"""Content adapters for scratch_work / get_cell_outputs.

`FlowBookTools` returns plain Python dicts (no MCP / Claude / OpenAI types).
The helpers in this module translate those dicts to each surface's native
multi-part content format. Keeping the conversion in one place means MIME
filtering, truncation, and ordering are defined exactly once.

Surfaces:
  to_mcp_content     — FastMCP list[TextContent | ImageContent]
  to_claude_content  — Claude multi-part tool_result blocks
  to_openai_content  — OpenAI role=tool content array
  to_markdown        — plain markdown with data: URIs (fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEEP_IMAGE_MIMES: tuple[str, ...] = ("image/png", "image/jpeg", "image/svg+xml")
KEEP_TEXT_MIMES: tuple[str, ...] = ("text/html", "text/plain")
KEEP_MIMES: tuple[str, ...] = KEEP_IMAGE_MIMES + KEEP_TEXT_MIMES

TEXT_TRUNCATE_BYTES = 8 * 1024        # stdout/stderr/result text
HTML_TRUNCATE_BYTES = 64 * 1024       # per text/html block
IMAGE_TRUNCATE_BYTES = 256 * 1024     # per decoded image
TOTAL_IMAGE_BUDGET_BYTES = 2 * 1024 * 1024  # total base64-decoded image payload per result


# ---------------------------------------------------------------------------
# ToolContent — structured return value for NBI tools
# ---------------------------------------------------------------------------

@dataclass
class ToolContent:
    """Structured tool result. NBI tools return this (instead of str) when
    they need to ship images or other rich content. The NBI participant
    wrappers (once patched in notebook-intelligence) dispatch to
    `to_claude_content` / `to_openai_content` / `text_summary` per provider.

    Until the upstream patch lands, NBI falls back to `text_summary`.
    """
    blocks: list[dict] = field(default_factory=list)
    text_summary: str = ""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _truncate_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n… [truncated, {len(s) - limit} more bytes]"


def _iter_outputs(result: dict) -> Iterable[tuple[str, dict]]:
    """Yield (label, output) pairs from a ScratchResult or per-cell section."""
    for out in result.get("outputs", []) or []:
        yield out.get("kind", "output"), out


def _output_blocks(output: dict, image_budget: list[int]) -> list[dict]:
    """Convert one Output dict into an ordered list of `ToolContent.blocks`
    items: `{type:"text", text:...}` or `{type:"image", mime:..., data:base64}`.

    `image_budget` is a list holding a single int (total bytes remaining for
    images) so we can track budget across multiple outputs in one call.
    """
    blocks: list[dict] = []
    kind = output.get("kind", "")
    data = output.get("data") or {}

    if kind == "stream":
        name = output.get("stream_name", "stdout")
        text = _truncate_text(str(output.get("text", "")), TEXT_TRUNCATE_BYTES)
        if text:
            blocks.append({"type": "text", "text": f"[{name}]\n{text}"})
        return blocks

    # execute_result / display_data: prefer images > html > text/plain
    for mime in KEEP_IMAGE_MIMES:
        payload = data.get(mime)
        if payload is None:
            continue
        b64 = payload.get("bytes", "")
        size = payload.get("size_bytes", 0) or len(b64) * 3 // 4
        if size > IMAGE_TRUNCATE_BYTES:
            blocks.append({"type": "text", "text": f"[{mime}: {size} bytes — exceeds per-image cap, omitted]"})
            continue
        if size > image_budget[0]:
            blocks.append({"type": "text", "text": f"[{mime}: {size} bytes — exceeds total image budget, omitted]"})
            continue
        image_budget[0] -= size
        blocks.append({"type": "image", "mime": mime, "data": b64})
        break  # one image payload per output is enough

    html = data.get("text/html")
    if html is not None:
        t = _truncate_text(str(html.get("text", "")), HTML_TRUNCATE_BYTES)
        if t:
            blocks.append({"type": "text", "text": f"[text/html]\n{t}"})

    # text/plain only if we did NOT already emit an image or html for this output
    if not blocks:
        txt = data.get("text/plain")
        if txt is not None:
            t = _truncate_text(str(txt.get("text", "")), TEXT_TRUNCATE_BYTES)
            if t:
                blocks.append({"type": "text", "text": t})

    # Markers for dropped MIMEs (the ones we explicitly don't round-trip)
    for mime, payload in data.items():
        if mime in KEEP_MIMES:
            continue
        size = payload.get("size_bytes") or len(str(payload)) if isinstance(payload, dict) else 0
        blocks.append({"type": "text", "text": f"[{mime}: {size} bytes — not transported; text/plain shown above if present]"})

    return blocks


# ---------------------------------------------------------------------------
# Result → blocks (shared backbone)
# ---------------------------------------------------------------------------

def _scratch_header(result: dict) -> list[dict]:
    status = result.get("status", "ok")
    t = result.get("execution_time_ms")
    header = f"status: {status}"
    if t is not None:
        header += f"  ({float(t):.1f} ms)"
    return [{"type": "text", "text": header}]


def _error_block(result: dict) -> list[dict]:
    err = result.get("error")
    if not err:
        return []
    lines = [f"error: {err.get('ename', '')}: {err.get('evalue', '')}"]
    tb = err.get("traceback", []) or []
    if tb:
        lines.append("")
        lines.extend(tb)
    return [{"type": "text", "text": "\n".join(lines)}]


def _scratch_blocks(result: dict) -> list[dict]:
    blocks: list[dict] = []
    blocks.extend(_scratch_header(result))
    budget = [TOTAL_IMAGE_BUDGET_BYTES]
    for _label, out in _iter_outputs(result):
        blocks.extend(_output_blocks(out, budget))
    blocks.extend(_error_block(result))
    return blocks


def _cell_outputs_blocks(result: dict) -> list[dict]:
    blocks: list[dict] = []
    budget = [TOTAL_IMAGE_BUDGET_BYTES]
    for cell in result.get("cells", []) or []:
        label = cell.get("label") or cell.get("cell_id", "?")
        header = f"=== {label}  [{cell.get('cell_id', '?')}] ==="
        blocks.append({"type": "text", "text": header})
        any_output = False
        for out in cell.get("outputs", []) or []:
            for b in _output_blocks(out, budget):
                blocks.append(b)
                any_output = True
        if not any_output:
            blocks.append({"type": "text", "text": "(no outputs)"})
    return blocks


def _result_to_blocks(result: dict) -> list[dict]:
    if "cells" in result:
        return _cell_outputs_blocks(result)
    return _scratch_blocks(result)


# ---------------------------------------------------------------------------
# Surface adapters
# ---------------------------------------------------------------------------

def to_mcp_content(result: dict) -> list[Any]:
    """Return a FastMCP content list (TextContent | ImageContent).

    Imports the MCP types lazily so this module doesn't require `mcp` at
    import time (useful for shared code used in environments without MCP).
    """
    from mcp.types import TextContent, ImageContent  # lazy

    content: list[Any] = []
    for b in _result_to_blocks(result):
        if b["type"] == "text":
            content.append(TextContent(type="text", text=b["text"]))
        else:
            content.append(ImageContent(type="image", mimeType=b["mime"], data=b["data"]))
    return content


def to_claude_content(result: dict) -> list[dict]:
    """Return Claude multi-part content blocks for a `tool_result`."""
    out: list[dict] = []
    for b in _result_to_blocks(result):
        if b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            out.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": b["mime"],
                    "data": b["data"],
                },
            })
    return out


def to_openai_content(result: dict) -> list[dict]:
    """Return an OpenAI-style content array for a role=tool message."""
    out: list[dict] = []
    for b in _result_to_blocks(result):
        if b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            out.append({
                "type": "image_url",
                "image_url": {"url": f"data:{b['mime']};base64,{b['data']}"},
            })
    return out


def to_markdown(result: dict) -> str:
    """Flatten to a single markdown string; images inline as data URIs."""
    parts: list[str] = []
    for b in _result_to_blocks(result):
        if b["type"] == "text":
            parts.append(b["text"])
        else:
            parts.append(f"![]({'data:' + b['mime'] + ';base64,' + b['data']})")
    return "\n\n".join(parts)


def build_tool_content(result: dict) -> ToolContent:
    """Build a ToolContent (for NBI use) from a ScratchResult / CellOutputsResult."""
    return ToolContent(
        blocks=_result_to_blocks(result),
        text_summary=to_markdown(result),
    )
