"""Read-only inspection tools for the AI fix suggester.

These functions operate on a notebook dict (the one the handler already has in
the request body) and return JSON-serializable summaries that the LLM can
reason about. They are designed for the agentic loop in fix_suggester.py:

    1. The LLM gets a list of available tools (TOOL_SCHEMAS).
    2. When the model emits a tool_use, the suggester calls dispatch().
    3. dispatch() returns a tool_result the model can read in the next turn.

The functions have no kernel access, no file I/O, no eval, and no mutation
surface. They are safe to expose to any LLM call we make for diagnosis.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# How many characters of cell source we show in list_cells_summary previews.
_PREVIEW_CHARS = 120

# Default cap on per-output text returned by get_cell_outputs. The model
# rarely needs more than this; longer outputs balloon token costs.
_DEFAULT_OUTPUT_MAX_CHARS = 2000


def _code_cells(notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]


def _find_cell(notebook: Dict[str, Any], cell_id: str) -> Optional[Dict[str, Any]]:
    for c in notebook.get("cells", []):
        if c.get("id") == cell_id:
            return c
    return None


def _alpha(idx: int) -> str:
    """0-based code-cell index → @-label (A..Z, AA..AZ, ...)."""
    result = ""
    n = idx
    while True:
        result = chr(ord("A") + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            return result


def _source_of(cell: Dict[str, Any]) -> str:
    s = cell.get("source", "")
    if isinstance(s, list):
        return "".join(s)
    return s or ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + "... [truncated]"


def list_cells_summary(notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One-line summary of every code cell in the notebook."""
    out: List[Dict[str, Any]] = []
    for idx, cell in enumerate(_code_cells(notebook)):
        flowbook_meta = (cell.get("metadata") or {}).get("flowbook") or {}
        errors = flowbook_meta.get("errors") or []
        out.append({
            "cell_id": cell.get("id"),
            "alpha": _alpha(idx),
            "source_preview": _truncate(_source_of(cell), _PREVIEW_CHARS),
            "has_violation": bool(errors),
            "violation_types": [e.get("error_type") for e in errors],
            "has_outputs": bool(cell.get("outputs")),
            "execution_count": cell.get("execution_count"),
        })
    return out


def get_cell_source(notebook: Dict[str, Any], cell_id: str) -> str:
    """Full source code of one cell."""
    cell = _find_cell(notebook, cell_id)
    if cell is None:
        raise ToolError(f"No cell with id '{cell_id}'")
    return _source_of(cell)


def get_cell_outputs(
    notebook: Dict[str, Any],
    cell_id: str,
    max_chars: int = _DEFAULT_OUTPUT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Truncated, text-only summaries of each output on a cell.

    For each nbformat output we extract a single text-shaped summary:
      - 'stream' outputs become {kind:'stream', name, text}
      - 'execute_result' / 'display_data' use 'text/plain' (or 'text/html'
        stripped to plain), truncated to max_chars
      - 'error' outputs become {kind:'error', ename, evalue, traceback}
    Image outputs become {kind:'image', mime} with no payload.
    """
    cell = _find_cell(notebook, cell_id)
    if cell is None:
        raise ToolError(f"No cell with id '{cell_id}'")
    out: List[Dict[str, Any]] = []
    for raw in cell.get("outputs") or []:
        otype = raw.get("output_type")
        if otype == "stream":
            text = raw.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            out.append({
                "kind": "stream",
                "name": raw.get("name", "stdout"),
                "text": _truncate(text, max_chars),
            })
        elif otype in ("execute_result", "display_data"):
            data = raw.get("data") or {}
            text = data.get("text/plain") or data.get("text/html") or ""
            if isinstance(text, list):
                text = "".join(text)
            mimes = sorted(data.keys())
            out.append({
                "kind": otype,
                "mimes": mimes,
                "text": _truncate(str(text), max_chars),
            })
        elif otype == "error":
            tb = raw.get("traceback") or []
            if isinstance(tb, list):
                tb_text = "\n".join(tb)
            else:
                tb_text = str(tb)
            out.append({
                "kind": "error",
                "ename": raw.get("ename", ""),
                "evalue": raw.get("evalue", ""),
                "traceback": _truncate(tb_text, max_chars),
            })
        else:
            out.append({"kind": otype or "unknown"})
    return out


def get_cell_flowbook_meta(
    notebook: Dict[str, Any], cell_id: str
) -> Dict[str, Any]:
    """Return the cell's flowbook metadata (read_locs, write_locs, errors, ...).

    Strips large arrays we don't expect the model to need (e.g. dense timing
    histograms). Returns an empty dict for cells with no flowbook metadata
    rather than raising — the LLM may legitimately probe an un-executed cell.
    """
    cell = _find_cell(notebook, cell_id)
    if cell is None:
        raise ToolError(f"No cell with id '{cell_id}'")
    meta = (cell.get("metadata") or {}).get("flowbook") or {}
    # Project to fields useful for diagnosis.
    projected = {}
    for key in (
        "read_locs",
        "write_locs",
        "changed_locs",
        "errors",
        "stale_cells",
        "structural_warnings",
        "execution_seq",
    ):
        if key in meta:
            projected[key] = meta[key]
    return projected


def get_cell_traceback(
    notebook: Dict[str, Any], cell_id: str
) -> Optional[Dict[str, Any]]:
    """If a cell's last execution raised, return the error output. Else None."""
    cell = _find_cell(notebook, cell_id)
    if cell is None:
        raise ToolError(f"No cell with id '{cell_id}'")
    for raw in cell.get("outputs") or []:
        if raw.get("output_type") == "error":
            tb = raw.get("traceback") or []
            tb_text = "\n".join(tb) if isinstance(tb, list) else str(tb)
            return {
                "ename": raw.get("ename", ""),
                "evalue": raw.get("evalue", ""),
                "traceback": _truncate(tb_text, _DEFAULT_OUTPUT_MAX_CHARS),
            }
    return None


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / litellm function-calling shape)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_cells_summary",
            "description": (
                "Return a one-line summary of every code cell in the notebook: "
                "id, @-label, short source preview, whether it has a violation, "
                "and the execution count. Call this first when you need an "
                "overview of the notebook structure."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell_source",
            "description": "Return the full source code of one cell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {
                        "type": "string",
                        "description": "The cell_id from list_cells_summary.",
                    },
                },
                "required": ["cell_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell_outputs",
            "description": (
                "Return text-shaped summaries of a cell's outputs (stream, "
                "execute_result, display_data, error). Long outputs are "
                "truncated to max_chars characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string"},
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            "Per-output character limit. Default 2000."
                        ),
                        "default": 2000,
                    },
                },
                "required": ["cell_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell_flowbook_meta",
            "description": (
                "Return the cell's flowbook reproducibility metadata: "
                "read_locs, write_locs, changed_locs, errors, stale_cells. "
                "Useful when you need precise locations from a recent execution."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cell_id": {"type": "string"}},
                "required": ["cell_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell_traceback",
            "description": (
                "If the cell's last execution raised, return the error "
                "(ename, evalue, traceback). Returns null otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cell_id": {"type": "string"}},
                "required": ["cell_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class ToolError(ValueError):
    """Raised when a tool call has bad args. Returned to the LLM as a string."""


_DISPATCH: Dict[str, Callable[..., Any]] = {
    "list_cells_summary": lambda notebook: list_cells_summary(notebook),
    "get_cell_source": lambda notebook, cell_id: get_cell_source(notebook, cell_id),
    "get_cell_outputs": lambda notebook, cell_id, max_chars=_DEFAULT_OUTPUT_MAX_CHARS: get_cell_outputs(
        notebook, cell_id, max_chars
    ),
    "get_cell_flowbook_meta": lambda notebook, cell_id: get_cell_flowbook_meta(
        notebook, cell_id
    ),
    "get_cell_traceback": lambda notebook, cell_id: get_cell_traceback(
        notebook, cell_id
    ),
}


def dispatch(notebook: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> Any:
    """Run the named read-only tool against the notebook. Returns plain JSON.

    Raises ToolError for unknown tools or bad args; the suggester loop turns
    that into a tool_result error string the LLM can read and recover from.
    """
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        raise ToolError(
            f"Unknown read-only tool '{tool_name}'. Available: {sorted(_DISPATCH)}"
        )
    try:
        return handler(notebook, **(args or {}))
    except ToolError:
        raise
    except TypeError as e:
        raise ToolError(f"Bad args for '{tool_name}': {e}")


def tool_names() -> List[str]:
    return list(_DISPATCH.keys())
