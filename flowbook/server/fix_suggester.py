"""AI-driven fix suggester for FlowBook reproducibility violations.

Builds a prompt grounded in REPRODUCIBILITY_PRIMER.md and the violating cell's
context, streams a diagnosis from an LLM (via litellm, so any
provider/model is supported), then parses a structured FixPlan out of a
trailing <FIX_PLAN>...</FIX_PLAN> JSON block.

The suggester never *applies* fixes — it only proposes them. Application
happens in handlers.py against the AST helpers in
flowbook/scripts/fix_repro_errors.py, behind a strict allowlist.

Model selection lives in the FlowBookExtension.fix_model traitlet. Provider
API keys come from standard env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY,
etc.) — litellm picks them up automatically based on the model prefix.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, Union

from flowbook.server.fix_models import (
    FixPlan,
    FixSuggestion,
    PlanValidationError,
    ViolationContext,
    validate_plan,
)
from flowbook.server.fix_tools_readonly import (
    TOOL_SCHEMAS as READ_ONLY_TOOL_SCHEMAS,
    ToolError as ReadOnlyToolError,
    dispatch as dispatch_read_only_tool,
)
from flowbook.server.fix_tools_mutator import (
    TOOL_SCHEMAS as MUTATOR_TOOL_SCHEMAS,
    MutationLog,
    MutatorError,
    dispatch as dispatch_mutator_tool,
    tool_names as mutator_tool_names,
)
from flowbook.tools.prompt import (
    FIX_TAXONOMY,
    render_function_schemas,
    render_tool_catalog,
)

# Default model when the FlowBookExtension.fix_model traitlet is not overridden.
# Opus is the right default here even though it's pricier than Haiku — this is
# a judgment-heavy task (picking the right fix from the taxonomy, naming
# variables sensibly). Users can override via --FlowBookExtension.fix_model.
DEFAULT_MODEL = "anthropic/claude-opus-4-7"
MAX_OUTPUT_TOKENS = 800

# Provider prefix → env var that must be set for that provider to work.
# Used by get_provider_for_model() to decide whether the feature should be
# advertised as enabled to the frontend.
_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure": "AZURE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Stream-end sentinel: the model is instructed to emit its structured fix
# plan inside this tag at the end of its response.
FIX_PLAN_OPEN = "<FIX_PLAN>"
FIX_PLAN_CLOSE = "</FIX_PLAN>"

_PRIMER_PATH = Path(__file__).parent.parent / "docs" / "REPRODUCIBILITY_PRIMER.md"


def load_primer() -> str:
    """Load the reproducibility primer. Raises at module-use time if missing."""
    if not _PRIMER_PATH.exists():
        raise FileNotFoundError(
            f"Reproducibility primer not found at {_PRIMER_PATH}. "
            "The flowbook package may be installed without its docs directory."
        )
    return _PRIMER_PATH.read_text(encoding="utf-8")


def get_model(settings: Optional[dict] = None) -> str:
    """Resolve the configured model identifier for litellm.

    ``settings["flowbook"]`` is normally the FlowBookExtension instance
    (jupyter_server's ExtensionApp writes ``self`` to that key as part of
    ``_prepare_settings``, which runs after our ``initialize_settings``
    hook). We also accept a plain dict to keep tests + bench harnesses
    simple — they can pass ``{"flowbook": {"fix_model": "..."}}``.
    """
    if settings is not None:
        bag = settings.get("flowbook")
        model = _bag_get(bag, "fix_model", "")
        if model:
            return model
    return DEFAULT_MODEL


def _bag_get(bag, key: str, default):
    """Read ``key`` from either a dict or an attribute-bearing object."""
    if bag is None:
        return default
    if isinstance(bag, dict):
        return bag.get(key, default)
    return getattr(bag, key, default)


def _provider_prefix(model: str) -> str:
    """Extract the provider prefix from a litellm model string.

    'anthropic/claude-opus-4-7' -> 'anthropic'
    'openai/gpt-4o'             -> 'openai'
    'gpt-4o'                    -> 'openai' (litellm default for un-prefixed
                                  OpenAI model names)
    """
    if "/" in model:
        return model.split("/", 1)[0].lower()
    # Heuristic: un-prefixed gpt-* / o1-* / o3-* names are OpenAI.
    lower = model.lower()
    if lower.startswith(("gpt-", "o1-", "o3-", "text-")):
        return "openai"
    if lower.startswith("claude-"):
        return "anthropic"
    return ""


def feature_enabled(settings: Optional[dict] = None) -> bool:
    """Return True iff the configured model's provider API key is present.

    The frontend uses this (via a probe response from the suggest-fix
    handler) to decide whether to render the suggestion UI at all. We do
    NOT instantiate a litellm client here — that's deferred to the actual
    request so import errors don't break the rest of the server.
    """
    model = get_model(settings)
    provider = _provider_prefix(model)
    env_var = _PROVIDER_ENV_VARS.get(provider)
    if env_var is None:
        # Unknown provider — assume the user knows what they're doing if any
        # of the common keys is set.
        return any(os.environ.get(v) for v in _PROVIDER_ENV_VARS.values())
    return bool(os.environ.get(env_var))


def build_system_prompt() -> str:
    """Compose the full system prompt: primer + task instructions.

    The fix-tool list is rendered from the unified registry
    (`flowbook.tools`) so it cannot drift from the tools that are actually
    validated and applied; per-tool craft guidance comes from FIX_TAXONOMY.
    """
    primer = load_primer()
    return f"""You are diagnosing a reproducibility violation in a Jupyter notebook
tracked by FlowBook. You will be given the violating cell, surrounding cells,
and the violation details. Your job is to (1) explain the root cause in one
short phrase and (2) propose 1-3 concrete fixes from a fixed taxonomy.

# Background: what FlowBook enforces

{primer}

# Available fix tools

You may ONLY propose fixes from this taxonomy. Anything else will be rejected.

{render_tool_catalog()}

{FIX_TAXONOMY}

# Inspection tools

Before answering you may call the available inspection tools to read any
cell's source, outputs, traceback, or flowbook metadata. Use them when:

- You need to see a cell that wasn't in the surrounding window I gave you.
- You need to look at a cell's outputs (e.g. to learn the shape of a
  DataFrame, the type of an object, the contents of an error).
- You need precise read/write locations from a cell's last execution.

Call as many tools as you need. Each call returns a JSON result you can
read in your next response. Stop calling tools when you are confident
about the root cause.

# Response format

Stream your diagnosis as plain text first — one short phrase explaining the root cause. Do not use markdown headers or bullet points.

Then, on a final line, emit exactly one block in this format:

{FIX_PLAN_OPEN}{{"fixes": [
  {{"label": "...", "rationale": "...", "tool": "...", "args": {{...}}}}
]}}{FIX_PLAN_CLOSE}

Rules:
- Use only the six tools listed above.
- `cell_id` (and `after_cell_id`, members of `cell_ids`) must be drawn from
  the cell order I give you.
- For `alpha_rename` / `remove_inplace` / `insert_deepcopy`, the variable
  name you choose must literally appear in the named cell's source.
- Order fixes from most to least likely to resolve the violation.
- Propose at most 3 fixes. Often 1 is enough — only suggest alternatives
  when there is a meaningful tradeoff.
- Labels are short (max ~60 chars) and start with a verb.
- Output nothing after the closing tag.
"""


def build_user_message(context: ViolationContext) -> str:
    """Render the violation context as the user-turn prompt."""
    surrounding_text = "\n\n".join(
        f"## Cell {cid}\n```python\n{src}\n```"
        for cid, src in context.surrounding_sources.items()
    )

    return f"""# Violation

**Cell**: @{context.cell_alpha} (cell_id: `{context.cell_id}`)
**Predicate**: `{context.error_type}`
**Locations**: {", ".join(f"`{loc}`" for loc in context.locations) or "(none)"}
**Other cells involved**: {", ".join(f"@{c}" for c in context.causer_cells) or "(none)"}

# Violating cell source

```python
{context.cell_source}
```

# Surrounding cells

Cell order in this notebook: {context.cell_order}

{surrounding_text if surrounding_text else "(no surrounding cells provided)"}

# Your task

Diagnose the root cause in one short phrase, then emit a {FIX_PLAN_OPEN}...{FIX_PLAN_CLOSE} block.
"""


# ---------------------------------------------------------------------------
# Streaming output
# ---------------------------------------------------------------------------


@dataclass
class TextEvent:
    """Streamed diagnosis text chunk (what the user sees being typed)."""

    text: str


@dataclass
class PlanEvent:
    """Final validated FixPlan, emitted once stream completes."""

    plan: FixPlan


@dataclass
class ErrorEvent:
    """Terminal event indicating the plan could not be produced.

    The frontend should display whatever TextEvents arrived (the diagnosis
    is still useful) but hide the button row.
    """

    message: str


@dataclass
class CustomDoneEvent:
    """Terminal event for a custom-fix run.

    `log` is the accumulated MutationLog containing pre-fix snapshots and the
    ordered list of mutation entries. The handler uses it to build the
    CustomFixResponse.
    """

    summary: str
    log: "MutationLog"


SuggesterEvent = Union[TextEvent, PlanEvent, ErrorEvent, CustomDoneEvent]


CUSTOM_FIX_SYSTEM_PROMPT_TEMPLATE = """You are applying a user-requested fix
to a Jupyter notebook tracked by FlowBook. The user has explicitly approved
this change by clicking "Other Fix…" and typing the instruction below.

# Background: what FlowBook enforces

{primer}

# Your task

Apply the user's request. You may freely inspect the notebook first with the
read-only tools, then change it with the mutator tools.

Read-only tools (inspect):
{read_tools}

Mutator tools (change):
{mutator_tools}

Rules:
- Make the smallest set of changes that satisfies the request.
- Code cells must still parse as Python after every edit — the server
  will reject malformed code and you should retry.
- Use semantically meaningful variable names if you rename.
- When you are done, stop calling tools and write a one-sentence summary
  of what you changed. Do not write the summary until everything is done.
- Do not run cells; do not invent tools; do not read or write files.

# The user's request

The user, looking at cell @{cell_alpha} (cell_id: {cell_id}), asked:

> {instruction}

Inspect, mutate, then write your one-sentence summary.
"""


class FixSuggester:
    """Provider-agnostic LLM client for fix suggestion, via litellm.

    The suggester is intentionally thin — it owns prompt construction,
    streaming, and FIX_PLAN extraction. It does not apply fixes; it does not
    talk to the kernel; it does not know about Tornado or SSE. Those concerns
    belong to handlers.py.

    Model + provider are configured via FlowBookExtension.fix_model; litellm
    handles API key resolution from standard env vars per provider.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model = model
        self._system_prompt = build_system_prompt()

    async def stream(
        self,
        context: ViolationContext,
        notebook: Optional[dict] = None,
    ) -> AsyncIterator[SuggesterEvent]:
        """Yield TextEvents as the model writes its diagnosis, then a PlanEvent
        (on successful parse + validate) or an ErrorEvent (on failure).

        The agentic loop runs as many turns as the model wants. On each turn:
          - Stream text chunks (with FIX_PLAN-boundary detection, identical to
            the single-shot path — text accumulates across turns).
          - Accumulate any tool_use deltas.
          - If the turn ended with tool calls, execute them, append the
            tool_result messages, and loop.
          - Otherwise this is the final turn — parse FIX_PLAN, validate,
            and emit a PlanEvent.

        If ``notebook`` is None, no tools are passed to the model and the loop
        terminates after the first turn — behavior equivalent to the original
        single-shot stream(). This keeps existing tests valid.
        """
        # Lazy import: a missing 'litellm' install shouldn't break the rest of
        # the server extension.
        try:
            from litellm import acompletion
        except ImportError as e:
            yield ErrorEvent(
                message=(
                    f"litellm not installed; the AI fix feature is disabled. "
                    f"Install with `pip install litellm`. ({e})"
                )
            )
            return

        user_message = build_user_message(context)
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        tools_arg = READ_ONLY_TOOL_SCHEMAS if notebook is not None else None

        # Shared text-emission state spans every turn so FIX_PLAN can land
        # anywhere across the streamed conversation.
        full_response_text: list[str] = []
        emitted_up_to = 0
        in_plan = False

        # Agentic loop. No turn cap — the model decides when it's done by
        # producing a turn with no tool calls.
        while True:
            turn_text: list[str] = []
            tool_call_buffers: dict[int, _ToolCallBuffer] = {}
            finish_reason: Optional[str] = None

            try:
                response = await acompletion(
                    model=self._model,
                    messages=messages,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    stream=True,
                    **({"tools": tools_arg} if tools_arg else {}),
                )

                async for chunk in response:
                    fr = _extract_finish_reason(chunk)
                    if fr:
                        finish_reason = fr

                    delta = _extract_chunk_text(chunk)
                    if delta:
                        turn_text.append(delta)
                        full_response_text.append(delta)
                        joined = "".join(full_response_text)
                        if not in_plan:
                            tag_idx = joined.find(FIX_PLAN_OPEN)
                            if tag_idx == -1:
                                safe_end = max(
                                    emitted_up_to,
                                    len(joined) - len(FIX_PLAN_OPEN),
                                )
                                if safe_end > emitted_up_to:
                                    yield TextEvent(text=joined[emitted_up_to:safe_end])
                                    emitted_up_to = safe_end
                            else:
                                if tag_idx > emitted_up_to:
                                    yield TextEvent(text=joined[emitted_up_to:tag_idx])
                                    emitted_up_to = tag_idx
                                in_plan = True

                    for tc_delta in _extract_tool_call_deltas(chunk):
                        _accumulate_tool_call(tool_call_buffers, tc_delta)

            except Exception as e:
                yield ErrorEvent(message=f"LLM call failed: {type(e).__name__}: {e}")
                return

            # End of turn. Did the model call tools?
            if tool_call_buffers and tools_arg is not None:
                finalized = _finalize_tool_calls(tool_call_buffers)
                # Append assistant message (text + tool calls) so the next
                # turn has the full conversation history.
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": "".join(turn_text) or None,
                    "tool_calls": finalized,
                }
                messages.append(assistant_msg)
                # Execute each tool against the notebook; feed results back.
                for call in finalized:
                    result_str = _run_read_only_tool(notebook, call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result_str,
                        }
                    )
                # Loop for the next turn.
                continue

            # Terminal turn (no tool calls). Parse FIX_PLAN and emit it.
            final_text = "".join(full_response_text)
            try:
                plan = _extract_and_parse_plan(final_text)
                plan = validate_plan(plan, context)
            except PlanValidationError as e:
                yield ErrorEvent(message=f"Plan validation failed: {e}")
                return
            except Exception as e:
                yield ErrorEvent(message=f"Could not parse plan: {e}")
                return
            yield PlanEvent(plan=plan)
            return

    async def custom_stream(
        self,
        notebook: dict,
        cell_id: str,
        cell_alpha: str,
        instruction: str,
    ) -> AsyncIterator[SuggesterEvent]:
        """Run the custom-fix agentic loop.

        Exposes BOTH read-only and mutator tools. The LLM may make any number
        of calls; the loop terminates when the model emits a turn with no
        tool calls. The trailing text of that final turn becomes the summary.

        Mutations are applied directly to the `notebook` dict and recorded in
        a MutationLog. The handler uses the log to build a CustomFixResponse
        (pre/post sources, cells added/removed, etc.).
        """
        try:
            from litellm import acompletion
        except ImportError as e:
            yield ErrorEvent(
                message=f"litellm not installed; the AI fix feature is disabled. ({e})"
            )
            return

        primer = load_primer()
        system_prompt = CUSTOM_FIX_SYSTEM_PROMPT_TEMPLATE.format(
            primer=primer,
            read_tools=render_function_schemas(READ_ONLY_TOOL_SCHEMAS),
            mutator_tools=render_function_schemas(MUTATOR_TOOL_SCHEMAS),
            cell_alpha=cell_alpha,
            cell_id=cell_id,
            instruction=instruction,
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Apply this fix to the notebook. The violating cell is "
                    f"@{cell_alpha} (cell_id: {cell_id}). Start by inspecting "
                    f"whatever you need, then mutate."
                ),
            },
        ]
        tools_arg = list(READ_ONLY_TOOL_SCHEMAS) + list(MUTATOR_TOOL_SCHEMAS)
        mutator_names = set(mutator_tool_names())
        log = MutationLog()
        full_response_text: list[str] = []
        emitted_up_to = 0

        # Agentic loop. No turn cap.
        while True:
            turn_text: list[str] = []
            tool_call_buffers: dict[int, _ToolCallBuffer] = {}
            finish_reason: Optional[str] = None

            try:
                response = await acompletion(
                    model=self._model,
                    messages=messages,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    stream=True,
                    tools=tools_arg,
                )
                async for chunk in response:
                    fr = _extract_finish_reason(chunk)
                    if fr:
                        finish_reason = fr
                    delta = _extract_chunk_text(chunk)
                    if delta:
                        turn_text.append(delta)
                        full_response_text.append(delta)
                        joined_new = delta
                        if joined_new:
                            yield TextEvent(text=joined_new)
                            emitted_up_to += len(joined_new)
                    for tc_delta in _extract_tool_call_deltas(chunk):
                        _accumulate_tool_call(tool_call_buffers, tc_delta)
            except Exception as e:
                yield ErrorEvent(message=f"LLM call failed: {type(e).__name__}: {e}")
                return

            if tool_call_buffers:
                finalized = _finalize_tool_calls(tool_call_buffers)
                messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(turn_text) or None,
                        "tool_calls": finalized,
                    }
                )
                for call in finalized:
                    name = (call.get("function") or {}).get("name") or ""
                    if name in mutator_names:
                        result_str = _run_mutator_tool(notebook, log, call)
                    else:
                        result_str = _run_read_only_tool(notebook, call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result_str,
                        }
                    )
                continue

            # Terminal turn — model is done. Trailing text is the summary.
            summary = "".join(full_response_text).strip()
            yield CustomDoneEvent(summary=summary, log=log)
            return


def _run_mutator_tool(notebook: dict, log: "MutationLog", call: dict) -> str:
    """Execute one mutator tool call and return a string result.

    Same error-as-string handling as _run_read_only_tool so the LLM can read
    and recover.
    """
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    raw_args = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError as e:
        return f'{{"error": "Could not parse arguments JSON: {e}"}}'
    try:
        result = dispatch_mutator_tool(notebook, log, name, args)
    except MutatorError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    try:
        return json.dumps(result)
    except TypeError:
        return json.dumps({"error": "tool returned non-serializable value"})


def _extract_chunk_text(chunk) -> str:
    """Pull the text delta out of a litellm streaming chunk.

    litellm returns OpenAI-shaped chunks for every provider:
        chunk.choices[0].delta.content
    But the chunk object can be a dict or an attrs-style object depending on
    the provider, so we accept both.
    """
    try:
        choices = chunk.choices if hasattr(chunk, "choices") else chunk["choices"]
        if not choices:
            return ""
        first = choices[0]
        delta = first.delta if hasattr(first, "delta") else first["delta"]
        content = delta.content if hasattr(delta, "content") else delta.get("content")
        return content or ""
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""


def _extract_finish_reason(chunk) -> Optional[str]:
    """Pull finish_reason out of a chunk if present."""
    try:
        choices = chunk.choices if hasattr(chunk, "choices") else chunk["choices"]
        if not choices:
            return None
        first = choices[0]
        fr = (
            first.finish_reason
            if hasattr(first, "finish_reason")
            else first.get("finish_reason")
        )
        return fr or None
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _extract_tool_call_deltas(chunk) -> list[dict]:
    """Pull the list of tool-call delta fragments from a streaming chunk.

    Each fragment looks like:
        {"index": 0, "id": "call_abc", "function": {"name": "...", "arguments": "..."}}
    """
    try:
        choices = chunk.choices if hasattr(chunk, "choices") else chunk["choices"]
        if not choices:
            return []
        first = choices[0]
        delta = first.delta if hasattr(first, "delta") else first["delta"]
        tcs = (
            delta.tool_calls
            if hasattr(delta, "tool_calls")
            else delta.get("tool_calls")
        )
        if not tcs:
            return []
        out: list[dict] = []
        for tc in tcs:
            # Each item may be an object or a dict.
            if hasattr(tc, "model_dump"):
                out.append(tc.model_dump())
            elif hasattr(tc, "__dict__"):
                out.append(
                    {
                        "index": getattr(tc, "index", 0),
                        "id": getattr(tc, "id", None),
                        "function": _to_dict(getattr(tc, "function", None)),
                    }
                )
            else:
                out.append(dict(tc))
        return out
    except (AttributeError, KeyError, IndexError, TypeError):
        return []


def _to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {
        "name": getattr(obj, "name", None),
        "arguments": getattr(obj, "arguments", ""),
    }


class _ToolCallBuffer:
    """Accumulates fragments of a single streamed tool call."""

    __slots__ = ("id", "name", "arguments")

    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: str = ""


def _accumulate_tool_call(buffers: dict[int, _ToolCallBuffer], delta: dict) -> None:
    """Merge one fragment into the per-index buffer."""
    idx = delta.get("index", 0)
    buf = buffers.setdefault(idx, _ToolCallBuffer())
    if delta.get("id"):
        buf.id = delta["id"]
    fn = delta.get("function") or {}
    if fn.get("name"):
        buf.name = fn["name"]
    args_fragment = fn.get("arguments")
    if args_fragment:
        buf.arguments += args_fragment


def _finalize_tool_calls(
    buffers: dict[int, _ToolCallBuffer],
) -> list[dict]:
    """Convert per-index buffers into OpenAI-format tool_call message entries."""
    out: list[dict] = []
    for idx in sorted(buffers.keys()):
        buf = buffers[idx]
        out.append(
            {
                "id": buf.id or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": buf.name or "",
                    "arguments": buf.arguments or "{}",
                },
            }
        )
    return out


def _run_read_only_tool(notebook: Optional[dict], call: dict) -> str:
    """Execute one read-only tool call and return a string result.

    Errors are returned as a string (not raised) so the model can read them
    in its next turn and recover (e.g. by retrying with a different cell_id).
    """
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    raw_args = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError as e:
        return f'{{"error": "Could not parse arguments JSON: {e}"}}'

    try:
        result = dispatch_read_only_tool(notebook or {}, name, args)
    except ReadOnlyToolError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    try:
        return json.dumps(result)
    except TypeError:
        return json.dumps({"error": "tool returned non-serializable value"})


def _extract_and_parse_plan(text: str) -> FixPlan:
    """Pull the JSON out of <FIX_PLAN>...</FIX_PLAN> and parse it.

    The regex is forgiving about whitespace and trailing tokens.
    """
    match = re.search(
        re.escape(FIX_PLAN_OPEN) + r"(.*?)" + re.escape(FIX_PLAN_CLOSE),
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise PlanValidationError("no FIX_PLAN block found in response")

    raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"FIX_PLAN block is not valid JSON: {e}")

    if not isinstance(data, dict) or "fixes" not in data:
        raise PlanValidationError("FIX_PLAN must be an object with a 'fixes' array")

    return FixPlan.model_validate(data)


# ---------------------------------------------------------------------------
# Convenience: build a ViolationContext from a raw notebook + cell_id
# ---------------------------------------------------------------------------


def build_context_from_notebook(
    notebook: dict, cell_id: str, neighbor_window: int = 3
) -> Optional[ViolationContext]:
    """Assemble a ViolationContext from notebook JSON + violating cell_id.

    Returns None if the cell has no violation metadata (caller can treat as
    "nothing to suggest").
    """
    code_cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
    cell_order = [c.get("id", "") for c in code_cells]

    if cell_id not in cell_order:
        return None

    idx = cell_order.index(cell_id)
    cell = code_cells[idx]
    cell_alpha = _indices_to_alpha_safe(idx)

    meta = cell.get("metadata", {}).get("flowbook", {})
    errors = meta.get("errors") or []
    if not errors:
        return None

    # If there are multiple errors on one cell, take the first — the
    # frontend triggers suggest-fix per (cell, violation) so we always
    # diagnose one at a time. The model gets all locations regardless.
    primary = errors[0]
    error_type = primary.get("error_type", "unknown")
    locations = list(primary.get("locations") or [])

    # Causer cells across all errors on this cell (deduped, as alpha labels).
    causer_alphas: list[str] = []
    for err in errors:
        causer = err.get("causer_cell")
        if not causer:
            continue
        raw = (
            causer[1:] if isinstance(causer, str) and causer.startswith("@") else causer
        )
        if raw in cell_order:
            alpha = _indices_to_alpha_safe(cell_order.index(raw))
            if alpha not in causer_alphas:
                causer_alphas.append(alpha)

    surrounding: dict[str, str] = {}
    lo = max(0, idx - neighbor_window)
    hi = min(len(code_cells), idx + neighbor_window + 1)
    for j in range(lo, hi):
        if j == idx:
            continue
        cid = cell_order[j]
        surrounding[cid] = _get_source(code_cells[j])

    return ViolationContext(
        cell_id=cell_id,
        cell_alpha=cell_alpha,
        cell_source=_get_source(cell),
        error_type=error_type,
        locations=locations,
        causer_cells=causer_alphas,
        cell_order=cell_order,
        surrounding_sources=surrounding,
    )


def _get_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def _indices_to_alpha_safe(idx: int) -> str:
    """Convert a 0-based code-cell index to its @-label (A, B, ..., Z, AA, ...).

    Mirrors src/cellindexutils.ts indexToAlpha so the frontend and backend
    agree on labels in violation messages.
    """
    result = ""
    n = idx
    while True:
        result = chr(ord("A") + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            return result
