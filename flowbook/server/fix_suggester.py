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
    """Compose the full system prompt: primer + task instructions."""
    primer = load_primer()
    return f"""You are diagnosing a reproducibility violation in a Jupyter notebook
tracked by FlowBook. You will be given the violating cell, surrounding cells,
and the violation details. Your job is to (1) explain the root cause in one
sentence and (2) propose 1–3 concrete fixes from a fixed taxonomy.

# Background: what FlowBook enforces

{primer}

# Available fix tools

You may ONLY propose fixes from this taxonomy. Anything else will be rejected.

- `alpha_rename(cell_id, old_name, new_name)` — AST-rename a variable from
  the given cell onwards. Use when a name is reused for different purposes,
  or when a sequential transform overwrites a name already read above.
  Choose a *semantically meaningful* new_name (e.g. `train_combined`,
  `lr_model`, `df_featured`) — not `df2`, `df_new`, or `model_v2`.

- `remove_inplace(cell_id, variable)` — Convert `df.method(..., inplace=True)`
  to `df = df.method(...)`. Use for UNRECOVERABLE_MUTATION caused by pandas
  inplace operations.

- `insert_deepcopy(cell_id, variable)` — Insert `import copy; var_copy =
  copy.deepcopy(var)` at the top of the cell and rename `var` → `var_copy`
  in the rest of that cell and downstream. Use for UNRECOVERABLE_MUTATION on
  objects with mutating methods like `model.fit()`.

- `mark_diagnostic(cell_id)` — Prefix the cell with `%diagnostic` so it is
  treated as a read-only inspection cell that doesn't participate in
  reproducibility tracking. Use when a `df.info()` / `df.head()` / `print(...)`
  cell sits above a cell that mutates the variable.

- `merge_cells([cell_id_a, cell_id_b, ...])` — Combine adjacent cells into
  one. Use when an allocation and a tightly-coupled transformation are split
  across two cells and would be simpler as one logical unit.

- `move_cell(cell_id, after_cell_id)` — Reorder cells. Use when a read-only
  inspection cell should be placed after the cell that mutates the variable.

# Response format

Stream your diagnosis as plain text first — one or two short sentences
explaining the root cause. Do not use markdown headers or bullet points.

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

Diagnose the root cause in one or two sentences, then emit a {FIX_PLAN_OPEN}...{FIX_PLAN_CLOSE} block.
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


SuggesterEvent = Union[TextEvent, PlanEvent, ErrorEvent]


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
        self, context: ViolationContext
    ) -> AsyncIterator[SuggesterEvent]:
        """Yield TextEvents as the model writes its diagnosis, then a PlanEvent
        (on successful parse + validate) or an ErrorEvent (on failure).

        The user-facing "diagnosis" is whatever appears *before* the
        FIX_PLAN_OPEN tag. Everything from the tag onward is structured data
        and is buffered, not yielded as text.
        """
        # Lazy import: a missing 'litellm' extra shouldn't break the rest of
        # the server extension.
        try:
            from litellm import acompletion
        except ImportError as e:
            yield ErrorEvent(
                message=(
                    f"litellm not installed; the AI fix feature is disabled. "
                    f"Install with `pip install 'flowbook-python[ai]'`. ({e})"
                )
            )
            return

        user_message = build_user_message(context)

        # Buffer used to (a) hold the partial FIX_PLAN block and (b) detect
        # the boundary where text-mode ends and JSON-mode begins.
        full_response: list[str] = []
        emitted_up_to = 0  # offset in joined response that we've already yielded
        in_plan = False

        try:
            response = await acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
            )

            async for chunk in response:
                # litellm normalizes chunks to OpenAI-shape regardless of provider.
                delta = _extract_chunk_text(chunk)
                if not delta:
                    continue
                full_response.append(delta)
                joined = "".join(full_response)

                if not in_plan:
                    tag_idx = joined.find(FIX_PLAN_OPEN)
                    if tag_idx == -1:
                        # Still in diagnosis text. Emit everything new
                        # *except* the last few chars, in case the tag
                        # is straddling a chunk boundary.
                        safe_end = max(
                            emitted_up_to, len(joined) - len(FIX_PLAN_OPEN)
                        )
                        if safe_end > emitted_up_to:
                            new_text = joined[emitted_up_to:safe_end]
                            if new_text:
                                yield TextEvent(text=new_text)
                                emitted_up_to = safe_end
                    else:
                        # Flush any remaining diagnosis text before the tag.
                        if tag_idx > emitted_up_to:
                            yield TextEvent(text=joined[emitted_up_to:tag_idx])
                            emitted_up_to = tag_idx
                        in_plan = True
                # When in_plan, we just keep buffering — no more TextEvents.

            final_text = "".join(full_response)

        except Exception as e:
            yield ErrorEvent(message=f"LLM call failed: {type(e).__name__}: {e}")
            return

        # Parse + validate the FIX_PLAN block.
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
        raw = causer[1:] if isinstance(causer, str) and causer.startswith("@") else causer
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
