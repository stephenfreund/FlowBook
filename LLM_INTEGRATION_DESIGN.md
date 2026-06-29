# LLM Integration — Unification Design & Implementation Plan

**Status:** Phases 1–3 done & tested. Phase 4 done: edit + execution AI-attribution wired and
compiling, with **zero dependency on LogBook** (decoupled string contracts only). Remaining: the
optional Option-A agent rewrite.

> **Decoupling invariant:** FlowBook imports nothing from LogBook and works whether or not
> LogBook is installed. Integration is two one-way string contracts: the Yjs transaction origin
> `'flowbook'` (edits) and the `'ai-notebook-activity'` DOM CustomEvent (executions). Neither
> repo imports the other; LogBook just recognizes the origin and listens for the event. Do not
> reintroduce a token/package dependency in either direction.
> **Author:** Design assessment (2026-05-29)
> **Related:** `MCP_ARCHITECTURE.md`, `FORMAL_DEVELOPMENT.md`, `flowbook/docs/REPRODUCIBILITY_PRIMER.md`, `../LogBook` (event-logging extension)

## Implementation status (2026-05-29)

**Done — Phase 1 (the tool-catalog collapse), all three surfaces:**

- `flowbook/tools/` package: `controller.py` (`NotebookController` Protocol + `ToolError`/
  `CellNotFoundError`/`NoEffectError`), `registry.py` (`REGISTRY` of 6 refactoring tools),
  `reproducibility.py` (single-source handlers), `adapters/dict_controller.py`,
  `adapters/kernel_controller.py`.
- **Fix-it** (`server/fix_dispatcher.py`) and the **custom-fix mutator**
  (`server/fix_tools_mutator.py`) now route through the registry over a `DictController`.
- **MCP** (`mcp/session.py`) refactoring methods are thin adapters over the handlers via
  `KernelController` (return shapes + tolerant no-op handling preserved); added
  `session._notify_structure()`.
- **NBI** (`nbi/tools.py`) AST tools (alpha_rename, remove_inplace, insert_deepcopy) snapshot
  the notebook via the bridge, run the shared handler over a `DictController`, and replay edits.
- The six refactoring algorithms now have **one** orchestration definition instead of three.
  `insert_deepcopy`'s copy is now uniformly named `{var}_{cell_id}` (was `{var}_copy` in
  MCP/NBI) — collision-safe.
- **Verified:** `flowbook/tools` unit tests; server suite (`test_fix_*`); MCP headless +
  real-kernel integration tests (incl. new `TestRefactoringToolsOverKernelController`); NBI
  mocked tool tests (require `notebook-intelligence` installed in the env).

**Done — Phase 2 (unified prompt + taxonomy):**

- `flowbook/tools/prompt.py`: `render_tool_catalog()` renders the tool list from `REGISTRY`;
  `FIX_TAXONOMY` holds the craft guidance. `fix_suggester.build_system_prompt()` now composes
  from these — the hand-maintained tool list is gone.
- Generated `flowbook/docs/_generated/tool_catalog.md` + a drift-guard test
  (`tools/tests/test_prompt_and_registry.py`).

**Done — Phase 3 (validation single-sourced on the registry):**

- `fix_models.TOOL_ARG_SCHEMAS` is now derived from `REGISTRY` (was a hand-maintained dup); a
  test asserts `FixToolName` ⟷ registry agreement. The built-in fix path's _description_
  (prompt), _arg contract_ (validation), and _application_ (dispatcher) all now flow from one
  source. (The custom-fix agent's broader inspect/mutate toolset — `fix_tools_readonly` /
  `fix_tools_mutator` — is a distinct, non-duplicated surface and was left as-is; folding it in
  is the Option-A agent rewrite, deferred.)

**Done — Phase 4 (LogBook AI attribution), Python + frontend:**

- Kernel protocol: `build_metadata_message(metadata, actor=None)` adds an optional `actor`
  field (omitted by default); mirrored in `src/flowbook/protocol.ts` `IMetadataMessage`.
- `src/flowbook/aiattribution.ts` (general AI-attribution helpers — no LogBook in the name or
  imports):
  - **Edits:** `FLOWBOOK_TX_ORIGIN = 'flowbook'` + `aiTransact(sharedModel, fn)` tags the
    underlying Yjs `Doc` transaction; an observer that watches Yjs origins (LogBook does)
    attributes the edit to the AI. Wrapped around the Fix-it applier (`fixsuggester.ts`
    `_applySources`/`_removeCells`/`_applyMoveCell`) and the NBI bridge (`nbibridge.ts`
    `edit-cell-source`/`move-cell`).
  - **Executions:** `emitAiActivity({path, cellId, kind:'execute'})` dispatches the
    `'ai-notebook-activity'` DOM CustomEvent (with `source:'flowbook'`); fired from
    `executionhook.ts` when an incoming `flowbook_update` carries `actor === 'ai'` (i.e. an
    out-of-process MCP run on the shared kernel). No listener → no-op.
- **Kernel `actor` data path (done & tested):** MCP `run_cell` sends `actor="ai"` →
  `KernelHelper.execute_code(actor=...)` → execute metadata → kernel stashes `self._actor`
  (default `"user"`) → echoed on the `flowbook_update` metadata message via
  `build_metadata_message(metadata, actor=...)`. Real-kernel tests
  (`TestActorAttribution`) confirm `"ai"` for MCP runs and `"user"` by default.
- **Verified:** `jlpm build:lib` compiles cleanly (`aiattribution.js` + all touched files emit,
  zero tsc errors).
- **LogBook side (sibling repo, decoupled):** `origin.ts` recognizes the `'flowbook'` Yjs origin
  (edits); `ai-activity-relay.ts` (`installAiActivityRelay`) listens for the
  `'ai-notebook-activity'` event and **emits an explicit `cell_execute_completed` (origin
  `'ai'`)** for the run. This is necessary because LogBook's execution listener uses
  `NotebookActions.executed` — a _frontend_ signal that never fires for an MCP ZMQ run — so
  there is nothing to attribute via a window; the relay records the run directly from the event
  detail (`status`/`executionCount`/`outputCount`, carried from `executionhook.ts`). Tested in
  `../LogBook/src/__tests__/ai-activity-relay.spec.ts` (4 cases). The earlier `ILogBookExternal`
  token / `external.ts` draft was **removed** — it would have required FlowBook to import a
  LogBook token, violating the decoupling invariant.

**Done — Phase 3 / Option A (verifiable slice): the fix agent's catalog is single-sourced.**

- The custom-fix agent's system prompt (`CUSTOM_FIX_SYSTEM_PROMPT_TEMPLATE`) no longer hardcodes
  its tool list in prose; it renders the read-only + mutator tools from their actual schemas via
  `render_function_schemas()`, so the prompt cannot drift from the tools the agent is given.
  Together with the registry-derived overlap schemas (above) and the built-in fix prompt
  (Phase 2), every fix-tool surface the LLM sees is now generated from a single source.

**Done — Phase 3 follow-through (registry as single schema source):**

- The custom-fix mutator's tools that overlap the registry (`merge_cells`, `move_cell`,
  `mark_diagnostic`, all applied via `apply_fix`) now generate their LLM-facing function
  schemas from the registry (`_registry_fn_schema`), with a drift-guard test. The three
  genuinely mutator-specific tools (`edit_cell_source`, `insert_cell_after`, `delete_cell`)
  have no registry counterpart and stay hand-defined — there is no duplication to remove there.

**Remaining (deliberately deferred, with rationale):**

- **Kernel-backed fix _verification_** (the high-value half of Option A — the agent re-runs
  changed cells and confirms the violation is gone). Deferred because it is a genuine new
  capability, not a refactor: the in-product fix handler is currently stateless (operates on the
  request-body dict, no kernel), so this needs the handler to gain a kernel connection to the
  open notebook + a changed request contract (the frontend supplies the kernel/session id). It
  cannot be end-to-end verified in this environment (needs live JupyterLab + kernel + an LLM
  key), so it should be built as its own focused, live-verified change rather than shipped blind.
- **Wholesale replacement of the litellm streaming loop** (`stream`/`custom_stream` +
  `_ToolCallBuffer` + FIX_PLAN parsing) with a generic agent. The loop works and is tested;
  with the catalog/prompt/validation now single-sourced, replacing the harness is churn with no
  remaining dedup payoff and no local way to verify the swap.
- Folding the single-consumer inspection tools (`fix_tools_readonly`) into the registry would
  centralize but eliminates no duplication; deliberately left as-is.

---

## 1. Motivation

FlowBook now exposes its reproducibility tooling to LLMs through **three independent
surfaces**, each of which re-implements the same catalog of notebook operations over a
different transport and (in one case) hosts its own agent loop:

| Surface                 | Code                                                     | Who runs the LLM                                            | Transport to notebook/kernel                             |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| **MCP server**          | `flowbook/mcp/`                                          | External (Claude Code / CLI)                                | ZMQ kernel (discovery) + Contents API / Y.js             |
| **NBI extension**       | `flowbook/nbi/`                                          | External (NBI chat participant)                             | Frontend bridge `run_ui_command()` → JupyterLab commands |
| **In-product "Fix it"** | `flowbook/server/fix_*` + `src/flowbook/fixsuggester.ts` | **FlowBook itself** (litellm → `anthropic/claude-opus-4-7`) | Request-body notebook **dict** (no kernel)               |

This works, but the design has fragmented along three axes, and a fourth concern — **event
logging / AI attribution via the LogBook extension** — is not yet satisfied by any of the
three paths. This document covers all four.

### 1.1 The tool catalog is defined three times

The six refactoring tools (`alpha_rename`, `remove_inplace`, `insert_deepcopy`,
`mark_diagnostic`, `merge_cells`, `move_cell`) plus the inspection/execution primitives
each exist in three places. The **algorithm cores** are already shared
(`flowbook/scripts/fix_repro_errors.py`: `rename_variable_in_code`, `InplaceRemover`,
`find_actual_variable_name`, `split_cell_magic`, …) — but the **per-tool orchestration**
is triplicated. The three `alpha_rename` bodies are identical except for how they read and
write a cell:

```python
# flowbook/server/fix_dispatcher.py  (dict transport)
for cid in targets:
    cell = _find_code_cell(notebook, cid)
    src  = get_cell_source(cell)
    new_src, renamed = rename_variable_in_code(src, old_name, new_name)
    if renamed:
        set_cell_source(cell, new_src)

# flowbook/mcp/session.py  (kernel + Contents API transport)
for cid in code_order[start_idx:]:
    _, cell = self._find_cell(cid)
    source  = get_cell_source(cell)
    new_source, renamed = rename_variable_in_code(source, old_name, new_name)
    if renamed:
        set_cell_source(cell, new_source)
        self._mark_cell_edited(cid)          # transport-specific hook
self._put_contents_api()                     # transport-specific hook

# flowbook/nbi/tools.py  (frontend-bridge transport)
for i in range(start_idx, num_code_cells):
    cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": i})
    source = cell_data.get('source', '')
    new_source, was_renamed = rename_variable_in_code(source, old_name, new_name)
    if was_renamed:
        await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": i, "source": new_source})
```

The only differences are **(a)** how a cell's source is read, **(b)** how it is written, and
**(c)** optional post-write hooks (`_mark_cell_edited`, `_put_contents_api`). Consequences:

- Every new tool, or change to an existing one, must be written and tested **three times**.
- The catalogs have already drifted: **27** MCP tools vs **30** NBI tools vs **~17** Fix-it tools.
- Tool descriptions are authored independently and can diverge from behavior.

### 1.2 Two different "what is a notebook" abstractions

MCP and NBI operate on a **live, kernel-connected** notebook; they can re-run cells and read
fresh reproducibility metadata. Fix-it operates on an **inert dict** carried in the request
body and never touches the kernel — so it cannot verify that a fix actually worked. This is a
capability gap, not merely duplication.

### 1.3 Two LLM-invocation philosophies

MCP/NBI = "expose tools, let the user's agent drive." Fix-it = "FlowBook embeds its own agent
with its own API key and model." So `fix_suggester.py` re-derives a fourth mini-harness
(streaming-delta accumulation, tool-call buffering, `<FIX_PLAN>` parsing, a read-only/mutator
tool split) that overlaps heavily with what Claude Code already does.

### 1.4 None of the three paths is correctly attributed to LogBook (see §6)

LogBook (`../LogBook`) is a frontend JupyterLab extension that logs every notebook event with
an `origin` of `'system' | 'user' | 'ai'`. It marks an event `'ai'` only through three hooks,
all of which are currently keyed to **NBI**, not FlowBook. As a result, FlowBook's
LLM-initiated actions are today logged as `user`/`system` (or, for MCP, not logged at all).
The requirement: **when LogBook is enabled, every action taken by an LLM through any of the
three FlowBook paths must produce a LogBook event marked `origin: 'ai'`.** §6 specifies how.

### What is already good (and must be preserved)

- **One algorithm core** in `flowbook/scripts/fix_repro_errors.py`, imported by all surfaces.
- **One grounding document** (`REPRODUCIBILITY_PRIMER.md`) referenced everywhere.
- **Three transports for real reasons:** MCP must run headless/standalone; NBI must edit the
  _live_ frontend document and preserve cell identity in the UI; Fix-it must stream into a
  browser with a 30-second surgical undo. The plan does **not** collapse the transports.

---

## 2. Design goals

> **One tool catalog + one agent, behind three thin transport adapters — and a single,
> uniform point where every LLM action is stamped as AI so LogBook attributes it correctly.**

1. The reproducibility tool catalog is defined **once**, declaratively, over an abstract
   `NotebookController` interface.
2. Each transport provides a `NotebookController`. The MCP server and NBI toolset are
   **generated** from the catalog rather than hand-written.
3. The in-product Fix-it agent dispatches against the **same** catalog and (Phase 3) can be
   backed by a real kernel-connected controller so it can verify fixes.
4. The system prompt and fix taxonomy are built **once** from the primer + the catalog's own
   descriptions, so what the LLM is told cannot drift from what exists.
5. Every mutation/execution performed by an LLM is stamped with an **AI actor** at the
   controller layer, and surfaced to LogBook as an `origin: 'ai'` event — best-effort, so
   FlowBook never hard-depends on LogBook being installed.

Non-goals: merging the three transports; changing the kernel reproducibility enforcer;
changing the formal model in `FORMAL_DEVELOPMENT.md`.

---

## 3. Target architecture

```
                 ┌───────────────────────────────────────────────┐
                 │  Tool registry  (flowbook/tools/registry.py)   │
                 │  Tool(name, description, json_schema,          │   ← single source of truth
                 │       handler(controller, **args) -> dict)     │
                 └───────────────────────────────────────────────┘
                                     │  handlers call ↓
                 ┌───────────────────────────────────────────────┐
                 │  NotebookController  (Protocol)                │
                 │  cell_order · read_source · write_source ·     │
                 │  run_cell · metadata · status · insert/delete/ │
                 │  move · checkpoint   +  actor / ai-attribution │
                 └───────────────────────────────────────────────┘
          ┌──────────────────┬──────────────────────┬──────────────────────┐
   KernelController     BridgeController        DictController
   (MCP / CLI:          (NBI: run_ui_command    (Fix-it request dict;
    ZMQ + Contents API)  frontend bridge)         Phase 3: alias to Kernel)
          │                     │                       │
   FastMCP tools          NBI toolset             in-product agent loop
   generated from         generated from          dispatches against the
   registry               registry                same registry
          │                     │                       │
          └──── each controller stamps actor='ai' and feeds LogBook (see §6) ───┘
```

### 3.1 Module layout (new)

```
flowbook/tools/
├── __init__.py
├── controller.py      # NotebookController Protocol + ToolError + AI-actor context
├── registry.py        # Tool dataclass + REGISTRY + lookup/schema helpers
├── reproducibility.py # the 6 refactoring tool handlers (controller-based)
├── inspection.py      # read/list/status/metadata handlers
├── execution.py       # run_cell / run_actionable* handlers
├── structure.py       # insert/delete/merge/move handlers
└── prompt.py          # build_system_prompt(primer, tools) + fix taxonomy
flowbook/tools/adapters/
├── kernel_controller.py   # wraps NotebookSession (MCP/CLI)
├── bridge_controller.py   # wraps NBI run_ui_command response object
└── dict_controller.py     # wraps an in-memory notebook dict (Fix-it)
```

`flowbook/scripts/fix_repro_errors.py` stays as the AST algorithm library.

### 3.2 The `NotebookController` Protocol

The least common denominator of what the three transports already do, keyed by **cell id**
(the `BridgeController` translates id↔index internally, as `nbi/tools.py` already does).

```python
# flowbook/tools/controller.py
from typing import Protocol, Sequence, Optional

class ToolError(Exception):
    """Handler-level failure; transports map this to their own error channel."""

class NotebookController(Protocol):
    # --- identity / attribution ---
    actor: str                 # "user" | "ai" — every write/exec is stamped with this (§6)

    # --- read ---
    def cell_order(self) -> Sequence[str]: ...
    def read_source(self, cell_id: str) -> str: ...
    def read_outputs_text(self, cell_id: str, max_chars: int = 2000) -> str: ...
    def metadata(self, cell_id: str) -> dict: ...
    def status(self) -> dict: ...
    def next_actionable(self) -> Optional[dict]: ...

    # --- write (source/structure) — each implementation emits an AI event (§6) ---
    def write_source(self, cell_id: str, source: str) -> None: ...
    def insert_after(self, after_cell_id: Optional[str], source: str, kind: str = "code") -> str: ...
    def delete_cell(self, cell_id: str) -> None: ...
    def move_after(self, cell_id: str, after_cell_id: str) -> None: ...

    # --- execution (may raise on read-only controllers) ---
    def supports_execution(self) -> bool: ...
    def run_cell(self, cell_id: str) -> dict: ...

    # --- checkpoint (optional) ---
    def checkpoint(self) -> str: ...
    def restore(self, checkpoint_id: str) -> None: ...
```

- **`supports_execution()`** lets the registry expose execution tools only on controllers with
  a kernel. `DictController` returns `False`; this is what lets Fix-it reuse the catalog now
  (no kernel) and gain execution later (Phase 3) with no handler changes.
- Transport quirks that are **not** part of a tool's contract (Contents API push,
  `_mark_cell_edited`, comm `notebook_structure`, **and AI-attribution emission to LogBook**)
  live **inside** the controller's `write_source`/`insert_after`/`move_after`/`run_cell`, where
  `session.py` and `nbi/tools.py` already put their transport hooks. Handlers stay
  transport-agnostic.

### 3.3 The `Tool` registry

```python
# flowbook/tools/registry.py
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class Tool:
    name: str
    description: str                 # the ONE description, reused for MCP/NBI/agent/prompt
    parameters: dict                 # JSON Schema (TOOL_ARG_SCHEMAS, expanded to full schema)
    handler: Callable[..., Any]      # handler(controller, **args) -> dict
    category: str                    # "inspect"|"execute"|"refactor"|"structure"|"checkpoint"
    requires_execution: bool = False # gated by controller.supports_execution()
    mutates: bool = False            # read-only tools available to every agent phase

REGISTRY: list[Tool] = [...]
def get(name: str) -> Tool: ...
def schemas(*, mutating: bool | None, executing: bool) -> list[dict]: ...
```

A refactoring handler is the de-duplicated body of the three `alpha_rename`s:

```python
# flowbook/tools/reproducibility.py
from flowbook.scripts.fix_repro_errors import rename_variable_in_code
from flowbook.tools.controller import NotebookController, ToolError

def alpha_rename(ctrl: NotebookController, *, cell_id: str, old_name: str, new_name: str) -> dict:
    order = list(ctrl.cell_order())
    if cell_id not in order:
        raise ToolError(f"Cell '{cell_id}' not found")
    modified = []
    for cid in order[order.index(cell_id):]:
        new_src, renamed = rename_variable_in_code(ctrl.read_source(cid), old_name, new_name)
        if renamed:
            ctrl.write_source(cid, new_src)   # transport hooks + AI attribution live here
            modified.append(cid)
    if not modified:
        raise ToolError(f"'{old_name}' not found from {cell_id} onward")
    return {"modified_cells": modified, "old_name": old_name, "new_name": new_name}
```

This single function replaces `fix_dispatcher._alpha_rename`, `NotebookSession.alpha_rename`,
and `nbi.tools.alpha_rename`.

### 3.4 Generated transport adapters

- **MCP** (`flowbook/mcp/server.py`): replace ~27 `@mcp.tool` functions with a loop that
  registers each `Tool` against a `KernelController`. Keep the `_logged_tool` event-logging
  decorator and `_cell_label` formatting inside the generated wrapper.
- **NBI** (`flowbook/nbi/tools.py`): same loop over a `BridgeController` built from the NBI
  `response`. The generator applies `@nbapi.tool`/`@nbapi.auto_approve`. The toolset-disabling
  calls stay in `extension.py`.

The MCP and NBI tool lists become **the same list by construction**; drift is impossible.

### 3.5 Unified prompt & taxonomy

```python
# flowbook/tools/prompt.py
def build_system_prompt(*, mutating: bool, executing: bool) -> str:
    return TEMPLATE.format(
        primer=load_primer(),                              # REPRODUCIBILITY_PRIMER.md
        tools=render(registry.schemas(mutating=mutating, executing=executing)),
        taxonomy=FIX_TAXONOMY,                              # one copy of the fix taxonomy
    )
```

`fix_suggester.py` builds its prompt from this; the skill markdown references a generated
`flowbook/docs/_generated/tool_catalog.md`. The LLM is told about exactly the tools that exist.

---

## 4. Implementation plan

Phased so each phase ships independently and never leaves the tree broken. Phases 1–2 are pure
refactors with no user-visible behavior change.

### Phase 0 — Scaffolding (no behavior change)

1. Create `flowbook/tools/` package and `flowbook/tools/tests/__init__.py`.
2. Add `controller.py` (`NotebookController`, `ToolError`, AI-actor context).
3. Add `registry.py` with an empty `REGISTRY`, `Tool`, `get`, `schemas`.

**Exit:** package imports; CI green.

### Phase 1 — Collapse the tool catalog (high payoff, low risk)

For each category, in order `refactor → inspect → structure → execute → checkpoint`:

1. Write the controller-based handler, lifting the body from the most complete existing impl.
2. Add a `Tool(...)` entry with the JSON schema migrated from `fix_models.TOOL_ARG_SCHEMAS`
   (expanded from `Set[str]` to full JSON Schema with types/descriptions for the LLM).
3. Implement/extend the three controllers, delegating to existing code:
   - `KernelController` wraps `NotebookSession` (`_find_cell`, `set_cell_source`,
     `_mark_cell_edited`, `_put_contents_api`).
   - `BridgeController` wraps the NBI `response` (`run_ui_command('flowbook:…')`, id↔index).
   - `DictController` operates on the request dict (lifts `_find_code_cell`, `get/set_cell_source`).
4. Unit-test each handler against an in-memory `DictController` and the existing fixtures.

Cut the three surfaces over one at a time, keeping old code until each new path is green:

5. **MCP:** generator in `server.py` over `KernelController`; keep `_logged_tool`. `pytest flowbook/mcp/`.
6. **NBI:** generator in `nbi/tools.py` over `BridgeController`. `pytest flowbook/nbi/`.
7. **Fix-it apply:** point `fix_dispatcher.apply_fix` at registry handlers over a
   `DictController` (becomes a ~10-line shim; keep `ApplyFixResponse` as the wire type).
   `pytest flowbook/server/`.
8. Delete the dead duplicate bodies in `session.py`, `nbi/tools.py`, `fix_dispatcher.py`.

**Exit:** one definition per tool; MCP/NBI lists generated; `git grep "def alpha_rename"`
returns one handler (+ optional controller shims); all tests pass.

### Phase 2 — Unify prompt & taxonomy (low risk)

1. Move the fix taxonomy into `flowbook/tools/prompt.py` as `FIX_TAXONOMY`.
2. Rewrite `fix_suggester.build_system_prompt` / `CUSTOM_FIX_SYSTEM_PROMPT_TEMPLATE` to call
   `prompt.build_system_prompt(...)`, rendering tool descriptions from `REGISTRY`.
3. Generate `flowbook/docs/_generated/tool_catalog.md`; a `pytest` test fails if `REGISTRY`
   changes without regenerating it. Skills reference the generated file.
4. Extend the `sync-spec` discipline to check the registry against primer/taxonomy.

**Exit:** one taxonomy; prompt cannot drift from tools; a test guards it.

### Phase 3 — Unify the agent (higher payoff, more design)

**Option A (recommended end state):** Replace the bespoke litellm tool-plumbing in
`fix_suggester.py` (`stream`/`custom_stream`, `_ToolCallBuffer`, `<FIX_PLAN>` parsing) with a
thin agent that dispatches against `REGISTRY`, gated by `controller.supports_execution()`. Keep
litellm as the model client (provider-agnostic, env-var keys, `fix_model` traitlet). Give
Fix-it a **`KernelController`** when a kernel is reachable so it can **re-run and verify** fixes
(closing §1.2), falling back to `DictController` (propose-only) otherwise. The 30-second undo in
`fixsuggester.ts` is unaffected (it uses pre/post snapshots the handlers already return).

**Option B (smaller intermediate):** Keep the loop but point its `dispatch_read_only_tool` /
`dispatch_mutator_tool` at `REGISTRY` over a `DictController`. Unifies the tool _set/schemas_
even though the harness stays bespoke; `fix_tools_readonly.py` / `fix_tools_mutator.py` become
adapters, then are deleted.

**Exit (A):** one agent; Fix-it verifies fixes; most of `fix_suggester.py`'s plumbing,
`fix_tools_readonly.py`, and `fix_tools_mutator.py` are deleted.

### Phase 4 — LogBook AI attribution (cross-cutting; see §6 for the design)

Can be done immediately after Phase 1 (it rides on the controller seam). Concretely:

1. **FlowBook frontend** — add a loosely-coupled LogBook bridge in `src/flowbook/`:
   - Fix-it and NBI-bridge mutations run inside an **AI-attributed Yjs transaction**
     (`sharedModel.transact(fn, FLOWBOOK_AI_ORIGIN)`), and frontend-driven executions open an
     **AI window** around the run.
   - A new `src/flowbook/logbookbridge.ts` relays **MCP-origin** actions (received on the
     existing comm / `flowbook_update` IOPub channel, now carrying `actor: 'ai'`) into LogBook.
2. **Kernel/MCP protocol** — add an `actor` field to `flowbook_update` and to MCP's
   `session.log_event`, stamped from `controller.actor`. This is what lets the frontend relay
   distinguish AI-initiated kernel activity from human activity.
3. **LogBook (sibling repo, additive change)** — generalize the hardcoded NBI hooks into a
   configurable AI-origin set and export a minimal attribution token (see §6.4). Ship behind
   LogBook's existing optional-integration pattern so FlowBook degrades gracefully when LogBook
   is absent.

**Exit:** with LogBook enabled, every LLM action from all three paths appears in the JSONL log
with `origin: 'ai'`; with LogBook absent, FlowBook behaves exactly as before.

---

## 5. File-by-file change map

| File                                                                 | Phase | Change                                                                                                                               |
| -------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `flowbook/tools/controller.py`                                       | 0     | **New.** Protocol + errors + AI-actor context.                                                                                       |
| `flowbook/tools/registry.py`                                         | 0–1   | **New.** `Tool`, `REGISTRY`, lookup/schema helpers.                                                                                  |
| `flowbook/tools/{reproducibility,inspection,execution,structure}.py` | 1     | **New.** Single-source handlers.                                                                                                     |
| `flowbook/tools/adapters/{kernel,bridge,dict}_controller.py`         | 1     | **New.** Transport adapters; AI-attribution hooks.                                                                                   |
| `flowbook/tools/prompt.py`                                           | 2     | **New.** `build_system_prompt`, `FIX_TAXONOMY`.                                                                                      |
| `flowbook/mcp/server.py`                                             | 1     | Generator replaces per-tool functions; keep `_logged_tool`, lifespan.                                                                |
| `flowbook/mcp/session.py`                                            | 1, 4  | Keep session/kernel/Contents-API; delete the 6 refactor methods; `log_event` gains `actor`.                                          |
| `flowbook/nbi/tools.py`                                              | 1     | Generator over `BridgeController`.                                                                                                   |
| `flowbook/server/fix_dispatcher.py`                                  | 1     | Reduce to a shim over registry handlers, or delete.                                                                                  |
| `flowbook/server/fix_models.py`                                      | 1–2   | `TOOL_ARG_SCHEMAS` derived from `REGISTRY`; keep wire types + `validate_plan`.                                                       |
| `flowbook/server/fix_suggester.py`                                   | 2–3   | Prompt → `prompt.py` (P2); loop → registry dispatch (P3).                                                                            |
| `flowbook/server/fix_tools_readonly.py`, `fix_tools_mutator.py`      | 3     | Adapters over registry, then delete.                                                                                                 |
| `flowbook/kernel/protocol.py`                                        | 4     | Add `actor` to `flowbook_update` / comm messages.                                                                                    |
| `src/flowbook/protocol.ts`                                           | 4     | Mirror the `actor` field.                                                                                                            |
| `src/flowbook/logbookbridge.ts`                                      | 4     | **New.** Best-effort relay of FlowBook AI actions → LogBook.                                                                         |
| `src/flowbook/fixsuggester.ts`                                       | 4     | Apply fixes inside an AI-attributed transaction / AI window.                                                                         |
| `src/flowbook/nbibridge.ts`                                          | 4     | `flowbook:edit-cell-source` / `flowbook:run-cell` mutate inside an AI-attributed transaction.                                        |
| `flowbook/docs/_generated/tool_catalog.md`                           | 2     | **New, generated.** Referenced by skills.                                                                                            |
| `.claude/commands/*.md`, `.claude/agents/reproducibility-fixer.md`   | 2     | Reference the generated catalog.                                                                                                     |
| `../LogBook/src/origin.ts`                                           | 4     | **(sibling repo, drafted)** AI-marker registries + `register*`/`isAi*`/`correlationIdFromArgs` helpers; seed `'flowbook'` tx origin. |
| `../LogBook/src/tokens.ts`                                           | 4     | **(sibling repo, drafted, new)** `ILogBookExternal` token + `IExternalToolCall`.                                                     |
| `../LogBook/src/external.ts`                                         | 4     | **(sibling repo, drafted, new)** `LogBookExternal` impl (`beginAiBlock`, `emit`, `emitToolCall`).                                    |
| `../LogBook/src/index.ts`                                            | 4     | **(sibling repo, drafted)** Provide `ILogBookExternal`.                                                                              |
| `../LogBook/src/__tests__/origin.spec.ts`                            | 4     | **(sibling repo, drafted)** Tests for FlowBook attribution + opt-in prefix.                                                          |

---

## 6. LogBook integration — attributing all three paths as `origin: 'ai'`

### 6.1 What LogBook is, and the constraint it imposes

LogBook (`../LogBook`) is a **frontend-only** JupyterLab extension. Per-notebook listeners
(`src/listeners/{edits,execution,structure,kernel,lifecycle,…}.ts`) observe the live notebook
model and JupyterLab signals and emit events with this envelope (`src/types/events.ts`,
`logbook/models.py`):

```jsonc
{
  "event_id": "...",
  "timestamp": "...",
  "origin": "system|user|ai",
  "correlation_id": "msg-… | null",
  "kind": "cell_source_changed | cell_execute_completed | …"
  /* payload */
}
```

Events are buffered and flushed to JSONL on disk via the Contents API; `logbook/ingest.py`
is an **offline** CLI that indexes JSONL → SQLite. **There is no server-side ingestion endpoint
— no external process can POST an event to LogBook.** Everything LogBook records, it observes
in the browser.

AI attribution is decided in `src/origin.ts` by an `aiInFlight` counter per panel; while it is
`> 0`, events are emitted with `origin: 'ai'`. The counter is driven by **three NBI-specific
hooks** (constants in `src/config.ts` / `src/origin.ts`):

1. **Yjs transaction origin** — `ydoc.on('beforeTransaction')`; if `tr.origin === NBI_TX_ORIGIN`
   (`'nbi'`), `aiInFlight++` (decremented on `afterTransaction`).
2. **Command prefix** — `app.commands.commandExecuted` with an id starting
   `NBI_COMMAND_PREFIX` (`'notebook-intelligence:'`) opens an AI window across the command's
   promise.
3. **NBI chat token** — `INbiChatObservable` signals produce the `ai_prompt_sent` /
   `ai_tool_call_*` events and supply `correlation_id` via the `__nbiChatMessageId` arg.

`OriginTracker` already exposes `beginAiBlock(panel, correlationId?) -> dispose` — the exact
primitive FlowBook needs — but it is **not currently reachable** from another extension (LogBook
provides no token).

### 6.2 Gap analysis per FlowBook path

| Path                  | Where the mutation/execution actually happens                                                      | Does LogBook see it?                                                                                                             | Marked AI today?                                                                                                     | Gap                                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **In-product Fix-it** | **Frontend** (`fixsuggester.ts` edits the notebook model)                                          | Yes                                                                                                                              | **No** — edits aren't tagged `'nbi'` and aren't `notebook-intelligence:` commands                                    | Tag the apply with an AI origin / open an AI window.                                                                      |
| **NBI extension**     | **Frontend** via `flowbook:`-prefixed bridge commands (`nbibridge.ts` → `sharedModel.setSource()`) | Yes                                                                                                                              | **No** — `'flowbook:'` ≠ the recognized `'notebook-intelligence:'` prefix; bridge transactions aren't tagged `'nbi'` | Same as Fix-it, in the bridge handlers; add chat correlation.                                                             |
| **MCP server**        | **Out-of-process**: Contents API PUT → Y.js, and ZMQ exec on shared kernel                         | **Mostly no** — Contents-API/Y.js sync transactions don't carry an AI origin; ZMQ executions bypass the frontend execute signals | **No**                                                                                                               | Needs a frontend **relay**: the kernel/MCP must signal `actor: 'ai'`, and the FlowBook frontend re-emits it into LogBook. |

The MCP path is the hard one: a frontend-only observer fundamentally cannot attribute an action
taken by a separate process over ZMQ + Contents API. It must be told.

### 6.3 The unifying mechanism: stamp the actor once, surface it per transport

The controller layer (§3.2) is the single choke point. Set `controller.actor = "ai"` whenever a
controller is created on behalf of an LLM (always true for MCP and NBI; true for Fix-it when the
agent is driving). Each controller's write/exec methods are then responsible for surfacing that
actor to LogBook in the way its transport allows:

- **`BridgeController` (NBI) and the Fix-it frontend apply** — both mutate the **frontend**
  model, so they use LogBook's own attribution hooks directly:
  - Wrap every source/structure mutation in `sharedModel.transact(fn, FLOWBOOK_AI_ORIGIN)`.
  - Wrap executions in an AI window (`OriginTracker.beginAiBlock`, obtained via the token in
    §6.4, or — interim — by issuing the action under an AI-recognized command).
  - Pass a correlation id when one exists (the NBI chat message id; a Fix-it "fix session" id).

- **`KernelController` (MCP)** — out-of-process, so it cannot touch the frontend. Instead:
  1. The kernel's `flowbook_update` protocol message (`flowbook/kernel/protocol.py`,
     `src/flowbook/protocol.ts`) gains an **`actor`** field, stamped from `controller.actor`.
     Likewise MCP's `session.log_event` records `actor`.
  2. A new **`src/flowbook/logbookbridge.ts`** in the FlowBook frontend plugin listens on the
     existing comm / IOPub `flowbook_update` channel (the plugin already subscribes for
     reproducibility metadata) and, for any update with `actor === 'ai'`, opens an AI window /
     emits the corresponding LogBook event. This turns out-of-band MCP actions into properly
     attributed frontend events **without** giving LogBook a server endpoint and without MCP
     knowing LogBook exists.

This keeps the coupling minimal and one-directional: FlowBook stamps `actor`; the FlowBook
frontend (the only component co-located with LogBook) does the attribution.

### 6.4 LogBook-side change (small, additive, optional-safe) — **drafted**

FlowBook cannot fully self-attribute without one additive change in LogBook, because the AI
hooks were hardcoded to NBI and `OriginTracker` was not reachable from another extension. The
LogBook-side draft (implemented in `../LogBook`, all changes additive and behind the existing
optional-integration pattern) is:

1. **Generalize the AI markers into registries** (`src/origin.ts`). The hardcoded
   `tr.origin === NBI_TX_ORIGIN` / `id.startsWith(NBI_COMMAND_PREFIX)` checks become lookups
   against module-level sets, with `registerAiTransactionOrigin(origin)`,
   `registerAiCommandPrefix(prefix)`, and `registerCorrelationArg(key)` so a new assistant
   needs **no** edit to LogBook. Helpers `isAiTransactionOrigin`, `isAiCommandId`,
   `correlationIdFromArgs` are exported. Seeding:
   - **Transaction origins** seeded with `'nbi'` **and** `'flowbook'` — both tags are used
     _only_ for AI mutations, so they are safe to treat as AI unconditionally. FlowBook wraps
     its in-frontend AI edits (Fix-it applier + NBI-bridge handlers) in
     `sharedModel.transact(fn, 'flowbook')`; this is what flips those `cell_source_changed`
     events to `origin: 'ai'`.
   - **Command prefixes** seeded with `'notebook-intelligence:'` **only**. `'flowbook:'` is
     deliberately **not** auto-registered, because FlowBook dispatches `flowbook:` commands
     from both the AI bridge _and_ human UI (toolbar/panels) — keying AI off the prefix would
     mislabel human clicks. `FLOWBOOK_COMMAND_PREFIX` is exported for a deployment that uses
     `flowbook:` exclusively for AI to opt in. FlowBook's AI executions instead open an
     explicit window via `beginAiBlock` (below).
2. **~~A public token `ILogBookExternal`~~ — SUPERSEDED; see the "Implementation status" section
   at the top, which is authoritative.** This subsection (and the token/`emitToolCall` design in
   §5/§6.3 below) was the _original plan_. It was **rejected and replaced** because having
   FlowBook consume a LogBook-provided token makes FlowBook import (depend on) LogBook. The
   shipped design is fully decoupled: executions are announced via a one-way
   `CustomEvent('ai-notebook-activity', …)` that LogBook listens for in `ai-activity-relay.ts`;
   no token, no import in either direction; `tokens.ts`/`external.ts` were removed.

Tests added to `../LogBook/src/__tests__/origin.spec.ts` cover the `'flowbook'` transaction
origin, the deliberate non-attribution of bare `flowbook:` commands, the opt-in path, and
`correlationIdFromArgs`. The TS suite is Node/Jest (no conda env); LogBook Python tests/CLI use
the `logbook` conda env.

**Why `ai_tool_call_*` for the MCP relay (not re-emitting `cell_source_changed`):** when MCP
PUTs via the Contents API, the Y.js sync still produces a `cell_source_changed` in this browser
(attributed `user`/`system` because the sync transaction origin isn't `'flowbook'`). Emitting
`ai_tool_call_*` records the AI action **without duplicating** that cell event — it satisfies
"produce events marked AI" cleanly. Flipping the synced cell event itself to `'ai'` is a
nice-to-have that depends on the collaboration provider's transaction origin and is out of
scope for this draft.

**Interim, zero-LogBook-change fallback** (if the LogBook change must lag): FlowBook frontend
mutations can tag their Yjs transactions with the already-recognized `'nbi'` origin — correct
`origin: 'ai'` immediately, but it mislabels the agent as NBI and does nothing for the MCP
relay. Treat as a stopgap; the registry generalization above is the real fix.

### 6.5 Acceptance criteria (the user's requirement, made testable)

With LogBook enabled, for each of the three paths, an end-to-end test asserts that an
LLM-initiated edit/execution produces JSONL events whose `origin == "ai"`:

- **Fix-it:** click a suggested fix → `cell_source_changed` events for the modified cells carry
  `origin: "ai"` (and, on Phase-3 verify, the re-run `cell_execute_*` events too).
- **NBI:** an NBI chat turn that calls `edit_cell_source` / `run_cell` → corresponding
  `cell_source_changed` / `cell_execute_*` events carry `origin: "ai"` and a `correlation_id`
  linking to the chat message.
- **MCP:** a Claude Code session running `alpha_rename` / `run_cell` over MCP → the FlowBook
  relay emits `cell_source_changed` / `cell_execute_*` with `origin: "ai"`.
- **Negative:** with LogBook **not** installed, all three paths behave exactly as today
  (no errors, no new dependency).

---

## 7. Testing strategy

- **Handler unit tests (P1):** each registry handler against a `DictController` fixture — fast,
  no kernel. Reuse `flowbook/server/tests` + `flowbook/mcp/tests` notebook fixtures.
- **Cross-controller parity test:** the same handler against `DictController` and a stub
  controller, asserting identical `modified_cells`/sources — guards transport behavioral drift.
- **Generator tests:** assert MCP tool count == NBI tool count == registry count; all schemas valid.
- **Prompt-drift test (P2):** fails if `REGISTRY` changes without regenerating `tool_catalog.md`.
- **LogBook attribution tests (P4):** the §6.5 acceptance criteria, plus the negative case.
- **Existing suites green** at every phase boundary: `pytest flowbook/{mcp,nbi,server,kernel}/`,
  plus LogBook's `jest` origin specs after the `origin.ts` generalization.
- Run all Python tests in the **`flowbook` conda env** (system python3.9 fails on `str | None`).
- Re-run `flowbook/nbi/MANUAL_UI_TESTS.md` after the NBI cutover and add a LogBook-attribution
  manual check.

---

## 8. Risks & trade-offs

| Risk                                                          | Likelihood | Mitigation                                                                                                                                                       |
| ------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Behavior drift between transports during cutover              | Medium     | One surface at a time behind its suite; cross-controller parity test; controllers isolate quirks.                                                                |
| `BridgeController` id↔index bugs                              | Medium     | NBI already does index math; focused tests; keep `@A` conversion in `flowbook/util/cell_index.py`.                                                               |
| Phase 3 changes Fix-it UX subtly                              | Medium     | Do Option B first; gate Option A behind the `fix_model` traitlet + feature flag.                                                                                 |
| LogBook change lands late / coordination across two repos     | Medium     | Ship the §6.4 interim `'nbi'`-tag fallback for frontend paths; land the relay + token next.                                                                      |
| MCP relay double-counts or mis-times AI windows               | Medium     | Drive windows off discrete `actor:'ai'` update messages with explicit begin/end, not heuristics; unit-test the relay against recorded `flowbook_update` streams. |
| Over-abstraction: a tool needs a primitive the Protocol lacks | Low        | Protocol is the LCD of three working impls; extend additively on real need.                                                                                      |
| FlowBook hard-depending on LogBook                            | Low        | LogBook integration is `optional` in the plugin graph; all hooks are best-effort no-ops when absent.                                                             |

---

## 9. Outcome

- **After Phase 1:** every tool defined and tested **once**; MCP/NBI lists generated; catalogs
  cannot drift.
- **After Phase 2:** the LLM is told about exactly the tools that exist; one fix taxonomy; a
  test keeps prose in sync.
- **After Phase 3:** one agent instead of three harnesses; Fix-it can re-run and **verify** fixes.
- **After Phase 4:** with LogBook enabled, **all three LLM paths produce events marked
  `origin: 'ai'`** — including the out-of-process MCP path, via a best-effort frontend relay —
  and FlowBook still runs unchanged when LogBook is absent.

The three transports remain, because they serve three real runtime contexts. What goes away is
everything that did not need to be three: the catalog, the agent, the prompt — and the silent
gap where LLM actions went unattributed.
