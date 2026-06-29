# FlowBook · NotebookIntelligence · LogBook — Ecosystem Architecture

**Audience:** an engineer new to these repos who needs to understand how the three
JupyterLab extensions fit together, how an LLM drives a notebook through each of them, and
where the code lives.

**Scope:** the _interaction_ architecture. Per-subsystem detail lives in the documents linked
in [§11](#11-related-documents); this doc is the map that ties them together.

---

## Table of contents

1. [The three extensions at a glance](#1-the-three-extensions-at-a-glance)
2. [The shared substrate: one kernel, one document](#2-the-shared-substrate-one-kernel-one-document)
3. [The three ways an LLM drives a notebook](#3-the-three-ways-an-llm-drives-a-notebook)
4. [The unified tool layer (`flowbook/tools`)](#4-the-unified-tool-layer-flowbooktools)
5. [The kernel protocol](#5-the-kernel-protocol)
6. [LogBook and AI attribution](#6-logbook-and-ai-attribution)
7. [NotebookIntelligence integration](#7-notebookintelligence-integration)
8. [End-to-end data flows](#8-end-to-end-data-flows)
9. [Implementation structure (file map)](#9-implementation-structure-file-map)
10. [Invariants, conventions, and gotchas](#10-invariants-conventions-and-gotchas)
11. [Related documents](#11-related-documents)
12. [Glossary](#12-glossary)

---

## 1. The three extensions at a glance

Three independently-installable JupyterLab 4 extensions, developed in sibling repos
(`FlowBook/`, `LogBook/`, and the third-party `notebook-intelligence/`):

| Extension                      | Repo                     | What it is                                                                                                                                                                         | Role in the ecosystem                                                                              |
| ------------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **FlowBook**                   | `FlowBook/`              | rerun-consistency engine: a custom IPython kernel that tracks reads/writes per cell and enforces rerun-consistency, plus UI, a server extension, an MCP server, and an NBI plugin. | The thing being driven. Defines _what a correct notebook is_ and exposes tools to fix it.          |
| **NotebookIntelligence (NBI)** | `notebook-intelligence/` | A third-party in-JupyterLab AI chat assistant with a tool/toolset system.                                                                                                          | One of the ways an LLM drives the notebook (an in-lab chat). FlowBook registers a toolset with it. |
| **LogBook**                    | `LogBook/`               | A passive observer that records every notebook event (edits, executions, AI chat) to a JSONL event log, tagging each with an origin (`system`/`user`/`ai`).                        | The audit/telemetry layer. Records _who did what_ — including which actions were AI-driven.        |

Each runs standalone. Installed together they compose, but **no extension imports another**
(see [§10](#10-invariants-conventions-and-gotchas) — the decoupling invariant). They cooperate
through three shared substrates — the kernel, the notebook document, and a small set of string
contracts — never through code dependencies.

```
                        ┌───────────────────────────────────────────────┐
                        │                 JupyterLab                      │
                        │                                                 │
   external LLM         │   ┌─────────┐   ┌─────────┐   ┌─────────────┐   │
   (Claude Code) ──MCP──┼─▶ │ FlowBook│   │   NBI   │   │   LogBook   │   │
                        │   │ frontend│   │  chat   │   │ (observer)  │   │
                        │   └────┬────┘   └────┬────┘   └──────┬──────┘   │
                        │        │  shared notebook (Y.js)     │ observes │
                        │        ▼             ▼               ▼          │
                        │   ┌─────────────────────────────────────────┐  │
                        │   │     shared FlowBook kernel (ZMQ/comm)     │  │
                        │   └─────────────────────────────────────────┘  │
                        └───────────────────────────────────────────────┘
```

---

## 2. The shared substrate: one kernel, one document

FlowBook's value depends on a _single source of truth_ for the notebook and a _single kernel_
that tracks rerun-consistency. Multiple clients (the JupyterLab UI, an out-of-process MCP server)
can attach to both at once.

- **Shared notebook (Y.js / `jupyter-collaboration`).** The notebook is a CRDT document. The
  JupyterLab UI edits it directly; an out-of-process client edits it through the Contents API
  (`GET`/`PUT /api/contents/{path}`), which `jupyter-collaboration` projects onto the live Y.js
  doc. All observers (including LogBook) see the same cell sources.
- **Shared kernel (`flowbook_kernel`).** A custom IPython kernel that, on every execution,
  records read/write _locations_ and enforces the four validity predicates (see
  `FORMAL_DEVELOPMENT.md`). Whoever starts the kernel writes a **discovery file**
  (`~/.jupyter/runtime/flowbook-{sha}.json`); a second participant reads it and attaches as an
  additional ZMQ client. The kernel broadcasts rerun-consistency results to _all_ attached clients
  over IOPub (see [§5](#5-the-kernel-protocol)).
- **Cell identity.** Every cell has a stable 4-char id (`flowbook/util/cell_ids.py`). Identity
  must survive edits — tools mutate sources in place rather than delete+reinsert — because the
  kernel keys rerun-consistency state by cell id.

`MCP_ARCHITECTURE.md` is the authoritative description of the MCP↔JupyterLab sharing protocol
(discovery, Contents-API sync, IOPub polling, graceful degradation).

---

## 3. The three ways an LLM drives a notebook

There are three _surfaces_ through which an LLM acts on a notebook. They differ in **who hosts
the LLM** and **how edits/executions reach the kernel and document** — but they share one tool
implementation ([§4](#4-the-unified-tool-layer-flowbooktools)).

| Surface                 | Code                                                     | Who hosts the LLM                                                      | Transport to notebook/kernel                                           | Driven by                                                 |
| ----------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------- |
| **MCP server**          | `flowbook/mcp/`                                          | External (Claude Code / CLI)                                           | ZMQ kernel via discovery + Contents API / Y.js                         | MCP tool calls; skills like `/fix-notebook`, `/basic-run` |
| **NBI extension**       | `flowbook/nbi/`                                          | External (NBI chat participant)                                        | Frontend bridge: `run_ui_command('flowbook:…')` → JupyterLab commands  | NBI chat; `/basic-run-nbi`, `flowbook-nb-fix`             |
| **In-product "Fix it"** | `flowbook/server/fix_*` + `src/flowbook/fixsuggester.ts` | **FlowBook itself** (litellm → `anthropic/claude-opus-4-7` by default) | Request-body notebook dict (stateless); frontend applies edits to Y.js | The violation notice's "fix" / "Other Fix…" buttons       |

Notes:

- **MCP** is for an external agent working a notebook end-to-end (possibly headless, possibly
  alongside a live JupyterLab). It owns or joins a kernel and can run cells.
- **NBI** is for in-lab chat. FlowBook exposes its tools to NBI and _disables NBI's built-in
  notebook-edit/execute toolsets_ so cell identity is preserved (see [§7](#7-notebookintelligence-integration)).
- **In-product Fix-it** is the only surface where _FlowBook hosts the model_. It is deliberately
  stateless (operates on the notebook JSON in the request, no kernel) and streams a diagnosis +
  proposed fixes (or applies a free-form "Other Fix") to the browser, with a 30-second surgical
  undo. Model selection is the `FlowBookExtension.fix_model` traitlet; the provider API key comes
  from standard env vars (litellm resolves them). The feature self-disables if no key is present.

---

## 4. The unified tool layer (`flowbook/tools`)

The rerun-consistency tools (rename a variable across cells, remove a pandas `inplace=True`, insert
a deepcopy, mark a cell diagnostic, merge cells, move a cell, plus inspection/execution) are the
_same operations_ regardless of surface. They are defined **once** and reused by all three.

```
            ┌──────────────────────────────────────────────┐
            │  Tool registry  (flowbook/tools/registry.py)   │  one catalog: name + JSON
            │  Tool(name, description, parameters, handler)  │  schema + handler
            └──────────────────────────────────────────────┘
                              │ handlers call ↓
            ┌──────────────────────────────────────────────┐
            │  NotebookController  (Protocol)                │  cell_order · read_source ·
            │  flowbook/tools/controller.py                  │  write_source · delete_cell ·
            └──────────────────────────────────────────────┘  move_after  (+ actor)
       ┌──────────────────┬──────────────────────┬──────────────────────┐
  KernelController    BridgeController*       DictController
  (MCP/CLI: wraps     (NBI: snapshot the      (in-product fix-it: an
   NotebookSession;    notebook via the        in-memory dict + a
   batches Contents-   bridge, run handler,    mutation log used to
   API push + kernel   replay edits by index)  build ApplyFixResponse)
   notify)
       │                       │                        │
  FastMCP tools          NBI async tools          server fix dispatcher
  call session           call the handler         calls the handler over
  methods that           over a DictController    a DictController
  delegate to handlers   (snapshot/replay)
```

\* NBI is async and its bridge is index-based, so rather than a long-lived controller it uses a
**snapshot → run handler over a `DictController` → replay edits through the bridge** pattern
(`flowbook/nbi/tools.py`). Same handler, different plumbing.

Key modules:

- `controller.py` — the `NotebookController` `Protocol` and the error types `ToolError` /
  `CellNotFoundError` / `NoEffectError`. Handlers raise these; adapters translate them (e.g. the
  MCP layer tolerates a no-op rename, the stateless dispatcher turns them into 4xx).
- `reproducibility.py` — the six refactoring handlers (the single definition that replaced three
  near-identical copies in `mcp/session.py`, `server/fix_dispatcher.py`, `nbi/tools.py`).
- `registry.py` — `REGISTRY: list[Tool]`, with `get(name)` and `names()`.
- `prompt.py` — `render_tool_catalog()` / `render_function_schemas()` and `FIX_TAXONOMY`:
  the LLM-facing tool descriptions, rendered from the registry/schemas so prompts cannot drift
  from the tools that actually exist.
- `adapters/dict_controller.py`, `adapters/kernel_controller.py` — the controller
  implementations.

**Single-source guarantees (enforced by tests in `flowbook/tools/tests/`):**

- The six refactoring tools have one handler each.
- `fix_models.TOOL_ARG_SCHEMAS` (the validation allowlist) is _derived_ from `REGISTRY`.
- The in-product fix prompt and the custom-fix agent prompt render their tool lists from the
  registry / the actual schemas.
- `flowbook/docs/_generated/tool_catalog.md` is generated from the registry (a test fails if it
  drifts).

So: descriptions (prompt), arg contracts (validation), and application (handlers) all flow from
one place. See `LLM_INTEGRATION_DESIGN.md` for the migration history and the parts deliberately
left for later (kernel-backed fix _verification_; a generic-agent rewrite of the litellm loop).

---

## 5. The kernel protocol

`flowbook/kernel/protocol.py` (Python) and `src/flowbook/protocol.ts` (TypeScript) define a
JSON protocol with a `"type"` discriminator, transported two ways:

| Direction       | Transport                                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| Client → kernel | Execute-request metadata (`cell_meta.flowbook`) for Python clients; comm channel (`comm.send`) for the frontend |
| Kernel → client | Custom IOPub message `flowbook_update` (broadcast to **all** attached clients) + the comm channel               |

**Kernel → client** message types: `metadata` (post-execution reads/writes/changed locs, stale
cells, timing, errors), `violation` (a predicate violation), `status` (icon + summary line).

**Client → kernel** message types: `notebook_structure` (set cell order), `cell_edited` (mark
stale), `continue_after_violation`, `sync`, `exec_restore`.

**The `actor` field.** The `metadata` message carries an optional `actor` (`"ai"` | `"user"`).
The kernel reads it from the execute-request metadata (`cell_meta["actor"]`, default `"user"`)
and echoes it on `flowbook_update`. This is what lets a co-located observer attribute an
_out-of-process_ execution (an MCP run on the shared kernel) to the AI even though that run never
touched the frontend. MCP's `run_cell` sends `actor="ai"`; the JupyterLab UI sends nothing
(→ `"user"`). See [§6](#6-logbook-and-ai-attribution).

---

## 6. LogBook and AI attribution

LogBook is a **frontend-only passive observer**. Per-notebook listeners
(`LogBook/src/listeners/`) watch the live document and JupyterLab signals and emit events
(`event_id`, `timestamp`, `origin`, `correlation_id`, `kind`, payload) into a buffered JSONL log
(`logs/<notebook>/<session>/events-*.jsonl`); `LogBook/logbook/` is an offline CLI that indexes
the JSONL into SQLite. There is **no server-side ingestion endpoint** — everything LogBook
records, it observes in the browser.

The interesting field is **`origin: 'system' | 'user' | 'ai'`**. `OriginTracker`
(`LogBook/src/origin.ts`) keeps a per-panel `aiInFlight` counter; while it is `> 0`, observed
events are attributed `'ai'`. It is driven by recognized markers:

- **Yjs transaction origins** in a configurable set (`'nbi'`, `'flowbook'`, extensible via
  `registerAiTransactionOrigin`).
- **Command-id prefixes** (`'notebook-intelligence:'`; `'flowbook:'` is _intentionally excluded_
  because FlowBook's `flowbook:` commands are dispatched by both AI and human UI).
- **The NBI chat token** (`INbiChatObservable`), which also supplies `correlation_id`.

### How FlowBook activity reaches LogBook — the decoupling invariant

**FlowBook imports nothing from LogBook, and vice versa.** Integration is two one-way _string
contracts_, each a no-op when the other side is absent:

| FlowBook activity                         | Mechanism (FlowBook side)                                                                                                                                                                                              | LogBook side                                                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Edits** (fix applier, NBI bridge edits) | `aiTransact(sharedModel, fn)` runs the mutation inside a Yjs transaction tagged `'flowbook'` (`src/flowbook/aiattribution.ts`)                                                                                         | `origin.ts` has `'flowbook'` in its AI-origin set → the observed `cell_source_changed` is `origin: 'ai'`            |
| **Executions** (out-of-process MCP runs)  | `emitAiActivity()` dispatches a DOM `CustomEvent('ai-notebook-activity', {detail:{source:'flowbook', path, cellId, status, …}})`, fired from `executionhook.ts` when a `flowbook_update` arrives with `actor === 'ai'` | `ai-activity-relay.ts` (`installAiActivityRelay`) listens and **emits** a `cell_execute_completed` (`origin: 'ai'`) |

Why the execution path _emits_ rather than opens a window: LogBook's execution listener uses
`NotebookActions.executed` — a **frontend** signal that never fires for an MCP ZMQ run — so there
is no event to re-attribute. The relay must record the run itself, using the detail carried on
the CustomEvent. The earlier design (a LogBook-provided `ILogBookExternal` token that FlowBook
would import) was **rejected and removed** because it created a dependency; the event contract
replaces it.

> **Do not reintroduce a token or package dependency in either direction.** The contracts are
> the strings `'flowbook'` (Yjs origin) and `'ai-notebook-activity'` (DOM event). Either
> extension must build and run with the other absent.

### How NBI activity reaches LogBook

LogBook consumes NBI's `INbiChatObservable` token _optionally_ (`AiChatTap`,
`LogBook/src/ai-chat.ts`): it records `ai_prompt_sent` / `ai_tool_call_*` / `ai_response_*`
events and brackets each tool call in an AI window. NBI's own shared-model mutations are tagged
with the `'nbi'` Yjs origin, and its tool-call commands carry a `__nbiChatMessageId` correlation
arg. (This is a genuine token dependency, but it is _LogBook → NBI_ and optional; it predates and
is independent of FlowBook.)

---

## 7. NotebookIntelligence integration

FlowBook ships an NBI plugin (`flowbook/nbi/`, optional dependency `notebook-intelligence`). On
activation (`flowbook/nbi/extension.py`):

- **Registers a toolset** (`"flowbook-reproducibility"`) of FlowBook tools — metadata/status,
  identity-safe cell ops, execution, the six refactorings, checkpoint/log.
- **Disables NBI's built-in `nbi-notebook-edit` and `nbi-notebook-execute` toolsets**, whose
  delete+reinsert edit pattern would destroy cell identity and break rerun-consistency tracking.
  FlowBook's replacements edit in place.
- Installs Claude slash-commands and registers the FlowBook MCP server into `~/.claude.json`.

NBI tools are `async` and reach the notebook through the **frontend bridge**: they call
`response.run_ui_command('flowbook:…')`, which executes a JupyterLab command registered in
`src/flowbook/nbibridge.ts` (e.g. `flowbook:get-cell`, `flowbook:edit-cell-source`,
`flowbook:run-cell`, `flowbook:move-cell`). All bridge indexing is **code-cell-only** with
`@A` labels. The refactoring tools reuse the unified handlers via snapshot→replay ([§4](#4-the-unified-tool-layer-flowbooktools)).

---

## 8. End-to-end data flows

### A. External agent fixes a notebook over MCP (alongside a live JupyterLab + LogBook)

```
Claude Code ── alpha_rename(cell, old, new) ──▶ MCP server (flowbook/mcp)
  → KernelController over NotebookSession → unified handler edits cell sources
  → Contents API PUT ──▶ Y.js doc ──▶ JupyterLab UI updates; LogBook sees the edit
Claude Code ── run_cell ──▶ session.run_cell(actor="ai")
  → kernel executes, enforces predicates, broadcasts flowbook_update{actor:"ai"} on IOPub
  → FlowBook frontend executionhook receives it → emitAiActivity('ai-notebook-activity')
  → LogBook ai-activity-relay emits cell_execute_completed{origin:"ai"}
```

(Note: an MCP edit pushed via the Contents API carries the _collaboration_ transaction origin,
not `'flowbook'`; attributing MCP _edits_ as AI in LogBook is a known gap — see
`LLM_INTEGRATION_DESIGN.md`. MCP _executions_ are attributed via the `actor` path above.)

### B. In-product "Fix it" (FlowBook hosts the model)

```
violation notice "fix" click ─▶ POST /flowbook/suggest-fix {notebook, cell_id}
  → FixSuggester.stream(): litellm diagnosis + <FIX_PLAN> (prompt rendered from the registry)
  → SSE: diagnosis text, then validated FixPlan ─▶ buttons
button click ─▶ POST /flowbook/apply-fix {notebook, tool, args}
  → fix_dispatcher.apply_fix → unified handler over a DictController → ApplyFixResponse
  → frontend applies sources inside aiTransact('flowbook') ─▶ LogBook sees origin:"ai"
  → 30s undo available (pre-fix snapshot)
```

### C. NBI chat fixes a notebook

```
NBI chat turn ─▶ FlowBook NBI tool alpha_rename(@C, old, new)
  → snapshot notebook via flowbook:get-cell* → unified handler over DictController
  → replay edits via flowbook:edit-cell-source (wrapped in aiTransact('flowbook'))
  → LogBook attributes edits origin:"ai"; NBI chat events recorded via INbiChatObservable
```

---

## 9. Implementation structure (file map)

### FlowBook (`FlowBook/`)

```
flowbook/
├── kernel/                     # the rerun-consistency kernel
│   ├── flowbook_kernel.py      #   do_execute, comm, reads cell_meta["actor"]
│   ├── reproducibility_enforcer.py, models.py, locations.py, change_detector.py
│   └── protocol.py             #   flowbook_update message builders (incl. actor)
├── tools/                      # THE UNIFIED TOOL LAYER (§4)
│   ├── controller.py           #   NotebookController Protocol + error types
│   ├── registry.py             #   REGISTRY of Tool
│   ├── reproducibility.py      #   the six single-source handlers
│   ├── prompt.py               #   render_tool_catalog / render_function_schemas / FIX_TAXONOMY
│   └── adapters/{dict,kernel}_controller.py
├── mcp/                        # MCP server surface (§3)
│   ├── server.py               #   FastMCP tools
│   └── session.py              #   NotebookSession; refactor methods delegate to handlers
├── nbi/                        # NotebookIntelligence surface (§7)
│   ├── extension.py            #   toolset registration; disables NBI edit/execute toolsets
│   └── tools.py                #   async tools; snapshot→handler→replay
├── server/                     # HTTP server extension + in-product Fix-it (§3B)
│   ├── handlers.py             #   /flowbook/{suggest,apply,custom}-fix, execute, kernel-discovery
│   ├── fix_suggester.py        #   litellm streaming agent (built-in + custom-fix loops)
│   ├── fix_dispatcher.py       #   apply_fix → registry handler over DictController
│   ├── fix_models.py           #   wire types + validate_plan; TOOL_ARG_SCHEMAS derived from REGISTRY
│   ├── fix_tools_readonly.py / fix_tools_mutator.py   # custom-fix agent's inspect/mutate tools
│   └── kernel_helper.py        #   execute_code(..., actor=...)
├── scripts/fix_repro_errors.py # the AST transformation primitives (shared by the handlers)
├── util/cell_ids.py, cell_index.py
└── docs/rerun-consistency_PRIMER.md, _generated/tool_catalog.md

src/flowbook/                   # TypeScript frontend
├── plugin.ts                   # activation, kernel discovery, wiring
├── executionhook.ts            # comm handling; fires emitAiActivity on actor:"ai"
├── fixsuggester.ts             # in-product fix UI; applies edits inside aiTransact
├── nbibridge.ts                # flowbook:* commands NBI calls; edits inside aiTransact
├── aiattribution.ts            # aiTransact (Yjs origin 'flowbook') + emitAiActivity (CustomEvent)
└── protocol.ts                 # mirrors kernel protocol incl. actor

# Companion docs at repo root: MCP_ARCHITECTURE.md, FORMAL_DEVELOPMENT.md,
# LLM_INTEGRATION_DESIGN.md, and this file.
```

### LogBook (`LogBook/`)

```
src/
├── index.ts                    # plugin activation; installAiActivityRelay({manager})
├── origin.ts                   # OriginTracker; AI-origin/prefix registries (incl. 'flowbook')
├── ai-activity-relay.ts        # listens for 'ai-notebook-activity' → cell_execute_completed(ai)
├── ai-chat.ts                  # AiChatTap: optional INbiChatObservable consumer
├── listeners/                  # edits.ts, execution.ts, structure.ts, kernel.ts, lifecycle.ts
├── manager.ts, session.ts, emitter.ts, contents.ts, blobs.ts
└── types/events.ts             # the event schema (origin, kinds)
logbook/                        # Python: offline JSONL → SQLite (ingest.py, reader.py, cli.py)
```

### NotebookIntelligence (`notebook-intelligence/`)

Third-party. FlowBook depends on its **public extension API** (`notebook_intelligence.api`:
`@nbapi.tool`, toolset registration, `run_ui_command`) and LogBook depends on its
`INbiChatObservable` token — both optionally.

---

## 10. Invariants, conventions, and gotchas

1. **Decoupling invariant.** FlowBook ⊥ LogBook: no imports, no `package.json`/`pyproject.toml`
   entry, either direction. Integration is the two string contracts in [§6](#6-logbook-and-ai-attribution).
   Each extension builds and runs with the other absent.
2. **One tool definition.** A rerun-consistency operation is implemented once in
   `flowbook/tools/`. Don't re-implement it per surface — add/extend a handler + controller.
3. **Prompts are generated, not hand-maintained.** Tool lists in LLM prompts and
   `tool_catalog.md` render from the registry/schemas. A drift-guard test enforces this.
4. **Cell identity is sacred.** Edit sources in place (`setSource`); never delete+reinsert a
   cell you mean to keep. The kernel and LogBook both key state by cell id.
5. **`actor` defaults to `"user"`.** Only a client acting for an LLM sets `actor="ai"` on the
   execute request (MCP does). The `flowbook:` _command_ prefix is deliberately NOT an AI marker
   because humans use those commands too.
6. **Frontend signals miss out-of-process work.** `NotebookActions.executed` fires only for
   in-UI runs. Anything an MCP client does over ZMQ is invisible to frontend-signal observers —
   hence the `actor` echo + `ai-notebook-activity` relay.
7. **Graceful degradation everywhere.** No Jupyter server → MCP runs standalone (own kernel,
   file-based). No LogBook → attribution calls are no-ops. No `notebook-intelligence` → the NBI
   plugin simply isn't built. No model API key → the in-product fix feature self-disables.
8. **Spec ↔ code sync.** rerun-consistency semantics live in `FORMAL_DEVELOPMENT.md` /
   `REPRODUCIBILITY_PRIMER.md`; keep them in sync with the kernel and the fix prompts.

---

## 11. Related documents

- **`MCP_ARCHITECTURE.md`** — MCP ↔ JupyterLab sharing: discovery files, Contents-API sync, IOPub
  polling, kernel ownership, graceful degradation.
- **`LLM_INTEGRATION_DESIGN.md`** — the design + migration plan for unifying the three LLM
  surfaces on one tool layer, the prompt/validation single-sourcing, the LogBook attribution
  work, and the parts deliberately deferred (kernel-backed fix verification; generic-agent
  rewrite). Read this for _why_ the structure is as it is and what is intentionally unfinished.
- **`FORMAL_DEVELOPMENT.md`** — the formal rerun-consistency model (the four validity predicates,
  staleness propagation, UNRECOVERABLE_MUTATION) with an implementation map.
- **`flowbook/docs/REPRODUCIBILITY_PRIMER.md`** — the canonical prose explanation, embedded in
  the fix prompts.
- **`CLAUDE.md`** — repo conventions and developer commands.
- **LogBook `AGENTS.md` / `docs/origin-attribution-plan.md`** — LogBook's own conventions and the
  origin-attribution design.

---

## 12. Glossary

- **Surface** — one of the three ways an LLM drives the notebook (MCP, NBI, in-product Fix-it).
- **Controller** — a `NotebookController` implementation; the transport-specific adapter a tool
  handler runs against.
- **Handler** — the single, transport-agnostic implementation of a tool (in `flowbook/tools/`).
- **Actor** — `"ai"` or `"user"`; who drove an execution. Carried on the execute request and
  echoed on `flowbook_update`.
- **Origin** — LogBook's `system`/`user`/`ai` attribution on a logged event.
- **Discovery file** — `~/.jupyter/runtime/flowbook-{sha}.json`; lets a second client find and
  attach to a running kernel.
- **`flowbook_update`** — the custom IOPub message type the kernel broadcasts (metadata /
  violation / status).
- **`aiTransact` / `ai-notebook-activity`** — FlowBook's two decoupled attribution contracts (a
  Yjs transaction origin for edits; a DOM event for executions).
