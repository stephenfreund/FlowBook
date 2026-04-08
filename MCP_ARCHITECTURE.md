# FlowBook Tool Architecture

## Overview

FlowBook exposes notebook reproducibility analysis through two AI tool surfaces, a shared tool library, and a custom IPython kernel:

1. **MCP Server** (`flowbook_mcp` CLI) — standalone tool surface for Claude Code CLI; manages its own kernel and notebook state via `NotebookSession`
2. **NBI Extension** (`flowbook/nbi/`) — tool surface for JupyterLab via Notebook Intelligence; communicates with JupyterLab's frontend via `run_ui_command()` bridge
3. **Shared Library** (`flowbook/tools/`) — `FlowBookTools` class and formatters used by MCP; formatters also used by NBI
4. **FlowBook Kernel** — instrumented IPython kernel with reproducibility enforcement, shared between MCP and JupyterLab via kernel discovery

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code / Claude AI                      │
├──────────────────────┬──────────────────────────────────────────┤
│  MCP Server          │  NBI Extension (JupyterLab)              │
│  mcp__flowbook__*    │  mcp__nbi__*                             │
│                      │                                          │
│  flowbook/mcp/       │  flowbook/nbi/                           │
│   server.py          │   tools.py                               │
│     ↓                │     ↓                                    │
│  FlowBookTools       │  run_ui_command() bridge                 │
│   (tools/tools.py)   │     ↓                                    │
│     ↓                │  nbibridge.ts (JupyterLab commands)      │
│  NotebookSession     │     ↓                                    │
│   (mcp/session.py)   │  JupyterLab UI (sharedModel, widgets)   │
│     ↓                │     ↓                                    │
│  ZMQ + Contents API  │  ZMQ + Comm Channel                     │
├──────────────────────┴──────────────────────────────────────────┤
│                    FlowBook Kernel (shared)                      │
│  flowbook/kernel/flowbook_kernel.py                              │
│  ReproducibilityEnforcer → NotebookState                         │
└─────────────────────────────────────────────────────────────────┘
```

## Two Tool Surfaces, Unified API

Both surfaces expose the same tool names and parameters. The implementations differ because NBI must go through JupyterLab's frontend for reliable UI updates, while MCP manages state directly.

### Unified Tool Set

| Tool | Params | Category |
|------|--------|----------|
| `read_cell` | `cell: str = ""` | Cell access (empty = all cells) |
| `edit_cell_source` | `cell, new_source` | Cell editing |
| `add_cell` | `source, cell_type?, after_cell?` | Cell editing |
| `delete_cell` | `cell` | Cell editing |
| `run_cell` | `cell` | Execution |
| `run_actionable_cell` | — | Execution |
| `run_actionable_cells` | — | Execution |
| `run_all_cells` | — | Execution |
| `run_from` | `cell` | Execution (MCP only) |
| `continue_after_violation` | `enabled: bool` | Execution config |
| `get_status` | — | Status |
| `get_next_actionable_cell` | — | Status |
| `get_flowbook_metadata` | `cell` | Status |
| `list_cells` | — | Status (MCP only) |
| `alpha_rename` | `cell, old_name, new_name` | Refactoring |
| `remove_inplace` | `cell, variable` | Refactoring |
| `insert_deepcopy` | `cell, variable` | Refactoring |
| `mark_diagnostic` | `cell` | Refactoring |
| `merge_cells` | `cell_ids: list[str]` | Refactoring |
| `move_cell` | `cell, after_cell` | Refactoring |
| `checkpoint` | — | Checkpoint |
| `restore` | `checkpoint_id` | Checkpoint |
| `list_checkpoints` | — | Checkpoint |
| `save_notebook` | `path?` | Lifecycle |
| `get_log` / `save_log` / `print_log` | — | Logging |

**MCP-only tools:** `load_notebook(path)`, `close_notebook()`, `get_notebook_path()`, `list_cells()`, `run_from(cell)`

All `cell` params accept **@A notation** (code-cell-only indexing) or 4-char cell IDs.

### MCP Path (`flowbook/mcp/`)

```
server.py  →  FlowBookTools  →  NotebookSession  →  Kernel (ZMQ)
                                                  →  JupyterLab (Contents API)
```

- **`server.py`**: 30 `@mcp.tool()` functions, each a thin wrapper delegating to `FlowBookTools`
- **`FlowBookTools`** (`flowbook/tools/tools.py`): Synchronous class with all tool logic; takes `NotebookSession`; handles @A ↔ cell_id resolution and result formatting
- **`NotebookSession`** (`mcp/session.py`): Manages kernel (ZMQ), in-memory notebook, Contents API sync, event log, checkpoints
- **Error handling**: `@_logged_tool` catches all exceptions, returns `"ERROR: ..."` text — never raises
- **Logging**: All tool calls logged to `mcp.log` (working directory) with args, duration, result preview

### NBI Path (`flowbook/nbi/`)

```
tools.py  →  run_ui_command()  →  nbibridge.ts  →  JupyterLab UI  →  Kernel
```

- **`tools.py`**: 27 `@nbapi.tool` async functions using `run_ui_command()` bridge to JupyterLab
- **Bridge commands** (`src/flowbook/nbibridge.ts`): ~20 JupyterLab commands (`flowbook:get-cell`, `flowbook:edit-cell-source`, `flowbook:run-cell`, etc.) that manipulate notebook widgets directly via `sharedModel.setSource()`, `NotebookActions.run()`, etc.
- **`FlowBookSession`** (`nbi/session.py`): Lightweight — only checkpoints and event log (no kernel, no notebook state)
- **Error handling**: `@_safe_tool` catches all exceptions, returns `"ERROR in {tool}: ..."` text
- **Why not NotebookSession?** NBI must use `run_ui_command()` for all notebook operations because the Contents API path doesn't reliably update JupyterLab's UI

### Shared Components (`flowbook/tools/`)

- **`format.py`**: Formatters for locations, errors, metadata, run results, status — used by both MCP (via FlowBookTools) and NBI (direct import)
- **`tools.py`**: `FlowBookTools` class — used by MCP only; NBI has its own implementations via the bridge
- **`nbi/cell_addressing.py`**: `index_to_alpha()`, `alpha_to_index()`, `parse_cell_ref()` — used by both

## Kernel Sharing (ZMQ + Discovery Files)

Both MCP and JupyterLab connect to the **same kernel process** as ZMQ clients. Whoever starts first writes a discovery file; the other reads it and joins.

**Discovery files:** `{jupyter_runtime_dir}/flowbook-{sha256(abs_path)[:12]}.json`

```json
{
  "notebook_path": "/abs/path/notebook.ipynb",
  "connection_file": "/abs/path/kernel-{uuid}.json",
  "kernel_name": "flowbook_kernel",
  "pid": 12345,
  "started_by": "mcp" | "jupyterlab",
  "started_at": 1234567890.5
}
```

**Validation:** On read, PID liveness and connection file existence are checked. Stale files are auto-cleaned.

**Key files:** `flowbook/kernel_discovery.py`, `flowbook/server/handlers.py` (KernelDiscoveryHandler), `src/flowbook/plugin.ts` (_writeKernelDiscovery)

## Comm Protocol (Kernel ↔ Clients)

The kernel communicates with all clients via a dual-send mechanism: custom IOPub messages (for Python clients like MCP) and comm channel (for the JupyterLab frontend).

**Kernel → Client:**

| Message Type | Purpose |
|---|---|
| `metadata` | Post-execution: read/write locs, stale cells, timing, staleness reasons |
| `violation` | Predicate violation: which predicate, locations, message, accepted flag |
| `status` | Status line: icon + text |
| `enforcer_checkpoint_result` | Response with checkpoint ID after enforcer snapshot |

**Client → Kernel:**

| Message Type | Purpose |
|---|---|
| `notebook_structure` | Set cell order (sent before each execution) |
| `cell_edited` | Mark cell stale (sent on source change, debounced 1s) |
| `continue_after_violation` | Toggle violation rejection vs. reporting |
| `sync` | Request full current state broadcast |
| `enforcer_checkpoint` | Snapshot enforcer's reproducibility state (deepcopy NotebookState) |
| `enforcer_restore` | Restore enforcer state from a snapshot, broadcast sync |

**Transport:** IOPub msg_type `flowbook_update`, comm target `flowbook`

**Key files:** `flowbook/kernel/protocol.py`, `src/flowbook/protocol.ts`, `flowbook/kernel/flowbook_kernel.py` (`_send_flowbook_message`, `_handle_flowbook_message`)

## Enforcer Checkpoint/Restore

Checkpoints preserve the full reproducibility state (not just cell sources). Both MCP and NBI trigger the same kernel mechanism:

1. **Client sends** `enforcer_checkpoint` → kernel deepcopies `NotebookState` + `seq_counter` + `continue_after_violation` → stores in `_enforcer_snapshots` → responds with `enforcer_checkpoint_result`
2. **Client sends** `enforcer_restore` with checkpoint ID → kernel replaces state → broadcasts `sync` to update all client UIs

**MCP checkpoint flow:**
- `session.checkpoint()` saves cell sources locally + sends `enforcer_checkpoint` to kernel
- Stores both the MCP checkpoint ID and kernel's `enforcer_snapshot_id`
- `session.restore()` restores sources + sends `enforcer_restore` + restores local metadata

**NBI checkpoint flow:**
- `checkpoint()` reads cell sources via bridge + calls `flowbook:enforcer-checkpoint` bridge command
- `restore()` edits only changed cells via bridge + calls `flowbook:enforcer-restore`
- Only edits cells whose source actually changed to avoid false staleness

## Contents API Sync (MCP ↔ JupyterLab)

MCP syncs notebook state with JupyterLab via the Jupyter Contents API. With `jupyter-collaboration`, the API returns/accepts the live Y.js document state.

**Reading (JupyterLab → MCP):** `GET /api/contents/{path}` → MCP merges sources into in-memory notebook (rate-limited to 0.2s)

**Writing (MCP → JupyterLab):** `PUT /api/contents/{path}` → updates Y.js document → JupyterLab sees edits

**Key files:** `flowbook/mcp/session.py` (`_setup_contents_api`, `_refresh_from_contents_api`, `_put_contents_api`), `flowbook/mcp/jupyter_config.py`

## NBI Extension Activation

`FlowBookNBIExtension.activate(host)` (`flowbook/nbi/extension.py`):

1. Disables NBI's built-in `nbi-notebook-edit` and `nbi-notebook-execute` toolsets (they destroy cell identity)
2. Creates `FlowBookSession` for checkpoints/logging
3. Registers FlowBook toolset with NBI host (tools appear as `mcp__nbi__*` to Claude)
4. Installs Claude slash commands from `flowbook/nbi/claude_commands/` to `{jupyter_root}/.claude/commands/`
5. Registers FlowBook MCP server in `~/.claude.json` (if not already present)

**Important:** Extensions must be initialized before `ClaudeCodeChatParticipant` is created (fixed in `notebook_intelligence/ai_service_manager.py`) so FlowBook tools appear in the `mcp__nbi__` namespace.

## MCP Server Startup

`main()` in `flowbook/mcp/server.py`:

1. Opens `mcp.log` in working directory (append mode)
2. Redirects FlowBook output module to log file (prevents STDIO corruption — MCP uses STDIO for JSON protocol)
3. Runs MCP server on STDIO transport

**Monitoring:** `tail -f mcp.log` shows kernel timer messages, tool call logs (args + duration + result preview), and errors.

## Error Handling

Both surfaces are **bullet-proof** — exceptions never propagate to the transport:

- **MCP** (`@_logged_tool`): Catches all exceptions, returns `"ERROR: {type}: {message}"` as text, logs to session event log + mcp.log
- **NBI** (`@_safe_tool`): Catches all exceptions, returns `"ERROR in {tool}: {type}: {message}"` as text, logs traceback via Python logging

## Graceful Degradation

| Scenario | Behavior |
|---|---|
| No Jupyter server running | MCP works standalone: own kernel, disk I/O, no live sync |
| `jupyter-collaboration` not installed | Contents API returns disk state, not live edits |
| JupyterLab not open | MCP starts own kernel, writes discovery for later |
| MCP not running | JupyterLab works normally, writes discovery for later |
| Comm channel fails | Metadata still arrives via IOPub fallback |
| Contents API fails | Best-effort: failures logged, never thrown |
| Checkpoint save fails | Cell execution continues without checkpoint protection |
| Tool throws exception | Error returned as text to Claude, never hangs |

## Slash Commands

Two sets of slash commands for the two contexts:

**NBI-installed** (`flowbook/nbi/claude_commands/` → copied to `{jupyter_root}/.claude/commands/`):
- `flowbook-fix.md` — Fix violations in active JupyterLab notebook (uses `mcp__nbi__*` tools)
- `basic-run-nbi.md` — Run active notebook and report status

**Repo-level** (`.claude/commands/`):
- `cli-flowbook-fix.md` — Fix violations via MCP CLI (uses `mcp__flowbook__*` tools, requires file path)
- `basic-run.md` — Run notebook via MCP CLI
- `categorize-repro-errors.md` — Batch error categorization
- `sync-spec.md` — Sync formal spec with code

## Key Files

| Component | Path |
|---|---|
| MCP server | `flowbook/mcp/server.py` |
| MCP session | `flowbook/mcp/session.py` |
| Jupyter config discovery | `flowbook/mcp/jupyter_config.py` |
| Kernel discovery | `flowbook/kernel_discovery.py` |
| FlowBookTools | `flowbook/tools/tools.py` |
| Shared formatters | `flowbook/tools/format.py` |
| Cell addressing | `flowbook/nbi/cell_addressing.py` |
| NBI tools | `flowbook/nbi/tools.py` |
| NBI extension | `flowbook/nbi/extension.py` |
| NBI session | `flowbook/nbi/session.py` |
| NBI bridge (TypeScript) | `src/flowbook/nbibridge.ts` |
| Protocol (Python) | `flowbook/kernel/protocol.py` |
| Protocol (TypeScript) | `src/flowbook/protocol.ts` |
| Kernel | `flowbook/kernel/flowbook_kernel.py` |
| Enforcer | `flowbook/kernel/reproducibility_enforcer.py` |
| Execution hook (frontend) | `src/flowbook/executionhook.ts` |
| Plugin (frontend) | `src/flowbook/plugin.ts` |
| AST refactoring utils | `flowbook/scripts/fix_repro_errors.py` |
