# MCP ↔ JupyterLab ↔ Kernel Collaboration Architecture

## Overview

FlowBook implements three-party bidirectional collaboration between:

1. **MCP Server** (`flowbook_mcp` CLI) — AI-driven reproducibility analysis
2. **JupyterLab Frontend** — notebook editor with FlowBook plugin
3. **Shared FlowBook Kernel** — instrumented IPython kernel with reproducibility enforcement

Any two of these can run independently, but all three collaborate when available.

## Synchronization Paths

```
┌──────────────────┐          ┌──────────────────┐
│   MCP Server     │          │   JupyterLab     │
│  (flowbook_mcp)  │          │   Frontend       │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
    Contents API                  Comm Channel
   GET (read edits)             (bidirectional)
   PUT (push edits)                    │
         │         ┌──────────┐        │
         │         │  Jupyter │        │
         ├────────►│  Server  │◄───────┤
         │         │ (REST)   │        │
         │         └──────────┘        │
         │                             │
    ZMQ (execute,     Shared       ZMQ (execute,
     poll IOPub)      Kernel        comm, IOPub)
         │         ┌──────────┐        │
         └────────►│ FlowBook │◄───────┘
                   │  Kernel  │
                   └──────────┘
```

### Path 1: Kernel Sharing (ZMQ + Discovery Files)

Both MCP and JupyterLab connect to the **same kernel process** as ZMQ clients.
Whoever starts the kernel writes a discovery file; the second participant reads it
and connects as a second client.

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

**Validation:** On read, PID liveness and connection file existence are checked.
Stale files are auto-cleaned.

**Key files:**
- `flowbook/kernel_discovery.py` — `read_discovery()`, `write_discovery()`, `remove_discovery()`
- `flowbook/server/handlers.py` — `KernelDiscoveryHandler` (GET/PUT for frontend)
- `src/flowbook/plugin.ts` — `_writeKernelDiscovery()` (writes on activation + kernel restart)

### Path 2: Contents API (Notebook State Sync)

MCP syncs notebook cell sources with JupyterLab via the Jupyter Contents API.
With `jupyter-collaboration`, the API returns/accepts the live Y.js document state
(not the disk file), enabling real-time edit propagation.

**Reading (JupyterLab → MCP):**
```
GET /api/contents/{path}?content=1&type=notebook
→ Returns live Y.js notebook with all cell sources reflecting JupyterLab edits
→ MCP merges sources into self.notebook["cells"] (preserves MCP-local outputs/metadata)
→ Rate-limited to max once per 0.5s
```

**Writing (MCP → JupyterLab):**
```
PUT /api/contents/{path}
Body: {"type": "notebook", "format": "json", "content": notebook}
→ Server updates Y.js document → JupyterLab sees MCP edits instantly
→ Called after: edit_cell, alpha_rename, remove_inplace, insert_deepcopy,
  mark_diagnostic, merge_cells, move_cell, save_notebook
```

**Key files:**
- `flowbook/mcp/session.py` — `_setup_contents_api()`, `_refresh_from_contents_api()`, `_put_contents_api()`
- `flowbook/mcp/jupyter_config.py` — `discover_jupyter_server()`, `discover_jupyter_server_root()`

### Path 3: Comm Protocol (Kernel ↔ Frontend)

The kernel and JupyterLab frontend communicate via a Jupyter comm channel
(`target_name="flowbook"`). This carries reproducibility metadata, violations,
and status updates.

**Kernel → Frontend:**

| Message Type | Purpose |
|---|---|
| `metadata` | Post-execution: read/write locs, stale cells, timing, staleness_reasons |
| `violation` | Predicate violation: which predicate, locations, message, accepted flag |
| `status` | Status line: icon + text (displayed in metadata panel header) |

**Frontend → Kernel:**

| Message Type | Purpose |
|---|---|
| `notebook_structure` | Set cell order (sent before each execution) |
| `cell_edited` | Mark cell stale (sent on source change, debounced 1s) |
| `continue_after_violation` | Toggle violation rejection vs. reporting |
| `sync` | Request full current state |

**Key files:**
- `flowbook/kernel/flowbook_kernel.py` — `_send_flowbook_message()` (dual: comm + IOPub)
- `flowbook/kernel/protocol.py` — Message builders and types (Python)
- `src/flowbook/protocol.ts` — Message types (TypeScript)
- `src/flowbook/executionhook.ts` — `_onCommMessage()`, `_onExecutionScheduled()`

### Path 4: IOPub Monitoring (External Execution Capture)

MCP polls the kernel's IOPub channel to catch executions initiated by JupyterLab.
The kernel sends `flowbook_update` messages on IOPub (in addition to comm) so both
clients see the same metadata regardless of who initiated the execution.

```
_poll_iopub()  →  drain pending flowbook_update messages
                  →  update cell_flowbook_meta, cell_status, _stale_cells
```

Called automatically at the start of: `get_cell`, `list_cells`, `get_status`,
`get_next_actionable`.

## MCP Server Architecture

### Tool Categories (23 tools)

| Category | Tools |
|---|---|
| **Lifecycle** | `load_notebook`, `close_notebook`, `get_notebook_path`, `continue_after_violation` |
| **Cell Access** | `list_cells`, `get_cell`, `get_next_actionable_cell` |
| **Execution** | `run_cell`, `run_all_cells`, `run_from`, `get_status` |
| **Editing** | `edit_cell` |
| **Save** | `save_notebook` |
| **Checkpoints** | `checkpoint`, `restore`, `list_checkpoints` |
| **Refactoring** | `alpha_rename`, `remove_inplace`, `insert_deepcopy`, `mark_diagnostic`, `merge_cells`, `move_cell` |
| **Logging** | `get_log`, `save_log`, `print_log` |

All tools are **synchronous** functions wrapped by `@_logged_tool` (captures name,
args, result, duration, errors into `session._event_log`).

### NotebookSession Lifecycle

```
load(path)
  ├─ Read notebook from disk
  ├─ Check kernel discovery file
  │   ├─ Found (PID alive, connection file exists) → join existing kernel
  │   │   └─ Skip cell ID normalization (preserve JupyterLab's IDs)
  │   └─ Not found → normalize IDs, start new kernel, write discovery file
  ├─ _setup_contents_api(abs_path)
  │   ├─ discover_jupyter_server() → (url, token)
  │   ├─ discover_jupyter_server_root() → root_dir
  │   ├─ Compute relative path: os.path.relpath(abs_path, root)
  │   └─ Test GET to verify API connectivity
  └─ Return: cell_ids, joined_existing, contents_api_connected

close()
  ├─ Auto-save event log
  ├─ If _owns_kernel: shutdown kernel + remove discovery file
  └─ Else: just disconnect ZMQ channels (leave kernel for JupyterLab)
```

### Save Behavior

When a Jupyter server is available, `save()` uses Contents API PUT instead of
writing to disk. This updates JupyterLab's live Y.js document and lets the
server handle persistence. Falls back to direct disk write when no server is
available or saving to a custom path.

## Frontend Plugin Architecture

### Activation

`flowbook:plugin` activates only when the notebook kernel is `flowbook_kernel`.
On activation:

1. Creates metadata panel, dependencies panel, cell highlighter
2. Creates execution hook manager (comm + edit listeners)
3. Sends `notebook_structure` + `sync` to kernel via comm
4. Writes kernel discovery file via `PUT /flowbook/kernel-discovery/{path}`

### External Execution Handling

When MCP runs a cell on the shared kernel, the frontend receives metadata and
violations via the comm channel. `_onCommMessage` handles these directly:

- **Metadata**: Stores on cell, calls `updateCell()` + `refreshDependencies()`
- **Violations**: Stores on cell, calls `updateCell()` to render violation notice

This is necessary because `_onCellExecuted` (which normally renders cell UI)
only fires for locally-initiated executions.

### Discovery File Rewrite

`_writeKernelDiscovery()` is called:
- On plugin activation (initial kernel detection)
- On `_onStatusChanged()` when kernel is active (catches kernel restarts)

The server handler (`KernelDiscoveryHandler.put`) resolves the notebook path
with `expanduser()` on `server_root_dir`, and looks up the actual kernel PID
from the Jupyter kernel manager.

## Server Extension (handlers.py)

### KernelDiscoveryHandler

**Path resolution** (`_resolve_notebook_path`):
- Expands `~` in both the path and `server_root_dir` setting
- Resolves relative paths against the server root
- Returns canonical absolute path

**PID lookup** (`_get_kernel_pid`):
- Extracts kernel UUID from connection filename
- Queries `serverapp.kernel_manager.get_kernel(uuid)`
- Returns `(pid, abs_connection_file)` — the frontend sends `pid=0` and the
  server fills in the real values

## Graceful Degradation

| Scenario | Behavior |
|---|---|
| No Jupyter server running | MCP works standalone: own kernel, disk I/O, no live sync |
| jupyter-collaboration not installed | Contents API returns disk state, not live edits |
| JupyterLab not open | MCP starts own kernel, writes discovery for later |
| MCP not running | JupyterLab works normally, writes discovery for later |
| Comm channel fails | Metadata still arrives via IOPub fallback |
| Contents API fails | Best-effort: failures logged, never thrown |

## Known Issues and Plan

### Issue 1: Cell Add/Delete Not Synced — FIXED

`_refresh_from_contents_api()` now detects structural changes (cells added,
deleted, or reordered in JupyterLab). It rebuilds the cell list from API order,
preserving MCP-local state (outputs, metadata), notifies the kernel of the new
cell order, and cleans up tracking for removed cells.

### Issue 2: Contents API PUT Race Condition — MITIGATED

`_put_contents_api()` now calls `_refresh_from_contents_api()` immediately
before sending the PUT, minimizing the window for concurrent edits. Full
optimistic locking (`If-Match` headers) is deferred as the race window is
typically milliseconds.

### Issue 3: CLAUDE.md Out of Date — FIXED

CLAUDE.md updated: Y.js references replaced with Contents API architecture,
`ydoc_sync.py` references removed, dependencies updated, collaboration section
rewritten with Contents API data flow.

### Issue 4: No Test Coverage for Contents API Integration — FIXED

14 tests added in `flowbook/mcp/tests/test_contents_api.py` covering source
merge, rate limiting, structural sync (add/delete/reorder), graceful failure,
PUT body format, save routing, and setup verification. Total: 45 tests.

**Original problem:** The 31 existing MCP tests cover standalone mode (own kernel, no
Jupyter server). There are no tests for Contents API refresh/push, kernel
discovery handler path resolution, or PID lookup.

**Plan:** Add integration tests:
- `test_contents_api.py`: Mock Contents API responses, verify source merge
  behavior, rate limiting, graceful failure
- `test_discovery_handler.py`: Test `_resolve_notebook_path` with tilde paths,
  relative paths, missing `server_root_dir`
- `test_put_contents_api.py`: Verify PUT body format, fallback to disk on failure

### Issue 5: MCP → JupyterLab Output Propagation Path — VERIFIED WORKING

**Original concern:** Removing `_sync_outputs_to_ydoc()` might break output
visibility in JupyterLab.

**Finding:** Outputs DO appear in JupyterLab via the shared kernel. Both MCP
and JupyterLab connect to the same ZMQ IOPub socket (multicast). JupyterLab's
kernel connection receives MCP-initiated outputs. FlowBook metadata/violations
propagate via the comm channel (inherently session-safe). No code change needed.

### Issue 6: Rate Limiting May Delay First Read — FIXED

Rate limit threshold reduced from 0.5s to 0.2s. The Contents API call to
localhost takes ~10ms, so 0.2s is still a safe rate limit while being responsive
enough that back-to-back tool calls both see fresh data.
