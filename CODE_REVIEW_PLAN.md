# FlowBook Code Review — Detailed Improvement Plan

## Context

Full code review of the FlowBook project identified 20 improvement areas across security, reliability, testing, dead code, code quality, and frontend. This plan organizes them into 7 phases ordered by severity and dependency. Each item includes exact file locations, what to change, and verification steps.

---

## Phase 1: Security & CI Fixes

### 1.1 Fix Path Traversal in `handlers.py`

**File:** `flowbook/server/handlers.py:223-237`

**Problem:** `_resolve_notebook_path()` resolves paths via `os.path.abspath()` but never validates the result stays within `server_root_dir`. Input like `../../etc/passwd` escapes the root.

**Fix:** After resolving, validate the path is under the root:
```python
resolved = os.path.abspath(path)
if root:
    abs_root = os.path.abspath(root)
    if not resolved.startswith(abs_root + os.sep) and resolved != abs_root:
        raise ValueError(f"Path escapes server root directory")
return resolved
```

**Verify:** Add test in `flowbook/server/commands/tests/test_path_traversal.py` with cases: normal path, `../` escape, tilde expansion, absolute path outside root.

---

### 1.2 Fix Broken `release.yml`

**File:** `.github/workflows/release.yml:32,38,46`

**Problem:** References non-existent `placeholder-package/` directory. Build/publish will fail.

**Fix:** Replace `cd placeholder-package` with building from project root:
- Line 32: `cd placeholder-package` → remove (already in root)
- Line 38: `packages-dir: placeholder-package/dist/` → `packages-dir: dist/`
- Line 46: `files: placeholder-package/dist/*` → `files: dist/*`
- Also add `python -m build` at root level and npm publish step for the frontend extension.

**Verify:** Dry-run `python -m build` locally to confirm it produces artifacts in `dist/`.

---

## Phase 2: Critical Reliability Fixes

### 2.1 Fix MCP `ctx: Context = None` Defaults

**File:** `flowbook/mcp/server.py` — lines 337, 401, 610, 845

**Problem:** Four tool functions (`run_cell`, `run_from`, `save_notebook`, `save_log`) default `ctx=None`. If called without context, `_get_session(ctx)` crashes with an unhelpful error.

**Fix:** Remove the `= None` default from all four signatures. FastMCP injects context automatically — the default is unnecessary and misleading.

**Verify:** Run `pytest flowbook/mcp/tests/` — existing tests should still pass since they provide context.

---

### 2.2 Add Contents API Conflict Warning

**File:** `flowbook/mcp/session.py:1004-1039` (`_put_contents_api`)

**Problem:** No conflict detection when MCP and JupyterLab edit the same cell concurrently. MCP's refresh-then-PUT can silently overwrite JupyterLab edits.

**Fix (pragmatic):** Rather than implementing full OT/CRDT (which requires a Y.js client), add:
1. Track `_last_known_sources: Dict[str, str]` per cell after each refresh
2. Before PUT, re-GET and compare sources against `_last_known_sources`
3. If a cell was modified by JupyterLab (source differs from last known AND differs from what MCP is about to write), log a warning and include it in the tool response
4. Add a docstring note about this limitation

**Verify:** Add test in `flowbook/mcp/tests/test_contents_api.py` simulating concurrent edit detection.

---

### 2.3 Fix `alpha_rename` / `move_cell` ValueError

**File:** `flowbook/mcp/session.py:1207` and similar locations

**Problem:** `code_order.index(cell_id)` raises raw `ValueError` if cell_id not found. User sees unhelpful stack trace.

**Fix:** Wrap in validation:
```python
if cell_id not in code_order:
    raise ValueError(f"Cell '{cell_id}' not found in notebook. Available cells: {code_order}")
```

Apply same pattern to all `code_order.index()` calls in session.py.

**Verify:** Add test calling `alpha_rename` with invalid cell_id, assert helpful error message.

---

### 2.4 Fix Silent `_poll_iopub` Exception Swallowing

**File:** `flowbook/mcp/session.py:613`

**Problem:** `except Exception: pass` swallows all errors including kernel crashes, malformed messages. Makes debugging impossible.

**Fix:** Replace with:
```python
except Empty:
    pass  # No more messages — normal
except Exception as e:
    import logging
    logging.getLogger(__name__).debug("IOPub poll error: %s", e)
```

**Verify:** Existing MCP tests should pass. Add debug logging test if feasible.

---

## Phase 3: Exception Handling Cleanup

### 3.1 Fix 40 Bare `except Exception` in `compare_baseline.py`

**File:** `flowbook/server/commands/compare_baseline.py` (3,062 lines)

**Problem:** 40+ `except Exception` blocks silently continue, masking real bugs. This is the single largest code quality issue.

**Fix (incremental — do NOT refactor the whole file):**
1. Replace bare `except Exception: continue` with `except Exception as e: logger.warning("...: %s", e); continue`
2. For exception blocks that return fallback values, add logging
3. Where possible, narrow to specific exceptions (e.g., `KeyError`, `TypeError`, `json.JSONDecodeError`)
4. Do NOT restructure the file — just add logging and narrow exceptions

**Verify:** Run `pytest flowbook/server/commands/tests/` — existing tests should pass.

---

### 3.2 Fix Star Imports and Bare Excepts in `locals.py`

**File:** `flowbook/kernel_support/locals.py:10-11, 51, 165`

**Fix:**
- Lines 10-11: Replace `from types import *` / `from typing import *` with explicit imports
- Line 51: Replace `except:` with `except (OSError, SyntaxError, TypeError):`
- Line 165: Replace `except:` with `except Exception:`

**Verify:** Run `pytest flowbook/kernel_support/tests/` — ensure no regressions.

---

## Phase 4: Dead Code Removal

### 4.1 Delete `src/_archived/` Directory

**Files:** 17 files, ~4,200 lines of dead TypeScript/TSX

**Confirmed:** No imports from active code reference `_archived/`. Already excluded from tsconfig.json and eslint.

**Fix:** `git rm -r src/_archived/`

**Verify:** `jlpm build` succeeds.

---

### 4.2 Delete `style/_archived.css`

**File:** `style/_archived.css` — 20KB of unused CSS

**Confirmed:** Not imported by `base.css` or any TS file.

**Fix:** `git rm style/_archived.css`

**Verify:** `jlpm build` succeeds.

---

### 4.3 Remove Unused `IOPUB_MSG_TYPE` Export

**File:** `src/flowbook/protocol.ts:19`

**Problem:** `export const IOPUB_MSG_TYPE = 'flowbook_update'` is never imported anywhere.

**Fix:** Delete line 19.

**Verify:** `jlpm build` succeeds.

---

### 4.4 Move Test Dependencies to Optional Group

**File:** `pyproject.toml:43-45`

**Problem:** `pytest`, `pytest-asyncio`, `hypothesis` are in main `dependencies` — installed for all users.

**Fix:** Move to optional dependencies:
```toml
[project.optional-dependencies]
test = [
    "pytest",
    "pytest-asyncio",
    "hypothesis",
]
```

**Verify:** `pip install -e ".[test]"` then `pytest flowbook/` passes.

---

## Phase 5: Test Coverage (Highest-Impact Gaps)

### 5.1 Add Tests for `flowbook/server/handlers.py`

**File to create:** `flowbook/server/tests/test_handlers.py` (+ `__init__.py`)

**What to test (264 LOC, 5 handler classes):**
- `FlowbookCommandHandler.post()`: missing command field (400), missing notebook (400), successful execution, kernel connection failure, command exception (500)
- `CommandListHandler.get()`: returns command list
- `KernelDiscoveryHandler._resolve_notebook_path()`: normal path, tilde expansion, relative resolution, path traversal rejection (from Phase 1)
- `KernelDiscoveryHandler.get()`: found vs not found
- `KernelDiscoveryHandler.put()`: successful write

**Pattern:** Use `tornado.testing` or mock the handler's `self.request`, `self.finish()`, `self.set_status()`. Follow patterns from existing server command tests.

---

### 5.2 Add Tests for `flowbook/server/kernel_manager.py`

**File to create:** `flowbook/server/tests/test_kernel_manager.py`

**What to test (87 LOC, 2 classes):**
- `FlowbookKernelClient.execute()`: message construction, cell_id injection, metadata propagation
- `KernelConnectionManager.get_kernel_client()`: caching behavior, channel startup
- `KernelConnectionManager.cleanup_client()`: removes from cache, stops channels

**Approach:** Mock `BlockingKernelClient` and `ServerApp.kernel_manager` since we can't start real kernels in unit tests.

---

### 5.3 Add Tests for `flowbook/server/message_broadcaster.py`

**File to create:** `flowbook/server/tests/test_message_broadcaster.py`

**What to test (241 LOC, 3 classes):**
- `MessageBroadcaster`: register/unregister clients, send to specific client, broadcast to all, QueueFull handling, singleton pattern
- `Message.to_json()`: all message types serialize correctly
- `BroadcastStream.write()`: splits newlines, sends APPEND/NEWLINE messages, handles empty text

**Approach:** Direct unit tests — `MessageBroadcaster` has no external dependencies besides `asyncio.Queue`.

---

### 5.4 Add Tests for `kernel_command_handlers.py`

**File to create:** `flowbook/kernel_support/tests/test_kernel_command_handlers.py`

**What to test (512 LOC, 12 handler methods):**
- Each `handle_*` method: success path, exception path (returns error response)
- `get_handler()`: valid command, invalid command raises ValueError
- Focus on error response formatting — ensure tracebacks are captured

**Approach:** Mock the `kernel` object (needs `shell`, `_checkpoint`, `_use_scalene`, etc.). The handlers are pure functions given a request object — straightforward to test.

---

### 5.5 Add Tests for `kernel_command_client.py`

**File to create:** `flowbook/kernel_support/tests/test_kernel_command_client.py`

**What to test (646 LOC, 10+ methods):**
- `_send_command()`: timeout handling, retry behavior, progress callbacks, comm_open/comm_close lifecycle
- Each checkpoint method: success path, error response path (status="error"), retry logic with sleep
- `KernelCommandError` raised after retries exhausted

**Approach:** Mock `BlockingKernelClient` to simulate comm messages. Test timeout by providing no response.

---

### 5.6 Investigate and Fix Skipped Tests

**5 hard-skipped tests to investigate:**

| File | Line | Reason | Action |
|------|------|--------|--------|
| `test_reproducibility_structural.py` | 301 | "staleness computation interaction with structural tracking" | Investigate if the underlying issue is fixed; unskip or document as known limitation |
| `test_reproducibility_structural.py` | 356 | Same | Same |
| `test_reproducibility_structural.py` | 503 | "expects no backward mutation violation" | Determine if this is a test issue or a real enforcer limitation |
| `test_monotonicity_structural.py` | 523 | "error message should mention structural nature" | This is an enhancement — file as issue or implement |
| `test_execution_error_capture.py` | 499 | "Requires running Jupyter kernel" | Leave skipped — this is a manual integration test |

---

## Phase 6: Frontend Improvements

### 6.1 Add Error Handling to `nbibridge.ts` Command Loop

**File:** `src/flowbook/nbibridge.ts:603-654`

**Problem:** The `flowbook:run-actionable-cells` command loop has no try-catch. Single command failure crashes the entire sequence with no user feedback.

**Fix:** Wrap the loop body in try-catch:
```typescript
while (totalRun < maxIterations) {
  try {
    const actionable = (await app.commands.execute('flowbook:get-next-actionable')) as any;
    if (actionable.done) break;
    // ... run cell ...
  } catch (error) {
    console.error('Error in run-actionable-cells:', error);
    break;
  }
}
```

**Verify:** `jlpm build` succeeds. Manual test: run actionable cells with a cell that errors.

---

### 6.2 Migrate to React 18 `createRoot`

**Files:**
- `src/flowbook/metadatapanel.tsx:518` — `ReactDOM.render(...)`
- `src/flowbook/dependenciespanel.tsx:547` — `ReactDOM.render(...)`

**Fix:** In each file:
1. Import `createRoot` from `react-dom/client`
2. Store root as instance variable: `private _root: Root | null = null`
3. In `render()`: create root on first call, then `root.render(<Component />)`
4. In `dispose()`: call `this._root?.unmount()` before `super.dispose()`

**Verify:** `jlpm build` succeeds. Manual test: open metadata panel, verify rendering.

---

### 6.3 Reduce `as any` Assertions (Lower Priority)

**Files:** 20+ occurrences across `executionhook.ts`, `stalenessnotice.ts`, `violationnotice.ts`, `nbibridge.ts`, `toolbar.ts`, `plugin.ts`

**Approach:** Create type guards and interfaces in `types.ts` for common patterns:
- `IOutputWithMetadata` for cell outputs that may have flowbook metadata
- Type guard `hasFlowbookMetadata(obj)` to replace `(output as any).metadata?.flowbook`
- Proper typing for command execution results in nbibridge

**This is a gradual improvement — address the most-used patterns first.**

---

## Phase 7: Polish

### 7.1 Remove Unused `import pprint`

**File:** `flowbook/server/handlers.py:6`

**Note:** Actually used at line 312 in `setup_handlers()`. Verify before removing — if it's only used in debug logging, consider wrapping in `if DEBUG:` instead.

---

### 7.2 Unify MCP Refactoring Tool Patterns

**File:** `flowbook/mcp/session.py` — `alpha_rename`, `remove_inplace`, `insert_deepcopy`

**Problem:** Each follows a slightly different mark-stale / notify-kernel / push-to-API pattern.

**Fix:** Extract common post-edit helper:
```python
def _after_cell_edit(self, cell_id: str) -> None:
    """Mark cell stale, notify kernel, and sync to Contents API."""
    self._stale_cells.add(cell_id)
    self._notify_kernel_cell_edited(cell_id)
    self._put_contents_api()
```

---

### 7.3 Fix `executor.shutdown(wait=False)` in Handlers

**File:** `flowbook/server/handlers.py:110`

**Problem:** `wait=False` can interrupt tasks mid-execution.

**Fix:** Change to `executor.shutdown(wait=True)` — the executor has `max_workers=1` and the task has already completed (result is captured on line 109), so `wait=True` is safe and nearly instant.

---

### 7.4 Update README to Mention MCP Server

**File:** `README.md`

**Add:** Brief section about MCP server support for AI-powered notebook analysis via Claude Code.

---

## Verification Plan

After all changes:

1. **Python tests:** `pytest flowbook/ -x -q` — all existing + new tests pass
2. **TypeScript build:** `jlpm build` — no errors
3. **Lint:** `jlpm lint:check` — no new warnings
4. **Extension check:** `jupyter server extension list` and `jupyter labextension list` — flowbook enabled
5. **Manual smoke test:** Open a notebook in JupyterLab with flowbook kernel, execute cells, verify highlighting/staleness works
6. **MCP test:** Load notebook via MCP tools, run cells, verify metadata returned correctly

---

## Summary by Phase

| Phase | Items | Effort | Risk |
|-------|-------|--------|------|
| 1. Security & CI | 2 items | Small | Low (isolated fixes) |
| 2. Critical Reliability | 4 items | Medium | Low-Medium (MCP session changes) |
| 3. Exception Handling | 2 items | Medium | Low (additive logging) |
| 4. Dead Code | 4 items | Small | Very Low (deletions) |
| 5. Test Coverage | 6 items | Large | Very Low (new files only) |
| 6. Frontend | 3 items | Medium | Low (isolated TS changes) |
| 7. Polish | 4 items | Small | Very Low |
