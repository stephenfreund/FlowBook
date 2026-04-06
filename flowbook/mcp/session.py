"""
NotebookSession — manages a single (notebook, kernel) pair for the MCP server.

This is the stateful core: it holds the in-memory notebook, a running FlowBook
kernel, and accumulated reproducibility metadata from cell executions.
"""

import ast
import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from queue import Empty
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from flowbook.cli.helpers import (
    cleanup_kernel,
    save_notebook as cli_save_notebook,
    setup_kernel,
)
from flowbook.kernel_discovery import read_discovery, write_discovery, remove_discovery
import urllib.request

from flowbook.mcp.jupyter_config import discover_jupyter_server, discover_jupyter_server_root
from flowbook.util.cell_ids import normalize_notebook_alpha, next_insertion_id
from flowbook.server.kernel_helper import KernelHelper
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.scripts.fix_repro_errors import (
    rename_variable_in_code,
    find_actual_variable_name,
    InplaceRemover,
    prepend_to_cell_source,
    split_cell_magic,
    get_cell_source,
    set_cell_source,
)


# ---------------------------------------------------------------------------
# Metadata formatting — canonical definitions in flowbook.tools.format
# Re-exported here for backward compatibility.
# ---------------------------------------------------------------------------

from flowbook.tools.format import (  # noqa: F401
    format_loc,
    format_loc_list,
    format_error,
    format_staleness_reasons,
    format_flowbook_meta,
    format_outputs_text,
)


def _truncate_dict(d: Dict[str, Any], max_str_len: int = 500) -> Dict[str, Any]:
    """Truncate string values in a dict for log readability."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_str_len:
            out[k] = v[:max_str_len] + "..."
        elif isinstance(v, dict):
            out[k] = _truncate_dict(v, max_str_len)
        elif isinstance(v, list) and len(v) > 20:
            out[k] = v[:20] + [f"... ({len(v)} total)"]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# NotebookSession
# ---------------------------------------------------------------------------

class NotebookSession:
    """Manages a single notebook + kernel pair."""

    def __init__(self):
        self.notebook: Optional[Dict[str, Any]] = None
        self.notebook_path: Optional[str] = None
        self.kernel_manager = None
        self.kernel_client: Optional[FlowbookKernelClient] = None
        self._owns_kernel: bool = False  # True if we started the kernel
        self._jupyter_server_url: Optional[str] = None
        self._jupyter_token: Optional[str] = None
        self._jupyter_contents_path: Optional[str] = None
        self._last_contents_refresh: float = 0
        self.executed_cells: Set[str] = set()
        self.cell_flowbook_meta: Dict[str, Dict] = {}
        self.cell_status: Dict[str, str] = {}  # cell_id -> "ok" | "error"
        self._stale_cells: Set[str] = set()
        self._continue_after_violation: bool = False
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._event_log: List[Dict[str, Any]] = []
        self._session_start: float = time.time()
        self._last_known_api_sources: Dict[str, str] = {}  # cell_id -> source from last API refresh
        self._conflict_warnings: List[str] = []  # populated by _put_contents_api

    @property
    def is_loaded(self) -> bool:
        return self.notebook is not None

    def _require_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("No notebook loaded. Call load_notebook first.")

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool invocation in the event log."""
        entry = {
            "seq": len(self._event_log),
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self._session_start, 3),
            "tool": tool,
            "args": args,
            "duration_ms": round(duration_ms, 1),
        }
        if error:
            entry["error"] = error
        else:
            # Truncate large result strings to keep the log manageable
            if isinstance(result, str):
                entry["result"] = result[:2000] + ("..." if len(result) > 2000 else "")
            elif isinstance(result, dict):
                entry["result"] = _truncate_dict(result)
            else:
                entry["result"] = str(result)[:2000]
        self._event_log.append(entry)

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Return the full event log."""
        return list(self._event_log)

    def save_event_log(self, path: Optional[str] = None) -> str:
        """Write the event log to a JSON file.

        Default path: {notebook_stem}-mcp-log.json alongside the notebook.
        """
        if path is None:
            if self.notebook_path:
                stem = self.notebook_path.rsplit(".", 1)[0]
                path = f"{stem}-mcp-log.json"
            else:
                path = "flowbook-mcp-log.json"

        log_doc = {
            "session_start": self._session_start,
            "notebook_path": self.notebook_path,
            "total_events": len(self._event_log),
            "events": self._event_log,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_doc, f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, path: str) -> Dict[str, Any]:
        """Load notebook from disk, start or join kernel, assign alpha cell IDs.

        Kernel discovery: checks the Jupyter runtime directory for an existing
        kernel (started by JupyterLab or another MCP session). If found and
        alive, connects as a second client. Otherwise starts a fresh kernel
        and writes a discovery file for others to find.

        Cell ID normalization is skipped when joining an existing session
        to avoid clobbering IDs that JupyterLab is already using.
        """
        # Close any previous session
        if self.is_loaded:
            self.close()

        abs_path = os.path.abspath(os.path.expanduser(path))

        with open(path, "r", encoding="utf-8") as f:
            raw_notebook = json.load(f)

        # Check for existing kernel via discovery file
        discovery = read_discovery(abs_path)

        self.notebook_path = abs_path
        self.executed_cells = set()
        self.cell_flowbook_meta = {}
        self.cell_status = {}
        self._stale_cells = set()
        self._checkpoints = {}
        self._event_log = []
        self._session_start = time.time()

        if discovery:
            # Join existing kernel — skip ID normalization to preserve
            # the cell IDs that the other client is already using
            self.notebook = raw_notebook
            self._owns_kernel = False
            self.kernel_manager, self.kernel_client = setup_kernel(
                connection_file=discovery["connection_file"],
                kernel_name="flowbook_kernel",
            )
        else:
            # Start fresh — normalize IDs and start our own kernel
            self.notebook = normalize_notebook_alpha(raw_notebook)
            notebook_dir = os.path.dirname(abs_path)
            self.kernel_manager, self.kernel_client = setup_kernel(
                connection_file=None,
                kernel_name="flowbook_kernel",
                cwd=notebook_dir,
            )
            self._owns_kernel = True

            # Write discovery file so JupyterLab (or another MCP) can find us
            if self.kernel_manager is not None:
                kernel_pid = getattr(
                    self.kernel_manager.provisioner, "pid", None
                ) or 0
                write_discovery(
                    notebook_path=abs_path,
                    connection_file=self.kernel_manager.connection_file,
                    kernel_name="flowbook_kernel",
                    pid=kernel_pid,
                    started_by="mcp",
                )

        cells = self.notebook.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        joined = " (joined existing kernel)" if discovery else ""

        # Set up Contents API for live sync with JupyterLab
        contents_status = self._setup_contents_api(abs_path)

        return {
            "path": path,
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "cell_ids": [c["id"] for c in code_cells],
            "joined_existing": discovery is not None,
            "contents_api_connected": self._jupyter_contents_path is not None,
            "info": f"Loaded{joined}{contents_status}",
        }

    def _setup_contents_api(self, notebook_abs_path: str) -> str:
        """Configure Contents API for reading live notebook state from JupyterLab.

        With jupyter-collaboration, the Contents API returns the live Y.js
        document state, reflecting JupyterLab edits instantly.

        Returns a status string for the load result message.
        """
        server_url, token = discover_jupyter_server()
        if not server_url:
            return ""

        try:
            server_root = discover_jupyter_server_root()
            if server_root:
                contents_path = os.path.relpath(notebook_abs_path, server_root)
            else:
                contents_path = os.path.basename(notebook_abs_path)

            # Verify the Contents API works with a test request
            url = f"{server_url}/api/contents/{contents_path}?content=0"
            headers = {}
            if token:
                headers["Authorization"] = f"token {token}"
            req = urllib.request.Request(url, headers=headers)
            urllib.request.urlopen(req, timeout=3)

            self._jupyter_server_url = server_url
            self._jupyter_token = token
            self._jupyter_contents_path = contents_path
            return " [live sync]"
        except Exception as e:
            logger.debug(f"Contents API not available: {e}")
            return ""

    def close(self):
        """Shutdown kernel (if we own it), auto-save log, and clear state.

        If we started the kernel, shuts it down and removes the discovery file.
        If we joined an existing kernel, just disconnects (leaves kernel running).
        """
        # Auto-save event log if there were any events
        if self._event_log and self.notebook_path:
            try:
                self.save_event_log()
            except Exception:
                pass  # best-effort log save on close

        if self._owns_kernel:
            # We started the kernel — shut it down and remove discovery file
            if self.kernel_client or self.kernel_manager:
                cleanup_kernel(self.kernel_client, self.kernel_manager)
            if self.notebook_path:
                remove_discovery(self.notebook_path)
        else:
            # We joined an existing kernel — just disconnect, don't kill it
            if self.kernel_client:
                try:
                    self.kernel_client.stop_channels()
                except Exception:
                    pass

        self.kernel_client = None
        self.kernel_manager = None
        self._owns_kernel = False
        self.notebook = None
        self.notebook_path = None
        self.executed_cells = set()
        self.cell_flowbook_meta = {}
        self.cell_status = {}
        self._stale_cells = set()
        self._checkpoints = {}

    # ------------------------------------------------------------------
    # Contents API sync (read live JupyterLab edits)
    # ------------------------------------------------------------------

    def _fetch_contents_api(self) -> Optional[Dict[str, Any]]:
        """Fetch the live notebook from Jupyter Contents API.

        With jupyter-collaboration, this returns the live Y.js document
        state, reflecting JupyterLab edits instantly (not the disk file).

        Returns:
            Notebook dict (nbformat structure), or None on failure.
        """
        if not self._jupyter_server_url or not self._jupyter_contents_path:
            return None
        try:
            url = (
                f"{self._jupyter_server_url}/api/contents/"
                f"{self._jupyter_contents_path}?content=1&type=notebook"
            )
            headers = {}
            if self._jupyter_token:
                headers["Authorization"] = f"token {self._jupyter_token}"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            return data.get("content")
        except Exception as e:
            logger.debug(f"Contents API fetch failed: {e}")
            return None

    def _refresh_from_contents_api(self) -> None:
        """Refresh in-memory notebook from the Jupyter Contents API.

        Merges cell sources from the API response into self.notebook["cells"],
        preserving MCP-local state (outputs, execution_count, flowbook metadata).
        Also handles structural changes (cells added/deleted in JupyterLab).
        Rate-limited to avoid excessive API calls.
        """
        if not self._jupyter_contents_path:
            return
        now = time.time()
        if now - self._last_contents_refresh < 0.2:
            return
        self._last_contents_refresh = now

        api_notebook = self._fetch_contents_api()
        if not api_notebook:
            return

        api_cells_list = api_notebook.get("cells", [])
        api_cells_by_id = {
            c.get("id"): c for c in api_cells_list if c.get("id")
        }

        # Snapshot local cells by ID for lookup during rebuild
        local_cells_by_id = {
            c.get("id"): c for c in self.notebook.get("cells", []) if c.get("id")
        }
        local_ids = set(local_cells_by_id.keys())
        api_ids = set(api_cells_by_id.keys())

        # Update sources for existing cells
        for cell_id in local_ids & api_ids:
            local_cell = local_cells_by_id[cell_id]
            api_source = api_cells_by_id[cell_id].get("source", "")
            if api_source != get_cell_source(local_cell):
                set_cell_source(local_cell, api_source)

        # Snapshot API sources for conflict detection on next PUT
        self._last_known_api_sources = {
            c.get("id"): c.get("source", "")
            for c in api_cells_list if c.get("id")
        }

        # Handle structural changes (cells added, deleted, or reordered)
        added_ids = api_ids - local_ids
        removed_ids = local_ids - api_ids
        api_order = [c.get("id") for c in api_cells_list if c.get("id")]
        local_order = [c.get("id") for c in self.notebook.get("cells", []) if c.get("id")]

        if not added_ids and not removed_ids and api_order == local_order:
            return

        # Rebuild cell list in API order, preserving MCP-local state
        new_cells = []
        for api_cell in api_cells_list:
            cid = api_cell.get("id")
            if cid in local_cells_by_id:
                new_cells.append(local_cells_by_id[cid])
            else:
                new_cells.append(api_cell)

        self.notebook["cells"] = new_cells

        # Notify kernel of new cell order
        if self.kernel_client:
            new_order = [
                c["id"] for c in new_cells if c.get("cell_type") == "code"
            ]
            KernelHelper.execute_code(
                self.kernel_client, "", timeout=10, store_history=False,
                flowbook_msg={
                    "type": "notebook_structure", "cell_order": new_order
                },
            )

        # Clean up tracking for removed cells
        for rid in removed_ids:
            self.executed_cells.discard(rid)
            self.cell_flowbook_meta.pop(rid, None)
            self.cell_status.pop(rid, None)
            self._stale_cells.discard(rid)

    def refresh_from_jupyter(self) -> None:
        """Public API to refresh notebook from JupyterLab (via Contents API)."""
        self._refresh_from_contents_api()

    def set_continue_after_violation(self, enabled: bool) -> None:
        """Configure whether violations reject execution or just report.

        When enabled (True): violations are reported but execution continues,
        cell stays CLEAN. Good for /basic-run where you want a full picture.

        When disabled (False, default): violations cause rollback and the cell
        is rejected. Good for /fix-notebook where you want a clean namespace.
        """
        self._require_loaded()
        self._continue_after_violation = enabled
        KernelHelper.execute_code(
            self.kernel_client,
            "",
            timeout=10,
            store_history=False,
            flowbook_msg={"type": "continue_after_violation", "enabled": enabled},
        )

    # ------------------------------------------------------------------
    # IOPub polling (catch JupyterLab-initiated executions)
    # ------------------------------------------------------------------

    def _poll_iopub(self) -> None:
        """Drain pending IOPub messages to catch external executions.

        When JupyterLab runs a cell on the shared kernel, the flowbook_update
        messages appear on IOPub. This method processes them to keep MCP's
        staleness and metadata state current.

        Called automatically at the start of get_cell, list_cells, get_status, etc.
        """
        if not self.kernel_client:
            return
        try:
            while True:
                msg = self.kernel_client.get_iopub_msg(timeout=0)
                msg_type = msg.get("msg_type", "")
                if msg_type == "flowbook_update":
                    content = msg.get("content", {})
                    data = content.get("data", content)
                    if isinstance(data, dict) and data.get("type") == "metadata":
                        cell_id = data.get("cell_id")
                        if cell_id:
                            self.cell_flowbook_meta[cell_id] = data
                            self.executed_cells.add(cell_id)
                            stale = set(data.get("stale_cells", []))
                            self._stale_cells = (self._stale_cells | stale) - {cell_id}
                            # Update cell outputs if we have them
                            # (outputs come via separate IOPub messages, not flowbook_update)
        except Empty:
            pass  # No more messages — normal
        except Exception as e:
            print(f"IOPub poll error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def get_cell_order(self) -> List[str]:
        """Return ordered list of code cell IDs."""
        self._require_loaded()
        return [
            c["id"]
            for c in self.notebook["cells"]
            if c.get("cell_type") == "code"
        ]

    def _all_cell_ids(self) -> Set[str]:
        """Return set of all cell IDs in the notebook."""
        return {c.get("id", "") for c in self.notebook.get("cells", [])}

    def _next_insert_id(self, after_id: str) -> str:
        """Generate an insertion ID after the given cell (e.g., B → B1, B2, ...)."""
        return next_insertion_id(after_id, self._all_cell_ids())

    def _find_cell(self, cell_id: str) -> Tuple[int, Dict[str, Any]]:
        """Find cell by ID, return (index_in_cells_list, cell_dict)."""
        self._require_loaded()
        for i, cell in enumerate(self.notebook["cells"]):
            if cell.get("id") == cell_id:
                return i, cell
        raise ValueError(f"Cell not found: {cell_id}")

    def _find_code_cell_index(self, cell_id: str) -> int:
        """Return the index of cell_id among code cells only."""
        for idx, cid in enumerate(self.get_cell_order()):
            if cid == cell_id:
                return idx
        raise ValueError(f"Code cell not found: {cell_id}")

    def get_cell(self, cell_id: str) -> Dict[str, Any]:
        """Get a cell's source, outputs, and flowbook metadata."""
        self._refresh_from_contents_api()
        self._poll_iopub()
        _, cell = self._find_cell(cell_id)
        source = get_cell_source(cell)
        outputs = cell.get("outputs", [])

        result = {
            "cell_id": cell_id,
            "cell_type": cell.get("cell_type", "code"),
            "source": source,
            "outputs_text": format_outputs_text(outputs),
            "execution_count": cell.get("execution_count"),
        }

        fb_meta = self.cell_flowbook_meta.get(cell_id)
        if fb_meta:
            result["flowbook"] = format_flowbook_meta(fb_meta)
            result["flowbook_raw"] = fb_meta

        if cell_id in self._stale_cells:
            result["status"] = "stale"
        elif cell_id in self.executed_cells:
            result["status"] = self.cell_status.get(cell_id, "ok")
        else:
            result["status"] = "unexecuted"

        return result

    def get_next_actionable(self) -> Optional[Dict[str, Any]]:
        """Return the first cell that needs attention.

        Priority: error > violation > stale > unexecuted.
        """
        self._require_loaded()
        self._refresh_from_contents_api()
        self._poll_iopub()
        code_cell_ids = self.get_cell_order()

        # 1. Runtime errors
        for cid in code_cell_ids:
            if self.cell_status.get(cid) == "error":
                cell_info = self.get_cell(cid)
                cell_info["reason"] = "runtime_error"
                return cell_info

        # 2. Reproducibility violations
        for cid in code_cell_ids:
            meta = self.cell_flowbook_meta.get(cid, {})
            errors = meta.get("errors", [])
            if errors:
                cell_info = self.get_cell(cid)
                cell_info["reason"] = "violation"
                cell_info["violation_summary"] = "; ".join(
                    format_error(e) for e in errors
                )
                return cell_info

        # 3. Stale cells
        for cid in code_cell_ids:
            if cid in self._stale_cells:
                cell_info = self.get_cell(cid)
                cell_info["reason"] = "stale"
                return cell_info

        # 4. Unexecuted cells
        for cid in code_cell_ids:
            if cid not in self.executed_cells:
                source = get_cell_source(self._find_cell(cid)[1])
                if source.strip():  # skip empty cells
                    cell_info = self.get_cell(cid)
                    cell_info["reason"] = "unexecuted"
                    return cell_info

        return None  # all clean

    def get_next_actionable_cell_id(self) -> Optional[str]:
        """Return the cell_id of the next actionable cell, or None if all clean."""
        result = self.get_next_actionable()
        if result is None:
            return None
        return result["cell_id"]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _extract_flowbook_meta(
        self, flowbook_messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract flowbook metadata from protocol messages.

        Args:
            flowbook_messages: Protocol messages from KernelHelper.execute_code()
        """
        for msg in flowbook_messages:
            if msg.get("type") == "metadata":
                return msg
        return None

    def run_cell(self, cell_id: str, timeout: float = 300) -> Dict[str, Any]:
        """Execute a single cell and return outputs + flowbook metadata."""
        self._require_loaded()
        self._refresh_from_contents_api()
        _, cell = self._find_cell(cell_id)
        if cell.get("cell_type") != "code":
            return {"cell_id": cell_id, "error": "Not a code cell"}

        source = get_cell_source(cell)
        if not source.strip():
            return {"cell_id": cell_id, "status": "skipped", "reason": "empty cell"}

        cell_order = self.get_cell_order()
        cell_metadata = {
            "cell_id": cell_id,
            "cell_order": cell_order,
        }

        result = KernelHelper.execute_code(
            self.kernel_client,
            source,
            timeout,
            cell_id=cell_id,
            cell_metadata=cell_metadata,
        )

        # Update cell in notebook
        cell["outputs"] = result["outputs"]
        cell["execution_count"] = result["execution_count"]
        self._put_contents_api()  # Push outputs to JupyterLab via Y.js
        self.executed_cells.add(cell_id)

        # Extract flowbook metadata
        fb_meta = self._extract_flowbook_meta(
            result.get("flowbook_messages", [])
        )
        if fb_meta:
            self.cell_flowbook_meta[cell_id] = fb_meta
            # Update stale cell tracking
            stale = set(fb_meta.get("stale_cells", []))
            self._stale_cells = (self._stale_cells | stale) - {cell_id}
            if cell_id in self._stale_cells:
                self._stale_cells.discard(cell_id)

        # Track execution status
        if result["status"] == "error":
            self.cell_status[cell_id] = "error"
        else:
            self.cell_status[cell_id] = "ok"

        # Build response
        response = {
            "cell_id": cell_id,
            "status": result["status"],
            "outputs_text": format_outputs_text(result["outputs"]),
        }
        if result.get("error_message"):
            response["error_message"] = result["error_message"]
        if fb_meta:
            response["flowbook"] = format_flowbook_meta(fb_meta)

        return response

    def run_all(self, timeout: float = 300) -> Dict[str, Any]:
        """Execute all code cells top-to-bottom."""
        self._require_loaded()
        self._refresh_from_contents_api()
        cell_order = self.get_cell_order()

        results = []
        all_errors = []
        all_stale: Set[str] = set()
        total_executed = 0
        overall_status = "success"

        for cell_id in cell_order:
            _, cell = self._find_cell(cell_id)
            source = get_cell_source(cell)
            if not source.strip():
                continue

            cell_result = self.run_cell(cell_id, timeout=timeout)
            results.append(cell_result)
            total_executed += 1

            # Collect violations
            fb_meta = self.cell_flowbook_meta.get(cell_id, {})
            errors = fb_meta.get("errors", [])
            if errors:
                for e in errors:
                    all_errors.append({"cell_id": cell_id, **e})

            # Update stale: add newly stale, remove the cell we just executed
            stale = fb_meta.get("stale_cells", [])
            all_stale.update(stale)
            all_stale.discard(cell_id)  # this cell just ran successfully

            # Stop on runtime error
            if cell_result.get("status") == "error":
                overall_status = "error"
                break

        # Build summary — use _stale_cells which is the authoritative
        # post-execution stale set (already maintained by run_cell)
        final_stale = self._stale_cells & set(cell_order)
        summary_lines = [
            f"Executed {total_executed}/{len(cell_order)} code cells",
            f"Status: {overall_status}",
            f"Violations: {len(all_errors)}",
            f"Stale cells: {len(final_stale)} ({', '.join(sorted(final_stale)) if final_stale else 'none'})",
        ]
        if all_errors:
            summary_lines.append("Violation details:")
            for e in all_errors:
                summary_lines.append(f"  - Cell {e['cell_id']}: {format_error(e)}")

        return {
            "status": overall_status,
            "summary": "\n".join(summary_lines),
            "total_executed": total_executed,
            "total_code_cells": len(cell_order),
            "violations": all_errors,
            "stale_cells": sorted(final_stale),
            "cell_results": results,
        }

    def run_from(self, cell_id: str, timeout: float = 300) -> Dict[str, Any]:
        """Run cell_id and subsequent cells that need execution, stopping on error.

        Refreshes from JupyterLab first to pick up any edits. Skips cells
        that are already clean. Only runs cells that are
        unexecuted, stale, or have violations.

        Args:
            cell_id: Cell to start from.
            timeout: Per-cell timeout in seconds.

        Returns:
            Summary with list of executed cells, skipped count, error cell
            (if any), and violation/stale counts.
        """
        self._require_loaded()
        self._refresh_from_contents_api()
        cell_order = self.get_cell_order()
        try:
            start = cell_order.index(cell_id)
        except ValueError:
            raise ValueError(f"Cell not found in code cell order: {cell_id}")

        executed = []
        skipped = 0
        error_cell = None
        violations = []

        for cid in cell_order[start:]:
            _, cell = self._find_cell(cid)
            source = get_cell_source(cell)
            if not source.strip():
                continue

            # Skip clean cells (executed, not stale, no violations)
            has_violation = bool(self.cell_flowbook_meta.get(cid, {}).get("errors"))
            is_clean = (
                cid in self.executed_cells
                and cid not in self._stale_cells
                and not has_violation
            )
            if is_clean:
                skipped += 1
                continue

            result = self.run_cell(cid, timeout=timeout)
            executed.append(cid)

            # Collect violations
            fb_meta = self.cell_flowbook_meta.get(cid, {})
            for e in fb_meta.get("errors", []):
                violations.append({"cell_id": cid, **e})

            if result.get("status") == "error":
                error_cell = cid
                break

        final_stale = self._stale_cells & set(cell_order)
        return {
            "executed": executed,
            "skipped": skipped,
            "error_cell": error_cell,
            "violations": violations,
            "stale_remaining": len(final_stale),
        }

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _mark_cell_edited(self, cell_id: str) -> None:
        """Mark a cell as stale and notify the kernel, if previously executed."""
        if cell_id in self.executed_cells:
            self._stale_cells.add(cell_id)
            KernelHelper.execute_code(
                self.kernel_client,
                "",
                timeout=10,
                store_history=False,
                flowbook_msg={"type": "cell_edited", "cell_id": cell_id},
            )

    def edit_cell(self, cell_id: str, new_source: str) -> Dict[str, Any]:
        """Update a cell's source and mark it stale if previously executed."""
        self._require_loaded()
        self._refresh_from_contents_api()
        _, cell = self._find_cell(cell_id)
        old_source = get_cell_source(cell)
        set_cell_source(cell, new_source)

        self._mark_cell_edited(cell_id)
        self._put_contents_api()

        return {
            "cell_id": cell_id,
            "old_source_preview": old_source[:200],
            "new_source_preview": new_source[:200],
            "marked_stale": cell_id in self._stale_cells,
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save notebook via Contents API (if available) or to disk.

        When a Jupyter server is running with jupyter-collaboration,
        saves via the Contents API which updates the live Y.js document
        and lets the server handle disk persistence. Falls back to
        direct disk write when no server is available or a custom
        path is specified.
        """
        self._require_loaded()
        save_path = path or self.notebook_path
        if not save_path:
            raise ValueError("No path specified and no original path available")

        # Use Contents API when available and saving to original path
        if not path and self._jupyter_contents_path:
            result = self._put_contents_api()
            if result:
                return result

        return cli_save_notebook(self.notebook, output_path=save_path)

    def _put_contents_api(self) -> Optional[str]:
        """Push the current notebook to JupyterLab via Contents API PUT.

        With jupyter-collaboration, this updates the live Y.js document,
        making MCP edits visible in JupyterLab. The server handles
        disk persistence.

        Before the PUT, checks whether JupyterLab has modified any cells
        since the last API refresh. If a cell was changed by JupyterLab AND
        differs from what MCP is about to send, a warning is logged and
        stored in ``_conflict_warnings``.

        Note: This is best-effort conflict detection, not true conflict
        resolution. A small race window remains between the check and the
        PUT. Full conflict-free editing would require a Y.js client.

        Returns:
            Status message on success, None on failure (caller falls back to disk).
        """
        if not self._jupyter_server_url or not self._jupyter_contents_path:
            return None

        # Check for concurrent JupyterLab edits by comparing what the API
        # has now against what we last saw from it.
        self._conflict_warnings = []
        if self._last_known_api_sources:
            api_notebook = self._fetch_contents_api()
            if api_notebook:
                local_sources = {
                    c["id"]: get_cell_source(c)
                    for c in self.notebook.get("cells", [])
                    if c.get("id") and c.get("cell_type") == "code"
                }
                for api_cell in api_notebook.get("cells", []):
                    cid = api_cell.get("id")
                    if not cid or api_cell.get("cell_type") != "code":
                        continue
                    last_known = self._last_known_api_sources.get(cid)
                    if last_known is None:
                        continue
                    api_source = api_cell.get("source", "")
                    local_source = local_sources.get(cid)
                    if local_source is None:
                        continue
                    # JupyterLab changed this cell AND MCP has something different
                    if api_source != last_known and api_source != local_source:
                        self._conflict_warnings.append(
                            f"Cell {cid}: JupyterLab edited this cell concurrently; "
                            f"MCP's version will overwrite the JupyterLab edit"
                        )
                        logger.warning(
                            "Contents API conflict on cell %s: JupyterLab source "
                            "changed since last refresh, MCP edit may overwrite it",
                            cid,
                        )

        try:
            url = (
                f"{self._jupyter_server_url}/api/contents/"
                f"{self._jupyter_contents_path}"
            )
            body = json.dumps({
                "type": "notebook",
                "format": "json",
                "content": self.notebook,
            }).encode()
            headers = {"Content-Type": "application/json"}
            if self._jupyter_token:
                headers["Authorization"] = f"token {self._jupyter_token}"
            req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
            urllib.request.urlopen(req, timeout=10)
            return self.notebook_path
        except Exception as e:
            logger.debug(f"Contents API PUT failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current notebook reproducibility status."""
        self._require_loaded()
        self._refresh_from_contents_api()
        self._poll_iopub()
        cell_order = self.get_cell_order()

        violations = []
        stale_with_reasons = {}

        # Determine which cells are actually stale right now:
        # - in _stale_cells (set by run_cell / edit_cell), OR
        # - unexecuted non-empty code cells
        actually_stale = set(self._stale_cells)
        for cid in cell_order:
            if cid not in self.executed_cells:
                source = get_cell_source(self._find_cell(cid)[1])
                if source.strip():
                    actually_stale.add(cid)

        for cid in cell_order:
            meta = self.cell_flowbook_meta.get(cid, {})
            errors = meta.get("errors", [])
            if errors:
                for e in errors:
                    violations.append({"cell_id": cid, **e})

            # Only collect staleness reasons for cells that are STILL stale.
            # Historical reasons (e.g., "never_executed" for a cell that has
            # since been executed) must be filtered out.
            reasons = meta.get("staleness_reasons", {})
            for stale_cid, reason_list in reasons.items():
                if stale_cid in actually_stale:
                    if stale_cid not in stale_with_reasons:
                        stale_with_reasons[stale_cid] = []
                    # Deduplicate: only add reasons not already present
                    existing = {json.dumps(r, sort_keys=True) for r in stale_with_reasons[stale_cid]}
                    for r in reason_list:
                        key = json.dumps(r, sort_keys=True)
                        if key not in existing:
                            stale_with_reasons[stale_cid].append(r)
                            existing.add(key)

        # Merge in edit-stale cells that don't already have reasons
        for cid in self._stale_cells:
            if cid in set(cell_order) and cid not in stale_with_reasons:
                stale_with_reasons[cid] = [{"type": "code_changed"}]

        executed_count = len(self.executed_cells & set(cell_order))
        clean_count = executed_count - len(
            set(self.cell_status.keys())
            & {k for k, v in self.cell_status.items() if v == "error"}
        ) - len(self._stale_cells & self.executed_cells)

        lines = [
            f"Notebook: {self.notebook_path}",
            f"Code cells: {len(cell_order)}",
            f"Executed: {executed_count}",
            f"Clean: {clean_count}",
            f"Violations: {len(violations)}",
            f"Stale: {len(stale_with_reasons)}",
        ]

        if violations:
            lines.append("\nViolations:")
            for v in violations:
                lines.append(f"  - Cell {v['cell_id']}: {format_error(v)}")

        if stale_with_reasons:
            lines.append(f"\nStale cells:\n{format_staleness_reasons(stale_with_reasons)}")

        return {
            "summary": "\n".join(lines),
            "violations": violations,
            "stale_cells": stale_with_reasons,
            "executed": executed_count,
            "total_code_cells": len(cell_order),
        }

    # ------------------------------------------------------------------
    # Checkpoint / Restore
    # ------------------------------------------------------------------

    def checkpoint(self) -> str:
        """Snapshot current notebook cell sources. Returns checkpoint_id."""
        self._require_loaded()
        ckpt_id = f"ckpt_{uuid.uuid4().hex[:8]}"
        cell_sources = {}
        for cell in self.notebook["cells"]:
            cell_sources[cell.get("id", "")] = get_cell_source(cell)

        self._checkpoints[ckpt_id] = {
            "timestamp": time.time(),
            "cell_sources": cell_sources,
            "cell_order": [c.get("id", "") for c in self.notebook["cells"]],
        }
        return ckpt_id

    def restore(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore notebook cell sources to a checkpoint.

        Does NOT restart the kernel. Changed cells are marked stale so
        they can be re-run incrementally via run_until_clean() or manually.
        """
        self._require_loaded()
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Unknown checkpoint: {checkpoint_id}")

        ckpt = self._checkpoints[checkpoint_id]
        cell_sources = ckpt["cell_sources"]

        # Restore cell sources and mark changed cells as stale
        changed_cells = []
        for cell in self.notebook["cells"]:
            cid = cell.get("id", "")
            if cid in cell_sources:
                old = get_cell_source(cell)
                if old != cell_sources[cid]:
                    set_cell_source(cell, cell_sources[cid])
                    changed_cells.append(cid)
                    # Mark as stale and notify kernel
                    self._mark_cell_edited(cid)
                    # Clear old violation metadata for changed cells
                    self.cell_flowbook_meta.pop(cid, None)
                    self.cell_status.pop(cid, None)

        return {
            "checkpoint_id": checkpoint_id,
            "cells_restored": len(changed_cells),
            "changed_cells": changed_cells,
        }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        result = []
        for ckpt_id, ckpt in self._checkpoints.items():
            result.append({
                "checkpoint_id": ckpt_id,
                "timestamp": ckpt["timestamp"],
                "cell_count": len(ckpt["cell_sources"]),
            })
        return result

    # ------------------------------------------------------------------
    # Algorithmic refactoring
    # ------------------------------------------------------------------

    def alpha_rename(
        self, cell_id: str, old_name: str, new_name: str
    ) -> Dict[str, Any]:
        """Rename a variable from cell_id onwards using AST-based rename."""
        self._require_loaded()
        self._refresh_from_contents_api()
        code_order = self.get_cell_order()
        if cell_id not in code_order:
            raise ValueError(
                f"Cell '{cell_id}' not found in notebook. "
                f"Available code cells: {code_order}"
            )
        start_idx = code_order.index(cell_id)

        modified_cells = []
        for cid in code_order[start_idx:]:
            _, cell = self._find_cell(cid)
            source = get_cell_source(cell)
            new_source, renamed = rename_variable_in_code(source, old_name, new_name)
            if renamed:
                set_cell_source(cell, new_source)
                modified_cells.append(cid)
                self._mark_cell_edited(cid)

        if modified_cells:
            self._put_contents_api()

        return {
            "old_name": old_name,
            "new_name": new_name,
            "modified_cells": modified_cells,
            "total_modified": len(modified_cells),
        }

    def remove_inplace(self, cell_id: str, variable: str) -> Dict[str, Any]:
        """Convert df.method(inplace=True) to df = df.method() in a cell."""
        self._require_loaded()
        self._refresh_from_contents_api()
        _, cell = self._find_cell(cell_id)
        source = get_cell_source(cell)

        # Find actual variable name (may have been renamed)
        actual_var = find_actual_variable_name(source, variable)

        try:
            tree = ast.parse(source)
            remover = InplaceRemover(actual_var)
            new_tree = remover.visit(tree)
            if remover.modified:
                new_source = ast.unparse(new_tree)
                set_cell_source(cell, new_source)
                self._mark_cell_edited(cell_id)
                self._put_contents_api()
                return {
                    "cell_id": cell_id,
                    "variable": actual_var,
                    "methods_fixed": remover.method_calls_fixed,
                    "new_source": new_source,
                }
        except SyntaxError:
            pass

        # Regex fallback
        pattern = rf"(\b{re.escape(actual_var)}\.(\w+)\([^)]*),\s*inplace\s*=\s*True([^)]*)\)"
        new_source, count = re.subn(
            pattern,
            rf"{actual_var} = \1\3)",
            source,
        )
        if count > 0:
            set_cell_source(cell, new_source)
            self._mark_cell_edited(cell_id)
            self._put_contents_api()
            return {
                "cell_id": cell_id,
                "variable": actual_var,
                "methods_fixed": ["(regex fallback)"],
                "new_source": new_source,
            }

        return {
            "cell_id": cell_id,
            "error": f"No inplace=True found for variable '{actual_var}'",
        }

    def insert_deepcopy(self, cell_id: str, variable: str) -> Dict[str, Any]:
        """Insert deepcopy of variable at top of cell, rename downstream."""
        self._require_loaded()
        self._refresh_from_contents_api()
        _, cell = self._find_cell(cell_id)
        source = get_cell_source(cell)

        actual_var = find_actual_variable_name(source, variable)
        new_name = f"{variable}_copy"

        # Insert deepcopy at top of cell
        copy_line = f"import copy; {new_name} = copy.deepcopy({actual_var})\n"
        new_source = prepend_to_cell_source(source, copy_line)

        # Rename variable in this cell (after the copy line)
        new_source, _ = rename_variable_in_code(new_source, actual_var, new_name)
        # But the deepcopy line itself should keep the original name on RHS
        # Fix: the copy line uses actual_var on RHS which is correct since
        # rename_variable_in_code would rename it. We need to be smarter here.
        # Actually, we want: import copy; new_name = copy.deepcopy(actual_var)
        # then rename actual_var -> new_name in the REST of the cell.
        # Let's do this more carefully:
        magic_prefix, rest = split_cell_magic(source)
        renamed_rest, _ = rename_variable_in_code(rest, actual_var, new_name)
        new_source = magic_prefix + copy_line + renamed_rest

        set_cell_source(cell, new_source)
        self._mark_cell_edited(cell_id)

        # Rename in all downstream cells
        code_order = self.get_cell_order()
        if cell_id not in code_order:
            raise ValueError(
                f"Cell '{cell_id}' not found in notebook. "
                f"Available code cells: {code_order}"
            )
        start_idx = code_order.index(cell_id)
        modified_downstream = []
        for cid in code_order[start_idx + 1:]:
            _, dcell = self._find_cell(cid)
            dsource = get_cell_source(dcell)
            dnew, renamed = rename_variable_in_code(dsource, actual_var, new_name)
            if renamed:
                set_cell_source(dcell, dnew)
                modified_downstream.append(cid)
                self._mark_cell_edited(cid)

        self._put_contents_api()

        return {
            "cell_id": cell_id,
            "variable": actual_var,
            "new_name": new_name,
            "modified_downstream": modified_downstream,
        }

    def mark_diagnostic(self, cell_id: str) -> Dict[str, Any]:
        """Add %diagnostic magic to a cell."""
        self._require_loaded()
        self._refresh_from_contents_api()
        _, cell = self._find_cell(cell_id)
        source = get_cell_source(cell)

        if source.lstrip().startswith("%diagnostic"):
            return {"cell_id": cell_id, "already_diagnostic": True}

        new_source = prepend_to_cell_source(source, "%diagnostic\n")
        set_cell_source(cell, new_source)

        self._mark_cell_edited(cell_id)
        self._put_contents_api()
        return {"cell_id": cell_id, "new_source_preview": new_source[:200]}

    def add_cell(self, source: str, cell_type: str = "code",
                 after_cell_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a new cell to the notebook.

        Generates a unique cell ID, inserts the cell, notifies the kernel
        of the new structure, and syncs to JupyterLab via Contents API.

        Args:
            source: Source code (or markdown) for the new cell.
            cell_type: "code" or "markdown".
            after_cell_id: Insert after this cell. If None, appends to end.

        Returns:
            Dict with cell_id, cell_type, and new_cell_order.
        """
        self._require_loaded()
        self._refresh_from_contents_api()

        # Generate unique cell ID
        existing_ids = {c.get("id", "") for c in self.notebook["cells"]}
        if after_cell_id:
            new_id = next_insertion_id(after_cell_id, existing_ids)
        else:
            # Append: use last cell's ID as base, or "a" if empty
            code_cells = [c for c in self.notebook["cells"] if c.get("cell_type") == "code"]
            base = code_cells[-1]["id"] if code_cells else "a"
            new_id = next_insertion_id(base, existing_ids)

        new_cell = {
            "id": new_id,
            "cell_type": cell_type,
            "source": source,
            "metadata": {},
            "outputs": [] if cell_type == "code" else [],
        }

        if after_cell_id:
            # Insert after the specified cell
            insert_idx = None
            for i, c in enumerate(self.notebook["cells"]):
                if c.get("id") == after_cell_id:
                    insert_idx = i + 1
                    break
            if insert_idx is None:
                raise ValueError(f"Cell not found: {after_cell_id}")
            self.notebook["cells"].insert(insert_idx, new_cell)
        else:
            self.notebook["cells"].append(new_cell)

        # Notify kernel of updated structure
        if cell_type == "code":
            new_order = self.get_cell_order()
            KernelHelper.execute_code(
                self.kernel_client,
                "",
                timeout=10,
                store_history=False,
                flowbook_msg={"type": "notebook_structure", "cell_order": new_order},
            )

        self._put_contents_api()

        return {
            "cell_id": new_id,
            "cell_type": cell_type,
            "new_cell_order": self.get_cell_order(),
        }

    def delete_cell(self, cell_id: str) -> Dict[str, Any]:
        """Remove a cell from the notebook.

        Removes the cell, cleans up tracking state, notifies the kernel,
        and syncs to JupyterLab via Contents API.

        Args:
            cell_id: The cell ID to delete.

        Returns:
            Dict with cell_id and new_cell_order.
        """
        self._require_loaded()
        self._refresh_from_contents_api()

        # Find and remove the cell
        _, cell = self._find_cell(cell_id)
        cell_type = cell.get("cell_type", "code")
        self.notebook["cells"] = [
            c for c in self.notebook["cells"] if c.get("id") != cell_id
        ]

        # Clean up tracking state
        self.executed_cells.discard(cell_id)
        self.cell_flowbook_meta.pop(cell_id, None)
        self.cell_status.pop(cell_id, None)
        self._stale_cells.discard(cell_id)

        # Notify kernel of updated structure
        if cell_type == "code":
            new_order = self.get_cell_order()
            KernelHelper.execute_code(
                self.kernel_client,
                "",
                timeout=10,
                store_history=False,
                flowbook_msg={"type": "notebook_structure", "cell_order": new_order},
            )

        self._put_contents_api()

        return {
            "cell_id": cell_id,
            "new_cell_order": self.get_cell_order(),
        }

    def merge_cells(self, cell_ids: List[str]) -> Dict[str, Any]:
        """Merge multiple cells into the first one."""
        self._require_loaded()
        self._refresh_from_contents_api()
        if len(cell_ids) < 2:
            raise ValueError("Need at least 2 cell IDs to merge")

        # Collect sources in order
        cells_to_merge = []
        for cid in cell_ids:
            _, cell = self._find_cell(cid)
            cells_to_merge.append((cid, cell))

        # Merge sources
        sources = [get_cell_source(c) for _, c in cells_to_merge]
        merged_source = "\n\n".join(s for s in sources if s.strip())

        # Update first cell
        first_id, first_cell = cells_to_merge[0]
        set_cell_source(first_cell, merged_source)

        # Remove subsequent cells from notebook
        ids_to_remove = set(cell_ids[1:])
        self.notebook["cells"] = [
            c for c in self.notebook["cells"]
            if c.get("id") not in ids_to_remove
        ]

        self._mark_cell_edited(first_id)

        # Clean up tracking for removed cells
        for cid in ids_to_remove:
            self.executed_cells.discard(cid)
            self.cell_flowbook_meta.pop(cid, None)
            self.cell_status.pop(cid, None)
            self._stale_cells.discard(cid)

        # Update kernel with new cell order
        new_order = self.get_cell_order()
        order_str = " ".join(new_order)
        KernelHelper.execute_code(
            self.kernel_client,
            "",
            timeout=10,
            store_history=False,
            flowbook_msg={"type": "notebook_structure", "cell_order": order_str.split()},
        )

        self._put_contents_api()

        return {
            "merged_cell_id": first_id,
            "cells_removed": list(ids_to_remove),
            "new_source_preview": merged_source[:300],
            "new_cell_order": new_order,
        }

    def move_cell(self, cell_id: str, after_cell_id: str) -> Dict[str, Any]:
        """Move a cell to after another cell in the notebook."""
        self._require_loaded()
        self._refresh_from_contents_api()
        cells = self.notebook["cells"]

        # Find and remove the cell to move
        src_idx, cell_to_move = self._find_cell(cell_id)
        cells.pop(src_idx)

        # Find destination
        dst_idx = None
        for i, c in enumerate(cells):
            if c.get("id") == after_cell_id:
                dst_idx = i + 1
                break
        if dst_idx is None:
            # Put it back and raise
            cells.insert(src_idx, cell_to_move)
            raise ValueError(f"Destination cell not found: {after_cell_id}")

        cells.insert(dst_idx, cell_to_move)

        # Update kernel with new cell order
        new_order = self.get_cell_order()
        order_str = " ".join(new_order)
        KernelHelper.execute_code(
            self.kernel_client,
            "",
            timeout=10,
            store_history=False,
            flowbook_msg={"type": "notebook_structure", "cell_order": order_str.split()},
        )

        self._put_contents_api()

        return {
            "cell_id": cell_id,
            "moved_after": after_cell_id,
            "new_cell_order": new_order,
        }
