"""
NotebookSession — manages a single (notebook, kernel) pair for the MCP server.

This is the stateful core: it holds the in-memory notebook, a running FlowBook
kernel, and accumulated reproducibility metadata from cell executions.
"""

import ast
import copy
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from flowbook.cli.helpers import (
    cleanup_kernel,
    save_notebook as cli_save_notebook,
    setup_kernel,
)
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
# Metadata formatting
# ---------------------------------------------------------------------------

def format_loc(loc) -> str:
    """Format a ReadLoc/WriteLoc dict (or pre-formatted string) as a human-readable string.

    {"type": "var", "name": "df"} -> "df"
    {"type": "col", "name": "df", "qualifier": "price"} -> "df.price"
    {"type": "struct", "name": "df", "qualifier": "columns"} -> "df[columns]"
    "df['age']" -> "df['age']"  (already formatted, from ReproducibilityError.locations)
    """
    if isinstance(loc, str):
        return loc
    name = loc.get("name", "?")
    loc_type = loc.get("type", "var")
    qualifier = loc.get("qualifier")
    if loc_type == "col" and qualifier:
        return f"{name}.{qualifier}"
    if loc_type == "struct" and qualifier:
        return f"{name}[{qualifier}]"
    return name


def format_loc_list(locs: list) -> str:
    """Format a list of locs as comma-separated readable names."""
    if not locs:
        return "(none)"
    return ", ".join(format_loc(loc) for loc in locs)


def format_error(error: Dict[str, Any]) -> str:
    """Format a reproducibility error dict as a readable string."""
    etype = error.get("error_type", "unknown")
    msg = error.get("message", "")
    locs = error.get("locations", [])
    causer = error.get("causer_cell", "")
    parts = [f"{etype}: {msg}"]
    if locs:
        parts.append(f"  Locations: {format_loc_list(locs)}")
    if causer:
        parts.append(f"  Causer cell: {causer}")
    return "\n".join(parts)


def format_staleness_reasons(reasons: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format staleness reasons dict as readable text."""
    if not reasons:
        return "(none)"
    lines = []
    for cell_id, reason_list in reasons.items():
        reason_strs = []
        for r in reason_list:
            rtype = r.get("type", "unknown")
            loc = r.get("loc", "")
            cause = r.get("cell_id", "")
            s = rtype
            if loc:
                s += f": {loc}"
            if cause:
                s += f" (from cell {cause})"
            reason_strs.append(s)
        lines.append(f"  {cell_id}: {', '.join(reason_strs)}")
    return "\n".join(lines)


def format_flowbook_meta(meta: Dict[str, Any]) -> str:
    """Format raw flowbook metadata into human+LLM-readable text."""
    lines = []

    reads = meta.get("read_locs", [])
    writes = meta.get("write_locs", [])
    changed = meta.get("changed_locs", [])
    errors = meta.get("errors", [])
    stale = meta.get("stale_cells", [])
    reasons = meta.get("staleness_reasons", {})

    lines.append(f"Reads: {format_loc_list(reads)}")
    lines.append(f"Writes: {format_loc_list(writes)}")
    if changed:
        lines.append(f"Changed: {format_loc_list(changed)}")

    if errors:
        lines.append("Errors:")
        for e in errors:
            lines.append(f"  - {format_error(e)}")
    else:
        lines.append("Errors: (none)")

    if stale:
        lines.append(f"Stale cells: {', '.join(stale)}")
        if reasons:
            lines.append(f"Staleness reasons:\n{format_staleness_reasons(reasons)}")
    else:
        lines.append("Stale cells: (none)")

    # Timing
    exec_ms = meta.get("execute_duration_ms")
    code_ms = meta.get("code_duration_ms")
    if exec_ms is not None:
        timing_parts = [f"total={exec_ms:.0f}ms"]
        if code_ms is not None:
            timing_parts.append(f"code={code_ms:.0f}ms")
        lines.append(f"Timing: {', '.join(timing_parts)}")

    return "\n".join(lines)


def _to_str(value) -> str:
    """Convert a value that may be a string or list of strings to a string."""
    if isinstance(value, list):
        return "".join(value)
    return str(value) if value is not None else ""


def format_outputs_text(outputs: List[Dict[str, Any]]) -> str:
    """Extract human-readable text from cell outputs."""
    parts = []
    for output in outputs:
        otype = output.get("output_type", "")
        if otype == "stream":
            parts.append(_to_str(output.get("text", "")))
        elif otype == "execute_result":
            data = output.get("data", {})
            if "text/plain" in data:
                parts.append(_to_str(data["text/plain"]))
        elif otype == "display_data":
            data = output.get("data", {})
            meta = output.get("metadata", {})
            # Skip flowbook metadata display_data
            if "flowbook" not in meta:
                if "text/plain" in data:
                    parts.append(_to_str(data["text/plain"]))
                elif "text/html" in data:
                    parts.append("[HTML output]")
                elif "image/png" in data:
                    parts.append("[Image output]")
        elif otype == "error":
            ename = output.get("ename", "Error")
            evalue = output.get("evalue", "")
            parts.append(f"{ename}: {evalue}")
    return "\n".join(parts) if parts else "(no output)"


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
        self.executed_cells: Set[str] = set()
        self.cell_flowbook_meta: Dict[str, Dict] = {}
        self.cell_status: Dict[str, str] = {}  # cell_id -> "ok" | "error"
        self._stale_cells: Set[str] = set()
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._event_log: List[Dict[str, Any]] = []
        self._session_start: float = time.time()

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
        """Load notebook from disk, start kernel, assign alpha cell IDs."""
        # Close any previous session
        if self.is_loaded:
            self.close()

        with open(path, "r", encoding="utf-8") as f:
            raw_notebook = json.load(f)
        self.notebook = normalize_notebook_alpha(raw_notebook)
        self.notebook_path = path
        self.executed_cells = set()
        self.cell_flowbook_meta = {}
        self.cell_status = {}
        self._stale_cells = set()
        self._checkpoints = {}
        self._event_log = []
        self._session_start = time.time()

        # Start FlowBook kernel with cwd set to the notebook's directory
        # so relative paths (pd.read_csv("data.csv")) resolve correctly
        import os
        notebook_dir = os.path.dirname(os.path.abspath(path))
        self.kernel_manager, self.kernel_client = setup_kernel(
            connection_file=None,
            kernel_name="flowbook_kernel",
            cwd=notebook_dir,
        )

        cells = self.notebook.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        return {
            "path": path,
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "cell_ids": [c["id"] for c in code_cells],
        }

    def close(self):
        """Shutdown kernel, auto-save log, and clear state."""
        # Auto-save event log if there were any events
        if self._event_log and self.notebook_path:
            try:
                self.save_event_log()
            except Exception:
                pass  # best-effort log save on close
        if self.kernel_client or self.kernel_manager:
            cleanup_kernel(self.kernel_client, self.kernel_manager)
        self.kernel_client = None
        self.kernel_manager = None
        self.notebook = None
        self.notebook_path = None
        self.executed_cells = set()
        self.cell_flowbook_meta = {}
        self.cell_status = {}
        self._stale_cells = set()
        self._checkpoints = {}

    def set_continue_after_violation(self, enabled: bool) -> None:
        """Configure whether violations reject execution or just report.

        When enabled (True): violations are reported but execution continues,
        cell stays CLEAN. Good for /basic-run where you want a full picture.

        When disabled (False, default): violations cause rollback and the cell
        is rejected. Good for /fix-notebook where you want a clean namespace.
        """
        self._require_loaded()
        flag = "on" if enabled else "off"
        KernelHelper.execute_code(
            self.kernel_client,
            f"%continue_after_violation {flag}",
            timeout=10,
            store_history=False,
        )

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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _extract_flowbook_meta(
        self, outputs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract flowbook metadata from display_data outputs."""
        for output in outputs:
            if output.get("output_type") == "display_data":
                output_meta = output.get("metadata", {})
                if "flowbook" in output_meta:
                    return output_meta["flowbook"]
        return None

    def run_cell(self, cell_id: str, timeout: float = 300) -> Dict[str, Any]:
        """Execute a single cell and return outputs + flowbook metadata."""
        self._require_loaded()
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
        self.executed_cells.add(cell_id)

        # Extract flowbook metadata
        fb_meta = self._extract_flowbook_meta(result["outputs"])
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

        Skips cells that are already clean. Only runs cells that are
        unexecuted, stale, or have violations.

        Args:
            cell_id: Cell to start from.
            timeout: Per-cell timeout in seconds.

        Returns:
            Summary with list of executed cells, skipped count, error cell
            (if any), and violation/stale counts.
        """
        self._require_loaded()
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

    def edit_cell(self, cell_id: str, new_source: str) -> Dict[str, Any]:
        """Update a cell's source and mark it stale if previously executed."""
        self._require_loaded()
        _, cell = self._find_cell(cell_id)
        old_source = get_cell_source(cell)
        set_cell_source(cell, new_source)

        newly_stale = []

        # Notify kernel if this cell was previously executed
        if cell_id in self.executed_cells:
            self._stale_cells.add(cell_id)
            KernelHelper.execute_code(
                self.kernel_client,
                f"%cell_edited {cell_id}",
                timeout=10,
                store_history=False,
            )

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
        """Save notebook to disk."""
        self._require_loaded()
        save_path = path or self.notebook_path
        if not save_path:
            raise ValueError("No path specified and no original path available")
        return cli_save_notebook(self.notebook, output_path=save_path)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current notebook reproducibility status."""
        self._require_loaded()
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
                    if cid in self.executed_cells:
                        self._stale_cells.add(cid)
                        KernelHelper.execute_code(
                            self.kernel_client,
                            f"%cell_edited {cid}",
                            timeout=10,
                            store_history=False,
                        )
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
        code_order = self.get_cell_order()
        start_idx = code_order.index(cell_id)

        modified_cells = []
        for cid in code_order[start_idx:]:
            _, cell = self._find_cell(cid)
            source = get_cell_source(cell)
            new_source, renamed = rename_variable_in_code(source, old_name, new_name)
            if renamed:
                set_cell_source(cell, new_source)
                modified_cells.append(cid)
                # Mark stale if previously executed
                if cid in self.executed_cells:
                    self._stale_cells.add(cid)
                    KernelHelper.execute_code(
                        self.kernel_client,
                        f"%cell_edited {cid}",
                        timeout=10,
                        store_history=False,
                    )

        return {
            "old_name": old_name,
            "new_name": new_name,
            "modified_cells": modified_cells,
            "total_modified": len(modified_cells),
        }

    def remove_inplace(self, cell_id: str, variable: str) -> Dict[str, Any]:
        """Convert df.method(inplace=True) to df = df.method() in a cell."""
        self._require_loaded()
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
                if cell_id in self.executed_cells:
                    self._stale_cells.add(cell_id)
                    KernelHelper.execute_code(
                        self.kernel_client,
                        f"%cell_edited {cell_id}",
                        timeout=10,
                        store_history=False,
                    )
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
            if cell_id in self.executed_cells:
                self._stale_cells.add(cell_id)
                KernelHelper.execute_code(
                    self.kernel_client,
                    f"%cell_edited {cell_id}",
                    timeout=10,
                    store_history=False,
                )
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

        # Mark stale
        if cell_id in self.executed_cells:
            self._stale_cells.add(cell_id)
            KernelHelper.execute_code(
                self.kernel_client,
                f"%cell_edited {cell_id}",
                timeout=10,
                store_history=False,
            )

        # Rename in all downstream cells
        code_order = self.get_cell_order()
        start_idx = code_order.index(cell_id)
        modified_downstream = []
        for cid in code_order[start_idx + 1:]:
            _, dcell = self._find_cell(cid)
            dsource = get_cell_source(dcell)
            dnew, renamed = rename_variable_in_code(dsource, actual_var, new_name)
            if renamed:
                set_cell_source(dcell, dnew)
                modified_downstream.append(cid)
                if cid in self.executed_cells:
                    self._stale_cells.add(cid)
                    KernelHelper.execute_code(
                        self.kernel_client,
                        f"%cell_edited {cid}",
                        timeout=10,
                        store_history=False,
                    )

        return {
            "cell_id": cell_id,
            "variable": actual_var,
            "new_name": new_name,
            "modified_downstream": modified_downstream,
        }

    def mark_diagnostic(self, cell_id: str) -> Dict[str, Any]:
        """Add %diagnostic magic to a cell."""
        self._require_loaded()
        _, cell = self._find_cell(cell_id)
        source = get_cell_source(cell)

        if source.lstrip().startswith("%diagnostic"):
            return {"cell_id": cell_id, "already_diagnostic": True}

        new_source = prepend_to_cell_source(source, "%diagnostic\n")
        set_cell_source(cell, new_source)

        if cell_id in self.executed_cells:
            self._stale_cells.add(cell_id)
            KernelHelper.execute_code(
                self.kernel_client,
                f"%cell_edited {cell_id}",
                timeout=10,
                store_history=False,
            )

        return {"cell_id": cell_id, "new_source_preview": new_source[:200]}

    def merge_cells(self, cell_ids: List[str]) -> Dict[str, Any]:
        """Merge multiple cells into the first one."""
        self._require_loaded()
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

        # Mark stale and notify kernel
        if first_id in self.executed_cells:
            self._stale_cells.add(first_id)
            KernelHelper.execute_code(
                self.kernel_client,
                f"%cell_edited {first_id}",
                timeout=10,
                store_history=False,
            )

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
            f"%notebook_structure {order_str}",
            timeout=10,
            store_history=False,
        )

        return {
            "merged_cell_id": first_id,
            "cells_removed": list(ids_to_remove),
            "new_source_preview": merged_source[:300],
            "new_cell_order": new_order,
        }

    def move_cell(self, cell_id: str, after_cell_id: str) -> Dict[str, Any]:
        """Move a cell to after another cell in the notebook."""
        self._require_loaded()
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
            f"%notebook_structure {order_str}",
            timeout=10,
            store_history=False,
        )

        return {
            "cell_id": cell_id,
            "moved_after": after_cell_id,
            "new_cell_order": new_order,
        }
