"""FlowBookTools — unified tool logic for MCP and NBI surfaces.

All methods are synchronous, taking a NotebookSession directly.
NBI tools wrap calls with run_in_executor for async compatibility.
MCP tools call methods directly with no overhead.

Cell references accept either @A notation or 4-char cell IDs.
"""

from typing import Dict, List, Optional

from flowbook.nbi.cell_addressing import index_to_alpha, alpha_to_index
from flowbook.tools.format import (
    format_flowbook_meta,
    format_loc,
    format_outputs_text,
    format_rename_result,
    format_run_result,
    format_status,
    format_violation_line,
)


class FlowBookTools:
    """Shared tool implementations backed by NotebookSession.

    Constructed with a loaded NotebookSession. Both MCP server.py and
    NBI tools.py create thin registration wrappers around this class.
    """

    def __init__(self, session):
        """Initialize with a NotebookSession (must already be loaded for most tools)."""
        self.session = session

    # ------------------------------------------------------------------
    # Cell reference resolution
    # ------------------------------------------------------------------

    def _resolve_ref(self, ref: str) -> str:
        """Convert @A notation or numeric string to cell_id.

        Accepts:
        - @A / @AA / @AAA: alpha label → index → cell_id
        - Plain letters (A, AA): treated as alpha label
        - 4-char alphanumeric (e.g. 'ab3f'): treated as cell_id directly
        - Numeric string ('2'): treated as code-cell index

        Raises ValueError if the reference can't be resolved.
        """
        if not ref or not isinstance(ref, str):
            raise ValueError(f"Invalid cell reference: {ref!r}")

        ref = ref.strip()
        order = self.session.get_cell_order()

        # @-prefixed alpha label
        if ref.startswith('@'):
            idx = alpha_to_index(ref)
            if idx >= len(order):
                raise ValueError(f"Cell {ref} out of range (notebook has {len(order)} code cells)")
            return order[idx]

        # Numeric string → index
        if ref.isdigit():
            idx = int(ref)
            if idx >= len(order):
                raise ValueError(f"Cell index {idx} out of range (notebook has {len(order)} code cells)")
            return order[idx]

        # If it looks like an existing cell_id (in the order list), use directly
        if ref in order:
            return ref

        # Try as alpha label without @
        try:
            idx = alpha_to_index(ref)
            if idx < len(order):
                return order[idx]
        except ValueError:
            pass

        raise ValueError(f"Cannot resolve cell reference: {ref!r}")

    def _label(self, cell_id: str) -> str:
        """Convert cell_id to @A label."""
        order = self.session.get_cell_order()
        try:
            return index_to_alpha(order.index(cell_id))
        except ValueError:
            return cell_id

    # ------------------------------------------------------------------
    # Status & metadata tools
    # ------------------------------------------------------------------

    def get_all_cell_sources(self) -> str:
        """Return source code of all code cells with @-labels and status."""
        self.session._require_loaded()
        self.session.refresh_from_jupyter()
        cell_order = self.session.get_cell_order()
        if not cell_order:
            return "No code cells in notebook."

        parts = []
        for idx, cid in enumerate(cell_order):
            label = index_to_alpha(idx)
            _, cell = self.session._find_cell(cid)
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            status = self._cell_status(cid)
            parts.append(f"\u2500\u2500 {label} [{cid}] ({status}) \u2500\u2500\n{source}")

        return "\n\n".join(parts)

    def read_cell(self, cell: str = "") -> str:
        """Read cell source, outputs, and flowbook metadata.

        If cell is provided, reads that single cell.
        If cell is empty/omitted, returns all code cells with labels and status.
        """
        if not cell:
            return self.get_all_cell_sources()

        cell_id = self._resolve_ref(cell)
        result = self.session.get_cell(cell_id)
        label = self._label(cell_id)

        line = f"{label} [{result['cell_id']}] ({result.get('status', '?')})"
        line += f"\n>>> {result['source']}"

        output_text = result.get("outputs_text", "").strip()
        if output_text:
            preview = output_text[:300]
            if len(output_text) > 300:
                preview += "..."
            line += f"\nOutput: {preview}"

        if "flowbook" in result:
            line += f"\n{result['flowbook']}"

        return line

    def get_next_actionable_cell(self) -> str:
        """Get the first cell needing attention, or 'All clean.'"""
        result = self.session.get_next_actionable()
        if result is None:
            return "All clean."

        label = self._label(result['cell_id'])
        line = f"{label} [{result['cell_id']}]: {result['reason']}"
        if "violation_summary" in result:
            line += f" \u2014 {result['violation_summary']}"
        line += f"\n>>> {result['source']}"
        return line

    def get_flowbook_metadata(self, cell: str) -> str:
        """Return reproducibility metadata for a specific cell."""
        cell_id = self._resolve_ref(cell)
        self.session._require_loaded()
        label = self._label(cell_id)
        meta = self.session.cell_flowbook_meta.get(cell_id)
        if meta is None:
            return f"Cell {label} [{cell_id}] has not been executed yet \u2014 no metadata available."
        return f"{label} [{cell_id}]:\n{format_flowbook_meta(meta)}"

    def get_status(self) -> str:
        """Get the notebook's current reproducibility status."""
        result = self.session.get_status()
        return format_status(result, self._label)

    def list_cells(self) -> str:
        """List all cells with index, ID, type, first line, and status."""
        self.session._require_loaded()
        self.session.refresh_from_jupyter()
        lines = []
        code_idx = 0
        for cell in self.session.notebook["cells"]:
            cid = cell.get("id", "?")
            ctype = cell.get("cell_type", "?")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            first_line = source.split("\n")[0][:60] if source.strip() else "(empty)"

            if ctype == "code":
                label = index_to_alpha(code_idx)
                status = self._cell_status(cid)
                fb_meta = self.session.cell_flowbook_meta.get(cid, {})
                viol = " !" if fb_meta.get("errors") else ""
                lines.append(f"{label} [{cid}] {status}{viol}: {first_line}")
                code_idx += 1
            else:
                lines.append(f"[ ] {cid} ({ctype}): {first_line}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cell editing tools
    # ------------------------------------------------------------------

    def edit_cell_source(self, cell: str, new_source: str) -> str:
        """Replace a cell's source code."""
        cell_id = self._resolve_ref(cell)
        result = self.session.edit_cell(cell_id, new_source)
        label = self._label(cell_id)
        stale_note = " (marked stale)" if result["marked_stale"] else ""
        return (
            f"Updated cell {label} [{result['cell_id']}]{stale_note}\n"
            f"New source preview: {result['new_source_preview']}"
        )

    def add_cell(self, source: str, cell_type: str = "code",
                 after_cell: Optional[str] = None) -> str:
        """Add a new cell to the notebook."""
        after_id = self._resolve_ref(after_cell) if after_cell else None
        result = self.session.add_cell(source, cell_type, after_cell_id=after_id)
        label = self._label(result["cell_id"])
        return f"Added {cell_type} cell {label} [{result['cell_id']}]"

    def delete_cell(self, cell: str) -> str:
        """Remove a cell from the notebook."""
        cell_id = self._resolve_ref(cell)
        label = self._label(cell_id)
        self.session.delete_cell(cell_id)
        return f"Deleted cell {label} [{cell_id}]"

    # ------------------------------------------------------------------
    # Execution tools
    # ------------------------------------------------------------------

    def run_cell(self, cell: str) -> str:
        """Execute a single cell and return outputs + flowbook metadata."""
        cell_id = self._resolve_ref(cell)
        result = self.session.run_cell(cell_id)
        label = self._label(cell_id)
        return format_run_result(result, label)

    def run_all_cells(self) -> str:
        """Execute all code cells top-to-bottom."""
        result = self.session.run_all()
        n = result["total_executed"]
        total = result["total_code_cells"]
        violations = result["violations"]
        stale = result["stale_cells"]
        status = result["status"]

        line = f"Executed {n}/{total} code cells"
        if status == "error":
            line += " (stopped on error)"
        line += f" | {len(violations)} violations | {len(stale)} stale"

        for e in violations:
            cid = e.get("cell_id", "?")
            line += "\n" + format_violation_line(e, self._label(cid), cid)

        return line

    def run_from(self, cell: str) -> str:
        """Run from a cell through the end of the notebook."""
        cell_id = self._resolve_ref(cell)
        result = self.session.run_from(cell_id)
        start_label = self._label(cell_id)
        n = len(result["executed"])
        v = len(result["violations"])
        s = result["stale_remaining"]
        sk = result["skipped"]
        line = f"Ran {n} cells from {start_label} [{cell_id}]"
        if sk:
            line += f" ({sk} clean skipped)"
        if result["error_cell"]:
            err_label = self._label(result['error_cell'])
            line += f" | error at {err_label} [{result['error_cell']}]"
        line += f" | {v} violations | {s} stale"
        for e in result["violations"]:
            cid = e.get("cell_id", "?")
            line += "\n" + format_violation_line(e, self._label(cid), cid)
        return line

    def run_actionable_cell(self) -> str:
        """Find and run the next actionable cell."""
        next_id = self.session.get_next_actionable_cell_id()
        if next_id is None:
            return "All clean \u2014 no actionable cells."
        label = self._label(next_id)
        result = self.session.run_cell(next_id)
        return f"Ran {label} [{result['cell_id']}]: " + format_run_result(result, label)

    def run_actionable_cells(self) -> str:
        """Run all actionable cells until clean or error."""
        self.session._require_loaded()

        cells_ran = []
        violations_seen = []
        error_cell = None

        while True:
            next_id = self.session.get_next_actionable_cell_id()
            if next_id is None:
                break

            result = self.session.run_cell(next_id)
            label = self._label(next_id)
            cells_ran.append(f"{label} [{next_id}]")

            if result.get("status") == "error":
                error_cell = next_id
                break

            fb_meta = self.session.cell_flowbook_meta.get(next_id, {})
            errors = fb_meta.get("errors", [])
            if errors:
                for e in errors:
                    violations_seen.append({"cell_id": next_id, **e})
                if not self.session._continue_after_violation:
                    break

        n = len(cells_ran)
        line = f"Ran {n} cells"
        if error_cell:
            err_label = self._label(error_cell)
            line += f" | error at {err_label} [{error_cell}]"
        line += f" | {len(violations_seen)} violations"

        status = self.session.get_status()
        stale_count = len(status["stale_cells"])
        line += f" | {stale_count} stale"

        if error_cell is None and not violations_seen and stale_count == 0:
            line += "\nAll clean!"
        elif violations_seen:
            for e in violations_seen:
                cid = e.get("cell_id", "?")
                line += "\n" + format_violation_line(e, self._label(cid), cid)

        if cells_ran:
            line += f"\nCells: {', '.join(cells_ran)}"

        return line

    def continue_after_violation(self, enabled: bool) -> str:
        """Configure violation handling mode."""
        self.session.set_continue_after_violation(enabled)
        mode = "continue (report only)" if enabled else "reject (rollback)"
        return f"Violation mode: {mode}"

    # ------------------------------------------------------------------
    # Refactoring tools
    # ------------------------------------------------------------------

    def alpha_rename(self, cell: str, old_name: str, new_name: str) -> str:
        """Rename a variable from a cell onwards using AST-based transformation."""
        cell_id = self._resolve_ref(cell)
        result = self.session.alpha_rename(cell_id, old_name, new_name)
        start_label = self._label(cell_id)
        mod_labels = [f"{self._label(c)} [{c}]" for c in result['modified_cells']]
        return format_rename_result(old_name, new_name, mod_labels, start_label)

    def remove_inplace(self, cell: str, variable: str) -> str:
        """Convert df.method(inplace=True) to df = df.method()."""
        cell_id = self._resolve_ref(cell)
        result = self.session.remove_inplace(cell_id, variable)
        if "error" in result:
            return f"Error: {result['error']}"
        label = self._label(result['cell_id'])
        return (
            f"Removed inplace=True for '{result['variable']}' in cell {label} [{result['cell_id']}]\n"
            f"Methods fixed: {', '.join(result['methods_fixed'])}\n"
            f"New source:\n{result['new_source']}"
        )

    def insert_deepcopy(self, cell: str, variable: str) -> str:
        """Insert copy.deepcopy() at the top of a cell and rename downstream."""
        cell_id = self._resolve_ref(cell)
        result = self.session.insert_deepcopy(cell_id, variable)
        label = self._label(result['cell_id'])
        downstream = result.get("modified_downstream", [])
        ds_labels = [f"{self._label(c)} [{c}]" for c in downstream] if downstream else []
        return (
            f"Inserted deepcopy: {result['variable']} \u2192 {result['new_name']} in cell {label} [{result['cell_id']}]\n"
            f"Downstream cells renamed: {', '.join(ds_labels) if ds_labels else 'none'}"
        )

    def mark_diagnostic(self, cell: str) -> str:
        """Add %diagnostic magic to exclude a cell from tracking."""
        cell_id = self._resolve_ref(cell)
        result = self.session.mark_diagnostic(cell_id)
        label = self._label(cell_id)
        if result.get("already_diagnostic"):
            return f"Cell {label} [{cell_id}] is already marked as diagnostic."
        return f"Marked cell {label} [{cell_id}] as diagnostic.\nPreview: {result['new_source_preview']}"

    def merge_cells(self, cell_ids: List[str]) -> str:
        """Merge multiple cells into the first one."""
        resolved = [self._resolve_ref(c) for c in cell_ids]
        result = self.session.merge_cells(resolved)
        merged_label = self._label(result['merged_cell_id'])
        removed_labels = [f"{self._label(c)} [{c}]" for c in result['cells_removed']]
        return (
            f"Merged into cell {merged_label} [{result['merged_cell_id']}]\n"
            f"Removed cells: {', '.join(removed_labels)}\n"
            f"New source preview:\n{result['new_source_preview']}"
        )

    def move_cell(self, cell: str, after_cell: str) -> str:
        """Move a cell to after another cell."""
        cell_id = self._resolve_ref(cell)
        after_id = self._resolve_ref(after_cell)
        result = self.session.move_cell(cell_id, after_id)
        moved_label = self._label(result['cell_id'])
        after_label = self._label(result['moved_after'])
        order_labels = [self._label(c) for c in result['new_cell_order']]
        return (
            f"Moved cell {moved_label} [{result['cell_id']}] to after {after_label} [{result['moved_after']}]\n"
            f"New cell order: {', '.join(order_labels)}"
        )

    # ------------------------------------------------------------------
    # Checkpoint tools
    # ------------------------------------------------------------------

    def checkpoint(self) -> str:
        """Create a snapshot of the current notebook state."""
        ckpt_id = self.session.checkpoint()
        return f"Checkpoint created: {ckpt_id}"

    def restore(self, checkpoint_id: str) -> str:
        """Restore notebook to a previous checkpoint."""
        result = self.session.restore(checkpoint_id)
        changed = ", ".join(result["changed_cells"]) or "none"
        return f"Restored {result['cells_restored']} cells (stale: {changed})"

    def list_checkpoints(self) -> str:
        """List all saved checkpoints."""
        ckpts = self.session.list_checkpoints()
        if not ckpts:
            return "No checkpoints saved."
        lines = [f"  {c['checkpoint_id']} (cells: {c['cell_count']})" for c in ckpts]
        return "Checkpoints:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Lifecycle tools
    # ------------------------------------------------------------------

    def save_notebook(self, path: str = "") -> str:
        """Save the notebook to disk."""
        save_path = path if path else None
        saved = self.session.save(save_path)
        return f"Saved: {saved}"

    def get_notebook_path(self) -> str:
        """Return the currently loaded notebook's file path."""
        if not self.session.is_loaded:
            return "No notebook is loaded."
        return self.session.notebook_path

    # ------------------------------------------------------------------
    # Log tools
    # ------------------------------------------------------------------

    def get_log(self) -> str:
        """Return the full session event log as JSON."""
        import json
        events = self.session.get_event_log()
        if not events:
            return "No events logged yet."
        return json.dumps(events, indent=2, default=str)

    def save_log(self, path: str = "") -> str:
        """Save the session event log to a file."""
        save_path = path if path else None
        saved = self.session.save_event_log(save_path)
        return f"Log saved: {saved} ({len(self.session.get_event_log())} events)"

    def print_log(self) -> str:
        """Print the session event log in a human-readable format."""
        events = self.session.get_event_log()
        if not events:
            return "No events logged yet."

        lines = [f"{len(events)} events:"]
        for e in events:
            seq = e.get("seq", "?")
            elapsed = e.get("elapsed_s", 0)
            dur_ms = e.get("duration_ms", 0)
            tool = e.get("tool", "?")
            err = e.get("error")

            if err:
                result_str = f"ERROR: {err[:60]}"
            else:
                raw = e.get("result", "")
                if isinstance(raw, str):
                    first = next((l.strip() for l in raw.split("\n") if l.strip()), "")
                    result_str = first[:60]
                else:
                    result_str = str(raw)[:60]

            prefix = "!" if err else " "
            lines.append(f"{prefix}{seq} {elapsed:>5.1f}s {dur_ms:>5.0f}ms {tool} \u2192 {result_str}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cell_status(self, cell_id: str) -> str:
        """Get display status for a cell: ok, stale, error, or —."""
        if cell_id in self.session._stale_cells:
            return "stale"
        if self.session.cell_status.get(cell_id) == "error":
            return "error"
        if cell_id in self.session.executed_cells:
            return "ok"
        return "\u2014"
