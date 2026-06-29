"""`NotebookController` over a live, kernel-connected MCP `NotebookSession`.

This is the controller for the MCP server (and the CLI): edits go to the
session's in-memory notebook, are marked stale + pushed to JupyterLab via the
Contents API, and structural changes are reported to the kernel.

To keep the `flowbook.tools` layer free of any dependency on `flowbook.mcp`,
this controller is duck-typed: it calls a small set of methods/attrs the
session already exposes (`get_cell_order`, `_find_cell`, `_mark_cell_edited`,
`_put_contents_api`, `_notify_structure`, `notebook`, and the tracking sets).
Side effects are *batched*: handlers call read/write many times, then the
session wrapper calls `flush()` once.
"""

from __future__ import annotations

from typing import Any, List

from flowbook.scripts.fix_repro_errors import get_cell_source, set_cell_source
from flowbook.tools.controller import CellNotFoundError


class KernelController:
    """Controller backed by an MCP NotebookSession."""

    def __init__(self, session: Any, actor: str = "ai") -> None:
        self.session = session
        self.actor = actor
        self.dirty = False
        self.structure_changed = False

    # --- helpers ---------------------------------------------------------
    def _cell(self, cell_id: str):
        try:
            _, cell = self.session._find_cell(cell_id)
        except ValueError as exc:
            raise CellNotFoundError(str(exc))
        return cell

    # --- NotebookController ----------------------------------------------
    def cell_order(self) -> List[str]:
        return self.session.get_cell_order()

    def read_source(self, cell_id: str) -> str:
        return get_cell_source(self._cell(cell_id))

    def write_source(self, cell_id: str, source: str) -> None:
        set_cell_source(self._cell(cell_id), source)
        self.session._mark_cell_edited(cell_id)
        self.dirty = True

    def delete_cell(self, cell_id: str) -> None:
        # Validate presence first (raises CellNotFoundError if absent).
        self._cell(cell_id)
        self.session.notebook["cells"] = [
            c for c in self.session.notebook["cells"] if c.get("id") != cell_id
        ]
        # Drop the removed cell from all per-cell tracking.
        self.session.executed_cells.discard(cell_id)
        self.session.cell_flowbook_meta.pop(cell_id, None)
        self.session.cell_status.pop(cell_id, None)
        self.session._stale_cells.discard(cell_id)
        self.dirty = True
        self.structure_changed = True

    def insert_after(
        self, after_cell_id: str, source: str, cell_type: str = "code"
    ) -> str:
        cells = self.session.notebook["cells"]
        idx = next(
            (i for i, c in enumerate(cells) if c.get("id") == after_cell_id), None
        )
        if idx is None:
            raise CellNotFoundError(f"after_cell_id '{after_cell_id}' not found")
        new_id = self.session._next_insert_id(after_cell_id)
        new_cell = {
            "cell_type": cell_type,
            "id": new_id,
            "source": source,
            "metadata": {},
        }
        if cell_type == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None
        cells.insert(idx + 1, new_cell)
        self.dirty = True
        self.structure_changed = True
        return new_id

    def move_after(self, cell_id: str, after_cell_id: str) -> None:
        cells = self.session.notebook["cells"]
        src_idx = next(
            (i for i, c in enumerate(cells) if c.get("id") == cell_id), None
        )
        if src_idx is None:
            raise CellNotFoundError(f"Cell '{cell_id}' not found in notebook")
        cell = cells.pop(src_idx)
        dst_idx = next(
            (i for i, c in enumerate(cells) if c.get("id") == after_cell_id), None
        )
        if dst_idx is None:
            cells.insert(src_idx, cell)  # restore
            raise CellNotFoundError(f"Destination cell not found: {after_cell_id}")
        cells.insert(dst_idx + 1, cell)
        self.dirty = True
        self.structure_changed = True

    # --- batched side effects -------------------------------------------
    def flush(self) -> None:
        """Apply batched side effects after a handler runs."""
        if self.structure_changed:
            self.session._notify_structure()
        if self.dirty:
            self.session._put_contents_api()
