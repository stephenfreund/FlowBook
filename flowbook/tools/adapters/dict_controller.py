"""`NotebookController` over an in-memory notebook dict.

This is the controller for the in-product "fix it" feature: the server already
has the notebook JSON in the request body and applies a fix to it statelessly
(no kernel, no Contents API). It also backs the unit tests, since it needs no
kernel or JupyterLab.

The controller records a *mutation log* — the pre/post source of every cell it
changes, which cells it removed, and whether order changed — so the caller can
build the `ApplyFixResponse` the frontend uses for its surgical undo.
"""

from __future__ import annotations

from typing import Any, Dict, List

from flowbook.scripts.fix_repro_errors import get_cell_source, set_cell_source
from flowbook.tools.controller import CellNotFoundError
from flowbook.util.cell_ids import next_insertion_id


class DictController:
    """Controller backed by a notebook dict; tracks a mutation log."""

    def __init__(self, notebook: Dict[str, Any], actor: str = "ai") -> None:
        self.notebook = notebook
        self.actor = actor
        # Mutation log (source of truth for ApplyFixResponse).
        self.pre_sources: Dict[str, str] = {}   # cell_id -> source before first change
        self.post_sources: Dict[str, str] = {}  # cell_id -> source after change
        self.removed: List[str] = []
        self.order_changed: bool = False

    # --- helpers ---------------------------------------------------------
    def _code_cells(self) -> List[Dict[str, Any]]:
        return [
            c for c in self.notebook.get("cells", [])
            if c.get("cell_type") == "code"
        ]

    def _find(self, cell_id: str) -> Dict[str, Any]:
        for c in self._code_cells():
            if c.get("id") == cell_id:
                return c
        raise CellNotFoundError(f"Cell '{cell_id}' not found in notebook")

    def _stash_pre(self, cell_id: str, cell: Dict[str, Any]) -> None:
        if cell_id not in self.pre_sources:
            self.pre_sources[cell_id] = get_cell_source(cell)

    # --- NotebookController ----------------------------------------------
    def cell_order(self) -> List[str]:
        return [c.get("id", "") for c in self._code_cells()]

    def read_source(self, cell_id: str) -> str:
        return get_cell_source(self._find(cell_id))

    def write_source(self, cell_id: str, source: str) -> None:
        cell = self._find(cell_id)
        self._stash_pre(cell_id, cell)
        set_cell_source(cell, source)
        self.post_sources[cell_id] = source

    def delete_cell(self, cell_id: str) -> None:
        cell = self._find(cell_id)
        self._stash_pre(cell_id, cell)
        self.notebook["cells"] = [
            c for c in self.notebook.get("cells", []) if c.get("id") != cell_id
        ]
        self.post_sources.pop(cell_id, None)
        self.removed.append(cell_id)
        self.order_changed = True

    def insert_after(
        self, after_cell_id: str, source: str, cell_type: str = "code"
    ) -> str:
        cells = self.notebook.get("cells", [])
        idx = next(
            (i for i, c in enumerate(cells) if c.get("id") == after_cell_id), None
        )
        if idx is None:
            raise CellNotFoundError(f"after_cell_id '{after_cell_id}' not found")
        existing = {c.get("id", "") for c in cells}
        new_id = next_insertion_id(after_cell_id, existing)
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
        self.post_sources[new_id] = source
        self.order_changed = True
        return new_id

    def move_after(self, cell_id: str, after_cell_id: str) -> None:
        cells = self.notebook.get("cells", [])
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
            cells.insert(src_idx, cell)  # put it back
            raise CellNotFoundError(f"after_cell_id '{after_cell_id}' not found")
        cells.insert(dst_idx + 1, cell)
        self.order_changed = True
