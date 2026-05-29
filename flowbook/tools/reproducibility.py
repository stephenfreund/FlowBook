"""The six reproducibility refactoring tools, defined once.

Each handler takes a `NotebookController` and keyword args, mutates the notebook
through the controller, and returns a normalized result dict. Handlers raise
`ToolError` when a fix has no effect or an argument is invalid; transport
adapters translate that into their own error channel.

The AST work is delegated to `flowbook.scripts.fix_repro_errors`, which remains
the single home of the transformation algorithms. This module is the single
home of the *orchestration* (the "from this cell onward" loops) that used to be
copy-pasted across `mcp/session.py`, `server/fix_dispatcher.py`, and
`nbi/tools.py`.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from flowbook.scripts.fix_repro_errors import (
    InplaceRemover,
    find_actual_variable_name,
    prepend_to_cell_source,
    rename_variable_in_code,
    split_cell_magic,
)
from flowbook.tools.controller import (
    CellNotFoundError,
    NoEffectError,
    NotebookController,
    ToolError,
)


def alpha_rename(
    ctrl: NotebookController, *, cell_id: str, old_name: str, new_name: str
) -> Dict[str, Any]:
    """Rename ``old_name`` to ``new_name`` from ``cell_id`` onward (AST-based)."""
    order = ctrl.cell_order()
    if cell_id not in order:
        raise CellNotFoundError(
            f"Cell '{cell_id}' not found in notebook. "
            f"Available code cells: {order}"
        )

    modified: List[str] = []
    for cid in order[order.index(cell_id):]:
        new_src, renamed = rename_variable_in_code(
            ctrl.read_source(cid), old_name, new_name
        )
        if renamed:
            ctrl.write_source(cid, new_src)
            modified.append(cid)

    if not modified:
        raise NoEffectError(
            f"alpha_rename had no effect: '{old_name}' not found from "
            f"cell {cell_id} onwards"
        )
    return {
        "modified_cells": modified,
        "total_modified": len(modified),
        "old_name": old_name,
        "new_name": new_name,
    }


def remove_inplace(
    ctrl: NotebookController, *, cell_id: str, variable: str
) -> Dict[str, Any]:
    """Convert ``df.method(inplace=True)`` to ``df = df.method()`` in one cell."""
    source = ctrl.read_source(cell_id)
    actual_var = find_actual_variable_name(source, variable)

    new_source = None
    methods_fixed: List[str] = []
    try:
        tree = ast.parse(source)
        remover = InplaceRemover(actual_var)
        new_tree = remover.visit(tree)
        if remover.modified:
            new_source = ast.unparse(new_tree)
            methods_fixed = list(remover.method_calls_fixed)
    except SyntaxError:
        pass

    if new_source is None:
        # Regex fallback for code that doesn't parse cleanly.
        pattern = (
            rf"(\b{re.escape(actual_var)}\.(\w+)\([^)]*),"
            rf"\s*inplace\s*=\s*True([^)]*)\)"
        )
        candidate, count = re.subn(pattern, rf"{actual_var} = \1\3)", source)
        if count > 0:
            new_source = candidate
            methods_fixed = ["(regex fallback)"]

    if new_source is None or new_source == source:
        raise NoEffectError(
            f"remove_inplace had no effect: no inplace=True call on "
            f"'{actual_var}' found in cell {cell_id}"
        )

    ctrl.write_source(cell_id, new_source)
    return {
        "cell_id": cell_id,
        "variable": actual_var,
        "methods_fixed": methods_fixed,
        "new_source": new_source,
    }


def insert_deepcopy(
    ctrl: NotebookController, *, cell_id: str, variable: str
) -> Dict[str, Any]:
    """Deep-copy ``variable`` at the top of ``cell_id`` and rename downstream.

    The copy is named ``{var}_{cell_id}`` to avoid collisions across cells.
    """
    order = ctrl.cell_order()
    if cell_id not in order:
        raise CellNotFoundError(
            f"Cell '{cell_id}' not found in notebook. "
            f"Available code cells: {order}"
        )

    source = ctrl.read_source(cell_id)
    actual_var = find_actual_variable_name(source, variable)
    new_name = f"{actual_var}_{cell_id}"
    copy_line = f"import copy; {new_name} = copy.deepcopy({actual_var})\n"

    magic_prefix, rest = split_cell_magic(source)
    renamed_rest, _ = rename_variable_in_code(rest, actual_var, new_name)
    ctrl.write_source(cell_id, magic_prefix + copy_line + renamed_rest)

    modified: List[str] = [cell_id]
    downstream: List[str] = []
    for cid in order[order.index(cell_id) + 1:]:
        new_src, renamed = rename_variable_in_code(
            ctrl.read_source(cid), actual_var, new_name
        )
        if renamed:
            ctrl.write_source(cid, new_src)
            modified.append(cid)
            downstream.append(cid)

    return {
        "cell_id": cell_id,
        "variable": actual_var,
        "new_name": new_name,
        "modified_cells": modified,
        "modified_downstream": downstream,
    }


def mark_diagnostic(ctrl: NotebookController, *, cell_id: str) -> Dict[str, Any]:
    """Prepend ``%diagnostic`` so the cell is excluded from tracking."""
    source = ctrl.read_source(cell_id)
    if source.lstrip().startswith("%diagnostic"):
        raise NoEffectError(f"Cell {cell_id} is already marked %diagnostic")

    new_source = prepend_to_cell_source(source, "%diagnostic\n")
    ctrl.write_source(cell_id, new_source)
    return {"cell_id": cell_id, "new_source": new_source}


def merge_cells(
    ctrl: NotebookController, *, cell_ids: List[str]
) -> Dict[str, Any]:
    """Merge ``cell_ids`` into the first; remove the rest."""
    if not isinstance(cell_ids, list) or len(cell_ids) < 2:
        raise ToolError("merge_cells requires at least 2 cell ids")
    # `ToolError` (not NoEffectError) — too few ids is a bad arg, not a no-op.

    sources = [ctrl.read_source(cid) for cid in cell_ids]  # validates existence
    merged = "\n\n".join(s for s in sources if s.strip())

    first = cell_ids[0]
    ctrl.write_source(first, merged)
    for cid in cell_ids[1:]:
        ctrl.delete_cell(cid)

    return {
        "merged_cell_id": first,
        "cells_removed": list(cell_ids[1:]),
        "new_source": merged,
        "new_cell_order": ctrl.cell_order(),
    }


def move_cell(
    ctrl: NotebookController, *, cell_id: str, after_cell_id: str
) -> Dict[str, Any]:
    """Reorder ``cell_id`` to directly after ``after_cell_id`` (no source change)."""
    ctrl.move_after(cell_id, after_cell_id)
    return {
        "cell_id": cell_id,
        "after_cell_id": after_cell_id,
        "new_cell_order": ctrl.cell_order(),
    }
