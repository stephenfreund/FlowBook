"""Stateless dispatcher that applies a FixSuggestion to a notebook dict.

This is the server-side counterpart of the MCP NotebookSession's fix methods,
but it does NOT manage a long-lived session, a kernel, or Contents API sync.
It takes a notebook JSON, transforms cell sources / order in place using the
AST helpers from flowbook.scripts.fix_repro_errors, and returns a description
of what changed. The handler is responsible for shipping the modified
notebook back to the frontend, which updates the Y.js document.

Each dispatcher function returns an ApplyFixResponse so the handler can
serialize it directly. On any failure, it raises ValueError — the handler
turns that into a 4xx with the original notebook unchanged.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Tuple

from flowbook.scripts.fix_repro_errors import (
    InplaceRemover,
    find_actual_variable_name,
    get_cell_source,
    prepend_to_cell_source,
    rename_variable_in_code,
    set_cell_source,
    split_cell_magic,
)
from flowbook.server.fix_models import ApplyFixResponse, FixToolName


def apply_fix(
    notebook: Dict[str, Any], tool: FixToolName, args: Dict[str, Any]
) -> ApplyFixResponse:
    """Dispatch to the named tool. Mutates `notebook` in place."""
    if tool == "alpha_rename":
        return _alpha_rename(notebook, **args)
    if tool == "remove_inplace":
        return _remove_inplace(notebook, **args)
    if tool == "insert_deepcopy":
        return _insert_deepcopy(notebook, **args)
    if tool == "mark_diagnostic":
        return _mark_diagnostic(notebook, **args)
    if tool == "merge_cells":
        return _merge_cells(notebook, **args)
    if tool == "move_cell":
        return _move_cell(notebook, **args)
    raise ValueError(f"Unknown tool: {tool}")


# ---------------------------------------------------------------------------
# Helpers (operating on notebook["cells"])
# ---------------------------------------------------------------------------

def _code_cells(notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]


def _code_cell_order(notebook: Dict[str, Any]) -> List[str]:
    return [c.get("id", "") for c in _code_cells(notebook)]


def _find_code_cell(notebook: Dict[str, Any], cell_id: str) -> Dict[str, Any]:
    for c in _code_cells(notebook):
        if c.get("id") == cell_id:
            return c
    raise ValueError(f"Cell '{cell_id}' not found in notebook")


def _snapshot_sources(
    notebook: Dict[str, Any], cell_ids: List[str]
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cid in cell_ids:
        try:
            cell = _find_code_cell(notebook, cid)
        except ValueError:
            continue
        out[cid] = get_cell_source(cell)
    return out


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _alpha_rename(
    notebook: Dict[str, Any], cell_id: str, old_name: str, new_name: str
) -> ApplyFixResponse:
    order = _code_cell_order(notebook)
    if cell_id not in order:
        raise ValueError(f"Cell '{cell_id}' not found in notebook")
    start = order.index(cell_id)
    targets = order[start:]

    pre = _snapshot_sources(notebook, targets)
    modified: List[str] = []
    post: Dict[str, str] = {}

    for cid in targets:
        cell = _find_code_cell(notebook, cid)
        src = get_cell_source(cell)
        new_src, renamed = rename_variable_in_code(src, old_name, new_name)
        if renamed:
            set_cell_source(cell, new_src)
            modified.append(cid)
            post[cid] = new_src

    if not modified:
        raise ValueError(
            f"alpha_rename had no effect: '{old_name}' not found from cell {cell_id} onwards"
        )

    return ApplyFixResponse(
        ok=True,
        tool="alpha_rename",
        args={"cell_id": cell_id, "old_name": old_name, "new_name": new_name},
        modified_cells=modified,
        pre_fix_sources={k: v for k, v in pre.items() if k in modified},
        post_fix_sources=post,
    )


def _remove_inplace(
    notebook: Dict[str, Any], cell_id: str, variable: str
) -> ApplyFixResponse:
    cell = _find_code_cell(notebook, cell_id)
    source = get_cell_source(cell)
    actual_var = find_actual_variable_name(source, variable)

    new_source = None
    try:
        tree = ast.parse(source)
        remover = InplaceRemover(actual_var)
        new_tree = remover.visit(tree)
        if remover.modified:
            new_source = ast.unparse(new_tree)
    except SyntaxError:
        pass

    if new_source is None:
        # Regex fallback for code that doesn't parse cleanly.
        pattern = (
            rf"(\b{re.escape(actual_var)}\.(\w+)\([^)]*),\s*inplace\s*=\s*True([^)]*)\)"
        )
        candidate, count = re.subn(pattern, rf"{actual_var} = \1\3)", source)
        if count > 0:
            new_source = candidate

    if new_source is None or new_source == source:
        raise ValueError(
            f"remove_inplace had no effect: no inplace=True call on '{actual_var}' "
            f"found in cell {cell_id}"
        )

    pre = {cell_id: source}
    set_cell_source(cell, new_source)
    return ApplyFixResponse(
        ok=True,
        tool="remove_inplace",
        args={"cell_id": cell_id, "variable": variable},
        modified_cells=[cell_id],
        pre_fix_sources=pre,
        post_fix_sources={cell_id: new_source},
    )


def _insert_deepcopy(
    notebook: Dict[str, Any], cell_id: str, variable: str
) -> ApplyFixResponse:
    cell = _find_code_cell(notebook, cell_id)
    source = get_cell_source(cell)
    actual_var = find_actual_variable_name(source, variable)
    # Use cell_id-derived suffix per convention in fix-notebook docs.
    new_name = f"{actual_var}_{cell_id}"
    copy_line = f"import copy; {new_name} = copy.deepcopy({actual_var})\n"

    magic_prefix, rest = split_cell_magic(source)
    renamed_rest, _ = rename_variable_in_code(rest, actual_var, new_name)
    new_source = magic_prefix + copy_line + renamed_rest

    order = _code_cell_order(notebook)
    if cell_id not in order:
        raise ValueError(f"Cell '{cell_id}' not found in notebook")
    start = order.index(cell_id)

    pre: Dict[str, str] = {cell_id: source}
    post: Dict[str, str] = {cell_id: new_source}
    modified: List[str] = [cell_id]

    set_cell_source(cell, new_source)

    for cid in order[start + 1:]:
        dcell = _find_code_cell(notebook, cid)
        dsrc = get_cell_source(dcell)
        dnew, renamed = rename_variable_in_code(dsrc, actual_var, new_name)
        if renamed:
            pre[cid] = dsrc
            set_cell_source(dcell, dnew)
            post[cid] = dnew
            modified.append(cid)

    return ApplyFixResponse(
        ok=True,
        tool="insert_deepcopy",
        args={"cell_id": cell_id, "variable": variable},
        modified_cells=modified,
        pre_fix_sources=pre,
        post_fix_sources=post,
    )


def _mark_diagnostic(notebook: Dict[str, Any], cell_id: str) -> ApplyFixResponse:
    cell = _find_code_cell(notebook, cell_id)
    source = get_cell_source(cell)
    if source.lstrip().startswith("%diagnostic"):
        raise ValueError(f"Cell {cell_id} is already marked %diagnostic")

    new_source = prepend_to_cell_source(source, "%diagnostic\n")
    set_cell_source(cell, new_source)
    return ApplyFixResponse(
        ok=True,
        tool="mark_diagnostic",
        args={"cell_id": cell_id},
        modified_cells=[cell_id],
        pre_fix_sources={cell_id: source},
        post_fix_sources={cell_id: new_source},
    )


def _merge_cells(
    notebook: Dict[str, Any], cell_ids: List[str]
) -> ApplyFixResponse:
    if len(cell_ids) < 2:
        raise ValueError("merge_cells requires at least 2 cell ids")

    cells_to_merge: List[Dict[str, Any]] = []
    for cid in cell_ids:
        cells_to_merge.append(_find_code_cell(notebook, cid))

    sources = [get_cell_source(c) for c in cells_to_merge]
    merged_source = "\n\n".join(s for s in sources if s.strip())

    first_id = cell_ids[0]
    pre = {cid: get_cell_source(_find_code_cell(notebook, cid)) for cid in cell_ids}

    # Update first cell's source.
    set_cell_source(cells_to_merge[0], merged_source)

    # Remove subsequent cells.
    ids_to_remove = set(cell_ids[1:])
    notebook["cells"] = [
        c for c in notebook.get("cells", []) if c.get("id") not in ids_to_remove
    ]

    return ApplyFixResponse(
        ok=True,
        tool="merge_cells",
        args={"cell_ids": cell_ids},
        modified_cells=[first_id],
        pre_fix_sources=pre,
        post_fix_sources={first_id: merged_source},
        cells_removed=cell_ids[1:],
        new_cell_order=_code_cell_order(notebook),
    )


def _move_cell(
    notebook: Dict[str, Any], cell_id: str, after_cell_id: str
) -> ApplyFixResponse:
    cells = notebook.get("cells", [])

    src_idx = None
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            src_idx = i
            break
    if src_idx is None:
        raise ValueError(f"Cell '{cell_id}' not found in notebook")

    cell_to_move = cells.pop(src_idx)

    dst_idx = None
    for i, c in enumerate(cells):
        if c.get("id") == after_cell_id:
            dst_idx = i + 1
            break
    if dst_idx is None:
        cells.insert(src_idx, cell_to_move)  # put it back
        raise ValueError(f"after_cell_id '{after_cell_id}' not found")

    cells.insert(dst_idx, cell_to_move)

    return ApplyFixResponse(
        ok=True,
        tool="move_cell",
        args={"cell_id": cell_id, "after_cell_id": after_cell_id},
        modified_cells=[],  # sources unchanged; only order
        pre_fix_sources={},
        post_fix_sources={},
        new_cell_order=_code_cell_order(notebook),
    )
