"""Mutator tools for the AI custom-fix path.

These functions are exposed to the LLM ONLY when the user explicitly invokes
"Other Fix…" and provides a natural-language instruction. Each call mutates
the notebook dict in place (the same dict the handler holds) and records a
MutationEntry in the supplied MutationLog so the handler can summarize what
changed and the frontend can drive Undo correctly.

Safety constraints, hard-coded:
  - A cell_id referenced by a mutator must exist (caller validates with
    _require_existing_cell).
  - Code-cell edits must yield Python that ast.parse can accept; otherwise
    the call is rejected (the LLM can retry in the next turn).
  - There is NO per-invocation mutation cap — the suggester loop terminates
    when the LLM stops calling tools.
  - Mutator tools never touch the kernel, never run code, never read or
    write files. They operate purely on the notebook dict.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from flowbook.server.fix_dispatcher import (
    _mark_diagnostic as _disp_mark_diagnostic,
    _merge_cells as _disp_merge_cells,
    _move_cell as _disp_move_cell,
)
from flowbook.scripts.fix_repro_errors import get_cell_source, set_cell_source
from flowbook.util.cell_ids import next_insertion_id


class MutatorError(ValueError):
    """Raised when a mutator call has bad args or fails validation."""


# ---------------------------------------------------------------------------
# Mutation log
# ---------------------------------------------------------------------------

@dataclass
class MutationEntry:
    """One record of a successful mutator call.

    The handler accumulates these into a MutationLog and uses them to build
    the CustomFixResponse (sources changed, cells added/removed, new order).
    """

    tool: str
    args: Dict[str, Any]
    summary: str
    # Per-event diffs the response needs:
    modified_cells: List[str] = field(default_factory=list)
    cells_added: List[str] = field(default_factory=list)
    cells_removed: List[str] = field(default_factory=list)


@dataclass
class MutationLog:
    """Aggregator of every mutator call within one custom-fix invocation."""

    entries: List[MutationEntry] = field(default_factory=list)
    # Pre-fix source snapshots, captured the first time a cell is mutated
    # so that even if the same cell is mutated twice we keep the original.
    pre_fix_sources: Dict[str, str] = field(default_factory=dict)
    # Pre-fix metadata snapshots, same reason.
    pre_fix_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record(self, entry: MutationEntry) -> None:
        self.entries.append(entry)

    def snapshot_if_first(self, cell: Dict[str, Any]) -> None:
        """Remember a cell's pre-fix source + metadata the first time we touch it."""
        cid = cell.get("id")
        if not cid or cid in self.pre_fix_sources:
            return
        self.pre_fix_sources[cid] = get_cell_source(cell)
        meta = cell.get("metadata") or {}
        # Shallow copy is enough — we only diff against the snapshot, never mutate it.
        self.pre_fix_metadata[cid] = dict(meta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_existing_cell(notebook: Dict[str, Any], cell_id: str) -> Dict[str, Any]:
    for c in notebook.get("cells", []):
        if c.get("id") == cell_id:
            return c
    raise MutatorError(f"No cell with id '{cell_id}'")


def _check_python_parses(source: str, cell_id: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise MutatorError(
            f"new_source for cell '{cell_id}' is not valid Python: "
            f"{type(e).__name__} at line {e.lineno}: {e.msg}"
        )


def _existing_ids(notebook: Dict[str, Any]) -> set:
    return {c.get("id") for c in notebook.get("cells", []) if c.get("id")}


# ---------------------------------------------------------------------------
# Mutator implementations
# ---------------------------------------------------------------------------

def edit_cell_source(
    notebook: Dict[str, Any], log: MutationLog, cell_id: str, new_source: str
) -> Dict[str, Any]:
    """Replace the source of an existing cell. Code cells must still parse."""
    cell = _require_existing_cell(notebook, cell_id)
    kind = cell.get("cell_type")
    if kind not in ("code", "markdown"):
        raise MutatorError(
            f"Cannot edit cell '{cell_id}' of type '{kind}'; only code and markdown are allowed."
        )
    if kind == "code":
        _check_python_parses(new_source, cell_id)
    log.snapshot_if_first(cell)
    set_cell_source(cell, new_source)
    log.record(
        MutationEntry(
            tool="edit_cell_source",
            args={"cell_id": cell_id, "new_source": new_source},
            summary=f"Rewrote {kind} cell {cell_id}",
            modified_cells=[cell_id],
        )
    )
    return {"cell_id": cell_id, "kind": kind, "new_source": new_source}


def insert_cell_after(
    notebook: Dict[str, Any],
    log: MutationLog,
    after_cell_id: str,
    source: str,
    kind: str = "code",
) -> Dict[str, Any]:
    """Insert a new cell immediately after the named cell.

    Returns the new cell's id so the LLM can refer to it in subsequent calls.
    """
    if kind not in ("code", "markdown"):
        raise MutatorError(f"kind must be 'code' or 'markdown', got '{kind}'")
    cells = notebook.get("cells", [])
    src_idx = None
    for i, c in enumerate(cells):
        if c.get("id") == after_cell_id:
            src_idx = i
            break
    if src_idx is None:
        raise MutatorError(f"after_cell_id '{after_cell_id}' not found")
    if kind == "code":
        _check_python_parses(source, "<new>")
    new_id = next_insertion_id(after_cell_id, _existing_ids(notebook))
    new_cell: Dict[str, Any] = {
        "cell_type": kind,
        "id": new_id,
        "source": source,
        "metadata": {},
    }
    if kind == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    cells.insert(src_idx + 1, new_cell)
    log.record(
        MutationEntry(
            tool="insert_cell_after",
            args={"after_cell_id": after_cell_id, "source": source, "kind": kind},
            summary=f"Inserted {kind} cell {new_id} after {after_cell_id}",
            cells_added=[new_id],
        )
    )
    return {"new_cell_id": new_id, "after_cell_id": after_cell_id, "kind": kind}


def delete_cell(
    notebook: Dict[str, Any], log: MutationLog, cell_id: str
) -> Dict[str, Any]:
    """Remove a cell from the notebook."""
    cells = notebook.get("cells", [])
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            log.snapshot_if_first(c)
            cells.pop(i)
            log.record(
                MutationEntry(
                    tool="delete_cell",
                    args={"cell_id": cell_id},
                    summary=f"Deleted cell {cell_id}",
                    cells_removed=[cell_id],
                )
            )
            return {"cell_id": cell_id, "removed": True}
    raise MutatorError(f"No cell with id '{cell_id}'")


def merge_cells(
    notebook: Dict[str, Any], log: MutationLog, cell_ids: List[str]
) -> Dict[str, Any]:
    """Combine adjacent code cells into the first one. Wraps the dispatcher's
    _merge_cells so the behaviour matches the built-in fix tool exactly."""
    for cid in cell_ids:
        _require_existing_cell(notebook, cid)
        log.snapshot_if_first(_require_existing_cell(notebook, cid))
    response = _disp_merge_cells(notebook, cell_ids)
    log.record(
        MutationEntry(
            tool="merge_cells",
            args={"cell_ids": cell_ids},
            summary=f"Merged {cell_ids} into {response.modified_cells[0] if response.modified_cells else cell_ids[0]}",
            modified_cells=list(response.modified_cells),
            cells_removed=list(response.cells_removed or []),
        )
    )
    return {
        "merged_into": response.modified_cells[0] if response.modified_cells else cell_ids[0],
        "cells_removed": response.cells_removed,
    }


def move_cell(
    notebook: Dict[str, Any],
    log: MutationLog,
    cell_id: str,
    after_cell_id: str,
) -> Dict[str, Any]:
    """Reorder a cell to sit immediately after another. Same semantics as the
    built-in move_cell fix tool."""
    _require_existing_cell(notebook, cell_id)
    _require_existing_cell(notebook, after_cell_id)
    response = _disp_move_cell(notebook, cell_id, after_cell_id)
    log.record(
        MutationEntry(
            tool="move_cell",
            args={"cell_id": cell_id, "after_cell_id": after_cell_id},
            summary=f"Moved {cell_id} to after {after_cell_id}",
        )
    )
    return {"cell_id": cell_id, "after_cell_id": after_cell_id, "new_order": response.new_cell_order}


def mark_diagnostic(
    notebook: Dict[str, Any], log: MutationLog, cell_id: str
) -> Dict[str, Any]:
    """Prefix the cell with %diagnostic. Same semantics as the built-in fix tool."""
    cell = _require_existing_cell(notebook, cell_id)
    log.snapshot_if_first(cell)
    try:
        response = _disp_mark_diagnostic(notebook, cell_id)
    except ValueError as e:
        # Already diagnostic, or other dispatch error.
        raise MutatorError(str(e))
    log.record(
        MutationEntry(
            tool="mark_diagnostic",
            args={"cell_id": cell_id},
            summary=f"Marked cell {cell_id} as %diagnostic",
            modified_cells=list(response.modified_cells),
        )
    )
    return {"cell_id": cell_id, "marked": True}


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / litellm function-calling shape)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "edit_cell_source",
            "description": (
                "Replace the source of an existing cell. Code cells must "
                "still parse as Python after the edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string"},
                    "new_source": {"type": "string"},
                },
                "required": ["cell_id", "new_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_cell_after",
            "description": (
                "Insert a new cell immediately after another. Returns "
                "{new_cell_id} so subsequent calls can reference it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "after_cell_id": {"type": "string"},
                    "source": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["code", "markdown"],
                        "default": "code",
                    },
                },
                "required": ["after_cell_id", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_cell",
            "description": "Remove a cell from the notebook.",
            "parameters": {
                "type": "object",
                "properties": {"cell_id": {"type": "string"}},
                "required": ["cell_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_cells",
            "description": (
                "Combine multiple code cells into the first one. The "
                "second+ cells are removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["cell_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_cell",
            "description": "Reorder a cell to sit immediately after another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string"},
                    "after_cell_id": {"type": "string"},
                },
                "required": ["cell_id", "after_cell_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_diagnostic",
            "description": (
                "Prepend %diagnostic to a code cell so FlowBook treats it as "
                "a read-only inspection cell."
            ),
            "parameters": {
                "type": "object",
                "properties": {"cell_id": {"type": "string"}},
                "required": ["cell_id"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH: Dict[str, Callable[..., Any]] = {
    "edit_cell_source": edit_cell_source,
    "insert_cell_after": insert_cell_after,
    "delete_cell": delete_cell,
    "merge_cells": merge_cells,
    "move_cell": move_cell,
    "mark_diagnostic": mark_diagnostic,
}


def dispatch(
    notebook: Dict[str, Any],
    log: MutationLog,
    tool_name: str,
    args: Dict[str, Any],
) -> Any:
    handler = _DISPATCH.get(tool_name)
    if handler is None:
        raise MutatorError(
            f"Unknown mutator tool '{tool_name}'. Available: {sorted(_DISPATCH)}"
        )
    try:
        return handler(notebook, log, **(args or {}))
    except MutatorError:
        raise
    except TypeError as e:
        raise MutatorError(f"Bad args for '{tool_name}': {e}")


def tool_names() -> List[str]:
    return list(_DISPATCH.keys())
