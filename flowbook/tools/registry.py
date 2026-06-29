"""The single declarative catalog of FlowBook notebook tools.

Each `Tool` pairs a name + JSON-Schema parameters + one-line description with a
handler written against `NotebookController`. Transports (MCP, NBI, the fix
agent) iterate `REGISTRY` to generate their tool surfaces and build LLM prompts,
so the catalog cannot drift between them.

This initial registry covers the six refactoring tools that were previously
triplicated. Inspection/execution/structure tools are added in later phases
(see `LLM_INTEGRATION_DESIGN.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from flowbook.tools import reproducibility as _r


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for the tool's args
    handler: Callable[..., Dict[str, Any]]
    category: str = "refactor"
    mutates: bool = True
    requires_execution: bool = False
    aliases: tuple = field(default=())


def _schema(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


_CELL_ID = {"type": "string", "description": "Target code-cell id."}


REGISTRY: List[Tool] = [
    Tool(
        name="alpha_rename",
        description=(
            "Rename a variable from a cell onward (AST-based). Fixes variable "
            "reuse across cells: renames every occurrence in the target cell "
            "and all later code cells."
        ),
        parameters=_schema(
            {
                "cell_id": _CELL_ID,
                "old_name": {"type": "string", "description": "Current name."},
                "new_name": {"type": "string", "description": "Replacement name."},
            },
            ["cell_id", "old_name", "new_name"],
        ),
        handler=_r.alpha_rename,
    ),
    Tool(
        name="remove_inplace",
        description=(
            "Convert df.method(inplace=True) into df = df.method(). Fixes "
            "UNRECOVERABLE_MUTATION from pandas in-place operations."
        ),
        parameters=_schema(
            {
                "cell_id": _CELL_ID,
                "variable": {"type": "string", "description": "e.g. 'df'."},
            },
            ["cell_id", "variable"],
        ),
        handler=_r.remove_inplace,
    ),
    Tool(
        name="insert_deepcopy",
        description=(
            "Insert copy.deepcopy() of a variable at the top of a cell and "
            "rename it downstream, breaking an aliasing/backward-mutation chain."
        ),
        parameters=_schema(
            {
                "cell_id": _CELL_ID,
                "variable": {"type": "string", "description": "Variable to copy."},
            },
            ["cell_id", "variable"],
        ),
        handler=_r.insert_deepcopy,
    ),
    Tool(
        name="mark_diagnostic",
        description=(
            "Prepend %diagnostic so a pure-inspection cell (df.info(), plots) "
            "is excluded from reproducibility tracking."
        ),
        parameters=_schema({"cell_id": _CELL_ID}, ["cell_id"]),
        handler=_r.mark_diagnostic,
    ),
    Tool(
        name="merge_cells",
        description=(
            "Merge consecutive code cells into the first, removing the rest. "
            "Use to keep a definition and its mutation in one atomic cell."
        ),
        parameters=_schema(
            {
                "cell_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Cell ids to merge; first is kept.",
                }
            },
            ["cell_ids"],
        ),
        handler=_r.merge_cells,
    ),
    Tool(
        name="move_cell",
        description=(
            "Reorder a cell to directly after another (no source change). Use "
            "to move a diagnostic below the mutation it inspects."
        ),
        parameters=_schema(
            {
                "cell_id": _CELL_ID,
                "after_cell_id": {
                    "type": "string",
                    "description": "Move target to just after this cell.",
                },
            },
            ["cell_id", "after_cell_id"],
        ),
        handler=_r.move_cell,
    ),
]


_BY_NAME: Dict[str, Tool] = {t.name: t for t in REGISTRY}


def get(name: str) -> Tool:
    """Return the Tool with this name, or raise ValueError for an unknown tool."""
    tool = _BY_NAME.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")
    return tool


def names() -> List[str]:
    return [t.name for t in REGISTRY]
