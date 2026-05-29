"""Stateless dispatcher that applies a FixSuggestion to a notebook dict.

This is the in-product "fix it" entry point: it takes a notebook JSON, applies
a named refactoring to it, and returns an `ApplyFixResponse` the handler ships
back to the frontend (which updates the Y.js document and offers undo).

The actual transformations live in the unified tool layer
(`flowbook.tools`): this module just runs the registry handler over a
`DictController` and turns that controller's mutation log into the wire
response. The handler raises `ToolError` (a `ValueError`) on any failure, which
the HTTP handler turns into a 4xx with the original notebook unchanged.
"""

from __future__ import annotations

from typing import Any, Dict

from flowbook.server.fix_models import ApplyFixResponse, FixToolName
from flowbook.tools import get
from flowbook.tools.adapters.dict_controller import DictController


def apply_fix(
    notebook: Dict[str, Any], tool: FixToolName, args: Dict[str, Any]
) -> ApplyFixResponse:
    """Apply the named tool to ``notebook`` in place; return what changed.

    Raises ``ValueError`` (``flowbook.tools.ToolError`` or "Unknown tool: ...")
    on failure, leaving the notebook in a partially-applied state only if the
    underlying handler did — the same contract as before the cutover.
    """
    handler = get(tool).handler  # raises ValueError for an unknown tool
    ctrl = DictController(notebook)
    handler(ctrl, **(args or {}))  # raises ToolError(ValueError) on bad input

    return ApplyFixResponse(
        ok=True,
        tool=tool,
        args=args,
        modified_cells=list(ctrl.post_sources.keys()),
        pre_fix_sources=dict(ctrl.pre_sources),
        post_fix_sources=dict(ctrl.post_sources),
        cells_removed=list(ctrl.removed),
        new_cell_order=ctrl.cell_order() if ctrl.order_changed else None,
    )
