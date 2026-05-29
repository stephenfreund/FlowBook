"""Unified FlowBook notebook-tool layer.

A single declarative catalog of reproducibility tools, defined once over an
abstract `NotebookController`. Each transport (MCP server, NBI extension, the
in-product "fix it" feature) provides its own controller adapter and reuses
these handlers, so the tool *algorithm* lives in exactly one place.

See `LLM_INTEGRATION_DESIGN.md` for the full design and migration plan.
"""

from flowbook.tools.controller import NotebookController, ToolError
from flowbook.tools.registry import REGISTRY, Tool, get, names

__all__ = [
    "NotebookController",
    "ToolError",
    "REGISTRY",
    "Tool",
    "get",
    "names",
]
