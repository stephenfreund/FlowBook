"""The abstract notebook interface every transport adapter implements.

A `NotebookController` is the least-common-denominator of what the MCP server
(kernel + Contents API), the NBI extension (frontend bridge), and the
in-product fix feature (in-memory dict) all already do. Tool handlers in this
package are written *only* against this interface, so the same handler runs
unchanged on every transport.

Controllers are keyed by **cell id** (4-char code-cell id). Index-based
transports (NBI) translate id<->index inside their adapter.

Transport-specific side effects that are not part of a tool's contract — pushing
to the Contents API, marking cells stale, notifying the kernel of structure
changes, opening a LogBook AI-attribution window — live *inside* the
controller's `write_source` / `delete_cell` / `move_after`, never in the
handler. That keeps handlers pure and identical across transports.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


class ToolError(ValueError):
    """A tool could not be applied (bad args, no effect, missing cell).

    Subclasses `ValueError` so existing server handlers that catch `ValueError`
    and return a 4xx keep working unchanged after the cutover.
    """


class CellNotFoundError(ToolError):
    """A referenced cell id is not in the notebook."""


class NoEffectError(ToolError):
    """The tool ran but changed nothing (no match / already applied).

    Adapters that historically *tolerated* a no-op (e.g. the MCP tools, which
    return a descriptive result rather than erroring) catch this specifically;
    the stateless fix dispatcher lets it propagate as a `ValueError`.
    """


@runtime_checkable
class NotebookController(Protocol):
    """Minimal notebook surface the refactoring/structure handlers need.

    `actor` identifies who is driving the controller — ``"ai"`` when an LLM is
    acting (MCP, NBI tool calls, the fix agent), ``"user"`` otherwise. Adapters
    use it to attribute LogBook events; handlers never read it.
    """

    actor: str

    def cell_order(self) -> List[str]:
        """Code-cell ids in execution order."""

    def read_source(self, cell_id: str) -> str:
        """Source of one code cell. Raises ToolError if the cell is absent."""

    def write_source(self, cell_id: str, source: str) -> None:
        """Replace a code cell's source. Raises ToolError if absent."""

    def delete_cell(self, cell_id: str) -> None:
        """Remove a cell. Raises ToolError if absent."""

    def insert_after(
        self, after_cell_id: str, source: str, cell_type: str = "code"
    ) -> str:
        """Insert a new ``cell_type`` ('code'|'markdown') cell directly after
        ``after_cell_id`` and return the new cell's id.

        ``after_cell_id`` may name any cell (code or markdown). Raises ToolError
        if it is absent.
        """

    def move_after(self, cell_id: str, after_cell_id: str) -> None:
        """Move ``cell_id`` to directly after ``after_cell_id``.

        Raises ToolError (leaving order unchanged) if either id is absent.
        """
