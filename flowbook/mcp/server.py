"""
FlowBook MCP Server — exposes notebook reproducibility analysis as MCP tools.

Tools delegate to FlowBookTools (flowbook.tools.tools) for shared logic.
MCP-only tools (load/close/get_notebook_path) call NotebookSession directly.

Every tool invocation is recorded in a JSON event log (auto-saved to
{notebook_stem}-mcp-log.json on close).
"""

import inspect
import os
import time as _time
from contextlib import asynccontextmanager
from functools import wraps

from mcp.server.fastmcp import FastMCP, Context

from flowbook.mcp.session import NotebookSession
from flowbook.tools.tools import FlowBookTools


def _get_session(ctx: Context) -> NotebookSession:
    """Extract the NotebookSession from the MCP lifespan context."""
    return ctx.request_context.lifespan_context["session"]


def _get_tools(ctx: Context) -> FlowBookTools:
    """Get the FlowBookTools instance from the MCP lifespan context."""
    return ctx.request_context.lifespan_context["tools"]


def _logged_tool(fn):
    """Decorator that logs every tool call to the session event log."""
    sig = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        ctx = bound.arguments.get("ctx")
        tool_args = {k: v for k, v in bound.arguments.items() if k != "ctx"}

        safe_args = {}
        for k, v in tool_args.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                safe_args[k] = v
            elif isinstance(v, list):
                safe_args[k] = v
            else:
                safe_args[k] = str(v)

        from flowbook.util.output import log as _log

        arg_str = ", ".join(f"{k}={v!r}" for k, v in safe_args.items())
        _log(f"[MCP] {fn.__name__}({arg_str})")

        t0 = _time.time()
        error_str = None
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            result = f"ERROR: {error_str}"
            return result
        finally:
            duration_ms = (_time.time() - t0) * 1000
            # Log to mcp.log
            preview = str(result)[:200] if result else ""
            if error_str:
                _log(f"[MCP]   ERROR {duration_ms:.0f}ms: {error_str}")
            else:
                _log(f"[MCP]   -> {duration_ms:.0f}ms: {preview}")
            # Log to session event log
            session = _get_session(ctx) if ctx else None
            if session:
                session.log_event(
                    tool=fn.__name__,
                    args=safe_args,
                    result=result,
                    duration_ms=duration_ms,
                    error=error_str,
                )

    return wrapper


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage a single NotebookSession + FlowBookTools for the server's lifetime."""
    session = NotebookSession()
    tools = FlowBookTools(session)
    try:
        yield {"session": session, "tools": tools}
    finally:
        session.close()


mcp = FastMCP(
    "flowbook",
    instructions=(
        "FlowBook MCP server for Jupyter notebook reproducibility analysis. "
        "Load a notebook, run cells, inspect violations, apply fixes, and save."
    ),
    lifespan=lifespan,
)


# =====================================================================
# MCP-only lifecycle tools
# =====================================================================


@mcp.tool()
@_logged_tool
def load_notebook(path: str, ctx: Context) -> str:
    """Load a notebook from disk and start a FlowBook kernel.

    The notebook is normalized (4-char cell IDs) and a FlowBook kernel is
    started with violation reporting enabled. Any previously loaded notebook
    is closed first.

    Args:
        path: Path to a .ipynb file (absolute or relative to cwd).
    """
    session = _get_session(ctx)
    abs_path = os.path.abspath(path)
    result = session.load(abs_path)
    ids = ", ".join(result["cell_ids"])
    joined = " [joined existing kernel]" if result.get("joined_existing") else ""
    live = " [live sync]" if result.get("contents_api_connected") else ""
    return f"Loaded {result['code_cells']} code cells ({result['total_cells']} total){joined}{live}: {ids}"


@mcp.tool()
@_logged_tool
def close_notebook(ctx: Context) -> str:
    """Close the current notebook and shutdown the kernel."""
    session = _get_session(ctx)
    if not session.is_loaded:
        return "No notebook is loaded."
    path = session.notebook_path
    session.close()
    return f"Closed: {path}"


@mcp.tool()
@_logged_tool
def get_notebook_path(ctx: Context) -> str:
    """Return the currently loaded notebook's file path."""
    return _get_tools(ctx).get_notebook_path()


# =====================================================================
# Shared tools — delegate to FlowBookTools
# =====================================================================


@mcp.tool()
@_logged_tool
def continue_after_violation(enabled: bool, ctx: Context) -> str:
    """Configure whether violations reject execution or just report.

    When True: violations are reported but execution continues and the
    cell stays CLEAN.  Use this for analysis runs where you want a full
    picture of all violations without halting.

    When False (default after load): violations cause rollback — the cell
    is rejected and the namespace is restored to pre-execution state.

    Args:
        enabled: True to continue after violations, False to reject.
    """
    return _get_tools(ctx).continue_after_violation(enabled)


@mcp.tool()
@_logged_tool
def list_cells(ctx: Context) -> str:
    """List all cells with index, ID, type, first line, and status.

    Status is one of: unexecuted, ok, error, stale.
    """
    return _get_tools(ctx).list_cells()


@mcp.tool()
@_logged_tool
def read_cell(cell: str, ctx: Context) -> str:
    """Read cell source, outputs, and flowbook metadata.

    If cell is provided, reads that single cell. If cell is empty,
    returns all code cells with @-labels, status, and source — much
    cheaper than calling read_cell for each cell individually.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID). Empty string for all cells.
    """
    return _get_tools(ctx).read_cell(cell)


@mcp.tool()
@_logged_tool
def get_next_actionable_cell(ctx: Context) -> str:
    """Get the first cell that needs attention.

    Priority: runtime error > reproducibility violation > stale > unexecuted.
    Returns the cell's source, ID, and reason it needs attention.
    Returns "All clean." if every cell is executed and clean.
    """
    return _get_tools(ctx).get_next_actionable_cell()


@mcp.tool()
@_logged_tool
def edit_cell_source(cell: str, new_source: str, ctx: Context) -> str:
    """Replace a cell's source code.

    If the cell was previously executed, it is automatically marked stale
    in the FlowBook kernel.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
        new_source: The new source code for the cell.
    """
    return _get_tools(ctx).edit_cell_source(cell, new_source)


@mcp.tool()
@_logged_tool
def add_cell(source: str, ctx: Context, cell_type: str = "code",
             after_cell: str = "") -> str:
    """Add a new cell to the notebook.

    Args:
        source: Source code (or markdown) for the new cell.
        cell_type: "code" or "markdown".
        after_cell: Insert after this cell (optional; appends if empty).
    """
    return _get_tools(ctx).add_cell(source, cell_type, after_cell or None)


@mcp.tool()
@_logged_tool
def delete_cell(cell: str, ctx: Context) -> str:
    """Remove a cell from the notebook.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
    """
    return _get_tools(ctx).delete_cell(cell)


@mcp.tool()
@_logged_tool
def run_cell(cell: str, ctx: Context) -> str:
    """Execute a single cell and return outputs + flowbook metadata.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
    """
    return _get_tools(ctx).run_cell(cell)


@mcp.tool()
@_logged_tool
def run_all_cells(ctx: Context) -> str:
    """Execute all code cells top-to-bottom with reproducibility tracking.

    Stops on the first runtime error. Returns violation and stale counts.
    """
    return _get_tools(ctx).run_all_cells()


@mcp.tool()
@_logged_tool
def run_from(cell: str, ctx: Context) -> str:
    """Run from a cell through the end of the notebook, stopping on error.

    Skips clean cells. Stops on the first runtime error or rejected violation.

    Args:
        cell: Cell reference to start from (@A notation or 4-char cell ID).
    """
    return _get_tools(ctx).run_from(cell)


@mcp.tool()
@_logged_tool
def get_flowbook_metadata(cell: str, ctx: Context) -> str:
    """Return reproducibility metadata for a specific cell.

    Shows read/write locations, errors, staleness, and timing information
    from the cell's most recent execution.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
    """
    return _get_tools(ctx).get_flowbook_metadata(cell)


@mcp.tool()
@_logged_tool
def run_actionable_cell(ctx: Context) -> str:
    """Find and run the next actionable cell.

    Finds the first cell needing attention (error > violation > stale >
    unexecuted), runs it, and returns the result. Returns "All clean" if
    no cells need attention.
    """
    return _get_tools(ctx).run_actionable_cell()


@mcp.tool()
@_logged_tool
def run_actionable_cells(ctx: Context) -> str:
    """Run all actionable cells in sequence until the notebook is clean.

    Stops on hard errors always. Stops on violations if
    continue_after_violation is False.
    """
    return _get_tools(ctx).run_actionable_cells()


@mcp.tool()
@_logged_tool
def get_status(ctx: Context) -> str:
    """Get the notebook's current reproducibility status.

    Shows: which cells are stale (with reasons), current violations,
    execution counts. Does not run anything.
    """
    return _get_tools(ctx).get_status()


@mcp.tool()
@_logged_tool
def save_notebook(ctx: Context, path: str = "") -> str:
    """Save the notebook to disk.

    Args:
        path: Output path. If empty, saves to the original path.
    """
    return _get_tools(ctx).save_notebook(path)


@mcp.tool()
@_logged_tool
def checkpoint(ctx: Context) -> str:
    """Create a snapshot of the current notebook state.

    Captures all cell sources so you can restore later if a fix attempt
    makes things worse. Returns a checkpoint ID to use with restore().
    """
    return _get_tools(ctx).checkpoint()


@mcp.tool()
@_logged_tool
def restore(checkpoint_id: str, ctx: Context) -> str:
    """Restore the notebook to a previous checkpoint.

    Reverts cell sources to the snapshot without restarting the kernel.
    Changed cells are marked stale so they can be re-run incrementally.

    Args:
        checkpoint_id: The checkpoint ID returned by checkpoint().
    """
    return _get_tools(ctx).restore(checkpoint_id)


@mcp.tool()
@_logged_tool
def list_checkpoints(ctx: Context) -> str:
    """List all saved checkpoints."""
    return _get_tools(ctx).list_checkpoints()


# =====================================================================
# Refactoring tools
# =====================================================================


@mcp.tool()
@_logged_tool
def alpha_rename(cell: str, old_name: str, new_name: str, ctx: Context) -> str:
    """Rename a variable from a cell onwards using AST-based transformation.

    Renames all occurrences of old_name to new_name in the target cell
    and every code cell after it. Uses AST parsing for reliability.

    Args:
        cell: Cell reference where renaming starts (@A notation or 4-char cell ID).
        old_name: Current variable name to rename.
        new_name: New variable name.
    """
    return _get_tools(ctx).alpha_rename(cell, old_name, new_name)


@mcp.tool()
@_logged_tool
def remove_inplace(cell: str, variable: str, ctx: Context) -> str:
    """Convert df.method(inplace=True) to df = df.method() in a cell.

    Fixes UNRECOVERABLE_MUTATION violations from pandas inplace operations.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
        variable: The DataFrame variable name (e.g., "df").
    """
    return _get_tools(ctx).remove_inplace(cell, variable)


@mcp.tool()
@_logged_tool
def insert_deepcopy(cell: str, variable: str, ctx: Context) -> str:
    """Insert a deepcopy of a variable at the top of a cell and rename downstream.

    Inserts `from copy import deepcopy; {var}_copy = deepcopy({var})` and renames
    all uses in the target cell and downstream cells.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
        variable: Variable name to copy.
    """
    return _get_tools(ctx).insert_deepcopy(cell, variable)


@mcp.tool()
@_logged_tool
def mark_diagnostic(cell: str, ctx: Context) -> str:
    """Add %diagnostic magic to a cell to exclude it from tracking.

    Use for inspection/visualization cells that don't affect computation.

    Args:
        cell: Cell reference (@A notation or 4-char cell ID).
    """
    return _get_tools(ctx).mark_diagnostic(cell)


@mcp.tool()
@_logged_tool
def merge_cells(cell_ids: list[str], ctx: Context) -> str:
    """Merge multiple cells into the first one.

    Concatenates sources and removes the rest from the notebook.

    Args:
        cell_ids: List of cell references to merge (@A notation or 4-char cell IDs).
    """
    return _get_tools(ctx).merge_cells(cell_ids)


@mcp.tool()
@_logged_tool
def move_cell(cell: str, after_cell: str, ctx: Context) -> str:
    """Move a cell to after another cell in the notebook.

    Args:
        cell: Cell reference to move (@A notation or 4-char cell ID).
        after_cell: Cell reference to place it after.
    """
    return _get_tools(ctx).move_cell(cell, after_cell)


# =====================================================================
# Log tools
# =====================================================================


@mcp.tool()
@_logged_tool
def get_log(ctx: Context) -> str:
    """Return the full session event log as JSON."""
    return _get_tools(ctx).get_log()


@mcp.tool()
@_logged_tool
def save_log(ctx: Context, path: str = "") -> str:
    """Save the session event log to a JSON file.

    Args:
        path: Output path. If empty, saves to {notebook_stem}-mcp-log.json.
    """
    return _get_tools(ctx).save_log(path)


@mcp.tool()
@_logged_tool
def print_log(ctx: Context) -> str:
    """Print the session event log in a human-readable format."""
    return _get_tools(ctx).print_log()


# =====================================================================
# Entry point
# =====================================================================


def main():
    """Entry point for the flowbook_mcp console script."""
    import os
    import sys

    # MCP uses STDIO for JSON protocol — any writes to stdout corrupt the stream.
    # Redirect FlowBook's output module to a log file so kernel/timer messages
    # are preserved for debugging. Tail with: tail -f mcp.log
    log_file = open("mcp.log", "a")

    from flowbook.util.output import output
    output._get_output_file = lambda: log_file

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
