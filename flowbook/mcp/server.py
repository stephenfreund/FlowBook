"""
FlowBook MCP Server — exposes notebook reproducibility analysis as MCP tools.

23 tools: 14 core (load/close, list/get/edit cells, run, status, save,
checkpoint/restore, get_next_actionable) + 6 algorithmic refactoring
(alpha_rename, remove_inplace, insert_deepcopy, mark_diagnostic,
merge_cells, move_cell) + 3 log tools (get_log, save_log, print_log).

Every tool invocation is recorded in a JSON event log (auto-saved to
{notebook_stem}-mcp-log.json on close).
"""

import inspect
import json
import os
import time as _time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

from flowbook.mcp.session import (
    NotebookSession,
    format_error,
    format_flowbook_meta,
    format_loc,
    format_outputs_text,
)


def _get_session(ctx: Context) -> NotebookSession:
    """Extract the NotebookSession from the MCP lifespan context."""
    return ctx.request_context.lifespan_context["session"]


def _logged_tool(fn):
    """Decorator that logs every tool call to the session event log.

    Captures: tool name, arguments (excluding ctx), result text,
    duration, and any errors. The log is JSON-serializable.
    """
    sig = inspect.signature(fn)
    param_names = [
        p for p in sig.parameters if p != "ctx"
    ]

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Bind arguments to figure out which is ctx vs tool args
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        ctx = bound.arguments.get("ctx")
        tool_args = {k: v for k, v in bound.arguments.items() if k != "ctx"}

        # Serialize args (handle lists, etc.)
        safe_args = {}
        for k, v in tool_args.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                safe_args[k] = v
            elif isinstance(v, list):
                safe_args[k] = v
            else:
                safe_args[k] = str(v)

        t0 = _time.time()
        error_str = None
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = (_time.time() - t0) * 1000
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
    """Manage a single NotebookSession for the server's lifetime."""
    session = NotebookSession()
    try:
        yield {"session": session}
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
# Core tools (1-14)
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
    return f"Loaded {result['code_cells']} code cells ({result['total_cells']} total): {ids}"


@mcp.tool()
@_logged_tool
def continue_after_violation(enabled: bool, ctx: Context) -> str:
    """Configure whether violations reject execution or just report.

    When True: violations are reported but execution continues and the
    cell stays CLEAN.  Use this for analysis runs (/basic-run) where you
    want a full picture of all violations without halting.

    When False (default after load): violations cause rollback — the cell
    is rejected and the namespace is restored to pre-execution state.
    Use this for fix runs (/fix-notebook) where you need a clean namespace.

    Args:
        enabled: True to continue after violations, False to reject.
    """
    session = _get_session(ctx)
    session.set_continue_after_violation(enabled)
    mode = "continue (report only)" if enabled else "reject (rollback)"
    return f"Violation mode: {mode}"


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
def list_cells(ctx: Context) -> str:
    """List all cells with index, ID, type, first line, and status.

    Status is one of: unexecuted, ok, error, stale.
    """
    session = _get_session(ctx)
    session._require_loaded()
    lines = []
    code_idx = 0
    for cell in session.notebook["cells"]:
        cid = cell.get("id", "?")
        ctype = cell.get("cell_type", "?")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        first_line = source.split("\n")[0][:60] if source.strip() else "(empty)"

        if ctype == "code":
            if cid in session._stale_cells:
                status = "stale"
            elif session.cell_status.get(cid) == "error":
                status = "error"
            elif cid in session.executed_cells:
                status = "ok"
            else:
                status = "—"

            fb_meta = session.cell_flowbook_meta.get(cid, {})
            viol = " !" if fb_meta.get("errors") else ""

            lines.append(f"[{code_idx}] {cid} {status}{viol}: {first_line}")
            code_idx += 1
        else:
            lines.append(f"[ ] {cid} ({ctype}): {first_line}")

    return "\n".join(lines)


@mcp.tool()
@_logged_tool
def get_cell(cell_id: str, ctx: Context) -> str:
    """Get a cell's full source code, outputs, and flowbook metadata.

    Args:
        cell_id: The 4-character cell ID.
    """
    session = _get_session(ctx)
    result = session.get_cell(cell_id)

    line = f"{result['cell_id']} ({result.get('status', '?')})"
    line += f"\n>>> {result['source']}"

    output_text = result.get("outputs_text", "").strip()
    if output_text:
        preview = output_text[:300]
        if len(output_text) > 300:
            preview += "..."
        line += f"\nOutput: {preview}"

    if "flowbook" in result:
        line += f"\n{result['flowbook']}"

    return line


@mcp.tool()
@_logged_tool
def get_next_actionable_cell(ctx: Context) -> str:
    """Get the first cell that needs attention.

    Priority: runtime error > reproducibility violation > stale > unexecuted.
    Returns the cell's source, ID, and reason it needs attention.
    Returns "all clean" if every cell is executed and clean.
    """
    session = _get_session(ctx)
    result = session.get_next_actionable()
    if result is None:
        return "All clean."

    line = f"{result['cell_id']}: {result['reason']}"
    if "violation_summary" in result:
        line += f" — {result['violation_summary']}"
    line += f"\n>>> {result['source']}"

    return line


@mcp.tool()
@_logged_tool
def edit_cell(cell_id: str, new_source: str, ctx: Context) -> str:
    """Replace a cell's source code.

    If the cell was previously executed, it is automatically marked stale
    in the FlowBook kernel (matching the frontend's edit detection behavior).

    Args:
        cell_id: The 4-character cell ID.
        new_source: The new source code for the cell.
    """
    session = _get_session(ctx)
    result = session.edit_cell(cell_id, new_source)
    stale_note = " (marked stale)" if result["marked_stale"] else ""
    return (
        f"Updated cell {result['cell_id']}{stale_note}\n"
        f"New source preview: {result['new_source_preview']}"
    )


@mcp.tool()
@_logged_tool
def run_cell(cell_id: str, ctx: Context = None) -> str:
    """Execute a single cell and return outputs + flowbook metadata.

    Args:
        cell_id: The cell ID to execute.
    """
    session = _get_session(ctx)
    result = session.run_cell(cell_id)

    line = f"{result['cell_id']}: {result['status']}"
    if result.get("error_message"):
        line += f" — {result['error_message']}"

    output_text = result.get("outputs_text", "").strip()
    if output_text:
        preview = output_text[:200]
        if len(output_text) > 200:
            preview += "..."
        line += f"\nOutput: {preview}"

    if "flowbook" in result:
        line += f"\n{result['flowbook']}"

    return line


@mcp.tool()
@_logged_tool
def run_all_cells(ctx: Context) -> str:
    """Execute all code cells top-to-bottom with reproducibility tracking.

    Returns per-cell results with flowbook metadata, an aggregate violation
    list, and the final stale cell list. Stops on the first runtime error.
    """
    session = _get_session(ctx)
    result = session.run_all()
    n = result["total_executed"]
    total = result["total_code_cells"]
    violations = result["violations"]
    stale = result["stale_cells"]
    status = result["status"]

    line = f"Executed {n}/{total} code cells"
    if status == "error":
        line += " (stopped on error)"
    line += f" | {len(violations)} violations | {len(stale)} stale"

    if violations:
        for e in violations:
            etype = e.get("error_type", "?")
            cid = e.get("cell_id", "?")
            locs = e.get("locations", [])
            loc_str = ", ".join(format_loc(l) for l in locs) if locs else ""
            line += f"\n  {cid}: {etype}"
            if loc_str:
                line += f" [{loc_str}]"

    return line


@mcp.tool()
@_logged_tool
def run_from(cell_id: str, ctx: Context = None) -> str:
    """Run from a cell through the end of the notebook, stopping on error.

    Executes cell_id and all subsequent non-empty code cells in order.
    Stops on the first runtime error or rejected violation.

    Args:
        cell_id: Cell ID to start from.
    """
    session = _get_session(ctx)
    result = session.run_from(cell_id)
    n = len(result["executed"])
    v = len(result["violations"])
    s = result["stale_remaining"]
    sk = result["skipped"]
    line = f"Ran {n} cells from {cell_id}"
    if sk:
        line += f" ({sk} clean skipped)"
    if result["error_cell"]:
        line += f" | error at {result['error_cell']}"
    line += f" | {v} violations | {s} stale"
    if result["violations"]:
        for e in result["violations"]:
            etype = e.get("error_type", "?")
            cid = e.get("cell_id", "?")
            locs = e.get("locations", [])
            loc_str = ", ".join(format_loc(l) for l in locs) if locs else ""
            line += f"\n  {cid}: {etype}"
            if loc_str:
                line += f" [{loc_str}]"
    return line


@mcp.tool()
@_logged_tool
def get_status(ctx: Context) -> str:
    """Get the notebook's current reproducibility status.

    Shows: which cells are stale (with reasons), current violations,
    execution counts. Does not run anything — reads from accumulated state.
    """
    session = _get_session(ctx)
    result = session.get_status()
    n_exec = result["executed"]
    n_total = result["total_code_cells"]
    violations = result["violations"]
    stale = result["stale_cells"]

    line = f"{n_exec}/{n_total} executed | {len(violations)} violations | {len(stale)} stale"

    if violations:
        for v in violations:
            etype = v.get("error_type", "?")
            cid = v.get("cell_id", "?")
            locs = v.get("locations", [])
            loc_str = ", ".join(format_loc(l) for l in locs) if locs else ""
            line += f"\n  {cid}: {etype}"
            if loc_str:
                line += f" [{loc_str}]"

    if stale:
        for cid, reasons in stale.items():
            reason_strs = []
            for r in reasons:
                rtype = r.get("type", "?")
                loc = r.get("loc", "")
                s = rtype
                if loc:
                    s += f": {loc}"
                reason_strs.append(s)
            line += f"\n  {cid} stale: {', '.join(reason_strs)}"

    return line


@mcp.tool()
@_logged_tool
def save_notebook(path: str = "", ctx: Context = None) -> str:
    """Save the notebook to disk.

    Args:
        path: Output path. If empty, saves to the original path.
    """
    session = _get_session(ctx)
    save_path = path if path else None
    saved = session.save(save_path)
    return f"Saved: {saved}"


@mcp.tool()
@_logged_tool
def get_notebook_path(ctx: Context) -> str:
    """Return the currently loaded notebook's file path."""
    session = _get_session(ctx)
    if not session.is_loaded:
        return "No notebook is loaded."
    return session.notebook_path


@mcp.tool()
@_logged_tool
def checkpoint(ctx: Context) -> str:
    """Create a snapshot of the current notebook state.

    Captures all cell sources so you can restore later if a fix attempt
    makes things worse. Returns a checkpoint ID to use with restore().
    """
    session = _get_session(ctx)
    ckpt_id = session.checkpoint()
    return f"Checkpoint created: {ckpt_id}"


@mcp.tool()
@_logged_tool
def restore(checkpoint_id: str, ctx: Context) -> str:
    """Restore the notebook to a previous checkpoint.

    Reverts cell sources to the snapshot without restarting the kernel.
    Changed cells are marked stale so they can be re-run incrementally.

    Args:
        checkpoint_id: The checkpoint ID returned by checkpoint().
    """
    session = _get_session(ctx)
    result = session.restore(checkpoint_id)
    changed = ", ".join(result["changed_cells"]) or "none"
    return f"Restored {result['cells_restored']} cells (stale: {changed})"


@mcp.tool()
@_logged_tool
def list_checkpoints(ctx: Context) -> str:
    """List all saved checkpoints."""
    session = _get_session(ctx)
    ckpts = session.list_checkpoints()
    if not ckpts:
        return "No checkpoints saved."
    lines = []
    for c in ckpts:
        lines.append(f"  {c['checkpoint_id']} (cells: {c['cell_count']})")
    return "Checkpoints:\n" + "\n".join(lines)


# =====================================================================
# Algorithmic refactoring tools (15-20)
# =====================================================================


@mcp.tool()
@_logged_tool
def alpha_rename(cell_id: str, old_name: str, new_name: str, ctx: Context) -> str:
    """Rename a variable from a cell onwards using AST-based transformation.

    This is the primary fix for backward mutations and variable reuse.
    Renames all occurrences of old_name to new_name in the target cell
    and every code cell after it in the notebook. Uses AST parsing for
    reliability (won't miss references in comprehensions, f-strings, etc.).

    Args:
        cell_id: Cell ID where renaming starts.
        old_name: Current variable name to rename.
        new_name: New variable name.
    """
    session = _get_session(ctx)
    result = session.alpha_rename(cell_id, old_name, new_name)
    if result["total_modified"] == 0:
        return f"No occurrences of '{old_name}' found from cell {cell_id} onwards."
    return (
        f"Renamed '{result['old_name']}' → '{result['new_name']}'\n"
        f"Modified {result['total_modified']} cells: {', '.join(result['modified_cells'])}"
    )


@mcp.tool()
@_logged_tool
def remove_inplace(cell_id: str, variable: str, ctx: Context) -> str:
    """Convert df.method(inplace=True) to df = df.method() in a cell.

    This fixes UNRECOVERABLE_MUTATION violations caused by pandas inplace
    operations. Uses AST transformation with regex fallback.

    Args:
        cell_id: Cell ID containing the inplace operation.
        variable: The DataFrame variable name (e.g., "df").
    """
    session = _get_session(ctx)
    result = session.remove_inplace(cell_id, variable)
    if "error" in result:
        return f"Error: {result['error']}"
    return (
        f"Removed inplace=True for '{result['variable']}' in cell {result['cell_id']}\n"
        f"Methods fixed: {', '.join(result['methods_fixed'])}\n"
        f"New source:\n{result['new_source']}"
    )


@mcp.tool()
@_logged_tool
def insert_deepcopy(cell_id: str, variable: str, ctx: Context) -> str:
    """Insert a deepcopy of a variable at the top of a cell and rename downstream.

    Inserts `import copy; {var}_copy = copy.deepcopy({var})` and renames
    all uses of the variable in the target cell and all downstream cells.
    Useful for in-place reassignment and sequential transformation chains.

    Args:
        cell_id: Cell ID where the copy should be inserted.
        variable: Variable name to copy.
    """
    session = _get_session(ctx)
    result = session.insert_deepcopy(cell_id, variable)
    downstream = result.get("modified_downstream", [])
    return (
        f"Inserted deepcopy: {result['variable']} → {result['new_name']} in cell {result['cell_id']}\n"
        f"Downstream cells renamed: {', '.join(downstream) if downstream else 'none'}"
    )


@mcp.tool()
@_logged_tool
def mark_diagnostic(cell_id: str, ctx: Context) -> str:
    """Add %diagnostic magic to a cell to exclude it from tracking.

    Use this for inspection/visualization cells (e.g., df.info(), df.head())
    that read variables but don't contribute to computation. This prevents
    them from creating false backward mutation violations.

    Args:
        cell_id: Cell ID to mark as diagnostic.
    """
    session = _get_session(ctx)
    result = session.mark_diagnostic(cell_id)
    if result.get("already_diagnostic"):
        return f"Cell {cell_id} is already marked as diagnostic."
    return f"Marked cell {cell_id} as diagnostic.\nPreview: {result['new_source_preview']}"


@mcp.tool()
@_logged_tool
def merge_cells(cell_ids: list[str], ctx: Context) -> str:
    """Merge multiple cells into the first one.

    Concatenates the source code of all specified cells into the first cell
    and removes the rest from the notebook. Useful for consolidating
    tightly coupled transformation steps that form a sequential chain.

    Args:
        cell_ids: List of cell IDs to merge (in notebook order).
    """
    session = _get_session(ctx)
    result = session.merge_cells(cell_ids)
    return (
        f"Merged into cell {result['merged_cell_id']}\n"
        f"Removed cells: {', '.join(result['cells_removed'])}\n"
        f"New source preview:\n{result['new_source_preview']}"
    )


@mcp.tool()
@_logged_tool
def move_cell(cell_id: str, after_cell_id: str, ctx: Context) -> str:
    """Move a cell to after another cell in the notebook.

    Useful for moving diagnostic/inspection cells after transformations
    to resolve backward mutation violations.

    Args:
        cell_id: Cell ID to move.
        after_cell_id: Cell ID to place it after.
    """
    session = _get_session(ctx)
    result = session.move_cell(cell_id, after_cell_id)
    return (
        f"Moved cell {result['cell_id']} to after {result['moved_after']}\n"
        f"New cell order: {', '.join(result['new_cell_order'])}"
    )


# =====================================================================
# Log tool (21)
# =====================================================================


@mcp.tool()
@_logged_tool
def get_log(ctx: Context) -> str:
    """Return the full session event log as JSON.

    Every tool call is recorded with: sequence number, timestamp, elapsed
    time, tool name, arguments, result (truncated), and duration. The log
    is also auto-saved to {notebook}-mcp-log.json when the session closes.
    """
    session = _get_session(ctx)
    events = session.get_event_log()
    if not events:
        return "No events logged yet."
    return json.dumps(events, indent=2, default=str)


@mcp.tool()
@_logged_tool
def save_log(path: str = "", ctx: Context = None) -> str:
    """Save the session event log to a JSON file.

    Args:
        path: Output path. If empty, saves to {notebook_stem}-mcp-log.json.
    """
    session = _get_session(ctx)
    save_path = path if path else None
    saved = session.save_event_log(save_path)
    return f"Log saved: {saved} ({len(session.get_event_log())} events)"


@mcp.tool()
@_logged_tool
def print_log(ctx: Context) -> str:
    """Print the session event log in a human-readable format.

    Shows a compact timeline of every tool call with elapsed time, duration,
    tool name, and a short result summary.
    """
    session = _get_session(ctx)
    events = session.get_event_log()
    if not events:
        return "No events logged yet."

    lines = [f"{len(events)} events:"]

    for e in events:
        seq = e.get("seq", "?")
        elapsed = e.get("elapsed_s", 0)
        dur_ms = e.get("duration_ms", 0)
        tool = e.get("tool", "?")
        err = e.get("error")

        if err:
            result_str = f"ERROR: {err[:60]}"
        else:
            raw = e.get("result", "")
            if isinstance(raw, str):
                first = next((l.strip() for l in raw.split("\n") if l.strip()), "")
                result_str = first[:60]
            else:
                result_str = str(raw)[:60]

        prefix = "!" if err else " "
        lines.append(f"{prefix}{seq} {elapsed:>5.1f}s {dur_ms:>5.0f}ms {tool} → {result_str}")

    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================


def main():
    """Entry point for the flowbook_mcp console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
