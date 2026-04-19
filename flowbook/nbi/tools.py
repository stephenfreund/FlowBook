"""NBI extension tool implementations for FlowBook.

All tools use run_ui_command() for notebook I/O via FlowBook's frontend bridge
commands. Cell references use @A notation (code-cell-only indexing).

Tool names and parameter signatures match the MCP server (flowbook/mcp/server.py)
for API consistency. Shared formatting from flowbook/tools/format.py.
"""

import ast
import time
import logging

import notebook_intelligence.api as nbapi

from flowbook.nbi.cell_addressing import index_to_alpha, parse_cell_ref
from flowbook.nbi.session import FlowBookSession
from flowbook.tools.format import format_flowbook_meta
from flowbook.tools.mcp_content import (
    ToolContent,
    build_tool_content,
    to_markdown,
)
from flowbook.scripts.fix_repro_errors import (
    rename_variable_in_code,
    find_actual_variable_name,
    split_cell_magic,
    prepend_to_cell_source,
    InplaceRemover,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared session instance — set by create_tools()
# ---------------------------------------------------------------------------

_session: FlowBookSession = None


import functools
import traceback


def _safe_tool(fn):
    """Decorator that catches exceptions and returns error text instead of raising.

    Checks that the FlowBook kernel is active before running the tool.
    Ensures the LLM always gets a response, even on internal errors.
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            # Check FlowBook kernel is active before running any tool
            response = kwargs.get('response')
            if response and not getattr(response, '_flowbook_active', False):
                try:
                    active = await response.run_ui_command('flowbook:is-active', {})
                    is_active = active.get('active', False) if isinstance(active, dict) else bool(active)
                except Exception:
                    is_active = False
                if not is_active:
                    return (
                        "ERROR: FlowBook kernel is not active. "
                        "Switch the notebook kernel to 'flowbook_kernel' and try again."
                    )
            return await fn(*args, **kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("Tool %s failed: %s\n%s", fn.__name__, exc, tb)
            return f"ERROR in {fn.__name__}: {type(exc).__name__}: {exc}"
    return wrapper


def _logged(tool_name, args_dict, fn):
    """Decorator-like helper for event logging."""
    t0 = time.time()
    error_str = None
    result = None
    try:
        result = fn()
        return result
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = (time.time() - t0) * 1000
        if _session:
            _session.log_event(
                tool=tool_name,
                args=args_dict,
                result=str(result)[:500] if result else None,
                duration_ms=duration_ms,
                error=error_str,
            )


# ===================================================================
# Category 1: Metadata & Status
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def get_flowbook_metadata(cell: str, **args) -> str:
    """Get FlowBook reproducibility metadata for a code cell. Returns read/write locations, errors, timing, and staleness info.

    Args:
        cell: Cell reference in @A notation (code cells only, e.g., @A, @C, @AA)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    meta = await response.run_ui_command('flowbook:get-metadata', {"cellIndex": idx})
    label = index_to_alpha(idx)
    if not meta:
        return f"Cell {label} has not been executed yet — no metadata available."
    return f"{label}:\n{format_flowbook_meta(meta)}"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def get_next_actionable_cell(**args) -> str:
    """Get the next cell that needs attention. Priority: error > stale > unexecuted. Returns cell label and reason, or 'All clean.' if all cells are clean.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:get-next-actionable', {})
    if result.get('done', False):
        return "All clean."
    label = result.get('label', '?')
    reason = result.get('reason', '?')
    source = result.get('source', '')
    line = f"{label}: {reason}"
    if result.get('error'):
        line += f" — {result['error']}"
    if source:
        line += f"\n>>> {source}"
    return line


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def get_status(**args) -> str:
    """Get the notebook's current reproducibility status. Shows execution counts, violations, and stale cells.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:get-status', {})
    total = result.get('total_cells', 0)
    executed = result.get('executed', 0)
    stale = result.get('stale', 0)
    clean = result.get('clean', 0)
    violations = result.get('violations', 0)
    reproducible = result.get('reproducible', False)

    line = f"{executed}/{total} executed | {violations} violations | {stale} stale"
    if reproducible:
        line += " | reproducible ✓"
    return line


# ===================================================================
# Category 2: Cell Operations
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def read_cell(cell: str = "", **args) -> str:
    """Read cell source, outputs, and FlowBook metadata.

    If cell is provided, reads that single cell. If cell is empty,
    returns all code cells with @-labels, status, and source.

    Args:
        cell: Cell reference in @A notation (code cells only). Empty for all cells.
    """
    response = args["response"]
    if not cell:
        # Return all cells
        counts = await response.run_ui_command('flowbook:get-cell-count', {})
        num_code = counts['code_cells']
        if num_code == 0:
            return "No code cells in notebook."

        parts = []
        for i in range(num_code):
            cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": i})
            label = index_to_alpha(i)
            source = cell_data.get('source', '')
            ec = cell_data.get('execution_count')
            meta = cell_data.get('flowbook_meta') or {}
            errors = meta.get('errors', [])
            stale = meta.get('stale_cells', [])
            cell_id = cell_data.get('cell_id', '?')

            if errors:
                status = 'error'
            elif cell_id in stale:
                status = 'stale'
            elif ec is not None:
                status = 'ok'
            else:
                status = '\u2014'

            parts.append(f"\u2500\u2500 {label} [{cell_id}] ({status}) \u2500\u2500\n{source}")

        return "\n\n".join(parts)

    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    label = index_to_alpha(idx)
    cell_id = result.get('cell_id', '?')
    source = result.get('source', '')
    outputs = result.get('outputs_text', '').strip() if isinstance(result.get('outputs_text'), str) else ''

    line = f"{label} [{cell_id}]"
    line += f"\n>>> {source}"
    if outputs:
        preview = outputs[:300]
        if len(outputs) > 300:
            preview += "..."
        line += f"\nOutput: {preview}"
    meta = result.get('flowbook_meta')
    if meta:
        line += f"\n{format_flowbook_meta(meta)}"
    return line


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def edit_cell_source(cell: str, new_source: str, **args) -> str:
    """Edit a code cell's source. Uses identity-safe in-place modification that preserves cell ID and triggers FlowBook's edit detection.

    Args:
        cell: Cell reference in @A notation (code cells only)
        new_source: New source code for the cell
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": idx, "source": new_source})
    label = result.get('label', index_to_alpha(idx))
    cell_id = result.get('cell_id', '?')
    return f"Updated cell {label} [{cell_id}]"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def add_cell(source: str, cell_type: str = "code",
                   after_cell: str = "", **args) -> str:
    """Add a new cell to the notebook.

    Args:
        source: Source code (or markdown) for the new cell
        cell_type: "code" or "markdown"
        after_cell: Insert after this cell (optional; appends if empty)
    """
    response = args["response"]
    if after_cell:
        after_idx = parse_cell_ref(after_cell)
        if cell_type == "code":
            await response.run_ui_command('notebook-intelligence:add-code-cell-to-active-notebook', {"source": source})
        else:
            await response.run_ui_command('notebook-intelligence:add-markdown-cell-to-active-notebook', {"source": source})
    else:
        if cell_type == "code":
            await response.run_ui_command('notebook-intelligence:add-code-cell-to-active-notebook', {"source": source})
        else:
            await response.run_ui_command('notebook-intelligence:add-markdown-cell-to-active-notebook', {"source": source})
    await response.run_ui_command('flowbook:notify-structure', {})
    return f"Added {cell_type} cell"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def delete_cell(cell: str, **args) -> str:
    """Delete a cell from the notebook.

    Args:
        cell: Cell reference in @A notation (code cells only)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    # Get cell info before deleting
    cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    label = index_to_alpha(idx)
    await response.run_ui_command('notebook-intelligence:delete-cell-at-index', {"cellIndex": idx})
    await response.run_ui_command('flowbook:notify-structure', {})
    return f"Deleted cell {label}"


# ===================================================================
# Category 3: Execution
# ===================================================================


def _format_run_result(result: dict) -> str:
    """Format a bridge run-cell result dict into a readable string.

    Matches the MCP format: "@A [cell_id]: status — error_message\\nOutput: ...\\nFlowBook: ..."
    """
    label = result.get('label', '?')
    cell_id = result.get('cell_id', '?')
    status = result.get('status', '?')

    line = f"{label} [{cell_id}]: {status}"

    # Show error details from outputs
    outputs = result.get('outputs_text', '').strip()
    if status == 'error' and outputs:
        # For errors, show the traceback prominently
        preview = outputs[:500]
        if len(outputs) > 500:
            preview += "..."
        line += f"\n{preview}"
    elif outputs:
        preview = outputs[:200]
        if len(outputs) > 200:
            preview += "..."
        line += f"\nOutput: {preview}"

    # Show FlowBook violations
    errors = result.get('errors', [])
    if errors:
        for e in errors:
            etype = e.get('error_type', e.get('predicate', '?'))
            msg = e.get('message', '')
            locs = e.get('locations', [])
            line += f"\n  Violation: {etype}"
            if msg:
                line += f" — {msg}"
            if locs:
                line += f" [{', '.join(str(l) for l in locs)}]"

    return line


def _format_run_actionable_cells_result(result: dict) -> str:
    """Format the bridge run-actionable-cells result."""
    total = result.get('total_run', 0)
    cells = result.get('cells_run', [])
    violations = result.get('violations', [])
    with_errors = result.get('with_errors', [])
    done = result.get('done', False)

    line = f"Ran {total} cells"
    if with_errors and isinstance(with_errors, (list, tuple)):
        line += f" | errors in: {', '.join(str(e) for e in with_errors)}"
    elif with_errors:
        line += f" | {with_errors} errors"
    violations = violations if isinstance(violations, (list, tuple)) else []
    line += f" | {len(violations)} violations"

    if done and not violations and not with_errors:
        line += "\nAll clean!"
    elif violations:
        for v in violations:
            label = v.get('label', '?')
            etype = v.get('error_type', v.get('predicate', '?'))
            locs = v.get('locations', [])
            line += f"\n  {label}: {etype}"
            if locs:
                line += f" [{', '.join(str(l) for l in locs)}]"

    if cells and isinstance(cells, (list, tuple)):
        line += f"\nCells: {', '.join(str(c) for c in cells)}"

    return line


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def run_cell(cell: str, **args) -> str:
    """Execute a code cell and return outputs + FlowBook reproducibility metadata.

    Args:
        cell: Cell reference in @A notation (code cells only)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:run-cell', {"cellIndex": idx})
    return _format_run_result(result)


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def run_actionable_cell(**args) -> str:
    """Find and run the next actionable cell. Priority: error > violation > stale > unexecuted.
    """
    response = args["response"]
    actionable = await response.run_ui_command('flowbook:get-next-actionable', {})
    if actionable.get('done', False):
        return "All clean \u2014 no actionable cells."
    idx = actionable.get('index', 0)
    result = await response.run_ui_command('flowbook:run-cell', {"cellIndex": idx})
    return _format_run_result(result)


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def run_actionable_cells(**args) -> str:
    """Run all stale and unexecuted cells until the notebook is reproducible or an error occurs. Stops on hard errors always. Stops on violations if continue_after_violation is disabled.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:run-actionable-cells', {})
    return _format_run_actionable_cells_result(result)


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def run_all_cells(**args) -> str:
    """Execute all code cells top-to-bottom with reproducibility tracking. Stops on the first runtime error.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:run-actionable-cells', {})
    return _format_run_actionable_cells_result(result)


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def continue_after_violation(enabled: bool, **args) -> str:
    """Set whether to continue execution after a predicate violation (True = report only, False = reject and rollback).

    Args:
        enabled: True to continue after violations, False to reject
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:set-continue-after-violation', {"enabled": enabled})
    mode = "continue (report only)" if enabled else "reject (rollback)"
    return f"Violation mode: {mode}"


# ===================================================================
# Category 4: Source Refactoring Tools
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def alpha_rename(cell: str, old_name: str, new_name: str, **args) -> str:
    """Rename a variable from a cell onward using AST-based transformation. Renames all occurrences in the specified cell and all subsequent code cells.

    Args:
        cell: Cell reference in @A notation — rename starts from this cell
        old_name: Current variable name
        new_name: New variable name
    """
    response = args["response"]
    start_idx = parse_cell_ref(cell)
    counts = await response.run_ui_command('flowbook:get-cell-count', {})
    num_code_cells = counts['code_cells']

    modified = []
    for i in range(start_idx, num_code_cells):
        cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": i})
        source = cell_data.get('source', '')
        if not source.strip():
            continue
        new_source, was_renamed = rename_variable_in_code(source, old_name, new_name)
        if was_renamed:
            await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": i, "source": new_source})
            modified.append(index_to_alpha(i))

    if modified:
        return f"Renamed '{old_name}' \u2192 '{new_name}' in {len(modified)} cells: {', '.join(modified)}"
    return f"No occurrences of '{old_name}' found from {index_to_alpha(start_idx)} onward"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def remove_inplace(cell: str, variable: str, **args) -> str:
    """Convert df.method(inplace=True) to df = df.method() in a cell. Fixes unrecoverable mutation violations from pandas inplace operations.

    Args:
        cell: Cell reference in @A notation
        variable: Variable name (e.g., 'df')
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    source = cell_data.get('source', '')
    label = index_to_alpha(idx)

    actual_var = find_actual_variable_name(source, variable)

    try:
        tree = ast.parse(source)
        remover = InplaceRemover(actual_var)
        new_tree = remover.visit(tree)
        if remover.modified:
            new_source = ast.unparse(new_tree)
            await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": idx, "source": new_source})
            methods = ', '.join(remover.method_calls_fixed)
            return f"Removed inplace=True from {methods} on '{actual_var}' in {label}"
        return f"No inplace=True found for '{actual_var}' in {label}"
    except SyntaxError:
        return f"Could not parse cell {label} \u2014 source has syntax errors"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def insert_deepcopy(cell: str, variable: str, **args) -> str:
    """Insert copy.deepcopy() at the top of a cell and rename the variable downstream. Fixes aliasing violations.

    Args:
        cell: Cell reference in @A notation
        variable: Variable to deepcopy (e.g., 'df')
    """
    response = args["response"]
    start_idx = parse_cell_ref(cell)
    cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": start_idx})
    source = cell_data.get('source', '')
    label = index_to_alpha(start_idx)

    actual_var = find_actual_variable_name(source, variable)
    new_name = f"{actual_var}_copy"

    # Modify the target cell: insert deepcopy and rename
    magic_prefix, rest = split_cell_magic(source)
    new_rest, _ = rename_variable_in_code(rest, actual_var, new_name)
    copy_line = f"from copy import deepcopy; {new_name} = deepcopy({actual_var})\n"
    new_source = magic_prefix + copy_line + new_rest
    await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": start_idx, "source": new_source})

    # Rename in all downstream cells
    counts = await response.run_ui_command('flowbook:get-cell-count', {})
    num_code_cells = counts['code_cells']
    modified_downstream = []

    for i in range(start_idx + 1, num_code_cells):
        ds_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": i})
        ds_source = ds_data.get('source', '')
        if not ds_source.strip():
            continue
        ds_new, was_renamed = rename_variable_in_code(ds_source, actual_var, new_name)
        if was_renamed:
            await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": i, "source": ds_new})
            modified_downstream.append(index_to_alpha(i))

    downstream_msg = f", renamed in {', '.join(modified_downstream)}" if modified_downstream else ""
    return f"Inserted deepcopy of '{actual_var}' as '{new_name}' in {label}{downstream_msg}"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def mark_diagnostic(cell: str, **args) -> str:
    """Add %diagnostic magic to a cell to exclude it from reproducibility tracking. Use for inspection cells (df.info(), df.head(), plots) that don't affect computation.

    Args:
        cell: Cell reference in @A notation
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    source = cell_data.get('source', '')
    label = index_to_alpha(idx)

    if source.lstrip().startswith('%diagnostic'):
        return f"Cell {label} is already marked as diagnostic"

    new_source = prepend_to_cell_source(source, "%diagnostic\n")
    await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": idx, "source": new_source})
    return f"Marked cell {label} as diagnostic"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def merge_cells(cell_ids: list[str], **args) -> str:
    """Merge multiple code cells into the first one. Concatenates sources with blank line separators. Removes the other cells.

    Args:
        cell_ids: List of cell references in @A notation (e.g., ['@A', '@B', '@C'])
    """
    response = args["response"]
    if len(cell_ids) < 2:
        return "Need at least 2 cells to merge"

    indices = sorted(parse_cell_ref(r) for r in cell_ids)

    # Read all cell sources
    sources = []
    for idx in indices:
        cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
        sources.append(cell_data.get('source', ''))

    # Write merged source to first cell
    merged = '\n\n'.join(s for s in sources if s.strip())
    await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": indices[0], "source": merged})

    # Delete remaining cells in reverse order (so indices stay valid)
    for idx in reversed(indices[1:]):
        await response.run_ui_command('notebook-intelligence:delete-cell-at-index', {"cellIndex": idx})

    await response.run_ui_command('flowbook:notify-structure', {})
    labels = ', '.join(index_to_alpha(i) for i in indices)
    return f"Merged cells {labels} into {index_to_alpha(indices[0])}"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def move_cell(cell: str, after_cell: str, **args) -> str:
    """Move a code cell to a new position (after another cell).

    Args:
        cell: Cell reference to move (in @A notation)
        after_cell: Move to position after this cell (in @A notation)
    """
    response = args["response"]
    from_idx = parse_cell_ref(cell)
    to_idx = parse_cell_ref(after_cell)
    result = await response.run_ui_command('flowbook:move-cell', {"fromIndex": from_idx, "toIndex": to_idx})
    await response.run_ui_command('flowbook:notify-structure', {})
    return f"Moved {index_to_alpha(from_idx)} to after {index_to_alpha(to_idx)}"


# ===================================================================
# Category 5: Notebook Lifecycle
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def save_notebook(**args) -> str:
    """Save the active notebook to disk.
    """
    response = args["response"]
    await response.run_ui_command('docmanager:save')
    return "Saved notebook"


# ===================================================================
# Category 6: Checkpoint & Logging
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def checkpoint(**args) -> str:
    """Create a checkpoint (snapshot) of all cell sources and reproducibility state. Use before making refactoring changes so you can restore if needed.
    """
    response = args["response"]
    counts = await response.run_ui_command('flowbook:get-cell-count', {})
    num_code = counts['code_cells']

    cells = []
    for i in range(num_code):
        cell_data = await response.run_ui_command('flowbook:get-cell', {"cellIndex": i})
        cells.append({
            'label': index_to_alpha(i),
            'cell_type': cell_data.get('cell_type', 'code'),
            'source': cell_data.get('source', ''),
            'flowbook_meta': cell_data.get('flowbook_meta'),
        })

    # Also checkpoint the kernel's enforcer state
    enforcer_result = await response.run_ui_command('flowbook:enforcer-checkpoint', {})
    enforcer_snapshot_id = None
    if isinstance(enforcer_result, dict):
        enforcer_snapshot_id = enforcer_result.get('checkpoint_id')

    cp_id = _session.save_checkpoint(cells, enforcer_snapshot_id=enforcer_snapshot_id)

    return f"Checkpoint created: {cp_id}"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def restore(checkpoint_id: str, **args) -> str:
    """Restore cell sources and reproducibility state from a checkpoint.

    Args:
        checkpoint_id: Checkpoint ID (e.g., 'ckpt_0')
    """
    response = args["response"]
    cells = _session.get_checkpoint(checkpoint_id)

    # Only edit cells whose source actually changed
    changed = 0
    for i, cell_data in enumerate(cells):
        current = await response.run_ui_command('flowbook:get-cell', {"cellIndex": i})
        if current.get('source', '') != cell_data['source']:
            await response.run_ui_command('flowbook:edit-cell-source', {
                "cellIndex": i,
                "source": cell_data['source'],
            })
            changed += 1

    # Restore kernel enforcer state (overwrites any staleness from edits above)
    enforcer_snapshot_id = _session.get_enforcer_snapshot_id(checkpoint_id)
    if enforcer_snapshot_id:
        await response.run_ui_command('flowbook:enforcer-restore', {
            "checkpointId": enforcer_snapshot_id,
        })

    return f"Restored {changed} cells from checkpoint '{checkpoint_id}'"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def list_checkpoints(**args) -> str:
    """List all saved checkpoints with their IDs and cell counts.
    """
    listing = _session.list_checkpoints()
    if not listing:
        return "No checkpoints saved"
    lines = [f"  {cp['id']}: {cp['cell_count']} cells" for cp in listing]
    return "Checkpoints:\n" + '\n'.join(lines)


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def get_log(**args) -> str:
    """Get the FlowBook session event log as a human-readable timeline.
    """
    return _session.format_log() or "No events logged"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def save_log(path: str = "", **args) -> str:
    """Save the session event log to a JSON file.

    Args:
        path: File path to save the log to (optional)
    """
    if not path:
        return "Path is required"
    written = _session.save_log_to_file(path)
    return f"Log saved to {written}"


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def print_log(**args) -> str:
    """Print the full session event log as a human-readable timeline.
    """
    return _session.format_log() or "No events logged"


# ===================================================================
# Inspection — scratch_work & get_cell_outputs
# ===================================================================

def _render_for_participant(result, request) -> object:
    """Return a ToolContent for structured surfaces; else a markdown string.

    If `result` is not a dict (e.g. notebook-intelligence stringified a
    frontend-command error per chat-sidebar.tsx), surface it as a readable
    error instead of crashing on result.get(...).
    """
    if not isinstance(result, dict):
        msg = str(result) if result else "(empty response)"
        if msg.startswith("Error executing command:"):
            hint = (
                " — the JupyterLab extension may be out of date. "
                "Rebuild with `jlpm build` (or `jlpm watch`) and hard-refresh JupyterLab."
            )
            msg = msg + hint
        return msg
    try:
        return build_tool_content(result)
    except Exception as exc:
        log.warning("build_tool_content failed: %s", exc)
        try:
            return to_markdown(result)
        except Exception as exc2:
            return f"Internal error rendering result: {exc2}"


@nbapi.tool
@_safe_tool
async def scratch_work(code: str, **args) -> object:
    """Run code against the live kernel WITHOUT affecting reproducibility state.

    The user namespace is checkpointed before the call and restored afterwards,
    so any assignments, deletions, imports, or in-place mutations inside the
    scratch code are rolled back. Does NOT create a cell, does NOT record
    reads/writes, does NOT stale any cell. Returns full outputs including
    images and HTML. Use for ad-hoc inspection (shapes, values, quick plots).

    Args:
        code: Python code to execute.
    """
    response = args["response"]
    request = args.get("request")
    result = await response.run_ui_command(
        'flowbook:scratch-work',
        {"code": code},
    )
    rendered = _render_for_participant(result, request)
    # Until notebook-intelligence is patched, ToolContent gets stringified.
    # Calling to_markdown here ensures we always surface something readable.
    if isinstance(rendered, ToolContent):
        return rendered
    return rendered


@nbapi.auto_approve
@nbapi.tool
@_safe_tool
async def get_cell_outputs(cells: list, **args) -> object:
    """Return the full outputs of one or more cells, including images and
    HTML tables. Companion to read_cell, which shows compact markers for
    non-text outputs.

    Args:
        cells: List of cell references in @A notation or 4-char cell IDs.
    """
    response = args["response"]
    request = args.get("request")
    # Resolve @A refs to cell IDs by reading the current cell order.
    order_counts = await response.run_ui_command('flowbook:get-cell-count', {})
    num_code = order_counts.get('code_cells', 0)
    cell_ids: list[str] = []
    for ref in cells or []:
        ref = (ref or "").strip()
        if ref.startswith("@") or (ref and ref.isalpha() and ref.isupper()):
            idx = parse_cell_ref(ref)
            if idx >= num_code:
                cell_ids.append(ref)  # will surface as not-found
                continue
            cell_data = await response.run_ui_command(
                'flowbook:get-cell', {"cellIndex": idx}
            )
            cell_ids.append(cell_data.get('cell_id', ref))
        else:
            cell_ids.append(ref)
    result = await response.run_ui_command(
        'flowbook:get-cell-outputs',
        {"cellIds": cell_ids},
    )
    return _render_for_participant(result, request)


# ===================================================================
# Tool list builder
# ===================================================================

FLOWBOOK_BACKGROUND = """# FlowBook — reproducibility for Jupyter notebooks

FlowBook is a custom Jupyter kernel (`flowbook_kernel`) and toolset that turns a notebook
into a reproducibility-checked program. It records the read and write set of every cell
execution at fine granularity (variable, column, column-set, row-set, file) and enforces
**rerun consistency**: if every cell is marked CLEAN, running the notebook top-to-bottom
will reproduce the current outputs bit-for-bit.

## Programming model

The notebook is an ordered list of code cells (markdown cells are skipped for indexing).
Cells share a single Python namespace. Each execution of cell `i` is observed by the
kernel, which records:

  - `R[i]` — the set of locations the cell read.
  - `W[i]` — the set of locations the cell wrote.

A *location* is one of: a variable, a named column of a dataframe, a set of columns, a
set of rows, or a file path. When a cell is rejected as violating reproducibility, the
kernel rolls the namespace back to the state it had before the cell ran, so the
notebook's semantics match what a top-to-bottom re-run would produce.

## The four validity predicates

After every execution of cell `i`, the kernel checks four predicates over the current
`R` and `W`. If any predicate fails the execution is rejected (or flagged as a
violation, if continue-after-violation mode is on).

  1. **NoReadAndWrite** — `R[i] ∩ W[i] = ∅`. A cell that both reads and writes the
     same location accumulates state across re-runs (e.g. `train = pd.concat([train, extra])`
     grows `train` every time).
  2. **WriteBeforeRead** — `R[i] ⊆ W[1..i-1]`. Every location a cell reads must have
     been written by some earlier cell. Catches dangling references and typos.
  3. **NoReadBeforeWrite** (forward contamination) — `R[i] ∩ W[i+1..n] = ∅`. A cell
     must not read a location that is only written by a *later* cell; that implies the
     notebook was executed out of order.
  4. **NoWriteAfterRead** (backward mutation) — `W[i] ∩ R[1..i-1] = ∅`. A cell must
     not write a location that an *earlier* cell read; re-running cell `i` would change
     the earlier cell's inputs and break rerun consistency.

In addition, **UNRECOVERABLE_MUTATION** flags in-place mutations (e.g. `df.drop(inplace=True)`,
`model.fit(...)`, `list.append(...)`) that FlowBook cannot roll back. Such cells are
rejected so the namespace is never corrupted.

## Staleness

A CLEAN cell becomes STALE automatically when a nearby cell is edited or re-run in a
way that invalidates its recorded outputs:

  - **Forward staleness** — cell `j > i` reads or writes a location cell `i` just wrote;
    `j`'s outputs may no longer reflect what a top-to-bottom rerun would produce.
  - **Backward staleness** — cell `j < i` was the last writer of a location that `i` no
    longer writes; `j` must be re-run to re-establish its outputs.

Staleness is a *symptom*, not a cause — it resolves automatically once the underlying
violation is fixed and the affected cells are re-run.

## Fix taxonomy

Each violation has a preferred algorithmic fix (AST-based tools — reliable, prefer
over manual edits):

  - `alpha_rename(@X, old, new)` — rename a variable from cell `@X` onwards.
    Fixes `NoReadAndWrite` on variables and `NoWriteAfterRead` from reusing a name.
  - `insert_deepcopy(@X, var)` — introduce `var = var.copy()` to break aliasing.
    Fixes `NoReadAndWrite` on columns (`df['x'] = df['x'] + 1`) and object mutation.
  - `remove_inplace(@X, var)` — rewrite `df.method(inplace=True)` as `df = df.method()`.
    Fixes `UNRECOVERABLE_MUTATION` from pandas `inplace=True`.
  - `mark_diagnostic(@X)` — exclude a pure-inspection cell (`df.info()`, plots, `print(...)`)
    from read/write tracking. Fixes `NoWriteAfterRead` when the inspection cell precedes
    a legitimate writer.
  - `merge_cells([@X, @Y])` — merge adjacent cells when an allocation and its mutation
    were split across cells. Fixes `UNRECOVERABLE_MUTATION` from `obj = X(); obj.fit(...)`.
  - `move_cell(@X, @Y)` — reorder cells. Useful when inspection cells would be correct
    if placed after the mutating cell.

## Cell addressing

Cells are always referenced by **@-label**: `@A` is the first code cell, `@B` the
second, continuing `@Z, @AA, @AB, ...`. Markdown cells are not counted. Never use raw
integer indices or the kernel's internal cell IDs — always use `@A` notation, and
always use the FlowBook tools (never write-through tools that rewrite cells without
preserving identity).
"""


FLOWBOOK_INSTRUCTIONS = FLOWBOOK_BACKGROUND + """
## Workflow

IMPORTANT: Before each tool call, print a one-line status message explaining what you're doing
(e.g., "Running all actionable cells...", "Checkpointing before fix...", "Renaming 'df' to 'df_clean' from @C...").

To read the full notebook, call read_cell() with no arguments — returns all cells in one call.
To read a single cell, pass its @-label (e.g., read_cell("@C")).

Run-and-fix loop:
1. run_actionable_cells() — runs all stale/unexecuted cells, stops on first error or violation.
2. If violations found:
   a. checkpoint() — save state before attempting fix.
   b. get_next_actionable_cell() — find the problem cell.
   c. read_cell("@X") — read the problematic cell.
   d. Apply the appropriate fix from the taxonomy above.
   e. run_actionable_cells() — re-run to check the fix worked.
   f. If worse, restore(checkpoint_id) and try a different strategy.
3. save_notebook() when done.

Use checkpoint() before making changes, restore() if needed.
Always use FlowBook tools for cell operations — they preserve cell identity and track reproducibility.
Never use indices or cell IDs directly — always use @A notation.
"""


def create_tools(session: FlowBookSession) -> list:
    """Create the list of all FlowBook NBI tools."""
    global _session
    _session = session

    return [
        # Metadata & Status
        get_flowbook_metadata,
        get_next_actionable_cell,
        get_status,
        # Cell Operations
        read_cell,
        edit_cell_source,
        add_cell,
        delete_cell,
        # Execution
        run_cell,
        run_actionable_cell,
        run_actionable_cells,
        run_all_cells,
        continue_after_violation,
        # Inspection
        scratch_work,
        get_cell_outputs,
        # Source Refactoring
        alpha_rename,
        remove_inplace,
        insert_deepcopy,
        mark_diagnostic,
        merge_cells,
        move_cell,
        # Lifecycle
        save_notebook,
        # Checkpoint & Logging
        checkpoint,
        restore,
        list_checkpoints,
        get_log,
        save_log,
        print_log,
    ]
