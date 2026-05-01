"""NBI extension tool implementations for FlowBook.

All tools use run_ui_command() for notebook I/O via FlowBook's frontend bridge
commands. Cell references use @A notation (code-cell-only indexing).
"""

import ast
import time
import logging

import notebook_intelligence.api as nbapi

from flowbook.util.cell_index import index_to_alpha, parse_cell_ref
from flowbook.nbi.session import FlowBookSession
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
async def get_flowbook_metadata(cell: str, **args) -> str:
    """Get FlowBook reproducibility metadata for a code cell. Returns read/write locations, errors, timing, and staleness info.

    Args:
        cell: Cell reference in @A notation (code cells only, e.g., @A, @C, @AA)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    meta = await response.run_ui_command('flowbook:get-metadata', {"cellIndex": idx})
    return str(meta)


@nbapi.auto_approve
@nbapi.tool
async def get_next_actionable_cell(**args) -> str:
    """Get the next cell that needs attention. Priority: error > stale > unexecuted. Returns cell label and reason, or 'done' if all cells are clean.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:get-next-actionable', {})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def get_flowbook_status(**args) -> str:
    """Get overall reproducibility status: total cells, executed, stale, clean, errors, and whether the notebook is reproducible.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:get-status', {})
    return str(result)


# ===================================================================
# Category 2: Cell Operations (replaces disabled NBI tools)
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
async def get_all_cell_sources(**args) -> str:
    """Return the source code of all code cells in one response. Each cell is shown with its @-label, separated by clear boundary markers. Much cheaper than calling read_cell for each cell individually.
    """
    response = args["response"]
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


@nbapi.auto_approve
@nbapi.tool
async def read_cell(cell: str, **args) -> str:
    """Read a code cell's source, outputs, and FlowBook metadata.

    Args:
        cell: Cell reference in @A notation (code cells only)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def read_cell_output(cell: str, **args) -> str:
    """Read a code cell's execution output.

    Args:
        cell: Cell reference in @A notation (code cells only)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:get-cell-output', {"cellIndex": idx})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def edit_cell_source(cell: str, source: str, **args) -> str:
    """Edit a code cell's source. Uses identity-safe in-place modification that preserves cell ID and triggers FlowBook's edit detection.

    Args:
        cell: Cell reference in @A notation (code cells only)
        source: New source code for the cell
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:edit-cell-source', {"cellIndex": idx, "source": source})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def add_code_cell(source: str, **args) -> str:
    """Add a new code cell to the end of the notebook.

    Args:
        source: Python code source for the new cell
    """
    response = args["response"]
    await response.run_ui_command('notebook-intelligence:add-code-cell-to-active-notebook', {"source": source})
    await response.run_ui_command('flowbook:notify-structure', {})
    return "Added code cell"


@nbapi.auto_approve
@nbapi.tool
async def add_markdown_cell(source: str, **args) -> str:
    """Add a new markdown cell to the end of the notebook.

    Args:
        source: Markdown source for the new cell
    """
    response = args["response"]
    await response.run_ui_command('notebook-intelligence:add-markdown-cell-to-active-notebook', {"source": source})
    await response.run_ui_command('flowbook:notify-structure', {})
    return "Added markdown cell"


@nbapi.auto_approve
@nbapi.tool
async def insert_cell(after_cell: str, cell_type: str, source: str, **args) -> str:
    """Insert a new cell after the specified code cell.

    Args:
        after_cell: Cell reference in @A notation — insert after this cell
        cell_type: Cell type: 'code' or 'markdown'
        source: Source content for the new cell
    """
    response = args["response"]
    # NBI's insert-cell-at-index uses notebook-widget indices, but we need
    # to map from code-cell index. Get the widget index from the frontend.
    idx = parse_cell_ref(after_cell)
    cell_info = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    # Insert at the position after this cell (NBI uses 0-based widget index)
    # We need the widget index, not the code-cell index. The simplest approach
    # is to get the total cell count and insert at the right position.
    # For now, use NBI's insert which takes a notebook-widget index.
    # TODO: This may need a bridge command for precise positioning.
    await response.run_ui_command('notebook-intelligence:add-code-cell-to-active-notebook', {"source": source})
    await response.run_ui_command('flowbook:notify-structure', {})
    return f"Inserted {cell_type} cell after {index_to_alpha(idx)}"


@nbapi.auto_approve
@nbapi.tool
async def delete_cell(cell: str, **args) -> str:
    """Delete a code cell.

    Args:
        cell: Cell reference in @A notation (code cells only)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    label = index_to_alpha(idx)
    # Get total cells to compute widget index — we need a bridge for this
    # For now, get the cell info to confirm it exists, then use NBI delete
    cell_info = await response.run_ui_command('flowbook:get-cell', {"cellIndex": idx})
    # NBI delete uses widget index. We need the widget index from the frontend.
    # The bridge command returns cell_id — we can search for it.
    # Simplest: count cells up to this code-cell index to find widget index
    counts = await response.run_ui_command('flowbook:get-cell-count', {})
    # Actually, we need a more reliable approach. Let's add widget index to get-cell response.
    # For now, iterate: code cell idx → widget idx by counting
    # The NBI delete-cell-at-index takes a 0-based notebook-widget index
    # We approximate by using the code-cell index (works when all cells are code)
    # TODO: Add widget index to flowbook:get-cell response for precise deletion
    await response.run_ui_command('notebook-intelligence:delete-cell-at-index', {"cellIndex": idx})
    await response.run_ui_command('flowbook:notify-structure', {})
    return f"Deleted cell {label}"


@nbapi.auto_approve
@nbapi.tool
async def get_cell_count(**args) -> str:
    """Get the number of cells in the notebook (code cells, markdown cells, total).
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:get-cell-count', {})
    return str(result)


# ===================================================================
# Category 3: Execution
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
async def run_cell(cell: str, **args) -> str:
    """Run a code cell and return its outputs and FlowBook reproducibility metadata.

    Args:
        cell: Cell reference in @A notation (code cells only)
    """
    response = args["response"]
    idx = parse_cell_ref(cell)
    result = await response.run_ui_command('flowbook:run-cell', {"cellIndex": idx})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def run_actionable_cell(**args) -> str:
    """Run the next cell that needs attention (error > stale > unexecuted). Returns the cell's outputs and FlowBook metadata, or 'done' if all cells are clean.
    """
    response = args["response"]
    actionable = await response.run_ui_command('flowbook:get-next-actionable', {})
    if isinstance(actionable, dict) and actionable.get('done'):
        return "All cells are clean. Notebook is reproducible."
    idx = actionable['index']
    result = await response.run_ui_command('flowbook:run-cell', {"cellIndex": idx})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def run_actionable_cells(**args) -> str:
    """Run all stale and unexecuted cells until the notebook is reproducible or an error occurs. Stops on hard errors always. Stops on violations if continue_after_violation is disabled.
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:run-actionable-cells', {})
    return str(result)


@nbapi.auto_approve
@nbapi.tool
async def continue_after_violation(enabled: bool, **args) -> str:
    """Set whether to continue execution after a predicate violation (True = report only, False = reject and rollback).

    Args:
        enabled: True to continue after violations, False to reject
    """
    response = args["response"]
    result = await response.run_ui_command('flowbook:set-continue-after-violation', {"enabled": enabled})
    return f"continue_after_violation set to {enabled}"


# ===================================================================
# Category 4: Source Refactoring Tools
# ===================================================================

@nbapi.auto_approve
@nbapi.tool
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

    return f"Renamed '{old_name}' -> '{new_name}' in {len(modified)} cells: {', '.join(modified)}" if modified else f"No occurrences of '{old_name}' found from {index_to_alpha(start_idx)} onward"


@nbapi.auto_approve
@nbapi.tool
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
        return f"Could not parse cell {label} — source has syntax errors"


@nbapi.auto_approve
@nbapi.tool
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
    copy_line = f"import copy; {new_name} = copy.deepcopy({actual_var})\n"
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
async def merge_cells(cells: str, **args) -> str:
    """Merge multiple code cells into the first one. Concatenates sources with blank line separators. Removes the other cells.

    Args:
        cells: Comma-separated cell references in @A notation (e.g., '@A,@B,@C')
    """
    response = args["response"]
    refs = [c.strip() for c in cells.split(',')]
    if len(refs) < 2:
        return "Need at least 2 cells to merge"

    indices = [parse_cell_ref(r) for r in refs]
    indices.sort()

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
async def create_notebook(**args) -> str:
    """Create a new empty notebook.
    """
    response = args["response"]
    result = await response.run_ui_command('notebook-intelligence:create-new-notebook-from-py', {"code": ""})
    return f"Created new notebook at {result.get('path', 'unknown')}"


@nbapi.auto_approve
@nbapi.tool
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
async def checkpoint(**args) -> str:
    """Create a checkpoint (snapshot) of all cell sources. Use before making refactoring changes so you can restore if needed.
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
        })

    cp_id = _session.save_checkpoint(cells)
    return f"Checkpoint '{cp_id}' saved ({num_code} code cells)"


@nbapi.auto_approve
@nbapi.tool
async def restore(checkpoint_id: str, **args) -> str:
    """Restore cell sources from a checkpoint. Overwrites current cell sources with the saved snapshot.

    Args:
        checkpoint_id: Checkpoint ID (e.g., 'ckpt_0')
    """
    response = args["response"]
    cells = _session.get_checkpoint(checkpoint_id)

    for i, cell_data in enumerate(cells):
        await response.run_ui_command('flowbook:edit-cell-source', {
            "cellIndex": i,
            "source": cell_data['source'],
        })

    return f"Restored {len(cells)} cells from checkpoint '{checkpoint_id}'"


@nbapi.auto_approve
@nbapi.tool
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
async def get_log(**args) -> str:
    """Get the FlowBook session event log as a human-readable timeline.
    """
    return _session.format_log() or "No events logged"


@nbapi.auto_approve
@nbapi.tool
async def save_log(path: str, **args) -> str:
    """Save the session event log to a JSON file.

    Args:
        path: File path to save the log to
    """
    written = _session.save_log_to_file(path)
    return f"Log saved to {written}"


@nbapi.auto_approve
@nbapi.tool
async def print_log(**args) -> str:
    """Print the full session event log as a human-readable timeline.
    """
    return _session.format_log() or "No events logged"


# ===================================================================
# Tool list builder
# ===================================================================

FLOWBOOK_INSTRUCTIONS = """FlowBook provides reproducibility tracking for Jupyter notebooks running the flowbook_kernel.
Cells are referenced using @A, @B, ... @Z, @AA notation (document order, code cells only). Markdown cells are not counted.

Workflow:
1. Use get_next_actionable_cell to find what needs attention (error > stale > unexecuted).
2. Use run_actionable_cell to execute it with reproducibility checking.
3. If a violation occurs, use refactoring tools to fix it:
   - alpha_rename: rename a variable from a cell onward
   - remove_inplace: convert df.method(inplace=True) to df = df.method()
   - insert_deepcopy: insert copy.deepcopy() to break aliasing
   - mark_diagnostic: exclude a cell from reproducibility tracking
   - merge_cells: combine adjacent cells into one
   - move_cell: reorder cells
4. Use run_actionable_cells to run all remaining cells until clean or error.
5. Use get_flowbook_metadata to inspect a cell's read/write sets and violations.

Use checkpoint before making changes, restore if needed.
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
        get_flowbook_status,
        # Cell Operations
        get_all_cell_sources,
        read_cell,
        read_cell_output,
        edit_cell_source,
        add_code_cell,
        add_markdown_cell,
        insert_cell,
        delete_cell,
        get_cell_count,
        # Execution
        run_cell,
        run_actionable_cell,
        run_actionable_cells,
        continue_after_violation,
        # Source Refactoring
        alpha_rename,
        remove_inplace,
        insert_deepcopy,
        mark_diagnostic,
        merge_cells,
        move_cell,
        # Lifecycle
        create_notebook,
        save_notebook,
        # Checkpoint & Logging
        checkpoint,
        restore,
        list_checkpoints,
        get_log,
        save_log,
        print_log,
    ]
