"""
SDC Enforcer - Sequential Dataflow Consistency enforcement.

Implements the three SDC rules:
1. Reproducibility Invariant (structural) - the goal
2. Staleness Propagation Rule (computed here) - tracks which cells need re-execution
3. No Backward Mutation Constraint (enforced here) - prevents hidden dependencies

See ferret_sdc_kernel.py module docstring for detailed architecture documentation.
"""

import pprint
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData
from data_ferret.kernel.types import DiffResult, DiffNode, ValueComparison, CompoundDiff
from data_ferret.util.cell_index import index_to_alpha

from .models import SDCExecutionRecord, SDCResult, SDCViolation

# Checkpoint naming constants
PRE_CHECKPOINT_PREFIX = "_pre_"
POST_CHECKPOINT_PREFIX = "_post_"


class SDCEnforcer:
    """
    Enforces Sequential Dataflow Consistency.

    Tracks cell executions and their read/write sets.
    On each execution, checks for backward mutations and computes staleness.
    """

    def __init__(self, checkpoints: Checkpoints):
        self.checkpoints = checkpoints
        self.records: Dict[str, SDCExecutionRecord] = {}
        self.seq_counter: int = 0
        self._cell_order: List[str] = []
        self._stale_cells: Set[str] = set()  # Cache for absolute staleness state

    @property
    def cell_order(self) -> List[str]:
        return self._cell_order

    def set_cell_order(self, order: List[str]) -> None:
        """Update notebook structure. Called via magic or metadata."""
        self._cell_order = order
        self._prune_deleted_cells()

    def _cell_id_to_alpha(self, cell_id: str) -> str:
        """Convert cell ID to @A notation using cell_order position."""
        try:
            index = self._cell_order.index(cell_id)
            return index_to_alpha(index)
        except ValueError:
            # Cell not in order, just return the ID
            return cell_id

    def _prune_deleted_cells(self) -> None:
        """Remove records for cells no longer in notebook."""
        current = set(self._cell_order)
        deleted = [c for c in self.records if c not in current]
        for c in deleted:
            del self.records[c]
            self._stale_cells.discard(c)  # Also remove from stale cache

    def check(
        self,
        cell_id: str,
        pre_checkpoint: Checkpoint,
        post_checkpoint: Checkpoint,
        tracking: TrackingData,
        continue_on_violation: bool = False,
    ) -> SDCResult:
        """
        Main entry point. Call after cell execution.

        Args:
            cell_id: ID of the cell that just executed
            pre_checkpoint: Snapshot of namespace before execution
            post_checkpoint: Snapshot of namespace after execution
            tracking: TrackingData with reads/writes
            continue_on_violation: If True, compute staleness even when violation detected

        Returns:
            SDCResult with violation info, absolute set of stale cells, and changed variables
        """
        self.seq_counter += 1

        # Get position in document order
        try:
            my_position = self._cell_order.index(cell_id)
        except ValueError:
            # Cell not in order list - can't enforce SDC
            my_position = -1

        # Rule 3: Check backward mutation (also returns the diff for reuse)
        violation = None
        current_diff = None
        if my_position >= 0:
            violation, current_diff = self._check_backward_mutation(
                cell_id, my_position, pre_checkpoint, post_checkpoint, tracking
            )

        stale = []
        changed_vars = []
        column_changed = {}

        if not violation or continue_on_violation:
            # Update our record for this cell BEFORE computing staleness
            # (so this cell is considered "fresh" in the computation)
            self.records[cell_id] = SDCExecutionRecord(
                cell_id=cell_id,
                tracking=tracking,
                execution_seq=self.seq_counter,
            )

            # This cell just executed, so it's now fresh
            self._stale_cells.discard(cell_id)

            # Reuse diff from backward mutation check, or compute if not available
            if current_diff is None:
                current_diff = Checkpoint.diff(
                    pre_checkpoint,
                    post_checkpoint,
                    use_leq=True,
                    column_rbw=tracking.column_reads_before_writes,
                )

            # Check if diff was truncated - if so, return violation
            truncated_vars = _check_for_truncation(current_diff)
            if truncated_vars:
                formatted_diff = _format_diff_for_display(current_diff, truncated_vars)
                mutating_alpha = self._cell_id_to_alpha(cell_id)
                return SDCResult(
                    violation=SDCViolation(
                        mutating_cell=cell_id,
                        affected_cell=cell_id,
                        variables=truncated_vars,
                        message=(
                            f"Cell {mutating_alpha}: SDC diff was truncated for variables: {truncated_vars}. "
                            "Tracking may be incomplete. Consider increasing max_diffs_per_container."
                        ),
                        truncation_details=formatted_diff,
                    ),
                    stale_cells=[],
                    changed_variables=[],
                    column_changed={},
                )

            if current_diff.differences:
                changed_vars = list(current_diff.differences.keys())

            # Extract column-level changes from diff result
            column_changed = _extract_column_changes(current_diff, tracking)

            # Update staleness INCREMENTALLY (only check cells that might have become stale)
            stale = self._update_staleness_incremental(
                post_checkpoint, set(changed_vars), column_changed, cell_id
            )

        return SDCResult(
            violation=violation,
            stale_cells=stale,
            changed_variables=changed_vars,
            column_changed=column_changed,
        )

    def _check_backward_mutation(
        self,
        cell_id: str,
        my_position: int,
        pre_checkpoint: Checkpoint,
        post_checkpoint: Checkpoint,
        tracking: TrackingData,
    ) -> Tuple[Optional[SDCViolation], DiffResult]:
        """
        Check if current cell causes a backward mutation (Rule 3 violation).

        A backward mutation occurs when a cell modifies a variable that an
        earlier cell (in notebook order) reads. This prevents hidden dependencies
        where earlier cells depend on later cells having run first.

        Includes column-aware conflict detection for DataFrames: modifying
        df['price'] doesn't conflict with a cell that only reads df['quantity'].

        Returns:
            Tuple of (violation, diff_result) - diff_result is returned for reuse
            by caller to avoid redundant computation.
        """
        # Compute what THIS cell actually modified
        # For diff detection, check all accessed columns (read OR written)
        all_accessed_columns = {}
        for var, cols in tracking.column_reads_before_writes.items():
            all_accessed_columns[var] = set(cols)
        for var, cols in tracking.column_writes.items():
            if var in all_accessed_columns:
                all_accessed_columns[var].update(cols)
            else:
                all_accessed_columns[var] = set(cols)

        current_diff = Checkpoint.diff(
            pre_checkpoint,
            post_checkpoint,
            use_leq=True,
            column_rbw=all_accessed_columns,
        )

        # Check if diff was truncated - if so, return violation
        truncated_vars = _check_for_truncation(current_diff)
        if truncated_vars:
            formatted_diff = _format_diff_for_display(current_diff, truncated_vars)
            mutating_alpha = self._cell_id_to_alpha(cell_id)
            return (
                SDCViolation(
                    mutating_cell=cell_id,
                    affected_cell=cell_id,
                    variables=truncated_vars,
                    message=(
                        f"⚠️ Cell {mutating_alpha}: SDC diff was truncated for variables: {truncated_vars}. "
                        "Tracking may be incomplete. Consider increasing max_diffs_per_container."
                    ),
                    truncation_details=formatted_diff,
                ),
                current_diff,
            )

        if not current_diff.differences:
            # This cell didn't modify anything, no backward mutation possible
            return (None, current_diff)

        # Variable-level modifications
        modified_vars = set(current_diff.differences.keys())

        # Column-level modifications (for DataFrames)
        modified_columns = _extract_column_changes(current_diff, tracking)

        # Check if any earlier cell reads something we modified
        for prior_cell_id in self._cell_order[:my_position]:
            prior_record = self.records.get(prior_cell_id)
            if prior_record is None:
                continue

            # Column-aware conflict detection
            conflicts = _check_read_write_conflict(
                modified_vars=modified_vars,
                modified_columns=modified_columns,
                prior_reads=prior_record.tracking.reads_before_writes,
                prior_column_reads=prior_record.tracking.column_reads_before_writes,
            )

            if conflicts:
                mutating_alpha = self._cell_id_to_alpha(cell_id)
                affected_alpha = self._cell_id_to_alpha(prior_cell_id)
                return (
                    SDCViolation(
                        mutating_cell=cell_id,
                        affected_cell=prior_cell_id,
                        variables=conflicts,
                        message=(
                            f"Cell {mutating_alpha} modified {conflicts} "
                            f"which cell {affected_alpha} (earlier in notebook) reads. "
                            f"This violates Sequential Dataflow Consistency."
                        ),
                    ),
                    current_diff,
                )

        return (None, current_diff)

    def _update_staleness_incremental(
        self,
        current_checkpoint: Checkpoint,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
    ) -> List[str]:
        """
        Incrementally update staleness cache (Rule 2 computation).

        Only checks cells that could have become stale from this execution:
        - Skips cells already marked stale
        - Skips cells whose reads don't overlap with changed variables/columns

        Args:
            current_checkpoint: The current state of the namespace
            changed_vars: Set of variable names that changed in this execution
            column_changed: Dict mapping var names to lists of changed column names
            just_executed: The cell_id that just executed (already marked fresh)

        Returns:
            List of all currently stale cell IDs (in document order)
        """
        for cell_id, record in self.records.items():
            if cell_id == just_executed:
                continue  # This cell just ran, already marked fresh

            if cell_id in self._stale_cells:
                continue  # Already stale, no need to re-check

            # Skip cells whose reads don't overlap with changed vars
            if not self._has_relevant_overlap(record, changed_vars, column_changed):
                continue

            # Cell MIGHT be stale - do the expensive diff check
            pre_checkpoint = self.checkpoints.get(f"{PRE_CHECKPOINT_PREFIX}{cell_id}")
            if pre_checkpoint is None:
                continue

            diff_result = Checkpoint.diff(
                pre_checkpoint,
                current_checkpoint,
                keys_to_include=record.tracking.reads_before_writes,
                use_leq=True,
                column_rbw=record.tracking.column_reads_before_writes,
            )

            if diff_result.differences:
                self._stale_cells.add(cell_id)

        # Return in document order
        return [cid for cid in self._cell_order if cid in self._stale_cells]

    def _has_relevant_overlap(
        self,
        record: SDCExecutionRecord,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
    ) -> bool:
        """
        Check if cell's reads overlap with changes at variable or column level.

        Returns True if:
        - Cell reads a variable that changed AND either:
          - No column-level info available (conservative: assume overlap)
          - Column-level info shows actual column overlap

        Args:
            record: The cell's execution record with tracking info
            changed_vars: Variables that changed in current execution
            column_changed: Dict mapping var names to changed column names

        Returns:
            True if the cell might be affected by the changes
        """
        reads = record.tracking.reads_before_writes
        var_overlap = reads & changed_vars

        if not var_overlap:
            return False  # No variable-level overlap at all

        # Check column-level overlap for each overlapping variable
        for var in var_overlap:
            changed_cols = set(column_changed.get(var, []))
            read_cols = record.tracking.column_reads_before_writes.get(var, None)

            if not changed_cols or read_cols is None:
                # No column info on one or both sides - conservative: assume overlap
                return True

            if changed_cols & read_cols:
                # Actual column overlap found
                return True

        # All overlapping vars have column info and no column overlap
        return False

    def get_stale_cells(self) -> List[str]:
        """
        Get the current set of stale cells (in document order).

        Returns the cached staleness state without recomputing. For a full
        recomputation from scratch, use compute_all_stale_cells().

        Returns:
            List of cell IDs that are currently stale (in document order)
        """
        return [cid for cid in self._cell_order if cid in self._stale_cells]

    def compute_all_stale_cells(self, current_checkpoint: Checkpoint) -> List[str]:
        """
        Recompute staleness for ALL cells from scratch.

        Unlike incremental updates, this checks every executed cell against
        the current namespace state. Use this when you need guaranteed
        accuracy (e.g., after external namespace modifications).

        Args:
            current_checkpoint: The current state of the namespace

        Returns:
            List of cell IDs that are currently stale (in document order)
        """
        self._stale_cells.clear()

        for cell_id, record in self.records.items():
            pre_checkpoint = self.checkpoints.get(f"{PRE_CHECKPOINT_PREFIX}{cell_id}")
            if pre_checkpoint is None:
                continue

            diff_result = Checkpoint.diff(
                pre_checkpoint,
                current_checkpoint,
                keys_to_include=record.tracking.reads_before_writes,
                use_leq=True,
                column_rbw=record.tracking.column_reads_before_writes,
            )

            if diff_result.differences:
                self._stale_cells.add(cell_id)

        return [cid for cid in self._cell_order if cid in self._stale_cells]

    def reset(self) -> None:
        """Clear all state. Called on kernel restart."""
        self.records.clear()
        self.seq_counter = 0
        self._cell_order = []
        self._stale_cells.clear()


def _check_read_write_conflict(
    modified_vars: Set[str],
    modified_columns: Dict[str, List[str]],
    prior_reads: Set[str],
    prior_column_reads: Dict[str, Set[str]],
) -> List[str]:
    """
    Check if modifications conflict with prior reads at variable or column level.

    Implements three-level conflict detection:
    1. Variable-only conflicts: Prior reads var, no column info on either side
    2. Mixed conflicts: Column info on one side only (conservative: flag as conflict)
    3. Column-level conflicts: Both have column info, check column overlap

    Args:
        modified_vars: Set of variable names modified by current cell
        modified_columns: Dict of variable -> list of modified columns
        prior_reads: Set of variables read by prior cell
        prior_column_reads: Dict of variable -> set of read columns for prior cell

    Returns:
        List of conflicting variable/column references (e.g., ["df.price", "config"])
    """
    conflicts = []

    # Check each variable that was modified
    for var in modified_vars:
        # Does prior cell read this variable?
        if var not in prior_reads:
            continue

        # Variable is read by prior cell - check if we have column-level info

        current_cols = set(modified_columns.get(var, []))
        prior_cols = prior_column_reads.get(var, None)

        if prior_cols is None or not current_cols:
            # Level 1 or 2: No column info on one or both sides
            # Conservative: flag as conflict at variable level
            conflicts.append(var)
        else:
            # Level 3: Both have column info - check overlap
            overlap_cols = current_cols & prior_cols
            if overlap_cols:
                # Conflict at column level - report each column
                for col in sorted(overlap_cols):
                    conflicts.append(f"{var}.{col}")
            # else: No overlap, no conflict

    return sorted(conflicts)


def _extract_column_changes(
    diff_result: DiffResult, tracking: TrackingData
) -> Dict[str, List[str]]:
    """
    Extract which DataFrame columns changed values from diff tree.

    Args:
        diff_result: The diff result from Checkpoint.diff()
        tracking: TrackingData with column_reads_before_writes and column_writes

    Returns:
        Dict mapping variable paths to lists of changed column names
    """
    column_changed = {}

    # Only process variables that have column tracking
    tracked_vars = set(tracking.column_reads_before_writes.keys()) | set(
        tracking.column_writes.keys()
    )

    for var_name in tracked_vars:
        if var_name not in diff_result.differences:
            # Variable didn't change at all
            continue

        diff_node = diff_result.differences[var_name]
        changed_cols = _get_changed_columns_from_diff_node(diff_node)

        if changed_cols:
            column_changed[var_name] = sorted(changed_cols)

    return column_changed


def _get_changed_columns_from_diff_node(node: DiffNode) -> Set[str]:
    """
    Parse a DiffNode to extract changed DataFrame column names.

    For DataFrames, the diff structure can be:
    1. Nested dict with column-level changes: {"['column_name']": ..., "['column_name'][0]": ..., etc.}
    2. ValueComparison with message about missing/added columns (legacy format)

    We extract column names from both formats.

    Args:
        node: DiffNode (either ValueComparison or Dict)

    Returns:
        Set of column names that changed
    """
    changed_cols = set()

    if isinstance(node, ValueComparison):
        # Check if this is a DataFrame diff with column information in the message
        # Legacy format: "DataFrame missing RBW columns in pre-state at var: ['col1', 'col2']"
        # New format: "DataFrame column 'col' missing in pre-state at var"
        if hasattr(node, 'message') and node.message:
            # Try legacy format first
            missing_match = re.search(r"missing RBW columns in (?:pre|post)-state at \w+: \[([^\]]+)\]", node.message)
            if missing_match:
                # Parse the column list
                cols_str = missing_match.group(1)
                # Remove quotes and split by comma
                for col in cols_str.split(','):
                    col = col.strip().strip("'\"")
                    if col:
                        changed_cols.add(col)
            else:
                # Try new format: "DataFrame column 'col' missing/deleted ..."
                col_match = re.search(r"DataFrame column '([^']+)' (?:missing|deleted)", node.message)
                if col_match:
                    changed_cols.add(col_match.group(1))
                # Also try: "Column 'col' only in first/second DataFrame"
                col_match2 = re.search(r"Column '([^']+)' only in", node.message)
                if col_match2:
                    changed_cols.add(col_match2.group(1))
        return changed_cols

    if isinstance(node, CompoundDiff):
        # Walk the CompoundDiff children and extract column names
        for key in node.children.keys():
            # DataFrame column diffs have keys like:
            # - "['column_name']" - column-level difference
            # - "['column_name'][0]" - row-level difference within column
            # - "['column_name']._dtype" - dtype difference for column
            # Extract the column name using regex
            match = re.match(r"\['([^']+)'\]", key)
            if match:
                column_name = match.group(1)
                changed_cols.add(column_name)
        return changed_cols

    if isinstance(node, dict):
        # Walk the diff dict and extract column names (legacy format)
        for key in node.keys():
            # Skip special markers
            if key == "_truncated" or key == "_index":
                continue

            # DataFrame column diffs have keys like:
            # - "['column_name']" - column-level difference
            # - "['column_name'][0]" - row-level difference within column
            # - "['column_name']._dtype" - dtype difference for column
            # Extract the column name using regex
            match = re.match(r"\['([^']+)'\]", key)
            if match:
                column_name = match.group(1)
                changed_cols.add(column_name)

    return changed_cols


def _check_for_truncation(diff_result: DiffResult) -> List[str]:
    """
    Check if truncation might cause us to miss which columns/keys changed.

    We only care about truncation at the variable's top level for certain types:
    - DataFrame: truncation would mean we might miss columns
    - dict: truncation would mean we might miss keys
    - object: truncation would mean we might miss attributes

    We do NOT care about truncation for:
    - array, list, tuple, series: these are value containers where truncation
      just means we don't know ALL the changed values (but we know it changed)

    Args:
        diff_result: The DiffResult to check

    Returns:
        List of variable names that had truncated diffs (empty if no truncation)
    """
    # Types where truncation could cause us to miss keys/columns
    STRUCTURAL_TYPES = {"dataframe", "dict", "object"}

    truncated_vars = []

    for var_name, diff_node in diff_result.differences.items():
        if isinstance(diff_node, CompoundDiff):
            # Only flag truncation for types where we might miss columns/keys
            if diff_node.truncated and diff_node.source_type in STRUCTURAL_TYPES:
                truncated_vars.append(var_name)

    return truncated_vars


def _format_diff_for_display(
    diff_result: DiffResult, truncated_vars: List[str], max_width: int = 120
) -> str:
    """
    Format truncated diff for human-readable display.

    Args:
        diff_result: The DiffResult containing differences
        truncated_vars: List of variable names that were truncated
        max_width: Maximum width for pretty printing

    Returns:
        Formatted string showing the truncated diffs
    """
    lines = ["=" * 60, "TRUNCATED DIFF DETAILS", "=" * 60]

    for var_name in truncated_vars:
        if var_name not in diff_result.differences:
            continue

        diff_node = diff_result.differences[var_name]
        lines.append(f"\nVariable: {var_name}")
        lines.append("-" * 40)

        # Pretty print the diff structure
        formatted = pprint.pformat(
            _diff_node_to_dict(diff_node),
            width=max_width,
            depth=4,  # Limit depth to avoid overwhelming output
            compact=True,
        )
        lines.append(formatted)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _diff_node_to_dict(node: DiffNode) -> dict:
    """
    Convert a DiffNode to a plain dict for pretty printing.

    Handles ValueComparison objects and CompoundDiff objects.
    """
    if isinstance(node, ValueComparison):
        result = {"status": node.status}
        if node.message:
            result["message"] = node.message
        if node.value1 is not None:
            result["before"] = _truncate_value(node.value1)
        if node.value2 is not None:
            result["after"] = _truncate_value(node.value2)
        return result
    elif isinstance(node, CompoundDiff):
        result = {
            "_type": node.source_type,
            "_truncated": node.truncated,
        }
        for k, v in node.children.items():
            result[k] = _diff_node_to_dict(v)
        return result
    elif isinstance(node, dict):
        # Fallback for any plain dicts (shouldn't happen with new structure)
        return {k: _diff_node_to_dict(v) for k, v in node.items()}
    else:
        return _truncate_value(node)


def _truncate_value(value, max_len: int = 100) -> str:
    """Truncate long values for display."""
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s
