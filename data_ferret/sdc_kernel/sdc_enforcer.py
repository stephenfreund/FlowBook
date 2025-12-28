"""
SDC Enforcer - Sequential Dataflow Consistency enforcement.

================================================================================
OVERVIEW
================================================================================

This module implements the core SDC enforcement logic, responsible for:
1. Recording cell execution history with access patterns
2. Detecting backward mutation violations (Rule 3)
3. Computing staleness for downstream cells (Rule 2)
4. Managing checkpoints for rollback on violation

================================================================================
THE THREE SDC RULES
================================================================================

Rule 1: Reproducibility Invariant (Goal)
    A notebook is reproducible if running cells in document order from a fresh
    kernel always produces the same result. This is the property SDC guarantees.

Rule 2: Staleness Propagation (Computed)
    A cell is "stale" when variables it read have changed since execution.
    Staleness is informational - stale cells are highlighted for user awareness.

Rule 3: No Backward Mutation (Enforced)
    A cell may NOT modify variables that earlier cells (in document order) read.
    This prevents hidden dependencies where earlier cells depend on later cells.

================================================================================
CONFLICT RESOLUTION HIERARCHY
================================================================================

When checking for backward mutations, conflicts are evaluated in precedence order:

1. COLUMN-LEVEL EXEMPTION (most specific)
   If both cells have column tracking for variable V:
   - Modifying df['col_a'] does NOT conflict with reading df['col_b']
   - Only overlapping columns cause violations

2. STRUCTURAL CONFLICTS (if structural tracking enabled)
   If earlier cell read structural attributes (df.columns, df.shape, len(df)):
   - Adding/removing columns triggers conflict with .columns reads
   - Adding/removing rows triggers conflict with .shape/len reads
   - Mode determines response: WARN (warning) or ENFORCE (violation)

3. VARIABLE-LEVEL FALLBACK (most conservative)
   If column information is unavailable on either side:
   - Any modification to a read variable is treated as a conflict
   - This ensures safety when precise tracking isn't possible

================================================================================
KEY COMPONENTS
================================================================================

SDCEnforcer: Main enforcement class
    - records: Dict[cell_id, SDCExecutionRecord] - execution history
    - checkpoints: Checkpoints - pre/post state snapshots
    - cell_order: List[cell_id] - document order from notebook

SDCExecutionRecord: Per-cell execution data
    - tracking: TrackingData - reads/writes at variable, column, structural levels
    - pre_checkpoint_name: Reference to pre-execution state
    - structural_reads_values: Captured values for error messages

ConflictResolver: Declarative conflict rule evaluation
    - Evaluates access events against change events
    - Applies rules in precedence order
    - Returns conflict/no-conflict decision with explanation

================================================================================
ALGORITHM OVERVIEW
================================================================================

On cell execution:
1. SDCEnforcer.check() receives:
   - cell_id, cell_order (document position)
   - tracking_data (access record from TrackingDict)
   - pre_checkpoint, post_checkpoint (state snapshots)

2. Backward mutation check:
   - Diff pre vs post to find actual changes
   - For each earlier cell that was previously executed:
     - Check if changes conflict with that cell's reads
     - Apply conflict resolution hierarchy
   - If conflict found: return violation, caller rolls back

3. Staleness computation:
   - For each previously executed cell:
     - Compare current state vs that cell's pre-checkpoint
     - If any read variable differs: cell is stale
   - Return set of stale cell IDs for UI display

4. Record keeping:
   - Store execution record for this cell
   - Capture structural read values for future error messages

See analysis.md for formal specification and pseudocode algorithms.


================================================================================
DEEP ALIAS DETECTION
================================================================================

When OPT_ACCESSED_VARS_ONLY is enabled (default), we only diff variables that
the cell actually accessed (reads + writes) plus their DEEP aliases.

WHY ALIAS DETECTION MATTERS
---------------------------
A deep alias is a variable that shares ANY internal reference with an accessed
variable - not just top-level object identity. For example:
  - If a["b"] and c["b"] point to the same object
  - If df1 and df2 share a column's underlying array
  - If x.attr and y.attr point to the same mutable object

This is critical for correctness: if cell C modifies a["b"]["f"], and c["b"]
is the same object as a["b"], then c also changed! We must diff c to detect
that change (for backward mutation checks).

ARCHITECTURE
------------
Alias detection uses precomputed indexes stored in Checkpoint objects:

  Checkpoint._reachable_ids: Dict[var_name, Set[obj_id]]
      All object IDs reachable from each variable via nested containers.

  Checkpoint._id_to_vars: Dict[obj_id, Set[var_name]]
      Reverse index: maps each object ID to variables containing it.

  Checkpoint._id_to_paths: Dict[obj_id, Dict[var_name, path_str]]
      Path tracking for detailed logging (e.g., "a['b'] ↔ c['b']").

The index is built LAZILY on first query (via get_aliases_for_vars) and
provides O(accessed + aliases) lookup instead of O(total_objects_in_namespace).

KEY FUNCTION
------------
_expand_with_deep_aliases(accessed_vars, pre_checkpoint, log_aliases=True)
  - Takes set of accessed variable names and the pre-execution checkpoint
  - Returns expanded set including all deep aliases
  - Uses pre-state checkpoint because alias relationships existed before cell ran

WHAT GETS TRACKED
-----------------
  - Containers: dict, list, tuple, set, frozenset
  - Pandas: DataFrame (via _mgr), Series, object-dtype columns
  - NumPy: ndarray (via .base for views), object-dtype arrays
  - Custom objects: via __dict__ and __slots__

WHAT GETS SKIPPED
-----------------
  - Immutable atomics: None, bool, int, float, str, bytes
  - Temporary objects: .values, .data (id can be reused after GC)

See checkpoint.py section 12 for full implementation details.

================================================================================
CUDF AND KERAS SUPPORT
================================================================================

cuDF Objects
------------
cuDF (GPU DataFrames) are transparently handled:
- Checkpoints convert cuDF to pandas via cudf_compat.to_pandas()
- Works with both native cuDF and cudf.pandas proxy objects
- Diff comparisons operate on pandas representations

Keras Models
------------
Keras models use the opaque object pattern (see checkpoint.py section 14):
- Only weights are checkpointed, not internal TensorFlow objects
- Deferred Keras import avoids ~3s penalty for non-Keras notebooks
- _is_keras_model() detects Keras via module inspection

================================================================================
PERFORMANCE TUNING
================================================================================

Checkpoint.diff() Timers
------------------------
The diff operation includes timing phases for debugging:
- [diff] Setup - Import and initialization
- [diff] Create Diff object - Comparator construction
- [diff] Compare namespaces - Actual comparison

Enable FERRET_PROFILE_DIFF=1 for per-variable timing breakdowns.

Optimization Flags
------------------
Controlled via environment variables:
- FERRET_OPT_CONFLICT_LOOP_SKIP: Skip O(n) loop when no overlap (default: on)
- FERRET_OPT_ACCESSED_VARS_ONLY: Only diff accessed vars + aliases (default: on)
"""

import os
import pprint
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData
from data_ferret.kernel.structural_tracking import StructuralTrackingMode
from data_ferret.kernel.types import DiffResult, DiffNode, ValueComparison, CompoundDiff
from data_ferret.util.cell_index import index_to_alpha

from .models import SDCExecutionRecord, SDCResult, SDCViolation

# Conflict resolution imports
from .access_events import StructuralRead, VariableRead
from .conflict_resolver import ConflictResolver
from .conflict_rules import StructuralMode
from .change_detector import detect_changes
from data_ferret.util.output import output, timer

# Checkpoint naming constants
PRE_CHECKPOINT_PREFIX = "_pre_"
POST_CHECKPOINT_PREFIX = "_post_"

# ============================================================================
# OPTIMIZATION FLAGS (controlled via environment variables)
# ============================================================================
# Set to "0" or "false" to disable; any other value (or unset) enables

def _env_flag(name: str, default: bool = True) -> bool:
    """Check environment variable for optimization flag."""
    val = os.environ.get(name, "").lower()
    if val in ("0", "false", "no", "off"):
        return False
    return default

# OPT_CONFLICT_LOOP_SKIP: Skip the O(n) conflict detection loop when there's
# no variable-level overlap between changed variables and prior reads.
OPT_CONFLICT_LOOP_SKIP = _env_flag("FERRET_OPT_CONFLICT_LOOP_SKIP", default=True)

# OPT_ACCESSED_VARS_ONLY: Only diff variables that the cell actually accessed
# (reads + writes) plus their aliases, instead of diffing the entire namespace.
# This can provide 5-10x speedup when cells access few variables.
OPT_ACCESSED_VARS_ONLY = _env_flag("FERRET_OPT_ACCESSED_VARS_ONLY", default=True)


def _expand_with_deep_aliases(
    accessed_vars: Set[str],
    pre_checkpoint: Checkpoint,
    log_aliases: bool = True,
) -> Set[str]:
    """
    Expand a set of accessed variable names to include all DEEP aliases.

    A deep alias is a variable that shares ANY internal reference with an
    accessed variable - not just top-level identity. For example:
    - If a["b"] and c["b"] point to the same object
    - If df1 and df2 share a column's underlying array
    - If x.attr and y.attr point to the same mutable object

    This is critical for correctness: if cell C modifies a["b"]["f"], and
    c["b"] is the same object as a["b"], then c also changed! We need to
    diff c to detect that it changed (for backward mutation checks).

    We use the pre-state checkpoint's precomputed alias index because:
    1. Alias relationships existed before the cell ran
    2. The index was built once during checkpoint creation (immutable)
    3. Lookup is O(accessed + aliases) instead of O(total_objects)

    Args:
        accessed_vars: Set of variable names the cell accessed (reads + writes)
        pre_checkpoint: The pre-execution checkpoint (has precomputed alias index)
        log_aliases: If True, log discovered alias relationships

    Returns:
        Expanded set including accessed_vars plus all their deep aliases
    """
    # Use the checkpoint's precomputed deep alias index
    return pre_checkpoint.get_aliases_for_vars(accessed_vars, log_aliases=log_aliases)


def _tracking_mode_to_structural_mode(mode: StructuralTrackingMode) -> StructuralMode:
    """Convert StructuralTrackingMode to StructuralMode for the new resolver."""
    if mode == StructuralTrackingMode.ENFORCE:
        return StructuralMode.ENFORCE
    elif mode == StructuralTrackingMode.WARN:
        return StructuralMode.WARN
    else:
        return StructuralMode.OFF


class SDCEnforcer:
    """
    Enforces Sequential Dataflow Consistency.

    Tracks cell executions and their read/write sets.
    On each execution, checks for backward mutations and computes staleness.

    Supports structural tracking mode for detecting structural changes
    (like df.columns, df.shape) when those attributes were read.
    """

    def __init__(
        self,
        checkpoints: Checkpoints,
        structural_mode: StructuralTrackingMode = StructuralTrackingMode.WARN,
    ):
        self.checkpoints = checkpoints
        self.records: Dict[str, SDCExecutionRecord] = {}
        self.seq_counter: int = 0
        self._cell_order: List[str] = []
        self._stale_cells: Set[str] = set()  # Cache for absolute staleness state
        self._structural_mode = structural_mode
        # New declarative conflict resolver
        self._conflict_resolver = ConflictResolver(
            structural_mode=_tracking_mode_to_structural_mode(structural_mode)
        )

    @property
    def structural_mode(self) -> StructuralTrackingMode:
        """Get the current structural tracking mode."""
        return self._structural_mode

    def set_structural_mode(self, mode: StructuralTrackingMode) -> None:
        """Set the structural tracking mode."""
        self._structural_mode = mode
        # Update the conflict resolver's mode too
        self._conflict_resolver = ConflictResolver(
            structural_mode=_tracking_mode_to_structural_mode(mode)
        )

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
        namespace: Optional[dict] = None,
    ) -> SDCResult:
        """
        Main entry point. Call after cell execution.

        Args:
            cell_id: ID of the cell that just executed
            pre_checkpoint: Snapshot of namespace before execution
            post_checkpoint: Snapshot of namespace after execution
            tracking: TrackingData with reads/writes
            continue_on_violation: If True, compute staleness even when violation detected
            namespace: Optional user namespace for capturing structural read values

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

        # Rule 3: Check backward mutation (also returns diff and typed_changes for reuse)
        violation = None
        current_diff = None
        typed_changes = []
        if my_position >= 0:
            with timer(key="sdc:backward_mutation", message=f"[sdc] Backward mutation check for {cell_id}") as t:
                violation, current_diff, typed_changes = self._check_backward_mutation(
                    cell_id, my_position, pre_checkpoint, post_checkpoint, tracking
                )

            if violation is not None:
                output.add_timing("sdc:backward_mutation_violation", t.duration())
            else:
                output.add_timing("sdc:backward_mutation_no_violation", t.duration())


        # Check forward dependency (reading from later cells that already executed)
        forward_violation = None
        if my_position >= 0:
            with timer(key="sdc:forward_dependency", message=f"[sdc] Forward dependency check for {cell_id}") as t:
                forward_violation = self._check_forward_dependency(
                    cell_id, my_position, tracking
                )

            if forward_violation is not None:
                output.add_timing("sdc:forward_dependency_violation", t.duration())
            else:
                output.add_timing("sdc:forward_dependency_no_violation", t.duration())

        stale = []
        changed_vars = []
        column_changed = {}
        structural_warnings = []

        # Extract structural warnings from diff (always, even with violations)
        # This ensures users see warnings about structural changes regardless of
        # whether there's also a violation
        if current_diff is not None and current_diff.warnings:
            structural_warnings = list(current_diff.warnings)

        if not violation or continue_on_violation:
            # Capture structural read values for better error messages later
            structural_read_values = {}
            if namespace is not None and tracking.structural_reads:
                structural_read_values = capture_structural_read_values(
                    namespace, tracking.structural_reads
                )

            # Update our record for this cell BEFORE computing staleness
            # (so this cell is considered "fresh" in the computation)
            # typed_changes is cached for fast forward dependency checks
            self.records[cell_id] = SDCExecutionRecord(
                cell_id=cell_id,
                tracking=tracking,
                execution_seq=self.seq_counter,
                structural_reads_values=structural_read_values,
                typed_changes=typed_changes,
            )

            # This cell just executed, so it's now fresh
            self._stale_cells.discard(cell_id)

            # Reuse diff from backward mutation check, or compute if not available
            if current_diff is None:
                # Note: Don't pass current cell's structural_reads - intra-cell
                # structural reads are not backward mutations. Structural warnings
                # for prior cells are handled in staleness computation.
                current_diff = Checkpoint.diff(
                    pre_checkpoint,
                    post_checkpoint,
                    use_leq=True,
                    column_rbw=tracking.column_reads_before_writes,
                    structural_reads={},  # Empty - no intra-cell structural warnings
                    structural_mode=self._structural_mode,
                )

            # Check if diff was truncated - if so, return violation
            truncated_vars = _check_for_truncation(current_diff)
            if truncated_vars:
                with timer(key="violation:truncation_check", message=f"[violation] Building truncation violation (check)"):
                    formatted_diff = _format_diff_for_display(current_diff, truncated_vars)
                    mutating_alpha = self._cell_id_to_alpha(cell_id)
                    return SDCResult(
                        violation=SDCViolation(
                            mutating_cell=cell_id,
                            affected_cell=cell_id,
                            variables=truncated_vars,
                            message=format_truncation_error(mutating_alpha, truncated_vars),
                            truncation_details=formatted_diff,
                        ),
                        stale_cells=[],
                        changed_variables=[],
                        column_changed={},
                        structural_warnings=structural_warnings,
                    )

            if current_diff.differences:
                changed_vars = list(current_diff.differences.keys())

            # Extract column-level changes from diff result
            column_changed = _extract_column_changes(current_diff, tracking)

            # Update staleness INCREMENTALLY (only check cells below that might have become stale)
            # Also captures structural warnings from affected cells
            with timer(key="sdc:staleness", message=f"[sdc] Staleness update for {cell_id}"):
                stale, staleness_warnings = self._update_staleness_incremental(
                    post_checkpoint, set(changed_vars), column_changed, cell_id, my_position
                )
            # Merge warnings from staleness checks
            structural_warnings.extend(staleness_warnings)

        return SDCResult(
            violation=violation,
            stale_cells=stale,
            changed_variables=changed_vars,
            column_changed=column_changed,
            structural_warnings=structural_warnings,
            forward_violation=forward_violation,
        )

    def _check_backward_mutation(
        self,
        cell_id: str,
        my_position: int,
        pre_checkpoint: Checkpoint,
        post_checkpoint: Checkpoint,
        tracking: TrackingData,
    ) -> Tuple[Optional[SDCViolation], DiffResult, List]:
        """
        Check if current cell causes a backward mutation (Rule 3 violation).

        A backward mutation occurs when a cell modifies a variable that an
        earlier cell (in notebook order) reads. This prevents hidden dependencies
        where earlier cells depend on later cells having run first.

        Includes column-aware conflict detection for DataFrames: modifying
        df['price'] doesn't conflict with a cell that only reads df['quantity'].

        Returns:
            Tuple of (violation, diff_result, typed_changes):
            - diff_result is returned for reuse by caller
            - typed_changes is cached in the record for fast forward dependency checks
        """
        # Compute what THIS cell actually modified
        # For diff detection, check all accessed columns (read OR written)
        with timer(key="bwm:prepare_columns", message=f"[bwm] Prepare accessed columns"):
            all_accessed_columns = {}
            for var, cols in tracking.column_reads_before_writes.items():
                all_accessed_columns[var] = set(cols)
            for var, cols in tracking.column_writes.items():
                if var in all_accessed_columns:
                    all_accessed_columns[var].update(cols)
                else:
                    all_accessed_columns[var] = set(cols)

        # Compute diff without structural_reads - intra-cell structural reads
        # are not backward mutations. Structural conflict detection is handled
        # by the ConflictResolver which uses prior cells' structural reads.
        # Note: use_leq=False to detect created variables too - these are cached
        # for forward dependency checks. Created variables won't cause false
        # positive backward mutations (earlier cells couldn't have read them).

        # ======================================================================
        # OPTIMIZATION: OPT_ACCESSED_VARS_ONLY
        # Only diff variables that the cell accessed (reads + writes) plus their
        # aliases, instead of diffing the entire namespace. This can provide
        # 5-10x speedup when cells access few variables out of many.
        # ======================================================================
        keys_to_include: Optional[Set[str]] = None
        if OPT_ACCESSED_VARS_ONLY:
            with timer(key="bwm:expand_aliases", message=f"[bwm] OPT expand accessed vars with DEEP aliases"):
                # Get variables this cell accessed (reads + writes)
                # Note: reads_before_writes is a Set[str], writes is also Set[str]
                accessed_vars = set(tracking.reads_before_writes) | set(tracking.writes)

                # Expand to include DEEP aliases (vars sharing ANY internal reference)
                # Uses precomputed index from checkpoint (O(accessed + aliases))
                keys_to_include = _expand_with_deep_aliases(accessed_vars, pre_checkpoint)

                # Log the optimization impact with variable names
                total_vars = len(pre_checkpoint.user_ns)
                all_vars_sorted = sorted(pre_checkpoint.user_ns.keys())
                from data_ferret.util.output import log
                accessed_sorted = sorted(accessed_vars)
                expanded_sorted = sorted(keys_to_include)
                log(f"[bwm] OPT_ACCESSED_VARS_ONLY: diffing {len(keys_to_include)} of {total_vars} vars")
                log(f"[bwm]   all_vars={all_vars_sorted}")
                log(f"[bwm]   accessed={accessed_sorted}")
                log(f"[bwm]   deep_aliases_added={sorted(set(expanded_sorted) - set(accessed_sorted))}")
        # ======================================================================

        with timer(key="bwm:checkpoint_diff", message=f"[bwm] Checkpoint.diff (pre vs post)"):
            current_diff = Checkpoint.diff(
                pre_checkpoint,
                post_checkpoint,
                keys_to_include=keys_to_include,
                use_leq=False,  # Detect creations for forward dependency caching
                column_rbw=all_accessed_columns,
                structural_reads={},  # Empty - ConflictResolver handles this
                structural_mode=self._structural_mode,
            )

        # Check if diff was truncated - if so, return violation
        with timer(key="bwm:check_truncation", message=f"[bwm] Check truncation"):
            truncated_vars = _check_for_truncation(current_diff)
        if truncated_vars:
            with timer(key="violation:truncation", message=f"[violation] Building truncation violation"):
                formatted_diff = _format_diff_for_display(current_diff, truncated_vars)
                mutating_alpha = self._cell_id_to_alpha(cell_id)
                return (
                    SDCViolation(
                        mutating_cell=cell_id,
                        affected_cell=cell_id,
                        variables=truncated_vars,
                        message=format_truncation_error(mutating_alpha, truncated_vars),
                        truncation_details=formatted_diff,
                    ),
                    current_diff,
                    [],  # No typed_changes on truncation
                )

        if not current_diff.differences:
            # This cell didn't modify anything, no backward mutation possible
            return (None, current_diff, [])

        # Column-level modifications (for DataFrames)
        with timer(key="bwm:extract_columns", message=f"[bwm] Extract column changes"):
            modified_columns = _extract_column_changes(current_diff, tracking)

        # Convert diff to typed Changes for conflict detection
        # These are also cached in the record for forward dependency checks
        with timer(key="bwm:detect_changes", message=f"[bwm] detect_changes (diff -> typed)"):
            typed_changes = detect_changes(current_diff)
        if not typed_changes:
            return (None, current_diff, [])

        # ======================================================================
        # OPTIMIZATION: OPT_CONFLICT_LOOP_SKIP
        # Skip the O(n) conflict detection loop when there's no variable-level
        # overlap between changed variables and prior cell reads.
        # ======================================================================
        if OPT_CONFLICT_LOOP_SKIP:
            with timer(key="bwm:opt_skip_check", message=f"[bwm] OPT skip check ({my_position} prior cells)"):
                changed_var_names = {c.variable for c in typed_changes}
                all_prior_var_reads: Set[str] = set()
                for prior_cell_id_check in self._cell_order[:my_position]:
                    prior_record_check = self.records.get(prior_cell_id_check)
                    if prior_record_check:
                        all_prior_var_reads.update(prior_record_check.tracking.reads_before_writes)

                # If no overlap at variable level, no conflict is possible
                has_overlap = bool(changed_var_names & all_prior_var_reads)

            if not has_overlap:
                return (None, current_diff, typed_changes)
        # ======================================================================

        # Check if any earlier cell reads something we modified
        with timer(key="bwm:conflict_loop", message=f"[bwm] Conflict detection loop ({my_position} prior cells)"):
            conflict_checks = 0
            for prior_cell_id in self._cell_order[:my_position]:
                prior_record = self.records.get(prior_cell_id)
                if prior_record is None:
                    continue

                # Convert prior cell's tracking to typed AccessEvents
                prior_reads = prior_record.tracking.to_read_events()
                if not prior_reads:
                    continue

                conflict_checks += 1

                # Use declarative ConflictResolver to detect conflicts
                violations = self._conflict_resolver.get_violations(typed_changes, prior_reads)
                if not violations:
                    continue

                # Extract conflict names in the format expected by messages
                # When the read is a VariableRead or StructuralRead, report just
                # the variable name (for backward compatibility with old behavior)
                conflicts = []
                for v in violations:
                    var = v.change.variable
                    # Check if the read was a VariableRead or StructuralRead
                    if isinstance(v.read, (VariableRead, StructuralRead)):
                        # Report just the variable name for variable/structural reads
                        conflicts.append(var)
                    elif hasattr(v.change, 'column'):
                        conflicts.append(f"{var}.{v.change.column}")
                    else:
                        conflicts.append(var)
                conflicts = sorted(set(conflicts))

                if conflicts:
                    with timer(key="violation:backward_mutation", message=f"[violation] Building backward mutation violation"):
                        mutating_alpha = self._cell_id_to_alpha(cell_id)
                        affected_alpha = self._cell_id_to_alpha(prior_cell_id)

                        # Get structural reads values from the prior record
                        prior_structural_values = prior_record.structural_reads_values

                        # Extract change descriptions from diff
                        changes = _extract_change_descriptions(current_diff, modified_columns)

                        # Build the detailed message
                        message = format_structural_violation(
                            mutating_alpha,
                            affected_alpha,
                            conflicts,
                            prior_structural_values,
                            changes,
                        )

                        return (
                            SDCViolation(
                                mutating_cell=cell_id,
                                affected_cell=prior_cell_id,
                                variables=conflicts,
                                message=message,
                                structural_reads_detail=prior_structural_values,
                                changes_detail=changes,
                            ),
                            current_diff,
                            typed_changes,
                        )

        return (None, current_diff, typed_changes)

    def _check_forward_dependency(
        self,
        cell_id: str,
        my_position: int,
        tracking: TrackingData,
    ) -> Optional[SDCViolation]:
        """
        Check if current cell reads from a later cell that already executed.

        A forward dependency occurs when a cell reads a variable that a later
        cell (in document order) has already written. This means the reading
        cell is seeing "future" state that wouldn't exist in top-to-bottom order.

        Uses cached typed_changes from each cell's execution record, avoiding
        expensive checkpoint diffs. The typed_changes are computed once during
        backward mutation check and cached for reuse here.

        Args:
            cell_id: ID of the cell that just executed (the reading cell)
            my_position: Position in document order
            tracking: TrackingData with reads for this cell

        Returns:
            SDCViolation with violation_type="forward_dependency" if detected, None otherwise
        """
        # Convert current cell's reads to typed AccessEvents
        my_read_events = tracking.to_read_events()
        if not my_read_events:
            return None

        # Check later cells (in document order) that already executed
        for later_cell_id in self._cell_order[my_position + 1:]:
            later_record = self.records.get(later_cell_id)
            if later_record is None:
                continue  # Later cell hasn't executed yet - OK

            # Use cached typed_changes from the later cell's execution
            # These were computed during backward mutation check and include
            # both modifications and creations (use_leq=False)
            later_changes = later_record.typed_changes
            if not later_changes:
                continue  # Later cell didn't actually change anything

            # Check if later cell's changes conflict with current cell's reads
            violations = self._conflict_resolver.get_violations(later_changes, my_read_events)
            if not violations:
                continue

            # Extract conflict names from violations
            conflicts = []
            for v in violations:
                var = v.change.variable
                if hasattr(v.change, 'column') and v.change.column:
                    conflicts.append(f"{var}['{v.change.column}']")
                else:
                    conflicts.append(var)
            conflicts = sorted(set(conflicts))

            if conflicts:
                with timer(key="violation:forward_dependency", message=f"[violation] Building forward dependency violation"):
                    reading_alpha = self._cell_id_to_alpha(cell_id)
                    writing_alpha = self._cell_id_to_alpha(later_cell_id)

                    message = format_forward_dependency_message(
                        reading_alpha, writing_alpha, conflicts
                    )

                    return SDCViolation(
                        mutating_cell=later_cell_id,
                        affected_cell=cell_id,
                        variables=conflicts,
                        message=message,
                        violation_type="forward_dependency",
                    )

        return None

    def _update_staleness_incremental(
        self,
        current_checkpoint: Checkpoint,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
    ) -> Tuple[List[str], List[str]]:
        """
        Incrementally update staleness cache (Rule 2 computation).

        Only checks cells BELOW the executed cell in document order (forward propagation).
        Cells above are not checked - backward dependencies are handled by violation detection.

        Skips cells that don't need checking:
        - Cells already marked stale
        - Cells whose reads don't overlap with changed variables/columns

        Args:
            current_checkpoint: The current state of the namespace
            changed_vars: Set of variable names that changed in this execution
            column_changed: Dict mapping var names to lists of changed column names
            just_executed: The cell_id that just executed (already marked fresh)
            my_position: Position of the executed cell in document order

        Returns:
            Tuple of:
            - List of all currently stale cell IDs (in document order)
            - List of structural warnings from affected cells
        """
        all_warnings: List[str] = []
        cells_checked = 0
        cells_skipped_stale = 0
        cells_skipped_no_overlap = 0
        diffs_performed = 0

        # Only check cells BELOW the executed cell (forward staleness only)
        cells_below = self._cell_order[my_position + 1:]
        for cell_id in cells_below:
            record = self.records.get(cell_id)
            if record is None:
                continue  # Cell below hasn't executed yet

            if cell_id in self._stale_cells:
                cells_skipped_stale += 1
                continue  # Already stale, no need to re-check

            # Skip cells whose reads don't overlap with changed vars
            if not self._has_relevant_overlap(record, changed_vars, column_changed):
                cells_skipped_no_overlap += 1
                continue

            cells_checked += 1

            # Cell MIGHT be stale - do the expensive diff check
            pre_checkpoint = self.checkpoints.get(f"{PRE_CHECKPOINT_PREFIX}{cell_id}")
            if pre_checkpoint is None:
                continue

            diffs_performed += 1
            with timer(key="sdc:staleness_diff", message=f"[staleness] Diff for cell {cell_id}"):
                diff_result = Checkpoint.diff(
                    pre_checkpoint,
                    current_checkpoint,
                    keys_to_include=record.tracking.reads_before_writes,
                    use_leq=True,
                    column_rbw=record.tracking.column_reads_before_writes,
                    structural_reads=record.tracking.structural_reads,
                    structural_mode=self._structural_mode,
                )

            if diff_result.differences:
                self._stale_cells.add(cell_id)

            # Capture warnings (these come from WARN mode structural tracking)
            if diff_result.warnings:
                affected_alpha = self._cell_id_to_alpha(cell_id)
                mutating_alpha = self._cell_id_to_alpha(just_executed)

                for warning in diff_result.warnings:
                    # Parse the warning to extract variable name and changes
                    # Format: "Structural change at var_name: details (read: attrs)"
                    var_match = re.match(r"Structural change at (\w+):", warning)
                    if var_match:
                        var_name = var_match.group(1)
                        # Get saved structural values from the affected cell's record
                        read_values = record.structural_reads_values.get(var_name, {})
                        # Extract change descriptions
                        changes = []
                        if "Columns added:" in warning:
                            match = re.search(r"Columns added: \[([^\]]+)\]", warning)
                            if match:
                                changes.append(f"Column(s) added: [{match.group(1)}]")
                        if "Rows added:" in warning:
                            match = re.search(r"Rows added: (\d+)", warning)
                            if match:
                                changes.append(f"Row(s) added: {match.group(1)}")
                        if "Shape:" in warning:
                            match = re.search(r"Shape: (\([^)]+\)) → (\([^)]+\))", warning)
                            if match:
                                changes.append(f"Shape: {match.group(1)} → {match.group(2)}")
                        if not changes:
                            # Fallback: use the raw warning detail
                            detail_match = re.search(r"Structural change at \w+: (.+)$", warning)
                            if detail_match:
                                changes.append(detail_match.group(1))

                        # Format the detailed warning
                        formatted = format_structural_warning(
                            mutating_alpha,
                            affected_alpha,
                            var_name,
                            read_values,
                            changes,
                        )
                        all_warnings.append(formatted)
                    else:
                        # Fallback for unrecognized warning format
                        all_warnings.append(f"Cell {affected_alpha}: {warning}")

        # Return in document order
        stale_cells = [cid for cid in self._cell_order if cid in self._stale_cells]

        # Log summary
        from data_ferret.util.output import log
        log(f"[staleness] Summary: checked={cells_checked}, skipped_stale={cells_skipped_stale}, "
            f"skipped_no_overlap={cells_skipped_no_overlap}, diffs={diffs_performed}, "
            f"newly_stale={len(stale_cells) - cells_skipped_stale}")

        return stale_cells, all_warnings

    def _has_relevant_overlap(
        self,
        record: SDCExecutionRecord,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
    ) -> bool:
        """
        Check if cell's reads overlap with changes at variable, column, or structural level.

        Returns True if:
        - Cell reads a variable that changed AND either:
          - No column-level info available (conservative: assume overlap)
          - Column-level info shows actual column overlap
          - Cell has structural reads for this variable (structural changes possible)

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

            # Check if cell has structural reads for this variable
            # If columns were added/changed and cell read structure, it might be affected
            if var in record.tracking.structural_reads and changed_cols:
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
                structural_reads=record.tracking.structural_reads,
                structural_mode=self._structural_mode,
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


def _extract_change_descriptions(
    diff_result: DiffResult,
    modified_columns: Dict[str, List[str]],
) -> List[str]:
    """
    Extract human-readable change descriptions from diff result.

    Args:
        diff_result: The diff result from Checkpoint.diff()
        modified_columns: Dict of variable -> list of modified columns

    Returns:
        List of change description strings
    """
    changes: List[str] = []

    for var_name, diff_node in diff_result.differences.items():
        # Report column changes
        cols = modified_columns.get(var_name, [])
        if cols:
            if len(cols) == 1:
                changes.append(f"Column '{cols[0]}' modified")
            else:
                changes.append(f"Columns modified: {cols}")

        # Check for structural changes in the diff
        if isinstance(diff_node, CompoundDiff):
            # Look for shape/size/index changes in children
            for key, child in diff_node.children.items():
                if key == "._shape" and isinstance(child, ValueComparison):
                    changes.append(f"Shape: {child.value1} → {child.value2}")
                elif key == "._len" and isinstance(child, ValueComparison):
                    changes.append(f"Length: {child.value1} → {child.value2}")
                elif key == "._columns" and isinstance(child, ValueComparison):
                    # Don't duplicate if we already reported column changes
                    if not cols:
                        changes.append(f"Columns changed")

        # Check warnings from the diff (these contain structural change info)
        if diff_result.warnings:
            for warning in diff_result.warnings:
                # Parse out useful info from warnings like:
                # "Structural change at var: Columns added: ['y'] (read: columns, shape)"
                if var_name in warning and warning not in changes:
                    # Extract the change description part
                    if "Columns added:" in warning:
                        match = re.search(r"Columns added: \[([^\]]+)\]", warning)
                        if match:
                            changes.append(f"Column(s) added: [{match.group(1)}]")
                    elif "Rows added:" in warning:
                        match = re.search(r"Rows added: (\d+)", warning)
                        if match:
                            changes.append(f"Row(s) added: {match.group(1)}")

    return changes


def capture_structural_read_values(
    namespace: dict,
    structural_reads: Dict[str, Set[str]],
) -> Dict[str, Dict[str, str]]:
    """
    Capture the current values of structural attributes that were read.

    Args:
        namespace: The user namespace containing the variables
        structural_reads: Dict mapping var names to sets of structural attrs read

    Returns:
        Dict mapping var names to dicts of {attr_name: repr_value}
    """
    result: Dict[str, Dict[str, str]] = {}

    for var_name, attrs in structural_reads.items():
        obj = namespace.get(var_name)
        if obj is None:
            continue

        result[var_name] = {}
        for attr in attrs:
            try:
                value = getattr(obj, attr)
                # Truncate long values
                result[var_name][attr] = _truncate_value(value, max_len=80)
            except Exception:
                result[var_name][attr] = "<unavailable>"

    return result


def format_variable_list(variables: List[str]) -> str:
    """
    Format a list of variable names as a human-readable string (no brackets).

    Examples:
        ["df"] -> "df"
        ["df", "other"] -> "df and other"
        ["df", "other", "third"] -> "df, other, and third"
    """
    if len(variables) == 0:
        return ""
    elif len(variables) == 1:
        return variables[0]
    elif len(variables) == 2:
        return f"{variables[0]} and {variables[1]}"
    else:
        return ", ".join(variables[:-1]) + f", and {variables[-1]}"


def format_structural_violation(
    mutating_cell_alpha: str,
    affected_cell_alpha: str,
    variables: List[str],
    structural_reads_values: Dict[str, Dict[str, str]],
    changes: List[str],
) -> str:
    """
    Format a detailed structural violation message.

    Args:
        mutating_cell_alpha: Cell that caused violation (@A notation)
        affected_cell_alpha: Earlier cell whose reads were mutated (@A notation)
        variables: List of affected variable names
        structural_reads_values: Values of structural attrs at read time
        changes: List of change descriptions

    Returns:
        Formatted multi-line message string
    """
    lines = [
        "❌ SDC Violation: Backward Structural Mutation",
        "",
        f"Cell {mutating_cell_alpha} modified {format_variable_list(variables)} which Cell {affected_cell_alpha} (earlier) reads.",
    ]

    # What was read
    if structural_reads_values:
        lines.append("")
        lines.append(f"What Cell {affected_cell_alpha} read:")
        for var_name, attrs in structural_reads_values.items():
            for attr, value in sorted(attrs.items()):
                lines.append(f"  • {var_name}.{attr} → {value}")

    # What changed
    if changes:
        lines.append("")
        lines.append(f"What Cell {mutating_cell_alpha} changed:")
        for change in changes:
            lines.append(f"  • {change}")

    # Explanation
    lines.append("")
    lines.append(f"Why blocked: Cell {affected_cell_alpha} depends on Cell {mutating_cell_alpha} "
                 "having run first, breaking top-to-bottom reproducibility.")
    lines.append("")
    lines.append("Fix: Move the modification before the read, or avoid reading "
                 "structural attributes that will change.")

    return "\n".join(lines)


def format_structural_warning(
    mutating_cell_alpha: str,
    affected_cell_alpha: str,
    var_name: str,
    structural_reads_values: Dict[str, str],
    changes: List[str],
) -> str:
    """
    Format a detailed structural warning message.

    Args:
        mutating_cell_alpha: Cell that caused the change (@A notation)
        affected_cell_alpha: Earlier cell whose reads were affected (@A notation)
        var_name: Name of the affected variable
        structural_reads_values: Dict of {attr_name: value_repr} at read time
        changes: List of change descriptions

    Returns:
        Formatted multi-line message string
    """
    lines = [
        "⚠️ Structural Warning",
        "",
        f"Cell {mutating_cell_alpha} modified '{var_name}' which Cell {affected_cell_alpha} previously read.",
    ]

    # What was read
    if structural_reads_values:
        lines.append("")
        lines.append(f"What Cell {affected_cell_alpha} read:")
        for attr, value in sorted(structural_reads_values.items()):
            lines.append(f"  • {var_name}.{attr} → {value}")

    # What changed
    if changes:
        lines.append("")
        lines.append(f"What Cell {mutating_cell_alpha} changed:")
        for change in changes:
            lines.append(f"  • {change}")

    # Impact
    lines.append("")
    lines.append(f"Impact: Re-running from top will give Cell {affected_cell_alpha} different results.")

    return "\n".join(lines)


def format_forward_dependency_message(
    reading_cell_alpha: str,
    writing_cell_alpha: str,
    variables: List[str],
) -> str:
    """
    Format a forward dependency violation message.

    A forward dependency occurs when a cell reads a variable that a later cell
    (in document order) has already written. This means the reading cell is
    seeing "future" state that wouldn't exist in top-to-bottom order.

    Args:
        reading_cell_alpha: Cell that read the variable (@A notation)
        writing_cell_alpha: Later cell that wrote the variable (@A notation)
        variables: List of affected variable names

    Returns:
        Formatted multi-line message string
    """
    vars_str = format_variable_list(variables)

    lines = [
        "❌ SDC Violation: Forward Dependency",
        "",
        f"Cell {reading_cell_alpha} reads {vars_str} which was written by "
        f"Cell {writing_cell_alpha} (later in notebook).",
        "",
        f"In top-to-bottom order, Cell {writing_cell_alpha} runs after "
        f"Cell {reading_cell_alpha}, so {reading_cell_alpha} would see a "
        "different (or undefined) value.",
        "",
        "Why blocked: This out-of-order execution breaks reproducibility.",
        "",
        f"Fix: Re-run from Cell {reading_cell_alpha} downward, or move the "
        "write before the read.",
    ]

    return "\n".join(lines)


def format_truncation_error(
    cell_alpha: str,
    variables: List[str],
) -> str:
    """
    Format a truncation error message.

    Truncation occurs when the diff tree exceeds max_diffs_per_container,
    meaning we may miss some column/key changes. This is treated as an error
    because SDC tracking may be incomplete.

    Args:
        cell_alpha: Cell where truncation occurred (@A notation)
        variables: List of variable names that had truncated diffs

    Returns:
        Formatted multi-line message string
    """
    vars_str = format_variable_list(variables)

    lines = [
        "⚠️ SDC Warning: Diff Truncated",
        "",
        f"Cell {cell_alpha}: SDC diff was truncated for: {vars_str}.",
        "",
        "Tracking may be incomplete. Some column or key changes may not be detected.",
        "",
        "Fix: Consider increasing max_diffs_per_container or simplifying the data structure.",
    ]

    return "\n".join(lines)
