"""
Reproducibility Enforcer - Reproducibility enforcement.

================================================================================
OVERVIEW
================================================================================

This module implements the core reproducibility enforcement logic, responsible for:
1. Recording cell execution history with access patterns
2. Detecting backward mutation violations (Rule 3)
3. Computing staleness for downstream cells (Rule 2)
4. Managing checkpoints for rollback on violation

================================================================================
THE THREE REPRODUCIBILITY RULES
================================================================================

Rule 1: Reproducibility Invariant (Goal)
    A notebook is reproducible if running cells in document order from a fresh
    kernel always produces the same result. This is the property reproducibility guarantees.

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

ReproducibilityEnforcer: Main enforcement class
    - _notebook_state: NotebookState - single source of truth for formal model
    - checkpoints: Checkpoints - pre/post state snapshots
    - cell_order: List[cell_id] - document order from notebook

NotebookState: Single source of truth for formal model S = ⟨C, O, Σ, T, R, W, L⟩
    - status: T (Cell → CellStatus) - clean/stale per cell
    - reads/writes: R, W (Cell → P(Loc)) - per-cell reads and writes
    - last_writer: L (Loc → Cell) - provenance tracking
    - tracking_data: Per-cell TrackingData for conflict detection

ConflictResolver: Declarative conflict rule evaluation
    - Evaluates access events against change events
    - Applies rules in precedence order
    - Returns conflict/no-conflict decision with explanation

================================================================================
ALGORITHM OVERVIEW
================================================================================

On cell execution:
1. ReproducibilityEnforcer.check() receives:
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

Enable FLOWBOOK_PROFILE_DIFF=1 for per-variable timing breakdowns.

Optimization Flags
------------------
Controlled via environment variables:
- FLOWBOOK_OPT_CONFLICT_LOOP_SKIP: Skip O(n) loop when no overlap (default: on)
- FLOWBOOK_OPT_ACCESSED_VARS_ONLY: Only diff accessed vars + aliases (default: on)

================================================================================
STALENESS COMPUTATION MODES
================================================================================

The enforcer supports two staleness computation modes, controlled via
%staleness_mode magic command:

SYNTACTIC MODE
--------------
Use checkpoint once to compute accurate R and W, then discard it.

On cell i execution:
    1. Capture pre_checkpoint
    2. Execute cell, get tracking (reads_before_writes, writes)
    3. Compute W_i = { v : pre_checkpoint[v] ≠ namespace[v] }  # actual changes
    4. R_i = tracking.reads_before_writes
    5. Discard pre_checkpoint

Stored state per cell:
    - R[i]: Set[str] — variables read before write
    - W[i]: Set[str] — variables that actually changed

Predicates (pure set operations):
    - NoReadAndWrite:    R_i ∩ W_i = ∅
    - WriteBeforeRead:   R_i ⊆ W_{1..i-1}
    - NoReadBeforeWrite: R_i ∩ W_{i+1..n} = ∅
    - NoWriteAfterRead:  W_i ∩ R_j = ∅ for all clean j < i

Forward Staleness (cells after i):
    for j > i where j was executed:
        if W_i ∩ R_j ≠ ∅:
            mark j stale

Properties:
    - Monotonic: Once stale, stays stale until re-executed
    - Conservative: Over-approximates staleness
    - Memory: O(cells × |variables|) — just string sets

SEMANTIC MODE (default)
-----------------------
Use checkpoint to compute R and W, store pre-checkpoints for semantic comparison.

On cell i execution:
    1. Capture pre_checkpoint
    2. Execute cell, get tracking
    3. Compute W_i via diff
    4. R_i = tracking.reads_before_writes
    5. Store pre_checkpoint[i] permanently

Stored state per cell:
    - R[i], W[i]: Set[str] — for quick filtering
    - pre_checkpoint[i]: Checkpoint — actual values cell saw

Predicates (syntactic filter + semantic check):
    - NoWriteAfterRead: W_i ∩ R_j ≠ ∅ AND diff(pre_checkpoint[j], namespace, R_j) ≠ ∅

Forward Staleness (cells after i, semantic):
    for j > i where j was executed:
        if W_i ∩ R_j = ∅:
            continue  # Quick filter
        diff_result = diff(pre_checkpoint[j], namespace, keys=R_j)
        if diff_result.differences:
            mark j stale
        else:
            mark j clean  # Converged!

Convergence detection:
    After any cell execution, check all stale cells:
        if diff(pre_checkpoint[j], namespace, R_j) is empty:
            mark j clean  # Inputs match what j originally saw

Properties:
    - Non-monotonic: Staleness can be cleared when values converge
    - Precise: Only marks stale when values actually differ
    - Memory: O(cells × values) — stores checkpoint values

COMPARISON
----------
| Aspect              | Syntactic           | Semantic              |
|---------------------|---------------------|-----------------------|
| Checkpoint storage  | Discard after W     | Keep permanently      |
| Staleness check     | W_i ∩ R_j ≠ ∅       | diff(pre_ckpt, ns)    |
| NoWriteAfterRead    | Set intersection    | + convergence check   |
| Un-staleness        | Never (monotonic)   | When values converge  |
| Memory cost         | Low (sets only)     | High (values)         |
| False positives     | More (conservative) | Fewer (precise)       |

See FORMAL_DEVELOPMENT.md §10 for the full formal specification.
"""

import os
import pprint
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from flowbook.kernel_support.checkpoint import Checkpoint, CheckpointDiffResult
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode, StalenessMode
from flowbook.kernel_support.types import MemoryCheckpointDiffResult, DiffNode, ValueComparison, CompoundDiff
from flowbook.util.cell_index import index_to_alpha

from flowbook.kernel.models import (
    CellStateSnapshot,
    ErrorType,
    MovedCell,
    OrderChangeResult,
    OrderDelta,
    Reason,
    ReasonType,
    ReproducibilityError,
    ReproducibilityResult,
    ReproducibilityViolation,
)
from flowbook.kernel.notebook_state import NotebookState

# Conflict resolution imports
from flowbook.kernel.access_events import StructuralRead, VariableRead
from flowbook.kernel.conflict_resolver import ConflictResolver
from flowbook.kernel.conflict_rules import StructuralMode
from flowbook.kernel.change_detector import detect_changes
from flowbook.util.output import output, timer

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
OPT_CONFLICT_LOOP_SKIP = _env_flag("FLOWBOOK_OPT_CONFLICT_LOOP_SKIP", default=True)

# OPT_ACCESSED_VARS_ONLY: Only diff variables that the cell actually accessed
# (reads + writes) plus their aliases, instead of diffing the entire namespace.
# This can provide 5-10x speedup when cells access few variables.
OPT_ACCESSED_VARS_ONLY = _env_flag("FLOWBOOK_OPT_ACCESSED_VARS_ONLY", default=True)

# ENABLE_SKIPPED_UPSTREAM: When True, checks if cells read from the "wrong" writer
# (runtime provenance differs from expected notebook-order provenance). This is a
# UX convenience that warns users proactively when cells are executed out of order.
# When False (default), ForwardStale handles these cases reactively when the
# skipped cell is eventually executed. Disabling simplifies the model without
# losing soundness.
ENABLE_SKIPPED_UPSTREAM = _env_flag("FLOWBOOK_ENABLE_SKIPPED_UPSTREAM", default=False)


# ============================================================================
# FORMAL PREDICATE HELPERS
# ============================================================================
# These functions implement the formal predicates from main.tex and
# FORMAL_DEVELOPMENT.md §3.2-3.3. They provide a direct mapping between
# the formal specification and the implementation.
#
# Notation:
#   R, W = read/write sets indexed by cell position
#   i, j = cell positions in document order
#   n = total number of cells
#
# Location types (Loc):
#   Var(x)           - Variable
#   Col(df, c)       - DataFrame column
#   File(path)       - File path
#   Structural(df, a) - Structural attribute
#
# The predicates below work with Set[str] for backward compatibility.
# For full Loc-based predicates, see flowbook.kernel.models:
#   tracking_to_read_locs(), tracking_to_write_locs(), locs_intersect()
# ============================================================================


def _writes_in_range(
    notebook_state: "NotebookState",
    cell_order: List[str],
    start: int,
    end: int,
) -> Set[str]:
    """
    Compute W_{start..end} = ⋃_{k ∈ [start..end]} Wₖ

    Formal ref: FORMAL_DEVELOPMENT.md §1.3
    """
    result: Set[str] = set()
    for k in range(start, min(end + 1, len(cell_order))):
        cell_id = cell_order[k]
        tracking = notebook_state.get_tracking(cell_id)
        if tracking is not None:
            result.update(tracking.writes)
    return result


def _reads_in_range(
    notebook_state: "NotebookState",
    cell_order: List[str],
    start: int,
    end: int,
) -> Set[str]:
    """
    Compute R_{start..end} = ⋃_{k ∈ [start..end]} Rₖ

    Formal ref: FORMAL_DEVELOPMENT.md §1.3
    """
    result: Set[str] = set()
    for k in range(start, min(end + 1, len(cell_order))):
        cell_id = cell_order[k]
        tracking = notebook_state.get_tracking(cell_id)
        if tracking is not None:
            result.update(tracking.reads_before_writes)
    return result


def _overwritten(
    notebook_state: "NotebookState",
    cell_order: List[str],
    i: int,
) -> Set[str]:
    """
    Overwritten(W, i) ≝ W_{i+1..n}

    The set of locations written by cells after position i.

    Formal ref: main.tex Definition (Overwritten), FORMAL_DEVELOPMENT.md §1.4.1
    """
    return _writes_in_range(notebook_state, cell_order, i + 1, len(cell_order) - 1)


def _forward_stale(
    R_j: Set[str],
    W_j: Set[str],
    W_i_old: Set[str],
    W_i_new: Set[str],
    i: int,
    j: int,
) -> bool:
    """
    ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅

    Cell j (after i) becomes stale if cell i's old OR new writes overlap with
    what cell j reads or writes.

    Formal ref: main.tex §Staleness predicates, FORMAL_DEVELOPMENT.md §3.3

    Args:
        R_j: Read set of cell j
        W_j: Write set of cell j
        W_i_old: Old write set of cell i (Wᵢ, from previous execution)
        W_i_new: New write set of cell i (W'ᵢ, from current execution)
        i: Position of executing cell
        j: Position of cell to check

    Returns:
        True if cell j should become stale due to cell i's execution
    """
    if j <= i:
        return False
    W_i_union = W_i_old | W_i_new
    return bool(W_i_union & (R_j | W_j))


def _backward_stale(
    W_old: Dict[str, Set[str]],
    W_new_i: Set[str],
    W_old_i: Set[str],
    last_writer_func,
    i: int,
    j: int,
) -> bool:
    """
    BackwardStale(W, W', i, j) ≝ j < i ∧ j = LastWriter(W, i, y) for some y ∈ Wᵢ \\ W'ᵢ

    Cell j (before i) becomes stale if it was the last writer of a location
    that cell i no longer writes (i.e., i's write set shrank).

    Formal ref: main.tex §Staleness predicates, FORMAL_DEVELOPMENT.md §3.3

    Args:
        W_old: Old write sets by cell_id
        W_new_i: New write set of cell i
        W_old_i: Old write set of cell i
        last_writer_func: Function(var, cell_id) -> last writer cell_id
        i: Position of executing cell
        j: Position of cell to check

    Returns:
        True if cell j should become stale due to cell i's changed writes
    """
    if j >= i:
        return False
    # Find locations that i used to write but no longer writes
    removed_writes = W_old_i - W_new_i
    for y in removed_writes:
        # Check if j was the last writer of y before cell i
        writer = last_writer_func(y, i)
        if writer is not None and writer == j:
            return True
    return False



def _write_before_read(
    R_i: Set[str],
    W_before_i: Set[str],
) -> bool:
    """
    WriteBeforeRead(R, W, i) ≝ Rᵢ ⊆ W_{1..i-1}

    All reads come from writes by earlier cells.

    Formal ref: main.tex §Validity predicates, FORMAL_DEVELOPMENT.md §3.2

    Args:
        R_i: Read set of cell i
        W_before_i: Union of write sets of cells 1..i-1

    Returns:
        True if the validity predicate holds
    """
    return R_i <= W_before_i


def _no_read_before_write(
    R_i: Set[str],
    W_after_i: Set[str],
) -> bool:
    """
    NoReadBeforeWrite(R, W, i) ≝ Rᵢ ∩ W_{i+1..n} = ∅

    Cell i does not read locations that will be written by later cells.
    (This detects forward contamination at the variable level.)

    Formal ref: main.tex §Validity predicates, FORMAL_DEVELOPMENT.md §3.2

    Args:
        R_i: Read set of cell i
        W_after_i: Union of write sets of cells i+1..n

    Returns:
        True if the validity predicate holds
    """
    return not bool(R_i & W_after_i)


def _no_write_after_read(
    W_i: Set[str],
    R_before_i: Set[str],
) -> bool:
    """
    NoWriteAfterRead(R, W, i) ≝ Wᵢ ∩ R_{1..i-1} = ∅

    Cell i does not write locations that earlier cells read.
    (This is the backward mutation check at the variable level.)

    Formal ref: main.tex §Validity predicates, FORMAL_DEVELOPMENT.md §3.2

    Args:
        W_i: Write set of cell i
        R_before_i: Union of read sets of cells 1..i-1

    Returns:
        True if the validity predicate holds (no backward mutation)
    """
    return not bool(W_i & R_before_i)


def _no_read_and_write(
    R_i: Set[str],
    W_i: Set[str],
) -> bool:
    """
    NoReadAndWrite(R, W, i) ≝ Rᵢ ∩ Wᵢ = ∅

    Cell i does not both read and write the same location.
    (This simplifies reasoning; actual impl allows read-then-write.)

    Formal ref: main.tex §Validity predicates, FORMAL_DEVELOPMENT.md §3.2

    Note: The implementation uses reads_before_writes which already
    excludes locations that are written before being read.

    Args:
        R_i: Read set of cell i
        W_i: Write set of cell i

    Returns:
        True if the validity predicate holds
    """
    return not bool(R_i & W_i)


# ============================================================================
# ALIAS EXPANSION
# ============================================================================


def _expand_with_deep_aliases(
    accessed_vars: Set[str],
    pre_checkpoint,
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
        pre_checkpoint: The pre-execution checkpoint (Checkpoint or Checkpoint)
        log_aliases: If True, log discovered alias relationships

    Returns:
        Expanded set including accessed_vars plus all their deep aliases
    """
    # Use the checkpoint's precomputed deep alias index
    # Checkpoint delegates get_aliases_for_vars to its memory checkpoint
    return pre_checkpoint.get_aliases_for_vars(accessed_vars, log_aliases=log_aliases)


def _expand_var_set_dict_with_aliases(
    var_set_dict: Dict[str, Set[str]],
    pre_checkpoint,
) -> Dict[str, Set[str]]:
    """
    Expand a var->set dict to include all aliases of each variable.

    Used for column reads/writes and structural reads where we need to
    record that accessing p['col'] or p.shape implicitly accesses
    x['col'] or x.shape for any alias x of p.

    Example:
        If p and x are aliases for the same DataFrame, and
        var_set_dict = {'p': {'col1'}}, returns {'p': {'col1'}, 'x': {'col1'}}.

    Args:
        var_set_dict: Dict mapping var names to sets of attributes/columns
        pre_checkpoint: Pre-execution checkpoint for alias lookup

    Returns:
        Expanded dict including all aliases with their attributes/columns
    """
    if not var_set_dict or pre_checkpoint is None:
        return var_set_dict

    # Get all aliases for variables in the dict
    vars_to_expand = set(var_set_dict.keys())
    expanded_vars = pre_checkpoint.get_aliases_for_vars(vars_to_expand, log_aliases=False)

    # Find new aliases (vars in expanded_vars but not in original)
    new_aliases = expanded_vars - vars_to_expand

    if not new_aliases:
        return var_set_dict

    # Build expanded dict - start with copy of original
    result: Dict[str, Set[str]] = {k: set(v) for k, v in var_set_dict.items()}

    # For each new alias, determine which original var it aliases
    # and copy that var's attributes to the alias
    for alias in new_aliases:
        # Find which original vars this alias shares IDs with
        alias_set = pre_checkpoint.get_aliases_for_vars({alias}, log_aliases=False)
        for orig_var in vars_to_expand:
            if orig_var in alias_set:
                # alias is an alias of orig_var - copy attributes
                result[alias] = set(var_set_dict[orig_var])
                break

    return result


def _tracking_mode_to_structural_mode(mode: StructuralTrackingMode) -> StructuralMode:
    """Convert StructuralTrackingMode to StructuralMode for the new resolver."""
    if mode == StructuralTrackingMode.ENFORCE:
        return StructuralMode.ENFORCE
    elif mode == StructuralTrackingMode.WARN:
        return StructuralMode.WARN
    else:
        return StructuralMode.OFF


class ReproducibilityEnforcer:
    """
    Enforces Reproducibility.

    Tracks cell executions and their read/write sets.
    On each execution, checks for backward mutations and computes staleness.

    Supports structural tracking mode for detecting structural changes
    (like df.columns, df.shape) when those attributes were read.
    """

    def __init__(
        self,
        checkpoints: MemoryCheckpoints,
        structural_mode: StructuralTrackingMode = StructuralTrackingMode.ENFORCE,
        staleness_mode: StalenessMode = StalenessMode.SYNTACTIC,
    ):
        self.checkpoints = checkpoints
        self.seq_counter: int = 0
        self._cell_order: List[str] = []
        self._structural_mode = structural_mode
        self._staleness_mode = staleness_mode
        # NotebookState is the single source of truth for formal model state:
        # T (status), R (reads), W (writes), L (last_writer), and per-cell TrackingData
        self._notebook_state = NotebookState()
        # Declarative conflict resolver
        self._conflict_resolver = ConflictResolver(
            structural_mode=_tracking_mode_to_structural_mode(structural_mode)
        )
        # Deferred checkpoint deletion for syntactic mode - keeps last checkpoint
        # until next cell executes, allowing size queries between executions
        self._pending_checkpoint_deletion: Optional[str] = None
        # Snapshot for rollback if execution is rejected
        self._pending_snapshot: Optional[CellStateSnapshot] = None

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
    def staleness_mode(self) -> StalenessMode:
        """Get the current staleness computation mode."""
        return self._staleness_mode

    def set_staleness_mode(self, mode: StalenessMode) -> None:
        """Set the staleness computation mode.

        When switching from SEMANTIC to SYNTACTIC, clears all stored
        pre-checkpoints since syntactic mode doesn't need them.
        """
        old_mode = self._staleness_mode
        self._staleness_mode = mode

        # Clear checkpoints when switching from semantic to syntactic
        # (syntactic mode doesn't need stored checkpoints)
        if old_mode == StalenessMode.SEMANTIC and mode == StalenessMode.SYNTACTIC:
            self._clear_all_pre_checkpoints()

    def _clear_all_pre_checkpoints(self) -> None:
        """Clear all stored pre-checkpoints (called when switching to syntactic mode)."""
        keys_to_delete = [
            key for key in self.checkpoints.list()
            if key.startswith(PRE_CHECKPOINT_PREFIX)
        ]
        for key in keys_to_delete:
            self.checkpoints.delete(key)
        # Also clear any pending deletion
        self._pending_checkpoint_deletion = None

    @property
    def cell_order(self) -> List[str]:
        return self._cell_order

    def set_cell_order(self, order: List[str]) -> OrderChangeResult:
        """Update notebook structure. Called via magic or metadata.

        Implements DELETE, INSERT, and MOVE transitions (§2.4-§2.6).

        Args:
            order: New cell order (list of cell IDs)

        Returns:
            OrderChangeResult with newly_stale cells, warnings, and delta
        """
        from flowbook.util.output import log, timer

        with timer(key="order:set_cell_order", message="[Order] Processing order change"):
            old_order = self._cell_order
            delta = self._compute_order_delta(old_order, order)

            # Update order first (needed for position lookups in handlers)
            self._cell_order = order

            all_newly_stale: List[str] = []
            all_warnings: List[str] = []

            # Handle deletions (§2.4)
            if delta.deleted:
                newly_stale, warnings = self._handle_deletions(delta.deleted, old_order)
                all_newly_stale.extend(newly_stale)
                all_warnings.extend(warnings)
                # Note: NotebookState cleanup (tracking_data, etc.) happens in set_cell_order below

            # Handle moves (§2.6)
            if delta.moved:
                newly_stale, warnings = self._handle_moves(delta.moved, old_order)
                all_newly_stale.extend(newly_stale)
                all_warnings.extend(warnings)

            # INSERT (§2.5): no action needed (new cells have no records)

            # Sync NotebookState with new order (handles its own insert/delete/reorder tracking)
            self._notebook_state.set_cell_order(order)

            if all_newly_stale:
                log(f"[ORDER] Cells marked stale: {all_newly_stale}")

            return OrderChangeResult(
                newly_stale=all_newly_stale,
                warnings=all_warnings,
                delta=delta,
            )

    def _compute_order_delta(
        self, old_order: List[str], new_order: List[str]
    ) -> OrderDelta:
        """Compute delta between old and new cell order.

        Args:
            old_order: Previous cell order
            new_order: New cell order

        Returns:
            OrderDelta with deleted, inserted, and moved cells
        """
        old_set = set(old_order)
        new_set = set(new_order)

        deleted = [c for c in old_order if c not in new_set]
        inserted = [c for c in new_order if c not in old_set]

        # Build position maps for cells that exist in both orders
        old_positions = {c: i for i, c in enumerate(old_order)}
        new_positions = {c: i for i, c in enumerate(new_order)}

        # Find moved cells: cells in both orders whose relative position changed
        # We use a stable algorithm: for each cell, check if its position changed
        # relative to cells that were adjacent to it
        moved: List[MovedCell] = []
        common_cells = old_set & new_set

        for cell_id in common_cells:
            old_pos = old_positions[cell_id]
            new_pos = new_positions[cell_id]
            if old_pos != new_pos:
                moved.append(MovedCell(cell_id=cell_id, old_position=old_pos, new_position=new_pos))

        return OrderDelta(deleted=deleted, inserted=inserted, moved=moved)

    def _handle_deletions(
        self, deleted_cells: List[str], old_order: List[str]
    ) -> tuple:
        """Handle DELETE transitions — Inst-Delete rule.

        Formal ref: FORMAL_DEVELOPMENT.md §3.3, §3.5 [Inst-Delete]

        Deleting cell i is modeled as W''=W[i:={}], R''=R[i:={}], then:
        - ForwardStale(R, W, W'', i, j): j > i ∧ Wᵢ ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
        - BackwardStale(W, W'', i, j):   j < i ∧ j = LastWriter(W, i, y) for y ∈ Wᵢ

        Reuses the same staleness predicates as [Inst-Run].

        Args:
            deleted_cells: List of deleted cell IDs (cell i being deleted)
            old_order: Cell order before deletion

        Returns:
            Tuple of (newly_stale cell IDs, warnings)
        """
        from flowbook.util.output import log, timer

        with timer(key="order:Inst-Delete", message=f"[Inst-Delete] Handling {len(deleted_cells)} deletions"):
            newly_stale: List[str] = []
            warnings: List[str] = []
            deleted_set = set(deleted_cells)

            for deleted_id in deleted_cells:
                deleted_tracking = self._notebook_state.get_tracking(deleted_id)
                if deleted_tracking is None:
                    continue  # No execution record, nothing to propagate

                deleted_writes = deleted_tracking.writes

                if not deleted_writes:
                    continue  # Deleted cell didn't write anything

                my_position = old_order.index(deleted_id)
                fwd_marked = 0
                bwd_marked = 0

                # ForwardStale: j > i, Wᵢ ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
                for cell_id in old_order[my_position + 1:]:
                    if cell_id in deleted_set:
                        continue
                    if not self._notebook_state.is_clean(cell_id):
                        continue

                    other_tracking = self._notebook_state.get_tracking(cell_id)
                    if other_tracking is None:
                        continue

                    other_reads = other_tracking.reads_before_writes or set()
                    other_writes = other_tracking.writes or set()
                    read_overlap = deleted_writes & other_reads
                    write_overlap = deleted_writes & other_writes

                    if read_overlap or write_overlap:
                        if cell_id not in newly_stale:
                            newly_stale.append(cell_id)
                        fwd_marked += 1
                        for var in read_overlap:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=deleted_id)
                            )
                        for var in write_overlap - read_overlap:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.WRITE_OVERLAP, loc=var, cell_id=deleted_id)
                            )
                        alpha_deleted = self._cell_id_to_alpha(deleted_id)
                        alpha_other = self._cell_id_to_alpha(cell_id)
                        overlap = read_overlap | write_overlap
                        warning = (
                            f"Cell @{alpha_other} marked stale: "
                            f"deleted cell @{alpha_deleted} wrote {sorted(overlap)}"
                        )
                        warnings.append(warning)
                        log(f"[DELETE-FWD] {warning}")

                # BackwardStale: j < i, j = LastWriter(W, i, y) for y ∈ Wᵢ
                # Snapshot clean state before marking — the formal rule checks
                # against the original Tⱼ, not the updated T''ⱼ.
                originally_clean = {
                    cell_id for cell_id in old_order[:my_position]
                    if cell_id not in deleted_set
                    and self._notebook_state.is_clean(cell_id)
                }
                for y in deleted_writes:
                    # Find last writer of y among cells before i
                    last_j = None
                    for cell_id in old_order[:my_position]:
                        if cell_id in deleted_set:
                            continue
                        cell_writes = self._notebook_state.writes.get(cell_id, set())
                        if y in cell_writes:
                            last_j = cell_id  # Keep scanning; last one wins

                    if last_j is not None and last_j in originally_clean:
                        if last_j not in newly_stale:
                            newly_stale.append(last_j)
                        bwd_marked += 1
                        self._notebook_state.add_reason(
                            last_j,
                            Reason(ReasonType.BACKWARD_STALE, loc=y, cell_id=deleted_id)
                        )
                        alpha_deleted = self._cell_id_to_alpha(deleted_id)
                        alpha_last = self._cell_id_to_alpha(last_j)
                        warning = (
                            f"Cell @{alpha_last} marked stale (backward): "
                            f"was last writer of '{y}' before deleted cell @{alpha_deleted}"
                        )
                        warnings.append(warning)
                        log(f"[DELETE-BWD] {warning}")

                log(f"[Inst-Delete] Cell {deleted_id}: ForwardStale marked {fwd_marked}, BackwardStale marked {bwd_marked} cells stale")

            return (newly_stale, warnings)

    def _handle_moves(
        self, moved_cells: List[MovedCell], old_order: List[str]
    ) -> tuple:
        """Handle MOVE transitions — Inst-Move-Down/Up rules.

        Formal ref: main.tex §3.5 [Inst-Move-Down], [Inst-Move-Up],
                    FORMAL_DEVELOPMENT.md §3.5

        Move is the composition of Inst-Delete followed by Inst-Insert.

        Move forward (p < q):
            - Crossed cells that read moved cell's writes → stale (lost dependency)
            - Moved cell that reads from crossed cells' writes → stale (gains input)

        Move backward (q < p):
            - Moved cell that reads from crossed cells' writes → stale (forward contamination)
            - Crossed cells that read moved cell's writes → stale (gains input)

        IMPORTANT: "Crossed" means cells whose relative order to the moved cell
        actually changed. If all cells shift together (e.g., due to insertion),
        they don't cross each other.

        Args:
            moved_cells: List of MovedCell records
            old_order: Cell order before moves

        Returns:
            Tuple of (newly_stale cell IDs, warnings)
        """
        from flowbook.util.output import log, timer

        with timer(key="order:Inst-Move", message=f"[Inst-Move] Handling {len(moved_cells)} moves"):
            newly_stale: List[str] = []
            warnings: List[str] = []
            new_order = self._cell_order

            # Build position maps
            old_positions = {c: i for i, c in enumerate(old_order)}
            new_positions = {c: i for i, c in enumerate(new_order)}

            for move in moved_cells:
                cell_id = move.cell_id

                cell_tracking = self._notebook_state.get_tracking(cell_id)
                if cell_tracking is None:
                    continue  # No execution record, nothing to check

                cell_reads = cell_tracking.reads_before_writes
                cell_writes = cell_tracking.writes

                old_pos = move.old_position
                new_pos = move.new_position
                is_forward = new_pos > old_pos

                # Determine truly crossed cells: cells whose relative order to cell_id changed
                # A cell is "crossed" if:
                #   - It was AFTER cell_id in old order but is now BEFORE in new order, OR
                #   - It was BEFORE cell_id in old order but is now AFTER in new order
                crossed_ids = []
                for other_id in self._cell_order:
                    if other_id == cell_id:
                        continue
                    if other_id not in old_positions or other_id not in new_positions:
                        continue  # Cell was deleted or inserted, not moved

                    other_old_pos = old_positions[other_id]
                    other_new_pos = new_positions[other_id]

                    # Check if relative order flipped
                    was_after = other_old_pos > old_pos
                    is_after = other_new_pos > new_pos

                    if was_after != is_after:
                        crossed_ids.append(other_id)

                # Count cells marked stale for this move
                cells_marked = 0

                for other_id in crossed_ids:
                    other_tracking = self._notebook_state.get_tracking(other_id)
                    if other_tracking is None:
                        continue

                    other_reads = other_tracking.reads_before_writes
                    other_writes = other_tracking.writes

                    # Determine direction of crossing for this specific pair
                    other_old_pos = old_positions[other_id]
                    other_new_pos = new_positions[other_id]
                    was_after = other_old_pos > old_pos
                    # is_after = other_new_pos > new_pos  # Must be opposite of was_after

                    if was_after:
                        # other_id was after cell_id, now before: cell_id moved forward past other_id
                        # (Ex1) Crossed cells that read moved cell's writes → stale
                        overlap1 = other_reads & cell_writes
                        if overlap1 and self._notebook_state.is_clean(other_id):
                            newly_stale.append(other_id)
                            cells_marked += 1
                            # Track reason: ORDER_CHANGED
                            self._notebook_state.add_reason(
                                other_id, Reason(ReasonType.ORDER_CHANGED)
                            )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            warning = (
                                f"Cell @{alpha_other} marked stale: "
                                f"cell @{alpha_moved} moved forward past it, "
                                f"lost dependency on {sorted(overlap1)}"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                        # (Ex2) Moved cell reads from crossed cells' writes → stale
                        overlap2 = cell_reads & other_writes
                        if overlap2 and self._notebook_state.is_clean(cell_id):
                            newly_stale.append(cell_id)
                            cells_marked += 1
                            # Track reason: ORDER_CHANGED
                            self._notebook_state.add_reason(
                                cell_id, Reason(ReasonType.ORDER_CHANGED)
                            )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            warning = (
                                f"Cell @{alpha_moved} marked stale: "
                                f"moved forward past @{alpha_other}, "
                                f"now reads {sorted(overlap2)} from it"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                    else:
                        # other_id was before cell_id, now after: cell_id moved backward past other_id
                        # (Ex3) Moved cell reads from crossed cells' writes → stale
                        overlap3 = cell_reads & other_writes
                        if overlap3 and self._notebook_state.is_clean(cell_id):
                            newly_stale.append(cell_id)
                            cells_marked += 1
                            # Track reason: NO_READ_BEFORE_WRITE (forward contamination)
                            for var in overlap3:
                                self._notebook_state.add_reason(
                                    cell_id,
                                    Reason(ReasonType.NO_READ_BEFORE_WRITE, loc=var, cell_id=other_id)
                                )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            warning = (
                                f"Cell @{alpha_moved} marked stale: "
                                f"moved backward before @{alpha_other}, "
                                f"forward contamination on {sorted(overlap3)}"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                        # (Ex4) Crossed cells that read moved cell's writes → stale
                        overlap4 = other_reads & cell_writes
                        if overlap4 and self._notebook_state.is_clean(other_id):
                            newly_stale.append(other_id)
                            cells_marked += 1
                            # Track reason: FORWARD_STALE (gains input from moved cell)
                            for var in overlap4:
                                self._notebook_state.add_reason(
                                    other_id,
                                    Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=cell_id)
                                )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            warning = (
                                f"Cell @{alpha_other} marked stale: "
                                f"cell @{alpha_moved} moved backward before it, "
                                f"gains input from {sorted(overlap4)}"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                direction = "Down" if is_forward else "Up"
                log(f"[Inst-Move-{direction}] Cell {cell_id}: crossed {len(crossed_ids)} cells, {cells_marked} marked stale")

            return (newly_stale, warnings)

    def _cell_id_to_alpha(self, cell_id: str) -> str:
        """Convert cell ID to @A notation using cell_order position."""
        try:
            index = self._cell_order.index(cell_id)
            return index_to_alpha(index)
        except ValueError:
            # Cell not in order, just return the ID
            return cell_id

    def check(
        self,
        cell_id: str,
        pre_checkpoint,
        namespace: dict,
        tracking: TrackingData,
        continue_on_violation: bool = False,
    ) -> ReproducibilityResult:
        """
        Main entry point. Implements [Inst-Run] from FORMAL_DEVELOPMENT.md §3.4.

        Formal ref: FORMAL_DEVELOPMENT.md §3.4, lines 205-219

        [Inst-Run] Transition Rule:
        1. Cᵢ; Σ ⇓ o · Σ' · r · w  (cell execution produces r, w)
        2. R' = R[i := r], W' = W[i := w]  (update state)
        3. Check validity predicates (§3.2, lines 176-179):
           - NoReadAndWrite(R', W', i)    ≝ Rᵢ ∩ Wᵢ = ∅
           - WriteBeforeRead(R', W', i)   ≝ Rᵢ ⊆ W_{1..i-1}
           - NoReadBeforeWrite(R', W', i) ≝ Rᵢ ∩ W_{i+1..n} = ∅
           - NoWriteAfterRead(R', W', i)  ≝ Wᵢ ∩ R_{1..i-1} = ∅
        4. If all pass: T'ᵢ = CLEAN; else T'ᵢ = STALE with problem list
        5. For j ≠ i, compute staleness (§3.3, lines 187-188):
           - ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
           - BackwardStale(W, W', i, j)   ≝ j < i ∧ j = LastWriter(W,i,y) for y ∈ Wᵢ\\W'ᵢ

        Args:
            cell_id: ID of the cell that just executed
            pre_checkpoint: Snapshot before execution (Checkpoint)
            namespace: Live user namespace dict (post-execution state)
            tracking: TrackingData with reads/writes
            continue_on_violation: If True, compute staleness even when violation detected

        Returns:
            ReproducibilityResult with violation info, stale cells, and changed variables
        """
        from flowbook.kernel.models import (
            tracking_to_read_locs, diff_to_write_locs, check_loc_conflicts,
            get_var_locs, get_loc_variables,
        )
        from flowbook.util.output import log

        # Process deferred checkpoint deletion from previous cell (syntactic mode)
        # This allows checkpoint size queries after a cell completes but before the next runs
        # IMPORTANT: Skip deletion if the pending checkpoint is for the CURRENT cell.
        # This happens when the same cell is executed twice in a row - the first execution
        # sets pending deletion, and the second execution creates a new checkpoint with
        # the same name. We must not delete the newly created checkpoint!
        current_cell_checkpoint = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
        if self._pending_checkpoint_deletion is not None:
            if self._pending_checkpoint_deletion != current_cell_checkpoint:
                self.checkpoints.delete(self._pending_checkpoint_deletion)
                from flowbook.kernel_support.deepcopy import clear_container_cache
                clear_container_cache()
            self._pending_checkpoint_deletion = None

        self.seq_counter += 1
        my_position = self._get_position(cell_id)
        problems: List[Reason] = []

        # ================================================================
        # Expand column and structural tracking with aliases early
        # This ensures the expanded versions are used throughout.
        # When x and y are aliases for p, reading p['col'] should also record
        # x['col'] and y['col'] in the tracking data for proper staleness detection.
        # ================================================================
        expanded_column_reads = _expand_var_set_dict_with_aliases(
            {k: set(v) for k, v in tracking.column_reads_before_writes.items()},
            pre_checkpoint,
        )
        expanded_column_writes = _expand_var_set_dict_with_aliases(
            {k: set(v) for k, v in tracking.column_writes.items()},
            pre_checkpoint,
        )
        expanded_structural_reads = _expand_var_set_dict_with_aliases(
            {k: set(v) for k, v in tracking.structural_reads.items()},
            pre_checkpoint,
        )

        # Replace tracking with expanded version for use throughout
        tracking = TrackingData(
            reads_before_writes=tracking.reads_before_writes,
            writes=tracking.writes,
            column_reads_before_writes=expanded_column_reads,
            column_writes=expanded_column_writes,
            structural_reads=expanded_structural_reads,
            file_reads_before_writes=tracking.file_reads_before_writes,
            file_writes=tracking.file_writes,
        )

        # ================================================================
        # STEP 1: Compute r (reads) and w (writes) from tracking
        # Ref: FORMAL_DEVELOPMENT.md §3.1, line 169
        # ================================================================
        R_i_locs = tracking_to_read_locs(tracking)  # r as LocSet
        W_i_old = self._notebook_state.writes.get(cell_id, set())  # Old W_i (strings)

        # Compute diff to get actual changes
        with timer(key="check:compute_diff", message=f"[Inst-Run] Computing diff for {cell_id}"):
            current_diff, typed_changes = self._compute_diff_and_changes(
                pre_checkpoint, namespace, tracking
            )

        # Check for truncation first
        truncated_vars = _check_for_truncation(current_diff)
        if truncated_vars:
            formatted_diff = _format_diff_for_display(current_diff, truncated_vars)
            mutating_alpha = self._cell_id_to_alpha(cell_id)
            return ReproducibilityResult(
                violation=ReproducibilityViolation(
                    mutating_cell=cell_id,
                    affected_cell=cell_id,
                    variables=truncated_vars,
                    message=format_truncation_error(mutating_alpha, truncated_vars),
                    truncation_details=formatted_diff,
                ),
                stale_cells=[],
                changed_variables=[],
                column_changed={},
                structural_warnings=list(current_diff.warnings) if current_diff.warnings else [],
            )

        # W_i from diff (what actually changed)
        W_i_locs = diff_to_write_locs(current_diff, tracking)  # w as LocSet
        changed_vars = list(current_diff.differences.keys()) if current_diff.differences else []
        column_changed = _extract_column_changes(current_diff, tracking)

        # Separate value-level changes from column-only changes for last_writer tracking.
        # Value-level changes (ValueChanged, RowsAdded, etc.) update last_writer[var].
        # Column-only changes (ColumnAdded, ColumnModified, etc.) only update column_last_writer.
        # This prevents false SKIPPED_UPSTREAM when a cell only mutates columns.
        value_level_changed_vars = _get_value_level_changed_vars(typed_changes)

        # Classify changed vars as recoverable vs unrecoverable.
        # Recoverable: var was rebound (in tracking.writes) → can be restored by re-execution.
        # Unrecoverable: diff-detected change NOT in tracking.writes → in-place mutation.
        #
        # We only flag a variable as unrecoverable if it was VALUE-LEVEL changed
        # (not just column-level), not rebound, AND existed before execution.
        # Variables that are NEW (added to namespace by this cell) are creations,
        # not mutations — they should be in tracking.writes but if they aren't,
        # they are not "unrecoverable mutations" of existing state.
        current_writes_set = tracking.writes or set()
        recoverable_changed_vars = set(changed_vars) & current_writes_set

        # Determine which diff-detected variables are truly new (didn't exist before)
        from flowbook.kernel_support.types import ValueComparison
        _new_vars = set()
        if current_diff.differences:
            for var_name, diff_node in current_diff.differences.items():
                if isinstance(diff_node, ValueComparison) and diff_node.value1 is None:
                    _new_vars.add(var_name)

        unrecoverable_changed_vars = (
            (value_level_changed_vars - current_writes_set - _new_vars)
            if value_level_changed_vars else set()
        )

        # Column-level: recoverable iff column is in tracking.column_writes
        tracked_col_writes = tracking.column_writes or {}
        recoverable_column_changed: Dict[str, List[str]] = {}
        unrecoverable_column_changed: Dict[str, List[str]] = {}
        for var, cols in column_changed.items():
            tw = set(tracked_col_writes.get(var, []))
            rec = [c for c in cols if c in tw]
            unrec = [c for c in cols if c not in tw]
            if rec:
                recoverable_column_changed[var] = rec
            if unrec:
                unrecoverable_column_changed[var] = unrec

        # Variables that changed at column level only (not value level)
        # are recoverable if ALL their changed columns are in column_writes
        for var in list(unrecoverable_changed_vars):
            if var in column_changed and var not in value_level_changed_vars:
                # Only column-level changes for this var — check if all columns recoverable
                if var not in unrecoverable_column_changed:
                    unrecoverable_changed_vars.discard(var)
                    recoverable_changed_vars.add(var)

        # Catch untracked column-level mutations: variables in changed_vars that are
        # NOT rebound, NOT in value_level_changed_vars, and NOT in column_changed.
        # These are DataFrame mutations (e.g., df.iloc[0,0]=999) with no column tracking
        # — the diff detects a column-level change but _extract_column_changes skips
        # them because the variable has no column tracking. They are unrecoverable.
        for var in set(changed_vars) - current_writes_set - _new_vars:
            if var not in value_level_changed_vars and var not in column_changed:
                if var not in unrecoverable_changed_vars:
                    unrecoverable_changed_vars.add(var)

        # Convert LocSet to Set[str] for backward compatibility with NotebookState
        R_i_vars = tracking.reads_before_writes  # Use tracking directly
        # W_i_vars: recoverable changes that should propagate staleness.
        # Includes rebound variables AND variables with recoverable column changes
        # (e.g., df['col'] = val where col is in column_writes but df is not rebound).
        W_i_vars = recoverable_changed_vars | set(recoverable_column_changed.keys())

        # Extract structural warnings from diff
        structural_warnings = list(current_diff.warnings) if current_diff.warnings else []

        if my_position < 0:
            # Cell not in order - store state and return early
            structural_read_values = {}
            if namespace is not None and tracking.structural_reads:
                structural_read_values = capture_structural_read_values(namespace, tracking.structural_reads)

            # tracking is already alias-expanded at the start of check()
            self._notebook_state.record_execution(
                cell_id,
                tracking=tracking,
                changed_vars=value_level_changed_vars if value_level_changed_vars else None,
                column_changed={k: set(v) for k, v in column_changed.items()} if column_changed else None,
                execution_seq=self.seq_counter,
                structural_reads_values=structural_read_values,
                typed_changes=typed_changes,
            )
            return ReproducibilityResult(
                violation=None,
                stale_cells=self._notebook_state.get_stale_cells(),
                changed_variables=changed_vars,
                column_changed=column_changed,
                structural_warnings=structural_warnings,
                staleness_reasons=self._notebook_state.get_all_reasons(),
            )

        # ================================================================
        # STEP 2: Check validity predicates BEFORE updating state
        # Ref: FORMAL_DEVELOPMENT.md §3.2, lines 176-179
        # ================================================================
        violation = None
        forward_violation = None
        writer_violation = None
        errors: List[ReproducibilityError] = []

        # NoReadAndWrite(R', W', i) ≝ Rᵢ ∩ Wᵢ = ∅
        # Ref: FORMAL_DEVELOPMENT.md §3.2, line 176
        # (Cell reads and writes same location - potential issue for reproducibility)
        no_read_and_write_error = self._check_no_read_and_write(cell_id, tracking)
        if no_read_and_write_error:
            errors.append(no_read_and_write_error)
            log(f"[Inst-Run] {cell_id}: NoReadAndWrite=fail")
        else:
            log(f"[Inst-Run] {cell_id}: NoReadAndWrite=pass")

        # WriteBeforeRead(R', W', i) ≝ Rᵢ ⊆ W_{1..i-1}
        # Ref: FORMAL_DEVELOPMENT.md §3.2, line 177
        # (Reads user variable not written by earlier cell)
        write_before_read_error = self._check_write_before_read(cell_id, my_position, tracking, namespace)
        if write_before_read_error:
            errors.append(write_before_read_error)
            log(f"[Inst-Run] {cell_id}: WriteBeforeRead=fail")
        else:
            log(f"[Inst-Run] {cell_id}: WriteBeforeRead=pass")

        # NoWriteAfterRead(R', W', i) ≝ Wᵢ ∩ R_{1..i-1} = ∅
        # Ref: FORMAL_DEVELOPMENT.md §3.2, line 179
        # (Backward mutation check - only against CLEAN cells)
        backward_violation = None
        if typed_changes:
            with timer(key="check:NoWriteAfterRead", message=f"[Inst-Run] NoWriteAfterRead check for {cell_id}"):
                backward_violation = self._check_backward_mutation_new(
                    cell_id, my_position, typed_changes, current_diff, column_changed
                )
        if backward_violation:
            errors.append(ReproducibilityError(
                error_type=ErrorType.NO_WRITE_AFTER_READ,
                cell_id=cell_id,
                locations=backward_violation.variables,
                message=backward_violation.message,
                causer_cell=backward_violation.affected_cell,
                detail={
                    "structural_reads_detail": backward_violation.structural_reads_detail,
                    "changes_detail": backward_violation.changes_detail,
                } if backward_violation.structural_reads_detail or backward_violation.changes_detail else None,
            ))
        log(f"[Inst-Run] {cell_id}: NoWriteAfterRead={'fail' if backward_violation else 'pass'}")

        # RecoverableMutation: diff(pre_i, ns) ⊆ W_i ∪ ColW_i
        # In-place mutations without rebinding are unrecoverable errors.
        unrecoverable_error = self._check_unrecoverable_mutation(
            cell_id, unrecoverable_changed_vars, unrecoverable_column_changed
        )
        if unrecoverable_error:
            errors.append(unrecoverable_error)
            log(f"[Inst-Run] {cell_id}: RecoverableMutation=fail ({unrecoverable_error.locations})")
        else:
            log(f"[Inst-Run] {cell_id}: RecoverableMutation=pass")

        # NoReadBeforeWrite(R', W', i) ≝ Rᵢ ∩ W_{i+1..n} = ∅
        # Ref: FORMAL_DEVELOPMENT.md §3.2, line 178
        # (Forward contamination check)
        with timer(key="check:NoReadBeforeWrite", message=f"[Inst-Run] NoReadBeforeWrite check for {cell_id}"):
            forward_violation = self._check_forward_contamination(cell_id, my_position, tracking)
        if forward_violation:
            errors.append(ReproducibilityError(
                error_type=ErrorType.NO_READ_BEFORE_WRITE,
                cell_id=cell_id,
                locations=forward_violation.variables,
                message=forward_violation.message,
                causer_cell=forward_violation.mutating_cell,
            ))
        log(f"[Inst-Run] {cell_id}: NoReadBeforeWrite={'fail' if forward_violation else 'pass'}")

        # ================================================================
        # STEP 3: Update state R' = R[i := r], W' = W[i := w]
        # Ref: FORMAL_DEVELOPMENT.md §3.4, lines 208-209
        # ALWAYS update state (new semantics: no rejection)
        # ================================================================
        structural_read_values = {}
        if namespace is not None and tracking.structural_reads:
            structural_read_values = capture_structural_read_values(namespace, tracking.structural_reads)

        # Snapshot state before update (for potential rollback)
        self._pending_snapshot = self._notebook_state.snapshot_cell_state(cell_id)

        # tracking is already alias-expanded at the start of check()
        # Pass only recoverable changes for last_writer updates in record_execution.
        # Unrecoverable mutations must not become last_writer of any variable.
        recoverable_value_level_for_record = value_level_changed_vars & current_writes_set if value_level_changed_vars else None
        self._notebook_state.record_execution(
            cell_id,
            tracking=tracking,
            changed_vars=recoverable_value_level_for_record if recoverable_value_level_for_record else None,
            column_changed={k: set(v) for k, v in recoverable_column_changed.items()} if recoverable_column_changed else None,
            execution_seq=self.seq_counter,
            structural_reads_values=structural_read_values,
            typed_changes=typed_changes,
        )

        # Clear pre-execution reasons (NEVER_EXECUTED, CODE_CHANGED) since the cell
        # has now been executed. These reasons are no longer accurate - if the cell
        # has errors, those will be recorded separately; if it's clean, set_clean()
        # will be called below.
        self._notebook_state.clear_pre_execution_reasons(cell_id)

        # ================================================================
        # STEP 4: Determine cell status T'ᵢ
        # T'ᵢ = CLEAN only if ALL validity predicates pass
        # Otherwise T'ᵢ = STALE with appropriate reasons
        # Ref: FORMAL_DEVELOPMENT.md §3.4, line 214
        #
        # IMPORTANT: When continue_on_violation=True, predicate violations
        # are ACCEPTED and the cell stays CLEAN. The only exception is
        # skipped_writers which always causes staleness.
        # ================================================================
        # Check for skipped writers (provenance mismatch)
        # When ENABLE_SKIPPED_UPSTREAM is False, we skip this check - ForwardStale
        # handles these cases reactively when the skipped cell is eventually executed.
        if ENABLE_SKIPPED_UPSTREAM:
            skipped_writers = self._check_skipped_writers(cell_id)
        else:
            skipped_writers = []

        # Collect staleness reasons for this cell
        # Only add predicate violation reasons if NOT continuing (i.e., they will be rejected)
        if not continue_on_violation:
            if backward_violation is not None:
                # NoWriteAfterRead failed - cell reads values it then modifies, breaking reproducibility
                for var in backward_violation.variables:
                    self._notebook_state.add_reason(
                        cell_id, Reason(ReasonType.NO_WRITE_AFTER_READ, loc=var, cell_id=backward_violation.affected_cell)
                    )
            if unrecoverable_error is not None:
                # UnrecoverableMutation failed - cell mutated state without rebinding
                for loc in unrecoverable_error.locations:
                    self._notebook_state.add_reason(
                        cell_id, Reason(ReasonType.UNRECOVERABLE_MUTATION, loc=loc)
                    )

        # Build writer_violation for UI purposes if forward contamination detected
        writer_violation = None
        if forward_violation is not None:
            # Forward contamination ALWAYS marks cell stale (even when continue_on_violation=True)
            # because re-running top-to-bottom would produce different results.
            # This is different from backward_violation which only affects other cells.
            writer = forward_violation.mutating_cell
            for var in forward_violation.variables:
                self._notebook_state.add_reason(
                    cell_id, Reason(ReasonType.NO_READ_BEFORE_WRITE, loc=var, cell_id=writer if writer != "<later>" else None)
                )
            # Build writer_violation for UI (shows backward_mutation-style message on writer cell)
            writer = forward_violation.mutating_cell
            if writer and writer != "<later>":
                writer_alpha = self._cell_id_to_alpha(writer)
                reader_alpha = self._cell_id_to_alpha(cell_id)
                writer_violation = ReproducibilityViolation(
                    mutating_cell=writer,
                    affected_cell=cell_id,
                    variables=forward_violation.variables,
                    message=format_backward_mutation_message(writer_alpha, reader_alpha, forward_violation.variables),
                    violation_type="backward_mutation",
                )

        # Skipped writers always cause staleness (regardless of continue_on_violation)
        if skipped_writers:
            for loc, actual_writer, expected_writer in skipped_writers:
                self._notebook_state.add_reason(
                    cell_id,
                    Reason(ReasonType.SKIPPED_UPSTREAM, loc=loc, cell_id=actual_writer, expected_cell_id=expected_writer)
                )

        # Set cell status
        # When continue_on_violation=True, only NoWriteAfterRead (backward_violation) can be accepted.
        # Other errors (forward contamination, read-and-write, undefined vars) always cause staleness
        # because they affect THIS cell's reproducibility.
        has_any_errors = len(errors) > 0

        # Separate errors into backward-only (can be accepted) and forward/other (always stale)
        _acceptable_error_types = {ErrorType.NO_WRITE_AFTER_READ, ErrorType.UNRECOVERABLE_MUTATION}
        backward_only_errors = [e for e in errors if e.error_type in _acceptable_error_types]
        other_errors = [e for e in errors if e.error_type not in _acceptable_error_types]

        # Backward-only errors can be accepted; other errors always cause staleness
        has_staleness_causing_errors = len(other_errors) > 0 or (len(backward_only_errors) > 0 and not continue_on_violation)

        if not has_staleness_causing_errors and not skipped_writers:
            self._notebook_state.set_clean(cell_id)
            if len(backward_only_errors) > 0:
                log(f"[Inst-Run] {cell_id}: T'=CLEAN (backward errors accepted via continue_on_violation)")
            else:
                log(f"[Inst-Run] {cell_id}: T'=CLEAN")
        else:
            # Build error-based staleness reasons (replaces any prior staleness reasons)
            error_reasons: Set[Reason] = set()
            reason_names = []
            for err in errors:
                reason_type = ReasonType(err.error_type.value)
                # Create a Reason for each location in the error
                for loc in err.locations:
                    error_reasons.add(Reason(reason_type, loc=loc, cell_id=err.causer_cell))
                reason_names.append(err.error_type.value)
            # Add skipped_writers reasons
            for loc, actual_writer, expected_writer in skipped_writers:
                error_reasons.add(Reason(
                    ReasonType.SKIPPED_UPSTREAM,
                    loc=loc,
                    cell_id=actual_writer,
                    expected_cell_id=expected_writer
                ))
            if skipped_writers:
                reason_names.append("skipped_writers")
            # Replace existing staleness with error-based reasons
            self._notebook_state.set_stale(cell_id, error_reasons)
            log(f"[Inst-Run] {cell_id}: T'=STALE ({', '.join(reason_names)})")

        # ================================================================
        # STEP 5: Compute staleness for all j ≠ i
        # Ref: FORMAL_DEVELOPMENT.md §3.4, lines 215-217
        # ================================================================
        # ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
        # Ref: FORMAL_DEVELOPMENT.md §3.3, line 187
        #
        # Skip staleness propagation when the cell will be rejected (rolled back).
        # When continue_on_violation=False and errors are present, the kernel will
        # rollback this cell's execution. Propagating staleness to other cells would
        # be incorrect since the writes never actually persist — and rollback only
        # restores the executing cell's state, not other cells' status.
        stale: List[str] = []
        staleness_warnings: List[str] = []

        # Determine if this cell will be rejected by the kernel
        will_be_rejected = has_any_errors and not continue_on_violation

        _changed_file_paths = tracking.file_writes if tracking.file_writes else None

        # W_i_current = current tracking.writes (what cell claims to write now)
        W_i_current = tracking.writes or set()

        if not will_be_rejected:
            with timer(key="check:ForwardStale", message=f"[Inst-Run] ForwardStale computation for {cell_id}"):
                stale, staleness_warnings = self._compute_forward_staleness(
                    namespace, W_i_old, W_i_current, W_i_vars, recoverable_column_changed, cell_id, my_position,
                    changed_file_paths=_changed_file_paths,
                )
            log(f"[Inst-Run] {cell_id}: ForwardStale marked {len(stale)} cells")
            structural_warnings.extend(staleness_warnings)

            # BackwardStale: mark cells j < i as stale if W_i ∩ R_j ≠ ∅
            # This handles the case where a later cell writes to a variable
            # that an earlier (clean) cell had read.
            with timer(key="check:BackwardStale", message=f"[Inst-Run] BackwardStale computation for {cell_id}"):
                backward_stale = self._compute_backward_staleness(
                    namespace, W_i_vars, recoverable_column_changed, cell_id, my_position
                )
            if backward_stale:
                log(f"[Inst-Run] {cell_id}: BackwardStale marked {len(backward_stale)} cells")
                stale.extend(backward_stale)
        else:
            log(f"[Inst-Run] {cell_id}: Skipping staleness propagation (cell will be rejected)")

        # Update last_writer (L) for recoverable value-level changed variables only.
        # Unrecoverable mutations (in-place without rebinding) must NOT become last_writer
        # because the cell cannot restore the full value on re-execution.
        # Column-only changes update column_last_writer below, not last_writer.
        recoverable_value_level = value_level_changed_vars & current_writes_set
        if recoverable_value_level:
            for loc in recoverable_value_level:
                self._notebook_state.last_writer[loc] = cell_id
        if recoverable_column_changed:
            for var, cols in recoverable_column_changed.items():
                if var not in self._notebook_state.column_last_writer:
                    self._notebook_state.column_last_writer[var] = {}
                for col in cols:
                    self._notebook_state.column_last_writer[var][col] = cell_id

        # In syntactic mode, defer checkpoint deletion until next cell executes.
        # This allows checkpoint size queries after cell execution completes.
        # (we've already computed W_i from the diff, so checkpoint is no longer needed
        # for reproducibility checks, but we keep it for metrics collection)
        if self._staleness_mode == StalenessMode.SYNTACTIC:
            self._pending_checkpoint_deletion = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"

        # In semantic mode, check for convergence (stale cells that have converged)
        if self._staleness_mode == StalenessMode.SEMANTIC:
            cleared = self._check_convergence(namespace)
            if cleared:
                log(f"[Inst-Run] {cell_id}: Convergence cleared staleness for {cleared}")
                # Update stale list after convergence
                stale = self._notebook_state.get_stale_cells()

        # ================================================================
        # Return result
        # ================================================================
        staleness_reasons = self._notebook_state.get_all_reasons()

        return ReproducibilityResult(
            violation=backward_violation,  # Return backward mutation info (for UI, not for rejection)
            stale_cells=stale,
            changed_variables=changed_vars,
            column_changed=column_changed,
            structural_warnings=structural_warnings,
            forward_violation=forward_violation,
            writer_violation=writer_violation,  # For UI: shows backward_mutation-style info on writer cell
            staleness_reasons=staleness_reasons,
            errors=errors,  # All formal predicate violations
        )

    # =========================================================================
    # Helper methods for check() - implementing formal predicates
    # =========================================================================

    def _get_position(self, cell_id: str) -> int:
        """Get cell position in document order."""
        try:
            return self._cell_order.index(cell_id)
        except ValueError:
            return -1

    def _compute_diff_and_changes(
        self,
        pre_checkpoint,
        namespace: dict,
        tracking: TrackingData,
    ) -> Tuple[MemoryCheckpointDiffResult, List]:
        """
        Compute diff and typed changes for a cell execution.

        Returns: (diff_result, typed_changes)
        """
        # Prepare accessed columns for diff
        all_accessed_columns = {}
        for var, cols in tracking.column_reads_before_writes.items():
            all_accessed_columns[var] = set(cols)
        for var, cols in tracking.column_writes.items():
            if var in all_accessed_columns:
                all_accessed_columns[var].update(cols)
            else:
                all_accessed_columns[var] = set(cols)

        # Optimization: only diff accessed variables + deep aliases
        keys_to_include: Optional[Set[str]] = None
        if OPT_ACCESSED_VARS_ONLY:
            accessed_vars = set(tracking.reads_before_writes) | set(tracking.writes)
            keys_to_include = _expand_with_deep_aliases(accessed_vars, pre_checkpoint)

        # Compute diff
        _is_combined = isinstance(pre_checkpoint, Checkpoint)
        if _is_combined:
            total_diff = Checkpoint.diff(
                pre_checkpoint,
                namespace,
                keys_to_include=keys_to_include,
                use_leq=False,
                column_rbw=all_accessed_columns,
                structural_reads={},
                structural_mode=self._structural_mode,
            )
            current_diff = total_diff.memory
        else:
            current_diff = MemoryCheckpoint.diff(
                pre_checkpoint,
                namespace,
                keys_to_include=keys_to_include,
                use_leq=False,
                column_rbw=all_accessed_columns,
                structural_reads={},
                structural_mode=self._structural_mode,
            )

        # Convert to typed changes
        typed_changes = detect_changes(current_diff) if current_diff.differences else []

        return current_diff, typed_changes

    def _check_forward_contamination(
        self,
        cell_id: str,
        my_position: int,
        tracking: TrackingData,
    ) -> Optional[ReproducibilityViolation]:
        """
        Check NoReadBeforeWrite predicate.

        Formal ref: NoReadBeforeWrite(R, W, i) ≝ Rᵢ ∩ W_{i+1..n} = ∅
        FORMAL_DEVELOPMENT.md §3.2, line 178
        """
        # Use existing implementation (consolidated from _check_forward_dependency)
        my_read_events = tracking.to_read_events()
        vars_covered_by_typed_changes: Set[str] = set()

        # Check later cells that already executed
        for later_cell_id in self._cell_order[my_position + 1:]:
            if not self._notebook_state.has_record(later_cell_id):
                continue

            later_changes = self._notebook_state.get_typed_changes(later_cell_id)
            if not later_changes:
                continue

            for change in later_changes:
                vars_covered_by_typed_changes.add(change.variable)

            if my_read_events:
                violations_result = self._conflict_resolver.get_violations(later_changes, my_read_events)
                if violations_result:
                    conflicts = []
                    for v in violations_result.violations:
                        var = v.change.variable
                        if hasattr(v.change, 'column') and v.change.column:
                            conflicts.append(f"{var}['{v.change.column}']")
                        else:
                            conflicts.append(var)
                    conflicts = sorted(set(conflicts))

                    # Add truncation notice if needed
                    if violations_result.truncated:
                        conflicts.append(f"... and {violations_result.truncated_count} more")

                    reading_alpha = self._cell_id_to_alpha(cell_id)
                    writing_alpha = self._cell_id_to_alpha(later_cell_id)
                    message = format_forward_dependency_message(reading_alpha, writing_alpha, conflicts)

                    return ReproducibilityViolation(
                        mutating_cell=later_cell_id,
                        affected_cell=cell_id,
                        variables=conflicts,
                        message=message,
                        violation_type="forward_dependency",
                    )

        # Provenance check for uncovered variables
        provenance_conflicts: List[str] = []
        deleted_cell_conflicts: List[str] = []
        writer_cell_for_message: Optional[str] = None
        deleted_writer_cell: Optional[str] = None

        for read_var in (tracking.reads_before_writes or set()):
            if read_var in vars_covered_by_typed_changes:
                continue

            writer_cell = self._notebook_state.last_writer.get(read_var)
            if writer_cell and writer_cell != cell_id:
                try:
                    writer_pos = self._cell_order.index(writer_cell)
                    if writer_pos > my_position:
                        provenance_conflicts.append(read_var)
                        if writer_cell_for_message is None:
                            writer_cell_for_message = writer_cell
                except ValueError:
                    deleted_cell_conflicts.append(read_var)
                    if deleted_writer_cell is None:
                        deleted_writer_cell = writer_cell

        if deleted_cell_conflicts:
            reading_alpha = self._cell_id_to_alpha(cell_id)
            conflict_names = sorted(set(deleted_cell_conflicts))
            vars_str = format_variable_list(conflict_names)
            message = (
                f"⚠️ Deleted cell conflict: Cell {reading_alpha} reads {vars_str} "
                f"written by cell {deleted_writer_cell} which is no longer in the notebook."
            )
            return ReproducibilityViolation(
                mutating_cell=deleted_writer_cell or "<deleted>",
                affected_cell=cell_id,
                variables=conflict_names,
                message=message,
                violation_type="deleted_cell_dependency",
            )

        if provenance_conflicts:
            reading_alpha = self._cell_id_to_alpha(cell_id)
            conflict_names = sorted(set(provenance_conflicts))
            writing_alpha = self._cell_id_to_alpha(writer_cell_for_message)
            message = format_forward_dependency_message(reading_alpha, writing_alpha, conflict_names)

            return ReproducibilityViolation(
                mutating_cell=writer_cell_for_message,
                affected_cell=cell_id,
                variables=conflict_names,
                message=message,
                violation_type="forward_dependency",
            )

        # File forward dependency check
        if tracking.file_reads_before_writes:
            for later_cell_id in self._cell_order[my_position + 1:]:
                later_tracking = self._notebook_state.get_tracking(later_cell_id)
                if later_tracking is None:
                    continue
                file_overlap = tracking.file_reads_before_writes & later_tracking.file_writes
                if file_overlap:
                    reading_alpha = self._cell_id_to_alpha(cell_id)
                    writing_alpha = self._cell_id_to_alpha(later_cell_id)
                    conflict_names = sorted(os.path.basename(p) for p in file_overlap)
                    message = format_forward_dependency_message(reading_alpha, writing_alpha, conflict_names)
                    return ReproducibilityViolation(
                        mutating_cell=later_cell_id,
                        affected_cell=cell_id,
                        variables=conflict_names,
                        message=message,
                        violation_type="forward_dependency",
                    )

        return None

    def _check_backward_mutation_new(
        self,
        cell_id: str,
        my_position: int,
        typed_changes: List,
        current_diff: MemoryCheckpointDiffResult,
        modified_columns: Dict[str, List[str]],
    ) -> Optional[ReproducibilityViolation]:
        """
        Check NoWriteAfterRead predicate.

        Formal ref: NoWriteAfterRead(R, W, i) ≝ Wᵢ ∩ R_{1..i-1} = ∅
        FORMAL_DEVELOPMENT.md §3.2, line 179

        Only checks against CLEAN cells per [Inst-Run] semantics.
        """
        if not typed_changes:
            return None

        # Optimization: skip if no variable-level overlap
        if OPT_CONFLICT_LOOP_SKIP:
            changed_var_names = {c.variable for c in typed_changes}
            all_prior_var_reads: Set[str] = set()
            for prior_cell_id in self._cell_order[:my_position]:
                prior_tracking = self._notebook_state.get_tracking(prior_cell_id)
                if prior_tracking and self._notebook_state.is_clean(prior_cell_id):
                    all_prior_var_reads.update(prior_tracking.reads_before_writes)
            if not (changed_var_names & all_prior_var_reads):
                return None

        # Check each prior CLEAN cell
        for prior_cell_id in self._cell_order[:my_position]:
            prior_tracking = self._notebook_state.get_tracking(prior_cell_id)
            if prior_tracking is None:
                continue
            if not self._notebook_state.is_clean(prior_cell_id):
                continue

            prior_reads = prior_tracking.to_read_events()
            if not prior_reads:
                continue

            violations_result = self._conflict_resolver.get_violations(typed_changes, prior_reads)
            if not violations_result:
                continue

            # Build conflict list - always show column info when change was at column level
            conflicts = []
            for v in violations_result.violations:
                var = v.change.variable
                if hasattr(v.change, 'column') and v.change.column:
                    conflicts.append(f"{var}.{v.change.column}")
                else:
                    conflicts.append(var)
            conflicts = sorted(set(conflicts))

            # Add truncation notice if needed
            if violations_result.truncated:
                conflicts.append(f"... and {violations_result.truncated_count} more")

            if conflicts:
                mutating_alpha = self._cell_id_to_alpha(cell_id)
                affected_alpha = self._cell_id_to_alpha(prior_cell_id)
                prior_structural_values = self._notebook_state.get_structural_reads_values(prior_cell_id)
                changes = _extract_change_descriptions(current_diff, modified_columns)
                message = format_structural_violation(
                    mutating_alpha, affected_alpha, conflicts, prior_structural_values, changes
                )

                return ReproducibilityViolation(
                    mutating_cell=cell_id,
                    affected_cell=prior_cell_id,
                    variables=conflicts,
                    message=message,
                    structural_reads_detail=prior_structural_values,
                    changes_detail=changes,
                )

        return None

    def _check_write_before_read(
        self,
        cell_id: str,
        cell_position: int,
        tracking: TrackingData,
        user_ns,
    ) -> Optional[ReproducibilityError]:
        """
        Check WriteBeforeRead predicate: Rᵢ ⊆ W_{1..i-1}

        All reads should come from writes by earlier cells.
        Excludes:
        - Builtins (print, len, range, etc.)
        - Imported modules and functions
        - "Ambient" variables (exist in namespace but not written by any cell)

        The "ambient" exclusion handles practical cases where notebooks start
        with pre-existing data (loaded datasets, injected variables, etc.)
        that wasn't written by earlier cells. The formal model assumes an
        empty starting namespace, but real notebooks often have initial state.

        Formal ref: main.tex §3.2, FORMAL_DEVELOPMENT.md §3.2, line 177
        """
        import builtins
        import types

        R_i = tracking.reads_before_writes or set()
        if not R_i:
            return None

        # Handle case where user_ns is a checkpoint (in tests)
        if hasattr(user_ns, 'namespace'):
            ns_dict = user_ns.namespace
        elif isinstance(user_ns, dict):
            ns_dict = user_ns
        else:
            # Can't check WriteBeforeRead without a proper namespace
            return None

        # Compute W_{1..i-1} (what earlier cells wrote)
        all_writes_before = _writes_in_range(
            self._notebook_state, self._cell_order, 0, cell_position - 1
        )

        # Find reads not covered by earlier writes
        missing = R_i - all_writes_before

        # Flag variables that are NOT in the namespace (would cause NameError)
        # This is the WriteBeforeRead violation: reading something that doesn't exist
        user_missing: Set[str] = set()
        for var in missing:
            # Skip builtins (print, len, range, etc.)
            if hasattr(builtins, var):
                continue
            # Flag if variable is NOT in namespace - this is an undefined read
            if var not in ns_dict:
                user_missing.add(var)

        if user_missing:
            cell_alpha = self._cell_id_to_alpha(cell_id)
            vars_str = format_variable_list(sorted(user_missing))
            message = f"Cell {cell_alpha} reads {vars_str} not written by earlier cells"
            return ReproducibilityError(
                error_type=ErrorType.WRITE_BEFORE_READ,
                cell_id=cell_id,
                locations=sorted(user_missing),
                message=message,
            )
        return None

    def _check_no_read_and_write(
        self,
        cell_id: str,
        tracking: TrackingData,
    ) -> Optional[ReproducibilityError]:
        """
        Check NoReadAndWrite predicate: Rᵢ ∩ Wᵢ = ∅

        Cell should not both read and write the same location.

        Formal ref: main.tex §3.2, FORMAL_DEVELOPMENT.md §3.2, line 176

        Note: The implementation uses reads_before_writes which already
        excludes locations that are written before being read. So this
        check catches variables that are first read, then written - patterns
        like `x = x + 1` or `df['col'] = df['col'] * 2`.

        This predicate ensures cells have clear input/output boundaries.

        Implementation note: We include column_writes.keys() in W_i because
        in-place DataFrame column modifications (e.g., df['a'] = df['a'] * 2)
        don't trigger namespace-level writes (tracking.writes), only column-level
        writes. Without this, patterns like `df['a'] = df['a'] * 10` would pass
        the check even though they read and write the same location.
        """
        # R_i: variables read (including those with column-level reads)
        R_i = (tracking.reads_before_writes or set()) | set(
            (tracking.column_reads_before_writes or {}).keys()
        )

        # W_i: variables written (including those with column-level writes)
        # Column-level writes (e.g., df['a'] = ...) don't appear in tracking.writes
        # because the variable binding doesn't change, only the object's contents.
        W_i = (tracking.writes or set()) | set(
            (tracking.column_writes or {}).keys()
        )

        # Build detailed location list with column info
        locations: List[str] = []

        # Variable-level overlap (now includes vars with column-only access)
        var_overlap = R_i & W_i

        # For each overlapping variable, check if we have column-level detail
        col_reads = tracking.column_reads_before_writes or {}
        col_writes = tracking.column_writes or {}

        for var in sorted(var_overlap):
            var_col_reads = col_reads.get(var, set())
            var_col_writes = col_writes.get(var, set())
            col_overlap = var_col_reads & var_col_writes

            if col_overlap:
                # Have column-level detail with overlap - show each overlapping column
                for col in sorted(col_overlap):
                    locations.append(f"{var}.{col}")
            elif var_col_reads and var_col_writes:
                # Have column detail but no overlap - different columns read vs written.
                # Check if the variable binding itself was written (not just columns).
                if var in (tracking.writes or set()):
                    # Variable was read AND the binding was written (e.g., df = transform(df))
                    # This is a NoReadAndWrite violation at the variable level, even though
                    # different columns were read vs written internally.
                    locations.append(var)
                # Otherwise, just column operations with no variable reassignment - this is fine
                # (e.g., reading df['a'] and writing df['b'] without reassigning df).
            elif var_col_writes and not var_col_reads:
                # Have column writes but no column reads.
                # Check if the variable binding itself was written (not just columns).
                if var in (tracking.writes or set()):
                    # Variable was read AND the binding was written (e.g., a = feature_engineer(a))
                    # This is a NoReadAndWrite violation at the variable level.
                    # The column writes may be from internal function operations that got
                    # attributed to this variable when the new object was assigned.
                    locations.append(var)
                # Otherwise, variable was read to access the object, but only columns were written.
                # This is NOT a NoReadAndWrite violation because reading the variable binding
                # (e.g., `df`) and writing to its column (e.g., `df['age']`) are semantically
                # different locations. The formal predicate Rᵢ ∩ Wᵢ = ∅ requires the SAME
                # location to be both read and written.
                # Example: `df['age'] = 1` reads `df` (binding) and writes `df.age` (column).
            elif var_col_reads and not var_col_writes:
                # Have column reads but no column writes
                # Check if variable-level write overlaps with column reads
                if var in (tracking.writes or set()):
                    # Columns were read, then variable itself was written
                    locations.append(var)
            else:
                # No column info - just show variable
                locations.append(var)

        if locations:
            cell_alpha = self._cell_id_to_alpha(cell_id)
            vars_str = format_variable_list(locations)
            message = f"Cell {cell_alpha} reads and writes the same locations: {vars_str}"
            return ReproducibilityError(
                error_type=ErrorType.NO_READ_AND_WRITE,
                cell_id=cell_id,
                locations=locations,
                message=message,
            )
        return None

    def _check_unrecoverable_mutation(
        self,
        cell_id: str,
        unrecoverable_changed_vars: Set[str],
        unrecoverable_column_changed: Dict[str, List[str]],
    ) -> Optional[ReproducibilityError]:
        """
        Check that all mutations are recoverable (rebound or column-tracked).

        In-place mutations (diff-detected changes NOT in tracking.writes) cannot
        be restored by re-executing the cell. These are errors — the cell mutated
        state it doesn't own.

        Formal ref: RecoverableMutation(R', W', i)
        """
        locations: List[str] = []
        for var in sorted(unrecoverable_changed_vars):
            locations.append(var)
        for var, cols in sorted(unrecoverable_column_changed.items()):
            for col in sorted(cols):
                locations.append(f"{var}.{col}")
        if not locations:
            return None
        cell_alpha = self._cell_id_to_alpha(cell_id)
        vars_str = ", ".join(locations)
        return ReproducibilityError(
            error_type=ErrorType.UNRECOVERABLE_MUTATION,
            cell_id=cell_id,
            locations=locations,
            message=(
                f"Cell {cell_alpha} mutated {vars_str} in place without rebinding. "
                f"Re-executing this cell cannot restore these values."
            ),
        )

    def _check_skipped_writers(self, cell_id: str) -> List[Tuple[str, Optional[str], str]]:
        """
        Check for skipped intermediate writers.

        Returns list of (loc, actual_writer, expected_writer) tuples.
        """
        skipped_writers: List[Tuple[str, Optional[str], str]] = []
        cell_reads = self._notebook_state.reads.get(cell_id, set())
        for loc in cell_reads:
            actual_writer = self._notebook_state.last_writer.get(loc)
            expected_writer = self._notebook_state.last_writer_for(loc, cell_id)
            if expected_writer is not None and actual_writer != expected_writer:
                skipped_writers.append((loc, actual_writer, expected_writer))
        return skipped_writers

    def _compute_forward_staleness(
        self,
        current_namespace: dict,
        old_writes: Set[str],
        current_writes: Set[str],
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
        changed_file_paths: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Compute ForwardStale for all cells j > i.

        Dispatches to syntactic or semantic implementation based on staleness_mode.

        Formal ref: ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
        FORMAL_DEVELOPMENT.md §3.3, line 187
        FORMAL_DEVELOPMENT.md §10 (Staleness Computation Modes)
        """
        if self._staleness_mode == StalenessMode.SYNTACTIC:
            return self._compute_forward_staleness_syntactic(
                old_writes, current_writes, changed_vars, column_changed, just_executed, my_position, changed_file_paths
            )
        else:
            return self._compute_forward_staleness_semantic(
                current_namespace, old_writes, current_writes, changed_vars, column_changed, just_executed, my_position, changed_file_paths
            )

    def _compute_forward_staleness_syntactic(
        self,
        old_writes: Set[str],
        current_writes: Set[str],
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
        changed_file_paths: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Syntactic ForwardStale: (Wᵢ ∪ W'ᵢ ∪ ΔV) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅ for j > i.

        Cell j becomes stale if cell i's old writes, new writes, OR diff-detected
        changes overlap with what cell j reads or writes.

        Uses pure set intersection on R/W sets. Does not use checkpoints for
        staleness comparison. Staleness is monotonic (once stale, stays stale
        until re-executed).

        Note: changed_vars (from diff) is essential for detecting in-place mutations
        like df['col'] = ... which may not appear in TrackingDict's writes set.

        Formal ref: FORMAL_DEVELOPMENT.md §3.3, §10.1
        """
        all_warnings: List[str] = []

        # Wᵢ ∪ W'ᵢ ∪ ΔV: all locations cell i has written (old, new, or diff-detected)
        # Include changed_vars to catch in-place mutations that TrackingDict misses
        W_i_union = old_writes | current_writes | changed_vars

        cells_below = self._cell_order[my_position + 1:]
        for cell_id in cells_below:
            cell_tracking = self._notebook_state.get_tracking(cell_id)
            if cell_tracking is None:
                continue

            if not self._notebook_state.is_clean(cell_id):
                # Update SKIPPED_UPSTREAM → FORWARD_STALE if expected cell just ran
                if ENABLE_SKIPPED_UPSTREAM:
                    reasons = self._notebook_state.get_reasons(cell_id)
                    for reason in list(reasons):
                        if (reason.type == ReasonType.SKIPPED_UPSTREAM and
                                reason.expected_cell_id == just_executed):
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.FORWARD_STALE, loc=reason.loc, cell_id=just_executed)
                            )
                continue

            # File staleness check
            if changed_file_paths and cell_tracking.file_reads_before_writes:
                overlap_files = changed_file_paths & cell_tracking.file_reads_before_writes
                if overlap_files:
                    for fpath in overlap_files:
                        self._notebook_state.add_reason(
                            cell_id, Reason(ReasonType.FORWARD_STALE, loc=fpath, cell_id=just_executed)
                        )
                    continue

            # Syntactic check: (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
            cell_reads = cell_tracking.reads_before_writes or set()
            cell_writes = cell_tracking.writes or set()
            write_overlap = W_i_union & cell_writes  # Overlap with writes

            # Use column-aware overlap check for read staleness.
            # _has_relevant_overlap_by_id handles BOTH regular variables AND DataFrames:
            # - Regular vars (no column info): returns True conservatively
            # - DataFrames with column tracking: returns True only if columns overlap
            # Example: Cell B reads df['eruptions'], Cell C writes df['cluster']
            #   → columns don't overlap, so B should NOT be stale
            has_relevant_read_overlap = self._has_relevant_overlap_by_id(cell_id, W_i_union, column_changed)

            if has_relevant_read_overlap or write_overlap:
                # Determine which variables caused staleness by checking each one
                cell_column_reads = self._notebook_state.get_column_reads(cell_id)
                stale_vars: set = set()
                for var in W_i_union & cell_reads:
                    changed_cols = set(column_changed.get(var, []))
                    read_cols = cell_column_reads.get(var, None)
                    # Include var if no column info (conservative) or columns overlap
                    if not changed_cols or read_cols is None or (changed_cols & read_cols):
                        stale_vars.add(var)

                # Handle read overlaps (FORWARD_STALE or SKIPPED_UPSTREAM)
                for var in stale_vars:
                    if ENABLE_SKIPPED_UPSTREAM:
                        expected_writer = self._notebook_state.last_writer_for(var, cell_id)
                        if expected_writer != just_executed and expected_writer is not None:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.SKIPPED_UPSTREAM, loc=var, cell_id=just_executed, expected_cell_id=expected_writer)
                            )
                        else:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=just_executed)
                            )
                    else:
                        self._notebook_state.add_reason(
                            cell_id,
                            Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=just_executed)
                        )
                # Then handle write-only overlaps (WRITE_OVERLAP - no convergence)
                for var in write_overlap - stale_vars:
                    self._notebook_state.add_reason(
                        cell_id,
                        Reason(ReasonType.WRITE_OVERLAP, loc=var, cell_id=just_executed)
                    )

        return self._notebook_state.get_stale_cells(), all_warnings

    def _compute_forward_staleness_semantic(
        self,
        current_namespace: dict,
        old_writes: Set[str],
        current_writes: Set[str],
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
        changed_file_paths: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Semantic ForwardStale: (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅ with semantic diff for reads.

        For read overlaps: uses checkpoint diff comparison for precise staleness detection.
        For write-only overlaps: marks stale immediately (no convergence for writes).
        For removed writes (W'_i - W_i): marks stale immediately (dependency removed).
        Staleness is non-monotonic for reads (can be cleared when values converge).

        Formal ref: FORMAL_DEVELOPMENT.md §3.3, §10.2
        """
        all_warnings: List[str] = []

        # Wᵢ ∪ W'ᵢ: all locations cell i has written (old or new)
        W_i_union = old_writes | current_writes

        cells_below = self._cell_order[my_position + 1:]
        for cell_id in cells_below:
            cell_tracking = self._notebook_state.get_tracking(cell_id)
            if cell_tracking is None:
                continue

            if not self._notebook_state.is_clean(cell_id):
                # Update SKIPPED_UPSTREAM → FORWARD_STALE if expected cell just ran
                if ENABLE_SKIPPED_UPSTREAM:
                    reasons = self._notebook_state.get_reasons(cell_id)
                    for reason in list(reasons):
                        if (reason.type == ReasonType.SKIPPED_UPSTREAM and
                                reason.expected_cell_id == just_executed):
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.FORWARD_STALE, loc=reason.loc, cell_id=just_executed)
                            )
                continue

            # File staleness check
            if changed_file_paths and cell_tracking.file_reads_before_writes:
                overlap_files = changed_file_paths & cell_tracking.file_reads_before_writes
                if overlap_files:
                    for fpath in overlap_files:
                        self._notebook_state.add_reason(
                            cell_id, Reason(ReasonType.FORWARD_STALE, loc=fpath, cell_id=just_executed)
                        )
                    continue

            # Compute syntactic overlap: (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ)
            cell_reads = cell_tracking.reads_before_writes or set()
            cell_writes = cell_tracking.writes or set()
            read_overlap = W_i_union & cell_reads
            write_overlap = W_i_union & cell_writes

            # Skip if no overlap at all
            if not read_overlap and not write_overlap:
                continue

            # Write-only overlap: mark stale immediately (no convergence for writes)
            # Use WRITE_OVERLAP reason type so convergence won't clear this staleness
            if write_overlap and not read_overlap:
                for var in write_overlap:
                    self._notebook_state.add_reason(
                        cell_id,
                        Reason(ReasonType.WRITE_OVERLAP, loc=var, cell_id=just_executed)
                    )
                continue

            # Check for removed-writes overlap with reads: (W'_i - W_i) ∩ R_j
            # If cell reads a variable that executing cell USED TO write but no longer
            # writes, the dependency relationship has changed. Mark stale with WRITE_OVERLAP
            # since this can't converge (the source of the variable has changed).
            removed_writes = old_writes - current_writes
            old_writes_read_overlap = removed_writes & cell_reads
            if old_writes_read_overlap:
                for var in old_writes_read_overlap:
                    self._notebook_state.add_reason(
                        cell_id,
                        Reason(ReasonType.WRITE_OVERLAP, loc=var, cell_id=just_executed)
                    )
                # Also handle any write_overlap
                for var in write_overlap:
                    self._notebook_state.add_reason(
                        cell_id,
                        Reason(ReasonType.WRITE_OVERLAP, loc=var, cell_id=just_executed)
                    )
                continue

            # Read overlap: use semantic diff check (skip if no relevant column overlap)
            if not self._has_relevant_overlap_by_id(cell_id, changed_vars, column_changed):
                # No column-level overlap for reads, but mark stale for write overlap
                if write_overlap:
                    for var in write_overlap:
                        self._notebook_state.add_reason(
                            cell_id,
                            Reason(ReasonType.WRITE_OVERLAP, loc=var, cell_id=just_executed)
                        )
                continue

            # Semantic check: expensive diff comparison
            pre_checkpoint = self.checkpoints.get(f"{PRE_CHECKPOINT_PREFIX}{cell_id}")
            if pre_checkpoint is None:
                continue

            diff_result = MemoryCheckpoint.diff(
                pre_checkpoint,
                current_namespace,
                keys_to_include=cell_tracking.reads_before_writes,
                use_leq=True,
                column_rbw=cell_tracking.column_reads_before_writes,
                structural_reads=cell_tracking.structural_reads,
                structural_mode=self._structural_mode,
            )

            if diff_result.differences:
                for var in diff_result.differences.keys():
                    if ENABLE_SKIPPED_UPSTREAM:
                        expected_writer = self._notebook_state.last_writer_for(var, cell_id)
                        if expected_writer != just_executed and expected_writer is not None:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.SKIPPED_UPSTREAM, loc=var, cell_id=just_executed, expected_cell_id=expected_writer)
                            )
                        else:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=just_executed)
                            )
                    else:
                        self._notebook_state.add_reason(
                            cell_id,
                            Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=just_executed)
                        )

            # Capture warnings
            if diff_result.warnings:
                affected_alpha = self._cell_id_to_alpha(cell_id)
                mutating_alpha = self._cell_id_to_alpha(just_executed)

                for warning in diff_result.warnings:
                    var_match = re.match(r"Structural change at (\w+):", warning)
                    if var_match:
                        var_name = var_match.group(1)
                        cell_structural_values = self._notebook_state.get_structural_reads_values(cell_id)
                        read_values = cell_structural_values.get(var_name, {})
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
                            detail_match = re.search(r"Structural change at \w+: (.+)$", warning)
                            if detail_match:
                                changes.append(detail_match.group(1))

                        formatted = format_structural_warning(
                            mutating_alpha, affected_alpha, var_name, read_values, changes
                        )
                        all_warnings.append(formatted)
                    else:
                        all_warnings.append(f"Cell {affected_alpha}: {warning}")

        return self._notebook_state.get_stale_cells(), all_warnings

    def _check_convergence(self, current_namespace: dict) -> List[str]:
        """Check all stale cells for convergence and clear staleness if inputs match.

        Only called in SEMANTIC mode. A stale cell j converges when:
            1. diff(pre_checkpoint[j], namespace, R_j) = ∅
            2. The cell's only staleness reasons are FORWARD_STALE

        Cells with other staleness reasons (like SKIPPED_UPSTREAM, CODE_CHANGED)
        should NOT be cleared by convergence - those reasons indicate structural
        issues beyond just value changes.

        Formal ref: FORMAL_DEVELOPMENT.md §10.2 Convergence rule

        Returns:
            List of cell IDs that were cleared (no longer stale)
        """
        if self._staleness_mode != StalenessMode.SEMANTIC:
            return []

        cleared: List[str] = []
        for cell_id in self._notebook_state.get_stale_cells():
            # Only consider cells whose only reasons are FORWARD_STALE
            # (convergence doesn't fix SKIPPED_UPSTREAM, CODE_CHANGED, etc.)
            reasons = self._notebook_state.get_reasons(cell_id)
            if not reasons:
                continue
            if not all(r.type == ReasonType.FORWARD_STALE for r in reasons):
                continue

            cell_tracking = self._notebook_state.get_tracking(cell_id)
            if cell_tracking is None:
                continue

            pre_checkpoint = self.checkpoints.get(f"{PRE_CHECKPOINT_PREFIX}{cell_id}")
            if pre_checkpoint is None:
                continue

            diff_result = MemoryCheckpoint.diff(
                pre_checkpoint,
                current_namespace,
                keys_to_include=cell_tracking.reads_before_writes,
                use_leq=True,
                column_rbw=cell_tracking.column_reads_before_writes,
                structural_reads=cell_tracking.structural_reads,
                structural_mode=self._structural_mode,
            )

            if not diff_result.differences:
                # Converged! Clear staleness
                self._notebook_state.set_clean(cell_id)
                cleared.append(cell_id)

        return cleared

    def _compute_backward_staleness(
        self,
        current_namespace: dict,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
    ) -> List[str]:
        """
        Compute BackwardStale for all cells j < i.

        When cell i writes to a variable that earlier cell j read, j becomes stale.
        Dispatches to syntactic or semantic implementation based on staleness_mode.

        Formal ref: FORMAL_DEVELOPMENT.md §10.1, §10.2

        Returns:
            List of cell IDs that were marked stale
        """
        if self._staleness_mode == StalenessMode.SYNTACTIC:
            return self._compute_backward_staleness_syntactic(
                changed_vars, column_changed, just_executed, my_position
            )
        else:
            return self._compute_backward_staleness_semantic(
                current_namespace, changed_vars, column_changed, just_executed, my_position
            )

    def _compute_backward_staleness_syntactic(
        self,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
    ) -> List[str]:
        """
        Syntactic BackwardStale: W_i ∩ R_j ≠ ∅ for j < i.

        Uses pure set intersection on R/W sets. Marks cells before i as stale
        if they read variables that i wrote to. Also checks column-level overlap
        for DataFrame column changes.

        Formal ref: FORMAL_DEVELOPMENT.md §10.1
        """
        newly_stale: List[str] = []

        for prior_cell_id in self._cell_order[:my_position]:
            if not self._notebook_state.is_clean(prior_cell_id):
                continue  # Already stale

            if not self._notebook_state.has_record(prior_cell_id):
                continue  # Never executed

            prior_reads = self._notebook_state.reads.get(prior_cell_id, set())

            # Use column-aware overlap check for backward staleness.
            # This handles BOTH regular variables AND DataFrames:
            # - Regular vars: no column info → returns True (conservative)
            # - DataFrames: only returns True if columns actually overlap
            has_relevant_overlap = self._has_relevant_overlap_by_id(
                prior_cell_id, changed_vars, column_changed
            )

            if has_relevant_overlap:
                # Determine which variables caused staleness (column-aware)
                prior_column_reads = self._notebook_state.get_column_reads(prior_cell_id)
                for var in changed_vars & prior_reads:
                    changed_cols = set(column_changed.get(var, []))
                    read_cols = prior_column_reads.get(var, None)
                    # Include var if no column info (conservative) or columns overlap
                    if not changed_cols or read_cols is None or (changed_cols & read_cols):
                        self._notebook_state.add_reason(
                            prior_cell_id,
                            Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=just_executed)
                        )
                newly_stale.append(prior_cell_id)

        return newly_stale

    def _compute_backward_staleness_semantic(
        self,
        current_namespace: dict,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
    ) -> List[str]:
        """
        Semantic BackwardStale: W_i ∩ R_j ≠ ∅ AND diff(pre_checkpoint[j], namespace, R_j) ≠ ∅.

        Uses checkpoint diff comparison for precise staleness detection.
        Only marks cells stale if their input values actually changed.

        Formal ref: FORMAL_DEVELOPMENT.md §10.2
        """
        newly_stale: List[str] = []

        for prior_cell_id in self._cell_order[:my_position]:
            if not self._notebook_state.is_clean(prior_cell_id):
                continue  # Already stale

            if not self._notebook_state.has_record(prior_cell_id):
                continue  # Never executed

            prior_reads = self._notebook_state.reads.get(prior_cell_id, set())

            # Quick filter: check variable-level overlap
            if not (changed_vars & prior_reads):
                continue

            # Also check column-level overlap for precision
            if not self._has_relevant_overlap_by_id(prior_cell_id, changed_vars, column_changed):
                continue

            # Semantic check: did prior cell's inputs actually change?
            pre_checkpoint = self.checkpoints.get(f"{PRE_CHECKPOINT_PREFIX}{prior_cell_id}")
            if pre_checkpoint is None:
                continue

            prior_tracking = self._notebook_state.get_tracking(prior_cell_id)
            if prior_tracking is None:
                continue

            diff_result = MemoryCheckpoint.diff(
                pre_checkpoint,
                current_namespace,
                keys_to_include=prior_tracking.reads_before_writes,
                use_leq=True,
                column_rbw=prior_tracking.column_reads_before_writes,
                structural_reads=prior_tracking.structural_reads,
                structural_mode=self._structural_mode,
            )

            if diff_result.differences:
                for var in diff_result.differences.keys():
                    self._notebook_state.add_reason(
                        prior_cell_id,
                        Reason(ReasonType.FORWARD_STALE, loc=var, cell_id=just_executed)
                    )
                newly_stale.append(prior_cell_id)

        return newly_stale

    def _has_relevant_overlap_by_id(
        self,
        cell_id: str,
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
            cell_id: The cell to check
            changed_vars: Variables that changed in current execution (value-level)
            column_changed: Dict mapping var names to changed column names

        Returns:
            True if the cell might be affected by the changes
        """
        reads = self._notebook_state.reads.get(cell_id, set())
        # Include both value-level changes AND column-level changes in overlap check
        all_changed = changed_vars | set(column_changed.keys())
        var_overlap = reads & all_changed

        if not var_overlap:
            return False  # No variable-level overlap at all

        cell_column_reads = self._notebook_state.get_column_reads(cell_id)
        cell_structural_reads = self._notebook_state.get_structural_reads(cell_id)

        # Check column-level overlap for each overlapping variable
        for var in var_overlap:
            changed_cols = set(column_changed.get(var, []))
            read_cols = cell_column_reads.get(var, None)

            if not changed_cols or read_cols is None:
                # No column info on one or both sides - conservative: assume overlap
                return True

            if changed_cols & read_cols:
                # Actual column overlap found
                return True

            # Check if cell has structural reads for this variable
            # If columns were added/changed and cell read structure, it might be affected
            if var in cell_structural_reads and changed_cols:
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
        return self._notebook_state.get_stale_cells()

    def compute_all_stale_cells(self, current_namespace: dict) -> List[str]:
        """
        Recompute staleness for ALL cells from scratch.

        Unlike incremental updates, this checks every executed cell against
        the current namespace state. Use this when you need guaranteed
        accuracy (e.g., after external namespace modifications).

        Args:
            current_namespace: The current live user namespace dict

        Returns:
            List of cell IDs that are currently stale (in document order)
        """
        
        for cell_id in self._cell_order:
            cell_tracking = self._notebook_state.get_tracking(cell_id)
            if cell_tracking is None:
                continue
            ckpt_name = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
            if not self.checkpoints.exists(ckpt_name):
                continue  # Checkpoint may have been deleted (syntactic mode)
            pre_checkpoint = self.checkpoints.get(ckpt_name)

            diff_result = MemoryCheckpoint.diff(
                pre_checkpoint,
                current_namespace,
                keys_to_include=cell_tracking.reads_before_writes,
                use_leq=True,
                column_rbw=cell_tracking.column_reads_before_writes,
                structural_reads=cell_tracking.structural_reads,
                structural_mode=self._structural_mode,
            )

            if diff_result.differences:
                # Track reasons: FORWARD_STALE for each changed variable
                for var in diff_result.differences.keys():
                    self._notebook_state.add_reason(
                        cell_id, Reason(ReasonType.FORWARD_STALE, loc=var)
                    )

        return self._notebook_state.get_stale_cells()

    def mark_cell_edited(self, cell_id: str) -> List[str]:
        """[EDIT] Mark edited cell stale (§2.3).

        With provenance tracking (§1.8.5), no special handling is needed on edit.
        Provenance persists until another cell writes to those locations, so
        forward contamination is automatically detected when earlier cells read
        values whose provenance points to later cells.

        Returns current stale cells list.
        """
        from flowbook.util.output import log, timer

        with timer(key="order:Inst-Edit", message=f"[Inst-Edit] Cell {cell_id}"):
            if not self._notebook_state.has_record(cell_id):
                return self.get_stale_cells()  # Unexecuted cell — no-op

            # Track reason: CODE_CHANGED via NotebookState
            self._notebook_state.handle_edit(cell_id)
            log(f"[Inst-Edit] Cell {cell_id} marked stale (CODE_CHANGED)")

            return self.get_stale_cells()

    def get_execution_records_size(self) -> int:
        """
        Calculate approximate memory size of execution records in bytes.

        This measures the overhead of storing per-cell execution metadata
        including tracking data (reads, writes, column tracking).

        Returns:
            Approximate memory size in bytes.
        """
        import sys

        total = sys.getsizeof(self._notebook_state.tracking_data)

        for cell_id, td in self._notebook_state.tracking_data.items():
            total += sys.getsizeof(cell_id)
            total += sys.getsizeof(td)

            # Sets for reads/writes
            if td.reads_before_writes:
                total += sys.getsizeof(td.reads_before_writes)
                for r in td.reads_before_writes:
                    total += sys.getsizeof(r)
            if td.writes:
                total += sys.getsizeof(td.writes)
                for w in td.writes:
                    total += sys.getsizeof(w)

            # Column tracking dicts
            for attr in ['column_writes', 'column_reads_before_writes']:
                d = getattr(td, attr, None)
                if d:
                    total += sys.getsizeof(d)
                    for k, v in d.items():
                        total += sys.getsizeof(k) + sys.getsizeof(v)

            # Structural tracking
            if td.structural_reads:
                total += sys.getsizeof(td.structural_reads)
                for k, v in td.structural_reads.items():
                    total += sys.getsizeof(k) + sys.getsizeof(v)

            # structural_reads_values dict (stored separately in NotebookState)
            structural_vals = self._notebook_state.get_structural_reads_values(cell_id)
            if structural_vals:
                total += sys.getsizeof(structural_vals)
                for k, v in structural_vals.items():
                    total += sys.getsizeof(k) + sys.getsizeof(v)

            # typed_changes list (stored separately in NotebookState)
            typed_changes = self._notebook_state.get_typed_changes(cell_id)
            if typed_changes:
                total += sys.getsizeof(typed_changes)
                for change in typed_changes:
                    total += sys.getsizeof(change)

        return total

    def measure_rerun_overhead(
        self,
        cell_id: str,
        namespace: dict,
    ) -> Dict[str, Any]:
        """
        Measure the overhead of re-running a cell without actually executing code.

        This is used by compare-baseline's --rerun=N option to measure worst-case
        overhead at quartile-boundary cells. It performs:
        1. Take a full checkpoint (timed)
        2. Full diff against the checkpoint (timed - will be empty but measures work)
        3. Full check using the cell's original R/W (timed)

        Note: This method does NOT restore state. The checkpoint taken becomes
        part of the checkpoint store and accumulates overhead.

        Args:
            cell_id: ID of the cell to measure rerun overhead for
            namespace: Current user namespace (for checkpoint/diff)

        Returns:
            Dictionary with timing and checkpoint cost data:
            {
                "cell_id": str,
                "checkpoint_ms": float,
                "diff_ms": float,
                "check_ms": float,
                "total_overhead_ms": float,
                "checkpoint_by_var": {var: mb, ...},
                "checkpoint_var_costs": {var: {"bytes": int, "deepcopy_ms": float}, ...}
            }
        """
        from flowbook.util.output import timer

        result = {
            "cell_id": cell_id,
            "checkpoint_ms": 0.0,
            "check_ms": 0.0,
            "total_overhead_ms": 0.0,
            "checkpoint_by_var": {},
            "checkpoint_var_costs": {},
        }

        # Get the cell's tracking data (original R/W)
        tracking = self._notebook_state.get_tracking(cell_id)
        if tracking is None:
            # Cell has no execution record - return zeros
            return result

        # 1. Take a full checkpoint (timed)
        checkpoint_name = f"_rerun_overhead_{cell_id}"
        with timer(key="rerun:checkpoint", message=f"[Rerun] Checkpoint for {cell_id}") as ckpt_timer:
            ns_dict = dict(namespace) if not isinstance(namespace, dict) else namespace
            self.checkpoints.save(checkpoint_name, ns_dict, max_size_mb=None)
            pre_checkpoint = self.checkpoints.get(checkpoint_name)

        result["checkpoint_ms"] = ckpt_timer.duration()

        # Get checkpoint variable costs
        if hasattr(self.checkpoints, '_var_memory_costs_by_checkpoint'):
            var_costs = self.checkpoints._var_memory_costs_by_checkpoint.get(checkpoint_name, {})
            # Convert to checkpoint_by_var (MB) and checkpoint_var_costs
            for var_name, cost_info in var_costs.items():
                result["checkpoint_by_var"][var_name] = cost_info.get("bytes", 0) / (1024 * 1024)
                result["checkpoint_var_costs"][var_name] = {
                    "bytes": cost_info.get("bytes", 0),
                    "deepcopy_ms": cost_info.get("deepcopy_ms", 0.0),
                }

        # 2. Check phase: diff + conflict resolution (timed together, like normal execution)
        with timer(key="rerun:check", message=f"[Rerun] Check for {cell_id}") as check_timer:
            # Prepare accessed columns for diff
            all_accessed_columns = {}
            for var, cols in tracking.column_reads_before_writes.items():
                all_accessed_columns[var] = set(cols)
            for var, cols in tracking.column_writes.items():
                if var in all_accessed_columns:
                    all_accessed_columns[var].update(cols)
                else:
                    all_accessed_columns[var] = set(cols)

            # Get accessed variables (like normal execution does)
            accessed_vars = tracking.reads_before_writes | tracking.writes

            # Diff only accessed variables (like normal execution)
            # Note: pass empty set directly, not None (None means "diff everything")
            current_diff = MemoryCheckpoint.diff(
                pre_checkpoint,
                namespace,
                keys_to_include=accessed_vars,
                use_leq=False,
                column_rbw=all_accessed_columns,
                structural_reads={},
                structural_mode=self._structural_mode,
            )

            # Conflict resolution
            my_position = self._get_position(cell_id)
            if my_position >= 0:
                from flowbook.kernel.changes import ValueChanged

                # Simulate forward contamination check
                my_read_events = tracking.to_read_events()
                for later_cell_id in self._cell_order[my_position + 1:]:
                    if not self._notebook_state.has_record(later_cell_id):
                        continue
                    later_tracking = self._notebook_state.get_tracking(later_cell_id)
                    if later_tracking is None:
                        continue
                    later_changes = [ValueChanged(variable=var) for var in later_tracking.writes]
                    if later_changes and my_read_events:
                        self._conflict_resolver.get_violations(later_changes, my_read_events)

                # Simulate backward mutation check
                for prior_cell_id in self._cell_order[:my_position]:
                    if not self._notebook_state.has_record(prior_cell_id):
                        continue
                    if not self._notebook_state.is_clean(prior_cell_id):
                        continue
                    prior_tracking = self._notebook_state.get_tracking(prior_cell_id)
                    if prior_tracking is None:
                        continue
                    prior_reads = prior_tracking.to_read_events()
                    if prior_reads:
                        fake_changes = [ValueChanged(variable=var) for var in prior_tracking.reads_before_writes]
                        if fake_changes:
                            self._conflict_resolver.get_violations(fake_changes, prior_reads)

        result["check_ms"] = check_timer.duration()
        result["total_overhead_ms"] = result["checkpoint_ms"] + result["check_ms"]

        # Clean up the temporary checkpoint to avoid memory accumulation
        self.checkpoints.delete(checkpoint_name)

        return result

    def reset(self) -> None:
        """Clear all state. Called on kernel restart."""
        self.seq_counter = 0
        self._cell_order = []
        self._notebook_state.clear()  # Clear status, R, W, last_writer, tracking_data
        self._pending_checkpoint_deletion = None
        self._pending_snapshot = None

    def rollback_last_check(self) -> None:
        """
        Rollback state changes from the most recent check() call.

        Called by kernel when execution is rejected and namespace is rolled back.
        This ensures the enforcer's analysis state matches the rolled-back namespace.

        The rollback restores:
        - Per-cell state (reads, writes, status, tracking_data, etc.)
        - last_writer entries that pointed to the cell
        - column_last_writer entries that pointed to the cell
        - Clears pending checkpoint deletion (since the cell was rejected,
          we shouldn't delete its checkpoint on the next execution)
        """
        if self._pending_snapshot is not None:
            self._notebook_state.restore_cell_state(self._pending_snapshot)
            self._pending_snapshot = None
        # Clear pending checkpoint deletion since the cell execution was rolled back.
        # If we don't clear this, re-executing the same cell would delete the newly
        # created checkpoint (same name) at the start of check(), causing a crash
        # when trying to restore later.
        self._pending_checkpoint_deletion = None


def _extract_column_changes(
    diff_result: MemoryCheckpointDiffResult, tracking: TrackingData
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

        # For newly created variables ("Variable was added"), all written columns
        # are considered "changed" since they didn't exist before
        if isinstance(diff_node, ValueComparison):
            if diff_node.message and "was added" in diff_node.message:
                # Variable is new - all tracked column_writes are "changed"
                if var_name in tracking.column_writes:
                    column_changed[var_name] = sorted(tracking.column_writes[var_name])
                continue

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


def _get_value_level_changed_vars(typed_changes: List) -> Set[str]:
    """
    Extract variables that had value-level changes (not just column changes).

    Value-level changes indicate the variable identity or structure changed:
    - ValueChanged: Variable was reassigned or replaced
    - RowsAdded/RowsRemoved: DataFrame row count changed
    - IndexChanged: DataFrame/Series index changed

    Column-only changes do NOT make a variable "changed" at value level:
    - ColumnAdded/ColumnModified/ColumnRemoved: Only specific columns changed
    - DtypeChanged: Only column dtype changed

    This distinction matters for last_writer tracking: if only columns changed,
    the variable's "owner" shouldn't change. The original creator still owns
    the variable; the mutator just modified columns within it.

    Args:
        typed_changes: List of Change objects from change_detector

    Returns:
        Set of variable names with value-level changes
    """
    from flowbook.kernel.changes import (
        ValueChanged, RowsAdded, RowsRemoved, IndexChanged
    )

    value_level_types = (ValueChanged, RowsAdded, RowsRemoved, IndexChanged)

    vars_with_value_change: Set[str] = set()
    for change in typed_changes:
        if isinstance(change, value_level_types):
            vars_with_value_change.add(change.variable)

    return vars_with_value_change


def _check_for_truncation(diff_result: MemoryCheckpointDiffResult) -> List[str]:
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
        diff_result: The MemoryCheckpointDiffResult to check

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
    diff_result: MemoryCheckpointDiffResult, truncated_vars: List[str], max_width: int = 120
) -> str:
    """
    Format truncated diff for human-readable display.

    Args:
        diff_result: The MemoryCheckpointDiffResult containing differences
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
    diff_result: MemoryCheckpointDiffResult,
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
        "❌ Reproducibility Violation: Backward Structural Mutation",
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
    Format a forward dependency (contamination) error message.

    A forward dependency occurs when a cell reads a variable that a later cell
    (in document order) has already written. This means the reading cell would
    see "future" state that wouldn't exist in top-to-bottom order.

    Forward contamination now blocks execution with an error.

    Args:
        reading_cell_alpha: Cell that attempted to read the variable (@A notation)
        writing_cell_alpha: Later cell that wrote the variable (@A notation)
        variables: List of affected variable names

    Returns:
        Formatted multi-line message string
    """
    vars_str = format_variable_list(variables)

    lines = [
        "❌ Forward Contamination",
        "",
        f"Cell {reading_cell_alpha} reads {vars_str} which was written by "
        f"downstream cell {writing_cell_alpha}.",
        "",
        "Execution blocked because this cell would read out-of-order state "
        "that would not exist in a top-to-bottom run.",
        "",
        "Fix: Re-run upstream cells to restore reproducible values for these variables.",
    ]

    return "\n".join(lines)


def format_backward_mutation_message(
    mutating_cell_alpha: str,
    affected_cell_alpha: str,
    variables: List[str],
) -> str:
    """
    Format a backward mutation error message.

    A backward mutation occurs when a cell modifies a variable that an
    earlier cell (in document order) reads. This creates a hidden dependency
    where the earlier cell depends on the later cell having run first.

    This format is used both for:
    1. Direct backward mutation (writer runs after reader)
    2. Writer violation on forward contamination (writer ran before reader)

    Args:
        mutating_cell_alpha: Cell that modified the variable (@A notation)
        affected_cell_alpha: Earlier cell that reads the variable (@A notation)
        variables: List of affected variable names

    Returns:
        Formatted message string
    """
    vars_str = format_variable_list(variables)

    return (
        f"Cell {mutating_cell_alpha} modified {vars_str} which Cell {affected_cell_alpha} "
        f"(earlier) reads."
    )


def format_truncation_error(
    cell_alpha: str,
    variables: List[str],
) -> str:
    """
    Format a truncation error message.

    Truncation occurs when the diff tree exceeds max_diffs_per_container,
    meaning we may miss some column/key changes. This is treated as an error
    because reproducibility tracking may be incomplete.

    Args:
        cell_alpha: Cell where truncation occurred (@A notation)
        variables: List of variable names that had truncated diffs

    Returns:
        Formatted multi-line message string
    """
    vars_str = format_variable_list(variables)

    lines = [
        "⚠️ Reproducibility Warning: Diff Truncated",
        "",
        f"Cell {cell_alpha}: Reproducibility diff was truncated for: {vars_str}.",
        "",
        "Tracking may be incomplete. Some column or key changes may not be detected.",
        "",
        "Fix: Consider increasing max_diffs_per_container or simplifying the data structure.",
    ]

    return "\n".join(lines)
