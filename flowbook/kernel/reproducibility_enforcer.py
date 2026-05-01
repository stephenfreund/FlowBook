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

NotebookState: Single source of truth for formal model S = ⟨C, O, Σ, T, R, W⟩
    - status: T (Cell → CellStatus) - clean/stale per cell
    - reads/writes: R, W (Cell → P(Loc)) - per-cell reads and writes
    - tracking_data: Per-cell TrackingData for conflict detection

LocSet operations (wlocs_conflict_rlocs): Column-aware conflict detection
    - Uses typed ReadLoc/WriteLoc with ▷ operator
    - Column-level precision for DataFrame operations
    - Structural attribute awareness for shape/columns changes

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
STALENESS COMPUTATION
================================================================================

Uses checkpoint once to compute accurate R and W sets, then discards it.
Staleness is determined by pure set intersection — monotonic and low memory.

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

See FORMAL_DEVELOPMENT.md §10 for the full formal specification.
"""

import logging
import os
import pprint
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from flowbook.kernel_support.checkpoint import Checkpoint, CheckpointDiffResult
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
import pandas as pd

from flowbook.kernel_support.models import TrackingData
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
)
from flowbook.kernel.loc_ids import StableIdMap
from flowbook.kernel.notebook_state import NotebookState

from flowbook.kernel.change_detector import detect_changes, changes_to_write_locs
from flowbook.kernel.locations import (
    WriteLoc, WriteLocType, WriteLocSet, ReadLocSet,
    wlocs_conflict_rlocs, wlocs_conflict_wlocs,
    tracking_to_readlocset, tracking_to_writelocset,
    readlocset_to_list, writelocset_to_list,
)
from flowbook.util.output import output, timer

# Structural tracking is always ENFORCE — import once for passing to diff functions
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode as _StructuralTrackingMode
_STRUCTURAL_ENFORCE = _StructuralTrackingMode.ENFORCE

_logger = logging.getLogger(__name__)

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
# For full LocSet-based checks, see flowbook.kernel.locations:
#   tracking_to_readlocset(), wlocs_conflict_rlocs()
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



def _changes_to_writelocset(
    changed_vars: Set[str],
    column_changed: Dict[str, List[str]],
) -> WriteLocSet:
    """Convert diff-based changes to a WriteLocSet for staleness checks.

    For variables with column-level change info, emits Col write locs.
    For variables without column info, emits Var write locs (conservative).

    Note: ValueChanged (complete replacement) is handled separately via
    typed_changes in _compute_forward_staleness_syntactic, which adds
    Var(x) for variables whose identity/binding changed.
    """
    locs: Set[WriteLoc] = set()
    for var in changed_vars:
        if var in column_changed:
            for col in column_changed[var]:
                locs.add(WriteLoc.col(var, col))
        else:
            locs.add(WriteLoc.var(var))
    # Also include column changes for vars not in changed_vars
    # (column_changed may have vars that only changed at column level)
    for var, cols in column_changed.items():
        if var not in changed_vars:
            for col in cols:
                locs.add(WriteLoc.col(var, col))
    return frozenset(locs)


class ReproducibilityEnforcer:
    """
    Enforces Reproducibility.

    Tracks cell executions and their read/write sets.
    On each execution, checks for backward mutations and computes staleness.

    Structural attribute conflicts (df.columns, df.shape, etc.) are always enforced.
    """

    def __init__(
        self,
        checkpoints: MemoryCheckpoints,
    ):
        self.checkpoints = checkpoints
        self.seq_counter: int = 0
        self._cell_order: List[str] = []
        # NotebookState is the single source of truth for formal model state:
        # T (status), R (reads), W (writes), and per-cell TrackingData
        self._notebook_state = NotebookState()
        # Deferred checkpoint deletion for syntactic mode - keeps last checkpoint
        # until next cell executes, allowing size queries between executions
        self._pending_checkpoint_deletion: Optional[str] = None
        # Snapshot for rollback if execution is rejected
        self._pending_snapshot: Optional[CellStateSnapshot] = None
        # Stable object identity map for DataFrame location qualifiers.
        # Maps Python id() → stable int via weakref. See loc_ids.py.
        self._stable_map = StableIdMap()

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
        - ForwardStale: j > i, Wᵢ ▷ Rⱼ ≠ ∅ or Wᵢ ▷ output*(Wⱼ) ≠ ∅
        - BackwardStale: j < i, j = LastWriter(W, i, y) for y ∈ Wᵢ

        Uses typed ▷ conflict relation for column-level precision.

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
                deleted_wlocs = self._notebook_state.writes.get(deleted_id, frozenset())
                if not deleted_wlocs:
                    continue  # Deleted cell didn't write anything

                my_position = old_order.index(deleted_id)
                fwd_marked = 0
                bwd_marked = 0

                # ForwardStale: j > i, Wᵢ ▷ Rⱼ ≠ ∅ or Wᵢ ▷ output*(Wⱼ) ≠ ∅
                for cell_id in old_order[my_position + 1:]:
                    if cell_id in deleted_set:
                        continue
                    if not self._notebook_state.is_clean(cell_id):
                        continue

                    cell_read_locs = self._notebook_state.reads.get(cell_id, frozenset())
                    cell_write_locs = self._notebook_state.writes.get(cell_id, frozenset())

                    read_conflicting = wlocs_conflict_rlocs(deleted_wlocs, cell_read_locs)
                    write_conflicting = wlocs_conflict_wlocs(deleted_wlocs, cell_write_locs)

                    if read_conflicting or write_conflicting:
                        if cell_id not in newly_stale:
                            newly_stale.append(cell_id)
                        fwd_marked += 1
                        stale_var_names: set = set()
                        for wloc in read_conflicting:
                            self._notebook_state.add_reason(
                                cell_id,
                                Reason(ReasonType.FORWARD_STALE, loc=wloc.display_name(), cell_id=deleted_id)
                            )
                            stale_var_names.add(wloc.var_name())
                        for wloc in write_conflicting:
                            if wloc.var_name() not in stale_var_names:
                                self._notebook_state.add_reason(
                                    cell_id,
                                    Reason(ReasonType.WRITE_OVERLAP, loc=wloc.display_name(), cell_id=deleted_id)
                                )
                        alpha_deleted = self._cell_id_to_alpha(deleted_id)
                        alpha_other = self._cell_id_to_alpha(cell_id)
                        all_locs = sorted({w.display_name() for w in read_conflicting | write_conflicting})
                        warning = (
                            f"Cell @{alpha_other} marked stale: "
                            f"deleted cell @{alpha_deleted} wrote {all_locs}"
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
                # BackwardStale iterates over variable names from the typed WriteLocs
                seen_vars: Set[str] = set()
                for wloc in deleted_wlocs:
                    var_name = wloc.var_name()
                    if var_name in seen_vars:
                        continue
                    seen_vars.add(var_name)
                    # Find last writer of var_name among cells before i
                    last_j = None
                    for cell_id in old_order[:my_position]:
                        if cell_id in deleted_set:
                            continue
                        cell_writes = self._notebook_state.writes.get(cell_id, frozenset())
                        if any(w.var_name() == var_name for w in cell_writes):
                            last_j = cell_id  # Keep scanning; last one wins

                    if last_j is not None and last_j in originally_clean:
                        if last_j not in newly_stale:
                            newly_stale.append(last_j)
                        bwd_marked += 1
                        self._notebook_state.add_reason(
                            last_j,
                            Reason(ReasonType.BACKWARD_STALE, loc=var_name, cell_id=deleted_id)
                        )
                        alpha_deleted = self._cell_id_to_alpha(deleted_id)
                        alpha_last = self._cell_id_to_alpha(last_j)
                        warning = (
                            f"Cell @{alpha_last} marked stale (backward): "
                            f"was last writer of '{var_name}' before deleted cell @{alpha_deleted}"
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

                # Use typed ReadLocSet/WriteLocSet from notebook_state
                cell_rlocs = self._notebook_state.reads.get(cell_id, frozenset())
                cell_wlocs = self._notebook_state.writes.get(cell_id, frozenset())
                if not cell_rlocs and not cell_wlocs:
                    continue  # No execution record, nothing to check

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
                    other_rlocs = self._notebook_state.reads.get(other_id, frozenset())
                    other_wlocs = self._notebook_state.writes.get(other_id, frozenset())
                    if not other_rlocs and not other_wlocs:
                        continue

                    # Determine direction of crossing for this specific pair
                    other_old_pos = old_positions[other_id]
                    was_after = other_old_pos > old_pos

                    if was_after:
                        # other_id was after cell_id, now before: cell_id moved forward past other_id
                        # (Ex1) Crossed cells that read moved cell's writes → stale
                        # cell_wlocs ▷ other_rlocs
                        overlap1 = wlocs_conflict_rlocs(cell_wlocs, other_rlocs)
                        if overlap1 and self._notebook_state.is_clean(other_id):
                            newly_stale.append(other_id)
                            cells_marked += 1
                            self._notebook_state.add_reason(
                                other_id, Reason(ReasonType.ORDER_CHANGED)
                            )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            locs = sorted({w.display_name() for w in overlap1})
                            warning = (
                                f"Cell @{alpha_other} marked stale: "
                                f"cell @{alpha_moved} moved forward past it, "
                                f"lost dependency on {locs}"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                        # (Ex2) Moved cell reads from crossed cells' writes → stale
                        # other_wlocs ▷ cell_rlocs
                        overlap2 = wlocs_conflict_rlocs(other_wlocs, cell_rlocs)
                        if overlap2 and self._notebook_state.is_clean(cell_id):
                            newly_stale.append(cell_id)
                            cells_marked += 1
                            self._notebook_state.add_reason(
                                cell_id, Reason(ReasonType.ORDER_CHANGED)
                            )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            locs = sorted({w.display_name() for w in overlap2})
                            warning = (
                                f"Cell @{alpha_moved} marked stale: "
                                f"moved forward past @{alpha_other}, "
                                f"now reads {locs} from it"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                    else:
                        # other_id was before cell_id, now after: cell_id moved backward past other_id
                        # (Ex3) Moved cell reads from crossed cells' writes → stale
                        # other_wlocs ▷ cell_rlocs
                        overlap3 = wlocs_conflict_rlocs(other_wlocs, cell_rlocs)
                        if overlap3 and self._notebook_state.is_clean(cell_id):
                            newly_stale.append(cell_id)
                            cells_marked += 1
                            for wloc in overlap3:
                                self._notebook_state.add_reason(
                                    cell_id,
                                    Reason(ReasonType.NO_READ_BEFORE_WRITE, loc=wloc.display_name(), cell_id=other_id)
                                )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            locs = sorted({w.display_name() for w in overlap3})
                            warning = (
                                f"Cell @{alpha_moved} marked stale: "
                                f"moved backward before @{alpha_other}, "
                                f"forward contamination on {locs}"
                            )
                            warnings.append(warning)
                            log(f"[MOVE] {warning}")

                        # (Ex4) Crossed cells that read moved cell's writes → stale
                        # cell_wlocs ▷ other_rlocs
                        overlap4 = wlocs_conflict_rlocs(cell_wlocs, other_rlocs)
                        if overlap4 and self._notebook_state.is_clean(other_id):
                            newly_stale.append(other_id)
                            cells_marked += 1
                            for wloc in overlap4:
                                self._notebook_state.add_reason(
                                    other_id,
                                    Reason(ReasonType.FORWARD_STALE, loc=wloc.display_name(), cell_id=cell_id)
                                )
                            alpha_moved = self._cell_id_to_alpha(cell_id)
                            alpha_other = self._cell_id_to_alpha(other_id)
                            locs = sorted({w.display_name() for w in overlap4})
                            warning = (
                                f"Cell @{alpha_other} marked stale: "
                                f"cell @{alpha_moved} moved backward before it, "
                                f"gains input from {locs}"
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
            row_mutations=tracking.row_mutations,
            index_mutations=tracking.index_mutations,
            dtype_changes=tracking.dtype_changes,
            column_deletions=tracking.column_deletions,
            file_reads_before_writes=tracking.file_reads_before_writes,
            file_writes=tracking.file_writes,
        )

        # ================================================================
        # STEP 1: Compute r (reads) and w (writes) from tracking
        # Ref: FORMAL_DEVELOPMENT.md §3.1, line 169
        # ================================================================
        W_i_old = self._writes_var_names(cell_id)  # Old W_i as variable names (Set[str])

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
                stale_cells=[],
                changed_variables=[],
                column_changed={},
                structural_warnings=list(current_diff.warnings) if current_diff.warnings else [],
                read_locs=readlocset_to_list(tracking_to_readlocset(tracking, namespace, self._stable_map)),
                write_locs=writelocset_to_list(tracking_to_writelocset(tracking, namespace, self._stable_map)),
                changed_locs=writelocset_to_list(changes_to_write_locs(typed_changes, namespace, self._stable_map)),
                errors=[ReproducibilityError(
                    error_type=ErrorType.UNRECOVERABLE_MUTATION,
                    cell_id=cell_id,
                    locations=truncated_vars,
                    message=format_truncation_error(mutating_alpha, truncated_vars),
                    detail={"truncation_details": formatted_diff} if formatted_diff else None,
                )],
            )

        changed_vars = list(current_diff.differences.keys()) if current_diff.differences else []
        column_changed = _extract_column_changes(current_diff, tracking)

        # Separate value-level changes from column-only changes for recoverability classification.
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

        # Column-level: recoverable iff variable was rebound OR column is in tracking.column_writes.
        # If the variable was rebound (e.g., df = df.merge(...)), ALL column changes are
        # recoverable — re-executing recreates the entire DataFrame including all columns.
        # Column-level tracking only matters for in-place column mutations (df['col'] = val).
        tracked_col_writes = tracking.column_writes or {}
        recoverable_column_changed: Dict[str, List[str]] = {}
        unrecoverable_column_changed: Dict[str, List[str]] = {}
        for var, cols in column_changed.items():
            if var in current_writes_set:
                # Variable rebound — all columns recoverable
                recoverable_column_changed[var] = cols
            else:
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
                execution_seq=self.seq_counter,
                structural_reads_values=structural_read_values,
                typed_changes=typed_changes,
                namespace=namespace,
                stable_map=self._stable_map,
            )
            return ReproducibilityResult(
                stale_cells=self._notebook_state.get_stale_cells(),
                changed_variables=changed_vars,
                column_changed=column_changed,
                structural_warnings=structural_warnings,
                staleness_reasons=self._notebook_state.get_all_reasons(),
                read_locs=readlocset_to_list(tracking_to_readlocset(tracking, namespace, self._stable_map)),
                write_locs=writelocset_to_list(tracking_to_writelocset(tracking, namespace, self._stable_map)),
                changed_locs=writelocset_to_list(changes_to_write_locs(typed_changes, namespace, self._stable_map)),
            )

        # ================================================================
        # STEP 2: Check validity predicates BEFORE updating state
        # Ref: FORMAL_DEVELOPMENT.md §3.2, lines 176-179
        # ================================================================
        errors: List[ReproducibilityError] = []

        # NoReadAndWrite(R', W', i) ≝ Rᵢ ∩ Wᵢ = ∅
        # Ref: FORMAL_DEVELOPMENT.md §3.2, line 176
        # (Cell reads and writes same location - potential issue for reproducibility)
        no_read_and_write_error = self._check_no_read_and_write(cell_id, tracking, namespace)
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
        backward_error = None
        if typed_changes:
            with timer(key="check:NoWriteAfterRead", message=f"[Inst-Run] NoWriteAfterRead check for {cell_id}"):
                backward_error = self._check_backward_mutation_new(
                    cell_id, my_position, typed_changes, current_diff, column_changed, namespace
                )
        if backward_error:
            errors.append(backward_error)
        log(f"[Inst-Run] {cell_id}: NoWriteAfterRead={'fail' if backward_error else 'pass'}")

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
            forward_error = self._check_forward_contamination(cell_id, my_position, tracking, namespace)
        if forward_error:
            errors.append(forward_error)
        log(f"[Inst-Run] {cell_id}: NoReadBeforeWrite={'fail' if forward_error else 'pass'}")

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
        self._notebook_state.record_execution(
            cell_id,
            tracking=tracking,
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
        # are ACCEPTED and the cell stays CLEAN.
        # ================================================================
        # Collect staleness reasons for this cell
        # Only add predicate violation reasons if NOT continuing (i.e., they will be rejected)
        if not continue_on_violation:
            if backward_error is not None:
                # NoWriteAfterRead failed - cell reads values it then modifies, breaking reproducibility
                for loc in backward_error.locations:
                    self._notebook_state.add_reason(
                        cell_id, Reason(ReasonType.NO_WRITE_AFTER_READ, loc=loc, cell_id=backward_error.causer_cell)
                    )
            if unrecoverable_error is not None:
                # UnrecoverableMutation failed - cell mutated state without rebinding
                for loc in unrecoverable_error.locations:
                    self._notebook_state.add_reason(
                        cell_id, Reason(ReasonType.UNRECOVERABLE_MUTATION, loc=loc)
                    )

        # Forward contamination ALWAYS marks cell stale (even when continue_on_violation=True)
        # because re-running top-to-bottom would produce different results.
        if forward_error is not None:
            for loc in forward_error.locations:
                self._notebook_state.add_reason(
                    cell_id, Reason(ReasonType.NO_READ_BEFORE_WRITE, loc=loc, cell_id=forward_error.causer_cell)
                )

        # Set cell status
        # When continue_on_violation=True, only NoWriteAfterRead (backward_error) can be accepted.
        # Other errors (forward contamination, read-and-write, undefined vars) always cause staleness
        # because they affect THIS cell's reproducibility.
        has_any_errors = len(errors) > 0

        # Separate errors into backward-only (can be accepted) and forward/other (always stale)
        _acceptable_error_types = {ErrorType.NO_WRITE_AFTER_READ, ErrorType.UNRECOVERABLE_MUTATION}
        backward_only_errors = [e for e in errors if e.error_type in _acceptable_error_types]
        other_errors = [e for e in errors if e.error_type not in _acceptable_error_types]

        # Backward-only errors can be accepted; other errors always cause staleness
        has_staleness_causing_errors = len(other_errors) > 0 or (len(backward_only_errors) > 0 and not continue_on_violation)

        if not has_staleness_causing_errors:
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

        # W_i_current: current writes at variable-name granularity.
        # Must include DataFrame names from tracking.column_writes to be
        # consistent with how W_i_old is computed (via writelocset_var_names
        # on the stored WriteLocSet, which extracts 'df' from Col locs).
        # tracking.column_writes records column mutations (df['col'] = ...)
        # even when the diff detects no change (same values re-written).
        # Without this, 'df' appears in W_i_old but not W_i_current, causing
        # a spurious "removed write" that marks the df creator stale.
        W_i_current = (
            (tracking.writes or set())
            | set((tracking.column_writes or {}).keys())
        )

        if not will_be_rejected:
            with timer(key="check:ForwardStale", message=f"[Inst-Run] ForwardStale computation for {cell_id}"):
                stale, staleness_warnings = self._compute_forward_staleness(
                    namespace, W_i_old, W_i_current, W_i_vars, recoverable_column_changed, cell_id, my_position,
                    changed_file_paths=_changed_file_paths,
                    typed_changes=typed_changes,
                )
            log(f"[Inst-Run] {cell_id}: ForwardStale marked {len(stale)} cells")
            structural_warnings.extend(staleness_warnings)

            # BackwardStale: mark cells j < i as stale if W_i ∩ R_j ≠ ∅
            # This handles the case where a later cell writes to a variable
            # that an earlier (clean) cell had read.
            # Also handles removed writes: if cell i used to write y but no longer
            # does, the last writer of y before i should be marked stale (its
            # value is now "exposed" to downstream cells).
            with timer(key="check:BackwardStale", message=f"[Inst-Run] BackwardStale computation for {cell_id}"):
                backward_stale = self._compute_backward_staleness(
                    namespace, W_i_vars, recoverable_column_changed, cell_id, my_position,
                    old_writes=W_i_old, current_writes=W_i_current,
                )
            if backward_stale:
                log(f"[Inst-Run] {cell_id}: BackwardStale marked {len(backward_stale)} cells")
                stale.extend(backward_stale)
        else:
            log(f"[Inst-Run] {cell_id}: Skipping staleness propagation (cell will be rejected)")

        # Defer checkpoint deletion until next cell executes.
        # This allows checkpoint size queries after cell execution completes.
        # (we've already computed W_i from the diff, so checkpoint is no longer needed
        # for reproducibility checks, but we keep it for metrics collection)
        self._pending_checkpoint_deletion = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"

        # ================================================================
        # Return result
        # ================================================================
        staleness_reasons = self._notebook_state.get_all_reasons()

        return ReproducibilityResult(
            stale_cells=stale,
            changed_variables=changed_vars,
            column_changed=column_changed,
            structural_warnings=structural_warnings,
            staleness_reasons=staleness_reasons,
            read_locs=readlocset_to_list(tracking_to_readlocset(tracking, namespace, self._stable_map)),
            write_locs=writelocset_to_list(tracking_to_writelocset(tracking, namespace, self._stable_map)),
            changed_locs=writelocset_to_list(changes_to_write_locs(typed_changes, namespace, self._stable_map)),
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

    def _writes_var_names(self, cell_id: str) -> Set[str]:
        """Extract variable names from a cell's WriteLocSet."""
        from flowbook.kernel.locations import writelocset_var_names
        writes = self._notebook_state.writes.get(cell_id, frozenset())
        return writelocset_var_names(writes)

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

            # Safety-net logging: check if aliases are already unified by StableIdMap
            alias_only = keys_to_include - accessed_vars
            if alias_only and _logger.isEnabledFor(logging.DEBUG):
                already_unified = set()
                for alias in alias_only:
                    alias_obj = namespace.get(alias)
                    if alias_obj is None:
                        continue
                    alias_sid = self._stable_map.get_stable(alias_obj)
                    for orig in accessed_vars:
                        orig_obj = namespace.get(orig)
                        if orig_obj is not None and self._stable_map.get_stable(orig_obj) == alias_sid:
                            already_unified.add(alias)
                            break
                not_unified = alias_only - already_unified
                if already_unified:
                    _logger.debug(
                        "Alias expansion found %d aliases already unified by StableIdMap: %s",
                        len(already_unified), already_unified,
                    )
                if not_unified:
                    _logger.debug(
                        "Alias expansion found %d aliases NOT unified by StableIdMap: %s",
                        len(not_unified), not_unified,
                    )

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
                structural_mode=_STRUCTURAL_ENFORCE,
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
                structural_mode=_STRUCTURAL_ENFORCE,
            )

        # Convert to typed changes
        typed_changes = detect_changes(current_diff) if current_diff.differences else []

        return current_diff, typed_changes

    def _check_forward_contamination(
        self,
        cell_id: str,
        my_position: int,
        tracking: TrackingData,
        namespace: Optional[dict] = None,
    ) -> Optional[ReproducibilityError]:
        """
        Check NoReadBeforeWrite predicate.

        Formal ref: NoReadBeforeWrite(R, W, i) ≝ Rᵢ ∩ W_{i+1..n} = ∅
        FORMAL_DEVELOPMENT.md §3.2, line 178

        Uses LocSet ▷ operator for column-aware conflict detection.
        Prefers diff-based WriteLocSet (from typed_changes) when available
        for column-level precision, falls back to tracking-based writes.

        Uses df.attrs provenance to inject structural WriteLocs (Rows,
        Attr(index), Attr(dtypes)) that checkpoint diffs
        miss on re-execution, ensuring structural conflicts are detected.
        """
        R_i = tracking_to_readlocset(tracking, namespace, self._stable_map)
        if not R_i:
            return None

        for later_cell_id in self._cell_order[my_position + 1:]:
            if not self._notebook_state.has_record(later_cell_id):
                continue

            # Prefer diff-based writes for column-level precision
            later_changes = self._notebook_state.get_typed_changes(later_cell_id)
            if later_changes:
                W_later = changes_to_write_locs(later_changes, namespace, self._stable_map)
            else:
                W_later = self._notebook_state.writes.get(later_cell_id, frozenset())

            if not W_later:
                continue

            conflicting = wlocs_conflict_rlocs(W_later, R_i)
            if conflicting:
                conflicts = sorted(w.display_name() for w in conflicting)
                reading_alpha = self._cell_id_to_alpha(cell_id)
                writing_alpha = self._cell_id_to_alpha(later_cell_id)
                message = format_forward_dependency_message(reading_alpha, writing_alpha, conflicts)

                return ReproducibilityError(
                    error_type=ErrorType.NO_READ_BEFORE_WRITE,
                    cell_id=cell_id,
                    locations=conflicts,
                    message=message,
                    causer_cell=later_cell_id,
                )

        return None

    def _check_backward_mutation_new(
        self,
        cell_id: str,
        my_position: int,
        typed_changes: List,
        current_diff: MemoryCheckpointDiffResult,
        modified_columns: Dict[str, List[str]],
        namespace: Optional[dict] = None,
    ) -> Optional[ReproducibilityError]:
        """
        Check NoWriteAfterRead predicate.

        Formal ref: NoWriteAfterRead(R, W, i) ≝ Wᵢ ∩ R_{1..i-1} = ∅
        FORMAL_DEVELOPMENT.md §3.2, line 179

        Only checks against CLEAN cells per [Inst-Run] semantics.
        """
        W_i_diff = changes_to_write_locs(typed_changes, namespace, self._stable_map) if typed_changes else frozenset()

        if not W_i_diff:
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
            if not self._notebook_state.has_record(prior_cell_id):
                continue
            if not self._notebook_state.is_clean(prior_cell_id):
                continue

            R_prior = self._notebook_state.reads.get(prior_cell_id, frozenset())
            if not R_prior:
                continue

            conflicting = wlocs_conflict_rlocs(W_i_diff, R_prior)
            if not conflicting:
                continue

            conflicts = sorted(w.display_name() for w in conflicting)

            mutating_alpha = self._cell_id_to_alpha(cell_id)
            affected_alpha = self._cell_id_to_alpha(prior_cell_id)
            prior_structural_values = self._notebook_state.get_structural_reads_values(prior_cell_id)
            changes = _extract_change_descriptions(current_diff, modified_columns)
            message = format_structural_violation(
                mutating_alpha, affected_alpha, conflicts, prior_structural_values, changes
            )

            detail = {}
            if prior_structural_values:
                detail["structural_reads_detail"] = prior_structural_values
            if changes:
                detail["changes_detail"] = changes

            return ReproducibilityError(
                error_type=ErrorType.NO_WRITE_AFTER_READ,
                cell_id=cell_id,
                locations=conflicts,
                message=message,
                causer_cell=prior_cell_id,
                detail=detail if detail else None,
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
        namespace: Optional[dict] = None,
    ) -> Optional[ReproducibilityError]:
        """
        Check NoReadAndWrite predicate: Wᵢ ▷ Rᵢ = ∅

        Cell should not both read and write the same location.
        Uses the ▷ conflict relation for proper Var(x) = binding-only semantics:
        - Col(df, y) ▷ Var(df) = false → column write + binding read is OK
        - Col(df, x) ▷ Col(df, x) = true → same column read+write is a violation
        - Var(x) ▷ Var(x) = true → variable reassignment + read is a violation

        Formal ref: main.tex §3.2, FORMAL_DEVELOPMENT.md §3.2, line 176
        """
        R_i = tracking_to_readlocset(tracking, namespace, self._stable_map)
        W_i = self._tracking_to_noreadandwrite_wlocs(tracking, namespace)

        conflicting = wlocs_conflict_rlocs(W_i, R_i)
        if conflicting:
            locations = sorted(w.display_name() for w in conflicting)
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

    def _tracking_to_noreadandwrite_wlocs(
        self,
        tracking: TrackingData,
        namespace: Optional[dict] = None,
    ) -> WriteLocSet:
        """Build WriteLocSet for NoReadAndWrite check.

        For column assignments like df["y"] = ..., the tracking reports
        writes={"df"} but the actual write is Col(df, y), not Var(df).
        We use column_writes to distinguish: if a variable has column_writes,
        emit Col locs; otherwise emit Var.
        """
        from flowbook.kernel.loc_ids import get_qualifier
        locs: set = set()
        col_writes = tracking.column_writes or {}
        var_writes = tracking.writes or set()

        for var in var_writes:
            if var in col_writes:
                # Has column detail → emit Col for each column written
                q = get_qualifier(var, namespace, self._stable_map)
                for col in col_writes[var]:
                    locs.add(WriteLoc.col(q, col))
            else:
                # No column detail → Var (whole variable reassignment)
                locs.add(WriteLoc.var(var))

        # Also include column writes for vars not in tracking.writes
        # (column writes via in-place mutation without rebinding)
        for var, cols in col_writes.items():
            if var not in var_writes:
                q = get_qualifier(var, namespace, self._stable_map)
                for col in cols:
                    locs.add(WriteLoc.col(q, col))

        return frozenset(locs)

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
        typed_changes: Optional[List] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Compute ForwardStale for all cells j > i.

        Formal ref: ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
        FORMAL_DEVELOPMENT.md §3.3, line 187
        FORMAL_DEVELOPMENT.md §10 (Staleness Computation)
        """
        return self._compute_forward_staleness_syntactic(
            old_writes, current_writes, changed_vars, column_changed, just_executed, my_position,
            changed_file_paths, typed_changes=typed_changes, namespace=current_namespace,
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
        typed_changes: Optional[List] = None,
        namespace: Optional[dict] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Syntactic ForwardStale: (Wᵢ ∪ W'ᵢ ∪ ΔV) ▷ (Rⱼ ∪ Wⱼ) ≠ ∅ for j > i.

        Cell j becomes stale if cell i's writes conflict with what cell j reads
        or writes, using the ▷ relation for column-level precision.

        Uses typed Change objects (when available) for precise WriteLoc types:
        Col conflicts with same-column reads and Cols reads.
        Rows conflicts with all column reads and Rows reads.

        Formal ref: FORMAL_DEVELOPMENT.md §3.3, §10.1
        """
        all_warnings: List[str] = []

        # Wᵢ ∪ W'ᵢ ∪ ΔV: all locations cell i has written (old, new, or diff-detected)
        W_i_union = old_writes | current_writes | changed_vars

        # Build WriteLocSet for ▷-based staleness checks.
        # Base: use tracking-based column info (column_changed) for column-level precision.
        # Augment: add Col from typed changes (diff-based) when available,
        # because these affect structural attributes (shape, columns) that Col does not.
        change_wlocs = _changes_to_writelocset(W_i_union, column_changed)

        cells_below = self._cell_order[my_position + 1:]
        for cell_id in cells_below:
            cell_tracking = self._notebook_state.get_tracking(cell_id)
            if cell_tracking is None:
                continue

            if not self._notebook_state.is_clean(cell_id):
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

            # Use ▷ operator for column-aware overlap check.
            # change_wlocs captures what actually changed at the right granularity:
            # - Var(x) for whole-variable changes
            # - Col(df, c) for column-level changes
            # Read sets always include Var(x) alongside Col/Attr reads, so
            # variable rebinding is caught by Var(x) ▷ Var(x) = true.
            cell_read_locs = self._notebook_state.reads.get(cell_id, frozenset())
            conflicting_wlocs = wlocs_conflict_rlocs(change_wlocs, cell_read_locs)

            # Write-write overlap: W'_i ▷▷ W_j — direct write-write conflict check.
            # Uses stored WriteLocSet (includes diff-derived Col/Rows/Attr).
            cell_write_locs = self._notebook_state.writes.get(cell_id, frozenset())
            write_conflicting = wlocs_conflict_wlocs(change_wlocs, cell_write_locs)

            if conflicting_wlocs or write_conflicting:
                # Build staleness reasons from conflicting WriteLocs
                stale_var_names: set = set()
                for wloc in conflicting_wlocs:
                    self._notebook_state.add_reason(
                        cell_id,
                        Reason(ReasonType.FORWARD_STALE, loc=wloc.display_name(), cell_id=just_executed)
                    )
                    stale_var_names.add(wloc.var_name())

                # Then handle write-only overlaps (WRITE_OVERLAP - no convergence)
                for wloc in write_conflicting:
                    if wloc.var_name() not in stale_var_names:
                        self._notebook_state.add_reason(
                            cell_id,
                            Reason(ReasonType.WRITE_OVERLAP, loc=wloc.display_name(), cell_id=just_executed)
                        )

        return self._notebook_state.get_stale_cells(), all_warnings

    def _compute_backward_staleness(
        self,
        current_namespace: dict,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
        old_writes: Optional[Set[str]] = None,
        current_writes: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Compute BackwardStale for all cells j < i.

        When cell i writes to a variable that earlier cell j read, j becomes stale.
        Also handles removed writes: when cell i used to write y but no longer does,
        the last writer of y before i (j) should be marked stale — its value is
        now "exposed" to downstream cells.

        Formal ref: FORMAL_DEVELOPMENT.md §10, §3.3

        Returns:
            List of cell IDs that were marked stale
        """
        return self._compute_backward_staleness_syntactic(
            changed_vars, column_changed, just_executed, my_position,
            old_writes=old_writes, current_writes=current_writes,
        )

    def _compute_backward_staleness_syntactic(
        self,
        changed_vars: Set[str],
        column_changed: Dict[str, List[str]],
        just_executed: str,
        my_position: int,
        old_writes: Optional[Set[str]] = None,
        current_writes: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Syntactic BackwardStale: W_i ∩ R_j ≠ ∅ for j < i.

        Uses pure set intersection on R/W sets. Marks cells before i as stale
        if they read variables that i wrote to. Also checks column-level overlap
        for DataFrame column changes.

        Also handles removed writes: BackwardStale(W, W', i, j) ≝
        j < i ∧ j = LastWriter(W, i, y) for some y ∈ Wᵢ \\ W'ᵢ.
        When cell i drops a write to y, the last writer of y before i
        should be marked stale — its value is now "exposed".

        Formal ref: FORMAL_DEVELOPMENT.md §10.1, §3.3
        """
        newly_stale: List[str] = []

        # Convert changes to WriteLocSet for ▷-based conflict detection
        change_wlocs = _changes_to_writelocset(changed_vars, column_changed)

        for prior_cell_id in self._cell_order[:my_position]:
            if not self._notebook_state.is_clean(prior_cell_id):
                continue  # Already stale

            if not self._notebook_state.has_record(prior_cell_id):
                continue  # Never executed

            # Use ▷ operator: check if changes conflict with prior cell's reads.
            # Read sets always include Var(x) alongside Col/Attr reads, so
            # variable rebinding is caught by Var(x) ▷ Var(x) = true.
            prior_read_locs = self._notebook_state.reads.get(prior_cell_id, frozenset())
            conflicting_wlocs = wlocs_conflict_rlocs(change_wlocs, prior_read_locs)

            if conflicting_wlocs:
                # Build staleness reasons from conflicting WriteLocs
                for wloc in conflicting_wlocs:
                    self._notebook_state.add_reason(
                        prior_cell_id,
                        Reason(ReasonType.FORWARD_STALE, loc=wloc.display_name(), cell_id=just_executed)
                    )
                newly_stale.append(prior_cell_id)

        # Removed writes backward staleness:
        # BackwardStale(W, W', i, j) ≝ j < i ∧ j = LastWriter(W, i, y) for y in W_old - W_new
        # When cell i drops a write to y, the previous writer of y becomes stale
        # because its value is now "exposed" to downstream cells.
        if old_writes is not None and current_writes is not None:
            removed_writes = old_writes - current_writes
            if removed_writes:
                for y in removed_writes:
                    last_j = self._notebook_state.last_writer_for(y, just_executed)
                    if last_j is not None and self._notebook_state.is_clean(last_j):
                        self._notebook_state.add_reason(
                            last_j,
                            Reason(ReasonType.BACKWARD_STALE, loc=y, cell_id=just_executed)
                        )
                        if last_j not in newly_stale:
                            newly_stale.append(last_j)

        return newly_stale

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
                structural_mode=_STRUCTURAL_ENFORCE,
            )

            if diff_result.differences:
                # Track reasons: FORWARD_STALE for each changed variable
                for var in diff_result.differences.keys():
                    self._notebook_state.add_reason(
                        cell_id, Reason(ReasonType.FORWARD_STALE, loc=var)
                    )

        return self._notebook_state.get_stale_cells()

    def mark_cell_edited(self, cell_id: str) -> List[str]:
        """[Inst-Edit] Mark edited cell stale.

        Per the formal semantics: T' = T[i := stale], R and W unchanged.
        R and W are preserved so that rerunning the cell can compute
        removed writes (W_i \\ W'_i) for BackwardStale propagation.

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
                structural_mode=_STRUCTURAL_ENFORCE,
            )

            # Conflict resolution (simulated for timing measurement)
            my_position = self._get_position(cell_id)
            if my_position >= 0:
                R_i = tracking_to_readlocset(tracking, namespace, self._stable_map)

                # Simulate forward contamination check
                for later_cell_id in self._cell_order[my_position + 1:]:
                    if not self._notebook_state.has_record(later_cell_id):
                        continue
                    W_later = self._notebook_state.writes.get(later_cell_id, frozenset())
                    if W_later and R_i:
                        wlocs_conflict_rlocs(W_later, R_i)

                # Simulate backward mutation check
                for prior_cell_id in self._cell_order[:my_position]:
                    if not self._notebook_state.has_record(prior_cell_id):
                        continue
                    if not self._notebook_state.is_clean(prior_cell_id):
                        continue
                    R_prior = self._notebook_state.reads.get(prior_cell_id, frozenset())
                    W_i = self._notebook_state.writes.get(cell_id, frozenset())
                    if R_prior and W_i:
                        wlocs_conflict_rlocs(W_i, R_prior)

        result["check_ms"] = check_timer.duration()
        result["total_overhead_ms"] = result["checkpoint_ms"] + result["check_ms"]

        # Clean up the temporary checkpoint to avoid memory accumulation
        self.checkpoints.delete(checkpoint_name)

        return result

    def reset(self) -> None:
        """Clear all state. Called on kernel restart."""
        self.seq_counter = 0
        self._cell_order = []
        self._notebook_state.clear()  # Clear status, R, W, tracking_data
        self._pending_checkpoint_deletion = None
        self._pending_snapshot = None
        self._stable_map.clear()

    def rollback_last_check(self) -> None:
        """
        Rollback state changes from the most recent check() call.

        Called by kernel when execution is rejected and namespace is rolled back.
        This ensures the enforcer's analysis state matches the rolled-back namespace.

        The rollback restores:
        - Per-cell state (reads, writes, status, tracking_data, etc.)
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

    This distinction matters for recoverability classification: value-level
    changes affect the entire variable, while column-only changes are localized.

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

    # Add column assignment hints for modified columns
    for var in variables:
        if "." in var:
            df_name, col_name = var.rsplit(".", 1)
            lines.append(f'  Use {df_name}["{col_name}"] = ... for full-column assignment')

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
