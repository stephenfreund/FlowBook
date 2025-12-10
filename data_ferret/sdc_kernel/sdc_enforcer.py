"""
SDC Enforcer - Sequential Dataflow Consistency enforcement.

Implements the three SDC rules:
1. Reproducibility Invariant (structural)
2. Staleness Propagation Rule (computed here)
3. No Backward Mutation Constraint (enforced here)
"""

from typing import Dict, List, Optional, Set

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData
from data_ferret.util.cell_index import index_to_alpha
from data_ferret.util.output import log, timer

from .models import SDCExecutionRecord, SDCResult, SDCViolation


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

    def check(
        self,
        cell_id: str,
        pre_checkpoint: Checkpoint,
        post_checkpoint: Checkpoint,
        tracking: TrackingData,
    ) -> SDCResult:
        """
        Main entry point. Call after cell execution.

        Args:
            cell_id: ID of the cell that just executed
            pre_checkpoint: Snapshot of namespace before execution
            post_checkpoint: Snapshot of namespace after execution
            tracking: TrackingData with reads/writes

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

        # Rule 3: Check backward mutation
        violation = None
        if my_position >= 0:
            violation = self._check_backward_mutation(
                cell_id, my_position, pre_checkpoint, post_checkpoint, tracking
            )

        stale = []
        changed_vars = []

        if not violation:
            # Update our record for this cell BEFORE computing staleness
            # (so this cell is considered "fresh" in the computation)
            self.records[cell_id] = SDCExecutionRecord(
                cell_id=cell_id,
                tracking=tracking,
                execution_seq=self.seq_counter,
            )

            # Compute ABSOLUTE staleness - all cells that are currently stale
            stale = self.compute_all_stale_cells(post_checkpoint)

            # Compute changed variables (what this execution modified)
            with timer(key="sdc_changed_vars", message="[sdc] Computing changed variables"):
                diff_result = Checkpoint.diff(pre_checkpoint, post_checkpoint)
                if diff_result.differences:
                    changed_vars = list(diff_result.differences.keys())

        return SDCResult(
            violation=violation,
            stale_cells=stale,  # ABSOLUTE list of all currently stale cells
            changed_variables=changed_vars,
        )

    def _check_backward_mutation(
        self,
        cell_id: str,
        my_position: int,
        pre_checkpoint: Checkpoint,
        post_checkpoint: Checkpoint,
        tracking: TrackingData,
    ) -> Optional[SDCViolation]:
        """
        Check if current cell causes a backward mutation.

        A backward mutation occurs when a cell modifies a variable that an
        earlier cell (in notebook order) reads. This is Rule 3 of SDC.

        The key insight: we must check what THIS cell actually modified,
        not just whether values are different from when earlier cells ran.
        """
        # First, compute what THIS cell actually modified
        with timer(key="sdc_current_diff", message="[sdc] Computing current cell diff"):
            current_diff = Checkpoint.diff(pre_checkpoint, post_checkpoint)

        if not current_diff.differences:
            # This cell didn't modify anything, no backward mutation possible
            return None

        modified_vars = set(current_diff.differences.keys())
        log(f"[sdc] Cell modified variables: {sorted(modified_vars)}")

        # Check if any earlier cell reads something we modified
        for prior_cell_id in self._cell_order[:my_position]:
            prior_record = self.records.get(prior_cell_id)
            if prior_record is None:
                continue

            # Check intersection of our modifications with prior cell's reads
            prior_reads = set(prior_record.tracking.reads_before_writes)
            overlap = modified_vars & prior_reads

            if overlap:
                mutating_alpha = self._cell_id_to_alpha(cell_id)
                affected_alpha = self._cell_id_to_alpha(prior_cell_id)
                vars_list = sorted(overlap)

                return SDCViolation(
                    mutating_cell=cell_id,
                    affected_cell=prior_cell_id,
                    variables=vars_list,
                    message=(
                        f"Cell {mutating_alpha} modified {vars_list} "
                        f"which cell {affected_alpha} (earlier in notebook) reads. "
                        f"This violates Sequential Dataflow Consistency."
                    ),
                )

        return None

    def compute_all_stale_cells(self, current_checkpoint: Checkpoint) -> List[str]:
        """
        Compute the COMPLETE set of all currently stale cells.

        A cell is stale if the values it read when it last executed
        have changed since then.

        This compares each executed cell's pre-execution checkpoint against
        the current state to determine staleness.

        Args:
            current_checkpoint: The current state of the namespace

        Returns:
            List of cell IDs that are currently stale (in document order)
        """
        stale = set()

        log(f"[sdc] compute_all_stale_cells: Checking {len(self.records)} executed cells")
        log(f"[sdc] Cell order: {[self._cell_id_to_alpha(cid) for cid in self._cell_order]}")

        # Check every cell that has been executed
        for cell_id, record in self.records.items():
            cell_alpha = self._cell_id_to_alpha(cell_id)

            # Get the checkpoint from when this cell last ran
            pre_checkpoint = self.checkpoints.get(f"_pre_{cell_id}")
            if pre_checkpoint is None:
                log(f"[sdc] Cell {cell_alpha}: No pre-checkpoint found, skipping")
                continue

            # Compare what this cell READ THEN vs what those values are NOW
            with timer(key="sdc_stale_check", message=f"[sdc] Checking staleness for {cell_alpha}"):
                diff_result = Checkpoint.diff(
                    pre_checkpoint,
                    current_checkpoint,
                    keys_to_include=record.tracking.reads_before_writes,
                    use_leq=True,
                    column_rbw=record.tracking.column_reads_before_writes,
                )

            if diff_result.differences:
                stale.add(cell_id)
                log(f"[sdc] Cell {cell_alpha}: STALE (changed: {list(diff_result.differences.keys())})")
            else:
                log(f"[sdc] Cell {cell_alpha}: FRESH")

        # Return in document order (for consistency)
        result = [cid for cid in self._cell_order if cid in stale]
        result_alpha = [self._cell_id_to_alpha(cid) for cid in result]
        log(f"[sdc] compute_all_stale_cells: Result = {result_alpha}")
        return result

    def _compute_stale(
        self,
        cell_id: str,
        my_position: int,
        post_checkpoint: Checkpoint,
    ) -> List[str]:
        """
        Compute cells that are now stale because their inputs changed.

        A cell Y (after cell_id in document order) is stale if:
        - Y has been previously executed (has a record)
        - The values Y reads have changed since Y last ran

        This mirrors _check_backward_mutation but looks forward instead of backward.

        Returns cell IDs in document order.
        """
        directly_stale = []

        # Check cells after us in document order
        for later_cell_id in self._cell_order[my_position + 1:]:
            later_record = self.records.get(later_cell_id)
            if later_record is None:
                continue  # Cell not yet executed

            # Get the pre-checkpoint from when this cell last ran
            later_pre = self.checkpoints.get(f"_pre_{later_cell_id}")
            if later_pre is None:
                continue  # No checkpoint available

            # Compare: has what this cell reads changed since it last ran?
            with timer(key="sdc_stale_diff", message="[sdc] Computing staleness diff"):
                diff_result = Checkpoint.diff(
                    later_pre,
                    post_checkpoint,
                    keys_to_include=later_record.tracking.reads_before_writes,
                    use_leq=True,
                    column_rbw=later_record.tracking.column_reads_before_writes,
                )

            if diff_result.differences:
                directly_stale.append(later_cell_id)

        return directly_stale

    def reset(self) -> None:
        """Clear all state. Called on kernel restart."""
        self.records.clear()
        self.seq_counter = 0
        self._cell_order = []
