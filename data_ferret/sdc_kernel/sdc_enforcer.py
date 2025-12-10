"""
SDC Enforcer - Sequential Dataflow Consistency enforcement.

Implements the three SDC rules:
1. Reproducibility Invariant (structural)
2. Staleness Propagation Rule (computed here)
3. No Backward Mutation Constraint (enforced here)
"""

from typing import Dict, List, Optional, Set

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.diff import Diff
from data_ferret.kernel.models import TrackingData
from data_ferret.kernel.tracking import TrackingDict
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

    @property
    def cell_order(self) -> List[str]:
        return self._cell_order

    def set_cell_order(self, order: List[str]) -> None:
        """Update notebook structure. Called via magic or metadata."""
        self._cell_order = order
        self._prune_deleted_cells()

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
            reads: Variables read by the cell (reads_before_writes)
            writes: Variables written by the cell

        Returns:
            SDCResult with violation info and stale cells
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
                cell_id, my_position, post_checkpoint, tracking
            )

        stale = []
        if not violation:
            # Compute staleness before updating our record
            if my_position >= 0:
                stale = self._compute_stale(cell_id, my_position, post_checkpoint)

            self.records[cell_id] = SDCExecutionRecord(
                cell_id=cell_id,
                tracking=tracking,
                execution_seq=self.seq_counter,
            )

        return SDCResult(
            violation=violation,
            stale_cells=stale,
            changed_variables=[],
        )

    def _do_check(
        self, prior: SDCExecutionRecord, cell_id: str, post_checkpoint: Checkpoint
    ) -> Optional[SDCViolation]:
        """Internal implementation of SDC check."""

        # Compare pre and post states
        with timer(key="sdc_diff", message="[sdc] Computing diff"):
            prior_checkpoint = self.checkpoints.get(f"_pre_{prior.cell_id}")
            diff_result = Checkpoint.diff(
                prior_checkpoint,
                post_checkpoint,
                keys_to_include=prior.tracking.reads_before_writes,
                use_leq=True,
                column_rbw=prior.tracking.column_reads_before_writes,
            )

        if diff_result.differences:
            return SDCViolation(
                mutating_cell=cell_id,
                affected_cell=prior.cell_id,
                variables=list(diff_result.differences.keys()),
                message=(
                    f"Cell '{cell_id}' modified {sorted(diff_result.differences.keys())} "
                    f"which cell '{prior.cell_id}' (earlier in notebook) reads. "
                    f"This violates Sequential Dataflow Consistency."
                ),
            )
        return None

    def _check_backward_mutation(
        self,
        cell_id: str,
        my_position: int,
        post_checkpoint: Checkpoint,
        tracking: TrackingData,
    ) -> Optional[SDCViolation]:
        for prior_cell_id in self._cell_order[:my_position]:
            prior_record = self.records.get(prior_cell_id)
            if prior_record is None:
                continue
            violation = self._do_check(prior_record, cell_id, post_checkpoint)
            if violation:
                return violation
        return None

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
