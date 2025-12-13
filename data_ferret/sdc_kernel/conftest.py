"""
Shared pytest fixtures for SDC kernel tests.
"""

import pytest

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData

from .sdc_enforcer import SDCEnforcer, PRE_CHECKPOINT_PREFIX


def make_tracking(
    reads: set = None,
    writes: set = None,
    column_reads: dict = None,
    column_writes: dict = None,
    structural_reads: dict = None,
) -> TrackingData:
    """
    Helper to create TrackingData with optional column tracking.

    Args:
        reads: Set of variable names read before write
        writes: Set of variable names written
        column_reads: Dict mapping var names to sets of read column names
        column_writes: Dict mapping var names to sets of written column names
        structural_reads: Dict mapping var names to sets of structural attrs read

    Returns:
        TrackingData instance
    """
    return TrackingData(
        reads_before_writes=reads or set(),
        writes=writes or set(),
        column_reads_before_writes=column_reads or {},
        column_writes=column_writes or {},
        structural_reads=structural_reads or {},
    )


class SDCTestHelper:
    """
    Helper class for SDC enforcer tests.

    Provides convenient methods for creating checkpoints and running
    SDC checks in tests.
    """

    def __init__(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)

    def set_cell_order(self, order: list) -> None:
        """Set the cell order for SDC enforcement."""
        self.sdc.set_cell_order(order)

    def save_pre_checkpoint(self, cell_id: str, namespace: dict) -> None:
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create and return a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def get_pre_checkpoint(self, cell_id: str) -> Checkpoint:
        """Get the pre-checkpoint for a cell."""
        return self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"]

    def execute_cell(
        self,
        cell_id: str,
        pre_namespace: dict,
        post_namespace: dict,
        reads: set = None,
        writes: set = None,
        column_reads: dict = None,
        column_writes: dict = None,
        structural_reads: dict = None,
        continue_on_violation: bool = False,
    ):
        """
        Simulate executing a cell with SDC tracking.

        This is a convenience method that:
        1. Saves a pre-checkpoint
        2. Creates a post-checkpoint
        3. Runs the SDC check

        Args:
            cell_id: ID of the cell being executed
            pre_namespace: Namespace state before execution
            post_namespace: Namespace state after execution
            reads: Variables read by the cell
            writes: Variables written by the cell
            column_reads: Dict of var -> set of read columns
            column_writes: Dict of var -> set of written columns
            structural_reads: Dict of var -> set of structural attrs read
            continue_on_violation: Whether to continue after violations

        Returns:
            SDCResult from the check
        """
        self.save_pre_checkpoint(cell_id, pre_namespace)
        post_checkpoint = self.make_post_checkpoint(f"post_{cell_id}", post_namespace)

        tracking = make_tracking(
            reads=reads,
            writes=writes,
            column_reads=column_reads,
            column_writes=column_writes,
            structural_reads=structural_reads,
        )

        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.get_pre_checkpoint(cell_id),
            post_checkpoint=post_checkpoint,
            tracking=tracking,
            continue_on_violation=continue_on_violation,
        )


@pytest.fixture
def sdc_helper():
    """
    Fixture providing an SDCTestHelper instance.

    Usage:
        def test_something(sdc_helper):
            sdc_helper.set_cell_order(["a", "b", "c"])
            result = sdc_helper.execute_cell(
                "a", pre_namespace={}, post_namespace={"x": 1},
                writes={"x"}
            )
            assert result.violation is None
    """
    return SDCTestHelper()


@pytest.fixture
def sdc_helper_with_order():
    """
    Fixture providing an SDCTestHelper with a default cell order ["a", "b", "c", "d"].
    """
    helper = SDCTestHelper()
    helper.set_cell_order(["a", "b", "c", "d"])
    return helper
