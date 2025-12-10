"""Tests for SDC Enforcer."""

import pytest

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData

from .sdc_enforcer import SDCEnforcer


def make_tracking(reads: set = None, writes: set = None) -> TrackingData:
    """Helper to create TrackingData."""
    return TrackingData(
        reads_before_writes=reads or set(),
        writes=writes or set(),
        column_reads_before_writes={},
        column_writes={},
    )


class TestSDCEnforcer:

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"_pre_{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_no_violation_forward_dependency(self):
        """Cell B reads what cell A writes - valid."""
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result_a.violation is None

        # Cell B reads x - valid (forward dependency)
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert result_b.violation is None

    def test_violation_backward_mutation(self):
        """Cell B modifies what cell A reads - violation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B (after A) modifies x - violation!
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.violation is not None
        assert result_b.violation.mutating_cell == "b"
        assert result_b.violation.affected_cell == "a"
        assert "x" in result_b.violation.variables

    def test_staleness_computation(self):
        """Re-running cell A makes cell B stale if B reads A's output."""
        # First run: A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale
        assert "b" in result.stale_cells

    def test_no_staleness_if_value_unchanged(self):
        """Semantic check: no staleness if value didn't actually change."""
        # A writes x=1
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with same value x=1
        # Note: pre-checkpoint for A now reflects current state
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 1, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should NOT be stale (x didn't change)
        assert "b" not in result.stale_cells

    def test_cell_order_update_affects_violation_check(self):
        """Cell order can be updated, affecting position-based checks."""
        # Initially [a, b, c, d]
        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - A is before B in order, so NOT a violation
        # (B can depend on A, that's forward dependency)
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.violation is None

        # Now reorder: [b, a, c, d] - B is now before A
        self.sdc.set_cell_order(["b", "a", "c", "d"])

        # A modifies x - now A is AFTER B, so this IS a violation
        # (A is mutating what B, an earlier cell, reads)
        self._save_pre_checkpoint("a", {"x": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.violation is not None
        assert result.violation.affected_cell == "b"

    def test_cell_deletion_prunes_records(self):
        """Deleted cells are removed from tracking."""
        # Execute cells a and b
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert "a" in self.sdc.records
        assert "b" in self.sdc.records

        # Remove cell b from order
        self.sdc.set_cell_order(["a", "c", "d"])

        # b should be pruned
        assert "a" in self.sdc.records
        assert "b" not in self.sdc.records

    def test_reset_clears_all_state(self):
        """Reset clears all tracking state."""
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert len(self.sdc.records) == 1
        assert self.sdc.seq_counter == 1

        self.sdc.reset()

        assert len(self.sdc.records) == 0
        assert self.sdc.seq_counter == 0
        assert self.sdc.cell_order == []

    def test_stale_cells_in_document_order(self):
        """Stale cells should be returned in document order, not execution order."""
        # Execute in order: a, d, b, c (but document order is a, b, c, d)
        # All read x

        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("d", {"x": 1})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "w": 4})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved["_pre_d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads={"x"}, writes={"w"}),
        )

        self._save_pre_checkpoint("b", {"x": 1, "w": 4})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "w": 4, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        self._save_pre_checkpoint("c", {"x": 1, "w": 4, "y": 2})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "w": 4, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved["_pre_c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Re-run A with different x - should make b, c, d stale (they all read x)
        self._save_pre_checkpoint("a", {"x": 1, "w": 4, "y": 2, "z": 3})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "w": 4, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Should be in document order [b, c, d], not execution order [d, b, c]
        assert result.stale_cells == ["b", "c", "d"]

    def test_no_violation_when_cell_not_in_order(self):
        """If cell_id is not in cell_order, no violation check happens."""
        # Set cell order that doesn't include 'x'
        self.sdc.set_cell_order(["a", "b", "c"])

        # Cell 'a' reads variable 'var'
        self._save_pre_checkpoint("a", {"var": 1})
        post_a = self._make_post_checkpoint("post_a", {"var": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"var"}, writes=set()),
        )

        # Cell 'x' (not in order) modifies 'var' - should not trigger violation
        self._save_pre_checkpoint("x", {"var": 1})
        post_x = self._make_post_checkpoint("post_x", {"var": 999})
        result = self.sdc.check(
            cell_id="x",
            pre_checkpoint=self.checkpoints.saved["_pre_x"],
            post_checkpoint=post_x,
            tracking=make_tracking(reads=set(), writes={"var"}),
        )

        # No violation because 'x' is not in cell_order
        assert result.violation is None
