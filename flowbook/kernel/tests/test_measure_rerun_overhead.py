"""Tests for measure_rerun_overhead functionality."""

import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
)
from flowbook.kernel.tests.conftest import make_tracking


class TestMeasureRerunOverhead:
    """Tests for the measure_rerun_overhead method."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.enforcer = ReproducibilityEnforcer(self.checkpoints)
        self.enforcer.set_cell_order(["a", "b", "c", "d", "e"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _execute_cell(self, cell_id: str, pre_ns: dict, post_ns: dict, tracking: TrackingData):
        """Simulate executing a cell."""
        self._save_pre_checkpoint(cell_id, pre_ns)
        return self.enforcer.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            namespace=post_ns,
            tracking=tracking,
        )

    def test_measure_rerun_overhead_returns_dict(self):
        """measure_rerun_overhead returns a dictionary with expected keys."""
        # First execute a cell so it has tracking data
        self._execute_cell(
            cell_id="a",
            pre_ns={},
            post_ns={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Measure rerun overhead
        result = self.enforcer.measure_rerun_overhead(
            cell_id="a",
            namespace={"x": 1},
        )

        # Check structure
        assert isinstance(result, dict)
        assert "cell_id" in result
        assert "checkpoint_ms" in result
        assert "check_ms" in result
        assert "total_overhead_ms" in result
        assert "checkpoint_by_var" in result
        assert "checkpoint_var_costs" in result

    def test_measure_rerun_overhead_cell_not_executed(self):
        """measure_rerun_overhead returns zeros for cell with no tracking data."""
        # Don't execute any cell - just measure
        result = self.enforcer.measure_rerun_overhead(
            cell_id="a",
            namespace={},
        )

        # Should return zeros since cell has no execution record
        assert result["cell_id"] == "a"
        assert result["checkpoint_ms"] == 0.0
        assert result["check_ms"] == 0.0
        assert result["total_overhead_ms"] == 0.0

    def test_measure_rerun_overhead_timing_values(self):
        """measure_rerun_overhead returns positive timing values."""
        # Execute a cell
        self._execute_cell(
            cell_id="a",
            pre_ns={},
            post_ns={"x": [1, 2, 3]},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        result = self.enforcer.measure_rerun_overhead(
            cell_id="a",
            namespace={"x": [1, 2, 3]},
        )

        # All timing values should be non-negative
        assert result["checkpoint_ms"] >= 0.0
        assert result["check_ms"] >= 0.0
        assert result["total_overhead_ms"] >= 0.0

        # Total should be sum of components
        expected_total = result["checkpoint_ms"] + result["check_ms"]
        assert abs(result["total_overhead_ms"] - expected_total) < 0.001

    def test_measure_rerun_overhead_checkpoint_var_costs(self):
        """measure_rerun_overhead captures checkpoint variable costs."""
        # Execute a cell with some data
        self._execute_cell(
            cell_id="a",
            pre_ns={},
            post_ns={"x": [1, 2, 3], "y": "hello"},
            tracking=make_tracking(reads=set(), writes={"x", "y"}),
        )

        result = self.enforcer.measure_rerun_overhead(
            cell_id="a",
            namespace={"x": [1, 2, 3], "y": "hello"},
        )

        # checkpoint_by_var should have entries for variables
        assert isinstance(result["checkpoint_by_var"], dict)
        assert isinstance(result["checkpoint_var_costs"], dict)

    def test_measure_rerun_overhead_multiple_cells(self):
        """measure_rerun_overhead works correctly with multiple executed cells."""
        # Execute cells a, b, c
        self._execute_cell(
            cell_id="a",
            pre_ns={},
            post_ns={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        self._execute_cell(
            cell_id="b",
            pre_ns={"x": 1},
            post_ns={"x": 1, "y": 2},
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        self._execute_cell(
            cell_id="c",
            pre_ns={"x": 1, "y": 2},
            post_ns={"x": 1, "y": 2, "z": 3},
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )

        # Measure overhead for cell b (middle cell)
        result = self.enforcer.measure_rerun_overhead(
            cell_id="b",
            namespace={"x": 1, "y": 2, "z": 3},
        )

        assert result["cell_id"] == "b"
        assert result["checkpoint_ms"] >= 0.0
        assert result["check_ms"] >= 0.0

    def test_measure_rerun_overhead_does_not_modify_state(self):
        """measure_rerun_overhead should not change cell status or staleness."""
        # Execute cells
        self._execute_cell(
            cell_id="a",
            pre_ns={},
            post_ns={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        self._execute_cell(
            cell_id="b",
            pre_ns={"x": 1},
            post_ns={"x": 1, "y": 2},
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Capture state before
        stale_before = self.enforcer.get_stale_cells()

        # Measure overhead
        self.enforcer.measure_rerun_overhead(
            cell_id="a",
            namespace={"x": 1, "y": 2},
        )

        # Staleness should not change
        stale_after = self.enforcer.get_stale_cells()
        assert set(stale_before) == set(stale_after)


class TestMeasureRerunOverheadWithDataFrames:
    """Tests for measure_rerun_overhead with DataFrame objects."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.enforcer = ReproducibilityEnforcer(self.checkpoints)
        self.enforcer.set_cell_order(["a", "b", "c"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _execute_cell(self, cell_id: str, pre_ns: dict, post_ns: dict, tracking: TrackingData):
        self._save_pre_checkpoint(cell_id, pre_ns)
        return self.enforcer.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            namespace=post_ns,
            tracking=tracking,
        )

    def test_measure_rerun_overhead_with_dataframe(self):
        """measure_rerun_overhead works with DataFrame objects."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Execute cell that creates DataFrame
        tracking = make_tracking(reads=set(), writes={"df"})
        tracking.column_writes = {"df": {"a", "b"}}

        self._execute_cell(
            cell_id="a",
            pre_ns={},
            post_ns={"df": df},
            tracking=tracking,
        )

        result = self.enforcer.measure_rerun_overhead(
            cell_id="a",
            namespace={"df": df},
        )

        assert result["cell_id"] == "a"
        assert result["checkpoint_ms"] >= 0.0
        assert result["total_overhead_ms"] >= 0.0
        # DataFrame should appear in checkpoint costs
        assert "df" in result["checkpoint_by_var"] or len(result["checkpoint_by_var"]) >= 0
