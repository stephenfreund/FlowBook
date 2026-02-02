"""
Tests for kernel/monotonicity.py - MonotonicityEnforcer class.

Tests cover:
- Basic monotonicity checking (pass/fail scenarios)
- Pre-state saving
- State restoration on violation
- Column-level monotonicity
- Error message formatting
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from flowbook.kernel_support.checkpoint import Checkpoints
from flowbook.kernel_support.monotonicity import MonotonicityEnforcer
from flowbook.kernel_support.models import TrackingData, MonotonicityViolation


class TestMonotonicityEnforcerBasics:
    """Basic tests for MonotonicityEnforcer."""

    def test_init(self):
        """MonotonicityEnforcer can be initialized."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        assert enforcer is not None

    def test_save_pre_state(self):
        """save_pre_state saves a checkpoint."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1, "y": 2}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        enforcer.save_pre_state("test_cell")

        # Checkpoint should exist
        assert "_monotone_pre" in checkpoints.list()

    def test_check_passes_no_rbw(self):
        """Check passes when there are no RBW variables."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # No RBW variables
        tracking = TrackingData(reads_before_writes=[], writes=["y"])

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None  # No violation


class TestMonotonicityPassing:
    """Tests for cases where monotonicity check passes."""

    def test_rbw_var_unchanged(self):
        """Check passes when RBW variable is unchanged."""
        checkpoints = Checkpoints()
        user_ns = {"x": 10, "y": 20}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Read x, but don't modify it
        tracking = TrackingData(
            reads_before_writes=["x"],
            writes=["z"],  # Write different variable
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None  # No violation

    def test_multiple_rbw_vars_unchanged(self):
        """Check passes when multiple RBW variables are unchanged."""
        checkpoints = Checkpoints()
        user_ns = {"a": 1, "b": 2, "c": 3}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData(
            reads_before_writes=["a", "b", "c"],
            writes=["d"],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_write_only_vars_can_change(self):
        """Write-only variables can change without violation."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1, "y": 2}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify y (not read first)
        user_ns["y"] = 100

        tracking = TrackingData(
            reads_before_writes=["x"],  # Only x is RBW
            writes=["y"],  # y was written
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None  # y change doesn't violate monotonicity


class TestMonotonicityViolations:
    """Tests for monotonicity violation detection."""

    def test_rbw_var_modified(self):
        """Check fails when RBW variable is modified."""
        checkpoints = Checkpoints()
        user_ns = {"x": 10}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify x after it was read
        user_ns["x"] = 999

        tracking = TrackingData(
            reads_before_writes=["x"],
            writes=["x"],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None
        assert isinstance(result, MonotonicityViolation)
        assert "x" in result.violated_vars

    def test_violation_restores_state(self):
        """Violation restores pre-execution state."""
        checkpoints = Checkpoints()
        user_ns = {"x": 10, "y": 20}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify x
        user_ns["x"] = 999
        user_ns["y"] = 888

        tracking = TrackingData(
            reads_before_writes=["x"],
            writes=["x", "y"],
        )

        result = enforcer.check_and_enforce(tracking, "test")

        # State should be restored
        assert user_ns["x"] == 10  # Restored
        # Note: y might also be restored depending on checkpoint behavior

    def test_multiple_violations(self):
        """Multiple RBW variables can be violated."""
        checkpoints = Checkpoints()
        user_ns = {"a": 1, "b": 2, "c": 3}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify all
        user_ns["a"] = 100
        user_ns["b"] = 200
        user_ns["c"] = 300

        tracking = TrackingData(
            reads_before_writes=["a", "b", "c"],
            writes=["a", "b", "c"],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None
        assert len(result.violated_vars) == 3

    def test_violation_cleans_up_checkpoint(self):
        """Violation cleans up the temporary checkpoint."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        user_ns["x"] = 999

        tracking = TrackingData(reads_before_writes=["x"], writes=["x"])
        enforcer.check_and_enforce(tracking, "test")

        # Temporary checkpoint should be cleaned up
        assert "_monotone_pre" not in checkpoints.list()


class TestMonotonicityWithDataFrames:
    """Tests for monotonicity with pandas DataFrames."""

    def test_dataframe_unchanged_passes(self):
        """DataFrame that's unchanged passes check."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        user_ns = {"df": df}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData(
            reads_before_writes=["df"],
            writes=[],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_dataframe_modified_fails(self):
        """DataFrame that's modified fails check."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({"a": [1, 2, 3]})
        user_ns = {"df": df}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify the DataFrame
        user_ns["df"]["a"] = [100, 200, 300]

        tracking = TrackingData(
            reads_before_writes=["df"],
            writes=["df"],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None
        assert "df" in result.violated_vars

    def test_dataframe_column_level_check(self):
        """Column-level RBW allows adding columns."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({"a": [1, 2, 3]})
        user_ns = {"df": df}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Add a new column (should be OK if only 'a' was read)
        user_ns["df"]["b"] = [4, 5, 6]

        # Column-level RBW says only 'a' was read
        tracking = TrackingData(
            reads_before_writes=["df"],
            writes=["df"],
            column_reads_before_writes={"df": ["a"]},
            column_writes={"df": ["b"]},
        )

        result = enforcer.check_and_enforce(tracking, "test")
        # Should pass because 'a' wasn't modified
        assert result is None

    def test_dataframe_column_modified_fails(self):
        """Modifying a read column fails check."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        user_ns = {"df": df}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify column 'a' which was read
        user_ns["df"]["a"] = [100, 200, 300]

        tracking = TrackingData(
            reads_before_writes=["df"],
            writes=["df"],
            column_reads_before_writes={"df": ["a"]},
            column_writes={"df": ["a"]},
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None


class TestMonotonicityWithNumpy:
    """Tests for monotonicity with numpy arrays."""

    def test_numpy_array_unchanged_passes(self):
        """Numpy array unchanged passes check."""
        checkpoints = Checkpoints()
        arr = np.array([1, 2, 3])
        user_ns = {"arr": arr.copy()}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData(
            reads_before_writes=["arr"],
            writes=[],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_numpy_array_modified_fails(self):
        """Numpy array modified fails check."""
        checkpoints = Checkpoints()
        arr = np.array([1, 2, 3])
        user_ns = {"arr": arr}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify array in place
        user_ns["arr"][0] = 999

        tracking = TrackingData(
            reads_before_writes=["arr"],
            writes=["arr"],
        )

        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None


class TestViolationFormatting:
    """Tests for violation error message formatting."""

    def test_format_diff_details_simple(self):
        """Violation details are formatted correctly."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        user_ns["x"] = 999

        tracking = TrackingData(reads_before_writes=["x"], writes=["x"])
        result = enforcer.check_and_enforce(tracking, "test")

        assert "x" in result.diff_details
        assert "Monotonicity violation" in result.diff_details

    def test_violation_error_summary(self):
        """Violation has proper error summary."""
        checkpoints = Checkpoints()
        user_ns = {"a": 1, "b": 2}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        user_ns["a"] = 100
        user_ns["b"] = 200

        tracking = TrackingData(reads_before_writes=["a", "b"], writes=["a", "b"])
        result = enforcer.check_and_enforce(tracking, "test")

        assert "Monotonicity violation" in result.error_summary
        assert "a" in result.error_summary or "b" in result.error_summary


class TestMonotonicityEdgeCases:
    """Edge case tests for MonotonicityEnforcer."""

    def test_empty_namespace(self):
        """Works with empty namespace."""
        checkpoints = Checkpoints()
        user_ns = {}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData()
        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_no_writes(self):
        """Works when there are no writes."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData(reads_before_writes=["x"], writes=[])
        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_none_value(self):
        """Works with None values."""
        checkpoints = Checkpoints()
        user_ns = {"x": None}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData(reads_before_writes=["x"], writes=[])
        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_complex_nested_structure(self):
        """Works with complex nested structures."""
        checkpoints = Checkpoints()
        user_ns = {
            "data": {
                "config": {"a": 1, "b": 2},
                "values": [1, 2, 3],
            }
        }
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        tracking = TrackingData(reads_before_writes=["data"], writes=[])
        result = enforcer.check_and_enforce(tracking, "test")
        assert result is None

    def test_nested_structure_modified(self):
        """Detects modification of nested structure."""
        checkpoints = Checkpoints()
        user_ns = {"data": {"x": 1}}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Modify nested value
        user_ns["data"]["x"] = 999

        tracking = TrackingData(reads_before_writes=["data"], writes=["data"])
        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None

    def test_multiple_cells(self):
        """Works across multiple cell executions."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1}

        # Cell 1
        enforcer1 = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer1.save_pre_state("cell1")
        tracking1 = TrackingData(reads_before_writes=["x"], writes=[])
        result1 = enforcer1.check_and_enforce(tracking1, "cell1")
        assert result1 is None

        # Cell 2
        user_ns["y"] = 2
        enforcer2 = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer2.save_pre_state("cell2")
        tracking2 = TrackingData(reads_before_writes=["y"], writes=[])
        result2 = enforcer2.check_and_enforce(tracking2, "cell2")
        assert result2 is None


class TestMonotonicityFloatTolerance:
    """Tests for float tolerance in monotonicity checking."""

    def test_float_within_tolerance_passes(self):
        """Floats within tolerance pass check."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1.0}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Very small change (within tolerance)
        user_ns["x"] = 1.0 + 1e-10

        tracking = TrackingData(reads_before_writes=["x"], writes=["x"])
        result = enforcer.check_and_enforce(tracking, "test")
        # Should pass due to tolerance
        assert result is None

    def test_float_outside_tolerance_fails(self):
        """Floats outside tolerance fail check."""
        checkpoints = Checkpoints()
        user_ns = {"x": 1.0}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("test")

        # Large change
        user_ns["x"] = 2.0

        tracking = TrackingData(reads_before_writes=["x"], writes=["x"])
        result = enforcer.check_and_enforce(tracking, "test")
        assert result is not None
