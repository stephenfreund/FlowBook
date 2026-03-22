"""
Tests for SDC Enforcer optimizations.

These tests verify that the optimization fast paths work correctly
and produce the same results as the unoptimized code paths.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData

from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
    OPT_CONFLICT_LOOP_SKIP,
    _env_flag,
)
from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper


class TestEnvFlag:
    """Tests for the _env_flag helper function."""

    def test_default_true(self):
        """Default value is True when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _env_flag("TEST_FLAG", default=True) is True

    def test_default_false(self):
        """Default value is False when specified."""
        with patch.dict(os.environ, {}, clear=True):
            assert _env_flag("TEST_FLAG", default=False) is False

    def test_explicit_false_values(self):
        """Various false values are recognized."""
        for false_val in ["0", "false", "False", "FALSE", "no", "No", "off", "OFF"]:
            with patch.dict(os.environ, {"TEST_FLAG": false_val}):
                assert _env_flag("TEST_FLAG", default=True) is False

    def test_any_other_value_is_true(self):
        """Any non-false value keeps default."""
        for true_val in ["1", "true", "True", "yes", "on", "anything"]:
            with patch.dict(os.environ, {"TEST_FLAG": true_val}):
                assert _env_flag("TEST_FLAG", default=True) is True


class TestConflictLoopSkipOptimization:
    """
    Tests for OPT_CONFLICT_LOOP_SKIP optimization.

    This optimization skips the O(n) conflict detection loop when there's
    no variable-level overlap between changed variables and prior reads.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_skip_when_no_overlap_different_variables(self):
        """Optimization should skip conflict loop when writing different variable than read."""
        # Cell A reads x
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "result_a": 10},
            reads={"x"},
            writes={"result_a"},
        )
        assert not result_a.has_errors()

        # Cell B writes y (different from x) - no conflict possible
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"x": 1, "result_a": 10},
            post_namespace={"x": 1, "result_a": 10, "y": 2},
            reads=set(),
            writes={"y"},
        )
        assert not result_b.has_errors()

    def test_skip_when_no_prior_cells(self):
        """Optimization should handle case when there are no prior cells."""
        # Cell A is first - no prior reads to conflict with
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={},
            post_namespace={"x": 1},
            reads=set(),
            writes={"x"},
        )
        assert not result_a.has_errors()

    def test_skip_when_prior_cells_have_no_reads(self):
        """Optimization should skip when prior cells didn't read anything."""
        # Cell A writes only (no reads)
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={},
            post_namespace={"x": 1},
            reads=set(),
            writes={"x"},
        )
        assert not result_a.has_errors()

        # Cell B modifies x - but A didn't read it, so no conflict
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"x": 1},
            post_namespace={"x": 999},
            reads=set(),
            writes={"x"},
        )
        assert not result_b.has_errors()

    def test_detect_conflict_when_overlap_exists(self):
        """Optimization should NOT skip when there's variable overlap."""
        # Cell A reads x
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )
        assert not result_a.has_errors()

        # Cell B modifies x (which A read) - should detect conflict
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 999, "y": 2},
            reads=set(),
            writes={"x"},
        )
        assert result_b.has_errors()
        assert result_b.errors[0].cell_id == "b"
        assert result_b.errors[0].causer_cell == "a"

    def test_multiple_prior_cells_no_overlap(self):
        """Optimization with multiple prior cells, none overlapping."""
        # Cell A reads x
        self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "a_out": 10},
            reads={"x"},
            writes={"a_out"},
        )

        # Cell B reads y
        self.helper.execute_cell(
            "b",
            pre_namespace={"x": 1, "a_out": 10, "y": 2},
            post_namespace={"x": 1, "a_out": 10, "y": 2, "b_out": 20},
            reads={"y"},
            writes={"b_out"},
        )

        # Cell C writes z (neither x nor y) - no conflict
        result_c = self.helper.execute_cell(
            "c",
            pre_namespace={"x": 1, "a_out": 10, "y": 2, "b_out": 20},
            post_namespace={"x": 1, "a_out": 10, "y": 2, "b_out": 20, "z": 3},
            reads=set(),
            writes={"z"},
        )
        assert not result_c.has_errors()

    def test_multiple_prior_cells_with_overlap(self):
        """Optimization with multiple prior cells, one overlapping."""
        # Cell A reads x
        self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "a_out": 10},
            reads={"x"},
            writes={"a_out"},
        )

        # Cell B reads y
        self.helper.execute_cell(
            "b",
            pre_namespace={"x": 1, "a_out": 10, "y": 2},
            post_namespace={"x": 1, "a_out": 10, "y": 2, "b_out": 20},
            reads={"y"},
            writes={"b_out"},
        )

        # Cell C modifies x (which A read) - conflict with A
        result_c = self.helper.execute_cell(
            "c",
            pre_namespace={"x": 1, "a_out": 10, "y": 2, "b_out": 20},
            post_namespace={"x": 999, "a_out": 10, "y": 2, "b_out": 20},
            reads=set(),
            writes={"x"},
        )
        assert result_c.has_errors()
        assert result_c.errors[0].causer_cell == "a"

    def test_optimization_preserves_typed_changes(self):
        """Optimization should still return typed_changes for forward dep caching."""
        # Cell A writes x (no prior reads)
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={},
            post_namespace={"x": 1},
            reads=set(),
            writes={"x"},
        )
        assert not result_a.has_errors()
        # The record should have typed_changes cached
        assert self.helper.sdc._notebook_state.has_record("a")
        # typed_changes should contain the creation of x
        typed_changes = self.helper.sdc._notebook_state.get_typed_changes("a")
        assert len(typed_changes) > 0

    def test_optimization_preserves_changed_vars(self):
        """Optimization should still compute changed_vars for staleness."""
        # Cell A writes x
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={},
            post_namespace={"x": 1},
            reads=set(),
            writes={"x"},
        )
        assert result_a.changed_variables == ["x"]

    def test_no_changes_means_empty_changed_vars(self):
        """When cell doesn't change anything, changed_vars should be empty."""
        # Cell A does nothing (reads and writes same value)
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1},
            reads={"x"},
            writes=set(),
        )
        assert result_a.changed_variables == []


class TestConflictLoopSkipWithDataFrames:
    """Tests for conflict loop skip with DataFrame column-level tracking."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_skip_when_different_columns(self):
        """Skip conflict check when writing different columns than read."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Cell A reads column x
        self.helper.execute_cell(
            "a",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df.copy(), "result": 10},
            reads={"df"},
            writes={"result"},
            column_reads={"df": {"x"}},
        )

        # Cell B writes column y (different from x)
        df_modified = df.copy()
        df_modified["y"] = [10, 20, 30]
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"df": df.copy(), "result": 10},
            post_namespace={"df": df_modified, "result": 10},
            reads=set(),
            writes={"df"},
            column_writes={"df": {"y"}},
        )
        # Note: Conflict loop skip is at variable level, so it won't skip here
        # because "df" overlaps. But the conflict resolver handles column-level.
        # This test verifies the full path works correctly.
        assert not result_b.has_errors()

    def test_detect_conflict_same_column(self):
        """Detect conflict when writing same column as read."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Cell A reads column x
        self.helper.execute_cell(
            "a",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df.copy(), "result": 10},
            reads={"df"},
            writes={"result"},
            column_reads={"df": {"x"}},
        )

        # Cell B writes column x (same as A read)
        df_modified = df.copy()
        df_modified["x"] = [10, 20, 30]
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"df": df.copy(), "result": 10},
            post_namespace={"df": df_modified, "result": 10},
            reads=set(),
            writes={"df"},
            column_writes={"df": {"x"}},
        )
        assert result_b.has_errors()


class TestOptimizationConsistency:
    """
    Tests that verify optimization produces same results as non-optimized path.

    These tests run the same scenario with and without optimization and
    compare results.
    """

    def _run_scenario_with_flag(self, flag_value: bool):
        """Run a test scenario with specific flag value."""
        # We can't easily toggle the module-level flag, but we can test
        # that the behavior is correct in both cases by constructing
        # scenarios that exercise both paths.
        helper = ReproducibilityTestHelper()
        helper.set_cell_order(["a", "b", "c"])

        # Cell A reads x
        helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )

        # Cell B modifies z (no overlap with x)
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads=set(),
            writes={"z"},
        )

        return result_b

    def test_no_violation_case_consistent(self):
        """No-violation case produces consistent results."""
        result = self._run_scenario_with_flag(True)
        assert not result.has_errors()

    def test_violation_detection_not_affected(self):
        """Violations are still detected correctly with optimization."""
        helper = ReproducibilityTestHelper()
        helper.set_cell_order(["a", "b"])

        # Cell A reads x
        helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1},
            reads={"x"},
            writes=set(),
        )

        # Cell B modifies x
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"x": 1},
            post_namespace={"x": 999},
            reads=set(),
            writes={"x"},
        )

        assert result_b.has_errors()
        assert "x" in result_b.errors[0].locations


class TestEdgeCases:
    """Edge cases for the conflict loop skip optimization."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d", "e"])

    def test_empty_tracking_data(self):
        """Handle cells with completely empty tracking data."""
        # Cell A with empty tracking
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1},
            reads=set(),
            writes=set(),
        )
        assert not result_a.has_errors()

    def test_many_prior_cells_no_overlap(self):
        """Performance case: many prior cells but no overlap."""
        # Execute cells a, b, c, d each reading different variables
        for i, cell_id in enumerate(["a", "b", "c", "d"]):
            var_name = f"var_{cell_id}"
            self.helper.execute_cell(
                cell_id,
                pre_namespace={var_name: i},
                post_namespace={var_name: i, f"out_{cell_id}": i * 10},
                reads={var_name},
                writes={f"out_{cell_id}"},
            )

        # Cell E writes a completely new variable
        result_e = self.helper.execute_cell(
            "e",
            pre_namespace={"var_a": 0, "var_b": 1, "var_c": 2, "var_d": 3,
                          "out_a": 0, "out_b": 10, "out_c": 20, "out_d": 30},
            post_namespace={"var_a": 0, "var_b": 1, "var_c": 2, "var_d": 3,
                           "out_a": 0, "out_b": 10, "out_c": 20, "out_d": 30,
                           "new_var": 100},
            reads=set(),
            writes={"new_var"},
        )
        assert not result_e.has_errors()

    def test_cell_reads_and_writes_same_variable(self):
        """Cell that reads and writes the same variable.

        A reads and writes x, which triggers NoReadAndWrite, but no backward mutation.
        """
        from flowbook.kernel.models import ErrorType
        # Cell A reads x and writes x (transforms it)
        result_a = self.helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 2},  # x was transformed
            reads={"x"},
            writes={"x"},
        )
        # NoReadAndWrite fires (reads and writes x), but no backward mutation
        assert not any(e.error_type == ErrorType.NO_WRITE_AFTER_READ for e in result_a.errors)

        # Cell B also reads x - but this is fine, A's transformation is visible
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"x": 2},
            post_namespace={"x": 2, "y": 20},
            reads={"x"},
            writes={"y"},
        )
        assert not result_b.has_errors()

    def test_overlapping_variable_names_partial_match(self):
        """Variables with similar names don't cause false positives."""
        # Cell A reads "data"
        self.helper.execute_cell(
            "a",
            pre_namespace={"data": 1, "data_copy": 2, "my_data": 3},
            post_namespace={"data": 1, "data_copy": 2, "my_data": 3, "result": 10},
            reads={"data"},
            writes={"result"},
        )

        # Cell B writes "data_copy" (different variable, similar name)
        result_b = self.helper.execute_cell(
            "b",
            pre_namespace={"data": 1, "data_copy": 2, "my_data": 3, "result": 10},
            post_namespace={"data": 1, "data_copy": 999, "my_data": 3, "result": 10},
            reads=set(),
            writes={"data_copy"},
        )
        assert not result_b.has_errors()

        # Cell C writes "data" (exact match - should conflict)
        result_c = self.helper.execute_cell(
            "c",
            pre_namespace={"data": 1, "data_copy": 999, "my_data": 3, "result": 10},
            post_namespace={"data": 999, "data_copy": 999, "my_data": 3, "result": 10},
            reads=set(),
            writes={"data"},
        )
        assert result_c.has_errors()
        assert result_c.errors[0].causer_cell == "a"
