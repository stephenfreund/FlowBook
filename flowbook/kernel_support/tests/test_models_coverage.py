"""Tests for models.py - Targeting uncovered methods.

Coverage gaps include:
- TrackingData.to_access_events()
- TrackingData.to_read_events()
- TrackingData.to_json_friendly()
- TrackingData.get_read_variables()
- TrackingData.get_written_variables()
- ExecutionContext.has_cell_magics, has_shell_magics, should_profile
"""

import os
import pytest

from flowbook.kernel_support.models import (
    TrackingData,
    ExecutionProfile,
    MonotonicityViolation,
    ExecutionContext,
)


class TestTrackingDataAccessEvents:
    """Tests for TrackingData.to_access_events method."""

    def test_empty_data_no_events(self):
        """Empty tracking data yields no access events."""
        data = TrackingData()
        events = data.to_access_events()
        assert events == []

    def test_column_read_events(self):
        """Column reads are converted to ColumnRead events."""
        data = TrackingData(
            column_reads_before_writes={"df": {"price", "quantity"}},
        )
        events = data.to_access_events()
        event_strs = [(e.__class__.__name__, e.variable, e.column) for e in events]
        assert ("ColumnRead", "df", "price") in event_strs
        assert ("ColumnRead", "df", "quantity") in event_strs

    def test_column_write_events(self):
        """Column writes are converted to ColumnWrite events."""
        data = TrackingData(
            column_writes={"df": {"total"}},
        )
        events = data.to_access_events()
        event_strs = [(e.__class__.__name__, e.variable, e.column) for e in events]
        assert ("ColumnWrite", "df", "total") in event_strs

    def test_structural_read_events(self):
        """Structural reads are converted to StructuralRead events."""
        data = TrackingData(
            structural_reads={"df": {"columns", "shape"}},
        )
        events = data.to_access_events()
        event_strs = [(e.__class__.__name__, e.variable, e.attr) for e in events]
        assert ("StructuralRead", "df", "columns") in event_strs
        assert ("StructuralRead", "df", "shape") in event_strs

    def test_events_are_sorted(self):
        """Events are returned in deterministic sorted order."""
        data = TrackingData(
            column_reads_before_writes={"z_df": {"b_col", "a_col"}, "a_df": {"col"}},
            column_writes={"df": {"x"}},
            structural_reads={"df": {"shape"}},
        )
        events = data.to_access_events()
        # Column reads come first, sorted by variable then column
        assert events[0].__class__.__name__ == "ColumnRead"
        assert events[0].variable == "a_df"


class TestTrackingDataReadEvents:
    """Tests for TrackingData.to_read_events method."""

    def test_empty_data_no_events(self):
        """Empty tracking data yields no read events."""
        data = TrackingData()
        events = data.to_read_events()
        assert events == []

    def test_column_reads_included(self):
        """Column reads are included in read events."""
        data = TrackingData(
            column_reads_before_writes={"df": {"price"}},
        )
        events = data.to_read_events()
        assert len(events) == 1
        assert events[0].__class__.__name__ == "ColumnRead"

    def test_structural_reads_included(self):
        """Structural reads are included in read events."""
        data = TrackingData(
            structural_reads={"df": {"shape"}},
        )
        events = data.to_read_events()
        assert len(events) == 1
        assert events[0].__class__.__name__ == "StructuralRead"

    def test_variable_reads_without_detail(self):
        """Variable-level reads for vars without column/structural detail."""
        data = TrackingData(
            reads_before_writes={"x", "y"},
        )
        events = data.to_read_events()
        assert len(events) == 2
        names = {e.__class__.__name__ for e in events}
        assert names == {"VariableRead"}

    def test_variable_reads_excluded_when_detail_exists(self):
        """Variables with column or structural detail don't get VariableRead."""
        data = TrackingData(
            reads_before_writes={"df", "x"},
            column_reads_before_writes={"df": {"col"}},
        )
        events = data.to_read_events()
        variable_reads = [e for e in events if e.__class__.__name__ == "VariableRead"]
        # Only x should get a VariableRead; df has column detail
        assert len(variable_reads) == 1
        assert variable_reads[0].variable == "x"

    def test_column_writes_not_in_read_events(self):
        """Column writes are not included in read events."""
        data = TrackingData(
            column_writes={"df": {"total"}},
        )
        events = data.to_read_events()
        assert events == []


class TestTrackingDataJsonFriendly:
    """Tests for TrackingData.to_json_friendly method."""

    def test_empty_data(self):
        """Empty tracking data returns empty sorted lists."""
        data = TrackingData()
        result = data.to_json_friendly()
        assert result["reads"] == []
        assert result["writes"] == []
        assert result["column_reads"] == {}
        assert result["column_writes"] == {}
        assert result["structural_reads"] == {}

    def test_sorted_output(self):
        """Output lists and dicts are sorted."""
        data = TrackingData(
            reads_before_writes={"z", "a", "m"},
            writes={"b", "c"},
            column_reads_before_writes={"df": {"z_col", "a_col"}},
            structural_reads={"df": {"shape", "columns"}},
        )
        result = data.to_json_friendly()
        assert result["reads"] == ["a", "m", "z"]
        assert result["writes"] == ["b", "c"]
        assert result["column_reads"]["df"] == ["a_col", "z_col"]
        assert result["structural_reads"]["df"] == ["columns", "shape"]

    def test_file_reads_and_writes(self):
        """File paths are included as relative paths."""
        cwd = os.getcwd()
        abs_path = os.path.join(cwd, "data.csv")
        data = TrackingData(
            file_reads_before_writes={abs_path},
            file_writes={abs_path},
        )
        result = data.to_json_friendly()
        assert "data.csv" in result["file_reads"]
        assert "data.csv" in result["file_writes"]


class TestTrackingDataHelpers:
    """Tests for remaining TrackingData helper methods."""

    def test_get_read_variables(self):
        """get_read_variables combines all read sources."""
        data = TrackingData(
            reads_before_writes={"x", "y"},
            column_reads_before_writes={"df": {"col"}},
            structural_reads={"sr": {"shape"}},
        )
        result = data.get_read_variables()
        assert result == {"x", "y", "df", "sr"}

    def test_get_written_variables(self):
        """get_written_variables returns set of written vars."""
        data = TrackingData(writes={"a", "b", "c"})
        result = data.get_written_variables()
        assert result == {"a", "b", "c"}

    def test_has_structural_read_true(self):
        """has_structural_read returns True when attrs exist."""
        data = TrackingData(structural_reads={"df": {"shape"}})
        assert data.has_structural_read("df")

    def test_has_structural_read_false(self):
        """has_structural_read returns False when no attrs."""
        data = TrackingData()
        assert not data.has_structural_read("df")

    def test_has_column_structure_read(self):
        """has_column_structure_read detects column-revealing attrs."""
        data = TrackingData(structural_reads={"df": {"columns"}})
        assert data.has_column_structure_read("df")

    def test_has_column_structure_read_false(self):
        """has_column_structure_read returns False for non-column attrs."""
        data = TrackingData(structural_reads={"df": {"some_random_attr"}})
        assert not data.has_column_structure_read("df")

    def test_has_row_structure_read(self):
        """has_row_structure_read detects row-revealing attrs."""
        data = TrackingData(structural_reads={"df": {"index", "shape"}})
        assert data.has_row_structure_read("df")

    def test_has_row_structure_read_false(self):
        """has_row_structure_read returns False for non-row attrs."""
        data = TrackingData(structural_reads={"df": {"columns"}})
        assert not data.has_row_structure_read("df")


class TestExecutionContext:
    """Tests for ExecutionContext model."""

    def test_has_cell_magics_true(self):
        """has_cell_magics returns True when code starts with %."""
        ctx = ExecutionContext(code="%magic command", timeout=30, original_code="%magic command")
        assert ctx.has_cell_magics

    def test_has_cell_magics_midline(self):
        """has_cell_magics detects magic commands mid-code."""
        ctx = ExecutionContext(
            code="x = 1\n%timeit x + 1",
            timeout=30,
            original_code="x = 1\n%timeit x + 1",
        )
        assert ctx.has_cell_magics

    def test_has_cell_magics_false(self):
        """has_cell_magics returns False for normal code."""
        ctx = ExecutionContext(code="x = 1", timeout=30, original_code="x = 1")
        assert not ctx.has_cell_magics

    def test_has_shell_magics_true(self):
        """has_shell_magics returns True when code starts with !."""
        ctx = ExecutionContext(code="!ls", timeout=30, original_code="!ls")
        assert ctx.has_shell_magics

    def test_has_shell_magics_midline(self):
        """has_shell_magics detects shell commands mid-code."""
        ctx = ExecutionContext(code="x = 1\n!echo hello", timeout=30, original_code="")
        assert ctx.has_shell_magics

    def test_has_shell_magics_false(self):
        """has_shell_magics returns False for normal code."""
        ctx = ExecutionContext(code="x = 1", timeout=30, original_code="x = 1")
        assert not ctx.has_shell_magics

    def test_should_profile_true(self):
        """should_profile returns True for normal code with cell_id."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="x = 1 + 2",
            timeout=30,
            original_code="x = 1 + 2",
        )
        assert ctx.should_profile

    def test_should_profile_false_no_cell_id(self):
        """should_profile returns False when cell_id is None."""
        ctx = ExecutionContext(code="x = 1", timeout=30, original_code="x = 1")
        assert not ctx.should_profile

    def test_should_profile_false_with_magic(self):
        """should_profile returns False when code has magics."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="%timeit x + 1",
            timeout=30,
            original_code="%timeit x + 1",
        )
        assert not ctx.should_profile

    def test_should_profile_false_with_shell(self):
        """should_profile returns False when code has shell commands."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="!ls",
            timeout=30,
            original_code="!ls",
        )
        assert not ctx.should_profile


class TestMonotonicityViolation:
    """Tests for MonotonicityViolation.to_error_result."""

    def test_to_error_result(self):
        """to_error_result produces kernel error format."""
        violation = MonotonicityViolation(
            violated_vars=["x", "y"],
            diff_details="x changed from 1 to 2",
            error_summary="Monotonicity violation: ['x', 'y']",
        )
        result = violation.to_error_result(execution_count=5)
        assert result["status"] == "error"
        assert result["execution_count"] == 5
        assert result["ename"] == "MonotonicityError"
        assert result["evalue"] == "Monotonicity violation: ['x', 'y']"
        assert "x changed from 1 to 2" in result["traceback"]
