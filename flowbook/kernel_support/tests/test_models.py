"""
Tests for kernel/models.py - Pydantic models for kernel execution.

Tests cover:
- TrackingData: Variable access pattern tracking
- ExecutionProfile: Profiling and timing data
- ExecutionContext: Pre-execution state and configuration
"""

import pytest
from flowbook.kernel_support.models import (
    TrackingData,
    ExecutionProfile,
    ExecutionContext,
)


class TestTrackingData:
    """Tests for TrackingData model."""

    def test_empty_tracking_data(self):
        """Empty tracking data has default empty collections."""
        data = TrackingData()
        assert data.reads_before_writes == set()
        assert data.writes == set()
        assert data.column_reads_before_writes == {}
        assert data.column_writes == {}

    def test_tracking_data_with_values(self):
        """TrackingData stores provided values correctly."""
        data = TrackingData(
            reads_before_writes=["x", "y"],
            writes=["z", "w"],
            column_reads_before_writes={"df": ["col1", "col2"]},
            column_writes={"df": ["col3"]},
        )
        assert data.reads_before_writes == {"x", "y"}
        assert data.writes == {"z", "w"}
        assert data.column_reads_before_writes == {"df": {"col1", "col2"}}
        assert data.column_writes == {"df": {"col3"}}

    def test_get_rbw_vars_empty(self):
        """get_rbw_vars returns empty set for empty data."""
        data = TrackingData()
        assert data.get_rbw_vars() == set()

    def test_get_rbw_vars_with_values(self):
        """get_rbw_vars returns set of RBW variables."""
        data = TrackingData(reads_before_writes=["x", "y", "z"])
        assert data.get_rbw_vars() == {"x", "y", "z"}

    def test_get_column_rbw_sets_empty(self):
        """get_column_rbw_sets returns empty dict for empty data."""
        data = TrackingData()
        assert data.get_column_rbw_sets() == {}

    def test_get_column_rbw_sets_with_values(self):
        """get_column_rbw_sets converts lists to sets."""
        data = TrackingData(
            column_reads_before_writes={
                "df1": ["a", "b", "c"],
                "df2": ["x", "y"],
            }
        )
        result = data.get_column_rbw_sets()
        assert result == {
            "df1": {"a", "b", "c"},
            "df2": {"x", "y"},
        }

    def test_tracking_data_serialization(self):
        """TrackingData can be serialized to dict and back."""
        data = TrackingData(
            reads_before_writes=["a", "b"],
            writes=["c"],
            column_reads_before_writes={"df": ["col1"]},
            column_writes={"df": ["col2"]},
        )
        serialized = data.model_dump()
        restored = TrackingData.model_validate(serialized)
        assert restored == data

    def test_tracking_data_json_serialization(self):
        """TrackingData can be serialized to JSON and back."""
        data = TrackingData(
            reads_before_writes=["x"],
            writes=["y"],
        )
        json_str = data.model_dump_json()
        restored = TrackingData.model_validate_json(json_str)
        assert restored == data


class TestExecutionProfile:
    """Tests for ExecutionProfile model."""

    def test_execution_profile_required_fields(self):
        """ExecutionProfile requires duration."""
        with pytest.raises(Exception):  # ValidationError
            ExecutionProfile()

    def test_execution_profile_defaults(self):
        """ExecutionProfile has sensible defaults."""
        profile = ExecutionProfile(duration=1.5)
        assert profile.duration == 1.5
        assert profile.profile == ""
        assert profile.env == {}
        assert profile.env_after == {}

    def test_execution_profile_with_all_fields(self):
        """ExecutionProfile stores all provided values."""
        profile = ExecutionProfile(
            duration=2.5,
            profile="CPU: 50%, Memory: 100MB",
            env={"x": "int", "y": "float"},
            env_after={"x": "int", "y": "float", "z": "str"},
        )
        assert profile.duration == 2.5
        assert profile.profile == "CPU: 50%, Memory: 100MB"
        assert profile.env == {"x": "int", "y": "float"}
        assert profile.env_after == {"x": "int", "y": "float", "z": "str"}

    def test_execution_profile_serialization(self):
        """ExecutionProfile can be serialized."""
        profile = ExecutionProfile(
            duration=1.0,
            profile="test",
            env={"a": "int"},
            env_after={"a": "int", "b": "str"},
        )
        serialized = profile.model_dump()
        assert serialized["duration"] == 1.0
        assert serialized["profile"] == "test"


class TestExecutionContext:
    """Tests for ExecutionContext model."""

    def test_execution_context_required_fields(self):
        """ExecutionContext requires code, timeout, original_code."""
        with pytest.raises(Exception):
            ExecutionContext()

    def test_execution_context_minimal(self):
        """ExecutionContext works with minimal fields."""
        ctx = ExecutionContext(
            code="x = 1",
            timeout=30.0,
            original_code="x = 1",
        )
        assert ctx.cell_id is None
        assert ctx.code == "x = 1"
        assert ctx.timeout == 30.0
        assert ctx.original_code == "x = 1"

    def test_execution_context_with_cell_id(self):
        """ExecutionContext stores cell_id."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="y = 2",
            timeout=60.0,
            original_code="y = 2",
        )
        assert ctx.cell_id == "abcd"

    def test_has_cell_magics_false(self):
        """has_cell_magics returns False for regular code."""
        ctx = ExecutionContext(
            code="x = 1\ny = 2",
            timeout=30.0,
            original_code="x = 1\ny = 2",
        )
        assert ctx.has_cell_magics is False

    def test_has_cell_magics_starts_with_percent(self):
        """has_cell_magics returns True for code starting with %."""
        ctx = ExecutionContext(
            code="%timeit x = 1",
            timeout=30.0,
            original_code="%timeit x = 1",
        )
        assert ctx.has_cell_magics is True

    def test_has_cell_magics_contains_percent(self):
        """has_cell_magics returns True for code containing newline+%."""
        ctx = ExecutionContext(
            code="x = 1\n%time y = 2",
            timeout=30.0,
            original_code="x = 1\n%time y = 2",
        )
        assert ctx.has_cell_magics is True

    def test_has_shell_magics_false(self):
        """has_shell_magics returns False for regular code."""
        ctx = ExecutionContext(
            code="x = 1",
            timeout=30.0,
            original_code="x = 1",
        )
        assert ctx.has_shell_magics is False

    def test_has_shell_magics_starts_with_bang(self):
        """has_shell_magics returns True for code starting with !."""
        ctx = ExecutionContext(
            code="!ls -la",
            timeout=30.0,
            original_code="!ls -la",
        )
        assert ctx.has_shell_magics is True

    def test_has_shell_magics_contains_bang(self):
        """has_shell_magics returns True for code containing newline+!."""
        ctx = ExecutionContext(
            code="x = 1\n!pip install foo",
            timeout=30.0,
            original_code="x = 1\n!pip install foo",
        )
        assert ctx.has_shell_magics is True

    def test_should_profile_true(self):
        """should_profile returns True for regular code with cell_id."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="x = 1",
            timeout=30.0,
            original_code="x = 1",
        )
        assert ctx.should_profile is True

    def test_should_profile_false_no_cell_id(self):
        """should_profile returns False without cell_id."""
        ctx = ExecutionContext(
            cell_id=None,
            code="x = 1",
            timeout=30.0,
            original_code="x = 1",
        )
        assert ctx.should_profile is False

    def test_should_profile_false_cell_magics(self):
        """should_profile returns False for cell magics."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="%timeit x = 1",
            timeout=30.0,
            original_code="%timeit x = 1",
        )
        assert ctx.should_profile is False

    def test_should_profile_false_shell_magics(self):
        """should_profile returns False for shell commands."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="!ls",
            timeout=30.0,
            original_code="!ls",
        )
        assert ctx.should_profile is False

    def test_execution_context_serialization(self):
        """ExecutionContext can be serialized."""
        ctx = ExecutionContext(
            cell_id="test",
            code="x = 1",
            timeout=30.0,
            original_code="# timeout 60\nx = 1",
        )
        serialized = ctx.model_dump()
        restored = ExecutionContext.model_validate(serialized)
        assert restored.cell_id == ctx.cell_id
        assert restored.code == ctx.code
        assert restored.timeout == ctx.timeout
        assert restored.original_code == ctx.original_code
