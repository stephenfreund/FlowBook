"""
Tests for kernel/models.py - Pydantic models for kernel execution.

Tests cover:
- TrackingData: Variable access pattern tracking
- ExecutionProfile: Profiling and timing data
- ExecutionMetadata: Complete execution metadata
- MonotonicityViolation: Monotonicity constraint violations
- ExecutionContext: Pre-execution state and configuration
"""

import pytest
from flowbook.kernel_support.models import (
    TrackingData,
    ExecutionProfile,
    ExecutionMetadata,
    MonotonicityViolation,
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


class TestExecutionMetadata:
    """Tests for ExecutionMetadata model."""

    def test_execution_metadata_required_fields(self):
        """ExecutionMetadata requires profile."""
        with pytest.raises(Exception):  # ValidationError
            ExecutionMetadata()

    def test_execution_metadata_with_profile_only(self):
        """ExecutionMetadata works with just profile."""
        profile = ExecutionProfile(duration=1.0)
        metadata = ExecutionMetadata(profile=profile)
        assert metadata.profile == profile
        assert metadata.dynamic_dependencies is None

    def test_execution_metadata_with_tracking(self):
        """ExecutionMetadata stores tracking data."""
        profile = ExecutionProfile(duration=1.0)
        tracking = TrackingData(reads_before_writes=["x"])
        metadata = ExecutionMetadata(
            profile=profile,
            dynamic_dependencies=tracking,
        )
        assert metadata.profile == profile
        assert metadata.dynamic_dependencies == tracking

    def test_to_display_metadata_without_tracking(self):
        """to_display_metadata works without tracking data."""
        profile = ExecutionProfile(duration=1.5, profile="test")
        metadata = ExecutionMetadata(profile=profile)
        display = metadata.to_display_metadata()

        assert "profile" in display
        assert display["profile"]["duration"] == 1.5
        assert display["profile"]["profile"] == "test"
        assert "dynamic_dependencies" not in display

    def test_to_display_metadata_with_tracking(self):
        """to_display_metadata includes tracking data when present."""
        profile = ExecutionProfile(duration=1.0)
        tracking = TrackingData(reads_before_writes=["x"], writes=["y"])
        metadata = ExecutionMetadata(
            profile=profile,
            dynamic_dependencies=tracking,
        )
        display = metadata.to_display_metadata()

        assert "profile" in display
        assert "dynamic_dependencies" in display
        assert display["dynamic_dependencies"]["reads_before_writes"] == {"x"}
        assert display["dynamic_dependencies"]["writes"] == {"y"}


class TestMonotonicityViolation:
    """Tests for MonotonicityViolation model."""

    def test_monotonicity_violation_required_fields(self):
        """MonotonicityViolation requires all fields."""
        with pytest.raises(Exception):
            MonotonicityViolation()

    def test_monotonicity_violation_creation(self):
        """MonotonicityViolation stores provided values."""
        violation = MonotonicityViolation(
            violated_vars=["x", "y"],
            diff_details="x changed from 1 to 2",
            error_summary="Monotonicity violation: ['x', 'y']",
        )
        assert violation.violated_vars == ["x", "y"]
        assert violation.diff_details == "x changed from 1 to 2"
        assert violation.error_summary == "Monotonicity violation: ['x', 'y']"

    def test_to_error_result(self):
        """to_error_result creates proper kernel error format."""
        violation = MonotonicityViolation(
            violated_vars=["x"],
            diff_details="Variable x was modified",
            error_summary="Monotonicity violation: ['x']",
        )
        result = violation.to_error_result(execution_count=5)

        assert result["status"] == "error"
        assert result["execution_count"] == 5
        assert result["ename"] == "MonotonicityError"
        assert result["evalue"] == "Monotonicity violation: ['x']"
        assert result["traceback"] == ["Variable x was modified"]

    def test_to_error_result_multiple_vars(self):
        """to_error_result handles multiple violated variables."""
        violation = MonotonicityViolation(
            violated_vars=["a", "b", "c"],
            diff_details="Multiple variables modified:\na: 1->2\nb: 3->4\nc: 5->6",
            error_summary="Monotonicity violation: ['a', 'b', 'c']",
        )
        result = violation.to_error_result(execution_count=10)

        assert result["status"] == "error"
        assert "a" in result["evalue"]
        assert "b" in result["evalue"]
        assert "c" in result["evalue"]


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
