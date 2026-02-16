"""Tests for types.py - Type definitions for structured diff results.

Targets uncovered code paths including:
- PathComponent classes (RootComponent, IndexComponent, KeyComponent, etc.)
- ValueComparison model_post_init tolerance bug detection
- MemoryCheckpointDiffResult filtering methods (close_only, different_only)
- CompoundDiff handling in filters
- serialize_diff_result function
- format_diff_as_markdown function
- ExecutionError, TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash models
"""

import math
import pytest

from flowbook.kernel_support.types import (
    RootComponent,
    IndexComponent,
    KeyComponent,
    AttributeComponent,
    DataFrameLocation,
    ValueComparison,
    CompoundDiff,
    MemoryCheckpointDiffResult,
    serialize_diff_result,
    format_diff_as_markdown,
    ExecutionError,
    TestCodeSuccess,
    TestCodeOriginalCrash,
    TestCodeModifiedCrash,
)


class TestPathComponents:
    """Tests for PathComponent subclasses."""

    def test_root_component_str(self):
        """RootComponent __str__ returns the name."""
        c = RootComponent("myvar")
        assert str(c) == "myvar"

    def test_index_component_str(self):
        """IndexComponent __str__ returns bracket notation."""
        c = IndexComponent(5)
        assert str(c) == "[5]"

    def test_key_component_str(self):
        """KeyComponent __str__ returns bracket with repr."""
        c = KeyComponent("name")
        assert str(c) == "['name']"

    def test_attribute_component_str(self):
        """AttributeComponent __str__ returns dot notation."""
        c = AttributeComponent("shape")
        assert str(c) == ".shape"

    def test_dataframe_location_str(self):
        """DataFrameLocation __str__ returns row, col format."""
        c = DataFrameLocation(0, "col_a")
        assert str(c) == "[0, 'col_a']"


class TestValueComparison:
    """Tests for ValueComparison model."""

    def test_basic_different(self):
        """ValueComparison with status 'different'."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="1 vs 2")
        assert vc.status == "different"
        assert not vc.is_close

    def test_basic_close(self):
        """ValueComparison with status 'close'."""
        vc = ValueComparison(
            status="close", value1=1.0, value2=1.0001, message="close"
        )
        assert vc.is_close

    def test_model_post_init_non_float_skipped(self):
        """model_post_init tolerance check skips non-float values."""
        # Integer values - should not trigger tolerance bug
        vc = ValueComparison(status="different", value1=1, value2=2, message="int diff")
        assert "TOLERANCE BUG DETECTED" not in vc.message

    def test_model_post_init_nan_skipped(self):
        """model_post_init tolerance check skips NaN values."""
        vc = ValueComparison(
            status="different",
            value1=float("nan"),
            value2=1.0,
            message="nan diff",
        )
        assert "TOLERANCE BUG DETECTED" not in vc.message

    def test_model_post_init_close_status_skipped(self):
        """model_post_init skips when status is 'close'."""
        vc = ValueComparison(
            status="close", value1=1.0, value2=1.0, message="close"
        )
        assert "TOLERANCE BUG DETECTED" not in vc.message

    def test_model_post_init_detects_tolerance_bug(self):
        """model_post_init detects when floats are marked different but within tolerance."""
        # These are within 1e-5 tolerance but marked as 'different'
        vc = ValueComparison(
            status="different",
            value1=1.0,
            value2=1.000001,
            message="should be close",
        )
        assert "TOLERANCE BUG DETECTED" in vc.message

    def test_model_post_init_actually_different_floats(self):
        """model_post_init does not trigger for genuinely different floats."""
        vc = ValueComparison(
            status="different",
            value1=1.0,
            value2=2.0,
            message="big diff",
        )
        assert "TOLERANCE BUG DETECTED" not in vc.message


class TestCompoundDiff:
    """Tests for CompoundDiff model."""

    def test_compound_diff_basic(self):
        """CompoundDiff stores source_type and children."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        cd = CompoundDiff(source_type="list", children={"[0]": vc})
        assert cd.source_type == "list"
        assert "[0]" in cd.children
        assert not cd.truncated

    def test_compound_diff_truncated(self):
        """CompoundDiff can be marked as truncated."""
        cd = CompoundDiff(source_type="array", children={}, truncated=True)
        assert cd.truncated

    def test_compound_diff_with_warnings(self):
        """CompoundDiff can have structural warnings."""
        cd = CompoundDiff(
            source_type="dataframe",
            children={},
            warnings=["columns differ"],
        )
        assert "columns differ" in cd.warnings


class TestMemoryCheckpointDiffResult:
    """Tests for MemoryCheckpointDiffResult."""

    def test_empty_result(self):
        """Empty MemoryCheckpointDiffResult is falsy."""
        result = MemoryCheckpointDiffResult()
        assert not result
        assert len(result) == 0

    def test_bool_true(self):
        """Non-empty MemoryCheckpointDiffResult is truthy."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        assert result
        assert len(result) == 1

    def test_contains(self):
        """__contains__ checks for variable names."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        assert "x" in result
        assert "y" not in result

    def test_getitem(self):
        """__getitem__ returns diff tree for a variable."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        assert result["x"] == vc

    def test_setitem(self):
        """__setitem__ sets diff tree for a variable."""
        result = MemoryCheckpointDiffResult()
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result["x"] = vc
        assert "x" in result

    def test_iter(self):
        """__iter__ iterates over variable names."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"a": vc, "b": vc})
        assert set(result) == {"a", "b"}

    def test_eq_with_dict(self):
        """__eq__ allows comparison with plain dicts."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        assert result == {"x": vc}

    def test_eq_with_other_result(self):
        """__eq__ allows comparison with another MemoryCheckpointDiffResult."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result1 = MemoryCheckpointDiffResult(differences={"x": vc})
        result2 = MemoryCheckpointDiffResult(differences={"x": vc})
        assert result1 == result2

    def test_eq_with_unrelated_type(self):
        """__eq__ returns False for unrelated types."""
        result = MemoryCheckpointDiffResult()
        assert result != "string"
        assert result != 42

    def test_keys_values_items(self):
        """keys(), values(), items() delegate to differences dict."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": vc, "y": vc})
        assert set(result.keys()) == {"x", "y"}
        assert list(result.values()) == [vc, vc]
        items = list(result.items())
        assert len(items) == 2
        assert dict(items) == {"x": vc, "y": vc}

    def test_get(self):
        """get() returns diff tree or default."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        assert result.get("x") == vc
        assert result.get("y") is None
        assert result.get("y", "default") == "default"

    def test_close_only(self):
        """close_only() filters to only 'close' status comparisons."""
        vc_diff = ValueComparison(status="different", value1=1, value2=2, message="diff")
        vc_close = ValueComparison(status="close", value1=1.0, value2=1.001, message="close")
        result = MemoryCheckpointDiffResult(differences={"x": vc_diff, "y": vc_close})
        close_result = result.close_only()
        assert "x" not in close_result
        assert "y" in close_result

    def test_different_only(self):
        """different_only() filters to only 'different' status comparisons."""
        vc_diff = ValueComparison(status="different", value1=1, value2=2, message="diff")
        vc_close = ValueComparison(status="close", value1=1.0, value2=1.001, message="close")
        result = MemoryCheckpointDiffResult(differences={"x": vc_diff, "y": vc_close})
        diff_result = result.different_only()
        assert "x" in diff_result
        assert "y" not in diff_result

    def test_close_only_with_compound_diff(self):
        """close_only handles CompoundDiff nodes with mixed children."""
        vc_diff = ValueComparison(status="different", value1=1, value2=2, message="diff")
        vc_close = ValueComparison(status="close", value1=1.0, value2=1.001, message="close")
        cd = CompoundDiff(
            source_type="list",
            children={"[0]": vc_diff, "[1]": vc_close},
        )
        result = MemoryCheckpointDiffResult(differences={"data": cd})
        close = result.close_only()
        assert "data" in close
        # The compound diff should only have the close child
        node = close["data"]
        assert isinstance(node, CompoundDiff)
        assert "[1]" in node.children
        assert "[0]" not in node.children

    def test_filter_empty_compound_removed(self):
        """Filtering removes CompoundDiff nodes that become empty."""
        vc_diff = ValueComparison(status="different", value1=1, value2=2, message="diff")
        cd = CompoundDiff(source_type="list", children={"[0]": vc_diff})
        result = MemoryCheckpointDiffResult(differences={"data": cd})
        close = result.close_only()
        assert "data" not in close

    def test_filter_with_legacy_dict_node(self):
        """Filtering handles legacy dict nodes."""
        vc_diff = ValueComparison(status="different", value1=1, value2=2, message="diff")
        vc_close = ValueComparison(status="close", value1=1.0, value2=1.001, message="close")
        result = MemoryCheckpointDiffResult(
            differences={"x": {"sub1": vc_diff, "sub2": vc_close}}
        )
        close = result.close_only()
        assert "x" in close
        node = close["x"]
        assert "sub2" in node
        assert "sub1" not in node

    def test_filter_removes_empty_dict(self):
        """Filtering removes dict nodes that become empty."""
        vc_diff = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(
            differences={"x": {"sub": vc_diff}}
        )
        close = result.close_only()
        assert "x" not in close

    def test_filter_unknown_node_type(self):
        """Filtering passes through unknown node types."""
        result = MemoryCheckpointDiffResult(differences={"x": "some_string"})
        # Should not crash
        filtered = result.close_only()
        # Unknown node type gets passed through as-is
        assert "x" in filtered

    def test_convert_dicts_to_comparisons_validator(self):
        """Field validator converts dicts to ValueComparison during deserialization."""
        data = {
            "differences": {
                "x": {
                    "status": "different",
                    "value1": 1,
                    "value2": 2,
                    "message": "1 vs 2",
                }
            }
        }
        result = MemoryCheckpointDiffResult(**data)
        assert isinstance(result["x"], ValueComparison)

    def test_convert_dicts_to_comparisons_nested(self):
        """Field validator handles nested dicts that are not ValueComparison."""
        data = {
            "differences": {
                "x": {
                    "sub_key": {
                        "status": "different",
                        "value1": 1,
                        "value2": 2,
                        "message": "1 vs 2",
                    }
                }
            }
        }
        result = MemoryCheckpointDiffResult(**data)
        assert isinstance(result["x"]["sub_key"], ValueComparison)


class TestSerializeDiffResult:
    """Tests for serialize_diff_result function."""

    def test_serialize_value_comparison(self):
        """Serialize a ValueComparison leaf node."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="1 vs 2")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        serialized = serialize_diff_result(result)
        assert serialized["x"]["type"] == "comparison"
        assert serialized["x"]["status"] == "different"
        assert serialized["x"]["message"] == "1 vs 2"

    def test_serialize_compound_diff(self):
        """Serialize a CompoundDiff node."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        cd = CompoundDiff(source_type="list", children={"[0]": vc}, truncated=True)
        result = MemoryCheckpointDiffResult(differences={"data": cd})
        serialized = serialize_diff_result(result)
        assert serialized["data"]["type"] == "compound"
        assert serialized["data"]["source_type"] == "list"
        assert serialized["data"]["truncated"] is True
        assert "[0]" in serialized["data"]["children"]

    def test_serialize_legacy_dict(self):
        """Serialize legacy dict nodes."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": {"sub": vc}})
        serialized = serialize_diff_result(result)
        assert "sub" in serialized["x"]
        assert serialized["x"]["sub"]["type"] == "comparison"

    def test_serialize_unknown_node(self):
        """Serialize unknown node types gracefully."""
        result = MemoryCheckpointDiffResult(differences={"x": 42})
        serialized = serialize_diff_result(result)
        assert serialized["x"]["type"] == "unknown"

    def test_serialize_empty(self):
        """Serialize empty diff result."""
        result = MemoryCheckpointDiffResult()
        serialized = serialize_diff_result(result)
        assert serialized == {}


class TestFormatDiffAsMarkdown:
    """Tests for format_diff_as_markdown function."""

    def test_empty_diff(self):
        """Empty diff result shows 'No Differences Found'."""
        result = MemoryCheckpointDiffResult()
        md = format_diff_as_markdown(result)
        assert "No Differences Found" in md

    def test_single_value_comparison(self):
        """Single ValueComparison formatted as markdown bullet."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="1 vs 2")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        md = format_diff_as_markdown(result)
        assert "**x**" in md
        assert "1 vs 2" in md
        assert "Differences Found" in md

    def test_close_value_indicator(self):
        """Close value comparisons show *(close)* indicator."""
        vc = ValueComparison(status="close", value1=1.0, value2=1.001, message="close")
        result = MemoryCheckpointDiffResult(differences={"x": vc})
        md = format_diff_as_markdown(result)
        assert "*(close)*" in md

    def test_compound_diff_format(self):
        """CompoundDiff children are formatted with paths."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        cd = CompoundDiff(source_type="list", children={"[0]": vc})
        result = MemoryCheckpointDiffResult(differences={"data": cd})
        md = format_diff_as_markdown(result)
        assert "data[0]" in md

    def test_compound_diff_truncated(self):
        """Truncated CompoundDiff shows truncation message."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        cd = CompoundDiff(source_type="list", children={"[0]": vc}, truncated=True)
        result = MemoryCheckpointDiffResult(differences={"data": cd})
        md = format_diff_as_markdown(result)
        assert "truncated" in md

    def test_legacy_dict_format(self):
        """Legacy dict nodes are formatted with paths."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"x": {"[0]": vc}})
        md = format_diff_as_markdown(result)
        assert "x[0]" in md

    def test_legacy_dict_truncated_marker(self):
        """Legacy dict _truncated marker is formatted."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="truncated info")
        result = MemoryCheckpointDiffResult(
            differences={"x": {"_truncated": vc}}
        )
        md = format_diff_as_markdown(result)
        assert "truncated info" in md

    def test_sorted_output(self):
        """Variables are sorted alphabetically in output."""
        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        result = MemoryCheckpointDiffResult(differences={"z": vc, "a": vc, "m": vc})
        md = format_diff_as_markdown(result)
        # Find positions
        pos_a = md.index("**a**")
        pos_m = md.index("**m**")
        pos_z = md.index("**z**")
        assert pos_a < pos_m < pos_z


class TestExecutionError:
    """Tests for ExecutionError model."""

    def test_execution_error_basic(self):
        """ExecutionError stores error details."""
        err = ExecutionError(
            error_type="ValueError",
            error_message="invalid value",
            traceback="Traceback...",
        )
        assert err.error_type == "ValueError"
        assert err.error_message == "invalid value"
        assert err.code_snippet is None

    def test_execution_error_with_snippet(self):
        """ExecutionError with code_snippet."""
        err = ExecutionError(
            error_type="TypeError",
            error_message="wrong type",
            traceback="Traceback...",
            code_snippet="x = 1 + 'a'",
        )
        assert err.code_snippet == "x = 1 + 'a'"


class TestTestCodeResults:
    """Tests for TestCodeResult discriminated union types."""

    def test_success_result(self):
        """TestCodeSuccess stores diff and timing info."""
        result = TestCodeSuccess(
            diff=MemoryCheckpointDiffResult(),
            original_duration=1.5,
            modified_duration=0.5,
            speedup=3.0,
        )
        assert result.status == "success"
        assert result.speedup == 3.0

    def test_original_crash(self):
        """TestCodeOriginalCrash stores error details."""
        err = ExecutionError(
            error_type="ValueError",
            error_message="bad",
            traceback="Traceback...",
        )
        result = TestCodeOriginalCrash(error=err)
        assert result.status == "original_crash"
        assert result.original_duration is None

    def test_modified_crash(self):
        """TestCodeModifiedCrash stores error and timing."""
        err = ExecutionError(
            error_type="RuntimeError",
            error_message="failed",
            traceback="Traceback...",
        )
        result = TestCodeModifiedCrash(
            error=err,
            original_duration=1.0,
        )
        assert result.status == "modified_crash"
        assert result.original_duration == 1.0
        assert result.modified_duration is None
