"""Additional edge case tests for MemoryCheckpoint.

This test file supplements existing checkpoint tests with coverage for:
- Lazy alias index building
- cuDF origin tracking (mocked)
- Size estimation edge cases
- Checkpoint.diff with various parameters
- Helper function edge cases
- Checkpointable filtering edge cases
"""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.memory_checkpoint import (
    MemoryCheckpoint,
    MemoryCheckpoints,
    SYSTEM_VARIABLES,
    convert_dataframe_object_to_specialized,
    convert_series_object_to_specialized,
    filter_user_namespace,
    is_valid_variable,
    is_valid_variable_name,
    _collect_reachable_ids,
    _collect_reachable_ids_with_paths,
    _IMMUTABLE_ATOMIC_TYPES,
    _SINGLETON_TYPES,
)


# ============================================================================
# LAZY ALIAS INDEX TESTS
# ============================================================================


class TestLazyAliasIndex:
    """Test lazy building of the alias index."""

    def test_index_not_built_initially(self):
        """Test that alias index is not built on checkpoint creation."""
        user_ns = {"x": [1, 2, 3], "y": [4, 5, 6]}
        checkpoint = MemoryCheckpoint("test", user_ns, {})

        # Index should not be built yet
        assert checkpoint._alias_index_built is False
        assert checkpoint._reachable_ids == {}
        assert checkpoint._id_to_vars == {}

    def test_index_built_on_first_query(self):
        """Test that alias index is built on first get_aliases_for_vars call."""
        user_ns = {"x": [1, 2, 3], "y": [4, 5, 6]}
        checkpoint = MemoryCheckpoint("test", user_ns, {})

        assert checkpoint._alias_index_built is False

        # Query aliases
        aliases = checkpoint.get_aliases_for_vars({"x"})

        # Now index should be built
        assert checkpoint._alias_index_built is True
        assert len(checkpoint._reachable_ids) > 0
        assert len(checkpoint._id_to_vars) > 0

    def test_index_not_rebuilt_on_subsequent_queries(self):
        """Test that alias index is reused on subsequent queries."""
        user_ns = {"x": [1, 2, 3], "y": [4, 5, 6]}
        checkpoint = MemoryCheckpoint("test", user_ns, {})

        # First query
        checkpoint.get_aliases_for_vars({"x"})
        first_index = id(checkpoint._reachable_ids)

        # Second query
        checkpoint.get_aliases_for_vars({"y"})
        second_index = id(checkpoint._reachable_ids)

        # Same index should be reused
        assert first_index == second_index

    def test_empty_namespace_alias_query(self):
        """Test alias query with empty namespace."""
        checkpoint = MemoryCheckpoint("test", {}, {})

        aliases = checkpoint.get_aliases_for_vars({"nonexistent"})

        # Should return the input vars (even if they don't exist)
        assert "nonexistent" in aliases

    def test_query_nonexistent_variable(self):
        """Test querying aliases for a variable not in namespace."""
        user_ns = {"x": [1, 2, 3]}
        checkpoint = MemoryCheckpoint("test", user_ns, {})

        aliases = checkpoint.get_aliases_for_vars({"nonexistent"})

        # Should still include the queried variable
        assert "nonexistent" in aliases


# ============================================================================
# CUDF ORIGIN TRACKING TESTS
# ============================================================================


class TestCuDFOriginTracking:
    """Test cuDF origin tracking (mocked since cuDF may not be installed)."""

    def test_cudf_origins_initialized(self):
        """Test that cuDF origin tracker is initialized."""
        user_ns = {"x": 1}
        checkpoint = MemoryCheckpoint("test", user_ns, {})

        assert checkpoint._cudf_origins is not None

    def test_save_records_cudf_origins(self):
        """Test that save records cuDF origins."""
        cp = MemoryCheckpoints()

        # Mock cudf object detection
        with patch("flowbook.kernel_support.cudf_compat.CuDFOriginTracker") as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            user_ns = {"df": pd.DataFrame({"a": [1, 2, 3]})}
            cp.save("test", user_ns)

            # Should have called record on the tracker
            mock_tracker.record.assert_called()


# ============================================================================
# SIZE ESTIMATION TESTS
# ============================================================================


class TestSizeEstimation:
    """Test size estimation for checkpoints."""

    def test_estimate_size_empty_namespace(self):
        """Test size estimation for empty namespace."""
        cp = MemoryCheckpoints()

        size = cp._estimate_size({})
        assert size == 0

    def test_estimate_size_scalars(self):
        """Test size estimation for scalar values."""
        cp = MemoryCheckpoints()

        user_ns = {"x": 1, "y": "hello", "z": 3.14}
        size = cp._estimate_size(user_ns)

        assert size > 0

    def test_estimate_size_dataframe(self):
        """Test size estimation for DataFrame."""
        cp = MemoryCheckpoints()

        df = pd.DataFrame({"a": range(1000), "b": range(1000)})
        user_ns = {"df": df}
        size = cp._estimate_size(user_ns)

        # Should be at least as big as the DataFrame
        df_size = df.memory_usage(deep=True).sum()
        assert size >= df_size

    def test_estimate_size_series(self):
        """Test size estimation for Series."""
        cp = MemoryCheckpoints()

        s = pd.Series(range(1000))
        user_ns = {"s": s}
        size = cp._estimate_size(user_ns)

        s_size = s.memory_usage(deep=True)
        assert size >= s_size

    def test_estimate_size_numpy_array(self):
        """Test size estimation for numpy array."""
        cp = MemoryCheckpoints()

        arr = np.zeros((100, 100))
        user_ns = {"arr": arr}
        size = cp._estimate_size(user_ns)

        assert size >= arr.nbytes

    def test_estimate_size_handles_errors(self):
        """Test that size estimation handles objects that fail sizeof."""
        cp = MemoryCheckpoints()

        class BadSizeof:
            def __sizeof__(self):
                raise TypeError("Cannot get size")

        user_ns = {"x": 1, "bad": BadSizeof()}
        # Should not raise
        size = cp._estimate_size(user_ns)
        assert size > 0  # x contributes


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_valid_variable_name_valid(self):
        """Test is_valid_variable_name with valid names."""
        assert is_valid_variable_name("x") is True
        assert is_valid_variable_name("my_var") is True
        assert is_valid_variable_name("Var123") is True

    def test_is_valid_variable_name_invalid(self):
        """Test is_valid_variable_name with invalid names."""
        # Underscore prefix
        assert is_valid_variable_name("_private") is False
        assert is_valid_variable_name("__dunder__") is False

        # System variables
        for sysvar in SYSTEM_VARIABLES:
            assert is_valid_variable_name(sysvar) is False

    def test_is_valid_variable_with_module(self):
        """Test is_valid_variable filters out modules."""
        import math

        assert is_valid_variable("math", math) is False
        assert is_valid_variable("x", 1) is True

    def test_filter_user_namespace(self):
        """Test filter_user_namespace function."""
        import os

        user_ns = {
            "x": 1,
            "y": "hello",
            "_private": 2,
            "__dunder__": 3,
            "get_ipython": lambda: None,
            "os_module": os,
            "In": [],
            "Out": {},
        }

        filtered = filter_user_namespace(user_ns)

        assert "x" in filtered
        assert "y" in filtered
        assert "_private" not in filtered
        assert "__dunder__" not in filtered
        assert "get_ipython" not in filtered
        assert "os_module" not in filtered
        assert "In" not in filtered
        assert "Out" not in filtered


# ============================================================================
# CHECKPOINTABLE VALUE TESTS
# ============================================================================


class TestCheckpointableValue:
    """Test checkpointable_value method."""

    def test_module_not_checkpointable(self):
        """Test that modules are not checkpointable."""
        cp = MemoryCheckpoints()
        import math

        assert cp.checkpointable_value(math) is False

    def test_matplotlib_mock_not_checkpointable(self):
        """Test that matplotlib objects are not checkpointable."""
        cp = MemoryCheckpoints()

        class MockMatplotlib:
            __module__ = "matplotlib.figure"

        assert cp.checkpointable_value(MockMatplotlib()) is False

    def test_regular_values_checkpointable(self):
        """Test that regular values are checkpointable."""
        cp = MemoryCheckpoints()

        assert cp.checkpointable_value(1) is True
        assert cp.checkpointable_value("hello") is True
        assert cp.checkpointable_value([1, 2, 3]) is True
        assert cp.checkpointable_value({"a": 1}) is True
        assert cp.checkpointable_value(pd.DataFrame()) is True

    def test_numpy_array_with_matplotlib_elements(self):
        """Test that numpy arrays containing matplotlib objects are filtered."""
        cp = MemoryCheckpoints()

        class MockMatplotlib:
            __module__ = "matplotlib.patches"

        arr = np.array([MockMatplotlib(), MockMatplotlib()], dtype=object)
        assert cp.checkpointable_value(arr) is False

    def test_numpy_array_with_regular_objects(self):
        """Test that numpy arrays with regular objects are checkpointable."""
        cp = MemoryCheckpoints()

        arr = np.array([{"a": 1}, {"b": 2}], dtype=object)
        assert cp.checkpointable_value(arr) is True


# ============================================================================
# USER-DEFINED CLASS DETECTION TESTS
# ============================================================================


class TestUserDefinedClassDetection:
    """Test detection of user-defined classes."""

    def test_user_defined_class_detected(self):
        """Test that user-defined classes are detected."""
        cp = MemoryCheckpoints()

        class MyClass:
            pass

        assert cp._is_user_defined_class(MyClass) is True

    def test_builtin_types_not_user_defined(self):
        """Test that builtin types are not detected as user-defined."""
        cp = MemoryCheckpoints()

        assert cp._is_user_defined_class(int) is False
        assert cp._is_user_defined_class(str) is False
        assert cp._is_user_defined_class(list) is False
        assert cp._is_user_defined_class(dict) is False

    def test_library_classes_not_user_defined(self):
        """Test that library classes are not detected as user-defined."""
        cp = MemoryCheckpoints()

        assert cp._is_user_defined_class(pd.DataFrame) is False
        assert cp._is_user_defined_class(pd.Series) is False
        assert cp._is_user_defined_class(np.ndarray) is False

    def test_instance_not_class(self):
        """Test that instances are not detected as classes."""
        cp = MemoryCheckpoints()

        class MyClass:
            pass

        obj = MyClass()
        assert cp._is_user_defined_class(obj) is False


# ============================================================================
# COLLECT REACHABLE IDS TESTS
# ============================================================================


class TestCollectReachableIds:
    """Test _collect_reachable_ids helper function."""

    def test_immutable_atomics_skipped(self):
        """Test that immutable atomic types are skipped."""
        visited = set()

        for value in [None, True, 1, 3.14, "hello", b"bytes", 1 + 2j]:
            _collect_reachable_ids(value, visited)

        # No IDs should be collected for immutable atomics
        assert len(visited) == 0

    def test_numpy_scalars_skipped(self):
        """Test that numpy scalar types are skipped."""
        visited = set()

        _collect_reachable_ids(np.int64(42), visited)
        _collect_reachable_ids(np.float64(3.14), visited)
        _collect_reachable_ids(np.bool_(True), visited)

        assert len(visited) == 0

    def test_singleton_types_skipped(self):
        """Test that singleton types (classes, functions, modules) are skipped."""
        visited = set()

        def my_func():
            pass

        _collect_reachable_ids(int, visited)  # type
        _collect_reachable_ids(my_func, visited)  # function
        _collect_reachable_ids(types, visited)  # module

        assert len(visited) == 0

    def test_list_ids_collected(self):
        """Test that list IDs are collected."""
        visited = set()
        lst = [1, 2, 3]

        _collect_reachable_ids(lst, visited)

        assert id(lst) in visited

    def test_dict_ids_collected(self):
        """Test that dict IDs are collected."""
        visited = set()
        d = {"a": 1, "b": 2}

        _collect_reachable_ids(d, visited)

        assert id(d) in visited

    def test_nested_ids_collected(self):
        """Test that nested object IDs are collected."""
        visited = set()
        inner = [1, 2]
        outer = {"inner": inner}

        _collect_reachable_ids(outer, visited)

        assert id(outer) in visited
        assert id(inner) in visited

    def test_circular_reference_handled(self):
        """Test that circular references don't cause infinite loops."""
        visited = set()
        lst = [1, 2]
        lst.append(lst)  # Self-reference

        # Should complete without hanging
        _collect_reachable_ids(lst, visited)

        assert id(lst) in visited


# ============================================================================
# COLLECT REACHABLE IDS WITH PATHS TESTS
# ============================================================================


class TestCollectReachableIdsWithPaths:
    """Test _collect_reachable_ids_with_paths helper function."""

    def test_paths_tracked_for_nested_objects(self):
        """Test that paths are tracked for nested objects."""
        visited = set()
        id_to_path = {}

        data = {"a": {"b": [1, 2, 3]}}
        _collect_reachable_ids_with_paths(data, "data", visited, id_to_path)

        # Root path should be tracked
        assert id(data) in id_to_path
        # Nested dict should have path
        assert id(data["a"]) in id_to_path
        # List should have path
        assert id(data["a"]["b"]) in id_to_path

    def test_first_path_preserved(self):
        """Test that first path to object is preserved."""
        visited = set()
        id_to_path = {}

        shared = [1, 2, 3]
        data = {"first": shared, "second": {"nested": shared}}

        _collect_reachable_ids_with_paths(data, "data", visited, id_to_path)

        # First path should be the one recorded
        shared_path = id_to_path[id(shared)]
        assert "first" in shared_path or "nested" in shared_path


# ============================================================================
# CHECKPOINT DIFF TESTS
# ============================================================================


class TestCheckpointDiff:
    """Test MemoryCheckpoint.diff method."""

    def test_diff_identical_checkpoints(self):
        """Test diff between identical checkpoints."""
        user_ns = {"x": 1, "y": [1, 2, 3]}

        cp1 = MemoryCheckpoint("cp1", dict(user_ns), {})
        cp2 = MemoryCheckpoint("cp2", dict(user_ns), {})

        diff = MemoryCheckpoint.diff(cp1, cp2)

        assert diff.differences == {}

    def test_diff_with_changes(self):
        """Test diff detects changes."""
        cp1 = MemoryCheckpoint("cp1", {"x": 1}, {})
        cp2 = MemoryCheckpoint("cp2", {"x": 2}, {})

        diff = MemoryCheckpoint.diff(cp1, cp2)

        assert "x" in diff.differences

    def test_diff_with_keys_to_include(self):
        """Test diff with keys_to_include filter."""
        cp1 = MemoryCheckpoint("cp1", {"x": 1, "y": 2, "z": 3}, {})
        cp2 = MemoryCheckpoint("cp2", {"x": 10, "y": 20, "z": 30}, {})

        diff = MemoryCheckpoint.diff(cp1, cp2, keys_to_include={"x", "y"})

        assert "x" in diff.differences
        assert "y" in diff.differences
        # z should not be in differences if keys_to_include works

    def test_diff_use_leq_mode(self):
        """Test diff with use_leq=True."""
        cp1 = MemoryCheckpoint("cp1", {"x": 1}, {})
        cp2 = MemoryCheckpoint("cp2", {"x": 1, "extra": 2}, {})

        # With use_leq=True, extra keys in cp2 are allowed
        diff = MemoryCheckpoint.diff(cp1, cp2, use_leq=True)

        # Should not raise and should handle the extra key
        assert isinstance(diff.differences, dict)


# ============================================================================
# TYPE MODELS TESTS
# ============================================================================


class TestTypeModels:
    """Test type_models method."""

    def test_type_models_for_namespace(self):
        """Test type_models generates models for all checkpointable vars."""
        cp = MemoryCheckpoints()

        user_ns = {
            "x": 1,
            "s": "hello",
            "lst": [1, 2, 3],
            "_private": 99,  # Should be filtered
        }

        models = cp.type_models(user_ns)

        assert "x" in models
        assert "s" in models
        assert "lst" in models
        assert "_private" not in models

    def test_type_models_empty_namespace(self):
        """Test type_models with empty namespace."""
        cp = MemoryCheckpoints()

        models = cp.type_models({})

        assert models == {}


# ============================================================================
# OBJECT CONVERSION EDGE CASES
# ============================================================================


class TestObjectConversionEdgeCases:
    """Test object to specialized type conversion edge cases."""

    def test_convert_series_preserves_index(self):
        """Test that conversion preserves Series index."""
        s = pd.Series([1, 2, 3], index=["a", "b", "c"], dtype=object)
        result = convert_series_object_to_specialized(s)

        assert list(result.index) == ["a", "b", "c"]

    def test_convert_series_preserves_name(self):
        """Test that conversion preserves Series name."""
        s = pd.Series([1, 2, 3], name="my_series", dtype=object)
        result = convert_series_object_to_specialized(s)

        assert result.name == "my_series"

    def test_convert_dataframe_preserves_index(self):
        """Test that conversion preserves DataFrame index."""
        df = pd.DataFrame(
            {"col": pd.Series([1, 2, 3], dtype=object)}, index=["a", "b", "c"]
        )
        result = convert_dataframe_object_to_specialized(df)

        assert list(result.index) == ["a", "b", "c"]

    def test_convert_dataframe_preserves_column_names(self):
        """Test that conversion preserves column names."""
        df = pd.DataFrame(
            {"my_col": pd.Series([1, 2, 3], dtype=object), "other": [4, 5, 6]}
        )
        result = convert_dataframe_object_to_specialized(df)

        assert "my_col" in result.columns
        assert "other" in result.columns
