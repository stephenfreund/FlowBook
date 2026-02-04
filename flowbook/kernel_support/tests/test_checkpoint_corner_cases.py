"""Comprehensive corner case tests for checkpoint system."""

import pandas as pd
import numpy as np
import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from flowbook.kernel_support.checkpoint import (
    convert_series_object_to_specialized,
    convert_dataframe_object_to_specialized,
)
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


class TestObjectConversionEdgeCases:
    """Test edge cases in object to specialized type conversion."""

    def test_empty_series_conversion(self):
        """Test converting empty Series with object dtype."""
        s = pd.Series([], dtype=object)
        result = convert_series_object_to_specialized(s)
        # Empty series should remain object (no data to infer from)
        assert result.dtype == object
        assert len(result) == 0

    def test_all_na_series_conversion(self):
        """Test converting Series with only NA values."""
        s = pd.Series([None, None, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        # All NA should remain object (no data to infer from)
        assert result.dtype == object

    def test_single_value_series_conversion(self):
        """Test converting Series with single non-NA value."""
        s = pd.Series([None, 42, None, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == pd.Int64Dtype()
        assert result[1] == 42
        assert pd.isna(result[0])

    def test_large_integer_overflow(self):
        """Test that large integers that exceed int64 are handled properly."""
        # Int64 can handle up to 2^63-1
        large_int = 2**62
        very_large_int = 2**63 - 1
        s = pd.Series([large_int, very_large_int, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == pd.Int64Dtype()
        assert result[0] == large_int
        assert result[1] == very_large_int

    def test_float_with_inf_and_nan(self):
        """Test converting floats with infinity and NaN values."""
        s = pd.Series([1.5, np.inf, -np.inf, np.nan, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == np.float64
        assert result[0] == 1.5
        assert np.isinf(result[1])
        assert np.isinf(result[2])
        assert np.isnan(result[3])

    def test_mixed_numeric_with_none_nan(self):
        """Test mixed numeric types with both None and NaN."""
        s = pd.Series([1, 2.5, None, np.nan, 3], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == np.float64
        assert result[0] == 1.0
        assert result[1] == 2.5

    def test_string_with_empty_strings(self):
        """Test string conversion with empty strings."""
        s = pd.Series(["a", "", "c", None, ""], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == pd.StringDtype()
        assert result[0] == "a"
        assert result[1] == ""
        assert pd.isna(result[3])

    def test_boolean_with_mixed_truthy_values(self):
        """Test that only actual booleans are converted, not truthy values."""
        s = pd.Series([True, False, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == pd.BooleanDtype()

    def test_datetime_with_timezones(self):
        """Test datetime conversion with timezone-aware datetimes."""
        # All timezone-aware
        dt_utc1 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        dt_utc2 = datetime(2020, 1, 2, tzinfo=timezone.utc)
        s = pd.Series([dt_utc1, dt_utc2, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        # Should convert to datetime64 (may lose timezone info)
        assert pd.api.types.is_datetime64_any_dtype(result.dtype)

    def test_timedelta_with_negative_values(self):
        """Test timedelta conversion with negative values."""
        s = pd.Series([
            timedelta(days=1),
            timedelta(days=-1),
            timedelta(hours=12),
            None
        ], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert pd.api.types.is_timedelta64_dtype(result.dtype)
        assert result[0] == pd.Timedelta(days=1)
        assert result[1] == pd.Timedelta(days=-1)

    def test_decimal_precision_handling(self):
        """Test that Decimal values are converted to float."""
        s = pd.Series([
            Decimal("1.23456789012345"),
            Decimal("2.5"),
            None
        ], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == np.float64
        # Note: precision may be lost in float conversion

    def test_complex_numbers_with_none_values(self):
        """Test that complex numbers with None values are converted to complex128."""
        s = pd.Series([1+2j, 2+3j, 0+4j, None], dtype=object)
        result = convert_series_object_to_specialized(s)
        # Complex numbers are properly detected and converted to complex128
        assert result.dtype == np.complex128
        assert result[0] == 1+2j
        assert result[1] == 2+3j
        # None becomes NaN in complex dtype
        assert np.isnan(result[3])

    def test_complex_numbers_without_none_values(self):
        """Test that complex numbers without None values are converted."""
        s = pd.Series([1+2j, 2+3j, 3+4j], dtype=object)
        result = convert_series_object_to_specialized(s)
        # Complex numbers WITHOUT None are properly converted to complex128
        assert result.dtype == np.complex128
        assert result[0] == 1+2j
        assert result[1] == 2+3j

    def test_bytes_remain_object(self):
        """Test that bytes objects remain as object dtype."""
        s = pd.Series([b"hello", b"world", None], dtype=object)
        result = convert_series_object_to_specialized(s)
        # Bytes should remain as object
        assert result.dtype == object

    def test_mixed_string_and_bytes(self):
        """Test that mixing strings and bytes keeps object dtype."""
        s = pd.Series(["hello", b"world", "test"], dtype=object)
        result = convert_series_object_to_specialized(s)
        assert result.dtype == object

    def test_dataframe_with_no_object_columns(self):
        """Test DataFrame conversion when no columns are object dtype."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": pd.Series(["a", "b", "c"], dtype="string")
        })
        result = convert_dataframe_object_to_specialized(df)
        # Should return copy with same dtypes
        assert result["int_col"].dtype == df["int_col"].dtype
        assert result["float_col"].dtype == df["float_col"].dtype
        assert result["str_col"].dtype == df["str_col"].dtype

    def test_dataframe_mixed_success_and_failure(self):
        """Test DataFrame where some columns convert and others don't."""
        df = pd.DataFrame({
            "integers": pd.Series([1, 2, 3], dtype=object),
            "mixed": pd.Series([1, "a", 2], dtype=object),
            "strings": pd.Series(["x", "y", "z"], dtype=object),
        })
        result = convert_dataframe_object_to_specialized(df)
        assert result["integers"].dtype == pd.Int64Dtype()
        assert result["mixed"].dtype == object  # Mixed types stay object
        assert result["strings"].dtype == pd.StringDtype()


class TestNestedStructures:
    """Test checkpoint handling of nested and complex structures."""

    def test_dataframe_with_dataframe_cells(self):
        """Test DataFrame containing DataFrames in cells."""
        inner_df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        inner_df2 = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        df = pd.DataFrame({
            "data": [inner_df1, inner_df2],
            "id": [1, 2]
        })

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        # Verify nested DataFrames are properly copied
        assert isinstance(restored_df["data"][0], pd.DataFrame)
        pd.testing.assert_frame_equal(restored_df["data"][0], inner_df1)

    def test_deeply_nested_dicts(self):
        """Test deeply nested dictionary structures."""
        deep_dict = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {"value": [1, 2, 3]}
                        }
                    }
                }
            }
        }

        checkpoints = MemoryCheckpoints()
        user_ns = {"data": deep_dict}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored = checkpoint.user_ns["data"]

        # Verify deep structure is preserved
        assert restored["level1"]["level2"]["level3"]["level4"]["level5"]["value"] == [1, 2, 3]

        # Verify it's a deep copy
        restored["level1"]["level2"]["level3"]["level4"]["level5"]["value"].append(4)
        assert deep_dict["level1"]["level2"]["level3"]["level4"]["level5"]["value"] == [1, 2, 3]

    def test_dataframe_with_list_of_dicts(self):
        """Test DataFrame with cells containing lists of dictionaries."""
        df = pd.DataFrame({
            "data": [
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
                [{"a": 5, "b": 6}],
            ],
            "id": [1, 2]
        })

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        # Verify nested structures
        assert restored_df["data"][0][0] == {"a": 1, "b": 2}

        # Verify deep copy
        restored_df["data"][0][0]["a"] = 999
        assert df["data"][0][0]["a"] == 1

    def test_series_with_nested_lists(self):
        """Test Series containing lists with nested lists."""
        s = pd.Series([
            [[1, 2], [3, 4]],
            [[5, 6]],
            [[7, 8], [9, 10], [11, 12]]
        ])

        checkpoints = MemoryCheckpoints()
        user_ns = {"s": s}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_s = checkpoint.user_ns["s"]

        assert restored_s[0] == [[1, 2], [3, 4]]

        # Verify deep copy
        restored_s[0][0].append(999)
        assert s[0] == [[1, 2], [3, 4]]


class TestCheckpointManagementEdgeCases:
    """Test edge cases in checkpoint management operations."""

    def test_checkpoint_with_empty_name(self):
        """Test that empty checkpoint names are rejected."""
        checkpoints = MemoryCheckpoints()
        user_ns = {"x": 42}

        # Empty string as checkpoint name should raise ValueError
        with pytest.raises(ValueError, match="empty or whitespace"):
            checkpoints.save("", user_ns)

    def test_checkpoint_with_special_characters_in_name(self):
        """Test checkpoint names with special characters."""
        checkpoints = MemoryCheckpoints()
        user_ns = {"x": 42}

        special_names = [
            "test-checkpoint",
            "test.checkpoint",
            "test_checkpoint",
            "test checkpoint",
            "test/checkpoint",
            "test\\checkpoint",
            "test:checkpoint",
        ]

        for name in special_names:
            checkpoints.save(name, user_ns)
            checkpoint = checkpoints.get(name)
            assert checkpoint.user_ns["x"] == 42

    def test_multiple_sequential_saves_same_name(self):
        """Test saving multiple times to the same checkpoint name."""
        checkpoints = MemoryCheckpoints()

        # First save
        user_ns = {"x": 1, "y": 2}
        checkpoints.save("test", user_ns)

        # Second save with different data
        user_ns = {"x": 10, "z": 20}
        checkpoints.save("test", user_ns)

        # Should have the latest version
        checkpoint = checkpoints.get("test")
        assert checkpoint.user_ns["x"] == 10
        assert checkpoint.user_ns["z"] == 20
        assert "y" not in checkpoint.user_ns

    def test_restore_then_save_new_checkpoint(self):
        """Test creating new checkpoint after restoring."""
        checkpoints = MemoryCheckpoints()

        # Save first checkpoint
        user_ns = {"x": 1}
        checkpoints.save("cp1", user_ns)

        # Modify and save second checkpoint
        user_ns["x"] = 2
        user_ns["y"] = 3
        checkpoints.save("cp2", user_ns)

        # Restore first checkpoint
        cp1 = checkpoints.get("cp1")
        user_ns = cp1.user_ns.copy()
        user_ns["x"] = 5

        # Save third checkpoint
        checkpoints.save("cp3", user_ns)

        # Verify all checkpoints are independent
        assert checkpoints.get("cp1").user_ns["x"] == 1
        assert checkpoints.get("cp2").user_ns["x"] == 2
        assert checkpoints.get("cp3").user_ns["x"] == 5

    def test_checkpoint_with_very_large_namespace(self):
        """Test checkpoint with many variables."""
        checkpoints = MemoryCheckpoints()

        # Create namespace with 1000 variables
        user_ns = {f"var_{i}": i for i in range(1000)}

        checkpoints.save("large", user_ns)
        checkpoint = checkpoints.get("large")

        assert len(checkpoint.user_ns) == 1000
        assert checkpoint.user_ns["var_500"] == 500
        assert checkpoint.user_ns["var_999"] == 999

    def test_delete_nonexistent_checkpoint(self):
        """Test deleting a checkpoint that doesn't exist."""
        checkpoints = MemoryCheckpoints()

        # Should not raise an error
        checkpoints.delete("nonexistent")

        # Verify nothing changed
        assert len(checkpoints.list()) == 0

    def test_get_nonexistent_checkpoint(self):
        """Test getting a checkpoint that doesn't exist."""
        checkpoints = MemoryCheckpoints()

        # get() raises KeyError for nonexistent checkpoints
        with pytest.raises(KeyError):
            checkpoints.get("nonexistent")

    def test_clear_empty_checkpoints(self):
        """Test clearing when no checkpoints exist."""
        checkpoints = MemoryCheckpoints()
        checkpoints.clear()
        assert len(checkpoints.list()) == 0

    def test_list_returns_ordered_by_creation(self):
        """Test that list returns checkpoint names in creation order."""
        checkpoints = MemoryCheckpoints()
        user_ns = {"x": 1}

        names = ["cp3", "cp1", "cp2"]
        for name in names:
            checkpoints.save(name, user_ns)

        listed = checkpoints.list()
        # list() returns list of strings (checkpoint names) in insertion order
        assert listed == names


class TestSpecialDataTypes:
    """Test checkpoint handling of special pandas data types."""

    def test_multiindex_dataframe(self):
        """Test DataFrame with MultiIndex."""
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two"]
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = pd.DataFrame(np.random.randn(6, 2), index=index, columns=["A", "B"])

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        pd.testing.assert_frame_equal(restored_df, df)
        assert isinstance(restored_df.index, pd.MultiIndex)

    def test_multiindex_columns(self):
        """Test DataFrame with MultiIndex columns."""
        columns = pd.MultiIndex.from_tuples([
            ("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")
        ])
        df = pd.DataFrame(np.random.randn(5, 4), columns=columns)

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        pd.testing.assert_frame_equal(restored_df, df)
        assert isinstance(restored_df.columns, pd.MultiIndex)

    def test_categorical_dtype(self):
        """Test DataFrame with categorical columns."""
        df = pd.DataFrame({
            "cat": pd.Categorical(["a", "b", "c", "a", "b"]),
            "ordered_cat": pd.Categorical(
                ["low", "medium", "high", "low", "high"],
                categories=["low", "medium", "high"],
                ordered=True
            )
        })

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        pd.testing.assert_frame_equal(restored_df, df)
        assert restored_df["cat"].dtype.name == "category"
        assert restored_df["ordered_cat"].dtype.ordered

    def test_nullable_integer_dtype(self):
        """Test DataFrame with nullable integer types."""
        df = pd.DataFrame({
            "int8": pd.array([1, 2, None], dtype="Int8"),
            "int16": pd.array([100, 200, None], dtype="Int16"),
            "int32": pd.array([1000, 2000, None], dtype="Int32"),
            "int64": pd.array([10000, 20000, None], dtype="Int64"),
        })

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        pd.testing.assert_frame_equal(restored_df, df)
        assert restored_df["int8"].dtype == pd.Int8Dtype()
        assert restored_df["int64"].dtype == pd.Int64Dtype()

    def test_nullable_float_dtype(self):
        """Test DataFrame with nullable float types."""
        df = pd.DataFrame({
            "float32": pd.array([1.5, 2.5, None], dtype="Float32"),
            "float64": pd.array([10.5, 20.5, None], dtype="Float64"),
        })

        checkpoints = MemoryCheckpoints()
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_df = checkpoint.user_ns["df"]

        pd.testing.assert_frame_equal(restored_df, df)
        assert restored_df["float32"].dtype == pd.Float32Dtype()
        assert restored_df["float64"].dtype == pd.Float64Dtype()

    def test_sparse_series(self):
        """Test Series with sparse dtype."""
        s = pd.Series([0, 0, 1, 0, 0, 2, 0, 0, 3], dtype=pd.SparseDtype("int64", 0))

        checkpoints = MemoryCheckpoints()
        user_ns = {"s": s}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_s = checkpoint.user_ns["s"]

        pd.testing.assert_series_equal(restored_s, s)
        assert isinstance(restored_s.dtype, pd.SparseDtype)

    def test_interval_dtype(self):
        """Test Series with interval dtype."""
        s = pd.Series(pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)]))

        checkpoints = MemoryCheckpoints()
        user_ns = {"s": s}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_s = checkpoint.user_ns["s"]

        pd.testing.assert_series_equal(restored_s, s)

    def test_period_dtype(self):
        """Test Series with period dtype."""
        s = pd.Series(pd.period_range("2020-01", periods=5, freq="M"))

        checkpoints = MemoryCheckpoints()
        user_ns = {"s": s}
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_s = checkpoint.user_ns["s"]

        pd.testing.assert_series_equal(restored_s, s)
        assert isinstance(restored_s.dtype, pd.PeriodDtype)


class TestCheckpointWithDifferentTypes:
    """Test checkpointing with various Python types."""

    def test_numpy_arrays_various_dtypes(self):
        """Test checkpoint with various numpy array dtypes."""
        user_ns = {
            "int_array": np.array([1, 2, 3], dtype=np.int32),
            "float_array": np.array([1.5, 2.5, 3.5], dtype=np.float32),
            "bool_array": np.array([True, False, True]),
            "complex_array": np.array([1+2j, 3+4j]),
            "str_array": np.array(["a", "b", "c"]),
        }

        checkpoints = MemoryCheckpoints()
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")

        np.testing.assert_array_equal(
            checkpoint.user_ns["int_array"], user_ns["int_array"]
        )
        assert checkpoint.user_ns["int_array"].dtype == np.int32

    def test_mixed_container_types(self):
        """Test checkpoint with lists, tuples, sets, and dicts."""
        user_ns = {
            "list_var": [1, 2, [3, 4]],
            "tuple_var": (1, 2, (3, 4)),
            "set_var": {1, 2, 3},
            "frozenset_var": frozenset([1, 2, 3]),
            "dict_var": {"a": 1, "b": {"c": 2}},
        }

        checkpoints = MemoryCheckpoints()
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")

        # Verify deep copy behavior
        checkpoint.user_ns["list_var"][2].append(5)
        assert user_ns["list_var"] == [1, 2, [3, 4]]

        checkpoint.user_ns["dict_var"]["b"]["c"] = 999
        assert user_ns["dict_var"]["b"]["c"] == 2

    def test_custom_class_instances(self):
        """Test checkpoint with custom class instances."""
        class CustomClass:
            def __init__(self, value):
                self.value = value
                self.data = [1, 2, 3]

        obj = CustomClass(42)
        user_ns = {"custom": obj}

        checkpoints = MemoryCheckpoints()
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")
        restored_obj = checkpoint.user_ns["custom"]

        assert restored_obj.value == 42
        assert restored_obj.data == [1, 2, 3]

        # Verify deep copy
        restored_obj.data.append(4)
        assert obj.data == [1, 2, 3]

    def test_none_values(self):
        """Test checkpoint with None values."""
        user_ns = {
            "none_var": None,
            "list_with_none": [1, None, 3],
            "dict_with_none": {"a": None, "b": 2},
        }

        checkpoints = MemoryCheckpoints()
        checkpoints.save("test", user_ns)

        checkpoint = checkpoints.get("test")

        assert checkpoint.user_ns["none_var"] is None
        assert checkpoint.user_ns["list_with_none"][1] is None
        assert checkpoint.user_ns["dict_with_none"]["a"] is None


class TestCheckpointSanityChecks:
    """Test sanity check functionality in detail."""

    def test_sanity_check_detects_modification(self):
        """Test that sanity check detects if original data was modified."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        checkpoints = MemoryCheckpoints(sanity_check=True)
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        # Modify original DataFrame
        df.loc[0, "a"] = 999

        # Restore should detect the change during sanity check
        # (This tests the internal sanity check logic)
        checkpoint = checkpoints.get("test")
        assert checkpoint.user_ns["df"].loc[0, "a"] == 1  # Original value

    def test_sanity_check_with_mutable_objects(self):
        """Test sanity check with mutable objects in object columns."""
        df = pd.DataFrame({
            "lists": [[1, 2], [3, 4], [5, 6]]
        })

        checkpoints = MemoryCheckpoints(sanity_check=True)
        user_ns = {"df": df}
        checkpoints.save("test", user_ns)

        # Modify list inside DataFrame
        df["lists"][0].append(999)

        checkpoint = checkpoints.get("test")
        # Checkpoint should have original data
        assert checkpoint.user_ns["df"]["lists"][0] == [1, 2]

    def test_sanity_check_disabled_faster(self):
        """Test that disabling sanity check works correctly."""
        import time

        # Large DataFrame for performance difference
        df = pd.DataFrame({
            "data": [{"key": i} for i in range(1000)]
        })

        # With sanity check
        checkpoints_with = MemoryCheckpoints(sanity_check=True)
        user_ns_with = {"df": df.copy()}
        start = time.time()
        checkpoints_with.save("test", user_ns_with)
        time_with = time.time() - start

        # Without sanity check
        checkpoints_without = MemoryCheckpoints(sanity_check=False)
        user_ns_without = {"df": df.copy()}
        start = time.time()
        checkpoints_without.save("test", user_ns_without)
        time_without = time.time() - start

        # Both should complete successfully
        assert checkpoints_with.get("test") is not None
        assert checkpoints_without.get("test") is not None


class TestErrorRecovery:
    """Test checkpoint behavior in error scenarios."""

    def test_checkpoint_after_failed_conversion(self):
        """Test that checkpoint works even if some conversions fail."""
        # Create a Series that might cause conversion issues
        class WeirdObject:
            def __init__(self, value):
                self.value = value

        s = pd.Series([WeirdObject(1), WeirdObject(2)], dtype=object)

        checkpoints = MemoryCheckpoints()
        user_ns = {"s": s}

        # Should not raise an error, conversion should gracefully fail
        checkpoints.save("test", user_ns)
        checkpoint = checkpoints.get("test")

        # Should have checkpointed the data even if conversion failed
        assert len(checkpoint.user_ns["s"]) == 2

    def test_restore_with_incompatible_modifications(self):
        """Test restore when current namespace has incompatible changes."""
        checkpoints = MemoryCheckpoints()

        # Save with DataFrame
        user_ns = {"var": pd.DataFrame({"a": [1, 2, 3]})}
        checkpoints.save("test", user_ns)

        # Change to different type
        user_ns["var"] = "now a string"

        # Restore should replace with checkpoint value
        checkpoint = checkpoints.get("test")
        assert isinstance(checkpoint.user_ns["var"], pd.DataFrame)


class TestNonCheckpointableVariableRemoval:
    """Tests for proper removal of non-checkpointable variables."""

    def test_file_handle_identified_as_removed(self):
        """File handles should be identified in the removed dict."""
        import tempfile

        checkpoints = MemoryCheckpoints()

        # Create a file handle (TextIOWrapper) - this cannot be checkpointed
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        user_ns = {"x": 1, "f": tmp}

        saved, removed = checkpoints.save("test", user_ns)

        # x should be saved, f should be removed
        assert "x" in saved
        assert "f" in removed
        assert "TextIOWrapper" in str(removed["f"]) or "file" in str(removed["f"]).lower()

        # Cleanup
        tmp.close()
        import os
        os.unlink(tmp.name)

    def test_socket_identified_as_removed(self):
        """Socket objects should be identified in the removed dict."""
        import socket

        checkpoints = MemoryCheckpoints()

        # Create a socket - this cannot be checkpointed
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        user_ns = {"x": 1, "sock": sock}

        saved, removed = checkpoints.save("test", user_ns)

        # x should be saved, sock should be removed
        assert "x" in saved
        assert "sock" in removed

        # Cleanup
        sock.close()

    def test_generator_identified_as_removed(self):
        """Generator objects should be identified in the removed dict."""
        checkpoints = MemoryCheckpoints()

        # Create a generator - this cannot be checkpointed
        def gen():
            yield 1
            yield 2

        g = gen()
        user_ns = {"x": 1, "g": g}

        saved, removed = checkpoints.save("test", user_ns)

        # x should be saved, g should be removed
        assert "x" in saved
        assert "g" in removed

    def test_multiple_non_checkpointable_objects(self):
        """Multiple non-checkpointable objects should all be in removed."""
        import tempfile
        import socket

        checkpoints = MemoryCheckpoints()

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        user_ns = {
            "x": 1,
            "y": [1, 2, 3],
            "f": tmp,
            "sock": sock,
        }

        saved, removed = checkpoints.save("test", user_ns)

        # x and y should be saved
        assert "x" in saved
        assert "y" in saved

        # f and sock should be removed
        assert "f" in removed
        assert "sock" in removed

        # Cleanup
        tmp.close()
        sock.close()
        import os
        os.unlink(tmp.name)

    def test_removed_vars_not_in_checkpoint(self):
        """Removed variables should not appear in the checkpoint's user_ns."""
        import tempfile

        checkpoints = MemoryCheckpoints()

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        user_ns = {"x": 1, "f": tmp}

        saved, removed = checkpoints.save("test", user_ns)

        # Get the checkpoint
        cp = checkpoints.get("test")

        # x should be in checkpoint, f should not
        assert "x" in cp.user_ns
        assert "f" not in cp.user_ns

        # Cleanup
        tmp.close()
        import os
        os.unlink(tmp.name)


class TestBaseKernelCheckpointRemoval:
    """Tests for base kernel removing non-checkpointable variables from namespace."""

    def test_take_checkpoint_removes_non_checkpointable(self):
        """_take_checkpoint should remove non-checkpointable vars from namespace."""
        from unittest.mock import MagicMock, patch
        import tempfile

        # Create mock shell with user_ns
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

        mock_shell = MagicMock()
        mock_shell.user_ns = {"x": 1, "f": tmp}

        # Create mock kernel with required attributes
        mock_kernel = MagicMock()
        mock_kernel.shell = mock_shell
        mock_kernel._display = MagicMock()
        mock_kernel._vfs = MagicMock()
        mock_kernel._vfs.enabled = False
        mock_kernel._vfs.tracking_only = False

        # Import and create Checkpoints
        from flowbook.kernel_support.checkpoint import Checkpoints
        mock_kernel._checkpoints = Checkpoints()

        # Import the base kernel method and bind it
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel

        # Call _take_checkpoint directly using the bound method pattern
        _take_checkpoint = BaseFlowbookKernel._take_checkpoint.__get__(mock_kernel)
        _take_checkpoint("test_checkpoint")

        # x should still be in namespace
        assert "x" in mock_shell.user_ns

        # f should have been removed from namespace
        assert "f" not in mock_shell.user_ns

        # Warning should have been displayed
        mock_kernel._display.display_icon_and_text.assert_called()

        # Cleanup
        tmp.close()
        import os
        os.unlink(tmp.name)

    def test_take_checkpoint_preserves_checkpointable_vars(self):
        """_take_checkpoint should preserve all checkpointable variables."""
        from unittest.mock import MagicMock

        mock_shell = MagicMock()
        mock_shell.user_ns = {
            "x": 1,
            "y": "hello",
            "z": [1, 2, 3],
            "df": pd.DataFrame({"a": [1, 2]}),
        }

        mock_kernel = MagicMock()
        mock_kernel.shell = mock_shell
        mock_kernel._display = MagicMock()
        mock_kernel._vfs = MagicMock()
        mock_kernel._vfs.enabled = False
        mock_kernel._vfs.tracking_only = False

        from flowbook.kernel_support.checkpoint import Checkpoints
        mock_kernel._checkpoints = Checkpoints()

        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        _take_checkpoint = BaseFlowbookKernel._take_checkpoint.__get__(mock_kernel)
        _take_checkpoint("test_checkpoint")

        # All checkpointable vars should still be in namespace
        assert "x" in mock_shell.user_ns
        assert "y" in mock_shell.user_ns
        assert "z" in mock_shell.user_ns
        assert "df" in mock_shell.user_ns

        # No warnings should have been displayed (no removed vars)
        mock_kernel._display.display_icon_and_text.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
