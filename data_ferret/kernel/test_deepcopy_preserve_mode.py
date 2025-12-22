"""
Tests for FERRET_OBJECT_MODE=preserve deepcopy behavior.

Tests the immutability detection and shallow copy optimization for
object dtype columns that contain only immutable values.
"""

import datetime
import decimal
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from data_ferret.kernel.deepcopy import (
    is_immutable,
    _object_column_is_all_immutable,
    _IMMUTABLE_INFERRED_KINDS,
    deepcopy,
)


class TestIsImmutable:
    """Tests for is_immutable() function."""

    def test_none_is_immutable(self):
        assert is_immutable(None) is True

    def test_bool_is_immutable(self):
        assert is_immutable(True) is True
        assert is_immutable(False) is True

    def test_int_is_immutable(self):
        assert is_immutable(0) is True
        assert is_immutable(42) is True
        assert is_immutable(-100) is True

    def test_float_is_immutable(self):
        assert is_immutable(0.0) is True
        assert is_immutable(3.14) is True
        assert is_immutable(float('inf')) is True
        assert is_immutable(float('nan')) is True

    def test_complex_is_immutable(self):
        assert is_immutable(1 + 2j) is True

    def test_str_is_immutable(self):
        assert is_immutable("") is True
        assert is_immutable("hello") is True

    def test_bytes_is_immutable(self):
        assert is_immutable(b"") is True
        assert is_immutable(b"hello") is True

    def test_range_is_immutable(self):
        assert is_immutable(range(10)) is True

    def test_datetime_types_immutable(self):
        assert is_immutable(datetime.date(2024, 1, 1)) is True
        assert is_immutable(datetime.datetime(2024, 1, 1, 12, 0, 0)) is True
        assert is_immutable(datetime.time(12, 0, 0)) is True
        assert is_immutable(datetime.timedelta(days=1)) is True

    def test_decimal_is_immutable(self):
        assert is_immutable(decimal.Decimal("3.14")) is True

    def test_numpy_scalars_immutable(self):
        assert is_immutable(np.int64(42)) is True
        assert is_immutable(np.float64(3.14)) is True
        assert is_immutable(np.bool_(True)) is True

    def test_pandas_timestamps_immutable(self):
        assert is_immutable(pd.Timestamp("2024-01-01")) is True
        assert is_immutable(pd.Timedelta("1 day")) is True
        assert is_immutable(pd.Period("2024-01", freq="M")) is True

    def test_pandas_na_immutable(self):
        assert is_immutable(pd.NA) is True

    def test_tuple_immutable_if_contents_immutable(self):
        assert is_immutable(()) is True
        assert is_immutable((1, 2, 3)) is True
        assert is_immutable(("a", "b", "c")) is True
        assert is_immutable((1, "a", 3.14)) is True
        assert is_immutable((1, (2, 3))) is True  # Nested tuples

    def test_tuple_not_immutable_if_contains_mutable(self):
        assert is_immutable((1, [2, 3])) is False
        assert is_immutable((1, {"a": 1})) is False
        assert is_immutable((1, {1, 2, 3})) is False

    def test_frozenset_immutable_if_contents_immutable(self):
        assert is_immutable(frozenset()) is True
        assert is_immutable(frozenset([1, 2, 3])) is True
        assert is_immutable(frozenset(["a", "b"])) is True

    def test_list_not_immutable(self):
        assert is_immutable([]) is False
        assert is_immutable([1, 2, 3]) is False

    def test_dict_not_immutable(self):
        assert is_immutable({}) is False
        assert is_immutable({"a": 1}) is False

    def test_set_not_immutable(self):
        assert is_immutable(set()) is False
        assert is_immutable({1, 2, 3}) is False

    def test_ndarray_not_immutable(self):
        assert is_immutable(np.array([1, 2, 3])) is False

    def test_custom_object_not_immutable(self):
        class MyClass:
            pass
        assert is_immutable(MyClass()) is False

    def test_max_depth_protection(self):
        # Deeply nested tuple - should return False at depth limit
        deep = (1,)
        for _ in range(15):
            deep = (deep,)
        # With default max_depth=10, this should return False
        assert is_immutable(deep) is False


class TestObjectColumnIsAllImmutable:
    """Tests for _object_column_is_all_immutable() function."""

    def test_fast_path_string_column(self):
        series = pd.Series(["a", "b", "c"], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_fast_path_integer_column(self):
        series = pd.Series([1, 2, 3], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_fast_path_float_column(self):
        series = pd.Series([1.1, 2.2, 3.3], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_fast_path_boolean_column(self):
        series = pd.Series([True, False, True], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_fast_path_datetime_column(self):
        series = pd.Series([
            datetime.datetime(2024, 1, 1),
            datetime.datetime(2024, 1, 2),
        ], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_fast_path_with_na(self):
        series = pd.Series(["a", None, "b"], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_slow_path_mixed_immutable(self):
        # Mixed column with strings and ints - infer_dtype returns "mixed"
        series = pd.Series(["a", 1, "b", 2], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_slow_path_mixed_with_tuple(self):
        # Mixed with immutable tuple
        series = pd.Series(["a", (1, 2), "b"], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_slow_path_mixed_mutable(self):
        # Mixed column with mutable list
        series = pd.Series(["a", [1, 2], "b"], dtype=object)
        assert _object_column_is_all_immutable(series) is False

    def test_slow_path_mixed_mutable_dict(self):
        series = pd.Series(["a", {"x": 1}, "b"], dtype=object)
        assert _object_column_is_all_immutable(series) is False

    def test_empty_column(self):
        series = pd.Series([], dtype=object)
        assert _object_column_is_all_immutable(series) is True

    def test_non_object_dtype_returns_true(self):
        series = pd.Series([1, 2, 3], dtype=int)
        assert _object_column_is_all_immutable(series) is True


class TestImmutableInferredKinds:
    """Verify the _IMMUTABLE_INFERRED_KINDS set is correct."""

    def test_string_in_set(self):
        assert "string" in _IMMUTABLE_INFERRED_KINDS

    def test_integer_in_set(self):
        assert "integer" in _IMMUTABLE_INFERRED_KINDS
        assert "mixed-integer" in _IMMUTABLE_INFERRED_KINDS

    def test_floating_in_set(self):
        assert "floating" in _IMMUTABLE_INFERRED_KINDS
        assert "mixed-integer-float" in _IMMUTABLE_INFERRED_KINDS

    def test_datetime_in_set(self):
        assert "datetime" in _IMMUTABLE_INFERRED_KINDS
        assert "datetime64" in _IMMUTABLE_INFERRED_KINDS
        assert "date" in _IMMUTABLE_INFERRED_KINDS

    def test_timedelta_in_set(self):
        assert "timedelta" in _IMMUTABLE_INFERRED_KINDS
        assert "timedelta64" in _IMMUTABLE_INFERRED_KINDS

    def test_mixed_not_in_set(self):
        # "mixed" triggers slow path
        assert "mixed" not in _IMMUTABLE_INFERRED_KINDS


class TestPreserveModeDeepCopy:
    """Integration tests for FERRET_OBJECT_MODE=preserve."""

    def test_preserve_mode_keeps_object_dtype(self):
        """Object columns with strings remain object dtype in preserve mode."""
        with mock.patch.dict(os.environ, {"FERRET_OBJECT_MODE": "preserve"}):
            # Need to reload to pick up env var change
            import importlib
            import data_ferret.kernel.deepcopy as dc_module
            importlib.reload(dc_module)

            try:
                df = pd.DataFrame({"col": ["a", "b", "c"]}, dtype=object)
                df_copy = dc_module.deepcopy(df)

                assert df_copy["col"].dtype == object
                assert list(df_copy["col"]) == ["a", "b", "c"]
            finally:
                # Restore default mode
                importlib.reload(dc_module)

    def test_preserve_mode_mutable_deep_copy(self):
        """Mutable content still uses element-wise deepcopy in preserve mode."""
        with mock.patch.dict(os.environ, {"FERRET_OBJECT_MODE": "preserve"}):
            import importlib
            import data_ferret.kernel.deepcopy as dc_module
            importlib.reload(dc_module)

            try:
                original_list = [1, 2, 3]
                df = pd.DataFrame({"col": [original_list, [4, 5, 6]]}, dtype=object)
                df_copy = dc_module.deepcopy(df)

                # Should be a deep copy - modifying original shouldn't affect copy
                original_list.append(99)
                assert 99 not in df_copy["col"].iloc[0]
            finally:
                importlib.reload(dc_module)

    def test_convert_mode_converts_strings(self):
        """Convert mode (default) converts object strings to string dtype."""
        with mock.patch.dict(os.environ, {"FERRET_OBJECT_MODE": "convert"}):
            import importlib
            import data_ferret.kernel.deepcopy as dc_module
            importlib.reload(dc_module)

            try:
                df = pd.DataFrame({"col": ["a", "b", "c"]}, dtype=object)
                df_copy = dc_module.deepcopy(df)

                # Convert mode should convert to string dtype
                assert df_copy["col"].dtype == "string" or df["col"].dtype == "string"
            finally:
                importlib.reload(dc_module)


class TestPreserveModeSeries:
    """Tests for Series deepcopy in preserve mode."""

    def test_preserve_mode_series_keeps_object_dtype(self):
        """Object Series with strings remains object dtype in preserve mode."""
        with mock.patch.dict(os.environ, {"FERRET_OBJECT_MODE": "preserve"}):
            import importlib
            import data_ferret.kernel.deepcopy as dc_module
            importlib.reload(dc_module)

            try:
                series = pd.Series(["a", "b", "c"], dtype=object)
                series_copy = dc_module.deepcopy(series)

                assert series_copy.dtype == object
                assert list(series_copy) == ["a", "b", "c"]
            finally:
                importlib.reload(dc_module)

    def test_preserve_mode_series_mutable_deep_copy(self):
        """Mutable content in Series still uses element-wise deepcopy."""
        with mock.patch.dict(os.environ, {"FERRET_OBJECT_MODE": "preserve"}):
            import importlib
            import data_ferret.kernel.deepcopy as dc_module
            importlib.reload(dc_module)

            try:
                original_list = [1, 2, 3]
                series = pd.Series([original_list, [4, 5, 6]], dtype=object)
                series_copy = dc_module.deepcopy(series)

                # Should be a deep copy
                original_list.append(99)
                assert 99 not in series_copy.iloc[0]
            finally:
                importlib.reload(dc_module)
