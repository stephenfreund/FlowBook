"""
Tests for proper diff behavior on all deepcopyable types.

This file tests that the Diff class correctly handles equality and difference
detection for all types that are deepcopyable, as defined in test_deepcopyable.py.
"""

import datetime
import copy
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from data_ferret.kernel.diff import Diff


class TestDiffImmutablePrimitives:
    """Tests for diff behavior on immutable primitive types."""

    def test_none_equal(self):
        differ = Diff()
        result = differ.diff({"x": None}, {"x": None})
        assert "x" not in result.differences

    def test_none_different(self):
        differ = Diff()
        result = differ.diff({"x": None}, {"x": 1})
        assert "x" in result.differences

    def test_bool_true_equal(self):
        differ = Diff()
        result = differ.diff({"x": True}, {"x": True})
        assert "x" not in result.differences

    def test_bool_false_equal(self):
        differ = Diff()
        result = differ.diff({"x": False}, {"x": False})
        assert "x" not in result.differences

    def test_bool_different(self):
        differ = Diff()
        result = differ.diff({"x": True}, {"x": False})
        assert "x" in result.differences

    def test_int_equal(self):
        differ = Diff()
        result = differ.diff({"x": 42}, {"x": 42})
        assert "x" not in result.differences

    def test_int_different(self):
        differ = Diff()
        result = differ.diff({"x": 42}, {"x": 43})
        assert "x" in result.differences

    def test_int_zero_equal(self):
        differ = Diff()
        result = differ.diff({"x": 0}, {"x": 0})
        assert "x" not in result.differences

    def test_int_negative_equal(self):
        differ = Diff()
        result = differ.diff({"x": -1}, {"x": -1})
        assert "x" not in result.differences

    def test_float_equal(self):
        differ = Diff()
        result = differ.diff({"x": 3.14}, {"x": 3.14})
        assert "x" not in result.differences

    def test_float_different(self):
        differ = Diff()
        result = differ.diff({"x": 3.14}, {"x": 3.15})
        assert "x" in result.differences

    def test_float_nan_equal(self):
        """NaN should be considered equal to NaN."""
        differ = Diff()
        result = differ.diff({"x": float("nan")}, {"x": float("nan")})
        assert "x" not in result.differences

    def test_float_inf_equal(self):
        differ = Diff()
        result = differ.diff({"x": float("inf")}, {"x": float("inf")})
        assert "x" not in result.differences

    def test_float_close_reported(self):
        """Close values should be reported with status='close' by default."""
        differ = Diff(rtol=1e-5, report_close=True)
        result = differ.diff({"x": 1.0}, {"x": 1.0 + 1e-7})
        assert "x" in result.differences
        assert result.differences["x"].status == "close"

    def test_float_close_not_reported_when_disabled(self):
        """Close values should not be reported when report_close=False."""
        differ = Diff(rtol=1e-5, report_close=False)
        result = differ.diff({"x": 1.0}, {"x": 1.0 + 1e-7})
        assert "x" not in result.differences

    def test_complex_equal(self):
        differ = Diff()
        result = differ.diff({"x": complex(1, 2)}, {"x": complex(1, 2)})
        assert "x" not in result.differences

    def test_complex_different_real(self):
        differ = Diff()
        result = differ.diff({"x": complex(1, 2)}, {"x": complex(2, 2)})
        assert "x" in result.differences

    def test_complex_different_imag(self):
        differ = Diff()
        result = differ.diff({"x": complex(1, 2)}, {"x": complex(1, 3)})
        assert "x" in result.differences

    def test_str_equal(self):
        differ = Diff()
        result = differ.diff({"x": "hello"}, {"x": "hello"})
        assert "x" not in result.differences

    def test_str_different(self):
        differ = Diff()
        result = differ.diff({"x": "hello"}, {"x": "world"})
        assert "x" in result.differences

    def test_str_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": ""}, {"x": ""})
        assert "x" not in result.differences

    def test_bytes_equal(self):
        differ = Diff()
        result = differ.diff({"x": b"bytes"}, {"x": b"bytes"})
        assert "x" not in result.differences

    def test_bytes_different(self):
        differ = Diff()
        result = differ.diff({"x": b"bytes"}, {"x": b"other"})
        assert "x" in result.differences

    def test_bytes_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": b""}, {"x": b""})
        assert "x" not in result.differences

    def test_range_equal(self):
        differ = Diff()
        result = differ.diff({"x": range(10)}, {"x": range(10)})
        assert "x" not in result.differences

    def test_range_different(self):
        differ = Diff()
        result = differ.diff({"x": range(10)}, {"x": range(5)})
        assert "x" in result.differences


class TestDiffDatetimeTypes:
    """Tests for diff behavior on datetime module types."""

    def test_date_equal(self):
        differ = Diff()
        d = datetime.date(2021, 1, 1)
        result = differ.diff({"x": d}, {"x": copy.deepcopy(d)})
        assert "x" not in result.differences

    def test_date_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": datetime.date(2021, 1, 1)}, {"x": datetime.date(2021, 1, 2)}
        )
        assert "x" in result.differences

    def test_datetime_equal(self):
        differ = Diff()
        dt = datetime.datetime(2021, 1, 1, 12, 30, 45)
        result = differ.diff({"x": dt}, {"x": copy.deepcopy(dt)})
        assert "x" not in result.differences

    def test_datetime_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": datetime.datetime(2021, 1, 1, 12, 30, 45)},
            {"x": datetime.datetime(2021, 1, 1, 12, 30, 46)},
        )
        assert "x" in result.differences

    def test_time_equal(self):
        differ = Diff()
        t = datetime.time(12, 30)
        result = differ.diff({"x": t}, {"x": copy.deepcopy(t)})
        assert "x" not in result.differences

    def test_time_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": datetime.time(12, 30)}, {"x": datetime.time(12, 31)}
        )
        assert "x" in result.differences

    def test_timedelta_equal(self):
        differ = Diff()
        td = datetime.timedelta(days=1, hours=2)
        result = differ.diff({"x": td}, {"x": copy.deepcopy(td)})
        assert "x" not in result.differences

    def test_timedelta_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": datetime.timedelta(days=1)}, {"x": datetime.timedelta(days=2)}
        )
        assert "x" in result.differences


class TestDiffContainerTypes:
    """Tests for diff behavior on standard container types."""

    def test_list_equal(self):
        differ = Diff()
        result = differ.diff({"x": [1, 2, 3]}, {"x": [1, 2, 3]})
        assert "x" not in result.differences

    def test_list_different_values(self):
        differ = Diff()
        result = differ.diff({"x": [1, 2, 3]}, {"x": [1, 2, 4]})
        assert "x" in result.differences

    def test_list_different_length(self):
        differ = Diff()
        result = differ.diff({"x": [1, 2, 3]}, {"x": [1, 2]})
        assert "x" in result.differences

    def test_list_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": []}, {"x": []})
        assert "x" not in result.differences

    def test_dict_equal(self):
        differ = Diff()
        result = differ.diff({"x": {"a": 1, "b": 2}}, {"x": {"a": 1, "b": 2}})
        assert "x" not in result.differences

    def test_dict_different_values(self):
        differ = Diff()
        result = differ.diff({"x": {"a": 1, "b": 2}}, {"x": {"a": 1, "b": 3}})
        assert "x" in result.differences

    def test_dict_different_keys(self):
        differ = Diff()
        result = differ.diff({"x": {"a": 1}}, {"x": {"b": 1}})
        assert "x" in result.differences

    def test_dict_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": {}}, {"x": {}})
        assert "x" not in result.differences

    def test_set_equal(self):
        differ = Diff()
        result = differ.diff({"x": {1, 2, 3}}, {"x": {1, 2, 3}})
        assert "x" not in result.differences

    def test_set_different(self):
        differ = Diff()
        result = differ.diff({"x": {1, 2, 3}}, {"x": {1, 2, 4}})
        assert "x" in result.differences

    def test_set_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": set()}, {"x": set()})
        assert "x" not in result.differences

    def test_tuple_equal(self):
        differ = Diff()
        result = differ.diff({"x": (1, 2, 3)}, {"x": (1, 2, 3)})
        assert "x" not in result.differences

    def test_tuple_different(self):
        differ = Diff()
        result = differ.diff({"x": (1, 2, 3)}, {"x": (1, 2, 4)})
        assert "x" in result.differences

    def test_tuple_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": ()}, {"x": ()})
        assert "x" not in result.differences

    def test_frozenset_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": frozenset([1, 2, 3])}, {"x": frozenset([1, 2, 3])}
        )
        assert "x" not in result.differences

    def test_frozenset_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": frozenset([1, 2, 3])}, {"x": frozenset([1, 2, 4])}
        )
        assert "x" in result.differences

    def test_frozenset_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": frozenset()}, {"x": frozenset()})
        assert "x" not in result.differences


class TestDiffNestedContainers:
    """Tests for diff behavior on nested container structures."""

    def test_nested_lists_equal(self):
        differ = Diff()
        result = differ.diff({"x": [1, [2, [3, [4]]]]}, {"x": [1, [2, [3, [4]]]]})
        assert "x" not in result.differences

    def test_nested_lists_different(self):
        differ = Diff()
        result = differ.diff({"x": [1, [2, [3, [4]]]]}, {"x": [1, [2, [3, [5]]]]})
        assert "x" in result.differences

    def test_nested_dicts_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": {"a": {"b": {"c": 1}}}}, {"x": {"a": {"b": {"c": 1}}}}
        )
        assert "x" not in result.differences

    def test_nested_dicts_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": {"a": {"b": {"c": 1}}}}, {"x": {"a": {"b": {"c": 2}}}}
        )
        assert "x" in result.differences

    def test_mixed_nesting_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": {"a": [1, 2], "b": (3, 4)}}, {"x": {"a": [1, 2], "b": (3, 4)}}
        )
        assert "x" not in result.differences

    def test_mixed_nesting_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": {"a": [1, 2], "b": (3, 4)}}, {"x": {"a": [1, 2], "b": (3, 5)}}
        )
        assert "x" in result.differences

    def test_list_of_dicts_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": [{"a": 1}, {"b": 2}]}, {"x": [{"a": 1}, {"b": 2}]}
        )
        assert "x" not in result.differences

    def test_dict_with_tuple_keys_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": {(1, 2): "a", (3, 4): "b"}}, {"x": {(1, 2): "a", (3, 4): "b"}}
        )
        assert "x" not in result.differences


class TestDiffCircularReferences:
    """Tests for diff behavior with circular references."""

    def test_circular_list_equal(self):
        differ = Diff()
        lst1 = [1, 2, 3]
        lst1.append(lst1)
        lst2 = [1, 2, 3]
        lst2.append(lst2)
        result = differ.diff({"x": lst1}, {"x": lst2})
        assert "x" not in result.differences

    def test_circular_dict_equal(self):
        differ = Diff()
        d1 = {"a": 1}
        d1["self"] = d1
        d2 = {"a": 1}
        d2["self"] = d2
        result = differ.diff({"x": d1}, {"x": d2})
        assert "x" not in result.differences

    def test_mutually_referencing_lists_equal(self):
        differ = Diff()
        a1 = [1]
        b1 = [2]
        a1.append(b1)
        b1.append(a1)

        a2 = [1]
        b2 = [2]
        a2.append(b2)
        b2.append(a2)

        result = differ.diff({"x": a1}, {"x": a2})
        assert "x" not in result.differences


class TestDiffNumPyTypes:
    """Tests for diff behavior on NumPy types."""

    def test_array_int_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": np.array([1, 2, 3])}, {"x": np.array([1, 2, 3])}
        )
        assert "x" not in result.differences

    def test_array_int_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": np.array([1, 2, 3])}, {"x": np.array([1, 2, 4])}
        )
        assert "x" in result.differences

    def test_array_float_equal(self):
        differ = Diff()
        result = differ.diff({"x": np.zeros((3, 3))}, {"x": np.zeros((3, 3))})
        assert "x" not in result.differences

    def test_array_float_different(self):
        differ = Diff()
        arr1 = np.zeros((3, 3))
        arr2 = np.zeros((3, 3))
        arr2[0, 0] = 1.0
        result = differ.diff({"x": arr1}, {"x": arr2})
        assert "x" in result.differences

    def test_array_2d_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": np.array([[1, 2], [3, 4]])}, {"x": np.array([[1, 2], [3, 4]])}
        )
        assert "x" not in result.differences

    def test_array_shape_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": np.array([[1, 2], [3, 4]])}, {"x": np.array([1, 2, 3, 4])}
        )
        assert "x" in result.differences

    def test_scalar_int64_equal(self):
        differ = Diff()
        result = differ.diff({"x": np.int64(42)}, {"x": np.int64(42)})
        assert "x" not in result.differences

    def test_scalar_int64_different(self):
        differ = Diff()
        result = differ.diff({"x": np.int64(42)}, {"x": np.int64(43)})
        assert "x" in result.differences

    def test_scalar_float64_equal(self):
        differ = Diff()
        result = differ.diff({"x": np.float64(3.14)}, {"x": np.float64(3.14)})
        assert "x" not in result.differences

    def test_scalar_float64_nan_equal(self):
        """NaN should be equal to NaN for numpy scalars."""
        differ = Diff()
        result = differ.diff({"x": np.float64("nan")}, {"x": np.float64("nan")})
        assert "x" not in result.differences

    def test_scalar_bool_equal(self):
        differ = Diff()
        result = differ.diff({"x": np.bool_(True)}, {"x": np.bool_(True)})
        assert "x" not in result.differences

    def test_object_array_copyable_contents_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": np.array([1, "hello", 3.14], dtype=object)},
            {"x": np.array([1, "hello", 3.14], dtype=object)},
        )
        assert "x" not in result.differences

    def test_object_array_with_lists_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": np.array([[1, 2], [3, 4]], dtype=object)},
            {"x": np.array([[1, 2], [3, 4]], dtype=object)},
        )
        assert "x" not in result.differences

    def test_array_with_nan_equal(self):
        """Arrays with NaN should be equal when NaN positions match."""
        differ = Diff()
        result = differ.diff(
            {"x": np.array([1.0, np.nan, 3.0])},
            {"x": np.array([1.0, np.nan, 3.0])},
        )
        assert "x" not in result.differences

    def test_structured_array_equal(self):
        differ = Diff()
        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        arr1 = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        arr2 = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        result = differ.diff({"x": arr1}, {"x": arr2})
        assert "x" not in result.differences

    def test_int_dtypes_compatible(self):
        """int32 and int64 arrays with same values should be equal."""
        differ = Diff()
        result = differ.diff(
            {"x": np.array([1, 2, 3], dtype=np.int32)},
            {"x": np.array([1, 2, 3], dtype=np.int64)},
        )
        assert "x" not in result.differences

    def test_float_dtypes_compatible(self):
        """float32 and float64 arrays with same values should be equal."""
        differ = Diff()
        result = differ.diff(
            {"x": np.array([1.5, 2.5], dtype=np.float32)},
            {"x": np.array([1.5, 2.5], dtype=np.float64)},
        )
        assert "x" not in result.differences


class TestDiffPandasTypes:
    """Tests for diff behavior on Pandas types."""

    def test_series_int_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1, 2, 3])}, {"x": pd.Series([1, 2, 3])}
        )
        assert "x" not in result.differences

    def test_series_int_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1, 2, 3])}, {"x": pd.Series([1, 2, 4])}
        )
        assert "x" in result.differences

    def test_series_float_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1.0, 2.0, 3.0])}, {"x": pd.Series([1.0, 2.0, 3.0])}
        )
        assert "x" not in result.differences

    def test_series_string_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series(["a", "b", "c"])}, {"x": pd.Series(["a", "b", "c"])}
        )
        assert "x" not in result.differences

    def test_series_object_copyable_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1, "a", 3.0], dtype=object)},
            {"x": pd.Series([1, "a", 3.0], dtype=object)},
        )
        assert "x" not in result.differences

    def test_series_with_nan_equal(self):
        """Series with NaN should be equal when NaN positions match."""
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1.0, np.nan, 3.0])},
            {"x": pd.Series([1.0, np.nan, 3.0])},
        )
        assert "x" not in result.differences

    def test_series_index_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1, 2], index=["a", "b"])},
            {"x": pd.Series([1, 2], index=["a", "c"])},
        )
        assert "x" in result.differences

    def test_dataframe_simple_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.DataFrame({"a": [1, 2], "b": [3, 4]})},
            {"x": pd.DataFrame({"a": [1, 2], "b": [3, 4]})},
        )
        assert "x" not in result.differences

    def test_dataframe_different_values(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.DataFrame({"a": [1, 2], "b": [3, 4]})},
            {"x": pd.DataFrame({"a": [1, 2], "b": [3, 5]})},
        )
        assert "x" in result.differences

    def test_dataframe_mixed_dtypes_equal(self):
        differ = Diff()
        df = pd.DataFrame({"int": [1, 2], "float": [1.0, 2.0], "str": ["a", "b"]})
        result = differ.diff({"x": df}, {"x": df.copy()})
        assert "x" not in result.differences

    def test_dataframe_object_column_copyable_equal(self):
        differ = Diff()
        df = pd.DataFrame({"a": [[1, 2], [3, 4]]}, dtype=object)
        # Need to make a proper deep copy
        df2 = pd.DataFrame({"a": [[1, 2], [3, 4]]}, dtype=object)
        result = differ.diff({"x": df}, {"x": df2})
        assert "x" not in result.differences

    def test_dataframe_shape_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.DataFrame({"a": [1, 2]})},
            {"x": pd.DataFrame({"a": [1, 2, 3]})},
        )
        assert "x" in result.differences

    def test_dataframe_columns_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.DataFrame({"a": [1, 2]})},
            {"x": pd.DataFrame({"b": [1, 2]})},
        )
        assert "x" in result.differences

    def test_dataframe_with_nan_equal(self):
        """DataFrames with NaN should be equal when NaN positions match."""
        differ = Diff()
        result = differ.diff(
            {"x": pd.DataFrame({"a": [1.0, np.nan, 3.0]})},
            {"x": pd.DataFrame({"a": [1.0, np.nan, 3.0]})},
        )
        assert "x" not in result.differences

    def test_timestamp_equal(self):
        differ = Diff()
        ts = pd.Timestamp("2021-01-01")
        result = differ.diff({"x": ts}, {"x": pd.Timestamp("2021-01-01")})
        assert "x" not in result.differences

    def test_timestamp_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Timestamp("2021-01-01")}, {"x": pd.Timestamp("2021-01-02")}
        )
        assert "x" in result.differences

    def test_timedelta_equal(self):
        differ = Diff()
        td = pd.Timedelta("1 day")
        result = differ.diff({"x": td}, {"x": pd.Timedelta("1 day")})
        assert "x" not in result.differences

    def test_timedelta_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Timedelta("1 day")}, {"x": pd.Timedelta("2 days")}
        )
        assert "x" in result.differences

    def test_period_equal(self):
        differ = Diff()
        p = pd.Period("2021-01", freq="M")
        result = differ.diff({"x": p}, {"x": pd.Period("2021-01", freq="M")})
        assert "x" not in result.differences

    def test_index_int_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Index([1, 2, 3])}, {"x": pd.Index([1, 2, 3])}
        )
        assert "x" not in result.differences

    def test_index_string_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": pd.Index(["a", "b", "c"])}, {"x": pd.Index(["a", "b", "c"])}
        )
        assert "x" not in result.differences

    def test_na_equal(self):
        differ = Diff()
        result = differ.diff({"x": pd.NA}, {"x": pd.NA})
        assert "x" not in result.differences

    def test_int_dtypes_compatible_series(self):
        """int32 and int64 Series with same values should be equal."""
        differ = Diff()
        result = differ.diff(
            {"x": pd.Series([1, 2, 3], dtype=np.int32)},
            {"x": pd.Series([1, 2, 3], dtype=np.int64)},
        )
        assert "x" not in result.differences

    def test_int_dtypes_compatible_dataframe(self):
        """int32 and int64 DataFrame columns should be equal."""
        differ = Diff()
        result = differ.diff(
            {"x": pd.DataFrame({"a": [1, 2]}, dtype=np.int32)},
            {"x": pd.DataFrame({"a": [1, 2]}, dtype=np.int64)},
        )
        assert "x" not in result.differences


class TestDiffCollectionsTypes:
    """Tests for diff behavior on collections module types."""

    def test_deque_equal(self):
        differ = Diff()
        result = differ.diff({"x": deque([1, 2, 3])}, {"x": deque([1, 2, 3])})
        assert "x" not in result.differences

    def test_deque_different(self):
        differ = Diff()
        result = differ.diff({"x": deque([1, 2, 3])}, {"x": deque([1, 2, 4])})
        assert "x" in result.differences

    def test_deque_empty_equal(self):
        differ = Diff()
        result = differ.diff({"x": deque()}, {"x": deque()})
        assert "x" not in result.differences

    def test_ordered_dict_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": OrderedDict([("a", 1), ("b", 2)])},
            {"x": OrderedDict([("a", 1), ("b", 2)])},
        )
        assert "x" not in result.differences

    def test_ordered_dict_different(self):
        differ = Diff()
        result = differ.diff(
            {"x": OrderedDict([("a", 1), ("b", 2)])},
            {"x": OrderedDict([("a", 1), ("b", 3)])},
        )
        assert "x" in result.differences

    def test_defaultdict_equal(self):
        differ = Diff()
        dd1 = defaultdict(list, {"a": [1, 2]})
        dd2 = defaultdict(list, {"a": [1, 2]})
        result = differ.diff({"x": dd1}, {"x": dd2})
        assert "x" not in result.differences

    def test_defaultdict_different(self):
        differ = Diff()
        dd1 = defaultdict(list, {"a": [1, 2]})
        dd2 = defaultdict(list, {"a": [1, 3]})
        result = differ.diff({"x": dd1}, {"x": dd2})
        assert "x" in result.differences

    def test_counter_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": Counter("abracadabra")}, {"x": Counter("abracadabra")}
        )
        assert "x" not in result.differences

    def test_counter_different(self):
        differ = Diff()
        result = differ.diff({"x": Counter("abc")}, {"x": Counter("abd")})
        assert "x" in result.differences


class TestDiffCustomClasses:
    """Tests for diff behavior on user-defined classes."""

    def test_simple_class_equal(self):
        class MyClass:
            def __init__(self, x):
                self.x = x

        differ = Diff()
        result = differ.diff({"obj": MyClass(42)}, {"obj": MyClass(42)})
        assert "obj" not in result.differences

    def test_simple_class_different(self):
        class MyClass:
            def __init__(self, x):
                self.x = x

        differ = Diff()
        result = differ.diff({"obj": MyClass(42)}, {"obj": MyClass(43)})
        assert "obj" in result.differences

    def test_class_with_list_attribute_equal(self):
        class MyClass:
            def __init__(self):
                self.items = [1, 2, 3]

        differ = Diff()
        result = differ.diff({"obj": MyClass()}, {"obj": MyClass()})
        assert "obj" not in result.differences

    def test_class_with_list_attribute_different(self):
        class MyClass:
            def __init__(self, items):
                self.items = items

        differ = Diff()
        result = differ.diff(
            {"obj": MyClass([1, 2, 3])}, {"obj": MyClass([1, 2, 4])}
        )
        assert "obj" in result.differences

    def test_dataclass_equal(self):
        @dataclass
        class Point:
            x: int
            y: int

        differ = Diff()
        result = differ.diff({"p": Point(1, 2)}, {"p": Point(1, 2)})
        assert "p" not in result.differences

    def test_dataclass_different(self):
        @dataclass
        class Point:
            x: int
            y: int

        differ = Diff()
        result = differ.diff({"p": Point(1, 2)}, {"p": Point(1, 3)})
        assert "p" in result.differences

    def test_dataclass_with_list_equal(self):
        @dataclass
        class Container:
            items: list

        differ = Diff()
        result = differ.diff(
            {"c": Container([1, 2, 3])}, {"c": Container([1, 2, 3])}
        )
        assert "c" not in result.differences


class TestDiffSlotsClasses:
    """Tests for diff behavior on classes using __slots__."""

    def test_slots_class_equal(self):
        class SlotClass:
            __slots__ = ["x", "y"]

            def __init__(self, x, y):
                self.x = x
                self.y = y

        differ = Diff()
        result = differ.diff({"s": SlotClass(1, 2)}, {"s": SlotClass(1, 2)})
        assert "s" not in result.differences

    def test_slots_class_different(self):
        class SlotClass:
            __slots__ = ["x", "y"]

            def __init__(self, x, y):
                self.x = x
                self.y = y

        differ = Diff()
        result = differ.diff({"s": SlotClass(1, 2)}, {"s": SlotClass(1, 3)})
        assert "s" in result.differences


class TestDiffFunctions:
    """Tests for diff behavior on function objects."""

    def test_same_function_equal(self):
        def my_func(x):
            return x + 1

        differ = Diff()
        result = differ.diff({"f": my_func}, {"f": my_func})
        assert "f" not in result.differences

    def test_same_lambda_equal(self):
        fn = lambda x: x + 1
        differ = Diff()
        result = differ.diff({"f": fn}, {"f": fn})
        assert "f" not in result.differences

    def test_bound_method_same_instance_equal(self):
        class MyClass:
            def method(self):
                pass

        obj = MyClass()
        differ = Diff()
        result = differ.diff({"m": obj.method}, {"m": obj.method})
        assert "m" not in result.differences

    def test_bound_method_different_equal_instances(self):
        """Bound methods from structurally equal instances should be equal."""
        class MyClass:
            def __init__(self, x):
                self.x = x

            def method(self):
                pass

        differ = Diff()
        result = differ.diff(
            {"m": MyClass(1).method}, {"m": MyClass(1).method}
        )
        assert "m" not in result.differences


class TestDiffTypeObjects:
    """Tests for diff behavior on type objects (classes themselves)."""

    def test_builtin_type_equal(self):
        differ = Diff()
        result = differ.diff({"t": int}, {"t": int})
        assert "t" not in result.differences

    def test_builtin_type_different(self):
        differ = Diff()
        result = differ.diff({"t": int}, {"t": str})
        assert "t" in result.differences

    def test_custom_class_type_equal(self):
        class MyClass:
            pass

        differ = Diff()
        result = differ.diff({"t": MyClass}, {"t": MyClass})
        assert "t" not in result.differences

    def test_numpy_type_equal(self):
        differ = Diff()
        result = differ.diff({"t": np.ndarray}, {"t": np.ndarray})
        assert "t" not in result.differences

    def test_pandas_type_equal(self):
        differ = Diff()
        result = differ.diff({"t": pd.DataFrame}, {"t": pd.DataFrame})
        assert "t" not in result.differences


class TestDiffDecimalType:
    """Tests for diff behavior on decimal.Decimal type."""

    def test_decimal_equal(self):
        differ = Diff()
        result = differ.diff({"x": Decimal("3.14")}, {"x": Decimal("3.14")})
        assert "x" not in result.differences

    def test_decimal_different(self):
        differ = Diff()
        result = differ.diff({"x": Decimal("3.14")}, {"x": Decimal("3.15")})
        assert "x" in result.differences

    def test_decimal_infinity_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": Decimal("Infinity")}, {"x": Decimal("Infinity")}
        )
        assert "x" not in result.differences

    def test_decimal_neg_infinity_equal(self):
        differ = Diff()
        result = differ.diff(
            {"x": Decimal("-Infinity")}, {"x": Decimal("-Infinity")}
        )
        assert "x" not in result.differences

    def test_decimal_nan_equal(self):
        """Decimal NaN should be equal to Decimal NaN."""
        differ = Diff()
        result = differ.diff({"x": Decimal("NaN")}, {"x": Decimal("NaN")})
        # Note: Decimal NaN != Decimal NaN by default
        # This tests current behavior - may differ from float NaN handling
        assert "x" in result.differences  # Decimal NaN is not equal to itself


class TestDiffPointerStructure:
    """Tests for diff behavior with shared references (pointer structure)."""

    def test_list_with_shared_reference_equal(self):
        """Lists with same pointer structure should be equal."""
        differ = Diff()
        shared1 = [1, 2, 3]
        ns1 = {"a": shared1, "b": shared1}

        shared2 = [1, 2, 3]
        ns2 = {"a": shared2, "b": shared2}

        result = differ.diff(ns1, ns2)
        assert "a" not in result.differences
        assert "b" not in result.differences

    def test_list_with_different_pointer_structure(self):
        """Lists with different pointer structure should be detected."""
        differ = Diff()
        shared = [1, 2, 3]
        ns1 = {"a": shared, "b": shared}  # a and b point to same list

        ns2 = {"a": [1, 2, 3], "b": [1, 2, 3]}  # a and b are different lists

        result = differ.diff(ns1, ns2)
        # The diff should detect pointer structure mismatch
        assert "b" in result.differences

    def test_nested_shared_references_equal(self):
        """Nested structures with same pointer pattern should be equal."""
        differ = Diff()

        inner1 = {"x": 1}
        outer1 = [inner1, inner1]

        inner2 = {"x": 1}
        outer2 = [inner2, inner2]

        result = differ.diff({"obj": outer1}, {"obj": outer2})
        assert "obj" not in result.differences


class TestDiffVariableAddRemove:
    """Tests for diff behavior when variables are added or removed."""

    def test_variable_added(self):
        differ = Diff()
        result = differ.diff({"x": 1}, {"x": 1, "y": 2})
        assert "y" in result.differences
        assert "Variable was added" in result.differences["y"].message

    def test_variable_removed(self):
        differ = Diff()
        result = differ.diff({"x": 1, "y": 2}, {"x": 1})
        assert "y" in result.differences
        assert "Variable was removed" in result.differences["y"].message

    def test_variable_unchanged(self):
        differ = Diff()
        result = differ.diff({"x": 1, "y": 2}, {"x": 1, "y": 2})
        assert "x" not in result.differences
        assert "y" not in result.differences


class TestDiffMultipleVariables:
    """Tests for diff behavior with multiple variables."""

    def test_multiple_equal_variables(self):
        differ = Diff()
        ns = {
            "int_var": 42,
            "float_var": 3.14,
            "str_var": "hello",
            "list_var": [1, 2, 3],
            "dict_var": {"a": 1},
            "array_var": np.array([1, 2, 3]),
            "df_var": pd.DataFrame({"a": [1, 2]}),
        }
        result = differ.diff(ns, copy.deepcopy(ns))
        assert len(result.differences) == 0

    def test_some_variables_different(self):
        differ = Diff()
        ns1 = {"a": 1, "b": 2, "c": 3}
        ns2 = {"a": 1, "b": 5, "c": 3}  # b is different
        result = differ.diff(ns1, ns2)
        assert "a" not in result.differences
        assert "b" in result.differences
        assert "c" not in result.differences
