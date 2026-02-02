"""
Tests for object dtype / float dtype compatibility in diff.py

Tests that object dtype arrays/series/dataframes containing all floats
are considered equal to float64 arrays/series/dataframes.
"""

import numpy as np
import pandas as pd
import pytest
from flowbook.kernel_support.diff import Diff


def test_numpy_array_object_vs_float64():
    """Test that object array with floats equals float64 array"""
    # Create object array with float elements
    arr_obj = np.array([1.0, 2.5, 3.7], dtype=object)
    arr_float = np.array([1.0, 2.5, 3.7], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object array with floats should equal float64 array"


def test_numpy_array_object_vs_float32():
    """Test that object array with floats equals float32 array"""
    # Create object array with float elements
    arr_obj = np.array([1.0, 2.5, 3.7], dtype=object)
    arr_float = np.array([1.0, 2.5, 3.7], dtype=np.float32)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object array with floats should equal float32 array"


def test_numpy_array_object_with_nan():
    """Test that object array with floats and NaN equals float64 array with NaN"""
    # Create object array with float elements including NaN
    arr_obj = np.array([1.0, np.nan, 3.7], dtype=object)
    arr_float = np.array([1.0, np.nan, 3.7], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object array with floats and NaN should equal float64 array"


def test_numpy_array_object_with_mixed_types():
    """Test that object array with mixed types does NOT equal float64 array"""
    # Create object array with mixed types (floats and strings)
    arr_obj = np.array([1.0, "hello", 3.7], dtype=object)
    arr_float = np.array([1.0, 2.0, 3.7], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) > 0, "Object array with mixed types should NOT equal float64 array"
    assert "arr" in result.differences


def test_numpy_array_object_with_integers():
    """Test that object array with integers does NOT equal float64 array"""
    # Create object array with integer elements
    arr_obj = np.array([1, 2, 3], dtype=object)
    arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) > 0, "Object array with integers should NOT equal float64 array"
    assert "arr" in result.differences


def test_pandas_series_object_vs_float64():
    """Test that Series with object dtype containing floats equals float64 Series"""
    # Create Series with object dtype containing floats
    s_obj = pd.Series([1.0, 2.5, 3.7], dtype=object)
    s_float = pd.Series([1.0, 2.5, 3.7], dtype=np.float64)

    differ = Diff()
    a = {"s": s_obj}
    b = {"s": s_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object Series with floats should equal float64 Series"


def test_pandas_series_object_with_nan():
    """Test that Series with object dtype containing floats and NaN equals float64 Series"""
    # Create Series with object dtype containing floats and NaN
    s_obj = pd.Series([1.0, np.nan, 3.7], dtype=object)
    s_float = pd.Series([1.0, np.nan, 3.7], dtype=np.float64)

    differ = Diff()
    a = {"s": s_obj}
    b = {"s": s_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object Series with floats and NaN should equal float64 Series"


def test_pandas_series_object_with_mixed_types():
    """Test that Series with mixed types does NOT equal float64 Series"""
    # Create Series with object dtype containing mixed types
    s_obj = pd.Series([1.0, "hello", 3.7], dtype=object)
    s_float = pd.Series([1.0, 2.0, 3.7], dtype=np.float64)

    differ = Diff()
    a = {"s": s_obj}
    b = {"s": s_float}
    result = differ.diff(a, b)

    assert len(result.differences) > 0, "Object Series with mixed types should NOT equal float64 Series"
    assert "s" in result.differences


def test_pandas_dataframe_object_vs_float64():
    """Test that DataFrame with object dtype columns containing floats equals float64 DataFrame"""
    # Create DataFrame with object dtype columns containing floats
    df_obj = pd.DataFrame({"a": [1.0, 2.5], "b": [3.7, 4.2]}, dtype=object)
    df_float = pd.DataFrame({"a": [1.0, 2.5], "b": [3.7, 4.2]}, dtype=np.float64)

    differ = Diff()
    a = {"df": df_obj}
    b = {"df": df_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object DataFrame with floats should equal float64 DataFrame"


def test_pandas_dataframe_object_with_nan():
    """Test that DataFrame with object dtype containing floats and NaN equals float64 DataFrame"""
    # Create DataFrame with object dtype columns containing floats and NaN
    df_obj = pd.DataFrame({"a": [1.0, np.nan], "b": [3.7, 4.2]}, dtype=object)
    df_float = pd.DataFrame({"a": [1.0, np.nan], "b": [3.7, 4.2]}, dtype=np.float64)

    differ = Diff()
    a = {"df": df_obj}
    b = {"df": df_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object DataFrame with floats and NaN should equal float64 DataFrame"


def test_pandas_dataframe_mixed_columns():
    """Test DataFrame with one object column (floats) and one float64 column"""
    # Create DataFrame with mixed column dtypes
    df_mixed = pd.DataFrame({"a": pd.Series([1.0, 2.5], dtype=object), "b": pd.Series([3.7, 4.2], dtype=np.float64)})
    df_float = pd.DataFrame({"a": [1.0, 2.5], "b": [3.7, 4.2]}, dtype=np.float64)

    differ = Diff()
    a = {"df": df_mixed}
    b = {"df": df_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "DataFrame with mixed object/float columns should equal float64 DataFrame"


def test_pandas_dataframe_object_with_mixed_types():
    """Test that DataFrame with object column containing mixed types does NOT equal float64 DataFrame"""
    # Create DataFrame with object dtype column containing mixed types
    df_obj = pd.DataFrame({"a": [1.0, "hello"], "b": [3.7, 4.2]})
    df_float = pd.DataFrame({"a": [1.0, 2.0], "b": [3.7, 4.2]}, dtype=np.float64)

    differ = Diff()
    a = {"df": df_obj}
    b = {"df": df_float}
    result = differ.diff(a, b)

    assert len(result.differences) > 0, "DataFrame with mixed types should NOT equal float64 DataFrame"
    assert "df" in result.differences


def test_numpy_floats_in_object_array():
    """Test object array with numpy float types equals float64 array"""
    # Create object array with numpy float types
    # Use values that can be represented exactly in all float types
    arr_obj = np.array([np.float64(1.0), np.float32(2.5), np.float16(3.5)], dtype=object)
    arr_float = np.array([1.0, 2.5, 3.5], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Object array with numpy floats should equal float64 array"


def test_empty_object_array_vs_empty_float_array():
    """Test that empty object array equals empty float array"""
    # Create empty arrays
    arr_obj = np.array([], dtype=object)
    arr_float = np.array([], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Empty object array should equal empty float array"


def test_multidimensional_object_array():
    """Test multidimensional object array with floats equals float64 array"""
    # Create 2D object array with float elements
    arr_obj = np.array([[1.0, 2.5], [3.7, 4.2]], dtype=object)
    arr_float = np.array([[1.0, 2.5], [3.7, 4.2]], dtype=np.float64)

    differ = Diff()
    a = {"arr": arr_obj}
    b = {"arr": arr_float}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "Multidimensional object array with floats should equal float64 array"


def test_reversed_comparison():
    """Test that comparison works in both directions (float64 first, object second)"""
    # Create arrays
    arr_float = np.array([1.0, 2.5, 3.7], dtype=np.float64)
    arr_obj = np.array([1.0, 2.5, 3.7], dtype=object)

    differ = Diff()
    a = {"arr": arr_float}
    b = {"arr": arr_obj}
    result = differ.diff(a, b)

    assert len(result.differences) == 0, "float64 array should equal object array with floats (reversed order)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
