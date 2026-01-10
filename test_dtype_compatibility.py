"""
Test dtype compatibility in diff.py
"""
import numpy as np
import pandas as pd
from flowbook.kernel.diff import Diff

def test_integer_dtypes():
    """Test that different integer dtypes are considered equal."""
    print("Testing numpy arrays with different integer dtypes...")

    # Test int32 vs int64
    a = {'arr': np.array([1, 2, 3], dtype=np.int32)}
    b = {'arr': np.array([1, 2, 3], dtype=np.int64)}

    differ = Diff()
    result = differ.diff(a, b)

    assert len(result.differences) == 0, f"Expected no differences, got: {result.differences}"
    print("✓ int32 vs int64 arrays are equal")

    # Test uint16 vs uint32
    a = {'arr': np.array([1, 2, 3], dtype=np.uint16)}
    b = {'arr': np.array([1, 2, 3], dtype=np.uint32)}

    result = differ.diff(a, b)
    assert len(result.differences) == 0, f"Expected no differences, got: {result.differences}"
    print("✓ uint16 vs uint32 arrays are equal")

def test_float_dtypes():
    """Test that different float dtypes are considered equal."""
    print("\nTesting numpy arrays with different float dtypes...")

    a = {'arr': np.array([1.5, 2.5, 3.5], dtype=np.float32)}
    b = {'arr': np.array([1.5, 2.5, 3.5], dtype=np.float64)}

    differ = Diff()
    result = differ.diff(a, b)

    assert len(result.differences) == 0, f"Expected no differences, got: {result.differences}"
    print("✓ float32 vs float64 arrays are equal")

def test_incompatible_dtypes():
    """Test that int vs float are NOT considered equal."""
    print("\nTesting incompatible dtypes...")

    a = {'arr': np.array([1, 2, 3], dtype=np.int32)}
    b = {'arr': np.array([1, 2, 3], dtype=np.float32)}

    differ = Diff()
    result = differ.diff(a, b)

    assert len(result.differences) > 0, "Expected differences for int vs float"
    assert 'dtype mismatch' in str(result.differences['arr']).lower(), \
        f"Expected dtype mismatch message, got: {result.differences['arr']}"
    print("✓ int32 vs float32 arrays are NOT equal (correct)")

def test_series_dtypes():
    """Test pandas Series with different integer dtypes."""
    print("\nTesting pandas Series with different dtypes...")

    a = {'s': pd.Series([1, 2, 3], dtype=np.int32)}
    b = {'s': pd.Series([1, 2, 3], dtype=np.int64)}

    differ = Diff()
    result = differ.diff(a, b)

    assert len(result.differences) == 0, f"Expected no differences, got: {result.differences}"
    print("✓ int32 vs int64 Series are equal")

def test_dataframe_dtypes():
    """Test pandas DataFrame with different integer dtypes."""
    print("\nTesting pandas DataFrame with different dtypes...")

    a = {'df': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, dtype=np.int32)}
    b = {'df': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, dtype=np.int64)}

    differ = Diff()
    result = differ.diff(a, b)

    assert len(result.differences) == 0, f"Expected no differences, got: {result.differences}"
    print("✓ int32 vs int64 DataFrames are equal")

def test_value_differences():
    """Test that value differences are still detected."""
    print("\nTesting that value differences are still detected...")

    a = {'arr': np.array([1, 2, 3], dtype=np.int32)}
    b = {'arr': np.array([1, 2, 4], dtype=np.int64)}

    differ = Diff()
    result = differ.diff(a, b)

    assert len(result.differences) > 0, "Expected differences for different values"
    assert 'values mismatch' in str(result.differences['arr']).lower(), \
        f"Expected values mismatch message, got: {result.differences['arr']}"
    print("✓ Arrays with different values are NOT equal (correct)")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing dtype compatibility in diff.py")
    print("=" * 60)

    test_integer_dtypes()
    test_float_dtypes()
    test_incompatible_dtypes()
    test_series_dtypes()
    test_dataframe_dtypes()
    test_value_differences()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
