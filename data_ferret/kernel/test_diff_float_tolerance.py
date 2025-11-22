"""
Test cases for float tolerance bug in diff.py.

The bug: When a pandas Series has object dtype but contains float values,
the comparison logic doesn't apply rtol/atol tolerance because it only checks
the Series dtype, not the individual value types.
"""

import pytest
import numpy as np
import pandas as pd
from data_ferret.kernel.diff import Diff
from data_ferret.kernel.types import ValueComparison


class TestFloatToleranceBug:
    """Test cases for the float tolerance bug."""

    def test_series_float_dtype_uses_tolerance(self):
        """
        PASSING TEST: Series with float64 dtype correctly applies tolerance.
        """
        # Create series with explicit float dtype
        s1 = pd.Series([251.42894439346296], index=[94309], dtype=np.float64)
        s2 = pd.Series([251.42894439346298], index=[94309], dtype=np.float64)

        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_series(s1, s2, path="test")

        # Should be None (equal within tolerance) or have status='close'
        if result is not None:
            assert isinstance(result, dict)
            for key, comp in result.items():
                if isinstance(comp, ValueComparison):
                    # Should be 'close', not 'different'
                    assert comp.status == 'close', f"Expected 'close' but got '{comp.status}': {comp.message}"

    def test_series_object_dtype_with_floats_BUG(self):
        """
        FAILING TEST: Series with object dtype containing floats fails to apply tolerance.

        This reproduces the bug from the stack trace where a Series containing
        float values but with object dtype uses != comparison instead of tolerance.
        """
        # Create series with object dtype but float values
        s1 = pd.Series([251.42894439346296], index=[94309], dtype=object)
        s2 = pd.Series([251.42894439346298], index=[94309], dtype=object)

        # Verify they are actually floats
        assert isinstance(s1.iloc[0], float)
        assert isinstance(s2.iloc[0], float)

        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_series(s1, s2, path="test")

        # BUG: result will show 'different' even though values are within tolerance
        if result is not None:
            assert isinstance(result, dict)
            for key, comp in result.items():
                if isinstance(comp, ValueComparison):
                    # BUG: This will be 'different' when it should be 'close' or None
                    print(f"Status: {comp.status}")
                    print(f"Message: {comp.message}")
                    if comp.status == 'different':
                        pytest.fail(f"BUG DETECTED: Float values within tolerance marked as 'different':\n{comp.message}")

    def test_dataframe_object_column_with_floats_BUG(self):
        """
        FAILING TEST: DataFrame with object dtype column containing floats.

        This is the exact scenario from the stack trace.
        """
        # Create DataFrame where the column has object dtype but contains floats
        df1 = pd.DataFrame({'ratio': pd.Series([251.42894439346296], index=[94309], dtype=object)})
        df2 = pd.DataFrame({'ratio': pd.Series([251.42894439346298], index=[94309], dtype=object)})

        # Verify dtype is object
        assert df1['ratio'].dtype == object
        assert df2['ratio'].dtype == object

        # Verify values are floats
        assert isinstance(df1['ratio'].iloc[0], float)
        assert isinstance(df2['ratio'].iloc[0], float)

        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_dataframe(df1, df2, path="df")

        # BUG: result will show 'different' even though values are within tolerance
        if result is not None:
            assert isinstance(result, dict)
            has_bug = False
            for key, comp in result.items():
                if isinstance(comp, ValueComparison):
                    if comp.status == 'different' and 'TOLERANCE BUG DETECTED' in comp.message:
                        has_bug = True
                        print(f"\nBUG DETECTED in key '{key}':")
                        print(comp.message)

            if has_bug:
                pytest.fail("Float tolerance bug detected in DataFrame comparison")

    def test_mixed_types_in_object_series(self):
        """
        Test Series with mixed types (some floats, some ints, some strings).

        This tests whether the fix handles mixed-type object Series correctly.
        """
        s1 = pd.Series([1.0000001, 2, "hello"], dtype=object)
        s2 = pd.Series([1.0000002, 2, "hello"], dtype=object)

        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_series(s1, s2, path="test")

        # First element: floats within tolerance (should be close or None)
        # Second element: ints, should be equal
        # Third element: strings, should be equal
        if result is not None:
            print(f"Result keys: {result.keys()}")
            for key, comp in result.items():
                if isinstance(comp, ValueComparison):
                    print(f"  {key}: status={comp.status}, message={comp.message[:100]}")

    def test_dtype_not_float_but_values_are_floats(self):
        """
        Test the root cause: dtype check vs value type check.

        This demonstrates that pd.api.types.is_float_dtype() returns False
        for object dtype even when all values are floats.
        """
        s = pd.Series([1.5, 2.5, 3.5], dtype=object)

        # This returns False even though all values are floats
        assert not pd.api.types.is_float_dtype(s.dtype)

        # But all values are actually floats
        assert all(isinstance(val, float) for val in s)

        print(f"Series dtype: {s.dtype}")
        print(f"is_float_dtype: {pd.api.types.is_float_dtype(s.dtype)}")
        print(f"Values are floats: {[isinstance(val, float) for val in s]}")

    def test_individual_value_comparison_with_tolerance(self):
        """
        Test that individual float values should use tolerance even in object Series.
        """
        val_a = 251.42894439346296
        val_b = 251.42894439346298

        # These are floats
        assert isinstance(val_a, float)
        assert isinstance(val_b, float)

        # They're different with ==
        assert val_a != val_b

        # But they're within tolerance
        diff = abs(val_a - val_b)
        threshold = 1e-5 + 1e-5 * abs(val_b)
        assert diff <= threshold, f"diff={diff}, threshold={threshold}"

        # Test with Diff._compare_float
        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_float(val_a, val_b, "test")

        if result is not None:
            print(f"_compare_float result: status={result.status}, message={result.message}")
            # Should be 'close' or None (depending on report_close setting)
            if result.status == 'different':
                pytest.fail(f"_compare_float incorrectly marked values as different: {result.message}")

    def test_compare_values_dispatches_to_compare_float(self):
        """
        Test that _compare_values correctly dispatches to _compare_float for floats.
        """
        val_a = 251.42894439346296
        val_b = 251.42894439346298

        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_values(val_a, val_b, "test")

        if result is not None:
            print(f"_compare_values result: status={result.status}")
            if hasattr(result, 'message'):
                print(f"  message: {result.message[:100]}")

    def test_ndarray_object_dtype_with_floats(self):
        """
        Test that numpy arrays with object dtype but containing floats apply tolerance.

        This is less common than pandas Series with object dtype, but can happen.
        """
        # Create arrays with object dtype but float values
        arr1 = np.array([251.42894439346296, 100.0], dtype=object)
        arr2 = np.array([251.42894439346298, 100.0], dtype=object)

        # Verify dtype is object
        assert arr1.dtype == object
        assert arr2.dtype == object

        # Verify values are floats
        assert isinstance(arr1[0], float)
        assert isinstance(arr2[0], float)

        differ = Diff(rtol=1e-5, atol=1e-5)
        result = differ._compare_ndarray(arr1, arr2, "test")

        # Should apply tolerance to float values
        if result is not None:
            print(f"Result keys: {result.keys() if isinstance(result, dict) else 'not dict'}")
            for key, comp in result.items() if isinstance(result, dict) else []:
                if isinstance(comp, ValueComparison):
                    print(f"  {key}: status={comp.status}")
                    # Should be 'close' not 'different'
                    if '[0]' in key:  # First element should be close
                        assert comp.status == 'close', f"Expected 'close' but got '{comp.status}'"


def run_diagnostic():
    """
    Run diagnostic to understand the bug.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: Float Tolerance Bug in diff.py")
    print("="*80)

    # Create the problematic scenario
    s1 = pd.Series([251.42894439346296], index=[94309], dtype=object)
    s2 = pd.Series([251.42894439346298], index=[94309], dtype=object)

    print(f"\nSeries 1 dtype: {s1.dtype}")
    print(f"Series 2 dtype: {s2.dtype}")
    print(f"Value 1 type: {type(s1.iloc[0])}")
    print(f"Value 2 type: {type(s2.iloc[0])}")
    print(f"Value 1: {s1.iloc[0]}")
    print(f"Value 2: {s2.iloc[0]}")

    print(f"\npd.api.types.is_float_dtype(s1.dtype): {pd.api.types.is_float_dtype(s1.dtype)}")

    val_a = s1.iloc[0]
    val_b = s2.iloc[0]
    diff = abs(val_a - val_b)
    threshold = 1e-5 + 1e-5 * abs(val_b)

    print(f"\nDifference: {diff:.2e}")
    print(f"Threshold (rtol=1e-5, atol=1e-5): {threshold:.2e}")
    print(f"Within tolerance: {diff <= threshold}")

    print("\nTesting _compare_series logic path:")
    differ = Diff(rtol=1e-5, atol=1e-5)

    # Check which branch the code takes
    if pd.api.types.is_float_dtype(s1.dtype):
        print("  -> Takes FLOAT branch (line 908-956)")
        print("  -> Uses np.allclose with tolerance")
    else:
        print("  -> Takes NON-FLOAT branch (line 957-975)")
        print("  -> Uses direct != comparison (BUG!)")

    result = differ._compare_series(s1, s2, path="df['ratio']")

    if result is None:
        print("\nResult: None (series considered equal)")
    else:
        print(f"\nResult: {type(result)}")
        for key, comp in result.items():
            if isinstance(comp, ValueComparison):
                print(f"  Key: {key}")
                print(f"  Status: {comp.status}")
                print(f"  Message preview: {comp.message[:200]}")
                if 'TOLERANCE BUG DETECTED' in comp.message:
                    print("\n  *** BUG CONFIRMED ***")


if __name__ == "__main__":
    import sys
    # Move to parent directory to avoid circular import with types.py
    sys.path.insert(0, '/Users/freund/other/DataFerret')
    run_diagnostic()

    # Run the tests
    print("\n\nRunning pytest tests...")
    pytest.main([__file__, "-v", "-s"])
