"""Test object dtype to specialized type conversion in checkpoint.py"""

import pandas as pd
import numpy as np
from flowbook.kernel_support.checkpoint import (
    convert_series_object_to_specialized,
    convert_dataframe_object_to_specialized,
)
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


def test_convert_series_integers():
    """Test converting Series with integer objects to Int64."""
    # Create Series with object dtype containing integers
    s = pd.Series([1, 2, 3, None, 5], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to Int64 (nullable integer)
    assert result.dtype == pd.Int64Dtype()
    assert result.tolist() == [1, 2, 3, pd.NA, 5]
    print("✓ Integer conversion test passed")


def test_convert_series_floats():
    """Test converting Series with float objects to float64."""
    # Create Series with object dtype containing floats
    s = pd.Series([1.5, 2.3, None, 4.7], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to float64
    assert result.dtype == np.float64
    assert result[0] == 1.5
    assert result[1] == 2.3
    assert pd.isna(result[2])
    assert result[3] == 4.7
    print("✓ Float conversion test passed")


def test_convert_series_mixed_numeric():
    """Test converting Series with mixed int/float objects."""
    # Create Series with mixed int and float objects
    s = pd.Series([1, 2.5, 3, None, 4.7], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to float64
    assert result.dtype == np.float64
    assert result[0] == 1.0
    assert result[1] == 2.5
    print("✓ Mixed numeric conversion test passed")


def test_convert_series_strings_to_string_dtype():
    """Test that Series with strings are converted to string dtype."""
    # Create Series with strings
    s = pd.Series(["a", "b", "c"], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to string dtype
    assert result.dtype == pd.StringDtype()
    assert result.tolist() == ["a", "b", "c"]
    print("✓ String dtype conversion test passed")


def test_convert_dataframe():
    """Test converting DataFrame with multiple object columns."""
    # Create DataFrame with mixed object columns
    df = pd.DataFrame({
        "integers": pd.Series([1, 2, 3, None, 5], dtype=object),
        "floats": pd.Series([1.5, 2.3, 3.7, None, 5.1], dtype=object),
        "strings": pd.Series(["a", "b", "c", "d", "e"], dtype=object),
        "mixed": pd.Series([1, 2.5, 3, 4.7, 5], dtype=object),
        "normal_int": [1, 2, 3, 4, 5],  # Already int64
    })

    # Verify initial dtypes
    assert df["integers"].dtype == object
    assert df["floats"].dtype == object
    assert df["strings"].dtype == object
    assert df["mixed"].dtype == object

    result = convert_dataframe_object_to_specialized(df)

    # Check conversions
    assert result["integers"].dtype == pd.Int64Dtype()
    assert result["floats"].dtype == np.float64
    assert result["strings"].dtype == pd.StringDtype()  # Should be string dtype
    assert result["mixed"].dtype == np.float64  # Mixed int/float -> float64
    assert result["normal_int"].dtype == np.int64  # Should remain int64

    # Verify values
    assert result["integers"][0] == 1
    assert pd.isna(result["integers"][3])
    assert result["floats"][0] == 1.5
    assert result["strings"][0] == "a"

    print("✓ DataFrame conversion test passed")


def test_checkpoint_with_conversion():
    """Test that Checkpoints applies object dtype conversion."""
    checkpoints = MemoryCheckpoints()

    # Create test data with object dtypes
    user_ns = {
        "df": pd.DataFrame({
            "ints": pd.Series([1, 2, 3], dtype=object),
            "floats": pd.Series([1.5, 2.5, 3.5], dtype=object),
        }),
        "series": pd.Series([10, 20, 30], dtype=object),
    }

    # Save checkpoint
    saved, removed = checkpoints.save("test", user_ns)

    # Retrieve checkpoint
    checkpoint = checkpoints.get("test")

    # Check that conversions were applied
    assert checkpoint.user_ns["df"]["ints"].dtype == pd.Int64Dtype()
    assert checkpoint.user_ns["df"]["floats"].dtype == np.float64
    assert checkpoint.user_ns["series"].dtype == pd.Int64Dtype()

    print("✓ Checkpoint conversion test passed")


def test_in_place_modification():
    """Test that saving a checkpoint does NOT modify the original DataFrame dtypes."""
    checkpoints = MemoryCheckpoints()

    # Create test data with object dtypes
    df_original = pd.DataFrame({
        "ints": pd.Series([1, 2, 3], dtype=object),
        "strings": pd.Series(["a", "b", "c"], dtype=object),
    })

    user_ns = {"df": df_original}

    # Store the original DataFrame ID
    original_df_id = id(user_ns["df"])

    # Save checkpoint - should NOT modify the original DataFrame
    saved, removed = checkpoints.save("test", user_ns)

    # Original DataFrame dtypes must be preserved (not mutated)
    assert user_ns["df"]["ints"].dtype == object, \
        f"Expected object dtype, got {user_ns['df']['ints'].dtype}"
    assert user_ns["df"]["strings"].dtype == object, \
        f"Expected object dtype, got {user_ns['df']['strings'].dtype}"

    # The DataFrame object itself should still be the same object
    assert id(user_ns["df"]) == original_df_id

    print("✓ No in-place modification test passed")


def test_convert_decimal():
    """Test converting Series with Decimal objects to float."""
    from decimal import Decimal

    s = pd.Series([Decimal("1.5"), Decimal("2.3"), None], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to float64
    assert result.dtype == np.float64
    assert result[0] == 1.5
    assert result[1] == 2.3
    assert pd.isna(result[2])
    print("✓ Decimal conversion test passed")


def test_convert_complex():
    """Test converting Series with complex numbers (no NaN values)."""
    s = pd.Series([1+2j, 3+4j, 5+6j], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to complex128 when no NaN values
    assert result.dtype == np.complex128
    assert result[0] == 1+2j
    assert result[1] == 3+4j
    print("✓ Complex conversion test passed")


def test_convert_boolean():
    """Test converting Series with boolean objects."""
    s = pd.Series([True, False, None, True], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to boolean dtype
    assert result.dtype == pd.BooleanDtype()
    assert result[0] == True
    assert result[1] == False
    assert pd.isna(result[2])
    print("✓ Boolean conversion test passed")


def test_convert_datetime():
    """Test converting Series with datetime objects."""
    from datetime import datetime, date

    s = pd.Series([datetime(2020, 1, 1), datetime(2020, 1, 2), None], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to datetime64[ns]
    assert pd.api.types.is_datetime64_any_dtype(result.dtype)
    assert result[0] == pd.Timestamp("2020-01-01")
    assert result[1] == pd.Timestamp("2020-01-02")
    print("✓ Datetime conversion test passed")


def test_convert_timedelta():
    """Test converting Series with timedelta objects."""
    from datetime import timedelta

    s = pd.Series([timedelta(days=1), timedelta(days=2), None], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should be converted to timedelta64[ns]
    assert pd.api.types.is_timedelta64_dtype(result.dtype)
    assert result[0] == pd.Timedelta(days=1)
    assert result[1] == pd.Timedelta(days=2)
    print("✓ Timedelta conversion test passed")


def test_convert_categorical():
    """Test converting Series with categorical data."""
    # Create a categorical Series first, then convert to object
    cat_series = pd.Series(pd.Categorical(["a", "b", "a", "c", "b"]))

    # Convert to object dtype but preserve categorical nature
    # Note: When converting categorical to object, infer_dtype will see it as "string"
    # So this test verifies that regular strings are handled correctly
    s = pd.Series(["a", "b", "a", "c", "b"], dtype=object)

    result = convert_series_object_to_specialized(s)

    # Will be converted to string dtype, not categorical
    # (categorical inference happens during data loading, not from object dtype)
    assert result.dtype == pd.StringDtype()
    print("✓ Categorical conversion test passed (converted to string)")


def test_mixed_types_unchanged():
    """Test that truly mixed types remain as object."""
    # Mix of strings and numbers - should remain object
    s = pd.Series([1, "a", 2, "b"], dtype=object)
    assert s.dtype == object

    result = convert_series_object_to_specialized(s)

    # Should remain object dtype
    assert result.dtype == object
    print("✓ Mixed types preservation test passed")


if __name__ == "__main__":
    test_convert_series_integers()
    test_convert_series_floats()
    test_convert_series_mixed_numeric()
    test_convert_series_strings_to_string_dtype()
    test_convert_dataframe()
    test_checkpoint_with_conversion()
    test_in_place_modification()
    test_convert_decimal()
    test_convert_complex()
    test_convert_boolean()
    test_convert_datetime()
    test_convert_timedelta()
    test_convert_categorical()
    test_mixed_types_unchanged()
    print("\n✅ All tests passed!")
