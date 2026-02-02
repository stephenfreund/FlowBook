"""
Test extension dtype handling in checkpoint and diff systems.

These tests ensure that pandas extension dtypes (StringDtype, Int64, etc.)
work correctly when:
1. Converting during deepcopy
2. Comparing in diff
3. Checkpointing and restoring
"""
import pytest
import pandas as pd
import numpy as np
from flowbook.kernel_support.checkpoint import Checkpoints, Checkpoint
from flowbook.kernel_support.diff import Diff


class TestExtensionDtypeDeepCopy:
    """Test that deepcopy correctly handles and converts extension dtypes."""

    def test_string_dtype_conversion(self):
        """Test converting object strings to StringDtype during deepcopy."""
        checkpoints = Checkpoints()

        # Object dtype with strings (may be object or StringDtype depending on pandas version)
        df = pd.DataFrame({'country': ['USA', 'UK', 'FR']})
        assert pd.api.types.is_string_dtype(df['country'].dtype)

        user_ns = {'df': df}
        checkpoints.save('v1', user_ns)
        cp = checkpoints.get('v1')

        # Should be converted to StringDtype
        assert pd.api.types.is_string_dtype(cp.user_ns['df']['country'].dtype)

    def test_int64_dtype_conversion(self):
        """Test converting object integers to Int64 during deepcopy."""
        checkpoints = Checkpoints()

        # Object dtype with integers (no None, which would make it float)
        df = pd.DataFrame({'numbers': pd.array([1, 2, 3], dtype=object)})
        assert df['numbers'].dtype == object

        user_ns = {'df': df}
        checkpoints.save('v1', user_ns)
        cp = checkpoints.get('v1')

        # Should be converted to Int64
        assert pd.api.types.is_integer_dtype(cp.user_ns['df']['numbers'].dtype)

    def test_datetime_dtype_conversion(self):
        """Test converting object timestamps to datetime64 during deepcopy."""
        checkpoints = Checkpoints()

        # Object dtype with Timestamps
        df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
        })

        user_ns = {'df': df}
        checkpoints.save('v1', user_ns)
        cp = checkpoints.get('v1')

        # Should be converted to datetime64
        assert pd.api.types.is_datetime64_any_dtype(cp.user_ns['df']['date'].dtype)

    def test_nested_dataframe_conversion(self):
        """Test that DataFrames nested in lists also get converted."""
        checkpoints = Checkpoints()

        df = pd.DataFrame({'country': ['USA', 'UK']})
        # List of DataFrames - all should be converted
        user_ns = {'dfs': [df, df, df]}

        checkpoints.save('v1', user_ns)
        cp = checkpoints.get('v1')

        # All nested DataFrames should have StringDtype
        for nested_df in cp.user_ns['dfs']:
            assert pd.api.types.is_string_dtype(nested_df['country'].dtype)


class TestExtensionDtypeDiff:
    """Test that diff handles extension dtypes correctly."""

    def test_string_dtype_compatible_with_object(self):
        """StringDtype and object dtype should be compatible for diff."""
        differ = Diff()

        # One with object, one with StringDtype
        df_object = pd.DataFrame({'country': ['USA', 'UK']})
        df_string = pd.DataFrame({'country': pd.array(['USA', 'UK'], dtype='string')})

        # Should not show as different
        result = differ.diff({'df': df_object}, {'df': df_string})
        assert result.differences == {}

    def test_int64_dtype_compatible_with_int(self):
        """Int64 and int dtype should be compatible for diff."""
        differ = Diff()

        # One with int, one with Int64
        df_int = pd.DataFrame({'numbers': [1, 2, 3]})
        df_int64 = pd.DataFrame({'numbers': pd.array([1, 2, 3], dtype='Int64')})

        # Should not show as different (compatible dtypes)
        result = differ.diff({'df': df_int}, {'df': df_int64})
        assert result.differences == {}

    def test_string_dtype_different_values(self):
        """StringDtype columns with different values should show as different."""
        differ = Diff()

        df1 = pd.DataFrame({'country': pd.array(['USA', 'UK'], dtype='string')})
        df2 = pd.DataFrame({'country': pd.array(['USA', 'FR'], dtype='string')})

        result = differ.diff({'df': df1}, {'df': df2})
        assert 'df' in result.differences

    def test_array_with_extension_dtype(self):
        """Test comparing numpy arrays vs pandas arrays with extension dtypes."""
        differ = Diff()

        # Regular numpy array vs pandas array with StringDtype
        arr1 = np.array(['a', 'b', 'c'])
        arr2 = pd.array(['a', 'b', 'c'], dtype='string')

        # These have different types, so should be different
        result = differ.diff({'arr': arr1}, {'arr': arr2})
        # Could be equal or different depending on type strictness

    def test_series_with_extension_dtype(self):
        """Test comparing Series with extension dtypes."""
        differ = Diff()

        # Both StringDtype, same values
        s1 = pd.Series(pd.array(['a', 'b', 'c'], dtype='string'))
        s2 = pd.Series(pd.array(['a', 'b', 'c'], dtype='string'))

        result = differ.diff({'s': s1}, {'s': s2})
        assert result.differences == {}

    def test_dataframe_column_extension_dtype_no_crash(self):
        """Ensure DataFrame comparison doesn't crash with extension dtypes."""
        differ = Diff()

        # Mixed dtypes in DataFrame
        df1 = pd.DataFrame({
            'country': pd.array(['USA', 'UK'], dtype='string'),
            'value': [1.0, 2.0]
        })
        df2 = pd.DataFrame({
            'country': pd.array(['USA', 'UK'], dtype='string'),
            'value': [1.0, 2.0]
        })

        # Should not crash
        result = differ.diff({'df': df1}, {'df': df2})
        assert result.differences == {}

    def test_mixed_extension_and_numpy_dtypes(self):
        """Test DataFrame with mix of extension and numpy dtypes."""
        differ = Diff()

        df1 = pd.DataFrame({
            'strings': pd.array(['a', 'b'], dtype='string'),
            'ints': pd.array([1, 2], dtype='Int64'),
            'floats': [1.0, 2.0]  # Regular numpy dtype
        })
        df2 = pd.DataFrame({
            'strings': pd.array(['a', 'b'], dtype='string'),
            'ints': pd.array([1, 2], dtype='Int64'),
            'floats': [1.0, 2.0]
        })

        result = differ.diff({'df': df1}, {'df': df2})
        assert result.differences == {}


class TestExtensionDtypeCheckpointIntegration:
    """Integration tests for extension dtypes in full checkpoint workflow."""

    def test_save_restore_with_extension_dtypes(self):
        """Test full save/restore cycle with extension dtypes."""
        checkpoints = Checkpoints()

        # Create DataFrame with various dtypes
        df = pd.DataFrame({
            'country': ['USA', 'UK', 'FR'],  # Will convert to string
            'numbers': [1, 2, None],  # Will convert to Int64
            'values': [1.0, 2.0, 3.0]  # Stays float64
        })

        user_ns = {'df': df}
        checkpoints.save('v1', user_ns)

        # Modify
        user_ns['df'].loc[0, 'country'] = 'MODIFIED'

        # Restore
        checkpoints.restore('v1', user_ns)

        # Should be restored to original
        assert user_ns['df'].loc[0, 'country'] == 'USA'

    def test_checkpoint_diff_with_extension_dtypes(self):
        """Test checkpoint diffing with extension dtypes."""
        checkpoints = Checkpoints()

        df1 = pd.DataFrame({'country': ['USA', 'UK']})
        df2 = pd.DataFrame({'country': ['USA', 'FR']})

        user_ns1 = {'df': df1}
        user_ns2 = {'df': df2}

        checkpoints.save('v1', user_ns1)
        checkpoints.save('v2', user_ns2)

        cp1 = checkpoints.get('v1')
        cp2 = checkpoints.get('v2')

        # Should not crash
        diff = Checkpoint.diff(cp1, cp2)
        # Should show difference in country column
        assert 'df' in diff.differences

    def test_nested_dataframes_with_extension_dtypes(self):
        """Test nested DataFrames all get converted properly."""
        checkpoints = Checkpoints()

        df = pd.DataFrame({'country': ['USA', 'UK']})
        user_ns = {'dfs': [df.copy(), df.copy(), df.copy()]}

        checkpoints.save('v1', user_ns)

        # Modify original
        df.loc[0, 'country'] = 'MODIFIED'

        # Restore
        checkpoints.restore('v1', user_ns)

        # All nested DataFrames should be unmodified
        for nested_df in user_ns['dfs']:
            assert nested_df.loc[0, 'country'] == 'USA'
            # And should have StringDtype
            assert pd.api.types.is_string_dtype(nested_df['country'].dtype)


class TestExtensionDtypeCompatibility:
    """Test are_compatible_dtypes function with extension dtypes."""

    def test_string_dtype_variants_compatible(self):
        """Test that different string dtype representations are compatible."""
        from flowbook.kernel_support.diff import are_compatible_dtypes

        s_object = pd.Series(['a', 'b'])
        s_string = pd.Series(pd.array(['a', 'b'], dtype='string'))

        assert are_compatible_dtypes(s_object, s_string)

    def test_integer_dtype_variants_compatible(self):
        """Test that different integer dtype representations are compatible."""
        from flowbook.kernel_support.diff import are_compatible_dtypes

        s_int32 = pd.Series([1, 2, 3], dtype='int32')
        s_int64 = pd.Series([1, 2, 3], dtype='int64')
        s_Int64 = pd.Series(pd.array([1, 2, 3], dtype='Int64'))

        assert are_compatible_dtypes(s_int32, s_int64)
        assert are_compatible_dtypes(s_int64, s_Int64)
        assert are_compatible_dtypes(s_int32, s_Int64)

    def test_float_dtype_variants_compatible(self):
        """Test that different float dtype representations are compatible."""
        from flowbook.kernel_support.diff import are_compatible_dtypes

        s_float32 = pd.Series([1.0, 2.0], dtype='float32')
        s_float64 = pd.Series([1.0, 2.0], dtype='float64')

        assert are_compatible_dtypes(s_float32, s_float64)

    def test_incompatible_extension_dtypes(self):
        """Test that incompatible extension dtypes are detected."""
        from flowbook.kernel_support.diff import are_compatible_dtypes

        s_string = pd.Series(pd.array(['a', 'b'], dtype='string'))
        s_int = pd.Series(pd.array([1, 2], dtype='Int64'))

        assert not are_compatible_dtypes(s_string, s_int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
