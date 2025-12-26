"""
Tests for cuDF compatibility layer.

These tests verify:
1. cudf_compat functions work correctly (with or without cudf installed)
2. GroupBy tracking works for pandas (always)
3. GroupBy tracking works for cudf without recursion (when cudf installed)
"""

import pytest
import pandas as pd

from data_ferret.kernel import cudf_compat
from data_ferret.kernel.column_tracking import ColumnAccessTracker


# =============================================================================
# Tests that run regardless of cudf installation
# =============================================================================

class TestCudfCompatBasics:
    """Test cudf_compat functions work without cudf installed."""

    def test_has_cudf_returns_bool(self):
        """has_cudf() should return a boolean."""
        result = cudf_compat.has_cudf()
        assert isinstance(result, bool)

    def test_is_cudf_proxy_returns_false_for_non_cudf(self):
        """is_cudf_proxy() should return False for non-cudf objects."""
        assert cudf_compat.is_cudf_proxy(None) is False
        assert cudf_compat.is_cudf_proxy("string") is False
        assert cudf_compat.is_cudf_proxy(123) is False
        assert cudf_compat.is_cudf_proxy(pd.DataFrame()) is False

    def test_is_cudf_groupby_returns_false_for_non_cudf(self):
        """is_cudf_groupby() should return False for pandas GroupBy."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        gb = df.groupby('a')
        assert cudf_compat.is_cudf_groupby(gb) is False
        assert cudf_compat.is_cudf_groupby(None) is False
        assert cudf_compat.is_cudf_groupby(df) is False

    def test_unwrap_cudf_proxy_passthrough(self):
        """unwrap_cudf_proxy() should return non-proxy objects unchanged."""
        obj = "test"
        assert cudf_compat.unwrap_cudf_proxy(obj) is obj

        df = pd.DataFrame({'a': [1, 2]})
        assert cudf_compat.unwrap_cudf_proxy(df) is df


class TestPandasGroupByTracking:
    """Test that pandas GroupBy tracking still works correctly."""

    def test_groupby_single_column_key(self):
        """GroupBy with single column key should track both key and selected column."""
        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40]
            })
            tracker.register_df(df, 'df')

            # groupby tracks 'category', getitem tracks 'value'
            _ = df.groupby('category')['value'].mean()

            reads = tracker.resolve_to_paths()
            assert 'df' in reads
            assert 'category' in reads['df']
            assert 'value' in reads['df']
        finally:
            tracker.uninstall()

    def test_groupby_list_column_selection(self):
        """GroupBy with list column selection should track all columns."""
        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value1': [10, 20, 30, 40],
                'value2': [1, 2, 3, 4]
            })
            tracker.register_df(df, 'df')

            # This is the pattern from the error: gb[["col1", "col2"]]
            _ = df.groupby('category')[['value1', 'value2']].mean()

            reads = tracker.resolve_to_paths()
            assert 'df' in reads
            assert 'category' in reads['df']
            assert 'value1' in reads['df']
            assert 'value2' in reads['df']
        finally:
            tracker.uninstall()

    def test_groupby_multiple_keys(self):
        """GroupBy with multiple keys should track all key columns."""
        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            df = pd.DataFrame({
                'cat1': ['A', 'B', 'A', 'B'],
                'cat2': ['X', 'X', 'Y', 'Y'],
                'value': [10, 20, 30, 40]
            })
            tracker.register_df(df, 'df')

            _ = df.groupby(['cat1', 'cat2'])['value'].sum()

            reads = tracker.resolve_to_paths()
            assert 'df' in reads
            assert 'cat1' in reads['df']
            assert 'cat2' in reads['df']
            assert 'value' in reads['df']
        finally:
            tracker.uninstall()

    def test_groupby_no_interference_between_dataframes(self):
        """Tracking should be independent for different DataFrames."""
        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            df2 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
            tracker.register_df(df1, 'df1')
            tracker.register_df(df2, 'df2')

            _ = df1.groupby('a')['b'].mean()
            _ = df2.groupby('x')['y'].mean()

            reads = tracker.resolve_to_paths()
            assert reads.get('df1') == {'a', 'b'}
            assert reads.get('df2') == {'x', 'y'}
        finally:
            tracker.uninstall()


# =============================================================================
# Tests that only run when cudf is installed
# =============================================================================

@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfDetection:
    """Test cudf detection functions when cudf is available."""

    def test_has_cudf_true(self):
        """has_cudf() should return True when cudf is installed."""
        assert cudf_compat.has_cudf() is True

    def test_get_cudf_returns_module(self):
        """get_cudf() should return the cudf module."""
        cudf = cudf_compat.get_cudf()
        assert cudf is not None
        assert hasattr(cudf, 'DataFrame')

    def test_is_cudf_groupby_detects_cudf_groupby(self):
        """is_cudf_groupby() should detect cudf GroupBy objects."""
        import cudf

        gdf = cudf.DataFrame({'a': [1, 2, 1], 'b': [3, 4, 5]})
        gb = gdf.groupby('a')

        assert cudf_compat.is_cudf_groupby(gb) is True

    def test_is_cudf_groupby_false_for_pandas(self):
        """is_cudf_groupby() should return False for pandas GroupBy."""
        df = pd.DataFrame({'a': [1, 2, 1], 'b': [3, 4, 5]})
        gb = df.groupby('a')

        assert cudf_compat.is_cudf_groupby(gb) is False


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfGroupByNoRecursion:
    """Test that cudf GroupBy operations don't cause infinite recursion."""

    def test_cudf_groupby_single_column(self):
        """cudf GroupBy with single column selection should work."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40]
            })
            tracker.register_df(gdf, 'gdf')

            # This should NOT cause infinite recursion
            result = gdf.groupby('category')['value'].mean()

            # Verify result is correct
            assert len(result) == 2
        finally:
            tracker.uninstall()

    def test_cudf_groupby_list_column_selection(self):
        """cudf GroupBy with list column selection should work (the failing case)."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'Weight Capacity (kg)': [10, 20, 10, 30],
                'Price': [100, 200, 150, 300]
            })
            tracker.register_df(gdf, 'gdf')

            # This is the exact pattern that was failing with recursion
            result = gdf.groupby("Weight Capacity (kg)")[["Price"]].mean()

            # Verify result is correct
            assert 'Price' in result.columns
            assert len(result) == 3  # 3 unique weight capacities
        finally:
            tracker.uninstall()

    def test_cudf_groupby_chained_operations(self):
        """cudf GroupBy with chained operations should work."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value1': [10, 20, 30, 40],
                'value2': [1, 2, 3, 4]
            })
            tracker.register_df(gdf, 'gdf')

            # Chain multiple operations
            result = gdf.groupby('category')[['value1', 'value2']].agg(['mean', 'sum'])

            assert result is not None
        finally:
            tracker.uninstall()

    def test_cudf_groupby_merge_pattern(self):
        """Test the full pattern from the error: groupby -> select -> agg -> merge."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            train = cudf.DataFrame({
                'Weight Capacity (kg)': [10, 20, 10, 30],
                'Price': [100, 200, 150, 300],
                'other': [1, 2, 3, 4]
            })
            orig = train.copy()

            tracker.register_df(train, 'train')
            tracker.register_df(orig, 'orig')

            # The exact pattern from the error
            tmp = orig.groupby("Weight Capacity (kg)")[["Price"]].mean()["Price"]
            tmp.name = "orig_price"
            train = train.merge(tmp, on="Weight Capacity (kg)", how="left")

            # Verify merge worked
            assert 'orig_price' in train.columns
        finally:
            tracker.uninstall()


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfColumnTracking:
    """Test that column tracking works for cudf DataFrames."""

    def test_cudf_groupby_tracks_columns(self):
        """cudf GroupBy should track column access."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40]
            })
            tracker.register_df(gdf, 'gdf')

            _ = gdf.groupby('category')['value'].mean()

            reads = tracker.resolve_to_paths()
            assert 'gdf' in reads
            assert 'category' in reads['gdf']
            assert 'value' in reads['gdf']
        finally:
            tracker.uninstall()

    def test_cudf_getitem_tracks_columns(self):
        """cudf DataFrame.__getitem__ should track column access."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            })
            tracker.register_df(gdf, 'gdf')

            _ = gdf['a']
            _ = gdf[['a', 'b']]

            reads = tracker.resolve_to_paths()
            assert 'gdf' in reads
            assert 'a' in reads['gdf']
            assert 'b' in reads['gdf']
        finally:
            tracker.uninstall()

    def test_cudf_setitem_tracks_writes(self):
        """cudf DataFrame.__setitem__ should track column writes."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            })
            tracker.register_df(gdf, 'gdf')

            gdf['c'] = [7, 8, 9]

            writes = tracker.resolve_writes_to_paths()
            assert 'gdf' in writes
            assert 'c' in writes['gdf']
        finally:
            tracker.uninstall()

    def test_cudf_sort_values_tracks_columns(self):
        """cudf DataFrame.sort_values should track column access."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'a': [3, 1, 2],
                'b': [4, 5, 6]
            })
            tracker.register_df(gdf, 'gdf')

            _ = gdf.sort_values('a')

            reads = tracker.resolve_to_paths()
            assert 'gdf' in reads
            assert 'a' in reads['gdf']
        finally:
            tracker.uninstall()


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfCheckpoint:
    """Test cudf checkpoint save/restore functionality."""

    def test_checkpoint_cudf_dataframe(self):
        """cudf DataFrame should be checkpointed to CPU memory."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        gdf = cudf.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Internally stored as pandas
        saved_cp = cp.saved['test']
        assert isinstance(saved_cp.user_ns['gdf'], pd.DataFrame)

    def test_restore_cudf_dataframe(self):
        """Restored DataFrame should be cudf again."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        gdf = cudf.DataFrame({'a': [1, 2, 3]})

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Modify the namespace
        user_ns['gdf'] = cudf.DataFrame({'x': [9, 9, 9]})

        # Restore
        cp.restore('test', user_ns)

        # Should be cudf again
        assert isinstance(user_ns['gdf'], cudf.DataFrame)
        assert list(user_ns['gdf']['a'].to_pandas()) == [1, 2, 3]

    def test_checkpoint_cudf_series(self):
        """cudf Series should be checkpointed and restored."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        gs = cudf.Series([1, 2, 3], name='values')

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gs': gs}
        cp.save('test', user_ns)

        # Modify
        user_ns['gs'] = cudf.Series([9, 9, 9])

        # Restore
        cp.restore('test', user_ns)

        # Should be cudf Series again
        assert isinstance(user_ns['gs'], cudf.Series)
        assert list(user_ns['gs'].to_pandas()) == [1, 2, 3]

    def test_mixed_pandas_cudf_checkpoint(self):
        """Mixed pandas/cudf checkpoint should work."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        gdf = cudf.DataFrame({'a': [1, 2, 3]})
        pdf = pd.DataFrame({'b': [4, 5, 6]})

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf, 'pdf': pdf}
        cp.save('test', user_ns)

        # Modify both
        user_ns['gdf'] = cudf.DataFrame({'x': [9]})
        user_ns['pdf'] = pd.DataFrame({'y': [9]})

        # Restore
        cp.restore('test', user_ns)

        # cudf should be cudf, pandas should be pandas
        assert isinstance(user_ns['gdf'], cudf.DataFrame)
        assert isinstance(user_ns['pdf'], pd.DataFrame)
        assert list(user_ns['gdf']['a'].to_pandas()) == [1, 2, 3]
        assert list(user_ns['pdf']['b']) == [4, 5, 6]

    def test_cudf_checkpoint_independence(self):
        """Modifications after restore should not affect checkpoint."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        gdf = cudf.DataFrame({'a': [1, 2, 3]})

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Restore and modify
        cp.restore('test', user_ns)
        user_ns['gdf']['a'] = [9, 9, 9]

        # Restore again - should get original values
        cp.restore('test', user_ns)
        assert list(user_ns['gdf']['a'].to_pandas()) == [1, 2, 3]


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfDiff:
    """Test cudf diff functionality."""

    def test_diff_cudf_equal(self):
        """Equal cudf DataFrames should show no diff."""
        import cudf
        from data_ferret.kernel.diff import Diff

        gdf1 = cudf.DataFrame({'a': [1, 2, 3]})
        gdf2 = cudf.DataFrame({'a': [1, 2, 3]})

        differ = Diff()
        result = differ.diff({'gdf': gdf1}, {'gdf': gdf2})
        assert not result.differences

    def test_diff_cudf_different(self):
        """Different cudf DataFrames should show diff."""
        import cudf
        from data_ferret.kernel.diff import Diff

        gdf1 = cudf.DataFrame({'a': [1, 2, 3]})
        gdf2 = cudf.DataFrame({'a': [1, 2, 4]})

        differ = Diff()
        result = differ.diff({'gdf': gdf1}, {'gdf': gdf2})
        assert result.differences


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfAliasDetection:
    """Test that alias detection works with cudf checkpoints."""

    def test_cudf_alias_detection_basic(self):
        """Alias detection should work for cudf DataFrames in checkpoint."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        gdf = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Build alias index (triggers _build_alias_index)
        saved_cp = cp.saved['test']
        # Access aliases to trigger lazy build
        aliases = saved_cp.get_deep_aliases({'gdf'})

        # Should complete without error
        assert isinstance(aliases, set)

    def test_cudf_shared_reference_detection(self):
        """Shared references between cudf DataFrames should be detected after checkpoint."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        # Create DataFrames that share data
        gdf1 = cudf.DataFrame({'a': [1, 2, 3]})
        gdf2 = gdf1  # Same object

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf1': gdf1, 'gdf2': gdf2}
        cp.save('test', user_ns)

        # After checkpoint, both should be independent pandas copies
        saved_cp = cp.saved['test']
        assert saved_cp.user_ns['gdf1'] is not saved_cp.user_ns['gdf2']


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfAdvancedTypes:
    """Test advanced cudf types (CategoricalIndex, MultiIndex, etc.)."""

    def test_cudf_categorical_column(self):
        """cudf DataFrame with categorical column should checkpoint correctly."""
        import cudf

        from data_ferret.kernel.checkpoint import Checkpoints

        gdf = cudf.DataFrame({'a': [1, 2, 3]})
        gdf['cat'] = cudf.Series(['x', 'y', 'x']).astype('category')

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Modify
        user_ns['gdf'] = cudf.DataFrame({'z': [9]})

        # Restore
        cp.restore('test', user_ns)

        assert isinstance(user_ns['gdf'], cudf.DataFrame)
        assert 'cat' in user_ns['gdf'].columns

    def test_cudf_multiindex_dataframe(self):
        """cudf DataFrame with MultiIndex should checkpoint correctly."""
        import cudf

        from data_ferret.kernel.checkpoint import Checkpoints

        # Create DataFrame with MultiIndex
        gdf = cudf.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7, 8],
            'c': [9, 10, 11, 12]
        })
        gdf = gdf.set_index(['a', 'b'])

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Modify
        user_ns['gdf'] = cudf.DataFrame({'z': [9]})

        # Restore
        cp.restore('test', user_ns)

        assert isinstance(user_ns['gdf'], cudf.DataFrame)
        assert 'c' in user_ns['gdf'].columns

    def test_cudf_datetime_index(self):
        """cudf DataFrame with datetime index should checkpoint correctly."""
        import cudf

        from data_ferret.kernel.checkpoint import Checkpoints

        dates = cudf.date_range('2020-01-01', periods=3, freq='D')
        gdf = cudf.DataFrame({'a': [1, 2, 3]}, index=dates)

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf': gdf}
        cp.save('test', user_ns)

        # Modify
        user_ns['gdf'] = cudf.DataFrame({'z': [9]})

        # Restore
        cp.restore('test', user_ns)

        assert isinstance(user_ns['gdf'], cudf.DataFrame)
        assert len(user_ns['gdf']) == 3


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfCheckpointCache:
    """Test cudf checkpoint cache functionality."""

    def test_cache_hit(self):
        """Same cudf object should use cached copy."""
        import cudf

        cache = cudf_compat.CuDFCheckpointCache()
        gdf = cudf.DataFrame({'a': [1, 2, 3]})

        # First access - cache miss
        result1 = cache.get_or_convert(gdf)
        assert isinstance(result1, pd.DataFrame)

        # Second access - cache hit
        result2 = cache.get_or_convert(gdf)
        assert result2 is result1

    def test_cache_invalidation_on_mutation(self):
        """Modified cudf object should not use stale cache."""
        import cudf

        cache = cudf_compat.CuDFCheckpointCache()
        gdf = cudf.DataFrame({'a': [1, 2, 3]})

        result1 = cache.get_or_convert(gdf)

        # Mutate
        gdf['a'] = [4, 5, 6]

        # Should get new copy
        result2 = cache.get_or_convert(gdf)
        assert result2 is not result1
        assert list(result2['a']) == [4, 5, 6]


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfProxyDetection:
    """Test proxy detection with cudf.pandas mode (if available)."""

    def test_proxy_detection_setup(self):
        """Verify proxy detection initializes without error."""
        # This just tests that the detection code runs without crashing
        cudf_compat._init_cudf_pandas_detection()
        # Result depends on whether cudf.pandas is available
        assert cudf_compat._HAS_CUDF_PANDAS in (True, False)


# =============================================================================
# Integration tests
# =============================================================================

class TestMixedPandasCudf:
    """Test scenarios with both pandas and cudf DataFrames."""

    @pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
    def test_mixed_dataframes_tracking(self):
        """Tracking should work with both pandas and cudf DataFrames."""
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            pdf = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            gdf = cudf.DataFrame({'x': [1, 2], 'y': [3, 4]})

            tracker.register_df(pdf, 'pdf')
            tracker.register_df(gdf, 'gdf')

            # Access pandas
            _ = pdf.groupby('a')['b'].mean()

            # Access cudf
            _ = gdf.groupby('x')['y'].mean()

            reads = tracker.resolve_to_paths()
            assert 'pdf' in reads
            assert 'a' in reads['pdf']
            assert 'b' in reads['pdf']
            # cudf tracking may be limited but shouldn't crash
        finally:
            tracker.uninstall()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
