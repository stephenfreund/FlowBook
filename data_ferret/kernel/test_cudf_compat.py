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
        aliases = saved_cp.get_aliases_for_vars({'gdf'})

        # Should complete without error and include the variable itself
        assert isinstance(aliases, set)
        assert 'gdf' in aliases

    def test_cudf_shared_reference_detection(self):
        """Shared references between cudf DataFrames should be detected as aliases."""
        import cudf
        from data_ferret.kernel.checkpoint import Checkpoints

        # Create DataFrames that share data (same object)
        gdf1 = cudf.DataFrame({'a': [1, 2, 3]})
        gdf2 = gdf1  # Same object

        cp = Checkpoints(sanity_check=False)
        user_ns = {'gdf1': gdf1, 'gdf2': gdf2}
        cp.save('test', user_ns)

        # After checkpoint, the memo preserves sharing - both point to same copy
        # This is correct behavior: alias detection should find them as related
        saved_cp = cp.saved['test']
        aliases = saved_cp.get_aliases_for_vars({'gdf1'})

        # Both variables should be detected as aliases of each other
        assert 'gdf1' in aliases
        assert 'gdf2' in aliases


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


# =============================================================================
# Tests for structural tracking cudf proxy unwrapping
# =============================================================================

class TestStructuralTrackingProxyUnwrap:
    """Test that structural tracking doesn't cause recursion with cudf proxies."""

    def test_unwrap_cudf_proxy_returns_non_proxy_unchanged(self):
        """_unwrap_cudf_proxy should return non-proxy objects unchanged."""
        from data_ferret.kernel.structural_tracking import _unwrap_cudf_proxy

        # Test with various non-proxy objects
        assert _unwrap_cudf_proxy(None) is None
        assert _unwrap_cudf_proxy("string") == "string"
        assert _unwrap_cudf_proxy(123) == 123

        df = pd.DataFrame({'a': [1, 2, 3]})
        assert _unwrap_cudf_proxy(df) is df

        series = pd.Series([1, 2, 3])
        assert _unwrap_cudf_proxy(series) is series

    def test_unwrap_cudf_proxy_with_mock_proxy(self):
        """_unwrap_cudf_proxy should unwrap objects with _fsproxy_slow attribute."""
        from data_ferret.kernel.structural_tracking import _unwrap_cudf_proxy

        # Create a mock proxy object
        underlying_df = pd.DataFrame({'a': [1, 2, 3]})

        class MockProxy:
            _fsproxy_slow = underlying_df

        proxy = MockProxy()
        result = _unwrap_cudf_proxy(proxy)
        assert result is underlying_df

    def test_unwrap_cudf_proxy_with_callable_fsproxy_slow(self):
        """_unwrap_cudf_proxy should handle callable _fsproxy_slow."""
        from data_ferret.kernel.structural_tracking import _unwrap_cudf_proxy

        underlying_df = pd.DataFrame({'a': [1, 2, 3]})

        class MockProxy:
            def _fsproxy_slow(self):
                return underlying_df

        proxy = MockProxy()
        result = _unwrap_cudf_proxy(proxy)
        assert result is underlying_df

    def test_structural_tracking_no_recursion_with_pandas(self):
        """Structural tracking should work without recursion for pandas objects."""
        from data_ferret.kernel.structural_tracking import StructuralAccessTracker

        tracker = StructuralAccessTracker()
        tracker.install()
        try:
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            tracker.register(df, 'df')

            # These operations should complete without recursion
            _ = df.groupby('a')['b'].mean()
            _ = df.loc[0]
            _ = df.iloc[0]

            # Verify no crash
            reads = tracker.resolve_to_paths()
            assert isinstance(reads, dict)
        finally:
            tracker.uninstall()

    def test_structural_tracking_groupby_operations(self):
        """GroupBy operations should work with structural tracking installed."""
        from data_ferret.kernel.structural_tracking import StructuralAccessTracker

        tracker = StructuralAccessTracker()
        tracker.install()
        try:
            df = pd.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40]
            })
            tracker.register(df, 'df')

            # Various groupby operations that previously could cause issues
            result1 = df.groupby('category')['value'].sum()
            result2 = df.groupby('category')[['value']].mean()
            result3 = df.groupby('category').agg({'value': 'sum'})

            assert len(result1) == 2
            assert 'value' in result2.columns
            assert 'value' in result3.columns
        finally:
            tracker.uninstall()


@pytest.mark.skipif(not cudf_compat.has_cudf(), reason="cuDF not installed")
class TestCudfStructuralTracking:
    """Test structural tracking with cudf objects."""

    def test_cudf_structural_tracking_no_recursion(self):
        """Structural tracking should not cause recursion with cudf."""
        import cudf
        from data_ferret.kernel.structural_tracking import StructuralAccessTracker

        tracker = StructuralAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({'a': [1, 2], 'b': [3, 4]})
            tracker.register(gdf, 'gdf')

            # These should complete without recursion
            _ = gdf.groupby('a')['b'].mean()

            reads = tracker.resolve_to_paths()
            assert isinstance(reads, dict)
        finally:
            tracker.uninstall()

    def test_cudf_groupby_list_selection_no_recursion(self):
        """The specific pattern that was causing recursion should work."""
        import cudf
        from data_ferret.kernel.structural_tracking import StructuralAccessTracker

        tracker = StructuralAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'Weight Capacity (kg)': [10, 20, 10, 30],
                'Price': [100, 200, 150, 300]
            })
            tracker.register(gdf, 'gdf')

            # This exact pattern was causing RecursionError
            result = gdf.groupby("Weight Capacity (kg)")[["Price"]].mean()

            assert 'Price' in result.columns
        finally:
            tracker.uninstall()


# =============================================================================
# Tests for cudf.pandas proxy detection and handling
# =============================================================================

class MockFastSlowProxy:
    """
    Mock base class that simulates cudf.pandas _FastSlowProxy behavior.

    cudf.pandas creates proxy objects that:
    - Have _fsproxy_slow attribute containing the pandas object
    - Have _fsproxy_fast attribute containing the cudf object (may be None)
    - Report their type name as the pandas type name
    """
    pass


@pytest.fixture
def mock_cudf_proxy_env():
    """
    Fixture that sets up a complete mock environment for cudf.pandas proxy testing.

    This properly mocks all cudf_compat internals so that proxy detection works
    without requiring cudf to be installed.
    """
    from data_ferret.kernel import cudf_compat

    # Save originals
    original_proxy_type = cudf_compat._cudf_pandas_proxy_type
    original_has_cudf_pandas = cudf_compat._HAS_CUDF_PANDAS
    original_has_cudf = cudf_compat.has_cudf
    original_is_cudf_proxy = cudf_compat.is_cudf_proxy
    original_is_cudf_dataframe = cudf_compat.is_cudf_dataframe
    original_is_cudf_series = cudf_compat.is_cudf_series
    original_is_cudf_index = cudf_compat.is_cudf_index

    # Set up mocks
    cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
    cudf_compat._HAS_CUDF_PANDAS = True
    cudf_compat.has_cudf = lambda: True
    cudf_compat.is_cudf_proxy = lambda obj: isinstance(obj, MockFastSlowProxy)

    # These should return False for our mock proxies (they're not native cudf)
    cudf_compat.is_cudf_dataframe = lambda obj: False
    cudf_compat.is_cudf_series = lambda obj: False
    cudf_compat.is_cudf_index = lambda obj: False

    yield cudf_compat

    # Restore originals
    cudf_compat._cudf_pandas_proxy_type = original_proxy_type
    cudf_compat._HAS_CUDF_PANDAS = original_has_cudf_pandas
    cudf_compat.has_cudf = original_has_cudf
    cudf_compat.is_cudf_proxy = original_is_cudf_proxy
    cudf_compat.is_cudf_dataframe = original_is_cudf_dataframe
    cudf_compat.is_cudf_series = original_is_cudf_series
    cudf_compat.is_cudf_index = original_is_cudf_index


class MockDataFrameProxy(MockFastSlowProxy):
    """Mock cudf.pandas DataFrame proxy."""

    def __init__(self, data):
        self._underlying_df = pd.DataFrame(data)
        self._fsproxy_slow = self._underlying_df
        self._fsproxy_fast = None  # Would be cudf.DataFrame in real proxy

    def copy(self):
        return MockDataFrameProxy(self._underlying_df.to_dict())


class MockSeriesProxy(MockFastSlowProxy):
    """Mock cudf.pandas Series proxy."""

    def __init__(self, data, name=None):
        self._underlying_series = pd.Series(data, name=name)
        self._fsproxy_slow = self._underlying_series
        self._fsproxy_fast = None

    def copy(self):
        return MockSeriesProxy(self._underlying_series.tolist(), self._underlying_series.name)


class MockIndexProxy(MockFastSlowProxy):
    """Mock cudf.pandas Index proxy."""

    def __init__(self, data):
        self._underlying_index = pd.Index(data)
        self._fsproxy_slow = self._underlying_index
        self._fsproxy_fast = None


class MockCallableFsprosySlow(MockFastSlowProxy):
    """Mock proxy where _fsproxy_slow is a callable."""

    def __init__(self, data):
        self._underlying_df = pd.DataFrame(data)

    def _fsproxy_slow(self):
        return self._underlying_df


class TestProxyDetectionHelpers:
    """Test the proxy detection helper functions."""

    def test_is_proxy_dataframe_with_mock(self):
        """_is_proxy_dataframe should detect DataFrame proxies."""
        from data_ferret.kernel import cudf_compat

        # Temporarily make our mock look like a real proxy
        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            # Create a mock that reports as DataFrame
            class DataFrameMock(MockFastSlowProxy):
                pass
            DataFrameMock.__name__ = 'DataFrame'

            proxy = DataFrameMock()
            assert cudf_compat._is_proxy_dataframe(proxy) is True

            # Series proxy should not match
            class SeriesMock(MockFastSlowProxy):
                pass
            SeriesMock.__name__ = 'Series'

            series_proxy = SeriesMock()
            assert cudf_compat._is_proxy_dataframe(series_proxy) is False

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type

    def test_is_proxy_series_with_mock(self):
        """_is_proxy_series should detect Series proxies."""
        from data_ferret.kernel import cudf_compat

        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            class SeriesMock(MockFastSlowProxy):
                pass
            SeriesMock.__name__ = 'Series'

            proxy = SeriesMock()
            assert cudf_compat._is_proxy_series(proxy) is True

            # DataFrame proxy should not match
            class DataFrameMock(MockFastSlowProxy):
                pass
            DataFrameMock.__name__ = 'DataFrame'

            df_proxy = DataFrameMock()
            assert cudf_compat._is_proxy_series(df_proxy) is False

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type

    def test_is_proxy_index_with_mock(self):
        """_is_proxy_index should detect Index proxies."""
        from data_ferret.kernel import cudf_compat

        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            # Test various Index types
            for index_name in ['Index', 'RangeIndex', 'DatetimeIndex', 'MultiIndex']:
                class IndexMock(MockFastSlowProxy):
                    pass
                IndexMock.__name__ = index_name

                proxy = IndexMock()
                assert cudf_compat._is_proxy_index(proxy) is True, f"Failed for {index_name}"

            # DataFrame should not match
            class DataFrameMock(MockFastSlowProxy):
                pass
            DataFrameMock.__name__ = 'DataFrame'

            df_proxy = DataFrameMock()
            assert cudf_compat._is_proxy_index(df_proxy) is False

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type

    def test_non_proxy_objects_return_false(self):
        """Non-proxy objects should return False for all proxy checks."""
        from data_ferret.kernel import cudf_compat

        # Regular pandas objects
        df = pd.DataFrame({'a': [1, 2, 3]})
        series = pd.Series([1, 2, 3])
        index = pd.Index([1, 2, 3])

        assert cudf_compat._is_proxy_dataframe(df) is False
        assert cudf_compat._is_proxy_series(series) is False
        assert cudf_compat._is_proxy_index(index) is False

        # Other types
        assert cudf_compat._is_proxy_dataframe(None) is False
        assert cudf_compat._is_proxy_dataframe("string") is False
        assert cudf_compat._is_proxy_dataframe(123) is False
        assert cudf_compat._is_proxy_dataframe([1, 2, 3]) is False


class TestAreBothCudfSameTypeWithProxies:
    """Test are_both_cudf_same_type with proxy objects."""

    def test_both_dataframe_proxies(self, mock_cudf_proxy_env):
        """Two DataFrame proxies should return True."""
        cudf_compat = mock_cudf_proxy_env

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy1 = DataFrameMock()
        proxy2 = DataFrameMock()

        result = cudf_compat.are_both_cudf_same_type(proxy1, proxy2)
        assert result is True

    def test_both_series_proxies(self, mock_cudf_proxy_env):
        """Two Series proxies should return True."""
        cudf_compat = mock_cudf_proxy_env

        class SeriesMock(MockFastSlowProxy):
            pass
        SeriesMock.__name__ = 'Series'

        proxy1 = SeriesMock()
        proxy2 = SeriesMock()

        result = cudf_compat.are_both_cudf_same_type(proxy1, proxy2)
        assert result is True

    def test_both_index_proxies(self, mock_cudf_proxy_env):
        """Two Index proxies should return True."""
        cudf_compat = mock_cudf_proxy_env

        class IndexMock(MockFastSlowProxy):
            pass
        IndexMock.__name__ = 'RangeIndex'

        proxy1 = IndexMock()
        proxy2 = IndexMock()

        result = cudf_compat.are_both_cudf_same_type(proxy1, proxy2)
        assert result is True

    def test_mixed_proxy_types_return_false(self, mock_cudf_proxy_env):
        """DataFrame proxy + Series proxy should return False."""
        cudf_compat = mock_cudf_proxy_env

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        class SeriesMock(MockFastSlowProxy):
            pass
        SeriesMock.__name__ = 'Series'

        df_proxy = DataFrameMock()
        series_proxy = SeriesMock()

        result = cudf_compat.are_both_cudf_same_type(df_proxy, series_proxy)
        assert result is False

    def test_proxy_and_regular_pandas_return_false(self, mock_cudf_proxy_env):
        """Proxy DataFrame + regular pandas DataFrame should return False."""
        cudf_compat = mock_cudf_proxy_env

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy = DataFrameMock()
        regular_df = pd.DataFrame({'a': [1, 2, 3]})

        result = cudf_compat.are_both_cudf_same_type(proxy, regular_df)
        assert result is False


class TestIsCudfObjectWithProxies:
    """Test is_cudf_object with proxy objects."""

    def test_proxy_dataframe_is_cudf_object(self, mock_cudf_proxy_env):
        """Proxy DataFrame should be detected as cudf object."""
        cudf_compat = mock_cudf_proxy_env

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy = DataFrameMock()
        result = cudf_compat.is_cudf_object(proxy)
        assert result is True

    def test_proxy_series_is_cudf_object(self, mock_cudf_proxy_env):
        """Proxy Series should be detected as cudf object."""
        cudf_compat = mock_cudf_proxy_env

        class SeriesMock(MockFastSlowProxy):
            pass
        SeriesMock.__name__ = 'Series'

        proxy = SeriesMock()
        result = cudf_compat.is_cudf_object(proxy)
        assert result is True

    def test_proxy_index_is_cudf_object(self, mock_cudf_proxy_env):
        """Proxy Index should be detected as cudf object."""
        cudf_compat = mock_cudf_proxy_env

        class IndexMock(MockFastSlowProxy):
            pass
        IndexMock.__name__ = 'RangeIndex'

        proxy = IndexMock()
        result = cudf_compat.is_cudf_object(proxy)
        assert result is True

    def test_regular_pandas_not_cudf_object(self):
        """Regular pandas objects should not be detected as cudf objects."""
        from data_ferret.kernel import cudf_compat

        df = pd.DataFrame({'a': [1, 2, 3]})
        series = pd.Series([1, 2, 3])
        index = pd.Index([1, 2, 3])

        # These should return False regardless of cudf availability
        # (assuming no cudf is installed on test machine)
        assert cudf_compat.is_cudf_object(df) is False or cudf_compat.has_cudf()
        assert cudf_compat.is_cudf_object(series) is False or cudf_compat.has_cudf()
        assert cudf_compat.is_cudf_object(index) is False or cudf_compat.has_cudf()


class TestToPandasWithProxies:
    """Test to_pandas function with proxy objects."""

    def test_to_pandas_with_proxy_dataframe(self, mock_cudf_proxy_env):
        """to_pandas should extract underlying DataFrame from proxy."""
        cudf_compat = mock_cudf_proxy_env

        underlying_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy = DataFrameMock()
        proxy._fsproxy_slow = underlying_df

        result = cudf_compat.to_pandas(proxy)

        # Should be a copy, not the same object
        assert result is not underlying_df
        # But should have same data
        pd.testing.assert_frame_equal(result, underlying_df)

    def test_to_pandas_with_proxy_series(self, mock_cudf_proxy_env):
        """to_pandas should extract underlying Series from proxy."""
        cudf_compat = mock_cudf_proxy_env

        underlying_series = pd.Series([1, 2, 3], name='values')

        class SeriesMock(MockFastSlowProxy):
            pass
        SeriesMock.__name__ = 'Series'

        proxy = SeriesMock()
        proxy._fsproxy_slow = underlying_series

        result = cudf_compat.to_pandas(proxy)

        # Should be a copy
        assert result is not underlying_series
        pd.testing.assert_series_equal(result, underlying_series)

    def test_to_pandas_with_callable_fsproxy_slow(self, mock_cudf_proxy_env):
        """to_pandas should handle callable _fsproxy_slow."""
        cudf_compat = mock_cudf_proxy_env

        underlying_df = pd.DataFrame({'a': [1, 2, 3]})

        class DataFrameMock(MockFastSlowProxy):
            def _fsproxy_slow(self):
                return underlying_df
        DataFrameMock.__name__ = 'DataFrame'

        proxy = DataFrameMock()

        result = cudf_compat.to_pandas(proxy)

        # Should return the underlying DataFrame (copied)
        pd.testing.assert_frame_equal(result, underlying_df)

    def test_to_pandas_returns_copy(self, mock_cudf_proxy_env):
        """to_pandas should return a copy for independence."""
        cudf_compat = mock_cudf_proxy_env

        underlying_df = pd.DataFrame({'a': [1, 2, 3]})

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy = DataFrameMock()
        proxy._fsproxy_slow = underlying_df

        result = cudf_compat.to_pandas(proxy)

        # Modify the result
        result['a'] = [9, 9, 9]

        # Original should be unchanged
        assert list(underlying_df['a']) == [1, 2, 3]

    def test_to_pandas_passthrough_for_regular_objects(self):
        """to_pandas should return regular pandas objects unchanged."""
        from data_ferret.kernel import cudf_compat

        df = pd.DataFrame({'a': [1, 2, 3]})
        result = cudf_compat.to_pandas(df)

        # Regular pandas should pass through unchanged
        assert result is df

    def test_to_pandas_passthrough_for_non_pandas(self):
        """to_pandas should return non-pandas objects unchanged."""
        from data_ferret.kernel import cudf_compat

        assert cudf_compat.to_pandas(None) is None
        assert cudf_compat.to_pandas("string") == "string"
        assert cudf_compat.to_pandas(123) == 123
        assert cudf_compat.to_pandas([1, 2, 3]) == [1, 2, 3]


class TestCacheWithProxies:
    """Test CuDFCheckpointCache with proxy objects."""

    def test_cache_get_or_convert_with_proxy(self, mock_cudf_proxy_env):
        """Cache should handle proxy objects correctly."""
        cudf_compat = mock_cudf_proxy_env

        underlying_df = pd.DataFrame({'a': [1, 2, 3]})

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy = DataFrameMock()
        proxy._fsproxy_slow = underlying_df

        cache = cudf_compat.CuDFCheckpointCache()

        result = cache.get_or_convert(proxy)

        # Should return a pandas DataFrame
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, underlying_df)


class TestDiffWithProxies:
    """Test diff functionality with proxy objects."""

    def test_diff_cudf_with_proxy_dataframes(self, mock_cudf_proxy_env):
        """diff_cudf should work with proxy DataFrames."""
        from data_ferret.kernel.diff import Diff

        cudf_compat = mock_cudf_proxy_env

        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy1 = DataFrameMock()
        proxy1._fsproxy_slow = df1

        proxy2 = DataFrameMock()
        proxy2._fsproxy_slow = df2

        differ = Diff()

        result = cudf_compat.diff_cudf(proxy1, proxy2, 'test', differ)

        # Equal DataFrames should return None
        assert result is None

    def test_diff_cudf_with_different_proxy_dataframes(self, mock_cudf_proxy_env):
        """diff_cudf should detect differences in proxy DataFrames."""
        from data_ferret.kernel.diff import Diff

        cudf_compat = mock_cudf_proxy_env

        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 999]})  # Different!

        class DataFrameMock(MockFastSlowProxy):
            pass
        DataFrameMock.__name__ = 'DataFrame'

        proxy1 = DataFrameMock()
        proxy1._fsproxy_slow = df1

        proxy2 = DataFrameMock()
        proxy2._fsproxy_slow = df2

        differ = Diff()

        result = cudf_compat.diff_cudf(proxy1, proxy2, 'test', differ)

        # Different DataFrames should return a diff
        assert result is not None

    def test_diff_cudf_with_proxy_series(self, mock_cudf_proxy_env):
        """diff_cudf should work with proxy Series."""
        from data_ferret.kernel.diff import Diff

        cudf_compat = mock_cudf_proxy_env

        s1 = pd.Series([1, 2, 3], name='values')
        s2 = pd.Series([1, 2, 3], name='values')

        class SeriesMock(MockFastSlowProxy):
            pass
        SeriesMock.__name__ = 'Series'

        proxy1 = SeriesMock()
        proxy1._fsproxy_slow = s1

        proxy2 = SeriesMock()
        proxy2._fsproxy_slow = s2

        differ = Diff()

        result = cudf_compat.diff_cudf(proxy1, proxy2, 'test', differ)

        # Equal Series should return None
        assert result is None


class TestProxyEdgeCases:
    """Test edge cases for proxy handling."""

    def test_proxy_with_none_fsproxy_slow(self):
        """Handle proxy with None _fsproxy_slow gracefully."""
        from data_ferret.kernel import cudf_compat

        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            class DataFrameMock(MockFastSlowProxy):
                _fsproxy_slow = None
            DataFrameMock.__name__ = 'DataFrame'

            proxy = DataFrameMock()

            original_has_cudf_func = cudf_compat.has_cudf
            cudf_compat.has_cudf = lambda: True

            try:
                # Should not crash
                result = cudf_compat.to_pandas(proxy)
                # With None _fsproxy_slow, should return the proxy itself
                assert result is proxy
            finally:
                cudf_compat.has_cudf = original_has_cudf_func

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type

    def test_proxy_without_fsproxy_slow_attribute(self):
        """Handle proxy without _fsproxy_slow attribute."""
        from data_ferret.kernel import cudf_compat

        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            # Create a proxy-like object without _fsproxy_slow
            class WeirdProxy(MockFastSlowProxy):
                pass
            WeirdProxy.__name__ = 'DataFrame'

            # Remove _fsproxy_slow if inherited
            proxy = WeirdProxy()
            if hasattr(proxy, '_fsproxy_slow'):
                delattr(proxy, '_fsproxy_slow')

            original_has_cudf_func = cudf_compat.has_cudf
            cudf_compat.has_cudf = lambda: True

            try:
                # Should not crash, should return proxy unchanged
                result = cudf_compat.to_pandas(proxy)
                assert result is proxy
            finally:
                cudf_compat.has_cudf = original_has_cudf_func

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type

    def test_proxy_with_to_pandas_method(self):
        """Proxy with to_pandas method should use it as fallback."""
        from data_ferret.kernel import cudf_compat

        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            underlying_df = pd.DataFrame({'a': [1, 2, 3]})

            class DataFrameMock(MockFastSlowProxy):
                def to_pandas(self):
                    return underlying_df.copy()
            DataFrameMock.__name__ = 'DataFrame'

            proxy = DataFrameMock()
            # No _fsproxy_slow, but has to_pandas

            original_has_cudf_func = cudf_compat.has_cudf
            cudf_compat.has_cudf = lambda: True

            try:
                result = cudf_compat.to_pandas(proxy)
                pd.testing.assert_frame_equal(result, underlying_df)
            finally:
                cudf_compat.has_cudf = original_has_cudf_func

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type

    def test_empty_dataframe_proxy(self):
        """Handle empty DataFrame proxy."""
        from data_ferret.kernel import cudf_compat

        original_proxy_type = cudf_compat._cudf_pandas_proxy_type
        cudf_compat._cudf_pandas_proxy_type = MockFastSlowProxy
        cudf_compat._HAS_CUDF_PANDAS = True

        try:
            empty_df = pd.DataFrame()

            class DataFrameMock(MockFastSlowProxy):
                pass
            DataFrameMock.__name__ = 'DataFrame'

            proxy = DataFrameMock()
            proxy._fsproxy_slow = empty_df

            original_has_cudf_func = cudf_compat.has_cudf
            cudf_compat.has_cudf = lambda: True

            try:
                result = cudf_compat.to_pandas(proxy)
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            finally:
                cudf_compat.has_cudf = original_has_cudf_func

        finally:
            cudf_compat._cudf_pandas_proxy_type = original_proxy_type


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
