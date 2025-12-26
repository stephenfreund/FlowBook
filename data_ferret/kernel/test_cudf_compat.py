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
    """Test column tracking behavior with cudf DataFrames."""

    def test_cudf_groupby_no_crash(self):
        """cudf GroupBy operations should not crash (tracking is limited).

        Note: We only patch pd.DataFrame methods, so native cudf DataFrames
        won't have column tracking. This test verifies the operation completes
        without the recursion error, not that tracking works.
        """
        import cudf

        tracker = ColumnAccessTracker()
        tracker.install()
        try:
            gdf = cudf.DataFrame({
                'category': ['A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40]
            })
            tracker.register_df(gdf, 'gdf')

            # This should complete without recursion error
            result = gdf.groupby('category')['value'].mean()

            # Verify the operation worked
            assert len(result) == 2

            # Note: cudf native operations don't go through our pandas patches,
            # so tracking won't capture these accesses. This is expected.
            # Full cudf tracking would require patching cudf methods (Phase 1+).
        finally:
            tracker.uninstall()


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
