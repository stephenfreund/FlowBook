"""
Tests for cudf.pandas proxy column tracking.

These tests verify that column reads and writes through cudf.pandas proxy
objects are properly tracked. The proxy intercepts __setitem__/__getitem__
before they reach the underlying pandas/cudf methods.
"""

import pytest
import pandas as pd

# Try to import cudf.pandas - skip all tests if not available
try:
    from cudf.pandas.fast_slow_proxy import _FastSlowProxy
    HAS_CUDF_PANDAS = True
except ImportError:
    HAS_CUDF_PANDAS = False

pytestmark = pytest.mark.skipif(
    not HAS_CUDF_PANDAS,
    reason="cudf.pandas not available"
)


class TestCudfPandasProxyTracking:
    """Test column tracking through cudf.pandas proxy objects."""

    def test_proxy_setitem_tracked(self):
        """Column writes via cudf.pandas proxy should be tracked."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker
        from flowbook.kernel_support import cudf_compat

        # Enable cudf.pandas mode
        import cudf.pandas
        cudf.pandas.install()

        try:
            tracker = ColumnAccessTracker()
            tracker.activate()

            # Create a DataFrame (will be wrapped in proxy)
            df = pd.DataFrame({'a': [1, 2, 3]})
            tracker.register_df(df, 'df')

            # This should be tracked via proxy patch
            df['b'] = [4, 5, 6]

            writes = tracker.resolve_writes_to_paths()
            assert 'df' in writes, f"Expected 'df' in writes, got {writes}"
            assert 'b' in writes['df'], f"Expected 'b' in writes['df'], got {writes}"

        finally:
            tracker.deactivate()

    def test_proxy_getitem_tracked(self):
        """Column reads via cudf.pandas proxy should be tracked."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker
        from flowbook.kernel_support import cudf_compat

        # Enable cudf.pandas mode
        import cudf.pandas
        cudf.pandas.install()

        try:
            tracker = ColumnAccessTracker()
            tracker.activate()

            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            tracker.register_df(df, 'df')

            # Read column - should be tracked
            _ = df['a']

            reads = tracker.resolve_to_paths()
            assert 'df' in reads, f"Expected 'df' in reads, got {reads}"
            assert 'a' in reads['df'], f"Expected 'a' in reads['df'], got {reads}"

        finally:
            tracker.deactivate()

    def test_proxy_multi_column_write_tracked(self):
        """Multi-column writes via proxy should be tracked."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker

        import cudf.pandas
        cudf.pandas.install()

        try:
            tracker = ColumnAccessTracker()
            tracker.activate()

            df = pd.DataFrame({'a': [1, 2, 3]})
            tracker.register_df(df, 'df')

            # Write multiple columns
            df['b'] = [4, 5, 6]
            df['c'] = [7, 8, 9]

            writes = tracker.resolve_writes_to_paths()
            assert 'df' in writes
            assert 'b' in writes['df']
            assert 'c' in writes['df']

        finally:
            tracker.deactivate()

    def test_proxy_read_before_write(self):
        """Read-before-write should be tracked correctly via proxy."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker

        import cudf.pandas
        cudf.pandas.install()

        try:
            tracker = ColumnAccessTracker()
            tracker.activate()

            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            tracker.register_df(df, 'df')

            # Read a, then write a (should track read)
            _ = df['a']
            df['a'] = [10, 20, 30]

            # Read b only (no write, should be in reads)
            _ = df['b']

            reads = tracker.resolve_to_paths()
            assert 'df' in reads
            # 'a' was read before written - should be in reads
            assert 'a' in reads['df']
            # 'b' was read but not written - should be in reads
            assert 'b' in reads['df']

        finally:
            tracker.deactivate()

    def test_proxy_write_then_read_not_rbw(self):
        """Write-then-read should NOT be recorded as read-before-write."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker

        import cudf.pandas
        cudf.pandas.install()

        try:
            tracker = ColumnAccessTracker()
            tracker.activate()

            df = pd.DataFrame({'a': [1, 2, 3]})
            tracker.register_df(df, 'df')

            # Write first, then read
            df['b'] = [4, 5, 6]
            _ = df['b']

            reads = tracker.resolve_to_paths()
            # 'b' was written first, so should NOT be in reads
            if 'df' in reads:
                assert 'b' not in reads['df'], "Write-then-read should not be RBW"

        finally:
            tracker.deactivate()

    def test_proxy_only_dataframe_tracked(self):
        """Only DataFrame proxies should be tracked, not Series."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker
        from flowbook.kernel_support import cudf_compat

        import cudf.pandas
        cudf.pandas.install()

        try:
            tracker = ColumnAccessTracker()
            tracker.activate()

            df = pd.DataFrame({'a': [1, 2, 3]})
            tracker.register_df(df, 'df')

            # Access as Series - this creates a Series proxy
            series = df['a']

            # Verify the DataFrame read was tracked
            reads = tracker.resolve_to_paths()
            assert 'df' in reads
            assert 'a' in reads['df']

        finally:
            tracker.deactivate()


class TestProxyDetection:
    """Test proxy type detection functions."""

    def test_is_cudf_proxy_non_proxy_objects(self):
        """Test is_cudf_proxy correctly rejects non-proxy objects."""
        from flowbook.kernel_support import cudf_compat

        # Non-proxy objects should always return False
        assert not cudf_compat.is_cudf_proxy(42)
        assert not cudf_compat.is_cudf_proxy("string")
        assert not cudf_compat.is_cudf_proxy([1, 2, 3])
        assert not cudf_compat.is_cudf_proxy({'a': 1})

    def test_proxy_tracking_works_regardless_of_detection(self):
        """Test that column tracking works even if is_cudf_proxy detection varies.

        The key requirement is that column writes/reads are tracked, not that
        is_cudf_proxy returns True. The tracking patches are installed on the
        proxy class itself, so they work regardless of isinstance checks.
        """
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker

        import cudf.pandas
        cudf.pandas.install()

        tracker = ColumnAccessTracker()
        tracker.activate()

        try:
            df = pd.DataFrame({'a': [1, 2, 3]})
            tracker.register_df(df, 'df')

            # These operations should be tracked regardless of proxy detection
            df['b'] = [4, 5, 6]
            _ = df['a']

            writes = tracker.resolve_writes_to_paths()
            reads = tracker.resolve_to_paths()

            assert 'df' in writes, "Column write should be tracked"
            assert 'b' in writes['df'], "Column 'b' should be in writes"
            assert 'df' in reads, "Column read should be tracked"
            assert 'a' in reads['df'], "Column 'a' should be in reads"

        finally:
            tracker.deactivate()


class TestProxyPatchInstallation:
    """Test that proxy patches are correctly installed and uninstalled."""

    def test_install_proxy_tracking(self):
        """Test that proxy tracking can be installed."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker
        from flowbook.kernel_support import cudf_compat

        tracker = ColumnAccessTracker()

        # Install proxy tracking
        cudf_compat.install_cudf_pandas_proxy_tracking(tracker)

        # Should be marked as installed
        assert cudf_compat._cudf_pandas_proxy_patches_installed

        # Clean up
        cudf_compat.uninstall_cudf_pandas_proxy_tracking()
        assert not cudf_compat._cudf_pandas_proxy_patches_installed

    def test_install_idempotent(self):
        """Installing proxy tracking multiple times should be safe."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker
        from flowbook.kernel_support import cudf_compat

        tracker = ColumnAccessTracker()

        # Install multiple times
        cudf_compat.install_cudf_pandas_proxy_tracking(tracker)
        cudf_compat.install_cudf_pandas_proxy_tracking(tracker)
        cudf_compat.install_cudf_pandas_proxy_tracking(tracker)

        # Should still be installed once
        assert cudf_compat._cudf_pandas_proxy_patches_installed

        # Clean up
        cudf_compat.uninstall_cudf_pandas_proxy_tracking()
        assert not cudf_compat._cudf_pandas_proxy_patches_installed

    def test_uninstall_idempotent(self):
        """Uninstalling proxy tracking multiple times should be safe."""
        from flowbook.kernel_support import cudf_compat

        # Uninstall without installing first - should be safe
        cudf_compat.uninstall_cudf_pandas_proxy_tracking()
        cudf_compat.uninstall_cudf_pandas_proxy_tracking()

        assert not cudf_compat._cudf_pandas_proxy_patches_installed
