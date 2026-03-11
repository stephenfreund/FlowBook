"""
Test GPU checkpoint mode for cudf objects.

Tests the GPU-side checkpointing path that keeps cudf objects on GPU
via deep copy instead of converting to pandas (CPU). This is controlled
by the FLOWBOOK_CUDF_GPU_CHECKPOINT flag.

Since cudf is not available in the test environment, these tests use
mock objects to verify the control flow and data model integration.
"""

import pytest
import numpy as np
import pandas as pd

from flowbook.kernel_support.cudf_compat import (
    is_gpu_checkpoint_mode,
    set_gpu_checkpoint_mode,
    _gpu_deep_copy,
    are_both_cudf_same_type,
    CuDFOriginTracker,
    is_cudf_object,
)
from flowbook.kernel_support.heap_size import CheckpointOverhead
from flowbook.cli.models import V5CellMemory, Plot3Data, Plot6Data, CDFData


class TestGPUCheckpointFlag:
    """Test the GPU checkpoint mode flag."""

    def test_default_off(self):
        """GPU checkpoint mode should be off by default."""
        set_gpu_checkpoint_mode(False)
        assert not is_gpu_checkpoint_mode()

    def test_set_on(self):
        """Setting on should enable GPU checkpoint mode (if cudf available)."""
        from flowbook.kernel_support.cudf_compat import has_cudf
        set_gpu_checkpoint_mode(True)
        # is_gpu_checkpoint_mode() returns _CUDF_GPU_CHECKPOINT and has_cudf()
        if has_cudf():
            assert is_gpu_checkpoint_mode()
        else:
            assert not is_gpu_checkpoint_mode()
        set_gpu_checkpoint_mode(False)

    def test_set_and_get(self):
        """set_gpu_checkpoint_mode should update the flag."""
        from flowbook.kernel_support import cudf_compat
        set_gpu_checkpoint_mode(True)
        assert cudf_compat._CUDF_GPU_CHECKPOINT is True
        set_gpu_checkpoint_mode(False)
        assert cudf_compat._CUDF_GPU_CHECKPOINT is False


class TestGPUDeepCopy:
    """Test _gpu_deep_copy function."""

    def test_copy_dataframe(self):
        """Should use .copy(deep=True) when available."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        result = _gpu_deep_copy(df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        # Should be a deep copy
        assert result is not df
        assert result['a'].values is not df['a'].values

    def test_copy_series(self):
        """Should use .copy(deep=True) for Series."""
        s = pd.Series([1, 2, 3], name='test')
        result = _gpu_deep_copy(s)
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, s)
        assert result is not s

    def test_no_copy_method(self):
        """Should return object as-is if no copy method."""
        obj = 42
        assert _gpu_deep_copy(obj) == 42


class TestGPUDeepCopyProxyFallback:
    """Test that _gpu_deep_copy handles cudf.pandas proxy slow-path fallback efficiently.

    When cudf.pandas proxies materialize their slow (pandas) side — e.g., after
    calling .values or .to_numpy() — proxy.copy(deep=True) dispatches to a full
    pandas deep copy, which is ~1000x slower than GPU copy for large DataFrames.

    The fix: _gpu_deep_copy detects when _fsproxy_wrapped points to a pandas
    object and uses a fast shallow CoW copy instead of proxy.copy(deep=True).
    """

    def _make_proxy_with_pandas_wrapped(self, df: pd.DataFrame):
        """Create a mock cudf.pandas proxy whose _fsproxy_wrapped is a pandas DataFrame.

        This simulates the state of a proxy after .values has been called,
        which causes the slow (pandas) side to be materialized.
        """
        from unittest.mock import MagicMock
        from flowbook.kernel_support import cudf_compat

        proxy = MagicMock()
        proxy._fsproxy_wrapped = df  # pandas side is materialized
        proxy.copy = MagicMock(side_effect=lambda deep=False: df.copy(deep=deep))

        # Make is_cudf_proxy return True for this mock
        original_is_proxy = cudf_compat.is_cudf_proxy
        cudf_compat.is_cudf_proxy = lambda obj: obj is proxy
        self._cleanup = lambda: setattr(cudf_compat, 'is_cudf_proxy', original_is_proxy)
        return proxy

    def _make_proxy_with_cudf_wrapped(self):
        """Create a mock cudf.pandas proxy whose _fsproxy_wrapped is a cudf (non-pandas) object."""
        from unittest.mock import MagicMock
        from flowbook.kernel_support import cudf_compat

        # Simulate a native cudf DataFrame (not pandas)
        cudf_df = MagicMock()
        cudf_df.__module__ = 'cudf.core.dataframe'
        type(cudf_df).__module__ = 'cudf.core.dataframe'
        cudf_copy = MagicMock()
        cudf_df.copy = MagicMock(return_value=cudf_copy)

        proxy = MagicMock()
        proxy._fsproxy_wrapped = cudf_df  # GPU side is materialized

        original_is_proxy = cudf_compat.is_cudf_proxy
        cudf_compat.is_cudf_proxy = lambda obj: obj is proxy
        self._cleanup = lambda: setattr(cudf_compat, 'is_cudf_proxy', original_is_proxy)
        return proxy, cudf_df, cudf_copy

    def teardown_method(self):
        if hasattr(self, '_cleanup'):
            self._cleanup()

    def test_proxy_with_pandas_wrapped_uses_shallow_copy(self):
        """When proxy's _fsproxy_wrapped is pandas, should use shallow CoW copy (not deep)."""
        df = pd.DataFrame({
            'a': range(100_000),
            'b': np.random.randn(100_000),
            'c': ['x'] * 100_000,
        })
        proxy = self._make_proxy_with_pandas_wrapped(df)

        result = _gpu_deep_copy(proxy)

        # Should NOT have called proxy.copy (the slow path)
        proxy.copy.assert_not_called()

        # Result should be a shallow copy of the pandas DataFrame
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        assert result is not df

        # Shallow copy: underlying arrays should be shared (CoW)
        assert np.shares_memory(result['a'].values, df['a'].values)

    def test_proxy_with_pandas_wrapped_is_fast(self):
        """Shallow CoW copy should be orders of magnitude faster than deep copy."""
        import time

        # Create a large DataFrame to amplify timing differences
        n_rows = 500_000
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(n_rows) for i in range(50)
        })
        proxy = self._make_proxy_with_pandas_wrapped(df)

        # Time the optimized path (shallow CoW copy)
        start = time.perf_counter()
        for _ in range(10):
            _gpu_deep_copy(proxy)
        optimized_ms = (time.perf_counter() - start) * 1000

        # Time a deep copy for comparison
        start = time.perf_counter()
        for _ in range(10):
            df.copy(deep=True)
        deep_copy_ms = (time.perf_counter() - start) * 1000

        # Optimized path should be at least 5x faster than deep copy
        assert optimized_ms < deep_copy_ms / 5, (
            f"Optimized path ({optimized_ms:.1f}ms) should be >5x faster "
            f"than deep copy ({deep_copy_ms:.1f}ms)"
        )

    def test_proxy_with_cudf_wrapped_copies_gpu_side(self):
        """When proxy's _fsproxy_wrapped is cudf, should copy the GPU side directly."""
        proxy, cudf_df, cudf_copy = self._make_proxy_with_cudf_wrapped()

        result = _gpu_deep_copy(proxy)

        # Should have called cudf_df.copy(deep=True), not proxy.copy
        cudf_df.copy.assert_called_once_with(deep=True)
        proxy.copy.assert_not_called()
        assert result is cudf_copy

    def test_proxy_without_fsproxy_wrapped_falls_through(self):
        """Proxy without _fsproxy_wrapped should fall through to proxy.copy()."""
        from unittest.mock import MagicMock
        from flowbook.kernel_support import cudf_compat

        proxy = MagicMock(spec=['copy'])  # No _fsproxy_wrapped
        del proxy._fsproxy_wrapped  # Ensure attribute doesn't exist
        copy_result = MagicMock()
        proxy.copy = MagicMock(return_value=copy_result)

        original_is_proxy = cudf_compat.is_cudf_proxy
        cudf_compat.is_cudf_proxy = lambda obj: obj is proxy
        try:
            result = _gpu_deep_copy(proxy)
            proxy.copy.assert_called_once_with(deep=True)
            assert result is copy_result
        finally:
            cudf_compat.is_cudf_proxy = original_is_proxy


class TestCuDFOriginTrackerGPUGuard:
    """Test that restore_value skips from_pandas when value is already cudf."""

    def test_restore_non_cudf_value(self):
        """Non-cudf values should pass through normally."""
        tracker = CuDFOriginTracker()
        # No origins recorded, should return as-is
        result = tracker.restore_value('x', 42)
        assert result == 42

    def test_restore_value_pandas_no_origin(self):
        """Pandas value with no origin should pass through."""
        tracker = CuDFOriginTracker()
        df = pd.DataFrame({'a': [1, 2]})
        result = tracker.restore_value('x', df)
        assert result is df


class TestCheckpointOverheadGPUFields:
    """Test that CheckpointOverhead has GPU fields."""

    def test_default_gpu_fields(self):
        """GPU fields should default to 0/empty."""
        co = CheckpointOverhead(
            total_mb=100,
            by_checkpoint={'c1': 50, 'c2': 50},
            by_variable={'x': 100},
            cumulative={'c1': 50, 'c2': 100},
            by_checkpoint_by_var={'c1': {'x': 50}},
        )
        assert co.gpu_total_mb == 0.0
        assert co.gpu_by_checkpoint == {}
        assert co.gpu_by_variable == {}
        assert co.gpu_cumulative == {}
        assert co.gpu_by_checkpoint_by_var == {}

    def test_gpu_fields_populated(self):
        """GPU fields should be settable."""
        co = CheckpointOverhead(
            total_mb=100,
            by_checkpoint={'c1': 50},
            by_variable={'x': 100},
            cumulative={'c1': 50},
            by_checkpoint_by_var={'c1': {'x': 50}},
            gpu_total_mb=200,
            gpu_by_checkpoint={'c1': 200},
            gpu_by_variable={'x': 200},
            gpu_cumulative={'c1': 200},
            gpu_by_checkpoint_by_var={'c1': {'x': 200}},
        )
        assert co.gpu_total_mb == 200
        assert co.gpu_by_checkpoint == {'c1': 200}


class TestV5CellMemoryGPUFields:
    """Test V5CellMemory GPU checkpoint fields."""

    def test_total_mb_includes_gpu(self):
        """total_mb should include gpu_checkpoint_mb."""
        cell = V5CellMemory(
            cell_id='test', cell_index=0,
            user_ns_mb=100, gpu_mb=50, checkpoint_mb=30,
            gpu_checkpoint_mb=20,
        )
        # total = user_ns(100) + gpu(50) + checkpoint(30) + gpu_checkpoint(20) = 200
        assert cell.total_mb == 200

    def test_serialization_roundtrip(self):
        """GPU fields should survive to_dict/from_dict."""
        cell = V5CellMemory(
            cell_id='abc', cell_index=1,
            user_ns_mb=100, gpu_mb=50, checkpoint_mb=30,
            gpu_checkpoint_mb=25,
            gpu_checkpoint_vars={'df': 20, 'arr': 5},
        )
        d = cell.to_dict()
        assert d['gpu_checkpoint_mb'] == 25
        assert d['gpu_checkpoint_vars'] == {'df': 20, 'arr': 5}

        cell2 = V5CellMemory.from_dict(d)
        assert cell2.gpu_checkpoint_mb == 25
        assert cell2.gpu_checkpoint_vars == {'df': 20, 'arr': 5}

    def test_backward_compat_no_gpu_fields(self):
        """from_dict should handle missing GPU fields gracefully."""
        d = {
            'cell_id': 'old', 'cell_index': 0,
            'user_ns_mb': 100, 'gpu_mb': 50, 'checkpoint_mb': 30,
        }
        cell = V5CellMemory.from_dict(d)
        assert cell.gpu_checkpoint_mb == 0.0
        assert cell.gpu_checkpoint_vars == {}

    def test_zero_gpu_not_serialized(self):
        """GPU fields with value 0 should not appear in to_dict."""
        cell = V5CellMemory(
            cell_id='test', cell_index=0,
            user_ns_mb=100, gpu_mb=50, checkpoint_mb=30,
        )
        d = cell.to_dict()
        assert 'gpu_checkpoint_mb' not in d
        assert 'gpu_checkpoint_vars' not in d


class TestPlot3DataGPUField:
    """Test Plot3Data gpu_checkpoint_mb field."""

    def test_default_empty(self):
        """gpu_checkpoint_mb should default to empty list."""
        p3 = Plot3Data(
            cells=[1, 2], user_ns_mb=[100, 200], gpu_mb=[50, 60],
            overhead_mb=[30, 40], has_baseline=False,
            peak_overhead_mb=40, peak_overhead_pct=20,
            peak_cell=1, initial_count=2,
        )
        assert p3.gpu_checkpoint_mb == []

    def test_with_gpu_data(self):
        """gpu_checkpoint_mb should accept data."""
        p3 = Plot3Data(
            cells=[1, 2], user_ns_mb=[100, 200], gpu_mb=[50, 60],
            overhead_mb=[30, 40], has_baseline=False,
            peak_overhead_mb=40, peak_overhead_pct=20,
            peak_cell=1, initial_count=2,
            gpu_checkpoint_mb=[10, 25],
        )
        assert p3.gpu_checkpoint_mb == [10, 25]


class TestPlot6DataGPURatios:
    """Test Plot6Data gpu_ratios field."""

    def test_default_empty(self):
        """gpu_ratios should default to empty list."""
        p6 = Plot6Data(cells=[1, 2], ratios=[0.5, 0.3], initial_count=2)
        assert p6.gpu_ratios == []

    def test_with_gpu_ratios(self):
        """gpu_ratios should accept data."""
        p6 = Plot6Data(
            cells=[1, 2], ratios=[0.5, 0.3],
            gpu_ratios=[0.2, 0.1], initial_count=2,
        )
        assert p6.gpu_ratios == [0.2, 0.1]


class TestCDFDataGPUFields:
    """Test CDFData GPU checkpoint fields."""

    def test_default_empty(self):
        """GPU CDF fields should default to empty."""
        cdf = CDFData(
            time_overhead_ms=[10], time_sorted=[10], time_percentiles=[1.0],
            memory_ratios=[0.1], memory_sorted=[0.1], memory_percentiles=[1.0],
            peak_memory_pct=[5.0], peak_sorted=[5.0], peak_percentiles=[1.0],
        )
        assert cdf.gpu_memory_ratios == []
        assert cdf.gpu_memory_sorted == []
        assert cdf.gpu_memory_percentiles == []
        assert cdf.gpu_peak_memory_pct == []
        assert cdf.gpu_peak_sorted == []
        assert cdf.gpu_peak_percentiles == []

    def test_with_gpu_data(self):
        """GPU CDF fields should accept data."""
        cdf = CDFData(
            time_overhead_ms=[], time_sorted=[], time_percentiles=[],
            memory_ratios=[], memory_sorted=[], memory_percentiles=[],
            peak_memory_pct=[], peak_sorted=[], peak_percentiles=[],
            gpu_memory_ratios=[0.1, 0.2],
            gpu_memory_sorted=[0.1, 0.2],
            gpu_memory_percentiles=[0.5, 1.0],
            gpu_peak_memory_pct=[3.0],
            gpu_peak_sorted=[3.0],
            gpu_peak_percentiles=[1.0],
        )
        assert len(cdf.gpu_memory_ratios) == 2
        assert len(cdf.gpu_peak_memory_pct) == 1
