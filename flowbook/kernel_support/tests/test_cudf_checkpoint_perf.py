"""
Performance test for cudf checkpoint overhead.

This test reproduces the 15+ second per-cell overhead observed with cudf.pandas
proxy DataFrames in the RAPIDS feature engineering notebook.

Root cause analysis:
1. cudf.pandas proxies are not fingerprinted correctly - _fingerprint() only
   handles native cudf.DataFrame, not proxy objects
2. Even with caching, accessing _fsproxy_slow may trigger GPU→CPU sync
3. The .copy() call on the slow object duplicates the data

Expected findings:
- Without proxy-aware fingerprinting, cache misses on every checkpoint
- GPU→CPU transfer happens on every checkpoint, not just first time
- Large DataFrames (millions of rows) take seconds to transfer
"""

import time
import pytest
import pandas as pd
import numpy as np

# Check if cudf is available
try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

# Check if cudf.pandas proxy is available
try:
    from cudf.pandas.fast_slow_proxy import _FastSlowProxy
    HAS_CUDF_PANDAS = True
except ImportError:
    HAS_CUDF_PANDAS = False


def create_large_dataframe(n_rows: int = 1_000_000, n_cols: int = 50) -> pd.DataFrame:
    """Create a large DataFrame similar to the RAPIDS notebook."""
    np.random.seed(42)
    data = {
        f'col_{i}': np.random.randn(n_rows) if i % 2 == 0
        else np.random.randint(0, 100, n_rows)
        for i in range(n_cols)
    }
    return pd.DataFrame(data)


@pytest.mark.skipif(not HAS_CUDF, reason="cudf not available")
class TestCudfCheckpointPerformance:
    """Test checkpoint performance with cudf objects."""

    def test_fingerprint_native_cudf(self):
        """Test that native cudf DataFrames get fingerprinted."""
        from flowbook.kernel_support.cudf_compat import (
            CuDFCheckpointCache,
            is_cudf_dataframe,
        )

        # Create native cudf DataFrame
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        gdf = cudf.DataFrame.from_pandas(pdf)

        assert is_cudf_dataframe(gdf), "Should detect native cudf DataFrame"

        cache = CuDFCheckpointCache()
        fp = cache._fingerprint(gdf)

        assert fp is not None, "Native cudf should have fingerprint"
        assert fp[0] == 'DataFrame'
        assert fp[1] == gdf.shape

    @pytest.mark.skipif(not HAS_CUDF_PANDAS, reason="cudf.pandas not available")
    def test_fingerprint_proxy_dataframe(self):
        """Test that cudf.pandas proxy DataFrames get fingerprinted.

        CURRENT BUG: _fingerprint() returns None for proxy objects because
        is_cudf_dataframe() only checks isinstance(obj, cudf.DataFrame).
        """
        from flowbook.kernel_support.cudf_compat import (
            CuDFCheckpointCache,
            is_cudf_dataframe,
            is_cudf_proxy,
            _is_proxy_dataframe,
        )

        # Create proxy DataFrame (simulating cudf.pandas mode)
        # In actual cudf.pandas mode, pd.DataFrame becomes a proxy
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # For this test, we need an actual proxy object
        # This requires cudf.pandas to be loaded
        # Skip if we can't create a real proxy
        if not _is_proxy_dataframe(pdf):
            pytest.skip("Need cudf.pandas loaded to test proxy fingerprinting")

        cache = CuDFCheckpointCache()
        fp = cache._fingerprint(pdf)

        # BUG: This assertion will fail - fingerprint returns None for proxies
        assert fp is not None, (
            "Proxy DataFrame should have fingerprint! "
            "Current implementation only fingerprints native cudf objects."
        )

    def test_cache_hit_performance_native(self):
        """Test cache hit performance with native cudf DataFrame."""
        from flowbook.kernel_support.cudf_compat import (
            CuDFCheckpointCache,
            to_pandas_cached,
        )

        # Create a moderately large cudf DataFrame
        pdf = create_large_dataframe(n_rows=100_000, n_cols=20)
        gdf = cudf.DataFrame.from_pandas(pdf)

        cache = CuDFCheckpointCache()

        # First call - should be slow (GPU→CPU transfer)
        start = time.perf_counter()
        result1 = cache.get_or_convert(gdf)
        first_time = time.perf_counter() - start

        # Second call - should be fast (cache hit)
        start = time.perf_counter()
        result2 = cache.get_or_convert(gdf)
        second_time = time.perf_counter() - start

        print(f"\nNative cudf cache test:")
        print(f"  First call (cache miss): {first_time*1000:.1f}ms")
        print(f"  Second call (cache hit): {second_time*1000:.1f}ms")
        print(f"  Speedup: {first_time/second_time:.1f}x")

        # Cache hit should be much faster
        assert second_time < first_time / 2, (
            f"Cache hit should be at least 2x faster. "
            f"First: {first_time*1000:.1f}ms, Second: {second_time*1000:.1f}ms"
        )

    def test_deepcopy_cudf_performance(self):
        """Test deepcopy performance with cudf objects.

        This simulates what happens during checkpoint save.
        """
        from flowbook.kernel_support.cudf_compat import deepcopy_cudf
        from flowbook.kernel_support.cudf_compat import get_checkpoint_cache

        # Create a large cudf DataFrame
        pdf = create_large_dataframe(n_rows=500_000, n_cols=30)
        gdf = cudf.DataFrame.from_pandas(pdf)

        memo = {}

        # Clear cache to simulate fresh checkpoint
        get_checkpoint_cache().clear()

        # First deepcopy
        start = time.perf_counter()
        result1 = deepcopy_cudf(gdf, memo)
        first_time = time.perf_counter() - start

        # Second deepcopy with same memo
        start = time.perf_counter()
        result2 = deepcopy_cudf(gdf, memo)
        second_time = time.perf_counter() - start

        print(f"\nDeepopy cudf test (500K rows x 30 cols):")
        print(f"  First deepcopy: {first_time*1000:.1f}ms")
        print(f"  Second deepcopy (same memo): {second_time*1000:.1f}ms")

        # Memo should provide the cached object
        assert second_time < first_time / 5, (
            f"Second deepcopy should use memo cache. "
            f"First: {first_time*1000:.1f}ms, Second: {second_time*1000:.1f}ms"
        )

    def test_repeated_checkpoint_overhead(self):
        """Simulate multiple cell executions with checkpointing.

        This reproduces the 15+ second overhead observed in the RAPIDS notebook.
        """
        from flowbook.kernel_support.cudf_compat import (
            deepcopy_cudf,
            get_checkpoint_cache,
        )

        # Create DataFrames similar to RAPIDS notebook
        # train: ~3M rows, test: ~700K rows
        print("\nCreating large DataFrames...")
        train = cudf.DataFrame.from_pandas(
            create_large_dataframe(n_rows=500_000, n_cols=30)
        )
        test = cudf.DataFrame.from_pandas(
            create_large_dataframe(n_rows=100_000, n_cols=30)
        )

        namespace = {'train': train, 'test': test}

        # Simulate 5 cell executions with checkpointing
        checkpoint_times = []

        for i in range(5):
            get_checkpoint_cache().clear()  # Clear cache between cells
            memo = {}

            start = time.perf_counter()
            for name, obj in namespace.items():
                deepcopy_cudf(obj, memo)
            elapsed = time.perf_counter() - start

            checkpoint_times.append(elapsed)
            print(f"  Cell {i+1} checkpoint: {elapsed*1000:.1f}ms")

        avg_time = sum(checkpoint_times) / len(checkpoint_times)
        print(f"\nAverage checkpoint time: {avg_time*1000:.1f}ms")

        # If cache is working, times should decrease after first checkpoint
        # If not working, all times will be similar (the bug)

        # Check if we're seeing the performance issue
        if all(t > 1.0 for t in checkpoint_times):
            print("\n⚠️  HIGH OVERHEAD DETECTED!")
            print("Each checkpoint takes >1s, indicating cache is not effective.")
            print("This explains the 15+ second per-cell overhead in RAPIDS notebooks.")


@pytest.mark.skipif(not HAS_CUDF_PANDAS, reason="cudf.pandas not available")
class TestCudfPandasProxyPerformance:
    """Test performance with cudf.pandas proxy mode."""

    def test_proxy_to_pandas_overhead(self):
        """Measure the overhead of accessing _fsproxy_slow on proxies.

        Even with caching, accessing _fsproxy_slow may trigger GPU sync.
        """
        from flowbook.kernel_support.cudf_compat import (
            is_cudf_proxy,
            to_pandas,
        )

        # This test requires cudf.pandas to be active
        # Load it if not already loaded
        try:
            import cudf.pandas
            cudf.pandas.install()
        except Exception:
            pytest.skip("Could not activate cudf.pandas")

        # Create DataFrame - should be a proxy in cudf.pandas mode
        df = pd.DataFrame({
            'a': np.random.randn(100_000),
            'b': np.random.randint(0, 100, 100_000),
        })

        if not is_cudf_proxy(df):
            pytest.skip("DataFrame is not a cudf.pandas proxy")

        # Measure to_pandas conversion time
        times = []
        for i in range(5):
            start = time.perf_counter()
            pdf = to_pandas(df)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  to_pandas call {i+1}: {elapsed*1000:.1f}ms")

        print(f"\nAverage to_pandas time: {sum(times)/len(times)*1000:.1f}ms")


class TestProposedFixes:
    """Test proposed fixes for the cudf checkpoint overhead."""

    @pytest.mark.skipif(not HAS_CUDF, reason="cudf not available")
    def test_proxy_aware_fingerprint(self):
        """Test a proposed proxy-aware fingerprint implementation.

        The fix: Add proxy handling to _fingerprint() that extracts
        shape/dtype from the proxy without triggering GPU→CPU transfer.
        """
        from flowbook.kernel_support.cudf_compat import (
            is_cudf_proxy,
            _is_proxy_dataframe,
        )

        def improved_fingerprint(obj):
            """Proposed fix: fingerprint that works with proxies."""
            # Handle proxy DataFrames
            if _is_proxy_dataframe(obj):
                # Access shape and dtypes without triggering GPU transfer
                # These are cached on the proxy and don't need GPU access
                try:
                    return ('DataFrame', obj.shape, tuple(obj.dtypes.items()), None)
                except Exception:
                    return None

            # Handle native cudf (existing logic)
            # ... existing implementation ...
            return None

        # Create a proxy DataFrame for testing
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

        # Test the improved fingerprint
        fp = improved_fingerprint(pdf)

        if _is_proxy_dataframe(pdf):
            assert fp is not None, "Improved fingerprint should work for proxies"
            assert fp[0] == 'DataFrame'
            assert fp[1] == pdf.shape
            print(f"\nProxy fingerprint: {fp}")

    @pytest.mark.skipif(not HAS_CUDF, reason="cudf not available")
    def test_incremental_checkpoint_strategy(self):
        """Test incremental checkpointing strategy.

        Instead of copying entire DataFrames, track which ones changed
        and only checkpoint those.
        """
        from flowbook.kernel_support.cudf_compat import get_checkpoint_cache

        # Create DataFrames
        df1 = cudf.DataFrame.from_pandas(
            create_large_dataframe(n_rows=100_000, n_cols=10)
        )
        df2 = cudf.DataFrame.from_pandas(
            create_large_dataframe(n_rows=100_000, n_cols=10)
        )

        cache = get_checkpoint_cache()

        # First checkpoint - both need to be copied
        start = time.perf_counter()
        pdf1 = cache.get_or_convert(df1)
        pdf2 = cache.get_or_convert(df2)
        first_time = time.perf_counter() - start

        # Simulate cell execution that only modifies df1
        df1['new_col'] = df1['col_0'] * 2

        # Second checkpoint - only df1 should need recopy
        # df2 should come from cache
        start = time.perf_counter()
        pdf1_new = cache.get_or_convert(df1)  # Cache miss (modified)
        pdf2_new = cache.get_or_convert(df2)  # Cache hit (unchanged)
        second_time = time.perf_counter() - start

        print(f"\nIncremental checkpoint test:")
        print(f"  First checkpoint (both DFs): {first_time*1000:.1f}ms")
        print(f"  Second checkpoint (1 modified): {second_time*1000:.1f}ms")

        # Second should be faster if cache works for unchanged df2
        # Note: This will only work if fingerprinting detects the change correctly
