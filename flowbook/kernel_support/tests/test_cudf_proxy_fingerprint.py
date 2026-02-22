"""
Test demonstrating the cudf.pandas proxy fingerprinting issue.

PROBLEM:
When using `%load_ext cudf.pandas`, DataFrames become proxy objects that wrap
cudf DataFrames. The CuDFCheckpointCache._fingerprint() method doesn't handle
these proxy objects correctly, causing:
1. _fingerprint() returns None for proxy objects
2. Cache stores None as fingerprint
3. Cache "hits" because None == None, BUT...
4. The cached pandas copy may be stale if the underlying data changed

More critically, even when cache "works", accessing the proxy's _fsproxy_slow
attribute may trigger expensive operations.

IMPACT:
- 15+ second overhead per cell in notebooks with large cudf DataFrames
- GPU→CPU transfers happen on every checkpoint instead of being cached

SOLUTION:
1. Add proxy-aware fingerprinting that uses shape/dtype from proxy metadata
2. Track data hash via proxy without triggering full GPU transfer
3. Consider incremental checkpointing (only checkpoint changed DataFrames)
"""

import time
import pytest
import pandas as pd
import numpy as np

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

try:
    from cudf.pandas.fast_slow_proxy import _FastSlowProxy
    HAS_CUDF_PANDAS = True
except ImportError:
    HAS_CUDF_PANDAS = False


@pytest.mark.skipif(not HAS_CUDF, reason="cudf not available")
class TestProxyFingerprintingBug:
    """Demonstrate the proxy fingerprinting bug."""

    def test_fingerprint_returns_none_for_proxy(self):
        """BUG: _fingerprint returns None for cudf.pandas proxy objects.

        The _fingerprint method checks:
        - is_cudf_dataframe(obj) -> only true for native cudf.DataFrame
        - is_cudf_series(obj) -> only true for native cudf.Series

        For proxy objects, isinstance(proxy, cudf.DataFrame) is False,
        so _fingerprint returns None.
        """
        from flowbook.kernel_support.cudf_compat import (
            CuDFCheckpointCache,
            is_cudf_dataframe,
            is_cudf_proxy,
            _is_proxy_dataframe,
        )

        # Create a cudf.pandas proxy if available
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

        cache = CuDFCheckpointCache()

        # For native cudf, fingerprint works
        gdf = cudf.DataFrame.from_pandas(pdf)
        native_fp = cache._fingerprint(gdf)
        assert native_fp is not None, "Native cudf should have fingerprint"
        print(f"Native cudf fingerprint: {native_fp}")

        # For proxy, fingerprint returns None (THE BUG)
        if _is_proxy_dataframe(pdf):
            proxy_fp = cache._fingerprint(pdf)
            print(f"Proxy fingerprint: {proxy_fp}")
            # This is the bug - proxy fingerprint is None
            assert proxy_fp is None, "Expected None for proxy (this is the bug)"

    def test_cache_appears_to_work_but_doesnt(self):
        """BUG: Cache "hits" for proxies but with None fingerprint.

        When fingerprint is None:
        - First cache stores: (None, pandas_copy, weak_ref)
        - Second lookup: current_fp = None, cached_fp = None
        - None != None is False, so cache "hits"

        But this is a false positive - we're not actually validating
        that the data hasn't changed!
        """
        from flowbook.kernel_support.cudf_compat import (
            CuDFCheckpointCache,
            _is_proxy_dataframe,
        )

        # Create objects
        gdf = cudf.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

        cache = CuDFCheckpointCache()

        # First call
        result1 = cache.get_or_convert(gdf)
        assert result1 is not None

        # Modify the data
        gdf['a'] = [100, 200, 300]

        # Second call - should MISS because data changed
        result2 = cache.get_or_convert(gdf)

        # For native cudf, fingerprint should detect the change
        # The hash_values() sum changed, so fingerprint differs
        print(f"Result1 'a': {result1['a'].tolist()}")
        print(f"Result2 'a': {result2['a'].tolist()}")

        # The results should be different (cache miss due to data change)
        assert result2['a'].tolist() == [100, 200, 300], "Cache should detect data change"

    def test_repeated_conversion_overhead(self):
        """Measure repeated conversion overhead when fingerprint is None."""
        from flowbook.kernel_support.cudf_compat import (
            CuDFCheckpointCache,
            get_checkpoint_cache,
        )

        # Create a moderately large DataFrame
        n_rows = 200_000
        gdf = cudf.DataFrame({
            f'col_{i}': np.random.randn(n_rows) for i in range(20)
        })

        cache = CuDFCheckpointCache()

        # Measure 5 conversions
        times = []
        for i in range(5):
            start = time.perf_counter()
            _ = cache.get_or_convert(gdf)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Conversion {i+1}: {elapsed*1000:.1f}ms")

        # With working cache, times 2-5 should be nearly instant
        # With broken cache, all times will be similar
        avg_first = times[0]
        avg_rest = sum(times[1:]) / len(times[1:])

        print(f"\nFirst conversion: {avg_first*1000:.1f}ms")
        print(f"Average subsequent: {avg_rest*1000:.1f}ms")
        print(f"Speedup: {avg_first/avg_rest:.1f}x")

        # Check if cache is actually working
        if avg_rest > avg_first * 0.1:
            print("\n⚠️  Cache not effective - subsequent conversions still slow")


@pytest.mark.skipif(not HAS_CUDF, reason="cudf not available")
class TestProposedFix:
    """Test proposed fix for proxy fingerprinting."""

    def test_proxy_aware_fingerprint(self):
        """Proposed fix: fingerprint that works with proxy objects."""
        from flowbook.kernel_support.cudf_compat import (
            is_cudf_dataframe,
            is_cudf_series,
            is_cudf_index,
            _is_proxy_dataframe,
            _is_proxy_series,
            _is_proxy_index,
            is_cudf_proxy,
            unwrap_cudf_proxy,
        )

        def improved_fingerprint(obj):
            """
            Improved fingerprint that handles both native cudf and proxies.

            For proxies, we can still compute fingerprint without GPU transfer
            by accessing metadata that the proxy caches.
            """
            # Check native cudf types first
            if is_cudf_dataframe(obj):
                try:
                    data_hash = obj.hash_values().sum()
                    if hasattr(data_hash, 'item'):
                        data_hash = data_hash.item()
                except Exception:
                    data_hash = None
                return ('DataFrame', obj.shape, tuple(obj.dtypes.items()), data_hash)

            if is_cudf_series(obj):
                try:
                    data_hash = obj.hash_values().sum()
                    if hasattr(data_hash, 'item'):
                        data_hash = data_hash.item()
                except Exception:
                    data_hash = None
                return ('Series', len(obj), str(obj.dtype), data_hash)

            if is_cudf_index(obj):
                try:
                    data_hash = hash(str(obj[:10].to_pandas()))
                except Exception:
                    data_hash = None
                return ('Index', len(obj), str(obj.dtype), data_hash)

            # NEW: Handle cudf.pandas proxy objects
            if _is_proxy_dataframe(obj):
                # Unwrap to get the underlying cudf DataFrame
                unwrapped = unwrap_cudf_proxy(obj)
                if is_cudf_dataframe(unwrapped):
                    # Use native cudf fingerprinting
                    try:
                        data_hash = unwrapped.hash_values().sum()
                        if hasattr(data_hash, 'item'):
                            data_hash = data_hash.item()
                    except Exception:
                        data_hash = None
                    return ('DataFrame', unwrapped.shape, tuple(unwrapped.dtypes.items()), data_hash)
                else:
                    # Proxy with pandas slow object - use shape/dtype as fingerprint
                    # This won't detect value changes, but at least validates structure
                    return ('ProxyDataFrame', obj.shape, tuple(obj.dtypes.items()), None)

            if _is_proxy_series(obj):
                unwrapped = unwrap_cudf_proxy(obj)
                if is_cudf_series(unwrapped):
                    try:
                        data_hash = unwrapped.hash_values().sum()
                        if hasattr(data_hash, 'item'):
                            data_hash = data_hash.item()
                    except Exception:
                        data_hash = None
                    return ('Series', len(unwrapped), str(unwrapped.dtype), data_hash)
                else:
                    return ('ProxySeries', len(obj), str(obj.dtype), None)

            return None

        # Test with native cudf
        gdf = cudf.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        native_fp = improved_fingerprint(gdf)
        assert native_fp is not None
        print(f"Native fingerprint: {native_fp}")

        # Test with proxy (if available)
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        if _is_proxy_dataframe(pdf):
            proxy_fp = improved_fingerprint(pdf)
            print(f"Proxy fingerprint: {proxy_fp}")
            assert proxy_fp is not None, "Improved fingerprint should work for proxies"

    def test_incremental_checkpoint_detection(self):
        """Test detecting which DataFrames actually changed.

        Strategy: Instead of checkpointing all DataFrames every time,
        track which ones have been modified and only checkpoint those.
        """
        # Track object IDs and their last-known fingerprint
        last_fingerprints = {}

        def has_changed(name, obj, cache):
            """Check if object has changed since last checkpoint."""
            current_fp = cache._fingerprint(obj)
            last_fp = last_fingerprints.get(name)

            if last_fp is None:
                # First time seeing this object
                last_fingerprints[name] = current_fp
                return True

            if current_fp != last_fp:
                last_fingerprints[name] = current_fp
                return True

            return False

        from flowbook.kernel_support.cudf_compat import CuDFCheckpointCache

        cache = CuDFCheckpointCache()

        # Create DataFrames
        df1 = cudf.DataFrame({'a': [1, 2, 3]})
        df2 = cudf.DataFrame({'b': [4, 5, 6]})

        # First check - both new
        assert has_changed('df1', df1, cache) == True
        assert has_changed('df2', df2, cache) == True

        # Second check - neither changed
        assert has_changed('df1', df1, cache) == False
        assert has_changed('df2', df2, cache) == False

        # Modify df1
        df1['a'] = [10, 20, 30]

        # Third check - only df1 changed
        assert has_changed('df1', df1, cache) == True, "df1 should be detected as changed"
        assert has_changed('df2', df2, cache) == False, "df2 should not be changed"

        print("Incremental detection works correctly!")
