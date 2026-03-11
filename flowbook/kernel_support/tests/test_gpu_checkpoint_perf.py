"""
Performance test for GPU checkpoint deepcopy with cudf.pandas proxy fallback.

Reproduces the bottleneck found in the first-place-single-model-lb-38-81 notebook:
after `train.Price.values` materializes the slow (pandas) side of cudf.pandas
proxies, checkpoint deepcopy jumped from ~200ms to ~4400ms because
_gpu_deep_copy called proxy.copy(deep=True) which dispatched to a full pandas
deep copy instead of a fast GPU copy or shallow CoW copy.

The fix makes _gpu_deep_copy unwrap the proxy, detect which side is active,
and use _shallow_copy_for_checkpoint for the pandas case.
"""

import time
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.cudf_compat import (
    _gpu_deep_copy,
    deepcopy_cudf,
    is_cudf_proxy,
    set_gpu_checkpoint_mode,
)
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


# ---------------------------------------------------------------------------
# Helpers: fake cudf.pandas proxy objects
# ---------------------------------------------------------------------------

class _FakeProxy:
    """
    Simulates a cudf.pandas _FastSlowProxy.

    cudf.pandas wraps every DataFrame/Series in a proxy.  The proxy
    holds two representations:
      - _fsproxy_fast  (cudf / GPU)   — populated when GPU path is used
      - _fsproxy_slow  (pandas / CPU) — populated when slow path is used
      - _fsproxy_wrapped — whichever is currently active

    After an operation like `.values`, the proxy materializes the slow
    side and _fsproxy_wrapped points to the pandas DataFrame.
    """

    def __init__(self, pandas_df: pd.DataFrame, *, slow_active: bool = False):
        self._pandas = pandas_df
        self._slow_active = slow_active
        # Mimic proxy attribute used by unwrap_cudf_proxy
        if slow_active:
            self._fsproxy_wrapped = pandas_df  # pandas side active
        else:
            self._fsproxy_wrapped = _FakeCudfDataFrame(pandas_df)  # GPU side

    def copy(self, deep: bool = False) -> pd.DataFrame:
        """
        Mimic proxy.copy() dispatch.

        When slow side is active this goes through pandas, which is
        exactly the bottleneck we're fixing.
        """
        if self._slow_active:
            return self._pandas.copy(deep=deep)
        else:
            # GPU path — would be ~3ms with real cudf
            return self._pandas.copy(deep=False)

    @property
    def __class_name__(self):
        return 'DataFrame'


class _FakeCudfDataFrame:
    """Mimics a native cudf DataFrame (non-pandas)."""

    def __init__(self, pandas_df: pd.DataFrame):
        self._pandas = pandas_df

    def copy(self, deep: bool = False) -> '_FakeCudfDataFrame':
        # Real cudf.copy(deep=True) is ~3ms on GPU
        result = _FakeCudfDataFrame(self._pandas)
        return result


class _FakeProxySeries:
    """Simulates a cudf.pandas Series proxy with slow side active."""

    def __init__(self, series: pd.Series, *, slow_active: bool = False):
        self._series = series
        self._slow_active = slow_active
        if slow_active:
            self._fsproxy_wrapped = series
        else:
            self._fsproxy_wrapped = _FakeCudfSeries(series)

    def copy(self, deep: bool = False):
        if self._slow_active:
            return self._series.copy(deep=deep)
        else:
            return self._series.copy(deep=False)


class _FakeCudfSeries:
    """Mimics a native cudf Series."""
    def __init__(self, s):
        self._s = s
    def copy(self, deep=False):
        return _FakeCudfSeries(self._s)


def _is_fake_proxy(obj: Any) -> bool:
    """Check if obj is one of our fake proxy types."""
    return isinstance(obj, (_FakeProxy, _FakeProxySeries))


def _is_fake_cudf_object(obj: Any) -> bool:
    """Check if obj is a fake cudf object (proxy or native)."""
    return isinstance(obj, (_FakeProxy, _FakeProxySeries,
                            _FakeCudfDataFrame, _FakeCudfSeries))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_large_namespace(n_rows: int = 200_000, n_cols: int = 100) -> Dict[str, Any]:
    """
    Build a namespace that mirrors the first-place notebook after
    feature engineering + XGBoost training.

    Variables:
      train        — large DataFrame proxy (slow side active after .values)
      test         — large DataFrame proxy (slow side active)
      train2       — medium DataFrame proxy (slow side active)
      orig         — medium DataFrame proxy (slow side active)
      oof          — numpy array
      pred         — numpy array
      model_params — plain dict
      CATS, COMBO, FEATURES, COLS — lists
      kf           — object
    """
    rng = np.random.default_rng(42)

    # Large DataFrames as proxies with slow side active (simulates post-.values state)
    train_df = pd.DataFrame(
        rng.standard_normal((n_rows, n_cols)).astype(np.float32),
        columns=[f'col_{i}' for i in range(n_cols)],
    )
    test_df = pd.DataFrame(
        rng.standard_normal((n_rows // 2, n_cols)).astype(np.float32),
        columns=[f'col_{i}' for i in range(n_cols)],
    )
    train2_df = pd.DataFrame(
        rng.standard_normal((n_rows // 4, n_cols // 2)).astype(np.float32),
        columns=[f'col_{i}' for i in range(n_cols // 2)],
    )
    orig_df = pd.DataFrame(
        rng.standard_normal((n_rows // 4, 10)).astype(np.float32),
        columns=[f'orig_{i}' for i in range(10)],
    )

    ns = {
        # Large cudf proxy DataFrames — slow side active (the bottleneck case)
        'train': _FakeProxy(train_df, slow_active=True),
        'test': _FakeProxy(test_df, slow_active=True),
        'train2': _FakeProxy(train2_df, slow_active=True),
        'orig': _FakeProxy(orig_df, slow_active=True),
        # Numpy arrays (from oof/pred)
        'oof': rng.standard_normal(n_rows),
        'pred': rng.standard_normal(n_rows // 2),
        # Scalars and small objects
        'VER': 1,
        's': 38.81,
        'FOLDS': 7,
        'CATS': [f'cat_{i}' for i in range(8)],
        'COMBO': ['NaNs'],
        'FEATURES': [f'col_{i}' for i in range(n_cols)],
        'COLS': [f'col_{i}' for i in range(n_cols + 38)],
        'BINS': 10,
    }
    return ns


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

class TestGPUDeepCopyProxyPerformance:
    """
    End-to-end performance tests verifying that checkpoint deepcopy stays
    fast even when cudf.pandas proxies have their slow (pandas) side active.
    """

    def _patch_cudf_detection(self):
        """Patch cudf detection to recognise our fake proxies."""
        return [
            patch('flowbook.kernel_support.cudf_compat.is_cudf_proxy', side_effect=_is_fake_proxy),
            patch('flowbook.kernel_support.cudf_compat.is_cudf_object', side_effect=_is_fake_cudf_object),
            patch('flowbook.kernel_support.cudf_compat.has_cudf', return_value=True),
        ]

    # ---------------------------------------------------------------
    # Test 1: _gpu_deep_copy on a single large proxy (slow-side)
    # ---------------------------------------------------------------
    def test_single_proxy_slow_side_is_fast(self):
        """
        _gpu_deep_copy on a slow-side proxy must use shallow CoW copy,
        NOT pandas deep copy.

        Benchmark: 500K×100 float32 DataFrame
          - pandas deep copy:  ~80-200ms
          - shallow CoW copy:  ~0.1-1ms  (1000x faster)
        """
        n_rows, n_cols = 500_000, 100
        df = pd.DataFrame(
            np.random.randn(n_rows, n_cols).astype(np.float32),
            columns=[f'c{i}' for i in range(n_cols)],
        )
        proxy = _FakeProxy(df, slow_active=True)

        patches = self._patch_cudf_detection()
        for p in patches:
            p.start()
        set_gpu_checkpoint_mode(True)
        try:
            # Warm up
            _gpu_deep_copy(proxy)

            # Measure optimized path
            iters = 20
            start = time.perf_counter()
            for _ in range(iters):
                _gpu_deep_copy(proxy)
            elapsed_ms = (time.perf_counter() - start) * 1000
            per_call_ms = elapsed_ms / iters

            # Measure pandas deep copy for comparison
            start = time.perf_counter()
            for _ in range(iters):
                df.copy(deep=True)
            deep_ms = (time.perf_counter() - start) * 1000
            per_deep_ms = deep_ms / iters

            # Optimized path must be at least 10x faster than deep copy
            assert per_call_ms < per_deep_ms / 10, (
                f"_gpu_deep_copy with slow-side proxy: {per_call_ms:.2f}ms/call, "
                f"pandas deep copy: {per_deep_ms:.2f}ms/call — "
                f"expected at least 10x speedup"
            )
            # And must be under 5ms absolute (shallow copy is O(1))
            assert per_call_ms < 5.0, (
                f"_gpu_deep_copy with slow-side proxy: {per_call_ms:.2f}ms/call, "
                f"expected <5ms (shallow CoW copy)"
            )
        finally:
            set_gpu_checkpoint_mode(False)
            for p in patches:
                p.stop()

    # ---------------------------------------------------------------
    # Test 2: deepcopy_cudf through the full call chain
    # ---------------------------------------------------------------
    def test_deepcopy_cudf_slow_proxy_is_fast(self):
        """
        deepcopy_cudf (the entry point called by our custom deepcopy)
        should be fast for slow-side proxies in GPU checkpoint mode.
        """
        n_rows = 300_000
        df = pd.DataFrame({
            'price': np.random.randn(n_rows).astype(np.float32),
            'weight': np.random.randn(n_rows).astype(np.float32),
        })
        proxy = _FakeProxy(df, slow_active=True)

        patches = self._patch_cudf_detection()
        for p in patches:
            p.start()
        set_gpu_checkpoint_mode(True)
        try:
            memo: dict = {}
            start = time.perf_counter()
            result = deepcopy_cudf(proxy, memo)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Should be fast (shallow copy, <5ms)
            assert elapsed_ms < 10.0, (
                f"deepcopy_cudf on slow-side proxy took {elapsed_ms:.2f}ms, expected <10ms"
            )
            # Result should be a pandas DataFrame (shallow copy of the wrapped pandas df)
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, df)
        finally:
            set_gpu_checkpoint_mode(False)
            for p in patches:
                p.stop()

    # ---------------------------------------------------------------
    # Test 3: Full MemoryCheckpoints.save() with mixed namespace
    # ---------------------------------------------------------------
    def test_full_checkpoint_save_with_slow_proxies(self):
        """
        End-to-end: MemoryCheckpoints.save() on a namespace containing
        multiple large cudf.pandas proxies with slow side active.

        This reproduces the first-place-single-model-lb-38-81 scenario:
        - 4 large DataFrame proxies with slow side active
        - numpy arrays, scalars, lists
        - ~15 variables total

        Before fix: ~4000ms (pandas deep copy of each DataFrame)
        After fix:  ~10-50ms (shallow CoW copies + normal deepcopy for rest)
        """
        ns = _make_large_namespace(n_rows=200_000, n_cols=80)

        patches = self._patch_cudf_detection()
        for p in patches:
            p.start()
        set_gpu_checkpoint_mode(True)
        try:
            checkpoints = MemoryCheckpoints(sanity_check=False, warn_classes=False)

            # Warm up (first call may be slower due to imports)
            checkpoints.save('warmup', ns, max_size_mb=None)

            # Measure
            iters = 5
            start = time.perf_counter()
            for i in range(iters):
                checkpoints.save(f'bench_{i}', ns, max_size_mb=None)
            elapsed_ms = (time.perf_counter() - start) * 1000
            per_save_ms = elapsed_ms / iters

            # With the fix, checkpoint should complete in well under 500ms.
            # Before the fix, this took ~4000ms due to pandas deep copies.
            assert per_save_ms < 500, (
                f"MemoryCheckpoints.save() with slow-side proxies: {per_save_ms:.0f}ms/save, "
                f"expected <500ms (was ~4000ms before fix)"
            )
        finally:
            set_gpu_checkpoint_mode(False)
            for p in patches:
                p.stop()

    # ---------------------------------------------------------------
    # Test 4: Measure the before/after difference directly
    # ---------------------------------------------------------------
    def test_slow_vs_fast_proxy_checkpoint_parity(self):
        """
        Checkpointing a namespace with slow-active proxies should take
        roughly the same time as fast-active proxies (both use shallow
        or fast copy, not deep copy).
        """
        n_rows, n_cols = 200_000, 60
        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            rng.standard_normal((n_rows, n_cols)).astype(np.float32),
            columns=[f'c{i}' for i in range(n_cols)],
        )

        ns_fast = {'df': _FakeProxy(df, slow_active=False), 'x': 42}
        ns_slow = {'df': _FakeProxy(df, slow_active=True), 'x': 42}

        patches = self._patch_cudf_detection()
        for p in patches:
            p.start()
        set_gpu_checkpoint_mode(True)
        try:
            cp = MemoryCheckpoints(sanity_check=False, warn_classes=False)

            # Warm up
            cp.save('w1', ns_fast, max_size_mb=None)
            cp.save('w2', ns_slow, max_size_mb=None)

            iters = 10

            start = time.perf_counter()
            for i in range(iters):
                cp.save(f'fast_{i}', ns_fast, max_size_mb=None)
            fast_ms = (time.perf_counter() - start) * 1000 / iters

            start = time.perf_counter()
            for i in range(iters):
                cp.save(f'slow_{i}', ns_slow, max_size_mb=None)
            slow_ms = (time.perf_counter() - start) * 1000 / iters

            # Slow-side proxies should be no more than 3x slower than fast-side
            # (both use lightweight copy paths; before the fix, slow was ~1000x worse)
            assert slow_ms < fast_ms * 3 + 20, (
                f"Slow-active proxy checkpoint: {slow_ms:.1f}ms, "
                f"fast-active: {fast_ms:.1f}ms — "
                f"slow should be within 3x of fast (was ~1000x before fix)"
            )
        finally:
            set_gpu_checkpoint_mode(False)
            for p in patches:
                p.stop()

    # ---------------------------------------------------------------
    # Test 5: Values access pattern (the actual trigger)
    # ---------------------------------------------------------------
    def test_values_access_pattern(self):
        """
        Simulate the exact pattern from the notebook:
        1. Checkpoint with GPU-side active (fast)
        2. Access .values (switches to slow side)
        3. Checkpoint again — should still be fast

        This is the scenario that caused the 21x regression.
        """
        n_rows = 200_000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'Price': rng.standard_normal(n_rows).astype(np.float32),
            'Weight': rng.standard_normal(n_rows).astype(np.float32),
        })

        # Phase 1: proxy in fast mode
        proxy = _FakeProxy(df, slow_active=False)
        ns = {'train': proxy, 'VER': 1}

        patches = self._patch_cudf_detection()
        for p in patches:
            p.start()
        set_gpu_checkpoint_mode(True)
        try:
            cp = MemoryCheckpoints(sanity_check=False, warn_classes=False)

            start = time.perf_counter()
            cp.save('before_values', ns, max_size_mb=None)
            before_ms = (time.perf_counter() - start) * 1000

            # Phase 2: simulate .values access — proxy switches to slow side
            proxy_slow = _FakeProxy(df, slow_active=True)
            ns_after = {
                'train': proxy_slow,
                'VER': 1,
                'true': df['Price'].values,  # numpy array from .values
                's': 38.81,
            }

            start = time.perf_counter()
            cp.save('after_values', ns_after, max_size_mb=None)
            after_ms = (time.perf_counter() - start) * 1000

            # After .values, checkpoint should not regress by more than 3x
            # Before the fix, this was ~20x slower
            assert after_ms < before_ms * 3 + 50, (
                f"Checkpoint after .values: {after_ms:.1f}ms, "
                f"before .values: {before_ms:.1f}ms — "
                f"should not regress by more than 3x (was ~21x before fix)"
            )
        finally:
            set_gpu_checkpoint_mode(False)
            for p in patches:
                p.stop()
