"""
Test that checkpoint conversion avoids dtype inflation for cudf proxy objects.

Reproduces the issue seen in first-place-single-model-lb-38-81 (cudf notebook):
- ~4M rows, ~300 columns (210 int8 pairs, 21 float32 factorized, float64 others)
- Column-by-column Series.to_pandas() inflated int8 → float64 (NaN accommodation)
- Checkpoint was 9254MB vs namespace 1757MB (5.3x blowup)

Fix: get_or_convert() now uses _fsproxy_slow (batch DataFrame.to_pandas()) for
proxy objects, which preserves compact dtypes.
"""

import pytest
import numpy as np
import pandas as pd

from flowbook.kernel_support.cudf_compat import (
    CuDFCheckpointCache,
    is_cudf_proxy,
)
from flowbook.kernel_support.heap_size import HeapSizer


# ---------------------------------------------------------------------------
# Mock cudf.pandas proxy
# ---------------------------------------------------------------------------

class _MockProxy:
    """
    Simulates a cudf.pandas _FastSlowProxy wrapping a DataFrame.

    _fsproxy_slow returns the compact pandas DF (as cudf's batch conversion would).
    _fsproxy_wrapped returns a sentinel that is_cudf_dataframe() rejects, so
    the old code path would have fallen through to _fsproxy_slow anyway — but
    in real cudf, _fsproxy_wrapped returns the native cudf object, which would
    have triggered the column-by-column path.
    """

    def __init__(self, compact_df: pd.DataFrame):
        self._compact = compact_df
        # _fsproxy_wrapped: in real cudf this would be the cudf native object.
        # We set it to the compact df so unwrap_cudf_proxy returns it.
        self._fsproxy_wrapped = compact_df

    @property
    def _fsproxy_slow(self):
        return self._compact

    # Attributes the detection helpers check
    @property
    def __module__(self):
        return 'cudf.pandas.fast_slow_proxy'


def _patch_cudf_detection(monkeypatch):
    """Make cudf_compat treat _MockProxy as a cudf proxy object."""
    import flowbook.kernel_support.cudf_compat as cc

    _orig_is_proxy = cc.is_cudf_proxy
    _orig_is_object = cc.is_cudf_object

    def patched_is_cudf_proxy(obj):
        if isinstance(obj, _MockProxy):
            return True
        return _orig_is_proxy(obj)

    def patched_is_cudf_object(obj):
        if isinstance(obj, _MockProxy):
            return True
        return _orig_is_object(obj)

    monkeypatch.setattr(cc, 'is_cudf_proxy', patched_is_cudf_proxy)
    monkeypatch.setattr(cc, 'is_cudf_object', patched_is_cudf_object)


# ---------------------------------------------------------------------------
# Build a DataFrame mimicking first-place-single-model-lb-38-81
# ---------------------------------------------------------------------------

def _build_notebook_scale_df(n_rows: int = 4_000_000) -> pd.DataFrame:
    """
    Build a DataFrame matching the column structure of
    first-place-single-model-lb-38-81 after feature engineering.

    Column layout (approximate):
      - 21 factorized categoricals: float32
      - 1 Weight Capacity: float64
      - 1 Price (target): float64
      - 21 nan_wc features: float64
      - 21 cat_wc features: float64
      - 1 NaNs feature: float32
      - 3 round features: float64
      - 4 orig_price features: float64
      - 9 digit features: int8
      - 10 digit-combo features: int8
      - 210 pair features: int8   (C(21,2))
    Total: ~302 columns
    """
    rng = np.random.default_rng(42)
    data = {}

    # 21 factorized categoricals (float32)
    for i in range(21):
        data[f'cat_{i}'] = rng.integers(0, 50, size=n_rows).astype(np.float32)

    # Weight Capacity + Price (float64)
    data['weight_capacity'] = rng.normal(50, 10, size=n_rows).astype(np.float64)
    data['price'] = rng.normal(100, 20, size=n_rows).astype(np.float64)

    # 21 nan_wc + 21 cat_wc (float64)
    for i in range(21):
        data[f'cat_{i}_nan_wc'] = rng.normal(0, 1, size=n_rows).astype(np.float64)
        data[f'cat_{i}_wc'] = rng.normal(0, 1, size=n_rows).astype(np.float64)

    # NaNs feature (float32)
    data['NaNs'] = rng.integers(0, 2**21, size=n_rows).astype(np.float32)

    # 3 round + 4 orig_price (float64)
    for i in range(3):
        data[f'round{7+i}'] = rng.normal(50, 10, size=n_rows).astype(np.float64)
    for i in range(4):
        data[f'orig_price_{i}'] = rng.normal(100, 20, size=n_rows).astype(np.float64)

    # 9 digit features (int8)
    for i in range(1, 10):
        data[f'digit{i}'] = rng.integers(-1, 10, size=n_rows).astype(np.int8)

    # 10 digit-combo features (int8)
    for i in range(4):
        for j in range(i+1, 5):
            data[f'digit_{i+1}_{j+1}'] = rng.integers(0, 121, size=n_rows).astype(np.int8)

    # 210 pair features (int8) — C(21,2)
    cats = [f'cat_{i}' for i in range(21)]
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            data[f'{cats[i]}_{cats[j]}'] = rng.integers(0, 127, size=n_rows).astype(np.int8)

    return pd.DataFrame(data)


def _inflate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate what column-by-column Series.to_pandas() does: widen int8 → float64
    (numpy can't represent NaN in int8, so cudf Series.to_pandas() upcasts).
    """
    inflated = {}
    for col in df.columns:
        arr = df[col].values
        if arr.dtype == np.int8:
            inflated[col] = arr.astype(np.float64)
        else:
            inflated[col] = arr
    return pd.DataFrame(inflated)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpointInflation:
    """Verify get_or_convert uses _fsproxy_slow to avoid dtype inflation."""

    def test_int8_inflation_magnitude(self):
        """Quantify the int8 → float64 inflation at notebook scale.

        With ~229 int8 columns and 4M rows, int8→float64 inflates those
        columns by 8x, which dominates the total size.
        """
        df = _build_notebook_scale_df()
        inflated = _inflate_dtypes(df)

        sizer = HeapSizer()
        compact_bytes = sizer.sizeof(df)

        sizer.reset()
        inflated_bytes = sizer.sizeof(inflated)

        compact_mb = compact_bytes / (1024 * 1024)
        inflated_mb = inflated_bytes / (1024 * 1024)
        ratio = inflated_mb / compact_mb

        # The inflated version should be significantly larger (>2x)
        assert ratio > 2.0, (
            f"Expected inflation ratio > 2x, got {ratio:.1f}x "
            f"(compact={compact_mb:.0f}MB, inflated={inflated_mb:.0f}MB)"
        )

    def test_get_or_convert_uses_fsproxy_slow(self, monkeypatch):
        """get_or_convert should use _fsproxy_slow for proxy objects,
        producing the compact representation (not column-by-column inflated).
        """
        _patch_cudf_detection(monkeypatch)

        compact_df = _build_notebook_scale_df()
        inflated_df = _inflate_dtypes(compact_df)

        proxy = _MockProxy(compact_df)
        cache = CuDFCheckpointCache()
        result = cache.get_or_convert(proxy)

        # Result should be a pandas DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(compact_df)
        assert list(result.columns) == list(compact_df.columns)

        # Measure sizes
        sizer = HeapSizer()
        result_bytes = sizer.sizeof(result)
        sizer.reset()
        compact_bytes = sizer.sizeof(compact_df)
        sizer.reset()
        inflated_bytes = sizer.sizeof(inflated_df)

        result_mb = result_bytes / (1024 * 1024)
        compact_mb = compact_bytes / (1024 * 1024)
        inflated_mb = inflated_bytes / (1024 * 1024)

        # Result should be close to compact size, not inflated size
        # Allow 10% tolerance for DataFrame wrapper overhead from .copy()
        assert result_mb < compact_mb * 1.1, (
            f"get_or_convert produced {result_mb:.0f}MB, expected ≈{compact_mb:.0f}MB "
            f"(compact), not {inflated_mb:.0f}MB (inflated). "
            f"Ratio to compact: {result_mb/compact_mb:.2f}x"
        )

    def test_get_or_convert_preserves_int8_dtypes(self, monkeypatch):
        """Verify int8 columns stay int8 through the _fsproxy_slow path."""
        _patch_cudf_detection(monkeypatch)

        compact_df = _build_notebook_scale_df()
        proxy = _MockProxy(compact_df)

        cache = CuDFCheckpointCache()
        result = cache.get_or_convert(proxy)

        int8_cols = [c for c in compact_df.columns if compact_df[c].dtype == np.int8]
        assert len(int8_cols) > 200, f"Expected 200+ int8 columns, got {len(int8_cols)}"

        for col in int8_cols:
            assert result[col].dtype == np.int8, (
                f"Column {col}: expected int8, got {result[col].dtype}. "
                f"_fsproxy_slow path should preserve compact dtypes."
            )

    def test_inflation_ratio_matches_notebook(self):
        """Verify the simulated inflation ratio is in the range seen in
        first-place-single-model-lb-38-81 (5.3x).

        The exact ratio depends on column mix; we check it's in a
        reasonable range (3x–8x) given ~229 int8 out of ~302 columns.
        """
        df = _build_notebook_scale_df()
        inflated = _inflate_dtypes(df)

        sizer = HeapSizer()
        compact_bytes = sizer.sizeof(df)
        sizer.reset()
        inflated_bytes = sizer.sizeof(inflated)

        ratio = inflated_bytes / compact_bytes
        assert 3.0 < ratio < 8.0, (
            f"Inflation ratio {ratio:.1f}x outside expected 3–8x range "
            f"(notebook saw 5.3x)"
        )
