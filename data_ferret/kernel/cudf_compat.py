"""
cuDF compatibility layer.

All cuDF-specific logic is isolated here to keep core modules clean.
When cuDF is not installed, all functions gracefully return False/None/passthrough.

Phase 0: Proxy detection and GroupBy handling (fixes recursion with cudf.pandas)
Phase 1+: Checkpoint/diff support (to be added)
"""

from __future__ import annotations

from typing import Any, Optional

# =============================================================================
# Lazy Import and Detection
# =============================================================================

_cudf_module = None
_HAS_CUDF: Optional[bool] = None


def has_cudf() -> bool:
    """Check if cuDF is available. Result is cached."""
    global _HAS_CUDF, _cudf_module
    if _HAS_CUDF is None:
        try:
            import cudf
            _cudf_module = cudf
            _HAS_CUDF = True
        except ImportError:
            _HAS_CUDF = False
    return _HAS_CUDF


def get_cudf():
    """Get cudf module, or None if not available."""
    if has_cudf():
        return _cudf_module
    return None


# =============================================================================
# Proxy Detection for cudf.pandas compatibility mode
# =============================================================================

_cudf_pandas_proxy_type = None
_HAS_CUDF_PANDAS: Optional[bool] = None


def _init_cudf_pandas_detection():
    """Initialize detection of cudf.pandas proxy types."""
    global _cudf_pandas_proxy_type, _HAS_CUDF_PANDAS
    if _HAS_CUDF_PANDAS is None:
        try:
            from cudf.pandas.fast_slow_proxy import _FastSlowProxy
            _cudf_pandas_proxy_type = _FastSlowProxy
            _HAS_CUDF_PANDAS = True
        except ImportError:
            _HAS_CUDF_PANDAS = False


def is_cudf_proxy(obj: Any) -> bool:
    """Check if object is a cudf.pandas proxy wrapper."""
    _init_cudf_pandas_detection()
    if not _HAS_CUDF_PANDAS or _cudf_pandas_proxy_type is None:
        return False
    return isinstance(obj, _cudf_pandas_proxy_type)


def is_cudf_groupby(obj: Any) -> bool:
    """Check if object is a cudf GroupBy (native or proxied)."""
    # Quick module check first (fast path)
    type_module = getattr(type(obj), '__module__', '')
    if 'cudf' in type_module:
        type_name = type(obj).__name__
        if 'GroupBy' in type_name:
            return True

    if not has_cudf():
        return False

    # Check for native cudf GroupBy
    try:
        from cudf.core.groupby.groupby import GroupBy as CudfGroupBy
        if isinstance(obj, CudfGroupBy):
            return True
    except ImportError:
        pass

    # Check for proxied cudf GroupBy
    if is_cudf_proxy(obj):
        # Check if the wrapped object is a GroupBy
        type_name = type(obj).__name__
        if 'GroupBy' in type_name:
            return True

    return False


def unwrap_cudf_proxy(obj: Any) -> Any:
    """
    Get the underlying cudf object from a proxy.

    In cudf.pandas mode, objects are wrapped in _FastSlowProxy.
    This returns the fast (cudf) object if available.
    """
    if not is_cudf_proxy(obj):
        return obj

    # _FastSlowProxy stores the fast object in _fsproxy_fast
    if hasattr(obj, '_fsproxy_fast'):
        fast_obj = obj._fsproxy_fast
        if fast_obj is not None:
            return fast_obj

    # Fallback to slow (pandas) object
    if hasattr(obj, '_fsproxy_slow'):
        return obj._fsproxy_slow

    return obj


def call_native_groupby_getitem(gb: Any, key: Any) -> Any:
    """
    Call GroupBy.__getitem__ without going through our wrapper.

    This is used when we detect a cudf proxy to avoid recursion.
    The cudf.pandas proxy system causes infinite recursion when our
    monkey-patched DataFrameGroupBy.__getitem__ calls the original method,
    because the proxy intercepts and calls our wrapper again.

    Solution: detect cudf objects and call their native __getitem__ directly.
    """
    # Get the underlying object (unwrap proxy if needed)
    unwrapped = unwrap_cudf_proxy(gb)

    # Try to call cudf's native __getitem__
    try:
        from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
        if isinstance(unwrapped, CudfDataFrameGroupBy):
            return CudfDataFrameGroupBy.__getitem__(unwrapped, key)
    except (ImportError, AttributeError, TypeError):
        pass

    # Try SeriesGroupBy
    try:
        from cudf.core.groupby.groupby import SeriesGroupBy as CudfSeriesGroupBy
        if isinstance(unwrapped, CudfSeriesGroupBy):
            return CudfSeriesGroupBy.__getitem__(unwrapped, key)
    except (ImportError, AttributeError, TypeError):
        pass

    # Fallback: use the type's __getitem__ directly
    # This avoids going through our patched pandas method
    obj_type = type(unwrapped)
    if hasattr(obj_type, '__getitem__'):
        return obj_type.__getitem__(unwrapped, key)

    # Last resort: just index directly (may still cause issues)
    return unwrapped[key]
