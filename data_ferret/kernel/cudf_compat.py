"""
cuDF compatibility layer.

All cuDF-specific logic is isolated here to keep core modules clean.
When cuDF is not installed, all functions gracefully return False/None/passthrough.

Phase 0: Proxy detection and GroupBy handling (fixes recursion with cudf.pandas)
Phase 0.5: cuDF column tracking via method patching
Phase 1+: Checkpoint/diff support (to be added)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from data_ferret.kernel.column_tracking import ColumnAccessTracker

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


# =============================================================================
# cuDF Column Tracking via Method Patching
# =============================================================================

# Storage for original cudf methods
_cudf_original_methods: Dict[str, Any] = {}
_cudf_patches_installed: bool = False
_cudf_tracker: Optional['ColumnAccessTracker'] = None
# Mapping from cudf GroupBy id -> source DataFrame id
_cudf_groupby_to_df: Dict[int, int] = {}


def install_cudf_tracking(tracker: 'ColumnAccessTracker') -> None:
    """
    Install column tracking patches on cudf DataFrame methods.

    This mirrors the pandas patching in ColumnAccessTracker but for cudf.
    All cudf-specific patching logic is contained here.

    Args:
        tracker: The ColumnAccessTracker instance to record reads/writes to
    """
    global _cudf_patches_installed, _cudf_tracker, _cudf_groupby_to_df

    if not has_cudf():
        return

    if _cudf_patches_installed:
        return

    _cudf_tracker = tracker
    _cudf_groupby_to_df.clear()

    cudf = get_cudf()

    # ========== cudf.DataFrame.__getitem__ ==========
    _cudf_original_methods['DataFrame.__getitem__'] = cudf.DataFrame.__getitem__
    original_getitem = _cudf_original_methods['DataFrame.__getitem__']

    def tracked_cudf_getitem(df, key):
        if _cudf_tracker is not None:
            if isinstance(key, str):
                _cudf_tracker.record_read(id(df), [key])
            elif isinstance(key, list):
                str_keys = [k for k in key if isinstance(k, str)]
                if str_keys:
                    _cudf_tracker.record_read(id(df), str_keys)
        return original_getitem(df, key)

    cudf.DataFrame.__getitem__ = tracked_cudf_getitem

    # ========== cudf.DataFrame.__setitem__ ==========
    _cudf_original_methods['DataFrame.__setitem__'] = cudf.DataFrame.__setitem__
    original_setitem = _cudf_original_methods['DataFrame.__setitem__']

    def tracked_cudf_setitem(df, key, value):
        if _cudf_tracker is not None:
            if isinstance(key, str):
                _cudf_tracker.record_write(id(df), [key])
            elif isinstance(key, list):
                str_keys = [k for k in key if isinstance(k, str)]
                if str_keys:
                    _cudf_tracker.record_write(id(df), str_keys)
        return original_setitem(df, key, value)

    cudf.DataFrame.__setitem__ = tracked_cudf_setitem

    # ========== cudf.DataFrame.groupby ==========
    _cudf_original_methods['DataFrame.groupby'] = cudf.DataFrame.groupby
    original_groupby = _cudf_original_methods['DataFrame.groupby']

    def tracked_cudf_groupby(df, by=None, *args, **kwargs):
        if _cudf_tracker is not None and by is not None:
            if isinstance(by, str):
                _cudf_tracker.record_read(id(df), [by])
            elif isinstance(by, list):
                str_keys = [k for k in by if isinstance(k, str)]
                if str_keys:
                    _cudf_tracker.record_read(id(df), str_keys)
        result = original_groupby(df, by=by, *args, **kwargs)
        # Store mapping from GroupBy -> DataFrame
        _cudf_groupby_to_df[id(result)] = id(df)
        return result

    cudf.DataFrame.groupby = tracked_cudf_groupby

    # ========== cudf DataFrameGroupBy.__getitem__ ==========
    try:
        from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
        _cudf_original_methods['DataFrameGroupBy.__getitem__'] = CudfDataFrameGroupBy.__getitem__
        original_gb_getitem = _cudf_original_methods['DataFrameGroupBy.__getitem__']

        def tracked_cudf_gb_getitem(gb, key):
            if _cudf_tracker is not None:
                df_id = _cudf_groupby_to_df.get(id(gb))
                if df_id is not None:
                    if isinstance(key, str):
                        _cudf_tracker.record_read(df_id, [key])
                    elif isinstance(key, list):
                        str_keys = [k for k in key if isinstance(k, str)]
                        if str_keys:
                            _cudf_tracker.record_read(df_id, str_keys)
            return original_gb_getitem(gb, key)

        CudfDataFrameGroupBy.__getitem__ = tracked_cudf_gb_getitem
    except (ImportError, AttributeError):
        pass

    # ========== cudf.DataFrame.merge ==========
    _cudf_original_methods['DataFrame.merge'] = cudf.DataFrame.merge
    original_merge = _cudf_original_methods['DataFrame.merge']

    def tracked_cudf_merge(df, right, how='inner', on=None, left_on=None,
                           right_on=None, *args, **kwargs):
        if _cudf_tracker is not None:
            # Track columns read from left DataFrame
            if on is not None:
                cols = [on] if isinstance(on, str) else list(on)
                _cudf_tracker.record_read(id(df), cols)
            if left_on is not None:
                cols = [left_on] if isinstance(left_on, str) else list(left_on)
                _cudf_tracker.record_read(id(df), cols)

            # Track columns read from right DataFrame
            if hasattr(right, '__class__') and 'DataFrame' in right.__class__.__name__:
                if on is not None:
                    cols = [on] if isinstance(on, str) else list(on)
                    _cudf_tracker.record_read(id(right), cols)
                if right_on is not None:
                    cols = [right_on] if isinstance(right_on, str) else list(right_on)
                    _cudf_tracker.record_read(id(right), cols)

        return original_merge(df, right, how=how, on=on, left_on=left_on,
                              right_on=right_on, *args, **kwargs)

    cudf.DataFrame.merge = tracked_cudf_merge

    # ========== cudf.DataFrame.sort_values ==========
    _cudf_original_methods['DataFrame.sort_values'] = cudf.DataFrame.sort_values
    original_sort = _cudf_original_methods['DataFrame.sort_values']

    def tracked_cudf_sort_values(df, by, *args, **kwargs):
        if _cudf_tracker is not None:
            cols = [by] if isinstance(by, str) else list(by)
            _cudf_tracker.record_read(id(df), cols)
        return original_sort(df, by, *args, **kwargs)

    cudf.DataFrame.sort_values = tracked_cudf_sort_values

    # ========== cudf.DataFrame.drop_duplicates ==========
    _cudf_original_methods['DataFrame.drop_duplicates'] = cudf.DataFrame.drop_duplicates
    original_drop_dup = _cudf_original_methods['DataFrame.drop_duplicates']

    def tracked_cudf_drop_duplicates(df, subset=None, *args, **kwargs):
        if _cudf_tracker is not None and subset is not None:
            cols = [subset] if isinstance(subset, str) else list(subset)
            _cudf_tracker.record_read(id(df), cols)
        return original_drop_dup(df, subset=subset, *args, **kwargs)

    cudf.DataFrame.drop_duplicates = tracked_cudf_drop_duplicates

    _cudf_patches_installed = True


def uninstall_cudf_tracking() -> None:
    """
    Restore original cudf DataFrame methods.
    """
    global _cudf_patches_installed, _cudf_tracker, _cudf_groupby_to_df

    if not has_cudf():
        return

    if not _cudf_patches_installed:
        return

    cudf = get_cudf()

    # Restore DataFrame methods
    if 'DataFrame.__getitem__' in _cudf_original_methods:
        cudf.DataFrame.__getitem__ = _cudf_original_methods['DataFrame.__getitem__']
    if 'DataFrame.__setitem__' in _cudf_original_methods:
        cudf.DataFrame.__setitem__ = _cudf_original_methods['DataFrame.__setitem__']
    if 'DataFrame.groupby' in _cudf_original_methods:
        cudf.DataFrame.groupby = _cudf_original_methods['DataFrame.groupby']
    if 'DataFrame.merge' in _cudf_original_methods:
        cudf.DataFrame.merge = _cudf_original_methods['DataFrame.merge']
    if 'DataFrame.sort_values' in _cudf_original_methods:
        cudf.DataFrame.sort_values = _cudf_original_methods['DataFrame.sort_values']
    if 'DataFrame.drop_duplicates' in _cudf_original_methods:
        cudf.DataFrame.drop_duplicates = _cudf_original_methods['DataFrame.drop_duplicates']

    # Restore GroupBy methods
    try:
        from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
        if 'DataFrameGroupBy.__getitem__' in _cudf_original_methods:
            CudfDataFrameGroupBy.__getitem__ = _cudf_original_methods['DataFrameGroupBy.__getitem__']
    except (ImportError, AttributeError):
        pass

    _cudf_original_methods.clear()
    _cudf_groupby_to_df.clear()
    _cudf_tracker = None
    _cudf_patches_installed = False


def reset_cudf_tracking() -> None:
    """Reset cudf tracking state (groupby mappings) for new cell execution."""
    global _cudf_groupby_to_df
    _cudf_groupby_to_df.clear()


def is_cudf_dataframe(obj: Any) -> bool:
    """Check if object is a cudf DataFrame."""
    if not has_cudf():
        return False
    return isinstance(obj, _cudf_module.DataFrame)
