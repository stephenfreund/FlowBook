"""
cuDF Compatibility Layer

All cuDF-specific logic is isolated here to keep core modules clean.
When cuDF is not installed, all functions gracefully return False/None/passthrough.

================================================================================
OVERVIEW
================================================================================

This module provides transparent support for NVIDIA cuDF (GPU-accelerated pandas)
in the FlowBook kernel. It handles two modes of cuDF usage:

1. Native cuDF: Direct use of cudf.DataFrame, cudf.Series, cudf.Index
2. cudf.pandas: Proxy mode where pandas API is transparently accelerated

The key challenge is that cudf.pandas uses proxy objects that wrap pandas
objects but appear as pandas types to isinstance checks. This module provides
detection and conversion utilities that work with both modes.

================================================================================
ARCHITECTURE
================================================================================

Lazy Import Pattern
-------------------
cuDF is imported lazily to avoid startup overhead when not used:
- has_cudf() → cached check for cudf availability
- get_cudf() → returns cudf module or None
- _cudf_module → cached module reference

Proxy Detection
---------------
cudf.pandas wraps objects in _FastSlowProxy from cudf.pandas.fast_slow_proxy:
- is_cudf_proxy(obj) → True if obj is a proxy wrapper
- Proxies have _fsproxy_slow (pandas) and _fsproxy_fast (cudf) attributes

Type-Specific Proxy Detection
-----------------------------
For dispatch purposes, we need to know WHAT TYPE of proxy:
- _is_proxy_dataframe(obj) → proxy wrapping a DataFrame
- _is_proxy_series(obj) → proxy wrapping a Series
- _is_proxy_index(obj) → proxy wrapping an Index

These check is_cudf_proxy() AND type(obj).__name__ to determine the wrapped type.

Unified Detection
-----------------
is_cudf_object(obj) returns True for ANY cudf-related object:
- Native cudf.DataFrame, cudf.Series, cudf.Index
- cudf.pandas proxy DataFrames, Series, Indexes
- Guards against _cudf_module being None (proxy-only mode)

================================================================================
KEY FUNCTIONS
================================================================================

Detection Functions
-------------------
- has_cudf() → bool: Is cuDF installed?
- is_cudf_proxy(obj) → bool: Is obj a cudf.pandas proxy?
- is_cudf_object(obj) → bool: Is obj any cuDF type (native or proxy)?
- is_cudf_dataframe(obj) → bool: Native cudf.DataFrame?
- is_cudf_series(obj) → bool: Native cudf.Series?
- is_cudf_groupby(obj) → bool: Native or proxied GroupBy?
- are_both_cudf_same_type(a, b) → bool: Both same cuDF type? (for diff dispatch)

Conversion Functions
--------------------
- to_pandas(obj) → pandas object:
    Converts cuDF objects to pandas for checkpointing/diffing.
    For proxies: extracts _fsproxy_slow (the underlying pandas object)
    For native: calls obj.to_pandas()
    Returns copy to ensure independence.

- get_or_convert(obj, cache) → pandas object:
    Like to_pandas() but with caching for repeated conversions.
    Uses CuDFCheckpointCache for efficient checkpoint operations.

Diff Integration
----------------
- diff_cudf(val_a, val_b, path, differ) → DiffNode:
    Entry point for comparing cuDF objects.
    Converts both to pandas and delegates to differ._compare_dataframe/series.
    Handles DataFrame, Series, and Index types.

Deepcopy Integration
--------------------
- deepcopy_cudf(obj, memo) → object:
    Deep copies cuDF objects by converting to pandas.
    Uses get_or_convert() with memo-based caching.

================================================================================
CUDF.PANDAS PROXY HANDLING
================================================================================

cudf.pandas provides transparent GPU acceleration by wrapping pandas objects.
The proxy pattern creates challenges:

1. Type Identity: type(proxy).__name__ == 'DataFrame' but
   isinstance(proxy, pd.DataFrame) may be False

2. Method Access: Proxies intercept attribute access, which can cause
   infinite recursion if we use hasattr() in our own __getattribute__

3. Underlying Object: The _fsproxy_slow attribute contains the pandas object
   - May be a callable (method) or direct reference
   - May be None in edge cases

Solution: Extract pandas object via _fsproxy_slow:
    if is_cudf_proxy(obj):
        slow_obj = obj._fsproxy_slow
        if callable(slow_obj):
            slow_obj = slow_obj()
        return slow_obj.copy()  # Ensure independence

================================================================================
CHECKPOINT/RESTORE INTEGRATION
================================================================================

CuDFOriginTracker
-----------------
Tracks which variables were originally cuDF objects before checkpoint:
- record(var_name, obj): Store origin info if obj is cuDF
- get_origin(var_name) → CuDFOrigin: Get original type info
- should_convert_back(var_name) → bool: Should restore as cuDF?

During checkpoint, cuDF objects are converted to pandas (GPU→CPU).
During restore, CuDFOriginTracker enables conversion back to cuDF.

CuDFCheckpointCache
-------------------
Caches to_pandas() results during a single checkpoint operation:
- Avoids redundant GPU→CPU transfers
- Keyed by object ID
- Cleared after each checkpoint

================================================================================
COLUMN TRACKING INTEGRATION
================================================================================

patch_cudf_column_tracking(tracker)
-----------------------------------
Installs method patches on cudf.DataFrame for column access tracking:
- __getitem__ → track column reads
- __setitem__ → track column writes
- Similar to pandas patching in column_tracking.py

unpatch_cudf_column_tracking()
------------------------------
Removes patches after cell execution.

================================================================================
USAGE
================================================================================

Checking for cuDF:
    >>> if cudf_compat.is_cudf_object(df):
    ...     pandas_df = cudf_compat.to_pandas(df)

In diff dispatch:
    >>> if cudf_compat.are_both_cudf_same_type(val_a, val_b):
    ...     return cudf_compat.diff_cudf(val_a, val_b, path, self)

In deepcopy:
    >>> if cudf_compat.is_cudf_object(obj):
    ...     return cudf_compat.deepcopy_cudf(obj, memo)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from flowbook.util.output import timer

if TYPE_CHECKING:
    from flowbook.kernel_support.column_tracking import ColumnAccessTracker

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
    Get the underlying object from a cudf.pandas proxy.

    In cudf.pandas mode, objects are wrapped in _FastSlowProxy.
    Returns the fast (cudf) object if it is already materialized, which is
    needed for GPU-based fingerprinting, buffer deduplication, and efficient
    conversion.  Falls back to the slow (pandas) object otherwise.

    IMPORTANT: We must NOT access _fsproxy_fast directly — it is a property
    that triggers slow→fast conversion, which can raise NotImplementedError
    for certain column types (e.g., category dtype after factorize()).
    Instead we read _fsproxy_wrapped, which holds whichever version is
    currently materialized without triggering any conversion.
    """
    if not is_cudf_proxy(obj):
        return obj

    # _fsproxy_wrapped holds the currently materialized object (fast or slow)
    # without triggering any conversion.  If it's a cudf object, use it
    # for fingerprinting and buffer dedup.  If it's pandas, that's fine too.
    if hasattr(obj, '_fsproxy_wrapped'):
        wrapped = obj._fsproxy_wrapped
        if wrapped is not None:
            return wrapped

    # Fallback to slow (pandas) object — safe, never triggers fast conversion
    if hasattr(obj, '_fsproxy_slow'):
        slow_obj = obj._fsproxy_slow
        if callable(slow_obj):
            slow_obj = slow_obj()
        if slow_obj is not None:
            return slow_obj

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
    if not has_cudf() or _cudf_module is None:
        return False
    return isinstance(obj, _cudf_module.DataFrame)


def is_cudf_series(obj: Any) -> bool:
    """Check if object is a cudf Series."""
    if not has_cudf() or _cudf_module is None:
        return False
    return isinstance(obj, _cudf_module.Series)


def is_cudf_index(obj: Any) -> bool:
    """Check if object is a cudf Index."""
    if not has_cudf() or _cudf_module is None:
        return False
    return isinstance(obj, _cudf_module.Index)


def is_cudf_object(obj: Any) -> bool:
    """Check if object is any cudf type that needs special checkpoint handling.

    Handles both native cudf objects and cudf.pandas proxy objects.
    """
    if not has_cudf():
        return False
    # Check native cudf types (guard against _cudf_module being None in proxy-only mode)
    if _cudf_module is not None:
        if isinstance(obj, (_cudf_module.DataFrame, _cudf_module.Series, _cudf_module.Index)):
            return True
    # Check cudf.pandas proxy types (DataFrame, Series, or Index proxies)
    if is_cudf_proxy(obj):
        type_name = type(obj).__name__
        return type_name in ('DataFrame', 'Series') or 'Index' in type_name
    return False


def get_cudf_type(obj: Any) -> Optional[type]:
    """Get the cudf type of an object, or None if not cudf."""
    if is_cudf_dataframe(obj):
        return _cudf_module.DataFrame
    elif is_cudf_series(obj):
        return _cudf_module.Series
    elif is_cudf_index(obj):
        return _cudf_module.Index
    return None


# =============================================================================
# GPU Buffer Detection for Deduplication
# =============================================================================

def _get_column_buffer_info(col: Any) -> Optional[tuple]:
    """
    Get GPU buffer info for a cudf column.

    Returns (buffer_ptr, offset, size) or None if not detectable.
    - buffer_ptr: GPU memory pointer for the BASE underlying buffer (shared by views)
    - offset: Offset into the buffer (for views/slices)
    - size: Number of elements in this column

    Columns sharing the same buffer_ptr with different offsets are views
    of the same underlying data and can share CPU memory after conversion.
    """
    try:
        # cudf columns have a _column attribute with the actual data
        if hasattr(col, '_column'):
            col = col._column

        # Get the BASE data buffer (the shared buffer that views point to)
        # This is critical for detecting views - use base_data first, not data
        data = None
        if hasattr(col, 'base_data') and col.base_data is not None:
            data = col.base_data
        elif hasattr(col, 'data'):
            data = col.data

        if data is None:
            return None

        # Get buffer pointer from base buffer
        buf_ptr = None
        if hasattr(data, '__cuda_array_interface__'):
            buf_ptr = data.__cuda_array_interface__['data'][0]
        elif hasattr(data, 'ptr'):
            buf_ptr = data.ptr

        if buf_ptr is None:
            return None

        # Get offset (for views/slices)
        offset = getattr(col, 'offset', 0) or 0

        # Get size
        size = len(col) if hasattr(col, '__len__') else 0

        return (buf_ptr, offset, size)
    except Exception:
        return None


def get_dataframe_buffer_map(df: Any) -> Dict[str, tuple]:
    """
    Get GPU buffer info for all columns in a cudf DataFrame.

    Returns dict mapping column_name -> (buffer_ptr, offset, size).
    Columns with the same buffer_ptr share underlying GPU memory.

    Works with both native cudf.DataFrame and cudf.pandas proxy DataFrames.
    """
    result = {}

    # Unwrap proxy if needed
    if is_cudf_proxy(df):
        unwrapped = unwrap_cudf_proxy(df)
        if unwrapped is not None and unwrapped is not df:
            df = unwrapped

    # Get column data
    try:
        if hasattr(df, '_data'):
            # Native cudf DataFrame
            for col_name in df._data.names:
                col = df._data[col_name]
                info = _get_column_buffer_info(col)
                if info is not None:
                    result[col_name] = info
        elif hasattr(df, 'columns'):
            # Try accessing columns directly
            for col_name in df.columns:
                try:
                    col = df[col_name]
                    info = _get_column_buffer_info(col)
                    if info is not None:
                        result[col_name] = info
                except Exception:
                    pass
    except Exception:
        pass

    return result


# =============================================================================
# Hash-Based Checkpoint Cache
# =============================================================================

import weakref


class CuDFCheckpointCache:
    """
    Cache for GPU→CPU conversions to avoid repeated expensive copies.

    Uses a fingerprint (shape + dtypes + data hash) to detect changes.
    If the same cuDF object is checkpointed again and hasn't changed,
    we return the cached pandas copy instead of copying from GPU again.

    The cache uses weak references to cuDF objects as keys, so entries
    are automatically removed when the cuDF object is garbage collected.

    Two-Level Caching:
    - Level 1 (_cache): GPU→pandas conversion cache
    - Level 2 (_deepcopy_cache): Stores already-deepcopied pandas for O(1) reuse

    On cache HIT, we return a shallow copy of the cached deepcopy, which is
    safe because:
    1. pandas CoW protects data buffers (mutations create new arrays)
    2. pandas Index is immutable ("Index does not support mutable operations")
    3. cudf object columns contain only immutable types (strings, tuples)
    """

    def __init__(self):
        # Maps id(cudf_obj) -> (fingerprint, pandas_copy, weak_ref)
        self._cache: Dict[int, tuple] = {}
        # Maps id(cudf_obj) -> deepcopied pandas object
        self._deepcopy_cache: Dict[int, Any] = {}
        # Buffer-based cache for deduplication: maps (buffer_ptr, dtype) -> numpy array
        # This allows detecting when multiple cudf objects share underlying GPU data
        # and reusing the same numpy array (or views of it) in checkpoints.
        self._buffer_cache: Dict[tuple, 'np.ndarray'] = {}

    def _fingerprint(self, obj: Any) -> Optional[tuple]:
        """
        Compute a cheap fingerprint of a cuDF object.

        This runs on the GPU and is much faster than a full GPU→CPU copy.
        The fingerprint includes shape, dtypes, and a hash of the data.

        Handles both native cudf objects and cudf.pandas proxy objects.
        """
        # Handle native cudf types
        if is_cudf_dataframe(obj):
            try:
                data_hash = obj.hash_values().sum()
                if hasattr(data_hash, 'item'):
                    data_hash = data_hash.item()
            except Exception:
                data_hash = None
            return ('DataFrame', obj.shape, tuple(obj.dtypes.items()), data_hash)

        elif is_cudf_series(obj):
            try:
                data_hash = obj.hash_values().sum()
                if hasattr(data_hash, 'item'):
                    data_hash = data_hash.item()
            except Exception:
                data_hash = None
            return ('Series', len(obj), str(obj.dtype), data_hash)

        elif is_cudf_index(obj):
            try:
                data_hash = hash(str(obj[:10].to_pandas()))
            except Exception:
                data_hash = None
            return ('Index', len(obj), str(obj.dtype), data_hash)

        # Handle cudf.pandas proxy objects
        if _is_proxy_dataframe(obj):
            try:
                unwrapped = unwrap_cudf_proxy(obj)
                if is_cudf_dataframe(unwrapped):
                    try:
                        data_hash = unwrapped.hash_values().sum()
                        if hasattr(data_hash, 'item'):
                            data_hash = data_hash.item()
                    except Exception:
                        data_hash = None
                    return ('ProxyDataFrame', unwrapped.shape,
                            tuple(unwrapped.dtypes.items()), data_hash)
                else:
                    # Proxy with pandas slow object - use shape/dtype
                    return ('ProxyDataFrame', obj.shape, tuple(obj.dtypes.items()), None)
            except Exception:
                return None

        if _is_proxy_series(obj):
            try:
                unwrapped = unwrap_cudf_proxy(obj)
                if is_cudf_series(unwrapped):
                    try:
                        data_hash = unwrapped.hash_values().sum()
                        if hasattr(data_hash, 'item'):
                            data_hash = data_hash.item()
                    except Exception:
                        data_hash = None
                    return ('ProxySeries', len(unwrapped), str(unwrapped.dtype), data_hash)
                else:
                    return ('ProxySeries', len(obj), str(obj.dtype), None)
            except Exception:
                return None

        if _is_proxy_index(obj):
            try:
                unwrapped = unwrap_cudf_proxy(obj)
                if is_cudf_index(unwrapped):
                    try:
                        data_hash = hash(str(unwrapped[:10].to_pandas()))
                    except Exception:
                        data_hash = None
                    return ('ProxyIndex', len(unwrapped), str(unwrapped.dtype), data_hash)
                else:
                    try:
                        data_hash = hash(str(obj[:10]))
                    except Exception:
                        data_hash = None
                    return ('ProxyIndex', len(obj), str(obj.dtype), data_hash)
            except Exception:
                return None

        return None

    def get_cached(self, obj: Any) -> Optional[Any]:
        """
        Get cached pandas copy if object hasn't changed.

        Returns None if not in cache or fingerprint changed.
        Also invalidates the deepcopy cache when data changes.
        """
        obj_id = id(obj)

        if obj_id not in self._cache:
            return None

        cached_fp, pandas_copy, weak_ref = self._cache[obj_id]

        # Verify weak reference still points to same object
        ref_obj = weak_ref()
        if ref_obj is None or ref_obj is not obj:
            del self._cache[obj_id]
            self._deepcopy_cache.pop(obj_id, None)  # Invalidate deepcopy cache
            return None

        # Verify fingerprint matches
        current_fp = self._fingerprint(obj)
        if current_fp != cached_fp:
            del self._cache[obj_id]
            self._deepcopy_cache.pop(obj_id, None)  # Invalidate deepcopy cache
            return None

        return pandas_copy

    def has_valid_cache(self, obj: Any) -> bool:
        """Check if we have a valid cached copy without triggering conversion."""
        return self.get_cached(obj) is not None

    def get_deepcopied(self, obj_id: int) -> Optional[Any]:
        """Get the cached deepcopied version if available."""
        return self._deepcopy_cache.get(obj_id)

    def cache_deepcopy(self, obj_id: int, deepcopied: Any) -> None:
        """Cache the deepcopied version for future use."""
        self._deepcopy_cache[obj_id] = deepcopied

    def cache(self, obj: Any, pandas_copy: Any) -> None:
        """Store a pandas copy in the cache."""
        obj_id = id(obj)
        fp = self._fingerprint(obj)

        try:
            weak_ref = weakref.ref(obj)
        except TypeError:
            return

        self._cache[obj_id] = (fp, pandas_copy, weak_ref)

    def _get_or_create_column_array(
        self, col: Any, col_name: str, buf_info: Optional[tuple], dtype_hint: Optional[str] = None
    ) -> 'np.ndarray':
        """
        Get or create a numpy array for a cudf column, with buffer-based deduplication.

        If this column shares a GPU buffer with a previously converted column,
        returns a view into the cached array instead of creating a new copy.
        """
        import numpy as np

        if buf_info is not None:
            buf_ptr, offset, size = buf_info

            # Try to get dtype from the column without converting
            if dtype_hint is None:
                try:
                    if hasattr(col, 'dtype'):
                        dtype_hint = str(col.dtype)
                except Exception:
                    pass

            # Check cache BEFORE converting to avoid unnecessary GPU->CPU copy
            if dtype_hint:
                cache_key = (buf_ptr, dtype_hint)
                if cache_key in self._buffer_cache:
                    cached_arr = self._buffer_cache[cache_key]

                    # If sizes match and offset is 0, return cached array directly
                    if offset == 0 and size == len(cached_arr):
                        return cached_arr

                    # If this is a prefix/view of a larger cached array, return a view
                    if size <= len(cached_arr):
                        if offset == 0:
                            # Prefix view (e.g., train.iloc[:80000])
                            return cached_arr[:size]
                        elif offset + size <= len(cached_arr):
                            # Slice view (e.g., train.iloc[80000:100000])
                            return cached_arr[offset:offset + size]

                    # If the cached array is smaller, we need to convert and update cache
                    # Fall through to conversion below

        # No cache hit - convert column to numpy
        if hasattr(col, 'to_pandas'):
            pandas_col = col.to_pandas()
            arr = pandas_col.values if hasattr(pandas_col, 'values') else np.asarray(pandas_col)
        else:
            arr = np.asarray(col)

        if buf_info is None:
            return arr

        buf_ptr, offset, size = buf_info
        cache_key = (buf_ptr, str(arr.dtype))

        if cache_key in self._buffer_cache:
            cached_arr = self._buffer_cache[cache_key]
            # We got here because the cached array was smaller - update cache if this is larger
            if len(arr) > len(cached_arr):
                self._buffer_cache[cache_key] = arr
                return arr
            # Otherwise return the newly converted array (shouldn't happen often)
            return arr

        # Not in cache - store and return
        self._buffer_cache[cache_key] = arr
        return arr

    def _convert_dataframe_with_sharing(self, df: Any) -> Any:
        """
        Convert a cudf DataFrame to pandas with buffer-based deduplication.

        Columns that share GPU memory will share numpy arrays in the result.
        """
        import pandas as pd

        # Get buffer map for this DataFrame
        buffer_map = get_dataframe_buffer_map(df)

        # Unwrap proxy for conversion
        cudf_df = df
        if is_cudf_proxy(df):
            unwrapped = unwrap_cudf_proxy(df)
            if unwrapped is not None:
                cudf_df = unwrapped

        # Convert each column with deduplication
        result_data = {}
        for col_name in df.columns:
            buf_info = buffer_map.get(col_name)
            try:
                col = cudf_df[col_name] if hasattr(cudf_df, '__getitem__') else df[col_name]
                arr = self._get_or_create_column_array(col, col_name, buf_info)
                result_data[col_name] = arr
            except Exception:
                # Fallback: convert normally
                if hasattr(df[col_name], 'to_pandas'):
                    result_data[col_name] = df[col_name].to_pandas()
                else:
                    result_data[col_name] = df[col_name]

        # Convert index
        try:
            if hasattr(df, 'index') and hasattr(df.index, 'to_pandas'):
                index = df.index.to_pandas()
            else:
                index = df.index if hasattr(df, 'index') else None
        except Exception:
            index = None

        # Use copy=False to preserve view relationships with cached arrays
        # This enables memory deduplication: views into cached arrays remain views
        return pd.DataFrame(result_data, index=index, copy=False)

    def get_or_convert(self, obj: Any) -> Any:
        """Get cached pandas copy or convert from GPU.

        For proxy objects, prefer _fsproxy_slow (cudf's batch DataFrame.to_pandas())
        over column-by-column conversion via _convert_dataframe_with_sharing.

        The batch path preserves compact dtypes (e.g., int8, float32), while
        column-by-column Series.to_pandas() can inflate dtypes (e.g., int8 →
        float64 to accommodate NaN in numpy). In first-place-single-model-lb-38-81
        (cudf notebook with ~210 int8 pair columns + ~21 float32 factorized
        columns), this inflation caused checkpoint_mb = 9254MB vs namespace
        1757MB — a 5.3x blowup from dtype widening alone.
        """
        cached = self.get_cached(obj)
        if cached is not None:
            return cached

        # Handle cudf.pandas proxy objects — use _fsproxy_slow for compact conversion
        if is_cudf_proxy(obj):
            if hasattr(obj, '_fsproxy_slow'):
                slow_obj = obj._fsproxy_slow
                if callable(slow_obj):
                    slow_obj = slow_obj()
                if slow_obj is not None:
                    pandas_copy = slow_obj.copy() if hasattr(slow_obj, 'copy') else slow_obj
                else:
                    pandas_copy = obj.to_pandas() if hasattr(obj, 'to_pandas') else obj
            elif hasattr(obj, 'to_pandas'):
                pandas_copy = obj.to_pandas()
            else:
                pandas_copy = obj
        elif is_cudf_dataframe(obj):
            # Native cudf DataFrame (not a proxy) - use buffer-aware conversion
            pandas_copy = self._convert_dataframe_with_sharing(obj)
        else:
            # Native cudf Series/Index
            pandas_copy = obj.to_pandas()

        self.cache(obj, pandas_copy)
        return pandas_copy

    def clear(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self._deepcopy_cache.clear()
        self._buffer_cache.clear()


# Global cache instance
_checkpoint_cache = CuDFCheckpointCache()


def get_checkpoint_cache() -> CuDFCheckpointCache:
    """Get the global checkpoint cache."""
    return _checkpoint_cache


# =============================================================================
# Origin Tracking for Restore
# =============================================================================

class CuDFOriginTracker:
    """
    Tracks which checkpoint values originated from cuDF objects.

    Used during restore to convert pandas back to cuDF.
    """

    def __init__(self):
        self._origins: Dict[str, type] = {}

    def record(self, name: str, obj: Any) -> None:
        """Record that a variable originated from a cuDF object."""
        cudf_type = get_cudf_type(obj)
        if cudf_type is not None:
            self._origins[name] = cudf_type

    def get_origin(self, name: str) -> Optional[type]:
        """Get the original cuDF type for a variable, or None."""
        return self._origins.get(name)

    def restore_value(self, name: str, value: Any) -> Any:
        """Restore a value, converting to cuDF if it originated from cuDF."""
        origin_type = self.get_origin(name)
        if origin_type is not None:
            return from_pandas(value, origin_type)
        return value

    def clear(self) -> None:
        """Clear all origin records."""
        self._origins.clear()

    def to_dict(self) -> Dict[str, str]:
        """Serialize origins to dict (for pickling checkpoints)."""
        return {name: t.__name__ for name, t in self._origins.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'CuDFOriginTracker':
        """Deserialize origins from dict."""
        tracker = cls()
        cudf = get_cudf()
        if cudf is not None:
            type_map = {
                'DataFrame': cudf.DataFrame,
                'Series': cudf.Series,
                'Index': cudf.Index,
            }
            for name, type_name in d.items():
                if type_name in type_map:
                    tracker._origins[name] = type_map[type_name]
        return tracker


# =============================================================================
# Conversion Functions
# =============================================================================

def to_pandas_cached(obj: Any) -> Any:
    """
    Convert cuDF object to pandas, using cache to avoid repeated copies.

    If obj is not a cuDF object, returns it unchanged.
    """
    if not is_cudf_object(obj):
        return obj
    return _checkpoint_cache.get_or_convert(obj)


def to_pandas(obj: Any) -> Any:
    """
    Convert cuDF object to pandas (uncached, always copies).

    If obj is not a cuDF object, returns it unchanged.
    Handles both native cudf objects and cudf.pandas proxy objects.
    """
    if not is_cudf_object(obj):
        return obj

    # For cudf.pandas proxy objects, get the underlying pandas object
    if is_cudf_proxy(obj):
        # Get the slow (pandas) object from the proxy
        if hasattr(obj, '_fsproxy_slow'):
            slow_obj = obj._fsproxy_slow
            if callable(slow_obj):
                slow_obj = slow_obj()
            # Only use slow_obj if it's not None
            if slow_obj is not None:
                # Return a copy to ensure independence
                if hasattr(slow_obj, 'copy'):
                    return slow_obj.copy()
                return slow_obj
        # Fallback: proxy might have to_pandas
        if hasattr(obj, 'to_pandas'):
            return obj.to_pandas()
        return obj

    # Native cudf object - use to_pandas()
    return obj.to_pandas()


def from_pandas(obj: Any, target_type: type) -> Any:
    """
    Convert pandas object back to cuDF.

    Args:
        obj: A pandas DataFrame, Series, or Index
        target_type: The cuDF type to convert to

    Returns:
        cuDF object, or original object if cuDF not available
    """
    cudf = get_cudf()
    if cudf is None:
        return obj

    type_name = target_type.__name__
    if type_name == 'DataFrame':
        return cudf.DataFrame.from_pandas(obj)
    elif type_name == 'Series':
        return cudf.Series.from_pandas(obj)
    elif type_name == 'Index':
        return cudf.Index.from_pandas(obj)
    return obj


# =============================================================================
# Deepcopy Support
# =============================================================================

import os

_DISABLE_DEEPCOPY_CACHE = os.environ.get('FLOWBOOK_DISABLE_CUDF_DEEPCOPY_CACHE', '0') == '1'


def _shallow_copy_for_checkpoint(df: Any) -> Any:
    """
    Create a shallow copy suitable for checkpointing.

    This is O(1) and safe because:
    1. pandas CoW: data buffers are copy-on-write, mutations create new arrays
    2. pandas Index is IMMUTABLE: "Index does not support mutable operations"
    3. cudf object columns: contain only immutable types (strings, tuples)
       per RAPIDS docs: "cuDF does not support the arbitrary object dtype"
    4. Cached copy is read-only: only used for diff/comparison, never modified

    Note: We use type name checks instead of isinstance because in cudf.pandas mode,
    `pd.DataFrame` is the proxy class, but the cached object is a real pandas DataFrame.
    """
    type_name = type(df).__name__

    if type_name in ('DataFrame', 'Series'):
        return df.copy(deep=False)
    elif 'Index' in type_name:
        return df.copy()
    else:
        return df


def deepcopy_cudf(obj: Any, memo: Dict[int, Any]) -> Any:
    """
    Deep copy a cuDF object by converting to pandas (CPU memory).

    Uses buffer-based deduplication to ensure that cudf objects sharing GPU
    memory also share numpy arrays in the checkpoint. This dramatically reduces
    checkpoint size when notebooks have views/subsets (e.g., train, X_train, X_valid).

    Memory sharing is preserved via numpy views and pandas CoW:
    - Arrays from the same GPU buffer share memory via views
    - CoW protects against mutations creating unexpected changes
    - The resulting pandas DataFrames are safe to use independently

    Args:
        obj: cuDF DataFrame, Series, or Index
        memo: deepcopy memo dict

    Returns:
        pandas equivalent (stored in CPU memory)
    """
    obj_id = id(obj)

    # Level 0: Check memo (same checkpoint, same object)
    if obj_id in memo:
        return memo[obj_id]

    cache = get_checkpoint_cache()

    # Convert GPU→pandas with buffer-based deduplication
    # This may return a DataFrame whose arrays are views into cached arrays
    pandas_copy = to_pandas_cached(obj)

    # Use shallow copy to preserve buffer-sharing views
    # This is safe because:
    # 1. pandas CoW: data buffers are copy-on-write, mutations create new arrays
    # 2. pandas Index is immutable
    # 3. cudf object columns contain only immutable types (strings, tuples)
    # 4. Checkpoint values are not modified after creation
    result = _shallow_copy_for_checkpoint(pandas_copy)

    # Register underlying arrays in memo to handle deepcopy traversal
    # This ensures HeapSizer sees the shared arrays
    if hasattr(result, '_data') or hasattr(result, '_mgr'):
        try:
            # Get the underlying block arrays and add to memo
            if hasattr(result, '_mgr') and hasattr(result._mgr, 'arrays'):
                for arr in result._mgr.arrays:
                    if hasattr(arr, '_ndarray'):
                        arr_id = id(arr._ndarray)
                        if arr_id not in memo:
                            memo[arr_id] = arr._ndarray
        except Exception:
            pass  # Best effort - don't fail checkpoint on memo issues

    memo[obj_id] = result
    return result


# =============================================================================
# Diff Support
# =============================================================================

def _is_proxy_dataframe(obj: Any) -> bool:
    """Check if object is a cudf.pandas proxy wrapping a DataFrame."""
    if not is_cudf_proxy(obj):
        return False
    # Check the type name - proxies report as pandas types
    type_name = type(obj).__name__
    return type_name == 'DataFrame'


def _is_proxy_series(obj: Any) -> bool:
    """Check if object is a cudf.pandas proxy wrapping a Series."""
    if not is_cudf_proxy(obj):
        return False
    type_name = type(obj).__name__
    return type_name == 'Series'


def _is_proxy_index(obj: Any) -> bool:
    """Check if object is a cudf.pandas proxy wrapping an Index."""
    if not is_cudf_proxy(obj):
        return False
    type_name = type(obj).__name__
    # Index has various subclasses
    return 'Index' in type_name


def are_both_cudf_same_type(obj1: Any, obj2: Any) -> bool:
    """Check if both objects are cuDF objects of the same type.

    Handles both native cudf objects and cudf.pandas proxy objects.
    """
    if not has_cudf():
        return False

    # Check native cudf types
    if is_cudf_dataframe(obj1) and is_cudf_dataframe(obj2):
        return True
    if is_cudf_series(obj1) and is_cudf_series(obj2):
        return True
    if is_cudf_index(obj1) and is_cudf_index(obj2):
        return True

    # Check cudf.pandas proxy types
    if _is_proxy_dataframe(obj1) and _is_proxy_dataframe(obj2):
        return True
    if _is_proxy_series(obj1) and _is_proxy_series(obj2):
        return True
    if _is_proxy_index(obj1) and _is_proxy_index(obj2):
        return True

    return False


def diff_cudf(obj1: Any, obj2: Any, path: str, differ: Any) -> Optional[Any]:
    """
    Compare two cuDF objects by converting to pandas.

    Args:
        obj1: First cudf object (native cudf or cudf.pandas proxy)
        obj2: Second cudf object (native cudf or cudf.pandas proxy)
        path: Variable path for error messages
        differ: The Diff instance (has _compare_dataframe etc. methods)

    Returns None if equal, or a DiffNode if different.
    """
    # Convert both to pandas (uncached - we want fresh copies for comparison)
    with timer(key="cudf:diff", message="Convert cuDF to pandas"):
        pdf1 = to_pandas(obj1)
        pdf2 = to_pandas(obj2)

    # Use the Diff instance's comparison methods
    # Check both native cudf types and cudf.pandas proxy types
    if is_cudf_dataframe(obj1) or _is_proxy_dataframe(obj1):
        return differ._compare_dataframe(pdf1, pdf2, path)
    elif is_cudf_series(obj1) or _is_proxy_series(obj1):
        return differ._compare_series(pdf1, pdf2, path)
    elif is_cudf_index(obj1) or _is_proxy_index(obj1):
        return differ._compare_index(pdf1, pdf2, path)

    return None
