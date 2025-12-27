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

from data_ferret.util.output import timer

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
    """

    def __init__(self):
        # Maps id(cudf_obj) -> (fingerprint, pandas_copy, weak_ref)
        self._cache: Dict[int, tuple] = {}

    def _fingerprint(self, obj: Any) -> Optional[tuple]:
        """
        Compute a cheap fingerprint of a cuDF object.

        This runs on the GPU and is much faster than a full GPU→CPU copy.
        The fingerprint includes shape, dtypes, and a hash of the data.
        """
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

        return None

    def get_cached(self, obj: Any) -> Optional[Any]:
        """
        Get cached pandas copy if object hasn't changed.

        Returns None if not in cache or fingerprint changed.
        """
        obj_id = id(obj)

        if obj_id not in self._cache:
            return None

        cached_fp, pandas_copy, weak_ref = self._cache[obj_id]

        # Verify weak reference still points to same object
        ref_obj = weak_ref()
        if ref_obj is None or ref_obj is not obj:
            del self._cache[obj_id]
            return None

        # Verify fingerprint matches
        current_fp = self._fingerprint(obj)
        if current_fp != cached_fp:
            del self._cache[obj_id]
            return None

        return pandas_copy

    def cache(self, obj: Any, pandas_copy: Any) -> None:
        """Store a pandas copy in the cache."""
        obj_id = id(obj)
        fp = self._fingerprint(obj)

        try:
            weak_ref = weakref.ref(obj)
        except TypeError:
            return

        self._cache[obj_id] = (fp, pandas_copy, weak_ref)

    def get_or_convert(self, obj: Any) -> Any:
        """Get cached pandas copy or convert from GPU."""
        cached = self.get_cached(obj)
        if cached is not None:
            return cached

        # Handle cudf.pandas proxy objects
        if is_cudf_proxy(obj):
            if hasattr(obj, '_fsproxy_slow'):
                slow_obj = obj._fsproxy_slow
                if callable(slow_obj):
                    slow_obj = slow_obj()
                pandas_copy = slow_obj.copy() if hasattr(slow_obj, 'copy') else slow_obj
            elif hasattr(obj, 'to_pandas'):
                pandas_copy = obj.to_pandas()
            else:
                pandas_copy = obj
        else:
            # Native cudf object
            pandas_copy = obj.to_pandas()

        self.cache(obj, pandas_copy)
        return pandas_copy

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


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

def deepcopy_cudf(obj: Any, memo: Dict[int, Any]) -> Any:
    """
    Deep copy a cuDF object by converting to pandas (CPU memory).

    Args:
        obj: cuDF DataFrame, Series, or Index
        memo: deepcopy memo dict

    Returns:
        pandas equivalent (stored in CPU memory)
    """
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    # Convert to pandas using cache
    pandas_copy = to_pandas_cached(obj)

    # The pandas copy may need its own deep copy (e.g., object columns)
    from .deepcopy import deepcopy as ferret_deepcopy
    result = ferret_deepcopy(pandas_copy, memo)

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
    with timer(key="diff_cudf", message="Convert cuDF to pandas"):
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
