"""
Ownership-aware heap traversal for accurate memory measurement.

This module provides the HeapSizer class for measuring memory usage of Python
objects with proper handling of:
- NumPy array views and shared data buffers
- Pandas DataFrame/Series Copy-on-Write sharing
- Object deduplication across shared references
- ML model opaque patterns (Keras, PyTorch, CatBoost)

Unlike pympler, this implementation:
- Never reads buffer contents (safe for read-only arrays)
- Uses metadata only (nbytes, base, ctypes.data, dtype)
- Properly handles ownership for accurate attribution
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Set, Optional, List
import sys
import types

# Lazy imports to avoid circular dependencies
_np = None
_pd = None


def _get_numpy():
    """Lazy import numpy."""
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError:
            pass
    return _np


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        try:
            import pandas as pd
            _pd = pd
        except ImportError:
            pass
    return _pd


@dataclass
class NamespaceSize:
    """Result of measuring a namespace's memory."""
    total_bytes: int
    by_variable: Dict[str, int]
    by_type: Dict[str, int]
    shared_bytes: int  # Memory counted once but referenced by multiple vars


@dataclass
class CheckpointSize:
    """Result of measuring a checkpoint's memory."""
    total_bytes: int
    by_variable: Dict[str, int]
    by_type: Dict[str, int]
    new_bytes: int  # Memory not shared with live namespace


@dataclass
class AllCheckpointsSize:
    """Result of measuring all checkpoints together, accounting for sharing."""
    total_bytes: int  # Total memory used by all checkpoints (deduplicated)
    by_variable: Dict[str, int]  # Last variable assignment wins for attribution
    by_type: Dict[str, int]  # Aggregated by type
    by_checkpoint: Dict[str, int]  # Per-checkpoint contribution (after dedup)


@dataclass
class CheckpointOverhead:
    """Result of measuring checkpoint memory beyond namespace."""
    total_mb: float  # Total checkpoint memory beyond namespace
    by_checkpoint: Dict[str, float]  # Delta per checkpoint (in MB)
    by_variable: Dict[str, float]  # Per-variable totals (in MB)
    cumulative: Dict[str, float]  # Running total at each checkpoint (in MB)
    by_checkpoint_by_var: Dict[str, Dict[str, float]]  # {checkpoint: {var: mb}}


# Types that are atomic (no internal references to measure)
_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    range,
    type,
    types.CodeType,
    types.BuiltinFunctionType,
    property,
)


class HeapSizer:
    """
    Ownership-aware heap traversal for accurate memory measurement.

    Key features:
    - Tracks object identity to avoid double-counting
    - Tracks numpy data buffer pointers for view deduplication
    - Handles pandas CoW sharing via ctypes.data pointer deduplication
    - Safe for read-only buffers (never reads contents)

    Usage:
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        ns_size = sizer.sizeof_namespace({'a': arr, 'b': df})
    """

    def __init__(self):
        self._seen_ids: Set[int] = set()
        self._seen_data_ptrs: Set[int] = set()  # numpy data buffer pointers

    def reset(self):
        """Clear seen tracking for new measurement."""
        self._seen_ids.clear()
        self._seen_data_ptrs.clear()

    def sizeof(self, obj: Any, owned_only: bool = True) -> int:
        """
        Get deep size of object in bytes.

        Note: This method accumulates seen IDs across calls. Use reset() to
        clear tracking state, or use the module-level sizeof() function for
        independent measurements.

        Args:
            obj: Object to measure
            owned_only: If True, skip numpy views and shared pandas blocks

        Returns:
            Size in bytes
        """
        return self._sizeof(obj, owned_only)

    def sizeof_namespace(
        self,
        ns: Dict[str, Any],
        *,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> NamespaceSize:
        """
        Measure memory of namespace variables.

        Args:
            ns: Namespace dictionary
            include: If provided, only measure these variable names
            exclude: If provided, skip these variable names

        Returns:
            NamespaceSize with total, per-variable, per-type, and shared bytes
        """
        self.reset()

        by_variable: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        total_bytes = 0

        # Filter variables
        vars_to_measure = set(ns.keys())
        if include is not None:
            vars_to_measure &= include
        if exclude is not None:
            vars_to_measure -= exclude

        # NOTE: Removed the first pass (_count_refs) that was only used for a rough
        # shared_bytes estimate. This eliminates unnecessary traversal overhead.

        # Measure each variable with deduplication via _seen_ids and _seen_data_ptrs
        # Use owned_only=False to count actual memory, even for arrays with .base
        # (e.g., pandas CoW internal arrays).
        for var_name in vars_to_measure:
            obj = ns[var_name]
            size = self._sizeof(obj, owned_only=False)
            by_variable[var_name] = size
            total_bytes += size

            # Track by type
            type_name = type(obj).__name__
            by_type[type_name] = by_type.get(type_name, 0) + size

        return NamespaceSize(
            total_bytes=total_bytes,
            by_variable=by_variable,
            by_type=by_type,
            shared_bytes=0,  # No longer computed (was just a rough estimate)
        )

    def sizeof_user_namespace(self, globals_dict: Dict[str, Any]) -> NamespaceSize:
        """
        Filter and measure user namespace from globals().

        Filters out:
        - Private variables (starting with '_')
        - IPython/ipykernel/zmq internal objects (by module check)
        - In/Out lists, get_ipython
        - Modules
        - Functions (regular and builtin)
        - Types/classes

        This method handles the filtering internally to avoid requiring
        __import__('types') in user_expressions, which can trigger dbm
        imports on some systems.

        Args:
            globals_dict: The globals() dictionary from the kernel

        Returns:
            NamespaceSize with total, per-variable, per-type, and shared bytes
        """
        # Filter out system/internal variables
        # Note: IPython/ipykernel objects (like ZMQExitAutocall for quit/exit) can have
        # huge __dict__ containing references to the entire kernel. Filter by module.
        def _is_ipython_internal(v):
            """Check if value is from IPython/ipykernel internals."""
            try:
                mod = getattr(type(v), '__module__', '') or ''
                return mod.startswith(('IPython', 'ipykernel', 'zmq'))
            except Exception:
                return False

        filtered = {}
        for k, v in globals_dict.items():
            if k.startswith('_'):
                continue
            if k in ('In', 'Out', 'get_ipython'):
                continue
            if isinstance(v, types.ModuleType):
                continue
            if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, type)):
                continue
            if _is_ipython_internal(v):
                continue
            filtered[k] = v

        return self.sizeof_namespace(filtered)

    def sizeof_checkpoint(
        self,
        checkpoint,
        exclude_cached: bool = True
    ) -> CheckpointSize:
        """
        Measure memory contribution of a checkpoint.

        Cached objects from deepcopy are excluded by default because they
        are shared across multiple checkpoints and should be counted
        separately as cache overhead rather than checkpoint memory.

        Args:
            checkpoint: MemoryCheckpoint object
            exclude_cached: If True, exclude objects in deepcopy caches

        Returns:
            CheckpointSize with total, per-variable, per-type, and new_bytes
        """
        self.reset()

        if not hasattr(checkpoint, 'user_ns'):
            return CheckpointSize(0, {}, {}, 0)

        # Pre-populate seen_ids with cached objects so they're skipped
        if exclude_cached:
            try:
                from flowbook.kernel_support.deepcopy import get_cached_object_ids
                cached_ids = get_cached_object_ids()
                self._seen_ids.update(cached_ids)
            except ImportError:
                pass

        user_ns = checkpoint.user_ns
        by_variable: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        total_bytes = 0

        for var_name, obj in user_ns.items():
            size = self._sizeof(obj, owned_only=True)
            by_variable[var_name] = size
            total_bytes += size

            type_name = type(obj).__name__
            by_type[type_name] = by_type.get(type_name, 0) + size

        # new_bytes = total for now (would need live namespace to compare)
        return CheckpointSize(
            total_bytes=total_bytes,
            by_variable=by_variable,
            by_type=by_type,
            new_bytes=total_bytes,
        )

    def sizeof_all_checkpoints(
        self,
        checkpoints: Dict[str, Any],
        exclude_cached: bool = True
    ) -> AllCheckpointsSize:
        """
        Measure memory of ALL checkpoints together, accounting for sharing.

        This is the correct way to measure cumulative checkpoint memory because
        checkpoints often share objects (via the deepcopy memo dict). Measuring
        each checkpoint separately and summing would overcount shared memory.

        Args:
            checkpoints: Dict mapping checkpoint names to MemoryCheckpoint objects
            exclude_cached: If True, exclude objects in deepcopy caches

        Returns:
            AllCheckpointsSize with deduplicated total and breakdowns
        """
        self.reset()

        # Pre-populate seen_ids with cached objects so they're skipped
        if exclude_cached:
            try:
                from flowbook.kernel_support.deepcopy import get_cached_object_ids
                cached_ids = get_cached_object_ids()
                self._seen_ids.update(cached_ids)
            except ImportError:
                pass

        by_variable: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_checkpoint: Dict[str, int] = {}
        total_bytes = 0

        # Measure all checkpoints in a single pass WITHOUT resetting between them.
        # Objects seen in earlier checkpoints won't be counted again in later ones.
        for ckpt_name, ckpt in checkpoints.items():
            if not hasattr(ckpt, 'user_ns'):
                by_checkpoint[ckpt_name] = 0
                continue

            ckpt_bytes = 0
            for var_name, obj in ckpt.user_ns.items():
                # Use owned_only=False to follow views to their base arrays
                # and deduplicate by data pointer. This correctly handles:
                # - Pandas CoW: checkpoints share underlying data with originals
                # - Modified data: CoW creates new arrays that ARE counted
                # - Same variable across checkpoints: deduplicated if sharing
                size = self._sizeof(obj, owned_only=False)
                ckpt_bytes += size

                # Track by variable (size added only if not deduplicated)
                by_variable[var_name] = by_variable.get(var_name, 0) + size

                # Track by type
                type_name = type(obj).__name__
                by_type[type_name] = by_type.get(type_name, 0) + size

            by_checkpoint[ckpt_name] = ckpt_bytes
            total_bytes += ckpt_bytes

        return AllCheckpointsSize(
            total_bytes=total_bytes,
            by_variable=by_variable,
            by_type=by_type,
            by_checkpoint=by_checkpoint,
        )

    def sizeof_checkpoints_beyond_namespace(
        self,
        namespace: Dict[str, Any],
        checkpoints: List[tuple],
    ) -> CheckpointOverhead:
        """
        Measure checkpoint memory BEYOND what's in the namespace.

        Uses cumulative seen_ids to deduplicate:
        1. Objects in namespace (measured first, marked as seen)
        2. Objects shared between checkpoints (only counted once)

        This correctly handles Copy-on-Write sharing between checkpoints
        and the namespace - shared memory is not double-counted.

        Args:
            namespace: Current user namespace dict (filtered to exclude private/callable)
            checkpoints: List of (checkpoint_name, MemoryCheckpoint) in execution order

        Returns:
            CheckpointOverhead with:
            - total_mb: Total bytes in checkpoints not in namespace
            - by_checkpoint: Dict of checkpoint_name -> delta MB
            - by_variable: Dict of var_name -> total MB across all checkpoints
            - cumulative: Dict of checkpoint_name -> running total MB
        """
        self.reset()

        # Step 1: Measure namespace, mark all objects as seen
        # This populates seen_ids with all namespace object IDs
        self.sizeof_namespace(namespace)

        # Step 2: Measure each checkpoint cumulatively (don't reset between)
        # Objects already seen (from namespace or prior checkpoints) return 0
        by_checkpoint: Dict[str, float] = {}
        by_variable: Dict[str, float] = {}
        by_checkpoint_by_var: Dict[str, Dict[str, float]] = {}
        cumulative: Dict[str, float] = {}
        running_total_bytes = 0

        for ckpt_name, ckpt in checkpoints:
            if not hasattr(ckpt, 'user_ns'):
                by_checkpoint[ckpt_name] = 0.0
                by_checkpoint_by_var[ckpt_name] = {}
                cumulative[ckpt_name] = running_total_bytes / (1024 * 1024)
                continue

            ckpt_delta_bytes = 0
            ckpt_vars: Dict[str, float] = {}
            for var_name, obj in ckpt.user_ns.items():
                # owned_only=True: don't follow views to base arrays owned elsewhere
                size = self._sizeof(obj, owned_only=True)
                ckpt_delta_bytes += size
                if size > 0:
                    var_mb = size / (1024 * 1024)
                    by_variable[var_name] = by_variable.get(var_name, 0.0) + var_mb
                    ckpt_vars[var_name] = var_mb

            delta_mb = ckpt_delta_bytes / (1024 * 1024)
            by_checkpoint[ckpt_name] = delta_mb
            by_checkpoint_by_var[ckpt_name] = ckpt_vars
            running_total_bytes += ckpt_delta_bytes
            cumulative[ckpt_name] = running_total_bytes / (1024 * 1024)

        return CheckpointOverhead(
            total_mb=running_total_bytes / (1024 * 1024),
            by_checkpoint=by_checkpoint,
            by_variable=by_variable,
            cumulative=cumulative,
            by_checkpoint_by_var=by_checkpoint_by_var,
        )

    def _sizeof(self, obj: Any, owned_only: bool) -> int:
        """
        Internal recursive size calculation.

        Dispatches to type-specific handlers.
        """
        # Check for None first
        if obj is None:
            return 0

        # Check if already seen
        obj_id = id(obj)
        if obj_id in self._seen_ids:
            return 0
        self._seen_ids.add(obj_id)

        # Atomic types
        if isinstance(obj, _ATOMIC_TYPES):
            return sys.getsizeof(obj)

        # cuDF proxy objects: convert to pandas via _fsproxy_slow to avoid
        # triggering _fsproxy_fast access, which can fail with
        # NotImplementedError for certain column types (e.g., category dtype
        # after factorize()).  Measure the pandas representation instead.
        from flowbook.kernel_support import cudf_compat
        if cudf_compat.is_cudf_object(obj):
            pandas_obj = cudf_compat.to_pandas(obj)
            return self._sizeof(pandas_obj, owned_only)

        # NumPy array
        np = _get_numpy()
        if np is not None and isinstance(obj, np.ndarray):
            return self._sizeof_ndarray(obj, owned_only)

        # Pandas types
        pd = _get_pandas()
        if pd is not None:
            if isinstance(obj, pd.DataFrame):
                return self._sizeof_dataframe(obj, owned_only)
            if isinstance(obj, pd.Series):
                return self._sizeof_series(obj, owned_only)
            if isinstance(obj, pd.Index):
                return self._sizeof_index(obj, owned_only)
            # ExtensionArray (ArrowExtensionArray, StringArray, etc.)
            # Check via api.extensions to handle all EA subclasses
            if hasattr(pd.api, 'extensions') and hasattr(pd.api.extensions, 'ExtensionArray'):
                if isinstance(obj, pd.api.extensions.ExtensionArray):
                    return self._sizeof_extension_array(obj, owned_only)

        # Containers
        if isinstance(obj, dict):
            return self._sizeof_dict(obj, owned_only)
        if isinstance(obj, list):
            return self._sizeof_list(obj, owned_only)
        if isinstance(obj, tuple):
            return self._sizeof_tuple(obj, owned_only)
        if isinstance(obj, (set, frozenset)):
            return self._sizeof_set(obj, owned_only)

        # Functions
        if isinstance(obj, types.FunctionType):
            return self._sizeof_function(obj, owned_only)

        # Check for ML models (lazy)
        if self._is_keras_model(obj):
            return self._sizeof_keras_model(obj, owned_only)
        if self._is_pytorch_model(obj):
            return self._sizeof_pytorch_model(obj, owned_only)
        if self._is_catboost_pool(obj):
            return self._sizeof_catboost_pool(obj, owned_only)

        # Matplotlib objects - opaque, don't traverse internals
        if self._is_matplotlib_object(obj):
            return self._sizeof_matplotlib(obj)

        # Generic object with __dict__
        return self._sizeof_object(obj, owned_only)

    def _sizeof_ndarray(self, arr, owned_only: bool) -> int:
        """
        Safe measurement of numpy array memory.

        Never reads buffer contents - uses metadata only:
        - arr.nbytes: computed from shape * itemsize (no buffer read)
        - arr.base: reference check (no buffer read)
        - arr.ctypes.data: pointer value as int (no buffer read)

        This avoids BufferError on read-only/mapped arrays.

        Object arrays (dtype=object) require element traversal since
        nbytes only counts the pointer array, not referenced objects.
        """
        np = _get_numpy()
        if np is None:
            return 0

        # For views, behavior depends on owned_only:
        # - owned_only=True: view doesn't own data, return just wrapper
        # - owned_only=False: measure the base array (full buffer)
        if arr.base is not None:
            if owned_only:
                return 128  # Just wrapper overhead
            else:
                # Follow to the root owner
                base = arr.base
                while hasattr(base, 'base') and base.base is not None:
                    base = base.base
                if isinstance(base, np.ndarray):
                    return 128 + self._sizeof_ndarray(base, owned_only=False)
                # Non-numpy base (e.g., memoryview)
                return 128 + getattr(base, 'nbytes', sys.getsizeof(base))

        # Object arrays: must traverse elements
        if arr.dtype == np.object_:
            total = 128 + arr.size * 8  # Wrapper + pointer array
            for obj in arr.flat:
                if obj is not None:
                    # Let _sizeof handle seen_ids checking
                    total += self._sizeof(obj, owned_only)
            return total

        # Numeric arrays: deduplicate by data pointer
        # ctypes.data deduplication handles views of same array and CoW sharing
        try:
            data_ptr = arr.ctypes.data
            if data_ptr in self._seen_data_ptrs:
                return 128  # Already counted this buffer
            self._seen_data_ptrs.add(data_ptr)
        except Exception:
            # ctypes.data can fail for some array types
            pass

        # NOTE: Removed np.shares_memory() check - it was O(n²) and caused
        # massive slowdowns (4+ minutes) for large namespaces with many arrays.
        # The ctypes.data check above handles 99.99% of cases. The shares_memory
        # check only caught rare edge cases (e.g., np.frombuffer with overlapping
        # regions). Accepting minor overcounting in those cases for performance.

        return arr.nbytes + 128

    def _sizeof_dataframe(self, df, owned_only: bool) -> int:
        """Measure DataFrame memory, handling CoW sharing."""
        total = 200  # DataFrame wrapper overhead

        # Measure backing arrays via BlockManager
        if hasattr(df, '_mgr') and hasattr(df._mgr, 'arrays'):
            for arr in df._mgr.arrays:
                nd = self._get_backing_ndarray(arr)
                if nd is not None:
                    # Use _sizeof() to get proper object identity tracking via _seen_ids.
                    # This ensures that shared ndarray objects across checkpoints
                    # (e.g., pandas CoW) are only counted once.
                    total += self._sizeof(nd, owned_only)
                else:
                    # ExtensionArray without numpy backing (e.g., ArrowExtensionArray
                    # for StringDtype). Route through _sizeof() for identity tracking
                    # to avoid double-counting shared arrays across checkpoints.
                    total += self._sizeof(arr, owned_only)

        # Index and columns
        total += self._sizeof(df.index, owned_only)
        total += self._sizeof(df.columns, owned_only)

        return total

    def _sizeof_series(self, series, owned_only: bool) -> int:
        """Measure Series memory."""
        total = 100  # Series wrapper overhead

        # Get backing array
        if hasattr(series, '_values'):
            values = series._values
            nd = self._get_backing_ndarray(values)
            if nd is not None:
                # Use _sizeof() to get proper object identity tracking via _seen_ids.
                # This ensures that shared ndarray objects across checkpoints
                # are only counted once.
                total += self._sizeof(nd, owned_only)
            else:
                # ExtensionArray without numpy backing - route through _sizeof()
                # for identity tracking to avoid double-counting.
                total += self._sizeof(values, owned_only)
        elif hasattr(series, 'values'):
            total += self._sizeof(series.values, owned_only)

        # Index
        total += self._sizeof(series.index, owned_only)

        return total

    def _sizeof_index(self, index, owned_only: bool) -> int:
        """
        Measure pandas Index memory with proper deduplication.

        Extracts the underlying numpy array and measures it via _sizeof_ndarray,
        which handles memory deduplication for shared arrays (e.g., from deepcopy).
        This prevents double-counting Index memory across checkpoints when the
        underlying data is shared.
        """
        pd = _get_pandas()
        np = _get_numpy()
        if pd is None:
            return 0

        total = 128  # Index wrapper overhead

        try:
            # RangeIndex is special - it doesn't store data, just start/stop/step
            # Accessing _data would materialize a full array, which we don't want
            if isinstance(index, pd.RangeIndex):
                # RangeIndex only stores start, stop, step (3 ints = 24 bytes)
                # Plus some object overhead
                return total + 100

            # Extract underlying numpy array for proper deduplication
            # Most Index types have a _data attribute containing the values
            backing_array = None

            if hasattr(index, '_data'):
                data = index._data
                # For NumericIndex, _data is typically an ndarray
                if np is not None and isinstance(data, np.ndarray):
                    backing_array = data
                # For DatetimeIndex, _data is DatetimeArray with _ndarray
                elif hasattr(data, '_ndarray'):
                    backing_array = data._ndarray
                # Check for PyArrow-backed arrays (use _pa_array, not deprecated _data)
                elif hasattr(data, '_pa_array'):
                    # Arrow arrays - skip numpy extraction, will use fallback sizing
                    pass
                # Some types have nested _data (legacy path with warning suppression)
                elif hasattr(data, '_data'):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=FutureWarning)
                        nested = getattr(data, '_data', None)
                        if isinstance(nested, np.ndarray):
                            backing_array = nested

            # Fallback: try .values or ._values
            if backing_array is None and hasattr(index, '_values'):
                vals = index._values
                if np is not None and isinstance(vals, np.ndarray):
                    backing_array = vals

            if backing_array is None and hasattr(index, 'values'):
                vals = index.values
                if np is not None and isinstance(vals, np.ndarray):
                    backing_array = vals

            if backing_array is not None:
                # Use _sizeof which goes through _sizeof_ndarray for deduplication
                total += self._sizeof(backing_array, owned_only)
            else:
                # Fallback for complex Index types (MultiIndex, etc.)
                try:
                    usage = index.memory_usage(deep=True)
                    total += int(usage) - 128  # Subtract wrapper already counted
                except Exception:
                    total += sys.getsizeof(index) - 128

        except Exception:
            # Final fallback
            return sys.getsizeof(index)

        return total

    def _get_backing_ndarray(self, arr):
        """Extract numpy array from pandas array types."""
        import warnings

        np = _get_numpy()
        if np is None:
            return None

        if isinstance(arr, np.ndarray):
            return arr
        if hasattr(arr, '_ndarray'):  # DatetimeArray, TimedeltaArray
            return arr._ndarray

        # Check for PyArrow-backed arrays first (ArrowStringArray, etc.)
        # These use _pa_array instead of _data (which is deprecated)
        if hasattr(arr, '_pa_array'):
            # Arrow arrays don't have numpy backing - return None to use fallback sizing
            return None

        # Legacy path for older pandas array types
        # Suppress FutureWarning for deprecated _data attribute access
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            if hasattr(arr, '_data') and hasattr(arr._data, '_ndarray'):
                return arr._data._ndarray
            if hasattr(arr, '_data') and isinstance(arr._data, np.ndarray):
                return arr._data

        return None

    def _sizeof_extension_array(self, arr, owned_only: bool) -> int:
        """
        Measure pandas ExtensionArray memory with proper deduplication.

        ExtensionArrays (StringArray, ArrowExtensionArray, etc.) often share their
        underlying data with copies via CoW. We extract the backing array and
        measure IT (via _sizeof) to get proper identity tracking.
        """
        total = 64  # Wrapper overhead

        # Try to extract backing array for proper deduplication
        backing = self._get_backing_ndarray(arr)
        if backing is not None:
            # Has numpy backing - measure via _sizeof for identity tracking
            total += self._sizeof(backing, owned_only)
            return total

        # For ArrowExtensionArray, extract the PyArrow chunked array
        if hasattr(arr, '_ndarray'):
            # StringArray, IntegerArray, etc. have _ndarray
            backing_id = id(arr._ndarray)
            if backing_id in self._seen_ids:
                return total  # Already counted
            self._seen_ids.add(backing_id)
            try:
                if hasattr(arr, 'memory_usage'):
                    return total + int(arr.memory_usage())
                if hasattr(arr, 'nbytes'):
                    return total + int(arr.nbytes)
            except Exception:
                pass

        # Fallback: use memory_usage or nbytes directly
        try:
            if hasattr(arr, 'memory_usage'):
                return total + int(arr.memory_usage())
            if hasattr(arr, 'nbytes'):
                return total + int(arr.nbytes)
        except Exception:
            pass
        return sys.getsizeof(arr)

    def _sizeof_list(self, lst: list, owned_only: bool) -> int:
        """Measure list memory including elements."""
        total = sys.getsizeof(lst)  # List overhead + pointer array
        for item in lst:
            total += self._sizeof(item, owned_only)
        return total

    def _sizeof_tuple(self, tup: tuple, owned_only: bool) -> int:
        """Measure tuple memory including elements."""
        total = sys.getsizeof(tup)
        for item in tup:
            total += self._sizeof(item, owned_only)
        return total

    def _sizeof_dict(self, d: dict, owned_only: bool) -> int:
        """Measure dict memory including keys and values."""
        total = sys.getsizeof(d)
        # Use list() snapshot to avoid "dictionary changed size during iteration"
        # which can happen if traversal triggers side effects or in concurrent contexts
        for k, v in list(d.items()):
            total += self._sizeof(k, owned_only)
            total += self._sizeof(v, owned_only)
        return total

    def _sizeof_set(self, s, owned_only: bool) -> int:
        """Measure set/frozenset memory including elements."""
        total = sys.getsizeof(s)
        # Use list() snapshot to avoid "set changed size during iteration"
        for item in list(s):
            total += self._sizeof(item, owned_only)
        return total

    def _sizeof_function(self, func, owned_only: bool) -> int:
        """Measure function including closure and defaults."""
        total = sys.getsizeof(func)

        # Closure cells
        if func.__closure__:
            for cell in func.__closure__:
                total += sys.getsizeof(cell)
                try:
                    total += self._sizeof(cell.cell_contents, owned_only)
                except ValueError:
                    # Empty cell
                    pass

        # Default arguments
        if func.__defaults__:
            for default in func.__defaults__:
                total += self._sizeof(default, owned_only)

        if func.__kwdefaults__:
            total += self._sizeof(func.__kwdefaults__, owned_only)

        return total

    def _sizeof_object(self, obj: Any, owned_only: bool) -> int:
        """Measure generic object with __dict__ and/or __slots__."""
        total = sys.getsizeof(obj)

        # Wrap attribute access in try/except to handle lazy import proxies
        # (e.g., six.moves) that trigger module imports when accessed.
        # If the lazy import fails (e.g., _gdbm not installed), skip the object.
        try:
            # __dict__
            if hasattr(obj, '__dict__'):
                obj_dict = obj.__dict__
                if isinstance(obj_dict, dict):
                    total += self._sizeof_dict(obj_dict, owned_only)

            # __slots__
            if hasattr(obj, '__slots__') and obj.__slots__ is not None:
                slots = obj.__slots__
                # Validate __slots__ is actually iterable (some objects have sentinel values)
                if not isinstance(slots, (tuple, list, set, frozenset)):
                    try:
                        slots = tuple(slots)
                    except TypeError:
                        slots = ()  # Not iterable, skip
                for slot in slots:
                    try:
                        val = getattr(obj, slot, None)
                        if val is not None:
                            total += self._sizeof(val, owned_only)
                    except Exception:
                        pass
        except (ImportError, ModuleNotFoundError, RuntimeError):
            # Lazy import proxy (e.g., six.moves.dbm_gnu) tried to import
            # an unavailable module (e.g., _gdbm), or a library raised an error
            # during lazy initialization (e.g., grpc version mismatch). Skip.
            pass

        return total

    # ML Model type checks and handlers

    def _is_keras_model(self, obj) -> bool:
        """Check if object is a Keras model."""
        try:
            # Check for keras.Model or tensorflow.keras.Model
            type_name = type(obj).__name__
            if type_name in ('Sequential', 'Model', 'Functional'):
                module = type(obj).__module__
                if 'keras' in module:
                    return True
        except Exception:
            pass
        return False

    def _sizeof_keras_model(self, model, owned_only: bool) -> int:
        """Measure Keras model via weights only."""
        total = 1024  # Model structure overhead estimate
        try:
            for w in model.get_weights():
                # Use _sizeof() for proper object identity tracking
                total += self._sizeof(w, owned_only)
        except Exception:
            pass
        return total

    def _is_pytorch_model(self, obj) -> bool:
        """Check if object is a PyTorch model."""
        try:
            type_name = type(obj).__name__
            module = type(obj).__module__
            if 'torch' in module and hasattr(obj, 'state_dict'):
                return True
        except Exception:
            pass
        return False

    def _sizeof_pytorch_model(self, model, owned_only: bool) -> int:
        """Measure PyTorch model via state_dict."""
        total = 1024  # Model structure overhead estimate
        try:
            for name, param in model.state_dict().items():
                if hasattr(param, 'numpy'):
                    # Use _sizeof() for proper object identity tracking
                    total += self._sizeof(param.numpy(), owned_only)
                elif hasattr(param, 'nbytes'):
                    total += param.nbytes
                elif hasattr(param, 'numel') and hasattr(param, 'element_size'):
                    total += param.numel() * param.element_size()
        except Exception:
            pass
        return total

    def _is_catboost_pool(self, obj) -> bool:
        """Check if object is a CatBoost Pool."""
        try:
            type_name = type(obj).__name__
            if type_name == 'Pool':
                module = type(obj).__module__
                if 'catboost' in module:
                    return True
        except Exception:
            pass
        return False

    def _sizeof_catboost_pool(self, pool, owned_only: bool) -> int:
        """Measure CatBoost Pool memory."""
        total = 1024  # Pool overhead estimate
        try:
            # Get features if available
            if hasattr(pool, 'get_features'):
                features = pool.get_features()
                if features is not None:
                    total += self._sizeof(features, owned_only)
            # Get labels if available
            if hasattr(pool, 'get_label'):
                labels = pool.get_label()
                if labels is not None:
                    total += self._sizeof(labels, owned_only)
        except Exception:
            pass
        return total

    # Matplotlib handling

    _MATPLOTLIB_MODULES = ('matplotlib',)
    _MATPLOTLIB_TYPES = ('Figure', 'Axes', 'AxesSubplot', 'Subplot',
                         'FigureCanvas', 'FigureCanvasBase')

    def _is_matplotlib_object(self, obj) -> bool:
        """Check if object is a matplotlib Figure, Axes, or related type.

        Matplotlib objects have deeply nested internal structures (transforms,
        artists, event handlers, etc.) that can cause HeapSizer to traverse
        millions of objects and time out. Treat as opaque.
        """
        try:
            module = type(obj).__module__ or ''
            if not any(m in module for m in self._MATPLOTLIB_MODULES):
                return False
            type_name = type(obj).__name__
            # Match known types or any Axes subclass
            if type_name in self._MATPLOTLIB_TYPES:
                return True
            # Catch subclasses like AxesSubplot, Axes3D, etc.
            if 'Axes' in type_name or 'Figure' in type_name:
                return True
            # Check MRO for Artist base class (all plot elements)
            for cls in type(obj).__mro__:
                if cls.__name__ == 'Artist' and 'matplotlib' in (cls.__module__ or ''):
                    return True
        except Exception:
            pass
        return False

    def _sizeof_matplotlib(self, obj) -> int:
        """Return a flat estimate for matplotlib objects without traversal."""
        # Matplotlib objects use negligible memory compared to data arrays.
        # The actual data (numpy arrays) is tracked via the variables they came from.
        return 4096


# Convenience function
def sizeof(obj: Any, owned_only: bool = True) -> int:
    """
    Get deep size of object in bytes.

    Convenience wrapper around HeapSizer.sizeof().

    Args:
        obj: Object to measure
        owned_only: If True, skip numpy views and shared pandas blocks

    Returns:
        Size in bytes
    """
    return HeapSizer().sizeof(obj, owned_only)
