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
    - Handles pandas CoW sharing correctly via np.shares_memory()
    - Safe for read-only buffers (never reads contents)

    Usage:
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        ns_size = sizer.sizeof_namespace({'a': arr, 'b': df})
    """

    def __init__(self):
        self._seen_ids: Set[int] = set()
        self._seen_data_ptrs: Set[int] = set()  # numpy data buffer pointers
        self._seen_arrays: List[Any] = []  # numpy arrays for shares_memory check
        self._ref_counts: Dict[int, int] = {}  # Track reference counts for shared detection

    def reset(self):
        """Clear seen tracking for new measurement."""
        self._seen_ids.clear()
        self._seen_data_ptrs.clear()
        self._seen_arrays.clear()
        self._ref_counts.clear()

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

        # First pass: count references to detect sharing
        for var_name in vars_to_measure:
            obj = ns[var_name]
            self._count_refs(obj)

        # Reset seen for actual measurement
        seen_before = len(self._seen_ids)
        self._seen_ids.clear()
        self._seen_data_ptrs.clear()

        # Second pass: measure each variable
        for var_name in vars_to_measure:
            obj = ns[var_name]
            size = self._sizeof(obj, owned_only=True)
            by_variable[var_name] = size
            total_bytes += size

            # Track by type
            type_name = type(obj).__name__
            by_type[type_name] = by_type.get(type_name, 0) + size

        # Calculate shared bytes (objects seen multiple times)
        shared_bytes = sum(
            1 for count in self._ref_counts.values() if count > 1
        ) * 100  # Rough estimate

        return NamespaceSize(
            total_bytes=total_bytes,
            by_variable=by_variable,
            by_type=by_type,
            shared_bytes=shared_bytes,
        )

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

    def _count_refs(self, obj: Any, depth: int = 0) -> None:
        """Count references for shared object detection."""
        if depth > 100:  # Prevent infinite recursion
            return

        obj_id = id(obj)
        self._ref_counts[obj_id] = self._ref_counts.get(obj_id, 0) + 1

        if obj_id in self._seen_ids:
            return
        self._seen_ids.add(obj_id)

        # Recurse into containers
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._count_refs(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._count_refs(item, depth + 1)

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

        # Numeric arrays: deduplicate by data pointer AND shares_memory check
        # ctypes.data deduplication handles views of same array
        try:
            data_ptr = arr.ctypes.data
            if data_ptr in self._seen_data_ptrs:
                return 128  # Already counted this buffer
            self._seen_data_ptrs.add(data_ptr)
        except Exception:
            # ctypes.data can fail for some array types
            pass

        # shares_memory check handles CoW copies where ctypes.data differs
        # but underlying buffer is shared
        try:
            for seen_arr in self._seen_arrays:
                if np.shares_memory(arr, seen_arr):
                    return 128  # Already counted this buffer
            self._seen_arrays.append(arr)
        except Exception:
            pass

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
                    # ExtensionArray without numpy backing
                    total += self._sizeof_extension_array(arr, owned_only)

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
                total += self._sizeof_extension_array(values, owned_only)
        elif hasattr(series, 'values'):
            total += self._sizeof(series.values, owned_only)

        # Index
        total += self._sizeof(series.index, owned_only)

        return total

    def _sizeof_index(self, index, owned_only: bool) -> int:
        """Measure pandas Index memory."""
        pd = _get_pandas()
        if pd is None:
            return 0

        try:
            # Use pandas built-in memory_usage
            usage = index.memory_usage(deep=True)
            return int(usage)
        except Exception:
            # Fallback
            return sys.getsizeof(index)

    def _get_backing_ndarray(self, arr):
        """Extract numpy array from pandas array types."""
        np = _get_numpy()
        if np is None:
            return None

        if isinstance(arr, np.ndarray):
            return arr
        if hasattr(arr, '_ndarray'):  # DatetimeArray, TimedeltaArray
            return arr._ndarray
        if hasattr(arr, '_data') and hasattr(arr._data, '_ndarray'):
            return arr._data._ndarray
        if hasattr(arr, '_data') and isinstance(arr._data, np.ndarray):
            return arr._data
        return None

    def _sizeof_extension_array(self, arr, owned_only: bool) -> int:
        """Measure pandas ExtensionArray memory."""
        try:
            # Try memory_usage if available
            if hasattr(arr, 'memory_usage'):
                return int(arr.memory_usage())
            # Try nbytes
            if hasattr(arr, 'nbytes'):
                return int(arr.nbytes) + 64
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
        for k, v in d.items():
            total += self._sizeof(k, owned_only)
            total += self._sizeof(v, owned_only)
        return total

    def _sizeof_set(self, s, owned_only: bool) -> int:
        """Measure set/frozenset memory including elements."""
        total = sys.getsizeof(s)
        for item in s:
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

        # __dict__
        if hasattr(obj, '__dict__'):
            obj_dict = obj.__dict__
            if isinstance(obj_dict, dict):
                total += self._sizeof_dict(obj_dict, owned_only)

        # __slots__
        if hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                try:
                    val = getattr(obj, slot, None)
                    if val is not None:
                        total += self._sizeof(val, owned_only)
                except Exception:
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
