"""
Data FlowBook Kernel Namespace Diff Comparator

Compares Jupyter kernel user namespaces for equality with isomorphic pointer
structure. Supports structured diff results with detailed difference reporting.

================================================================================
OVERVIEW
================================================================================

The Diff module compares two kernel namespaces (dictionaries of variables) and
returns a structured tree of differences. It handles complex Python objects
including pandas DataFrames/Series, numpy arrays, nested containers, and
special types like Keras models and cudf objects.

Key capabilities:
- Deep structural comparison with pointer tracking (detects aliasing)
- Type-specific comparison methods for optimal performance
- Column-level DataFrame comparison for precise change detection
- LEQ mode for monotonicity checking (allows additions, blocks modifications)
- Integration with structural tracking for shape/columns/index changes

================================================================================
ARCHITECTURE
================================================================================

Type Dispatch System
--------------------
The module uses a two-tier dispatch system for O(1) type lookups:

1. _COMPARE_DISPATCH (dict): Maps exact types to method names
   - Pre-populated for common types (int, float, str, DataFrame, etc.)
   - Keras models registered lazily to avoid import overhead

2. _DISPATCH_CACHE (dict): Caches subclass lookups
   - When a type isn't in _COMPARE_DISPATCH, isinstance checks are used
   - Results are cached to avoid repeated isinstance chains

Fallback order for unknown types:
   1. Check _COMPARE_DISPATCH for exact type match
   2. Check _DISPATCH_CACHE for previously resolved subclass
   3. isinstance checks: timedelta64 → integer → floating → complex →
      GroupBy → Index → DataFrame → Series → Keras model → callable → object

Deferred Keras Import
---------------------
Keras/TensorFlow import is expensive (~3 seconds). To avoid this penalty:

1. _is_keras_model(obj) checks if obj is a Keras model WITHOUT importing Keras
   - Uses module name inspection: 'keras' in type(obj).__module__
   - Checks MRO for 'Model' or 'Sequential' base classes

2. _register_keras_dispatch_if_needed() only called when Keras model detected
   - Imports Keras and registers types in _COMPARE_DISPATCH
   - Subsequent Keras models use fast dispatch path

cudf Support
------------
cudf objects (GPU DataFrames) are handled via cudf_compat module:
- are_both_cudf_same_type() detects matching cudf types (native or proxy)
- diff_cudf() converts to pandas and delegates to standard comparison
- Supports cudf.pandas proxy mode (transparent GPU acceleration)

================================================================================
KEY FEATURES
================================================================================

LEQ Mode (use_leq=True)
-----------------------
For monotonicity checking, LEQ mode allows "conservative extensions":
- Extra keys in namespace b are allowed (variable creation OK)
- Extra columns in DataFrames are allowed (column creation OK)
- Modifications to existing values are still detected as differences

Column-Level Comparison (column_rbw parameter)
----------------------------------------------
When column_rbw is provided, only specified columns are compared:
- column_rbw = {'df': {'price', 'quantity'}} → only compare these columns
- Unspecified columns are ignored even if different
- Enables precise backward mutation detection for DataFrame operations

Structural Tracking (structural_reads, structural_mode)
-------------------------------------------------------
Detects changes to DataFrame/Series structure:
- Tracks reads of .columns, .shape, .index, len()
- WARN mode: Log warnings for structural changes
- ENFORCE mode: Treat structural changes as differences

================================================================================
PERFORMANCE OPTIMIZATIONS
================================================================================

Fast Path for DataFrames
------------------------
_fast_dataframe_equal() provides vectorized comparison:
1. Check shape, columns, index match
2. For each column, try identity check (same object)
3. For numeric columns, use array_equal
4. For object columns, check if all values are immutable strings/ints
5. Fall back to element-wise comparison only when needed

Pointer Structure Tracking
--------------------------
The differ tracks object identity to detect aliasing:
- id_map_a/id_map_b map object IDs to canonical IDs
- If same object appears multiple times, only compared once
- Detects pointer structure mismatches (a[0] is a[1] but b[0] is not b[1])

Primitive Container Cache Integration
-------------------------------------
For large lists, sets, and dicts containing only primitive types (int, float,
str, etc.), the diff leverages deepcopy's container cache to short-circuit
comparisons:
- are_primitive_containers_equal(a, b) checks if containers match via cache
- If comparing an original container to its cached checkpoint copy, O(1) check
- Avoids O(n) or O(n²) element-wise comparison for unchanged containers
- Used in _compare_list(), _compare_set(), _compare_dict()

Profiling
---------
Set FLOWBOOK_PROFILE_DIFF=1 to enable detailed timing:
- Per-variable comparison times
- Per-column breakdown for DataFrames
- Aggregated statistics by type

================================================================================
MULTIINDEX COLUMN SUPPORT
================================================================================

DataFrames with MultiIndex columns (hierarchical column labels) are fully
supported. The `_get_column_as_series()` helper function safely extracts
columns using `get_loc()` to find the column position, then `iloc` to access
it. This ensures we always get a Series, even when column names are tuples
that pandas might otherwise interpret as partial keys.

The comparison methods handle:
- 2-level and 3+ level MultiIndex columns
- Mixed-type level values (strings, integers, None)
- Duplicate column names in MultiIndex

================================================================================
USAGE
================================================================================

Basic comparison:
    >>> differ = Diff()
    >>> result = differ.diff({'x': 1, 'y': [1,2]}, {'x': 1, 'y': [1,3]})
    >>> 'y' in result.differences  # True

LEQ mode for monotonicity:
    >>> differ = Diff(use_leq=True)
    >>> result = differ.diff({'x': 1}, {'x': 1, 'new_var': 2})
    >>> result.differences  # {} - new_var allowed in LEQ mode

Column-level comparison:
    >>> differ = Diff(use_leq=True, column_rbw={'df': {'price'}})
    >>> # Only compares 'price' column, ignores other columns
"""

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas.core.groupby.ops import BaseGrouper
from typing import Any, Dict, List, Set, Tuple, Optional
import math

# Import immutable kinds for fast path in object column comparison
from flowbook.kernel_support.deepcopy import _IMMUTABLE_INFERRED_KINDS, are_primitive_containers_equal

from flowbook.kernel_support.structural_tracking import StructuralTrackingMode
from flowbook.kernel_support import cudf_compat
from flowbook.kernel_support.types import (
    ValueComparison,
    CompoundDiff,
    DiffNode,
    MemoryCheckpointDiffResult,
    IndexComponent,
    KeyComponent,
    AttributeComponent,
    DataFrameLocation,
)
from flowbook.util.output import log, output, timer
import time
import os

# Environment variable to enable detailed diff profiling
_PROFILE_DIFF = os.environ.get("FLOWBOOK_PROFILE_DIFF", "0") == "1"

# Threshold in seconds - only log comparisons taking longer than this
_PROFILE_THRESHOLD_SEC = float(os.environ.get("FLOWBOOK_PROFILE_THRESHOLD", "0.001"))


# =============================================================================
# TYPE DISPATCH OPTIMIZATION
# Using frozenset/dict lookups instead of isinstance chains for ~3x speedup
# =============================================================================

# Immutable atomic types - these don't need pointer tracking
# Note: We use exact type matching here, subclass handling is done separately
_IMMUTABLE_ATOMIC_TYPES = frozenset({
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    # numpy scalar types
    np.bool_,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
    np.complex64, np.complex128,
})

# Type category mapping for immutable atomics
# Maps exact type to category string
_TYPE_CATEGORY_MAP = {
    type(None): "none",
    bool: "bool",
    np.bool_: "bool",
    int: "integer",
    np.int8: "integer",
    np.int16: "integer",
    np.int32: "integer",
    np.int64: "integer",
    np.uint8: "integer",
    np.uint16: "integer",
    np.uint32: "integer",
    np.uint64: "integer",
    float: "float",
    np.float16: "float",
    np.float32: "float",
    np.float64: "float",
    complex: "complex",
    np.complex64: "complex",
    np.complex128: "complex",
    str: "str",
    bytes: "bytes",
}

# Dispatch table for _compare_values
# Maps exact type to method name (string, resolved at runtime)
# Order matters for subclasses - we handle that separately
_COMPARE_DISPATCH = {
    type(None): "_dispatch_none",
    bool: "_compare_bool",
    np.bool_: "_compare_bool",
    int: "_compare_int",
    np.int8: "_compare_int",
    np.int16: "_compare_int",
    np.int32: "_compare_int",
    np.int64: "_compare_int",
    np.uint8: "_compare_int",
    np.uint16: "_compare_int",
    np.uint32: "_compare_int",
    np.uint64: "_compare_int",
    float: "_compare_float",
    np.float16: "_compare_float",
    np.float32: "_compare_float",
    np.float64: "_compare_float",
    complex: "_compare_complex",
    np.complex64: "_compare_complex",
    np.complex128: "_compare_complex",
    str: "_compare_str",
    bytes: "_compare_bytes",
    # Pandas scalar types
    pd.Timestamp: "_compare_timestamp",
    pd.Timedelta: "_compare_timedelta",
    # Numpy datetime types
    np.datetime64: "_compare_datetime64",
    np.timedelta64: "_compare_timedelta64",
    # Container types
    np.ndarray: "_compare_ndarray",
    pd.Series: "_compare_series",
    pd.DataFrame: "_compare_dataframe",
    pd.Index: "_compare_index",
    list: "_compare_list",
    tuple: "_compare_tuple",
    dict: "_compare_dict",
    set: "_compare_set",
    frozenset: "_compare_frozenset",
}

# CatBoost Pool - add to dispatch table if available
try:
    from catboost import Pool as CatBoostPool
    _COMPARE_DISPATCH[CatBoostPool] = "_compare_catboost_pool"
except ImportError:
    CatBoostPool = None  # type: ignore

# Keras models - add to dispatch table if available
# We register lazily to avoid import-time side effects (matplotlib backend, etc.)
_keras_dispatch_registered = False


def _is_keras_model(x) -> bool:
    """Check if x is a Keras model without importing Keras.

    Uses module name checking to avoid the expensive Keras import (~3s).
    """
    cls = type(x)
    module = getattr(cls, '__module__', '') or ''
    # Ensure module is a string (can be getset_descriptor for some C types)
    if not isinstance(module, str):
        return False
    # Check for tensorflow.keras or standalone keras
    return 'keras' in module and any(
        base.__name__ in ('Model', 'Sequential')
        for base in cls.__mro__
        if hasattr(base, '__name__')
    )


def _register_keras_dispatch_if_needed():
    """Register Keras model types in dispatch table lazily.

    IMPORTANT: This is expensive (~3s) due to Keras import. Only call when
    we know we're dealing with a Keras model (use _is_keras_model first).
    """
    global _keras_dispatch_registered
    if _keras_dispatch_registered:
        return
    _keras_dispatch_registered = True

    # Try tensorflow.keras first (more common), then standalone keras
    try:
        from tensorflow.keras.models import Sequential as TFSequential
        from tensorflow.keras.models import Model as TFModel
        _COMPARE_DISPATCH[TFSequential] = "_compare_keras_model"
        _COMPARE_DISPATCH[TFModel] = "_compare_keras_model"
    except ImportError:
        pass

    try:
        from keras.models import Sequential as KerasSequential
        from keras.models import Model as KerasModel
        _COMPARE_DISPATCH[KerasSequential] = "_compare_keras_model"
        _COMPARE_DISPATCH[KerasModel] = "_compare_keras_model"
    except ImportError:
        pass


# PyTorch models - add to dispatch table if available
# We register lazily to avoid import-time side effects
_pytorch_dispatch_registered = False


def _is_pytorch_model(x) -> bool:
    """Check if x is a PyTorch nn.Module without importing torch.

    Uses module name and MRO checking to avoid the PyTorch import.
    """
    cls = type(x)
    module = getattr(cls, '__module__', '') or ''
    # Ensure module is a string
    if not isinstance(module, str):
        return False
    if not module.startswith('torch'):
        return False
    # Check MRO for nn.Module base class
    for base in cls.__mro__:
        base_module = getattr(base, '__module__', '') or ''
        if base.__name__ == 'Module' and 'torch.nn' in base_module:
            return True
    return False


def _register_pytorch_dispatch_if_needed():
    """Register PyTorch model types in dispatch table lazily."""
    global _pytorch_dispatch_registered
    if _pytorch_dispatch_registered:
        return
    _pytorch_dispatch_registered = True

    try:
        import torch.nn as nn
        _COMPARE_DISPATCH[nn.Module] = "_compare_pytorch_model"
    except ImportError:
        pass


# LightGBM models - add to dispatch table if available
# We register lazily to avoid import-time side effects
_lightgbm_dispatch_registered = False


def _is_lightgbm_model(x) -> bool:
    """Check if x is a LightGBM model without importing lightgbm.

    Uses module name checking to avoid the lightgbm import (~1s).
    """
    cls = type(x)
    module = getattr(cls, '__module__', '') or ''
    # Ensure module is a string
    if not isinstance(module, str):
        return False
    # Check for lightgbm module
    if not module.startswith('lightgbm'):
        return False
    # Check for sklearn estimator classes
    return cls.__name__ in ('LGBMRegressor', 'LGBMClassifier', 'LGBMRanker', 'LGBMModel')


def _register_lightgbm_dispatch_if_needed():
    """Register LightGBM model types in dispatch table lazily."""
    global _lightgbm_dispatch_registered
    if _lightgbm_dispatch_registered:
        return
    _lightgbm_dispatch_registered = True

    try:
        import lightgbm as lgb
        _COMPARE_DISPATCH[lgb.LGBMRegressor] = "_compare_lightgbm_model"
        _COMPARE_DISPATCH[lgb.LGBMClassifier] = "_compare_lightgbm_model"
        _COMPARE_DISPATCH[lgb.LGBMRanker] = "_compare_lightgbm_model"
    except ImportError:
        pass


# Cache for subclass dispatch lookups
_DISPATCH_CACHE: Dict[type, Optional[str]] = {}


def _get_value_shape_info(val: Any) -> str:
    """
    Get a human-readable shape/size description for a value.
    Used for profiling to identify which values are most expensive to compare.
    """
    t = type(val)
    tname = t.__name__

    try:
        if isinstance(val, pd.DataFrame):
            return f"DataFrame({val.shape[0]}x{val.shape[1]})"
        elif isinstance(val, pd.Series):
            return f"Series({len(val)})"
        elif isinstance(val, np.ndarray):
            return f"ndarray{val.shape}"
        elif isinstance(val, dict):
            return f"dict({len(val)} keys)"
        elif isinstance(val, (list, tuple)):
            return f"{tname}({len(val)} items)"
        elif isinstance(val, set):
            return f"set({len(val)} items)"
        elif isinstance(val, frozenset):
            return f"frozenset({len(val)} items)"
        elif isinstance(val, pd.Index):
            return f"Index({len(val)})"
        elif isinstance(val, (DataFrameGroupBy, SeriesGroupBy)):
            return f"GroupBy({val.ngroups} groups)"
        else:
            return tname
    except Exception:
        return tname


# Profiling stats accumulator (reset per diff() call)
class DiffProfileStats:
    """Accumulates per-type timing statistics for diff profiling."""

    def __init__(self):
        self.reset()

    def reset(self):
        # type_name -> (total_time_sec, count, max_time_sec, max_info)
        self.by_type: Dict[str, List[float, int, float, str]] = {}
        self.slow_comparisons: List[Tuple[float, str, str]] = []  # (time, path, info)

    def record(self, type_info: str, elapsed_sec: float, path: str):
        """Record a comparison timing."""
        if type_info not in self.by_type:
            self.by_type[type_info] = [0.0, 0, 0.0, ""]

        stats = self.by_type[type_info]
        stats[0] += elapsed_sec
        stats[1] += 1
        if elapsed_sec > stats[2]:
            stats[2] = elapsed_sec
            stats[3] = path

        # Track slow comparisons
        if elapsed_sec >= _PROFILE_THRESHOLD_SEC:
            self.slow_comparisons.append((elapsed_sec, path, type_info))

    def log_summary(self):
        """Log a summary of timing statistics."""
        if not self.by_type:
            return

        # Sort by total time
        sorted_types = sorted(
            self.by_type.items(),
            key=lambda x: x[1][0],
            reverse=True
        )

        total_time = sum(s[0] for _, s in sorted_types)

        # for type_info, (total, count, max_t, max_path) in sorted_types:
        #     type_name = type_info.split('(')[0].replace(' ', '_')
        #     output.add_timing(f"diff:{type_name}", total)

        log(f"[diff profile] Total comparison time: {total_time*1000:.2f}ms")
        log(f"[diff profile] By type (top 10):")
        for type_info, (total, count, max_t, max_path) in sorted_types[:10]:
            pct = (total / total_time * 100) if total_time > 0 else 0
            log(f"  {type_info}: {total*1000:.2f}ms ({pct:.1f}%) | "
                f"{count} calls | max={max_t*1000:.2f}ms at {max_path}")

        # Log slowest individual comparisons
        if self.slow_comparisons:
            self.slow_comparisons.sort(reverse=True)
            log(f"[diff profile] Slowest comparisons (>{_PROFILE_THRESHOLD_SEC*1000:.1f}ms):")
            for elapsed, path, type_info in self.slow_comparisons[:10]:
                log(f"  {elapsed*1000:.2f}ms: {path} ({type_info})")


# Global profile stats instance (reused per diff call)
_profile_stats = DiffProfileStats()


def _is_floating_dtype(dtype) -> bool:
    """
    Safely check if a dtype is a floating point type.

    Handles both numpy dtypes and pandas extension dtypes.

    Args:
        dtype: A numpy dtype or pandas extension dtype

    Returns:
        True if the dtype represents floating point numbers
    """
    try:
        # Try pandas first (works for both numpy and extension dtypes)
        from pandas.api.types import is_float_dtype
        return is_float_dtype(dtype)
    except Exception:
        # Fallback to numpy (but this will fail for extension dtypes)
        try:
            return np.issubdtype(dtype, np.floating)
        except (TypeError, AttributeError):
            return False


def _is_integer_dtype(dtype) -> bool:
    """
    Safely check if a dtype is an integer type.

    Handles both numpy dtypes and pandas extension dtypes.

    Args:
        dtype: A numpy dtype or pandas extension dtype

    Returns:
        True if the dtype represents integers
    """
    try:
        # Try pandas first (works for both numpy and extension dtypes)
        from pandas.api.types import is_integer_dtype
        return is_integer_dtype(dtype)
    except Exception:
        # Fallback to numpy (but this will fail for extension dtypes)
        try:
            return np.issubdtype(dtype, np.integer)
        except (TypeError, AttributeError):
            return False


def _all_elements_are_floats(arr) -> bool:
    """
    Check if all elements in an object-dtype array/series are floats.

    Args:
        arr: numpy array or pandas Series with object dtype

    Returns:
        True if all non-NaN elements are float instances, False otherwise
    """
    if isinstance(arr, pd.Series):
        arr = arr.values

    # Check each element
    for elem in arr.flat:
        # Skip NaN values (they're allowed in float arrays)
        if pd.isna(elem):
            continue
        # Check if element is a float (Python float or numpy floating)
        if not isinstance(elem, (float, np.floating)):
            return False

    return True


def are_compatible_dtypes(arr1, arr2) -> bool:
    """
    Check if two numpy arrays or pandas Series have compatible dtypes for equality comparison.

    Returns True if:
    - Both are integer types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, Int64, etc.)
    - Both are floating types (float16, float32, float64, Float64, etc.)
    - Both are string types (object with strings, StringDtype)
    - One is object dtype containing all floats, and the other is a floating type
    - Both are the exact same type

    Args:
        arr1: First numpy array or pandas Series
        arr2: Second numpy array or pandas Series
    """
    # Extract dtypes
    dtype1 = arr1.dtype
    dtype2 = arr2.dtype

    # Same dtype - always compatible
    if dtype1 == dtype2:
        return True

    # Handle pandas extension dtypes (StringDtype, Int64Dtype, etc.)
    # These need special handling before numpy dtype checks
    from pandas.api.types import is_extension_array_dtype, is_string_dtype, is_integer_dtype, is_float_dtype

    # Both are string types (object with strings or StringDtype)
    if is_string_dtype(dtype1) and is_string_dtype(dtype2):
        return True

    # Both are integer types (numpy int or pandas Int64, etc.)
    if is_integer_dtype(dtype1) and is_integer_dtype(dtype2):
        return True

    # Both are floating types (numpy float or pandas Float64, etc.)
    if is_float_dtype(dtype1) and is_float_dtype(dtype2):
        return True

    # If either is an extension dtype that we haven't handled above,
    # they're not compatible unless they're exactly the same (already checked)
    if is_extension_array_dtype(dtype1) or is_extension_array_dtype(dtype2):
        return False

    # For numpy dtypes only (not extension dtypes):
    # Both are integer types (signed or unsigned)
    if _is_integer_dtype(dtype1) and _is_integer_dtype(dtype2):
        return True

    # Both are floating types
    if _is_floating_dtype(dtype1) and _is_floating_dtype(dtype2):
        return True

    # Check object dtype compatibility with float dtypes
    # If one is object and the other is floating, check if object contains all floats
    if dtype1 == np.object_ and _is_floating_dtype(dtype2):
        return _all_elements_are_floats(arr1)

    if dtype2 == np.object_ and _is_floating_dtype(dtype1):
        return _all_elements_are_floats(arr2)

    return False


def _get_column_as_series(df: pd.DataFrame, col) -> pd.Series:
    """
    Safely get a column from a DataFrame as a Series.

    Handles MultiIndex columns by using iloc with get_loc to ensure
    we always get a single Series, not a DataFrame.

    Args:
        df: The DataFrame to get the column from
        col: The column name (can be a tuple for MultiIndex)

    Returns:
        The column as a pandas Series
    """
    # Use get_loc to find the integer position, then iloc to access
    # This handles MultiIndex columns correctly
    col_idx = df.columns.get_loc(col)
    # get_loc can return an int, slice, or boolean array for duplicates
    if isinstance(col_idx, int):
        return df.iloc[:, col_idx]
    else:
        # For duplicates or other cases, fall back to loc which handles it
        # but ensure we get a Series (take first match if multiple)
        result = df.loc[:, col]
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0]
        return result


class Diff:
    """
    Compare two Jupyter kernel user namespaces for equality.
    Checks value equality and isomorphic pointer structure.
    Returns structured diff results with all differences found.
    """

    def __init__(
        self,
        rtol=1e-5,
        atol=1e-8,
        max_diffs_per_container: int = 10,
        max_diffs_per_structure: int = 100,
        sample_large_arrays: bool = False,
        strict: bool = True,
        report_close: bool = True,
        use_leq: bool = False,
        column_rbw: Optional[Dict[str, Set[str]]] = None,
        structural_reads: Optional[Dict[str, Set[str]]] = None,
        structural_mode: StructuralTrackingMode = StructuralTrackingMode.OFF,
    ):
        """
        Initialize the Diff comparator.

        Args:
            rtol: Relative tolerance for floating point comparisons (default: 1e-5)
            atol: Absolute tolerance for floating point comparisons (default: 1e-8)
            max_diffs_per_container: Maximum differences to collect per container (default: 1000)
            max_diffs_per_structure: Maximum differences to collect per structured data
                                     (arrays, Series, DataFrames) (default: 5)
            sample_large_arrays: Whether to sample large arrays instead of full comparison (default: False)
            strict: If True, require exact type matches. If False, allow compatible types
                    (e.g., int vs float, list vs ndarray) (default: True)
            report_close: If True, report floats that are close (within tolerance) with status='close'.
                         If False, treat close values as equal and don't report them (default: True)
            use_leq: If True, check if b is a conservative extension of a (default: False).
                     This means: (1) extra keys in b are allowed, (2) DataFrames in b can have
                     extra columns as long as all columns from a are present and equal.
            column_rbw: Column-level reads-before-writes for DataFrames (default: None).
                       Maps variable path (e.g., 'df', 'data["train"]') to set of column names
                       that were read-before-write. When provided with use_leq=True, only these
                       columns are compared for each DataFrame.
            structural_reads: Structural attribute accesses per variable path (default: None).
                       Maps variable path to set of structural attributes accessed (e.g., 'columns',
                       'shape', 'len', 'iter'). Used with structural_mode to enforce or warn about
                       structural changes to variables where these attributes were read.
            structural_mode: How to handle structural reads (default: OFF).
                       - OFF: Ignore structural reads, use standard LEQ behavior
                       - WARN: Track structural changes as warnings but don't fail
                       - ENFORCE: Treat structural changes as differences when attributes were read

        Example:
            >>> # Default behavior - reports close values
            >>> differ = Diff(rtol=1e-5)
            >>> result = differ.diff({'x': 1.0000001}, {'x': 1.0000002})
            >>> 'x' in result  # True - close value is reported

            >>> # With report_close=False - treats close as equal
            >>> differ = Diff(rtol=1e-5, report_close=False)
            >>> result = differ.diff({'x': 1.0000001}, {'x': 1.0000002})
            >>> 'x' in result  # False - close value not reported
        """
        self.rtol = rtol
        self.atol = atol
        self.max_diffs_per_container = max_diffs_per_container
        self.max_diffs_per_structure = max_diffs_per_structure
        self.sample_large_arrays = sample_large_arrays
        self.strict = strict
        self.report_close = report_close
        self.use_leq = use_leq
        self.column_rbw = column_rbw or {}
        self.structural_reads = structural_reads or {}
        self.structural_mode = structural_mode
        # Track object identities to ensure pointer structure matches
        self.id_map_a = {}  # Maps id(obj_a) -> canonical_id
        self.id_map_b = {}  # Maps id(obj_b) -> canonical_id
        self.next_canonical_id = 0
        # Accumulated warnings for WARN mode
        self._warnings: List[str] = []

    def _is_immutable_atomic(self, val: Any) -> bool:
        """
        Check if a value is an immutable atomic type that doesn't need pointer tracking.

        Immutable atomics are values where identity doesn't matter - only value matters.
        These include: None, bool, int, float, complex, str, bytes

        Args:
            val: The value to check

        Returns:
            True if val is an immutable atomic, False otherwise
        """
        # Fast path: exact type lookup in frozenset (O(1))
        t = type(val)
        if t in _IMMUTABLE_ATOMIC_TYPES:
            return True

        # Fallback for numpy subclasses not in the set
        # (e.g., np.longdouble on some platforms)
        if isinstance(val, (np.integer, np.floating, np.complexfloating)):
            return True

        return False

    def _get_type_category(self, val: Any) -> str:
        """
        Get the type category for immutable atomic values.

        This groups semantically equivalent types together:
        - All integer types (int, np.int8, np.int16, np.int32, np.int64) → "integer"
        - All float types (float, np.float16, np.float32, np.float64) → "float"
        - All complex types (complex, np.complex64, np.complex128) → "complex"

        This allows numpy scalar type variants to be compared as the same type category,
        so that np.int64(5) == np.int32(5) or int(5) == np.int64(5).

        Returns:
            String representing the type category
        """
        # Fast path: exact type lookup in dict (O(1))
        t = type(val)
        category = _TYPE_CATEGORY_MAP.get(t)
        if category is not None:
            return category

        # Fallback for numpy subclasses not in the map
        # (e.g., np.longdouble on some platforms, or user-defined subclasses)
        if isinstance(val, np.integer):
            return "integer"
        if isinstance(val, np.floating):
            return "float"
        if isinstance(val, np.complexfloating):
            return "complex"

        return "other"

    def diff(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        keys_to_include: Set[str] | None = None,
    ) -> MemoryCheckpointDiffResult:
        """
        Compare two user namespaces.

        Args:
            a: First namespace dictionary
            b: Second namespace dictionary
            keys_to_include: Optional set of keys to compare (default: all keys)

        Returns:
            MemoryCheckpointDiffResult instance containing diff trees for variables with differences.
            The differences dict is empty if all variables are equal.
            The warnings list contains structural warnings when structural_mode is WARN.
        """
        if keys_to_include is None:
            keys_to_include = set(a.keys()) | set(b.keys())

        # Reset identity tracking for each comparison
        self.id_map_a = {}
        self.id_map_b = {}
        self.next_canonical_id = 0
        self._warnings = []  # Reset warnings

        # Reset profiling stats if profiling is enabled
        if _PROFILE_DIFF:
            _profile_stats.reset()
            diff_start = time.perf_counter()

        differences: Dict[str, DiffNode] = {}

        # Check for variables only in a
        only_in_a = set(a.keys()) - set(b.keys())
        for var in only_in_a & keys_to_include:
            differences[var] = ValueComparison(
                status="different",
                value1=a[var],
                value2=None,
                message="Variable was removed",
            )

        # Check for variables only in b (skip if use_leq since extra keys are allowed)
        if not self.use_leq:
            only_in_b = set(b.keys()) - set(a.keys())
            for var in only_in_b & keys_to_include:
                differences[var] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=b[var],
                    message="Variable was added",
                )

        # Compare common variables - only add to differences if not equal
        common_vars = set(a.keys()) & set(b.keys())
        with timer(key="diff:compare_loop", message=f"Comparing {len(common_vars & keys_to_include)} common variables"):
            for var in sorted(
                common_vars & keys_to_include
            ):  # Sort for deterministic output
                if _PROFILE_DIFF:
                    var_start = time.perf_counter()
                    type_info = _get_value_shape_info(a[var])

                    mod_a = type(a[var]).__module__
                    mod_b = type(b[var]).__module__
                    ta = (mod_a if isinstance(mod_a, str) else '<unknown>') + "." + type(a[var]).__name__
                    tb = (mod_b if isinstance(mod_b, str) else '<unknown>') + "." + type(b[var]).__name__
                    with timer(key=f"diff:{ta}-{tb}", message=f"Comparing {var} ({ta} vs {tb})"):
                        diff_result = self._compare_values(a[var], b[var], path=var)

                    elapsed = time.perf_counter() - var_start
                    _profile_stats.record(type_info, elapsed, var)

                else:

                    diff_result = self._compare_values(a[var], b[var], path=var)

                if diff_result:  # Only include if there are differences
                    differences[var] = diff_result

        # Log profiling summary if enabled
        if _PROFILE_DIFF:
            total_elapsed = time.perf_counter() - diff_start
            log(f"[diff profile] Total diff time: {total_elapsed*1000:.2f}ms "
                f"({len(common_vars & keys_to_include)} variables compared)")
            _profile_stats.log_summary()

        return MemoryCheckpointDiffResult(differences=differences, warnings=list(self._warnings))

    def _log_column_timings(
        self, df_path: str, column_timings: List[Tuple[float, str, str]]
    ) -> None:
        """
        Log timing information for DataFrame column comparisons.

        Args:
            df_path: The path to the DataFrame variable
            column_timings: List of (elapsed_sec, column_name, dtype) tuples
        """
        if not column_timings:
            return

        # Sort by time descending
        column_timings.sort(reverse=True)
        total_time = sum(t[0] for t in column_timings)

        # Only log if total time is significant
        if total_time < _PROFILE_THRESHOLD_SEC:
            return

        log(f"[diff profile] DataFrame {df_path} column breakdown ({total_time*1000:.2f}ms total, {len(column_timings)} cols):")

        # Log top 5 slowest columns
        for elapsed, col_name, dtype in column_timings[:5]:
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            log(f"    {col_name} ({dtype}): {elapsed*1000:.3f}ms ({pct:.1f}%)")

        # If there are more columns, show summary
        if len(column_timings) > 5:
            remaining_time = sum(t[0] for t in column_timings[5:])
            log(f"    ... and {len(column_timings) - 5} more columns: {remaining_time*1000:.3f}ms")

    def _check_structural_change(
        self, path: str, change_type: str, detail: str
    ) -> Optional[ValueComparison]:
        """
        Check if a structural change should be reported based on structural_mode.

        Args:
            path: Variable path (e.g., 'df', 'data["train"]')
            change_type: Type of structural change ('columns', 'rows', 'index', etc.)
            detail: Human-readable description of the change

        Returns:
            ValueComparison if ENFORCE mode and structural read exists, None otherwise.
            In WARN mode, adds to self._warnings instead.
        """
        if self.structural_mode == StructuralTrackingMode.OFF:
            return None

        # Check if this path has any structural reads
        structural_attrs = self.structural_reads.get(path, set())
        if not structural_attrs:
            return None

        # Determine which structural reads are relevant for this change type
        # Note: shape and size reveal BOTH row and column structure
        column_revealing = {
            'columns', 'keys', 'iter', 'dtypes', 'T', 'axes', 'values',
            'describe', 'to_dict', 'info', 'head', 'tail', 'sample',
            'select_dtypes', 'to_records', 'memory_usage',
            'shape', 'size',  # shape=(rows,cols), size=rows*cols
        }
        row_revealing = {'index', 'len', 'shape', 'size', 'empty'}

        relevant_attrs = set()
        if change_type in ('columns', 'column_add', 'column_remove'):
            relevant_attrs = structural_attrs & column_revealing
        elif change_type in ('rows', 'row_add', 'row_remove', 'shape', 'len'):
            relevant_attrs = structural_attrs & row_revealing
        elif change_type == 'index':
            relevant_attrs = structural_attrs & {'index'}
        elif change_type == 'dtype':
            relevant_attrs = structural_attrs & {'dtype', 'dtypes'}
        else:
            # Generic structural change
            relevant_attrs = structural_attrs

        if not relevant_attrs:
            return None

        message = f"Structural change at {path}: {detail} (read: {', '.join(sorted(relevant_attrs))})"

        if self.structural_mode == StructuralTrackingMode.WARN:
            self._warnings.append(message)
            return None
        else:  # ENFORCE
            return ValueComparison(
                status="different",
                value1=None,
                value2=None,
                message=message,
            )

    def _compare_values(
        self, val_a: Any, val_b: Any, path: str = ""
    ) -> Optional[DiffNode]:
        """
        Compare two values, dispatching to type-specific methods.
        Returns None if equal, otherwise returns DiffNode with differences.
        """
        # Fast path: same object means definitely equal (identity short-circuit)
        # This eliminates O(n) comparison for reused objects in incremental checkpoints
        if val_a is val_b:
            return None

        ta = type(val_a)
        tb = type(val_b)

        # Handle cudf objects by converting to pandas (all cudf logic in cudf_compat)
        if cudf_compat.are_both_cudf_same_type(val_a, val_b):
            return cudf_compat.diff_cudf(val_a, val_b, path, self)

        # Skip pointer tracking for immutable atomic values
        # For these types, only value equality matters, not object identity
        both_immutable_atomic = self._is_immutable_atomic(
            val_a
        ) and self._is_immutable_atomic(val_b)

        # Get IDs for all values (needed for error handling even if not tracking)
        id_a, id_b = id(val_a), id(val_b)
        registered_ids = False  # Track whether we registered these IDs

        if not both_immutable_atomic:
            # Pointer tracking for mutable containers and complex objects

            # If we've seen val_a before, check if pointer structure matches
            if id_a in self.id_map_a:
                canonical_a = self.id_map_a[id_a]
                if id_b in self.id_map_b:
                    canonical_b = self.id_map_b[id_b]
                    if canonical_a != canonical_b:
                        return ValueComparison(
                            status="different",
                            value1=val_a,
                            value2=val_b,
                            message=f"Pointer structure mismatch at {path}",
                        )
                else:
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Pointer structure mismatch at {path} (first namespace has reference to earlier object)",
                    )
                return None  # Already compared, and structure matches

            # If we've seen val_b before but not val_a
            if id_b in self.id_map_b:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Pointer structure mismatch at {path} (second namespace has reference to earlier object)",
                )

            # Register these objects with the same canonical ID
            # We do this before comparing to handle circular references
            canonical_id = self.next_canonical_id
            self.next_canonical_id += 1
            self.id_map_a[id_a] = canonical_id
            self.id_map_b[id_b] = canonical_id
            registered_ids = True  # Mark that we registered these IDs

        # Type checking
        # For immutable atomics, use category-based type checking
        # This allows np.int64 to match np.int32, int to match np.int64, etc.
        if both_immutable_atomic:
            type_a_category = self._get_type_category(val_a)
            type_b_category = self._get_type_category(val_b)

            if type_a_category != type_b_category:
                # Different type categories (e.g., int vs str, int vs float)
                # In non-strict mode, check if types are compatible (e.g., int vs float)
                if not self.strict:
                    is_compatible, compat_type = self._types_compatible(val_a, val_b)
                    if is_compatible:
                        # Use flexible comparison for compatible types
                        if compat_type == "numeric":
                            result = self._compare_numeric_flexible(val_a, val_b, path)
                        else:
                            # Should not happen for atomics, but handle gracefully
                            result = ValueComparison(
                                status="different",
                                value1=val_a,
                                value2=val_b,
                                message=f"Type category mismatch at {path}: {type_a_category} vs {type_b_category} ({type(val_a).__name__} vs {type(val_b).__name__})",
                            )
                        return result

                # Strict mode or incompatible types
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Type category mismatch at {path}: {type_a_category} vs {type_b_category} ({type(val_a).__name__} vs {type(val_b).__name__})",
                )
            # Same type category - continue to value comparison below
        else:
            # For non-atomic types (containers), use exact type matching
            if type(val_a) != type(val_b):
                # Special case: pd.NaT (NaTType) vs Timestamp/Timedelta
                # pd.NaT is a separate type that represents missing time values
                # We should report this as "one is NaT" rather than a type mismatch
                # Note: Check type name first to avoid calling pd.isna() on non-scalar types
                is_nat_a = type(val_a).__name__ == 'NaTType'
                is_nat_b = type(val_b).__name__ == 'NaTType'
                is_timestamp_a = isinstance(val_a, pd.Timestamp)
                is_timestamp_b = isinstance(val_b, pd.Timestamp)
                is_timedelta_a = isinstance(val_a, pd.Timedelta)
                is_timedelta_b = isinstance(val_b, pd.Timedelta)

                # NaT vs Timestamp
                if (is_nat_a and is_timestamp_b) or (is_timestamp_a and is_nat_b):
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Timestamp mismatch at {path}: {val_a} vs {val_b} (one is NaT)",
                    )

                # NaT vs Timedelta
                if (is_nat_a and is_timedelta_b) or (is_timedelta_a and is_nat_b):
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Timedelta mismatch at {path}: {val_a} vs {val_b} (one is NaT)",
                    )

                # In non-strict mode, check if types are compatible
                if not self.strict:
                    is_compatible, compat_type = self._types_compatible(val_a, val_b)
                    if is_compatible:
                        # Use flexible comparison for compatible types
                        if compat_type == "numeric":
                            result = self._compare_numeric_flexible(val_a, val_b, path)
                        elif compat_type in ("list_array", "tuple_array"):
                            result = self._compare_list_array_flexible(
                                val_a, val_b, path
                            )
                        else:
                            # Should not happen, but handle gracefully
                            result = ValueComparison(
                                status="different",
                                value1=val_a,
                                value2=val_b,
                                message=f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}",
                            )

                        # If comparison found a difference, unregister these objects
                        if result is not None and registered_ids:
                            del self.id_map_a[id_a]
                            del self.id_map_b[id_b]

                        return result

                # Strict mode or incompatible types - unregister and return type mismatch
                if registered_ids:
                    del self.id_map_a[id_a]
                    del self.id_map_b[id_b]
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}",
                )

        # Dispatch to type-specific methods using optimized dict lookup
        result: Optional[DiffNode] = None
        t = type(val_a)

        # Fast path: exact type lookup in dispatch table (O(1))
        method_name = _COMPARE_DISPATCH.get(t)

        if method_name is not None:
            # Found exact match - dispatch directly
            if method_name == "_dispatch_none":
                result = None  # Both None, so equal
            else:
                method = getattr(self, method_name)
                result = method(val_a, val_b, path)
        else:
            # Check cache for previously resolved subclasses
            method_name = _DISPATCH_CACHE.get(t)
            if method_name is not None:
                method = getattr(self, method_name)
                result = method(val_a, val_b, path)
            else:
                # Fallback: isinstance checks for subclasses and special cases
                # Order matters for inheritance relationships
                if isinstance(val_a, np.timedelta64):
                    _DISPATCH_CACHE[t] = "_compare_timedelta64"
                    result = self._compare_timedelta64(val_a, val_b, path)
                elif isinstance(val_a, np.integer):
                    _DISPATCH_CACHE[t] = "_compare_int"
                    result = self._compare_int(val_a, val_b, path)
                elif isinstance(val_a, np.floating):
                    _DISPATCH_CACHE[t] = "_compare_float"
                    result = self._compare_float(val_a, val_b, path)
                elif isinstance(val_a, np.complexfloating):
                    _DISPATCH_CACHE[t] = "_compare_complex"
                    result = self._compare_complex(val_a, val_b, path)
                elif isinstance(val_a, (DataFrameGroupBy, SeriesGroupBy)):
                    _DISPATCH_CACHE[t] = "_compare_groupby"
                    result = self._compare_groupby(val_a, val_b, path)
                elif isinstance(val_a, pd.Index):
                    # Handle Index subclasses like IntervalIndex, MultiIndex, etc.
                    _DISPATCH_CACHE[t] = "_compare_index"
                    result = self._compare_index(val_a, val_b, path)
                elif isinstance(val_a, pd.DataFrame):
                    # Handle DataFrame subclasses and cudf.pandas proxy types
                    _DISPATCH_CACHE[t] = "_compare_dataframe"
                    result = self._compare_dataframe(val_a, val_b, path)
                elif isinstance(val_a, pd.Series):
                    # Handle Series subclasses and cudf.pandas proxy types
                    _DISPATCH_CACHE[t] = "_compare_series"
                    result = self._compare_series(val_a, val_b, path)
                elif _is_keras_model(val_a):
                    # Keras model - register handler lazily to avoid expensive import
                    _register_keras_dispatch_if_needed()
                    # Retry dispatch after registration
                    method_name = _COMPARE_DISPATCH.get(t)
                    if method_name is not None:
                        method = getattr(self, method_name)
                        result = method(val_a, val_b, path)
                    else:
                        # Fallback to object comparison
                        result = self._compare_object(val_a, val_b, path)
                elif _is_pytorch_model(val_a):
                    # PyTorch model - call comparison directly since subclasses
                    # (nn.Linear, nn.Sequential, etc.) won't match nn.Module in dispatch
                    _register_pytorch_dispatch_if_needed()
                    _DISPATCH_CACHE[t] = "_compare_pytorch_model"
                    result = self._compare_pytorch_model(val_a, val_b, path)
                elif _is_lightgbm_model(val_a):
                    # LightGBM model - call comparison directly
                    _register_lightgbm_dispatch_if_needed()
                    _DISPATCH_CACHE[t] = "_compare_lightgbm_model"
                    result = self._compare_lightgbm_model(val_a, val_b, path)
                elif isinstance(val_a, dict):
                    # Handle dict subclasses (OrderedDict, Counter, defaultdict, etc.)
                    _DISPATCH_CACHE[t] = "_compare_dict"
                    result = self._compare_dict(val_a, val_b, path)
                elif callable(val_a):
                    # Don't cache callables - too many different types
                    result = self._compare_callable(val_a, val_b, path)
                else:
                    # User-defined objects
                    result = self._compare_object(val_a, val_b, path)

        # If comparison found a difference, unregister these objects
        # so they don't pollute future comparisons
        # Only unregister if we actually registered (i.e., not immutable atomics)
        if result is not None and registered_ids:
            del self.id_map_a[id_a]
            del self.id_map_b[id_b]

        return result

    def _types_compatible(self, val_a: Any, val_b: Any) -> Tuple[bool, str]:
        """
        Check if two values have compatible types for non-strict comparison.

        Returns:
            Tuple of (is_compatible, compatibility_type) where compatibility_type is:
            - "numeric": int vs float compatibility
            - "list_array": list vs ndarray compatibility
            - "tuple_array": tuple vs ndarray compatibility
            - "": not compatible
        """
        # Check numeric compatibility (int vs float)
        is_int_a = isinstance(val_a, (int, np.integer)) and not isinstance(val_a, bool)
        is_int_b = isinstance(val_b, (int, np.integer)) and not isinstance(val_b, bool)
        is_float_a = isinstance(val_a, (float, np.floating))
        is_float_b = isinstance(val_b, (float, np.floating))

        if (is_int_a and is_float_b) or (is_float_a and is_int_b):
            return (True, "numeric")

        # Check list vs array compatibility
        if isinstance(val_a, list) and isinstance(val_b, np.ndarray):
            return (True, "list_array")
        if isinstance(val_a, np.ndarray) and isinstance(val_b, list):
            return (True, "list_array")

        # Check tuple vs array compatibility
        if isinstance(val_a, tuple) and isinstance(val_b, np.ndarray):
            return (True, "tuple_array")
        if isinstance(val_a, np.ndarray) and isinstance(val_b, tuple):
            return (True, "tuple_array")

        return (False, "")

    def _get_list_depth(self, lst: Any) -> int:
        """
        Determine the nesting depth of a list/tuple.

        Returns:
            0 for non-list/tuple
            1 for flat list/tuple
            2 for list/tuple of lists/tuples
            etc.
        """
        if not isinstance(lst, (list, tuple)):
            return 0

        if len(lst) == 0:
            return 1

        # Check first element to determine depth
        max_depth = 0
        for item in lst:
            if isinstance(item, (list, tuple)):
                depth = 1 + self._get_list_depth(item)
                max_depth = max(max_depth, depth)
            else:
                max_depth = max(max_depth, 1)

        return max_depth

    def _compare_numeric_flexible(
        self, val_a: Any, val_b: Any, path: str
    ) -> Optional[ValueComparison]:
        """
        Compare int vs float values in non-strict mode.
        Converts int to float and uses float comparison logic.
        """
        # Convert both to float for comparison
        float_a = float(val_a)
        float_b = float(val_b)

        return self._compare_float(float_a, float_b, path)

    def _compare_list_array_flexible(
        self, val_a: Any, val_b: Any, path: str
    ) -> Optional[DiffNode]:
        """
        Compare list/tuple vs ndarray in non-strict mode.
        Validates structure matches and compares element-by-element.
        """
        # Determine which is the list/tuple and which is the array
        if isinstance(val_a, (list, tuple)):
            lst, arr = val_a, val_b
            lst_is_a = True
        else:
            lst, arr = val_b, val_a
            lst_is_a = False

        # Check that list depth matches array dimensions
        list_depth = self._get_list_depth(lst)
        array_ndim = arr.ndim

        if list_depth != array_ndim:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Structure mismatch at {path}: {'list' if isinstance(lst, list) else 'tuple'} depth {list_depth} vs array ndim {array_ndim}",
            )

        # Convert list/tuple to array for shape comparison
        try:
            lst_as_array = np.array(lst)
        except (ValueError, TypeError) as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Cannot convert {'list' if isinstance(lst, list) else 'tuple'} to array at {path}: {str(e)}",
            )

        # Check shapes match
        if lst_as_array.shape != arr.shape:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Shape mismatch at {path}: {'list' if isinstance(lst, list) else 'tuple'} shape {lst_as_array.shape} vs array shape {arr.shape}",
            )

        # Compare element-by-element
        # Flatten both for easier iteration
        flat_lst = lst_as_array.ravel()
        flat_arr = arr.ravel()

        for i in range(len(flat_lst)):
            lst_val = flat_lst[i]
            arr_val = flat_arr[i]

            # Use flexible comparison for elements
            elem_diff = self._compare_values(lst_val, arr_val, f"{path}[{i}]")
            if elem_diff:
                message = elem_diff.message
                # Return first difference found
                idx = np.unravel_index(i, arr.shape)
                idx_tuple = tuple(int(x) for x in idx)
                return ValueComparison(
                    status="different",
                    value1=lst_val,
                    value2=arr_val,
                    message=f"Element mismatch at {path}[{idx_tuple}]: {lst_val} vs {arr_val} - {type(lst_val)} vs {type(arr_val)}"
                    + message,
                )

        # All elements match
        return None

    def _compare_bool(
        self, val_a: bool, val_b: bool, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Bool mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_int(
        self, val_a: int, val_b: int, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Integer mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_float(
        self, val_a: float, val_b: float, path: str
    ) -> Optional[ValueComparison]:
        # Handle NaN
        is_nan_a = math.isnan(val_a) if isinstance(val_a, float) else np.isnan(val_a)
        is_nan_b = math.isnan(val_b) if isinstance(val_b, float) else np.isnan(val_b)

        if is_nan_a and is_nan_b:
            return None  # Both NaN, considered equal
        if is_nan_a or is_nan_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Float mismatch at {path}: {val_a} vs {val_b} (one is NaN)",
            )

        # Check exact equality first
        if val_a == val_b:
            return None  # Exactly equal

        # Check if close within tolerance
        is_close = (
            math.isclose(val_a, val_b, rel_tol=self.rtol, abs_tol=self.atol)
            if isinstance(val_a, float)
            else np.isclose(val_a, val_b, rel_tol=self.rtol, abs_tol=self.atol)
        )
        if is_close:
            # If report_close is False, treat close values as equal (no difference)
            if not self.report_close:
                return None
            return ValueComparison(
                status="close",
                value1=val_a,
                value2=val_b,
                message=f"Float close at {path}: {val_a} vs {val_b} (within tolerance)",
            )

        # Not equal and not close
        return ValueComparison(
            status="different",
            value1=val_a,
            value2=val_b,
            message=f"Float mismatch at {path}: {val_a} vs {val_b}",
        )

    def _compare_complex(
        self, val_a: complex, val_b: complex, path: str
    ) -> Optional[DiffNode]:
        children: Dict[str, DiffNode] = {}
        real_diff = self._compare_float(val_a.real, val_b.real, f"{path}.real")
        if real_diff:
            children[".real"] = real_diff
        imag_diff = self._compare_float(val_a.imag, val_b.imag, f"{path}.imag")
        if imag_diff:
            children[".imag"] = imag_diff
        if children:
            return CompoundDiff(source_type="complex", children=children, truncated=False)
        return None

    def _compare_str(
        self, val_a: str, val_b: str, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"String mismatch at {path}: '{val_a}' vs '{val_b}'",
            )
        return None

    def _compare_bytes(
        self, val_a: bytes, val_b: bytes, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Bytes mismatch at {path}",
            )
        return None

    def _compare_callable(
        self, val_a: Any, val_b: Any, path: str
    ) -> Optional[DiffNode]:
        """
        Compare callables.
        - For type objects (classes): compare by identity
        - For functions: ignore differences (redefinitions are common)
        - For bound methods: compare __func__ and __self__
        """
        # Check if both are type objects (classes)
        if isinstance(val_a, type) and isinstance(val_b, type):
            if val_a is not val_b:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Type mismatch at {path}: {val_a.__name__} vs {val_b.__name__}",
                )
            return None

        # Check if both are bound methods
        is_method_a = hasattr(val_a, "__self__") and hasattr(val_a, "__func__")
        is_method_b = hasattr(val_b, "__self__") and hasattr(val_b, "__func__")

        if is_method_a and is_method_b:
            # Both are bound methods - compare the underlying function and instance
            diffs = {}
            func_diff = self._compare_values(
                val_a.__func__, val_b.__func__, f"{path}.__func__"
            )
            if func_diff:
                diffs[".__func__"] = func_diff

            self_diff = self._compare_values(
                val_a.__self__, val_b.__self__, f"{path}.__self__"
            )
            if self_diff:
                diffs[".__self__"] = self_diff

            return CompoundDiff(source_type="callable", children=diffs) if diffs else None
        elif is_method_a != is_method_b:
            # One is a bound method, the other isn't
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Callable type mismatch at {path}: bound method vs function",
            )
        else:
            # ignore function/callable mismatch
            return None

    def _compare_ndarray(
        self, val_a: np.ndarray, val_b: np.ndarray, path: str
    ) -> Optional[DiffNode]:
        """Compare numpy arrays, collecting up to max_diffs_per_container differences.

        This method records structural mismatches (shape, dtype) but continues
        comparing elements where possible.
        """
        # ======================================================================
        # FAST PATH: If arrays have same shape and dtype, try vectorized
        # equality check before any other processing.
        # ======================================================================
        if val_a.shape == val_b.shape and val_a.dtype == val_b.dtype:
            try:
                if _is_floating_dtype(val_a.dtype) or np.issubdtype(val_a.dtype, np.complexfloating):
                    if np.allclose(val_a, val_b, rtol=self.rtol, atol=self.atol, equal_nan=True):
                        return None  # Arrays are equal
                else:
                    if np.array_equal(val_a, val_b):
                        return None  # Arrays are equal
            except Exception:
                pass  # Fall through to detailed comparison
        # ======================================================================

        children: Dict[str, DiffNode] = {}
        truncated = False

        # Check shape - record mismatch but can't compare elements with different shapes
        if val_a.shape != val_b.shape:
            children["_shape"] = ValueComparison(
                status="different",
                value1=val_a.shape,
                value2=val_b.shape,
                message=f"Array shape mismatch at {path}: {val_a.shape} vs {val_b.shape}",
            )
            # Can't compare elements with different shapes
            return CompoundDiff(source_type="array", children=children, truncated=False)

        # Check dtype compatibility - record mismatch but can't compare incompatible dtypes
        if not are_compatible_dtypes(val_a, val_b):
            children["_dtype"] = ValueComparison(
                status="different",
                value1=val_a.dtype,
                value2=val_b.dtype,
                message=f"Array dtype mismatch at {path}: {val_a.dtype} vs {val_b.dtype}",
            )
            # Can't compare elements with incompatible dtypes
            return CompoundDiff(source_type="array", children=children, truncated=False)

        # Cast to common type if dtypes are compatible but different
        if val_a.dtype != val_b.dtype:
            # Special handling for object dtype with floats
            if val_a.dtype == np.object_ and _is_floating_dtype(val_b.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            elif val_b.dtype == np.object_ and _is_floating_dtype(val_a.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            else:
                # Check if either is an extension dtype
                from pandas.api.types import is_extension_array_dtype
                if is_extension_array_dtype(val_a.dtype) or is_extension_array_dtype(val_b.dtype):
                    # Extension dtypes can't use np.promote_types, compare as-is
                    val_a_cmp = val_a
                    val_b_cmp = val_b
                else:
                    # Safe to use np.promote_types for numpy dtypes
                    common_dtype = np.promote_types(val_a.dtype, val_b.dtype)
                    val_a_cmp = val_a.astype(common_dtype)
                    val_b_cmp = val_b.astype(common_dtype)
        else:
            val_a_cmp = val_a
            val_b_cmp = val_b

        # Fast path: use vectorized operations to check if arrays are equal
        try:
            if _is_floating_dtype(val_a_cmp.dtype) or np.issubdtype(
                val_a_cmp.dtype, np.complexfloating
            ):
                # Fast vectorized check for floats
                if np.allclose(
                    val_a_cmp, val_b_cmp, rtol=self.rtol, atol=self.atol, equal_nan=True
                ):
                    return None  # Arrays are equal
            else:
                # Fast vectorized check for other types
                if np.array_equal(val_a_cmp, val_b_cmp):
                    return None  # Arrays are equal
        except Exception:
            pass  # Fall through to element-by-element comparison

        # Arrays are different - collect up to max_diffs_per_container differences
        diff_count = 0

        try:
            if _is_floating_dtype(val_a_cmp.dtype) or np.issubdtype(
                val_a_cmp.dtype, np.complexfloating
            ):
                # Iterate to find specific differences
                flat_a = val_a_cmp.ravel()
                flat_b = val_b_cmp.ravel()
                flat_a_orig = val_a.ravel()
                flat_b_orig = val_b.ravel()

                for i in range(len(flat_a)):
                    a_val, b_val = flat_a[i], flat_b[i]
                    both_nan = np.isnan(a_val) and np.isnan(b_val)

                    if not both_nan and not np.allclose(
                        [a_val], [b_val], rtol=self.rtol, atol=self.atol, equal_nan=True
                    ):
                        idx = np.unravel_index(i, val_a.shape)
                        idx_tuple = tuple(int(x) for x in idx)

                        children[f"[{idx_tuple}]"] = ValueComparison(
                            status="different",
                            value1=flat_a_orig[i],
                            value2=flat_b_orig[i],
                            message=f"Array values mismatch at {path}[{idx_tuple}]: {flat_a_orig[i]} vs {flat_b_orig[i]}",
                        )

                        diff_count += 1
                        if diff_count >= self.max_diffs_per_container:
                            truncated = True
                            break
            else:
                # Iterate to find specific differences
                # Use _compare_values() to properly handle floats with tolerance even in object dtype arrays
                flat_a = val_a_cmp.ravel()
                flat_b = val_b_cmp.ravel()
                flat_a_orig = val_a.ravel()
                flat_b_orig = val_b.ravel()

                for i in range(len(flat_a)):
                    idx = np.unravel_index(i, val_a.shape)
                    idx_tuple = tuple(int(x) for x in idx)

                    elem_diff = self._compare_values(flat_a_orig[i], flat_b_orig[i], f"{path}[{idx_tuple}]")
                    if elem_diff:
                        children[f"[{idx_tuple}]"] = elem_diff

                        diff_count += 1
                        if diff_count >= self.max_diffs_per_container:
                            truncated = True
                            break
        except Exception as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Array comparison error at {path}: {str(e)}",
            )

        if children:
            return CompoundDiff(source_type="array", children=children, truncated=truncated)
        return None

    def _compare_series(
        self, val_a: pd.Series, val_b: pd.Series, path: str
    ) -> Optional[DiffNode]:
        """Compare pandas Series, collecting up to max_diffs_per_container differences.

        This method records structural mismatches (index, name, dtype) but continues
        comparing values where possible.

        Structural tracking: When structural_mode is WARN or ENFORCE and structural reads
        were made on this Series, changes to structure (length, index) are reported.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Check for structural changes (length)
        if len(val_a) != len(val_b):
            structural_diff = self._check_structural_change(
                path, 'len',
                f"Series length changed from {len(val_a)} to {len(val_b)}"
            )
            if structural_diff:
                children["_structural_len"] = structural_diff

        # Check for structural changes (dtype)
        if val_a.dtype != val_b.dtype:
            structural_diff = self._check_structural_change(
                path, 'dtype',
                f"Series dtype changed from {val_a.dtype} to {val_b.dtype}"
            )
            if structural_diff:
                children["_structural_dtype"] = structural_diff

        # Check index - if indexes differ, can't compare values by label
        indexes_match = val_a.index.equals(val_b.index)
        if not indexes_match:
            children["_index"] = ValueComparison(
                status="different",
                value1=val_a.index,
                value2=val_b.index,
                message=f"Series index mismatch at {path}",
            )
            # Can't compare values when indexes differ (labels don't match)
            return CompoundDiff(source_type="series", children=children, truncated=False)

        # Check name - record mismatch but continue
        if val_a.name != val_b.name:
            children["_name"] = ValueComparison(
                status="different",
                value1=val_a.name,
                value2=val_b.name,
                message=f"Series name mismatch at {path}: {val_a.name} vs {val_b.name}",
            )

        # Check dtype compatibility - record mismatch but can't compare incompatible dtypes
        if not are_compatible_dtypes(val_a, val_b):
            children["_dtype"] = ValueComparison(
                status="different",
                value1=val_a.dtype,
                value2=val_b.dtype,
                message=f"Series dtype mismatch at {path}: {val_a.dtype} vs {val_b.dtype}",
            )
            # Can't compare values with incompatible dtypes
            return CompoundDiff(source_type="series", children=children, truncated=False)

        # Cast to common type if dtypes are compatible but different
        if val_a.dtype != val_b.dtype:
            # Special handling for object dtype with floats
            if val_a.dtype == np.object_ and _is_floating_dtype(val_b.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            elif val_b.dtype == np.object_ and _is_floating_dtype(val_a.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            else:
                # Check if either is an extension dtype
                from pandas.api.types import is_extension_array_dtype
                if is_extension_array_dtype(val_a.dtype) or is_extension_array_dtype(val_b.dtype):
                    # Extension dtypes can't use np.promote_types, compare as-is
                    val_a_cmp = val_a
                    val_b_cmp = val_b
                else:
                    # Safe to use np.promote_types for numpy dtypes
                    common_dtype = np.promote_types(val_a.dtype, val_b.dtype)
                    val_a_cmp = val_a.astype(common_dtype)
                    val_b_cmp = val_b.astype(common_dtype)
        else:
            val_a_cmp = val_a
            val_b_cmp = val_b

        # Fast path: check if series are equal using vectorized operations
        try:
            if pd.api.types.is_float_dtype(val_a_cmp.dtype):
                # For floats, check NaN positions and values
                mask_nan_a = pd.isna(val_a_cmp)
                mask_nan_b = pd.isna(val_b_cmp)
                if mask_nan_a.equals(mask_nan_b):
                    non_nan_a = val_a_cmp[~mask_nan_a]
                    non_nan_b = val_b_cmp[~mask_nan_b]
                    if len(non_nan_a) == 0 or np.allclose(
                        non_nan_a, non_nan_b, rtol=self.rtol, atol=self.atol
                    ):
                        # Values equal, return any structural diffs
                        if children:
                            return CompoundDiff(source_type="series", children=children, truncated=False)
                        return None
            else:
                # For non-float types, use equals
                if val_a_cmp.equals(val_b_cmp):
                    # Values equal, return any structural diffs
                    if children:
                        return CompoundDiff(source_type="series", children=children, truncated=False)
                    return None
        except Exception:
            pass  # Fall through to element-by-element comparison

        # Series are different - collect up to max_diffs_per_container differences
        # Use vectorized operations to find diff indices, then only iterate over actual diffs

        try:
            if pd.api.types.is_float_dtype(val_a_cmp.dtype):
                # OPTIMIZED: Vectorized diff detection for float dtypes
                # Extract numpy arrays for fast access
                arr_a = val_a_cmp.values
                arr_b = val_b_cmp.values
                arr_a_orig = val_a.values  # Original values for output
                arr_b_orig = val_b.values
                index = val_a_cmp.index

                # Vectorized NaN detection
                nan_a = np.isnan(arr_a)
                nan_b = np.isnan(arr_b)

                # Find all difference indices at once:
                # 1. NaN position mismatches (one is NaN, other is not)
                nan_mismatch = nan_a != nan_b
                # 2. Value mismatches (both non-NaN but values differ beyond tolerance)
                both_valid = ~nan_a & ~nan_b
                value_mismatch = both_valid & ~np.isclose(
                    arr_a, arr_b, rtol=self.rtol, atol=self.atol
                )

                # Combined: all differences
                all_diffs = nan_mismatch | value_mismatch
                diff_indices = np.where(all_diffs)[0]

                # Check if truncation needed
                truncated = len(diff_indices) > self.max_diffs_per_container

                # Only iterate over actual differences (up to max_diffs)
                for i in diff_indices[: self.max_diffs_per_container]:
                    idx_label = index[i]
                    v1, v2 = arr_a_orig[i], arr_b_orig[i]

                    if nan_mismatch[i]:
                        children[f"[{repr(idx_label)}]"] = ValueComparison(
                            status="different",
                            value1=v1,
                            value2=v2,
                            message=f"Series NaN positions mismatch at {path}[{repr(idx_label)}]: is_nan={nan_a[i]} vs is_nan={nan_b[i]}",
                        )
                    else:
                        children[f"[{repr(idx_label)}]"] = ValueComparison(
                            status="different",
                            value1=v1,
                            value2=v2,
                            message=f"Series values mismatch at {path}[{repr(idx_label)}]: {v1} vs {v2}",
                        )
            else:
                # For non-float dtypes, use vectorized equality check first
                # then only call _compare_values on different elements
                arr_a = val_a_cmp.values
                arr_b = val_b_cmp.values
                index = val_a_cmp.index

                # Try vectorized comparison first for simple types
                try:
                    # For object dtype or other types, try pandas not-equal
                    not_equal = val_a_cmp != val_b_cmp
                    # Handle case where comparison returns non-boolean (e.g., nested objects)
                    if hasattr(not_equal, 'values'):
                        diff_mask = not_equal.values
                    else:
                        diff_mask = np.array([True] * len(arr_a))  # Fall back to check all
                except (TypeError, ValueError):
                    # Comparison failed, check all elements
                    diff_mask = np.array([True] * len(arr_a))

                diff_indices = np.where(diff_mask)[0]
                diff_count = 0

                # FAST PATH for homogeneous immutable object columns (strings, ints, etc.)
                # Skip _compare_values dispatch overhead and record diffs directly
                if val_a_cmp.dtype == object and len(diff_indices) > 0:
                    kind = infer_dtype(val_a_cmp, skipna=True)
                    if kind in _IMMUTABLE_INFERRED_KINDS:
                        with timer(key="diff:series_immutable_fast_path", message=f"[diff] Series immutable fast path ({kind}, {len(diff_indices)} diffs)"):
                            # Direct recording without dispatch - much faster for large diffs
                            truncated = len(diff_indices) > self.max_diffs_per_container
                            for i in diff_indices[:self.max_diffs_per_container]:
                                idx_label = index[i]
                                v1, v2 = arr_a[i], arr_b[i]

                                # Check if both values are floats - apply tolerance
                                if isinstance(v1, (float, np.floating)) and isinstance(v2, (float, np.floating)):
                                    # Handle NaN equality
                                    if math.isnan(v1) and math.isnan(v2):
                                        continue  # Both NaN - consider equal
                                    # Apply tolerance comparison
                                    if math.isclose(v1, v2, rel_tol=self.rtol, abs_tol=self.atol):
                                        continue  # Within tolerance - consider equal

                                children[f"[{repr(idx_label)}]"] = ValueComparison(
                                    status="different",
                                    value1=v1,
                                    value2=v2,
                                    message=f"Series value mismatch at {path}[{repr(idx_label)}]: {v1!r} vs {v2!r}",
                                )
                            # Skip the normal loop
                            if children:
                                return CompoundDiff(source_type="series", children=children, truncated=truncated)
                            return None

                # Only iterate over potentially different elements
                for i in diff_indices:
                    idx_label = index[i]
                    elem_diff = self._compare_values(
                        arr_a[i], arr_b[i], f"{path}[{repr(idx_label)}]"
                    )
                    if elem_diff:
                        children[f"[{repr(idx_label)}]"] = elem_diff
                        diff_count += 1
                        if diff_count >= self.max_diffs_per_container:
                            truncated = True
                            break
        except Exception as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Series comparison error at {path}: {str(e)}",
            )

        if children:
            return CompoundDiff(source_type="series", children=children, truncated=truncated)
        return None

    # ==========================================================================
    # FAST PATH HELPERS: Vectorized equality checks for DataFrames and Series
    # ==========================================================================

    def _fast_series_equal(self, s_a: pd.Series, s_b: pd.Series, path: str = "") -> bool:
        """
        Fast vectorized equality check for two Series.

        Assumes: same length, same index (caller should verify).
        Uses exact equality (not tolerance-based) for speed.

        Returns:
            True if series values are exactly equal (NaN == NaN for floats).

        Raises:
            Exception if comparison fails (caller should catch and fall back).
        """
        if s_a.dtype != s_b.dtype:
            # Different dtypes - can't use fast path
            return False

        # Identity check: if underlying arrays are the same object, they're equal
        # This is nearly instant (0.002ms) when arrays share memory
        try:
            arr_a = s_a.values
            arr_b = s_b.values
            if arr_a is arr_b:
                if _PROFILE_DIFF:
                    log(f"[diff profile] {path}: Fast path identity (same object)")
                return True
            if np.shares_memory(arr_a, arr_b):
                if _PROFILE_DIFF:
                    log(f"[diff profile] {path}: Fast path identity (shared memory)")
                return True
        except (TypeError, ValueError):
            pass  # Some array types don't support shares_memory

        # For float types, use numpy's array_equal with NaN handling
        # This is ~3x faster than mask+allclose for float columns
        if pd.api.types.is_float_dtype(s_a.dtype):
            return np.array_equal(s_a.to_numpy(), s_b.to_numpy(), equal_nan=True)

        # For all other types (int, string, object, etc.), pandas equals is faster
        # because it avoids the to_numpy() conversion overhead
        return s_a.equals(s_b)

    def _fast_dataframe_equal(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, path: str = ""
    ) -> bool:
        """
        Fast vectorized equality check for two DataFrames.

        Assumes: same shape, same columns, same index (caller should verify).
        Uses tolerance (rtol, atol) for float columns.

        Args:
            df_a: First DataFrame
            df_b: Second DataFrame
            path: Variable path for profiling output

        Returns:
            True if all column values are equal (within tolerance for floats).

        Raises:
            Exception if comparison fails (caller should catch and fall back).
        """
        # Collect column timings if profiling is enabled
        column_timings: List[Tuple[float, str, str]] = []

        # Use iloc to avoid issues with MultiIndex columns
        if _PROFILE_DIFF:
            with timer(key="diff:fast_dataframe_equal", message=f"[diff] Fast path DataFrame equal ({len(df_a.columns)} cols compared)"):
                # Profiling enabled: collect timing, pass path names
                for i in range(len(df_a.columns)):
                    col_name = str(df_a.columns[i])
                    col_dtype = str(df_a.iloc[:, i].dtype)
                    col_start = time.perf_counter()
                    path_col = f"{path}['{col_name}']" if path else col_name
                    is_equal = self._fast_series_equal(df_a.iloc[:, i], df_b.iloc[:, i], path_col)
                    col_elapsed = time.perf_counter() - col_start
                    column_timings.append((col_elapsed, col_name, col_dtype))
                    if not is_equal:
                        # Log timings collected so far before returning
                        if column_timings:
                            self._log_column_timings(path, column_timings)
                        return False

                # Log all column timings
                if column_timings:
                    self._log_column_timings(path, column_timings)
                    log(f"[diff profile] Fast path: DataFrame {path} equal ({len(df_a.columns)} cols compared)")
        else:
            # No profiling: just compare columns
            for i in range(len(df_a.columns)):
                if not self._fast_series_equal(df_a.iloc[:, i], df_b.iloc[:, i]):
                    return False

        return True

    # ==========================================================================

    def _compare_dataframe(
        self, val_a: pd.DataFrame, val_b: pd.DataFrame, path: str
    ) -> Optional[DiffNode]:
        """Compare pandas DataFrames, collecting up to max_diffs_per_structure differences.

        This method NEVER returns early - it always compares all columns that exist in both
        DataFrames, accumulating all differences found.

        Structural tracking: When structural_mode is WARN or ENFORCE and structural reads
        were made on this DataFrame, changes to structure (columns, rows) are reported.
        """
        # ======================================================================
        # FAST PATH: If DataFrames are structurally identical, try vectorized
        # equality check before column-by-column comparison.
        # ======================================================================

        if (val_a.shape == val_b.shape and
            val_a.columns.equals(val_b.columns) and
            val_a.index.equals(val_b.index)):
            try:
                # Try fast equality check for all columns at once
                if self._fast_dataframe_equal(val_a, val_b, path):
                    return None  # DataFrames are equal, no differences
            except Exception:
                pass  # Fall through to detailed comparison
        # ======================================================================

        children: Dict[str, DiffNode] = {}
        truncated = False
        total_diff_count = 0

        # Determine which columns to compare
        cols_a = set(val_a.columns)
        cols_b = set(val_b.columns)

        # Check for structural changes (row count)
        if len(val_a) != len(val_b):
            structural_diff = self._check_structural_change(
                path, 'rows',
                f"Row count changed from {len(val_a)} to {len(val_b)}"
            )
            if structural_diff:
                children["_structural_rows"] = structural_diff
                total_diff_count += 1

        # Check for structural changes (column additions)
        added_cols = cols_b - cols_a
        if added_cols:
            structural_diff = self._check_structural_change(
                path, 'columns',
                f"Columns added: {sorted(added_cols)}"
            )
            if structural_diff:
                children["_structural_columns"] = structural_diff
                total_diff_count += 1

        if self.use_leq:
            # Check if we have column-level RBW info for this variable
            if path in self.column_rbw:
                rbw_cols = self.column_rbw[path]

                if not rbw_cols:
                    # Empty set means write-only - skip DataFrame comparison entirely
                    # (only report structural changes if structural tracking is enabled)
                    if children:
                        return CompoundDiff(source_type="dataframe", children=children, truncated=False)
                    return None
                else:
                    # Track missing RBW columns as differences (but don't return early!)
                    missing_in_a = rbw_cols - cols_a
                    for col in sorted(missing_in_a):
                        children[f"['{col}']"] = ValueComparison(
                            status="different",
                            value1=None,
                            value2=None,
                            message=f"DataFrame column '{col}' missing in pre-state at {path}",
                        )
                        total_diff_count += 1

                    missing_in_b = rbw_cols - cols_b
                    for col in sorted(missing_in_b):
                        children[f"['{col}']"] = ValueComparison(
                            status="different",
                            value1=None,
                            value2=None,
                            message=f"DataFrame column '{col}' deleted in post-state at {path}",
                        )
                        total_diff_count += 1

                    # Compare RBW columns that exist in both
                    cols_to_compare = rbw_cols & cols_a & cols_b
            else:
                # No column-level RBW info - in leq mode, compare all of val_a's columns
                # Track columns missing in b
                missing_cols = cols_a - cols_b
                for col in sorted(missing_cols):
                    children[f"['{col}']"] = ValueComparison(
                        status="different",
                        value1=_get_column_as_series(val_a, col) if col in val_a.columns else None,
                        value2=None,
                        message=f"DataFrame column '{col}' missing in second DataFrame at {path}",
                    )
                    total_diff_count += 1

                # Compare columns that exist in both
                cols_to_compare = cols_a & cols_b
        else:
            # Standard mode: track column differences
            only_in_a = cols_a - cols_b
            only_in_b = cols_b - cols_a

            for col in sorted(only_in_a):
                children[f"['{col}']"] = ValueComparison(
                    status="different",
                    value1=_get_column_as_series(val_a, col),
                    value2=None,
                    message=f"Column '{col}' only in first DataFrame",
                )
                total_diff_count += 1

            for col in sorted(only_in_b):
                children[f"['{col}']"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=_get_column_as_series(val_b, col),
                    message=f"Column '{col}' only in second DataFrame",
                )
                total_diff_count += 1

            # Compare common columns
            cols_to_compare = cols_a & cols_b

        # Check index mismatch (record as difference but continue comparing columns)
        if not val_a.index.equals(val_b.index):
            children["_index"] = ValueComparison(
                status="different",
                value1=val_a.index,
                value2=val_b.index,
                message=f"DataFrame index mismatch at {path}",
            )
            total_diff_count += 1
            # We still try to compare columns if they have the same length
            if len(val_a) != len(val_b):
                # Can't compare columns with different row counts
                if children:
                    return CompoundDiff(source_type="dataframe", children=children, truncated=False)
                return None

        # Check dtype compatibility for each column (record mismatches but continue)
        cols_with_dtype_issues = set()
        for col in cols_to_compare:
            col_a = _get_column_as_series(val_a, col)
            col_b = _get_column_as_series(val_b, col)
            if not are_compatible_dtypes(col_a, col_b):
                children[f"['{col}']._dtype"] = ValueComparison(
                    status="different",
                    value1=col_a.dtype,
                    value2=col_b.dtype,
                    message=f"DataFrame column '{col}' dtype mismatch at {path}: {col_a.dtype} vs {col_b.dtype}",
                )
                cols_with_dtype_issues.add(col)
                total_diff_count += 1

        # Only compare columns that don't have dtype issues
        cols_to_compare_values = cols_to_compare - cols_with_dtype_issues

        # Compare each column - _compare_series handles dtype casting internally
        # Track column timings for profiling
        column_timings: List[Tuple[float, str, str]] = []  # (elapsed, col_name, dtype)

        for col in sorted(cols_to_compare_values):
            if _PROFILE_DIFF:
                col_start = time.perf_counter()

            col_a = _get_column_as_series(val_a, col)
            col_b = _get_column_as_series(val_b, col)
            col_diff = self._compare_series(
                col_a, col_b, f"{path}['{col}']"
            )

            if _PROFILE_DIFF:
                col_elapsed = time.perf_counter() - col_start
                col_dtype = str(col_a.dtype)
                column_timings.append((col_elapsed, str(col), col_dtype))

            if col_diff:
                # Keep nested structure - don't flatten series diffs
                children[f"['{col}']"] = col_diff

                # Count diffs for truncation limit
                if isinstance(col_diff, CompoundDiff):
                    # Count non-metadata keys in the nested children
                    total_diff_count += len(col_diff.children)
                else:
                    total_diff_count += 1

                # Check if we've hit the limit
                if total_diff_count >= self.max_diffs_per_structure:
                    truncated = True
                    # Log column timings before returning
                    if _PROFILE_DIFF and column_timings:
                        self._log_column_timings(path, column_timings)
                    return CompoundDiff(source_type="dataframe", children=children, truncated=truncated)

        # Log column timings for this DataFrame
        if _PROFILE_DIFF and column_timings:
            self._log_column_timings(path, column_timings)

        if children:
            return CompoundDiff(source_type="dataframe", children=children, truncated=truncated)
        return None

    def _compare_index(
        self, val_a: pd.Index, val_b: pd.Index, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas Index objects using .equals() method."""
        if not val_a.equals(val_b):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Index mismatch at {path}: {list(val_a)} vs {list(val_b)}",
            )
        return None

    def _compare_timestamp(
        self, val_a: pd.Timestamp, val_b: pd.Timestamp, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas Timestamp objects."""
        # Handle NaT (Not a Time) - similar to NaN, NaT != NaT is True
        is_nat_a = pd.isna(val_a)
        is_nat_b = pd.isna(val_b)

        if is_nat_a and is_nat_b:
            return None  # Both NaT, considered equal
        if is_nat_a or is_nat_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Timestamp mismatch at {path}: {val_a} vs {val_b} (one is NaT)",
            )

        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Timestamp mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_timedelta(
        self, val_a: pd.Timedelta, val_b: pd.Timedelta, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas Timedelta objects."""
        # Handle NaT (Not a Time) - similar to NaN, NaT != NaT is True
        is_nat_a = pd.isna(val_a)
        is_nat_b = pd.isna(val_b)

        if is_nat_a and is_nat_b:
            return None  # Both NaT, considered equal
        if is_nat_a or is_nat_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Timedelta mismatch at {path}: {val_a} vs {val_b} (one is NaT)",
            )

        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Timedelta mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_datetime64(
        self, val_a: np.datetime64, val_b: np.datetime64, path: str
    ) -> Optional[ValueComparison]:
        """Compare numpy datetime64 objects."""
        # Handle NaT (Not a Time) - similar to NaN, NaT != NaT is True
        is_nat_a = np.isnat(val_a)
        is_nat_b = np.isnat(val_b)

        if is_nat_a and is_nat_b:
            return None  # Both NaT, considered equal
        if is_nat_a or is_nat_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"datetime64 mismatch at {path}: {val_a} vs {val_b} (one is NaT)",
            )

        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"datetime64 mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_timedelta64(
        self, val_a: np.timedelta64, val_b: np.timedelta64, path: str
    ) -> Optional[ValueComparison]:
        """Compare numpy timedelta64 objects."""
        # Handle NaT (Not a Time) - similar to NaN, NaT != NaT is True
        is_nat_a = np.isnat(val_a)
        is_nat_b = np.isnat(val_b)

        if is_nat_a and is_nat_b:
            return None  # Both NaT, considered equal
        if is_nat_a or is_nat_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"timedelta64 mismatch at {path}: {val_a} vs {val_b} (one is NaT)",
            )

        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"timedelta64 mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_catboost_pool(
        self, val_a: "CatBoostPool", val_b: "CatBoostPool", path: str
    ) -> Optional[DiffNode]:
        """
        Compare CatBoost Pool objects by their content.

        Pools are compared by:
        - Shape (num_row, num_col)
        - Quantization state (is_quantized)
        - Feature names (get_feature_names)
        - Categorical feature indices (get_cat_feature_indices)
        - Features (get_features)
        - Labels (get_label)
        - Weights (get_weight)

        Returns None if pools are equal, CompoundDiff with differences otherwise.
        """
        children: Dict[str, DiffNode] = {}

        # Fast path: shape check first (cheapest)
        num_row_a, num_row_b = val_a.num_row(), val_b.num_row()
        num_col_a, num_col_b = val_a.num_col(), val_b.num_col()

        if num_row_a != num_row_b:
            children["_num_row"] = ValueComparison(
                status="different",
                value1=num_row_a,
                value2=num_row_b,
                message=f"Pool row count mismatch at {path}: {num_row_a} vs {num_row_b}",
            )
            # Can't compare data with different row counts
            return CompoundDiff(source_type="catboost_pool", children=children, truncated=False)

        if num_col_a != num_col_b:
            children["_num_col"] = ValueComparison(
                status="different",
                value1=num_col_a,
                value2=num_col_b,
                message=f"Pool column count mismatch at {path}: {num_col_a} vs {num_col_b}",
            )
            # Can't compare features with different column counts
            return CompoundDiff(source_type="catboost_pool", children=children, truncated=False)

        # Metadata checks (cheap)
        quant_a, quant_b = val_a.is_quantized(), val_b.is_quantized()
        if quant_a != quant_b:
            children["_quantized"] = ValueComparison(
                status="different",
                value1=quant_a,
                value2=quant_b,
                message=f"Pool quantization state mismatch at {path}: {quant_a} vs {quant_b}",
            )

        feat_names_a, feat_names_b = val_a.get_feature_names(), val_b.get_feature_names()
        if feat_names_a != feat_names_b:
            children["_feature_names"] = ValueComparison(
                status="different",
                value1=feat_names_a,
                value2=feat_names_b,
                message=f"Pool feature names mismatch at {path}",
            )

        cat_feat_a, cat_feat_b = val_a.get_cat_feature_indices(), val_b.get_cat_feature_indices()
        if cat_feat_a != cat_feat_b:
            children["_cat_features"] = ValueComparison(
                status="different",
                value1=cat_feat_a,
                value2=cat_feat_b,
                message=f"Pool categorical feature indices mismatch at {path}",
            )

        # Data checks (expensive - extract arrays)
        # Only compare features if shape matches (already checked above)
        # Note: Quantized pools don't support get_features(), so skip if either is quantized
        if num_row_a > 0 and num_col_a > 0 and not quant_a and not quant_b:
            try:
                features_a = np.array(val_a.get_features())
                features_b = np.array(val_b.get_features())
                feat_diff = self._compare_ndarray(features_a, features_b, f"{path}._features")
                if feat_diff:
                    children["_features"] = feat_diff
            except Exception:
                # Can't extract features (e.g., quantized pool)
                pass

        # Compare labels if present
        labels_a = val_a.get_label()
        labels_b = val_b.get_label()
        if labels_a is not None or labels_b is not None:
            if labels_a is None:
                children["_labels"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=labels_b,
                    message=f"Pool labels mismatch at {path}: first has no labels",
                )
            elif labels_b is None:
                children["_labels"] = ValueComparison(
                    status="different",
                    value1=labels_a,
                    value2=None,
                    message=f"Pool labels mismatch at {path}: second has no labels",
                )
            else:
                labels_diff = self._compare_ndarray(
                    np.array(labels_a), np.array(labels_b), f"{path}._labels"
                )
                if labels_diff:
                    children["_labels"] = labels_diff

        # Compare weights if present
        weights_a = val_a.get_weight()
        weights_b = val_b.get_weight()
        if weights_a is not None or weights_b is not None:
            if weights_a is None:
                children["_weights"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=weights_b,
                    message=f"Pool weights mismatch at {path}: first has no weights",
                )
            elif weights_b is None:
                children["_weights"] = ValueComparison(
                    status="different",
                    value1=weights_a,
                    value2=None,
                    message=f"Pool weights mismatch at {path}: second has no weights",
                )
            else:
                weights_diff = self._compare_ndarray(
                    np.array(weights_a), np.array(weights_b), f"{path}._weights"
                )
                if weights_diff:
                    children["_weights"] = weights_diff

        if children:
            return CompoundDiff(source_type="catboost_pool", children=children, truncated=False)
        return None

    def _compare_keras_model(
        self, val_a, val_b, path: str
    ) -> Optional[DiffNode]:
        """
        Compare Keras model objects by their weights and configuration.

        Models are compared by:
        - Number of layers
        - Layer configurations (type, units, activation, etc.)
        - Model weights (layer by layer comparison)

        Returns None if models are equal, CompoundDiff with differences otherwise.
        """
        children: Dict[str, DiffNode] = {}

        # Compare number of layers
        layers_a = val_a.layers
        layers_b = val_b.layers

        if len(layers_a) != len(layers_b):
            children["_num_layers"] = ValueComparison(
                status="different",
                value1=len(layers_a),
                value2=len(layers_b),
                message=f"Model layer count mismatch at {path}: {len(layers_a)} vs {len(layers_b)}",
            )
            # Can't meaningfully compare weights with different architectures
            return CompoundDiff(source_type="keras_model", children=children, truncated=False)

        # Compare layer configurations
        for i, (layer_a, layer_b) in enumerate(zip(layers_a, layers_b)):
            layer_path = f"{path}.layer[{i}]"

            # Compare layer types
            type_a = type(layer_a).__name__
            type_b = type(layer_b).__name__
            if type_a != type_b:
                children[f"_layer_{i}_type"] = ValueComparison(
                    status="different",
                    value1=type_a,
                    value2=type_b,
                    message=f"Layer type mismatch at {layer_path}: {type_a} vs {type_b}",
                )
                continue

            # Compare layer configs (units, activation, etc.)
            try:
                config_a = layer_a.get_config()
                config_b = layer_b.get_config()

                # Exclude 'name' from config comparison as auto-generated names may differ
                config_a_filtered = {k: v for k, v in config_a.items() if k != 'name'}
                config_b_filtered = {k: v for k, v in config_b.items() if k != 'name'}

                if config_a_filtered != config_b_filtered:
                    # Find specific differences
                    all_keys = set(config_a_filtered.keys()) | set(config_b_filtered.keys())
                    for key in all_keys:
                        val_cfg_a = config_a_filtered.get(key)
                        val_cfg_b = config_b_filtered.get(key)
                        if val_cfg_a != val_cfg_b:
                            children[f"_layer_{i}_config_{key}"] = ValueComparison(
                                status="different",
                                value1=val_cfg_a,
                                value2=val_cfg_b,
                                message=f"Layer config mismatch at {layer_path}.{key}",
                            )
            except Exception:
                # Some layers may not support get_config
                pass

        # Compare weights - this is the critical comparison for trained models
        try:
            weights_a = val_a.get_weights()
            weights_b = val_b.get_weights()

            if len(weights_a) != len(weights_b):
                children["_num_weight_arrays"] = ValueComparison(
                    status="different",
                    value1=len(weights_a),
                    value2=len(weights_b),
                    message=f"Model weight array count mismatch at {path}: {len(weights_a)} vs {len(weights_b)}",
                )
            else:
                # Compare each weight array
                for i, (w_a, w_b) in enumerate(zip(weights_a, weights_b)):
                    weight_path = f"{path}._weights[{i}]"
                    weight_diff = self._compare_ndarray(
                        np.asarray(w_a), np.asarray(w_b), weight_path
                    )
                    if weight_diff:
                        children[f"_weights_{i}"] = weight_diff
        except Exception as e:
            # Handle edge cases where weights can't be extracted
            children["_weights_error"] = ValueComparison(
                status="different",
                value1=str(e),
                value2=None,
                message=f"Could not compare weights at {path}: {e}",
            )

        if children:
            return CompoundDiff(source_type="keras_model", children=children, truncated=False)
        return None

    def _compare_pytorch_model(
        self, val_a, val_b, path: str
    ) -> Optional[DiffNode]:
        """
        Compare PyTorch nn.Module objects by their state_dict and structure.

        Models are compared by:
        - Module structure (named_modules keys)
        - State dict (parameters and buffers)
        - Training mode

        Returns None if models are equal, CompoundDiff with differences otherwise.
        """
        import torch

        children: Dict[str, DiffNode] = {}

        # Compare module structure
        modules_a = dict(val_a.named_modules())
        modules_b = dict(val_b.named_modules())

        keys_a = set(modules_a.keys())
        keys_b = set(modules_b.keys())

        if keys_a != keys_b:
            only_in_a = keys_a - keys_b
            only_in_b = keys_b - keys_a
            if only_in_a or only_in_b:
                children["_structure"] = ValueComparison(
                    status="different",
                    value1=sorted(keys_a),
                    value2=sorted(keys_b),
                    message=f"Module structure mismatch at {path}: "
                            f"only in first: {sorted(only_in_a)}, only in second: {sorted(only_in_b)}",
                )

        # Compare state_dicts (parameters and buffers)
        try:
            sd_a = val_a.state_dict()
            sd_b = val_b.state_dict()

            keys_sd_a = set(sd_a.keys())
            keys_sd_b = set(sd_b.keys())

            # Check for missing keys
            only_in_sd_a = keys_sd_a - keys_sd_b
            only_in_sd_b = keys_sd_b - keys_sd_a

            for key in only_in_sd_a:
                children[f"_param_{key}"] = ValueComparison(
                    status="different",
                    value1=f"<tensor shape={tuple(sd_a[key].shape)}>",
                    value2="<missing>",
                    message=f"Parameter missing in second model at {path}.{key}",
                )

            for key in only_in_sd_b:
                children[f"_param_{key}"] = ValueComparison(
                    status="different",
                    value1="<missing>",
                    value2=f"<tensor shape={tuple(sd_b[key].shape)}>",
                    message=f"Parameter missing in first model at {path}.{key}",
                )

            # Compare common keys
            for key in keys_sd_a & keys_sd_b:
                t_a, t_b = sd_a[key], sd_b[key]

                # Shape mismatch
                if t_a.shape != t_b.shape:
                    children[f"_param_{key}"] = ValueComparison(
                        status="different",
                        value1=f"<tensor shape={tuple(t_a.shape)}>",
                        value2=f"<tensor shape={tuple(t_b.shape)}>",
                        message=f"Parameter shape mismatch at {path}.{key}: {tuple(t_a.shape)} vs {tuple(t_b.shape)}",
                    )
                    continue

                # Value comparison using torch.allclose for numerical stability
                # Convert to float for comparison (handles int tensors)
                try:
                    if not torch.allclose(t_a.float(), t_b.float(), rtol=1e-5, atol=1e-8):
                        # Find max difference for debugging
                        max_diff = (t_a.float() - t_b.float()).abs().max().item()
                        children[f"_param_{key}"] = ValueComparison(
                            status="different",
                            value1=f"<tensor shape={tuple(t_a.shape)}>",
                            value2=f"<tensor shape={tuple(t_b.shape)}>",
                            message=f"Parameter values differ at {path}.{key}: max_diff={max_diff:.6e}",
                        )
                except Exception as e:
                    children[f"_param_{key}"] = ValueComparison(
                        status="different",
                        value1=str(t_a),
                        value2=str(t_b),
                        message=f"Could not compare parameter at {path}.{key}: {e}",
                    )

        except Exception as e:
            children["_state_dict_error"] = ValueComparison(
                status="different",
                value1=str(e),
                value2=None,
                message=f"Could not compare state_dict at {path}: {e}",
            )

        # Compare training mode
        if val_a.training != val_b.training:
            children["_training"] = ValueComparison(
                status="different",
                value1=val_a.training,
                value2=val_b.training,
                message=f"Training mode mismatch at {path}: {val_a.training} vs {val_b.training}",
            )

        if children:
            return CompoundDiff(source_type="pytorch_model", children=children, truncated=False)
        return None

    def _compare_lightgbm_model(
        self, val_a, val_b, path: str
    ) -> Optional[DiffNode]:
        """
        Compare LightGBM model objects using their model string representation.

        For fitted models, the booster's model_to_string() provides a complete
        representation of the tree ensemble. Comparing these strings is much
        faster than traversing the Python object graph.

        Models are compared by:
        - Fitted status (both must be fitted or both unfitted)
        - Model string (for fitted models)
        - Parameters (for unfitted models)

        Returns None if models are equal, CompoundDiff with differences otherwise.
        """
        children: Dict[str, DiffNode] = {}

        # Check type match
        if type(val_a) != type(val_b):
            return ValueComparison(
                status="different",
                value1=type(val_a).__name__,
                value2=type(val_b).__name__,
                message=f"LightGBM model type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}",
            )

        # Check fitted status
        a_fitted = hasattr(val_a, 'booster_') and val_a.booster_ is not None
        b_fitted = hasattr(val_b, 'booster_') and val_b.booster_ is not None

        if a_fitted != b_fitted:
            return ValueComparison(
                status="different",
                value1="fitted" if a_fitted else "unfitted",
                value2="fitted" if b_fitted else "unfitted",
                message=f"LightGBM fitted status mismatch at {path}: {'fitted' if a_fitted else 'unfitted'} vs {'fitted' if b_fitted else 'unfitted'}",
            )

        if not a_fitted:
            # Both unfitted - compare parameters only
            params_a = val_a.get_params()
            params_b = val_b.get_params()
            if params_a != params_b:
                children["_params"] = ValueComparison(
                    status="different",
                    value1=str(params_a),
                    value2=str(params_b),
                    message=f"LightGBM parameters mismatch at {path}",
                )
        else:
            # Both fitted - compare model strings
            str_a = val_a.booster_.model_to_string()
            str_b = val_b.booster_.model_to_string()

            if str_a != str_b:
                # Models are different - don't include full strings in diff
                # (they can be very large)
                children["_booster"] = ValueComparison(
                    status="different",
                    value1=f"<LightGBM model with {val_a.booster_.num_trees()} trees>",
                    value2=f"<LightGBM model with {val_b.booster_.num_trees()} trees>",
                    message=f"LightGBM model trees differ at {path}",
                )

        if children:
            return CompoundDiff(source_type="lightgbm_model", children=children, truncated=False)
        return None

    def _compare_groupby(self, val_a, val_b, path: str) -> Optional[ValueComparison]:
        """Compare pandas GroupBy objects (TODO: implement full diff collection)."""
        result = self._compare_groupby_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(
                status="different", value1=val_a, value2=val_b, message=result
            )
        return None

    def _compare_groupby_legacy(self, val_a, val_b, path: str) -> str:
        """
        Compare DataFrameGroupBy or SeriesGroupBy objects by their semantic properties.

        Excludes internal cache fields (_cache, _grouper._cache) that can differ
        between otherwise equivalent groupby objects.

        Compares:
        - The underlying data object (obj)
        - The grouper configuration (keys, sort, dropna)
        - Selection state
        """
        # Compare the underlying object (DataFrame or Series)
        if not hasattr(val_a, "obj") or not hasattr(val_b, "obj"):
            return f"GroupBy structure mismatch at {path}: missing obj attribute"

        obj_diff = self._compare_values(val_a.obj, val_b.obj, f"{path}.obj")
        if obj_diff:
            # _compare_values returns DiffNode now, extract message for legacy
            if isinstance(obj_diff, ValueComparison):
                return obj_diff.message
            elif isinstance(obj_diff, dict):
                # Compound diff - just indicate there's a difference
                return f"GroupBy data mismatch at {path}.obj"
            return obj_diff

        # Compare the grouper (excluding cache)
        if not hasattr(val_a, "_grouper") or not hasattr(val_b, "_grouper"):
            return f"GroupBy structure mismatch at {path}: missing _grouper attribute"

        grouper_diff = self._compare_grouper(
            val_a._grouper, val_b._grouper, f"{path}._grouper"
        )
        if grouper_diff:
            # _compare_grouper returns ValueComparison now, but we need string for legacy
            if isinstance(grouper_diff, ValueComparison):
                return grouper_diff.message
            return grouper_diff

        # Compare selection (which columns are selected)
        if hasattr(val_a, "_selection") and hasattr(val_b, "_selection"):
            if val_a._selection != val_b._selection:
                return f"GroupBy selection mismatch at {path}: {val_a._selection} vs {val_b._selection}"

        return ""

    def _compare_grouper(
        self, val_a: BaseGrouper, val_b: BaseGrouper, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas BaseGrouper objects (legacy wrapper)."""
        result = self._compare_grouper_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(
                status="different", value1=val_a, value2=val_b, message=result
            )
        return None

    def _compare_grouper_legacy(
        self, val_a: BaseGrouper, val_b: BaseGrouper, path: str
    ) -> str:
        """
        Compare BaseGrouper objects, excluding their internal cache.

        Compares the semantic properties that define the grouping:
        - groupings (keys)
        - sort flag
        - dropna flag
        - axis
        """
        # Compare axis (typically a pandas Index)
        if hasattr(val_a, "axis") and hasattr(val_b, "axis"):
            # Use pandas Index.equals() for proper comparison
            axes_equal = False
            if isinstance(val_a.axis, pd.Index) and isinstance(val_b.axis, pd.Index):
                axes_equal = val_a.axis.equals(val_b.axis)
            elif isinstance(val_a.axis, np.ndarray) and isinstance(
                val_b.axis, np.ndarray
            ):
                axes_equal = np.array_equal(val_a.axis, val_b.axis)
            else:
                axes_equal = val_a.axis == val_b.axis

            if not axes_equal:
                return f"Grouper axis mismatch at {path}: {val_a.axis} vs {val_b.axis}"

        # Compare sort flag
        if hasattr(val_a, "_sort") and hasattr(val_b, "_sort"):
            if val_a._sort != val_b._sort:
                return (
                    f"Grouper sort mismatch at {path}: {val_a._sort} vs {val_b._sort}"
                )

        # Compare dropna flag
        if hasattr(val_a, "dropna") and hasattr(val_b, "dropna"):
            if val_a.dropna != val_b.dropna:
                return f"Grouper dropna mismatch at {path}: {val_a.dropna} vs {val_b.dropna}"

        # Compare groupings (the actual grouping keys)
        if hasattr(val_a, "_groupings") and hasattr(val_b, "_groupings"):
            groupings_a = val_a._groupings
            groupings_b = val_b._groupings

            if len(groupings_a) != len(groupings_b):
                return f"Grouper groupings count mismatch at {path}: {len(groupings_a)} vs {len(groupings_b)}"

            for i, (grp_a, grp_b) in enumerate(zip(groupings_a, groupings_b)):
                # Compare key/name
                if hasattr(grp_a, "name") and hasattr(grp_b, "name"):
                    if grp_a.name != grp_b.name:
                        return f"Grouping name mismatch at {path}._groupings[{i}]: {grp_a.name} vs {grp_b.name}"

                # Compare key object if available
                if hasattr(grp_a, "key") and hasattr(grp_b, "key"):
                    if grp_a.key != grp_b.key:
                        return f"Grouping key mismatch at {path}._groupings[{i}]: {grp_a.key} vs {grp_b.key}"

        return ""

    def _compare_list(self, val_a: list, val_b: list, path: str) -> Optional[DiffNode]:
        """Compare lists, recording length mismatch but comparing common elements."""
        # Fast path: check if both are from the same cached primitive list
        if are_primitive_containers_equal(val_a, val_b):
            return None

        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record length mismatch but continue comparing common elements
        if len(val_a) != len(val_b):
            children["_length"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"List length mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Compare common elements, collecting differences
        min_len = min(len(val_a), len(val_b))
        diff_count = 0
        for i in range(min_len):
            item_a, item_b = val_a[i], val_b[i]
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                children[f"[{i}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    break

        if children:
            return CompoundDiff(source_type="list", children=children, truncated=truncated)
        return None

    def _compare_tuple(
        self, val_a: tuple, val_b: tuple, path: str
    ) -> Optional[DiffNode]:
        """Compare tuples, recording length mismatch but comparing common elements."""
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record length mismatch but continue comparing common elements
        if len(val_a) != len(val_b):
            children["_length"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"Tuple length mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Compare common elements, collecting differences
        min_len = min(len(val_a), len(val_b))
        diff_count = 0
        for i in range(min_len):
            item_a, item_b = val_a[i], val_b[i]
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                children[f"[{i}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    break

        if children:
            return CompoundDiff(source_type="tuple", children=children, truncated=truncated)
        return None

    def _compare_set(
        self, val_a: set, val_b: set, path: str
    ) -> Optional[DiffNode]:
        """
        Compare sets by finding matching elements and comparing them recursively.
        This properly handles pointer structure within sets.
        Records size mismatch but continues to find unmatched elements.
        """
        # Fast path: check if both are from the same cached primitive set
        if are_primitive_containers_equal(val_a, val_b):
            return None

        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record size mismatch but continue
        if len(val_a) != len(val_b):
            children["_size"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"Set size mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Convert to lists for matching
        list_a = list(val_a)
        list_b = list(val_b)

        # Try to find a matching between elements
        used_b = set()
        unmatched_a = []
        diff_count = len(children)

        for i, item_a in enumerate(list_a):
            found_match = False

            for j, item_b in enumerate(list_b):
                if j in used_b:
                    continue

                # Try to match item_a with item_b
                diff = self._compare_values(item_a, item_b, f"{path}{{element {i}}}")
                if not diff:
                    # Found a match
                    used_b.add(j)
                    found_match = True
                    break

            if not found_match:
                unmatched_a.append((i, item_a))

        # Record unmatched elements from set A
        for i, item_a in unmatched_a:
            children[f"{{unmatched_a_{i}}}"] = ValueComparison(
                status="different",
                value1=item_a,
                value2=None,
                message=f"Set element {repr(item_a)} from first set has no match in second set",
            )
            diff_count += 1
            if diff_count >= self.max_diffs_per_container:
                truncated = True
                return CompoundDiff(source_type="set", children=children, truncated=truncated)

        # Record unmatched elements from set B
        for j, item_b in enumerate(list_b):
            if j not in used_b:
                children[f"{{unmatched_b_{j}}}"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=item_b,
                    message=f"Set element {repr(item_b)} from second set has no match in first set",
                )
                diff_count += 1
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    return CompoundDiff(source_type="set", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="set", children=children, truncated=truncated)
        return None

    def _compare_frozenset(
        self, val_a: frozenset, val_b: frozenset, path: str
    ) -> Optional[DiffNode]:
        """
        Compare frozensets by finding matching elements and comparing them recursively.
        Records size mismatch but continues to find unmatched elements.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record size mismatch but continue
        if len(val_a) != len(val_b):
            children["_size"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"Frozenset size mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Convert to lists for matching
        list_a = list(val_a)
        list_b = list(val_b)

        # Try to find a matching between elements
        used_b = set()
        unmatched_a = []
        diff_count = len(children)

        for i, item_a in enumerate(list_a):
            found_match = False

            for j, item_b in enumerate(list_b):
                if j in used_b:
                    continue

                # Try to match item_a with item_b
                diff = self._compare_values(item_a, item_b, f"{path}{{element {i}}}")
                if not diff:
                    # Found a match
                    used_b.add(j)
                    found_match = True
                    break

            if not found_match:
                unmatched_a.append((i, item_a))

        # Record unmatched elements from frozenset A
        for i, item_a in unmatched_a:
            children[f"{{unmatched_a_{i}}}"] = ValueComparison(
                status="different",
                value1=item_a,
                value2=None,
                message=f"Frozenset element {repr(item_a)} from first set has no match in second set",
            )
            diff_count += 1
            if diff_count >= self.max_diffs_per_container:
                truncated = True
                return CompoundDiff(source_type="frozenset", children=children, truncated=truncated)

        # Record unmatched elements from frozenset B
        for j, item_b in enumerate(list_b):
            if j not in used_b:
                children[f"{{unmatched_b_{j}}}"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=item_b,
                    message=f"Frozenset element {repr(item_b)} from second set has no match in first set",
                )
                diff_count += 1
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    return CompoundDiff(source_type="frozenset", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="frozenset", children=children, truncated=truncated)
        return None

    def _compare_dict(self, val_a: dict, val_b: dict, path: str) -> Optional[DiffNode]:
        # Fast path: check if both are from the same cached primitive dict
        if are_primitive_containers_equal(val_a, val_b):
            return None

        children: Dict[str, DiffNode] = {}
        truncated = False
        keys_a = set(val_a.keys())
        keys_b = set(val_b.keys())

        # Check for key mismatches
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a

        for key in only_a:
            children[f"[{repr(key)}]"] = ValueComparison(
                status="different",
                value1=val_a[key],
                value2=None,
                message=f"Key {repr(key)} only in first dict",
            )

        for key in only_b:
            children[f"[{repr(key)}]"] = ValueComparison(
                status="different",
                value1=None,
                value2=val_b[key],
                message=f"Key {repr(key)} only in second dict",
            )

        # Compare common keys, collecting ALL differences
        common_keys = keys_a & keys_b
        diff_count = len(children)  # Count keys already different
        for key in sorted(common_keys, key=str):  # Sort for deterministic output
            diff = self._compare_values(val_a[key], val_b[key], f"{path}[{repr(key)}]")
            if diff:
                children[f"[{repr(key)}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    break

        if children:
            return CompoundDiff(source_type="dict", children=children, truncated=truncated)
        return None

    def _compare_object(self, val_a: Any, val_b: Any, path: str) -> Optional[DiffNode]:
        """
        Compare user-defined objects by recursively comparing their attributes.
        Handles both __dict__ and __slots__ based objects.
        """
        has_dict_a = hasattr(val_a, "__dict__")
        has_dict_b = hasattr(val_b, "__dict__")

        # Get __slots__ from the class, not the instance
        slots_a = getattr(type(val_a), "__slots__", None)
        slots_b = getattr(type(val_b), "__slots__", None)

        # If neither has __dict__ nor __slots__, try direct equality
        if not has_dict_a and not slots_a:
            try:
                if val_a != val_b:
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Object mismatch at {path}: {val_a} != {val_b}",
                    )
                return None
            except Exception:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Object comparison not supported at {path} (type: {type(val_a).__name__})",
                )

        children: Dict[str, DiffNode] = {}
        truncated = False
        diff_count = 0

        # Compare __dict__ attributes if available
        if has_dict_a and has_dict_b:
            dict_a = val_a.__dict__
            dict_b = val_b.__dict__

            keys_a = set(dict_a.keys())
            keys_b = set(dict_b.keys())

            # Check for attribute mismatches
            only_a = keys_a - keys_b
            only_b = keys_b - keys_a

            for key in only_a:
                children[f".{key}"] = ValueComparison(
                    status="different",
                    value1=dict_a[key],
                    value2=None,
                    message=f"Attribute {key} only in first object",
                )
                diff_count += 1

            for key in only_b:
                children[f".{key}"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=dict_b[key],
                    message=f"Attribute {key} only in second object",
                )
                diff_count += 1

            # Compare common attributes
            common_keys = keys_a & keys_b
            for key in sorted(common_keys, key=str):
                diff = self._compare_values(dict_a[key], dict_b[key], f"{path}.{key}")
                if diff:
                    children[f".{key}"] = diff
                    diff_count += 1
                    if diff_count >= self.max_diffs_per_container:
                        truncated = True
                        return CompoundDiff(source_type="object", children=children, truncated=truncated)
        elif has_dict_a != has_dict_b:
            # One has __dict__, the other doesn't (and no slots to fall back on)
            if not slots_a and not slots_b:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Object mismatch at {path}: __dict__ availability differs",
                )

        # Compare __slots__ attributes
        if slots_a or slots_b:
            # Normalize slots to tuples
            if slots_a is None:
                slots_a = ()
            elif isinstance(slots_a, str):
                slots_a = (slots_a,)
            else:
                slots_a = tuple(slots_a)

            if slots_b is None:
                slots_b = ()
            elif isinstance(slots_b, str):
                slots_b = (slots_b,)
            else:
                slots_b = tuple(slots_b)

            all_slots = set(slots_a) | set(slots_b)

            for slot in sorted(all_slots):
                has_a = hasattr(val_a, slot)
                has_b = hasattr(val_b, slot)

                if has_a and not has_b:
                    children[f".{slot}"] = ValueComparison(
                        status="different",
                        value1=getattr(val_a, slot),
                        value2=None,
                        message=f"Slot {slot} only in first object",
                    )
                    diff_count += 1
                elif has_b and not has_a:
                    children[f".{slot}"] = ValueComparison(
                        status="different",
                        value1=None,
                        value2=getattr(val_b, slot),
                        message=f"Slot {slot} only in second object",
                    )
                    diff_count += 1
                elif has_a and has_b:
                    diff = self._compare_values(
                        getattr(val_a, slot), getattr(val_b, slot), f"{path}.{slot}"
                    )
                    if diff:
                        children[f".{slot}"] = diff
                        diff_count += 1

                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    return CompoundDiff(source_type="object", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="object", children=children, truncated=truncated)
        return None


# Example usage
if __name__ == "__main__":
    # Create test namespaces
    import numpy as np
    import pandas as pd

    # Namespace A
    a = {}
    a["x"] = 42
    a["y"] = 3.14159
    a["z"] = np.array([1.0, 2.0, np.nan, 4.0])
    a["df"] = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, np.nan]})
    a["list_obj"] = [1, 2, 3]
    a["ref1"] = a["list_obj"]  # Create pointer reference

    # Namespace B (identical)
    b = {}
    b["x"] = 42
    b["y"] = 3.14159
    b["z"] = np.array([1.0, 2.0, np.nan, 4.0])
    b["df"] = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, np.nan]})
    b["list_obj"] = [1, 2, 3]
    b["ref1"] = b["list_obj"]  # Maintain pointer structure

    differ = Diff()
    differences = differ.diff(a, b)

    if differences:
        print("Differences found:")
        for var, msg in differences.items():
            print(f"  {var}: {msg}")
    else:
        print("Namespaces are equal!")

    # Test with differences
    b["x"] = 43  # Change value
    differences = differ.diff(a, b)
    print("\nAfter changing b['x']:")
    for var, msg in differences.items():
        print(f"  {var}: {msg}")
