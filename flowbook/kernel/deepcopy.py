"""
Custom Deepcopy Implementation

Extends Python's standard copy.deepcopy with optimized handlers for data science
objects including pandas, Keras models, CatBoost, and cuDF.

================================================================================
OVERVIEW
================================================================================

This module provides a drop-in replacement for copy.deepcopy() that is optimized
for Jupyter notebook checkpointing. It handles complex objects that standard
deepcopy either fails on or handles inefficiently.

Key optimizations:
- Pandas DataFrame/Series: Uses Copy-on-Write (CoW) for non-object columns
- Keras models: Extracts weights only via opaque object pattern
- CatBoost Pool: Uses slice workaround for unpicklable objects
- cuDF objects: Converts to pandas for GPU→CPU transfer
- Functions: Deep copies closures and mutable default arguments

================================================================================
ARCHITECTURE
================================================================================

Dispatch Table Pattern
----------------------
Like the standard library's copy module, we use a dispatch table:

_deepcopy_dispatch (dict): Maps types to handler functions
    - Pre-populated for atomic types (int, str, float, etc.)
    - Pandas types (DataFrame, Series, Index, etc.)
    - Special types (CatBoost Pool, Keras models)

Handler signature: handler(obj, memo) → copied_obj

The memo dictionary tracks already-copied objects to handle:
- Circular references (a contains b contains a)
- Shared references (multiple vars pointing to same object)

Deferred Keras Import
---------------------
Keras/TensorFlow import is expensive (~3 seconds). To avoid this penalty:

1. _is_keras_model(obj): Detects Keras models WITHOUT importing Keras
   - Checks type(obj).__module__ for 'keras' substring
   - Checks MRO for 'Model' or 'Sequential' base classes
   - O(1) check that never triggers import

2. _register_keras_handlers_if_needed(): Only called when Keras model found
   - Imports Keras and registers handlers in _deepcopy_dispatch
   - Subsequent Keras models use fast dispatch path

Opaque Object Pattern
---------------------
Some objects have complex internal structure but simple mutable state.
The OpaqueRegistry pattern handles these:

1. OpaqueHandler.can_handle(obj) → True if handler applies
2. OpaqueHandler.get_mutable_state(obj) → extract state (e.g., weights)
3. OpaqueHandler.copy_with_state(obj, state, memo) → create copy

Currently registered handlers:
- KerasModelHandler: Keras Sequential/Functional models
  - Extracts: model.get_weights()
  - Copies via: clone_model() + set_weights()
  - Rejects unbuilt models (architecture not frozen)

================================================================================
PANDAS HANDLING
================================================================================

DataFrame Deepcopy
------------------
_deepcopy_dataframe(df, memo):
1. Create shallow copy: df.copy(deep=False)
   - With CoW enabled, non-object columns share memory until mutated
2. For each object-dtype column:
   - Deep copy elements recursively
   - Handles nested lists, dicts, custom objects
3. Register in memo for circular reference handling

Optimization: If object column contains only immutable values (strings, ints,
None), skip deep copy entirely (_object_column_is_all_immutable).

Series Deepcopy
---------------
_deepcopy_series(series, memo):
1. If object dtype: deep copy element by element
2. Otherwise: shallow copy (CoW handles mutations)

================================================================================
SPECIAL TYPE HANDLING
================================================================================

CatBoost Pool
-------------
CatBoost Pool objects cannot be pickled directly, but can be sliced.
_deepcopy_catboost_pool uses pool.slice(range(len(pool))) to create a copy.

Keras Models
------------
_deepcopy_keras_model uses the opaque handler pattern:
1. Check if model is built (architecture frozen)
2. Clone model structure via keras.models.clone_model()
3. Copy weights via get_weights()/set_weights()

This avoids copying millions of internal TensorFlow objects.

cuDF Objects
------------
cuDF objects are handled via cudf_compat.deepcopy_cudf:
1. Convert to pandas (GPU→CPU transfer)
2. Standard pandas deepcopy
3. Optionally convert back on restore

Functions
---------
_deepcopy_function handles:
- Closure contents (__closure__)
- Mutable default arguments (__defaults__, __kwdefaults__)
- Does NOT copy __globals__ (would duplicate module state)

================================================================================
MULTIINDEX COLUMN SUPPORT
================================================================================

DataFrames with MultiIndex columns (hierarchical column labels) are fully
supported. When iterating over DataFrame columns, we use positional indexing
(`.iloc[:, i]`) to read columns, which always returns a Series regardless of
column name type. For writing, we use column name indexing (`df[col]`) to
preserve dtype correctly.

================================================================================
IMMUTABILITY CHECKING
================================================================================

For "preserve mode" deepcopy (used in some optimizations):

is_immutable(obj, max_depth=10) → bool:
    Recursively checks if obj and all nested contents are immutable.
    Returns True for: int, float, str, bytes, tuple (of immutables), frozenset

_IMMUTABLE_INFERRED_KINDS: Set of pandas inferred_dtype values that are immutable
    Used by diff module for fast-path column comparisons.

================================================================================
LARGE PRIMITIVE LIST CACHING
================================================================================

For large lists (>= 1000 elements) containing only primitive immutable types
(None, bool, int, float, complex, str, bytes), we cache the checkpoint copy
and reuse it on subsequent checkpoints if the list hasn't changed.

IMPORTANT: Only primitive types are cached - NOT tuples or frozensets.
This keeps the eligibility check O(n) with very low constant factor (just
type membership checks, no recursion).

How It Works
------------
1. When a large list is first checkpointed:
   - Check if all elements are primitive types (_list_has_only_primitives)
   - If yes, compute content hash via hash(tuple(list))
   - Create shallow copy (safe since elements are immutable primitives)
   - Cache: (original_list_ref, cached_copy, content_hash)

2. On subsequent checkpoints of the same list:
   - Check if list is still the same object (identity check)
   - Compute current content hash
   - If hash matches cached hash → reuse cached copy (cache hit)
   - If hash differs → create new copy and update cache (cache miss)

3. Alias traversal optimization (checkpoint.py):
   - is_list_in_immutable_cache() checks if a list is cached
   - Cached lists contain only primitives → no nested mutable refs
   - Alias detection can skip element-by-element traversal for cached lists

Primitive Types (cached)
------------------------
None, bool, int, float, complex, str, bytes

NOT Cached (even if immutable)
------------------------------
Tuples, frozensets, datetime objects, numpy scalars, pandas types, etc.
These are still correctly deep-copied, just not cached.

Cache Management
----------------
- _large_list_cache: Dict mapping id(list) -> cache entry
- Cache is cleared when checkpoints are deleted/cleared (via clear_list_cache())
- Cache is pruned when it exceeds _MAX_LIST_CACHE_SIZE entries
- Stale entries (id reuse after GC) are detected via identity check

Performance Characteristics
---------------------------
- Threshold: _LARGE_LIST_THRESHOLD = 1000 elements
- Eligibility check: O(n) with very low constant (type(x) in set)
- First checkpoint: O(n) type check + O(n) hash + O(n) copy
- Subsequent unchanged: O(n) hash only (copy is skipped)
- Alias traversal: O(1) cache lookup + O(n) hash (skips O(n) traversal)

API
---
- clear_list_cache(): Clear all cached list copies (called by checkpoint.py)
- is_list_in_immutable_cache(lst): Check if list is cached (for alias optimization)
- _list_has_only_primitives(lst): Check if list contains only primitive types
- _list_is_all_immutable(lst): Check if list contains any immutable types (general)

================================================================================
USAGE
================================================================================

Basic usage:
    >>> from flowbook.kernel.deepcopy import deepcopy
    >>> memo = {}
    >>> copy = deepcopy(original, memo)

With Keras model:
    >>> model = keras.Sequential([...])
    >>> model_copy = deepcopy(model, {})  # Uses opaque handler

Shared references preserved:
    >>> shared = [1, 2, 3]
    >>> a = {'x': shared}
    >>> b = {'y': shared}
    >>> memo = {}
    >>> a_copy = deepcopy(a, memo)
    >>> b_copy = deepcopy(b, memo)
    >>> a_copy['x'] is b_copy['y']  # True - sharing preserved
"""

from __future__ import annotations

import datetime
import decimal
import os
import types
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

from flowbook.util.output import log, timer
from flowbook.kernel.column_tracking import suspend_column_tracking
from flowbook.kernel.opaque import OpaqueRegistry


# Sentinel for memo lookups
_nil = []

_convert_object_dtypes = True


# =============================================================================
# Large Immutable List Caching
# =============================================================================
# For large lists containing only immutable values, we cache the checkpoint
# copy and reuse it if the list hasn't changed. This avoids O(n) deep copy
# on repeated checkpoints of unchanged data.
#
# See documentation section "LARGE IMMUTABLE LIST CACHING" above for details.
# =============================================================================

from typing import Dict, Tuple

# Threshold for "large" list optimization (number of elements).
# Lists smaller than this use standard deep copy without caching.
# Rationale: At 1000 elements, hash cost (~0.1ms) is negligible compared
# to deep copy cost (~1-10ms), and the cache lookup overhead is justified.
_LARGE_LIST_THRESHOLD = 1000

# Maximum number of lists to cache to prevent unbounded memory growth.
# When exceeded, stale entries are pruned; if still too large, cache is cleared.
_MAX_LIST_CACHE_SIZE = 100

# Cache structure: id(original_list) -> (original_list_ref, cached_copy, content_hash)
# - original_list_ref: Strong reference to detect id reuse after GC
# - cached_copy: The shallow copy we return on cache hits
# - content_hash: hash(tuple(list)) for change detection
_large_list_cache: Dict[int, Tuple[list, list, int]] = {}

# Primitive immutable types for cache eligibility.
# Only lists containing ONLY these types are cached.
# This keeps the cache check O(n) with very low constant factor (just type checks).
# We intentionally exclude tuples/frozensets to avoid recursion.
_PRIMITIVE_IMMUTABLE_TYPES = frozenset({
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
})


def clear_list_cache() -> None:
    """
    Clear the large list checkpoint cache.

    Called by checkpoint.py when checkpoints are deleted or cleared to:
    1. Free memory held by cached list copies
    2. Prevent stale cache entries from being used

    This should be called when:
    - All checkpoints are cleared (Checkpoints.clear())
    - The last checkpoint is deleted (Checkpoints.delete())
    """
    _large_list_cache.clear()


def get_list_cache_stats() -> Dict[str, int]:
    """
    Get statistics about the large list cache.

    Returns:
        Dict with 'size' (number of cached lists) and 'threshold' (min list size)
    """
    return {
        'size': len(_large_list_cache),
        'threshold': _LARGE_LIST_THRESHOLD,
        'max_size': _MAX_LIST_CACHE_SIZE,
    }


def _prune_list_cache() -> None:
    """
    Remove stale entries from the list cache.

    Stale entries occur when:
    - A list is garbage collected and its id is reused for a different object
    - We detect this by checking if the stored reference still has the same id

    This is called automatically when the cache grows too large.
    """
    stale_ids = []
    for lst_id, (cached_list, _, _) in _large_list_cache.items():
        # If the reference's current id doesn't match the key, the original
        # list was GC'd and the id was reused for a different object
        if id(cached_list) != lst_id:
            stale_ids.append(lst_id)

    for lst_id in stale_ids:
        del _large_list_cache[lst_id]

    if stale_ids:
        log(f"[deepcopy] Pruned {len(stale_ids)} stale entries from list cache")


def _maybe_prune_list_cache() -> None:
    """
    Prune the list cache if it exceeds the maximum size.

    Strategy:
    1. First, remove stale entries (lists that were GC'd)
    2. If still too large, clear the entire cache (simple but effective)

    A more sophisticated LRU strategy could be implemented if needed,
    but for typical notebook usage, full clear is acceptable.
    """
    if len(_large_list_cache) > _MAX_LIST_CACHE_SIZE:
        _prune_list_cache()
        # If still too large after pruning stale entries, clear everything
        if len(_large_list_cache) > _MAX_LIST_CACHE_SIZE:
            log(f"[deepcopy] List cache exceeded {_MAX_LIST_CACHE_SIZE} entries, clearing")
            _large_list_cache.clear()


def _list_is_all_immutable(lst: list) -> bool:
    """
    Check if all elements in a list are immutable (safe to shallow copy).

    For very large lists (>10000 elements), uses sampling to quickly reject
    lists with mutable elements before doing a full scan.

    Args:
        lst: List to check

    Returns:
        True if all elements are definitely immutable, False otherwise.
        Returns False conservatively if unable to determine.
    """
    if not lst:
        return True

    n = len(lst)

    # For very large lists, sample first to quickly reject mutable lists
    # This avoids O(n) scan when the list clearly has mutable elements
    if n > 10000:
        # Sample: first 10, last 10, and 10 evenly spaced in middle
        sample_indices = (
            list(range(min(10, n))) +
            list(range(max(0, n - 10), n)) +
            [n * i // 12 for i in range(1, 11)]
        )
        # Deduplicate and bound
        sample_indices = sorted(set(i for i in sample_indices if 0 <= i < n))

        for i in sample_indices:
            if not is_immutable(lst[i]):
                return False

    # Full check - required for correctness
    for item in lst:
        if not is_immutable(item):
            return False

    return True


def _list_has_only_primitives(lst: list) -> bool:
    """
    Check if all elements in a list are primitive immutable types.

    Only checks for: None, bool, int, float, complex, str, bytes.
    Does NOT recurse into tuples or frozensets - returns False for those.

    This is stricter than _list_is_all_immutable() but much faster (O(n)
    with very low constant factor - just type membership checks).

    Used to determine cache eligibility for the large list optimization.
    The cache can then be used to skip alias traversal in checkpoint.py.

    Args:
        lst: List to check

    Returns:
        True if all elements are primitive immutable types, False otherwise.
    """
    if not lst:
        return True

    for item in lst:
        if type(item) not in _PRIMITIVE_IMMUTABLE_TYPES:
            return False
    return True


def is_list_in_immutable_cache(lst: list) -> bool:
    """
    Check if a list is in the immutable list cache with matching content.

    Used by checkpoint.py to skip alias traversal for lists known to
    contain only immutable primitives. Since the cache only stores lists
    that pass _list_has_only_primitives(), cached lists are guaranteed
    to contain no nested mutable references.

    Returns True if:
    1. List is in cache (was identified as primitive-only during deepcopy)
    2. Identity matches (not a stale entry from GC id reuse)
    3. Content hash matches (list hasn't been modified)

    Performance: O(n) for hash computation, but this allows skipping
    O(n) alias traversal which is more expensive per-element.

    Args:
        lst: List to check

    Returns:
        True if list is cached and unchanged, False otherwise.
    """
    obj_id = id(lst)
    if obj_id not in _large_list_cache:
        return False

    cached_list, cached_copy, cached_hash = _large_list_cache[obj_id]

    # Verify identity (detect GC id reuse)
    if cached_list is not lst:
        return False

    # Verify content unchanged
    try:
        return hash(tuple(lst)) == cached_hash
    except TypeError:
        return False


# ============================================================================
# Immutability checking for preserve mode
# ============================================================================

def is_immutable(obj: Any, _seen: set[int] | None = None, max_depth: int = 10) -> bool:
    """
    Check if an object is deeply immutable (safe to share references).

    Unlike check_deepcopyable, this checks if the object cannot be mutated,
    meaning a shallow copy is sufficient for isolation.

    Args:
        obj: Any Python object to check
        _seen: Internal set for cycle detection (do not pass externally)
        max_depth: Maximum recursion depth for container types

    Returns:
        True if the object is immutable, False otherwise
    """
    # Fast path for common immutable primitives
    if obj is None or obj is True or obj is False:
        return True

    obj_type = type(obj)

    # Immutable primitive types
    if obj_type in (int, float, complex, str, bytes, range):
        return True

    # datetime/time types are immutable
    if obj_type in (
        datetime.date,
        datetime.time,
        datetime.datetime,
        datetime.timedelta,
    ):
        return True

    # Decimal is immutable
    if obj_type is decimal.Decimal:
        return True

    # NumPy scalars are immutable
    if isinstance(obj, np.generic):
        return True

    # Pandas immutable types
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return True
    if obj is pd.NA:
        return True

    # tuple - immutable if all elements are immutable
    if obj_type is tuple:
        if max_depth <= 0:
            return False  # Too deep, assume mutable
        # Handle cycles
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return True  # Already checking this object
        _seen.add(obj_id)
        return all(is_immutable(item, _seen, max_depth - 1) for item in obj)

    # frozenset - immutable if all elements are immutable
    if obj_type is frozenset:
        if max_depth <= 0:
            return False
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return True
        _seen.add(obj_id)
        return all(is_immutable(item, _seen, max_depth - 1) for item in obj)

    # Everything else is considered mutable (list, dict, set, ndarray, custom objects, etc.)
    return False


# These infer_dtype results guarantee all elements are immutable
_IMMUTABLE_INFERRED_KINDS = frozenset({
    "string",
    "bytes",
    "integer",
    "mixed-integer",       # still all ints
    "floating",
    "mixed-integer-float",  # ints and floats - both immutable
    "decimal",
    "complex",
    "boolean",
    "datetime64",
    "datetime",
    "date",
    "timedelta64",
    "timedelta",
    "time",
})


def _object_column_is_all_immutable(series: pd.Series, col_name: str = None) -> bool:
    """
    Check if ALL values in an object column are immutable.

    Uses fast path via infer_dtype for homogeneous columns,
    falls back to element-by-element check only for mixed columns.

    Args:
        series: Series to check (must be object dtype)
        col_name: Optional column name for logging

    Returns:
        True if all values are immutable, False otherwise
    """
    if series.dtype != object or series.empty:
        return True

    col_label = f" ({col_name})" if col_name else ""
    n_rows = len(series)

    with timer(key="deepcopy:immutability_check", message=f"[deepcopy] Immutability check{col_label} ({n_rows} rows)"):
        # FAST PATH: infer_dtype tells us the homogeneous type - O(1)
        kind = infer_dtype(series, skipna=True)
        if kind in _IMMUTABLE_INFERRED_KINDS:
            log(f"Immutability fast path{col_label}: {kind}")
            return True  # All elements are known-immutable types

        # SLOW PATH: mixed or unknown - check element by element - O(n)
        if kind == "mixed":
            log(f"Immutability slow path{col_label}: {kind}, checking {n_rows} elements")
            for val in series.dropna():
                if not is_immutable(val):
                    return False
            return True

        # Unknown kind (e.g., "empty", "unknown-array") - be conservative
        log(f"Immutability unknown kind{col_label}: {kind}, assuming mutable")
        return False


def deepcopy(x, memo=None):
    """Deep copy operation on arbitrary Python objects.

    This is identical to copy.deepcopy except for special handling of:
    - pandas DataFrames: shallow copy + deep copy object columns
    - pandas Series: shallow copy + deep copy if object dtype
    - Functions: deep copy closure and mutable defaults

    Args:
        x: Object to deep copy
        memo: Dictionary tracking already-copied objects (for circular references)

    Returns:
        Deep copy of x
    """
    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    # Handle cudf objects by converting to pandas (all cudf logic in cudf_compat)
    from . import cudf_compat
    if cudf_compat.is_cudf_object(x):
        return cudf_compat.deepcopy_cudf(x, memo)

    cls = type(x)

    copier = _deepcopy_dispatch.get(cls)
    if copier is not None:
        y = copier(x, memo)
    else:
        # Check if this is a Keras model - register handlers lazily to avoid
        # expensive Keras import (~3s) on every deepcopy call
        if _is_keras_model(x):
            _register_keras_handlers_if_needed()
            # Retry dispatch after registration
            copier = _deepcopy_dispatch.get(cls)
            if copier is not None:
                y = copier(x, memo)
                # Skip to the end
                if y is not x:
                    memo[d] = y
                    _keep_alive(x, memo)
                return y

        # Check if this is a PyTorch model - register handlers lazily
        if _is_pytorch_model(x):
            _register_pytorch_handlers_if_needed()
            # Look up dispatch using MRO (dispatch is registered for nn.Module,
            # but we may have nn.Linear, nn.Sequential, etc.)
            for base in cls.__mro__:
                copier = _deepcopy_dispatch.get(base)
                if copier is not None:
                    y = copier(x, memo)
                    if y is not x:
                        memo[d] = y
                        _keep_alive(x, memo)
                    return y

        if issubclass(cls, type):
            y = _deepcopy_atomic(x, memo)
        else:
            copier = getattr(x, "__deepcopy__", None)
            if copier is not None:
                y = copier(memo)
            else:
                reductor = getattr(x, "__reduce_ex__", None)
                if reductor is not None:
                    rv = reductor(4)
                else:
                    reductor = getattr(x, "__reduce__", None)
                    if reductor:
                        rv = reductor()
                    else:
                        raise TypeError(
                            f"un(deep)copyable object of type {cls}"
                        )
                if isinstance(rv, str):
                    y = x
                else:
                    y = _reconstruct(x, memo, *rv)

    # If is its own copy, don't memoize.
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo)  # Make sure x lives at least as long as d
    return y


_deepcopy_dispatch = d = {}


def _deepcopy_atomic(x, memo):
    """Return object unchanged (for immutable types)."""
    return x


# Register atomic types (immutable, safe to share)
d[type(None)] = _deepcopy_atomic
d[type(Ellipsis)] = _deepcopy_atomic
d[type(NotImplemented)] = _deepcopy_atomic
d[int] = _deepcopy_atomic
d[float] = _deepcopy_atomic
d[bool] = _deepcopy_atomic
d[complex] = _deepcopy_atomic
d[bytes] = _deepcopy_atomic
d[str] = _deepcopy_atomic
d[types.CodeType] = _deepcopy_atomic
d[type] = _deepcopy_atomic
d[range] = _deepcopy_atomic
d[types.BuiltinFunctionType] = _deepcopy_atomic
# NOTE: FunctionType is NOT atomic - we override it below
d[property] = _deepcopy_atomic


def _deepcopy_list(x, memo):
    """
    Deep copy a list with optimization for large primitive-content lists.

    For lists with >= _LARGE_LIST_THRESHOLD elements containing only primitive
    immutable types (None, bool, int, float, complex, str, bytes), we cache
    the copy and reuse it on subsequent checkpoints if the content hash matches.
    This avoids O(n) deep copy on repeated checkpoints of unchanged data.

    Optimization flow for large lists:
    1. Check if already in memo (handles circular references)
    2. Check if all elements are primitive immutables (fast type check)
    3. If primitive: compute hash, check cache, return cached or create new
    4. If non-primitive: fall through to standard deep copy

    The cache is also used by checkpoint.py to skip alias traversal for
    cached lists (since primitive-only lists have no nested mutable refs).

    Args:
        x: List to deep copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        Deep copy of the list (or cached copy for large primitive lists)
    """
    obj_id = id(x)

    # Already copied in this deepcopy operation (handles circular references)
    if obj_id in memo:
        return memo[obj_id]

    n = len(x)

    # Optimization for large lists with primitive immutable contents
    if n >= _LARGE_LIST_THRESHOLD:
        # Check if all elements are primitive types (fast O(n) type check)
        if _list_has_only_primitives(x):
            try:
                # Compute content hash for change detection
                content_hash = hash(tuple(x))

                # Check cache for existing copy
                if obj_id in _large_list_cache:
                    cached_list, cached_copy, cached_hash = _large_list_cache[obj_id]
                    # Verify it's the same list object (not id reuse after GC)
                    # and the contents haven't changed
                    if cached_list is x and cached_hash == content_hash:
                        # Cache hit! Reuse the same copy
                        log(f"[deepcopy] Cache hit for primitive list ({n:,} elements)")
                        memo[obj_id] = cached_copy
                        return cached_copy

                # Cache miss or stale entry - create new shallow copy
                log(f"[deepcopy] Caching primitive list ({n:,} elements)")

                # Prune cache if needed before adding new entry
                _maybe_prune_list_cache()

                # Shallow copy is safe since all elements are immutable
                y = x.copy()
                _large_list_cache[obj_id] = (x, y, content_hash)
                memo[obj_id] = y
                return y

            except TypeError:
                # Unhashable elements despite passing immutability check
                # (shouldn't happen, but be safe)
                log(f"[deepcopy] List has unhashable elements, using standard copy")

    # Standard deep copy for small lists or lists with mutable contents
    y = []
    memo[obj_id] = y
    append = y.append
    for a in x:
        append(deepcopy(a, memo))
    return y


d[list] = _deepcopy_list


def _deepcopy_tuple(x, memo):
    """Deep copy a tuple (or return original if contents unchanged)."""
    y = [deepcopy(a, memo) for a in x]
    # We're not going to put the tuple in the memo, but it's still important we
    # check for it, in case the tuple contains recursive mutable structures.
    try:
        return memo[id(x)]
    except KeyError:
        pass
    for k, j in zip(x, y):
        if k is not j:
            y = tuple(y)
            break
    else:
        y = x
    return y


d[tuple] = _deepcopy_tuple


def _deepcopy_dict(x, memo):
    """Deep copy a dictionary."""
    y = {}
    memo[id(x)] = y
    for key, value in x.items():
        y[deepcopy(key, memo)] = deepcopy(value, memo)
    return y


d[dict] = _deepcopy_dict


def _deepcopy_method(x, memo):
    """Copy instance methods."""
    return type(x)(x.__func__, deepcopy(x.__self__, memo))


d[types.MethodType] = _deepcopy_method


# Custom handlers for pandas and functions


def _has_mutable_defaults(func: types.FunctionType) -> bool:
    """
    Check if a function has any mutable default argument values.

    Mutable defaults (like [], {}) are a common source of bugs and need
    to be deep copied to ensure isolation across checkpoints.

    Args:
        func: Function to check

    Returns:
        True if the function has any mutable default arguments
    """
    # Check positional defaults
    if func.__defaults__:
        for default in func.__defaults__:
            if isinstance(default, (list, dict, set)):
                return True
            # Check for other mutable types
            if hasattr(default, '__dict__') or hasattr(default, '__setitem__'):
                if not isinstance(default, (str, bytes, tuple, frozenset, range)):
                    return True

    # Check keyword-only defaults
    if func.__kwdefaults__:
        for default in func.__kwdefaults__.values():
            if isinstance(default, (list, dict, set)):
                return True
            if hasattr(default, '__dict__') or hasattr(default, '__setitem__'):
                if not isinstance(default, (str, bytes, tuple, frozenset, range)):
                    return True

    return False


def _deepcopy_function(func: types.FunctionType, memo: dict[int, Any]) -> types.FunctionType:
    """
    Deep copy a function, including its closure contents and mutable defaults.

    Standard deepcopy doesn't copy closure cell contents or mutable defaults,
    leaving the copied function referencing the same objects as the original.
    This function creates a true deep copy by:
    1. Deep copying each closure cell's contents using the shared memo
    2. Creating new cell objects with the copied contents
    3. Deep copying mutable default arguments
    4. Building a new function with the new closure and defaults

    Args:
        func: Function to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        New function with deep-copied closure and defaults
    """
    # If no closure and no mutable defaults, return the same function
    if func.__closure__ is None and not _has_mutable_defaults(func):
        return func

    # If no closure but has mutable defaults, need to copy defaults only
    if func.__closure__ is None:
        # Create the function first
        new_func = types.FunctionType(
            func.__code__,
            func.__globals__,
            func.__name__,
            func.__defaults__,
            None,
        )

        # Register in memo BEFORE any recursive operations
        memo[id(func)] = new_func

        # Now do the recursive deep copies
        new_defaults = (
            tuple(deepcopy(d, memo) for d in func.__defaults__)
            if func.__defaults__
            else None
        )
        new_kwdefaults = (
            {k: deepcopy(v, memo) for k, v in func.__kwdefaults__.items()}
            if func.__kwdefaults__
            else None
        )

        # Update with deep copied values
        new_func.__defaults__ = new_defaults
        new_func.__kwdefaults__ = new_kwdefaults
        new_func.__annotations__ = func.__annotations__.copy() if func.__annotations__ else {}
        # Deep copy the function's __dict__
        for k, v in func.__dict__.items():
            new_func.__dict__[k] = deepcopy(v, memo)
        new_func.__doc__ = func.__doc__

        return new_func

    # Create a temporary function and register it in memo first to handle circular references
    temp_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    # Register BEFORE recursive operations to handle circular references
    memo[id(func)] = temp_func

    # Now deep copy closure contents
    new_cells = []
    for cell in func.__closure__:
        try:
            copied_contents = deepcopy(cell.cell_contents, memo)
            new_cells.append(types.CellType(copied_contents))
        except ValueError:
            # Empty cell (variable referenced but not yet bound)
            new_cells.append(types.CellType())

    new_closure = tuple(new_cells)

    # Deep copy defaults (may contain mutable objects)
    new_defaults = (
        tuple(deepcopy(d, memo) for d in func.__defaults__)
        if func.__defaults__
        else None
    )
    new_kwdefaults = (
        {k: deepcopy(v, memo) for k, v in func.__kwdefaults__.items()}
        if func.__kwdefaults__
        else None
    )

    # Create the actual new function with deep-copied closure and defaults
    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,  # Share globals (modules, builtins, etc.)
        func.__name__,
        new_defaults,
        new_closure,
    )
    new_func.__kwdefaults__ = new_kwdefaults
    new_func.__annotations__ = func.__annotations__.copy() if func.__annotations__ else {}
    # Deep copy the function's __dict__
    for k, v in func.__dict__.items():
        new_func.__dict__[k] = deepcopy(v, memo)
    new_func.__doc__ = func.__doc__

    # Update memo to point to the actual function
    memo[id(func)] = new_func

    return new_func


d[types.FunctionType] = _deepcopy_function


def _convert_object_column_dtype(series: pd.Series) -> pd.Series:
    """
    Convert object dtype Series to specialized dtype if possible.

    This is done inline during deepcopy to ensure all DataFrames
    (including nested ones) get converted, not just top-level variables.

    Args:
        series: Series to convert

    Returns:
        Converted Series or original if conversion not possible
    """
    if series.dtype != object or series.empty:
        return series

    kind = infer_dtype(series, skipna=True)

    try:
        if kind in {"integer", "mixed-integer"}:
            return series.astype("Int64")
        elif kind in {"floating", "mixed-integer-float"}:
            return series.astype(float)
        elif kind == "decimal":
            return series.astype(float)
        elif kind == "complex":
            return series.astype(complex)
        elif kind == "string":
            return series.astype("string")
        elif kind == "boolean":
            return series.astype("boolean")
        elif kind in {"datetime64", "datetime", "date"}:
            return pd.to_datetime(series)
        elif kind in {"timedelta64", "timedelta"}:
            return pd.to_timedelta(series)
        elif kind == "categorical":
            return series.astype("category")
        else:
            # Mixed types or unknown - leave as object
            return series
    except (TypeError, ValueError, Exception):
        # Conversion failed - return original
        return series


def _deepcopy_dataframe(df: pd.DataFrame, memo: dict[int, Any]) -> pd.DataFrame:
    """
    Deep copy a DataFrame with special object column handling.

    Object dtype columns are handled specially:
    - If all values are immutable (strings, ints, etc.), shallow copy is used
    - If mutable objects exist, element-wise deep copy is performed

    Args:
        df: DataFrame to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        Deep copy of the DataFrame
    """
    obj_id = id(df)
    if obj_id in memo:
        return memo[obj_id]

    # Suspend column tracking during deepcopy to avoid recording internal accesses
    with suspend_column_tracking():
        # Convert object columns to specialized dtypes on the original DataFrame first
        # (only when _convert_object_dtypes=True)
        if _convert_object_dtypes:
            # Use iloc to read columns (handles MultiIndex) but column name to write
            # (preserves dtype correctly)
            for i in range(len(df.columns)):
                col = df.columns[i]
                col_series = df.iloc[:, i]
                if col_series.dtype == object:
                    converted = _convert_object_column_dtype(col_series)
                    if converted.dtype != object:
                        log(f"Converted column {col} from object to {converted.dtype}")
                        # Use column name for assignment to preserve dtype
                        df[col] = converted

        # Shallow copy: CoW handles non-object columns efficiently
        df_copy = df.copy(deep=False)

        # Process remaining object columns
        # Use iloc to read columns (handles MultiIndex) but column name to write
        for i in range(len(df_copy.columns)):
            col = df_copy.columns[i]
            col_series = df_copy.iloc[:, i]
            if col_series.dtype == object:
                # In preserve mode, check if all values are immutable
                if not _convert_object_dtypes and _object_column_is_all_immutable(col_series, col_name=str(col)):
                    # All immutable - shallow copy is sufficient (already done by df.copy)
                    log(f"Shallow copying immutable object column {col}")
                    # Force a copy of the underlying array to ensure independence
                    df_copy[col] = col_series.copy(deep=False)
                else:
                    # Has mutable objects - need element-wise deepcopy
                    num_rows = len(df_copy)
                    if num_rows > 10000:
                        log(f"Deep copying large object column {col} with {num_rows:,} rows...")
                    else:
                        log(f"Deep copying object column {col}")

                    # Apply deep copy and explicitly preserve object dtype
                    result = col_series.apply(lambda x: deepcopy(x, memo))
                    df_copy[col] = result.astype(object)

        memo[obj_id] = df_copy
        return df_copy


d[pd.DataFrame] = _deepcopy_dataframe


def _deepcopy_series(series: pd.Series, memo: dict[int, Any]) -> pd.Series:
    """
    Deep copy a Series with special object dtype handling.

    Object dtype Series are handled specially:
    - If all values are immutable (strings, ints, etc.), shallow copy is used
    - If mutable objects exist, element-wise deep copy is performed

    Args:
        series: Series to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        Deep copy of the Series
    """
    obj_id = id(series)
    if obj_id in memo:
        return memo[obj_id]

    # Suspend column tracking during deepcopy to avoid recording internal accesses
    with suspend_column_tracking():
        # Convert object Series to specialized dtype on the original Series first
        # (only when _convert_object_dtypes=True)
        if _convert_object_dtypes and series.dtype == object:
            # Try to convert to specialized dtype first
            converted = _convert_object_column_dtype(series)

            if converted.dtype != object:
                # Successfully converted - update original Series
                log(f"Converted Series from object to {converted.dtype}")
                series = converted

        # Shallow copy: CoW handles non-object Series efficiently
        series_copy = series.copy(deep=False)

        # Process if still object dtype
        if series_copy.dtype == object:
            # In preserve mode, check if all values are immutable
            if not _convert_object_dtypes and _object_column_is_all_immutable(series_copy, col_name=series_copy.name):
                # All immutable - shallow copy is sufficient
                log(f"Shallow copying immutable object Series")
                # Force a copy of the underlying array to ensure independence
                series_copy = series_copy.copy(deep=False)
            else:
                # Has mutable objects - need element-wise deepcopy
                num_rows = len(series_copy)
                if num_rows > 10000:
                    log(f"Deep copying large object Series with {num_rows:,} rows...")
                else:
                    log(f"Deep copying object Series")

                # Apply deep copy and explicitly preserve object dtype
                result = series_copy.apply(lambda x: deepcopy(x, memo))
                series_copy = result.astype(object)

        memo[obj_id] = series_copy
        return series_copy


d[pd.Series] = _deepcopy_series


# CatBoost Pool handler - Pool explicitly blocks deepcopy, use slice workaround
try:
    from catboost import Pool as CatBoostPool
    from _catboost import _PoolBase

    def _deepcopy_catboost_pool(pool: "CatBoostPool", memo: dict[int, Any]) -> "CatBoostPool":
        """
        Deep copy a CatBoost Pool using the slice workaround.

        CatBoost Pool explicitly raises CatBoostError on __deepcopy__ and has no
        __reduce__ support. However, pool.slice() creates an independent copy
        that doesn't share memory with the original.

        Verified to preserve: features, labels, weights, categorical features,
        feature names, and group_id.

        Args:
            pool: CatBoost Pool to copy (Pool or _PoolBase)
            memo: Shared memo dict for tracking copied objects

        Returns:
            Independent copy of the Pool
        """
        obj_id = id(pool)
        if obj_id in memo:
            return memo[obj_id]

        # slice() with all indices creates a full independent copy
        pool_copy = pool.slice(list(range(pool.num_row())))
        memo[obj_id] = pool_copy
        return pool_copy

    d[CatBoostPool] = _deepcopy_catboost_pool
    d[_PoolBase] = _deepcopy_catboost_pool  # Also handle the base class
except ImportError:
    pass  # CatBoost not installed


# Keras model handler - uses opaque handler pattern for efficient copying
# NOTE: We don't import Keras at module load time to avoid triggering
# matplotlib backend initialization before the kernel is ready.
def _deepcopy_keras_model(model, memo: dict[int, Any]):
    """
    Deep copy a Keras model using the opaque handler pattern.

    For built models:
    - Uses clone_model() to share architecture (immutable after build)
    - Only copies weights via get_weights()/set_weights()
    - Much faster than full deepcopy for large models

    For unbuilt models:
    - Raises TypeError (cannot checkpoint unbuilt models)

    Args:
        model: Keras model (Sequential, Functional, or Model subclass)
        memo: Shared memo dict for tracking copied objects

    Returns:
        Independent copy of the model with weights

    Raises:
        TypeError: If model is not built
    """
    obj_id = id(model)
    if obj_id in memo:
        return memo[obj_id]

    # Get the opaque handler for this model
    handler = OpaqueRegistry.get_handler(model)
    if handler is None:
        # Fallback to stdlib deepcopy if no handler (shouldn't happen)
        import copy as stdlib_copy
        model_copy = stdlib_copy.deepcopy(model)
        memo[obj_id] = model_copy
        return model_copy

    # Check if model can be checkpointed (must be built)
    can_cp, error = handler.is_checkpointable(model)
    if not can_cp:
        raise TypeError(f"Cannot checkpoint Keras model: {error}")

    # Extract mutable state (weights) and create copy with shared architecture
    state = handler.get_mutable_state(model)
    model_copy = handler.copy_with_state(model, state, memo)

    return model_copy


# Flag to track if we've registered Keras handlers (done lazily on first use)
_keras_handlers_registered = False


def _is_keras_model(x) -> bool:
    """Check if x is a Keras model without importing Keras.

    Uses module name checking to avoid the expensive Keras import.
    """
    cls = type(x)
    module = getattr(cls, '__module__', '') or ''
    # Check for tensorflow.keras or standalone keras
    return 'keras' in module and any(
        base.__name__ in ('Model', 'Sequential')
        for base in cls.__mro__
        if hasattr(base, '__name__')
    )


def _register_keras_handlers_if_needed():
    """Register Keras model handlers lazily to avoid import-time side effects.

    IMPORTANT: This is expensive (~3s) due to Keras import. Only call when
    we know we're dealing with a Keras model (use _is_keras_model first).
    """
    global _keras_handlers_registered
    if _keras_handlers_registered:
        return
    _keras_handlers_registered = True

    # Try tensorflow.keras first (more common), then standalone keras
    try:
        from tensorflow.keras.models import Sequential as TFSequential
        from tensorflow.keras.models import Model as TFModel
        _deepcopy_dispatch[TFSequential] = _deepcopy_keras_model
        _deepcopy_dispatch[TFModel] = _deepcopy_keras_model
    except ImportError:
        pass

    try:
        from keras.models import Sequential as KerasSequential
        from keras.models import Model as KerasModel
        _deepcopy_dispatch[KerasSequential] = _deepcopy_keras_model
        _deepcopy_dispatch[KerasModel] = _deepcopy_keras_model
    except ImportError:
        pass  # Keras not installed


def reset_keras_deepcopy_handler():
    """Reset the Keras deepcopy handler registration. For testing."""
    global _keras_handlers_registered
    _keras_handlers_registered = False
    # Also remove from dispatch table
    try:
        from tensorflow.keras.models import Sequential as TFSequential
        from tensorflow.keras.models import Model as TFModel
        if TFSequential in _deepcopy_dispatch:
            del _deepcopy_dispatch[TFSequential]
        if TFModel in _deepcopy_dispatch:
            del _deepcopy_dispatch[TFModel]
    except ImportError:
        pass
    try:
        from keras.models import Sequential as KerasSequential
        from keras.models import Model as KerasModel
        if KerasSequential in _deepcopy_dispatch:
            del _deepcopy_dispatch[KerasSequential]
        if KerasModel in _deepcopy_dispatch:
            del _deepcopy_dispatch[KerasModel]
    except ImportError:
        pass


# PyTorch model handler - uses opaque handler pattern for efficient copying
# NOTE: We don't import PyTorch at module load time to avoid unnecessary loading.

def _deepcopy_pytorch_model(model, memo: dict[int, Any]):
    """
    Deep copy a PyTorch model using the opaque handler pattern.

    For initialized models:
    - Uses stdlib copy.deepcopy() for architecture cloning
    - Restores state via state_dict()
    - Preserves training mode and device placement
    - Preserves custom attributes

    For models with uninitialized lazy modules:
    - Raises TypeError (cannot checkpoint uninitialized models)

    Args:
        model: PyTorch nn.Module
        memo: Shared memo dict for tracking copied objects

    Returns:
        Independent copy of the model

    Raises:
        TypeError: If model has uninitialized lazy modules
    """
    obj_id = id(model)
    if obj_id in memo:
        return memo[obj_id]

    # Get the opaque handler for this model
    handler = OpaqueRegistry.get_handler(model)
    if handler is None:
        # Fallback to stdlib deepcopy if no handler
        import copy as stdlib_copy
        model_copy = stdlib_copy.deepcopy(model)
        memo[obj_id] = model_copy
        return model_copy

    # Check if model can be checkpointed (no uninitialized lazy modules)
    can_cp, error = handler.is_checkpointable(model)
    if not can_cp:
        raise TypeError(f"Cannot checkpoint PyTorch model: {error}")

    # Extract mutable state (state_dict, training mode, etc.) and create copy
    state = handler.get_mutable_state(model)
    model_copy = handler.copy_with_state(model, state, memo)

    return model_copy


# Flag to track if we've registered PyTorch handlers (done lazily on first use)
_pytorch_handlers_registered = False


def _is_pytorch_model(x) -> bool:
    """Check if x is a PyTorch nn.Module without importing torch.

    Uses module name and MRO checking to avoid the expensive PyTorch import.
    """
    cls = type(x)
    module = getattr(cls, '__module__', '') or ''
    if not module.startswith('torch'):
        return False
    # Check MRO for nn.Module base class
    for base in cls.__mro__:
        base_module = getattr(base, '__module__', '') or ''
        if base.__name__ == 'Module' and 'torch.nn' in base_module:
            return True
    return False


def _register_pytorch_handlers_if_needed():
    """Register PyTorch model handlers lazily to avoid import-time side effects.

    Less expensive than Keras import, but still deferred to avoid unnecessary
    loading when PyTorch models aren't used.
    """
    global _pytorch_handlers_registered
    if _pytorch_handlers_registered:
        return
    _pytorch_handlers_registered = True

    try:
        import torch.nn as nn
        _deepcopy_dispatch[nn.Module] = _deepcopy_pytorch_model
    except ImportError:
        pass  # PyTorch not installed


def reset_pytorch_deepcopy_handler():
    """Reset the PyTorch deepcopy handler registration. For testing."""
    global _pytorch_handlers_registered
    _pytorch_handlers_registered = False
    # Also remove from dispatch table
    try:
        import torch.nn as nn
        if nn.Module in _deepcopy_dispatch:
            del _deepcopy_dispatch[nn.Module]
    except ImportError:
        pass


del d  # Clean up namespace


def _keep_alive(x, memo):
    """Keeps a reference to the object x in the memo.

    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)] = [x]


def _reconstruct(x, memo, func, args,
                 state=None, listiter=None, dictiter=None):
    """Reconstruct an object from pickle-like reduction."""
    deep = memo is not None
    if deep and args:
        args = tuple(deepcopy(arg, memo) for arg in args)
    y = func(*args)
    if deep:
        memo[id(x)] = y

    if state is not None:
        if deep:
            state = deepcopy(state, memo)
        if hasattr(y, '__setstate__'):
            y.__setstate__(state)
        else:
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            else:
                slotstate = None
            if state is not None:
                y.__dict__.update(state)
            if slotstate is not None:
                for key, value in slotstate.items():
                    setattr(y, key, value)

    if listiter is not None:
        if deep:
            for item in listiter:
                item = deepcopy(item, memo)
                y.append(item)
        else:
            for item in listiter:
                y.append(item)
    if dictiter is not None:
        if deep:
            for key, value in dictiter:
                key = deepcopy(key, memo)
                value = deepcopy(value, memo)
                y[key] = value
        else:
            for key, value in dictiter:
                y[key] = value
    return y
