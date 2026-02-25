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
LARGE PRIMITIVE CONTAINER CACHING
================================================================================

For large containers (>= 1000 elements) containing only primitive immutable
types (None, bool, int, float, complex, str, bytes), we cache checkpoint copies
and reuse them on subsequent checkpoints if the container hasn't changed.

Supported container types:
- list: cache shallow copy if all elements are primitive
- set: cache shallow copy if all elements are primitive
- dict: cache shallow copy if all VALUES are primitive (keys always immutable)
- tuple: return original immediately if all elements are primitive (no copy needed)

This optimization also enables alias traversal to skip these containers since
they contain no nested mutable references.

How It Works
------------
1. When a large container is first checkpointed:
   - Check if all elements/values are primitive types (O(n) type checks)
   - If yes, compute content hash for change detection
   - Create shallow copy (or return original for tuples)
   - Cache: (original_ref, cached_copy, content_hash, length)
   - Track copy in _primitive_*_copies dict for O(1) lookup

2. On subsequent checkpoints (cache-first optimization):
   - Check cache by id FIRST (O(1)) before any O(n) operations
   - If found, verify identity and length (O(1) fast rejection)
   - Only then compute hash to verify content unchanged (O(n))
   - If hash matches → reuse cached copy (cache hit)
   - If not in cache → check primitives and create new entry

3. Alias traversal optimization (checkpoint.py):
   - is_primitive_container() checks if a container is cached/primitive
   - Cached containers have no nested mutable refs
   - Alias detection skips element-by-element traversal for these

4. Diff optimization (diff.py):
   - are_primitive_containers_equal(a, b) checks if two containers match via cache
   - If original and its cached copy are being compared, skip element-wise diff
   - Enables O(1) equality check for unchanged large containers

Content Hash Computation
------------------------
- list: hash(tuple(lst))
- set: hash(frozenset(s))
- dict: hash(tuple(sorted(d.items())))
- tuple: _tuple_has_only_primitives() check (no hash needed)

Primitive Types
---------------
None, bool, int, float, complex, str, bytes

NOT Cached (even if immutable)
------------------------------
Nested tuples, frozensets, datetime objects, numpy scalars, pandas types.
These are still correctly deep-copied, just not cached.

Cache Management
----------------
- _large_list_cache, _large_set_cache, _large_dict_cache: Main caches
- _primitive_list_copies, _primitive_set_copies, _primitive_dict_copies: Copy tracking
- Caches cleared via clear_container_cache() when checkpoints deleted
- Pruning when exceeding _MAX_CONTAINER_CACHE_SIZE entries

Performance Characteristics
---------------------------
- Threshold: _LARGE_LIST_THRESHOLD = 1000 elements
- Eligibility check: O(n) with very low constant (type(x) in set)
- First checkpoint: O(n) type check + O(n) hash + O(n) copy
- Subsequent unchanged: O(n) hash only (copy is skipped)
- Alias traversal on copies: O(1) dict lookup
- Alias traversal on originals: O(1) cache lookup + O(n) hash

API
---
- clear_container_cache(): Clear all container caches
- is_primitive_container(obj): Check if list/set/dict/tuple is cached/primitive
- are_primitive_containers_equal(a, b): Check if two containers are equal via cache
- _list_has_only_primitives(lst): Check if list has only primitive elements
- _set_has_only_primitives(s): Check if set has only primitive elements
- _dict_has_only_primitive_values(d): Check if dict has only primitive values
- _tuple_has_only_primitives(t): Check if tuple has only primitive elements

================================================================================
USAGE
================================================================================

Basic usage:
    >>> from flowbook.kernel_support.deepcopy import deepcopy
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
from flowbook.kernel_support.column_tracking import suspend_column_tracking
from flowbook.kernel_support.opaque import OpaqueRegistry


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

# Maximum number of containers to cache (per type) to prevent unbounded memory growth.
# When exceeded, stale entries are pruned; if still too large, cache is cleared.
_MAX_CONTAINER_CACHE_SIZE = 10000

# Primitive immutable types for cache eligibility.
# Only containers with ONLY these types are cached.
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

# =============================================================================
# List caching
# =============================================================================
# Cache structure: id(original) -> (original_ref, cached_copy, content_hash)
_large_list_cache: Dict[int, Tuple[list, list, int]] = {}
# Dict mapping id(copy) -> copy for recognizing copies during alias traversal
_primitive_list_copies: Dict[int, list] = {}

# =============================================================================
# Set caching
# =============================================================================
# Cache structure: id(original) -> (original_ref, cached_copy, content_hash)
# content_hash computed via hash(frozenset(s))
_large_set_cache: Dict[int, Tuple[set, set, int]] = {}
# Dict mapping id(copy) -> copy for recognizing copies during alias traversal
_primitive_set_copies: Dict[int, set] = {}

# =============================================================================
# Dict caching
# =============================================================================
# Cache structure: id(original) -> (original_ref, cached_copy, content_hash)
# content_hash computed via hash(tuple(sorted(d.items())))
_large_dict_cache: Dict[int, Tuple[dict, dict, int]] = {}
# Dict mapping id(copy) -> copy for recognizing copies during alias traversal
_primitive_dict_copies: Dict[int, dict] = {}

# =============================================================================
# NumPy ndarray caching
# =============================================================================
# For numeric (non-object) ndarrays unchanged between checkpoints, we reuse
# the previous checkpoint's copy to avoid O(n) memcpy + allocation.
#
# Cache structure: id(original) -> (cached_copy, shape, dtype, nbytes)
# Note: We do NOT store a reference to the original array, so it can be GC'd.
# If a new array reuses the same id with matching metadata, the content
# equality check ensures correctness.
_ndarray_cache: Dict[int, Tuple[np.ndarray, tuple, np.dtype, int]] = {}
# Dict mapping id(copy) -> True for recognizing copies during alias traversal
_ndarray_copies: Dict[int, np.ndarray] = {}

# Maximum cached ndarrays before clearing (prevents unbounded memory growth)
_MAX_NDARRAY_CACHE_SIZE = 1000

# Chunk size in bytes for _fast_array_equal comparison.
# Limits the boolean intermediate array to ~1 MB per chunk and allows
# early termination on first mismatched chunk.
_ARRAY_EQUAL_CHUNK_BYTES = 1 << 20  # 1 MB


def clear_container_cache() -> None:
    """
    Clear all primitive container caches (lists, sets, dicts) and ndarray cache.

    Called by checkpoint.py when checkpoints are deleted or cleared to:
    1. Free memory held by cached copies
    2. Prevent stale cache entries from being used

    This should be called when:
    - All checkpoints are cleared (Checkpoints.clear())
    - The last checkpoint is deleted (Checkpoints.delete())
    """
    _large_list_cache.clear()
    _primitive_list_copies.clear()
    _large_set_cache.clear()
    _primitive_set_copies.clear()
    _large_dict_cache.clear()
    _primitive_dict_copies.clear()
    _ndarray_cache.clear()
    _ndarray_copies.clear()


# Backwards compatibility alias
clear_list_cache = clear_container_cache


def get_container_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the primitive container caches.

    Returns:
        Dict with cache sizes for lists, sets, dicts and threshold info.
    """
    return {
        'list_cache_size': len(_large_list_cache),
        'set_cache_size': len(_large_set_cache),
        'dict_cache_size': len(_large_dict_cache),
        'ndarray_cache_size': len(_ndarray_cache),
        'threshold': _LARGE_LIST_THRESHOLD,
        'max_size': _MAX_CONTAINER_CACHE_SIZE,
    }


# Backwards compatibility alias
def get_list_cache_stats() -> Dict[str, int]:
    """Get statistics about the large list cache (deprecated, use get_container_cache_stats)."""
    return {
        'size': len(_large_list_cache),
        'threshold': _LARGE_LIST_THRESHOLD,
        'max_size': _MAX_CONTAINER_CACHE_SIZE,
    }


def get_cache_sizes() -> Dict[str, int]:
    """
    Get memory sizes of all deepcopy caches in bytes.

    This provides visibility into the memory overhead of the deepcopy
    caching system, which can be significant for large notebooks.

    Returns:
        Dictionary with sizes in bytes for each cache:
        - large_list_cache: Main list cache entries
        - large_set_cache: Main set cache entries
        - large_dict_cache: Main dict cache entries
        - ndarray_cache: NumPy array cache entries
        - primitive_list_copies: List copy references
        - primitive_set_copies: Set copy references
        - primitive_dict_copies: Dict copy references
        - ndarray_copies: NumPy array copy references
    """
    import sys

    def cache_size(cache: dict) -> int:
        """Estimate size of a cache dictionary."""
        total = sys.getsizeof(cache)
        for k, v in cache.items():
            total += 8  # id key
            if isinstance(v, tuple):
                total += sys.getsizeof(v)
                # Don't count actual objects - they're user data
        return total

    return {
        'large_list_cache': cache_size(_large_list_cache),
        'large_set_cache': cache_size(_large_set_cache),
        'large_dict_cache': cache_size(_large_dict_cache),
        'ndarray_cache': cache_size(_ndarray_cache),
        'primitive_list_copies': sys.getsizeof(_primitive_list_copies),
        'primitive_set_copies': sys.getsizeof(_primitive_set_copies),
        'primitive_dict_copies': sys.getsizeof(_primitive_dict_copies),
        'ndarray_copies': sys.getsizeof(_ndarray_copies),
    }


def get_cached_object_ids() -> set:
    """
    Get IDs of all objects stored in deepcopy caches.

    These objects are shared across multiple checkpoints and should not
    be double-counted when measuring checkpoint memory. Instead, they
    should be attributed to general cache overhead.

    Returns:
        Set of object IDs currently in any cache
    """
    cached_ids = set()

    # Large container caches store tuples of (copy, original, ...)
    for _, (cached_copy, _, _, _) in _large_list_cache.items():
        cached_ids.add(id(cached_copy))

    for _, (cached_copy, _, _, _) in _large_set_cache.items():
        cached_ids.add(id(cached_copy))

    for _, (cached_copy, _, _, _) in _large_dict_cache.items():
        cached_ids.add(id(cached_copy))

    for _, (cached_copy, _, _, _) in _ndarray_cache.items():
        cached_ids.add(id(cached_copy))

    # Primitive copy caches store copies directly
    for copy_id, copy_obj in _primitive_list_copies.items():
        cached_ids.add(id(copy_obj))

    for copy_id, copy_obj in _primitive_set_copies.items():
        cached_ids.add(id(copy_obj))

    for copy_id, copy_obj in _primitive_dict_copies.items():
        cached_ids.add(id(copy_obj))

    for copy_id, copy_obj in _ndarray_copies.items():
        cached_ids.add(id(copy_obj))

    return cached_ids


def get_cached_objects_size() -> int:
    """
    Get total memory size of all objects stored in deepcopy caches.

    This measures the actual cached data, not just the cache structure.
    Should be used to report cache overhead separately from checkpoint memory.

    Returns:
        Total bytes of cached objects
    """
    from flowbook.kernel_support.heap_size import HeapSizer

    sizer = HeapSizer()
    total = 0

    # Large container caches
    for _, (cached_copy, _, _, _) in _large_list_cache.items():
        total += sizer.sizeof(cached_copy)

    for _, (cached_copy, _, _, _) in _large_set_cache.items():
        total += sizer.sizeof(cached_copy)

    for _, (cached_copy, _, _, _) in _large_dict_cache.items():
        total += sizer.sizeof(cached_copy)

    for _, (cached_copy, _, _, _) in _ndarray_cache.items():
        total += sizer.sizeof(cached_copy)

    # Primitive copies (smaller objects, but count them)
    for copy_id, copy_obj in _primitive_list_copies.items():
        total += sizer.sizeof(copy_obj)

    for copy_id, copy_obj in _primitive_set_copies.items():
        total += sizer.sizeof(copy_obj)

    for copy_id, copy_obj in _primitive_dict_copies.items():
        total += sizer.sizeof(copy_obj)

    for copy_id, copy_obj in _ndarray_copies.items():
        total += sizer.sizeof(copy_obj)

    return total


def _prune_list_cache() -> None:
    """
    Remove stale entries from the list cache.

    Stale entries occur when:
    - A list is garbage collected and its id is reused for a different object
    - We detect this by checking if the stored reference still has the same id

    This is called automatically when the cache grows too large.
    """
    stale_ids = []
    for lst_id, (cached_list, _, _, _) in _large_list_cache.items():
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
    if len(_large_list_cache) > _MAX_CONTAINER_CACHE_SIZE:
        _prune_list_cache()
        # If still too large after pruning stale entries, clear everything
        if len(_large_list_cache) > _MAX_CONTAINER_CACHE_SIZE:
            log(f"[deepcopy] List cache exceeded {_MAX_CONTAINER_CACHE_SIZE} entries, clearing")
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


def _set_has_only_primitives(s: set) -> bool:
    """
    Check if all elements in a set are primitive immutable types.

    Only checks for: None, bool, int, float, complex, str, bytes.
    This is O(n) with very low constant factor (just type membership checks).

    Used to determine cache eligibility for the large set optimization.

    Args:
        s: Set to check

    Returns:
        True if all elements are primitive immutable types, False otherwise.
    """
    for item in s:
        if type(item) not in _PRIMITIVE_IMMUTABLE_TYPES:
            return False
    return True


def _dict_has_only_primitive_values(d: dict) -> bool:
    """
    Check if all VALUES in a dict are primitive immutable types.

    Keys are always immutable (dict requirement), so we only check values.
    Only checks for: None, bool, int, float, complex, str, bytes.
    This is O(n) with very low constant factor (just type membership checks).

    Used to determine cache eligibility for the large dict optimization.

    Args:
        d: Dict to check

    Returns:
        True if all values are primitive immutable types, False otherwise.
    """
    for value in d.values():
        if type(value) not in _PRIMITIVE_IMMUTABLE_TYPES:
            return False
    return True


def _tuple_has_only_primitives(t: tuple) -> bool:
    """
    Check if all elements in a tuple are primitive immutable types.

    Only checks for: None, bool, int, float, complex, str, bytes.
    This is O(n) with very low constant factor (just type membership checks).

    Used to short-circuit tuple deepcopy and skip alias traversal.

    Args:
        t: Tuple to check

    Returns:
        True if all elements are primitive immutable types, False otherwise.
    """
    for item in t:
        if type(item) not in _PRIMITIVE_IMMUTABLE_TYPES:
            return False
    return True


def is_primitive_container(obj) -> bool:
    """
    Check if a container is known to contain only primitive immutable types.

    Used by checkpoint.py to skip alias traversal for containers known to
    contain only immutable primitives (no nested mutable references).

    Supports: list, set, dict, tuple

    For lists, sets, dicts:
    - Checks the copies dict first (O(1) - for checkpoint copies)
    - Then checks the main cache with length check before expensive hash

    For tuples:
    - No caching needed (tuples are immutable)
    - Just checks if all elements are primitive types

    Args:
        obj: Container to check (list, set, dict, or tuple)

    Returns:
        True if container is known to contain only primitives, False otherwise.
    """
    obj_id = id(obj)

    if isinstance(obj, list):
        # Fast path: check if this is a known primitive copy
        if obj_id in _primitive_list_copies:
            if _primitive_list_copies[obj_id] is obj:
                return True
            del _primitive_list_copies[obj_id]

        # Check main cache for original (with length check before hash)
        if obj_id in _large_list_cache:
            cached, cached_copy, cached_hash, cached_len = _large_list_cache[obj_id]
            if cached is obj and len(obj) == cached_len:
                try:
                    return hash(tuple(obj)) == cached_hash
                except TypeError:
                    pass
        return False

    elif isinstance(obj, set):
        # Fast path: check if this is a known primitive copy
        if obj_id in _primitive_set_copies:
            if _primitive_set_copies[obj_id] is obj:
                return True
            del _primitive_set_copies[obj_id]

        # Check main cache for original (with length check before hash)
        if obj_id in _large_set_cache:
            cached, cached_copy, cached_hash, cached_len = _large_set_cache[obj_id]
            if cached is obj and len(obj) == cached_len:
                try:
                    return hash(frozenset(obj)) == cached_hash
                except TypeError:
                    pass
        return False

    elif isinstance(obj, dict):
        # Fast path: check if this is a known primitive copy
        if obj_id in _primitive_dict_copies:
            if _primitive_dict_copies[obj_id] is obj:
                return True
            del _primitive_dict_copies[obj_id]

        # Check main cache for original (with length check before hash)
        if obj_id in _large_dict_cache:
            cached, cached_copy, cached_hash, cached_len = _large_dict_cache[obj_id]
            if cached is obj and len(obj) == cached_len:
                try:
                    return hash(tuple(sorted(obj.items()))) == cached_hash
                except TypeError:
                    pass
        return False

    elif isinstance(obj, tuple):
        # Tuples don't need caching - just check if primitive
        # Only do this for large tuples to avoid overhead on small ones
        if len(obj) >= _LARGE_LIST_THRESHOLD:
            return _tuple_has_only_primitives(obj)
        return False

    return False


# Backwards compatibility alias
def is_list_in_immutable_cache(lst: list) -> bool:
    """Check if list is cached (deprecated, use is_primitive_container)."""
    return is_primitive_container(lst)


def are_primitive_containers_equal(a, b) -> bool:
    """
    Check if two containers are equal via the primitive container cache.

    Used by diff.py to short-circuit comparison of large primitive containers.
    If one is an original in the cache and the other is its copy (or vice versa),
    and the content hash still matches, they must be equal.

    This avoids O(n) element-by-element comparison for large primitive lists/sets/dicts.

    Args:
        a: First container (list, set, or dict)
        b: Second container (list, set, or dict)

    Returns:
        True if containers are known equal via cache, False if unknown (need full comparison)
    """
    if type(a) is not type(b):
        return False

    id_a, id_b = id(a), id(b)

    # Same object - trivially equal (but this should be caught by identity check in diff)
    if id_a == id_b:
        return True

    if isinstance(a, list):
        # Quick length check
        if len(a) != len(b):
            return False

        # Check if a is a known copy
        a_is_copy = id_a in _primitive_list_copies and _primitive_list_copies[id_a] is a
        # Check if b is a known copy
        b_is_copy = id_b in _primitive_list_copies and _primitive_list_copies[id_b] is b
        # Check if a is an original in cache
        a_is_original = id_a in _large_list_cache and _large_list_cache[id_a][0] is a
        # Check if b is an original in cache
        b_is_original = id_b in _large_list_cache and _large_list_cache[id_b][0] is b

        # If neither is in the cache system, we can't short-circuit
        if not (a_is_copy or a_is_original or b_is_copy or b_is_original):
            return False

        # Get content hashes
        hash_a = hash_b = None

        if a_is_original:
            _, _, hash_a, _ = _large_list_cache[id_a]
        elif a_is_copy:
            # Find original that a is a copy of
            for orig_id, (orig, copy, h, _) in _large_list_cache.items():
                if copy is a:
                    hash_a = h
                    break

        if b_is_original:
            _, _, hash_b, _ = _large_list_cache[id_b]
        elif b_is_copy:
            # Find original that b is a copy of
            for orig_id, (orig, copy, h, _) in _large_list_cache.items():
                if copy is b:
                    hash_b = h
                    break

        # If we have both hashes and they match, containers are equal
        if hash_a is not None and hash_b is not None and hash_a == hash_b:
            return True

        return False

    elif isinstance(a, set):
        if len(a) != len(b):
            return False

        a_is_copy = id_a in _primitive_set_copies and _primitive_set_copies[id_a] is a
        b_is_copy = id_b in _primitive_set_copies and _primitive_set_copies[id_b] is b
        a_is_original = id_a in _large_set_cache and _large_set_cache[id_a][0] is a
        b_is_original = id_b in _large_set_cache and _large_set_cache[id_b][0] is b

        if not (a_is_copy or a_is_original or b_is_copy or b_is_original):
            return False

        hash_a = hash_b = None

        if a_is_original:
            _, _, hash_a, _ = _large_set_cache[id_a]
        elif a_is_copy:
            for orig_id, (orig, copy, h, _) in _large_set_cache.items():
                if copy is a:
                    hash_a = h
                    break

        if b_is_original:
            _, _, hash_b, _ = _large_set_cache[id_b]
        elif b_is_copy:
            for orig_id, (orig, copy, h, _) in _large_set_cache.items():
                if copy is b:
                    hash_b = h
                    break

        if hash_a is not None and hash_b is not None and hash_a == hash_b:
            return True

        return False

    elif isinstance(a, dict):
        if len(a) != len(b):
            return False

        a_is_copy = id_a in _primitive_dict_copies and _primitive_dict_copies[id_a] is a
        b_is_copy = id_b in _primitive_dict_copies and _primitive_dict_copies[id_b] is b
        a_is_original = id_a in _large_dict_cache and _large_dict_cache[id_a][0] is a
        b_is_original = id_b in _large_dict_cache and _large_dict_cache[id_b][0] is b

        if not (a_is_copy or a_is_original or b_is_copy or b_is_original):
            return False

        hash_a = hash_b = None

        if a_is_original:
            _, _, hash_a, _ = _large_dict_cache[id_a]
        elif a_is_copy:
            for orig_id, (orig, copy, h, _) in _large_dict_cache.items():
                if copy is a:
                    hash_a = h
                    break

        if b_is_original:
            _, _, hash_b, _ = _large_dict_cache[id_b]
        elif b_is_copy:
            for orig_id, (orig, copy, h, _) in _large_dict_cache.items():
                if copy is b:
                    hash_b = h
                    break

        if hash_a is not None and hash_b is not None and hash_a == hash_b:
            return True

        return False

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
    from flowbook.kernel_support import cudf_compat
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

        # Check if this is a LightGBM model - register handlers lazily
        if _is_lightgbm_model(x):
            _register_lightgbm_handlers_if_needed()
            # Retry dispatch after registration
            copier = _deepcopy_dispatch.get(cls)
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
        # Check cache FIRST by id (O(1)) before expensive O(n) operations
        if obj_id in _large_list_cache:
            cached_list, cached_copy, cached_hash, cached_len = _large_list_cache[obj_id]
            # Check identity (O(1)) and length (O(1)) before expensive hash
            if cached_list is x and cached_len == n:
                try:
                    if hash(tuple(x)) == cached_hash:
                        # Cache hit! Reuse the same copy
                        log(f"[deepcopy] Cache hit for primitive list ({n:,} elements)")
                        memo[obj_id] = cached_copy
                        return cached_copy
                except TypeError:
                    pass  # List became unhashable, fall through

        # No cache hit - check if eligible for caching
        if _list_has_only_primitives(x):
            try:
                # Compute content hash for change detection
                content_hash = hash(tuple(x))

                # Cache miss or stale entry - create new shallow copy
                log(f"[deepcopy] Caching primitive list ({n:,} elements)")

                # Prune cache if needed before adding new entry
                _maybe_prune_list_cache()

                # Shallow copy is safe since all elements are immutable
                y = x.copy()
                _large_list_cache[obj_id] = (x, y, content_hash, n)
                # Track copy so alias traversal can recognize it (keyed by copy's id)
                _primitive_list_copies[id(y)] = y
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
    """
    Deep copy a tuple (or return original if contents unchanged).

    Optimization for large tuples:
    - If all elements are primitive types, return original immediately
    - No need to iterate through elements since tuples are immutable
    - Alias traversal can also skip these tuples
    """
    n = len(x)

    # Optimization for large tuples with primitive contents
    if n >= _LARGE_LIST_THRESHOLD:
        if _tuple_has_only_primitives(x):
            # Tuple is immutable and contains only primitives
            # No need to iterate - just return original
            log(f"[deepcopy] Primitive tuple ({n:,} elements) - returning original")
            return x

    # Standard tuple deepcopy - iterate through elements
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


def _deepcopy_set(x, memo):
    """
    Deep copy a set with optimization for large primitive-content sets.

    For sets with >= _LARGE_LIST_THRESHOLD elements containing only primitive
    immutable types, we cache the copy and reuse it on subsequent checkpoints
    if the content hash matches.
    """
    obj_id = id(x)

    # Already copied in this deepcopy operation (handles circular references)
    if obj_id in memo:
        return memo[obj_id]

    n = len(x)

    # Optimization for large sets with primitive immutable contents
    if n >= _LARGE_LIST_THRESHOLD:
        # Check cache FIRST by id (O(1)) before expensive O(n) operations
        if obj_id in _large_set_cache:
            cached_set, cached_copy, cached_hash, cached_len = _large_set_cache[obj_id]
            # Check identity (O(1)) and length (O(1)) before expensive hash
            if cached_set is x and cached_len == n:
                try:
                    if hash(frozenset(x)) == cached_hash:
                        # Cache hit! Reuse the same copy
                        log(f"[deepcopy] Cache hit for primitive set ({n:,} elements)")
                        memo[obj_id] = cached_copy
                        return cached_copy
                except TypeError:
                    pass  # Set became unhashable, fall through

        # No cache hit - check if eligible for caching
        if _set_has_only_primitives(x):
            try:
                # Compute content hash for change detection
                content_hash = hash(frozenset(x))

                # Cache miss or stale entry - create new shallow copy
                log(f"[deepcopy] Caching primitive set ({n:,} elements)")

                # Shallow copy is safe since all elements are immutable
                y = x.copy()
                _large_set_cache[obj_id] = (x, y, content_hash, n)
                _primitive_set_copies[id(y)] = y
                memo[obj_id] = y
                return y

            except TypeError:
                log(f"[deepcopy] Set has unhashable elements, using standard copy")

    # Standard deep copy for small sets or sets with mutable contents
    y = set()
    memo[obj_id] = y
    for a in x:
        y.add(deepcopy(a, memo))
    return y


d[set] = _deepcopy_set


def _deepcopy_dict(x, memo):
    """
    Deep copy a dictionary with optimization for large primitive-value dicts.

    For dicts with >= _LARGE_LIST_THRESHOLD items where all VALUES are primitive
    immutable types, we cache the copy and reuse it on subsequent checkpoints
    if the content hash matches.
    """
    obj_id = id(x)

    # Already copied in this deepcopy operation (handles circular references)
    if obj_id in memo:
        return memo[obj_id]

    n = len(x)

    # Optimization for large dicts with primitive immutable values
    if n >= _LARGE_LIST_THRESHOLD:
        # Check cache FIRST by id (O(1)) before expensive O(n) operations
        if obj_id in _large_dict_cache:
            cached_dict, cached_copy, cached_hash, cached_len = _large_dict_cache[obj_id]
            # Check identity (O(1)) and length (O(1)) before expensive hash
            if cached_dict is x and cached_len == n:
                try:
                    if hash(tuple(sorted(x.items()))) == cached_hash:
                        # Cache hit! Reuse the same copy
                        log(f"[deepcopy] Cache hit for primitive dict ({n:,} items)")
                        memo[obj_id] = cached_copy
                        return cached_copy
                except TypeError:
                    pass  # Dict became unhashable, fall through

        # No cache hit - check if eligible for caching
        if _dict_has_only_primitive_values(x):
            try:
                # Compute content hash for change detection
                # Keys are always immutable, values are primitives
                content_hash = hash(tuple(sorted(x.items())))

                # Cache miss or stale entry - create new shallow copy
                log(f"[deepcopy] Caching primitive dict ({n:,} items)")

                # Shallow copy is safe since all values are immutable
                y = x.copy()
                _large_dict_cache[obj_id] = (x, y, content_hash, n)
                _primitive_dict_copies[id(y)] = y
                memo[obj_id] = y
                return y

            except TypeError:
                log(f"[deepcopy] Dict has unhashable keys, using standard copy")

    # Standard deep copy for small dicts or dicts with mutable values
    y = {}
    memo[obj_id] = y
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

        # Deep copy the Index to ensure isolation between checkpoints
        # Without this, df.copy(deep=False) shares the Index's underlying array
        df_copy.index = _deepcopy_index(df.index, memo)

        # Also deep copy column labels if they're an Index (not just strings)
        if isinstance(df.columns, pd.Index) and not isinstance(df.columns, pd.RangeIndex):
            df_copy.columns = _deepcopy_index(df.columns, memo)

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

        # Deep copy the Index to ensure isolation between checkpoints
        series_copy.index = _deepcopy_index(series.index, memo)

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
                name_str = f" '{series_copy.name}'" if series_copy.name else ""
                if num_rows > 10000:
                    log(f"Deep copying large object Series{name_str} with {num_rows:,} rows...")
                else:
                    log(f"Deep copying object Series{name_str} ({num_rows:,} rows)")

                # Apply deep copy and explicitly preserve object dtype
                result = series_copy.apply(lambda x: deepcopy(x, memo))
                series_copy = result.astype(object)

        memo[obj_id] = series_copy
        return series_copy


d[pd.Series] = _deepcopy_series


def _deepcopy_index(index: pd.Index, memo: dict[int, Any]) -> pd.Index:
    """
    Deep copy a pandas Index with proper data isolation and cross-checkpoint caching.

    This ensures that the Index's underlying data array is properly copied,
    preventing shared memory between checkpoints. Without this handler,
    df.copy(deep=False) shares Index data, which could lead to checkpoint
    corruption if the Index is modified in place.

    Uses memo for caching to avoid multiple copies of the same Index object
    (e.g., when both df and df.groupby() reference the same Index).

    For cross-checkpoint efficiency, we extract the Index's backing array
    and use the ndarray cache to avoid re-copying unchanged arrays.

    Args:
        index: Index to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        Deep copy of the Index with isolated data
    """
    obj_id = id(index)
    if obj_id in memo:
        return memo[obj_id]

    # RangeIndex is special - it doesn't store data, just start/stop/step
    # Creating a new RangeIndex with same parameters is sufficient
    if isinstance(index, pd.RangeIndex):
        # RangeIndex is immutable and doesn't have shared data
        # Just create a new one with the same parameters
        index_copy = pd.RangeIndex(
            start=index.start,
            stop=index.stop,
            step=index.step,
            name=index.name
        )
        memo[obj_id] = index_copy
        return index_copy

    # For numeric Index types, use ndarray cache for cross-checkpoint deduplication
    # This avoids re-copying the same unchanged Index data across checkpoints
    if isinstance(index, pd.MultiIndex):
        # MultiIndex: copy codes and levels separately
        # Codes are small int8/int16 arrays, levels are Index objects
        new_codes = []
        for code_arr in index.codes:
            # Use ndarray cache for codes (they're numpy arrays)
            code_arr = np.asarray(code_arr)
            new_codes.append(_deepcopy_ndarray(code_arr, memo))

        new_levels = []
        for level in index.levels:
            new_levels.append(_deepcopy_index(level, memo))

        index_copy = pd.MultiIndex(
            levels=new_levels,
            codes=new_codes,
            names=index.names,
            verify_integrity=False
        )
    elif hasattr(index, '_data') and isinstance(index._data, np.ndarray):
        # Standard numeric Index (Int64Index, Float64Index, etc.)
        # The backing array is directly at index._data
        backing_array = _deepcopy_ndarray(index._data, memo)
        # IMPORTANT: copy=False is required to avoid pd.Index making another copy
        index_copy = pd.Index(backing_array, name=index.name, copy=False)
    elif hasattr(index, '_data') and hasattr(index._data, '_ndarray'):
        # DatetimeIndex, TimedeltaIndex, PeriodIndex
        # These have _data which is a DatetimeArray/TimedeltaArray with _ndarray
        backing_array = _deepcopy_ndarray(index._data._ndarray, memo)
        # Reconstruct the DatetimeArray/TimedeltaArray and then the Index
        # We need to use the proper constructor to preserve timezone, freq, etc.
        # IMPORTANT: copy=False is required to avoid the Index making another copy
        if isinstance(index, pd.DatetimeIndex):
            # DatetimeIndex preserves tz and freq
            index_copy = pd.DatetimeIndex(
                backing_array,
                tz=index.tz,
                freq=index.freq,
                name=index.name,
                copy=False
            )
        elif isinstance(index, pd.TimedeltaIndex):
            index_copy = pd.TimedeltaIndex(
                backing_array,
                freq=index.freq,
                name=index.name,
                copy=False
            )
        elif isinstance(index, pd.PeriodIndex):
            # PeriodIndex needs different handling
            index_copy = index.copy(deep=True)
        else:
            # Fallback for other datetime-like types
            index_copy = index.copy(deep=True)
    elif isinstance(index, pd.CategoricalIndex):
        # CategoricalIndex: copy codes and categories separately
        cat = index._data
        new_codes = _deepcopy_ndarray(cat.codes, memo)
        new_categories = _deepcopy_index(cat.categories, memo)
        index_copy = pd.CategoricalIndex._simple_new(
            pd.Categorical._simple_new(
                new_codes,
                dtype=pd.CategoricalDtype(categories=new_categories, ordered=cat.ordered)
            ),
            name=index.name
        )
    else:
        # Fallback for other Index types
        index_copy = index.copy(deep=True)

    memo[obj_id] = index_copy
    return index_copy


# Register for base Index class - subclasses will also use this handler
# due to MRO lookup in the dispatch logic
d[pd.Index] = _deepcopy_index


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


# =============================================================================
# LightGBM model handler - uses fast serialization for fitted models
# =============================================================================
# LightGBM models (LGBMRegressor, LGBMClassifier, LGBMRanker) have an internal
# Booster object with C++ backing. Standard deepcopy traverses the Python object
# graph inefficiently. Instead, we use model_to_string() for fast serialization.
#
# Key insight: Fitted models are immutable - the tree ensemble doesn't change
# after training, so we can safely use string serialization for copying.

def _deepcopy_lightgbm_model(model, memo: dict[int, Any]):
    """
    Deep copy a LightGBM model by sharing the immutable booster.

    Key insight: Fitted LightGBM models have an immutable tree ensemble.
    The _Booster object cannot be modified after fit() - trees can't be
    added, splits can't be changed. So we can SHARE the booster reference
    and only copy the mutable sklearn wrapper attributes.

    For fitted models (with booster_):
    - SHARES the _Booster reference (it's immutable!)
    - Only copies the sklearn wrapper's __dict__ attributes
    - O(1) for the booster, O(num_attrs) for the wrapper

    For unfitted models:
    - Falls back to standard deepcopy (just copies parameters)

    This is much faster than serialization because we avoid any
    booster copying entirely.
    """
    import copy

    model_id = id(model)
    if model_id in memo:
        return memo[model_id]

    # Check if model is fitted (has a trained booster)
    if not hasattr(model, 'booster_') or model.booster_ is None:
        # Unfitted model - use standard deepcopy
        result = copy.deepcopy(model, memo)
        memo[model_id] = result
        return result

    # Fitted model - SHARE the immutable booster, copy wrapper only
    # Use __new__ to create instance without calling __init__
    new_model = object.__new__(model.__class__)

    # Share the immutable booster reference (this is the key optimization!)
    new_model._Booster = model._Booster

    # Copy all __dict__ attributes except _Booster
    for attr, val in model.__dict__.items():
        if attr == '_Booster':
            continue  # Already shared

        # Deep copy mutable attributes for isolation
        if isinstance(val, (list, dict, np.ndarray)):
            new_model.__dict__[attr] = copy.deepcopy(val, memo)
        else:
            new_model.__dict__[attr] = val

    memo[model_id] = new_model
    return new_model


# Flag to track if we've registered LightGBM handlers (done lazily on first use)
_lightgbm_handlers_registered = False


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


def _register_lightgbm_handlers_if_needed():
    """Register LightGBM model handlers lazily to avoid import-time side effects.

    Less expensive than Keras import (~1s vs ~3s), but still deferred to avoid
    unnecessary loading when LightGBM models aren't used.
    """
    global _lightgbm_handlers_registered
    if _lightgbm_handlers_registered:
        return
    _lightgbm_handlers_registered = True

    try:
        import lightgbm as lgb
        _deepcopy_dispatch[lgb.LGBMRegressor] = _deepcopy_lightgbm_model
        _deepcopy_dispatch[lgb.LGBMClassifier] = _deepcopy_lightgbm_model
        _deepcopy_dispatch[lgb.LGBMRanker] = _deepcopy_lightgbm_model
    except ImportError:
        pass  # LightGBM not installed


def reset_lightgbm_deepcopy_handler():
    """Reset the LightGBM deepcopy handler registration. For testing."""
    global _lightgbm_handlers_registered
    _lightgbm_handlers_registered = False
    # Also remove from dispatch table
    try:
        import lightgbm as lgb
        for cls in (lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.LGBMRanker):
            if cls in _deepcopy_dispatch:
                del _deepcopy_dispatch[cls]
    except ImportError:
        pass


# =============================================================================
# NumPy ndarray handler with cross-checkpoint caching
# =============================================================================


def _fast_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Fast vectorized byte-level array comparison.

    Compares arrays as raw uint8 views to:
    - Handle NaN correctly (byte-identical NaN values compare equal)
    - Leverage numpy's vectorized ``==`` and ``all()``
    - Chunk large arrays for bounded memory (~1 MB bool intermediate)
      and early termination on first mismatched chunk

    For non-contiguous arrays, falls back to ``np.array_equal``.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.size == 0:
        return True

    # Both C-contiguous: compare as raw bytes (fast, NaN-safe)
    if a.flags.c_contiguous and b.flags.c_contiguous:
        try:
            a_bytes = a.view(np.uint8)
            b_bytes = b.view(np.uint8)
        except ValueError:
            # Structured / void dtypes may not support a uint8 view
            return bool(np.array_equal(a, b, equal_nan=True))

        n = a_bytes.size
        chunk = _ARRAY_EQUAL_CHUNK_BYTES

        if n <= chunk:
            # Single vectorized comparison — no Python loop
            return bool(np.all(a_bytes == b_bytes))

        # Chunked: bounded intermediate allocation + early exit
        for i in range(0, n, chunk):
            if not np.all(a_bytes[i:i + chunk] == b_bytes[i:i + chunk]):
                return False
        return True

    # Non-contiguous: fall back to numpy (handles strides, NaN)
    return bool(np.array_equal(a, b, equal_nan=True))


def _deepcopy_ndarray(arr: np.ndarray, memo: dict) -> np.ndarray:
    """
    Deep copy an ndarray with cross-checkpoint caching for numeric arrays.

    For numeric (non-object) dtype arrays:
    - Checks the ndarray cache for an identical copy from a prior checkpoint
    - If the source array is unchanged (fast byte-level comparison): reuses
      the cached copy — zero allocation, zero memcpy
    - If changed or not cached: creates a new copy via ``arr.copy()`` and
      updates the cache

    For object-dtype arrays:
    - Delegates to numpy's built-in ``__deepcopy__`` which recurses into
      each element (existing behaviour, unchanged)

    The cache is keyed by ``id(original)`` and validated by shape / dtype /
    nbytes metadata plus a fast vectorized content comparison.
    """
    obj_id = id(arr)

    # Already copied in this deepcopy pass (alias dedup within one checkpoint)
    if obj_id in memo:
        return memo[obj_id]

    # Object-dtype arrays may contain mutable Python objects — skip cache,
    # delegate to numpy's built-in deepcopy for correct element recursion
    if arr.dtype == object:
        y = arr.__deepcopy__(memo)
        memo[obj_id] = y
        return y

    # --- Cross-checkpoint cache lookup ---
    if obj_id in _ndarray_cache:
        cached_copy, cached_shape, cached_dtype, cached_nbytes = _ndarray_cache[obj_id]
        if (arr.shape == cached_shape
                and arr.dtype == cached_dtype
                and arr.nbytes == cached_nbytes
                and _fast_array_equal(arr, cached_copy)):
            log(f"[deepcopy] ndarray cache hit "
                f"({arr.nbytes / (1024 * 1024):.1f} MB, {arr.shape} {arr.dtype})")
            memo[obj_id] = cached_copy
            return cached_copy

    # --- Cache miss: full copy ---
    y = arr.copy()

    # Cache arrays >= 1 KB (avoids overhead for tiny scalars / 0-d arrays)
    if arr.nbytes >= 1024:
        if len(_ndarray_cache) > _MAX_NDARRAY_CACHE_SIZE:
            _ndarray_cache.clear()
            _ndarray_copies.clear()
            log(f"[deepcopy] ndarray cache exceeded {_MAX_NDARRAY_CACHE_SIZE} entries, cleared")

        _ndarray_cache[obj_id] = (y, arr.shape, arr.dtype, arr.nbytes)
        _ndarray_copies[id(y)] = y

    memo[obj_id] = y

    if arr.nbytes > 1024 * 1024:
        log(f"[deepcopy] ndarray cache miss, copied "
            f"{arr.nbytes / (1024 * 1024):.1f} MB {arr.shape} {arr.dtype}")

    return y


d[np.ndarray] = _deepcopy_ndarray

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
