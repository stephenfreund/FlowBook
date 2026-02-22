# Plan: Skip Re-Deepcopy for Cached cuDF Objects

## Problem Statement

In `deepcopy_cudf()`, even when the cudf cache returns a HIT (unchanged data), we still call `flowbook_deepcopy(pandas_copy)`. This is wasteful:

```python
def deepcopy_cudf(obj, memo):
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    pandas_copy = to_pandas_cached(obj)  # Cache HIT - returns cached pandas
    result = flowbook_deepcopy(pandas_copy, memo)  # WASTEFUL - deepcopies again!

    memo[obj_id] = result
    return result
```

**Impact:** For a 1.5 GB DataFrame, cache HIT should be ~10ms but is currently ~600ms due to the redundant deepcopy.

## Solution Design

### Core Insight

The cudf cache currently stores the **raw GPU→pandas conversion**. Multiple cache HITs return the **same pandas object**, so we can't skip deepcopy without risking shared state between checkpoints.

**Solution:** Store the **already-deepcopied pandas** in the cache. On cache HIT, return a shallow copy (safe with CoW).

### Architecture

```
Current Flow:
  cudf_obj → [cudf cache: stores raw pandas] → raw_pandas → deepcopy → result

New Flow:
  cudf_obj → [cudf cache: stores deepcopied pandas] → deepcopied_pandas → shallow_copy → result
```

### Note on cudf's Copy-on-Write

cudf has its own CoW mechanism at the GPU level (`ExposureTrackedBuffer`, `BufferOwner`) separate from pandas CoW. When we call `.to_pandas()` or access `_fsproxy_slow`, we get a CPU copy that is independent of the GPU buffers. Our caching operates on the pandas side, so cudf's GPU-level CoW doesn't affect our optimization.

### Performance Comparison

| Operation | Current (Cache HIT) | Optimized (Cache HIT) |
|-----------|--------------------|-----------------------|
| GPU→CPU transfer | Skipped (cached) | Skipped (cached) |
| pandas deepcopy | ~600ms for 1.5GB | Skipped |
| Shallow copy | N/A | O(1) ~0.1ms |
| Index copy | Included in deepcopy | Skipped (immutable) |
| **Total** | **~600ms** | **<1ms** |

## Implementation Steps

### Step 1: Add `was_cache_hit` Tracking to `CuDFCheckpointCache`

**File:** `flowbook/kernel_support/cudf_compat.py`

Add a method to check cache without converting:

```python
class CuDFCheckpointCache:
    def __init__(self):
        self._cache: Dict[int, tuple] = {}
        self._deepcopy_cache: Dict[int, Any] = {}  # NEW: stores deepcopied versions

    def has_valid_cache(self, obj: Any) -> bool:
        """Check if we have a valid cached copy without triggering conversion."""
        return self.get_cached(obj) is not None

    def get_deepcopied(self, obj_id: int) -> Optional[Any]:
        """Get the cached deepcopied version if available."""
        return self._deepcopy_cache.get(obj_id)

    def cache_deepcopy(self, obj_id: int, deepcopied: Any) -> None:
        """Cache the deepcopied version for future use."""
        self._deepcopy_cache[obj_id] = deepcopied

    def clear(self) -> None:
        """Clear both caches."""
        self._cache.clear()
        self._deepcopy_cache.clear()
```

### Step 2: Modify `deepcopy_cudf` to Use Two-Level Caching

**File:** `flowbook/kernel_support/cudf_compat.py`

```python
def deepcopy_cudf(obj: Any, memo: Dict[int, Any]) -> Any:
    """
    Deep copy a cuDF object by converting to pandas (CPU memory).

    Uses two-level caching:
    1. cudf cache: GPU→pandas conversion (avoids GPU transfer)
    2. deepcopy cache: pandas deepcopy result (avoids redundant deepcopy)

    On cache HIT, returns a shallow copy of the cached deepcopy (fast with CoW).
    """
    obj_id = id(obj)

    # Level 0: Check memo (same checkpoint, same object)
    if obj_id in memo:
        return memo[obj_id]

    cache = get_checkpoint_cache()

    # Level 1: Check if cudf cache has valid entry (data unchanged)
    is_cache_hit = cache.has_valid_cache(obj)

    if is_cache_hit:
        # Level 2: Check if we have a cached deepcopy
        cached_deepcopy = cache.get_deepcopied(obj_id)

        if cached_deepcopy is not None:
            # Fast path: O(1) shallow copy of cached deepcopy
            # Safe because: CoW protects data, Index is immutable, cache is read-only
            result = _shallow_copy_for_checkpoint(cached_deepcopy)
            memo[obj_id] = result
            return result

    # Cache MISS or no deepcopy cached - do full conversion + deepcopy
    pandas_copy = to_pandas_cached(obj)

    from flowbook.kernel_support.deepcopy import deepcopy as flowbook_deepcopy
    result = flowbook_deepcopy(pandas_copy, memo)

    # Cache the deepcopy for future HITs
    if is_cache_hit or cache.has_valid_cache(obj):
        cache.cache_deepcopy(obj_id, result)

    memo[obj_id] = result
    return result


def _shallow_copy_for_checkpoint(df: Any) -> Any:
    """
    Create a shallow copy suitable for checkpointing.

    This is safe because:
    1. pandas CoW: data buffers are copy-on-write, mutations create new arrays
    2. pandas Index is IMMUTABLE: "Index does not support mutable operations"
    3. cudf object columns: contain only immutable strings/tuples (per RAPIDS docs)
    4. Cached copy is read-only: only used for diff/comparison, never modified

    Shallow copy shares Index object with original, which is fine since Index
    is immutable. This makes the copy O(1) - just creates new DataFrame wrapper.
    """
    import pandas as pd

    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df.copy(deep=False)

    elif isinstance(df, pd.Index):
        # Index is immutable, but copy() returns same type
        return df.copy()

    else:
        return df
```

### Step 3: Handle Cache Invalidation

When data changes, the cudf cache fingerprint check fails, invalidating the entry. We need to also invalidate the deepcopy cache:

```python
def get_cached(self, obj: Any) -> Optional[Any]:
    """Get cached pandas copy if object hasn't changed."""
    obj_id = id(obj)

    if obj_id not in self._cache:
        return None

    cached_fp, pandas_copy, weak_ref = self._cache[obj_id]

    # Verify weak reference still points to same object
    ref_obj = weak_ref()
    if ref_obj is None or ref_obj is not obj:
        del self._cache[obj_id]
        self._deepcopy_cache.pop(obj_id, None)  # NEW: invalidate deepcopy cache too
        return None

    # Verify fingerprint matches
    current_fp = self._fingerprint(obj)
    if current_fp != cached_fp:
        del self._cache[obj_id]
        self._deepcopy_cache.pop(obj_id, None)  # NEW: invalidate deepcopy cache too
        return None

    return pandas_copy
```

### Step 4: Add Proxy Fingerprinting (Bonus Fix)

For correctness, also fix proxy fingerprinting:

```python
def _fingerprint(self, obj: Any) -> Optional[tuple]:
    """Compute fingerprint for cudf object (native or proxy)."""

    # Handle native cudf types (existing code)
    if is_cudf_dataframe(obj):
        # ... existing implementation ...

    # NEW: Handle cudf.pandas proxy objects
    if _is_proxy_dataframe(obj):
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

    if _is_proxy_series(obj):
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

    return None
```

## Testing Plan

### Unit Tests

1. **Test cache HIT returns shallow copy:**
   ```python
   def test_cache_hit_returns_shallow_copy():
       gdf = cudf.DataFrame({'a': [1, 2, 3]})
       memo1 = {}
       result1 = deepcopy_cudf(gdf, memo1)

       memo2 = {}  # New memo
       result2 = deepcopy_cudf(gdf, memo2)

       # Results should be independent
       result1['a'] = [10, 20, 30]
       assert result2['a'].tolist() == [1, 2, 3]
   ```

2. **Test cache invalidation on data change:**
   ```python
   def test_cache_invalidated_on_change():
       gdf = cudf.DataFrame({'a': [1, 2, 3]})
       memo1 = {}
       result1 = deepcopy_cudf(gdf, memo1)

       gdf['a'] = [100, 200, 300]  # Modify data

       memo2 = {}
       result2 = deepcopy_cudf(gdf, memo2)

       assert result2['a'].tolist() == [100, 200, 300]
   ```

3. **Performance test:**
   ```python
   def test_cache_hit_performance():
       gdf = cudf.DataFrame({f'c{i}': np.random.randn(1_000_000) for i in range(30)})

       # First call - cache miss
       memo1 = {}
       t1 = time.perf_counter()
       deepcopy_cudf(gdf, memo1)
       first_time = time.perf_counter() - t1

       # Second call - cache hit
       memo2 = {}
       t2 = time.perf_counter()
       deepcopy_cudf(gdf, memo2)
       second_time = time.perf_counter() - t2

       # Cache hit should be >10x faster
       assert second_time < first_time / 10
   ```

### Integration Tests

1. Run existing cudf tests: `pytest flowbook/kernel_support/tests/test_cudf_compat.py`
2. Run checkpoint tests with cudf: `pytest flowbook/kernel_support/tests/test_cudf_checkpoint_perf.py`
3. Verify RAPIDS notebook overhead improves

## Files to Modify

1. `flowbook/kernel_support/cudf_compat.py`:
   - Add `_deepcopy_cache` to `CuDFCheckpointCache`
   - Add `has_valid_cache()`, `get_deepcopied()`, `cache_deepcopy()` methods
   - Modify `get_cached()` to invalidate deepcopy cache
   - Modify `clear()` to clear both caches
   - Modify `deepcopy_cudf()` to use two-level caching
   - Add `_shallow_copy_for_checkpoint()` helper
   - Fix `_fingerprint()` for proxy objects

2. `flowbook/kernel_support/tests/test_cudf_compat.py`:
   - Add tests for two-level caching
   - Add performance regression tests

## Rollback Plan

If issues arise:
1. Set `FLOWBOOK_DISABLE_CUDF_DEEPCOPY_CACHE=1` environment variable
2. Check for this in `deepcopy_cudf()` and skip optimization

```python
import os
_DISABLE_DEEPCOPY_CACHE = os.environ.get('FLOWBOOK_DISABLE_CUDF_DEEPCOPY_CACHE', '0') == '1'

def deepcopy_cudf(obj, memo):
    if _DISABLE_DEEPCOPY_CACHE:
        # Original behavior
        pandas_copy = to_pandas_cached(obj)
        return flowbook_deepcopy(pandas_copy, memo)
    # ... optimized path ...
```

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Cache HIT deepcopy time (1.5 GB) | ~600ms | <1ms |
| Total checkpoint overhead per cell | ~15s | ~1-2s (first cell), <0.5s (subsequent) |
| Memory usage | Same | Same (CoW) |

## Risks

1. **CoW not enabled:** Shallow copy would share data. Mitigation: Check CoW is enabled at startup, warn if not. (FlowBook already enables CoW in `memory_checkpoint.py`)

2. **Index mutations:** Some code mutates index in place. Mitigation: Always deep copy index.

3. **cudf spilling:** With `CUDF_SPILL=on`, cudf may spill GPU buffers to host memory. This shouldn't affect our caching since we work with the pandas conversion, not the GPU buffers directly.

## Why Object Columns Are Safe

Per [RAPIDS Library Design docs](https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/):

> "cuDF does not support the arbitrary `object` dtype"

cudf stores data in Arrow format:
- **Strings**: `StringColumn` with UTF-8 buffer + offsets (immutable when converted to pandas)
- **Lists**: `ListDtype` with homogeneous elements (becomes tuples - immutable)
- **Structs**: `StructDtype` (becomes dicts with immutable values)

Therefore, object columns from cudf→pandas conversion contain **only immutable objects**, making shallow copy always safe.
