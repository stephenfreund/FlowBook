# cuDF Checkpoint Deduplication Fix Plan

## Problem Statement

When checkpointing cudf DataFrames that share GPU memory (e.g., views, subsets), each DataFrame is converted to pandas independently via `to_pandas()`. This creates separate numpy arrays with **identical content but no memory sharing**, causing checkpoint sizes to be much larger than the actual unique data.

### Example
```python
train = cudf.read_csv(...)           # 1000 MB on GPU
X_train = train.iloc[:80%]           # View - shares GPU memory (no extra memory)
X_valid = train.iloc[80%:]           # View - shares GPU memory (no extra memory)

# After to_pandas() conversion in checkpoint:
train_pd = train.to_pandas()         # 1000 MB (new numpy arrays)
X_train_pd = X_train.to_pandas()     # 800 MB (separate numpy arrays, SAME CONTENT)
X_valid_pd = X_valid.to_pandas()     # 200 MB (separate numpy arrays, SAME CONTENT)
# Total: 2000 MB instead of 1000 MB
```

### Impact
- 48x memory overhead ratio observed in `feature-engineering-with-rapids` notebook
- Checkpoint stores 4182 MB for data that should be ~460 MB
- Makes FlowBook impractical for cudf-heavy notebooks

## Root Cause Analysis

1. **cudf views share GPU memory**: `train.iloc[...]`, `train.loc[...]` create views that reference the same GPU buffers

2. **`to_pandas()` allocates fresh memory**: Each call allocates new numpy arrays, losing the sharing relationship

3. **HeapSizer deduplication doesn't help**: It uses `np.shares_memory()` which only detects actual memory sharing, not equal content

4. **Current cudf cache is ID-based**: `CuDFCheckpointCache` keys by `id(obj)`, so different cudf objects (even if they share GPU data) get independent conversions

## Proposed Solution

### Approach: GPU Buffer Tracking

Track which GPU memory buffers back each cudf column, and reuse pandas arrays when the same buffer is encountered.

### Implementation Steps

#### Phase 1: GPU Buffer Detection

Add buffer tracking to `cudf_compat.py`:

```python
def get_gpu_buffer_ids(df: cudf.DataFrame) -> Dict[str, int]:
    """
    Get GPU buffer IDs for each column in a cudf DataFrame.

    Columns sharing the same buffer will have the same ID.
    This allows detecting when multiple cudf objects share underlying data.
    """
    buffer_ids = {}
    for col_name in df.columns:
        col = df._data[col_name]
        # cudf columns have a data buffer with a unique pointer
        if hasattr(col, 'data') and hasattr(col.data, 'ptr'):
            buffer_ids[col_name] = col.data.ptr
        elif hasattr(col, '_column') and hasattr(col._column, 'data'):
            buffer_ids[col_name] = col._column.data.ptr
    return buffer_ids
```

#### Phase 2: Shared Conversion Cache

Modify `CuDFCheckpointCache` to cache by GPU buffer, not object ID:

```python
class CuDFCheckpointCache:
    def __init__(self):
        # Existing caches
        self._cache: Dict[int, tuple] = {}
        self._deepcopy_cache: Dict[int, Any] = {}

        # NEW: Buffer-based cache for column arrays
        # Maps GPU buffer ptr -> (pandas array, slice info)
        self._buffer_cache: Dict[int, np.ndarray] = {}

    def get_or_convert_with_sharing(self, obj: cudf.DataFrame) -> pd.DataFrame:
        """
        Convert cudf DataFrame to pandas, reusing arrays for shared buffers.
        """
        buffer_ids = get_gpu_buffer_ids(obj)

        result_columns = {}
        for col_name, buf_ptr in buffer_ids.items():
            if buf_ptr in self._buffer_cache:
                # Reuse existing array (possibly as a view)
                cached_array = self._buffer_cache[buf_ptr]
                # Create view if this is a subset
                result_columns[col_name] = self._create_view_if_subset(
                    cached_array, obj[col_name]
                )
            else:
                # Convert this column and cache
                pandas_col = obj[col_name].to_pandas()
                self._buffer_cache[buf_ptr] = pandas_col.values
                result_columns[col_name] = pandas_col.values

        return pd.DataFrame(result_columns, index=obj.index.to_pandas())
```

#### Phase 3: View Creation for Subsets

When a cudf column is a view of a larger buffer, create a numpy view:

```python
def _create_view_if_subset(self, cached_array: np.ndarray,
                            cudf_col: cudf.Series) -> np.ndarray:
    """
    If cudf_col is a view/subset of the cached array's source,
    return a numpy view instead of copying.
    """
    # Get offset and length from cudf column
    if hasattr(cudf_col, '_column'):
        col = cudf_col._column
        if hasattr(col, 'offset') and col.offset > 0:
            # This is a view starting at an offset
            start = col.offset
            end = start + len(cudf_col)
            return cached_array[start:end]

    # Check if lengths match
    if len(cached_array) == len(cudf_col):
        return cached_array

    # Subset - find the matching slice
    # This may require content comparison for non-contiguous subsets
    return cached_array[:len(cudf_col)]
```

#### Phase 4: Integration with deepcopy_cudf

Modify `deepcopy_cudf()` to use the buffer-aware conversion:

```python
def deepcopy_cudf(obj: Any, memo: Dict[int, Any]) -> Any:
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    cache = get_checkpoint_cache()

    # Use buffer-aware conversion for DataFrames
    if is_cudf_dataframe(obj) or _is_proxy_dataframe(obj):
        pandas_df = cache.get_or_convert_with_sharing(obj)
    else:
        pandas_df = to_pandas_cached(obj)

    # Shallow copy is now safe because arrays may be shared
    # CoW protects against mutation
    result = pandas_df.copy(deep=False)
    memo[obj_id] = result
    return result
```

### Alternative Approaches Considered

#### A. Content-Based Hashing (Rejected)
- Hash array contents to detect duplicates
- **Rejected**: Too slow for large arrays (must read all data)

#### B. Track Derivation Relationships (Rejected)
- Monitor `iloc`, `loc` calls to track parent-child relationships
- **Rejected**: Complex, fragile, doesn't handle all cases

#### C. Post-Conversion Deduplication (Partial)
- After converting all objects, compare arrays and deduplicate
- **Issue**: Still requires content comparison; O(n²) comparisons
- **Possible enhancement**: Use as fallback for edge cases

### Testing Strategy

1. **Unit tests** (added in `test_cudf_checkpoint_dedup.py`):
   - Verify buffer ID detection works
   - Verify shared conversion produces views
   - Verify checkpoint size is close to unique data size

2. **Integration tests**:
   - Run `feature-engineering-with-rapids` notebook
   - Verify checkpoint overhead ratio < 2x (not 48x)

3. **Edge cases**:
   - Non-contiguous subsets (fancy indexing)
   - Column subsets (`df[['a', 'b']]`)
   - Multiple levels of derivation (`X = train.iloc[...]; Y = X.iloc[...]`)
   - Mixed cudf/pandas in same checkpoint

### Rollout Plan

1. **Phase 1**: Add buffer tracking, keep existing behavior as fallback
2. **Phase 2**: Enable buffer-aware caching behind feature flag
3. **Phase 3**: Run benchmarks on cudf-heavy notebooks
4. **Phase 4**: Enable by default, remove fallback

### Success Criteria

- Checkpoint size for overlapping cudf DataFrames ≤ 1.5x unique data size
- No regression in checkpoint/restore correctness
- No significant performance regression for non-cudf notebooks

### Files to Modify

1. `flowbook/kernel_support/cudf_compat.py`:
   - Add `get_gpu_buffer_ids()`
   - Modify `CuDFCheckpointCache` with buffer-aware caching
   - Modify `deepcopy_cudf()` to use buffer-aware conversion

2. `flowbook/kernel_support/tests/test_cudf_checkpoint_dedup.py`:
   - Already created with xfail tests

3. `flowbook/kernel_support/heap_size.py`:
   - No changes needed (existing deduplication will work once arrays share memory)

### Estimated Effort

- Phase 1 (Buffer Detection): 2-3 hours
- Phase 2 (Shared Conversion): 4-6 hours
- Phase 3 (View Creation): 2-3 hours
- Phase 4 (Integration): 1-2 hours
- Testing & Edge Cases: 4-6 hours
- **Total**: 2-3 days
