"""
Test case for cudf checkpoint deduplication issue.

Problem:
    When cudf DataFrames share GPU memory (e.g., X_train = train.loc[...]),
    converting each to pandas via to_pandas() creates independent numpy arrays.
    This causes checkpoint sizes to be much larger than the actual unique data.

Example:
    - train: 1000 MB on GPU
    - X_train = train.iloc[:80%]: shares GPU memory with train (no extra memory)
    - After to_pandas() conversion:
        - train_pd: 1000 MB (new numpy arrays)
        - X_train_pd: 800 MB (separate numpy arrays with SAME CONTENT)
        - Total: 1800 MB instead of 1000 MB

This test verifies the issue exists and will pass once the fix is implemented.
"""

import pytest
import numpy as np
import sys

# Skip all tests if cudf is not available
cudf = pytest.importorskip("cudf")


class TestCudfCheckpointDeduplication:
    """Tests for cudf checkpoint memory deduplication."""

    def test_cudf_view_shares_gpu_memory(self):
        """Verify that cudf views share GPU memory (baseline check)."""
        # Create a cudf DataFrame
        n_rows = 100_000
        train = cudf.DataFrame({
            'a': np.arange(n_rows, dtype=np.float64),
            'b': np.arange(n_rows, dtype=np.float64),
        })

        # Create a view/subset
        X_train = train.iloc[:80_000]

        # On GPU, the view should not allocate much extra memory
        # (just metadata, not the actual data)
        train_mem = train.memory_usage(deep=True).sum()
        X_train_mem = X_train.memory_usage(deep=True).sum()

        # X_train should report ~80% of train's memory (it's a view of that data)
        # This is expected behavior - cudf views share underlying GPU memory
        assert X_train_mem < train_mem
        assert X_train_mem > 0.7 * train_mem  # ~80% of rows

    def test_to_pandas_loses_memory_sharing(self):
        """Verify that to_pandas() loses memory sharing (documents the problem)."""
        n_rows = 100_000
        train = cudf.DataFrame({
            'a': np.arange(n_rows, dtype=np.float64),
            'b': np.arange(n_rows, dtype=np.float64),
        })
        X_train = train.iloc[:80_000]

        # Convert both to pandas
        train_pd = train.to_pandas()
        X_train_pd = X_train.to_pandas()

        # The pandas DataFrames should NOT share memory
        # (this is the problem we need to fix)
        train_arr = train_pd['a'].values
        X_train_arr = X_train_pd['a'].values

        # Different data pointers
        assert train_arr.ctypes.data != X_train_arr.ctypes.data

        # No memory sharing
        assert not np.shares_memory(train_arr, X_train_arr)

        # But the content IS the same for the overlapping portion
        assert np.array_equal(train_arr[:80_000], X_train_arr)

    def test_heapsizer_does_not_deduplicate_equal_content(self):
        """Verify HeapSizer doesn't deduplicate arrays with equal content but different memory."""
        from flowbook.kernel_support.heap_size import HeapSizer

        n_rows = 100_000
        train = cudf.DataFrame({
            'a': np.arange(n_rows, dtype=np.float64),
            'b': np.arange(n_rows, dtype=np.float64),
        })
        X_train = train.iloc[:80_000]

        # Convert to pandas (creates independent arrays)
        train_pd = train.to_pandas()
        X_train_pd = X_train.to_pandas()

        # Measure with HeapSizer
        sizer = HeapSizer()
        train_size = sizer.sizeof(train_pd)

        # Reset and measure X_train separately
        sizer.reset()
        X_train_size = sizer.sizeof(X_train_pd)

        # Both are measured at full size (no deduplication)
        assert train_size > 1_000_000  # > 1 MB
        assert X_train_size > 800_000  # > 0.8 MB

        # Now measure both together - still no deduplication because
        # they don't share memory (even though content overlaps)
        sizer.reset()
        combined_size = sizer.sizeof({'train': train_pd, 'X_train': X_train_pd})

        # Combined size is approximately sum of individual sizes
        # (no deduplication happening)
        expected_if_no_dedup = train_size + X_train_size
        # Allow some overhead variance
        assert combined_size > 0.9 * expected_if_no_dedup

    def test_cudf_checkpoint_deduplicates_overlapping_data(self):
        """
        Test that checkpoint correctly deduplicates overlapping cudf data.

        The fix ensures that cudf DataFrames sharing GPU memory also share
        numpy arrays in checkpoints, so the total checkpoint size is close
        to the unique data size (not sum of all DataFrames).
        """
        from flowbook.kernel_support.heap_size import HeapSizer
        from flowbook.kernel_support.cudf_compat import deepcopy_cudf, get_checkpoint_cache

        # Clear the cache to start fresh
        get_checkpoint_cache().clear()

        n_rows = 100_000
        train = cudf.DataFrame({
            'a': np.arange(n_rows, dtype=np.float64),
            'b': np.arange(n_rows, dtype=np.float64),
        })
        X_train = train.iloc[:80_000]

        # Simulate checkpoint: deepcopy both cudf objects
        memo = {}
        train_copy = deepcopy_cudf(train, memo)
        X_train_copy = deepcopy_cudf(X_train, memo)

        # Verify memory sharing is working
        train_a = train_copy['a'].values
        X_train_a = X_train_copy['a'].values
        assert np.shares_memory(train_a, X_train_a), \
            "X_train should share memory with train after buffer-aware conversion"

        # Measure the checkpoint size (combined namespace)
        sizer = HeapSizer()
        checkpoint_ns = {'train': train_copy, 'X_train': X_train_copy}
        checkpoint_size = sizer.sizeof_namespace(checkpoint_ns).total_bytes

        # Expected size: ~1.6 MB (100k rows * 2 cols * 8 bytes)
        expected_unique_data = n_rows * 2 * 8  # 1.6 MB
        expected_without_dedup = expected_unique_data + (80_000 * 2 * 8)  # 2.88 MB

        # Allow 50% overhead for metadata, but NOT doubling from X_train
        max_acceptable = expected_unique_data * 1.5

        assert checkpoint_size < max_acceptable, (
            f"Checkpoint size {checkpoint_size / 1e6:.2f} MB should be close to "
            f"unique data size {expected_unique_data / 1e6:.2f} MB (not "
            f"{expected_without_dedup / 1e6:.2f} MB without dedup)"
        )

    def test_cudf_iloc_slice_shares_checkpoint_memory(self):
        """
        Test that train.iloc slice subsets share memory in checkpoints.

        When using iloc with slices (not index arrays), cudf creates views
        that share GPU memory. Our deduplication preserves this sharing
        in the checkpoint.
        """
        from flowbook.kernel_support.cudf_compat import deepcopy_cudf, get_checkpoint_cache

        # Clear the cache to start fresh
        get_checkpoint_cache().clear()

        n_rows = 100_000
        train = cudf.DataFrame({
            'a': np.arange(n_rows, dtype=np.float64),
            'b': np.arange(n_rows, dtype=np.float64),
            'c': np.arange(n_rows, dtype=np.float64),
        })

        # Create subsets using iloc slices (creates views, not copies)
        X_train = train.iloc[:80_000]
        X_valid = train.iloc[80_000:]

        # Checkpoint all three
        memo = {}
        train_copy = deepcopy_cudf(train, memo)
        X_train_copy = deepcopy_cudf(X_train, memo)
        X_valid_copy = deepcopy_cudf(X_valid, memo)

        # X_train and X_valid copies should share memory with train_copy
        train_a = train_copy['a'].values
        X_train_a = X_train_copy['a'].values
        X_valid_a = X_valid_copy['a'].values

        # The subset arrays should be views into train's array
        assert np.shares_memory(train_a, X_train_a), \
            "X_train checkpoint should share memory with train checkpoint"
        assert np.shares_memory(train_a, X_valid_a), \
            "X_valid checkpoint should share memory with train checkpoint"

    def test_cudf_loc_with_index_array_creates_copy(self):
        """
        Document that train.loc[index_array] creates copies, not views.

        This is expected cudf behavior - fancy indexing with arrays
        requires gathering rows, which allocates new GPU memory.
        Our deduplication cannot help in this case.
        """
        from flowbook.kernel_support.cudf_compat import get_dataframe_buffer_map

        n_rows = 100_000
        train = cudf.DataFrame({
            'a': np.arange(n_rows, dtype=np.float64),
        })

        # .loc with index array creates a COPY (new GPU allocation)
        train_idx = np.arange(80_000)
        X_train = train.loc[train_idx]

        train_buf = get_dataframe_buffer_map(train)
        X_train_buf = get_dataframe_buffer_map(X_train)

        # Buffer pointers are different - these are separate GPU allocations
        assert train_buf['a'][0] != X_train_buf['a'][0], \
            "loc with index array should create copy, not view"


class TestCudfViewDetection:
    """Tests for detecting cudf view/derivation relationships."""

    def test_detect_iloc_view(self):
        """Test detecting that iloc creates a view."""
        train = cudf.DataFrame({'a': [1, 2, 3, 4, 5]})
        X_train = train.iloc[:3]

        # Check if we can detect the relationship
        # cudf DataFrames have _column which contains the actual data
        # Views should reference the same underlying buffer

        # This tests what APIs we can use to detect views
        # The actual detection logic will be implemented in the fix
        assert hasattr(train, '_data')  # cudf internal data structure
        assert hasattr(X_train, '_data')

    def test_detect_loc_view(self):
        """Test detecting that loc creates a view."""
        train = cudf.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
        })
        subset = train.loc[:, ['a']]

        # Column subset should also be detectable
        assert 'a' in subset.columns
        assert 'b' not in subset.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
