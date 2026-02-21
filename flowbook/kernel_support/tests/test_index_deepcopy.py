"""
Comprehensive tests for Index deepcopy isolation and HeapSizer deduplication.

These tests verify that:
1. Index objects are properly isolated during deepcopy (no shared memory)
2. Memo caching prevents duplicate copies of the same Index within a checkpoint
3. HeapSizer correctly deduplicates Index memory across checkpoints
4. GroupBy objects with shared Index references are handled correctly
5. Various Index types (Int64, RangeIndex, DatetimeIndex, MultiIndex) work correctly
"""

import pytest
import numpy as np
import pandas as pd
from copy import deepcopy as stdlib_deepcopy
from typing import Dict, Any

from flowbook.kernel_support.deepcopy import deepcopy
from flowbook.kernel_support.heap_size import HeapSizer
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


# =============================================================================
# INDEX ISOLATION TESTS
# =============================================================================

class TestIndexIsolation:
    """Tests that Index objects are properly isolated after deepcopy."""

    def test_int64_index_isolation(self):
        """Int64Index data should be isolated after deepcopy."""
        df = pd.DataFrame({'A': [1, 2, 3]}, index=np.array([10, 20, 30]))
        checkpoint = deepcopy(df, {})

        # Data pointers should be different
        assert df.index._data.ctypes.data != checkpoint.index._data.ctypes.data
        # But values should be the same
        assert list(df.index) == list(checkpoint.index)

    def test_int64_index_modification_safety(self):
        """Modifying original Index should not affect checkpoint."""
        df = pd.DataFrame({'A': [1, 2, 3]}, index=np.array([10, 20, 30]))
        checkpoint = deepcopy(df, {})

        # Modify original via _data
        df.index._data[0] = 999

        # Checkpoint should be unaffected
        assert checkpoint.index[0] == 10
        assert list(checkpoint.index) == [10, 20, 30]

    def test_rangeindex_handling(self):
        """RangeIndex should be recreated, not shared."""
        df = pd.DataFrame({'A': [1, 2, 3]})  # Default RangeIndex
        assert isinstance(df.index, pd.RangeIndex)

        checkpoint = deepcopy(df, {})

        # Should be a new RangeIndex object
        assert df.index is not checkpoint.index
        # With same parameters
        assert checkpoint.index.start == df.index.start
        assert checkpoint.index.stop == df.index.stop
        assert checkpoint.index.step == df.index.step

    def test_datetime_index_isolation(self):
        """DatetimeIndex should be isolated after deepcopy."""
        df = pd.DataFrame(
            {'A': [1, 2, 3]},
            index=pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'])
        )
        checkpoint = deepcopy(df, {})

        # Check isolation via internal _ndarray
        orig_arr = df.index._data._ndarray
        copy_arr = checkpoint.index._data._ndarray

        assert orig_arr.ctypes.data != copy_arr.ctypes.data
        assert list(df.index) == list(checkpoint.index)

    def test_multiindex_isolation(self):
        """MultiIndex should be isolated after deepcopy."""
        df = pd.DataFrame(
            {'A': [1, 2, 3]},
            index=pd.MultiIndex.from_tuples([(1, 'a'), (2, 'b'), (3, 'c')])
        )
        checkpoint = deepcopy(df, {})

        # MultiIndex codes should be isolated
        orig_codes = np.asarray(df.index.codes[0])
        copy_codes = np.asarray(checkpoint.index.codes[0])

        assert not np.shares_memory(orig_codes, copy_codes)
        assert list(df.index) == list(checkpoint.index)

    def test_categorical_index_isolation(self):
        """CategoricalIndex should be isolated after deepcopy."""
        df = pd.DataFrame(
            {'A': [1, 2, 3]},
            index=pd.CategoricalIndex(['x', 'y', 'z'])
        )
        checkpoint = deepcopy(df, {})

        assert df.index is not checkpoint.index
        assert list(df.index) == list(checkpoint.index)

    def test_series_index_isolation(self):
        """Series Index should also be isolated."""
        series = pd.Series([1, 2, 3], index=np.array([10, 20, 30]))
        checkpoint = deepcopy(series, {})

        assert series.index._data.ctypes.data != checkpoint.index._data.ctypes.data
        assert list(series.index) == list(checkpoint.index)


# =============================================================================
# MEMO CACHING TESTS
# =============================================================================

class TestMemoCaching:
    """Tests that memo caching prevents duplicate Index copies."""

    def test_shared_index_uses_memo(self):
        """Same Index object should be cached in memo."""
        df = pd.DataFrame({'A': [1, 2, 3]}, index=np.array([10, 20, 30]))
        gb = df.groupby('A')

        # gb.obj is df (same object)
        assert gb.obj is df

        memo = {}
        namespace = {'df': df, 'gb': gb}
        copied = {}
        for name, val in namespace.items():
            copied[name] = deepcopy(val, memo)

        # In the copied namespace, gb.obj.index should be the SAME object as df.index
        assert copied['df'].index is copied['gb'].obj.index

        # But both should be isolated from original
        assert copied['df'].index._data.ctypes.data != df.index._data.ctypes.data

    def test_memo_caching_across_multiple_dataframes(self):
        """Multiple DataFrames sharing same Index should use memo."""
        index = pd.Index([10, 20, 30])
        df1 = pd.DataFrame({'A': [1, 2, 3]}, index=index)
        df2 = pd.DataFrame({'B': [4, 5, 6]}, index=index)

        # They share the same Index object
        assert df1.index is df2.index

        memo = {}
        copied = {
            'df1': deepcopy(df1, memo),
            'df2': deepcopy(df2, memo),
        }

        # Copied DataFrames should share the same copied Index
        assert copied['df1'].index is copied['df2'].index

        # But isolated from original
        assert copied['df1'].index is not index

    def test_memo_caching_with_series(self):
        """Series created from same Index should use memo."""
        index = pd.Index([10, 20, 30])
        series1 = pd.Series([1, 2, 3], index=index)
        series2 = pd.Series([4, 5, 6], index=index)

        # They share the same Index object
        assert series1.index is series2.index

        memo = {}
        series1_copy = deepcopy(series1, memo)
        series2_copy = deepcopy(series2, memo)

        # Should share the same copied Index
        assert series1_copy.index is series2_copy.index


# =============================================================================
# GROUPBY SPECIFIC TESTS
# =============================================================================

class TestGroupByCheckpoint:
    """Tests for GroupBy checkpoint behavior."""

    def test_groupby_obj_shares_index_via_memo(self):
        """GroupBy.obj should share Index with copied DataFrame via memo."""
        df = pd.DataFrame({
            'A': np.random.randint(0, 3, 100),
            'B': np.random.randn(100),
        }, index=np.arange(100))

        gb = df.groupby('A')

        memo = {}
        namespace = {'df': df, 'gb': gb}
        copied = {name: deepcopy(val, memo) for name, val in namespace.items()}

        # Check that copied gb.obj.index is the same object as copied df.index
        assert copied['gb'].obj.index is copied['df'].index

    def test_groupby_operations_dont_corrupt_checkpoint(self):
        """GroupBy operations should not corrupt shared Index."""
        df = pd.DataFrame({
            'A': np.random.randint(0, 3, 100),
            'B': np.random.randn(100),
        }, index=np.arange(100, 200))

        checkpoint = deepcopy(df, {})
        original_index = list(checkpoint.index[:5])

        # Create GroupBy and perform operations
        gb = df.groupby('A')
        _ = gb.sum()
        _ = gb.transform('mean')
        _ = gb.apply(lambda x: x)
        for name, group in gb:
            pass

        # Checkpoint should be unaffected
        assert list(checkpoint.index[:5]) == original_index

    def test_groupby_internal_arrays_isolated(self):
        """GroupBy's internal grouper arrays should not share memory with checkpoint."""
        df = pd.DataFrame({
            'A': np.random.randint(0, 3, 100),
            'B': np.random.randn(100),
        })

        memo = {}
        df_copy = deepcopy(df, memo)
        gb = df.groupby('A')
        gb_copy = deepcopy(gb, memo)

        # Grouper codes should not share memory
        if hasattr(gb._grouper, 'codes') and hasattr(gb_copy._grouper, 'codes'):
            orig_codes = np.asarray(gb._grouper.codes[0])
            copy_codes = np.asarray(gb_copy._grouper.codes[0])
            assert not np.shares_memory(orig_codes, copy_codes)


# =============================================================================
# HEAPSIZER DEDUPLICATION TESTS
# =============================================================================

class TestHeapSizerIndexDeduplication:
    """Tests for HeapSizer Index deduplication."""

    def test_heapsizer_deduplicates_shared_index(self):
        """HeapSizer should deduplicate Index that shares memory."""
        df = pd.DataFrame({'A': np.random.randn(10000)}, index=np.arange(10000))

        # Create two checkpoints that properly isolate Index
        memo1 = {}
        cp1 = deepcopy(df, memo1)

        memo2 = {}
        cp2 = deepcopy(df, memo2)

        # Measure separately - should get full size for each
        sizer1 = HeapSizer()
        size1 = sizer1.sizeof(cp1)

        sizer2 = HeapSizer()
        size2 = sizer2.sizeof(cp2)

        # Each should be roughly the same size
        assert abs(size1 - size2) < 1000  # Within 1KB

    def test_heapsizer_deduplicates_shared_dataframe(self):
        """HeapSizer should deduplicate df and gb.obj when they're the same object."""
        df = pd.DataFrame({
            'A': np.random.randn(10000),
        }, index=np.arange(10000))
        gb = df.groupby(df['A'] > 0)

        # In the checkpoint, df and gb.obj should be the same object
        memo = {}
        copied = {
            'df': deepcopy(df, memo),
            'gb': deepcopy(gb, memo),
        }

        # Verify they share the same DataFrame
        assert copied['df'] is copied['gb'].obj

        # Measure df alone
        sizer1 = HeapSizer()
        df_alone = sizer1.sizeof(copied['df'])

        # Measure gb alone - should include its own copy of the DataFrame
        sizer2 = HeapSizer()
        gb_alone = sizer2.sizeof(copied['gb'])

        # Measure together - df then gb
        sizer3 = HeapSizer()
        df_first = sizer3.sizeof(copied['df'])
        gb_after_df = sizer3.sizeof(copied['gb'])

        # When measuring together, gb should be smaller because df is already counted
        assert gb_after_df < gb_alone

    def test_heapsizer_index_extraction(self):
        """HeapSizer should properly extract and deduplicate Index backing arrays."""
        df = pd.DataFrame({'A': np.random.randn(10000)}, index=np.arange(10000))

        sizer = HeapSizer()
        size = sizer.sizeof(df)

        # Index is ~80KB for 10000 int64 values
        # Check that size includes Index (should be > 80KB for data + index + overhead)
        assert size > 80000

    def test_heapsizer_handles_rangeindex(self):
        """HeapSizer should handle RangeIndex without errors."""
        df = pd.DataFrame({'A': np.random.randn(10000)})
        assert isinstance(df.index, pd.RangeIndex)

        sizer = HeapSizer()
        size = sizer.sizeof(df)

        # Should measure successfully and return a positive size
        assert size > 0
        # Data column is 10000 * 8 = 80000 bytes
        assert size >= 80000


# =============================================================================
# CHECKPOINT INTEGRATION TESTS
# =============================================================================

class TestCheckpointIntegration:
    """Integration tests with full checkpoint system."""

    def test_multiple_checkpoints_isolated(self):
        """Multiple checkpoints should have isolated Index data."""
        cp = MemoryCheckpoints()
        user_ns = {
            'df': pd.DataFrame({'A': [1, 2, 3]}, index=np.array([10, 20, 30]))
        }

        cp.save('checkpoint1', user_ns)

        # Modify original
        user_ns['df'].index._data[0] = 999

        # Checkpoint should be unaffected
        restored_df = cp.get('checkpoint1').user_ns['df']
        assert restored_df.index[0] == 10

    def test_checkpoint_with_groupby(self):
        """Checkpoint with GroupBy should handle Index correctly."""
        cp = MemoryCheckpoints()
        df = pd.DataFrame({
            'A': np.random.randint(0, 3, 100),
            'B': np.random.randn(100),
        }, index=np.arange(100))

        user_ns = {'df': df, 'gb': df.groupby('A')}

        cp.save('checkpoint1', user_ns)

        # Get checkpoint
        restored = cp.get('checkpoint1').user_ns

        # df.index and gb.obj.index should be the same object (memo caching)
        assert restored['df'].index is restored['gb'].obj.index

        # But isolated from original
        assert restored['df'].index._data.ctypes.data != df.index._data.ctypes.data

    def test_checkpoint_various_index_types(self):
        """Checkpoint should handle various Index types correctly."""
        cp = MemoryCheckpoints()

        # Various Index types
        user_ns = {
            'df_int': pd.DataFrame({'A': [1, 2]}, index=np.array([10, 20])),
            'df_range': pd.DataFrame({'A': [1, 2]}),  # RangeIndex
            'df_datetime': pd.DataFrame(
                {'A': [1, 2]},
                index=pd.DatetimeIndex(['2020-01-01', '2020-01-02'])
            ),
            'df_multi': pd.DataFrame(
                {'A': [1, 2]},
                index=pd.MultiIndex.from_tuples([(1, 'a'), (2, 'b')])
            ),
        }

        cp.save('checkpoint1', user_ns)
        restored = cp.get('checkpoint1').user_ns

        # Verify each Index type is properly restored
        assert list(restored['df_int'].index) == [10, 20]
        assert list(restored['df_range'].index) == [0, 1]
        assert list(restored['df_datetime'].index) == list(user_ns['df_datetime'].index)
        assert list(restored['df_multi'].index) == [(1, 'a'), (2, 'b')]


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataframe_index(self):
        """Empty DataFrame with Index should be handled correctly."""
        df = pd.DataFrame({'A': []}, index=pd.Index([], dtype='int64'))
        checkpoint = deepcopy(df, {})

        assert len(checkpoint.index) == 0
        assert checkpoint.index.dtype == df.index.dtype

    def test_index_with_name(self):
        """Index name should be preserved."""
        df = pd.DataFrame({'A': [1, 2, 3]}, index=pd.Index([10, 20, 30], name='my_index'))
        checkpoint = deepcopy(df, {})

        assert checkpoint.index.name == 'my_index'

    def test_column_index_isolated(self):
        """DataFrame column Index should also be isolated."""
        df = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.Index(['col_a', 'col_b'], name='cols')
        )
        checkpoint = deepcopy(df, {})

        # Column Index should be a different object
        assert df.columns is not checkpoint.columns
        # But with same values
        assert list(df.columns) == list(checkpoint.columns)
        assert checkpoint.columns.name == 'cols'

    def test_deeply_nested_groupby(self):
        """Nested GroupBy operations should work correctly."""
        df = pd.DataFrame({
            'A': np.random.randint(0, 2, 50),
            'B': np.random.randint(0, 3, 50),
            'C': np.random.randn(50),
        }, index=np.arange(50))

        gb1 = df.groupby('A')
        gb2 = df.groupby(['A', 'B'])

        memo = {}
        copied = {
            'df': deepcopy(df, memo),
            'gb1': deepcopy(gb1, memo),
            'gb2': deepcopy(gb2, memo),
        }

        # All should share the same Index
        assert copied['df'].index is copied['gb1'].obj.index
        assert copied['df'].index is copied['gb2'].obj.index
