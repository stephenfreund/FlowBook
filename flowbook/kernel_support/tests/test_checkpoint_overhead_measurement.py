"""
Tests for the checkpoint overhead measurement fix.

This module tests the cumulative checkpoint measurement approach that properly
handles Copy-on-Write (CoW) sharing between checkpoints and the namespace.

The key insight is that checkpoint measurements must be done CUMULATIVELY:
1. First measure namespace (marks objects as seen)
2. Then measure checkpoints in order (only count NEW objects)

This fixes the issue where checkpoint overhead was incorrectly reported as 4x+
the namespace size because CoW-shared memory was being double-counted.

NOTE: numpy arrays are fully deep-copied in checkpoints (no CoW).
DataFrames use CoW sharing, so they have minimal overhead.
"""

import pytest
import numpy as np
import pandas as pd

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.heap_size import HeapSizer


@pytest.fixture(autouse=True)
def disable_infer_string():
    """Disable infer_string to avoid pyarrow dependency issues."""
    old_value = pd.options.future.infer_string
    pd.options.future.infer_string = False
    yield
    pd.options.future.infer_string = old_value


class TestGetOverheadBeyondNamespace:
    """Tests for MemoryCheckpoints.get_overhead_beyond_namespace method."""

    def test_numpy_array_is_deep_copied(self):
        """Verify that numpy arrays are deep copied (not shared) in checkpoints.

        Unlike DataFrames, numpy arrays are fully deep-copied, so they don't
        benefit from CoW sharing. This is expected behavior.
        """
        cp = MemoryCheckpoints()

        # Create a large array in namespace
        arr = np.zeros(100_000)  # ~800KB
        user_ns = {'arr': arr}

        # Save checkpoint (deep copies the array)
        cp.save('_pre_cell1', user_ns)

        # For numpy arrays, deep copy creates new memory, so overhead equals array size
        result = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # Numpy arrays are deep copied, so overhead should be ~800KB
        assert result['total_mb'] > 0.7, \
            f"Expected ~800KB overhead for deep-copied numpy array, got {result['total_mb']}MB"

    def test_new_data_counted_correctly(self):
        """Verify that new data in checkpoint is counted correctly."""
        cp = MemoryCheckpoints()

        # Namespace with one variable
        ns_arr = np.zeros(50_000)  # ~400KB
        user_ns = {'ns_arr': ns_arr}

        # Checkpoint with different variable
        cp.saved['_pre_cell1'] = type('MockCheckpoint', (), {
            'user_ns': {'ckpt_arr': np.zeros(50_000)}  # Different array ~400KB
        })()

        result = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # Should count the new array (~400KB)
        assert result['total_mb'] > 0.35, \
            f"Expected ~400KB for new data, got {result['total_mb']}MB"
        assert result['total_mb'] < 0.5

    def test_checkpoint_stops_at_cell(self):
        """Verify that measurement stops at the specified cell's checkpoint."""
        cp = MemoryCheckpoints()

        user_ns = {'x': 1}

        # Add checkpoints for multiple cells in order
        # Note: saved is an ordered dict, so order matters
        cp.saved['_pre_cell1'] = type('MockCheckpoint', (), {
            'user_ns': {'arr1': np.zeros(10_000)}  # ~80KB
        })()
        cp.saved['_pre_cell2'] = type('MockCheckpoint', (), {
            'user_ns': {'arr2': np.zeros(10_000)}  # ~80KB
        })()

        # Get overhead for cell1 only
        result1 = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # Should only include cell1's checkpoint
        assert '_pre_cell1' in result1['by_checkpoint']
        # Should NOT include cell2 (stops at first match)
        assert '_pre_cell2' not in result1['by_checkpoint']

        # Get overhead for cell2 (includes cell1 and cell2)
        result2 = cp.get_overhead_beyond_namespace('cell2', user_ns)
        assert '_pre_cell1' in result2['by_checkpoint']
        assert '_pre_cell2' in result2['by_checkpoint']
        # Total should be larger since it includes both
        assert result2['total_mb'] > result1['total_mb']

    def test_cow_dataframe_sharing(self):
        """Test CoW sharing with pandas DataFrames.

        DataFrames use Copy-on-Write, so the checkpoint shares underlying data
        with the namespace until either is modified.
        """
        cp = MemoryCheckpoints()

        # Create DataFrame in namespace
        df = pd.DataFrame({'a': np.zeros(100_000)})  # ~800KB
        user_ns = {'df': df}

        # Save checkpoint (uses CoW deepcopy for DataFrames)
        cp.save('_pre_cell1', user_ns)

        result = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # With CoW sharing, overhead should be much smaller than 800KB
        # Allow some overhead for DataFrame wrapper objects
        assert result['total_mb'] < 0.5, \
            f"CoW DataFrame should have minimal data overhead, got {result['total_mb']}MB"

    def test_no_checkpoints_returns_zero(self):
        """Test that empty checkpoints return zero overhead."""
        cp = MemoryCheckpoints()
        user_ns = {'arr': np.zeros(10_000)}

        result = cp.get_overhead_beyond_namespace('nonexistent_cell', user_ns)

        assert result['total_mb'] == 0
        assert len(result['by_checkpoint']) == 0

    def test_by_variable_breakdown(self):
        """Test per-variable breakdown in results."""
        cp = MemoryCheckpoints()

        user_ns = {'x': 1}

        # Checkpoint with multiple new variables
        cp.saved['_pre_cell1'] = type('MockCheckpoint', (), {
            'user_ns': {
                'var1': np.zeros(40_000),  # ~320KB
                'var2': np.zeros(20_000),  # ~160KB
            }
        })()

        result = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # Should have breakdown by variable
        assert 'var1' in result['by_variable']
        assert 'var2' in result['by_variable']
        assert result['by_variable']['var1'] > result['by_variable']['var2']


class TestCheckpointOverheadRatioDataFrames:
    """Tests verifying that checkpoint overhead ratios are small for DataFrames."""

    def test_dataframe_overhead_ratio_small(self):
        """Verify overhead ratio is small for DataFrames due to CoW sharing."""
        cp = MemoryCheckpoints()

        # Create substantial namespace with DataFrames
        df1 = pd.DataFrame({
            'a': np.zeros(100_000),
            'b': np.ones(100_000),
        })
        df2 = pd.DataFrame({
            'x': np.random.randn(50_000),
            'y': np.random.randn(50_000),
        })
        user_ns = {'df1': df1, 'df2': df2, 'value': 42}

        # Save checkpoint
        cp.save('_pre_cell1', user_ns)

        # Get overhead
        result = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # Calculate namespace size
        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace(user_ns)
        ns_mb = ns_size.total_bytes / (1024 * 1024)

        # For DataFrames with CoW, overhead should be smaller than namespace
        # (not 4x+ like the original bug)
        ratio = result['total_mb'] / ns_mb if ns_mb > 0 else 0
        assert ratio < 0.5, \
            f"Overhead ratio {ratio:.2f} is too high for DataFrames (expected < 0.5 with CoW sharing)"

    def test_no_4x_overhead_bug_for_dataframes(self):
        """Regression test: verify the 4x+ overhead bug is fixed for DataFrames.

        The original bug was that checkpoint overhead was reported as 4x+ the
        namespace size because CoW-shared memory was being double-counted.
        """
        cp = MemoryCheckpoints()

        # Simulate syntactic mode: single checkpoint with DataFrame
        df = pd.DataFrame({
            'col1': np.zeros(200_000),
            'col2': np.ones(200_000),
        })
        user_ns = {'df': df}

        # Save pre-checkpoint (syntactic mode only has pre)
        cp.save('_pre_cell1', user_ns)

        # Measure overhead
        result = cp.get_overhead_beyond_namespace('cell1', user_ns)

        # Measure namespace
        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace(user_ns)
        ns_mb = ns_size.total_bytes / (1024 * 1024)

        # Before the fix, this would be 4x+ due to not handling CoW sharing
        # After the fix, it should be much smaller (CoW sharing)
        # Allow up to 50% ratio - still much better than 4x
        assert result['total_mb'] < ns_mb * 0.5, \
            f"Overhead {result['total_mb']:.2f}MB should be << namespace {ns_mb:.2f}MB for DataFrames"


class TestIntegrationWithCompareBaseline:
    """Integration tests simulating compare_baseline workflow."""

    def test_cumulative_checkpoints_workflow(self):
        """Test that checkpoints accumulate correctly."""
        cp = MemoryCheckpoints()

        # Create namespace
        user_ns = {'df': pd.DataFrame({'a': np.zeros(50_000)})}

        # Save checkpoint for cell1
        cp.save('_pre_cell1', user_ns)

        # Add more data, save checkpoint for cell2
        user_ns['df2'] = pd.DataFrame({'b': np.ones(50_000)})
        cp.save('_pre_cell2', user_ns)

        # Add even more, save checkpoint for cell3
        user_ns['df3'] = pd.DataFrame({'c': np.zeros(50_000)})
        cp.save('_pre_cell3', user_ns)

        # Measure at each cell
        oh1 = cp.get_overhead_beyond_namespace('cell1', user_ns)
        oh2 = cp.get_overhead_beyond_namespace('cell2', user_ns)
        oh3 = cp.get_overhead_beyond_namespace('cell3', user_ns)

        # Each subsequent measurement should include more checkpoints
        # so cumulative overhead should grow (or stay same if sharing)
        assert oh1['total_mb'] <= oh2['total_mb']
        assert oh2['total_mb'] <= oh3['total_mb']

        # Verify by_checkpoint includes expected checkpoints
        assert len(oh1['by_checkpoint']) == 1
        assert len(oh2['by_checkpoint']) == 2
        assert len(oh3['by_checkpoint']) == 3


class TestHeapSizerCheckpointOverhead:
    """Direct tests of HeapSizer.sizeof_checkpoints_beyond_namespace."""

    def test_cumulative_deduplication(self):
        """Test that cumulative measurement deduplicates across checkpoints."""
        from flowbook.kernel_support.heap_size import HeapSizer

        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        # Same array referenced by both checkpoints
        shared_arr = np.zeros(100_000)  # ~800KB

        ns = {'x': 1}
        ckpt1 = MockCheckpoint({'arr': shared_arr})
        ckpt2 = MockCheckpoint({'arr': shared_arr})  # Same object

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(
            ns, [('ckpt1', ckpt1), ('ckpt2', ckpt2)]
        )

        # Total should be ~800KB (not 1.6MB) because array is deduplicated
        assert result.total_mb > 0.7
        assert result.total_mb < 1.0  # Should not count twice

        # First checkpoint gets credit, second gets 0 or minimal
        assert result.by_checkpoint['ckpt1'] > 0.7
        assert result.by_checkpoint['ckpt2'] < 0.1

    def test_namespace_objects_excluded(self):
        """Test that namespace objects are excluded from checkpoint measurement."""
        from flowbook.kernel_support.heap_size import HeapSizer

        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        # Same array in namespace and checkpoint
        arr = np.zeros(100_000)  # ~800KB

        ns = {'arr': arr}
        ckpt = MockCheckpoint({'arr': arr})  # Same object

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [('ckpt', ckpt)])

        # Array is already in namespace, so checkpoint overhead should be minimal
        assert result.total_mb < 0.1, \
            f"Expected minimal overhead since array is in namespace, got {result.total_mb}MB"
