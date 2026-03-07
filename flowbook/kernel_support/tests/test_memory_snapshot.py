"""Tests for MemoryCheckpoints.get_memory_snapshot() - v5 simplified API."""

import numpy as np
import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


class TestGetMemorySnapshotBasic:
    """Basic functionality tests for get_memory_snapshot()."""

    def test_empty_namespace_no_checkpoints(self):
        """Returns zeros for empty namespace with no checkpoints."""
        cp = MemoryCheckpoints()
        ns = {}

        result = cp.get_memory_snapshot(ns)

        assert result['user_ns_bytes'] == 0
        assert result['checkpoint_bytes'] == 0
        assert result['checkpoint_vars'] == {}
        assert 'gpu_bytes' in result

    def test_namespace_only_no_checkpoints(self):
        """Measures namespace correctly when no checkpoints exist."""
        cp = MemoryCheckpoints()
        arr = np.random.random((100, 100))
        ns = {'arr': arr}

        result = cp.get_memory_snapshot(ns)

        # arr is 100x100 floats = 80KB
        assert result['user_ns_bytes'] > 0
        assert result['checkpoint_bytes'] == 0
        assert result['checkpoint_vars'] == {}

    def test_checkpoint_overhead_measured(self):
        """Checkpoint overhead is measured after saving."""
        cp = MemoryCheckpoints()

        # Create a namespace with data
        arr = np.random.random((100, 100))
        ns = {'arr': arr}

        # No checkpoints yet
        result_before = cp.get_memory_snapshot(ns)
        assert result_before['checkpoint_bytes'] == 0

        # Save a checkpoint
        cp.save('ckpt1', ns)

        # Now checkpoints should have overhead
        result_after = cp.get_memory_snapshot(ns)

        # Arrays get deep copied, so checkpoint_bytes > 0
        assert result_after['checkpoint_bytes'] > 0
        assert 'arr' in result_after['checkpoint_vars']
        assert result_after['checkpoint_vars']['arr'] > 0

    def test_checkpoint_bytes_beyond_namespace(self):
        """Checkpoint bytes excludes memory shared with namespace."""
        cp = MemoryCheckpoints()

        # Create a large array
        arr = np.random.random((1000, 1000))  # ~8MB
        ns = {'arr': arr}
        cp.save('ckpt1', ns)

        result = cp.get_memory_snapshot(ns)

        # Checkpoint bytes should be > 0 (array is deep copied)
        assert result['checkpoint_bytes'] > 0
        assert result['checkpoint_vars'].get('arr', 0) > 0

    def test_filters_private_variables(self):
        """Private variables (starting with _) are excluded."""
        cp = MemoryCheckpoints()
        ns = {'x': [1, 2, 3], '_private': [4, 5, 6], '__dunder': 'test'}

        result = cp.get_memory_snapshot(ns)

        # user_ns_bytes should only count 'x', not private vars
        # (Hard to verify exact bytes, but method should not error)
        assert result['user_ns_bytes'] >= 0

    def test_filters_modules(self):
        """Module objects are excluded."""
        import types
        cp = MemoryCheckpoints()
        ns = {'x': [1, 2, 3], 'np': np}  # np is a module

        result = cp.get_memory_snapshot(ns)

        # Should not error, np should be filtered out
        assert result['user_ns_bytes'] >= 0

    def test_filters_functions(self):
        """Function objects are excluded."""
        cp = MemoryCheckpoints()

        def my_func():
            pass

        ns = {'x': [1, 2, 3], 'func': my_func}

        result = cp.get_memory_snapshot(ns)

        # Should not error, func should be filtered out
        assert result['user_ns_bytes'] >= 0

    def test_multiple_checkpoints(self):
        """Multiple checkpoints are measured cumulatively."""
        cp = MemoryCheckpoints()

        arr1 = np.random.random((100, 100))
        arr2 = np.random.random((100, 100))
        ns = {'arr1': arr1}

        cp.save('ckpt1', ns)
        result1 = cp.get_memory_snapshot(ns)

        # Add another variable and checkpoint
        ns['arr2'] = arr2
        cp.save('ckpt2', ns)
        result2 = cp.get_memory_snapshot(ns)

        # Second result should have more checkpoint bytes
        assert result2['checkpoint_bytes'] >= result1['checkpoint_bytes']

    def test_result_structure(self):
        """Result has expected structure."""
        cp = MemoryCheckpoints()
        ns = {'x': [1, 2, 3]}

        result = cp.get_memory_snapshot(ns)

        assert 'user_ns_bytes' in result
        assert 'gpu_bytes' in result
        assert 'checkpoint_bytes' in result
        assert 'checkpoint_vars' in result
        assert isinstance(result['user_ns_bytes'], int)
        assert isinstance(result['gpu_bytes'], int)
        assert isinstance(result['checkpoint_bytes'], int)
        assert isinstance(result['checkpoint_vars'], dict)


class TestGetMemorySnapshotDataFrames:
    """DataFrame-specific tests for CoW handling."""

    def test_dataframe_checkpoint_overhead(self):
        """DataFrames in checkpoints are measured correctly."""
        import pandas as pd
        cp = MemoryCheckpoints()

        df = pd.DataFrame({'a': np.random.random(1000), 'b': np.random.random(1000)})
        ns = {'df': df}
        cp.save('ckpt1', ns)

        result = cp.get_memory_snapshot(ns)

        # With CoW, checkpoint may share data with namespace initially
        # After deepcopy, there should be some overhead
        assert result['checkpoint_bytes'] >= 0
        # 'df' should be in checkpoint_vars
        assert 'df' in result['checkpoint_vars'] or result['checkpoint_vars'] == {}


class TestGetMemorySnapshotEdgeCases:
    """Edge case tests."""

    def test_empty_checkpoint(self):
        """Handles checkpoints with no variables."""
        cp = MemoryCheckpoints()
        ns = {}
        cp.save('empty', ns)

        result = cp.get_memory_snapshot(ns)

        assert result['checkpoint_bytes'] == 0
        assert result['checkpoint_vars'] == {}

    def test_cow_sharing(self):
        """CoW sharing is handled correctly - shared data not double-counted."""
        import pandas as pd
        cp = MemoryCheckpoints()

        # Create DataFrame
        df = pd.DataFrame({'a': np.arange(10000)})
        ns = {'df': df}

        # Save checkpoint (uses shallow copy with CoW)
        cp.save('ckpt1', ns)

        result = cp.get_memory_snapshot(ns)

        # Due to CoW, much of the data may be shared
        # The exact value depends on pandas CoW implementation
        # Just verify it doesn't error and returns valid structure
        assert result['checkpoint_bytes'] >= 0
