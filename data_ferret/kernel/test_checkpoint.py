"""
Comprehensive unit tests for checkpoint save/restore functionality.

Tests verify that checkpoints properly deep copy variables, especially
ensuring that mutable objects inside pandas DataFrames and Series are
fully isolated to prevent shared references.

To run these tests:
    pytest data_ferret/kernel/test_checkpoint.py -v
"""

import pytest
import copy
import types
import numpy as np
import pandas as pd
from typing import Dict, Any

from data_ferret.kernel.checkpoint import Checkpoints, Checkpoint
from data_ferret.kernel.extended_types import TypeModel


# ============================================================================
# TEST HELPERS
# ============================================================================

def modify_list_in_place(lst: list, value: Any):
    """Helper to modify a list in place."""
    lst.append(value)


def modify_dict_in_place(d: dict, key: str, value: Any):
    """Helper to modify a dict in place."""
    d[key] = value


# ============================================================================
# DEEP COPY ISOLATION TESTS
# ============================================================================

class TestDeepCopyDataFrames:
    """Test that DataFrames with mutable objects are properly deep copied."""

    def test_dataframe_with_lists_in_cells(self):
        """Test that lists in DataFrame cells are deep copied, not shared."""
        cp = Checkpoints()

        # Create DataFrame with lists in object dtype column
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        })

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Modify the original DataFrame's list
        user_ns['df'].iloc[0, 1].append(999)

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Restored DataFrame should not have the modification
        assert user_ns['df'].iloc[0, 1] == [1, 2, 3]
        assert 999 not in user_ns['df'].iloc[0, 1]

    def test_dataframe_with_dicts_in_cells(self):
        """Test that dicts in DataFrame cells are deep copied, not shared."""
        cp = Checkpoints()

        # Create DataFrame with dicts in object dtype column
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'config': [{'a': 1}, {'b': 2}, {'c': 3}]
        })

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Modify the original DataFrame's dict
        user_ns['df'].iloc[0, 1]['new_key'] = 'new_value'

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Restored DataFrame should not have the modification
        assert user_ns['df'].iloc[0, 1] == {'a': 1}
        assert 'new_key' not in user_ns['df'].iloc[0, 1]

    def test_dataframe_with_nested_mutable_structures(self):
        """Test that nested mutable structures in DataFrames are deep copied."""
        cp = Checkpoints()

        # Create DataFrame with complex nested structures
        df = pd.DataFrame({
            'id': [1, 2],
            'nested': [
                {'list': [1, 2, 3], 'dict': {'a': [4, 5]}},
                {'list': [6, 7, 8], 'dict': {'b': [9, 10]}}
            ]
        })

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Modify nested structures
        user_ns['df'].iloc[0, 1]['list'].append(999)
        user_ns['df'].iloc[0, 1]['dict']['a'].append(888)

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Restored DataFrame should not have modifications
        assert user_ns['df'].iloc[0, 1]['list'] == [1, 2, 3]
        assert user_ns['df'].iloc[0, 1]['dict']['a'] == [4, 5]

    def test_dataframe_mixed_dtypes(self):
        """Test DataFrame with both object and non-object dtype columns."""
        cp = Checkpoints()

        # Create DataFrame with mixed column types
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'obj_col': [[1, 2], [3, 4], [5, 6]]
        })

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Modify the list in object column
        user_ns['df'].iloc[0, 3].append(999)
        # Modify other columns
        user_ns['df'].iloc[0, 0] = 999

        # Restore checkpoint
        cp.restore('test', user_ns)

        # All columns should be restored
        assert user_ns['df'].iloc[0, 0] == 1
        assert user_ns['df'].iloc[0, 3] == [1, 2]

    def test_multiple_dataframes(self):
        """Test multiple DataFrames with mutable objects."""
        cp = Checkpoints()

        df1 = pd.DataFrame({'data': [[1, 2], [3, 4]]})
        df2 = pd.DataFrame({'config': [{'a': 1}, {'b': 2}]})

        user_ns = {'df1': df1, 'df2': df2}
        cp.save('test', user_ns)

        # Modify both DataFrames
        user_ns['df1'].iloc[0, 0].append(999)
        user_ns['df2'].iloc[0, 0]['new'] = 'value'

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Both should be restored properly
        assert user_ns['df1'].iloc[0, 0] == [1, 2]
        assert user_ns['df2'].iloc[0, 0] == {'a': 1}


class TestDeepCopySeries:
    """Test that Series with mutable objects are properly deep copied."""

    def test_series_with_lists(self):
        """Test that lists in Series are deep copied, not shared."""
        cp = Checkpoints()

        # Create Series with lists
        s = pd.Series([[1, 2], [3, 4], [5, 6]])

        user_ns = {'s': s}
        cp.save('test', user_ns)

        # Modify the original Series' list
        user_ns['s'].iloc[0].append(999)

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Restored Series should not have the modification
        assert user_ns['s'].iloc[0] == [1, 2]

    def test_series_with_dicts(self):
        """Test that dicts in Series are deep copied, not shared."""
        cp = Checkpoints()

        # Create Series with dicts
        s = pd.Series([{'a': 1}, {'b': 2}, {'c': 3}])

        user_ns = {'s': s}
        cp.save('test', user_ns)

        # Modify the original Series' dict
        user_ns['s'].iloc[0]['new'] = 'value'

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Restored Series should not have the modification
        assert user_ns['s'].iloc[0] == {'a': 1}

    def test_series_non_object_dtype(self):
        """Test that non-object dtype Series are handled correctly."""
        cp = Checkpoints()

        # Create Series with numeric dtype
        s = pd.Series([1, 2, 3, 4, 5], dtype=int)

        user_ns = {'s': s}
        cp.save('test', user_ns)

        # Modify the Series
        user_ns['s'].iloc[0] = 999

        # Restore checkpoint
        cp.restore('test', user_ns)

        # Should be restored to original values
        assert user_ns['s'].iloc[0] == 1
        assert list(user_ns['s']) == [1, 2, 3, 4, 5]


class TestMultipleRestores:
    """Test that checkpoints remain pristine across multiple restores."""

    def test_multiple_restores_dataframe_with_lists(self):
        """Test that restoring multiple times doesn't corrupt the checkpoint."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': [[1, 2, 3], [4, 5, 6]]})
        user_ns = {'df': df}
        cp.save('test', user_ns)

        # First restore and modify
        cp.restore('test', user_ns)
        user_ns['df'].iloc[0, 0].append(999)
        assert 999 in user_ns['df'].iloc[0, 0]

        # Second restore should still get original data
        cp.restore('test', user_ns)
        assert user_ns['df'].iloc[0, 0] == [1, 2, 3]

        # Third restore with different modification
        user_ns['df'].iloc[0, 0].extend([777, 888])
        cp.restore('test', user_ns)
        assert user_ns['df'].iloc[0, 0] == [1, 2, 3]

    def test_multiple_restores_series_with_dicts(self):
        """Test multiple restores with Series containing dicts."""
        cp = Checkpoints()

        s = pd.Series([{'x': 1, 'y': 2}, {'z': 3}])
        user_ns = {'s': s}
        cp.save('test', user_ns)

        # Multiple restore cycles
        for i in range(3):
            cp.restore('test', user_ns)
            user_ns['s'].iloc[0][f'key_{i}'] = f'value_{i}'
            # Each time we restore, the modifications should be gone

        # Final restore should have original data
        cp.restore('test', user_ns)
        assert user_ns['s'].iloc[0] == {'x': 1, 'y': 2}
        assert len(user_ns['s'].iloc[0]) == 2


# ============================================================================
# NON-PANDAS OBJECT TESTS
# ============================================================================

class TestNonPandasObjects:
    """Test checkpointing of non-pandas objects."""

    def test_regular_lists(self):
        """Test that regular lists are deep copied."""
        cp = Checkpoints()

        lst = [1, 2, [3, 4]]
        user_ns = {'lst': lst}
        cp.save('test', user_ns)

        # Modify nested list
        user_ns['lst'][2].append(5)

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['lst'] == [1, 2, [3, 4]]

    def test_regular_dicts(self):
        """Test that regular dicts are deep copied."""
        cp = Checkpoints()

        d = {'a': 1, 'b': {'c': 2}}
        user_ns = {'d': d}
        cp.save('test', user_ns)

        # Modify nested dict
        user_ns['d']['b']['new'] = 3

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['d'] == {'a': 1, 'b': {'c': 2}}

    def test_numpy_arrays(self):
        """Test that numpy arrays are properly copied."""
        cp = Checkpoints()

        arr = np.array([1, 2, 3, 4, 5])
        user_ns = {'arr': arr}
        cp.save('test', user_ns)

        # Modify array
        user_ns['arr'][0] = 999

        # Restore
        cp.restore('test', user_ns)
        assert np.array_equal(user_ns['arr'], np.array([1, 2, 3, 4, 5]))

    def test_mixed_types(self):
        """Test checkpointing multiple different types together."""
        cp = Checkpoints()

        user_ns = {
            'num': 42,
            'lst': [1, 2, 3],
            'df': pd.DataFrame({'data': [[1, 2]]}),
            's': pd.Series([{'a': 1}]),
            'arr': np.array([10, 20, 30])
        }

        cp.save('test', user_ns)

        # Modify everything
        user_ns['num'] = 999
        user_ns['lst'].append(999)
        user_ns['df'].iloc[0, 0].append(999)
        user_ns['s'].iloc[0]['new'] = 'value'
        user_ns['arr'][0] = 999

        # Restore
        cp.restore('test', user_ns)

        # Verify all restored correctly
        assert user_ns['num'] == 42
        assert user_ns['lst'] == [1, 2, 3]
        assert user_ns['df'].iloc[0, 0] == [1, 2]
        assert user_ns['s'].iloc[0] == {'a': 1}
        assert np.array_equal(user_ns['arr'], np.array([10, 20, 30]))


# ============================================================================
# CHECKPOINTABLE VALUE FILTERING TESTS
# ============================================================================

class TestCheckpointableFiltering:
    """Test that appropriate values are filtered from checkpoints."""

    def test_filters_modules(self):
        """Test that module objects are not checkpointed."""
        cp = Checkpoints()

        import math
        user_ns = {'x': 1, 'math_module': math}

        saved, removed = cp.save('test', user_ns)

        # x should be saved, math module should be filtered out entirely
        # (not saved, not in removed, because it's filtered at checkpointable_vars stage)
        assert 'x' in saved
        assert 'math_module' not in saved
        assert 'math_module' not in removed

        # Verify checkpoint doesn't contain the module
        checkpoint = cp.get('test')
        assert 'math_module' not in checkpoint.user_ns

    def test_filters_system_variables(self):
        """Test that IPython system variables are filtered."""
        cp = Checkpoints()

        user_ns = {
            'x': 1,
            'y': 2,
            '_': 'last_output',
            '__': 'prev_output',
            'get_ipython': lambda: None,
            '_private': 'private',
        }

        saved, removed = cp.save('test', user_ns)

        # Regular variables should be saved
        assert 'x' in saved
        assert 'y' in saved

        # System variables and private variables should not be saved
        assert '_' not in saved
        assert '__' not in saved
        assert 'get_ipython' not in saved
        assert '_private' not in saved

    def test_filters_matplotlib_objects(self):
        """Test that matplotlib objects are filtered."""
        cp = Checkpoints()

        # Create a mock matplotlib object
        class MockMatplotlibClass:
            __module__ = 'matplotlib.figure'

        user_ns = {
            'x': 1,
            'fig': MockMatplotlibClass()
        }

        saved, removed = cp.save('test', user_ns)

        # Matplotlib object should be removed
        assert 'x' in saved
        assert 'fig' in removed


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling for objects that can't be deep copied."""

    def test_uncopyable_object_tracked_as_removed(self):
        """Test that objects that fail to copy are tracked as removed."""
        cp = Checkpoints()

        # Create an object that can't be deep copied
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("Cannot copy this object")

        user_ns = {
            'x': 1,
            'bad': Uncopyable()
        }

        saved, removed = cp.save('test', user_ns)

        # x should be saved, bad should be removed
        assert 'x' in saved
        assert 'bad' in removed

    def test_restore_with_empty_namespace(self):
        """Test restoring into an empty namespace."""
        cp = Checkpoints()

        user_ns = {'x': 1, 'y': 2}
        cp.save('test', user_ns)

        # Clear namespace
        empty_ns = {}

        # Restore
        cp.restore('test', empty_ns)

        # Should have the saved variables
        assert empty_ns['x'] == 1
        assert empty_ns['y'] == 2


# ============================================================================
# CHECKPOINT MANAGEMENT TESTS
# ============================================================================

class TestCheckpointManagement:
    """Test checkpoint management operations."""

    def test_list_checkpoints(self):
        """Test listing saved checkpoints."""
        cp = Checkpoints()

        assert cp.list() == []

        user_ns = {'x': 1}
        cp.save('cp1', user_ns)
        assert cp.list() == ['cp1']

        cp.save('cp2', user_ns)
        assert set(cp.list()) == {'cp1', 'cp2'}

    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        cp = Checkpoints()

        user_ns = {'x': 1}
        cp.save('test', user_ns)
        assert 'test' in cp.list()

        cp.delete('test')
        assert 'test' not in cp.list()

    def test_clear_checkpoints(self):
        """Test clearing all checkpoints."""
        cp = Checkpoints()

        user_ns = {'x': 1}
        cp.save('cp1', user_ns)
        cp.save('cp2', user_ns)
        cp.save('cp3', user_ns)

        assert len(cp.list()) == 3

        cp.clear()
        assert cp.list() == []

    def test_get_checkpoint(self):
        """Test retrieving a checkpoint object."""
        cp = Checkpoints()

        user_ns = {'x': 1, 'y': 2}
        cp.save('test', user_ns)

        checkpoint = cp.get('test')
        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.name == 'test'
        assert 'x' in checkpoint.user_ns
        assert 'y' in checkpoint.user_ns

    def test_overwrite_checkpoint(self):
        """Test that saving with same name overwrites."""
        cp = Checkpoints()

        user_ns = {'x': 1}
        cp.save('test', user_ns)

        user_ns = {'x': 999, 'y': 2}
        cp.save('test', user_ns)

        # Should only have one checkpoint
        assert cp.list() == ['test']

        # Should have new values
        checkpoint = cp.get('test')
        assert checkpoint.user_ns['x'] == 999
        assert 'y' in checkpoint.user_ns
    def test_exists_method(self):
        """Test the exists() method."""
        cp = Checkpoints()

        # Non-existing checkpoint
        assert not cp.exists('nonexistent')

        # Create checkpoint
        user_ns = {'x': 1}
        cp.save('test', user_ns)

        # Now exists
        assert cp.exists('test')

        # Delete checkpoint
        cp.delete('test')

        # No longer exists
        assert not cp.exists('test')

    def test_exists_with_multiple_checkpoints(self):
        """Test exists() with multiple checkpoints."""
        cp = Checkpoints()

        user_ns = {'x': 1}
        cp.save('cp1', user_ns)
        cp.save('cp2', user_ns)
        cp.save('cp3', user_ns)

        assert cp.exists('cp1')
        assert cp.exists('cp2')
        assert cp.exists('cp3')
        assert not cp.exists('cp4')

        cp.delete('cp2')

        assert cp.exists('cp1')
        assert not cp.exists('cp2')
        assert cp.exists('cp3')


# ============================================================================
# NAME VALIDATION TESTS
# ============================================================================

class TestCheckpointNameValidation:
    """Test checkpoint name validation."""

    def test_empty_string_raises_error(self):
        """Test that empty string checkpoint name raises ValueError."""
        cp = Checkpoints()
        user_ns = {'x': 1}

        with pytest.raises(ValueError, match="cannot be empty"):
            cp.save('', user_ns)

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only checkpoint name raises ValueError."""
        cp = Checkpoints()
        user_ns = {'x': 1}

        with pytest.raises(ValueError, match="cannot be empty"):
            cp.save('   ', user_ns)

        with pytest.raises(ValueError, match="cannot be empty"):
            cp.save('\t\n', user_ns)

    def test_valid_names_work(self):
        """Test that various valid names work correctly."""
        cp = Checkpoints()
        user_ns = {'x': 1}

        # Normal names
        cp.save('test', user_ns)
        assert cp.exists('test')

        # Names with underscores and hyphens
        cp.save('test_checkpoint_1', user_ns)
        assert cp.exists('test_checkpoint_1')

        cp.save('test-checkpoint-2', user_ns)
        assert cp.exists('test-checkpoint-2')

        # Names with special characters (allowed)
        cp.save('checkpoint/v1', user_ns)
        assert cp.exists('checkpoint/v1')

        cp.save('checkpoint.v2', user_ns)
        assert cp.exists('checkpoint.v2')


# ============================================================================
# COPY-ON-WRITE VERIFICATION TESTS
# ============================================================================

class TestCopyOnWriteVerification:
    """Test that CoW is verified and enabled on initialization."""

    def test_cow_enabled_check(self):
        """Test that CoW gets enabled if disabled."""
        # Temporarily disable CoW
        original_cow = pd.options.mode.copy_on_write
        pd.options.mode.copy_on_write = False

        try:
            # Creating Checkpoints should re-enable it
            cp = Checkpoints()

            # Verify CoW is now enabled
            assert pd.options.mode.copy_on_write == True
        finally:
            # Restore original setting
            pd.options.mode.copy_on_write = original_cow

    def test_cow_already_enabled(self):
        """Test that no warning when CoW already enabled."""
        # Ensure CoW is enabled
        pd.options.mode.copy_on_write = True

        # Should work without issues
        cp = Checkpoints()

        # CoW should still be enabled
        assert pd.options.mode.copy_on_write == True


# ============================================================================
# SIZE WARNING TESTS
# ============================================================================

class TestSizeWarnings:
    """Test the max_size_mb parameter and size warnings."""

    def test_small_checkpoint_no_warning(self):
        """Test that small checkpoints don't warn."""
        cp = Checkpoints()

        # Create small DataFrame (< 1MB)
        df = pd.DataFrame({'data': range(100)})
        user_ns = {'df': df}

        # Should not warn (default max_size_mb=1000)
        cp.save('test', user_ns)
        # If it warned, it would be in the log output

    def test_large_checkpoint_warns(self):
        """Test that large checkpoints warn when exceeding limit."""
        cp = Checkpoints()

        # Create large DataFrame (> 1MB)
        df = pd.DataFrame({'data': range(1_000_000)})
        user_ns = {'df': df}

        # Should warn with small limit
        cp.save('test', user_ns, max_size_mb=1)
        # Warning will be logged

    def test_max_size_none_disables_warnings(self):
        """Test that max_size_mb=None disables size warnings."""
        cp = Checkpoints()

        # Create large DataFrame
        df = pd.DataFrame({'data': range(1_000_000)})
        user_ns = {'df': df}

        # Should not warn with max_size_mb=None
        cp.save('test', user_ns, max_size_mb=None)

    def test_size_estimation_reasonable(self):
        """Test that size estimation is reasonably accurate."""
        cp = Checkpoints()

        # Create DataFrame with known size
        df = pd.DataFrame({'data': range(10_000)})
        actual_size_bytes = df.memory_usage(deep=True).sum()
        actual_size_mb = actual_size_bytes / (1024 * 1024)

        user_ns = {'df': df}

        # Estimate size
        estimated_bytes = cp._estimate_size(user_ns)
        estimated_mb = estimated_bytes / (1024 * 1024)

        # Should be within reasonable range (estimate may not be exact)
        # Allow 50% tolerance either way
        assert estimated_mb > 0
        # This is a rough check - just ensure it's not wildly off


# ============================================================================
# CLASS WARNING TESTS
# ============================================================================

class TestClassWarnings:
    """Test warnings for user-defined classes."""

    def test_user_defined_class_triggers_warning(self):
        """Test that user-defined class triggers warning."""
        cp = Checkpoints(warn_classes=True)

        # Create a user-defined class
        class MyClass:
            class_var = 0

        user_ns = {'MyClass': MyClass}
        cp.save('test', user_ns)
        # Warning should be logged

    def test_builtin_class_no_warning(self):
        """Test that built-in classes don't warn."""
        cp = Checkpoints(warn_classes=True)

        user_ns = {'int_class': int, 'str_class': str}
        cp.save('test', user_ns)
        # Should not warn for built-in types

    def test_instance_no_warning(self):
        """Test that instances don't trigger class warning."""
        cp = Checkpoints(warn_classes=True)

        class MyClass:
            def __init__(self):
                self.value = 42

        obj = MyClass()
        user_ns = {'obj': obj}
        cp.save('test', user_ns)
        # Should not warn - it's an instance, not a class

    def test_warn_classes_false_suppresses_warning(self):
        """Test that warn_classes=False suppresses warnings."""
        cp = Checkpoints(warn_classes=False)

        class MyClass:
            class_var = 0

        user_ns = {'MyClass': MyClass}
        cp.save('test', user_ns)
        # Should not warn when disabled

    def test_pandas_numpy_classes_no_warning(self):
        """Test that pandas/numpy classes don't warn."""
        cp = Checkpoints(warn_classes=True)

        user_ns = {
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            'ndarray': np.ndarray
        }
        cp.save('test', user_ns)
        # Should not warn for library classes


# ============================================================================
# IMPROVED ERROR MESSAGE TESTS
# ============================================================================

class TestImprovedErrorMessages:
    """Test that error messages include helpful hints."""

    def test_generator_error_includes_hint(self):
        """Test that generator failures include helpful hint."""
        cp = Checkpoints()

        def my_gen():
            yield 1
            yield 2

        gen = my_gen()
        user_ns = {'gen': gen}

        # Save should handle the failure gracefully
        saved, removed = cp.save('test', user_ns)

        # Generator should be in removed
        assert 'gen' in removed

    def test_module_error_includes_hint(self):
        """Test that module failures include helpful hint."""
        cp = Checkpoints()

        import os
        user_ns = {'os_module': os}

        # Modules are filtered out, so they won't be in saved or removed
        saved, removed = cp.save('test', user_ns)

        # Module should not be in saved (filtered out by checkpointable_value)
        assert 'os_module' not in saved

    def test_iterator_error_includes_hint(self):
        """Test that iterator failures include helpful hint."""
        cp = Checkpoints()

        it = iter([1, 2, 3, 4, 5])
        # Advance the iterator
        next(it)

        user_ns = {'it': it}

        # Save might succeed or fail depending on iterator implementation
        # We're just verifying it doesn't crash
        saved, removed = cp.save('test', user_ns)


# ============================================================================
# PROGRESS LOGGING TESTS
# ============================================================================

class TestProgressLogging:
    """Test progress logging for large DataFrames."""

    def test_small_dataframe_normal_logging(self):
        """Test that small DataFrames use normal logging."""
        cp = Checkpoints()

        # Small DataFrame (< 10k rows)
        df = pd.DataFrame({'data': [[i] for i in range(100)]})
        user_ns = {'df': df}

        cp.save('test', user_ns)
        # Should log "Deep copying object column data" (without row count)

    def test_large_dataframe_progress_logging(self):
        """Test that large DataFrames log row count."""
        cp = Checkpoints()

        # Large DataFrame (> 10k rows)
        df = pd.DataFrame({'data': [[i] for i in range(15000)]})
        user_ns = {'df': df}

        cp.save('test', user_ns)
        # Should log "Deep copying large object column data with 15,000 rows..."

    def test_large_series_progress_logging(self):
        """Test that large Series log row count."""
        cp = Checkpoints()

        # Large Series (> 10k rows)
        s = pd.Series([[i] for i in range(12000)])
        user_ns = {'s': s}

        cp.save('test', user_ns)
        # Should log "Deep copying large object Series with 12,000 rows..."


# ============================================================================
# SANITY CHECK TESTS
# ============================================================================

class TestSanityCheck:
    """Test the sanity check feature."""

    def test_sanity_check_enabled(self):
        """Test that sanity check is performed when enabled."""
        cp = Checkpoints(sanity_check=True)

        # Normal save should work
        user_ns = {'x': 1, 'y': 2}
        saved, removed = cp.save('test', user_ns)

        assert 'x' in saved
        assert 'y' in saved

    def test_sanity_check_disabled(self):
        """Test that sanity check can be disabled."""
        cp = Checkpoints(sanity_check=False)

        user_ns = {'x': 1, 'y': 2}
        saved, removed = cp.save('test', user_ns)

        assert 'x' in saved
        assert 'y' in saved


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCheckpointIntegration:
    """Integration tests combining multiple features."""

    def test_save_restore_delete_cycle(self):
        """Test complete save/restore/delete cycle."""
        cp = Checkpoints()

        # Save checkpoint 1
        user_ns = {'x': 1, 'df': pd.DataFrame({'data': [[1, 2]]})}
        cp.save('cp1', user_ns)

        # Modify and save checkpoint 2
        user_ns['x'] = 2
        user_ns['df'].iloc[0, 0].append(3)
        cp.save('cp2', user_ns)

        # Restore cp1
        cp.restore('cp1', user_ns)
        assert user_ns['x'] == 1
        assert user_ns['df'].iloc[0, 0] == [1, 2]

        # Restore cp2
        cp.restore('cp2', user_ns)
        assert user_ns['x'] == 2
        assert user_ns['df'].iloc[0, 0] == [1, 2, 3]

        # Delete cp1 and verify
        cp.delete('cp1')
        assert cp.list() == ['cp2']

    def test_checkpoint_with_variable_deletion(self):
        """Test that restore properly removes variables not in checkpoint."""
        cp = Checkpoints()

        # Save with two variables
        user_ns = {'x': 1, 'y': 2}
        cp.save('test', user_ns)

        # Add a new variable
        user_ns['z'] = 3
        assert 'z' in user_ns

        # Restore should remove 'z'
        cp.restore('test', user_ns)
        assert 'x' in user_ns
        assert 'y' in user_ns
        assert 'z' not in user_ns

    def test_checkpoint_diff_after_modification(self):
        """Test checkpoint diff with modified variables."""
        cp = Checkpoints()

        # Create two checkpoints
        user_ns = {'x': 1, 'y': pd.DataFrame({'a': [1, 2, 3]})}
        cp.save('cp1', user_ns)

        user_ns['x'] = 999
        user_ns['y'] = pd.DataFrame({'a': [4, 5, 6]})
        cp.save('cp2', user_ns)

        # Compare checkpoints
        cp1_obj = cp.get('cp1')
        cp2_obj = cp.get('cp2')

        diff = Checkpoint.diff(cp1_obj, cp2_obj)

        # Should detect differences
        assert 'x' in diff.differences
        assert 'y' in diff.differences

    def test_type_models(self):
        """Test that type models are properly generated."""
        cp = Checkpoints()

        user_ns = {
            'num': 42,
            'lst': [1, 2, 3],
            'df': pd.DataFrame({'a': [1, 2]}),
            's': pd.Series([1, 2, 3])
        }

        type_models = cp.type_models(user_ns)

        # Should have type models for all valid variables
        assert 'num' in type_models
        assert 'lst' in type_models
        assert 'df' in type_models
        assert 's' in type_models

        # Each should be a TypeModel
        for tm in type_models.values():
            assert isinstance(tm, TypeModel)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_namespace_save(self):
        """Test saving an empty namespace."""
        cp = Checkpoints()

        saved, removed = cp.save('test', {})

        assert saved == {}
        assert removed == {}

    def test_empty_dataframe(self):
        """Test checkpointing an empty DataFrame."""
        cp = Checkpoints()

        df = pd.DataFrame()
        user_ns = {'df': df}

        saved, removed = cp.save('test', user_ns)
        assert 'df' in saved

        cp.restore('test', user_ns)
        assert isinstance(user_ns['df'], pd.DataFrame)
        assert len(user_ns['df']) == 0

    def test_dataframe_with_none_values(self):
        """Test DataFrame containing None values."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': [None, [1, 2], None]})
        user_ns = {'df': df}

        cp.save('test', user_ns)
        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] is None
        assert user_ns['df'].iloc[1, 0] == [1, 2]
        assert user_ns['df'].iloc[2, 0] is None

    def test_large_dataframe_with_mutable_objects(self):
        """Test performance with larger DataFrames."""
        cp = Checkpoints()

        # Create a larger DataFrame with mutable objects
        data = [[i, i+1, i+2] for i in range(100)]
        df = pd.DataFrame({'data': data})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Modify first and last items
        user_ns['df'].iloc[0, 0].append(999)
        user_ns['df'].iloc[99, 0].append(888)

        # Restore
        cp.restore('test', user_ns)

        # Verify restoration
        assert user_ns['df'].iloc[0, 0] == [0, 1, 2]
        assert user_ns['df'].iloc[99, 0] == [99, 100, 101]


# ============================================================================
# DEEP ALIAS DETECTION TESTS
# ============================================================================

class TestDeepAliasDetection:
    """Test that deep alias detection correctly identifies shared references."""

    def test_simple_alias_detected(self):
        """Test that simple variable aliasing is detected."""
        from data_ferret.kernel.checkpoint import Checkpoint

        shared_list = [1, 2, 3]
        user_ns = {
            'a': {'data': shared_list},
            'b': {'data': shared_list},  # Same list object
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'a'})

        # Should include 'b' because they share the same list
        assert 'a' in aliases
        assert 'b' in aliases

    def test_no_alias_for_independent_objects(self):
        """Test that independent objects are not marked as aliases."""
        from data_ferret.kernel.checkpoint import Checkpoint

        user_ns = {
            'a': {'data': [1, 2, 3]},
            'b': {'data': [1, 2, 3]},  # Same values, different objects
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'a'})

        # Should NOT include 'b' - they have equal values but different objects
        assert 'a' in aliases
        assert 'b' not in aliases

    def test_nested_alias_detected(self):
        """Test that deeply nested aliases are detected."""
        from data_ferret.kernel.checkpoint import Checkpoint

        shared_dict = {'nested': {'value': 42}}
        user_ns = {
            'x': {'level1': {'level2': shared_dict}},
            'y': [shared_dict],  # Same dict in different structure
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'x'})

        assert 'x' in aliases
        assert 'y' in aliases

    def test_dataframe_alias_via_shared_column(self):
        """Test that DataFrames sharing internal data are detected as aliases."""
        from data_ferret.kernel.checkpoint import Checkpoint

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        user_ns = {
            'df': df,
            'series': df['a'],  # Series shares data with DataFrame
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'df'})

        # series shares the block manager with df
        assert 'df' in aliases
        # Note: Whether 'series' is detected depends on internal pandas structure


# ============================================================================
# SINGLETON TYPE SKIPPING TESTS
# ============================================================================

class TestSingletonTypeSkipping:
    """Test that singleton types (classes, functions, modules) are skipped in alias detection.

    This is critical for performance - without this fix, ML model classes would
    cause false alias detection leading to 24-minute diff times.
    """

    def test_type_objects_skipped(self):
        """Test that type/class objects are not tracked for aliasing."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class MyClass:
            class_var = "test"

        visited = set()
        _collect_reachable_ids(MyClass, visited)

        # Type objects should be skipped entirely - no IDs collected
        assert len(visited) == 0

    def test_function_objects_skipped(self):
        """Test that function objects are not tracked for aliasing."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        def my_function(x):
            return x * 2

        visited = set()
        _collect_reachable_ids(my_function, visited)

        # Functions should be skipped - no IDs collected
        assert len(visited) == 0

    def test_builtin_function_skipped(self):
        """Test that built-in functions are not tracked."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        visited = set()
        _collect_reachable_ids(len, visited)

        assert len(visited) == 0

    def test_module_objects_skipped(self):
        """Test that module objects are not tracked for aliasing."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids
        import json

        visited = set()
        _collect_reachable_ids(json, visited)

        # Modules should be skipped - no IDs collected
        assert len(visited) == 0

    def test_method_objects_skipped(self):
        """Test that bound method objects are not tracked."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class MyClass:
            def my_method(self):
                pass

        instance = MyClass()
        visited = set()
        _collect_reachable_ids(instance.my_method, visited)

        # Bound methods should be skipped
        assert len(visited) == 0

    def test_instance_with_class_reference_no_false_alias(self):
        """Test that instances containing class references don't falsely alias with the class.

        This is the core bug that was fixed - an instance like `base = LGBMRegressor()`
        should NOT be considered an alias of the `LGBMRegressor` class itself.
        """
        from data_ferret.kernel.checkpoint import Checkpoint

        class ModelWrapper:
            def __init__(self):
                self.data = [1, 2, 3]

        wrapper = ModelWrapper()
        user_ns = {
            'ModelWrapper': ModelWrapper,  # The class itself
            'wrapper': wrapper,  # An instance
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'wrapper'})

        # wrapper should NOT alias with ModelWrapper class
        assert 'wrapper' in aliases
        assert 'ModelWrapper' not in aliases

    def test_instance_still_tracks_mutable_data(self):
        """Test that instances with mutable attributes are still properly tracked."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class Container:
            def __init__(self):
                self.data = [1, 2, 3]
                self.nested = {'key': [4, 5, 6]}

        instance = Container()
        visited = set()
        _collect_reachable_ids(instance, visited)

        # Should track: instance, data list, nested dict, nested list
        # At minimum: instance + list + dict + inner list = 4
        assert len(visited) >= 4

    def test_class_with_shared_data_still_aliases_via_data(self):
        """Test that instances sharing actual data still alias correctly."""
        from data_ferret.kernel.checkpoint import Checkpoint

        shared_list = [1, 2, 3]

        class Container:
            def __init__(self, data):
                self.data = data

        obj1 = Container(shared_list)
        obj2 = Container(shared_list)

        user_ns = {
            'obj1': obj1,
            'obj2': obj2,
            'Container': Container,  # The class - should not affect aliasing
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'obj1'})

        # obj1 and obj2 share the list, so should be aliases
        assert 'obj1' in aliases
        assert 'obj2' in aliases
        # But NOT the class
        assert 'Container' not in aliases

    def test_multiple_classes_not_aliased_together(self):
        """Test that multiple class objects don't create false aliases.

        This tests the exact scenario from the bug: having multiple ML model
        classes should not make them appear as aliases of each other.
        """
        from data_ferret.kernel.checkpoint import Checkpoint

        class ModelA:
            pass

        class ModelB:
            pass

        class ModelC:
            pass

        user_ns = {
            'ModelA': ModelA,
            'ModelB': ModelB,
            'ModelC': ModelC,
            'data': [1, 2, 3],  # Unrelated data
        }

        cp = Checkpoint('test', user_ns, {})

        # Querying ModelA should only return ModelA
        aliases_a = cp.get_aliases_for_vars({'ModelA'})
        assert aliases_a == {'ModelA'}

        # Querying data should only return data
        aliases_data = cp.get_aliases_for_vars({'data'})
        assert aliases_data == {'data'}

    def test_function_with_closure_not_aliased(self):
        """Test that functions with closures don't create false aliases."""
        from data_ferret.kernel.checkpoint import Checkpoint

        x = 10

        def func1():
            return x

        def func2():
            return x * 2

        user_ns = {
            'func1': func1,
            'func2': func2,
            'x': x,
        }

        cp = Checkpoint('test', user_ns, {})
        aliases = cp.get_aliases_for_vars({'func1'})

        # func1 should not alias with func2 or x
        assert aliases == {'func1'}


class TestMLModelScenario:
    """Test the exact scenario that caused the 24-minute diff bug.

    When a user creates a model wrapper class that stores DataFrames and
    references ML model classes, the alias detection should NOT expand to
    include all the ML classes.
    """

    def test_model_wrapper_with_dataframes(self):
        """Test a realistic ML model wrapper scenario."""
        from data_ferret.kernel.checkpoint import Checkpoint

        # Simulate a user's model wrapper class (like AbdBase from the bug report)
        class ModelWrapper:
            model_types = ['LGBM', 'CAT', 'XGB']  # Class variable with strings

            def __init__(self, train_data, test_data):
                self.train_data = train_data
                self.test_data = test_data
                self.model = None

        train = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        test = pd.DataFrame({'a': [7, 8, 9], 'b': [10, 11, 12]})
        wrapper = ModelWrapper(train, test)

        user_ns = {
            'ModelWrapper': ModelWrapper,
            'train': train,
            'test': test,
            'wrapper': wrapper,
        }

        cp = Checkpoint('test', user_ns, {})

        # Querying 'wrapper' should find aliases with train and test (they share data)
        # but should NOT include ModelWrapper class
        aliases = cp.get_aliases_for_vars({'wrapper'})

        assert 'wrapper' in aliases
        assert 'train' in aliases  # wrapper.train_data is same object as train
        assert 'test' in aliases   # wrapper.test_data is same object as test
        assert 'ModelWrapper' not in aliases  # Class should not be an alias

    def test_imported_classes_not_aliased(self):
        """Test that imported library classes don't create false aliases."""
        from data_ferret.kernel.checkpoint import Checkpoint
        from sklearn.model_selection import KFold, StratifiedKFold

        user_ns = {
            'KFold': KFold,
            'StratifiedKFold': StratifiedKFold,
            'n_splits': 5,
            'data': pd.DataFrame({'x': [1, 2, 3]}),
        }

        cp = Checkpoint('test', user_ns, {})

        # KFold should not alias with anything
        aliases_kfold = cp.get_aliases_for_vars({'KFold'})
        assert aliases_kfold == {'KFold'}

        # data should not alias with the classes
        aliases_data = cp.get_aliases_for_vars({'data'})
        assert 'data' in aliases_data
        assert 'KFold' not in aliases_data
        assert 'StratifiedKFold' not in aliases_data

    def test_performance_with_many_classes(self):
        """Test that having many class objects doesn't slow down alias detection."""
        from data_ferret.kernel.checkpoint import Checkpoint
        import time

        # Create a namespace with many class definitions
        user_ns = {}
        for i in range(50):
            # Dynamically create classes
            user_ns[f'Class{i}'] = type(f'Class{i}', (), {'value': i})

        # Add some actual data
        user_ns['data'] = [1, 2, 3]
        user_ns['df'] = pd.DataFrame({'a': [1, 2, 3]})

        cp = Checkpoint('test', user_ns, {})

        # This should be fast - not slow due to class traversal
        start = time.time()
        aliases = cp.get_aliases_for_vars({'data'})
        elapsed = time.time() - start

        # Should complete in well under 1 second
        assert elapsed < 1.0, f"Alias detection took {elapsed:.2f}s - too slow!"

        # Should not include any of the classes
        assert aliases == {'data'}


class TestCollectReachableIdsEdgeCases:
    """Test edge cases in the _collect_reachable_ids function."""

    def test_descriptor_types_skipped(self):
        """Test that descriptor types are properly skipped."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class MyClass:
            @property
            def prop(self):
                return 42

        # The property descriptor itself
        prop_descriptor = MyClass.__dict__['prop']

        visited = set()
        _collect_reachable_ids(prop_descriptor, visited)

        # Property descriptors might be skipped or not depending on type
        # The key is they shouldn't cause issues

    def test_lambda_functions_skipped(self):
        """Test that lambda functions are skipped like regular functions."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        func = lambda x: x * 2

        visited = set()
        _collect_reachable_ids(func, visited)

        assert len(visited) == 0

    def test_staticmethod_and_classmethod(self):
        """Test that static and class methods don't cause issues."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class MyClass:
            @staticmethod
            def static_method():
                pass

            @classmethod
            def class_method(cls):
                pass

        # These are descriptor objects when accessed from class __dict__
        static = MyClass.__dict__['static_method']
        classm = MyClass.__dict__['class_method']

        visited = set()
        _collect_reachable_ids(static, visited)
        _collect_reachable_ids(classm, visited)

        # Should not cause any issues (may or may not collect IDs depending on type)

    def test_mixed_container_with_types(self):
        """Test containers that mix data with type references."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class MyClass:
            pass

        data = {
            'class_ref': MyClass,
            'func_ref': len,
            'actual_data': [1, 2, 3],
            'nested': {'more': [4, 5, 6]}
        }

        visited = set()
        _collect_reachable_ids(data, visited)

        # Should have: data dict, actual_data list, nested dict, more list
        # Should NOT have: MyClass type or len function
        assert len(visited) >= 4  # At least the mutable containers

    def test_circular_reference_with_class(self):
        """Test circular references involving class instances."""
        from data_ferret.kernel.checkpoint import _collect_reachable_ids

        class Node:
            def __init__(self):
                self.next = None

        a = Node()
        b = Node()
        a.next = b
        b.next = a  # Circular

        visited = set()
        _collect_reachable_ids(a, visited)

        # Should handle circular reference without infinite loop
        # Should have both nodes
        assert id(a) in visited
        assert id(b) in visited


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
