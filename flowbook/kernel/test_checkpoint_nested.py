"""
Tests for nested DataFrame/Series/Function deep copying in checkpoints.

This module tests that the custom deepcopy correctly handles DataFrames,
Series, and Functions at any nesting level within custom objects.
"""

import pandas as pd
import pytest

from flowbook.kernel.checkpoint import Checkpoints


class TestNestedDataFrames:
    """Test that DataFrames nested in objects are properly deep copied."""

    def test_nested_dataframe_with_mutable_objects(self):
        """Test DataFrame nested in an object with mutable object columns."""
        class Container:
            def __init__(self, df):
                self.data = df

        # Create DataFrame with object column containing mutable lists
        df = pd.DataFrame({'col': [[1, 2], [3, 4]]})
        container = Container(df)

        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Mutate the original
        container.data['col'].iloc[0].append(999)

        # Restore and verify isolation
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['container'].data['col'].iloc[0]
        assert user_ns['container'].data['col'].iloc[0] == [1, 2]

    def test_nested_dataframe_with_immutable_columns(self):
        """Test DataFrame nested in an object with immutable columns (optimization)."""
        class Container:
            def __init__(self, df):
                self.data = df

        # Create DataFrame with object column containing only strings (immutable)
        df = pd.DataFrame({'col': ['a', 'b', 'c']})
        container = Container(df)

        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Modify the DataFrame
        container.data['col'].iloc[0] = 'z'

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert user_ns['container'].data['col'].iloc[0] == 'a'

    def test_deeply_nested_dataframe(self):
        """Test DataFrame nested 3 levels deep."""
        class Inner:
            def __init__(self):
                self.df = pd.DataFrame({'col': [[1, 2]]})

        class Middle:
            def __init__(self):
                self.inner = Inner()

        class Outer:
            def __init__(self):
                self.middle = Middle()

        outer = Outer()
        checkpoints = Checkpoints()
        user_ns = {'outer': outer}
        checkpoints.save('test', user_ns)

        # Mutate deeply nested mutable object
        outer.middle.inner.df['col'].iloc[0].append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['outer'].middle.inner.df['col'].iloc[0]
        assert user_ns['outer'].middle.inner.df['col'].iloc[0] == [1, 2]

    def test_dataframe_in_list(self):
        """Test DataFrame stored in a list."""
        df1 = pd.DataFrame({'col': [[1, 2]]})
        df2 = pd.DataFrame({'col': [[3, 4]]})
        data_list = [df1, df2]

        checkpoints = Checkpoints()
        user_ns = {'data_list': data_list}
        checkpoints.save('test', user_ns)

        # Mutate
        data_list[0]['col'].iloc[0].append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['data_list'][0]['col'].iloc[0]

    def test_dataframe_in_dict(self):
        """Test DataFrame stored in a dict."""
        df = pd.DataFrame({'col': [[1, 2]]})
        data_dict = {'key1': df}

        checkpoints = Checkpoints()
        user_ns = {'data_dict': data_dict}
        checkpoints.save('test', user_ns)

        # Mutate
        data_dict['key1']['col'].iloc[0].append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['data_dict']['key1']['col'].iloc[0]


class TestNestedSeries:
    """Test that Series nested in objects are properly deep copied."""

    def test_nested_series_with_objects(self):
        """Test Series nested in an object."""
        class Container:
            def __init__(self):
                self.series = pd.Series([{'a': 1}, {'b': 2}])

        container = Container()
        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Mutate
        container.series.iloc[0]['c'] = 3

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert 'c' not in user_ns['container'].series.iloc[0]
        assert user_ns['container'].series.iloc[0] == {'a': 1}

    def test_series_in_list(self):
        """Test Series stored in a list."""
        s1 = pd.Series([[1, 2]])
        s2 = pd.Series([[3, 4]])
        series_list = [s1, s2]

        checkpoints = Checkpoints()
        user_ns = {'series_list': series_list}
        checkpoints.save('test', user_ns)

        # Mutate
        series_list[0].iloc[0].append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['series_list'][0].iloc[0]


class TestNestedFunctions:
    """Test that functions nested in objects are properly deep copied."""

    def test_nested_function_with_closure(self):
        """Test function nested in an object with closure."""
        class CallbackContainer:
            def __init__(self):
                data = [1, 2, 3]
                self.callback = lambda: data

        container = CallbackContainer()
        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Mutate closure
        original_data = container.callback()
        original_data.append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert user_ns['container'].callback() == [1, 2, 3]

    def test_nested_function_with_mutable_default(self):
        """Test function with mutable default argument nested in object."""
        class Container:
            def __init__(self):
                def accumulate(val, acc=[]):
                    acc.append(val)
                    return acc
                self.func = accumulate

        container = Container()
        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Call function to mutate default
        container.func(1)
        container.func(2)

        # Restore and verify default is reset
        checkpoints.restore('test', user_ns)
        result = user_ns['container'].func(100)
        assert result == [100]  # Not [1, 2, 100]

    def test_function_in_list(self):
        """Test function stored in a list."""
        data = [1, 2, 3]
        func = lambda: data
        func_list = [func]

        checkpoints = Checkpoints()
        user_ns = {'func_list': func_list}
        checkpoints.save('test', user_ns)

        # Mutate closure
        data.append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert user_ns['func_list'][0]() == [1, 2, 3]


class TestCircularReferences:
    """Test that circular references work with nested structures."""

    def test_circular_reference_with_dataframe(self):
        """Test circular references with nested DataFrames."""
        class Node:
            def __init__(self, name):
                self.name = name
                self.df = pd.DataFrame({'col': [[1, 2]]})
                self.next = None

        node1 = Node('node1')
        node2 = Node('node2')
        node1.next = node2
        node2.next = node1  # Circular!

        checkpoints = Checkpoints()
        user_ns = {'node1': node1, 'node2': node2}
        checkpoints.save('test', user_ns)

        # Mutate
        node1.df['col'].iloc[0].append(999)

        # Restore and verify circular structure preserved
        checkpoints.restore('test', user_ns)
        assert user_ns['node1'].next is user_ns['node2']
        assert user_ns['node2'].next is user_ns['node1']
        assert 999 not in user_ns['node1'].df['col'].iloc[0]

    def test_self_referential_structure(self):
        """Test object that references itself."""
        class SelfRef:
            def __init__(self):
                self.data = [1, 2, 3]
                self.self_ref = None

        obj = SelfRef()
        obj.self_ref = obj

        checkpoints = Checkpoints()
        user_ns = {'obj': obj}
        checkpoints.save('test', user_ns)

        # Mutate
        obj.data.append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert user_ns['obj'].self_ref is user_ns['obj']
        assert user_ns['obj'].data == [1, 2, 3]


class TestMixedNestedStructures:
    """Test complex nested structures with mixed types."""

    def test_dataframe_with_series_and_function(self):
        """Test object containing DataFrame, Series, and function."""
        class MixedContainer:
            def __init__(self):
                self.df = pd.DataFrame({'col': [[1, 2]]})
                self.series = pd.Series([{'a': 1}])
                data = [10, 20]
                self.func = lambda: data

        container = MixedContainer()
        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Mutate all three
        container.df['col'].iloc[0].append(999)
        container.series.iloc[0]['b'] = 999
        container.func().append(999)

        # Restore and verify all are isolated
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['container'].df['col'].iloc[0]
        assert 'b' not in user_ns['container'].series.iloc[0]
        assert user_ns['container'].func() == [10, 20]

    def test_nested_containers_with_dataframes(self):
        """Test nested lists/dicts containing DataFrames."""
        df = pd.DataFrame({'col': [[1, 2]]})
        structure = {
            'list': [df],
            'dict': {'inner': df},
            'nested': [[df]]
        }

        checkpoints = Checkpoints()
        user_ns = {'structure': structure}
        checkpoints.save('test', user_ns)

        # Mutate the DataFrame (all references should point to same object)
        df['col'].iloc[0].append(999)

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert 999 not in user_ns['structure']['list'][0]['col'].iloc[0]
        assert 999 not in user_ns['structure']['dict']['inner']['col'].iloc[0]
        assert 999 not in user_ns['structure']['nested'][0][0]['col'].iloc[0]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe_nested(self):
        """Test empty DataFrame in nested structure."""
        class Container:
            def __init__(self):
                self.df = pd.DataFrame()

        container = Container()
        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)
        checkpoints.restore('test', user_ns)

        assert user_ns['container'].df.empty

    def test_object_with_slots(self):
        """Test object using __slots__ instead of __dict__."""
        class SlotsClass:
            __slots__ = ['data', 'value']

            def __init__(self):
                self.data = [1, 2, 3]
                self.value = 42

        obj = SlotsClass()
        checkpoints = Checkpoints()
        user_ns = {'obj': obj}
        checkpoints.save('test', user_ns)

        # Mutate
        obj.data.append(999)
        obj.value = 0

        # Restore and verify
        checkpoints.restore('test', user_ns)
        assert user_ns['obj'].data == [1, 2, 3]
        assert user_ns['obj'].value == 42

    def test_nested_dataframe_shared_references(self):
        """Test that shared DataFrame references are preserved."""
        df = pd.DataFrame({'col': [[1, 2]]})

        class Container:
            def __init__(self, df):
                self.df1 = df
                self.df2 = df  # Same reference

        container = Container(df)
        checkpoints = Checkpoints()
        user_ns = {'container': container}
        checkpoints.save('test', user_ns)

        # Restore and verify both point to same copied object
        checkpoints.restore('test', user_ns)
        assert user_ns['container'].df1 is user_ns['container'].df2

        # Mutate one and verify other changes too (shared reference)
        user_ns['container'].df1['col'].iloc[0].append(999)
        assert 999 in user_ns['container'].df2['col'].iloc[0]


class TestPerformance:
    """Test that performance is reasonable for nested structures."""

    def test_deeply_nested_list_performance(self):
        """Test performance with very deep nesting."""
        # Create a deeply nested structure
        current = [1, 2, 3]
        for _ in range(10):
            current = [current]

        checkpoints = Checkpoints()
        user_ns = {'nested': current}

        # This should complete without stack overflow
        checkpoints.save('test', user_ns)
        checkpoints.restore('test', user_ns)

        # Verify structure is correct
        restored = user_ns['nested']
        for _ in range(10):
            assert isinstance(restored, list)
            restored = restored[0]
        assert restored == [1, 2, 3]

    def test_large_nested_structure(self):
        """Test performance with many nested objects."""
        class Container:
            def __init__(self, df):
                self.df = df

        # Create 100 containers with DataFrames
        containers = [Container(pd.DataFrame({'col': [[i]]})) for i in range(100)]

        checkpoints = Checkpoints()
        user_ns = {'containers': containers}

        # Should complete in reasonable time
        checkpoints.save('test', user_ns)
        checkpoints.restore('test', user_ns)

        # Verify all are independent
        assert len(user_ns['containers']) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
