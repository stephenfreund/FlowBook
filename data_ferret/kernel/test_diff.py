"""
Comprehensive test suite for data_ferret.kernel.diff.Diff
Run with: python test_diff.py
or: pytest test_diff.py -v
"""

import pytest
import numpy as np
import pandas as pd
import math
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.extra.pandas import column, data_frames, series
import sys

# Import the Diff class
from data_ferret.kernel.diff import Diff


# ============================================================================
# TEST SUITE
# ============================================================================

class TestBasicTypes:
    """Test comparison of basic Python types."""
    
    def test_empty_namespaces(self):
        differ = Diff()
        assert differ.diff({}, {}) == {}
    
    def test_none_equal(self):
        differ = Diff()
        a = {'x': None}
        b = {'x': None}
        assert differ.diff(a, b) == {}
    
    def test_bool_equal(self):
        differ = Diff()
        a = {'flag': True}
        b = {'flag': True}
        assert differ.diff(a, b) == {}
    
    def test_bool_not_equal(self):
        differ = Diff()
        a = {'flag': True}
        b = {'flag': False}
        result = differ.diff(a, b)
        assert 'flag' in result
        assert 'Bool mismatch' in result['flag']
    
    def test_int_equal(self):
        differ = Diff()
        a = {'x': 42}
        b = {'x': 42}
        assert differ.diff(a, b) == {}
    
    def test_int_not_equal(self):
        differ = Diff()
        a = {'x': 42}
        b = {'x': 43}
        result = differ.diff(a, b)
        assert 'x' in result
        assert 'Integer mismatch' in result['x']
    
    def test_float_equal(self):
        differ = Diff()
        a = {'pi': 3.14159}
        b = {'pi': 3.14159}
        assert differ.diff(a, b) == {}
    
    def test_float_close_enough(self):
        differ = Diff(rtol=1e-6)
        a = {'x': 1.0000001}
        b = {'x': 1.0000002}
        assert differ.diff(a, b) == {}
    
    def test_float_not_close_enough(self):
        differ = Diff(rtol=1e-9, atol=0)
        a = {'x': 1.0000001}
        b = {'x': 1.0000002}
        result = differ.diff(a, b)
        assert 'x' in result
    
    def test_nan_equality(self):
        differ = Diff()
        a = {'x': float('nan')}
        b = {'x': float('nan')}
        assert differ.diff(a, b) == {}
    
    def test_nan_vs_number(self):
        differ = Diff()
        a = {'x': float('nan')}
        b = {'x': 1.0}
        result = differ.diff(a, b)
        assert 'x' in result
        assert 'one is NaN' in result['x']
    
    def test_complex_equal(self):
        differ = Diff()
        a = {'z': 3 + 4j}
        b = {'z': 3 + 4j}
        assert differ.diff(a, b) == {}
    
    def test_complex_with_nan(self):
        differ = Diff()
        a = {'z': complex(float('nan'), 1.0)}
        b = {'z': complex(float('nan'), 1.0)}
        assert differ.diff(a, b) == {}
    
    def test_string_equal(self):
        differ = Diff()
        a = {'msg': 'hello'}
        b = {'msg': 'hello'}
        assert differ.diff(a, b) == {}
    
    def test_string_not_equal(self):
        differ = Diff()
        a = {'msg': 'hello'}
        b = {'msg': 'world'}
        result = differ.diff(a, b)
        assert 'msg' in result
        assert 'String mismatch' in result['msg']
    
    def test_bytes_equal(self):
        differ = Diff()
        a = {'data': b'hello'}
        b = {'data': b'hello'}
        assert differ.diff(a, b) == {}
    
    def test_bytes_not_equal(self):
        differ = Diff()
        a = {'data': b'hello'}
        b = {'data': b'world'}
        result = differ.diff(a, b)
        assert 'data' in result


class TestCollections:
    """Test comparison of collection types."""
    
    def test_list_equal(self):
        differ = Diff()
        a = {'lst': [1, 2, 3]}
        b = {'lst': [1, 2, 3]}
        assert differ.diff(a, b) == {}
    
    def test_list_different_length(self):
        differ = Diff()
        a = {'lst': [1, 2, 3]}
        b = {'lst': [1, 2]}
        result = differ.diff(a, b)
        assert 'lst' in result
        assert 'length mismatch' in result['lst']
    
    def test_list_different_values(self):
        differ = Diff()
        a = {'lst': [1, 2, 3]}
        b = {'lst': [1, 2, 4]}
        result = differ.diff(a, b)
        assert 'lst' in result
    
    def test_nested_list(self):
        differ = Diff()
        a = {'lst': [[1, 2], [3, 4]]}
        b = {'lst': [[1, 2], [3, 4]]}
        assert differ.diff(a, b) == {}
    
    def test_tuple_equal(self):
        differ = Diff()
        a = {'tpl': (1, 2, 3)}
        b = {'tpl': (1, 2, 3)}
        assert differ.diff(a, b) == {}
    
    def test_tuple_not_equal(self):
        differ = Diff()
        a = {'tpl': (1, 2, 3)}
        b = {'tpl': (1, 2, 4)}
        result = differ.diff(a, b)
        assert 'tpl' in result
    
    def test_set_equal(self):
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {3, 2, 1}}  # Different order, but same set
        assert differ.diff(a, b) == {}
    
    def test_set_not_equal(self):
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 4}}
        result = differ.diff(a, b)
        assert 's' in result
    
    def test_set_with_nested_lists(self):
        """Test that sets containing mutable objects are compared recursively."""
        differ = Diff()
        inner_a1 = [1, 2]
        inner_a2 = [3, 4]
        # Can't actually put lists in sets, but we can test with tuples
        a = {'s': {(1, 2), (3, 4)}}
        b = {'s': {(3, 4), (1, 2)}}
        assert differ.diff(a, b) == {}
    
    def test_frozenset_equal(self):
        differ = Diff()
        a = {'fs': frozenset([1, 2, 3])}
        b = {'fs': frozenset([3, 2, 1])}
        assert differ.diff(a, b) == {}
    
    def test_dict_equal(self):
        differ = Diff()
        a = {'d': {'x': 1, 'y': 2}}
        b = {'d': {'x': 1, 'y': 2}}
        assert differ.diff(a, b) == {}
    
    def test_dict_missing_key(self):
        differ = Diff()
        a = {'d': {'x': 1, 'y': 2}}
        b = {'d': {'x': 1}}
        result = differ.diff(a, b)
        assert 'd' in result
        assert 'keys mismatch' in result['d']
    
    def test_dict_extra_key(self):
        differ = Diff()
        a = {'d': {'x': 1}}
        b = {'d': {'x': 1, 'y': 2}}
        result = differ.diff(a, b)
        assert 'd' in result


class TestNumpy:
    """Test comparison of NumPy arrays."""
    
    def test_array_equal(self):
        differ = Diff()
        a = {'arr': np.array([1, 2, 3])}
        b = {'arr': np.array([1, 2, 3])}
        assert differ.diff(a, b) == {}
    
    def test_array_different_shape(self):
        differ = Diff()
        a = {'arr': np.array([1, 2, 3])}
        b = {'arr': np.array([[1, 2, 3]])}
        result = differ.diff(a, b)
        assert 'arr' in result
        assert 'shape mismatch' in result['arr']
    
    def test_array_different_dtype(self):
        differ = Diff()
        a = {'arr': np.array([1, 2, 3], dtype=np.int32)}
        b = {'arr': np.array([1, 2, 3], dtype=np.int64)}
        result = differ.diff(a, b)
        assert 'arr' in result
        assert 'dtype mismatch' in result['arr']
    
    def test_array_different_values(self):
        differ = Diff()
        a = {'arr': np.array([1, 2, 3])}
        b = {'arr': np.array([1, 2, 4])}
        result = differ.diff(a, b)
        assert 'arr' in result
    
    def test_float_array_with_nan(self):
        differ = Diff()
        a = {'arr': np.array([1.0, np.nan, 3.0])}
        b = {'arr': np.array([1.0, np.nan, 3.0])}
        assert differ.diff(a, b) == {}
    
    def test_float_array_close(self):
        differ = Diff(rtol=1e-6)
        a = {'arr': np.array([1.0000001, 2.0000001])}
        b = {'arr': np.array([1.0000002, 2.0000002])}
        assert differ.diff(a, b) == {}
    
    def test_multidimensional_array(self):
        differ = Diff()
        a = {'arr': np.array([[1, 2], [3, 4]])}
        b = {'arr': np.array([[1, 2], [3, 4]])}
        assert differ.diff(a, b) == {}


class TestPandas:
    """Test comparison of Pandas objects."""
    
    def test_series_equal(self):
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3])}
        b = {'s': pd.Series([1, 2, 3])}
        assert differ.diff(a, b) == {}
    
    def test_series_different_index(self):
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1, 2, 3], index=['x', 'y', 'z'])}
        result = differ.diff(a, b)
        assert 's' in result
        assert 'index mismatch' in result['s']
    
    def test_series_different_name(self):
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], name='foo')}
        b = {'s': pd.Series([1, 2, 3], name='bar')}
        result = differ.diff(a, b)
        assert 's' in result
        assert 'name mismatch' in result['s']
    
    def test_series_with_nan(self):
        differ = Diff()
        a = {'s': pd.Series([1.0, np.nan, 3.0])}
        b = {'s': pd.Series([1.0, np.nan, 3.0])}
        assert differ.diff(a, b) == {}
    
    def test_series_different_nan_positions(self):
        differ = Diff()
        a = {'s': pd.Series([1.0, np.nan, 3.0])}
        b = {'s': pd.Series([1.0, 2.0, np.nan])}
        result = differ.diff(a, b)
        assert 's' in result
    
    def test_dataframe_equal(self):
        differ = Diff()
        a = {'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})}
        b = {'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})}
        assert differ.diff(a, b) == {}
    
    def test_dataframe_different_shape(self):
        differ = Diff()
        a = {'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})}
        b = {'df': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})}
        result = differ.diff(a, b)
        assert 'df' in result
        assert 'shape mismatch' in result['df']
    
    def test_dataframe_different_columns(self):
        differ = Diff()
        a = {'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})}
        b = {'df': pd.DataFrame({'A': [1, 2], 'C': [3, 4]})}
        result = differ.diff(a, b)
        assert 'df' in result
        assert 'columns mismatch' in result['df']
    
    def test_dataframe_with_nan(self):
        differ = Diff()
        a = {'df': pd.DataFrame({'A': [1.0, np.nan], 'B': [3.0, 4.0]})}
        b = {'df': pd.DataFrame({'A': [1.0, np.nan], 'B': [3.0, 4.0]})}
        assert differ.diff(a, b) == {}


class TestPointerStructure:
    """Test isomorphic pointer structure checking."""
    
    def test_simple_reference(self):
        differ = Diff()
        lst = [1, 2, 3]
        a = {'x': lst, 'y': lst}  # Same object
        
        lst2 = [1, 2, 3]
        b = {'x': lst2, 'y': lst2}  # Same object in b too
        
        assert differ.diff(a, b) == {}
    
    def test_broken_reference(self):
        differ = Diff()
        lst = [1, 2, 3]
        a = {'x': lst, 'y': lst}  # Same object
        
        b = {'x': [1, 2, 3], 'y': [1, 2, 3]}  # Different objects
        
        result = differ.diff(a, b)
        assert 'y' in result
        assert 'Pointer structure mismatch' in result['y']
    
    def test_mismatched_then_correct_reference(self):
        """
        Test that if object A is compared with wrong object B1,
        then later compared with correct object B2, we don't get
        a false pointer structure error.
        """
        differ = Diff()
        obj_a = [1, 2, 3]
        obj_b_wrong = [1, 2, 999]  # Different value
        obj_b_correct = [1, 2, 3]  # Correct value
        
        a = {'x': obj_a, 'y': obj_a}
        b = {'x': obj_b_wrong, 'y': obj_b_correct}
        
        result = differ.diff(a, b)
        # Should report 'x' is different (values don't match)
        assert 'x' in result
        assert 'Integer mismatch' in result['x'] or 'values mismatch' in result['x']
        # Should NOT report pointer structure mismatch for 'y'
        # since both obj_a and obj_b_correct haven't been successfully matched before
        if 'y' in result:
            assert 'Pointer structure mismatch' not in result['y']
    
    def test_nested_reference(self):
        differ = Diff()
        inner = [1, 2]
        lst = [inner, inner]
        a = {'x': lst}
        
        inner2 = [1, 2]
        lst2 = [inner2, inner2]
        b = {'x': lst2}
        
        assert differ.diff(a, b) == {}
    
    def test_circular_reference(self):
        differ = Diff()
        lst_a = [1, 2]
        lst_a.append(lst_a)  # Circular reference
        a = {'x': lst_a}
        
        lst_b = [1, 2]
        lst_b.append(lst_b)  # Circular reference
        b = {'x': lst_b}
        
        assert differ.diff(a, b) == {}
    
    def test_dict_reference(self):
        differ = Diff()
        d = {'key': 'value'}
        a = {'x': d, 'y': d}
        
        d2 = {'key': 'value'}
        b = {'x': d2, 'y': d2}
        
        assert differ.diff(a, b) == {}
    
    def test_set_with_shared_references(self):
        """Test pointer structure within sets is preserved."""
        differ = Diff()
        
        # Create shared object
        shared_a = [1, 2]
        # Use tuples containing the same list (conceptually)
        a = {'items': ([shared_a],), 'ref': shared_a}
        
        shared_b = [1, 2]
        b = {'items': ([shared_b],), 'ref': shared_b}
        
        assert differ.diff(a, b) == {}


class TestUserObjects:
    """Test comparison of user-defined objects."""
    
    def test_simple_object_equal(self):
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        differ = Diff()
        a = {'p': Point(1, 2)}
        b = {'p': Point(1, 2)}
        assert differ.diff(a, b) == {}
    
    def test_simple_object_not_equal(self):
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        differ = Diff()
        a = {'p': Point(1, 2)}
        b = {'p': Point(1, 3)}
        result = differ.diff(a, b)
        assert 'p' in result
    
    def test_nested_object(self):
        class Inner:
            def __init__(self, val):
                self.val = val
        
        class Outer:
            def __init__(self, inner):
                self.inner = inner
        
        differ = Diff()
        a = {'obj': Outer(Inner(42))}
        b = {'obj': Outer(Inner(42))}
        assert differ.diff(a, b) == {}
    
    def test_object_with_numpy(self):
        class Container:
            def __init__(self, arr):
                self.arr = arr
        
        differ = Diff()
        a = {'c': Container(np.array([1, 2, 3]))}
        b = {'c': Container(np.array([1, 2, 3]))}
        assert differ.diff(a, b) == {}
    
    def test_object_with_shared_reference(self):
        """Test that shared references in objects are tracked."""
        class Container:
            def __init__(self, item1, item2):
                self.item1 = item1
                self.item2 = item2
        
        differ = Diff()
        shared_a = [1, 2, 3]
        a = {'c': Container(shared_a, shared_a)}
        
        shared_b = [1, 2, 3]
        b = {'c': Container(shared_b, shared_b)}
        
        assert differ.diff(a, b) == {}
    
    def test_object_with_broken_reference(self):
        """Test that broken references in objects are detected."""
        class Container:
            def __init__(self, item1, item2):
                self.item1 = item1
                self.item2 = item2
        
        differ = Diff()
        shared_a = [1, 2, 3]
        a = {'c': Container(shared_a, shared_a)}
        
        # Different objects in b
        b = {'c': Container([1, 2, 3], [1, 2, 3])}
        
        result = differ.diff(a, b)
        assert 'c' in result
        assert 'Pointer structure mismatch' in result['c']


class TestNamespaceLevel:
    """Test namespace-level differences."""
    
    def test_variable_only_in_first(self):
        differ = Diff()
        a = {'x': 1, 'y': 2}
        b = {'x': 1}
        result = differ.diff(a, b)
        assert 'y' in result
        assert 'removed' in result['y']
    
    def test_variable_only_in_second(self):
        differ = Diff()
        a = {'x': 1}
        b = {'x': 1, 'y': 2}
        result = differ.diff(a, b)
        assert 'y' in result
        assert 'added' in result['y']
    
    def test_multiple_differences(self):
        differ = Diff()
        a = {'x': 1, 'y': 2, 'z': 3}
        b = {'x': 1, 'y': 99, 'w': 4}
        result = differ.diff(a, b)
        assert 'y' in result  # Different value
        assert 'z' in result  # Removed (only in first)
        assert 'w' in result  # Added (only in second)
        assert 'x' not in result  # Same in both


class TestTypeMismatch:
    """Test type mismatches."""
    
    def test_int_vs_float(self):
        differ = Diff()
        a = {'x': 1}
        b = {'x': 1.0}
        result = differ.diff(a, b)
        assert 'x' in result
        assert 'Type mismatch' in result['x']
    
    def test_list_vs_tuple(self):
        differ = Diff()
        a = {'x': [1, 2, 3]}
        b = {'x': (1, 2, 3)}
        result = differ.diff(a, b)
        assert 'x' in result
        assert 'Type mismatch' in result['x']
    
    def test_array_vs_list(self):
        differ = Diff()
        a = {'x': np.array([1, 2, 3])}
        b = {'x': [1, 2, 3]}
        result = differ.diff(a, b)
        assert 'x' in result
        assert 'Type mismatch' in result['x']


class TestSetRecursion:
    """Test recursive comparison within sets."""
    
    def test_set_with_tuple_elements(self):
        """Sets can contain tuples, test recursive comparison."""
        differ = Diff()
        a = {'s': {(1, 2), (3, 4)}}
        b = {'s': {(3, 4), (1, 2)}}  # Different order
        assert differ.diff(a, b) == {}
    
    def test_set_nested_tuples(self):
        """Test sets with nested tuple structures."""
        differ = Diff()
        a = {'s': {((1, 2), 3), ((4, 5), 6)}}
        b = {'s': {((4, 5), 6), ((1, 2), 3)}}
        assert differ.diff(a, b) == {}
    
    def test_frozenset_with_nested(self):
        """Test frozensets with nested structures."""
        differ = Diff()
        a = {'fs': frozenset([(1, 2), (3, 4)])}
        b = {'fs': frozenset([(3, 4), (1, 2)])}
        assert differ.diff(a, b) == {}


class TestCallables:
    """Test comparison of callable objects."""
    
    def test_same_function(self):
        """Same function should be equal."""
        def foo():
            return 42
        
        differ = Diff()
        a = {'f': foo}
        b = {'f': foo}
        assert differ.diff(a, b) == {}
    
    def test_different_functions(self):
        """Different functions should not be equal."""
        def foo():
            return 42
        
        def bar():
            return 42
        
        differ = Diff()
        a = {'f': foo}
        b = {'f': bar}
        result = differ.diff(a, b)
        assert 'f' in result
        assert 'Callable mismatch' in result['f']
    
    def test_lambda_same(self):
        """Same lambda should be equal."""
        lam = lambda x: x + 1
        
        differ = Diff()
        a = {'f': lam}
        b = {'f': lam}
        assert differ.diff(a, b) == {}
    
    def test_lambda_different(self):
        """Different lambdas should not be equal even if equivalent."""
        differ = Diff()
        a = {'f': lambda x: x + 1}
        b = {'f': lambda x: x + 1}
        result = differ.diff(a, b)
        assert 'f' in result
        assert 'Callable mismatch' in result['f']
    
    def test_method_same(self):
        """Bound methods to same method on same instance should be equal."""
        class Foo:
            def bar(self):
                return 42
        
        obj = Foo()
        differ = Diff()
        # Even though these create different bound method objects,
        # they refer to the same method on the same instance
        a = {'m': obj.bar}
        b = {'m': obj.bar}
        assert differ.diff(a, b) == {}
    
    def test_method_different_instances(self):
        """Methods from different instances with different values should not be equal."""
        class Foo:
            def __init__(self, x: int):
                self.x = x
            def bar(self):
                return self.x
        
        obj1 = Foo(1)
        obj2 = Foo(2)
        differ = Diff()
        a = {'m': obj1.bar}
        b = {'m': obj2.bar}
        result = differ.diff(a, b)
        assert 'm' in result
        # Should detect that __self__ differs
        assert '__self__' in result['m'] or 'Pointer structure mismatch' in result['m']
    
    def test_method_comparable_instances(self):
        """Methods from different instances with same values should be equal."""
        class Foo:
            def __init__(self, x: int):
                self.x = x
            def bar(self):
                return self.x
        
        obj1 = Foo(1)
        obj2 = Foo(1)
        differ = Diff()
        a = {'m': obj1.bar}
        b = {'m': obj2.bar}
        result = differ.diff(a, b)
        assert result == {} 
        
    def test_builtin_same(self):
        """Same builtin function should be equal."""
        differ = Diff()
        a = {'f': len}
        b = {'f': len}
        assert differ.diff(a, b) == {}
    
    def test_builtin_different(self):
        """Different builtin functions should not be equal."""
        differ = Diff()
        a = {'f': len}
        b = {'f': sum}
        result = differ.diff(a, b)
        assert 'f' in result
        assert 'Callable mismatch' in result['f']
    
    def test_callable_reference_structure(self):
        """Pointer structure with callables should be tracked."""
        def foo():
            return 42
        
        differ = Diff()
        a = {'x': foo, 'y': foo}  # Same function object
        b = {'x': foo, 'y': foo}  # Same function object
        assert differ.diff(a, b) == {}
    
    def test_callable_broken_reference(self):
        """Broken pointer structure with callables should be detected."""
        def foo():
            return 42
        
        def bar():
            return 42
        
        differ = Diff()
        a = {'x': foo, 'y': foo}  # Same function
        b = {'x': foo, 'y': bar}  # Different functions
        result = differ.diff(a, b)
        assert 'y' in result
    
    def test_method_reference_structure(self):
        """Test that bound method aliasing is tracked through __func__ and __self__."""
        class Foo:
            def bar(self):
                return 42
        
        obj = Foo()
        differ = Diff()
        # Both create separate bound method wrappers but point to same method + instance
        a = {'x': obj.bar, 'y': obj.bar}
        b = {'x': obj.bar, 'y': obj.bar}
        assert differ.diff(a, b) == {}


class TestDetailedErrorMessages:
    """Test that error messages include specific locations and values."""
    
    def test_array_shows_index_and_values(self):
        """Array mismatch should show which index differs and the values."""
        differ = Diff()
        a = {'arr': np.array([1, 2, 3, 4, 5])}
        b = {'arr': np.array([1, 2, 99, 4, 5])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the index
        assert '[2]' in result['arr'] or '(2,)' in result['arr']
        # Should show the values
        assert '3' in result['arr'] and '99' in result['arr']
    
    def test_multidim_array_shows_index(self):
        """Multidimensional array should show full index."""
        differ = Diff()
        a = {'arr': np.array([[1, 2], [3, 4]])}
        b = {'arr': np.array([[1, 2], [3, 99]])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show 2D index
        assert '1' in result['arr'] and '1' in result['arr']
        assert '4' in result['arr'] and '99' in result['arr']
    
    def test_series_shows_label_and_values(self):
        """Series mismatch should show the index label and values."""
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1, 99, 3], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # Should show the index label
        assert "'b'" in result['s'] or 'b' in result['s']
        # Should show the values
        assert '2' in result['s'] and '99' in result['s']
    
    def test_series_nan_position_shows_label(self):
        """Series NaN position mismatch should show the label."""
        differ = Diff()
        a = {'s': pd.Series([1.0, np.nan, 3.0], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # Should show which label has the NaN mismatch
        assert "'b'" in result['s'] or 'b' in result['s']
        assert 'NaN' in result['s'] or 'is_nan' in result['s']
    
    def test_set_shows_unmatched_element(self):
        """Set mismatch should show which element couldn't be matched."""
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 99}}
        result = differ.diff(a, b)
        assert 's' in result
        # Should mention an element value (either 3 or 99)
        msg = result['s']
        assert ('3' in msg or '99' in msg)
    
    def test_float_array_shows_values(self):
        """Float array with tolerance mismatch should show values."""
        differ = Diff(rtol=1e-9, atol=0)
        a = {'arr': np.array([1.0, 2.0, 3.00001])}
        b = {'arr': np.array([1.0, 2.0, 3.00002])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the differing values
        assert '3.0000' in result['arr']
        """Array mismatch should show which index differs and the values."""
        differ = Diff()
        a = {'arr': np.array([1, 2, 3, 4, 5])}
        b = {'arr': np.array([1, 2, 99, 4, 5])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the index
        assert '[2]' in result['arr'] or '(2,)' in result['arr']
        # Should show the values
        assert '3' in result['arr'] and '99' in result['arr']
    
    def test_multidim_array_shows_index(self):
        """Multidimensional array should show full index."""
        differ = Diff()
        a = {'arr': np.array([[1, 2], [3, 4]])}
        b = {'arr': np.array([[1, 2], [3, 99]])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show 2D index
        assert '1' in result['arr'] and '1' in result['arr']
        assert '4' in result['arr'] and '99' in result['arr']
    
    def test_series_shows_label_and_values(self):
        """Series mismatch should show the index label and values."""
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1, 99, 3], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # Should show the index label
        assert "'b'" in result['s'] or 'b' in result['s']
        # Should show the values
        assert '2' in result['s'] and '99' in result['s']
    
    def test_series_nan_position_shows_label(self):
        """Series NaN position mismatch should show the label."""
        differ = Diff()
        a = {'s': pd.Series([1.0, np.nan, 3.0], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # Should show which label has the NaN mismatch
        assert "'b'" in result['s'] or 'b' in result['s']
        assert 'NaN' in result['s'] or 'is_nan' in result['s']
    
    def test_set_shows_unmatched_element(self):
        """Set mismatch should show which element couldn't be matched."""
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 99}}
        result = differ.diff(a, b)
        assert 's' in result
        # Should mention an element value (either 3 or 99)
        msg = result['s']
        assert ('3' in msg or '99' in msg)
    
    def test_float_array_shows_values(self):
        """Float array with tolerance mismatch should show values."""
        differ = Diff(rtol=1e-9, atol=0)
        a = {'arr': np.array([1.0, 2.0, 3.00001])}
        b = {'arr': np.array([1.0, 2.0, 3.00002])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the differing values
        assert '3.0000' in result['arr']


# ============================================================================
# GROUPBY TESTS
# ============================================================================

class TestGroupBy:
    """Test comparison of pandas GroupBy objects."""

    def test_dataframe_groupby_equal(self):
        """Two equivalent DataFrameGroupBy objects should be equal."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7], 'C': [8, 9, 10, 11]})

        # Create two groupby objects
        gb_a = df.groupby('A')
        gb_b = df.groupby('A')

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        # Should be equal even though they have different cache states
        assert differ.diff(a, b) == {}

    def test_dataframe_groupby_different_keys(self):
        """GroupBy objects with different keys should be different."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7], 'C': [8, 9, 10, 11]})

        gb_a = df.groupby('A')
        gb_b = df.groupby('B')

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        result = differ.diff(a, b)
        assert 'gb' in result
        assert 'Grouping' in result['gb'] or 'name' in result['gb']

    def test_dataframe_groupby_different_data(self):
        """GroupBy objects with different underlying data should be different."""
        differ = Diff()
        df_a = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
        df_b = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 8]})  # Different B value

        gb_a = df_a.groupby('A')
        gb_b = df_b.groupby('A')

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        result = differ.diff(a, b)
        assert 'gb' in result
        assert 'DataFrame' in result['gb'] or 'Series' in result['gb']

    def test_dataframe_groupby_sort_difference(self):
        """GroupBy objects with different sort flags should be different."""
        differ = Diff()
        df = pd.DataFrame({'A': [2, 1, 2, 1], 'B': [4, 5, 6, 7]})

        gb_a = df.groupby('A', sort=True)
        gb_b = df.groupby('A', sort=False)

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        result = differ.diff(a, b)
        assert 'gb' in result
        assert 'sort' in result['gb']

    def test_dataframe_groupby_dropna_difference(self):
        """GroupBy objects with different dropna flags should be different."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, None], 'B': [4, 5, 6, 7]})

        gb_a = df.groupby('A', dropna=True)
        gb_b = df.groupby('A', dropna=False)

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        result = differ.diff(a, b)
        assert 'gb' in result
        assert 'dropna' in result['gb']

    def test_dataframe_groupby_with_cache_access(self):
        """GroupBy objects should be equal even after cache is populated."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})

        gb_a = df.groupby('A')
        gb_b = df.groupby('A')

        # Access some properties to populate caches
        _ = gb_a.ngroups  # This will populate some cache
        # gb_b's cache remains empty

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        # Should still be equal despite different cache states
        assert differ.diff(a, b) == {}

    def test_series_groupby_equal(self):
        """Two equivalent SeriesGroupBy objects should be equal."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})

        gb_a = df.groupby('A')['B']
        gb_b = df.groupby('A')['B']

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        assert differ.diff(a, b) == {}

    def test_groupby_multiple_keys(self):
        """GroupBy with multiple keys should compare correctly."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [1, 2, 1, 2], 'C': [4, 5, 6, 7]})

        gb_a = df.groupby(['A', 'B'])
        gb_b = df.groupby(['A', 'B'])

        a = {'gb': gb_a}
        b = {'gb': gb_b}

        assert differ.diff(a, b) == {}

    def test_groupby_pointer_structure(self):
        """Pointer structure should be maintained for GroupBy objects."""
        differ = Diff()
        df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})

        gb = df.groupby('A')

        # Create references in a
        a = {'gb1': gb, 'gb2': gb}

        # Create matching references in b
        gb_copy = df.groupby('A')
        b = {'gb1': gb_copy, 'gb2': gb_copy}

        # Should be equal with matching pointer structure
        assert differ.diff(a, b) == {}

        # Break pointer structure
        gb_copy2 = df.groupby('A')
        b['gb2'] = gb_copy2

        result = differ.diff(a, b)
        assert 'gb2' in result
        assert 'Pointer structure mismatch' in result['gb2']


# ============================================================================
# HYPOTHESIS PROPERTY TESTS
# ============================================================================

class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(st.integers())
    def test_reflexivity_int(self, x):
        """A namespace should equal itself."""
        differ = Diff()
        a = {'x': x}
        assert differ.diff(a, a) == {}
    
    @given(st.floats(allow_nan=True, allow_infinity=True))
    def test_reflexivity_float(self, x):
        """A namespace should equal itself, even with NaN."""
        differ = Diff()
        a = {'x': x}
        assert differ.diff(a, a) == {}
    
    @given(st.text())
    def test_reflexivity_string(self, s):
        """String reflexivity."""
        differ = Diff()
        a = {'s': s}
        assert differ.diff(a, a) == {}
    
    @given(st.lists(st.integers(), max_size=20))
    def test_reflexivity_list(self, lst):
        """List reflexivity."""
        differ = Diff()
        a = {'lst': lst}
        assert differ.diff(a, a) == {}
    
    @given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=10))
    def test_reflexivity_dict(self, d):
        """Dict reflexivity."""
        differ = Diff()
        a = {'d': d}
        assert differ.diff(a, a) == {}
    
    @given(arrays(dtype=np.float64, shape=array_shapes(max_dims=3, max_side=10)))
    def test_reflexivity_array(self, arr):
        """NumPy array reflexivity."""
        differ = Diff()
        a = {'arr': arr}
        assert differ.diff(a, a) == {}
    
    @given(st.integers(), st.integers())
    def test_symmetry_int(self, x, y):
        """If a != b, then b != a."""
        assume(x != y)
        differ = Diff()
        a = {'x': x}
        b = {'x': y}
        
        diff_ab = differ.diff(a, b)
        diff_ba = differ.diff(b, a)
        
        # Both should report differences
        assert ('x' in diff_ab) == ('x' in diff_ba)
    
    @given(st.lists(st.integers(), min_size=1, max_size=20))
    def test_copy_equality(self, lst):
        """A list and its shallow copy should be equal in value but different in identity."""
        differ = Diff()
        a = {'lst': lst}
        b = {'lst': lst.copy()}
        # Values are equal
        assert differ.diff(a, b) == {}
    
    @given(st.lists(st.integers(), min_size=1, max_size=20))
    def test_modification_creates_difference(self, lst):
        """Modifying a value should create a difference."""
        assume(len(lst) > 0)
        differ = Diff()
        a = {'lst': lst.copy()}
        b = {'lst': lst.copy()}
        b['lst'][0] = b['lst'][0] + 1  # Modify first element
        
        result = differ.diff(a, b)
        assert 'lst' in result
    
    @given(st.integers(), st.integers(), st.integers())
    def test_namespace_variable_independence(self, x, y, z):
        """Different variables should be compared independently."""
        differ = Diff()
        a = {'x': x, 'y': y, 'z': z}
        b = {'x': x, 'y': y, 'z': z}
        assert differ.diff(a, b) == {}
        
        # Change one variable
        b['y'] = y + 1 if isinstance(y, int) else 999
        result = differ.diff(a, b)
        assert 'y' in result
        assert 'x' not in result
        assert 'z' not in result
    
    @given(st.floats(allow_nan=False, allow_infinity=False, 
                     min_value=-1e100, max_value=1e100))
    def test_float_tolerance(self, x):
        """Floats within tolerance should be equal."""
        assume(not math.isnan(x) and not math.isinf(x))
        differ = Diff(rtol=1e-6)
        a = {'x': x}
        b = {'x': x * (1 + 1e-8)}  # Very close
        # Might be equal depending on magnitude
        result = differ.diff(a, b)
        # Should either be equal or have a float mismatch
        if result:
            assert 'x' in result
    
    @settings(deadline=None)
    @given(arrays(dtype=np.float64, shape=array_shapes(max_dims=2, max_side=5)))
    def test_array_with_nan_positions(self, arr):
        """Arrays with NaN in same positions should be equal."""
        # Replace some values with NaN
        arr_a = arr.copy()
        arr_b = arr.copy()
        
        if arr.size > 0:
            flat_a = arr_a.ravel()
            flat_b = arr_b.ravel()
            # Set first element to NaN in both
            flat_a[0] = np.nan
            flat_b[0] = np.nan
        
        differ = Diff()
        a = {'arr': arr_a}
        b = {'arr': arr_b}
        assert differ.diff(a, b) == {}
    
    @given(st.lists(st.integers(), min_size=2, max_size=10))
    def test_pointer_structure_preserved(self, lst):
        """Pointer structure must be isomorphic."""
        differ = Diff()
        # Create reference in a
        a = {'x': lst, 'y': lst}
        
        # Create matching reference in b
        lst_copy = lst.copy()
        b = {'x': lst_copy, 'y': lst_copy}
        
        assert differ.diff(a, b) == {}
        
        # Break pointer structure in b
        b['y'] = lst_copy.copy()
        result = differ.diff(a, b)
        assert 'y' in result
        assert 'Pointer structure mismatch' in result['y']
    
    @given(st.sets(st.integers(), max_size=10))
    def test_set_reflexivity(self, s):
        """Sets should equal themselves."""
        differ = Diff()
        a = {'s': s}
        assert differ.diff(a, a) == {}
    
    @given(st.sets(st.integers(), min_size=1, max_size=10))
    def test_set_copy_equality(self, s):
        """A set and its copy should be equal."""
        differ = Diff()
        a = {'s': s}
        b = {'s': s.copy()}
        assert differ.diff(a, b) == {}


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
    # Run pytest with verbose output
    import pytest
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))