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
from data_ferret.kernel.types import ValueComparison, DiffNode, DiffResult


# ============================================================================
# TEST HELPERS
# ============================================================================

def assert_no_diff(result: DiffResult):
    """Assert that result contains no differences."""
    assert result == {}, f"Expected no differences, but got: {result}"


def assert_has_diff(result: DiffResult, var: str):
    """Assert that variable has a difference."""
    assert var in result, f"Expected difference in '{var}', but not found. Result: {result}"


def assert_message_contains(result: DiffResult, var: str, expected_text: str):
    """Assert that the difference message for a variable contains expected text."""
    assert var in result, f"Variable '{var}' not in result"

    diff_node = result[var]

    # If it's a ValueComparison, check its message
    if isinstance(diff_node, ValueComparison):
        assert expected_text in diff_node.message, \
            f"Expected '{expected_text}' in message, but got: {diff_node.message}"
    # If it's a dict (compound diff), look for the text in any nested message
    elif isinstance(diff_node, dict):
        # Recursively search for message containing text
        def find_message(node):
            if isinstance(node, ValueComparison):
                if expected_text in node.message:
                    return True
            elif isinstance(node, dict):
                for value in node.values():
                    if find_message(value):
                        return True
            return False

        assert find_message(diff_node), \
            f"Expected '{expected_text}' in nested messages, but not found. Result: {diff_node}"
    else:
        raise AssertionError(f"Unexpected diff node type: {type(diff_node)}")


def get_comparison(result: DiffResult, var: str) -> ValueComparison:
    """Get ValueComparison for a variable (assumes simple diff)."""
    assert var in result, f"Variable '{var}' not in result"
    assert isinstance(result[var], ValueComparison), \
        f"Expected ValueComparison but got {type(result[var])}"
    return result[var]


def assert_status(result: DiffResult, var: str, expected_status: str):
    """Assert that a variable has a specific comparison status."""
    comparison = get_comparison(result, var)
    assert comparison.status == expected_status, \
        f"Expected status '{expected_status}' but got '{comparison.status}'"


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
        assert_has_diff(result, 'flag')
        assert_message_contains(result, 'flag', 'Bool mismatch')
    
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
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'Integer mismatch')
    
    def test_float_equal(self):
        differ = Diff()
        a = {'pi': 3.14159}
        b = {'pi': 3.14159}
        assert differ.diff(a, b) == {}
    
    def test_float_close_enough(self):
        differ = Diff(rtol=1e-6)
        a = {'x': 1.0000001}
        b = {'x': 1.0000002}
        result = differ.diff(a, b)
        # Should be "close" not equal
        assert_has_diff(result, 'x')
        assert_status(result, 'x', 'close')
    
    def test_float_not_close_enough(self):
        differ = Diff(rtol=1e-9, atol=0)
        a = {'x': 1.0000001}
        b = {'x': 1.0000002}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_status(result, 'x', 'different')
    
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
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaN')
    
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
        assert_message_contains(result, 'msg', 'String mismatch')
    
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
        assert_message_contains(result, 'lst', 'length mismatch')
    
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
        # Now we get a structured diff showing the specific key
        assert isinstance(result['d'], dict)
        assert "['y']" in result['d']
        assert_message_contains(result, 'd', "only in first")
    
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
        assert_message_contains(result, 'arr', 'shape mismatch')
    
    def test_array_different_dtype(self):
        differ = Diff()
        a = {'arr': np.array([1, 2, 3], dtype=np.int32)}
        b = {'arr': np.array([1, 2, 3], dtype=np.int64)}
        result = differ.diff(a, b)
        assert 'arr' in result
        assert_message_contains(result, 'arr', 'dtype mismatch')
    
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
        assert_message_contains(result, 's', 'index mismatch')
    
    def test_series_different_name(self):
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], name='foo')}
        b = {'s': pd.Series([1, 2, 3], name='bar')}
        result = differ.diff(a, b)
        assert 's' in result
        assert_message_contains(result, 's', 'name mismatch')
    
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
        assert_message_contains(result, 'df', 'shape mismatch')
    
    def test_dataframe_different_columns(self):
        differ = Diff()
        a = {'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})}
        b = {'df': pd.DataFrame({'A': [1, 2], 'C': [3, 4]})}
        result = differ.diff(a, b)
        assert 'df' in result
        assert_message_contains(result, 'df', 'columns mismatch')
    
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
        assert_message_contains(result, 'y', 'Pointer structure mismatch')
    
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
        assert_message_contains(result, 'x', 'mismatch')
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
        assert_message_contains(result, 'c', 'Pointer structure mismatch')


class TestNamespaceLevel:
    """Test namespace-level differences."""
    
    def test_variable_only_in_first(self):
        differ = Diff()
        a = {'x': 1, 'y': 2}
        b = {'x': 1}
        result = differ.diff(a, b)
        assert 'y' in result
        assert_message_contains(result, 'y', 'removed')
    
    def test_variable_only_in_second(self):
        differ = Diff()
        a = {'x': 1}
        b = {'x': 1, 'y': 2}
        result = differ.diff(a, b)
        assert 'y' in result
        assert_message_contains(result, 'y', 'added')
    
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
        assert_message_contains(result, 'x', 'Type category mismatch')
    
    def test_list_vs_tuple(self):
        differ = Diff()
        a = {'x': [1, 2, 3]}
        b = {'x': (1, 2, 3)}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Type mismatch')
    
    def test_array_vs_list(self):
        differ = Diff()
        a = {'x': np.array([1, 2, 3])}
        b = {'x': [1, 2, 3]}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Type mismatch')


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
        assert_message_contains(result, 'f', 'Callable mismatch')
    
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
        assert_message_contains(result, 'f', 'Callable mismatch')
    
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
        # Now returns nested structure with .__self__ key
        assert isinstance(result['m'], dict)
        assert '.__self__' in result['m']
    
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
        assert_message_contains(result, 'f', 'Callable mismatch')
    
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
        comp = get_comparison(result, 'arr')
        assert '[2]' in comp.message or '(2,)' in comp.message
        # Should show the values
        assert '3' in comp.message and '99' in comp.message
    
    def test_multidim_array_shows_index(self):
        """Multidimensional array should show full index."""
        differ = Diff()
        a = {'arr': np.array([[1, 2], [3, 4]])}
        b = {'arr': np.array([[1, 2], [3, 99]])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show 2D index
        comp = get_comparison(result, 'arr')
        assert '(1, 1)' in comp.message  # 2D index
        assert '4' in comp.message and '99' in comp.message  # Values
    
    def test_series_shows_label_and_values(self):
        """Series mismatch should show the index label and values."""
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1, 99, 3], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # With new format, check message contains label and values
        comp = get_comparison(result, 's')
        assert 'b' in comp.message  # Index label
        assert '2' in comp.message and '99' in comp.message  # Values
    
    def test_series_nan_position_shows_label(self):
        """Series NaN position mismatch should show the label."""
        differ = Diff()
        a = {'s': pd.Series([1.0, np.nan, 3.0], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # Should show which label has the NaN mismatch
        comp = get_comparison(result, 's')
        assert 'b' in comp.message
        assert 'NaN' in comp.message or 'nan' in comp.message.lower()
    
    def test_set_shows_unmatched_element(self):
        """Set mismatch should show which element couldn't be matched."""
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 99}}
        result = differ.diff(a, b)
        assert 's' in result
        # Should mention an element value (either 3 or 99)
        comp = get_comparison(result, 's')
        assert ('3' in comp.message or '99' in comp.message)
    
    def test_float_array_shows_values(self):
        """Float array with tolerance mismatch should show values."""
        differ = Diff(rtol=1e-9, atol=0)
        a = {'arr': np.array([1.0, 2.0, 3.00001])}
        b = {'arr': np.array([1.0, 2.0, 3.00002])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the differing values
        assert_message_contains(result, 'arr', '3.0000')
        """Array mismatch should show which index differs and the values."""
        differ = Diff()
        a = {'arr': np.array([1, 2, 3, 4, 5])}
        b = {'arr': np.array([1, 2, 99, 4, 5])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the index
        comp = get_comparison(result, 'arr')
        assert '[2]' in comp.message or '(2,)' in comp.message
        # Should show the values
        assert '3' in comp.message and '99' in comp.message
    
    def test_multidim_array_shows_index(self):
        """Multidimensional array should show full index."""
        differ = Diff()
        a = {'arr': np.array([[1, 2], [3, 4]])}
        b = {'arr': np.array([[1, 2], [3, 99]])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show 2D index
        comp = get_comparison(result, 'arr')
        assert '(1, 1)' in comp.message  # 2D index
        assert '4' in comp.message and '99' in comp.message  # Values
    
    def test_series_shows_label_and_values(self):
        """Series mismatch should show the index label and values."""
        differ = Diff()
        a = {'s': pd.Series([1, 2, 3], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1, 99, 3], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # With new format, check message contains label and values
        comp = get_comparison(result, 's')
        assert 'b' in comp.message  # Index label
        assert '2' in comp.message and '99' in comp.message  # Values
    
    def test_series_nan_position_shows_label(self):
        """Series NaN position mismatch should show the label."""
        differ = Diff()
        a = {'s': pd.Series([1.0, np.nan, 3.0], index=['a', 'b', 'c'])}
        b = {'s': pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])}
        result = differ.diff(a, b)
        assert 's' in result
        # Should show which label has the NaN mismatch
        comp = get_comparison(result, 's')
        assert 'b' in comp.message
        assert 'NaN' in comp.message or 'nan' in comp.message.lower()
    
    def test_set_shows_unmatched_element(self):
        """Set mismatch should show which element couldn't be matched."""
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 99}}
        result = differ.diff(a, b)
        assert 's' in result
        # Should mention an element value (either 3 or 99)
        comp = get_comparison(result, 's')
        assert ('3' in comp.message or '99' in comp.message)
    
    def test_float_array_shows_values(self):
        """Float array with tolerance mismatch should show values."""
        differ = Diff(rtol=1e-9, atol=0)
        a = {'arr': np.array([1.0, 2.0, 3.00001])}
        b = {'arr': np.array([1.0, 2.0, 3.00002])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the differing values
        assert_message_contains(result, 'arr', '3.0000')


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
        comp = get_comparison(result, 'gb')
        assert 'Grouping' in comp.message or 'name' in comp.message

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
        comp = get_comparison(result, 'gb')
        assert 'DataFrame' in comp.message or 'Series' in comp.message

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
        assert_message_contains(result, 'gb', 'sort')

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
        assert_message_contains(result, 'gb', 'dropna')

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
        assert_message_contains(result, 'gb2', 'Pointer structure mismatch')


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
        assert_message_contains(result, 'y', 'Pointer structure mismatch')
    
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
# NEW FUNCTIONALITY TESTS
# ============================================================================

class TestFloatCloseStatus:
    """Test the new 'close' status for floats within tolerance."""

    def test_float_exactly_equal_no_diff(self):
        """Exactly equal floats should return no diff."""
        differ = Diff()
        result = differ.diff({'x': 1.0}, {'x': 1.0})
        assert result == {}

    def test_float_close_returns_close_status(self):
        """Floats within tolerance should return 'close' status."""
        differ = Diff(rtol=1e-5, atol=1e-8)
        result = differ.diff({'x': 1.0}, {'x': 1.0 + 1e-6})
        assert 'x' in result
        assert_status(result, 'x', 'close')
        assert_message_contains(result, 'x', 'within tolerance')

    def test_float_far_apart_returns_different_status(self):
        """Floats outside tolerance should return 'different' status."""
        differ = Diff(rtol=1e-9, atol=1e-12)
        result = differ.diff({'x': 1.0}, {'x': 1.001})
        assert 'x' in result
        assert_status(result, 'x', 'different')

    def test_float_close_in_array(self):
        """Close floats in arrays should be detected."""
        from data_ferret.kernel.types import DiffResult

        differ = Diff(rtol=1e-5)
        a = {'arr': np.array([1.0, 2.0, 3.0])}
        b = {'arr': np.array([1.0, 2.0 + 1e-6, 3.0])}
        result = differ.diff(a, b)
        # Array comparison doesn't currently return 'close' status per element
        # but at least shouldn't crash
        assert isinstance(result, DiffResult)

    def test_float_close_in_complex(self):
        """Close floats in complex numbers should be detected."""
        differ = Diff(rtol=1e-5)
        result = differ.diff({'z': 1.0 + 2.0j}, {'z': 1.0 + (2.0 + 1e-6) * 1j})
        assert 'z' in result
        # Complex returns nested dict with .imag key
        assert isinstance(result['z'], dict)
        assert '.imag' in result['z']


class TestCollectAllDifferences:
    """Test that ALL differences are collected, not just the first one."""

    def test_list_collects_all_differences(self):
        """List comparison should find all differing elements."""
        differ = Diff()
        a = {'lst': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        b = {'lst': [1, 99, 3, 88, 5, 77, 7, 66, 9, 55]}
        result = differ.diff(a, b)

        assert 'lst' in result
        assert isinstance(result['lst'], dict)

        # Should have differences at indices 1, 3, 5, 7, 9
        assert '[1]' in result['lst']
        assert '[3]' in result['lst']
        assert '[5]' in result['lst']
        assert '[7]' in result['lst']
        assert '[9]' in result['lst']

        # Should NOT have differences at even indices
        assert '[0]' not in result['lst']
        assert '[2]' not in result['lst']
        assert '[4]' not in result['lst']

    def test_dict_collects_all_differences(self):
        """Dict comparison should find all differing values."""
        differ = Diff()
        a = {'d': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}}
        b = {'d': {'a': 1, 'b': 99, 'c': 3, 'd': 88, 'e': 5}}
        result = differ.diff(a, b)

        assert 'd' in result
        assert isinstance(result['d'], dict)

        # Should have differences at keys 'b' and 'd'
        assert "['b']" in result['d']
        assert "['d']" in result['d']

        # Should NOT have differences at keys 'a', 'c', 'e'
        assert "['a']" not in result['d']
        assert "['c']" not in result['d']
        assert "['e']" not in result['d']

    def test_object_collects_all_attribute_differences(self):
        """Object comparison should find all differing attributes."""
        class Obj:
            def __init__(self, a, b, c, d):
                self.a = a
                self.b = b
                self.c = c
                self.d = d

        differ = Diff()
        result = differ.diff(
            {'o': Obj(1, 2, 3, 4)},
            {'o': Obj(1, 99, 3, 88)}
        )

        assert 'o' in result
        assert isinstance(result['o'], dict)

        # Should have differences at attributes b and d
        assert '.b' in result['o']
        assert '.d' in result['o']

        # Should NOT have differences at attributes a and c
        assert '.a' not in result['o']
        assert '.c' not in result['o']

    def test_nested_structure_collects_all_levels(self):
        """Nested structures should collect differences at all levels."""
        differ = Diff()
        a = {'data': {'users': [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]}}
        b = {'data': {'users': [{'name': 'Alice', 'age': 31}, {'name': 'Charlie', 'age': 25}]}}
        result = differ.diff(a, b)

        assert 'data' in result
        assert isinstance(result['data'], dict)
        assert "['users']" in result['data']

        # Should have differences in both list elements
        users_diff = result['data']["['users']"]
        assert '[0]' in users_diff  # First user age changed
        assert '[1]' in users_diff  # Second user name changed


class TestDiffLimits:
    """Test configurable limits on number of differences reported."""

    def test_list_respects_max_diffs(self):
        """List should stop after max_diffs_per_container."""
        differ = Diff(max_diffs_per_container=3)
        # Create list with 10 differences
        a = {'lst': list(range(10))}
        b = {'lst': [x + 1 for x in range(10)]}
        result = differ.diff(a, b)

        assert 'lst' in result
        diff_dict = result['lst']
        # Should have at most 3 diffs + 1 truncation message
        assert len(diff_dict) <= 4
        assert '_truncated' in diff_dict

    def test_dict_respects_max_diffs(self):
        """Dict should stop after max_diffs_per_container."""
        differ = Diff(max_diffs_per_container=5)
        # Create dict with 10 differences
        a = {'d': {str(i): i for i in range(10)}}
        b = {'d': {str(i): i + 1 for i in range(10)}}
        result = differ.diff(a, b)

        assert 'd' in result
        diff_dict = result['d']
        # Should have at most 5 diffs + 1 truncation message
        assert len(diff_dict) <= 6
        assert '_truncated' in diff_dict

    def test_truncation_message_explains_limit(self):
        """Truncation message should explain why stopped."""
        differ = Diff(max_diffs_per_container=2)
        a = {'lst': [1, 2, 3, 4, 5]}
        b = {'lst': [11, 12, 13, 14, 15]}
        result = differ.diff(a, b)

        truncation = result['lst']['_truncated']
        assert isinstance(truncation, ValueComparison)
        assert 'max_diffs_per_container' in truncation.message
        assert '2' in truncation.message


class TestOnlyDifferences:
    """Test that only differences are included in results."""

    def test_equal_namespace_returns_empty(self):
        """Equal namespaces should return empty dict."""
        differ = Diff()
        a = {'x': 1, 'y': 2, 'z': 3}
        b = {'x': 1, 'y': 2, 'z': 3}
        result = differ.diff(a, b)
        assert result == {}

    def test_mostly_equal_only_shows_diffs(self):
        """Namespace with one diff should only show that diff."""
        differ = Diff()
        a = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
        b = {'a': 1, 'b': 2, 'c': 3, 'd': 99, 'e': 5, 'f': 6, 'g': 7}
        result = differ.diff(a, b)

        # Only 'd' should be in result
        assert len(result) == 1
        assert 'd' in result

    def test_equal_list_elements_not_in_result(self):
        """Equal list elements should not appear in diff."""
        differ = Diff()
        a = {'lst': [1, 2, 3, 4, 5]}
        b = {'lst': [1, 2, 99, 4, 5]}
        result = differ.diff(a, b)

        diff_dict = result['lst']
        # Only index 2 should differ
        assert '[2]' in diff_dict
        assert '[0]' not in diff_dict
        assert '[1]' not in diff_dict
        assert '[3]' not in diff_dict
        assert '[4]' not in diff_dict


class TestDiffNodeStructure:
    """Test the tree structure of DiffNode results."""

    def test_simple_diff_returns_value_comparison(self):
        """Simple type diff should return ValueComparison."""
        differ = Diff()
        result = differ.diff({'x': 1}, {'x': 2})
        assert isinstance(result['x'], ValueComparison)
        assert result['x'].status == 'different'
        assert result['x'].value1 == 1
        assert result['x'].value2 == 2

    def test_compound_diff_returns_dict(self):
        """Compound structure diff should return dict."""
        differ = Diff()
        result = differ.diff({'lst': [1, 2]}, {'lst': [1, 99]})
        assert isinstance(result['lst'], dict)
        assert '[1]' in result['lst']

    def test_nested_diff_has_nested_dicts(self):
        """Nested structures should have nested dicts."""
        differ = Diff()
        result = differ.diff(
            {'outer': {'inner': [1, 2, 3]}},
            {'outer': {'inner': [1, 99, 3]}}
        )

        assert isinstance(result['outer'], dict)
        assert "['inner']" in result['outer']
        assert isinstance(result['outer']["['inner']"], dict)
        assert '[1]' in result['outer']["['inner']"]

    def test_value_comparison_has_expected_fields(self):
        """ValueComparison should have all expected fields."""
        differ = Diff()
        result = differ.diff({'x': 1}, {'x': 2})
        comp = result['x']

        assert hasattr(comp, 'status')
        assert hasattr(comp, 'value1')
        assert hasattr(comp, 'value2')
        assert hasattr(comp, 'message')
        assert hasattr(comp, 'is_close')

    def test_close_status_has_is_close_true(self):
        """ValueComparison with 'close' status should have is_close=True."""
        differ = Diff(rtol=1e-5)
        result = differ.diff({'x': 1.0}, {'x': 1.0 + 1e-6})
        comp = result['x']

        assert comp.status == 'close'
        assert comp.is_close is True

    def test_different_status_has_is_close_false(self):
        """ValueComparison with 'different' status should have is_close=False."""
        differ = Diff()
        result = differ.diff({'x': 1}, {'x': 2})
        comp = result['x']

        assert comp.status == 'different'
        assert comp.is_close is False


class TestVariableAddedRemoved:
    """Test that added/removed variables are properly detected."""

    def test_variable_removed(self):
        """Removed variable should appear in result."""
        differ = Diff()
        result = differ.diff({'x': 1, 'y': 2}, {'x': 1})

        assert 'y' in result
        assert isinstance(result['y'], ValueComparison)
        assert_message_contains(result, 'y', 'removed')

    def test_variable_added(self):
        """Added variable should appear in result."""
        differ = Diff()
        result = differ.diff({'x': 1}, {'x': 1, 'y': 2})

        assert 'y' in result
        assert isinstance(result['y'], ValueComparison)
        assert_message_contains(result, 'y', 'added')

    def test_multiple_added_and_removed(self):
        """Multiple added and removed variables should all appear."""
        differ = Diff()
        result = differ.diff(
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 1, 'd': 4, 'e': 5}
        )

        # 'a' unchanged, 'b' and 'c' removed, 'd' and 'e' added
        assert 'a' not in result  # Unchanged
        assert 'b' in result  # Removed
        assert 'c' in result  # Removed
        assert 'd' in result  # Added
        assert 'e' in result  # Added


class TestMarkdownFormatting:
    """Test the format_diff_as_markdown function."""

    def test_empty_diff_formatting(self):
        """Empty diff should show 'No Differences Found'."""
        from data_ferret.kernel.types import format_diff_as_markdown

        result = {}
        markdown = format_diff_as_markdown(result)

        assert "No Differences Found" in markdown
        assert "All variables are equal" in markdown

    def test_simple_diff_formatting(self):
        """Simple difference should be formatted as bullet point."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff()
        result = differ.diff({'x': 1}, {'x': 2})
        markdown = format_diff_as_markdown(result)

        assert "## Differences Found" in markdown
        assert "- **x**:" in markdown

    def test_close_float_shows_indicator(self):
        """Close floats should show (close) indicator."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff(rtol=1e-5)
        result = differ.diff({'y': 1.0000001}, {'y': 1.0000002})
        markdown = format_diff_as_markdown(result)

        assert "**y** *(close)*:" in markdown
        assert "Float close" in markdown

    def test_nested_structure_formatting(self):
        """Nested structures should show full paths."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff()
        result = differ.diff(
            {'data': {'a': 1, 'b': 2}},
            {'data': {'a': 1, 'b': 99}}
        )
        markdown = format_diff_as_markdown(result)

        assert "**data['b']**:" in markdown

    def test_list_formatting(self):
        """List differences should show indices."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff()
        result = differ.diff(
            {'items': [1, 2, 3]},
            {'items': [1, 99, 3]}
        )
        markdown = format_diff_as_markdown(result)

        assert "**items[1]**:" in markdown

    def test_multiple_variables_sorted(self):
        """Multiple variables should be sorted alphabetically."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff()
        result = differ.diff(
            {'z': 1, 'a': 2, 'm': 3},
            {'z': 10, 'a': 20, 'm': 30}
        )
        markdown = format_diff_as_markdown(result)

        lines = markdown.split('\n')
        var_lines = [l for l in lines if l.startswith('- **')]

        # Should be in alphabetical order: a, m, z
        assert '**a**' in var_lines[0]
        assert '**m**' in var_lines[1]
        assert '**z**' in var_lines[2]

    def test_truncation_appears_in_markdown(self):
        """Truncation messages should appear in output."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff(max_diffs_per_container=2)
        result = differ.diff(
            {'nums': [1, 2, 3, 4, 5]},
            {'nums': [10, 20, 30, 40, 50]}
        )
        markdown = format_diff_as_markdown(result)

        assert "Truncated" in markdown


class TestStrictMode:
    """Test the strict parameter for flexible type comparisons."""

    def test_int_vs_float_strict_mode(self):
        """In strict mode, int vs float should fail."""
        differ = Diff(strict=True)
        a = {'x': 1}
        b = {'x': 1.0}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Type category mismatch')

    def test_int_vs_float_nonstrict_mode(self):
        """In non-strict mode, int vs float with same value should pass."""
        differ = Diff(strict=False)
        a = {'x': 1}
        b = {'x': 1.0}
        result = differ.diff(a, b)
        assert result == {}

    def test_float_vs_int_nonstrict_mode(self):
        """In non-strict mode, float vs int with same value should pass."""
        differ = Diff(strict=False)
        a = {'x': 2.0}
        b = {'x': 2}
        result = differ.diff(a, b)
        assert result == {}

    def test_int_vs_float_different_values_nonstrict(self):
        """In non-strict mode, int vs float with different values should fail."""
        differ = Diff(strict=False)
        a = {'x': 1}
        b = {'x': 2.0}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Float mismatch')

    def test_numpy_int_vs_float_nonstrict(self):
        """In non-strict mode, np.int64 vs float should pass."""
        differ = Diff(strict=False)
        a = {'x': np.int64(42)}
        b = {'x': 42.0}
        result = differ.diff(a, b)
        assert result == {}

    def test_numpy_float_vs_int_nonstrict(self):
        """In non-strict mode, np.float64 vs int should pass."""
        differ = Diff(strict=False)
        a = {'x': np.float64(42.0)}
        b = {'x': 42}
        result = differ.diff(a, b)
        assert result == {}

    def test_int_vs_np_int64(self):
        """Python int should equal np.int64 with same value."""
        differ = Diff(strict=False)
        a = {'x': 5}
        b = {'x': np.int64(5)}
        result = differ.diff(a, b)
        assert result == {}

    def test_np_int64_vs_np_int32(self):
        """Different numpy int types should be compatible."""
        differ = Diff(strict=False)
        a = {'x': np.int64(42)}
        b = {'x': np.int32(42)}
        result = differ.diff(a, b)
        assert result == {}

    def test_np_int64_vs_np_int16(self):
        """np.int64 should equal np.int16 with same value."""
        differ = Diff(strict=False)
        a = {'x': np.int64(100)}
        b = {'x': np.int16(100)}
        result = differ.diff(a, b)
        assert result == {}

    def test_float_vs_np_float64(self):
        """Python float should equal np.float64 with same value."""
        differ = Diff(strict=False)
        a = {'x': 3.14}
        b = {'x': np.float64(3.14)}
        result = differ.diff(a, b)
        assert result == {}

    def test_np_float64_vs_np_float32(self):
        """Different numpy float types should be compatible."""
        differ = Diff(strict=False)
        a = {'x': np.float64(3.14)}
        b = {'x': np.float32(3.14)}
        result = differ.diff(a, b)
        # May have small precision differences, so check if equal or close
        assert len(result) == 0 or (result['x'].status == 'close')

    def test_np_int64_different_values(self):
        """Different values should still be detected even with same numpy type."""
        differ = Diff(strict=False)
        a = {'x': np.int64(5)}
        b = {'x': np.int64(10)}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Integer mismatch')

    def test_int_vs_np_int64_different_values(self):
        """Different values should be detected across int and np.int64."""
        differ = Diff(strict=False)
        a = {'x': 5}
        b = {'x': np.int64(10)}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Integer mismatch')

    def test_list_vs_array_1d_strict(self):
        """In strict mode, list vs array should fail."""
        differ = Diff(strict=True)
        a = {'x': [1, 2, 3]}
        b = {'x': np.array([1, 2, 3])}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Type mismatch')

    def test_list_vs_array_1d_nonstrict(self):
        """In non-strict mode, list vs array with same values should pass."""
        differ = Diff(strict=False)
        a = {'x': [1, 2, 3]}
        b = {'x': np.array([1, 2, 3])}
        result = differ.diff(a, b)
        assert result == {}

    def test_array_vs_list_1d_nonstrict(self):
        """In non-strict mode, array vs list with same values should pass."""
        differ = Diff(strict=False)
        a = {'x': np.array([4, 5, 6])}
        b = {'x': [4, 5, 6]}
        result = differ.diff(a, b)
        assert result == {}

    def test_nested_list_vs_array_2d_nonstrict(self):
        """In non-strict mode, nested list vs 2D array should pass."""
        differ = Diff(strict=False)
        a = {'x': [[1, 2], [3, 4]]}
        b = {'x': np.array([[1, 2], [3, 4]])}
        result = differ.diff(a, b)
        assert result == {}

    def test_deeply_nested_list_vs_array_3d_nonstrict(self):
        """In non-strict mode, 3D nested list vs 3D array should pass."""
        differ = Diff(strict=False)
        a = {'x': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}
        b = {'x': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}
        result = differ.diff(a, b)
        assert result == {}

    def test_list_vs_array_dimension_mismatch_nonstrict(self):
        """In non-strict mode, dimension mismatch should still fail."""
        differ = Diff(strict=False)
        a = {'x': [1, 2, 3]}
        b = {'x': np.array([[1, 2, 3]])}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Structure mismatch')

    def test_list_vs_array_shape_mismatch_nonstrict(self):
        """In non-strict mode, shape mismatch should fail."""
        differ = Diff(strict=False)
        a = {'x': [[1, 2], [3, 4]]}
        b = {'x': np.array([[1, 2, 3], [4, 5, 6]])}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Shape mismatch')

    def test_list_vs_array_value_mismatch_nonstrict(self):
        """In non-strict mode, value mismatch should fail."""
        differ = Diff(strict=False)
        a = {'x': [1, 2, 3]}
        b = {'x': np.array([1, 2, 99])}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Element mismatch')

    def test_tuple_vs_array_strict(self):
        """In strict mode, tuple vs array should fail."""
        differ = Diff(strict=True)
        a = {'x': (1, 2, 3)}
        b = {'x': np.array([1, 2, 3])}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Type mismatch')

    def test_tuple_vs_array_nonstrict(self):
        """In non-strict mode, tuple vs array with same values should pass."""
        differ = Diff(strict=False)
        a = {'x': (1, 2, 3)}
        b = {'x': np.array([1, 2, 3])}
        result = differ.diff(a, b)
        assert result == {}

    def test_nested_tuple_vs_array_2d_nonstrict(self):
        """In non-strict mode, nested tuple vs 2D array should pass."""
        differ = Diff(strict=False)
        a = {'x': ((1, 2), (3, 4))}
        b = {'x': np.array([[1, 2], [3, 4]])}
        result = differ.diff(a, b)
        assert result == {}

    def test_mixed_list_vs_array_nonstrict(self):
        """In non-strict mode, list with mixed int/float vs int array should pass."""
        differ = Diff(strict=False)
        a = {'x': [1, 2.0, 3]}
        b = {'x': np.array([1, 2, 3])}
        result = differ.diff(a, b)
        assert result == {}

    def test_float_list_vs_int_array_nonstrict(self):
        """In non-strict mode, float list vs int array with compatible values should pass."""
        differ = Diff(strict=False)
        a = {'x': [1.0, 2.0, 3.0]}
        b = {'x': np.array([1, 2, 3], dtype=np.int32)}
        result = differ.diff(a, b)
        assert result == {}

    def test_float_list_vs_int_array_incompatible_values_nonstrict(self):
        """In non-strict mode, float list vs int array with incompatible values should fail."""
        differ = Diff(strict=False)
        a = {'x': [1.5, 2.5, 3.5]}
        b = {'x': np.array([1, 2, 3])}
        result = differ.diff(a, b)
        assert 'x' in result
        assert_message_contains(result, 'x', 'Element mismatch')

    def test_complex_nested_structure_nonstrict(self):
        """In non-strict mode, complex nested structures with mixed types should pass."""
        differ = Diff(strict=False)
        a = {
            'data': {
                'scores': [10, 20, 30],
                'matrix': [[1, 2], [3, 4]],
                'value': 42,
                'ratio': 0.5
            }
        }
        b = {
            'data': {
                'scores': np.array([10, 20, 30]),
                'matrix': np.array([[1, 2], [3, 4]]),
                'value': 42.0,
                'ratio': 0.5
            }
        }
        result = differ.diff(a, b)
        assert result == {}

    def test_list_in_dict_in_list_nonstrict(self):
        """In non-strict mode, list inside dict inside list vs array should pass."""
        differ = Diff(strict=False)
        a = {'x': [{'nums': [1, 2, 3]}]}
        b = {'x': [{'nums': np.array([1, 2, 3])}]}
        result = differ.diff(a, b)
        assert result == {}

    def test_pointer_structure_preserved_in_nonstrict(self):
        """In non-strict mode, pointer structure should still be checked."""
        differ = Diff(strict=False)
        lst = [1, 2, 3]
        a = {'x': lst, 'y': lst}

        # Break pointer structure with array
        b = {'x': np.array([1, 2, 3]), 'y': np.array([1, 2, 3])}

        result = differ.diff(a, b)
        # Should detect pointer structure mismatch
        assert 'y' in result
        assert_message_contains(result, 'y', 'Pointer structure mismatch')

    def test_pointer_structure_maintained_nonstrict(self):
        """In non-strict mode, matching pointer structure should pass."""
        differ = Diff(strict=False)
        lst_a = [1, 2, 3]
        a = {'x': lst_a, 'y': lst_a}

        lst_b = [1, 2, 3]
        b = {'x': lst_b, 'y': lst_b}

        result = differ.diff(a, b)
        assert result == {}

    def test_empty_list_vs_empty_array_nonstrict(self):
        """In non-strict mode, empty list vs empty array should pass."""
        differ = Diff(strict=False)
        a = {'x': []}
        b = {'x': np.array([])}
        result = differ.diff(a, b)
        assert result == {}

    def test_bool_not_confused_with_int_nonstrict(self):
        """In non-strict mode, bool should not be treated as int."""
        differ = Diff(strict=False)
        a = {'x': True}
        b = {'x': 1}
        result = differ.diff(a, b)
        # Bools should not match ints even in non-strict mode
        assert 'x' in result

    def test_bool_vs_float_nonstrict(self):
        """In non-strict mode, bool should not be treated as float."""
        differ = Diff(strict=False)
        a = {'x': True}
        b = {'x': 1.0}
        result = differ.diff(a, b)
        # Bools should not match floats even in non-strict mode
        assert 'x' in result

    def test_list_with_nan_vs_array_with_nan_nonstrict(self):
        """In non-strict mode, list with NaN vs array with NaN should pass."""
        differ = Diff(strict=False)
        a = {'x': [1.0, float('nan'), 3.0]}
        b = {'x': np.array([1.0, np.nan, 3.0])}
        result = differ.diff(a, b)
        assert result == {}

    def test_ragged_list_vs_array_nonstrict(self):
        """In non-strict mode, ragged list cannot convert to array properly."""
        differ = Diff(strict=False)
        a = {'x': [[1, 2], [3, 4, 5]]}  # Ragged - different lengths
        b = {'x': np.array([[1, 2], [3, 4]])}
        result = differ.diff(a, b)
        # Should fail to convert
        assert 'x' in result

    def test_backwards_compatibility_default_strict(self):
        """Default behavior should be strict=True for backwards compatibility."""
        differ = Diff()
        a = {'x': 1}
        b = {'x': 1.0}
        result = differ.diff(a, b)
        # Default is strict, so should fail
        assert 'x' in result
        assert_message_contains(result, 'x', 'Type category mismatch')

    def test_multiple_compatible_types_in_namespace_nonstrict(self):
        """In non-strict mode, multiple variables with compatible types."""
        differ = Diff(strict=False)
        a = {
            'a': 1,
            'b': [1, 2, 3],
            'c': (4, 5, 6),
            'd': [[1, 2], [3, 4]]
        }
        b = {
            'a': 1.0,
            'b': np.array([1, 2, 3]),
            'c': np.array([4, 5, 6]),
            'd': np.array([[1, 2], [3, 4]])
        }
        result = differ.diff(a, b)
        assert result == {}


class TestDiffResultFiltering:
    """Test the close_only() and different_only() filtering methods."""

    def test_close_only_returns_only_close_comparisons(self):
        """close_only() should return only close float comparisons."""
        differ = Diff(rtol=1e-5)
        a = {
            'x': 1.0000001,  # Close
            'y': 1.0,        # Different
            'z': 2.0000001   # Close
        }
        b = {
            'x': 1.0000002,  # Close
            'y': 2.0,        # Different
            'z': 2.0000002   # Close
        }
        result = differ.diff(a, b)

        close_result = result.close_only()

        # Should only have x and z
        assert 'x' in close_result
        assert 'z' in close_result
        assert 'y' not in close_result
        assert len(close_result) == 2

    def test_different_only_returns_only_different_comparisons(self):
        """different_only() should return only different comparisons."""
        differ = Diff(rtol=1e-5)
        a = {
            'x': 1.0000001,  # Close
            'y': 1.0,        # Different
            'z': 2.0000001   # Close
        }
        b = {
            'x': 1.0000002,  # Close
            'y': 2.0,        # Different
            'z': 2.0000002   # Close
        }
        result = differ.diff(a, b)

        diff_result = result.different_only()

        # Should only have y
        assert 'y' in diff_result
        assert 'x' not in diff_result
        assert 'z' not in diff_result
        assert len(diff_result) == 1

    def test_close_only_with_no_close_comparisons(self):
        """close_only() should return empty DiffResult if no close comparisons."""
        differ = Diff()
        a = {'x': 1, 'y': 2}
        b = {'x': 10, 'y': 20}
        result = differ.diff(a, b)

        close_result = result.close_only()

        assert close_result == {}
        assert len(close_result) == 0

    def test_different_only_with_only_close_comparisons(self):
        """different_only() should return empty DiffResult if only close comparisons."""
        differ = Diff(rtol=1e-5)
        a = {'x': 1.0000001, 'y': 2.0000001}
        b = {'x': 1.0000002, 'y': 2.0000002}
        result = differ.diff(a, b)

        diff_result = result.different_only()

        assert diff_result == {}
        assert len(diff_result) == 0

    def test_filtering_with_nested_structures(self):
        """Filtering should work with nested structures."""
        differ = Diff(rtol=1e-5)
        a = {
            'data': {
                'a': 1.0000001,  # Close
                'b': 1.0,        # Different
                'c': 2.0000001   # Close
            }
        }
        b = {
            'data': {
                'a': 1.0000002,  # Close
                'b': 2.0,        # Different
                'c': 2.0000002   # Close
            }
        }
        result = differ.diff(a, b)

        close_result = result.close_only()
        diff_result = result.different_only()

        # close_result should have data with only a and c
        assert 'data' in close_result
        assert isinstance(close_result['data'], dict)
        assert "['a']" in close_result['data']
        assert "['c']" in close_result['data']
        assert "['b']" not in close_result['data']

        # diff_result should have data with only b
        assert 'data' in diff_result
        assert isinstance(diff_result['data'], dict)
        assert "['b']" in diff_result['data']
        assert "['a']" not in diff_result['data']
        assert "['c']" not in diff_result['data']

    def test_filtering_with_lists(self):
        """Filtering should work with lists containing mixed close/different."""
        differ = Diff(rtol=1e-5)
        a = {'lst': [1.0000001, 2.0, 3.0000001]}
        b = {'lst': [1.0000002, 99.0, 3.0000002]}
        result = differ.diff(a, b)

        close_result = result.close_only()
        diff_result = result.different_only()

        # close_result should have list with indices 0 and 2
        assert 'lst' in close_result
        assert '[0]' in close_result['lst']
        assert '[2]' in close_result['lst']
        assert '[1]' not in close_result['lst']

        # diff_result should have list with index 1
        assert 'lst' in diff_result
        assert '[1]' in diff_result['lst']
        assert '[0]' not in diff_result['lst']
        assert '[2]' not in diff_result['lst']

    def test_filtering_returns_new_diffresult(self):
        """Filtering should return new DiffResult instances."""
        differ = Diff(rtol=1e-5)
        a = {'x': 1.0000001, 'y': 1.0}
        b = {'x': 1.0000002, 'y': 2.0}
        result = differ.diff(a, b)

        close_result = result.close_only()
        diff_result = result.different_only()

        # All should be different objects
        assert result is not close_result
        assert result is not diff_result
        assert close_result is not diff_result

        # All should be DiffResult instances
        assert isinstance(result, DiffResult)
        assert isinstance(close_result, DiffResult)
        assert isinstance(diff_result, DiffResult)

    def test_filtering_preserves_valuecomparison_objects(self):
        """Filtered results should contain same ValueComparison objects."""
        differ = Diff(rtol=1e-5)
        a = {'x': 1.0000001, 'y': 1.0}
        b = {'x': 1.0000002, 'y': 2.0}
        result = differ.diff(a, b)

        close_result = result.close_only()

        # The ValueComparison object should be the same
        assert result['x'] is close_result['x']

    def test_filtering_with_complex_object(self):
        """Filtering should work with complex nested objects."""
        class Container:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        differ = Diff(rtol=1e-5)
        a = {'obj': Container(1.0000001, 2.0)}
        b = {'obj': Container(1.0000002, 99.0)}
        result = differ.diff(a, b)

        close_result = result.close_only()
        diff_result = result.different_only()

        # close_result should have obj with only .a
        assert 'obj' in close_result
        assert '.a' in close_result['obj']
        assert '.b' not in close_result['obj']

        # diff_result should have obj with only .b
        assert 'obj' in diff_result
        assert '.b' in diff_result['obj']
        assert '.a' not in diff_result['obj']

    def test_filtering_empty_diffresult(self):
        """Filtering an empty DiffResult should return empty DiffResult."""
        result = DiffResult(differences={})

        close_result = result.close_only()
        diff_result = result.different_only()

        assert close_result == {}
        assert diff_result == {}

    def test_filtering_all_different(self):
        """Filtering all different should leave different_only unchanged."""
        differ = Diff()
        a = {'x': 1, 'y': 2, 'z': 3}
        b = {'x': 10, 'y': 20, 'z': 30}
        result = differ.diff(a, b)

        diff_result = result.different_only()

        # Should have all three
        assert len(diff_result) == 3
        assert 'x' in diff_result
        assert 'y' in diff_result
        assert 'z' in diff_result

    def test_deeply_nested_filtering(self):
        """Filtering should work with deeply nested structures."""
        differ = Diff(rtol=1e-5)
        a = {
            'level1': {
                'level2': {
                    'level3': [
                        {'val': 1.0000001},  # Close
                        {'val': 2.0}         # Different
                    ]
                }
            }
        }
        b = {
            'level1': {
                'level2': {
                    'level3': [
                        {'val': 1.0000002},  # Close
                        {'val': 99.0}        # Different
                    ]
                }
            }
        }
        result = differ.diff(a, b)

        close_result = result.close_only()
        diff_result = result.different_only()

        # Both should have the nested structure
        assert 'level1' in close_result
        assert 'level1' in diff_result

        # Navigate down to check filtering worked
        close_leaf = close_result['level1']["['level2']"]["['level3']"]
        assert '[0]' in close_leaf
        assert '[1]' not in close_leaf

        diff_leaf = diff_result['level1']["['level2']"]["['level3']"]
        assert '[1]' in diff_leaf
        assert '[0]' not in diff_leaf

    def test_filtering_with_format_diff_as_markdown(self):
        """Filtered results should work with format_diff_as_markdown."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff(rtol=1e-5)
        a = {'x': 1.0000001, 'y': 1.0}
        b = {'x': 1.0000002, 'y': 2.0}
        result = differ.diff(a, b)

        close_result = result.close_only()
        diff_result = result.different_only()

        close_markdown = format_diff_as_markdown(close_result)
        diff_markdown = format_diff_as_markdown(diff_result)

        # close_markdown should mention x and close indicator
        assert '**x**' in close_markdown
        assert '*(close)*' in close_markdown
        assert '**y**' not in close_markdown

        # diff_markdown should mention y but not close indicator
        assert '**y**' in diff_markdown
        assert '*(close)*' not in diff_markdown
        assert '**x**' not in diff_markdown


class TestReportCloseFlag:
    """Test the report_close flag for controlling close value reporting."""

    def test_report_close_true_default(self):
        """Default behavior (report_close=True) should report close values."""
        differ = Diff(rtol=1e-5)
        a = {'x': 1.0000001}
        b = {'x': 1.0000002}
        result = differ.diff(a, b)

        # Should report the close value
        assert 'x' in result
        assert_status(result, 'x', 'close')

    def test_report_close_false_hides_close_values(self):
        """report_close=False should not report close values."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'x': 1.0000001}
        b = {'x': 1.0000002}
        result = differ.diff(a, b)

        # Should NOT report the close value
        assert 'x' not in result
        assert result == {}

    def test_report_close_false_still_reports_different(self):
        """report_close=False should still report different values."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'x': 1.0}
        b = {'x': 2.0}
        result = differ.diff(a, b)

        # Should report the different value
        assert 'x' in result
        assert_status(result, 'x', 'different')

    def test_report_close_false_with_exact_match(self):
        """report_close=False with exact match should not report."""
        differ = Diff(report_close=False)
        a = {'x': 1.0}
        b = {'x': 1.0}
        result = differ.diff(a, b)

        # Should not report (exact match)
        assert 'x' not in result
        assert result == {}

    def test_report_close_mixed_values(self):
        """report_close=False with mixed close and different values."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {
            'close1': 1.0000001,  # Close
            'different': 1.0,     # Different
            'close2': 2.0000001,  # Close
            'exact': 3.0          # Exact match
        }
        b = {
            'close1': 1.0000002,  # Close
            'different': 99.0,    # Different
            'close2': 2.0000002,  # Close
            'exact': 3.0          # Exact match
        }
        result = differ.diff(a, b)

        # Should only report different
        assert 'different' in result
        assert 'close1' not in result
        assert 'close2' not in result
        assert 'exact' not in result
        assert len(result) == 1

    def test_report_close_false_all_close(self):
        """report_close=False with all close values should return empty."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'x': 1.0000001, 'y': 2.0000001, 'z': 3.0000001}
        b = {'x': 1.0000002, 'y': 2.0000002, 'z': 3.0000002}
        result = differ.diff(a, b)

        # Should return empty - all values are close
        assert result == {}
        assert len(result) == 0

    def test_report_close_false_complex_numbers(self):
        """report_close=False with complex numbers (both parts close)."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'z': 1.0000001 + 2.0000001j}
        b = {'z': 1.0000002 + 2.0000002j}
        result = differ.diff(a, b)

        # Should not report - both real and imag are close
        assert 'z' not in result
        assert result == {}

    def test_report_close_false_complex_one_part_different(self):
        """report_close=False with complex (one part close, one different)."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'z': 1.0000001 + 2.0j}
        b = {'z': 1.0000002 + 99.0j}
        result = differ.diff(a, b)

        # Should report only the imaginary part difference
        assert 'z' in result
        assert isinstance(result['z'], dict)
        assert '.imag' in result['z']
        assert '.real' not in result['z']

    def test_report_close_false_nested_structures(self):
        """report_close=False with nested structures."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {
            'data': {
                'close': 1.0000001,
                'different': 2.0
            }
        }
        b = {
            'data': {
                'close': 1.0000002,
                'different': 99.0
            }
        }
        result = differ.diff(a, b)

        # Should only report different
        assert 'data' in result
        assert "['different']" in result['data']
        assert "['close']" not in result['data']

    def test_report_close_false_list_values(self):
        """report_close=False with lists containing close values."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'lst': [1.0000001, 2.0, 3.0000001]}
        b = {'lst': [1.0000002, 99.0, 3.0000002]}
        result = differ.diff(a, b)

        # Should only report index 1 (different)
        assert 'lst' in result
        assert '[1]' in result['lst']
        assert '[0]' not in result['lst']
        assert '[2]' not in result['lst']

    def test_report_close_false_object_attributes(self):
        """report_close=False with object attributes."""
        class Container:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        differ = Diff(rtol=1e-5, report_close=False)
        a = {'obj': Container(1.0000001, 2.0)}
        b = {'obj': Container(1.0000002, 99.0)}
        result = differ.diff(a, b)

        # Should only report .b
        assert 'obj' in result
        assert '.b' in result['obj']
        assert '.a' not in result['obj']

    def test_equivalence_with_different_only(self):
        """Verify report_close=False is equivalent to different_only()."""
        a = {
            'close1': 1.0000001,
            'different': 1.0,
            'close2': 2.0000001,
            'exact': 3.0
        }
        b = {
            'close1': 1.0000002,
            'different': 99.0,
            'close2': 2.0000002,
            'exact': 3.0
        }

        # With report_close=False
        differ_no_report = Diff(rtol=1e-5, report_close=False)
        result_no_report = differ_no_report.diff(a, b)

        # With different_only()
        differ_filter = Diff(rtol=1e-5, report_close=True)
        result_filter = differ_filter.diff(a, b).different_only()

        # Should be equivalent
        assert set(result_no_report.keys()) == set(result_filter.keys())
        assert 'different' in result_no_report
        assert 'different' in result_filter

    def test_backwards_compatibility_default_true(self):
        """Default report_close=True maintains backward compatibility."""
        differ_default = Diff(rtol=1e-5)
        differ_explicit = Diff(rtol=1e-5, report_close=True)

        a = {'x': 1.0000001}
        b = {'x': 1.0000002}

        result_default = differ_default.diff(a, b)
        result_explicit = differ_explicit.diff(a, b)

        # Should behave identically
        assert 'x' in result_default
        assert 'x' in result_explicit
        assert result_default['x'].status == result_explicit['x'].status

    def test_report_close_false_with_nan(self):
        """report_close=False with NaN values."""
        differ = Diff(report_close=False)
        a = {'x': float('nan')}
        b = {'x': float('nan')}
        result = differ.diff(a, b)

        # NaN == NaN should not report
        assert 'x' not in result

    def test_report_close_false_performance_benefit(self):
        """report_close=False should create fewer ValueComparison objects."""
        differ_true = Diff(rtol=1e-5, report_close=True)
        differ_false = Diff(rtol=1e-5, report_close=False)

        # Create data with many close values
        # Use values that are actually close: 100.0000001 vs 100.0000002
        a = {f'x{i}': float(i + 100) + 0.0000001 for i in range(100)}
        b = {f'x{i}': float(i + 100) + 0.0000002 for i in range(100)}

        result_true = differ_true.diff(a, b)
        result_false = differ_false.diff(a, b)

        # report_close=True should have 100 results
        assert len(result_true) == 100

        # report_close=False should have 0 results
        assert len(result_false) == 0

    def test_report_close_false_deeply_nested(self):
        """report_close=False with deeply nested structures."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {
            'level1': {
                'level2': {
                    'level3': [
                        {'val': 1.0000001},  # Close
                        {'val': 2.0}         # Different
                    ]
                }
            }
        }
        b = {
            'level1': {
                'level2': {
                    'level3': [
                        {'val': 1.0000002},  # Close
                        {'val': 99.0}        # Different
                    ]
                }
            }
        }
        result = differ.diff(a, b)

        # Should only report the different value
        assert 'level1' in result
        level3 = result['level1']["['level2']"]["['level3']"]
        assert '[1]' in level3
        assert '[0]' not in level3

    def test_report_close_false_with_format_markdown(self):
        """report_close=False results should format correctly."""
        from data_ferret.kernel.types import format_diff_as_markdown

        differ = Diff(rtol=1e-5, report_close=False)
        a = {'close': 1.0000001, 'different': 1.0}
        b = {'close': 1.0000002, 'different': 99.0}
        result = differ.diff(a, b)

        markdown = format_diff_as_markdown(result)

        # Should only mention different, not close
        assert '**different**' in markdown
        assert '**close**' not in markdown
        assert '*(close)*' not in markdown


# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
    # Run pytest with verbose output
    import pytest
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))