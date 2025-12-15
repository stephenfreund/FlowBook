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
from data_ferret.kernel.types import ValueComparison, CompoundDiff, DiffNode, DiffResult


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
    # If it's a CompoundDiff or dict, look for the text in any nested message
    elif isinstance(diff_node, (CompoundDiff, dict)):
        # Recursively search for message containing text
        def find_message(node):
            if isinstance(node, ValueComparison):
                if expected_text in node.message:
                    return True
            elif isinstance(node, CompoundDiff):
                for value in node.children.values():
                    if find_message(value):
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


def get_any_comparison(result: DiffResult, var: str) -> ValueComparison:
    """Get any ValueComparison from a variable (handles nested diffs)."""
    assert var in result, f"Variable '{var}' not in result"
    diff_node = result[var]

    if isinstance(diff_node, ValueComparison):
        return diff_node
    elif isinstance(diff_node, CompoundDiff):
        # Return first ValueComparison found in children
        for value in diff_node.children.values():
            if isinstance(value, ValueComparison):
                return value
            elif isinstance(value, CompoundDiff):
                # Recursively search nested CompoundDiff
                for nested_value in value.children.values():
                    if isinstance(nested_value, ValueComparison):
                        return nested_value
        raise AssertionError(f"No ValueComparison found in CompoundDiff for '{var}'")
    elif isinstance(diff_node, dict):
        # Return first non-truncated ValueComparison found
        for key, value in diff_node.items():
            if key != '_truncated' and isinstance(value, ValueComparison):
                return value
        raise AssertionError(f"No ValueComparison found in nested diff for '{var}'")
    else:
        raise AssertionError(f"Unexpected diff node type: {type(diff_node)}")


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
        assert isinstance(result['d'], CompoundDiff)
        assert "['y']" in result['d'].children
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
        # int32 and int64 are compatible, so use int and float for incompatible types
        a = {'arr': np.array([1, 2, 3], dtype=np.int32)}
        b = {'arr': np.array([1, 2, 3], dtype=np.float32)}
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
        # Now reports index mismatch (different number of rows)
        assert_message_contains(result, 'df', 'index mismatch')

    def test_dataframe_different_columns(self):
        differ = Diff()
        a = {'df': pd.DataFrame({'A': [1, 2], 'B': [3, 4]})}
        b = {'df': pd.DataFrame({'A': [1, 2], 'C': [3, 4]})}
        result = differ.diff(a, b)
        assert 'df' in result
        # Now reports individual column differences
        assert_message_contains(result, 'df', 'only in first DataFrame')
        assert_message_contains(result, 'df', 'only in second DataFrame')
    
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
        """Different functions are now ignored and considered equal."""
        def foo():
            return 42

        def bar():
            return 42

        differ = Diff()
        a = {'f': foo}
        b = {'f': bar}
        result = differ.diff(a, b)
        # Functions/callables are now ignored
        assert 'f' not in result
    
    def test_lambda_same(self):
        """Same lambda should be equal."""
        lam = lambda x: x + 1
        
        differ = Diff()
        a = {'f': lam}
        b = {'f': lam}
        assert differ.diff(a, b) == {}
    
    def test_lambda_different(self):
        """Different lambdas are now ignored and considered equal."""
        differ = Diff()
        a = {'f': lambda x: x + 1}
        b = {'f': lambda x: x + 1}
        result = differ.diff(a, b)
        # Lambdas/callables are now ignored
        assert 'f' not in result
    
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
        """Different builtin functions are now ignored and considered equal."""
        differ = Diff()
        a = {'f': len}
        b = {'f': sum}
        result = differ.diff(a, b)
        # Builtin callables are now ignored
        assert 'f' not in result
    
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
        comp = get_any_comparison(result, 'arr')
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
        comp = get_any_comparison(result, 'arr')
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
        comp = get_any_comparison(result, 's')
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
        comp = get_any_comparison(result, 's')
        assert 'b' in comp.message
        assert 'NaN' in comp.message or 'nan' in comp.message.lower()
    
    def test_set_shows_unmatched_element(self):
        """Set mismatch should show which element couldn't be matched."""
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 99}}
        result = differ.diff(a, b)
        assert 's' in result
        # Should mention an element value (either 3 or 99) - now returns CompoundDiff with unmatched elements
        diff_node = result['s']
        assert isinstance(diff_node, CompoundDiff), f"Expected CompoundDiff but got {type(diff_node)}"
        messages = [v.message for v in diff_node.children.values() if isinstance(v, ValueComparison)]
        all_messages = ' '.join(messages)
        assert ('3' in all_messages or '99' in all_messages), f"Expected 3 or 99 in messages: {messages}"

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
        # Should show the index - now returns dict with element diffs
        assert_message_contains(result, 'arr', '(2,)')
        # Should show the values
        assert_message_contains(result, 'arr', '3')
        assert_message_contains(result, 'arr', '99')
    
    def test_multidim_array_shows_index(self):
        """Multidimensional array should show full index."""
        differ = Diff()
        a = {'arr': np.array([[1, 2], [3, 4]])}
        b = {'arr': np.array([[1, 2], [3, 99]])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show 2D index
        comp = get_any_comparison(result, 'arr')
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
        comp = get_any_comparison(result, 's')
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
        comp = get_any_comparison(result, 's')
        assert 'b' in comp.message
        assert 'NaN' in comp.message or 'nan' in comp.message.lower()

    def test_set_shows_unmatched_element_v2(self):
        """Set mismatch should show which element couldn't be matched (v2)."""
        differ = Diff()
        a = {'s': {1, 2, 3}}
        b = {'s': {1, 2, 99}}
        result = differ.diff(a, b)
        assert 's' in result
        # Should mention an element value (either 3 or 99) - now returns CompoundDiff with unmatched elements
        diff_node = result['s']
        assert isinstance(diff_node, CompoundDiff), f"Expected CompoundDiff but got {type(diff_node)}"
        messages = [v.message for v in diff_node.children.values() if isinstance(v, ValueComparison)]
        all_messages = ' '.join(messages)
        assert ('3' in all_messages or '99' in all_messages), f"Expected 3 or 99 in messages: {messages}"

    def test_float_array_shows_values_v2(self):
        """Float array with tolerance mismatch should show values (v2)."""
        differ = Diff(rtol=1e-9, atol=0)
        a = {'arr': np.array([1.0, 2.0, 3.00001])}
        b = {'arr': np.array([1.0, 2.0, 3.00002])}
        result = differ.diff(a, b)
        assert 'arr' in result
        # Should show the differing values
        assert_message_contains(result, 'arr', '3.0000')


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
        # Complex returns CompoundDiff with .imag key
        assert isinstance(result['z'], CompoundDiff)
        assert '.imag' in result['z'].children


class TestCollectAllDifferences:
    """Test that ALL differences are collected, not just the first one."""

    def test_list_collects_all_differences(self):
        """List comparison should find all differing elements."""
        differ = Diff()
        a = {'lst': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        b = {'lst': [1, 99, 3, 88, 5, 77, 7, 66, 9, 55]}
        result = differ.diff(a, b)

        assert 'lst' in result
        assert isinstance(result['lst'], CompoundDiff)

        # Should have differences at indices 1, 3, 5, 7, 9
        assert '[1]' in result['lst'].children
        assert '[3]' in result['lst'].children
        assert '[5]' in result['lst'].children
        assert '[7]' in result['lst'].children
        assert '[9]' in result['lst'].children

        # Should NOT have differences at even indices
        assert '[0]' not in result['lst'].children
        assert '[2]' not in result['lst'].children
        assert '[4]' not in result['lst'].children

    def test_dict_collects_all_differences(self):
        """Dict comparison should find all differing values."""
        differ = Diff()
        a = {'d': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}}
        b = {'d': {'a': 1, 'b': 99, 'c': 3, 'd': 88, 'e': 5}}
        result = differ.diff(a, b)

        assert 'd' in result
        assert isinstance(result['d'], CompoundDiff)

        # Should have differences at keys 'b' and 'd'
        assert "['b']" in result['d'].children
        assert "['d']" in result['d'].children

        # Should NOT have differences at keys 'a', 'c', 'e'
        assert "['a']" not in result['d'].children
        assert "['c']" not in result['d'].children
        assert "['e']" not in result['d'].children

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
        assert isinstance(result['o'], CompoundDiff)

        # Should have differences at attributes b and d
        assert '.b' in result['o'].children
        assert '.d' in result['o'].children

        # Should NOT have differences at attributes a and c
        assert '.a' not in result['o'].children
        assert '.c' not in result['o'].children

    def test_nested_structure_collects_all_levels(self):
        """Nested structures should collect differences at all levels."""
        differ = Diff()
        a = {'data': {'users': [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]}}
        b = {'data': {'users': [{'name': 'Alice', 'age': 31}, {'name': 'Charlie', 'age': 25}]}}
        result = differ.diff(a, b)

        assert 'data' in result
        assert isinstance(result['data'], CompoundDiff)
        assert "['users']" in result['data'].children

        # Should have differences in both list elements
        users_diff = result['data'].children["['users']"]
        assert isinstance(users_diff, CompoundDiff)
        assert '[0]' in users_diff.children  # First user age changed
        assert '[1]' in users_diff.children  # Second user name changed


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
        diff_compound = result['lst']
        assert isinstance(diff_compound, CompoundDiff)
        # Should have at most 3 diffs and truncated flag set
        assert len(diff_compound.children) <= 3
        assert diff_compound.truncated

    def test_dict_respects_max_diffs(self):
        """Dict should stop after max_diffs_per_container."""
        differ = Diff(max_diffs_per_container=5)
        # Create dict with 10 differences
        a = {'d': {str(i): i for i in range(10)}}
        b = {'d': {str(i): i + 1 for i in range(10)}}
        result = differ.diff(a, b)

        assert 'd' in result
        diff_compound = result['d']
        assert isinstance(diff_compound, CompoundDiff)
        # Should have at most 5 diffs and truncated flag set
        assert len(diff_compound.children) <= 5
        assert diff_compound.truncated

    def test_truncation_message_explains_limit(self):
        """Truncation message - now just checks truncated flag is set."""
        differ = Diff(max_diffs_per_container=2)
        a = {'lst': [1, 2, 3, 4, 5]}
        b = {'lst': [11, 12, 13, 14, 15]}
        result = differ.diff(a, b)

        diff_compound = result['lst']
        assert isinstance(diff_compound, CompoundDiff)
        assert diff_compound.truncated
        # With CompoundDiff, truncation is indicated by the truncated field


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

        diff_compound = result['lst']
        assert isinstance(diff_compound, CompoundDiff)
        # Only index 2 should differ
        assert '[2]' in diff_compound.children
        assert '[0]' not in diff_compound.children
        assert '[1]' not in diff_compound.children
        assert '[3]' not in diff_compound.children
        assert '[4]' not in diff_compound.children


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

    def test_compound_diff_returns_compound_diff(self):
        """Compound structure diff should return CompoundDiff."""
        differ = Diff()
        result = differ.diff({'lst': [1, 2]}, {'lst': [1, 99]})
        assert isinstance(result['lst'], CompoundDiff)
        assert '[1]' in result['lst'].children

    def test_nested_diff_has_nested_compound_diffs(self):
        """Nested structures should have nested CompoundDiffs."""
        differ = Diff()
        result = differ.diff(
            {'outer': {'inner': [1, 2, 3]}},
            {'outer': {'inner': [1, 99, 3]}}
        )

        assert isinstance(result['outer'], CompoundDiff)
        assert "['inner']" in result['outer'].children
        assert isinstance(result['outer'].children["['inner']"], CompoundDiff)
        assert '[1]' in result['outer'].children["['inner']"].children

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

        assert "truncated" in markdown


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
        assert isinstance(close_result['data'], CompoundDiff)
        assert "['a']" in close_result['data'].children
        assert "['c']" in close_result['data'].children
        assert "['b']" not in close_result['data'].children

        # diff_result should have data with only b
        assert 'data' in diff_result
        assert isinstance(diff_result['data'], CompoundDiff)
        assert "['b']" in diff_result['data'].children
        assert "['a']" not in diff_result['data'].children
        assert "['c']" not in diff_result['data'].children

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
        assert isinstance(close_result['lst'], CompoundDiff)
        assert '[0]' in close_result['lst'].children
        assert '[2]' in close_result['lst'].children
        assert '[1]' not in close_result['lst'].children

        # diff_result should have list with index 1
        assert 'lst' in diff_result
        assert isinstance(diff_result['lst'], CompoundDiff)
        assert '[1]' in diff_result['lst'].children
        assert '[0]' not in diff_result['lst'].children
        assert '[2]' not in diff_result['lst'].children

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
        assert isinstance(close_result['obj'], CompoundDiff)
        assert '.a' in close_result['obj'].children
        assert '.b' not in close_result['obj'].children

        # diff_result should have obj with only .b
        assert 'obj' in diff_result
        assert isinstance(diff_result['obj'], CompoundDiff)
        assert '.b' in diff_result['obj'].children
        assert '.a' not in diff_result['obj'].children

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

        # Navigate down to check filtering worked - use .children for CompoundDiff
        close_level1 = close_result['level1']
        assert isinstance(close_level1, CompoundDiff)
        close_level2 = close_level1.children["['level2']"]
        assert isinstance(close_level2, CompoundDiff)
        close_leaf = close_level2.children["['level3']"]
        assert isinstance(close_leaf, CompoundDiff)
        assert '[0]' in close_leaf.children
        assert '[1]' not in close_leaf.children

        diff_level1 = diff_result['level1']
        assert isinstance(diff_level1, CompoundDiff)
        diff_level2 = diff_level1.children["['level2']"]
        assert isinstance(diff_level2, CompoundDiff)
        diff_leaf = diff_level2.children["['level3']"]
        assert isinstance(diff_leaf, CompoundDiff)
        assert '[1]' in diff_leaf.children
        assert '[0]' not in diff_leaf.children

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
        assert isinstance(result['z'], CompoundDiff)
        assert '.imag' in result['z'].children
        assert '.real' not in result['z'].children

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
        assert isinstance(result['data'], CompoundDiff)
        assert "['different']" in result['data'].children
        assert "['close']" not in result['data'].children

    def test_report_close_false_list_values(self):
        """report_close=False with lists containing close values."""
        differ = Diff(rtol=1e-5, report_close=False)
        a = {'lst': [1.0000001, 2.0, 3.0000001]}
        b = {'lst': [1.0000002, 99.0, 3.0000002]}
        result = differ.diff(a, b)

        # Should only report index 1 (different)
        assert 'lst' in result
        assert isinstance(result['lst'], CompoundDiff)
        assert '[1]' in result['lst'].children
        assert '[0]' not in result['lst'].children
        assert '[2]' not in result['lst'].children

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
        assert isinstance(result['obj'], CompoundDiff)
        assert '.b' in result['obj'].children
        assert '.a' not in result['obj'].children

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
        level1 = result['level1']
        assert isinstance(level1, CompoundDiff)
        level2 = level1.children["['level2']"]
        assert isinstance(level2, CompoundDiff)
        level3 = level2.children["['level3']"]
        assert isinstance(level3, CompoundDiff)
        assert '[1]' in level3.children
        assert '[0]' not in level3.children

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

class TestMultiDiffSupport:
    """Test the max_diffs_per_container/max_diffs_per_structure parameters."""

    def test_array_multiple_diffs_default(self):
        """Test that arrays collect up to max_diffs_per_container differences."""
        # Arrays now use max_diffs_per_container (treated as containers)
        differ = Diff(max_diffs_per_container=5)
        a = {'arr': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
        b = {'arr': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])}
        result = differ.diff(a, b)

        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have 5 diffs + truncated flag set
        assert len(result['arr'].children) == 5
        assert result['arr'].truncated

    def test_array_multiple_diffs_custom_limit(self):
        """Test custom max_diffs_per_container parameter for arrays."""
        differ = Diff(max_diffs_per_container=3)
        a = {'arr': np.array([1, 2, 3, 4, 5, 6, 7, 8])}
        b = {'arr': np.array([10, 20, 30, 40, 50, 60, 70, 80])}
        result = differ.diff(a, b)

        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have 3 diffs + truncated flag set
        assert len(result['arr'].children) == 3
        assert result['arr'].truncated

    def test_array_no_truncation_when_below_limit(self):
        """Test that no truncation occurs when diffs are below limit."""
        differ = Diff(max_diffs_per_container=5)
        a = {'arr': np.array([1, 2, 3, 4, 5])}
        b = {'arr': np.array([10, 2, 30, 4, 5])}
        result = differ.diff(a, b)

        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have exactly 2 diffs, no truncation
        assert len(result['arr'].children) == 2
        assert not result['arr'].truncated

    def test_series_multiple_diffs(self):
        """Test that Series collect multiple differences (uses max_diffs_per_container)."""
        differ = Diff(max_diffs_per_container=3)
        a = {'s': pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])}
        b = {'s': pd.Series([10, 2, 30, 4, 50, 6, 70, 80], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])}
        result = differ.diff(a, b)

        assert 's' in result
        assert isinstance(result['s'], CompoundDiff)
        # Should have 3 diffs + truncated flag set
        assert len(result['s'].children) == 3
        assert result['s'].truncated

    def test_dataframe_multiple_diffs_across_columns(self):
        """Test that DataFrames collect differences across multiple columns."""
        differ = Diff(max_diffs_per_structure=5)
        a = {'df': pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })}
        b = {'df': pd.DataFrame({
            'A': [1, 20, 3, 40, 5],  # 2 diffs
            'B': [10, 200, 30, 400, 50],  # 2 diffs
            'C': [1000, 200, 3000, 400, 5000]  # 3 diffs
        })}
        result = differ.diff(a, b)

        assert 'df' in result
        assert isinstance(result['df'], CompoundDiff)
        # With nested structure, we have 3 column keys (A, B, C)
        col_keys = list(result['df'].children.keys())
        assert len(col_keys) == 3  # A, B, C columns (nested structure)
        # Total element diffs across columns >= max_diffs_per_structure (5)
        total_element_diffs = sum(
            len(result['df'].children[col].children) if isinstance(result['df'].children[col], CompoundDiff) else 0
            for col in col_keys
        )
        assert total_element_diffs >= 5  # 2 + 2 + 3 = 7 total
        assert result['df'].truncated

    def test_multidim_array_multiple_diffs(self):
        """Test that multidimensional arrays report multiple differences."""
        differ = Diff(max_diffs_per_container=3)
        a = {'arr': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}
        b = {'arr': np.array([[10, 2, 30], [4, 50, 6], [70, 8, 90]])}
        result = differ.diff(a, b)

        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have 3 diffs + truncated flag set (5 total diffs truncated to 3)
        assert len(result['arr'].children) == 3
        assert result['arr'].truncated
        # Check that indices are properly formatted as tuples
        for key in result['arr'].children.keys():
            assert '(' in key and ')' in key  # Should have (row, col) format

    def test_float_array_multiple_diffs(self):
        """Test that float arrays with multiple differences work correctly."""
        differ = Diff(max_diffs_per_container=4, rtol=1e-5, atol=1e-8)
        a = {'arr': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
        b = {'arr': np.array([1.0, 2.5, 3.0, 4.5, 5.0, 6.5])}
        result = differ.diff(a, b)

        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have 3 actual diffs, no truncation
        assert len(result['arr'].children) == 3
        assert not result['arr'].truncated

    def test_mixed_structures_independent_limits(self):
        """Test that arrays and series use max_diffs_per_container."""
        # Arrays and Series now use max_diffs_per_container (treated as containers)
        differ = Diff(max_diffs_per_container=2)
        a = {
            'arr': np.array([1, 2, 3, 4, 5]),
            's': pd.Series([1, 2, 3, 4, 5])
        }
        b = {
            'arr': np.array([10, 20, 30, 40, 50]),
            's': pd.Series([10, 20, 30, 40, 50])
        }
        result = differ.diff(a, b)

        # Both should have independent limits
        assert 'arr' in result
        assert 's' in result
        assert isinstance(result['arr'], CompoundDiff)
        assert isinstance(result['s'], CompoundDiff)

        # Each should be truncated at 2
        assert len(result['arr'].children) == 2
        assert len(result['s'].children) == 2
        assert result['arr'].truncated
        assert result['s'].truncated

    def test_zero_max_diffs_per_container(self):
        """Test edge case with max_diffs_per_container=0 for arrays."""
        differ = Diff(max_diffs_per_container=0)
        a = {'arr': np.array([1, 2, 3])}
        b = {'arr': np.array([10, 20, 30])}
        result = differ.diff(a, b)

        # Should still detect differences exist and report at least one + truncation
        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have 1 diff + truncated flag (reports at least one before truncating)
        assert len(result['arr'].children) >= 1  # Reports at least one diff
        assert result['arr'].truncated

    def test_large_max_diffs_per_container(self):
        """Test with very large max_diffs_per_container (arrays use container limit)."""
        differ = Diff(max_diffs_per_container=1000)
        a = {'arr': np.arange(100)}
        b = {'arr': np.arange(100, 200)}
        result = differ.diff(a, b)

        assert 'arr' in result
        assert isinstance(result['arr'], CompoundDiff)
        # Should have all 100 diffs, no truncation
        assert len(result['arr'].children) == 100
        assert not result['arr'].truncated


# ============================================================================
# USE_LEQ TESTS - Conservative Extension Semantics
# ============================================================================

class TestUseLeqNamespace:
    """Tests for use_leq at the namespace (top-level dict) level."""

    def test_extra_keys_allowed_in_b(self):
        """Extra keys in b should not be reported as differences."""
        differ = Diff(use_leq=True)
        a = {'x': 1, 'y': 2}
        b = {'x': 1, 'y': 2, 'z': 3}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_multiple_extra_keys_allowed(self):
        """Multiple extra keys in b should all be allowed."""
        differ = Diff(use_leq=True)
        a = {'x': 1}
        b = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_missing_key_in_b_detected(self):
        """Keys in a but not in b should still be detected."""
        differ = Diff(use_leq=True)
        a = {'x': 1, 'y': 2}
        b = {'x': 1}  # y is missing
        result = differ.diff(a, b)
        assert_has_diff(result, 'y')
        assert_message_contains(result, 'y', 'removed')

    def test_value_difference_detected(self):
        """Value differences should still be detected with use_leq."""
        differ = Diff(use_leq=True)
        a = {'x': 1, 'y': 2}
        b = {'x': 1, 'y': 999, 'z': 3}  # y has different value, z is extra
        result = differ.diff(a, b)
        assert 'z' not in result  # extra key allowed
        assert_has_diff(result, 'y')  # value difference detected

    def test_empty_a_always_succeeds(self):
        """Empty a should always succeed (b can have anything)."""
        differ = Diff(use_leq=True)
        a = {}
        b = {'x': 1, 'y': 2, 'z': 3}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_empty_b_with_nonempty_a_fails(self):
        """Empty b with non-empty a should fail (missing keys)."""
        differ = Diff(use_leq=True)
        a = {'x': 1}
        b = {}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')

    def test_both_empty_succeeds(self):
        """Both empty namespaces should succeed."""
        differ = Diff(use_leq=True)
        a = {}
        b = {}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_identical_namespaces_succeed(self):
        """Identical namespaces should succeed with use_leq."""
        differ = Diff(use_leq=True)
        a = {'x': 1, 'y': [1, 2, 3], 'z': 'hello'}
        b = {'x': 1, 'y': [1, 2, 3], 'z': 'hello'}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_strict_mode_still_detects_extra_keys(self):
        """Default mode (use_leq=False) should still detect extra keys."""
        differ = Diff(use_leq=False)
        a = {'x': 1}
        b = {'x': 1, 'y': 2}
        result = differ.diff(a, b)
        assert_has_diff(result, 'y')
        assert_message_contains(result, 'y', 'added')


class TestUseLeqDataFrame:
    """Tests for use_leq with pandas DataFrames."""

    def test_extra_columns_allowed(self):
        """Extra columns in b's DataFrame should be allowed."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_multiple_extra_columns_allowed(self):
        """Multiple extra columns should all be allowed."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3]})
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_missing_column_detected(self):
        """Missing columns in b should be detected."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df_b = pd.DataFrame({'A': [1, 2, 3]})  # B is missing
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')
        # Now reports individual column differences
        assert_message_contains(result, 'df', 'missing in second DataFrame')

    def test_multiple_missing_columns_detected(self):
        """Multiple missing columns should be reported."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        df_b = pd.DataFrame({'A': [1, 2, 3]})  # B and C are missing
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')
        # Now reports individual column differences
        assert_message_contains(result, 'df', 'missing in second DataFrame')

    def test_column_value_difference_detected(self):
        """Value differences in columns should still be detected."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df_b = pd.DataFrame({'A': [1, 2, 999], 'B': [4, 5, 6], 'C': [7, 8, 9]})  # A[2] differs
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')

    def test_index_difference_detected(self):
        """Index differences should still be detected."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[0, 1, 3])  # index differs
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')
        assert_message_contains(result, 'df', 'index')

    def test_row_count_difference_detected(self):
        """Different row counts should still be detected."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3]})
        df_b = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})  # extra row
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')

    def test_column_order_independent(self):
        """Column order in b shouldn't matter as long as a's columns exist."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        # b has columns in different order plus extra
        df_b = pd.DataFrame({'C': [7, 8, 9], 'B': [4, 5, 6], 'A': [1, 2, 3]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_empty_dataframes_both(self):
        """Both empty DataFrames should succeed."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame()
        df_b = pd.DataFrame()
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_single_column_dataframe(self):
        """Single column DataFrame comparison should work."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3]})
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_with_nan_values(self):
        """NaN values should be handled correctly."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        df_b = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan], 'C': [7, 8, 9]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_with_different_dtypes_compatible(self):
        """Compatible dtypes should work (e.g., int32 vs int64)."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': np.array([1, 2, 3], dtype=np.int32)})
        df_b = pd.DataFrame({
            'A': np.array([1, 2, 3], dtype=np.int64),
            'B': [4, 5, 6]
        })
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_strict_mode_detects_extra_columns(self):
        """Default mode (use_leq=False) should detect extra columns."""
        differ = Diff(use_leq=False)
        df_a = pd.DataFrame({'A': [1, 2, 3]})
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')


class TestUseLeqCombined:
    """Tests combining namespace and DataFrame use_leq behavior."""

    def test_extra_key_with_extra_columns(self):
        """Extra keys AND extra columns should both be allowed."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3]})
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        a = {'df': df_a, 'x': 1}
        b = {'df': df_b, 'x': 1, 'y': 2}  # extra key y, extra column B
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_multiple_dataframes(self):
        """Multiple DataFrames should all allow extra columns."""
        differ = Diff(use_leq=True)
        a = {
            'df1': pd.DataFrame({'A': [1, 2]}),
            'df2': pd.DataFrame({'X': [3, 4]}),
        }
        b = {
            'df1': pd.DataFrame({'A': [1, 2], 'B': [5, 6]}),
            'df2': pd.DataFrame({'X': [3, 4], 'Y': [7, 8], 'Z': [9, 10]}),
            'df3': pd.DataFrame({'P': [1, 2]}),  # extra DataFrame
        }
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_nested_dict_not_extended(self):
        """Nested dicts should NOT allow extra keys (only top-level)."""
        differ = Diff(use_leq=True)
        a = {'outer': {'inner': 1}}
        b = {'outer': {'inner': 1, 'extra': 2}}  # extra key in nested dict
        result = differ.diff(a, b)
        # Nested dict extra key should be detected
        assert_has_diff(result, 'outer')

    def test_list_not_extended(self):
        """Lists should NOT allow extra elements."""
        differ = Diff(use_leq=True)
        a = {'lst': [1, 2, 3]}
        b = {'lst': [1, 2, 3, 4]}  # extra element
        result = differ.diff(a, b)
        assert_has_diff(result, 'lst')

    def test_dataframe_in_list(self):
        """DataFrame inside list - standard behavior (no extension)."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2, 3]})
        df_b = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        a = {'lst': [df_a]}
        b = {'lst': [df_b]}
        # DataFrame inside list should still use leq for the DataFrame itself
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_mixed_success_and_failure(self):
        """Some variables pass, some fail."""
        differ = Diff(use_leq=True)
        a = {
            'ok1': 1,
            'ok2': pd.DataFrame({'A': [1, 2]}),
            'fail1': 2,
            'fail2': pd.DataFrame({'X': [1, 2], 'Y': [3, 4]}),
        }
        b = {
            'ok1': 1,
            'ok2': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),  # extra col OK
            'fail1': 999,  # different value
            'fail2': pd.DataFrame({'X': [1, 2]}),  # missing column Y
            'extra': 'ignored',  # extra key OK
        }
        result = differ.diff(a, b)
        assert 'ok1' not in result
        assert 'ok2' not in result
        assert 'extra' not in result
        assert_has_diff(result, 'fail1')
        assert_has_diff(result, 'fail2')


class TestUseLeqWithOtherOptions:
    """Tests for use_leq combined with other Diff options."""

    def test_with_report_close_false(self):
        """use_leq should work with report_close=False."""
        differ = Diff(use_leq=True, report_close=False)
        df_a = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        df_b = pd.DataFrame({'A': [1.0000001, 2.0, 3.0], 'B': [4, 5, 6]})  # close value + extra col
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_with_strict_false(self):
        """use_leq should work with strict=False."""
        differ = Diff(use_leq=True, strict=False)
        a = {'x': 1}  # int
        b = {'x': 1.0, 'y': 2}  # float (compatible in non-strict) + extra key
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_with_custom_tolerance(self):
        """use_leq should work with custom rtol/atol."""
        differ = Diff(use_leq=True, rtol=0.1, atol=0.1, report_close=False)
        df_a = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        df_b = pd.DataFrame({'A': [1.05, 2.0, 3.0], 'B': [4, 5, 6]})  # within tolerance
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_with_keys_to_include(self):
        """use_leq should work with keys_to_include filter."""
        differ = Diff(use_leq=True)
        a = {'x': 1, 'y': 2}
        b = {'x': 1, 'y': 999, 'z': 3}  # y differs, z is extra
        # Only check x
        result = differ.diff(a, b, keys_to_include={'x'})
        assert_no_diff(result)
        # Check y too
        result = differ.diff(a, b, keys_to_include={'x', 'y'})
        assert_has_diff(result, 'y')


class TestUseLeqEdgeCases:
    """Edge cases for use_leq."""

    def test_identical_is_leq(self):
        """Identical namespaces should satisfy leq."""
        differ = Diff(use_leq=True)
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        a = {'x': 1, 'df': df}
        b = {'x': 1, 'df': df.copy()}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_subset_columns_same_order(self):
        """a's columns are prefix of b's columns."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df_b = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_subset_columns_different_order(self):
        """a's columns are subset but not prefix of b's columns."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'B': [3, 4], 'D': [7, 8]})
        df_b = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_dataframe_with_string_columns(self):
        """DataFrame with string data should work."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({'name': ['alice', 'bob']})
        df_b = pd.DataFrame({'name': ['alice', 'bob'], 'age': [30, 25]})
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_dataframe_with_mixed_types(self):
        """DataFrame with mixed column types should work."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
        })
        df_b = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'extra_col': [True, False, True],
        })
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_numpy_array_not_extended(self):
        """numpy arrays should NOT allow extra elements."""
        differ = Diff(use_leq=True)
        a = {'arr': np.array([1, 2, 3])}
        b = {'arr': np.array([1, 2, 3, 4])}
        result = differ.diff(a, b)
        assert_has_diff(result, 'arr')

    def test_series_not_extended(self):
        """pandas Series should NOT allow extra elements."""
        differ = Diff(use_leq=True)
        a = {'s': pd.Series([1, 2, 3])}
        b = {'s': pd.Series([1, 2, 3, 4])}
        result = differ.diff(a, b)
        assert_has_diff(result, 's')

    def test_dataframe_with_multiindex_columns(self):
        """DataFrame with MultiIndex columns should work."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_tuples([('A', 'x'), ('A', 'y')])
        )
        df_b = pd.DataFrame(
            [[1, 2, 5], [3, 4, 6]],
            columns=pd.MultiIndex.from_tuples([('A', 'x'), ('A', 'y'), ('B', 'z')])
        )
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_dataframe_datetime_columns(self):
        """DataFrame with datetime columns should work."""
        differ = Diff(use_leq=True)
        df_a = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'value': [1, 2]
        })
        df_b = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'value': [1, 2],
            'extra': [3, 4]
        })
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)


# ============================================================================
# NaT (Not a Time) HANDLING TESTS
# ============================================================================

class TestNaTHandling:
    """Test that NaT (Not a Time) values are compared like NaN - NaT equals NaT."""

    # -------------------------------------------------------------------------
    # numpy.datetime64 scalar tests
    # -------------------------------------------------------------------------

    def test_datetime64_both_nat_equal(self):
        """Two numpy datetime64 NaT values should be equal."""
        differ = Diff()
        a = {'x': np.datetime64('NaT')}
        b = {'x': np.datetime64('NaT')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_datetime64_nat_vs_date_different(self):
        """datetime64 NaT vs actual date should be different."""
        differ = Diff()
        a = {'x': np.datetime64('NaT')}
        b = {'x': np.datetime64('2024-01-01')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaT')

    def test_datetime64_date_vs_nat_different(self):
        """datetime64 actual date vs NaT should be different."""
        differ = Diff()
        a = {'x': np.datetime64('2024-01-01')}
        b = {'x': np.datetime64('NaT')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaT')

    def test_datetime64_same_dates_equal(self):
        """Two identical datetime64 values should be equal."""
        differ = Diff()
        a = {'x': np.datetime64('2024-01-15')}
        b = {'x': np.datetime64('2024-01-15')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_datetime64_different_dates_different(self):
        """Two different datetime64 values should be different."""
        differ = Diff()
        a = {'x': np.datetime64('2024-01-15')}
        b = {'x': np.datetime64('2024-01-16')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'datetime64 mismatch')

    def test_datetime64_different_units_same_time(self):
        """datetime64 with different units but same time should be equal."""
        differ = Diff()
        a = {'x': np.datetime64('2024-01-15', 'D')}
        b = {'x': np.datetime64('2024-01-15', 'D')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    # -------------------------------------------------------------------------
    # numpy.timedelta64 scalar tests
    # -------------------------------------------------------------------------

    def test_timedelta64_both_nat_equal(self):
        """Two numpy timedelta64 NaT values should be equal."""
        differ = Diff()
        a = {'x': np.timedelta64('NaT')}
        b = {'x': np.timedelta64('NaT')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_timedelta64_nat_vs_value_different(self):
        """timedelta64 NaT vs actual duration should be different."""
        differ = Diff()
        a = {'x': np.timedelta64('NaT')}
        b = {'x': np.timedelta64(5, 'D')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaT')

    def test_timedelta64_value_vs_nat_different(self):
        """timedelta64 actual duration vs NaT should be different."""
        differ = Diff()
        a = {'x': np.timedelta64(5, 'D')}
        b = {'x': np.timedelta64('NaT')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaT')

    def test_timedelta64_same_values_equal(self):
        """Two identical timedelta64 values should be equal."""
        differ = Diff()
        a = {'x': np.timedelta64(10, 'D')}
        b = {'x': np.timedelta64(10, 'D')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_timedelta64_different_values_different(self):
        """Two different timedelta64 values should be different."""
        differ = Diff()
        a = {'x': np.timedelta64(10, 'D')}
        b = {'x': np.timedelta64(20, 'D')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'timedelta64 mismatch')

    # -------------------------------------------------------------------------
    # pandas Timestamp tests
    # -------------------------------------------------------------------------

    def test_pandas_timestamp_both_nat_equal(self):
        """Two pandas NaT values should be equal."""
        differ = Diff()
        a = {'x': pd.NaT}
        b = {'x': pd.NaT}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_pandas_timestamp_nat_vs_timestamp_different(self):
        """pandas NaT vs actual Timestamp should be different."""
        differ = Diff()
        a = {'x': pd.NaT}
        b = {'x': pd.Timestamp('2024-01-15')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaT')

    def test_pandas_timestamp_same_equal(self):
        """Two identical pandas Timestamps should be equal."""
        differ = Diff()
        a = {'x': pd.Timestamp('2024-01-15 10:30:00')}
        b = {'x': pd.Timestamp('2024-01-15 10:30:00')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_pandas_timestamp_different_different(self):
        """Two different pandas Timestamps should be different."""
        differ = Diff()
        a = {'x': pd.Timestamp('2024-01-15')}
        b = {'x': pd.Timestamp('2024-01-16')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'Timestamp mismatch')

    # -------------------------------------------------------------------------
    # pandas Timedelta tests
    # -------------------------------------------------------------------------

    def test_pandas_timedelta_both_nat_equal(self):
        """Two pandas Timedelta NaT values should be equal."""
        differ = Diff()
        a = {'x': pd.Timedelta('NaT')}
        b = {'x': pd.Timedelta('NaT')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_pandas_timedelta_nat_vs_value_different(self):
        """pandas Timedelta NaT vs actual value should be different."""
        differ = Diff()
        a = {'x': pd.Timedelta('NaT')}
        b = {'x': pd.Timedelta('5 days')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'one is NaT')

    def test_pandas_timedelta_same_equal(self):
        """Two identical pandas Timedeltas should be equal."""
        differ = Diff()
        a = {'x': pd.Timedelta('5 days 3 hours')}
        b = {'x': pd.Timedelta('5 days 3 hours')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_pandas_timedelta_different_different(self):
        """Two different pandas Timedeltas should be different."""
        differ = Diff()
        a = {'x': pd.Timedelta('5 days')}
        b = {'x': pd.Timedelta('10 days')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'x')
        assert_message_contains(result, 'x', 'Timedelta mismatch')

    # -------------------------------------------------------------------------
    # numpy array tests with NaT
    # -------------------------------------------------------------------------

    def test_datetime64_array_with_nat_same_positions_equal(self):
        """datetime64 arrays with NaT at same positions should be equal."""
        differ = Diff()
        a = {'arr': np.array(['2024-01-01', 'NaT', '2024-01-03'], dtype='datetime64')}
        b = {'arr': np.array(['2024-01-01', 'NaT', '2024-01-03'], dtype='datetime64')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_datetime64_array_with_nat_different_positions_different(self):
        """datetime64 arrays with NaT at different positions should be different."""
        differ = Diff()
        a = {'arr': np.array(['2024-01-01', 'NaT', '2024-01-03'], dtype='datetime64')}
        b = {'arr': np.array(['NaT', '2024-01-02', '2024-01-03'], dtype='datetime64')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'arr')

    def test_datetime64_array_nat_vs_date_different(self):
        """datetime64 array with NaT vs date at same position should be different."""
        differ = Diff()
        a = {'arr': np.array(['2024-01-01', 'NaT', '2024-01-03'], dtype='datetime64')}
        b = {'arr': np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'arr')
        assert_message_contains(result, 'arr', 'one is NaT')

    def test_datetime64_array_multiple_nats_equal(self):
        """datetime64 arrays with multiple NaTs at same positions should be equal."""
        differ = Diff()
        a = {'arr': np.array(['NaT', 'NaT', 'NaT'], dtype='datetime64')}
        b = {'arr': np.array(['NaT', 'NaT', 'NaT'], dtype='datetime64')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_timedelta64_array_with_nat_equal(self):
        """timedelta64 arrays with NaT at same positions should be equal."""
        differ = Diff()
        a = {'arr': np.array([1, 'NaT', 3], dtype='timedelta64[D]')}
        b = {'arr': np.array([1, 'NaT', 3], dtype='timedelta64[D]')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_timedelta64_array_nat_vs_value_different(self):
        """timedelta64 array with NaT vs value should be different."""
        differ = Diff()
        a = {'arr': np.array([1, 'NaT', 3], dtype='timedelta64[D]')}
        b = {'arr': np.array([1, 2, 3], dtype='timedelta64[D]')}
        result = differ.diff(a, b)
        assert_has_diff(result, 'arr')
        assert_message_contains(result, 'arr', 'one is NaT')

    def test_datetime64_2d_array_with_nat_equal(self):
        """2D datetime64 arrays with NaT should be equal."""
        differ = Diff()
        a = {'arr': np.array([['2024-01-01', 'NaT'], ['NaT', '2024-01-04']], dtype='datetime64')}
        b = {'arr': np.array([['2024-01-01', 'NaT'], ['NaT', '2024-01-04']], dtype='datetime64')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    # -------------------------------------------------------------------------
    # pandas Series tests with NaT
    # -------------------------------------------------------------------------

    def test_datetime_series_with_nat_equal(self):
        """datetime Series with NaT at same positions should be equal."""
        differ = Diff()
        a = {'s': pd.Series(pd.to_datetime(['2024-01-01', 'NaT', '2024-01-03']))}
        b = {'s': pd.Series(pd.to_datetime(['2024-01-01', 'NaT', '2024-01-03']))}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_datetime_series_nat_vs_date_different(self):
        """datetime Series with NaT vs date should be different."""
        differ = Diff()
        a = {'s': pd.Series(pd.to_datetime(['2024-01-01', 'NaT', '2024-01-03']))}
        b = {'s': pd.Series(pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))}
        result = differ.diff(a, b)
        assert_has_diff(result, 's')

    def test_timedelta_series_with_nat_equal(self):
        """timedelta Series with NaT at same positions should be equal."""
        differ = Diff()
        a = {'s': pd.Series(pd.to_timedelta(['1 day', 'NaT', '3 days']))}
        b = {'s': pd.Series(pd.to_timedelta(['1 day', 'NaT', '3 days']))}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_timedelta_series_nat_vs_value_different(self):
        """timedelta Series with NaT vs value should be different."""
        differ = Diff()
        a = {'s': pd.Series(pd.to_timedelta(['1 day', 'NaT', '3 days']))}
        b = {'s': pd.Series(pd.to_timedelta(['1 day', '2 days', '3 days']))}
        result = differ.diff(a, b)
        assert_has_diff(result, 's')

    # -------------------------------------------------------------------------
    # pandas DataFrame tests with NaT
    # -------------------------------------------------------------------------

    def test_dataframe_datetime_column_with_nat_equal(self):
        """DataFrame with datetime column containing NaT should be equal."""
        differ = Diff()
        a = {'df': pd.DataFrame({'date': pd.to_datetime(['2024-01-01', 'NaT', '2024-01-03'])})}
        b = {'df': pd.DataFrame({'date': pd.to_datetime(['2024-01-01', 'NaT', '2024-01-03'])})}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_dataframe_datetime_column_nat_vs_date_different(self):
        """DataFrame datetime column with NaT vs date should be different."""
        differ = Diff()
        a = {'df': pd.DataFrame({'date': pd.to_datetime(['2024-01-01', 'NaT', '2024-01-03'])})}
        b = {'df': pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])})}
        result = differ.diff(a, b)
        assert_has_diff(result, 'df')

    def test_dataframe_timedelta_column_with_nat_equal(self):
        """DataFrame with timedelta column containing NaT should be equal."""
        differ = Diff()
        a = {'df': pd.DataFrame({'duration': pd.to_timedelta(['1 day', 'NaT', '3 days'])})}
        b = {'df': pd.DataFrame({'duration': pd.to_timedelta(['1 day', 'NaT', '3 days'])})}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_dataframe_mixed_columns_with_nat_equal(self):
        """DataFrame with mixed columns including NaT should be equal."""
        differ = Diff()
        df_a = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', 'NaT']),
            'value': [1.5, 2.5],
            'name': ['alice', 'bob']
        })
        df_b = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', 'NaT']),
            'value': [1.5, 2.5],
            'name': ['alice', 'bob']
        })
        a = {'df': df_a}
        b = {'df': df_b}
        result = differ.diff(a, b)
        assert_no_diff(result)

    # -------------------------------------------------------------------------
    # Edge cases and nested structures with NaT
    # -------------------------------------------------------------------------

    def test_nat_in_nested_dict(self):
        """NaT values in nested dictionaries should be handled correctly."""
        differ = Diff()
        a = {'data': {'timestamp': np.datetime64('NaT')}}
        b = {'data': {'timestamp': np.datetime64('NaT')}}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_nat_in_list(self):
        """NaT values in lists should be handled correctly."""
        differ = Diff()
        a = {'times': [np.datetime64('2024-01-01'), np.datetime64('NaT')]}
        b = {'times': [np.datetime64('2024-01-01'), np.datetime64('NaT')]}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_nat_in_tuple(self):
        """NaT values in tuples should be handled correctly."""
        differ = Diff()
        a = {'times': (np.datetime64('NaT'), np.datetime64('2024-01-01'))}
        b = {'times': (np.datetime64('NaT'), np.datetime64('2024-01-01'))}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_mixed_nat_and_nan_equal(self):
        """Mixed NaT and NaN values should be handled correctly."""
        differ = Diff()
        a = {
            'timestamp': np.datetime64('NaT'),
            'value': float('nan'),
            'duration': np.timedelta64('NaT')
        }
        b = {
            'timestamp': np.datetime64('NaT'),
            'value': float('nan'),
            'duration': np.timedelta64('NaT')
        }
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_nat_with_different_datetime64_units(self):
        """NaT with different datetime64 units should still be equal."""
        differ = Diff()
        # NaT is NaT regardless of unit
        a = {'x': np.datetime64('NaT', 'D')}
        b = {'x': np.datetime64('NaT', 's')}
        result = differ.diff(a, b)
        assert_no_diff(result)

    def test_nat_with_different_timedelta64_units(self):
        """NaT with different timedelta64 units should still be equal."""
        differ = Diff()
        a = {'x': np.timedelta64('NaT', 'D')}
        b = {'x': np.timedelta64('NaT', 'h')}
        result = differ.diff(a, b)
        assert_no_diff(result)
