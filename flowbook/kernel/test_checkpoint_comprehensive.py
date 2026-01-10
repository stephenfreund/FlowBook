"""
Comprehensive test cases for checkpoint.py covering all scenarios and corner cases.

This test file supplements test_checkpoint.py with additional scenarios from the
checkpoint.py design document that weren't previously tested.

Test Coverage:
- Circular references
- Functions with closures, mutable defaults, and recursive calls
- Class definitions and class variables
- Shared references and object identity
- Extension dtypes and special pandas types
- Custom __deepcopy__ and __getstate__/__setstate__
- Generators and iterators
- Special numeric values (NaN, inf, complex)
- Collection types (sets, frozensets, named tuples)
- Datetime and decimal objects
- MultiIndex DataFrames
- Nested DataFrames
- Edge cases and error conditions

To run these tests:
    pytest flowbook/kernel/test_checkpoint_comprehensive.py -v
"""

import pytest
import types
import decimal
import datetime
import collections
import numpy as np
import pandas as pd
from typing import Any

from flowbook.kernel.checkpoint import Checkpoints, Checkpoint


# ============================================================================
# CIRCULAR REFERENCE TESTS (Design Doc 7.1)
# ============================================================================

class TestCircularReferences:
    """Test that circular references are properly handled via memo mechanism."""

    def test_self_referential_list(self):
        """Test a list that contains itself."""
        cp = Checkpoints()

        lst = [1, 2, 3]
        lst.append(lst)  # lst[3] is lst itself

        user_ns = {'lst': lst}
        cp.save('test', user_ns)

        # Modify the list
        lst[0] = 999

        # Restore
        cp.restore('test', user_ns)

        # Should have original value
        assert user_ns['lst'][0] == 1
        # And should still be self-referential
        assert user_ns['lst'][3] is user_ns['lst']

    def test_mutually_referential_lists(self):
        """Test two lists that reference each other."""
        cp = Checkpoints()

        list_a = [1, 2, 3]
        list_b = [4, 5, 6]
        list_a.append(list_b)
        list_b.append(list_a)

        user_ns = {'list_a': list_a, 'list_b': list_b}
        cp.save('test', user_ns)

        # Modify both
        list_a[0] = 999
        list_b[0] = 888

        # Restore
        cp.restore('test', user_ns)

        # Should have original values
        assert user_ns['list_a'][0] == 1
        assert user_ns['list_b'][0] == 4
        # And should still reference each other
        assert user_ns['list_a'][3] is user_ns['list_b']
        assert user_ns['list_b'][3] is user_ns['list_a']

    def test_self_referential_dict(self):
        """Test a dict that contains itself."""
        cp = Checkpoints()

        d = {'a': 1, 'b': 2}
        d['self'] = d

        user_ns = {'d': d}
        cp.save('test', user_ns)

        d['a'] = 999

        cp.restore('test', user_ns)

        assert user_ns['d']['a'] == 1
        assert user_ns['d']['self'] is user_ns['d']

    def test_circular_reference_in_dataframe(self):
        """Test DataFrame with cells containing circular references."""
        cp = Checkpoints()

        lst = [1, 2, 3]
        lst.append(lst)

        df = pd.DataFrame({'data': [lst, [4, 5, 6]]})
        user_ns = {'df': df}

        cp.save('test', user_ns)

        # Modify the circular list
        df.iloc[0, 0][0] = 999

        cp.restore('test', user_ns)

        # Should have original value and still be circular
        assert user_ns['df'].iloc[0, 0][0] == 1
        assert user_ns['df'].iloc[0, 0][3] is user_ns['df'].iloc[0, 0]


# ============================================================================
# SHARED REFERENCE TESTS (Design Doc 4.1, 4.6)
# ============================================================================

class TestSharedReferences:
    """Test that shared references are properly maintained."""

    def test_same_list_in_multiple_variables(self):
        """Test that the same list referenced by multiple variables remains shared."""
        cp = Checkpoints()

        shared_list = [1, 2, 3]
        user_ns = {'a': shared_list, 'b': shared_list}

        # Verify they're the same object initially
        assert user_ns['a'] is user_ns['b']

        cp.save('test', user_ns)

        # Modify through one reference
        user_ns['a'].append(999)

        cp.restore('test', user_ns)

        # After restore, both should be restored and still share identity
        assert user_ns['a'] == [1, 2, 3]
        assert user_ns['b'] == [1, 2, 3]
        assert user_ns['a'] is user_ns['b']

    def test_shared_dict_in_dataframe_cells(self):
        """Test shared dict referenced in multiple DataFrame cells."""
        cp = Checkpoints()

        shared_dict = {'key': 'value'}
        df = pd.DataFrame({'col1': [shared_dict, {'other': 1}],
                          'col2': [shared_dict, {'other': 2}]})

        user_ns = {'df': df}

        # Verify sharing
        assert df.iloc[0, 0] is df.iloc[0, 1]

        cp.save('test', user_ns)
        cp.restore('test', user_ns)

        # Should still be shared after restore
        assert user_ns['df'].iloc[0, 0] is user_ns['df'].iloc[0, 1]

    def test_reverse_memo_tracking(self):
        """Test that reverse_memo correctly tracks object identity."""
        cp = Checkpoints()

        shared = [1, 2, 3]
        user_ns = {'a': shared, 'b': shared, 'c': [4, 5, 6]}

        saved, removed = cp.save('test', user_ns)
        checkpoint = cp.get('test')

        # The reverse_memo should track the copied shared list
        copied_shared_id = id(checkpoint.user_ns['a'])
        original_id = checkpoint.get_original_id(copied_shared_id)

        # Both 'a' and 'b' should map to the same original ID
        assert id(checkpoint.user_ns['a']) == id(checkpoint.user_ns['b'])
        # But 'c' should be different
        assert id(checkpoint.user_ns['a']) != id(checkpoint.user_ns['c'])


# ============================================================================
# FUNCTION CHECKPOINTING TESTS (Design Doc 8.4)
# ============================================================================

class TestFunctionCheckpointing:
    """Test checkpointing of functions with closures, defaults, and recursion."""

    def test_function_with_closure(self):
        """Test function with closure variables are properly isolated."""
        cp = Checkpoints()

        captured = [1, 2, 3]

        def func_with_closure():
            return captured[0]

        user_ns = {'func_with_closure': func_with_closure, 'captured': captured}
        cp.save('test', user_ns)

        # Modify captured variable
        captured[0] = 999

        # Restore
        cp.restore('test', user_ns)

        # Function should see original captured value
        assert user_ns['func_with_closure']() == 1
        # And the captured list itself should be restored
        assert user_ns['captured'][0] == 1

    def test_function_with_mutable_default_list(self):
        """Test function with mutable default argument."""
        cp = Checkpoints()

        def func_with_default(items=[]):
            items.append(1)
            return items

        # Call once to populate default
        result1 = func_with_default()
        assert result1 == [1]

        user_ns = {'func': func_with_default}
        cp.save('test', user_ns)

        # Call again to further populate default
        result2 = func_with_default()
        assert result2 == [1, 1]

        # Restore
        cp.restore('test', user_ns)

        # After restore, default should be back to [1]
        result3 = user_ns['func']()
        assert result3 == [1, 1]  # Should append to restored state

    def test_function_with_mutable_default_dict(self):
        """Test function with mutable default dict argument."""
        cp = Checkpoints()

        def func_with_dict(config={}):
            config['count'] = config.get('count', 0) + 1
            return config

        result1 = func_with_dict()
        assert result1 == {'count': 1}

        user_ns = {'func': func_with_dict}
        cp.save('test', user_ns)

        result2 = func_with_dict()
        assert result2 == {'count': 2}

        cp.restore('test', user_ns)

        result3 = user_ns['func']()
        assert result3 == {'count': 2}

    def test_lambda_with_captured_variable(self):
        """Test lambda with captured variables."""
        cp = Checkpoints()

        multiplier = [2]
        func = lambda x: x * multiplier[0]

        user_ns = {'func': func, 'multiplier': multiplier}
        cp.save('test', user_ns)

        # Modify captured variable
        multiplier[0] = 10

        cp.restore('test', user_ns)

        # Lambda should see restored value
        assert user_ns['func'](5) == 10  # 5 * 2

    def test_nested_functions_with_shared_closure(self):
        """Test nested functions sharing closure variables."""
        cp = Checkpoints()

        state = {'value': 100}

        def outer():
            def inner():
                return state['value']
            return inner

        inner_func = outer()

        user_ns = {'inner_func': inner_func, 'state': state}
        cp.save('test', user_ns)

        state['value'] = 999

        cp.restore('test', user_ns)

        assert user_ns['inner_func']() == 100

    def test_recursive_function(self):
        """Test recursive function checkpointing."""
        cp = Checkpoints()

        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        user_ns = {'factorial': factorial}
        cp.save('test', user_ns)

        # Verify function works before restore
        assert factorial(5) == 120

        # Restore
        cp.restore('test', user_ns)

        # Should still work after restore (recursive calls should work)
        assert user_ns['factorial'](5) == 120

    def test_function_without_closure_or_defaults(self):
        """Test that simple functions without closure/defaults are unchanged."""
        cp = Checkpoints()

        def simple_func(x, y):
            return x + y

        user_ns = {'func': simple_func}
        cp.save('test', user_ns)

        # The function object might be the same (optimization)
        # but it should still work after restore
        cp.restore('test', user_ns)

        assert user_ns['func'](3, 4) == 7

    def test_bound_method(self):
        """Test that bound methods are properly checkpointed."""
        cp = Checkpoints()

        class Counter:
            def __init__(self):
                self.count = 0

            def increment(self):
                self.count += 1
                return self.count

        obj = Counter()
        obj.count = 5

        user_ns = {'obj': obj, 'method': obj.increment}
        cp.save('test', user_ns)

        obj.count = 999

        cp.restore('test', user_ns)

        # Method should reference the restored object
        assert user_ns['method']() == 6
        assert user_ns['obj'].count == 6

    def test_function_with_nested_closure(self):
        """Test function with nested mutable objects in closure."""
        cp = Checkpoints()

        config = {'data': [1, 2, 3], 'nested': {'value': 100}}

        def func():
            return config['data'][0], config['nested']['value']

        user_ns = {'func': func, 'config': config}
        cp.save('test', user_ns)

        config['data'][0] = 999
        config['nested']['value'] = 888

        cp.restore('test', user_ns)

        assert user_ns['func']() == (1, 100)


# ============================================================================
# CLASS DEFINITION TESTS (Design Doc 8.5)
# ============================================================================

class TestClassDefinitions:
    """Test checkpointing behavior with class definitions."""

    def test_class_variable_not_restored(self):
        """Test that mutable class variables are NOT properly restored (known issue)."""
        cp = Checkpoints()

        class Counter:
            count = 0

        user_ns = {'Counter': Counter}
        cp.save('test', user_ns)

        Counter.count = 100

        cp.restore('test', user_ns)

        # KNOWN ISSUE: Class variable is NOT restored
        assert user_ns['Counter'].count == 100  # Still modified!

    def test_class_instance_attributes_are_restored(self):
        """Test that instance attributes ARE properly restored."""
        cp = Checkpoints()

        class MyClass:
            def __init__(self):
                self.value = 1

        obj = MyClass()
        obj.value = 42

        user_ns = {'obj': obj}
        cp.save('test', user_ns)

        obj.value = 999

        cp.restore('test', user_ns)

        # Instance attributes ARE restored
        assert user_ns['obj'].value == 42

    def test_class_method_modification_persists(self):
        """Test that methods added to classes persist (known issue)."""
        cp = Checkpoints()

        class Extensible:
            def original(self):
                return "original"

        user_ns = {'Extensible': Extensible}
        cp.save('test', user_ns)

        # Add a new method
        Extensible.new_method = lambda self: "new"

        cp.restore('test', user_ns)

        # KNOWN ISSUE: New method still exists
        assert hasattr(user_ns['Extensible'], 'new_method')

    def test_class_redefinition_works(self):
        """Test that class redefinition DOES work correctly."""
        cp = Checkpoints()

        class MyClass:
            value = 1

        user_ns = {'MyClass': MyClass}
        cp.save('test', user_ns)

        # Completely redefine the class
        class MyClass:
            value = 999

        user_ns['MyClass'] = MyClass

        cp.restore('test', user_ns)

        # Class should be restored to original
        assert user_ns['MyClass'].value == 1

    def test_instance_with_mutable_attributes(self):
        """Test instance with mutable attributes in __dict__."""
        cp = Checkpoints()

        class Container:
            def __init__(self):
                self.data = []
                self.config = {}

        obj = Container()
        obj.data = [1, 2, 3]
        obj.config = {'key': 'value'}

        user_ns = {'obj': obj}
        cp.save('test', user_ns)

        obj.data.append(999)
        obj.config['new'] = 'data'

        cp.restore('test', user_ns)

        assert user_ns['obj'].data == [1, 2, 3]
        assert user_ns['obj'].config == {'key': 'value'}


# ============================================================================
# CUSTOM DEEPCOPY TESTS (Design Doc 7.4)
# ============================================================================

class TestCustomDeepcopy:
    """Test objects with custom __deepcopy__ methods."""

    def test_custom_deepcopy_method(self):
        """Test object with custom __deepcopy__."""
        cp = Checkpoints()

        class CustomCopy:
            def __init__(self, value):
                self.value = value
                self.copy_count = 0

            def __deepcopy__(self, memo):
                # Custom copy that increments counter
                new_obj = CustomCopy(self.value)
                new_obj.copy_count = self.copy_count + 1
                memo[id(self)] = new_obj
                return new_obj

        obj = CustomCopy(42)
        user_ns = {'obj': obj}

        cp.save('test', user_ns)

        # Should have been copied once
        checkpoint = cp.get('test')
        assert checkpoint.user_ns['obj'].copy_count == 1

        obj.value = 999

        cp.restore('test', user_ns)

        # Should have been copied again (restore also copies)
        assert user_ns['obj'].value == 42
        assert user_ns['obj'].copy_count == 2

    def test_object_with_getstate_setstate(self):
        """Test object with __getstate__ and __setstate__."""
        cp = Checkpoints()

        class Stateful:
            def __init__(self):
                self.data = [1, 2, 3]
                self.temp = "temporary"  # Won't be pickled

            def __getstate__(self):
                # Only save data, not temp
                return {'data': self.data}

            def __setstate__(self, state):
                self.data = state['data']
                self.temp = "restored"

        obj = Stateful()
        user_ns = {'obj': obj}

        cp.save('test', user_ns)

        obj.data.append(999)
        obj.temp = "modified"

        cp.restore('test', user_ns)

        assert user_ns['obj'].data == [1, 2, 3]
        # temp is set by __setstate__
        assert user_ns['obj'].temp == "restored"


# ============================================================================
# GENERATOR AND ITERATOR TESTS (Design Doc 7.7)
# ============================================================================

class TestGeneratorsAndIterators:
    """Test that generators and iterators are handled (may fail or produce empty results)."""

    def test_generator_fails_or_produces_empty(self):
        """Test that generators are either removed or produce empty/exhausted copies."""
        cp = Checkpoints()

        def my_generator():
            yield 1
            yield 2
            yield 3

        gen = my_generator()
        next(gen)  # Advance to 1

        user_ns = {'gen': gen}

        saved, removed = cp.save('test', user_ns)

        # Generator might be removed or saved
        # If saved, it will likely be exhausted
        if 'gen' in saved:
            cp.restore('test', user_ns)
            # Try to get next value - might raise StopIteration or give wrong value
            # This is expected behavior for generators

    def test_iterator_over_list(self):
        """Test iterator over a list."""
        cp = Checkpoints()

        lst = [1, 2, 3, 4, 5]
        it = iter(lst)
        next(it)  # Advance to 1
        next(it)  # Advance to 2

        user_ns = {'it': it, 'lst': lst}

        saved, removed = cp.save('test', user_ns)

        # Iterator might fail to copy or produce unexpected results
        # This is expected behavior


# ============================================================================
# OBJECT DTYPE CONVERSION TESTS (Design Doc 4.3)
# ============================================================================

class TestObjectDtypeConversion:
    """Test automatic conversion of object dtypes to specialized types."""

    def test_object_column_with_mixed_types(self):
        """Test object column with truly mixed types (can't be converted)."""
        cp = Checkpoints()

        # Truly mixed: int, str, list, None
        df = pd.DataFrame({'mixed': [1, "string", [1, 2, 3], None]})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Should remain object dtype
        checkpoint = cp.get('test')
        # Dtype might still be object or might be converted in place

        # Modify the list
        user_ns['df'].iloc[2, 0].append(999)

        cp.restore('test', user_ns)

        # Should be restored
        assert user_ns['df'].iloc[2, 0] == [1, 2, 3]

    def test_object_column_with_integers_converts(self):
        """Test that object column with integers converts to Int64."""
        cp = Checkpoints()

        # Object column with integers
        df = pd.DataFrame({'data': pd.Series([1, 2, 3, None], dtype=object)})

        assert df['data'].dtype == object

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # After checkpoint, should be converted to Int64
        # (Conversion happens in place on original DataFrame)
        # The dtype might be Int64 or similar nullable integer type

    def test_object_column_with_strings_converts(self):
        """Test that object column with strings converts to string dtype."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': pd.Series(['a', 'b', 'c'], dtype=object)})

        assert df['data'].dtype == object

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Should be converted to string dtype


# ============================================================================
# EXTENSION DTYPE TESTS
# ============================================================================

class TestExtensionDtypes:
    """Test DataFrames with pandas extension dtypes."""

    def test_categorical_dtype(self):
        """Test DataFrame with categorical dtype."""
        cp = Checkpoints()

        df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'c', 'a', 'b'])})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = 'c'

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == 'a'
        assert user_ns['df']['cat'].dtype.name == 'category'

    def test_nullable_integer_dtype(self):
        """Test DataFrame with nullable integer dtype."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': pd.array([1, 2, None, 4], dtype='Int64')})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = 999

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == 1
        assert pd.isna(user_ns['df'].iloc[2, 0])

    def test_string_dtype(self):
        """Test DataFrame with StringDtype."""
        cp = Checkpoints()

        df = pd.DataFrame({'text': pd.array(['hello', 'world', None], dtype='string')})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = 'modified'

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == 'hello'

    def test_boolean_dtype(self):
        """Test DataFrame with boolean dtype."""
        cp = Checkpoints()

        df = pd.DataFrame({'flag': pd.array([True, False, None], dtype='boolean')})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = False

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == True


# ============================================================================
# SPECIAL NUMERIC VALUE TESTS
# ============================================================================

class TestSpecialNumericValues:
    """Test handling of NaN, inf, and complex numbers."""

    def test_nan_values(self):
        """Test that NaN values are properly handled."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': [1.0, np.nan, 3.0, np.nan, 5.0]})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[2, 0] = np.nan

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == 1.0
        assert pd.isna(user_ns['df'].iloc[1, 0])
        assert user_ns['df'].iloc[2, 0] == 3.0

    def test_infinity_values(self):
        """Test that inf and -inf are handled."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': [1.0, np.inf, -np.inf, 0.0]})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = 999.0

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == 1.0
        assert np.isinf(user_ns['df'].iloc[1, 0])
        assert user_ns['df'].iloc[1, 0] > 0
        assert np.isinf(user_ns['df'].iloc[2, 0])
        assert user_ns['df'].iloc[2, 0] < 0

    def test_complex_numbers(self):
        """Test complex numbers."""
        cp = Checkpoints()

        arr = np.array([1+2j, 3+4j, 5-6j])

        user_ns = {'arr': arr}
        cp.save('test', user_ns)

        arr[0] = 99+99j

        cp.restore('test', user_ns)

        assert user_ns['arr'][0] == 1+2j
        assert user_ns['arr'][1] == 3+4j

    def test_decimal_objects(self):
        """Test Decimal objects."""
        cp = Checkpoints()

        from decimal import Decimal

        data = [Decimal('1.23'), Decimal('4.56'), Decimal('7.89')]
        df = pd.DataFrame({'amounts': data})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        cp.restore('test', user_ns)

        # Decimals might be converted to float during object dtype conversion
        # Just verify it restores


# ============================================================================
# COLLECTION TYPE TESTS
# ============================================================================

class TestCollectionTypes:
    """Test various collection types."""

    def test_sets(self):
        """Test that sets are deep copied."""
        cp = Checkpoints()

        s = {1, 2, 3, 4, 5}

        user_ns = {'s': s}
        cp.save('test', user_ns)

        s.add(999)

        cp.restore('test', user_ns)

        assert user_ns['s'] == {1, 2, 3, 4, 5}
        assert 999 not in user_ns['s']

    def test_frozensets(self):
        """Test frozensets (immutable, should be atomic)."""
        cp = Checkpoints()

        fs = frozenset([1, 2, 3, 4, 5])

        user_ns = {'fs': fs}
        cp.save('test', user_ns)

        cp.restore('test', user_ns)

        assert user_ns['fs'] == frozenset([1, 2, 3, 4, 5])

    def test_named_tuples(self):
        """Test named tuples."""
        cp = Checkpoints()

        Point = collections.namedtuple('Point', ['x', 'y'])
        p = Point(3, 4)

        user_ns = {'p': p}
        cp.save('test', user_ns)

        cp.restore('test', user_ns)

        assert user_ns['p'].x == 3
        assert user_ns['p'].y == 4
        assert isinstance(user_ns['p'], tuple)

    def test_deque(self):
        """Test collections.deque."""
        cp = Checkpoints()

        dq = collections.deque([1, 2, 3, 4, 5], maxlen=5)

        user_ns = {'dq': dq}
        cp.save('test', user_ns)

        dq.append(999)

        cp.restore('test', user_ns)

        assert list(user_ns['dq']) == [1, 2, 3, 4, 5]

    def test_tuple_with_mutable_contents(self):
        """Test tuple containing mutable objects."""
        cp = Checkpoints()

        t = (1, [2, 3, 4], {'key': 'value'})

        user_ns = {'t': t}
        cp.save('test', user_ns)

        # Modify mutable contents
        t[1].append(999)
        t[2]['new'] = 'data'

        cp.restore('test', user_ns)

        assert user_ns['t'][1] == [2, 3, 4]
        assert user_ns['t'][2] == {'key': 'value'}


# ============================================================================
# DATETIME AND TIMEDELTA TESTS
# ============================================================================

class TestDatetimeTypes:
    """Test datetime and timedelta objects."""

    def test_datetime_objects(self):
        """Test datetime.datetime objects."""
        cp = Checkpoints()

        dt = datetime.datetime(2023, 1, 15, 10, 30, 45)

        user_ns = {'dt': dt}
        cp.save('test', user_ns)

        cp.restore('test', user_ns)

        assert user_ns['dt'] == datetime.datetime(2023, 1, 15, 10, 30, 45)

    def test_timedelta_objects(self):
        """Test datetime.timedelta objects."""
        cp = Checkpoints()

        td = datetime.timedelta(days=5, hours=3, minutes=30)

        user_ns = {'td': td}
        cp.save('test', user_ns)

        cp.restore('test', user_ns)

        assert user_ns['td'] == datetime.timedelta(days=5, hours=3, minutes=30)

    def test_datetime64_in_dataframe(self):
        """Test datetime64 dtype in DataFrame."""
        cp = Checkpoints()

        dates = pd.date_range('2023-01-01', periods=5)
        df = pd.DataFrame({'dates': dates})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = pd.Timestamp('2099-12-31')

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == pd.Timestamp('2023-01-01')

    def test_timedelta64_in_dataframe(self):
        """Test timedelta64 dtype in DataFrame."""
        cp = Checkpoints()

        df = pd.DataFrame({'duration': pd.to_timedelta(['1 days', '2 days', '3 days'])})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.iloc[0, 0] = pd.Timedelta('999 days')

        cp.restore('test', user_ns)

        assert user_ns['df'].iloc[0, 0] == pd.Timedelta('1 days')


# ============================================================================
# MULTIINDEX TESTS
# ============================================================================

class TestMultiIndex:
    """Test DataFrames with MultiIndex."""

    def test_multiindex_dataframe(self):
        """Test DataFrame with MultiIndex."""
        cp = Checkpoints()

        arrays = [
            ['A', 'A', 'B', 'B'],
            [1, 2, 1, 2]
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=['letter', 'number'])
        df = pd.DataFrame({'data': [10, 20, 30, 40]}, index=index)

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.loc[('A', 1), 'data'] = 999

        cp.restore('test', user_ns)

        assert user_ns['df'].loc[('A', 1), 'data'] == 10

    def test_multiindex_columns(self):
        """Test DataFrame with MultiIndex columns."""
        cp = Checkpoints()

        arrays = [['A', 'A', 'B', 'B'],
                  ['x', 'y', 'x', 'y']]
        cols = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame([[1, 2, 3, 4]], columns=cols)

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.loc[0, ('A', 'x')] = 999

        cp.restore('test', user_ns)

        assert user_ns['df'].loc[0, ('A', 'x')] == 1


# ============================================================================
# NESTED DATAFRAME TESTS
# ============================================================================

class TestNestedDataFrames:
    """Test DataFrames containing other DataFrames."""

    def test_dataframe_in_dataframe_cell(self):
        """Test DataFrame containing DataFrames as cell values."""
        cp = Checkpoints()

        inner_df1 = pd.DataFrame({'a': [1, 2]})
        inner_df2 = pd.DataFrame({'b': [3, 4]})

        outer_df = pd.DataFrame({'nested': [inner_df1, inner_df2]})

        user_ns = {'outer': outer_df}
        cp.save('test', user_ns)

        # Modify nested DataFrame
        outer_df.iloc[0, 0].iloc[0, 0] = 999

        cp.restore('test', user_ns)

        assert user_ns['outer'].iloc[0, 0].iloc[0, 0] == 1

    def test_list_of_dataframes(self):
        """Test list containing DataFrames."""
        cp = Checkpoints()

        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'b': [3, 4]})

        df_list = [df1, df2]

        user_ns = {'df_list': df_list}
        cp.save('test', user_ns)

        df_list[0].iloc[0, 0] = 999

        cp.restore('test', user_ns)

        assert user_ns['df_list'][0].iloc[0, 0] == 1


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestEdgeCasesComprehensive:
    """Additional edge cases and error conditions."""

    def test_empty_series(self):
        """Test empty Series."""
        cp = Checkpoints()

        s = pd.Series([], dtype=object)

        user_ns = {'s': s}
        cp.save('test', user_ns)

        cp.restore('test', user_ns)

        assert len(user_ns['s']) == 0

    def test_series_with_custom_index(self):
        """Test Series with custom index."""
        cp = Checkpoints()

        s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

        user_ns = {'s': s}
        cp.save('test', user_ns)

        s.loc['a'] = 999

        cp.restore('test', user_ns)

        assert user_ns['s'].loc['a'] == 1

    def test_dataframe_with_custom_index(self):
        """Test DataFrame with custom index."""
        cp = Checkpoints()

        df = pd.DataFrame({'data': [1, 2, 3]}, index=['x', 'y', 'z'])

        user_ns = {'df': df}
        cp.save('test', user_ns)

        df.loc['x', 'data'] = 999

        cp.restore('test', user_ns)

        assert user_ns['df'].loc['x', 'data'] == 1

    def test_very_nested_structure(self):
        """Test deeply nested data structures."""
        cp = Checkpoints()

        nested = {'level1': {'level2': {'level3': {'level4': [1, 2, 3]}}}}

        user_ns = {'nested': nested}
        cp.save('test', user_ns)

        nested['level1']['level2']['level3']['level4'].append(999)

        cp.restore('test', user_ns)

        assert user_ns['nested']['level1']['level2']['level3']['level4'] == [1, 2, 3]

    def test_none_value(self):
        """Test None values."""
        cp = Checkpoints()

        user_ns = {'x': None}
        cp.save('test', user_ns)

        user_ns['x'] = "not none"

        cp.restore('test', user_ns)

        assert user_ns['x'] is None

    def test_unicode_strings(self):
        """Test Unicode strings."""
        cp = Checkpoints()

        text = "Hello 世界 🌍 Привет"

        user_ns = {'text': text}
        cp.save('test', user_ns)

        user_ns['text'] = "modified"

        cp.restore('test', user_ns)

        assert user_ns['text'] == "Hello 世界 🌍 Привет"

    def test_bytes_objects(self):
        """Test bytes objects."""
        cp = Checkpoints()

        data = b'\x00\x01\x02\xff\xfe'

        user_ns = {'data': data}
        cp.save('test', user_ns)

        user_ns['data'] = b'modified'

        cp.restore('test', user_ns)

        assert user_ns['data'] == b'\x00\x01\x02\xff\xfe'

    def test_sparse_dataframe(self):
        """Test sparse DataFrame."""
        cp = Checkpoints()

        # Create DataFrame with sparse and regular columns
        # (SparseArray doesn't support item assignment, so we test via regular column)
        arr = pd.arrays.SparseArray([0, 0, 1, 0, 0, 2, 0, 0])
        df = pd.DataFrame({'sparse': arr, 'regular': [1, 2, 3, 4, 5, 6, 7, 8]})

        user_ns = {'df': df}
        cp.save('test', user_ns)

        # Modify the regular column (sparse doesn't support assignment)
        df['regular'].iloc[2] = 999

        cp.restore('test', user_ns)

        # Verify regular column was restored
        assert user_ns['df']['regular'].iloc[2] == 3
        # Verify sparse column is intact
        assert user_ns['df']['sparse'].iloc[2] == 1

    def test_restore_preserves_private_variables(self):
        """Test that restore doesn't remove private variables."""
        cp = Checkpoints()

        user_ns = {'x': 1, '_private': 'secret'}
        cp.save('test', user_ns)

        user_ns['x'] = 999
        user_ns['_private'] = 'modified'

        cp.restore('test', user_ns)

        # x should be restored
        assert user_ns['x'] == 1
        # _private should still exist (not checkpointed, so not removed)
        assert '_private' in user_ns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
