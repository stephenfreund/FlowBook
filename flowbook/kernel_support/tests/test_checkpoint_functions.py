"""
Unit tests for function and closure checkpointing behavior.

Tests verify that functions with closures are properly deep copied,
ensuring that closure state is isolated across checkpoint/restore cycles.

To run these tests:
    pytest flowbook/kernel/test_checkpoint_functions.py -v
"""

import pytest
import types
import numpy as np
import pandas as pd
from typing import Any, List

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints, _deep_copy_function


# ============================================================================
# CLOSURE ISOLATION TESTS
# ============================================================================

class TestFunctionClosureIsolation:
    """Test that function closures are properly deep copied and isolated."""

    def test_function_with_list_closure(self):
        """Test that list variables captured in closures are isolated."""
        cp = MemoryCheckpoints()

        # Create function with list closure
        counter = [0]
        def increment():
            counter[0] += 1
            return counter[0]

        user_ns = {'increment': increment, 'counter': counter}
        cp.save('before', user_ns)

        # Call original function
        assert increment() == 1
        assert increment() == 2
        assert counter == [2]

        # Restore checkpoint
        cp.restore('before', user_ns)

        # Restored function should have its own counter at [0]
        assert user_ns['counter'] == [0]
        assert user_ns['increment']() == 1
        assert user_ns['counter'] == [1]

        # Original checkpoint should still be pristine
        checkpoint = cp.get('before')
        assert checkpoint.user_ns['counter'] == [0]

    def test_function_with_dict_closure(self):
        """Test that dict variables captured in closures are isolated."""
        cp = MemoryCheckpoints()

        # Create function with dict closure
        state = {'count': 0, 'history': []}
        def track(value):
            state['count'] += 1
            state['history'].append(value)
            return state['count']

        user_ns = {'track': track, 'state': state}
        cp.save('initial', user_ns)

        # Use original function
        track('a')
        track('b')
        assert state == {'count': 2, 'history': ['a', 'b']}

        # Restore
        cp.restore('initial', user_ns)

        # Should be reset
        assert user_ns['state'] == {'count': 0, 'history': []}
        user_ns['track']('x')
        assert user_ns['state'] == {'count': 1, 'history': ['x']}

    def test_function_closure_shared_with_variable(self):
        """Test when a function's closure variable is also a namespace variable."""
        cp = MemoryCheckpoints()

        data = [1, 2, 3]
        def get_sum():
            return sum(data)

        user_ns = {'data': data, 'get_sum': get_sum}
        cp.save('test', user_ns)

        # Both the function's closure and the variable reference the same list
        # After checkpoint, they should both be copied and still reference
        # the same (copied) list
        data.append(4)
        assert get_sum() == 10

        cp.restore('test', user_ns)
        assert user_ns['data'] == [1, 2, 3]
        assert user_ns['get_sum']() == 6  # Should use restored data

    def test_lambda_with_closure(self):
        """Test that lambdas with closures are properly handled."""
        cp = MemoryCheckpoints()

        multiplier = [2]
        double = lambda x: x * multiplier[0]

        user_ns = {'double': double, 'multiplier': multiplier}
        cp.save('test', user_ns)

        # Modify multiplier
        multiplier[0] = 10
        assert double(5) == 50

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['multiplier'] == [2]
        assert user_ns['double'](5) == 10

    def test_nested_functions_shared_closure(self):
        """Test nested functions that share closure variables."""
        cp = MemoryCheckpoints()

        shared = [0]

        def outer():
            def inner():
                shared[0] += 1
                return shared[0]
            return inner()

        user_ns = {'outer': outer, 'shared': shared}
        cp.save('test', user_ns)

        # Call function
        assert outer() == 1
        assert outer() == 2
        assert shared == [2]

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['shared'] == [0]
        assert user_ns['outer']() == 1

    def test_multiple_functions_same_closure(self):
        """Test multiple functions sharing the same closure variable."""
        cp = MemoryCheckpoints()

        data = [0]

        def increment():
            data[0] += 1
            return data[0]

        def decrement():
            data[0] -= 1
            return data[0]

        def get_value():
            return data[0]

        user_ns = {
            'increment': increment,
            'decrement': decrement,
            'get_value': get_value,
            'data': data
        }
        cp.save('test', user_ns)

        # Modify through functions
        increment()
        increment()
        decrement()
        assert data == [1]

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['data'] == [0]
        assert user_ns['get_value']() == 0

        # All restored functions should share the same restored closure
        user_ns['increment']()
        assert user_ns['get_value']() == 1
        assert user_ns['data'] == [1]


class TestFunctionWithMutableDefaults:
    """Test that mutable default arguments are properly handled."""

    def test_function_with_list_default(self):
        """Test that mutable default arguments are deep copied."""
        cp = MemoryCheckpoints()

        def append_to(item, lst=[]):
            lst.append(item)
            return lst

        user_ns = {'append_to': append_to}
        cp.save('test', user_ns)

        # Modify default
        append_to('a')
        append_to('b')
        assert append_to('c') == ['a', 'b', 'c']

        # Restore
        cp.restore('test', user_ns)

        # Restored function should have empty default
        assert user_ns['append_to']('x') == ['x']

    def test_function_with_dict_default(self):
        """Test that dict default arguments are deep copied."""
        cp = MemoryCheckpoints()

        def update_config(key, value, config={}):
            config[key] = value
            return config

        user_ns = {'update_config': update_config}
        cp.save('test', user_ns)

        # Modify default
        update_config('a', 1)
        update_config('b', 2)
        assert update_config('c', 3) == {'a': 1, 'b': 2, 'c': 3}

        # Restore
        cp.restore('test', user_ns)

        # Restored function should have empty default
        assert user_ns['update_config']('x', 10) == {'x': 10}

    def test_function_with_kwonly_mutable_default(self):
        """Test that keyword-only mutable defaults are deep copied."""
        cp = MemoryCheckpoints()

        def process(*, items=[]):
            items.append(len(items))
            return items

        user_ns = {'process': process}
        cp.save('test', user_ns)

        # Modify kwonly default
        process()
        process()
        assert process() == [0, 1, 2]

        # Restore
        cp.restore('test', user_ns)

        # Should be reset
        assert user_ns['process']() == [0]


class TestFunctionWithoutClosure:
    """Test that functions without closures are handled correctly."""

    def test_simple_function_no_closure(self):
        """Test that simple functions without closures work correctly."""
        cp = MemoryCheckpoints()

        def add(a, b):
            return a + b

        user_ns = {'add': add}
        cp.save('test', user_ns)
        cp.restore('test', user_ns)

        assert user_ns['add'](2, 3) == 5

    def test_function_with_immutable_defaults(self):
        """Test that functions with immutable defaults are handled correctly."""
        cp = MemoryCheckpoints()

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        user_ns = {'greet': greet}
        cp.save('test', user_ns)
        cp.restore('test', user_ns)

        assert user_ns['greet']("World") == "Hello, World!"
        assert user_ns['greet']("World", "Hi") == "Hi, World!"


class TestMultipleRestoresWithFunctions:
    """Test that multiple restores don't corrupt function checkpoints."""

    def test_multiple_restores_preserve_closure(self):
        """Test that restoring multiple times keeps closure pristine."""
        cp = MemoryCheckpoints()

        counter = [0]
        def inc():
            counter[0] += 1
            return counter[0]

        user_ns = {'inc': inc, 'counter': counter}
        cp.save('test', user_ns)

        # Multiple restore cycles
        for i in range(3):
            cp.restore('test', user_ns)
            # Each restore should give us counter at [0]
            result = user_ns['inc']()
            assert result == 1, f"Iteration {i}: expected 1, got {result}"
            # Call again to verify it's incrementing the restored copy
            result2 = user_ns['inc']()
            assert result2 == 2, f"Iteration {i}: expected 2, got {result2}"

        # Final restore should still work
        cp.restore('test', user_ns)
        assert user_ns['counter'] == [0]

    def test_multiple_restores_with_mutable_default(self):
        """Test multiple restores with mutable default arguments."""
        cp = MemoryCheckpoints()

        def accumulate(val, acc=[]):
            acc.append(val)
            return acc.copy()

        user_ns = {'accumulate': accumulate}
        cp.save('test', user_ns)

        for i in range(3):
            cp.restore('test', user_ns)
            # Each restore should give us empty default
            result = user_ns['accumulate']('a')
            assert result == ['a'], f"Iteration {i}: expected ['a'], got {result}"


class TestDeepCopyFunctionDirectly:
    """Test the _deep_copy_function helper directly."""

    def test_copy_function_with_closure(self):
        """Test direct use of _deep_copy_function."""
        data = [1, 2, 3]
        def get_data():
            return data.copy()

        memo = {}
        copied_func = _deep_copy_function(get_data, memo)

        # Modify original
        data.append(4)
        assert get_data() == [1, 2, 3, 4]

        # Copied function should have isolated closure
        assert copied_func() == [1, 2, 3]

    def test_copy_function_without_closure(self):
        """Test _deep_copy_function with function that has no closure."""
        def add(a, b):
            return a + b

        memo = {}
        copied_func = _deep_copy_function(add, memo)

        assert copied_func(2, 3) == 5
        # No closure and no mutable defaults, should return same function
        assert copied_func is add

    def test_copy_function_with_mutable_default_no_closure(self):
        """Test _deep_copy_function with mutable default but no closure."""
        def append(item, lst=[]):
            lst.append(item)
            return lst

        memo = {}
        copied_func = _deep_copy_function(append, memo)

        # Should NOT be the same function (has mutable default)
        assert copied_func is not append

        # Original should have its own default
        append('a')
        assert append('b') == ['a', 'b']

        # Copied should have isolated default
        assert copied_func('x') == ['x']

    def test_copy_function_preserves_attributes(self):
        """Test that function attributes are preserved."""
        data = []
        def func():
            return data

        func.__doc__ = "Test docstring"
        func.custom_attr = "custom_value"

        memo = {}
        copied_func = _deep_copy_function(func, memo)

        assert copied_func.__doc__ == "Test docstring"
        assert copied_func.custom_attr == "custom_value"
        assert copied_func.__name__ == "func"

    def test_copy_function_memo_tracking(self):
        """Test that copied functions are tracked in memo."""
        data = [1]
        def func():
            return data

        memo = {}
        copied_func = _deep_copy_function(func, memo)

        # Original function's id should be in memo
        assert id(func) in memo
        assert memo[id(func)] is copied_func


class TestFunctionWithComplexClosures:
    """Test functions with more complex closure scenarios."""

    def test_closure_with_numpy_array(self):
        """Test closure containing numpy array."""
        cp = MemoryCheckpoints()

        arr = np.array([1, 2, 3])
        def get_sum():
            return arr.sum()

        user_ns = {'arr': arr, 'get_sum': get_sum}
        cp.save('test', user_ns)

        # Modify array
        arr[0] = 100
        assert get_sum() == 105

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['get_sum']() == 6

    def test_closure_with_dataframe(self):
        """Test closure containing pandas DataFrame."""
        cp = MemoryCheckpoints()

        df = pd.DataFrame({'a': [1, 2, 3]})
        def get_mean():
            return df['a'].mean()

        user_ns = {'df': df, 'get_mean': get_mean}
        cp.save('test', user_ns)

        # Modify dataframe
        df['a'] = [10, 20, 30]
        assert get_mean() == 20.0

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['get_mean']() == 2.0

    def test_closure_with_custom_object(self):
        """Test closure containing custom class instance."""
        cp = MemoryCheckpoints()

        class Counter:
            def __init__(self):
                self.value = 0
            def increment(self):
                self.value += 1
                return self.value

        counter = Counter()
        def inc():
            return counter.increment()

        user_ns = {'counter': counter, 'inc': inc}
        cp.save('test', user_ns)

        # Use counter
        inc()
        inc()
        assert counter.value == 2

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['counter'].value == 0
        assert user_ns['inc']() == 1

    def test_closure_with_nested_mutable_structures(self):
        """Test closure with deeply nested mutable structures."""
        cp = MemoryCheckpoints()

        nested = {'level1': {'level2': [1, 2, 3]}}
        def get_nested():
            return nested['level1']['level2'].copy()

        user_ns = {'nested': nested, 'get_nested': get_nested}
        cp.save('test', user_ns)

        # Modify deeply
        nested['level1']['level2'].append(4)
        assert get_nested() == [1, 2, 3, 4]

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['get_nested']() == [1, 2, 3]


class TestGlobalReferencesInFunctions:
    """Test functions that reference globals (not closures) work correctly."""

    def test_function_referencing_global_variable(self):
        """Test that functions referencing globals see restored values after restore."""
        cp = MemoryCheckpoints()

        # Simulate Jupyter-style execution where __globals__ IS user_ns
        user_ns = {}
        exec("""
data = [1, 2, 3]
def get_sum():
    return sum(data)  # 'data' looked up in __globals__
""", user_ns)

        cp.save('test', user_ns)

        # Modify global
        user_ns['data'].append(100)
        assert user_ns['get_sum']() == 106

        # Restore
        cp.restore('test', user_ns)

        # Function should see restored value because __globals__ IS user_ns
        assert user_ns['data'] == [1, 2, 3]
        assert user_ns['get_sum']() == 6

    def test_recursive_function_via_globals(self):
        """Test recursive function where recursion uses __globals__ lookup."""
        cp = MemoryCheckpoints()

        user_ns = {}
        exec("""
counter = [0]
def factorial(n):
    counter[0] += 1
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # factorial looked up in __globals__
""", user_ns)

        cp.save('test', user_ns)

        # Use original
        user_ns['factorial'](5)
        assert user_ns['counter'] == [5]

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['counter'] == [0]

        # Recursive calls should use the RESTORED function
        result = user_ns['factorial'](4)
        assert result == 24
        assert user_ns['counter'] == [4]

    def test_function_modifying_global_after_restore(self):
        """Test that restored function can modify restored globals."""
        cp = MemoryCheckpoints()

        user_ns = {}
        exec("""
state = {'count': 0}
def increment():
    state['count'] += 1
    return state['count']
""", user_ns)

        cp.save('test', user_ns)

        # Modify
        user_ns['increment']()
        user_ns['increment']()
        assert user_ns['state']['count'] == 2

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['state']['count'] == 0

        # Restored function should modify restored state
        assert user_ns['increment']() == 1
        assert user_ns['state']['count'] == 1


class TestEdgeCases:
    """Test edge cases in function checkpointing."""

    def test_recursive_function_with_closure(self):
        """Test recursive function that uses closure."""
        cp = MemoryCheckpoints()

        call_count = [0]
        def factorial(n):
            call_count[0] += 1
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        user_ns = {'factorial': factorial, 'call_count': call_count}
        cp.save('test', user_ns)

        # Call factorial
        result = factorial(5)
        assert result == 120
        assert call_count[0] == 5

        # Restore
        cp.restore('test', user_ns)
        assert user_ns['call_count'] == [0]
        # Note: The restored factorial still references the original factorial
        # because __globals__ is shared, but the call_count should be isolated
        result = user_ns['factorial'](3)
        assert result == 6

    def test_function_referencing_itself_in_closure(self):
        """Test function that captures itself in closure (through variable)."""
        cp = MemoryCheckpoints()

        log = []
        def func():
            log.append('called')
            return len(log)

        user_ns = {'func': func, 'log': log}
        cp.save('test', user_ns)

        func()
        func()
        assert log == ['called', 'called']

        cp.restore('test', user_ns)
        assert user_ns['log'] == []
        assert user_ns['func']() == 1

    def test_generator_function_with_closure(self):
        """Test generator function with closure."""
        cp = MemoryCheckpoints()

        data = [1, 2, 3, 4, 5]
        def gen():
            for item in data:
                yield item * 2

        user_ns = {'gen': gen, 'data': data}
        cp.save('test', user_ns)

        # Modify data
        data.extend([6, 7])

        # Restore
        cp.restore('test', user_ns)
        assert list(user_ns['gen']()) == [2, 4, 6, 8, 10]

    def test_method_bound_to_object_with_closure(self):
        """Test that bound methods work correctly after restore."""
        cp = MemoryCheckpoints()

        class MyClass:
            def __init__(self, value):
                self.value = value
            def get_value(self):
                return self.value

        obj = MyClass(42)
        method = obj.get_value

        user_ns = {'obj': obj, 'method': method}
        cp.save('test', user_ns)

        obj.value = 100
        assert method() == 100

        cp.restore('test', user_ns)
        assert user_ns['obj'].value == 42
        # Note: method is bound to original obj, but obj was also restored
        # The bound method's __self__ should be deep copied


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
