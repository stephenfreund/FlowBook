"""Tests for HeapSizer memory measurement."""

import sys
import pytest
import numpy as np
import pandas as pd

from flowbook.kernel_support.heap_size import HeapSizer, sizeof, NamespaceSize


class TestNumPyArrays:
    """Tests for numpy array memory measurement."""

    def test_basic_array_size(self):
        """Test that basic array size is accurate."""
        arr = np.zeros(1_000_000)  # 8 MB
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        # Should be close to 8MB + overhead
        assert size > 8_000_000
        assert size < 9_000_000

    def test_view_not_double_counted(self):
        """Views should only count wrapper, not data."""
        arr = np.zeros(1_000_000)  # 8 MB
        view = arr[::2]  # View of half the array

        sizer = HeapSizer()
        arr_size = sizer.sizeof(arr)
        assert arr_size > 8_000_000

        sizer.reset()
        view_size = sizer.sizeof(view, owned_only=True)
        assert view_size < 200  # Just wrapper

    def test_view_counted_when_owned_only_false(self):
        """Views should count data when owned_only=False."""
        arr = np.zeros(1_000_000)
        view = arr[::2]

        sizer = HeapSizer()
        view_size = sizer.sizeof(view, owned_only=False)
        # Should count the full underlying buffer
        assert view_size > 8_000_000

    def test_object_array_traversed(self):
        """Object arrays should traverse and measure elements."""
        arr = np.array([{'a': 1, 'b': 2}, [1, 2, 3, 4, 5], "hello"], dtype=object)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        # Should be more than just 3 pointers
        assert size > 100

    def test_large_object_array(self):
        """Large object arrays should measure all elements."""
        data = [{'key': i, 'value': list(range(10))} for i in range(100)]
        arr = np.array(data, dtype=object)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        # Should be substantial due to dict/list contents
        assert size > 10_000

    def test_shared_data_pointer_deduplication(self):
        """Arrays sharing data buffer should deduplicate."""
        arr = np.zeros(1_000_000)
        # Create another reference to same data
        arr2 = arr.view()
        arr2.flags.writeable = False

        sizer = HeapSizer()
        # Measure both
        ns_size = sizer.sizeof_namespace({'arr': arr, 'arr2': arr2})
        # Should count data once, not twice
        assert ns_size.total_bytes < 10_000_000


class TestPandasDataFrame:
    """Tests for pandas DataFrame memory measurement."""

    def test_basic_dataframe(self):
        """Test basic DataFrame size."""
        df = pd.DataFrame({'a': np.zeros(100_000), 'b': np.ones(100_000)})
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        # Two columns of 100k floats = ~1.6MB
        assert size > 1_500_000
        assert size < 2_500_000

    def test_same_dataframe_not_double_counted(self):
        """Same DataFrame referenced by multiple variables should count once."""
        df = pd.DataFrame({'a': np.zeros(1_000_000)})
        # Two variables referencing the same DataFrame
        df1 = df
        df2 = df

        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace({'df1': df1, 'df2': df2})
        # Should count data once, not twice (under 12MB for ~8MB of data)
        assert ns_size.total_bytes < 12_000_000

    def test_object_column_traversed(self):
        """Object columns should have elements measured."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [{'x': 1}, {'y': 2}, {'z': 3}]  # Object column
        })
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        # Should be more than just array overhead
        assert size > 500

    def test_multiindex_columns(self):
        """MultiIndex columns should be handled."""
        arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame(np.random.randn(3, 4), columns=index)

        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 0


class TestPandasSeries:
    """Tests for pandas Series memory measurement."""

    def test_basic_series(self):
        """Test basic Series size."""
        s = pd.Series(np.zeros(100_000))
        sizer = HeapSizer()
        size = sizer.sizeof(s)
        assert size > 800_000  # ~0.8MB for 100k floats

    def test_object_series(self):
        """Object series should traverse elements."""
        s = pd.Series([{'a': 1}, [1, 2, 3], 'hello'])
        sizer = HeapSizer()
        size = sizer.sizeof(s)
        assert size > 100


class TestContainers:
    """Tests for container memory measurement."""

    def test_list_with_elements(self):
        """List should include element sizes."""
        lst = [np.zeros(10_000) for _ in range(10)]
        sizer = HeapSizer()
        size = sizer.sizeof(lst)
        # 10 arrays of 80KB each = ~800KB
        assert size > 700_000

    def test_nested_dict(self):
        """Nested dict should measure all levels."""
        d = {
            'level1': {
                'level2': {
                    'data': np.zeros(10_000)
                }
            }
        }
        sizer = HeapSizer()
        size = sizer.sizeof(d)
        assert size > 80_000

    def test_shared_list_counted_once(self):
        """Shared list should be counted once."""
        shared = [1, 2, 3, 4, 5] * 1000  # Large list
        a = {'x': shared}
        b = {'y': shared}

        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace({'a': a, 'b': b})
        # shared list counted once, not twice
        assert ns_size.total_bytes < sys.getsizeof(shared) * 2

    def test_tuple_immutable_traversed(self):
        """Tuples should still traverse mutable elements."""
        arr = np.zeros(10_000)
        t = (arr, arr)  # Same array twice

        sizer = HeapSizer()
        size = sizer.sizeof(t)
        # Array should be counted once
        assert size < 100_000


class TestFunctions:
    """Tests for function memory measurement."""

    def test_function_closure_measured(self):
        """Function closure should be measured."""
        large_data = list(range(10_000))

        def f():
            return large_data

        sizer = HeapSizer()
        size = sizer.sizeof(f)
        # Should include closure contents
        assert size > sys.getsizeof(f)
        assert size > 100_000  # list(range(10000)) is substantial

    def test_function_defaults_measured(self):
        """Function default arguments should be measured."""
        large_default = list(range(1000))

        def f(x=large_default):
            return x

        sizer = HeapSizer()
        size = sizer.sizeof(f)
        assert size > sys.getsizeof(f)


class TestNamespace:
    """Tests for namespace measurement."""

    def test_basic_namespace(self):
        """Test basic namespace measurement."""
        ns = {
            'arr': np.zeros(100_000),
            'df': pd.DataFrame({'a': [1, 2, 3]}),
            'x': 42,
        }
        sizer = HeapSizer()
        result = sizer.sizeof_namespace(ns)

        assert isinstance(result, NamespaceSize)
        assert result.total_bytes > 800_000
        assert 'arr' in result.by_variable
        assert result.by_variable['arr'] > 800_000
        assert 'ndarray' in result.by_type

    def test_include_filter(self):
        """Test include filter."""
        ns = {'a': np.zeros(100_000), 'b': np.zeros(100_000)}
        sizer = HeapSizer()
        result = sizer.sizeof_namespace(ns, include={'a'})

        assert 'a' in result.by_variable
        assert 'b' not in result.by_variable

    def test_exclude_filter(self):
        """Test exclude filter."""
        ns = {'a': np.zeros(100_000), 'b': np.zeros(100_000)}
        sizer = HeapSizer()
        result = sizer.sizeof_namespace(ns, exclude={'b'})

        assert 'a' in result.by_variable
        assert 'b' not in result.by_variable


class TestEdgeCases:
    """Tests for edge cases."""

    def test_none(self):
        """None should return 0."""
        sizer = HeapSizer()
        assert sizer.sizeof(None) == 0

    def test_empty_containers(self):
        """Empty containers should have minimal size."""
        sizer = HeapSizer()
        assert sizer.sizeof([]) < 100
        assert sizer.sizeof({}) < 300
        assert sizer.sizeof(()) < 100

    def test_circular_reference(self):
        """Circular references should not cause infinite loop."""
        a = {'self': None}
        a['self'] = a

        sizer = HeapSizer()
        size = sizer.sizeof(a)
        assert size > 0
        assert size < 10_000  # Should not explode

    def test_deep_nesting(self):
        """Deep nesting should not cause stack overflow."""
        obj = {'data': 1}
        for _ in range(100):
            obj = {'nested': obj}

        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        assert size > 0

    def test_convenience_function(self):
        """Test sizeof convenience function."""
        arr = np.zeros(1000)
        size = sizeof(arr)
        assert size > 8000


class TestAccuracy:
    """Tests for measurement accuracy."""

    def test_accuracy_within_10_percent(self):
        """Memory measurements should be within 10% of actual."""
        # Create array with known size
        arr = np.zeros(10_000_000)  # 80 MB exactly
        expected = 80_000_000

        sizer = HeapSizer()
        measured = sizer.sizeof(arr)

        # Allow for wrapper overhead
        assert abs(measured - expected) / expected < 0.10

    def test_dataframe_accuracy(self):
        """DataFrame measurement should be reasonably accurate."""
        # Create DataFrame with known column sizes
        n = 100_000
        df = pd.DataFrame({
            'a': np.zeros(n),  # 800KB
            'b': np.ones(n),   # 800KB
        })
        expected_min = 1_500_000  # At least 1.5MB

        sizer = HeapSizer()
        measured = sizer.sizeof(df)

        assert measured > expected_min


class TestNumPyDtypes:
    """Tests for various numpy dtypes."""

    def test_int8_array(self):
        """Int8 arrays should measure 1 byte per element."""
        arr = np.zeros(1_000_000, dtype=np.int8)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 1_000_000
        assert size < 1_100_000

    def test_int16_array(self):
        """Int16 arrays should measure 2 bytes per element."""
        arr = np.zeros(1_000_000, dtype=np.int16)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 2_000_000
        assert size < 2_100_000

    def test_int32_array(self):
        """Int32 arrays should measure 4 bytes per element."""
        arr = np.zeros(1_000_000, dtype=np.int32)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 4_000_000
        assert size < 4_100_000

    def test_float32_array(self):
        """Float32 arrays should measure 4 bytes per element."""
        arr = np.zeros(1_000_000, dtype=np.float32)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 4_000_000
        assert size < 4_100_000

    def test_complex128_array(self):
        """Complex128 arrays should measure 16 bytes per element."""
        arr = np.zeros(100_000, dtype=np.complex128)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 1_600_000
        assert size < 1_700_000

    def test_bool_array(self):
        """Bool arrays should measure 1 byte per element."""
        arr = np.zeros(1_000_000, dtype=bool)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 1_000_000
        assert size < 1_100_000

    def test_structured_array(self):
        """Structured arrays should handle compound dtypes."""
        dt = np.dtype([('x', np.float64), ('y', np.float64), ('label', 'U10')])
        arr = np.zeros(10_000, dtype=dt)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        # Each element is 8 + 8 + 40 = 56 bytes
        assert size > 500_000

    def test_string_array(self):
        """String (unicode) arrays should be measured."""
        arr = np.array(['hello', 'world', 'test'] * 1000)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 0

    def test_bytes_array(self):
        """Bytes arrays should be measured."""
        arr = np.array([b'hello', b'world', b'test'] * 1000)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 0


class TestNumPyShapes:
    """Tests for various numpy array shapes."""

    def test_1d_array(self):
        """1D arrays should work."""
        arr = np.zeros(10_000)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 80_000

    def test_2d_array(self):
        """2D arrays should work."""
        arr = np.zeros((100, 100))
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 80_000

    def test_3d_array(self):
        """3D arrays should work."""
        arr = np.zeros((10, 10, 100))
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 80_000

    def test_empty_array(self):
        """Empty arrays should have minimal size."""
        arr = np.array([])
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size < 200

    def test_scalar_array(self):
        """0-d scalar arrays should work."""
        arr = np.array(42.0)
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 0
        assert size < 200

    def test_fortran_order_array(self):
        """Fortran-ordered arrays should work."""
        arr = np.asfortranarray(np.zeros((100, 100)))
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 80_000

    def test_non_contiguous_array(self):
        """Non-contiguous arrays should work."""
        arr = np.zeros((100, 100))
        non_contig = arr[::2, ::2]  # Every other element
        sizer = HeapSizer()
        # This is a view
        size = sizer.sizeof(non_contig, owned_only=True)
        assert size < 200


class TestNumPyAdvanced:
    """Tests for advanced numpy features."""

    def test_masked_array(self):
        """Masked arrays should be measured."""
        arr = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 0

    def test_matrix(self):
        """Matrix objects should be measured."""
        mat = np.matrix([[1, 2], [3, 4]])
        sizer = HeapSizer()
        size = sizer.sizeof(mat)
        assert size > 0

    def test_recarray(self):
        """Record arrays should be measured."""
        arr = np.rec.array([(1, 2.0, 'Hello'), (2, 3.0, 'World')],
                           dtype=[('a', int), ('b', float), ('c', 'U10')])
        sizer = HeapSizer()
        size = sizer.sizeof(arr)
        assert size > 0

    def test_memmap_like(self):
        """Memory-mapped-like arrays (views) should be handled."""
        # Create a large array and a view that simulates mmap behavior
        base = np.zeros(1_000_000)
        view = base[:]
        view.flags.writeable = False

        sizer = HeapSizer()
        # View should be small since it doesn't own data
        size = sizer.sizeof(view, owned_only=True)
        assert size < 200


class TestPandasAdvanced:
    """Tests for advanced pandas features."""

    def test_categorical_column(self):
        """Categorical columns should be measured."""
        df = pd.DataFrame({
            'cat': pd.Categorical(['a', 'b', 'c'] * 10000)
        })
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 0
        # Categorical should be smaller than object type
        assert size < 1_000_000

    def test_datetime_column(self):
        """Datetime columns should be measured."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10000)
        })
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 80_000  # 8 bytes per datetime

    def test_timedelta_column(self):
        """Timedelta columns should be measured."""
        df = pd.DataFrame({
            'td': pd.to_timedelta(range(10000), unit='D')
        })
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 80_000

    def test_nullable_int(self):
        """Nullable integer columns should be measured."""
        df = pd.DataFrame({
            'nullable': pd.array([1, 2, None, 4, 5] * 2000, dtype=pd.Int64Dtype())
        })
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 0

    def test_string_dtype(self):
        """String dtype columns should be measured."""
        df = pd.DataFrame({
            'str': pd.array(['hello', 'world', None] * 1000, dtype=pd.StringDtype())
        })
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 0

    def test_multiindex_rows(self):
        """MultiIndex on rows should be measured."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(np.random.randn(4, 3), index=index)
        sizer = HeapSizer()
        size = sizer.sizeof(df)
        assert size > 0

    def test_sparse_series(self):
        """Sparse series should be measured."""
        s = pd.arrays.SparseArray([0, 0, 1, 0, 0, 0, 0, 2, 0, 0])
        sizer = HeapSizer()
        size = sizer.sizeof(s)
        assert size > 0

    def test_period_index(self):
        """Period index should be measured."""
        idx = pd.period_range('2020-01', periods=100, freq='M')
        sizer = HeapSizer()
        size = sizer.sizeof(idx)
        assert size > 0

    def test_interval_index(self):
        """Interval index should be measured."""
        idx = pd.IntervalIndex.from_breaks(range(101))
        sizer = HeapSizer()
        size = sizer.sizeof(idx)
        assert size > 0


class TestClassInstances:
    """Tests for class instances."""

    def test_slots_class(self):
        """Classes with __slots__ should be measured."""
        class SlotClass:
            __slots__ = ['x', 'y', 'data']
            def __init__(self):
                self.x = 1
                self.y = 2
                self.data = np.zeros(1000)

        obj = SlotClass()
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        assert size > 8000  # Should include data array

    def test_dict_class(self):
        """Classes with __dict__ should be measured."""
        class DictClass:
            def __init__(self):
                self.x = 1
                self.y = 2
                self.data = np.zeros(1000)

        obj = DictClass()
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        assert size > 8000  # Should include data array

    def test_nested_objects(self):
        """Nested custom objects should be measured."""
        class Node:
            def __init__(self, value, left=None, right=None):
                self.value = value
                self.left = left
                self.right = right

        # Build a small tree
        tree = Node(1,
            Node(2, Node(4), Node(5)),
            Node(3, Node(6), Node(7))
        )

        sizer = HeapSizer()
        size = sizer.sizeof(tree)
        assert size > 0

    def test_object_with_custom_sizeof(self):
        """Objects with custom __sizeof__ should use it."""
        class CustomSize:
            def __sizeof__(self):
                return 12345

        obj = CustomSize()
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        # Should be at least the custom size
        assert size >= 12345


class TestPrimitives:
    """Tests for primitive types."""

    def test_int(self):
        """Integers should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof(42)
        assert size > 0

    def test_large_int(self):
        """Large integers should have larger size."""
        sizer = HeapSizer()
        small_size = sizer.sizeof(42)
        sizer.reset()
        large_size = sizer.sizeof(10**100)
        assert large_size > small_size

    def test_float(self):
        """Floats should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof(3.14159)
        assert size > 0

    def test_complex(self):
        """Complex numbers should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof(complex(1, 2))
        assert size > 0

    def test_string(self):
        """Strings should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof("hello world")
        assert size > 0

    def test_large_string(self):
        """Large strings should have proportionally larger size."""
        sizer = HeapSizer()
        small_size = sizer.sizeof("x")
        sizer.reset()
        large_size = sizer.sizeof("x" * 10000)
        assert large_size > small_size * 100

    def test_bytes(self):
        """Bytes should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof(b"hello world")
        assert size > 0

    def test_bytearray(self):
        """Bytearrays should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof(bytearray(10000))
        assert size > 10000

    def test_bool(self):
        """Booleans should have a size."""
        sizer = HeapSizer()
        size = sizer.sizeof(True)
        assert size > 0


class TestMoreContainers:
    """Additional container tests."""

    def test_frozenset(self):
        """Frozensets should be measured."""
        fs = frozenset(range(1000))
        sizer = HeapSizer()
        size = sizer.sizeof(fs)
        assert size > 0

    def test_set_with_objects(self):
        """Sets with object elements should be measured."""
        s = {(i, i*2) for i in range(1000)}
        sizer = HeapSizer()
        size = sizer.sizeof(s)
        assert size > 0

    def test_dict_with_complex_keys(self):
        """Dicts with complex keys should be measured."""
        d = {(i, i*2): [i, i+1, i+2] for i in range(100)}
        sizer = HeapSizer()
        size = sizer.sizeof(d)
        assert size > 0

    def test_nested_lists(self):
        """Deeply nested lists should be measured."""
        lst = [[[[i for i in range(10)] for _ in range(10)] for _ in range(10)] for _ in range(10)]
        sizer = HeapSizer()
        size = sizer.sizeof(lst)
        assert size > 0

    def test_mixed_container(self):
        """Mixed containers should be measured."""
        obj = {
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'set': {1, 2, 3},
            'tuple': (1, 2, 3),
            'array': np.zeros(100),
            'df': pd.DataFrame({'x': [1, 2, 3]}),
        }
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        assert size > 0

    def test_empty_nested(self):
        """Empty nested containers should work."""
        obj = {'a': [], 'b': {}, 'c': (), 'd': set()}
        sizer = HeapSizer()
        size = sizer.sizeof(obj)
        assert size > 0


class TestCallables:
    """Tests for callable objects."""

    def test_lambda(self):
        """Lambda functions should be measured."""
        large_list = list(range(1000))
        f = lambda: large_list  # noqa: E731

        sizer = HeapSizer()
        size = sizer.sizeof(f)
        assert size > sys.getsizeof(f)

    def test_nested_closure(self):
        """Nested closures should be measured."""
        def outer():
            x = list(range(1000))
            def inner():
                y = list(range(500))
                def innermost():
                    return x, y
                return innermost
            return inner

        f = outer()()
        sizer = HeapSizer()
        size = sizer.sizeof(f)
        assert size > 0

    def test_method(self):
        """Bound methods should be measured."""
        class MyClass:
            def __init__(self):
                self.data = list(range(1000))
            def method(self):
                return self.data

        obj = MyClass()
        method = obj.method

        sizer = HeapSizer()
        size = sizer.sizeof(method)
        assert size > 0

    def test_builtin_function(self):
        """Built-in functions should not crash."""
        sizer = HeapSizer()
        size = sizer.sizeof(len)
        assert size >= 0


class TestCircularReferences:
    """Tests for circular reference handling."""

    def test_self_referential_list(self):
        """Self-referential list should not cause infinite loop."""
        lst = [1, 2, 3]
        lst.append(lst)

        sizer = HeapSizer()
        size = sizer.sizeof(lst)
        assert size > 0
        assert size < 10_000

    def test_mutual_reference(self):
        """Mutually referential objects should be handled."""
        a = {'b': None}
        b = {'a': a}
        a['b'] = b

        sizer = HeapSizer()
        size = sizer.sizeof(a)
        assert size > 0
        assert size < 10_000

    def test_class_circular(self):
        """Circular class references should be handled."""
        class Node:
            def __init__(self):
                self.next = None
                self.prev = None

        a = Node()
        b = Node()
        a.next = b
        b.prev = a

        sizer = HeapSizer()
        size = sizer.sizeof(a)
        assert size > 0

    def test_complex_graph(self):
        """Complex object graph should be handled."""
        # Create a web of interconnected objects
        nodes = [{'id': i, 'connections': []} for i in range(10)]
        for i, node in enumerate(nodes):
            node['connections'] = [nodes[(i+1) % 10], nodes[(i+2) % 10]]

        sizer = HeapSizer()
        size = sizer.sizeof(nodes)
        assert size > 0


class TestCheckpointSize:
    """Tests for checkpoint size measurement."""

    def test_checkpoint_excludes_cached(self):
        """Checkpoint measurement should exclude cached objects by default."""
        from flowbook.kernel_support.heap_size import CheckpointSize

        # Create a mock checkpoint
        class MockCheckpoint:
            def __init__(self):
                self.user_ns = {'arr': np.zeros(1000)}

        ckpt = MockCheckpoint()
        sizer = HeapSizer()
        result = sizer.sizeof_checkpoint(ckpt)

        assert isinstance(result, CheckpointSize)
        assert result.total_bytes > 0
        assert 'arr' in result.by_variable

    def test_checkpoint_empty(self):
        """Empty checkpoint should return zeros."""
        from flowbook.kernel_support.heap_size import CheckpointSize

        class MockCheckpoint:
            pass

        ckpt = MockCheckpoint()
        sizer = HeapSizer()
        result = sizer.sizeof_checkpoint(ckpt)

        assert result.total_bytes == 0


class TestReset:
    """Tests for HeapSizer reset functionality."""

    def test_reset_clears_seen(self):
        """Reset should clear seen IDs."""
        arr = np.zeros(1000)
        sizer = HeapSizer()

        size1 = sizer.sizeof(arr)
        assert size1 > 8000

        # Without reset, same object returns 0
        size2 = sizer.sizeof(arr)
        assert size2 == 0

        # After reset, object is measured again
        sizer.reset()
        size3 = sizer.sizeof(arr)
        assert size3 > 8000

    def test_fresh_sizer_each_call(self):
        """Each HeapSizer instance should start fresh."""
        arr = np.zeros(1000)

        sizer1 = HeapSizer()
        size1 = sizer1.sizeof(arr)

        sizer2 = HeapSizer()
        size2 = sizer2.sizeof(arr)

        assert size1 == size2


class TestOwnershipTracking:
    """Tests for ownership-aware measurement."""

    def test_owned_only_true(self):
        """owned_only=True should not count views."""
        base = np.zeros(10000)
        view = base[1000:5000]

        sizer = HeapSizer()
        view_size = sizer.sizeof(view, owned_only=True)
        assert view_size < 200  # Just wrapper

    def test_owned_only_false(self):
        """owned_only=False should follow to base."""
        base = np.zeros(10000)
        view = base[1000:5000]

        sizer = HeapSizer()
        view_size = sizer.sizeof(view, owned_only=False)
        assert view_size > 80000  # Full base array

    def test_chained_views(self):
        """Chained views should follow to ultimate base."""
        base = np.zeros(10000)
        view1 = base[::2]
        view2 = view1[::2]

        sizer = HeapSizer()
        size = sizer.sizeof(view2, owned_only=False)
        assert size > 80000  # Should measure full base


class TestDeduplication:
    """Tests for object deduplication."""

    def test_same_array_in_multiple_vars(self):
        """Same array in multiple vars counted once."""
        arr = np.zeros(10000)
        ns = {'a': arr, 'b': arr, 'c': arr}

        sizer = HeapSizer()
        result = sizer.sizeof_namespace(ns)

        # Should be about 80KB, not 240KB
        assert result.total_bytes < 100_000

    def test_shared_list_elements(self):
        """Shared list elements counted once."""
        shared = list(range(1000))
        container = [shared, shared, shared]

        sizer = HeapSizer()
        size = sizer.sizeof(container)

        # Should not be 3x the shared list size
        single_size = sizer.reset() or HeapSizer().sizeof(shared)
        assert size < single_size * 2

    def test_dataframe_shares_memory(self):
        """DataFrames sharing numpy buffer deduped via shares_memory."""
        arr = np.zeros(10000)
        # Create view that might share memory
        view = arr.view()

        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace({'arr': arr, 'view': view})

        # Should count buffer once
        assert ns_size.total_bytes < 100_000


class TestExtensionArrayDeduplication:
    """Tests for ExtensionArray (StringDtype, etc.) memory deduplication.

    This is a regression test for an issue where StringDtype columns with
    pd.options.future.infer_string = True caused memory overhead to grow
    linearly across checkpoints instead of deduplicating shared arrays.
    """

    def test_string_array_identity_tracking(self):
        """StringArray should be tracked by identity to avoid double-counting."""
        arr = pd.array(['hello', 'world', 'test'] * 1000, dtype='string')

        sizer = HeapSizer()
        size1 = sizer.sizeof(arr)
        size2 = sizer.sizeof(arr)

        # First measurement should be non-zero
        assert size1 > 0
        # Second measurement of same object should be 0 (already counted)
        assert size2 == 0

    def test_string_array_underlying_ndarray_tracked(self):
        """StringArray's underlying _ndarray should be tracked for dedup."""
        arr = pd.array(['hello', 'world'] * 1000, dtype='string')

        # Verify it has _ndarray attribute
        assert hasattr(arr, '_ndarray'), "StringArray should have _ndarray"

        sizer = HeapSizer()
        size1 = sizer.sizeof(arr)

        # Create a new wrapper around the same _ndarray (simulates CoW sharing)
        # This is what happens when df.copy(deep=False) is used
        arr2 = pd.array(['different'], dtype='string')  # Just to get the type
        # Now measure the underlying array directly
        sizer2 = HeapSizer()
        underlying_size = sizer2.sizeof(arr._ndarray)
        underlying_size2 = sizer2.sizeof(arr._ndarray)

        assert underlying_size > 0
        assert underlying_size2 == 0, "Same _ndarray should be deduplicated"

    def test_dataframe_stringdtype_column_dedup(self):
        """DataFrame with StringDtype columns should deduplicate across measurements."""
        df = pd.DataFrame({
            'name': pd.array(['Alice', 'Bob', 'Charlie'] * 1000, dtype='string'),
            'city': pd.array(['NYC', 'LA', 'Chicago'] * 1000, dtype='string'),
        })

        sizer = HeapSizer()
        size1 = sizer.sizeof(df)
        size2 = sizer.sizeof(df)

        assert size1 > 1000  # Meaningful size
        assert size2 == 0, "Same DataFrame should return 0 on second measurement"

    def test_dataframe_copy_shares_string_column_data(self):
        """DataFrame.copy(deep=False) should share StringDtype column data."""
        from flowbook.kernel_support.deepcopy import deepcopy

        df = pd.DataFrame({
            'name': pd.array(['Alice'] * 5000, dtype='string'),
        })

        # Simulate checkpoint deepcopy
        df_copy = deepcopy(df, {})

        # The underlying _ndarray should be shared
        orig_arr = df._mgr.arrays[0]
        copy_arr = df_copy._mgr.arrays[0]

        assert hasattr(orig_arr, '_ndarray'), "Expected StringArray with _ndarray"
        assert orig_arr._ndarray is copy_arr._ndarray, \
            "Deepcopy should share underlying _ndarray for StringArray"

    def test_checkpoint_deduplication_stringdtype(self):
        """Cross-checkpoint measurement should deduplicate shared StringDtype arrays."""
        from flowbook.kernel_support.deepcopy import deepcopy

        df = pd.DataFrame({
            'name': pd.array(['Alice'] * 10000, dtype='string'),
            'city': pd.array(['NYC'] * 10000, dtype='string'),
        })

        # Create multiple "checkpoints" via deepcopy
        df_copy1 = deepcopy(df, {})
        df_copy2 = deepcopy(df, {})

        # Measure all with a single sizer (simulating cross-checkpoint measurement)
        sizer = HeapSizer()
        size_orig = sizer.sizeof(df)
        size_copy1 = sizer.sizeof(df_copy1)
        size_copy2 = sizer.sizeof(df_copy2)

        # Original should have full size
        assert size_orig > 10000, f"Original should be substantial, got {size_orig}"

        # Copies should have minimal overhead (wrapper only) since data is shared
        assert size_copy1 < size_orig * 0.1, \
            f"Copy1 should be much smaller ({size_copy1}) than original ({size_orig})"
        assert size_copy2 < size_orig * 0.1, \
            f"Copy2 should be much smaller ({size_copy2}) than original ({size_orig})"

        # Total should be much less than 3x the original size
        total = size_orig + size_copy1 + size_copy2
        assert total < size_orig * 1.5, \
            f"Total ({total}) should be < 1.5x original ({size_orig * 1.5})"

    def test_nullable_integer_array_dedup(self):
        """Nullable integer ExtensionArray should be deduplicated."""
        arr = pd.array([1, 2, None, 4, 5] * 2000, dtype=pd.Int64Dtype())

        sizer = HeapSizer()
        size1 = sizer.sizeof(arr)
        size2 = sizer.sizeof(arr)

        assert size1 > 0
        assert size2 == 0, "Same nullable int array should return 0 on second measurement"

    def test_categorical_array_dedup(self):
        """Categorical ExtensionArray should be deduplicated."""
        arr = pd.Categorical(['a', 'b', 'c'] * 10000)

        sizer = HeapSizer()
        size1 = sizer.sizeof(arr)
        size2 = sizer.sizeof(arr)

        assert size1 > 0
        assert size2 == 0, "Same Categorical should return 0 on second measurement"

    def test_datetime_array_dedup(self):
        """DatetimeArray should be deduplicated."""
        arr = pd.array(pd.date_range('2020-01-01', periods=10000))

        sizer = HeapSizer()
        size1 = sizer.sizeof(arr)
        size2 = sizer.sizeof(arr)

        assert size1 > 0
        assert size2 == 0, "Same DatetimeArray should return 0 on second measurement"

    def test_extension_array_in_series_dedup(self):
        """Series with ExtensionArray should deduplicate the backing array."""
        s = pd.Series(pd.array(['hello'] * 5000, dtype='string'))

        sizer = HeapSizer()
        size1 = sizer.sizeof(s)
        size2 = sizer.sizeof(s)

        assert size1 > 0
        assert size2 == 0, "Same Series should return 0 on second measurement"

    def test_mixed_dtype_dataframe_dedup(self):
        """DataFrame with mixed dtypes including ExtensionArrays should deduplicate."""
        df = pd.DataFrame({
            'int_col': np.arange(10000),
            'float_col': np.random.randn(10000),
            'str_col': pd.array(['test'] * 10000, dtype='string'),
            'nullable_int': pd.array([1, 2, None] * 3333 + [4], dtype=pd.Int64Dtype()),
        })

        sizer = HeapSizer()
        size1 = sizer.sizeof(df)
        size2 = sizer.sizeof(df)

        assert size1 > 100000  # Meaningful size for mixed data
        assert size2 == 0, "Same DataFrame should return 0 on second measurement"


class TestCheckpointOverheadBeyondNamespace:
    """Tests for sizeof_checkpoints_beyond_namespace method.

    This method properly measures checkpoint memory BEYOND what's in the namespace,
    correctly handling Copy-on-Write sharing. It fixes the issue where checkpoint
    overhead was incorrectly reported as 4x+ the namespace size.
    """

    def test_basic_checkpoint_overhead(self):
        """Test basic checkpoint overhead measurement."""
        from flowbook.kernel_support.heap_size import CheckpointOverhead

        # Create namespace
        ns = {'arr': np.zeros(10_000)}  # ~80KB

        # Create mock checkpoint with same data
        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        ckpt = MockCheckpoint({'arr': ns['arr']})  # Same object - CoW sharing

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [('ckpt_0', ckpt)])

        assert isinstance(result, CheckpointOverhead)
        # Checkpoint should have minimal overhead since it shares the same object
        assert result.total_mb < 0.01, f"Expected minimal overhead, got {result.total_mb}MB"
        assert 'ckpt_0' in result.by_checkpoint
        assert 'ckpt_0' in result.cumulative

    def test_checkpoint_with_new_data(self):
        """Test checkpoint with data not in namespace."""
        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        # Namespace has one array
        ns = {'arr1': np.zeros(10_000)}  # ~80KB

        # Checkpoint has different array
        ckpt = MockCheckpoint({'arr2': np.zeros(10_000)})  # Different object

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [('ckpt_0', ckpt)])

        # Checkpoint should have ~80KB overhead (new data)
        assert result.total_mb > 0.07, f"Expected ~80KB overhead, got {result.total_mb}MB"
        assert result.total_mb < 0.1
        assert result.by_checkpoint['ckpt_0'] > 0.07

    def test_multiple_checkpoints_cumulative(self):
        """Test that multiple checkpoints measure cumulatively."""
        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        # Namespace
        ns = {'x': 1}

        # Three checkpoints, each with new data
        arr1 = np.zeros(10_000)  # ~80KB
        arr2 = np.zeros(10_000)  # ~80KB
        arr3 = np.zeros(10_000)  # ~80KB

        ckpt1 = MockCheckpoint({'arr1': arr1})
        ckpt2 = MockCheckpoint({'arr2': arr2})
        ckpt3 = MockCheckpoint({'arr3': arr3})

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(
            ns, [('ckpt_1', ckpt1), ('ckpt_2', ckpt2), ('ckpt_3', ckpt3)]
        )

        # Total should be ~240KB (all three arrays)
        assert result.total_mb > 0.2, f"Expected ~240KB total, got {result.total_mb}MB"
        assert result.total_mb < 0.3

        # Each checkpoint adds ~80KB
        assert result.by_checkpoint['ckpt_1'] > 0.07
        assert result.by_checkpoint['ckpt_2'] > 0.07
        assert result.by_checkpoint['ckpt_3'] > 0.07

        # Cumulative should grow
        assert result.cumulative['ckpt_1'] < result.cumulative['ckpt_2']
        assert result.cumulative['ckpt_2'] < result.cumulative['ckpt_3']

    def test_checkpoint_sharing_between_checkpoints(self):
        """Test that shared data between checkpoints is deduplicated."""
        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        ns = {'x': 1}

        # Two checkpoints sharing the same array
        shared_arr = np.zeros(100_000)  # ~800KB
        ckpt1 = MockCheckpoint({'arr': shared_arr})
        ckpt2 = MockCheckpoint({'arr': shared_arr})  # Same object

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(
            ns, [('ckpt_1', ckpt1), ('ckpt_2', ckpt2)]
        )

        # Total should be ~800KB (not 1.6MB)
        assert result.total_mb > 0.7, f"Expected ~800KB, got {result.total_mb}MB"
        assert result.total_mb < 1.0

        # First checkpoint gets credit, second gets 0
        assert result.by_checkpoint['ckpt_1'] > 0.7
        assert result.by_checkpoint['ckpt_2'] < 0.01

    def test_cow_sharing_with_dataframe(self):
        """Test CoW sharing with pandas DataFrames."""
        from flowbook.kernel_support.deepcopy import deepcopy

        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        # Namespace with DataFrame
        df = pd.DataFrame({'a': np.zeros(100_000)})  # ~800KB
        ns = {'df': df}

        # Checkpoint with CoW copy
        df_copy = deepcopy(df, {})  # Should share underlying data
        ckpt = MockCheckpoint({'df': df_copy})

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [('ckpt_0', ckpt)])

        # Checkpoint should have minimal overhead (CoW sharing)
        assert result.total_mb < 0.1, \
            f"CoW copy should have minimal overhead, got {result.total_mb}MB"

    def test_empty_checkpoint(self):
        """Test checkpoint without user_ns."""
        class MockCheckpointNoNs:
            pass

        ns = {'x': 1}
        ckpt = MockCheckpointNoNs()

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [('ckpt_0', ckpt)])

        assert result.total_mb == 0
        assert result.by_checkpoint['ckpt_0'] == 0.0

    def test_no_checkpoints(self):
        """Test with empty checkpoint list."""
        ns = {'x': np.zeros(10_000)}

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [])

        assert result.total_mb == 0
        assert len(result.by_checkpoint) == 0
        assert len(result.cumulative) == 0

    def test_per_variable_breakdown(self):
        """Test by_variable breakdown in result."""
        class MockCheckpoint:
            def __init__(self, data):
                self.user_ns = data

        ns = {'x': 1}

        arr1 = np.zeros(50_000)  # ~400KB
        arr2 = np.zeros(30_000)  # ~240KB
        ckpt = MockCheckpoint({'var1': arr1, 'var2': arr2})

        sizer = HeapSizer()
        result = sizer.sizeof_checkpoints_beyond_namespace(ns, [('ckpt_0', ckpt)])

        # Should have breakdown by variable
        assert 'var1' in result.by_variable
        assert 'var2' in result.by_variable
        assert result.by_variable['var1'] > 0.3  # ~400KB
        assert result.by_variable['var2'] > 0.2  # ~240KB


class TestExtensionArrayRegression:
    """Regression tests for the infer_string memory growth bug.

    When pd.options.future.infer_string = True, string columns become StringDtype
    backed by StringArray. Without proper deduplication, checkpoint overhead
    grows linearly instead of staying constant.
    """

    def test_infer_string_dataframe_checkpoint_overhead(self):
        """Verify that infer_string DataFrames don't cause linear memory growth."""
        from flowbook.kernel_support.deepcopy import deepcopy

        # Create DataFrame with string columns (like read_csv would with infer_string)
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'] * 5000,
            'city': ['NYC', 'LA', 'Chicago'] * 5000,
        })

        # With infer_string, these would be StringDtype
        # Even without it, test the deduplication behavior
        df['name'] = df['name'].astype('string')
        df['city'] = df['city'].astype('string')

        # Simulate multiple checkpoints
        checkpoints = [deepcopy(df, {}) for _ in range(5)]

        # Measure total memory across all checkpoints
        sizer = HeapSizer()
        sizes = [sizer.sizeof(ckpt) for ckpt in [df] + checkpoints]

        # First checkpoint should be large
        assert sizes[0] > 50000, f"Original should be substantial, got {sizes[0]}"

        # Subsequent checkpoints should have minimal overhead
        for i, size in enumerate(sizes[1:], 1):
            assert size < sizes[0] * 0.2, \
                f"Checkpoint {i} size ({size}) should be << original ({sizes[0]})"

        # Total should be much less than 6x the original (would be 6x without dedup)
        total = sum(sizes)
        assert total < sizes[0] * 2, \
            f"Total ({total}) should be < 2x original ({sizes[0] * 2})"

    def test_large_string_dataframe_no_memory_explosion(self):
        """Large DataFrames with string columns shouldn't cause memory explosion."""
        from flowbook.kernel_support.deepcopy import deepcopy

        # Larger DataFrame
        n_rows = 50000
        df = pd.DataFrame({
            'col1': pd.array(['value'] * n_rows, dtype='string'),
            'col2': pd.array(['data'] * n_rows, dtype='string'),
        })

        # Create two checkpoints
        ckpt1 = deepcopy(df, {})
        ckpt2 = deepcopy(df, {})

        sizer = HeapSizer()
        orig_size = sizer.sizeof(df)
        ckpt1_size = sizer.sizeof(ckpt1)
        ckpt2_size = sizer.sizeof(ckpt2)

        # Without the fix, this would be approximately:
        # total ≈ orig_size * 3 (each checkpoint fully counted)
        # With the fix:
        # total ≈ orig_size + 2 * small_overhead

        total = orig_size + ckpt1_size + ckpt2_size
        expected_max = orig_size * 1.5  # Allow 50% overhead for wrappers

        assert total < expected_max, \
            f"Total memory ({total:,}) exceeds expected max ({expected_max:,}). " \
            f"Breakdown: orig={orig_size:,}, ckpt1={ckpt1_size:,}, ckpt2={ckpt2_size:,}"


# ============================================================================
# HEAP SIZER SUBSET ACCOUNTING TESTS
# ============================================================================


class TestHeapSizerSubsetAccounting:
    """Test HeapSizer memory accounting with DataFrame subset relations."""

    def test_row_subset_counted_separately(self):
        """Test that parent and child (row subset) both have full memory counted.

        When DataFrames are NOT CoW copies (different underlying data),
        each should be counted at their actual size.
        """
        parent = pd.DataFrame({
            'a': np.arange(100_000),
            'b': np.arange(100_000, 200_000),
        })
        # Create a row subset (not a view - .copy() creates new arrays)
        child = parent.iloc[::2].copy()

        sizer = HeapSizer()
        parent_size = sizer.sizeof(parent)
        sizer.reset()
        child_size = sizer.sizeof(child)

        # Parent should be ~1.6MB (2 cols * 100K * 8 bytes)
        assert parent_size > 1_500_000
        # Child should be ~800KB (2 cols * 50K * 8 bytes)
        assert child_size > 750_000
        assert child_size < parent_size

    def test_memory_view_deduplicated(self):
        """Test that view slice counts wrapper only when measuring owned data."""
        base_arr = np.zeros(1_000_000)  # 8 MB
        view = base_arr[::2]  # View of every other element

        sizer = HeapSizer()
        # Measure view with owned_only=True
        view_size = sizer.sizeof(view, owned_only=True)
        # Should only count the wrapper, not the shared buffer
        assert view_size < 500  # Just overhead

        sizer.reset()
        # Measure full buffer
        base_size = sizer.sizeof(base_arr)
        assert base_size > 8_000_000

    def test_multiple_subsets_from_same_parent(self):
        """Test multiple children from same parent are measured correctly."""
        parent = pd.DataFrame({
            'a': np.arange(100_000),
            'b': np.arange(100_000, 200_000),
        })

        child1 = parent.iloc[:25000].copy()
        child2 = parent.iloc[25000:50000].copy()
        child3 = parent.iloc[50000:75000].copy()

        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace({
            'parent': parent,
            'child1': child1,
            'child2': child2,
            'child3': child3,
        })

        # Should measure each DataFrame at its actual size
        assert 'parent' in ns_size.by_variable
        assert 'child1' in ns_size.by_variable
        assert 'child2' in ns_size.by_variable
        assert 'child3' in ns_size.by_variable

        # Parent should be largest
        assert ns_size.by_variable['parent'] > ns_size.by_variable['child1']

    def test_chain_of_subsets_accounting(self):
        """Test df -> df_a -> df_b chain memory accounting."""
        df = pd.DataFrame({'x': np.arange(100_000)})
        df_a = df.iloc[::2].copy()  # 50K rows
        df_b = df_a.iloc[::2].copy()  # 25K rows

        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace({
            'df': df,
            'df_a': df_a,
            'df_b': df_b,
        })

        # Each should have its own size
        assert ns_size.by_variable['df'] > ns_size.by_variable['df_a']
        assert ns_size.by_variable['df_a'] > ns_size.by_variable['df_b']

    def test_cow_copy_accounting(self):
        """Test CoW copy before and after mutation."""
        from flowbook.kernel_support.deepcopy import deepcopy

        df = pd.DataFrame({'a': np.zeros(100_000)})
        df_cow = deepcopy(df, {})  # CoW copy shares data

        sizer = HeapSizer()
        ns_size = sizer.sizeof_namespace({'df': df, 'df_cow': df_cow})

        # Total should be much less than 2x (data is shared)
        orig_size = HeapSizer().sizeof(df)
        assert ns_size.total_bytes < orig_size * 1.5


class TestHeapSizerCheckpointSubsetInteraction:
    """Test HeapSizer with checkpoint and subset interaction."""

    def test_checkpoint_with_subset_relation_memory(self):
        """Test that checkpoint storing subset relation measures indices, not full child."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints

        parent = pd.DataFrame({
            'a': np.arange(50_000),
            'b': np.arange(50_000, 100_000),
        })
        child = parent.iloc[::2].copy()

        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=0,
        )

        user_ns = {'parent': parent, 'child': child}
        cp.save('test', user_ns)

        # Get the checkpoint
        checkpoint = cp.get('test')

        # Measure the checkpoint's memory
        sizer = HeapSizer()
        ckpt_size = sizer.sizeof_checkpoint(checkpoint)

        assert ckpt_size.total_bytes > 0
        assert 'parent' in ckpt_size.by_variable

    def test_reconstructed_subset_memory(self):
        """Test that after restore, child has full DataFrame memory."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints

        parent = pd.DataFrame({
            'a': np.arange(10_000),
            'b': np.arange(10_000, 20_000),
        })
        child = parent.iloc[::2].copy()

        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=0,
        )

        user_ns = {'parent': parent, 'child': child}
        cp.save('test', user_ns)

        # Clear and restore
        user_ns.clear()
        cp.restore('test', user_ns)

        # Measure restored child
        sizer = HeapSizer()
        restored_child_size = sizer.sizeof(user_ns['child'])

        # Should be substantial (at least 20KB for 5K rows * 2 cols)
        # Note: reconstructed DataFrames may have different internal structure
        assert restored_child_size > 20_000

        # Verify data integrity
        assert len(user_ns['child']) == 5_000
        assert list(user_ns['child'].columns) == ['a', 'b']

    def test_memory_before_and_after_restore(self):
        """Test that pre/post restore DataFrames have similar memory (within 2x)."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints

        parent = pd.DataFrame({
            'a': np.arange(5_000),
            'b': np.arange(5_000, 10_000),
        })
        child = parent.iloc[::2].copy()

        # Measure before
        sizer1 = HeapSizer()
        pre_child_size = sizer1.sizeof(child)

        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=0,
        )

        user_ns = {'parent': parent, 'child': child}
        cp.save('test', user_ns)
        user_ns.clear()
        cp.restore('test', user_ns)

        # Measure after restore
        sizer2 = HeapSizer()
        post_child_size = sizer2.sizeof(user_ns['child'])

        # Reconstructed DataFrames may have different memory layout
        # but should be within 2x of original (not wildly different)
        assert post_child_size > 0
        assert post_child_size < pre_child_size * 2

        # More importantly: verify data is correct
        assert len(user_ns['child']) == 2_500
        assert list(user_ns['child'].columns) == ['a', 'b']
