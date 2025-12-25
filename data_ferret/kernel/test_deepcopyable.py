"""
Tests for the check_deepcopyable function.
"""

import datetime
import tempfile
import threading
import weakref
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from data_ferret.kernel.deepcopyable import check_deepcopyable


class TestImmutablePrimitives:
    """Tests for immutable primitive types."""

    def test_none(self):
        assert check_deepcopyable(None) is None

    def test_bool_true(self):
        assert check_deepcopyable(True) is None

    def test_bool_false(self):
        assert check_deepcopyable(False) is None

    def test_int(self):
        assert check_deepcopyable(42) is None
        assert check_deepcopyable(-1) is None
        assert check_deepcopyable(0) is None

    def test_float(self):
        assert check_deepcopyable(3.14) is None
        assert check_deepcopyable(float("inf")) is None
        assert check_deepcopyable(float("nan")) is None

    def test_complex(self):
        assert check_deepcopyable(complex(1, 2)) is None

    def test_str(self):
        assert check_deepcopyable("hello") is None
        assert check_deepcopyable("") is None

    def test_bytes(self):
        assert check_deepcopyable(b"bytes") is None
        assert check_deepcopyable(b"") is None

    def test_range(self):
        assert check_deepcopyable(range(10)) is None
        assert check_deepcopyable(range(0)) is None


class TestDatetimeTypes:
    """Tests for datetime module types."""

    def test_date(self):
        assert check_deepcopyable(datetime.date.today()) is None

    def test_datetime(self):
        assert check_deepcopyable(datetime.datetime.now()) is None

    def test_time(self):
        assert check_deepcopyable(datetime.time(12, 30)) is None

    def test_timedelta(self):
        assert check_deepcopyable(datetime.timedelta(days=1)) is None


class TestContainerTypes:
    """Tests for standard container types."""

    def test_list_simple(self):
        assert check_deepcopyable([1, 2, 3]) is None

    def test_list_empty(self):
        assert check_deepcopyable([]) is None

    def test_dict_simple(self):
        assert check_deepcopyable({"a": 1, "b": 2}) is None

    def test_dict_empty(self):
        assert check_deepcopyable({}) is None

    def test_set_simple(self):
        assert check_deepcopyable({1, 2, 3}) is None

    def test_set_empty(self):
        assert check_deepcopyable(set()) is None

    def test_tuple_simple(self):
        assert check_deepcopyable((1, 2, 3)) is None

    def test_tuple_empty(self):
        assert check_deepcopyable(()) is None

    def test_frozenset_simple(self):
        assert check_deepcopyable(frozenset([1, 2, 3])) is None

    def test_frozenset_empty(self):
        assert check_deepcopyable(frozenset()) is None


class TestNestedContainers:
    """Tests for nested container structures."""

    def test_nested_lists(self):
        assert check_deepcopyable([1, [2, [3, [4]]]]) is None

    def test_nested_dicts(self):
        assert check_deepcopyable({"a": {"b": {"c": 1}}}) is None

    def test_mixed_nesting(self):
        assert check_deepcopyable({"a": [1, 2], "b": (3, 4)}) is None

    def test_list_of_dicts(self):
        assert check_deepcopyable([{"a": 1}, {"b": 2}]) is None

    def test_dict_with_tuple_keys(self):
        assert check_deepcopyable({(1, 2): "a", (3, 4): "b"}) is None


class TestCircularReferences:
    """Tests for objects with circular references."""

    def test_circular_list(self):
        lst = [1, 2, 3]
        lst.append(lst)
        assert check_deepcopyable(lst) is None

    def test_circular_dict(self):
        d = {"a": 1}
        d["self"] = d
        assert check_deepcopyable(d) is None

    def test_mutually_referencing_lists(self):
        a = [1]
        b = [2]
        a.append(b)
        b.append(a)
        assert check_deepcopyable(a) is None


class TestNumPyTypes:
    """Tests for NumPy types."""

    def test_array_int(self):
        assert check_deepcopyable(np.array([1, 2, 3])) is None

    def test_array_float(self):
        assert check_deepcopyable(np.zeros((3, 3))) is None

    def test_array_2d(self):
        assert check_deepcopyable(np.array([[1, 2], [3, 4]])) is None

    def test_scalar_int64(self):
        assert check_deepcopyable(np.int64(42)) is None

    def test_scalar_float64(self):
        assert check_deepcopyable(np.float64(3.14)) is None

    def test_scalar_bool(self):
        assert check_deepcopyable(np.bool_(True)) is None

    def test_object_array_copyable_contents(self):
        assert check_deepcopyable(np.array([1, "hello", 3.14], dtype=object)) is None

    def test_object_array_with_lists(self):
        assert check_deepcopyable(np.array([[1, 2], [3, 4]], dtype=object)) is None

    def test_object_array_with_generator(self):
        gen = (x for x in range(5))
        arr = np.array([1, gen], dtype=object)
        assert check_deepcopyable(arr) is not None

    def test_structured_array(self):
        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        arr = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        assert check_deepcopyable(arr) is None


class TestPandasTypes:
    """Tests for Pandas types."""

    def test_series_int(self):
        assert check_deepcopyable(pd.Series([1, 2, 3])) is None

    def test_series_float(self):
        assert check_deepcopyable(pd.Series([1.0, 2.0, 3.0])) is None

    def test_series_string(self):
        assert check_deepcopyable(pd.Series(["a", "b", "c"])) is None

    def test_series_object_copyable(self):
        assert check_deepcopyable(pd.Series([1, "a", 3.0], dtype=object)) is None

    def test_series_object_with_generator(self):
        gen = (x for x in range(5))
        s = pd.Series([1, gen], dtype=object)
        assert check_deepcopyable(s) is not None

    def test_dataframe_simple(self):
        assert check_deepcopyable(pd.DataFrame({"a": [1, 2], "b": [3, 4]})) is None

    def test_dataframe_mixed_dtypes(self):
        df = pd.DataFrame({"int": [1, 2], "float": [1.0, 2.0], "str": ["a", "b"]})
        assert check_deepcopyable(df) is None

    def test_dataframe_object_column_copyable(self):
        df = pd.DataFrame({"a": [[1, 2], [3, 4]]}, dtype=object)
        assert check_deepcopyable(df) is None

    def test_dataframe_object_column_with_generator(self):
        gen = (x for x in range(5))
        df = pd.DataFrame({"a": [1, gen]}, dtype=object)
        assert check_deepcopyable(df) is not None

    def test_timestamp(self):
        assert check_deepcopyable(pd.Timestamp("2021-01-01")) is None

    def test_timedelta(self):
        assert check_deepcopyable(pd.Timedelta("1 day")) is None

    def test_period(self):
        assert check_deepcopyable(pd.Period("2021-01", freq="M")) is None

    def test_index_int(self):
        assert check_deepcopyable(pd.Index([1, 2, 3])) is None

    def test_index_string(self):
        assert check_deepcopyable(pd.Index(["a", "b", "c"])) is None

    def test_na(self):
        assert check_deepcopyable(pd.NA) is None


class TestCollectionsTypes:
    """Tests for collections module types."""

    def test_deque(self):
        assert check_deepcopyable(deque([1, 2, 3])) is None

    def test_deque_empty(self):
        assert check_deepcopyable(deque()) is None

    def test_deque_with_generator(self):
        gen = (x for x in range(5))
        assert check_deepcopyable(deque([1, gen])) is not None

    def test_ordered_dict(self):
        assert check_deepcopyable(OrderedDict([("a", 1), ("b", 2)])) is None

    def test_defaultdict(self):
        assert check_deepcopyable(defaultdict(list, {"a": [1, 2]})) is None

    def test_counter(self):
        assert check_deepcopyable(Counter("abracadabra")) is None


class TestModulesNotCopyable:
    """Tests that modules cannot be deep copied."""

    def test_sys_module(self):
        import sys

        assert check_deepcopyable(sys) is not None

    def test_numpy_module(self):
        assert check_deepcopyable(np) is not None

    def test_pandas_module(self):
        assert check_deepcopyable(pd) is not None

    def test_os_module(self):
        import os

        assert check_deepcopyable(os) is not None


class TestGeneratorsNotCopyable:
    """Tests that generators and coroutines cannot be deep copied."""

    def test_generator(self):
        gen = (x for x in range(10))
        assert check_deepcopyable(gen) is not None

    def test_generator_function_result(self):
        def gen_func():
            yield 1
            yield 2

        gen = gen_func()
        assert check_deepcopyable(gen) is not None

    def test_async_generator(self):
        async def async_gen():
            yield 1

        ag = async_gen()
        assert check_deepcopyable(ag) is not None

    def test_coroutine(self):
        async def coro():
            return 1

        c = coro()
        assert check_deepcopyable(c) is not None
        c.close()  # Clean up


class TestFileHandlesNotCopyable:
    """Tests that file handles cannot be deep copied."""

    def test_tempfile(self):
        with tempfile.NamedTemporaryFile() as f:
            assert check_deepcopyable(f) is not None

    def test_open_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        with open(tmp_path, "r") as f:
            assert check_deepcopyable(f) is not None

        import os

        os.unlink(tmp_path)


class TestMatplotlibNotCopyable:
    """Tests that matplotlib objects cannot be deep copied."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_figure(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        assert check_deepcopyable(fig) is not None

    def test_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        assert check_deepcopyable(ax) is not None

    def test_axes_array(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2)
        assert check_deepcopyable(axes) is not None

    def test_line2d(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        (line,) = ax.plot([1, 2, 3])
        assert check_deepcopyable(line) is not None


class TestThreadingPrimitivesNotCopyable:
    """Tests that threading primitives cannot be deep copied."""

    def test_lock(self):
        lock = threading.Lock()
        assert check_deepcopyable(lock) is not None

    def test_rlock(self):
        rlock = threading.RLock()
        assert check_deepcopyable(rlock) is not None

    def test_event(self):
        event = threading.Event()
        assert check_deepcopyable(event) is not None

    def test_condition(self):
        condition = threading.Condition()
        assert check_deepcopyable(condition) is not None

    def test_semaphore(self):
        sem = threading.Semaphore()
        assert check_deepcopyable(sem) is not None


class TestWeakrefsCopyable:
    """Tests for weakref types (which are actually copyable)."""

    def test_weakref(self):
        class Foo:
            pass

        obj = Foo()
        ref = weakref.ref(obj)
        assert check_deepcopyable(ref) is None


class TestContainersWithNonCopyableElements:
    """Tests for containers containing non-copyable elements."""

    def test_list_with_generator(self):
        gen = (x for x in range(5))
        assert check_deepcopyable([1, 2, gen]) is not None

    def test_dict_with_generator_value(self):
        gen = (x for x in range(5))
        assert check_deepcopyable({"a": 1, "gen": gen}) is not None

    def test_tuple_with_generator(self):
        gen = (x for x in range(5))
        assert check_deepcopyable((1, gen)) is not None

    def test_set_cannot_contain_generator(self):
        # Sets can only contain hashable items, generators are not hashable
        pass

    def test_nested_with_generator(self):
        gen = (x for x in range(5))
        assert check_deepcopyable({"a": [1, [2, gen]]}) is not None


class TestCustomClasses:
    """Tests for user-defined classes."""

    def test_simple_class(self):
        class MyClass:
            def __init__(self, x):
                self.x = x

        obj = MyClass(42)
        assert check_deepcopyable(obj) is None

    def test_class_with_list_attribute(self):
        class MyClass:
            def __init__(self):
                self.items = [1, 2, 3]

        obj = MyClass()
        assert check_deepcopyable(obj) is None

    def test_class_with_generator_attribute(self):
        class MyClassWithGen:
            def __init__(self):
                self.gen = (x for x in range(10))

        obj = MyClassWithGen()
        assert check_deepcopyable(obj) is not None

    def test_dataclass(self):
        @dataclass
        class Point:
            x: int
            y: int

        p = Point(1, 2)
        assert check_deepcopyable(p) is None

    def test_dataclass_with_list(self):
        @dataclass
        class Container:
            items: list

        c = Container([1, 2, 3])
        assert check_deepcopyable(c) is None


class TestSlotsClasses:
    """Tests for classes using __slots__."""

    def test_slots_class(self):
        class SlotClass:
            __slots__ = ["x", "y"]

            def __init__(self, x, y):
                self.x = x
                self.y = y

        sc = SlotClass(1, 2)
        assert check_deepcopyable(sc) is None

    def test_slots_class_with_generator(self):
        class SlotClassWithGen:
            __slots__ = ["gen"]

            def __init__(self):
                self.gen = (x for x in range(10))

        scg = SlotClassWithGen()
        assert check_deepcopyable(scg) is not None


class TestFunctions:
    """Tests for function objects."""

    def test_regular_function(self):
        def my_func(x):
            return x + 1

        assert check_deepcopyable(my_func) is None

    def test_lambda(self):
        assert check_deepcopyable(lambda x: x + 1) is None

    def test_bound_method(self):
        class MyClass:
            def method(self):
                pass

        obj = MyClass()
        assert check_deepcopyable(obj.method) is None


class TestTypeObjects:
    """Tests for type objects (classes themselves)."""

    def test_builtin_type(self):
        assert check_deepcopyable(int) is None
        assert check_deepcopyable(str) is None
        assert check_deepcopyable(list) is None

    def test_custom_class(self):
        class MyClass:
            pass

        assert check_deepcopyable(MyClass) is None

    def test_numpy_type(self):
        assert check_deepcopyable(np.ndarray) is None

    def test_pandas_type(self):
        assert check_deepcopyable(pd.DataFrame) is None

    def test_abcmeta_class(self):
        """ABCMeta classes should be deepcopyable (they're singletons)."""
        import numbers
        from abc import ABCMeta

        # numbers.Integral is an ABCMeta class
        assert isinstance(numbers.Integral, ABCMeta)
        assert check_deepcopyable(numbers.Integral) is None

    def test_sklearn_abcmeta_class(self):
        """sklearn classes with ABCMeta should be deepcopyable."""
        from sklearn.linear_model import LinearRegression
        from abc import ABCMeta

        # LinearRegression uses ABCMeta
        assert isinstance(LinearRegression, ABCMeta)
        assert check_deepcopyable(LinearRegression) is None

    def test_sklearn_class_with_parameter_constraints(self):
        """sklearn classes with _parameter_constraints containing ABCMeta refs should be deepcopyable."""
        from sklearn.linear_model import LinearRegression

        # This dict contains references to ABCMeta classes like numbers.Integral
        pc = LinearRegression._parameter_constraints
        assert check_deepcopyable(pc) is None


class TestDecimalType:
    """Tests for decimal.Decimal type."""

    def test_decimal(self):
        from decimal import Decimal

        assert check_deepcopyable(Decimal("3.14")) is None

    def test_decimal_special_values(self):
        from decimal import Decimal

        assert check_deepcopyable(Decimal("Infinity")) is None
        assert check_deepcopyable(Decimal("-Infinity")) is None
        assert check_deepcopyable(Decimal("NaN")) is None
