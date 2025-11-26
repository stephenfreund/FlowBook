"""
Tests for the is_deepcopyable function.
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

from data_ferret.kernel.deepcopyable import is_deepcopyable


class TestImmutablePrimitives:
    """Tests for immutable primitive types."""

    def test_none(self):
        assert is_deepcopyable(None) is True

    def test_bool_true(self):
        assert is_deepcopyable(True) is True

    def test_bool_false(self):
        assert is_deepcopyable(False) is True

    def test_int(self):
        assert is_deepcopyable(42) is True
        assert is_deepcopyable(-1) is True
        assert is_deepcopyable(0) is True

    def test_float(self):
        assert is_deepcopyable(3.14) is True
        assert is_deepcopyable(float("inf")) is True
        assert is_deepcopyable(float("nan")) is True

    def test_complex(self):
        assert is_deepcopyable(complex(1, 2)) is True

    def test_str(self):
        assert is_deepcopyable("hello") is True
        assert is_deepcopyable("") is True

    def test_bytes(self):
        assert is_deepcopyable(b"bytes") is True
        assert is_deepcopyable(b"") is True

    def test_range(self):
        assert is_deepcopyable(range(10)) is True
        assert is_deepcopyable(range(0)) is True


class TestDatetimeTypes:
    """Tests for datetime module types."""

    def test_date(self):
        assert is_deepcopyable(datetime.date.today()) is True

    def test_datetime(self):
        assert is_deepcopyable(datetime.datetime.now()) is True

    def test_time(self):
        assert is_deepcopyable(datetime.time(12, 30)) is True

    def test_timedelta(self):
        assert is_deepcopyable(datetime.timedelta(days=1)) is True


class TestContainerTypes:
    """Tests for standard container types."""

    def test_list_simple(self):
        assert is_deepcopyable([1, 2, 3]) is True

    def test_list_empty(self):
        assert is_deepcopyable([]) is True

    def test_dict_simple(self):
        assert is_deepcopyable({"a": 1, "b": 2}) is True

    def test_dict_empty(self):
        assert is_deepcopyable({}) is True

    def test_set_simple(self):
        assert is_deepcopyable({1, 2, 3}) is True

    def test_set_empty(self):
        assert is_deepcopyable(set()) is True

    def test_tuple_simple(self):
        assert is_deepcopyable((1, 2, 3)) is True

    def test_tuple_empty(self):
        assert is_deepcopyable(()) is True

    def test_frozenset_simple(self):
        assert is_deepcopyable(frozenset([1, 2, 3])) is True

    def test_frozenset_empty(self):
        assert is_deepcopyable(frozenset()) is True


class TestNestedContainers:
    """Tests for nested container structures."""

    def test_nested_lists(self):
        assert is_deepcopyable([1, [2, [3, [4]]]]) is True

    def test_nested_dicts(self):
        assert is_deepcopyable({"a": {"b": {"c": 1}}}) is True

    def test_mixed_nesting(self):
        assert is_deepcopyable({"a": [1, 2], "b": (3, 4)}) is True

    def test_list_of_dicts(self):
        assert is_deepcopyable([{"a": 1}, {"b": 2}]) is True

    def test_dict_with_tuple_keys(self):
        assert is_deepcopyable({(1, 2): "a", (3, 4): "b"}) is True


class TestCircularReferences:
    """Tests for objects with circular references."""

    def test_circular_list(self):
        lst = [1, 2, 3]
        lst.append(lst)
        assert is_deepcopyable(lst) is True

    def test_circular_dict(self):
        d = {"a": 1}
        d["self"] = d
        assert is_deepcopyable(d) is True

    def test_mutually_referencing_lists(self):
        a = [1]
        b = [2]
        a.append(b)
        b.append(a)
        assert is_deepcopyable(a) is True


class TestNumPyTypes:
    """Tests for NumPy types."""

    def test_array_int(self):
        assert is_deepcopyable(np.array([1, 2, 3])) is True

    def test_array_float(self):
        assert is_deepcopyable(np.zeros((3, 3))) is True

    def test_array_2d(self):
        assert is_deepcopyable(np.array([[1, 2], [3, 4]])) is True

    def test_scalar_int64(self):
        assert is_deepcopyable(np.int64(42)) is True

    def test_scalar_float64(self):
        assert is_deepcopyable(np.float64(3.14)) is True

    def test_scalar_bool(self):
        assert is_deepcopyable(np.bool_(True)) is True

    def test_object_array_copyable_contents(self):
        assert is_deepcopyable(np.array([1, "hello", 3.14], dtype=object)) is True

    def test_object_array_with_lists(self):
        assert is_deepcopyable(np.array([[1, 2], [3, 4]], dtype=object)) is True

    def test_object_array_with_generator(self):
        gen = (x for x in range(5))
        arr = np.array([1, gen], dtype=object)
        assert is_deepcopyable(arr) is False

    def test_structured_array(self):
        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        arr = np.array([(1, 2.0), (3, 4.0)], dtype=dt)
        assert is_deepcopyable(arr) is True


class TestPandasTypes:
    """Tests for Pandas types."""

    def test_series_int(self):
        assert is_deepcopyable(pd.Series([1, 2, 3])) is True

    def test_series_float(self):
        assert is_deepcopyable(pd.Series([1.0, 2.0, 3.0])) is True

    def test_series_string(self):
        assert is_deepcopyable(pd.Series(["a", "b", "c"])) is True

    def test_series_object_copyable(self):
        assert is_deepcopyable(pd.Series([1, "a", 3.0], dtype=object)) is True

    def test_series_object_with_generator(self):
        gen = (x for x in range(5))
        s = pd.Series([1, gen], dtype=object)
        assert is_deepcopyable(s) is False

    def test_dataframe_simple(self):
        assert is_deepcopyable(pd.DataFrame({"a": [1, 2], "b": [3, 4]})) is True

    def test_dataframe_mixed_dtypes(self):
        df = pd.DataFrame({"int": [1, 2], "float": [1.0, 2.0], "str": ["a", "b"]})
        assert is_deepcopyable(df) is True

    def test_dataframe_object_column_copyable(self):
        df = pd.DataFrame({"a": [[1, 2], [3, 4]]}, dtype=object)
        assert is_deepcopyable(df) is True

    def test_dataframe_object_column_with_generator(self):
        gen = (x for x in range(5))
        df = pd.DataFrame({"a": [1, gen]}, dtype=object)
        assert is_deepcopyable(df) is False

    def test_timestamp(self):
        assert is_deepcopyable(pd.Timestamp("2021-01-01")) is True

    def test_timedelta(self):
        assert is_deepcopyable(pd.Timedelta("1 day")) is True

    def test_period(self):
        assert is_deepcopyable(pd.Period("2021-01", freq="M")) is True

    def test_index_int(self):
        assert is_deepcopyable(pd.Index([1, 2, 3])) is True

    def test_index_string(self):
        assert is_deepcopyable(pd.Index(["a", "b", "c"])) is True

    def test_na(self):
        assert is_deepcopyable(pd.NA) is True


class TestCollectionsTypes:
    """Tests for collections module types."""

    def test_deque(self):
        assert is_deepcopyable(deque([1, 2, 3])) is True

    def test_deque_empty(self):
        assert is_deepcopyable(deque()) is True

    def test_deque_with_generator(self):
        gen = (x for x in range(5))
        assert is_deepcopyable(deque([1, gen])) is False

    def test_ordered_dict(self):
        assert is_deepcopyable(OrderedDict([("a", 1), ("b", 2)])) is True

    def test_defaultdict(self):
        assert is_deepcopyable(defaultdict(list, {"a": [1, 2]})) is True

    def test_counter(self):
        assert is_deepcopyable(Counter("abracadabra")) is True


class TestModulesNotCopyable:
    """Tests that modules cannot be deep copied."""

    def test_sys_module(self):
        import sys

        assert is_deepcopyable(sys) is False

    def test_numpy_module(self):
        assert is_deepcopyable(np) is False

    def test_pandas_module(self):
        assert is_deepcopyable(pd) is False

    def test_os_module(self):
        import os

        assert is_deepcopyable(os) is False


class TestGeneratorsNotCopyable:
    """Tests that generators and coroutines cannot be deep copied."""

    def test_generator(self):
        gen = (x for x in range(10))
        assert is_deepcopyable(gen) is False

    def test_generator_function_result(self):
        def gen_func():
            yield 1
            yield 2

        gen = gen_func()
        assert is_deepcopyable(gen) is False

    def test_async_generator(self):
        async def async_gen():
            yield 1

        ag = async_gen()
        assert is_deepcopyable(ag) is False

    def test_coroutine(self):
        async def coro():
            return 1

        c = coro()
        assert is_deepcopyable(c) is False
        c.close()  # Clean up


class TestFileHandlesNotCopyable:
    """Tests that file handles cannot be deep copied."""

    def test_tempfile(self):
        with tempfile.NamedTemporaryFile() as f:
            assert is_deepcopyable(f) is False

    def test_open_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        with open(tmp_path, "r") as f:
            assert is_deepcopyable(f) is False

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
        assert is_deepcopyable(fig) is False

    def test_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        assert is_deepcopyable(ax) is False

    def test_axes_array(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2)
        assert is_deepcopyable(axes) is False

    def test_line2d(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        (line,) = ax.plot([1, 2, 3])
        assert is_deepcopyable(line) is False


class TestThreadingPrimitivesNotCopyable:
    """Tests that threading primitives cannot be deep copied."""

    def test_lock(self):
        lock = threading.Lock()
        assert is_deepcopyable(lock) is False

    def test_rlock(self):
        rlock = threading.RLock()
        assert is_deepcopyable(rlock) is False

    def test_event(self):
        event = threading.Event()
        assert is_deepcopyable(event) is False

    def test_condition(self):
        condition = threading.Condition()
        assert is_deepcopyable(condition) is False

    def test_semaphore(self):
        sem = threading.Semaphore()
        assert is_deepcopyable(sem) is False


class TestWeakrefsCopyable:
    """Tests for weakref types (which are actually copyable)."""

    def test_weakref(self):
        class Foo:
            pass

        obj = Foo()
        ref = weakref.ref(obj)
        assert is_deepcopyable(ref) is True


class TestContainersWithNonCopyableElements:
    """Tests for containers containing non-copyable elements."""

    def test_list_with_generator(self):
        gen = (x for x in range(5))
        assert is_deepcopyable([1, 2, gen]) is False

    def test_dict_with_generator_value(self):
        gen = (x for x in range(5))
        assert is_deepcopyable({"a": 1, "gen": gen}) is False

    def test_tuple_with_generator(self):
        gen = (x for x in range(5))
        assert is_deepcopyable((1, gen)) is False

    def test_set_cannot_contain_generator(self):
        # Sets can only contain hashable items, generators are not hashable
        pass

    def test_nested_with_generator(self):
        gen = (x for x in range(5))
        assert is_deepcopyable({"a": [1, [2, gen]]}) is False


class TestCustomClasses:
    """Tests for user-defined classes."""

    def test_simple_class(self):
        class MyClass:
            def __init__(self, x):
                self.x = x

        obj = MyClass(42)
        assert is_deepcopyable(obj) is True

    def test_class_with_list_attribute(self):
        class MyClass:
            def __init__(self):
                self.items = [1, 2, 3]

        obj = MyClass()
        assert is_deepcopyable(obj) is True

    def test_class_with_generator_attribute(self):
        class MyClassWithGen:
            def __init__(self):
                self.gen = (x for x in range(10))

        obj = MyClassWithGen()
        assert is_deepcopyable(obj) is False

    def test_dataclass(self):
        @dataclass
        class Point:
            x: int
            y: int

        p = Point(1, 2)
        assert is_deepcopyable(p) is True

    def test_dataclass_with_list(self):
        @dataclass
        class Container:
            items: list

        c = Container([1, 2, 3])
        assert is_deepcopyable(c) is True


class TestSlotsClasses:
    """Tests for classes using __slots__."""

    def test_slots_class(self):
        class SlotClass:
            __slots__ = ["x", "y"]

            def __init__(self, x, y):
                self.x = x
                self.y = y

        sc = SlotClass(1, 2)
        assert is_deepcopyable(sc) is True

    def test_slots_class_with_generator(self):
        class SlotClassWithGen:
            __slots__ = ["gen"]

            def __init__(self):
                self.gen = (x for x in range(10))

        scg = SlotClassWithGen()
        assert is_deepcopyable(scg) is False


class TestFunctions:
    """Tests for function objects."""

    def test_regular_function(self):
        def my_func(x):
            return x + 1

        assert is_deepcopyable(my_func) is True

    def test_lambda(self):
        assert is_deepcopyable(lambda x: x + 1) is True

    def test_bound_method(self):
        class MyClass:
            def method(self):
                pass

        obj = MyClass()
        assert is_deepcopyable(obj.method) is True


class TestTypeObjects:
    """Tests for type objects (classes themselves)."""

    def test_builtin_type(self):
        assert is_deepcopyable(int) is True
        assert is_deepcopyable(str) is True
        assert is_deepcopyable(list) is True

    def test_custom_class(self):
        class MyClass:
            pass

        assert is_deepcopyable(MyClass) is True

    def test_numpy_type(self):
        assert is_deepcopyable(np.ndarray) is True

    def test_pandas_type(self):
        assert is_deepcopyable(pd.DataFrame) is True


class TestDecimalType:
    """Tests for decimal.Decimal type."""

    def test_decimal(self):
        from decimal import Decimal

        assert is_deepcopyable(Decimal("3.14")) is True

    def test_decimal_special_values(self):
        from decimal import Decimal

        assert is_deepcopyable(Decimal("Infinity")) is True
        assert is_deepcopyable(Decimal("-Infinity")) is True
        assert is_deepcopyable(Decimal("NaN")) is True
