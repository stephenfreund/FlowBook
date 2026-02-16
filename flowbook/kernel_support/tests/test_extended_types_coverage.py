"""Tests for extended_types.py - Targeting uncovered __str__ methods and helpers.

Coverage gaps include __str__ for all type models, _make_union, and get_type_model
edge cases.
"""

import numpy as np
import pandas as pd
import inspect
import pytest
from typing import Any, Optional
from collections import deque

from flowbook.kernel_support.extended_types import (
    AtomicType,
    ArrayType,
    DataFrameType,
    DataFrameColumn,
    SeriesType,
    DictType,
    SequenceType,
    SetType,
    FunctionType,
    ParameterModel,
    FallbackType,
    UnionType,
    get_type_model,
    _make_union,
)


class TestAtomicTypeStr:
    """Tests for AtomicType __str__."""

    def test_int_str(self):
        assert str(AtomicType(kind="Atomic", type_name="int")) == "int"

    def test_float_str(self):
        assert str(AtomicType(kind="Atomic", type_name="float")) == "float"


class TestArrayTypeStr:
    """Tests for ArrayType __str__."""

    def test_array_str(self):
        at = ArrayType(kind="ndarray", dtype="float64", shape=(3, 4))
        assert str(at) == "ndarray[float64, shape=(3, 4)]"


class TestDataFrameTypeStr:
    """Tests for DataFrameType __str__."""

    def test_dataframe_str(self):
        cols = [
            DataFrameColumn(name="age", dtype="int64"),
            DataFrameColumn(name="score", dtype="float64"),
        ]
        dt = DataFrameType(kind="DataFrame", n_rows=100, columns=cols)
        result = str(dt)
        assert "100 rows" in result
        assert "'age': int64" in result
        assert "'score': float64" in result


class TestSeriesTypeStr:
    """Tests for SeriesType __str__."""

    def test_series_str(self):
        st = SeriesType(kind="Series", dtype="float64", length=50)
        assert str(st) == "Series[float64, length=50]"


class TestDictTypeStr:
    """Tests for DictType __str__."""

    def test_empty_dict_str(self):
        dt = DictType(kind="Dict", key_types=[], value_types=[])
        result = str(dt)
        assert "Dict[Unknown, Unknown]" == result

    def test_dict_with_types(self):
        kt = AtomicType(kind="Atomic", type_name="str")
        vt = AtomicType(kind="Atomic", type_name="int")
        dt = DictType(kind="Dict", key_types=[kt], value_types=[vt])
        assert str(dt) == "Dict[str, int]"


class TestSequenceTypeStr:
    """Tests for SequenceType __str__."""

    def test_list_str(self):
        et = AtomicType(kind="Atomic", type_name="int")
        st = SequenceType(kind="List", element_types=[et])
        assert str(st) == "List[int]"

    def test_tuple_str(self):
        et = AtomicType(kind="Atomic", type_name="float")
        st = SequenceType(kind="Tuple", element_types=[et])
        assert str(st) == "Tuple[float]"

    def test_deque_str(self):
        et = AtomicType(kind="Atomic", type_name="str")
        st = SequenceType(kind="Deque", element_types=[et])
        assert str(st) == "Deque[str]"

    def test_empty_elements(self):
        st = SequenceType(kind="List", element_types=[])
        assert str(st) == "List[Unknown]"


class TestSetTypeStr:
    """Tests for SetType __str__."""

    def test_set_str(self):
        et = AtomicType(kind="Atomic", type_name="int")
        st = SetType(kind="Set", element_types=[et])
        assert str(st) == "Set[int]"

    def test_empty_set(self):
        st = SetType(kind="Set", element_types=[])
        assert str(st) == "Set[Unknown]"


class TestFunctionTypeStr:
    """Tests for FunctionType __str__."""

    def test_function_str(self):
        params = [
            ParameterModel(name="x", annotation=AtomicType(kind="Atomic", type_name="int"), default=None),
        ]
        ft = FunctionType(
            kind="Function",
            name="foo",
            parameters=params,
            return_type=AtomicType(kind="Atomic", type_name="str"),
        )
        result = str(ft)
        assert "foo" in result
        assert "x: int" in result
        assert "-> str" in result

    def test_function_no_return(self):
        ft = FunctionType(kind="Function", name="bar", parameters=[], return_type=None)
        assert "-> None" in str(ft)


class TestParameterModelStr:
    """Tests for ParameterModel __str__."""

    def test_name_only(self):
        pm = ParameterModel(name="x", annotation=None, default=None)
        assert str(pm) == "x"

    def test_with_annotation(self):
        pm = ParameterModel(
            name="x",
            annotation=AtomicType(kind="Atomic", type_name="int"),
            default=None,
        )
        assert str(pm) == "x: int"

    def test_with_default(self):
        pm = ParameterModel(name="x", annotation=None, default=42)
        assert str(pm) == "x=42"

    def test_with_annotation_and_default(self):
        pm = ParameterModel(
            name="x",
            annotation=AtomicType(kind="Atomic", type_name="int"),
            default=0,
        )
        assert str(pm) == "x: int=0"


class TestFallbackTypeStr:
    """Tests for FallbackType __str__."""

    def test_fallback_str(self):
        ft = FallbackType(kind="Class", class_name="MyClass")
        assert str(ft) == "MyClass"


class TestUnionTypeStr:
    """Tests for UnionType __str__."""

    def test_optional_pattern(self):
        """Union of T and None becomes Optional[T]."""
        types = [
            AtomicType(kind="Atomic", type_name="int"),
            AtomicType(kind="Atomic", type_name="None"),
        ]
        ut = UnionType(kind="Union", types=types)
        assert str(ut) == "Optional[int]"

    def test_regular_union(self):
        """Non-optional union shows Union[...]."""
        types = [
            AtomicType(kind="Atomic", type_name="int"),
            AtomicType(kind="Atomic", type_name="str"),
        ]
        ut = UnionType(kind="Union", types=types)
        assert str(ut) == "Union[int, str]"

    def test_three_element_union(self):
        """Three-element union (not optional pattern)."""
        types = [
            AtomicType(kind="Atomic", type_name="int"),
            AtomicType(kind="Atomic", type_name="str"),
            AtomicType(kind="Atomic", type_name="None"),
        ]
        ut = UnionType(kind="Union", types=types)
        result = str(ut)
        assert "Union[" in result


class TestMakeUnion:
    """Tests for _make_union helper."""

    def test_single_type(self):
        """Single unique type returns that type directly."""
        t = AtomicType(kind="Atomic", type_name="int")
        result = _make_union([t])
        assert result == t

    def test_duplicate_removal(self):
        """Duplicate types are deduplicated."""
        t = AtomicType(kind="Atomic", type_name="int")
        result = _make_union([t, t, t])
        assert result == t

    def test_int64_with_int(self):
        """int64 is removed when int is present."""
        t_int = AtomicType(kind="Atomic", type_name="int")
        t_int64 = AtomicType(kind="Atomic", type_name="int64")
        result = _make_union([t_int64, t_int])
        assert result == t_int

    def test_float64_with_float(self):
        """float64 is removed when float is present."""
        t_float = AtomicType(kind="Atomic", type_name="float")
        t_float64 = AtomicType(kind="Atomic", type_name="float64")
        result = _make_union([t_float64, t_float])
        assert result == t_float

    def test_str96_with_str(self):
        """str96 is removed when str is present."""
        t_str = AtomicType(kind="Atomic", type_name="str")
        t_str96 = AtomicType(kind="Atomic", type_name="str96")
        result = _make_union([t_str96, t_str])
        assert result == t_str

    def test_multiple_distinct_types(self):
        """Multiple distinct types create a UnionType."""
        t_int = AtomicType(kind="Atomic", type_name="int")
        t_str = AtomicType(kind="Atomic", type_name="str")
        result = _make_union([t_int, t_str])
        assert isinstance(result, UnionType)
        assert len(result.types) == 2


class TestGetTypeModel:
    """Tests for get_type_model function."""

    def test_int(self):
        result = get_type_model(42)
        assert str(result) == "int"

    def test_float(self):
        result = get_type_model(3.14)
        assert str(result) == "float"

    def test_bool(self):
        result = get_type_model(True)
        assert str(result) == "bool"

    def test_str(self):
        result = get_type_model("hello")
        assert str(result) == "str"

    def test_none(self):
        result = get_type_model(None)
        assert str(result) == "None"

    def test_numpy_int64(self):
        result = get_type_model(np.int64(7))
        assert "int64" in str(result)

    def test_numpy_float64(self):
        result = get_type_model(np.float64(2.7))
        assert "float64" in str(result)

    def test_numpy_array(self):
        arr = np.zeros((3, 4))
        result = get_type_model(arr)
        assert isinstance(result, ArrayType)
        assert result.shape == (3, 4)

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = get_type_model(df)
        assert isinstance(result, DataFrameType)
        assert result.n_rows == 2
        assert len(result.columns) == 2

    def test_series(self):
        s = pd.Series([1, 2, 3])
        result = get_type_model(s)
        assert isinstance(result, SeriesType)
        assert result.length == 3

    def test_dict(self):
        result = get_type_model({"a": 1, "b": 2})
        assert isinstance(result, DictType)

    def test_empty_dict(self):
        result = get_type_model({})
        assert isinstance(result, DictType)
        assert result.key_types == []
        assert result.value_types == []

    def test_list(self):
        result = get_type_model([1, 2, 3])
        assert isinstance(result, SequenceType)
        assert result.kind == "List"

    def test_tuple(self):
        result = get_type_model((1, 2, 3))
        assert isinstance(result, SequenceType)
        assert result.kind == "Tuple"

    def test_set(self):
        result = get_type_model({1, 2, 3})
        assert isinstance(result, SetType)

    def test_empty_set(self):
        result = get_type_model(set())
        assert isinstance(result, SetType)

    def test_deque(self):
        result = get_type_model(deque([1, 2, 3]))
        assert isinstance(result, SequenceType)
        assert result.kind == "Deque"

    def test_empty_deque(self):
        result = get_type_model(deque())
        assert isinstance(result, SequenceType)

    def test_function(self):
        def foo(x: int, y: float = 1.0) -> str:
            return str(x + y)
        result = get_type_model(foo)
        assert isinstance(result, FunctionType)
        assert result.name == "foo"

    def test_function_no_annotations(self):
        def bar(x, y):
            return x + y
        result = get_type_model(bar)
        assert isinstance(result, FunctionType)

    def test_type_class_int(self):
        """Type object for int."""
        result = get_type_model(int)
        assert str(result) == "int"

    def test_type_class_none_type(self):
        """type(None) returns AtomicType None."""
        result = get_type_model(type(None))
        assert str(result) == "None"

    def test_type_class_any(self):
        """Any type returns AtomicType Any."""
        result = get_type_model(Any)
        assert str(result) == "Any"

    def test_type_class_custom(self):
        """Custom class returns FallbackType."""
        class MyClass:
            pass
        result = get_type_model(MyClass)
        assert isinstance(result, FallbackType)
        assert result.class_name == "MyClass"

    def test_circular_reference(self):
        """Circular reference is handled gracefully."""
        lst = [1, 2]
        lst.append(lst)
        result = get_type_model(lst)
        # Should not infinite loop
        assert result is not None

    def test_custom_object(self):
        """Custom object without special handling returns FallbackType."""
        class MyObj:
            pass
        result = get_type_model(MyObj())
        assert isinstance(result, FallbackType)
        assert result.class_name == "MyObj"

    def test_numpy_scalar_type_class(self):
        """numpy scalar type class."""
        result = get_type_model(np.float64)
        assert "float64" in str(result)
