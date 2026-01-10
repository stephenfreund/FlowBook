import pytest
from dataclasses import dataclass
from types import SimpleNamespace
from hypothesis import given, strategies as st
from flowbook.kernel.equality import user_ns_diff


import numpy as np
import pandas as pd


# ======================================================================
# Example‐based tests
# ======================================================================


def test_simple_scalars_equal():
    a = {"x": 1, "y": "hello"}
    b = {"x": 1, "y": "hello"}
    assert user_ns_diff(a, b) == {}


def test_simple_scalars_not_equal():
    a = {"x": 1}
    b = {"x": 2}
    errs = user_ns_diff(a, b)
    assert "x" in errs
    assert "Value mismatch" in errs["x"]


def test_aliasing_preserved():
    obj = [1, 2, 3]
    a = {"x": obj, "y": obj}
    obj2 = [1, 2, 3]
    b = {"x": obj2, "y": obj2}
    assert user_ns_diff(a, b) == {}

    b_bad = {"x": [1, 2, 3], "y": [1, 2, 3]}
    errs = user_ns_diff(a, b_bad)
    assert "x" in errs or "y" in errs
    assert "Aliasing mismatch" in next(iter(errs.values()))


def test_nested_structures_equal():
    shared = [42]
    a = {"a": [shared, {"b": shared}]}
    shared2 = [42]
    b = {"a": [shared2, {"b": shared2}]}
    assert user_ns_diff(a, b) == {}


def test_cycles():
    a = {}
    a["self"] = a
    b = {}
    b["self"] = b
    assert user_ns_diff(a, b) == {}

    b2 = {}
    b2["self"] = {}
    errs = user_ns_diff(a, b2)
    assert "self" in errs
    # assert "Aliasing mismatch" in errs["self"]


# --- NumPy -------------------------------------------------------------


def test_numpy_equal():
    arr1 = np.arange(5)
    arr2 = arr1
    a = {"x": arr1, "y": arr2}
    arr3 = np.arange(5)
    b = {"x": arr3, "y": arr3}
    assert user_ns_diff(a, b) == {}


def test_numpy_not_equal():
    a = {"x": np.arange(5)}
    b = {"x": np.arange(5) + 1}
    errs = user_ns_diff(a, b)
    assert "x" in errs
    assert "NumPy arrays differ" in errs["x"]


def test_numpy_aliasing():
    arr = np.zeros((2, 2))
    a = {"a": arr, "b": arr}
    arr2 = np.zeros((2, 2))
    b = {"a": arr2, "b": arr2}
    assert user_ns_diff(a, b) == {}

    b_bad = {"a": np.zeros((2, 2)), "b": np.zeros((2, 2))}
    errs = user_ns_diff(a, b_bad)
    assert any("Aliasing mismatch" in msg for msg in errs.values())


# --- Pandas ------------------------------------------------------------


def test_pandas_series_equal():
    s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
    s2 = s1
    a = {"s": s1, "alias": s2}

    s1b = pd.Series([1, 2, 3], index=["a", "b", "c"])
    s2b = s1b
    b = {"s": s1b, "alias": s2b}
    assert user_ns_diff(a, b) == {}


def test_pandas_dataframe_not_equal():
    df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pd.DataFrame({"x": [1, 2], "y": [3, 5]})
    a = {"df": df1}
    b = {"df": df2}
    errs = user_ns_diff(a, b)
    assert "df" in errs
    assert "DataFrame differ" in errs["df"]


def test_pandas_index_equal():
    idx1 = pd.Index(["a", "b", "c"])
    idx2 = idx1
    a = {"idx": idx1, "alias": idx2}

    idx3 = pd.Index(["a", "b", "c"])
    idx4 = idx3
    b = {"idx": idx3, "alias": idx4}
    assert user_ns_diff(a, b) == {}


# --- Dataclasses / Objects ---------------------------------------------


@dataclass
class D:
    x: int
    y: list


def test_dataclass_equal():
    d1 = D(1, [1, 2])
    d2 = d1
    a = {"d": d1, "alias": d2}

    d1b = D(1, [1, 2])
    d2b = d1b
    b = {"d": d1b, "alias": d2b}
    assert user_ns_diff(a, b) == {}


def test_slots_objects_equal():
    class S:
        __slots__ = ("x", "y")

        def __init__(self, x, y):
            self.x = x
            self.y = y

    s1 = S(1, 2)
    s2 = s1
    a = {"obj": s1, "alias": s2}
    s3 = S(1, 2)
    s4 = s3
    b = {"obj": s3, "alias": s4}
    assert user_ns_diff(a, b) == {}


def test_namespace_equal():
    n1 = SimpleNamespace(a=1, b=[2, 3])
    n2 = n1
    a = {"n": n1, "alias": n2}
    n3 = SimpleNamespace(a=1, b=[2, 3])
    n4 = n3
    b = {"n": n3, "alias": n4}
    assert user_ns_diff(a, b) == {}


# --- Misc container types ---------------------------------------------


# def test_dicts_with_tuple_keys():
#     a = {("x", 1): 10}
#     b = {("x", 1): 10}
#     assert user_ns_diff(a, b) == {}


def test_sets_equal():
    a = {"x": {1, 2, 3}}
    b = {"x": {3, 2, 1}}
    assert user_ns_diff(a, b) == {}


def test_mismatched_types_fail():
    a = {"x": [1, 2, 3]}
    b = {"x": (1, 2, 3)}
    errs = user_ns_diff(a, b)
    assert "x" in errs
    # assert "Type mismatch" in errs["x"]


def test_missing_key():
    a = {"x": 1}
    b = {"x": 1, "y": 2}
    errs = user_ns_diff(a, b)
    assert "y" in errs or "x" in errs
    assert "present only" in next(iter(errs.values()))


def test_functions_equal():
    def f(x):
        return x + 1

    a = {"f": f}
    b = {"f": f}
    assert user_ns_diff(a, b) == {}


def test_functions_not_equal():
    def f(x):
        return x + 1

    def g(x):
        return x + 2

    a = {"f": f}
    b = {"f": g}
    errs = user_ns_diff(a, b)
    assert "f" in errs
    assert "Value mismatch" in errs["f"]


# ======================================================================
# Property‐based tests (Hypothesis)
# ======================================================================


def make_recursive_strategy():
    base = st.one_of(st.none(), st.booleans(), st.integers(), st.text())

    def extend_fn(children):
        return st.one_of(
            st.lists(children, max_size=3),
            st.tuples(children, children),
            st.dictionaries(st.text(), children, max_size=3),
        )

    return st.recursive(base, extend_fn, max_leaves=10)


@given(make_recursive_strategy())
def test_reflexivity(x):
    a = {"x": x}
    b = {"x": x}
    assert user_ns_diff(a, b) == {}


@given(make_recursive_strategy(), make_recursive_strategy())
def test_symmetry(x, y):
    a = {"x": x}
    b = {"x": y}
    assert (user_ns_diff(a, b) == {}) == (user_ns_diff(b, a) == {})


@given(make_recursive_strategy())
def test_aliasing_pattern_preserved(x):
    a = {"v1": x, "v2": x}
    b_good = {"v1": x, "v2": x}
    assert user_ns_diff(a, b_good) == {}

    if isinstance(x, (list, dict, set)):
        b_bad = {"v1": x, "v2": x.copy()}
        errs = user_ns_diff(a, b_bad)
        assert "v1" in errs or "v2" in errs
        assert "Aliasing mismatch" in next(iter(errs.values()))


# ======================================================================
# Nan handling
# ======================================================================

# test_user_ns_diff_nan.py
import math
import pytest


def test_scalar_python_nan_equal():
    a = {"x": float("nan")}
    b = {"x": float("nan")}
    assert user_ns_diff(a, b) == {}


def test_scalar_nan_vs_number_diff():
    a = {"x": float("nan")}
    b = {"x": 1.0}
    errs = user_ns_diff(a, b)
    assert "x" in errs
    # don't assert exact message text; just that it flags a difference
    assert isinstance(errs["x"], str) and errs["x"]


@pytest.mark.parametrize(
    "dtype", [float, np.float32, np.float64]
)  # numpy added below if available
def test_numpy_scalar_nan_equal_with_python_float(dtype):
    a = {"x": dtype("nan")}
    b = {"x": dtype("nan")}
    assert user_ns_diff(a, b) == {}


def test_list_with_nans_equal():
    a = {"x": [1.0, float("nan"), 3.0]}
    b = {"x": [1.0, float("nan"), 3.0]}
    assert user_ns_diff(a, b) == {}


def test_list_nan_position_mismatch():
    a = {"x": [1.0, float("nan"), 3.0]}
    b = {"x": [1.0, 2.0, 3.0]}
    errs = user_ns_diff(a, b)
    assert "x" in errs


# ---------- NumPy tests ----------


@pytest.mark.parametrize("ctor", [np.float32, np.float64])
def test_numpy_scalar_nan_equal(ctor):
    a = {"x": ctor("nan")}
    b = {"x": ctor("nan")}
    assert user_ns_diff(a, b) == {}


def test_numpy_array_nan_equal():
    a = {"x": np.array([1.0, np.nan, 3.0])}
    b = {"x": np.array([1.0, np.nan, 3.0])}
    assert user_ns_diff(a, b) == {}


def test_numpy_array_value_mismatch_ignores_nan_pairs_but_catches_real_diffs():
    a = {"x": np.array([1.0, np.nan, 3.0])}
    b = {"x": np.array([1.0, np.nan, 4.0])}
    errs = user_ns_diff(a, b)
    assert "x" in errs


def test_numpy_array_shape_mismatch():
    a = {"x": np.array([1.0, np.nan])}
    b = {"x": np.array([[1.0, np.nan]])}
    errs = user_ns_diff(a, b)
    assert "x" in errs


# ---------- pandas tests ----------
pd = pytest.importorskip("pandas")


def test_pandas_series_nan_equal():
    a = {"s": pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])}
    b = {"s": pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])}
    assert user_ns_diff(a, b) == {}


def test_pandas_series_nan_equal_with_dtype_variation():
    # ensure NaN equality survives dtype differences only if types match per your function
    s1 = pd.Series([1.0, np.nan], dtype="float64")
    s2 = pd.Series([1.0, np.nan], dtype="float64")
    assert user_ns_diff({"s": s1}, {"s": s2}) == {}


def test_pandas_series_value_mismatch():
    a = {"s": pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])}
    b = {"s": pd.Series([1.0, np.nan, 4.0], index=["a", "b", "c"])}
    errs = user_ns_diff(a, b)
    assert "s" in errs


def test_pandas_dataframe_nan_equal():
    a = {
        "df": pd.DataFrame(
            {"A": [1.0, np.nan], "B": [np.nan, 2.0]}, index=pd.Index(["r1", "r2"])
        )
    }
    b = {
        "df": pd.DataFrame(
            {"A": [1.0, np.nan], "B": [np.nan, 2.0]}, index=pd.Index(["r1", "r2"])
        )
    }
    assert user_ns_diff(a, b) == {}


def test_pandas_dataframe_value_mismatch():
    a = {"df": pd.DataFrame({"A": [1.0, np.nan]})}
    b = {"df": pd.DataFrame({"A": [2.0, np.nan]})}
    errs = user_ns_diff(a, b)
    assert "df" in errs


def test_pandas_index_nan_equal():
    idx1 = pd.Index([1, np.nan, 3])
    idx2 = pd.Index([1, np.nan, 3])
    a = {"i": idx1}
    b = {"i": idx2}
    assert user_ns_diff(a, b) == {}


def test_pandas_dataframe_nan_in_columns_and_index_equal():
    df1 = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        columns=pd.Index(["A", np.nan]),
        index=pd.Index([np.nan, "r2"]),
    )
    df2 = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        columns=pd.Index(["A", np.nan]),
        index=pd.Index([np.nan, "r2"]),
    )
    assert user_ns_diff({"df": df1}, {"df": df2}) == {}


# ======================================================================
# Main runner
# ======================================================================


def main():
    """
    Allows this test file to be run directly:
        python test_user_ns_equal.py
    """
    import sys

    print("Running pytest on this file...")
    sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
