from dataclasses import is_dataclass, asdict
from typing import Any, Callable, Dict, Tuple, Set


def user_ns_diff(
    a: Dict[str, Any], b: Dict[str, Any], ignore_keys: Set[str] = set()
) -> Dict[str, str]:
    """
    Compare two Jupyter user_ns dictionaries and return a map of keys that differ,
    with detailed messages explaining why (including indices and values).

    Returns an empty dict if namespaces are fully equal (including aliasing).
    """
    errors: Dict[str, str] = {}

    # Check same top-level keys
    keys_a = set(a.keys()) - ignore_keys
    keys_b = set(b.keys()) - ignore_keys
    if keys_a != keys_b:
        for k in keys_a - keys_b:
            errors[k] = f"'removed"
        for k in keys_b - keys_a:
            errors[k] = f"'added"

    shared_keys = keys_a & keys_b - ignore_keys

    # Maps for alias tracking
    map_ab: Dict[int, Any] = {}
    map_ba: Dict[int, Any] = {}

    def is_atomic(x: Any) -> bool:
        # Keep your original notion of atomic
        return isinstance(
            x, (type(None), bool, int, float, complex, str, bytes, Callable)
        )

    def is_nan_value(v: Any) -> bool:
        # True if v is a NaN-like scalar (Python float NaN or NumPy scalar NaN)
        import math

        try:
            import numpy as np  # noqa: F401
        except Exception:
            np = None  # type: ignore
        # Python float NaN
        if isinstance(v, float):
            return math.isnan(v)
        # NumPy scalar NaN
        if np is not None:
            try:
                # bool(...) to handle numpy.bool_ return type
                return bool(np.isnan(v))  # works for numpy floating scalars
            except Exception:
                pass
        return False

    def try_numpy_eq(x: Any, y: Any, path: str) -> Tuple[bool, str] | None:
        try:
            import numpy as np
        except ImportError:
            return None
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            map_ab[id(x)] = y
            map_ba[id(y)] = x
            if x.shape != y.shape:
                return False, f"NumPy shape mismatch at {path}: {x.shape} vs {y.shape}"
            # Treat NaNs as equal
            try:
                if np.array_equal(x, y, equal_nan=True):
                    return True, ""
                # find first mismatch index for message
                eq = np.equal(x, y) | (np.isnan(x) & np.isnan(y))
            except TypeError:
                # Fallback for dtypes not supporting isnan
                eq = np.equal(x, y)
            if eq.all():
                return True, ""
            idx = tuple(dim[0] for dim in np.where(~eq))
            return (
                False,
                f"NumPy arrays differ at {path}{idx}: {x[idx]!r} != {y[idx]!r}",
            )
        # NumPy scalar vs scalar: treat NaN == NaN
        if (
            hasattr(np, "floating")
            and isinstance(x, np.floating)
            and isinstance(y, np.floating)
        ):
            if is_nan_value(x) and is_nan_value(y):
                return True, ""
            return (
                (True, "")
                if x == y
                else (False, f"Value mismatch at {path}: {x!r} != {y!r}")
            )
        return None

    def try_pandas_eq(x: Any, y: Any, path: str) -> Tuple[bool, str] | None:
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            return None
        types = (pd.Series, pd.DataFrame, pd.Index)
        if isinstance(x, types) and isinstance(y, types):
            if type(x) is not type(y):
                return (
                    False,
                    f"Pandas type mismatch at {path}: {type(x).__name__} vs {type(y).__name__}",
                )
            map_ab[id(x)] = y
            map_ba[id(y)] = x
            if isinstance(x, pd.DataFrame):
                assert isinstance(y, pd.DataFrame)
                # use .equals for columns/index so NaN==NaN there too
                if not x.columns.equals(y.columns):
                    removed = set(x.columns) - set(y.columns)
                    added = set(y.columns) - set(x.columns)
                    return (
                        False,
                        f"DataFrame columns differ at {path}: added {added} removed {removed}",
                    )
                if not x.index.equals(y.index):

                    return (
                        False,
                        f"DataFrame index differ at {path}: {list(x.index)} vs {list(y.index)}",
                    )
                # .equals treats NaN==NaN
                if x.equals(y):
                    return True, ""
                # Build a mismatch mask that treats NaN==NaN as equal
                comp = x.eq(y) | (x.isna() & y.isna())
                mismatch = ~comp.values
                i, j = np.where(mismatch)[0][0], np.where(mismatch)[1][0]
                col = x.columns[j]
                idx = x.index[i]
                return False, (
                    f"DataFrame differ at {path}[{idx!r}, '{col}']: "
                    f"{x.iat[i, j]!r} != {y.iat[i, j]!r}"
                )
            # Series
            if isinstance(x, pd.Series):
                if not x.index.equals(y.index):
                    first_diff = next(
                        (i for i, (a, b) in enumerate(zip(x.index, y.index)) if a != b),
                        "end",
                    )
                    if first_diff != "end":
                        return (
                            False,
                            f"Series index differ at {path}: first diff at {first_diff}",
                        )
                    else:
                        return (
                            False,
                            f"Series index differ at {path}: {repr(x.index)} vs {repr(y.index)}",
                        )
                if x.equals(y):  # treats NaN==NaN
                    return True, ""
                comp = x.eq(y) | (x.isna() & y.isna())
                diff_pos = (~comp).to_numpy().nonzero()[0][0]
                diff_idx = x.index[diff_pos]
                return (
                    False,
                    f"Series differ at {path}[{diff_idx!r}]: {x.loc[diff_idx]!r} != {y.loc[diff_idx]!r}",
                )
            # Index
            if isinstance(x, pd.Index):
                if x.equals(y):  # treats NaN==NaN
                    return True, ""
                # Show only the first difference instead of the whole lists
                for i, (a, b) in enumerate(zip(x, y)):
                    if a != b:
                        return False, f"Index differ at {path}[{i}]: {a!r} != {b!r}"
                # If lengths differ, show the first extra element
                if len(x) != len(y):
                    shorter, longer, name = (
                        (x, y, "b") if len(x) < len(y) else (y, x, "a")
                    )
                    idx = len(shorter)
                    return (
                        False,
                        f"Index differ at {path}[{idx}]: {name} has extra value {longer[idx]!r}",
                    )
        return None

    def equal_iso(x: Any, y: Any, path: str) -> Tuple[bool, str]:
        ix, iy = id(x), id(y)
        if ix in map_ab:
            return (
                (True, "")
                if map_ab[ix] is y
                else (False, f"Aliasing mismatch at {path}")
            )
        if iy in map_ba:
            return (
                (True, "")
                if map_ba[iy] is x
                else (False, f"Aliasing mismatch at {path}")
            )

        # Atomic/scalar path (treat NaN==NaN)
        if is_atomic(x) and is_atomic(y):
            if type(x) is not type(y):
                return (
                    False,
                    f"Type mismatch at {path}: {type(x).__name__} vs {type(y).__name__}",
                )
            # special-case NaNs
            if is_nan_value(x) and is_nan_value(y):
                return True, ""
            if x != y:
                return False, f"Value mismatch at {path}: {x!r} != {y!r}"
            return True, ""

        # NumPy / pandas paths (handle NaN in helpers)
        np_res = try_numpy_eq(x, y, path)
        if np_res is not None:
            return np_res
        pd_res = try_pandas_eq(x, y, path)
        if pd_res is not None:
            return pd_res

        map_ab[ix] = y
        map_ba[iy] = x

        if (isinstance(x, list) and isinstance(y, list)) or (
            isinstance(x, tuple) and isinstance(y, tuple)
        ):
            if len(x) != len(y):
                return False, f"Length mismatch at {path}: {len(x)} vs {len(y)}"
            for idx, (xi, yi) in enumerate(zip(x, y)):
                ok, msg = equal_iso(xi, yi, f"{path}[{idx}]")
                if not ok:
                    return False, msg
            return True, ""

        if (isinstance(x, set) and isinstance(y, set)) or (
            isinstance(x, frozenset) and isinstance(y, frozenset)
        ):
            if len(x) != len(y):
                return False, f"Set size mismatch at {path}: {len(x)} vs {len(y)}"
            sx = sorted(map(str, x))
            sy = sorted(map(str, y))
            if sx != sy:
                return False, f"Set contents differ at {path}: {sx} vs {sy}"
            return True, ""

        if isinstance(x, dict) and isinstance(y, dict):
            if set(x.keys()) != set(y.keys()):
                removed = set(x.keys()) - set(y.keys())
                added = set(y.keys()) - set(x.keys())
                return (
                    False,
                    f"Dict keys differ at {path}: added {added} removed {removed}",
                )
            for key in x:
                ok, msg = equal_iso(x[key], y[key], f"{path}['{key}']")
                if not ok:
                    return False, msg
            return True, ""

        if is_dataclass(x):
            return equal_iso(asdict(x), asdict(y), path)

        if hasattr(x, "__dict__") and hasattr(y, "__dict__"):
            return equal_iso(vars(x), vars(y), path)
        if hasattr(x, "__slots__") and hasattr(y, "__slots__"):
            slots = (
                x.__slots__
                if isinstance(x.__slots__, (list, tuple))
                else (x.__slots__,)
            )
            for slot in slots:
                ok, msg = equal_iso(
                    getattr(x, slot), getattr(y, slot), f"{path}.{slot}"
                )
                if not ok:
                    return False, msg
            return True, ""

        try:
            # Final fallback (scalars that slipped through; treat NaN==NaN)
            if is_nan_value(x) and is_nan_value(y):
                return True, ""
            if type(x) is not type(y):
                return (
                    False,
                    f"Type mismatch at {path}: {type(x).__name__} vs {type(y).__name__}",
                )
            if x != y:
                return False, f"Fallback mismatch at {path}: {x!r} != {y!r}"
            return True, ""
        except Exception as e:
            return False, f"Exception comparing at {path}: {e}"

    # top-level compare
    for k in shared_keys:
        ok, msg = equal_iso(a[k], b[k], f"key '{k}'")
        if not ok:
            errors[k] = msg
    return errors


if __name__ == "__main__":
    import numpy as np, pandas as pd

    # aliasing check with ndarray
    x = np.arange(5)
    y = x  # alias
    a = {"arr": x, "alias": y}

    x2 = np.arange(5)
    y2 = x2
    b = {"arr": x2, "alias": y2}
    assert user_ns_diff(a, b) == {}

    # fails if aliasing pattern differs
    b_bad = {"arr": np.arange(5), "alias": np.arange(5)}
    assert user_ns_diff(a, b_bad) != {}

    # pandas Series / DataFrame
    s1 = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))
    s2 = s1  # alias
    df1 = pd.DataFrame({"u": [1, 2], "v": [3, 4]}, index=pd.Index(["r", "s"]))
    a = {"s": s1, "s_alias": s2, "df": df1}

    s1b = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))
    s2b = s1b
    df1b = pd.DataFrame({"u": [1, 2], "v": [3, 4]}, index=pd.Index(["r", "s"]))
    b = {"s": s1b, "s_alias": s2b, "df": df1b}
    assert user_ns_diff(a, b) == {}

    # Index comparison
    a = {"idx": df1.index}
    b = {"idx": df1b.index.copy()}
    assert user_ns_diff(a, b) == {}
