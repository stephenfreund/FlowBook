"""
Custom deepcopy implementation with special handling for pandas and functions.

This module extends Python's standard copy.deepcopy with custom handlers for:
- pandas DataFrame: shallow copy with CoW + deep copy of object columns
- pandas Series: shallow copy with CoW + deep copy if object dtype
- FunctionType: deep copy of closure contents and mutable defaults

The implementation follows the same dispatch pattern as the standard library's
copy module for consistency and extensibility.
"""

from __future__ import annotations

import types
from typing import Any

import pandas as pd
from pandas.api.types import infer_dtype

from data_ferret.util.output import log
from data_ferret.kernel.column_tracking import suspend_column_tracking


# Sentinel for memo lookups
_nil = []

# Global flag to control dtype conversion (set by Checkpoints.__init__)
_convert_object_dtypes = True


def deepcopy(x, memo=None):
    """Deep copy operation on arbitrary Python objects.

    This is identical to copy.deepcopy except for special handling of:
    - pandas DataFrames: shallow copy + deep copy object columns
    - pandas Series: shallow copy + deep copy if object dtype
    - Functions: deep copy closure and mutable defaults

    Args:
        x: Object to deep copy
        memo: Dictionary tracking already-copied objects (for circular references)

    Returns:
        Deep copy of x
    """
    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    cls = type(x)

    copier = _deepcopy_dispatch.get(cls)
    if copier is not None:
        y = copier(x, memo)
    else:
        if issubclass(cls, type):
            y = _deepcopy_atomic(x, memo)
        else:
            copier = getattr(x, "__deepcopy__", None)
            if copier is not None:
                y = copier(memo)
            else:
                reductor = getattr(x, "__reduce_ex__", None)
                if reductor is not None:
                    rv = reductor(4)
                else:
                    reductor = getattr(x, "__reduce__", None)
                    if reductor:
                        rv = reductor()
                    else:
                        raise TypeError(
                            f"un(deep)copyable object of type {cls}"
                        )
                if isinstance(rv, str):
                    y = x
                else:
                    y = _reconstruct(x, memo, *rv)

    # If is its own copy, don't memoize.
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo)  # Make sure x lives at least as long as d
    return y


_deepcopy_dispatch = d = {}


def _deepcopy_atomic(x, memo):
    """Return object unchanged (for immutable types)."""
    return x


# Register atomic types (immutable, safe to share)
d[type(None)] = _deepcopy_atomic
d[type(Ellipsis)] = _deepcopy_atomic
d[type(NotImplemented)] = _deepcopy_atomic
d[int] = _deepcopy_atomic
d[float] = _deepcopy_atomic
d[bool] = _deepcopy_atomic
d[complex] = _deepcopy_atomic
d[bytes] = _deepcopy_atomic
d[str] = _deepcopy_atomic
d[types.CodeType] = _deepcopy_atomic
d[type] = _deepcopy_atomic
d[range] = _deepcopy_atomic
d[types.BuiltinFunctionType] = _deepcopy_atomic
# NOTE: FunctionType is NOT atomic - we override it below
d[property] = _deepcopy_atomic


def _deepcopy_list(x, memo):
    """Deep copy a list."""
    y = []
    memo[id(x)] = y
    append = y.append
    for a in x:
        append(deepcopy(a, memo))
    return y


d[list] = _deepcopy_list


def _deepcopy_tuple(x, memo):
    """Deep copy a tuple (or return original if contents unchanged)."""
    y = [deepcopy(a, memo) for a in x]
    # We're not going to put the tuple in the memo, but it's still important we
    # check for it, in case the tuple contains recursive mutable structures.
    try:
        return memo[id(x)]
    except KeyError:
        pass
    for k, j in zip(x, y):
        if k is not j:
            y = tuple(y)
            break
    else:
        y = x
    return y


d[tuple] = _deepcopy_tuple


def _deepcopy_dict(x, memo):
    """Deep copy a dictionary."""
    y = {}
    memo[id(x)] = y
    for key, value in x.items():
        y[deepcopy(key, memo)] = deepcopy(value, memo)
    return y


d[dict] = _deepcopy_dict


def _deepcopy_method(x, memo):
    """Copy instance methods."""
    return type(x)(x.__func__, deepcopy(x.__self__, memo))


d[types.MethodType] = _deepcopy_method


# Custom handlers for pandas and functions


def _has_mutable_defaults(func: types.FunctionType) -> bool:
    """
    Check if a function has any mutable default argument values.

    Mutable defaults (like [], {}) are a common source of bugs and need
    to be deep copied to ensure isolation across checkpoints.

    Args:
        func: Function to check

    Returns:
        True if the function has any mutable default arguments
    """
    # Check positional defaults
    if func.__defaults__:
        for default in func.__defaults__:
            if isinstance(default, (list, dict, set)):
                return True
            # Check for other mutable types
            if hasattr(default, '__dict__') or hasattr(default, '__setitem__'):
                if not isinstance(default, (str, bytes, tuple, frozenset, range)):
                    return True

    # Check keyword-only defaults
    if func.__kwdefaults__:
        for default in func.__kwdefaults__.values():
            if isinstance(default, (list, dict, set)):
                return True
            if hasattr(default, '__dict__') or hasattr(default, '__setitem__'):
                if not isinstance(default, (str, bytes, tuple, frozenset, range)):
                    return True

    return False


def _deepcopy_function(func: types.FunctionType, memo: dict[int, Any]) -> types.FunctionType:
    """
    Deep copy a function, including its closure contents and mutable defaults.

    Standard deepcopy doesn't copy closure cell contents or mutable defaults,
    leaving the copied function referencing the same objects as the original.
    This function creates a true deep copy by:
    1. Deep copying each closure cell's contents using the shared memo
    2. Creating new cell objects with the copied contents
    3. Deep copying mutable default arguments
    4. Building a new function with the new closure and defaults

    Args:
        func: Function to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        New function with deep-copied closure and defaults
    """
    # If no closure and no mutable defaults, return the same function
    if func.__closure__ is None and not _has_mutable_defaults(func):
        return func

    # If no closure but has mutable defaults, need to copy defaults only
    if func.__closure__ is None:
        # Create the function first
        new_func = types.FunctionType(
            func.__code__,
            func.__globals__,
            func.__name__,
            func.__defaults__,
            None,
        )

        # Register in memo BEFORE any recursive operations
        memo[id(func)] = new_func

        # Now do the recursive deep copies
        new_defaults = (
            tuple(deepcopy(d, memo) for d in func.__defaults__)
            if func.__defaults__
            else None
        )
        new_kwdefaults = (
            {k: deepcopy(v, memo) for k, v in func.__kwdefaults__.items()}
            if func.__kwdefaults__
            else None
        )

        # Update with deep copied values
        new_func.__defaults__ = new_defaults
        new_func.__kwdefaults__ = new_kwdefaults
        new_func.__annotations__ = func.__annotations__.copy() if func.__annotations__ else {}
        # Deep copy the function's __dict__
        for k, v in func.__dict__.items():
            new_func.__dict__[k] = deepcopy(v, memo)
        new_func.__doc__ = func.__doc__

        return new_func

    # Create a temporary function and register it in memo first to handle circular references
    temp_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    # Register BEFORE recursive operations to handle circular references
    memo[id(func)] = temp_func

    # Now deep copy closure contents
    new_cells = []
    for cell in func.__closure__:
        try:
            copied_contents = deepcopy(cell.cell_contents, memo)
            new_cells.append(types.CellType(copied_contents))
        except ValueError:
            # Empty cell (variable referenced but not yet bound)
            new_cells.append(types.CellType())

    new_closure = tuple(new_cells)

    # Deep copy defaults (may contain mutable objects)
    new_defaults = (
        tuple(deepcopy(d, memo) for d in func.__defaults__)
        if func.__defaults__
        else None
    )
    new_kwdefaults = (
        {k: deepcopy(v, memo) for k, v in func.__kwdefaults__.items()}
        if func.__kwdefaults__
        else None
    )

    # Create the actual new function with deep-copied closure and defaults
    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,  # Share globals (modules, builtins, etc.)
        func.__name__,
        new_defaults,
        new_closure,
    )
    new_func.__kwdefaults__ = new_kwdefaults
    new_func.__annotations__ = func.__annotations__.copy() if func.__annotations__ else {}
    # Deep copy the function's __dict__
    for k, v in func.__dict__.items():
        new_func.__dict__[k] = deepcopy(v, memo)
    new_func.__doc__ = func.__doc__

    # Update memo to point to the actual function
    memo[id(func)] = new_func

    return new_func


d[types.FunctionType] = _deepcopy_function


def _convert_object_column_dtype(series: pd.Series) -> pd.Series:
    """
    Convert object dtype Series to specialized dtype if possible.

    This is done inline during deepcopy to ensure all DataFrames
    (including nested ones) get converted, not just top-level variables.

    Args:
        series: Series to convert

    Returns:
        Converted Series or original if conversion not possible
    """
    if series.dtype != object or series.empty:
        return series

    kind = infer_dtype(series, skipna=True)

    try:
        if kind in {"integer", "mixed-integer"}:
            return series.astype("Int64")
        elif kind in {"floating", "mixed-integer-float"}:
            return series.astype(float)
        elif kind == "decimal":
            return series.astype(float)
        elif kind == "complex":
            return series.astype(complex)
        elif kind == "string":
            return series.astype("category")
        elif kind == "boolean":
            return series.astype("boolean")
        elif kind in {"datetime64", "datetime", "date"}:
            return pd.to_datetime(series)
        elif kind in {"timedelta64", "timedelta"}:
            return pd.to_timedelta(series)
        elif kind == "categorical":
            return series.astype("category")
        else:
            # Mixed types or unknown - leave as object
            return series
    except (TypeError, ValueError, Exception):
        # Conversion failed - return original
        return series


def _deepcopy_dataframe(df: pd.DataFrame, memo: dict[int, Any]) -> pd.DataFrame:
    """
    Deep copy a DataFrame with special object column handling.

    Converts object columns to specialized dtypes (Int64, string, datetime64, etc.)
    on the original DataFrame before copying. Uses shallow copy with copy-on-write
    for non-object columns. For object columns that remain object dtype after
    conversion, deep copy to ensure mutable objects are isolated.

    Args:
        df: DataFrame to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        Deep copy of the DataFrame
    """
    obj_id = id(df)
    if obj_id in memo:
        return memo[obj_id]

    # Suspend column tracking during deepcopy to avoid recording internal accesses
    with suspend_column_tracking():
        # Convert object columns to specialized dtypes on the original DataFrame first
        if _convert_object_dtypes:
            for col in df.columns:
                if df[col].dtype == object:
                    converted = _convert_object_column_dtype(df[col])
                    if converted.dtype != object:
                        log(f"Converted column {col} from object to {converted.dtype}")
                        df[col] = converted

        # Shallow copy: CoW handles non-object columns efficiently
        df_copy = df.copy(deep=False)

        # Process remaining object columns: deep copy for mutable objects
        for col in df_copy.columns:
            if df_copy[col].dtype == object:
                num_rows = len(df_copy)
                if num_rows > 10000:
                    log(f"Deep copying large object column {col} with {num_rows:,} rows...")
                else:
                    log(f"Deep copying object column {col}")

                # Apply deep copy and explicitly preserve object dtype
                result = df_copy[col].apply(lambda x: deepcopy(x, memo))
                df_copy[col] = result.astype(object)

        memo[obj_id] = df_copy
        return df_copy


d[pd.DataFrame] = _deepcopy_dataframe


def _deepcopy_series(series: pd.Series, memo: dict[int, Any]) -> pd.Series:
    """
    Deep copy a Series with special object dtype handling.

    Converts object Series to specialized dtypes (Int64, string, datetime64, etc.)
    on the original Series before copying. Uses shallow copy with copy-on-write
    for non-object Series. For object Series that remain object dtype after
    conversion, deep copy to ensure mutable objects are isolated.

    Args:
        series: Series to copy
        memo: Shared memo dict for tracking copied objects

    Returns:
        Deep copy of the Series
    """
    obj_id = id(series)
    if obj_id in memo:
        return memo[obj_id]

    # Suspend column tracking during deepcopy to avoid recording internal accesses
    with suspend_column_tracking():
        # Convert object Series to specialized dtype on the original Series first
        # (only if conversion is enabled globally)
        if _convert_object_dtypes and series.dtype == object:
            # Try to convert to specialized dtype first
            converted = _convert_object_column_dtype(series)

            if converted.dtype != object:
                # Successfully converted - update original Series
                log(f"Converted Series from object to {converted.dtype}")
                series = converted

        # Shallow copy: CoW handles non-object Series efficiently
        series_copy = series.copy(deep=False)

        # Process if still object dtype: deep copy for mutable objects
        if series_copy.dtype == object:
            # Still object dtype - need to deep copy for mutable objects
            num_rows = len(series_copy)
            if num_rows > 10000:
                log(f"Deep copying large object Series with {num_rows:,} rows...")
            else:
                log(f"Deep copying object Series")

            # Apply deep copy and explicitly preserve object dtype
            result = series_copy.apply(lambda x: deepcopy(x, memo))
            series_copy = result.astype(object)

        memo[obj_id] = series_copy
        return series_copy


d[pd.Series] = _deepcopy_series

del d  # Clean up namespace


def _keep_alive(x, memo):
    """Keeps a reference to the object x in the memo.

    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)] = [x]


def _reconstruct(x, memo, func, args,
                 state=None, listiter=None, dictiter=None):
    """Reconstruct an object from pickle-like reduction."""
    deep = memo is not None
    if deep and args:
        args = tuple(deepcopy(arg, memo) for arg in args)
    y = func(*args)
    if deep:
        memo[id(x)] = y

    if state is not None:
        if deep:
            state = deepcopy(state, memo)
        if hasattr(y, '__setstate__'):
            y.__setstate__(state)
        else:
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            else:
                slotstate = None
            if state is not None:
                y.__dict__.update(state)
            if slotstate is not None:
                for key, value in slotstate.items():
                    setattr(y, key, value)

    if listiter is not None:
        if deep:
            for item in listiter:
                item = deepcopy(item, memo)
                y.append(item)
        else:
            for item in listiter:
                y.append(item)
    if dictiter is not None:
        if deep:
            for key, value in dictiter:
                key = deepcopy(key, memo)
                value = deepcopy(value, memo)
                y[key] = value
        else:
            for key, value in dictiter:
                y[key] = value
    return y
