"""
Custom deepcopy implementation with special handling for pandas and functions.

This module extends Python's standard copy.deepcopy with custom handlers for:
- pandas DataFrame: shallow copy with CoW + deep copy of object columns
- pandas Series: shallow copy with CoW + deep copy if object dtype
- FunctionType: deep copy of closure contents and mutable defaults

The implementation follows the same dispatch pattern as the standard library's
copy module for consistency and extensibility.

MultiIndex Column Support
-------------------------
DataFrames with MultiIndex columns (hierarchical column labels) are fully
supported. When iterating over DataFrame columns, we use positional indexing
(`.iloc[:, i]`) to read columns, which always returns a Series regardless of
column name type. For writing, we use column name indexing (`df[col]`) to
preserve dtype correctly.

This approach handles edge cases where `df[tuple]` might return a DataFrame
instead of a Series when pandas interprets the tuple as a partial key in a
MultiIndex.

Example of supported MultiIndex DataFrame:
    >>> arrays = [['A', 'A', 'B'], ['one', 'two', 'one']]
    >>> tuples = list(zip(*arrays))
    >>> columns = pd.MultiIndex.from_tuples(tuples)
    >>> df = pd.DataFrame([[1, 2, 3]], columns=columns)
    >>> memo = {}
    >>> copy = deepcopy(df, memo)  # Works correctly
"""

from __future__ import annotations

import datetime
import decimal
import os
import types
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

from data_ferret.util.output import log, timer
from data_ferret.kernel.column_tracking import suspend_column_tracking
from data_ferret.kernel.opaque import OpaqueRegistry


# Sentinel for memo lookups
_nil = []

_convert_object_dtypes = True


# ============================================================================
# Immutability checking for preserve mode
# ============================================================================

def is_immutable(obj: Any, _seen: set[int] | None = None, max_depth: int = 10) -> bool:
    """
    Check if an object is deeply immutable (safe to share references).

    Unlike check_deepcopyable, this checks if the object cannot be mutated,
    meaning a shallow copy is sufficient for isolation.

    Args:
        obj: Any Python object to check
        _seen: Internal set for cycle detection (do not pass externally)
        max_depth: Maximum recursion depth for container types

    Returns:
        True if the object is immutable, False otherwise
    """
    # Fast path for common immutable primitives
    if obj is None or obj is True or obj is False:
        return True

    obj_type = type(obj)

    # Immutable primitive types
    if obj_type in (int, float, complex, str, bytes, range):
        return True

    # datetime/time types are immutable
    if obj_type in (
        datetime.date,
        datetime.time,
        datetime.datetime,
        datetime.timedelta,
    ):
        return True

    # Decimal is immutable
    if obj_type is decimal.Decimal:
        return True

    # NumPy scalars are immutable
    if isinstance(obj, np.generic):
        return True

    # Pandas immutable types
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return True
    if obj is pd.NA:
        return True

    # tuple - immutable if all elements are immutable
    if obj_type is tuple:
        if max_depth <= 0:
            return False  # Too deep, assume mutable
        # Handle cycles
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return True  # Already checking this object
        _seen.add(obj_id)
        return all(is_immutable(item, _seen, max_depth - 1) for item in obj)

    # frozenset - immutable if all elements are immutable
    if obj_type is frozenset:
        if max_depth <= 0:
            return False
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return True
        _seen.add(obj_id)
        return all(is_immutable(item, _seen, max_depth - 1) for item in obj)

    # Everything else is considered mutable (list, dict, set, ndarray, custom objects, etc.)
    return False


# These infer_dtype results guarantee all elements are immutable
_IMMUTABLE_INFERRED_KINDS = frozenset({
    "string",
    "bytes",
    "integer",
    "mixed-integer",       # still all ints
    "floating",
    "mixed-integer-float",  # ints and floats - both immutable
    "decimal",
    "complex",
    "boolean",
    "datetime64",
    "datetime",
    "date",
    "timedelta64",
    "timedelta",
    "time",
})


def _object_column_is_all_immutable(series: pd.Series, col_name: str = None) -> bool:
    """
    Check if ALL values in an object column are immutable.

    Uses fast path via infer_dtype for homogeneous columns,
    falls back to element-by-element check only for mixed columns.

    Args:
        series: Series to check (must be object dtype)
        col_name: Optional column name for logging

    Returns:
        True if all values are immutable, False otherwise
    """
    if series.dtype != object or series.empty:
        return True

    col_label = f" ({col_name})" if col_name else ""
    n_rows = len(series)

    with timer(key="immutability_check", message=f"[deepcopy] Immutability check{col_label} ({n_rows} rows)"):
        # FAST PATH: infer_dtype tells us the homogeneous type - O(1)
        kind = infer_dtype(series, skipna=True)
        if kind in _IMMUTABLE_INFERRED_KINDS:
            log(f"Immutability fast path{col_label}: {kind}")
            return True  # All elements are known-immutable types

        # SLOW PATH: mixed or unknown - check element by element - O(n)
        if kind == "mixed":
            log(f"Immutability slow path{col_label}: {kind}, checking {n_rows} elements")
            for val in series.dropna():
                if not is_immutable(val):
                    return False
            return True

        # Unknown kind (e.g., "empty", "unknown-array") - be conservative
        log(f"Immutability unknown kind{col_label}: {kind}, assuming mutable")
        return False


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

    # Register Keras handlers lazily on first deepcopy call
    # (avoids import-time side effects that can break kernel startup)
    _register_keras_handlers_if_needed()

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    # Handle cudf objects by converting to pandas (all cudf logic in cudf_compat)
    from . import cudf_compat
    if cudf_compat.is_cudf_object(x):
        return cudf_compat.deepcopy_cudf(x, memo)

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
            return series.astype("string")
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

    Object dtype columns are handled specially:
    - If all values are immutable (strings, ints, etc.), shallow copy is used
    - If mutable objects exist, element-wise deep copy is performed

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
        # (only when _convert_object_dtypes=True)
        if _convert_object_dtypes:
            # Use iloc to read columns (handles MultiIndex) but column name to write
            # (preserves dtype correctly)
            for i in range(len(df.columns)):
                col = df.columns[i]
                col_series = df.iloc[:, i]
                if col_series.dtype == object:
                    converted = _convert_object_column_dtype(col_series)
                    if converted.dtype != object:
                        log(f"Converted column {col} from object to {converted.dtype}")
                        # Use column name for assignment to preserve dtype
                        df[col] = converted

        # Shallow copy: CoW handles non-object columns efficiently
        df_copy = df.copy(deep=False)

        # Process remaining object columns
        # Use iloc to read columns (handles MultiIndex) but column name to write
        for i in range(len(df_copy.columns)):
            col = df_copy.columns[i]
            col_series = df_copy.iloc[:, i]
            if col_series.dtype == object:
                # In preserve mode, check if all values are immutable
                if not _convert_object_dtypes and _object_column_is_all_immutable(col_series, col_name=str(col)):
                    # All immutable - shallow copy is sufficient (already done by df.copy)
                    log(f"Shallow copying immutable object column {col}")
                    # Force a copy of the underlying array to ensure independence
                    df_copy[col] = col_series.copy(deep=False)
                else:
                    # Has mutable objects - need element-wise deepcopy
                    num_rows = len(df_copy)
                    if num_rows > 10000:
                        log(f"Deep copying large object column {col} with {num_rows:,} rows...")
                    else:
                        log(f"Deep copying object column {col}")

                    # Apply deep copy and explicitly preserve object dtype
                    result = col_series.apply(lambda x: deepcopy(x, memo))
                    df_copy[col] = result.astype(object)

        memo[obj_id] = df_copy
        return df_copy


d[pd.DataFrame] = _deepcopy_dataframe


def _deepcopy_series(series: pd.Series, memo: dict[int, Any]) -> pd.Series:
    """
    Deep copy a Series with special object dtype handling.

    Object dtype Series are handled specially:
    - If all values are immutable (strings, ints, etc.), shallow copy is used
    - If mutable objects exist, element-wise deep copy is performed

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
        # (only when _convert_object_dtypes=True)
        if _convert_object_dtypes and series.dtype == object:
            # Try to convert to specialized dtype first
            converted = _convert_object_column_dtype(series)

            if converted.dtype != object:
                # Successfully converted - update original Series
                log(f"Converted Series from object to {converted.dtype}")
                series = converted

        # Shallow copy: CoW handles non-object Series efficiently
        series_copy = series.copy(deep=False)

        # Process if still object dtype
        if series_copy.dtype == object:
            # In preserve mode, check if all values are immutable
            if not _convert_object_dtypes and _object_column_is_all_immutable(series_copy, col_name=series_copy.name):
                # All immutable - shallow copy is sufficient
                log(f"Shallow copying immutable object Series")
                # Force a copy of the underlying array to ensure independence
                series_copy = series_copy.copy(deep=False)
            else:
                # Has mutable objects - need element-wise deepcopy
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


# CatBoost Pool handler - Pool explicitly blocks deepcopy, use slice workaround
try:
    from catboost import Pool as CatBoostPool
    from _catboost import _PoolBase

    def _deepcopy_catboost_pool(pool: "CatBoostPool", memo: dict[int, Any]) -> "CatBoostPool":
        """
        Deep copy a CatBoost Pool using the slice workaround.

        CatBoost Pool explicitly raises CatBoostError on __deepcopy__ and has no
        __reduce__ support. However, pool.slice() creates an independent copy
        that doesn't share memory with the original.

        Verified to preserve: features, labels, weights, categorical features,
        feature names, and group_id.

        Args:
            pool: CatBoost Pool to copy (Pool or _PoolBase)
            memo: Shared memo dict for tracking copied objects

        Returns:
            Independent copy of the Pool
        """
        obj_id = id(pool)
        if obj_id in memo:
            return memo[obj_id]

        # slice() with all indices creates a full independent copy
        pool_copy = pool.slice(list(range(pool.num_row())))
        memo[obj_id] = pool_copy
        return pool_copy

    d[CatBoostPool] = _deepcopy_catboost_pool
    d[_PoolBase] = _deepcopy_catboost_pool  # Also handle the base class
except ImportError:
    pass  # CatBoost not installed


# Keras model handler - uses opaque handler pattern for efficient copying
# NOTE: We don't import Keras at module load time to avoid triggering
# matplotlib backend initialization before the kernel is ready.
def _deepcopy_keras_model(model, memo: dict[int, Any]):
    """
    Deep copy a Keras model using the opaque handler pattern.

    For built models:
    - Uses clone_model() to share architecture (immutable after build)
    - Only copies weights via get_weights()/set_weights()
    - Much faster than full deepcopy for large models

    For unbuilt models:
    - Raises TypeError (cannot checkpoint unbuilt models)

    Args:
        model: Keras model (Sequential, Functional, or Model subclass)
        memo: Shared memo dict for tracking copied objects

    Returns:
        Independent copy of the model with weights

    Raises:
        TypeError: If model is not built
    """
    obj_id = id(model)
    if obj_id in memo:
        return memo[obj_id]

    # Get the opaque handler for this model
    handler = OpaqueRegistry.get_handler(model)
    if handler is None:
        # Fallback to stdlib deepcopy if no handler (shouldn't happen)
        import copy as stdlib_copy
        model_copy = stdlib_copy.deepcopy(model)
        memo[obj_id] = model_copy
        return model_copy

    # Check if model can be checkpointed (must be built)
    can_cp, error = handler.is_checkpointable(model)
    if not can_cp:
        raise TypeError(f"Cannot checkpoint Keras model: {error}")

    # Extract mutable state (weights) and create copy with shared architecture
    state = handler.get_mutable_state(model)
    model_copy = handler.copy_with_state(model, state, memo)

    return model_copy


# Flag to track if we've registered Keras handlers (done lazily on first use)
_keras_handlers_registered = False


def _register_keras_handlers_if_needed():
    """Register Keras model handlers lazily to avoid import-time side effects."""
    global _keras_handlers_registered
    if _keras_handlers_registered:
        return
    _keras_handlers_registered = True

    # Try tensorflow.keras first (more common), then standalone keras
    try:
        from tensorflow.keras.models import Sequential as TFSequential
        from tensorflow.keras.models import Model as TFModel
        _deepcopy_dispatch[TFSequential] = _deepcopy_keras_model
        _deepcopy_dispatch[TFModel] = _deepcopy_keras_model
    except ImportError:
        pass

    try:
        from keras.models import Sequential as KerasSequential
        from keras.models import Model as KerasModel
        _deepcopy_dispatch[KerasSequential] = _deepcopy_keras_model
        _deepcopy_dispatch[KerasModel] = _deepcopy_keras_model
    except ImportError:
        pass  # Keras not installed


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
