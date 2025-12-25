"""
Determine if a Python object can be deep copied without actually copying it.

This module provides a function to check if an object is deepcopyable,
avoiding the overhead of actual deepcopy for known types while falling
back to try/except for user-defined types.

MultiIndex Column Support
-------------------------
DataFrames with MultiIndex columns (hierarchical column labels) are fully
supported. When checking DataFrame columns, we use positional indexing
(`.iloc[:, i]`) rather than label-based indexing (`df[col]`) to avoid
issues where `df[tuple]` might return a DataFrame instead of a Series
when the tuple is interpreted as a partial key.

Example of supported MultiIndex DataFrame:
    >>> arrays = [['A', 'A', 'B'], ['one', 'two', 'one']]
    >>> tuples = list(zip(*arrays))
    >>> columns = pd.MultiIndex.from_tuples(tuples)
    >>> df = pd.DataFrame([[1, 2, 3]], columns=columns)
    >>> check_deepcopyable(df)  # Returns None (is deepcopyable)
"""

from typing import Any, Set
import types
import copy
import io
import datetime
import decimal


def check_deepcopyable(obj: Any, _seen: Set[int] | None = None) -> str | None:
    """
    Check if an object can be deepcopied without actually copying it.

    This function avoids calling deepcopy for known types, reasoning about
    their copyability statically. For user-defined types, it falls back to
    attempting an actual deepcopy.

    Args:
        obj: Any Python object to check
        _seen: Internal set for cycle detection (do not pass externally)

    Returns:
        None if deepcopy would succeed, or a string explaining why it would fail
    """
    # Handle cycle detection
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return None  # Already processing this object, assume copyable

    obj_type = type(obj)

    # === 1. Immutable atomics - always deepcopyable ===
    # These are singletons or immutable, deepcopy returns them unchanged
    if obj is None or obj is True or obj is False:
        return None

    if obj_type in (int, float, complex, str, bytes, range):
        return None

    # datetime/time types are immutable
    if obj_type in (
        datetime.date,
        datetime.time,
        datetime.datetime,
        datetime.timedelta,
    ):
        return None

    if obj_type is decimal.Decimal:
        return None

    # === 2. Never deepcopyable types ===

    # Modules cannot be deepcopied
    if obj_type is types.ModuleType:
        return "module objects cannot be deepcopied"

    # Generator/coroutine types
    if obj_type is types.GeneratorType:
        return "generator objects cannot be deepcopied"
    if obj_type is types.CoroutineType:
        return "coroutine objects cannot be deepcopied"
    if obj_type is types.AsyncGeneratorType:
        return "async generator objects cannot be deepcopied"

    # Code, frame, and traceback objects
    if obj_type is types.CodeType:
        return "code objects cannot be deepcopied"
    if obj_type is types.FrameType:
        return "frame objects cannot be deepcopied"
    if obj_type is types.TracebackType:
        return "traceback objects cannot be deepcopied"

    # Method wrapper and other internal types
    if obj_type in (
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.MethodWrapperType,
        types.WrapperDescriptorType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
    ):
        return "built-in function/method objects cannot be deepcopied"

    # File and I/O objects
    if isinstance(obj, io.IOBase):
        return "file/IO objects cannot be deepcopied"

    # Check for socket (without importing socket module unless needed)
    module_name = obj_type.__module__
    type_name = obj_type.__name__

    if module_name == "socket" and type_name == "socket":
        return "socket objects cannot be deepcopied"

    if module_name == "ssl" and type_name in ("SSLSocket", "SSLContext"):
        return "SSL objects cannot be deepcopied"

    # Threading primitives
    if module_name == "_thread":
        if type_name in ("lock", "RLock", "LockType"):
            return "threading primitives cannot be deepcopied"

    if module_name == "threading":
        if type_name in (
            "Lock",
            "RLock",
            "Condition",
            "Semaphore",
            "BoundedSemaphore",
            "Event",
            "Barrier",
            "Thread",
            "Timer",
        ):
            return "threading primitives cannot be deepcopied"

    # Multiprocessing primitives
    if module_name.startswith("multiprocessing"):
        if type_name in (
            "Lock",
            "RLock",
            "Condition",
            "Semaphore",
            "BoundedSemaphore",
            "Event",
            "Barrier",
            "Queue",
            "JoinableQueue",
            "Pool",
            "Process",
        ):
            return "multiprocessing primitives cannot be deepcopied"

    # Weakrefs - note: basic weakref.ref (ReferenceType) IS actually deepcopyable
    # WeakValueDictionary etc. are also copyable but contain weak references
    # We don't need to special-case them

    # === 3. Matplotlib types - never deepcopyable ===
    if module_name.startswith("matplotlib"):
        return "matplotlib objects cannot be deepcopied"

    # === 4. NumPy types ===
    try:
        import numpy as np

        # NumPy scalars are immutable
        if isinstance(obj, np.generic):
            return None

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            # Object dtype arrays need element-by-element check
            if obj.dtype == object:
                _seen.add(obj_id)
                try:
                    for item in obj.flat:
                        reason = check_deepcopyable(item, _seen)
                        if reason:
                            return f"numpy array contains non-copyable element: {reason}"
                    return None
                except (TypeError, ValueError):
                    # If we can't iterate, be conservative
                    return "numpy array with object dtype cannot be iterated"
            # Non-object dtype arrays are always copyable
            return None

        # NumPy matrix (deprecated but still exists)
        if obj_type.__name__ == "matrix" and module_name == "numpy":
            return None

    except ImportError:
        pass

    # === 5. Pandas types ===
    try:
        import pandas as pd

        # Pandas timestamps and timedeltas are immutable
        if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
            return None

        # pd.NA is immutable
        if obj is pd.NA:
            return None

        # Pandas Index types
        if isinstance(obj, pd.Index):
            if obj.dtype == object:
                _seen.add(obj_id)
                for item in obj:
                    reason = check_deepcopyable(item, _seen)
                    if reason:
                        return f"pandas Index contains non-copyable element: {reason}"
            return None

        # Pandas Series
        if isinstance(obj, pd.Series):
            if obj.dtype == object:
                _seen.add(obj_id)
                for item in obj:
                    reason = check_deepcopyable(item, _seen)
                    if reason:
                        return f"pandas Series contains non-copyable element: {reason}"
            return None

        # Pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            _seen.add(obj_id)
            # Use .iloc to avoid issues with MultiIndex columns
            for i in range(len(obj.columns)):
                col_series = obj.iloc[:, i]
                if col_series.dtype == object:
                    for item in col_series:
                        reason = check_deepcopyable(item, _seen)
                        if reason:
                            return f"pandas DataFrame contains non-copyable element: {reason}"
            return None

    except ImportError:
        pass

    # === 6. Standard container types ===

    # frozenset - check elements
    if obj_type is frozenset:
        _seen.add(obj_id)
        for item in obj:
            reason = check_deepcopyable(item, _seen)
            if reason:
                return f"frozenset contains non-copyable element: {reason}"
        return None

    # tuple - check elements
    if obj_type is tuple:
        _seen.add(obj_id)
        for item in obj:
            reason = check_deepcopyable(item, _seen)
            if reason:
                return f"tuple contains non-copyable element: {reason}"
        return None

    # list - check elements
    if obj_type is list:
        _seen.add(obj_id)
        for item in obj:
            reason = check_deepcopyable(item, _seen)
            if reason:
                return f"list contains non-copyable element: {reason}"
        return None

    # dict - check keys and values
    if obj_type is dict:
        _seen.add(obj_id)
        for k, v in obj.items():
            reason = check_deepcopyable(k, _seen)
            if reason:
                return f"dict contains non-copyable key: {reason}"
            reason = check_deepcopyable(v, _seen)
            if reason:
                return f"dict contains non-copyable value: {reason}"
        return None

    # set - check elements
    if obj_type is set:
        _seen.add(obj_id)
        for item in obj:
            reason = check_deepcopyable(item, _seen)
            if reason:
                return f"set contains non-copyable element: {reason}"
        return None

    # === 7. Collections module types ===
    try:
        from collections import deque, OrderedDict, defaultdict, Counter

        if obj_type is deque:
            _seen.add(obj_id)
            for item in obj:
                reason = check_deepcopyable(item, _seen)
                if reason:
                    return f"deque contains non-copyable element: {reason}"
            return None

        if obj_type in (OrderedDict, defaultdict, Counter):
            _seen.add(obj_id)
            for k, v in obj.items():
                reason = check_deepcopyable(k, _seen)
                if reason:
                    return f"{obj_type.__name__} contains non-copyable key: {reason}"
                reason = check_deepcopyable(v, _seen)
                if reason:
                    return f"{obj_type.__name__} contains non-copyable value: {reason}"
            return None

    except ImportError:
        pass

    # === 8. Functions - regular functions are copyable, but closures may not be ===
    if obj_type is types.FunctionType:
        # Functions with closures might have non-copyable references
        # but generally functions themselves are copyable
        return None

    if obj_type is types.LambdaType:
        return None

    if obj_type is types.MethodType:
        # Bound methods - check if the instance is copyable
        _seen.add(obj_id)
        reason = check_deepcopyable(obj.__self__, _seen)
        if reason:
            return f"bound method's instance is not copyable: {reason}"
        return None

    # === 9. Type objects (classes themselves) ===
    # Use isinstance to catch metaclasses like ABCMeta (used by sklearn, numbers, etc.)
    if isinstance(obj, type):
        # Class objects are copyable (deepcopy returns the same singleton)
        return None

    # === 10. User-defined types - try deepcopy ===
    # For unknown types, we need to actually try deepcopy

    # First, check for common indicators of copyability
    # Has __deepcopy__? Likely explicitly designed to be copied
    if hasattr(obj_type, "__deepcopy__"):
        try:
            copy.deepcopy(obj)
            return None
        except Exception as e:
            return f"deepcopy failed: {type(e).__name__}: {e}"

    # Has __reduce__ or __reduce_ex__? Likely picklable = copyable
    # Note: all objects inherit these, but custom implementations suggest
    # the author thought about serialization

    # For dataclasses and objects with __dict__, check attributes
    _seen.add(obj_id)

    if hasattr(obj, "__dict__"):
        for attr_name, v in obj.__dict__.items():
            reason = check_deepcopyable(v, _seen)
            if reason:
                return f"attribute '{attr_name}' is not copyable: {reason}"
        # Also check class-level __slots__ if present
        if hasattr(obj_type, "__slots__"):
            for slot in obj_type.__slots__:
                if hasattr(obj, slot):
                    reason = check_deepcopyable(getattr(obj, slot), _seen)
                    if reason:
                        return f"slot '{slot}' is not copyable: {reason}"
        return None

    if hasattr(obj_type, "__slots__"):
        for slot in obj_type.__slots__:
            if hasattr(obj, slot):
                reason = check_deepcopyable(getattr(obj, slot), _seen)
                if reason:
                    return f"slot '{slot}' is not copyable: {reason}"
        return None

    # Last resort: actually try to deepcopy
    try:
        copy.deepcopy(obj)
        return None
    except Exception as e:
        return f"deepcopy failed: {type(e).__name__}: {e}"
