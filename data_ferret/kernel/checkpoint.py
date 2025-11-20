import copy
import datetime
import decimal
import types
from typing import Any, Dict, Set

from data_ferret.kernel.diff import Diff
from data_ferret.kernel.equality import user_ns_diff
from data_ferret.kernel.extended_types import TypeModel, get_type_model

import pandas as pd
import numpy as np

# Enable copy-on-write mode for better performance with DataFrame copies
pd.options.mode.copy_on_write = True


# System variables to filter out from user namespace
SYSTEM_VARIABLES = {
    "get_ipython",
    "In",
    "Out",
    "exit",
    "quit",
    "_",
    "__",
    "___",
    "_i",
    "_ii",
    "_iii",
    "_dh",
}


def is_valid_variable_name(name: str) -> bool:
    """
    Check if a variable name should be included in processing.

    Filters out:
    - Names starting with underscore (private/internal)
    - IPython system variables

    Args:
        name: Variable name to check

    Returns:
        True if the variable should be included, False otherwise
    """
    return not name.startswith("_") and name not in SYSTEM_VARIABLES


def is_valid_variable(name: str, value: Any) -> bool:
    """
    Check if a variable (name and value) should be included in processing.

    Filters out:
    - Names starting with underscore (private/internal)
    - IPython system variables
    - Module objects

    Args:
        name: Variable name to check
        value: Variable value to check

    Returns:
        True if the variable should be included, False otherwise
    """
    return is_valid_variable_name(name) and not isinstance(value, types.ModuleType)


def filter_user_namespace(user_ns: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a user namespace to include only valid variables.

    This is a convenience function that applies is_valid_variable() to
    an entire namespace dictionary.

    Args:
        user_ns: User namespace dictionary

    Returns:
        Filtered dictionary with only valid variables
    """
    return {k: v for k, v in user_ns.items() if is_valid_variable(k, v)}


def is_immutable_type(obj: Any) -> bool:
    """
    Check if an object is of an immutable type that's safe to skip copying.

    Returns True for basic immutable types commonly found in data science code,
    including Python primitives, NumPy scalars, and pandas temporal types.

    Args:
        obj: Object to check

    Returns:
        True if the object is immutable and safe to skip deep copying
    """
    if obj is None or obj is pd.NA:
        return True

    # Basic immutable types
    if isinstance(obj, (int, float, str, bool, bytes, complex, frozenset, range)):
        return True

    # NumPy scalar types (all are immutable)
    if isinstance(obj, np.generic):
        return True

    # Date/time types
    if isinstance(obj, (datetime.date, datetime.time, datetime.timedelta)):
        return True

    # Pandas temporal types
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return True

    # Decimal
    if isinstance(obj, decimal.Decimal):
        return True

    return False


def is_column_all_immutable(series: pd.Series) -> bool:
    """
    Check if all non-null values in a Series are immutable types.

    This function assumes homogeneous columns (all values of the same type),
    which is common in data science workflows. It performs a quick type check
    on the first non-null value, then verifies all other values are the same type.

    Args:
        series: pandas Series to check

    Returns:
        True if all non-null values are immutable types
    """
    if len(series) == 0:
        return True

    # Get non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return True

    # Check if first value is immutable
    first_val = non_null.iloc[0]
    if not is_immutable_type(first_val):
        return False

    # Fast path: check if all values are same type
    first_type = type(first_val)
    if all(type(x) is first_type for x in non_null.iloc[1:]):
        return True

    # Fallback: check each value individually (for heterogeneous columns)
    return all(is_immutable_type(x) for x in non_null.iloc[1:])


class Checkpoint:
    def __init__(self, name: str, user_ns: Dict[str, Any], memo: Dict[int, Any]):
        self.name = name
        self.user_ns = user_ns
        self.reverse_memo = {id(v): k for k, v in memo.items()}

    def original(self, id: int) -> int:
        return self.reverse_memo.get(id, id)


def checkpoint_diff(
    a: Checkpoint, b: Checkpoint, keys_to_include: Set[str] | None = None
):
    """
    Compare two checkpoints and return structured diff results.

    Returns:
        DiffResult: Structured diff tree with only differences
    """
    # all_keys = set(a.user_ns.keys()) | set(b.user_ns.keys())
    # diffs = {}
    # for k in all_keys:
    #     if not (k in a.user_ns and k in b.user_ns):
    #         if k in a.user_ns:
    #             diffs[k] = f"'removed"
    #         else:
    #             diffs[k] = f"'added"

    # ignore_keys = set(diffs.keys())
    # return user_ns_diff(a.user_ns, b.user_ns, ignore_keys) | diffs

    differ = Diff(strict=False, report_close=False, atol=1e-5, rtol=1e-5)
    return differ.diff(a.user_ns, b.user_ns, keys_to_include)


class Checkpoints:

    def __init__(self, sanity_check: bool = False, skip_immutable_copy: bool = False):
        self.sanity_check = sanity_check
        self.skip_immutable_copy = skip_immutable_copy
        self.saved = {}

    def _deep_copy_user_ns(
        self, variables: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[int, Any], Dict[str, Exception]]:
        """
        Deep copy a dictionary of variables, with special handling for pandas objects.

        Ensures that mutable objects inside pandas DataFrames and Series are fully
        deep copied to prevent shared references. For pandas objects with object dtype,
        this method ensures that mutable objects stored in cells (like lists, dicts,
        custom objects) are properly deep copied rather than creating shallow references.

        Args:
            variables: Dictionary of variables to copy

        Returns:
            Tuple of (copied dictionary, memo dictionary for tracking copied objects,
                     dictionary of failed variables with their exceptions)
        """
        copied = {}
        memo = {}
        failed = {}

        for k, v in variables.items():
            try:
                if isinstance(v, pd.DataFrame):
                    # Check if DataFrame has any object dtype columns
                    has_object_columns = any(
                        v[col].dtype == object for col in v.columns
                    )
                    df_copy = v.copy(deep=True)

                    if has_object_columns:
                        # For DataFrames with object columns, use pandas deep copy
                        # which is faster than manual apply + deepcopy

                        # Additional deep copy for object columns to ensure mutable objects
                        # in cells are truly independent
                        for col in df_copy.columns:
                            if df_copy[col].dtype == object:
                                # Skip deepcopy if optimization enabled and column contains only immutable objects
                                if (
                                    self.skip_immutable_copy
                                    and is_column_all_immutable(df_copy[col])
                                ):
                                    continue  # No need to deepcopy immutables
                                df_copy[col] = df_copy[col].apply(
                                    lambda x: copy.deepcopy(x, memo=memo)
                                )

                    memo[id(v)] = df_copy
                    copied[k] = df_copy

                elif isinstance(v, pd.Series):
                    series_copy = v.copy(deep=True)
                    if v.dtype == object:
                        # For object dtype Series, use pandas deep copy + manual deepcopy for cells
                        # Skip deepcopy if optimization enabled and series contains only immutable objects
                        if not (
                            self.skip_immutable_copy
                            and is_column_all_immutable(series_copy)
                        ):
                            series_copy = series_copy.apply(
                                lambda x: copy.deepcopy(x, memo=memo)
                            )
                    memo[id(v)] = series_copy
                    copied[k] = series_copy
                else:
                    # For all other types, use standard deepcopy with memo tracking
                    copied[k] = copy.deepcopy(v, memo=memo)
            except Exception as e:
                # Track variables that failed to copy
                failed[k] = e

        return copied, memo, failed

    def checkpointable_value(self, v: Any) -> bool:
        # Skip modules
        if isinstance(v, types.ModuleType):
            return False

        # Skip matplotlib objects
        if type(v).__module__.startswith("matplotlib"):
            return False

        # Skip numpy arrays containing matplotlib objects
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                # Check if any element is a matplotlib object
                try:
                    # Use flat iterator to avoid issues with multi-dimensional arrays
                    for item in v.flat:
                        if hasattr(
                            item, "__class__"
                        ) and item.__class__.__module__.startswith("matplotlib"):
                            return False
                except (AttributeError, TypeError):
                    # If we can't iterate or access module, be conservative and skip
                    return False

        return True

    def checkpointable_vars(self, user_ns: Dict[str, Any]) -> Dict[str, Any]:
        return filter_user_namespace(user_ns)

    def checkpointable_values(self, user_ns: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in user_ns.items() if self.checkpointable_value(v)}

    def save(
        self, name, user_ns: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        saved = {}
        removed = {}
        checkpointable_vars = self.checkpointable_vars(user_ns)

        checkpointable_values = self.checkpointable_values(checkpointable_vars)
        for k in checkpointable_vars.keys() - checkpointable_values.keys():
            removed[k] = get_type_model(user_ns[k])

        # Use helper to deep copy all variables with pandas awareness
        cp, memo, failed = self._deep_copy_user_ns(checkpointable_values)

        # Track successfully copied variables
        for k in cp:
            saved[k] = get_type_model(checkpointable_values[k])

        # Track variables that failed to copy
        for k in failed:
            removed[k] = get_type_model(checkpointable_values[k])

        if self.sanity_check:
            original = {k: v for k, v in checkpointable_values.items() if k in saved}
            diff = user_ns_diff(original, cp)
            if diff:
                raise ValueError(f"Sanity check failed: {diff}")

        self.saved[name] = Checkpoint(name, cp, memo)

        return saved, removed

    def restore(self, name, user_ns: Dict[str, Any]):
        cp = self.saved[name]
        checkpointable_vars = self.checkpointable_vars(user_ns)

        for k in checkpointable_vars.keys():
            del user_ns[k]

        # Deep copy the checkpoint before restoring to keep the checkpoint pristine
        # This ensures that modifications to restored variables don't affect the checkpoint
        restored_vars, _, _ = self._deep_copy_user_ns(cp.user_ns)
        user_ns.update(restored_vars)

    def type_models(self, user_ns: Dict[str, Any]) -> None:
        return {
            k: get_type_model(v) for k, v in self.checkpointable_vars(user_ns).items()
        }

    def delete(self, name):
        if name in self.saved:
            del self.saved[name]

    def list(self):
        return list(self.saved.keys())

    def clear(self):
        self.saved.clear()

    def get(self, name) -> Checkpoint:
        return self.saved[name]
