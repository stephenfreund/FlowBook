import copy
import types
from typing import Any, Dict, Set

from data_ferret.kernel.diff import Diff
from data_ferret.kernel.equality import user_ns_diff
from data_ferret.kernel.extended_types import TypeModel, get_type_model

import pandas as pd
import numpy as np

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
    return (
        is_valid_variable_name(name)
        and not isinstance(value, types.ModuleType)
    )


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


class Checkpoint:
    def __init__(self, name: str, user_ns: Dict[str, Any], memo: Dict[int, Any]):
        self.name = name
        self.user_ns = user_ns
        self.reverse_memo = { id(v): k for k, v in memo.items() }

    def original(self, id: int) -> int:
        return self.reverse_memo.get(id, id)

def checkpoint_diff(a: Checkpoint, b: Checkpoint, keys_to_include: Set[str] | None = None):
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
    
    differ = Diff(strict=False, report_close=False, atol=1e-6, rtol=1e-5)
    return differ.diff(a.user_ns, b.user_ns, keys_to_include)


class Checkpoints:

    def __init__(self, sanity_check: bool = False):
        self.sanity_check = sanity_check
        self.saved = {}

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
                        if hasattr(item, '__class__') and item.__class__.__module__.startswith("matplotlib"):
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
        cp = {}
        memo = {}
        saved = {}
        removed = {}
        checkpointable_vars = self.checkpointable_vars(user_ns)

        checkpointable_values = self.checkpointable_values(checkpointable_vars)
        for k in checkpointable_vars.keys() - checkpointable_values.keys():
            removed[k] = get_type_model(user_ns[k])

        for k, v in checkpointable_values.items():
            try:
                cp[k] = copy.deepcopy(v, memo=memo)
                saved[k] = get_type_model(v)
            except Exception as e:
                removed[k] = get_type_model(v)

        if self.sanity_check:
            original = {k: v for k, v in checkpointable_values.items() if k in saved}
            diff = user_ns_diff(original, cp)
            if diff:
                raise ValueError(f"Sanity check failed: {diff}")

        self.saved[name] = Checkpoint(name, cp, memo)

        return saved, removed

    def type_models(self, user_ns: Dict[str, Any]) -> Dict[str, TypeModel]:
        return {k: get_type_model(v) for k, v in self.checkpointable_vars(user_ns).items()}

    def restore(self, name, user_ns: Dict[str, Any]):
        cp = self.saved[name] 
        checkpointable_vars = self.checkpointable_vars(user_ns)

        for k in checkpointable_vars.keys():
            del user_ns[k]

        user_ns.update(cp.user_ns)

    def delete(self, name):
        del self.saved[name]

    def list(self):
        return list(self.saved.keys())

    def clear(self):
        self.saved.clear()

    def get(self, name) -> Checkpoint:
        return self.saved[name]
