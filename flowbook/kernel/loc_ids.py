"""
Stable Object Identity for Location Qualifiers.

This module provides the infrastructure for matching the paper's formal model
where Loc ::= x | d.c and d ∈ Address is a stable DataFrame address.

The challenge: Python's id() breaks on checkpoint deep copy (every deepcopy()
creates new objects with new ids). And df.attrs propagates through user copies,
conflating independent DataFrames.

Solution: A weakref-based side-table (StableIdMap) that:
- Maps Python id() to stable integer identifiers
- Uses weakref to detect id reuse after GC
- Transfers stable_ids to deep-copy targets via checkpoint memo dicts

Formal ref: FORMAL_DEVELOPMENT.md §9.1
"""

import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class LocRef:
    """
    Qualifier for DataFrame sub-locations, combining stable identity with
    the variable name used to access the object.

    - loc_id: Stable identity from StableIdMap (for intra-DataFrame conflicts)
    - var_name: Variable name at access time (for Var ▷ Col bridging)

    Two LocRefs with the same loc_id refer to the same DataFrame object,
    even if accessed through different variable names (aliases).

    Formal ref: FORMAL_DEVELOPMENT.md §9.1
    """

    loc_id: int
    var_name: str

    def __repr__(self) -> str:
        return f"LocRef({self.loc_id}, {self.var_name!r})"


class StableIdMap:
    """
    Maps Python object identity to stable location identifiers.

    Survives checkpoint deep copy via memo-based transfer.
    Detects id() reuse via weakref validation.

    Correctness by scenario:
        Same object          → ref() is obj → return existing stable_id
        Alias (df2 = df)     → same object  → return same stable_id
        User copy (df.copy()) → different obj → assign new stable_id
        id reuse after GC    → ref dead      → assign new stable_id
        Our deepcopy (checkpoint) → apply_memo() transfers stable_id

    Formal ref: FORMAL_DEVELOPMENT.md §9.1
    """

    def __init__(self) -> None:
        # python_id → (stable_id, weakref_to_object)
        self._entries: Dict[int, Tuple[int, weakref.ref]] = {}
        self._next_id: int = 0

    def get_stable(self, obj: Any) -> int:
        """Get or assign a stable id for obj.

        If obj has been seen before (same object, verified by weakref),
        returns the existing stable_id. Otherwise assigns a new one.
        """
        pid = id(obj)
        entry = self._entries.get(pid)
        if entry is not None:
            stable_id, ref = entry
            if ref() is obj:  # Same object, not id reuse
                return stable_id
        # New object or id reuse → assign fresh stable_id
        stable_id = self._next_id
        self._next_id += 1
        self._entries[pid] = (stable_id, weakref.ref(obj))
        return stable_id

    def apply_memo(self, memo: Dict[int, Any]) -> None:
        """Transfer stable_ids from original objects to their deep copies.

        Called after _deep_copy_user_ns() with its memo dict.
        memo maps id(original) → copy_object.

        This ensures that checkpoint copies retain the same stable_id as
        the original objects they were copied from.
        """
        for old_id, new_obj in memo.items():
            entry = self._entries.get(old_id)
            if entry is not None:
                stable_id, _ = entry
                # Only transfer to weakref-able objects
                try:
                    new_pid = id(new_obj)
                    self._entries[new_pid] = (stable_id, weakref.ref(new_obj))
                except TypeError:
                    # Object doesn't support weakref (e.g., int, str)
                    pass
                # Don't delete old entry — original may still be alive

    def lookup(self, obj: Any) -> Optional[int]:
        """Look up the stable id for obj without assigning one.

        Returns None if obj has no stable id assigned.
        """
        pid = id(obj)
        entry = self._entries.get(pid)
        if entry is not None:
            stable_id, ref = entry
            if ref() is obj:
                return stable_id
        return None

    def clear(self) -> None:
        """Clear all mappings (e.g., on kernel restart)."""
        self._entries.clear()
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)


def get_qualifier(
    var_name: str,
    namespace: Optional[dict] = None,
    stable_map: Optional[StableIdMap] = None,
) -> "str | LocRef":
    """Get the qualifier for a variable: LocRef if it's a DataFrame/Series, else str.

    This is the bridge between variable names and stable object identity.
    When namespace and stable_map are available, DataFrames and Series get
    LocRef qualifiers (with stable loc_ids). Other objects fall back to
    variable name strings.

    Args:
        var_name: The variable name
        namespace: Current kernel namespace (optional)
        stable_map: StableIdMap instance (optional)

    Returns:
        LocRef for DataFrames/Series, str for everything else
    """
    if namespace is not None and stable_map is not None:
        obj = namespace.get(var_name)
        if obj is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
            loc_id = stable_map.get_stable(obj)
            return LocRef(loc_id, var_name)
    return var_name


def build_loc_context(
    namespace: dict,
    stable_map: StableIdMap,
) -> Dict[int, Set[str]]:
    """Build a mapping from loc_id to variable names pointing to that object.

    This is used for display purposes — given a loc_id, we can show the user
    which variable names currently refer to that DataFrame.

    Args:
        namespace: Current kernel namespace
        stable_map: StableIdMap instance

    Returns:
        Dict mapping loc_id → set of variable names
    """
    result: Dict[int, Set[str]] = {}
    for var_name, obj in namespace.items():
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            loc_id = stable_map.get_stable(obj)
            result.setdefault(loc_id, set()).add(var_name)
    return result
