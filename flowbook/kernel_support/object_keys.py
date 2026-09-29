"""
Per-object keys for the column and structural trackers.

The trackers record accesses during a cell and resolve them to variable paths
only when the cell ends. They used to key everything by id(obj). A tracked
object that dies during the cell (a temporary, or a variable's previous value
after rebinding) frees its address, and CPython routinely gives that address
to a new object. The two objects' records then shared one key and were merged
and attributed to whichever path the new object was registered under, e.g. a
column write to a temporary `ss` reported as a write to `X_train`.

ObjectKeys gives each object it is asked about a token that is never reused:
while the object is alive, key(obj) returns the same token; once it dies (a
weakref callback, which runs at deallocation, before the address can be
reused) its address maps to a fresh token for the next object. Records keyed
by a dead object's token stay separate and keep the path that object was
registered under. Objects that cannot be weakly referenced are kept alive
until reset() so their address cannot be reused within the cell.
"""
import itertools
import weakref
from typing import Any, Callable, Dict, List, Optional

_tokens = itertools.count(1)  # global, so tokens never repeat across cells or trackers


class ObjectKeys:
    def __init__(self, id_func: Callable[[Any], int] = id):
        # id_func is id(); tests substitute a function that gives several objects
        # the same address, to simulate address reuse deterministically.
        self._id = id_func
        self._token_by_id: Dict[int, int] = {}
        self._refs: Dict[int, Any] = {}
        self._pinned: List[Any] = []

    def key(self, obj) -> int:
        """Token for obj, assigned on first use in the current cell."""
        i = self._id(obj)
        tok = self._token_by_id.get(i)
        if tok is not None:
            return tok
        tok = self._token_by_id[i] = next(_tokens)
        try:
            self._refs[i] = weakref.ref(obj, self._make_drop(i))
        except TypeError:
            self._pinned.append(obj)
        return tok

    def lookup(self, obj) -> Optional[int]:
        """Token already assigned to obj (a live object), or None. Does not assign one."""
        return self._token_by_id.get(self._id(obj))

    def _make_drop(self, i):
        def drop(ref):
            if self._refs.get(i) is ref:
                del self._refs[i]
                self._token_by_id.pop(i, None)
        return drop

    def reset(self) -> None:
        self._token_by_id.clear()
        self._refs.clear()
        self._pinned.clear()
