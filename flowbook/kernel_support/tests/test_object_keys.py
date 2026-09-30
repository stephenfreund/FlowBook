"""Tracker records must not merge across objects that reuse an address within a cell.

The column and structural trackers used to key records by id(obj). When a tracked
object died during the cell, CPython could give its address to a new object; the
dead object's records were then attributed to the new object's path (a column
write to a temporary `ss` reported as a write to `X_train`). They now key records
by ObjectKeys tokens (object_keys.py).

CPython does not guarantee that a freed address is reused, so these tests give
ObjectKeys an id function that assigns every object the same address: each new
object then "reuses" the address of the previous one once that one has died.
mcp/tests/test_address_reuse.py covers the real allocator in a kernel.
"""
import gc

import pandas as pd

from flowbook.kernel_support.column_tracking import ColumnAccessTracker
from flowbook.kernel_support.object_keys import ObjectKeys
from flowbook.kernel_support.structural_tracking import StructuralAccessTracker

ADDR = 0x1000


def one_address(_obj):
    return ADDR


def test_same_live_object_same_token():
    keys = ObjectKeys()
    a = pd.DataFrame({"x": [1]})
    t = keys.key(a)
    assert keys.key(a) == t and keys.lookup(a) == t


def test_dead_objects_address_gets_a_new_token():
    keys = ObjectKeys(id_func=one_address)
    a = pd.DataFrame({"x": [1]})
    t1 = keys.key(a)
    del a
    gc.collect()
    b = pd.DataFrame({"y": [2]})
    assert keys.lookup(b) is None
    assert keys.key(b) != t1


def test_unweakrefable_object_is_pinned_until_reset():
    keys = ObjectKeys()
    obj = [1, 2]  # lists cannot be weakly referenced
    t = keys.key(obj)
    assert keys.key(obj) == t and len(keys._pinned) == 1
    keys.reset()
    assert keys._pinned == [] and keys.lookup(obj) is None


def test_column_write_to_dead_temporary_not_attributed_to_object_at_same_address():
    tracker = ColumnAccessTracker(namespace_ref={})
    tracker._keys = ObjectKeys(id_func=one_address)
    ss = pd.DataFrame()
    tracker.register_df(ss, "ss")
    tracker.record_write(tracker.key(ss), ["num_sold"])
    del ss
    gc.collect()
    x_train = pd.DataFrame({"a": [1], "b": [2]})
    tracker.register_df(x_train, "X_train")
    tracker.record_read(tracker.key(x_train), ["a"])
    writes = tracker.resolve_writes_to_paths()
    assert writes.get("ss") == {"num_sold"}
    assert "num_sold" not in writes.get("X_train", set())
    assert tracker.resolve_to_paths().get("X_train") == {"a"}


def test_unregistered_temporary_not_resolved_to_object_found_by_namespace_walk():
    ns = {}
    tracker = ColumnAccessTracker(namespace_ref=ns)
    tracker._keys = ObjectKeys(id_func=one_address)
    tmp = pd.DataFrame({"x": [1]})
    tracker.record_read(tracker.key(tmp), ["x"])  # never registered: a temporary
    del tmp
    gc.collect()
    ns["df"] = pd.DataFrame({"y": [1]})  # found only by the lazy namespace walk
    assert "df" not in tracker.resolve_to_paths()


def test_structural_read_of_dead_temporary_not_attributed_to_object_at_same_address():
    tracker = StructuralAccessTracker(namespace_ref={})
    tracker._keys = ObjectKeys(id_func=one_address)
    tmp = pd.DataFrame({"x": [1, 2]})
    tracker.record_structural_read(tracker.key(tmp), "len")
    del tmp
    gc.collect()
    df = pd.DataFrame({"y": [1]})
    tracker.register(df, "df")
    assert "len" not in tracker.resolve_to_paths().get("df", set())


def test_rebinding_keeps_reads_of_the_previous_value():
    """`df = df[df.a > 0]`: the old df dies, but its column read is still a read of df."""
    tracker = ColumnAccessTracker(namespace_ref={})
    tracker._keys = ObjectKeys(id_func=one_address)
    old = pd.DataFrame({"a": [1, -1]})
    tracker.register_df(old, "df")
    tracker.record_read(tracker.key(old), ["a"])
    new = old[old.a > 0]
    del old
    gc.collect()
    tracker.register_df(new, "df")
    assert tracker.resolve_to_paths().get("df") == {"a"}
