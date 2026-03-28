"""
Tests for setitem NOT producing spurious column reads.

The tracked_df_setitem wrapper snapshots dtypes before/after writes for
provenance tracking.  Previously it used df[key].dtype which triggered
tracked_df_getitem, recording a spurious read.  After the fix, dtype
snapshots use df.dtypes[key] which bypasses the __getitem__ patch.

These tests verify that write-only operations produce NO column reads.
"""

import pandas as pd
import pytest

from flowbook.kernel_support.column_tracking import ColumnAccessTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_tracker(df, path="df", cell_id="test"):
    """Create, install, and activate a tracker with a registered DataFrame."""
    tracker = ColumnAccessTracker()
    tracker.register_df(df, path)
    tracker.install()
    tracker.activate(cell_id=cell_id)
    return tracker


def _teardown(tracker):
    tracker.deactivate()
    tracker.uninstall()


# ---------------------------------------------------------------------------
# setitem: write-only operations must NOT record reads
# ---------------------------------------------------------------------------

class TestSetitemNoSpuriousReads:
    """df['col'] = value must record a write but NOT a read."""

    def test_setitem_existing_column_scalar(self):
        """df['age'] = 4 — overwrite existing column with scalar."""
        df = pd.DataFrame({"age": [30, 25]})
        tracker = _setup_tracker(df)
        try:
            df["age"] = 4

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert reads.get("df", set()) == set(), (
                "Write-only setitem should not record any column reads"
            )
            assert "age" in writes.get("df", set()), (
                "setitem should record a column write"
            )
        finally:
            _teardown(tracker)

    def test_setitem_existing_column_list(self):
        """df['age'] = [1, 2] — overwrite existing column with list."""
        df = pd.DataFrame({"age": [30, 25]})
        tracker = _setup_tracker(df)
        try:
            df["age"] = [1, 2]

            reads = tracker.resolve_to_paths()
            assert reads.get("df", set()) == set()
        finally:
            _teardown(tracker)

    def test_setitem_new_column(self):
        """df['new_col'] = 1 — add a new column."""
        df = pd.DataFrame({"a": [1, 2]})
        tracker = _setup_tracker(df)
        try:
            df["new_col"] = 1

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert reads.get("df", set()) == set(), (
                "Adding a new column should not record any reads"
            )
            assert "new_col" in writes.get("df", set())
        finally:
            _teardown(tracker)

    def test_setitem_dtype_change_no_spurious_read(self):
        """df['age'] = 'old' — changes dtype from int to object, no read."""
        df = pd.DataFrame({"age": [30, 25]})
        tracker = _setup_tracker(df)
        try:
            df["age"] = "old"  # int64 -> object

            reads = tracker.resolve_to_paths()
            assert reads.get("df", set()) == set(), (
                "Dtype-changing write should not record a read"
            )
        finally:
            _teardown(tracker)

    def test_setitem_multi_column_write(self):
        """df[['a', 'b']] = values — multi-column write, no reads."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        tracker = _setup_tracker(df)
        try:
            df[["a", "b"]] = pd.DataFrame({"a": [10, 20], "b": [30, 40]})

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert reads.get("df", set()) == set(), (
                "Multi-column write should not record any reads"
            )
            assert writes.get("df", set()) == {"a", "b"}
        finally:
            _teardown(tracker)


# ---------------------------------------------------------------------------
# setitem: read-then-write SHOULD record a read
# ---------------------------------------------------------------------------

class TestSetitemWithReads:
    """Operations that genuinely read a column before writing should track it."""

    def test_read_modify_write(self):
        """df['age'] = df['age'] + 1 — reads age then writes age."""
        df = pd.DataFrame({"age": [30, 25]})
        tracker = _setup_tracker(df)
        try:
            df["age"] = df["age"] + 1  # RHS reads, LHS writes

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            # 'age' is read (by df['age'] on RHS) BEFORE the write
            assert "age" in reads.get("df", set()), (
                "Read-modify-write should record a column read"
            )
            assert "age" in writes.get("df", set())
        finally:
            _teardown(tracker)

    def test_read_one_write_another(self):
        """df['b'] = df['a'] * 2 — reads 'a', writes 'b'."""
        df = pd.DataFrame({"a": [1, 2], "b": [0, 0]})
        tracker = _setup_tracker(df)
        try:
            df["b"] = df["a"] * 2

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert "a" in reads.get("df", set()), "Should record read on 'a'"
            assert "b" not in reads.get("df", set()), (
                "'b' is only written, not read"
            )
            assert "b" in writes.get("df", set())
        finally:
            _teardown(tracker)

    def test_getitem_then_setitem(self):
        """x = df['a']; df['a'] = x + 1 — read then write in sequence."""
        df = pd.DataFrame({"a": [1, 2]})
        tracker = _setup_tracker(df)
        try:
            x = df["a"]       # explicit read
            df["a"] = x + 1   # write

            reads = tracker.resolve_to_paths()
            assert "a" in reads.get("df", set()), (
                "Explicit getitem should record a read even though setitem follows"
            )
        finally:
            _teardown(tracker)


# ---------------------------------------------------------------------------
# loc/iloc setitem: write-only operations must NOT record reads
# ---------------------------------------------------------------------------

class TestLocSetitemNoSpuriousReads:
    """loc/iloc writes should not produce spurious reads."""

    def test_loc_setitem_column(self):
        """df.loc[:, 'a'] = 10 — write via loc, no read."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracker = _setup_tracker(df)
        try:
            df.loc[:, "a"] = 10

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert "a" not in reads.get("df", set()), (
                "loc write should not record a read on the written column"
            )
            assert "a" in writes.get("df", set())
        finally:
            _teardown(tracker)

    def test_iloc_setitem(self):
        """df.iloc[:, 0] = 10 — write via iloc, no read."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracker = _setup_tracker(df)
        try:
            df.iloc[:, 0] = 10

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert "a" not in reads.get("df", set()), (
                "iloc write should not record a read on the written column"
            )
            assert "a" in writes.get("df", set())
        finally:
            _teardown(tracker)


# ---------------------------------------------------------------------------
# insert: write-only, must NOT record reads
# ---------------------------------------------------------------------------

class TestInsertNoSpuriousReads:
    """DataFrame.insert() is a pure write — no reads."""

    def test_insert_no_read(self):
        """df.insert(0, 'new', [1, 2]) — no reads."""
        df = pd.DataFrame({"a": [1, 2]})
        tracker = _setup_tracker(df)
        try:
            df.insert(0, "new", [1, 2])

            reads = tracker.resolve_to_paths()
            writes = tracker.resolve_writes_to_paths()

            assert reads.get("df", set()) == set()
            assert "new" in writes.get("df", set())
        finally:
            _teardown(tracker)
