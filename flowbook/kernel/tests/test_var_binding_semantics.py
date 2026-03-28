"""
Tests for Var(x) = binding-only semantics.

Var(x) means "read the namespace binding" — the pointer from name x to an object.
Sub-variable writes (Col, ColAdd, ColDel, Rows, Attr) do NOT conflict with
Var reads because they don't change the binding.

This file tests:
1. ▷ matrix: sub-variable writes vs Var reads = false
2. NoReadAndWrite: column assignment doesn't fire (Col ▷ Var = false)
3. ForwardStale: column write doesn't stale binding-only readers
4. BackwardMutation: column write doesn't violate binding-only readers
5. Method interception: df.sum() produces Col reads, not Var reads
6. End-to-end: method interception + staleness propagation
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper
from flowbook.kernel.models import ErrorType, ReasonType
from flowbook.kernel.locations import (
    ReadLoc, WriteLoc, ReadLocSet, WriteLocSet,
    write_conflicts_read, has_conflict, wlocs_conflict_rlocs,
    tracking_to_readlocset,
)
from flowbook.kernel_support.models import TrackingData


def _find_error(result, error_type):
    for e in result.errors:
        if e.error_type == error_type:
            return e
    return None


def _has_error(result, error_type):
    return _find_error(result, error_type) is not None


# =============================================================================
# 1. ▷ matrix: all sub-variable writes vs Var = false
# =============================================================================


class TestSubVariableWritesDontConflictWithVar:
    """Every sub-variable write type should NOT conflict with Var reads."""

    def test_col_vs_var(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.var("df"))

    def test_col_add_vs_var(self):
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.var("df"))

    def test_col_del_vs_var(self):
        assert not write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.var("df"))

    def test_rows_vs_var(self):
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.var("df"))

    def test_attr_vs_var(self):
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.var("df"))

    def test_file_vs_var(self):
        assert not write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.var("df"))


class TestVarWriteDoesConflictWithVar:
    """Var(x) write DOES conflict with Var(x) read — replacing the binding."""

    def test_var_vs_var_same(self):
        assert write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("x"))

    def test_var_vs_var_different(self):
        assert not write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("y"))


class TestVarWriteDoesNotConflictWithSubVariableReads:
    """Var(x) write does NOT directly conflict with Col/Attr reads.

    Rebinding detection works because Var(x) is always present in read
    sets alongside Col/Attr reads, so Var(x) ▷ Var(x) catches it.
    """

    def test_var_vs_col(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("df", "price"))

    def test_var_vs_attr(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.attr("df", "shape"))


class TestSubVariableWritesDoConflictWithColReads:
    """Col/ColAdd/Rows writes DO conflict with Col reads (at column level)."""

    def test_col_vs_col_same(self):
        assert write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.col("df", "price"))

    def test_col_vs_col_different(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.col("df", "qty"))

    def test_rows_vs_col(self):
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.col("df", "price"))

    def test_col_add_vs_col(self):
        """ColAdd doesn't conflict with existing column reads."""
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.col("df", "price"))

    def test_col_del_vs_col_same(self):
        assert write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.col("df", "old"))

    def test_col_del_vs_col_different(self):
        assert not write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.col("df", "price"))


# =============================================================================
# 2. NoReadAndWrite: column assignment doesn't fire
# =============================================================================


class TestNoReadAndWriteColumnAssignment:
    """df["y"] = expr should NOT trigger NoReadAndWrite."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_column_assignment_no_error(self):
        """df["y"] = [1,2,3]: reads Var(df), writes Col(df,y) → no NoReadAndWrite."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x"}})

        df2 = df.copy(); df2["y"] = [10, 20, 30]
        result = self.helper.execute_cell("b", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        assert not _has_error(result, ErrorType.NO_READ_AND_WRITE), \
            "Column assignment should NOT trigger NoReadAndWrite"

    def test_read_and_write_same_column_does_fire(self):
        """df["x"] = df["x"] + 1: reads Col(df,x), writes Col(df,x) → NoReadAndWrite."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x"}})

        df2 = df.copy(); df2["x"] = df["x"] + 1
        result = self.helper.execute_cell("b", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"},
            column_reads={"df": {"x"}}, column_writes={"df": {"x"}},
            continue_on_violation=True)

        assert _has_error(result, ErrorType.NO_READ_AND_WRITE), \
            "Reading and writing same column SHOULD trigger NoReadAndWrite"

    def test_var_reassignment_does_fire(self):
        """x = x + 1: reads Var(x), writes Var(x) → NoReadAndWrite."""
        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        result = self.helper.execute_cell("b", {"x": 1}, {"x": 2},
            reads={"x"}, writes={"x"})
        assert _has_error(result, ErrorType.NO_READ_AND_WRITE)


# =============================================================================
# 3. ForwardStale: column write doesn't stale binding-only readers
# =============================================================================


class TestForwardStalenessBindingSemantics:
    """Col writes should NOT stale cells with only Var reads."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_col_write_doesnt_stale_var_reader(self):
        """B reads Var(df) only. A writes Col(df, price). B NOT stale."""
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}})

        # B: z = df (binding-only read, no column detail)
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy(), "z": df.copy()},
            reads={"df"}, writes={"z"})

        # Edit A, modify price column
        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = df.copy(); df2["price"] = [10, 20]
        result = self.helper.execute_cell("a", {}, {"df": df2},
            writes={"df"}, column_writes={"df": {"price", "qty"}})

        # B reads Var(df), A writes Col changes → Var not stale
        assert "b" not in result.stale_cells, \
            "Binding-only reader should NOT be staled by column write"

    def test_col_write_does_not_stale_var_reader(self):
        """B reads Var(df). A replaces df entirely but with column_writes detail.

        When A has column_writes, the forward staleness uses Col-level precision.
        Col(df, other) does NOT conflict with Var(df) because Var(df) is a
        binding-only read and column writes don't change the binding.

        If B needs to be staled by column changes, B should have column_reads.
        """
        df = pd.DataFrame({"price": [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price"}})

        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy(), "z": True},
            reads={"df"}, writes={"z"})

        # Edit A, completely replace df
        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = pd.DataFrame({"other": [99]})
        result = self.helper.execute_cell("a", {}, {"df": df2},
            writes={"df"}, column_writes={"df": {"other"}})

        # Col(df, other) does NOT conflict with Var(df) — binding unchanged semantics
        assert "b" not in result.stale_cells, \
            "Col write should NOT stale binding-only (Var) reader"

    def test_var_write_without_column_detail_does_stale_var_reader(self):
        """B reads Var(df). A replaces df entirely without column_writes.

        When A has NO column_writes, the write is Var(df) which conflicts
        with Var(df) read. B IS stale.
        """
        df = pd.DataFrame({"price": [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"})

        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy(), "z": True},
            reads={"df"}, writes={"z"})

        # Edit A, completely replace df (no column_writes → Var-level write)
        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = pd.DataFrame({"other": [99]})
        result = self.helper.execute_cell("a", {}, {"df": df2},
            writes={"df"})

        # Var(df) write stales Var(df) reader
        assert "b" in result.stale_cells, \
            "Var write (no column detail) SHOULD stale Var reader"

    def test_col_write_does_stale_col_reader(self):
        """B reads Col(df, price). A modifies Col(df, price). B IS stale."""
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}})

        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}})

        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = df.copy(); df2["price"] = [10, 20]
        result = self.helper.execute_cell("a", {}, {"df": df2},
            writes={"df"}, column_writes={"df": {"price", "qty"}})

        assert "b" in result.stale_cells, \
            "Col reader SHOULD be staled by same-col write"

    def test_col_write_doesnt_stale_different_col_reader(self):
        """B reads Col(df, qty). A writes Col(df, price). B NOT stale."""
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}})

        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"qty"}})

        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = df.copy(); df2["price"] = [10, 20]
        result = self.helper.execute_cell("a", {"df": df.copy()}, {"df": df2},
            writes={"df"}, column_writes={"df": {"price"}})

        assert "b" not in result.stale_cells, \
            "Different-column reader should NOT be staled"


# =============================================================================
# 4. BackwardMutation: column write vs binding-only reader
# =============================================================================


class TestBackwardMutationBindingSemantics:
    """Col writes should NOT violate cells with only Var reads."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_col_write_no_violation_for_var_reader(self):
        """A reads Var(df). B writes Col(df, price). No NoWriteAfterRead violation."""
        df = pd.DataFrame({"price": [1, 2]})
        self.helper.execute_cell("a", {"df": df.copy()}, {"df": df.copy(), "z": True},
            reads={"df"}, writes={"z"})

        df2 = df.copy(); df2["price"] = [10, 20]
        result = self.helper.execute_cell("b", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True)

        assert not _has_error(result, ErrorType.NO_WRITE_AFTER_READ), \
            "Col write should NOT violate Var binding-only reader"

    def test_col_write_does_violate_col_reader(self):
        """A reads Col(df, price). B writes Col(df, price). Violation."""
        df = pd.DataFrame({"price": [1, 2]})
        self.helper.execute_cell("a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}})

        df2 = df.copy(); df2["price"] = [10, 20]
        result = self.helper.execute_cell("b", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True)

        assert _has_error(result, ErrorType.NO_WRITE_AFTER_READ), \
            "Col write SHOULD violate same-col reader"

    def test_var_write_does_violate_var_reader(self):
        """A reads Var(x). B writes Var(x). Violation."""
        self.helper.execute_cell("a", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"})

        result = self.helper.execute_cell("b", {"x": 1, "y": 2}, {"x": 99, "y": 2},
            writes={"x"})

        assert _has_error(result, ErrorType.NO_WRITE_AFTER_READ), \
            "Var write SHOULD violate Var reader"


# =============================================================================
# 5. tracking_to_readlocset: Var always emitted alongside Col/Attr
# =============================================================================


class TestTrackingReadLocConversion:
    """Verify that Var(x) is always emitted alongside Col/Attr reads."""

    def test_column_reads_produce_col_and_var(self):
        """TrackingData with column reads → both Col and Var locs."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={"df": {"price", "qty"}},
            column_writes={},
        )
        locs = tracking_to_readlocset(td)
        assert ReadLoc.col("df", "price") in locs
        assert ReadLoc.col("df", "qty") in locs
        assert ReadLoc.var("df") in locs  # always emitted

    def test_no_column_reads_produce_var(self):
        """TrackingData without column reads → Var only."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
        )
        locs = tracking_to_readlocset(td)
        assert ReadLoc.var("df") in locs

    def test_structural_reads_include_var(self):
        """Structural reads include Var alongside Attr."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            structural_reads={"df": {"shape"}},
        )
        locs = tracking_to_readlocset(td)
        assert ReadLoc.attr("df", "shape") in locs
        assert ReadLoc.var("df") in locs  # always emitted

    def test_mixed_vars_all_have_var(self):
        """All variables in reads_before_writes get Var, regardless of detail."""
        td = TrackingData(
            reads_before_writes={"df", "config"},
            writes=set(),
            column_reads_before_writes={"df": {"price"}},
            column_writes={},
        )
        locs = tracking_to_readlocset(td)
        assert ReadLoc.col("df", "price") in locs
        assert ReadLoc.var("df") in locs  # always emitted
        assert ReadLoc.var("config") in locs


# =============================================================================
# 6. Method interception integration
# =============================================================================


class TestMethodInterceptionProducesColReads:
    """Verify that intercepted DataFrame methods record column reads."""

    def test_sum_produces_col_reads(self):
        """After df.sum(), tracking has column reads for all columns."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker

        tracker = ColumnAccessTracker()
        tracker.activate()
        try:
            df = pd.DataFrame({"price": [1, 2, 3], "qty": [4, 5, 6]})
            tracker.register_df(df, "df")

            df.sum(numeric_only=True)

            reads = tracker._reads_by_id.get(id(df), set())
            assert "price" in reads
            assert "qty" in reads
        finally:
            tracker.deactivate()

    def test_describe_produces_col_reads(self):
        """After df.describe(), tracking has column reads."""
        from flowbook.kernel_support.column_tracking import ColumnAccessTracker

        tracker = ColumnAccessTracker()
        tracker.activate()
        try:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            tracker.register_df(df, "df")

            df.describe()

            reads = tracker._reads_by_id.get(id(df), set())
            assert "a" in reads
            assert "b" in reads
        finally:
            tracker.deactivate()


# =============================================================================
# 7. End-to-end: staleness with method-intercepted reads
# =============================================================================


class TestEndToEndMethodStaleness:
    """Full pipeline: method interception → Col reads → correct staleness."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_col_write_stales_cell_with_col_reads(self):
        """
        A: df = DataFrame({"x": [1], "y": [2]})
        B: result = df["x"].sum()  (reads Col(df, x))
        C: df["x"] = [99]          (writes Col(df, x))

        After running C, B should be stale (same column).
        """
        self.helper.set_cell_order(["a", "b", "c"])
        df = pd.DataFrame({"x": [1], "y": [2]})

        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x", "y"}})
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy(), "result": 1},
            reads={"df"}, writes={"result"}, column_reads={"df": {"x"}})

        df2 = df.copy(); df2["x"] = [99]
        result = self.helper.execute_cell("c", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"x"}},
            continue_on_violation=True)

        state = self.helper.sdc._notebook_state
        # B reads Col(df, x), C wrote Col(df, x) → ForwardStale
        # But C is after B, so this is ForwardStale from C's writes
        # Actually, C is below B but the check for C's write staling B
        # would be in B's forward contamination check (NoReadBeforeWrite)
        # or C's forward staleness propagation.

    def test_binding_only_reader_not_staled(self):
        """
        A: df = DataFrame(...)
        B: z = df  (binding-only read)
        A re-executed with column change → B NOT stale.
        """
        self.helper.set_cell_order(["a", "b"])
        df = pd.DataFrame({"x": [1, 2]})

        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x"}})
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy(), "z": True},
            reads={"df"}, writes={"z"})

        state = self.helper.sdc._notebook_state
        b_reads = state.reads.get("b", frozenset())
        # B should have Var(df) only (no column detail)
        assert any(r.type.value == "var" and r.name == "df" for r in b_reads)
        assert not any(r.type.value == "col" for r in b_reads)

        # Edit and rerun A with column change
        state.handle_edit("a")
        df2 = df.copy(); df2["x"] = [10, 20]
        result = self.helper.execute_cell("a", {"df": df.copy()}, {"df": df2},
            writes={"df"}, column_writes={"df": {"x"}})

        # B has Var(df) read, A's change is at column level → B NOT stale
        assert "b" not in result.stale_cells
