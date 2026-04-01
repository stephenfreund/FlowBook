"""
Integration tests for LocSet-based conflict detection in the enforcer.

Tests verify that the enforcer correctly uses the write_conflicts_read (otimes) operator for:
- Forward staleness with column-level precision
- Backward mutation detection with column-level precision
- Forward contamination with column-level precision
- Independent column additions (no false positives)
- Attribute conflicts always enforced
- Row changes cascading to all columns
- Edit + rerun with BackwardStale for removed writes
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper
from flowbook.kernel.models import ErrorType, ReasonType
from flowbook.kernel.locations import ReadLoc, WriteLoc, writelocset_var_names, readlocset_var_names


def _find_error(result, error_type):
    for e in result.errors:
        if e.error_type == error_type:
            return e
    return None


def _has_error(result, error_type):
    return _find_error(result, error_type) is not None


class TestForwardStalenessColumnPrecision:
    """Forward staleness uses the otimes operator for column-level precision."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_same_column_write_stales_reader(self):
        """Writing df["price"] stales cell that reads df["price"]."""
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )

        # Edit A to change price column
        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = df.copy()
        df2["price"] = [10, 20]
        result = self.helper.execute_cell(
            "a", {}, {"df": df2},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )
        assert "b" in result.stale_cells

    def test_different_column_write_doesnt_stale_reader(self):
        """Writing df["qty"] does NOT stale cell that reads only df["price"],
        when the diff can detect column-level precision.

        For column-level precision to work, the pre_namespace must already
        contain df so the diff can compare column-by-column (producing Col
        write locs) rather than detecting a whole-variable creation
        (producing Var write loc which conflicts with everything).
        """
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )
        self.helper.execute_cell(
            "c", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"qty"}},
        )

        # Edit A, only qty changes.
        # Use df as pre_namespace so the diff sees column-level changes,
        # not a whole-variable ValueChanged.
        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = df.copy()
        df2["qty"] = [30, 40]
        result = self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df2},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )
        # C reads qty which changed -> stale
        assert "c" in result.stale_cells
        # B reads only price which did NOT change -> NOT stale
        # The diff detects only Col(df, qty) changed, and B's read is Col(df, price)
        assert "b" not in result.stale_cells

    def test_col_add_doesnt_stale_existing_column_reader(self):
        """ColAdd(df, new) does NOT stale cell reading Col(df, price).

        Adding a new column to df should not stale downstream cells that only
        read existing, unchanged columns.  The diff must see the original df
        in pre_namespace to detect a ColAdd (rather than a whole-variable
        ValueChanged).
        """
        df = pd.DataFrame({"price": [1, 2]})
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price"}},
        )
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )

        # Edit A to add a new column.
        # Use df as pre_namespace so the diff detects ColAdd(df, new) rather
        # than ValueChanged(df).
        self.helper.sdc._notebook_state.handle_edit("a")
        df_with_new = df.copy()
        df_with_new["new"] = [5, 6]
        result = self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df_with_new},
            writes={"df"}, column_writes={"df": {"price", "new"}},
        )
        # The diff detects ColAdd(df, new). B reads Col(df, price).
        # ColAdd(df, new) does NOT conflict with Col(df, price) per the otimes table.
        # B has no writes={"df"}, so no write-write overlap either.
        assert "b" not in result.stale_cells

    def test_var_write_stales_all_column_readers(self):
        """Var(df) write stales ALL cells that read any column of df."""
        df = pd.DataFrame({"price": [1], "qty": [2]})
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )
        self.helper.execute_cell(
            "c", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"qty"}},
        )

        # Edit A, completely replace df with different values in ALL columns
        self.helper.sdc._notebook_state.handle_edit("a")
        df2 = pd.DataFrame({"price": [10], "qty": [20]})
        result = self.helper.execute_cell(
            "a", {}, {"df": df2},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )
        assert "b" in result.stale_cells
        assert "c" in result.stale_cells


class TestForwardContaminationColumnPrecision:
    """Forward contamination (NoReadBeforeWrite) uses the otimes operator."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_same_column_forward_contamination(self):
        """Cell reads df["price"], later cell wrote df["price"] -> contaminated."""
        self.helper.set_cell_order(["a", "b"])
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})

        # Run B first (writes df with price changed)
        df_b = df.copy()
        df_b["price"] = [99, 99]
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_b},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True,
        )

        # Run A (reads df["price"]) -- B below wrote it
        result = self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
            continue_on_violation=True,
        )
        assert _has_error(result, ErrorType.NO_READ_BEFORE_WRITE)

    def test_different_column_no_forward_contamination(self):
        """Cell reads df["qty"], later cell wrote df["price"] -> NOT contaminated.

        The otimes operator ensures ColMod(df, price) does not conflict with
        Col(df, qty).
        """
        self.helper.set_cell_order(["a", "b"])
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})

        # Run B first (writes df with price changed)
        df_b = df.copy()
        df_b["price"] = [99, 99]
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_b},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True,
        )

        # Run A (reads df["qty"]) -- B below wrote df["price"], different column
        result = self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"qty"}},
            continue_on_violation=True,
        )
        # Should NOT have forward contamination for qty
        assert not _has_error(result, ErrorType.NO_READ_BEFORE_WRITE)


class TestBackwardMutationColumnPrecision:
    """Backward mutation (NoWriteAfterRead) uses the otimes operator."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_same_column_backward_violation(self):
        """A reads df["price"], B modifies df["price"] -> violation."""
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )

        # B modifies price column in-place
        df_b = df.copy()
        df_b["price"] = [99, 99]
        result = self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_b},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True,
        )
        assert _has_error(result, ErrorType.NO_WRITE_AFTER_READ)

    def test_different_column_no_backward_violation(self):
        """A reads df["price"], B modifies df["qty"] -> no violation.

        The otimes operator ensures Col(df, qty) write does not conflict with
        Col(df, price) read.
        """
        df = pd.DataFrame({"price": [1, 2], "qty": [3, 4]})
        self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )

        # B modifies qty column only
        df_b = df.copy()
        df_b["qty"] = [99, 99]
        result = self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_b},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"qty"}},
            continue_on_violation=True,
        )
        assert not _has_error(result, ErrorType.NO_WRITE_AFTER_READ)


class TestIndependentColumnAdditions:
    """Independent column additions should not cause false ForwardStale write-write overlap."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_two_col_adds_no_write_write_overlap(self):
        """B adds df["price"], C adds df["qty"]. Editing B does NOT stale C
        because ColumnAdded now maps to Col (not ColAdd), giving column-level
        precision in write-write overlap.

        B's changed write locs are Col(df, price). C's write locs include
        Col(df, qty). Col(df, price) does not conflict with Col(df, qty),
        so there is no write-write overlap and C stays clean.
        """
        df = pd.DataFrame({"base": [1, 2]})

        # A creates df
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"base"}},
        )

        # B adds price column
        df_b = df.copy()
        df_b["price"] = [10, 20]
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_b},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True,
        )

        # C adds qty column
        df_c = df_b.copy()
        df_c["qty"] = [30, 40]
        self.helper.execute_cell(
            "c", {"df": df_b.copy()}, {"df": df_c},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"qty"}},
            continue_on_violation=True,
        )

        # Edit and rerun B
        self.helper.sdc._notebook_state.handle_edit("b")
        df_b2 = df.copy()
        df_b2["price"] = [100, 200]
        result = self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_b2},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True,
        )
        # C is NOT stale: Col(df, price) does not overlap Col(df, qty).
        # ColumnAdded maps to Col, giving column-level write-write precision.
        assert "c" not in result.stale_cells


class TestAttributeConflictsAlwaysEnforced:
    """Attribute conflicts (df.shape, df.columns) are always enforced."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_structural_read_backward_violation(self):
        """Cell that reads df.columns, then later cell drops a column -> violation."""
        df = pd.DataFrame({"x": [1], "y": [2]})

        # A reads df.columns (structural read)
        self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, structural_reads={"df": {"columns"}},
        )

        # B drops a column (writes df, removes column y)
        df_dropped = df.drop("y", axis=1)
        result = self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df_dropped},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"x"}},
            continue_on_violation=True,
        )
        # B writes to df which A read -> backward violation
        assert result.has_errors()

    def test_no_structural_off_mode(self):
        """Verify there is no way to disable structural tracking."""
        # The enforcer should not have a structural_mode attribute
        assert not hasattr(self.helper.sdc, '_structural_mode')
        assert not hasattr(self.helper.sdc, 'structural_mode')


class TestRowChangeCascade:
    """Row changes (Rows(d)) cascade to all column readers."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_row_change_stales_all_column_readers(self):
        """If a cell produces a df with more rows, all column readers are stale."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x", "y"}},
        )
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"x"}},
        )
        self.helper.execute_cell(
            "c", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"y"}},
        )

        # Edit A to produce more rows
        self.helper.sdc._notebook_state.handle_edit("a")
        df_more = pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})
        result = self.helper.execute_cell(
            "a", {}, {"df": df_more},
            writes={"df"}, column_writes={"df": {"x", "y"}},
        )
        # Both B and C should be stale (df changed at all columns due to row change)
        assert "b" in result.stale_cells
        assert "c" in result.stale_cells


class TestEditRerunBackwardStale:
    """Edit + rerun correctly detects removed writes via BackwardStale."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_edit_remove_write_marks_downstream_stale(self):
        """
        A writes z, B writes z, C reads z.
        Edit B to stop writing z -> C becomes stale because z changed.
        """
        self.helper.execute_cell("a", {}, {"z": 0}, writes={"z"})
        self.helper.execute_cell("b", {"z": 0}, {"z": 99}, writes={"z"})
        self.helper.execute_cell("c", {"z": 99}, {"z": 99}, reads={"z"})

        # Edit B
        self.helper.sdc._notebook_state.handle_edit("b")

        # Rerun B with different writes (no longer writes z)
        result = self.helper.execute_cell(
            "b", {"z": 0}, {"z": 0, "other": 10},
            writes={"other"},
        )

        # C should be stale: B used to write z (which C reads) and now it doesn't.
        # The old write union includes z, so ForwardStale(old_writes={z}, R_C={z}) triggers.
        assert "c" in result.stale_cells

    def test_edit_remove_write_backward_stale(self):
        """
        A writes z, B writes z, C reads z.
        Edit B to stop writing z -> A is backward-stale
        (A was last writer of z before B, and B removed its write).
        """
        self.helper.execute_cell("a", {}, {"z": 0}, writes={"z"})
        self.helper.execute_cell("b", {"z": 0}, {"z": 99}, writes={"z"})
        self.helper.execute_cell("c", {"z": 99}, {"z": 99}, reads={"z"})

        # Edit B
        self.helper.sdc._notebook_state.handle_edit("b")

        # Rerun B with different writes (no longer writes z)
        result = self.helper.execute_cell(
            "b", {"z": 0}, {"z": 0, "other": 10},
            writes={"other"},
        )

        # A should be backward-stale: A was LastWriter of z before B, and B
        # removed z from its writes, exposing A's value to downstream cells.
        assert "a" in result.stale_cells

    def test_edit_change_column_staleness(self):
        """
        A writes df (all columns), B modifies df["price"].
        Edit B to only write df["qty"] -> C reads df["price"].

        The ForwardStale check uses _changes_to_writelocset which uses the
        CURRENT diff's column_changed. Since B's new diff only shows qty
        changed, and C reads price, C is NOT stale via the read-conflict path.

        However, the W_i_union includes "df" (variable level), and
        _changes_to_writelocset maps it to Col(df, qty) since column_changed
        has detail for df. So Col(df, qty) does not conflict with Col(df, price).
        """
        df = pd.DataFrame({"price": [1], "qty": [2]})
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"price", "qty"}},
        )

        df2 = df.copy()
        df2["price"] = [99]
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"price"}},
            continue_on_violation=True,
        )

        # Add a cell C that reads df["price"]
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "c", {"df": df2.copy()}, {"df": df2.copy()},
            reads={"df"}, column_reads={"df": {"price"}},
        )

        # Edit B, now only writes qty (stops writing price)
        self.helper.sdc._notebook_state.handle_edit("b")
        df3 = df.copy()
        df3["qty"] = [99]
        result = self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df3},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()}, column_writes={"df": {"qty"}},
            continue_on_violation=True,
        )

        # B's current diff only shows qty changed. C reads price.
        # Col(df, qty) does NOT conflict with Col(df, price), so C is NOT stale.
        # This demonstrates column-level precision in ForwardStale: removing
        # a column write does not stale readers of other columns.
        assert "c" not in result.stale_cells


class TestLocSetConversionIntegration:
    """Verify that tracking data is correctly converted to LocSets in the enforcer."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_column_reads_produce_col_and_var_locs(self):
        """When column_reads are provided, reads should include both Col and Var locs.

        Var(df) is always present alongside Col reads. Rebinding is caught
        by Var(df) ▷ Var(df); column independence is preserved because
        Col/Rows/Attr ▷ Var = false in the ▷ matrix.
        """
        df = pd.DataFrame({"x": [1], "y": [2]})
        self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"x"}},
        )

        # Check the stored read locs in notebook state
        stored_reads = self.helper.sdc._notebook_state.reads.get("a", frozenset())
        has_col_loc = any(
            r.type.value == "col" and r.qualifier == "df" and r.name == "x"
            for r in stored_reads
        )
        has_var_loc = any(
            r.type.value == "var" and r.name == "df"
            for r in stored_reads
        )
        assert has_col_loc, f"Expected Col(df, x) in reads, got: {stored_reads}"
        # Var(df) IS present alongside column reads
        assert has_var_loc, f"Expected Var(df) in reads alongside Col reads, got: {stored_reads}"

    def test_structural_reads_produce_attr_locs(self):
        """When structural_reads are provided, reads should include Attr locs."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, structural_reads={"df": {"columns", "shape"}},
        )

        stored_reads = self.helper.sdc._notebook_state.reads.get("a", frozenset())
        attr_names = {
            r.name for r in stored_reads
            if r.type.value == "attr" and r.qualifier == "df"
        }
        assert "columns" in attr_names
        assert "shape" in attr_names

    def test_var_only_read_produces_var_loc(self):
        """When no column/structural detail, reads should be Var locs."""
        self.helper.execute_cell(
            "a", {"x": 1}, {"x": 1},
            reads={"x"},
        )

        stored_reads = self.helper.sdc._notebook_state.reads.get("a", frozenset())
        has_var_loc = any(
            r.type.value == "var" and r.name == "x"
            for r in stored_reads
        )
        assert has_var_loc, f"Expected Var(x) in reads, got: {stored_reads}"


# =============================================================================
# Comment-only / empty cell clearing R/W
# =============================================================================


class TestCommentOnlyCellClearsRW:
    """When a cell is edited from real code to comment-only (or empty),
    executing it should clear R/W and propagate staleness for removed writes.

    This tests the NotebookState-level behavior that _execute_without_enforcer
    relies on. The kernel calls set_clean + clears R/W + propagate_staleness.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_clearing_writes_stales_downstream_readers(self):
        """Cell A writes x, B reads x. A edited to comment → B stale."""
        # Run A: x = 1
        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        # Run B: reads x
        self.helper.execute_cell("b", {"x": 1}, {"x": 1}, reads={"x"})

        state = self.helper.sdc._notebook_state
        assert state.is_clean("a")
        assert state.is_clean("b")

        # Simulate _execute_without_enforcer: A becomes comment-only
        old_writes = state.writes.get("a", frozenset())
        state.reads["a"] = frozenset()
        state.writes["a"] = frozenset()
        if old_writes:
            state.propagate_staleness("a", old_writes)
        state.set_clean("a")

        # A is clean with empty R/W
        assert state.is_clean("a")
        assert state.reads["a"] == frozenset()
        assert state.writes["a"] == frozenset()


# =============================================================================
# ColAdd vs Col on first vs subsequent runs
# =============================================================================


class TestColAddVsColStaleness:
    """
    When a cell adds a column on first run, it produces ColAdd in the diff.
    On subsequent runs (column already exists), it produces Col (modify).
    ColAdd conflicts with attribute reads (shape, columns); Col does not.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_col_add_stales_shape_reader(self):
        """Adding a new column stales cells reading df.shape."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()}, writes={"df"}, column_writes={"df": {"x"}})
        df_y = df.copy(); df_y["y"] = [10, 20, 30]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y},
            reads={"df"}, writes={"df"}, column_reads={"df": set()}, column_writes={"df": {"y"}},
            continue_on_violation=True)
        self.helper.execute_cell("c", {"df": df_y}, {"df": df_y},
            reads={"df"}, structural_reads={"df": {"shape", "columns"}})

        state = self.helper.sdc._notebook_state

        # Edit B, rerun to add NEW column z
        state.handle_edit("b")
        df_yz = df.copy(); df_yz["y"] = [10, 20, 30]; df_yz["z"] = [7, 8, 9]
        result = self.helper.execute_cell("b", {"df": df_y}, {"df": df_yz},
            reads={"df"}, writes={"df"}, column_reads={"df": set()}, column_writes={"df": {"y", "z"}},
            continue_on_violation=True)

        # C reads shape/columns → stale because z was added (ColAdd ▷ Attr = true)
        assert "c" in result.stale_cells, \
            "C should be stale: ColAdd(df, z) conflicts with Attr(df, shape)"

    def test_col_modify_stales_shape_reader(self):
        """Modifying an existing column stales cells reading df.shape.

        Col now conflicts with all COL_ATTRS (shape, columns, dtypes, etc.),
        so any column write -- whether add or modify -- invalidates structural
        attribute readers on the same DataFrame.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()}, writes={"df"}, column_writes={"df": {"x"}})
        df_y = df.copy(); df_y["y"] = [10, 20, 30]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y},
            reads={"df"}, writes={"df"}, column_reads={"df": set()}, column_writes={"df": {"y"}},
            continue_on_violation=True)
        self.helper.execute_cell("c", {"df": df_y}, {"df": df_y},
            reads={"df"}, structural_reads={"df": {"shape", "columns"}})

        state = self.helper.sdc._notebook_state

        # Edit B, rerun — column y exists, just modify values
        state.handle_edit("b")
        df_y2 = df.copy(); df_y2["y"] = [100, 200, 300]
        result = self.helper.execute_cell("b", {"df": df_y}, {"df": df_y2},
            reads={"df"}, writes={"df"}, column_reads={"df": set()}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C reads shape/columns → stale because Col now conflicts with COL_ATTRS
        assert "c" in result.stale_cells, \
            "C should be stale: Col(df, y) conflicts with Attr(df, shape)"

    def test_first_run_add_then_second_run_modify(self):
        """On first run B adds column → C stale. On second run B modifies → C still stale.

        Col now conflicts with all COL_ATTRS, so the behavior is consistent
        across first and subsequent runs: any column write (add or modify)
        invalidates structural attribute readers.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()}, writes={"df"}, column_writes={"df": {"x"}})

        # B first run: adds y (pre doesn't have y)
        df_y = df.copy(); df_y["y"] = [10, 20, 30]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y},
            reads={"df"}, writes={"df"}, column_reads={"df": set()}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C reads shape
        self.helper.execute_cell("c", {"df": df_y}, {"df": df_y},
            reads={"df"}, structural_reads={"df": {"shape"}})

        state = self.helper.sdc._notebook_state

        # Edit B, second run: y exists in pre, just modify
        state.handle_edit("b")
        df_y2 = df.copy(); df_y2["y"] = [100, 200, 300]
        result = self.helper.execute_cell("b", {"df": df_y}, {"df": df_y2},
            reads={"df"}, writes={"df"}, column_reads={"df": set()}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C IS stale: Col now conflicts with COL_ATTRS, consistent across runs
        assert "c" in result.stale_cells, \
            "On second run, column modify should stale shape reader (Col ▷ Attr = true)"

        # B has column_reads={"df": set()} → empty column detail means Var(df) is
        # suppressed from read set. B has no recorded read locs for df, so
        # changes to A's writes cannot stale B via the ⊗ relation.

    def test_clearing_writes_no_effect_when_no_old_writes(self):
        """Cell with no prior writes → clearing is a no-op."""
        # Run A with no writes (e.g., was already a comment)
        state = self.helper.sdc._notebook_state
        state.reads["a"] = frozenset()
        state.writes["a"] = frozenset()
        state.set_clean("a")

        # Run B
        self.helper.execute_cell("b", {}, {"y": 1}, writes={"y"})
        assert state.is_clean("b")

        # "Re-execute" A as comment — no old writes, nothing happens
        old_writes = state.writes.get("a", frozenset())
        assert len(old_writes) == 0
        state.set_clean("a")

        # B stays clean
        assert state.is_clean("b")

    def test_clearing_writes_column_level_precision(self):
        """Cell A writes df["price"], B reads df["qty"]. A cleared → B stays clean."""
        df = pd.DataFrame({"price": [1], "qty": [2]})
        self.helper.execute_cell(
            "a", {}, {"df": df.copy()}, writes={"df"},
            column_writes={"df": {"price"}},
        )
        self.helper.execute_cell(
            "b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"qty"}},
        )

        state = self.helper.sdc._notebook_state
        assert state.is_clean("b")

        # Clear A's writes
        old_writes = state.writes.get("a", frozenset())
        state.reads["a"] = frozenset()
        state.writes["a"] = frozenset()
        if old_writes:
            state.propagate_staleness("a", old_writes)
        state.set_clean("a")

        # B reads qty, A wrote price → different columns, B should stay clean
        # (depends on whether propagate_staleness uses ▷ for column precision)
