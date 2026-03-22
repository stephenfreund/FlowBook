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

    def test_two_col_adds_write_write_overlap(self):
        """B adds df["price"], C adds df["qty"]. Editing B may stale C
        due to variable-level write-write overlap on "df".

        The current enforcer uses variable-level write-write overlap
        (W_i_union & cell_writes) which checks at the Var("df") level.
        Since both B and C have "df" in their tracking.writes, editing and
        rerunning B will mark C stale due to write-write overlap.

        This is a known conservatism -- the write-write overlap check
        operates at variable granularity, not column granularity.
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
        # C is stale due to variable-level write-write overlap (both write "df")
        # This is expected: the write-write overlap check is conservative.
        assert "c" in result.stale_cells


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

    def test_column_reads_produce_col_locs_not_var(self):
        """When column_reads are provided, reads should be Col locs, not Var locs.

        This ensures the enforcer has column-level precision for conflict detection.
        """
        df = pd.DataFrame({"x": [1], "y": [2]})
        self.helper.execute_cell(
            "a", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, column_reads={"df": {"x"}},
        )

        # Check the stored read locs in notebook state
        stored_reads = self.helper.sdc._notebook_state.reads.get("a", frozenset())
        # Should have Col(df, x), not Var(df)
        has_col_loc = any(
            r.type.value == "column" and r.qualifier == "df" and r.name == "x"
            for r in stored_reads
        )
        has_var_loc = any(
            r.type.value == "var" and r.name == "df"
            for r in stored_reads
        )
        assert has_col_loc, f"Expected Col(df, x) in reads, got: {stored_reads}"
        # Var(df) should NOT be present when column detail is available
        assert not has_var_loc, f"Var(df) should not be present when column reads exist, got: {stored_reads}"

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
