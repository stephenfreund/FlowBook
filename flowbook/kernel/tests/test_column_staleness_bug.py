"""
Test for column-independent staleness bug.

This test demonstrates the bug where adding a new column to a DataFrame
incorrectly marks cells that read OTHER columns as stale.

Example scenario (from geyser.ipynb):
- Cell A: df = pd.read_csv('old_faithful.csv')  # creates df with 'eruptions', 'waiting'
- Cell B: plt.scatter(df['eruptions'], df['waiting'])  # reads only these columns
- Cell C: df['cluster'] = kmeans.fit_predict(X)  # adds NEW column 'cluster'

BUG: After running C, cell B is incorrectly marked stale because:
- C writes to 'df' (variable-level)
- B reads from 'df' (variable-level)
- Variable-level overlap triggers staleness, bypassing the column-level check

EXPECTED: Cell B should NOT be stale because:
- B reads columns: {'eruptions', 'waiting'}
- C writes column: {'cluster'}
- {'cluster'} ∩ {'eruptions', 'waiting'} = ∅ (no column overlap)
"""

import pytest
import pandas as pd

from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper


class TestColumnIndependentStaleness:
    """Tests for column-level staleness precision."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_adding_column_does_not_stale_readers_of_other_columns(self):
        """
        Adding a new column should NOT mark cells reading other columns as stale.

        This is the exact scenario from geyser.ipynb that triggered the bug.
        """
        # Create a DataFrame with two columns
        df = pd.DataFrame({
            'eruptions': [3.6, 1.8, 3.3],
            'waiting': [79, 54, 74],
        })

        # Cell A: Creates the DataFrame
        result_a = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
            column_writes={"df": {"eruptions", "waiting"}},
        )
        assert not result_a.has_errors()

        # Cell B: Reads df['eruptions'] and df['waiting'] (makes a scatter plot)
        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},  # No change to namespace
            reads={"df"},
            column_reads={"df": {"eruptions", "waiting"}},
        )
        assert not result_b.has_errors()
        assert "b" not in result_b.stale_cells

        # Cell C: Adds df['cluster'] - a completely different column
        df_with_cluster = df.copy()
        df_with_cluster['cluster'] = [0, 1, 0]  # Simulated KMeans output

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df_with_cluster},
            reads={"df"},
            writes={"df"},
            column_reads={"df": {"eruptions", "waiting"}},  # C also reads these to compute clusters
            column_writes={"df": {"cluster"}},  # C only WRITES to 'cluster'
            continue_on_violation=True,  # C reads and writes df (NoReadAndWrite), but staleness should still propagate
        )

        # THE BUG: Cell B should NOT be stale!
        # B only read 'eruptions' and 'waiting'
        # C only wrote 'cluster'
        # There is no column overlap, so B should remain fresh
        assert "b" not in result_c.stale_cells, (
            "BUG: Cell B was incorrectly marked stale. "
            "It reads ['eruptions', 'waiting'] but Cell C only wrote ['cluster']. "
            "No column overlap means no staleness should occur."
        )

    def test_modifying_same_column_does_stale_readers(self):
        """
        Modifying a column SHOULD mark cells reading that column as stale.

        This is the correct behavior - only readers of the modified column become stale.
        """
        df = pd.DataFrame({
            'eruptions': [3.6, 1.8, 3.3],
            'waiting': [79, 54, 74],
        })

        # Cell A: Creates the DataFrame
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
        )

        # Cell B: Reads df['waiting']
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"waiting"}},
        )

        # Cell C: MODIFIES df['waiting'] (same column B reads)
        df_modified = df.copy()
        df_modified['waiting'] = df_modified['waiting'] + 10

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df_modified},
            reads={"df"},
            writes={"df"},
            column_reads={"df": {"waiting"}},
            column_writes={"df": {"waiting"}},  # Modifies 'waiting'
            continue_on_violation=True,  # Allow staleness propagation even with violations
        )

        # Cell B SHOULD be stale because C modified 'waiting' which B reads
        assert "b" in result_c.stale_cells, (
            "Cell B should be marked stale because Cell C modified 'waiting' "
            "which Cell B reads."
        )

    def test_no_column_info_var_read_not_staled_by_column_write(self):
        """
        When column info is missing, the reader has Var(df) which is binding-only.

        A column write (Col) does not conflict with a binding-only read (Var),
        so B should NOT be staled.
        """
        df = pd.DataFrame({
            'eruptions': [3.6, 1.8, 3.3],
            'waiting': [79, 54, 74],
        })

        # Cell A: Creates the DataFrame
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
        )

        # Cell B: Reads df but WITHOUT column-level tracking (Var(df) only)
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            # NO column_reads - produces Var(df), a binding-only read
        )

        # Cell C: Adds a new column WITH column tracking
        df_with_cluster = df.copy()
        df_with_cluster['cluster'] = [0, 1, 0]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df_with_cluster},
            reads={"df"},
            writes={"df"},
            column_writes={"df": {"cluster"}},
            continue_on_violation=True,  # Allow staleness propagation
        )

        # Cell B should NOT be stale: Var(df) is a binding-only read,
        # and Col(df, cluster) write does not conflict with it.
        assert "b" not in result_c.stale_cells, (
            "Cell B has Var(df) (binding-only read). "
            "A column write does not affect the binding, so B should not be stale."
        )


class TestForwardStalenessColumnAware:
    """Tests for forward staleness (later cell becomes stale when earlier cell writes)."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_forward_stale_independent_columns(self):
        """
        Forward staleness: Cell B reads col_a, Cell A (re-executed) writes col_b.
        Cell B should NOT become stale.
        """
        df = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})

        # Cell A: Creates df
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
            column_writes={"df": {"col_a", "col_b"}},
        )

        # Cell B: Reads df['col_a']
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"col_a"}},
        )

        # Re-run Cell A but only modify col_b
        df_modified = df.copy()
        df_modified['col_b'] = [30, 40]

        result_a2 = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df_modified},
            writes={"df"},
            column_writes={"df": {"col_b"}},  # Only wrote col_b
        )

        # Cell B should NOT be stale (reads col_a, A wrote col_b)
        assert "b" not in result_a2.stale_cells, (
            "Cell B reads col_a, Cell A wrote col_b. No overlap, no staleness."
        )

    def test_forward_stale_overlapping_columns(self):
        """
        Forward staleness: Cell B reads col_a, Cell A (re-executed) writes col_a.
        Cell B SHOULD become stale.
        """
        df = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})

        # Cell A: Creates df
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
            column_writes={"df": {"col_a", "col_b"}},
        )

        # Cell B: Reads df['col_a']
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"col_a"}},
        )

        # Re-run Cell A, modifying col_a
        df_modified = df.copy()
        df_modified['col_a'] = [10, 20]

        result_a2 = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df_modified},
            writes={"df"},
            column_writes={"df": {"col_a"}},  # Wrote col_a (same as B reads)
        )

        # Cell B SHOULD be stale (reads col_a, A wrote col_a)
        assert "b" in result_a2.stale_cells, (
            "Cell B reads col_a, Cell A wrote col_a. Overlap causes staleness."
        )

    def test_forward_stale_multiple_downstream_cells(self):
        """
        Forward staleness with multiple downstream cells reading different columns.
        Only cells reading the modified column should become stale.
        """
        df = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})

        # Cell A: Creates df
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
        )

        # Cell B: Reads df['x']
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"x"}},
        )

        # Cell C: Reads df['y']
        self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"y"}},
        )

        # Cell D: Reads df['z']
        self.helper.execute_cell(
            cell_id="d",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"z"}},
        )

        # Re-run Cell A, modifying only col y
        df_modified = df.copy()
        df_modified['y'] = [200]

        result_a2 = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df_modified},
            writes={"df"},
            column_writes={"df": {"y"}},
        )

        # Only Cell C should be stale (reads y)
        assert "b" not in result_a2.stale_cells, "B reads x, not affected by y change"
        assert "c" in result_a2.stale_cells, "C reads y, should be stale"
        assert "d" not in result_a2.stale_cells, "D reads z, not affected by y change"


class TestBackwardStalenessColumnAware:
    """Tests for backward staleness (earlier cell becomes stale when later cell writes)."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_backward_stale_independent_columns(self):
        """
        Backward staleness: Cell A reads col_a, Cell C writes col_b.
        Cell A should NOT become stale.
        """
        df = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})

        # Cell A: Reads df['col_a']
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"col_a"}},
        )

        # Cell B: Some intermediate cell
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
        )

        # Cell C: Writes to df['col_b'] (different column than A reads)
        df_modified = df.copy()
        df_modified['col_b'] = [30, 40]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df_modified},
            writes={"df"},
            column_writes={"df": {"col_b"}},
        )

        # Cell A should NOT be stale (reads col_a, C wrote col_b)
        assert "a" not in result_c.stale_cells, (
            "Cell A reads col_a, Cell C wrote col_b. No overlap, no staleness."
        )

    def test_backward_stale_overlapping_columns(self):
        """
        Backward staleness: Cell A reads col_a, Cell C writes col_a.
        Cell A SHOULD become stale.
        """
        df = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})

        # Cell A: Reads df['col_a']
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"col_a"}},
        )

        # Cell B: Some intermediate cell
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
        )

        # Cell C: Writes to df['col_a'] (same column A reads)
        df_modified = df.copy()
        df_modified['col_a'] = [10, 20]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df_modified},
            writes={"df"},
            column_writes={"df": {"col_a"}},
            continue_on_violation=True,  # Allow staleness propagation
        )

        # Cell A SHOULD be stale (reads col_a, C wrote col_a)
        assert "a" in result_c.stale_cells, (
            "Cell A reads col_a, Cell C wrote col_a. Overlap causes staleness."
        )


class TestMixedVariablesAndDataFrames:
    """Tests for scenarios mixing regular variables and DataFrames with columns."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_regular_variable_still_triggers_staleness(self):
        """
        Regular variables (without column tracking) should still trigger staleness.
        """
        # Cell A: Writes x
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )

        # Cell B: Reads x
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )

        # Re-run Cell A with new value for x
        result_a2 = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 100},  # Changed value
            writes={"x"},
        )

        # Cell B SHOULD be stale (reads x, A wrote x)
        assert "b" in result_a2.stale_cells, (
            "Regular variable x changed. Cell B reads x, should be stale."
        )

    def test_mixed_df_and_regular_var_independent(self):
        """
        DataFrame column change should not affect cells only reading regular variables.
        """
        df = pd.DataFrame({'col': [1, 2]})

        # Cell A: Creates df and x
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df, "x": 10},
            writes={"df", "x"},
        )

        # Cell B: Reads only x (not df)
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df, "x": 10},
            post_namespace={"df": df, "x": 10},
            reads={"x"},
        )

        # Cell C: Modifies df['col'] (not x)
        df_modified = df.copy()
        df_modified['col'] = [100, 200]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df, "x": 10},
            post_namespace={"df": df_modified, "x": 10},
            writes={"df"},
            column_writes={"df": {"col"}},
        )

        # Cell B should NOT be stale (reads x, C only wrote to df)
        assert "b" not in result_c.stale_cells, (
            "Cell B reads x, Cell C wrote df['col']. No overlap."
        )

    def test_mixed_df_and_regular_var_both_change(self):
        """
        When both df column and regular var change, only affected cells become stale.
        """
        df = pd.DataFrame({'col_a': [1], 'col_b': [2]})

        # Cell A: Creates df and x
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df, "x": 10},
            writes={"df", "x"},
        )

        # Cell B: Reads df['col_a'] and x
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df, "x": 10},
            post_namespace={"df": df, "x": 10},
            reads={"df", "x"},
            column_reads={"df": {"col_a"}},
        )

        # Cell C: Modifies df['col_b'] (not col_a), does NOT change x
        df_modified = df.copy()
        df_modified['col_b'] = [200]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df, "x": 10},
            post_namespace={"df": df_modified, "x": 10},
            writes={"df"},
            column_writes={"df": {"col_b"}},
        )

        # Cell B should NOT be stale:
        # - B reads df['col_a'], C wrote df['col_b'] (no overlap)
        # - B reads x, but x didn't change
        assert "b" not in result_c.stale_cells, (
            "Cell B reads col_a and x. Cell C wrote col_b. No relevant overlap."
        )

    def test_mixed_df_and_regular_var_regular_changes(self):
        """
        When regular var changes but df column doesn't overlap, cell reading both becomes stale.
        """
        df = pd.DataFrame({'col_a': [1], 'col_b': [2]})

        # Cell A: Creates df and x
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df, "x": 10},
            writes={"df", "x"},
        )

        # Cell B: Reads df['col_a'] and x
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df, "x": 10},
            post_namespace={"df": df, "x": 10},
            reads={"df", "x"},
            column_reads={"df": {"col_a"}},
        )

        # Cell C: Modifies df['col_b'] AND changes x
        df_modified = df.copy()
        df_modified['col_b'] = [200]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df, "x": 10},
            post_namespace={"df": df_modified, "x": 999},  # x changed!
            writes={"df", "x"},
            column_writes={"df": {"col_b"}},
            continue_on_violation=True,  # Allow staleness propagation
        )

        # Cell B SHOULD be stale because x changed (even though df columns don't overlap)
        assert "b" in result_c.stale_cells, (
            "Cell B reads x. Cell C changed x. B should be stale."
        )


class TestEdgeCases:
    """Edge cases for column-aware staleness."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_writer_no_column_info_reader_has_column_info(self):
        """
        Writer has no column info, reader has column info.
        Should be conservative (mark stale).
        """
        df = pd.DataFrame({'x': [1], 'y': [2]})

        # Cell A: Creates df
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
        )

        # Cell B: Reads df['x'] with column tracking
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": {"x"}},
        )

        # Cell C: Writes df WITHOUT column info
        df_modified = df.copy()
        df_modified['y'] = [200]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df},
            post_namespace={"df": df_modified},
            writes={"df"},
            # NO column_writes - conservative case
        )

        # Cell B SHOULD be stale (conservative: writer has no column info)
        assert "b" in result_c.stale_cells, (
            "Writer has no column info, must be conservative and mark stale."
        )

    def test_empty_column_sets(self):
        """
        Empty column read/write sets should not cause errors.
        """
        df = pd.DataFrame({'x': [1]})

        # Cell A: Creates df
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
            column_writes={"df": set()},  # Empty column set
        )

        # Cell B: Reads df with empty column set
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
            column_reads={"df": set()},  # Empty column set
        )

        # Re-run Cell A
        result_a2 = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
            column_writes={"df": {"x"}},
        )

        # With empty column reads, should be conservative
        # (empty set means "didn't track columns", not "reads nothing")
        # This depends on implementation - empty set is treated as "no info"
        # Just verify no crash
        assert not result_a2.has_errors()

    def test_multiple_dataframes_independent(self):
        """
        Changes to one DataFrame should not affect cells reading a different DataFrame.
        """
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]})

        # Cell A: Creates both DataFrames
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df1": df1, "df2": df2},
            writes={"df1", "df2"},
        )

        # Cell B: Reads df1['a']
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df1": df1, "df2": df2},
            post_namespace={"df1": df1, "df2": df2},
            reads={"df1"},
            column_reads={"df1": {"a"}},
        )

        # Cell C: Modifies df2['b'] (different DataFrame entirely)
        df2_modified = df2.copy()
        df2_modified['b'] = [200]

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df1": df1, "df2": df2},
            post_namespace={"df1": df1, "df2": df2_modified},
            writes={"df2"},
            column_writes={"df2": {"b"}},
        )

        # Cell B should NOT be stale (reads df1, C wrote df2)
        assert "b" not in result_c.stale_cells, (
            "Cell B reads df1, Cell C wrote df2. Different variables, no staleness."
        )
