"""
Exploration: Operation-Based "May-Modify" Write Sets.

PROBLEM
-------
The current system determines write location types (ColAdd vs Col) by diffing
pre/post checkpoints. This means the SAME code produces DIFFERENT write sets
on first vs second execution:

    df['y'] = [1, 2, 3]

  First run:  pre has no 'y' → diff says ColumnAdded  → ColAdd(df, y) → stales shape readers
  Second run: pre has 'y'    → diff says ColumnModified → Col(df, y)  → does NOT stale shape readers

This inconsistency means staleness propagation is non-deterministic — it depends
on history, not just on the code.

PROPOSED ALTERNATIVE
--------------------
Instead of using the diff to classify writes, use the OPERATION TYPE to
determine what the operation MAY modify. For df['col'] = val (__setitem__),
the operation may add OR modify, so we produce a conservative write that
conflicts with both ColAdd and Col targets.

This is modeled here as a hypothetical "ColSet" write type:

    ColSet(d, c) ▷ r  =  Col(d, c) ▷ r  OR  ColAdd(d, c) ▷ r

Concretely:
    ColSet(df, c) ▷ Col(df, c)      = true  (same column data)
    ColSet(df, c) ▷ Col(df, other)  = false  (column independence)
    ColSet(df, c) ▷ Attr(df, shape) = true  (may add → shape change)
    ColSet(df, c) ▷ Attr(df, cols)  = true  (may add → columns change)
    ColSet(df, c) ▷ Attr(df, vals)  = true  (data or structure changed)
    ColSet(df, c) ▷ Var(x)         = false  (binding not changed)

TEST STRUCTURE
--------------
Each test documents:
  - The scenario
  - Current behavior (diff-based)
  - May-modify behavior (operation-based)
  - Whether the change is a "fix" (removes inconsistency) or
    a "false positive" (over-conservative staleness/violation)

The operations we catalog:

  Operation                  | Current Write Type   | May-Modify Write Type
  ---------------------------|----------------------|----------------------
  df['col'] = val            | Col OR ColAdd (diff) | ColSet (conservative)
  df.loc[:, 'col'] = val     | Col OR ColAdd (diff) | ColSet (conservative)
  df.insert(i, col, val)     | ColAdd (always)      | ColAdd (same, unambiguous)
  del df['col']              | ColDel (always)      | ColDel (same, unambiguous)
  df.drop(columns=[c])       | ColDel (always)      | ColDel (same, unambiguous)
  sort_values(inplace=True)  | Rows + Col* (diff)   | Rows + Col* (same)
  reset_index(inplace=True)  | ColAdd + Index (diff) | ColAdd + Index (same-ish)
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper
from flowbook.kernel.models import ErrorType
from flowbook.kernel.locations import (
    ReadLoc, WriteLoc, ReadLocType, WriteLocType,
    write_conflicts_read, wlocs_conflict_rlocs,
    COL_ATTRS, COL_VALUE_ATTRS, ROW_ATTRS,
)


# ============================================================================
# Helpers: Simulate ColSet behavior
# ============================================================================

def col_set_conflicts_read(qualifier, column, r: ReadLoc) -> bool:
    """
    ColSet(d, c) ▷ r  =  Col(d, c) ▷ r  OR  ColAdd(d, c) ▷ r

    The conservative "may add or modify" conflict check.
    """
    col_w = WriteLoc.col(qualifier, column)
    col_add_w = WriteLoc.col_add(qualifier, column)
    return (
        write_conflicts_read(col_w, r) or
        write_conflicts_read(col_add_w, r)
    )


def col_set_write_set(qualifier, column):
    """Return both Col and ColAdd writes to simulate ColSet behavior."""
    return frozenset({
        WriteLoc.col(qualifier, column),
        WriteLoc.col_add(qualifier, column),
    })


# ============================================================================
# Category 1: The Inconsistency Problem (motivating cases)
# ============================================================================

class TestInconsistencyProblem:
    """
    These tests demonstrate the CORE problem: same code produces different
    write types on first vs second execution.

    With may-modify (ColSet), behavior is CONSISTENT across runs.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_first_run_add_stales_shape_reader(self):
        """
        First run: B adds column y → C reads df.shape → C stale.
        This is CORRECT — shape changed from (3,1) to (3,2).

        Current: ColAdd(df,y) ▷ Attr(df,shape) = true  → C stale ✓
        May-mod: ColSet(df,y) ▷ Attr(df,shape) = true  → C stale ✓  (same)
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x"}})

        # B adds column y (not in pre)
        df_y = df.copy(); df_y["y"] = [10, 20, 30]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        self.helper.execute_cell("c", {"df": df_y}, {"df": df_y},
            reads={"df"}, structural_reads={"df": {"shape"}})

        state = self.helper.sdc._notebook_state

        # Edit B → rerun (still first conceptual run — pre lacks y)
        state.handle_edit("b")
        df_y2 = df.copy(); df_y2["y"] = [100, 200, 300]
        result = self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y2},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C stale: ColAdd(df,y) ▷ Attr(df,shape) = true
        assert "c" in result.stale_cells

    def test_second_run_modify_does_not_stale_shape_reader__inconsistency(self):
        """
        Second run: B modifies existing column y → C reads df.shape → C NOT stale.

        Current: Col(df,y) ▷ Attr(df,shape) = false → C not stale
        May-mod: ColSet(df,y) ▷ Attr(df,shape) = true → C WOULD be stale

        This is the INCONSISTENCY: same code `df['y'] = values`, but different
        staleness because the pre-checkpoint state differs.

        *** May-modify FIXES this inconsistency (at cost of false positive here) ***
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x"}})

        # B first run: adds y
        df_y = df.copy(); df_y["y"] = [10, 20, 30]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C reads shape
        self.helper.execute_cell("c", {"df": df_y}, {"df": df_y},
            reads={"df"}, structural_reads={"df": {"shape"}})

        state = self.helper.sdc._notebook_state

        # Edit B, second run: y already in pre, just modify values
        state.handle_edit("b")
        df_y2 = df.copy(); df_y2["y"] = [100, 200, 300]
        result = self.helper.execute_cell("b", {"df": df_y}, {"df": df_y2},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # CURRENT: C not stale (Col ▷ shape = false)
        assert "c" not in result.stale_cells

        # MAY-MODIFY WOULD CHANGE THIS:
        # ColSet(df,y) ▷ Attr(df,shape) = true → C would be stale
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "shape"))

    def test_inconsistency_with_columns_reader(self):
        """
        Same inconsistency but with df.columns reader.

        Current: Col(df,y) ▷ Attr(df,columns) = false → not stale on rerun
        May-mod: ColSet(df,y) ▷ Attr(df,columns) = true → stale (consistent)
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x"}})

        # B adds y
        df_y = df.copy(); df_y["y"] = [10, 20, 30]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df_y},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C reads df.columns
        self.helper.execute_cell("c", {"df": df_y}, {"df": df_y},
            reads={"df"}, structural_reads={"df": {"columns"}})

        state = self.helper.sdc._notebook_state

        # Edit B, re-run with y already present
        state.handle_edit("b")
        df_y2 = df.copy(); df_y2["y"] = [100, 200, 300]
        result = self.helper.execute_cell("b", {"df": df_y}, {"df": df_y2},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # CURRENT: not stale
        assert "c" not in result.stale_cells

        # MAY-MODIFY: would be stale
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "columns"))

    def test_inconsistency_with_dtypes_reader(self):
        """
        Same pattern with df.dtypes reader.

        Current: Col(df,y) ▷ Attr(df,dtypes) = false → not stale on rerun
        May-mod: ColSet(df,y) ▷ Attr(df,dtypes) = true → stale (consistent)
        """
        assert not write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "dtypes"))
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "dtypes"))


# ============================================================================
# Category 2: False Positives (over-conservative staleness)
# ============================================================================

class TestFalsePositives:
    """
    Cases where may-modify produces staleness that current system correctly avoids.

    These are the COST of the conservative approach: cells get marked stale
    when they don't need to be re-run.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_modify_existing_col_false_stales_shape_reader(self):
        """
        B modifies existing column y (pure value update, no structural change).
        C reads df.shape. Shape didn't change.

        Current: Col(df,y) ▷ Attr(df,shape) = false → C not stale ✓ CORRECT
        May-mod: ColSet(df,y) ▷ Attr(df,shape) = true → C stale (FALSE POSITIVE)
        """
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x", "y"}})

        # B modifies y values (y already exists in df)
        df2 = df.copy(); df2["y"] = [100, 200, 300]
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # C reads shape
        self.helper.execute_cell("c", {"df": df2}, {"df": df2},
            reads={"df"}, structural_reads={"df": {"shape"}})

        state = self.helper.sdc._notebook_state

        # Edit B, rerun — y exists, just change values
        state.handle_edit("b")
        df3 = df.copy(); df3["y"] = [999, 888, 777]
        result = self.helper.execute_cell("b", {"df": df2}, {"df": df3},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # CURRENT: C not stale (correct — shape unchanged)
        assert "c" not in result.stale_cells

        # MAY-MODIFY: C would be stale (false positive — shape didn't change)
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "shape"))

    def test_modify_existing_col_false_stales_columns_reader(self):
        """
        B modifies existing column values. C reads df.columns.
        The set of columns didn't change.

        Current: not stale ✓
        May-mod: stale (FALSE POSITIVE — columns Index unchanged)
        """
        df = pd.DataFrame({"x": [1], "y": [2]})

        # Col write does NOT conflict with columns attr
        assert not write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "columns"))

        # ColSet DOES conflict with columns attr (from ColAdd arm)
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "columns"))

    def test_modify_existing_col_false_stales_size_reader(self):
        """
        Same false positive for df.size reader.

        size = rows * cols. Modifying existing column values doesn't change size.

        Current: not stale ✓
        May-mod: stale (FALSE POSITIVE)
        """
        assert not write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "size"))
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "size"))

    def test_modify_existing_col_false_stales_describe_reader(self):
        """
        B modifies column y. C calls df.describe().

        describe() shows statistics. Modifying ANY column DOES change describe().
        Current ALSO flags this (values/T/describe are in COL_VALUE_ATTRS).

        Current: Col(df,y) ▷ Attr(df,describe) = true  → stale ✓
        May-mod: ColSet(df,y) ▷ Attr(df,describe) = true → stale ✓ (same)
        """
        # Both approaches agree: modifying column values affects describe()
        assert write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "describe"))
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "describe"))

    def test_modify_existing_col_false_stales_values_reader(self):
        """
        B modifies column y. C reads df.values (full 2D array).

        df.values exposes all column data. Both approaches agree this is stale.

        Current: Col(df,y) ▷ Attr(df,values) = true → stale ✓
        May-mod: ColSet(df,y) ▷ Attr(df,values) = true → stale ✓ (same)
        """
        assert write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "values"))
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "values"))


# ============================================================================
# Category 3: Preserved Column Independence
# ============================================================================

class TestColumnIndependence:
    """
    Column independence is preserved by BOTH approaches.
    Writing column A does not stale readers of column B.
    """

    def test_col_vs_other_col_no_conflict(self):
        """Both Col and ColSet for column A don't conflict with reading column B."""
        # Current: Col(df, price) ▷ Col(df, qty) = false
        assert not write_conflicts_read(
            WriteLoc.col("df", "price"), ReadLoc.col("df", "qty"))

        # May-modify: ColSet(df, price) ▷ Col(df, qty) = false
        assert not col_set_conflicts_read("df", "price", ReadLoc.col("df", "qty"))

    def test_col_add_vs_other_col_no_conflict(self):
        """Adding column A doesn't conflict with reading column B data."""
        # ColAdd(df, new) ▷ Col(df, existing) = false
        assert not write_conflicts_read(
            WriteLoc.col_add("df", "new"), ReadLoc.col("df", "existing"))

        # ColSet(df, new) ▷ Col(df, existing) = false
        assert not col_set_conflicts_read("df", "new", ReadLoc.col("df", "existing"))

    def test_same_column_always_conflicts(self):
        """Writing column A always stales readers of column A."""
        # Current: Col(df, y) ▷ Col(df, y) = true
        assert write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.col("df", "y"))

        # May-modify: ColSet(df, y) ▷ Col(df, y) = true
        assert col_set_conflicts_read("df", "y", ReadLoc.col("df", "y"))


# ============================================================================
# Category 4: Validity Predicate Impact (NoWriteAfterRead)
# ============================================================================

class TestValidityPredicateImpact:
    """
    This is the most SERIOUS consequence of the conservative approach.

    The four validity predicates use the ▷ relation to detect conflicts.
    If we use ColSet instead of Col, we get FALSE VIOLATIONS that BLOCK
    execution (not just false staleness).
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_no_write_after_read_false_violation(self):
        """
        Cell A reads df.shape. Cell B modifies existing column.
        NoWriteAfterRead checks: W_B ∩ R_{before B}

        Current: Col(df,y) ▷ Attr(df,shape) = false → NO violation ✓
        May-mod: ColSet(df,y) ▷ Attr(df,shape) = true → VIOLATION (false!)

        *** This is a BLOCKING false positive — execution would be rejected ***
        """
        df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})

        # A creates df and reads shape
        self.helper.execute_cell("a", {}, {"df": df.copy()},
            writes={"df"}, column_writes={"df": {"x", "y"}})

        # B reads shape in a pass-through (e.g., print(df.shape))
        self.helper.execute_cell("b", {"df": df.copy()}, {"df": df.copy()},
            reads={"df"}, structural_reads={"df": {"shape"}})

        # C modifies existing column y
        df2 = df.copy(); df2["y"] = [99, 88]
        result = self.helper.execute_cell("c", {"df": df.copy()}, {"df": df2},
            reads={"df"}, writes={"df"}, column_writes={"df": {"y"}},
            continue_on_violation=True)

        # Current: NO violation — Col(df,y) doesn't conflict with shape
        no_write_after_read_errors = [
            e for e in result.errors
            if e.error_type == ErrorType.NO_WRITE_AFTER_READ
        ]
        assert len(no_write_after_read_errors) == 0

        # MAY-MODIFY WOULD CAUSE a false NoWriteAfterRead violation here:
        # ColSet(df,y) ▷ Attr(df,shape) = true
        # But shape didn't actually change! Purely a value update.
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "shape"))

    def test_no_read_before_write_false_violation(self):
        """
        Cell A reads df['x']. Cell B (earlier in order) sets df['y'] = val.
        NoReadBeforeWrite checks: R_A ∩ W_{after A}

        If B writes ColSet(df,y), does it conflict with A's shape read?
        Not with Col reads, but with structural reads.

        Current: not a violation (Col ▷ shape = false)
        May-mod: violation if A reads df.shape (false positive)
        """
        # Structural read of shape conflicts with ColSet
        assert not write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "shape"))
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "shape"))

        # But column reads are fine — column independence preserved
        assert not col_set_conflicts_read("df", "y", ReadLoc.col("df", "x"))

    def test_no_read_and_write_same_col_still_detected(self):
        """
        df['x'] = df['x'] + 1  (reads and writes same column)

        Both approaches detect this correctly.

        Current: Col(df,x) ▷ Col(df,x) = true → NoReadAndWrite violation ✓
        May-mod: ColSet(df,x) ▷ Col(df,x) = true → violation ✓ (same)
        """
        assert write_conflicts_read(
            WriteLoc.col("df", "x"), ReadLoc.col("df", "x"))
        assert col_set_conflicts_read("df", "x", ReadLoc.col("df", "x"))


# ============================================================================
# Category 5: Unambiguous Operations (no behavior change)
# ============================================================================

class TestUnambiguousOperations:
    """
    Some DataFrame operations are inherently unambiguous about what they modify.
    The may-modify approach would NOT change their behavior.
    """

    def test_insert_always_adds(self):
        """
        df.insert(0, 'col', val) — always adds. No ambiguity.
        Both approaches produce ColAdd.
        """
        # ColAdd is the correct and only interpretation
        w = WriteLoc.col_add("df", "new_col")
        assert write_conflicts_read(w, ReadLoc.attr("df", "shape"))
        assert write_conflicts_read(w, ReadLoc.attr("df", "columns"))
        assert not write_conflicts_read(w, ReadLoc.col("df", "existing"))

    def test_del_always_removes(self):
        """
        del df['col'] — always removes. No ambiguity.
        Both approaches produce ColDel.
        """
        w = WriteLoc.col_del("df", "old_col")
        assert write_conflicts_read(w, ReadLoc.attr("df", "shape"))
        assert write_conflicts_read(w, ReadLoc.attr("df", "columns"))
        assert write_conflicts_read(w, ReadLoc.col("df", "old_col"))
        assert not write_conflicts_read(w, ReadLoc.col("df", "other"))

    def test_drop_columns_always_removes(self):
        """
        df.drop(columns=['col'], inplace=True) — always removes.
        Both approaches produce ColDel.
        """
        w = WriteLoc.col_del("df", "dropped")
        assert write_conflicts_read(w, ReadLoc.attr("df", "columns"))

    def test_rows_change_unambiguous(self):
        """
        Row additions/removals — always Rows write. Unambiguous.
        Both approaches produce Rows(df).
        """
        w = WriteLoc.rows("df")
        assert write_conflicts_read(w, ReadLoc.col("df", "any_col"))
        assert write_conflicts_read(w, ReadLoc.attr("df", "shape"))
        assert write_conflicts_read(w, ReadLoc.attr("df", "index"))
        assert not write_conflicts_read(w, ReadLoc.attr("df", "columns"))


# ============================================================================
# Category 6: Complete Conflict Matrix for ColSet
# ============================================================================

class TestColSetConflictMatrix:
    """
    Exhaustive test of ColSet ▷ r for all ReadLoc types.
    Documents the full behavior of the proposed may-modify approach.
    """

    def test_col_set_vs_var_read(self):
        """ColSet(df, c) ▷ Var(x) = false for all x.
        Column operations never change variable bindings."""
        assert not col_set_conflicts_read("df", "y", ReadLoc.var("df"))
        assert not col_set_conflicts_read("df", "y", ReadLoc.var("other"))

    def test_col_set_vs_same_col_read(self):
        """ColSet(df, c) ▷ Col(df, c) = true.
        Writing a column (add or modify) invalidates reading that column."""
        assert col_set_conflicts_read("df", "y", ReadLoc.col("df", "y"))

    def test_col_set_vs_different_col_read(self):
        """ColSet(df, c) ▷ Col(df, other) = false.
        Column independence: writing column A doesn't affect column B."""
        assert not col_set_conflicts_read("df", "y", ReadLoc.col("df", "x"))

    def test_col_set_vs_different_df_col_read(self):
        """ColSet(df1, c) ▷ Col(df2, c) = false.
        Different DataFrames are independent."""
        assert not col_set_conflicts_read("df1", "y", ReadLoc.col("df2", "y"))

    @pytest.mark.parametrize("attr", sorted(COL_ATTRS))
    def test_col_set_vs_col_attrs(self, attr):
        """ColSet(df, c) ▷ Attr(df, a) = true for all a in COL_ATTRS.

        This is the UNION of Col and ColAdd behavior:
        - Col only conflicts with COL_VALUE_ATTRS (values, T, describe)
        - ColAdd conflicts with all COL_ATTRS

        ColSet is conservative: conflicts with all COL_ATTRS.

        For attrs NOT in COL_VALUE_ATTRS (like shape, columns, dtypes, size,
        keys, axes, iter), this is MORE conservative than current Col behavior.
        """
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", attr))

        # Show which are "new" conflicts vs already existing
        col_already_conflicts = write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", attr))
        if not col_already_conflicts:
            # This attr is a NEW conflict added by the ColAdd arm
            assert attr not in COL_VALUE_ATTRS

    def test_col_set_vs_non_col_attr(self):
        """ColSet(df, c) ▷ Attr(df, 'index') = false.
        Index is a ROW_ATTR, not a COL_ATTR. Not affected by column writes."""
        # 'index' is in ROW_ATTRS but not COL_ATTRS
        if "index" not in COL_ATTRS:
            assert not col_set_conflicts_read("df", "y", ReadLoc.attr("df", "index"))

    def test_col_set_vs_file_read(self):
        """ColSet(df, c) ▷ File(p) = false."""
        assert not col_set_conflicts_read("df", "y", ReadLoc.file("data.csv"))

    def test_attrs_that_differ_between_col_and_col_set(self):
        """
        Enumerate exactly which attributes produce DIFFERENT results
        between Col and ColSet.

        These are the attrs in COL_ATTRS but NOT in COL_VALUE_ATTRS:
        the attrs where ColSet is MORE conservative than Col.
        """
        new_conflict_attrs = COL_ATTRS - COL_VALUE_ATTRS
        for attr in sorted(new_conflict_attrs):
            # Col does NOT conflict with these
            assert not write_conflicts_read(
                WriteLoc.col("df", "y"), ReadLoc.attr("df", attr)), \
                f"Col(df,y) should NOT conflict with Attr(df,{attr})"

            # ColSet DOES conflict with these
            assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", attr)), \
                f"ColSet(df,y) SHOULD conflict with Attr(df,{attr})"

        # These are exactly: columns, keys, dtypes, axes, shape, size, iter
        expected_new = {"columns", "keys", "dtypes", "axes", "shape", "size", "iter"}
        assert new_conflict_attrs == expected_new, \
            f"Expected new conflicts: {expected_new}, got: {new_conflict_attrs}"


# ============================================================================
# Category 7: Mixed Operations in One Cell
# ============================================================================

class TestMixedOperations:
    """
    A single cell may do multiple DataFrame operations.
    Test how the combined write set differs.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_cell_modifies_existing_and_adds_new(self):
        """
        Cell B: df['existing'] = new_vals; df['brand_new'] = vals

        Current (diff):
          Col(df, existing) + ColAdd(df, brand_new)
          → shape reader stale because of ColAdd

        May-modify:
          ColSet(df, existing) + ColSet(df, brand_new)
          → shape reader stale because BOTH are ColSet

        Same result here because the ColAdd from brand_new already stales.
        But if we remove the brand_new line, behavior diverges (see Category 2).
        """
        # Both approaches stale shape reader when there's a genuine new column
        col_w = WriteLoc.col("df", "existing")
        col_add_w = WriteLoc.col_add("df", "brand_new")
        writes = frozenset({col_w, col_add_w})

        shape_read = frozenset({ReadLoc.attr("df", "shape")})
        conflicts = wlocs_conflict_rlocs(writes, shape_read)
        assert col_add_w in conflicts  # ColAdd conflicts with shape
        assert col_w not in conflicts  # Col does not

    def test_cell_modifies_two_existing_cols(self):
        """
        Cell B: df['x'] = vals; df['y'] = vals  (both exist)

        Current: Col(df,x) + Col(df,y) → no shape staleness ✓
        May-mod: ColSet(df,x) + ColSet(df,y) → shape staleness (false positive!)

        With may-modify, ANY cell that writes to ANY column of a DataFrame
        will stale all downstream structural readers. This is a significant
        false positive rate for notebooks that do feature engineering.
        """
        # Current: no structural conflict
        writes_current = frozenset({
            WriteLoc.col("df", "x"),
            WriteLoc.col("df", "y"),
        })
        shape_reads = frozenset({ReadLoc.attr("df", "shape")})
        assert not wlocs_conflict_rlocs(writes_current, shape_reads)

        # May-modify: structural conflict from BOTH columns
        writes_may = col_set_write_set("df", "x") | col_set_write_set("df", "y")
        assert wlocs_conflict_rlocs(writes_may, shape_reads)


# ============================================================================
# Category 8: The Attribute Catalog
# ============================================================================

class TestOperationMayModifyCatalog:
    """
    Catalog of DataFrame operations and what they MAY modify.

    This documents the "hardcoded may-modify table" that the alternative
    approach would use. Some operations are unambiguous (insert = always add),
    some are ambiguous (setitem = may add or modify).

    For each operation, we list:
    - The operation
    - What it may produce as WriteLocs
    - Whether this differs from diff-based detection
    """

    def test_setitem_may_modify(self):
        """
        df['col'] = val

        May produce: ColSet(df, col) = Col(df,col) ∪ ColAdd(df,col)
        Differs from diff-based: YES — diff distinguishes add vs modify
        """
        # The whole point: setitem is ambiguous
        assert col_set_conflicts_read("df", "c", ReadLoc.attr("df", "shape"))

    def test_insert_always_adds(self):
        """
        df.insert(loc, col, val)

        Always produces: ColAdd(df, col)
        Differs from diff-based: NO — always an add
        """
        w = WriteLoc.col_add("df", "c")
        assert write_conflicts_read(w, ReadLoc.attr("df", "shape"))

    def test_delitem_always_removes(self):
        """
        del df['col']

        Always produces: ColDel(df, col)
        Differs from diff-based: NO — always a delete
        """
        w = WriteLoc.col_del("df", "c")
        assert write_conflicts_read(w, ReadLoc.attr("df", "shape"))

    def test_drop_columns_always_removes(self):
        """
        df.drop(columns=['col'], inplace=True)

        Always produces: ColDel(df, col) for each dropped column
        Differs from diff-based: NO — always a delete
        """
        w = WriteLoc.col_del("df", "c")
        assert write_conflicts_read(w, ReadLoc.attr("df", "columns"))

    def test_loc_setitem_may_modify(self):
        """
        df.loc[:, 'col'] = val

        Same ambiguity as __setitem__: may add or modify.
        May produce: ColSet(df, col)
        Differs from diff-based: YES
        """
        assert col_set_conflicts_read("df", "c", ReadLoc.attr("df", "shape"))

    def test_sort_values_inplace_reorders_rows(self):
        """
        df.sort_values('col', inplace=True)

        Modifies: index/row order. Column values stay aligned but index changes.
        Produces: Rows(df) [+ possibly Attr(df, index)]
        Differs from diff-based: NO — always a row/index change
        """
        w = WriteLoc.rows("df")
        assert write_conflicts_read(w, ReadLoc.attr("df", "index"))
        assert write_conflicts_read(w, ReadLoc.col("df", "any"))

    def test_reset_index_may_add_column(self):
        """
        df.reset_index(inplace=True)

        Moves index to column (adds column), resets to RangeIndex.
        Produces: ColAdd(df, old_index_name) + Attr(df, index)
        Differs from diff-based: Potentially — if index was already reset.
                                  Usually unambiguous (always adds the index column).
        """
        w_col = WriteLoc.col_add("df", "old_index")
        w_idx = WriteLoc.attr("df", "index")
        assert write_conflicts_read(w_col, ReadLoc.attr("df", "columns"))
        assert write_conflicts_read(w_idx, ReadLoc.attr("df", "index"))

    def test_rename_columns_adds_and_removes(self):
        """
        df.rename(columns={'old': 'new'}, inplace=True)

        Renames: removes old name, adds new name. Values don't change.
        Produces: ColDel(df, old) + ColAdd(df, new)
        Differs from diff-based: NO — always a rename
        """
        w_del = WriteLoc.col_del("df", "old")
        w_add = WriteLoc.col_add("df", "new")
        assert write_conflicts_read(w_del, ReadLoc.col("df", "old"))
        assert write_conflicts_read(w_add, ReadLoc.attr("df", "columns"))
        # The new column name doesn't conflict with reads of unrelated columns
        assert not write_conflicts_read(w_add, ReadLoc.col("df", "unrelated"))

    def test_assign_returns_new_df(self):
        """
        df2 = df.assign(new_col=val)

        Returns a NEW DataFrame. The original df is NOT modified.
        Produces: Var(df2) — whole-variable write (reassignment)
        No ambiguity about in-place modification.
        """
        w = WriteLoc.var("df2")
        assert write_conflicts_read(w, ReadLoc.var("df2"))
        assert not write_conflicts_read(w, ReadLoc.var("df"))


# ============================================================================
# Category 9: Quantifying the False Positive Impact
# ============================================================================

class TestFalsePositiveImpact:
    """
    Tests that quantify how many common patterns would get new false positives
    under the may-modify approach.
    """

    def test_typical_feature_engineering_pattern(self):
        """
        Common pattern: transform existing columns, then read shape.

            Cell A: df = pd.read_csv(...)         # creates df
            Cell B: df['price'] = df['price'] * 1.1  # modify existing
            Cell C: print(f"Shape: {df.shape}")   # structural read

        Current: C not stale after B reruns (correct — shape unchanged)
        May-mod: C stale after B reruns (false positive)
        """
        # This is a false positive because price already exists
        assert not write_conflicts_read(
            WriteLoc.col("df", "price"), ReadLoc.attr("df", "shape"))
        assert col_set_conflicts_read("df", "price", ReadLoc.attr("df", "shape"))

    def test_feature_engineering_multiple_transforms(self):
        """
        Cell B transforms 5 existing columns. Cell C reads df.shape.

        With may-modify, C is falsely stale because each column write
        produces a ColSet that conflicts with shape.

        In a typical data science notebook, this pattern is VERY common.
        """
        cols = ["price", "qty", "revenue", "cost", "profit"]

        # Current: none of these conflict with shape
        for col in cols:
            assert not write_conflicts_read(
                WriteLoc.col("df", col), ReadLoc.attr("df", "shape"))

        # May-modify: ALL of them conflict with shape
        for col in cols:
            assert col_set_conflicts_read("df", col, ReadLoc.attr("df", "shape"))

    def test_len_check_pattern(self):
        """
        Common pattern: cell reads len(df) to validate data.

            Cell B: df['score'] = compute_score(df)  # modify existing
            Cell C: assert len(df) == expected_rows   # structural read (len)

        Current: not stale (Col ▷ len = false)
        May-mod: stale (ColSet ▷ len = true, via ColAdd ▷ size)

        len is in COL_ATTRS? Actually len is in ROW_ATTRS. Let's check.
        """
        # len is a ROW_ATTR, not a COL_ATTR
        assert "len" not in COL_ATTRS
        # So ColAdd doesn't conflict with len
        assert not write_conflicts_read(
            WriteLoc.col_add("df", "score"), ReadLoc.attr("df", "len"))
        # And neither does ColSet
        assert not col_set_conflicts_read("df", "score", ReadLoc.attr("df", "len"))
        # Good — len is NOT a false positive! It's purely row-based.

    def test_iter_pattern(self):
        """
        Cell B modifies existing column. Cell C iterates: for col in df.

        iter is in COL_ATTRS (yields column names).

        Current: Col ▷ iter = false (just values changed, columns unchanged)
        May-mod: ColSet ▷ iter = true (may have added column)
        """
        assert "iter" in COL_ATTRS
        assert not write_conflicts_read(
            WriteLoc.col("df", "y"), ReadLoc.attr("df", "iter"))
        assert col_set_conflicts_read("df", "y", ReadLoc.attr("df", "iter"))
