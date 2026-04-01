"""
Tests for ▷ conflict relation with LocRef aliasing.

Covers every scenario from ALIAS_CONFLICT_ANALYSIS.md:
- Same-object aliases (X = df) → same loc_id
- User copies (df2 = df.copy()) → different loc_id
- All 7 write types × 4 read types with alias/copy/self combinations
- Var(x) only conflicts with Var(x) reads (no cross-domain bridge)

See ALIAS_CONFLICT_ANALYSIS.md for the full correctness analysis.
"""

import pytest

from flowbook.kernel.loc_ids import LocRef
from flowbook.kernel.locations import ReadLoc, WriteLoc, write_conflicts_read


# ============================================================================
# Fixtures: LocRef aliases and copies
# ============================================================================

# df and X are aliases (same object, loc_id=1)
LR_DF = LocRef(loc_id=1, var_name="df")
LR_X = LocRef(loc_id=1, var_name="X")

# df2 is a user copy (different object, loc_id=2)
LR_DF2 = LocRef(loc_id=2, var_name="df2")


# ============================================================================
# Var(x) ▷ Var(x')
# ============================================================================

class TestVarVarConflicts:
    """Var rebinding only invalidates reads of the SAME variable name."""

    def test_same_name(self):
        """Var("df") ▷ Var("df") → True"""
        assert write_conflicts_read(WriteLoc.var("df"), ReadLoc.var("df"))

    def test_alias_name_no_conflict(self):
        """Var("df") ▷ Var("X") → False — rebinding df doesn't affect X."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.var("X"))

    def test_reverse_alias_no_conflict(self):
        """Var("X") ▷ Var("df") → False — rebinding X doesn't affect df."""
        assert not write_conflicts_read(WriteLoc.var("X"), ReadLoc.var("df"))

    def test_copy_name_no_conflict(self):
        """Var("df") ▷ Var("df2") → False — different variables entirely."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.var("df2"))


# ============================================================================
# Var(x) ▷ Col(d', c') — no cross-domain conflict
# ============================================================================

class TestVarColConflicts:
    """Var(x) does NOT directly conflict with Col/Attr reads.

    Rebinding detection works because Var(x) is always present in read
    sets alongside Col reads (via tracking_to_readlocset). So Var(x)
    write ▷ Var(x) read catches the rebinding case.
    """

    def test_rebind_does_not_conflict_with_col_reads(self):
        """Var("df") ▷ Col(LR(1,"df"), "price") → False"""
        w = WriteLoc.var("df")
        r = ReadLoc.col(LR_DF, "price")
        assert not write_conflicts_read(w, r)

    def test_rebind_does_not_invalidate_alias_col_reads(self):
        """Var("df") ▷ Col(LR(1,"X"), "price") → False"""
        w = WriteLoc.var("df")
        r = ReadLoc.col(LR_X, "price")
        assert not write_conflicts_read(w, r)

    def test_rebind_alias_does_not_invalidate_original_col_reads(self):
        """Var("X") ▷ Col(LR(1,"df"), "price") → False"""
        w = WriteLoc.var("X")
        r = ReadLoc.col(LR_DF, "price")
        assert not write_conflicts_read(w, r)

    def test_rebind_alias_does_not_conflict_with_own_col_reads(self):
        """Var("X") ▷ Col(LR(1,"X"), "price") → False"""
        w = WriteLoc.var("X")
        r = ReadLoc.col(LR_X, "price")
        assert not write_conflicts_read(w, r)

    def test_rebind_does_not_invalidate_copy_col_reads(self):
        """Var("df") ▷ Col(LR(2,"df2"), "price") → False"""
        w = WriteLoc.var("df")
        r = ReadLoc.col(LR_DF2, "price")
        assert not write_conflicts_read(w, r)

    def test_rebind_no_col_conflict_any_column(self):
        """Var(x) does not conflict with any Col read."""
        w = WriteLoc.var("df")
        assert not write_conflicts_read(w, ReadLoc.col(LR_DF, "price"))
        assert not write_conflicts_read(w, ReadLoc.col(LR_DF, "qty"))
        assert not write_conflicts_read(w, ReadLoc.col(LR_DF, "anything"))


# ============================================================================
# Var(x) ▷ Attr(d', a') — no cross-domain conflict
# ============================================================================

class TestVarAttrConflicts:
    """Var(x) does NOT directly conflict with Attr reads."""

    def test_rebind_does_not_conflict_with_attr_reads(self):
        """Var("df") ▷ Attr(LR(1,"df"), "shape") → False"""
        w = WriteLoc.var("df")
        r = ReadLoc.attr(LR_DF, "shape")
        assert not write_conflicts_read(w, r)

    def test_rebind_does_not_invalidate_alias_attr_reads(self):
        """Var("df") ▷ Attr(LR(1,"X"), "shape") → False"""
        w = WriteLoc.var("df")
        r = ReadLoc.attr(LR_X, "shape")
        assert not write_conflicts_read(w, r)

    def test_rebind_alias_does_not_invalidate_original_attr_reads(self):
        """Var("X") ▷ Attr(LR(1,"df"), "shape") → False"""
        w = WriteLoc.var("X")
        r = ReadLoc.attr(LR_DF, "shape")
        assert not write_conflicts_read(w, r)

    def test_rebind_no_attr_conflict_any_attr(self):
        """Var(x) does not conflict with any Attr read."""
        w = WriteLoc.var("df")
        assert not write_conflicts_read(w, ReadLoc.attr(LR_DF, "shape"))
        assert not write_conflicts_read(w, ReadLoc.attr(LR_DF, "columns"))
        assert not write_conflicts_read(w, ReadLoc.attr(LR_DF, "index"))

    def test_var_does_not_conflict_with_file(self):
        """Var("df") ▷ File("data.csv") → False — always."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.file("data.csv"))


# ============================================================================
# Col(d, c) ▷ Col(d', c') — loc_id identity comparison
# ============================================================================

class TestColColConflicts:
    """Column write invalidates column read iff same object AND same column.

    Uses d ≡ d' (loc_id comparison) — handles aliases correctly.
    """

    def test_same_var_same_col(self):
        """Col(LR(1,"df"), "price") ▷ Col(LR(1,"df"), "price") → True"""
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, r)

    def test_alias_same_col(self):
        """Col(LR(1,"df"), "price") ▷ Col(LR(1,"X"), "price") → True

        KEY ALIAS TEST: write through df, read through X — same object
        (loc_id=1), same column. StableIdMap unifies them.
        """
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col(LR_X, "price")
        assert write_conflicts_read(w, r)

    def test_alias_reverse_same_col(self):
        """Col(LR(1,"X"), "price") ▷ Col(LR(1,"df"), "price") → True

        Write through alias, read through original.
        """
        w = WriteLoc.col(LR_X, "price")
        r = ReadLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, r)

    def test_same_var_different_col(self):
        """Col(LR(1,"df"), "price") ▷ Col(LR(1,"df"), "qty") → False

        Column independence: modifying price doesn't affect qty.
        """
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col(LR_DF, "qty")
        assert not write_conflicts_read(w, r)

    def test_alias_different_col(self):
        """Col(LR(1,"df"), "price") ▷ Col(LR(1,"X"), "qty") → False

        Same object via alias, but different column — no conflict.
        """
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col(LR_X, "qty")
        assert not write_conflicts_read(w, r)

    def test_copy_same_col_no_conflict(self):
        """Col(LR(1,"df"), "price") ▷ Col(LR(2,"df2"), "price") → False

        User copy: different object (loc_id=2), no conflict even with
        same column name.
        """
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col(LR_DF2, "price")
        assert not write_conflicts_read(w, r)

    def test_col_does_not_conflict_with_var(self):
        """Col(d,c) ▷ Var(x') → False (always, per ▷ matrix)."""
        w = WriteLoc.col(LR_DF, "price")
        assert not write_conflicts_read(w, ReadLoc.var("df"))
        assert not write_conflicts_read(w, ReadLoc.var("X"))

    def test_col_conflicts_with_structural_attr(self):
        """Col(d,c) ▷ Attr(d',a') → True for COL_ATTRS (shape, columns, dtypes).

        Col now conflicts with ALL COL_ATTRS, not just COL_VALUE_ATTRS.
        Modifying a column can change shape, columns, and dtypes.
        """
        w = WriteLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, ReadLoc.attr(LR_DF, "shape"))
        assert write_conflicts_read(w, ReadLoc.attr(LR_X, "columns"))
        assert write_conflicts_read(w, ReadLoc.attr(LR_DF, "dtypes"))

    def test_col_does_not_conflict_with_row_attr(self):
        """Col(d,c) ▷ Attr(d',a') → False for ROW_ATTRS (index).

        index is a row attribute, not a column attribute.
        """
        w = WriteLoc.col(LR_DF, "price")
        assert not write_conflicts_read(w, ReadLoc.attr(LR_DF, "index"))

    def test_col_conflicts_with_value_attrs(self):
        """Col(d,c) ▷ Attr(d',a') → True when a' ∈ COL_VALUE_ATTRS.

        Modifying column values changes df.values, df.T, df.describe().
        """
        w = WriteLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, ReadLoc.attr(LR_DF, "values"))
        assert write_conflicts_read(w, ReadLoc.attr(LR_DF, "T"))
        assert write_conflicts_read(w, ReadLoc.attr(LR_DF, "describe"))

    def test_col_conflicts_with_alias_value_attrs(self):
        """Col(d,c) ▷ Attr(d',a') → True for alias with value attr.

        Same object via loc_id, value-dependent attr.
        """
        w = WriteLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, ReadLoc.attr(LR_X, "values"))
        assert write_conflicts_read(w, ReadLoc.attr(LR_X, "T"))
        assert write_conflicts_read(w, ReadLoc.attr(LR_X, "describe"))

    def test_col_does_not_conflict_with_copy_value_attrs(self):
        """Col(d,c) ▷ Attr(d',a') → False for copy, even value attrs."""
        w = WriteLoc.col(LR_DF, "price")
        assert not write_conflicts_read(w, ReadLoc.attr(LR_DF2, "values"))
        assert not write_conflicts_read(w, ReadLoc.attr(LR_DF2, "T"))

    def test_col_does_not_conflict_with_file(self):
        """Col(d,c) ▷ File(p') → False."""
        w = WriteLoc.col(LR_DF, "price")
        assert not write_conflicts_read(w, ReadLoc.file("data.csv"))


# ============================================================================
# ColAdd(d, c) ▷ Attr(d', a') — structural impact of adding a column
# ============================================================================

class TestColAddConflicts:
    """Adding a column invalidates COL_ATTRS reads on the same object."""

    def test_coladd_invalidates_own_col_attrs(self):
        """ColAdd(LR(1,"df"), "new") ▷ Attr(LR(1,"df"), "columns") → True"""
        w = WriteLoc.col_add(LR_DF, "new")
        r = ReadLoc.attr(LR_DF, "columns")
        assert write_conflicts_read(w, r)

    def test_coladd_invalidates_alias_col_attrs(self):
        """ColAdd(LR(1,"df"), "new") ▷ Attr(LR(1,"X"), "columns") → True

        Alias: X.columns also changed because X IS df.
        """
        w = WriteLoc.col_add(LR_DF, "new")
        r = ReadLoc.attr(LR_X, "columns")
        assert write_conflicts_read(w, r)

    def test_coladd_does_not_invalidate_copy_col_attrs(self):
        """ColAdd(LR(1,"df"), "new") ▷ Attr(LR(2,"df2"), "columns") → False"""
        w = WriteLoc.col_add(LR_DF, "new")
        r = ReadLoc.attr(LR_DF2, "columns")
        assert not write_conflicts_read(w, r)

    def test_coladd_does_not_invalidate_non_col_attrs(self):
        """ColAdd doesn't affect row-structural attrs like index."""
        w = WriteLoc.col_add(LR_DF, "new")
        r = ReadLoc.attr(LR_DF, "index")
        assert not write_conflicts_read(w, r)

    def test_coladd_does_not_invalidate_col_reads(self):
        """ColAdd(d,c) ▷ Col(d',c') → False — existing columns unaffected."""
        w = WriteLoc.col_add(LR_DF, "new")
        assert not write_conflicts_read(w, ReadLoc.col(LR_DF, "price"))
        assert not write_conflicts_read(w, ReadLoc.col(LR_X, "price"))

    def test_coladd_does_not_invalidate_var(self):
        """ColAdd(d,c) ▷ Var(x') → False."""
        w = WriteLoc.col_add(LR_DF, "new")
        assert not write_conflicts_read(w, ReadLoc.var("df"))


# ============================================================================
# ColDel(d, c) ▷ Col(d', c') and Attr(d', a')
# ============================================================================

class TestColDelConflicts:
    """Deleting a column invalidates reads of that column AND COL_ATTRS."""

    def test_coldel_invalidates_own_col_read(self):
        """ColDel(LR(1,"df"), "price") ▷ Col(LR(1,"df"), "price") → True"""
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, r)

    def test_coldel_invalidates_alias_col_read(self):
        """ColDel(LR(1,"df"), "price") ▷ Col(LR(1,"X"), "price") → True

        Alias: column deleted from same object.
        """
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.col(LR_X, "price")
        assert write_conflicts_read(w, r)

    def test_coldel_does_not_invalidate_copy_col_read(self):
        """ColDel(LR(1,"df"), "price") ▷ Col(LR(2,"df2"), "price") → False"""
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.col(LR_DF2, "price")
        assert not write_conflicts_read(w, r)

    def test_coldel_does_not_invalidate_different_col(self):
        """ColDel(LR(1,"df"), "price") ▷ Col(LR(1,"df"), "qty") → False"""
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.col(LR_DF, "qty")
        assert not write_conflicts_read(w, r)

    def test_coldel_invalidates_own_col_attrs(self):
        """ColDel(LR(1,"df"), "price") ▷ Attr(LR(1,"df"), "columns") → True"""
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.attr(LR_DF, "columns")
        assert write_conflicts_read(w, r)

    def test_coldel_invalidates_alias_col_attrs(self):
        """ColDel(LR(1,"df"), "price") ▷ Attr(LR(1,"X"), "columns") → True"""
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.attr(LR_X, "columns")
        assert write_conflicts_read(w, r)

    def test_coldel_does_not_invalidate_copy_col_attrs(self):
        """ColDel(LR(1,"df"), "price") ▷ Attr(LR(2,"df2"), "columns") → False"""
        w = WriteLoc.col_del(LR_DF, "price")
        r = ReadLoc.attr(LR_DF2, "columns")
        assert not write_conflicts_read(w, r)


# ============================================================================
# Rows(d) ▷ Col(d', c') and Attr(d', a')
# ============================================================================

class TestRowsConflicts:
    """Row changes affect ALL columns and ROW_ATTRS on the same object."""

    def test_rows_invalidates_own_col_read(self):
        """Rows(LR(1,"df")) ▷ Col(LR(1,"df"), "price") → True"""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, r)

    def test_rows_invalidates_alias_col_read(self):
        """Rows(LR(1,"df")) ▷ Col(LR(1,"X"), "price") → True

        Alias: same DataFrame, rows changed → all columns affected.
        """
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.col(LR_X, "price")
        assert write_conflicts_read(w, r)

    def test_rows_invalidates_all_alias_cols(self):
        """Rows invalidates ALL column reads on aliases."""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        assert write_conflicts_read(w, ReadLoc.col(LR_X, "price"))
        assert write_conflicts_read(w, ReadLoc.col(LR_X, "qty"))
        assert write_conflicts_read(w, ReadLoc.col(LR_X, "anything"))

    def test_rows_does_not_invalidate_copy_col_read(self):
        """Rows(LR(1,"df")) ▷ Col(LR(2,"df2"), "price") → False"""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.col(LR_DF2, "price")
        assert not write_conflicts_read(w, r)

    def test_rows_invalidates_own_row_attrs(self):
        """Rows(LR(1,"df")) ▷ Attr(LR(1,"df"), "shape") → True"""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.attr(LR_DF, "shape")
        assert write_conflicts_read(w, r)

    def test_rows_invalidates_alias_row_attrs(self):
        """Rows(LR(1,"df")) ▷ Attr(LR(1,"X"), "shape") → True"""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.attr(LR_X, "shape")
        assert write_conflicts_read(w, r)

    def test_rows_does_not_invalidate_non_row_attrs(self):
        """Rows doesn't affect column-structural attrs like 'columns'."""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.attr(LR_DF, "columns")
        assert not write_conflicts_read(w, r)

    def test_rows_does_not_invalidate_copy_row_attrs(self):
        """Rows(LR(1,"df")) ▷ Attr(LR(2,"df2"), "shape") → False"""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        r = ReadLoc.attr(LR_DF2, "shape")
        assert not write_conflicts_read(w, r)


# ============================================================================
# Attr(d, a) ▷ Attr(d', a')
# ============================================================================

class TestAttrAttrConflicts:
    """Attr write invalidates attr read iff same object AND same attribute."""

    def test_same_var_same_attr(self):
        """Attr(LR(1,"df"), "index") ▷ Attr(LR(1,"df"), "index") → True"""
        w = WriteLoc.attr(LR_DF, "index")
        r = ReadLoc.attr(LR_DF, "index")
        assert write_conflicts_read(w, r)

    def test_alias_same_attr(self):
        """Attr(LR(1,"df"), "index") ▷ Attr(LR(1,"X"), "index") → True

        Alias: same attribute on same object.
        """
        w = WriteLoc.attr(LR_DF, "index")
        r = ReadLoc.attr(LR_X, "index")
        assert write_conflicts_read(w, r)

    def test_alias_different_attr(self):
        """Attr(LR(1,"df"), "index") ▷ Attr(LR(1,"X"), "shape") → False"""
        w = WriteLoc.attr(LR_DF, "index")
        r = ReadLoc.attr(LR_X, "shape")
        assert not write_conflicts_read(w, r)

    def test_copy_same_attr_no_conflict(self):
        """Attr(LR(1,"df"), "index") ▷ Attr(LR(2,"df2"), "index") → False"""
        w = WriteLoc.attr(LR_DF, "index")
        r = ReadLoc.attr(LR_DF2, "index")
        assert not write_conflicts_read(w, r)

    def test_attr_does_not_conflict_with_col(self):
        """Attr(d,a) ▷ Col(d',c') → False (always)."""
        w = WriteLoc.attr(LR_DF, "index")
        assert not write_conflicts_read(w, ReadLoc.col(LR_DF, "price"))
        assert not write_conflicts_read(w, ReadLoc.col(LR_X, "price"))

    def test_attr_does_not_conflict_with_var(self):
        """Attr(d,a) ▷ Var(x') → False (always)."""
        w = WriteLoc.attr(LR_DF, "index")
        assert not write_conflicts_read(w, ReadLoc.var("df"))


# ============================================================================
# File(p) ▷ File(p')
# ============================================================================

class TestFileConflicts:
    """File write invalidates file read of same path. No aliasing possible."""

    def test_same_path(self):
        assert write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.file("data.csv"))

    def test_different_path(self):
        assert not write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.file("other.csv"))

    def test_file_does_not_conflict_with_var(self):
        assert not write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.var("df"))

    def test_file_does_not_conflict_with_col(self):
        assert not write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.col(LR_DF, "price"))


# ============================================================================
# Mixed qualifier types (LocRef vs string) — backward compatibility
# ============================================================================

class TestMixedQualifiers:
    """When one side has LocRef and the other has string, fall back to var_name."""

    def test_locref_vs_string_same_name(self):
        """Col(LocRef(1,"df"), "price") ▷ Col("df", "price") → True

        Mixed types: extract var_name from LocRef, compare as strings.
        """
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col("df", "price")
        assert write_conflicts_read(w, r)

    def test_string_vs_locref_same_name(self):
        """Col("df", "price") ▷ Col(LocRef(1,"df"), "price") → True"""
        w = WriteLoc.col("df", "price")
        r = ReadLoc.col(LR_DF, "price")
        assert write_conflicts_read(w, r)

    def test_locref_vs_string_different_name(self):
        """Col(LocRef(1,"df"), "price") ▷ Col("X", "price") → False

        Different var_names, and string qualifier has no loc_id.
        """
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col("X", "price")
        assert not write_conflicts_read(w, r)

    def test_string_vs_string(self):
        """Col("df", "price") ▷ Col("df", "price") → True

        Pure string comparison (legacy behavior).
        """
        w = WriteLoc.col("df", "price")
        r = ReadLoc.col("df", "price")
        assert write_conflicts_read(w, r)

    def test_var_vs_string_qualifier(self):
        """Var("df") ▷ Col("df", "price") → False (Var only conflicts with Var)."""
        w = WriteLoc.var("df")
        r = ReadLoc.col("df", "price")
        assert not write_conflicts_read(w, r)

    def test_var_vs_locref_alias(self):
        """Var("df") ▷ Col(LocRef(1,"X"), "price") → False"""
        w = WriteLoc.var("df")
        r = ReadLoc.col(LR_X, "price")
        assert not write_conflicts_read(w, r)


# ============================================================================
# Three-way alias scenarios
# ============================================================================

class TestThreeWayAliases:
    """Three aliases of the same object: df, X, Y (all loc_id=1)."""

    LR_Y = LocRef(loc_id=1, var_name="Y")

    def test_col_write_through_first_read_through_third(self):
        """Col(df, "price") ▷ Col(Y, "price") → True — transitive via loc_id."""
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.col(self.LR_Y, "price")
        assert write_conflicts_read(w, r)

    def test_col_write_through_second_read_through_third(self):
        """Col(X, "price") ▷ Col(Y, "price") → True"""
        w = WriteLoc.col(LR_X, "price")
        r = ReadLoc.col(self.LR_Y, "price")
        assert write_conflicts_read(w, r)

    def test_var_rebind_does_not_conflict_with_any_col(self):
        """Var("df") does not conflict with any Col read."""
        w = WriteLoc.var("df")
        assert not write_conflicts_read(w, ReadLoc.col(LR_DF, "price"))
        assert not write_conflicts_read(w, ReadLoc.col(LR_X, "price"))
        assert not write_conflicts_read(w, ReadLoc.col(self.LR_Y, "price"))

    def test_rows_affects_all_aliases(self):
        """Rows(df) invalidates all column reads through any alias."""
        w = WriteLoc.rows(LR_DF.var_name, qualifier=LR_DF)
        assert write_conflicts_read(w, ReadLoc.col(LR_DF, "price"))
        assert write_conflicts_read(w, ReadLoc.col(LR_X, "qty"))
        assert write_conflicts_read(w, ReadLoc.col(self.LR_Y, "anything"))


# ============================================================================
# Multiple independent DataFrames
# ============================================================================

class TestIndependentDataFrames:
    """Two unrelated DataFrames with different loc_ids never conflict."""

    LR_A = LocRef(loc_id=10, var_name="sales")
    LR_B = LocRef(loc_id=20, var_name="inventory")

    def test_col_independent(self):
        """Col writes to one DataFrame don't affect another, even same column name."""
        w = WriteLoc.col(self.LR_A, "price")
        r = ReadLoc.col(self.LR_B, "price")
        assert not write_conflicts_read(w, r)

    def test_rows_independent(self):
        w = WriteLoc.rows(self.LR_A.var_name, qualifier=self.LR_A)
        r = ReadLoc.col(self.LR_B, "price")
        assert not write_conflicts_read(w, r)

    def test_coladd_independent(self):
        w = WriteLoc.col_add(self.LR_A, "new")
        r = ReadLoc.attr(self.LR_B, "columns")
        assert not write_conflicts_read(w, r)

    def test_attr_independent(self):
        w = WriteLoc.attr(self.LR_A, "index")
        r = ReadLoc.attr(self.LR_B, "index")
        assert not write_conflicts_read(w, r)
