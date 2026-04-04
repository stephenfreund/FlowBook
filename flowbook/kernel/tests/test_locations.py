"""
Tests for ReadLoc, WriteLoc, and the ▷ conflict relation.

Tests cover:
1. All 25 cells of the ▷ matrix (5 write types × 5 read types)
2. The ▷▷ write-write conflict relation (write_conflicts_write, wlocs_conflict_wlocs)
3. Set-level operations (wlocs_conflict_rlocs, has_conflict)
4. Extraction helpers (var_names, column_map, file_list)
5. Conversion from TrackingData
6. display_name() for all types
7. Worked examples (independent column writes, etc.)
"""

import pytest

from flowbook.kernel.locations import (
    ReadLoc,
    ReadLocType,
    ReadLocSet,
    WriteLoc,
    WriteLocType,
    WriteLocSet,
    write_conflicts_read,
    write_conflicts_write,
    wlocs_conflict_rlocs,
    wlocs_conflict_wlocs,
    has_conflict,
    readlocset_var_names,
    writelocset_var_names,
    readlocset_to_column_map,
    writelocset_to_column_map,
    readlocset_to_file_list,
    writelocset_to_file_list,
    readlocset_to_list,
    writelocset_to_list,
    tracking_to_readlocset,
    tracking_to_writelocset,
    COLS_READ_ATTRS,
    ROWS_READ_ATTRS,
    BOTH_READ_ATTRS,
)
from flowbook.kernel.loc_ids import LocRef
from flowbook.kernel_support.models import TrackingData


# =============================================================================
# ReadLoc construction and properties
# =============================================================================


class TestReadLoc:
    def test_var(self):
        r = ReadLoc.var("x")
        assert r.type == ReadLocType.VAR
        assert r.name == "x"
        assert r.qualifier is None
        assert r.var_name() == "x"

    def test_col(self):
        r = ReadLoc.col("df", "price")
        assert r.type == ReadLocType.COLUMN
        assert r.name == "price"
        assert r.qualifier == "df"
        assert r.var_name() == "df"

    def test_cols(self):
        r = ReadLoc.cols("df")
        assert r.type == ReadLocType.COLS
        assert r.name == "df"
        assert r.qualifier == "df"
        assert r.var_name() == "df"

    def test_cols_with_qualifier(self):
        ref = LocRef(42, "df")
        r = ReadLoc.cols("df", qualifier=ref)
        assert r.type == ReadLocType.COLS
        assert r.name == "df"
        assert r.qualifier == ref
        assert r.var_name() == "df"

    def test_rows(self):
        r = ReadLoc.rows("df")
        assert r.type == ReadLocType.ROWS
        assert r.name == "df"
        assert r.qualifier == "df"
        assert r.var_name() == "df"

    def test_rows_with_qualifier(self):
        ref = LocRef(42, "df")
        r = ReadLoc.rows("df", qualifier=ref)
        assert r.type == ReadLocType.ROWS
        assert r.name == "df"
        assert r.qualifier == ref
        assert r.var_name() == "df"

    def test_file(self):
        r = ReadLoc.file("data.csv")
        assert r.type == ReadLocType.FILE
        assert r.name == "data.csv"
        assert r.var_name() == "data.csv"

    def test_frozen(self):
        r = ReadLoc.var("x")
        with pytest.raises(AttributeError):
            r.name = "y"

    def test_hashable(self):
        """ReadLocs can be put in sets."""
        s = {ReadLoc.var("x"), ReadLoc.col("df", "price"), ReadLoc.var("x")}
        assert len(s) == 2  # var("x") deduped

    def test_display_name(self):
        assert ReadLoc.var("x").display_name() == "x"
        assert ReadLoc.col("df", "price").display_name() == "df['price']"
        assert ReadLoc.cols("df").display_name() == "df (cols structure)"
        assert ReadLoc.rows("df").display_name() == "df (rows structure)"
        assert ReadLoc.file("data.csv").display_name() == "File(data.csv)"


# =============================================================================
# WriteLoc construction and properties
# =============================================================================


class TestWriteLoc:
    def test_var(self):
        w = WriteLoc.var("x")
        assert w.type == WriteLocType.VAR
        assert w.name == "x"
        assert w.var_name() == "x"

    def test_col(self):
        w = WriteLoc.col("df", "price")
        assert w.type == WriteLocType.COL
        assert w.name == "price"
        assert w.qualifier == "df"
        assert w.var_name() == "df"

    def test_cols(self):
        w = WriteLoc.cols("df")
        assert w.type == WriteLocType.COLS
        assert w.name == "df"
        assert w.var_name() == "df"

    def test_rows(self):
        w = WriteLoc.rows("df")
        assert w.type == WriteLocType.ROWS
        assert w.name == "df"
        assert w.var_name() == "df"

    def test_file(self):
        w = WriteLoc.file("out.csv")
        assert w.type == WriteLocType.FILE
        assert w.var_name() == "out.csv"

    def test_display_name(self):
        assert WriteLoc.var("x").display_name() == "x"
        assert WriteLoc.col("df", "price").display_name() == "df['price']"
        assert WriteLoc.cols("df").display_name() == "df (cols changed)"
        assert WriteLoc.rows("df").display_name() == "df (rows changed)"
        assert WriteLoc.file("out.csv").display_name() == "File(out.csv)"


# =============================================================================
# ▷▷ Write-Write Conflict Relation
# =============================================================================


class TestWriteConflictsWrite:
    """write_conflicts_write() (▷▷) — direct write-write conflict relation."""

    def test_var_same(self):
        assert write_conflicts_write(WriteLoc.var("x"), WriteLoc.var("x"))

    def test_var_different(self):
        assert not write_conflicts_write(WriteLoc.var("x"), WriteLoc.var("y"))

    def test_col_same(self):
        assert write_conflicts_write(WriteLoc.col("df", "price"), WriteLoc.col("df", "price"))

    def test_col_different_column(self):
        """Independent columns do not conflict."""
        assert not write_conflicts_write(WriteLoc.col("df", "price"), WriteLoc.col("df", "qty"))

    def test_col_different_df(self):
        assert not write_conflicts_write(WriteLoc.col("df", "price"), WriteLoc.col("other", "price"))

    def test_col_vs_cols(self):
        """Col(df, price) ▷▷ Cols(df) = True."""
        assert write_conflicts_write(WriteLoc.col("df", "price"), WriteLoc.cols("df"))

    def test_cols_vs_col(self):
        """Cols(df) ▷▷ Col(df, price) = True."""
        assert write_conflicts_write(WriteLoc.cols("df"), WriteLoc.col("df", "price"))

    def test_col_vs_rows(self):
        """Col(df, price) ▷▷ Rows(df) = True."""
        assert write_conflicts_write(WriteLoc.col("df", "price"), WriteLoc.rows("df"))

    def test_rows_vs_col(self):
        """Rows(df) ▷▷ Col(df, price) = True."""
        assert write_conflicts_write(WriteLoc.rows("df"), WriteLoc.col("df", "price"))

    def test_cols_same(self):
        assert write_conflicts_write(WriteLoc.cols("df"), WriteLoc.cols("df"))

    def test_cols_different_df(self):
        assert not write_conflicts_write(WriteLoc.cols("df"), WriteLoc.cols("other"))

    def test_rows_same(self):
        assert write_conflicts_write(WriteLoc.rows("df"), WriteLoc.rows("df"))

    def test_rows_different_df(self):
        assert not write_conflicts_write(WriteLoc.rows("df"), WriteLoc.rows("other"))

    def test_cols_vs_rows_no_conflict(self):
        """Cols(df) ▷▷ Rows(df) = False — independent structural domains."""
        assert not write_conflicts_write(WriteLoc.cols("df"), WriteLoc.rows("df"))

    def test_rows_vs_cols_no_conflict(self):
        """Rows(df) ▷▷ Cols(df) = False — symmetric."""
        assert not write_conflicts_write(WriteLoc.rows("df"), WriteLoc.cols("df"))

    def test_file_same(self):
        assert write_conflicts_write(WriteLoc.file("out.csv"), WriteLoc.file("out.csv"))

    def test_file_different(self):
        assert not write_conflicts_write(WriteLoc.file("out.csv"), WriteLoc.file("other.csv"))

    def test_var_vs_col_no_conflict(self):
        """Var(df) ▷▷ Col(df, price) = False."""
        assert not write_conflicts_write(WriteLoc.var("df"), WriteLoc.col("df", "price"))


# =============================================================================
# ▷ Conflict Matrix — All 25 cells (5×5)
# =============================================================================


class TestConflictMatrix_Var:
    """Var(x) writes: only conflict with Var(x) reads."""

    def test_var_vs_var_same(self):
        assert write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("x"))

    def test_var_vs_var_different(self):
        assert not write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("y"))

    def test_var_vs_col_no_conflict(self):
        """Var(x) does NOT directly conflict with Col reads."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("df", "price"))

    def test_var_vs_col_different_df(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("other", "price"))

    def test_var_vs_cols_no_conflict(self):
        """Var(x) does NOT directly conflict with Cols reads."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.cols("df"))

    def test_var_vs_rows_no_conflict(self):
        """Var(x) does NOT directly conflict with Rows reads."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.rows("df"))

    def test_var_vs_file(self):
        assert not write_conflicts_read(WriteLoc.var("x"), ReadLoc.file("data.csv"))


class TestConflictMatrix_Col:
    """Col(d, c) writes: column values modified."""

    def test_col_vs_var_same_df(self):
        """Column write does NOT conflict with Var read (binding unchanged)."""
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.var("df"))

    def test_col_vs_var_different_df(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.var("other"))

    def test_col_vs_col_same(self):
        assert write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.col("df", "price"))

    def test_col_vs_col_different_column(self):
        """Key: modifying price does NOT affect quantity reads."""
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.col("df", "qty"))

    def test_col_vs_col_different_df(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.col("other", "price"))

    def test_col_vs_cols(self):
        """Col(df, price) ▷ Cols(df) = d≡d'."""
        assert write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.cols("df"))

    def test_col_vs_cols_different_df(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.cols("other"))

    def test_col_vs_rows_no_conflict(self):
        """Col(df, price) ▷ Rows(df) = False."""
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.rows("df"))

    def test_col_vs_file(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.file("data.csv"))


class TestConflictMatrix_Cols:
    """Cols(d) writes: column structure changed."""

    def test_cols_vs_var(self):
        assert not write_conflicts_read(WriteLoc.cols("df"), ReadLoc.var("df"))

    def test_cols_vs_col(self):
        """Cols(df) ▷ Col(df, price) = d≡d'."""
        assert write_conflicts_read(WriteLoc.cols("df"), ReadLoc.col("df", "price"))

    def test_cols_vs_col_different_df(self):
        assert not write_conflicts_read(WriteLoc.cols("df"), ReadLoc.col("other", "price"))

    def test_cols_vs_cols(self):
        """Cols(df) ▷ Cols(df) = d≡d'."""
        assert write_conflicts_read(WriteLoc.cols("df"), ReadLoc.cols("df"))

    def test_cols_vs_cols_different_df(self):
        assert not write_conflicts_read(WriteLoc.cols("df"), ReadLoc.cols("other"))

    def test_cols_vs_rows_no_conflict(self):
        """Cols(df) ▷ Rows(df) = False — cross-domain independence."""
        assert not write_conflicts_read(WriteLoc.cols("df"), ReadLoc.rows("df"))

    def test_cols_vs_file(self):
        assert not write_conflicts_read(WriteLoc.cols("df"), ReadLoc.file("f"))


class TestConflictMatrix_Rows:
    """Rows(d) writes: rows added or removed."""

    def test_rows_vs_var(self):
        """Rows write does NOT conflict with Var read (binding unchanged)."""
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.var("df"))

    def test_rows_vs_var_different(self):
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.var("other"))

    def test_rows_vs_col(self):
        """Row change affects ALL column data."""
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.col("df", "price"))
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.col("df", "qty"))

    def test_rows_vs_col_different_df(self):
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.col("other", "price"))

    def test_rows_vs_cols_no_conflict(self):
        """Rows(df) ▷ Cols(df) = False — cross-domain independence."""
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.cols("df"))

    def test_rows_vs_rows(self):
        """Rows(df) ▷ Rows(df) = d≡d'."""
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.rows("df"))

    def test_rows_vs_rows_different_df(self):
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.rows("other"))

    def test_rows_vs_file(self):
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.file("f"))


class TestConflictMatrix_File:
    """File(p) writes."""

    def test_file_vs_var(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.var("x"))

    def test_file_vs_col(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.col("df", "price"))

    def test_file_vs_cols(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.cols("df"))

    def test_file_vs_rows(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.rows("df"))

    def test_file_vs_file_same(self):
        assert write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.file("data.csv"))

    def test_file_vs_file_different(self):
        assert not write_conflicts_read(WriteLoc.file("data.csv"), ReadLoc.file("other.csv"))


# =============================================================================
# Set-Level Operations
# =============================================================================


class TestSetOperations:
    def test_wlocs_conflict_rlocs_basic(self):
        writes = frozenset({WriteLoc.var("x"), WriteLoc.var("y")})
        reads = frozenset({ReadLoc.var("x")})
        result = wlocs_conflict_rlocs(writes, reads)
        assert result == frozenset({WriteLoc.var("x")})

    def test_wlocs_conflict_rlocs_empty_writes(self):
        assert wlocs_conflict_rlocs(frozenset(), frozenset({ReadLoc.var("x")})) == frozenset()

    def test_wlocs_conflict_rlocs_empty_reads(self):
        assert wlocs_conflict_rlocs(frozenset({WriteLoc.var("x")}), frozenset()) == frozenset()

    def test_wlocs_conflict_rlocs_column_precision(self):
        """Column-level: only conflicting writes returned."""
        writes = frozenset({WriteLoc.col("df", "price"), WriteLoc.col("df", "qty")})
        reads = frozenset({ReadLoc.col("df", "price")})
        result = wlocs_conflict_rlocs(writes, reads)
        assert result == frozenset({WriteLoc.col("df", "price")})

    def test_has_conflict_true(self):
        writes = frozenset({WriteLoc.var("x")})
        reads = frozenset({ReadLoc.var("x")})
        assert has_conflict(writes, reads)

    def test_has_conflict_false(self):
        writes = frozenset({WriteLoc.col("df", "price")})
        reads = frozenset({ReadLoc.col("df", "qty")})
        assert not has_conflict(writes, reads)

    def test_wlocs_conflict_wlocs_basic(self):
        """wlocs_conflict_wlocs returns writes from W1 that overlap with W2."""
        W1 = frozenset({WriteLoc.var("x"), WriteLoc.var("y")})
        W2 = frozenset({WriteLoc.var("x")})
        result = wlocs_conflict_wlocs(W1, W2)
        assert result == frozenset({WriteLoc.var("x")})

    def test_wlocs_conflict_wlocs_empty(self):
        """Empty sets produce no conflicts."""
        assert wlocs_conflict_wlocs(frozenset(), frozenset({WriteLoc.var("x")})) == frozenset()
        assert wlocs_conflict_wlocs(frozenset({WriteLoc.var("x")}), frozenset()) == frozenset()

    def test_wlocs_conflict_wlocs_col_precision(self):
        """Independent columns do not conflict at write-write level."""
        W1 = frozenset({WriteLoc.col("df", "price")})
        W2 = frozenset({WriteLoc.col("df", "qty")})
        assert not wlocs_conflict_wlocs(W1, W2)


# =============================================================================
# Worked Examples
# =============================================================================


class TestWorkedExamples:
    def test_independent_column_writes_no_write_write_overlap(self):
        """
        B writes df["price"], C writes df["qty"].
        Col(df, price) ▷▷ Col(df, qty) = False (independent columns).
        """
        W_B = frozenset({WriteLoc.col("df", "price")})
        W_C = frozenset({WriteLoc.col("df", "qty")})
        assert not wlocs_conflict_wlocs(W_B, W_C)

    def test_column_write_conflicts_with_cols_read(self):
        """
        Col(df, price) ▷ Cols(df) = True.
        Column writes DO conflict with column-structure reads.
        """
        W_B = frozenset({WriteLoc.col("df", "price")})
        R_C = frozenset({ReadLoc.cols("df")})
        assert has_conflict(W_B, R_C)

    def test_col_write_doesnt_conflict_with_var_read(self):
        """
        Col(df, price) ▷ Var(df) = false.
        Column write doesn't change the variable binding.
        """
        W_B = frozenset({WriteLoc.col("df", "price")})
        R_C = frozenset({ReadLoc.var("df")})
        assert not has_conflict(W_B, R_C)

    def test_column_modify_affects_cols(self):
        """Col(df, price) ▷ Cols(df) = true."""
        W = frozenset({WriteLoc.col("df", "price")})
        R = frozenset({ReadLoc.cols("df")})
        assert has_conflict(W, R)

    def test_column_write_does_not_affect_rows(self):
        """Col(df, price) ▷ Rows(df) = false."""
        W = frozenset({WriteLoc.col("df", "price")})
        R = frozenset({ReadLoc.rows("df")})
        assert not has_conflict(W, R)

    def test_row_change_affects_all_columns(self):
        """
        Rows(df) ▷ Col(df, c) = true for any c.
        """
        W = frozenset({WriteLoc.rows("df")})
        R = frozenset({ReadLoc.col("df", "price"), ReadLoc.col("df", "qty")})
        conflicting = wlocs_conflict_rlocs(W, R)
        assert WriteLoc.rows("df") in conflicting

    def test_rows_change_does_not_affect_cols(self):
        """Rows(df) ▷ Cols(df) = false — cross-domain independence."""
        W = frozenset({WriteLoc.rows("df")})
        R = frozenset({ReadLoc.cols("df")})
        assert not has_conflict(W, R)

    def test_cols_change_does_not_affect_rows(self):
        """Cols(df) ▷ Rows(df) = false — cross-domain independence."""
        W = frozenset({WriteLoc.cols("df")})
        R = frozenset({ReadLoc.rows("df")})
        assert not has_conflict(W, R)

    def test_disjoint_column_modify_no_conflict(self):
        """
        Col(df, price) ▷ Col(df, qty) = false.
        Disjoint column modifications don't conflict.
        """
        W = frozenset({WriteLoc.col("df", "price")})
        R = frozenset({ReadLoc.col("df", "qty")})
        assert not has_conflict(W, R)


# =============================================================================
# Extraction Helpers
# =============================================================================


class TestExtractionHelpers:
    def test_readlocset_var_names(self):
        locs = frozenset({
            ReadLoc.var("x"),
            ReadLoc.col("df", "price"),
            ReadLoc.cols("df"),
            ReadLoc.rows("df2"),
            ReadLoc.file("data.csv"),
        })
        names = readlocset_var_names(locs)
        assert names == {"x", "df", "df2", "data.csv"}

    def test_writelocset_var_names(self):
        locs = frozenset({
            WriteLoc.var("x"),
            WriteLoc.col("df", "new"),
            WriteLoc.cols("df"),
            WriteLoc.rows("df2"),
            WriteLoc.file("out.csv"),
        })
        names = writelocset_var_names(locs)
        assert names == {"x", "df", "df2", "out.csv"}

    def test_readlocset_to_column_map(self):
        locs = frozenset({
            ReadLoc.col("df", "price"),
            ReadLoc.col("df", "qty"),
            ReadLoc.col("other", "name"),
            ReadLoc.var("x"),
        })
        result = readlocset_to_column_map(locs)
        assert result == {"df": ["price", "qty"], "other": ["name"]}

    def test_writelocset_to_column_map(self):
        locs = frozenset({
            WriteLoc.col("df", "price"),
            WriteLoc.col("df", "new"),
            WriteLoc.col("df", "old"),
            WriteLoc.var("x"),
        })
        result = writelocset_to_column_map(locs)
        assert result == {"df": ["new", "old", "price"]}

    def test_readlocset_to_file_list(self):
        locs = frozenset({
            ReadLoc.file("b.csv"),
            ReadLoc.file("a.csv"),
            ReadLoc.var("x"),
        })
        assert readlocset_to_file_list(locs) == ["a.csv", "b.csv"]

    def test_writelocset_to_file_list(self):
        locs = frozenset({
            WriteLoc.file("out.csv"),
            WriteLoc.var("x"),
        })
        assert writelocset_to_file_list(locs) == ["out.csv"]


# =============================================================================
# TrackingData Conversion
# =============================================================================


class TestTrackingConversion:
    def test_tracking_to_readlocset(self):
        td = TrackingData(
            reads_before_writes={"df", "config"},
            writes=set(),
            column_reads_before_writes={"df": {"price", "qty"}},
            column_writes={},
            structural_reads={"df": {"shape"}},
            file_reads_before_writes={"data.csv"},
            file_writes=set(),
        )
        result = tracking_to_readlocset(td)
        # shape is in BOTH_READ_ATTRS → emits both Cols(df) and Rows(df)
        expected = frozenset({
            ReadLoc.var("df"),
            ReadLoc.var("config"),
            ReadLoc.col("df", "price"),
            ReadLoc.col("df", "qty"),
            ReadLoc.cols("df"),
            ReadLoc.rows("df"),
            ReadLoc.file("data.csv"),
        })
        assert result == expected

    def test_tracking_to_readlocset_cols_only_attr(self):
        """Attribute in COLS_READ_ATTRS only produces Cols(d)."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            structural_reads={"df": {"columns"}},
        )
        result = tracking_to_readlocset(td)
        assert ReadLoc.cols("df") in result
        assert ReadLoc.rows("df") not in result

    def test_tracking_to_readlocset_rows_only_attr(self):
        """Attribute in ROWS_READ_ATTRS only produces Rows(d)."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            structural_reads={"df": {"index"}},
        )
        result = tracking_to_readlocset(td)
        assert ReadLoc.rows("df") in result
        assert ReadLoc.cols("df") not in result

    def test_tracking_to_readlocset_both_attr(self):
        """Attribute in BOTH_READ_ATTRS produces both Cols(d) and Rows(d)."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            structural_reads={"df": {"shape"}},
        )
        result = tracking_to_readlocset(td)
        assert ReadLoc.cols("df") in result
        assert ReadLoc.rows("df") in result

    def test_tracking_to_readlocset_mixed_attrs(self):
        """Multiple attrs from different domains."""
        td = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            structural_reads={"df": {"columns", "index"}},
        )
        result = tracking_to_readlocset(td)
        assert ReadLoc.cols("df") in result
        assert ReadLoc.rows("df") in result

    def test_tracking_to_writelocset(self):
        td = TrackingData(
            reads_before_writes=set(),
            writes={"df", "result"},
            column_reads_before_writes={},
            column_writes={"df": {"price"}},
            file_reads_before_writes=set(),
            file_writes={"out.csv"},
        )
        result = tracking_to_writelocset(td)
        expected = frozenset({
            WriteLoc.var("df"),
            WriteLoc.var("result"),
            WriteLoc.col("df", "price"),
            WriteLoc.file("out.csv"),
        })
        assert result == expected

    def test_tracking_to_writelocset_index_mutations(self):
        """index_mutations produce Rows(d) write."""
        td = TrackingData(
            reads_before_writes=set(),
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            index_mutations={"df"},
        )
        result = tracking_to_writelocset(td)
        assert WriteLoc.rows("df") in result

    def test_tracking_to_writelocset_dtype_changes(self):
        """dtype_changes produce Cols(d) write."""
        td = TrackingData(
            reads_before_writes=set(),
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            dtype_changes={"df": {"price"}},
        )
        result = tracking_to_writelocset(td)
        assert WriteLoc.cols("df") in result

    def test_tracking_to_readlocset_empty(self):
        td = TrackingData(
            reads_before_writes=set(),
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
        )
        assert tracking_to_readlocset(td) == frozenset()

    def test_tracking_to_writelocset_empty(self):
        td = TrackingData(
            reads_before_writes=set(),
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
        )
        assert tracking_to_writelocset(td) == frozenset()


# =============================================================================
# Attribute classification sanity checks
# =============================================================================


class TestAttrClassification:
    def test_shape_in_both(self):
        """shape is in BOTH_READ_ATTRS."""
        assert "shape" in BOTH_READ_ATTRS

    def test_columns_cols_only(self):
        """columns is in COLS_READ_ATTRS."""
        assert "columns" in COLS_READ_ATTRS

    def test_index_rows_only(self):
        """index is in ROWS_READ_ATTRS."""
        assert "index" in ROWS_READ_ATTRS

    def test_no_overlap_cols_rows(self):
        """COLS_READ_ATTRS and ROWS_READ_ATTRS do not overlap."""
        assert not COLS_READ_ATTRS & ROWS_READ_ATTRS

    def test_no_overlap_with_both(self):
        """BOTH_READ_ATTRS is disjoint from COLS and ROWS."""
        assert not BOTH_READ_ATTRS & COLS_READ_ATTRS
        assert not BOTH_READ_ATTRS & ROWS_READ_ATTRS


# =============================================================================
# Serialization: readlocset_to_list / writelocset_to_list
# =============================================================================


class TestLocsetToList:
    """Tests for serialization functions that sort locs into JSON-friendly dicts."""

    def test_readlocset_to_list_basic(self):
        locs = frozenset({ReadLoc.var("x"), ReadLoc.var("a")})
        result = readlocset_to_list(locs)
        assert len(result) == 2
        assert result[0]["name"] == "a"  # sorted by name
        assert result[1]["name"] == "x"

    def test_writelocset_to_list_basic(self):
        locs = frozenset({WriteLoc.var("y"), WriteLoc.var("b")})
        result = writelocset_to_list(locs)
        assert len(result) == 2
        assert result[0]["name"] == "b"
        assert result[1]["name"] == "y"

    def test_readlocset_mixed_locref_and_string_qualifier(self):
        """Sorting col locs where one has LocRef (int) and another has string qualifier."""
        locref = LocRef(loc_id=42, var_name="df")
        locs = frozenset({
            ReadLoc.col(locref, "price"),
            ReadLoc.col("other_df", "qty"),
        })
        result = readlocset_to_list(locs)
        assert len(result) == 2

    def test_writelocset_mixed_locref_and_string_qualifier(self):
        """Same regression test for writelocset_to_list."""
        locref = LocRef(loc_id=7, var_name="df")
        locs = frozenset({
            WriteLoc.col(locref, "amount"),
            WriteLoc.col("other", "total"),
        })
        result = writelocset_to_list(locs)
        assert len(result) == 2

    def test_readlocset_integer_column_names(self):
        """DataFrames with integer column names (e.g., df[0], df[1])."""
        locref = LocRef(loc_id=1, var_name="df")
        locs = frozenset({
            ReadLoc.col(locref, 0),
            ReadLoc.col(locref, 1),
            ReadLoc.col(locref, "label"),
        })
        result = readlocset_to_list(locs)
        assert len(result) == 3

    def test_writelocset_integer_column_names(self):
        """Same regression test for writelocset_to_list with int column names."""
        locref = LocRef(loc_id=1, var_name="df")
        locs = frozenset({
            WriteLoc.col(locref, 0),
            WriteLoc.col(locref, "price"),
        })
        result = writelocset_to_list(locs)
        assert len(result) == 2

    def test_readlocset_mixed_qualifier_types_sorted(self):
        """Multiple LocRef qualifiers with different loc_ids sort correctly."""
        ref1 = LocRef(loc_id=1, var_name="df1")
        ref2 = LocRef(loc_id=2, var_name="df2")
        locs = frozenset({
            ReadLoc.col(ref2, "b"),
            ReadLoc.col(ref1, "a"),
            ReadLoc.var("x"),
        })
        result = readlocset_to_list(locs)
        assert len(result) == 3
        col_results = [d for d in result if d["type"] == "col"]
        var_results = [d for d in result if d["type"] == "var"]
        assert len(col_results) == 2
        assert len(var_results) == 1
