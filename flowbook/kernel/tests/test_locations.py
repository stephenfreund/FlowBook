"""
Tests for ReadLoc, WriteLoc, and the ▷ conflict relation.

Tests cover:
1. All 20 cells of the ▷ matrix (5 write types × 4 read types)
2. The ▷▷ write-write conflict relation (write_conflicts_write, wlocs_conflict_wlocs)
3. Set-level operations (wlocs_conflict_rlocs, has_conflict)
4. Extraction helpers (var_names, column_map, file_list)
5. Conversion from TrackingData
6. display_name() for all types
7. Worked examples from the plan (independent column writes, etc.)
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
    readlocset_to_attr_map,
    readlocset_to_file_list,
    writelocset_to_file_list,
    readlocset_to_list,
    writelocset_to_list,
    tracking_to_readlocset,
    tracking_to_writelocset,
    COL_ATTRS,
    ROW_ATTRS,
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

    def test_attr(self):
        r = ReadLoc.attr("df", "shape")
        assert r.type == ReadLocType.ATTR
        assert r.name == "shape"
        assert r.qualifier == "df"
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
        assert ReadLoc.attr("df", "shape").display_name() == "df.shape"
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

    def test_rows(self):
        w = WriteLoc.rows("df")
        assert w.type == WriteLocType.ROWS
        assert w.name == "df"
        assert w.var_name() == "df"

    def test_attr(self):
        w = WriteLoc.attr("df", "index")
        assert w.type == WriteLocType.ATTR
        assert w.name == "index"
        assert w.qualifier == "df"
        assert w.var_name() == "df"

    def test_file(self):
        w = WriteLoc.file("out.csv")
        assert w.type == WriteLocType.FILE
        assert w.var_name() == "out.csv"

    def test_display_name(self):
        assert WriteLoc.var("x").display_name() == "x"
        assert WriteLoc.col("df", "price").display_name() == "df['price']"
        assert WriteLoc.rows("df").display_name() == "df (rows changed)"
        assert WriteLoc.attr("df", "index").display_name() == "df.index"
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

    def test_col_vs_rows(self):
        """Col(df, price) ▷▷ Rows(df) = True."""
        assert write_conflicts_write(WriteLoc.col("df", "price"), WriteLoc.rows("df"))

    def test_rows_vs_col(self):
        """Rows(df) ▷▷ Col(df, price) = True."""
        assert write_conflicts_write(WriteLoc.rows("df"), WriteLoc.col("df", "price"))

    def test_rows_same(self):
        assert write_conflicts_write(WriteLoc.rows("df"), WriteLoc.rows("df"))

    def test_rows_different_df(self):
        assert not write_conflicts_write(WriteLoc.rows("df"), WriteLoc.rows("other"))

    def test_attr_same(self):
        assert write_conflicts_write(WriteLoc.attr("df", "index"), WriteLoc.attr("df", "index"))

    def test_attr_different(self):
        assert not write_conflicts_write(WriteLoc.attr("df", "index"), WriteLoc.attr("df", "shape"))

    def test_file_same(self):
        assert write_conflicts_write(WriteLoc.file("out.csv"), WriteLoc.file("out.csv"))

    def test_file_different(self):
        assert not write_conflicts_write(WriteLoc.file("out.csv"), WriteLoc.file("other.csv"))

    def test_var_vs_col_no_conflict(self):
        """Var(df) ▷▷ Col(df, price) = False."""
        assert not write_conflicts_write(WriteLoc.var("df"), WriteLoc.col("df", "price"))



# =============================================================================
# ▷ Conflict Matrix — All 28 cells
# =============================================================================


class TestConflictMatrix_Var:
    """Var(x) writes: only conflict with Var(x) reads.

    Rebinding detection for Col/Attr readers is handled by always
    including Var(x) in read sets alongside Col/Attr reads.
    """

    def test_var_vs_var_same(self):
        assert write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("x"))

    def test_var_vs_var_different(self):
        assert not write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("y"))

    def test_var_vs_col_no_conflict(self):
        """Var(x) does NOT directly conflict with Col reads."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("df", "price"))

    def test_var_vs_col_different_df(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("other", "price"))

    def test_var_vs_attr_no_conflict(self):
        """Var(x) does NOT directly conflict with Attr reads."""
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.attr("df", "shape"))

    def test_var_vs_attr_different_df(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.attr("other", "shape"))

    def test_var_vs_file(self):
        assert not write_conflicts_read(WriteLoc.var("x"), ReadLoc.file("data.csv"))


class TestConflictMatrix_Col:
    """Col(d, c) writes: column values modified. Conflicts with all COL_ATTRS."""

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

    def test_col_vs_attr(self):
        """Col conflicts with ALL COL_ATTRS on the same DataFrame."""
        for a in COL_ATTRS:
            assert write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.attr("df", a)), (
                f"Col(df, price) should conflict with Attr(df, {a})"
            )

    def test_col_vs_attr_non_col_attr(self):
        """Col does NOT conflict with attrs outside COL_ATTRS."""
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.attr("df", "index"))
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.attr("df", "len"))

    def test_col_vs_attr_different_df(self):
        """Col does NOT conflict with attrs on a different DataFrame."""
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.attr("other", "shape"))

    def test_col_vs_file(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.file("data.csv"))


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

    def test_rows_vs_attr_row_structure(self):
        """Row change affects row-structure attributes."""
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.attr("df", "index"))
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.attr("df", "shape"))
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.attr("df", "len"))
        assert write_conflicts_read(WriteLoc.rows("df"), ReadLoc.attr("df", "empty"))

    def test_rows_vs_attr_col_only(self):
        """Row change does NOT affect column-only attributes (that aren't also row attrs)."""
        # columns is COL_ATTRS but NOT ROW_ATTRS
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.attr("df", "columns"))
        # dtypes is COL_ATTRS but NOT ROW_ATTRS
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.attr("df", "dtypes"))

    def test_rows_vs_file(self):
        assert not write_conflicts_read(WriteLoc.rows("df"), ReadLoc.file("f"))


class TestConflictMatrix_Attr:
    """Attr(d, a) writes: attribute changed."""

    def test_attr_vs_var(self):
        """Attr does NOT conflict with Var read (binding unchanged)."""
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.var("df"))

    def test_attr_vs_var_different(self):
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.var("other"))

    def test_attr_vs_col(self):
        """Attr change does NOT affect column value reads."""
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.col("df", "price"))

    def test_attr_vs_attr_same(self):
        assert write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.attr("df", "index"))

    def test_attr_vs_attr_different(self):
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.attr("df", "shape"))

    def test_attr_vs_attr_different_df(self):
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.attr("other", "index"))

    def test_attr_vs_file(self):
        assert not write_conflicts_read(WriteLoc.attr("df", "index"), ReadLoc.file("f"))


class TestConflictMatrix_File:
    """File(p) writes."""

    def test_file_vs_var(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.var("x"))

    def test_file_vs_col(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.col("df", "price"))

    def test_file_vs_attr(self):
        assert not write_conflicts_read(WriteLoc.file("out.csv"), ReadLoc.attr("df", "shape"))

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
# Worked Examples from Plan
# =============================================================================


class TestWorkedExamples:
    def test_independent_column_writes_no_write_write_overlap(self):
        """
        B writes df["price"], C writes df["qty"].
        Col(df, price) ▷▷ Col(df, qty) = False (independent columns).
        Write-write overlap is NOT detected for disjoint columns.
        """
        W_B = frozenset({WriteLoc.col("df", "price")})
        W_C = frozenset({WriteLoc.col("df", "qty")})

        # No write-write overlap for disjoint columns
        assert not wlocs_conflict_wlocs(W_B, W_C)

    def test_column_write_conflicts_with_structural_read(self):
        """
        Col(df, price) ▷ Attr(df, columns) = True.
        Column writes DO conflict with structural attribute reads directly.
        """
        W_B = frozenset({WriteLoc.col("df", "price")})
        R_C = frozenset({ReadLoc.attr("df", "columns")})
        assert has_conflict(W_B, R_C)

    def test_col_write_doesnt_conflict_with_var_read(self):
        """
        Col(df, price) ▷ Var(df) = false.
        Column write doesn't change the variable binding.
        """
        W_B = frozenset({WriteLoc.col("df", "price")})
        R_C = frozenset({ReadLoc.var("df")})
        assert not has_conflict(W_B, R_C)

    def test_column_modify_affects_structural_attrs(self):
        """
        Modifying column values DOES conflict with structural attribute reads.
        Col(df, price) ▷ Attr(df, shape) = true.
        Col(df, price) ▷ Attr(df, columns) = true.
        """
        W = frozenset({WriteLoc.col("df", "price")})
        R = frozenset({ReadLoc.attr("df", "shape"), ReadLoc.attr("df", "columns")})
        assert has_conflict(W, R)

    def test_column_modify_affects_value_attrs(self):
        """
        Modifying column values DOES conflict with value-dependent attribute reads.
        Col(df, price) ▷ Attr(df, values) = true.
        Col(df, price) ▷ Attr(df, T) = true.
        Col(df, price) ▷ Attr(df, describe) = true.
        """
        W = frozenset({WriteLoc.col("df", "price")})
        assert has_conflict(W, frozenset({ReadLoc.attr("df", "values")}))
        assert has_conflict(W, frozenset({ReadLoc.attr("df", "T")}))
        assert has_conflict(W, frozenset({ReadLoc.attr("df", "describe")}))

    def test_column_write_affects_structure(self):
        """
        Writing a column DOES conflict with column-structure attributes.
        Col(df, new) ▷ Attr(df, columns) = true.
        """
        W = frozenset({WriteLoc.col("df", "new")})
        R = frozenset({ReadLoc.attr("df", "columns")})
        assert has_conflict(W, R)

    def test_row_change_affects_all_columns(self):
        """
        Rows(df) ▷ Col(df, c) = true for any c.
        """
        W = frozenset({WriteLoc.rows("df")})
        R = frozenset({ReadLoc.col("df", "price"), ReadLoc.col("df", "qty")})
        conflicting = wlocs_conflict_rlocs(W, R)
        assert WriteLoc.rows("df") in conflicting

    def test_index_change_doesnt_affect_columns(self):
        """
        Attr(df, index) ▷ Col(df, price) = false.
        """
        W = frozenset({WriteLoc.attr("df", "index")})
        R = frozenset({ReadLoc.col("df", "price")})
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
            ReadLoc.attr("df", "shape"),
            ReadLoc.file("data.csv"),
        })
        names = readlocset_var_names(locs)
        assert names == {"x", "df", "data.csv"}

    def test_writelocset_var_names(self):
        locs = frozenset({
            WriteLoc.var("x"),
            WriteLoc.col("df", "new"),
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

    def test_readlocset_to_attr_map(self):
        locs = frozenset({
            ReadLoc.attr("df", "shape"),
            ReadLoc.attr("df", "columns"),
            ReadLoc.attr("other", "index"),
            ReadLoc.var("x"),
        })
        result = readlocset_to_attr_map(locs)
        assert result == {"df": ["columns", "shape"], "other": ["index"]}

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
        # Var(x) is always emitted for every variable in reads_before_writes,
        # alongside any Col/Attr detail.
        expected = frozenset({
            ReadLoc.var("df"),
            ReadLoc.var("config"),
            ReadLoc.col("df", "price"),
            ReadLoc.col("df", "qty"),
            ReadLoc.attr("df", "shape"),
            ReadLoc.file("data.csv"),
        })
        assert result == expected

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
# Attribute constant sanity checks
# =============================================================================


class TestAttrConstants:
    def test_shape_in_both(self):
        """shape and size are in both COL_ATTRS and ROW_ATTRS."""
        assert "shape" in COL_ATTRS
        assert "shape" in ROW_ATTRS
        assert "size" in COL_ATTRS
        assert "size" in ROW_ATTRS

    def test_columns_col_only(self):
        """columns is in COL_ATTRS but not ROW_ATTRS."""
        assert "columns" in COL_ATTRS
        assert "columns" not in ROW_ATTRS

    def test_index_row_only(self):
        """index is in ROW_ATTRS but not COL_ATTRS."""
        assert "index" in ROW_ATTRS
        assert "index" not in COL_ATTRS


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
        """Sorting col locs where one has LocRef (int) and another has string qualifier.

        Regression: LocRef.loc_id is int, so to_dict() produces qualifier=int.
        A col loc with a plain string qualifier produces qualifier=str.
        When both locs have type="col", sorted() compares qualifiers directly,
        raising TypeError: '<' not supported between instances of 'str' and 'int'.
        """
        locref = LocRef(loc_id=42, var_name="df")
        locs = frozenset({
            ReadLoc.col(locref, "price"),          # LocRef qualifier → int in dict
            ReadLoc.col("other_df", "qty"),         # string qualifier
        })
        result = readlocset_to_list(locs)
        assert len(result) == 2

    def test_writelocset_mixed_locref_and_string_qualifier(self):
        """Same regression test for writelocset_to_list."""
        locref = LocRef(loc_id=7, var_name="df")
        locs = frozenset({
            WriteLoc.col(locref, "amount"),         # LocRef qualifier → int in dict
            WriteLoc.col("other", "total"),          # string qualifier
        })
        result = writelocset_to_list(locs)
        assert len(result) == 2

    def test_readlocset_integer_column_names(self):
        """DataFrames with integer column names (e.g., df[0], df[1]).

        Regression test: column name becomes loc.name which can be int.
        """
        locref = LocRef(loc_id=1, var_name="df")
        locs = frozenset({
            ReadLoc.col(locref, 0),
            ReadLoc.col(locref, 1),
            ReadLoc.col(locref, "label"),
        })
        result = readlocset_to_list(locs)
        assert len(result) == 3
        # All names should be sortable (str coercion in sort key)

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
        # str(1) < str(2) < "" doesn't hold, but type comes first in sort key
        # col < var alphabetically, so col entries come first
        col_results = [d for d in result if d["type"] == "col"]
        var_results = [d for d in result if d["type"] == "var"]
        assert len(col_results) == 2
        assert len(var_results) == 1
