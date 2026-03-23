"""
Tests for ReadLoc, WriteLoc, and the ▷ conflict relation.

Tests cover:
1. All 28 cells of the ▷ matrix (7 write types × 4 read types)
2. The output() function for all 7 write types
3. Set-level operations (wlocs_conflict_rlocs, has_conflict, output_set)
4. Extraction helpers (var_names, column_map, file_list)
5. Conversion from TrackingData
6. display_name() for all types
7. Worked examples from the plan (independent column additions, etc.)
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
    wlocs_conflict_rlocs,
    has_conflict,
    output_set,
    readlocset_var_names,
    writelocset_var_names,
    readlocset_to_column_map,
    writelocset_to_column_map,
    readlocset_to_attr_map,
    readlocset_to_file_list,
    writelocset_to_file_list,
    tracking_to_readlocset,
    tracking_to_writelocset,
    COL_ATTRS,
    ROW_ATTRS,
)
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

    def test_col_add(self):
        w = WriteLoc.col_add("df", "new")
        assert w.type == WriteLocType.COL_ADD
        assert w.name == "new"
        assert w.qualifier == "df"
        assert w.var_name() == "df"

    def test_col_del(self):
        w = WriteLoc.col_del("df", "old")
        assert w.type == WriteLocType.COL_DEL
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
        assert WriteLoc.col_add("df", "new").display_name() == "df['new'] (added)"
        assert WriteLoc.col_del("df", "old").display_name() == "df['old'] (removed)"
        assert WriteLoc.rows("df").display_name() == "df (rows changed)"
        assert WriteLoc.attr("df", "index").display_name() == "df.index"
        assert WriteLoc.file("out.csv").display_name() == "File(out.csv)"


# =============================================================================
# output() function
# =============================================================================


class TestOutput:
    """output() now returns FrozenSet[ReadLoc] — the reads that observe the write."""

    def test_var(self):
        assert WriteLoc.var("x").output() == frozenset({ReadLoc.var("x")})

    def test_col(self):
        """Col produces just Col(d,c) — no attr inflation for column independence."""
        result = WriteLoc.col("df", "price").output()
        assert result == frozenset({ReadLoc.col("df", "price")})

    def test_col_add(self):
        """ColAdd produces Attr reads for all COL_ATTRS."""
        result = WriteLoc.col_add("df", "new").output()
        # Should contain Attr(df, a) for each a in COL_ATTRS
        assert all(r.type == ReadLocType.ATTR for r in result)
        assert all(r.qualifier == "df" for r in result)
        from flowbook.kernel.locations import COL_ATTRS
        assert {r.name for r in result} == COL_ATTRS

    def test_col_del(self):
        """ColDel produces Col(d,c) plus Attr reads for all COL_ATTRS."""
        result = WriteLoc.col_del("df", "old").output()
        assert ReadLoc.col("df", "old") in result
        from flowbook.kernel.locations import COL_ATTRS
        for a in COL_ATTRS:
            assert ReadLoc.attr("df", a) in result

    def test_rows(self):
        """Rows produces Attr reads for all ROW_ATTRS."""
        result = WriteLoc.rows("df").output()
        assert all(r.type == ReadLocType.ATTR for r in result)
        assert all(r.qualifier == "df" for r in result)
        from flowbook.kernel.locations import ROW_ATTRS
        assert {r.name for r in result} == ROW_ATTRS

    def test_attr(self):
        assert WriteLoc.attr("df", "index").output() == frozenset({ReadLoc.attr("df", "index")})

    def test_file(self):
        assert WriteLoc.file("out.csv").output() == frozenset({ReadLoc.file("out.csv")})

    def test_col_add_same_df_same_output(self):
        """ColAdd for same df produces same outputs (both affect COL_ATTRS)."""
        o1 = WriteLoc.col_add("df", "price").output()
        o2 = WriteLoc.col_add("df", "qty").output()
        assert o1 == o2  # Both produce {Attr(df, a) | a ∈ COL_ATTRS}


# =============================================================================
# ▷ Conflict Matrix — All 28 cells
# =============================================================================


class TestConflictMatrix_Var:
    """Var(x) writes: invalidate any read involving x."""

    def test_var_vs_var_same(self):
        assert write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("x"))

    def test_var_vs_var_different(self):
        assert not write_conflicts_read(WriteLoc.var("x"), ReadLoc.var("y"))

    def test_var_vs_col_same_df(self):
        assert write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("df", "price"))

    def test_var_vs_col_different_df(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.col("other", "price"))

    def test_var_vs_attr_same_df(self):
        assert write_conflicts_read(WriteLoc.var("df"), ReadLoc.attr("df", "shape"))

    def test_var_vs_attr_different_df(self):
        assert not write_conflicts_read(WriteLoc.var("df"), ReadLoc.attr("other", "shape"))

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

    def test_col_vs_attr(self):
        """Key: modifying column values does NOT change structure."""
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.attr("df", "shape"))
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.attr("df", "columns"))

    def test_col_vs_file(self):
        assert not write_conflicts_read(WriteLoc.col("df", "price"), ReadLoc.file("data.csv"))


class TestConflictMatrix_ColAdd:
    """ColAdd(d, c) writes: new column added."""

    def test_col_add_vs_var_same_df(self):
        """ColAdd does NOT conflict with Var read (binding unchanged)."""
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.var("df"))

    def test_col_add_vs_var_different_df(self):
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.var("other"))

    def test_col_add_vs_col_existing(self):
        """Key: adding new column does NOT affect existing column reads."""
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.col("df", "price"))

    def test_col_add_vs_col_same_name(self):
        """Adding column 'new' doesn't conflict with reading 'new' (column didn't exist before)."""
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.col("df", "new"))

    def test_col_add_vs_attr_col_structure(self):
        """Adding column changes column-structure attributes."""
        assert write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("df", "columns"))
        assert write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("df", "dtypes"))
        assert write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("df", "shape"))
        assert write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("df", "size"))

    def test_col_add_vs_attr_row_only(self):
        """Adding column does NOT affect row-only attributes (that aren't also col attrs)."""
        # index is ROW_ATTRS but NOT COL_ATTRS
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("df", "index"))
        # len is ROW_ATTRS but NOT COL_ATTRS
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("df", "len"))

    def test_col_add_vs_attr_different_df(self):
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.attr("other", "columns"))

    def test_col_add_vs_file(self):
        assert not write_conflicts_read(WriteLoc.col_add("df", "new"), ReadLoc.file("data.csv"))


class TestConflictMatrix_ColDel:
    """ColDel(d, c) writes: column removed."""

    def test_col_del_vs_var_same_df(self):
        """ColDel does NOT conflict with Var read (binding unchanged)."""
        assert not write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.var("df"))

    def test_col_del_vs_col_same(self):
        """Removing column invalidates reads of that column."""
        assert write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.col("df", "old"))

    def test_col_del_vs_col_different(self):
        """Removing column does NOT affect reads of other columns."""
        assert not write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.col("df", "price"))

    def test_col_del_vs_attr_col_structure(self):
        """Removing column changes column-structure attributes."""
        assert write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.attr("df", "columns"))
        assert write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.attr("df", "shape"))

    def test_col_del_vs_attr_row_only(self):
        assert not write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.attr("df", "index"))

    def test_col_del_vs_file(self):
        assert not write_conflicts_read(WriteLoc.col_del("df", "old"), ReadLoc.file("f"))


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

    def test_output_set(self):
        writes = frozenset({
            WriteLoc.var("x"),
            WriteLoc.col("df", "price"),
            WriteLoc.attr("df", "index"),
        })
        result = output_set(writes)
        # Col output is just {Col(df, price)} — no CVA inflation
        expected = frozenset({
            ReadLoc.var("x"),
            ReadLoc.col("df", "price"),
            ReadLoc.attr("df", "index"),
        })
        assert result == expected

    def test_output_set_structural_writes(self):
        """Structural writes expand to multiple output reads."""
        writes = frozenset({WriteLoc.rows("df")})
        result = output_set(writes)
        from flowbook.kernel.locations import ROW_ATTRS
        # Should contain Attr(df, a) for each a in ROW_ATTRS
        assert len(result) == len(ROW_ATTRS)
        assert all(r.type == ReadLocType.ATTR for r in result)


# =============================================================================
# Worked Examples from Plan
# =============================================================================


class TestWorkedExamples:
    def test_independent_column_additions_have_structural_overlap(self):
        """
        B adds df["price"], C adds df["qty"].
        Both affect df.columns, df.shape, etc. — so write-write overlap IS detected.
        ColAdd(df, price) ▷ Attr(df, columns) = True.
        """
        W_B = frozenset({WriteLoc.col_add("df", "price")})
        W_C = frozenset({WriteLoc.col_add("df", "qty")})

        # Write-write overlap via output: W_B ▷ output*(W_C)
        assert has_conflict(W_B, output_set(W_C))

    def test_col_add_doesnt_conflict_with_var_read(self):
        """
        ColAdd(df, price) ▷ Var(df) = false.
        Column add doesn't change the variable binding.
        """
        W_B = frozenset({WriteLoc.col_add("df", "price")})
        R_C = frozenset({ReadLoc.var("df")})
        assert not has_conflict(W_B, R_C)

    def test_column_modify_doesnt_affect_structural_attrs(self):
        """
        Modifying column values doesn't conflict with structural attribute reads.
        Col(df, price) ▷ Attr(df, shape) = false.
        Col(df, price) ▷ Attr(df, columns) = false.
        """
        W = frozenset({WriteLoc.col("df", "price")})
        R = frozenset({ReadLoc.attr("df", "shape"), ReadLoc.attr("df", "columns")})
        assert not has_conflict(W, R)

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

    def test_column_add_affects_structure(self):
        """
        Adding a column DOES conflict with column-structure attributes.
        ColAdd(df, new) ▷ Attr(df, columns) = true.
        """
        W = frozenset({WriteLoc.col_add("df", "new")})
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
            WriteLoc.col_add("df", "new"),
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
            WriteLoc.col_add("df", "new"),
            WriteLoc.col_del("df", "old"),
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
        # "df" has column/structural detail, so no Var(df) — only Col and Attr locs.
        # "config" has no detail, so it gets Var(config).
        expected = frozenset({
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
