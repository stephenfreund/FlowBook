"""
Tests for typed formal predicates and write-write overlap.

These test the column-level ▷-based predicate helpers in locations.py,
and verify that write-write overlap uses typed locs (not variable names).

Formal ref: FORMAL_DEVELOPMENT.md §3.2-3.3, CONFLICT_RELATION.md
"""

import pytest

from flowbook.kernel.loc_ids import LocRef
from flowbook.kernel.locations import (
    ReadLoc, WriteLoc, ReadLocSet, WriteLocSet,
    no_read_and_write,
    no_read_before_write,
    no_write_after_read,
    forward_stale_reads,
    forward_stale_writes,
    write_conflicts_write,
    wlocs_conflict_wlocs,
    wlocs_conflict_rlocs,
    COLS_READ_ATTRS,
    ROWS_READ_ATTRS,
    BOTH_READ_ATTRS,
)


# ============================================================================
# Fixtures
# ============================================================================

LR_DF = LocRef(loc_id=1, var_name="df")
LR_X = LocRef(loc_id=1, var_name="X")    # alias of df
LR_DF2 = LocRef(loc_id=2, var_name="df2")  # user copy


def _rset(*locs) -> ReadLocSet:
    return frozenset(locs)


def _wset(*locs) -> WriteLocSet:
    return frozenset(locs)


# ============================================================================
# NoReadAndWrite — Rᵢ ∩ Wᵢ = ∅
# ============================================================================

class TestNoReadAndWrite:
    """Cell should not read and write the same location."""

    def test_no_overlap(self):
        """Reading x, writing y → no conflict."""
        R = _rset(ReadLoc.var("x"))
        W = _wset(WriteLoc.var("y"))
        assert not no_read_and_write(R, W)

    def test_var_overlap(self):
        """Reading and writing same variable → conflict."""
        R = _rset(ReadLoc.var("x"))
        W = _wset(WriteLoc.var("x"))
        result = no_read_and_write(R, W)
        assert result  # Non-empty → violation

    def test_col_overlap(self):
        """Reading and writing same column → conflict."""
        R = _rset(ReadLoc.col(LR_DF, "price"))
        W = _wset(WriteLoc.col(LR_DF, "price"))
        assert no_read_and_write(R, W)

    def test_col_independent(self):
        """Reading col_a, writing col_b → no conflict (column independence)."""
        R = _rset(ReadLoc.col(LR_DF, "col_a"))
        W = _wset(WriteLoc.col(LR_DF, "col_b"))
        assert not no_read_and_write(R, W)

    def test_col_alias_overlap(self):
        """Reading Col(X, price), writing Col(df, price) → conflict via loc_id."""
        R = _rset(ReadLoc.col(LR_X, "price"))
        W = _wset(WriteLoc.col(LR_DF, "price"))
        assert no_read_and_write(R, W)

    def test_col_copy_no_overlap(self):
        """Reading Col(df2, price), writing Col(df, price) → no conflict (different object)."""
        R = _rset(ReadLoc.col(LR_DF2, "price"))
        W = _wset(WriteLoc.col(LR_DF, "price"))
        assert not no_read_and_write(R, W)

    def test_var_write_col_read(self):
        """Var("df") write conflicts via Var("df") read (always in read set).

        Col(df, price) alone doesn't conflict with Var("df") write.
        The read set must include Var("df") for rebinding detection.
        """
        R = _rset(ReadLoc.col(LR_DF, "price"), ReadLoc.var("df"))
        W = _wset(WriteLoc.var("df"))
        assert no_read_and_write(R, W)

    def test_col_write_cols_read(self):
        """Col(df, price) write conflicts with Cols(df) read."""
        R = _rset(ReadLoc.cols("df", qualifier=LR_DF))
        W = _wset(WriteLoc.col(LR_DF, "price"))
        assert no_read_and_write(R, W)

    def test_col_write_does_not_conflict_with_rows_read(self):
        """Col(df, price) write does NOT conflict with Rows(df) read."""
        R = _rset(ReadLoc.rows("df", qualifier=LR_DF))
        W = _wset(WriteLoc.col(LR_DF, "price"))
        assert not no_read_and_write(R, W)


# ============================================================================
# NoReadBeforeWrite — Rᵢ ∩ W_{i+1..n} = ∅ (forward contamination)
# ============================================================================

class TestNoReadBeforeWrite:
    """Cell i should not read locations written by later cells."""

    def test_no_contamination(self):
        """Cell i reads x, later cells don't write x → no contamination."""
        R_i = _rset(ReadLoc.var("x"))
        W_after = _wset(WriteLoc.var("y"))
        assert not no_read_before_write(R_i, W_after)

    def test_var_contamination(self):
        """Cell i reads x, later cell writes x → contamination."""
        R_i = _rset(ReadLoc.var("x"))
        W_after = _wset(WriteLoc.var("x"))
        assert no_read_before_write(R_i, W_after)

    def test_col_contamination(self):
        """Cell i reads Col(df, price), later cell writes Col(df, price) → contamination."""
        R_i = _rset(ReadLoc.col(LR_DF, "price"))
        W_after = _wset(WriteLoc.col(LR_DF, "price"))
        assert no_read_before_write(R_i, W_after)

    def test_col_independent_no_contamination(self):
        """Cell i reads Col(df, price), later cell writes Col(df, qty) → no contamination."""
        R_i = _rset(ReadLoc.col(LR_DF, "price"))
        W_after = _wset(WriteLoc.col(LR_DF, "qty"))
        assert not no_read_before_write(R_i, W_after)

    def test_col_alias_contamination(self):
        """Cell i reads Col(X, price), later cell writes Col(df, price) → contamination via loc_id."""
        R_i = _rset(ReadLoc.col(LR_X, "price"))
        W_after = _wset(WriteLoc.col(LR_DF, "price"))
        assert no_read_before_write(R_i, W_after)

    def test_rows_contaminates_all_cols(self):
        """Cell i reads Col(df, price), later cell writes Rows(df) → contamination."""
        R_i = _rset(ReadLoc.col(LR_DF, "price"))
        W_after = _wset(WriteLoc.rows("df", qualifier=LR_DF))
        assert no_read_before_write(R_i, W_after)

    def test_col_contaminates_cols_reads(self):
        """Cell i reads Cols(df), later cell writes Col(df, new) → contamination.

        Col conflicts with Cols (column structure), so writing any column
        invalidates Cols reads.
        """
        R_i = _rset(ReadLoc.cols("df", qualifier=LR_DF))
        W_after = _wset(WriteLoc.col(LR_DF, "new"))
        assert no_read_before_write(R_i, W_after)


# ============================================================================
# NoWriteAfterRead — Wᵢ ∩ R_{1..i-1} = ∅ (backward mutation)
# ============================================================================

class TestNoWriteAfterRead:
    """Cell i should not write locations that earlier cells read."""

    def test_no_backward_mutation(self):
        """Cell i writes y, earlier cells read x → no mutation."""
        W_i = _wset(WriteLoc.var("y"))
        R_before = _rset(ReadLoc.var("x"))
        assert not no_write_after_read(W_i, R_before)

    def test_var_backward_mutation(self):
        """Cell i writes x, earlier cell read x → backward mutation."""
        W_i = _wset(WriteLoc.var("x"))
        R_before = _rset(ReadLoc.var("x"))
        assert no_write_after_read(W_i, R_before)

    def test_col_backward_mutation(self):
        """Cell i writes Col(df, price), earlier cell read Col(df, price) → mutation."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_before = _rset(ReadLoc.col(LR_DF, "price"))
        assert no_write_after_read(W_i, R_before)

    def test_col_independent_no_mutation(self):
        """Cell i writes Col(df, qty), earlier cell read Col(df, price) → no mutation."""
        W_i = _wset(WriteLoc.col(LR_DF, "qty"))
        R_before = _rset(ReadLoc.col(LR_DF, "price"))
        assert not no_write_after_read(W_i, R_before)

    def test_col_alias_backward_mutation(self):
        """Cell i writes Col(df, price), earlier cell read Col(X, price) → mutation via loc_id."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_before = _rset(ReadLoc.col(LR_X, "price"))
        assert no_write_after_read(W_i, R_before)

    def test_rows_backward_mutation_all_cols(self):
        """Cell i writes Rows(df), earlier cell read Col(df, price) → mutation."""
        W_i = _wset(WriteLoc.rows("df", qualifier=LR_DF))
        R_before = _rset(ReadLoc.col(LR_DF, "price"))
        assert no_write_after_read(W_i, R_before)

    def test_col_write_backward_mutation_cols_read(self):
        """Cell i writes Col(df, price), earlier cell read Cols(df) → mutation."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_before = _rset(ReadLoc.cols("df", qualifier=LR_DF))
        assert no_write_after_read(W_i, R_before)

    def test_col_write_no_mutation_rows_read(self):
        """Cell i writes Col(df, price), earlier cell read Rows(df) → no mutation.

        Col does not conflict with Rows (row structure is unaffected).
        """
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_before = _rset(ReadLoc.rows("df", qualifier=LR_DF))
        assert not no_write_after_read(W_i, R_before)


# ============================================================================
# ForwardStale reads — W'ᵢ ▷ Rⱼ
# ============================================================================

class TestForwardStaleReads:
    """Cell j becomes stale if cell i's writes invalidate j's reads."""

    def test_no_stale(self):
        """Cell i writes y, cell j reads x → not stale."""
        W_i = _wset(WriteLoc.var("y"))
        R_j = _rset(ReadLoc.var("x"))
        assert not forward_stale_reads(W_i, R_j)

    def test_var_stale(self):
        """Cell i writes x, cell j reads x → stale."""
        W_i = _wset(WriteLoc.var("x"))
        R_j = _rset(ReadLoc.var("x"))
        assert forward_stale_reads(W_i, R_j)

    def test_col_stale_same_col(self):
        """Cell i writes Col(df, price), cell j reads Col(df, price) → stale."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_j = _rset(ReadLoc.col(LR_DF, "price"))
        assert forward_stale_reads(W_i, R_j)

    def test_col_independent_not_stale(self):
        """Cell i writes Col(df, price), cell j reads Col(df, qty) → NOT stale.

        This is the key column-independence test.
        """
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_j = _rset(ReadLoc.col(LR_DF, "qty"))
        assert not forward_stale_reads(W_i, R_j)

    def test_col_alias_stale(self):
        """Cell i writes Col(df, price), cell j reads Col(X, price) → stale via loc_id."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_j = _rset(ReadLoc.col(LR_X, "price"))
        assert forward_stale_reads(W_i, R_j)

    def test_col_copy_not_stale(self):
        """Cell i writes Col(df, price), cell j reads Col(df2, price) → NOT stale."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_j = _rset(ReadLoc.col(LR_DF2, "price"))
        assert not forward_stale_reads(W_i, R_j)

    def test_rows_stales_all_col_reads(self):
        """Rows(df) stales all column reads on df."""
        W_i = _wset(WriteLoc.rows("df", qualifier=LR_DF))
        R_j = _rset(ReadLoc.col(LR_DF, "price"), ReadLoc.col(LR_DF, "qty"))
        result = forward_stale_reads(W_i, R_j)
        assert len(result) == 1  # Rows(df) is the one conflicting write

    def test_col_stales_cols_reads(self):
        """Col(df, price) stales Cols(df) reads."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_j = _rset(ReadLoc.cols("df", qualifier=LR_DF))
        assert forward_stale_reads(W_i, R_j)

    def test_col_does_not_stale_rows_reads(self):
        """Col(df, price) does NOT stale Rows(df) reads.

        Col does not conflict with Rows (row structure unaffected).
        """
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        R_j = _rset(ReadLoc.rows("df", qualifier=LR_DF))
        assert not forward_stale_reads(W_i, R_j)


# ============================================================================
# ForwardStale writes — W'ᵢ ▷ output*(Wⱼ) (write-write overlap)
# ============================================================================

class TestForwardStaleWrites:
    """Cell j becomes stale if cell i's writes overlap with j's writes.

    This is the key test class for the typed write-write overlap upgrade.
    Previously this was checked via variable-name set intersection, now
    it uses ▷ with output() for column-level precision.
    """

    def test_same_var(self):
        """Both write Var(x) → overlap."""
        W_i = _wset(WriteLoc.var("x"))
        W_j = _wset(WriteLoc.var("x"))
        assert forward_stale_writes(W_i, W_j)

    def test_different_var(self):
        """Write different variables → no overlap."""
        W_i = _wset(WriteLoc.var("x"))
        W_j = _wset(WriteLoc.var("y"))
        assert not forward_stale_writes(W_i, W_j)

    def test_same_col(self):
        """Both write Col(df, price) → overlap."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        W_j = _wset(WriteLoc.col(LR_DF, "price"))
        assert forward_stale_writes(W_i, W_j)

    def test_independent_cols_no_overlap(self):
        """Cell i writes Col(df, price), cell j writes Col(df, qty) → NO overlap.

        THIS IS THE KEY TEST: with variable-name-level checking, both
        write "df" so they would overlap. With typed ▷, they don't because
        output(Col(df, qty)) = {Col(df, qty)} and
        Col(df, price) ▷ Col(df, qty) = False (different columns).
        Column independence is preserved at the write-write level.
        """
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        W_j = _wset(WriteLoc.col(LR_DF, "qty"))
        # Independent column writes: no overlap
        assert not forward_stale_writes(W_i, W_j)

    def test_col_alias_overlap(self):
        """Cell i writes Col(df, price), cell j writes Col(X, price) → overlap via loc_id."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        W_j = _wset(WriteLoc.col(LR_X, "price"))
        assert forward_stale_writes(W_i, W_j)

    def test_col_copy_no_overlap(self):
        """Cell i writes Col(df, price), cell j writes Col(df2, price) → no overlap."""
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        W_j = _wset(WriteLoc.col(LR_DF2, "price"))
        assert not forward_stale_writes(W_i, W_j)

    def test_rows_vs_col_overlap(self):
        """Cell i writes Rows(df), cell j writes Col(df, price) → overlap.

        Rows(df) ▷ output(Col(df, price)) includes Rows(df) ▷ Col(df, price) = True.
        """
        W_i = _wset(WriteLoc.rows("df", qualifier=LR_DF))
        W_j = _wset(WriteLoc.col(LR_DF, "price"))
        assert forward_stale_writes(W_i, W_j)

    def test_col_vs_rows_overlap(self):
        """Cell i writes Col(df, price), cell j writes Rows(df) → overlap.

        Col(df, price) ▷▷ Rows(df) = True (per ▷▷ matrix).
        """
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        W_j = _wset(WriteLoc.rows("df", qualifier=LR_DF))
        assert forward_stale_writes(W_i, W_j)

    def test_cols_vs_cols_same(self):
        """Both write Cols(df) → overlap."""
        W_i = _wset(WriteLoc.cols("df", qualifier=LR_DF))
        W_j = _wset(WriteLoc.cols("df", qualifier=LR_DF))
        assert forward_stale_writes(W_i, W_j)

    def test_cols_vs_rows_no_overlap(self):
        """Write Cols(df), write Rows(df) → no overlap."""
        W_i = _wset(WriteLoc.cols("df", qualifier=LR_DF))
        W_j = _wset(WriteLoc.rows("df", qualifier=LR_DF))
        assert not forward_stale_writes(W_i, W_j)

    def test_file_overlap(self):
        """Both write File(data.csv) → overlap."""
        W_i = _wset(WriteLoc.file("data.csv"))
        W_j = _wset(WriteLoc.file("data.csv"))
        assert forward_stale_writes(W_i, W_j)

    def test_file_no_overlap(self):
        """Write different files → no overlap."""
        W_i = _wset(WriteLoc.file("data.csv"))
        W_j = _wset(WriteLoc.file("other.csv"))
        assert not forward_stale_writes(W_i, W_j)

    def test_var_write_vs_col_write(self):
        """Var("df") does NOT overlap with Col(df, price) via write-write path.

        Var("df") ▷ output(Col(df, price)) = Var("df") ▷ Col(df, price) = False.
        Rebinding detection is handled by the read overlap path instead:
        the read set always includes Var("df"), so Var("df") ▷ Var("df") catches it.
        """
        W_i = _wset(WriteLoc.var("df"))
        W_j = _wset(WriteLoc.col(LR_DF, "price"))
        assert not forward_stale_writes(W_i, W_j)

    def test_col_write_vs_var_write(self):
        """Col(df, price) vs Var("df"): Col ▷ output(Var("df")) = Col ▷ Var("df") = False.

        But wait — Col(df, price) has output that includes Attr(df, values), and
        Var("df") ▷ Attr(df, values) = True. So the reverse direction catches it.
        Actually, forward_stale_writes checks W_i ▷ output*(W_j), so:
        Col(df, price) ▷ output(Var("df")) = Col(df, price) ▷ {Var("df")} = False.

        This is asymmetric: Var("df") dominates Col, but not vice versa via output.
        The full ForwardStale checks BOTH read and write overlap, so this case
        is caught by the read overlap check (Col(df, price) ▷ R_j).
        """
        W_i = _wset(WriteLoc.col(LR_DF, "price"))
        W_j = _wset(WriteLoc.var("df"))
        # Col(df, price) ▷ output(Var("df")) = Col(df, price) ▷ {Var("df")} = False
        assert not forward_stale_writes(W_i, W_j)


# ============================================================================
# ▷▷ write_conflicts_write relation
# ============================================================================

class TestWriteConflictsWrite:
    """Verify the ▷▷ write-write conflict relation with LocRef qualifiers."""

    def test_var_same(self):
        assert write_conflicts_write(WriteLoc.var("x"), WriteLoc.var("x"))

    def test_var_different(self):
        assert not write_conflicts_write(WriteLoc.var("x"), WriteLoc.var("y"))

    def test_col_same(self):
        """Col(df, price) ▷▷ Col(df, price) = True."""
        assert write_conflicts_write(WriteLoc.col(LR_DF, "price"), WriteLoc.col(LR_DF, "price"))

    def test_col_different_column(self):
        """Col(df, price) ▷▷ Col(df, qty) = False (independent columns)."""
        assert not write_conflicts_write(WriteLoc.col(LR_DF, "price"), WriteLoc.col(LR_DF, "qty"))

    def test_col_vs_rows(self):
        """Col(df, price) ▷▷ Rows(df) = True."""
        assert write_conflicts_write(WriteLoc.col(LR_DF, "price"), WriteLoc.rows("df", qualifier=LR_DF))

    def test_rows_vs_col(self):
        """Rows(df) ▷▷ Col(df, any) = True."""
        assert write_conflicts_write(WriteLoc.rows("df", qualifier=LR_DF), WriteLoc.col(LR_DF, "any"))

    def test_cols_same(self):
        """Cols(df) ▷▷ Cols(df) = True."""
        assert write_conflicts_write(WriteLoc.cols("df", qualifier=LR_DF), WriteLoc.cols("df", qualifier=LR_DF))

    def test_cols_vs_rows_no_conflict(self):
        """Cols(df) ▷▷ Rows(df) = False."""
        assert not write_conflicts_write(WriteLoc.cols("df", qualifier=LR_DF), WriteLoc.rows("df", qualifier=LR_DF))

    def test_rows_same(self):
        """Rows(df) ▷▷ Rows(df) = True."""
        assert write_conflicts_write(WriteLoc.rows("df", qualifier=LR_DF), WriteLoc.rows("df", qualifier=LR_DF))

    def test_file_same(self):
        assert write_conflicts_write(WriteLoc.file("data.csv"), WriteLoc.file("data.csv"))

    def test_file_different(self):
        assert not write_conflicts_write(WriteLoc.file("data.csv"), WriteLoc.file("other.csv"))

    def test_col_alias_overlap(self):
        """Col(df, price) ▷▷ Col(X, price) = True via shared loc_id."""
        assert write_conflicts_write(WriteLoc.col(LR_DF, "price"), WriteLoc.col(LR_X, "price"))

    def test_col_copy_no_overlap(self):
        """Col(df, price) ▷▷ Col(df2, price) = False (different objects)."""
        assert not write_conflicts_write(WriteLoc.col(LR_DF, "price"), WriteLoc.col(LR_DF2, "price"))


# ============================================================================
# Attribute classification coverage
# ============================================================================

class TestReadAttrClassification:
    """Verify attribute classification constants are correct."""

    def test_cols_read_attrs_contents(self):
        """COLS_READ_ATTRS contains column-structure-revealing attributes."""
        assert COLS_READ_ATTRS == frozenset({
            "columns", "keys", "dtypes", "iter",
            "head", "tail", "sample", "info",
            "select_dtypes", "memory_usage",
        })

    def test_rows_read_attrs_contents(self):
        """ROWS_READ_ATTRS contains index, len, empty."""
        assert ROWS_READ_ATTRS == frozenset({"index", "len", "empty"})

    def test_both_read_attrs_contents(self):
        """BOTH_READ_ATTRS contains cross-cutting structural attributes."""
        assert BOTH_READ_ATTRS == frozenset({
            "shape", "size", "axes", "values", "T", "describe",
            "to_dict", "to_records", "to_numpy",
        })

    def test_col_conflicts_with_cols_read(self):
        """Col(df, c) ▷ Cols(df) = True."""
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.cols("df", qualifier=LR_DF)
        assert wlocs_conflict_rlocs(frozenset({w}), frozenset({r}))

    def test_col_does_not_conflict_with_rows_read(self):
        """Col(df, c) ▷ Rows(df) = False."""
        w = WriteLoc.col(LR_DF, "price")
        r = ReadLoc.rows("df", qualifier=LR_DF)
        assert not wlocs_conflict_rlocs(frozenset({w}), frozenset({r}))


# ============================================================================
# Integration: NotebookState.writes stores diff-derived locs
# ============================================================================

class TestNotebookStateWriteStorage:
    """Verify that NotebookState.writes includes diff-derived WriteLocs."""

    def test_writes_include_col_for_column_added(self):
        """record_execution with ColumnAdded typed_change stores Col WriteLoc.

        ColumnAdded now maps to WriteLoc.col() (not col_add) in
        changes_to_write_locs(), so the stored write type is 'col'.
        """
        from flowbook.kernel.notebook_state import NotebookState
        from flowbook.kernel.changes import ColumnAdded
        from flowbook.kernel_support.models import TrackingData

        state = NotebookState()
        state.cell_order = ["a"]
        tracking = TrackingData()
        tracking.writes = {"df"}
        tracking.column_writes = {"df": {"new_col"}}

        state.record_execution(
            "a",
            tracking=tracking,
            typed_changes=[ColumnAdded(variable="df", column="new_col")],
        )

        writes = state.writes["a"]
        write_types = {w.type.value for w in writes}
        assert "col" in write_types, f"Expected col in {write_types}"
        assert "var" in write_types  # From tracking.writes

    def test_writes_include_rows(self):
        """record_execution with RowsAdded typed_change stores Rows WriteLoc."""
        from flowbook.kernel.notebook_state import NotebookState
        from flowbook.kernel.changes import RowsAdded
        from flowbook.kernel_support.models import TrackingData

        state = NotebookState()
        state.cell_order = ["a"]
        tracking = TrackingData()
        tracking.writes = {"df"}

        state.record_execution(
            "a",
            tracking=tracking,
            typed_changes=[RowsAdded(variable="df", count=2)],
        )

        writes = state.writes["a"]
        write_types = {w.type.value for w in writes}
        assert "rows" in write_types

    def test_unrecoverable_mutation_not_stored(self):
        """In-place mutation (diff-detected but not in tracking writes) NOT stored.

        If tracking says writes=set() but diff detects ValueChanged("x"),
        the diff-derived WriteLoc should NOT be stored because it's an
        unrecoverable mutation, not a tracked write.
        """
        from flowbook.kernel.notebook_state import NotebookState
        from flowbook.kernel.changes import ValueChanged
        from flowbook.kernel_support.models import TrackingData

        state = NotebookState()
        state.cell_order = ["a"]
        tracking = TrackingData()
        tracking.writes = set()  # Cell doesn't claim to write x
        tracking.reads_before_writes = {"x"}  # Cell reads x

        state.record_execution(
            "a",
            tracking=tracking,
            typed_changes=[ValueChanged(variable="x")],
        )

        writes = state.writes["a"]
        # x should NOT be in writes because tracking didn't record it as a write
        write_var_names = {w.var_name() for w in writes}
        assert "x" not in write_var_names

    def test_no_typed_changes_falls_back(self):
        """Without typed_changes, writes come from tracking only."""
        from flowbook.kernel.notebook_state import NotebookState
        from flowbook.kernel_support.models import TrackingData

        state = NotebookState()
        state.cell_order = ["a"]
        tracking = TrackingData()
        tracking.writes = {"x"}

        state.record_execution("a", tracking=tracking)

        writes = state.writes["a"]
        assert len(writes) == 1
        assert list(writes)[0].type.value == "var"
