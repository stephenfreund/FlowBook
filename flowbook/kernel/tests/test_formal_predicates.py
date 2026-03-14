"""
Tests for formal predicate helpers and Loc types.

These tests verify the implementation matches the formal specification in:
- main.tex (LaTeX proof document)
- FORMAL_DEVELOPMENT.md (Markdown specification)

Each test references the specific formal definition it verifies.
"""

import pytest
from typing import Set, Dict, Optional

from flowbook.kernel.models import (
    Loc,
    LocType,
    LocSet,
    tracking_to_read_locs,
    tracking_to_write_locs,
    locs_intersect,
    get_var_locs,
)
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel.reproducibility_enforcer import (
    _forward_stale,
    _backward_stale,
    _write_before_read,
    _no_read_before_write,
    _no_write_after_read,
    _no_read_and_write,
    _writes_in_range,
    _reads_in_range,
    _overwritten,
)
from flowbook.kernel.notebook_state import NotebookState


# =============================================================================
# Tests for Loc class (FORMAL_DEVELOPMENT.md §1.1, §8.1-8.3)
# =============================================================================


class TestLoc:
    """Tests for the Loc location type."""

    def test_var_location(self):
        """Test Var(x) location creation."""
        loc = Loc.var("x")
        assert loc.type == LocType.VAR
        assert loc.name == "x"
        assert loc.qualifier is None
        assert str(loc) == "Var(x)"

    def test_column_location(self):
        """Test Col(df, c) location creation."""
        loc = Loc.column("df", "price")
        assert loc.type == LocType.COLUMN
        assert loc.name == "price"
        assert loc.qualifier == "df"
        assert str(loc) == "Col(df, price)"

    def test_file_location(self):
        """Test File(path) location creation."""
        loc = Loc.file("/data/input.csv")
        assert loc.type == LocType.FILE
        assert loc.name == "/data/input.csv"
        assert loc.qualifier is None
        assert str(loc) == "File(/data/input.csv)"

    def test_structural_location(self):
        """Test Structural(df, attr) location creation."""
        loc = Loc.structural("df", "shape")
        assert loc.type == LocType.STRUCTURAL
        assert loc.name == "shape"
        assert loc.qualifier == "df"
        assert str(loc) == "Structural(df, shape)"

    def test_loc_equality(self):
        """Locs with same type/name/qualifier are equal (frozen dataclass)."""
        loc1 = Loc.var("x")
        loc2 = Loc.var("x")
        loc3 = Loc.var("y")
        assert loc1 == loc2
        assert loc1 != loc3

    def test_loc_hashable(self):
        """Locs can be used in sets (frozen dataclass)."""
        locs = {Loc.var("x"), Loc.var("x"), Loc.var("y")}
        assert len(locs) == 2


class TestTrackingToLocs:
    """Tests for converting TrackingData to LocSet."""

    def test_empty_tracking(self):
        """Empty tracking produces empty LocSet."""
        tracking = TrackingData()
        reads = tracking_to_read_locs(tracking)
        writes = tracking_to_write_locs(tracking)
        assert reads == frozenset()
        assert writes == frozenset()

    def test_variable_reads(self):
        """Variable reads become Var locations."""
        tracking = TrackingData(reads_before_writes={"x", "y"})
        reads = tracking_to_read_locs(tracking)
        assert Loc.var("x") in reads
        assert Loc.var("y") in reads
        assert len(reads) == 2

    def test_variable_writes(self):
        """Variable writes become Var locations."""
        tracking = TrackingData(writes={"result", "df"})
        writes = tracking_to_write_locs(tracking)
        assert Loc.var("result") in writes
        assert Loc.var("df") in writes
        assert len(writes) == 2

    def test_column_reads(self):
        """Column reads become Col locations."""
        tracking = TrackingData(
            column_reads_before_writes={"df": {"price", "quantity"}}
        )
        reads = tracking_to_read_locs(tracking)
        assert Loc.column("df", "price") in reads
        assert Loc.column("df", "quantity") in reads

    def test_column_writes(self):
        """Column writes become Col locations."""
        tracking = TrackingData(
            column_writes={"df": {"total"}}
        )
        writes = tracking_to_write_locs(tracking)
        assert Loc.column("df", "total") in writes

    def test_file_reads(self):
        """File reads become File locations."""
        tracking = TrackingData(
            file_reads_before_writes={"/data/input.csv"}
        )
        reads = tracking_to_read_locs(tracking)
        assert Loc.file("/data/input.csv") in reads

    def test_file_writes(self):
        """File writes become File locations."""
        tracking = TrackingData(
            file_writes={"/data/output.csv"}
        )
        writes = tracking_to_write_locs(tracking)
        assert Loc.file("/data/output.csv") in writes

    def test_structural_reads(self):
        """Structural reads become Structural locations."""
        tracking = TrackingData(
            structural_reads={"df": {"shape", "columns"}}
        )
        reads = tracking_to_read_locs(tracking)
        assert Loc.structural("df", "shape") in reads
        assert Loc.structural("df", "columns") in reads

    def test_mixed_tracking(self):
        """All location types are captured together."""
        tracking = TrackingData(
            reads_before_writes={"x"},
            writes={"y"},
            column_reads_before_writes={"df": {"col1"}},
            column_writes={"df": {"col2"}},
            file_reads_before_writes={"/in.csv"},
            file_writes={"/out.csv"},
            structural_reads={"df": {"shape"}},
        )
        reads = tracking_to_read_locs(tracking)
        writes = tracking_to_write_locs(tracking)

        # Reads should include var, column, file, structural
        assert Loc.var("x") in reads
        assert Loc.column("df", "col1") in reads
        assert Loc.file("/in.csv") in reads
        assert Loc.structural("df", "shape") in reads

        # Writes should include var, column, file
        assert Loc.var("y") in writes
        assert Loc.column("df", "col2") in writes
        assert Loc.file("/out.csv") in writes


class TestLocsIntersect:
    """Tests for locs_intersect helper."""

    def test_empty_sets(self):
        """Empty sets don't intersect."""
        assert not locs_intersect(frozenset(), frozenset())

    def test_disjoint_sets(self):
        """Disjoint sets don't intersect."""
        a = frozenset({Loc.var("x")})
        b = frozenset({Loc.var("y")})
        assert not locs_intersect(a, b)

    def test_overlapping_sets(self):
        """Overlapping sets intersect."""
        a = frozenset({Loc.var("x"), Loc.var("y")})
        b = frozenset({Loc.var("y"), Loc.var("z")})
        assert locs_intersect(a, b)

    def test_same_var_different_types(self):
        """Same name but different types don't match."""
        a = frozenset({Loc.var("df")})
        b = frozenset({Loc.column("df", "col")})
        assert not locs_intersect(a, b)


class TestGetVarLocs:
    """Tests for get_var_locs helper."""

    def test_extracts_var_names(self):
        """Extracts only Var location names."""
        locs = frozenset({
            Loc.var("x"),
            Loc.var("y"),
            Loc.column("df", "col"),
            Loc.file("/path"),
        })
        vars = get_var_locs(locs)
        assert vars == {"x", "y"}


# =============================================================================
# Tests for Validity Predicates (main.tex §Validity predicates, FORMAL_DEVELOPMENT.md §3.2)
# =============================================================================


class TestNoReadAndWrite:
    """Tests for NoReadAndWrite(R, W, i) ≝ Rᵢ ∩ Wᵢ = ∅"""

    def test_disjoint_sets(self):
        """Disjoint R and W satisfy predicate."""
        R_i = {"x", "y"}
        W_i = {"z", "w"}
        assert _no_read_and_write(R_i, W_i) is True

    def test_overlapping_sets(self):
        """Overlapping R and W violate predicate."""
        R_i = {"x", "y"}
        W_i = {"y", "z"}
        assert _no_read_and_write(R_i, W_i) is False

    def test_empty_sets(self):
        """Empty sets satisfy predicate."""
        assert _no_read_and_write(set(), set()) is True


class TestWriteBeforeRead:
    """Tests for WriteBeforeRead(R, W, i) ≝ Rᵢ ⊆ W_{1..i-1}"""

    def test_all_reads_from_prior_writes(self):
        """Reads subset of prior writes satisfies predicate."""
        R_i = {"x", "y"}
        W_before_i = {"x", "y", "z"}
        assert _write_before_read(R_i, W_before_i) is True

    def test_read_not_in_prior_writes(self):
        """Read not in prior writes violates predicate."""
        R_i = {"x", "y"}
        W_before_i = {"x"}  # Missing y
        assert _write_before_read(R_i, W_before_i) is False

    def test_empty_reads(self):
        """Empty reads always satisfy predicate."""
        assert _write_before_read(set(), {"x"}) is True
        assert _write_before_read(set(), set()) is True


class TestNoReadBeforeWrite:
    """Tests for NoReadBeforeWrite(R, W, i) ≝ Rᵢ ∩ W_{i+1..n} = ∅"""

    def test_no_overlap_with_future_writes(self):
        """Reads disjoint from future writes satisfies predicate."""
        R_i = {"x", "y"}
        W_after_i = {"z", "w"}
        assert _no_read_before_write(R_i, W_after_i) is True

    def test_reads_from_future_writes(self):
        """Reads overlapping future writes violates predicate (forward contamination)."""
        R_i = {"x", "y"}
        W_after_i = {"y", "z"}
        assert _no_read_before_write(R_i, W_after_i) is False


class TestNoWriteAfterRead:
    """Tests for NoWriteAfterRead(R, W, i) ≝ Wᵢ ∩ R_{1..i-1} = ∅"""

    def test_no_overlap_with_prior_reads(self):
        """Writes disjoint from prior reads satisfies predicate."""
        W_i = {"z", "w"}
        R_before_i = {"x", "y"}
        assert _no_write_after_read(W_i, R_before_i) is True

    def test_writes_to_prior_reads(self):
        """Writes overlapping prior reads violates predicate (backward mutation)."""
        W_i = {"x", "z"}
        R_before_i = {"x", "y"}
        assert _no_write_after_read(W_i, R_before_i) is False


# =============================================================================
# Tests for Staleness Predicates (main.tex §Staleness predicates, FORMAL_DEVELOPMENT.md §3.3)
# =============================================================================


class TestForwardStale:
    """Tests for ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅"""

    def test_later_cell_reads_written_var(self):
        """Later cell reading a written var becomes stale."""
        W_i_old = set()
        W_i_new = {"x", "y"}
        R_j = {"x"}
        W_j = set()
        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=2, j=5) is True

    def test_later_cell_writes_written_var(self):
        """Later cell writing a written var becomes stale."""
        W_i_old = set()
        W_i_new = {"x"}
        R_j = set()
        W_j = {"x", "z"}
        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=2, j=5) is True

    def test_no_overlap(self):
        """No overlap means no staleness."""
        W_i_old = set()
        W_i_new = {"x"}
        R_j = {"y"}
        W_j = {"z"}
        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=2, j=5) is False

    def test_earlier_cell_not_affected(self):
        """Cell before i is not affected by ForwardStale."""
        W_i_old = set()
        W_i_new = {"x"}
        R_j = {"x"}
        W_j = set()
        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=5, j=2) is False  # j < i

    def test_same_cell_not_affected(self):
        """Same cell (j=i) is not affected."""
        W_i_old = set()
        W_i_new = {"x"}
        R_j = {"x"}
        assert _forward_stale(R_j, set(), W_i_old, W_i_new, i=3, j=3) is False

    def test_old_writes_cause_staleness(self):
        """Old writes (no longer written) still cause staleness."""
        W_i_old = {"x"}  # Cell used to write x
        W_i_new = set()  # Cell no longer writes x
        R_j = {"x"}
        W_j = set()
        # Even though cell i doesn't write x anymore, j reading x is still stale
        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=2, j=5) is True

    def test_union_of_old_and_new_writes(self):
        """Both old and new writes contribute to staleness check."""
        W_i_old = {"x"}  # Cell used to write x
        W_i_new = {"y"}  # Cell now writes y (not x)
        R_j = {"x", "y", "z"}  # Reads all three
        W_j = set()
        # j reads both x (old write) and y (new write), so stale
        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=2, j=5) is True
        # If j only reads z, not stale
        assert _forward_stale({"z"}, set(), W_i_old, W_i_new, i=2, j=5) is False


class TestBackwardStale:
    r"""Tests for BackwardStale(W, W', i, j) ≝ j < i ∧ j = LastWriter(W, i, y) for some y ∈ Wᵢ \ W'ᵢ"""

    def test_removed_write_affects_last_writer(self):
        """Cell that was last writer of removed write becomes stale."""
        W_old_i = {"x", "y"}
        W_new_i = {"y"}  # x was removed

        # Mock last_writer: cell 2 wrote x before cell 5
        def last_writer(var, i):
            if var == "x":
                return 2
            return None

        assert _backward_stale({}, W_new_i, W_old_i, last_writer, i=5, j=2) is True

    def test_no_removed_writes(self):
        """No removed writes means no backward staleness."""
        W_old_i = {"x"}
        W_new_i = {"x", "y"}  # Added y, didn't remove

        def last_writer(var, i):
            return None

        assert _backward_stale({}, W_new_i, W_old_i, last_writer, i=5, j=2) is False

    def test_later_cell_not_affected(self):
        """Cell after i is not affected by BackwardStale."""
        W_old_i = {"x"}
        W_new_i = set()

        def last_writer(var, i):
            return 7  # After i=5

        assert _backward_stale({}, W_new_i, W_old_i, last_writer, i=5, j=7) is False


# =============================================================================
# Tests for Range Functions (FORMAL_DEVELOPMENT.md §1.3)
# =============================================================================


class TestWritesInRange:
    """Tests for W_{start..end} = ⋃_{k ∈ [start..end]} Wₖ"""

    def test_collects_writes_from_range(self):
        """Collects writes from all cells in range."""
        state = NotebookState()
        cell_order = ["a", "b", "c", "d"]

        # Set up tracking data (used by get_tracking() which _writes_in_range calls)
        state.tracking_data["a"] = TrackingData(writes={"x"})
        state.tracking_data["b"] = TrackingData(writes={"y"})
        state.tracking_data["c"] = TrackingData(writes={"z"})
        state.tracking_data["d"] = TrackingData(writes={"w"})

        # W_{1..2} should include writes from b and c (indices 1, 2)
        writes = _writes_in_range(state, cell_order, 1, 2)
        assert writes == {"y", "z"}

    def test_empty_range(self):
        """Empty range returns empty set."""
        state = NotebookState()
        writes = _writes_in_range(state, [], 0, 0)
        assert writes == set()


class TestOverwritten:
    """Tests for Overwritten(W, i) ≝ W_{i+1..n}"""

    def test_collects_writes_after_position(self):
        """Collects writes from all cells after position i."""
        state = NotebookState()
        cell_order = ["a", "b", "c", "d"]

        # Set up tracking data
        state.tracking_data["a"] = TrackingData(writes={"x"})
        state.tracking_data["b"] = TrackingData(writes={"y"})
        state.tracking_data["c"] = TrackingData(writes={"z"})
        state.tracking_data["d"] = TrackingData(writes={"w"})

        # Overwritten(W, 1) = W_{2..3} = writes from c and d
        overwritten = _overwritten(state, cell_order, 1)
        assert overwritten == {"z", "w"}


# =============================================================================
# Integration Tests
# =============================================================================


class TestFormalPredicatesIntegration:
    """Integration tests combining multiple predicates."""

    def test_forward_stale_with_tracking_data(self):
        """ForwardStale works with actual TrackingData."""
        # Cell i writes x (new writes)
        tracking_i = TrackingData(writes={"x", "y"})
        # Cell j reads x
        tracking_j = TrackingData(reads_before_writes={"x", "z"})

        W_i_old = set()  # No previous writes
        W_i_new = tracking_i.writes
        R_j = tracking_j.reads_before_writes
        W_j = tracking_j.writes

        assert _forward_stale(R_j, W_j, W_i_old, W_i_new, i=1, j=3) is True

    def test_backward_mutation_with_tracking_data(self):
        """NoWriteAfterRead detects backward mutation."""
        # Earlier cell reads x
        tracking_earlier = TrackingData(reads_before_writes={"x", "y"})
        # Current cell writes x
        tracking_current = TrackingData(writes={"x", "z"})

        R_before = tracking_earlier.reads_before_writes
        W_i = tracking_current.writes

        # This would be a backward mutation
        assert _no_write_after_read(W_i, R_before) is False

    def test_complete_validity_check(self):
        """All validity predicates pass for valid execution."""
        # Cell 3 in a notebook
        R_i = {"a", "b"}  # Reads a and b
        W_i = {"c"}       # Writes c (no overlap with reads)
        W_before = {"a", "b", "d"}  # Prior cells wrote a, b, d
        W_after = {"e", "f"}  # Later cells write e, f
        R_before = {"d"}  # Prior cells read d

        # All validity predicates should pass
        assert _no_read_and_write(R_i, W_i) is True
        assert _write_before_read(R_i, W_before) is True
        assert _no_read_before_write(R_i, W_after) is True
        assert _no_write_after_read(W_i, R_before) is True

    def test_forward_contamination_detection(self):
        """NoReadBeforeWrite detects forward contamination."""
        # Cell i reads from a variable that a later cell already wrote
        R_i = {"future_var"}
        W_after_i = {"future_var", "other"}

        assert _no_read_before_write(R_i, W_after_i) is False


# =============================================================================
# Tests for Loc-based predicates with full location types
# =============================================================================


class TestLocBasedChecks:
    """Tests ensuring column/file locations work in predicates."""

    def test_column_level_forward_stale(self):
        """Column-level tracking prevents false positive staleness."""
        # Cell i writes df['price']
        tracking_i = TrackingData(
            writes={"df"},
            column_writes={"df": {"price"}}
        )
        # Cell j reads df['quantity'] (different column)
        tracking_j = TrackingData(
            reads_before_writes={"df"},
            column_reads_before_writes={"df": {"quantity"}}
        )

        # At variable level, this looks like overlap
        W_i_vars = tracking_i.writes
        R_j_vars = tracking_j.reads_before_writes
        assert bool(W_i_vars & R_j_vars)  # df overlaps at var level

        # But at Loc level, they're different locations
        W_i_locs = tracking_to_write_locs(tracking_i)
        R_j_locs = tracking_to_read_locs(tracking_j)

        # Check that column locations don't intersect
        col_write = Loc.column("df", "price")
        col_read = Loc.column("df", "quantity")
        assert col_write in W_i_locs
        assert col_read in R_j_locs
        assert col_write != col_read

    def test_file_tracking_in_locs(self):
        """File locations are properly tracked."""
        tracking = TrackingData(
            file_reads_before_writes={"/data/train.csv"},
            file_writes={"/data/model.pkl"}
        )

        reads = tracking_to_read_locs(tracking)
        writes = tracking_to_write_locs(tracking)

        assert Loc.file("/data/train.csv") in reads
        assert Loc.file("/data/model.pkl") in writes
        assert not locs_intersect(reads, writes)

    def test_structural_tracking_in_locs(self):
        """Structural attribute locations are tracked."""
        tracking = TrackingData(
            structural_reads={"df": {"shape", "columns"}}
        )

        reads = tracking_to_read_locs(tracking)

        assert Loc.structural("df", "shape") in reads
        assert Loc.structural("df", "columns") in reads
