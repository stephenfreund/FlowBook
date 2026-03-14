"""
Tests for [Inst-Delete] transition rule.

The new [Inst-Delete] uses the same ForwardStale and BackwardStale predicates
as [Inst-Run], instead of the old ReadsResidualWrite predicate.

    S →^{Delete(i)} S'
    w = Wᵢ
    R'' = R[i:={}]
    W'' = W[i:={}]
    T''ⱼ = STALE           if ForwardStale(R, W, W'', i, j)
         = STALE           if BackwardStale(W, W'', i, j)
         = Tⱼ              otherwise
    R' = R_{1..i-1}, R_{i+1..n}
    W' = W_{1..i-1}, W_{i+1..n}
    T' = T''_{1..i-1}, T''_{i+1..n}

Since W''ᵢ = {}:
  ForwardStale simplifies to:  j > i ∧ Wᵢ ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
  BackwardStale simplifies to: j < i ∧ j = LastWriter(W, i, y) for y ∈ Wᵢ
"""

import pytest

from flowbook.kernel.models import ReasonType
from flowbook.kernel.notebook_state import NotebookState, CellStatus
from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper


class TestDeleteForwardStale:
    """ForwardStale(R, W, W'', i, j): j > i ∧ Wᵢ ∩ (Rⱼ ∪ Wⱼ) ≠ ∅"""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_forward_read_overlap(self):
        """Deleting A marks downstream reader B stale (Wᵢ ∩ Rⱼ ≠ ∅)."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        reason_types = {r.type for r in reasons}
        assert ReasonType.FORWARD_STALE in reason_types

    def test_forward_write_overlap(self):
        """Deleting A marks downstream writer B stale (Wᵢ ∩ Wⱼ ≠ ∅, Wᵢ ∩ Rⱼ = ∅).

        This is a NEW behavior — old ReadsResidualWrite only checked reads.
        """
        self.helper.set_cell_order(["a", "b"])
        # A writes x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        # B writes x but doesn't read it
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        reason_types = {r.type for r in reasons}
        assert ReasonType.WRITE_OVERLAP in reason_types

    def test_forward_read_and_write_overlap(self):
        """Both read and write overlap — FORWARD_STALE takes priority for read vars."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1, "y": 2},
            writes={"x", "y"},
        )
        # B reads x and writes y
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 3},
            reads={"x"}, writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        locs_by_type = {}
        for r in reasons:
            locs_by_type.setdefault(r.type, set()).add(r.loc)
        # x causes FORWARD_STALE (read overlap)
        assert "x" in locs_by_type.get(ReasonType.FORWARD_STALE, set())
        # y causes WRITE_OVERLAP (write-only overlap)
        assert "y" in locs_by_type.get(ReasonType.WRITE_OVERLAP, set())

    def test_forward_no_overlap(self):
        """No overlap between deleted cell's writes and downstream — no staleness."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"y"}, writes={"z"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" not in result.newly_stale

    def test_forward_multiple_downstream(self):
        """Deleting A marks multiple downstream cells stale."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"x"}, writes={"z"},
        )

        result = self.helper.sdc.set_cell_order(["b", "c"])

        assert "b" in result.newly_stale
        assert "c" in result.newly_stale

    def test_forward_skips_already_stale(self):
        """Already-stale cells are not re-marked."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        # Mark B stale first (e.g., via edit)
        self.helper.sdc.mark_cell_edited("b")
        assert not self.helper.sdc._notebook_state.is_clean("b")

        result = self.helper.sdc.set_cell_order(["b"])

        # B was already stale, should not appear in newly_stale
        assert "b" not in result.newly_stale

    def test_forward_only_affects_cells_after(self):
        """ForwardStale only applies to j > i, not j < i."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
        )

        # Delete B (middle cell)
        result = self.helper.sdc.set_cell_order(["a", "c"])

        # C reads y (written by B) → ForwardStale
        assert "c" in result.newly_stale
        # A is before B → not affected by ForwardStale
        assert "a" not in result.newly_stale


class TestDeleteBackwardStale:
    """BackwardStale(W, W'', i, j): j < i ∧ j = LastWriter(W, i, y) for y ∈ Wᵢ"""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_backward_last_writer_stale(self):
        """Deleting B marks A stale when A was last writer of y before B.

        A writes y, B writes y. Delete B → A is exposed as last writer of y.
        """
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"y": 1},
            writes={"y"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"y": 1}, post_namespace={"y": 2},
            writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert "a" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("a")
        reason_types = {r.type for r in reasons}
        assert ReasonType.BACKWARD_STALE in reason_types

    def test_backward_not_last_writer(self):
        """A is not marked stale when it's not the last writer of y before deleted cell.

        A writes y, C writes y, B writes y. Delete B → C is last writer, not A.
        """
        self.helper.set_cell_order(["a", "c", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"y": 1},
            writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"y": 1}, post_namespace={"y": 2},
            writes={"y"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"y": 2}, post_namespace={"y": 3},
            writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        # C is last writer before B → C is stale
        assert "c" in result.newly_stale
        # A is not the last writer → not affected
        assert "a" not in result.newly_stale

    def test_backward_no_prior_writer(self):
        """No prior writer exists → no backward staleness."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        # B writes y, but no prior cell writes y
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert "a" not in result.newly_stale

    def test_backward_already_stale_skipped(self):
        """Already-stale last writer is not re-marked."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"y": 1},
            writes={"y"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"y": 1}, post_namespace={"y": 2},
            writes={"y"},
        )
        # Mark A stale first
        self.helper.sdc.mark_cell_edited("a")

        result = self.helper.sdc.set_cell_order(["a"])

        # A was already stale
        assert "a" not in result.newly_stale

    def test_backward_multiple_variables(self):
        """BackwardStale checks each y ∈ Wᵢ independently."""
        self.helper.set_cell_order(["a", "c", "b"])
        # A writes x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        # C writes y
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )
        # B writes both x and y
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 10, "y": 20},
            writes={"x", "y"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        # A is last writer of x before B → BackwardStale
        assert "a" in result.newly_stale
        # C is last writer of y before B → BackwardStale
        assert "c" in result.newly_stale


class TestDeleteCombined:
    """Tests combining ForwardStale and BackwardStale in a single delete."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_both_forward_and_backward(self):
        """Delete marks both upstream and downstream cells stale.

        A writes x, B reads x and writes x, C reads x.
        Delete B → A stale (BackwardStale: was last writer of x before B),
                   C stale (ForwardStale: reads x from deleted B).
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            reads={"x"}, writes={"x"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 2}, post_namespace={"x": 2, "y": 3},
            reads={"x"}, writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        assert "a" in result.newly_stale  # BackwardStale
        assert "c" in result.newly_stale  # ForwardStale

    def test_delete_cell_with_disjoint_reads_writes(self):
        """Delete cell that reads y and writes x.

        A writes x, B reads y and writes z, C reads z.
        Delete B → C stale (ForwardStale on z), A clean (no overlap).
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "z": 3},
            reads={"y"}, writes={"z"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "z": 3},
            post_namespace={"x": 1, "z": 3, "w": 4},
            reads={"z"}, writes={"w"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        assert "c" in result.newly_stale  # reads z, B wrote z
        assert "a" not in result.newly_stale  # no overlap


class TestDeleteEdgeCases:
    """Edge cases for [Inst-Delete]."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_delete_unexecuted_cell(self):
        """Deleting a never-executed cell causes no staleness."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        # B is never executed

        result = self.helper.sdc.set_cell_order(["a"])

        assert result.newly_stale == []

    def test_delete_cell_with_no_writes(self):
        """Deleting a cell that wrote nothing causes no staleness."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        # B reads but doesn't write
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1},
            reads={"x"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert result.newly_stale == []

    def test_delete_multiple_cells(self):
        """Deleting multiple cells at once applies staleness for each."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"x", "y"}, writes={"z"},
        )

        # Delete both A and B
        result = self.helper.sdc.set_cell_order(["c"])

        # C reads x (from A) and y (from B) → ForwardStale from both
        assert "c" in result.newly_stale

    def test_delete_skips_co_deleted_cells(self):
        """When deleting A and B together, staleness from A doesn't target B."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )

        # Delete both A and B — no remaining cells to mark stale
        result = self.helper.sdc.set_cell_order([])

        assert result.newly_stale == []

    def test_delete_no_duplicate_in_newly_stale(self):
        """A cell should not appear twice in newly_stale even if affected by multiple deletions."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2},
            reads={"x", "y"},
        )

        # Delete both A and B — C reads from both
        result = self.helper.sdc.set_cell_order(["c"])

        # C should appear only once
        assert result.newly_stale.count("c") == 1


class TestDeleteNotebookState:
    """Tests for NotebookState.handle_delete() using same predicates."""

    def test_handle_delete_forward_stale(self):
        """NotebookState.handle_delete uses FORWARD_STALE for downstream readers."""
        from flowbook.kernel.notebook_state import NotebookState, CellStatus
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.status["C"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("C")
        reasons = state.get_reasons("C")
        assert any(r.type == ReasonType.FORWARD_STALE and r.loc == "x" for r in reasons)

    def test_handle_delete_write_overlap(self):
        """NotebookState.handle_delete uses WRITE_OVERLAP for downstream writers."""
        from flowbook.kernel.notebook_state import NotebookState, CellStatus
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["B"] = {"x"}
        state.writes["C"] = {"x"}
        state.reads["C"] = set()
        state.status["C"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("C")
        reasons = state.get_reasons("C")
        assert any(r.type == ReasonType.WRITE_OVERLAP and r.loc == "x" for r in reasons)

    def test_handle_delete_backward_stale(self):
        """NotebookState.handle_delete uses BACKWARD_STALE for upstream last writer."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        state.writes["B"] = {"x"}
        state.status["A"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("A")
        reasons = state.get_reasons("A")
        assert any(r.type == ReasonType.BACKWARD_STALE and r.loc == "x" for r in reasons)

    def test_handle_delete_cleanup(self):
        """handle_delete removes all state for the deleted cell."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        state.reads["B"] = {"x"}
        state.writes["B"] = {"y"}

        state.handle_delete("B")

        assert "B" not in state.cell_order
        assert "B" not in state.status
        assert "B" not in state.reads
        assert "B" not in state.writes

    def test_handle_delete_no_writes_no_staleness(self):
        """Deleting a cell with no writes causes no staleness."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = set()
        state.reads["B"] = {"x"}
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.handle_delete("A")

        assert state.is_clean("B")
        assert state.is_clean("C")

    def test_handle_delete_preserves_last_writer(self):
        """Deleting a cell keeps last_writer entries for orphan detection."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        state.last_writer["x"] = "A"
        state.status["B"] = CellStatus.clean()

        state.handle_delete("A")

        # last_writer intentionally NOT cleared for orphan detection
        assert state.last_writer.get("x") == "A"

    def test_handle_delete_multiple_readers(self):
        """Multiple downstream readers all get marked stale."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C", "D"]
        state.writes["A"] = {"x"}
        state.reads["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.reads["D"] = {"y"}  # no overlap
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()
        state.status["D"] = CellStatus.clean()

        state.handle_delete("A")

        assert not state.is_clean("B")
        assert not state.is_clean("C")
        assert state.is_clean("D")  # no overlap, stays clean


# =============================================================================
# ForwardStale: Additional Scenarios
# =============================================================================


class TestDeleteForwardStaleAdvanced:
    """Additional ForwardStale tests for [Inst-Delete]."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_forward_reason_includes_deleted_cell_id(self):
        """FORWARD_STALE reason includes cell_id pointing to the deleted cell."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )

        self.helper.sdc.set_cell_order(["b"])

        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        fwd_reasons = [r for r in reasons if r.type == ReasonType.FORWARD_STALE]
        assert len(fwd_reasons) >= 1
        assert fwd_reasons[0].cell_id == "a"
        assert fwd_reasons[0].loc == "x"

    def test_forward_multiple_variables(self):
        """Deleting cell that writes multiple variables marks reader of any stale."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1, "y": 2, "z": 3},
            writes={"x", "y", "z"},
        )
        # B only reads y
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads={"y"}, writes={"w"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        fwd = [r for r in reasons if r.type == ReasonType.FORWARD_STALE]
        assert any(r.loc == "y" for r in fwd)

    def test_forward_chain_a_b_c_delete_middle(self):
        """Chain A→B→C: delete B marks C stale, not A.

        A writes x. B reads x, writes y. C reads y.
        Delete B → C stale (reads y from B), A clean.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        assert "c" in result.newly_stale
        assert "a" not in result.newly_stale

    def test_forward_delete_first_cell(self):
        """Deleting the first cell can only produce ForwardStale (no backward possible)."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
        )

        result = self.helper.sdc.set_cell_order(["b", "c"])

        assert "b" in result.newly_stale  # reads x from A
        assert "c" in result.newly_stale  # reads x from A

    def test_forward_partial_variable_overlap(self):
        """Only variables that overlap cause staleness; others are unaffected."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1, "y": 2},
            writes={"x", "y"},
        )
        # B reads x only, not y
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"x"}, writes={"z"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        fwd = [r for r in reasons if r.type == ReasonType.FORWARD_STALE]
        # Only x should appear as reason, not y
        assert any(r.loc == "x" for r in fwd)
        assert not any(r.loc == "y" for r in fwd)

    def test_forward_write_overlap_reason_includes_cell_id(self):
        """WRITE_OVERLAP reason also includes the deleted cell's ID."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )

        self.helper.sdc.set_cell_order(["b"])

        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        wo = [r for r in reasons if r.type == ReasonType.WRITE_OVERLAP]
        assert len(wo) >= 1
        assert wo[0].cell_id == "a"

    def test_forward_cell_reads_and_writes_same_var(self):
        """Cell that reads AND writes same var triggers NoReadAndWrite, so is already stale.

        When B reads x and writes x, NoReadAndWrite marks B stale during execution.
        Deleting A can't newly-stale B because B is already stale.
        """
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        # B both reads and writes x — triggers NoReadAndWrite
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            reads={"x"}, writes={"x"},
        )
        # B is already stale from NoReadAndWrite
        assert not self.helper.sdc._notebook_state.is_clean("b")

        result = self.helper.sdc.set_cell_order(["b"])

        # B was already stale — not in newly_stale
        assert "b" not in result.newly_stale

    def test_forward_delete_with_five_cells(self):
        """Delete middle cell in a 5-cell notebook."""
        self.helper.set_cell_order(["a", "b", "c", "d", "e"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            writes={"z"},
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"z"}, writes={"w"},
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"},
        )

        # Delete C (writes z)
        result = self.helper.sdc.set_cell_order(["a", "b", "d", "e"])

        # D reads z → stale
        assert "d" in result.newly_stale
        # E reads y, not z → clean
        assert "e" not in result.newly_stale
        # A, B before C → not ForwardStale from C
        assert "a" not in result.newly_stale
        assert "b" not in result.newly_stale


# =============================================================================
# BackwardStale: Additional Scenarios
# =============================================================================


class TestDeleteBackwardStaleAdvanced:
    """Additional BackwardStale tests for [Inst-Delete]."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_backward_reason_includes_deleted_cell_id(self):
        """BACKWARD_STALE reason includes cell_id pointing to the deleted cell."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )

        self.helper.sdc.set_cell_order(["a"])

        reasons = self.helper.sdc._notebook_state.get_reasons("a")
        bwd = [r for r in reasons if r.type == ReasonType.BACKWARD_STALE]
        assert len(bwd) >= 1
        assert bwd[0].cell_id == "b"
        assert bwd[0].loc == "x"

    def test_backward_delete_last_cell(self):
        """Deleting the last cell can only produce BackwardStale (no forward possible)."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert "a" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("a")
        assert any(r.type == ReasonType.BACKWARD_STALE for r in reasons)
        # No forward stale possible (nothing after B)
        assert not any(r.type == ReasonType.FORWARD_STALE for r in reasons)

    def test_backward_chain_three_writers(self):
        """Three cells all write x. Delete last → only immediate predecessor is stale.

        A writes x, B writes x, C writes x. Delete C.
        LastWriter(W, C, x) = B → B stale, A clean.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 2}, post_namespace={"x": 3},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["a", "b"])

        assert "b" in result.newly_stale  # B is last writer of x before C
        assert "a" not in result.newly_stale  # A is not last writer

    def test_backward_delete_middle_writer(self):
        """Delete middle writer in a chain of three.

        A writes x, B writes x, C writes x. Delete B.
        LastWriter(W, B, x) = A → A is marked BackwardStale.
        C is after B → ForwardStale (write overlap on x).
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 2}, post_namespace={"x": 3},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        assert "a" in result.newly_stale  # BackwardStale: was last writer of x before B
        assert "c" in result.newly_stale  # ForwardStale: write overlap on x

    def test_backward_independent_variables(self):
        """BackwardStale only triggers for shared variables.

        A writes x, B writes y. Delete B. A doesn't write y → no BackwardStale.
        """
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert "a" not in result.newly_stale

    def test_backward_multiple_vars_different_last_writers(self):
        """Deleted cell writes x and y. Different cells are last writers for each.

        A writes x, B writes y, C writes x and y. Delete C.
        LastWriter(W, C, x) = A → A stale.
        LastWriter(W, C, y) = B → B stale.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 10, "y": 20},
            writes={"x", "y"},
        )

        result = self.helper.sdc.set_cell_order(["a", "b"])

        assert "a" in result.newly_stale  # last writer of x before C
        assert "b" in result.newly_stale  # last writer of y before C

    def test_backward_same_cell_is_last_writer_for_multiple_vars(self):
        """One upstream cell is last writer for multiple variables the deleted cell writes.

        A writes x and y. B writes x and y. Delete B → A stale for both.
        """
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1, "y": 2},
            writes={"x", "y"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 10, "y": 20},
            writes={"x", "y"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert "a" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("a")
        bwd_locs = {r.loc for r in reasons if r.type == ReasonType.BACKWARD_STALE}
        assert "x" in bwd_locs
        assert "y" in bwd_locs

    def test_backward_skips_co_deleted_last_writer(self):
        """When deleting B and C together, B should not be marked stale for C's BackwardStale.

        A writes x, B writes x, C writes x. Delete B and C together.
        LastWriter(W, C, x) = B, but B is also deleted → skip.
        LastWriter(W, B, x) = A → A stale.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 2}, post_namespace={"x": 3},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        # A should be stale (last non-deleted writer of x before B)
        assert "a" in result.newly_stale


# =============================================================================
# Combined ForwardStale + BackwardStale Scenarios
# =============================================================================


class TestDeleteCombinedAdvanced:
    """Advanced tests combining ForwardStale and BackwardStale."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_diamond_pattern(self):
        """Diamond: A writes x, B reads x writes y, C reads x writes z, D reads y and z.

        Delete A → B stale (reads x), C stale (reads x). D not directly affected.
        """
        self.helper.set_cell_order(["a", "b", "c", "d"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"x"}, writes={"z"},
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y", "z"},
        )

        result = self.helper.sdc.set_cell_order(["b", "c", "d"])

        assert "b" in result.newly_stale  # ForwardStale: reads x
        assert "c" in result.newly_stale  # ForwardStale: reads x
        # D doesn't read x directly, not stale from this delete
        assert "d" not in result.newly_stale

    def test_overwrite_chain_delete_middle(self):
        """A writes x=1, B writes x=2, C reads x (gets 2). Delete B.

        ForwardStale: C reads x, B wrote x → C stale.
        BackwardStale: LastWriter(W, B, x) = A → A stale.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 2}, post_namespace={"x": 2, "y": 3},
            reads={"x"}, writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["a", "c"])

        assert "a" in result.newly_stale  # BackwardStale
        assert "c" in result.newly_stale  # ForwardStale
        # Verify reason types
        a_reasons = {r.type for r in self.helper.sdc._notebook_state.get_reasons("a")}
        c_reasons = {r.type for r in self.helper.sdc._notebook_state.get_reasons("c")}
        assert ReasonType.BACKWARD_STALE in a_reasons
        assert ReasonType.FORWARD_STALE in c_reasons

    def test_swap_pattern(self):
        """A writes x, B writes y reads x. Delete A.

        ForwardStale: B reads x, A wrote x → B stale.
        Note: A cannot read y (B writes y → NoWriteAfterRead would trigger).
        """
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale

    def test_delete_produces_warnings(self):
        """Deleting a cell generates warning messages."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert len(result.warnings) > 0
        assert any("stale" in w.lower() for w in result.warnings)

    def test_delete_backward_produces_warnings(self):
        """BackwardStale from delete generates warning messages."""
        self.helper.set_cell_order(["a", "b"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["a"])

        assert len(result.warnings) > 0
        assert any("backward" in w.lower() for w in result.warnings)


# =============================================================================
# Multi-deletion and Interaction Tests
# =============================================================================


class TestDeleteMultipleInteractions:
    """Tests for multiple simultaneous deletions and their interactions."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_delete_two_adjacent_writers(self):
        """Delete two adjacent cells that both write x.

        A writes x, B writes x, C writes x. Delete A and B.
        ForwardStale from A: B was after A but co-deleted → skip. C writes x → stale.
        ForwardStale from B: C writes x → stale (already).
        BackwardStale from B: A was last writer before B but co-deleted → skip.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 2},
            writes={"x"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 2}, post_namespace={"x": 3},
            writes={"x"},
        )

        result = self.helper.sdc.set_cell_order(["c"])

        assert "c" in result.newly_stale  # ForwardStale from A or B

    def test_delete_two_cells_affecting_different_downstream(self):
        """Delete A and C. B reads from A, D reads from C.

        A writes x, B reads x, C writes y, D reads y.
        Delete A → B stale (reads x).
        Delete C → D stale (reads y).
        """
        self.helper.set_cell_order(["a", "b", "c", "d"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "w": 2},
            reads={"x"}, writes={"w"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "w": 2},
            post_namespace={"x": 1, "w": 2, "y": 3},
            writes={"y"},
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "w": 2, "y": 3},
            post_namespace={"x": 1, "w": 2, "y": 3},
            reads={"y"},
        )

        result = self.helper.sdc.set_cell_order(["b", "d"])

        assert "b" in result.newly_stale
        assert "d" in result.newly_stale

    def test_delete_all_cells(self):
        """Deleting all cells results in empty newly_stale."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
        )

        result = self.helper.sdc.set_cell_order([])

        assert result.newly_stale == []

    def test_delete_with_forward_and_backward_from_different_deletions(self):
        """Two deletions: one causes ForwardStale, another would cause BackwardStale.

        A writes x, B reads x writes y, C writes y. Delete A and C.
        Deletions are processed sequentially:
        From A: B stale (ForwardStale, reads x).
        From C: B is already stale → BackwardStale skipped.
        B only gets ForwardStale reason.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 20},
            writes={"y"},
        )

        result = self.helper.sdc.set_cell_order(["b"])

        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        reason_types = {r.type for r in reasons}
        assert ReasonType.FORWARD_STALE in reason_types  # from deleting A

    def test_backward_stale_from_deletion_when_not_already_stale(self):
        """BackwardStale fires from deletion when target cell is still clean.

        A writes y, B writes y, C reads y. Delete B and C.
        B is processed first: C is after B (ForwardStale on y), but C is co-deleted → skip.
        C is processed second: BackwardStale from C finds B as last writer of y before C,
        but B is co-deleted → skip. A is also a writer of y but not last writer → skip.
        So no staleness (all cells are co-deleted or not last writer).

        Use a simpler case: A writes x, B writes x. Delete B (only B).
        BackwardStale: A was last writer of x before B → A stale.
        """
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            writes={"y"},
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 20},
            writes={"y"},
        )

        # Delete only C (B stays)
        result = self.helper.sdc.set_cell_order(["a", "b"])

        # B was last writer of y before C → BackwardStale
        assert "b" in result.newly_stale
        reasons = self.helper.sdc._notebook_state.get_reasons("b")
        assert any(r.type == ReasonType.BACKWARD_STALE and r.loc == "y" for r in reasons)


# =============================================================================
# NotebookState.handle_delete: Additional Tests
# =============================================================================


class TestDeleteNotebookStateAdvanced:
    """Additional tests for NotebookState.handle_delete()."""

    def test_handle_delete_no_overlap_stays_clean(self):
        """Downstream cell with no overlap stays clean."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = {"x"}
        state.reads["B"] = {"y"}
        state.writes["B"] = {"z"}
        state.reads["C"] = {"z"}
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.handle_delete("A")

        assert state.is_clean("B")
        assert state.is_clean("C")

    def test_handle_delete_already_stale_downstream(self):
        """Already-stale downstream gets additional reason from handle_delete."""
        from flowbook.kernel.models import Reason
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        state.reads["B"] = {"x"}
        state.status["B"] = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})

        state.handle_delete("A")

        # B was already stale, now has additional FORWARD_STALE reason
        assert not state.is_clean("B")
        reasons = state.get_reasons("B")
        assert any(r.type == ReasonType.CODE_CHANGED for r in reasons)
        assert any(r.type == ReasonType.FORWARD_STALE and r.loc == "x" for r in reasons)

    def test_handle_delete_upstream_before_not_writer(self):
        """Upstream cell that doesn't write the same var stays clean."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"y"}  # different var
        state.writes["B"] = {"x"}
        state.status["A"] = CellStatus.clean()

        state.handle_delete("B")

        assert state.is_clean("A")

    def test_handle_delete_first_cell_no_backward(self):
        """Deleting first cell: no upstream → no BackwardStale possible."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        state.reads["B"] = {"x"}
        state.status["B"] = CellStatus.clean()

        state.handle_delete("A")

        assert not state.is_clean("B")  # ForwardStale
        reasons = state.get_reasons("B")
        assert any(r.type == ReasonType.FORWARD_STALE for r in reasons)
        assert not any(r.type == ReasonType.BACKWARD_STALE for r in reasons)

    def test_handle_delete_last_cell_no_forward(self):
        """Deleting last cell: no downstream → only BackwardStale possible."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        state.writes["B"] = {"x"}
        state.status["A"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("A")
        reasons = state.get_reasons("A")
        assert any(r.type == ReasonType.BACKWARD_STALE for r in reasons)

    def test_handle_delete_removes_from_cell_order(self):
        """handle_delete removes the cell from cell_order."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        state.handle_delete("B")

        assert state.cell_order == ["A", "C"]

    def test_handle_delete_cell_not_in_order(self):
        """Deleting a cell not in cell_order is a no-op for staleness."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["X"] = {"z"}
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()

        state.handle_delete("X")  # X not in cell_order

        assert state.is_clean("A")
        assert state.is_clean("B")

    def test_handle_delete_backward_last_writer_scan(self):
        """BackwardStale correctly finds the LAST writer, not the first.

        Order: [A, B, C, D]. A writes x, B writes x, C writes x, D writes x.
        Delete D → LastWriter(W, D, x) = C (not A or B).
        """
        state = NotebookState()
        state.cell_order = ["A", "B", "C", "D"]
        state.writes["A"] = {"x"}
        state.writes["B"] = {"x"}
        state.writes["C"] = {"x"}
        state.writes["D"] = {"x"}
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.handle_delete("D")

        # Only C should be stale (last writer)
        assert not state.is_clean("C")
        assert state.is_clean("A")
        assert state.is_clean("B")

    def test_handle_delete_forward_and_backward_same_var(self):
        """Same variable causes both ForwardStale and BackwardStale.

        Order: [A, B, C]. A writes x, B writes x, C reads x.
        Delete B → A stale (BackwardStale), C stale (ForwardStale).
        """
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = {"x"}
        state.writes["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.status["A"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("A")  # BackwardStale
        assert not state.is_clean("C")  # ForwardStale
        a_reasons = {r.type for r in state.get_reasons("A")}
        c_reasons = {r.type for r in state.get_reasons("C")}
        assert ReasonType.BACKWARD_STALE in a_reasons
        assert ReasonType.FORWARD_STALE in c_reasons

    def test_handle_delete_multiple_vars_mixed(self):
        """Deleted cell writes x and y. Downstream reads x, upstream writes y.

        Order: [A, B, C]. A writes y, B writes x and y, C reads x.
        Delete B → A stale (BackwardStale on y), C stale (ForwardStale on x).
        """
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = {"y"}
        state.writes["B"] = {"x", "y"}
        state.reads["C"] = {"x"}
        state.status["A"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("A")
        assert not state.is_clean("C")
        a_reasons = state.get_reasons("A")
        assert any(r.type == ReasonType.BACKWARD_STALE and r.loc == "y" for r in a_reasons)
        c_reasons = state.get_reasons("C")
        assert any(r.type == ReasonType.FORWARD_STALE and r.loc == "x" for r in c_reasons)
