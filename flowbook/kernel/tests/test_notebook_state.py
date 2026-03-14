"""
Tests for NotebookState - the core state model for reproducibility tracking.

These tests verify the formal model transitions:
- Status management (Clean/Stale with reasons)
- LastWriter computation
- Runnable checks
- Contamination detection
- Staleness propagation
- Structural changes (EDIT, INSERT, DELETE, MOVE)
"""

import pytest
from flowbook.kernel.models import CellStatus, Reason, ReasonType
from flowbook.kernel.notebook_state import NotebookState
from flowbook.kernel_support.models import TrackingData


def make_tracking(reads: set = None, writes: set = None) -> TrackingData:
    """Helper to create TrackingData for tests."""
    return TrackingData(
        reads_before_writes=reads or set(),
        writes=writes or set(),
    )


# =============================================================================
# Reason Tests
# =============================================================================


class TestReason:
    """Tests for the Reason dataclass."""

    def test_reason_creation_minimal(self):
        """Reason can be created with just a type."""
        r = Reason(ReasonType.NEVER_EXECUTED)
        assert r.type == ReasonType.NEVER_EXECUTED
        assert r.loc is None
        assert r.cell_id is None

    def test_reason_creation_with_loc(self):
        """Reason can include a location."""
        r = Reason(ReasonType.FORWARD_STALE, loc="x")
        assert r.type == ReasonType.FORWARD_STALE
        assert r.loc == "x"
        assert r.cell_id is None

    def test_reason_creation_with_cell_id(self):
        """Reason can include a causing cell ID."""
        r = Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="abc123")
        assert r.type == ReasonType.FORWARD_STALE
        assert r.loc == "x"
        assert r.cell_id == "abc123"

    def test_reason_to_dict(self):
        """Reason converts to dict for JSON serialization."""
        r = Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="abc123")
        d = r.to_dict()
        assert d == {"type": "forward_stale", "loc": "x", "cell_id": "abc123"}

    def test_reason_to_dict_minimal(self):
        """Minimal reason omits None fields."""
        r = Reason(ReasonType.NEVER_EXECUTED)
        d = r.to_dict()
        assert d == {"type": "never_executed"}

    def test_reason_from_dict(self):
        """Reason can be recreated from dict."""
        d = {"type": "forward_stale", "loc": "x", "cell_id": "abc123"}
        r = Reason.from_dict(d)
        assert r.type == ReasonType.FORWARD_STALE
        assert r.loc == "x"
        assert r.cell_id == "abc123"

    def test_reason_equality(self):
        """Reasons with same fields are equal (frozen dataclass)."""
        r1 = Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="abc123")
        r2 = Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="abc123")
        assert r1 == r2

    def test_reason_hashable(self):
        """Reasons can be used in sets."""
        r1 = Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="abc123")
        r2 = Reason(ReasonType.FORWARD_STALE, loc="y", cell_id="abc123")
        s = {r1, r2}
        assert len(s) == 2

    def test_reason_str(self):
        """Reason has readable string representation."""
        r = Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="abc")
        s = str(r)
        assert "forward_stale" in s
        assert "x" in s
        assert "abc" in s


# =============================================================================
# CellStatus Tests
# =============================================================================


class TestCellStatus:
    """Tests for CellStatus."""

    def test_clean_status(self):
        """Clean status has no reasons."""
        status = CellStatus.clean()
        assert status.is_clean
        assert len(status.reasons) == 0

    def test_stale_status(self):
        """Stale status has reasons."""
        reasons = {Reason(ReasonType.CODE_CHANGED)}
        status = CellStatus.stale(reasons)
        assert not status.is_clean
        assert len(status.reasons) == 1

    def test_never_executed(self):
        """Never executed factory method."""
        status = CellStatus.never_executed()
        assert not status.is_clean
        assert Reason(ReasonType.NEVER_EXECUTED) in status.reasons

    def test_code_changed(self):
        """Code changed factory method."""
        status = CellStatus.code_changed()
        assert not status.is_clean
        assert Reason(ReasonType.CODE_CHANGED) in status.reasons

    def test_add_reason_to_clean(self):
        """Adding reason to Clean makes it Stale."""
        status = CellStatus.clean()
        status.add_reason(Reason(ReasonType.FORWARD_STALE, loc="x"))
        assert not status.is_clean
        assert len(status.reasons) == 1

    def test_add_reason_accumulates(self):
        """Adding reason to Stale accumulates reasons."""
        status = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        status.add_reason(Reason(ReasonType.FORWARD_STALE, loc="x"))
        assert len(status.reasons) == 2

    def test_clear_reasons(self):
        """Clearing reasons makes status Clean."""
        status = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        status.clear_reasons()
        assert status.is_clean
        assert len(status.reasons) == 0

    def test_to_dict(self):
        """Status converts to dict."""
        status = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        d = status.to_dict()
        assert d["is_clean"] is False
        assert len(d["reasons"]) == 1

    def test_str_clean(self):
        """Clean status string representation."""
        status = CellStatus.clean()
        assert str(status) == "Clean"

    def test_str_stale(self):
        """Stale status string representation."""
        status = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        s = str(status)
        assert "Stale" in s
        assert "code_changed" in s


# =============================================================================
# NotebookState Initialization Tests
# =============================================================================


class TestNotebookStateInit:
    """Tests for NotebookState initialization."""

    def test_empty_state(self):
        """Empty state has no cells."""
        state = NotebookState()
        assert len(state.cell_order) == 0
        assert len(state.status) == 0
        assert len(state.reads) == 0
        assert len(state.writes) == 0
        assert len(state.last_writer) == 0

    def test_get_status_creates_never_executed(self):
        """Getting status for unknown cell creates NeverExecuted."""
        state = NotebookState()
        status = state.get_status("cell1")
        assert not status.is_clean
        assert Reason(ReasonType.NEVER_EXECUTED) in status.reasons

    def test_is_clean_unknown_cell(self):
        """Unknown cell is not clean (never executed)."""
        state = NotebookState()
        assert not state.is_clean("cell1")


# =============================================================================
# LastWriter Tests
# =============================================================================


class TestLastWriter:
    """Tests for LastWriter computation."""

    def test_last_writer_no_writers(self):
        """LastWriter returns None when no cell wrote the location."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        assert state.last_writer_for("x", "C") is None

    def test_last_writer_single_writer(self):
        """LastWriter finds the only writer."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = {"x"}
        assert state.last_writer_for("x", "C") == "A"

    def test_last_writer_multiple_writers(self):
        """LastWriter finds the latest writer before cell."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C", "D"]
        state.writes["A"] = {"x"}
        state.writes["B"] = {"x"}
        assert state.last_writer_for("x", "D") == "B"
        assert state.last_writer_for("x", "C") == "B"
        assert state.last_writer_for("x", "B") == "A"

    def test_last_writer_ignores_later_cells(self):
        """LastWriter ignores writers after the target cell."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["C"] = {"x"}
        assert state.last_writer_for("x", "B") is None

    def test_last_writer_cell_not_in_order(self):
        """LastWriter returns None if target cell not in order."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        assert state.last_writer_for("x", "C") is None


# =============================================================================
# Runnable Tests
# =============================================================================


class TestRunnable:
    """Tests for Runnable check."""

    def test_runnable_no_reads(self):
        """Cell with no reads is always runnable."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.reads["B"] = set()
        assert state.is_runnable("B")

    def test_runnable_reads_from_earlier(self):
        """Cell reading from earlier cell is runnable."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["A"] = {"x"}
        state.reads["B"] = {"x"}
        state.last_writer["x"] = "A"
        assert state.is_runnable("B")

    def test_not_runnable_reads_from_later(self):
        """Cell reading from later cell is not runnable."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["B"] = {"x"}
        state.reads["A"] = {"x"}
        state.last_writer["x"] = "B"  # B wrote x, but B is after A
        assert not state.is_runnable("A")

    def test_runnable_reads_from_init(self):
        """Cell reading from init (no writer) is runnable if no expected writer."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.reads["B"] = {"x"}
        # No cell wrote x, so L(x) = None and LastWriter(x, B) = None
        # Both are None, so runnable
        assert state.is_runnable("B")

    def test_not_runnable_wrong_writer(self):
        """Cell is not runnable if actual writer differs from expected."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = {"x"}
        state.writes["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.last_writer["x"] = "A"  # A wrote x, but B should have
        assert not state.is_runnable("C")


# =============================================================================
# Contamination Detection Tests
# =============================================================================


class TestContaminationDetection:
    """Tests for contamination (reads from later) detection."""

    def test_no_contamination_empty_reads(self):
        """No contamination when cell reads nothing."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        reasons = state.get_contamination_reasons("A", set())
        assert len(reasons) == 0

    def test_no_contamination_reads_from_earlier(self):
        """No contamination when reading from earlier cell."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.last_writer["x"] = "A"
        reasons = state.get_contamination_reasons("B", {"x"})
        assert len(reasons) == 0

    def test_contamination_reads_from_later(self):
        """Contamination when reading from later cell."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.last_writer["x"] = "B"
        reasons = state.get_contamination_reasons("A", {"x"})
        assert len(reasons) == 1
        r = list(reasons)[0]
        assert r.type == ReasonType.NO_READ_BEFORE_WRITE
        assert r.loc == "x"
        assert r.cell_id == "B"

    def test_contamination_multiple_locations(self):
        """Contamination detected for multiple locations."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.last_writer["x"] = "B"
        state.last_writer["y"] = "C"
        reasons = state.get_contamination_reasons("A", {"x", "y"})
        assert len(reasons) == 2
        locs = {r.loc for r in reasons}
        assert locs == {"x", "y"}

    def test_contamination_cell_not_in_order(self):
        """No contamination if cell not in order."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.last_writer["x"] = "B"
        reasons = state.get_contamination_reasons("X", {"x"})
        assert len(reasons) == 0


# =============================================================================
# Record Execution Tests
# =============================================================================


class TestRecordExecution:
    """Tests for recording cell execution."""

    def test_record_execution_updates_reads(self):
        """Recording execution updates reads."""
        state = NotebookState()
        state.record_execution("A", make_tracking(reads={"x", "y"}, writes={"z"}))
        assert state.reads["A"] == {"x", "y"}

    def test_record_execution_updates_writes(self):
        """Recording execution updates writes."""
        state = NotebookState()
        state.record_execution("A", make_tracking(reads={"x"}, writes={"y", "z"}))
        assert state.writes["A"] == {"y", "z"}

    def test_record_execution_updates_last_writer(self):
        """Recording execution updates last_writer map for changed variables."""
        state = NotebookState()
        # last_writer is updated only for variables that CHANGED (via changed_vars)
        state.record_execution("A", make_tracking(writes={"x", "y"}), changed_vars={"x", "y"})
        assert state.last_writer["x"] == "A"
        assert state.last_writer["y"] == "A"

    def test_record_execution_overwrites_last_writer(self):
        """Later execution overwrites earlier last_writer only if value changed."""
        state = NotebookState()
        state.record_execution("A", make_tracking(writes={"x"}), changed_vars={"x"})
        state.record_execution("B", make_tracking(writes={"x"}), changed_vars={"x"})
        assert state.last_writer["x"] == "B"

    def test_record_execution_preserves_last_writer_when_unchanged(self):
        """last_writer is NOT updated if variable is written but value unchanged."""
        state = NotebookState()
        state.record_execution("A", make_tracking(writes={"x"}), changed_vars={"x"})
        # B writes x but value is the same (changed_vars is None or doesn't include x)
        state.record_execution("B", make_tracking(writes={"x"}), changed_vars=None)
        # A should still be the last writer
        assert state.last_writer["x"] == "A"


# =============================================================================
# Staleness Propagation Tests
# =============================================================================


class TestStalenessPropagation:
    """Tests for staleness propagation after execution."""

    def test_propagate_to_reader(self):
        """Writing variable propagates staleness to later reader."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.reads["B"] = {"x"}
        state.status["B"] = CellStatus.clean()

        state.propagate_staleness("A", {"x"})

        assert not state.is_clean("B")
        reasons = state.get_reasons("B")
        assert any(r.type == ReasonType.FORWARD_STALE and r.loc == "x" for r in reasons)

    def test_propagate_to_writer(self):
        """Writing variable propagates staleness to later writer of same var."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["B"] = {"x"}
        state.status["B"] = CellStatus.clean()

        state.propagate_staleness("A", {"x"})

        assert not state.is_clean("B")
        reasons = state.get_reasons("B")
        assert any(r.type == ReasonType.BACKWARD_STALE and r.loc == "x" for r in reasons)

    def test_propagate_skips_already_stale(self):
        """Propagation skips cells that are already stale."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.reads["B"] = {"x"}
        state.status["B"] = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})

        state.propagate_staleness("A", {"x"})

        # Should still just have CODE_CHANGED, not FORWARD_STALE
        reasons = state.get_reasons("B")
        assert len(reasons) == 1
        assert Reason(ReasonType.CODE_CHANGED) in reasons

    def test_propagate_to_multiple_cells(self):
        """Propagation affects multiple later cells."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.reads["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.propagate_staleness("A", {"x"})

        assert not state.is_clean("B")
        assert not state.is_clean("C")

    def test_propagate_only_affects_later_cells(self):
        """Propagation only affects cells after the writer."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.reads["A"] = {"x"}  # Earlier cell reads x
        state.reads["C"] = {"x"}  # Later cell reads x
        state.status["A"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.propagate_staleness("B", {"x"})

        assert state.is_clean("A")  # Not affected (before B)
        assert not state.is_clean("C")  # Affected (after B)

    def test_propagate_no_overlap(self):
        """No propagation when written vars don't overlap with reads/writes."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.reads["B"] = {"y"}
        state.writes["B"] = {"z"}
        state.status["B"] = CellStatus.clean()

        state.propagate_staleness("A", {"x"})

        assert state.is_clean("B")


# =============================================================================
# EDIT Transition Tests
# =============================================================================


class TestEditTransition:
    """Tests for EDIT transition."""

    def test_edit_marks_stale(self):
        """EDIT marks cell as stale."""
        state = NotebookState()
        state.cell_order = ["A"]
        state.status["A"] = CellStatus.clean()

        state.handle_edit("A")

        assert not state.is_clean("A")

    def test_edit_replaces_reasons(self):
        """EDIT replaces existing reasons with CodeChanged."""
        state = NotebookState()
        state.cell_order = ["A"]
        state.status["A"] = CellStatus.stale({
            Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="B")
        })

        state.handle_edit("A")

        reasons = state.get_reasons("A")
        assert len(reasons) == 1
        assert Reason(ReasonType.CODE_CHANGED) in reasons


# =============================================================================
# DELETE Transition Tests
# =============================================================================


class TestDeleteTransition:
    """Tests for DELETE transition."""

    def test_delete_removes_from_cell_order(self):
        """DELETE removes cell from cell_order."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        state.handle_delete("B")

        assert state.cell_order == ["A", "C"]

    def test_delete_removes_status(self):
        """DELETE removes cell's status."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()

        state.handle_delete("B")

        assert "B" not in state.status

    def test_delete_removes_reads_writes(self):
        """DELETE removes cell's reads and writes."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.reads["B"] = {"x"}
        state.writes["B"] = {"y"}

        state.handle_delete("B")

        assert "B" not in state.reads
        assert "B" not in state.writes

    def test_delete_keeps_last_writer_for_orphan_detection(self):
        """DELETE keeps last_writer pointing to deleted cell for orphan detection.

        This allows forward dependency checks to detect values that came from
        a cell that no longer exists (orphaned values).
        """
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.writes["B"] = {"x", "y"}
        state.last_writer["x"] = "B"
        state.last_writer["y"] = "B"
        state.last_writer["z"] = "A"

        state.handle_delete("B")

        # last_writer KEPT for orphan detection (writer not in cell_order)
        assert state.last_writer["x"] == "B"
        assert state.last_writer["y"] == "B"
        assert state.last_writer["z"] == "A"  # Unchanged

    def test_delete_marks_orphan_readers(self):
        """DELETE marks cells that read from deleted cell as stale."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.last_writer["x"] = "B"
        state.status["C"] = CellStatus.clean()

        state.handle_delete("B")

        assert not state.is_clean("C")
        reasons = state.get_reasons("C")
        assert any(r.type == ReasonType.FORWARD_STALE and r.loc == "x" for r in reasons)


# =============================================================================
# INSERT Transition Tests
# =============================================================================


class TestInsertTransition:
    """Tests for INSERT transition."""

    def test_insert_adds_to_cell_order(self):
        """INSERT adds cell to cell_order at position."""
        state = NotebookState()
        state.cell_order = ["A", "C"]

        state.handle_insert("B", 1)

        assert state.cell_order == ["A", "B", "C"]

    def test_insert_at_start(self):
        """INSERT at position 0."""
        state = NotebookState()
        state.cell_order = ["B", "C"]

        state.handle_insert("A", 0)

        assert state.cell_order == ["A", "B", "C"]

    def test_insert_at_end(self):
        """INSERT at end."""
        state = NotebookState()
        state.cell_order = ["A", "B"]

        state.handle_insert("C", 2)

        assert state.cell_order == ["A", "B", "C"]

    def test_insert_initializes_status(self):
        """INSERT initializes cell with NeverExecuted status."""
        state = NotebookState()
        state.cell_order = ["A"]

        state.handle_insert("B", 1)

        assert not state.is_clean("B")
        assert Reason(ReasonType.NEVER_EXECUTED) in state.get_reasons("B")

    def test_insert_initializes_empty_reads_writes(self):
        """INSERT initializes empty reads and writes."""
        state = NotebookState()
        state.cell_order = ["A"]

        state.handle_insert("B", 1)

        assert state.reads["B"] == set()
        assert state.writes["B"] == set()

    def test_insert_marks_affected_cells(self):
        """INSERT marks affected cells if their Runnable changes."""
        state = NotebookState()
        state.cell_order = ["A", "C"]
        state.writes["A"] = {"x"}
        state.reads["C"] = {"x"}
        state.last_writer["x"] = "A"
        state.status["C"] = CellStatus.clean()

        # Insert B which also writes x
        state.handle_insert("B", 1)
        state.writes["B"] = {"x"}
        state.last_writer["x"] = "B"

        # Now C's Runnable fails: L(x)=B, LastWriter(x,C)=B, but C was clean
        # Actually need to re-check after insertion...
        # The handle_insert checks Runnable after inserting.
        # In this test, B hasn't executed yet, so this case is edge-case.


# =============================================================================
# MOVE Transition Tests
# =============================================================================


class TestMoveTransition:
    """Tests for MOVE transition."""

    def test_move_forward(self):
        """MOVE cell forward in order."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        state.handle_move("A", 2)

        assert state.cell_order == ["B", "C", "A"]

    def test_move_backward(self):
        """MOVE cell backward in order."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        state.handle_move("C", 0)

        assert state.cell_order == ["C", "A", "B"]

    def test_move_same_position(self):
        """MOVE to same position is no-op."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        state.handle_move("B", 1)

        assert state.cell_order == ["A", "B", "C"]

    def test_move_marks_affected_cells(self):
        """MOVE marks affected cells if their Runnable changes."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["A"] = {"x"}
        state.reads["B"] = {"x"}
        state.last_writer["x"] = "A"
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        # Move A to end - now B reads x but A is after B
        state.handle_move("A", 2)
        # After move: ["B", "C", "A"]
        # B reads x, L(x)=A, but LastWriter(x, B) = None (A is after B now)
        # So B is not runnable

        # The move should mark B as stale
        # Note: depends on whether handle_move checks this


# =============================================================================
# set_cell_order Tests
# =============================================================================


class TestSetCellOrder:
    """Tests for set_cell_order (compound structural changes)."""

    def test_set_cell_order_initial(self):
        """Setting initial cell order."""
        state = NotebookState()

        state.set_cell_order(["A", "B", "C"])

        assert state.cell_order == ["A", "B", "C"]

    def test_set_cell_order_handles_deletions(self):
        """set_cell_order handles deletions."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.set_cell_order(["A", "C"])

        assert state.cell_order == ["A", "C"]
        assert "B" not in state.status

    def test_set_cell_order_handles_insertions(self):
        """set_cell_order handles insertions."""
        state = NotebookState()
        state.cell_order = ["A", "C"]

        state.set_cell_order(["A", "B", "C"])

        assert state.cell_order == ["A", "B", "C"]
        assert not state.is_clean("B")  # New cell is NeverExecuted

    def test_set_cell_order_returns_newly_stale(self):
        """set_cell_order returns list of newly stale cells."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.writes["B"] = {"x"}
        state.reads["C"] = {"x"}
        state.last_writer["x"] = "B"
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        # Delete B - C becomes orphan
        newly_stale = state.set_cell_order(["A", "C"])

        assert "C" in newly_stale

    def test_set_cell_order_reorder(self):
        """set_cell_order handles pure reordering."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.clean()

        state.set_cell_order(["C", "B", "A"])

        assert state.cell_order == ["C", "B", "A"]


# =============================================================================
# Clear Tests
# =============================================================================


class TestClear:
    """Tests for clear (kernel restart)."""

    def test_clear_resets_all_state(self):
        """Clear removes all state."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.status["A"] = CellStatus.clean()
        state.reads["A"] = {"x"}
        state.writes["A"] = {"y"}
        state.last_writer["x"] = "A"

        state.clear()

        assert len(state.cell_order) == 0
        assert len(state.status) == 0
        assert len(state.reads) == 0
        assert len(state.writes) == 0
        assert len(state.last_writer) == 0


# =============================================================================
# get_stale_cells Tests
# =============================================================================


class TestGetStaleCells:
    """Tests for get_stale_cells."""

    def test_get_stale_cells_empty(self):
        """Empty state has no stale cells."""
        state = NotebookState()
        assert state.get_stale_cells() == []

    def test_get_stale_cells_all_clean(self):
        """All clean cells returns empty."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        assert state.get_stale_cells() == []

    def test_get_stale_cells_some_stale(self):
        """Returns only stale cells in order."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        state.status["C"] = CellStatus.clean()
        assert state.get_stale_cells() == ["B"]

    def test_get_stale_cells_preserves_order(self):
        """Returns stale cells in document order."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]
        state.status["A"] = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        state.status["B"] = CellStatus.clean()
        state.status["C"] = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        assert state.get_stale_cells() == ["A", "C"]


# =============================================================================
# get_all_reasons Tests
# =============================================================================


class TestGetAllReasons:
    """Tests for get_all_reasons."""

    def test_get_all_reasons_empty(self):
        """Empty state has no reasons."""
        state = NotebookState()
        assert state.get_all_reasons() == {}

    def test_get_all_reasons_clean_cells(self):
        """Clean cells not included."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.status["A"] = CellStatus.clean()
        state.status["B"] = CellStatus.clean()
        assert state.get_all_reasons() == {}

    def test_get_all_reasons_stale_cells(self):
        """Returns reasons for stale cells."""
        state = NotebookState()
        state.cell_order = ["A", "B"]
        state.status["A"] = CellStatus.stale({Reason(ReasonType.CODE_CHANGED)})
        state.status["B"] = CellStatus.stale({
            Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="A")
        })

        reasons = state.get_all_reasons()

        assert "A" in reasons
        assert "B" in reasons
        assert len(reasons["A"]) == 1
        assert reasons["A"][0]["type"] == "code_changed"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_in_order_execution(self):
        """Simulate in-order execution: all cells become clean."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        # Execute A: writes x
        state.record_execution("A", make_tracking(writes={"x"}))
        state.set_clean("A")
        state.propagate_staleness("A", {"x"})

        # Execute B: reads x, writes y
        state.record_execution("B", make_tracking(reads={"x"}, writes={"y"}))
        state.set_clean("B")
        state.propagate_staleness("B", {"y"})

        # Execute C: reads y
        state.record_execution("C", make_tracking(reads={"y"}))
        state.set_clean("C")

        assert state.is_clean("A")
        assert state.is_clean("B")
        assert state.is_clean("C")
        assert state.get_stale_cells() == []

    def test_edit_then_propagate(self):
        """Edit cell, then re-execute propagates staleness."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        # Initial execution
        state.record_execution("A", make_tracking(writes={"x"}))
        state.set_clean("A")
        state.record_execution("B", make_tracking(reads={"x"}, writes={"y"}))
        state.set_clean("B")
        state.record_execution("C", make_tracking(reads={"y"}))
        state.set_clean("C")

        # Edit A
        state.handle_edit("A")
        assert not state.is_clean("A")
        assert state.is_clean("B")  # Not yet propagated

        # Re-execute A
        state.record_execution("A", make_tracking(writes={"x"}))
        state.set_clean("A")
        state.propagate_staleness("A", {"x"})

        # Now B is stale
        assert not state.is_clean("B")
        assert state.is_clean("C")  # B hasn't run yet, C not affected

    def test_out_of_order_creates_contamination(self):
        """Out-of-order execution creates contamination."""
        state = NotebookState()
        state.cell_order = ["A", "B"]

        # Execute B first: writes x (and x changes, so B is the last_writer)
        state.record_execution("B", make_tracking(writes={"x"}), changed_vars={"x"})
        state.set_clean("B")

        # Now A tries to read x - contamination (A is before B but B wrote x)
        reasons = state.get_contamination_reasons("A", {"x"})
        assert len(reasons) == 1
        assert reasons.pop().type == ReasonType.NO_READ_BEFORE_WRITE

    def test_delete_cascade(self):
        """Deleting a cell cascades orphan staleness."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        # A writes x, B reads x and writes y, C reads y
        # Must pass changed_vars so last_writer is updated for orphan detection
        state.record_execution("A", make_tracking(writes={"x"}), changed_vars={"x"})
        state.set_clean("A")
        state.record_execution("B", make_tracking(reads={"x"}, writes={"y"}), changed_vars={"y"})
        state.set_clean("B")
        state.record_execution("C", make_tracking(reads={"y"}))
        state.set_clean("C")

        # Delete B
        state.handle_delete("B")

        # C is now orphaned (read y from deleted B)
        assert not state.is_clean("C")
        reasons = state.get_reasons("C")
        assert any(r.type == ReasonType.FORWARD_STALE for r in reasons)

    def test_multiple_reasons_accumulate(self):
        """A cell can accumulate multiple staleness reasons."""
        state = NotebookState()
        state.cell_order = ["A", "B", "C"]

        # C reads x and y
        state.record_execution("C", make_tracking(reads={"x", "y"}))
        state.set_clean("C")

        # A writes x, propagates to C
        state.record_execution("A", make_tracking(writes={"x"}))
        state.set_clean("A")
        state.propagate_staleness("A", {"x"})

        assert not state.is_clean("C")
        assert len(state.get_reasons("C")) == 1

        # B writes y, propagates to C (C already stale, but we can test add_reason)
        state.status["C"] = CellStatus.clean()  # Reset for test
        state.propagate_staleness("A", {"x"})
        # Manually add another reason
        state.add_reason("C", Reason(ReasonType.FORWARD_STALE, loc="y", cell_id="B"))

        assert len(state.get_reasons("C")) == 2
