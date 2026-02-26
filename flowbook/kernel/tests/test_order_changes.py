"""Tests for DELETE, INSERT, and MOVE transitions (§2.4-§2.6)."""

import pytest

from flowbook.kernel.models import MovedCell, OrderChangeResult, OrderDelta
from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper


class TestComputeOrderDelta:
    """Tests for _compute_order_delta method."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_no_changes(self):
        """Same order produces empty delta."""
        delta = self.helper.sdc._compute_order_delta(
            ["a", "b", "c", "d"],
            ["a", "b", "c", "d"],
        )
        assert delta.deleted == []
        assert delta.inserted == []
        assert delta.moved == []

    def test_deletion(self):
        """Deleted cells are detected."""
        delta = self.helper.sdc._compute_order_delta(
            ["a", "b", "c", "d"],
            ["a", "c", "d"],
        )
        assert delta.deleted == ["b"]
        assert delta.inserted == []

    def test_insertion(self):
        """Inserted cells are detected."""
        delta = self.helper.sdc._compute_order_delta(
            ["a", "b", "c"],
            ["a", "b", "x", "c"],
        )
        assert delta.deleted == []
        assert delta.inserted == ["x"]

    def test_move_forward(self):
        """Moving cell forward is detected."""
        delta = self.helper.sdc._compute_order_delta(
            ["a", "b", "c", "d"],
            ["a", "c", "d", "b"],  # b moved from pos 1 to pos 3
        )
        assert delta.deleted == []
        assert delta.inserted == []
        # b moved from 1 to 3, c moved from 2 to 1, d moved from 3 to 2
        moved_ids = {m.cell_id for m in delta.moved}
        assert "b" in moved_ids

    def test_move_backward(self):
        """Moving cell backward is detected."""
        delta = self.helper.sdc._compute_order_delta(
            ["a", "b", "c", "d"],
            ["a", "d", "b", "c"],  # d moved from pos 3 to pos 1
        )
        assert delta.deleted == []
        assert delta.inserted == []
        moved_ids = {m.cell_id for m in delta.moved}
        assert "d" in moved_ids


class TestDeleteTransition:
    """Tests for DELETE transition (§2.4)."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_delete_marks_dependent_cells_stale(self):
        """Deleting a cell marks cells that read its writes as stale."""
        # Execute A→B→C where B writes x, C reads x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "b", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Delete B
        result = self.helper.sdc.set_cell_order(["a", "c", "d"])

        assert "c" in result.newly_stale
        assert "b" not in self.helper.sdc.records  # Record pruned

    def test_delete_unexecuted_cell_no_staleness(self):
        """Deleting an unexecuted cell doesn't mark anything stale."""
        # Only execute A
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )

        # Delete B (never executed)
        result = self.helper.sdc.set_cell_order(["a", "c", "d"])

        assert result.newly_stale == []

    def test_delete_cell_no_readers_no_staleness(self):
        """Deleting a cell with no downstream readers doesn't mark anything stale."""
        # Execute A and B where B writes x but nothing reads x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "b", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )

        # Delete B
        result = self.helper.sdc.set_cell_order(["a", "c", "d"])

        assert result.newly_stale == []


class TestInsertTransition:
    """Tests for INSERT transition (§2.5)."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_insert_no_staleness(self):
        """Inserting a new cell doesn't mark anything stale."""
        # Execute A and B
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Insert new cell X
        result = self.helper.sdc.set_cell_order(["a", "x", "b", "c"])

        assert result.newly_stale == []
        assert "x" in result.delta.inserted


class TestMoveForwardTransition:
    """Tests for MOVE forward transition (§2.6, Examples 1-2)."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_move_forward_downstream_loses_dependency(self):
        """Example 1: Moving B forward makes C stale (C read B's write)."""
        # Execute A→B→C→D where B writes x, C reads x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "b", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2},
            reads=set(), writes=set()
        )

        # Move B from pos 1 to pos 3 (after D)
        result = self.helper.sdc.set_cell_order(["a", "c", "d", "b"])

        assert "c" in result.newly_stale
        assert "b" not in result.newly_stale  # B's write is still valid

    def test_move_forward_moved_cell_gains_input(self):
        """Example 2: Moving B forward past C makes B stale (B reads what C writes)."""
        # Execute A→B→C→D where B reads y (initial value), C writes y
        # Note: Use continue_on_violation=True for C because it writes y which B reads
        # (this would normally be a backward violation, but we want to test MOVE logic)
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"y": 0},
            reads=set(), writes={"y"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"y": 0}, post_namespace={"y": 0, "z": 1},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"y": 0, "z": 1}, post_namespace={"y": 99, "z": 1},
            reads=set(), writes={"y"},
            continue_on_violation=True  # Allow record despite backward violation
        )
        self.helper.execute_cell(
            "d", pre_namespace={"y": 99, "z": 1}, post_namespace={"y": 99, "z": 1},
            reads=set(), writes=set()
        )

        # Move B from pos 1 to pos 3 (after D)
        result = self.helper.sdc.set_cell_order(["a", "c", "d", "b"])

        assert "b" in result.newly_stale  # B now reads C's y


class TestMoveBackwardTransition:
    """Tests for MOVE backward transition (§2.6, Examples 3-4)."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_move_backward_forward_contamination(self):
        """Example 3: Moving D backward makes D stale (D reads B's write)."""
        # Execute A→B→C→D where B writes x, D reads x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "b", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1}, post_namespace={"x": 1},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Move D from pos 3 to pos 1 (before B)
        result = self.helper.sdc.set_cell_order(["a", "d", "b", "c"])

        assert "d" in result.newly_stale  # D is now contaminated

    def test_move_backward_downstream_gains_input(self):
        """Example 4: Moving D backward makes B stale (B reads x, D writes x)."""
        # Execute A→B→C→D where A writes x, B reads x, D writes x
        # Note: Use continue_on_violation=True for D because it writes x which B reads
        # (this would normally be a backward violation, but we want to test MOVE logic)
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 99, "y": 2},
            reads=set(), writes={"x"},
            continue_on_violation=True  # Allow record despite backward violation
        )

        # Move D from pos 3 to pos 1 (before B)
        result = self.helper.sdc.set_cell_order(["a", "d", "b", "c"])

        assert "b" in result.newly_stale  # B now reads D's x instead of A's


class TestMoveEdgeCases:
    """Edge case tests for MOVE transition."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_move_unexecuted_cell_no_staleness(self):
        """Moving an unexecuted cell doesn't cause staleness."""
        # Only execute A
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )

        # Move B (never executed)
        result = self.helper.sdc.set_cell_order(["a", "c", "b", "d"])

        # B is in moved but should not cause staleness
        assert result.newly_stale == []

    def test_move_already_stale_not_duplicated(self):
        """Moving a cell that's already stale doesn't add it again."""
        # Execute A and B
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Mark B stale manually
        self.helper.sdc._stale_cells.add("b")

        # Move that could make B stale
        result = self.helper.sdc.set_cell_order(["b", "a", "c", "d"])

        # B should not be in newly_stale since it was already stale
        assert "b" not in result.newly_stale

    def test_move_no_overlap_no_staleness(self):
        """Moving cells with no variable overlap doesn't cause staleness."""
        # Execute A (writes x) and B (writes y, unrelated)
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads=set(), writes={"y"}  # B doesn't read x
        )

        # Move B before A
        result = self.helper.sdc.set_cell_order(["b", "a", "c", "d"])

        assert result.newly_stale == []


class TestOrderChangeResult:
    """Tests for OrderChangeResult data structure."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_result_has_delta(self):
        """OrderChangeResult includes the computed delta."""
        result = self.helper.sdc.set_cell_order(["a", "c", "d"])  # Delete b

        assert isinstance(result, OrderChangeResult)
        assert isinstance(result.delta, OrderDelta)
        assert "b" in result.delta.deleted

    def test_result_has_warnings(self):
        """OrderChangeResult includes warnings for staleness."""
        # Execute cells with dependencies
        self.helper.execute_cell(
            "b", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Delete B
        result = self.helper.sdc.set_cell_order(["a", "c", "d"])

        assert len(result.warnings) > 0
        assert any("stale" in w.lower() for w in result.warnings)


class TestMovedCellProperties:
    """Tests for MovedCell dataclass."""

    def test_moved_forward_property(self):
        """moved_forward is True when new_position > old_position."""
        move = MovedCell(cell_id="x", old_position=1, new_position=3)
        assert move.moved_forward is True
        assert move.moved_backward is False

    def test_moved_backward_property(self):
        """moved_backward is True when new_position < old_position."""
        move = MovedCell(cell_id="x", old_position=3, new_position=1)
        assert move.moved_forward is False
        assert move.moved_backward is True

    def test_same_position_neither(self):
        """Neither property is True when position unchanged."""
        move = MovedCell(cell_id="x", old_position=2, new_position=2)
        assert move.moved_forward is False
        assert move.moved_backward is False
