"""Complex tests for DELETE, INSERT, and MOVE transitions.

Tests cover:
- Block moves (moving multiple adjacent cells as a group)
- Copy/paste scenarios (new cells from duplication)
- Complex reorderings (shuffles, reversals)
- Cascade effects from multiple dependencies
- Real-world notebook editing patterns
"""

import pytest

from flowbook.kernel.models import MovedCell, OrderChangeResult, OrderDelta
from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper


class TestBlockMoves:
    """Tests for moving multiple cells as a block."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d", "e", "f"])

    def test_move_block_forward(self):
        """Move cells B,C as a block forward past D,E."""
        # Setup: A→B→C→D→E→F
        # A writes x, B reads x and writes y, C reads y and writes z
        # D reads z, E reads nothing, F reads nothing
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads={"z"}, writes={"w"}
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "f", pre_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads=set(), writes=set()
        )

        # Move B,C block after E: A→D→E→B→C→F
        result = self.helper.sdc.set_cell_order(["a", "d", "e", "b", "c", "f"])

        # D should be stale: it read z from C, but C is now after D
        assert "d" in result.newly_stale

    def test_move_block_backward(self):
        """Move cells D,E as a block backward before B."""
        # Setup: A→B→C→D→E→F where D writes to var that B reads
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
            continue_on_violation=True  # D writes x which was read by... nobody after A
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 99, "y": 2}, post_namespace={"x": 99, "y": 2, "z": 3},
            reads={"x"}, writes={"z"}
        )
        self.helper.execute_cell(
            "f", pre_namespace={"x": 99, "y": 2, "z": 3},
            post_namespace={"x": 99, "y": 2, "z": 3},
            reads=set(), writes=set()
        )

        # Move D,E block before B: A→D→E→B→C→F
        result = self.helper.sdc.set_cell_order(["a", "d", "e", "b", "c", "f"])

        # B should be stale: D now precedes B and writes x, B reads x
        assert "b" in result.newly_stale

    def test_swap_adjacent_blocks(self):
        """Swap two adjacent blocks: [B,C] <-> [D,E]."""
        # Setup dependency chain
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads={"z"}, writes={"w"}
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4, "v": 5},
            reads={"w"}, writes={"v"}
        )
        self.helper.execute_cell(
            "f", pre_namespace={"x": 1, "y": 2, "z": 3, "w": 4, "v": 5},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4, "v": 5},
            reads=set(), writes=set()
        )

        # Swap: A→B→C→D→E→F becomes A→D→E→B→C→F
        result = self.helper.sdc.set_cell_order(["a", "d", "e", "b", "c", "f"])

        # D read z from C, now D is before C → D is stale (forward contamination)
        assert "d" in result.newly_stale
        # E read w from D, D is still before E → E might be OK
        # But E also moved, and D is stale, so chain effect


class TestCopyPasteScenarios:
    """Tests simulating copy/paste operations.

    When a cell is copy/pasted:
    - A new cell ID is generated (the pasted cell)
    - The pasted cell has no execution record
    - It appears in delta.inserted
    - Existing cells may need to be marked stale based on new order
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d"])

    def test_paste_new_cell_at_end(self):
        """Paste a new cell at the end - no staleness."""
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Paste new cell "e" at end
        result = self.helper.sdc.set_cell_order(["a", "b", "c", "d", "e"])

        assert "e" in result.delta.inserted
        assert result.newly_stale == []  # No staleness from insertion at end

    def test_paste_new_cell_in_middle(self):
        """Paste a new cell in the middle - existing cells shift but no data dependency."""
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )

        # Paste new cell "e" between A and B
        result = self.helper.sdc.set_cell_order(["a", "e", "b", "c", "d"])

        assert "e" in result.delta.inserted
        # B and C shifted but their data dependencies are intact
        # B still reads x from A, C still reads y from B
        assert result.newly_stale == []

    def test_paste_duplicate_creates_new_unexecuted_cell(self):
        """Duplicating a cell creates a new cell with no execution history."""
        # Execute A which writes x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        # Execute B which reads x
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # User duplicates cell A (gets new ID "a_copy")
        # New cell has same code but no execution record
        result = self.helper.sdc.set_cell_order(["a", "a_copy", "b", "c", "d"])

        assert "a_copy" in result.delta.inserted
        assert not self.helper.sdc._notebook_state.has_record("a_copy")  # No execution record
        # No staleness - the new cell hasn't run yet
        assert result.newly_stale == []

    def test_cut_paste_moves_cell(self):
        """Cut and paste is equivalent to a move."""
        # Setup chain: A writes x, B reads x writes y, C reads y
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads=set(), writes=set()
        )

        # Cut B, paste after C (B moves from position 1 to position 2)
        result = self.helper.sdc.set_cell_order(["a", "c", "b", "d"])

        # C read y from B, now B is after C → C is stale
        assert "c" in result.newly_stale

    def test_paste_over_selection_deletes_and_inserts(self):
        """Pasting over a selection: delete selected cells, insert new cell."""
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2}, post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads={"z"}, writes={"w"}
        )

        # Select B,C and paste new cell "e" over them
        # This deletes B,C and inserts E in their place
        result = self.helper.sdc.set_cell_order(["a", "e", "d"])

        assert "b" in result.delta.deleted
        assert "c" in result.delta.deleted
        assert "e" in result.delta.inserted
        # D read z from C, C is deleted → D is stale
        assert "d" in result.newly_stale


class TestComplexReorderings:
    """Tests for complex reordering scenarios."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d", "e"])

    def test_reverse_order(self):
        """Reverse the entire notebook order."""
        # Chain: A→B→C→D→E where each reads from previous
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"v1": 1},
            reads=set(), writes={"v1"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"v1": 1}, post_namespace={"v1": 1, "v2": 2},
            reads={"v1"}, writes={"v2"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"v1": 1, "v2": 2},
            post_namespace={"v1": 1, "v2": 2, "v3": 3},
            reads={"v2"}, writes={"v3"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"v1": 1, "v2": 2, "v3": 3},
            post_namespace={"v1": 1, "v2": 2, "v3": 3, "v4": 4},
            reads={"v3"}, writes={"v4"}
        )
        self.helper.execute_cell(
            "e", pre_namespace={"v1": 1, "v2": 2, "v3": 3, "v4": 4},
            post_namespace={"v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5},
            reads={"v4"}, writes={"v5"}
        )

        # Reverse: E→D→C→B→A
        result = self.helper.sdc.set_cell_order(["e", "d", "c", "b", "a"])

        # Every cell except E should be stale (forward contamination)
        # E now reads v4 but D (which writes v4) is after E
        assert "e" in result.newly_stale
        # D reads v3 but C is after D
        assert "d" in result.newly_stale
        # C reads v2 but B is after C
        assert "c" in result.newly_stale
        # B reads v1 but A is after B
        assert "b" in result.newly_stale

    def test_shuffle_with_dependencies(self):
        """Shuffle cells that have complex dependencies."""
        # A writes x, B writes y, C reads x and y
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads=set(), writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"x", "y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads=set(), writes=set()
        )

        # Move C before A: C→A→B→D→E
        result = self.helper.sdc.set_cell_order(["c", "a", "b", "d", "e"])

        # C reads x and y, but A and B (which write them) are now after C
        assert "c" in result.newly_stale

    def test_interleave_cells(self):
        """Interleave cells from two groups."""
        # Original: A→B→C→D→E (odd positions: A,C,E write; even: B,D read)
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "bval": 1},
            reads={"x"}, writes={"bval"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "bval": 1},
            post_namespace={"x": 1, "bval": 1, "y": 2},
            reads=set(), writes={"y"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "bval": 1, "y": 2},
            post_namespace={"x": 1, "bval": 1, "y": 2, "dval": 2},
            reads={"y"}, writes={"dval"}
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "bval": 1, "y": 2, "dval": 2},
            post_namespace={"x": 1, "bval": 1, "y": 2, "dval": 2, "z": 3},
            reads=set(), writes={"z"}
        )

        # Reorder to: A→C→E→B→D (writers first, then readers)
        result = self.helper.sdc.set_cell_order(["a", "c", "e", "b", "d"])

        # B reads x, A still before B → B OK
        # D reads y, C still before D → D OK
        # Actually both B and D should be OK since their dependencies are preserved
        # But wait - B moved from position 1 to 3, and C moved from 2 to 1
        # B's dependency on A is still satisfied (A at 0, B at 3)
        # D's dependency on C is still satisfied (C at 1, D at 4)
        # So neither should be stale!
        assert "b" not in result.newly_stale
        assert "d" not in result.newly_stale


class TestCascadeEffects:
    """Tests for cascade effects when moves affect multiple cells."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d", "e"])

    def test_move_affects_multiple_downstream(self):
        """Moving one cell makes multiple downstream cells stale."""
        # A writes x, B, C, D all read x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "b_out": 1},
            reads={"x"}, writes={"b_out"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "b_out": 1},
            post_namespace={"x": 1, "b_out": 1, "c_out": 2},
            reads={"x"}, writes={"c_out"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "b_out": 1, "c_out": 2},
            post_namespace={"x": 1, "b_out": 1, "c_out": 2, "d_out": 3},
            reads={"x"}, writes={"d_out"}
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "b_out": 1, "c_out": 2, "d_out": 3},
            post_namespace={"x": 1, "b_out": 1, "c_out": 2, "d_out": 3},
            reads=set(), writes=set()
        )

        # Move A to end: B→C→D→E→A
        result = self.helper.sdc.set_cell_order(["b", "c", "d", "e", "a"])

        # B, C, D all read x from A, but A is now at the end
        assert "b" in result.newly_stale
        assert "c" in result.newly_stale
        assert "d" in result.newly_stale

    def test_delete_affects_multiple_downstream(self):
        """Deleting one cell makes multiple downstream cells stale."""
        # A writes x, B, C, D all read x
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "b_out": 1},
            reads={"x"}, writes={"b_out"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "b_out": 1},
            post_namespace={"x": 1, "b_out": 1, "c_out": 2},
            reads={"x"}, writes={"c_out"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "b_out": 1, "c_out": 2},
            post_namespace={"x": 1, "b_out": 1, "c_out": 2, "d_out": 3},
            reads={"x"}, writes={"d_out"}
        )

        # Delete A
        result = self.helper.sdc.set_cell_order(["b", "c", "d", "e"])

        # B, C, D all read x which was written by deleted A
        assert "b" in result.newly_stale
        assert "c" in result.newly_stale
        assert "d" in result.newly_stale


class TestMultipleDeletesAndInserts:
    """Tests for multiple simultaneous deletions and insertions."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "d", "e"])

    def test_delete_multiple_cells(self):
        """Delete multiple cells at once."""
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads={"z"}, writes={"w"}
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            post_namespace={"x": 1, "y": 2, "z": 3, "w": 4},
            reads={"w"}, writes=set()
        )

        # Delete B and C
        result = self.helper.sdc.set_cell_order(["a", "d", "e"])

        assert "b" in result.delta.deleted
        assert "c" in result.delta.deleted
        # D read z from C (deleted) → D stale
        assert "d" in result.newly_stale
        # E read w from D, D is still there but stale

    def test_insert_multiple_cells(self):
        """Insert multiple cells at once."""
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )

        # Insert X, Y, Z between A and B
        result = self.helper.sdc.set_cell_order(["a", "x", "y", "z", "b", "c", "d", "e"])

        assert set(result.delta.inserted) == {"x", "y", "z"}
        # No staleness from insertions alone
        assert result.newly_stale == []

    def test_simultaneous_delete_insert_move(self):
        """Complex operation: delete some, insert some, move some."""
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )
        self.helper.execute_cell(
            "d", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads=set(), writes=set()
        )
        self.helper.execute_cell(
            "e", pre_namespace={"x": 1, "y": 2, "z": 3},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads=set(), writes=set()
        )

        # Delete B, insert X, move C to end: A→X→D→E→C
        result = self.helper.sdc.set_cell_order(["a", "x", "d", "e", "c"])

        assert "b" in result.delta.deleted
        assert "x" in result.delta.inserted
        # C read y from B, B is deleted → C stale
        assert "c" in result.newly_stale


class TestEdgeCasesAndBoundaries:
    """Edge cases and boundary conditions."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_single_cell_notebook(self):
        """Operations on a single-cell notebook."""
        self.helper.set_cell_order(["a"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )

        # Add a cell
        result = self.helper.sdc.set_cell_order(["a", "b"])
        assert result.delta.inserted == ["b"]
        assert result.newly_stale == []

        # Remove original
        result = self.helper.sdc.set_cell_order(["b"])
        assert result.delta.deleted == ["a"]

    def test_empty_to_non_empty(self):
        """Add cells to empty notebook."""
        self.helper.set_cell_order([])

        result = self.helper.sdc.set_cell_order(["a", "b", "c"])

        assert set(result.delta.inserted) == {"a", "b", "c"}
        assert result.newly_stale == []

    def test_non_empty_to_empty(self):
        """Remove all cells from notebook."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )

        result = self.helper.sdc.set_cell_order([])

        assert "a" in result.delta.deleted
        # No staleness since no cells left

    def test_move_first_cell_to_end(self):
        """Move first cell to end."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"}
        )

        result = self.helper.sdc.set_cell_order(["b", "c", "a"])

        # B read x from A, A is now at end → B stale
        assert "b" in result.newly_stale

    def test_move_last_cell_to_beginning(self):
        """Move last cell to beginning."""
        self.helper.set_cell_order(["a", "b", "c"])
        self.helper.execute_cell(
            "a", pre_namespace={}, post_namespace={"x": 1},
            reads=set(), writes={"x"}
        )
        self.helper.execute_cell(
            "b", pre_namespace={"x": 1}, post_namespace={"x": 1, "y": 2},
            reads={"x"}, writes={"y"}
        )
        self.helper.execute_cell(
            "c", pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"x", "y"}, writes={"z"}
        )

        result = self.helper.sdc.set_cell_order(["c", "a", "b"])

        # C read x and y, both A and B are now after C → C stale
        assert "c" in result.newly_stale


class TestRealWorldPatterns:
    """Tests mimicking real-world notebook editing patterns."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["imports", "data", "clean", "model", "eval"])

    def test_add_data_exploration_cell(self):
        """Add a data exploration cell between loading and cleaning."""
        # Typical notebook: imports → load data → clean → model → evaluate
        self.helper.execute_cell(
            "imports", pre_namespace={}, post_namespace={"pd": "pandas"},
            reads=set(), writes={"pd"}
        )
        self.helper.execute_cell(
            "data", pre_namespace={"pd": "pandas"},
            post_namespace={"pd": "pandas", "df": "dataframe"},
            reads={"pd"}, writes={"df"}
        )
        self.helper.execute_cell(
            "clean", pre_namespace={"pd": "pandas", "df": "dataframe"},
            post_namespace={"pd": "pandas", "df": "clean_df"},
            reads={"df"}, writes={"df"}
        )

        # Insert exploration cell between data and clean
        result = self.helper.sdc.set_cell_order(
            ["imports", "data", "explore", "clean", "model", "eval"]
        )

        assert "explore" in result.delta.inserted
        # Clean still depends on df from data, which is still before clean
        assert "clean" not in result.newly_stale

    def test_reorder_model_before_clean(self):
        """Mistakenly reorder model cell before data cleaning."""
        self.helper.execute_cell(
            "imports", pre_namespace={}, post_namespace={"pd": "pandas", "sklearn": "sklearn"},
            reads=set(), writes={"pd", "sklearn"}
        )
        self.helper.execute_cell(
            "data", pre_namespace={"pd": "pandas", "sklearn": "sklearn"},
            post_namespace={"pd": "pandas", "sklearn": "sklearn", "df": "raw_df"},
            reads={"pd"}, writes={"df"}
        )
        self.helper.execute_cell(
            "clean", pre_namespace={"pd": "pandas", "sklearn": "sklearn", "df": "raw_df"},
            post_namespace={"pd": "pandas", "sklearn": "sklearn", "df": "clean_df"},
            reads={"df"}, writes={"df"}
        )
        self.helper.execute_cell(
            "model", pre_namespace={"pd": "pandas", "sklearn": "sklearn", "df": "clean_df"},
            post_namespace={"pd": "pandas", "sklearn": "sklearn", "df": "clean_df", "model": "trained"},
            reads={"df", "sklearn"}, writes={"model"}
        )

        # Accidentally move model before clean
        result = self.helper.sdc.set_cell_order(
            ["imports", "data", "model", "clean", "eval"]
        )

        # Model reads df, clean writes df, clean is now after model
        # But model used to read AFTER clean wrote → model is now stale (contaminated)
        assert "model" in result.newly_stale

    def test_delete_unused_exploration_cells(self):
        """Delete multiple exploration cells that are no longer needed."""
        self.helper.set_cell_order(
            ["imports", "data", "explore1", "explore2", "explore3", "clean", "model"]
        )
        self.helper.execute_cell(
            "imports", pre_namespace={}, post_namespace={"pd": "pandas"},
            reads=set(), writes={"pd"}
        )
        self.helper.execute_cell(
            "data", pre_namespace={"pd": "pandas"},
            post_namespace={"pd": "pandas", "df": "dataframe"},
            reads={"pd"}, writes={"df"}
        )
        # Exploration cells just read df, don't write anything important
        self.helper.execute_cell(
            "explore1", pre_namespace={"pd": "pandas", "df": "dataframe"},
            post_namespace={"pd": "pandas", "df": "dataframe"},
            reads={"df"}, writes=set()
        )
        self.helper.execute_cell(
            "explore2", pre_namespace={"pd": "pandas", "df": "dataframe"},
            post_namespace={"pd": "pandas", "df": "dataframe"},
            reads={"df"}, writes=set()
        )
        self.helper.execute_cell(
            "explore3", pre_namespace={"pd": "pandas", "df": "dataframe"},
            post_namespace={"pd": "pandas", "df": "dataframe"},
            reads={"df"}, writes=set()
        )
        self.helper.execute_cell(
            "clean", pre_namespace={"pd": "pandas", "df": "dataframe"},
            post_namespace={"pd": "pandas", "df": "clean_df"},
            reads={"df"}, writes={"df"}
        )

        # Delete all exploration cells
        result = self.helper.sdc.set_cell_order(["imports", "data", "clean", "model"])

        assert set(result.delta.deleted) == {"explore1", "explore2", "explore3"}
        # Clean still depends on df from data, no staleness
        assert "clean" not in result.newly_stale

    def test_move_helper_function_up(self):
        """Move a helper function cell to be before cells that use it."""
        self.helper.set_cell_order(["imports", "data", "process", "helper", "model"])
        self.helper.execute_cell(
            "imports", pre_namespace={}, post_namespace={"pd": "pandas"},
            reads=set(), writes={"pd"}
        )
        self.helper.execute_cell(
            "data", pre_namespace={"pd": "pandas"},
            post_namespace={"pd": "pandas", "df": "dataframe"},
            reads={"pd"}, writes={"df"}
        )
        # Process tries to use helper function (would fail at runtime, but let's track)
        self.helper.execute_cell(
            "process", pre_namespace={"pd": "pandas", "df": "dataframe"},
            post_namespace={"pd": "pandas", "df": "dataframe", "result": "processed"},
            reads={"df", "helper_fn"}, writes={"result"},
            continue_on_violation=True  # helper_fn doesn't exist yet
        )
        self.helper.execute_cell(
            "helper", pre_namespace={"pd": "pandas", "df": "dataframe", "result": "processed"},
            post_namespace={"pd": "pandas", "df": "dataframe", "result": "processed", "helper_fn": "fn"},
            reads=set(), writes={"helper_fn"},
            continue_on_violation=True  # Backward violation: helper writes what process reads
        )

        # Move helper before process (where it should be)
        result = self.helper.sdc.set_cell_order(["imports", "data", "helper", "process", "model"])

        # Process reads helper_fn, helper writes helper_fn
        # Helper was after process, now before → process gains new input → process stale
        assert "process" in result.newly_stale
