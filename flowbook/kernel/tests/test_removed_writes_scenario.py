"""Test for the specific scenario: B changes what it writes, C should become stale.

Scenario:
    A: x = 10
    B: x = 20
    C: print(x)

    Run A → B → C
    Change B to "y = 10" and run B

Expected: C should be marked stale because:
    - C read x from B
    - B no longer writes x
    - C's dependency on x is now "orphaned" (source changed)
"""

import pytest
from flowbook.kernel.tests.conftest import ReproducibilityTestHelper


class TestRemovedWritesMarksDependentsStale:
    """Test that changing what a cell writes marks dependent cells stale."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["A", "B", "C"])

    def test_changed_writes_marks_reader_stale(self):
        """
        Scenario:
            A: x = 10
            B: x = 20
            C: print(x)  # reads x

            Run A → B → C
            Change B to y = 10 (no longer writes x)
            Run B

        Expected: C should be stale (reads x, but B no longer writes x)
        """
        # Execute A: writes x
        result_a = self.helper.execute_cell(
            "A",
            pre_namespace={},
            post_namespace={"x": 10},
            reads=set(),
            writes={"x"},
        )
        print(f"After A: stale_cells={result_a.stale_cells}")

        # Execute B (first time): writes x = 20
        result_b1 = self.helper.execute_cell(
            "B",
            pre_namespace={"x": 10},
            post_namespace={"x": 20},
            reads=set(),
            writes={"x"},
        )
        print(f"After B (first): stale_cells={result_b1.stale_cells}")

        # Execute C: reads x
        result_c = self.helper.execute_cell(
            "C",
            pre_namespace={"x": 20},
            post_namespace={"x": 20},
            reads={"x"},
            writes=set(),
        )
        print(f"After C: stale_cells={result_c.stale_cells}")

        # All cells should be fresh at this point
        state = self.helper.sdc._notebook_state
        assert state.status.get("A").is_clean, f"A should be clean, got {state.status.get('A')}"
        assert state.status.get("B").is_clean, f"B should be clean, got {state.status.get('B')}"
        assert state.status.get("C").is_clean, f"C should be clean, got {state.status.get('C')}"

        # Check that B's writes include x
        print(f"B's writes before change: {state.writes.get('B')}")
        assert any(w.name == "x" for w in state.writes.get("B", set())), \
            "B should have x in its writes"

        # Execute B (second time): now writes y instead of x
        # This simulates changing B's code from "x = 20" to "y = 10"
        result_b2 = self.helper.execute_cell(
            "B",
            pre_namespace={"x": 10},  # Restored from checkpoint (A's value)
            post_namespace={"x": 10, "y": 10},  # x unchanged, y added
            reads=set(),
            writes={"y"},  # Now B only writes y, not x
        )
        print(f"After B (second): stale_cells={result_b2.stale_cells}")

        # C should now be stale because:
        # - C reads x
        # - B used to write x but no longer does
        # - C's dependency has changed (now reads from A, not B)
        assert "C" in result_b2.stale_cells, \
            f"Expected C to be stale after B changed its writes, got stale_cells={result_b2.stale_cells}"

        # Verify C is marked stale in state
        assert not state.status.get("C").is_clean, \
            f"Expected C to be stale (is_clean=False), got {state.status.get('C')}"

    def test_removed_writes_tracked_correctly(self):
        """Verify W_i_old is tracked correctly for removed-writes detection."""
        # Execute A: writes x
        self.helper.execute_cell(
            "A",
            pre_namespace={},
            post_namespace={"x": 10},
            reads=set(),
            writes={"x"},
        )

        # Execute B: writes x
        self.helper.execute_cell(
            "B",
            pre_namespace={"x": 10},
            post_namespace={"x": 20},
            reads=set(),
            writes={"x"},
        )

        state = self.helper.sdc._notebook_state

        # Before second execution, B should have x in writes
        W_i_old = state.writes.get("B", set())
        print(f"W_i_old (B's writes before re-execution): {W_i_old}")
        assert any(w.name == "x" for w in W_i_old), \
            f"Expected x in B's old writes, got {W_i_old}"

    def test_detailed_removed_writes_flow(self):
        """
        Detailed test showing exactly what happens at each step.
        This mirrors the real kernel's behavior more closely.
        """
        state = self.helper.sdc._notebook_state

        # Step 1: Execute A - writes x = 10
        print("\n=== Step 1: Execute A ===")
        self.helper.execute_cell(
            "A",
            pre_namespace={},
            post_namespace={"x": 10},
            reads=set(),
            writes={"x"},
        )
        print(f"  A's writes in state: {state.writes.get('A')}")
        print(f"  Stale cells: {state.get_stale_cells()}")

        # Step 2: Execute B - writes x = 20
        print("\n=== Step 2: Execute B (writes x) ===")
        self.helper.execute_cell(
            "B",
            pre_namespace={"x": 10},
            post_namespace={"x": 20},
            reads=set(),
            writes={"x"},
        )
        print(f"  B's writes in state: {state.writes.get('B')}")
        print(f"  Stale cells: {state.get_stale_cells()}")

        # Step 3: Execute C - reads x
        print("\n=== Step 3: Execute C (reads x) ===")
        self.helper.execute_cell(
            "C",
            pre_namespace={"x": 20},
            post_namespace={"x": 20},
            reads={"x"},
            writes=set(),
        )
        print(f"  C's reads in state: {state.reads.get('C')}")
        print(f"  Stale cells: {state.get_stale_cells()}")
        print(f"  All cells should be clean now")

        # Verify all cells are clean
        assert state.status.get("A").is_clean
        assert state.status.get("B").is_clean
        assert state.status.get("C").is_clean

        # Step 4: Re-execute B with different writes (y instead of x)
        print("\n=== Step 4: Re-execute B (now writes y, not x) ===")
        print(f"  BEFORE re-execution:")
        print(f"    B's writes (W_i_old): {state.writes.get('B')}")

        result = self.helper.execute_cell(
            "B",
            pre_namespace={"x": 10},  # Restored from A's checkpoint
            post_namespace={"x": 10, "y": 10},  # x from A, y newly written
            reads=set(),
            writes={"y"},  # Now only writes y
        )

        print(f"  AFTER re-execution:")
        print(f"    B's writes (updated): {state.writes.get('B')}")
        print(f"    stale_cells from result: {result.stale_cells}")
        print(f"    staleness_reasons: {result.staleness_reasons}")

        # Check that C is stale
        assert "C" in result.stale_cells, \
            f"C should be stale, got stale_cells={result.stale_cells}"

        # Check that the reason is WRITE_OVERLAP with loc='x'
        c_reasons = result.staleness_reasons.get("C", [])
        print(f"    C's staleness reasons: {c_reasons}")
        assert len(c_reasons) > 0, "C should have staleness reasons"

        # Verify the reason mentions x (the removed write)
        reason_locs = [r.get("loc") for r in c_reasons if r.get("loc")]
        print(f"    Reason locations: {reason_locs}")
        assert "x" in reason_locs, \
            f"C's staleness reason should mention 'x', got {c_reasons}"
