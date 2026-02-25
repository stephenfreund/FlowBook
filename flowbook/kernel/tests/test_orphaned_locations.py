"""
Tests for Orphaned Location Tracking (§1.8.5, §2.3).

Orphaned locations are variables that were written by a cell whose code has
since changed. Values at these locations don't correspond to any current
cell's behavior, making them a source of forward contamination.

These tests use a NotebookDriver class that simulates the notebook execution
workflow, including editing cells and the interplay between execution and
staleness/contamination detection.
"""

import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData

from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
    POST_CHECKPOINT_PREFIX,
)
from flowbook.kernel.tests.conftest import make_tracking


class NotebookDriver:
    """
    Simulates notebook execution workflow for testing.

    This driver maintains:
    - A simulated live store (namespace)
    - Cell code definitions (can be edited)
    - Execution state via ReproducibilityEnforcer

    It allows testing scenarios like:
    - Running cells in various orders
    - Editing cell code
    - Re-executing edited cells
    - Observing staleness and contamination
    """

    def __init__(self, cell_order: list):
        """
        Initialize the notebook driver.

        Args:
            cell_order: List of cell IDs in notebook order (e.g., ["a", "b", "c"])
        """
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.enforcer = ReproducibilityEnforcer(self.checkpoints)
        self.enforcer.set_cell_order(cell_order)

        # Simulated live store (kernel namespace)
        self.live_store: dict = {}

        # Cell definitions: cell_id -> (code_description, reads, writes, effect_fn)
        # effect_fn takes live_store and returns the modified store
        self._cell_defs: dict = {}

    def define_cell(
        self,
        cell_id: str,
        description: str,
        reads: set = None,
        writes: set = None,
        effect: callable = None,
    ):
        """
        Define or redefine a cell's behavior.

        Args:
            cell_id: ID of the cell
            description: Human-readable description (e.g., "x = 0")
            reads: Set of variables the cell reads
            writes: Set of variables the cell writes
            effect: Function that takes live_store and returns modified store
        """
        self._cell_defs[cell_id] = {
            "description": description,
            "reads": reads or set(),
            "writes": writes or set(),
            "effect": effect or (lambda s: s),
        }

    def edit_cell(
        self,
        cell_id: str,
        description: str,
        reads: set = None,
        writes: set = None,
        effect: callable = None,
    ):
        """
        Edit a cell's code (triggers EDIT transition in enforcer).

        This simulates the user editing a cell in the notebook UI.
        """
        # First define the new cell behavior
        self.define_cell(cell_id, description, reads, writes, effect)

        # Then trigger the EDIT transition
        self.enforcer.mark_cell_edited(cell_id)

    def run_cell(self, cell_id: str):
        """
        Execute a cell and return the ReproducibilityResult.

        This simulates:
        1. Taking a pre-checkpoint
        2. Executing the cell code (via effect function)
        3. Taking a post-checkpoint
        4. Running the reproducibility check

        Returns:
            ReproducibilityResult from the enforcer
        """
        cell_def = self._cell_defs.get(cell_id)
        if cell_def is None:
            raise ValueError(f"Cell {cell_id} not defined")

        # Take pre-checkpoint
        pre_name = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
        self.checkpoints.save(pre_name, dict(self.live_store), max_size_mb=None)
        pre_checkpoint = self.checkpoints.get(pre_name)

        # Execute the cell (apply its effect to the live store)
        self.live_store = cell_def["effect"](dict(self.live_store))

        # Take post-checkpoint
        post_name = f"{POST_CHECKPOINT_PREFIX}{cell_id}"
        self.checkpoints.save(post_name, dict(self.live_store), max_size_mb=None)
        post_checkpoint = self.checkpoints.get(post_name)

        # Create tracking data
        tracking = make_tracking(
            reads=cell_def["reads"],
            writes=cell_def["writes"],
        )

        # Run the check
        result = self.enforcer.check(
            cell_id=cell_id,
            pre_checkpoint=pre_checkpoint,
            post_checkpoint=post_checkpoint,
            tracking=tracking,
        )

        return result

    @property
    def orphaned_locations(self) -> list:
        """Get the current set of orphaned locations."""
        return self.enforcer.get_orphaned_locations()

    @property
    def stale_cells(self) -> list:
        """Get the current set of stale cells."""
        return self.enforcer.get_stale_cells()


class TestOrphanedLocationsBasic:
    """Basic orphaned location tracking tests."""

    def test_edit_creates_orphaned_locations(self):
        """Editing a cell orphans its written locations."""
        driver = NotebookDriver(["a", "b", "c"])

        # Define and run cell A: x = 1
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # No orphaned locations yet
        assert driver.orphaned_locations == []

        # Edit cell A to "y = 1" (no longer writes x)
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})

        # x should now be orphaned
        assert "x" in driver.orphaned_locations

    def test_execution_clears_orphaned_locations(self):
        """Executing a cell clears orphaned status for its writes."""
        driver = NotebookDriver(["a", "b"])

        # Cell A writes x
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # Edit A to write y instead
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        assert "x" in driver.orphaned_locations

        # Define cell B that writes x
        driver.define_cell("b", "x = 2", writes={"x"}, effect=lambda s: {**s, "x": 2})
        driver.run_cell("b")

        # x should no longer be orphaned (B claimed it)
        assert "x" not in driver.orphaned_locations

    def test_reset_clears_orphaned_locations(self):
        """Kernel restart clears all orphaned locations."""
        driver = NotebookDriver(["a"])

        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})

        assert "x" in driver.orphaned_locations

        # Simulate kernel restart
        driver.enforcer.reset()

        assert driver.orphaned_locations == []


class TestOrphanedContamination:
    """Tests for orphan contamination detection."""

    def test_reading_orphaned_location_triggers_contamination(self):
        """Reading an orphaned location triggers forward contamination."""
        driver = NotebookDriver(["a", "b"])

        # A writes x=1
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # Edit A to write y instead (x becomes orphaned)
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        driver.run_cell("a")

        assert "x" in driver.orphaned_locations
        assert driver.live_store.get("x") == 1  # x still in store from old execution

        # B reads x (orphaned)
        driver.define_cell("b", "print(x)", reads={"x"}, effect=lambda s: s)
        result = driver.run_cell("b")

        # Should trigger forward contamination
        assert result.forward_violation is not None
        assert result.cell_is_contaminated
        assert "x" in result.forward_violation.variables
        assert "orphaned" in result.forward_violation.message.lower()


class TestBugScenario:
    """
    Tests for the specific bug scenario that motivated this feature.

    @A: x = 0
    @B: print(x)
    @C: x = 1

    1. Run @A → x=0
    2. Run @C → x=1
    3. Run @B → contaminated (correct)
    4. Edit @C to "y=1"
    5. Run @C → y=1 (x still 1 in store)
    6. Run @B → should STILL be contaminated (x is orphaned)
    """

    def test_bug_scenario_edit_removes_write(self):
        """
        The main bug scenario: edit removes a write, orphaning the location.

        Before the fix, step 6 would NOT detect contamination because:
        - C's new Δ = {y}, not {x}
        - FwdContaminated checked RBW(B) ∩ Δ(C) = {x} ∩ {y} = ∅

        With the fix, x is tracked as orphaned, and B reading x triggers contamination.
        """
        driver = NotebookDriver(["a", "b", "c"])

        # Define cells
        driver.define_cell("a", "x = 0", writes={"x"}, effect=lambda s: {**s, "x": 0})
        driver.define_cell("b", "print(x)", reads={"x"}, effect=lambda s: s)
        driver.define_cell("c", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})

        # Step 1: Run A
        result_a = driver.run_cell("a")
        assert result_a.violation is None
        assert result_a.forward_violation is None
        assert driver.live_store["x"] == 0

        # Step 2: Run C (out of order)
        result_c = driver.run_cell("c")
        assert result_c.violation is None  # No backward conflict (A didn't read x)
        assert driver.live_store["x"] == 1

        # Step 3: Run B
        result_b = driver.run_cell("b")
        assert result_b.forward_violation is not None  # Contaminated by C
        assert result_b.cell_is_contaminated
        assert "b" in driver.stale_cells

        # Step 4: Edit C to "y = 1"
        driver.edit_cell("c", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        assert "x" in driver.orphaned_locations  # x is now orphaned

        # Step 5: Run C with new code
        result_c2 = driver.run_cell("c")
        assert result_c2.violation is None
        assert driver.live_store["x"] == 1  # x STILL 1 from old execution!
        assert driver.live_store["y"] == 1
        assert "x" in driver.orphaned_locations  # x still orphaned (C didn't write it)

        # Step 6: Run B - THE FIX
        result_b2 = driver.run_cell("b")
        # B should STILL be contaminated because x is orphaned
        assert result_b2.forward_violation is not None
        assert result_b2.cell_is_contaminated
        assert "x" in result_b2.forward_violation.variables
        assert "orphaned" in result_b2.forward_violation.message.lower()

    def test_bug_scenario_cleared_by_rerun_original_writer(self):
        """
        After the bug scenario, re-running A (which writes x) should clear the orphan.
        """
        driver = NotebookDriver(["a", "b", "c"])

        # Setup: same as bug scenario up to step 5
        driver.define_cell("a", "x = 0", writes={"x"}, effect=lambda s: {**s, "x": 0})
        driver.define_cell("b", "print(x)", reads={"x"}, effect=lambda s: s)
        driver.define_cell("c", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})

        driver.run_cell("a")
        driver.run_cell("c")
        driver.run_cell("b")
        driver.edit_cell("c", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        driver.run_cell("c")

        assert "x" in driver.orphaned_locations

        # Re-run A (which writes x)
        result_a2 = driver.run_cell("a")
        assert result_a2.violation is None

        # x should no longer be orphaned
        assert "x" not in driver.orphaned_locations
        assert driver.live_store["x"] == 0  # A restored x to 0

        # Now B should be clean (x is no longer orphaned, and C doesn't write x)
        result_b3 = driver.run_cell("b")
        assert result_b3.forward_violation is None
        assert not result_b3.cell_is_contaminated


class TestOrphanedWithMultipleCells:
    """Tests involving multiple cells and complex scenarios."""

    def test_multiple_orphaned_locations(self):
        """Multiple locations can be orphaned from the same cell."""
        driver = NotebookDriver(["a", "b"])

        # A writes x and y
        driver.define_cell(
            "a", "x, y = 1, 2",
            writes={"x", "y"},
            effect=lambda s: {**s, "x": 1, "y": 2}
        )
        driver.run_cell("a")

        # Edit A to write only z
        driver.edit_cell(
            "a", "z = 3",
            writes={"z"},
            effect=lambda s: {**s, "z": 3}
        )

        # Both x and y should be orphaned
        assert "x" in driver.orphaned_locations
        assert "y" in driver.orphaned_locations
        assert "z" not in driver.orphaned_locations

    def test_partial_orphan_clearing(self):
        """Executing a cell clears only the locations it writes."""
        driver = NotebookDriver(["a", "b", "c"])

        # A writes x and y
        driver.define_cell(
            "a", "x, y = 1, 2",
            writes={"x", "y"},
            effect=lambda s: {**s, "x": 1, "y": 2}
        )
        driver.run_cell("a")

        # Edit A to write nothing relevant
        driver.edit_cell("a", "pass", writes=set(), effect=lambda s: s)

        assert "x" in driver.orphaned_locations
        assert "y" in driver.orphaned_locations

        # B writes only x
        driver.define_cell("b", "x = 10", writes={"x"}, effect=lambda s: {**s, "x": 10})
        driver.run_cell("b")

        # Only x should be cleared
        assert "x" not in driver.orphaned_locations
        assert "y" in driver.orphaned_locations

    def test_chain_of_edits(self):
        """
        Multiple edits accumulate orphaned locations.

        A: x = 1  →  A: y = 2  →  A: z = 3
        Orphaned: {}  →  {x}    →  {x, y}
        """
        driver = NotebookDriver(["a"])

        # First version: x = 1
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")
        assert driver.orphaned_locations == []

        # Edit to y = 2
        driver.edit_cell("a", "y = 2", writes={"y"}, effect=lambda s: {**s, "y": 2})
        driver.run_cell("a")
        assert "x" in driver.orphaned_locations
        assert "y" not in driver.orphaned_locations

        # Edit to z = 3
        driver.edit_cell("a", "z = 3", writes={"z"}, effect=lambda s: {**s, "z": 3})
        driver.run_cell("a")
        assert "x" in driver.orphaned_locations
        assert "y" in driver.orphaned_locations
        assert "z" not in driver.orphaned_locations


class TestOrphanedInMetadata:
    """Tests for orphaned location metadata propagation."""

    def test_orphaned_reads_in_result(self):
        """ReproducibilityResult includes orphaned_reads field."""
        driver = NotebookDriver(["a", "b"])

        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        driver.run_cell("a")

        driver.define_cell("b", "print(x)", reads={"x"}, effect=lambda s: s)
        result = driver.run_cell("b")

        # orphaned_reads should contain x
        assert "x" in result.orphaned_reads

    def test_get_orphaned_locations_method(self):
        """Enforcer exposes get_orphaned_locations() method."""
        driver = NotebookDriver(["a"])

        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        assert driver.enforcer.get_orphaned_locations() == []

        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})

        assert "x" in driver.enforcer.get_orphaned_locations()


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_edit_unexecuted_cell_no_orphan(self):
        """Editing an unexecuted cell doesn't create orphans."""
        driver = NotebookDriver(["a", "b"])

        # Define but don't run A
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})

        # Edit A without running it first
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})

        # No orphaned locations (A was never executed)
        assert driver.orphaned_locations == []

    def test_orphan_from_deleted_cell(self):
        """
        If a cell is deleted, its orphaned locations persist.

        The orphaned values are still in the store even if the cell is gone.
        """
        driver = NotebookDriver(["a", "b", "c"])

        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        assert "x" in driver.orphaned_locations

        # Simulate cell deletion by changing cell order
        driver.enforcer.set_cell_order(["b", "c"])

        # x is still orphaned (the value is still in the store)
        assert "x" in driver.orphaned_locations

    def test_redefine_with_same_writes_no_new_orphan(self):
        """Re-running a cell that writes the same variables doesn't create orphans."""
        driver = NotebookDriver(["a"])

        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # Re-run with same writes but different value
        driver.define_cell("a", "x = 2", writes={"x"}, effect=lambda s: {**s, "x": 2})
        driver.run_cell("a")

        # No orphaned locations
        assert driver.orphaned_locations == []
