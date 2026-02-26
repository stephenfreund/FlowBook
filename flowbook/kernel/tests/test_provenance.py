"""
Tests for Provenance Tracking (SS1.8.5).

Provenance tracks which cell last wrote each location. This enables detection
of forward contamination even after cells are edited, without requiring
explicit orphan tracking.

Key insight: When cell C writes x, then C is edited to write y and re-executed:
- Prov["x"] = C (from old execution, NOT cleared on edit)
- Prov["y"] = C (new)
- When B (earlier cell) reads x: Prov["x"] = C, C is after B -> contaminated

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
        column_writes: dict = None,
        effect: callable = None,
    ):
        """
        Define or redefine a cell's behavior.

        Args:
            cell_id: ID of the cell
            description: Human-readable description (e.g., "x = 0")
            reads: Set of variables the cell reads
            writes: Set of variables the cell writes
            column_writes: Dict of var -> set of columns written
            effect: Function that takes live_store and returns modified store
        """
        self._cell_defs[cell_id] = {
            "description": description,
            "reads": reads or set(),
            "writes": writes or set(),
            "column_writes": column_writes or {},
            "effect": effect or (lambda s: s),
        }

    def edit_cell(
        self,
        cell_id: str,
        description: str,
        reads: set = None,
        writes: set = None,
        column_writes: dict = None,
        effect: callable = None,
    ):
        """
        Edit a cell's code (triggers EDIT transition in enforcer).

        This simulates the user editing a cell in the notebook UI.
        """
        # First define the new cell behavior
        self.define_cell(cell_id, description, reads, writes, column_writes, effect)

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
            column_writes=cell_def["column_writes"],
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
    def provenance(self):
        """Get the current provenance map."""
        return self.enforcer.get_provenance()

    @property
    def stale_cells(self) -> list:
        """Get the current set of stale cells."""
        return self.enforcer.get_stale_cells()


class TestProvenanceBasic:
    """Basic provenance tracking tests."""

    def test_provenance_updates_on_write(self):
        """Executing a cell updates provenance for its written locations."""
        driver = NotebookDriver(["a", "b", "c"])

        # Define and run cell A: x = 1
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # Provenance should show A wrote x
        assert driver.provenance.get_variable_writer("x") == "a"

    def test_provenance_overwritten_by_later_cell(self):
        """A later cell overwrites provenance when it writes the same variable."""
        driver = NotebookDriver(["a", "b", "c"])

        # A writes x
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")
        assert driver.provenance.get_variable_writer("x") == "a"

        # C writes x (later in order)
        driver.define_cell("c", "x = 2", writes={"x"}, effect=lambda s: {**s, "x": 2})
        driver.run_cell("c")

        # Now provenance shows C wrote x
        assert driver.provenance.get_variable_writer("x") == "c"

    def test_provenance_persists_after_edit(self):
        """
        Provenance persists after cell edit.

        This is the key insight: when C is edited from "x = 1" to "y = 1",
        Prov["x"] = C still holds (it's not cleared).
        """
        driver = NotebookDriver(["a", "b", "c"])

        # C writes x
        driver.define_cell("c", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("c")
        assert driver.provenance.get_variable_writer("x") == "c"

        # Edit C to write y instead (x is NOT cleared from provenance)
        driver.edit_cell("c", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})

        # Provenance for x still points to C (even though C now writes y)
        assert driver.provenance.get_variable_writer("x") == "c"

        # Run C with new code
        driver.run_cell("c")

        # Now C also has provenance for y, but x still points to C
        assert driver.provenance.get_variable_writer("x") == "c"
        assert driver.provenance.get_variable_writer("y") == "c"


class TestProvenanceContamination:
    """Tests for provenance-based contamination detection."""

    def test_contamination_from_later_cell(self):
        """Reading a variable written by a later cell triggers contamination."""
        driver = NotebookDriver(["a", "b", "c"])

        # C writes x (later cell)
        driver.define_cell("c", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("c")

        # A reads x (earlier cell, but C already ran)
        driver.define_cell("a", "print(x)", reads={"x"}, effect=lambda s: s)
        result = driver.run_cell("a")

        # A should be contaminated: it reads x whose provenance points to C (later)
        assert result.forward_violation is not None
        assert result.cell_is_contaminated
        assert "x" in result.forward_violation.variables

    def test_no_contamination_from_earlier_cell(self):
        """Reading a variable written by an earlier cell is OK."""
        driver = NotebookDriver(["a", "b", "c"])

        # A writes x (earlier cell)
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # C reads x (later cell reading from earlier)
        driver.define_cell("c", "print(x)", reads={"x"}, effect=lambda s: s)
        result = driver.run_cell("c")

        # C should NOT be contaminated
        assert result.forward_violation is None
        assert not result.cell_is_contaminated


class TestBugScenario:
    """
    Tests for the specific bug scenario that motivated provenance tracking.

    @A: x = 0
    @B: print(x)
    @C: x = 1

    1. Run @A -> x=0
    2. Run @C -> x=1
    3. Run @B -> contaminated (correct)
    4. Edit @C to "y=1"
    5. Run @C -> y=1 (x still 1 in store)
    6. Run @B -> should STILL be contaminated (Prov["x"] = C, C > B)
    """

    def test_bug_scenario_edit_removes_write(self):
        """
        The main bug scenario: edit removes a write, but provenance persists.

        Before the fix (with orphan tracking), step 6 would NOT detect contamination
        because C's new trace doesn't write x.

        With provenance tracking, x still has Prov["x"] = C, and since C > B in
        document order, B is correctly marked contaminated.
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
        assert driver.provenance.get_variable_writer("x") == "a"

        # Step 2: Run C (out of order)
        result_c = driver.run_cell("c")
        assert result_c.violation is None  # No backward conflict (A didn't read x)
        assert driver.live_store["x"] == 1
        assert driver.provenance.get_variable_writer("x") == "c"

        # Step 3: Run B
        result_b = driver.run_cell("b")
        assert result_b.forward_violation is not None  # Contaminated by C
        assert result_b.cell_is_contaminated
        assert "b" in driver.stale_cells

        # Step 4: Edit C to "y = 1"
        driver.edit_cell("c", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        # Provenance for x STILL points to C (not cleared on edit)
        assert driver.provenance.get_variable_writer("x") == "c"

        # Step 5: Run C with new code
        result_c2 = driver.run_cell("c")
        assert result_c2.violation is None
        assert driver.live_store["x"] == 1  # x STILL 1 from old execution!
        assert driver.live_store["y"] == 1
        # Provenance for x still points to C
        assert driver.provenance.get_variable_writer("x") == "c"
        assert driver.provenance.get_variable_writer("y") == "c"

        # Step 6: Run B - THE FIX
        result_b2 = driver.run_cell("b")
        # B should STILL be contaminated because Prov["x"] = C and C > B
        assert result_b2.forward_violation is not None
        assert result_b2.cell_is_contaminated
        assert "x" in result_b2.forward_violation.variables

    def test_bug_scenario_cleared_by_earlier_cell_rerun(self):
        """
        After the bug scenario, re-running A (which writes x) updates provenance.
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

        # x has provenance pointing to C
        assert driver.provenance.get_variable_writer("x") == "c"

        # Re-run A (which writes x)
        result_a2 = driver.run_cell("a")
        assert result_a2.violation is None

        # x now has provenance pointing to A (earlier cell)
        assert driver.provenance.get_variable_writer("x") == "a"
        assert driver.live_store["x"] == 0  # A restored x to 0

        # Now B should be clean (Prov["x"] = A, A < B)
        result_b3 = driver.run_cell("b")
        assert result_b3.forward_violation is None
        assert not result_b3.cell_is_contaminated


class TestProvenanceWithMultipleCells:
    """Tests involving multiple cells and complex scenarios."""

    def test_multiple_variables_provenance(self):
        """Multiple variables can have different provenance."""
        driver = NotebookDriver(["a", "b", "c"])

        # A writes x
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # B writes y
        driver.define_cell("b", "y = 2", writes={"y"}, effect=lambda s: {**s, "y": 2})
        driver.run_cell("b")

        # C writes z
        driver.define_cell("c", "z = 3", writes={"z"}, effect=lambda s: {**s, "z": 3})
        driver.run_cell("c")

        assert driver.provenance.get_variable_writer("x") == "a"
        assert driver.provenance.get_variable_writer("y") == "b"
        assert driver.provenance.get_variable_writer("z") == "c"

    def test_provenance_overwrite_chain(self):
        """
        Multiple cells writing the same variable in sequence.
        Provenance tracks the most recent writer.
        """
        driver = NotebookDriver(["a", "b", "c"])

        # A writes x
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")
        assert driver.provenance.get_variable_writer("x") == "a"

        # B writes x
        driver.define_cell("b", "x = 2", writes={"x"}, effect=lambda s: {**s, "x": 2})
        driver.run_cell("b")
        assert driver.provenance.get_variable_writer("x") == "b"

        # C writes x
        driver.define_cell("c", "x = 3", writes={"x"}, effect=lambda s: {**s, "x": 3})
        driver.run_cell("c")
        assert driver.provenance.get_variable_writer("x") == "c"

        # Run A again - x now points to A
        driver.run_cell("a")
        assert driver.provenance.get_variable_writer("x") == "a"


class TestProvenanceReset:
    """Tests for provenance reset."""

    def test_reset_clears_provenance(self):
        """Kernel restart clears all provenance."""
        driver = NotebookDriver(["a"])

        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        assert driver.provenance.get_variable_writer("x") == "a"

        # Simulate kernel restart
        driver.enforcer.reset()

        assert driver.provenance.get_variable_writer("x") is None


class TestProvenanceColumnLevel:
    """Tests for column-level provenance tracking."""

    def test_column_provenance_basic(self):
        """Column-level provenance is tracked separately."""
        driver = NotebookDriver(["a", "b"])

        # A writes df['x']
        driver.define_cell(
            "a", "df['x'] = 1",
            writes={"df"},
            column_writes={"df": {"x"}},
            effect=lambda s: {**s, "df": {"x": 1}}
        )
        driver.run_cell("a")

        assert driver.provenance.get_column_writer("df", "x") == "a"

        # B writes df['y']
        driver.define_cell(
            "b", "df['y'] = 2",
            writes={"df"},
            column_writes={"df": {"y"}},
            effect=lambda s: {**s, "df": {**s.get("df", {}), "y": 2}}
        )
        driver.run_cell("b")

        assert driver.provenance.get_column_writer("df", "x") == "a"
        assert driver.provenance.get_column_writer("df", "y") == "b"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_edit_unexecuted_cell_no_provenance_change(self):
        """Editing an unexecuted cell doesn't affect provenance."""
        driver = NotebookDriver(["a", "b"])

        # Define but don't run A
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})

        # Edit A without running it first
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})

        # No provenance (A was never executed)
        assert driver.provenance.get_variable_writer("x") is None
        assert driver.provenance.get_variable_writer("y") is None

    def test_self_contamination_not_detected(self):
        """A cell doesn't contaminate itself when it writes and reads the same var."""
        driver = NotebookDriver(["a", "b"])

        # A writes x
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")

        # A now reads and writes x (increment)
        driver.define_cell(
            "a", "x = x + 1",
            reads={"x"}, writes={"x"},
            effect=lambda s: {**s, "x": s["x"] + 1}
        )
        result = driver.run_cell("a")

        # A should NOT be contaminated by itself
        assert result.forward_violation is None
        assert not result.cell_is_contaminated
