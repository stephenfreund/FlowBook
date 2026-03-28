"""
Tests for notebook execution workflow using NotebookDriver.

These tests verify basic execution behavior, staleness detection,
and the interaction between cell editing and execution.

Note: Provenance tracking (last_writer, column_last_writer) has been removed.
Tests that depended on ProvenanceAdapter or forward_violation/violation fields
have been removed as part of the provenance removal refactoring.
"""

import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData

from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
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
    - Observing staleness
    """

    def __init__(self, cell_order: list):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.enforcer = ReproducibilityEnforcer(self.checkpoints)
        self.enforcer.set_cell_order(cell_order)

        # Simulated live store (kernel namespace)
        self.live_store: dict = {}

        # Cell definitions: cell_id -> (code_description, reads, writes, effect_fn)
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
        self.define_cell(cell_id, description, reads, writes, column_writes, effect)
        self.enforcer.mark_cell_edited(cell_id)

    def run_cell(self, cell_id: str):
        cell_def = self._cell_defs.get(cell_id)
        if cell_def is None:
            raise ValueError(f"Cell {cell_id} not defined")

        pre_name = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
        self.checkpoints.save(pre_name, dict(self.live_store), max_size_mb=None)
        pre_checkpoint = self.checkpoints.get(pre_name)

        self.live_store = cell_def["effect"](dict(self.live_store))

        tracking = make_tracking(
            reads=cell_def["reads"],
            writes=cell_def["writes"],
            column_writes=cell_def["column_writes"],
        )

        result = self.enforcer.check(
            cell_id=cell_id,
            pre_checkpoint=pre_checkpoint,
            namespace=dict(self.live_store),
            tracking=tracking,
        )

        return result

    @property
    def stale_cells(self) -> list:
        return self.enforcer.get_stale_cells()


class TestNotebookDriverBasic:
    """Basic tests for NotebookDriver execution."""

    def test_simple_execution_no_errors(self):
        """Simple in-order execution produces no errors."""
        driver = NotebookDriver(["a", "b"])
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        result = driver.run_cell("a")
        assert not result.has_errors()

    def test_edit_marks_stale(self):
        """Editing a cell marks it stale."""
        driver = NotebookDriver(["a", "b"])
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")
        driver.edit_cell("a", "x = 2", writes={"x"}, effect=lambda s: {**s, "x": 2})
        assert "a" in driver.stale_cells

    def test_multiple_variables_different_cells(self):
        """Multiple cells writing different variables produces no errors."""
        driver = NotebookDriver(["a", "b", "c"])
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.define_cell("b", "y = 2", writes={"y"}, effect=lambda s: {**s, "y": 2})
        driver.define_cell("c", "z = 3", writes={"z"}, effect=lambda s: {**s, "z": 3})
        result_a = driver.run_cell("a")
        result_b = driver.run_cell("b")
        result_c = driver.run_cell("c")
        assert not result_a.has_errors()
        assert not result_b.has_errors()
        assert not result_c.has_errors()

    def test_reset_clears_state(self):
        """Kernel restart clears all state."""
        driver = NotebookDriver(["a"])
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.run_cell("a")
        driver.enforcer.reset()
        # After reset, stale cells list is empty (no cells in order)
        assert driver.stale_cells == []


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_edit_unexecuted_cell(self):
        """Editing an unexecuted cell keeps it stale."""
        driver = NotebookDriver(["a", "b"])
        driver.define_cell("a", "x = 1", writes={"x"}, effect=lambda s: {**s, "x": 1})
        driver.edit_cell("a", "y = 1", writes={"y"}, effect=lambda s: {**s, "y": 1})
        # Cell was never executed, so still stale
        assert "a" in driver.stale_cells
