"""The offline ReproducibilitySimulator mirrors the kernel's execute path.

These run without a Jupyter kernel: exec() plus the real checkpoint,
tracking, and enforcer machinery.
"""
import os

import pytest

from flowbook.testing.correctness import run_correctness_test_from_notebook
from flowbook.testing.notebook_loader import Cell
from flowbook.testing.runner import ReproducibilitySimulator

NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")


def _cells(sources):
    return [
        Cell(cell_id=cid, source=src, cell_type="code", index=i)
        for i, (cid, src) in enumerate(sources)
    ]


def _sim(cells, continue_on_violation=True):
    sim = ReproducibilitySimulator()
    sim.continue_on_violation = continue_on_violation
    sim.cells = cells
    sim.enforcer.set_cell_order([c.cell_id for c in cells])
    return sim


@pytest.mark.parametrize("name", ["deterministic.ipynb", "dependencies.ipynb"])
def test_bundled_notebooks_pass_correctness(name):
    simulator, results = run_correctness_test_from_notebook(
        os.path.join(NOTEBOOKS, name), iterations_per_cell=1, seed=42
    )
    assert simulator.cell_records
    assert results


def test_edit_then_rerun_propagates_staleness():
    cells = _cells([("A", "x = 1"), ("B", "y = x + 1"), ("C", "z = y * 2")])
    sim = _sim(cells)
    for c in cells:
        rec = sim.execute_cell(c)
        assert rec.error is None
        assert not rec.sdc_result.has_errors()
    assert sim.enforcer.get_stale_cells() == []

    assert sim.enforcer.mark_cell_edited("A") == ["A"]

    rec = sim.execute_cell(Cell(cell_id="A", source="x = 99", cell_type="code", index=0))
    # ForwardStale marks direct readers of what A wrote; C follows when B reruns
    assert "B" in rec.sdc_result.stale_cells
    assert "C" not in rec.sdc_result.stale_cells
    assert sim.namespace["x"] == 99

    rec = sim.execute_cell(cells[1])
    assert "C" in rec.sdc_result.stale_cells
    assert "B" not in rec.sdc_result.stale_cells


def test_rejected_violation_rolls_back_namespace():
    cells = _cells([("A", "x = 1"), ("B", "print(x)"), ("C", "x = 2")])
    sim = _sim(cells, continue_on_violation=False)
    sim.execute_cell(cells[0])
    sim.execute_cell(cells[1])
    rec = sim.execute_cell(cells[2])
    assert rec.sdc_result.has_errors()
    assert sim.namespace["x"] == 1, "rejected execution must be rolled back"


def test_accepted_violation_keeps_namespace():
    cells = _cells([("A", "x = 1"), ("B", "print(x)"), ("C", "x = 2")])
    sim = _sim(cells, continue_on_violation=True)
    for c in cells:
        rec = sim.execute_cell(c)
    assert rec.sdc_result.has_errors()
    assert sim.namespace["x"] == 2


def test_exception_restores_namespace_and_skips_check():
    cells = _cells([("A", "x = 1"), ("B", "x = 5\nraise ValueError('boom')")])
    sim = _sim(cells)
    sim.execute_cell(cells[0])
    stale_before = sim.enforcer.get_stale_cells()
    rec = sim.execute_cell(cells[1])
    assert rec.error and "ValueError" in rec.error
    assert sim.namespace["x"] == 1
    assert not rec.sdc_result.has_errors()
    # No check ran, so the enforcer's view is unchanged
    assert sim.enforcer.get_stale_cells() == stale_before
