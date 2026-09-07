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


def test_rebinding_a_shared_list_does_not_mutate_its_holders():
    """A cell that re-runs `cols = [...]` must not be blamed for changing objects that hold the old list."""
    cells = _cells([
        ("A", "class H:\n    def __init__(self, c): self.c = c"),
        ("B", "cols = ['x', 'y']"),
        ("C", "holder = H(cols)"),
    ])
    sim = _sim(cells)
    for c in cells:
        sim.execute_cell(c)
    rec = sim.execute_cell(cells[1])  # rebinds cols; holder still references the old list
    assert not rec.sdc_result.has_errors(), [e.error_type.value for e in rec.sdc_result.errors]
    assert "holder" not in rec.sdc_result.changed_variables
    assert "C" in rec.sdc_result.stale_cells  # C read cols, so it is forward-stale: correct


def test_use_before_import_across_cells_is_forward_contamination():
    """A cell that uses a module before the cell that imports it (in order) is rejected."""
    cells = _cells([("A", 's = json.dumps({"a": 1})'), ("B", "import json")])
    sim = _sim(cells)
    sim.execute_cell(cells[1])          # import first in time...
    rec = sim.execute_cell(cells[0])    # ...but A comes first in order and reads json
    assert [e.error_type.value for e in rec.sdc_result.errors] == ["no_read_before_write"]


def test_reimport_in_a_later_cell_neither_violates_nor_propagates_staleness():
    cells = _cells([("A", "import json"), ("B", "s = json.dumps({})"), ("C", "import json\nt = json.dumps([])")])
    sim = _sim(cells)
    for c in cells:
        rec = sim.execute_cell(c)
        assert not rec.sdc_result.has_errors(), [e.error_type.value for e in rec.sdc_result.errors]
    rec = sim.execute_cell(cells[2])    # re-running the re-import must not mark B stale
    assert "B" not in rec.sdc_result.stale_cells
    assert not rec.sdc_result.has_errors()


def test_rejected_import_is_undone_on_rollback():
    cells = _cells([("A", "x = 1"), ("B", "print(x)"), ("C", "import json\nx = 2")])
    sim = _sim(cells, continue_on_violation=False)
    sim.execute_cell(cells[0]); sim.execute_cell(cells[1])
    rec = sim.execute_cell(cells[2])    # rejected: writes x after B read it
    assert rec.sdc_result.has_errors()
    assert "json" not in sim.namespace, "the rejected cell's import must not linger"


def test_rerunning_the_import_cell_does_not_stale_its_readers_but_still_guards_them():
    cells = _cells([("A", "import json"), ("B", "s = json.dumps({})"), ("X", "u = json.dumps(1)")])
    sim = _sim(cells)
    sim.execute_cell(cells[0]); sim.execute_cell(cells[1])
    rec = sim.execute_cell(cells[0])            # re-run the importer: an idempotent write
    assert "B" not in rec.sdc_result.stale_cells
    assert "json" in rec.tracking.writes        # still recorded as the importer...
    sim.enforcer.set_cell_order(["X", "A", "B"])
    rec = sim.execute_cell(cells[2])            # ...so a user placed above it is contamination
    assert [e.error_type.value for e in rec.sdc_result.errors] == ["no_read_before_write"]
