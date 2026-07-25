"""Validate the examples/progress notebooks against a live kernel.

Each notebook documents an expected run-until-clean outcome; these
tests drive that exact workflow (run all, meaningful edit, run all
stale) through run_actionable_cells and assert the documented result:
the positive notebooks (01-03) always terminate clean with no report,
and the negative notebooks (10-11) produce the potential
non-termination report at cell B.

Like test_session_integration.py, these need the FlowBook kernel and
take a few seconds each.
"""

import shutil
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from flowbook.mcp.server import run_actionable_cells
from flowbook.mcp.session import NotebookSession

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "progress"


def _ctx(session):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"session": session}
    return ctx


@pytest.fixture
def session():
    s = NotebookSession()
    yield s
    s.close()


def _load_example(session, tmp_path, name):
    """Copy the example to tmp (so runs do not dirty the repo) and load."""
    src = EXAMPLES / name
    dst = tmp_path / name
    shutil.copy(src, dst)
    session.load(str(dst))


def _cell_source(session, cell_id):
    _, cell = session._find_cell(cell_id)
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


class TestPositiveExamples:
    """The positive notebooks terminate clean, before and after edits."""

    @pytest.mark.parametrize(
        "name,edit_old,edit_new",
        [
            (
                "01_deterministic_pipeline.ipynb",
                "prices = [1.0, 2.0, 3.0, 4.0]",
                "prices = [5.0, 6.0, 7.0]",
            ),
            (
                "02_random_values.ipynb",
                "range(1000)",
                "range(2000)",
            ),
            (
                "03_random_dataframe.ipynb",
                "range(100)",
                "range(200)",
            ),
        ],
    )
    def test_terminates_clean(self, tmp_path, session, name, edit_old, edit_new):
        _load_example(session, tmp_path, name)

        result1 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result1
        assert "All clean!" in result1

        source = _cell_source(session, "A")
        assert edit_old in source, f"{name}: expected {edit_old!r} in cell A"
        edit = session.edit_cell("A", source.replace(edit_old, edit_new))
        assert edit["marked_stale"] is True

        result2 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result2
        assert "All clean!" in result2


class TestUnstableWrites:
    """10_unstable_writes: the deterministic flip is reported at B."""

    def test_reported_at_b(self, tmp_path, session):
        _load_example(session, tmp_path, "10_unstable_writes.ipynb")

        result1 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result1
        assert "All clean!" in result1

        source = _cell_source(session, "A")
        edit = session.edit_cell("A", source.replace("high = 100", "high = 101"))
        assert edit["marked_stale"] is True

        result2 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" in result2
        assert "@B" in result2.split("POTENTIAL NON-TERMINATION", 1)[1]
        # The sweep ran A, B, A, B and stopped — it did not cycle.
        assert "Ran 4 cells" in result2


class TestRandomBranch:
    """11_random_branch: reported at B once the branch actually flips."""

    def test_eventually_reported_at_b(self, tmp_path, session):
        _load_example(session, tmp_path, "11_random_branch.ipynb")

        result = run_actionable_cells(_ctx(session))
        assert "All clean!" in result

        # Re-edit A meaningfully each attempt (a fresh constant, so the
        # kernel never classifies the edit as cosmetic); each edited
        # sweep has a fair chance of flipping B's branch between its
        # reruns. P(no report in 25 sweeps) < 0.1%.
        base = (
            "import random\nscore = random.random()\n"
            "grade_a = 0\ngrade_b = 0\n"
        )
        for attempt in range(25):
            edit = session.edit_cell("A", base + f"_edit = {attempt}\n")
            assert edit["marked_stale"] is True
            result = run_actionable_cells(_ctx(session))
            if "POTENTIAL NON-TERMINATION" in result:
                assert "@B" in result.split("POTENTIAL NON-TERMINATION", 1)[1]
                return
            # Not flipped this time: the sweep must still have ended clean.
            assert "All clean!" in result
        pytest.fail("branch never flipped in 25 edited sweeps")
