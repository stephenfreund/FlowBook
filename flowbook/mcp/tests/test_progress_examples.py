"""Validate the examples/progress notebooks against a live kernel.

Each notebook documents an expected run-until-clean outcome; these
tests drive that exact workflow (run all, meaningful edit, run all
stale) through run_actionable_cells and assert the documented result:
the positive notebooks (01-03) always terminate clean with no warning,
and the negative notebooks (10-11) produce rerun warnings naming cell
B, with the deterministic flipper (10) stopping at the per-cell run
cap.

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
    """10_unstable_writes: the deterministic flip warns at B and the
    sweep stops at the per-cell run cap."""

    def test_warned_at_b_and_capped(self, tmp_path, session):
        _load_example(session, tmp_path, "10_unstable_writes.ipynb")

        result1 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result1
        assert "All clean!" in result1

        source = _cell_source(session, "A")
        edit = session.edit_cell("A", source.replace("high = 100", "high = 101"))
        assert edit["marked_stale"] is True

        result2 = run_actionable_cells(_ctx(session))
        # Every B rerun flips its write set and re-marks A: B's reruns
        # warn, and the sweep alternates A, B until A (which runs first
        # each cycle) exceeds the per-cell cap on its 11th run.
        assert "warning:" in result2
        warning_line = next(
            ln for ln in result2.splitlines() if ln.startswith("warning:")
        )
        assert "@B" in warning_line
        assert "POTENTIAL NON-TERMINATION" in result2
        tail = result2.split("POTENTIAL NON-TERMINATION", 1)[1]
        assert "@A" in tail
        assert "warnings above" in tail
        assert "Ran 21 cells" in result2


class TestRandomBranch:
    """11_random_branch: warned at B once the branch actually flips."""

    def test_eventually_warned_at_b(self, tmp_path, session):
        _load_example(session, tmp_path, "11_random_branch.ipynb")

        result = run_actionable_cells(_ctx(session))
        assert "All clean!" in result

        # Re-edit A meaningfully each attempt (a fresh constant, so the
        # kernel never classifies the edit as cosmetic); each edited
        # sweep has a fair chance of flipping B's branch between its
        # reruns. P(no warning in 25 sweeps) = 2^-25.
        base = (
            "import random\nscore = random.random()\n"
            "grade_a = 0\ngrade_b = 0\n"
        )
        for attempt in range(25):
            edit = session.edit_cell("A", base + f"_edit = {attempt}\n")
            assert edit["marked_stale"] is True
            result = run_actionable_cells(_ctx(session))
            if "warning:" in result:
                warning_line = next(
                    ln
                    for ln in result.splitlines()
                    if ln.startswith("warning:")
                )
                assert "@B" in warning_line
                # The sweep warns and continues: it ends clean unless
                # the branch flipped 10+ times in a row (p ≈ 2^-10).
                assert (
                    "All clean!" in result
                    or "POTENTIAL NON-TERMINATION" in result
                )
                return
            # No flip this time: the sweep must still have ended clean.
            assert "All clean!" in result
        pytest.fail("branch never flipped in 25 edited sweeps")
