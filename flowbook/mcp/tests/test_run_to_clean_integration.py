"""End-to-end tests for the RunToClean rerun check with a real kernel.

These drive run_actionable_cells against live FlowBook kernels (like
test_session_integration.py, they require the flowbook kernel to be
installable and take a few seconds each):

- a notebook containing a cell whose write set flips across reruns —
  driven deterministically through untracked module state, standing in
  for `if random() > 0.5:` — must be reported as potential
  non-termination, with the loop stopping at the flipping cell;
- value-nondeterministic notebooks with fixed footprints (random
  numbers, DataFrame recreation) must rerun to clean with no report.
"""

import json

import pytest
from unittest.mock import MagicMock

from flowbook.mcp.server import run_actionable_cells
from flowbook.mcp.session import NotebookSession


def _write_nb(tmp_path, cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "flowbook_kernel"}},
        "cells": [
            {
                "cell_type": "code",
                "source": src,
                "id": cid,
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            for cid, src in cells
        ],
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))
    return str(path)


def _ctx(session):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"session": session}
    return ctx


@pytest.fixture
def session():
    s = NotebookSession()
    yield s
    s.close()


# Cell A: writes a fixed set {builtins, flag, a, b}, but the VALUE of
# flag alternates on every execution via module state the tracker does
# not follow (exactly how random.random() behaves, but deterministic so
# the test is not flaky).
FLIPPER = (
    "import builtins\n"
    "flag = getattr(builtins, '_fb_flip', False)\n"
    "builtins._fb_flip = not flag\n"
    "a = 1\n"
    "b = 2\n"
)

# Cell B: an input-dependent footprint — writes {a} or {b} depending on
# the value of flag. This is the cell whose recorded results cannot be
# trusted to reproduce, and the one the report must name.
BRANCHER = "if flag:\n    a = 10\nelse:\n    b = 20\n"


class TestRerunAssignmentCell:
    def test_immediate_rerun_of_assignment_accepted(self, tmp_path, session):
        """Regression: IPython's prefilter looks up the first token of a
        cell in user_ns during transform_cell. With tracking enabled that
        lookup was recorded as a user read, so any rerun of ``y = x * 2``
        (whose write target already exists) was falsely rejected as
        reading and writing y (NoReadAndWrite). Reruns are the entire
        point of the staleness workflow, so this must be accepted."""
        path = _write_nb(
            tmp_path,
            [("A", "x = 1\n"), ("B", "y = x * 2\n")],
        )
        session.load(path)
        assert session.run_cell("A")["status"] == "ok"
        assert session.run_cell("B")["status"] == "ok"
        rerun = session.run_cell("B")
        assert rerun["status"] == "ok", rerun.get("outputs_text", "")


class TestRunToCleanEndToEnd:
    def test_footprint_flip_reported_at_culprit(self, tmp_path, session):
        """The nonterminating pattern: B's rerun drops a write, which
        backward-marks A; A's rerun re-marks B; B flips again. The loop
        must stop with a report at B instead of cycling."""
        path = _write_nb(tmp_path, [("A", FLIPPER), ("B", BRANCHER)])
        session.load(path)

        # Establish recorded state: flag=False, so B writes {b}.
        assert session.run_cell("A")["status"] == "ok"
        assert session.run_cell("B")["status"] == "ok"

        # Edit A meaningfully (the kernel ignores cosmetic edits) but
        # without changing its footprint; A becomes stale.
        edited = FLIPPER.replace("a = 1", "a = 3")
        result_edit = session.edit_cell("A", edited)
        assert result_edit["marked_stale"] is True

        result = run_actionable_cells(_ctx(session))

        # Sweep: A (flag→True, marks B), B (writes {a}, drops b →
        # backward-marks A), A again (same footprint: passes the check),
        # B again (flag→False, writes {b}: FLIP) — then stop.
        assert "POTENTIAL NON-TERMINATION" in result
        tail = result.split("POTENTIAL NON-TERMINATION", 1)[1]
        assert "@B" in tail
        assert "Ran 4 cells" in result

    def test_random_value_chain_reruns_clean(self, tmp_path, session):
        """Random values written to fixed variables: reruns cascade
        forward and finish clean with no report."""
        path = _write_nb(
            tmp_path,
            [
                ("A", "import random\nx = random.random()\n"),
                ("B", "y = x * 2\n"),
                ("C", "z = y + 1\n"),
            ],
        )
        session.load(path)

        # First sweep from all-unexecuted: three runs, clean.
        result1 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result1
        assert "All clean!" in result1

        # Edit A meaningfully; the rerun writes a different random value
        # into the same variable, and the staleness cascade reruns B and C.
        result_edit = session.edit_cell(
            "A", "import random\nx = random.uniform(0.0, 1.0)\n"
        )
        assert result_edit["marked_stale"] is True
        result2 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result2
        assert "All clean!" in result2
        assert "Ran 3 cells" in result2

    def test_dataframe_recreation_reruns_clean(self, tmp_path, session):
        """Rerunning df = pd.DataFrame(...) allocates a new object with
        a new loc_id; the name-level footprint is unchanged and the
        rerun must not be reported."""
        path = _write_nb(
            tmp_path,
            [
                (
                    "A",
                    "import pandas as pd\n"
                    "df = pd.DataFrame({'price': [1.0, 2.0, 3.0]})\n",
                ),
                ("B", "m = df['price'].mean()\n"),
            ],
        )
        session.load(path)

        result1 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result1
        assert "All clean!" in result1

        session.edit_cell(
            "A",
            "import pandas as pd\n"
            "df = pd.DataFrame({'price': [4.0, 5.0, 6.0]})\n",
        )
        result2 = run_actionable_cells(_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result2
        assert "All clean!" in result2
