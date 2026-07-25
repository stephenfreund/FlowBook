"""Validation suite for the RunToClean rerun check.

The run-until-clean loop (run_actionable_cells) implements the RunToClean
algorithm from the formal development: it executes the first stale cell in
notebook order, and if it executes a cell a second time and that run marks
an earlier cell stale (the only way staleness moves backward — a dropped
write that the earlier cell must restore), the loop may never terminate,
so it reports potential non-termination. Footprints are remembered only to
explain the report; the trigger never compares them.

These tests validate both directions:
- good programs are NOT rejected: deterministic chains, backward-mark
  repairs, cells writing varying values (random numbers) to fixed
  variables, DataFrame cells whose object identity (loc_id qualifier)
  changes across reruns, and even footprint drift that marks nothing
  backward;
- bad programs ARE caught: reruns that re-mark earlier cells (the
  nonterminating counterexample), with the loop stopping in a bounded
  number of runs and naming both the culprit and the re-marked cells.
"""

from unittest.mock import MagicMock, patch

from flowbook.mcp.server import run_actionable_cells
from flowbook.mcp.session import NotebookSession
from flowbook.util.footprint import (
    RunToCleanGuard,
    canonical_footprint,
    canonical_loc_key,
    format_footprint_change,
)


# ------------------------------------------------------------------
# Helpers (mirroring test_new_tools.py)
# ------------------------------------------------------------------

ORDER = ["A", "B", "C"]


def _make_mock_session(cell_order=None, continue_after_violation=False):
    session = MagicMock(spec=NotebookSession)
    session.is_loaded = True
    session.get_cell_order.return_value = cell_order or []
    session.cell_flowbook_meta = {}
    session._stale_cells = set()
    session.executed_cells = set()
    session.cell_status = {}
    session._continue_after_violation = continue_after_violation
    session.notebook_path = "/tmp/test.ipynb"
    session.log_event = MagicMock()
    return session


def _make_ctx(session):
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"session": session}
    return ctx


def _var(name):
    return {"type": "var", "name": name}


def _col(df, col, loc_id):
    return {"type": "col", "name": col, "qualifier": loc_id, "var_name": df}


def _meta(reads, writes, stale_cells=None, errors=None):
    return {
        "read_locs": reads,
        "write_locs": writes,
        "stale_cells": stale_cells or [],
        "errors": errors or [],
    }


def _scripted_session(script, cell_order):
    """Mock session whose run_cell replays scripted per-run metadata.

    ``script`` is a list of (cell_id, meta_or_None) pairs: each run_cell
    call consumes the next entry (asserting the expected cell) and
    installs the metadata. get_next_run_target follows the script, then
    reports all-clean.
    """
    session = _make_mock_session(cell_order=cell_order)
    session.get_next_run_target.side_effect = [cid for cid, _ in script] + [None]
    state = {"i": 0}

    def mock_run_cell(cell_id, **kwargs):
        idx = state["i"]
        assert idx < len(script), "run_cell called more times than scripted"
        expected_cid, meta = script[idx]
        assert cell_id == expected_cid, (
            f"run {idx}: expected {expected_cid}, got {cell_id}"
        )
        state["i"] += 1
        if meta is not None:
            session.cell_flowbook_meta[cell_id] = meta
        return {"cell_id": cell_id, "status": "ok", "outputs_text": ""}

    session.run_cell.side_effect = mock_run_cell
    session.get_status.return_value = {
        "stale_cells": {},
        "violations": [],
        "executed": len(cell_order),
        "total_code_cells": len(cell_order),
    }
    return session


# ==================================================================
# Canonicalization unit tests (used for report messages)
# ==================================================================


class TestCanonicalFootprint:
    def test_var_key(self):
        assert canonical_loc_key(_var("x")) == ("var", None, "x")

    def test_loc_id_qualifier_dropped(self):
        """Object identity changes across reruns; the key must not."""
        a = canonical_loc_key(_col("df", "price", 101))
        b = canonical_loc_key(_col("df", "price", 202))
        assert a == b == ("col", "df", "price")

    def test_string_qualifier_kept(self):
        """A string qualifier is a variable name, not an object id."""
        loc = {"type": "col", "name": "price", "qualifier": "df"}
        assert canonical_loc_key(loc) == ("col", "df", "price")

    def test_string_and_locref_forms_agree(self):
        via_str = {"type": "col", "name": "price", "qualifier": "df"}
        via_ref = _col("df", "price", 7)
        assert canonical_loc_key(via_str) == canonical_loc_key(via_ref)

    def test_footprint_set(self):
        fp = canonical_footprint([_var("x"), _var("x"), _col("df", "a", 1)])
        assert fp == {("var", None, "x"), ("col", "df", "a")}


# ==================================================================
# Guard unit tests
# ==================================================================


class TestRunToCleanGuard:
    def test_first_run_never_reports(self):
        """A first execution may mark backward freely (repairs after an
        edit legitimately do)."""
        guard = RunToCleanGuard()
        meta = _meta([], [_var("y")], stale_cells=["A"])
        assert guard.note_run("C", meta, ORDER) is None

    def test_rerun_forward_marks_only_not_reported(self):
        guard = RunToCleanGuard()
        guard.note_run("B", _meta([], [_var("y")], stale_cells=["C"]), ORDER)
        assert (
            guard.note_run("B", _meta([], [_var("y")], stale_cells=["C"]), ORDER)
            is None
        )

    def test_rerun_backward_mark_reported(self):
        guard = RunToCleanGuard()
        guard.note_run("C", _meta([], [_var("b")]), ORDER)
        report = guard.note_run(
            "C", _meta([], [_var("a")], stale_cells=["B"]), ORDER
        )
        assert report is not None
        assert report["backward_stale"] == ["B"]
        # Footprint change is included to explain the report.
        assert report["prev_writes"] == ["b"]
        assert report["new_writes"] == ["a"]

    def test_footprint_drift_without_backward_mark_not_reported(self):
        """Set changes that mark nothing backward re-record silently:
        staleness still only moves forward, so termination holds."""
        guard = RunToCleanGuard()
        guard.note_run("B", _meta([_var("x")], [_var("y")]), ORDER)
        assert (
            guard.note_run(
                "B", _meta([_var("x")], [_var("z")], stale_cells=["C"]), ORDER
            )
            is None
        )

    def test_report_without_footprint_metadata(self):
        """The trigger needs only staleness marks — it works even when
        read/write tracking is unavailable."""
        guard = RunToCleanGuard()
        guard.note_run("C", {"stale_cells": []}, ORDER)
        report = guard.note_run("C", {"stale_cells": ["A", "B"]}, ORDER)
        assert report is not None
        assert report["backward_stale"] == ["A", "B"]
        assert "prev_writes" not in report

    def test_missing_metadata_never_reports(self):
        guard = RunToCleanGuard()
        guard.note_run("C", _meta([], [_var("a")]), ORDER)
        assert guard.note_run("C", None, ORDER) is None

    def test_loc_id_churn_does_not_pollute_report(self):
        """df recreation changes loc_ids; the explanation must not show
        a phantom footprint change."""
        guard = RunToCleanGuard()
        guard.note_run(
            "C", _meta([_col("df", "a", 101)], [_col("df", "b", 101)]), ORDER
        )
        report = guard.note_run(
            "C",
            _meta(
                [_col("df", "a", 202)], [_col("df", "b", 202)],
                stale_cells=["B"],
            ),
            ORDER,
        )
        assert report is not None  # the backward mark still reports
        assert "prev_writes" not in report  # but no footprint change shown

    def test_cells_tracked_independently(self):
        guard = RunToCleanGuard()
        guard.note_run("B", _meta([], [_var("y")]), ORDER)
        assert (
            guard.note_run("C", _meta([], [_var("z")], stale_cells=["A"]), ORDER)
            is None  # first execution of C
        )

    def test_format_change_spells_out_sets(self):
        guard = RunToCleanGuard()
        guard.note_run("C", _meta([_var("x")], [_var("a")]), ORDER)
        report = guard.note_run(
            "C", _meta([], [_var("b")], stale_cells=["B"]), ORDER
        )
        text = format_footprint_change(report)
        assert text == (
            "the previous run read x and wrote a, "
            "but this run read nothing and wrote b"
        )


# ==================================================================
# Loop scenario tests (good programs must not be rejected)
# ==================================================================


class TestRunToCleanLoopAccepts:
    def test_deterministic_chain_runs_clean(self):
        """Edit at the top, staleness flows forward, each cell runs once."""
        script = [
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("y")], stale_cells=["C"])),
            ("C", _meta([_var("y")], [_var("z")])),
        ]
        session = _scripted_session(script, ["A", "B", "C"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 3 cells" in result
        assert "All clean!" in result

    def test_backward_mark_repair_not_flagged(self):
        """A first-in-sweep run may drop a write and mark an earlier
        cell stale (the repair path after an edit); the repair rerun
        marks nothing backward and is not flagged."""
        script = [
            # C's first run drops a write, backward-marking A...
            ("C", _meta([], [_var("b")], stale_cells=["A"])),
            # ...A's repair reproduces its behavior, marks nothing back.
            ("A", _meta([], [_var("a"), _var("x")], stale_cells=["C"])),
            ("C", _meta([], [_var("b")])),
        ]
        session = _scripted_session(script, ["A", "B", "C"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 3 cells" in result

    def test_random_cell_rerun_not_flagged(self):
        """x = random() reruns write different values, same variables."""
        script = [
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("y")])),
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("y")])),
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 4 cells" in result

    def test_footprint_drift_without_backward_marks_accepted(self):
        """A rerun whose sets changed but marked nothing backward is
        re-recorded silently — staleness still flows only forward."""
        script = [
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("y")])),
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("z")])),  # drift, no back-mark
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result

    def test_dataframe_recreation_not_flagged(self):
        """Rerunning df = pd.read_csv(...) changes the loc_id qualifier
        of every column location; nothing is marked backward."""
        script = [
            ("A", _meta([], [_col("df", "price", 101)], stale_cells=["B"])),
            ("B", _meta([_col("df", "price", 101)], [_var("m")])),
            ("A", _meta([], [_col("df", "price", 202)], stale_cells=["B"])),
            ("B", _meta([_col("df", "price", 202)], [_var("m")])),
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result

    def test_edited_cell_new_footprint_not_flagged(self):
        """An edited cell runs once per sweep; its first run this sweep
        may change behavior freely (E is per-invocation)."""
        session = _scripted_session(
            [("A", _meta([], [_var("q")]))], ["A"]
        )
        session.cell_flowbook_meta["A"] = _meta([], [_var("old")])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 1 cell" in result


# ==================================================================
# Loop scenario tests (bad programs must be caught)
# ==================================================================


class TestRunToCleanLoopRejects:
    def test_flip_flop_flagged_and_terminates(self):
        """The paper's nonterminating counterexample: a cell whose write
        set alternates keeps backward-marking an earlier cell. The loop
        must stop with a report at the flipping cell, naming the cell it
        re-marked."""
        script = [
            ("C", _meta([], [_var("b")], stale_cells=["B"])),
            ("B", _meta([], [_var("a"), _var("b")], stale_cells=["C"])),
            ("C", _meta([], [_var("a")], stale_cells=["B"])),  # re-marks B
        ]
        session = _scripted_session(script, ["A", "B", "C"])
        session.get_next_run_target.side_effect = ["C", "B", "C", "B", "C"]
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" in result
        tail = result.split("POTENTIAL NON-TERMINATION", 1)[1]
        assert "@C" in tail  # the culprit
        assert "@B" in tail  # the re-marked cell
        # Stopped at the flip: exactly 3 runs, not the whole tail.
        assert session.run_cell.call_count == 3

    def test_input_dependent_footprint_downstream_of_random_flagged(self):
        """Random values feeding a cell whose write set depends on its
        input: the rerun that drops a write re-marks its owner."""
        script = [
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("a")], stale_cells=["A"])),
            ("A", _meta([], [_var("x")], stale_cells=["B"])),
            ("B", _meta([_var("x")], [_var("b")], stale_cells=["A"])),
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" in result
        assert session.run_cell.call_count == 4

    def test_untracked_cells_hit_backstop_cap(self):
        """Cells with no metadata at all cannot trigger the check; the
        loop still terminates via the n(n+2) backstop."""
        session = _make_mock_session(cell_order=["A"])
        session.get_next_run_target.return_value = "A"  # forever stale
        session.run_cell.return_value = {
            "cell_id": "A", "status": "ok", "outputs_text": ""
        }
        session.get_status.return_value = {
            "stale_cells": {"A": []},
            "violations": [],
            "executed": 1,
            "total_code_cells": 1,
        }
        result = run_actionable_cells(_make_ctx(session))
        assert "Stopped after" in result
        assert session.run_cell.call_count == max(25, 1 * (1 + 3))


# ==================================================================
# get_next_run_target ordering (first-stale in document order)
# ==================================================================


def _make_notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}


def _make_code_cell(cell_id, source):
    return {
        "id": cell_id,
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }


def _bare_session(cells):
    session = NotebookSession()
    session.notebook = _make_notebook(cells)
    session.notebook_path = "/abs/path/test.ipynb"
    return session


def _quiet(session):
    """Patch out kernel/contents refreshes for bare sessions."""
    return (
        patch.object(NotebookSession, "_refresh_from_contents_api"),
        patch.object(NotebookSession, "_poll_iopub"),
    )


class TestGetNextRunTarget:
    def _target(self, session):
        p1, p2 = _quiet(session)
        with p1, p2:
            return session.get_next_run_target()

    def test_unexecuted_earlier_beats_stale_later(self):
        """The formal algorithm counts unexecuted cells as stale and runs
        the first one in document order, so every cell sees its final
        inputs before its first run (no false rerun flags)."""
        session = _bare_session(
            [_make_code_cell("A", "x = 1"), _make_code_cell("B", "y = x")]
        )
        session.executed_cells = {"B"}
        session._stale_cells = {"B"}  # stale, later
        # A unexecuted, earlier
        assert self._target(session) == "A"

    def test_stale_earlier_beats_unexecuted_later(self):
        session = _bare_session(
            [_make_code_cell("A", "x = 1"), _make_code_cell("B", "y = x")]
        )
        session.executed_cells = {"A"}
        session._stale_cells = {"A"}
        assert self._target(session) == "A"

    def test_error_cell_first(self):
        session = _bare_session(
            [_make_code_cell("A", "x = 1"), _make_code_cell("B", "boom")]
        )
        session.executed_cells = {"A", "B"}
        session.cell_status = {"B": "error"}
        session._stale_cells = {"A"}
        assert self._target(session) == "B"

    def test_empty_unexecuted_cells_skipped(self):
        session = _bare_session(
            [_make_code_cell("A", "   "), _make_code_cell("B", "y = 1")]
        )
        assert self._target(session) == "B"

    def test_all_clean_returns_none(self):
        session = _bare_session([_make_code_cell("A", "x = 1")])
        session.executed_cells = {"A"}
        assert self._target(session) is None

    def test_empty_stale_cell_skipped(self):
        """An empty cell marked stale must not be a run target: running
        it is a no-op that never clears its staleness, so returning it
        would loop forever. (Seen in practice: JupyterLab appends an
        empty cell when the last cell is run with shift+enter.)"""
        session = _bare_session(
            [_make_code_cell("A", "x = 1"), _make_code_cell("B", "   ")]
        )
        session.executed_cells = {"A"}
        session._stale_cells = {"B"}
        assert self._target(session) is None
