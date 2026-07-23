"""Validation suite for the RunToClean rerun check.

The run-until-clean loop (run_actionable_cells) implements the RunToClean
algorithm from the formal development: it executes the first stale cell in
notebook order, and if it executes a cell a second time, the run's read
and write sets must match those recorded by the previous run; otherwise
the loop may never terminate, so it reports potential non-termination.

These tests validate both directions:
- good programs are NOT rejected: deterministic chains, backward-mark
  repairs, cells writing varying values (random numbers) to fixed
  variables, DataFrame cells whose object identity (loc_id qualifier)
  changes across reruns;
- bad programs ARE caught: cells whose write sets flip across reruns
  (the nonterminating counterexample), with the loop stopping in a
  bounded number of runs and naming the culprit cell.
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


def _meta(reads, writes, errors=None):
    return {"read_locs": reads, "write_locs": writes, "errors": errors or []}


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
# Guard unit tests
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


class TestRunToCleanGuard:
    def test_first_run_never_flags(self):
        guard = RunToCleanGuard()
        assert guard.note_run("A", _meta([_var("x")], [_var("y")])) is None

    def test_identical_rerun_not_flagged(self):
        guard = RunToCleanGuard()
        guard.note_run("A", _meta([_var("x")], [_var("y")]))
        assert guard.note_run("A", _meta([_var("x")], [_var("y")])) is None

    def test_random_values_fixed_variables_not_flagged(self):
        """x = random() writes the same variable set on every run."""
        guard = RunToCleanGuard()
        guard.note_run("A", _meta([], [_var("x")]))
        assert guard.note_run("A", _meta([], [_var("x")])) is None

    def test_loc_id_churn_not_flagged(self):
        """df = pd.read_csv(...) allocates a new object every run."""
        guard = RunToCleanGuard()
        guard.note_run("A", _meta([_col("df", "a", 101)], [_col("df", "b", 101)]))
        assert (
            guard.note_run(
                "A", _meta([_col("df", "a", 202)], [_col("df", "b", 202)])
            )
            is None
        )

    def test_write_set_flip_flagged(self):
        guard = RunToCleanGuard()
        guard.note_run("A", _meta([], [_var("a")]))
        change = guard.note_run("A", _meta([], [_var("b")]))
        assert change is not None
        assert change["writes_added"] == ["b"]
        assert change["writes_removed"] == ["a"]
        assert change["reads_added"] == []

    def test_read_set_change_flagged(self):
        guard = RunToCleanGuard()
        guard.note_run("A", _meta([_var("x")], [_var("y")]))
        change = guard.note_run("A", _meta([_var("z")], [_var("y")]))
        assert change is not None
        assert change["reads_added"] == ["z"]
        assert change["reads_removed"] == ["x"]

    def test_missing_metadata_skips_and_forgets(self):
        """No tracking info: nothing to compare, and the stale record is
        dropped so a later run is not compared against outdated data."""
        guard = RunToCleanGuard()
        guard.note_run("A", _meta([], [_var("a")]))
        assert guard.note_run("A", None) is None
        assert guard.note_run("A", {"errors": []}) is None
        # Record was dropped: this differing run counts as a first run.
        assert guard.note_run("A", _meta([], [_var("b")])) is None
        # But from here on, comparisons resume.
        assert guard.note_run("A", _meta([], [_var("c")])) is not None

    def test_format_change(self):
        text = format_footprint_change(
            {"writes_added": ["b"], "writes_removed": ["a"],
             "reads_added": [], "reads_removed": []}
        )
        assert "writes +{b}" in text and "writes -{a}" in text


# ==================================================================
# Loop scenario tests (good programs must not be rejected)
# ==================================================================


class TestRunToCleanLoopAccepts:
    def test_deterministic_chain_runs_clean(self):
        """Edit at the top, staleness flows forward, each cell runs once."""
        script = [
            ("A", _meta([], [_var("x")])),
            ("B", _meta([_var("x")], [_var("y")])),
            ("C", _meta([_var("y")], [_var("z")])),
        ]
        session = _scripted_session(script, ["A", "B", "C"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 3 cells" in result
        assert "All clean!" in result

    def test_backward_mark_repair_not_flagged(self):
        """A dropped write marks an earlier cell stale; its rerun
        reproduces its recorded sets (a repair) and is not flagged."""
        script = [
            # C's run drops a write, backward-marking A...
            ("C", _meta([], [_var("b")])),
            # ...A reruns with its recorded footprint (same sets)...
            ("A", _meta([], [_var("a"), _var("x")])),
            ("A", _meta([], [_var("a"), _var("x")])),
        ]
        session = _scripted_session(script, ["A", "B", "C"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 3 cells" in result

    def test_random_cell_rerun_not_flagged(self):
        """x = random() reruns write different values, same variables."""
        script = [
            ("A", _meta([], [_var("x")])),
            ("B", _meta([_var("x")], [_var("y")])),
            ("A", _meta([], [_var("x")])),
            ("B", _meta([_var("x")], [_var("y")])),
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result
        assert "Ran 4 cells" in result

    def test_dataframe_recreation_not_flagged(self):
        """Rerunning df = pd.read_csv(...) changes the loc_id qualifier
        of every column location; name-level identity is unchanged."""
        script = [
            ("A", _meta([], [_col("df", "price", 101)])),
            ("B", _meta([_col("df", "price", 101)], [_var("m")])),
            ("A", _meta([], [_col("df", "price", 202)])),
            ("B", _meta([_col("df", "price", 202)], [_var("m")])),
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" not in result

    def test_edited_cell_new_footprint_not_flagged(self):
        """An edited cell runs once per sweep; its first run this sweep
        records new sets freely (E is per-invocation)."""
        session = _scripted_session(
            [("A", _meta([], [_var("q")]))], ["A"]
        )
        # Simulate a previous sweep having recorded a different footprint.
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
        must stop with a report at the flipping cell."""
        script = [
            ("C", _meta([], [_var("b")])),   # first run: writes {b}
            ("B", _meta([], [_var("a"), _var("b")])),
            ("C", _meta([], [_var("a")])),   # rerun: writes {a} — flip!
        ]
        session = _scripted_session(script, ["A", "B", "C"])
        # get_next_run_target would keep returning cells forever if the
        # loop did not stop itself; simulate that with a long script tail.
        session.get_next_run_target.side_effect = ["C", "B", "C", "B", "C"]
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" in result
        assert "[C]" in result or "@C" in result
        # Stopped at the flip: exactly 3 runs, not the whole tail.
        assert session.run_cell.call_count == 3

    def test_input_dependent_footprint_downstream_of_random_flagged(self):
        """Random values feeding a cell whose write set depends on its
        input: each individual step looks justified, but the rerun check
        catches the second footprint."""
        script = [
            ("A", _meta([], [_var("x")])),                 # x = random()
            ("B", _meta([_var("x")], [_var("a")])),        # if x>0: a=1
            ("A", _meta([], [_var("x")])),                 # rerun, new x
            ("B", _meta([_var("x")], [_var("b")])),        # now writes b!
        ]
        session = _scripted_session(script, ["A", "B"])
        result = run_actionable_cells(_make_ctx(session))
        assert "POTENTIAL NON-TERMINATION" in result
        assert session.run_cell.call_count == 4

    def test_untracked_cells_hit_backstop_cap(self):
        """Cells with no tracking metadata cannot be checked; the loop
        still terminates via the n(n+2) backstop."""
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
