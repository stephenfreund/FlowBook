"""Tests for FlowBookTools — unified tool logic."""

import pytest
from unittest.mock import MagicMock, PropertyMock

from flowbook.tools.tools import FlowBookTools


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_session(cell_order=None, cells=None, stale=None, executed=None,
                  meta=None, cell_status=None, continue_after=False):
    """Create a mock NotebookSession with configurable state."""
    session = MagicMock()
    session.is_loaded = True
    session.notebook_path = "/tmp/test.ipynb"
    session._continue_after_violation = continue_after
    session._stale_cells = set(stale or [])
    session.executed_cells = set(executed or [])
    session.cell_flowbook_meta = dict(meta or {})
    session.cell_status = dict(cell_status or {})

    if cell_order is None and cells is not None:
        order = [c["id"] for c in cells if c.get("cell_type") == "code"]
    else:
        order = cell_order if cell_order is not None else ["aa", "bb", "cc"]
    session.get_cell_order.return_value = order

    if cells is None:
        cells = [
            {"id": cid, "cell_type": "code", "source": f"# cell {cid}", "outputs": []}
            for cid in order
        ]
    notebook = {"cells": cells}
    session.notebook = notebook

    def find_cell(cid):
        for i, c in enumerate(cells):
            if c["id"] == cid:
                return i, c
        raise ValueError(f"Cell {cid} not found")
    session._find_cell.side_effect = find_cell

    return session


@pytest.fixture
def session():
    return _make_session()


@pytest.fixture
def tools(session):
    return FlowBookTools(session)


# ==================================================================
# Cell reference resolution
# ==================================================================

class TestResolveRef:
    def test_at_label(self, tools):
        assert tools._resolve_ref("@A") == "aa"
        assert tools._resolve_ref("@B") == "bb"
        assert tools._resolve_ref("@C") == "cc"

    def test_plain_alpha(self, tools):
        assert tools._resolve_ref("A") == "aa"
        assert tools._resolve_ref("B") == "bb"

    def test_numeric_string(self, tools):
        assert tools._resolve_ref("0") == "aa"
        assert tools._resolve_ref("2") == "cc"

    def test_cell_id_direct(self, tools):
        assert tools._resolve_ref("bb") == "bb"

    def test_out_of_range_raises(self, tools):
        with pytest.raises(ValueError, match="out of range"):
            tools._resolve_ref("@Z")

    def test_invalid_raises(self, tools):
        with pytest.raises(ValueError):
            tools._resolve_ref("")

    def test_case_insensitive_alpha(self, tools):
        assert tools._resolve_ref("@a") == "aa"
        assert tools._resolve_ref("@b") == "bb"


class TestLabel:
    def test_known_cell(self, tools):
        assert tools._label("aa") == "@A"
        assert tools._label("cc") == "@C"

    def test_unknown_cell(self, tools):
        assert tools._label("zz") == "zz"


# ==================================================================
# Status & metadata tools
# ==================================================================

class TestGetAllCellSources:
    def test_returns_all_cells(self, tools, session):
        session.refresh_from_jupyter = MagicMock()
        result = tools.get_all_cell_sources()
        assert "@A" in result
        assert "@B" in result
        assert "@C" in result
        assert "[aa]" in result

    def test_empty_notebook(self):
        session = _make_session(cell_order=[], cells=[])
        t = FlowBookTools(session)
        session.refresh_from_jupyter = MagicMock()
        assert t.get_all_cell_sources() == "No code cells in notebook."

    def test_shows_status(self):
        session = _make_session(executed={"aa"}, stale={"bb"})
        session.refresh_from_jupyter = MagicMock()
        t = FlowBookTools(session)
        result = t.get_all_cell_sources()
        assert "(ok)" in result
        assert "(stale)" in result


class TestReadCell:
    def test_delegates_to_session(self, tools, session):
        session.get_cell.return_value = {
            "cell_id": "aa",
            "source": "x = 1",
            "status": "ok",
            "outputs_text": "1",
        }
        result = tools.read_cell("@A")
        session.get_cell.assert_called_once_with("aa")
        assert "@A" in result
        assert "x = 1" in result

    def test_accepts_cell_id(self, tools, session):
        session.get_cell.return_value = {
            "cell_id": "bb",
            "source": "y = 2",
            "status": "ok",
            "outputs_text": "",
        }
        result = tools.read_cell("bb")
        session.get_cell.assert_called_once_with("bb")


class TestGetNextActionableCell:
    def test_all_clean(self, tools, session):
        session.get_next_actionable.return_value = None
        assert tools.get_next_actionable_cell() == "All clean."

    def test_returns_cell(self, tools, session):
        session.get_next_actionable.return_value = {
            "cell_id": "bb",
            "reason": "stale",
            "source": "y = 2",
        }
        result = tools.get_next_actionable_cell()
        assert "@B" in result
        assert "stale" in result


class TestGetFlowbookMetadata:
    def test_no_metadata(self, tools, session):
        result = tools.get_flowbook_metadata("@A")
        assert "not been executed" in result

    def test_with_metadata(self):
        meta = {"aa": {"read_locs": [{"type": "var", "name": "x"}],
                        "write_locs": [], "errors": [], "stale_cells": []}}
        session = _make_session(meta=meta)
        t = FlowBookTools(session)
        result = t.get_flowbook_metadata("@A")
        assert "Reads: x" in result


class TestGetStatus:
    def test_delegates(self, tools, session):
        session.get_status.return_value = {
            "executed": 2,
            "total_code_cells": 3,
            "violations": [],
            "stale_cells": {},
        }
        result = tools.get_status()
        assert "2/3 executed" in result


# ==================================================================
# Cell editing tools
# ==================================================================

class TestEditCellSource:
    def test_delegates(self, tools, session):
        session.edit_cell.return_value = {
            "cell_id": "aa",
            "marked_stale": True,
            "new_source_preview": "x = 2",
        }
        result = tools.edit_cell_source("@A", "x = 2")
        session.edit_cell.assert_called_once_with("aa", "x = 2")
        assert "(marked stale)" in result


# ==================================================================
# Execution tools
# ==================================================================

class TestRunCell:
    def test_delegates(self, tools, session):
        session.run_cell.return_value = {
            "cell_id": "aa",
            "status": "ok",
            "outputs_text": "done",
        }
        result = tools.run_cell("@A")
        session.run_cell.assert_called_once_with("aa")
        assert "@A" in result
        assert "ok" in result


class TestRunAllCells:
    def test_ok(self, tools, session):
        session.run_all.return_value = {
            "total_executed": 3,
            "total_code_cells": 3,
            "violations": [],
            "stale_cells": [],
            "status": "ok",
        }
        result = tools.run_all_cells()
        assert "3/3" in result
        assert "0 violations" in result

    def test_with_error(self, tools, session):
        session.run_all.return_value = {
            "total_executed": 1,
            "total_code_cells": 3,
            "violations": [],
            "stale_cells": [],
            "status": "error",
        }
        result = tools.run_all_cells()
        assert "stopped on error" in result


class TestRunFrom:
    def test_delegates(self, tools, session):
        session.run_from.return_value = {
            "executed": ["aa", "bb"],
            "violations": [],
            "stale_remaining": 0,
            "skipped": 1,
            "error_cell": None,
        }
        result = tools.run_from("@A")
        session.run_from.assert_called_once_with("aa")
        assert "2 cells" in result
        assert "1 clean skipped" in result


class TestRunActionableCell:
    def test_all_clean(self, tools, session):
        session.get_next_actionable_cell_id.return_value = None
        result = tools.run_actionable_cell()
        assert "All clean" in result

    def test_runs_next(self, tools, session):
        session.get_next_actionable_cell_id.return_value = "bb"
        session.run_cell.return_value = {
            "cell_id": "bb",
            "status": "ok",
            "outputs_text": "",
        }
        result = tools.run_actionable_cell()
        session.run_cell.assert_called_once_with("bb")
        assert "@B" in result


class TestRunActionableCells:
    def test_all_clean_initially(self, tools, session):
        session.get_next_actionable_cell_id.return_value = None
        session.get_status.return_value = {
            "executed": 3, "total_code_cells": 3,
            "violations": [], "stale_cells": {},
        }
        result = tools.run_actionable_cells()
        assert "Ran 0 cells" in result

    def test_runs_until_clean(self, tools, session):
        call_count = [0]
        def next_actionable():
            call_count[0] += 1
            if call_count[0] <= 2:
                return ["aa", "bb"][call_count[0] - 1]
            return None
        session.get_next_actionable_cell_id.side_effect = next_actionable
        session.run_cell.return_value = {
            "cell_id": "aa", "status": "ok", "outputs_text": "",
        }
        session.get_status.return_value = {
            "executed": 3, "total_code_cells": 3,
            "violations": [], "stale_cells": {},
        }
        result = tools.run_actionable_cells()
        assert "Ran 2 cells" in result
        assert "All clean!" in result

    def test_stops_on_error(self, tools, session):
        session.get_next_actionable_cell_id.return_value = "aa"
        session.run_cell.return_value = {
            "cell_id": "aa", "status": "error", "outputs_text": "",
            "error_message": "NameError",
        }
        session.get_status.return_value = {
            "executed": 1, "total_code_cells": 3,
            "violations": [], "stale_cells": {},
        }
        result = tools.run_actionable_cells()
        assert "error at @A" in result


    def test_empty_cells_dont_cause_infinite_loop(self, tools, session):
        """Empty cells should be executed by the kernel (not skipped) and marked clean."""
        call_count = [0]
        def next_actionable():
            call_count[0] += 1
            if call_count[0] <= 3:
                # Return cells: aa (normal), bb (normal), cc (empty but kernel handles it)
                return ["aa", "bb", "cc"][call_count[0] - 1]
            return None
        session.get_next_actionable_cell_id.side_effect = next_actionable
        session.run_cell.return_value = {
            "cell_id": "aa", "status": "ok", "outputs_text": "",
        }
        session.get_status.return_value = {
            "executed": 3, "total_code_cells": 3,
            "violations": [], "stale_cells": {},
        }
        result = tools.run_actionable_cells()
        # All 3 cells should be run (including the empty one)
        assert "Ran 3 cells" in result
        assert session.run_cell.call_count == 3

    def test_safety_limit_prevents_infinite_loop(self, tools, session):
        """If get_next_actionable keeps returning the same cell, the loop stops at 500."""
        session.get_next_actionable_cell_id.return_value = "aa"
        session.run_cell.return_value = {
            "cell_id": "aa", "status": "ok", "outputs_text": "",
        }
        session.cell_flowbook_meta = {}
        session.get_status.return_value = {
            "executed": 1, "total_code_cells": 3,
            "violations": [], "stale_cells": {"bb": [], "cc": []},
        }
        result = tools.run_actionable_cells()
        # Should stop at 500 iterations, not run forever
        assert session.run_cell.call_count == 500


class TestContinueAfterViolation:
    def test_enable(self, tools, session):
        result = tools.continue_after_violation(True)
        session.set_continue_after_violation.assert_called_once_with(True)
        assert "continue" in result

    def test_disable(self, tools, session):
        result = tools.continue_after_violation(False)
        session.set_continue_after_violation.assert_called_once_with(False)
        assert "reject" in result


# ==================================================================
# Refactoring tools
# ==================================================================

class TestAlphaRename:
    def test_no_occurrences(self, tools, session):
        session.alpha_rename.return_value = {
            "old_name": "x", "new_name": "y",
            "total_modified": 0, "modified_cells": [],
        }
        result = tools.alpha_rename("@A", "x", "y")
        assert "No occurrences" in result

    def test_with_modifications(self, tools, session):
        session.alpha_rename.return_value = {
            "old_name": "x", "new_name": "y",
            "total_modified": 2, "modified_cells": ["aa", "cc"],
        }
        result = tools.alpha_rename("@A", "x", "y")
        assert "Renamed 'x'" in result
        assert "@A [aa]" in result
        assert "@C [cc]" in result


class TestRemoveInplace:
    def test_success(self, tools, session):
        session.remove_inplace.return_value = {
            "cell_id": "aa", "variable": "df",
            "methods_fixed": ["drop", "fillna"],
            "new_source": "df = df.drop()",
        }
        result = tools.remove_inplace("@A", "df")
        assert "Removed inplace=True" in result
        assert "drop, fillna" in result

    def test_error(self, tools, session):
        session.remove_inplace.return_value = {"error": "syntax error"}
        result = tools.remove_inplace("@A", "df")
        assert "Error: syntax error" in result


class TestInsertDeepcopy:
    def test_success(self, tools, session):
        session.insert_deepcopy.return_value = {
            "cell_id": "aa", "variable": "df", "new_name": "df_copy",
            "modified_downstream": ["bb", "cc"],
        }
        result = tools.insert_deepcopy("@A", "df")
        assert "df_copy" in result
        assert "@B [bb]" in result


class TestMarkDiagnostic:
    def test_already_diagnostic(self, tools, session):
        session.mark_diagnostic.return_value = {"already_diagnostic": True}
        result = tools.mark_diagnostic("@A")
        assert "already marked" in result

    def test_success(self, tools, session):
        session.mark_diagnostic.return_value = {
            "already_diagnostic": False,
            "new_source_preview": "%diagnostic\nx = 1",
        }
        result = tools.mark_diagnostic("@A")
        assert "Marked cell @A" in result


class TestMergeCells:
    def test_success(self, tools, session):
        session.merge_cells.return_value = {
            "merged_cell_id": "aa",
            "cells_removed": ["bb"],
            "new_source_preview": "x = 1\ny = 2",
        }
        result = tools.merge_cells(["@A", "@B"])
        session.merge_cells.assert_called_once_with(["aa", "bb"])
        assert "Merged into cell @A" in result
        assert "@B [bb]" in result


class TestMoveCell:
    def test_success(self, tools, session):
        session.move_cell.return_value = {
            "cell_id": "aa",
            "moved_after": "cc",
            "new_cell_order": ["bb", "cc", "aa"],
        }
        result = tools.move_cell("@A", "@C")
        session.move_cell.assert_called_once_with("aa", "cc")
        assert "Moved cell @A" in result
        assert "after @C" in result


# ==================================================================
# Checkpoint tools
# ==================================================================

class TestCheckpoint:
    def test_creates(self, tools, session):
        session.checkpoint.return_value = "ckpt_abc123"
        result = tools.checkpoint()
        assert "ckpt_abc123" in result


class TestRestore:
    def test_restores(self, tools, session):
        session.restore.return_value = {
            "cells_restored": 3,
            "changed_cells": ["aa", "bb"],
        }
        result = tools.restore("ckpt_abc123")
        session.restore.assert_called_once_with("ckpt_abc123")
        assert "Restored 3 cells" in result


class TestListCheckpoints:
    def test_empty(self, tools, session):
        session.list_checkpoints.return_value = []
        assert "No checkpoints" in tools.list_checkpoints()

    def test_with_checkpoints(self, tools, session):
        session.list_checkpoints.return_value = [
            {"checkpoint_id": "ckpt_1", "cell_count": 3},
            {"checkpoint_id": "ckpt_2", "cell_count": 3},
        ]
        result = tools.list_checkpoints()
        assert "ckpt_1" in result
        assert "ckpt_2" in result


# ==================================================================
# Lifecycle tools
# ==================================================================

class TestSaveNotebook:
    def test_default_path(self, tools, session):
        session.save.return_value = "/tmp/test.ipynb"
        result = tools.save_notebook()
        session.save.assert_called_once_with(None)
        assert "Saved: /tmp/test.ipynb" in result

    def test_custom_path(self, tools, session):
        session.save.return_value = "/tmp/out.ipynb"
        result = tools.save_notebook("/tmp/out.ipynb")
        session.save.assert_called_once_with("/tmp/out.ipynb")


class TestGetNotebookPath:
    def test_loaded(self, tools, session):
        assert tools.get_notebook_path() == "/tmp/test.ipynb"

    def test_not_loaded(self):
        session = _make_session()
        session.is_loaded = False
        t = FlowBookTools(session)
        assert "No notebook" in t.get_notebook_path()


# ==================================================================
# Log tools
# ==================================================================

class TestGetLog:
    def test_empty(self, tools, session):
        session.get_event_log.return_value = []
        assert "No events" in tools.get_log()

    def test_with_events(self, tools, session):
        session.get_event_log.return_value = [{"seq": 1, "tool": "run_cell"}]
        result = tools.get_log()
        assert "run_cell" in result


class TestPrintLog:
    def test_empty(self, tools, session):
        session.get_event_log.return_value = []
        assert "No events" in tools.print_log()

    def test_with_events(self, tools, session):
        session.get_event_log.return_value = [
            {"seq": 1, "elapsed_s": 0.5, "duration_ms": 100, "tool": "run_cell", "result": "ok"},
        ]
        result = tools.print_log()
        assert "1 events" in result
        assert "run_cell" in result


class TestCellStatus:
    def test_stale(self):
        session = _make_session(stale={"aa"})
        t = FlowBookTools(session)
        assert t._cell_status("aa") == "stale"

    def test_error(self):
        session = _make_session(cell_status={"aa": "error"})
        t = FlowBookTools(session)
        assert t._cell_status("aa") == "error"

    def test_ok(self):
        session = _make_session(executed={"aa"})
        t = FlowBookTools(session)
        assert t._cell_status("aa") == "ok"

    def test_unexecuted(self):
        session = _make_session()
        t = FlowBookTools(session)
        assert t._cell_status("aa") == "\u2014"
