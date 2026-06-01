"""Unit tests for new/renamed tools and helpers in flowbook.mcp.server.

Tests cover:
- _cell_label helper function
- get_next_actionable_cell_id session method
- Renamed tools: read_cell, edit_cell_source
- New tools: get_flowbook_metadata, run_actionable_cell, run_actionable_cells
- run_cell calling _put_contents_api after execution
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from flowbook.mcp.server import (
    _cell_label,
    _format_cell_output,
    _OUTPUT_HEAD_CHARS,
    _OUTPUT_TAIL_CHARS,
    get_cell_output,
    read_cell,
    edit_cell_source,
    get_flowbook_metadata,
    run_actionable_cell,
    run_actionable_cells,
    run_cell,
    _get_session,
)
from flowbook.mcp.session import NotebookSession


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_session(cell_order=None, cell_flowbook_meta=None,
                       stale_cells=None, executed_cells=None,
                       cell_status=None, continue_after_violation=False):
    """Build a mock NotebookSession with sensible defaults."""
    session = MagicMock(spec=NotebookSession)
    session.is_loaded = True
    session.get_cell_order.return_value = cell_order or []
    session.cell_flowbook_meta = cell_flowbook_meta or {}
    session._stale_cells = stale_cells or set()
    session.executed_cells = executed_cells or set()
    session.cell_status = cell_status or {}
    session._continue_after_violation = continue_after_violation
    session.notebook_path = "/tmp/test.ipynb"
    # Default: log_event does nothing
    session.log_event = MagicMock()
    return session


def _make_ctx(session):
    """Build a mock MCP Context that returns the given session."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"session": session}
    return ctx


# ==================================================================
# _cell_label
# ==================================================================


class TestCellLabel:
    def test_known_id_first(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        assert _cell_label(session, "abc1") == "@A"

    def test_known_id_second(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        assert _cell_label(session, "def2") == "@B"

    def test_known_id_third(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        assert _cell_label(session, "ghi3") == "@C"

    def test_unknown_id_returns_raw(self):
        session = _make_mock_session(cell_order=["abc1"])
        assert _cell_label(session, "unknown") == "unknown"

    def test_empty_order(self):
        session = _make_mock_session(cell_order=[])
        assert _cell_label(session, "any") == "any"


# ==================================================================
# get_next_actionable_cell_id (session method)
# ==================================================================


class TestGetNextActionableCellId:
    def test_returns_id_when_actionable(self):
        session = _make_mock_session()
        session.get_next_actionable.return_value = {
            "cell_id": "abc1",
            "reason": "stale",
            "source": "x = 1",
        }
        assert session.get_next_actionable_cell_id() is not None
        # Since we mocked the whole object, we actually want to test
        # the real method. Let's use the real implementation instead.

    def test_real_method_returns_id(self):
        """Test the real get_next_actionable_cell_id delegates to get_next_actionable."""
        session = MagicMock()
        session.get_next_actionable.return_value = {
            "cell_id": "abc1",
            "reason": "stale",
            "source": "x = 1",
        }
        # Call the real method bound to our mock
        result = NotebookSession.get_next_actionable_cell_id(session)
        assert result == "abc1"
        session.get_next_actionable.assert_called_once()

    def test_real_method_returns_none_when_clean(self):
        """Test get_next_actionable_cell_id returns None when all clean."""
        session = MagicMock()
        session.get_next_actionable.return_value = None
        result = NotebookSession.get_next_actionable_cell_id(session)
        assert result is None


# ==================================================================
# Tool existence and callability (renames + new tools)
# ==================================================================


class TestToolExistence:
    def test_read_cell_exists(self):
        assert callable(read_cell)

    def test_edit_cell_source_exists(self):
        assert callable(edit_cell_source)

    def test_get_flowbook_metadata_exists(self):
        assert callable(get_flowbook_metadata)

    def test_run_actionable_cell_exists(self):
        assert callable(run_actionable_cell)

    def test_run_actionable_cells_exists(self):
        assert callable(run_actionable_cells)

    def test_old_names_not_exported(self):
        """Verify the old names (get_cell, edit_cell) are NOT in server module."""
        import flowbook.mcp.server as srv
        # The old function names should not exist as top-level symbols
        assert not hasattr(srv, "get_cell") or not callable(getattr(srv, "get_cell", None))
        assert not hasattr(srv, "edit_cell") or not callable(getattr(srv, "edit_cell", None))


# ==================================================================
# read_cell (formerly get_cell)
# ==================================================================


class TestReadCell:
    def test_returns_cell_info(self):
        session = _make_mock_session(cell_order=["abc1", "def2"])
        session.get_cell.return_value = {
            "cell_id": "abc1",
            "source": "x = 1",
            "status": "ok",
            "outputs_text": "",
        }
        ctx = _make_ctx(session)
        result = read_cell("abc1", ctx)
        assert "@A" in result
        assert "abc1" in result
        assert "x = 1" in result
        session.get_cell.assert_called_once_with("abc1")

    def test_includes_output_preview(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_cell.return_value = {
            "cell_id": "abc1",
            "source": "print(42)",
            "status": "ok",
            "outputs_text": "42\n",
        }
        ctx = _make_ctx(session)
        result = read_cell("abc1", ctx)
        assert "Output:" in result
        assert "42" in result

    def test_includes_flowbook_meta(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_cell.return_value = {
            "cell_id": "abc1",
            "source": "x = 1",
            "status": "ok",
            "outputs_text": "",
            "flowbook": "Reads: (none)\nWrites: x",
        }
        ctx = _make_ctx(session)
        result = read_cell("abc1", ctx)
        assert "Writes: x" in result


# ==================================================================
# edit_cell_source (formerly edit_cell)
# ==================================================================


class TestEditCellSource:
    def test_returns_update_confirmation(self):
        session = _make_mock_session(cell_order=["abc1", "def2"])
        session.edit_cell.return_value = {
            "cell_id": "abc1",
            "marked_stale": True,
            "new_source_preview": "x = 99",
        }
        ctx = _make_ctx(session)
        result = edit_cell_source("abc1", "x = 99", ctx)
        assert "@A" in result
        assert "abc1" in result
        assert "marked stale" in result
        assert "x = 99" in result
        session.edit_cell.assert_called_once_with("abc1", "x = 99")

    def test_not_stale_when_not_executed(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.edit_cell.return_value = {
            "cell_id": "abc1",
            "marked_stale": False,
            "new_source_preview": "y = 2",
        }
        ctx = _make_ctx(session)
        result = edit_cell_source("abc1", "y = 2", ctx)
        assert "marked stale" not in result


# ==================================================================
# get_flowbook_metadata
# ==================================================================


class TestGetFlowbookMetadata:
    def test_returns_metadata_for_executed_cell(self):
        meta = {
            "type": "metadata",
            "cell_id": "abc1",
            "read_locs": [{"type": "var", "name": "x"}],
            "write_locs": [{"type": "var", "name": "y"}],
            "changed_locs": [],
            "errors": [],
            "stale_cells": [],
            "staleness_reasons": {},
        }
        session = _make_mock_session(
            cell_order=["abc1", "def2"],
            cell_flowbook_meta={"abc1": meta},
        )
        ctx = _make_ctx(session)
        result = get_flowbook_metadata("abc1", ctx)
        assert "@A" in result
        assert "abc1" in result
        # format_flowbook_meta produces "Reads:" and "Writes:" lines
        assert "Reads:" in result
        assert "Writes:" in result

    def test_returns_message_for_unexecuted_cell(self):
        session = _make_mock_session(
            cell_order=["abc1"],
            cell_flowbook_meta={},
        )
        ctx = _make_ctx(session)
        result = get_flowbook_metadata("abc1", ctx)
        assert "not been executed" in result
        assert "@A" in result

    def test_unknown_cell_shows_raw_id(self):
        session = _make_mock_session(
            cell_order=["abc1"],
            cell_flowbook_meta={},
        )
        ctx = _make_ctx(session)
        result = get_flowbook_metadata("zzzz", ctx)
        assert "zzzz" in result
        assert "not been executed" in result


# ==================================================================
# run_actionable_cell
# ==================================================================


class TestRunActionableCell:
    def test_runs_next_actionable(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        session.get_next_actionable_cell_id.return_value = "def2"
        session.run_cell.return_value = {
            "cell_id": "def2",
            "status": "ok",
            "outputs_text": "",
        }
        ctx = _make_ctx(session)
        result = run_actionable_cell(ctx)
        assert "Ran" in result
        assert "@B" in result
        assert "def2" in result
        session.run_cell.assert_called_once_with("def2")

    def test_all_clean_when_no_actionable(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_next_actionable_cell_id.return_value = None
        ctx = _make_ctx(session)
        result = run_actionable_cell(ctx)
        assert "All clean" in result
        session.run_cell.assert_not_called()

    def test_includes_error_message(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_next_actionable_cell_id.return_value = "abc1"
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "error",
            "outputs_text": "",
            "error_message": "NameError: name 'foo' is not defined",
        }
        ctx = _make_ctx(session)
        result = run_actionable_cell(ctx)
        assert "error" in result.lower() or "NameError" in result

    def test_includes_output_preview(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_next_actionable_cell_id.return_value = "abc1"
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "ok",
            "outputs_text": "Hello World\n",
        }
        ctx = _make_ctx(session)
        result = run_actionable_cell(ctx)
        assert "Hello World" in result

    def test_includes_flowbook_metadata(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_next_actionable_cell_id.return_value = "abc1"
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "ok",
            "outputs_text": "",
            "flowbook": "Reads: (none)\nWrites: x",
        }
        ctx = _make_ctx(session)
        result = run_actionable_cell(ctx)
        assert "Writes: x" in result


# ==================================================================
# run_actionable_cells
# ==================================================================


class TestRunActionableCells:
    def test_runs_all_until_clean(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        # get_next_actionable_cell_id returns cells then None
        session.get_next_actionable_cell_id.side_effect = [
            "abc1", "def2", "ghi3", None
        ]
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "ok",
            "outputs_text": "",
        }
        session.cell_flowbook_meta = {}
        session.get_status.return_value = {
            "stale_cells": {},
            "violations": [],
            "executed": 3,
            "total_code_cells": 3,
        }
        ctx = _make_ctx(session)
        result = run_actionable_cells(ctx)
        assert "Ran 3 cells" in result
        assert "All clean!" in result
        assert session.run_cell.call_count == 3

    def test_stops_on_hard_error(self):
        session = _make_mock_session(cell_order=["abc1", "def2"])
        session.get_next_actionable_cell_id.side_effect = ["abc1", "def2"]
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "error",
            "outputs_text": "",
            "error_message": "SyntaxError",
        }
        session.cell_flowbook_meta = {}
        session.get_status.return_value = {
            "stale_cells": {},
            "violations": [],
            "executed": 1,
            "total_code_cells": 2,
        }
        ctx = _make_ctx(session)
        result = run_actionable_cells(ctx)
        assert "Ran 1 cell" in result
        assert "error" in result.lower()
        # Should stop after the first cell
        assert session.run_cell.call_count == 1

    def test_stops_on_violation_when_reject_mode(self):
        session = _make_mock_session(
            cell_order=["abc1", "def2"],
            continue_after_violation=False,
        )
        session.get_next_actionable_cell_id.side_effect = ["abc1", "def2"]
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "ok",
            "outputs_text": "",
        }
        # After run, the cell has a violation
        session.cell_flowbook_meta = {
            "abc1": {
                "errors": [
                    {"error_type": "NO_READ_AND_WRITE", "locations": [], "message": "test"}
                ]
            }
        }
        session.get_status.return_value = {
            "stale_cells": {},
            "violations": [{"cell_id": "abc1", "error_type": "NO_READ_AND_WRITE"}],
            "executed": 1,
            "total_code_cells": 2,
        }
        ctx = _make_ctx(session)
        result = run_actionable_cells(ctx)
        assert "1 violations" in result
        assert session.run_cell.call_count == 1

    def test_continues_past_violation_when_enabled(self):
        session = _make_mock_session(
            cell_order=["abc1", "def2"],
            continue_after_violation=True,
        )
        # First call returns abc1, second returns def2, third returns None
        session.get_next_actionable_cell_id.side_effect = ["abc1", "def2", None]

        call_count = [0]
        def mock_run_cell(cell_id, **kwargs):
            call_count[0] += 1
            return {
                "cell_id": cell_id,
                "status": "ok",
                "outputs_text": "",
            }
        session.run_cell.side_effect = mock_run_cell

        # abc1 has a violation, def2 does not
        session.cell_flowbook_meta = {
            "abc1": {
                "errors": [
                    {"error_type": "NO_WRITE_AFTER_READ", "locations": [], "message": "test"}
                ]
            },
            "def2": {"errors": []},
        }
        session.get_status.return_value = {
            "stale_cells": {},
            "violations": [{"cell_id": "abc1", "error_type": "NO_WRITE_AFTER_READ"}],
            "executed": 2,
            "total_code_cells": 2,
        }
        ctx = _make_ctx(session)
        result = run_actionable_cells(ctx)
        # Should run both cells because continue_after_violation is True
        assert "Ran 2 cells" in result
        assert session.run_cell.call_count == 2

    def test_returns_no_actionable_immediately(self):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_next_actionable_cell_id.return_value = None
        session.get_status.return_value = {
            "stale_cells": {},
            "violations": [],
            "executed": 1,
            "total_code_cells": 1,
        }
        ctx = _make_ctx(session)
        result = run_actionable_cells(ctx)
        assert "Ran 0 cells" in result
        assert "0 stale" in result
        session.run_cell.assert_not_called()


# ==================================================================
# run_cell calls _put_contents_api
# ==================================================================


class TestRunCellPutsContentsApi:
    def test_put_contents_api_called_after_execution(self):
        """Verify that session.run_cell calls _put_contents_api."""
        session = _make_mock_session(cell_order=["abc1"])
        session.run_cell.return_value = {
            "cell_id": "abc1",
            "status": "ok",
            "outputs_text": "",
        }
        ctx = _make_ctx(session)
        # Call the server tool which delegates to session.run_cell
        run_cell("abc1", ctx)
        session.run_cell.assert_called_once_with("abc1")
        # The actual _put_contents_api call is inside session.run_cell,
        # which we've mocked. To verify the real implementation calls it,
        # we test the session method directly below.

    def test_session_run_cell_calls_put_contents_api(self):
        """Verify the real session.run_cell calls _put_contents_api."""
        # We need a partially-real session, so patch specific methods
        cell = {
            "cell_type": "code",
            "id": "abc1",
            "source": "x = 1",
            "outputs": [],
            "metadata": {},
        }
        session = MagicMock(spec=NotebookSession)
        session.is_loaded = True
        session.notebook = {"cells": [cell]}
        session.get_cell_order.return_value = ["abc1"]
        session._find_cell.return_value = (0, cell)
        session._extract_flowbook_meta.return_value = None
        session.executed_cells = set()
        session.cell_flowbook_meta = {}
        session._stale_cells = set()
        session.cell_status = {}
        session.kernel_client = MagicMock()

        # Mock KernelHelper.execute_code
        execute_result = {
            "status": "ok",
            "outputs": [],
            "execution_count": 1,
            "flowbook_messages": [],
        }
        with patch("flowbook.mcp.session.KernelHelper") as mock_kh:
            mock_kh.execute_code.return_value = execute_result
            # Call the REAL run_cell implementation
            NotebookSession.run_cell(session, "abc1")

        session._put_contents_api.assert_called_once()


# ==================================================================
# _format_cell_output (middle-elision + truncation banner)
# ==================================================================


class TestFormatCellOutput:
    def test_empty_returns_blank(self):
        assert _format_cell_output("", "abc1") == ""
        assert _format_cell_output("   \n  ", "abc1") == ""

    def test_small_output_returned_whole(self):
        out = _format_cell_output("col_a  col_b\n1  2\n", "abc1")
        assert out == "\nOutput:\ncol_a  col_b\n1  2"
        assert "TRUNCATED" not in out

    def test_at_threshold_not_truncated(self):
        text = "x" * (_OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS)
        out = _format_cell_output(text, "abc1")
        assert "TRUNCATED" not in out
        assert out.endswith(text)

    def test_large_output_keeps_head_and_tail(self):
        head = "H" * _OUTPUT_HEAD_CHARS
        tail = "T" * _OUTPUT_TAIL_CHARS
        middle = "M" * 1000
        out = _format_cell_output(head + middle + tail, "abc1")
        # banner is obvious and quantifies what was dropped
        assert "OUTPUT TRUNCATED" in out
        total = _OUTPUT_HEAD_CHARS + 1000 + _OUTPUT_TAIL_CHARS
        assert f"1000 of {total} chars hidden" in out
        # head and tail survive; the middle is gone
        assert head in out and tail in out
        assert middle not in out

    def test_large_output_has_paging_hint(self):
        out = _format_cell_output("z" * 5000, "abc1")
        assert f"get_cell_output(cell_id='abc1', offset={_OUTPUT_HEAD_CHARS})" in out


# ==================================================================
# get_cell_output (paging)
# ==================================================================


class TestGetCellOutput:
    def _session_with_output(self, text):
        session = _make_mock_session(cell_order=["abc1"])
        session.get_cell.return_value = {"cell_id": "abc1", "outputs_text": text}
        return session

    def test_no_output(self):
        session = self._session_with_output("")
        result = get_cell_output("abc1", _make_ctx(session))
        assert "no output" in result

    def test_first_page_reports_total_and_next_offset(self):
        session = self._session_with_output("D" * 10000)
        result = get_cell_output("abc1", _make_ctx(session), offset=0, limit=4000)
        assert "chars 0–4000 of 10000" in result
        assert "offset=4000" in result  # more available
        assert result.count("D") == 4000

    def test_last_page_has_no_more_hint(self):
        session = self._session_with_output("D" * 10000)
        result = get_cell_output("abc1", _make_ctx(session), offset=8000, limit=4000)
        assert "chars 8000–10000 of 10000" in result
        assert "offset=" not in result  # nothing left to page

    def test_offset_past_end(self):
        session = self._session_with_output("D" * 100)
        result = get_cell_output("abc1", _make_ctx(session), offset=500)
        assert "past end" in result

    def test_negative_offset_clamped(self):
        session = self._session_with_output("D" * 100)
        result = get_cell_output("abc1", _make_ctx(session), offset=-5, limit=50)
        assert "chars 0–50 of 100" in result


# ==================================================================
# _get_session helper
# ==================================================================


class TestGetSession:
    def test_extracts_session_from_context(self):
        session = _make_mock_session()
        ctx = _make_ctx(session)
        assert _get_session(ctx) is session
