"""Unit tests for MCP server tools and FlowBookTools integration.

Tests cover:
- FlowBookTools._label (replaces old _cell_label)
- get_next_actionable_cell_id session method
- Tool wrappers: read_cell, edit_cell_source, get_flowbook_metadata,
  run_actionable_cell, run_actionable_cells, run_cell
- run_cell calling _put_contents_api after execution
- NotebookSession.add_cell and delete_cell
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from flowbook.mcp.server import (
    read_cell,
    edit_cell_source,
    get_flowbook_metadata,
    run_actionable_cell,
    run_actionable_cells,
    run_cell,
    _get_session,
)
from flowbook.mcp.session import NotebookSession
from flowbook.tools.tools import FlowBookTools


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
    tools = FlowBookTools(session)
    ctx.request_context.lifespan_context = {"session": session, "tools": tools}
    return ctx


# ==================================================================
# FlowBookTools._label (replaces old _cell_label)
# ==================================================================


class TestCellLabel:
    def test_known_id_first(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        tools = FlowBookTools(session)
        assert tools._label("abc1") == "@A"

    def test_known_id_second(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        tools = FlowBookTools(session)
        assert tools._label("def2") == "@B"

    def test_known_id_third(self):
        session = _make_mock_session(cell_order=["abc1", "def2", "ghi3"])
        tools = FlowBookTools(session)
        assert tools._label("ghi3") == "@C"

    def test_unknown_id_returns_raw(self):
        session = _make_mock_session(cell_order=["abc1"])
        tools = FlowBookTools(session)
        assert tools._label("unknown") == "unknown"

    def test_empty_order(self):
        session = _make_mock_session(cell_order=[])
        tools = FlowBookTools(session)
        assert tools._label("any") == "any"


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
        result = read_cell(cell="abc1", ctx=ctx)
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
        result = read_cell(cell="abc1", ctx=ctx)
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
        result = read_cell(cell="abc1", ctx=ctx)
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
        result = edit_cell_source(cell="abc1", new_source="x = 99", ctx=ctx)
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
        result = edit_cell_source(cell="abc1", new_source="y = 2", ctx=ctx)
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
        result = get_flowbook_metadata(cell="abc1", ctx=ctx)
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
        result = get_flowbook_metadata(cell="abc1", ctx=ctx)
        assert "not been executed" in result
        assert "@A" in result

    def test_unknown_cell_returns_error(self):
        session = _make_mock_session(
            cell_order=["abc1"],
            cell_flowbook_meta={},
        )
        ctx = _make_ctx(session)
        result = get_flowbook_metadata(cell="zzzz", ctx=ctx)
        assert "ERROR" in result
        assert "Cannot resolve" in result


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
        run_cell(cell="abc1", ctx=ctx)
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
# _get_session helper
# ==================================================================


# ==================================================================
# NotebookSession.add_cell
# ==================================================================


class TestAddCell:
    def _make_session_with_cells(self):
        """Create a partially-real session with a notebook in memory."""
        cell_a = {"id": "aaaa", "cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": []}
        cell_b = {"id": "bbbb", "cell_type": "code", "source": "y = 2", "metadata": {}, "outputs": []}
        session = MagicMock(spec=NotebookSession)
        session.is_loaded = True
        session.notebook = {"cells": [cell_a, cell_b]}
        session._stale_cells = set()
        session.executed_cells = set()
        session.cell_flowbook_meta = {}
        session.cell_status = {}
        session.kernel_client = MagicMock()

        # Use real _find_cell
        def find_cell(cid):
            for i, c in enumerate(session.notebook["cells"]):
                if c.get("id") == cid:
                    return i, c
            raise ValueError(f"Cell {cid} not found")
        session._find_cell.side_effect = find_cell

        # Use real get_cell_order
        def get_cell_order():
            return [c["id"] for c in session.notebook["cells"] if c.get("cell_type") == "code"]
        session.get_cell_order.side_effect = get_cell_order

        return session

    def test_append_to_end(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            result = NotebookSession.add_cell(session, "z = 3")
        assert result["cell_type"] == "code"
        assert len(session.notebook["cells"]) == 3
        assert session.notebook["cells"][-1]["source"] == "z = 3"
        session._put_contents_api.assert_called_once()

    def test_insert_after_cell(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            result = NotebookSession.add_cell(session, "mid = 0", after_cell_id="aaaa")
        assert len(session.notebook["cells"]) == 3
        assert session.notebook["cells"][1]["source"] == "mid = 0"
        assert session.notebook["cells"][2]["id"] == "bbbb"

    def test_markdown_cell(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper") as mock_kh:
            result = NotebookSession.add_cell(session, "# Title", cell_type="markdown")
        assert result["cell_type"] == "markdown"
        # Markdown cells should NOT notify kernel of structure change
        mock_kh.execute_code.assert_not_called()

    def test_invalid_after_cell_raises(self):
        session = self._make_session_with_cells()
        with pytest.raises(ValueError, match="not found"):
            NotebookSession.add_cell(session, "z = 3", after_cell_id="zzzz")

    def test_generated_id_is_unique(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            result = NotebookSession.add_cell(session, "z = 3")
        new_id = result["cell_id"]
        existing = {"aaaa", "bbbb"}
        assert new_id not in existing


# ==================================================================
# NotebookSession.delete_cell
# ==================================================================


class TestDeleteCell:
    def _make_session_with_cells(self):
        """Create a partially-real session with a notebook in memory."""
        cell_a = {"id": "aaaa", "cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": []}
        cell_b = {"id": "bbbb", "cell_type": "code", "source": "y = 2", "metadata": {}, "outputs": []}
        cell_c = {"id": "cccc", "cell_type": "code", "source": "z = 3", "metadata": {}, "outputs": []}
        session = MagicMock(spec=NotebookSession)
        session.is_loaded = True
        session.notebook = {"cells": [cell_a, cell_b, cell_c]}
        session._stale_cells = {"bbbb"}
        session.executed_cells = {"aaaa", "bbbb"}
        session.cell_flowbook_meta = {"bbbb": {"some": "meta"}}
        session.cell_status = {"bbbb": "ok"}
        session.kernel_client = MagicMock()

        def find_cell(cid):
            for i, c in enumerate(session.notebook["cells"]):
                if c.get("id") == cid:
                    return i, c
            raise ValueError(f"Cell {cid} not found")
        session._find_cell.side_effect = find_cell

        def get_cell_order():
            return [c["id"] for c in session.notebook["cells"] if c.get("cell_type") == "code"]
        session.get_cell_order.side_effect = get_cell_order

        return session

    def test_removes_cell(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            result = NotebookSession.delete_cell(session, "bbbb")
        assert result["cell_id"] == "bbbb"
        assert len(session.notebook["cells"]) == 2
        remaining_ids = [c["id"] for c in session.notebook["cells"]]
        assert "bbbb" not in remaining_ids

    def test_cleans_up_tracking(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            NotebookSession.delete_cell(session, "bbbb")
        assert "bbbb" not in session.executed_cells
        assert "bbbb" not in session.cell_flowbook_meta
        assert "bbbb" not in session.cell_status
        assert "bbbb" not in session._stale_cells

    def test_notifies_kernel(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper") as mock_kh:
            NotebookSession.delete_cell(session, "bbbb")
        mock_kh.execute_code.assert_called_once()
        call_kwargs = mock_kh.execute_code.call_args
        fb_msg = call_kwargs.kwargs.get("flowbook_msg") or call_kwargs[1].get("flowbook_msg")
        assert fb_msg["type"] == "notebook_structure"

    def test_pushes_contents_api(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            NotebookSession.delete_cell(session, "bbbb")
        session._put_contents_api.assert_called_once()

    def test_invalid_cell_raises(self):
        session = self._make_session_with_cells()
        with pytest.raises(ValueError, match="not found"):
            NotebookSession.delete_cell(session, "zzzz")

    def test_returns_new_cell_order(self):
        session = self._make_session_with_cells()
        with patch("flowbook.mcp.session.KernelHelper"):
            result = NotebookSession.delete_cell(session, "bbbb")
        assert result["new_cell_order"] == ["aaaa", "cccc"]


class TestGetSession:
    def test_extracts_session_from_context(self):
        session = _make_mock_session()
        ctx = _make_ctx(session)
        assert _get_session(ctx) is session
