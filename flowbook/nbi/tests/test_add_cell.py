"""Tests for the NBI add_cell tool.

Guards the after_cell routing. Previously add_cell ignored after_cell and
delegated to `notebook-intelligence:add-code-cell-to-active-notebook`, which
always inserted at the end of the notebook — so "insert after @A" silently
put the cell in the wrong place. Now add_cell drives a new
`flowbook:add-cell` frontend command and passes the target code-cell index
explicitly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("notebook_intelligence")

from flowbook.nbi import tools as nbi_tools


def _response(ui_commands=None):
    """Mock response whose run_ui_command looks up handlers by command name."""
    ui_commands = ui_commands or {}
    response = MagicMock()
    calls: list[tuple[str, dict]] = []

    async def run_ui_command(name, args=None):
        calls.append((name, args or {}))
        handler = ui_commands.get(name, {})
        if callable(handler):
            return handler(args or {})
        return handler

    response.run_ui_command = AsyncMock(side_effect=run_ui_command)
    response.calls = calls
    return response


class TestAddCellAfter:
    @pytest.mark.asyncio
    async def test_after_A_sends_afterCodeCellIndex_zero(self):
        response = _response({
            "flowbook:add-cell": {"cell_id": "nnnn", "inserted_at": 1},
            "flowbook:notify-structure": {},
        })
        await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "x = 1", "cell_type": "code", "after_cell": "@A"},
        )
        payloads = [args for name, args in response.calls if name == "flowbook:add-cell"]
        assert payloads, "flowbook:add-cell was never called"
        assert payloads[0]["afterCodeCellIndex"] == 0
        assert payloads[0]["source"] == "x = 1"
        assert payloads[0]["cellType"] == "code"

    @pytest.mark.asyncio
    async def test_after_C_sends_afterCodeCellIndex_two(self):
        response = _response({"flowbook:add-cell": {"cell_id": "x"}})
        await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "y = 2", "cell_type": "code", "after_cell": "@C"},
        )
        payloads = [args for name, args in response.calls if name == "flowbook:add-cell"]
        assert payloads[0]["afterCodeCellIndex"] == 2

    @pytest.mark.asyncio
    async def test_no_after_cell_omits_field_and_appends(self):
        response = _response({"flowbook:add-cell": {"cell_id": "y"}})
        await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "z = 3", "cell_type": "code"},
        )
        payloads = [args for name, args in response.calls if name == "flowbook:add-cell"]
        assert "afterCodeCellIndex" not in payloads[0]
        assert "cellIndex" not in payloads[0]

    @pytest.mark.asyncio
    async def test_markdown_cell_type_propagates(self):
        response = _response({"flowbook:add-cell": {"cell_id": "m"}})
        await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "# Title", "cell_type": "markdown", "after_cell": "@A"},
        )
        payloads = [args for name, args in response.calls if name == "flowbook:add-cell"]
        assert payloads[0]["cellType"] == "markdown"
        assert payloads[0]["afterCodeCellIndex"] == 0

    @pytest.mark.asyncio
    async def test_notify_structure_called_after_insert(self):
        """The kernel needs the new cell order; notify-structure must run after
        flowbook:add-cell."""
        response = _response({"flowbook:add-cell": {"cell_id": "x"}})
        await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "x = 1", "after_cell": "@A"},
        )
        command_order = [name for name, _ in response.calls]
        assert command_order.index("flowbook:add-cell") < command_order.index(
            "flowbook:notify-structure"
        )

    @pytest.mark.asyncio
    async def test_result_message_includes_position(self):
        response = _response({"flowbook:add-cell": {"cell_id": "nnnn"}})
        msg = await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "x = 1", "after_cell": "@A"},
        )
        assert "@A" in msg
        assert "nnnn" in msg

    @pytest.mark.asyncio
    async def test_result_message_end_when_no_after(self):
        response = _response({"flowbook:add-cell": {"cell_id": "qqqq"}})
        msg = await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={},
            tool_args={"source": "x = 1"},
        )
        assert "end" in msg.lower()


class TestAddCellFailsLoudWhenFrontendBroken:
    @pytest.mark.asyncio
    async def test_stringified_command_error_is_surfaced(self):
        """If flowbook:add-cell isn't registered (stale bundle), chat-sidebar.tsx
        returns an error string. add_cell must report failure, not success with
        no cell_id."""
        response = _response({
            "flowbook:add-cell": "Error executing command: Command 'flowbook:add-cell' not registered.",
        })
        msg = await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"source": "x = 1", "after_cell": "@A"},
        )
        assert "Added" not in msg, f"Expected failure message, got: {msg!r}"
        assert "stale" in msg.lower() or "rebuild" in msg.lower()

    @pytest.mark.asyncio
    async def test_missing_cell_id_is_surfaced(self):
        """If flowbook:add-cell returns a dict without cell_id, the tool must
        surface that as an error — it means something went wrong on the frontend."""
        response = _response({"flowbook:add-cell": {}})
        msg = await nbi_tools.add_cell.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"source": "x = 1", "after_cell": "@A"},
        )
        assert "Added" not in msg
        assert "unexpected" in msg.lower() or "stale" in msg.lower() or "rebuild" in msg.lower()
