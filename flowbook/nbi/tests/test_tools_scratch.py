"""Tests for NBI tools: scratch_work and get_cell_outputs."""

import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("notebook_intelligence")

from flowbook.nbi import tools as nbi_tools
from flowbook.tools.mcp_content import ToolContent


def _response(ui_commands: dict):
    """Build a mock `response` whose run_ui_command returns the mapped payload
    per command name. Unknown commands return {}."""
    response = MagicMock()

    async def run_ui_command(name, args=None):
        handler = ui_commands.get(name)
        if callable(handler):
            return handler(args or {})
        return handler if handler is not None else {}

    response.run_ui_command = AsyncMock(side_effect=run_ui_command)
    return response


# --------------------------------------------------------------------------
# scratch_work
# --------------------------------------------------------------------------

class TestScratchWork:
    @pytest.mark.asyncio
    async def test_calls_frontend_command_with_code(self):
        response = _response({
            "flowbook:scratch-work": {
                "status": "ok", "execution_time_ms": 1.0, "outputs": [], "error": None,
            }
        })
        await nbi_tools.scratch_work.handle_tool_call(  # type: ignore[attr-defined]
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "x = 1"},
        )
        response.run_ui_command.assert_any_call(
            "flowbook:scratch-work", {"code": "x = 1"}
        )

    @pytest.mark.asyncio
    async def test_returns_tool_content(self):
        response = _response({
            "flowbook:scratch-work": {
                "status": "ok", "execution_time_ms": 2.5, "outputs": [
                    {"kind": "stream", "stream_name": "stdout", "text": "hi"}
                ], "error": None,
            }
        })
        result = await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "print('hi')"},
        )
        assert isinstance(result, ToolContent)
        assert any("hi" in b.get("text", "") for b in result.blocks if b.get("type") == "text")
        assert "status: ok" in result.text_summary


# --------------------------------------------------------------------------
# get_cell_outputs
# --------------------------------------------------------------------------

class TestGetCellOutputs:
    @pytest.mark.asyncio
    async def test_resolves_alpha_to_cell_ids(self):
        fake_cells = {0: "aaaa", 1: "bbbb", 2: "cccc"}
        def get_cell(args):
            idx = args["cellIndex"]
            return {"cell_id": fake_cells[idx], "source": "", "outputs_text": ""}

        response = _response({
            "flowbook:get-cell-count": {"code_cells": 3},
            "flowbook:get-cell": get_cell,
            "flowbook:get-cell-outputs": {"cells": []},
        })

        await nbi_tools.get_cell_outputs.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"cells": ["@A", "@C"]},
        )

        # Verify the eventual call to get-cell-outputs carried resolved IDs.
        calls = [c.args for c in response.run_ui_command.call_args_list]
        final = [a for a in calls if a[0] == "flowbook:get-cell-outputs"]
        assert final, "get-cell-outputs was never called"
        assert final[-1][1] == {"cellIds": ["aaaa", "cccc"]}

    @pytest.mark.asyncio
    async def test_passes_raw_cell_ids_through(self):
        response = _response({
            "flowbook:get-cell-count": {"code_cells": 3},
            "flowbook:get-cell-outputs": {"cells": []},
        })
        await nbi_tools.get_cell_outputs.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"cells": ["abcd"]},
        )
        calls = [c.args for c in response.run_ui_command.call_args_list]
        final = [a for a in calls if a[0] == "flowbook:get-cell-outputs"]
        assert final[-1][1] == {"cellIds": ["abcd"]}

    @pytest.mark.asyncio
    async def test_returns_tool_content_with_cells(self):
        response = _response({
            "flowbook:get-cell-count": {"code_cells": 1},
            "flowbook:get-cell": {"cell_id": "aaaa"},
            "flowbook:get-cell-outputs": {
                "cells": [{
                    "cell_id": "aaaa", "label": "@A",
                    "outputs": [{"kind": "stream", "stream_name": "stdout", "text": "hello"}],
                }],
            },
        })
        result = await nbi_tools.get_cell_outputs.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"cells": ["@A"]},
        )
        assert isinstance(result, ToolContent)
        assert "hello" in result.text_summary
        assert "@A" in result.text_summary

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_cells(self):
        response = _response({
            "flowbook:get-cell-count": {"code_cells": 0},
            "flowbook:get-cell-outputs": {"cells": []},
        })
        result = await nbi_tools.get_cell_outputs.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"cells": []},
        )
        assert isinstance(result, ToolContent)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

class TestFrontendErrorFallback:
    @pytest.mark.asyncio
    async def test_string_result_does_not_crash(self):
        """notebook-intelligence's chat-sidebar.tsx wraps command errors into
        a plain string like 'Error executing command: ...'. The tool must
        surface it readably, not crash on result.get()."""
        response = _response({
            "flowbook:scratch-work": "Error executing command: Command 'flowbook:scratch-work' not registered.",
        })
        result = await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "x = 1"},
        )
        assert isinstance(result, str)
        # _FrontendError is caught by _safe_tool and formatted with a hint.
        assert "stale" in result.lower() or "rebuild" in result.lower()

    @pytest.mark.asyncio
    async def test_get_cell_outputs_resilient_to_string_result(self):
        response = _response({
            "flowbook:get-cell-count": {"code_cells": 0},
            "flowbook:get-cell-outputs": "Error executing command: Something broke",
        })
        result = await nbi_tools.get_cell_outputs.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"cells": []},
        )
        assert isinstance(result, str)
        assert "stale" in result.lower() or "rebuild" in result.lower()


class TestRegistration:
    def test_scratch_work_registered(self):
        from flowbook.nbi.tools import create_tools

        tools = create_tools(MagicMock())
        names = {t.name for t in tools}
        assert "scratch_work" in names
        assert "get_cell_outputs" in names
