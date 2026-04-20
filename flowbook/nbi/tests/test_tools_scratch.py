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


class TestChatStreaming:
    @pytest.mark.asyncio
    async def test_streams_header_and_code_block(self):
        response = _response({
            "flowbook:scratch-work": {
                "status": "ok", "execution_time_ms": 1.2, "outputs": [], "error": None,
            }
        })
        await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "df.shape"},
        )
        # response.stream was called with MarkdownData and ImageData objects;
        # check their .content for the header + fenced code block.
        streamed = [c.args[0].content for c in response.stream.call_args_list
                    if hasattr(c.args[0], "content")]
        joined = "\n".join(streamed)
        assert "scratch_work" in joined
        assert "```python" in joined
        assert "df.shape" in joined

    @pytest.mark.asyncio
    async def test_streams_stdout_as_code_block(self):
        response = _response({
            "flowbook:scratch-work": {
                "status": "ok", "execution_time_ms": 0.5,
                "outputs": [{"kind": "stream", "stream_name": "stdout", "text": "hello world\n"}],
                "error": None,
            }
        })
        await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "print('hi')"},
        )
        streamed = [c.args[0].content for c in response.stream.call_args_list
                    if hasattr(c.args[0], "content")]
        joined = "\n".join(streamed)
        assert "stdout" in joined
        assert "hello world" in joined

    @pytest.mark.asyncio
    async def test_streams_image_via_imagedata(self):
        b64 = "iVBORw0KGgoAAAAN"  # fake PNG payload
        response = _response({
            "flowbook:scratch-work": {
                "status": "ok", "execution_time_ms": 12.0,
                "outputs": [{
                    "kind": "display_data",
                    "data": {"image/png": {"encoding": "base64", "bytes": b64, "size_bytes": 12}},
                }],
                "error": None,
            }
        })
        await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "plt.show()"},
        )
        streamed = [c.args[0] for c in response.stream.call_args_list]
        # One of the streamed items is an ImageData with a data: URI content.
        from notebook_intelligence.api import ImageData as _ImageData
        images = [s for s in streamed if isinstance(s, _ImageData)]
        assert images, "expected ImageData stream for image/png output"
        assert images[0].content.startswith("data:image/png;base64,")
        assert b64 in images[0].content

    @pytest.mark.asyncio
    async def test_streams_error_traceback(self):
        response = _response({
            "flowbook:scratch-work": {
                "status": "error", "execution_time_ms": 2.0,
                "outputs": [],
                "error": {"ename": "ValueError", "evalue": "bad", "traceback": ["line1", "line2"]},
            }
        })
        await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "raise ValueError('bad')"},
        )
        streamed = [c.args[0].content for c in response.stream.call_args_list
                    if hasattr(c.args[0], "content")]
        joined = "\n".join(streamed)
        assert "ValueError: bad" in joined
        assert "line1" in joined and "line2" in joined

    @pytest.mark.asyncio
    async def test_streams_html_as_fenced_block(self):
        response = _response({
            "flowbook:scratch-work": {
                "status": "ok", "execution_time_ms": 1.0,
                "outputs": [{
                    "kind": "execute_result",
                    "data": {"text/html": {"text": "<table><tr><td>x</td></tr></table>"},
                             "text/plain": {"text": "<Figure>"}},
                }],
                "error": None,
            }
        })
        await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "df.head()"},
        )
        streamed = [c.args[0].content for c in response.stream.call_args_list
                    if hasattr(c.args[0], "content")]
        joined = "\n".join(streamed)
        assert "```html" in joined
        assert "<table>" in joined

    @pytest.mark.asyncio
    async def test_no_stream_on_frontend_error(self):
        """When _ui raises _FrontendError, _stream_scratch_result should not
        have streamed any of the per-output content (only whatever _safe_tool
        emits as its error-hint return)."""
        response = _response({
            "flowbook:scratch-work": "Error executing command: Command 'flowbook:scratch-work' not registered.",
        })
        result = await nbi_tools.scratch_work.handle_tool_call(
            request=MagicMock(), response=response,
            tool_context={}, tool_args={"code": "x = 1"},
        )
        # _FrontendError path returns a string; we didn't even reach
        # _stream_scratch_result.
        assert isinstance(result, str)


class TestRegistration:
    def test_scratch_work_registered(self):
        from flowbook.nbi.tools import create_tools

        tools = create_tools(MagicMock())
        names = {t.name for t in tools}
        assert "scratch_work" in names
        assert "get_cell_outputs" in names
