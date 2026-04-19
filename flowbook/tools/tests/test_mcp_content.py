"""Tests for flowbook.tools.mcp_content — the shared content adapter."""

import base64

import pytest

from flowbook.tools.mcp_content import (
    KEEP_IMAGE_MIMES,
    TEXT_TRUNCATE_BYTES,
    TOTAL_IMAGE_BUDGET_BYTES,
    ToolContent,
    build_tool_content,
    to_claude_content,
    to_markdown,
    to_openai_content,
)


PNG_1x1 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa3\xee\x1a\xb5\x00\x00\x00\x00IEND\xaeB`\x82"
).decode()


def _scratch(outputs=None, status="ok", error=None, time_ms=12.3):
    return {
        "status": status,
        "execution_time_ms": time_ms,
        "outputs": outputs or [],
        "error": error,
    }


def _stream(name, text):
    return {"kind": "stream", "stream_name": name, "text": text}


def _display(data):
    return {"kind": "display_data", "data": data}


def _execute(data):
    return {"kind": "execute_result", "data": data}


class TestToMarkdown:
    def test_header_present(self):
        md = to_markdown(_scratch())
        assert "status: ok" in md
        assert "12.3 ms" in md

    def test_stream_text(self):
        md = to_markdown(_scratch([_stream("stdout", "hello")]))
        assert "[stdout]" in md and "hello" in md

    def test_execute_result_plain(self):
        md = to_markdown(_scratch([_execute({"text/plain": {"text": "42"}})]))
        assert "42" in md

    def test_image_rendered_as_data_uri(self):
        payload = {"encoding": "base64", "bytes": PNG_1x1, "size_bytes": 100}
        md = to_markdown(_scratch([_display({"image/png": payload})]))
        assert "![](data:image/png;base64," in md
        assert PNG_1x1 in md

    def test_error_appended(self):
        r = _scratch(status="error",
                     error={"ename": "RuntimeError", "evalue": "boom", "traceback": ["line1", "line2"]})
        md = to_markdown(r)
        assert "RuntimeError: boom" in md
        assert "line1" in md and "line2" in md

    def test_html_included_as_text_block(self):
        md = to_markdown(_scratch([_display({"text/html": {"text": "<table><tr><td>x</td></tr></table>"}})]))
        assert "[text/html]" in md
        assert "<table>" in md

    def test_text_truncation(self):
        big = "x" * (TEXT_TRUNCATE_BYTES + 500)
        md = to_markdown(_scratch([_stream("stdout", big)]))
        assert "truncated" in md
        assert md.count("x") <= TEXT_TRUNCATE_BYTES + 10

    def test_dropped_mime_marker(self):
        md = to_markdown(_scratch([_display({
            "text/plain": {"text": "fallback"},
            "application/vnd.something": {"size_bytes": 5000},
        })]))
        assert "fallback" not in md or "application/vnd.something" in md
        assert "application/vnd.something" in md


class TestToClaudeContent:
    def test_text_block_shape(self):
        blocks = to_claude_content(_scratch([_execute({"text/plain": {"text": "42"}})]))
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert text_blocks and "42" in text_blocks[-1]["text"]

    def test_image_block_shape(self):
        payload = {"encoding": "base64", "bytes": PNG_1x1, "size_bytes": 100}
        blocks = to_claude_content(_scratch([_display({"image/png": payload})]))
        img_blocks = [b for b in blocks if b["type"] == "image"]
        assert len(img_blocks) == 1
        assert img_blocks[0]["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": PNG_1x1,
        }


class TestToOpenaiContent:
    def test_image_becomes_image_url_data_uri(self):
        payload = {"encoding": "base64", "bytes": PNG_1x1, "size_bytes": 100}
        blocks = to_openai_content(_scratch([_display({"image/png": payload})]))
        img_blocks = [b for b in blocks if b["type"] == "image_url"]
        assert len(img_blocks) == 1
        url = img_blocks[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert PNG_1x1 in url


class TestToMcpContent:
    def test_mcp_imports_lazy(self):
        # If mcp isn't installed, to_mcp_content should fail at call time, not at import time.
        # We just check the module imported without issue above.
        from flowbook.tools import mcp_content  # noqa: F401

    def test_mcp_content_roundtrip(self):
        pytest.importorskip("mcp")
        from flowbook.tools.mcp_content import to_mcp_content

        payload = {"encoding": "base64", "bytes": PNG_1x1, "size_bytes": 100}
        r = _scratch([
            _stream("stdout", "hello"),
            _display({"image/png": payload}),
        ])
        content = to_mcp_content(r)
        types = [getattr(c, "type", None) for c in content]
        assert "text" in types
        assert "image" in types


class TestImageBudget:
    def test_budget_exhaustion_emits_marker(self):
        big = "A" * ((TOTAL_IMAGE_BUDGET_BYTES + 100) * 4 // 3)  # > budget
        r = _scratch([
            _display({"image/png": {"encoding": "base64", "bytes": big, "size_bytes": TOTAL_IMAGE_BUDGET_BYTES + 100}}),
            _display({"image/png": {"encoding": "base64", "bytes": PNG_1x1, "size_bytes": 100}}),
        ])
        md = to_markdown(r)
        assert "exceeds" in md  # first image exceeds per-image cap
        # second image is small, should come through
        assert PNG_1x1 in md


class TestCellOutputsShape:
    def test_cells_block_label_and_body(self):
        result = {
            "cells": [
                {
                    "cell_id": "abcd",
                    "label": "@C",
                    "outputs": [_stream("stdout", "hello")],
                },
                {"cell_id": "efgh", "label": "@D", "outputs": []},
            ]
        }
        md = to_markdown(result)
        assert "=== @C  [abcd] ===" in md
        assert "hello" in md
        assert "=== @D  [efgh] ===" in md
        assert "(no outputs)" in md


class TestBuildToolContent:
    def test_builds_blocks_and_summary(self):
        tc = build_tool_content(_scratch([_stream("stdout", "ok")]))
        assert isinstance(tc, ToolContent)
        assert tc.blocks
        assert "ok" in tc.text_summary
        assert "status: ok" in tc.text_summary
