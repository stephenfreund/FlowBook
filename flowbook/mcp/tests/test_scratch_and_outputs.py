"""Unit tests for NotebookSession.scratch_work and .get_cell_outputs.

These tests stub the kernel/refresh layers so they don't need a running
kernel or Jupyter server. Integration coverage is in
test_session_integration.py.
"""

import base64
from unittest.mock import patch, MagicMock

import pytest

from flowbook.mcp.session import NotebookSession


PNG_B64 = base64.b64encode(b"\x89PNGfake").decode()


# ---------------------------------------------------------------------------
# scratch_work
# ---------------------------------------------------------------------------

class TestScratchWork:
    def test_requires_session_loaded(self):
        session = NotebookSession()
        with pytest.raises(Exception):
            session.scratch_work("x = 1")

    def test_delegates_to_kernel_helper_execute_scratch(self):
        session = NotebookSession()
        session._require_loaded = MagicMock()
        session.kernel_client = MagicMock()

        fake_result = {
            "status": "ok",
            "execution_time_ms": 1.23,
            "outputs": [],
            "error": None,
        }
        with patch("flowbook.mcp.session.KernelHelper.execute_scratch",
                   return_value=fake_result) as m:
            result = session.scratch_work("x = 1")

        m.assert_called_once()
        assert m.call_args.args[0] is session.kernel_client
        assert m.call_args.args[1] == "x = 1"
        assert result is fake_result

    def test_passes_through_timeout(self):
        session = NotebookSession()
        session._require_loaded = MagicMock()
        session.kernel_client = MagicMock()

        with patch("flowbook.mcp.session.KernelHelper.execute_scratch",
                   return_value={}) as m:
            session.scratch_work("x = 1", timeout=5.0)

        assert m.call_args.kwargs.get("timeout") == 5.0


# ---------------------------------------------------------------------------
# get_cell_outputs
# ---------------------------------------------------------------------------

def _session_with_notebook(cells):
    session = NotebookSession()
    session._require_loaded = MagicMock()
    session._refresh_from_contents_api = MagicMock()
    session.notebook = {"cells": cells}
    session.get_cell_order = MagicMock(return_value=[c["id"] for c in cells if c.get("cell_type") == "code"])

    def find(cid):
        for i, c in enumerate(cells):
            if c.get("id") == cid:
                return i, c
        raise ValueError(f"not found: {cid}")
    session._find_cell = MagicMock(side_effect=find)
    return session


class TestGetCellOutputs:
    def test_empty_list(self):
        session = _session_with_notebook([])
        result = session.get_cell_outputs([])
        assert result == {"cells": []}

    def test_returns_stream_outputs(self):
        cells = [{
            "id": "aaaa", "cell_type": "code",
            "outputs": [{"output_type": "stream", "name": "stdout", "text": "hello"}],
        }]
        result = _session_with_notebook(cells).get_cell_outputs(["aaaa"])
        assert len(result["cells"]) == 1
        cell = result["cells"][0]
        assert cell["cell_id"] == "aaaa"
        assert cell["label"] == "@A"
        assert cell["outputs"][0] == {"kind": "stream", "stream_name": "stdout", "text": "hello"}

    def test_returns_execute_result_with_mime_bundle(self):
        cells = [{
            "id": "aaaa", "cell_type": "code",
            "outputs": [{
                "output_type": "execute_result",
                "data": {"text/plain": "42", "image/png": PNG_B64},
                "execution_count": 1,
            }],
        }]
        result = _session_with_notebook(cells).get_cell_outputs(["aaaa"])
        data = result["cells"][0]["outputs"][0]["data"]
        assert data["text/plain"]["text"] == "42"
        assert data["image/png"]["encoding"] == "base64"
        assert data["image/png"]["bytes"] == PNG_B64

    def test_returns_display_data(self):
        cells = [{
            "id": "aaaa", "cell_type": "code",
            "outputs": [{
                "output_type": "display_data",
                "data": {"text/html": "<p>x</p>"},
            }],
        }]
        result = _session_with_notebook(cells).get_cell_outputs(["aaaa"])
        out = result["cells"][0]["outputs"][0]
        assert out["kind"] == "display_data"
        assert out["data"]["text/html"] == {"text": "<p>x</p>"}

    def test_error_output_becomes_text_plain(self):
        cells = [{
            "id": "aaaa", "cell_type": "code",
            "outputs": [{
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "bad",
                "traceback": ["line1", "line2"],
            }],
        }]
        result = _session_with_notebook(cells).get_cell_outputs(["aaaa"])
        out = result["cells"][0]["outputs"][0]
        assert out["kind"] == "error"
        text = out["data"]["text/plain"]["text"]
        assert "ValueError: bad" in text
        assert "line1" in text and "line2" in text

    def test_missing_cell_returns_error_placeholder(self):
        session = _session_with_notebook([{"id": "aaaa", "cell_type": "code", "outputs": []}])
        result = session.get_cell_outputs(["zzzz"])
        assert result["cells"][0]["cell_id"] == "zzzz"
        assert result["cells"][0]["outputs"][0]["kind"] == "error"
        assert "not found" in result["cells"][0]["outputs"][0]["data"]["text/plain"]["text"]

    def test_multiple_cells_returned_in_order(self):
        cells = [
            {"id": "aaaa", "cell_type": "code", "outputs": [{"output_type": "stream", "name": "stdout", "text": "A"}]},
            {"id": "bbbb", "cell_type": "code", "outputs": [{"output_type": "stream", "name": "stdout", "text": "B"}]},
            {"id": "cccc", "cell_type": "code", "outputs": []},
        ]
        result = _session_with_notebook(cells).get_cell_outputs(["aaaa", "cccc"])
        assert [c["cell_id"] for c in result["cells"]] == ["aaaa", "cccc"]
        assert result["cells"][0]["outputs"][0]["text"] == "A"
        assert result["cells"][1]["outputs"] == []

    def test_no_kernel_call(self):
        """get_cell_outputs must not touch the kernel — it's a model read."""
        session = _session_with_notebook([
            {"id": "aaaa", "cell_type": "code", "outputs": []}
        ])
        session.kernel_client = MagicMock()  # if we touched it, assertions below would show

        session.get_cell_outputs(["aaaa"])
        session.kernel_client.execute.assert_not_called()
