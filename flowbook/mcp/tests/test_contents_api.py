"""Tests for Contents API integration in NotebookSession."""

import io
import json
import time
from http.client import HTTPResponse
from unittest.mock import MagicMock, patch

import pytest

from flowbook.scripts.fix_repro_errors import get_cell_source
from flowbook.mcp.session import NotebookSession


def _make_notebook(cells):
    """Create a minimal nbformat notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells,
    }


def _make_code_cell(cell_id, source, outputs=None, execution_count=None):
    """Create a minimal code cell dict."""
    return {
        "id": cell_id,
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": outputs or [],
        "execution_count": execution_count,
    }


def _make_markdown_cell(cell_id, source):
    """Create a minimal markdown cell dict."""
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "source": source,
        "metadata": {},
    }


def _mock_urlopen_response(notebook_dict):
    """Create a mock urlopen response that returns a Contents API response."""
    contents_response = json.dumps({"content": notebook_dict}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = contents_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _setup_session_with_api(cells, contents_path="demos/test.ipynb"):
    """Create a NotebookSession with Contents API configured."""
    session = NotebookSession()
    session.notebook = _make_notebook(cells)
    session.notebook_path = "/abs/path/test.ipynb"
    session._jupyter_server_url = "http://localhost:8888"
    session._jupyter_token = "test-token"
    session._jupyter_contents_path = contents_path
    session._last_contents_refresh = 0  # Allow immediate refresh
    return session


class TestRefreshFromContentsApi:
    """Tests for _refresh_from_contents_api."""

    def test_updates_source(self):
        """Cell source is updated from API, outputs/metadata preserved."""
        local_cells = [
            _make_code_cell("A", "x = 1", outputs=[{"output_type": "stream", "text": "1"}], execution_count=1),
        ]
        session = _setup_session_with_api(local_cells)
        session.cell_flowbook_meta["A"] = {"read_locs": []}

        api_cells = [_make_code_cell("A", "x = 42")]
        api_notebook = _make_notebook(api_cells)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(api_notebook)):
            session._refresh_from_contents_api()

        cell = session.notebook["cells"][0]
        assert get_cell_source(cell) == "x = 42"  # Source updated
        assert cell["outputs"] == [{"output_type": "stream", "text": "1"}]  # Preserved
        assert cell["execution_count"] == 1  # Preserved
        assert session.cell_flowbook_meta["A"] == {"read_locs": []}  # Preserved

    def test_rate_limited(self):
        """Second call within 0.2s is skipped."""
        session = _setup_session_with_api([_make_code_cell("A", "x = 1")])

        api_notebook = _make_notebook([_make_code_cell("A", "x = 99")])
        mock_resp = _mock_urlopen_response(api_notebook)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            session._refresh_from_contents_api()
            assert mock_open.call_count == 1

            # Immediate second call should be skipped
            session._refresh_from_contents_api()
            assert mock_open.call_count == 1  # No additional call

    def test_adds_new_cells(self):
        """Cells added in JupyterLab appear in MCP."""
        local_cells = [_make_code_cell("A", "x = 1")]
        session = _setup_session_with_api(local_cells)

        api_cells = [
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),  # New cell
        ]
        api_notebook = _make_notebook(api_cells)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(api_notebook)):
            session._refresh_from_contents_api()

        assert len(session.notebook["cells"]) == 2
        assert session.notebook["cells"][1]["id"] == "B"
        assert session.notebook["cells"][1]["source"] == "y = 2"

    def test_removes_deleted_cells(self):
        """Cells deleted in JupyterLab are removed from MCP."""
        local_cells = [
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ]
        session = _setup_session_with_api(local_cells)
        session.executed_cells.add("B")
        session.cell_flowbook_meta["B"] = {"some": "meta"}
        session.cell_status["B"] = "ok"
        session._stale_cells.add("B")

        # API only has cell A (B was deleted in JupyterLab)
        api_cells = [_make_code_cell("A", "x = 1")]
        api_notebook = _make_notebook(api_cells)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(api_notebook)):
            session._refresh_from_contents_api()

        assert len(session.notebook["cells"]) == 1
        assert session.notebook["cells"][0]["id"] == "A"
        # Tracking cleaned up
        assert "B" not in session.executed_cells
        assert "B" not in session.cell_flowbook_meta
        assert "B" not in session.cell_status
        assert "B" not in session._stale_cells

    def test_handles_api_failure(self):
        """API failure is silent — no exception raised."""
        session = _setup_session_with_api([_make_code_cell("A", "x = 1")])

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            session._refresh_from_contents_api()  # Should not raise

        assert get_cell_source(session.notebook["cells"][0]) == "x = 1"  # Unchanged

    def test_noop_without_contents_path(self):
        """Does nothing when Contents API not configured."""
        session = NotebookSession()
        session.notebook = _make_notebook([_make_code_cell("A", "x = 1")])
        # _jupyter_contents_path is None
        session._refresh_from_contents_api()  # Should not raise

    def test_preserves_cell_order_from_api(self):
        """Cells are reordered to match API order."""
        local_cells = [
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ]
        session = _setup_session_with_api(local_cells)

        # API has cells in reversed order
        api_cells = [
            _make_code_cell("B", "y = 2"),
            _make_code_cell("A", "x = 1"),
        ]
        api_notebook = _make_notebook(api_cells)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(api_notebook)):
            session._refresh_from_contents_api()

        assert session.notebook["cells"][0]["id"] == "B"
        assert session.notebook["cells"][1]["id"] == "A"


class TestPutContentsApi:
    """Tests for _put_contents_api."""

    def test_sends_notebook(self):
        """PUT sends the full notebook as JSON."""
        session = _setup_session_with_api([_make_code_cell("A", "x = 1")])

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            # Skip the refresh GET inside _put_contents_api
            session._last_contents_refresh = time.time()
            result = session._put_contents_api()

        assert result is not None
        # Verify the PUT request
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert req.method == "PUT"
        body = json.loads(req.data)
        assert body["type"] == "notebook"
        assert body["format"] == "json"
        assert "cells" in body["content"]

    def test_returns_none_on_failure(self):
        """Returns None on API failure (caller falls back to disk)."""
        session = _setup_session_with_api([_make_code_cell("A", "x = 1")])
        session._last_contents_refresh = time.time()  # Skip refresh

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = session._put_contents_api()

        assert result is None

    def test_noop_without_contents_path(self):
        """Returns None when Contents API not configured."""
        session = NotebookSession()
        session.notebook = _make_notebook([_make_code_cell("A", "x = 1")])
        result = session._put_contents_api()
        assert result is None


class TestSetupContentsApi:
    """Tests for _setup_contents_api."""

    def test_configures_on_success(self):
        """Sets up Contents API vars when server is available."""
        session = NotebookSession()
        session.notebook = _make_notebook([])

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"

        with patch("flowbook.mcp.session.discover_jupyter_server", return_value=("http://localhost:8888", "tok")), \
             patch("flowbook.mcp.session.discover_jupyter_server_root", return_value="/server/root"), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = session._setup_contents_api("/server/root/demos/test.ipynb")

        assert result == " [live sync]"
        assert session._jupyter_server_url == "http://localhost:8888"
        assert session._jupyter_token == "tok"
        assert session._jupyter_contents_path == "demos/test.ipynb"

    def test_returns_empty_without_server(self):
        """Returns empty string when no Jupyter server found."""
        session = NotebookSession()

        with patch("flowbook.mcp.session.discover_jupyter_server", return_value=(None, None)):
            result = session._setup_contents_api("/some/path.ipynb")

        assert result == ""
        assert session._jupyter_contents_path is None


class TestSaveWithContentsApi:
    """Tests for save() routing."""

    def test_save_uses_api_when_available(self):
        """save() uses Contents API PUT when available."""
        session = _setup_session_with_api([_make_code_cell("A", "x = 1")])

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("flowbook.mcp.session.cli_save_notebook") as mock_disk:
            session._last_contents_refresh = time.time()
            session.save()
            mock_disk.assert_not_called()

    def test_save_falls_back_to_disk(self):
        """save() writes to disk when Contents API not configured."""
        session = NotebookSession()
        session.notebook = _make_notebook([_make_code_cell("A", "x = 1")])
        session.notebook_path = "/tmp/test.ipynb"

        with patch("flowbook.mcp.session.cli_save_notebook", return_value="/tmp/test.ipynb") as mock_disk:
            result = session.save()
            mock_disk.assert_called_once()
            assert result == "/tmp/test.ipynb"
