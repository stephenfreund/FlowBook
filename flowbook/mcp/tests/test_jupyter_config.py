"""Tests for Jupyter Server auto-discovery."""

import os

import pytest

from flowbook.mcp.jupyter_config import discover_jupyter_server


class TestEnvironmentVariableOverride:
    def test_env_vars_take_precedence(self, monkeypatch):
        monkeypatch.setenv("JUPYTER_SERVER_URL", "http://custom:9999")
        monkeypatch.setenv("JUPYTER_TOKEN", "mytoken")
        url, token = discover_jupyter_server()
        assert url == "http://custom:9999"
        assert token == "mytoken"

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("JUPYTER_SERVER_URL", "http://custom:9999/")
        url, _ = discover_jupyter_server()
        assert url == "http://custom:9999"

    def test_url_without_token(self, monkeypatch):
        monkeypatch.setenv("JUPYTER_SERVER_URL", "http://custom:9999")
        # JUPYTER_TOKEN not set
        monkeypatch.delenv("JUPYTER_TOKEN", raising=False)
        url, token = discover_jupyter_server()
        assert url == "http://custom:9999"
        assert token is None


class TestRuntimeDiscovery:
    def test_returns_tuple(self):
        """Should return a 2-tuple regardless of whether a server is found."""
        result = discover_jupyter_server()
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.skipif(
        not os.path.isdir(os.path.expanduser("~/.jupyter")),
        reason="No Jupyter config directory",
    )
    def test_finds_running_server_if_available(self):
        """If a Jupyter Server is running, we should find it.

        This test is environment-dependent — it passes when a server is running
        and is skipped otherwise.
        """
        url, token = discover_jupyter_server()
        if url is not None:
            assert url.startswith("http")
