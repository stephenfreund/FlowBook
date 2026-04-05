"""Tests for flowbook server HTTP handlers."""

import json
import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from flowbook.server.handlers import (
    KernelDiscoveryHandler,
    CommandListHandler,
)


# ---------------------------------------------------------------------------
# KernelDiscoveryHandler._resolve_notebook_path
# (path traversal tests are in commands/tests/test_path_traversal.py —
#  these cover additional handler-level behaviour)
# ---------------------------------------------------------------------------


def _make_discovery_handler(server_root_dir=""):
    app = MagicMock()
    app.settings = {"server_root_dir": server_root_dir}
    handler = KernelDiscoveryHandler.__new__(KernelDiscoveryHandler)
    handler.application = app
    handler.request = MagicMock()
    # Bypass @tornado.web.authenticated
    handler._jupyter_current_user = "test-user"
    handler.current_user = "test-user"
    return handler


class TestKernelDiscoveryResolve:
    def test_relative_resolved_under_root(self, tmp_path):
        handler = _make_discovery_handler(str(tmp_path))
        result = handler._resolve_notebook_path("nb.ipynb")
        assert result == os.path.join(str(tmp_path), "nb.ipynb")

    def test_absolute_within_root_allowed(self, tmp_path):
        nb = os.path.join(str(tmp_path), "sub", "nb.ipynb")
        handler = _make_discovery_handler(str(tmp_path))
        assert handler._resolve_notebook_path(nb) == nb

    def test_traversal_blocked(self, tmp_path):
        handler = _make_discovery_handler(str(tmp_path))
        with pytest.raises(ValueError, match="escapes"):
            handler._resolve_notebook_path("../../etc/passwd")

    def test_no_root_allows_anything(self):
        handler = _make_discovery_handler("")
        result = handler._resolve_notebook_path("/any/path.ipynb")
        assert result == "/any/path.ipynb"


# ---------------------------------------------------------------------------
# KernelDiscoveryHandler.get / put
# ---------------------------------------------------------------------------


class TestKernelDiscoveryGet:
    def test_returns_discovery_when_found(self, tmp_path):
        handler = _make_discovery_handler(str(tmp_path))
        handler.finish = MagicMock()
        handler.set_status = MagicMock()
        disc_data = {"connection_file": "/tmp/kernel.json", "pid": 123}

        with patch("flowbook.server.handlers.read_discovery", return_value=disc_data):
            import asyncio
            asyncio.get_event_loop().run_until_complete(handler.get("nb.ipynb"))

        handler.finish.assert_called_once()
        body = json.loads(handler.finish.call_args[0][0])
        assert body["pid"] == 123
        handler.set_status.assert_not_called()

    def test_returns_404_when_not_found(self, tmp_path):
        handler = _make_discovery_handler(str(tmp_path))
        handler.finish = MagicMock()
        handler.set_status = MagicMock()

        with patch("flowbook.server.handlers.read_discovery", return_value=None):
            import asyncio
            asyncio.get_event_loop().run_until_complete(handler.get("nb.ipynb"))

        handler.set_status.assert_called_once_with(404)


class TestKernelDiscoveryPut:
    def test_writes_discovery(self, tmp_path):
        handler = _make_discovery_handler(str(tmp_path))
        handler.finish = MagicMock()
        handler.get_json_body = MagicMock(return_value={
            "connection_file": "kernel-abc.json",
            "kernel_name": "flowbook_kernel",
        })

        with patch("flowbook.server.handlers.write_discovery", return_value="/tmp/disc.json") as mock_write, \
             patch.object(handler, "_get_kernel_pid", return_value=(42, "/full/kernel-abc.json")):
            import asyncio
            asyncio.get_event_loop().run_until_complete(handler.put("nb.ipynb"))

        mock_write.assert_called_once()
        body = json.loads(handler.finish.call_args[0][0])
        assert body["discovery_file"] == "/tmp/disc.json"


# ---------------------------------------------------------------------------
# KernelDiscoveryHandler._get_kernel_pid
# ---------------------------------------------------------------------------


class TestGetKernelPid:
    def test_extracts_pid_from_kernel_manager(self):
        handler = _make_discovery_handler("")
        mock_kernel = MagicMock()
        mock_kernel.provisioner.pid = 999
        mock_kernel.connection_file = "/abs/kernel-abc.json"
        mock_km = MagicMock()
        mock_km.get_kernel.return_value = mock_kernel
        handler.application.settings["serverapp"] = MagicMock()
        handler.application.settings["serverapp"].kernel_manager = mock_km

        pid, conn = handler._get_kernel_pid("kernel-abc.json")
        assert pid == 999
        assert conn == "/abs/kernel-abc.json"

    def test_returns_zero_for_non_matching_filename(self):
        handler = _make_discovery_handler("")
        pid, conn = handler._get_kernel_pid("not-a-kernel-file.txt")
        assert pid == 0
        assert conn == "not-a-kernel-file.txt"

    def test_returns_zero_on_exception(self):
        handler = _make_discovery_handler("")
        handler.application.settings["serverapp"] = MagicMock()
        handler.application.settings["serverapp"].kernel_manager.get_kernel.side_effect = KeyError("nope")

        pid, conn = handler._get_kernel_pid("kernel-abc123.json")
        assert pid == 0
