"""Tests for flowbook server HTTP handlers."""

import json
import os
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from flowbook.server.handlers import (
    FlowbookCommandHandler,
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

        with patch("flowbook.server.handlers.write_discovery", return_value=True) as mock_write, \
             patch.object(handler, "_get_kernel_pid", return_value=(42, "/full/kernel-abc.json")):
            import asyncio
            asyncio.get_event_loop().run_until_complete(handler.put("nb.ipynb"))

        mock_write.assert_called_once()
        assert mock_write.call_args.kwargs["pid"] == 42
        body = json.loads(handler.finish.call_args[0][0])
        assert body["written"] is True

    def test_refuses_pid_zero(self, tmp_path):
        """pid=0 (lookup failure) must not produce a doomed discovery file."""
        handler = _make_discovery_handler(str(tmp_path))
        handler.finish = MagicMock()
        handler.set_status = MagicMock()
        handler.get_json_body = MagicMock(return_value={
            "connection_file": "not-a-kernel-file.txt",
            "kernel_name": "flowbook_kernel",
        })

        with patch("flowbook.server.handlers.write_discovery") as mock_write, \
             patch.object(handler, "_get_kernel_pid", return_value=(0, "not-a-kernel-file.txt")):
            import asyncio
            asyncio.get_event_loop().run_until_complete(handler.put("nb.ipynb"))

        mock_write.assert_not_called()
        # Still HTTP 200 — the frontend ignores the body
        handler.set_status.assert_not_called()
        body = json.loads(handler.finish.call_args[0][0])
        assert body["written"] is False
        assert "reason" in body

    def test_reports_refused_write(self, tmp_path):
        """write_discovery returning False is surfaced as written=false."""
        handler = _make_discovery_handler(str(tmp_path))
        handler.finish = MagicMock()
        handler.set_status = MagicMock()
        handler.get_json_body = MagicMock(return_value={
            "connection_file": "kernel-abc.json",
        })

        with patch("flowbook.server.handlers.write_discovery", return_value=False), \
             patch.object(handler, "_get_kernel_pid", return_value=(42, "/full/kernel-abc.json")):
            import asyncio
            asyncio.get_event_loop().run_until_complete(handler.put("nb.ipynb"))

        handler.set_status.assert_not_called()
        body = json.loads(handler.finish.call_args[0][0])
        assert body["written"] is False
        assert "reason" in body


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


# ---------------------------------------------------------------------------
# FlowbookCommandHandler.post — command timeout enforcement
# ---------------------------------------------------------------------------


def _make_command_handler(command):
    """Build a FlowbookCommandHandler wired to a stub registry/command."""
    registry = MagicMock()
    registry.get_command.return_value = command
    app = MagicMock()
    app.settings = {}
    handler = FlowbookCommandHandler.__new__(FlowbookCommandHandler)
    handler.application = app
    handler.request = MagicMock()
    handler._jupyter_current_user = "test-user"
    handler.current_user = "test-user"
    handler.initialize(registry=registry, connection_manager=None)
    handler.finish = MagicMock()
    handler.set_status = MagicMock()
    handler.get_json_body = MagicMock(return_value={
        "command": "stub",
        "notebook": {"cells": []},
    })
    return handler


class _StubCommand:
    """Minimal command stub: optionally blocks on an event until released."""

    command_name = "stub"
    requires_kernel = False

    def __init__(self, timeout, release=None, result=None):
        self.timeout = timeout
        self._release = release
        self._result = result or {"notebook": {"cells": []}, "metadata": {}}

    async def process(self, notebook_content, kernel_client=None,
                      selected_cell_ids=None, **kwargs):
        if self._release is not None:
            # Block the worker thread until the test releases it.
            self._release.wait(timeout=10)
        return self._result


class TestCommandTimeout:
    def test_hung_command_returns_504(self):
        import asyncio
        import threading

        release = threading.Event()
        command = _StubCommand(timeout=0.2, release=release)
        handler = _make_command_handler(command)

        try:
            asyncio.get_event_loop().run_until_complete(handler.post())
        finally:
            release.set()  # let the worker thread exit

        handler.set_status.assert_called_once_with(504)
        body = json.loads(handler.finish.call_args[0][0])
        assert "stub" in body["error"]
        assert "0.2" in body["error"]

    def test_fast_command_completes(self):
        import asyncio

        command = _StubCommand(timeout=30)
        handler = _make_command_handler(command)

        asyncio.get_event_loop().run_until_complete(handler.post())

        handler.set_status.assert_not_called()
        body = json.loads(handler.finish.call_args[0][0])
        assert body == {"notebook": {"cells": []}, "metadata": {}}
