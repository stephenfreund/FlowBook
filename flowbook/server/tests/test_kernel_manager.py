"""Tests for FlowbookKernelClient and KernelConnectionManager."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from jupyter_client.session import Session

from flowbook.server.kernel_manager import FlowbookKernelClient, KernelConnectionManager


# ---------------------------------------------------------------------------
# FlowbookKernelClient
# ---------------------------------------------------------------------------


class TestFlowbookKernelClient:
    """Test FlowbookKernelClient.execute() message construction.

    The execute() method builds a message and sends it on shell_channel.
    Because BlockingKernelClient uses traitlets/ZMQ properties that are
    hard to mock, we test the logic by calling execute() on a fully mocked
    object that preserves the method's code.
    """

    def test_stores_kernel_id(self):
        with patch("flowbook.server.kernel_manager.BlockingKernelClient.__init__", return_value=None):
            client = FlowbookKernelClient(kernel_id="test-id")
        assert client.kernel_id == "test-id"

    def _call_execute(self, code="x = 1", **kwargs):
        """Call FlowbookKernelClient.execute logic with fully mocked internals."""
        mock_session = MagicMock()
        mock_msg = {"header": {"msg_id": "msg-123"}}
        mock_session.msg.return_value = mock_msg
        mock_shell = MagicMock()

        # Create a plain object to act as self, with needed attributes
        self_obj = MagicMock()
        self_obj.session = mock_session
        self_obj.shell_channel = mock_shell
        self_obj.allow_stdin = False

        # Call the unbound method
        result = FlowbookKernelClient.execute(self_obj, code, **kwargs)
        return result, mock_session, mock_shell, mock_msg

    def test_execute_injects_cell_id(self):
        result, mock_session, _, _ = self._call_execute("x = 1", cell_id="cell-A")
        assert result == "msg-123"
        metadata = mock_session.msg.call_args[1]["metadata"]
        assert metadata["cell_id"] == "cell-A"

    def test_execute_content_fields(self):
        _, mock_session, _, _ = self._call_execute("x = 1", silent=True, store_history=False)
        content = mock_session.msg.call_args[0][1]
        assert content["code"] == "x = 1"
        assert content["silent"] is True
        assert content["store_history"] is False
        assert content["user_expressions"] == {}

    def test_execute_includes_cell_metadata(self):
        _, mock_session, _, _ = self._call_execute(
            "y = 2", cell_id="cell-B", cell_metadata={"flowbook": {"key": "val"}}
        )
        metadata = mock_session.msg.call_args[1]["metadata"]
        assert metadata["cell_id"] == "cell-B"
        assert metadata["flowbook"] == {"key": "val"}

    def test_execute_sends_on_shell_channel(self):
        _, _, mock_shell, mock_msg = self._call_execute("z = 3")
        mock_shell.send.assert_called_once_with(mock_msg)

    def test_execute_defaults_user_expressions_to_empty(self):
        _, mock_session, _, _ = self._call_execute("a = 1")
        content = mock_session.msg.call_args[0][1]
        assert content["user_expressions"] == {}


# ---------------------------------------------------------------------------
# KernelConnectionManager
# ---------------------------------------------------------------------------


class TestKernelConnectionManager:
    def _make_manager(self):
        mock_app = MagicMock()
        return KernelConnectionManager(mock_app), mock_app

    def test_caching_returns_same_client(self):
        """Once a client is created, subsequent calls return the cached instance."""
        mgr, _ = self._make_manager()
        mock_client = MagicMock(spec=FlowbookKernelClient)
        mgr._kernel_clients["k1"] = mock_client

        assert mgr.get_kernel_client("k1") is mock_client

    def test_cleanup_client_stops_channels(self):
        mgr, _ = self._make_manager()
        mock_client = MagicMock(spec=FlowbookKernelClient)
        mgr._kernel_clients["k1"] = mock_client

        mgr.cleanup_client("k1")
        mock_client.stop_channels.assert_called_once()
        assert "k1" not in mgr._kernel_clients

    def test_cleanup_nonexistent_is_noop(self):
        mgr, _ = self._make_manager()
        mgr.cleanup_client("nonexistent")  # Should not raise

    def test_different_ids_are_independent(self):
        mgr, _ = self._make_manager()
        c1 = MagicMock(spec=FlowbookKernelClient)
        c2 = MagicMock(spec=FlowbookKernelClient)
        mgr._kernel_clients["k1"] = c1
        mgr._kernel_clients["k2"] = c2
        assert mgr.get_kernel_client("k1") is not mgr.get_kernel_client("k2")
