"""Tests for KernelCommandClient."""

import time
from unittest.mock import MagicMock, patch, call

import pytest

from flowbook.kernel_support.kernel_command_client import (
    KernelCommandClient,
    KernelCommandError,
)
from flowbook.kernel_support.kernel_commands import (
    CheckpointSaveResponse,
    CheckpointListResponse,
    ForceCheckpointsResponse,
    FinalMessage,
    ProgressMessage,
)


def _mock_kernel_client():
    kc = MagicMock()
    kc.session = MagicMock()
    kc.shell_channel = MagicMock()
    kc.iopub_channel = MagicMock()
    return kc


def _make_final_response(response_dict, ok=True, error_msg=None):
    """Create a mock IOPub comm_msg containing a FinalMessage."""
    return {
        "msg_type": "comm_msg",
        "content": {
            "comm_id": None,  # Will be matched dynamically
            "data": {
                "type": "final",
                "ok": ok,
                "response": response_dict,
                "error": error_msg,
            },
        },
    }


def _make_progress_msg(message="working..."):
    return {
        "msg_type": "comm_msg",
        "content": {
            "comm_id": None,
            "data": {
                "type": "progress",
                "message": message,
            },
        },
    }


class TestSendCommand:
    def test_successful_command(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=5, retries=1)

        response_dict = {"status": "ok", "message": "done", "checkpoints": ["a", "b"]}
        final = _make_final_response(response_dict)

        def get_msg_side_effect(timeout=1.0):
            # Set comm_id to match whatever was opened
            comm_id = kc.session.msg.call_args[0][1]["comm_id"]
            final["content"]["comm_id"] = comm_id
            return final

        kc.iopub_channel.get_msg.side_effect = get_msg_side_effect

        resp = client.checkpoint_list(timeout=5)
        assert resp.status == "ok"
        assert resp.checkpoints == ["a", "b"]

    def test_timeout_raises(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=0.1, retries=1)

        # Never return a matching message
        kc.iopub_channel.get_msg.side_effect = TimeoutError()

        with pytest.raises(KernelCommandError, match="after 1 attempts"):
            client.checkpoint_list(timeout=0.1)

    def test_error_response_retries(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=5, retries=2)

        error_resp = {
            "status": "error",
            "message": "temporary failure",
            "checkpoints": [],
        }
        ok_resp = {
            "status": "ok",
            "message": "done",
            "checkpoints": ["cp1"],
        }

        call_count = [0]

        def get_msg_side_effect(timeout=1.0):
            comm_id = kc.session.msg.call_args[0][1]["comm_id"]
            call_count[0] += 1
            if call_count[0] <= 1:
                resp = _make_final_response(error_resp)
            else:
                resp = _make_final_response(ok_resp)
            resp["content"]["comm_id"] = comm_id
            return resp

        kc.iopub_channel.get_msg.side_effect = get_msg_side_effect

        with patch("time.sleep"):  # Don't actually sleep
            resp = client.checkpoint_list(timeout=5)

        assert resp.status == "ok"

    def test_progress_callback(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=5, retries=1)

        messages_received = []
        call_count = [0]

        def get_msg_side_effect(timeout=1.0):
            comm_id = kc.session.msg.call_args[0][1]["comm_id"]
            call_count[0] += 1
            if call_count[0] == 1:
                msg = _make_progress_msg("50%")
                msg["content"]["comm_id"] = comm_id
                return msg
            else:
                resp = _make_final_response({"status": "ok", "message": "done", "checkpoints": []})
                resp["content"]["comm_id"] = comm_id
                return resp

        kc.iopub_channel.get_msg.side_effect = get_msg_side_effect

        resp = client.checkpoint_list(timeout=5)
        assert resp.status == "ok"

    def test_sends_comm_close_on_success(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=5, retries=1)

        def get_msg_side_effect(timeout=1.0):
            comm_id = kc.session.msg.call_args[0][1]["comm_id"]
            resp = _make_final_response({"status": "ok", "message": "done", "checkpoints": []})
            resp["content"]["comm_id"] = comm_id
            return resp

        kc.iopub_channel.get_msg.side_effect = get_msg_side_effect

        client.checkpoint_list()

        # Should have sent comm_open and then comm_close
        msg_types = [c[0][0] for c in kc.session.msg.call_args_list]
        assert "comm_open" in msg_types
        assert "comm_close" in msg_types


class TestKernelCommandError:
    def test_is_exception(self):
        assert issubclass(KernelCommandError, Exception)

    def test_message(self):
        e = KernelCommandError("test error")
        assert str(e) == "test error"


class TestCheckpointSave:
    def test_success(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=5, retries=1)

        def get_msg_side_effect(timeout=1.0):
            comm_id = kc.session.msg.call_args[0][1]["comm_id"]
            resp = _make_final_response({
                "status": "ok",
                "message": "saved",
                "saved": {"x": {"kind": "Atomic", "type_name": "int"}},
                "removed": {},
                "duration": 0.5,
            })
            resp["content"]["comm_id"] = comm_id
            return resp

        kc.iopub_channel.get_msg.side_effect = get_msg_side_effect

        resp = client.checkpoint_save("test_cp")
        assert resp.status == "ok"
        assert "x" in resp.saved


class TestForceCheckpoints:
    def test_success(self):
        kc = _mock_kernel_client()
        client = KernelCommandClient(kc, timeout=5, retries=1)

        def get_msg_side_effect(timeout=1.0):
            comm_id = kc.session.msg.call_args[0][1]["comm_id"]
            resp = _make_final_response({
                "status": "ok",
                "message": "enabled",
                "enabled": True,
            })
            resp["content"]["comm_id"] = comm_id
            return resp

        kc.iopub_channel.get_msg.side_effect = get_msg_side_effect

        resp = client.force_checkpoints(True)
        assert resp.status == "ok"
        assert resp.enabled is True
