"""Tests for KernelHelper.execute_code message handling.

Regression tests for the 2026-07-12 audit findings C2/C3/R3:
- C2: IOPub messages from OTHER executions on a shared kernel were drained
  and discarded during execute_code; they are now handed to on_foreign_msg.
- C3: the shell reply was not matched to the request msg_id, so a stale
  reply from an earlier timed-out request could be mis-attributed.
"""

from queue import Empty

from flowbook.server.kernel_helper import KernelHelper


def _iopub(parent_id, msg_type, content):
    return {
        "parent_header": {"msg_id": parent_id},
        "header": {"msg_type": msg_type},
        "content": content,
    }


def _idle(parent_id):
    return _iopub(parent_id, "status", {"execution_state": "idle"})


def _shell_reply(parent_id, status="ok", **content):
    return {
        "parent_header": {"msg_id": parent_id},
        "content": {"status": status, **content},
    }


class FakeKernelClient:
    """Minimal kernel client: canned IOPub/shell message queues."""

    MSG_ID = "my-msg-id"

    def __init__(self, iopub_msgs, shell_msgs):
        self._iopub = list(iopub_msgs)
        self._shell = list(shell_msgs)

    def execute(self, code, **kwargs):
        return self.MSG_ID

    def get_iopub_msg(self, timeout=None):
        if self._iopub:
            return self._iopub.pop(0)
        raise Empty

    def get_shell_msg(self, timeout=None):
        if self._shell:
            return self._shell.pop(0)
        raise Empty


FOREIGN_UPDATE = _iopub(
    "jupyterlab-run",
    "flowbook_update",
    {"flowbook": {"type": "metadata", "cell_id": "abcd", "stale_cells": []}},
)


class TestForeignIopubMessages:
    def test_foreign_message_passed_to_handler(self):
        received = []
        client = FakeKernelClient(
            iopub_msgs=[FOREIGN_UPDATE, _idle(FakeKernelClient.MSG_ID)],
            shell_msgs=[_shell_reply(FakeKernelClient.MSG_ID)],
        )
        result = KernelHelper.execute_code(
            client, "x = 1", timeout=5, on_foreign_msg=received.append
        )

        assert result["status"] == "ok"
        assert received == [FOREIGN_UPDATE]
        # The foreign message must not leak into this execution's results
        assert result["flowbook_messages"] == []
        assert result["outputs"] == []

    def test_foreign_message_dropped_without_handler(self):
        client = FakeKernelClient(
            iopub_msgs=[FOREIGN_UPDATE, _idle(FakeKernelClient.MSG_ID)],
            shell_msgs=[_shell_reply(FakeKernelClient.MSG_ID)],
        )
        result = KernelHelper.execute_code(client, "x = 1", timeout=5)
        assert result["status"] == "ok"

    def test_handler_error_is_swallowed(self):
        def bad_handler(msg):
            raise RuntimeError("boom")

        client = FakeKernelClient(
            iopub_msgs=[FOREIGN_UPDATE, _idle(FakeKernelClient.MSG_ID)],
            shell_msgs=[_shell_reply(FakeKernelClient.MSG_ID)],
        )
        result = KernelHelper.execute_code(
            client, "x = 1", timeout=5, on_foreign_msg=bad_handler
        )
        assert result["status"] == "ok"

    def test_own_flowbook_update_still_collected(self):
        own_update = _iopub(
            FakeKernelClient.MSG_ID,
            "flowbook_update",
            {"flowbook": {"type": "metadata", "cell_id": "wxyz"}},
        )
        client = FakeKernelClient(
            iopub_msgs=[own_update, _idle(FakeKernelClient.MSG_ID)],
            shell_msgs=[_shell_reply(FakeKernelClient.MSG_ID)],
        )
        result = KernelHelper.execute_code(client, "x = 1", timeout=5)
        assert result["flowbook_messages"] == [
            {"type": "metadata", "cell_id": "wxyz"}
        ]


class TestShellReplyMatching:
    def test_stale_reply_discarded(self):
        """A reply abandoned by an earlier timed-out request must not be
        attributed to this execution."""
        client = FakeKernelClient(
            iopub_msgs=[_idle(FakeKernelClient.MSG_ID)],
            shell_msgs=[
                _shell_reply("earlier-abandoned-request", status="ok"),
                _shell_reply(
                    FakeKernelClient.MSG_ID,
                    status="error",
                    ename="ValueError",
                    evalue="bad",
                    traceback=["ValueError: bad"],
                ),
            ],
        )
        result = KernelHelper.execute_code(client, "raise ValueError", timeout=5)
        assert result["status"] == "error"
        assert "ValueError" in (result["error_message"] or "")

    def test_missing_reply_is_tolerated(self):
        client = FakeKernelClient(
            iopub_msgs=[_idle(FakeKernelClient.MSG_ID)],
            shell_msgs=[],
        )
        result = KernelHelper.execute_code(client, "x = 1", timeout=5)
        assert result["status"] == "ok"
