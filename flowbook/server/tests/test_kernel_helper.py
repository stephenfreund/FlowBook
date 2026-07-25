"""Tests for KernelHelper.execute_code message handling.

Regression tests for the 2026-07-12 audit findings C2/C3/R3:
- C2: IOPub messages from OTHER executions on a shared kernel were drained
  and discarded during execute_code; they are now handed to on_foreign_msg.
- C3: the shell reply was not matched to the request msg_id, so a stale
  reply from an earlier timed-out request could be mis-attributed.
"""

from queue import Empty
from unittest.mock import patch

import pytest

from flowbook.server.kernel_helper import KernelHelper, extract_flowbook_metadata


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


class TestInjectCsvDownsampling:
    """Audit S5: proportion is interpolated into kernel code — it must be
    coerced to float and range-checked so no string/expression can reach
    the code template."""

    def _run(self, proportion):
        captured = {}

        def fake_execute(kernel_client, code, *args, **kwargs):
            captured["code"] = code
            return {"status": "ok", "outputs": [], "flowbook_messages": [],
                    "execution_count": None, "error_message": None}

        with patch.object(KernelHelper, "execute_code", side_effect=fake_execute):
            result = KernelHelper.inject_csv_downsampling(object(), proportion)
        return result, captured.get("code")

    def test_valid_float(self):
        result, code = self._run(0.5)
        assert result["status"] == "ok"
        assert "int(len(df) * 0.5)" in code

    def test_string_number_coerced(self):
        result, code = self._run("0.5")
        assert result["status"] == "ok"
        assert "int(len(df) * 0.5)" in code

    def test_injection_string_rejected(self):
        with patch.object(KernelHelper, "execute_code") as mock_exec:
            with pytest.raises(ValueError):
                KernelHelper.inject_csv_downsampling(object(), "1); import os #")
            mock_exec.assert_not_called()

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            KernelHelper.inject_csv_downsampling(object(), 1.5)
        with pytest.raises(ValueError):
            KernelHelper.inject_csv_downsampling(object(), -0.1)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            KernelHelper.inject_csv_downsampling(object(), float("nan"))

    def test_none_rejected(self):
        with pytest.raises(ValueError):
            KernelHelper.inject_csv_downsampling(object(), None)


class TestExtractFlowbookMetadata:
    """Shared metadata-extraction helper (audit A4)."""

    def test_returns_first_metadata_message(self):
        result = {
            "flowbook_messages": [
                {"type": "status", "text": "..."},
                {"type": "metadata", "cell_id": "abcd"},
                {"type": "metadata", "cell_id": "wxyz"},
            ]
        }
        assert extract_flowbook_metadata(result) == {
            "type": "metadata", "cell_id": "abcd"
        }

    def test_returns_none_when_absent(self):
        assert extract_flowbook_metadata({"flowbook_messages": []}) is None
        assert extract_flowbook_metadata({}) is None
