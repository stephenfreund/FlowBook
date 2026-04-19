"""Tests for KernelHelper.execute_scratch and _filter_mime_bundle."""

import base64
from unittest.mock import MagicMock

import pytest

from flowbook.server.kernel_helper import KernelHelper, _filter_mime_bundle


PNG_B64 = base64.b64encode(b"\x89PNG\r\nfake").decode()


class TestFilterMimeBundle:
    def test_image_becomes_base64_payload(self):
        out = _filter_mime_bundle({"image/png": PNG_B64})
        assert out["image/png"]["encoding"] == "base64"
        assert out["image/png"]["bytes"] == PNG_B64
        assert out["image/png"]["size_bytes"] == (len(PNG_B64) * 3) // 4

    def test_text_plain_kept(self):
        out = _filter_mime_bundle({"text/plain": "hello"})
        assert out["text/plain"] == {"text": "hello"}

    def test_text_html_kept(self):
        html = "<table>x</table>"
        out = _filter_mime_bundle({"text/html": html})
        assert out["text/html"] == {"text": html}

    def test_unknown_mime_retained_as_size_stub(self):
        out = _filter_mime_bundle({"application/vnd.plotly.v1+json": "{\"foo\":1}"})
        assert out["application/vnd.plotly.v1+json"] == {"size_bytes": len("{\"foo\":1}")}


class _FakeMsg(dict):
    """Minimal fake kernel message — dict with the fields execute_scratch
    inspects. Convenience wrapper so we can enqueue them with kw args."""
    @classmethod
    def iopub(cls, parent_id, msg_type, **content):
        return {
            "parent_header": {"msg_id": parent_id},
            "header": {"msg_type": msg_type},
            "content": content,
        }


def _kernel_client(msg_id="req-1", iopub_msgs=None, shell_msg=None):
    """Build a mock kernel client whose execute() returns msg_id and whose
    get_iopub_msg() pops from iopub_msgs (an idle-status message ends the loop)."""
    client = MagicMock()
    client.execute.return_value = msg_id

    queue = list(iopub_msgs or [])
    queue.append(_FakeMsg.iopub(msg_id, "status", execution_state="idle"))
    def get_iopub(timeout=1.0):
        if not queue:
            raise Exception("no more messages")
        return queue.pop(0)
    client.get_iopub_msg.side_effect = get_iopub

    client.get_shell_msg.return_value = shell_msg or {"content": {"status": "ok"}}
    return client


class TestExecuteScratch:
    def test_sends_silent_no_history_and_isolate_flag(self):
        client = _kernel_client()
        KernelHelper.execute_scratch(client, "x = 1")
        kwargs = client.execute.call_args.kwargs
        args = client.execute.call_args.args
        assert args[0] == "x = 1" or kwargs.get("code") == "x = 1"
        assert kwargs["silent"] is True
        assert kwargs["store_history"] is False
        meta = kwargs["cell_metadata"]
        assert meta["flowbook_isolate"] is True
        assert "timeout" in meta
        # cell_id should NOT be set for scratch work
        assert "cell_id" not in kwargs or kwargs["cell_id"] is None

    def test_captures_stdout_and_stderr_separately(self):
        client = _kernel_client(iopub_msgs=[
            _FakeMsg.iopub("req-1", "stream", name="stdout", text="out1"),
            _FakeMsg.iopub("req-1", "stream", name="stderr", text="err1"),
        ])
        result = KernelHelper.execute_scratch(client, "print(1)")
        kinds = [(o["kind"], o.get("stream_name"), o.get("text")) for o in result["outputs"]]
        assert ("stream", "stdout", "out1") in kinds
        assert ("stream", "stderr", "err1") in kinds

    def test_captures_execute_result_with_mime_filter(self):
        client = _kernel_client(iopub_msgs=[
            _FakeMsg.iopub("req-1", "execute_result",
                           data={"text/plain": "42", "image/png": PNG_B64},
                           execution_count=1),
        ])
        result = KernelHelper.execute_scratch(client, "42")
        outs = [o for o in result["outputs"] if o["kind"] == "execute_result"]
        assert len(outs) == 1
        data = outs[0]["data"]
        assert data["text/plain"]["text"] == "42"
        assert data["image/png"]["bytes"] == PNG_B64
        assert data["image/png"]["encoding"] == "base64"

    def test_captures_display_data(self):
        client = _kernel_client(iopub_msgs=[
            _FakeMsg.iopub("req-1", "display_data",
                           data={"text/html": "<b>hi</b>"}),
        ])
        result = KernelHelper.execute_scratch(client, "display(...)")
        outs = [o for o in result["outputs"] if o["kind"] == "display_data"]
        assert outs and outs[0]["data"]["text/html"]["text"] == "<b>hi</b>"

    def test_captures_error_details(self):
        client = _kernel_client(iopub_msgs=[
            _FakeMsg.iopub("req-1", "error",
                           ename="ValueError", evalue="bad",
                           traceback=["line1 ", "line2"]),
        ])
        result = KernelHelper.execute_scratch(client, "raise ValueError('bad')")
        assert result["status"] == "error"
        assert result["error"]["ename"] == "ValueError"
        assert result["error"]["evalue"] == "bad"
        assert "line1" in result["error"]["traceback"]

    def test_ignores_messages_for_other_requests(self):
        client = _kernel_client(iopub_msgs=[
            _FakeMsg.iopub("OTHER-ID", "stream", name="stdout", text="not mine"),
            _FakeMsg.iopub("req-1", "stream", name="stdout", text="mine"),
        ])
        result = KernelHelper.execute_scratch(client, "print(1)")
        texts = [o.get("text") for o in result["outputs"] if o["kind"] == "stream"]
        assert "mine" in texts
        assert "not mine" not in texts

    def test_sets_execution_time_ms(self):
        client = _kernel_client()
        result = KernelHelper.execute_scratch(client, "x = 1")
        assert "execution_time_ms" in result
        assert result["execution_time_ms"] >= 0

    def test_no_outputs_for_empty_code(self):
        client = _kernel_client()
        result = KernelHelper.execute_scratch(client, "")
        assert result["status"] == "ok"
        assert result["outputs"] == []
        assert result["error"] is None
