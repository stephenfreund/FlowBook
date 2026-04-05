"""Tests for MessageBroadcaster, Message, and BroadcastStream."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from flowbook.server.message_broadcaster import (
    Message,
    MessageType,
    MessageBroadcaster,
    BroadcastStream,
)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_to_json_append(self):
        msg = Message(MessageType.APPEND, "hello")
        parsed = json.loads(msg.to_json())
        assert parsed["type"] == "append"
        assert parsed["content"] == "hello"
        assert "metadata" not in parsed

    def test_to_json_newline(self):
        msg = Message(MessageType.NEWLINE)
        parsed = json.loads(msg.to_json())
        assert parsed["type"] == "newline"
        assert parsed["content"] == ""

    def test_to_json_end(self):
        msg = Message(MessageType.END)
        assert json.loads(msg.to_json())["type"] == "end"

    def test_to_json_clear(self):
        msg = Message(MessageType.CLEAR)
        assert json.loads(msg.to_json())["type"] == "clear"

    def test_to_json_with_metadata(self):
        msg = Message(MessageType.APPEND, "text", metadata={"color": "red"})
        parsed = json.loads(msg.to_json())
        assert parsed["metadata"] == {"color": "red"}


# ---------------------------------------------------------------------------
# MessageBroadcaster (use fresh instances to avoid singleton issues)
# ---------------------------------------------------------------------------


def _fresh_broadcaster():
    """Create a fresh broadcaster bypassing the singleton."""
    b = object.__new__(MessageBroadcaster)
    b._queues = {}
    b._initialized = True
    return b


class TestMessageBroadcaster:
    def test_register_returns_queue(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        assert isinstance(q, asyncio.Queue)

    def test_register_same_id_returns_same_queue(self):
        b = _fresh_broadcaster()
        q1 = b.register_client("c1")
        q2 = b.register_client("c1")
        assert q1 is q2

    def test_unregister_removes_queue(self):
        b = _fresh_broadcaster()
        b.register_client("c1")
        b.unregister_client("c1")
        assert b.get_client_count() == 0

    def test_unregister_nonexistent_is_noop(self):
        b = _fresh_broadcaster()
        b.unregister_client("nope")  # Should not raise

    def test_send_message_to_specific_client(self):
        b = _fresh_broadcaster()
        q1 = b.register_client("c1")
        q2 = b.register_client("c2")
        msg = Message(MessageType.APPEND, "hi")
        b.send_message(msg, client_id="c1")
        assert q1.qsize() == 1
        assert q2.qsize() == 0

    def test_broadcast_to_all(self):
        b = _fresh_broadcaster()
        q1 = b.register_client("c1")
        q2 = b.register_client("c2")
        msg = Message(MessageType.END)
        b.send_message(msg)
        assert q1.qsize() == 1
        assert q2.qsize() == 1

    def test_send_to_nonexistent_client_is_noop(self):
        b = _fresh_broadcaster()
        msg = Message(MessageType.APPEND, "hi")
        b.send_message(msg, client_id="ghost")  # Should not raise

    def test_append_convenience(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        b.append("hello")
        got = q.get_nowait()
        assert got.type == MessageType.APPEND
        assert got.content == "hello"

    def test_newline_convenience(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        b.newline()
        assert q.get_nowait().type == MessageType.NEWLINE

    def test_end_convenience(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        b.end()
        assert q.get_nowait().type == MessageType.END

    def test_clear_convenience(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        b.clear()
        assert q.get_nowait().type == MessageType.CLEAR

    def test_client_count(self):
        b = _fresh_broadcaster()
        assert b.get_client_count() == 0
        b.register_client("a")
        b.register_client("b")
        assert b.get_client_count() == 2
        b.unregister_client("a")
        assert b.get_client_count() == 1


# ---------------------------------------------------------------------------
# BroadcastStream
# ---------------------------------------------------------------------------


class TestBroadcastStream:
    def test_write_empty_returns_zero(self):
        b = _fresh_broadcaster()
        stream = BroadcastStream(b)
        assert stream.write("") == 0

    def test_write_single_line(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        stream = BroadcastStream(b)
        n = stream.write("hello")
        assert n == 5
        msg = q.get_nowait()
        assert msg.type == MessageType.APPEND
        assert msg.content == "hello"

    def test_write_with_newline(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        stream = BroadcastStream(b)
        stream.write("line1\nline2")
        messages = []
        while not q.empty():
            messages.append(q.get_nowait())
        types = [m.type for m in messages]
        assert MessageType.NEWLINE in types
        contents = [m.content for m in messages if m.content]
        assert "line1" in contents
        assert "line2" in contents

    def test_write_trailing_newline(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        stream = BroadcastStream(b)
        stream.write("line1\n")
        messages = []
        while not q.empty():
            messages.append(q.get_nowait())
        # Should have APPEND "line1" then NEWLINE
        assert messages[0].content == "line1"
        assert messages[1].type == MessageType.NEWLINE

    def test_context_manager_flushes(self):
        b = _fresh_broadcaster()
        q = b.register_client("c1")
        stream = BroadcastStream(b)
        with stream:
            stream.write("pending")
        # After __exit__, should get a NEWLINE for pending content
        messages = []
        while not q.empty():
            messages.append(q.get_nowait())
        assert any(m.type == MessageType.NEWLINE for m in messages)

    def test_flush_is_noop(self):
        b = _fresh_broadcaster()
        stream = BroadcastStream(b)
        stream.flush()  # Should not raise

    def test_returns_length(self):
        b = _fresh_broadcaster()
        stream = BroadcastStream(b)
        assert stream.write("abc\ndef") == 7
