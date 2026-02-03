"""Tests for TotalCheckpoint and TotalCheckpoints."""

import os
import shutil
import tempfile

import pytest

from flowbook.kernel_support.total_checkpoint import (
    TotalCheckpoint,
    TotalCheckpoints,
    TotalDiffResult,
)


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp(prefix="test_tcp_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tcp():
    tc = TotalCheckpoints()
    yield tc
    tc.clear()


class TestTotalCheckpoints:
    def test_save_memory_only(self, tcp):
        user_ns = {"x": 1, "y": "hello"}
        total, removed = tcp.save("cp1", user_ns)
        assert total.memory is not None
        assert total.file is None
        assert tcp.exists("cp1")

    def test_save_with_file_checkpoints(self, tcp, tmpdir):
        tcp.file.enable()

        path = os.path.join(tmpdir, "data.txt")
        with open(path, "w") as f:
            f.write("checkpoint data")

        user_ns = {"x": 42}
        total, removed = tcp.save("cp1", user_ns, write_paths={path})
        assert total.memory is not None
        assert total.file is not None
        assert path in total.file.files

    def test_restore_memory(self, tcp):
        user_ns = {"x": 1}
        tcp.save("cp1", user_ns)

        user_ns["x"] = 999
        user_ns["y"] = "new"
        tcp.restore("cp1", user_ns)
        assert user_ns["x"] == 1
        assert "y" not in user_ns

    def test_restore_with_files(self, tcp, tmpdir):
        tcp.file.enable()

        path = os.path.join(tmpdir, "restore.txt")
        with open(path, "w") as f:
            f.write("original")

        user_ns = {"x": 1}
        tcp.save("cp1", user_ns, write_paths={path})

        # Modify both
        user_ns["x"] = 999
        with open(path, "w") as f:
            f.write("modified")

        tcp.restore("cp1", user_ns)
        assert user_ns["x"] == 1
        with open(path, "r") as f:
            assert f.read() == "original"

    def test_list_delete_clear(self, tcp):
        tcp.save("a", {"x": 1})
        tcp.save("b", {"y": 2})
        assert sorted(tcp.list()) == ["a", "b"]

        tcp.delete("a")
        assert tcp.list() == ["b"]

        tcp.clear()
        assert tcp.list() == []

    def test_get(self, tcp):
        tcp.save("cp1", {"x": 1})
        total = tcp.get("cp1")
        assert total.memory is not None
        assert "x" in total.user_ns


class TestTotalCheckpointDiff:
    def test_memory_only_diff(self, tcp):
        user_ns = {"x": 1}
        pre, _ = tcp.save("pre", dict(user_ns))

        user_ns["x"] = 2
        post, _ = tcp.save("post", dict(user_ns))

        diff = TotalCheckpoint.diff(pre, post)
        assert "x" in diff.differences
        assert not diff.has_file_changes

    def test_memory_plus_file_diff(self, tcp, tmpdir):
        tcp.file.enable()

        path = os.path.join(tmpdir, "diff.txt")
        with open(path, "w") as f:
            f.write("before")

        user_ns = {"x": 1}
        pre, _ = tcp.save("pre", dict(user_ns), write_paths={path})

        user_ns["x"] = 2
        with open(path, "w") as f:
            f.write("after")
        post, _ = tcp.save("post", dict(user_ns), write_paths={path})

        diff = TotalCheckpoint.diff(pre, post)
        assert "x" in diff.differences
        assert diff.has_file_changes
        assert path in diff.changed_file_paths

    def test_backward_compat_properties(self, tcp):
        user_ns = {"x": 1}
        pre, _ = tcp.save("pre", dict(user_ns))
        user_ns["x"] = 2
        post, _ = tcp.save("post", dict(user_ns))

        diff = TotalCheckpoint.diff(pre, post)
        # These delegate to memory
        assert isinstance(diff.differences, dict)
        assert isinstance(diff.warnings, list)
        assert isinstance(diff.changed_file_paths, set)
