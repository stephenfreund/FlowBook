"""Tests for FileCheckpoints."""

import os
import shutil
import tempfile

import pytest

from flowbook.kernel_support.file_checkpoint import (
    FileCheckpoint,
    FileCheckpoints,
    FileDiffResult,
    FileSnapshot,
)


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp(prefix="test_fcp_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fcp():
    fc = FileCheckpoints()
    fc.enable()
    yield fc
    fc.disable()


class TestFileCheckpointsBasics:
    def test_enable_disable(self):
        fc = FileCheckpoints()
        assert not fc._enabled
        fc.enable()
        assert fc._enabled
        fc.disable()
        assert not fc._enabled

    def test_save_and_exists(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "data.txt")
        with open(path, "w") as f:
            f.write("hello")

        cp = fcp.save("cp1", {path})
        assert fcp.exists("cp1")
        assert path in cp.files
        assert cp.files[path].size > 0
        assert len(cp.files[path].file_hash) == 64  # sha256

    def test_save_missing_file(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "nonexistent.txt")
        cp = fcp.save("cp1", {path})
        assert path not in cp.files
        assert path in cp.deleted_files

    def test_restore(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "restore.txt")
        with open(path, "w") as f:
            f.write("original")

        fcp.save("cp1", {path})

        # Modify
        with open(path, "w") as f:
            f.write("modified")

        # Restore
        fcp.restore("cp1")

        with open(path, "r") as f:
            assert f.read() == "original"

    def test_list_and_delete(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "list.txt")
        with open(path, "w") as f:
            f.write("data")

        fcp.save("a", {path})
        fcp.save("b", {path})
        assert sorted(fcp.list()) == ["a", "b"]

        fcp.delete("a")
        assert fcp.list() == ["b"]

    def test_clear(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "clear.txt")
        with open(path, "w") as f:
            f.write("data")

        fcp.save("cp1", {path})
        fcp.save("cp2", {path})
        fcp.clear()
        assert fcp.list() == []


class TestFileCheckpointDiff:
    def test_no_changes(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "same.txt")
        with open(path, "w") as f:
            f.write("unchanged")

        cp1 = fcp.save("cp1", {path})
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)
        assert not diff.has_changes
        assert len(diff.changed_paths) == 0

    def test_modified_file(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "modify.txt")
        with open(path, "w") as f:
            f.write("before")
        cp1 = fcp.save("cp1", {path})

        with open(path, "w") as f:
            f.write("after")
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)
        assert diff.has_changes
        assert len(diff.modified) == 1
        assert diff.modified[0].path == path
        assert diff.modified[0].status == "modified"
        assert diff.modified[0].content_diff is not None
        assert path in diff.changed_paths

    def test_added_file(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "added.txt")

        # cp1: file doesn't exist
        cp1 = fcp.save("cp1", set())

        with open(path, "w") as f:
            f.write("new file")
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)
        assert diff.has_changes
        assert len(diff.added) == 1
        assert diff.added[0].path == path

    def test_removed_file(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "removed.txt")
        with open(path, "w") as f:
            f.write("will be removed")
        cp1 = fcp.save("cp1", {path})

        os.remove(path)
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)
        assert diff.has_changes
        assert len(diff.removed) == 1
        assert diff.removed[0].path == path

    def test_binary_file(self, fcp, tmpdir):
        path = os.path.join(tmpdir, "binary.bin")
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)
        cp1 = fcp.save("cp1", {path})

        with open(path, "wb") as f:
            f.write(b"\x04\x05\x06\x07" * 100)
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)
        assert diff.has_changes
        assert diff.modified[0].is_binary
        assert diff.modified[0].content_diff is None

    def test_multiple_files(self, fcp, tmpdir):
        path_a = os.path.join(tmpdir, "a.txt")
        path_b = os.path.join(tmpdir, "b.txt")
        path_c = os.path.join(tmpdir, "c.txt")

        with open(path_a, "w") as f:
            f.write("a1")
        with open(path_b, "w") as f:
            f.write("b1")
        cp1 = fcp.save("cp1", {path_a, path_b})

        # Modify a, add c, leave b unchanged
        with open(path_a, "w") as f:
            f.write("a2")
        with open(path_c, "w") as f:
            f.write("c1")
        cp2 = fcp.save("cp2", {path_a, path_b, path_c})

        diff = FileCheckpoints.diff(cp1, cp2)
        assert diff.has_changes
        assert len(diff.modified) == 1  # a changed
        assert len(diff.added) == 1    # c added
        assert len(diff.removed) == 0  # nothing removed
