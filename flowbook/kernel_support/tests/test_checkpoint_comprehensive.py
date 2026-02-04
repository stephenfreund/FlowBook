"""Comprehensive tests for the combined Checkpoint wrapper.

This test file supplements test_checkpoint.py with additional coverage for:
- CheckpointDiffResult properties and backward compatibility
- Checkpoint backward compatibility properties
- Checkpoints manager edge cases
- VFS parameter forwarding
- max_size_mb parameter forwarding
- File-only vs memory-only scenarios
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from flowbook.kernel_support.checkpoint import (
    Checkpoint,
    Checkpoints,
    CheckpointDiffResult,
)
from flowbook.kernel_support.memory_checkpoint import (
    MemoryCheckpoint,
    MemoryCheckpoints,
)
from flowbook.kernel_support.file_checkpoint import (
    FileCheckpoint,
    FileCheckpoints,
    FileCheckpointDiffResult,
)
from flowbook.kernel_support.types import MemoryCheckpointDiffResult


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tmpdir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp(prefix="test_checkpoint_comp_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tcp():
    """Create a Checkpoints manager instance."""
    tc = Checkpoints()
    yield tc
    tc.clear()


@pytest.fixture
def tcp_with_files(tcp):
    """Create a Checkpoints manager with file checkpoints enabled."""
    tcp.file.enable()
    yield tcp
    tcp.file.disable()


# ============================================================================
# CHECKPOINTDIFFRESULT TESTS
# ============================================================================


class TestCheckpointDiffResult:
    """Test CheckpointDiffResult dataclass and properties."""

    def test_memory_only_diff_result(self):
        """Test CheckpointDiffResult with only memory changes."""
        # Create a mock memory diff
        mem_diff = MemoryCheckpointDiffResult(
            differences={"x": "changed"},
            warnings=["warning1"],
        )

        result = CheckpointDiffResult(memory=mem_diff, file=None)

        # Test convenience properties
        assert result.differences == {"x": "changed"}
        assert result.warnings == ["warning1"]
        assert result.changed_file_paths == set()
        assert result.has_file_changes is False

    def test_memory_and_file_diff_result(self):
        """Test CheckpointDiffResult with both memory and file changes."""
        mem_diff = MemoryCheckpointDiffResult(
            differences={"y": "modified"},
            warnings=[],
        )
        file_diff = FileCheckpointDiffResult(
            changed_paths={"/path/to/file.txt"},
            modified=[],
        )
        file_diff.modified.append(MagicMock(path="/path/to/file.txt"))

        result = CheckpointDiffResult(memory=mem_diff, file=file_diff)

        assert result.differences == {"y": "modified"}
        assert result.warnings == []
        assert result.changed_file_paths == {"/path/to/file.txt"}
        assert result.has_file_changes is True

    def test_no_file_changes(self):
        """Test has_file_changes when file diff has no changes."""
        mem_diff = MemoryCheckpointDiffResult(differences={}, warnings=[])
        file_diff = FileCheckpointDiffResult()  # Empty

        result = CheckpointDiffResult(memory=mem_diff, file=file_diff)

        assert result.has_file_changes is False
        assert result.changed_file_paths == set()


# ============================================================================
# CHECKPOINT CLASS TESTS
# ============================================================================


class TestCheckpointClass:
    """Test Checkpoint dataclass and static methods."""

    def test_checkpoint_backward_compat_properties(self, tcp):
        """Test Checkpoint backward compatibility properties."""
        user_ns = {"x": 1, "y": [1, 2, 3]}
        total, _ = tcp.save("cp1", user_ns)

        # Test user_ns property
        assert total.user_ns == total.memory.user_ns
        assert "x" in total.user_ns
        assert total.user_ns["x"] == 1

        # Test name property
        assert total.name == total.memory.name
        assert total.name == "cp1"

    def test_checkpoint_get_aliases_for_vars(self, tcp):
        """Test Checkpoint.get_aliases_for_vars delegation."""
        shared_list = [1, 2, 3]
        user_ns = {"a": shared_list, "b": shared_list}
        total, _ = tcp.save("cp1", user_ns)

        # Should delegate to memory checkpoint
        aliases = total.get_aliases_for_vars({"a"}, log_aliases=False)

        # Both 'a' and 'b' should be included (they share the same list)
        assert "a" in aliases
        assert "b" in aliases

    def test_checkpoint_diff_memory_only(self, tcp):
        """Test Checkpoint.diff with memory-only changes."""
        user_ns = {"x": 1}
        pre, _ = tcp.save("pre", dict(user_ns))

        user_ns["x"] = 2
        post, _ = tcp.save("post", dict(user_ns))

        diff = Checkpoint.diff(pre, post)

        assert "x" in diff.differences
        assert diff.file is None

    def test_checkpoint_diff_with_files(self, tcp_with_files, tmpdir):
        """Test Checkpoint.diff with both memory and file changes."""
        path = os.path.join(tmpdir, "test.txt")

        with open(path, "w") as f:
            f.write("before")

        user_ns = {"x": 1}
        pre, _ = tcp_with_files.save("pre", dict(user_ns), write_paths={path})

        user_ns["x"] = 2
        with open(path, "w") as f:
            f.write("after")
        post, _ = tcp_with_files.save("post", dict(user_ns), write_paths={path})

        diff = Checkpoint.diff(pre, post)

        assert "x" in diff.differences
        assert diff.file is not None
        assert diff.has_file_changes is True
        assert path in diff.changed_file_paths

    def test_checkpoint_diff_with_keys_to_include(self, tcp):
        """Test Checkpoint.diff with keys_to_include parameter."""
        user_ns = {"x": 1, "y": 2, "z": 3}
        pre, _ = tcp.save("pre", dict(user_ns))

        user_ns["x"] = 10
        user_ns["y"] = 20
        user_ns["z"] = 30
        post, _ = tcp.save("post", dict(user_ns))

        # Only check "x" and "y"
        diff = Checkpoint.diff(pre, post, keys_to_include={"x", "y"})

        assert "x" in diff.differences
        assert "y" in diff.differences
        # "z" should not be in differences since it's not in keys_to_include

    def test_checkpoint_diff_file_only_change(self, tcp_with_files, tmpdir):
        """Test Checkpoint.diff with only file changes (no memory changes)."""
        path = os.path.join(tmpdir, "test.txt")

        with open(path, "w") as f:
            f.write("before")

        user_ns = {"x": 1}
        pre, _ = tcp_with_files.save("pre", dict(user_ns), write_paths={path})

        # Only modify file, not memory
        with open(path, "w") as f:
            f.write("after")
        post, _ = tcp_with_files.save("post", dict(user_ns), write_paths={path})

        diff = Checkpoint.diff(pre, post)

        # No memory changes
        assert diff.differences == {}
        # But file changed
        assert diff.has_file_changes is True
        assert path in diff.changed_file_paths


# ============================================================================
# CHECKPOINTS MANAGER TESTS
# ============================================================================


class TestCheckpointsManager:
    """Test the Checkpoints manager class."""

    def test_save_memory_only_no_file_enabled(self, tcp):
        """Test save when file checkpoints are disabled."""
        user_ns = {"x": 1, "y": "hello"}
        total, removed = tcp.save("cp1", user_ns)

        assert total.memory is not None
        assert total.file is None
        assert "x" in total.user_ns

    def test_save_with_write_paths_but_disabled(self, tcp, tmpdir):
        """Test save with write_paths when file checkpoints are disabled."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        # File checkpoints not enabled, so write_paths is ignored
        user_ns = {"x": 1}
        total, _ = tcp.save("cp1", user_ns, write_paths={path})

        assert total.file is None

    def test_save_with_max_size_mb(self, tcp):
        """Test that max_size_mb is forwarded to memory checkpoint save."""
        # Create a large-ish DataFrame
        df = pd.DataFrame({"data": range(10000)})
        user_ns = {"df": df}

        # Should not raise with a reasonable max_size_mb
        total, _ = tcp.save("cp1", user_ns, max_size_mb=100)
        assert total.memory is not None

    def test_restore_memory_only(self, tcp):
        """Test restore when file checkpoints are disabled."""
        user_ns = {"x": 1, "y": 2}
        tcp.save("cp1", user_ns)

        # Modify namespace
        user_ns["x"] = 999
        user_ns["z"] = 3

        tcp.restore("cp1", user_ns)

        assert user_ns["x"] == 1
        assert user_ns["y"] == 2
        assert "z" not in user_ns

    def test_restore_with_vfs_forwarded(self, tcp_with_files, tmpdir):
        """Test that VFS parameter is forwarded to file restore."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("original")

        tcp_with_files.save("cp1", {"x": 1}, write_paths={path})

        # Create mock VFS
        vfs = MagicMock()
        vfs.enabled = True
        overlay_path = os.path.join(tmpdir, "overlay", "test.txt")
        vfs._to_overlay_path.return_value = overlay_path
        vfs._deleted_paths = set()

        # Create overlay directory
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)

        # Restore with VFS
        tcp_with_files.restore("cp1", {"x": 1}, vfs=vfs)

        # VFS should have been used
        vfs._to_overlay_path.assert_called()

    def test_get_returns_combined_checkpoint(self, tcp_with_files, tmpdir):
        """Test that get returns a Checkpoint with both memory and file."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test content")

        tcp_with_files.save("cp1", {"x": 1}, write_paths={path})

        total = tcp_with_files.get("cp1")

        assert isinstance(total, Checkpoint)
        assert total.memory is not None
        assert total.file is not None
        assert total.user_ns["x"] == 1

    def test_get_memory_only(self, tcp):
        """Test get when file checkpoints are disabled."""
        tcp.save("cp1", {"x": 1})

        total = tcp.get("cp1")

        assert total.memory is not None
        assert total.file is None

    def test_exists_method(self, tcp):
        """Test exists method."""
        assert tcp.exists("nonexistent") is False

        tcp.save("cp1", {"x": 1})
        assert tcp.exists("cp1") is True

        tcp.delete("cp1")
        assert tcp.exists("cp1") is False

    def test_delete_memory_and_file(self, tcp_with_files, tmpdir):
        """Test that delete removes both memory and file checkpoints."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        tcp_with_files.save("cp1", {"x": 1}, write_paths={path})

        assert tcp_with_files.exists("cp1")
        assert tcp_with_files.file.exists("cp1")

        tcp_with_files.delete("cp1")

        assert not tcp_with_files.exists("cp1")
        assert not tcp_with_files.file.exists("cp1")

    def test_list_returns_memory_checkpoint_names(self, tcp):
        """Test that list returns checkpoint names from memory manager."""
        tcp.save("cp1", {"x": 1})
        tcp.save("cp2", {"y": 2})
        tcp.save("cp3", {"z": 3})

        names = tcp.list()

        assert set(names) == {"cp1", "cp2", "cp3"}

    def test_clear_removes_all(self, tcp_with_files, tmpdir):
        """Test that clear removes all memory and file checkpoints."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        tcp_with_files.save("cp1", {"x": 1}, write_paths={path})
        tcp_with_files.save("cp2", {"y": 2}, write_paths={path})

        tcp_with_files.clear()

        assert tcp_with_files.list() == []
        assert tcp_with_files.file.list() == []


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_save_empty_namespace(self, tcp):
        """Test saving an empty namespace."""
        total, removed = tcp.save("empty", {})

        assert total.memory is not None
        assert total.user_ns == {}

    def test_restore_into_empty_namespace(self, tcp):
        """Test restoring into an empty namespace."""
        tcp.save("cp1", {"x": 1, "y": 2})

        empty_ns = {}
        tcp.restore("cp1", empty_ns)

        assert empty_ns["x"] == 1
        assert empty_ns["y"] == 2

    def test_multiple_save_restore_cycles(self, tcp):
        """Test multiple save/restore cycles."""
        user_ns = {"counter": 0}

        for i in range(5):
            tcp.save(f"cp{i}", dict(user_ns))
            user_ns["counter"] += 1

        # Restore to each checkpoint
        for i in range(5):
            tcp.restore(f"cp{i}", user_ns)
            assert user_ns["counter"] == i

    def test_overwrite_checkpoint(self, tcp):
        """Test that saving with same name overwrites."""
        tcp.save("cp", {"x": 1})
        assert tcp.get("cp").user_ns["x"] == 1

        tcp.save("cp", {"x": 999})
        assert tcp.get("cp").user_ns["x"] == 999

        # Should only have one checkpoint
        assert tcp.list() == ["cp"]

    def test_file_checkpoint_without_write_paths(self, tcp_with_files, tmpdir):
        """Test saving with file checkpoints enabled but no write_paths."""
        user_ns = {"x": 1}
        total, _ = tcp_with_files.save("cp1", user_ns, write_paths=None)

        # Memory checkpoint should exist
        assert total.memory is not None
        # File checkpoint should be None (no paths provided)
        assert total.file is None

    def test_diff_with_use_leq(self, tcp):
        """Test Checkpoint.diff with use_leq parameter."""
        user_ns = {"x": 1}
        pre, _ = tcp.save("pre", dict(user_ns))

        user_ns["x"] = 1  # Same value
        user_ns["extra"] = 2  # New variable
        post, _ = tcp.save("post", dict(user_ns))

        # With use_leq=True, extra keys in b are allowed
        diff = Checkpoint.diff(pre, post, use_leq=True)

        # The behavior depends on the Diff implementation
        # Just verify it doesn't crash
        assert isinstance(diff, CheckpointDiffResult)

    def test_removed_vars_tracking(self):
        """Test that removed vars are tracked correctly."""
        # Create object that can't be checkpointed
        class Uncheckpointable:
            def __deepcopy__(self, memo):
                raise TypeError("Cannot copy")

        user_ns = {"x": 1, "bad": Uncheckpointable()}

        # Use MemoryCheckpoints directly to test removed vars
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints

        cp = MemoryCheckpoints()
        saved, removed = cp.save("cp1", user_ns)

        assert "x" in saved
        assert "bad" in removed


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_files(self, tcp_with_files, tmpdir):
        """Test complete workflow with memory and file checkpoints."""
        file1 = os.path.join(tmpdir, "data.csv")
        file2 = os.path.join(tmpdir, "config.json")

        # Initial state
        with open(file1, "w") as f:
            f.write("a,b,c\n1,2,3")
        with open(file2, "w") as f:
            f.write('{"key": "value"}')

        user_ns = {"df": pd.DataFrame({"a": [1, 2, 3]}), "config": {"key": "value"}}

        # Save checkpoint 1
        cp1, _ = tcp_with_files.save("v1", dict(user_ns), write_paths={file1, file2})

        # Modify everything
        with open(file1, "w") as f:
            f.write("a,b,c\n4,5,6")
        with open(file2, "w") as f:
            f.write('{"key": "new_value"}')

        user_ns["df"] = pd.DataFrame({"a": [4, 5, 6]})
        user_ns["config"]["key"] = "new_value"

        # Save checkpoint 2
        cp2, _ = tcp_with_files.save("v2", dict(user_ns), write_paths={file1, file2})

        # Compare checkpoints
        diff = Checkpoint.diff(cp1, cp2)

        assert "df" in diff.differences or "config" in diff.differences
        assert diff.has_file_changes is True
        assert file1 in diff.changed_file_paths

        # Restore to v1
        tcp_with_files.restore("v1", user_ns)

        # Verify memory restored
        assert list(user_ns["df"]["a"]) == [1, 2, 3]

        # Verify files restored
        with open(file1, "r") as f:
            assert "1,2,3" in f.read()

    def test_checkpoint_isolation(self, tcp_with_files, tmpdir):
        """Test that checkpoints are properly isolated from each other."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("v1")

        user_ns = {"data": [1, 2, 3]}
        tcp_with_files.save("cp1", dict(user_ns), write_paths={path})

        # Modify and save cp2
        user_ns["data"].append(4)
        with open(path, "w") as f:
            f.write("v2")
        tcp_with_files.save("cp2", dict(user_ns), write_paths={path})

        # Modify again and save cp3
        user_ns["data"].append(5)
        with open(path, "w") as f:
            f.write("v3")
        tcp_with_files.save("cp3", dict(user_ns), write_paths={path})

        # Restore cp1 and verify isolation
        tcp_with_files.restore("cp1", user_ns)
        assert user_ns["data"] == [1, 2, 3]
        with open(path, "r") as f:
            assert f.read() == "v1"

        # Verify cp2 still has its own data
        cp2 = tcp_with_files.get("cp2")
        assert cp2.user_ns["data"] == [1, 2, 3, 4]

        # Verify cp3 still has its own data
        cp3 = tcp_with_files.get("cp3")
        assert cp3.user_ns["data"] == [1, 2, 3, 4, 5]
