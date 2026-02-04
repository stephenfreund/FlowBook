"""Comprehensive tests for FileCheckpoint covering all behaviors and edge cases.

This test file supplements test_file_checkpoint.py with additional coverage for:
- VFS (Virtual File System) integration
- Directory handling
- Error recovery and edge cases
- Idempotent operations
- Deleted files tracking
- Restore when saved copies are missing
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from flowbook.kernel_support.file_checkpoint import (
    FileCheckpoint,
    FileCheckpoints,
    FileCheckpointDiffResult,
    FileDiffEntry,
    FileSnapshot,
    _hash_file,
    _is_binary,
    _unified_diff,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tmpdir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp(prefix="test_fcp_comprehensive_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fcp():
    """Create an enabled FileCheckpoints instance."""
    fc = FileCheckpoints()
    fc.enable()
    yield fc
    fc.disable()


@pytest.fixture
def disabled_fcp():
    """Create a disabled FileCheckpoints instance."""
    fc = FileCheckpoints()
    yield fc
    if fc._enabled:
        fc.disable()


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================


class TestHashFile:
    """Test the _hash_file helper function."""

    def test_hash_empty_file(self, tmpdir):
        """Test hashing an empty file."""
        path = os.path.join(tmpdir, "empty.txt")
        with open(path, "w") as f:
            pass  # Create empty file

        file_hash = _hash_file(path)
        assert len(file_hash) == 64  # SHA-256 produces 64 hex characters
        # Empty file has a known hash
        assert file_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_hash_same_content_same_hash(self, tmpdir):
        """Test that identical content produces identical hash."""
        path1 = os.path.join(tmpdir, "file1.txt")
        path2 = os.path.join(tmpdir, "file2.txt")

        content = "Hello, World!" * 100
        for path in [path1, path2]:
            with open(path, "w") as f:
                f.write(content)

        assert _hash_file(path1) == _hash_file(path2)

    def test_hash_different_content_different_hash(self, tmpdir):
        """Test that different content produces different hash."""
        path1 = os.path.join(tmpdir, "file1.txt")
        path2 = os.path.join(tmpdir, "file2.txt")

        with open(path1, "w") as f:
            f.write("content1")
        with open(path2, "w") as f:
            f.write("content2")

        assert _hash_file(path1) != _hash_file(path2)

    def test_hash_large_file(self, tmpdir):
        """Test hashing a large file (tests chunked reading)."""
        path = os.path.join(tmpdir, "large.bin")
        # Create file larger than chunk size (8192 bytes)
        with open(path, "wb") as f:
            f.write(b"x" * 100000)

        file_hash = _hash_file(path)
        assert len(file_hash) == 64


class TestIsBinary:
    """Test the _is_binary helper function."""

    def test_text_file_not_binary(self, tmpdir):
        """Test that text files are not detected as binary."""
        path = os.path.join(tmpdir, "text.txt")
        with open(path, "w") as f:
            f.write("Hello, World!\nLine 2\nLine 3\n")

        assert _is_binary(path) is False

    def test_binary_file_detected(self, tmpdir):
        """Test that binary files (with null bytes) are detected."""
        path = os.path.join(tmpdir, "binary.bin")
        with open(path, "wb") as f:
            f.write(b"Some text\x00with null bytes")

        assert _is_binary(path) is True

    def test_empty_file_not_binary(self, tmpdir):
        """Test that empty files are not detected as binary."""
        path = os.path.join(tmpdir, "empty.txt")
        with open(path, "w") as f:
            pass

        assert _is_binary(path) is False

    def test_nonexistent_file_returns_true(self, tmpdir):
        """Test that nonexistent files return True (treated as binary)."""
        path = os.path.join(tmpdir, "nonexistent.txt")
        assert _is_binary(path) is True

    def test_unicode_text_not_binary(self, tmpdir):
        """Test that Unicode text files are not detected as binary."""
        path = os.path.join(tmpdir, "unicode.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Hello 世界 🌍 مرحبا")

        assert _is_binary(path) is False


class TestUnifiedDiff:
    """Test the _unified_diff helper function."""

    def test_identical_files_no_diff(self, tmpdir):
        """Test that identical files produce no diff."""
        path_a = os.path.join(tmpdir, "a.txt")
        path_b = os.path.join(tmpdir, "b.txt")

        content = "Line 1\nLine 2\nLine 3\n"
        for path in [path_a, path_b]:
            with open(path, "w") as f:
                f.write(content)

        result = _unified_diff(path_a, path_b)
        assert result is None  # No differences

    def test_different_files_produce_diff(self, tmpdir):
        """Test that different files produce a diff."""
        path_a = os.path.join(tmpdir, "a.txt")
        path_b = os.path.join(tmpdir, "b.txt")

        with open(path_a, "w") as f:
            f.write("Line 1\nLine 2\n")
        with open(path_b, "w") as f:
            f.write("Line 1\nModified Line 2\n")

        result = _unified_diff(path_a, path_b, "before", "after")
        assert result is not None
        assert "-Line 2" in result
        assert "+Modified Line 2" in result

    def test_binary_files_return_none(self, tmpdir):
        """Test that binary files return None for diff."""
        path_a = os.path.join(tmpdir, "a.bin")
        path_b = os.path.join(tmpdir, "b.bin")

        for path, content in [(path_a, b"\x00\x01"), (path_b, b"\x00\x02")]:
            with open(path, "wb") as f:
                f.write(content)

        result = _unified_diff(path_a, path_b)
        assert result is None

    def test_one_binary_file_returns_none(self, tmpdir):
        """Test that if either file is binary, return None."""
        path_a = os.path.join(tmpdir, "text.txt")
        path_b = os.path.join(tmpdir, "binary.bin")

        with open(path_a, "w") as f:
            f.write("text content")
        with open(path_b, "wb") as f:
            f.write(b"\x00binary content")

        assert _unified_diff(path_a, path_b) is None
        assert _unified_diff(path_b, path_a) is None


# ============================================================================
# ENABLE/DISABLE TESTS
# ============================================================================


class TestEnableDisable:
    """Test enable/disable operations."""

    def test_double_enable_is_idempotent(self, disabled_fcp):
        """Test that calling enable() twice is safe."""
        disabled_fcp.enable()
        first_storage = disabled_fcp._storage_dir

        disabled_fcp.enable()  # Second call should be no-op
        second_storage = disabled_fcp._storage_dir

        assert disabled_fcp._enabled is True
        assert first_storage == second_storage

    def test_double_disable_is_idempotent(self, fcp):
        """Test that calling disable() twice is safe."""
        fcp.disable()
        assert fcp._enabled is False

        fcp.disable()  # Second call should be no-op
        assert fcp._enabled is False

    def test_enable_creates_storage_directory(self, disabled_fcp):
        """Test that enable() creates a storage directory."""
        assert disabled_fcp._storage_dir is None

        disabled_fcp.enable()

        assert disabled_fcp._storage_dir is not None
        assert os.path.exists(disabled_fcp._storage_dir)

    def test_disable_cleans_up_storage(self, fcp):
        """Test that disable() removes the storage directory."""
        storage_dir = fcp._storage_dir
        assert os.path.exists(storage_dir)

        fcp.disable()

        assert not os.path.exists(storage_dir)
        assert fcp._storage_dir is None

    def test_disable_clears_saved_checkpoints(self, fcp, tmpdir):
        """Test that disable() clears all saved checkpoints."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        fcp.save("cp1", {path})
        fcp.save("cp2", {path})
        assert len(fcp.saved) == 2

        fcp.disable()

        assert len(fcp.saved) == 0


# ============================================================================
# SAVE OPERATION TESTS
# ============================================================================


class TestSaveOperation:
    """Test save operation edge cases."""

    def test_save_empty_write_paths(self, fcp):
        """Test saving with empty write_paths set."""
        cp = fcp.save("empty", set())

        assert cp.name == "empty"
        assert len(cp.files) == 0
        assert len(cp.deleted_files) == 0

    def test_save_with_directory_path_skipped(self, fcp, tmpdir):
        """Test that directory paths are skipped during save."""
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)

        cp = fcp.save("cp1", {subdir})

        # Directory should be skipped (not in files, not in deleted)
        assert subdir not in cp.files
        assert subdir not in cp.deleted_files

    def test_save_overwrite_existing_checkpoint(self, fcp, tmpdir):
        """Test that saving with same name overwrites the checkpoint."""
        path = os.path.join(tmpdir, "test.txt")

        with open(path, "w") as f:
            f.write("version 1")
        cp1 = fcp.save("cp", {path})
        hash1 = cp1.files[path].file_hash

        with open(path, "w") as f:
            f.write("version 2")
        cp2 = fcp.save("cp", {path})
        hash2 = cp2.files[path].file_hash

        # Hashes should be different
        assert hash1 != hash2
        # Only one checkpoint should exist
        assert fcp.list() == ["cp"]

    def test_save_with_special_characters_in_path(self, fcp, tmpdir):
        """Test saving files with special characters in path."""
        # Create file with spaces and special chars
        path = os.path.join(tmpdir, "file with spaces.txt")
        with open(path, "w") as f:
            f.write("test content")

        cp = fcp.save("cp1", {path})

        assert path in cp.files
        assert cp.files[path].size > 0

    def test_save_handles_copy_error_gracefully(self, fcp, tmpdir):
        """Test that save handles copy errors gracefully."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        # Mock shutil.copy2 to raise an error
        with patch("shutil.copy2") as mock_copy:
            mock_copy.side_effect = OSError("Permission denied")
            cp = fcp.save("cp1", {path})

        # File should not be in checkpoint (copy failed)
        assert path not in cp.files


# ============================================================================
# RESTORE OPERATION TESTS
# ============================================================================


class TestRestoreOperation:
    """Test restore operation edge cases."""

    def test_restore_creates_parent_directory(self, fcp, tmpdir):
        """Test that restore creates parent directory if needed."""
        subdir = os.path.join(tmpdir, "newdir")
        path = os.path.join(subdir, "file.txt")

        # Create file in subdir first
        os.makedirs(subdir)
        with open(path, "w") as f:
            f.write("original")

        fcp.save("cp1", {path})

        # Remove the entire subdir
        shutil.rmtree(subdir)
        assert not os.path.exists(subdir)

        # Restore should recreate the directory
        fcp.restore("cp1")

        assert os.path.exists(path)
        with open(path, "r") as f:
            assert f.read() == "original"

    def test_restore_deleted_file_removes_it(self, fcp, tmpdir):
        """Test that restoring with deleted_files removes the file."""
        path = os.path.join(tmpdir, "to_delete.txt")

        # Create checkpoint when file doesn't exist
        cp = fcp.save("cp1", {path})
        assert path in cp.deleted_files

        # Create the file
        with open(path, "w") as f:
            f.write("new content")
        assert os.path.exists(path)

        # Restore should delete the file
        fcp.restore("cp1")

        assert not os.path.exists(path)

    def test_restore_skips_missing_saved_copy(self, fcp, tmpdir):
        """Test restore handles missing saved_copy gracefully."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("original")

        fcp.save("cp1", {path})

        # Manually delete the saved copy
        cp = fcp.get("cp1")
        os.remove(cp.files[path].saved_copy)

        # Modify original file
        with open(path, "w") as f:
            f.write("modified")

        # Restore should not crash, but file won't be restored
        fcp.restore("cp1")

        # File still has modified content (restore was skipped)
        with open(path, "r") as f:
            assert f.read() == "modified"


# ============================================================================
# VFS INTEGRATION TESTS
# ============================================================================


class TestVFSIntegration:
    """Test VFS (Virtual File System) integration."""

    def test_save_with_vfs_resolves_paths(self, fcp, tmpdir):
        """Test that save with VFS resolves overlay paths."""
        real_path = os.path.join(tmpdir, "real.txt")
        overlay_path = os.path.join(tmpdir, "overlay.txt")

        with open(real_path, "w") as f:
            f.write("real content")

        # Create mock VFS
        vfs = MagicMock()
        vfs.enabled = True
        vfs._resolve_read_path.return_value = real_path

        # Save should use VFS to resolve path
        cp = fcp.save("cp1", {real_path}, vfs=vfs)

        # VFS's resolve method should have been called
        vfs._resolve_read_path.assert_called()

    def test_restore_with_vfs_writes_to_overlay(self, fcp, tmpdir):
        """Test that restore with VFS writes to overlay."""
        real_path = os.path.join(tmpdir, "real.txt")
        overlay_path = os.path.join(tmpdir, "overlay", "real.txt")

        with open(real_path, "w") as f:
            f.write("original")

        # Save without VFS
        fcp.save("cp1", {real_path})

        # Create mock VFS
        vfs = MagicMock()
        vfs.enabled = True
        vfs._to_overlay_path.return_value = overlay_path
        vfs._deleted_paths = set()

        # Create overlay directory
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)

        # Restore with VFS
        fcp.restore("cp1", vfs=vfs)

        # VFS methods should have been called
        vfs._to_overlay_path.assert_called_with(real_path)

    def test_restore_deleted_with_vfs_adds_to_deleted_paths(self, fcp, tmpdir):
        """Test that restoring deleted files with VFS adds to _deleted_paths."""
        path = os.path.join(tmpdir, "deleted.txt")

        # Save when file doesn't exist
        cp = fcp.save("cp1", {path})
        assert path in cp.deleted_files

        # Create mock VFS
        vfs = MagicMock()
        vfs.enabled = True
        vfs._deleted_paths = set()

        # Restore with VFS
        fcp.restore("cp1", vfs=vfs)

        # Deleted path should be added to VFS deleted_paths
        assert path in vfs._deleted_paths


# ============================================================================
# DIFF OPERATION TESTS
# ============================================================================


class TestDiffOperation:
    """Test diff operation edge cases."""

    def test_diff_with_empty_checkpoints(self, fcp):
        """Test diff between two empty checkpoints."""
        cp1 = fcp.save("cp1", set())
        cp2 = fcp.save("cp2", set())

        diff = FileCheckpoints.diff(cp1, cp2)

        assert not diff.has_changes
        assert len(diff.added) == 0
        assert len(diff.removed) == 0
        assert len(diff.modified) == 0

    def test_diff_deleted_file_to_existing(self, fcp, tmpdir):
        """Test diff when file goes from deleted to existing."""
        path = os.path.join(tmpdir, "file.txt")

        # cp1: file doesn't exist
        cp1 = fcp.save("cp1", {path})
        assert path in cp1.deleted_files

        # cp2: file exists
        with open(path, "w") as f:
            f.write("new content")
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)

        assert diff.has_changes
        assert len(diff.added) == 1
        assert diff.added[0].path == path

    def test_diff_existing_to_deleted(self, fcp, tmpdir):
        """Test diff when file goes from existing to deleted."""
        path = os.path.join(tmpdir, "file.txt")

        # cp1: file exists
        with open(path, "w") as f:
            f.write("original")
        cp1 = fcp.save("cp1", {path})

        # cp2: file deleted
        os.remove(path)
        cp2 = fcp.save("cp2", {path})
        assert path in cp2.deleted_files

        diff = FileCheckpoints.diff(cp1, cp2)

        assert diff.has_changes
        assert len(diff.removed) == 1
        assert diff.removed[0].path == path

    def test_diff_both_deleted_no_change(self, fcp, tmpdir):
        """Test diff when file is deleted in both checkpoints."""
        path = os.path.join(tmpdir, "deleted.txt")

        # Both checkpoints: file doesn't exist
        cp1 = fcp.save("cp1", {path})
        cp2 = fcp.save("cp2", {path})

        assert path in cp1.deleted_files
        assert path in cp2.deleted_files

        diff = FileCheckpoints.diff(cp1, cp2)

        # Should not be reported as a change
        assert path not in diff.changed_paths

    def test_diff_result_properties(self, fcp, tmpdir):
        """Test FileCheckpointDiffResult properties."""
        path = os.path.join(tmpdir, "test.txt")

        with open(path, "w") as f:
            f.write("v1")
        cp1 = fcp.save("cp1", {path})

        with open(path, "w") as f:
            f.write("v2")
        cp2 = fcp.save("cp2", {path})

        diff = FileCheckpoints.diff(cp1, cp2)

        assert diff.has_changes is True
        assert isinstance(diff.changed_paths, set)
        assert path in diff.changed_paths


# ============================================================================
# GET AND EXISTS TESTS
# ============================================================================


class TestGetAndExists:
    """Test get and exists operations."""

    def test_get_returns_correct_checkpoint(self, fcp, tmpdir):
        """Test that get returns the correct checkpoint."""
        path = os.path.join(tmpdir, "test.txt")

        with open(path, "w") as f:
            f.write("v1")
        fcp.save("cp1", {path})

        with open(path, "w") as f:
            f.write("v2")
        fcp.save("cp2", {path})

        cp1 = fcp.get("cp1")
        cp2 = fcp.get("cp2")

        assert cp1.name == "cp1"
        assert cp2.name == "cp2"
        assert cp1.files[path].file_hash != cp2.files[path].file_hash

    def test_get_raises_keyerror_for_nonexistent(self, fcp):
        """Test that get raises KeyError for nonexistent checkpoint."""
        with pytest.raises(KeyError):
            fcp.get("nonexistent")

    def test_exists_returns_true_for_saved(self, fcp, tmpdir):
        """Test exists returns True for saved checkpoints."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        fcp.save("cp1", {path})

        assert fcp.exists("cp1") is True

    def test_exists_returns_false_for_nonexistent(self, fcp):
        """Test exists returns False for nonexistent checkpoints."""
        assert fcp.exists("nonexistent") is False


# ============================================================================
# DELETE AND CLEAR TESTS
# ============================================================================


class TestDeleteAndClear:
    """Test delete and clear operations."""

    def test_delete_removes_storage_directory(self, fcp, tmpdir):
        """Test that delete removes the checkpoint's storage directory."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        fcp.save("cp1", {path})
        cp = fcp.get("cp1")
        cp_dir = os.path.join(fcp._storage_dir, "cp1")
        assert os.path.exists(cp_dir)

        fcp.delete("cp1")

        assert not os.path.exists(cp_dir)
        assert not fcp.exists("cp1")

    def test_delete_nonexistent_is_noop(self, fcp):
        """Test that deleting nonexistent checkpoint doesn't raise."""
        # Should not raise
        fcp.delete("nonexistent")
        assert True

    def test_clear_removes_all_checkpoints(self, fcp, tmpdir):
        """Test that clear removes all checkpoints."""
        path = os.path.join(tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("test")

        fcp.save("cp1", {path})
        fcp.save("cp2", {path})
        fcp.save("cp3", {path})

        fcp.clear()

        assert fcp.list() == []
        # Storage directory should still exist but be empty
        assert os.path.exists(fcp._storage_dir)

    def test_clear_when_disabled_does_nothing(self, disabled_fcp):
        """Test that clear on disabled FileCheckpoints is safe."""
        # Should not raise
        disabled_fcp.clear()
        assert True


# ============================================================================
# DATACLASS TESTS
# ============================================================================


class TestDataclasses:
    """Test the dataclass structures."""

    def test_file_snapshot_fields(self):
        """Test FileSnapshot dataclass fields."""
        snap = FileSnapshot(
            real_path="/path/to/file.txt",
            saved_copy="/tmp/copy.txt",
            file_hash="abc123",
            size=100,
        )

        assert snap.real_path == "/path/to/file.txt"
        assert snap.saved_copy == "/tmp/copy.txt"
        assert snap.file_hash == "abc123"
        assert snap.size == 100

    def test_file_checkpoint_fields(self):
        """Test FileCheckpoint dataclass fields."""
        cp = FileCheckpoint(name="test")

        assert cp.name == "test"
        assert cp.files == {}
        assert cp.deleted_files == set()

    def test_file_diff_entry_fields(self):
        """Test FileDiffEntry dataclass fields."""
        entry = FileDiffEntry(
            path="/path/to/file.txt",
            status="modified",
            size_a=100,
            size_b=150,
            is_binary=False,
            content_diff="--- a\n+++ b\n",
        )

        assert entry.path == "/path/to/file.txt"
        assert entry.status == "modified"
        assert entry.size_a == 100
        assert entry.size_b == 150
        assert entry.is_binary is False
        assert entry.content_diff is not None

    def test_file_checkpoint_diff_result_has_changes(self):
        """Test FileCheckpointDiffResult.has_changes property."""
        # Empty result
        result = FileCheckpointDiffResult()
        assert result.has_changes is False

        # With added files
        result = FileCheckpointDiffResult(
            added=[FileDiffEntry(path="/a", status="added")]
        )
        assert result.has_changes is True

        # With removed files
        result = FileCheckpointDiffResult(
            removed=[FileDiffEntry(path="/b", status="removed")]
        )
        assert result.has_changes is True

        # With modified files
        result = FileCheckpointDiffResult(
            modified=[FileDiffEntry(path="/c", status="modified")]
        )
        assert result.has_changes is True
