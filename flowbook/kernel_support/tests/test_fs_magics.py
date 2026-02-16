"""Tests for fs_magics.py - FileSystem magic commands.

Tests all three magic commands: %virtual_fs, %file_checkpoints, %commit_files.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from flowbook.kernel_support.fs_magics import FileSystemMagics


def _make_kernel_and_magics():
    """Create a mock kernel and FileSystemMagics instance."""
    shell = MagicMock()
    kernel = MagicMock()
    kernel._vfs = MagicMock()
    kernel._display = MagicMock()
    kernel._checkpoints = MagicMock()
    kernel._checkpoints.file = MagicMock()
    magics = FileSystemMagics.__new__(FileSystemMagics)
    magics._kernel = kernel
    magics.shell = shell
    return kernel, magics


class TestVirtualFsMagic:
    """Tests for %virtual_fs magic command."""

    def test_status_when_enabled(self):
        """Show status when VFS is fully enabled."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = True
        kernel._vfs.tracking_only = False
        magics.virtual_fs("")
        kernel._display.display_icon_and_text.assert_called_once()
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "full VFS mode" in args[1]

    def test_status_when_tracking_only(self):
        """Show status when VFS is in tracking-only mode."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = False
        kernel._vfs.tracking_only = True
        magics.virtual_fs("?")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "tracking-only" in args[1]

    def test_status_when_disabled(self):
        """Show status when VFS is disabled."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = False
        kernel._vfs.tracking_only = False
        magics.virtual_fs("")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "OFF" in args[1]

    def test_enable_vfs(self):
        """Enable VFS with 'on'."""
        kernel, magics = _make_kernel_and_magics()
        magics.virtual_fs("on")
        kernel._vfs.enable.assert_called_once()

    def test_enable_vfs_variations(self):
        """Enable VFS with various aliases."""
        for arg in ("true", "1", "enable"):
            kernel, magics = _make_kernel_and_magics()
            magics.virtual_fs(arg)
            kernel._vfs.enable.assert_called_once()

    def test_disable_vfs(self):
        """Disable VFS with 'off'."""
        kernel, magics = _make_kernel_and_magics()
        magics.virtual_fs("off")
        kernel._vfs.disable.assert_called_once()

    def test_disable_vfs_variations(self):
        """Disable VFS with various aliases."""
        for arg in ("false", "0", "disable"):
            kernel, magics = _make_kernel_and_magics()
            magics.virtual_fs(arg)
            kernel._vfs.disable.assert_called_once()

    def test_invalid_argument(self):
        """Invalid argument shows error message."""
        kernel, magics = _make_kernel_and_magics()
        magics.virtual_fs("invalid")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "error" in args[0]
        assert "invalid" in args[1].lower()


class TestFileCheckpointsMagic:
    """Tests for %file_checkpoints magic command."""

    def test_status_enabled(self):
        """Show status when file checkpoints are enabled."""
        kernel, magics = _make_kernel_and_magics()
        kernel._checkpoints.file._enabled = True
        magics.file_checkpoints("?")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "ON" in args[1]

    def test_status_disabled(self):
        """Show status when file checkpoints are disabled."""
        kernel, magics = _make_kernel_and_magics()
        kernel._checkpoints.file._enabled = False
        magics.file_checkpoints("")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "OFF" in args[1]

    def test_enable_file_checkpoints(self):
        """Enable file checkpoints with 'on'."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = False
        kernel._vfs.tracking_only = False
        magics.file_checkpoints("on")
        kernel._checkpoints.file.enable.assert_called_once()
        # Should also enable tracking-only VFS
        kernel._vfs.enable_tracking_only.assert_called_once()

    def test_enable_file_checkpoints_vfs_already_enabled(self):
        """Enable file checkpoints when VFS is already enabled."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = True
        kernel._vfs.tracking_only = False
        magics.file_checkpoints("on")
        kernel._checkpoints.file.enable.assert_called_once()
        # Should NOT try to enable tracking-only when full VFS is on
        kernel._vfs.enable_tracking_only.assert_not_called()

    def test_enable_file_checkpoints_tracking_already_on(self):
        """Enable file checkpoints when tracking-only is already on."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = False
        kernel._vfs.tracking_only = True
        magics.file_checkpoints("enable")
        kernel._checkpoints.file.enable.assert_called_once()
        # Should NOT try to enable tracking-only when it's already on
        kernel._vfs.enable_tracking_only.assert_not_called()

    def test_disable_file_checkpoints(self):
        """Disable file checkpoints with 'off'."""
        kernel, magics = _make_kernel_and_magics()
        magics.file_checkpoints("off")
        kernel._checkpoints.file.disable.assert_called_once()

    def test_invalid_argument(self):
        """Invalid argument shows error message."""
        kernel, magics = _make_kernel_and_magics()
        magics.file_checkpoints("bad")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "error" in args[0]


class TestCommitFilesMagic:
    """Tests for %commit_files magic command."""

    def test_commit_when_vfs_disabled(self):
        """Commit shows error when VFS is disabled."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = False
        magics.commit_files("")
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "error" in args[0]
        assert "not enabled" in args[1]

    def test_commit_when_vfs_enabled(self):
        """Commit applies overlay and shows count."""
        kernel, magics = _make_kernel_and_magics()
        kernel._vfs.enabled = True
        kernel._vfs.get_write_paths.return_value = {"/a.txt", "/b.txt"}
        magics.commit_files("")
        kernel._vfs.commit.assert_called_once()
        args = kernel._display.display_icon_and_text.call_args[0]
        assert "2" in args[1]
