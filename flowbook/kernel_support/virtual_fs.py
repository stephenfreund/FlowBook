"""
VirtualFileSystem - Copy-on-write overlay for file I/O interception.

Provides two modes:
1. Full VFS mode: redirects writes to a temp overlay, preserving the real FS
2. Tracking-only mode: records read/write paths without redirecting I/O

Per-cell tracking records which files were read-before-written (analogous to
TrackingDict for variables). This feeds into the reproducibility enforcer
for file-level backward mutation detection and staleness propagation.

Known limitations:
- os.open() (low-level fd) not intercepted
- Not thread-safe (consistent with existing kernel)
"""

import builtins
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class FileTrackingData:
    """Per-cell file tracking results."""
    file_reads_before_writes: Set[str] = field(default_factory=set)
    file_writes: Set[str] = field(default_factory=set)


class VirtualFileSystem:
    """
    Copy-on-write overlay filesystem for notebook file I/O.

    When enabled, writes go to a temp overlay directory. Reads resolve
    overlay-first, then fall back to the real FS. The real FS is
    preserved until explicit commit().

    When disabled but tracking is active (tracking_only mode), file
    paths are recorded without I/O redirection.
    """

    def __init__(self):
        self._enabled: bool = False
        self._tracking_only: bool = False
        self._overlay_dir: Optional[str] = None

        # Cumulative tracking (across all cells)
        self._read_paths: Set[str] = set()
        self._write_paths: Set[str] = set()
        self._deleted_paths: Set[str] = set()

        # Per-cell tracking (reset between cells)
        self._cell_reads_before_writes: Set[str] = set()
        self._cell_writes: Set[str] = set()

        # Original functions (saved for unpatching)
        self._originals: dict = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def tracking_only(self) -> bool:
        return self._tracking_only

    def enable(self) -> None:
        """Enable full VFS mode: create overlay and install patches."""
        if self._enabled:
            return
        self._overlay_dir = tempfile.mkdtemp(prefix="flowbook_vfs_")
        self._enabled = True
        self._tracking_only = False
        self._install_patches()

    def enable_tracking_only(self) -> None:
        """Enable tracking-only mode: record paths without redirecting I/O."""
        if self._enabled or self._tracking_only:
            return
        self._tracking_only = True
        self._install_tracking_patches()

    def disable(self) -> None:
        """Remove patches, discard overlay, clear all state."""
        if self._enabled:
            self._remove_patches()
            if self._overlay_dir and os.path.exists(self._overlay_dir):
                shutil.rmtree(self._overlay_dir, ignore_errors=True)
            self._overlay_dir = None
            self._enabled = False
        elif self._tracking_only:
            self._remove_patches()
            self._tracking_only = False
        self._read_paths.clear()
        self._write_paths.clear()
        self._deleted_paths.clear()
        self._cell_reads_before_writes.clear()
        self._cell_writes.clear()

    def commit(self) -> None:
        """Apply overlay to real FS, then clear overlay state."""
        if not self._enabled or not self._overlay_dir:
            return
        # Temporarily remove all patches so commit uses real FS functions
        self._remove_patches()
        try:
            overlay_root = self._overlay_dir
            for dirpath, dirnames, filenames in os.walk(overlay_root):
                for filename in filenames:
                    overlay_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(overlay_path, overlay_root)
                    real_path = os.path.join("/", rel_path)
                    real_dir = os.path.dirname(real_path)
                    if not os.path.exists(real_dir):
                        os.makedirs(real_dir, exist_ok=True)
                    shutil.copy2(overlay_path, real_path)
            # Apply deletions
            for path in self._deleted_paths:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            # Clear overlay
            shutil.rmtree(self._overlay_dir, ignore_errors=True)
            self._overlay_dir = tempfile.mkdtemp(prefix="flowbook_vfs_")
            self._deleted_paths.clear()
        finally:
            # Re-install patches
            self._install_patches()

    def rollback(self) -> None:
        """Discard overlay contents, clear state."""
        if not self._enabled or not self._overlay_dir:
            return
        # Temporarily remove patches so rollback uses real FS
        self._remove_patches()
        try:
            shutil.rmtree(self._overlay_dir, ignore_errors=True)
            self._overlay_dir = tempfile.mkdtemp(prefix="flowbook_vfs_")
            self._deleted_paths.clear()
        finally:
            self._install_patches()

    def get_read_paths(self) -> Set[str]:
        """Cumulative read paths."""
        return set(self._read_paths)

    def get_write_paths(self) -> Set[str]:
        """Cumulative write paths."""
        return set(self._write_paths)

    def get_cell_file_tracking(self) -> FileTrackingData:
        """Per-cell reads-before-writes and writes."""
        return FileTrackingData(
            file_reads_before_writes=set(self._cell_reads_before_writes),
            file_writes=set(self._cell_writes),
        )

    def reset_cell_tracking(self) -> None:
        """Clear per-cell sets (keep overlay + cumulative)."""
        self._cell_reads_before_writes.clear()
        self._cell_writes.clear()

    # =========================================================================
    # Path Mapping
    # =========================================================================

    def _to_overlay_path(self, real_path: str) -> str:
        """Map a real absolute path to its overlay location."""
        abs_path = os.path.abspath(real_path)
        # Strip leading / so join works correctly
        rel = abs_path.lstrip(os.sep)
        return os.path.join(self._overlay_dir, rel)

    def _resolve_read_path(self, real_path: str) -> str:
        """Resolve a path for reading: overlay first, then real FS."""
        abs_path = os.path.abspath(real_path)
        if abs_path in self._deleted_paths:
            # File was deleted in overlay - return overlay path (will fail on open)
            return self._to_overlay_path(abs_path)
        overlay = self._to_overlay_path(abs_path)
        if os.path.exists(overlay):
            return overlay
        return real_path

    def _track_read(self, path: str) -> None:
        """Record a file read."""
        abs_path = os.path.abspath(path)
        self._read_paths.add(abs_path)
        if abs_path not in self._cell_writes:
            self._cell_reads_before_writes.add(abs_path)

    def _track_write(self, path: str) -> None:
        """Record a file write."""
        abs_path = os.path.abspath(path)
        self._write_paths.add(abs_path)
        self._cell_writes.add(abs_path)

    # =========================================================================
    # Patches — Full VFS Mode
    # =========================================================================

    def _install_patches(self) -> None:
        """Install monkey-patches for full VFS mode."""
        self._originals = {
            "builtins.open": builtins.open,
            "os.remove": os.remove,
            "os.unlink": os.unlink,
            "os.rename": os.rename,
            "os.makedirs": os.makedirs,
            "os.mkdir": os.mkdir,
            "os.rmdir": os.rmdir,
            "os.path.exists": os.path.exists,
            "os.listdir": os.listdir,
            "shutil.copy": shutil.copy,
            "shutil.copy2": shutil.copy2,
            "shutil.move": shutil.move,
            "shutil.rmtree": shutil.rmtree,
        }

        vfs = self
        _orig_open = self._originals["builtins.open"]
        _orig_exists = self._originals["os.path.exists"]
        _orig_listdir = self._originals["os.listdir"]
        _orig_remove = self._originals["os.remove"]
        _orig_makedirs = self._originals["os.makedirs"]
        _orig_mkdir = self._originals["os.mkdir"]
        _orig_rmdir = self._originals["os.rmdir"]
        _orig_rename = self._originals["os.rename"]
        _orig_copy = self._originals["shutil.copy"]
        _orig_copy2 = self._originals["shutil.copy2"]
        _orig_move = self._originals["shutil.move"]
        _orig_rmtree = self._originals["shutil.rmtree"]

        def _ensure_overlay_dir(overlay_path):
            """Create overlay directory using original (unpatched) functions."""
            overlay_dir = os.path.dirname(overlay_path)
            if not _orig_exists(overlay_dir):
                # Temporarily restore to avoid recursion through patched mkdir
                saved_makedirs = os.makedirs
                saved_mkdir = os.mkdir
                os.makedirs = _orig_makedirs
                os.mkdir = _orig_mkdir
                try:
                    _orig_makedirs(overlay_dir, exist_ok=True)
                finally:
                    os.makedirs = saved_makedirs
                    os.mkdir = saved_mkdir

        def patched_open(file, mode="r", *args, **kwargs):
            path = str(file)
            is_write = any(c in mode for c in "wxa+")
            if is_write:
                vfs._track_write(path)
                overlay = vfs._to_overlay_path(path)
                _ensure_overlay_dir(overlay)
                return _orig_open(overlay, mode, *args, **kwargs)
            else:
                vfs._track_read(path)
                resolved = vfs._resolve_read_path(path)
                return _orig_open(resolved, mode, *args, **kwargs)

        def patched_remove(path, *args, **kwargs):
            abs_path = os.path.abspath(path)
            vfs._track_write(path)
            vfs._deleted_paths.add(abs_path)
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                _orig_remove(overlay, *args, **kwargs)

        def patched_rename(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            overlay_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            # If source is in overlay, move it; otherwise, copy from real
            if overlay_src != src:
                _orig_rename(overlay_src, overlay_dst, *args, **kwargs)
            else:
                _orig_copy2(src, overlay_dst)
            abs_src = os.path.abspath(src)
            vfs._deleted_paths.add(abs_src)

        def patched_makedirs(name, *args, **kwargs):
            overlay = vfs._to_overlay_path(name)
            # Temporarily restore originals to avoid recursion
            os.makedirs = _orig_makedirs
            os.mkdir = _orig_mkdir
            try:
                kwargs.setdefault("exist_ok", False)
                return _orig_makedirs(overlay, *args, **kwargs)
            finally:
                os.makedirs = patched_makedirs
                os.mkdir = patched_mkdir

        def patched_mkdir(path, *args, **kwargs):
            overlay = vfs._to_overlay_path(path)
            overlay_parent = os.path.dirname(overlay)
            # Temporarily restore originals to avoid recursion
            os.makedirs = _orig_makedirs
            os.mkdir = _orig_mkdir
            try:
                if not _orig_exists(overlay_parent):
                    _orig_makedirs(overlay_parent, exist_ok=True)
                return _orig_mkdir(overlay, *args, **kwargs)
            finally:
                os.makedirs = patched_makedirs
                os.mkdir = patched_mkdir

        def patched_rmdir(path, *args, **kwargs):
            abs_path = os.path.abspath(path)
            vfs._deleted_paths.add(abs_path)
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                _orig_rmdir(overlay, *args, **kwargs)

        def patched_exists(path):
            abs_path = os.path.abspath(path)
            if abs_path in vfs._deleted_paths:
                return False
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                return True
            return _orig_exists(path)

        def patched_listdir(path="."):
            abs_path = os.path.abspath(path)
            # Merge real FS and overlay
            real_entries = set()
            if _orig_exists(abs_path):
                real_entries = set(_orig_listdir(abs_path))
            overlay = vfs._to_overlay_path(abs_path)
            overlay_entries = set()
            if _orig_exists(overlay):
                overlay_entries = set(_orig_listdir(overlay))
            # Remove deleted entries
            merged = (real_entries | overlay_entries)
            result = []
            for entry in sorted(merged):
                entry_abs = os.path.join(abs_path, entry)
                if entry_abs not in vfs._deleted_paths:
                    result.append(entry)
            return result

        def patched_shutil_copy(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            resolved_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            return _orig_copy(resolved_src, overlay_dst, *args, **kwargs)

        def patched_shutil_copy2(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            resolved_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            return _orig_copy2(resolved_src, overlay_dst, *args, **kwargs)

        def patched_shutil_move(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            resolved_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            if resolved_src != src:
                _orig_move(resolved_src, overlay_dst, *args, **kwargs)
            else:
                _orig_copy2(src, overlay_dst)
            abs_src = os.path.abspath(src)
            vfs._deleted_paths.add(abs_src)

        def patched_shutil_rmtree(path, *args, **kwargs):
            abs_path = os.path.abspath(path)
            vfs._deleted_paths.add(abs_path)
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                _orig_rmtree(overlay, *args, **kwargs)

        builtins.open = patched_open
        os.remove = patched_remove
        os.unlink = patched_remove
        os.rename = patched_rename
        os.makedirs = patched_makedirs
        os.mkdir = patched_mkdir
        os.rmdir = patched_rmdir
        os.path.exists = patched_exists
        os.listdir = patched_listdir
        shutil.copy = patched_shutil_copy
        shutil.copy2 = patched_shutil_copy2
        shutil.move = patched_shutil_move
        shutil.rmtree = patched_shutil_rmtree

    # =========================================================================
    # Patches — Tracking-Only Mode
    # =========================================================================

    def _install_tracking_patches(self) -> None:
        """Install lightweight patches that only track paths (no I/O redirect)."""
        self._originals = {
            "builtins.open": builtins.open,
        }

        vfs = self
        _orig_open = self._originals["builtins.open"]

        def patched_open(file, mode="r", *args, **kwargs):
            path = str(file)
            is_write = any(c in mode for c in "wxa+")
            if is_write:
                vfs._track_write(path)
            else:
                vfs._track_read(path)
            return _orig_open(file, mode, *args, **kwargs)

        builtins.open = patched_open

    # =========================================================================
    # Unpatch
    # =========================================================================

    def _remove_patches(self) -> None:
        """Restore all original functions."""
        if "builtins.open" in self._originals:
            builtins.open = self._originals["builtins.open"]
        if "os.remove" in self._originals:
            os.remove = self._originals["os.remove"]
        if "os.unlink" in self._originals:
            os.unlink = self._originals["os.unlink"]
        if "os.rename" in self._originals:
            os.rename = self._originals["os.rename"]
        if "os.makedirs" in self._originals:
            os.makedirs = self._originals["os.makedirs"]
        if "os.mkdir" in self._originals:
            os.mkdir = self._originals["os.mkdir"]
        if "os.rmdir" in self._originals:
            os.rmdir = self._originals["os.rmdir"]
        if "os.path.exists" in self._originals:
            os.path.exists = self._originals["os.path.exists"]
        if "os.listdir" in self._originals:
            os.listdir = self._originals["os.listdir"]
        if "shutil.copy" in self._originals:
            shutil.copy = self._originals["shutil.copy"]
        if "shutil.copy2" in self._originals:
            shutil.copy2 = self._originals["shutil.copy2"]
        if "shutil.move" in self._originals:
            shutil.move = self._originals["shutil.move"]
        if "shutil.rmtree" in self._originals:
            shutil.rmtree = self._originals["shutil.rmtree"]
        self._originals.clear()
