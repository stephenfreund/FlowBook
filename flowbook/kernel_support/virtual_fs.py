"""
VirtualFileSystem - Copy-on-write overlay for file I/O interception.

Provides two modes:
1. Full VFS mode: redirects writes to a temp overlay, preserving the real FS
2. Tracking-only mode: records read/write paths without redirecting I/O

Per-cell tracking records which files were read-before-written (analogous to
TrackingDict for variables). This feeds into the reproducibility enforcer
for file-level backward mutation detection and staleness propagation.

Known limitations:
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


# Sentinel value for namespace patching
_NOT_PRESENT = object()


# --- Low-level fd flag helpers ---

# Access mode is stored in the lowest 2 bits of flags
_O_ACCMODE = os.O_RDONLY | os.O_WRONLY | os.O_RDWR


def _is_write_flags(flags: int) -> bool:
    """True if flags indicate a write-capable open."""
    access = flags & _O_ACCMODE
    if access in (os.O_WRONLY, os.O_RDWR):
        return True
    # O_CREAT, O_APPEND, O_TRUNC also imply write intent
    return bool(flags & (os.O_CREAT | os.O_APPEND | os.O_TRUNC))


def _is_read_flags(flags: int) -> bool:
    """True if flags indicate a read-capable open."""
    access = flags & _O_ACCMODE
    return access in (os.O_RDONLY, os.O_RDWR)


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

        # Path filtering: only track paths under _notebook_dir (if set)
        self._notebook_dir: Optional[str] = None
        self._excluded_prefixes: Set[str] = set()

        # Original functions (saved for unpatching)
        self._originals: dict = {}

        # Patched namespaces and their original 'open' values
        # Maps namespace id -> (namespace, original_open_or_NOT_PRESENT)
        self._patched_namespaces: dict = {}

        # The current patched_open function (created during enable)
        self._patched_open = None

        # Low-level fd tracking: maps open fd -> original abs path
        self._fd_to_path: dict = {}

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
        # If transitioning from tracking-only mode, save namespaces and clean up
        saved_namespaces = []
        if self._tracking_only:
            # Save namespace references before removing patches
            saved_namespaces = [ns for ns_id, (ns, orig) in self._patched_namespaces.items()]
            self._remove_patches()
            self._tracking_only = False
        self._overlay_dir = tempfile.mkdtemp(prefix="flowbook_vfs_")
        self._enabled = True
        self._install_patches()
        # Re-patch any previously patched namespaces with new patched_open
        for namespace in saved_namespaces:
            self.patch_namespace(namespace)

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
        self._fd_to_path.clear()

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

    def set_notebook_dir(self, path: str) -> None:
        """Set the notebook directory. Only files under this tree will be tracked."""
        self._notebook_dir = os.path.abspath(path).rstrip(os.sep) + os.sep

    def add_excluded_prefix(self, prefix: str) -> None:
        """Add a path prefix to exclude from tracking (e.g., checkpoint storage dir)."""
        abs_prefix = os.path.abspath(prefix)
        self._excluded_prefixes.add(abs_prefix)

    def _should_overlay(self, abs_path: str) -> bool:
        """Check if a path should be redirected to the overlay.

        Only redirects files under the notebook directory tree (if set).
        Paths outside (e.g., /dev/shm, /tmp) pass through to real FS.
        """
        if self._notebook_dir is None:
            return True
        return abs_path.startswith(self._notebook_dir)

    def _should_track_path(self, abs_path: str) -> bool:
        """Check if a path should be tracked.

        Only tracks files under the notebook directory tree (if set).
        Always excludes internal FlowBook paths and explicitly excluded prefixes.
        """
        # Filter out dynamically excluded prefixes (e.g., checkpoint storage dir)
        for prefix in self._excluded_prefixes:
            if abs_path.startswith(prefix):
                return False

        # Only track files under the notebook directory tree
        if self._notebook_dir is not None:
            if not abs_path.startswith(self._notebook_dir):
                return False

        return True

    def _track_read(self, path) -> None:
        """Record a file read."""
        # Convert bytes paths to str (e.g., psutil passes b"/proc" to os.listdir at shutdown)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        abs_path = os.path.abspath(path)
        if not self._should_track_path(abs_path):
            return
        self._read_paths.add(abs_path)
        if abs_path not in self._cell_writes:
            self._cell_reads_before_writes.add(abs_path)

    def _track_write(self, path) -> None:
        """Record a file write."""
        # Convert bytes paths to str (e.g., psutil passes b"/proc" to os.listdir at shutdown)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        abs_path = os.path.abspath(path)
        if not self._should_track_path(abs_path):
            return
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
            # Low-level fd operations
            "os.open": os.open,
            "os.close": os.close,
            "os.read": os.read,
            "os.write": os.write,
            "os.fdopen": os.fdopen,
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
            if isinstance(file, bytes):
                return _orig_open(file, mode, *args, **kwargs)
            path = str(file)
            abs_path = os.path.abspath(path)
            is_write = any(c in mode for c in "wxa+")
            if not vfs._should_overlay(abs_path):
                if is_write:
                    vfs._track_write(path)
                else:
                    vfs._track_read(path)
                return _orig_open(file, mode, *args, **kwargs)
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
            if isinstance(path, bytes):
                return _orig_remove(path, *args, **kwargs)
            abs_path = os.path.abspath(path)
            if not vfs._should_overlay(abs_path):
                vfs._track_write(path)
                return _orig_remove(path, *args, **kwargs)
            vfs._track_write(path)
            vfs._deleted_paths.add(abs_path)
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                _orig_remove(overlay, *args, **kwargs)

        def patched_rename(src, dst, *args, **kwargs):
            if isinstance(src, bytes) or isinstance(dst, bytes):
                return _orig_rename(src, dst, *args, **kwargs)
            abs_src = os.path.abspath(src)
            abs_dst = os.path.abspath(dst)
            if not vfs._should_overlay(abs_src) and not vfs._should_overlay(abs_dst):
                vfs._track_read(src)
                vfs._track_write(dst)
                return _orig_rename(src, dst, *args, **kwargs)
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
            vfs._deleted_paths.add(abs_src)

        def patched_makedirs(name, *args, **kwargs):
            if isinstance(name, bytes):
                return _orig_makedirs(name, *args, **kwargs)
            abs_path = os.path.abspath(name)
            if not vfs._should_overlay(abs_path):
                vfs._track_write(name)
                return _orig_makedirs(name, *args, **kwargs)
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
            if isinstance(path, bytes):
                return _orig_mkdir(path, *args, **kwargs)
            abs_path = os.path.abspath(path)
            if not vfs._should_overlay(abs_path):
                vfs._track_write(path)
                return _orig_mkdir(path, *args, **kwargs)
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
            if isinstance(path, bytes):
                return _orig_rmdir(path, *args, **kwargs)
            abs_path = os.path.abspath(path)
            if not vfs._should_overlay(abs_path):
                vfs._track_write(path)
                return _orig_rmdir(path, *args, **kwargs)
            vfs._deleted_paths.add(abs_path)
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                _orig_rmdir(overlay, *args, **kwargs)

        def patched_exists(path):
            if isinstance(path, bytes):
                return _orig_exists(path)
            abs_path = os.path.abspath(path)
            if not vfs._should_overlay(abs_path):
                return _orig_exists(path)
            if abs_path in vfs._deleted_paths:
                return False
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                return True
            return _orig_exists(path)

        def patched_listdir(path="."):
            if isinstance(path, bytes):
                return _orig_listdir(path)
            abs_path = os.path.abspath(path)
            if not vfs._should_overlay(abs_path):
                return _orig_listdir(path)
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
            if isinstance(src, bytes) or isinstance(dst, bytes):
                return _orig_copy(src, dst, *args, **kwargs)
            abs_src = os.path.abspath(src)
            abs_dst = os.path.abspath(dst)
            if not vfs._should_overlay(abs_src) and not vfs._should_overlay(abs_dst):
                vfs._track_read(src)
                vfs._track_write(dst)
                return _orig_copy(src, dst, *args, **kwargs)
            vfs._track_read(src)
            vfs._track_write(dst)
            resolved_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            return _orig_copy(resolved_src, overlay_dst, *args, **kwargs)

        def patched_shutil_copy2(src, dst, *args, **kwargs):
            if isinstance(src, bytes) or isinstance(dst, bytes):
                return _orig_copy2(src, dst, *args, **kwargs)
            abs_src = os.path.abspath(src)
            abs_dst = os.path.abspath(dst)
            if not vfs._should_overlay(abs_src) and not vfs._should_overlay(abs_dst):
                vfs._track_read(src)
                vfs._track_write(dst)
                return _orig_copy2(src, dst, *args, **kwargs)
            vfs._track_read(src)
            vfs._track_write(dst)
            resolved_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            return _orig_copy2(resolved_src, overlay_dst, *args, **kwargs)

        def patched_shutil_move(src, dst, *args, **kwargs):
            if isinstance(src, bytes) or isinstance(dst, bytes):
                return _orig_move(src, dst, *args, **kwargs)
            abs_src = os.path.abspath(src)
            abs_dst = os.path.abspath(dst)
            if not vfs._should_overlay(abs_src) and not vfs._should_overlay(abs_dst):
                vfs._track_read(src)
                vfs._track_write(dst)
                return _orig_move(src, dst, *args, **kwargs)
            vfs._track_read(src)
            vfs._track_write(dst)
            resolved_src = vfs._resolve_read_path(src)
            overlay_dst = vfs._to_overlay_path(dst)
            _ensure_overlay_dir(overlay_dst)
            if resolved_src != src:
                _orig_move(resolved_src, overlay_dst, *args, **kwargs)
            else:
                _orig_copy2(src, overlay_dst)
            vfs._deleted_paths.add(abs_src)

        def patched_shutil_rmtree(path, *args, **kwargs):
            if isinstance(path, bytes):
                return _orig_rmtree(path, *args, **kwargs)
            abs_path = os.path.abspath(path)
            if not vfs._should_overlay(abs_path):
                vfs._track_write(path)
                return _orig_rmtree(path, *args, **kwargs)
            vfs._deleted_paths.add(abs_path)
            overlay = vfs._to_overlay_path(abs_path)
            if _orig_exists(overlay):
                _orig_rmtree(overlay, *args, **kwargs)

        # --- Low-level fd patches ---
        _orig_os_open = self._originals["os.open"]
        _orig_os_close = self._originals["os.close"]
        _orig_os_read = self._originals["os.read"]
        _orig_os_write = self._originals["os.write"]
        _orig_os_fdopen = self._originals["os.fdopen"]

        def patched_os_open(path, flags, mode=0o777, *, dir_fd=None):
            if isinstance(path, bytes):
                return _orig_os_open(path, flags, mode, dir_fd=dir_fd)
            str_path = str(path)
            abs_path = os.path.abspath(str_path)
            is_write = _is_write_flags(flags)
            is_read = _is_read_flags(flags)
            if not vfs._should_overlay(abs_path):
                if is_write:
                    vfs._track_write(str_path)
                if is_read:
                    vfs._track_read(str_path)
                fd = _orig_os_open(path, flags, mode, dir_fd=dir_fd)
                vfs._fd_to_path[fd] = abs_path
                return fd
            if is_write:
                vfs._track_write(str_path)
                if is_read:
                    vfs._track_read(str_path)
                overlay = vfs._to_overlay_path(str_path)
                _ensure_overlay_dir(overlay)
                fd = _orig_os_open(overlay, flags, mode, dir_fd=dir_fd)
            else:
                vfs._track_read(str_path)
                resolved = vfs._resolve_read_path(str_path)
                fd = _orig_os_open(resolved, flags, mode, dir_fd=dir_fd)
            vfs._fd_to_path[fd] = abs_path
            return fd

        def patched_os_close(fd):
            vfs._fd_to_path.pop(fd, None)
            return _orig_os_close(fd)

        def patched_os_read(fd, n):
            path = vfs._fd_to_path.get(fd)
            if path is not None:
                vfs._track_read(path)
            return _orig_os_read(fd, n)

        def patched_os_write(fd, data):
            path = vfs._fd_to_path.get(fd)
            if path is not None:
                vfs._track_write(path)
            return _orig_os_write(fd, data)

        def patched_os_fdopen(fd, *args, **kwargs):
            return _orig_os_fdopen(fd, *args, **kwargs)

        builtins.open = patched_open
        self._patched_open = patched_open
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
        os.open = patched_os_open
        os.close = patched_os_close
        os.read = patched_os_read
        os.write = patched_os_write
        os.fdopen = patched_os_fdopen

    # =========================================================================
    # Patches — Tracking-Only Mode
    # =========================================================================

    def _install_tracking_patches(self) -> None:
        """Install lightweight patches that only track paths (no I/O redirect)."""
        self._originals = {
            # File I/O
            "builtins.open": builtins.open,
            # Write operations (modify filesystem)
            "os.remove": os.remove,
            "os.unlink": os.unlink,
            "os.rmdir": os.rmdir,
            "os.rename": os.rename,
            "os.makedirs": os.makedirs,
            "os.mkdir": os.mkdir,
            "shutil.copy": shutil.copy,
            "shutil.copy2": shutil.copy2,
            "shutil.move": shutil.move,
            "shutil.rmtree": shutil.rmtree,
            # Read operations (query filesystem)
            "os.path.exists": os.path.exists,
            "os.listdir": os.listdir,
            "os.stat": os.stat,
            "os.path.isfile": os.path.isfile,
            "os.path.isdir": os.path.isdir,
            "os.path.getsize": os.path.getsize,
            "os.path.getmtime": os.path.getmtime,
            # Low-level fd operations
            "os.open": os.open,
            "os.close": os.close,
            "os.read": os.read,
            "os.write": os.write,
            "os.fdopen": os.fdopen,
        }

        vfs = self
        _orig_open = self._originals["builtins.open"]
        _orig_remove = self._originals["os.remove"]
        _orig_unlink = self._originals["os.unlink"]
        _orig_rmdir = self._originals["os.rmdir"]
        _orig_rename = self._originals["os.rename"]
        _orig_makedirs = self._originals["os.makedirs"]
        _orig_mkdir = self._originals["os.mkdir"]
        _orig_shutil_copy = self._originals["shutil.copy"]
        _orig_shutil_copy2 = self._originals["shutil.copy2"]
        _orig_shutil_move = self._originals["shutil.move"]
        _orig_shutil_rmtree = self._originals["shutil.rmtree"]
        _orig_exists = self._originals["os.path.exists"]
        _orig_listdir = self._originals["os.listdir"]
        _orig_stat = self._originals["os.stat"]
        _orig_isfile = self._originals["os.path.isfile"]
        _orig_isdir = self._originals["os.path.isdir"]
        _orig_getsize = self._originals["os.path.getsize"]
        _orig_getmtime = self._originals["os.path.getmtime"]

        # --- File I/O ---
        def patched_open(file, mode="r", *args, **kwargs):
            path = str(file)
            is_write = any(c in mode for c in "wxa+")
            if is_write:
                vfs._track_write(path)
            else:
                vfs._track_read(path)
            return _orig_open(file, mode, *args, **kwargs)

        # --- Write operations (delete) ---
        def patched_remove(path, *args, **kwargs):
            vfs._track_write(path)
            return _orig_remove(path, *args, **kwargs)

        def patched_unlink(path, *args, **kwargs):
            vfs._track_write(path)
            return _orig_unlink(path, *args, **kwargs)

        def patched_rmdir(path, *args, **kwargs):
            vfs._track_write(path)
            return _orig_rmdir(path, *args, **kwargs)

        def patched_shutil_rmtree(path, *args, **kwargs):
            vfs._track_write(path)
            return _orig_shutil_rmtree(path, *args, **kwargs)

        # --- Write operations (create) ---
        def patched_makedirs(name, *args, **kwargs):
            vfs._track_write(name)
            return _orig_makedirs(name, *args, **kwargs)

        def patched_mkdir(path, *args, **kwargs):
            vfs._track_write(path)
            return _orig_mkdir(path, *args, **kwargs)

        # --- Read + Write operations (copy/move) ---
        def patched_rename(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            return _orig_rename(src, dst, *args, **kwargs)

        def patched_shutil_copy(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            return _orig_shutil_copy(src, dst, *args, **kwargs)

        def patched_shutil_copy2(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            return _orig_shutil_copy2(src, dst, *args, **kwargs)

        def patched_shutil_move(src, dst, *args, **kwargs):
            vfs._track_read(src)
            vfs._track_write(dst)
            return _orig_shutil_move(src, dst, *args, **kwargs)

        # --- Read operations ---
        def patched_exists(path):
            vfs._track_read(path)
            return _orig_exists(path)

        def patched_listdir(path="."):
            vfs._track_read(path)
            return _orig_listdir(path)

        def patched_stat(path, *args, **kwargs):
            vfs._track_read(path)
            return _orig_stat(path, *args, **kwargs)

        def patched_isfile(path):
            vfs._track_read(path)
            return _orig_isfile(path)

        def patched_isdir(path):
            vfs._track_read(path)
            return _orig_isdir(path)

        def patched_getsize(filename):
            vfs._track_read(filename)
            return _orig_getsize(filename)

        def patched_getmtime(filename):
            vfs._track_read(filename)
            return _orig_getmtime(filename)

        # --- Low-level fd patches ---
        _orig_os_open = self._originals["os.open"]
        _orig_os_close = self._originals["os.close"]
        _orig_os_read = self._originals["os.read"]
        _orig_os_write = self._originals["os.write"]
        _orig_os_fdopen = self._originals["os.fdopen"]

        def patched_os_open(path, flags, mode=0o777, *, dir_fd=None):
            if isinstance(path, bytes):
                return _orig_os_open(path, flags, mode, dir_fd=dir_fd)
            str_path = str(path)
            if _is_write_flags(flags):
                vfs._track_write(str_path)
            if _is_read_flags(flags):
                vfs._track_read(str_path)
            fd = _orig_os_open(path, flags, mode, dir_fd=dir_fd)
            abs_path = os.path.abspath(str_path)
            vfs._fd_to_path[fd] = abs_path
            return fd

        def patched_os_close(fd):
            vfs._fd_to_path.pop(fd, None)
            return _orig_os_close(fd)

        def patched_os_read(fd, n):
            path = vfs._fd_to_path.get(fd)
            if path is not None:
                vfs._track_read(path)
            return _orig_os_read(fd, n)

        def patched_os_write(fd, data):
            path = vfs._fd_to_path.get(fd)
            if path is not None:
                vfs._track_write(path)
            return _orig_os_write(fd, data)

        def patched_os_fdopen(fd, *args, **kwargs):
            return _orig_os_fdopen(fd, *args, **kwargs)

        # Install patches
        builtins.open = patched_open
        self._patched_open = patched_open
        os.remove = patched_remove
        os.unlink = patched_unlink
        os.rmdir = patched_rmdir
        os.rename = patched_rename
        os.makedirs = patched_makedirs
        os.mkdir = patched_mkdir
        shutil.copy = patched_shutil_copy
        shutil.copy2 = patched_shutil_copy2
        shutil.move = patched_shutil_move
        shutil.rmtree = patched_shutil_rmtree
        os.path.exists = patched_exists
        os.listdir = patched_listdir
        os.stat = patched_stat
        os.path.isfile = patched_isfile
        os.path.isdir = patched_isdir
        os.path.getsize = patched_getsize
        os.path.getmtime = patched_getmtime
        os.open = patched_os_open
        os.close = patched_os_close
        os.read = patched_os_read
        os.write = patched_os_write
        os.fdopen = patched_os_fdopen

    def patch_namespace(self, namespace: dict) -> None:
        """
        Patch a namespace's 'open' to use our tracking open.

        This is needed because IPython puts io.open directly in user_global_ns,
        bypassing builtins.open. We need to patch that namespace too.

        Safe to call multiple times on the same namespace.
        """
        if not (self._enabled or self._tracking_only):
            return
        if self._patched_open is None:
            return

        ns_id = id(namespace)
        if ns_id in self._patched_namespaces:
            return  # Already patched

        # Save original value (or sentinel if not present)
        original = namespace.get("open", _NOT_PRESENT)
        self._patched_namespaces[ns_id] = (namespace, original)

        # Install our patched open
        namespace["open"] = self._patched_open

    def _unpatch_namespaces(self) -> None:
        """Restore all patched namespaces to their original state."""
        for ns_id, (namespace, original) in self._patched_namespaces.items():
            if original is _NOT_PRESENT:
                namespace.pop("open", None)
            else:
                namespace["open"] = original
        self._patched_namespaces.clear()

    # =========================================================================
    # Unpatch
    # =========================================================================

    def _remove_patches(self) -> None:
        """Restore all original functions."""
        # Restore patched namespaces first
        self._unpatch_namespaces()
        self._patched_open = None

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
        if "os.stat" in self._originals:
            os.stat = self._originals["os.stat"]
        if "os.path.isfile" in self._originals:
            os.path.isfile = self._originals["os.path.isfile"]
        if "os.path.isdir" in self._originals:
            os.path.isdir = self._originals["os.path.isdir"]
        if "os.path.getsize" in self._originals:
            os.path.getsize = self._originals["os.path.getsize"]
        if "os.path.getmtime" in self._originals:
            os.path.getmtime = self._originals["os.path.getmtime"]
        if "os.open" in self._originals:
            os.open = self._originals["os.open"]
        if "os.close" in self._originals:
            os.close = self._originals["os.close"]
        if "os.read" in self._originals:
            os.read = self._originals["os.read"]
        if "os.write" in self._originals:
            os.write = self._originals["os.write"]
        if "os.fdopen" in self._originals:
            os.fdopen = self._originals["os.fdopen"]
        self._originals.clear()
