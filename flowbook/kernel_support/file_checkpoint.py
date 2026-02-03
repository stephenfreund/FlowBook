"""
File checkpoint management — save/restore/diff copies of files written by notebook code.

Mirrors the Checkpoints API but for files. Only tracks files the notebook has written.
"""

import difflib
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class FileSnapshot:
    """Snapshot of a single file."""
    real_path: str
    saved_copy: str
    file_hash: str
    size: int


@dataclass
class FileCheckpoint:
    """Snapshot of all tracked files at a point in time."""
    name: str
    files: Dict[str, FileSnapshot] = field(default_factory=dict)
    deleted_files: Set[str] = field(default_factory=set)


@dataclass
class FileDiffEntry:
    """One file's diff entry."""
    path: str
    status: str  # "added", "removed", "modified"
    size_a: Optional[int] = None
    size_b: Optional[int] = None
    is_binary: bool = False
    content_diff: Optional[str] = None  # unified diff for text; None for binary


@dataclass
class FileDiffResult:
    """Result of diffing two file checkpoints."""
    added: List[FileDiffEntry] = field(default_factory=list)
    removed: List[FileDiffEntry] = field(default_factory=list)
    modified: List[FileDiffEntry] = field(default_factory=list)
    changed_paths: Set[str] = field(default_factory=set)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)


def _hash_file(path: str) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_binary(path: str) -> bool:
    """Check if file is binary by looking for null bytes in first 8KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except (OSError, IOError):
        return True


def _unified_diff(path_a: str, path_b: str, label_a: str = "a", label_b: str = "b") -> Optional[str]:
    """Generate unified diff between two text files. Returns None if binary."""
    if _is_binary(path_a) or _is_binary(path_b):
        return None
    try:
        with open(path_a, "r", errors="replace") as f:
            lines_a = f.readlines()
        with open(path_b, "r", errors="replace") as f:
            lines_b = f.readlines()
        diff = difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b)
        result = "".join(diff)
        return result if result else None
    except (OSError, IOError):
        return None


class FileCheckpoints:
    """Manager for file checkpoints — save, restore, and diff file snapshots."""

    def __init__(self):
        self._enabled: bool = False
        self._storage_dir: Optional[str] = None
        self.saved: Dict[str, FileCheckpoint] = {}

    def enable(self) -> None:
        if self._enabled:
            return
        self._storage_dir = tempfile.mkdtemp(prefix="flowbook_file_cp_")
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        if self._storage_dir and os.path.exists(self._storage_dir):
            shutil.rmtree(self._storage_dir, ignore_errors=True)
        self._storage_dir = None
        self._enabled = False
        self.saved.clear()

    def save(self, name: str, write_paths: Set[str], vfs=None) -> FileCheckpoint:
        """
        Save copies of all files in write_paths.

        Args:
            name: Checkpoint name
            write_paths: Set of absolute file paths to snapshot
            vfs: Optional VirtualFileSystem (to resolve overlay paths)

        Returns:
            FileCheckpoint with snapshots of all written files
        """
        cp_dir = os.path.join(self._storage_dir, name)
        if os.path.exists(cp_dir):
            shutil.rmtree(cp_dir)
        os.makedirs(cp_dir, exist_ok=True)

        files = {}
        deleted = set()

        for real_path in write_paths:
            # Resolve to actual file location (overlay or real)
            if vfs and vfs.enabled:
                source = vfs._resolve_read_path(real_path)
            else:
                source = real_path

            if not os.path.exists(source):
                deleted.add(real_path)
                continue

            if os.path.isdir(source):
                continue

            # Create saved copy
            safe_name = real_path.replace(os.sep, "__").lstrip("_")
            saved_path = os.path.join(cp_dir, safe_name)

            try:
                shutil.copy2(source, saved_path)
                file_hash = _hash_file(saved_path)
                size = os.path.getsize(saved_path)
                files[real_path] = FileSnapshot(
                    real_path=real_path,
                    saved_copy=saved_path,
                    file_hash=file_hash,
                    size=size,
                )
            except (OSError, IOError):
                pass

        cp = FileCheckpoint(name=name, files=files, deleted_files=deleted)
        self.saved[name] = cp
        return cp

    def restore(self, name: str, vfs=None) -> None:
        """
        Restore files from a checkpoint.

        Args:
            name: Checkpoint name
            vfs: Optional VirtualFileSystem (to write to overlay instead of real FS)
        """
        cp = self.saved[name]

        for real_path, snapshot in cp.files.items():
            if not os.path.exists(snapshot.saved_copy):
                continue
            if vfs and vfs.enabled:
                # Write to overlay
                overlay = vfs._to_overlay_path(real_path)
                overlay_dir = os.path.dirname(overlay)
                os.makedirs(overlay_dir, exist_ok=True)
                shutil.copy2(snapshot.saved_copy, overlay)
            else:
                # Write directly to real FS
                real_dir = os.path.dirname(real_path)
                if not os.path.exists(real_dir):
                    os.makedirs(real_dir, exist_ok=True)
                shutil.copy2(snapshot.saved_copy, real_path)

        # Handle deleted files
        for deleted_path in cp.deleted_files:
            if vfs and vfs.enabled:
                vfs._deleted_paths.add(deleted_path)
            elif os.path.exists(deleted_path):
                os.remove(deleted_path)

    @staticmethod
    def diff(a: FileCheckpoint, b: FileCheckpoint) -> FileDiffResult:
        """
        Diff two file checkpoints.

        Args:
            a: First (earlier) checkpoint
            b: Second (later) checkpoint

        Returns:
            FileDiffResult with added, removed, and modified entries
        """
        result = FileDiffResult()
        all_paths = set(a.files.keys()) | set(b.files.keys())

        for path in sorted(all_paths):
            in_a = path in a.files and path not in a.deleted_files
            in_b = path in b.files and path not in b.deleted_files

            if in_a and not in_b:
                snap = a.files[path]
                binary = _is_binary(snap.saved_copy) if os.path.exists(snap.saved_copy) else True
                entry = FileDiffEntry(
                    path=path, status="removed",
                    size_a=snap.size, size_b=None,
                    is_binary=binary,
                )
                result.removed.append(entry)
                result.changed_paths.add(path)

            elif not in_a and in_b:
                snap = b.files[path]
                binary = _is_binary(snap.saved_copy) if os.path.exists(snap.saved_copy) else True
                entry = FileDiffEntry(
                    path=path, status="added",
                    size_a=None, size_b=snap.size,
                    is_binary=binary,
                )
                result.added.append(entry)
                result.changed_paths.add(path)

            elif in_a and in_b:
                snap_a = a.files[path]
                snap_b = b.files[path]
                if snap_a.file_hash != snap_b.file_hash:
                    binary = _is_binary(snap_a.saved_copy) or _is_binary(snap_b.saved_copy)
                    content_diff = None
                    if not binary:
                        content_diff = _unified_diff(
                            snap_a.saved_copy, snap_b.saved_copy,
                            label_a=f"a/{os.path.basename(path)}",
                            label_b=f"b/{os.path.basename(path)}",
                        )
                    entry = FileDiffEntry(
                        path=path, status="modified",
                        size_a=snap_a.size, size_b=snap_b.size,
                        is_binary=binary,
                        content_diff=content_diff,
                    )
                    result.modified.append(entry)
                    result.changed_paths.add(path)

        return result

    def exists(self, name: str) -> bool:
        return name in self.saved

    def get(self, name: str) -> FileCheckpoint:
        return self.saved[name]

    def delete(self, name: str) -> None:
        if name in self.saved:
            cp = self.saved[name]
            # Clean up stored copies
            cp_dir = os.path.join(self._storage_dir, name)
            if os.path.exists(cp_dir):
                shutil.rmtree(cp_dir, ignore_errors=True)
            del self.saved[name]

    def list(self) -> List[str]:
        return list(self.saved.keys())

    def clear(self) -> None:
        if self._storage_dir and os.path.exists(self._storage_dir):
            shutil.rmtree(self._storage_dir, ignore_errors=True)
            os.makedirs(self._storage_dir, exist_ok=True)
        self.saved.clear()
