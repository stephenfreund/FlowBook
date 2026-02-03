"""
TotalCheckpoint — unified memory + file checkpoint wrapper.

Composes a memory Checkpoint (namespace snapshot) with an optional
FileCheckpoint (file snapshot) into a single object. Provides a
combined diff() that returns both memory and file differences.

Naming:
- Checkpoint (from checkpoint.py) = memory/namespace snapshot
- FileCheckpoint (from file_checkpoint.py) = file snapshot
- TotalCheckpoint = combined memory + file
- Checkpoints (from checkpoint.py) = memory checkpoint manager
- FileCheckpoints (from file_checkpoint.py) = file checkpoint manager
- TotalCheckpoints = combined manager
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from flowbook.kernel_support.checkpoint import Checkpoint, Checkpoints
from flowbook.kernel_support.file_checkpoint import (
    FileCheckpoint,
    FileCheckpoints,
    FileDiffResult,
)
from flowbook.kernel_support.types import DiffResult


@dataclass
class TotalDiffResult:
    """Combined diff result for memory + files."""

    memory: DiffResult
    file: Optional[FileDiffResult] = None

    # Convenience accessors delegating to memory (backward compat for enforcer)
    @property
    def differences(self) -> dict:
        return self.memory.differences

    @property
    def warnings(self) -> list:
        return self.memory.warnings

    @property
    def changed_file_paths(self) -> Set[str]:
        return self.file.changed_paths if self.file else set()

    @property
    def has_file_changes(self) -> bool:
        return self.file is not None and self.file.has_changes


@dataclass
class TotalCheckpoint:
    """Combined memory + file snapshot."""

    memory: Checkpoint
    file: Optional[FileCheckpoint] = None

    # Forward common Checkpoint attributes for backward compat
    @property
    def user_ns(self):
        return self.memory.user_ns

    @property
    def name(self):
        return self.memory.name

    def get_aliases_for_vars(self, accessed_vars, log_aliases=True):
        return self.memory.get_aliases_for_vars(accessed_vars, log_aliases=log_aliases)

    @staticmethod
    def diff(
        a: "TotalCheckpoint",
        b: "TotalCheckpoint",
        keys_to_include=None,
        use_leq: bool = False,
        column_rbw=None,
        structural_reads=None,
        structural_mode=None,
    ) -> TotalDiffResult:
        """Diff both memory and files in one call."""
        mem_diff = Checkpoint.diff(
            a.memory,
            b.memory,
            keys_to_include=keys_to_include,
            use_leq=use_leq,
            column_rbw=column_rbw,
            structural_reads=structural_reads,
            structural_mode=structural_mode,
        )
        file_diff = None
        if a.file is not None and b.file is not None:
            file_diff = FileCheckpoints.diff(a.file, b.file)
        return TotalDiffResult(memory=mem_diff, file=file_diff)


class TotalCheckpoints:
    """Manager composing memory Checkpoints + FileCheckpoints."""

    def __init__(self):
        self.memory = Checkpoints(sanity_check=False, warn_classes=False)
        self.file = FileCheckpoints()

    def save(
        self,
        name: str,
        user_ns: dict,
        write_paths: Optional[Set[str]] = None,
        vfs=None,
        max_size_mb=None,
    ) -> Tuple[TotalCheckpoint, dict]:
        """
        Save memory + file checkpoint.

        Returns:
            Tuple of (TotalCheckpoint, removed_vars dict)
        """
        saved, removed = self.memory.save(name, user_ns, max_size_mb=max_size_mb)

        file_cp = None
        if self.file._enabled and write_paths is not None:
            file_cp = self.file.save(name, write_paths, vfs=vfs)

        total = TotalCheckpoint(
            memory=self.memory.saved[name],
            file=file_cp,
        )
        return total, removed

    def restore(self, name: str, user_ns: dict, vfs=None) -> None:
        """Restore memory + file checkpoint."""
        self.memory.restore(name, user_ns)
        if self.file._enabled and self.file.exists(name):
            self.file.restore(name, vfs=vfs)

    def get(self, name: str) -> TotalCheckpoint:
        return TotalCheckpoint(
            memory=self.memory.saved[name],
            file=self.file.saved.get(name),
        )

    def exists(self, name: str) -> bool:
        return name in self.memory.saved

    def delete(self, name: str) -> None:
        self.memory.delete(name)
        self.file.delete(name)

    def list(self) -> List[str]:
        return self.memory.list()

    def clear(self) -> None:
        self.memory.clear()
        self.file.clear()
