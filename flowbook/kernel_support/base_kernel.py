"""
BaseFlowbookKernel - Common base class for FlowBook and Checkpoint kernels.

Provides shared functionality:
- DisplayHelper initialization
- Cell ID tracking
- Checkpointing infrastructure (memory + file via Checkpoints)
- Virtual filesystem (VFS) for file I/O interception
- Cell ID extraction from metadata
- Pure magic detection
- Checkpoint taking
"""

import os
from typing import Optional

from ipykernel.ipkernel import IPythonKernel

from flowbook.kernel_support.checkpoint import Checkpoint, Checkpoints
from flowbook.kernel_support.display_helpers import DisplayHelper
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.virtual_fs import VirtualFileSystem


class BaseFlowbookKernel(IPythonKernel):
    """
    Base kernel with shared infrastructure for FlowBook kernels.

    Subclasses must set:
        implementation, implementation_version, banner

    Subclasses override:
        _do_execute_impl() for their specific execution logic
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Display helper
        self._display = DisplayHelper()

        # Current cell being executed
        self._cell_id: Optional[str] = None

        # Virtual filesystem for file tracking
        self._vfs = VirtualFileSystem()
        # Enable tracking-only mode by default (tracks reads/writes without snapshots)
        self._vfs.enable_tracking_only()
        # Full VFS mode (with file snapshots) can be enabled via environment variable
        if os.environ.get("FLOWBOOK_VIRTUAL_FS", "0") == "1":
            self._vfs.enable()

        # Unified checkpointing (memory + files) - always enabled
        self._checkpoints = Checkpoints()
        self._checkpoints.file.enable()

        # FS magics registration flag
        self._fs_magics_registered = False

    # Backward compat: expose memory checkpoints as _checkpoint
    @property
    def _checkpoint(self) -> MemoryCheckpoints:
        return self._checkpoints.memory

    def _ensure_fs_magics(self) -> None:
        """Register filesystem magics (lazy, once)."""
        if self._fs_magics_registered:
            return
        if self.shell is None:
            return
        from flowbook.kernel_support.fs_magics import FileSystemMagics
        self.shell.register_magics(FileSystemMagics(self.shell, self))
        self._fs_magics_registered = True

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
        *,
        cell_meta: Optional[dict] = None,
        cell_id: Optional[str] = None,
    ) -> dict:
        """
        Execute code, extracting cell ID then delegating to subclass.
        """
        self._cell_id = self._extract_cell_id(cell_id, cell_meta)
        return await self._do_execute_impl(
            code, silent, store_history, user_expressions, allow_stdin, cell_meta
        )

    async def _do_execute_impl(
        self,
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> dict:
        """
        Subclass-specific execution logic.

        Override this instead of do_execute().
        Use _ipython_do_execute() to call the IPythonKernel.do_execute().
        """
        raise NotImplementedError

    async def _ipython_do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
        *,
        cell_meta: Optional[dict] = None,
        cell_id: Optional[str] = None,
    ) -> dict:
        """
        Call IPythonKernel.do_execute() directly, bypassing our do_execute override.

        Use this from _do_execute_impl() to avoid infinite recursion.
        """
        return await IPythonKernel.do_execute(
            self,
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
            cell_meta=cell_meta,
            cell_id=cell_id,
        )

    def _extract_cell_id(
        self, cell_id: Optional[str], cell_meta: Optional[dict]
    ) -> Optional[str]:
        """Extract cell ID from arguments or metadata."""
        if cell_id is not None:
            return cell_id
        if cell_meta is not None:
            return cell_meta.get("cell_id")
        return None

    def _is_pure_magic(self, code: str) -> bool:
        """Check if code is only magic commands (with optional comments)."""
        if "__flowbook_force_checkpoint__" in code:
            return False
        lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
        return all(
            line.startswith("%") or line.startswith("!") or line.startswith("#")
            for line in lines
        )

    def _take_checkpoint(self, checkpoint_name: str) -> Checkpoint:
        """
        Take a snapshot of the namespace (and optionally files).

        Uses Checkpoints.save() to deep copy the namespace and
        snapshot written files.
        Returns the Checkpoint object.
        """
        write_paths = self._vfs.get_write_paths() if (self._vfs.enabled or self._vfs.tracking_only) else None
        total, removed = self._checkpoints.save(
            checkpoint_name,
            dict(self.shell.user_ns),
            write_paths=write_paths,
            vfs=self._vfs if self._vfs.enabled else None,
            max_size_mb=None,
        )

        for k, v in removed.items():
            from flowbook.util.output import log
            message = f"The object {k} (type {v}) cannot be checkpointed"
            log(message)
            self._display.display_icon_and_text("\u26a0\ufe0f", message)
            # Remove variables that couldn't be checkpointed from the namespace
            if k in self.shell.user_ns:
                del self.shell.user_ns[k]

        return total

    def _restore_checkpoint(self, checkpoint_name: str) -> None:
        """Restore memory + file checkpoint."""
        self._checkpoints.restore(
            checkpoint_name,
            self.shell.user_ns,
            vfs=self._vfs if self._vfs.enabled else None,
        )
