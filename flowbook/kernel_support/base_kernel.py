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
import warnings
from typing import Optional, Set, Tuple

from pandas.errors import ChainedAssignmentError

# Configure pandas ChainedAssignmentError to be raised as an error
warnings.filterwarnings('error', category=ChainedAssignmentError)

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
        # Only track files under the notebook's working directory
        self._vfs.set_notebook_dir(os.getcwd())
        # Full VFS mode by default (writes go to overlay, preserving real FS)
        # Opt out via FLOWBOOK_NO_VIRTUAL_FS=1 or FLOWBOOK_VIRTUAL_FS=0
        if os.environ.get("FLOWBOOK_NO_VIRTUAL_FS", "0") == "1" or \
                os.environ.get("FLOWBOOK_VIRTUAL_FS") == "0":
            self._vfs.enable_tracking_only()
        else:
            self._vfs.enable()

        # Unified checkpointing (memory + files) - always enabled
        self._checkpoints = Checkpoints()
        self._checkpoints.file.enable()

        # Exclude checkpoint storage dir from VFS tracking to prevent feedback loop
        if self._checkpoints.file._storage_dir:
            self._vfs.add_excluded_prefix(self._checkpoints.file._storage_dir)

        # FS magics registration flag
        self._fs_magics_registered = False
        # VFS namespace patching flag
        self._vfs_namespace_patched = False

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

    def _ensure_vfs_namespace_patched(self) -> None:
        """
        Ensure VFS patches are applied to the user namespace.

        IPython puts io.open directly in user_global_ns, bypassing builtins.open.
        This method patches the namespace's 'open' to use our tracking version.
        Called once when the shell is available.
        """
        if self._vfs_namespace_patched:
            return
        if self.shell is None:
            return
        if self._vfs.enabled or self._vfs.tracking_only:
            self._vfs.patch_namespace(self.shell.user_global_ns)
        self._vfs_namespace_patched = True

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

    def _take_checkpoint(self, checkpoint_name: str) -> Tuple[Checkpoint, Set[str]]:
        """
        Take a snapshot of the namespace (and optionally files).

        Uses Checkpoints.save() to deep copy the namespace and
        snapshot written files.

        Returns:
            Tuple of (Checkpoint, set of uncopyable variable names).
            The caller decides how to handle uncopyable variables:
            - Old behavior: remove from user_ns
            - New behavior (FLOWBOOK_UNCOPYABLE_AS_WRITE=1): add to writes
        """
        from flowbook.util.output import timer

        with timer(key="checkpoint:get_write_paths", message="Get VFS write paths"):
            write_paths = self._vfs.get_write_paths() if (self._vfs.enabled or self._vfs.tracking_only) else None

        with timer(key="checkpoint:dict_ns", message="Convert namespace to dict"):
            ns_dict = dict(self.shell.user_ns)

        try:
            total, removed = self._checkpoints.save(
                checkpoint_name,
                ns_dict,
                write_paths=write_paths,
                vfs=self._vfs if self._vfs.enabled else None,
                max_size_mb=None,
            )
        except Exception as e:
            # Checkpoint save can fail due to race conditions (OSError),
            # user variable shadowing stdlib modules (UnboundLocalError),
            # or other issues. Log and continue — the cell can still execute,
            # just without checkpoint protection.
            from flowbook.util.output import error as log_error
            log_error(f"Checkpoint save failed (continuing without checkpoint): {e}")
            total, removed = 0, {}

        uncopyable_vars: Set[str] = set()
        for k, v in removed.items():
            from flowbook.util.output import log
            message = f"The object {k} (type {v}) cannot be checkpointed"
            log(message)
            self._display.display_icon_and_text("\u26a0\ufe0f", message)
            uncopyable_vars.add(k)

        return total, uncopyable_vars

    def _take_checkpoint_incremental(
        self,
        checkpoint_name: str,
        accessed_vars: set,
        prior_checkpoint_name: str,
    ) -> Tuple[Checkpoint, Set[str]]:
        """
        Take an incremental snapshot optimized for untouched variables.

        Reuses deep copies from prior_checkpoint for variables that were not
        accessed during cell execution AND are known leaf objects.

        Args:
            checkpoint_name: Name for the new checkpoint
            accessed_vars: Set of variable names accessed during cell execution
            prior_checkpoint_name: Name of checkpoint to potentially reuse from

        Returns:
            Tuple of (Checkpoint, set of uncopyable variable names).
            The caller decides how to handle uncopyable variables.
        """
        from flowbook.util.output import timer

        with timer(key="checkpoint:get_write_paths", message="Get VFS write paths"):
            write_paths = self._vfs.get_write_paths() if (self._vfs.enabled or self._vfs.tracking_only) else None

        with timer(key="checkpoint:dict_ns", message="Convert namespace to dict"):
            ns_dict = dict(self.shell.user_ns)

        total, removed = self._checkpoints.save_incremental(
            checkpoint_name,
            ns_dict,
            accessed_vars,
            prior_checkpoint_name,
            write_paths=write_paths,
            vfs=self._vfs if self._vfs.enabled else None,
            max_size_mb=None,
        )

        uncopyable_vars: Set[str] = set()
        for k, v in removed.items():
            from flowbook.util.output import log
            message = f"The object {k} (type {v}) cannot be checkpointed"
            log(message)
            self._display.display_icon_and_text("\u26a0\ufe0f", message)
            uncopyable_vars.add(k)

        return total, uncopyable_vars

    def _restore_checkpoint(self, checkpoint_name: str) -> None:
        """Restore memory + file checkpoint."""
        self._checkpoints.restore(
            checkpoint_name,
            self.shell.user_ns,
            vfs=self._vfs if self._vfs.enabled else None,
        )
