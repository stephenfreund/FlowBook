"""
BaseFlowbookKernel - Common base class for FlowBook and Checkpoint kernels.

Provides shared functionality:
- DisplayHelper initialization
- Cell ID tracking
- Checkpointing infrastructure
- Cell ID extraction from metadata
- Pure magic detection
- Checkpoint taking
"""

from typing import Optional

from ipykernel.ipkernel import IPythonKernel

from flowbook.kernel_support.checkpoint import Checkpoint, Checkpoints
from flowbook.kernel_support.display_helpers import DisplayHelper


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

        # Checkpointing
        self._checkpoint = Checkpoints(
            sanity_check=False,
            warn_classes=False,
        )

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
        Take a snapshot of the namespace.

        Uses Checkpoints.save() to properly deep copy the namespace.
        Returns the Checkpoint object.
        """
        saved, removed = self._checkpoint.save(
            checkpoint_name, dict(self.shell.user_ns), max_size_mb=None
        )

        for k, v in removed.items():
            from flowbook.util.output import log
            message = f"The object {k} (type {v}) cannot be checkpointed"
            log(message)
            self._display.display_icon_and_text("\u26a0\ufe0f", message)

        return self._checkpoint.saved[checkpoint_name]
