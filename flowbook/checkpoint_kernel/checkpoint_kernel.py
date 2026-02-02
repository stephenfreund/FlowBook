"""
CheckpointKernel - IPython kernel that measures cell execution and checkpoint time.

This is a minimal kernel that:
1. Measures cell execution time
2. Takes a checkpoint after each execution
3. Measures checkpoint (commit) time
4. Reports timing via display_data metadata

No reproducibility tracking, no variable tracking - just execution and checkpoint timing.
"""

import time
from typing import Optional

from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from flowbook.kernel_support.checkpoint import Checkpoints
from flowbook.kernel_support.display_helpers import DisplayHelper

from flowbook.checkpoint_kernel.models import CheckpointMetadata


class CheckpointKernel(IPythonKernel):
    """
    IPython kernel with checkpoint timing measurement.

    Features:
    - Measures cell execution time
    - Measures checkpoint (state save) time
    - Reports timing via display_data metadata
    """

    implementation = "checkpoint_kernel"
    implementation_version = "0.1"
    banner = "FlowBook Checkpoint Kernel - Execution and checkpoint timing"

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

        # Expose checkpoint object to user code for rerun trials
        self.shell.user_ns["_flowbook_checkpoint"] = self._checkpoint

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
        Execute code and measure timing.
        """
        # Extract cell context
        self._cell_id = self._extract_cell_id(cell_id, cell_meta)

        # For empty code or pure magic, still report timing (with 0 values)
        is_trivial = not code.strip() or self._is_pure_magic(code)

        # Measure cell execution time
        exec_start = time.perf_counter()
        result = await super().do_execute(
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
            cell_meta=cell_meta,
            cell_id=self._cell_id,
        )
        exec_end = time.perf_counter()

        # For trivial cells, report 0 timings and skip checkpoint
        if is_trivial:
            cell_runtime_s = 0.0
            commit_time_s = 0.0
        else:
            cell_runtime_s = exec_end - exec_start

            # Measure checkpoint time
            commit_start = time.perf_counter()
            self._take_checkpoint(f"post_{self._cell_id}")
            commit_end = time.perf_counter()
            commit_time_s = commit_end - commit_start

        # Check for execution errors
        error_msg = None
        if result.get("status") == "error":
            error_msg = result.get("evalue", "Unknown error")

        # Send timing metadata
        if not silent:
            metadata = CheckpointMetadata(
                cell_id=self._cell_id or "",
                execution_count=self.execution_count,
                cell_runtime_s=cell_runtime_s,
                commit_time_s=commit_time_s,
                error=error_msg,
            )

            # Display timing info
            self._display.display_icon_and_text(
                "ok" if error_msg is None else "error",
                f"Run: {cell_runtime_s*1000:.1f}ms | Commit: {commit_time_s*1000:.1f}ms",
                metadata=metadata.to_display_metadata(),
            )

        return result

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
        # Force checkpoint on sentinel (for rerun trials)
        if "__flowbook_force_checkpoint__" in code:
            return False
        lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
        return all(
            line.startswith("%") or line.startswith("!") or line.startswith("#")
            for line in lines
        )

    def _take_checkpoint(self, checkpoint_name: str) -> None:
        """
        Take a snapshot of the namespace.

        Uses Checkpoints.save() to properly deep copy the namespace.
        """
        self._checkpoint.save(
            checkpoint_name, dict(self.shell.user_ns), max_size_mb=None
        )


# Entry point
if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=CheckpointKernel)
