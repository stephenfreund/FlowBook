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

from ipykernel.kernelapp import IPKernelApp

from flowbook.kernel_support.base_kernel import BaseFlowbookKernel

from flowbook.checkpoint_kernel.models import CheckpointMetadata


class CheckpointKernel(BaseFlowbookKernel):
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

        # Expose checkpoint object to user code for rerun trials
        self.shell.user_ns["_flowbook_checkpoint"] = self._checkpoint

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
        Execute code and measure timing.
        """
        self._ensure_fs_magics()
        self._ensure_vfs_namespace_patched()

        # For empty code or pure magic, still report timing (with 0 values)
        is_trivial = not code.strip() or self._is_pure_magic(code)

        # Measure cell execution time
        exec_start = time.perf_counter()
        result = await self._ipython_do_execute(
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
            _, uncopyable = self._take_checkpoint(f"post_{self._cell_id}")
            # Checkpoint kernel uses old behavior: remove uncopyable vars
            for k in uncopyable:
                if k in self.shell.user_ns:
                    del self.shell.user_ns[k]
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


# Entry point
if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=CheckpointKernel)
