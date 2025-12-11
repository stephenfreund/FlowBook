"""
FerretSDCKernel - IPython kernel with Sequential Dataflow Consistency enforcement.

A simplified kernel focused on SDC. No profiling, no checkpoint magics.
SDC is always enabled.
"""

import time
import traceback
from typing import Any, Dict, Optional

from IPython.core.magic import Magics, line_magic, magics_class
from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.display_helpers import DisplayHelper
from data_ferret.kernel.tracking import TrackingDict
from data_ferret.server.message_broadcaster import get_broadcaster
from data_ferret.util.cell_index import index_to_alpha
from data_ferret.util.output import log, timer

from .models import SDCMetadata
from .sdc_enforcer import SDCEnforcer


@magics_class
class FerretSDCKernel(IPythonKernel, Magics):
    """
    IPython kernel with Sequential Dataflow Consistency enforcement.

    Features:
    - Variable access tracking (reads/writes per cell)
    - SDC Rule 3 enforcement (no backward mutations)
    - Staleness computation and reporting
    - Cell order management via magic command

    SDC is always enabled. No profiling or checkpoint magics.
    """

    implementation = "ferret_sdc_kernel"
    implementation_version = "0.1"
    banner = "Ferret SDC Kernel - Sequential Dataflow Consistency"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.shell is not None
        self.shell.register_magics(self)

        # Display helper
        self._display = DisplayHelper()

        # Current cell being executed
        self._cell_id: Optional[str] = None

        # Checkpointing (for pre/post comparison)
        self._checkpoint = Checkpoints(
            sanity_check=False,
            convert_dtypes=True,
            warn_classes=False,
        )

        # Tracking
        self._tracking = TrackingDict(self.shell.user_ns)

        # SDC enforcement
        self._sdc = SDCEnforcer(self._checkpoint)

        # # Flag to track if we've initialized tracking
        # self._tracking_initialized = False

    # =========================================================================
    # Magic Commands
    # =========================================================================

    @line_magic
    def notebook_structure(self, line: str) -> None:
        """
        Set the notebook cell order for SDC enforcement.

        Usage:
            %notebook_structure cell1 cell2 cell3 ...
        """
        cell_order = line.split()
        self._sdc.set_cell_order(cell_order)

    @line_magic
    def sdc_status(self, line: str) -> None:
        """Display current SDC state."""
        order = self._sdc.cell_order
        records = self._sdc.records

        status_lines = [
            f"Cell order: {order}",
            f"Executed cells: {list(records.keys())}",
            f"Execution counter: {self._sdc.seq_counter}",
        ]

        for cell_id, record in records.items():
            status_lines.append(
                f"  {cell_id}: reads={sorted(record.reads)}, "
                f"writes={sorted(record.writes)}, seq={record.execution_seq}"
            )

        self._display.display_icon_and_text(
            "info", "SDC Status", "\n".join(status_lines)
        )

    # =========================================================================
    # Stub magics for compatibility with FerretKernel notebooks
    # =========================================================================

    @line_magic
    def enable_scalene(self, line: str) -> None:
        """Stub for Scalene profiling (not supported in SDC kernel)."""
        pass

    @line_magic
    def disable_scalene(self, line: str) -> None:
        """Stub for Scalene profiling (not supported in SDC kernel)."""
        pass

    @line_magic
    def enable_global_tracking(self, line: str) -> None:
        """Stub for global tracking (not supported in SDC kernel)."""
        pass

    @line_magic
    def disable_global_tracking(self, line: str) -> None:
        """Stub for global tracking (not supported in SDC kernel)."""
        pass

    @line_magic
    def checkpoint(self, line: str) -> None:
        """Stub for checkpoint magic (not supported in SDC kernel)."""
        pass

    @line_magic
    def restore(self, line: str) -> None:
        """Stub for restore magic (not supported in SDC kernel)."""
        pass

    @line_magic
    def list_checkpoints(self, line: str) -> None:
        """Stub for list checkpoints magic (not supported in SDC kernel)."""
        pass

    # =========================================================================
    # Tracking Initialization
    # =========================================================================

    def _ensure_tracking_initialized(self) -> None:
        """
        Initialize variable tracking with full comprehension support.

        This patches the shell's run_code method to use our TrackingDict for
        BOTH globals and locals. This is necessary because:

        1. Python's exec(code, globals, locals) uses globals for LOAD_GLOBAL
        2. List comprehensions and functions use LOAD_GLOBAL for free variables
        3. If globals != our TrackingDict, comprehension reads aren't tracked

        By using TrackingDict for both, all variable access is tracked including
        reads inside list comprehensions and nested functions.
        """
        if not isinstance(self.shell.user_ns, TrackingDict):
            # Create TrackingDict wrapping user_global_ns
            tracking_dict = TrackingDict(self.shell.user_global_ns)
            self.shell.user_ns = tracking_dict

            # Patch run_code to use TrackingDict for both globals and locals
            self._patch_run_code(tracking_dict)

    def _patch_run_code(self, tracking_dict: TrackingDict) -> None:
        """
        Patch shell.run_code to use TrackingDict for both globals and locals.

        This enables tracking of variable reads inside list comprehensions
        and nested functions, which would otherwise bypass our TrackingDict.
        """
        shell = self.shell
        original_run_code = shell.run_code

        def patched_run_code(code_obj, result=None, *, async_=False):
            """
            Execute code using TrackingDict for both globals and locals.

            This ensures all variable access is tracked, including reads
            inside list comprehensions which use LOAD_GLOBAL.
            """
            # Temporarily replace both user_ns and inject tracking_dict
            # as the globals dict for exec. We do this by temporarily
            # swapping what user_global_ns returns.

            # Store original
            old_user_ns = shell.user_ns

            try:
                # Set both to tracking_dict so exec sees it as globals
                shell.user_ns = tracking_dict
                # user_global_ns is a property, but we shadow it in __dict__
                shell.__dict__['user_global_ns'] = tracking_dict

                # Call original run_code - it will use our tracking_dict
                return original_run_code(code_obj, result, async_=async_)

            finally:
                # Restore
                shell.user_ns = old_user_ns
                # Remove the shadow
                if 'user_global_ns' in shell.__dict__:
                    del shell.__dict__['user_global_ns']

        # Replace the method
        shell.run_code = patched_run_code

    # =========================================================================
    # Execution
    # =========================================================================

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
        Execute code with SDC tracking and enforcement.
        """
        try:
            # Ensure tracking is initialized (done lazily on first execution)
            self._ensure_tracking_initialized()

            # Extract cell context
            self._cell_id = self._extract_cell_id(cell_id, cell_meta)
            cell_alpha = self._get_cell_alpha()
            log(f"[sdc] Executing cell {cell_alpha}")

            # Update cell order if provided in metadata
            if cell_meta and "cell_order" in cell_meta:
                self._sdc.set_cell_order(cell_meta["cell_order"])

            # Check for notebook_structure magic (parse and remove if present)
            code = self._process_structure_magic(code)

            # Skip SDC for empty code or pure magic
            if not code.strip() or self._is_pure_magic(code):
                log(f"[sdc] Skipping SDC for empty/magic cell {cell_alpha}")
                return await self._execute_without_sdc(
                    code,
                    silent,
                    store_history,
                    user_expressions,
                    allow_stdin,
                    cell_meta,
                )

            # Take pre-execution snapshot
            user_ns = self.shell.user_ns
            with timer(
                key="sdc_pre_checkpoint",
                message=f"[sdc] Taking pre-checkpoint for {cell_alpha}",
            ):
                pre_checkpoint = self._take_checkpoint(f"_pre_{self._cell_id}")

            # Reset tracking for this execution
            if isinstance(user_ns, TrackingDict):
                user_ns.reset_tracking()

            # Execute with tracking
            with timer(key="sdc_execute", message=f"[sdc] Executing cell {cell_alpha}"):
                start_time = time.time()
                if isinstance(user_ns, TrackingDict):
                    with user_ns.track_execution():
                        result = await super().do_execute(
                            code,
                            silent,
                            store_history,
                            user_expressions,
                            allow_stdin,
                            cell_meta=cell_meta,
                            cell_id=self._cell_id,
                        )
                    tracking = user_ns.get_tracking_data()
                else:
                    result = await super().do_execute(
                        code,
                        silent,
                        store_history,
                        user_expressions,
                        allow_stdin,
                        cell_meta=cell_meta,
                        cell_id=self._cell_id,
                    )
                    tracking = None
                duration = time.time() - start_time

            # If execution had an error, restore pre-state and skip SDC checks
            if result.get("status") == "error":
                log(f"[sdc] Cell {cell_alpha} had execution error, restoring pre-state")
                self._sdc.checkpoints.restore(
                    f"_pre_{self._cell_id}", self.shell.user_ns
                )
                return result

            # Log tracking results
            if tracking:
                log(
                    f"[sdc] Cell {cell_alpha} tracking: reads={sorted(tracking.reads_before_writes)}, writes={sorted(tracking.writes)}"
                )

            with timer(
                key="sdc_post_checkpoint",
                message=f"[sdc] Taking post-checkpoint for {cell_alpha}",
            ):
                post_checkpoint = self._take_checkpoint(f"_post_{self._cell_id}")

            # Run SDC check if we have tracking data and cell_id
            sdc_result = None
            if tracking and self._cell_id:
                with timer(
                    key="sdc_check", message=f"[sdc] Running SDC check for {cell_alpha}"
                ):
                    sdc_result = self._sdc.check(
                        cell_id=self._cell_id,
                        pre_checkpoint=pre_checkpoint,
                        post_checkpoint=post_checkpoint,
                        tracking=tracking,
                    )
                log(f"[sdc] Check completed for cell {cell_alpha}")
                if sdc_result and sdc_result.violation:
                    log(f"[sdc] VIOLATION DETECTED: {sdc_result.violation.message}")

            # Display results (only if not silent, no error, and no SDC violation)
            # Skip display on violation since state will be rolled back
            has_violation = sdc_result and sdc_result.violation
            if not silent and result.get("status") != "error" and not has_violation:
                self._display_execution_result(duration, tracking, sdc_result)

            # Handle violation - report as error
            if sdc_result and sdc_result.violation:
                log(f"[sdc] Restoring checkpoint and sending error")
                # restore to pre-checkpoint
                self._sdc.checkpoints.restore(
                    f"_pre_{self._cell_id}", self.shell.user_ns
                )
                self._send_violation_error(sdc_result.violation)
                return self._make_error_result(sdc_result.violation)

            return result
        except Exception as e:
            log(f"[sdc] Error executing cell {self._cell_id}: {e}")
            log(traceback.format_exc())
            raise e

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_cell_alpha(self) -> str:
        """Get @A notation for current cell."""
        if self._cell_id is None:
            return "unknown"
        try:
            index = self._sdc.cell_order.index(self._cell_id)
            return index_to_alpha(index)
        except (ValueError, IndexError):
            return self._cell_id

    def _extract_cell_id(
        self, cell_id: Optional[str], cell_meta: Optional[dict]
    ) -> Optional[str]:
        """Extract cell ID from arguments or metadata."""
        if cell_id is not None:
            return cell_id
        if cell_meta is not None:
            return cell_meta.get("cell_id")
        return None

    def _process_structure_magic(self, code: str) -> str:
        """
        Process %notebook_structure magic if present at start of code.
        Removes the magic line and updates cell order.
        Returns remaining code.
        """
        lines = code.split("\n")
        if lines and lines[0].strip().startswith("%notebook_structure"):
            # Extract cell order from magic line
            magic_line = lines[0].strip()
            parts = magic_line.split()[1:]  # Skip the magic name
            if parts:
                self._sdc.set_cell_order(parts)
            return "\n".join(lines[1:])
        return code

    def _is_pure_magic(self, code: str) -> bool:
        """Check if code is only magic commands."""
        lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
        return all(line.startswith("%") or line.startswith("!") for line in lines)

    def _take_checkpoint(self, checkpoint_name: str) -> Checkpoint:
        """
        Take a snapshot of checkpointable variables before execution.

        Uses Checkpoints.save() to properly deep copy the namespace.
        This is critical for correct operation with TrackingDict.
        """
        # Convert TrackingDict to regular dict for checkpointing
        self._checkpoint.save(
            checkpoint_name, dict(self.shell.user_ns), max_size_mb=None
        )
        return self._checkpoint.saved[checkpoint_name]

    async def _execute_without_sdc(
        self,
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> dict:
        """Execute without SDC tracking (for magics, empty code)."""
        return await super().do_execute(
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
            cell_meta=cell_meta,
            cell_id=self._cell_id,
        )

    def _display_execution_result(
        self,
        duration: float,
        tracking,
        sdc_result,
    ) -> None:
        """Display execution timing and SDC metadata."""
        # Build metadata for display
        metadata = SDCMetadata(
            cell_id=self._cell_id or "",
            execution_seq=self._sdc.seq_counter,
            reads=list(tracking.reads_before_writes) if tracking else [],
            writes=list(tracking.writes) if tracking else [],
            changed_variables=sdc_result.changed_variables if sdc_result else [],
            stale_cells=sdc_result.stale_cells if sdc_result else [],
            violation=(
                sdc_result.violation.to_dict()
                if (sdc_result and sdc_result.violation)
                else None
            ),
            cell_order=self._sdc.cell_order,
        )

        # Build display text
        parts = [f"{duration:.2f}s"]
        if tracking:
            if tracking.reads_before_writes:
                reads_preview = list(tracking.reads_before_writes)[:3]
                parts.append(f"R:{','.join(reads_preview)}")
            if tracking.writes:
                writes_preview = list(tracking.writes)[:3]
                parts.append(f"W:{','.join(writes_preview)}")

        if sdc_result and sdc_result.stale_cells:
            parts.append(f"stale:{','.join(sdc_result.stale_cells)}")

        icon = "check" if not (sdc_result and sdc_result.violation) else "error"

        self._display.display_icon_and_text(
            icon,
            " | ".join(parts),
            metadata=metadata.to_display_metadata(),
        )

    def _send_violation_error(self, violation) -> None:
        """Send SDC violation as error via iopub."""
        # Get @A notation for cells
        try:
            mutating_idx = self._sdc.cell_order.index(violation.mutating_cell)
            mutating_alpha = index_to_alpha(mutating_idx)
        except (ValueError, IndexError):
            mutating_alpha = violation.mutating_cell

        try:
            affected_idx = self._sdc.cell_order.index(violation.affected_cell)
            affected_alpha = index_to_alpha(affected_idx)
        except (ValueError, IndexError):
            affected_alpha = violation.affected_cell

        self.send_response(
            self.iopub_socket,
            "error",
            {
                "ename": "SDCViolation",
                "evalue": violation.message,
                "traceback": [
                    "Sequential Dataflow Consistency Violation",
                    "",
                    f"Cell {mutating_alpha} modified variables that "
                    f"cell {affected_alpha} (earlier in notebook) reads.",
                    "",
                    f"Affected variables: {violation.variables}",
                    "",
                    "This breaks reproducibility. The earlier cell's behavior "
                    "depends on this cell having run first.",
                ],
            },
        )

    def _make_error_result(self, violation) -> dict:
        """Create error result dict for SDC violation."""
        return {
            "status": "error",
            "execution_count": self.execution_count,
            "ename": "SDCViolation",
            "evalue": violation.message,
            "traceback": [violation.message],
        }


# Entry point
if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretSDCKernel)
