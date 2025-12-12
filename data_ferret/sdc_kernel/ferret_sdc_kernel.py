"""
FerretSDCKernel - IPython kernel with Sequential Dataflow Consistency enforcement.

================================================================================
SEQUENTIAL DATAFLOW CONSISTENCY (SDC) - ARCHITECTURE OVERVIEW
================================================================================

SDC ensures notebook reproducibility by enforcing dataflow rules that prevent
hidden state dependencies. When SDC is enforced, running cells top-to-bottom
always produces the same result.

THE THREE SDC RULES
-------------------

Rule 1: Reproducibility Invariant (Structural)
    A notebook is reproducible if running all cells in document order from a
    fresh kernel produces identical results every time. This is the goal.

Rule 2: Staleness Propagation Rule (Computed)
    A cell becomes "stale" when any variable it reads has changed since it
    last executed. Stale cells need re-execution to reflect current state.

    Example:
        Cell A: x = 1
        Cell B: y = x + 1  # B reads x
        Cell A: x = 2      # Re-run A -> B is now stale (x changed)

Rule 3: No Backward Mutation Constraint (Enforced)
    A cell may NOT modify a variable that an earlier cell (in document order)
    reads. This prevents "hidden" dependencies where earlier cells depend on
    later cells having run first.

    Example (VIOLATION):
        Cell A: y = x + 1  # A reads x
        Cell B: x = 10     # B modifies x -> VIOLATION! A depends on B.

    This is the key rule that makes notebooks reproducible. Without it, the
    order you run cells affects results in unpredictable ways.

COLUMN-LEVEL TRACKING
---------------------

For DataFrames, SDC tracks at the column level for precision:

    Cell A: total = df['price'].sum()     # Reads df.price
    Cell B: df['quantity'] = df['quantity'] * 2  # Modifies df.quantity

This is NOT a violation because different columns are involved. Without
column tracking, any DataFrame modification would trigger false violations.

DATA FLOW
---------

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         FerretSDCKernel                             │
    │                                                                     │
    │  1. do_execute() receives code + cell_id + cell_order               │
    │                         │                                           │
    │                         ▼                                           │
    │  2. Take PRE-checkpoint (snapshot namespace before execution)       │
    │                         │                                           │
    │                         ▼                                           │
    │  3. Execute code with TrackingDict (records reads/writes)           │
    │                         │                                           │
    │                         ▼                                           │
    │  4. Take POST-checkpoint (snapshot namespace after execution)       │
    │                         │                                           │
    │                         ▼                                           │
    │  5. SDCEnforcer.check() compares checkpoints + tracking:            │
    │     - Detect backward mutations (Rule 3 violations)                 │
    │     - Compute which cells are now stale (Rule 2)                    │
    │     - Extract column-level changes for DataFrames                   │
    │                         │                                           │
    │                         ▼                                           │
    │  6. On violation: restore PRE-checkpoint (rollback) + error         │
    │     On success: display metadata (reads, writes, stale cells)       │
    └─────────────────────────────────────────────────────────────────────┘

KEY COMPONENTS
--------------

- TrackingDict: Wraps user namespace to record variable reads/writes
- Checkpoints: Deep-copies namespace for before/after comparison
- SDCEnforcer: Implements Rule 2 (staleness) and Rule 3 (backward mutation)
- SDCMetadata: Data sent to frontend for UI display (stale cell highlighting)

MAGIC COMMANDS
--------------

%notebook_structure cell1 cell2 ...  - Set cell order (usually auto-injected)
%sdc_status                          - Display current SDC state
%sdc_stale                           - Show which cells are currently stale
%continue_after_violation [true|false] - Control violation handling

================================================================================
"""

import traceback
from typing import Any, Dict, List, Optional

from IPython.core.magic import Magics, line_magic, magics_class
from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.display_helpers import DisplayHelper
from data_ferret.kernel.tracking import TrackingDict
from data_ferret.util.cell_index import index_to_alpha
from data_ferret.util.output import error, timer

from .models import SDCMetadata
from .sdc_enforcer import SDCEnforcer, PRE_CHECKPOINT_PREFIX, POST_CHECKPOINT_PREFIX


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

        # Continue after violation flag (default: stop on violation)
        self._continue_after_violation: bool = False

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
            "ℹ️", "SDC Status", "\n".join(status_lines)
        )

    @line_magic
    def continue_after_violation(self, line: str) -> None:
        """
        Control whether execution continues after SDC violations.

        Usage:
            %continue_after_violation        # Enable (continue after violations)
            %continue_after_violation true   # Enable
            %continue_after_violation false  # Disable (default: stop on violation)
        """
        arg = line.strip().lower()
        if arg == "false":
            self._continue_after_violation = False
        elif arg == "true" or arg == "":
            self._continue_after_violation = True
        else:
            self._display.display_icon_and_text(
                "⚠️", f"Invalid argument: '{arg}'. Use 'true' or 'false'."
            )
            return
        state = "enabled" if self._continue_after_violation else "disabled"
        self._display.display_icon_and_text("ℹ️", f"Continue after violation: {state}")

    @line_magic
    def sdc_stale(self, line: str) -> None:
        """
        Show which cells are currently stale.

        Usage:
            %sdc_stale
        """
        stale_cells = self._sdc.get_stale_cells()
        if not stale_cells:
            self._display.display_icon_and_text("✓", "No stale cells")
            return

        # Convert to @A notation
        stale_refs = []
        for cell_id in stale_cells:
            try:
                idx = self._sdc.cell_order.index(cell_id)
                stale_refs.append(index_to_alpha(idx))
            except (ValueError, IndexError):
                stale_refs.append(cell_id)

        self._display.display_icon_and_text(
            "⚠️", f"Stale cells: {', '.join(stale_refs)}"
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
                shell.__dict__["user_global_ns"] = tracking_dict

                # Call original run_code - it will use our tracking_dict
                return original_run_code(code_obj, result, async_=async_)

            finally:
                # Restore
                shell.user_ns = old_user_ns
                # Remove the shadow
                if "user_global_ns" in shell.__dict__:
                    del shell.__dict__["user_global_ns"]

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

            # Update cell order if provided in metadata
            if cell_meta and "cell_order" in cell_meta:
                self._sdc.set_cell_order(cell_meta["cell_order"])

            # Check for notebook_structure magic (parse and remove if present)
            code = self._process_structure_magic(code)

            # Skip SDC for empty code or pure magic
            if not code.strip() or self._is_pure_magic(code):
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
            with timer(key="sdc_pre_checkpoint") as pre_timer:
                pre_checkpoint = self._take_checkpoint(f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}")

            # Reset tracking for this execution
            if isinstance(user_ns, TrackingDict):
                user_ns.reset_tracking()

            # Execute with tracking
            with timer(key="sdc_execute") as run_timer:
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

            # If execution had an error, restore pre-state and skip SDC checks
            if result.get("status") == "error":
                self._sdc.checkpoints.restore(
                    f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}", self.shell.user_ns
                )
                return result

            # Take post-execution snapshot
            with timer(key="sdc_post_checkpoint") as post_timer:
                post_checkpoint = self._take_checkpoint(f"{POST_CHECKPOINT_PREFIX}{self._cell_id}")

            # Run SDC check if we have tracking data and cell_id
            if tracking and self._cell_id:
                with timer(key="sdc_check") as check_timer:
                    sdc_result = self._sdc.check(
                        cell_id=self._cell_id,
                        pre_checkpoint=pre_checkpoint,
                        post_checkpoint=post_checkpoint,
                        tracking=tracking,
                        continue_on_violation=self._continue_after_violation,
                    )

                # Handle violation
                has_violation = sdc_result and sdc_result.violation
                if has_violation:
                    # Log truncation issues to terminal
                    if sdc_result.violation.truncation_details:
                        error(f"SDC truncation: {sdc_result.violation.message}")
                        self._send_truncation_details(sdc_result.violation.truncation_details)

                    if self._continue_after_violation:
                        error(f"SDC violation (continuing): {sdc_result.violation.message}")
                        self._send_violation_warning(sdc_result.violation)
                    else:
                        # Log violation to terminal, restore checkpoint, return error
                        error(f"SDC violation: {sdc_result.violation.message}")
                        self._sdc.checkpoints.restore(
                            f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}", self.shell.user_ns
                        )
                        self._send_violation_error(sdc_result.violation)
                        return self._make_error_result(sdc_result.violation)

                # Display results (skip if silent, error, or violation with rollback)
                skip_display = has_violation and not self._continue_after_violation
                if not silent and result.get("status") != "error" and not skip_display:
                    state_ms = pre_timer.duration() + post_timer.duration()
                    self._display_execution_result(
                        run_timer.duration(),
                        state_ms,
                        check_timer.duration(),
                        tracking,
                        sdc_result,
                    )

            return result
        except Exception as e:
            error(f"SDC error in cell {self._cell_id}: {e}\n{traceback.format_exc()}")
            raise

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

    def _format_var_with_columns(
        self,
        var: str,
        column_reads: dict,
        column_writes: dict,
    ) -> str:
        """
        Format variable with inline column info: df[price,qty].

        Args:
            var: Variable name
            column_reads: Dict mapping var names to sets/lists of read columns
            column_writes: Dict mapping var names to sets/lists of written columns

        Returns:
            Formatted string like "df[price,qty]" or "x" if no columns
        """
        # Handle both sets and lists
        read_cols = column_reads.get(var, [])
        write_cols = column_writes.get(var, [])
        all_cols = set(read_cols) | set(write_cols)

        if all_cols:
            cols_list = sorted(all_cols)[:3]  # Show first 3
            cols_str = ",".join(cols_list)
            if len(all_cols) > 3:
                cols_str += f",+{len(all_cols) - 3}"
            return f"{var}[{cols_str}]"
        return var

    def _display_execution_result(
        self,
        run_duration: float,
        state_duration: float,
        check_duration: float,
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
            column_reads=(
                {k: list(v) for k, v in tracking.column_reads_before_writes.items()}
                if tracking
                else {}
            ),
            column_writes=(
                {k: list(v) for k, v in tracking.column_writes.items()}
                if tracking
                else {}
            ),
            column_changed=sdc_result.column_changed if sdc_result else {},
            run_duration_ms=run_duration,
            state_duration_ms=state_duration,
            check_duration_ms=check_duration,
        )

        # Build display text
        parts = [
            f"Run: {run_duration:.0f} ms",
            f"State: {state_duration:.0f} ms",
            f"Check: {check_duration:.0f} ms",
        ]
        if tracking:
            if tracking.reads_before_writes:
                reads_preview = [
                    self._format_var_with_columns(v, metadata.column_reads, {})
                    for v in list(tracking.reads_before_writes)[:3]
                ]
                parts.append(f"Reads: {','.join(reads_preview)}")
            # Show writes if there are variable-level writes OR column-level writes
            write_vars = set(tracking.writes) | set(metadata.column_writes.keys())
            if write_vars:
                writes_preview = [
                    self._format_var_with_columns(v, {}, metadata.column_writes)
                    for v in list(write_vars)[:3]
                ]
                parts.append(f"Writes: {','.join(writes_preview)}")

        if sdc_result and sdc_result.stale_cells:
            # Convert cell IDs to @A references for display
            stale_refs = []
            for cell_id in sdc_result.stale_cells:
                try:
                    idx = self._sdc.cell_order.index(cell_id)
                    stale_refs.append(index_to_alpha(idx))
                except (ValueError, IndexError):
                    stale_refs.append(cell_id)  # Fallback to ID if not in order
            parts.append(f"Stale: {','.join(stale_refs)}")

        icon = "✓" if not (sdc_result and sdc_result.violation) else "✗"

        self._display.display_icon_and_text(
            icon,
            " | ".join(parts),
            metadata=metadata.to_display_metadata(),
        )

    def _send_violation_error(self, violation) -> None:
        """Send SDC violation as error via iopub."""
        mutating_alpha, affected_alpha = self._get_violation_alphas(violation)

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

    def _send_violation_warning(self, violation) -> None:
        """Send SDC violation as warning via iopub (when continue_after_violation is enabled)."""
        mutating_alpha, affected_alpha = self._get_violation_alphas(violation)

        warning_text = (
            f"⚠️ SDC Violation (continuing): Cell {mutating_alpha} modified "
            f"{violation.variables} which cell {affected_alpha} reads.\n"
        )
        self.send_response(
            self.iopub_socket,
            "stream",
            {"name": "stderr", "text": warning_text},
        )

    def _send_truncation_details(self, truncation_details: str) -> None:
        """Send truncation details to stderr for user visibility."""
        self.send_response(
            self.iopub_socket,
            "stream",
            {"name": "stderr", "text": f"\n{truncation_details}\n"},
        )

    def _get_violation_alphas(self, violation) -> tuple:
        """Get @A notation for violation cells."""
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

        return mutating_alpha, affected_alpha

    def _make_error_result(self, violation) -> dict:
        """Create error result dict for SDC violation."""
        return {
            "status": "error",
            "execution_count": self.execution_count,
            "ename": "SDCViolation",
            "evalue": violation.message,
            "traceback": [violation.message],
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def do_shutdown(self, restart: bool) -> dict:
        """Handle kernel shutdown/restart."""
        if restart:
            # Clear SDC state on restart for clean slate
            self._sdc.reset()
        return super().do_shutdown(restart)


# Entry point
if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretSDCKernel)
