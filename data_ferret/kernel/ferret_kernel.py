"""
FerretKernel - Enhanced IPython kernel with profiling, checkpoints, and tracking.

This module provides the core kernel implementation for DataFerret, extending
IPython's kernel with advanced features for notebook analysis and optimization.

Architecture Overview:
    FerretKernel extends IPythonKernel and Magics to provide:

    1. **Execution Profiling**: Integration with Scalene for CPU/memory profiling
       of cell execution. Profiles are captured and sent to the frontend.

    2. **Checkpointing**: Save and restore kernel state snapshots. Useful for
       comparing variable values before/after execution or rolling back changes.

    3. **Variable Tracking**: Track which variables are read/written during
       cell execution using TrackingDict. This enables dynamic dependency analysis.

    4. **Column-Level Tracking**: For pandas DataFrames, track which columns are
       accessed, enabling fine-grained dependency analysis.

    5. **Monotonicity Enforcement**: Optionally enforce that cells don't modify
       variables they read before writing (prevents non-deterministic behavior).

    6. **Timeout Handling**: Configurable per-cell timeouts with graceful cleanup.

Key Components:
    - FerretKernel: Main kernel class (this file)
    - TrackingDict: Variable access tracking (tracking.py)
    - MonotonicityEnforcer: Monotonicity constraint checking (monotonicity.py)
    - Checkpoints: State snapshot management (checkpoint.py)
    - ScaleneRunner: Profiler integration (scalene_runner.py)
    - KernelCommandHandlers: Handler implementations (kernel_command_handlers.py)

Communication:
    The kernel communicates with the frontend via:
    - Standard Jupyter messaging (execute_request, execute_reply)
    - Custom comm channels:
        - "debug_command": Debugger commands
        - "kernel_command": Checkpoint/toggle commands

Configuration:
    Key settings (as instance attributes):
    - _use_scalene: Enable/disable Scalene profiling
    - _use_global_tracking: Enable/disable variable tracking
    - _enforce_monotone_updates: Enable/disable monotonicity checking
    - _force_checkpoints: Auto-checkpoint before each cell

Usage:
    The kernel is launched via the standard Jupyter kernel mechanism:

        python -m data_ferret.kernel.ferret_kernel

    Or via the installed kernel spec "ferret".
"""

import re
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from IPython.core.magic import Magics, line_cell_magic, magics_class
from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.deepcopyable import check_deepcopyable
from data_ferret.kernel.display_helpers import DisplayHelper
from data_ferret.kernel.ferret_pdb import FerretPdb
from data_ferret.kernel.json_utils import make_json_safe
from data_ferret.kernel.kernel_command_handlers import KernelCommandHandlers
from data_ferret.kernel.kernel_commands import FinalMessage
from data_ferret.kernel.models import ExecutionContext, ExecutionMetadata, ExecutionProfile, TrackingData
from data_ferret.kernel.monotonicity import MonotonicityEnforcer
from data_ferret.kernel.scalene_runner import ScaleneRunner
from data_ferret.kernel.timeout_handler import CellTimeoutHandler
from data_ferret.kernel.tracking import TrackingDict
from data_ferret.util.output import log, output


# =============================================================================
# Checkpoint Command Configuration
# =============================================================================

# Maps checkpoint subcommand -> (arg_count, request_class_name, arg_names)
CHECKPOINT_COMMANDS = {
    "save": (1, "CheckpointSaveRequest", ["name"]),
    "restore": (1, "CheckpointRestoreRequest", ["name"]),
    "delete": (1, "CheckpointDeleteRequest", ["name"]),
    "list": (0, "CheckpointListRequest", []),
    "compare": (2, "CheckpointCompareRequest", ["name1", "name2"]),
    "clear": (0, "CheckpointClearRequest", []),
}


@magics_class
class FerretKernel(IPythonKernel, Magics):
    """
    Enhanced IPython kernel with profiling, checkpoints, and debugging support.

    This kernel extends IPython with features for notebook analysis:
    - Scalene profiling for CPU/memory analysis
    - State checkpointing and diffing
    - Variable access tracking
    - Monotonicity enforcement
    - Cell timeout handling

    Attributes:
        _cell_id: Current cell being executed
        _checkpoint: Checkpoint manager
        _use_scalene: Whether Scalene profiling is enabled
        _use_global_tracking: Whether variable tracking is enabled
        _enforce_monotone_updates: Whether monotonicity is enforced
        _force_checkpoints: Whether to auto-checkpoint before each cell
    """

    # =========================================================================
    # Configuration Constants
    # =========================================================================

    _default_cell_timeout = 30 * 60  # 30 minutes
    _post_kb_grace = 1.0  # Grace period after KeyboardInterrupt
    _kill_timeout = 3.0  # Time to wait before force kill
    _verbose = False
    _max_passes = 2  # Max timeout handler passes

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.shell is not None, "shell is not set"
        self.shell.register_magics(self)

        # Initialize helpers
        self._display = DisplayHelper()
        self.command_handlers = KernelCommandHandlers(self)
        self.pdb = FerretPdb()

        # Execution state
        self._cell_id: Optional[str] = None
        self._cell_timeout = self._default_cell_timeout
        self._executed_cell_ids: Dict[int, str] = {}

        # Feature flags
        self._use_scalene = True
        self._force_checkpoints = False
        self._use_global_tracking = False
        self._enforce_monotone_updates = False

        # Managers
        self._checkpoint = Checkpoints()
        self._scalene = ScaleneRunner(self.shell, self._executed_cell_ids)

        # Tracking state
        self._original_run_code = None  # Store original for unpatch

        # Register comm targets
        self.comm_manager.register_target("debug_command", self._debug_command_comm_open)
        self.comm_manager.register_target("kernel_command", self._kernel_command_comm_open)

    # =========================================================================
    # Display Helpers
    # =========================================================================

    def display_cell_id(self) -> None:
        """Display the current cell ID."""
        self._display.display_cell_id(self._cell_id)

    def display_icon_and_text(
        self,
        icon: str,
        text: str,
        contents: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Display an icon with text, optionally with expandable contents."""
        self._display.display_icon_and_text(icon, text, contents, metadata)

    def diff_checkpoints(self, old: Checkpoint, new: Checkpoint) -> None:
        """Display the diff between two checkpoints."""
        self._display.display_checkpoint_diff(old, new)

    # =========================================================================
    # Variable Tracking
    # =========================================================================

    def _patch_run_code(self, tracking_dict: TrackingDict) -> None:
        """
        Patch shell.run_code to use TrackingDict for both globals and locals.

        This enables tracking of variable reads inside list comprehensions
        and nested functions, which would otherwise bypass our TrackingDict.
        """
        if self._original_run_code is not None:
            # Already patched
            return

        shell = self.shell
        self._original_run_code = shell.run_code

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
                return self._original_run_code(code_obj, result, async_=async_)

            finally:
                # Restore
                shell.user_ns = old_user_ns
                # Remove the shadow
                if 'user_global_ns' in shell.__dict__:
                    del shell.__dict__['user_global_ns']

        # Replace the method
        shell.run_code = patched_run_code

    def _unpatch_run_code(self) -> None:
        """Restore original run_code method."""
        if self._original_run_code is not None:
            self.shell.run_code = self._original_run_code
            self._original_run_code = None

    # =========================================================================
    # Magic Commands - Feature Toggles
    # =========================================================================

    @line_cell_magic
    def scalene(self, line: str, cell: str = "") -> None:
        """
        Control Scalene profiling.

        Usage:
            %scalene        - Enable profiling (default)
            %scalene on     - Enable profiling
            %scalene off    - Disable profiling
            %scalene ?      - Show current status
        """
        arg = line.strip().lower()

        if arg == "?":
            status = "on" if self._use_scalene else "off"
            self.display_icon_and_text("\U0001F50D", f"Scalene profiling: {status}")
            return

        if not arg or arg in ("on", "true", "1", "enable"):
            self._use_scalene = True
            self.display_icon_and_text("\U0001F50D", "Scalene profiling enabled")
        elif arg in ("off", "false", "0", "disable"):
            self._use_scalene = False
            self.display_icon_and_text("\U0001F50D", "Scalene profiling disabled")
        else:
            self.display_icon_and_text("\u274C", f"Invalid: '{arg}'. Use 'on', 'off', or '?'")

    @line_cell_magic
    def force_checkpoints(self, line: str, cell: str = "") -> None:
        """
        Control force checkpoints mode.

        Usage:
            %force_checkpoints        - Enable force checkpoints (default)
            %force_checkpoints on     - Enable force checkpoints
            %force_checkpoints off    - Disable force checkpoints
            %force_checkpoints ?      - Show current status
        """
        arg = line.strip().lower()

        if arg == "?":
            status = "on" if self._force_checkpoints else "off"
            self.display_icon_and_text("\u2705", f"Force checkpoints: {status}")
            return

        if not arg or arg in ("on", "true", "1", "enable"):
            self._force_checkpoints = True
            self.display_icon_and_text("\u2705", "Force checkpoints enabled")
        elif arg in ("off", "false", "0", "disable"):
            self._force_checkpoints = False
            self.display_icon_and_text("\u2705", "Force checkpoints disabled")
        else:
            self.display_icon_and_text("\u274C", f"Invalid: '{arg}'. Use 'on', 'off', or '?'")

    @line_cell_magic
    def tracking(self, line: str, cell: str = "") -> None:
        """
        Control global variable tracking.

        Usage:
            %tracking        - Enable tracking (default)
            %tracking on     - Enable tracking
            %tracking off    - Disable tracking
            %tracking ?      - Show current status
        """
        arg = line.strip().lower()

        if arg == "?":
            status = "on" if self._use_global_tracking else "off"
            self.display_icon_and_text("\U0001F4CA", f"Global tracking: {status}")
            return

        if not arg or arg in ("on", "true", "1", "enable"):
            self._use_global_tracking = True
            # Initialize tracking if not already done
            if not isinstance(self.shell.user_ns, TrackingDict):
                tracking_dict = TrackingDict(self.shell.user_global_ns)
                self.shell.user_ns = tracking_dict
                self._patch_run_code(tracking_dict)
            self.display_icon_and_text("\U0001F4CA", "Global tracking enabled")
        elif arg in ("off", "false", "0", "disable"):
            self._use_global_tracking = False
            self.display_icon_and_text("\U0001F4CA", "Global tracking disabled")
        else:
            self.display_icon_and_text("\u274C", f"Invalid: '{arg}'. Use 'on', 'off', or '?'")

    @line_cell_magic
    def monotone(self, line: str, cell: str = "") -> None:
        """
        Control monotone updates enforcement.

        Usage:
            %monotone        - Enable enforcement (default)
            %monotone on     - Enable enforcement
            %monotone off    - Disable enforcement
            %monotone ?      - Show current status
        """
        arg = line.strip().lower()

        if arg == "?":
            status = "on" if self._enforce_monotone_updates else "off"
            self.display_icon_and_text("\u2705", f"Monotone enforcement: {status}")
            return

        if not arg or arg in ("on", "true", "1", "enable"):
            log("[monotone] Enabling monotone updates enforcement")
            if not self._use_global_tracking:
                log("[monotone] Auto-enabling global tracking (required for RBW detection)")
                self.tracking("on", "")
            self._enforce_monotone_updates = True
            self.display_icon_and_text("\u2705", "Monotone enforcement enabled")
        elif arg in ("off", "false", "0", "disable"):
            log("[monotone] Disabling monotone updates enforcement")
            self._enforce_monotone_updates = False
            self.display_icon_and_text("\u2705", "Monotone enforcement disabled")
        else:
            self.display_icon_and_text("\u274C", f"Invalid: '{arg}'. Use 'on', 'off', or '?'")

    @line_cell_magic
    def structural_tracking(self, line: str, cell: str = "") -> None:
        """
        Set structural tracking mode for DataFrame/Series attribute monitoring.

        Structural tracking detects when code accesses attributes that reveal
        DataFrame/Series structure (like df.columns, df.shape, len(df)).
        When structural tracking is enabled and these attributes are read,
        subsequent changes to the structure (adding columns, changing row count)
        are either warned about or treated as violations.

        Usage:
            %structural_tracking           - Show current mode
            %structural_tracking off       - Disable structural tracking
            %structural_tracking warn      - Track and warn only (default)
            %structural_tracking enforce   - Track and treat changes as violations
        """
        from contextlib import nullcontext

        from data_ferret.kernel.structural_tracking import StructuralTrackingMode

        # Suspend tracking during magic execution to avoid recording infrastructure reads
        user_ns = self.shell.user_ns
        if isinstance(user_ns, TrackingDict) and hasattr(user_ns, 'suspended'):
            ctx = user_ns.suspended()
        else:
            ctx = nullcontext()

        with ctx:
            mode_str = line.strip().lower()

            if not mode_str:
                # Show current mode
                current_mode = "off"
                if isinstance(user_ns, TrackingDict):
                    current_mode = user_ns.structural_tracking_mode.value
                elif hasattr(self, '_structural_mode'):
                    current_mode = self._structural_mode.value
                self.display_icon_and_text(
                    "\U0001F50D",
                    f"Structural tracking mode: {current_mode}"
                )
                return

            try:
                mode = StructuralTrackingMode(mode_str)
            except ValueError:
                self.display_icon_and_text(
                    "\u274C",
                    f"Invalid mode: {mode_str}. Use 'off', 'warn', or 'enforce'"
                )
                return

            # Store mode for use in MonotonicityEnforcer
            self._structural_mode = mode

            # Update TrackingDict if it exists
            if isinstance(user_ns, TrackingDict):
                user_ns.set_structural_tracking_mode(mode_str)

            self.display_icon_and_text(
                "\u2705",
                f"Structural tracking mode set to: {mode.value}"
            )

    # =========================================================================
    # Magic Commands - Checkpoints
    # =========================================================================

    @line_cell_magic
    def checkpoint(self, line: str, cell: str = "") -> None:
        """
        Checkpoint management magic command.

        Usage:
            %checkpoint save <name>     - Save current state
            %checkpoint restore <name>  - Restore saved state
            %checkpoint delete <name>   - Delete a checkpoint
            %checkpoint list            - List all checkpoints
            %checkpoint compare <a> <b> - Compare two checkpoints
            %checkpoint clear           - Delete all checkpoints
        """
        args = line.split()
        if not args:
            self.display_icon_and_text("\u274C", "Usage: checkpoint <command> [args]")
            return

        try:
            self._handle_checkpoint_command(args)
        except Exception as e:
            self.display_icon_and_text("\u274C", f"Error: {e}")

    def _handle_checkpoint_command(self, args: list) -> None:
        """
        Route checkpoint subcommand to appropriate handler.

        Uses CHECKPOINT_COMMANDS table for dispatch, reducing boilerplate.
        """
        from data_ferret.kernel import kernel_commands as cmd_module

        cmd = args[0]
        if cmd not in CHECKPOINT_COMMANDS:
            self.display_icon_and_text("\u274C", f"Unknown checkpoint command: {cmd}")
            return

        arg_count, request_class_name, arg_names = CHECKPOINT_COMMANDS[cmd]

        # Validate argument count
        if len(args) - 1 != arg_count:
            usage = f"checkpoint {cmd} " + " ".join(f"<{n}>" for n in arg_names)
            self.display_icon_and_text("\u274C", f"Usage: {usage}")
            return

        # Build request
        request_class = getattr(cmd_module, request_class_name)
        kwargs = {name: args[i + 1] for i, name in enumerate(arg_names)}
        req = request_class(**kwargs)

        # Get and invoke handler
        handler = self.command_handlers.get_handler(f"checkpoint_{cmd}")
        response = handler(req)

        # Display result (command-specific formatting)
        self._display_checkpoint_response(cmd, response, args)

    def _display_checkpoint_response(self, cmd: str, response, args: list) -> None:
        """Format and display checkpoint command response."""
        if cmd == "save":
            self._display_checkpoint_save_result(response)
        elif cmd == "list":
            self.display_icon_and_text(
                "\u2705", f"Checkpoints: {', '.join(sorted(response.checkpoints))}"
            )
        elif cmd == "compare":
            old = self._checkpoint.get(args[1])
            new = self._checkpoint.get(args[2])
            self.diff_checkpoints(old, new)
        else:
            self.display_icon_and_text("\u2705", response.message)

    def _display_checkpoint_save_result(self, response) -> None:
        """Format and display checkpoint save result."""
        added_text = "\n".join([f"* {k}: {v}" for k, v in response.saved.items()])
        removed_text = "\n".join([f"* {k}: {v}" for k, v in response.removed.items()])
        metadata = {
            "ferret": {
                "added": added_text,
                "removed": removed_text,
                "duration": response.duration,
            }
        }
        if response.removed:
            self.display_icon_and_text(
                "\u2705",
                f"{response.duration:.2f}s [removed: {', '.join(sorted(response.removed.keys()))}]",
                contents=added_text,
                metadata=metadata,
            )
        else:
            self.display_icon_and_text(
                "\u2705",
                f"Checkpoint saved in {response.duration:.2f}s",
                contents=added_text,
                metadata=metadata,
            )

    # =========================================================================
    # Comm Handlers
    # =========================================================================

    def _debug_command_comm_open(self, comm, open_msg) -> None:
        """Handle debug command comm requests."""
        cmd = open_msg["content"]["data"]["cmd"]
        try:
            result = self.pdb.capture_onecmd(cmd)
            comm.send({"type": "final", "ok": True, "result": result})
        except Exception as e:
            comm.send({"type": "final", "ok": False, "error": str(e)})

    def _kernel_command_comm_open(self, comm, open_msg) -> None:
        """Handle kernel command comm requests."""
        try:
            data = open_msg["content"]["data"]
            command = data.get("command")
            if not command:
                raise ValueError("Missing 'command' field in request")

            handler = self.command_handlers.get_handler(command)

            # Dynamically build request class name
            from data_ferret.kernel import kernel_commands as cmd_module

            parts = command.split("_")
            request_class_name = "".join(p.capitalize() for p in parts) + "Request"
            request_class = getattr(cmd_module, request_class_name)

            request = request_class(**data)
            response = handler(request)

            final_msg = FinalMessage(ok=True, response=response.model_dump())
            comm.send(final_msg.model_dump())

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            final_msg = FinalMessage(ok=False, error=error_msg)
            comm.send(final_msg.model_dump())

    # =========================================================================
    # Execution - Main Entry Point
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
        Execute code with profiling, tracking, and optional monotonicity enforcement.

        This is the main execution entry point, called for each cell. It orchestrates:
        1. Context preparation (cell ID, timeout parsing, tracking reset)
        2. Optional pre-execution checkpoint (force_checkpoints mode)
        3. Optional monotonicity pre-state save
        4. Code execution with profiling
        5. Result display
        6. Optional monotonicity check and enforcement
        7. Timeout handling

        Args:
            code: The code to execute
            silent: If True, suppress output
            store_history: If True, store in execution history
            user_expressions: Expressions to evaluate after execution
            allow_stdin: If True, allow stdin requests
            cell_meta: Cell metadata from frontend
            cell_id: Cell identifier

        Returns:
            Execution result dict with status, execution_count, etc.
        """
        # Prepare execution context
        context = self._prepare_execution(code, cell_id, cell_meta)
        self.display_cell_id()

        # Pre-execution checkpoints
        if self._force_checkpoints:
            self.checkpoint(f"save pre_{context.cell_id}")

        # Setup monotonicity enforcer if enabled
        monotone: Optional[MonotonicityEnforcer] = None
        if self._enforce_monotone_updates:
            monotone = MonotonicityEnforcer(self._checkpoint, self.shell.user_ns)
            monotone.save_pre_state(context.cell_id)

        # Setup timeout handler
        timeout_handler = CellTimeoutHandler(
            timeout=context.timeout,
            post_kb_grace=self._post_kb_grace,
            kill_timeout=self._kill_timeout,
            verbose=self._verbose,
            max_passes=self._max_passes,
        )
        timeout_handler.start()
        normal_exit = False

        try:
            start_time = time.time()
            result, tracking = await self._execute_with_profiling(
                context, silent, store_history, user_expressions, allow_stdin, cell_meta
            )
            normal_exit = True
            duration = time.time() - start_time

            # Display results (skip on error)
            if result.get("status") != "error":
                self._display_execution_result(context, duration, tracking)

                # Show checkpoint diff if force_checkpoints enabled
                if self._force_checkpoints:
                    self._show_checkpoint_diff(context)

                # Check monotonicity if enabled
                if monotone and tracking:
                    violation = monotone.check_and_enforce(tracking, context.cell_id)
                    if violation:
                        self._send_monotonicity_error(violation)
                        return violation.to_error_result(self.execution_count)

            return result

        except KeyboardInterrupt:
            return self._handle_timeout_error(context.timeout)

        finally:
            timeout_handler.cancel()
            if not normal_exit:
                await timeout_handler.cleanup_on_error()

    # =========================================================================
    # Execution - Helpers
    # =========================================================================

    def _prepare_execution(
        self,
        code: str,
        cell_id: Optional[str],
        cell_meta: Optional[dict],
    ) -> ExecutionContext:
        """
        Prepare execution context from request parameters.

        Extracts cell ID, parses timeout directive, resets tracking state.

        Returns:
            ExecutionContext with parsed parameters
        """
        # Extract cell ID
        if cell_id is not None:
            self._cell_id = cell_id
        elif cell_meta is not None:
            self._cell_id = cell_meta.get("cell_id", None)
        else:
            self._cell_id = None

        # Parse timeout from code (highest priority)
        parsed_code, code_timeout = self._parse_timeout_from_code(code)

        # Determine timeout: code directive > cell_meta > default
        if code_timeout != self._default_cell_timeout:
            # Code had explicit # timeout directive
            timeout = code_timeout
        elif cell_meta and "timeout" in cell_meta:
            # Use timeout from cell_metadata (from command)
            timeout = float(cell_meta["timeout"])
        else:
            # Fall back to default
            timeout = self._default_cell_timeout

        # Reset tracking for new execution
        if isinstance(self.shell.user_ns, TrackingDict):
            self.shell.user_ns.reset_tracking()

        return ExecutionContext(
            cell_id=self._cell_id,
            code=parsed_code,
            timeout=timeout,
            original_code=code,
        )

    def _parse_timeout_from_code(self, code: str) -> Tuple[str, float]:
        """Parse timeout directive from code if present."""
        match = re.match(r"# timeout (\d+)\n", code)
        if match:
            timeout = int(match.group(1))
            code = code.replace(match.group(0), "", 1)
        else:
            timeout = self._default_cell_timeout
        return code, timeout

    async def _execute_with_profiling(
        self,
        context: ExecutionContext,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> Tuple[dict, Optional[TrackingData]]:
        """
        Execute code with optional Scalene profiling and tracking.

        Returns:
            Tuple of (execution result, tracking data or None)
        """
        user_ns = self.shell.user_ns
        should_profile = self._use_scalene and context.should_profile

        # Capture pre-execution types (before column tracking to avoid pollution)
        if should_profile:
            self._pre_types = {
                k: str(v)
                for k, v in self._checkpoint.type_models(user_ns).items()
            }

        # Execute with tracking if enabled
        tracking_data: Optional[TrackingData] = None

        if self._use_global_tracking and isinstance(user_ns, TrackingDict):
            with user_ns.track_execution():
                result, self._profile_contents = await self._do_execute_code(
                    context, should_profile, silent, store_history,
                    user_expressions, allow_stdin, cell_meta
                )
            tracking_data = user_ns.get_tracking_data()
        else:
            result, self._profile_contents = await self._do_execute_code(
                context, should_profile, silent, store_history,
                user_expressions, allow_stdin, cell_meta
            )

        # Post-profiling cleanup (after column tracking stopped)
        if should_profile:
            self._remove_non_deepcopyable_objects()
            self._post_types = {
                k: str(v)
                for k, v in self._checkpoint.type_models(user_ns).items()
            }

        return result, tracking_data

    async def _do_execute_code(
        self,
        context: ExecutionContext,
        should_profile: bool,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> Tuple[dict, Optional[str]]:
        """
        Execute code with or without Scalene profiling.

        Returns:
            Tuple of (execution result, profile contents or None)
        """
        if should_profile:
            result, contents = await self._scalene.run(
                context.code, context.cell_id, store_history
            )
            # Send error to iopub if Scalene detected one
            if result.get("status") == "error":
                self.send_response(
                    self.iopub_socket,
                    "error",
                    {
                        "ename": result.get("ename", "UnknownError"),
                        "evalue": result.get("evalue", ""),
                        "traceback": result.get("traceback", []),
                    },
                )
            return result, contents
        else:
            result = await super().do_execute(
                context.code,
                silent,
                store_history,
                user_expressions,
                allow_stdin,
                cell_meta=cell_meta,
                cell_id=context.cell_id,
            )
            # Handle shell magic returning None
            if context.has_shell_magics and result is None:
                result = {"status": "ok", "execution_count": self.execution_count}
            return result, None

    def _remove_non_deepcopyable_objects(self) -> None:
        """Remove objects that can't be deep copied from user namespace."""
        from data_ferret.kernel.checkpoint import filter_user_namespace

        user_ns = filter_user_namespace(self.shell.user_ns)

        # Collect non-copyable variables with their types and reasons
        non_copyable = []
        for k, v in user_ns.items():
            reason = check_deepcopyable(v)
            if reason:
                non_copyable.append((k, type(v).__name__, reason))

        for k, _, _ in non_copyable:
            del self.shell.user_ns[k]

        if non_copyable:
            for k, typ, reason in non_copyable:
                message = f"The object {k}: {typ} cannot be passed between cells: {reason}"
                log(message)
                self.display_icon_and_text(
                    "\u26A0\uFE0F",
                    message
                )

    def _display_execution_result(
        self,
        context: ExecutionContext,
        duration: float,
        tracking: Optional[TrackingData],
    ) -> None:
        """Display execution timing and profile results."""
        # Build metadata
        profile = ExecutionProfile(
            duration=duration,
            profile=getattr(self, "_profile_contents", None) or "",
            env=getattr(self, "_pre_types", {}),
            env_after=getattr(self, "_post_types", {}),
        )
        metadata = ExecutionMetadata(
            profile=profile,
            dynamic_dependencies=tracking,
        )

        # Display appropriate output
        if self._profile_contents:
            self.display_icon_and_text(
                "\U0001F50D",
                f"{duration:0.2f}s",
                contents=self._profile_contents,
                metadata=metadata.to_display_metadata(),
            )
        elif not context.has_cell_magics:
            self.display_icon_and_text(
                "\u23F1\uFE0F",
                f"{duration:0.2f}s",
                metadata=metadata.to_display_metadata(),
            )

    def _show_checkpoint_diff(self, context: ExecutionContext) -> None:
        """Show diff between pre-checkpoint and current state."""
        user_ns = self._checkpoint.checkpointable_vars(self.shell.user_ns)
        user_ns = self._checkpoint.checkpointable_values(user_ns)
        old = self._checkpoint.get(f"pre_{context.cell_id}")
        start_time = time.time()
        new = Checkpoint("_tmp", user_ns, {})
        self.diff_checkpoints(old, new)
        duration = time.time() - start_time
        self._display.display_icon_and_text("\u23F1\uFE0F", f"checkpoint diff: {duration:0.2f}s")

    def _send_monotonicity_error(self, violation) -> None:
        """Send monotonicity error via iopub socket."""
        self.send_response(
            self.iopub_socket,
            "error",
            {
                "ename": "MonotonicityError",
                "evalue": violation.error_summary,
                "traceback": [violation.diff_details],
            },
        )

    def _handle_timeout_error(self, timeout: float) -> dict:
        """Create timeout error result."""
        timeout_msg = f"Cell execution timed out after {timeout} seconds"
        self.send_response(
            self.iopub_socket,
            "error",
            {"ename": "TimeoutError", "evalue": timeout_msg, "traceback": [timeout_msg]},
        )
        return {
            "status": "error",
            "execution_count": self.execution_count,
            "ename": "TimeoutError",
            "evalue": timeout_msg,
            "traceback": [timeout_msg],
        }

    # =========================================================================
    # JSON Utilities
    # =========================================================================

    def _make_json_safe(self, obj: Any) -> Any:
        """Convert an object to a JSON-safe format."""
        return make_json_safe(obj)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def do_shutdown(self, restart: bool) -> dict:
        """Handle kernel shutdown/restart."""
        # Explicitly flush timings before shutdown - atexit may not run
        # if the kernel is killed by jupyter_client after timeout
        output._print_timings()
        return super().do_shutdown(restart)


if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretKernel)
