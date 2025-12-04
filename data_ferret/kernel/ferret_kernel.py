"""FerretKernel - Enhanced IPython kernel with profiling and checkpoint support."""

import re
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from comm import create_comm
from IPython.core.magic import Magics, line_cell_magic, magics_class
from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints, filter_user_namespace
from data_ferret.kernel.deepcopyable import is_deepcopyable
from data_ferret.kernel.display_helpers import DisplayHelper
from data_ferret.kernel.ferret_pdb import FerretPdb
from data_ferret.kernel.json_utils import make_json_safe
from data_ferret.kernel.kernel_command_handlers import KernelCommandHandlers
from data_ferret.kernel.kernel_commands import FinalMessage
from data_ferret.kernel.scalene_runner import ScaleneRunner
from data_ferret.kernel.timeout_handler import CellTimeoutHandler


@magics_class
class FerretKernel(IPythonKernel, Magics):
    """Enhanced IPython kernel with profiling, checkpoints, and debugging support."""

    # Configuration
    _default_cell_timeout = 30 * 60  # 30 minutes
    _post_kb_grace = 1.0
    _kill_timeout = 3.0
    _verbose = False
    _max_passes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.shell is not None, "shell is not set"
        self.shell.register_magics(self)

        # Initialize helpers
        self._display = DisplayHelper()
        self.command_handlers = KernelCommandHandlers(self)
        self.pdb = FerretPdb()

        # State
        self._cell_id: Optional[str] = None
        self._cell_timeout = self._default_cell_timeout
        self._executed_cell_ids: Dict[int, str] = {}
        self._checkpoint = Checkpoints()
        self._use_scalene = True
        self._force_checkpoints = False

        # Initialize Scalene runner
        self._scalene = ScaleneRunner(self.shell, self._executed_cell_ids)

        # Register comm targets
        self.comm_manager.register_target("debug_command", self._debug_command_comm_open)
        self.comm_manager.register_target("kernel_command", self._kernel_command_comm_open)

    # =========================================================================
    # Display Delegation
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
    # Magic Commands
    # =========================================================================

    @line_cell_magic
    def enable_scalene(self, line: str, cell: str = "") -> None:
        """Enable Scalene profiling."""
        from data_ferret.kernel.kernel_commands import EnableScaleneRequest
        req = EnableScaleneRequest()
        response = self.command_handlers.handle_enable_scalene(req)
        self.display_icon_and_text("🔍", response.message)

    @line_cell_magic
    def disable_scalene(self, line: str, cell: str = "") -> None:
        """Disable Scalene profiling."""
        from data_ferret.kernel.kernel_commands import DisableScaleneRequest
        req = DisableScaleneRequest()
        response = self.command_handlers.handle_disable_scalene(req)
        self.display_icon_and_text("🔍", response.message)

    @line_cell_magic
    def force_checkpoints(self, line: str, cell: str = "") -> None:
        """Enable or disable force checkpoints mode."""
        from data_ferret.kernel.kernel_commands import ForceCheckpointsRequest
        enabled = line.strip().lower() not in ["false", "0", "disable", "off"]
        req = ForceCheckpointsRequest(enabled=enabled)
        response = self.command_handlers.handle_force_checkpoints(req)
        self.display_icon_and_text("✅", response.message)

    @line_cell_magic
    def checkpoint(self, line: str, cell: str = "") -> None:
        """
        Checkpoint cell magic.

        Usage:
            %checkpoint save <name>
            %checkpoint restore <name>
            %checkpoint delete <name>
            %checkpoint list
            %checkpoint compare <name1> <name2>
            %checkpoint clear
        """
        from data_ferret.kernel.kernel_commands import (
            CheckpointClearRequest,
            CheckpointCompareRequest,
            CheckpointDeleteRequest,
            CheckpointListRequest,
            CheckpointRestoreRequest,
            CheckpointSaveRequest,
        )

        args = line.split()
        if not args:
            self.display_icon_and_text("❌", "Usage: checkpoint <command> [args]")
            return

        try:
            self._handle_checkpoint_command(args)
        except Exception as e:
            self.display_icon_and_text("❌", f"Error: {e}")

    def _handle_checkpoint_command(self, args: list) -> None:
        """Route checkpoint subcommand to appropriate handler."""
        from data_ferret.kernel.kernel_commands import (
            CheckpointClearRequest,
            CheckpointCompareRequest,
            CheckpointDeleteRequest,
            CheckpointListRequest,
            CheckpointRestoreRequest,
            CheckpointSaveRequest,
        )

        cmd = args[0]

        if cmd == "save":
            if len(args) != 2:
                self.display_icon_and_text("❌", "Usage: checkpoint save <name>")
                return
            req = CheckpointSaveRequest(name=args[1])
            response = self.command_handlers.handle_checkpoint_save(req)
            self._display_checkpoint_save_result(response)

        elif cmd == "restore":
            if len(args) != 2:
                self.display_icon_and_text("❌", "Usage: checkpoint restore <name>")
                return
            req = CheckpointRestoreRequest(name=args[1])
            response = self.command_handlers.handle_checkpoint_restore(req)
            self.display_icon_and_text("✅", response.message)

        elif cmd == "delete":
            if len(args) != 2:
                self.display_icon_and_text("❌", "Usage: checkpoint delete <name>")
                return
            req = CheckpointDeleteRequest(name=args[1])
            response = self.command_handlers.handle_checkpoint_delete(req)
            self.display_icon_and_text("✅", response.message)

        elif cmd == "list":
            req = CheckpointListRequest()
            response = self.command_handlers.handle_checkpoint_list(req)
            self.display_icon_and_text(
                "✅", f"Checkpoints: {', '.join(sorted(response.checkpoints))}"
            )

        elif cmd == "compare":
            if len(args) != 3:
                self.display_icon_and_text("❌", "Usage: checkpoint compare <name1> <name2>")
                return
            req = CheckpointCompareRequest(name1=args[1], name2=args[2])
            self.command_handlers.handle_checkpoint_compare(req)
            old = self._checkpoint.get(args[1])
            new = self._checkpoint.get(args[2])
            self.diff_checkpoints(old, new)

        elif cmd == "clear":
            req = CheckpointClearRequest()
            response = self.command_handlers.handle_checkpoint_clear(req)
            self.display_icon_and_text("✅", response.message)

        else:
            self.display_icon_and_text("❌", f"Unknown checkpoint command: {cmd}")

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
                "✅",
                f"{response.duration:.2f}s [removed: {', '.join(sorted(response.removed.keys()))}]",
                contents=added_text,
                metadata=metadata,
            )
        else:
            self.display_icon_and_text(
                "✅",
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
        """Execute code with optional profiling and checkpointing."""
        self._extract_cell_id(cell_id, cell_meta)
        code, timeout = self._parse_timeout_from_code(code)
        self.display_cell_id()

        force_checkpoints = self._force_checkpoints

        if force_checkpoints:
            self.checkpoint(f"save pre_{self._cell_id}")

        timeout_handler = CellTimeoutHandler(
            timeout=timeout,
            post_kb_grace=self._post_kb_grace,
            kill_timeout=self._kill_timeout,
            verbose=self._verbose,
            max_passes=self._max_passes,
        )
        timeout_handler.start()
        normal_exit = False

        try:
            start_time = time.time()
            result, profile_contents, pre_types, post_types = await self._execute_with_profiling(
                code, silent, store_history, user_expressions, allow_stdin, cell_meta
            )
            normal_exit = True
            end_time = time.time()

            # Only display execution result if not an error
            if result.get('status') != 'error':
                self._display_execution_result(
                    end_time - start_time, profile_contents, pre_types, post_types, code
                )

            if force_checkpoints:
                self._show_checkpoint_diff()

            # print(f"[FerretKernel] do_execute returning status: {result.get('status')}", file=sys.__stderr__)
            return result

        except KeyboardInterrupt:
            # Cell execution timed out
            timeout_msg = f"Cell execution timed out after {timeout} seconds"

            # Send error to client via iopub socket
            self.send_response(self.iopub_socket, 'error', {
                'ename': 'TimeoutError',
                'evalue': timeout_msg,
                'traceback': [timeout_msg]
            })

            # Return error result
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'TimeoutError',
                'evalue': timeout_msg,
                'traceback': [timeout_msg]
            }

        finally:
            timeout_handler.cancel()
            if not normal_exit:
                await timeout_handler.cleanup_on_error()

    def _extract_cell_id(self, cell_id: Optional[str], cell_meta: Optional[dict]) -> None:
        """Extract and store the cell ID from execution parameters."""
        if cell_id is not None:
            self._cell_id = cell_id
        elif cell_meta is not None:
            self._cell_id = cell_meta.get("cell_id", None)
        else:
            self._cell_id = None

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
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> Tuple[dict, Optional[str], Optional[dict], Optional[dict]]:
        """Execute code with optional Scalene profiling."""
        has_cell_magics = code.startswith("%") or "\n%" in code
        has_shell_magics = code.startswith("!") or "\n!" in code
        should_profile = self._use_scalene and self._cell_id is not None and not has_cell_magics and not has_shell_magics

        if should_profile:
            pre_types = {k: str(v) for k, v in self._checkpoint.type_models(self.shell.user_ns).items()}
            result, contents = await self._scalene.run(code, self._cell_id, store_history)
            # print(f"[FerretKernel] scalene result status: {result.get('status')}", file=sys.__stderr__)

            # If scalene detected an error, send error message on iopub
            if result.get('status') == 'error':
                self.send_response(self.iopub_socket, 'error', {
                    'ename': result.get('ename', 'UnknownError'),
                    'evalue': result.get('evalue', ''),
                    'traceback': result.get('traceback', [])
                })

            self._remove_non_deepcopyable_objects()
            post_types = {k: str(v) for k, v in self._checkpoint.type_models(self.shell.user_ns).items()}
            return result, contents, pre_types, post_types
        else:
            result = await super().do_execute(
                code, silent, store_history, user_expressions, allow_stdin,
                cell_meta=cell_meta, cell_id=self._cell_id
            )
            return result, None, None, None

    def _remove_non_deepcopyable_objects(self) -> None:
        """Remove objects that can't be deep copied from user namespace."""
        user_ns = filter_user_namespace(self.shell.user_ns)
        non_copyable = [k for k, v in user_ns.items() if not is_deepcopyable(v)]
        for k in non_copyable:
            del self.shell.user_ns[k]
        if non_copyable:
            self.display_icon_and_text(
                "⚠️",
                f"The following objects cannot be passed between cells: {', '.join(non_copyable)}"
            )

    def _display_execution_result(
        self,
        duration: float,
        profile_contents: Optional[str],
        pre_types: Optional[dict],
        post_types: Optional[dict],
        code: str,
    ) -> None:
        """Display execution timing and profile results."""
        metadata = {
            'profile': {
                'duration': duration,
                'profile': profile_contents if profile_contents is not None else "",
                'env': pre_types if pre_types is not None else {},
                'env_after': post_types if post_types is not None else {},
            }
        }
        has_cell_magics = code.startswith("%") or "\n%" in code

        if profile_contents is not None:
            self.display_icon_and_text("🔍", f"{duration:0.2f}s", contents=profile_contents, metadata=metadata)
        elif not has_cell_magics:
            self.display_icon_and_text("⏱️", f"{duration:0.2f}s", metadata=metadata)

    def _show_checkpoint_diff(self) -> None:
        """Show diff between checkpoint and current state."""
        user_ns = self._checkpoint.checkpointable_vars(self.shell.user_ns)
        user_ns = self._checkpoint.checkpointable_values(user_ns)
        old = self._checkpoint.get(f"pre_{self._cell_id}")
        start_time = time.time()
        new = Checkpoint(f"_tmp", user_ns, {})
        self.diff_checkpoints(old, new)
        end_time = time.time()
        self._display.display_icon_and_text("⏱️", f"checkpoint diff: {end_time - start_time:0.2f}s")

    # =========================================================================
    # JSON Utilities (exposed for compatibility)
    # =========================================================================

    def _make_json_safe(self, obj: Any) -> Any:
        """Convert an object to a JSON-safe format."""
        return make_json_safe(obj)


if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretKernel)
