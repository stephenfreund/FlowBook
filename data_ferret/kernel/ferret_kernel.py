# debug_kernel.py
import ast
import copy
import json
from pathlib import Path
import pprint
import re
import threading
import types
from typing import Dict, List, Set, Tuple
import traceback
from typing import Any
from IPython.display import HTML, display, Markdown
from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp
from comm import create_comm
from data_ferret.kernel.checkpoint import Checkpoints, checkpoint_diff
from data_ferret.kernel.equality import user_ns_diff
from data_ferret.kernel.ferret_pdb import FerretPdb
from data_ferret.kernel.types import (
    TestCodeResult, TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash,
    DiffResult, ExecutionError
)
from data_ferret.kernel.kernel_commands import (
    KernelCommandRequest,
    ProgressMessage,
    FinalMessage,
)
from data_ferret.kernel.kernel_command_handlers import KernelCommandHandlers
import io
import sys
import time

# kernel_helpers.py
import asyncio, os, psutil
import threading, time, _thread, os, psutil
from ipykernel.ipkernel import IPythonKernel

from data_ferret.util.output import timer
from IPython.core.magic import Magics, cell_magic, line_cell_magic, magics_class
from data_ferret.kernel.extended_types import get_type_model
from data_ferret.kernel.checkpoint import Checkpoint
from data_ferret.util.ferret_metadata import FerretMetadata, ProfileData


async def stop_loky_and_all_children(timeout=3.0, verbose=False, max_passes=2):
    """
    1) Ask loky/joblib to shut down cleanly (cancel futures, wait for exit).
    2) Force-reset the global reusable executor so the next cell gets a fresh one.
    3) Kill *all* remaining child processes.
    """
    me = psutil.Process(os.getpid())

    async def _shutdown_and_reset_loky():
        try:
            # Import the module that holds the singleton
            from joblib.externals.loky import reusable_executor
        except Exception:
            print("Stopping loky and all children: no reusable executor found")
            return

        def _do():
            try:
                ex = reusable_executor.get_reusable_executor()
                if ex._executor_manager_thread is not None:
                    print("Killing loky workers")
                    ex._executor_manager_thread.kill_workers(
                        "executor shutting down in kernel"
                    )
                # cooperative shutdown; cancel queued tasks
                try:
                    print("Shutting down loky workers")
                    ex.shutdown(wait=True, kill_workers=True)
                except TypeError:
                    # older loky without cancel_futures
                    print("Waiting for loky workers to shut down")
                    ex.shutdown(wait=True)
                time.sleep(0.2)  # let resource_tracker unregister
            except Exception:
                pass
            # HARD RESET: drop the singleton so next use creates a brand-new executor
            try:
                reusable_executor._executor = None
            except Exception:
                pass
            try:
                # some versions also keep the args; clear them to avoid reuse mismatch
                reusable_executor._executor_args = None
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _do)

    # 1 & 2: clean shutdown + singleton reset
    if "joblib" in globals():
        await _shutdown_and_reset_loky()

    # 3) Reap any leftover children
    for _ in range(max_passes):
        try:
            kids = me.children(recursive=True)
        except Exception:
            kids = []
        if not kids:
            break

        if verbose:
            try:
                print("Terminating:", [(p.pid, p.name()) for p in kids])
            except Exception:
                print("Terminating:", [p.pid for p in kids])

        for p in kids:
            try:
                p.terminate()
            except Exception:
                pass

        _, alive = psutil.wait_procs(kids, timeout=timeout)

        for p in alive:
            if verbose:
                try:
                    print("Killing stubborn:", p.pid, p.name())
                except Exception:
                    print("Killing stubborn:", p.pid)
            try:
                p.kill()
            except Exception:
                pass

        psutil.wait_procs(alive, timeout=timeout)


@magics_class
class FerretKernel(IPythonKernel, Magics):
    _post_kb_grace = 1.0  # seconds to wait after interrupt before we escalate
    _verbose = False
    _passes = 2
    _kill_timeout = 3.0

    _div_style = "padding-left: 3em; font-size: 0.8em; background-color: #f0f0f8; margin-bottom: 0em;"

    def display_cell_id(self) -> None:
        display(
            Markdown(
                f"<div style='{self._div_style}'>"
                f"<b>Cell {self._cell_id}</b>"
                f"</div>"
            )
        )

    def display_icon_and_text(
        self,
        icon: str,
        text: str,
        contents: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        if contents is None:
            display(
                Markdown(f"<div style='{self._div_style}'>" f"{icon} {text}" f"</div>"),
                metadata=metadata,
            )
        else:
            display(
                Markdown(
                    f"<div style='{self._div_style}'>"
                    f"<details style='display: inline-block; text-align: left;'>"
                    f"<summary>{icon} {text}</summary>\n\n"
                    f"<pre style='margin: 0;'><code>{contents}</code></pre>\n\n"
                    f"</details>"
                    f"</div>"
                ),
                metadata=metadata,
            )

    @line_cell_magic
    def enable_scalene(self, line: str, cell: str = "") -> None:
        """Enable Scalene profiling using shared handler."""
        from data_ferret.kernel.kernel_commands import EnableScaleneRequest

        req = EnableScaleneRequest()
        response = self.command_handlers.handle_enable_scalene(req)
        self.display_icon_and_text("🔍", response.message)

    @line_cell_magic
    def disable_scalene(self, line: str, cell: str = "") -> None:
        """Disable Scalene profiling using shared handler."""
        from data_ferret.kernel.kernel_commands import DisableScaleneRequest

        req = DisableScaleneRequest()
        response = self.command_handlers.handle_disable_scalene(req)
        self.display_icon_and_text("🔍", response.message)

    @line_cell_magic
    def force_checkpoints(self, line: str, cell: str = "") -> None:
        """Enable force checkpoints mode using shared handler."""
        from data_ferret.kernel.kernel_commands import ForceCheckpointsRequest

        # Parse line for enable/disable (default: enable)
        enabled = True
        if line.strip().lower() in ["false", "0", "disable", "off"]:
            enabled = False

        req = ForceCheckpointsRequest(enabled=enabled)
        response = self.command_handlers.handle_force_checkpoints(req)
        self.display_icon_and_text("✅", response.message)

    def diff_checkpoints(self, old: Checkpoint, new: Checkpoint) -> None:
        diffs = checkpoint_diff(old, new)
        contents = pprint.pformat(diffs, indent=2)
        if diffs:
            self.display_icon_and_text(
                "↔️", f"Changed: {', '.join(sorted(diffs.keys()))}", contents=contents
            )
        else:
            self.display_icon_and_text("↔️", "No changes")

    @line_cell_magic
    def checkpoint(self, line: str, cell: str = "") -> None:
        """
        Checkpoint cell magic.

        Uses shared handler implementations for consistent behavior
        between cell magics and comm channel.

        Usage:
            %checkpoint save <name>
            %checkpoint restore <name>
            %checkpoint delete <name>
            %checkpoint list
            %checkpoint compare <name1> <name2>
            %checkpoint clear
        """
        from data_ferret.kernel.kernel_commands import (
            CheckpointSaveRequest,
            CheckpointRestoreRequest,
            CheckpointDeleteRequest,
            CheckpointListRequest,
            CheckpointCompareRequest,
            CheckpointClearRequest,
        )

        assert self.shell is not None, "shell is not set"
        args = line.split()

        if not args:
            self.display_icon_and_text("❌", "Usage: checkpoint <command> [args]")
            return

        try:
            if args[0] == "save":
                if len(args) != 2:
                    self.display_icon_and_text("❌", "Usage: checkpoint save <name>")
                    return

                req = CheckpointSaveRequest(name=args[1])
                response = self.command_handlers.handle_checkpoint_save(req)

                # Format saved/removed variables for display
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
                        f"{response.duration:.2f}s",
                        contents=added_text,
                        metadata=metadata,
                    )

            elif args[0] == "restore":
                if len(args) != 2:
                    self.display_icon_and_text("❌", "Usage: checkpoint restore <name>")
                    return

                req = CheckpointRestoreRequest(name=args[1])
                response = self.command_handlers.handle_checkpoint_restore(req)
                self.display_icon_and_text("✅", response.message)

            elif args[0] == "delete":
                if len(args) != 2:
                    self.display_icon_and_text("❌", "Usage: checkpoint delete <name>")
                    return

                req = CheckpointDeleteRequest(name=args[1])
                response = self.command_handlers.handle_checkpoint_delete(req)
                self.display_icon_and_text("✅", response.message)

            elif args[0] == "list":
                req = CheckpointListRequest()
                response = self.command_handlers.handle_checkpoint_list(req)
                self.display_icon_and_text(
                    "✅",
                    f"Checkpoints: {', '.join(sorted(response.checkpoints))}",
                )

            elif args[0] == "compare":
                if len(args) != 3:
                    self.display_icon_and_text(
                        "❌", "Usage: checkpoint compare <name1> <name2>"
                    )
                    return

                req = CheckpointCompareRequest(name1=args[1], name2=args[2])
                response = self.command_handlers.handle_checkpoint_compare(req)

                # Use existing diff display logic
                old = self._checkpoint.get(args[1])
                new = self._checkpoint.get(args[2])
                self.diff_checkpoints(old, new)

            elif args[0] == "clear":
                req = CheckpointClearRequest()
                response = self.command_handlers.handle_checkpoint_clear(req)
                self.display_icon_and_text("✅", response.message)

            else:
                self.display_icon_and_text("❌", f"Unknown checkpoint command: {args[0]}")

        except Exception as e:
            self.display_icon_and_text("❌", f"Error: {e}")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # any exception → our handler
        assert self.shell is not None, "shell is not set"
        self.shell.register_magics(self)

        # Initialize command handlers
        self.command_handlers = KernelCommandHandlers(self)

        # Register comm targets
        # Debug commands - separate channel for debugger operations
        self.comm_manager.register_target(
            "debug_command", self._debug_command_comm_open
        )
        # Test code - legacy comm, kept for backward compatibility
        self.comm_manager.register_target(
            "test_code", self._test_code_comm_open
        )
        # Kernel commands - new unified command channel
        self.comm_manager.register_target(
            "kernel_command", self._kernel_command_comm_open
        )

        self.pdb = FerretPdb()
        self._default_cell_timeout = 30 * 60
        self._cell_timeout = self._default_cell_timeout
        self._use_scalene = True

        self._cell_id = None
        self._executed_cell_ids = {}
        self._checkpoint = Checkpoints()
        self._force_checkpoints = False

        self.shell.set_custom_exc((Exception,), self._custom_exc_handler)

    def _custom_exc_handler(self, *args, **kwargs):
        try:
            _, etype, evalue, tb = args[:4]
            tb_offset = kwargs.get("tb_offset", None)
        except Exception:
            # should never happen!
            return None

        self.pdb.reset()
        frame = tb.tb_frame
        stack, idx = self.pdb.get_stack(frame, tb)
        self.pdb.stack = stack
        self.pdb.curindex = idx

        # move to the last user frame -- lame...
        self._do_cmd("up")
        self._do_cmd("down")

        traceback = self.pdb.enriched_stack_trace()
        globals = self.pdb.enriched_globals()

        # Build our debug payload
        data = {
            "etype": etype.__name__,
            "evalue": str(evalue),
            "traceback": traceback.rstrip(),
            "globals": globals.rstrip(),
        }

        # Send it over a Comm so the client can pick it up
        try:
            comm = create_comm(target_name="debug_event")
            comm.send(data)
        except Exception:
            # Don’t let our debug handler crash the kernel
            pass

        assert self.shell is not None, "shell is not set"
        stb = self.shell.InteractiveTB.structured_traceback(
            etype, evalue, tb, tb_offset=tb_offset
        )
        print("\n".join(stb), file=sys.__stderr__)

        return None

    def _do_cmd(self, cmd):
        return self.pdb.capture_onecmd(cmd)

    def _debug_command_comm_open(self, comm, open_msg):
        cmd = open_msg["content"]["data"]["cmd"]
        try:
            result = self._do_cmd(cmd)
            comm.send({"type": "final", "ok": True, "result": result})
        except Exception as e:
            comm.send({"type": "final", "ok": False, "error": str(e)})

    def test_code(self, comm, original_code: str, modified_code: str, output_variables: Set[str] | None = None) -> Dict[str, Any]:
        """
        Test the code and return the result, sending progress messages via comm.

        Returns a TestCodeResult (union type) that can be:
        - TestCodeSuccess: Both codes executed successfully
        - TestCodeOriginalCrash: Original code crashed
        - TestCodeModifiedCrash: Modified code crashed (original succeeded)
        """
        comm.send({"type": "progress", "message": "Saving original environment"})
        self.checkpoint(f"save original_environment")

        # Execute original code with crash handling
        comm.send({"type": "progress", "message": "Executing original code"})
        start_time = time.time()
        try:
            result = self.shell.run_cell(original_code)
            original_duration = time.time() - start_time

            # Check if execution had an error
            if result.error_in_exec is not None:
                raise result.error_in_exec

        except Exception as e:
            original_duration = time.time() - start_time
            comm.send({"type": "progress", "message": f"Original code crashed: {type(e).__name__}"})

            # Create error result for original code crash
            crash_result = TestCodeOriginalCrash(
                error=ExecutionError(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    code_snippet=original_code
                ),
                original_duration=original_duration
            )
            return crash_result.model_dump()

        comm.send({"type": "progress", "message": "Saving original result"})
        self.checkpoint(f"save original_result")

        comm.send({"type": "progress", "message": "Restoring original environment"})
        self.checkpoint(f"restore original_environment")

        # Execute modified code with crash handling
        comm.send({"type": "progress", "message": "Executing modified code"})
        start_time = time.time()
        try:
            result = self.shell.run_cell(modified_code)
            modified_duration = time.time() - start_time

            # Check if execution had an error
            if result.error_in_exec is not None:
                raise result.error_in_exec

        except Exception as e:
            modified_duration = time.time() - start_time
            comm.send({"type": "progress", "message": f"Modified code crashed: {type(e).__name__}"})

            # Create error result for modified code crash
            crash_result = TestCodeModifiedCrash(
                error=ExecutionError(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    code_snippet=modified_code
                ),
                original_duration=original_duration,
                modified_duration=modified_duration
            )
            return crash_result.model_dump()

        comm.send({"type": "progress", "message": "Saving modified result"})
        self.checkpoint(f"save modified_result")

        # Both codes succeeded - perform diff
        comm.send({"type": "progress", "message": "Diffing original and modified environments"})
        diff_result = checkpoint_diff(
            self._checkpoint.get(f"original_result"),
            self._checkpoint.get(f"modified_result"),
            keys_to_include=output_variables
        )

        # Calculate speedup (avoid division by zero)
        speedup = original_duration / modified_duration if modified_duration > 0 else 0.0

        comm.send({"type": "progress", "message": f"Speedup: {speedup:0.2f}x (Original duration: {original_duration:0.2f}s, Modified duration: {modified_duration:0.2f}s)"})

        # Create success result with timing information
        success_result = TestCodeSuccess(
            diff=diff_result,
            original_duration=original_duration,
            modified_duration=modified_duration,
            speedup=speedup
        )

        # Serialize to JSON-compatible format using Pydantic
        return success_result.model_dump()

    def _make_json_safe(self, obj):
        """Convert an object to a JSON-safe format, handling numpy arrays and NaN values."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # For large arrays, just return a summary instead of the full array
            if obj.size > 100:
                return {
                    "_type": "ndarray",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "size": int(obj.size),
                    "summary": f"Array of shape {obj.shape}"
                }
            # For small arrays, try to convert to list with NaN handling
            try:
                # Replace NaN with None for JSON compatibility
                result = obj.tolist()
                return self._make_json_safe(result)
            except:
                return {
                    "_type": "ndarray",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "size": int(obj.size)
                }
        elif isinstance(obj, (np.integer, np.floating)):
            # Convert numpy scalars to Python types
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            else:
                return obj.item()
        elif isinstance(obj, float):
            # Handle Python float NaN and Inf
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            else:
                return obj
        else:
            return obj

    def _test_code_comm_open(self, comm, open_msg):
        """
        Handle test_code comm requests.

        Note: test_code() now always returns a structured result (success or crash),
        so we always send ok=True. The result's 'status' field discriminates between
        success, original_crash, and modified_crash.
        """
        try:
            original_code = open_msg["content"]["data"]["original_code"]
            modified_code = open_msg["content"]["data"]["modified_code"]
            output_variables = set[str](open_msg["content"]["data"]["output_variables"])
            comm.send({"type": "progress", "message": f"Output variables: {output_variables}"})

            # test_code() returns a structured result (TestCodeSuccess/TestCodeOriginalCrash/TestCodeModifiedCrash)
            result = self.test_code(comm, original_code, modified_code, output_variables)

            # Make result JSON-safe before sending
            safe_result = self._make_json_safe(result)
            comm.send({"type": "final", "ok": True, "result": safe_result})

        except Exception as e:
            # This catches unexpected errors in comm message parsing or kernel internals,
            # not code execution errors (those are handled in test_code)
            error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            comm.send({"type": "final", "ok": False, "error": error_msg})

    def _kernel_command_comm_open(self, comm, open_msg):
        """
        Handle kernel_command comm requests.

        Routes commands to appropriate handlers based on the 'command' field.
        Supports progress messages for long-running operations.
        """
        try:
            data = open_msg["content"]["data"]
            command = data.get("command")

            if not command:
                raise ValueError("Missing 'command' field in request")

            # Get the appropriate handler
            handler = self.command_handlers.get_handler(command)

            # Create progress callback for operations that support it
            def send_progress(message: str):
                progress_msg = ProgressMessage(message=message)
                comm.send(progress_msg.model_dump())

            # Special handling for test_code which uses progress callback
            if command == "test_code":
                # Parse request
                from data_ferret.kernel.kernel_commands import TestCodeRequest
                request = TestCodeRequest(**data)

                # Execute with progress callback
                response = self.command_handlers.handle_test_code(
                    request,
                    progress_callback=send_progress,
                )

                # Make result JSON-safe before sending
                response_dict = response.model_dump()
                if "result" in response_dict:
                    response_dict["result"] = self._make_json_safe(response_dict["result"])

                # Send final response
                final_msg = FinalMessage(
                    ok=True,
                    response=response_dict,
                )
                comm.send(final_msg.model_dump())

            else:
                # For other commands, just execute handler
                # Dynamically import the appropriate request model
                from data_ferret.kernel import kernel_commands as cmd_module

                # Build request class name (e.g., "CheckpointSaveRequest")
                parts = command.split("_")
                request_class_name = "".join(p.capitalize() for p in parts) + "Request"
                request_class = getattr(cmd_module, request_class_name)

                # Parse and validate request
                request = request_class(**data)

                # Execute handler
                response = handler(request)

                # Send final response
                final_msg = FinalMessage(
                    ok=True,
                    response=response.model_dump(),
                )
                comm.send(final_msg.model_dump())

        except Exception as e:
            # Send error response
            error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            final_msg = FinalMessage(
                ok=False,
                error=error_msg,
            )
            comm.send(final_msg.model_dump())

    async def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
        *,
        cell_meta=None,
        cell_id=None,
    ):
        if cell_id is not None:
            self._cell_id = cell_id
        elif cell_meta is not None:
            self._cell_id = cell_meta.get("cell_id", None)
        else:
            self._cell_id = None

        cancel_flag = {"done": False}

        self.display_cell_id()

        # if the first line of code matches "# timeout <seconds>" grab the number of seconds, remove that line, and stash the timout in self._timeout
        match = re.match(r"# timeout (\d+)\n", code)
        if match:
            self._cell_timeout = int(match.group(1))
            code = code.replace(match.group(0), "", 1)
        else:
            self._cell_timeout = self._default_cell_timeout
        if self._verbose:
            print(f"Cell timeout: {self._cell_timeout}")

        force_checkpoints = self._force_checkpoints
        if force_checkpoints:
            self.checkpoint(f"save cell_{self._cell_id}")

        def _timeout_handler():
            # 1) raise KeyboardInterrupt in the main thread
            _thread.interrupt_main()

            # 2) optional: after a short grace, nuke stragglers so finally cleanup sees fewer
            def _escalate():
                if cancel_flag["done"]:
                    return
                try:
                    me = psutil.Process(os.getpid())
                    kids = me.children(recursive=True)
                    for p in kids:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                except Exception:
                    pass

            # fire escalation after a small grace window
            threading.Timer(self._post_kb_grace, _escalate).start()

        # Arm the per-cell watchdog
        t = threading.Timer(self._cell_timeout, _timeout_handler)
        t.daemon = True
        t.start()

        normal_exit = False
        try:
            has_cell_magics = code.startswith("%") or "\n%" in code
            start_time = time.time()
            if self._use_scalene and self._cell_id is not None and not has_cell_magics:
                pre_type_models = { k: str(v) for k, v in self._checkpoint.type_models(self.shell.user_ns).items() }
                result, contents = await self.do_scalene(code, self._cell_id, store_history)
                post_type_models = { k: str(v) for k, v in self._checkpoint.type_models(self.shell.user_ns).items() }
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
                contents = None
                pre_type_models = None
                post_type_models = None
            normal_exit = True
            end_time = time.time()

            # Serialize to dict for notebook metadata
            metadata = {
                'profile': {
                    'duration': end_time - start_time,
                    'profile': contents if contents is not None else "",
                    'env': pre_type_models if pre_type_models is not None else {},
                    'env_after': post_type_models if post_type_models is not None else {},
                }
            }

            if contents is not None:
                self.display_icon_and_text(
                    "🔍",
                    f"{end_time - start_time:0.2f}s",
                    contents=contents,
                    metadata=metadata,
                )
            else:
                if not has_cell_magics:
                    self.display_icon_and_text(
                        "⏱️",
                        f"{end_time - start_time:0.2f}s",
                        metadata=metadata,
                    )

            if force_checkpoints:
                assert self.shell is not None, "shell is not set"
                user_ns = self._checkpoint.checkpointable_vars(self.shell.user_ns)
                user_ns = self._checkpoint.checkpointable_values(user_ns)
                dummy_memo = {}
                old = self._checkpoint.get(f"cell_{self._cell_id}")
                self.diff_checkpoints(old, Checkpoint(f"_tmp", user_ns, dummy_memo))

            return result

        finally:
            # Disarm watchdog so it doesn't trip after we’ve finished
            cancel_flag["done"] = True
            try:
                t.cancel()
            except Exception:
                pass
            if not normal_exit:
                # Kernel-side cleanup: stop loky, reset singleton, kill ALL children
                try:
                    await stop_loky_and_all_children(
                        timeout=self._kill_timeout,
                        verbose=self._verbose,
                        max_passes=self._passes,
                    )
                except Exception:
                    pass

    def wrap_last_expr_with_print_repr(self, src: str) -> str:
        """
        Given the code for a Jupyter cell as a string, if the final statement
        is an expression statement, replace it with:
            _val = <expr>
            if _val is not None:
                print(repr(_val))
        Otherwise, return the code unchanged.
        """
        try:
            tree = ast.parse(src, mode="exec", type_comments=True)
        except SyntaxError:
            return src

        if not tree.body:
            return src

        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            # _val = <expr>
            assign = ast.Assign(
                targets=[ast.Name(id="_val", ctx=ast.Store())], value=last.value
            )

            # if _val is not None: print(repr(_val))
            cond = ast.Compare(
                left=ast.Name(id="_val", ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            )
            print_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id="repr", ctx=ast.Load()),
                            args=[ast.Name(id="_val", ctx=ast.Load())],
                            keywords=[],
                        )
                    ],
                    keywords=[],
                )
            )
            if_stmt = ast.If(test=cond, body=[print_call], orelse=[])

            # Replace the last expression with these two statements
            tree.body[-1:] = [assign, if_stmt]
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)

        return src

    async def do_scalene(
        self, code: str, cell_id: str, store_history: bool
    ) -> Tuple[dict[str, Any], str | None]:
        from scalene import ScaleneArguments, scalene_profiler

        code = self.wrap_last_expr_with_print_repr(code)

        cell_id = cell_id
        args = {"outfile": f"_ipython-profile-{cell_id}.txt", "memory": False}

        try:
            n = self.shell.execution_count - 1
            filename = f"_ipython-input-{n}-profile"
            with open(filename, "w") as tmpfile:
                tmpfile.write(code)

            self._executed_cell_ids[self.shell.execution_count - 1] = cell_id

            scalene_profiler.Scalene.set_initialized()

            args = ScaleneArguments()
            args.outfile = f"_ipython-profile-{cell_id}.txt"
            args.memory = False
            args.gpu = False
            args.json = False
            args.html = False
            args.web = False
            args.no_browser = True

            # Capture stderr
            stderr_buffer = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = stderr_buffer
            try:
                scalene_profiler.Scalene.run_profiler(args, [filename], is_jupyter=True)
                self.shell.execution_count += 1
                contents = self.display_contents(args.outfile)
            finally:
                sys.stderr = old_stderr
                scalene_output = stderr_buffer.getvalue()
                expected_msg = (
                    "Scalene: The specified code did not run for long enough to profile.\n"
                    "By default, Scalene only profiles code in the file executed and its subdirectories.\n"
                    "To track the time spent in all files, use the `--profile-all` option.\n"
                )
                if scalene_output and scalene_output != expected_msg:
                    print(scalene_output, file=sys.stderr)

            result = {"status": "ok", "execution_count": self.shell.execution_count - 1}
            return result, contents

        except Exception as e:
            result = {
                "status": "error",
                "execution_count": self.shell.execution_count - 1,
                "traceback": traceback.format_exception(type(e), e, e.__traceback__),
                "ename": str(type(e).__name__),
                "evalue": str(e),
            }
            return result, None
        finally:
            if os.path.exists(args.outfile):
                os.remove(args.outfile)
            if os.path.exists(filename):
                os.remove(filename)

    def replace_filenames_with_cell_ids(self, text: str) -> str:
        # This pattern matches any path ending in _ipython-input-<N>-profile
        pattern = r"/[^\s]*?_ipython-input-(\d+)-profile"

        # Replacement function: take the captured N, convert to zero‑based index, and format
        def repl(m):
            n = int(m.group(1))
            return f"Cell {self._executed_cell_ids[n]}"

        return re.sub(pattern, repl, text)

    def display_contents(self, filename: str) -> str | None:
        try:
            with open(filename, "r") as f:
                contents = f.read()
            contents = self.replace_filenames_with_cell_ids(contents)
            return contents
        except FileNotFoundError:
            return None


if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretKernel)
