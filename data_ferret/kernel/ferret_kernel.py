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
        self._use_scalene = True
        self.display_icon_and_text("🔍", "Scalene enabled")

    @line_cell_magic
    def disable_scalene(self, line: str, cell: str = "") -> None:
        self._use_scalene = False
        self.display_icon_and_text("🔍", "Scalene disabled")

    @line_cell_magic
    def force_checkpoints(self, line: str, cell: str = "") -> None:
        self._force_checkpoints = True
        self.display_icon_and_text("✅", "Force checkpoints enabled")

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
        assert self.shell is not None, "shell is not set"
        args = line.split()

        if args[0] == "save":
            if len(args) != 2:
                self.display_icon_and_text("Usage: checkpoint save <name>")
                return

            start_time = time.time()
            saved, removed = self._checkpoint.save(args[1], self.shell.user_ns)
            end_time = time.time()
            duration = end_time - start_time

            for k in removed:
                del self.shell.user_ns[k]

            added_text = "\n".join([f"* {k}: {v}" for k, v in saved.items()])
            removed_text = "\n".join([f"* {k}: {v}" for k, v in removed.items()])
            metadata = {
                "ferret": {
                    "added": added_text,
                    "removed": removed_text,
                    "duration": duration,
                }
            }

            if removed:
                self.display_icon_and_text(
                    "✅",
                    f"{duration:0.2f}s [removed: {', '.join(sorted(removed.keys()))}]",
                    contents=added_text,
                    metadata=metadata,
                )
            else:
                self.display_icon_and_text(
                    "✅",
                    f"{duration:0.2f}s",
                    contents=added_text,
                    metadata=metadata,
                )
        elif args[0] == "restore":
            if len(args) != 2:
                self.display_icon_and_text("❌", "Usage: checkpoint restore <name>")
                return

            try:
                self._checkpoint.restore(args[1], self.shell.user_ns)
            except Exception as e:
                self.display_icon_and_text("❌", f"Error restoring checkpoint: {e}")
                return
        elif args[0] == "delete":
            if len(args) != 2:
                self.display_icon_and_text("❌", "Usage: checkpoint delete <name>")
                return

            try:
                self._checkpoint.delete(args[1])
            except Exception as e:
                self.display_icon_and_text("❌", f"Error deleting checkpoint: {e}")
                return
        elif args[0] == "list":
            self.display_icon_and_text(
                "✅",
                f"Checkpoints: {', '.join(sorted(self._checkpoint.list()))}",
                metadata=metadata,
            )
        elif args[0] == "compare":
            if len(args) != 3:
                self.display_icon_and_text(
                    "❌", "Usage: checkpoint compare <name1> <name2>"
                )
                return

            try:
                old = self._checkpoint.get(args[1])
                new = self._checkpoint.get(args[2])
                self.diff_checkpoints(old, new)
            except Exception as e:
                self.display_icon_and_text("❌", f"Error comparing checkpoints: {e}")
                return
        elif args[0] == "clear":
            self._checkpoint.clear()
        else:
            self.display_icon_and_text("❌", f"Unknown checkpoint command: {line}")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # any exception → our handler
        assert self.shell is not None, "shell is not set"
        self.shell.register_magics(self)
        # register our new RPC target
        self.comm_manager.register_target(
            "debug_command", self._debug_command_comm_open
        )
        self.comm_manager.register_target(
            "test_code", self._test_code_comm_open
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
        """Test the code and return the result, sending progress messages via comm."""
        comm.send({"type": "progress", "message": "Saving original environment"})
        self.checkpoint(f"save original_environment")

        comm.send({"type": "progress", "message": "Executing original code"})
        self.shell.run_cell(original_code)

        comm.send({"type": "progress", "message": "Saving original result"})
        self.checkpoint(f"save original_result")

        comm.send({"type": "progress", "message": "Restoring original environment"})
        self.checkpoint(f"restore original_environment")

        comm.send({"type": "progress", "message": "Executing modified code"})
        self.shell.run_cell(modified_code)

        comm.send({"type": "progress", "message": "Saving modified result"})
        self.checkpoint(f"save modified_result")

        comm.send({"type": "progress", "message": "Diffing original and modified environments"})
        diff_result = checkpoint_diff(self._checkpoint.get(f"original_result"), self._checkpoint.get(f"modified_result"), keys_to_include=output_variables)

        # Serialize DiffResult to JSON-compatible format using Pydantic
        serialized = diff_result.model_dump()
        return serialized

    def _test_code_comm_open(self, comm, open_msg):
        try:
            original_code = open_msg["content"]["data"]["original_code"]
            modified_code = open_msg["content"]["data"]["modified_code"]
            output_variables = set[str](open_msg["content"]["data"]["output_variables"])
            comm.send({"type": "progress", "message": f"Output variables: {output_variables}"})
            result = self.test_code(comm, original_code, modified_code, output_variables)
            comm.send({"type": "final", "ok": True, "result": result})
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            comm.send({"type": "final", "ok": False, "error": error_msg})

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

        n = self.shell.execution_count - 1
        filename = f"_ipython-input-{n}-profile"
        with open(filename, "w") as tmpfile:
            tmpfile.write(code)

        self._executed_cell_ids[self.shell.execution_count - 1] = cell_id

        scalene_profiler.Scalene.set_initialized()

        try:
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
