"""Scalene profiler integration for kernel execution."""

import gc
import io
import os
import re
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

from data_ferret.kernel.ast_utils import wrap_last_expr_with_print_repr


class StopJupyterExecution(Exception):
    """Signal to stop Jupyter execution (used by Scalene)."""
    pass


class ScaleneRunner:
    """Runs code with Scalene profiling in a Jupyter kernel context."""

    def __init__(self, shell, executed_cell_ids: Dict[int, str]):
        """
        Initialize the Scalene runner.

        Args:
            shell: The IPython shell instance
            executed_cell_ids: Dict mapping execution counts to cell IDs
        """
        self.shell = shell
        self.executed_cell_ids = executed_cell_ids

    async def run(
        self, code: str, cell_id: str, store_history: bool
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Run code with Scalene profiling.

        This is a simplified implementation that directly uses shell.user_ns
        as the execution namespace, allowing TrackingDict to work properly.

        Args:
            code: The code to execute
            cell_id: The cell ID for this execution
            store_history: Whether to store in history

        Returns:
            Tuple of (result dict, profile contents or None)
        """
        from scalene import ScaleneArguments
        from scalene.scalene_profiler import Scalene
        from scalene.scalene_statistics import Filename

        code = wrap_last_expr_with_print_repr(code)

        args = ScaleneArguments()
        args.outfile = f"_ipython-profile-{cell_id}.txt"
        args.memory = False
        args.gpu = False
        args.json = False
        args.html = False
        args.web = False
        args.no_browser = True
        args.column_width = 132 * 4

        n = self.shell.execution_count - 1
        filename = f"_ipython-input-{n}-profile"

        try:
            # Write code to temp file for Scalene
            with open(filename, "w") as tmpfile:
                tmpfile.write(code)

            self.executed_cell_ids[self.shell.execution_count - 1] = cell_id

            # Compile the code
            prog_name = os.path.abspath(filename)
            compiled_code = compile(code, prog_name, "exec")

            # Use shell.user_ns directly - this is the key change!
            # This allows TrackingDict to capture variable accesses
            the_globals = self.shell.user_ns
            the_locals = self.shell.user_ns

            # Set up __file__ for the execution context
            old_file = the_globals.get("__file__")
            the_globals["__file__"] = prog_name

            # Initialize Scalene for Jupyter mode
            Scalene.set_initialized()
            Scalene.set_in_jupyter()

            # Clear stats and process args
            Scalene._Scalene__stats.clear_all()
            Scalene.process_args(args)

            # Do GC before starting
            gc.collect()

            # Create profiler and run
            profiler = Scalene(args, Filename(prog_name))

            # Capture stderr for error detection
            stderr_buffer = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = stderr_buffer

            try:
                # profile_code uses the_locals and the_globals we provide
                exit_status = profiler.profile_code(
                    compiled_code, the_locals, the_globals, [filename]
                )
                self.shell.execution_count += 1
                contents = self._read_profile(args.outfile)
            except StopJupyterExecution:
                # Normal exit for Jupyter
                self.shell.execution_count += 1
                contents = self._read_profile(args.outfile)
            finally:
                sys.stderr = old_stderr
                scalene_output = stderr_buffer.getvalue()

                # Restore __file__
                if old_file is not None:
                    the_globals["__file__"] = old_file
                elif "__file__" in the_globals:
                    del the_globals["__file__"]

                # Check for errors in output
                parsed_error = self._parse_scalene_error_output(scalene_output)
                if parsed_error:
                    ename, evalue, traceback_lines = parsed_error
                    result = {
                        "status": "error",
                        "execution_count": self.shell.execution_count - 1,
                        "traceback": traceback_lines,
                        "ename": ename,
                        "evalue": evalue,
                    }
                    return result, None

                # Log unexpected output
                expected_msg = (
                    "Scalene: The specified code did not run for long enough to profile.\n"
                    "By default, Scalene only profiles code in the file executed and its subdirectories.\n"
                    "To track the time spent in all files, use the `--profile-all` option.\n"
                )
                if scalene_output and scalene_output != expected_msg:
                    print(scalene_output, file=sys.__stderr__)

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

    def _read_profile(self, filename: str) -> Optional[str]:
        """Read and process the profile output file."""
        try:
            with open(filename, "r") as f:
                contents = f.read()
            contents = self._replace_filenames_with_cell_ids(contents)
            return contents
        except FileNotFoundError:
            return None

    def _replace_filenames_with_cell_ids(self, text: str) -> str:
        """Replace internal profile filenames with readable cell IDs."""
        pattern = r"/[^\s]*?_ipython-input-(\d+)-profile"

        def repl(m):
            n = int(m.group(1))
            return f"Cell {self.executed_cell_ids.get(n, n)}"

        return re.sub(pattern, repl, text)

    def _parse_scalene_error_output(
        self, output: str
    ) -> Optional[Tuple[str, str, list[str]]]:
        """
        Parse scalene error output to extract exception info.

        Args:
            output: The stderr output from scalene

        Returns:
            Tuple of (ename, evalue, traceback_lines) or None if no error found
        """
        if "Error in program being profiled:" not in output:
            return None

        lines = output.split("\n")

        # Find the traceback start
        traceback_start = None
        for i, line in enumerate(lines):
            if line.strip() == "Traceback (most recent call last):":
                traceback_start = i
                break

        if traceback_start is None:
            return None

        # Collect traceback frames and find the exception line
        traceback_lines = ["Traceback (most recent call last):"]
        exception_line = None
        i = traceback_start + 1

        while i < len(lines):
            line = lines[i]

            # Check if this is the exception line (no leading spaces, contains "Error:")
            if line and not line.startswith(" ") and ":" in line:
                # This looks like the exception line
                exception_line = line
                break

            # Skip empty lines at the start of traceback
            if not line.strip() and len(traceback_lines) == 1:
                i += 1
                continue

            # Check if this is a file line we should skip (scalene internals)
            if "scalene_profiler.py" in line and "in profile_code" in line:
                # Skip this frame - advance past the File line and the code line(s)
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    i += 1
                continue

            # Add this line to the traceback (without trailing newline)
            if line.strip():  # Only add non-empty lines
                # Replace filenames in File lines
                if line.strip().startswith("File "):
                    line = self._replace_filenames_with_cell_ids(line)
                traceback_lines.append(line)

            i += 1

        if exception_line is None:
            return None

        # Parse exception name and value
        if ":" in exception_line:
            ename, evalue = exception_line.split(":", 1)
            ename = ename.strip()
            evalue = evalue.strip()
        else:
            ename = exception_line.strip()
            evalue = ""

        # Add the exception line to traceback (without trailing newline)
        traceback_lines.append(f"{ename}: {evalue}")

        return ename, evalue, traceback_lines
