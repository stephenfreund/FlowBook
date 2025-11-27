"""Scalene profiler integration for kernel execution."""

import io
import os
import re
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

from data_ferret.kernel.ast_utils import wrap_last_expr_with_print_repr


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

        Args:
            code: The code to execute
            cell_id: The cell ID for this execution
            store_history: Whether to store in history

        Returns:
            Tuple of (result dict, profile contents or None)
        """
        from scalene import ScaleneArguments, scalene_profiler

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

        try:
            n = self.shell.execution_count - 1
            filename = f"_ipython-input-{n}-profile"
            with open(filename, "w") as tmpfile:
                tmpfile.write(code)

            self.executed_cell_ids[self.shell.execution_count - 1] = cell_id

            scalene_profiler.Scalene.set_initialized()

            # Capture stderr
            stderr_buffer = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = stderr_buffer
            try:
                scalene_profiler.Scalene.run_profiler(args, [filename], is_jupyter=True)
                self.shell.execution_count += 1
                contents = self._read_profile(args.outfile)
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
