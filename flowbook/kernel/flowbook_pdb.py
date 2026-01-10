import types
import ast
import atexit
import inspect
import linecache
import os
import pydoc
import sys
import textwrap
import traceback
from io import StringIO
from pathlib import Path

import IPython

import pdb

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.debugger import Pdb
from flowbook.util.text import strip_ansi
from flowbook.kernel.locals import print_locals, print_user_globals


class FlowbookPdb(Pdb):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chat_prefix = "   "
        self._text_width = 120
        self.context = 10

        # set this to True ONLY AFTER we have had access to stack frames
        self._show_locals = True

        self._library_paths = [os.path.dirname(os.__file__)] + [
            path
            for path in sys.path
            if "site-packages" in path or "dist-packages" in path
        ]

    def _is_user_frame(self, frame):
        if not self._is_user_file(frame.f_code.co_filename):
            return False
        name = frame.f_code.co_name
        return not name.startswith("<") or name == "<module>"

    def _is_user_file(self, file_name):
        if file_name.endswith(".pyx"):
            return False
        elif file_name == "<string>" or file_name.startswith("<frozen"):
            # synthetic entry point or frozen modules
            return False
        elif file_name.startswith("<ipython"):
            # stdin from ipython session
            return True

        for path in self._library_paths:
            if os.path.commonpath([file_name, path]) == path:
                return False

        return True

    def enriched_stack_trace(self) -> str:
        old_stdout = self.stdout
        buf = StringIO()
        self.stdout = buf
        try:
            self.print_stack_trace()
        finally:
            self.stdout = old_stdout
        return strip_ansi(buf.getvalue())

    def _hide_lib_frames(self):
        # hide lib frames
        for s in self.stack:
            s[0].f_locals["__tracebackhide__"] = not self._is_user_frame(s[0])

        # truncate huge stacks
        for frame in self.stack[0:-30]:
            frame[0].f_locals["__tracebackhide__"] = True

        # go up until we are not in a library
        while self.curindex > 0 and self.curframe_locals.get(
            "__tracebackhide__", False
        ):
            self.curindex -= 1
            self.curframe, self.lineno = self.stack[self.curindex]
            self.curframe_locals = self.curframe.f_locals

        # Assume assertions are correct and the code leading to them is not!
        if self.curframe.f_lineno != None:
            current_line = linecache.getline(
                self.curframe.f_code.co_filename, self.curframe.f_lineno
            )
            if current_line.strip().startswith("assert"):
                self._error_details = f"The code `{current_line.strip()}` is correct and MUST remain unchanged in your fix."

    def capture_onecmd(self, line) -> str:
        """
        Run one Pdb command, but capture and return stdout.
        """
        stdout = self.stdout
        try:
            self.stdout = StringIO()
            super().onecmd(line)
            result = self.stdout.getvalue().rstrip()
            result = strip_ansi(result)
            return result
        finally:
            self.stdout = stdout

    def _hidden_predicate(self, frame):
        """
        Given a frame return whether it it should be hidden or not by IPython.
        """

        if self._predicates["readonly"]:
            fname = frame.f_code.co_filename
            # we need to check for file existence and interactively define
            # function would otherwise appear as RO.
            if os.path.isfile(fname) and not os.access(fname, os.W_OK):
                return True

        if self._predicates["tbhide"]:
            if frame in (self.curframe, getattr(self, "initial_frame", None)):
                return False
            fname = frame.f_code.co_filename

            # Hack because the locals for this frame are shared with
            # the first user frame, so we can't rely on the flag
            # in frame_locals to be set properly.
            if fname == "<string>":
                return True

            frame_locals = self._get_frame_locals(frame)
            if "__tracebackhide__" not in frame_locals:
                return False
            return frame_locals["__tracebackhide__"]
        return False

    def print_stack_trace(self, locals=None):
        # override to print the skips into stdout instead of stderr...
        if locals is None:
            locals = self._show_locals
        else:
            locals = locals and self._show_locals

        skipped = 0
        for hidden, frame_lineno in zip(self.hidden_frames(self.stack), self.stack):
            if hidden and self.skip_hidden:
                skipped += 1
                continue
            if skipped:
                print(
                    f"    [... skipping {skipped} hidden frame(s)]\n",
                    file=self.stdout,
                )
                skipped = 0
            self.print_stack_entry(frame_lineno)
            if locals:
                print_locals(self.stdout, frame_lineno[0])
        if skipped:
            print(
                f"    [... skipping {skipped} hidden frame(s)]\n",
                file=self.stdout,
            )

    def enriched_globals(self) -> str:
        old_stdout = self.stdout
        buf = StringIO()
        self.stdout = buf
        try:
            print_user_globals(self.stdout, self.shell.user_ns)
        finally:
            self.stdout = old_stdout
        return strip_ansi(buf.getvalue())
