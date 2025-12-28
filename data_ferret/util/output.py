import json
from pathlib import Path
import sys
import os
import socket
import textwrap
import time
import traceback
import datetime

from attr import dataclass
from data_ferret.util.text import strip_ansi
import termcolor
import threading
import atexit
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, TypedDict


Timing = TypedDict("Timing", {"key": str, "duration": float})
Timings = List[Timing]


class SocketStream:
    """
    File-like object that writes to a Unix domain socket.

    Handles connection, reconnection, and buffering for streaming
    output from kernels to the ferret_lab terminal.
    """

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False

    def _connect(self) -> bool:
        """Attempt to connect to the output socket."""
        if self._connected and self._socket:
            return True

        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self.socket_path)
            self._connected = True
            return True
        except (OSError, ConnectionRefusedError):
            self._socket = None
            self._connected = False
            return False

    def write(self, text: str) -> int:
        """Write text to the socket."""
        with self._lock:
            if not self._connect():
                return 0

            try:
                data = text.encode("utf-8")
                self._socket.sendall(data)
                return len(text)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Connection lost, mark for reconnection
                self._connected = False
                self._socket = None
                return 0

    def flush(self):
        """Flush is a no-op for sockets (data is sent immediately)."""
        pass

    def close(self):
        """Close the socket connection."""
        with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None
                self._connected = False


class Output:
    def __init__(self, *, timings_file: str | None = None):
        if timings_file is None:
            # Check environment variable first, then fall back to default
            timings_file = os.environ.get("FERRET_TIMINGS_FILE", "ferret-times.json")
        self.pending = None
        self.contexts = []
        self.output_contexts = []
        # Don't store file reference - resolve lazily on each write
        self.file = None
        self.lock = threading.RLock()
        self.timings: Timings = []
        self.timings_file = timings_file
        self.quiet = 0
        self._timings_flushed = False  # Prevent double-write
        atexit.register(self._print_timings)

    def _is_kernel_context(self):
        """Detect if running inside a Jupyter/IPython kernel."""
        if not hasattr(self, "_in_kernel"):
            self._in_kernel = False
            try:
                from IPython import get_ipython

                ip = get_ipython()
                self._in_kernel = (
                    ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
                )
            except ImportError:
                pass
        return self._in_kernel

    def _get_output_file(self):
        """
        Get the output file handle.

        Priority:
        1. FERRET_OUTPUT_SOCKET - Unix socket for streaming to ferret_lab terminal
        2. /dev/tty - Direct terminal access (works in kernel contexts)
        3. sys.stdout - Default fallback

        Returns:
            File handle for output
        """
        # Check for ferret_lab socket first (works in any context)
        socket_path = os.environ.get("FERRET_OUTPUT_SOCKET")
        if socket_path and os.path.exists(socket_path):
            # Lazily create socket stream
            if not hasattr(self, "_socket_stream") or self._socket_stream is None:
                self._socket_stream = SocketStream(socket_path)
            return self._socket_stream

        return sys.stdout

    def set_timings_file(self, timings_file: str):
        self.timings_file = timings_file

    def _print_timings(self):
        # Prevent double-write (can be called explicitly from do_shutdown and by atexit)
        if self._timings_flushed:
            return
        self._timings_flushed = True

        timings = self.timings
        if timings:
            if os.path.exists(self.timings_file):
                saved_timings: Timings = json.load(open(self.timings_file))
                log(f"Output timer data loaded from {self.timings_file}")
                saved_timings.extend(timings)
                timings = saved_timings
                log(f"Output timer data extended by {len(self.timings)} entries")

            json.dump(timings, open(self.timings_file, "w"), indent=2)
            log(f"Output timer data saved to {self.timings_file}")

    def add_timing(self, key: str, duration: float):
        with self.lock:
            self.timings.append(Timing(key=key, duration=duration))

    def timing_context(self, *, key: str | None = None, message: str | None = None):
        return self.TimedOutputContext(
            self, key, message, color="cyan", start="[", end="]"
        )

    class TimedOutputContext:
        def __init__(
            self,
            outer,
            key: str | None = None,
            message: str | None = None,
            color="cyan",
            start="",
            end="",
        ):
            self.outer = outer
            self.key = key
            if message is None:
                self.message = None
            else:
                self.message = termcolor.colored(message + "...", color)
            self.color = color
            self.start = termcolor.colored(start, color)
            self.end = termcolor.colored(end, color)

        def __enter__(self):
            with self.outer.lock:
                if self.outer.quiet > 0:
                    self.suppressed = True
                else:
                    self.suppressed = False
                    if self.message is not None:
                        self.outer.print_enter(
                            self.message, start=self.start, end=self.end
                        )
                        self.outer.contexts += [self]
                self.start_time = time.time()
                self._duration_ms = None
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                end_time = time.time()
                duration = (end_time - self.start_time) * 1000
                self._duration_ms = duration
                if self.suppressed:
                    return
                if self.key is not None:
                    self.outer.timings.append(Timing(key=self.key, duration=duration))
                if self.message is not None:
                    self.outer.contexts.pop()
                    message = f"{termcolor.colored(f'{int(duration)} ms', self.color)}"
                    self.outer.print_exit(message, start=self.start, end=self.end)

        def duration(self) -> float:
            """
            Return elapsed duration in milliseconds.

            Returns:
                Duration in milliseconds as a float.

            Raises:
                ValueError: If called before the context has exited.
            """
            if self._duration_ms is None:
                raise ValueError("Duration not available - context not yet exited")
            return self._duration_ms

    class IndentedOutputContext:
        def __init__(self, outer, message: str, color="cyan", start="[", end="]"):
            self.outer = outer
            self.message = message
            self.color = color
            self.start = start
            self.end = end

        def __enter__(self):
            with self.outer.lock:
                message = termcolor.colored(self.start + self.message, self.color)
                self.outer.print_enter(message)
                self.outer.contexts.append(self)

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                self.outer.contexts.pop()
                if self.end is not None:
                    message = termcolor.colored(self.end, self.color)
                    self.outer.print_exit(message)

        def write(self, message):
            self.outer.write(message)
            self.outer.flush()

        def flush(self):
            self.outer.flush()

    class FileOutputContext:
        def __init__(self, outer, file_path: str):
            self.outer = outer
            self.file_path = file_path
            self.file = open(self.file_path, "w")

        def __enter__(self):
            with self.outer.lock:
                self.outer.output_contexts.append(self)
                self.outer.log(f"Entering file output context: {self.file_path}\n")

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                self.file.close()
                self.outer.output_contexts.pop()
                self.outer.log(f"Exiting file output context: {self.file_path}\n")

        def write(self, message):
            message = strip_ansi(message)
            self.file.write(message)
            self.file.flush()

        def flush(self):
            self.file.flush()

    class StreamOutputContext:
        """
        Output context that streams messages to a file-like object.

        This context captures all output and sends it to a stream object
        that has standard write() and flush() methods. The stream is
        responsible for any text processing (e.g., ANSI code handling).
        """

        def __init__(self, outer, stream):
            self.outer = outer
            self.stream = stream

        def __enter__(self):
            with self.outer.lock:
                self.outer.output_contexts.append(self)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                self.stream.flush()
                self.outer.output_contexts.pop()

        def write(self, message):
            """Write message to the stream (preserving ANSI codes for stream to process)."""
            self.stream.write(message)
            self.stream.flush()

        def flush(self):
            """Flush the stream."""
            self.stream.flush()

    class QuietOutputContext:
        """
        Context manager that suppresses log() and timer output.

        When active, log messages and timer messages are not displayed,
        and timing data is not collected. error() and print() remain visible.

        Can be nested - suppression is active while any QuietOutputContext
        is active.
        """

        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            with self.outer.lock:
                self.outer.quiet += 1
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                self.outer.quiet -= 1

    def write(self, message):
        file = self._get_output_file()
        file.write(message)
        file.flush()
        for context in self.output_contexts:
            context.write(message)
            context.flush()

    def flush(self):
        if self.pending is not None:
            file = self._get_output_file()
            file.write(self.pending)
            file.write("\n")
            for context in self.output_contexts:
                context.write(self.pending)
                context.write("\n")
            self.pending = None

    def get_pad(self):
        current_time = termcolor.colored(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}]", "cyan"
        )
        return f"{current_time}" + " " * (1 + self.pad_depth())

    def pad_depth(self):
        return len(self.contexts) * 2

    def print_enter(self, message, start="", end=""):
        self.flush()
        pad = self.get_pad()
        indented = textwrap.indent(f"{start}{message}", pad)
        self.write(indented)
        self.pending = end

    # assume message is one line
    def print_exit(self, message, start="", end=""):
        pad = self.get_pad()
        if self.pending is not None:
            if self.pending != end:
                self.write(f"Pending does not match end: {self.pending} != {end}")
            self.write(" ")
            self.write(message)
            self.write(self.pending)
            self.write("\n")
        else:
            self.write(f"{pad}{start}{message}{end}\n")
        self.pending = None

    def _format_exception(self, e):
        tb = e.__traceback__
        formatted_exception = "\nException:\n"
        formatted_exception += "".join(traceback.format_exception(type(e), e, tb))

        return formatted_exception

    def _print(self, color, args, start="", end=""):
        self.flush()
        start_len = len(start)
        pad = self.get_pad()

        message = " ".join(
            self._format_exception(a) if isinstance(a, Exception) else str(a)
            for a in args
        ).rstrip()

        lines = f"{start}{message}{termcolor.colored(end, color)}".split("\n")
        self.write(f"{pad}{termcolor.colored(lines[0], color)}")
        for line in lines[1:]:
            self.write("\n")
            self.write(f"{pad + (' ' * start_len)}{termcolor.colored(line, color)}")
        self.write("\n")

    def print(self, *args):
        with self.lock:
            self._print("light_yellow", args)

    def log(self, *args):
        with self.lock:
            if self.quiet > 0:
                return
            self._print("cyan", args, start="[", end="]")

    def error(self, *args):
        with self.lock:
            self._print("red", args)


output = Output()


def log(*message):
    output.log(*message)


def error(*message):
    output.error(*message)


def print(*message):
    output.print(*message)


class EmptyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def timer(*, key: str | None = None, message: str | None = None):
    return output.timing_context(key=key, message=message)


def indent(*, message: str):
    return output.IndentedOutputContext(
        output, message, color="light_yellow", start="", end=None
    )


def quiet():
    """
    Context manager that suppresses log() and timer output.

    When active, log messages and timer messages are not displayed,
    and timing data is not collected. error() and print() remain visible.

    Example:
        with quiet():
            log("This will not be shown")
            with timer(key="test", message="Test"):
                # Timer message and data collection suppressed
                pass
            error("This WILL be shown")
            print("This WILL be shown too")
    """
    return output.QuietOutputContext(output)


def tee_output(file_path: Path | str):
    return output.FileOutputContext(output, str(file_path))


def stream_output(stream):
    """
    Create an output context that streams to a file-like object.

    The stream object should have standard write() and flush() methods.

    Args:
        stream: A file-like object with write() and flush() methods

    Returns:
        StreamOutputContext that will capture and forward all output

    Example:
        class BroadcastStream:
            def write(self, text):
                # Custom write logic here
                pass

            def flush(self):
                # Custom flush logic here
                pass

        with stream_output(BroadcastStream()):
            log("This message will be streamed")
            print("This too!")
    """
    return output.StreamOutputContext(output, stream)


if __name__ == "__main__":
    # Test the logger
    # output = Output()
    with output.timing_context(key="A", message="Main"):
        output.log("This is a log message")

    with output.timing_context(key="A", message="No time"):
        output.log("This is a log message")
        with output.timing_context(key="A", message="Beep"):
            output.log("This is a log message")
            with output.timing_context(key="B", message="Boop"):
                output.print("Real messages")
                output.log("This is a log message")
                output.error("MOo")
            output.log("This is a log message")
        with output.timing_context(key="C", message="Bop"):
            pass
