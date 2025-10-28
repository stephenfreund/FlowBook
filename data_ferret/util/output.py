import json
from pathlib import Path
import sys
import os
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
from typing import List, Tuple, TypedDict


Timing = TypedDict("Timing", {"key": str, "duration": float})
Timings = List[Timing]


class Output:
    def __init__(self, *, timings_file: str = "ferret-times.json"):
        self.pending = None
        self.contexts = []
        self.output_contexts = []
        self.file = sys.stdout
        self.lock = threading.RLock()
        self.timings: Timings = []
        self.timings_file = timings_file
        atexit.register(self._print_timings)

    def set_timings_file(self, timings_file: str):
        self.timings_file = timings_file

    def _print_timings(self):
        # if os.path.exists(self.timings_file):
        #     timings: Timings = json.load(open(self.timings_file))
        #     timings.extend(self.timings)
        # else:
        timings = self.timings
        if timings:
            json.dump(timings, open(self.timings_file, "w"), indent=2)
            log(f"Output timer data saved to {self.timings_file}")
            

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
                if self.message is not None:
                    self.outer.print_enter(self.message, start=self.start, end=self.end)
                    self.outer.contexts += [self]
                self.start_time = time.time()

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                end_time = time.time()
                duration = (end_time - self.start_time) * 1000
                if self.key is not None:
                    self.outer.timings.append(Timing(key=self.key, duration=duration))
                if self.message is not None:
                    self.outer.contexts.pop()
                    message = f"{termcolor.colored(f'{int(duration)} ms', self.color)}"
                    self.outer.print_exit(message, start=self.start, end=self.end)

    class IndentedOutputContext:
        def __init__(self, outer, message: str, color="cyan"):
            self.outer = outer
            self.message = message
            self.color = color

        def __enter__(self):
            with self.outer.lock:
                message = termcolor.colored("[" +self.message, self.color)
                self.outer.print_enter(message)
                self.outer.contexts.append(self)

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                self.outer.contexts.pop()
                message = termcolor.colored("]", self.color)
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

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        for context in self.output_contexts:
            context.write(message)
            context.flush()

    def flush(self):
        if self.pending is not None:
            self.file.write(self.pending)
            self.file.write("\n")
            for context in self.output_contexts:
                context.write(self.pending)
                context.write("\n")
            self.pending = None

    def get_pad(self):
        current_time = termcolor.colored(f"[{datetime.datetime.now().strftime('%H:%M:%S')}]", "cyan")
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
            self._print("yellow", args)

    def log(self, *args):
        with self.lock:
            self._print("cyan", args, start="[", end="]")

    def error(self, *args):
        with self.lock:
            self._print("red", args, start="[", end="]")


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
    return output.IndentedOutputContext(output, message)

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
