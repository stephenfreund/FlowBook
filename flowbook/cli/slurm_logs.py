"""CLI tool to extract and display exceptions from FlowBook slurm log files.

Given a directory containing .out and .err files from flowbook_slurm runs,
this tool finds all Python tracebacks and reports them with:
- The log file name and the notebook/directory it corresponds to
- Context lines showing what was happening when the exception occurred
- The full traceback and final error line

Usage:
    flowbook_slurm_logs /path/to/slurm_logs/
    flowbook_slurm_logs /path/to/slurm_logs/ --summary
    flowbook_slurm_logs /path/to/slurm_logs/ --unique
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ANSI escape code pattern
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Timestamp prefix pattern: [HH:MM:SS] followed by at most one space.
# Preserves any additional indentation after the timestamp.
_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ?")


def strip_ansi(text: str) -> str:
    """Remove ANSI color/formatting escape codes from text."""
    return _ANSI_RE.sub("", text)


def strip_timestamp(text: str) -> str:
    """Remove leading timestamp prefix like '[09:13:30] ' from a line."""
    return _TIMESTAMP_RE.sub("", text)


def parse_job_label(filename: str) -> Tuple[str, str]:
    """Extract user and notebook name from slurm log filename.

    Filenames follow the pattern: user---parent_dir---notebook_stem.{out,err}

    Returns:
        (user, notebook_stem) or (filename, "") if pattern doesn't match.
    """
    stem = Path(filename).stem
    parts = stem.split("---")
    if len(parts) >= 3:
        return parts[0], parts[-1]
    return stem, ""


def extract_notebook_info(lines: List[str]) -> Optional[str]:
    """Extract notebook path from the srun/flowbook command line or Arguments block."""
    for line in lines:
        clean = strip_ansi(line)
        # srun flowbook ... execute notebook.ipynb
        m = re.search(r"(?:srun\s+)?flowbook\s+.*?(\S+\.ipynb)", clean)
        if m:
            return m.group(1)
        # Arguments block: paths : ['notebook.ipynb']
        m = re.match(r"paths\s*:\s*\['([^']+)'\]", clean.strip())
        if m:
            return m.group(1)
    return None


def extract_working_dir(lines: List[str]) -> Optional[str]:
    """Extract working directory from srun --chdir or conda activate path."""
    for line in lines:
        clean = strip_ansi(line)
        m = re.search(r"--chdir\s+(\S+)", clean)
        if m:
            return m.group(1)
        # conda activate often reveals the env/project path
        m = re.search(r"conda activate\s+(\S+)", clean)
        if m:
            return m.group(1)
    return None


class ExceptionInfo:
    """A single exception extracted from a log file."""

    def __init__(
        self,
        file_path: str,
        context_lines: List[str],
        traceback_lines: List[str],
        error_line: str,
        line_number: int,
    ):
        self.file_path = file_path
        self.context_lines = context_lines
        self.traceback_lines = traceback_lines
        self.error_line = error_line
        self.line_number = line_number

    @property
    def error_type(self) -> str:
        """Extract the exception class name from the error line."""
        m = re.match(r"(\w+(?:Error|Exception|Warning|Interrupt)\w*)", self.error_line)
        return m.group(1) if m else self.error_line.split(":")[0].strip()

    @property
    def short_message(self) -> str:
        """The error line without the exception class prefix."""
        parts = self.error_line.split(":", 1)
        return parts[1].strip() if len(parts) > 1 else self.error_line

    @property
    def signature(self) -> str:
        """A deduplication key: error type + innermost frame file + line."""
        # Use the last File "..." line as the location
        for line in reversed(self.traceback_lines):
            m = re.search(r'File "([^"]+)", line (\d+)', line)
            if m:
                filepath = Path(m.group(1)).name
                lineno = m.group(2)
                return f"{self.error_type} at {filepath}:{lineno}"
        return self.error_type


def extract_exceptions(file_path: str) -> List[ExceptionInfo]:
    """Parse a log file and extract all Python tracebacks.

    Handles:
    - Standard Python tracebacks ("Traceback (most recent call last):")
    - ANSI-colored output (stripped before matching)
    - Context lines before the traceback (up to 3 non-blank lines)
    """
    try:
        with open(file_path, "r", errors="replace") as f:
            raw_lines = f.readlines()
    except OSError:
        return []

    lines = [strip_ansi(line.rstrip()) for line in raw_lines]
    exceptions: List[ExceptionInfo] = []

    i = 0
    while i < len(lines):
        # Look for traceback start
        if "Traceback (most recent call last):" not in lines[i]:
            i += 1
            continue

        tb_start = i

        # Gather context: up to 3 non-empty lines before the traceback
        context = []
        for j in range(max(0, tb_start - 5), tb_start):
            stripped = lines[j].strip()
            if stripped and "Traceback" not in stripped:
                context.append(lines[j])
        context = context[-3:]  # keep at most 3

        # Collect traceback lines.
        # Lines may be prefixed with timestamps (e.g., "[09:13:30]   File ...")
        # from the FlowBook output logger. We strip timestamps for pattern
        # matching but keep the original line for display.
        tb_lines = [lines[i]]
        i += 1
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # Strip timestamp, then check the content structure
            detimed = strip_timestamp(line)
            content = strip_timestamp(stripped)
            # Traceback body: indented lines (File, code, ^^^)
            # Check BEFORE stripping: line or detimed starts with whitespace
            is_indented = detimed.startswith("  ") or detimed.startswith("\t")
            is_frame = content.startswith("File ") or content.startswith("Cell ")
            is_caret = content.startswith("^") or content.startswith("~")
            # Detect the final error line: "SomeError: message" pattern
            # This can appear indented in timestamped output
            is_error_line = bool(re.match(
                r"\w+(?:Error|Exception|Warning|Interrupt)\w*:", content.lstrip()
            ))
            if is_error_line:
                error_line = content.lstrip()
                tb_lines.append(line)
                i += 1
                break
            if is_frame or is_indented or is_caret:
                tb_lines.append(line)
                i += 1
                continue
            # The final error line is the first non-indented, non-File line
            if content:
                error_line = content
                tb_lines.append(line)
                i += 1
                break
            # Blank line — end of traceback
            if not stripped:
                i += 1
                break
        else:
            error_line = strip_timestamp(tb_lines[-1].strip()) if tb_lines else "Unknown error"

        exceptions.append(ExceptionInfo(
            file_path=file_path,
            context_lines=context,
            traceback_lines=tb_lines,
            error_line=error_line,
            line_number=tb_start + 1,  # 1-indexed
        ))

    return exceptions


def find_log_files(directory: str) -> List[str]:
    """Find all .out and .err files in the given directory."""
    d = Path(directory)
    files = sorted(
        str(f) for f in d.iterdir()
        if f.suffix in (".out", ".err") and f.is_file()
    )
    return files


def display_exceptions(
    exceptions_by_file: Dict[str, List[ExceptionInfo]],
    all_lines: Dict[str, List[str]],
    show_traceback: bool = True,
) -> int:
    """Display all exceptions grouped by file.

    Returns total exception count.
    """
    total = 0

    for file_path, exceptions in sorted(exceptions_by_file.items()):
        if not exceptions:
            continue

        p = Path(file_path)
        user, notebook = parse_job_label(p.name)
        lines = all_lines.get(file_path, [])
        nb_path = extract_notebook_info(lines)

        print(f"{'=' * 78}")
        print(f"  File:     {p.name}")
        if notebook:
            print(f"  User:     {user}")
            print(f"  Notebook: {notebook}")
        if nb_path and nb_path != notebook:
            print(f"  Path:     {nb_path}")
        print(f"  Errors:   {len(exceptions)}")
        print(f"{'=' * 78}")
        print()

        for idx, exc in enumerate(exceptions, 1):
            total += 1
            print(f"  [{idx}] {exc.error_type} (line {exc.line_number})")
            print(f"      {exc.error_line}")
            print()

            if exc.context_lines:
                print("      Context:")
                for cl in exc.context_lines:
                    print(f"        {cl.strip()}")
                print()

            if show_traceback:
                print("      Traceback:")
                for tl in exc.traceback_lines:
                    print(f"        {tl.strip()}")
                print()

            print(f"  {'-' * 74}")
            print()

    return total


def display_summary(
    exceptions_by_file: Dict[str, List[ExceptionInfo]],
) -> None:
    """Display a compact summary table."""
    print(f"{'File':<60} {'Errors':>6}")
    print("-" * 68)

    total = 0
    files_with_errors = 0
    for file_path in sorted(exceptions_by_file.keys()):
        exceptions = exceptions_by_file[file_path]
        if not exceptions:
            continue
        files_with_errors += 1
        p = Path(file_path)
        count = len(exceptions)
        total += count
        # Truncate long filenames
        name = p.name
        if len(name) > 58:
            name = name[:55] + "..."
        print(f"{name:<60} {count:>6}")

    print("-" * 68)
    all_files = len(exceptions_by_file)
    print(f"Total: {total} exceptions in {files_with_errors}/{all_files} files")


def display_unique(
    exceptions_by_file: Dict[str, List[ExceptionInfo]],
) -> None:
    """Display unique exception types with counts and example locations."""
    sig_counter: Counter = Counter()
    sig_examples: Dict[str, List[str]] = defaultdict(list)

    for file_path, exceptions in exceptions_by_file.items():
        for exc in exceptions:
            sig = exc.signature
            sig_counter[sig] += 1
            p = Path(file_path).name
            if len(sig_examples[sig]) < 3:
                sig_examples[sig].append(p)

    if not sig_counter:
        print("No exceptions found.")
        return

    print(f"{'Exception Signature':<55} {'Count':>6}")
    print("=" * 63)

    for sig, count in sig_counter.most_common():
        print(f"{sig:<55} {count:>6}")
        for example in sig_examples[sig]:
            name = example if len(example) <= 55 else example[:52] + "..."
            print(f"    {name}")
        print()

    print(f"Total: {sum(sig_counter.values())} exceptions, "
          f"{len(sig_counter)} unique signatures")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and display exceptions from FlowBook slurm log files."
    )
    parser.add_argument(
        "directory",
        help="Directory containing .out and .err log files",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only a compact summary table (file names and error counts)",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Group by unique exception signature and show counts",
    )
    parser.add_argument(
        "--no-traceback",
        action="store_true",
        help="Hide full tracebacks, show only error lines and context",
    )
    parser.add_argument(
        "--err-only",
        action="store_true",
        help="Only scan .err files (skip .out files)",
    )
    parser.add_argument(
        "--out-only",
        action="store_true",
        help="Only scan .out files (skip .err files)",
    )

    args = parser.parse_args()

    directory = args.directory
    if not Path(directory).is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        return 1

    # Find log files
    log_files = find_log_files(directory)
    if args.err_only:
        log_files = [f for f in log_files if f.endswith(".err")]
    elif args.out_only:
        log_files = [f for f in log_files if f.endswith(".out")]

    if not log_files:
        print(f"No log files found in {directory}")
        return 0

    # Extract exceptions from all files
    exceptions_by_file: Dict[str, List[ExceptionInfo]] = {}
    all_lines: Dict[str, List[str]] = {}

    for file_path in log_files:
        try:
            with open(file_path, "r", errors="replace") as f:
                raw_lines = f.readlines()
            all_lines[file_path] = [strip_ansi(l.rstrip()) for l in raw_lines]
        except OSError:
            all_lines[file_path] = []
        exceptions_by_file[file_path] = extract_exceptions(file_path)

    # Count
    total_exceptions = sum(len(excs) for excs in exceptions_by_file.values())
    files_with_errors = sum(1 for excs in exceptions_by_file.values() if excs)

    print(f"\nScanned {len(log_files)} files in {directory}")
    print(f"Found {total_exceptions} exceptions in {files_with_errors} files\n")

    if total_exceptions == 0:
        return 0

    # Display
    if args.unique:
        display_unique(exceptions_by_file)
    elif args.summary:
        display_summary(exceptions_by_file)
    else:
        display_exceptions(
            exceptions_by_file,
            all_lines,
            show_traceback=not args.no_traceback,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
