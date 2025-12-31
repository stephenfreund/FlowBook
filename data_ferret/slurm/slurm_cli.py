#!/usr/bin/env python3
"""Submit DataFerret CLI jobs to Slurm with per-notebook conda environment support.

Accepts any number of input files: .txt files (one notebook path per line) or
.ipynb files (treated as individual jobs). For each work item, resolves the
appropriate conda environment and launches a separate `sbatch` job using the
command that follows the `--` separator.

Environment Resolution (4 rules, in priority order):
    1. --env flag: Use specified environment for ALL notebooks (assumes it exists)
    2. .txt file: Use '_env' from same directory as the .txt file (must exist or use --make-env)
    3. Direct .ipynb: Use '_env' from great-grandparent directory (must exist or use --make-env)
       Example: user/notebook-slug/notebook.ipynb -> looks for _env in parent of 'user/' dir
    4. Fallback: Use currently active conda environment

Examples:
    # Use existing 'myenv' for all notebooks (rule 1)
    python slurm_cli.py notebooks.txt --env=myenv -- info

    # Use _env from same directory as notebooks.txt (rule 2, must exist)
    python slurm_cli.py batch1.txt -- execute_all

    # Create _env before running (rule 2 + --make-env)
    python slurm_cli.py batch1.txt --make-env -- info

    # Direct notebook with _env in great-grandparent dir (rule 3, must exist)
    # For user/notebook-slug/notebook.ipynb -> looks for _env in parent of 'user/' dir
    python slurm_cli.py user/notebook-slug/notebook.ipynb -- execute_all

    # Create _env in great-grandparent dir before running (rule 3 + --make-env)
    python slurm_cli.py user/notebook-slug/notebook.ipynb --make-env -- info

    # Multiple files with different _env locations (all must exist)
    python slurm_cli.py batch1.txt batch2.txt notebook.ipynb -- execute_all

    # Create all discovered _env environments before running
    python slurm_cli.py batch1.txt batch2.txt --make-env -- info

    # Run a different DataFerret command (optimize) without auto-appending notebook
    python slurm_cli.py notebooks.txt --no-append-target -- \
        optimize --plan-only --config configs/opt.json

    # Use GPU partition for compute-intensive operations
    python slurm_cli.py notebooks.txt --partition=gpu -- execute_all

Note: The 'data_ferret' command with --timings-file and --metadata-file flags
      is automatically prepended to your command. Just specify the subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")


@dataclass
class WorkItem:
    """A notebook to process and the conda environment to use.

    Attributes:
        notebook_path: Path to the notebook file to process
        env_name: Name of the conda environment to use for execution
        env_dir: Directory containing the _env (for --make-env setup), None if not applicable
        requires_env_check: True if this uses _env discovery (rules 2-3), False otherwise
    """

    notebook_path: Path
    env_name: str
    env_dir: Optional[Path] = None
    requires_env_check: bool = False


def get_ferret_env_exports() -> str:
    """Generate export statements for all FERRET_* environment variables.

    Iterates over the current process environment and returns shell export
    statements for any variables starting with 'FERRET_'.
    """
    exports = []
    for var, value in sorted(os.environ.items()):
        if var.startswith("FERRET_"):
            exports.append(f"export {var}={shlex.quote(value)}")
    return "\n".join(exports)


def get_current_conda_env() -> str:
    """Get the name of the currently active conda environment.

    Returns:
        Name of the active conda environment, defaults to "base" if not in a conda env
    """
    # Try environment variable first (most reliable)
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name

    # Fallback: try conda info --json
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        info = json.loads(result.stdout)
        env_name = info.get("active_prefix_name")
        if env_name:
            return env_name
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass

    # Final fallback
    return "base"


def check_environment_exists(env_name: str, env_dir: Optional[Path] = None) -> bool:
    """Check if a conda environment exists.

    Args:
        env_name: Name of the conda environment to check
        env_dir: Optional directory containing the environment (for path-based envs)

    Returns:
        True if the environment exists, False otherwise
    """
    # For path-based environments, check if the directory exists
    if env_dir is not None:
        env_path = env_dir / env_name
        return env_path.is_dir()

    # For named environments, check conda env list
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Look for env_name as a word boundary (to avoid partial matches)
        # conda env list format: "env_name   /path/to/env"
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == env_name:
                return True
        return False
    except subprocess.CalledProcessError:
        return False


def resolve_environment(
    notebook_path: Path,
    source_file: Optional[Path],
    cli_env: Optional[str],
) -> WorkItem:
    """Resolve the conda environment for a notebook using the 4-rule hierarchy.

    Rules (in priority order):
    1. CLI --env flag overrides everything (assumes env exists)
    2. .txt file -> use _env from same directory as the file
    3. Direct .ipynb -> use _env from great-grandparent directory
       (notebook.ipynb -> parent -> grandparent -> great-grandparent)
    4. Fallback -> use current active conda environment

    Args:
        notebook_path: Path to the notebook file
        source_file: Optional path to .txt file that contained this notebook
        cli_env: Optional environment name from --env CLI flag

    Returns:
        WorkItem with resolved environment and metadata
    """
    # Rule 1: CLI override
    if cli_env is not None:
        return WorkItem(
            notebook_path=notebook_path,
            env_name=cli_env,
            env_dir=None,
            requires_env_check=False,
        )

    # Rule 2: .txt file -> use _env from source file's directory
    if source_file is not None and source_file.suffix.lower() == ".txt":
        return WorkItem(
            notebook_path=notebook_path,
            env_name="_env",
            env_dir=source_file.parent,
            requires_env_check=True,
        )

    # Rule 3: Direct .ipynb -> use _env from great-grandparent directory
    # (notebook.ipynb -> parent -> grandparent -> great-grandparent)
    if source_file is None:
        return WorkItem(
            notebook_path=notebook_path,
            env_name="_env",
            env_dir=notebook_path.parent.parent.parent,
            requires_env_check=True,
        )

    # Rule 4: Fallback to current conda environment
    return WorkItem(
        notebook_path=notebook_path,
        env_name=get_current_conda_env(),
        env_dir=None,
        requires_env_check=False,
    )


def validate_work_item_environments(work_items: List[WorkItem]) -> None:
    """Validate that all required _env environments exist.

    Args:
        work_items: List of work items to validate

    Raises:
        SystemExit: If any required _env is missing
    """
    missing_envs: Dict[Path, str] = {}

    for item in work_items:
        if item.requires_env_check:
            if not check_environment_exists(item.env_name, item.env_dir):
                missing_envs[item.env_dir] = item.env_name

    if missing_envs:
        print("[ERROR] Required conda environments not found:")
        for env_dir, env_name in missing_envs.items():
            print(f"  - '{env_name}' in directory: {env_dir}")
        print()
        print("Solutions:")
        print("  1. Use --make-env to create the missing environments")
        print("  2. Use --env=<existing_env> to override with an existing environment")
        sys.exit(1)


def create_environments(work_items: List[WorkItem], ferret_source: Path) -> bool:
    """Create all discovered _env environments.

    Args:
        work_items: List of work items (only those with requires_env_check=True are processed)
        ferret_source: Path to DataFerret source directory (for pip install -e .)

    Returns:
        True if all environments were created successfully, False otherwise
    """
    # Collect unique (env_name, env_dir) pairs that need creation
    unique_envs: Dict[Tuple[str, Path], List[Path]] = {}
    for item in work_items:
        if item.requires_env_check and item.env_dir is not None:
            key = (item.env_name, item.env_dir)
            if key not in unique_envs:
                unique_envs[key] = []
            unique_envs[key].append(item.notebook_path)

    if not unique_envs:
        print("[INFO] No _env environments to create (using CLI --env or current env)")
        return True

    print(f"[ENV] Creating {len(unique_envs)} environment(s)...")
    print()

    for (env_name, env_dir), notebooks in unique_envs.items():
        env_path = env_dir / env_name
        print(f"[ENV] Setting up '{env_name}' in {env_dir}")
        print(f"[ENV]   Full path: {env_path}")
        print(f"[ENV]   Used by {len(notebooks)} notebook(s)")
        requirements_file = env_dir / "requirements.txt"

        if not setup_environment(env_path, ferret_source, requirements_file):
            print(f"[ERROR] Failed to create environment at {env_path}")
            return False

        print()

    print(f"[ENV] All {len(unique_envs)} environment(s) created successfully")
    return True


def parse_time_limit(time_str: str) -> int:
    """Parse SLURM time format (HH:MM:SS or D-HH:MM:SS) to seconds.

    Examples:
        "01:30:00" -> 5400 (1.5 hours)
        "24:00:00" -> 86400 (24 hours)
        "2-00:00:00" -> 172800 (2 days)
    """
    days = 0
    if "-" in time_str:
        day_part, time_part = time_str.split("-", 1)
        days = int(day_part)
        time_str = time_part

    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid time format: {time_str}. Expected HH:MM:SS or D-HH:MM:SS"
        )

    hours, minutes, seconds = map(int, parts)
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Handles color codes, cursor movement, and other terminal control sequences.

    Args:
        text: String potentially containing ANSI escape sequences

    Returns:
        String with all ANSI escape sequences removed
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def get_timer_count(timings_file_path: Optional[str]) -> Tuple[int, str]:
    """Get the count of cli_main_exit in a timings file.

    Args:
        timings_file_path: Full path to the timings file

    Returns:
        (count, status_str) where count is the number of cli_main_exit entries
        and status_str is a human-readable status
    """
    if timings_file_path is None:
        return (-1, "no timings file specified")

    timers_path = Path(timings_file_path)
    if not timers_path.exists():
        return (-1, "timings file not found")

    try:
        with open(timers_path) as f:
            timings = json.load(f)
    except (json.JSONDecodeError, Exception):
        return (-1, "error reading timings")

    if not isinstance(timings, list):
        return (-1, "invalid timings format")

    count = sum(1 for t in timings if t.get("key") == "cli_main_exit")
    return (count, "ok" if count == 1 else f"count={count}")


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS or MM:SS."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_expected_runtime(target_path: Path) -> Optional[float]:
    """Get expected runtime from Kaggle log file if available.

    Notebooks downloaded from Kaggle include a .log file with execution
    timestamps. The last entry's 'time' field is the total runtime.

    Args:
        target_path: Path to the notebook file

    Returns:
        Expected runtime in seconds, or None if not available
    """
    log_file = target_path.parent / (target_path.stem + ".log")
    if not log_file.exists():
        return None
    try:
        with open(log_file) as f:
            log = json.load(f)
            if log and isinstance(log, list) and len(log) > 0:
                last_entry = log[-1]
                if isinstance(last_entry, dict) and "time" in last_entry:
                    return float(last_entry["time"])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    return None


def build_job_id(target_path: Path) -> str:
    """Build a unique job identifier from the notebook path.

    Format: {username}---{slug}---{stem}

    Given: user/notebook-slug/notebook.ipynb
    Returns: user---notebook-slug---notebook

    Falls back to just the stem if path doesn't have enough components.

    Args:
        target_path: Path to the notebook file

    Returns:
        Job identifier string suitable for filenames
    """
    stem = target_path.stem or "target"
    parent_name = target_path.parent.name
    grandparent_name = target_path.parent.parent.name

    if grandparent_name and parent_name:
        return f"{grandparent_name}---{parent_name}---{stem}"
    elif parent_name:
        return f"{parent_name}---{stem}"
    else:
        return stem


def wait_for_jobs(
    job_ids: List[int],
    job_info: Dict[int, Tuple[Path, Optional[str]]],
    poll_interval: int = 5,
    log_dir: Optional[Path] = None,
) -> Dict[int, str]:
    """Wait for all SLURM jobs to complete, polling squeue.

    Uses squeue to monitor jobs. When a job disappears from squeue,
    it's assumed to have finished (marked as FINISHED since sacct
    may not be available to get the exact exit status).

    Args:
        job_ids: List of SLURM job IDs to monitor
        job_info: Dict mapping job_id -> (target_path, timings_file_path)
        poll_interval: Seconds between status checks
        log_dir: Directory containing job output files (for dumping on warnings)

    Returns:
        Dictionary mapping job_id -> final_state
    """
    results: Dict[int, str] = {}
    pending = set(job_ids)
    start_time = time.time()

    try:
        while pending:
            job_list = ",".join(str(j) for j in pending)
            still_in_queue: set[int] = set()

            try:
                # squeue shows jobs that are still running or pending
                squeue_cmd = [
                    "squeue",
                    "-j",
                    job_list,
                    "--noheader",
                    "--format=%i|%T",
                ]
                squeue_output = subprocess.check_output(
                    squeue_cmd, text=True, stderr=subprocess.DEVNULL
                )
                for line in squeue_output.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 1:
                        try:
                            still_in_queue.add(int(parts[0].strip()))
                        except ValueError:
                            continue
            except subprocess.CalledProcessError:
                # squeue may fail if all jobs in the list have finished
                # (returns non-zero when no matching jobs found)
                pass

            # Jobs not in squeue are finished
            finished_jobs = pending - still_in_queue
            for job_id in finished_jobs:
                results[job_id] = "FINISHED"
                pending.discard(job_id)
                target, timings_file = job_info.get(job_id, (Path("unknown"), None))
                count, status = get_timer_count(timings_file)
                if count == 1:
                    print(f"[STATUS] Job {job_id} -> FINISHED (ok) {target}")
                else:
                    # Warning case - dump output file for debugging
                    if count == 0:
                        print(
                            f"[WARNING] Job {job_id} -> FINISHED (cli_main_exit=0) {target}"
                        )
                    elif count > 1:
                        print(
                            f"[WARNING] Job {job_id} -> FINISHED (cli_main_exit={count}) {target}"
                        )
                    else:
                        print(f"[WARNING] Job {job_id} -> FINISHED ({status}) {target}")
                    # Dump the .out file contents for debugging
                    if log_dir is not None:
                        custom_job_id = build_job_id(target)
                        out_file = log_dir / f"{custom_job_id}.out"
                        if out_file.exists():
                            print(f"[WARNING]   --- {out_file} ---")
                            try:
                                content = out_file.read_text()
                                for line in content.splitlines():
                                    print(f"[WARNING]     {line}")
                                print(f"[WARNING]   --- end of {job_id}.out ---")
                            except Exception as e:
                                print(f"[WARNING]   (failed to read: {e})")

            if pending:
                elapsed = format_elapsed(time.time() - start_time)
                print(
                    f"[WAIT] {len(pending)} jobs still running... (elapsed: {elapsed})"
                )
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C received, stopping monitoring.")
        # Mark remaining jobs as interrupted
        for job_id in pending:
            results[job_id] = "MONITORING_INTERRUPTED"

    return results


def setup_environment(
    env_path: Path, ferret_source: Path, requirements_file: Path
) -> bool:
    """Create/recreate a conda environment with DataFerret and requirements.

    Args:
        env_path: Full path to the conda environment directory (e.g., /path/to/_env)
        ferret_source: Path to DataFerret source directory (for pip install -e .)
        requirements_file: Path to requirements.txt

    Returns:
        True if successful, False otherwise
    """
    # 1. Remove existing environment if it exists
    print(f"[ENV] Removing existing environment at '{env_path}' if present...")
    subprocess.run(
        ["conda", "env", "remove", "-p", str(env_path), "-y"],
        stderr=subprocess.DEVNULL,
    )

    # 2. Create new environment with Python 3.11
    print(f"[ENV] Creating new environment at '{env_path}' with Python 3.11...")
    print(f"[ENV] Running: conda create -p {env_path} python=3.11 -y", flush=True)
    result = subprocess.run(
        ["conda", "create", "-p", str(env_path), "python=3.11", "-y"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to create environment (exit code: {result.returncode})")
        return False

    # 3. Install requirements if file exists
    if requirements_file.exists():
        print(f"[ENV] Installing requirements from {requirements_file}...")
        print(
            f"[ENV] Running: conda run -p {env_path} pip install -v -r {requirements_file}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(
            [
                "conda",
                "run",
                "-p",
                str(env_path),
                "pip",
                "install",
                "-v",
                "-r",
                str(requirements_file),
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
        if result.returncode != 0:
            print(
                f"[ERROR] Failed to install requirements (exit code: {result.returncode})"
            )
            return False
    else:
        print(f"[ENV] No requirements file found at {requirements_file}, skipping")

    # 4. Install DataFerret from source -- do after requirements to avoid conflicts
    print(f"[ENV] Installing DataFerret from {ferret_source}...")
    print(
        f"[ENV] Running: conda run -p {env_path} pip install -v -e {ferret_source}",
        flush=True,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            "conda",
            "run",
            "-p",
            str(env_path),
            "pip",
            "install",
            "-v",
            "-e",
            str(ferret_source),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to install DataFerret (exit code: {result.returncode})")
        return False

    print(f"[ENV] Environment setup complete at {env_path}")
    return True


def extract_timings_file_from_command(cmd_tokens: List[str]) -> Optional[str]:
    """Extract --timings-file value from command tokens.

    Args:
        cmd_tokens: List of command tokens (after template substitution)

    Returns:
        The timings file path if found, None otherwise
    """
    for i, token in enumerate(cmd_tokens):
        # Handle --timings-file=value format
        if token.startswith("--timings-file="):
            return token.split("=", 1)[1]
        # Handle --timings-file value format
        if token == "--timings-file" and i + 1 < len(cmd_tokens):
            return cmd_tokens[i + 1]
    return None


def check_job_timers(timings_file_path: Optional[str]) -> Tuple[bool, str]:
    """Check if a job finished normally by examining its timers file.

    A job is considered to have finished normally if cli_main_exit was
    triggered exactly once.

    Args:
        timings_file_path: Full path to the timings file

    Returns:
        (success, message) tuple where success=True if completed normally
    """
    if timings_file_path is None:
        return (False, "no --timings-file in command")

    timers_path = Path(timings_file_path)

    if not timers_path.exists():
        return (False, f"timings file not found: {timers_path}")

    try:
        with open(timers_path) as f:
            timings = json.load(f)
    except json.JSONDecodeError as e:
        return (False, f"invalid JSON in timings file: {e}")
    except Exception as e:
        return (False, f"error reading timings file: {e}")

    if not isinstance(timings, list):
        return (False, "timings file is not a list")

    # Count occurrences of cli_main_exit
    exit_count = sum(1 for t in timings if t.get("key") == "cli_main_exit")

    if exit_count == 1:
        return (True, "completed normally (cli_main_exit triggered once)")
    elif exit_count == 0:
        return (False, "cli_main_exit not found in timings")
    else:
        return (False, f"cli_main_exit triggered {exit_count} times (expected 1)")


def split_cli_sections(argv: Sequence[str]) -> tuple[List[str], List[str]]:
    """Split CLI arguments into script options and command tokens."""
    if not argv:
        return [], []
    if "--" in argv:
        idx = argv.index("--")
        return list(argv[:idx]), list(argv[idx + 1 :])
    return list(argv), []


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments and return the populated namespace."""
    slurm_args, command_template = split_cli_sections(list(argv))
    parser = argparse.ArgumentParser(
        description=(
            "Submit one Slurm job per notebook with per-notebook conda environments. "
            "Specify the DataFerret subcommand after a literal `--` separator. "
            "The 'data_ferret' command with --timings-file and --metadata-file is "
            "automatically prepended."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example commands (after `--`):
              info                   -> runs: data_ferret --timings-file=... --metadata-file=... info <notebook>
              execute_all            -> runs: data_ferret --timings-file=... --metadata-file=... execute_all <notebook>
              optimize --plan-only   -> runs: data_ferret --timings-file=... --metadata-file=... optimize --plan-only <notebook>

            Formatting tokens available in your command:
              {target}               -> absolute path to the notebook file
              {target_dir}           -> absolute directory containing the notebook
              {target_name}          -> filename with extension (no directory)
              {target_stem}          -> filename stem (no extension)
              {target_with_extension} -> filename plus any --target-extension
              {log_dir}              -> directory where output files go
              {job_id}               -> unique job identifier (user---slug---stem)
            """
        ),
    )
    parser.add_argument(
        "--partition", default="gpmoo-b", help="Slurm partition (default: gpmoo-b)"
    )
    parser.add_argument(
        "--time",
        dest="time_limit",
        default="24:00:00",
        help="Time limit, e.g., 24:00:00 (default: 24h)",
    )
    parser.add_argument("--mem", default="16G", help="Memory request (default: 16G)")
    parser.add_argument(
        "--cpus", type=int, default=4, help="CPUs per task (default: 4)"
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="GPUs per task (default: 1)"
    )
    parser.add_argument(
        "--job-name",
        default="ferret-batch",
        help="Slurm job name prefix (default: ferret-batch)",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Conda environment to use for ALL notebooks (overrides _env discovery)",
    )
    parser.add_argument(
        "--make-env",
        action="store_true",
        help="Create/recreate _env environments in discovered locations before submitting jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run commands locally in sequence instead of submitting to Slurm (respects --time limit)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit immediately after submitting jobs (don't wait for completion)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=1,
        help="Seconds between SLURM status checks when waiting (default: 1)",
    )
    parser.add_argument(
        "--log-dir",
        default="slurm_logs",
        help="Directory for sbatch stdout/stderr files (default: slurm_logs)",
    )
    parser.add_argument(
        "--no-append-target",
        dest="append_target",
        action="store_false",
        help="Do not automatically append the path from the work file",
    )
    parser.set_defaults(append_target=True)
    parser.add_argument(
        "--target-extension",
        default=".ipynb",
        help="Suffix appended to the target name when missing (default: .ipynb; set to '' to disable)",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input files: .txt files (one target per line) or .ipynb files (direct jobs)",
    )
    args = parser.parse_args(slurm_args)

    if not command_template:
        parser.error("Missing command to run. Add `-- <command>` at the end.")
    args.command_template = command_template
    return args


def load_work_items(work_file: Path, cli_env: Optional[str]) -> List[WorkItem]:
    """Return work items from a .txt file, ignoring comments and blank lines.

    Args:
        work_file: Path to .txt file containing notebook paths (one per line)
        cli_env: Optional environment name from --env CLI flag

    Returns:
        List of WorkItem objects with resolved environments
    """
    if not work_file.is_file():
        raise FileNotFoundError(f"File not found: {work_file}")
    items: List[WorkItem] = []
    for raw in work_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        notebook_path = work_file.parent / line
        # Resolve environment for this notebook (source_file is the .txt file)
        work_item = resolve_environment(notebook_path, work_file, cli_env)
        items.append(work_item)
    return items


def collect_work_items(
    input_files: List[str], cli_env: Optional[str]
) -> List[WorkItem]:
    """Collect work items from multiple input files with resolved environments.

    Args:
        input_files: List of input file paths (.txt or .ipynb files)
        cli_env: Optional environment name from --env CLI flag

    Returns:
        List of WorkItem objects with resolved environments

    Behavior:
        - .txt files: read line-by-line, each line is a notebook path
        - .ipynb files: treated as single work items
    """
    items: List[WorkItem] = []
    for file_str in input_files:
        file_path = Path(file_str)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            # Load notebooks from text file with environment resolution
            items.extend(load_work_items(file_path, cli_env))
        elif suffix == ".ipynb":
            # Direct notebook file (source_file is None for direct .ipynb)
            work_item = resolve_environment(file_path, None, cli_env)
            items.append(work_item)
        else:
            raise ValueError(
                f"Unsupported file type: {file_path} (expected .txt or .ipynb)"
            )
    return items


def build_command_tokens(
    context: dict[str, str], args: argparse.Namespace
) -> List[str]:
    """Create the concrete command tokens for the given target context.

    Automatically prepends the standard data_ferret command with timings and metadata flags.

    Args:
        context: Dictionary with formatting variables (target, log_dir, job_id, etc.)
        args: Command-line arguments

    Returns:
        List of command tokens ready for execution
    """
    # Prepend standard data_ferret command with timings and metadata
    cmd_tokens = [
        "data_ferret",
        f"--timings-file={context['log_dir']}/{context['job_id']}.timers.json",
        f"--metadata-file={context['log_dir']}/{context['job_id']}.metadata.json",
    ]
    # Add user-provided command tokens
    cmd_tokens.extend([token.format(**context) for token in args.command_template])
    if args.append_target:
        cmd_tokens.append(context["target_name"])
    return cmd_tokens


def run_local_job(work_item: WorkItem, args: argparse.Namespace) -> bool:
    """Run the job locally in a subprocess with timeout.

    Args:
        work_item: WorkItem containing notebook path and environment info
        args: Command-line arguments

    Returns:
        True if the job completed successfully, False otherwise.
    """
    abs_target = work_item.notebook_path.expanduser()
    if not abs_target.is_absolute():
        abs_target = abs_target.resolve(strict=False)
    work_dir = abs_target.parent
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    job_id = build_job_id(abs_target)

    context = {
        "target": str(abs_target),
        "target_dir": str(work_dir),
        "target_name": abs_target.name,
        "target_stem": abs_target.stem,
        "log_dir": str(log_dir),
        "job_id": job_id,
    }

    job_label = f"{args.job_name}-{job_id}"
    command_tokens = build_command_tokens(context, args)

    # Parse timeout from time limit
    timeout_seconds = parse_time_limit(args.time_limit)

    # Build the shell script to run
    command_str = shlex.join(command_tokens)

    # Determine environment activation command (use full path if env_dir is set)
    if work_item.env_dir is not None:
        env_path = (work_item.env_dir / work_item.env_name).resolve()
        env_activate = f"conda activate ./{shlex.quote(str(env_path))}"
    else:
        env_activate = f"conda activate {shlex.quote(work_item.env_name)}"

    inner_cmd = textwrap.dedent(
        f"""
        set -x

        # ---- Python output & hang diagnostics ----
        export PYTHONUNBUFFERED=1
        export PYTHONFAULTHANDLER=1

        # ---- FERRET environment variables ----
        {get_ferret_env_exports()}

        # ---- Node info ----
        echo "===== LOCAL EXECUTION INFO ====="
        hostname
        date
        which python || true
        python -V || true
        echo "================================"

        # ---- Conda environment ----
        source ~/.bashrc || true
        {env_activate} || echo "Warning: Failed to activate conda environment"

        conda info

        # ---- Enable strict error handling after environment setup ----
        set -euo pipefail

        # ---- Requested command ----
        {command_str}
        """
    ).strip()

    stdout_path = log_dir / f"{job_id}.out"
    stderr_path = log_dir / f"{job_id}.err"

    # Report expected runtime if available from Kaggle logs
    expected_runtime = get_expected_runtime(abs_target)
    if expected_runtime is not None:
        print(f"[LOCAL] Expected runtime (Kaggle): {format_elapsed(expected_runtime)}")

    print(f"[LOCAL] Running: {command_str}")
    print(f"[LOCAL] Working directory: {work_dir}")
    print(f"[LOCAL] Timeout: {timeout_seconds}s ({args.time_limit})")
    print(f"[LOCAL] Stdout:  {stdout_path}")
    print(f"[LOCAL] Stderr:  {stderr_path}")

    try:
        # Capture output to strings for ANSI stripping
        result = subprocess.run(
            ["bash", "-lc", inner_cmd],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=True,
            check=False,
        )

        # Strip ANSI codes and write cleaned output to files
        stdout_clean = strip_ansi_codes(result.stdout or "")
        stderr_clean = strip_ansi_codes(result.stderr or "")

        with open(stdout_path, "w") as f:
            f.write(stdout_clean)
        with open(stderr_path, "w") as f:
            f.write(stderr_clean)

        if result.returncode == 0:
            print(f"[LOCAL] ✓ Completed successfully (exit code 0)")
            return True
        else:
            print(f"[LOCAL] ✗ Failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[LOCAL] ✗ Timed out after {timeout_seconds}s")
        return False
    except Exception as exc:
        print(f"[LOCAL] ✗ Error: {exc}")
        return False


def submit_single_job(work_item: WorkItem, args: argparse.Namespace) -> Optional[int]:
    """Submit the sbatch job for one work item.

    Args:
        work_item: WorkItem containing notebook path and environment info
        args: Command-line arguments

    Returns:
        SLURM job ID if successful, None otherwise
    """
    abs_target = work_item.notebook_path.expanduser()
    if not abs_target.is_absolute():
        abs_target = abs_target.resolve(strict=False)
    work_dir = abs_target.parent
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    job_id = build_job_id(abs_target)

    context = {
        "target": str(abs_target),
        "target_dir": str(work_dir),
        "target_name": abs_target.name,
        "target_stem": abs_target.stem,
        "log_dir": str(log_dir),
        "job_id": job_id,
    }

    job_label = f"{args.job_name}-{job_id}"
    command_tokens = build_command_tokens(context, args)
    command_str = shlex.join(command_tokens)

    # Determine environment activation command (use full path if env_dir is set)
    if work_item.env_dir is not None:
        env_path = (work_item.env_dir / work_item.env_name).resolve()
        env_activate = f"conda activate ./{shlex.quote(str(env_path))}"
    else:
        env_activate = f"conda activate {shlex.quote(work_item.env_name)}"

    inner_cmd = textwrap.dedent(
        f"""
        set -euo pipefail
        set -x

        # ---- Python output & hang diagnostics ----
        export PYTHONUNBUFFERED=1
        export PYTHONFAULTHANDLER=1

        # ---- Node info ----
        echo "===== NODE INFO ====="
        hostname
        date
        echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_NODELIST=$SLURM_NODELIST  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
        which python || true
        python -V || true
        nvidia-smi || true
        echo "====================="

        # ---- OpenMP/BLAS/MKL/NumExpr threads ----
        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
        export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK

        # ---- FERRET environment variables ----
        {get_ferret_env_exports()}

        # ---- Conda environment ----
        source ~/.bashrc
        {env_activate}

        # cd {shlex.quote(str(work_dir))}

        # ---- Requested command ----
        # srun --cpu-bind=cores {command_str}
        srun {command_str}
        """
    ).strip()

    stdout_path = log_dir / f"{job_id}.out"
    stderr_path = log_dir / f"{job_id}.err"

    sbatch_args: List[str] = [
        "sbatch",
        f"--job-name={job_label}",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus}",
        "--chdir",
        str(work_dir),
        f"--output={stdout_path}",
        f"--error={stderr_path}",
        f"--partition={args.partition}",
        f"--time={args.time_limit}",
        f"--mem={args.mem}",
    ]
    if args.gpus:
        sbatch_args.append(f"--gres=gpu:{args.gpus}")

    wrapped = f"bash -lc {shlex.quote(inner_cmd)}"
    cmd = sbatch_args + ["--wrap", wrapped]

    # Report expected runtime if available from Kaggle logs
    expected_runtime = get_expected_runtime(abs_target)
    if expected_runtime is not None:
        print(f"[SUBMIT] Expected runtime (Kaggle): {format_elapsed(expected_runtime)}")

    if args.dry_run:
        print("DRY RUN:", " ".join(shlex.quote(part) for part in cmd))
        return None

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print(out.strip())
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] sbatch failed for {target}:\n{exc.output}")
        return None

    match = SUBMITTED_RE.search(out)
    return int(match.group(1)) if match else None


def main() -> None:
    """Script entry point."""
    args = parse_args(sys.argv[1:])
    work_items = collect_work_items(args.input_files, args.env)
    if not work_items:
        print("[INFO] No valid work items found. Nothing to submit.")
        return

    # DataFerret source is the repo root (parent of data_ferret/slurm/)
    ferret_source = Path(__file__).parent.parent.parent

    # Handle --make-env: create/recreate _env environments before submitting jobs
    if args.make_env:
        if not create_environments(work_items, ferret_source):
            print("[ERROR] Environment creation failed, aborting")
            sys.exit(1)
        print()
    else:
        # Validate that all required _env environments exist
        validate_work_item_environments(work_items)

    if args.local:
        # Local execution mode
        print(f"[LOCAL MODE] Running {len(work_items)} jobs sequentially...")
        print()
        succeeded = 0
        failed = 0
        for i, work_item in enumerate(work_items, 1):
            print(
                f"[{i}/{len(work_items)}] {work_item.notebook_path} (env: {work_item.env_name})"
            )
            print("-" * 60)
            if run_local_job(work_item=work_item, args=args):
                succeeded += 1
            else:
                failed += 1
            print()

        print("=" * 60)
        print(f"[SUMMARY] Completed {succeeded + failed}/{len(work_items)} jobs")
        print(f"[SUMMARY] ✓ Succeeded: {succeeded}")
        print(f"[SUMMARY] ✗ Failed: {failed}")
    else:
        # SLURM submission mode
        submitted: List[int] = []
        job_targets: Dict[int, Path] = {}
        job_timings_files: Dict[int, Optional[str]] = {}
        log_dir = Path(args.log_dir).resolve()

        for work_item in work_items:
            print(f"{work_item.notebook_path} (env: {work_item.env_name})")
            print("-----------------------")
            job_id = submit_single_job(work_item=work_item, args=args)
            if job_id is not None:
                submitted.append(job_id)
                job_targets[job_id] = work_item.notebook_path

                # Build context and command tokens to extract timings file
                abs_target = work_item.notebook_path.expanduser()
                if not abs_target.is_absolute():
                    abs_target = abs_target.resolve(strict=False)
                context = {
                    "target": str(abs_target),
                    "target_dir": str(abs_target.parent),
                    "target_name": abs_target.name,
                    "target_stem": abs_target.stem,
                    "log_dir": str(log_dir),
                    "job_id": build_job_id(abs_target),
                }
                cmd_tokens = build_command_tokens(context, args)
                job_timings_files[job_id] = extract_timings_file_from_command(
                    cmd_tokens
                )
            print()

        if submitted:
            print(f"[OK] Submitted {len(submitted)}/{len(work_items)} jobs.")
            print("Job IDs:", " ".join(str(job) for job in submitted))
        else:
            print("[WARN] No jobs submitted.")

        # Wait for jobs and check completion status
        if not args.no_wait and submitted and not args.dry_run:
            print(f"\n[WAIT] Monitoring {len(submitted)} jobs...")
            print(f"[WAIT] Poll interval: {args.poll_interval}s")
            print()

            # Build job_info dict: job_id -> (target_path, timings_file_path)
            job_info: Dict[int, Tuple[Path, Optional[str]]] = {
                job_id: (job_targets[job_id], job_timings_files.get(job_id))
                for job_id in submitted
            }

            final_states = wait_for_jobs(
                submitted, job_info, poll_interval=args.poll_interval, log_dir=log_dir
            )

            print()
            print("=" * 60)
            print("[RESULTS] Job completion status:")
            print("=" * 60)

            normal_count = 0
            for job_id in submitted:
                state = final_states.get(job_id, "UNKNOWN")
                target = job_targets[job_id]
                timings_file = job_timings_files.get(job_id)
                if state == "FINISHED":
                    # Job left the queue - check timers to see if it completed normally
                    success, msg = check_job_timers(timings_file)
                    if success:
                        print(f"[OK] Job {job_id} ({target.stem}): {msg}")
                        normal_count += 1
                    else:
                        print(f"[WARN] Job {job_id} ({target.stem}): {msg}")
                        # Dump the .out file contents for debugging
                        out_file = log_dir / f"{job_id}.out"
                        if out_file.exists():
                            print(f"[WARN]   --- {out_file} ---")
                            try:
                                content = out_file.read_text()
                                for line in content.splitlines():
                                    print(f"[WARN]     {line}")
                                print(f"[WARN]   --- end of {job_id}.out ---")
                            except Exception as e:
                                print(f"[WARN]   (failed to read: {e})")
                else:
                    print(f"[SKIP] Job {job_id} ({target.stem}): {state}")

            print()
            print(f"[SUMMARY] {normal_count}/{len(submitted)} jobs finished normally")


if __name__ == "__main__":
    main()
