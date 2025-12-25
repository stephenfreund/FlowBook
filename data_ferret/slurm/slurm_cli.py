#!/usr/bin/env python3
"""Submit DataFerret CLI jobs to Slurm.

Reads a text file containing one work item (path) per line (for example,
`zyh1104/sticker-sales-solution-ensembling/nb.ipynb`), then launches a separate
`sbatch` job for each entry using the command that follows the `--` separator.
Every job activates a Conda environment, prints node diagnostics, changes into
the notebook’s directory, and runs the requested CLI on the notebook filename.

Examples:
    # Run `data_ferret info <notebook>` from its directory
    python slurm_cli.py notebooks.txt --env=ferret -- data_ferret info

    # Run optimize without automatically appending the notebook filename
    python slurm_cli.py notebooks.txt -env=moo --no-append-target -- \
        data_ferret_optimize --plan-only --config configs/opt.json

    # Collect timing stats for every work item
    python slurm_cli.py notebooks.txt --partition=gpu -env=perf -- \
        data_ferret_timers --summary
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")


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
) -> Dict[int, str]:
    """Wait for all SLURM jobs to complete, polling squeue.

    Uses squeue to monitor jobs. When a job disappears from squeue,
    it's assumed to have finished (marked as FINISHED since sacct
    may not be available to get the exact exit status).

    Args:
        job_ids: List of SLURM job IDs to monitor
        job_info: Dict mapping job_id -> (target_path, timings_file_path)
        poll_interval: Seconds between status checks

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
                elif count == 0:
                    print(f"[STATUS] Job {job_id} -> FINISHED (cli_main_exit=0) {target}")
                elif count > 1:
                    print(f"[STATUS] Job {job_id} -> FINISHED (cli_main_exit={count}) {target}")
                else:
                    print(f"[STATUS] Job {job_id} -> FINISHED ({status}) {target}")

            if pending:
                elapsed = format_elapsed(time.time() - start_time)
                print(f"[WAIT] {len(pending)} jobs still running... (elapsed: {elapsed})")
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C received, stopping monitoring.")
        # Mark remaining jobs as interrupted
        for job_id in pending:
            results[job_id] = "MONITORING_INTERRUPTED"

    return results


def setup_environment(env_name: str, ferret_source: Path, requirements_file: Path) -> bool:
    """Create/recreate a conda environment with DataFerret and requirements.

    Args:
        env_name: Name of the conda environment to create
        ferret_source: Path to DataFerret source directory (for pip install -e .)
        requirements_file: Path to requirements.txt

    Returns:
        True if successful, False otherwise
    """
    # 1. Remove existing environment if it exists
    print(f"[ENV] Removing existing environment '{env_name}' if present...")
    subprocess.run(
        ["conda", "env", "remove", "-n", env_name, "-y"],
        capture_output=True,
    )

    # 2. Create new environment with Python 3.11
    print(f"[ENV] Creating new environment '{env_name}' with Python 3.11...")
    result = subprocess.run(
        ["conda", "create", "-n", env_name, "python=3.11", "-y"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to create environment: {result.stderr}")
        return False

    # 3. Install DataFerret from source
    print(f"[ENV] Installing DataFerret from {ferret_source}...")
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install", "-e", str(ferret_source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to install DataFerret: {result.stderr}")
        return False

    # 4. Install requirements if file exists
    if requirements_file.exists():
        print(f"[ENV] Installing requirements from {requirements_file}...")
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                env_name,
                "pip",
                "install",
                "-r",
                str(requirements_file),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] Failed to install requirements: {result.stderr}")
            return False
    else:
        print(f"[ENV] No requirements file found at {requirements_file}, skipping")

    print(f"[ENV] Environment '{env_name}' setup complete")
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
            "Submit one Slurm job per path listed in the work file. "
            "Provide the command to run after a literal `--` separator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Formatting tokens for commands supplied after `--`:
              {target}               -> absolute path to the notebook file
              {target_dir}           -> absolute directory containing the notebook
              {target_name}          -> filename with extension (no directory)
              {target_stem}          -> filename stem (no extension)
              {target_with_extension} -> filename plus any --target-extension
              {log_dir}              -> directory where output files go
              {job_id}               -> unique job identifier (user---slug---stem)
              {metadata_file}        -> path for metadata output (in log_dir)
              {timers_file}          -> path for timers output (in log_dir)
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
        default="ferret",
        help="Conda environment to activate inside each job (default: ferret)",
    )
    parser.add_argument(
        "--reset-env",
        action="store_true",
        help="Create/recreate the conda environment before submitting jobs",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="Path to requirements.txt (default: requirements.txt in work_file directory)",
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
        default=5,
        help="Seconds between SLURM status checks when waiting (default: 5)",
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
        "work_file",
        help="Text file containing one target path per line (empty lines and #comments are ignored)",
    )
    args = parser.parse_args(slurm_args)

    if not command_template:
        parser.error("Missing command to run. Add `-- <command>` at the end.")
    args.command_template = command_template
    return args


def load_work_items(work_file: Path) -> List[Path]:
    """Return cleaned work items, ignoring comments and blank lines."""
    if not work_file.is_file():
        raise FileNotFoundError(f"File not found: {work_file}")
    items: List[Path] = []
    for raw in work_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Path(line))
    return items


def build_command_tokens(
    context: dict[str, str], args: argparse.Namespace
) -> List[str]:
    """Create the concrete command tokens for the given target context."""
    cmd_tokens = [token.format(**context) for token in args.command_template]
    if args.append_target:
        cmd_tokens.append(context["target_name"])
    return cmd_tokens


def run_local_job(target: Path, args: argparse.Namespace) -> bool:
    """Run the job locally in a subprocess with timeout.

    Returns:
        True if the job completed successfully, False otherwise.
    """
    abs_target = target.expanduser()
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
        "metadata_file": str(log_dir / f"{job_id}.metadata.json"),
        "timers_file": str(log_dir / f"{job_id}.timers.json"),
    }

    job_label = f"{args.job_name}-{job_id}"
    command_tokens = build_command_tokens(context, args)

    # Parse timeout from time limit
    timeout_seconds = parse_time_limit(args.time_limit)

    # Build the shell script to run
    command_str = shlex.join(command_tokens)
    inner_cmd = textwrap.dedent(
        f"""
        set -x

        # ---- Python output & hang diagnostics ----
        export PYTHONUNBUFFERED=1
        export PYTHONFAULTHANDLER=1

        # ---- Node info ----
        echo "===== LOCAL EXECUTION INFO ====="
        hostname
        date
        which python || true
        python -V || true
        echo "================================"

        # ---- Conda environment ----
        source ~/.bashrc || true
        conda activate {shlex.quote(args.env or 'base')} || echo "Warning: Failed to activate conda env {shlex.quote(args.env or 'base')}"

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


def submit_single_job(target: Path, args: argparse.Namespace) -> Optional[int]:
    """Submit the sbatch job for one work item."""
    abs_target = target.expanduser()
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
        "metadata_file": str(log_dir / f"{job_id}.metadata.json"),
        "timers_file": str(log_dir / f"{job_id}.timers.json"),
    }

    job_label = f"{args.job_name}-{job_id}"
    command_tokens = build_command_tokens(context, args)
    command_str = shlex.join(command_tokens)

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

        # ---- Conda environment ----
        source ~/.bashrc
        conda activate {shlex.quote(args.env or 'base')}

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
    targets = load_work_items(Path(args.work_file))
    if not targets:
        print("[INFO] No valid targets found. Nothing to submit.")
        return

    # Handle --reset-env: create/recreate conda environment before submitting jobs
    if args.reset_env:
        # DataFerret source is the repo root (parent of data_ferret/slurm/)
        ferret_source = Path(__file__).parent.parent.parent
        work_file_dir = Path(args.work_file).resolve().parent
        requirements_file = (
            Path(args.requirements)
            if args.requirements
            else work_file_dir / "requirements.txt"
        )

        print(f"[ENV] Setting up environment '{args.env}'...")
        print(f"[ENV] DataFerret source: {ferret_source}")
        print(f"[ENV] Requirements file: {requirements_file}")
        print()

        if not setup_environment(args.env, ferret_source, requirements_file):
            print("[ERROR] Environment setup failed, aborting")
            return

        print()

    if args.local:
        # Local execution mode
        print(f"[LOCAL MODE] Running {len(targets)} jobs sequentially...")
        print()
        succeeded = 0
        failed = 0
        for i, target_path in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] {target_path}")
            print("-" * 60)
            if run_local_job(target=target_path, args=args):
                succeeded += 1
            else:
                failed += 1
            print()

        print("=" * 60)
        print(f"[SUMMARY] Completed {succeeded + failed}/{len(targets)} jobs")
        print(f"[SUMMARY] ✓ Succeeded: {succeeded}")
        print(f"[SUMMARY] ✗ Failed: {failed}")
    else:
        # SLURM submission mode
        submitted: List[int] = []
        job_targets: Dict[int, Path] = {}
        job_timings_files: Dict[int, Optional[str]] = {}
        log_dir = Path(args.log_dir).resolve()

        for target_path in targets:
            print(target_path)
            print("-----------------------")
            job_id = submit_single_job(target=target_path, args=args)
            if job_id is not None:
                submitted.append(job_id)
                job_targets[job_id] = target_path

                # Build context and command tokens to extract timings file
                abs_target = target_path.expanduser()
                if not abs_target.is_absolute():
                    abs_target = abs_target.resolve(strict=False)
                context = {
                    "target": str(abs_target),
                    "target_dir": str(abs_target.parent),
                    "target_name": abs_target.name,
                    "target_stem": abs_target.stem,
                    "log_dir": str(log_dir),
                }
                cmd_tokens = build_command_tokens(context, args)
                job_timings_files[job_id] = extract_timings_file_from_command(cmd_tokens)
            print()

        if submitted:
            print(f"[OK] Submitted {len(submitted)}/{len(targets)} jobs.")
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
                submitted, job_info, poll_interval=args.poll_interval
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
