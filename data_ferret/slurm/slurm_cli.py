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
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")


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
              {log_dir}              -> directory where Slurm stdout/stderr files go
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
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task (default: 4)")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per task (default: 1)")
    parser.add_argument(
        "--job-name",
        default="ferret-batch",
        help="Slurm job name prefix (default: ferret-batch)",
    )
    parser.add_argument(
        "-env",
        "--env",
        default="ferret",
        help="Conda environment to activate inside each job (default: ferret)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them",
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


def build_command_tokens(context: dict[str, str], args: argparse.Namespace) -> List[str]:
    """Create the concrete command tokens for the given target context."""
    cmd_tokens = [token.format(**context) for token in args.command_template]
    if args.append_target:
        cmd_tokens.append(context["target_name"])
    return cmd_tokens


def submit_single_job(target: Path, args: argparse.Namespace) -> Optional[int]:
    """Submit the sbatch job for one work item."""
    abs_target = target.expanduser()
    if not abs_target.is_absolute():
        abs_target = abs_target.resolve(strict=False)
    work_dir = abs_target.parent
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "target": str(abs_target),
        "target_dir": str(work_dir),
        "target_name": abs_target.name,
        "target_stem": abs_target.stem,
        "log_dir": str(log_dir),
    }

    basename = abs_target.stem or "target"
    job_label = f"{args.job_name}-{basename}"
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
        srun --cpu-bind=cores {command_str}
        """
    ).strip()

    stdout_path = log_dir / f"slurm-{job_label}.out"
    stderr_path = log_dir / f"slurm-{job_label}.err"

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

    submitted: List[int] = []
    for target_path in targets:
        print(target_path)
        print("-----------------------")
        job_id = submit_single_job(target=target_path, args=args)
        if job_id is not None:
            submitted.append(job_id)
        print()

    if submitted:
        print(f"[OK] Submitted {len(submitted)}/{len(targets)} jobs.")
        print("Job IDs:", " ".join(str(job) for job in submitted))
    else:
        print("[WARN] No jobs submitted.")


if __name__ == "__main__":
    main()
