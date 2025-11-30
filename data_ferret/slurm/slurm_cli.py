#!/usr/bin/env python3
"""
enqueue_ferret.py
Submit ONE Slurm job per directory listed in nb.txt (one per line).
Each job:
  - runs in that directory (via --chdir)
  - conda activate <env>
  - ferret batch <dir_basename>
Resources: 1 task, configurable CPUs/GPUs.

Usage:
  python enqueue_ferret.py --partition gpu --time 24:00:00 --mem 32G nb.txt
"""

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit one Slurm job per directory (from nb.txt)"
    )
    p.add_argument(
        "--partition", type=str, default="gpmoo-b", help="Slurm partition (e.g., gpu)"
    )
    p.add_argument(
        "--time",
        dest="time_limit",
        type=str,
        default="24:00:00",
        help="Time limit, e.g., 24:00:00",
    )
    p.add_argument("--mem", type=str, default="16G", help="Memory request (default: 16G)")
    p.add_argument("--cpus", type=int, default=4, help="CPUs per task (default: 4)")
    p.add_argument("--gpus", type=int, default=1, help="GPUs per task (default: 1)")
    p.add_argument(
        "--job-name",
        default="ferret-batch",
        help="Slurm job name prefix (default: ferret-batch)",
    )
    p.add_argument(
        "--env",
        default="ferret",
        help="Conda environment to use (default: ferret)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    p.add_argument(
        "--outdir",
        default="outdir",
        help="Output directory (default: outdir)",
    )
    p.add_argument(
        "--ferret-args",
        default="info",
        help="Arguments to pass to ferret (default: 'info')",
    )
    p.add_argument("nb_file", help="File with notebook paths, one per line")
    return p.parse_args()


def load_valid_nb_paths(nb_file: Path) -> List[Path]:
    if not nb_file.is_file():
        raise FileNotFoundError(f"File not found: {nb_file}")
    dirs: List[Path] = []
    for raw in nb_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        dirs.append(p)
    return dirs


def submit_single_job(
    notebook_file: Path,
    args: argparse.Namespace,
) -> Optional[int]:
    # Per-job script (runs inside workdir due to --chdir)
    notebook_base = notebook_file.stem
    basename = notebook_base.replace(".ipynb", "")

    inner_cmd = f"""
set -euo pipefail
set -x

# ---- Python output & hang diagnostics ----
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1   # dumps tracebacks on fatal signals/timeouts

# ---- Node info ----
# Helpful node info up front
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

# ---- Run ferret ----
srun --cpu-bind=cores ferret --outdir={args.outdir}/{notebook_file.parent} {args.ferret_args} {shlex.quote(str(notebook_file))} 
""".strip()

    sbatch_args = [
        "sbatch",
        f"--job-name={args.job_name}-{basename}",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus}",
        f"--gres=gpu:{args.gpus}",
        "--chdir",
        str(os.getcwd()),
        f"--output={args.outdir}/slurm-{args.job_name}-{basename}.out",
        f"--error={args.outdir}/slurm-{args.job_name}-{basename}.err",
        f"--partition={args.partition}",
        f"--time={args.time_limit}",
        f"--mem={args.mem}",
    ]

    os.makedirs(args.outdir, exist_ok=True)

    wrapped = f"bash -lc {shlex.quote(inner_cmd)}"
    cmd = sbatch_args + ["--wrap", wrapped]

    if args.dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return None

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print(out.strip())
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] sbatch failed for {notebook_file}:\n{e.output}")
        return None

    m = SUBMITTED_RE.search(out)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    nb_paths = load_valid_nb_paths(Path(args.nb_file))
    if not nb_paths:
        print("[INFO] No valid directories found. Nothing to submit.")
        return

    submitted: List[int] = []
    for notebook_path in nb_paths:
        print(f"{notebook_path}")
        print(f"-----------------------")
        jobid = submit_single_job(
            notebook_file=notebook_path,
            args=args,
        )
        if jobid is not None:
            submitted.append(jobid)
        print()

    if submitted:
        print(f"[OK] Submitted {len(submitted)}/{len(nb_paths)} jobs.")
        print("Job IDs:", " ".join(str(j) for j in submitted))
    else:
        print("[WARN] No jobs submitted.")


if __name__ == "__main__":
    main()
