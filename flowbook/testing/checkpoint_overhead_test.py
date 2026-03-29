"""
Checkpoint overhead test - Compare baseline execution vs FlowBook (optionally vs Kishu).

Usage:
    python -m flowbook.testing.checkpoint_overhead_test notebook.ipynb
    python -m flowbook.testing.checkpoint_overhead_test notebook.ipynb --kishu
    python -m flowbook.testing.checkpoint_overhead_test notebook.ipynb --reruns 1000
    python -m flowbook.testing.checkpoint_overhead_test --plot-only  # use existing CSV files
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jupyter_client import KernelManager

from jupyter_client.blocking import BlockingKernelClient

from flowbook.cli.helpers import start_kernel
from flowbook.testing.benchmark_checkpoint import (
    _MEMORY_SETUP_CODE,
    measure_memory,
)
from flowbook.testing.notebook_loader import Cell, load_notebook
from flowbook.util.output import log


def create_baseline_kernel() -> tuple[KernelManager, any]:
    """
    Start a baseline python3 kernel.

    Returns:
        Tuple of (KernelManager, BlockingKernelClient)
    """
    return start_kernel(
        "python3",
        client_factory=lambda kid: BlockingKernelClient(),
    )


def cleanup_kernel(kernel_manager, kernel_client) -> None:
    """Clean up kernel resources."""
    if kernel_client:
        try:
            kernel_client.kernel_info()
            time.sleep(0.5)
        except Exception:
            pass
        try:
            kernel_client.stop_channels()
        except Exception:
            pass

    if kernel_manager:
        try:
            kernel_manager.shutdown_kernel()
        except Exception:
            pass


def execute_cell_timed(kernel_client, cell: Cell, timeout: float = 300.0) -> dict:
    """
    Execute a cell and measure execution time.

    Returns:
        Dict with cell_runtime_s and optional error
    """
    start = time.perf_counter()

    msg_id = kernel_client.execute(cell.source)

    error_msg = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return {"cell_runtime_s": None, "error": f"Timeout after {timeout}s"}

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        if msg_type == "error":
            content = msg["content"]
            error_msg = "\n".join(content.get("traceback", []))

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Get execute_reply
    try:
        reply = kernel_client.get_shell_msg(timeout=1.0)
        if reply["content"]["status"] == "error" and error_msg is None:
            error_content = reply["content"]
            error_msg = "\n".join(error_content.get("traceback", []))
    except Exception:
        pass

    elapsed = time.perf_counter() - start

    return {"cell_runtime_s": elapsed, "error": error_msg}


def _setup_baseline_memory(kernel_client, timeout: float = 60.0) -> bool:
    """Inject pympler measurement helper into baseline kernel.

    Uses the same helper as FlowBook; checkpoint_bytes will be 0
    since baseline has no _flowbook_checkpoint.
    """
    msg_id = kernel_client.execute(_MEMORY_SETUP_CODE, silent=True)
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return False
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'error':
            return False
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass
    return True


def run_baseline(notebook_path: str, output_csv: str, cell_timeout: float = 300.0) -> None:
    """
    Run notebook on baseline python3 kernel and save timings to CSV.
    """
    cells = load_notebook(notebook_path)
    log(f"Baseline: Loaded {len(cells)} code cells")

    kernel_manager = None
    kernel_client = None

    try:
        log("Baseline: Starting python3 kernel...")
        kernel_manager, kernel_client = create_baseline_kernel()
        log("Baseline: Kernel ready")

        # Setup memory measurement
        if _setup_baseline_memory(kernel_client):
            log("Baseline: Memory measurement helper injected")
        else:
            log("Baseline: WARNING: Failed to inject memory measurement helper")

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cell_id", "cell_runtime_s", "commit_time_s",
                "user_ns_bytes", "user_ns_and_checkpoint_bytes",
            ])

            for i, cell in enumerate(cells):
                log(f"Baseline: Executing cell {i+1}/{len(cells)} ({cell.cell_id})...")
                timing = execute_cell_timed(kernel_client, cell, cell_timeout)

                if timing.get("error"):
                    log(f"  Error: {timing['error'][:100]}...")
                else:
                    mem = measure_memory(kernel_client)
                    # Baseline has no commit time
                    writer.writerow([
                        cell.cell_id, timing["cell_runtime_s"], 0.0,
                        mem["user_ns_bytes"], mem["user_ns_and_checkpoint_bytes"],
                    ])
                    log(f"  Run: {timing['cell_runtime_s']*1000:.1f}ms")
                    log(f"  Memory: user_ns={mem['user_ns_bytes']:,}B, with_checkpoints={mem['user_ns_and_checkpoint_bytes']:,}B")

        log(f"Baseline: Results written to {output_csv}")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)


def run_flowbook_benchmark(
    notebook_path: str,
    output_csv: str,
    num_reruns: int = 0,
    rerun_modifications: int = 3,
    rerun_output_csv: str = "flowbook_rerun_timings.csv",
    rerun_seed: Optional[int] = None,
) -> None:
    """Run flowbook checkpoint benchmark."""
    log("FlowBook: Running benchmark...")

    cmd = [
        sys.executable, "-m", "flowbook.testing.benchmark_checkpoint",
        notebook_path, "-o", output_csv
    ]

    if num_reruns > 0:
        cmd.extend([
            "--reruns", str(num_reruns),
            "--modifications", str(rerun_modifications),
            "--rerun-output", rerun_output_csv,
        ])
        if rerun_seed is not None:
            cmd.extend(["--seed", str(rerun_seed)])

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        log(f"FlowBook benchmark failed with exit code {result.returncode}")
    log(f"FlowBook: Results written to {output_csv}")
    if num_reruns > 0:
        log(f"FlowBook: Rerun results written to {rerun_output_csv}")


def run_kishu_benchmark(notebook_path: str, output_csv: str) -> None:
    """Run kishu benchmark."""
    log("Kishu: Running benchmark...")
    result = subprocess.run(
        ["kishu-benchmark", notebook_path, "-o", output_csv],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        log(f"Kishu benchmark stderr: {result.stderr}")
        log(f"Kishu benchmark stdout: {result.stdout}")
    log(f"Kishu: Results written to {output_csv}")


def plot_single_runtime(ax, csv_path: str, colors, title: str) -> None:
    """Plot cumulative runtime only (no commit time)."""
    df = pd.read_csv(csv_path)
    df["cumulative_runtime"] = df["cell_runtime_s"].cumsum()
    df["cell"] = range(1, len(df) + 1)

    ax.fill_between(
        df["cell"],
        0,
        df["cumulative_runtime"],
        alpha=0.7,
        label="Cell Run Time",
        color=colors[0],
    )
    ax.plot(df["cell"], df["cumulative_runtime"], color=colors[0], linewidth=1.5)

    ax.set_xlabel("Cell Number")
    ax.set_ylabel("Cumulative Time (seconds)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def plot_stacked(ax, csv_path: str, colors, title: str) -> None:
    """Plot stacked cumulative runtime and commit time."""
    df = pd.read_csv(csv_path)
    df["cumulative_runtime"] = df["cell_runtime_s"].cumsum()
    df["cumulative_commit"] = df["commit_time_s"].cumsum()
    df["cumulative_total"] = df["cumulative_runtime"] + df["cumulative_commit"]
    df["cell"] = range(1, len(df) + 1)

    ax.fill_between(
        df["cell"],
        0,
        df["cumulative_runtime"],
        alpha=0.7,
        label="Cell Run Time",
        color=colors[0],
    )
    ax.fill_between(
        df["cell"],
        df["cumulative_runtime"],
        df["cumulative_total"],
        alpha=0.7,
        label="Checkpoint Time",
        color=colors[1],
    )

    ax.plot(df["cell"], df["cumulative_runtime"], color=colors[0], linewidth=1.5)
    ax.plot(df["cell"], df["cumulative_total"], color=colors[1], linewidth=1.5)

    ax.set_xlabel("Cell Number")
    ax.set_ylabel("Cumulative Time (seconds)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def plot_slowdown(ax, baseline_csv: str, checkpoint_csv: str, colors, title: str, large_fonts: bool = False) -> None:
    """Plot slowdown: baseline cell runtimes with checkpoint time stacked on top."""
    df_base = pd.read_csv(baseline_csv)
    df_ckpt = pd.read_csv(checkpoint_csv)

    # Merge by cell_id to align timings properly (fall back to positional if no cell_id)
    if "cell_id" in df_base.columns and "cell_id" in df_ckpt.columns:
        df = df_base[["cell_id", "cell_runtime_s"]].merge(
            df_ckpt[["cell_id", "commit_time_s"]],
            on="cell_id",
            how="inner"
        )
    else:
        # Fall back to positional alignment
        df = pd.DataFrame()
        df["cell_runtime_s"] = df_base["cell_runtime_s"]
        df["commit_time_s"] = df_ckpt["commit_time_s"]
    df["cumulative_runtime"] = df["cell_runtime_s"].cumsum()
    df["cumulative_commit"] = df["commit_time_s"].cumsum()
    df["cumulative_total"] = df["cumulative_runtime"] + df["cumulative_commit"]
    df["cell"] = range(1, len(df) + 1)

    # Font sizes for paper-ready plots
    label_size = 18 if large_fonts else None
    title_size = 20 if large_fonts else None
    legend_size = 16 if large_fonts else None
    tick_size = 14 if large_fonts else None

    # Plot with markers and lighter fill
    ax.fill_between(
        df["cell"],
        0,
        df["cumulative_runtime"],
        alpha=0.3,
        label="Cell Run Time",
        color=colors[0],
    )
    ax.fill_between(
        df["cell"],
        df["cumulative_runtime"],
        df["cumulative_total"],
        alpha=0.3,
        label="Checkpoint Time",
        color=colors[1],
    )

    ax.plot(df["cell"], df["cumulative_runtime"], color=colors[0], linewidth=2, marker='o', markersize=4)
    ax.plot(df["cell"], df["cumulative_total"], color=colors[1], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    if large_fonts:
        ax.tick_params(axis='both', labelsize=tick_size)


def plot_checkpoint_times(
    ax,
    initial_csv: str,
    rerun_csv: Optional[str],
    colors,
    title: str,
    large_fonts: bool = False,
) -> None:
    """
    Plot histogram of all checkpoint times (initial + rerun combined).

    Args:
        ax: Matplotlib axes
        initial_csv: Path to initial run timings CSV
        rerun_csv: Path to rerun timings CSV (optional)
        colors: Seaborn color palette
        title: Plot title
        large_fonts: Use larger fonts for paper-ready plots
    """
    # Font sizes for paper-ready plots
    label_size = 18 if large_fonts else None
    title_size = 20 if large_fonts else None
    tick_size = 14 if large_fonts else None

    # Load initial run data (convert to ms)
    df_initial = pd.read_csv(initial_csv)
    all_times = list(df_initial["commit_time_s"].dropna() * 1000)

    # Add rerun data if available (convert to ms)
    if rerun_csv and os.path.exists(rerun_csv):
        df_rerun = pd.read_csv(rerun_csv)
        all_times.extend(list(df_rerun["commit_time_s"].dropna() * 1000))

    # Plot histogram with percentage on y-axis (orange to match checkpoint time in left plot)
    weights = np.ones(len(all_times)) / len(all_times) * 100
    ax.hist(
        all_times,
        bins=20,
        weights=weights,
        alpha=0.7,
        color=colors[1],
        edgecolor='white',
    )

    ax.set_xlabel("Checkpoint Time (ms)", fontsize=label_size)
    ax.set_ylabel("Percent", fontsize=label_size)
    ax.set_title(f"{title} (N = {len(all_times):,})", fontsize=title_size)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if large_fonts:
        ax.tick_params(axis='both', labelsize=tick_size)


def _has_memory_columns(csv_path: str) -> bool:
    """Check if a CSV file has memory measurement columns."""
    try:
        df = pd.read_csv(csv_path, nrows=0)
        return 'user_ns_bytes' in df.columns and 'user_ns_and_checkpoint_bytes' in df.columns
    except Exception:
        return False


def plot_memory_comparison(
    ax,
    flowbook_csv: str,
    colors,
    title: str,
    large_fonts: bool = False,
) -> None:
    """Plot memory usage from the checkpoint run: user_ns + checkpoint overhead.

    Uses user_ns_bytes and user_ns_and_checkpoint_bytes from the FlowBook CSV.
    The difference is the checkpoint overhead.
    Matches the visual style of plot_slowdown (stacked fills + marker lines).
    """
    df = pd.read_csv(flowbook_csv)

    mb = 1024 * 1024

    df['cell'] = range(1, len(df) + 1)
    df['user_mb'] = df['user_ns_bytes'] / mb
    df['total_mb'] = df['user_ns_and_checkpoint_bytes'] / mb

    # Font sizes for paper-ready plots
    label_size = 18 if large_fonts else None
    title_size = 20 if large_fonts else None
    legend_size = 16 if large_fonts else None
    tick_size = 14 if large_fonts else None

    # Same structure as plot_slowdown:
    # Blue fill = user namespace (analogous to cell run time)
    # Orange fill = checkpoint overhead stacked on top
    ax.fill_between(
        df['cell'], 0, df['user_mb'],
        alpha=0.3, color=colors[0], label='User Namespace',
    )
    ax.fill_between(
        df['cell'], df['user_mb'], df['total_mb'],
        alpha=0.3, color=colors[1], label='Checkpoint Overhead',
    )

    ax.plot(df['cell'], df['user_mb'], color=colors[0], linewidth=2, marker='o', markersize=4)
    ax.plot(df['cell'], df['total_mb'], color=colors[1], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Cell Number', fontsize=label_size)
    ax.set_ylabel('Memory (MB)', fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    ax.legend(loc='upper left', fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    if large_fonts:
        ax.tick_params(axis='both', labelsize=tick_size)


def print_summary_statistics(
    baseline_csv: str,
    flowbook_csv: str,
    rerun_csv: Optional[str] = None,
) -> None:
    """Print summary statistics for the benchmark results."""
    df_base = pd.read_csv(baseline_csv)
    df_flow = pd.read_csv(flowbook_csv)

    # Slowdown statistics
    total_baseline_runtime = df_base["cell_runtime_s"].sum()
    total_checkpoint_time = df_flow["commit_time_s"].sum()
    total_with_checkpoint = total_baseline_runtime + total_checkpoint_time
    overhead_pct = (total_checkpoint_time / total_baseline_runtime) * 100 if total_baseline_runtime > 0 else 0

    log("")
    log("=" * 60)
    log("SUMMARY STATISTICS")
    log("=" * 60)
    log("")
    log("Cumulative Times:")
    log(f"  Total cell runtime:     {total_baseline_runtime*1000:,.1f} ms ({total_baseline_runtime:.2f} s)")
    log(f"  Total checkpoint time:  {total_checkpoint_time*1000:,.1f} ms ({total_checkpoint_time:.2f} s)")
    log(f"  Total with checkpoint:  {total_with_checkpoint*1000:,.1f} ms ({total_with_checkpoint:.2f} s)")
    log(f"  Checkpoint overhead:    {overhead_pct:.1f}%")
    log("")

    # Checkpoint time statistics (combine initial + rerun)
    all_checkpoint_times_ms = list(df_flow["commit_time_s"].dropna() * 1000)

    if rerun_csv and os.path.exists(rerun_csv):
        df_rerun = pd.read_csv(rerun_csv)
        all_checkpoint_times_ms.extend(list(df_rerun["commit_time_s"].dropna() * 1000))

    if all_checkpoint_times_ms:
        times = np.array(all_checkpoint_times_ms)
        log(f"Checkpoint Time Distribution (N = {len(times):,}):")
        log(f"  Mean:    {np.mean(times):,.2f} ms")
        log(f"  Median:  {np.median(times):,.2f} ms")
        log(f"  Std:     {np.std(times):,.2f} ms")
        log(f"  Min:     {np.min(times):,.2f} ms")
        log(f"  Max:     {np.max(times):,.2f} ms")
        log(f"  P25:     {np.percentile(times, 25):,.2f} ms")
        log(f"  P75:     {np.percentile(times, 75):,.2f} ms")
        log(f"  P95:     {np.percentile(times, 95):,.2f} ms")

    # Memory statistics (from FlowBook run only)
    mb = 1024 * 1024
    if _has_memory_columns(flowbook_csv):
        flow_user_final = df_flow["user_ns_bytes"].iloc[-1]
        flow_total_final = df_flow["user_ns_and_checkpoint_bytes"].iloc[-1]
        flow_cp_overhead = flow_total_final - flow_user_final
        log("")
        log("Memory Usage (final cell):")
        log(f"  User namespace:         {flow_user_final/mb:,.1f} MB")
        log(f"  With checkpoints:       {flow_total_final/mb:,.1f} MB")
        log(f"  Checkpoint overhead:    {flow_cp_overhead/mb:,.1f} MB (cross-checkpoint deduped)")
        if flow_user_final > 0:
            mem_overhead_pct = (flow_cp_overhead / flow_user_final) * 100
            log(f"  Checkpoint overhead:    {mem_overhead_pct:.1f}%")

    log("")
    log("=" * 60)
    log("")


def create_comparison_plot(
    baseline_csv: str,
    flowbook_csv: str,
    output_path: str,
    kishu_csv: Optional[str] = None,
    rerun_csv: Optional[str] = None,
    large_fonts: bool = False,
    show_checkpoint_dist: bool = False,
) -> None:
    """
    Create slowdown comparison plot.

    Builds a list of panels from left to right:
    - Always: FlowBook slowdown
    - If Kishu: Kishu slowdown
    - If --checkpoint-dist and rerun data: checkpoint time histogram
    - If memory data in both CSVs: memory comparison
    """
    # Print summary statistics
    print_summary_statistics(baseline_csv, flowbook_csv, rerun_csv)

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette()

    has_rerun = rerun_csv and os.path.exists(rerun_csv)
    has_memory = _has_memory_columns(flowbook_csv)

    # Collect panels as (plot_func, args) tuples
    panels = []

    # Always include FlowBook slowdown
    if kishu_csv:
        panels.append(
            lambda ax: plot_slowdown(ax, baseline_csv, flowbook_csv, colors, "Cumulative Times (FlowBook)", large_fonts)
        )
        panels.append(
            lambda ax: plot_slowdown(ax, baseline_csv, kishu_csv, colors, "Cumulative Times (Kishu)", large_fonts)
        )
    else:
        panels.append(
            lambda ax: plot_slowdown(ax, baseline_csv, flowbook_csv, colors, "Cumulative Cell Run and Checkpointing Times", large_fonts)
        )

    if show_checkpoint_dist and has_rerun:
        panels.append(
            lambda ax: plot_checkpoint_times(ax, flowbook_csv, rerun_csv, colors, "Per-Cell Checkpoint Times", large_fonts)
        )

    if has_memory:
        panels.append(
            lambda ax: plot_memory_comparison(ax, flowbook_csv, colors, "Memory Usage", large_fonts)
        )

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, draw in zip(axes, panels):
        draw(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    log(f"Plot saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare checkpoint overhead: baseline vs FlowBook (optionally vs Kishu)"
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        help="Path to notebook file (.ipynb)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output plot file (default: <notebook_name>.pdf or checkpoint_overhead.pdf)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout per cell in seconds (default: 300)"
    )
    parser.add_argument(
        "--kishu",
        action="store_true",
        help="Include Kishu benchmark in comparison"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip running benchmarks, use existing CSV files"
    )
    parser.add_argument(
        "--flowbook-only",
        action="store_true",
        help="Only rerun the FlowBook checkpoint benchmark (skip baseline, reuse existing baseline CSV)"
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=0,
        help="Number of rerun measurements to take (default: 0 = skip)"
    )
    parser.add_argument(
        "--modifications",
        type=int,
        default=3,
        help="Number of variables to modify per rerun (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for rerun cell selection (default: None)"
    )
    parser.add_argument(
        "--checkpoint-dist",
        action="store_true",
        help="Show checkpoint time distribution histogram panel"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.plot_only and not args.notebook:
        parser.error("notebook is required unless --plot-only is specified")

    if args.flowbook_only and not os.path.exists("baseline_timings.csv"):
        parser.error("--flowbook-only requires an existing baseline_timings.csv")

    # Determine output filename
    if args.output:
        output_path = args.output
    elif args.notebook:
        notebook_basename = os.path.splitext(os.path.basename(args.notebook))[0]
        output_path = f"{notebook_basename}.pdf"
    else:
        output_path = "checkpoint_overhead.pdf"

    # CSV files
    baseline_csv = "baseline_timings.csv"
    flowbook_csv = "flowbook_timings.csv"
    rerun_csv = "flowbook_rerun_timings.csv"

    # Run benchmarks unless --plot-only
    if not args.plot_only:
        if not args.flowbook_only:
            run_baseline(args.notebook, baseline_csv, args.timeout)
        else:
            log("Skipping baseline (--flowbook-only), reusing existing baseline_timings.csv")
        run_flowbook_benchmark(
            args.notebook,
            flowbook_csv,
            num_reruns=args.reruns,
            rerun_modifications=args.modifications,
            rerun_output_csv=rerun_csv,
            rerun_seed=args.seed,
        )

    kishu_csv = None
    if args.kishu:
        kishu_csv = "kishu_timings.csv"
        if not args.plot_only:
            run_kishu_benchmark(args.notebook, kishu_csv)

    # Determine if rerun data exists (check file exists for plot-only mode)
    rerun_csv_path = None
    if args.reruns > 0 or (args.plot_only and os.path.exists(rerun_csv)):
        rerun_csv_path = rerun_csv

    # Create comparison plot
    create_comparison_plot(
        baseline_csv, flowbook_csv, output_path, kishu_csv,
        rerun_csv=rerun_csv_path,
        large_fonts=args.plot_only,
        show_checkpoint_dist=args.checkpoint_dist,
    )


if __name__ == "__main__":
    main()
