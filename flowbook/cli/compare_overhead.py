#!/usr/bin/env python3
"""
Process FlowBook baseline comparison JSON files and generate statistics and plots.

Usage:
    flowbook_compare_overhead file1.json file2.json ...
    flowbook_compare_overhead *.json --format table
    flowbook_compare_overhead *.json --format json
    flowbook_compare_overhead *.json --plot
    flowbook_compare_overhead *.json --plot --output-dir plots/
    flowbook_compare_overhead user@server:/path/*.json  # Remote files
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Base directory for cached remote files
CACHE_BASE_DIR = "/tmp/flowbook_compare_overhead"


def parse_remote_path(path: str) -> Tuple[bool, str, str, str]:
    """
    Parse a path to detect if it's a remote path.

    Args:
        path: File path that may be local or remote

    Returns:
        Tuple of (is_remote, user, host, remote_path)
        For local paths, returns (False, '', '', path)
    """
    # Pattern: [user@]host:path
    match = re.match(r'^(?:([^@:]+)@)?([^:/]+):(.+)$', path)
    if match:
        user = match.group(1) or ''
        host = match.group(2)
        remote_path = match.group(3)
        # Exclude Windows drive letters (single letter before colon)
        if len(host) == 1 and host.isalpha():
            return (False, '', '', path)
        return (True, user, host, remote_path)
    return (False, '', '', path)


def get_cache_dir(remote_spec: str) -> str:
    """
    Get the deterministic cache directory for a remote path.

    Args:
        remote_spec: The full remote specification (e.g., "user@host:/path/*.json")

    Returns:
        Path to the cache directory
    """
    hash_val = hashlib.md5(remote_spec.encode()).hexdigest()[:8]
    return os.path.join(CACHE_BASE_DIR, hash_val)


def rsync_remote_files(
    remote_spec: str,
    force_download: bool = False
) -> Tuple[List[str], str]:
    """
    Rsync files from a remote location to local cache.

    Args:
        remote_spec: Remote path specification (e.g., "user@host:/path/*.json")
        force_download: If True, re-download even if cache exists

    Returns:
        Tuple of (list of local file paths, cache directory path)
    """
    is_remote, user, host, remote_path = parse_remote_path(remote_spec)
    if not is_remote:
        raise ValueError(f"Not a remote path: {remote_spec}")

    cache_dir = get_cache_dir(remote_spec)
    remote_host = f"{user}@{host}" if user else host

    # Check if cache exists
    cache_exists = os.path.exists(cache_dir) and os.listdir(cache_dir)

    print(f"Remote: {remote_spec}", file=sys.stderr)
    print(f"Cache:  {cache_dir}/", file=sys.stderr)

    if cache_exists and not force_download:
        print("Status: Using cached files (use --force-download to refresh)", file=sys.stderr)
    else:
        if force_download and cache_exists:
            print("Status: Force downloading (clearing cache)...", file=sys.stderr)
            shutil.rmtree(cache_dir)
        else:
            print("Status: Downloading...", file=sys.stderr)

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Build rsync command
        if '*' in remote_path or '?' in remote_path:
            remote_dir = os.path.dirname(remote_path)
            pattern = os.path.basename(remote_path)

            rsync_cmd = [
                'rsync', '-avz',
                '--include', pattern,
                '--exclude', '*',
                f"{remote_host}:{remote_dir}/",
                cache_dir + "/"
            ]
        else:
            rsync_cmd = [
                'rsync', '-avz',
                f"{remote_host}:{remote_path}",
                cache_dir + "/"
            ]

        try:
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                file_lines = [l for l in lines if l and not l.startswith(('sending', 'sent', 'total', 'receiving'))]
                if file_lines:
                    print(f"Files:  {len(file_lines)} file(s) synced", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error: rsync failed: {e.stderr}", file=sys.stderr)
            raise

    print("", file=sys.stderr)

    # Find the local files matching the pattern
    if '*' in remote_path or '?' in remote_path:
        pattern = os.path.basename(remote_path)
        local_files = glob.glob(os.path.join(cache_dir, pattern))
    else:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(cache_dir, filename)
        local_files = [local_path] if os.path.exists(local_path) else []

    return sorted(local_files), cache_dir


def resolve_file_paths(
    file_paths: List[str],
    force_download: bool = False
) -> List[str]:
    """
    Resolve file paths, downloading remote files as needed.

    Args:
        file_paths: List of local or remote file paths
        force_download: If True, re-download remote files even if cached

    Returns:
        List of local file paths (remote files are replaced with cached copies)
    """
    resolved = []

    for path in file_paths:
        is_remote, _, _, _ = parse_remote_path(path)

        if is_remote:
            local_files, _ = rsync_remote_files(path, force_download)
            resolved.extend(local_files)
        else:
            # Handle local wildcards
            if '*' in path or '?' in path:
                matched = glob.glob(path)
                resolved.extend(sorted(matched))
            else:
                resolved.append(path)

    return resolved


def clear_cache() -> None:
    """Remove all cached remote files."""
    if os.path.exists(CACHE_BASE_DIR):
        shutil.rmtree(CACHE_BASE_DIR)
        print(f"Cleared cache: {CACHE_BASE_DIR}", file=sys.stderr)
    else:
        print(f"Cache directory does not exist: {CACHE_BASE_DIR}", file=sys.stderr)


@dataclass
class FileStats:
    """Statistics for a single comparison file."""
    notebook_path: str
    notebook_name: str
    num_cells: int
    baseline_runtime_ms: float
    flowbook_runtime_ms: float
    state_overhead_ms: float
    check_overhead_ms: float
    flowbook_total_ms: float
    slowdown: float
    state_overhead_pct: float
    check_overhead_pct: float
    baseline_memory_bytes: int
    flowbook_memory_bytes: int
    memory_overhead_bytes: int
    memory_overhead_pct: float
    # Last cell overhead percentages
    last_cell_state_overhead_pct: float = 0.0
    last_cell_check_overhead_pct: float = 0.0
    last_cell_memory_overhead_pct: float = 0.0
    # Rerun stats (optional)
    num_reruns: int = 0
    rerun_baseline_runtime_ms: float = 0.0
    rerun_flowbook_runtime_ms: float = 0.0
    rerun_state_overhead_ms: float = 0.0
    rerun_check_overhead_ms: float = 0.0
    rerun_flowbook_total_ms: float = 0.0
    rerun_final_checkpoint_bytes: int = 0


@dataclass
class AggregateStats:
    """Aggregate statistics across multiple files."""
    num_files: int
    total_cells: int
    slowdown_mean: float
    slowdown_median: float
    slowdown_std: float
    slowdown_min: float
    slowdown_max: float
    slowdown_p90: float
    slowdown_p95: float
    state_overhead_pct_mean: float
    check_overhead_pct_mean: float
    memory_overhead_pct_mean: float


def load_comparison_json(file_path: str) -> Dict[str, Any]:
    """Load and validate a comparison JSON file."""
    with open(file_path) as f:
        data = json.load(f)

    # Validate structure
    if "kernels" not in data:
        raise ValueError(f"Invalid comparison file: missing 'kernels' key in {file_path}")
    if "baseline" not in data["kernels"] or "flowbook" not in data["kernels"]:
        raise ValueError(f"Invalid comparison file: missing baseline or flowbook results in {file_path}")

    return data


def extract_warnings(data: Dict[str, Any]) -> List[str]:
    """Extract all memory warnings from comparison data."""
    warnings = []
    for kernel_name in ["baseline", "flowbook"]:
        kernel_data = data.get("kernels", {}).get(kernel_name, {})
        for cell in kernel_data.get("cells", []) + kernel_data.get("rerun_cells", []):
            cell_warnings = cell.get("memory_warnings") or []
            for w in cell_warnings:
                warnings.append(f"{kernel_name} cell {cell.get('cell_id', '?')}: {w}")
    return warnings


def compute_file_stats(data: Dict[str, Any], file_path: str) -> FileStats:
    """Compute statistics from a single comparison file."""
    notebook_path = data.get("notebook_path", file_path)
    notebook_name = Path(notebook_path).name

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    baseline_totals = baseline.get("totals", {})
    flowbook_totals = flowbook.get("totals", {})

    baseline_runtime = baseline_totals.get("cell_runtime_ms", 0.0)
    flowbook_runtime = flowbook_totals.get("cell_runtime_ms", 0.0)
    state_overhead = flowbook_totals.get("state_duration_ms", 0.0)
    check_overhead = flowbook_totals.get("check_duration_ms", 0.0)
    flowbook_total = flowbook_runtime

    if baseline_runtime > 0:
        slowdown = flowbook_total / baseline_runtime
        state_pct = (state_overhead / baseline_runtime) * 100
        check_pct = (check_overhead / baseline_runtime) * 100
    else:
        slowdown = 0.0
        state_pct = 0.0
        check_pct = 0.0

    baseline_memory = baseline_totals.get("final_user_ns_bytes", 0)
    flowbook_memory = flowbook_totals.get("final_checkpoint_bytes", 0)
    memory_overhead = flowbook_memory - baseline_memory if flowbook_memory > baseline_memory else 0

    if baseline_memory > 0:
        memory_pct = (memory_overhead / baseline_memory) * 100
    else:
        memory_pct = 0.0

    num_cells = data.get("metadata", {}).get("num_cells", len(baseline.get("cells", [])))

    # Rerun stats
    baseline_rerun_totals = baseline.get("rerun_totals", {})
    flowbook_rerun_totals = flowbook.get("rerun_totals", {})
    num_reruns = len(baseline.get("rerun_cells", []))

    rerun_baseline_runtime = baseline_rerun_totals.get("cell_runtime_ms", 0.0)
    rerun_flowbook_runtime = flowbook_rerun_totals.get("cell_runtime_ms", 0.0)
    rerun_state_overhead = flowbook_rerun_totals.get("state_duration_ms", 0.0)
    rerun_check_overhead = flowbook_rerun_totals.get("check_duration_ms", 0.0)
    rerun_flowbook_total = rerun_flowbook_runtime
    rerun_final_checkpoint = flowbook_rerun_totals.get("final_checkpoint_bytes", 0)

    # Last cell overhead calculations
    baseline_cells = baseline.get("cells", [])
    flowbook_cells = flowbook.get("cells", [])

    last_cell_state_pct = 0.0
    last_cell_check_pct = 0.0
    last_cell_memory_pct = 0.0

    if flowbook_cells and baseline_cells:
        last_fc = flowbook_cells[-1]
        last_bc = baseline_cells[-1]

        last_baseline_runtime = last_bc.get("cell_runtime_ms", 0.0)
        last_state = last_fc.get("state_duration_ms", 0.0)
        last_check = last_fc.get("check_duration_ms", 0.0)

        if last_baseline_runtime > 0:
            last_cell_state_pct = (last_state / last_baseline_runtime) * 100
            last_cell_check_pct = (last_check / last_baseline_runtime) * 100

        last_user_ns = last_fc.get("user_ns_bytes", 0)
        # Use checkpoint_details.total_bytes if available
        last_details = last_fc.get("checkpoint_details") or {}
        last_total = last_details.get("total_bytes", 0) or last_fc.get("user_ns_and_checkpoint_bytes", 0)
        if last_user_ns > 0:
            last_cell_memory_pct = ((last_total - last_user_ns) / last_user_ns) * 100

    return FileStats(
        notebook_path=notebook_path,
        notebook_name=notebook_name,
        num_cells=num_cells,
        baseline_runtime_ms=baseline_runtime,
        flowbook_runtime_ms=flowbook_runtime,
        state_overhead_ms=state_overhead,
        check_overhead_ms=check_overhead,
        flowbook_total_ms=flowbook_total,
        slowdown=slowdown,
        state_overhead_pct=state_pct,
        check_overhead_pct=check_pct,
        baseline_memory_bytes=baseline_memory,
        flowbook_memory_bytes=flowbook_memory,
        memory_overhead_bytes=memory_overhead,
        memory_overhead_pct=memory_pct,
        last_cell_state_overhead_pct=last_cell_state_pct,
        last_cell_check_overhead_pct=last_cell_check_pct,
        last_cell_memory_overhead_pct=last_cell_memory_pct,
        num_reruns=num_reruns,
        rerun_baseline_runtime_ms=rerun_baseline_runtime,
        rerun_flowbook_runtime_ms=rerun_flowbook_runtime,
        rerun_state_overhead_ms=rerun_state_overhead,
        rerun_check_overhead_ms=rerun_check_overhead,
        rerun_flowbook_total_ms=rerun_flowbook_total,
        rerun_final_checkpoint_bytes=rerun_final_checkpoint,
    )


def compute_aggregate_stats(stats_list: List[FileStats]) -> AggregateStats:
    """Compute aggregate statistics across multiple files."""
    if not stats_list:
        return AggregateStats(
            num_files=0,
            total_cells=0,
            slowdown_mean=0.0,
            slowdown_median=0.0,
            slowdown_std=0.0,
            slowdown_min=0.0,
            slowdown_max=0.0,
            slowdown_p90=0.0,
            slowdown_p95=0.0,
            state_overhead_pct_mean=0.0,
            check_overhead_pct_mean=0.0,
            memory_overhead_pct_mean=0.0,
        )

    slowdowns = np.array([s.slowdown for s in stats_list])
    # Use last cell overhead percentages for aggregate stats
    state_pcts = np.array([s.last_cell_state_overhead_pct for s in stats_list])
    check_pcts = np.array([s.last_cell_check_overhead_pct for s in stats_list])
    memory_pcts = np.array([s.last_cell_memory_overhead_pct for s in stats_list])

    return AggregateStats(
        num_files=len(stats_list),
        total_cells=sum(s.num_cells for s in stats_list),
        slowdown_mean=float(np.mean(slowdowns)),
        slowdown_median=float(np.median(slowdowns)),
        slowdown_std=float(np.std(slowdowns)),
        slowdown_min=float(np.min(slowdowns)),
        slowdown_max=float(np.max(slowdowns)),
        slowdown_p90=float(np.percentile(slowdowns, 90)),
        slowdown_p95=float(np.percentile(slowdowns, 95)),
        state_overhead_pct_mean=float(np.mean(state_pcts)),
        check_overhead_pct_mean=float(np.mean(check_pcts)),
        memory_overhead_pct_mean=float(np.mean(memory_pcts)),
    )


def format_table(stats_list: List[FileStats], aggregate: AggregateStats) -> str:
    """Format results as ASCII table."""
    lines = []
    lines.append("=" * 100)
    lines.append("FLOWBOOK OVERHEAD COMPARISON")
    lines.append("=" * 100)
    lines.append(f"Notebooks: {aggregate.num_files}")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Notebook':<30} {'Cells':>5} {'Baseline':>10} {'FlowBook':>10} {'State':>8} {'Check':>8} {'Slowdown':>10}"
    lines.append(header)
    lines.append("-" * 100)

    # Per-file rows
    for s in stats_list:
        name = s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
        row = f"{name:<30} {s.num_cells:>5} {s.baseline_runtime_ms:>9.0f}ms {s.flowbook_total_ms:>9.0f}ms {s.state_overhead_ms:>7.0f}ms {s.check_overhead_ms:>7.0f}ms {s.slowdown:>9.2f}x"
        lines.append(row)

    lines.append("-" * 100)
    lines.append("")

    # Show rerun stats if any file has reruns
    has_reruns = any(s.num_reruns > 0 for s in stats_list)
    if has_reruns:
        lines.append("RERUN STATISTICS")
        lines.append("-" * 100)
        header = f"{'Notebook':<30} {'Reruns':>6} {'Baseline':>10} {'FlowBook':>10} {'State':>8} {'Check':>8} {'Final Ckpt':>12}"
        lines.append(header)
        lines.append("-" * 100)
        for s in stats_list:
            if s.num_reruns > 0:
                name = s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
                ckpt_mb = s.rerun_final_checkpoint_bytes / (1024 * 1024)
                row = f"{name:<30} {s.num_reruns:>6} {s.rerun_baseline_runtime_ms:>9.0f}ms {s.rerun_flowbook_total_ms:>9.0f}ms {s.rerun_state_overhead_ms:>7.0f}ms {s.rerun_check_overhead_ms:>7.0f}ms {ckpt_mb:>10.1f}MB"
                lines.append(row)
        lines.append("-" * 100)
        lines.append("")

    # Aggregate statistics
    lines.append(f"AGGREGATE (N={aggregate.num_files})")
    lines.append(f"  Mean Slowdown:      {aggregate.slowdown_mean:.3f}x")
    lines.append(f"  Median Slowdown:    {aggregate.slowdown_median:.3f}x")
    lines.append(f"  Std Dev:            {aggregate.slowdown_std:.3f}")
    lines.append(f"  Min Slowdown:       {aggregate.slowdown_min:.3f}x")
    lines.append(f"  Max Slowdown:       {aggregate.slowdown_max:.3f}x")
    lines.append(f"  P90 Slowdown:       {aggregate.slowdown_p90:.3f}x")
    lines.append(f"  P95 Slowdown:       {aggregate.slowdown_p95:.3f}x")
    lines.append("")
    lines.append(f"  Mean State Overhead:   {aggregate.state_overhead_pct_mean:.1f}%")
    lines.append(f"  Mean Check Overhead:   {aggregate.check_overhead_pct_mean:.1f}%")
    lines.append(f"  Mean Memory Overhead:  {aggregate.memory_overhead_pct_mean:.1f}%")
    lines.append("=" * 100)

    return "\n".join(lines)


def format_json_output(stats_list: List[FileStats], aggregate: AggregateStats) -> str:
    """Format results as JSON."""
    output = {
        "files": [
            {
                "notebook_path": s.notebook_path,
                "notebook_name": s.notebook_name,
                "num_cells": s.num_cells,
                "baseline_runtime_ms": s.baseline_runtime_ms,
                "flowbook_runtime_ms": s.flowbook_runtime_ms,
                "state_overhead_ms": s.state_overhead_ms,
                "check_overhead_ms": s.check_overhead_ms,
                "flowbook_total_ms": s.flowbook_total_ms,
                "slowdown": s.slowdown,
                "state_overhead_pct": s.state_overhead_pct,
                "check_overhead_pct": s.check_overhead_pct,
                "memory_overhead_bytes": s.memory_overhead_bytes,
                "memory_overhead_pct": s.memory_overhead_pct,
            }
            for s in stats_list
        ],
        "aggregate": {
            "num_files": aggregate.num_files,
            "total_cells": aggregate.total_cells,
            "slowdown": {
                "mean": aggregate.slowdown_mean,
                "median": aggregate.slowdown_median,
                "std": aggregate.slowdown_std,
                "min": aggregate.slowdown_min,
                "max": aggregate.slowdown_max,
                "p90": aggregate.slowdown_p90,
                "p95": aggregate.slowdown_p95,
            },
            "state_overhead_pct_mean": aggregate.state_overhead_pct_mean,
            "check_overhead_pct_mean": aggregate.check_overhead_pct_mean,
            "memory_overhead_pct_mean": aggregate.memory_overhead_pct_mean,
        },
    }
    return json.dumps(output, indent=2)


def format_csv(stats_list: List[FileStats], aggregate: AggregateStats) -> str:
    """Format results as CSV."""
    lines = []
    lines.append("notebook,cells,baseline_ms,flowbook_ms,state_ms,check_ms,slowdown,state_pct,check_pct,memory_pct")

    for s in stats_list:
        lines.append(
            f'"{s.notebook_name}",{s.num_cells},{s.baseline_runtime_ms:.1f},{s.flowbook_total_ms:.1f},'
            f'{s.state_overhead_ms:.1f},{s.check_overhead_ms:.1f},{s.slowdown:.3f},'
            f'{s.state_overhead_pct:.1f},{s.check_overhead_pct:.1f},{s.memory_overhead_pct:.1f}'
        )

    return "\n".join(lines)


def plot_slowdown(
    data: Dict[str, Any],
    output_path: str,
    large_fonts: bool = False
) -> None:
    """
    Plot slowdown: baseline cell runtimes with state/check time stacked on top.

    Creates a cumulative time plot showing baseline runtime and FlowBook overhead.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette()

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    baseline_cells = baseline.get("cells", [])
    flowbook_cells = flowbook.get("cells", [])

    # Build aligned data by cell_id
    cell_data = {}
    for c in baseline_cells:
        cell_data[c["cell_id"]] = {"baseline_runtime_ms": c["cell_runtime_ms"]}

    for c in flowbook_cells:
        if c["cell_id"] in cell_data:
            cell_data[c["cell_id"]]["state_ms"] = c["state_duration_ms"]
            cell_data[c["cell_id"]]["check_ms"] = c["check_duration_ms"]

    # Convert to arrays
    cell_ids = list(cell_data.keys())
    baseline_runtimes = [cell_data[cid].get("baseline_runtime_ms", 0) for cid in cell_ids]
    state_times = [cell_data[cid].get("state_ms", 0) for cid in cell_ids]
    check_times = [cell_data[cid].get("check_ms", 0) for cid in cell_ids]

    # Cumulative sums
    baseline_cumsum = np.cumsum(baseline_runtimes)
    state_cumsum = np.cumsum(state_times)
    check_cumsum = np.cumsum(check_times)
    total_cumsum = baseline_cumsum + state_cumsum + check_cumsum

    cells = np.arange(1, len(cell_ids) + 1)

    # Font sizes
    label_size = 18 if large_fonts else None
    title_size = 20 if large_fonts else None
    legend_size = 16 if large_fonts else None
    tick_size = 14 if large_fonts else None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked areas
    ax.fill_between(cells, 0, baseline_cumsum / 1000, alpha=0.3, color=colors[0], label="Cell Run Time")
    ax.fill_between(cells, baseline_cumsum / 1000, (baseline_cumsum + state_cumsum) / 1000, alpha=0.3, color=colors[1], label="State Checkpoint")
    ax.fill_between(cells, (baseline_cumsum + state_cumsum) / 1000, total_cumsum / 1000, alpha=0.3, color=colors[2], label="Reproducibility Check")

    # Lines with markers
    ax.plot(cells, baseline_cumsum / 1000, color=colors[0], linewidth=2, marker='o', markersize=4)
    ax.plot(cells, (baseline_cumsum + state_cumsum) / 1000, color=colors[1], linewidth=2, marker='o', markersize=4)
    ax.plot(cells, total_cumsum / 1000, color=colors[2], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)

    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    ax.set_title(f"Cumulative Cell Run and Checkpointing Times\n{notebook_name}", fontsize=title_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    if large_fonts:
        ax.tick_params(axis='both', labelsize=tick_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Slowdown plot saved to: {output_path}")


def plot_memory_comparison(
    data: Dict[str, Any],
    output_path: str,
    large_fonts: bool = False
) -> None:
    """
    Plot memory usage: user namespace + checkpoint overhead.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette()

    flowbook = data["kernels"]["flowbook"]
    flowbook_cells = flowbook.get("cells", [])

    if not flowbook_cells:
        print("No cell data available for memory plot")
        return

    cells = np.arange(1, len(flowbook_cells) + 1)
    user_ns_bytes = [c.get("user_ns_bytes", 0) for c in flowbook_cells]
    total_bytes = [c.get("user_ns_and_checkpoint_bytes", 0) for c in flowbook_cells]

    mb = 1024 * 1024
    user_mb = np.array(user_ns_bytes) / mb
    total_mb = np.array(total_bytes) / mb

    # Font sizes
    label_size = 18 if large_fonts else None
    title_size = 20 if large_fonts else None
    legend_size = 16 if large_fonts else None
    tick_size = 14 if large_fonts else None

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(cells, 0, user_mb, alpha=0.3, color=colors[0], label='User Namespace')
    ax.fill_between(cells, user_mb, total_mb, alpha=0.3, color=colors[1], label='Checkpoint Overhead')

    ax.plot(cells, user_mb, color=colors[0], linewidth=2, marker='o', markersize=4)
    ax.plot(cells, total_mb, color=colors[1], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Cell Number', fontsize=label_size)
    ax.set_ylabel('Memory (MB)', fontsize=label_size)

    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    ax.set_title(f'Memory Usage\n{notebook_name}', fontsize=title_size)
    ax.legend(loc='upper left', fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    if large_fonts:
        ax.tick_params(axis='both', labelsize=tick_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Memory plot saved to: {output_path}")


def extract_checkpoint_type_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract checkpoint type breakdown data from comparison JSON.

    Returns dict with:
        cells: list of cell indices
        by_type: dict mapping type name to list of bytes per cell
        types_ordered: list of type names ordered by total size descending
        total_bytes: list of total checkpoint bytes per cell
        initial_count: number of initial execution cells (for separator)

    Returns None if no checkpoint_details data available.
    """
    flowbook = data.get("kernels", {}).get("flowbook", {})
    flowbook_cells = flowbook.get("cells", [])
    rerun_cells = flowbook.get("rerun_cells", [])

    # Combine initial + rerun cells
    all_cells = flowbook_cells + rerun_cells
    initial_count = len(flowbook_cells)

    # Check if any cell has checkpoint_details
    has_details = any(c.get("checkpoint_details") for c in all_cells)
    if not has_details:
        return None

    # First pass: collect all type names
    all_type_names: set = set()
    for c in all_cells:
        details = c.get("checkpoint_details") or {}
        by_type = details.get("by_type", {})
        all_type_names.update(by_type.keys())

    # Initialize type tracking
    type_totals: Dict[str, int] = {t: 0 for t in all_type_names}
    type_by_cell: Dict[str, List[int]] = {t: [] for t in all_type_names}
    total_bytes: List[int] = []

    # Second pass: collect data
    for c in all_cells:
        details = c.get("checkpoint_details") or {}
        by_type = details.get("by_type", {})
        cell_total = details.get("total_bytes", 0)
        total_bytes.append(cell_total)

        # Add value for each type (0 if not present in this cell)
        for type_name in all_type_names:
            if type_name in by_type:
                type_bytes = by_type[type_name].get("bytes", 0)
                type_totals[type_name] += type_bytes
                type_by_cell[type_name].append(type_bytes)
            else:
                type_by_cell[type_name].append(0)

    # Order types by total size descending
    types_ordered = sorted(type_totals.keys(), key=lambda t: type_totals[t], reverse=True)

    # Limit to top types, aggregate rest as "other"
    TOP_N = 6
    if len(types_ordered) > TOP_N:
        top_types = types_ordered[:TOP_N]
        other_types = types_ordered[TOP_N:]

        # Aggregate "other"
        other_by_cell = [0] * len(all_cells)
        for t in other_types:
            for i, v in enumerate(type_by_cell[t]):
                other_by_cell[i] += v
            del type_by_cell[t]

        type_by_cell["other"] = other_by_cell
        types_ordered = top_types + ["other"]

    return {
        "cells": list(range(1, len(all_cells) + 1)),
        "by_type": type_by_cell,
        "types_ordered": types_ordered,
        "total_bytes": total_bytes,
        "initial_count": initial_count,
    }


def plot_checkpoint_types(
    data: Dict[str, Any],
    output_path: Optional[str] = None,
    large_fonts: bool = False
) -> Optional[Any]:
    """
    Plot checkpoint memory usage broken down by variable type.

    Creates a stacked area plot showing memory by type over cells.

    If output_path is provided, saves the plot and returns None.
    If output_path is None, returns the figure for use with PdfPages.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    type_data = extract_checkpoint_type_data(data)
    if type_data is None:
        print("No checkpoint_details data available for type breakdown plot")
        return None

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("husl", len(type_data["types_ordered"]))

    cells = np.array(type_data["cells"])
    mb = 1024 * 1024

    # Build stacked data
    stacked = []
    for t in type_data["types_ordered"]:
        stacked.append(np.array(type_data["by_type"][t]) / mb)

    # Font sizes
    label_size = 18 if large_fonts else None
    title_size = 20 if large_fonts else None
    legend_size = 12 if large_fonts else None
    tick_size = 14 if large_fonts else None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked areas
    cumulative = np.zeros(len(cells))
    for i, (t, data_mb) in enumerate(zip(type_data["types_ordered"], stacked)):
        ax.fill_between(cells, cumulative, cumulative + data_mb, alpha=0.7, color=colors[i], label=t)
        cumulative = cumulative + data_mb

    # Total line on top
    ax.plot(cells, cumulative, color='black', linewidth=1.5, linestyle='--', label='Total')

    # Add separator for rerun phase if present
    initial_count = type_data.get("initial_count", len(cells))
    if initial_count < len(cells):
        ax.axvline(x=initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Checkpoint Memory (MB)", fontsize=label_size)

    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    title = f"Checkpoint Memory by Type\n{notebook_name}"
    if initial_count < len(cells):
        title += f" (cells 1-{initial_count} + {len(cells) - initial_count} reruns)"
    ax.set_title(title, fontsize=title_size)
    ax.legend(loc="upper left", fontsize=legend_size, ncol=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    if large_fonts:
        ax.tick_params(axis='both', labelsize=tick_size)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Checkpoint types plot saved to: {output_path}")
        return None
    else:
        return fig


def plot_combined(
    data: Dict[str, Any],
    output_path: Optional[str] = None,
    large_fonts: bool = True
) -> Optional[Any]:
    """
    Create combined multi-panel plot (time + memory + checkpoint types).

    If output_path is provided, saves the plot and returns None.
    If output_path is None, returns the figure for use with PdfPages.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette()

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    baseline_cells = baseline.get("cells", [])
    flowbook_cells = flowbook.get("cells", [])
    baseline_rerun_cells = baseline.get("rerun_cells", [])
    flowbook_rerun_cells = flowbook.get("rerun_cells", [])

    initial_count = len(baseline_cells)

    # Build aligned data for initial cells
    cell_data = {}
    for c in baseline_cells:
        cell_data[c["cell_id"]] = {"baseline_runtime_ms": c["cell_runtime_ms"]}

    for c in flowbook_cells:
        if c["cell_id"] in cell_data:
            cell_data[c["cell_id"]]["flowbook_runtime_ms"] = c["cell_runtime_ms"]
            cell_data[c["cell_id"]]["state_ms"] = c["state_duration_ms"]
            cell_data[c["cell_id"]]["check_ms"] = c["check_duration_ms"]
            cell_data[c["cell_id"]]["user_ns_bytes"] = c.get("user_ns_bytes", 0)
            cell_data[c["cell_id"]]["total_bytes"] = c.get("user_ns_and_checkpoint_bytes", 0)

    cell_ids = list(cell_data.keys())
    baseline_runtimes = [cell_data[cid].get("baseline_runtime_ms", 0) for cid in cell_ids]
    flowbook_runtimes = [cell_data[cid].get("flowbook_runtime_ms", 0) for cid in cell_ids]
    state_times = [cell_data[cid].get("state_ms", 0) for cid in cell_ids]
    check_times = [cell_data[cid].get("check_ms", 0) for cid in cell_ids]
    user_ns_bytes = [cell_data[cid].get("user_ns_bytes", 0) for cid in cell_ids]
    total_bytes = [cell_data[cid].get("total_bytes", 0) for cid in cell_ids]

    # Add rerun cells data
    for i, (bc, fc) in enumerate(zip(baseline_rerun_cells, flowbook_rerun_cells)):
        baseline_runtimes.append(bc.get("cell_runtime_ms", 0))
        flowbook_runtimes.append(fc.get("cell_runtime_ms", 0))
        state_times.append(fc.get("state_duration_ms", 0))
        check_times.append(fc.get("check_duration_ms", 0))
        user_ns_bytes.append(fc.get("user_ns_bytes", 0))
        total_bytes.append(fc.get("user_ns_and_checkpoint_bytes", 0))

    cells = np.arange(1, len(baseline_runtimes) + 1)

    # Cumulative times - use actual measured times for both kernels
    baseline_cumsum = np.cumsum(baseline_runtimes)
    flowbook_cumsum = np.cumsum(flowbook_runtimes)

    # Memory
    mb = 1024 * 1024
    user_mb = np.array(user_ns_bytes) / mb
    total_mb = np.array(total_bytes) / mb

    # Font sizes
    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 16 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    # Check if checkpoint type data is available
    type_data = extract_checkpoint_type_data(data)
    n_panels = 3 if type_data is not None else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))

    # Panel 1: Time comparison (Baseline vs FlowBook)
    ax = axes[0]
    ax.fill_between(cells, 0, baseline_cumsum / 1000, alpha=0.3, color=colors[0], label="Baseline")
    ax.fill_between(cells, baseline_cumsum / 1000, flowbook_cumsum / 1000, alpha=0.3, color=colors[1], label="FlowBook Overhead")

    ax.plot(cells, baseline_cumsum / 1000, color=colors[0], linewidth=2, marker='o', markersize=4)
    ax.plot(cells, flowbook_cumsum / 1000, color=colors[1], linewidth=2, marker='o', markersize=4)

    # Add separator for rerun phase
    if initial_count < len(cells):
        ax.axvline(x=initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)
    title = "Cumulative Runtime"
    if initial_count < len(cells):
        title += f" (cells 1-{initial_count} + {len(cells) - initial_count} reruns)"
    ax.set_title(title, fontsize=title_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Add overhead percentage label at end of run
    if baseline_cumsum[-1] > 0:
        time_overhead_pct = (flowbook_cumsum[-1] - baseline_cumsum[-1]) / baseline_cumsum[-1] * 100
        ax.annotate(f'{time_overhead_pct:.1f}% overhead',
                    xy=(cells[-1], flowbook_cumsum[-1] / 1000),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=legend_size, va='center', ha='left',
                    color=colors[1])

    # Panel 2: Memory
    ax = axes[1]
    ax.fill_between(cells, 0, user_mb, alpha=0.3, color=colors[0], label='User Namespace')
    ax.fill_between(cells, user_mb, total_mb, alpha=0.3, color=colors[1], label='Checkpoint Overhead')

    ax.plot(cells, user_mb, color=colors[0], linewidth=2, marker='o', markersize=4)
    ax.plot(cells, total_mb, color=colors[1], linewidth=2, marker='o', markersize=4)

    # Add separator for rerun phase
    if initial_count < len(cells):
        ax.axvline(x=initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

    ax.set_xlabel('Cell Number', fontsize=label_size)
    ax.set_ylabel('Memory (MB)', fontsize=label_size)
    title = 'Memory Usage'
    if initial_count < len(cells):
        title += f' (cells 1-{initial_count} + {len(cells) - initial_count} reruns)'
    ax.set_title(title, fontsize=title_size)
    ax.legend(loc='upper left', fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Add overhead percentage label at end of run
    if user_mb[-1] > 0:
        mem_overhead_pct = (total_mb[-1] - user_mb[-1]) / user_mb[-1] * 100
        ax.annotate(f'{mem_overhead_pct:.1f}% overhead',
                    xy=(cells[-1], total_mb[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=legend_size, va='center', ha='left',
                    color=colors[1])

    # Panel 3: Checkpoint types (if data available)
    if type_data is not None:
        ax = axes[2]
        type_colors = sns.color_palette("husl", len(type_data["types_ordered"]))
        type_cells = np.array(type_data["cells"])

        # Build stacked data
        stacked = []
        for t in type_data["types_ordered"]:
            stacked.append(np.array(type_data["by_type"][t]) / mb)

        # Stacked areas
        cumulative = np.zeros(len(type_cells))
        for i, (t, data_mb) in enumerate(zip(type_data["types_ordered"], stacked)):
            ax.fill_between(type_cells, cumulative, cumulative + data_mb, alpha=0.7, color=type_colors[i], label=t)
            cumulative = cumulative + data_mb

        # Total line on top
        ax.plot(type_cells, cumulative, color='black', linewidth=1.5, linestyle='--', label='Total')

        # Add separator for rerun phase if present
        type_initial_count = type_data.get("initial_count", len(type_cells))
        if type_initial_count < len(type_cells):
            ax.axvline(x=type_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Checkpoint Memory (MB)", fontsize=label_size)
        title = "Checkpoint by Type"
        if type_initial_count < len(type_cells):
            title += f" (cells 1-{type_initial_count} + {len(type_cells) - type_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.legend(loc="upper left", fontsize=legend_size - 4, ncol=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=tick_size)

    # Add notebook name as figure title
    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    fig.suptitle(notebook_name, fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Combined plot saved to: {output_path}")
        return None
    else:
        return fig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process FlowBook baseline comparison JSON files"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Comparison JSON files to process (supports remote paths like user@host:/path/*.json)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--sort-by",
        choices=["slowdown", "memory", "runtime", "name"],
        default="slowdown",
        help="Sort files by metric (default: slowdown)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PDF plots for each file"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for plot output files (default: current directory)"
    )
    parser.add_argument(
        "--large-fonts",
        action="store_true",
        help="Use larger fonts for paper-ready plots"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of remote files (ignore cache)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached remote files and exit"
    )

    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        clear_cache()
        return

    # Check for input files
    if not args.files:
        parser.error("No input files specified")

    # Resolve file paths (handles remote files and wildcards)
    try:
        resolved_files = resolve_file_paths(args.files, args.force_download)
    except Exception as e:
        print(f"Error resolving file paths: {e}", file=sys.stderr)
        sys.exit(1)

    if not resolved_files:
        print("Error: No files found matching the specified paths", file=sys.stderr)
        sys.exit(1)

    # Load all files
    stats_list: List[FileStats] = []
    file_data: Dict[str, Dict[str, Any]] = {}

    for file_path in resolved_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        try:
            data = load_comparison_json(file_path)
            stats = compute_file_stats(data, file_path)
            stats_list.append(stats)
            file_data[file_path] = data
            # Print any memory measurement warnings
            warnings = extract_warnings(data)
            for w in warnings:
                print(f"Memory warning ({Path(file_path).name}): {w}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}", file=sys.stderr)
            continue

    if not stats_list:
        print("Error: No valid comparison files found", file=sys.stderr)
        sys.exit(1)

    # Sort
    sort_key = {
        "slowdown": lambda s: s.slowdown,
        "memory": lambda s: s.memory_overhead_pct,
        "runtime": lambda s: s.baseline_runtime_ms,
        "name": lambda s: s.notebook_name,
    }[args.sort_by]
    stats_list.sort(key=sort_key, reverse=(args.sort_by != "name"))

    # Compute aggregate
    aggregate = compute_aggregate_stats(stats_list)

    # Output statistics
    if args.format == "table":
        print(format_table(stats_list, aggregate))
    elif args.format == "json":
        print(format_json_output(stats_list, aggregate))
    elif args.format == "csv":
        print(format_csv(stats_list, aggregate))

    # Generate plots if requested
    if args.plot:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all combined plots (time + memory + types side by side) into one PDF
        combined_figures = []

        for file_path, data in file_data.items():
            # Combined plot with all 3 panels side by side
            try:
                fig = plot_combined(data, output_path=None, large_fonts=args.large_fonts)
                if fig is not None:
                    combined_figures.append(fig)
            except Exception as e:
                print(f"Warning: Could not generate plot for {file_path}: {e}", file=sys.stderr)

        # Save combined plots to a single PDF
        if combined_figures:
            combined_path = output_dir / "all_overhead.pdf"
            with PdfPages(str(combined_path)) as pdf:
                for fig in combined_figures:
                    pdf.savefig(fig, dpi=150)
                    plt.close(fig)
            print(f"Combined overhead plots saved to: {combined_path}")


if __name__ == "__main__":
    main()
