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
    """Load and validate a comparison JSON file (v1.0 or v2.0 format)."""
    with open(file_path) as f:
        data = json.load(f)

    # Validate structure
    if "kernels" not in data:
        raise ValueError(f"Invalid comparison file: missing 'kernels' key in {file_path}")
    if "baseline" not in data["kernels"] or "flowbook" not in data["kernels"]:
        raise ValueError(f"Invalid comparison file: missing baseline or flowbook results in {file_path}")

    # Detect version
    version = data.get("version", "1.0")
    data["_version"] = version

    return data


def is_v2_format(data: Dict[str, Any]) -> bool:
    """Check if data is v2.0 format (with separate timing/memory)."""
    # Check both _version (set by load_comparison_json) and version (in raw JSON)
    version = data.get("_version") or data.get("version", "1.0")
    return str(version).startswith("2")


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
    """Compute statistics from a v2.0 comparison file."""
    notebook_path = data.get("notebook_path", file_path)
    notebook_name = Path(notebook_path).name

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    # v2.0 format has separate timing/memory
    baseline_timing = baseline.get("timing", {})
    flowbook_timing = flowbook.get("timing", {})
    baseline_totals = baseline_timing.get("totals", {}) if baseline_timing else {}
    flowbook_totals = flowbook_timing.get("totals", {}) if flowbook_timing else {}

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

    # Memory from v2.0 Scalene data
    baseline_memory_data = baseline.get("memory", {})
    flowbook_memory_data = flowbook.get("memory", {})
    baseline_mem_totals = baseline_memory_data.get("totals", {}) if baseline_memory_data else {}
    flowbook_mem_totals = flowbook_memory_data.get("totals", {}) if flowbook_memory_data else {}

    # Convert MB to bytes for consistency with old format
    mb_to_bytes = 1024 * 1024
    baseline_memory = int(baseline_mem_totals.get("final_footprint_mb", 0) * mb_to_bytes)
    flowbook_memory = int(flowbook_mem_totals.get("final_footprint_mb", 0) * mb_to_bytes)
    memory_overhead = flowbook_memory - baseline_memory if flowbook_memory > baseline_memory else 0

    if baseline_memory > 0:
        memory_pct = (memory_overhead / baseline_memory) * 100
    else:
        memory_pct = 0.0

    num_cells = data.get("metadata", {}).get("num_cells", 0)
    if num_cells == 0 and baseline_timing:
        num_cells = len(baseline_timing.get("cells", []))

    # Extract rerun statistics from v2.0 format
    rerun_baseline_cells = baseline_timing.get("rerun_cells", []) if baseline_timing else []
    rerun_flowbook_cells = flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    num_reruns = len(rerun_flowbook_cells)

    rerun_baseline_runtime = sum(c.get("cell_runtime_ms", 0) for c in rerun_baseline_cells)
    rerun_flowbook_runtime = sum(c.get("cell_runtime_ms", 0) for c in rerun_flowbook_cells)
    rerun_state_overhead = sum(c.get("state_duration_ms", 0) for c in rerun_flowbook_cells)
    rerun_check_overhead = sum(c.get("check_duration_ms", 0) for c in rerun_flowbook_cells)
    rerun_flowbook_total = rerun_flowbook_runtime

    # Get final checkpoint size from last rerun cell (if available)
    rerun_final_checkpoint = 0
    flowbook_mem_rerun = flowbook_memory_data.get("rerun_cells", []) if flowbook_memory_data else []
    if flowbook_mem_rerun:
        last_rerun_mem = flowbook_mem_rerun[-1]
        overhead_breakdown = last_rerun_mem.get("overhead_breakdown", {})
        if overhead_breakdown:
            rerun_final_checkpoint = int(overhead_breakdown.get("checkpoints_mb", 0) * 1024 * 1024)

    # Last cell overhead from timing data
    baseline_timing_cells = baseline_timing.get("cells", []) if baseline_timing else []
    flowbook_timing_cells = flowbook_timing.get("cells", []) if flowbook_timing else []

    last_cell_state_pct = 0.0
    last_cell_check_pct = 0.0
    last_cell_memory_pct = 0.0

    if flowbook_timing_cells and baseline_timing_cells:
        last_fc = flowbook_timing_cells[-1]
        last_bc = baseline_timing_cells[-1]

        last_baseline_runtime = last_bc.get("cell_runtime_ms", 0.0)
        last_state = last_fc.get("state_duration_ms", 0.0)
        last_check = last_fc.get("check_duration_ms", 0.0)

        if last_baseline_runtime > 0:
            last_cell_state_pct = (last_state / last_baseline_runtime) * 100
            last_cell_check_pct = (last_check / last_baseline_runtime) * 100

    # Memory overhead from memory data
    baseline_mem_cells = baseline_memory_data.get("cells", []) if baseline_memory_data else []
    flowbook_mem_cells = flowbook_memory_data.get("cells", []) if flowbook_memory_data else []

    if flowbook_mem_cells and baseline_mem_cells:
        last_baseline_mem = baseline_mem_cells[-1].get("current_footprint_mb", 0)
        last_flowbook_mem = flowbook_mem_cells[-1].get("current_footprint_mb", 0)
        if last_baseline_mem > 0:
            last_cell_memory_pct = ((last_flowbook_mem - last_baseline_mem) / last_baseline_mem) * 100

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


def extract_checkpoint_type_data(data: Dict[str, Any], top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Extract checkpoint type breakdown data from comparison JSON.

    Args:
        data: Comparison data dict
        top_n: Number of top types to show individually (rest aggregated as "other")

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
    if len(types_ordered) > top_n:
        top_types = types_ordered[:top_n]
        other_types = types_ordered[top_n:]

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


def extract_checkpoint_type_data_v2(data: Dict[str, Any], top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Extract CUMULATIVE checkpoint type breakdown from v2.0 format.

    Uses cumulative_by_type field when available (TRUE cumulative accounting for sharing).
    Falls back to deriving from checkpoint_var_costs for backwards compatibility.

    Args:
        data: Comparison data dict
        top_n: Number of top types to show individually (rest aggregated as "other")

    Returns dict with:
        cells: list of cell indices
        by_type: dict mapping type name to list of CUMULATIVE bytes per cell
        types_ordered: list of type names ordered by total size descending
        total_bytes: list of CUMULATIVE total checkpoint bytes per cell
        initial_count: number of initial cells (before reruns)

    Returns None if no checkpoint data available.
    """
    if not is_v2_format(data):
        return None

    flowbook = data.get("kernels", {}).get("flowbook", {})
    memory_data = flowbook.get("memory", {})
    if not memory_data:
        return None

    initial_cells = memory_data.get("cells", [])
    rerun_cells = memory_data.get("rerun_cells", [])
    memory_cells = initial_cells + rerun_cells
    initial_count = len(initial_cells)

    # Check if we have the new cumulative_by_type field (TRUE cumulative)
    has_cumulative = any(c.get("cumulative_by_type") for c in memory_cells)

    if has_cumulative:
        # Use TRUE cumulative data that accounts for memory sharing
        all_type_names: set = set()
        for c in memory_cells:
            by_type = c.get("cumulative_by_type") or {}
            all_type_names.update(by_type.keys())

        if not all_type_names:
            return None

        type_by_cell: Dict[str, List[int]] = {t: [] for t in all_type_names}
        total_bytes: List[int] = []

        for c in memory_cells:
            by_type = c.get("cumulative_by_type") or {}
            cell_total = 0

            for t in all_type_names:
                type_bytes = by_type.get(t, 0)
                type_by_cell[t].append(type_bytes)
                cell_total += type_bytes

            total_bytes.append(cell_total)

        # Get final cumulative for ordering
        type_final: Dict[str, int] = {t: type_by_cell[t][-1] if type_by_cell[t] else 0 for t in all_type_names}
    else:
        # Fall back to old method - derives from checkpoint_var_costs (may overcount)
        has_costs = any(c.get("checkpoint_var_costs") for c in memory_cells)
        if not has_costs:
            return None

        # First pass: collect all type names
        all_type_names = set()
        for c in memory_cells:
            costs = c.get("checkpoint_var_costs") or {}
            for var_info in costs.values():
                all_type_names.add(var_info.get("type", "unknown"))

        if not all_type_names:
            return None

        # Initialize tracking - cumulative totals
        type_cumulative: Dict[str, int] = {t: 0 for t in all_type_names}
        type_by_cell = {t: [] for t in all_type_names}
        total_bytes = []
        cumulative_total = 0

        # Second pass: collect CUMULATIVE data per cell (OLD method - overcounts sharing)
        for c in memory_cells:
            costs = c.get("checkpoint_var_costs") or {}
            cell_total = 0

            for var_info in costs.values():
                var_type = var_info.get("type", "unknown")
                var_bytes = var_info.get("bytes", 0)
                if var_type in type_cumulative:
                    type_cumulative[var_type] += var_bytes
                cell_total += var_bytes

            cumulative_total += cell_total
            for t in all_type_names:
                type_by_cell[t].append(type_cumulative[t])
            total_bytes.append(cumulative_total)

        type_final = type_cumulative

    # Order types by total (final cumulative) size descending
    types_ordered = sorted(type_final.keys(), key=lambda t: type_final[t], reverse=True)

    # Limit to top types, aggregate rest as "other"
    if len(types_ordered) > top_n:
        top_types = types_ordered[:top_n]
        other_types = types_ordered[top_n:]

        # Aggregate "other" - sum cumulative values from excluded types
        other_by_cell = [0] * len(memory_cells)
        for t in other_types:
            for i, v in enumerate(type_by_cell[t]):
                other_by_cell[i] += v
            del type_by_cell[t]

        type_by_cell["other"] = other_by_cell
        types_ordered = top_types + ["other"]

    return {
        "cells": list(range(1, len(memory_cells) + 1)),
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
    # Use lines only (not shaded areas) since these are independent measurements
    ax = axes[0]
    ax.plot(cells, baseline_cumsum / 1000, color=colors[0], linewidth=2, marker='o', markersize=4, label="Baseline")
    ax.plot(cells, flowbook_cumsum / 1000, color=colors[1], linewidth=2, marker='o', markersize=4, label="FlowBook")

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


def extract_checkpoint_var_data(data: Dict[str, Any], top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Extract CUMULATIVE per-variable checkpoint costs from v2.0 comparison data.

    Uses cumulative_by_var field when available (TRUE cumulative accounting for sharing).
    Falls back to deriving from checkpoint_var_costs for backwards compatibility.

    Args:
        data: Comparison data dict
        top_n: Number of top variables to show individually (rest aggregated as "other")

    Returns dict with:
        cells: list of cell indices
        by_var: dict mapping variable name to list of CUMULATIVE bytes per cell
        vars_ordered: list of variable names ordered by total size descending
        initial_count: number of initial execution cells (before reruns)

    Returns None if no checkpoint data available.
    """
    if not is_v2_format(data):
        return None

    flowbook = data.get("kernels", {}).get("flowbook", {})
    memory_data = flowbook.get("memory", {})
    if not memory_data:
        return None

    initial_cells = memory_data.get("cells", [])
    rerun_cells = memory_data.get("rerun_cells", [])
    memory_cells = initial_cells + rerun_cells
    initial_count = len(initial_cells)

    # Check if we have the new cumulative_by_var field (TRUE cumulative)
    has_cumulative = any(c.get("cumulative_by_var") for c in memory_cells)

    if has_cumulative:
        # Use TRUE cumulative data that accounts for memory sharing
        all_var_names: set = set()
        for c in memory_cells:
            by_var = c.get("cumulative_by_var") or {}
            all_var_names.update(by_var.keys())

        if not all_var_names:
            return None

        var_by_cell: Dict[str, List[int]] = {v: [] for v in all_var_names}

        for c in memory_cells:
            by_var = c.get("cumulative_by_var") or {}

            for var_name in all_var_names:
                var_bytes = by_var.get(var_name, 0)
                var_by_cell[var_name].append(var_bytes)

        # Get final cumulative for ordering
        var_final: Dict[str, int] = {v: var_by_cell[v][-1] if var_by_cell[v] else 0 for v in all_var_names}
    else:
        # Fall back to old method - derives from checkpoint_var_costs (may overcount)
        has_costs = any(c.get("checkpoint_var_costs") for c in memory_cells)
        if not has_costs:
            return None

        # First pass: collect all variable names
        all_var_names = set()
        for c in memory_cells:
            costs = c.get("checkpoint_var_costs") or {}
            all_var_names.update(costs.keys())

        # Initialize tracking - cumulative totals per variable
        var_cumulative: Dict[str, int] = {v: 0 for v in all_var_names}
        var_by_cell = {v: [] for v in all_var_names}

        # Second pass: collect CUMULATIVE data (OLD method - overcounts sharing)
        for c in memory_cells:
            costs = c.get("checkpoint_var_costs") or {}

            for var_name in all_var_names:
                if var_name in costs:
                    var_bytes = costs[var_name].get("bytes", 0)
                    var_cumulative[var_name] += var_bytes
                # Append current cumulative value for this variable
                var_by_cell[var_name].append(var_cumulative[var_name])

        var_final = var_cumulative

    # Order variables by final cumulative total size descending
    vars_ordered = sorted(var_final.keys(), key=lambda v: var_final[v], reverse=True)

    # Limit to top variables, aggregate rest as "other"
    if len(vars_ordered) > top_n:
        top_vars = vars_ordered[:top_n]
        other_vars = vars_ordered[top_n:]

        # Aggregate "other" - sum cumulative values from excluded variables
        other_by_cell = [0] * len(memory_cells)
        for v in other_vars:
            for i, val in enumerate(var_by_cell[v]):
                other_by_cell[i] += val
            del var_by_cell[v]

        var_by_cell["other"] = other_by_cell
        vars_ordered = top_vars + ["other"]

    # Build var_types mapping from checkpoint_var_costs
    var_types: Dict[str, str] = {}
    for c in memory_cells:
        costs = c.get("checkpoint_var_costs") or {}
        for var_name, info in costs.items():
            if var_name not in var_types and isinstance(info, dict):
                var_types[var_name] = info.get("type", "?")

    return {
        "cells": list(range(1, len(memory_cells) + 1)),
        "by_var": var_by_cell,
        "vars_ordered": vars_ordered,
        "var_types": var_types,
        "initial_count": initial_count,
    }


def extract_checkpoint_timing_var_data(data: Dict[str, Any], top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Extract PER-CELL per-variable checkpoint TIMING from v2.0 comparison data.

    Gets deepcopy_ms from checkpoint_var_costs for each cell (not cumulative).

    Args:
        data: Comparison data dict
        top_n: Number of top variables to show individually (rest aggregated as "other")

    Returns dict with:
        cells: list of cell indices
        by_var: dict mapping variable name to list of ms per cell (NOT cumulative)
        vars_ordered: list of variable names ordered by total time descending
        initial_count: number of initial execution cells (before reruns)

    Returns None if no checkpoint timing data available.
    """
    if not is_v2_format(data):
        return None

    flowbook = data.get("kernels", {}).get("flowbook", {})
    memory_data = flowbook.get("memory", {})
    if not memory_data:
        return None

    initial_cells = memory_data.get("cells", [])
    rerun_cells = memory_data.get("rerun_cells", [])
    memory_cells = initial_cells + rerun_cells
    initial_count = len(initial_cells)

    # Get timing from checkpoint_var_costs (deepcopy_ms field)
    has_costs = any(c.get("checkpoint_var_costs") for c in memory_cells)
    if not has_costs:
        return None

    # First pass: collect all variable names and compute totals
    all_var_names: set = set()
    var_total: Dict[str, float] = {}
    for c in memory_cells:
        costs = c.get("checkpoint_var_costs") or {}
        for var_name, info in costs.items():
            all_var_names.add(var_name)
            var_ms = info.get("deepcopy_ms", 0)
            if var_name in var_total:
                var_total[var_name] += var_ms
            else:
                var_total[var_name] = var_ms

    if not all_var_names:
        return None

    # Second pass: collect PER-CELL timing data (not cumulative)
    var_by_cell: Dict[str, List[float]] = {v: [] for v in all_var_names}

    for c in memory_cells:
        costs = c.get("checkpoint_var_costs") or {}

        for var_name in all_var_names:
            if var_name in costs:
                var_ms = costs[var_name].get("deepcopy_ms", 0)
            else:
                var_ms = 0
            var_by_cell[var_name].append(var_ms)

    # Order variables by total time descending
    vars_ordered = sorted(var_total.keys(), key=lambda v: var_total[v], reverse=True)

    # Limit to top variables, aggregate rest as "other"
    if len(vars_ordered) > top_n:
        top_vars = vars_ordered[:top_n]
        other_vars = vars_ordered[top_n:]

        # Aggregate "other" - sum per-cell values from excluded variables
        other_by_cell = [0.0] * len(memory_cells)
        for v in other_vars:
            for i, val in enumerate(var_by_cell[v]):
                other_by_cell[i] += val
            del var_by_cell[v]

        var_by_cell["other"] = other_by_cell
        vars_ordered = top_vars + ["other"]

    # Build var_types mapping from checkpoint_var_costs
    var_types: Dict[str, str] = {}
    for c in memory_cells:
        costs = c.get("checkpoint_var_costs") or {}
        for var_name, info in costs.items():
            if var_name not in var_types and isinstance(info, dict):
                var_types[var_name] = info.get("type", "?")

    return {
        "cells": list(range(1, len(memory_cells) + 1)),
        "by_var": var_by_cell,
        "vars_ordered": vars_ordered,
        "var_types": var_types,
        "initial_count": initial_count,
    }


def plot_combined_v2(
    data: Dict[str, Any],
    output_path: Optional[str] = None,
    large_fonts: bool = True,
    top_n: int = 10
) -> Optional[Any]:
    """
    Create combined 4-panel plot for v2.0 data with HeapSizer memory:
    - Panel 1: Timing (baseline vs flowbook cumulative time)
    - Panel 2: Memory (HeapSizer data)
    - Panel 3: Checkpoint by Type
    - Panel 4: Checkpoint by Name (per-variable)

    Args:
        data: Comparison data dict
        output_path: If provided, saves to file; otherwise returns figure
        large_fonts: Use larger fonts for paper-ready plots
        top_n: Number of top types/variables to show individually

    If output_path is provided, saves the plot and returns None.
    If output_path is None, returns the figure for use with PdfPages.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette()

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    # Extract timing data (including reruns)
    baseline_timing = baseline.get("timing", {})
    flowbook_timing = flowbook.get("timing", {})
    baseline_initial_cells = baseline_timing.get("cells", []) if baseline_timing else []
    flowbook_initial_cells = flowbook_timing.get("cells", []) if flowbook_timing else []
    baseline_rerun_cells = baseline_timing.get("rerun_cells", []) if baseline_timing else []
    flowbook_rerun_cells = flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    baseline_cells = baseline_initial_cells + baseline_rerun_cells
    flowbook_cells = flowbook_initial_cells + flowbook_rerun_cells
    timing_initial_count = len(baseline_initial_cells)

    # Extract memory data (including reruns)
    baseline_memory = baseline.get("memory", {})
    flowbook_memory = flowbook.get("memory", {})
    baseline_mem_initial = baseline_memory.get("cells", []) if baseline_memory else []
    flowbook_mem_initial = flowbook_memory.get("cells", []) if flowbook_memory else []
    baseline_mem_rerun = baseline_memory.get("rerun_cells", []) if baseline_memory else []
    flowbook_mem_rerun = flowbook_memory.get("rerun_cells", []) if flowbook_memory else []
    baseline_mem_cells = baseline_mem_initial + baseline_mem_rerun
    flowbook_mem_cells = flowbook_mem_initial + flowbook_mem_rerun
    memory_initial_count = len(baseline_mem_initial)

    # Font sizes
    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    # Determine number of panels
    has_memory = bool(baseline_mem_cells and flowbook_mem_cells)
    # Extract timing by variable for Panel 3
    timing_var_data = extract_checkpoint_timing_var_data(data, top_n=top_n)
    # Extract memory by variable for Panel 4
    var_data = extract_checkpoint_var_data(data, top_n=top_n)

    n_panels = 1  # Always have timing
    if has_memory:
        n_panels += 1
    if timing_var_data:
        n_panels += 1
    if var_data:
        n_panels += 1

    # Use 2x2 grid layout for 4 panels, otherwise single row
    if n_panels == 4:
        fig, axes_2d = plt.subplots(2, 2, figsize=(14, 12))
        axes = [axes_2d[0, 0], axes_2d[0, 1], axes_2d[1, 0], axes_2d[1, 1]]
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

    panel_idx = 0

    # ========== Panel 1: Timing ==========
    if baseline_cells and flowbook_cells:
        ax = axes[panel_idx]
        panel_idx += 1

        # Align by cell_id - use execute_duration_ms and code_duration_ms
        cell_data = {}
        for c in baseline_cells:
            # Baseline: execute_duration_ms equals code_duration_ms (no FlowBook overhead)
            cell_data[c["cell_id"]] = {"baseline_ms": c.get("execute_duration_ms", c.get("cell_runtime_ms", 0))}
        for c in flowbook_cells:
            if c["cell_id"] in cell_data:
                cell_data[c["cell_id"]]["execute_ms"] = c.get("execute_duration_ms", c.get("cell_runtime_ms", 0))
                cell_data[c["cell_id"]]["code_ms"] = c.get("code_duration_ms", 0)
                cell_data[c["cell_id"]]["state_ms"] = c.get("state_duration_ms", 0)
                cell_data[c["cell_id"]]["check_ms"] = c.get("check_duration_ms", 0)

        cell_ids = list(cell_data.keys())
        baseline_runtimes = [cell_data[cid].get("baseline_ms", 0) for cid in cell_ids]
        flowbook_execute = [cell_data[cid].get("execute_ms", 0) for cid in cell_ids]
        flowbook_code = [cell_data[cid].get("code_ms", 0) for cid in cell_ids]
        state_times = [cell_data[cid].get("state_ms", 0) for cid in cell_ids]
        check_times = [cell_data[cid].get("check_ms", 0) for cid in cell_ids]

        cells = np.arange(1, len(cell_ids) + 1)

        # Per-cell arrays
        baseline_arr = np.array(baseline_runtimes)
        code_arr = np.array(flowbook_code)
        state_arr = np.array(state_times)
        check_arr = np.array(check_times)
        execute_arr = np.array(flowbook_execute)
        # Other overhead = execute - (code + state + check)
        other_arr = execute_arr - (code_arr + state_arr + check_arr)
        other_arr = np.maximum(other_arr, 0)  # Clamp to non-negative

        # Cumulative times
        baseline_cumsum = np.cumsum(baseline_arr)
        # FlowBook cumulative: code + state + check + other (same as execute)
        code_cumsum = np.cumsum(code_arr)
        state_cumsum = np.cumsum(state_arr)
        check_cumsum = np.cumsum(check_arr)
        other_cumsum = np.cumsum(other_arr)

        # Left y-axis: cumulative time - baseline line and FlowBook stacked areas
        ax.plot(cells, baseline_cumsum / 1000, color=colors[0], linewidth=2, marker='o', markersize=4, label="Baseline")

        # FlowBook as stacked area: code (bottom) + state + check + other (top)
        ax.fill_between(cells, 0, code_cumsum / 1000, alpha=0.3, color=colors[1], label="FlowBook Code")
        ax.fill_between(cells, code_cumsum / 1000, (code_cumsum + state_cumsum) / 1000, alpha=0.4, color=colors[2], label="State")
        ax.fill_between(cells, (code_cumsum + state_cumsum) / 1000, (code_cumsum + state_cumsum + check_cumsum) / 1000, alpha=0.4, color=colors[3], label="Check")
        ax.fill_between(cells, (code_cumsum + state_cumsum + check_cumsum) / 1000, (code_cumsum + state_cumsum + check_cumsum + other_cumsum) / 1000, alpha=0.4, color=colors[4], label="Other")

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)
        title = "Timing Comparison"
        if timing_initial_count < len(cells):
            title += f" (cells 1-{timing_initial_count} + {len(cells) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.legend(loc="lower right", fontsize=legend_size)

        # Add separator line for rerun phase
        if timing_initial_count < len(cells):
            ax.axvline(x=timing_initial_count + 0.5, color='red', linestyle='--', linewidth=2)

        # Add timing breakdown text box in upper-left (no overlap with legend now)
        total_baseline_s = baseline_cumsum[-1] / 1000
        total_code_s = code_cumsum[-1] / 1000
        total_state_s = state_cumsum[-1] / 1000
        total_check_s = check_cumsum[-1] / 1000
        total_other_s = other_cumsum[-1] / 1000
        total_flowbook_s = total_code_s + total_state_s + total_check_s + total_other_s

        textstr = f'Baseline: {total_baseline_s:.2f}s\n\nFlowBook:\n  Code: {total_code_s:.2f}s\n  State: {total_state_s:.2f}s\n  Check: {total_check_s:.2f}s\n  Other: {total_other_s:.2f}s\n  Total: {total_flowbook_s:.2f}s'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=legend_size,
                verticalalignment='top', horizontalalignment='left', bbox=props)

        if baseline_cumsum[-1] > 0:
            overhead_pct = (total_flowbook_s * 1000 - baseline_cumsum[-1]) / baseline_cumsum[-1] * 100
            ax.annotate(f'{overhead_pct:.1f}% overhead',
                        xy=(cells[-1], total_flowbook_s),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=legend_size, va='center', ha='left', color=colors[1])

    # ========== Panel 2: Memory + GPU ==========
    if has_memory:
        ax = axes[panel_idx]
        panel_idx += 1

        cells = np.arange(1, len(flowbook_mem_cells) + 1)
        baseline_footprint = np.array([c.get("current_footprint_mb", 0) for c in baseline_mem_cells])
        flowbook_footprint = np.array([c.get("current_footprint_mb", 0) for c in flowbook_mem_cells])
        baseline_gpu = np.array([c.get("gpu_mem_samples", 0) for c in baseline_mem_cells])

        # Check if overhead_breakdown data is available
        has_overhead_breakdown = any(
            c.get("overhead_breakdown") for c in flowbook_mem_cells
        )

        # Check if pre/post checkpoint breakdown is available
        has_pre_post = any(
            c.get("pre_only_bytes", 0) > 0 or c.get("post_savings_bytes", 0) > 0
            for c in flowbook_mem_cells
        )

        # Extract pre/post checkpoint sizes (for NO_POST analysis)
        # pre_only_mb: memory if only pre checkpoints existed (accounts for sharing)
        # post_savings_mb: actual memory saved by removing post checkpoints
        pre_only_mb = np.array([
            c.get("pre_only_bytes", 0) / (1024 * 1024)
            for c in flowbook_mem_cells
        ])
        post_savings_mb = np.array([
            c.get("post_savings_bytes", 0) / (1024 * 1024)
            for c in flowbook_mem_cells
        ])

        if has_overhead_breakdown:
            # Stacked overhead breakdown visualization
            # Extract overhead categories per cell
            checkpoints_mb = np.array([
                (c.get("overhead_breakdown") or {}).get("checkpoints_mb", 0)
                for c in flowbook_mem_cells
            ])
            execution_records_mb = np.array([
                (c.get("overhead_breakdown") or {}).get("execution_records_mb", 0)
                for c in flowbook_mem_cells
            ])
            tracking_metadata_mb = np.array([
                (c.get("overhead_breakdown") or {}).get("tracking_metadata_mb", 0)
                for c in flowbook_mem_cells
            ])
            other_mb = np.array([
                (c.get("overhead_breakdown") or {}).get("other_mb", 0)
                for c in flowbook_mem_cells
            ])

            # Colors for stacked areas
            stack_colors = sns.color_palette("Set2", 5)

            # Check for GPU memory
            has_gpu = any(g > 0 for g in baseline_gpu)

            # Layer 1: Baseline CPU memory (bottom - gray)
            ax.fill_between(cells, 0, baseline_footprint, alpha=0.3, color='gray', label='Baseline CPU')

            # Layer 2: GPU memory (middle - orange, stacked on baseline)
            cumulative = baseline_footprint.copy()
            if has_gpu:
                next_level = cumulative + baseline_gpu
                ax.fill_between(cells, cumulative, next_level, alpha=0.4, color='orange', label='GPU Memory')
                cumulative = next_level

            # Layer 3: FlowBook overhead categories (top - stacked on GPU)
            # Checkpoints (largest)
            next_level = cumulative + checkpoints_mb
            ax.fill_between(cells, cumulative, next_level, alpha=0.5, color=stack_colors[0], label='Checkpoints')
            cumulative = next_level

            # Execution records
            next_level = cumulative + execution_records_mb
            ax.fill_between(cells, cumulative, next_level, alpha=0.5, color=stack_colors[1], label='Exec Records')
            cumulative = next_level

            # Tracking metadata
            next_level = cumulative + tracking_metadata_mb
            ax.fill_between(cells, cumulative, next_level, alpha=0.5, color=stack_colors[2], label='Tracking')
            cumulative = next_level

            # Other
            next_level = cumulative + other_mb
            ax.fill_between(cells, cumulative, next_level, alpha=0.5, color=stack_colors[3], label='Other')
            cumulative = next_level

            # Draw lines for reference
            ax.plot(cells, baseline_footprint, color='gray', linewidth=2, linestyle='--')
            ax.plot(cells, cumulative, color=colors[1], linewidth=2, marker='o', markersize=4, label='Total')

            # Show pre-only line (what memory would be with NO post checkpoints)
            if has_pre_post:
                # Pre-only = baseline + GPU + pre_only checkpoints + other overhead (non-checkpoint)
                other_overhead = execution_records_mb + tracking_metadata_mb + other_mb
                gpu_component = baseline_gpu if has_gpu else np.zeros_like(baseline_footprint)
                pre_only_total = baseline_footprint + gpu_component + pre_only_mb + other_overhead
                ax.plot(cells, pre_only_total, color='green', linewidth=2, linestyle=':', marker='s', markersize=3, label='Pre-Only (no post)')

                # Annotate savings at end
                if cumulative[-1] > 0 and post_savings_mb[-1] > 0:
                    total_ckpt = checkpoints_mb[-1]
                    savings_pct = (post_savings_mb[-1] / total_ckpt * 100) if total_ckpt > 0 else 0
                    ax.annotate(f'Post savings: {post_savings_mb[-1]:.1f}MB ({savings_pct:.0f}%)',
                                xy=(cells[-1], pre_only_total[-1]),
                                xytext=(5, 10), textcoords='offset points',
                                fontsize=legend_size - 2, va='bottom', ha='left',
                                color='green')

            ax.set_title('Memory Overhead (Total vs Pre-Only)', fontsize=title_size)

            # Annotate FlowBook overhead percentage at the end
            if baseline_footprint[-1] > 0:
                mem_overhead_pct = (cumulative[-1] - baseline_footprint[-1]) / baseline_footprint[-1] * 100
                ax.annotate(f'{mem_overhead_pct:.1f}% overhead',
                            xy=(cells[-1], cumulative[-1]),
                            xytext=(5, 0), textcoords='offset points',
                            fontsize=legend_size, va='center', ha='left',
                            color=colors[1])
        else:
            # Original visualization without detailed breakdown
            # Check for GPU memory
            has_gpu = any(g > 0 for g in baseline_gpu)

            # Layer 1: Baseline CPU memory (bottom - gray)
            ax.fill_between(cells, 0, baseline_footprint, alpha=0.3, color=colors[0], label='Baseline CPU')
            cumulative = baseline_footprint.copy()

            # Layer 2: GPU memory (middle - orange)
            if has_gpu:
                next_level = cumulative + baseline_gpu
                ax.fill_between(cells, cumulative, next_level, alpha=0.4, color='orange', label='GPU Memory')
                cumulative = next_level

            # Layer 3: FlowBook overhead (top)
            flowbook_overhead = flowbook_footprint - baseline_footprint
            flowbook_overhead = np.maximum(flowbook_overhead, 0)  # Ensure non-negative
            next_level = cumulative + flowbook_overhead
            ax.fill_between(cells, cumulative, next_level, alpha=0.3, color=colors[1], label='FlowBook Overhead')

            ax.plot(cells, baseline_footprint, color=colors[0], linewidth=2, marker='o', markersize=4)
            ax.plot(cells, next_level, color=colors[1], linewidth=2, marker='o', markersize=4, label='Total')

            # Show pre-only line if available
            if has_pre_post:
                pre_only_total = baseline_footprint + (baseline_gpu if has_gpu else 0) + pre_only_mb
                ax.plot(cells, pre_only_total, color='green', linewidth=2, linestyle=':', marker='s', markersize=3, label='Pre-Only (no post)')

            ax.set_title('Memory (Total vs Pre-Only)', fontsize=title_size)

            # Annotate FlowBook overhead percentage at the end
            if baseline_footprint[-1] > 0:
                mem_overhead_pct = (next_level[-1] - baseline_footprint[-1]) / baseline_footprint[-1] * 100
                ax.annotate(f'{mem_overhead_pct:.1f}% overhead',
                            xy=(cells[-1], next_level[-1]),
                            xytext=(5, 0), textcoords='offset points',
                            fontsize=legend_size, va='center', ha='left',
                            color=colors[1])

        ax.set_xlabel('Cell Number', fontsize=label_size)
        ax.set_ylabel('Memory (MB)', fontsize=label_size)

        # Add separator line for rerun phase
        if memory_initial_count < len(cells):
            ax.axvline(x=memory_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.legend(loc='upper left', fontsize=legend_size - 2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 3: Checkpoint Time by Variable (if available) ==========
    if timing_var_data is not None:
        ax = axes[panel_idx]
        panel_idx += 1
        var_colors = sns.color_palette("husl", len(timing_var_data["vars_ordered"]))
        timing_cells = np.array(timing_var_data["cells"])
        timing_var_types = timing_var_data.get("var_types", {})

        # Stack checkpoint timing by variable (ms -> seconds)
        stacked = [np.array(timing_var_data["by_var"][v]) / 1000 for v in timing_var_data["vars_ordered"]]
        cumulative = np.zeros(len(timing_cells))
        for i, (v, data_sec) in enumerate(zip(timing_var_data["vars_ordered"], stacked)):
            # Include type in legend label
            var_type = timing_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(timing_cells, cumulative, cumulative + data_sec, alpha=0.7, color=var_colors[i], label=label)
            cumulative = cumulative + data_sec

        # Draw total line
        ax.plot(timing_cells, cumulative, color='black', linewidth=1.5, linestyle='--', label='Total')

        # Add separator line for rerun phase
        timing_var_initial_count = timing_var_data.get("initial_count", len(timing_cells))
        if timing_var_initial_count < len(timing_cells):
            ax.axvline(x=timing_var_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Checkpoint Time (seconds)", fontsize=label_size)
        title = "Checkpoint Time by Variable"
        if timing_var_initial_count < len(timing_cells):
            title += f" (cells 1-{timing_var_initial_count} + {len(timing_cells) - timing_var_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.legend(loc="upper left", fontsize=legend_size - 4, ncol=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 4: Checkpoint Memory by Variable (if available) ==========
    if var_data is not None:
        ax = axes[panel_idx]
        panel_idx += 1
        var_colors = sns.color_palette("husl", len(var_data["vars_ordered"]))
        var_cells = np.array(var_data["cells"])
        mb = 1024 * 1024
        mem_var_types = var_data.get("var_types", {})

        # Get baseline memory for reference (same length as var_cells)
        baseline_footprint = np.zeros(len(var_cells))
        if has_memory and len(baseline_mem_cells) >= len(var_cells):
            baseline_footprint = np.array([c.get("current_footprint_mb", 0) for c in baseline_mem_cells[:len(var_cells)]])

        # Draw baseline memory first (bottom layer)
        ax.fill_between(var_cells, 0, baseline_footprint, alpha=0.3, color=colors[0], label='Baseline Memory')

        # Stack checkpoint variables on top of baseline
        stacked = [np.array(var_data["by_var"][v]) / mb for v in var_data["vars_ordered"]]
        cumulative = baseline_footprint.copy()
        for i, (v, data_mb) in enumerate(zip(var_data["vars_ordered"], stacked)):
            # Include type in legend label
            var_type = mem_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(var_cells, cumulative, cumulative + data_mb, alpha=0.7, color=var_colors[i], label=label)
            cumulative = cumulative + data_mb

        # Draw lines for baseline and total
        ax.plot(var_cells, baseline_footprint, color=colors[0], linewidth=2, marker='o', markersize=4)
        ax.plot(var_cells, cumulative, color='black', linewidth=1.5, linestyle='--', label='Total')

        # Add separator line for rerun phase
        var_initial_count = var_data.get("initial_count", len(var_cells))
        if var_initial_count < len(var_cells):
            ax.axvline(x=var_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Memory (MB)", fontsize=label_size)
        title = "Checkpoint Memory by Variable"
        if var_initial_count < len(var_cells):
            title += f" (cells 1-{var_initial_count} + {len(var_cells) - var_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.legend(loc="upper left", fontsize=legend_size - 4, ncol=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=tick_size)

    # Add notebook name as figure title
    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    fig.suptitle(notebook_name, fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Combined v2 plot saved to: {output_path}")
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
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top types/variables to show individually in plots (default: 10)"
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
                fig = plot_combined_v2(data, output_path=None, large_fonts=args.large_fonts, top_n=args.top_n)
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
