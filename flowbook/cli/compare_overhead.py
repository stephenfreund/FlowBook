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
import copy
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import v4.0 models and extraction/rendering
from flowbook.cli.models import ComparisonResult as ComparisonResultV4
from flowbook.cli.plot_extraction import (
    extract_plot1_data,
    extract_plot2_data,
    extract_plot3_data,
    extract_plot4_data,
    extract_plot5_data,
    extract_plot6_data,
    extract_cdf_data,
    # V5 extraction functions
    extract_plot2_data_v5,
    extract_plot3_data_v5,
    extract_plot4_data_v5,
    extract_plot6_data_v5,
    extract_v5_memory_result,
    extract_baseline_cells,
    extract_gpu_overhead_from_timing,
)
from flowbook.cli.plot_rendering import render_combined_6panel, render_time_cdf, render_overhead_pct_cdf, render_base_runtime_cdf

# Base directory for cached remote files
CACHE_BASE_DIR = "/tmp/flowbook_compare_overhead"

# IPython traceback prefix for detecting cell errors
TRACEBACK_PREFIX = "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"


# =============================================================================
# Rerun Overhead Extraction and Plotting
# =============================================================================


@dataclass
class RerunOverheadCDFData:
    """Data for rerun overhead CDF and breakdown plots.

    Attributes:
        total_overhead_ms: All overhead values (flat list for CDF)
        total_sorted: Sorted overhead values for CDF
        total_percentiles: Percentile values for CDF
        checkpoint_ms: All checkpoint times (flat list)
        check_ms: All check times (flat list)
        cell_indices: Unique cell indices measured (for breakdown plot)
        num_iterations: Number of iterations per cell
        breakdown: Dict[cell_index][iteration] = (checkpoint_ms, check_ms)
    """

    total_overhead_ms: List[float]
    total_sorted: List[float]
    total_percentiles: List[float]
    checkpoint_ms: List[float]
    check_ms: List[float]
    # New fields for grouped bar chart breakdown
    cell_indices: List[int] = None  # Unique cell indices (quartiles)
    num_iterations: int = 0
    breakdown: Dict[int, Dict[int, tuple]] = (
        None  # breakdown[cell_idx][iter] = (ckpt, check)
    )


def extract_rerun_overhead_data(
    raw_data_list: List[Dict],
) -> Optional[RerunOverheadCDFData]:
    """Extract rerun overhead data from raw comparison JSON data.

    Args:
        raw_data_list: List of raw JSON dicts from comparison files

    Returns:
        RerunOverheadCDFData or None if no rerun overhead data found
    """
    total_overhead_ms = []
    checkpoint_ms = []
    check_ms = []

    for data in raw_data_list:
        rerun = data.get("rerun_overhead")
        if not rerun:
            continue

        measurements = rerun.get("measurements", [])
        for m in measurements:
            total_overhead_ms.append(m.get("total_overhead_ms", 0.0))
            checkpoint_ms.append(m.get("checkpoint_ms", 0.0))
            check_ms.append(m.get("check_ms", 0.0))

    if not total_overhead_ms:
        return None

    # Sort for CDF
    total_sorted = sorted(total_overhead_ms)
    n = len(total_sorted)
    total_percentiles = [(i + 1) / n for i in range(n)]

    return RerunOverheadCDFData(
        total_overhead_ms=total_overhead_ms,
        total_sorted=total_sorted,
        total_percentiles=total_percentiles,
        checkpoint_ms=checkpoint_ms,
        check_ms=check_ms,
    )


def render_rerun_overhead_cdf(
    ax,
    data: RerunOverheadCDFData,
    large_fonts: bool = True,
    show_sample_size: bool = True,
) -> None:
    """Render rerun overhead CDF panel.

    Uses the same style as Analysis Time Distribution but with orange color.

    Args:
        ax: Matplotlib axes
        data: RerunOverheadCDFData with timing values
        large_fonts: Use larger fonts
        show_sample_size: Whether to show N= annotation
    """
    render_time_cdf(
        ax,
        sorted_vals=list(data.total_sorted),
        percentiles=list(data.total_percentiles),
        n=len(data.total_overhead_ms),
        color="#E67E22",  # Orange
        title="Rerun Overhead Time Distribution",
        xlabel="Rerun Overhead (ms, log scale)",
        large_fonts=large_fonts,
        show_sample_size=show_sample_size,
    )


def render_rerun_checkpoint_breakdown(
    ax,
    data: RerunOverheadCDFData,
    notebook_name: str = "",
    large_fonts: bool = True,
) -> None:
    """Render rerun checkpoint breakdown as grouped stacked bar chart.

    Shows timing breakdown per cell and per iteration, with State (checkpoint)
    stacked below Check. X-axis groups bars by cell, with iterations side-by-side.

    Args:
        ax: Matplotlib axes
        data: RerunOverheadCDFData with breakdown data
        notebook_name: Name for title
        large_fonts: Use larger fonts
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Font sizes
    label_size = 16 if large_fonts else 14
    title_size = 18 if large_fonts else 16
    tick_size = 14 if large_fonts else 12
    legend_size = 12 if large_fonts else 10

    # Check for valid breakdown data
    if (
        data.breakdown is None
        or data.cell_indices is None
        or not data.cell_indices
        or data.num_iterations == 0
    ):
        ax.text(
            0.5,
            0.5,
            "No rerun overhead data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=label_size,
        )
        ax.set_title(
            f'Rerun Overhead Breakdown{" - " + notebook_name if notebook_name else ""}',
            fontsize=title_size,
        )
        return

    cell_indices = data.cell_indices
    num_iterations = data.num_iterations
    breakdown = data.breakdown

    # Colors for stacked components
    colors = sns.color_palette("muted")
    state_color = colors[0]  # Blue-ish
    check_color = colors[2]  # Green-ish

    # Bar positioning
    num_cells = len(cell_indices)
    group_width = 0.8  # Width of each cell group
    bar_width = group_width / num_iterations  # Width of each iteration bar
    group_spacing = 1.0  # Space between cell groups

    # X positions for cell groups
    group_positions = np.arange(num_cells) * group_spacing

    # Plot bars for each iteration
    for iter_idx in range(num_iterations):
        checkpoint_vals = []
        check_vals = []

        for cell_idx in cell_indices:
            if cell_idx in breakdown and iter_idx in breakdown[cell_idx]:
                ckpt, chk = breakdown[cell_idx][iter_idx]
                checkpoint_vals.append(ckpt)
                check_vals.append(chk)
            else:
                checkpoint_vals.append(0)
                check_vals.append(0)

        # Bar positions for this iteration within each group
        bar_offset = (iter_idx - (num_iterations - 1) / 2) * bar_width
        x_positions = group_positions + bar_offset

        # Stacked bars: State (checkpoint) on bottom, Check on top
        label_state = "State" if iter_idx == 0 else None
        label_check = "Check" if iter_idx == 0 else None

        ax.bar(
            x_positions,
            checkpoint_vals,
            bar_width * 0.9,
            color=state_color,
            edgecolor="white",
            linewidth=0.5,
            label=label_state,
            alpha=0.85,
        )
        ax.bar(
            x_positions,
            check_vals,
            bar_width * 0.9,
            bottom=checkpoint_vals,
            color=check_color,
            edgecolor="white",
            linewidth=0.5,
            label=label_check,
            alpha=0.85,
        )

    # X-axis labels: Cell numbers
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f"Cell {idx + 1}" for idx in cell_indices], fontsize=tick_size)

    # Add iteration labels below if multiple iterations
    if num_iterations > 1:
        # Add secondary x-axis info
        ax.set_xlabel(
            f"Cell (each with {num_iterations} iterations)", fontsize=label_size
        )
    else:
        ax.set_xlabel("Cell", fontsize=label_size)

    ax.set_ylabel("Time (ms)", fontsize=label_size)
    ax.set_title(
        f'Rerun Overhead Breakdown{" - " + notebook_name if notebook_name else ""}',
        fontsize=title_size,
    )
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-0.5, num_cells * group_spacing - 0.5)


# =============================================================================
# Trial Grouping and Averaging
# =============================================================================


def group_trial_files(file_paths: List[str]) -> Dict[str, List[str]]:
    """Group trial files by notebook stem.

    Files like 'notebook_comparison-1.json', 'notebook_comparison-2.json'
    are grouped under key 'notebook_comparison'.

    Args:
        file_paths: List of file paths

    Returns:
        Dict mapping stem to list of file paths
    """
    groups: Dict[str, List[str]] = defaultdict(list)

    for path in file_paths:
        name = Path(path).stem
        # Match pattern: name-N where N is a number (trial suffix)
        match = re.match(r"^(.+)-(\d+)$", name)
        if match:
            stem = match.group(1)
        else:
            stem = name
        groups[stem].append(path)

    return dict(groups)


def average_dict_values(dicts: List[Dict]) -> Dict:
    """Average numeric values across dicts.

    Args:
        dicts: List of dicts to average

    Returns:
        Dict with averaged numeric values, non-numeric values taken from first dict
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]

    result = {}
    all_keys: set = set()
    for d in dicts:
        if d:
            all_keys.update(d.keys())

    for key in all_keys:
        values = [d.get(key) for d in dicts if d and key in d]
        if values and all(
            isinstance(v, (int, float)) and v is not None for v in values
        ):
            result[key] = sum(values) / len(values)
        elif values:
            result[key] = values[0]  # Non-numeric, use first

    return result


def average_cell_metrics(cells: List[Dict]) -> Dict:
    """Average numeric fields across cell dicts.

    Args:
        cells: List of cell dicts from different trials

    Returns:
        Averaged cell dict
    """
    if len(cells) == 1:
        return cells[0]

    avg = {"cell_id": cells[0].get("cell_id"), "cell_index": cells[0].get("cell_index")}

    numeric_fields = [
        "execute_duration_ms",
        "code_duration_ms",
        "state_duration_ms",
        "check_duration_ms",
        "cell_runtime_ms",  # For compatibility with older formats
        "current_footprint_mb",
        "max_footprint_mb",
        "allocation_delta_mb",
        "gpu_mem_samples",
        "pre_only_bytes",
        "post_savings_bytes",
    ]
    for field in numeric_fields:
        values = [c.get(field) for c in cells if c.get(field) is not None]
        if values:
            avg[field] = sum(values) / len(values)

    # Copy non-numeric fields from first cell
    for key, val in cells[0].items():
        if key not in avg:
            avg[key] = val

    return avg


def average_trial_data(trial_datas: List[Dict]) -> Dict:
    """Average metrics across multiple trial JSON dicts.

    Args:
        trial_datas: List of comparison data dicts from different trials

    Returns:
        Single dict with averaged per-cell values
    """
    if len(trial_datas) == 1:
        return trial_datas[0]

    # Use first trial as template
    result = copy.deepcopy(trial_datas[0])
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["averaged_trials"] = len(trial_datas)

    # Average cells for each kernel/phase
    for kernel in ["baseline", "flowbook"]:
        for phase in ["timing", "memory"]:
            phase_data = result.get("kernels", {}).get(kernel, {}).get(phase)
            if not phase_data:
                continue

            for cell_list_key in ["cells", "rerun_cells"]:
                if cell_list_key not in phase_data:
                    continue

                # Collect cells from all trials by (cell_id, cell_index)
                all_cells_by_key: Dict[Tuple, List[Dict]] = defaultdict(list)
                for trial in trial_datas:
                    cells = (
                        trial.get("kernels", {})
                        .get(kernel, {})
                        .get(phase, {})
                        .get(cell_list_key, [])
                    )
                    for c in cells:
                        key = (c.get("cell_id"), c.get("cell_index"))
                        all_cells_by_key[key].append(c)

                # Average each cell group
                averaged_cells = []
                for (cell_id, cell_index), trial_cells in sorted(
                    all_cells_by_key.items(),
                    key=lambda x: x[0][1] if x[0][1] is not None else 0,
                ):
                    avg_cell = average_cell_metrics(trial_cells)
                    averaged_cells.append(avg_cell)

                phase_data[cell_list_key] = averaged_cells

            # Average totals
            all_totals = [
                t.get("kernels", {}).get(kernel, {}).get(phase, {}).get("totals", {})
                for t in trial_datas
            ]
            phase_data["totals"] = average_dict_values(all_totals)

            # Check staleness consistency across trials (flowbook timing only)
            if kernel == "flowbook" and phase == "timing":
                first_checking = (
                    all_totals[0].get("checking_summary", {}) if all_totals else {}
                )
                first_staleness = (
                    first_checking.get("clean_cells", 0),
                    first_checking.get("stale_cells", 0),
                    tuple(sorted(first_checking.get("reason_counts", {}).items())),
                )
                for i, totals in enumerate(all_totals[1:], start=2):
                    checking = totals.get("checking_summary", {})
                    staleness = (
                        checking.get("clean_cells", 0),
                        checking.get("stale_cells", 0),
                        tuple(sorted(checking.get("reason_counts", {}).items())),
                    )
                    if staleness != first_staleness:
                        notebook_path = result.get("notebook_path", "unknown")
                        print(
                            f"WARNING: Staleness differs across trials for {notebook_path}:",
                            file=sys.stderr,
                        )
                        print(
                            f"  Trial 1: clean={first_staleness[0]}, stale={first_staleness[1]}, reasons={dict(first_staleness[2])}",
                            file=sys.stderr,
                        )
                        print(
                            f"  Trial {i}: clean={staleness[0]}, stale={staleness[1]}, reasons={dict(staleness[2])}",
                            file=sys.stderr,
                        )
                        break  # Only warn once per notebook

                # Don't average checking_summary - use first trial's values (should be identical)
                if "checking_summary" in phase_data["totals"]:
                    phase_data["totals"]["checking_summary"] = first_checking

    return result


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
    match = re.match(r"^(?:([^@:]+)@)?([^:/]+):(.+)$", path)
    if match:
        user = match.group(1) or ""
        host = match.group(2)
        remote_path = match.group(3)
        # Exclude Windows drive letters (single letter before colon)
        if len(host) == 1 and host.isalpha():
            return (False, "", "", path)
        return (True, user, host, remote_path)
    return (False, "", "", path)


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
    remote_spec: str, force_download: bool = False
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
        print(
            "Status: Using cached files (use --force-download to refresh)",
            file=sys.stderr,
        )
    else:
        if force_download and cache_exists:
            print("Status: Force downloading (clearing cache)...", file=sys.stderr)
            shutil.rmtree(cache_dir)
        else:
            print("Status: Downloading...", file=sys.stderr)

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Build rsync command
        if "*" in remote_path or "?" in remote_path:
            remote_dir = os.path.dirname(remote_path)
            pattern = os.path.basename(remote_path)

            rsync_cmd = [
                "rsync",
                "-avz",
                "--include",
                pattern,
                "--exclude",
                "*",
                f"{remote_host}:{remote_dir}/",
                cache_dir + "/",
            ]
        else:
            rsync_cmd = [
                "rsync",
                "-avz",
                f"{remote_host}:{remote_path}",
                cache_dir + "/",
            ]

        try:
            result = subprocess.run(
                rsync_cmd, capture_output=True, text=True, check=True
            )
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                file_lines = [
                    l
                    for l in lines
                    if l and not l.startswith(("sending", "sent", "total", "receiving"))
                ]
                if file_lines:
                    print(f"Files:  {len(file_lines)} file(s) synced", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error: rsync failed: {e.stderr}", file=sys.stderr)
            raise

    print("", file=sys.stderr)

    # Find the local files matching the pattern
    if "*" in remote_path or "?" in remote_path:
        pattern = os.path.basename(remote_path)
        local_files = glob.glob(os.path.join(cache_dir, pattern))
    else:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(cache_dir, filename)
        local_files = [local_path] if os.path.exists(local_path) else []

    return sorted(local_files), cache_dir


def resolve_file_paths(
    file_paths: List[str], force_download: bool = False
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
            if "*" in path or "?" in path:
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
    memory_overhead_ratio: (
        float  # flowbook_memory / baseline_memory (like slowdown for time)
    )
    # Last cell overhead percentages
    last_cell_state_overhead_pct: float = 0.0
    last_cell_check_overhead_pct: float = 0.0
    last_cell_memory_overhead_pct: float = 0.0
    # Peak memory overhead (max checkpoint / base across all cells)
    peak_memory_overhead_pct: float = 0.0
    # Rerun stats (optional)
    num_reruns: int = 0
    rerun_baseline_runtime_ms: float = 0.0
    rerun_flowbook_runtime_ms: float = 0.0
    rerun_state_overhead_ms: float = 0.0
    rerun_check_overhead_ms: float = 0.0
    rerun_flowbook_total_ms: float = 0.0
    rerun_final_checkpoint_bytes: int = 0
    # Number of trials averaged (1 = single trial)
    num_trials: int = 1
    # Per-cell data for aggregate statistics
    per_cell_checkpoint_overhead_ms: List[float] = field(default_factory=list)
    per_cell_total_overhead_ms: List[float] = field(default_factory=list)
    per_cell_memory_overhead_mb: List[float] = field(
        default_factory=list
    )  # Actually ratio
    per_cell_checkpoint_mb: List[float] = field(default_factory=list)  # Raw MB values
    # Checking results (staleness summary)
    checking_clean_cells: int = 0
    checking_stale_cells: int = 0
    checking_error_cells: int = 0
    checking_reason_counts: Dict[str, int] = field(default_factory=dict)
    checking_error_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class AggregateStats:
    """Aggregate statistics across multiple files."""

    num_files: int
    total_cells: int
    # Overall slowdown
    slowdown_mean: float
    slowdown_median: float
    slowdown_std: float
    slowdown_min: float
    slowdown_max: float
    slowdown_p90: float
    slowdown_p95: float
    slowdown_p99: float
    # Overall overhead percentages
    state_overhead_pct_mean: float
    check_overhead_pct_mean: float
    memory_overhead_pct_mean: float
    memory_overhead_pct_median: float = 0.0
    memory_overhead_pct_p90: float = 0.0
    memory_overhead_pct_p95: float = 0.0
    # Memory overhead ratio (like slowdown)
    memory_overhead_mean: float = 0.0
    memory_overhead_median: float = 0.0
    memory_overhead_std: float = 0.0
    memory_overhead_min: float = 0.0
    memory_overhead_max: float = 0.0
    memory_overhead_p90: float = 0.0
    memory_overhead_p95: float = 0.0
    memory_overhead_p99: float = 0.0
    # Per-cell checkpoint overhead (ms)
    checkpoint_overhead_per_cell_mean: float = 0.0
    checkpoint_overhead_per_cell_median: float = 0.0
    checkpoint_overhead_per_cell_min: float = 0.0
    checkpoint_overhead_per_cell_max: float = 0.0
    checkpoint_overhead_per_cell_p90: float = 0.0
    checkpoint_overhead_per_cell_p95: float = 0.0
    checkpoint_overhead_per_cell_p99: float = 0.0
    # Per-cell total overhead (ms)
    total_overhead_per_cell_mean: float = 0.0
    total_overhead_per_cell_median: float = 0.0
    total_overhead_per_cell_min: float = 0.0
    total_overhead_per_cell_max: float = 0.0
    total_overhead_per_cell_p90: float = 0.0
    total_overhead_per_cell_p95: float = 0.0
    total_overhead_per_cell_p99: float = 0.0
    # Per-cell memory overhead (MB)
    memory_overhead_per_cell_mean: float = 0.0
    memory_overhead_per_cell_median: float = 0.0
    memory_overhead_per_cell_min: float = 0.0
    memory_overhead_per_cell_max: float = 0.0
    memory_overhead_per_cell_p90: float = 0.0
    memory_overhead_per_cell_p95: float = 0.0
    memory_overhead_per_cell_p99: float = 0.0
    # Raw per-cell data for histograms
    all_total_overhead_per_cell: List[float] = field(default_factory=list)
    all_memory_overhead_per_cell: List[float] = field(default_factory=list)  # Ratio
    all_checkpoint_mb_per_cell: List[float] = field(default_factory=list)  # Raw MB
    # Per-notebook peak memory overhead percentages (for CDF)
    all_peak_memory_overhead_pct: List[float] = field(default_factory=list)
    # Aggregate checking results (staleness summary across all files)
    total_checking_clean_cells: int = 0
    total_checking_stale_cells: int = 0
    total_checking_error_cells: int = 0
    total_checking_reason_counts: Dict[str, int] = field(default_factory=dict)
    total_checking_error_counts: Dict[str, int] = field(default_factory=dict)


def load_comparison_json(file_path: str) -> Dict[str, Any]:
    """Load and validate a comparison JSON file (v1.0 or v2.0 format)."""
    with open(file_path) as f:
        data = json.load(f)

    # Validate structure
    if "kernels" not in data:
        raise ValueError(
            f"Invalid comparison file: missing 'kernels' key in {file_path}"
        )
    if "baseline" not in data["kernels"] or "flowbook" not in data["kernels"]:
        raise ValueError(
            f"Invalid comparison file: missing baseline or flowbook results in {file_path}"
        )

    # Detect version
    version = data.get("version", "1.0")
    data["_version"] = version

    return data


def is_v2_format(data: Dict[str, Any]) -> bool:
    """Check if data is v2.0 format (with separate timing/memory)."""
    # Check both _version (set by load_comparison_json) and version (in raw JSON)
    version = data.get("_version") or data.get("version", "1.0")
    return str(version).startswith("2")


def is_v3_format(data: Dict[str, Any]) -> bool:
    """Check if data is v3.0 format (with pre-cell memory measurements for cross-run comparison)."""
    version = data.get("_version") or data.get("version", "1.0")
    return str(version).startswith("3")


def is_v4_format(data: Dict[str, Any]) -> bool:
    """Check if data is v4.0 format (dataclass-based with checkpoint_vars)."""
    version = data.get("_version") or data.get("version", "1.0")
    return str(version).startswith("4")


def is_v5_format(data: Dict[str, Any]) -> bool:
    """Check if data is v5.0 format (simplified memory structure)."""
    version = data.get("_version") or data.get("version", "1.0")
    return str(version).startswith("5")


def is_v4_or_v5_format(data: Dict[str, Any]) -> bool:
    """Check if data is v4 or v5 format."""
    return is_v4_format(data) or is_v5_format(data)


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


def has_baseline_errors(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if baseline run has any cell errors.

    Errors are detected by checking if the error field starts with the
    IPython traceback prefix (ANSI-colored separator line).

    Returns:
        Tuple of (has_errors, list_of_cell_ids_with_errors)
    """
    error_cells = []
    baseline = data.get("kernels", {}).get("baseline", {})
    timing = baseline.get("timing", {})

    for cell in timing.get("cells", []):
        error = cell.get("error")
        if error and isinstance(error, str) and error.startswith(TRACEBACK_PREFIX):
            error_cells.append(cell.get("cell_id", "unknown"))

    # Also check rerun_cells if present
    for cell in timing.get("rerun_cells", []):
        error = cell.get("error")
        if error and isinstance(error, str) and error.startswith(TRACEBACK_PREFIX):
            error_cells.append(cell.get("cell_id", "unknown"))

    return (len(error_cells) > 0, error_cells)


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

    # Support both v1.0 (cell_runtime_ms) and v2.0 (execute_duration_ms) key names
    baseline_runtime = baseline_totals.get("cell_runtime_ms") or baseline_totals.get(
        "execute_duration_ms", 0.0
    )
    flowbook_runtime = flowbook_totals.get("cell_runtime_ms") or flowbook_totals.get(
        "execute_duration_ms", 0.0
    )
    state_overhead = flowbook_totals.get("state_duration_ms", 0.0)
    check_overhead = flowbook_totals.get("check_duration_ms", 0.0)
    flowbook_total = flowbook_runtime

    # Extract checking summary (staleness data)
    checking_summary = flowbook_totals.get("checking_summary", {})
    checking_clean_cells = checking_summary.get("clean_cells", 0)
    checking_stale_cells = checking_summary.get("stale_cells", 0)
    checking_error_cells = checking_summary.get("error_cells", 0)
    checking_reason_counts = checking_summary.get("reason_counts", {})
    checking_error_counts = checking_summary.get("error_counts", {})

    if baseline_runtime > 0:
        slowdown = flowbook_total / baseline_runtime
        state_pct = (state_overhead / baseline_runtime) * 100
        check_pct = (check_overhead / baseline_runtime) * 100
    elif flowbook_runtime > 0:
        # No baseline: compute slowdown as overhead ratio relative to flowbook execution time
        slowdown = (
            flowbook_runtime + state_overhead + check_overhead
        ) / flowbook_runtime
        state_pct = (state_overhead / flowbook_runtime) * 100
        check_pct = (check_overhead / flowbook_runtime) * 100
    else:
        slowdown = 0.0
        state_pct = 0.0
        check_pct = 0.0

    # Memory from v2.0 Scalene data
    baseline_memory_data = baseline.get("memory", {})
    flowbook_memory_data = flowbook.get("memory", {})
    baseline_mem_totals = (
        baseline_memory_data.get("totals", {}) if baseline_memory_data else {}
    )
    flowbook_mem_totals = (
        flowbook_memory_data.get("totals", {}) if flowbook_memory_data else {}
    )

    # Convert MB to bytes for consistency with old format
    mb_to_bytes = 1024 * 1024
    baseline_memory = int(
        baseline_mem_totals.get("final_footprint_mb", 0) * mb_to_bytes
    )
    flowbook_memory = int(
        flowbook_mem_totals.get("final_footprint_mb", 0) * mb_to_bytes
    )
    memory_overhead = (
        flowbook_memory - baseline_memory if flowbook_memory > baseline_memory else 0
    )

    # Check if FlowBook JSON has precomputed memory_overhead_ratio (new format)
    if "memory_overhead_ratio" in flowbook_mem_totals:
        # Use precomputed ratio: (base_namespace + checkpoint_overhead) / base_namespace
        memory_overhead_ratio = flowbook_mem_totals["memory_overhead_ratio"]
        # Compute memory_pct from the ratio (ratio - 1.0 is the fractional overhead)
        memory_pct = (memory_overhead_ratio - 1.0) * 100
        # For baseline_memory, use base_namespace_mb if no baseline kernel was run
        if baseline_memory == 0:
            baseline_memory = int(
                flowbook_mem_totals.get("base_namespace_mb", 0) * mb_to_bytes
            )
            memory_overhead = int(
                flowbook_mem_totals.get("total_overhead_mb", 0) * mb_to_bytes
            )
    elif baseline_memory > 0:
        memory_pct = (memory_overhead / baseline_memory) * 100
        memory_overhead_ratio = flowbook_memory / baseline_memory
    else:
        # No baseline and no precomputed ratio: compute ratio from checkpoint overhead
        # Get checkpoint overhead from last memory cell's overhead_breakdown
        flowbook_mem_cells = (
            flowbook_memory_data.get("cells", []) if flowbook_memory_data else []
        )
        checkpoint_overhead_mb = 0.0
        if flowbook_mem_cells:
            last_mem_cell = flowbook_mem_cells[-1]
            overhead_breakdown = last_mem_cell.get("overhead_breakdown", {})
            checkpoint_overhead_mb = overhead_breakdown.get("checkpoints_mb", 0.0)

        flowbook_memory_mb = (
            flowbook_memory / mb_to_bytes if flowbook_memory > 0 else 0.0
        )

        if flowbook_memory_mb > 0 and checkpoint_overhead_mb > 0:
            # No baseline: report checkpoint/total as the overhead ratio
            # This shows what fraction of total memory is checkpoint overhead
            # 0.0 = no overhead, 1.0 = all memory is checkpoints
            memory_overhead_ratio = checkpoint_overhead_mb / flowbook_memory_mb
            memory_pct = memory_overhead_ratio * 100
        else:
            memory_pct = 0.0
            memory_overhead_ratio = 0.0

    num_cells = data.get("metadata", {}).get("num_cells", 0)
    if num_cells == 0 and baseline_timing:
        num_cells = len(baseline_timing.get("cells", []))

    # Get number of trials that were averaged (1 = single trial, >1 = averaged)
    num_trials = data.get("metadata", {}).get("averaged_trials", 1)

    # Extract rerun statistics from v2.0 format
    rerun_baseline_cells = (
        baseline_timing.get("rerun_cells", []) if baseline_timing else []
    )
    rerun_flowbook_cells = (
        flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    )
    num_reruns = len(rerun_flowbook_cells)

    rerun_baseline_runtime = sum(
        c.get("cell_runtime_ms") or c.get("execute_duration_ms", 0)
        for c in rerun_baseline_cells
    )
    rerun_flowbook_runtime = sum(
        c.get("cell_runtime_ms") or c.get("execute_duration_ms", 0)
        for c in rerun_flowbook_cells
    )
    rerun_state_overhead = sum(
        c.get("state_duration_ms", 0) for c in rerun_flowbook_cells
    )
    rerun_check_overhead = sum(
        c.get("check_duration_ms", 0) for c in rerun_flowbook_cells
    )
    rerun_flowbook_total = rerun_flowbook_runtime

    # Get final checkpoint size from last rerun cell (if available)
    rerun_final_checkpoint = 0
    flowbook_mem_rerun = (
        flowbook_memory_data.get("rerun_cells", []) if flowbook_memory_data else []
    )
    if flowbook_mem_rerun:
        last_rerun_mem = flowbook_mem_rerun[-1]
        overhead_breakdown = last_rerun_mem.get("overhead_breakdown", {})
        if overhead_breakdown:
            rerun_final_checkpoint = int(
                overhead_breakdown.get("checkpoints_mb", 0) * 1024 * 1024
            )

    # Last cell overhead from timing data
    baseline_timing_cells = baseline_timing.get("cells", []) if baseline_timing else []
    flowbook_timing_cells = flowbook_timing.get("cells", []) if flowbook_timing else []

    last_cell_state_pct = 0.0
    last_cell_check_pct = 0.0
    last_cell_memory_pct = 0.0

    if flowbook_timing_cells and baseline_timing_cells:
        last_fc = flowbook_timing_cells[-1]
        last_bc = baseline_timing_cells[-1]

        last_baseline_runtime = last_bc.get("cell_runtime_ms") or last_bc.get(
            "execute_duration_ms", 0.0
        )
        last_state = last_fc.get("state_duration_ms", 0.0)
        last_check = last_fc.get("check_duration_ms", 0.0)

        if last_baseline_runtime > 0:
            last_cell_state_pct = (last_state / last_baseline_runtime) * 100
            last_cell_check_pct = (last_check / last_baseline_runtime) * 100

    # Memory overhead from memory data
    baseline_mem_cells = (
        baseline_memory_data.get("cells", []) if baseline_memory_data else []
    )
    flowbook_mem_cells = (
        flowbook_memory_data.get("cells", []) if flowbook_memory_data else []
    )

    if flowbook_mem_cells and baseline_mem_cells:
        last_baseline_mem = baseline_mem_cells[-1].get("current_footprint_mb", 0)
        last_flowbook_mem = flowbook_mem_cells[-1].get("current_footprint_mb", 0)
        if last_baseline_mem > 0:
            last_cell_memory_pct = (
                (last_flowbook_mem - last_baseline_mem) / last_baseline_mem
            ) * 100

    # Collect per-cell overhead data for aggregate statistics
    per_cell_checkpoint_overhead_ms: List[float] = []
    per_cell_total_overhead_ms: List[float] = []
    per_cell_memory_overhead_mb: List[float] = []  # Actually ratio
    per_cell_checkpoint_mb: List[float] = []  # Raw MB values

    # Per-cell timing overhead (checkpoint = state_duration_ms, total = state + check + other)
    for fc in flowbook_timing_cells:
        state_ms = fc.get("state_duration_ms", 0.0)
        check_ms = fc.get("check_duration_ms", 0.0)
        execute_ms = fc.get("execute_duration_ms") or fc.get("cell_runtime_ms", 0.0)
        # If code_duration_ms is not available, derive it (so other_ms = 0)
        code_ms = fc.get("code_duration_ms")
        if code_ms is None:
            code_ms = max(execute_ms - state_ms - check_ms, 0)
        other_ms = max(execute_ms - (code_ms + state_ms + check_ms), 0)
        per_cell_checkpoint_overhead_ms.append(state_ms)
        per_cell_total_overhead_ms.append(state_ms + check_ms + other_ms)

    # Per-cell memory overhead as ratio of checkpoint_delta / (prev_namespace + prev_gpu)
    # This gives a proportion: 0 = no overhead, 1 = checkpoint equals base memory size
    from flowbook.cli.plot_extraction import MIN_BASE_MB

    min_meaningful_base_mb = MIN_BASE_MB  # 0.1 MB threshold

    # Detect v5 format
    version = data.get("version", "4.0")
    is_v5 = str(version).startswith("5")

    if is_v5:
        # V5 format: use simplified fields directly
        # checkpoint_mb is cumulative, so delta = cell[i].checkpoint_mb - cell[i-1].checkpoint_mb
        for i, cell in enumerate(flowbook_mem_cells):
            # Compute checkpoint delta (new checkpoint data at this cell)
            curr_ckpt = cell.get("checkpoint_mb", 0)
            prev_ckpt = (
                flowbook_mem_cells[i - 1].get("checkpoint_mb", 0) if i > 0 else 0
            )
            delta_mb = max(0, curr_ckpt - prev_ckpt)

            # Get base (prev cell's namespace + gpu)
            if i == 0:
                base_mb = 0
            else:
                prev_cell = flowbook_mem_cells[i - 1]
                base_mb = prev_cell.get("user_ns_mb", 0) + prev_cell.get("gpu_mb", 0)

            # Compute ratio (base_mb > 0 guards the division when
            # min_meaningful_base_mb is 0; the first cell always has base 0)
            if base_mb >= min_meaningful_base_mb and base_mb > 0:
                ratio = delta_mb / base_mb
            else:
                ratio = 0.0

            per_cell_memory_overhead_mb.append(ratio)
            per_cell_checkpoint_mb.append(delta_mb)

        # Peak memory overhead: (max_flowbook_total / max_base_total - 1) * 100
        # flowbook_total = user_ns_mb + gpu_mb + checkpoint_mb
        # base_total = user_ns_mb + gpu_mb
        if flowbook_mem_cells:
            max_flowbook_total = max(
                c.get("user_ns_mb", 0) + c.get("gpu_mb", 0) + c.get("checkpoint_mb", 0)
                for c in flowbook_mem_cells
            )
            max_base_total = max(
                c.get("user_ns_mb", 0) + c.get("gpu_mb", 0) for c in flowbook_mem_cells
            )
            if max_base_total >= min_meaningful_base_mb and max_base_total > 0:
                peak_memory_overhead_pct = (
                    max_flowbook_total / max_base_total - 1
                ) * 100
            else:
                peak_memory_overhead_pct = 0.0
        else:
            peak_memory_overhead_pct = 0.0
    else:
        # V4 format: use legacy field names and computation
        # Pre-compute cumulative checkpoint totals for delta calculation (fallback)
        def get_cumulative_total(c):
            checkpoint_cumulative = c.get("checkpoint_cumulative_mb")
            if checkpoint_cumulative is not None and checkpoint_cumulative > 0:
                return checkpoint_cumulative
            cumulative_by_var = c.get("cumulative_by_var") or {}
            if cumulative_by_var:
                return sum(cumulative_by_var.values()) / (1024 * 1024)
            cumulative_by_type = c.get("cumulative_by_type") or {}
            if cumulative_by_type:
                return sum(cumulative_by_type.values()) / (1024 * 1024)
            var_costs = c.get("checkpoint_var_costs") or {}
            if var_costs:
                return sum(v.get("bytes", 0) for v in var_costs.values()) / (
                    1024 * 1024
                )
            return 0

        cumulative_totals = [get_cumulative_total(c) for c in flowbook_mem_cells]

        for i, fc in enumerate(flowbook_mem_cells):
            if i == 0:
                base_mb = 0  # No prior namespace for first cell
            else:
                prev_cell = flowbook_mem_cells[i - 1]
                # Support both old and new field names for backward compatibility
                namespace = prev_cell.get(
                    "namespace_mb", prev_cell.get("current_footprint_mb", 0)
                )
                gpu = prev_cell.get("gpu_mb", prev_cell.get("gpu_mem_samples", 0))
                base_mb = namespace + gpu

            # Get checkpoint delta (new field) or compute from old fields
            delta_mb = fc.get("checkpoint_delta_mb")
            if not (delta_mb is not None and delta_mb > 0):
                overhead = fc.get("overhead_breakdown") or {}
                delta_mb = overhead.get("checkpoints_mb")
            if not (delta_mb is not None and delta_mb > 0):
                # Derive from cumulative totals
                delta_mb = cumulative_totals[i]

            if base_mb >= min_meaningful_base_mb and base_mb > 0:
                ratio = delta_mb / base_mb
            else:
                ratio = 0.0
            per_cell_memory_overhead_mb.append(ratio)
            per_cell_checkpoint_mb.append(delta_mb if delta_mb is not None else 0.0)

        # Compute peak memory overhead percentage (max ratio * 100)
        peak_memory_overhead_pct = (
            max(per_cell_memory_overhead_mb) * 100
            if per_cell_memory_overhead_mb
            else 0.0
        )

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
        memory_overhead_ratio=memory_overhead_ratio,
        last_cell_state_overhead_pct=last_cell_state_pct,
        last_cell_check_overhead_pct=last_cell_check_pct,
        last_cell_memory_overhead_pct=last_cell_memory_pct,
        peak_memory_overhead_pct=peak_memory_overhead_pct,
        num_reruns=num_reruns,
        rerun_baseline_runtime_ms=rerun_baseline_runtime,
        rerun_flowbook_runtime_ms=rerun_flowbook_runtime,
        rerun_state_overhead_ms=rerun_state_overhead,
        rerun_check_overhead_ms=rerun_check_overhead,
        rerun_flowbook_total_ms=rerun_flowbook_total,
        rerun_final_checkpoint_bytes=rerun_final_checkpoint,
        num_trials=num_trials,
        per_cell_checkpoint_overhead_ms=per_cell_checkpoint_overhead_ms,
        per_cell_total_overhead_ms=per_cell_total_overhead_ms,
        per_cell_memory_overhead_mb=per_cell_memory_overhead_mb,
        per_cell_checkpoint_mb=per_cell_checkpoint_mb,
        checking_clean_cells=checking_clean_cells,
        checking_stale_cells=checking_stale_cells,
        checking_error_cells=checking_error_cells,
        checking_reason_counts=checking_reason_counts,
        checking_error_counts=checking_error_counts,
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
            slowdown_p99=0.0,
            state_overhead_pct_mean=0.0,
            check_overhead_pct_mean=0.0,
            memory_overhead_pct_mean=0.0,
        )

    slowdowns = np.array([s.slowdown for s in stats_list])
    memory_overhead_ratios = np.array([s.memory_overhead_ratio for s in stats_list])
    # Use last cell overhead percentages for aggregate stats
    state_pcts = np.array([s.last_cell_state_overhead_pct for s in stats_list])
    check_pcts = np.array([s.last_cell_check_overhead_pct for s in stats_list])
    memory_pcts = np.array([s.last_cell_memory_overhead_pct for s in stats_list])

    # Collect per-notebook peak memory overhead percentages
    peak_memory_pcts = [s.peak_memory_overhead_pct for s in stats_list]

    # Collect all per-cell data across all files for aggregate per-cell statistics
    all_checkpoint_overhead = []
    all_total_overhead = []
    all_memory_overhead = []
    all_checkpoint_mb = []
    for s in stats_list:
        all_checkpoint_overhead.extend(s.per_cell_checkpoint_overhead_ms)
        all_total_overhead.extend(s.per_cell_total_overhead_ms)
        all_memory_overhead.extend(s.per_cell_memory_overhead_mb)
        all_checkpoint_mb.extend(s.per_cell_checkpoint_mb)

    # Compute per-cell statistics
    checkpoint_arr = (
        np.array(all_checkpoint_overhead)
        if all_checkpoint_overhead
        else np.array([0.0])
    )
    total_arr = np.array(all_total_overhead) if all_total_overhead else np.array([0.0])
    memory_arr = (
        np.array(all_memory_overhead) if all_memory_overhead else np.array([0.0])
    )
    checkpoint_mb_arr = (
        np.array(all_checkpoint_mb) if all_checkpoint_mb else np.array([0.0])
    )

    # Aggregate checking results (staleness summary)
    total_clean_cells = sum(s.checking_clean_cells for s in stats_list)
    total_stale_cells = sum(s.checking_stale_cells for s in stats_list)
    total_error_cells = sum(s.checking_error_cells for s in stats_list)
    total_reason_counts: Dict[str, int] = {}
    total_error_counts: Dict[str, int] = {}
    for s in stats_list:
        for rtype, count in s.checking_reason_counts.items():
            total_reason_counts[rtype] = total_reason_counts.get(rtype, 0) + count
        for etype, count in s.checking_error_counts.items():
            total_error_counts[etype] = total_error_counts.get(etype, 0) + count

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
        slowdown_p99=float(np.percentile(slowdowns, 99)),
        state_overhead_pct_mean=float(np.mean(state_pcts)),
        check_overhead_pct_mean=float(np.mean(check_pcts)),
        memory_overhead_pct_mean=float(np.mean(memory_pcts)),
        memory_overhead_pct_median=float(np.median(memory_pcts)),
        memory_overhead_pct_p90=float(np.percentile(memory_pcts, 90)),
        memory_overhead_pct_p95=float(np.percentile(memory_pcts, 95)),
        # Memory overhead ratio (like slowdown)
        memory_overhead_mean=float(np.mean(memory_overhead_ratios)),
        memory_overhead_median=float(np.median(memory_overhead_ratios)),
        memory_overhead_std=float(np.std(memory_overhead_ratios)),
        memory_overhead_min=float(np.min(memory_overhead_ratios)),
        memory_overhead_max=float(np.max(memory_overhead_ratios)),
        memory_overhead_p90=float(np.percentile(memory_overhead_ratios, 90)),
        memory_overhead_p95=float(np.percentile(memory_overhead_ratios, 95)),
        memory_overhead_p99=float(np.percentile(memory_overhead_ratios, 99)),
        # Per-cell checkpoint overhead (ms)
        checkpoint_overhead_per_cell_mean=float(np.mean(checkpoint_arr)),
        checkpoint_overhead_per_cell_median=float(np.median(checkpoint_arr)),
        checkpoint_overhead_per_cell_min=float(np.min(checkpoint_arr)),
        checkpoint_overhead_per_cell_max=float(np.max(checkpoint_arr)),
        checkpoint_overhead_per_cell_p90=float(np.percentile(checkpoint_arr, 90)),
        checkpoint_overhead_per_cell_p95=float(np.percentile(checkpoint_arr, 95)),
        checkpoint_overhead_per_cell_p99=float(np.percentile(checkpoint_arr, 99)),
        # Per-cell total overhead (ms)
        total_overhead_per_cell_mean=float(np.mean(total_arr)),
        total_overhead_per_cell_median=float(np.median(total_arr)),
        total_overhead_per_cell_min=float(np.min(total_arr)),
        total_overhead_per_cell_max=float(np.max(total_arr)),
        total_overhead_per_cell_p90=float(np.percentile(total_arr, 90)),
        total_overhead_per_cell_p95=float(np.percentile(total_arr, 95)),
        total_overhead_per_cell_p99=float(np.percentile(total_arr, 99)),
        # Per-cell memory overhead (MB)
        memory_overhead_per_cell_mean=float(np.mean(memory_arr)),
        memory_overhead_per_cell_median=float(np.median(memory_arr)),
        memory_overhead_per_cell_min=float(np.min(memory_arr)),
        memory_overhead_per_cell_max=float(np.max(memory_arr)),
        memory_overhead_per_cell_p90=float(np.percentile(memory_arr, 90)),
        memory_overhead_per_cell_p95=float(np.percentile(memory_arr, 95)),
        memory_overhead_per_cell_p99=float(np.percentile(memory_arr, 99)),
        # Raw per-cell data for histograms
        all_total_overhead_per_cell=list(total_arr),
        all_memory_overhead_per_cell=list(memory_arr),
        all_checkpoint_mb_per_cell=list(checkpoint_mb_arr),
        all_peak_memory_overhead_pct=peak_memory_pcts,
        # Aggregate checking results
        total_checking_clean_cells=total_clean_cells,
        total_checking_stale_cells=total_stale_cells,
        total_checking_error_cells=total_error_cells,
        total_checking_reason_counts=total_reason_counts,
        total_checking_error_counts=total_error_counts,
    )


def format_table(stats_list: List[FileStats], aggregate: AggregateStats) -> str:
    """Format results as ASCII table."""
    lines = []
    lines.append("=" * 110)
    lines.append("FLOWBOOK OVERHEAD COMPARISON")
    lines.append("=" * 110)
    lines.append(f"Notebooks: {aggregate.num_files}")
    lines.append("=" * 110)
    lines.append("")

    # Header
    header = f"{'Notebook':<30} {'Cells':>5} {'Trials':>6} {'Baseline':>10} {'FlowBook':>10} {'State':>8} {'Check':>8} {'Slowdown':>10}"
    lines.append(header)
    lines.append("-" * 110)

    # Per-file rows
    for s in stats_list:
        name = s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
        row = f"{name:<30} {s.num_cells:>5} {s.num_trials:>6} {s.baseline_runtime_ms:>9.0f}ms {s.flowbook_total_ms:>9.0f}ms {s.state_overhead_ms:>7.0f}ms {s.check_overhead_ms:>7.0f}ms {s.slowdown:>9.2f}x"
        lines.append(row)

    lines.append("-" * 110)
    lines.append("")

    # Show rerun stats if any file has reruns
    has_reruns = any(s.num_reruns > 0 for s in stats_list)
    if has_reruns:
        lines.append("RERUN STATISTICS")
        lines.append("-" * 110)
        header = f"{'Notebook':<30} {'Reruns':>6} {'Trials':>6} {'Baseline':>10} {'FlowBook':>10} {'State':>8} {'Check':>8} {'Final Ckpt':>12}"
        lines.append(header)
        lines.append("-" * 110)
        for s in stats_list:
            if s.num_reruns > 0:
                name = (
                    s.notebook_name[:28]
                    if len(s.notebook_name) > 28
                    else s.notebook_name
                )
                ckpt_mb = s.rerun_final_checkpoint_bytes / (1024 * 1024)
                row = f"{name:<30} {s.num_reruns:>6} {s.num_trials:>6} {s.rerun_baseline_runtime_ms:>9.0f}ms {s.rerun_flowbook_total_ms:>9.0f}ms {s.rerun_state_overhead_ms:>7.0f}ms {s.rerun_check_overhead_ms:>7.0f}ms {ckpt_mb:>10.1f}MB"
                lines.append(row)
        lines.append("-" * 110)
        lines.append("")

    # Per-benchmark overhead table
    lines.append("PER-BENCHMARK OVERHEAD SUMMARY")
    lines.append("-" * 110)
    header = f"{'Notebook':<30} {'Total OH Mean':>12} {'Total OH Med':>12} {'Total OH Max':>12} {'Mem OH Mean':>12} {'Mem OH Med':>12} {'Mem OH Max':>12}"
    lines.append(header)
    lines.append("-" * 110)
    for s in stats_list:
        name = s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
        if s.per_cell_total_overhead_ms:
            total_mean = np.mean(s.per_cell_total_overhead_ms)
            total_med = np.median(s.per_cell_total_overhead_ms)
            total_max = np.max(s.per_cell_total_overhead_ms)
        else:
            total_mean = total_med = total_max = 0.0
        if s.per_cell_memory_overhead_mb:
            mem_mean = np.mean(s.per_cell_memory_overhead_mb)
            mem_med = np.median(s.per_cell_memory_overhead_mb)
            mem_max = np.max(s.per_cell_memory_overhead_mb)
        else:
            mem_mean = mem_med = mem_max = 0.0
        row = f"{name:<30} {total_mean:>10.1f}ms {total_med:>10.1f}ms {total_max:>10.1f}ms {mem_mean:>10.1f}MB {mem_med:>10.1f}MB {mem_max:>10.1f}MB"
        lines.append(row)
    lines.append("-" * 110)
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
    lines.append(f"  P99 Slowdown:       {aggregate.slowdown_p99:.3f}x")
    lines.append("")
    lines.append("PER-CELL CHECKPOINT OVERHEAD (ms)")
    lines.append(f"  Mean:   {aggregate.checkpoint_overhead_per_cell_mean:.2f}ms")
    lines.append(f"  Median: {aggregate.checkpoint_overhead_per_cell_median:.2f}ms")
    lines.append(f"  Min:    {aggregate.checkpoint_overhead_per_cell_min:.2f}ms")
    lines.append(f"  Max:    {aggregate.checkpoint_overhead_per_cell_max:.2f}ms")
    lines.append(f"  P90:    {aggregate.checkpoint_overhead_per_cell_p90:.2f}ms")
    lines.append(f"  P95:    {aggregate.checkpoint_overhead_per_cell_p95:.2f}ms")
    lines.append(f"  P99:    {aggregate.checkpoint_overhead_per_cell_p99:.2f}ms")
    lines.append("")
    lines.append("PER-CELL TOTAL OVERHEAD (ms)")
    lines.append(f"  Mean:   {aggregate.total_overhead_per_cell_mean:.2f}ms")
    lines.append(f"  Median: {aggregate.total_overhead_per_cell_median:.2f}ms")
    lines.append(f"  Min:    {aggregate.total_overhead_per_cell_min:.2f}ms")
    lines.append(f"  Max:    {aggregate.total_overhead_per_cell_max:.2f}ms")
    lines.append(f"  P90:    {aggregate.total_overhead_per_cell_p90:.2f}ms")
    lines.append(f"  P95:    {aggregate.total_overhead_per_cell_p95:.2f}ms")
    lines.append(f"  P99:    {aggregate.total_overhead_per_cell_p99:.2f}ms")
    lines.append("")
    lines.append("PER-CELL MEMORY OVERHEAD RATIO (checkpoint / user_ns)")
    lines.append(f"  Mean:   {aggregate.memory_overhead_per_cell_mean:.2f}")
    lines.append(f"  Median: {aggregate.memory_overhead_per_cell_median:.2f}")
    lines.append(f"  Min:    {aggregate.memory_overhead_per_cell_min:.2f}")
    lines.append(f"  Max:    {aggregate.memory_overhead_per_cell_max:.2f}")
    lines.append(f"  P90:    {aggregate.memory_overhead_per_cell_p90:.2f}")
    lines.append(f"  P95:    {aggregate.memory_overhead_per_cell_p95:.2f}")
    lines.append(f"  P99:    {aggregate.memory_overhead_per_cell_p99:.2f}")
    # Check if we have baseline memory data (ratio > 0 and not ~1.0 from checkpoint fraction)
    has_baseline_memory = any(s.baseline_memory_bytes > 0 for s in stats_list)
    lines.append("")
    if has_baseline_memory:
        lines.append("MEMORY OVERHEAD")
        lines.append(f"AGGREGATE (N={aggregate.num_files})")
        lines.append(f"  Mean Overhead:      {aggregate.memory_overhead_mean:.3f}x")
        lines.append(f"  Median Overhead:    {aggregate.memory_overhead_median:.3f}x")
        lines.append(f"  Std Dev:            {aggregate.memory_overhead_std:.3f}")
        lines.append(f"  Min Overhead:       {aggregate.memory_overhead_min:.3f}x")
        lines.append(f"  Max Overhead:       {aggregate.memory_overhead_max:.3f}x")
        lines.append(f"  P90 Overhead:       {aggregate.memory_overhead_p90:.3f}x")
        lines.append(f"  P95 Overhead:       {aggregate.memory_overhead_p95:.3f}x")
        lines.append(f"  P99 Overhead:       {aggregate.memory_overhead_p99:.3f}x")
    else:
        # No baseline memory - show checkpoint overhead ratio
        lines.append("CHECKPOINT MEMORY RATIO")
        lines.append(f"AGGREGATE (N={aggregate.num_files})")
        lines.append(
            f"  Per-Cell Mean:      {aggregate.memory_overhead_per_cell_mean:.2f}"
        )
        lines.append(
            f"  Per-Cell Median:    {aggregate.memory_overhead_per_cell_median:.2f}"
        )
        lines.append(
            f"  Per-Cell Max:       {aggregate.memory_overhead_per_cell_max:.2f}"
        )
        lines.append(
            f"  Per-Cell P99:       {aggregate.memory_overhead_per_cell_p99:.2f}"
        )

    # Checking results summary (staleness)
    # Note: Staleness consistency across trials is checked during averaging in average_trial_data
    total_checked = (
        aggregate.total_checking_clean_cells
        + aggregate.total_checking_stale_cells
        + aggregate.total_checking_error_cells
    )
    if total_checked > 0:
        lines.append("")
        lines.append("CHECKING RESULTS")
        lines.append(f"  Clean cells:        {aggregate.total_checking_clean_cells}")
        lines.append(f"  Stale cells:        {aggregate.total_checking_stale_cells}")
        lines.append(f"  Error cells:        {aggregate.total_checking_error_cells}")
        if aggregate.total_checking_reason_counts:
            lines.append("  Staleness reasons:")
            for rtype, count in sorted(aggregate.total_checking_reason_counts.items()):
                lines.append(f"    {rtype}: {count}")
        if aggregate.total_checking_error_counts:
            lines.append("  Error types:")
            for etype, count in sorted(aggregate.total_checking_error_counts.items()):
                lines.append(f"    {etype}: {count}")

    # Per-notebook error table (if any notebook has errors)
    any_errors = any(s.checking_error_cells > 0 for s in stats_list)
    if any_errors:
        # Collect all error types across all notebooks
        all_error_types = set()
        for s in stats_list:
            all_error_types.update(s.checking_error_counts.keys())
        error_types = sorted(all_error_types)

        lines.append("")
        lines.append("PER-NOTEBOOK ERROR SUMMARY")
        lines.append("-" * 110)

        # Build header with dynamic error type columns
        header = f"{'Notebook':<30} {'Cells':>5} {'Errors':>6}"
        for etype in error_types:
            # Shorten error type names for column headers
            short_name = etype.replace("no_", "").replace("_", " ")[:12]
            header += f" {short_name:>12}"
        lines.append(header)
        lines.append("-" * 110)

        # Per-file rows (all files)
        for s in stats_list:
            name = (
                s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
            )
            row = f"{name:<30} {s.num_cells:>5} {s.checking_error_cells:>6}"
            for etype in error_types:
                count = s.checking_error_counts.get(etype, 0)
                row += f" {count:>12}"
            lines.append(row)

        lines.append("-" * 110)

    lines.append("=" * 110)

    return "\n".join(lines)


def format_json_output(stats_list: List[FileStats], aggregate: AggregateStats) -> str:
    """Format results as JSON."""
    output = {
        "files": [
            {
                "notebook_path": s.notebook_path,
                "notebook_name": s.notebook_name,
                "num_cells": s.num_cells,
                "num_trials": s.num_trials,
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
                "p99": aggregate.slowdown_p99,
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
    lines.append(
        "notebook,cells,trials,baseline_ms,flowbook_ms,state_ms,check_ms,slowdown,state_pct,check_pct,memory_pct"
    )

    for s in stats_list:
        lines.append(
            f'"{s.notebook_name}",{s.num_cells},{s.num_trials},{s.baseline_runtime_ms:.1f},{s.flowbook_total_ms:.1f},'
            f"{s.state_overhead_ms:.1f},{s.check_overhead_ms:.1f},{s.slowdown:.3f},"
            f"{s.state_overhead_pct:.1f},{s.check_overhead_pct:.1f},{s.memory_overhead_pct:.1f}"
        )

    return "\n".join(lines)


def plot_slowdown(
    data: Dict[str, Any], output_path: str, large_fonts: bool = False
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
    baseline_runtimes = [
        cell_data[cid].get("baseline_runtime_ms", 0) for cid in cell_ids
    ]
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
    ax.fill_between(
        cells,
        0,
        baseline_cumsum / 1000,
        alpha=0.3,
        color=colors[0],
        label="Cell Run Time",
    )
    ax.fill_between(
        cells,
        baseline_cumsum / 1000,
        (baseline_cumsum + state_cumsum) / 1000,
        alpha=0.3,
        color=colors[1],
        label="State Checkpoint",
    )
    ax.fill_between(
        cells,
        (baseline_cumsum + state_cumsum) / 1000,
        total_cumsum / 1000,
        alpha=0.3,
        color=colors[2],
        label="Reproducibility Check",
    )

    # Lines with markers
    ax.plot(
        cells,
        baseline_cumsum / 1000,
        color=colors[0],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.plot(
        cells,
        (baseline_cumsum + state_cumsum) / 1000,
        color=colors[1],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.plot(
        cells,
        total_cumsum / 1000,
        color=colors[2],
        linewidth=2,
        marker="o",
        markersize=4,
    )

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)

    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    ax.set_title(
        f"Cumulative Cell Run and Checkpointing Times\n{notebook_name}",
        fontsize=title_size,
    )
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    if large_fonts:
        ax.tick_params(axis="both", labelsize=tick_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Slowdown plot saved to: {output_path}")


def plot_memory_comparison(
    data: Dict[str, Any], output_path: str, large_fonts: bool = False
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

    ax.fill_between(
        cells, 0, user_mb, alpha=0.3, color=colors[0], label="User Namespace"
    )
    ax.fill_between(
        cells,
        user_mb,
        total_mb,
        alpha=0.3,
        color=colors[1],
        label="Checkpoint Overhead",
    )

    ax.plot(cells, user_mb, color=colors[0], linewidth=2, marker="o", markersize=4)
    ax.plot(cells, total_mb, color=colors[1], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Memory (MB)", fontsize=label_size)

    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    ax.set_title(f"Memory Usage\n{notebook_name}", fontsize=title_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    if large_fonts:
        ax.tick_params(axis="both", labelsize=tick_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Memory plot saved to: {output_path}")


def extract_checkpoint_type_data(
    data: Dict[str, Any], top_n: int = 10
) -> Optional[Dict[str, Any]]:
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
    types_ordered = sorted(
        type_totals.keys(), key=lambda t: type_totals[t], reverse=True
    )

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


def extract_checkpoint_type_data_v2(
    data: Dict[str, Any], top_n: int = 10
) -> Optional[Dict[str, Any]]:
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
        type_final: Dict[str, int] = {
            t: type_by_cell[t][-1] if type_by_cell[t] else 0 for t in all_type_names
        }
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
    data: Dict[str, Any], output_path: Optional[str] = None, large_fonts: bool = False
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
        ax.fill_between(
            cells, cumulative, cumulative + data_mb, alpha=0.7, color=colors[i], label=t
        )
        cumulative = cumulative + data_mb

    # Total line on top
    ax.plot(
        cells, cumulative, color="black", linewidth=1.5, linestyle="--", label="Total"
    )

    # Add separator for rerun phase if present
    initial_count = type_data.get("initial_count", len(cells))
    if initial_count < len(cells):
        ax.axvline(
            x=initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

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
        ax.tick_params(axis="both", labelsize=tick_size)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Checkpoint types plot saved to: {output_path}")
        return None
    else:
        return fig


def plot_combined(
    data: Dict[str, Any], output_path: Optional[str] = None, large_fonts: bool = True
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
            cell_data[c["cell_id"]]["total_bytes"] = c.get(
                "user_ns_and_checkpoint_bytes", 0
            )

    cell_ids = list(cell_data.keys())
    baseline_runtimes = [
        cell_data[cid].get("baseline_runtime_ms", 0) for cid in cell_ids
    ]
    flowbook_runtimes = [
        cell_data[cid].get("flowbook_runtime_ms", 0) for cid in cell_ids
    ]
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
    ax.plot(
        cells,
        baseline_cumsum / 1000,
        color=colors[0],
        linewidth=2,
        marker="o",
        markersize=4,
        label="Baseline",
    )
    ax.plot(
        cells,
        flowbook_cumsum / 1000,
        color=colors[1],
        linewidth=2,
        marker="o",
        markersize=4,
        label="FlowBook",
    )

    # Add separator for rerun phase
    if initial_count < len(cells):
        ax.axvline(
            x=initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

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
    ax.tick_params(axis="both", labelsize=tick_size)

    # Add overhead percentage label at end of run
    if baseline_cumsum[-1] > 0:
        time_overhead_pct = (
            (flowbook_cumsum[-1] - baseline_cumsum[-1]) / baseline_cumsum[-1] * 100
        )
        ax.annotate(
            f"{time_overhead_pct:.1f}% overhead",
            xy=(cells[-1], flowbook_cumsum[-1] / 1000),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=legend_size,
            va="center",
            ha="left",
            color=colors[1],
        )

    # Panel 2: Memory
    ax = axes[1]
    ax.fill_between(
        cells, 0, user_mb, alpha=0.3, color=colors[0], label="User Namespace"
    )
    ax.fill_between(
        cells,
        user_mb,
        total_mb,
        alpha=0.3,
        color=colors[1],
        label="Checkpoint Overhead",
    )

    ax.plot(cells, user_mb, color=colors[0], linewidth=2, marker="o", markersize=4)
    ax.plot(cells, total_mb, color=colors[1], linewidth=2, marker="o", markersize=4)

    # Add separator for rerun phase
    if initial_count < len(cells):
        ax.axvline(
            x=initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Memory (MB)", fontsize=label_size)
    title = "Memory Usage"
    if initial_count < len(cells):
        title += f" (cells 1-{initial_count} + {len(cells) - initial_count} reruns)"
    ax.set_title(title, fontsize=title_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Add overhead percentage label at end of run
    if user_mb[-1] > 0:
        mem_overhead_pct = (total_mb[-1] - user_mb[-1]) / user_mb[-1] * 100
        ax.annotate(
            f"{mem_overhead_pct:.1f}% overhead",
            xy=(cells[-1], total_mb[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=legend_size,
            va="center",
            ha="left",
            color=colors[1],
        )

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
            ax.fill_between(
                type_cells,
                cumulative,
                cumulative + data_mb,
                alpha=0.7,
                color=type_colors[i],
                label=t,
            )
            cumulative = cumulative + data_mb

        # Total line on top
        ax.plot(
            type_cells,
            cumulative,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Total",
        )

        # Add separator for rerun phase if present
        type_initial_count = type_data.get("initial_count", len(type_cells))
        if type_initial_count < len(type_cells):
            ax.axvline(
                x=type_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )

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
        ax.tick_params(axis="both", labelsize=tick_size)

    # Add notebook name as figure title
    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    fig.suptitle(notebook_name, fontsize=title_size + 2, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Combined plot saved to: {output_path}")
        return None
    else:
        return fig


def extract_checkpoint_var_data(
    data: Dict[str, Any], top_n: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Extract per-variable checkpoint memory from comparison data.

    Uses cumulative_by_var field when available (TRUE cumulative accounting for sharing).
    Otherwise uses checkpoint_by_var which contains the CURRENT checkpoint state at each cell.
    Falls back to deriving from checkpoint_var_costs for backwards compatibility.

    Args:
        data: Comparison data dict
        top_n: Number of top variables to show individually (rest aggregated as "other")

    Returns dict with:
        cells: list of cell indices
        by_var: dict mapping variable name to list of bytes per cell
        vars_ordered: list of variable names ordered by max size descending
        initial_count: number of initial execution cells (before reruns)

    Returns None if no checkpoint data available.
    """
    if not (is_v2_format(data) or is_v3_format(data)):
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

        # Get MAX cumulative for ordering (captures peak contribution, not just final)
        var_max: Dict[str, int] = {
            v: max(var_by_cell[v]) if var_by_cell[v] else 0 for v in all_var_names
        }
    else:
        # Try checkpoint_by_var first (contains correct CURRENT state in MB)
        has_by_var = any(c.get("checkpoint_by_var") for c in memory_cells)

        if has_by_var:
            all_var_names: set = set()
            for c in memory_cells:
                by_var = c.get("checkpoint_by_var") or {}
                all_var_names.update(by_var.keys())

            if not all_var_names:
                return None

            var_by_cell: Dict[str, List[int]] = {v: [] for v in all_var_names}
            mb_to_bytes = 1024 * 1024

            for c in memory_cells:
                by_var = c.get("checkpoint_by_var") or {}

                for var_name in all_var_names:
                    # checkpoint_by_var is in MB, convert to bytes for consistency
                    var_mb = by_var.get(var_name, 0)
                    var_by_cell[var_name].append(int(var_mb * mb_to_bytes))

            # Get MAX for ordering
            var_max: Dict[str, int] = {
                v: max(var_by_cell[v]) if var_by_cell[v] else 0 for v in all_var_names
            }
        else:
            # Fall back to old method - derives from checkpoint_var_costs (may overcount)
            has_costs = any(c.get("checkpoint_var_costs") for c in memory_cells)

            if not has_costs:
                return None

            # First pass: collect all variable names
            all_var_names: set = set()
            for c in memory_cells:
                costs = c.get("checkpoint_var_costs") or {}
                all_var_names.update(costs.keys())

            if not all_var_names:
                return None

            # Initialize tracking - per-cell values per variable
            var_by_cell: Dict[str, List[int]] = {v: [] for v in all_var_names}

            # Second pass: collect per-cell values (NOT cumulative)
            # checkpoint_var_costs contains the current checkpoint state at each cell, not deltas
            for c in memory_cells:
                costs = c.get("checkpoint_var_costs") or {}

                for var_name in all_var_names:
                    # Use per-cell value directly (checkpoint_var_costs is current state, not delta)
                    var_bytes = costs.get(var_name, {}).get("bytes", 0)
                    var_by_cell[var_name].append(var_bytes)

            # Get MAX for ordering (captures peak contribution)
            var_max: Dict[str, int] = {
                v: max(var_by_cell[v]) if var_by_cell[v] else 0 for v in all_var_names
            }

    # Order variables by MAX cumulative size descending (not final - captures vars that get cleaned up)
    vars_ordered = sorted(var_max.keys(), key=lambda v: var_max[v], reverse=True)

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


def extract_checkpoint_timing_var_data(
    data: Dict[str, Any], top_n: int = 10
) -> Optional[Dict[str, Any]]:
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
    if not (is_v2_format(data) or is_v3_format(data)):
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
    top_n: int = 10,
) -> Optional[Any]:
    """
    Create combined 6-panel plot (2x3 grid) for v2.0 data with HeapSizer memory:
    - Row 1: Timing Comparison | Checkpoint Time by Variable
    - Row 2: Memory Overhead | Checkpoint Memory by Variable
    - Row 3: Overhead Time per Cell | Checkpoint Memory Overhead per Cell

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
    baseline_rerun_cells = (
        baseline_timing.get("rerun_cells", []) if baseline_timing else []
    )
    flowbook_rerun_cells = (
        flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    )
    baseline_cells = baseline_initial_cells + baseline_rerun_cells
    flowbook_cells = flowbook_initial_cells + flowbook_rerun_cells
    timing_initial_count = len(baseline_initial_cells)

    # Extract memory data (including reruns)
    baseline_memory = baseline.get("memory", {})
    flowbook_memory = flowbook.get("memory", {})
    baseline_mem_initial = baseline_memory.get("cells", []) if baseline_memory else []
    flowbook_mem_initial = flowbook_memory.get("cells", []) if flowbook_memory else []
    baseline_mem_rerun = (
        baseline_memory.get("rerun_cells", []) if baseline_memory else []
    )
    flowbook_mem_rerun = (
        flowbook_memory.get("rerun_cells", []) if flowbook_memory else []
    )
    baseline_mem_cells = baseline_mem_initial + baseline_mem_rerun
    flowbook_mem_cells = flowbook_mem_initial + flowbook_mem_rerun
    memory_initial_count = len(baseline_mem_initial)

    # Font sizes
    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    # Check data availability
    has_memory = bool(flowbook_mem_cells)  # FlowBook memory data is sufficient
    has_baseline_memory = bool(baseline_mem_cells)
    timing_var_data = extract_checkpoint_timing_var_data(data, top_n=top_n)
    var_data = extract_checkpoint_var_data(data, top_n=top_n)

    # Always use 2x3 grid layout
    fig, axes_2d = plt.subplots(3, 2, figsize=(14, 18))
    axes = [
        axes_2d[0, 0],
        axes_2d[0, 1],  # Row 1: Timing, Checkpoint Time by Variable
        axes_2d[1, 0],
        axes_2d[1, 1],  # Row 2: Memory, Checkpoint Memory by Variable
        axes_2d[2, 0],
        axes_2d[2, 1],  # Row 3: Overhead per Cell, Memory Overhead per Cell
    ]

    # Prepare shared data for timing
    cell_data_map = {}
    has_baseline = bool(baseline_cells)
    if baseline_cells and flowbook_cells:
        # Both baseline and flowbook available - align by cell_id
        for c in baseline_cells:
            cell_data_map[c["cell_id"]] = {
                "baseline_ms": c.get("execute_duration_ms", c.get("cell_runtime_ms", 0))
            }
        for c in flowbook_cells:
            if c["cell_id"] in cell_data_map:
                execute_ms = c.get("execute_duration_ms", c.get("cell_runtime_ms", 0))
                state_ms = c.get("state_duration_ms", 0)
                check_ms = c.get("check_duration_ms", 0)
                # If code_duration_ms is not available, derive it from execute - state - check
                # This ensures "other" overhead is 0 when we don't have a separate code timing
                code_ms = c.get("code_duration_ms")
                if code_ms is None:
                    code_ms = max(execute_ms - state_ms - check_ms, 0)
                cell_data_map[c["cell_id"]]["execute_ms"] = execute_ms
                cell_data_map[c["cell_id"]]["code_ms"] = code_ms
                cell_data_map[c["cell_id"]]["state_ms"] = state_ms
                cell_data_map[c["cell_id"]]["check_ms"] = check_ms
    elif flowbook_cells:
        # FlowBook only - use code_duration_ms as the "baseline" for comparison
        for c in flowbook_cells:
            execute_ms = c.get("execute_duration_ms", c.get("cell_runtime_ms", 0))
            state_ms = c.get("state_duration_ms", 0)
            check_ms = c.get("check_duration_ms", 0)
            code_ms = c.get("code_duration_ms")
            if code_ms is None:
                code_ms = max(execute_ms - state_ms - check_ms, 0)
            cell_data_map[c["cell_id"]] = {
                "baseline_ms": code_ms,  # Use code time as "baseline" when no baseline kernel
                "execute_ms": execute_ms,
                "code_ms": code_ms,
                "state_ms": state_ms,
                "check_ms": check_ms,
            }

    cell_ids = list(cell_data_map.keys())
    cells = np.arange(1, len(cell_ids) + 1) if cell_ids else np.array([])

    # Arrays for timing data
    baseline_arr = (
        np.array([cell_data_map[cid].get("baseline_ms", 0) for cid in cell_ids])
        if cell_ids
        else np.array([])
    )
    code_arr = (
        np.array([cell_data_map[cid].get("code_ms", 0) for cid in cell_ids])
        if cell_ids
        else np.array([])
    )
    state_arr = (
        np.array([cell_data_map[cid].get("state_ms", 0) for cid in cell_ids])
        if cell_ids
        else np.array([])
    )
    check_arr = (
        np.array([cell_data_map[cid].get("check_ms", 0) for cid in cell_ids])
        if cell_ids
        else np.array([])
    )
    execute_arr = (
        np.array([cell_data_map[cid].get("execute_ms", 0) for cid in cell_ids])
        if cell_ids
        else np.array([])
    )
    other_arr = (
        np.maximum(execute_arr - (code_arr + state_arr + check_arr), 0)
        if cell_ids
        else np.array([])
    )

    # Cumulative sums
    baseline_cumsum = np.cumsum(baseline_arr) if len(baseline_arr) > 0 else np.array([])
    code_cumsum = np.cumsum(code_arr) if len(code_arr) > 0 else np.array([])
    state_cumsum = np.cumsum(state_arr) if len(state_arr) > 0 else np.array([])
    check_cumsum = np.cumsum(check_arr) if len(check_arr) > 0 else np.array([])
    other_cumsum = np.cumsum(other_arr) if len(other_arr) > 0 else np.array([])

    # ========== Panel 1: Timing Comparison (top-left) ==========
    ax = axes[0]
    if len(cells) > 0:
        # Plot baseline/code line (baseline if available, otherwise code time)
        baseline_label = "Baseline" if has_baseline else "Code (no baseline)"
        ax.plot(
            cells,
            baseline_cumsum / 1000,
            color=colors[0],
            linewidth=2,
            marker="o",
            markersize=4,
            label=baseline_label,
        )

        # FlowBook as stacked area: code (bottom) + state + check + other (top)
        ax.fill_between(
            cells,
            0,
            code_cumsum / 1000,
            alpha=0.3,
            color=colors[1],
            label="FlowBook Code",
        )
        ax.fill_between(
            cells,
            code_cumsum / 1000,
            (code_cumsum + state_cumsum) / 1000,
            alpha=0.4,
            color=colors[2],
            label="State",
        )
        ax.fill_between(
            cells,
            (code_cumsum + state_cumsum) / 1000,
            (code_cumsum + state_cumsum + check_cumsum) / 1000,
            alpha=0.4,
            color=colors[3],
            label="Check",
        )
        ax.fill_between(
            cells,
            (code_cumsum + state_cumsum + check_cumsum) / 1000,
            (code_cumsum + state_cumsum + check_cumsum + other_cumsum) / 1000,
            alpha=0.4,
            color=colors[4],
            label="Other",
        )

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)
        title = "Timing Comparison" if has_baseline else "Timing (FlowBook only)"
        if timing_initial_count < len(cells):
            title += f" (cells 1-{timing_initial_count} + {len(cells) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=tick_size)

        # Add separator line for rerun phase
        if timing_initial_count < len(cells):
            ax.axvline(
                x=timing_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )

        ax.legend(loc="upper left", fontsize=legend_size)

        # Add timing breakdown text box
        total_code_s = code_cumsum[-1] / 1000
        total_state_s = state_cumsum[-1] / 1000
        total_check_s = check_cumsum[-1] / 1000
        total_other_s = other_cumsum[-1] / 1000
        total_flowbook_s = total_code_s + total_state_s + total_check_s + total_other_s
        total_baseline_s = baseline_cumsum[-1] / 1000

        if has_baseline:
            textstr = (
                f"Baseline: {total_baseline_s:.2f}s\nFlowBook: {total_flowbook_s:.2f}s"
            )
        else:
            textstr = f"Code: {total_code_s:.2f}s\nTotal: {total_flowbook_s:.2f}s"
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        ax.text(
            0.02,
            0.70,
            textstr,
            transform=ax.transAxes,
            fontsize=legend_size,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=props,
        )

        # Annotate overhead percentage relative to FlowBook Code time
        total_overhead_s = total_state_s + total_check_s + total_other_s
        if total_code_s > 0:
            overhead_pct = (total_overhead_s / total_code_s) * 100
            ax.annotate(
                f"{overhead_pct:.1f}% overhead (vs code)",
                xy=(cells[-1], total_flowbook_s),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=legend_size,
                va="center",
                ha="left",
                color=colors[1],
            )
    else:
        ax.text(
            0.5, 0.5, "No timing data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Timing Comparison", fontsize=title_size)

    # ========== Panel 2: Checkpoint Time by Variable (top-right) ==========
    ax = axes[1]
    if timing_var_data is not None:
        var_colors = sns.color_palette("husl", len(timing_var_data["vars_ordered"]))
        timing_cells = np.array(timing_var_data["cells"])
        timing_var_types = timing_var_data.get("var_types", {})

        # Stack checkpoint timing by variable (ms -> seconds)
        stacked = [
            np.array(timing_var_data["by_var"][v]) / 1000
            for v in timing_var_data["vars_ordered"]
        ]
        cumulative_timing = np.zeros(len(timing_cells))
        for i, (v, data_sec) in enumerate(
            zip(timing_var_data["vars_ordered"], stacked)
        ):
            var_type = timing_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(
                timing_cells,
                cumulative_timing,
                cumulative_timing + data_sec,
                alpha=0.7,
                color=var_colors[i],
                label=label,
            )
            cumulative_timing = cumulative_timing + data_sec

        # Draw total line
        ax.plot(
            timing_cells,
            cumulative_timing,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Total",
        )

        # Add separator line for rerun phase
        timing_var_initial_count = timing_var_data.get(
            "initial_count", len(timing_cells)
        )
        if timing_var_initial_count < len(timing_cells):
            ax.axvline(
                x=timing_var_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )

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
    else:
        ax.text(
            0.5,
            0.5,
            "No checkpoint timing data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Checkpoint Time by Variable", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # ========== Panel 3: Memory Overhead (middle-left) ==========
    # Uses simplified fields: namespace_mb, checkpoint_cumulative_mb, gpu_mb
    ax = axes[2]
    if has_memory:
        mem_cells_arr = np.arange(1, len(flowbook_mem_cells) + 1)

        # Extract namespace, checkpoint, and GPU memory using new simplified fields
        # (with fallback to old field names for backward compatibility)
        namespace_mb = np.array(
            [
                c.get(
                    "namespace_mb",
                    c.get("current_footprint_mb", c.get("base_namespace_mb", 0)),
                )
                for c in flowbook_mem_cells
            ]
        )

        # Get checkpoint cumulative MB - try multiple sources
        def get_checkpoint_mb(c):
            # 1. Try explicit field - but only if > 0 (may be 0 when saved checkpoints empty)
            checkpoint_cumulative = c.get("checkpoint_cumulative_mb")
            if checkpoint_cumulative is not None and checkpoint_cumulative > 0:
                return checkpoint_cumulative
            # 2. Try overhead_breakdown.checkpoints_mb
            overhead = c.get("overhead_breakdown") or {}
            checkpoints_mb = overhead.get("checkpoints_mb")
            if checkpoints_mb is not None and checkpoints_mb > 0:
                return checkpoints_mb
            # 3. Fall back to summing cumulative_by_var (bytes -> MB)
            cumulative_by_var = c.get("cumulative_by_var") or {}
            if cumulative_by_var:
                return sum(cumulative_by_var.values()) / (1024 * 1024)
            # 4. Fall back to summing cumulative_by_type (bytes -> MB)
            cumulative_by_type = c.get("cumulative_by_type") or {}
            if cumulative_by_type:
                return sum(cumulative_by_type.values()) / (1024 * 1024)
            # 5. Fall back to summing checkpoint_var_costs (per-cell total)
            var_costs = c.get("checkpoint_var_costs") or {}
            if var_costs:
                return sum(v.get("bytes", 0) for v in var_costs.values()) / (
                    1024 * 1024
                )
            return 0

        checkpoint_cumulative_mb = np.array(
            [get_checkpoint_mb(c) for c in flowbook_mem_cells]
        )
        gpu_mb = np.array(
            [c.get("gpu_mb", c.get("gpu_mem_samples", 0)) for c in flowbook_mem_cells]
        )

        has_gpu = np.any(gpu_mb > 0)
        stack_colors = sns.color_palette("Set2", 5)

        # Layer 1: User namespace (bottom - gray)
        ax.fill_between(
            mem_cells_arr,
            0,
            namespace_mb,
            alpha=0.4,
            color="gray",
            label="User Namespace",
        )
        cumulative_mem = namespace_mb.copy()

        # Layer 2: GPU memory (if present)
        if has_gpu:
            next_level = cumulative_mem + gpu_mb
            ax.fill_between(
                mem_cells_arr,
                cumulative_mem,
                next_level,
                alpha=0.4,
                color="orange",
                label="GPU Memory",
            )
            cumulative_mem = next_level

        # Layer 3: Checkpoint overhead
        next_level = cumulative_mem + checkpoint_cumulative_mb
        ax.fill_between(
            mem_cells_arr,
            cumulative_mem,
            next_level,
            alpha=0.5,
            color="steelblue",
            label="Checkpoints",
        )
        cumulative_mem = next_level

        # Draw namespace line for reference
        ax.plot(mem_cells_arr, namespace_mb, color="gray", linewidth=2, linestyle="--")

        # Calculate and annotate PEAK overhead percentage
        base_mem = namespace_mb + gpu_mb  # Base = namespace + GPU
        peak_overhead = np.max(cumulative_mem - base_mem)
        peak_idx = np.argmax(cumulative_mem - base_mem)
        if base_mem[peak_idx] > 0:
            peak_overhead_pct = peak_overhead / base_mem[peak_idx] * 100
            ax.annotate(
                f"{peak_overhead_pct:.1f}% peak overhead",
                xy=(mem_cells_arr[peak_idx], cumulative_mem[peak_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=legend_size,
                va="bottom",
                ha="left",
                color=colors[1],
            )

        ax.set_title("Memory Overhead", fontsize=title_size)
        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Memory (MB)", fontsize=label_size)

        if memory_initial_count < len(mem_cells_arr):
            ax.axvline(
                x=memory_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )

        ax.legend(loc="upper left", fontsize=legend_size - 2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
    else:
        ax.text(
            0.5, 0.5, "No memory data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Memory Overhead", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # ========== Panel 4: Checkpoint Memory by Variable (middle-right) ==========
    ax = axes[3]
    if var_data is not None:
        var_colors = sns.color_palette("husl", len(var_data["vars_ordered"]))
        var_cells = np.array(var_data["cells"])
        mb = 1024 * 1024
        mem_var_types = var_data.get("var_types", {})

        # Get namespace memory for reference (use new simplified fields with fallback)
        namespace_var = np.zeros(len(var_cells))
        gpu_mem_var = np.zeros(len(var_cells))
        if has_baseline_memory and len(baseline_mem_cells) >= len(var_cells):
            namespace_var = np.array(
                [
                    c.get("namespace_mb", c.get("current_footprint_mb", 0))
                    for c in baseline_mem_cells[: len(var_cells)]
                ]
            )
            gpu_mem_var = np.array(
                [
                    c.get("gpu_mb", c.get("gpu_mem_samples", 0))
                    for c in baseline_mem_cells[: len(var_cells)]
                ]
            )
        elif has_memory and len(flowbook_mem_cells) >= len(var_cells):
            # Use FlowBook namespace
            namespace_var = np.array(
                [
                    c.get(
                        "namespace_mb",
                        c.get("current_footprint_mb", c.get("base_namespace_mb", 0)),
                    )
                    for c in flowbook_mem_cells[: len(var_cells)]
                ]
            )
            gpu_mem_var = np.array(
                [
                    c.get("gpu_mb", c.get("gpu_mem_samples", 0))
                    for c in flowbook_mem_cells[: len(var_cells)]
                ]
            )

        has_gpu_var = any(g > 0 for g in gpu_mem_var)

        # Draw namespace memory first (bottom layer)
        ax.fill_between(
            var_cells, 0, namespace_var, alpha=0.3, color="gray", label="User Namespace"
        )
        cumulative_var = namespace_var.copy()

        # Layer 2: GPU memory (if present) - between namespace and checkpoints
        if has_gpu_var:
            next_level = cumulative_var + gpu_mem_var
            ax.fill_between(
                var_cells,
                cumulative_var,
                next_level,
                alpha=0.4,
                color="orange",
                label="GPU Memory",
            )
            cumulative_var = next_level

        # Stack checkpoint variables on top
        stacked = [
            np.array(var_data["by_var"][v]) / mb for v in var_data["vars_ordered"]
        ]
        for i, (v, data_mb) in enumerate(zip(var_data["vars_ordered"], stacked)):
            var_type = mem_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(
                var_cells,
                cumulative_var,
                cumulative_var + data_mb,
                alpha=0.7,
                color=var_colors[i],
                label=label,
            )
            cumulative_var = cumulative_var + data_mb

        # Draw namespace line (for reference)
        ax.plot(var_cells, namespace_var, color="gray", linewidth=2, linestyle="--")

        # Add separator line for rerun phase
        var_initial_count = var_data.get("initial_count", len(var_cells))
        if var_initial_count < len(var_cells):
            ax.axvline(
                x=var_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )

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
    else:
        ax.text(
            0.5,
            0.5,
            "No checkpoint memory data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Checkpoint Memory by Variable", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # ========== Panel 5: Overhead Time per Cell (bottom-left) ==========
    ax = axes[4]
    if len(cells) > 0:
        # Overhead per cell = state + check + other (everything except code time)
        overhead_per_cell = state_arr + check_arr + other_arr
        bar_width = 0.6

        # Stacked bar chart with breakdown
        ax.bar(
            cells,
            state_arr / 1000,
            width=bar_width,
            alpha=0.7,
            color=colors[2],
            label="State",
        )
        ax.bar(
            cells,
            check_arr / 1000,
            width=bar_width,
            alpha=0.7,
            color=colors[3],
            label="Check",
            bottom=state_arr / 1000,
        )
        ax.bar(
            cells,
            other_arr / 1000,
            width=bar_width,
            alpha=0.7,
            color=colors[4],
            label="Other",
            bottom=(state_arr + check_arr) / 1000,
        )

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Overhead per Cell (seconds)", fontsize=label_size)
        title = "Overhead Time per Cell"
        if timing_initial_count < len(cells):
            title += f" (cells 1-{timing_initial_count} + {len(cells) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)

        if timing_initial_count < len(cells):
            ax.axvline(
                x=timing_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )

        ax.legend(loc="upper right", fontsize=legend_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5, right=len(cells) + 0.5)
        ax.set_ylim(bottom=0)
    else:
        ax.text(
            0.5, 0.5, "No timing data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Overhead Time per Cell", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # ========== Panel 6: Checkpoint Memory Overhead Ratio per Cell (bottom-right) ==========
    # Shows checkpoint_delta / (prev_namespace + prev_gpu) - the ratio of new checkpoint data to base memory
    # Uses simplified fields: checkpoint_delta_mb, namespace_mb, gpu_mb
    ax = axes[5]

    if flowbook_mem_cells:
        cell_nums = list(range(1, len(flowbook_mem_cells) + 1))
        min_meaningful_base_mb = 1.0

        # Pre-compute cumulative checkpoint totals for delta calculation
        # Use same fallback logic as Panel 3's get_checkpoint_mb()
        def get_cumulative_total(c):
            # 1. Try explicit field - but only if > 0 (may be 0 when saved checkpoints empty)
            checkpoint_cumulative = c.get("checkpoint_cumulative_mb")
            if checkpoint_cumulative is not None and checkpoint_cumulative > 0:
                return checkpoint_cumulative
            # 2. Fall back to summing cumulative_by_var (bytes -> MB)
            cumulative_by_var = c.get("cumulative_by_var") or {}
            if cumulative_by_var:
                return sum(cumulative_by_var.values()) / (1024 * 1024)
            # 3. Fall back to summing cumulative_by_type (bytes -> MB)
            cumulative_by_type = c.get("cumulative_by_type") or {}
            if cumulative_by_type:
                return sum(cumulative_by_type.values()) / (1024 * 1024)
            # 4. Fall back to summing checkpoint_var_costs (per-cell total)
            var_costs = c.get("checkpoint_var_costs") or {}
            if var_costs:
                return sum(v.get("bytes", 0) for v in var_costs.values()) / (
                    1024 * 1024
                )
            return 0

        cumulative_totals = [get_cumulative_total(c) for c in flowbook_mem_cells]

        ratios = []
        for i, c in enumerate(flowbook_mem_cells):
            # Get checkpoint delta (new field) or derive from cumulative
            # NOTE: Check > 0, not just "is not None", because field may be 0 when
            # saved checkpoints are empty but checkpoint_var_costs has data
            delta_mb = c.get("checkpoint_delta_mb")
            if not (delta_mb is not None and delta_mb > 0):
                overhead = c.get("overhead_breakdown") or {}
                delta_mb = overhead.get("checkpoints_mb")
            if not (delta_mb is not None and delta_mb > 0):
                # Derive from cumulative totals
                if i == 0:
                    delta_mb = cumulative_totals[0]
                else:
                    delta_mb = max(0, cumulative_totals[i] - cumulative_totals[i - 1])

            if i == 0:
                base_mb = 0  # No prior namespace for first cell
            else:
                prev_cell = flowbook_mem_cells[i - 1]
                # Support both old and new field names
                namespace = prev_cell.get(
                    "namespace_mb", prev_cell.get("current_footprint_mb", 0)
                )
                gpu = prev_cell.get("gpu_mb", prev_cell.get("gpu_mem_samples", 0))
                base_mb = namespace + gpu

            if base_mb >= min_meaningful_base_mb:
                ratio = delta_mb / base_mb
            else:
                ratio = 0.0  # Namespace too small for meaningful ratio
            ratios.append(ratio)

        bar_width = 0.6
        ax.bar(cell_nums, ratios, width=bar_width, alpha=0.7, color="#66c2a5")

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Checkpoint / Base Memory", fontsize=label_size)

        title = "Checkpoint Overhead Ratio"
        if memory_initial_count < len(flowbook_mem_cells):
            title += f" (cells 1-{memory_initial_count} + {len(flowbook_mem_cells) - memory_initial_count} reruns)"
            ax.axvline(
                x=memory_initial_count + 0.5,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Rerun Start",
            )
            ax.legend(loc="upper right", fontsize=legend_size)
        ax.set_title(title, fontsize=title_size)

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5, right=len(cell_nums) + 0.5)
        ax.set_ylim(bottom=0)
    else:
        ax.text(
            0.5, 0.5, "No memory data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Checkpoint Memory per Cell", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Add notebook name as figure title
    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    fig.suptitle(notebook_name, fontsize=title_size + 2, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Combined v2 plot saved to: {output_path}")
        return None
    else:
        return fig


def plot_overhead_histograms(
    aggregate: "AggregateStats",
    output_path: Optional[str] = None,
    large_fonts: bool = True,
) -> Optional[Any]:
    """
    Create histogram plots for per-cell overhead distributions.

    Creates two histograms:
    - Total overhead per cell (ms)
    - Memory overhead per cell (MB)

    Outliers are removed using the IQR method (1.5 * IQR).

    Args:
        aggregate: AggregateStats with per-cell data
        output_path: If provided, saves to file; otherwise returns figure
        large_fonts: Use larger fonts for paper-ready plots

    Returns:
        Figure if output_path is None, else None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # Font sizes
    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    tick_size = 14 if large_fonts else 10

    total_overhead = np.array(aggregate.all_total_overhead_per_cell)
    memory_overhead = np.array(aggregate.all_memory_overhead_per_cell)

    def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method (1.5 * IQR)."""
        if len(data) == 0:
            return data
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    # Remove outliers
    total_overhead_filtered = remove_outliers_iqr(total_overhead)
    memory_overhead_filtered = remove_outliers_iqr(memory_overhead)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram 1: Total Overhead per Cell
    ax = axes[0]
    if len(total_overhead_filtered) > 0:
        # Convert ms to seconds for better readability if values are large
        if np.max(total_overhead_filtered) > 1000:
            data_plot = total_overhead_filtered / 1000
            xlabel = "Total Overhead per Cell (seconds)"
        else:
            data_plot = total_overhead_filtered
            xlabel = "Total Overhead per Cell (ms)"

        ax.hist(data_plot, bins=30, alpha=0.3, color="steelblue", edgecolor="black")
        ax.axvline(
            np.median(data_plot),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(data_plot):.2f}",
        )
        ax.axvline(
            np.mean(data_plot),
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(data_plot):.2f}",
        )
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel("Frequency", fontsize=label_size)
        ax.set_title("Total Overhead per Cell Distribution", fontsize=title_size)
        ax.legend(fontsize=tick_size)

        # Add stats text box
        n_total = len(total_overhead)
        n_filtered = len(total_overhead_filtered)
        outliers_removed = n_total - n_filtered
        textstr = f"N={n_filtered} (removed {outliers_removed} outliers)"
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        ax.text(
            0.98,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=tick_size,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Total Overhead per Cell Distribution", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Histogram 2: Memory Overhead Ratio per Cell (checkpoint / user_ns)
    ax = axes[1]
    if len(memory_overhead_filtered) > 0:
        ax.hist(
            memory_overhead_filtered,
            bins=30,
            alpha=0.3,
            color="seagreen",
            edgecolor="black",
        )
        ax.axvline(
            np.median(memory_overhead_filtered),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(memory_overhead_filtered):.2f}",
        )
        ax.axvline(
            np.mean(memory_overhead_filtered),
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(memory_overhead_filtered):.2f}",
        )
        ax.set_xlabel("Checkpoint / User NS Ratio", fontsize=label_size)
        ax.set_ylabel("Frequency", fontsize=label_size)
        ax.set_title("Memory Overhead Ratio Distribution", fontsize=title_size)
        ax.legend(fontsize=tick_size)

        # Add stats text box
        n_total = len(memory_overhead)
        n_filtered = len(memory_overhead_filtered)
        outliers_removed = n_total - n_filtered
        textstr = f"N={n_filtered} (removed {outliers_removed} outliers)"
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        ax.text(
            0.98,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=tick_size,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Memory Overhead per Cell Distribution", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    fig.suptitle(
        "Per-Cell Overhead Distributions (Outliers Removed)",
        fontsize=title_size + 2,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Histogram plot saved to: {output_path}")
        return None
    else:
        return fig


def plot_overhead_cdfs(
    aggregate: "AggregateStats",
    output_path: Optional[str] = None,
    large_fonts: bool = True,
    show_sample_size: bool = True,
) -> Optional[List[Any]]:
    """
    Create CDF plots for per-cell overhead distributions.

    Creates two pages:
    1. Log scale CDFs showing full distribution with percentile markers
    2. Linear scale CDFs zoomed to P99

    Args:
        aggregate: AggregateStats with per-cell data
        output_path: If provided, saves to file; otherwise returns list of figures
        large_fonts: Use larger fonts for paper-ready plots

    Returns:
        List of figures if output_path is None, else None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use minimal style - we'll add our own subtle grid
    sns.set_theme(style="white")

    # Font sizes - larger for readability, legend stays compact
    label_size = 22 if large_fonts else 14
    title_size = 24 if large_fonts else 16
    tick_size = 18 if large_fonts else 12
    annotation_size = 18 if large_fonts else 12
    legend_fontsize = 14 if large_fonts else 10  # Keep legend compact

    total_overhead = np.array(aggregate.all_total_overhead_per_cell)
    memory_overhead = np.array(aggregate.all_memory_overhead_per_cell)

    # Prepare data - keep in ms
    if len(total_overhead) > 0:
        total_data = np.sort(total_overhead)  # Keep in ms
        total_cdf = np.arange(1, len(total_data) + 1) / len(total_data)
        total_stats = {
            "P50": np.percentile(total_data, 50),
            "P90": np.percentile(total_data, 90),
            "P95": np.percentile(total_data, 95),
            "P99": np.percentile(total_data, 99),
        }
    else:
        total_data = None

    if len(memory_overhead) > 0:
        # memory_overhead is now a ratio (checkpoint / user_ns), not MB
        memory_data_ratio = np.sort(memory_overhead)
        memory_cdf = np.arange(1, len(memory_data_ratio) + 1) / len(memory_data_ratio)
        memory_stats_ratio = {
            "P50": np.percentile(memory_data_ratio, 50),
            "P90": np.percentile(memory_data_ratio, 90),
            "P95": np.percentile(memory_data_ratio, 95),
            "P99": np.percentile(memory_data_ratio, 99),
        }
    else:
        memory_data_ratio = None

    def add_percentile_markers(ax, data, cdf, stats, unit_fmt, color, lfontsize):
        """Add percentile markers with vertical lines and labels, plus legend."""
        percentiles = ["P50", "P90", "P95", "P99"]
        y_positions = [0.5, 0.9, 0.95, 0.99]
        # Larger offsets to avoid curve overlap; use leader lines
        label_offsets = [(12, 8), (12, -18), (12, 8), (12, -18)]  # (x, y) offsets
        label_vas = ["bottom", "top", "bottom", "top"]  # vertical alignments

        for pname, y_val, offset, va in zip(
            percentiles, y_positions, label_offsets, label_vas
        ):
            if pname not in stats:
                continue
            x_val = stats[pname]
            # Small point on the curve
            ax.scatter(
                [x_val],
                [y_val],
                color=color,
                s=40,
                marker="o",
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
            )
            # Label with leader line (arrow)
            ax.annotate(
                pname,
                (x_val, y_val),
                textcoords="offset points",
                xytext=offset,
                fontsize=annotation_size,
                ha="left",
                va=va,
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.8,
                    shrinkA=0,
                    shrinkB=3,
                ),
            )

        # Add max point on the curve (at CDF = 1.0) with label
        if len(data) > 0:
            max_val = np.max(data)
            ax.scatter(
                [max_val],
                [1.0],
                color=color,
                s=40,
                marker="o",
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
            )
            ax.annotate(
                "Max",
                (max_val, 1.0),
                textcoords="offset points",
                xytext=(12, -18),
                fontsize=annotation_size,
                ha="left",
                va="top",
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.8,
                    shrinkA=0,
                    shrinkB=3,
                ),
            )

        # Add legend box with values in lower right (right-aligned values)
        formatted_values = {
            pname: unit_fmt(stats[pname]) for pname in percentiles if pname in stats
        }
        if len(data) > 0:
            formatted_values["Max"] = unit_fmt(np.max(data))
        max_val_len = (
            max(len(v) for v in formatted_values.values()) if formatted_values else 0
        )
        # Pad keys to align columns (P50, P90, P95, P99, Max -> all 3 chars)
        legend_keys = [p for p in percentiles if p in formatted_values] + (
            ["Max"] if "Max" in formatted_values else []
        )
        legend_lines = [
            f"{pname:>3}  {formatted_values[pname]:>{max_val_len}}"
            for pname in legend_keys
        ]
        legend_text = "\n".join(legend_lines)
        props = dict(
            boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"
        )
        ax.text(
            0.98,
            0.02,
            legend_text,
            transform=ax.transAxes,
            fontsize=lfontsize,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
            family="monospace",  # Monospace for column alignment
        )

    def add_subtle_grid(ax, stats=None):
        """Add subtle grid with fewer gridlines."""
        # Horizontal lines at key percentile levels
        for y in [0.5, 0.9, 0.99]:
            ax.axhline(y, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        # Vertical lines at percentile x-values if provided
        if stats:
            for pname in ["P50", "P90", "P95", "P99"]:
                if pname in stats:
                    ax.axvline(
                        stats[pname],
                        color="gray",
                        linestyle="--",
                        linewidth=0.6,
                        alpha=0.3,
                    )
        # Subtle spines
        for spine in ax.spines.values():
            spine.set_color("lightgray")
            spine.set_linewidth(0.5)

    figures = []

    # --- Figure 1: Log scale with full distribution ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # Total overhead - log scale
    ax = axes1[0]
    if total_data is not None:
        pos_mask = total_data > 0
        if np.any(pos_mask):
            ax.fill_between(
                total_data[pos_mask],
                0,
                total_cdf[pos_mask],
                alpha=0.12,  # Reduced opacity
                color="steelblue",
                edgecolor="none",
            )
            ax.plot(
                total_data[pos_mask],
                total_cdf[pos_mask],
                color="steelblue",
                linewidth=2.5,  # Slightly thicker line
            )

            add_percentile_markers(
                ax,
                total_data[pos_mask],
                total_cdf[pos_mask],
                total_stats,
                lambda x: f"{x:.1f}ms",
                "steelblue",
                legend_fontsize,
            )
            add_subtle_grid(ax, total_stats)

            ax.set_xscale("log")
            ax.set_xlabel(
                "Checkpoint + Analysis Time (ms, log scale)", fontsize=label_size
            )
            ax.set_ylabel("Cumulative Proportion", fontsize=label_size)
            ax.set_title(
                "Time Overhead Distribution",
                fontsize=title_size,
            )
            ax.set_ylim(0, 1.05)

            # Use plain formatting (no scientific notation) for x-axis
            from matplotlib.ticker import ScalarFormatter

            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)

            if show_sample_size:
                textstr = f"N={len(total_data):,}"
                props = dict(
                    boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"
                )
                ax.text(
                    0.02,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=tick_size,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=props,
                )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Time Overhead Distribution", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Memory overhead ratio (checkpoint / user_ns)
    ax = axes1[1]
    if memory_data_ratio is not None:
        pos_mask = memory_data_ratio > 0
        if np.any(pos_mask):
            ax.fill_between(
                memory_data_ratio[pos_mask],
                0,
                memory_cdf[pos_mask],
                alpha=0.12,  # Reduced opacity
                color="seagreen",
                edgecolor="none",
            )
            ax.plot(
                memory_data_ratio[pos_mask],
                memory_cdf[pos_mask],
                color="seagreen",
                linewidth=2.5,  # Slightly thicker line
            )

            # Format ratio as percentage for legend with adaptive precision
            def format_ratio_pct(r):
                if r >= 1:
                    return f"{r * 100:.0f}%"
                elif r >= 0.1:
                    return f"{r * 100:.0f}%"
                elif r >= 0.01:
                    return f"{r * 100:.1f}%"
                elif r >= 0.001:
                    return f"{r * 100:.2f}%"
                elif r >= 0.0001:
                    return f"{r * 100:.3f}%"
                elif r > 0:
                    return "<0.01%"
                else:
                    return "0%"

            add_percentile_markers(
                ax,
                memory_data_ratio[pos_mask],
                memory_cdf[pos_mask],
                memory_stats_ratio,
                format_ratio_pct,
                "seagreen",
                legend_fontsize,
            )
            add_subtle_grid(ax, memory_stats_ratio)

            ax.set_xlabel(
                "Checkpoint / Namespace Size",
                fontsize=label_size,
            )
            ax.set_ylabel("Cumulative Proportion", fontsize=label_size)
            ax.set_title(
                "Memory Overhead Distribution",
                fontsize=title_size,
            )
            ax.set_ylim(0, 1.05)

            # Log scale x-axis with percentage tick labels
            # ax.set_xscale("log")
            ax.set_xlim(-0.05, 1.05)  # Add whitespace on left and right
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

            if show_sample_size:
                textstr = f"N={len(memory_data_ratio):,}"
                props = dict(
                    boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"
                )
                ax.text(
                    0.02,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=tick_size,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=props,
                )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Memory Overhead Distribution", fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    plt.tight_layout()
    figures.append(fig1)

    # --- Figure 2: Peak Memory Overhead % CDF (per notebook) ---
    peak_pcts = np.array(aggregate.all_peak_memory_overhead_pct)
    if len(peak_pcts) > 0:
        peak_data = np.sort(peak_pcts)
        peak_cdf = np.arange(1, len(peak_data) + 1) / len(peak_data)
        peak_stats = {
            "P50": np.percentile(peak_data, 50),
            "P90": np.percentile(peak_data, 90),
            "P95": np.percentile(peak_data, 95),
            "P99": np.percentile(peak_data, 99),
        }

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

        ax2.fill_between(
            peak_data,
            0,
            peak_cdf,
            alpha=0.12,
            color="darkorange",
            edgecolor="none",
        )
        ax2.plot(
            peak_data,
            peak_cdf,
            color="darkorange",
            linewidth=2.5,
        )

        # Format percentage values for legend
        def format_pct(v):
            if v >= 100:
                return f"{v:.0f}%"
            elif v >= 10:
                return f"{v:.1f}%"
            elif v >= 1:
                return f"{v:.2f}%"
            else:
                return f"{v:.3f}%"

        add_percentile_markers(
            ax2,
            peak_data,
            peak_cdf,
            peak_stats,
            format_pct,
            "darkorange",
            legend_fontsize,
        )
        add_subtle_grid(ax2, peak_stats)

        ax2.set_xlabel("Peak Memory Overhead (%)", fontsize=label_size)
        ax2.set_ylabel("Cumulative Proportion", fontsize=label_size)
        ax2.set_title("Peak Memory Overhead Distribution", fontsize=title_size)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, 100)
        ax2.set_xticks([0, 25, 50, 75, 100])
        ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

        if show_sample_size:
            textstr = f"N={len(peak_data):,}"
            props = dict(
                boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"
            )
            ax2.text(
                0.02,
                0.98,
                textstr,
                transform=ax2.transAxes,
                fontsize=tick_size,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=props,
            )

        ax2.tick_params(axis="both", labelsize=tick_size)
        # Subtle spines
        for spine in ax2.spines.values():
            spine.set_color("lightgray")
            spine.set_linewidth(0.5)

        plt.tight_layout()
        figures.append(fig2)

    if output_path is not None:
        # Save first figure only (for standalone file output)
        plt.figure(fig1.number)
        plt.savefig(output_path, dpi=150)
        for f in figures:
            plt.close(f)
        print(f"CDF plot saved to: {output_path}")
        return None
    else:
        return figures


def process_v4(
    file_data: Dict[str, Dict[str, Any]],
    args,
    excluded_for_errors: Optional[List[Tuple[str, List[str]]]] = None,
) -> None:
    """Process v4.0 format files using new dataclass models and extraction functions.

    Args:
        file_data: Dict mapping file paths to loaded JSON dicts
        args: Parsed command line arguments
        excluded_for_errors: List of (path, cell_ids) for notebooks excluded due to baseline errors
    """
    if excluded_for_errors is None:
        excluded_for_errors = []

    # Compute show_sample_size from args
    show_sample_size = not getattr(args, "no_sample_size", False)

    # Convert to ComparisonResultV4 objects
    results: Dict[str, ComparisonResultV4] = {}
    raw_data: Dict[str, Dict] = {}  # Keep raw data for v5 detection
    for path, data in file_data.items():
        results[path] = ComparisonResultV4.from_dict(data)
        raw_data[path] = data

    # Only do plot work if --plot is specified
    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # Generate plots
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_figures = []
        peak_overhead_stats = (
            []
        )  # Collect (notebook_name, peak_flowbook_mb, peak_base_mb, peak_pct) for table

        for file_path, result in results.items():
            notebook_name = Path(file_path).stem.replace("_comparison", "")

            # Detect v5 format and use appropriate extraction
            data = raw_data[file_path]
            version = data.get("version", "4.0")

            # Extract plot data
            # Plot 1 and 5 are timing-only, version-independent
            p1 = extract_plot1_data(result)
            p5 = extract_plot5_data(result)

            # Memory plots: use v5 extraction if available
            # Extract baseline cells for cross-run comparison (Plot 3)
            baseline_cells = extract_baseline_cells(data)

            if version.startswith("5"):
                # Native v5 format
                v5_memory = extract_v5_memory_result(data)
                if v5_memory and v5_memory.all_cells:
                    # Plot 2: try v5 extraction (uses checkpoint_var_timing)
                    p2 = extract_plot2_data_v5(v5_memory.all_cells, top_n=args.top_n)
                    if p2 is None:
                        # Fall back to v4 extraction from timing data
                        p2 = extract_plot2_data(result, top_n=args.top_n)
                    gpu_from_timing = extract_gpu_overhead_from_timing(data)
                    p3 = extract_plot3_data_v5(
                        v5_memory.all_cells,
                        baseline_cells,
                        gpu_overhead_from_timing=gpu_from_timing,
                    )
                    p4 = extract_plot4_data_v5(v5_memory.all_cells, top_n=args.top_n)
                    p6 = extract_plot6_data_v5(v5_memory.all_cells)
                else:
                    # Fall back to v4 extraction
                    p2 = extract_plot2_data(result, top_n=args.top_n)
                    p3 = extract_plot3_data(result)
                    p4 = extract_plot4_data(result, top_n=args.top_n)
                    p6 = extract_plot6_data(result)
            else:
                # v4 format - try v5 extraction first (handles conversion)
                v5_memory = extract_v5_memory_result(data)
                if v5_memory and v5_memory.all_cells:
                    p2 = extract_plot2_data_v5(v5_memory.all_cells, top_n=args.top_n)
                    if p2 is None:
                        p2 = extract_plot2_data(result, top_n=args.top_n)
                    gpu_from_timing = extract_gpu_overhead_from_timing(data)
                    p3 = extract_plot3_data_v5(
                        v5_memory.all_cells,
                        baseline_cells,
                        gpu_overhead_from_timing=gpu_from_timing,
                    )
                    p4 = extract_plot4_data_v5(v5_memory.all_cells, top_n=args.top_n)
                    p6 = extract_plot6_data_v5(v5_memory.all_cells)
                else:
                    # Pure v4 extraction
                    p2 = extract_plot2_data(result, top_n=args.top_n)
                    p3 = extract_plot3_data(result)
                    p4 = extract_plot4_data(result, top_n=args.top_n)
                    p6 = extract_plot6_data(result)

            # Collect peak overhead stats for table
            if p3 is not None and p3.peak_base_mb > 0:
                peak_overhead_stats.append(
                    (
                        notebook_name,
                        p3.peak_flowbook_mb,
                        p3.peak_base_mb,
                        p3.peak_overhead_pct,
                    )
                )

            # Create 6-panel figure
            fig, axes_2d = plt.subplots(3, 2, figsize=(14, 18))
            axes = [
                axes_2d[0, 0],
                axes_2d[0, 1],
                axes_2d[1, 0],
                axes_2d[1, 1],
                axes_2d[2, 0],
                axes_2d[2, 1],
            ]

            try:
                render_combined_6panel(
                    fig,
                    axes,
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    large_fonts=args.large_fonts,
                    notebook_name=notebook_name,
                )
                combined_figures.append(fig)
            except Exception as e:
                print(
                    f"Warning: Could not generate plot for {file_path}: {e}",
                    file=sys.stderr,
                )
                plt.close(fig)

        # Print peak memory overhead table
        if peak_overhead_stats:
            print("\nPeak Memory Overhead by Notebook:")
            print("-" * 80)
            print(f"{'Notebook':<40} {'FlowBook':>12} {'Base':>12} {'Overhead':>10}")
            print(f"{'':<40} {'(MB)':>12} {'(MB)':>12} {'(%)':>10}")
            print("-" * 80)
            for name, fb_mb, base_mb, pct in sorted(
                peak_overhead_stats, key=lambda x: -x[3]
            ):
                # Truncate long names with ellipsis
                display_name = name[:37] + "..." if len(name) > 40 else name
                print(
                    f"{display_name:<40} {fb_mb:>12.1f} {base_mb:>12.1f} {pct:>10.1f}"
                )
            print("-" * 80)
            # Summary stats
            pcts = [x[3] for x in peak_overhead_stats]
            print(
                f"P50: {np.percentile(pcts, 50):.1f}%   P90: {np.percentile(pcts, 90):.1f}%"
            )
            print()

        # Save to PDF
        if combined_figures:
            combined_path = output_dir / args.output
            with PdfPages(str(combined_path)) as pdf:
                for fig in combined_figures:
                    pdf.savefig(fig, dpi=150)
                    plt.close(fig)

                # Add CDF plot if we have multiple notebooks
                if len(results) > 1:
                    # Pass raw data for v5 extraction
                    results_list = list(results.values())
                    raw_data_list = [raw_data[p] for p in results.keys()]
                    cdf_data = extract_cdf_data(results_list, raw_data_list)
                    if cdf_data:
                        from flowbook.cli.plot_rendering import render_cdf_panel
                        import seaborn as sns

                        sns.set_theme(style="whitegrid")

                        cdf_fig, cdf_axes = plt.subplots(1, 3, figsize=(15, 5))
                        render_cdf_panel(
                            cdf_axes[0], cdf_data, "time", large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size
                        )
                        render_cdf_panel(
                            cdf_axes[1],
                            cdf_data,
                            "memory_abs",
                            large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size,
                        )
                        render_cdf_panel(
                            cdf_axes[2], cdf_data, "peak", large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size
                        )
                        cdf_fig.tight_layout()
                        pdf.savefig(cdf_fig, dpi=150)
                        plt.close(cdf_fig)

                        # GPU checkpoint CDF page (if any GPU data exists)
                        if cdf_data.gpu_memory_ratios or cdf_data.gpu_peak_memory_pct:
                            gpu_cdf_fig, gpu_cdf_axes = plt.subplots(
                                1, 2, figsize=(12, 5)
                            )
                            render_cdf_panel(
                                gpu_cdf_axes[0],
                                cdf_data,
                                "gpu_memory",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                            )
                            render_cdf_panel(
                                gpu_cdf_axes[1],
                                cdf_data,
                                "gpu_peak",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                            )
                            gpu_cdf_fig.tight_layout()
                            pdf.savefig(gpu_cdf_fig, dpi=150)
                            plt.close(gpu_cdf_fig)

                        # Per-cell overhead percentage CDF page
                        if cdf_data.overhead_pct:
                            overhead_fig, overhead_ax = plt.subplots(figsize=(8, 6))
                            render_overhead_pct_cdf(overhead_ax, cdf_data, large_fonts=args.large_fonts, show_sample_size=show_sample_size)
                            overhead_fig.tight_layout()
                            pdf.savefig(overhead_fig, dpi=150)
                            plt.close(overhead_fig)

                        # Base runtime CDF page
                        if cdf_data.base_runtime_ms:
                            base_fig, base_ax = plt.subplots(figsize=(8, 6))
                            render_base_runtime_cdf(base_ax, cdf_data, large_fonts=args.large_fonts, show_sample_size=show_sample_size)
                            base_fig.tight_layout()
                            pdf.savefig(base_fig, dpi=150)
                            plt.close(base_fig)

                    # Rerun overhead plots (if any data exists)
                    rerun_cdf_data = extract_rerun_overhead_data(raw_data_list)
                    if rerun_cdf_data:
                        import seaborn as sns

                        sns.set_theme(style="whitegrid")

                        # Rerun checkpoint breakdown page for each notebook
                        for path, data in raw_data.items():
                            rerun = data.get("rerun_overhead")
                            if rerun and rerun.get("measurements"):
                                notebook_name = Path(path).stem.replace(
                                    "_comparison", ""
                                )
                                # Extract per-notebook rerun data with breakdown
                                measurements = rerun.get("measurements", [])

                                # Build breakdown dict: breakdown[cell_index][iteration] = (ckpt, check)
                                breakdown: Dict[int, Dict[int, tuple]] = defaultdict(
                                    dict
                                )
                                cell_indices_set = set()
                                iterations_set = set()

                                for m in measurements:
                                    cell_idx = m.get("cell_index", 0)
                                    iteration = m.get("iteration", 0)
                                    ckpt = m.get("checkpoint_ms", 0)
                                    chk = m.get("check_ms", 0)

                                    breakdown[cell_idx][iteration] = (ckpt, chk)
                                    cell_indices_set.add(cell_idx)
                                    iterations_set.add(iteration)

                                # Sort cell indices and determine iteration count
                                cell_indices = sorted(cell_indices_set)
                                num_iterations = (
                                    len(iterations_set) if iterations_set else 0
                                )

                                notebook_rerun = RerunOverheadCDFData(
                                    total_overhead_ms=[
                                        m.get("total_overhead_ms", 0)
                                        for m in measurements
                                    ],
                                    total_sorted=sorted(
                                        [
                                            m.get("total_overhead_ms", 0)
                                            for m in measurements
                                        ]
                                    ),
                                    total_percentiles=[],  # Not needed for breakdown
                                    checkpoint_ms=[
                                        m.get("checkpoint_ms", 0) for m in measurements
                                    ],
                                    check_ms=[
                                        m.get("check_ms", 0) for m in measurements
                                    ],
                                    cell_indices=cell_indices,
                                    num_iterations=num_iterations,
                                    breakdown=dict(breakdown),
                                )

                                # Use wider figure for grouped bars
                                fig_width = max(8, len(cell_indices) * 2)
                                breakdown_fig, breakdown_ax = plt.subplots(
                                    figsize=(fig_width, 6)
                                )
                                render_rerun_checkpoint_breakdown(
                                    breakdown_ax,
                                    notebook_rerun,
                                    notebook_name=notebook_name,
                                    large_fonts=args.large_fonts,
                                )
                                breakdown_fig.tight_layout()
                                pdf.savefig(breakdown_fig, dpi=150)
                                plt.close(breakdown_fig)

                        # Rerun overhead CDF page (combined across all notebooks)
                        rerun_cdf_fig, rerun_cdf_ax = plt.subplots(figsize=(8, 6))
                        render_rerun_overhead_cdf(
                            rerun_cdf_ax, rerun_cdf_data, large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size
                        )
                        rerun_cdf_fig.tight_layout()
                        pdf.savefig(rerun_cdf_fig, dpi=150)
                        plt.close(rerun_cdf_fig)

                        # Time CDFs side-by-side page (analysis + rerun)
                        if cdf_data:
                            time_cdf_fig, time_cdf_axes = plt.subplots(
                                1, 2, figsize=(12, 5), sharey=True
                            )
                            render_time_cdf(
                                time_cdf_axes[0],
                                sorted_vals=list(cdf_data.time_sorted),
                                percentiles=list(cdf_data.time_percentiles),
                                n=len(cdf_data.time_overhead_ms),
                                color="blue",
                                title="Per-Cell Analysis Time \nDistribution",
                                xlabel="Analysis Time (ms, log scale)",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                            )
                            render_time_cdf(
                                time_cdf_axes[1],
                                sorted_vals=list(rerun_cdf_data.total_sorted),
                                percentiles=list(rerun_cdf_data.total_percentiles),
                                n=len(rerun_cdf_data.total_overhead_ms),
                                color="green",
                                title="Per-Rerun-Cell Analysis Time \nDistribution",
                                xlabel="Rerun Overhead (ms, log scale)",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                            )
                            time_cdf_fig.tight_layout()
                            pdf.savefig(time_cdf_fig, dpi=150)
                            time_cdf_fig.savefig("time.pdf", dpi=150)
                            plt.close(time_cdf_fig)

                            # All CDFs in one row: 2 time + 1 memory, slightly
                            # larger fonts than the standalone time/mem.pdf files
                            all_cdf_scale = 1.3
                            all_cdf_fig, all_cdf_axes = plt.subplots(
                                1, 3, figsize=(18, 5), sharey=True
                            )
                            render_time_cdf(
                                all_cdf_axes[0],
                                sorted_vals=list(cdf_data.time_sorted),
                                percentiles=list(cdf_data.time_percentiles),
                                n=len(cdf_data.time_overhead_ms),
                                color="blue",
                                title="Per-Cell Analysis Time",
                                xlabel="Analysis Time (ms, log scale)",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                                font_scale=all_cdf_scale,
                            )
                            render_time_cdf(
                                all_cdf_axes[1],
                                sorted_vals=list(rerun_cdf_data.total_sorted),
                                percentiles=list(rerun_cdf_data.total_percentiles),
                                n=len(rerun_cdf_data.total_overhead_ms),
                                color="green",
                                title="Per-Rerun-Cell Analysis Time",
                                xlabel="Rerun Overhead (ms, log scale)",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                                font_scale=all_cdf_scale,
                            )
                            render_cdf_panel(
                                all_cdf_axes[2],
                                cdf_data,
                                "memory_abs",
                                color_override="orange",
                                title_override="Per-Cell Checkpoint Memory Size",
                                large_fonts=args.large_fonts,
                                show_sample_size=show_sample_size,
                                font_scale=all_cdf_scale,
                            )
                            # Drop the redundant y-axis label on the 2nd and
                            # 3rd panels (y-axis is shared via sharey=True).
                            all_cdf_axes[1].set_ylabel("")
                            all_cdf_axes[2].set_ylabel("")
                            all_cdf_fig.tight_layout()
                            pdf.savefig(all_cdf_fig, dpi=150)
                            all_cdf_fig.savefig("all-cdfs.pdf", dpi=150)
                            plt.close(all_cdf_fig)

                    # Memory CDFs side-by-side page (per-cell + per-notebook)
                    if cdf_data:
                        mem_cdf_fig, mem_cdf_axes = plt.subplots(
                            1, 2, figsize=(12, 5), sharey=True
                        )
                        render_cdf_panel(
                            mem_cdf_axes[0],
                            cdf_data,
                            "memory_abs",
                            color_override="orange",
                            title_override="Per-Cell Checkpoint Memory Size\nDistribution",
                            large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size,
                        )
                        render_cdf_panel(
                            mem_cdf_axes[1],
                            cdf_data,
                            "peak",
                            color_override="red",
                            title_override="Per-Notebook Peak Memory Overhead\nDistribution",
                            large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size,
                        )
                        mem_cdf_fig.tight_layout()
                        pdf.savefig(mem_cdf_fig, dpi=150)
                        mem_cdf_fig.savefig("mem.pdf", dpi=150)
                        plt.close(mem_cdf_fig)

                    # Absolute per-cell memory overhead page (Checkpoint - Base, MB)
                    if cdf_data and cdf_data.memory_abs_mb:
                        mem_abs_fig, mem_abs_ax = plt.subplots(figsize=(6, 5))
                        render_cdf_panel(
                            mem_abs_ax,
                            cdf_data,
                            "memory_abs",
                            color_override="orange",
                            title_override="Per-Cell Checkpoint Memory Size\nDistribution",
                            large_fonts=args.large_fonts,
                            show_sample_size=show_sample_size,
                        )
                        mem_abs_ax.set_xlim(0.01, 10000)
                        mem_abs_ax.set_xticks([0.01, 1, 100, 10000])
                        mem_abs_ax.set_xticklabels(
                            ["0.01", "1", "100", "10000"]
                        )
                        mem_abs_fig.tight_layout()
                        pdf.savefig(mem_abs_fig, dpi=150)
                        mem_abs_fig.savefig("mem-abs.pdf", dpi=150)
                        plt.close(mem_abs_fig)

            print(f"Combined plots saved to: {combined_path}")

    # Print summary table
    print_v5_summary(raw_data, results)

    # Print summary counts at the very end
    print()
    print(f"PROCESSED {len(results)} NOTEBOOK(S) WITHOUT ERRORS")
    print(f"EXCLUDED {len(excluded_for_errors)} NOTEBOOK(S) WITH BASELINE ERRORS:")
    print("-" * 60)
    if excluded_for_errors:
        for path, cell_ids in sorted(excluded_for_errors):
            notebook_name = Path(path).stem.replace("_comparison", "")
            print(
                f"  {notebook_name}: {len(cell_ids)} error(s) in cells {', '.join(cell_ids)}"
            )
    else:
        print("  (none)")

    # Print source directories
    source_dirs = sorted(set(str(Path(p).parent) for p in raw_data.keys()))
    print()
    print("SOURCE DIRECTORIES:")
    print("-" * 60)
    for d in source_dirs:
        print(f"  {d}")
    # Also show cache directory if it exists
    if os.path.exists(CACHE_BASE_DIR) and os.listdir(CACHE_BASE_DIR):
        print()
        print(f"CACHE DIRECTORY: {CACHE_BASE_DIR}")


def print_v5_summary(raw_data: Dict[str, Dict], results: Dict[str, Any]) -> None:
    """Print summary table for v5 format data.

    Args:
        raw_data: Dict mapping file paths to raw JSON dicts
        results: Dict mapping file paths to ComparisonResultV4 objects
    """
    from flowbook.cli.plot_extraction import extract_v5_memory_result

    print()
    print("=" * 120)
    print("FLOWBOOK OVERHEAD SUMMARY")
    print("=" * 120)
    print(f"Notebooks: {len(raw_data)}")
    print("=" * 120)
    print()

    # Per-notebook header
    header = (
        f"{'Notebook':<35} {'Cells':>5} "
        f"{'User NS':>10} {'GPU':>8} {'Ckpt':>10} {'Ckpt/Base':>10} "
        f"{'State ms':>10} {'Check ms':>10}"
    )
    print(header)
    print("-" * 120)

    # Collect aggregate data
    all_overhead_ms = []
    all_memory_ratios = []
    all_peak_pcts = []
    all_overhead_pct = []  # Per-cell overhead: (state + check) / (base + 150) * 100
    all_base_runtime_ms = []  # Per-cell base runtime (code execution time)

    # Staleness data per notebook
    staleness_data = []  # (name, clean, stale, error, reason_counts, error_counts)
    total_clean = 0
    total_stale = 0
    total_error = 0
    total_reason_counts: Dict[str, int] = {}
    total_error_counts: Dict[str, int] = {}

    for path, data in raw_data.items():
        notebook_name = Path(path).stem.replace("_comparison", "")
        if len(notebook_name) > 33:
            notebook_name = notebook_name[:30] + "..."

        # Get v5 memory data
        v5_memory = extract_v5_memory_result(data)
        if not v5_memory or not v5_memory.all_cells:
            continue

        cells = v5_memory.all_cells
        num_cells = len(cells)

        # Final cell values
        final = cells[-1]
        user_ns_mb = final.user_ns_mb
        gpu_mb = final.gpu_mb
        checkpoint_mb = final.checkpoint_mb
        base_mb = user_ns_mb + gpu_mb

        # Peak checkpoint ratio
        peak_ratio = 0.0
        for c in cells:
            c_base = c.user_ns_mb + c.gpu_mb
            if c_base > 0.1:
                ratio = c.checkpoint_mb / c_base
                peak_ratio = max(peak_ratio, ratio)

        all_peak_pcts.append(peak_ratio * 100)

        # Per-cell overhead data
        for i, c in enumerate(cells):
            c_base = c.user_ns_mb + c.gpu_mb
            if i > 0 and c_base > 0.1:
                prev = cells[i - 1]
                delta = max(0, c.checkpoint_mb - prev.checkpoint_mb)
                prev_base = prev.user_ns_mb + prev.gpu_mb
                if prev_base > 0.1:
                    all_memory_ratios.append(delta / prev_base)

        # Get timing data
        result = results.get(path)
        timing = result.timing if result else None
        state_ms = 0.0
        check_ms = 0.0
        clean_cells = 0
        stale_cells = 0
        error_cells = 0
        reason_counts: Dict[str, int] = {}
        error_counts: Dict[str, int] = {}
        if timing:
            fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
            for cell in fb_timing.get("cells", []):
                s_ms = cell.get("state_ms", 0) or cell.get("state_duration_ms", 0) or 0
                c_ms = cell.get("check_ms", 0) or cell.get("check_duration_ms", 0) or 0
                state_ms += s_ms
                check_ms += c_ms
                all_overhead_ms.append(s_ms + c_ms)

                # Collect per-cell overhead percentage: (state + check) / (base + 150) * 100
                # Add 150ms to base to reflect Jupyter frontend overhead
                code_ms = cell.get("code_duration_ms")
                if code_ms is None:
                    code_ms = cell.get("run_ms", 0) or cell.get("cell_runtime_ms", 0) or 0
                if code_ms > 0:
                    overhead_pct = (s_ms + c_ms) / (code_ms + 150) * 100
                    all_overhead_pct.append(overhead_pct)
                    all_base_runtime_ms.append(code_ms)

            # Get checking summary (staleness data)
            totals = fb_timing.get("totals", {})
            checking = totals.get("checking_summary", {})
            clean_cells = checking.get("clean_cells", 0)
            stale_cells = checking.get("stale_cells", 0)
            error_cells = checking.get("error_cells", 0)
            reason_counts = checking.get("reason_counts", {})
            error_counts = checking.get("error_counts", {})

            # Accumulate totals
            total_clean += clean_cells
            total_stale += stale_cells
            total_error += error_cells
            for r, c in reason_counts.items():
                total_reason_counts[r] = total_reason_counts.get(r, 0) + c
            for e, c in error_counts.items():
                total_error_counts[e] = total_error_counts.get(e, 0) + c

        staleness_data.append(
            (
                notebook_name,
                clean_cells,
                stale_cells,
                error_cells,
                reason_counts,
                error_counts,
            )
        )

        # Format ratio
        ratio_str = f"{peak_ratio * 100:.1f}%" if base_mb > 0.1 else "N/A"

        row = (
            f"{notebook_name:<35} {num_cells:>5} "
            f"{user_ns_mb:>9.1f}M {gpu_mb:>7.1f}M {checkpoint_mb:>9.1f}M {ratio_str:>10} "
            f"{state_ms:>9.0f}ms {check_ms:>9.0f}ms"
        )
        print(row)

    print("-" * 120)
    print()

    # Aggregate statistics
    print("AGGREGATE STATISTICS")
    print("-" * 60)

    if all_overhead_ms:
        arr = np.array(all_overhead_ms)
        print("Per-Cell Time Overhead (state + check):")
        print(f"  P50: {np.percentile(arr, 50):.1f}ms")
        print(f"  P95: {np.percentile(arr, 95):.1f}ms")
        print(f"  P99: {np.percentile(arr, 99):.1f}ms")
        print(f"  Max: {np.max(arr):.1f}ms")
        print()

    if all_overhead_pct:
        arr = np.array(all_overhead_pct)
        print("Per-Cell Overhead ((state + check) / (base + 150ms)):")
        print(f"  P50: {np.percentile(arr, 50):.2f}%")
        print(f"  P95: {np.percentile(arr, 95):.2f}%")
        print(f"  P99: {np.percentile(arr, 99):.2f}%")
        print(f"  Max: {np.max(arr):.2f}%")
        print()

    if all_base_runtime_ms:
        arr = np.array(all_base_runtime_ms)
        print("Per-Cell Base Runtime (code execution):")
        print(f"  P50: {np.percentile(arr, 50):.1f}ms")
        print(f"  P95: {np.percentile(arr, 95):.1f}ms")
        print(f"  P99: {np.percentile(arr, 99):.1f}ms")
        print(f"  Max: {np.max(arr):.1f}ms")
        print()

    if all_memory_ratios:
        arr = np.array(all_memory_ratios)
        print("Per-Cell Memory Overhead (checkpoint_delta / prev_base):")
        print(f"  P50: {np.percentile(arr, 50) * 100:.1f}%")
        print(f"  P95: {np.percentile(arr, 95) * 100:.1f}%")
        print(f"  P99: {np.percentile(arr, 99) * 100:.1f}%")
        print(f"  Max: {np.max(arr) * 100:.1f}%")
        print()

    if all_peak_pcts:
        arr = np.array(all_peak_pcts)
        print("Peak Memory Overhead (per notebook):")
        print(f"  P50: {np.percentile(arr, 50):.1f}%")
        print(f"  P95: {np.percentile(arr, 95):.1f}%")
        print(f"  P99: {np.percentile(arr, 99):.1f}%")
        print(f"  Max: {np.max(arr):.1f}%")
        print()

    # Staleness and error statistics (always print, even if no cells checked)
    total_checked = total_clean + total_stale + total_error
    print("CHECKING RESULTS")
    print("-" * 60)
    print(f"  Clean cells:        {total_clean}")
    print(f"  Stale cells:        {total_stale}")
    print(f"  Error cells:        {total_error}")
    if total_reason_counts:
        print("  Staleness reasons:")
        for rtype, count in sorted(total_reason_counts.items()):
            print(f"    {rtype}: {count}")
    if total_error_counts:
        print("  Error types:")
        for etype, count in sorted(total_error_counts.items()):
            print(f"    {etype}: {count}")
    print()

    # Per-notebook error table (always print, even if no errors)
    # Collect all error types across all notebooks
    all_error_types: set = set()
    for s in staleness_data:
        all_error_types.update(s[5].keys())  # s[5] = error_counts
    error_types = sorted(all_error_types)

    print("PER-NOTEBOOK ERROR SUMMARY")
    print("-" * 110)

    # Build header with dynamic error type columns (wider for readability)
    col_width = 22
    header = f"{'Notebook':<35} {'Cells':>5} {'Errors':>6}"
    for etype in error_types:
        # Format error type names for column headers
        short_name = etype.replace("no_", "").replace("_", " ")[:col_width]
        header += f" {short_name:>{col_width}}"
    print(header)
    print("-" * (47 + (col_width + 1) * len(error_types)))

    # Per-notebook rows (all notebooks, sorted by name)
    for name, clean, stale, errors, reasons, err_counts in sorted(
        staleness_data, key=lambda x: x[0]
    ):
        display_name = name[:33] if len(name) > 33 else name
        row = f"{display_name:<35} {clean + stale + errors:>5} {errors:>6}"
        for etype in error_types:
            count = err_counts.get(etype, 0)
            row += f" {count:>{col_width}}"
        print(row)

    print("-" * (47 + (col_width + 1) * len(error_types)))
    print()

    print("=" * 120)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process FlowBook baseline comparison JSON files"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Comparison JSON files to process (supports remote paths like user@host:/path/*.json)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["slowdown", "memory", "runtime", "name"],
        default="slowdown",
        help="Sort files by metric (default: slowdown)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate PDF plots for each file"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for plot output files (default: current directory)",
    )
    parser.add_argument(
        "--large-fonts",
        action="store_true",
        help="Use larger fonts for paper-ready plots",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of remote files (ignore cache)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached remote files and exit",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top types/variables to show individually in plots (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="all_overhead.pdf",
        help="Output PDF filename for combined plots (default: all_overhead.pdf)",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include notebooks where baseline run has cell errors (excluded by default)",
    )
    parser.add_argument(
        "--no-sample-size",
        action="store_true",
        help="Hide N=... sample size annotations from CDF plots",
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

    # Group trial files by notebook stem
    trial_groups = group_trial_files(resolved_files)

    # Load all files, averaging trials when multiple exist for the same notebook
    stats_list: List[FileStats] = []
    file_data: Dict[str, Dict[str, Any]] = {}
    excluded_for_errors: List[Tuple[str, List[str]]] = []

    for stem, trial_files in trial_groups.items():
        # Filter to existing files
        existing_files = [f for f in trial_files if os.path.exists(f)]
        if not existing_files:
            print(f"Warning: No files found for {stem}", file=sys.stderr)
            continue

        try:
            if len(existing_files) > 1:
                # Multiple trials - load all and average
                print(
                    f"Averaging {len(existing_files)} trials for {stem}",
                    file=sys.stderr,
                )
                trial_datas = []
                for trial_file in sorted(existing_files):
                    trial_data = load_comparison_json(trial_file)
                    trial_datas.append(trial_data)
                data = average_trial_data(trial_datas)
                # Use stem as the representative path
                representative_path = sorted(existing_files)[0]
            else:
                # Single file
                representative_path = existing_files[0]
                data = load_comparison_json(representative_path)

            # Check for baseline errors and exclude unless --include-errors
            has_errors, error_cell_ids = has_baseline_errors(data)
            if has_errors and not args.include_errors:
                excluded_for_errors.append((representative_path, error_cell_ids))
                continue

            stats = compute_file_stats(data, representative_path)
            stats_list.append(stats)
            file_data[representative_path] = data

            # Print any memory measurement warnings
            warnings = extract_warnings(data)
            for w in warnings:
                print(
                    f"Memory warning ({Path(representative_path).name}): {w}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"Warning: Error loading {stem}: {e}", file=sys.stderr)
            continue

    if not stats_list:
        print("Error: No valid comparison files found", file=sys.stderr)
        sys.exit(1)

    # Check for v4/v5 format - older formats not supported
    has_supported_format = all(is_v4_or_v5_format(d) for d in file_data.values())

    if not has_supported_format:
        print(
            "Error: Only v4.0 or v5.0 format is supported. "
            "Re-run notebooks with current compare-baseline to produce v4/v5 output.",
            file=sys.stderr,
        )
        sys.exit(1)

    process_v4(file_data, args, excluded_for_errors)  # Handles both v4 and v5


if __name__ == "__main__":
    main()
