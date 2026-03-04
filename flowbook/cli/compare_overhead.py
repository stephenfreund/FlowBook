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

# Base directory for cached remote files
CACHE_BASE_DIR = "/tmp/flowbook_compare_overhead"


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
        match = re.match(r'^(.+)-(\d+)$', name)
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
        if values and all(isinstance(v, (int, float)) and v is not None for v in values):
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
        "execute_duration_ms", "code_duration_ms", "state_duration_ms", "check_duration_ms",
        "cell_runtime_ms",  # For compatibility with older formats
        "current_footprint_mb", "max_footprint_mb", "allocation_delta_mb", "gpu_mem_samples",
        "pre_only_bytes", "post_savings_bytes",
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
                    cells = trial.get("kernels", {}).get(kernel, {}).get(phase, {}).get(cell_list_key, [])
                    for c in cells:
                        key = (c.get("cell_id"), c.get("cell_index"))
                        all_cells_by_key[key].append(c)

                # Average each cell group
                averaged_cells = []
                for (cell_id, cell_index), trial_cells in sorted(
                    all_cells_by_key.items(),
                    key=lambda x: x[0][1] if x[0][1] is not None else 0
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
                first_checking = all_totals[0].get("checking_summary", {}) if all_totals else {}
                first_staleness = (
                    first_checking.get("clean_cells", 0),
                    first_checking.get("stale_cells", 0),
                    tuple(sorted(first_checking.get("reason_counts", {}).items()))
                )
                for i, totals in enumerate(all_totals[1:], start=2):
                    checking = totals.get("checking_summary", {})
                    staleness = (
                        checking.get("clean_cells", 0),
                        checking.get("stale_cells", 0),
                        tuple(sorted(checking.get("reason_counts", {}).items()))
                    )
                    if staleness != first_staleness:
                        notebook_path = result.get("notebook_path", "unknown")
                        print(f"WARNING: Staleness differs across trials for {notebook_path}:", file=sys.stderr)
                        print(f"  Trial 1: clean={first_staleness[0]}, stale={first_staleness[1]}, reasons={dict(first_staleness[2])}", file=sys.stderr)
                        print(f"  Trial {i}: clean={staleness[0]}, stale={staleness[1]}, reasons={dict(staleness[2])}", file=sys.stderr)
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
    memory_overhead_ratio: float  # flowbook_memory / baseline_memory (like slowdown for time)
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
    # Number of trials averaged (1 = single trial)
    num_trials: int = 1
    # Per-cell data for aggregate statistics
    per_cell_checkpoint_overhead_ms: List[float] = field(default_factory=list)
    per_cell_total_overhead_ms: List[float] = field(default_factory=list)
    per_cell_memory_overhead_mb: List[float] = field(default_factory=list)
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
    all_memory_overhead_per_cell: List[float] = field(default_factory=list)
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

    # Support both v1.0 (cell_runtime_ms) and v2.0 (execute_duration_ms) key names
    baseline_runtime = baseline_totals.get("cell_runtime_ms") or baseline_totals.get("execute_duration_ms", 0.0)
    flowbook_runtime = flowbook_totals.get("cell_runtime_ms") or flowbook_totals.get("execute_duration_ms", 0.0)
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
        slowdown = (flowbook_runtime + state_overhead + check_overhead) / flowbook_runtime
        state_pct = (state_overhead / flowbook_runtime) * 100
        check_pct = (check_overhead / flowbook_runtime) * 100
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

    # Check if FlowBook JSON has precomputed memory_overhead_ratio (new format)
    if "memory_overhead_ratio" in flowbook_mem_totals:
        # Use precomputed ratio: (base_namespace + checkpoint_overhead) / base_namespace
        memory_overhead_ratio = flowbook_mem_totals["memory_overhead_ratio"]
        # Compute memory_pct from the ratio (ratio - 1.0 is the fractional overhead)
        memory_pct = (memory_overhead_ratio - 1.0) * 100
        # For baseline_memory, use base_namespace_mb if no baseline kernel was run
        if baseline_memory == 0:
            baseline_memory = int(flowbook_mem_totals.get("base_namespace_mb", 0) * mb_to_bytes)
            memory_overhead = int(flowbook_mem_totals.get("total_overhead_mb", 0) * mb_to_bytes)
    elif baseline_memory > 0:
        memory_pct = (memory_overhead / baseline_memory) * 100
        memory_overhead_ratio = flowbook_memory / baseline_memory
    else:
        # No baseline and no precomputed ratio: compute ratio from checkpoint overhead
        # Get checkpoint overhead from last memory cell's overhead_breakdown
        flowbook_mem_cells = flowbook_memory_data.get("cells", []) if flowbook_memory_data else []
        checkpoint_overhead_mb = 0.0
        if flowbook_mem_cells:
            last_mem_cell = flowbook_mem_cells[-1]
            overhead_breakdown = last_mem_cell.get("overhead_breakdown", {})
            checkpoint_overhead_mb = overhead_breakdown.get("checkpoints_mb", 0.0)

        flowbook_memory_mb = flowbook_memory / mb_to_bytes if flowbook_memory > 0 else 0.0

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
    rerun_baseline_cells = baseline_timing.get("rerun_cells", []) if baseline_timing else []
    rerun_flowbook_cells = flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    num_reruns = len(rerun_flowbook_cells)

    rerun_baseline_runtime = sum(c.get("cell_runtime_ms") or c.get("execute_duration_ms", 0) for c in rerun_baseline_cells)
    rerun_flowbook_runtime = sum(c.get("cell_runtime_ms") or c.get("execute_duration_ms", 0) for c in rerun_flowbook_cells)
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

        last_baseline_runtime = last_bc.get("cell_runtime_ms") or last_bc.get("execute_duration_ms", 0.0)
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

    # Collect per-cell overhead data for aggregate statistics
    per_cell_checkpoint_overhead_ms: List[float] = []
    per_cell_total_overhead_ms: List[float] = []
    per_cell_memory_overhead_mb: List[float] = []

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

    # Per-cell memory overhead (from cumulative_by_var - compute delta between consecutive cells)
    prev_cumulative = 0
    for fc in flowbook_mem_cells:
        cumulative_by_var = fc.get("cumulative_by_var", {})
        if cumulative_by_var:
            # Sum all variable cumulative values at this cell
            total_cumulative = sum(cumulative_by_var.values())
            # Per-cell is the delta from previous cell
            per_cell_bytes = max(0, total_cumulative - prev_cumulative)
            per_cell_memory_overhead_mb.append(per_cell_bytes / (1024 * 1024))
            prev_cumulative = total_cumulative
        else:
            per_cell_memory_overhead_mb.append(0.0)

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

    # Collect all per-cell data across all files for aggregate per-cell statistics
    all_checkpoint_overhead = []
    all_total_overhead = []
    all_memory_overhead = []
    for s in stats_list:
        all_checkpoint_overhead.extend(s.per_cell_checkpoint_overhead_ms)
        all_total_overhead.extend(s.per_cell_total_overhead_ms)
        all_memory_overhead.extend(s.per_cell_memory_overhead_mb)

    # Compute per-cell statistics
    checkpoint_arr = np.array(all_checkpoint_overhead) if all_checkpoint_overhead else np.array([0.0])
    total_arr = np.array(all_total_overhead) if all_total_overhead else np.array([0.0])
    memory_arr = np.array(all_memory_overhead) if all_memory_overhead else np.array([0.0])

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
                name = s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
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
    lines.append("PER-CELL MEMORY OVERHEAD (MB)")
    lines.append(f"  Mean:   {aggregate.memory_overhead_per_cell_mean:.2f}MB")
    lines.append(f"  Median: {aggregate.memory_overhead_per_cell_median:.2f}MB")
    lines.append(f"  Min:    {aggregate.memory_overhead_per_cell_min:.2f}MB")
    lines.append(f"  Max:    {aggregate.memory_overhead_per_cell_max:.2f}MB")
    lines.append(f"  P90:    {aggregate.memory_overhead_per_cell_p90:.2f}MB")
    lines.append(f"  P95:    {aggregate.memory_overhead_per_cell_p95:.2f}MB")
    lines.append(f"  P99:    {aggregate.memory_overhead_per_cell_p99:.2f}MB")
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
        # No baseline memory - show checkpoint overhead in MB instead
        total_checkpoint_mb = sum(s.flowbook_memory_bytes for s in stats_list) / (1024 * 1024)
        lines.append("CHECKPOINT MEMORY")
        lines.append(f"AGGREGATE (N={aggregate.num_files})")
        lines.append(f"  Total:              {total_checkpoint_mb:.2f}MB")
        lines.append(f"  Per-Cell Mean:      {aggregate.memory_overhead_per_cell_mean:.2f}MB")
        lines.append(f"  Per-Cell Median:    {aggregate.memory_overhead_per_cell_median:.2f}MB")
        lines.append(f"  Per-Cell Max:       {aggregate.memory_overhead_per_cell_max:.2f}MB")
        lines.append(f"  Per-Cell P99:       {aggregate.memory_overhead_per_cell_p99:.2f}MB")

    # Checking results summary (staleness)
    # Note: Staleness consistency across trials is checked during averaging in average_trial_data
    total_checked = (aggregate.total_checking_clean_cells +
                     aggregate.total_checking_stale_cells +
                     aggregate.total_checking_error_cells)
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

        # Per-file rows (only files with errors)
        for s in stats_list:
            if s.checking_error_cells > 0:
                name = s.notebook_name[:28] if len(s.notebook_name) > 28 else s.notebook_name
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
    lines.append("notebook,cells,trials,baseline_ms,flowbook_ms,state_ms,check_ms,slowdown,state_pct,check_pct,memory_pct")

    for s in stats_list:
        lines.append(
            f'"{s.notebook_name}",{s.num_cells},{s.num_trials},{s.baseline_runtime_ms:.1f},{s.flowbook_total_ms:.1f},'
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

        # Get MAX cumulative for ordering (captures peak contribution, not just final)
        var_max: Dict[str, int] = {v: max(var_by_cell[v]) if var_by_cell[v] else 0 for v in all_var_names}
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

        # Get MAX cumulative for ordering (captures peak contribution)
        var_max = {v: max(var_by_cell[v]) if var_by_cell[v] else 0 for v in all_var_names}

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

    # Check data availability
    has_memory = bool(flowbook_mem_cells)  # FlowBook memory data is sufficient
    has_baseline_memory = bool(baseline_mem_cells)
    timing_var_data = extract_checkpoint_timing_var_data(data, top_n=top_n)
    var_data = extract_checkpoint_var_data(data, top_n=top_n)

    # Always use 2x3 grid layout
    fig, axes_2d = plt.subplots(3, 2, figsize=(14, 18))
    axes = [
        axes_2d[0, 0], axes_2d[0, 1],  # Row 1: Timing, Checkpoint Time by Variable
        axes_2d[1, 0], axes_2d[1, 1],  # Row 2: Memory, Checkpoint Memory by Variable
        axes_2d[2, 0], axes_2d[2, 1],  # Row 3: Overhead per Cell, Memory Overhead per Cell
    ]

    # Prepare shared data for timing
    cell_data_map = {}
    has_baseline = bool(baseline_cells)
    if baseline_cells and flowbook_cells:
        # Both baseline and flowbook available - align by cell_id
        for c in baseline_cells:
            cell_data_map[c["cell_id"]] = {"baseline_ms": c.get("execute_duration_ms", c.get("cell_runtime_ms", 0))}
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
    baseline_arr = np.array([cell_data_map[cid].get("baseline_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    code_arr = np.array([cell_data_map[cid].get("code_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    state_arr = np.array([cell_data_map[cid].get("state_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    check_arr = np.array([cell_data_map[cid].get("check_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    execute_arr = np.array([cell_data_map[cid].get("execute_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    other_arr = np.maximum(execute_arr - (code_arr + state_arr + check_arr), 0) if cell_ids else np.array([])

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
        ax.plot(cells, baseline_cumsum / 1000, color=colors[0], linewidth=2, marker='o', markersize=4, label=baseline_label)

        # FlowBook as stacked area: code (bottom) + state + check + other (top)
        ax.fill_between(cells, 0, code_cumsum / 1000, alpha=0.3, color=colors[1], label="FlowBook Code")
        ax.fill_between(cells, code_cumsum / 1000, (code_cumsum + state_cumsum) / 1000, alpha=0.4, color=colors[2], label="State")
        ax.fill_between(cells, (code_cumsum + state_cumsum) / 1000, (code_cumsum + state_cumsum + check_cumsum) / 1000, alpha=0.4, color=colors[3], label="Check")
        ax.fill_between(cells, (code_cumsum + state_cumsum + check_cumsum) / 1000, (code_cumsum + state_cumsum + check_cumsum + other_cumsum) / 1000, alpha=0.4, color=colors[4], label="Other")

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)
        title = "Timing Comparison" if has_baseline else "Timing (FlowBook only)"
        if timing_initial_count < len(cells):
            title += f" (cells 1-{timing_initial_count} + {len(cells) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', labelsize=tick_size)

        # Add separator line for rerun phase
        if timing_initial_count < len(cells):
            ax.axvline(x=timing_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.legend(loc="upper left", fontsize=legend_size)

        # Add timing breakdown text box
        total_code_s = code_cumsum[-1] / 1000
        total_state_s = state_cumsum[-1] / 1000
        total_check_s = check_cumsum[-1] / 1000
        total_other_s = other_cumsum[-1] / 1000
        total_flowbook_s = total_code_s + total_state_s + total_check_s + total_other_s
        total_baseline_s = baseline_cumsum[-1] / 1000

        if has_baseline:
            textstr = f'Baseline: {total_baseline_s:.2f}s\nFlowBook: {total_flowbook_s:.2f}s'
        else:
            textstr = f'Code: {total_code_s:.2f}s\nTotal: {total_flowbook_s:.2f}s'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.70, textstr, transform=ax.transAxes, fontsize=legend_size,
                verticalalignment='top', horizontalalignment='left', bbox=props)

        # Annotate overhead percentage relative to FlowBook Code time
        total_overhead_s = total_state_s + total_check_s + total_other_s
        if total_code_s > 0:
            overhead_pct = (total_overhead_s / total_code_s) * 100
            ax.annotate(f'{overhead_pct:.1f}% overhead (vs code)',
                        xy=(cells[-1], total_flowbook_s),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=legend_size, va='center', ha='left', color=colors[1])
    else:
        ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Timing Comparison", fontsize=title_size)

    # ========== Panel 2: Checkpoint Time by Variable (top-right) ==========
    ax = axes[1]
    if timing_var_data is not None:
        var_colors = sns.color_palette("husl", len(timing_var_data["vars_ordered"]))
        timing_cells = np.array(timing_var_data["cells"])
        timing_var_types = timing_var_data.get("var_types", {})

        # Stack checkpoint timing by variable (ms -> seconds)
        stacked = [np.array(timing_var_data["by_var"][v]) / 1000 for v in timing_var_data["vars_ordered"]]
        cumulative_timing = np.zeros(len(timing_cells))
        for i, (v, data_sec) in enumerate(zip(timing_var_data["vars_ordered"], stacked)):
            var_type = timing_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(timing_cells, cumulative_timing, cumulative_timing + data_sec, alpha=0.7, color=var_colors[i], label=label)
            cumulative_timing = cumulative_timing + data_sec

        # Draw total line
        ax.plot(timing_cells, cumulative_timing, color='black', linewidth=1.5, linestyle='--', label='Total')

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
    else:
        ax.text(0.5, 0.5, 'No checkpoint timing data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Checkpoint Time by Variable", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 3: Memory Overhead (middle-left) ==========
    ax = axes[2]
    if has_memory:
        mem_cells_arr = np.arange(1, len(flowbook_mem_cells) + 1)
        # Handle case when baseline memory is not available
        if has_baseline_memory and len(baseline_mem_cells) == len(flowbook_mem_cells):
            baseline_footprint = np.array([c.get("current_footprint_mb", 0) for c in baseline_mem_cells])
            baseline_gpu = np.array([c.get("gpu_mem_samples", 0) for c in baseline_mem_cells])
        else:
            # No baseline - use base_namespace_mb from FlowBook as the "baseline"
            # This is the size of user_ns without checkpoint data
            baseline_footprint = np.array([c.get("base_namespace_mb", c.get("current_footprint_mb", 0)) for c in flowbook_mem_cells])
            baseline_gpu = np.zeros(len(flowbook_mem_cells))
        flowbook_footprint = np.array([c.get("current_footprint_mb", 0) for c in flowbook_mem_cells])

        has_overhead_breakdown = any(c.get("overhead_breakdown") for c in flowbook_mem_cells)

        if has_overhead_breakdown:
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
            other_overhead_mb = np.array([
                (c.get("overhead_breakdown") or {}).get("other_mb", 0)
                for c in flowbook_mem_cells
            ])

            stack_colors = sns.color_palette("Set2", 5)
            has_gpu = any(g > 0 for g in baseline_gpu)

            # Layer 1: Baseline CPU memory (bottom - gray)
            base_label = 'Baseline CPU' if has_baseline_memory else 'User Namespace'
            ax.fill_between(mem_cells_arr, 0, baseline_footprint, alpha=0.3, color='gray', label=base_label)
            cumulative_mem = baseline_footprint.copy()

            # Layer 2: GPU memory (if present)
            if has_gpu:
                next_level = cumulative_mem + baseline_gpu
                ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.4, color='orange', label='GPU Memory')
                cumulative_mem = next_level

            # Layer 3: FlowBook overhead categories
            next_level = cumulative_mem + checkpoints_mb
            ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.5, color=stack_colors[0], label='Checkpoints')
            cumulative_mem = next_level

            next_level = cumulative_mem + execution_records_mb
            ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.5, color=stack_colors[1], label='Exec Records')
            cumulative_mem = next_level

            next_level = cumulative_mem + tracking_metadata_mb
            ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.5, color=stack_colors[2], label='Tracking')
            cumulative_mem = next_level

            next_level = cumulative_mem + other_overhead_mb
            ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.5, color=stack_colors[3], label='Other')
            cumulative_mem = next_level

            # Draw baseline line for reference
            ax.plot(mem_cells_arr, baseline_footprint, color='gray', linewidth=2, linestyle='--')

            # Calculate and annotate PEAK overhead percentage
            peak_overhead = np.max(cumulative_mem - baseline_footprint)
            peak_idx = np.argmax(cumulative_mem - baseline_footprint)
            if baseline_footprint[peak_idx] > 0:
                peak_overhead_pct = peak_overhead / baseline_footprint[peak_idx] * 100
                ax.annotate(f'{peak_overhead_pct:.1f}% peak overhead',
                            xy=(mem_cells_arr[peak_idx], cumulative_mem[peak_idx]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=legend_size, va='bottom', ha='left', color=colors[1])
        else:
            has_gpu = any(g > 0 for g in baseline_gpu)
            base_label = 'Baseline CPU' if has_baseline_memory else 'User Namespace'
            ax.fill_between(mem_cells_arr, 0, baseline_footprint, alpha=0.3, color=colors[0], label=base_label)
            cumulative_mem = baseline_footprint.copy()

            if has_gpu:
                next_level = cumulative_mem + baseline_gpu
                ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.4, color='orange', label='GPU Memory')
                cumulative_mem = next_level

            flowbook_overhead_mem = np.maximum(flowbook_footprint - baseline_footprint, 0)
            next_level = cumulative_mem + flowbook_overhead_mem
            ax.fill_between(mem_cells_arr, cumulative_mem, next_level, alpha=0.3, color=colors[1], label='FlowBook Overhead')
            cumulative_mem = next_level

            ax.plot(mem_cells_arr, baseline_footprint, color=colors[0], linewidth=2, marker='o', markersize=4)

            # Calculate and annotate PEAK overhead percentage
            peak_overhead = np.max(cumulative_mem - baseline_footprint)
            peak_idx = np.argmax(cumulative_mem - baseline_footprint)
            if baseline_footprint[peak_idx] > 0:
                peak_overhead_pct = peak_overhead / baseline_footprint[peak_idx] * 100
                ax.annotate(f'{peak_overhead_pct:.1f}% peak overhead',
                            xy=(mem_cells_arr[peak_idx], cumulative_mem[peak_idx]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=legend_size, va='bottom', ha='left', color=colors[1])

        title = 'Memory Overhead' if has_baseline_memory else 'Memory (FlowBook only)'
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel('Cell Number', fontsize=label_size)
        ax.set_ylabel('Memory (MB)', fontsize=label_size)

        if memory_initial_count < len(mem_cells_arr):
            ax.axvline(x=memory_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.legend(loc='upper left', fontsize=legend_size - 2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 4: Checkpoint Memory by Variable (middle-right) ==========
    ax = axes[3]
    if var_data is not None:
        var_colors = sns.color_palette("husl", len(var_data["vars_ordered"]))
        var_cells = np.array(var_data["cells"])
        mb = 1024 * 1024
        mem_var_types = var_data.get("var_types", {})

        # Get baseline memory for reference
        baseline_footprint_var = np.zeros(len(var_cells))
        if has_baseline_memory and len(baseline_mem_cells) >= len(var_cells):
            baseline_footprint_var = np.array([c.get("current_footprint_mb", 0) for c in baseline_mem_cells[:len(var_cells)]])
        elif has_memory and len(flowbook_mem_cells) >= len(var_cells):
            # No baseline - use base_namespace_mb from FlowBook
            baseline_footprint_var = np.array([c.get("base_namespace_mb", c.get("current_footprint_mb", 0)) for c in flowbook_mem_cells[:len(var_cells)]])

        # Draw baseline memory first (bottom layer)
        base_label = 'Baseline Memory' if has_baseline_memory else 'User Namespace'
        ax.fill_between(var_cells, 0, baseline_footprint_var, alpha=0.3, color=colors[0], label=base_label)

        # Stack checkpoint variables on top of baseline
        stacked = [np.array(var_data["by_var"][v]) / mb for v in var_data["vars_ordered"]]
        cumulative_var = baseline_footprint_var.copy()
        for i, (v, data_mb) in enumerate(zip(var_data["vars_ordered"], stacked)):
            var_type = mem_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(var_cells, cumulative_var, cumulative_var + data_mb, alpha=0.7, color=var_colors[i], label=label)
            cumulative_var = cumulative_var + data_mb

        # Draw baseline line
        ax.plot(var_cells, baseline_footprint_var, color=colors[0], linewidth=2, marker='o', markersize=4)

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
    else:
        ax.text(0.5, 0.5, 'No checkpoint memory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Checkpoint Memory by Variable", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 5: Overhead Time per Cell (bottom-left) ==========
    ax = axes[4]
    if len(cells) > 0:
        # Overhead per cell = state + check + other (everything except code time)
        overhead_per_cell = state_arr + check_arr + other_arr
        bar_width = 0.6

        # Stacked bar chart with breakdown
        ax.bar(cells, state_arr / 1000, width=bar_width, alpha=0.7, color=colors[2], label='State')
        ax.bar(cells, check_arr / 1000, width=bar_width, alpha=0.7, color=colors[3], label='Check', bottom=state_arr / 1000)
        ax.bar(cells, other_arr / 1000, width=bar_width, alpha=0.7, color=colors[4], label='Other', bottom=(state_arr + check_arr) / 1000)

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Overhead per Cell (seconds)", fontsize=label_size)
        title = "Overhead Time per Cell"
        if timing_initial_count < len(cells):
            title += f" (cells 1-{timing_initial_count} + {len(cells) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)

        if timing_initial_count < len(cells):
            ax.axvline(x=timing_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.legend(loc="upper right", fontsize=legend_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5, right=len(cells) + 0.5)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Overhead Time per Cell", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 6: Checkpoint Memory Overhead per Cell (bottom-right) ==========
    ax = axes[5]
    if var_data is not None:
        # Use the same data as Panel 4: sum cumulative values across all variables per cell
        # Then compute per-cell checkpoint size as delta from previous cell
        mb = 1024 * 1024
        var_cells = np.array(var_data["cells"])

        # Compute total cumulative checkpoint size at each cell
        total_cumulative = np.zeros(len(var_cells))
        for v in var_data["vars_ordered"]:
            total_cumulative += np.array(var_data["by_var"][v]) / mb

        # Compute per-cell checkpoint size (delta from previous)
        per_cell_mem = np.zeros(len(var_cells))
        per_cell_mem[0] = total_cumulative[0]  # First cell: all of it
        for i in range(1, len(var_cells)):
            per_cell_mem[i] = max(0, total_cumulative[i] - total_cumulative[i-1])

        bar_width = 0.6
        ax.bar(var_cells, per_cell_mem, width=bar_width, alpha=0.7, color='#66c2a5')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Checkpoint Size (MB)", fontsize=label_size)

        var_initial_count = var_data.get("initial_count", len(var_cells))
        title = "Checkpoint Memory per Cell"
        if var_initial_count < len(var_cells):
            title += f" (cells 1-{var_initial_count} + {len(var_cells) - var_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)

        if var_initial_count < len(var_cells):
            ax.axvline(x=var_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')
            ax.legend(loc="upper right", fontsize=legend_size)

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5, right=len(var_cells) + 0.5)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Checkpoint Memory per Cell", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Add notebook name as figure title
    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    fig.suptitle(notebook_name, fontsize=title_size + 2, fontweight='bold')
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
    large_fonts: bool = True
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

        ax.hist(data_plot, bins=30, alpha=0.3, color='steelblue', edgecolor='black')
        ax.axvline(np.median(data_plot), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(data_plot):.2f}')
        ax.axvline(np.mean(data_plot), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(data_plot):.2f}')
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel("Frequency", fontsize=label_size)
        ax.set_title("Total Overhead per Cell Distribution", fontsize=title_size)
        ax.legend(fontsize=tick_size)

        # Add stats text box
        n_total = len(total_overhead)
        n_filtered = len(total_overhead_filtered)
        outliers_removed = n_total - n_filtered
        textstr = f'N={n_filtered} (removed {outliers_removed} outliers)'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=tick_size,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Total Overhead per Cell Distribution", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Histogram 2: Memory Overhead per Cell
    ax = axes[1]
    if len(memory_overhead_filtered) > 0:
        ax.hist(memory_overhead_filtered, bins=30, alpha=0.3, color='seagreen', edgecolor='black')
        ax.axvline(np.median(memory_overhead_filtered), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(memory_overhead_filtered):.2f}MB')
        ax.axvline(np.mean(memory_overhead_filtered), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(memory_overhead_filtered):.2f}MB')
        ax.set_xlabel("Memory Overhead per Cell (MB)", fontsize=label_size)
        ax.set_ylabel("Frequency", fontsize=label_size)
        ax.set_title("Memory Overhead per Cell Distribution", fontsize=title_size)
        ax.legend(fontsize=tick_size)

        # Add stats text box
        n_total = len(memory_overhead)
        n_filtered = len(memory_overhead_filtered)
        outliers_removed = n_total - n_filtered
        textstr = f'N={n_filtered} (removed {outliers_removed} outliers)'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=tick_size,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead per Cell Distribution", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    fig.suptitle("Per-Cell Overhead Distributions (Outliers Removed)", fontsize=title_size + 2, fontweight='bold')
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
    large_fonts: bool = True
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

    sns.set_theme(style="whitegrid")

    # Font sizes
    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    tick_size = 14 if large_fonts else 10
    annotation_size = 12 if large_fonts else 9

    total_overhead = np.array(aggregate.all_total_overhead_per_cell)
    memory_overhead = np.array(aggregate.all_memory_overhead_per_cell)

    # Prepare data - keep in ms, convert memory from MB to bytes for log scale
    if len(total_overhead) > 0:
        total_data = np.sort(total_overhead)  # Keep in ms
        total_xlabel = "Total Overhead per Cell (ms)"
        total_cdf = np.arange(1, len(total_data) + 1) / len(total_data)
        total_stats = {
            'P50': np.percentile(total_data, 50),
            'P90': np.percentile(total_data, 90),
            'P95': np.percentile(total_data, 95),
            'P99': np.percentile(total_data, 99),
        }
    else:
        total_data = None

    if len(memory_overhead) > 0:
        # Convert MB to bytes for log scale (avoids negatives)
        memory_data_bytes = np.sort(memory_overhead * 1024 * 1024)  # MB to bytes
        memory_data_mb = np.sort(memory_overhead)  # Keep MB for linear plot
        memory_cdf = np.arange(1, len(memory_data_bytes) + 1) / len(memory_data_bytes)
        memory_stats_bytes = {
            'P50': np.percentile(memory_data_bytes, 50),
            'P90': np.percentile(memory_data_bytes, 90),
            'P95': np.percentile(memory_data_bytes, 95),
            'P99': np.percentile(memory_data_bytes, 99),
        }
        memory_stats_mb = {
            'P50': np.percentile(memory_data_mb, 50),
            'P90': np.percentile(memory_data_mb, 90),
            'P95': np.percentile(memory_data_mb, 95),
            'P99': np.percentile(memory_data_mb, 99),
        }
    else:
        memory_data_bytes = None
        memory_data_mb = None

    def format_bytes(b):
        """Format bytes to human readable."""
        if b >= 1024 * 1024 * 1024:
            return f'{b / (1024**3):.1f}GB'
        elif b >= 1024 * 1024:
            return f'{b / (1024**2):.1f}MB'
        elif b >= 1024:
            return f'{b / 1024:.1f}KB'
        else:
            return f'{b:.0f}B'

    def add_percentile_markers(ax, data, cdf, stats, unit_fmt, color, legend_fontsize):
        """Add percentile markers with vertical lines and labels, plus legend."""
        percentiles = ['P50', 'P90', 'P95', 'P99']
        y_positions = [0.5, 0.9, 0.95, 0.99]
        # Stagger labels: alternate above/below to avoid crowding
        label_offsets = [(5, 5), (5, -15), (5, 5), (5, -15)]  # (x, y) offsets
        label_vas = ['bottom', 'top', 'bottom', 'top']  # vertical alignments

        for pname, y_val, offset, va in zip(percentiles, y_positions, label_offsets, label_vas):
            if pname not in stats:
                continue
            x_val = stats[pname]
            # Vertical line from x-axis to the point (lighter)
            ax.vlines(x_val, 0, y_val, color=color, linestyle='--', linewidth=1, alpha=0.4)
            # Small point on the curve
            ax.scatter([x_val], [y_val], color=color, s=30, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
            # Staggered label by the dot
            ax.annotate(pname, (x_val, y_val), textcoords='offset points',
                       xytext=offset, fontsize=annotation_size, ha='left', va=va, fontweight='bold')

        # Add max point on the curve (at CDF = 1.0) with label
        if len(data) > 0:
            max_val = np.max(data)
            ax.scatter([max_val], [1.0], color=color, s=30, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
            ax.annotate('Max', (max_val, 1.0), textcoords='offset points',
                       xytext=(5, -15), fontsize=annotation_size, ha='left', va='top', fontweight='bold')

        # Add legend box with values in lower right (right-aligned values)
        formatted_values = {pname: unit_fmt(stats[pname]) for pname in percentiles if pname in stats}
        if len(data) > 0:
            formatted_values['Max'] = unit_fmt(np.max(data))
        max_val_len = max(len(v) for v in formatted_values.values()) if formatted_values else 0
        legend_keys = [p for p in percentiles if p in formatted_values] + (['Max'] if 'Max' in formatted_values else [])
        legend_lines = [f'{pname}: {formatted_values[pname]:>{max_val_len}}' for pname in legend_keys]
        legend_text = '\n'.join(legend_lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=legend_fontsize,
                verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

    def add_percentile_gridlines(ax):
        """Add horizontal gridlines at percentile levels."""
        for y in [0.5, 0.9, 0.95, 0.99]:
            ax.axhline(y, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    figures = []

    # --- Figure 1: Log scale with full distribution ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # Total overhead - log scale
    ax = axes1[0]
    if total_data is not None:
        pos_mask = total_data > 0
        if np.any(pos_mask):
            ax.fill_between(total_data[pos_mask], 0, total_cdf[pos_mask], alpha=0.3, color='steelblue', edgecolor='none')
            ax.plot(total_data[pos_mask], total_cdf[pos_mask], color='steelblue', linewidth=2)

            add_percentile_markers(ax, total_data[pos_mask], total_cdf[pos_mask],
                                  total_stats, lambda x: f'{x:.1f}ms', 'black', tick_size)
            add_percentile_gridlines(ax)

            ax.set_xscale('log')
            ax.set_xlabel(total_xlabel, fontsize=label_size)
            ax.set_ylabel("Cumulative Probability", fontsize=label_size)
            ax.set_title("Total Overhead (Log Scale)", fontsize=title_size)
            ax.set_ylim(0, 1.05)

            textstr = f'N={len(total_data)}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=tick_size,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Total Overhead (Log Scale)", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Memory overhead - log scale (in bytes)
    ax = axes1[1]
    if memory_data_bytes is not None:
        pos_mask = memory_data_bytes > 0
        if np.any(pos_mask):
            ax.fill_between(memory_data_bytes[pos_mask], 0, memory_cdf[pos_mask], alpha=0.3, color='seagreen', edgecolor='none')
            ax.plot(memory_data_bytes[pos_mask], memory_cdf[pos_mask], color='seagreen', linewidth=2)

            add_percentile_markers(ax, memory_data_bytes[pos_mask], memory_cdf[pos_mask],
                                  memory_stats_bytes, format_bytes, 'black', tick_size)
            add_percentile_gridlines(ax)

            ax.set_xscale('log')
            ax.set_xlabel("Memory Overhead per Cell (bytes)", fontsize=label_size)
            ax.set_ylabel("Cumulative Probability", fontsize=label_size)
            ax.set_title("Memory Overhead (Log Scale)", fontsize=title_size)
            ax.set_ylim(0, 1.05)

            textstr = f'N={len(memory_data_bytes)}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=tick_size,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead (Log Scale)", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    fig1.suptitle("Per-Cell Overhead CDFs (Full Distribution)", fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig1)

    # --- Figure 2: Linear scale zoomed to P99 ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Total overhead - linear, zoomed to P99
    ax = axes2[0]
    if total_data is not None:
        p99_val = total_stats['P99']
        mask = total_data <= p99_val * 1.05
        ax.fill_between(total_data[mask], 0, total_cdf[mask], alpha=0.3, color='steelblue', edgecolor='none')
        ax.plot(total_data[mask], total_cdf[mask], color='steelblue', linewidth=2)

        # Filter stats to those within range
        stats_in_range = {k: v for k, v in total_stats.items() if v <= p99_val * 1.05}
        add_percentile_markers(ax, total_data[mask], total_cdf[mask],
                              stats_in_range, lambda x: f'{x:.1f}ms', 'black', tick_size)
        add_percentile_gridlines(ax)

        ax.set_xlabel(total_xlabel, fontsize=label_size)
        ax.set_ylabel("Cumulative Probability", fontsize=label_size)
        ax.set_title("Total Overhead (Zoomed to P99)", fontsize=title_size)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)

        n_excluded = np.sum(~mask)
        textstr = f'N={np.sum(mask)} (excluded {n_excluded} > P99)'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=tick_size,
                verticalalignment='top', horizontalalignment='left', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Total Overhead (Zoomed to P99)", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Memory overhead - linear, zoomed to P99 (in MB for readability)
    ax = axes2[1]
    if memory_data_mb is not None:
        p99_val = memory_stats_mb['P99']
        mask = memory_data_mb <= p99_val * 1.05
        ax.fill_between(memory_data_mb[mask], 0, memory_cdf[mask], alpha=0.3, color='seagreen', edgecolor='none')
        ax.plot(memory_data_mb[mask], memory_cdf[mask], color='seagreen', linewidth=2)

        stats_in_range = {k: v for k, v in memory_stats_mb.items() if v <= p99_val * 1.05}
        add_percentile_markers(ax, memory_data_mb[mask], memory_cdf[mask],
                              stats_in_range, lambda x: f'{x:.1f}MB', 'black', tick_size)
        add_percentile_gridlines(ax)

        ax.set_xlabel("Memory Overhead per Cell (MB)", fontsize=label_size)
        ax.set_ylabel("Cumulative Probability", fontsize=label_size)
        ax.set_title("Memory Overhead (Zoomed to P99)", fontsize=title_size)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)

        n_excluded = np.sum(~mask)
        textstr = f'N={np.sum(mask)} (excluded {n_excluded} > P99)'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=tick_size,
                verticalalignment='top', horizontalalignment='left', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead (Zoomed to P99)", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    fig2.suptitle("Per-Cell Overhead CDFs (Zoomed to P99)", fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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

    # Group trial files by notebook stem
    trial_groups = group_trial_files(resolved_files)

    # Load all files, averaging trials when multiple exist for the same notebook
    stats_list: List[FileStats] = []
    file_data: Dict[str, Dict[str, Any]] = {}

    for stem, trial_files in trial_groups.items():
        # Filter to existing files
        existing_files = [f for f in trial_files if os.path.exists(f)]
        if not existing_files:
            print(f"Warning: No files found for {stem}", file=sys.stderr)
            continue

        try:
            if len(existing_files) > 1:
                # Multiple trials - load all and average
                print(f"Averaging {len(existing_files)} trials for {stem}", file=sys.stderr)
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

            stats = compute_file_stats(data, representative_path)
            stats_list.append(stats)
            file_data[representative_path] = data

            # Print any memory measurement warnings
            warnings = extract_warnings(data)
            for w in warnings:
                print(f"Memory warning ({Path(representative_path).name}): {w}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error loading {stem}: {e}", file=sys.stderr)
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

                # Add histogram plots at the end if we have aggregate data
                if aggregate.all_total_overhead_per_cell or aggregate.all_memory_overhead_per_cell:
                    hist_fig = plot_overhead_histograms(aggregate, output_path=None, large_fonts=args.large_fonts)
                    if hist_fig is not None:
                        pdf.savefig(hist_fig, dpi=150)
                        plt.close(hist_fig)

                    # Add CDF plots (returns list of figures)
                    cdf_figs = plot_overhead_cdfs(aggregate, output_path=None, large_fonts=args.large_fonts)
                    if cdf_figs is not None:
                        for cdf_fig in cdf_figs:
                            pdf.savefig(cdf_fig, dpi=150)
                            plt.close(cdf_fig)

            print(f"Combined overhead plots saved to: {combined_path}")


if __name__ == "__main__":
    main()
