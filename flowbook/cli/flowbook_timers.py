#!/usr/bin/env python3
"""
Command line tool to analyze timing data from flowbook-times.json files.

Reads timing data and displays statistics for each timer type including:
- Count, Total, Mean, Median, Min, Max, Std Dev, P95

Supports single or multiple files with combined statistics.
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import termcolor

# Base directory for cached remote files
CACHE_BASE_DIR = "/tmp/flowbook_timers"


def truncate_timer_key(key: str, max_length: int = 50) -> str:
    """
    Truncate a timer key to a maximum length.

    If the key is longer than max_length, it's truncated to show
    prefix and suffix separated by '...'.

    Args:
        key: The timer key to potentially truncate
        max_length: Maximum length (default: 50)

    Returns:
        Truncated key in format "abc...xyz" if too long, otherwise original key

    Examples:
        >>> truncate_timer_key("short_key")
        'short_key'
        >>> truncate_timer_key("a" * 100, max_length=50)
        'aaaaaaaaaaaaaaaaaaaaaaa...aaaaaaaaaaaaaaaaaaaaaa'
    """
    if len(key) <= max_length:
        return key

    # Reserve 3 characters for '...'
    available = max_length - 3
    # Split remaining space between prefix and suffix
    prefix_len = available // 2
    suffix_len = available - prefix_len

    return f"{key[:prefix_len]}...{key[-suffix_len:]}"


def parse_remote_path(path: str) -> tuple[bool, str, str, str]:
    """
    Parse a path to detect if it's a remote path.

    Args:
        path: File path that may be local or remote

    Returns:
        Tuple of (is_remote, user, host, remote_path)
        For local paths, returns (False, '', '', path)
    """
    # Pattern: [user@]host:path
    # Must have a colon, and the part before colon must look like a host
    # (not a Windows drive letter like C:)
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
        Path to the cache directory (e.g., "/tmp/flowbook_timers/a1b2c3d4")
    """
    # Create a hash of the remote spec
    hash_val = hashlib.md5(remote_spec.encode()).hexdigest()[:8]
    return os.path.join(CACHE_BASE_DIR, hash_val)


def rsync_remote_files(
    remote_spec: str,
    force_download: bool = False
) -> tuple[list[str], str]:
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
        # For wildcards, we need to handle them specially
        if '*' in remote_path or '?' in remote_path:
            # Get the directory containing the pattern
            remote_dir = os.path.dirname(remote_path)
            pattern = os.path.basename(remote_path)

            # Rsync the directory with include pattern
            rsync_cmd = [
                'rsync', '-avz',
                '--include', pattern,
                '--exclude', '*',
                f"{remote_host}:{remote_dir}/",
                cache_dir + "/"
            ]
        else:
            # Simple file or directory
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
                # Show files transferred (but not verbose rsync output)
                lines = result.stdout.strip().split('\n')
                file_lines = [l for l in lines if l and not l.startswith(('sending', 'sent', 'total', 'receiving'))]
                if file_lines:
                    print(f"Files:  {len(file_lines)} file(s) synced", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error: rsync failed: {e.stderr}", file=sys.stderr)
            raise

    print("", file=sys.stderr)  # Blank line for readability

    # Find the local files matching the pattern
    if '*' in remote_path or '?' in remote_path:
        pattern = os.path.basename(remote_path)
        local_files = glob.glob(os.path.join(cache_dir, pattern))
    else:
        # Single file
        filename = os.path.basename(remote_path)
        local_path = os.path.join(cache_dir, filename)
        local_files = [local_path] if os.path.exists(local_path) else []

    return sorted(local_files), cache_dir


def resolve_file_paths(
    file_paths: list[str],
    force_download: bool = False
) -> list[str]:
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


def sanitize_filename(key: str) -> str:
    """
    Sanitize a timer key for use as a filename.

    Args:
        key: Timer key (e.g., "diff:compute")

    Returns:
        Sanitized filename (e.g., "diff_compute")
    """
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[:/\\<>"|?*\s]', '_', key)
    # Remove leading/trailing underscores and collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized


def create_histogram_pdf(
    durations: list[float],
    key: str,
    output_dir: str = '.',
    clip_percentile: Optional[float] = None
) -> str:
    """
    Create a seaborn-styled histogram and save as PDF.

    For highly skewed data, clips the x-axis to P99 for better visualization
    while still showing the full statistics.

    Args:
        durations: List of timing values
        key: Timer key name (used in title and filename)
        output_dir: Directory to save PDF
        clip_percentile: If specified, clip values above this percentile (e.g., 99)

    Returns:
        Path to created PDF file
    """
    # Calculate statistics on full data
    p90 = np.percentile(durations, 90)
    p95 = np.percentile(durations, 95)
    p99 = np.percentile(durations, 99)
    max_val = np.max(durations)
    min_val = np.min(durations)
    mean_val = np.mean(durations)
    median_val = np.median(durations)
    std_val = np.std(durations)

    # Determine clipping behavior
    if clip_percentile is not None:
        # User-specified clip percentile
        clip_threshold = np.percentile(durations, clip_percentile)
        plot_durations = [d for d in durations if d <= clip_threshold]
        clipped_count = len(durations) - len(plot_durations)
        is_clipped = clipped_count > 0
        clip_note = f"\n(clipped to P{clip_percentile:.0f}, {clipped_count} values removed)" if is_clipped else ""
    else:
        # Auto-clip: Check if data is highly skewed (max >> P99)
        # If so, clip to P99 for better visualization
        is_clipped = max_val > p99 * 2 and p99 > 0
        if is_clipped:
            plot_durations = [d for d in durations if d <= p99]
            clip_note = f"\n(showing up to P99, {len(durations) - len(plot_durations)} outliers clipped)"
        else:
            plot_durations = durations
            clip_note = ""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create histogram (percentage on y-axis)
    sns.histplot(plot_durations, kde=False, ax=ax, color='steelblue', edgecolor='white', stat='percent')

    # Add vertical lines for P90, P95 (only if within plot range)
    if p90 <= (p99 if is_clipped else max_val):
        ax.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90:.2f}')
    if p95 <= (p99 if is_clipped else max_val):
        ax.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'P95: {p95:.2f}')
    if not is_clipped:
        ax.axvline(max_val, color='darkred', linestyle='-', linewidth=2, label=f'Max: {max_val:.2f}')

    # Add legend for the lines
    ax.legend(loc='upper right', fontsize=10)

    # Labels and title
    ax.set_xlabel('Duration (ms)', fontsize=12)
    ax.set_ylabel('Percent', fontsize=12)
    ax.set_title(f'Distribution of {key}\n(n={len(durations)}){clip_note}', fontsize=14)

    # Add statistics annotation (always show full stats)
    stats_text = (
        f"Mean: {mean_val:.2f} ms\n"
        f"Median: {median_val:.2f} ms\n"
        f"Std: {std_val:.2f} ms\n"
        f"Min: {min_val:.2f} ms\n"
        f"P90: {p90:.2f} ms\n"
        f"P95: {p95:.2f} ms\n"
        f"P99: {p99:.2f} ms\n"
        f"Max: {max_val:.2f} ms"
    )
    ax.text(0.95, 0.65, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, family='monospace')

    plt.tight_layout()

    # Save to PDF
    filename = f"{sanitize_filename(key)}.pdf"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return filepath


def create_scatterplot_pdf(
    durations_x: list[float],
    durations_y: list[float],
    key_x: str,
    key_y: str,
    output_dir: str = '.'
) -> str:
    """
    Create a seaborn-styled scatter plot and save as PDF.

    NOTE: Points are paired by index position. This is only meaningful if
    the two timer keys are always recorded together in the same order
    during each event (e.g., both measured for every cell execution).

    Args:
        durations_x: X-axis timing values
        durations_y: Y-axis timing values
        key_x: X-axis timer key name
        key_y: Y-axis timer key name
        output_dir: Directory to save PDF

    Returns:
        Path to created PDF file
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create scatter plot with regression line
    sns.regplot(x=durations_x, y=durations_y, ax=ax,
                scatter_kws={'alpha': 0.6, 's': 50},
                line_kws={'color': 'red', 'linewidth': 1.5})

    # Labels and title
    ax.set_xlabel(f'{key_x} (ms)', fontsize=12)
    ax.set_ylabel(f'{key_y} (ms)', fontsize=12)
    ax.set_title(f'{key_x} vs {key_y}\n(n={len(durations_x)})', fontsize=14)

    # Calculate and display correlation
    if len(durations_x) > 1:
        correlation = np.corrcoef(durations_x, durations_y)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add warning note
    ax.text(0.5, -0.12, 'Note: Points paired by index. Only meaningful if keys are recorded in lockstep.',
            transform=ax.transAxes, fontsize=9, ha='center', style='italic', color='gray')

    plt.tight_layout()

    # Save to PDF
    filename = f"{sanitize_filename(key_x)}_{sanitize_filename(key_y)}.pdf"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return filepath


def create_ascii_histogram(data, bins=20, width=80, bar_char="#", clip_percentile=None):
    """
    Generates an ASCII histogram for the given data.

    Args:
        data: List of values to histogram
        bins: Number of bins
        width: Width of the bars
        bar_char: Character to use for bars
        clip_percentile: If specified, clip values above this percentile

    Returns:
        ASCII histogram string
    """
    if not data:
        return ""

    # Apply clipping if specified
    clipped_count = 0
    if clip_percentile is not None:
        clip_threshold = np.percentile(data, clip_percentile)
        original_len = len(data)
        data = [d for d in data if d <= clip_threshold]
        clipped_count = original_len - len(data)
        if not data:
            return f"All {original_len} values clipped at P{clip_percentile:.0f}"

    # 1. Determine the range of the data
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return "All data points are the same."

    bin_width = (max_val - min_val) / bins

    # 2. Count frequencies for each bin
    counts = [0] * bins
    for item in data:
        # Calculate the bin index
        if item == max_val:
            bin_index = bins - 1  # Handle the edge case of max value
        else:
            bin_index = int((item - min_val) // bin_width)
        counts[bin_index] += 1

    # 3. Generate the ASCII visualization
    max_count = max(counts)
    chart = []
    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        count = counts[i]
        percent = (count / len(data)) * 100 if max_count > 0 else 0
        # Scale the bar length
        bar_length = int((count / max_count) * width) if max_count > 0 else 0
        chart.append(
            f"{bin_start:7.3f} - {bin_end:7.3f} | {bar_char * bar_length} ({percent:.1f}%)"
        )

    result = "\n".join(chart)
    if clipped_count > 0:
        result += f"\n[{clipped_count} values clipped above P{clip_percentile:.0f}]"
    return result


@dataclass
class TimerStats:
    """Statistics for a single timer."""

    key: str
    count: int
    total: float
    mean: float
    median: float
    min: float
    max: float
    std: float
    p90: float
    p95: float


def load_timings(file_path: str) -> list[dict]:
    """
    Load timing data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of timing records, or empty list on error
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a list", file=sys.stderr)
            return []

        # Validate structure
        valid_records = []
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                print(
                    f"Warning: Record {i} in {file_path} is not a dict", file=sys.stderr
                )
                continue
            if "key" not in record or "duration" not in record:
                print(
                    f"Warning: Record {i} in {file_path} missing 'key' or 'duration'",
                    file=sys.stderr,
                )
                continue
            valid_records.append(record)

        return valid_records

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []


def load_multiple_timings(file_paths: list[str]) -> dict[str, list[dict]]:
    """
    Load timing data from multiple files.

    Args:
        file_paths: List of file paths

    Returns:
        Dict mapping file_path -> list of timing records
    """
    results = {}
    for file_path in file_paths:
        timings = load_timings(file_path)
        if timings:  # Only include files that loaded successfully
            results[file_path] = timings

    return results


def group_by_key(timings: list[dict]) -> dict[str, list[float]]:
    """
    Group timing durations by key.

    Args:
        timings: List of timing records

    Returns:
        Dict mapping timer key -> list of durations
    """
    grouped = {}
    for record in timings:
        key = record["key"]
        duration = float(record["duration"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(duration)
    return grouped


def compute_synthetic_metrics(grouped: dict[str, list[float]]) -> dict[str, list[float]]:
    """
    Compute synthetic metrics from grouped timing data.

    Currently computes:
    - Computed:Slowdown = sum(execute:reproducibility) / sum(kernel:execute)
    - Computed:CheckpointSlowdown = sum(kernel:checkpoint) / sum(kernel:execute)

    Args:
        grouped: Dict mapping timer key -> list of durations

    Returns:
        Updated dict with synthetic metrics added
    """
    kernel_execute_key = "kernel:execute"
    total_kernel_execute = sum(grouped.get(kernel_execute_key, []))

    # Compute Slowdown: execute:reproducibility / kernel:execute
    execute_key = "execute:reproducibility"
    if execute_key in grouped and total_kernel_execute > 0:
        total_execute = sum(grouped[execute_key])
        slowdown = total_execute / total_kernel_execute
        grouped["Computed:Slowdown"] = [slowdown]

    # Compute CheckpointSlowdown: 1 + kernel:checkpoint / kernel:execute
    checkpoint_key = "kernel:checkpoint"
    if checkpoint_key in grouped and total_kernel_execute > 0:
        total_checkpoint = sum(grouped[checkpoint_key])
        checkpoint_slowdown = 1 + total_checkpoint / total_kernel_execute
        grouped["Computed:CheckpointSlowdown"] = [checkpoint_slowdown]

    return grouped


# List of all synthetic metric keys
SYNTHETIC_METRIC_KEYS = ["Computed:Slowdown", "Computed:CheckpointSlowdown"]


def collect_synthetic_metrics(
    grouped_by_file: dict[str, dict[str, list[float]]],
    combined_grouped: dict[str, list[float]]
) -> dict[str, list[float]]:
    """
    Collect per-file synthetic metrics into combined grouped data.

    Instead of computing synthetic metrics from combined totals, this collects
    the individual per-file values so statistics reflect the distribution across files.

    Args:
        grouped_by_file: Dict mapping file_path -> grouped timing data (with synthetic metrics)
        combined_grouped: Combined grouped timing data (without synthetic metrics yet)

    Returns:
        Updated combined_grouped with synthetic metrics collected from each file
    """
    # Collect all synthetic metrics from each file
    for key in SYNTHETIC_METRIC_KEYS:
        values = []
        for file_path, grouped in grouped_by_file.items():
            if key in grouped:
                values.extend(grouped[key])
        if values:
            combined_grouped[key] = values

    return combined_grouped


def calculate_stats(durations: list[float]) -> dict:
    """
    Calculate statistics for a list of durations.

    Args:
        durations: List of duration values

    Returns:
        Dict with statistical measures
    """
    if not durations:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    arr = np.array(durations)
    return {
        "count": len(durations),
        "total": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def build_stats_table(
    timings: list[dict],
    compute_synthetic: bool = True
) -> list[TimerStats]:
    """
    Build statistics table from timing data.

    Args:
        timings: List of timing records
        compute_synthetic: Whether to compute synthetic metrics (default True).
            Set to False when combining multiple files to avoid computing from
            combined totals instead of per-file values.

    Returns:
        List of TimerStats objects
    """
    if not timings:
        return []

    grouped = group_by_key(timings)
    if compute_synthetic:
        grouped = compute_synthetic_metrics(grouped)
    stats_list = []

    for key, durations in grouped.items():
        stats = calculate_stats(durations)
        stats_list.append(TimerStats(key=key, **stats))

    return stats_list


def combine_timings(timings_by_file: dict[str, list[dict]]) -> list[dict]:
    """
    Combine timings from all files into a single list.

    Args:
        timings_by_file: Dict mapping file_path -> timings

    Returns:
        Combined list of all timing records
    """
    combined = []
    for timings in timings_by_file.values():
        combined.extend(timings)
    return combined


def format_time(ms: float, use_commas: bool = False) -> str:
    """
    Format time value in milliseconds.

    Args:
        ms: Time in milliseconds (values from JSON are already in ms)
        use_commas: Whether to use comma separators (for terminal output)

    Returns:
        Formatted string
    """
    if use_commas:
        return f"{ms:,.2f}"
    else:
        return f"{ms:.2f}"


def format_table(
    stats: list[TimerStats],
    sort_by: str = "total",
    title: Optional[str] = None,
    top: Optional[int] = None,
    grouped_timings: Optional[dict[str, list[float]]] = None,
    show_histograms: bool = False,
    clip_percentile: Optional[float] = None,
) -> str:
    """
    Format statistics as a table.

    Args:
        stats: List of TimerStats objects
        sort_by: Field to sort by ('total', 'mean', 'count', 'max', 'key')
        title: Optional title for the table
        top: Optional limit to show only top N timers
        grouped_timings: Optional dict mapping key -> list of durations (for histograms)
        show_histograms: Whether to show histograms for each key (requires grouped_timings)
        clip_percentile: Optional percentile to clip histogram values at

    Returns:
        Formatted table string
    """
    if not stats:
        return "No timing data available\n"

    # Parse GROUP:KEY format and group stats
    from collections import defaultdict

    def parse_key(key: str) -> tuple[str, str]:
        """Parse timer key into (group, subkey)."""
        if ":" in key:
            group, subkey = key.split(":", 1)
            return group, subkey
        return "", key  # No group

    # Group stats by group prefix
    grouped: dict[str, list[TimerStats]] = defaultdict(list)
    for s in stats:
        group, _ = parse_key(s.key)
        grouped[group].append(s)

    # Sort groups: empty group first, then alphabetically
    group_order = sorted(grouped.keys(), key=lambda g: (g != "", g))

    # Sort within each group by the specified field
    def sort_key(s: TimerStats):
        if sort_by == "count":
            return -s.count
        elif sort_by == "mean":
            return -s.mean
        elif sort_by == "max":
            return -s.max
        elif sort_by == "key":
            return s.key
        else:  # total (default)
            return -s.total

    for group in grouped:
        grouped[group] = sorted(grouped[group], key=sort_key)

    # Apply top N filter if specified (after grouping)
    if top is not None and top > 0:
        # Flatten, sort, take top N, then re-group
        all_stats = []
        for group in group_order:
            all_stats.extend(grouped[group])
        all_stats = sorted(all_stats, key=sort_key)[:top]
        grouped = defaultdict(list)
        for s in all_stats:
            group, _ = parse_key(s.key)
            grouped[group].append(s)
        group_order = sorted(grouped.keys(), key=lambda g: (g != "", g))
        for group in grouped:
            grouped[group] = sorted(grouped[group], key=sort_key)

    # Build table
    lines = []

    # Calculate key column width - use subkey length + 4 for indent, or full key for ungrouped
    # Keys are truncated to 50 chars max, so cap the width at 50
    max_subkey_len = 0
    for s in stats:
        group, subkey = parse_key(s.key)
        if group:
            display_len = min(len(subkey) + 4, 50)  # 4 spaces indent, max 50
            max_subkey_len = max(max_subkey_len, display_len)
        else:
            display_len = min(len(s.key), 50)
            max_subkey_len = max(max_subkey_len, display_len)
    # Also consider group names as headers
    for group in grouped:
        if group:
            max_subkey_len = max(max_subkey_len, min(len(group), 50))
    key_width = max(len("Timer Key"), max_subkey_len)

    # Total row width: key + space + Count(8) + space + 9 numeric columns (14 each) + 8 spaces + " | " separator
    row_width = key_width + 1 + 8 + 1 + 14 * 9 + 8 + 2

    # Title
    if title:
        lines.append(title)
    else:
        lines.append("Timer Statistics (all times in ms)")

    # Header separator
    lines.append("=" * row_width)

    # Column headers
    header = f"{'Timer Key':<{key_width}} {'Count':>8} {'Mean':>14} {'Median':>14} {'Max':>14} | {'P90':>14} {'P95':>14} {'Min':>14} {'Total':>14} {'Std Dev':>14}"
    lines.append(header)

    # Data separator
    lines.append("-" * row_width)

    # Data rows with alternating colors for readability
    row_idx = 0
    for group in group_order:
        group_stats = grouped[group]

        # Group header row (if group is not empty)
        if group:
            # Empty stats row with just the group name (truncated to 50 chars)
            truncated_group = truncate_timer_key(group, max_length=50)
            group_header = f"{truncated_group:<{key_width}} {'':>8} {'':>14} {'':>14} {'':>14} | {'':>14} {'':>14} {'':>14} {'':>14} {'':>14}"
            lines.append(termcolor.colored(group_header, "cyan", attrs=["bold"]))

        for s in group_stats:
            group_name, subkey = parse_key(s.key)
            # Indent subkeys if they have a group
            display_key = f"    {subkey}" if group_name else s.key
            # Truncate display key to max 50 characters
            display_key = truncate_timer_key(display_key, max_length=50)

            row = (
                f"{display_key:<{key_width}} "
                f"{s.count:>8} "
                f"{format_time(s.mean, use_commas=True):>14} "
                f"{format_time(s.median, use_commas=True):>14} "
                f"{format_time(s.max, use_commas=True):>14} | "
                f"{format_time(s.p90, use_commas=True):>14} "
                f"{format_time(s.p95, use_commas=True):>14} "
                f"{format_time(s.min, use_commas=True):>14} "
                f"{format_time(s.total, use_commas=True):>14} "
                f"{format_time(s.std, use_commas=True):>14}"
            )
            if row_idx % 2 == 0:
                row = termcolor.colored(row, "yellow")
            else:
                row = termcolor.colored(row, "white")
            lines.append(row)
            row_idx += 1

    # Footer separator
    lines.append("-" * row_width)

    # Total row
    all_stats_flat = [s for group in grouped.values() for s in group]
    total_count = sum(s.count for s in all_stats_flat)
    total_time = sum(s.total for s in all_stats_flat)
    mean_time = total_time / total_count if total_count > 0 else 0

    footer = (
        f"{'TOTAL':<{key_width}} "
        f"{total_count:>8} "
        f"{format_time(mean_time, use_commas=True):>14} "
        f"{'':>14} "  # Median
        f"{'':>14} | "  # Max
        f"{'':>14} "  # P90
        f"{'':>14} "  # P95
        f"{'':>14} "  # Min
        f"{format_time(total_time, use_commas=True):>14}"  # Total
    )
    lines.append(footer)

    # Add histograms if requested
    if show_histograms and grouped_timings:
        lines.append("")
        lines.append("=" * 80)
        lines.append("HISTOGRAMS (all times in ms)")
        lines.append("=" * 80)
        for s in all_stats_flat:
            if s.key in grouped_timings:
                durations = grouped_timings[s.key]
                if len(durations) > 1:  # Need at least 2 points for a histogram
                    lines.append("")
                    truncated_key = truncate_timer_key(s.key, max_length=50)
                    lines.append(f"--- {truncated_key} (n={len(durations)}) ---")
                    hist = create_ascii_histogram(durations, bar_char="█", clip_percentile=clip_percentile)
                    lines.append(hist)

    return "\n".join(lines)


def format_per_key_table(
    key: str,
    stats_by_file: dict[str, TimerStats],
    file_order: list[str],
) -> str:
    """
    Format a single timer key as a table with one row per file.

    Args:
        key: Timer key name
        stats_by_file: Dict mapping file_path -> TimerStats for this key (missing means empty row)
        file_order: Ordered list of file paths to include

    Returns:
        Formatted table string
    """
    if not file_order:
        return f"Timer Key: {key}\nNo data available\n"

    lines = []

    # Title
    lines.append(f"Timer Key: {key}")

    # Calculate key column width from file basenames
    def file_display_name(fp: str) -> str:
        name = os.path.basename(fp)
        for suffix in ['.timers.json', '.json']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        return truncate_timer_key(name, max_length=50)

    display_names = {fp: file_display_name(fp) for fp in file_order}
    max_name_len = max(len(n) for n in display_names.values()) if display_names else 0
    key_width = max(len("Timer Key"), max_name_len)

    # Row width: key + space + Count(8) + space + 9 numeric columns (14 each) + 8 spaces + " | " separator
    row_width = key_width + 1 + 8 + 1 + 14 * 9 + 8 + 2

    lines.append("=" * row_width)

    # Header
    header = (
        f"{'Timer Key':<{key_width}} "
        f"{'Count':>8} "
        f"{'Mean':>14} "
        f"{'Median':>14} "
        f"{'Max':>14} | "
        f"{'P90':>14} "
        f"{'P95':>14} "
        f"{'Min':>14} "
        f"{'Total':>14} "
        f"{'Std Dev':>14}"
    )
    lines.append(header)
    lines.append("-" * row_width)

    # Data rows
    row_idx = 0
    total_count = 0
    total_time = 0.0

    for fp in file_order:
        display_name = display_names[fp]
        s = stats_by_file.get(fp)

        if s is None:
            # Empty row for file without data (incomplete run)
            row = (
                f"{display_name:<{key_width}} "
                f"{'':>8} "
                f"{'':>14} "
                f"{'':>14} "
                f"{'':>14} | "
                f"{'':>14} "
                f"{'':>14} "
                f"{'':>14} "
                f"{'':>14} "
                f"{'':>14}"
            )
        else:
            row = (
                f"{display_name:<{key_width}} "
                f"{s.count:>8} "
                f"{format_time(s.mean, use_commas=True):>14} "
                f"{format_time(s.median, use_commas=True):>14} "
                f"{format_time(s.max, use_commas=True):>14} | "
                f"{format_time(s.p90, use_commas=True):>14} "
                f"{format_time(s.p95, use_commas=True):>14} "
                f"{format_time(s.min, use_commas=True):>14} "
                f"{format_time(s.total, use_commas=True):>14} "
                f"{format_time(s.std, use_commas=True):>14}"
            )
            total_count += s.count
            total_time += s.total

        if row_idx % 2 == 0:
            row = termcolor.colored(row, "yellow")
        else:
            row = termcolor.colored(row, "white")
        lines.append(row)
        row_idx += 1

    # Footer
    lines.append("-" * row_width)
    mean_time = total_time / total_count if total_count > 0 else 0
    footer = (
        f"{'TOTAL':<{key_width}} "
        f"{total_count:>8} "
        f"{format_time(mean_time, use_commas=True):>14} "
        f"{'':>14} "  # Median
        f"{'':>14} | "  # Max
        f"{'':>14} "  # P90
        f"{'':>14} "  # P95
        f"{'':>14} "  # Min
        f"{format_time(total_time, use_commas=True):>14}"  # Total
    )
    lines.append(footer)

    return "\n".join(lines)


def format_json_single(stats: list[TimerStats], timings: list[dict]) -> str:
    """
    Format statistics as JSON for a single file.

    Args:
        stats: List of TimerStats objects
        timings: Original timing records for summary info

    Returns:
        JSON string
    """
    total_time = sum(s.total for s in stats)

    result = {
        "summary": {
            "total_records": len(timings),
            "total_time": total_time,
            "unique_timers": len(stats),
        },
        "timers": [asdict(s) for s in stats],
    }

    return json.dumps(result, indent=2)


def format_json_multi(
    timings_by_file: dict[str, list[dict]],
    stats_by_file: dict[str, list[TimerStats]],
    combined_stats: list[TimerStats],
) -> str:
    """
    Format statistics as JSON for multiple files.

    Args:
        timings_by_file: Dict mapping file_path -> timings
        stats_by_file: Dict mapping file_path -> stats
        combined_stats: Combined statistics

    Returns:
        JSON string
    """
    files_data = []
    for file_path in timings_by_file.keys():
        timings = timings_by_file[file_path]
        stats = stats_by_file[file_path]
        total_time = sum(s.total for s in stats)

        files_data.append(
            {
                "file": file_path,
                "summary": {
                    "total_records": len(timings),
                    "total_time": total_time,
                    "unique_timers": len(stats),
                },
                "timers": [asdict(s) for s in stats],
            }
        )

    combined_total = sum(s.total for s in combined_stats)
    all_timings = combine_timings(timings_by_file)

    result = {
        "files": files_data,
        "combined": {
            "summary": {
                "total_files": len(timings_by_file),
                "total_records": len(all_timings),
                "total_time": combined_total,
                "unique_timers": len(combined_stats),
            },
            "timers": [asdict(s) for s in combined_stats],
        },
    }

    return json.dumps(result, indent=2)


def format_csv_single(stats: list[TimerStats]) -> str:
    """
    Format statistics as CSV for a single file.

    Args:
        stats: List of TimerStats objects

    Returns:
        CSV string
    """
    lines = ["key,count,total,mean,median,min,max,std,p90,p95"]

    for s in stats:
        line = f"{s.key},{s.count},{s.total},{s.mean},{s.median},{s.min},{s.max},{s.std},{s.p90},{s.p95}"
        lines.append(line)

    return "\n".join(lines)


def format_csv_multi(
    timings_by_file: dict[str, list[dict]],
    stats_by_file: dict[str, list[TimerStats]],
    combined_stats: list[TimerStats],
) -> str:
    """
    Format statistics as CSV for multiple files.

    Args:
        timings_by_file: Dict mapping file_path -> timings
        stats_by_file: Dict mapping file_path -> stats
        combined_stats: Combined statistics

    Returns:
        CSV string
    """
    lines = ["file,key,count,total,mean,median,min,max,std,p90,p95"]

    # Per-file stats
    for file_path, stats in stats_by_file.items():
        for s in stats:
            line = f"{file_path},{s.key},{s.count},{s.total},{s.mean},{s.median},{s.min},{s.max},{s.std},{s.p90},{s.p95}"
            lines.append(line)

    # Combined stats
    for s in combined_stats:
        line = f"COMBINED,{s.key},{s.count},{s.total},{s.mean},{s.median},{s.min},{s.max},{s.std},{s.p90},{s.p95}"
        lines.append(line)

    return "\n".join(lines)


def process_single_file(
    file_path: str, args
) -> tuple[list[dict], list[TimerStats], dict[str, list[float]]]:
    """
    Process a single timing file.

    Args:
        file_path: Path to the timing file
        args: Command line arguments

    Returns:
        Tuple of (timings, stats, grouped_timings)
    """
    timings = load_timings(file_path)
    if not timings:
        return [], [], {}

    # Filter by keys if specified
    if args.keys:
        keys_set = set(args.keys)
        timings = [t for t in timings if t["key"] in keys_set]

    grouped = group_by_key(timings)
    grouped = compute_synthetic_metrics(grouped)
    stats = build_stats_table(timings)

    # Sort stats
    if args.sort_by == "count":
        stats.sort(key=lambda s: s.count, reverse=True)
    elif args.sort_by == "mean":
        stats.sort(key=lambda s: s.mean, reverse=True)
    elif args.sort_by == "max":
        stats.sort(key=lambda s: s.max, reverse=True)
    elif args.sort_by == "key":
        stats.sort(key=lambda s: s.key, reverse=False)
    else:  # total (default)
        stats.sort(key=lambda s: s.total, reverse=True)

    # Apply top N filter
    if args.top is not None and args.top > 0:
        stats = stats[: args.top]

    return timings, stats, grouped


def process_multiple_files(file_paths: list[str], args):
    """
    Process multiple timing files and display results.

    Args:
        file_paths: List of file paths
        args: Command line arguments
    """
    # Load all files (unfiltered)
    all_timings_by_file = load_multiple_timings(file_paths)

    if not all_timings_by_file:
        print("Error: No valid timing files found", file=sys.stderr)
        sys.exit(1)

    # Check cli:main_exit BEFORE filtering by keys
    files_without_main_exit = set()
    for file_path, timings in all_timings_by_file.items():
        cli_main_exit = [t for t in timings if t["key"] == "cli:main_exit"]
        if len(cli_main_exit) > 1:
            print(
                f"Warning: Multiple cli:main_exit keys found in {file_path}",
                file=sys.stderr,
            )
        elif len(cli_main_exit) == 0:
            files_without_main_exit.add(file_path)

    # Filter by keys if specified
    if args.keys:
        keys_set = set(args.keys)
        timings_by_file = {
            fp: [t for t in timings if t["key"] in keys_set]
            for fp, timings in all_timings_by_file.items()
        }
        # Keep files that have matching timings OR are missing cli:main_exit (shown as empty rows)
        timings_by_file = {
            fp: t for fp, t in timings_by_file.items()
            if t or fp in files_without_main_exit
        }
    else:
        timings_by_file = all_timings_by_file

    # Build stats and grouped timings for each file
    stats_by_file = {}
    grouped_by_file = {}
    for file_path, timings in timings_by_file.items():
        grouped = group_by_key(timings)
        grouped = compute_synthetic_metrics(grouped)
        grouped_by_file[file_path] = grouped
        stats = build_stats_table(timings)
        stats_by_file[file_path] = stats

    # Build combined stats (skip synthetic metrics - we'll collect per-file values)
    combined_timings = combine_timings(timings_by_file)
    combined_grouped = group_by_key(combined_timings)
    # For synthetic metrics, collect per-file values instead of computing from combined totals
    combined_grouped = collect_synthetic_metrics(grouped_by_file, combined_grouped)
    combined_stats = build_stats_table(combined_timings, compute_synthetic=False)
    # Add synthetic metrics to combined stats from per-file values
    for key in SYNTHETIC_METRIC_KEYS:
        if key in combined_grouped:
            stats = calculate_stats(combined_grouped[key])
            combined_stats.append(TimerStats(key=key, **stats))

    # Output based on format
    if args.format == "table":
        if args.keys:
            # Consolidated per-key tables: one table per key with files as rows
            for key in args.keys:
                # Build per-file stats for this specific key
                key_stats_by_file = {}
                for file_path in timings_by_file:
                    key_timings = [t for t in timings_by_file[file_path] if t["key"] == key]
                    if key_timings:
                        durations = [float(t["duration"]) for t in key_timings]
                        key_stats = calculate_stats(durations)
                        key_stats_by_file[file_path] = TimerStats(key=key, **key_stats)

                # Include files that have data OR are missing cli:main_exit (empty rows)
                file_order = sorted([
                    fp for fp in timings_by_file
                    if fp in key_stats_by_file or fp in files_without_main_exit
                ])

                print(format_per_key_table(key, key_stats_by_file, file_order))
                print()
                print()

            # Print combined table
            title = f"COMBINED Timer Statistics ({len(timings_by_file)} files) (all times in ms)"
            print(
                format_table(
                    combined_stats,
                    args.sort_by,
                    title,
                    args.top,
                    grouped_timings=combined_grouped,
                    show_histograms=args.histograms,
                    clip_percentile=args.clip,
                )
            )
        else:
            # Original per-file tables (when no specific keys selected)
            for file_path in timings_by_file.keys():
                stats = stats_by_file[file_path]
                grouped = grouped_by_file[file_path]
                title = f"Timer Statistics: {file_path} (all times in ms)"
                print(
                    format_table(
                        stats,
                        args.sort_by,
                        title,
                        args.top,
                        grouped_timings=grouped,
                        show_histograms=args.histograms,
                        clip_percentile=args.clip,
                    )
                )
                print()  # Blank line between tables
                print()

            # Print combined table
            title = f"COMBINED Timer Statistics ({len(timings_by_file)} files) (all times in ms)"
            print(
                format_table(
                    combined_stats,
                    args.sort_by,
                    title,
                    args.top,
                    grouped_timings=combined_grouped,
                    show_histograms=args.histograms,
                    clip_percentile=args.clip,
                )
            )

    elif args.format == "json":
        output = format_json_multi(timings_by_file, stats_by_file, combined_stats)
        print(output)

    elif args.format == "csv":
        output = format_csv_multi(timings_by_file, stats_by_file, combined_stats)
        print(output)

    # Show warnings for files without cli:main_exit (only in non-keys mode,
    # since keys mode shows them as empty rows in the per-key tables)
    if not args.keys and files_without_main_exit:
        print()
        print("=" * 60)
        print("WARNINGS")
        print("=" * 60)
        for file_path in sorted(files_without_main_exit):
            print(
                f"Warning: No cli:main_exit key found in {file_path}",
                file=sys.stderr,
            )
        print("=" * 60)
        print()


def generate_plots(grouped_timings: dict[str, list[float]], args) -> None:
    """
    Generate histogram and scatter plot PDFs based on CLI arguments.

    Args:
        grouped_timings: Dict mapping timer key -> list of durations
        args: Command line arguments with histplot, scatterplot, output_dir
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate histograms
    clip_percentile = getattr(args, 'clip', None)
    if args.histplot:
        for key in args.histplot:
            if key not in grouped_timings:
                print(f"Warning: Key '{key}' not found in timing data", file=sys.stderr)
                continue
            durations = grouped_timings[key]
            if len(durations) < 2:
                print(f"Warning: Key '{key}' has only {len(durations)} data point(s), skipping histogram", file=sys.stderr)
                continue
            filepath = create_histogram_pdf(durations, key, output_dir, clip_percentile=clip_percentile)
            print(f"Created histogram: {filepath}")

    # Generate scatter plots
    if args.scatterplot:
        print("Note: Scatter plots pair points by index. Only meaningful if keys are recorded in lockstep.", file=sys.stderr)
        for spec in args.scatterplot:
            if '%' not in spec:
                print(f"Warning: Invalid scatterplot format '{spec}' (expected key1%key2)", file=sys.stderr)
                continue
            parts = spec.split('%', 1)
            if len(parts) != 2:
                print(f"Warning: Invalid scatterplot format '{spec}' (expected key1%key2)", file=sys.stderr)
                continue
            key_x, key_y = parts

            if key_x not in grouped_timings:
                print(f"Warning: Key '{key_x}' not found in timing data", file=sys.stderr)
                continue
            if key_y not in grouped_timings:
                print(f"Warning: Key '{key_y}' not found in timing data", file=sys.stderr)
                continue

            durations_x = grouped_timings[key_x]
            durations_y = grouped_timings[key_y]

            if len(durations_x) != len(durations_y):
                print(f"Warning: Keys '{key_x}' ({len(durations_x)} points) and '{key_y}' ({len(durations_y)} points) have different counts", file=sys.stderr)
                # Use the minimum length
                min_len = min(len(durations_x), len(durations_y))
                durations_x = durations_x[:min_len]
                durations_y = durations_y[:min_len]
                print(f"  Using first {min_len} points from each", file=sys.stderr)

            if len(durations_x) < 2:
                print(f"Warning: Not enough data points for scatter plot, skipping", file=sys.stderr)
                continue

            filepath = create_scatterplot_pdf(durations_x, durations_y, key_x, key_y, output_dir)
            print(f"Created scatter plot: {filepath}")


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Analyze timing data from flowbook-times.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default file (flowbook-times.json)
  flowbook_timers

  # Analyze specific file
  flowbook_timers timings.json

  # Analyze multiple files
  flowbook_timers run1.json run2.json run3.json

  # Remote files (cached after first download)
  flowbook_timers user@server:/data/runs/*.json

  # Force re-download remote files
  flowbook_timers user@server:/data/*.json --force-download

  # Clear all cached remote files
  flowbook_timers --clear-cache

  # Sort by mean time, show top 10
  flowbook_timers --sort-by mean --top 10

  # Show only specific timer keys
  flowbook_timers --keys diff.compute checkpoint.create

  # Show histograms for top 5 timers
  flowbook_timers --top 5 --histograms

  # Create histogram PDF for a specific timer
  flowbook_timers timings.json --histplot=diff:compute

  # Create histogram with top 1% outliers clipped
  flowbook_timers timings.json --histplot=diff:compute --clip 99

  # Create scatter plot PDF comparing two timers
  flowbook_timers timings.json --scatterplot=diff:compute%checkpoint:create

  # Output as JSON
  flowbook_timers --format json
        """,
    )

    parser.add_argument(
        "files",
        nargs="*",
        default=["flowbook-times.json"],
        help="Timing JSON files to analyze (local or remote, supports wildcards)",
    )

    parser.add_argument(
        "--sort-by",
        choices=["total", "mean", "count", "max", "key"],
        default="key",
        help="Sort timers by field (default: key)",
    )

    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument("--top", type=int, metavar="N", help="Show only top N timers")

    parser.add_argument(
        "--keys", nargs="+", metavar="KEY", help="Only show these specific timer keys"
    )

    parser.add_argument(
        "--histograms",
        action="store_true",
        help="Show ASCII histograms for each timer (table mode only)",
    )

    # Remote file options
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of remote files even if cached",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached remote files and exit",
    )

    # Plot options
    parser.add_argument(
        "--histplot",
        action="append",
        metavar="KEY",
        help="Create histogram PDF for specified timer key (can be used multiple times)",
    )

    parser.add_argument(
        "--scatterplot",
        action="append",
        metavar="KEY%KEY",
        help="Create scatter plot PDF for two keys (format: key1%%key2, can be used multiple times)",
    )

    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for plot output files (default: current directory)",
    )

    parser.add_argument(
        "--clip",
        type=float,
        metavar="PERCENTILE",
        help="Clip histogram values above this percentile (e.g., 99). Reports how many values were clipped.",
    )

    args = parser.parse_args()

    # Handle --clear-cache
    if args.clear_cache:
        clear_cache()
        sys.exit(0)

    # Resolve file paths (download remote files if needed)
    resolved_files = resolve_file_paths(args.files, args.force_download)

    if not resolved_files:
        print("Error: No files found", file=sys.stderr)
        sys.exit(1)

    # Update args.files with resolved paths
    args.files = resolved_files

    # Single or multi-file mode?
    if len(args.files) == 1:
        # Single file mode
        timings, stats, grouped = process_single_file(args.files[0], args)

        if not stats:
            print("No timing data available", file=sys.stderr)
            sys.exit(1)

        if args.format == "table":
            print(
                format_table(
                    stats,
                    args.sort_by,
                    top=args.top,
                    grouped_timings=grouped,
                    show_histograms=args.histograms,
                    clip_percentile=args.clip,
                )
            )
        elif args.format == "json":
            print(format_json_single(stats, timings))
        elif args.format == "csv":
            print(format_csv_single(stats))

        # Generate plots if requested
        if args.histplot or args.scatterplot:
            generate_plots(grouped, args)
    else:
        # Multi-file mode
        process_multiple_files(args.files, args)

        # For plots in multi-file mode, we need combined timings
        if args.histplot or args.scatterplot:
            # Reload and combine timings for plotting
            timings_by_file = load_multiple_timings(args.files)
            # Compute per-file synthetic metrics first
            grouped_by_file = {}
            for file_path, timings in timings_by_file.items():
                grouped = group_by_key(timings)
                grouped = compute_synthetic_metrics(grouped)
                grouped_by_file[file_path] = grouped
            # Combine and collect per-file synthetic metrics
            combined_timings = combine_timings(timings_by_file)
            combined_grouped = group_by_key(combined_timings)
            combined_grouped = collect_synthetic_metrics(grouped_by_file, combined_grouped)
            generate_plots(combined_grouped, args)


if __name__ == "__main__":
    main()
