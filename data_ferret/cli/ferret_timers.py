#!/usr/bin/env python3
"""
Command line tool to analyze timing data from ferret-times.json files.

Reads timing data and displays statistics for each timer type including:
- Count, Total, Mean, Median, Min, Max, Std Dev, P95

Supports single or multiple files with combined statistics.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import termcolor


def create_ascii_histogram(data, bins=20, width=80, bar_char='#'):
    """
    Generates an ASCII histogram for the given data.
    """
    if not data:
        return ""

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
            bin_index = bins - 1 # Handle the edge case of max value
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
        chart.append(f"{bin_start:7.3f} - {bin_end:7.3f} | {bar_char * bar_length} ({percent:.1f}%)")

    return "\n".join(chart)


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
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {file_path} does not contain a list", file=sys.stderr)
            return []

        # Validate structure
        valid_records = []
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                print(f"Warning: Record {i} in {file_path} is not a dict", file=sys.stderr)
                continue
            if 'key' not in record or 'duration' not in record:
                print(f"Warning: Record {i} in {file_path} missing 'key' or 'duration'", file=sys.stderr)
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
        key = record['key']
        duration = float(record['duration'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(duration)
    return grouped


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
            'count': 0,
            'total': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0,
            'p90': 0.0,
            'p95': 0.0,
        }

    arr = np.array(durations)
    return {
        'count': len(durations),
        'total': float(np.sum(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'std': float(np.std(arr)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
    }


def build_stats_table(timings: list[dict]) -> list[TimerStats]:
    """
    Build statistics table from timing data.

    Args:
        timings: List of timing records

    Returns:
        List of TimerStats objects
    """
    if not timings:
        return []

    grouped = group_by_key(timings)
    stats_list = []

    for key, durations in grouped.items():
        stats = calculate_stats(durations)
        stats_list.append(TimerStats(
            key=key,
            **stats
        ))

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


def format_table(stats: list[TimerStats], sort_by: str = 'total',
                 title: Optional[str] = None, top: Optional[int] = None,
                 grouped_timings: Optional[dict[str, list[float]]] = None,
                 show_histograms: bool = False) -> str:
    """
    Format statistics as a table.

    Args:
        stats: List of TimerStats objects
        sort_by: Field to sort by ('total', 'mean', 'count', 'max', 'key')
        title: Optional title for the table
        top: Optional limit to show only top N timers
        grouped_timings: Optional dict mapping key -> list of durations (for histograms)
        show_histograms: Whether to show histograms for each key (requires grouped_timings)

    Returns:
        Formatted table string
    """
    if not stats:
        return "No timing data available\n"

    # Parse GROUP:KEY format and group stats
    from collections import defaultdict

    def parse_key(key: str) -> tuple[str, str]:
        """Parse timer key into (group, subkey)."""
        if ':' in key:
            group, subkey = key.split(':', 1)
            return group, subkey
        return '', key  # No group

    # Group stats by group prefix
    grouped: dict[str, list[TimerStats]] = defaultdict(list)
    for s in stats:
        group, _ = parse_key(s.key)
        grouped[group].append(s)

    # Sort groups: empty group first, then alphabetically
    group_order = sorted(grouped.keys(), key=lambda g: (g != '', g))

    # Sort within each group by the specified field
    def sort_key(s: TimerStats):
        if sort_by == 'count':
            return -s.count
        elif sort_by == 'mean':
            return -s.mean
        elif sort_by == 'max':
            return -s.max
        elif sort_by == 'key':
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
        group_order = sorted(grouped.keys(), key=lambda g: (g != '', g))
        for group in grouped:
            grouped[group] = sorted(grouped[group], key=sort_key)

    # Build table
    lines = []

    # Calculate key column width - use subkey length + 4 for indent, or full key for ungrouped
    max_subkey_len = 0
    for s in stats:
        group, subkey = parse_key(s.key)
        if group:
            max_subkey_len = max(max_subkey_len, len(subkey) + 4)  # 4 spaces indent
        else:
            max_subkey_len = max(max_subkey_len, len(s.key))
    # Also consider group names as headers
    for group in grouped:
        if group:
            max_subkey_len = max(max_subkey_len, len(group))
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
            # Empty stats row with just the group name
            group_header = f"{group:<{key_width}} {'':>8} {'':>14} {'':>14} {'':>14} | {'':>14} {'':>14} {'':>14} {'':>14} {'':>14}"
            lines.append(termcolor.colored(group_header, 'cyan', attrs=['bold']))

        for s in group_stats:
            group_name, subkey = parse_key(s.key)
            # Indent subkeys if they have a group
            display_key = f"    {subkey}" if group_name else s.key

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
                row = termcolor.colored(row, 'yellow')
            else:
                row = termcolor.colored(row, 'white')
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
        for s in sorted_stats:
            if s.key in grouped_timings:
                durations = grouped_timings[s.key]
                if len(durations) > 1:  # Need at least 2 points for a histogram
                    lines.append("")
                    lines.append(f"--- {s.key} (n={len(durations)}) ---")
                    hist = create_ascii_histogram(durations, bar_char='█')
                    lines.append(hist)

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
        "timers": [asdict(s) for s in stats]
    }

    return json.dumps(result, indent=2)


def format_json_multi(timings_by_file: dict[str, list[dict]],
                      stats_by_file: dict[str, list[TimerStats]],
                      combined_stats: list[TimerStats]) -> str:
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

        files_data.append({
            "file": file_path,
            "summary": {
                "total_records": len(timings),
                "total_time": total_time,
                "unique_timers": len(stats),
            },
            "timers": [asdict(s) for s in stats]
        })

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
            "timers": [asdict(s) for s in combined_stats]
        }
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


def format_csv_multi(timings_by_file: dict[str, list[dict]],
                     stats_by_file: dict[str, list[TimerStats]],
                     combined_stats: list[TimerStats]) -> str:
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


def process_single_file(file_path: str, args) -> tuple[list[dict], list[TimerStats], dict[str, list[float]]]:
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
        timings = [t for t in timings if t['key'] in keys_set]

    grouped = group_by_key(timings)
    stats = build_stats_table(timings)

    # Sort stats
    if args.sort_by == 'count':
        stats.sort(key=lambda s: s.count, reverse=True)
    elif args.sort_by == 'mean':
        stats.sort(key=lambda s: s.mean, reverse=True)
    elif args.sort_by == 'max':
        stats.sort(key=lambda s: s.max, reverse=True)
    elif args.sort_by == 'key':
        stats.sort(key=lambda s: s.key, reverse=False)
    else:  # total (default)
        stats.sort(key=lambda s: s.total, reverse=True)

    # Apply top N filter
    if args.top is not None and args.top > 0:
        stats = stats[:args.top]

    return timings, stats, grouped


def process_multiple_files(file_paths: list[str], args):
    """
    Process multiple timing files and display results.

    Args:
        file_paths: List of file paths
        args: Command line arguments
    """
    # Load all files
    timings_by_file = load_multiple_timings(file_paths)

    if not timings_by_file:
        print("Error: No valid timing files found", file=sys.stderr)
        sys.exit(1)

    # Filter by keys if specified
    if args.keys:
        keys_set = set(args.keys)
        timings_by_file = {
            fp: [t for t in timings if t['key'] in keys_set]
            for fp, timings in timings_by_file.items()
        }
        # Remove files with no matching timings
        timings_by_file = {fp: t for fp, t in timings_by_file.items() if t}

    # Build stats and grouped timings for each file
    stats_by_file = {}
    grouped_by_file = {}
    for file_path, timings in timings_by_file.items():
        grouped_by_file[file_path] = group_by_key(timings)
        stats = build_stats_table(timings)
        stats_by_file[file_path] = stats

    # Build combined stats
    combined_timings = combine_timings(timings_by_file)
    combined_grouped = group_by_key(combined_timings)
    combined_stats = build_stats_table(combined_timings)

    # Output based on format
    if args.format == 'table':
        # Print table for each file
        for file_path in timings_by_file.keys():
            stats = stats_by_file[file_path]
            grouped = grouped_by_file[file_path]
            title = f"Timer Statistics: {file_path} (all times in ms)"
            print(format_table(
                stats, args.sort_by, title, args.top,
                grouped_timings=grouped, show_histograms=args.histograms
            ))
            print()  # Blank line between tables
            print()

        # Print combined table
        title = f"COMBINED Timer Statistics ({len(timings_by_file)} files) (all times in ms)"
        print(format_table(
            combined_stats, args.sort_by, title, args.top,
            grouped_timings=combined_grouped, show_histograms=args.histograms
        ))

    elif args.format == 'json':
        output = format_json_multi(timings_by_file, stats_by_file, combined_stats)
        print(output)

    elif args.format == 'csv':
        output = format_csv_multi(timings_by_file, stats_by_file, combined_stats)
        print(output)


    print()
    first_error = True
    for file_path, valid_records in timings_by_file.items():
        # Look for a cli_main_exit key
        cli_main_exit = [t for t in valid_records if t['key'] == 'cli_main_exit']
        if len(cli_main_exit) > 1:
            if first_error:
                print("=" * 60)
                print("WARNINGS")
                print("=" * 60)
            print(f"Warning: Multiple cli_main_exit keys found in {file_path}", file=sys.stderr)
            print(f"Warning: {len(cli_main_exit)} cli_main_exit keys found in {file_path}", file=sys.stderr)
            print(f"Warning: {cli_main_exit}", file=sys.stderr)
            first_error = False
        elif len(cli_main_exit) == 0:
            if first_error:
                print("=" * 60)
                print("WARNINGS")
                print("=" * 60)
            print(f"Warning: No cli_main_exit key found in {file_path}", file=sys.stderr)
            first_error = False
    if not first_error:
        print("=" * 60)
        print()


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description='Analyze timing data from ferret-times.json files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default file (ferret-times.json)
  ferret_timers

  # Analyze specific file
  ferret_timers timings.json

  # Analyze multiple files
  ferret_timers run1.json run2.json run3.json

  # Sort by mean time, show top 10
  ferret_timers --sort-by mean --top 10

  # Show only specific timer keys
  ferret_timers --keys diff.compute checkpoint.create

  # Show histograms for top 5 timers
  ferret_timers --top 5 --histograms

  # Output as JSON
  ferret_timers --format json
        """
    )

    parser.add_argument(
        'files',
        nargs='*',
        default=['ferret-times.json'],
        help='Timing JSON files to analyze (default: ferret-times.json)'
    )

    parser.add_argument(
        '--sort-by',
        choices=['total', 'mean', 'count', 'max', 'key'],
        default='key',
        help='Sort timers by field (default: key)'
    )

    parser.add_argument(
        '--format',
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format (default: table)'
    )

    parser.add_argument(
        '--top',
        type=int,
        metavar='N',
        help='Show only top N timers'
    )

    parser.add_argument(
        '--keys',
        nargs='+',
        metavar='KEY',
        help='Only show these specific timer keys'
    )

    parser.add_argument(
        '--histograms',
        action='store_true',
        help='Show ASCII histograms for each timer (table mode only)'
    )

    args = parser.parse_args()

    # Single or multi-file mode?
    if len(args.files) == 1:
        # Single file mode
        timings, stats, grouped = process_single_file(args.files[0], args)

        if not stats:
            print("No timing data available", file=sys.stderr)
            sys.exit(1)

        if args.format == 'table':
            print(format_table(
                stats, args.sort_by, top=args.top,
                grouped_timings=grouped, show_histograms=args.histograms
            ))
        elif args.format == 'json':
            print(format_json_single(stats, timings))
        elif args.format == 'csv':
            print(format_csv_single(stats))
    else:
        # Multi-file mode
        process_multiple_files(args.files, args)


if __name__ == '__main__':
    main()
