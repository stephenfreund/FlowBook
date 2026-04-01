#!/usr/bin/env python3
"""
Compare overhead between original and fixed notebooks.

Takes two directories of _comparison.json files and uses permutation tests
to determine if there's a statistically significant difference in running times.

Supports multiple runs per benchmark: X_comparison.json, X_comparison-1.json, etc.
Per-benchmark medians and means are computed, then paired permutation tests
are performed across benchmarks (the independent unit of analysis).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def load_comparison_json(path: Path) -> Optional[dict]:
    """Load a comparison JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
        return None


def extract_times(data: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract baseline and flowbook execution times from comparison data.

    Returns (baseline_ms, flowbook_ms), either may be None if not available.
    """
    baseline_ms = None
    flowbook_ms = None

    kernels = data.get("kernels", {})

    baseline = kernels.get("baseline", {})
    if baseline and baseline.get("timing"):
        totals = baseline["timing"].get("totals", {})
        baseline_ms = totals.get("execute_duration_ms")

    flowbook = kernels.get("flowbook", {})
    if flowbook and flowbook.get("timing"):
        totals = flowbook["timing"].get("totals", {})
        flowbook_ms = totals.get("execute_duration_ms")

    return baseline_ms, flowbook_ms


# IPython traceback prefix for detecting cell errors
TRACEBACK_PREFIX = "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"


def extract_errors(data: dict) -> tuple[bool, bool]:
    """Extract error status for baseline and flowbook.

    Returns (baseline_has_errors, flowbook_has_errors).
    Detects errors by looking for IPython traceback prefix in cell error field.
    """
    kernels = data.get("kernels", {})

    def has_traceback_errors(timing: dict) -> bool:
        """Check if any cell has a traceback error."""
        for cell in timing.get("cells", []):
            error = cell.get("error")
            if error and isinstance(error, str) and error.startswith(TRACEBACK_PREFIX):
                return True
        return False

    baseline_errors = False
    baseline = kernels.get("baseline", {})
    if baseline and baseline.get("timing"):
        baseline_errors = has_traceback_errors(baseline["timing"])

    flowbook_errors = False
    flowbook = kernels.get("flowbook", {})
    if flowbook and flowbook.get("timing"):
        flowbook_errors = has_traceback_errors(flowbook["timing"])

    return baseline_errors, flowbook_errors


def find_matching_groups(orig_dir: Path, fixed_dir: Path) -> list[tuple[str, list[Path], list[Path]]]:
    """Find matching groups of original and fixed comparison files.

    For each benchmark X, collects:
    - Orig: X_comparison.json, X_comparison-1.json, X_comparison-2.json, ...
    - Fixed: X-fixed_comparison.json, X-fixed_comparison-1.json, ...

    Returns list of (benchmark_name, orig_paths, fixed_paths).
    """
    orig_pattern = re.compile(r'^(.+)_comparison(?:-(\d+))?\.json$')
    fixed_pattern = re.compile(r'^(.+)-fixed_comparison(?:-(\d+))?\.json$')

    orig_groups: dict[str, list[Path]] = {}
    for f in sorted(orig_dir.glob("*_comparison*.json")):
        if "-fixed_comparison" in f.name:
            continue
        m = orig_pattern.match(f.name)
        if m:
            name = m.group(1)
            orig_groups.setdefault(name, []).append(f)

    fixed_groups: dict[str, list[Path]] = {}
    for f in sorted(fixed_dir.glob("*-fixed_comparison*.json")):
        m = fixed_pattern.match(f.name)
        if m:
            name = m.group(1)
            fixed_groups.setdefault(name, []).append(f)

    groups = []
    for name in sorted(orig_groups):
        if name in fixed_groups:
            groups.append((name, orig_groups[name], fixed_groups[name]))
        else:
            print(f"Warning: No fixed files found for {name}", file=sys.stderr)

    return groups


def load_run_data(paths: list[Path]) -> list[dict]:
    """Load timing data from multiple comparison files.

    Returns list of per-run dicts with keys: baseline, flowbook, base_err, flow_err.
    """
    runs = []
    for path in paths:
        data = load_comparison_json(path)
        if data is None:
            continue
        baseline_ms, flowbook_ms = extract_times(data)
        base_err, flow_err = extract_errors(data)
        runs.append({
            "baseline": baseline_ms,
            "flowbook": flowbook_ms,
            "base_err": base_err,
            "flow_err": flow_err,
        })
    return runs


def summarize_values(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"median": None, "mean": None, "std": None, "n": 0}
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    return {
        "median": float(np.median(arr)),
        "mean": mean_val,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }


def build_benchmark_row(name: str, orig_runs: list[dict], fixed_runs: list[dict]) -> dict:
    """Build a data row for one benchmark from its runs.

    Excludes individual runs with baseline errors from time aggregation.
    """
    def collect_times(runs):
        baseline_times = []
        flowbook_times = []
        for r in runs:
            if r["base_err"]:
                continue
            if r["baseline"] is not None:
                baseline_times.append(r["baseline"])
            if r["flowbook"] is not None:
                flowbook_times.append(r["flowbook"])
        return baseline_times, flowbook_times

    ob_times, of_times = collect_times(orig_runs)
    fb_times, ff_times = collect_times(fixed_runs)

    return {
        "name": name,
        "orig_baseline": summarize_values(ob_times),
        "orig_flowbook": summarize_values(of_times),
        "fixed_baseline": summarize_values(fb_times),
        "fixed_flowbook": summarize_values(ff_times),
        "orig_n": len(orig_runs),
        "fixed_n": len(fixed_runs),
        "orig_base_err": sum(1 for r in orig_runs if r["base_err"]),
        "orig_flow_err": sum(1 for r in orig_runs if r["flow_err"]),
        "fixed_base_err": sum(1 for r in fixed_runs if r["base_err"]),
        "fixed_flow_err": sum(1 for r in fixed_runs if r["flow_err"]),
    }


def paired_permutation_test_geometric(
    orig: np.ndarray,
    fixed: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42
) -> tuple[float, float, float, float]:
    """Paired permutation test using geometric mean of ratios.

    Uses log-ratios internally for symmetric treatment of speedup/slowdown.
    Tests whether geometric mean of (fixed/orig) differs from 1.0.

    Returns:
        (geom_mean_ratio, pct_change, p_value, effect_size)
    """
    rng = np.random.default_rng(seed)

    ratios = fixed / orig
    log_ratios = np.log(ratios)
    observed_mean = np.mean(log_ratios)

    # Paired permutation: randomly flip sign of each log-ratio
    n = len(log_ratios)
    perm_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        perm_means[i] = np.mean(signs * log_ratios)

    p_value = np.mean(np.abs(perm_means) >= np.abs(observed_mean))

    std_log = np.std(log_ratios, ddof=1)
    effect_size = observed_mean / std_log if std_log > 0 else 0.0

    geom_mean_ratio = np.exp(observed_mean)
    pct_change = (1 - geom_mean_ratio) * 100

    return geom_mean_ratio, pct_change, p_value, effect_size


def paired_permutation_test_arithmetic(
    orig: np.ndarray,
    fixed: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42
) -> tuple[float, float, float, float]:
    """Paired permutation test using arithmetic mean of ratios.

    Tests whether arithmetic mean of (fixed/orig) differs from 1.0.

    Returns:
        (arith_mean_ratio, pct_change, p_value, effect_size)
    """
    rng = np.random.default_rng(seed)

    ratios = fixed / orig
    observed_mean = np.mean(ratios)

    # Paired permutation: randomly swap orig/fixed within each pair
    # Equivalent to using 1/ratio for some pairs
    n = len(ratios)
    perm_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        swapped = np.where(rng.choice([True, False], size=n), ratios, 1.0 / ratios)
        perm_means[i] = np.mean(swapped)

    # Test against 1.0 (no difference)
    observed_diff = observed_mean - 1.0
    perm_diffs = perm_means - 1.0
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    std_ratios = np.std(ratios, ddof=1)
    effect_size = observed_diff / std_ratios if std_ratios > 0 else 0.0

    pct_change = (1 - observed_mean) * 100

    return observed_mean, pct_change, p_value, effect_size


def truncate_name(name: str, max_width: int) -> str:
    """Truncate name with ellipsis if too long."""
    if len(name) <= max_width:
        return name
    return name[:max_width - 3] + "..."


def format_ms(value: Optional[float], width: int = 12) -> str:
    """Format milliseconds value for display."""
    if value is None:
        return "N/A".rjust(width)
    return f"{value:.2f}".rjust(width)


def format_pct(orig: Optional[float], fixed: Optional[float], width: int = 10) -> str:
    """Format percentage change ((orig - fixed) / orig * 100)."""
    if orig is None or fixed is None or orig == 0:
        return "N/A".rjust(width)
    pct = (orig - fixed) / orig * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%".rjust(width)


def format_ratio_pct(ratio: float, width: int = 8) -> str:
    """Format ratio as percentage change."""
    pct = (1 - ratio) * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%".rjust(width)


def run_comparison_tests(
    label: str,
    data_rows: list[dict],
    orig_key: str,
    fixed_key: str,
    name_width: int,
    n_permutations: int,
    alpha: float,
    direction_labels: tuple[str, str] = ("FASTER", "SLOWER"),
):
    """Run and print paired permutation tests for a comparison.

    Tests with both median and mean aggregations, reporting geometric
    and arithmetic test variants for each.
    """
    print(f"\n{'-' * 80}")
    print(label)
    print(f"{'-' * 80}")

    for agg_name in ["median", "mean"]:
        paired = [
            (r["name"], r[orig_key][agg_name], r[fixed_key][agg_name])
            for r in data_rows
            if r[orig_key][agg_name] is not None
            and r[fixed_key][agg_name] is not None
            and r[orig_key][agg_name] > 0
        ]

        if len(paired) < 2:
            print(f"\n  [{agg_name.title()} aggregation] Insufficient data (n={len(paired)})")
            continue

        orig_arr = np.array([p[1] for p in paired])
        fixed_arr = np.array([p[2] for p in paired])

        if agg_name == "median":
            print(f"\nSample size (n): {len(paired)}")
            print(f"\nPer-benchmark ratios (fixed/orig, {agg_name}), sorted by |%change|:")
            print(f"  {'Benchmark':>{name_width}}   {'Ratio':>8}  {'%Chg':>8}")
            print(f"  {'-' * name_width}   {'-' * 8}  {'-' * 8}")
            sorted_pairs = sorted(paired, key=lambda x: abs(1 - x[2] / x[1]), reverse=True)
            for pname, porig, pfixed in sorted_pairs:
                ratio = pfixed / porig
                disp = truncate_name(pname, name_width)
                print(f"  {disp:>{name_width}}   {ratio:>8.4f}  {format_ratio_pct(ratio, 8)}")

        g_ratio, g_pct, g_pval, g_effect = paired_permutation_test_geometric(
            orig_arr, fixed_arr, n_permutations
        )
        a_ratio, a_pct, a_pval, a_effect = paired_permutation_test_arithmetic(
            orig_arr, fixed_arr, n_permutations
        )

        print(f"\n  [{agg_name.title()} aggregation]")
        print(f"  {'Test Method':<20} {'Mean Ratio':>12} {'% Change':>12} {'Effect Size':>12} {'p-value':>10}")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")
        print(f"  {'Geometric mean':<20} {g_ratio:>12.4f} {g_pct:>+11.2f}% {g_effect:>12.3f} {g_pval:>10.4f}")
        print(f"  {'Arithmetic mean':<20} {a_ratio:>12.4f} {a_pct:>+11.2f}% {a_effect:>12.3f} {a_pval:>10.4f}")

        pos_label, neg_label = direction_labels
        if g_pval < alpha:
            direction = pos_label if g_pct > 0 else neg_label
            print(f"\n  >>> SIGNIFICANT ({agg_name}): Fixed is {direction} than Original (p={g_pval:.4f})")
        else:
            print(f"\n  >>> NOT SIGNIFICANT ({agg_name}): No detectable difference (p={g_pval:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare overhead between original and fixed notebooks"
    )
    parser.add_argument(
        "--orig-dir",
        type=Path,
        required=True,
        help="Directory containing original _comparison.json files"
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        required=True,
        help="Directory containing fixed (n-fixed_comparison.json) files"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical test (default: 10000)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)"
    )

    args = parser.parse_args()

    # Validate directories
    if not args.orig_dir.is_dir():
        print(f"Error: {args.orig_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.fixed_dir.is_dir():
        print(f"Error: {args.fixed_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find matching groups
    groups = find_matching_groups(args.orig_dir, args.fixed_dir)

    if not groups:
        print("No matching benchmark groups found.", file=sys.stderr)
        sys.exit(1)

    # Collect data
    data_rows = []
    for name, orig_paths, fixed_paths in groups:
        orig_runs = load_run_data(orig_paths)
        fixed_runs = load_run_data(fixed_paths)

        if not orig_runs or not fixed_runs:
            print(f"Warning: No valid runs for {name}", file=sys.stderr)
            continue

        row = build_benchmark_row(name, orig_runs, fixed_runs)
        data_rows.append(row)

    if not data_rows:
        print("No valid data extracted.", file=sys.stderr)
        sys.exit(1)

    # Filter out benchmarks where all orig runs had baseline errors
    skipped = [r for r in data_rows if r["orig_baseline"]["n"] == 0]
    data_rows = [r for r in data_rows if r["orig_baseline"]["n"] > 0]

    if skipped:
        print(f"SKIPPED {len(skipped)} benchmarks with no valid orig baseline data:")
        for r in skipped:
            print(f"  - {r['name']} ({r['orig_n']} runs, all had baseline errors)")
        print()

    if not data_rows:
        print("No valid data remaining after filtering.", file=sys.stderr)
        sys.exit(1)

    total_orig = sum(r["orig_n"] for r in data_rows)
    total_fixed = sum(r["fixed_n"] for r in data_rows)
    print(f"Analyzing {len(data_rows)} benchmarks ({total_orig} orig runs, {total_fixed} fixed runs)\n")

    # Calculate dynamic name width (max 45 chars)
    max_name_width = min(45, max(len(r["name"]) for r in data_rows))
    name_width = max(max_name_width, 9)  # At least "Benchmark" width

    # Column widths
    num_w = 12  # numeric columns
    pct_w = 8   # percentage columns
    runs_w = 7  # runs column

    table_width = name_width + runs_w + 3 + (num_w * 2 + pct_w + 3) * 2 + 1

    # Error status table
    any_errors = any(
        r["orig_base_err"] > 0 or r["orig_flow_err"] > 0 or
        r["fixed_base_err"] > 0 or r["fixed_flow_err"] > 0
        for r in data_rows
    )

    if any_errors:
        err_w = 10
        err_table_width = name_width + 4 + err_w * 4 + 3
        print("=" * err_table_width)
        print("ERROR STATUS (error runs / total runs)")
        print("=" * err_table_width)
        print(
            f"  {'Benchmark':>{name_width}} | "
            f"{'Orig Base':^{err_w}} {'Orig Flow':^{err_w}} | "
            f"{'Fixed Base':^{err_w}} {'Fixed Flow':^{err_w}}"
        )
        print(
            f"  {'-' * name_width}-+-"
            f"{'-' * err_w}-{'-' * err_w}-+-"
            f"{'-' * err_w}-{'-' * err_w}"
        )

        error_count = 0
        for row in data_rows:
            has_any = (
                row["orig_base_err"] > 0 or row["orig_flow_err"] > 0 or
                row["fixed_base_err"] > 0 or row["fixed_flow_err"] > 0
            )
            if not has_any:
                continue
            error_count += 1
            disp_name = truncate_name(row["name"], name_width)
            ob = f"{row['orig_base_err']}/{row['orig_n']}" if row["orig_base_err"] else ""
            of = f"{row['orig_flow_err']}/{row['orig_n']}" if row["orig_flow_err"] else ""
            fb = f"{row['fixed_base_err']}/{row['fixed_n']}" if row["fixed_base_err"] else ""
            ff = f"{row['fixed_flow_err']}/{row['fixed_n']}" if row["fixed_flow_err"] else ""
            print(
                f"  {disp_name:>{name_width}} | "
                f"{ob:^{err_w}} {of:^{err_w}} | "
                f"{fb:^{err_w}} {ff:^{err_w}}"
            )

        print(
            f"  {'-' * name_width}-+-"
            f"{'-' * err_w}-{'-' * err_w}-+-"
            f"{'-' * err_w}-{'-' * err_w}"
        )
        print(f"  Benchmarks with errors: {error_count} / {len(data_rows)}")
        print("=" * err_table_width)
        print()

    # Full data table (median times)
    print("=" * table_width)
    print("FULL DATA TABLE (median times in ms)")
    print("=" * table_width)

    baseline_header = "BASELINE".center(num_w * 2 + pct_w + 2)
    flowbook_header = "FLOWBOOK".center(num_w * 2 + pct_w + 2)
    print(f"{'':>{name_width}} {'':>{runs_w}}  |{baseline_header}|{flowbook_header}")

    header = (
        f"{'Benchmark':>{name_width}} {'Runs':>{runs_w}}  "
        f"{'Orig':>{num_w}} {'Fixed':>{num_w}} {'%Chg':>{pct_w}} | "
        f"{'Orig':>{num_w}} {'Fixed':>{num_w}} {'%Chg':>{pct_w}}"
    )
    print(header)
    print("-" * table_width)

    for row in data_rows:
        name = truncate_name(row["name"], name_width)
        runs_str = f"{row['orig_n']}/{row['fixed_n']}"
        ob = row["orig_baseline"]["median"]
        fb = row["fixed_baseline"]["median"]
        of_ = row["orig_flowbook"]["median"]
        ff = row["fixed_flowbook"]["median"]
        line = (
            f"{name:>{name_width}} {runs_str:>{runs_w}}  "
            f"{format_ms(ob, num_w)} {format_ms(fb, num_w)} {format_pct(ob, fb, pct_w)} | "
            f"{format_ms(of_, num_w)} {format_ms(ff, num_w)} {format_pct(of_, ff, pct_w)}"
        )
        print(line)

    print("-" * table_width)

    # Summary row
    def safe_mean(lst):
        return np.mean(lst) if lst else None

    all_ob = [r["orig_baseline"]["median"] for r in data_rows if r["orig_baseline"]["median"] is not None]
    all_fb = [r["fixed_baseline"]["median"] for r in data_rows if r["fixed_baseline"]["median"] is not None]
    all_of = [r["orig_flowbook"]["median"] for r in data_rows if r["orig_flowbook"]["median"] is not None]
    all_ff = [r["fixed_flowbook"]["median"] for r in data_rows if r["fixed_flowbook"]["median"] is not None]

    summary = (
        f"{'MEAN':>{name_width}} {'':>{runs_w}}  "
        f"{format_ms(safe_mean(all_ob), num_w)} {format_ms(safe_mean(all_fb), num_w)} "
        f"{format_pct(safe_mean(all_ob), safe_mean(all_fb), pct_w)} | "
        f"{format_ms(safe_mean(all_of), num_w)} {format_ms(safe_mean(all_ff), num_w)} "
        f"{format_pct(safe_mean(all_of), safe_mean(all_ff), pct_w)}"
    )
    print(summary)
    print("=" * table_width)

    # Within-benchmark variability warnings
    high_cv = []
    for row in data_rows:
        for key_label, key in [("orig baseline", "orig_baseline"),
                                ("fixed baseline", "fixed_baseline"),
                                ("orig flowbook", "orig_flowbook"),
                                ("fixed flowbook", "fixed_flowbook")]:
            s = row[key]
            if s["n"] > 1 and s["mean"] and s["mean"] > 0:
                cv = s["std"] / s["mean"] * 100
                if cv > 10:
                    high_cv.append((row["name"], key_label, cv, s["n"]))

    if high_cv:
        print(f"\nWARNING: High within-benchmark variability (CV > 10%):")
        for bname, klabel, cv, n in sorted(high_cv, key=lambda x: -x[2]):
            print(f"  {bname} [{klabel}]: CV={cv:.1f}% (n={n})")
        print()

    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (Paired Permutation Test on Ratios)")
    print("=" * 80)
    print("\nMethod: For each benchmark, aggregate runs via median and mean (separately),")
    print("compute ratio = fixed/orig, test whether geometric mean of ratios differs from 1.0.")
    print("Unit of analysis: benchmark (not individual runs).")
    print(f"\nPermutations: {args.permutations}")
    print(f"Significance level (alpha): {args.alpha}")

    run_comparison_tests(
        "BASELINE KERNEL: Original vs Fixed",
        data_rows, "orig_baseline", "fixed_baseline",
        name_width, args.permutations, args.alpha,
        ("FASTER", "SLOWER"),
    )

    run_comparison_tests(
        "FLOWBOOK KERNEL: Original vs Fixed",
        data_rows, "orig_flowbook", "fixed_flowbook",
        name_width, args.permutations, args.alpha,
        ("FASTER", "SLOWER"),
    )

    # Overhead ratio comparison
    print(f"\n{'-' * 80}")
    print("FLOWBOOK OVERHEAD RATIO: (Flowbook/Baseline) for Original vs Fixed")
    print(f"{'-' * 80}")

    for agg_name in ["median", "mean"]:
        full_data = [
            (r["name"],
             r["orig_baseline"][agg_name], r["fixed_baseline"][agg_name],
             r["orig_flowbook"][agg_name], r["fixed_flowbook"][agg_name])
            for r in data_rows
            if all(r[k][agg_name] is not None and r[k][agg_name] > 0
                   for k in ["orig_baseline", "fixed_baseline",
                             "orig_flowbook", "fixed_flowbook"])
        ]

        if len(full_data) < 2:
            print(f"\n  [{agg_name.title()} aggregation] Insufficient data (n={len(full_data)})")
            continue

        orig_overhead = np.array([r[3] / r[1] for r in full_data])
        fixed_overhead = np.array([r[4] / r[2] for r in full_data])

        if agg_name == "median":
            print(f"\nSample size (n): {len(full_data)}")
            print(f"\nPer-benchmark overhead ratios (flowbook/baseline, {agg_name}), sorted by |delta|:")
            print(f"  {'Benchmark':>{name_width}}   {'Orig':>8}  {'Fixed':>8}  {'Delta':>8}")
            print(f"  {'-' * name_width}   {'-' * 8}  {'-' * 8}  {'-' * 8}")
            sorted_full = sorted(full_data, key=lambda r: abs(r[4] / r[2] - r[3] / r[1]), reverse=True)
            for pname, ob, fb, of_, ff in sorted_full:
                orig_r = of_ / ob
                fixed_r = ff / fb
                delta = fixed_r - orig_r
                sign = "+" if delta > 0 else ""
                disp = truncate_name(pname, name_width)
                print(f"  {disp:>{name_width}}   {orig_r:>8.4f}  {fixed_r:>8.4f}  {sign}{delta:>7.4f}")

        g_ratio, g_pct, g_pval, g_effect = paired_permutation_test_geometric(
            orig_overhead, fixed_overhead, args.permutations
        )
        a_ratio, a_pct, a_pval, a_effect = paired_permutation_test_arithmetic(
            orig_overhead, fixed_overhead, args.permutations
        )

        print(f"\n  [{agg_name.title()} aggregation]")
        print(f"  {'Test Method':<20} {'Mean Ratio':>12} {'% Change':>12} {'Effect Size':>12} {'p-value':>10}")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")
        print(f"  {'Geometric mean':<20} {g_ratio:>12.4f} {g_pct:>+11.2f}% {g_effect:>12.3f} {g_pval:>10.4f}")
        print(f"  {'Arithmetic mean':<20} {a_ratio:>12.4f} {a_pct:>+11.2f}% {a_effect:>12.3f} {a_pval:>10.4f}")

        if g_pval < args.alpha:
            direction = "LOWER" if g_pct > 0 else "HIGHER"
            print(f"\n  >>> SIGNIFICANT ({agg_name}): Fixed has {direction} overhead (p={g_pval:.4f})")
        else:
            print(f"\n  >>> NOT SIGNIFICANT ({agg_name}): No detectable difference (p={g_pval:.4f})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
