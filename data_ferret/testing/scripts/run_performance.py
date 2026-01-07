#!/usr/bin/env python3
"""
Run performance tests on a notebook.

Usage:
    python -m data_ferret.testing.scripts.run_performance notebook.ipynb [options]

Options:
    -n, --iterations N     Number of iterations (default: 10)
    -m, --modifications M  Variables to modify per test (default: 3)
    -s, --seed SEED        Random seed for reproducibility
    -o, --output DIR       Output directory (default: ./test_results)
    -v, --verbose          Verbose output
"""

import argparse
import sys
from pathlib import Path

from data_ferret.testing.notebook_loader import load_notebook
from data_ferret.testing.runner import SDCSimulator
from data_ferret.testing.performance import run_performance_test, summarize_performance_results
from data_ferret.testing.results import ResultLogger, TestConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run SDC performance tests on a notebook"
    )
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=10,
        help="Number of test iterations (default: 10)"
    )
    parser.add_argument(
        "-m", "--modifications",
        type=int,
        default=3,
        help="Variables to modify per test (default: 3)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-o", "--output",
        default="./test_results",
        help="Output directory (default: ./test_results)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Validate notebook exists
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    # Load and execute notebook
    print(f"Loading notebook: {args.notebook}")
    cells = load_notebook(args.notebook)
    print(f"Found {len(cells)} code cells")

    print("Executing notebook...")
    simulator = SDCSimulator(verbose=args.verbose)
    simulator.execute_notebook(cells)

    # Check for execution errors
    errors = [r for r in simulator.cell_records.values() if r.error]
    if errors:
        print(f"\nWarning: {len(errors)} cell(s) had execution errors")
        for record in errors[:3]:
            print(f"  Cell {record.cell_id}: {record.error}")

    # Run performance tests
    print(f"\nRunning {args.iterations} performance tests...")
    results = run_performance_test(
        simulator,
        n_iterations=args.iterations,
        seed=args.seed,
        modifications_per_test=args.modifications,
        verbose=args.verbose,
    )

    # Log results
    test_name = f"performance_{notebook_path.stem}"
    logger = ResultLogger(args.output, test_name, test_type="performance")
    logger.set_config(TestConfig(
        n_iterations=args.iterations,
        seed=args.seed,
        notebook=args.notebook,
        modifications_per_test=args.modifications,
    ))

    for result in results:
        logger.log(result)

    json_path, csv_path = logger.save_all()

    # Print detailed summary
    summary = summarize_performance_results(results)

    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Notebook: {args.notebook}")
    print(f"Cells: {len(cells)} | Iterations: {args.iterations} | Seed: {args.seed}")
    print(f"Modifications per test: {args.modifications}")
    print(f"{'=' * 60}")

    if summary["clean"]["count"] > 0:
        clean = summary["clean"]
        print(f"\nCLEAN Scenario (no modifications):")
        print(f"  Check Time (ms):  min={clean['check_time_ms']['min']:.3f}  "
              f"max={clean['check_time_ms']['max']:.3f}  "
              f"mean={clean['check_time_ms']['mean']:.3f}  "
              f"median={clean['check_time_ms']['median']:.3f}")
        print(f"  Total Time (ms):  min={clean['total_time_ms']['min']:.3f}  "
              f"max={clean['total_time_ms']['max']:.3f}  "
              f"mean={clean['total_time_ms']['mean']:.3f}")

    if summary["modified"]["count"] > 0:
        modified = summary["modified"]
        print(f"\nMODIFIED Scenario ({args.modifications} variables):")
        print(f"  Check Time (ms):  min={modified['check_time_ms']['min']:.3f}  "
              f"max={modified['check_time_ms']['max']:.3f}  "
              f"mean={modified['check_time_ms']['mean']:.3f}  "
              f"median={modified['check_time_ms']['median']:.3f}")
        print(f"  Total Time (ms):  min={modified['total_time_ms']['min']:.3f}  "
              f"max={modified['total_time_ms']['max']:.3f}  "
              f"mean={modified['total_time_ms']['mean']:.3f}")
        print(f"  Avg Modifications: {modified['avg_modifications']:.1f}")

    # Per-cell breakdown
    cell_stats = {}
    for r in results:
        if r.cell_id not in cell_stats:
            cell_stats[r.cell_id] = {"clean": [], "modified": [], "vars_checked": []}
        cell_stats[r.cell_id][r.scenario].append(r.check_time_ms)
        cell_stats[r.cell_id]["vars_checked"].append(r.num_variables_checked)

    print(f"\n{'=' * 60}")
    print(f"Per-Cell Breakdown:")
    print(f"{'Cell ID':<10} {'Clean (ms)':<12} {'Modified (ms)':<14} {'Vars Checked':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*14} {'-'*12}")
    for cell_id, stats in cell_stats.items():
        clean_avg = sum(stats["clean"]) / len(stats["clean"]) if stats["clean"] else 0
        mod_avg = sum(stats["modified"]) / len(stats["modified"]) if stats["modified"] else 0
        vars_avg = sum(stats["vars_checked"]) / len(stats["vars_checked"]) if stats["vars_checked"] else 0
        print(f"{cell_id:<10} {clean_avg:<12.3f} {mod_avg:<14.3f} {vars_avg:<12.1f}")

    # Namespace size info
    ns_sizes = [r.num_variables_in_namespace for r in results]
    print(f"\nNamespace Size: min={min(ns_sizes)} max={max(ns_sizes)} avg={sum(ns_sizes)/len(ns_sizes):.1f}")

    print(f"\n{'=' * 60}")
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
