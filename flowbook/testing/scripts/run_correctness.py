#!/usr/bin/env python3
"""
Run correctness tests on a notebook.

Usage:
    python -m flowbook.testing.scripts.run_correctness notebook.ipynb [options]

Options:
    -n, --iterations N     Number of iterations (default: 10)
    -s, --seed SEED        Random seed for reproducibility
    -o, --output DIR       Output directory (default: ./test_results)
"""

import argparse
import sys
from pathlib import Path

from flowbook.testing.notebook_loader import load_notebook
from flowbook.testing.runner import SDCSimulator
from flowbook.testing.correctness import run_correctness_test
from flowbook.testing.results import ResultLogger, TestConfig
from flowbook.util.output import log, timer


def main():
    parser = argparse.ArgumentParser(
        description="Run SDC correctness tests on a notebook"
    )
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=10,
        help="Number of test iterations (default: 10)"
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

    args = parser.parse_args()

    # Validate notebook exists
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    # Load and execute notebook
    with timer(key="script:load_notebook", message=f"Loading notebook {args.notebook}"):
        cells = load_notebook(args.notebook)

    log(f"Found {len(cells)} code cells")

    with timer(key="script:execute_notebook", message="Executing notebook"):
        simulator = SDCSimulator()
        simulator.execute_notebook(cells)

    # Check for execution errors
    errors = [r for r in simulator.cell_records.values() if r.error]
    if errors:
        log(f"Warning: {len(errors)} cell(s) had execution errors")
        for record in errors[:3]:
            log(f"  Cell {record.cell_id}: {record.error}")

    # Run correctness tests
    with timer(key="script:run_tests", message=f"Running {args.iterations} correctness tests"):
        results = run_correctness_test(
            simulator,
            n_iterations=args.iterations,
            seed=args.seed,
        )

    # Log results
    test_name = f"correctness_{notebook_path.stem}"
    logger = ResultLogger(args.output, test_name, test_type="correctness")
    logger.set_config(TestConfig(
        n_iterations=args.iterations,
        seed=args.seed,
        notebook=args.notebook,
    ))

    for result in results:
        logger.log(result)

    json_path, csv_path = logger.save_all()

    # Print detailed summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\n{'=' * 60}")
    print(f"CORRECTNESS TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Notebook: {args.notebook}")
    print(f"Cells: {len(cells)} | Iterations: {args.iterations} | Seed: {args.seed}")
    print(f"{'=' * 60}")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed} ({100*passed/len(results):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(results):.1f}%)")
    print(f"{'=' * 60}")

    # Show per-cell breakdown
    cell_stats = {}
    for r in results:
        if r.cell_id not in cell_stats:
            cell_stats[r.cell_id] = {"passed": 0, "failed": 0, "times": []}
        if r.passed:
            cell_stats[r.cell_id]["passed"] += 1
        else:
            cell_stats[r.cell_id]["failed"] += 1
        cell_stats[r.cell_id]["times"].append(r.re_execution_time_ms)

    print(f"\nPer-Cell Breakdown:")
    print(f"{'Cell ID':<12} {'Pass':<6} {'Fail':<6} {'Avg Time (ms)':<15}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*15}")
    for cell_id, stats in cell_stats.items():
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        print(f"{cell_id:<12} {stats['passed']:<6} {stats['failed']:<6} {avg_time:<15.2f}")

    # Show failures if any
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n{'=' * 60}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'=' * 60}")
        for f in failures[:5]:  # Show first 5 failures
            print(f"\n  Cell: {f.cell_id} (iteration {f.iteration})")
            if f.error:
                print(f"    Error: {f.error}")
            if f.unexpected_diffs:
                for var, diff in list(f.unexpected_diffs.items())[:3]:
                    diff_short = str(diff)[:100] + "..." if len(str(diff)) > 100 else str(diff)
                    print(f"    Diff in '{var}': {diff_short}")
        if len(failures) > 5:
            print(f"\n  ... and {len(failures) - 5} more failures")

    print(f"\n{'=' * 60}")
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"{'=' * 60}")

    # Exit with error code if any tests failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
