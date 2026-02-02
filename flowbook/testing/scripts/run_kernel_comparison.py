#!/usr/bin/env python3
"""
Compare notebook execution times across kernel types.

Usage:
    python -m flowbook.testing.scripts.run_kernel_comparison notebook.ipynb [options]

Examples:
    # Run on a sample notebook
    python -m flowbook.testing.scripts.run_kernel_comparison examples/titanic.ipynb

    # Skip Kishu if not installed
    python -m flowbook.testing.scripts.run_kernel_comparison examples/titanic.ipynb --skip-kishu

    # Just compare base vs flowbook
    python -m flowbook.testing.scripts.run_kernel_comparison examples/titanic.ipynb --skip-kishu
"""

import argparse
import sys
from pathlib import Path

from flowbook.testing.notebook_loader import load_notebook
from flowbook.testing.kernel_comparison import (
    KernelType,
    run_comparison,
    format_comparison_table,
)
from flowbook.util.output import log


def main():
    parser = argparse.ArgumentParser(
        description="Compare notebook execution times across kernel types"
    )
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=300.0,
        help="Timeout per cell in seconds (default: 300)"
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip baseline python3 kernel"
    )
    parser.add_argument(
        "--skip-flowbook",
        action="store_true",
        help="Skip flowbook_kernel"
    )
    parser.add_argument(
        "--skip-kishu",
        action="store_true",
        help="Skip Kishu-enabled kernel"
    )

    args = parser.parse_args()

    # Validate notebook exists
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    # Load notebook
    cells = load_notebook(str(notebook_path))
    log(f"Found {len(cells)} code cells")

    # Determine which kernels to run
    kernels_to_run = []
    if not args.skip_base:
        kernels_to_run.append(KernelType.BASE)
    if not args.skip_flowbook:
        kernels_to_run.append(KernelType.FLOWBOOK)
    if not args.skip_kishu:
        kernels_to_run.append(KernelType.KISHU)

    if not kernels_to_run:
        print("Error: No kernels selected to run. Use fewer --skip-* flags.", file=sys.stderr)
        sys.exit(1)

    # Run comparison
    result = run_comparison(
        cells=cells,
        kernels=kernels_to_run,
        cell_timeout=args.timeout,
    )

    # Print results
    print(format_comparison_table(result, notebook_path=str(notebook_path)))


if __name__ == "__main__":
    main()
