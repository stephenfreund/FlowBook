"""
Command-line interface for displaying optimization statistics from notebooks.

This CLI tool reads ferret optimization metadata from a notebook and displays
the split results, optimization results, and LLM cost summary tables.

Usage:
    data_ferret_stats notebook.ipynb
"""

import argparse
import sys
from pathlib import Path

from data_ferret.cli.helpers import load_notebook
from data_ferret.cli.optimization_metadata import FerretOptimizationMetadata
from data_ferret.cli.stats_display import display_all_stats
from data_ferret.util.output import error


def stats_cli_main():
    """Entry point for data_ferret_stats CLI."""
    parser = argparse.ArgumentParser(
        description="Display optimization statistics from a notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display stats from an optimized notebook
  data_ferret_stats notebook_optimized.ipynb

  # Display stats from a split and optimized notebook
  data_ferret_stats notebook_split_optimized.ipynb
        """,
    )
    parser.add_argument(
        "notebook_path",
        type=Path,
        help="Notebook file (.ipynb) to read statistics from",
    )

    args = parser.parse_args()

    # Validate input
    if not args.notebook_path.exists():
        error(f"Notebook not found: {args.notebook_path}")
        sys.exit(1)

    if args.notebook_path.suffix != ".ipynb":
        error(f"Input must be a .ipynb file: {args.notebook_path}")
        sys.exit(1)

    # Load notebook
    try:
        notebook_content = load_notebook(str(args.notebook_path))
    except FileNotFoundError:
        error(f"Notebook not found: {args.notebook_path}")
        sys.exit(1)
    except Exception as e:
        error(f"Error loading notebook: {e}")
        sys.exit(1)

    # Extract metadata
    if 'metadata' not in notebook_content:
        error("No metadata found in notebook")
        sys.exit(1)

    if 'ferret_optimization' not in notebook_content['metadata']:
        error("No ferret optimization metadata found in notebook")
        error("This notebook may not have been processed with data_ferret_optimize")
        sys.exit(1)

    # Parse metadata into Pydantic model
    try:
        ferret_metadata_dict = notebook_content['metadata']['ferret_optimization']
        ferret_metadata = FerretOptimizationMetadata.model_validate(ferret_metadata_dict)
    except Exception as e:
        error(f"Error parsing optimization metadata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display all statistics
    print(f"\n{'='*70}")
    print(f"Optimization Statistics for: {args.notebook_path.name}")
    print(f"{'='*70}")

    display_all_stats(ferret_metadata)

    return 0


if __name__ == "__main__":
    sys.exit(stats_cli_main())
