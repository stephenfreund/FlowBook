"""
Command-line interface for splitting notebook cells.

This CLI tool processes a Jupyter notebook and splits code cells into logical,
self-contained steps using LLM analysis to improve readability and maintainability.

Usage:
    data_ferret_split input.ipynb -o output.ipynb
    data_ferret_split input.ipynb  # Output: input_split.ipynb
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from data_ferret.cli.helpers import load_notebook, save_notebook
from data_ferret.server.registry import CommandRegistry
from data_ferret.server.config import FerretConfig
from data_ferret.util.output import error, log, timer


async def split_notebook(
    input_path: Path,
    output_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> None:
    """
    Split cells in a notebook.

    Args:
        input_path: Path to input notebook
        output_path: Path to save split notebook (default: input_split.ipynb)
        model: LLM model to use for splitting (optional)
    """
    with timer(key="split_cli_total", message="Split notebook CLI"):
        # Load notebook
        log(f"Loading notebook: {input_path}")
        try:
            notebook_content = load_notebook(str(input_path))
        except FileNotFoundError:
            error(f"Notebook not found: {input_path}")
            sys.exit(1)
        except Exception as e:
            error(f"Error loading notebook: {e}")
            sys.exit(1)

        # Create config with specified model
        if model:
            config = FerretConfig(model=model)
        else:
            config = FerretConfig()

        # Get split command from registry
        registry = CommandRegistry()
        split_cmd = registry.get_command("split")

        if not split_cmd:
            error("Split command not found in registry")
            error("This may indicate an installation issue")
            sys.exit(1)

        # Execute split
        log("Executing split command...")
        try:
            result = await split_cmd.process(
                notebook_content=notebook_content,
                kernel_client=None,  # Not needed for split
                config=config,
            )
        except Exception as e:
            error(f"Error executing split command: {e}")
            sys.exit(1)

        # Extract results from ProcessingResult
        split_notebook_content = result.notebook
        metadata = result.metadata
        total_cost = result.total_cost
        total_time = result.total_time

        # Check for errors
        if metadata.get("status") == "error":
            error(f"Split command failed: {metadata.get('error', 'Unknown error')}")
            sys.exit(1)

        # Determine output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_split.ipynb"

        # Save split notebook
        log(f"Saving split notebook: {output_path}")
        try:
            save_notebook(
                split_notebook_content,
                output_path=str(output_path),
            )
        except Exception as e:
            error(f"Error saving notebook: {e}")
            sys.exit(1)

        # Display results
        log("\n" + "=" * 60)
        log("SPLIT RESULTS")
        log("=" * 60)
        log(f"Cells analyzed:  {metadata['cells_analyzed']}")
        log(f"Cells split:     {metadata['cells_split']}")
        log(f"New cells added: {metadata['total_new_cells']}")
        log("")
        log("LLM Usage:")
        llm_stats = metadata["llm_stats"]
        log(f"  Model:         {llm_stats['model']}")
        log(f"  Input tokens:  {llm_stats['input_tokens']:,}")
        log(f"  Output tokens: {llm_stats['output_tokens']:,}")
        log("")
        log("Total Cost & Time:")
        log(f"  Total Cost:    ${total_cost:.4f}")
        log(f"  Total Time:    {total_time:.2f}s")
        log("=" * 60)
        log("")
        log(f"Split notebook saved to: {output_path}")


def split_cli_main():
    """Entry point for data_ferret_split CLI."""
    parser = argparse.ArgumentParser(
        description="Split Jupyter notebook cells into logical steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split a notebook and save to custom output
  data_ferret_split notebook.ipynb -o notebook_split.ipynb

  # Split a notebook with custom model
  data_ferret_split notebook.ipynb --model claude-3-5-sonnet-20241022

  # Split a notebook (output defaults to notebook_split.ipynb)
  data_ferret_split notebook.ipynb
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input notebook path (.ipynb)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output notebook path (default: <input>_split.ipynb)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use for splitting (default: gpt-4o)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        error(f"Input notebook not found: {args.input}")
        sys.exit(1)

    if args.input.suffix != ".ipynb":
        error(f"Input must be a .ipynb file: {args.input}")
        sys.exit(1)

    # Run async split
    try:
        asyncio.run(split_notebook(args.input, args.output, args.model))
    except KeyboardInterrupt:
        log("\nSplit interrupted by user")
        sys.exit(130)
    except Exception as e:
        error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    split_cli_main()
