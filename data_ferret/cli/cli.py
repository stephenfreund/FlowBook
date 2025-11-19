"""
Command-line interface for ferret notebook processing.

This CLI provides a unified interface for executing any registered notebook command.
"""

import argparse
import json
import sys
import asyncio
from typing import Optional

from data_ferret.server.registry import CommandRegistry
from data_ferret.server.config import FerretConfig
from data_ferret.util.output import error, log

from .helpers import (
    load_notebook,
    setup_kernel,
    save_notebook,
    cleanup_kernel,
    detect_file_type,
    convert_cell_indices_to_ids,
)


def parse_file_paths(paths: list) -> tuple[Optional[str], Optional[str]]:
    """
    Parse command-line paths to identify notebook and connection files.

    Args:
        paths: List of file paths from command line

    Returns:
        Tuple of (notebook_path, connection_file)

    Raises:
        SystemExit: If file detection fails or multiple files of same type provided
    """
    notebook_path = None
    connection_file = None

    for path in paths:
        file_type = detect_file_type(path)
        if file_type == "notebook":
            if notebook_path:
                error(f"Multiple notebook files provided: {notebook_path} and {path}")
                sys.exit(1)
            notebook_path = path
        elif file_type == "connection":
            if connection_file:
                error(f"Multiple connection files provided: {connection_file} and {path}")
                sys.exit(1)
            connection_file = path
        else:
            error(f"Could not determine file type for: {path}")
            sys.exit(1)

    if not notebook_path:
        error("No notebook file provided. Please provide a .ipynb file.")
        sys.exit(1)

    return notebook_path, connection_file


def cli_main():
    """Command-line interface for the ferret command processor."""
    parser = argparse.ArgumentParser(
        description="Process Jupyter notebooks with ferret commands"
    )

    registry = CommandRegistry()

    parser.add_argument(
        "command", choices=registry.list_commands(), help="Command to execute"
    )

    parser.add_argument(
        "paths",
        nargs='+',
        help="Notebook file (.ipynb) and/or kernel connection file (kernel-*.json). Provide one or both in any order."
    )

    parser.add_argument(
        "--kernel-name",
        default="ferret_kernel",
        help="Kernel name for new kernel (default: ferret_kernel). Only used if no connection file provided.",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for the new notebook (default: adds _processed suffix)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="AI model to use for commands (default: gpt-4o)",
    )

    parser.add_argument(
        "--fast-model",
        default="gpt-4o-mini",
        help="Fast AI model to use for lightweight operations (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--cell-ids",
        nargs="+",
        help="Optional list of cell IDs to process. Can use #N for Nth code cell (1-based), e.g., --cell-ids #1 #3, or mix with actual cell IDs (default: process all cells)",
    )

    args = parser.parse_args()

    # Parse file paths
    notebook_path, connection_file = parse_file_paths(args.paths)

    # Create config from CLI arguments
    config = FerretConfig(model=args.model, fast_model=args.fast_model)

    kernel_manager = None
    kernel_client = None

    try:
        # Load notebook
        notebook_content = load_notebook(notebook_path)

        # Convert cell indices (#1, #2) to actual cell IDs
        selected_cell_ids = args.cell_ids
        if selected_cell_ids:
            try:
                selected_cell_ids = convert_cell_indices_to_ids(notebook_content, selected_cell_ids)
                log(f"Processing cells: {selected_cell_ids}")
            except ValueError as e:
                error(str(e))
                return 1

        # Get command
        command = registry.get_command(args.command)

        # Setup kernel if needed
        if command.requires_kernel:
            kernel_manager, kernel_client = setup_kernel(
                connection_file=connection_file,
                kernel_name=args.kernel_name
            )

        # Run async command.process() in event loop
        result = asyncio.run(
            command.process(
                notebook_content,
                kernel_client=kernel_client,
                selected_cell_ids=selected_cell_ids,
                config=config,
            )
        )

        # Save processed notebook
        # Handle both dict and ProcessingResult (Pydantic model)
        notebook_data = result.get("notebook") if isinstance(result, dict) else result.notebook
        metadata_data = result.get("metadata") if isinstance(result, dict) else result.metadata
        total_cost = result.get("total_cost", 0.0) if isinstance(result, dict) else result.total_cost
        total_time = result.get("total_time", 0.0) if isinstance(result, dict) else result.total_time

        output_path = save_notebook(
            notebook_data,
            output_path=args.output,
            input_path=notebook_path
        )
        print(f"Processed notebook written to {output_path}")

        # Display execution summary
        print("\n" + "=" * 60)
        print("COMMAND EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Command:     {args.command}")
        print(f"Total Cost:  ${total_cost:.4f}")
        print(f"Total Time:  {total_time:.2f}s")
        print("=" * 60)

        # Display metadata
        metadata_output = json.dumps(metadata_data, indent=2)
        print("\nDetailed Metadata:")
        print(metadata_output)

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in notebook: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_kernel(kernel_client, kernel_manager, connection_file)


if __name__ == "__main__":
    sys.exit(cli_main())
