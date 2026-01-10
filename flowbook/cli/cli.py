"""
Command-line interface for flowbook notebook processing.

This CLI provides a unified interface for executing any registered notebook command.
Kernel output is streamed to the terminal in real-time via Unix socket.
"""

import argparse
import json
import os
import sys
import asyncio
from typing import Optional

from flowbook.server.registry import CommandRegistry
from flowbook.server.config import FlowbookConfig
from flowbook.util.output import error, log, output, timer
from flowbook.util.socket_receiver import setup_socket_receiver

from .helpers import (
    load_notebook,
    setup_kernel,
    save_notebook,
    cleanup_kernel,
    detect_file_type,
    convert_cell_indices_to_ids,
    format_metadata,
    save_metadata_file,
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
                error(
                    f"Multiple connection files provided: {connection_file} and {path}"
                )
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
    """Command-line interface for the flowbook command processor."""
    parser = argparse.ArgumentParser(
        description="Process Jupyter notebooks with flowbook commands"
    )

    registry = CommandRegistry()

    # Common arguments (before command)
    parser.add_argument(
        "--kernel-name",
        default="flowbook_kernel",
        help="Kernel name for new kernel (default: varies by command, typically flowbook_kernel). Only used if no connection file provided.",
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
        help="Optional list of cell IDs to process. Can use @A notation for 0-based code cell indexing (e.g., --cell-ids @A @C for cells 0 and 2), or mix with actual cell IDs (default: process all cells)",
    )

    parser.add_argument(
        "--timings-file",
        default="flowbook-times.json",
        help="Output file for timing data (default: flowbook-times.json)",
    )

    parser.add_argument(
        "--metadata-file",
        default="metadata.json",
        help="Output file for command metadata (default: metadata.json in current directory)",
    )

    parser.add_argument(
        "--force-checkpoints",
        action="store_true",
        help="Force checkpointing before every cell execution (stored as pre_{cell_id})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Subparsers for each command
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in registry.get_all_commands():
        # Command creates its subparser (may add command-specific args)
        subparser = command.make_subparser(subparsers)
        # CLI adds paths after command returns it
        subparser.add_argument(
            "paths",
            nargs="+",
            help="Notebook file (.ipynb) and/or kernel connection file (kernel-*.json). Provide one or both in any order.",
        )

    args = parser.parse_args()

    # log the args in a nice format (arg and value in columns)
    print("=" * 60)
    print("Arguments")
    print("=" * 60)
    for arg in args.__dict__:
        print(f"{arg:<20}: {getattr(args, arg)}")
    print("=" * 60)

    # Remove old timings file
    if os.path.exists(args.timings_file):
        os.remove(args.timings_file)
        log(f"Removed old timings file: {args.timings_file}")

    # Set environment variable so kernel process inherits the same timings file
    os.environ['FLOWBOOK_TIMINGS_FILE'] = args.timings_file

    # Set the timings file for the global output object in CLI process
    output.set_timings_file(args.timings_file)

    # Parse file paths
    notebook_path, connection_file = parse_file_paths(args.paths)

    # Create config from CLI arguments
    config = FlowbookConfig(model=args.model, fast_model=args.fast_model)

    kernel_manager = None
    kernel_client = None
    socket_receiver = None

    try:
        # Load notebook
        notebook_content = load_notebook(notebook_path)

        # Convert cell indices (@A, @B) to actual cell IDs
        selected_cell_ids = args.cell_ids
        if selected_cell_ids:
            try:
                selected_cell_ids = convert_cell_indices_to_ids(
                    notebook_content, selected_cell_ids
                )
                log(f"Processing cells: {selected_cell_ids}")
            except ValueError as e:
                error(str(e))
                return 1

        # Get command
        command = registry.get_command(args.command)

        # Setup kernel if needed
        if command.requires_kernel:
            # Set up socket receiver for kernel output
            socket_receiver, socket_path = setup_socket_receiver("flowbook_cli")
            log(f"Kernel output socket: {socket_path}")

            # Use command's preferred kernel if no --kernel-name was explicitly provided
            # (check if it's the default value)
            kernel_to_use = args.kernel_name
            if args.kernel_name == "flowbook_kernel" and hasattr(command, 'kernel_name'):
                # User didn't override, use command's preference
                kernel_to_use = command.kernel_name
                if kernel_to_use != args.kernel_name:
                    log(f"Command prefers kernel: {kernel_to_use} (overriding default)")

            log(f"Using kernel: {kernel_to_use}")

            kernel_manager, kernel_client = setup_kernel(
                connection_file=connection_file, kernel_name=kernel_to_use
            )

            # Enable force checkpoints if requested
            if args.force_checkpoints:
                from flowbook.kernel.kernel_command_client import KernelCommandClient

                command_client = KernelCommandClient(kernel_client, timeout=30)
                command_client.force_checkpoints(enabled=True)
                log("Force checkpoints enabled")

        # Extract command-specific kwargs (those not in common CLI args)
        common_args = {
            'command', 'paths', 'kernel_name', 'output', 'model', 'fast_model',
            'cell_ids', 'timings_file', 'metadata_file', 'force_checkpoints', 'verbose'
        }
        command_kwargs = {
            k: v for k, v in vars(args).items()
            if k not in common_args and v is not None
        }

        # Run async command.process() in event loop
        result = asyncio.run(
            command.process(
                notebook_content,
                kernel_client=kernel_client,
                selected_cell_ids=selected_cell_ids,
                config=config,
                **command_kwargs,
            )
        )

        # Save processed notebook
        # Handle both dict and ProcessingResult (Pydantic model)
        notebook_data = (
            result.get("notebook") if isinstance(result, dict) else result.notebook
        )
        metadata_data = (
            result.get("metadata") if isinstance(result, dict) else result.metadata
        )
        total_cost = (
            result.get("total_cost", 0.0)
            if isinstance(result, dict)
            else result.total_cost
        )
        total_time = (
            result.get("total_time", 0.0)
            if isinstance(result, dict)
            else result.total_time
        )

        output_path = save_notebook(
            notebook_data, output_path=args.output, input_path=notebook_path
        )
        print(f"Processed notebook written to {output_path}")

        # Display execution summary
        status = metadata_data.get("status", "unknown") if metadata_data else "unknown"
        error_msg = metadata_data.get("error_message") if metadata_data else None
        error_cell_id = metadata_data.get("error_cell_id") if metadata_data else None
        print("\n" + "=" * 60)
        print("COMMAND EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Command:     {args.command}")
        print(f"Status:      {status.upper()}")
        if error_msg:
            print(f"Cell ID:     {error_cell_id}")
            print(f"Error:       {error_msg}")
        print(f"Total Cost:  ${total_cost:.4f}")
        print(f"Total Time:  {total_time:.2f}s")
        print("=" * 60)

        # Display metadata
        if args.verbose:
            print("\n" + "=" * 60)
            print("DETAILED METADATA")
            print("=" * 60)
            print(format_metadata(metadata_data))
            print("=" * 60)

        # Save metadata to file
        try:
            metadata_file_path = save_metadata_file(
                metadata=metadata_data,
                command=args.command,
                total_cost=total_cost,
                total_time=total_time,
                output_path=args.metadata_file,
                notebook_path=os.path.abspath(notebook_path)
            )
            print(f"\nMetadata written to {metadata_file_path}")
        except Exception as e:
            error(f"Warning: Could not save metadata file: {e}")

        # if any of the metadata has a status of error or timeout, return 1

        if metadata_data is None or metadata_data.get("status") in ("error", "timeout"):
            return 1

        with timer(key="cli:main_exit", message="CLI main exit"):
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
        if socket_receiver:
            socket_receiver.stop()


if __name__ == "__main__":
    sys.exit(cli_main())
