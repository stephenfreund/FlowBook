"""
Command-line interface for ferret notebook processing.
"""

import argparse
import json
import sys
import asyncio
from typing import Any, Dict
from jupyter_client import KernelManager

from data_ferret import make_kernels
from data_ferret.util.output import error, log, timer

from .registry import CommandRegistry
from .kernel_manager import FerretKernelClient
from .config import FerretConfig


def convert_all_source_to_strings(notebook_content: Dict[str, Any]) -> Dict[str, Any]:
    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "code" and isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    return notebook_content


def detect_file_type(filepath: str) -> str:
    """Detect if a file is a notebook or kernel connection file."""
    import os

    if not os.path.exists(filepath):
        return "unknown"

    # Check by extension first
    if filepath.endswith('.ipynb'):
        return "notebook"

    # Check if it looks like a kernel connection file
    filename = os.path.basename(filepath)
    if filename.startswith('kernel-') and filename.endswith('.json'):
        return "connection"

    # Try to detect by content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Connection files have specific keys
            if all(key in data for key in ['transport', 'ip', 'shell_port', 'iopub_port']):
                return "connection"
            # Notebooks have cells
            if 'cells' in data and 'metadata' in data:
                return "notebook"
    except:
        pass

    return "unknown"


def convert_cell_indices_to_ids(notebook_content: Dict[str, Any], cell_id_args: list) -> list:
    """
    Convert cell index notation (#1, #2) to actual cell IDs.

    Supports #N notation where N is a 1-based index of code cells.
    For example, #1 refers to the first code cell, #2 to the second, etc.

    Args:
        notebook_content: The loaded notebook JSON
        cell_id_args: List of cell IDs or #-notation strings

    Returns:
        List of actual cell IDs with #-notation converted to UUIDs

    Raises:
        ValueError: If #-notation is invalid or out of range
    """
    # Build map of code cell index (1-based) to cell ID
    code_cell_map = {}
    code_cell_index = 1

    for cell in notebook_content.get('cells', []):
        if cell.get('cell_type') == 'code':
            code_cell_map[code_cell_index] = cell.get('id')
            code_cell_index += 1

    # Convert cell ID arguments
    converted_ids = []

    for cell_arg in cell_id_args:
        if isinstance(cell_arg, str) and cell_arg.startswith('#'):
            # Extract the number
            index_str = cell_arg[1:]

            try:
                index = int(index_str)
            except ValueError:
                raise ValueError(f"Invalid cell index format: '{cell_arg}'. Expected #N where N is a number.")

            if index < 1:
                raise ValueError(f"Invalid cell index: '{cell_arg}'. Index must be >= 1.")

            if index not in code_cell_map:
                max_index = len(code_cell_map)
                raise ValueError(
                    f"Cell index out of range: '{cell_arg}'. "
                    f"Notebook has {max_index} code cell{'s' if max_index != 1 else ''}."
                )

            # Convert to actual cell ID
            cell_id = code_cell_map[index]
            converted_ids.append(cell_id)
            log(f"Converted {cell_arg} -> {cell_id}")
        else:
            # Already a cell ID, pass through
            converted_ids.append(cell_arg)

    return converted_ids


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

    make_kernels()

    # Detect what files were provided
    notebook_path = None
    connection_file = None

    for path in args.paths:
        file_type = detect_file_type(path)
        if file_type == "notebook":
            if notebook_path:
                error(f"Multiple notebook files provided: {notebook_path} and {path}")
                return 1
            notebook_path = path
        elif file_type == "connection":
            if connection_file:
                error(f"Multiple connection files provided: {connection_file} and {path}")
                return 1
            connection_file = path
        else:
            error(f"Could not determine file type for: {path}")
            return 1

    if not notebook_path:
        error("No notebook file provided. Please provide a .ipynb file.")
        return 1

    # print(f"Notebook: {notebook_path}")
    # if connection_file:
    #     print(f"Connection file: {connection_file}")

    # Create config from CLI arguments with same defaults as Jupyter
    config = FerretConfig(model=args.model, fast_model=args.fast_model)

    kernel_manager = None
    kernel_client = None

    try:
        with timer(key="load_notebook", message=f"Loading notebook: {notebook_path}"):
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook_content = json.load(f)

        notebook_content = convert_all_source_to_strings(notebook_content)

        # Convert cell indices (#1, #2) to actual cell IDs
        selected_cell_ids = args.cell_ids
        if selected_cell_ids:
            try:
                selected_cell_ids = convert_cell_indices_to_ids(notebook_content, selected_cell_ids)
                log(f"Processing cells: {selected_cell_ids}")
            except ValueError as e:
                error(str(e))
                return 1

        command = registry.get_command(args.command)

        if command.requires_kernel:
            if connection_file:
                # Connect to existing kernel using connection file
                with timer(
                    key="connect_kernel",
                    message=f"Connecting to existing kernel",
                ):
                    try:
                        with timer(key="read_connection_file", message=f"Reading connection file: {connection_file}"):
                            # Read connection file
                            with open(connection_file, "r", encoding="utf-8") as f:
                                connection_info = json.load(f)

                        # Extract kernel ID from connection file (if available)
                        # The kernel ID is typically embedded in the filename
                        import os
                        filename = os.path.basename(connection_file)
                        # Format is typically kernel-<id>.json
                        if filename.startswith("kernel-") and filename.endswith(".json"):
                            kernel_id = filename[7:-5]  # Extract ID between "kernel-" and ".json"
                        else:
                            kernel_id = "unknown"

                        # Create kernel client and connect using the connection info
                        kernel_client = FerretKernelClient(kernel_id=kernel_id)
                        kernel_client.load_connection_info(connection_info)
                        kernel_client.start_channels()
                        kernel_client.wait_for_ready(timeout=30)
                        log(f"Connected to kernel successfully")

                    except FileNotFoundError:
                        error(f"Connection file not found: {connection_file}")
                        return 1
                    except json.JSONDecodeError as e:
                        error(f"Invalid JSON in connection file: {e}")
                        return 1
                    except Exception as e:
                        error(f"Error connecting to kernel: {e}")
                        return 1
            else:
                # Start new kernel
                with timer(
                    key="start_kernel",
                    message=f"Starting new kernel: {args.kernel_name}",
                ):
                    # Start kernel manager and create our custom FerretKernelClient
                    kernel_manager = KernelManager(kernel_name=args.kernel_name)
                    try:
                        kernel_manager.start_kernel()
                    except Exception as e:
                        error(f"Error starting kernel: {e}")
                        return 1

                    kernel_client = FerretKernelClient(
                        kernel_id=kernel_manager.kernel_id
                    )
                    kernel_client.load_connection_info(
                        kernel_manager.get_connection_info()
                    )
                    kernel_client.start_channels()
                    for i in range(3):
                        try:
                            kernel_client.wait_for_ready(timeout=30)
                            assert isinstance(kernel_client, FerretKernelClient)
                            log(f"Kernel started successfully")
                            break
                        except Exception as e:
                            log(f"Error waiting for kernel to be ready: {e}")
                            # Try to read kernel stderr/stdout for more details
                            if kernel_manager.is_alive():
                                log("Kernel is still running but not responding")
                            else:
                                log("Kernel has died")
                            if i < 2:
                                log(f"Retrying...")
                            else:
                                error(f"Giving up after 3 attempts")
                                return 1

        # Run async command.process() in event loop
        result = asyncio.run(
            command.process(
                notebook_content,
                kernel_client=kernel_client,
                selected_cell_ids=selected_cell_ids,
                config=config,
            )
        )

        if args.output:
            notebook_output = args.output
        else:
            base_name = notebook_path.rsplit(".", 1)[0]
            notebook_output = f"{base_name}_processed.ipynb"

        with open(notebook_output, "w", encoding="utf-8") as f:
            json.dump(result["notebook"], f, indent=2)
        print(f"Processed notebook written to {notebook_output}")

        metadata_output = json.dumps(result["metadata"], indent=2)

        print("\nMetadata:")
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
        if kernel_client:
            kernel_client.stop_channels()
        # Only shutdown kernel if we started it (not if connecting to existing kernel)
        if kernel_manager and not connection_file:
            kernel_manager.shutdown_kernel()


if __name__ == "__main__":
    sys.exit(cli_main())
