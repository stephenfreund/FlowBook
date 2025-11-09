"""
Helper functions for CLI operations.

This module provides shared utilities for notebook and kernel management
used by both cli.py and optimize_cli.py.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from jupyter_client import KernelManager

from data_ferret import make_kernels
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.output import error, log, timer


def convert_all_source_to_strings(notebook_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert cell sources from list format to string format.

    Args:
        notebook_content: The notebook JSON structure

    Returns:
        The notebook with all sources as strings
    """
    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "code" and isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    return notebook_content


def detect_file_type(filepath: str) -> str:
    """
    Detect if a file is a notebook or kernel connection file.

    Args:
        filepath: Path to the file

    Returns:
        One of: "notebook", "connection", "unknown"
    """
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


def convert_cell_indices_to_ids(notebook_content: Dict[str, Any], cell_id_args: List[str]) -> List[str]:
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


def load_notebook(notebook_path: str) -> Dict[str, Any]:
    """
    Load a notebook from disk and prepare it for processing.

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        Notebook content as a dictionary

    Raises:
        FileNotFoundError: If notebook doesn't exist
        json.JSONDecodeError: If notebook is invalid JSON
    """
    with timer(key="load_notebook", message=f"Loading notebook: {notebook_path}"):
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)

    return convert_all_source_to_strings(notebook_content)


def setup_kernel(
    connection_file: Optional[str] = None,
    kernel_name: str = "ferret_kernel"
) -> Tuple[Optional[KernelManager], FerretKernelClient]:
    """
    Start a new kernel or connect to an existing one.

    Args:
        connection_file: Path to kernel connection file (optional)
        kernel_name: Name of kernel to start if not connecting to existing

    Returns:
        Tuple of (KernelManager or None, FerretKernelClient)
        KernelManager is only returned if a new kernel was started.

    Raises:
        Exception: If kernel setup fails
    """
    make_kernels()

    kernel_manager = None
    kernel_client = None

    if connection_file:
        # Connect to existing kernel using connection file
        with timer(key="connect_kernel", message="Connecting to existing kernel"):
            try:
                with timer(key="read_connection_file", message=f"Reading connection file: {connection_file}"):
                    # Read connection file
                    with open(connection_file, "r", encoding="utf-8") as f:
                        connection_info = json.load(f)

                # Extract kernel ID from connection file
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
                log("Connected to kernel successfully")

            except FileNotFoundError:
                raise FileNotFoundError(f"Connection file not found: {connection_file}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in connection file: {e}")
            except Exception as e:
                raise Exception(f"Error connecting to kernel: {e}")
    else:
        # Start new kernel
        with timer(key="start_kernel", message=f"Starting new kernel: {kernel_name}"):
            kernel_manager = KernelManager(kernel_name=kernel_name)
            try:
                kernel_manager.start_kernel()
            except Exception as e:
                raise Exception(f"Error starting kernel: {e}")

            kernel_client = FerretKernelClient(kernel_id=kernel_manager.kernel_id)
            kernel_client.load_connection_info(kernel_manager.get_connection_info())
            kernel_client.start_channels()

            for i in range(3):
                try:
                    kernel_client.wait_for_ready(timeout=30)
                    assert isinstance(kernel_client, FerretKernelClient)
                    log("Kernel started successfully")
                    break
                except Exception as e:
                    log(f"Error waiting for kernel to be ready: {e}")
                    if kernel_manager.is_alive():
                        log("Kernel is still running but not responding")
                    else:
                        log("Kernel has died")
                    if i < 2:
                        log("Retrying...")
                    else:
                        raise Exception("Giving up after 3 attempts")

    return kernel_manager, kernel_client


def save_notebook(
    notebook_content: Dict[str, Any],
    output_path: Optional[str] = None,
    input_path: Optional[str] = None
) -> str:
    """
    Save a processed notebook to disk.

    Args:
        notebook_content: The notebook to save
        output_path: Explicit output path (optional)
        input_path: Original input path, used to generate default output path

    Returns:
        Path where notebook was saved

    Raises:
        ValueError: If neither output_path nor input_path is provided
    """
    if output_path:
        notebook_output = output_path
    elif input_path:
        base_name = input_path.rsplit(".", 1)[0]
        notebook_output = f"{base_name}_processed.ipynb"
    else:
        raise ValueError("Either output_path or input_path must be provided")

    with open(notebook_output, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    return notebook_output


def cleanup_kernel(
    kernel_client: Optional[FerretKernelClient],
    kernel_manager: Optional[KernelManager],
    connection_file: Optional[str] = None
) -> None:
    """
    Clean up kernel resources.

    Args:
        kernel_client: The kernel client to stop
        kernel_manager: The kernel manager (if we started the kernel)
        connection_file: Connection file path (if we connected to existing kernel)
    """
    if kernel_client:
        try:
            # Wait for kernel to be idle before shutting down to allow pending messages to complete
            # This prevents ZMQ errors from messages being sent on closed sockets
            kernel_client.kernel_info()  # Send a simple request to ensure channel is responsive
            import time
            time.sleep(0.5)  # Give time for any pending comm messages to complete
        except Exception:
            # If kernel is already dead or unresponsive, just continue with cleanup
            pass

        try:
            kernel_client.stop_channels()
        except Exception as e:
            # Ignore errors during channel stop - kernel may already be dead
            log(f"Warning: Error stopping kernel channels: {e}")

    # Only shutdown kernel if we started it (not if connecting to existing kernel)
    if kernel_manager and not connection_file:
        try:
            kernel_manager.shutdown_kernel()
        except Exception as e:
            # Ignore errors during shutdown - kernel may already be stopped
            log(f"Warning: Error shutting down kernel: {e}")
