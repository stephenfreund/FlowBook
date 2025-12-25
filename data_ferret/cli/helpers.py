"""
Helper functions for CLI operations.

This module provides shared utilities for notebook and kernel management
used by both cli.py and optimize_cli.py.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from jupyter_client import KernelManager

from data_ferret import make_kernels
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.output import error, log, timer


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
    Convert cell index notation (@A, @B, @AA) to actual cell IDs.

    Supports @A notation where indices are 0-based Excel-style for code cells.
    For example, @A refers to the first code cell, @B to the second, etc.

    Args:
        notebook_content: The loaded notebook JSON
        cell_id_args: List of cell IDs or @-notation strings

    Returns:
        List of actual cell IDs with @-notation converted to UUIDs

    Raises:
        ValueError: If @-notation is invalid or out of range
    """
    from data_ferret.util.cell_index import alpha_to_index, index_to_alpha

    # Build list of code cell IDs (0-based)
    code_cell_ids = []

    for cell in notebook_content.get('cells', []):
        if cell.get('cell_type') == 'code':
            code_cell_ids.append(cell.get('id'))

    # Convert cell ID arguments
    converted_ids = []

    for cell_arg in cell_id_args:
        if isinstance(cell_arg, str) and cell_arg.startswith('@'):
            # Parse the alpha index
            try:
                index = alpha_to_index(cell_arg)  # Returns 0-based index
            except ValueError as e:
                raise ValueError(f"Invalid cell index format: '{cell_arg}'. {str(e)}")

            if index < 0 or index >= len(code_cell_ids):
                max_index = len(code_cell_ids)
                if max_index > 0:
                    max_cell = index_to_alpha(max_index - 1)
                    raise ValueError(
                        f"Cell index out of range: '{cell_arg}'. "
                        f"Notebook has {max_index} code cell{'s' if max_index != 1 else ''} "
                        f"(valid range: @A to {max_cell})."
                    )
                else:
                    raise ValueError(
                        f"Cell index out of range: '{cell_arg}'. "
                        f"Notebook has no code cells."
                    )

            # Convert to actual cell ID
            cell_id = code_cell_ids[index]
            converted_ids.append(cell_id)
            log(f"Converted {cell_arg} -> {cell_id}")
        else:
            # Already a cell ID, pass through
            converted_ids.append(cell_arg)

    return converted_ids


def load_notebook(notebook_path: str) -> Dict[str, Any]:
    """
    Load a notebook from disk and normalize it.

    Normalization includes:
    - Adding unique 4-character IDs to cells without IDs
    - Replacing non-4-character IDs with new 4-character IDs
    - Ensuring all cell IDs are unique
    - Converting cell sources from list to string format

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        Normalized notebook content as a dictionary

    Raises:
        FileNotFoundError: If notebook doesn't exist
        json.JSONDecodeError: If notebook is invalid JSON
    """
    from data_ferret.util.cell_ids import normalize_notebook

    with timer(key="load_notebook", message=f"Loading notebook: {notebook_path}"):
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)

    # Normalize notebook (add IDs, ensure uniqueness, convert sources)
    notebook_content = normalize_notebook(notebook_content)

    return notebook_content


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
        # Start new kernel with retry logic
        max_attempts = 3
        kernel_manager = None
        kernel_client = None

        with timer(key="start_kernel", message=f"Starting new kernel: {kernel_name}"):
            for attempt in range(max_attempts):
                try:
                    # Clean up any previous failed attempt
                    if kernel_client is not None:
                        try:
                            kernel_client.stop_channels()
                        except Exception:
                            pass
                    if kernel_manager is not None:
                        try:
                            kernel_manager.shutdown_kernel(now=True)
                        except Exception:
                            pass

                    # Start fresh kernel
                    kernel_manager = KernelManager(kernel_name=kernel_name)
                    kernel_manager.start_kernel()

                    kernel_client = FerretKernelClient(kernel_id=kernel_manager.kernel_id)
                    kernel_client.load_connection_info(kernel_manager.get_connection_info())
                    kernel_client.start_channels()

                    kernel_client.wait_for_ready(timeout=30)
                    assert isinstance(kernel_client, FerretKernelClient)
                    log("Kernel started successfully")
                    break

                except Exception as e:
                    log(f"Error on attempt {attempt + 1}/{max_attempts}: {e}")
                    if kernel_manager is not None and kernel_manager.is_alive():
                        log("Kernel is still running but not responding")
                        kernel_manager.shutdown_kernel(now=True)

                        # Wait 5 seconds before restarting
                        while kernel_manager.is_alive():
                            log("Waiting for kernel to die...")
                            time.sleep(1)

                    log("Kernel has died")

                    if attempt < max_attempts - 1:
                        log("Restarting kernel...")
                        time.sleep(0.5)  # Give ZMQ sockets time to release
                    else:
                        # Clean up before raising
                        if kernel_client is not None:
                            try:
                                kernel_client.stop_channels()
                            except Exception:
                                pass
                        if kernel_manager is not None:
                            try:
                                kernel_manager.shutdown_kernel(now=True)
                            except Exception:
                                pass
                        raise Exception(f"Kernel failed to start after {max_attempts} attempts: {e}")
                    kernel_manager = None
                    kernel_client = None



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


def format_metadata(metadata: Dict[str, Any], indent: int = 0) -> str:
    """
    Format metadata in a human-readable way.

    Converts nested dictionaries and lists into a formatted string with
    indentation and bullet points instead of raw JSON.

    Args:
        metadata: The metadata dictionary to format
        indent: Current indentation level (internal use)

    Returns:
        Formatted string representation of metadata
    """
    lines = []
    indent_str = "  " * indent

    # Handle special cases for specific metadata structures
    if indent == 0:
        # Top-level formatting
        status = metadata.get("status", "unknown")
        command = metadata.get("command", "unknown")
        message = metadata.get("message", "")

        lines.append(f"Status: {status}")
        lines.append(f"Command: {command}")

        if message:
            lines.append(f"Message: {message}")
            lines.append("")

        # Handle remaining fields
        for key, value in metadata.items():
            if key in ["status", "command", "message"]:
                continue

            lines.append(f"{key.replace('_', ' ').title()}:")
            lines.append(format_metadata_value(value, indent=1))
    else:
        # Nested formatting
        for key, value in metadata.items():
            key_formatted = key.replace('_', ' ').title()
            lines.append(f"{indent_str}{key_formatted}:")
            lines.append(format_metadata_value(value, indent=indent+1))

    return "\n".join(lines)


def _is_simple_list(value: list) -> bool:
    """
    Check if a list contains only simple scalar values (strings, numbers, bools).

    Simple lists are formatted inline like [a, b, c] instead of with bullet points.
    """
    return all(
        isinstance(item, (str, int, float, bool)) and
        (not isinstance(item, str) or '\n' not in item)
        for item in value
    )


def format_metadata_value(value: Any, indent: int = 0) -> str:
    """
    Format a single metadata value (helper for format_metadata).

    Args:
        value: The value to format
        indent: Current indentation level

    Returns:
        Formatted string representation of the value
    """
    indent_str = "  " * indent
    lines = []

    if isinstance(value, dict):
        if not value:
            lines.append(f"{indent_str}{{}}")
        else:
            for k, v in value.items():
                k_formatted = k.replace('_', ' ').title()
                # Handle empty collections inline
                if isinstance(v, dict) and not v:
                    lines.append(f"{indent_str}{k_formatted}: {{}}")
                elif isinstance(v, list) and not v:
                    lines.append(f"{indent_str}{k_formatted}: []")
                elif isinstance(v, list) and _is_simple_list(v):
                    # Simple list of strings/numbers - format inline with key
                    formatted_items = [repr(item) if isinstance(item, str) else str(item) for item in v]
                    inline = f"[{', '.join(formatted_items)}]"
                    lines.append(f"{indent_str}{k_formatted}: {inline}")
                elif isinstance(v, (dict, list)):
                    lines.append(f"{indent_str}{k_formatted}:")
                    lines.append(format_metadata_value(v, indent=indent+1))
                elif isinstance(v, str) and '\n' in v:
                    # Multi-line string: use YAML literal block scalar (|)
                    lines.append(f"{indent_str}{k_formatted}: |")
                    for line in v.split('\n'):
                        lines.append(f"{indent_str}  {line}")
                else:
                    lines.append(f"{indent_str}{k_formatted}: {v}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{indent_str}(none)")
        # Simple list of strings/numbers - format on one line
        elif _is_simple_list(value):
            formatted_items = [repr(item) if isinstance(item, str) else str(item) for item in value]
            inline = f"[{', '.join(formatted_items)}]"
            lines.append(f"{indent_str}{inline}")
        else:
            for item in value:
                if isinstance(item, dict):
                    # YAML-style: first key-value on same line as dash
                    items_list = list(item.items())
                    if items_list:
                        first_key, first_value = items_list[0]
                        first_key_formatted = first_key.replace('_', ' ').title()

                        # If first value is simple, put it on the same line
                        if not isinstance(first_value, (dict, list)):
                            # Check if first value is multi-line string
                            if isinstance(first_value, str) and '\n' in first_value:
                                lines.append(f"{indent_str}- {first_key_formatted}: |")
                                for line in first_value.split('\n'):
                                    lines.append(f"{indent_str}    {line}")
                            else:
                                lines.append(f"{indent_str}- {first_key_formatted}: {first_value}")
                            # Add remaining keys indented
                            for k, v in items_list[1:]:
                                k_formatted = k.replace('_', ' ').title()
                                # Handle empty collections inline
                                if isinstance(v, dict) and not v:
                                    lines.append(f"{indent_str}  {k_formatted}: {{}}")
                                elif isinstance(v, list) and not v:
                                    lines.append(f"{indent_str}  {k_formatted}: []")
                                elif isinstance(v, list) and _is_simple_list(v):
                                    # Simple list - format inline
                                    formatted_items = [repr(item) if isinstance(item, str) else str(item) for item in v]
                                    inline = f"[{', '.join(formatted_items)}]"
                                    lines.append(f"{indent_str}  {k_formatted}: {inline}")
                                elif isinstance(v, (dict, list)):
                                    lines.append(f"{indent_str}  {k_formatted}:")
                                    lines.append(format_metadata_value(v, indent=indent+2))
                                elif isinstance(v, str) and '\n' in v:
                                    # Multi-line string
                                    lines.append(f"{indent_str}  {k_formatted}: |")
                                    for line in v.split('\n'):
                                        lines.append(f"{indent_str}    {line}")
                                else:
                                    lines.append(f"{indent_str}  {k_formatted}: {v}")
                        else:
                            # First value is complex, put dash alone then indent everything
                            lines.append(f"{indent_str}-")
                            for k, v in items_list:
                                k_formatted = k.replace('_', ' ').title()
                                # Handle empty collections inline
                                if isinstance(v, dict) and not v:
                                    lines.append(f"{indent_str}  {k_formatted}: {{}}")
                                elif isinstance(v, list) and not v:
                                    lines.append(f"{indent_str}  {k_formatted}: []")
                                elif isinstance(v, list) and _is_simple_list(v):
                                    # Simple list - format inline
                                    formatted_items = [repr(item) if isinstance(item, str) else str(item) for item in v]
                                    inline = f"[{', '.join(formatted_items)}]"
                                    lines.append(f"{indent_str}  {k_formatted}: {inline}")
                                elif isinstance(v, (dict, list)):
                                    lines.append(f"{indent_str}  {k_formatted}:")
                                    lines.append(format_metadata_value(v, indent=indent+2))
                                elif isinstance(v, str) and '\n' in v:
                                    # Multi-line string
                                    lines.append(f"{indent_str}  {k_formatted}: |")
                                    for line in v.split('\n'):
                                        lines.append(f"{indent_str}    {line}")
                                else:
                                    lines.append(f"{indent_str}  {k_formatted}: {v}")
                elif isinstance(item, str) and '\n' in item:
                    # Multi-line string in list: use YAML literal block scalar (|)
                    lines.append(f"{indent_str}- |")
                    for line in item.split('\n'):
                        lines.append(f"{indent_str}    {line}")
                else:
                    lines.append(f"{indent_str}- {item}")
    else:
        lines.append(f"{indent_str}{value}")

    return "\n".join(lines)


def save_metadata_file(
    metadata: Dict[str, Any],
    command: str,
    total_cost: float,
    total_time: float,
    output_path: str = "metadata.json",
    notebook_path: Optional[str] = None
) -> str:
    """
    Save command metadata to a JSON file.

    This function creates a comprehensive metadata file that includes:
    - Command that was executed
    - Notebook path (full path to the notebook file)
    - Status from the command's metadata
    - Timing information (total_time)
    - Cost information (total_cost)
    - All metadata from the command execution
    - Timestamp of when the command was executed

    Args:
        metadata: The metadata dictionary returned by the command
        command: The name of the command that was executed
        total_cost: Total cost in USD
        total_time: Total execution time in seconds
        output_path: Path where the metadata file should be saved
        notebook_path: Full path to the notebook file (optional)

    Returns:
        Path where the metadata file was saved

    Raises:
        IOError: If the metadata file cannot be written
    """
    import datetime
    from data_ferret.util.output import error

    output_metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": command,
        "notebook": notebook_path,
        "status": metadata.get("status", "unknown") if metadata else "unknown",
        "total_cost": total_cost,
        "total_time": total_time,
        "command_metadata": metadata or {}
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_metadata, f, indent=2)
        return output_path
    except IOError as e:
        error(f"Failed to write metadata file {output_path}: {e}")
        raise
