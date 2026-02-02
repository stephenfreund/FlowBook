"""
Benchmark checkpoint kernel - Measure cell execution and checkpoint times.

Usage:
    python -m flowbook.testing.benchmark_checkpoint notebook.ipynb
    python -m flowbook.testing.benchmark_checkpoint notebook.ipynb -o output.csv
    python -m flowbook.testing.benchmark_checkpoint notebook.ipynb --reruns 1000
"""

import argparse
import csv
import random
import sys
import time
from typing import List, Optional, TextIO

from jupyter_client import KernelManager

from flowbook import make_kernels
from flowbook.checkpoint_kernel import CheckpointKernelClient
from flowbook.testing.notebook_loader import Cell, load_notebook
from flowbook.util.output import log


def create_checkpoint_kernel() -> tuple[KernelManager, CheckpointKernelClient]:
    """
    Start the checkpoint kernel.

    Returns:
        Tuple of (KernelManager, CheckpointKernelClient)
    """
    make_kernels()

    max_attempts = 3
    kernel_manager = None
    kernel_client = None

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
            kernel_manager = KernelManager(kernel_name="checkpoint_kernel")
            kernel_manager.start_kernel()

            kernel_client = CheckpointKernelClient()
            kernel_client.load_connection_info(kernel_manager.get_connection_info())
            kernel_client.start_channels()

            # Race condition workaround
            time.sleep(2)
            while True:
                try:
                    kernel_client.wait_for_ready(timeout=30)
                    break
                except Exception as e:
                    log(f"Error waiting for kernel to be ready: {e}")
                    time.sleep(0.5)

            return kernel_manager, kernel_client

        except Exception as e:
            log(f"Error on attempt {attempt + 1}/{max_attempts}: {e}")
            if kernel_manager is not None and kernel_manager.is_alive():
                kernel_manager.shutdown_kernel(now=True)
                while kernel_manager.is_alive():
                    time.sleep(1)

            if attempt < max_attempts - 1:
                time.sleep(2)
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

    raise Exception("Kernel failed to start")


def cleanup_kernel(
    kernel_manager: Optional[KernelManager],
    kernel_client: Optional[CheckpointKernelClient]
) -> None:
    """Clean up kernel resources."""
    if kernel_client:
        try:
            kernel_client.kernel_info()
            time.sleep(0.5)
        except Exception:
            pass

        try:
            kernel_client.stop_channels()
        except Exception as e:
            log(f"Warning: Error stopping kernel channels: {e}")

    if kernel_manager:
        try:
            kernel_manager.shutdown_kernel()
        except Exception as e:
            log(f"Warning: Error shutting down kernel: {e}")


def execute_cell_and_extract_timing(
    kernel_client: CheckpointKernelClient,
    cell: Cell,
    timeout: float = 300.0
) -> dict:
    """
    Execute a cell and extract timing from metadata.

    Returns:
        Dict with keys: execution_count, cell_runtime_s, commit_time_s, error
    """
    msg_id = kernel_client.execute(cell.source, cell_id=cell.cell_id)

    timing_data = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return {
                "execution_count": None,
                "cell_runtime_s": None,
                "commit_time_s": None,
                "error": f"Timeout after {timeout}s"
            }

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        # Look for display_data with flowbook_checkpoint metadata
        if msg_type == "display_data":
            metadata = msg.get("content", {}).get("metadata", {})
            if "flowbook_checkpoint" in metadata:
                timing_data = metadata["flowbook_checkpoint"]

        # Check for errors
        if msg_type == "error":
            content = msg["content"]
            error_msg = "\n".join(content.get("traceback", []))
            if timing_data is None:
                timing_data = {"error": error_msg}
            else:
                timing_data["error"] = error_msg

        # Done when kernel is idle
        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Get the execute_reply message
    try:
        reply = kernel_client.get_shell_msg(timeout=1.0)
        if reply["content"]["status"] == "error" and timing_data:
            if "error" not in timing_data:
                error_content = reply["content"]
                timing_data["error"] = "\n".join(error_content.get("traceback", []))
    except Exception:
        pass

    if timing_data is None:
        return {
            "execution_count": None,
            "cell_runtime_s": None,
            "commit_time_s": None,
            "error": "No timing metadata received"
        }

    return timing_data


def execute_silent(
    kernel_client: CheckpointKernelClient,
    code: str,
    timeout: float = 60.0
) -> bool:
    """
    Execute code silently (no cell_id, no timing extraction).

    Returns:
        True if execution succeeded, False on error or timeout
    """
    msg_id = kernel_client.execute(code, silent=True)
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return False

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        if msg_type == "error":
            return False

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Drain shell reply
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass

    return True


def execute_and_get_error(
    kernel_client: CheckpointKernelClient,
    code: str,
    timeout: float = 60.0
) -> Optional[str]:
    """
    Execute code and return error message if any, None if success.
    """
    msg_id = kernel_client.execute(code, silent=True)
    start_time = time.time()
    error_msg = None

    while True:
        if time.time() - start_time > timeout:
            return f"Timeout after {timeout}s"

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        if msg_type == "error":
            content = msg["content"]
            error_msg = f"{content.get('ename', 'Error')}: {content.get('evalue', 'Unknown')}"

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Drain shell reply
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass

    return error_msg


def run_rerun_trials(
    kernel_client: CheckpointKernelClient,
    cells: List[Cell],
    num_reruns: int,
    num_modifications: int,
    output_file: TextIO,
    cell_timeout: float = 60.0,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Run rerun trials on randomly selected cells (with replacement).

    For each rerun:
    1. Pick a random cell (with replacement)
    2. Restore the post-checkpoint for that cell
    3. Randomly modify the namespace
    4. Trigger a checkpoint and measure timing

    Keeps trying until num_reruns successful measurements are collected.

    Args:
        kernel_client: Connected kernel client
        cells: List of cells that were executed
        num_reruns: Number of successful rerun measurements to collect
        num_modifications: Number of variables to modify per rerun
        output_file: File to write CSV output
        cell_timeout: Timeout per rerun in seconds
        seed: Random seed for reproducibility

    Returns:
        List of timing dicts for each rerun
    """
    if seed is not None:
        random.seed(seed)

    writer = csv.writer(output_file)
    writer.writerow(["cell_id", "commit_time_s", "num_modifications"])

    results = []
    attempts = 0
    max_attempts = num_reruns * 3  # Give up after 3x attempts

    while len(results) < num_reruns and attempts < max_attempts:
        attempts += 1
        # Pick a random cell (with replacement)
        cell = random.choice(cells)
        log(f"Rerun {len(results)+1}/{num_reruns} (attempt {attempts}): Cell {cell.cell_id}...")

        # 1. Restore checkpoint
        restore_code = f'_flowbook_checkpoint.restore("post_{cell.cell_id}", globals())'
        error = execute_and_get_error(kernel_client, restore_code, cell_timeout)
        if error:
            log(f"  FAILED restoring checkpoint: {error}")
            continue

        # 2. Modify namespace
        modify_code = f'''
from flowbook.testing.performance import _randomly_modify_namespace
_randomly_modify_namespace(globals(), {num_modifications},
    exclude={{"__builtins__", "__name__", "__doc__", "_flowbook_checkpoint"}})
'''
        error = execute_and_get_error(kernel_client, modify_code, cell_timeout)
        if error:
            log(f"  FAILED modifying namespace: {error}")
            continue

        # 3. Trigger checkpoint and extract timing
        trigger_cell = Cell(
            cell_id=f"{cell.cell_id}_rerun_{attempts}",
            source="# __flowbook_force_checkpoint__\nz=100",
            cell_type="code",
            index=-1,
        )
        timing = execute_cell_and_extract_timing(
            kernel_client,
            trigger_cell,
            timeout=cell_timeout
        )

        if timing.get("error"):
            log(f"  FAILED checkpoint: {timing['error'][:200]}")
            continue

        commit_time = timing.get("commit_time_s", 0)
        writer.writerow([
            cell.cell_id,
            commit_time,
            num_modifications,
        ])
        output_file.flush()

        results.append({
            "cell_id": cell.cell_id,
            "commit_time_s": commit_time,
            "num_modifications": num_modifications,
        })

        log(f"  Commit: {commit_time*1000:.1f}ms")

    if len(results) < num_reruns:
        log(f"WARNING: Only collected {len(results)}/{num_reruns} successful reruns after {attempts} attempts")

    return results


def run_benchmark(
    notebook_path: str,
    output_file: Optional[TextIO] = None,
    cell_timeout: float = 300.0,
    num_reruns: int = 0,
    rerun_modifications: int = 3,
    rerun_output_file: Optional[TextIO] = None,
    rerun_seed: Optional[int] = None,
) -> List[dict]:
    """
    Run benchmark on a notebook.

    Args:
        notebook_path: Path to .ipynb file
        output_file: File to write CSV output (default: stdout)
        cell_timeout: Timeout per cell in seconds
        num_reruns: Number of rerun measurements to take (0 = skip)
        rerun_modifications: Number of variables to modify per rerun
        rerun_output_file: File to write rerun CSV output
        rerun_seed: Random seed for rerun selection

    Returns:
        List of timing dicts for each cell
    """
    if output_file is None:
        output_file = sys.stdout

    # Load notebook cells
    cells = load_notebook(notebook_path)
    log(f"Loaded {len(cells)} code cells from {notebook_path}")

    # Start kernel
    kernel_manager = None
    kernel_client = None
    results = []
    executed_cells = []

    try:
        log("Starting checkpoint kernel...")
        kernel_manager, kernel_client = create_checkpoint_kernel()
        log("Kernel ready")

        # Write CSV header
        writer = csv.writer(output_file)
        writer.writerow(["cell_id", "execution_count", "cell_runtime_s", "commit_time_s"])

        # Execute each cell
        for i, cell in enumerate(cells):
            log(f"Executing cell {i+1}/{len(cells)} ({cell.cell_id})...")
            timing = execute_cell_and_extract_timing(kernel_client, cell, cell_timeout)
            results.append(timing)

            # Write CSV row
            if timing.get("error"):
                log(f"  Error: {timing['error'][:100]}...")
            else:
                writer.writerow([
                    cell.cell_id,
                    timing.get("execution_count", ""),
                    timing.get("cell_runtime_s", ""),
                    timing.get("commit_time_s", "")
                ])
                log(f"  Run: {timing.get('cell_runtime_s', 0)*1000:.1f}ms, Commit: {timing.get('commit_time_s', 0)*1000:.1f}ms")
                executed_cells.append(cell)

        # Flush output
        output_file.flush()

        # Run rerun trials if requested
        if num_reruns > 0 and executed_cells and rerun_output_file is not None:
            log(f"\nStarting {num_reruns} rerun measurements...")
            run_rerun_trials(
                kernel_client,
                executed_cells,
                num_reruns,
                rerun_modifications,
                rerun_output_file,
                cell_timeout=60.0,
                seed=rerun_seed,
            )

        return results

    finally:
        cleanup_kernel(kernel_manager, kernel_client)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark checkpoint kernel execution and commit times"
    )
    parser.add_argument(
        "notebook",
        help="Path to notebook file (.ipynb)"
    )
    parser.add_argument(
        "-o", "--output",
        default="flowbook_timings.csv",
        help="Output CSV file (default: flowbook_timings.csv)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout per cell in seconds (default: 300)"
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=0,
        help="Number of rerun measurements to take (default: 0 = skip)"
    )
    parser.add_argument(
        "--modifications",
        type=int,
        default=3,
        help="Number of variables to modify per rerun (default: 3)"
    )
    parser.add_argument(
        "--rerun-output",
        default="flowbook_rerun_timings.csv",
        help="Output CSV file for rerun timings (default: flowbook_rerun_timings.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for rerun cell selection (default: None)"
    )

    args = parser.parse_args()

    output_file = None
    rerun_output_file = None
    try:
        output_file = open(args.output, "w", newline="")
        log(f"Writing results to {args.output}")

        if args.reruns > 0:
            rerun_output_file = open(args.rerun_output, "w", newline="")
            log(f"Writing rerun results to {args.rerun_output}")

        run_benchmark(
            args.notebook,
            output_file,
            args.timeout,
            num_reruns=args.reruns,
            rerun_modifications=args.modifications,
            rerun_output_file=rerun_output_file,
            rerun_seed=args.seed,
        )
    finally:
        if output_file:
            output_file.close()
        if rerun_output_file:
            rerun_output_file.close()


if __name__ == "__main__":
    main()
