"""
Compare baseline (python3) vs FlowBook kernel execution with timing and memory metrics.

Usage via CLI:
    flowbook compare-baseline notebook.ipynb
    flowbook compare-baseline notebook.ipynb --timeout 300
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jupyter_client import KernelManager
from jupyter_client.blocking import BlockingKernelClient

from flowbook import make_kernels
from flowbook.kernel.flowbook_client import FlowbookKernelClient
from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.config import FlowbookConfig
from flowbook.testing.benchmark_checkpoint import (
    _MEMORY_SETUP_CODE,
    measure_memory,
    measure_checkpoint_details,
)
from flowbook.util.output import log


def generate_comparison_filename(notebook_path: str, num_components: int = 3) -> str:
    """
    Generate a comparison filename from the notebook path.

    Takes the last N path components (directories + filename without extension)
    and joins them with '---'.

    Example:
        /path/to/zyh1104/sticker-sales-solution-ensembling/notebook.ipynb
        -> zyh1104---sticker-sales-solution-ensembling---notebook_comparison.json

    Args:
        notebook_path: Path to the notebook
        num_components: Number of path components to include (default: 3)

    Returns:
        Filename string like 'dir1---dir2---filename_comparison.json'
    """
    path = Path(notebook_path)
    stem = path.stem

    # Get parent directory components
    parts = list(path.parent.parts)

    # Take the last (num_components - 1) directory parts + the stem
    dir_parts = parts[-(num_components - 1):] if len(parts) >= (num_components - 1) else parts
    all_parts = dir_parts + [stem]

    # Join with '---'
    name = '---'.join(all_parts)
    return f"{name}_comparison.json"


@dataclass
class CellMetrics:
    """Metrics for a single cell execution."""
    cell_id: str
    cell_index: int
    cell_runtime_ms: float
    state_duration_ms: float
    check_duration_ms: float
    user_ns_bytes: int
    user_ns_and_checkpoint_bytes: int
    status: str
    error: Optional[str] = None
    checkpoint_details: Optional[Dict[str, Any]] = None
    memory_warnings: Optional[List[str]] = None


@dataclass
class KernelResults:
    """Results from running a notebook on a single kernel."""
    kernel_name: str
    cells: List[CellMetrics] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    rerun_cells: List[CellMetrics] = field(default_factory=list)
    rerun_totals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Complete comparison result."""
    version: str = "1.0"
    notebook_path: str = ""
    timestamp: str = ""
    kernels: Dict[str, KernelResults] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_baseline_kernel() -> Tuple[KernelManager, BlockingKernelClient]:
    """
    Start a baseline python3 kernel.

    Returns:
        Tuple of (KernelManager, BlockingKernelClient)
    """
    max_attempts = 3
    kernel_manager = None
    kernel_client = None

    for attempt in range(max_attempts):
        try:
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

            kernel_manager = KernelManager(kernel_name="python3")
            kernel_manager.start_kernel()

            kernel_client = BlockingKernelClient()
            kernel_client.load_connection_info(kernel_manager.get_connection_info())
            kernel_client.start_channels()

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


def create_flowbook_kernel() -> Tuple[KernelManager, FlowbookKernelClient]:
    """
    Start a flowbook_kernel.

    Returns:
        Tuple of (KernelManager, FlowbookKernelClient)
    """
    make_kernels()

    max_attempts = 3
    kernel_manager = None
    kernel_client = None

    for attempt in range(max_attempts):
        try:
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

            kernel_manager = KernelManager(kernel_name="flowbook_kernel")
            kernel_manager.start_kernel()

            kernel_client = FlowbookKernelClient()
            kernel_client.load_connection_info(kernel_manager.get_connection_info())
            kernel_client.start_channels()

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


def cleanup_kernel(kernel_manager, kernel_client) -> None:
    """Clean up kernel resources."""
    if kernel_client:
        try:
            kernel_client.kernel_info()
            time.sleep(0.5)
        except Exception:
            pass
        try:
            kernel_client.stop_channels()
        except Exception:
            pass

    if kernel_manager:
        try:
            kernel_manager.shutdown_kernel()
        except Exception:
            pass


def execute_cell_baseline(
    kernel_client: BlockingKernelClient,
    source: str,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """
    Execute a cell on the baseline kernel and measure execution time.

    Returns:
        Dict with cell_runtime_ms and optional error
    """
    start = time.perf_counter()

    msg_id = kernel_client.execute(source)

    error_msg = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return {"cell_runtime_ms": None, "error": f"Timeout after {timeout}s"}

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        if msg_type == "error":
            content = msg["content"]
            error_msg = "\n".join(content.get("traceback", []))

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Get execute_reply
    try:
        reply = kernel_client.get_shell_msg(timeout=1.0)
        if reply["content"]["status"] == "error" and error_msg is None:
            error_content = reply["content"]
            error_msg = "\n".join(error_content.get("traceback", []))
    except Exception:
        pass

    elapsed = time.perf_counter() - start

    return {"cell_runtime_ms": elapsed * 1000, "error": error_msg}


def execute_cell_flowbook(
    kernel_client: FlowbookKernelClient,
    source: str,
    cell_id: str,
    cell_order: List[str],
    timeout: float = 300.0
) -> Dict[str, Any]:
    """
    Execute a cell on the flowbook_kernel and extract timing from metadata.

    Returns:
        Dict with cell_runtime_ms, state_duration_ms, check_duration_ms, and optional error
    """
    # Set cell order for reproducibility tracking
    kernel_client.set_cell_order(cell_order)

    msg_id = kernel_client.execute(source, cell_id=cell_id)

    flowbook_metadata = None
    error_msg = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return {
                "cell_runtime_ms": None,
                "state_duration_ms": None,
                "check_duration_ms": None,
                "error": f"Timeout after {timeout}s"
            }

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        # Look for display_data with flowbook metadata
        if msg_type == "display_data":
            output_meta = msg.get("content", {}).get("metadata", {})
            if "flowbook" in output_meta:
                flowbook_metadata = output_meta["flowbook"]

        if msg_type == "error":
            content = msg["content"]
            error_msg = "\n".join(content.get("traceback", []))

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Get execute_reply
    try:
        reply = kernel_client.get_shell_msg(timeout=1.0)
        if reply["content"]["status"] == "error" and error_msg is None:
            error_content = reply["content"]
            error_msg = "\n".join(error_content.get("traceback", []))
    except Exception:
        pass

    if flowbook_metadata:
        # Check for violations in metadata
        violation = flowbook_metadata.get("violation")
        violation_msg = None
        if violation:
            violation_msg = violation.get("message", "Reproducibility violation")

        return {
            "cell_runtime_ms": flowbook_metadata.get("run_duration_ms", 0.0),
            "state_duration_ms": flowbook_metadata.get("state_duration_ms", 0.0),
            "check_duration_ms": flowbook_metadata.get("check_duration_ms", 0.0),
            "error": error_msg,
            "violation": violation_msg,
            "stale_cells": flowbook_metadata.get("stale_cells", []),
        }
    else:
        return {
            "cell_runtime_ms": None,
            "state_duration_ms": None,
            "check_duration_ms": None,
            "error": error_msg or "No flowbook metadata received",
            "violation": None,
            "stale_cells": [],
        }


def _wait_for_idle(kernel_client, timeout: float = 30.0) -> None:
    """Wait for kernel to become idle after executing a command."""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break
    # Also drain shell reply
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass


def setup_memory_measurement(kernel_client, timeout: float = 60.0) -> bool:
    """Inject pympler measurement helper into the kernel."""
    msg_id = kernel_client.execute(_MEMORY_SETUP_CODE, silent=True)
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return False
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'error':
            return False
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass
    return True


def run_baseline_execution(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    rerun_indices: Optional[List[int]] = None
) -> KernelResults:
    """
    Run notebook on baseline python3 kernel and collect metrics.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        rerun_indices: List of cell indices to re-execute after initial pass (with replacement)
    """
    if rerun_indices is None:
        rerun_indices = []
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    log(f"Baseline: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = KernelResults(kernel_name="python3")

    try:
        log("Baseline: Starting python3 kernel...")
        kernel_manager, kernel_client = create_baseline_kernel()
        log("Baseline: Kernel ready")

        # Setup memory measurement
        if setup_memory_measurement(kernel_client):
            log("Baseline: Memory measurement helper injected")
        else:
            log("Baseline: WARNING: Failed to inject memory measurement helper")

        total_runtime_ms = 0.0
        final_user_ns_bytes = 0
        final_checkpoint_bytes = 0

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"Baseline: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            timing = execute_cell_baseline(kernel_client, source, cell_timeout)

            if timing.get("error"):
                log(f"  Error:\n{timing['error']}")
                results.cells.append(CellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=0.0,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    user_ns_bytes=0,
                    user_ns_and_checkpoint_bytes=0,
                    status="error",
                    error=timing["error"]
                ))
            else:
                mem = measure_memory(kernel_client)
                runtime_ms = timing["cell_runtime_ms"]
                total_runtime_ms += runtime_ms
                final_user_ns_bytes = mem["user_ns_bytes"]
                final_checkpoint_bytes = mem["user_ns_and_checkpoint_bytes"]

                results.cells.append(CellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=runtime_ms,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    user_ns_bytes=mem["user_ns_bytes"],
                    user_ns_and_checkpoint_bytes=mem["user_ns_and_checkpoint_bytes"],
                    status="ok",
                    error=None,
                    memory_warnings=mem.get("diagnostics", {}).get("warnings"),
                ))
                log(f"  Runtime: {runtime_ms:.1f}ms, Memory: {mem['user_ns_bytes']:,}B")

        results.totals = {
            "cell_runtime_ms": total_runtime_ms,
            "state_duration_ms": 0.0,
            "check_duration_ms": 0.0,
            "final_user_ns_bytes": final_user_ns_bytes,
            "final_checkpoint_bytes": final_checkpoint_bytes,
        }

        log(f"Baseline: Total runtime {total_runtime_ms:.1f}ms")

        # Run extra cells if requested
        if rerun_indices:
            log("")
            log(f"Baseline: Running {len(rerun_indices)} extra cell re-executions...")
            rerun_runtime_ms = 0.0
            last_mem = {}

            for rerun_idx, orig_idx in enumerate(rerun_indices):
                cell = code_cells[orig_idx]
                cell_id = cell.get("id", f"cell_{orig_idx}")
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)

                log(f"Baseline: Rerun {rerun_idx+1}/{len(rerun_indices)} (cell {cell_id})...")

                timing = execute_cell_baseline(kernel_client, source, cell_timeout)

                if timing.get("error"):
                    log(f"  Error:\n{timing['error']}")
                    results.rerun_cells.append(CellMetrics(
                        cell_id=cell_id,
                        cell_index=orig_idx,
                        cell_runtime_ms=0.0,
                        state_duration_ms=0.0,
                        check_duration_ms=0.0,
                        user_ns_bytes=0,
                        user_ns_and_checkpoint_bytes=0,
                        status="error",
                        error=timing["error"]
                    ))
                else:
                    last_mem = measure_memory(kernel_client)
                    runtime_ms = timing["cell_runtime_ms"]
                    rerun_runtime_ms += runtime_ms

                    results.rerun_cells.append(CellMetrics(
                        cell_id=cell_id,
                        cell_index=orig_idx,
                        cell_runtime_ms=runtime_ms,
                        state_duration_ms=0.0,
                        check_duration_ms=0.0,
                        user_ns_bytes=last_mem["user_ns_bytes"],
                        user_ns_and_checkpoint_bytes=last_mem["user_ns_and_checkpoint_bytes"],
                        status="ok",
                        error=None,
                        memory_warnings=last_mem.get("diagnostics", {}).get("warnings"),
                    ))
                    log(f"  Runtime: {runtime_ms:.1f}ms, Memory: {last_mem['user_ns_bytes']:,}B")

            results.rerun_totals = {
                "cell_runtime_ms": rerun_runtime_ms,
                "state_duration_ms": 0.0,
                "check_duration_ms": 0.0,
                "final_user_ns_bytes": last_mem.get("user_ns_bytes", 0),
                "final_checkpoint_bytes": last_mem.get("user_ns_and_checkpoint_bytes", 0),
            }
            log(f"Baseline: Rerun total runtime {rerun_runtime_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_execution(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    rerun_indices: Optional[List[int]] = None
) -> KernelResults:
    """
    Run notebook on flowbook_kernel and collect metrics.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        rerun_indices: List of cell indices to re-execute after initial pass (with replacement)
    """
    if rerun_indices is None:
        rerun_indices = []
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cell_order = [c.get("id", f"cell_{i}") for i, c in enumerate(code_cells)]

    log(f"FlowBook: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = KernelResults(kernel_name="flowbook_kernel")

    try:
        log("FlowBook: Starting flowbook_kernel...")
        kernel_manager, kernel_client = create_flowbook_kernel()
        log("FlowBook: Kernel ready")

        # Setup memory measurement
        if setup_memory_measurement(kernel_client):
            log("FlowBook: Memory measurement helper injected")
        else:
            log("FlowBook: WARNING: Failed to inject memory measurement helper")

        # Enable continue_after_violation so we can measure full execution even with repro issues
        kernel_client.execute("%continue_after_violation on", silent=True)
        # Wait for it to complete
        _wait_for_idle(kernel_client)
        log("FlowBook: continue_after_violation enabled")

        total_runtime_ms = 0.0
        total_state_ms = 0.0
        total_check_ms = 0.0
        final_user_ns_bytes = 0
        final_checkpoint_bytes = 0

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"FlowBook: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            timing = execute_cell_flowbook(
                kernel_client, source, cell_id, cell_order, cell_timeout
            )

            if timing.get("error") and timing.get("cell_runtime_ms") is None:
                log(f"  Error:\n{timing['error']}")
                results.cells.append(CellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=0.0,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    user_ns_bytes=0,
                    user_ns_and_checkpoint_bytes=0,
                    status="error",
                    error=timing["error"]
                ))
            else:
                # Log any violations (warnings with continue_after_violation)
                if timing.get("violation"):
                    log(f"  ⚠️  Violation: {timing['violation']}")

                mem = measure_memory(kernel_client)
                ckpt_details = measure_checkpoint_details(kernel_client)
                runtime_ms = timing["cell_runtime_ms"] or 0.0
                state_ms = timing["state_duration_ms"] or 0.0
                check_ms = timing["check_duration_ms"] or 0.0

                total_runtime_ms += runtime_ms
                total_state_ms += state_ms
                total_check_ms += check_ms
                final_user_ns_bytes = mem["user_ns_bytes"]
                final_checkpoint_bytes = mem["user_ns_and_checkpoint_bytes"]

                # Determine status
                if timing.get("error"):
                    status = "error"
                elif timing.get("violation"):
                    status = "violation"
                else:
                    status = "ok"

                results.cells.append(CellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=runtime_ms,
                    state_duration_ms=state_ms,
                    check_duration_ms=check_ms,
                    user_ns_bytes=mem["user_ns_bytes"],
                    user_ns_and_checkpoint_bytes=mem["user_ns_and_checkpoint_bytes"],
                    status=status,
                    error=timing.get("violation") or timing.get("error"),
                    checkpoint_details=ckpt_details,
                    memory_warnings=mem.get("diagnostics", {}).get("warnings"),
                ))
                log(f"  Runtime: {runtime_ms:.1f}ms, State: {state_ms:.1f}ms, Check: {check_ms:.1f}ms, Memory: {mem['user_ns_bytes']:,}B, Checkpoint: {mem['user_ns_and_checkpoint_bytes']:,}B")

        results.totals = {
            "cell_runtime_ms": total_runtime_ms,
            "state_duration_ms": total_state_ms,
            "check_duration_ms": total_check_ms,
            "final_user_ns_bytes": final_user_ns_bytes,
            "final_checkpoint_bytes": final_checkpoint_bytes,
        }

        log(f"FlowBook: Total runtime {total_runtime_ms:.1f}ms, state {total_state_ms:.1f}ms, check {total_check_ms:.1f}ms")

        # Run extra cells if requested
        if rerun_indices:
            log("")
            log(f"FlowBook: Running {len(rerun_indices)} extra cell re-executions...")
            rerun_runtime_ms = 0.0
            rerun_state_ms = 0.0
            rerun_check_ms = 0.0
            last_mem = {}
            last_ckpt_details = {}

            for rerun_idx, orig_idx in enumerate(rerun_indices):
                cell = code_cells[orig_idx]
                cell_id = cell.get("id", f"cell_{orig_idx}")
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)

                log(f"FlowBook: Rerun {rerun_idx+1}/{len(rerun_indices)} (cell {cell_id})...")

                timing = execute_cell_flowbook(
                    kernel_client, source, cell_id, cell_order, cell_timeout
                )

                if timing.get("error") and timing.get("cell_runtime_ms") is None:
                    log(f"  Error:\n{timing['error']}")
                    results.rerun_cells.append(CellMetrics(
                        cell_id=cell_id,
                        cell_index=orig_idx,
                        cell_runtime_ms=0.0,
                        state_duration_ms=0.0,
                        check_duration_ms=0.0,
                        user_ns_bytes=0,
                        user_ns_and_checkpoint_bytes=0,
                        status="error",
                        error=timing["error"]
                    ))
                else:
                    if timing.get("violation"):
                        log(f"  ⚠️  Violation: {timing['violation']}")

                    last_mem = measure_memory(kernel_client)
                    last_ckpt_details = measure_checkpoint_details(kernel_client)
                    runtime_ms = timing["cell_runtime_ms"] or 0.0
                    state_ms = timing["state_duration_ms"] or 0.0
                    check_ms = timing["check_duration_ms"] or 0.0

                    rerun_runtime_ms += runtime_ms
                    rerun_state_ms += state_ms
                    rerun_check_ms += check_ms

                    if timing.get("error"):
                        status = "error"
                    elif timing.get("violation"):
                        status = "violation"
                    else:
                        status = "ok"

                    results.rerun_cells.append(CellMetrics(
                        cell_id=cell_id,
                        cell_index=orig_idx,
                        cell_runtime_ms=runtime_ms,
                        state_duration_ms=state_ms,
                        check_duration_ms=check_ms,
                        user_ns_bytes=last_mem["user_ns_bytes"],
                        user_ns_and_checkpoint_bytes=last_mem["user_ns_and_checkpoint_bytes"],
                        status=status,
                        error=timing.get("violation") or timing.get("error"),
                        checkpoint_details=last_ckpt_details,
                        memory_warnings=last_mem.get("diagnostics", {}).get("warnings"),
                    ))
                    log(f"  Runtime: {runtime_ms:.1f}ms, State: {state_ms:.1f}ms, Check: {check_ms:.1f}ms, Memory: {last_mem['user_ns_bytes']:,}B, Checkpoint: {last_mem['user_ns_and_checkpoint_bytes']:,}B")

            results.rerun_totals = {
                "cell_runtime_ms": rerun_runtime_ms,
                "state_duration_ms": rerun_state_ms,
                "check_duration_ms": rerun_check_ms,
                "final_user_ns_bytes": last_mem.get("user_ns_bytes", 0),
                "final_checkpoint_bytes": last_mem.get("user_ns_and_checkpoint_bytes", 0),
            }
            log(f"FlowBook: Rerun total runtime {rerun_runtime_ms:.1f}ms, state {rerun_state_ms:.1f}ms, check {rerun_check_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


class CompareBaselineCommand(NotebookCommand):
    """
    Compare baseline (python3) vs FlowBook kernel execution.

    Runs the notebook on both kernels, collecting timing and memory metrics,
    and saves results to a JSON file.
    """

    @property
    def command_name(self) -> str:
        return "compare-baseline"

    @property
    def display_name(self) -> str:
        return "Compare Baseline"

    @property
    def icon_name(self) -> str:
        return "ui-components:compare"

    @property
    def tooltip(self) -> str:
        return "Compare execution between baseline python3 and FlowBook kernel"

    @property
    def requires_kernel(self) -> bool:
        # We manage our own kernels internally
        return False

    def make_subparser(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        """Add command-specific CLI arguments."""
        subparser = super().make_subparser(subparsers)
        subparser.add_argument(
            "--timeout",
            type=float,
            default=300.0,
            help="Timeout in seconds per cell (default: 300)",
        )
        subparser.add_argument(
            "--extra-cells", "-e",
            type=int,
            default=0,
            help="Number of additional random cells to re-execute after initial pass (default: 0)",
        )
        return subparser

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client=None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[FlowbookConfig] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Run notebook on both baseline and FlowBook kernels, compare results.

        Args:
            notebook_content: Notebook JSON
            kernel_client: Not used (we manage our own kernels)
            selected_cell_ids: Not used (we execute all cells)
            config: Optional configuration
            **kwargs: Additional arguments (e.g., timeout, notebook_path)

        Returns:
            ProcessingResult with comparison metadata
        """
        cell_timeout = kwargs.get("timeout", 300.0)
        extra_cells = kwargs.get("extra_cells", 0)
        notebook_path = kwargs.get("notebook_path", "unknown.ipynb")

        # Pre-generate random rerun sequence so both kernels execute the same cells
        cells = notebook_content.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        rerun_indices: List[int] = []
        if extra_cells > 0:
            executable_indices = [i for i, c in enumerate(code_cells)
                                  if c.get("source", "").strip()]
            if executable_indices:
                rerun_indices = random.choices(executable_indices, k=extra_cells)

        with self.timing_context() as get_elapsed:
            log(f"Starting baseline vs FlowBook comparison...")
            log(f"Notebook: {notebook_path}")
            log(f"Cell timeout: {cell_timeout}s")
            if extra_cells > 0:
                log(f"Extra cells: {extra_cells}")
            log("")

            # Run baseline
            log("=" * 60)
            log("BASELINE EXECUTION (python3)")
            log("=" * 60)
            baseline_results = run_baseline_execution(notebook_content, cell_timeout, rerun_indices)
            log("")

            # Run FlowBook
            log("=" * 60)
            log("FLOWBOOK EXECUTION (flowbook_kernel)")
            log("=" * 60)
            flowbook_results = run_flowbook_execution(notebook_content, cell_timeout, rerun_indices)
            log("")

            # Build comparison result
            cells = notebook_content.get("cells", [])
            code_cells = [c for c in cells if c.get("cell_type") == "code"]

            comparison = ComparisonResult(
                version="1.0",
                notebook_path=str(notebook_path),
                timestamp=datetime.now().isoformat(),
                kernels={
                    "baseline": baseline_results,
                    "flowbook": flowbook_results,
                },
                metadata={
                    "num_cells": len(code_cells),
                    "timeout_seconds": cell_timeout,
                    "extra_cells": extra_cells,
                }
            )

            # Convert to dict for JSON serialization
            def to_dict(obj):
                if hasattr(obj, '__dict__'):
                    result = {}
                    for key, value in obj.__dict__.items():
                        result[key] = to_dict(value)
                    return result
                elif isinstance(obj, list):
                    return [to_dict(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: to_dict(v) for k, v in obj.items()}
                else:
                    return obj

            comparison_dict = to_dict(comparison)

            # Save JSON output in the same directory as the timings file
            from flowbook.util.output import output as global_output
            timings_dir = Path(global_output.timings_file).parent
            notebook_stem = Path(notebook_path).stem
            json_output_path = timings_dir / f"{notebook_stem}_comparison.json"

            with open(json_output_path, "w") as f:
                json.dump(comparison_dict, f, indent=2)

            log("=" * 60)
            log("SUMMARY")
            log("=" * 60)
            baseline_total = baseline_results.totals.get("cell_runtime_ms", 0)
            flowbook_runtime = flowbook_results.totals.get("cell_runtime_ms", 0)
            flowbook_state = flowbook_results.totals.get("state_duration_ms", 0)
            flowbook_check = flowbook_results.totals.get("check_duration_ms", 0)
            flowbook_total = flowbook_runtime + flowbook_state + flowbook_check

            if baseline_total > 0:
                slowdown = flowbook_total / baseline_total
                state_overhead_pct = (flowbook_state / baseline_total) * 100
                check_overhead_pct = (flowbook_check / baseline_total) * 100
            else:
                slowdown = 0.0
                state_overhead_pct = 0.0
                check_overhead_pct = 0.0

            log(f"Baseline runtime:     {baseline_total:,.1f}ms")
            log(f"FlowBook runtime:     {flowbook_runtime:,.1f}ms")
            log(f"FlowBook state time:  {flowbook_state:,.1f}ms ({state_overhead_pct:.1f}%)")
            log(f"FlowBook check time:  {flowbook_check:,.1f}ms ({check_overhead_pct:.1f}%)")
            log(f"FlowBook total:       {flowbook_total:,.1f}ms")
            log(f"Slowdown:             {slowdown:.2f}x")
            log("")
            log(f"Results saved to: {json_output_path}")
            log("")

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=notebook_content,
            metadata={
                "status": "success",
                "command": self.command_name,
                "comparison_file": str(json_output_path),
                "baseline_runtime_ms": baseline_total,
                "flowbook_runtime_ms": flowbook_runtime,
                "flowbook_state_ms": flowbook_state,
                "flowbook_check_ms": flowbook_check,
                "slowdown": slowdown,
            },
            total_cost=0.0,
            total_time=total_time,
        )
