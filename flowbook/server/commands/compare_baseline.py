"""
Compare baseline (python3) vs FlowBook kernel execution with timing and memory metrics.

4-Phase Execution:
  Phase 1: FlowBook timing - collect cell_runtime_ms, state_duration_ms, check_duration_ms
  Phase 2: Baseline timing - collect cell_runtime_ms (skipped by default, use --run-baseline)
  Phase 3: Baseline memory (HeapSizer) - collect namespace_size_mb (skipped by default)
  Phase 4: FlowBook memory (HeapSizer) - collect namespace_size_mb, checkpoint overhead

Memory measurement uses HeapSizer for accurate heap traversal with proper handling of:
- NumPy array views and shared data buffers
- Pandas DataFrame/Series Copy-on-Write sharing
- Object deduplication across shared references

Usage via CLI:
    flowbook compare-baseline notebook.ipynb                              # FlowBook only (default)
    flowbook compare-baseline notebook.ipynb --run-baseline               # Include baseline comparison
    flowbook compare-baseline notebook.ipynb --staleness-mode syntactic   # Use syntactic mode
    flowbook compare-baseline notebook.ipynb --df-subset-optimization     # Enable DF subset optimization
    flowbook compare-baseline notebook.ipynb --timeout 14400              # optional timeout (default: 4 hours)

OUTPUT JSON SCHEMA (version 2.0):
{
  "version": "2.0",
  "notebook_path": str,                    # Path to the notebook file
  "timestamp": str,                        # ISO format timestamp
  "scalene_available": bool,               # Whether HeapSizer was available
  "metadata": {
    "num_cells": int,                      # Number of code cells
    "timeout_seconds": float,              # Cell execution timeout
    "staleness_mode": str,                 # "semantic" or "syntactic"
    "df_subset_optimization": bool,        # Whether DF subset optimization is enabled
    "rerun_k": int,                        # Number of rerun iterations (optional)
    "trial": int,                          # Trial number (optional, for multi-trial runs)
    "num_trials": int                      # Total trials (optional)
  },
  "kernels": {
    "baseline": {                          # Only present if --run-baseline
      "kernel_name": "baseline_kernel",
      "timing": TimingResults,             # See below
      "memory": MemoryResults              # See below (optional)
    },
    "flowbook": {
      "kernel_name": "flowbook_kernel",
      "timing": TimingResults,
      "memory": MemoryResults              # Only present if HeapSizer available
    }
  }
}

TimingResults:
{
  "kernel_name": str,
  "cells": [TimingCellMetrics, ...],       # Initial execution cells
  "rerun_cells": [TimingCellMetrics, ...], # Rerun cells (if rerun_k > 0)
  "totals": {
    "total_runtime_ms": float,
    "state_overhead_ms": float,            # FlowBook only
    "check_overhead_ms": float             # FlowBook only
  }
}

TimingCellMetrics:
{
  "cell_id": str,
  "cell_index": int,
  "execute_duration_ms": float,            # Total execution time
  "code_duration_ms": float,               # User code time (FlowBook only)
  "state_duration_ms": float,              # Checkpoint time (FlowBook only)
  "check_duration_ms": float,              # Reproducibility check time (FlowBook only)
  "status": str,                           # "ok" or "error"
  "error": str | null,
  "is_rerun": bool,
  "checking_result": {                     # FlowBook only (optional)
    "cell_status": str,                    # "clean", "stale", or "error"
    "reasons": [dict, ...],
    "errors": [dict, ...]
  }
}

MemoryResults:
{
  "kernel_name": str,
  "cells": [MemoryCellMetrics, ...],
  "rerun_cells": [MemoryCellMetrics, ...],
  "totals": {
    "final_footprint_mb": float,
    "max_footprint_mb": float,
    "total_allocation_mb": float,
    "gpu_mem_samples": float,
    "base_namespace_mb": float,            # FlowBook only
    "total_overhead_mb": float             # FlowBook only
  }
}

MemoryCellMetrics:
{
  "cell_id": str,
  "cell_index": int,
  "current_footprint_mb": float,           # Total memory at end of cell
  "max_footprint_mb": float,
  "allocation_delta_mb": float,
  "gpu_mem_samples": float,
  "status": str,
  "error": str | null,
  "is_rerun": bool,
  # FlowBook-only fields:
  "checkpoint_var_costs": {var: {bytes, deepcopy_ms}, ...} | null,
  "overhead_breakdown": {                  # Memory breakdown by category
    "checkpoints_mb": float,               # Cumulative checkpoint memory
    "execution_records_mb": float,
    "tracking_metadata_mb": float,
    "other_mb": float
  } | null,
  "cumulative_by_type": {type_name: bytes, ...} | null,
  "cumulative_by_var": {var_name: bytes, ...} | null,
  "pre_only_bytes": int,                   # Pre-checkpoint size (for syntactic mode)
  "post_savings_bytes": int,               # Memory saved without post checkpoints
  "base_namespace_mb": float,              # TODO: Currently set to current_footprint_mb, should be
                                           #       current_footprint_mb - total_overhead to represent
                                           #       user namespace without FlowBook overhead
  "total_overhead_mb": float               # Sum of overhead_breakdown values
}
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
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
class CheckingResult:
    """Result of reproducibility checking for a cell."""
    cell_status: str  # "clean", "stale", or "error"
    reasons: List[Dict[str, Any]] = field(default_factory=list)  # List of reason dicts if stale
    errors: List[Dict[str, Any]] = field(default_factory=list)  # List of error dicts if error

    def to_dict(self) -> Dict[str, Any]:
        return {"cell_status": self.cell_status, "reasons": self.reasons, "errors": self.errors}


@dataclass
class TimingCellMetrics:
    """Timing metrics for a single cell execution (Scalene OFF)."""
    cell_id: str
    cell_index: int
    execute_duration_ms: float  # Total time in _do_execute_impl (FlowBook) or client timing (baseline)
    code_duration_ms: float  # Time for _ipython_do_execute (FlowBook only, 0 for baseline)
    state_duration_ms: float  # FlowBook only
    check_duration_ms: float  # FlowBook only
    status: str
    error: Optional[str] = None
    is_rerun: bool = False
    checking_result: Optional[CheckingResult] = None  # Reproducibility checking result


@dataclass
class MemoryCellMetrics:
    """Memory metrics for a single cell execution.

    Simplified structure with clear semantics:
    - namespace_mb: Size of user namespace (what the user's code uses)
    - checkpoint_delta_mb: What THIS cell's checkpoint adds (beyond ns + prior ckpts)
    - checkpoint_cumulative_mb: Total checkpoint overhead so far
    - gpu_mb: GPU memory usage
    - checkpoint_by_var: Per-variable breakdown of checkpoint memory
    """
    cell_id: str
    cell_index: int
    namespace_mb: float              # User namespace size
    checkpoint_delta_mb: float       # This cell's checkpoint contribution
    checkpoint_cumulative_mb: float  # Total checkpoint overhead so far
    gpu_mb: float                    # GPU memory
    checkpoint_by_var: Optional[Dict[str, float]] = None  # Per-variable MB (for memory plots)
    checkpoint_var_costs: Optional[Dict[str, Any]] = None  # Per-variable timing (for timing plots)
    status: str = "ok"
    error: Optional[str] = None
    is_rerun: bool = False


@dataclass
class TimingResults:
    """Timing results from a notebook execution (Scalene OFF)."""
    kernel_name: str
    cells: List[TimingCellMetrics] = field(default_factory=list)
    rerun_cells: List[TimingCellMetrics] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryResults:
    """Memory results from a notebook execution (Scalene ON)."""
    kernel_name: str
    cells: List[MemoryCellMetrics] = field(default_factory=list)
    rerun_cells: List[MemoryCellMetrics] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelResults:
    """Combined timing and memory results for a kernel."""
    kernel_name: str
    timing: Optional[TimingResults] = None
    memory: Optional[MemoryResults] = None


@dataclass
class ComparisonResult:
    """Complete comparison result with 4-phase execution."""
    version: str = "2.0"
    notebook_path: str = ""
    timestamp: str = ""
    kernels: Dict[str, KernelResults] = field(default_factory=dict)
    scalene_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _is_heapsizer_available() -> bool:
    """Check if HeapSizer is available."""
    try:
        from flowbook.kernel_support.heap_size import HeapSizer
        return True
    except ImportError:
        return False


def create_baseline_kernel() -> Tuple[KernelManager, BlockingKernelClient]:
    """
    Start the baseline kernel.

    Returns:
        Tuple of (KernelManager, BlockingKernelClient)
    """
    # Import to ensure kernels are registered
    from flowbook.baseline_kernel import (
        install_baseline_kernel,
    )
    try:
        install_baseline_kernel()
    except Exception:
        pass

    kernel_name = "baseline_kernel"

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

            kernel_manager = KernelManager(kernel_name=kernel_name)
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
                    log(traceback.format_exc())
                    time.sleep(0.5)

            return kernel_manager, kernel_client

        except Exception as e:
            log(f"Error on attempt {attempt + 1}/{max_attempts}: {e}")
            log(traceback.format_exc())
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


def create_flowbook_kernel(
    extra_env: Optional[Dict[str, str]] = None
) -> Tuple[KernelManager, FlowbookKernelClient]:
    """
    Start a flowbook_kernel.

    Args:
        extra_env: Optional dict of extra environment variables to set for the kernel process.

    Returns:
        Tuple of (KernelManager, FlowbookKernelClient)
    """
    make_kernels()

    kernel_name = "flowbook_kernel"

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

            kernel_manager = KernelManager(kernel_name=kernel_name)
            kernel_manager.start_kernel(extra_env=extra_env)

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
                    log(traceback.format_exc())
                    time.sleep(0.5)

            return kernel_manager, kernel_client

        except Exception as e:
            log(f"Error on attempt {attempt + 1}/{max_attempts}: {e}")
            log(traceback.format_exc())
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
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Execute a cell on the baseline kernel and measure execution time.

    Returns:
        Dict with cell_runtime_ms (kernel-reported) and optional error
    """
    start = time.perf_counter()

    msg_id = kernel_client.execute(source)

    baseline_metadata = None
    error_msg = None
    start_time = time.time()

    while True:
        if timeout is not None and time.time() - start_time > timeout:
            return {"cell_runtime_ms": None, "error": f"Timeout after {timeout}s"}

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        # Look for display_data with baseline metadata (kernel-reported timing)
        if msg_type == "display_data":
            output_meta = msg.get("content", {}).get("metadata", {})
            if "baseline" in output_meta:
                baseline_metadata = output_meta["baseline"]

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

    # Fallback to client-side timing if kernel metadata not available
    elapsed = time.perf_counter() - start

    if baseline_metadata:
        # Use kernel-reported timing for fair comparison with FlowBook
        return {
            "cell_runtime_ms": baseline_metadata.get("code_duration_ms", elapsed * 1000),
            "error": error_msg
        }
    else:
        # Fallback to client-side timing (should only happen on old kernels)
        return {"cell_runtime_ms": elapsed * 1000, "error": error_msg}


def execute_cell_flowbook(
    kernel_client: FlowbookKernelClient,
    source: str,
    cell_id: str,
    cell_order: List[str],
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Execute a cell on the flowbook_kernel and measure execution time.

    Returns:
        Dict with execute_duration_ms, code_duration_ms, state_duration_ms, check_duration_ms, and optional error
    """
    # Set cell order for reproducibility tracking
    kernel_client.set_cell_order(cell_order)

    # Measure wall-clock time from client side (same as baseline)
    start = time.perf_counter()

    # Pass timeout to kernel via cell_metadata so kernel respects it
    cell_meta = {"timeout": timeout} if timeout else None
    msg_id = kernel_client.execute(source, cell_id=cell_id, cell_metadata=cell_meta)

    flowbook_metadata = None
    predicate_violations = []  # Collect all predicate violations (even when continue_after_violation=True)
    error_msg = None
    start_time = time.time()

    while True:
        if timeout is not None and time.time() - start_time > timeout:
            return {
                "execute_duration_ms": None,
                "code_duration_ms": None,
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

        # Look for display_data with flowbook metadata or predicate violations
        if msg_type == "display_data":
            output_meta = msg.get("content", {}).get("metadata", {})
            if "flowbook" in output_meta:
                flowbook_metadata = output_meta["flowbook"]
            # Capture predicate violations (sent even when continue_after_violation=True)
            if "predicate_violation" in output_meta:
                predicate_violations.append(output_meta["predicate_violation"])

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

    # Calculate client-side elapsed time (same methodology as baseline)
    elapsed = time.perf_counter() - start

    # Build violation message from predicate_violations or legacy violation field
    violation_msg = None
    if predicate_violations:
        # Format predicate violations as messages
        msgs = []
        for pv in predicate_violations:
            predicate = pv.get("predicate", "unknown")
            locations = pv.get("locations", [])
            accepted = pv.get("accepted", False)
            status = "accepted" if accepted else "rejected"
            locs_str = ", ".join(locations) if locations else "unknown"
            msgs.append(f"{predicate}: {locs_str} ({status})")
        violation_msg = "; ".join(msgs)

    if flowbook_metadata:
        # Also check for legacy violations in flowbook metadata
        if not violation_msg:
            violation = flowbook_metadata.get("violation")
            if violation:
                violation_msg = violation.get("message", "Reproducibility violation")

        # Extract checking result for this cell
        staleness_reasons = flowbook_metadata.get("staleness_reasons", {})
        cell_reasons = staleness_reasons.get(cell_id, [])

        # Convert predicate_violations to error dicts for checking_result
        cell_errors = []
        for pv in predicate_violations:
            cell_errors.append({
                "error_type": pv.get("predicate", "unknown"),
                "locations": pv.get("locations", []),
                "causer_cell": pv.get("causer_cell"),
                "accepted": pv.get("accepted", False),
            })

        # Determine cell status: error > stale > clean
        if cell_errors:
            cell_status = "error"
        elif cell_reasons:
            cell_status = "stale"
        else:
            cell_status = "clean"

        return {
            # Use kernel-reported timing for accuracy
            "execute_duration_ms": flowbook_metadata.get("execute_duration_ms", elapsed * 1000),
            "code_duration_ms": flowbook_metadata.get("code_duration_ms", 0.0),
            "state_duration_ms": flowbook_metadata.get("state_duration_ms", 0.0),
            "check_duration_ms": flowbook_metadata.get("check_duration_ms", 0.0),
            "error": error_msg,
            "violation": violation_msg,
            "predicate_violations": predicate_violations,  # Include full violation details
            "stale_cells": flowbook_metadata.get("stale_cells", []),
            "checking_result": {
                "cell_status": cell_status,
                "reasons": cell_reasons,
                "errors": cell_errors,
            },
        }
    else:
        return {
            "execute_duration_ms": elapsed * 1000,
            "code_duration_ms": 0.0,
            "state_duration_ms": None,
            "check_duration_ms": None,
            "error": error_msg or "No flowbook metadata received",
            "violation": violation_msg,  # May have predicate violations even without flowbook metadata
            "predicate_violations": predicate_violations,
            "stale_cells": [],
            "checking_result": None,
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


def get_namespace_size(kernel_client, timeout: float = 30.0) -> Dict[str, Any]:
    """Get user namespace memory size via HeapSizer.

    Uses empty code + user_expressions to avoid creating checkpoints in FlowBook kernel.

    Returns dict with:
        - total_bytes: Total memory used by user namespace
        - total_mb: Total memory in MB
        - by_variable: Dict mapping variable name to bytes
        - by_type: Dict mapping type name to bytes
    """
    # Use a single execute with user_expressions to get namespace size
    # Filter out private/system variables and callables
    expr_code = """(lambda: (
        __import__('flowbook.kernel_support.heap_size', fromlist=['HeapSizer'])
        .HeapSizer()
        .sizeof_namespace(
            {k: v for k, v in globals().items()
             if not k.startswith('_') and not callable(v) and not isinstance(v, type(__builtins__))}
        ).__dict__
    ))()"""

    msg_id = kernel_client.execute(
        '',  # Empty code - no checkpoints created
        user_expressions={'_ns_size': expr_code},
        silent=True,
    )

    # Wait for idle on iopub channel
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {"total_bytes": 0, "total_mb": 0.0, "by_variable": {}, "by_type": {}}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            user_exprs = reply['content'].get('user_expressions', {})
            expr = user_exprs.get('_ns_size', {})
            if expr.get('status') == 'ok':
                text = expr['data']['text/plain']
                result = ast.literal_eval(text)
                total_bytes = result.get('total_bytes', 0)
                total_mb = total_bytes / (1024 * 1024)
                log(f"Namespace size: {total_mb:.2f}MB ({len(result.get('by_variable', {}))} variables)")
                return {
                    "total_bytes": total_bytes,
                    "total_mb": total_mb,
                    "by_variable": result.get('by_variable', {}),
                    "by_type": result.get('by_type', {}),
                }
            else:
                log(f"Namespace size error: {expr.get('evalue', 'unknown')}")
                break
    except Exception as e:
        log(f"Failed to get namespace size: {e}")
        log(traceback.format_exc())

    return {"total_bytes": 0, "total_mb": 0.0, "by_variable": {}, "by_type": {}}


def get_kernel_gpu_memory_mb(kernel_client, timeout: float = 10.0) -> float:
    """Get GPU memory usage from the kernel process.

    GPU memory must be queried from the kernel process because that's where
    cuDF/RAPIDS allocations happen. The server process won't see kernel GPU usage.

    Args:
        kernel_client: Kernel client to execute code on
        timeout: Timeout in seconds

    Returns:
        GPU memory in MB, or 0.0 if unavailable
    """
    # Query GPU memory from kernel process using pynvml
    expr_code = """(lambda: (
        __import__('flowbook.util.gpu_memory', fromlist=['get_gpu_memory_mb'])
        .get_gpu_memory_mb()
    ))()"""

    msg_id = kernel_client.execute(
        '',  # Empty code - no side effects
        user_expressions={'_gpu_mem': expr_code},
        silent=True,
    )

    # Wait for idle on iopub channel
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return 0.0
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply
    try:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            user_exprs = reply['content'].get('user_expressions', {})
            expr = user_exprs.get('_gpu_mem', {})
            if expr.get('status') == 'ok':
                text = expr['data']['text/plain']
                return float(text)
            break
    except Exception:
        pass

    return 0.0


def get_flowbook_checkpoint_var_costs(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get per-variable checkpoint memory costs from FlowBook kernel.

    Gets combined costs from both pre and post checkpoints for a cell.
    This gives the total memory overhead of checkpointing for that cell.

    IMPORTANT: Uses user_expressions with empty code to avoid creating
    additional checkpoints. The FlowBook kernel creates pre/post checkpoints
    for every execute() call, so we must minimize kernel executions.

    Args:
        kernel_client: Kernel client to execute code on
        cell_id: Cell ID to get checkpoint costs for
        timeout: Timeout in seconds

    Returns:
        Dict with variable name -> {bytes, type, module, pre_bytes, post_bytes}
    """
    # Use a single user_expressions call with empty code to avoid checkpointing.
    # The expression imports and calls the method inline.
    expr = (
        f"(__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_cell_checkpoint_costs('{cell_id}') "
        f"if hasattr(__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints, '_instance') and "
        f"__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance else {{}})"
    )

    msg_id = kernel_client.execute(
        '',  # Empty code - goes through _execute_without_enforcer, no checkpoints
        user_expressions={'_costs': expr},
        silent=True,
    )

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply, matching by msg_id
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            expr_result = reply['content'].get('user_expressions', {}).get('_costs', {})
            if expr_result.get('status') == 'ok':
                text = expr_result['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get checkpoint var costs: {e}")
        log(traceback.format_exc())

    return {}


def get_flowbook_cumulative_checkpoint_size(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get cumulative checkpoint size for all checkpoints up to and including a cell.

    This is the CORRECT way to measure checkpoint memory because it measures all
    checkpoints together, properly accounting for memory sharing between checkpoints.
    Summing individual checkpoint sizes would overcount shared memory.

    Args:
        kernel_client: Kernel client to execute code on
        cell_id: Cell ID to get cumulative checkpoint size for
        timeout: Timeout in seconds

    Returns:
        Dict with:
        - total_bytes: Deduplicated total memory
        - by_variable: Memory by variable name
        - by_type: Memory by type name
        - by_checkpoint: Memory contribution per checkpoint
    """
    # Use user_expressions with empty code to avoid creating additional checkpoints
    expr = (
        f"(lambda: "
        f"dict("
        f"total_bytes=r.total_bytes, "
        f"by_variable=dict(r.by_variable), "
        f"by_type=dict(r.by_type), "
        f"by_checkpoint=dict(r.by_checkpoint)"
        f") "
        f"if (r := __import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_cumulative_checkpoint_size_at_cell('{cell_id}')) "
        f"else {{'total_bytes': 0, 'by_variable': {{}}, 'by_type': {{}}, 'by_checkpoint': {{}}}})()"
    )

    msg_id = kernel_client.execute(
        '',  # Empty code - no checkpoints created
        user_expressions={'_cumulative_size': expr},
        silent=True,
    )

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            expr_result = reply['content'].get('user_expressions', {}).get('_cumulative_size', {})
            if expr_result.get('status') == 'ok':
                text = expr_result['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get cumulative checkpoint size: {e}")
        log(traceback.format_exc())

    return {}


def get_checkpoint_overhead(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get checkpoint overhead beyond namespace (correctly handles CoW sharing).

    This is the CORRECT way to measure checkpoint memory overhead because it:
    1. Measures namespace first (marks objects as seen)
    2. Measures checkpoints cumulatively (only NEW objects counted)

    This properly handles Copy-on-Write sharing between checkpoints and
    the namespace - shared memory is not double-counted.

    Args:
        kernel_client: Kernel client to execute code on
        cell_id: Cell ID to measure up to (inclusive)
        timeout: Timeout in seconds

    Returns:
        Dict with:
        - total_mb: Total checkpoint memory beyond namespace
        - by_checkpoint: Per-checkpoint delta in MB
        - by_variable: Per-variable totals in MB
        - cumulative: Running total at each checkpoint in MB
    """
    # Filter namespace same way as sizeof_namespace (exclude private, callable, modules)
    ns_filter = "{k: v for k, v in globals().items() if not k.startswith('_') and not callable(v) and not isinstance(v, type(__builtins__))}"

    expr = (
        f"__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_overhead_beyond_namespace('{cell_id}', {ns_filter})"
    )

    msg_id = kernel_client.execute('', user_expressions={'_overhead': expr}, silent=True)

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {'total_mb': 0, 'by_checkpoint': {}, 'by_variable': {}, 'cumulative': {}}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            expr_result = reply['content'].get('user_expressions', {}).get('_overhead', {})
            if expr_result.get('status') == 'ok':
                text = expr_result['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get checkpoint overhead: {e}")
        log(traceback.format_exc())

    return {'total_mb': 0, 'by_checkpoint': {}, 'by_variable': {}, 'cumulative': {}}


def get_flowbook_pre_post_checkpoint_sizes(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get separate pre and post checkpoint sizes for a cell.

    This enables analyzing how much memory would be saved by eliminating
    post checkpoints (NO_POST plan).

    Args:
        kernel_client: Kernel client to execute code on
        cell_id: Cell ID to get checkpoint sizes for
        timeout: Timeout in seconds

    Returns:
        Dict with:
        - pre_bytes: Total bytes in _pre_* checkpoints only
        - post_bytes: Total bytes in _post_* checkpoints only
        - total_bytes: Total bytes in all checkpoints
        - pre_checkpoint_count: Number of pre checkpoints
        - post_checkpoint_count: Number of post checkpoints
    """
    expr = (
        f"(__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_pre_post_checkpoint_sizes_at_cell('{cell_id}') "
        f"if hasattr(__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints, '_instance') and "
        f"__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance else {{'pre_bytes': 0, 'post_bytes': 0, 'total_bytes': 0, "
        f"'pre_checkpoint_count': 0, 'post_checkpoint_count': 0}})"
    )

    msg_id = kernel_client.execute(
        '',  # Empty code - no checkpoints created
        user_expressions={'_pre_post_sizes': expr},
        silent=True,
    )

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            expr_result = reply['content'].get('user_expressions', {}).get('_pre_post_sizes', {})
            if expr_result.get('status') == 'ok':
                text = expr_result['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get pre/post checkpoint sizes: {e}")
        log(traceback.format_exc())

    return {}


def get_flowbook_overhead_breakdown(
    kernel_client, timeout: float = 30.0
) -> Dict[str, float]:
    """Get memory overhead breakdown by category from FlowBook kernel.

    Gets checkpoint storage, execution records, tracking metadata, and other
    overhead categories in megabytes.

    Args:
        kernel_client: Kernel client to execute code on
        timeout: Timeout in seconds

    Returns:
        Dict with category -> MB:
        {
            'checkpoints_mb': float,
            'execution_records_mb': float,
            'tracking_metadata_mb': float,
            'other_mb': float,
        }
    """
    code = """
import sys
_overhead_breakdown = {
    'checkpoints_mb': 0.0,
    'execution_records_mb': 0.0,
    'tracking_metadata_mb': 0.0,
    'other_mb': 0.0,
}

# Get checkpoint overhead from MemoryCheckpoints
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
if hasattr(MemoryCheckpoints, '_instance') and MemoryCheckpoints._instance:
    mc = MemoryCheckpoints._instance
    # Use get_total_checkpoint_size() to measure actual storage via HeapSizer
    all_sizes = mc.get_total_checkpoint_size()
    _overhead_breakdown['checkpoints_mb'] = all_sizes.total_bytes / (1024 * 1024)
    # Estimate tracking metadata: ~200 bytes per variable entry
    num_vars = sum(len(ckpt.user_ns) for ckpt in mc.saved.values())
    _overhead_breakdown['tracking_metadata_mb'] = (num_vars * 200) / (1024 * 1024)
    # Other overhead: ~1KB per checkpoint
    _overhead_breakdown['other_mb'] = len(mc.saved) * 1024 / (1024 * 1024)

# Get execution records overhead from ReproducibilityEnforcer (if accessible)
try:
    # The enforcer is stored in the kernel's namespace
    if '_flowbook_enforcer' in dir():
        enforcer = _flowbook_enforcer
        if hasattr(enforcer, 'get_execution_records_size'):
            _overhead_breakdown['execution_records_mb'] = enforcer.get_execution_records_size() / (1024 * 1024)
except Exception:
    pass
"""
    msg_id = kernel_client.execute(code, silent=True)
    _wait_for_idle(kernel_client, timeout)

    msg_id = kernel_client.execute(
        '',
        user_expressions={'_breakdown': '_overhead_breakdown'},
        silent=True,
    )

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply, matching by msg_id
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            if reply['parent_header'].get('msg_id') != msg_id:
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                continue

            expr = reply['content'].get('user_expressions', {}).get('_breakdown', {})
            if expr.get('status') == 'ok':
                text = expr['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get overhead breakdown: {e}")
        log(traceback.format_exc())

    return {}


def run_baseline_timing(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    rerun_k: int = 0,
) -> TimingResults:
    """
    Run notebook on baseline kernel and collect TIMING metrics only (Scalene OFF).

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds

    Returns:
        TimingResults with cell timing data
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    log(f"Baseline Timing: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = TimingResults(kernel_name="baseline_kernel")

    try:
        log("Baseline Timing: Starting baseline_kernel...")
        kernel_manager, kernel_client = create_baseline_kernel()
        log("Baseline Timing: Kernel ready")

        # Warmup: pre-import pandas to match FlowBook kernel's startup state
        warmup_code = "import pandas"
        kernel_client.execute(warmup_code, silent=True)
        _wait_for_idle(kernel_client)
        log("Baseline Timing: Warmup imports completed")

        total_runtime_ms = 0.0

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"Baseline Timing: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            timing = execute_cell_baseline(kernel_client, source, cell_timeout)

            if timing.get("error"):
                log(f"  Error:\n{timing['error']}")
                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    execute_duration_ms=0.0,
                    code_duration_ms=0.0,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    status="error",
                    error=timing["error"],
                    checking_result=None,  # Baseline has no checking
                ))
            else:
                runtime_ms = timing["cell_runtime_ms"]
                total_runtime_ms += runtime_ms

                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    execute_duration_ms=runtime_ms,
                    code_duration_ms=runtime_ms,  # For baseline, code time equals total time
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    status="ok",
                    checking_result=None,  # Baseline has no checking
                ))
                log(f"  Runtime: {runtime_ms:.1f}ms")

        # Execute reruns: run all cells k extra times (top-to-bottom)
        rerun_runtime_ms = 0.0
        if rerun_k > 0:
            total_rerun_cells = rerun_k * len(code_cells)
            log(f"Baseline Timing: Executing {rerun_k} rerun pass(es) ({total_rerun_cells} cell executions)...")
            rerun_count = 0
            for pass_num in range(rerun_k):
                log(f"Baseline Timing: Rerun pass {pass_num + 1}/{rerun_k}...")
                for idx, cell in enumerate(code_cells):
                    cell_id = cell.get("id", f"cell_{idx}")
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    if not source.strip():
                        continue

                    rerun_count += 1
                    log(f"Baseline Timing: Rerun {rerun_count}/{total_rerun_cells} - pass {pass_num+1}, cell {idx+1} ({cell_id})...")

                    timing = execute_cell_baseline(kernel_client, source, cell_timeout)

                    if timing.get("error"):
                        log(f"  Rerun Error:\n{timing['error']}")
                        results.rerun_cells.append(TimingCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            execute_duration_ms=0.0,
                            code_duration_ms=0.0,
                            state_duration_ms=0.0,
                            check_duration_ms=0.0,
                            status="error",
                            error=timing["error"],
                            is_rerun=True,
                            checking_result=None,  # Baseline has no checking
                        ))
                    else:
                        runtime_ms = timing["cell_runtime_ms"]
                        rerun_runtime_ms += runtime_ms

                        results.rerun_cells.append(TimingCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            execute_duration_ms=runtime_ms,
                            code_duration_ms=runtime_ms,  # For baseline, code time equals total time
                            state_duration_ms=0.0,
                            check_duration_ms=0.0,
                            status="ok",
                            is_rerun=True,
                            checking_result=None,  # Baseline has no checking
                        ))
                        log(f"  Rerun Runtime: {runtime_ms:.1f}ms")

        results.totals = {
            "execute_duration_ms": total_runtime_ms,
            "code_duration_ms": total_runtime_ms,  # For baseline, code time equals total time
            "rerun_runtime_ms": rerun_runtime_ms,
        }

        log(f"Baseline Timing: Total runtime {total_runtime_ms:.1f}ms, Rerun runtime {rerun_runtime_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_timing(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    rerun_k: int = 0,
    staleness_mode: str = "semantic",
    df_subset_optimization: bool = False,
) -> TimingResults:
    """
    Run notebook on FlowBook kernel and collect TIMING metrics only (Scalene OFF).

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        rerun_k: Number of rerun iterations
        staleness_mode: Staleness computation mode ('syntactic' or 'semantic')
        df_subset_optimization: Enable DataFrame subset optimization for checkpoints

    Returns:
        TimingResults with cell timing data including state_duration_ms, check_duration_ms
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cell_order = [c.get("id", f"cell_{i}") for i, c in enumerate(code_cells)]

    log(f"FlowBook Timing: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = TimingResults(kernel_name="flowbook_kernel")

    try:
        log("FlowBook Timing: Starting flowbook_kernel...")
        kernel_manager, kernel_client = create_flowbook_kernel()
        log("FlowBook Timing: Kernel ready")

        # Enable continue_after_violation so we can measure full execution even with repro issues
        kernel_client.execute("%continue_after_violation on", silent=True)
        _wait_for_idle(kernel_client)
        log("FlowBook Timing: continue_after_violation enabled")

        # Set staleness computation mode
        kernel_client.execute(f"%staleness_mode {staleness_mode}", silent=True)
        _wait_for_idle(kernel_client)
        log(f"FlowBook Timing: staleness_mode set to {staleness_mode}")

        # Enable DataFrame subset optimization if requested
        if df_subset_optimization:
            kernel_client.execute("%df_subset_checkpoints on", silent=True)
            _wait_for_idle(kernel_client)
            log("FlowBook Timing: df_subset_checkpoints enabled")

        # Run a warm-up cell to trigger lazy initialization (cudf import, tracking patches, etc.)
        # This ensures the overhead doesn't appear in the first real cell
        log("FlowBook Timing: Running warm-up cell...")
        warmup_result = execute_cell_flowbook(
            kernel_client,
            "_warmup_var_ = 1; del _warmup_var_",  # Trigger tracking pipeline, then clean up
            "_warmup_",
            ["_warmup_"],
            timeout=60.0
        )
        log(f"FlowBook Timing: Warm-up complete (took {warmup_result.get('execute_duration_ms', 0):.0f}ms)")

        total_execute_ms = 0.0
        total_code_ms = 0.0
        total_state_ms = 0.0
        total_check_ms = 0.0

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"FlowBook Timing: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            timing = execute_cell_flowbook(
                kernel_client, source, cell_id, cell_order, cell_timeout
            )

            if timing.get("error") and timing.get("execute_duration_ms") is None:
                log(f"  Error:\n{timing['error']}")
                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    execute_duration_ms=0.0,
                    code_duration_ms=0.0,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    status="error",
                    error=timing["error"],
                    checking_result=None,
                ))
            else:
                if timing.get("violation"):
                    log(f"  Violation: {timing['violation']}")

                execute_ms = timing["execute_duration_ms"] or 0.0
                code_ms = timing["code_duration_ms"] or 0.0
                state_ms = timing["state_duration_ms"] or 0.0
                check_ms = timing["check_duration_ms"] or 0.0

                total_execute_ms += execute_ms
                total_code_ms += code_ms
                total_state_ms += state_ms
                total_check_ms += check_ms

                status = "ok"
                if timing.get("error"):
                    status = "error"
                elif timing.get("violation"):
                    status = "violation"

                # Build checking result
                checking_result_data = timing.get("checking_result")
                checking_result = None
                if checking_result_data:
                    checking_result = CheckingResult(
                        cell_status=checking_result_data.get("cell_status", "unknown"),
                        reasons=checking_result_data.get("reasons", []),
                        errors=checking_result_data.get("errors", []),
                    )

                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    execute_duration_ms=execute_ms,
                    code_duration_ms=code_ms,
                    state_duration_ms=state_ms,
                    check_duration_ms=check_ms,
                    status=status,
                    error=timing.get("violation") or timing.get("error"),
                    checking_result=checking_result,
                ))
                log(f"  Execute: {execute_ms:.1f}ms, Code: {code_ms:.1f}ms, State: {state_ms:.1f}ms, Check: {check_ms:.1f}ms")

        # Execute reruns: run all cells k extra times (top-to-bottom)
        rerun_execute_ms = 0.0
        rerun_code_ms = 0.0
        rerun_state_ms = 0.0
        rerun_check_ms = 0.0
        if rerun_k > 0:
            total_rerun_cells = rerun_k * len(code_cells)
            log(f"FlowBook Timing: Executing {rerun_k} rerun pass(es) ({total_rerun_cells} cell executions)...")
            rerun_count = 0
            for pass_num in range(rerun_k):
                log(f"FlowBook Timing: Rerun pass {pass_num + 1}/{rerun_k}...")
                for idx, cell in enumerate(code_cells):
                    cell_id = cell.get("id", f"cell_{idx}")
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    if not source.strip():
                        continue

                    rerun_count += 1
                    log(f"FlowBook Timing: Rerun {rerun_count}/{total_rerun_cells} - pass {pass_num+1}, cell {idx+1} ({cell_id})...")

                    timing = execute_cell_flowbook(
                        kernel_client, source, cell_id, cell_order, cell_timeout
                    )

                    if timing.get("error") and timing.get("execute_duration_ms") is None:
                        log(f"  Rerun Error:\n{timing['error']}")
                        results.rerun_cells.append(TimingCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            execute_duration_ms=0.0,
                            code_duration_ms=0.0,
                            state_duration_ms=0.0,
                            check_duration_ms=0.0,
                            status="error",
                            error=timing["error"],
                            is_rerun=True,
                            checking_result=None,
                        ))
                    else:
                        if timing.get("violation"):
                            log(f"  Rerun Violation: {timing['violation']}")

                        execute_ms = timing["execute_duration_ms"] or 0.0
                        code_ms = timing["code_duration_ms"] or 0.0
                        state_ms = timing["state_duration_ms"] or 0.0
                        check_ms = timing["check_duration_ms"] or 0.0

                        rerun_execute_ms += execute_ms
                        rerun_code_ms += code_ms
                        rerun_state_ms += state_ms
                        rerun_check_ms += check_ms

                        status = "ok"
                        if timing.get("error"):
                            status = "error"
                        elif timing.get("violation"):
                            status = "violation"

                        # Build checking result
                        checking_result_data = timing.get("checking_result")
                        checking_result = None
                        if checking_result_data:
                            checking_result = CheckingResult(
                                cell_status=checking_result_data.get("cell_status", "unknown"),
                                reasons=checking_result_data.get("reasons", []),
                                errors=checking_result_data.get("errors", []),
                            )

                        results.rerun_cells.append(TimingCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            execute_duration_ms=execute_ms,
                            code_duration_ms=code_ms,
                            state_duration_ms=state_ms,
                            check_duration_ms=check_ms,
                            status=status,
                            error=timing.get("violation") or timing.get("error"),
                            is_rerun=True,
                            checking_result=checking_result,
                        ))
                        log(f"  Rerun Execute: {execute_ms:.1f}ms, Code: {code_ms:.1f}ms, State: {state_ms:.1f}ms, Check: {check_ms:.1f}ms")

        # Compute checking summary (exclude never_executed cells - they are empty code cells)
        clean_count = 0
        stale_count = 0
        error_count = 0
        reason_counts: Dict[str, int] = {}
        error_counts: Dict[str, int] = {}
        for cell in results.cells:
            if cell.checking_result:
                # Skip cells that only have never_executed reason (empty code cells)
                reasons = cell.checking_result.reasons
                errors = cell.checking_result.errors
                if reasons and all(r.get("type") == "never_executed" for r in reasons) and not errors:
                    continue
                if cell.checking_result.cell_status == "clean":
                    clean_count += 1
                elif cell.checking_result.cell_status == "error":
                    error_count += 1
                    for error in errors:
                        etype = error.get("error_type", "unknown")
                        error_counts[etype] = error_counts.get(etype, 0) + 1
                else:
                    stale_count += 1
                    for reason in reasons:
                        rtype = reason.get("type", "unknown")
                        if rtype != "never_executed":  # Don't count never_executed
                            reason_counts[rtype] = reason_counts.get(rtype, 0) + 1

        results.totals = {
            "execute_duration_ms": total_execute_ms,
            "code_duration_ms": total_code_ms,
            "state_duration_ms": total_state_ms,
            "check_duration_ms": total_check_ms,
            "rerun_execute_ms": rerun_execute_ms,
            "rerun_code_ms": rerun_code_ms,
            "rerun_state_ms": rerun_state_ms,
            "rerun_check_ms": rerun_check_ms,
            "checking_summary": {
                "clean_cells": clean_count,
                "stale_cells": stale_count,
                "error_cells": error_count,
                "reason_counts": reason_counts,
                "error_counts": error_counts,
            },
        }

        log(f"FlowBook Timing: Total execute {total_execute_ms:.1f}ms, code {total_code_ms:.1f}ms, state {total_state_ms:.1f}ms, check {total_check_ms:.1f}ms")
        if rerun_k > 0:
            log(f"FlowBook Timing: Rerun execute {rerun_execute_ms:.1f}ms, code {rerun_code_ms:.1f}ms, state {rerun_state_ms:.1f}ms, check {rerun_check_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_baseline_memory(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    rerun_k: int = 0,
) -> MemoryResults:
    """
    Run notebook on baseline kernel and collect MEMORY metrics using HeapSizer.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds

    Returns:
        MemoryResults with cell memory data from HeapSizer
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    log(f"Baseline Memory: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = MemoryResults(kernel_name="baseline_kernel")

    try:
        log("Baseline Memory: Starting baseline_kernel...")
        kernel_manager, kernel_client = create_baseline_kernel()
        log("Baseline Memory: Kernel ready")

        # Warmup
        warmup_code = "import pandas"
        kernel_client.execute(warmup_code, silent=True)
        _wait_for_idle(kernel_client)

        # Get initial namespace size
        before_stats = get_namespace_size(kernel_client)
        max_footprint_mb = 0.0

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"Baseline Memory: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            # Get namespace size before cell
            pre_stats = get_namespace_size(kernel_client)

            timing = execute_cell_baseline(kernel_client, source, cell_timeout)

            # Get namespace size after cell
            post_stats = get_namespace_size(kernel_client)

            current_mb = post_stats.get("total_mb", 0.0)
            max_footprint_mb = max(max_footprint_mb, current_mb)

            if timing.get("error"):
                log(f"  Error:\n{timing['error']}")
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    namespace_mb=0.0,
                    checkpoint_delta_mb=0.0,
                    checkpoint_cumulative_mb=0.0,
                    gpu_mb=get_kernel_gpu_memory_mb(kernel_client),
                    status="error",
                    error=timing["error"]
                ))
            else:
                gpu_mem = get_kernel_gpu_memory_mb(kernel_client)
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    namespace_mb=current_mb,
                    checkpoint_delta_mb=0.0,  # Baseline has no checkpoints
                    checkpoint_cumulative_mb=0.0,
                    gpu_mb=gpu_mem,
                    status="ok",
                ))
                log(f"  Namespace: {current_mb:.1f}MB")

        # Execute reruns: run all cells k extra times (top-to-bottom)
        if rerun_k > 0:
            total_rerun_cells = rerun_k * len(code_cells)
            log(f"Baseline Memory: Executing {rerun_k} rerun pass(es) ({total_rerun_cells} cell executions)...")
            rerun_count = 0
            for pass_num in range(rerun_k):
                log(f"Baseline Memory: Rerun pass {pass_num + 1}/{rerun_k}...")
                for idx, cell in enumerate(code_cells):
                    cell_id = cell.get("id", f"cell_{idx}")
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    if not source.strip():
                        continue

                    rerun_count += 1
                    log(f"Baseline Memory: Rerun {rerun_count}/{total_rerun_cells} - pass {pass_num+1}, cell {idx+1} ({cell_id})...")

                    # Get namespace size before cell
                    pre_stats = get_namespace_size(kernel_client)

                    timing = execute_cell_baseline(kernel_client, source, cell_timeout)

                    # Get namespace size after cell
                    post_stats = get_namespace_size(kernel_client)

                    current_mb = post_stats.get("total_mb", 0.0)
                    max_footprint_mb = max(max_footprint_mb, current_mb)

                    if timing.get("error"):
                        log(f"  Rerun Error:\n{timing['error']}")
                        results.rerun_cells.append(MemoryCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            namespace_mb=0.0,
                            checkpoint_delta_mb=0.0,
                            checkpoint_cumulative_mb=0.0,
                            gpu_mb=get_kernel_gpu_memory_mb(kernel_client),
                            status="error",
                            error=timing["error"],
                            is_rerun=True,
                        ))
                    else:
                        gpu_mem = get_kernel_gpu_memory_mb(kernel_client)
                        results.rerun_cells.append(MemoryCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            namespace_mb=current_mb,
                            checkpoint_delta_mb=0.0,
                            checkpoint_cumulative_mb=0.0,
                            gpu_mb=gpu_mem,
                            status="ok",
                            is_rerun=True,
                        ))
                        log(f"  Rerun Namespace: {current_mb:.1f}MB, Delta: {allocation_delta:.1f}MB")

        # Get final stats
        final_stats = get_namespace_size(kernel_client)
        final_gpu_mem = get_kernel_gpu_memory_mb(kernel_client)
        results.totals = {
            "final_footprint_mb": final_stats.get("total_mb", 0.0),
            "max_footprint_mb": max_footprint_mb,
            "total_allocation_mb": final_stats.get("total_mb", 0.0) - before_stats.get("total_mb", 0.0),
            "gpu_mem_samples": final_gpu_mem,
        }

        log(f"Baseline Memory: Final namespace {final_stats.get('total_mb', 0):.1f}MB, "
            f"Max {max_footprint_mb:.1f}MB" + (f", GPU {final_gpu_mem:.1f}MB" if final_gpu_mem > 0 else ""))

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_memory(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    rerun_k: int = 0,
    staleness_mode: str = "semantic",
    df_subset_optimization: bool = False,
) -> MemoryResults:
    """
    Run notebook on FlowBook kernel and collect MEMORY metrics using HeapSizer.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        rerun_k: Number of rerun iterations
        staleness_mode: Staleness computation mode ('syntactic' or 'semantic')
        df_subset_optimization: Enable DataFrame subset optimization for checkpoints

    Returns:
        MemoryResults with cell memory data from HeapSizer + checkpoint var costs
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cell_order = [c.get("id", f"cell_{i}") for i, c in enumerate(code_cells)]

    log(f"FlowBook Memory: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = MemoryResults(kernel_name="flowbook_kernel")

    try:
        log("FlowBook Memory: Starting flowbook_kernel...")
        # Enable FLOWBOOK_PROFILE_CHECKPOINT to populate per-variable memory bytes
        kernel_manager, kernel_client = create_flowbook_kernel(
            extra_env={"FLOWBOOK_PROFILE_CHECKPOINT": "1"}
        )
        log("FlowBook Memory: Kernel ready")

        # Enable continue_after_violation
        kernel_client.execute("%continue_after_violation on", silent=True)
        _wait_for_idle(kernel_client)

        # Set staleness computation mode
        kernel_client.execute(f"%staleness_mode {staleness_mode}", silent=True)
        _wait_for_idle(kernel_client)
        log(f"FlowBook Memory: staleness_mode set to {staleness_mode}")

        # Enable DataFrame subset optimization if requested
        if df_subset_optimization:
            kernel_client.execute("%df_subset_checkpoints on", silent=True)
            _wait_for_idle(kernel_client)
            log("FlowBook Memory: df_subset_checkpoints enabled")

        # Run a warm-up cell to trigger lazy initialization (cudf import, tracking patches, etc.)
        log("FlowBook Memory: Running warm-up cell...")
        warmup_result = execute_cell_flowbook(
            kernel_client,
            "_warmup_var_ = 1; del _warmup_var_",  # Trigger tracking pipeline, then clean up
            "_warmup_",
            ["_warmup_"],
            timeout=60.0
        )
        log(f"FlowBook Memory: Warm-up complete (took {warmup_result.get('execute_duration_ms', 0):.0f}ms)")

        # Get initial namespace size
        before_stats = get_namespace_size(kernel_client)
        max_footprint_mb = 0.0

        # Track cumulative checkpoint costs across all cells
        # Track var count for metadata overhead estimate
        cumulative_var_count = 0

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"FlowBook Memory: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            # Get namespace size before cell
            pre_stats = get_namespace_size(kernel_client)

            timing = execute_cell_flowbook(
                kernel_client, source, cell_id, cell_order, cell_timeout
            )

            # Get namespace size after cell
            post_stats = get_namespace_size(kernel_client)

            # Get namespace size (this is what the user's code uses)
            namespace_mb = post_stats.get("total_mb", 0.0)
            max_footprint_mb = max(max_footprint_mb, namespace_mb)

            # Get checkpoint overhead using the new cumulative measurement approach
            # This properly handles CoW sharing - measures checkpoints BEYOND namespace
            overhead = get_checkpoint_overhead(kernel_client, cell_id)

            # Extract this cell's checkpoint delta (pre + post checkpoints)
            pre_name = f"_pre_{cell_id}"
            post_name = f"_post_{cell_id}"
            checkpoint_delta_mb = (
                overhead['by_checkpoint'].get(pre_name, 0.0) +
                overhead['by_checkpoint'].get(post_name, 0.0)
            )
            checkpoint_cumulative_mb = overhead.get('total_mb', 0.0)
            checkpoint_by_var = overhead.get('by_variable') or None

            # Get per-variable checkpoint costs (includes deepcopy_ms for timing plots)
            checkpoint_var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id) or None

            # Get GPU memory
            gpu_mb = get_kernel_gpu_memory_mb(kernel_client)

            # Debug logging
            log(f"  Namespace: {namespace_mb:.1f}MB, Checkpoint delta: {checkpoint_delta_mb:.1f}MB, "
                f"Cumulative: {checkpoint_cumulative_mb:.1f}MB, GPU: {gpu_mb:.1f}MB")
            if checkpoint_by_var:
                top_vars = sorted(checkpoint_by_var.items(), key=lambda x: x[1], reverse=True)[:3]
                log(f"  Top checkpoint vars: {', '.join(f'{k}={v:.2f}MB' for k, v in top_vars)}")

            if timing.get("error") and timing.get("cell_runtime_ms") is None:
                log(f"  Error:\n{timing['error']}")
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    namespace_mb=0.0,
                    checkpoint_delta_mb=checkpoint_delta_mb,
                    checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                    gpu_mb=gpu_mb,
                    checkpoint_by_var=checkpoint_by_var,
                    checkpoint_var_costs=checkpoint_var_costs,
                    status="error",
                    error=timing["error"]
                ))
            else:
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    namespace_mb=namespace_mb,
                    checkpoint_delta_mb=checkpoint_delta_mb,
                    checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                    gpu_mb=gpu_mb,
                    checkpoint_by_var=checkpoint_by_var,
                    checkpoint_var_costs=checkpoint_var_costs,
                    status="ok",
                ))

        # Execute reruns: run all cells k extra times (top-to-bottom)
        if rerun_k > 0:
            total_rerun_cells = rerun_k * len(code_cells)
            log(f"FlowBook Memory: Executing {rerun_k} rerun pass(es) ({total_rerun_cells} cell executions)...")
            rerun_count = 0
            for pass_num in range(rerun_k):
                log(f"FlowBook Memory: Rerun pass {pass_num + 1}/{rerun_k}...")
                for idx, cell in enumerate(code_cells):
                    cell_id = cell.get("id", f"cell_{idx}")
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    if not source.strip():
                        continue

                    rerun_count += 1
                    log(f"FlowBook Memory: Rerun {rerun_count}/{total_rerun_cells} - pass {pass_num+1}, cell {idx+1} ({cell_id})...")

                    # Get namespace size before cell
                    pre_stats = get_namespace_size(kernel_client)

                    timing = execute_cell_flowbook(
                        kernel_client, source, cell_id, cell_order, cell_timeout
                    )

                    # Get namespace size after cell
                    post_stats = get_namespace_size(kernel_client)
                    namespace_mb = post_stats.get("total_mb", 0.0)
                    max_footprint_mb = max(max_footprint_mb, namespace_mb)

                    # Get checkpoint overhead using cumulative measurement
                    overhead = get_checkpoint_overhead(kernel_client, cell_id)
                    pre_name = f"_pre_{cell_id}"
                    post_name = f"_post_{cell_id}"
                    checkpoint_delta_mb = (
                        overhead['by_checkpoint'].get(pre_name, 0.0) +
                        overhead['by_checkpoint'].get(post_name, 0.0)
                    )
                    checkpoint_cumulative_mb = overhead.get('total_mb', 0.0)
                    checkpoint_by_var = overhead.get('by_variable') or None

                    # Get per-variable checkpoint costs (includes deepcopy_ms for timing plots)
                    checkpoint_var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id) or None

                    gpu_mb = get_kernel_gpu_memory_mb(kernel_client)

                    if timing.get("error") and timing.get("cell_runtime_ms") is None:
                        log(f"  Rerun Error:\n{timing['error']}")
                        results.rerun_cells.append(MemoryCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            namespace_mb=0.0,
                            checkpoint_delta_mb=checkpoint_delta_mb,
                            checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                            gpu_mb=gpu_mb,
                            checkpoint_by_var=checkpoint_by_var,
                            checkpoint_var_costs=checkpoint_var_costs,
                            status="error",
                            error=timing["error"],
                            is_rerun=True,
                        ))
                    else:
                        results.rerun_cells.append(MemoryCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            namespace_mb=namespace_mb,
                            checkpoint_delta_mb=checkpoint_delta_mb,
                            checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                            gpu_mb=gpu_mb,
                            checkpoint_by_var=checkpoint_by_var,
                            checkpoint_var_costs=checkpoint_var_costs,
                            status="ok",
                            is_rerun=True,
                        ))
                        log(f"  Rerun Namespace: {namespace_mb:.1f}MB, Delta: {checkpoint_delta_mb:.1f}MB, "
                            f"Cumulative: {checkpoint_cumulative_mb:.1f}MB")

        # Get final stats
        final_stats = get_namespace_size(kernel_client)
        final_gpu_mb = get_kernel_gpu_memory_mb(kernel_client)

        # Get final checkpoint overhead (for last cell)
        if code_cells:
            last_cell_id = code_cells[-1].get("id", f"cell_{len(code_cells)-1}")
            final_overhead = get_checkpoint_overhead(kernel_client, last_cell_id)
            total_checkpoint_mb = final_overhead.get('total_mb', 0.0)
        else:
            total_checkpoint_mb = 0.0

        namespace_mb = final_stats.get("total_mb", 0.0)
        if namespace_mb > 0:
            # Ratio = (namespace + checkpoint_overhead) / namespace
            memory_overhead_ratio = (namespace_mb + total_checkpoint_mb) / namespace_mb
        else:
            memory_overhead_ratio = 1.0

        results.totals = {
            "final_namespace_mb": namespace_mb,
            "max_namespace_mb": max_footprint_mb,
            "total_checkpoint_mb": total_checkpoint_mb,
            "gpu_mb": final_gpu_mb,
            "memory_overhead_ratio": memory_overhead_ratio,
        }

        log(f"FlowBook Memory: Final namespace {namespace_mb:.1f}MB, "
            f"Max {max_footprint_mb:.1f}MB, Checkpoints {total_checkpoint_mb:.1f}MB (ratio: {memory_overhead_ratio:.3f}x)" +
            (f", GPU {final_gpu_mb:.1f}MB" if final_gpu_mb > 0 else ""))

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
            default=None,
            help="Timeout in seconds per cell (default: no timeout).",
        )
        subparser.add_argument(
            "--skip-memory",
            action="store_true",
            help="Skip memory measurement phases (timing only)",
        )
        subparser.add_argument(
            "--run-baseline",
            action="store_true",
            help="Run baseline kernel (skipped by default)",
        )
        subparser.add_argument(
            "--rerun-k",
            type=int,
            default=0,
            help="Number of extra top-to-bottom passes after initial run (default: 0)",
        )
        subparser.add_argument(
            "--trials",
            type=int,
            default=1,
            help="Number of trials to run. Each trial saved to separate file (e.g., notebook-1.json, notebook-2.json)",
        )
        subparser.add_argument(
            "--start",
            type=int,
            default=1,
            help="Starting trial number (default: 1). Use negative numbers for counting down (e.g., --trials 3 --start -4 produces -4, -3, -2)",
        )
        subparser.add_argument(
            "--staleness-mode",
            type=str,
            choices=["syntactic", "semantic"],
            default="semantic",
            help="Staleness computation mode: 'syntactic' (set intersection, lower memory) or 'semantic' (checkpoint diff, precise). Default: semantic",
        )
        subparser.add_argument(
            "--df-subset-optimization",
            action="store_true",
            help="Enable DataFrame subset optimization for checkpoints. Detects row-subsets and stores indices instead of full copies.",
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
        Run notebook using 4-phase execution for timing and memory comparison.

        Phase 1: FlowBook timing (Scalene OFF)
        Phase 2: Baseline timing (Scalene OFF)
        Phase 3: Baseline memory (Scalene ON) - if Scalene available
        Phase 4: FlowBook memory (Scalene ON) - if Scalene available

        Args:
            notebook_content: Notebook JSON
            kernel_client: Not used (we manage our own kernels)
            selected_cell_ids: Not used (we execute all cells)
            config: Optional configuration
            **kwargs: Additional arguments (e.g., timeout, notebook_path)

        Returns:
            ProcessingResult with comparison metadata
        """
        cell_timeout = kwargs.get("timeout", 14400.0)  # 4 hours default
        skip_memory = kwargs.get("skip_memory", False)
        run_baseline = kwargs.get("run_baseline", False)
        rerun_k = kwargs.get("rerun_k", 0)
        num_trials = kwargs.get("trials", 1)
        start_trial = kwargs.get("start", 1)
        staleness_mode = kwargs.get("staleness_mode", "semantic")
        df_subset_optimization = kwargs.get("df_subset_optimization", False)
        notebook_path = kwargs.get("notebook_path", "unknown.ipynb")

        cells = notebook_content.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        # Check if HeapSizer is available for memory phases
        heapsizer_available = _is_heapsizer_available() and not skip_memory

        # Prepare output directory
        from flowbook.util.output import output as global_output
        timings_dir = Path(global_output.timings_file).parent
        notebook_stem = Path(notebook_path).stem

        # Track outputs across all trials
        all_json_paths: List[Path] = []
        last_baseline_total = 0.0
        last_flowbook_execute = 0.0
        last_flowbook_code = 0.0
        last_flowbook_state = 0.0
        last_flowbook_check = 0.0

        # Track staleness data across trials for consistency checking
        first_staleness_data: Optional[Dict[str, Any]] = None
        last_staleness_data: Optional[Dict[str, Any]] = None
        staleness_mismatch_warned = False

        with self.timing_context() as get_elapsed:
            if run_baseline:
                log(f"Starting 4-phase baseline vs FlowBook comparison...")
            else:
                log(f"Starting FlowBook-only comparison (use --run-baseline to include baseline)...")
            log(f"Notebook: {notebook_path}")
            log(f"Cell timeout: {cell_timeout}s" if cell_timeout else "Cell timeout: none")
            log(f"HeapSizer available: {heapsizer_available}")
            log(f"Run baseline: {run_baseline}")
            if rerun_k > 0:
                log(f"Rerun passes: {rerun_k} (will execute all {len(code_cells)} cells {rerun_k} extra time(s))")
            if num_trials > 1:
                log(f"Trials: {num_trials} (starting at {start_trial})")
            log("")

            # Generate trial numbers: start_trial, start_trial+1, ..., start_trial+num_trials-1
            trial_numbers = list(range(start_trial, start_trial + num_trials))

            for trial_num in trial_numbers:
                if num_trials > 1:
                    log("=" * 60)
                    log(f"TRIAL {trial_num} ({trial_numbers.index(trial_num) + 1}/{num_trials})")
                    log("=" * 60)
                    log("")

                # ============================================================
                # PHASE 1: FlowBook Timing (Scalene OFF)
                # ============================================================
                log("=" * 60)
                log("PHASE 1: FLOWBOOK TIMING (Scalene OFF)")
                log("=" * 60)
                flowbook_timing = run_flowbook_timing(notebook_content, cell_timeout, rerun_k, staleness_mode, df_subset_optimization)
                log("")

                # ============================================================
                # PHASE 2: Baseline Timing (Scalene OFF) - if run_baseline
                # ============================================================
                baseline_timing = None
                if run_baseline:
                    log("=" * 60)
                    log("PHASE 2: BASELINE TIMING (Scalene OFF)")
                    log("=" * 60)
                    baseline_timing = run_baseline_timing(notebook_content, cell_timeout, rerun_k)
                    log("")
                else:
                    log("=" * 60)
                    log("PHASE 2: BASELINE TIMING - SKIPPED (use --run-baseline to enable)")
                    log("=" * 60)
                    log("")

                # ============================================================
                # PHASE 3: Baseline Memory (HeapSizer) - if available and run_baseline
                # ============================================================
                baseline_memory = None
                if heapsizer_available and run_baseline:
                    log("=" * 60)
                    log("PHASE 3: BASELINE MEMORY (HeapSizer)")
                    log("=" * 60)
                    baseline_memory = run_baseline_memory(notebook_content, cell_timeout, rerun_k)
                    log("")
                elif heapsizer_available:
                    log("=" * 60)
                    log("PHASE 3: BASELINE MEMORY - SKIPPED (use --run-baseline to enable)")
                    log("=" * 60)
                    log("")

                # ============================================================
                # PHASE 4: FlowBook Memory (HeapSizer) - if available
                # ============================================================
                flowbook_memory = None
                if heapsizer_available:
                    log("=" * 60)
                    log("PHASE 4: FLOWBOOK MEMORY (HeapSizer)")
                    log("=" * 60)
                    flowbook_memory = run_flowbook_memory(notebook_content, cell_timeout, rerun_k, staleness_mode, df_subset_optimization)
                    log("")

                # Build comparison result with new structure
                metadata_dict: Dict[str, Any] = {
                    "num_cells": len(code_cells),
                    "timeout_seconds": cell_timeout,
                    "staleness_mode": staleness_mode,
                    "df_subset_optimization": df_subset_optimization,
                }
                if rerun_k > 0:
                    metadata_dict["rerun_k"] = rerun_k
                if num_trials > 1:
                    metadata_dict["trial"] = trial_num
                    metadata_dict["num_trials"] = num_trials

                comparison = ComparisonResult(
                    version="2.0",
                    notebook_path=str(notebook_path),
                    timestamp=datetime.now().isoformat(),
                    kernels={
                        "baseline": KernelResults(
                            kernel_name="baseline_kernel",
                            timing=baseline_timing,
                            memory=baseline_memory,
                        ),
                        "flowbook": KernelResults(
                            kernel_name="flowbook_kernel",
                            timing=flowbook_timing,
                            memory=flowbook_memory,
                        ),
                    },
                    scalene_available=heapsizer_available,  # Keep field name for compatibility
                    metadata=metadata_dict,
                )

                # Convert to dict for JSON serialization
                def to_dict(obj):
                    if obj is None:
                        return None
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

                # Save JSON output - numbered for multi-trial
                if num_trials > 1:
                    json_output_path = timings_dir / f"{notebook_stem}_comparison-{trial_num}.json"
                else:
                    json_output_path = timings_dir / f"{notebook_stem}_comparison.json"

                with open(json_output_path, "w") as f:
                    json.dump(comparison_dict, f, indent=2)

                all_json_paths.append(json_output_path)

                # Print summary for this trial
                log("=" * 60)
                if num_trials > 1:
                    log(f"TRIAL {trial_num} SUMMARY")
                else:
                    log("SUMMARY")
                log("=" * 60)

                baseline_total = baseline_timing.totals.get("execute_duration_ms", 0) if baseline_timing else 0
                flowbook_execute = flowbook_timing.totals.get("execute_duration_ms", 0)
                flowbook_code = flowbook_timing.totals.get("code_duration_ms", 0)
                flowbook_state = flowbook_timing.totals.get("state_duration_ms", 0)
                flowbook_check = flowbook_timing.totals.get("check_duration_ms", 0)

                # Track last trial for return value
                last_baseline_total = baseline_total
                last_flowbook_execute = flowbook_execute
                last_flowbook_code = flowbook_code
                last_flowbook_state = flowbook_state
                last_flowbook_check = flowbook_check

                log("TIMING:")
                if baseline_timing:
                    if baseline_total > 0:
                        state_overhead_pct = (flowbook_state / baseline_total) * 100
                        check_overhead_pct = (flowbook_check / baseline_total) * 100
                    else:
                        state_overhead_pct = 0.0
                        check_overhead_pct = 0.0
                    log(f"  Baseline code time:   {baseline_total:,.1f}ms")
                    log(f"  FlowBook execute:     {flowbook_execute:,.1f}ms")
                    log(f"  FlowBook code time:   {flowbook_code:,.1f}ms")
                    log(f"  FlowBook state time:  {flowbook_state:,.1f}ms ({state_overhead_pct:.1f}%)")
                    log(f"  FlowBook check time:  {flowbook_check:,.1f}ms ({check_overhead_pct:.1f}%)")
                else:
                    log(f"  Baseline:             SKIPPED")
                    log(f"  FlowBook execute:     {flowbook_execute:,.1f}ms")
                    log(f"  FlowBook code time:   {flowbook_code:,.1f}ms")
                    log(f"  FlowBook state time:  {flowbook_state:,.1f}ms")
                    log(f"  FlowBook check time:  {flowbook_check:,.1f}ms")
                log("")

                if heapsizer_available and flowbook_memory:
                    flowbook_mem = flowbook_memory.totals.get("final_footprint_mb", 0)
                    log("MEMORY (from HeapSizer):")
                    if baseline_memory:
                        baseline_mem = baseline_memory.totals.get("final_footprint_mb", 0)
                        mem_overhead = flowbook_mem - baseline_mem
                        log(f"  Baseline namespace:   {baseline_mem:,.1f}MB")
                        log(f"  FlowBook namespace:   {flowbook_mem:,.1f}MB")
                        log(f"  Memory overhead:      {mem_overhead:+,.1f}MB")
                    else:
                        log(f"  Baseline:             SKIPPED")
                        log(f"  FlowBook namespace:   {flowbook_mem:,.1f}MB")
                    log("")

                if rerun_k > 0:
                    rerun_baseline = baseline_timing.totals.get("rerun_runtime_ms", 0) if baseline_timing else 0
                    rerun_flowbook_execute = flowbook_timing.totals.get("rerun_execute_ms", 0)
                    rerun_flowbook_code = flowbook_timing.totals.get("rerun_code_ms", 0)
                    rerun_state = flowbook_timing.totals.get("rerun_state_ms", 0)
                    rerun_check = flowbook_timing.totals.get("rerun_check_ms", 0)
                    total_rerun_cells = rerun_k * len(code_cells)

                    log(f"RERUN TIMING ({rerun_k} pass(es) x {len(code_cells)} cells = {total_rerun_cells} executions):")
                    if baseline_timing:
                        log(f"  Baseline rerun:       {rerun_baseline:,.1f}ms")
                    else:
                        log(f"  Baseline rerun:       SKIPPED")
                    log(f"  FlowBook execute:     {rerun_flowbook_execute:,.1f}ms")
                    log(f"  FlowBook code:        {rerun_flowbook_code:,.1f}ms")
                    log(f"  FlowBook state time:  {rerun_state:,.1f}ms")
                    log(f"  FlowBook check time:  {rerun_check:,.1f}ms")
                    log("")

                # Checking results summary (exclude never_executed - empty code cells)
                if flowbook_timing and flowbook_timing.cells:
                    clean_count = 0
                    stale_count = 0
                    error_count = 0
                    reason_counts: Dict[str, int] = {}
                    error_counts: Dict[str, int] = {}

                    for cell in flowbook_timing.cells:
                        if cell.checking_result:
                            # Skip cells that only have never_executed reason (empty code cells)
                            reasons = cell.checking_result.reasons
                            errors = cell.checking_result.errors
                            if reasons and all(r.get("type") == "never_executed" for r in reasons) and not errors:
                                continue
                            if cell.checking_result.cell_status == "clean":
                                clean_count += 1
                            elif cell.checking_result.cell_status == "error":
                                error_count += 1
                                for error in errors:
                                    etype = error.get("error_type", "unknown")
                                    error_counts[etype] = error_counts.get(etype, 0) + 1
                            else:
                                stale_count += 1
                                for reason in reasons:
                                    rtype = reason.get("type", "unknown")
                                    if rtype != "never_executed":  # Don't count never_executed
                                        reason_counts[rtype] = reason_counts.get(rtype, 0) + 1

                    # Store staleness data for cross-trial comparison
                    current_staleness = {
                        "clean_count": clean_count,
                        "stale_count": stale_count,
                        "error_count": error_count,
                        "reason_counts": dict(reason_counts),
                        "error_counts": dict(error_counts),
                    }

                    # Check consistency across trials
                    if first_staleness_data is None:
                        first_staleness_data = current_staleness
                    elif not staleness_mismatch_warned and current_staleness != first_staleness_data:
                        staleness_mismatch_warned = True
                        log("WARNING: Staleness results differ from first trial!")
                        log(f"  First trial: clean={first_staleness_data['clean_count']}, stale={first_staleness_data['stale_count']}, errors={first_staleness_data.get('error_count', 0)}, reasons={first_staleness_data['reason_counts']}")
                        log(f"  This trial:  clean={clean_count}, stale={stale_count}, errors={error_count}, reasons={dict(reason_counts)}")

                    last_staleness_data = current_staleness

                    log("CHECKING RESULTS:")
                    log(f"  Clean cells:          {clean_count}")
                    log(f"  Stale cells:          {stale_count}")
                    log(f"  Error cells:          {error_count}")
                    if reason_counts:
                        log("  Staleness reasons:")
                        for rtype, count in sorted(reason_counts.items()):
                            log(f"    {rtype}: {count}")
                    if error_counts:
                        log("  Error types:")
                        for etype, count in sorted(error_counts.items()):
                            log(f"    {etype}: {count}")
                    log("")

                log(f"Results saved to: {json_output_path}")
                log("")

            # Final summary for multi-trial runs
            if num_trials > 1:
                log("=" * 60)
                log(f"COMPLETED {num_trials} TRIALS")
                log("=" * 60)
                log("")
                log("Output files:")
                for path in all_json_paths:
                    log(f"  {path}")
                log("")

                # Show staleness summary in multi-trial output
                if last_staleness_data:
                    log("CHECKING RESULTS (consistent across trials):" if not staleness_mismatch_warned else "CHECKING RESULTS (WARNING: varied across trials):")
                    log(f"  Clean cells:          {last_staleness_data['clean_count']}")
                    log(f"  Stale cells:          {last_staleness_data['stale_count']}")
                    log(f"  Error cells:          {last_staleness_data.get('error_count', 0)}")
                    if last_staleness_data['reason_counts']:
                        log("  Staleness reasons:")
                        for rtype, count in sorted(last_staleness_data['reason_counts'].items()):
                            log(f"    {rtype}: {count}")
                    if last_staleness_data.get('error_counts'):
                        log("  Error types:")
                        for etype, count in sorted(last_staleness_data['error_counts'].items()):
                            log(f"    {etype}: {count}")
                    log("")

            total_time = get_elapsed()

        # Return the last trial's results (or single trial results)
        return ProcessingResult(
            notebook=notebook_content,
            metadata={
                "status": "success",
                "command": self.command_name,
                "comparison_file": str(all_json_paths[-1]) if all_json_paths else "",
                "comparison_files": [str(p) for p in all_json_paths] if num_trials > 1 else None,
                "num_trials": num_trials if num_trials > 1 else None,
                "baseline_code_ms": last_baseline_total,
                "flowbook_execute_ms": last_flowbook_execute,
                "flowbook_code_ms": last_flowbook_code,
                "flowbook_state_ms": last_flowbook_state,
                "flowbook_check_ms": last_flowbook_check,
                "scalene_available": heapsizer_available,  # Keep field name for compatibility
            },
            total_cost=0.0,
            total_time=total_time,
        )
