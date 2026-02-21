"""
Compare baseline (python3) vs FlowBook kernel execution with timing and memory metrics.

4-Phase Execution:
  Phase 1: FlowBook timing - collect cell_runtime_ms, state_duration_ms, check_duration_ms
  Phase 2: Baseline timing - collect cell_runtime_ms
  Phase 3: Baseline memory (HeapSizer) - collect namespace_size_mb
  Phase 4: FlowBook memory (HeapSizer) - collect namespace_size_mb, checkpoint overhead

Memory measurement uses HeapSizer for accurate heap traversal with proper handling of:
- NumPy array views and shared data buffers
- Pandas DataFrame/Series Copy-on-Write sharing
- Object deduplication across shared references

Usage via CLI:
    flowbook compare-baseline notebook.ipynb
    flowbook compare-baseline notebook.ipynb --timeout 3600
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
class TimingCellMetrics:
    """Timing metrics for a single cell execution (Scalene OFF)."""
    cell_id: str
    cell_index: int
    cell_runtime_ms: float
    state_duration_ms: float  # FlowBook only
    check_duration_ms: float  # FlowBook only
    status: str
    error: Optional[str] = None


@dataclass
class MemoryCellMetrics:
    """Memory metrics for a single cell execution (Scalene ON)."""
    cell_id: str
    cell_index: int
    current_footprint_mb: float
    max_footprint_mb: float
    allocation_delta_mb: float
    gpu_mem_samples: float
    checkpoint_var_costs: Optional[Dict[str, Any]] = None  # FlowBook only: per-cell var costs (includes deepcopy_ms)
    overhead_breakdown: Optional[Dict[str, float]] = None  # FlowBook only: {category: MB}
    cumulative_by_type: Optional[Dict[str, int]] = None  # FlowBook only: cumulative bytes by type
    cumulative_by_var: Optional[Dict[str, int]] = None  # FlowBook only: cumulative bytes by variable
    # Pre/post checkpoint breakdown for NO_POST analysis (accounts for sharing)
    pre_only_bytes: int = 0  # Bytes if only pre checkpoints existed (with sharing)
    post_savings_bytes: int = 0  # Memory saved by removing post checkpoints
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class TimingResults:
    """Timing results from a notebook execution (Scalene OFF)."""
    kernel_name: str
    cells: List[TimingCellMetrics] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryResults:
    """Memory results from a notebook execution (Scalene ON)."""
    kernel_name: str
    cells: List[MemoryCellMetrics] = field(default_factory=list)
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


def create_flowbook_kernel() -> Tuple[KernelManager, FlowbookKernelClient]:
    """
    Start a flowbook_kernel.

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
    Execute a cell on the flowbook_kernel and measure execution time.

    Returns:
        Dict with cell_runtime_ms (client-side), state_duration_ms, check_duration_ms, and optional error
    """
    # Set cell order for reproducibility tracking
    kernel_client.set_cell_order(cell_order)

    # Measure wall-clock time from client side (same as baseline)
    start = time.perf_counter()

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

    # Calculate client-side elapsed time (same methodology as baseline)
    elapsed = time.perf_counter() - start

    if flowbook_metadata:
        # Check for violations in metadata
        violation = flowbook_metadata.get("violation")
        violation_msg = None
        if violation:
            violation_msg = violation.get("message", "Reproducibility violation")

        return {
            # Use client-side timing for fair comparison with baseline
            "cell_runtime_ms": elapsed * 1000,
            "state_duration_ms": flowbook_metadata.get("state_duration_ms", 0.0),
            "check_duration_ms": flowbook_metadata.get("check_duration_ms", 0.0),
            "error": error_msg,
            "violation": violation_msg,
            "stale_cells": flowbook_metadata.get("stale_cells", []),
        }
    else:
        return {
            "cell_runtime_ms": elapsed * 1000,
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
    breakdown = mc.get_overhead_breakdown()
    _overhead_breakdown['checkpoints_mb'] = breakdown.get('checkpoints_bytes', 0) / (1024 * 1024)
    _overhead_breakdown['tracking_metadata_mb'] = breakdown.get('tracking_metadata_bytes', 0) / (1024 * 1024)
    _overhead_breakdown['other_mb'] = breakdown.get('other_bytes', 0) / (1024 * 1024)

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
                    cell_runtime_ms=0.0,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    status="error",
                    error=timing["error"]
                ))
            else:
                runtime_ms = timing["cell_runtime_ms"]
                total_runtime_ms += runtime_ms

                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=runtime_ms,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    status="ok",
                ))
                log(f"  Runtime: {runtime_ms:.1f}ms")

        results.totals = {
            "cell_runtime_ms": total_runtime_ms,
        }

        log(f"Baseline Timing: Total runtime {total_runtime_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_timing(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
) -> TimingResults:
    """
    Run notebook on FlowBook kernel and collect TIMING metrics only (Scalene OFF).

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds

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

        total_runtime_ms = 0.0
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

            if timing.get("error") and timing.get("cell_runtime_ms") is None:
                log(f"  Error:\n{timing['error']}")
                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=0.0,
                    state_duration_ms=0.0,
                    check_duration_ms=0.0,
                    status="error",
                    error=timing["error"]
                ))
            else:
                if timing.get("violation"):
                    log(f"  Violation: {timing['violation']}")

                runtime_ms = timing["cell_runtime_ms"] or 0.0
                state_ms = timing["state_duration_ms"] or 0.0
                check_ms = timing["check_duration_ms"] or 0.0

                total_runtime_ms += runtime_ms
                total_state_ms += state_ms
                total_check_ms += check_ms

                status = "ok"
                if timing.get("error"):
                    status = "error"
                elif timing.get("violation"):
                    status = "violation"

                results.cells.append(TimingCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    cell_runtime_ms=runtime_ms,
                    state_duration_ms=state_ms,
                    check_duration_ms=check_ms,
                    status=status,
                    error=timing.get("violation") or timing.get("error"),
                ))
                log(f"  Runtime: {runtime_ms:.1f}ms, State: {state_ms:.1f}ms, Check: {check_ms:.1f}ms")

        results.totals = {
            "cell_runtime_ms": total_runtime_ms,
            "state_duration_ms": total_state_ms,
            "check_duration_ms": total_check_ms,
        }

        log(f"FlowBook Timing: Total runtime {total_runtime_ms:.1f}ms, state {total_state_ms:.1f}ms, check {total_check_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_baseline_memory(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
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
                    current_footprint_mb=0.0,
                    max_footprint_mb=0.0,
                    allocation_delta_mb=0.0,
                    gpu_mem_samples=0.0,
                    status="error",
                    error=timing["error"]
                ))
            else:
                allocation_delta = current_mb - pre_stats.get("total_mb", 0.0)
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    current_footprint_mb=current_mb,
                    max_footprint_mb=max_footprint_mb,
                    allocation_delta_mb=allocation_delta,
                    gpu_mem_samples=0.0,  # GPU tracking not supported without Scalene
                    status="ok",
                ))
                log(f"  Namespace: {current_mb:.1f}MB, Delta: {allocation_delta:.1f}MB")

        # Get final stats
        final_stats = get_namespace_size(kernel_client)
        results.totals = {
            "final_footprint_mb": final_stats.get("total_mb", 0.0),
            "max_footprint_mb": max_footprint_mb,
            "total_allocation_mb": final_stats.get("total_mb", 0.0) - before_stats.get("total_mb", 0.0),
            "gpu_mem_samples": 0.0,
        }

        log(f"Baseline Memory: Final namespace {final_stats.get('total_mb', 0):.1f}MB, "
            f"Max {max_footprint_mb:.1f}MB")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_memory(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
) -> MemoryResults:
    """
    Run notebook on FlowBook kernel and collect MEMORY metrics using HeapSizer.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds

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
        kernel_manager, kernel_client = create_flowbook_kernel()
        log("FlowBook Memory: Kernel ready")

        # Enable continue_after_violation
        kernel_client.execute("%continue_after_violation on", silent=True)
        _wait_for_idle(kernel_client)

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

            # Get checkpoint variable costs (combined pre + post) for per-variable breakdown
            var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id)

            # Get TRUE cumulative checkpoint size - measures ALL checkpoints together
            # This accounts for memory sharing between checkpoints (via memo dict)
            cumulative_size = get_flowbook_cumulative_checkpoint_size(kernel_client, cell_id)
            cumulative_checkpoint_bytes = cumulative_size.get('total_bytes', 0)

            # Get pre/post checkpoint breakdown for NO_POST analysis
            pre_post_sizes = get_flowbook_pre_post_checkpoint_sizes(kernel_client, cell_id)
            pre_only_bytes = pre_post_sizes.get('pre_only_bytes', 0)
            post_savings_bytes = pre_post_sizes.get('post_savings_bytes', 0)

            current_mb = post_stats.get("total_mb", 0.0)
            max_footprint_mb = max(max_footprint_mb, current_mb)

            # Track variable count for metadata overhead estimate
            if var_costs:
                cumulative_var_count += len(var_costs)

            # Build CUMULATIVE overhead breakdown
            # checkpoints_mb is the TRUE cumulative size accounting for sharing
            overhead_breakdown = {
                'checkpoints_mb': cumulative_checkpoint_bytes / (1024 * 1024),
                'execution_records_mb': 0.0,
                'tracking_metadata_mb': cumulative_var_count * 200 / (1024 * 1024),
                'other_mb': (idx + 1) * 0.001,  # ~1KB per checkpoint
            }

            # Debug logging for checkpoint costs
            cumulative_ckpt_mb = overhead_breakdown['checkpoints_mb']
            if var_costs:
                # Per-cell costs (for reference - may overcount due to sharing)
                cell_checkpoint_bytes = sum(v.get('bytes', 0) for v in var_costs.values())
                cell_ckpt_mb = cell_checkpoint_bytes / (1024 * 1024)
                log(f"  This cell's vars: {cell_ckpt_mb:.1f}MB, TRUE Cumulative: {cumulative_ckpt_mb:.1f}MB")
                log(f"  Variables in this checkpoint ({len(var_costs)}):")
                for var_name, info in list(var_costs.items())[:3]:
                    log(f"    {var_name}: {info.get('bytes', 0)/1024/1024:.2f}MB ({info.get('type')})")
                if len(var_costs) > 3:
                    log(f"    ... and {len(var_costs) - 3} more")
            else:
                log(f"  No checkpoint costs retrieved for cell {cell_id}, Cumulative: {cumulative_ckpt_mb:.1f}MB")

            # Extract cumulative breakdown by type and variable (for Plots 3/4)
            cumulative_by_type = cumulative_size.get('by_type', {})
            cumulative_by_var = cumulative_size.get('by_variable', {})

            if timing.get("error") and timing.get("cell_runtime_ms") is None:
                log(f"  Error:\n{timing['error']}")
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    current_footprint_mb=0.0,
                    max_footprint_mb=0.0,
                    allocation_delta_mb=0.0,
                    gpu_mem_samples=0.0,
                    checkpoint_var_costs=None,
                    overhead_breakdown=overhead_breakdown if overhead_breakdown else None,
                    cumulative_by_type=cumulative_by_type if cumulative_by_type else None,
                    cumulative_by_var=cumulative_by_var if cumulative_by_var else None,
                    pre_only_bytes=pre_only_bytes,
                    post_savings_bytes=post_savings_bytes,
                    status="error",
                    error=timing["error"]
                ))
            else:
                allocation_delta = current_mb - pre_stats.get("total_mb", 0.0)
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    current_footprint_mb=current_mb,
                    max_footprint_mb=max_footprint_mb,
                    allocation_delta_mb=allocation_delta,
                    gpu_mem_samples=0.0,  # GPU tracking not supported without Scalene
                    checkpoint_var_costs=var_costs if var_costs else None,
                    overhead_breakdown=overhead_breakdown if overhead_breakdown else None,
                    cumulative_by_type=cumulative_by_type if cumulative_by_type else None,
                    cumulative_by_var=cumulative_by_var if cumulative_by_var else None,
                    pre_only_bytes=pre_only_bytes,
                    post_savings_bytes=post_savings_bytes,
                    status="ok",
                ))
                cumulative_ckpt = overhead_breakdown.get('checkpoints_mb', 0) if overhead_breakdown else 0
                pre_only_mb = pre_only_bytes / (1024 * 1024)
                savings_mb = post_savings_bytes / (1024 * 1024)
                log(f"  Namespace: {current_mb:.1f}MB, Delta: {allocation_delta:.1f}MB, Checkpoints: {cumulative_ckpt:.1f}MB (pre-only: {pre_only_mb:.1f}MB, post-savings: {savings_mb:.1f}MB)")

        # Get final stats
        final_stats = get_namespace_size(kernel_client)
        results.totals = {
            "final_footprint_mb": final_stats.get("total_mb", 0.0),
            "max_footprint_mb": max_footprint_mb,
            "total_allocation_mb": final_stats.get("total_mb", 0.0) - before_stats.get("total_mb", 0.0),
            "gpu_mem_samples": 0.0,
        }

        log(f"FlowBook Memory: Final namespace {final_stats.get('total_mb', 0):.1f}MB, "
            f"Max {max_footprint_mb:.1f}MB")

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
            default=3600.0,
            help="Timeout in seconds per cell (default: 3600). Should be generous as FlowBook adds overhead.",
        )
        subparser.add_argument(
            "--skip-memory",
            action="store_true",
            help="Skip memory measurement phases (timing only)",
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
        cell_timeout = kwargs.get("timeout", 3600.0)
        skip_memory = kwargs.get("skip_memory", False)
        notebook_path = kwargs.get("notebook_path", "unknown.ipynb")

        cells = notebook_content.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        # Check if HeapSizer is available for memory phases
        heapsizer_available = _is_heapsizer_available() and not skip_memory

        with self.timing_context() as get_elapsed:
            log(f"Starting 4-phase baseline vs FlowBook comparison...")
            log(f"Notebook: {notebook_path}")
            log(f"Cell timeout: {cell_timeout}s")
            log(f"HeapSizer available: {heapsizer_available}")
            log("")

            # ============================================================
            # PHASE 1: FlowBook Timing (Scalene OFF)
            # ============================================================
            log("=" * 60)
            log("PHASE 1: FLOWBOOK TIMING (Scalene OFF)")
            log("=" * 60)
            flowbook_timing = run_flowbook_timing(notebook_content, cell_timeout)
            log("")

            # ============================================================
            # PHASE 2: Baseline Timing (Scalene OFF)
            # ============================================================
            log("=" * 60)
            log("PHASE 2: BASELINE TIMING (Scalene OFF)")
            log("=" * 60)
            baseline_timing = run_baseline_timing(notebook_content, cell_timeout)
            log("")

            # ============================================================
            # PHASE 3: Baseline Memory (HeapSizer) - if available
            # ============================================================
            baseline_memory = None
            if heapsizer_available:
                log("=" * 60)
                log("PHASE 3: BASELINE MEMORY (HeapSizer)")
                log("=" * 60)
                baseline_memory = run_baseline_memory(notebook_content, cell_timeout)
                log("")

            # ============================================================
            # PHASE 4: FlowBook Memory (HeapSizer) - if available
            # ============================================================
            flowbook_memory = None
            if heapsizer_available:
                log("=" * 60)
                log("PHASE 4: FLOWBOOK MEMORY (HeapSizer)")
                log("=" * 60)
                flowbook_memory = run_flowbook_memory(notebook_content, cell_timeout)
                log("")

            # Build comparison result with new structure
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
                metadata={
                    "num_cells": len(code_cells),
                    "timeout_seconds": cell_timeout,
                }
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

            # Save JSON output
            from flowbook.util.output import output as global_output
            timings_dir = Path(global_output.timings_file).parent
            notebook_stem = Path(notebook_path).stem
            json_output_path = timings_dir / f"{notebook_stem}_comparison.json"

            with open(json_output_path, "w") as f:
                json.dump(comparison_dict, f, indent=2)

            # Print summary
            log("=" * 60)
            log("SUMMARY")
            log("=" * 60)

            baseline_total = baseline_timing.totals.get("cell_runtime_ms", 0)
            flowbook_runtime = flowbook_timing.totals.get("cell_runtime_ms", 0)
            flowbook_state = flowbook_timing.totals.get("state_duration_ms", 0)
            flowbook_check = flowbook_timing.totals.get("check_duration_ms", 0)

            if baseline_total > 0:
                state_overhead_pct = (flowbook_state / baseline_total) * 100
                check_overhead_pct = (flowbook_check / baseline_total) * 100
            else:
                state_overhead_pct = 0.0
                check_overhead_pct = 0.0

            log("TIMING:")
            log(f"  Baseline runtime:     {baseline_total:,.1f}ms")
            log(f"  FlowBook runtime:     {flowbook_runtime:,.1f}ms")
            log(f"  FlowBook state time:  {flowbook_state:,.1f}ms ({state_overhead_pct:.1f}%)")
            log(f"  FlowBook check time:  {flowbook_check:,.1f}ms ({check_overhead_pct:.1f}%)")
            log("")

            if heapsizer_available and baseline_memory and flowbook_memory:
                baseline_mem = baseline_memory.totals.get("final_footprint_mb", 0)
                flowbook_mem = flowbook_memory.totals.get("final_footprint_mb", 0)
                mem_overhead = flowbook_mem - baseline_mem

                log("MEMORY (from HeapSizer):")
                log(f"  Baseline namespace:   {baseline_mem:,.1f}MB")
                log(f"  FlowBook namespace:   {flowbook_mem:,.1f}MB")
                log(f"  Memory overhead:      {mem_overhead:+,.1f}MB")
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
                "scalene_available": heapsizer_available,  # Keep field name for compatibility
            },
            total_cost=0.0,
            total_time=total_time,
        )
