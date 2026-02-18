"""
Compare baseline (python3) vs FlowBook kernel execution with timing and memory metrics.

4-Phase Execution:
  Phase 1: FlowBook timing (Scalene OFF) - collect cell_runtime_ms, state_duration_ms, check_duration_ms
  Phase 2: Baseline timing (Scalene OFF) - collect cell_runtime_ms
  Phase 3: Baseline memory (Scalene ON) - collect current_footprint_mb, gpu_mem
  Phase 4: FlowBook memory (Scalene ON) - collect current_footprint_mb, gpu_mem, checkpoint overhead

Usage via CLI:
    flowbook compare-baseline notebook.ipynb
    flowbook compare-baseline notebook.ipynb --timeout 300
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
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
    checkpoint_var_costs: Optional[Dict[str, Any]] = None  # FlowBook only
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


def _get_scalene_preload_env() -> Dict[str, str]:
    """Get environment variables needed for Scalene memory tracking."""
    try:
        import scalene
        from scalene.scalene_preload import ScalenePreload
    except ImportError:
        return {}

    args = argparse.Namespace(
        memory=True,
        allocation_sampling_window=10485767,
    )

    return ScalenePreload.get_preload_environ(args)


def _is_scalene_available() -> bool:
    """Check if Scalene is installed and can be used."""
    try:
        import scalene
        from scalene.scalene_preload import ScalenePreload
        return True
    except ImportError:
        return False


def create_baseline_kernel(
    with_scalene_preload: bool = False
) -> Tuple[KernelManager, BlockingKernelClient]:
    """
    Start the baseline scalene kernel.

    Args:
        with_scalene_preload: If True, use kernel spec with Scalene library preloaded

    Returns:
        Tuple of (KernelManager, BlockingKernelClient)
    """
    # Import to ensure kernels are registered
    from flowbook.baseline_scalene_kernel import (
        install_baseline_scalene_kernel,
        install_baseline_scalene_preload_kernel,
    )
    try:
        install_baseline_scalene_kernel()
        install_baseline_scalene_preload_kernel()
    except Exception:
        pass

    # Choose kernel name based on whether we want Scalene preloaded
    if with_scalene_preload:
        kernel_name = "baseline_scalene_preload_kernel"
    else:
        kernel_name = "baseline_scalene_kernel"

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


def create_flowbook_kernel(
    with_scalene_preload: bool = False
) -> Tuple[KernelManager, FlowbookKernelClient]:
    """
    Start a flowbook_kernel.

    Args:
        with_scalene_preload: If True, use kernel spec with Scalene library preloaded

    Returns:
        Tuple of (KernelManager, FlowbookKernelClient)
    """
    make_kernels()

    # Also install the scalene preload kernel if needed
    if with_scalene_preload:
        from flowbook.kernel import install_flowbook_scalene_preload_kernel
        try:
            install_flowbook_scalene_preload_kernel()
        except Exception:
            pass

    # Choose kernel name based on whether we want Scalene preloaded
    if with_scalene_preload:
        kernel_name = "flowbook_scalene_preload_kernel"
    else:
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


def enable_scalene_tracking(kernel_client, timeout: float = 30.0) -> bool:
    """Enable Scalene memory tracking via magic command."""
    # First check if Scalene is available in the kernel
    debug_code = """
import os
import platform
import sys
_debug_info = {
    'platform': platform.system(),
    'dyld': os.environ.get('DYLD_INSERT_LIBRARIES', 'NOT SET'),
    'ld_preload': os.environ.get('LD_PRELOAD', 'NOT SET'),
    'python': sys.executable,
}
try:
    from scalene.scalene_profiler import Scalene
    _debug_info['scalene_import'] = 'OK'
    _debug_info['scalene_initialized'] = Scalene._Scalene__initialized
except ImportError as e:
    _debug_info['scalene_import'] = f'FAILED: {e}'
except Exception as e:
    _debug_info['scalene_import'] = f'ERROR: {e}'
try:
    from flowbook.kernel_support.scalene_memory import ScaleneMemoryTracker
    ScaleneMemoryTracker._scalene_available = None  # Reset cache
    _debug_info['tracker_available'] = ScaleneMemoryTracker.is_available()
except Exception as e:
    _debug_info['tracker_available'] = f'ERROR: {e}'
print(_debug_info)
"""
    msg_id = kernel_client.execute(debug_code, silent=False)

    # Collect iopub messages to get print output
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            msg_type = msg['header']['msg_type']
            if msg_type == 'stream':
                text = msg['content'].get('text', '')
                if text:
                    log(f"Scalene debug: {text.strip()}")
            elif msg_type == 'status':
                if msg['content']['execution_state'] == 'idle':
                    break
        except Exception:
            continue

    msg_id = kernel_client.execute("%scalene_memory on", silent=False)
    start_time = time.time()
    success = False
    while True:
        if time.time() - start_time > timeout:
            log("Scalene enable: timeout waiting for response")
            return False
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        msg_type = msg['header']['msg_type']
        if msg_type == 'display_data':
            # Check for success/failure icons in display_data (FlowBook kernel)
            text = msg['content'].get('data', {}).get('text/plain', '')
            if '✅' in text or 'enabled' in text.lower():
                success = True
            elif '❌' in text or 'failed' in text.lower() or 'not available' in text.lower():
                log(f"Scalene enable failed: {text}")
                return False
        elif msg_type == 'stream':
            # Check for success/failure text in stream output (baseline kernel)
            text = msg['content'].get('text', '')
            if 'enabled' in text.lower():
                success = True
            elif 'failed' in text.lower() or 'not available' in text.lower():
                log(f"Scalene enable failed: {text}")
                return False
        elif msg_type == 'error':
            log(f"Scalene enable error: {msg['content']}")
            return False
        elif msg_type == 'status':
            if msg['content']['execution_state'] == 'idle':
                break
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass

    if success:
        return True
    else:
        log("Scalene enable: no success indicator found")
        return False


def disable_scalene_tracking(kernel_client, timeout: float = 30.0) -> bool:
    """Disable Scalene memory tracking via magic command."""
    msg_id = kernel_client.execute("%scalene_memory off", silent=True)
    _wait_for_idle(kernel_client, timeout)
    return True


def get_scalene_memory_stats(kernel_client, timeout: float = 30.0) -> Dict[str, Any]:
    """Get current Scalene memory statistics.

    Returns dict with current_footprint_mb, max_footprint_mb, gpu_mem_samples, etc.
    """
    # Use a single execute with user_expressions to get stats
    # The expression creates the dict inline to avoid variable persistence issues
    expr_code = """(lambda: (
        __import__('flowbook.kernel_support.scalene_memory', fromlist=['ScaleneMemoryTracker']).ScaleneMemoryTracker.get_memory()
    ))()"""

    msg_id = kernel_client.execute(
        '',
        user_expressions={'_stats': expr_code},
        silent=True,
    )

    # Wait for idle on iopub channel
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {"current_footprint_mb": 0.0, "max_footprint_mb": 0.0, "gpu_mem_samples": 0.0}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the execute_reply, matching by msg_id to ensure we get the right one
    try:
        import ast
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reply = kernel_client.get_shell_msg(timeout=1.0)
            except Exception:
                continue
            # Check if this is the reply to our specific request
            if reply['parent_header'].get('msg_id') != msg_id:
                # Not our reply, keep looking
                continue
            if reply['header']['msg_type'] != 'execute_reply':
                # Not an execute_reply, keep looking
                continue

            user_exprs = reply['content'].get('user_expressions', {})
            expr = user_exprs.get('_stats', {})
            if expr.get('status') == 'ok':
                text = expr['data']['text/plain']
                stats = ast.literal_eval(text)
                log(f"Scalene stats: footprint={stats.get('current_footprint_mb', 0):.2f}MB, "
                    f"malloc={stats.get('total_malloc_mb', 0):.2f}MB, "
                    f"max={stats.get('max_footprint_mb', 0):.2f}MB")
                return stats
            else:
                log(f"Scalene stats error: {expr.get('evalue', 'unknown')}")
                break
    except Exception as e:
        log(f"Failed to get Scalene stats: {e}")

    return {"current_footprint_mb": 0.0, "max_footprint_mb": 0.0, "gpu_mem_samples": 0.0}


def get_flowbook_checkpoint_var_costs(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get per-variable checkpoint memory costs from FlowBook kernel.

    Gets combined costs from both pre and post checkpoints for a cell.
    This gives the total memory overhead of checkpointing for that cell.

    Args:
        kernel_client: Kernel client to execute code on
        cell_id: Cell ID to get checkpoint costs for
        timeout: Timeout in seconds

    Returns:
        Dict with variable name -> {bytes, type, module, pre_bytes, post_bytes}
    """
    code = f"""
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
_ckpt_var_costs = {{}}
if hasattr(MemoryCheckpoints, '_instance') and MemoryCheckpoints._instance:
    _ckpt_var_costs = MemoryCheckpoints._instance.get_cell_checkpoint_costs("{cell_id}")
"""
    msg_id = kernel_client.execute(code, silent=True)
    _wait_for_idle(kernel_client, timeout)

    msg_id = kernel_client.execute(
        '',
        user_expressions={'_costs': '_ckpt_var_costs'},
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

            expr = reply['content'].get('user_expressions', {}).get('_costs', {})
            if expr.get('status') == 'ok':
                text = expr['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get checkpoint var costs: {e}")

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
    results = TimingResults(kernel_name="baseline_scalene_kernel")

    try:
        log("Baseline Timing: Starting baseline_scalene_kernel (no Scalene preload)...")
        kernel_manager, kernel_client = create_baseline_kernel(with_scalene_preload=False)
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
        log("FlowBook Timing: Starting flowbook_kernel (no Scalene preload)...")
        kernel_manager, kernel_client = create_flowbook_kernel(with_scalene_preload=False)
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
    Run notebook on baseline kernel with Scalene preload and collect MEMORY metrics.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds

    Returns:
        MemoryResults with cell memory data from Scalene
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    log(f"Baseline Memory: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = MemoryResults(kernel_name="baseline_scalene_kernel")

    try:
        log("Baseline Memory: Starting baseline_scalene_kernel WITH Scalene preload...")
        kernel_manager, kernel_client = create_baseline_kernel(with_scalene_preload=True)
        log("Baseline Memory: Kernel ready")

        # Enable Scalene tracking
        if enable_scalene_tracking(kernel_client):
            log("Baseline Memory: Scalene tracking enabled")
        else:
            log("Baseline Memory: WARNING: Failed to enable Scalene tracking")

        # Warmup
        warmup_code = "import pandas"
        kernel_client.execute(warmup_code, silent=True)
        _wait_for_idle(kernel_client)

        # Get baseline memory stats
        before_stats = get_scalene_memory_stats(kernel_client)

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"Baseline Memory: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            # Get stats before cell
            pre_stats = get_scalene_memory_stats(kernel_client)

            timing = execute_cell_baseline(kernel_client, source, cell_timeout)

            # Small delay for Scalene to process samples
            time.sleep(0.2)

            # Get stats after cell
            post_stats = get_scalene_memory_stats(kernel_client)

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
                allocation_delta = (
                    post_stats.get("total_malloc_mb", 0) - pre_stats.get("total_malloc_mb", 0)
                )
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    current_footprint_mb=post_stats.get("current_footprint_mb", 0.0),
                    max_footprint_mb=post_stats.get("max_footprint_mb", 0.0),
                    allocation_delta_mb=allocation_delta,
                    gpu_mem_samples=post_stats.get("gpu_mem_samples", 0.0),
                    status="ok",
                ))
                log(f"  Footprint: {post_stats.get('current_footprint_mb', 0):.1f}MB, "
                    f"Delta: {allocation_delta:.1f}MB, GPU: {post_stats.get('gpu_mem_samples', 0):.0f}")

        # Get final stats
        final_stats = get_scalene_memory_stats(kernel_client)
        results.totals = {
            "final_footprint_mb": final_stats.get("current_footprint_mb", 0.0),
            "max_footprint_mb": final_stats.get("max_footprint_mb", 0.0),
            "total_allocation_mb": final_stats.get("total_malloc_mb", 0.0) - before_stats.get("total_malloc_mb", 0.0),
            "gpu_mem_samples": final_stats.get("gpu_mem_samples", 0.0),
        }

        log(f"Baseline Memory: Final footprint {final_stats.get('current_footprint_mb', 0):.1f}MB, "
            f"Max {final_stats.get('max_footprint_mb', 0):.1f}MB")

        # Disable Scalene tracking
        disable_scalene_tracking(kernel_client)

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_memory(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
) -> MemoryResults:
    """
    Run notebook on FlowBook kernel with Scalene preload and collect MEMORY metrics.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds

    Returns:
        MemoryResults with cell memory data from Scalene + checkpoint var costs
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cell_order = [c.get("id", f"cell_{i}") for i, c in enumerate(code_cells)]

    log(f"FlowBook Memory: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    results = MemoryResults(kernel_name="flowbook_kernel")

    try:
        log("FlowBook Memory: Starting flowbook_kernel WITH Scalene preload...")
        kernel_manager, kernel_client = create_flowbook_kernel(with_scalene_preload=True)
        log("FlowBook Memory: Kernel ready")

        # Enable Scalene tracking
        if enable_scalene_tracking(kernel_client):
            log("FlowBook Memory: Scalene tracking enabled")
        else:
            log("FlowBook Memory: WARNING: Failed to enable Scalene tracking")

        # Enable continue_after_violation
        kernel_client.execute("%continue_after_violation on", silent=True)
        _wait_for_idle(kernel_client)

        # Small delay to let Scalene process any pending samples
        time.sleep(0.5)

        # Get baseline memory stats
        before_stats = get_scalene_memory_stats(kernel_client)

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"FlowBook Memory: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            # Get stats before cell
            pre_stats = get_scalene_memory_stats(kernel_client)

            timing = execute_cell_flowbook(
                kernel_client, source, cell_id, cell_order, cell_timeout
            )

            # Small delay for Scalene to process samples
            time.sleep(0.2)

            # Get stats after cell
            post_stats = get_scalene_memory_stats(kernel_client)

            # Get checkpoint variable costs (combined pre + post)
            var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id)

            # Debug logging for checkpoint costs
            if var_costs:
                log(f"  Checkpoint costs for {len(var_costs)} variables:")
                for var_name, info in list(var_costs.items())[:3]:
                    log(f"    {var_name}: {info.get('bytes', 0)/1024/1024:.2f}MB ({info.get('type')})")
                if len(var_costs) > 3:
                    log(f"    ... and {len(var_costs) - 3} more")
            else:
                log(f"  No checkpoint costs retrieved for cell {cell_id}")

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
                    status="error",
                    error=timing["error"]
                ))
            else:
                allocation_delta = (
                    post_stats.get("total_malloc_mb", 0) - pre_stats.get("total_malloc_mb", 0)
                )
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    current_footprint_mb=post_stats.get("current_footprint_mb", 0.0),
                    max_footprint_mb=post_stats.get("max_footprint_mb", 0.0),
                    allocation_delta_mb=allocation_delta,
                    gpu_mem_samples=post_stats.get("gpu_mem_samples", 0.0),
                    checkpoint_var_costs=var_costs if var_costs else None,
                    status="ok",
                ))
                ckpt_total = sum(v.get("bytes", 0) for v in var_costs.values()) / (1024 * 1024) if var_costs else 0
                log(f"  Footprint: {post_stats.get('current_footprint_mb', 0):.1f}MB, "
                    f"Delta: {allocation_delta:.1f}MB, Checkpoint: {ckpt_total:.1f}MB")

        # Get final stats
        final_stats = get_scalene_memory_stats(kernel_client)
        results.totals = {
            "final_footprint_mb": final_stats.get("current_footprint_mb", 0.0),
            "max_footprint_mb": final_stats.get("max_footprint_mb", 0.0),
            "total_allocation_mb": final_stats.get("total_malloc_mb", 0.0) - before_stats.get("total_malloc_mb", 0.0),
            "gpu_mem_samples": final_stats.get("gpu_mem_samples", 0.0),
        }

        log(f"FlowBook Memory: Final footprint {final_stats.get('current_footprint_mb', 0):.1f}MB, "
            f"Max {final_stats.get('max_footprint_mb', 0):.1f}MB")

        # Disable Scalene tracking
        disable_scalene_tracking(kernel_client)

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
        cell_timeout = kwargs.get("timeout", 300.0)
        skip_memory = kwargs.get("skip_memory", False)
        notebook_path = kwargs.get("notebook_path", "unknown.ipynb")

        cells = notebook_content.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        # Check if Scalene is available for memory phases
        scalene_available = _is_scalene_available() and not skip_memory

        with self.timing_context() as get_elapsed:
            log(f"Starting 4-phase baseline vs FlowBook comparison...")
            log(f"Notebook: {notebook_path}")
            log(f"Cell timeout: {cell_timeout}s")
            log(f"Scalene available: {scalene_available}")
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
            # PHASE 3: Baseline Memory (Scalene ON) - if available
            # ============================================================
            baseline_memory = None
            if scalene_available:
                log("=" * 60)
                log("PHASE 3: BASELINE MEMORY (Scalene ON)")
                log("=" * 60)
                baseline_memory = run_baseline_memory(notebook_content, cell_timeout)
                log("")

            # ============================================================
            # PHASE 4: FlowBook Memory (Scalene ON) - if available
            # ============================================================
            flowbook_memory = None
            if scalene_available:
                log("=" * 60)
                log("PHASE 4: FLOWBOOK MEMORY (Scalene ON)")
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
                        kernel_name="baseline_scalene_kernel",
                        timing=baseline_timing,
                        memory=baseline_memory,
                    ),
                    "flowbook": KernelResults(
                        kernel_name="flowbook_kernel",
                        timing=flowbook_timing,
                        memory=flowbook_memory,
                    ),
                },
                scalene_available=scalene_available,
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

            if scalene_available and baseline_memory and flowbook_memory:
                baseline_mem = baseline_memory.totals.get("final_footprint_mb", 0)
                flowbook_mem = flowbook_memory.totals.get("final_footprint_mb", 0)
                baseline_gpu = baseline_memory.totals.get("gpu_mem_samples", 0)
                flowbook_gpu = flowbook_memory.totals.get("gpu_mem_samples", 0)
                mem_overhead = flowbook_mem - baseline_mem

                log("MEMORY (from Scalene):")
                log(f"  Baseline footprint:   {baseline_mem:,.1f}MB")
                log(f"  FlowBook footprint:   {flowbook_mem:,.1f}MB")
                log(f"  Memory overhead:      {mem_overhead:+,.1f}MB")
                if baseline_gpu > 0 or flowbook_gpu > 0:
                    log(f"  Baseline GPU:         {baseline_gpu:,.0f} samples")
                    log(f"  FlowBook GPU:         {flowbook_gpu:,.0f} samples")
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
                "scalene_available": scalene_available,
            },
            total_cost=0.0,
            total_time=total_time,
        )
