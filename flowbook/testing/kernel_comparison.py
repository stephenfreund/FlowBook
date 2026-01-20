"""
Kernel comparison testing - Compare notebook execution times across kernel types.

This module provides utilities for comparing execution performance across:
- Base: Standard python3 kernel (baseline)
- FlowBook: flowbook_sdc_kernel (with SDC tracking overhead)
- Kishu: python3 kernel with Kishu extension enabled
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from jupyter_client import KernelManager

from flowbook import make_kernels
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.testing.notebook_loader import Cell
from flowbook.util.output import log, timer


class KernelType(Enum):
    """Types of kernels to compare."""
    BASE = "base"           # python3 kernel
    FLOWBOOK = "flowbook"   # flowbook_sdc_kernel
    KISHU = "kishu"         # python3 + Kishu extension


@dataclass
class CellTiming:
    """Timing result for a single cell execution."""
    cell_id: str
    time_ms: float
    status: str  # 'ok', 'error', 'timeout'
    error_message: Optional[str] = None


@dataclass
class KernelResult:
    """Results from running a notebook on a single kernel type."""
    kernel_type: KernelType
    cell_timings: List[CellTiming]
    total_time_ms: float
    setup_time_ms: float
    num_errors: int


@dataclass
class ComparisonResult:
    """Results from comparing multiple kernel types."""
    num_cells: int
    results: Dict[KernelType, KernelResult] = field(default_factory=dict)


def create_kernel(kernel_name: str) -> Tuple[KernelManager, FlowbookKernelClient]:
    """
    Start a kernel using jupyter_client.KernelManager.

    Follows pattern from flowbook/cli/helpers.py setup_kernel() with
    retry logic and wait_for_ready.

    Args:
        kernel_name: Name of kernel to start (e.g., 'python3', 'flowbook_sdc_kernel')

    Returns:
        Tuple of (KernelManager, FlowbookKernelClient)

    Raises:
        Exception: If kernel fails to start after max attempts
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
            kernel_manager = KernelManager(kernel_name=kernel_name)
            kernel_manager.start_kernel()

            kernel_client = FlowbookKernelClient(kernel_id=kernel_manager.kernel_id)
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

    # This should never be reached
    raise Exception("Kernel failed to start")


def cleanup_kernel(
    kernel_manager: Optional[KernelManager],
    kernel_client: Optional[FlowbookKernelClient]
) -> None:
    """
    Clean up kernel resources.

    Follows pattern from flowbook/cli/helpers.py cleanup_kernel().

    Args:
        kernel_manager: The kernel manager to shutdown
        kernel_client: The kernel client to stop
    """
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


def execute_cell_timed(
    kernel_client: FlowbookKernelClient,
    cell: Cell,
    timeout: float
) -> CellTiming:
    """
    Execute a cell and return timing information.

    Uses pattern from flowbook/server/kernel_helper.py KernelHelper.execute_code()
    with time.perf_counter() for timing.

    Args:
        kernel_client: The kernel client to use
        cell: The cell to execute
        timeout: Timeout in seconds

    Returns:
        CellTiming with execution results
    """
    start = time.perf_counter()

    msg_id = kernel_client.execute(cell.source, cell_id=cell.cell_id)

    status = 'ok'
    error_message = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            status = 'timeout'
            error_message = f'Execution timed out after {timeout} seconds'
            break

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg['parent_header'].get('msg_id') != msg_id:
            continue

        msg_type = msg['header']['msg_type']
        content = msg['content']

        if msg_type == 'error':
            status = 'error'
            error_message = '\n'.join([line.rstrip() for line in content.get('traceback', [])])

        elif msg_type == 'status':
            if content['execution_state'] == 'idle':
                break

    # Get the execute_reply message
    try:
        reply = kernel_client.get_shell_msg(timeout=1.0)
        reply_status = reply['content']['status']
        if reply_status == 'error':
            status = 'error'
            if error_message is None:
                error_content = reply['content']
                error_message = '\n'.join([
                    line.rstrip() for line in error_content.get('traceback', [])
                ])
    except Exception:
        pass

    elapsed_ms = (time.perf_counter() - start) * 1000

    return CellTiming(
        cell_id=cell.cell_id,
        time_ms=elapsed_ms,
        status=status,
        error_message=error_message
    )


def _wait_for_idle(kernel_client: FlowbookKernelClient, msg_id: str, timeout: float) -> bool:
    """Wait for kernel to become idle after an execution."""
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
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                return True
    return False


def run_kernel_execution(
    cells: List[Cell],
    kernel_type: KernelType,
    cell_timeout: float
) -> KernelResult:
    """
    Run complete notebook on one kernel type.

    Args:
        cells: List of cells to execute
        kernel_type: Which kernel type to use
        cell_timeout: Timeout per cell in seconds

    Returns:
        KernelResult with timing information
    """
    # Determine kernel name
    if kernel_type == KernelType.FLOWBOOK:
        kernel_name = 'flowbook_sdc_kernel'
    else:
        kernel_name = 'python3'

    kernel_manager = None
    kernel_client = None

    try:
        # Start kernel
        with timer(message=f"Starting {kernel_type.value} kernel") as setup_timer:
            kernel_manager, kernel_client = create_kernel(kernel_name)

            # For Kishu kernel, execute setup commands
            if kernel_type == KernelType.KISHU:
                log("Loading Kishu extension...")
                # Load the extension
                msg_id = kernel_client.execute("from kishu import init_kishu\ninit_kishu()\n")
                if not _wait_for_idle(kernel_client, msg_id, timeout=30.0):
                    log("Warning: Timeout waiting for Kishu extension to load")

                # # Enable Kishu
                # msg_id = kernel_client.execute("%kishu enable")
                # if not _wait_for_idle(kernel_client, msg_id, timeout=30.0):
                #     log("Warning: Timeout waiting for Kishu to enable")

                # Drain the shell channel
                try:
                    print(kernel_client.get_shell_msg(timeout=1.0))
                    # kernel_client.get_shell_msg(timeout=1.0)
                except Exception:
                    pass

        setup_time_ms = setup_timer.duration()

        # Execute cells
        cell_timings: List[CellTiming] = []
        total_time_ms = 0.0
        num_errors = 0

        for i, cell in enumerate(cells):
            log(f"Executing cell {i+1}/{len(cells)} ({cell.cell_id})...")
            timing = execute_cell_timed(kernel_client, cell, cell_timeout)
            cell_timings.append(timing)
            total_time_ms += timing.time_ms

            if timing.status != 'ok':
                num_errors += 1
                log(f"Cell {cell.cell_id} {timing.status}: {timing.error_message[:100] if timing.error_message else 'unknown'}")

        return KernelResult(
            kernel_type=kernel_type,
            cell_timings=cell_timings,
            total_time_ms=total_time_ms,
            setup_time_ms=setup_time_ms,
            num_errors=num_errors
        )

    finally:
        cleanup_kernel(kernel_manager, kernel_client)


def run_comparison(
    cells: List[Cell],
    kernels: List[KernelType],
    cell_timeout: float
) -> ComparisonResult:
    """
    Run comparison across specified kernel types.

    Args:
        cells: List of cells to execute
        kernels: List of kernel types to compare
        cell_timeout: Timeout per cell in seconds

    Returns:
        ComparisonResult with results from all kernels
    """
    result = ComparisonResult(num_cells=len(cells))

    for kernel_type in kernels:
        log(f"\n{'='*60}")
        log(f"Running on {kernel_type.value} kernel")
        log(f"{'='*60}")

        kernel_result = run_kernel_execution(cells, kernel_type, cell_timeout)
        result.results[kernel_type] = kernel_result

        log(f"Completed {kernel_type.value}: {kernel_result.total_time_ms:.1f}ms total, {kernel_result.num_errors} errors")

    return result


def format_comparison_table(
    result: ComparisonResult,
    notebook_path: str
) -> str:
    """
    Format results as a comparison table.

    Base column shows absolute time in ms.
    FlowBook and Kishu columns show slowdown ratio (e.g., 1.52x).

    Args:
        result: ComparisonResult from run_comparison
        notebook_path: Path to the notebook for display

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("KERNEL COMPARISON RESULTS")
    lines.append("=" * 80)
    lines.append(f"Notebook: {notebook_path}")
    lines.append(f"Cells: {result.num_cells}")
    lines.append("=" * 80)
    lines.append("")

    # Determine which columns to show
    has_base = KernelType.BASE in result.results
    has_flowbook = KernelType.FLOWBOOK in result.results
    has_kishu = KernelType.KISHU in result.results

    # Build header row
    header_parts = ["Cell Id   "]
    if has_base:
        header_parts.append("Base        ")
    if has_flowbook:
        header_parts.append("FlowBook    ")
    if has_kishu:
        header_parts.append("Kishu       ")
    lines.append("".join(header_parts))

    # Separator
    sep_parts = ["---------- "]
    if has_base:
        sep_parts.append("------------ ")
    if has_flowbook:
        sep_parts.append("------------ ")
    if has_kishu:
        sep_parts.append("------------ ")
    lines.append("".join(sep_parts))

    # Get base timings for ratio calculation
    base_timings: Dict[str, CellTiming] = {}
    if has_base:
        for ct in result.results[KernelType.BASE].cell_timings:
            base_timings[ct.cell_id] = ct

    # Get all cell IDs in order (from any kernel result)
    cell_ids = []
    for kernel_result in result.results.values():
        for ct in kernel_result.cell_timings:
            if ct.cell_id not in cell_ids:
                cell_ids.append(ct.cell_id)

    # Build timing lookups for each kernel
    flowbook_timings: Dict[str, CellTiming] = {}
    kishu_timings: Dict[str, CellTiming] = {}

    if has_flowbook:
        for ct in result.results[KernelType.FLOWBOOK].cell_timings:
            flowbook_timings[ct.cell_id] = ct

    if has_kishu:
        for ct in result.results[KernelType.KISHU].cell_timings:
            kishu_timings[ct.cell_id] = ct

    # Build rows
    for cell_id in cell_ids:
        row_parts = [f"{cell_id:<10} "]

        # Base column (absolute time)
        if has_base:
            bt = base_timings.get(cell_id)
            if bt is None:
                row_parts.append("N/A          ")
            elif bt.status != 'ok':
                row_parts.append("ERROR        ")
            else:
                row_parts.append(f"{bt.time_ms:>8.1f} ms  ")

        # FlowBook column (ratio)
        if has_flowbook:
            ft = flowbook_timings.get(cell_id)
            bt = base_timings.get(cell_id) if has_base else None

            if ft is None:
                row_parts.append("N/A          ")
            elif ft.status != 'ok':
                row_parts.append("ERROR        ")
            elif bt is None or bt.status != 'ok':
                # No base to compare, show absolute time
                row_parts.append(f"{ft.time_ms:>8.1f} ms  ")
            else:
                ratio = ft.time_ms / bt.time_ms if bt.time_ms > 0 else 0
                row_parts.append(f"{ratio:>8.2f}x    ")

        # Kishu column (ratio)
        if has_kishu:
            kt = kishu_timings.get(cell_id)
            bt = base_timings.get(cell_id) if has_base else None

            if kt is None:
                row_parts.append("N/A          ")
            elif kt.status != 'ok':
                row_parts.append("ERROR        ")
            elif bt is None or bt.status != 'ok':
                # No base to compare, show absolute time
                row_parts.append(f"{kt.time_ms:>8.1f} ms  ")
            else:
                ratio = kt.time_ms / bt.time_ms if bt.time_ms > 0 else 0
                row_parts.append(f"{ratio:>8.2f}x    ")

        lines.append("".join(row_parts))

    # Total row separator
    lines.append("".join(sep_parts))

    # Total row
    total_parts = ["Total      "]

    base_total = result.results[KernelType.BASE].total_time_ms if has_base else 0

    if has_base:
        total_parts.append(f"{base_total:>8.1f} ms  ")

    if has_flowbook:
        fb_total = result.results[KernelType.FLOWBOOK].total_time_ms
        if has_base and base_total > 0:
            ratio = fb_total / base_total
            total_parts.append(f"{ratio:>8.2f}x    ")
        else:
            total_parts.append(f"{fb_total:>8.1f} ms  ")

    if has_kishu:
        k_total = result.results[KernelType.KISHU].total_time_ms
        if has_base and base_total > 0:
            ratio = k_total / base_total
            total_parts.append(f"{ratio:>8.2f}x    ")
        else:
            total_parts.append(f"{k_total:>8.1f} ms  ")

    lines.append("".join(total_parts))

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
