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

v3.0 adds pre-cell memory measurements for cross-run comparison:
- Base_i = pre_namespace + pre_gpu (baseline run, before step i)
- Flow_i = pre_namespace + pre_gpu + pre_checkpoint + pre_enforcer (FlowBook run, before step i)
- MemoryOverhead_i = Flow_i - Base_i
- Checkpoint_i = MemoryOverhead_{i+1} - MemoryOverhead_i

Usage via CLI:
    flowbook compare-baseline notebook.ipynb                              # FlowBook only (default)
    flowbook compare-baseline notebook.ipynb --run-baseline               # Include baseline comparison
    flowbook compare-baseline notebook.ipynb --staleness-mode syntactic   # Use syntactic mode
    flowbook compare-baseline notebook.ipynb --df-subset-optimization     # Enable DF subset optimization
    flowbook compare-baseline notebook.ipynb --timeout 14400              # optional timeout (default: 4 hours)

OUTPUT JSON SCHEMA (version 3.0):
{
  "version": "3.0",
  "notebook_path": str,                    # Path to the notebook file
  "timestamp": str,                        # ISO format timestamp
  "scalene_available": bool,               # Whether HeapSizer was available
  "metadata": {
    "num_cells": int,                      # Number of code cells
    "timeout_seconds": float,              # Cell execution timeout
    "staleness_mode": str,                 # "semantic" or "syntactic"
    "rerun_k": int,                        # Number of rerun iterations (optional)
    "trial": int,                          # Trial number (optional, for multi-trial runs)
    "num_trials": int                      # Total trials (optional)
  },
  "kernels": {
    "baseline": {                          # Always present (memory/timing null if skipped)
      "kernel_name": "baseline_kernel",
      "timing": TimingResults | null,
      "memory": MemoryResults | null
    },
    "flowbook": {
      "kernel_name": "flowbook_kernel",
      "timing": TimingResults,
      "memory": MemoryResults | null       # null if --skip-memory
    }
  }
}

TimingResults: (unchanged from v2.0)
{
  "kernel_name": str,
  "cells": [TimingCellMetrics, ...],
  "rerun_cells": [TimingCellMetrics, ...],
  "totals": {
    "execute_duration_ms": float,
    "code_duration_ms": float,             # FlowBook only
    "state_duration_ms": float,            # FlowBook only
    "check_duration_ms": float             # FlowBook only
  }
}

TimingCellMetrics: (unchanged from v2.0)
{
  "cell_id": str,
  "cell_index": int,
  "execute_duration_ms": float,
  "code_duration_ms": float,
  "state_duration_ms": float,
  "check_duration_ms": float,
  "status": str,
  "error": str | null,
  "is_rerun": bool,
  "checking_result": { ... } | null
}

MemoryResults:
{
  "kernel_name": str,
  "cells": [MemoryCellMetrics, ...],
  "rerun_cells": [MemoryCellMetrics, ...],
  "totals": {
    "final_namespace_mb": float,           # User namespace after all cells
    "final_gpu_mb": float,                 # GPU memory after all cells
    "max_namespace_mb": float,             # Peak namespace across all cells
    # FlowBook only:
    "final_checkpoint_cumulative_mb": float,  # Checkpoint overhead after all cells
    "final_enforcer_overhead_mb": float,      # Enforcer metadata after all cells
    "memory_overhead_ratio": float            # (namespace + checkpoints) / namespace
  }
}

MemoryCellMetrics:
{
  "cell_id": str,
  "cell_index": int,
  "pre_namespace_mb": float,               # User namespace BEFORE this cell
  "pre_gpu_mb": float,                     # GPU memory BEFORE this cell
  "namespace_mb": float,                   # User namespace AFTER this cell
  "gpu_mb": float,                         # GPU memory AFTER this cell
  "checkpoint_delta_mb": float,            # This cell's checkpoint contribution (FlowBook only)
  "checkpoint_cumulative_mb": float,       # Cumulative checkpoint overhead (FlowBook only)
  "pre_checkpoint_cumulative_mb": float,   # Checkpoints BEFORE this cell (FlowBook only)
  "pre_enforcer_overhead_mb": float,       # Enforcer metadata BEFORE this cell (FlowBook only)
  "checkpoint_by_var": {var: mb} | null,   # Per-variable checkpoint breakdown
  "checkpoint_var_costs": {var: {bytes, deepcopy_ms}} | null,
  "status": str,
  "error": str | null,
  "is_rerun": bool
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
from flowbook.cli.models import (
    BaselineMemorySnapshot,
    BaselineCellMemory,
    BaselineMemoryResult,
    CheckpointVarInfo,
    CheckpointVarSizes,
    FlowBookMemorySnapshot,
    FlowBookCellMemory,
    FlowBookMemoryResult,
    ComparisonMetadata,
    ComparisonResult as ComparisonResultV4,
    # V5 simplified models
    V5CellMemory,
    V5MemoryResult,
)


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
    gpu_mb: float = 0.0  # GPU memory after this cell (pynvml, for GPU overhead diff)


@dataclass
class MemoryCellMetrics:
    """Memory metrics for a single cell execution.

    Simplified structure with clear semantics:
    - pre_namespace_mb / pre_gpu_mb: Memory BEFORE this cell executes (for cross-run comparison)
    - namespace_mb / gpu_mb: Memory AFTER this cell executes (for Plot 4 stacked area)
    - checkpoint_delta_mb: What THIS cell's checkpoint adds (beyond ns + prior ckpts)
    - checkpoint_cumulative_mb: Total checkpoint overhead so far (after this cell)
    - pre_checkpoint_cumulative_mb: Checkpoint overhead BEFORE this cell (FlowBook only)
    - pre_enforcer_overhead_mb: Enforcer metadata BEFORE this cell (FlowBook only)
    - checkpoint_by_var: Per-variable breakdown of checkpoint memory (aggregated)
    - by_checkpoint_by_var: Per-checkpoint, per-variable breakdown (for v4.0 format)
    """
    cell_id: str
    cell_index: int
    pre_namespace_mb: float          # User namespace BEFORE this cell
    pre_gpu_mb: float                # GPU memory BEFORE this cell
    namespace_mb: float              # User namespace AFTER this cell
    checkpoint_delta_mb: float       # This cell's checkpoint contribution
    checkpoint_cumulative_mb: float  # Total checkpoint overhead after this cell
    gpu_mb: float                    # GPU memory AFTER this cell
    pre_checkpoint_cumulative_mb: float = 0.0  # Checkpoints BEFORE this cell (FlowBook only)
    pre_enforcer_overhead_mb: float = 0.0      # Enforcer metadata BEFORE this cell (FlowBook only)
    checkpoint_by_var: Optional[Dict[str, float]] = None  # Per-variable MB (aggregated)
    checkpoint_var_costs: Optional[Dict[str, Any]] = None  # Per-variable timing (for timing plots)
    by_checkpoint_by_var: Optional[Dict[str, Dict[str, float]]] = None  # {ckpt: {var: mb}}
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
    version: str = "3.0"
    notebook_path: str = ""
    timestamp: str = ""
    kernels: Dict[str, KernelResults] = field(default_factory=dict)
    scalene_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerunOverheadMeasurement:
    """A single rerun overhead measurement."""
    iteration: int
    cell_id: str
    cell_index: int
    checkpoint_ms: float
    check_ms: float
    total_overhead_ms: float
    checkpoint_by_var: Dict[str, float] = field(default_factory=dict)
    checkpoint_var_costs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerunOverheadResult:
    """Result of rerun overhead measurements."""
    rerun_n: int
    quartile_indices: List[int]
    measurements: List[RerunOverheadMeasurement] = field(default_factory=list)


# ============ V4 Conversion Functions ============


def _convert_checkpoint_vars_to_v4(
    by_checkpoint_by_var: Dict[str, Dict[str, float]],
    checkpoint_var_costs: Optional[Dict[str, Any]] = None,
) -> Dict[str, CheckpointVarSizes]:
    """Convert per-checkpoint, per-variable data to v4.0 format.

    Args:
        by_checkpoint_by_var: {checkpoint_name: {var_name: size_mb}}
        checkpoint_var_costs: Optional timing info {var_name: {bytes, type, module, ...}}

    Returns:
        Dict[checkpoint_name, CheckpointVarSizes]
    """
    result: Dict[str, CheckpointVarSizes] = {}
    for ckpt_name, var_sizes in by_checkpoint_by_var.items():
        vars_info: Dict[str, CheckpointVarInfo] = {}
        for var_name, size_mb in var_sizes.items():
            # Get type info from checkpoint_var_costs if available
            type_name = ""
            module = ""
            if checkpoint_var_costs and var_name in checkpoint_var_costs:
                cost_info = checkpoint_var_costs[var_name]
                if isinstance(cost_info, dict):
                    type_name = cost_info.get("type", "")
                    module = cost_info.get("module", "")
            vars_info[var_name] = CheckpointVarInfo(
                size_mb=size_mb,
                type_name=type_name,
                module=module,
            )
        result[ckpt_name] = CheckpointVarSizes(vars=vars_info)
    return result


def _convert_flowbook_cell_to_v4(cell: MemoryCellMetrics) -> FlowBookCellMemory:
    """Convert internal MemoryCellMetrics to v4.0 FlowBookCellMemory.

    Args:
        cell: Internal cell metrics

    Returns:
        FlowBookCellMemory in v4.0 format
    """
    # Build checkpoint_vars from by_checkpoint_by_var if available
    post_checkpoint_vars = {}
    if cell.by_checkpoint_by_var:
        post_checkpoint_vars = _convert_checkpoint_vars_to_v4(
            cell.by_checkpoint_by_var, cell.checkpoint_var_costs
        )

    return FlowBookCellMemory(
        cell_id=cell.cell_id,
        cell_index=cell.cell_index,
        pre=FlowBookMemorySnapshot(
            user_ns_mb=cell.pre_namespace_mb,
            gpu_mb=cell.pre_gpu_mb,
            overhead_mb=cell.pre_checkpoint_cumulative_mb + cell.pre_enforcer_overhead_mb,
            checkpoint_vars={},  # Pre-checkpoints not tracked at this granularity
        ),
        post=FlowBookMemorySnapshot(
            user_ns_mb=cell.namespace_mb,
            gpu_mb=cell.gpu_mb,
            overhead_mb=cell.checkpoint_cumulative_mb,
            checkpoint_vars=post_checkpoint_vars,
        ),
        status=cell.status,
        error=cell.error,
    )


def _convert_baseline_cell_to_v4(cell: MemoryCellMetrics) -> BaselineCellMemory:
    """Convert internal MemoryCellMetrics to v4.0 BaselineCellMemory."""
    return BaselineCellMemory(
        cell_id=cell.cell_id,
        cell_index=cell.cell_index,
        pre=BaselineMemorySnapshot(
            user_ns_mb=cell.pre_namespace_mb,
            gpu_mb=cell.pre_gpu_mb,
        ),
        post=BaselineMemorySnapshot(
            user_ns_mb=cell.namespace_mb,
            gpu_mb=cell.gpu_mb,
        ),
        status=cell.status,
        error=cell.error,
    )


def _convert_memory_results_to_v4(
    results: MemoryResults,
    is_flowbook: bool,
):
    """Convert MemoryResults to v4.0 format.

    Args:
        results: Internal memory results
        is_flowbook: True for FlowBook kernel, False for baseline

    Returns:
        FlowBookMemoryResult or BaselineMemoryResult in v4.0 format
    """
    if is_flowbook:
        return FlowBookMemoryResult(
            cells=[_convert_flowbook_cell_to_v4(c) for c in results.cells],
            rerun_cells=[_convert_flowbook_cell_to_v4(c) for c in results.rerun_cells],
        )
    else:
        return BaselineMemoryResult(
            cells=[_convert_baseline_cell_to_v4(c) for c in results.cells],
            rerun_cells=[_convert_baseline_cell_to_v4(c) for c in results.rerun_cells],
        )


def create_v4_comparison_result(
    notebook_path: str,
    timestamp: str,
    metadata_dict: Dict[str, Any],
    baseline_timing: Optional[TimingResults],
    flowbook_timing: Optional[TimingResults],
    baseline_memory: Optional[MemoryResults],
    flowbook_memory: Optional[MemoryResults],
) -> Dict[str, Any]:
    """Create v4.0 format comparison result dict ready for JSON serialization.

    Args:
        notebook_path: Path to notebook
        timestamp: ISO timestamp
        metadata_dict: Metadata dict
        baseline_timing: Baseline timing results
        flowbook_timing: FlowBook timing results
        baseline_memory: Baseline memory results
        flowbook_memory: FlowBook memory results

    Returns:
        Dict in v4.0 format ready for json.dump()
    """
    # Build metadata
    metadata = ComparisonMetadata(
        staleness_mode=metadata_dict.get("staleness_mode", "semantic"),
        num_cells=metadata_dict.get("num_cells", 0),
        timeout_seconds=metadata_dict.get("timeout_seconds", 0.0),
        notebook_path=notebook_path,
        timestamp=timestamp,
    )

    # Convert memory results
    baseline_v4 = None
    if baseline_memory:
        baseline_v4 = _convert_memory_results_to_v4(baseline_memory, is_flowbook=False)

    flowbook_v4 = None
    if flowbook_memory:
        flowbook_v4 = _convert_memory_results_to_v4(flowbook_memory, is_flowbook=True)

    # Build result
    result = ComparisonResultV4(
        version="4.0",
        metadata=metadata,
        baseline=baseline_v4,
        flowbook=flowbook_v4,
        timing=None,  # Will add below
    )

    # Build dict
    result_dict = result.to_dict()

    # Add timing data to kernels (kept in original format for now)
    if "kernels" not in result_dict:
        result_dict["kernels"] = {}

    if baseline_timing or baseline_v4:
        if "baseline" not in result_dict["kernels"]:
            result_dict["kernels"]["baseline"] = {}
        if baseline_timing:
            result_dict["kernels"]["baseline"]["timing"] = _timing_results_to_dict(baseline_timing)

    if flowbook_timing or flowbook_v4:
        if "flowbook" not in result_dict["kernels"]:
            result_dict["kernels"]["flowbook"] = {}
        if flowbook_timing:
            result_dict["kernels"]["flowbook"]["timing"] = _timing_results_to_dict(flowbook_timing)

    return result_dict


# ============ V5 Conversion Functions ============


def create_v5_comparison_result(
    notebook_path: str,
    timestamp: str,
    metadata_dict: Dict[str, Any],
    baseline_timing: Optional[TimingResults],
    flowbook_timing: Optional[TimingResults],
    baseline_memory: Optional["V5MemoryResult"],
    flowbook_memory: Optional["V5MemoryResult"],
) -> Dict[str, Any]:
    """Create v5.0 format comparison result dict ready for JSON serialization.

    V5 simplifies memory data by:
    - Removing pre_* fields (only post-execution state needed)
    - Flattening nested checkpoint structure to single checkpoint_mb
    - Aggregating per-variable checkpoint sizes across all checkpoints

    Args:
        notebook_path: Path to notebook
        timestamp: ISO timestamp
        metadata_dict: Metadata dict
        baseline_timing: Baseline timing results
        flowbook_timing: FlowBook timing results
        baseline_memory: Baseline memory results (V5MemoryResult)
        flowbook_memory: FlowBook memory results (V5MemoryResult)

    Returns:
        Dict in v5.0 format ready for json.dump()
    """
    result: Dict[str, Any] = {
        "version": "5.0",
        "metadata": {
            "staleness_mode": metadata_dict.get("staleness_mode", "semantic"),
            "num_cells": metadata_dict.get("num_cells", 0),
            "timeout_seconds": metadata_dict.get("timeout_seconds", 0.0),
            "notebook_path": notebook_path,
            "timestamp": timestamp,
        },
        "kernels": {},
    }

    # Add baseline if present
    if baseline_timing or baseline_memory:
        result["kernels"]["baseline"] = {}
        if baseline_timing:
            result["kernels"]["baseline"]["timing"] = _timing_results_to_dict(baseline_timing)
        if baseline_memory:
            result["kernels"]["baseline"]["memory"] = baseline_memory.to_dict()

    # Add flowbook if present
    if flowbook_timing or flowbook_memory:
        result["kernels"]["flowbook"] = {}
        if flowbook_timing:
            result["kernels"]["flowbook"]["timing"] = _timing_results_to_dict(flowbook_timing)
        if flowbook_memory:
            result["kernels"]["flowbook"]["memory"] = flowbook_memory.to_dict()

    return result


def _timing_results_to_dict(timing: TimingResults) -> Dict[str, Any]:
    """Convert TimingResults to dict for JSON serialization."""
    def cell_to_dict(cell: TimingCellMetrics) -> Dict[str, Any]:
        d = {
            "cell_id": cell.cell_id,
            "cell_index": cell.cell_index,
            "execute_duration_ms": cell.execute_duration_ms,
            "code_duration_ms": cell.code_duration_ms,
            "state_duration_ms": cell.state_duration_ms,
            "check_duration_ms": cell.check_duration_ms,
            "status": cell.status,
            "error": cell.error,
            "is_rerun": cell.is_rerun,
        }
        if cell.gpu_mb > 0:
            d["gpu_mb"] = cell.gpu_mb
        if cell.checking_result:
            d["checking_result"] = cell.checking_result.to_dict()
        return d

    return {
        "kernel_name": timing.kernel_name,
        "cells": [cell_to_dict(c) for c in timing.cells],
        "rerun_cells": [cell_to_dict(c) for c in timing.rerun_cells],
        "totals": timing.totals,
    }


def _convert_memory_results_to_v5(memory: MemoryResults) -> "V5MemoryResult":
    """Convert MemoryResults to V5MemoryResult for baseline compatibility.

    Baseline doesn't have checkpoints, so checkpoint_mb is always 0.
    """
    v5_cells = []
    for cell in memory.cells:
        v5_cells.append(V5CellMemory(
            cell_id=cell.cell_id,
            cell_index=cell.cell_index,
            user_ns_mb=cell.namespace_mb,
            gpu_mb=cell.gpu_mb,
            checkpoint_mb=0.0,  # Baseline has no checkpoints
            checkpoint_vars={},
        ))
    return V5MemoryResult(cells=v5_cells)


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

    # Set extra environment variables (kernel inherits from parent process)
    old_env = {}
    if extra_env:
        for key, value in extra_env.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

    try:
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
    finally:
        # Restore original environment
        if extra_env:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value


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
    # Filter out private/system variables, functions, and modules.
    # NOTE: callable() is too aggressive - it excludes DataFrames (which have __call__
    # via the type system) and cudf.pandas proxy objects. Use explicit type checks instead.
    # NOTE: We use sizeof_user_namespace() which handles filtering internally
    # to avoid __import__('types') in user_expressions, which can trigger
    # dbm imports on systems without _gdbm/_dbm C extensions.
    expr_code = """__import__('flowbook.kernel_support.heap_size', fromlist=['HeapSizer']).HeapSizer().sizeof_user_namespace(globals()).__dict__"""

    msg_id = kernel_client.execute(
        '',  # Empty code - no checkpoints created
        user_expressions={'_ns_size': expr_code},
        silent=True,
    )

    # Wait for idle on iopub channel, capturing any stream output
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
        msg_type = msg['header']['msg_type']
        # Print any stream output (stdout/stderr) from the kernel
        if msg_type == 'stream':
            text = msg['content'].get('text', '')
            if text.strip():
                log(f"[kernel {msg['content'].get('name', 'output')}] {text.strip()}")
        if msg_type == 'status':
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
    # Use get_overhead_beyond_user_namespace() which handles filtering internally
    # to avoid __import__('types') in user_expressions, which can trigger
    # dbm imports on systems without _gdbm/_dbm C extensions.
    expr = (
        f"__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_overhead_beyond_user_namespace('{cell_id}', globals())"
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


def get_v5_memory_snapshot(kernel_client, timeout: float = 30.0) -> Dict[str, Any]:
    """Get unified v5 memory snapshot: namespace + checkpoint overhead.

    This is the simplified v5 API that measures everything in a single call:
    1. User namespace size (filtered)
    2. Total checkpoint overhead BEYOND namespace (handles CoW sharing)
    3. Per-variable checkpoint sizes aggregated across all checkpoints

    The key simplification is that HeapSizer measures namespace FIRST (populating
    seen_ids), then measures checkpoints (only counting IDs NOT already seen).
    This correctly handles CoW sharing without complex intermediate structures.

    Args:
        kernel_client: Kernel client to execute code on
        timeout: Timeout in seconds

    Returns:
        Dict with:
        - user_ns_bytes: int - Size of user namespace objects
        - gpu_bytes: int - GPU memory (if torch available)
        - checkpoint_bytes: int - Total checkpoint overhead beyond namespace
        - checkpoint_vars: Dict[str, int] - {var_name: bytes} aggregated across checkpoints
    """
    expr = (
        "__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        ".MemoryCheckpoints._instance.get_memory_snapshot(globals())"
    )

    msg_id = kernel_client.execute('', user_expressions={'_snapshot': expr}, silent=True)

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {'user_ns_bytes': 0, 'gpu_bytes': 0, 'checkpoint_bytes': 0, 'checkpoint_vars': {}, 'gpu_checkpoint_bytes': 0, 'gpu_checkpoint_vars': {}}
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

            expr_result = reply['content'].get('user_expressions', {}).get('_snapshot', {})
            if expr_result.get('status') == 'ok':
                text = expr_result['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get v5 memory snapshot: {e}")
        log(traceback.format_exc())

    return {'user_ns_bytes': 0, 'gpu_bytes': 0, 'checkpoint_bytes': 0, 'checkpoint_vars': {}, 'gpu_checkpoint_bytes': 0, 'gpu_checkpoint_vars': {}}


def get_logical_checkpoint_sizes(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get LOGICAL checkpoint sizes - what's stored, regardless of sharing.

    Unlike get_checkpoint_overhead which measures UNIQUE memory not shared
    with namespace, this measures what data IS in each checkpoint. Useful
    for understanding checkpoint content even when CoW means overhead is 0.

    Args:
        kernel_client: Kernel client to execute code on
        cell_id: Cell ID to measure up to (inclusive)
        timeout: Timeout in seconds

    Returns:
        Dict with:
        - total_mb: Sum of all checkpoint logical sizes
        - by_checkpoint: Per-checkpoint logical size in MB
        - by_variable: Per-variable totals in MB (across all checkpoints)
        - by_checkpoint_by_var: {checkpoint_name: {var_name: mb}}
    """
    expr = (
        f"__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_logical_checkpoint_sizes('{cell_id}')"
    )

    msg_id = kernel_client.execute('', user_expressions={'_logical': expr}, silent=True)

    # Wait for idle on iopub
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {'total_mb': 0, 'by_checkpoint': {}, 'by_variable': {}, 'by_checkpoint_by_var': {}}
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

            expr_result = reply['content'].get('user_expressions', {}).get('_logical', {})
            if expr_result.get('status') == 'ok':
                text = expr_result['data']['text/plain']
                return ast.literal_eval(text)
            break
    except Exception as e:
        log(f"Failed to get logical checkpoint sizes: {e}")
        log(traceback.format_exc())

    return {'total_mb': 0, 'by_checkpoint': {}, 'by_variable': {}, 'by_checkpoint_by_var': {}}


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
        del enforcer
except Exception:
    pass

# Clean up temp variables to avoid polluting namespace
for _v in ['mc', 'all_sizes', 'num_vars']:
    if _v in dir():
        exec(f'del {_v}')
del _v
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
                result = ast.literal_eval(text)
                # Clean up _overhead_breakdown from namespace
                kernel_client.execute('del _overhead_breakdown', silent=True)
                return result
            break
    except Exception as e:
        log(f"Failed to get overhead breakdown: {e}")
        log(traceback.format_exc())

    # Clean up even on failure
    try:
        kernel_client.execute('del _overhead_breakdown', silent=True)
    except Exception:
        pass

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

            # Capture GPU memory from kernel process (for diff-based overhead)
            cell_gpu_mb = get_kernel_gpu_memory_mb(kernel_client)

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
                    gpu_mb=cell_gpu_mb,
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
                    gpu_mb=cell_gpu_mb,
                ))
                log(f"  Runtime: {runtime_ms:.1f}ms")

        results.totals = {
            "execute_duration_ms": total_runtime_ms,
            "code_duration_ms": total_runtime_ms,  # For baseline, code time equals total time
        }

        log(f"Baseline Timing: Total runtime {total_runtime_ms:.1f}ms")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_timing(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    staleness_mode: str = "syntactic",
    rerun_n: int = 0,
) -> Tuple[TimingResults, Optional[RerunOverheadResult]]:
    """
    Run notebook on FlowBook kernel and collect TIMING metrics only (Scalene OFF).

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        staleness_mode: Staleness computation mode ('syntactic' or 'semantic')
        rerun_n: Number of rerun overhead measurement iterations at quartile cells

    Returns:
        Tuple of (TimingResults, Optional[RerunOverheadResult])
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

            # Capture GPU memory from kernel process (for diff-based overhead)
            cell_gpu_mb = get_kernel_gpu_memory_mb(kernel_client)

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
                    gpu_mb=cell_gpu_mb,
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
                    gpu_mb=cell_gpu_mb,
                ))
                log(f"  Execute: {execute_ms:.1f}ms, Code: {code_ms:.1f}ms, State: {state_ms:.1f}ms, Check: {check_ms:.1f}ms")

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
            "checking_summary": {
                "clean_cells": clean_count,
                "stale_cells": stale_count,
                "error_cells": error_count,
                "reason_counts": reason_counts,
                "error_counts": error_counts,
            },
        }

        log(f"FlowBook Timing: Total execute {total_execute_ms:.1f}ms, code {total_code_ms:.1f}ms, state {total_state_ms:.1f}ms, check {total_check_ms:.1f}ms")

        # Measure rerun overhead if requested (using the same kernel that has all state)
        rerun_overhead_result = None
        if rerun_n > 0:
            log("")
            log("=" * 60)
            log("RERUN OVERHEAD MEASUREMENT")
            log("=" * 60)
            rerun_overhead_result = measure_rerun_overhead(
                kernel_client, code_cells, rerun_n, timeout=cell_timeout
            )
            log("")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results, rerun_overhead_result


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

            # Measure BEFORE cell execution (for cross-run comparison)
            pre_stats = get_namespace_size(kernel_client)
            pre_gpu = get_kernel_gpu_memory_mb(kernel_client)

            timing = execute_cell_baseline(kernel_client, source, cell_timeout)

            # Measure AFTER cell execution (for Plot 4 stacked area)
            post_stats = get_namespace_size(kernel_client)
            post_gpu = get_kernel_gpu_memory_mb(kernel_client)

            pre_mb = pre_stats.get("total_mb", 0.0)
            current_mb = post_stats.get("total_mb", 0.0)
            max_footprint_mb = max(max_footprint_mb, current_mb)

            if timing.get("error"):
                log(f"  Error:\n{timing['error']}")
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    pre_namespace_mb=pre_mb,
                    pre_gpu_mb=pre_gpu,
                    namespace_mb=0.0,
                    checkpoint_delta_mb=0.0,
                    checkpoint_cumulative_mb=0.0,
                    gpu_mb=post_gpu,
                    status="error",
                    error=timing["error"]
                ))
            else:
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    pre_namespace_mb=pre_mb,
                    pre_gpu_mb=pre_gpu,
                    namespace_mb=current_mb,
                    checkpoint_delta_mb=0.0,  # Baseline has no checkpoints
                    checkpoint_cumulative_mb=0.0,
                    gpu_mb=post_gpu,
                    status="ok",
                ))
                log(f"  Pre: {pre_mb:.1f}MB, Post: {current_mb:.1f}MB")

        # Get final stats (used as Base_{N} in cross-run formula)
        final_stats = get_namespace_size(kernel_client)
        final_gpu_mem = get_kernel_gpu_memory_mb(kernel_client)
        results.totals = {
            "final_namespace_mb": final_stats.get("total_mb", 0.0),
            "final_gpu_mb": final_gpu_mem,
            "max_namespace_mb": max_footprint_mb,
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
    staleness_mode: str = "syntactic",
) -> MemoryResults:
    """
    Run notebook on FlowBook kernel and collect MEMORY metrics using HeapSizer.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        rerun_k: Number of rerun iterations
        staleness_mode: Staleness computation mode ('syntactic' or 'semantic')

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

        max_footprint_mb = 0.0
        prev_cell_id = None

        def _get_pre_flowbook_overhead(kc, prev_cid):
            """Get checkpoint + enforcer overhead BEFORE the current cell."""
            if prev_cid is None:
                return 0.0, 0.0
            pre_ckpt = get_checkpoint_overhead(kc, prev_cid)
            pre_ckpt_mb = pre_ckpt.get('total_mb', 0.0)
            breakdown = get_flowbook_overhead_breakdown(kc)
            pre_enforcer_mb = (breakdown.get("execution_records_mb", 0.0)
                               + breakdown.get("tracking_metadata_mb", 0.0)
                               + breakdown.get("other_mb", 0.0))
            return pre_ckpt_mb, pre_enforcer_mb

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"FlowBook Memory: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            # Measure BEFORE cell execution (for cross-run comparison)
            pre_stats = get_namespace_size(kernel_client)
            pre_gpu = get_kernel_gpu_memory_mb(kernel_client)
            pre_ckpt_mb, pre_enforcer_mb = _get_pre_flowbook_overhead(kernel_client, prev_cell_id)

            timing = execute_cell_flowbook(
                kernel_client, source, cell_id, cell_order, cell_timeout
            )

            # Measure AFTER cell execution (for Plot 4 stacked area)
            post_stats = get_namespace_size(kernel_client)
            post_gpu = get_kernel_gpu_memory_mb(kernel_client)

            pre_mb = pre_stats.get("total_mb", 0.0)
            namespace_mb = post_stats.get("total_mb", 0.0)
            max_footprint_mb = max(max_footprint_mb, namespace_mb)

            # Get checkpoint overhead using the cumulative measurement approach
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
            by_checkpoint_by_var = overhead.get('by_checkpoint_by_var') or None

            # Get per-variable checkpoint costs (includes deepcopy_ms for timing plots)
            checkpoint_var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id) or None

            # Debug logging
            log(f"  Pre: {pre_mb:.1f}MB, Post: {namespace_mb:.1f}MB, "
                f"Ckpt delta: {checkpoint_delta_mb:.1f}MB, Cumulative: {checkpoint_cumulative_mb:.1f}MB, "
                f"GPU: {post_gpu:.1f}MB, Pre-ckpt: {pre_ckpt_mb:.1f}MB, Enforcer: {pre_enforcer_mb:.3f}MB")
            if checkpoint_by_var:
                top_vars = sorted(checkpoint_by_var.items(), key=lambda x: x[1], reverse=True)[:3]
                log(f"  Top checkpoint vars: {', '.join(f'{k}={v:.2f}MB' for k, v in top_vars)}")

            if timing.get("error") and timing.get("cell_runtime_ms") is None:
                log(f"  Error:\n{timing['error']}")
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    pre_namespace_mb=pre_mb,
                    pre_gpu_mb=pre_gpu,
                    namespace_mb=0.0,
                    checkpoint_delta_mb=checkpoint_delta_mb,
                    checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                    gpu_mb=post_gpu,
                    pre_checkpoint_cumulative_mb=pre_ckpt_mb,
                    pre_enforcer_overhead_mb=pre_enforcer_mb,
                    checkpoint_by_var=checkpoint_by_var,
                    checkpoint_var_costs=checkpoint_var_costs,
                    by_checkpoint_by_var=by_checkpoint_by_var,
                    status="error",
                    error=timing["error"]
                ))
            else:
                results.cells.append(MemoryCellMetrics(
                    cell_id=cell_id,
                    cell_index=idx,
                    pre_namespace_mb=pre_mb,
                    pre_gpu_mb=pre_gpu,
                    namespace_mb=namespace_mb,
                    checkpoint_delta_mb=checkpoint_delta_mb,
                    checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                    gpu_mb=post_gpu,
                    pre_checkpoint_cumulative_mb=pre_ckpt_mb,
                    pre_enforcer_overhead_mb=pre_enforcer_mb,
                    checkpoint_by_var=checkpoint_by_var,
                    checkpoint_var_costs=checkpoint_var_costs,
                    by_checkpoint_by_var=by_checkpoint_by_var,
                    status="ok",
                ))

            prev_cell_id = cell_id

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

                    # Measure BEFORE cell execution
                    pre_stats = get_namespace_size(kernel_client)
                    pre_gpu = get_kernel_gpu_memory_mb(kernel_client)
                    pre_ckpt_mb, pre_enforcer_mb = _get_pre_flowbook_overhead(kernel_client, prev_cell_id)

                    timing = execute_cell_flowbook(
                        kernel_client, source, cell_id, cell_order, cell_timeout
                    )

                    # Measure AFTER cell execution
                    post_stats = get_namespace_size(kernel_client)
                    post_gpu = get_kernel_gpu_memory_mb(kernel_client)
                    pre_mb = pre_stats.get("total_mb", 0.0)
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
                    by_checkpoint_by_var = overhead.get('by_checkpoint_by_var') or None

                    # Get per-variable checkpoint costs (includes deepcopy_ms for timing plots)
                    checkpoint_var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id) or None

                    if timing.get("error") and timing.get("cell_runtime_ms") is None:
                        log(f"  Rerun Error:\n{timing['error']}")
                        results.rerun_cells.append(MemoryCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            pre_namespace_mb=pre_mb,
                            pre_gpu_mb=pre_gpu,
                            namespace_mb=0.0,
                            checkpoint_delta_mb=checkpoint_delta_mb,
                            checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                            gpu_mb=post_gpu,
                            pre_checkpoint_cumulative_mb=pre_ckpt_mb,
                            pre_enforcer_overhead_mb=pre_enforcer_mb,
                            checkpoint_by_var=checkpoint_by_var,
                            checkpoint_var_costs=checkpoint_var_costs,
                            by_checkpoint_by_var=by_checkpoint_by_var,
                            status="error",
                            error=timing["error"],
                            is_rerun=True,
                        ))
                    else:
                        results.rerun_cells.append(MemoryCellMetrics(
                            cell_id=cell_id,
                            cell_index=idx,
                            pre_namespace_mb=pre_mb,
                            pre_gpu_mb=pre_gpu,
                            namespace_mb=namespace_mb,
                            checkpoint_delta_mb=checkpoint_delta_mb,
                            checkpoint_cumulative_mb=checkpoint_cumulative_mb,
                            gpu_mb=post_gpu,
                            pre_checkpoint_cumulative_mb=pre_ckpt_mb,
                            pre_enforcer_overhead_mb=pre_enforcer_mb,
                            checkpoint_by_var=checkpoint_by_var,
                            checkpoint_var_costs=checkpoint_var_costs,
                            by_checkpoint_by_var=by_checkpoint_by_var,
                            status="ok",
                            is_rerun=True,
                        ))
                        log(f"  Rerun Pre: {pre_mb:.1f}MB, Post: {namespace_mb:.1f}MB, "
                            f"Delta: {checkpoint_delta_mb:.1f}MB, Cumulative: {checkpoint_cumulative_mb:.1f}MB")

                    prev_cell_id = cell_id

        # Get final stats (used as Flow_{N} in cross-run formula)
        final_stats = get_namespace_size(kernel_client)
        final_gpu_mb = get_kernel_gpu_memory_mb(kernel_client)

        # Get final checkpoint overhead (for last cell)
        if code_cells:
            last_cell_id = code_cells[-1].get("id", f"cell_{len(code_cells)-1}")
            final_overhead = get_checkpoint_overhead(kernel_client, last_cell_id)
            total_checkpoint_mb = final_overhead.get('total_mb', 0.0)
        else:
            total_checkpoint_mb = 0.0

        # Get final enforcer overhead
        final_breakdown = get_flowbook_overhead_breakdown(kernel_client)
        final_enforcer_mb = (final_breakdown.get("execution_records_mb", 0.0)
                             + final_breakdown.get("tracking_metadata_mb", 0.0)
                             + final_breakdown.get("other_mb", 0.0))

        namespace_mb = final_stats.get("total_mb", 0.0)
        if namespace_mb > 0:
            # Ratio = (namespace + checkpoint_overhead) / namespace
            memory_overhead_ratio = (namespace_mb + total_checkpoint_mb) / namespace_mb
        else:
            memory_overhead_ratio = 1.0

        results.totals = {
            "final_namespace_mb": namespace_mb,
            "final_gpu_mb": final_gpu_mb,
            "final_checkpoint_cumulative_mb": total_checkpoint_mb,
            "final_enforcer_overhead_mb": final_enforcer_mb,
            "max_namespace_mb": max_footprint_mb,
            "memory_overhead_ratio": memory_overhead_ratio,
        }

        log(f"FlowBook Memory: Final namespace {namespace_mb:.1f}MB, "
            f"Max {max_footprint_mb:.1f}MB, Checkpoints {total_checkpoint_mb:.1f}MB, "
            f"Enforcer {final_enforcer_mb:.3f}MB (ratio: {memory_overhead_ratio:.3f}x)" +
            (f", GPU {final_gpu_mb:.1f}MB" if final_gpu_mb > 0 else ""))

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return results


def run_flowbook_memory_v5(
    notebook_content: Dict[str, Any],
    cell_timeout: float,
    staleness_mode: str = "syntactic",
) -> V5MemoryResult:
    """
    Run notebook on FlowBook kernel and collect v5 simplified MEMORY metrics.

    This is a simplified version of run_flowbook_memory() that uses a single
    get_memory_snapshot() call per cell instead of multiple separate API calls.
    The result is a cleaner data flow that correctly handles CoW sharing.

    Args:
        notebook_content: Notebook JSON
        cell_timeout: Timeout per cell in seconds
        staleness_mode: Staleness computation mode ('syntactic' or 'semantic')

    Returns:
        V5MemoryResult with simplified cell memory data
    """
    cells = notebook_content.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cell_order = [c.get("id", f"cell_{i}") for i, c in enumerate(code_cells)]

    log(f"FlowBook Memory v5: Found {len(code_cells)} code cells")

    kernel_manager = None
    kernel_client = None
    result = V5MemoryResult()

    try:
        log("FlowBook Memory v5: Starting flowbook_kernel...")
        kernel_manager, kernel_client = create_flowbook_kernel()
        log("FlowBook Memory v5: Kernel ready")

        # Enable continue_after_violation
        kernel_client.execute("%continue_after_violation on", silent=True)
        _wait_for_idle(kernel_client)

        # Set staleness computation mode
        kernel_client.execute(f"%staleness_mode {staleness_mode}", silent=True)
        _wait_for_idle(kernel_client)
        log(f"FlowBook Memory v5: staleness_mode set to {staleness_mode}")

        # Run a warm-up cell
        log("FlowBook Memory v5: Running warm-up cell...")
        warmup_result = execute_cell_flowbook(
            kernel_client,
            "_warmup_var_ = 1; del _warmup_var_",
            "_warmup_",
            ["_warmup_"],
            timeout=60.0
        )
        log(f"FlowBook Memory v5: Warm-up complete")

        for idx, cell in enumerate(code_cells):
            cell_id = cell.get("id", f"cell_{idx}")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if not source.strip():
                continue

            log(f"FlowBook Memory v5: Executing cell {idx+1}/{len(code_cells)} ({cell_id})...")

            # Execute the cell
            timing = execute_cell_flowbook(
                kernel_client, source, cell_id, cell_order, cell_timeout
            )

            # Get unified memory snapshot AFTER cell execution
            snapshot = get_v5_memory_snapshot(kernel_client)

            # Get per-variable timing data (deepcopy_ms per variable)
            var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id) or {}
            checkpoint_var_timing = {}
            for var_name, cost_info in var_costs.items():
                if isinstance(cost_info, dict):
                    checkpoint_var_timing[var_name] = cost_info.get('deepcopy_ms', 0.0)

            # Convert bytes to MB
            user_ns_mb = snapshot['user_ns_bytes'] / (1024 * 1024)
            gpu_mb = snapshot['gpu_bytes'] / (1024 * 1024)
            checkpoint_mb = snapshot['checkpoint_bytes'] / (1024 * 1024)
            checkpoint_vars = {k: v / (1024 * 1024) for k, v in snapshot['checkpoint_vars'].items()}
            checkpoint_var_types = snapshot.get('checkpoint_var_types', {})
            gpu_checkpoint_mb = snapshot.get('gpu_checkpoint_bytes', 0) / (1024 * 1024)
            gpu_checkpoint_vars = {k: v / (1024 * 1024) for k, v in snapshot.get('gpu_checkpoint_vars', {}).items()}

            log(f"  NS: {user_ns_mb:.1f}MB, Ckpt: {checkpoint_mb:.1f}MB, GPU: {gpu_mb:.1f}MB")
            if checkpoint_vars:
                top_vars = sorted(checkpoint_vars.items(), key=lambda x: x[1], reverse=True)[:3]
                log(f"  Top checkpoint vars: {', '.join(f'{k}={v:.2f}MB' for k, v in top_vars)}")

            if timing.get("error") and timing.get("cell_runtime_ms") is None:
                log(f"  Error:\n{timing['error']}")
                result.cells.append(V5CellMemory(
                    cell_id=cell_id,
                    cell_index=idx,
                    user_ns_mb=0.0,
                    gpu_mb=gpu_mb,
                    checkpoint_mb=checkpoint_mb,
                    checkpoint_vars=checkpoint_vars,
                    checkpoint_var_timing=checkpoint_var_timing,
                    checkpoint_var_types=checkpoint_var_types,
                    gpu_checkpoint_mb=gpu_checkpoint_mb,
                    gpu_checkpoint_vars=gpu_checkpoint_vars,
                ))
            else:
                result.cells.append(V5CellMemory(
                    cell_id=cell_id,
                    cell_index=idx,
                    user_ns_mb=user_ns_mb,
                    gpu_mb=gpu_mb,
                    checkpoint_mb=checkpoint_mb,
                    checkpoint_vars=checkpoint_vars,
                    checkpoint_var_timing=checkpoint_var_timing,
                    checkpoint_var_types=checkpoint_var_types,
                    gpu_checkpoint_mb=gpu_checkpoint_mb,
                    gpu_checkpoint_vars=gpu_checkpoint_vars,
                ))

        # Log final summary
        if result.all_cells:
            final = result.all_cells[-1]
            peak_ckpt = result.peak_checkpoint_mb
            log(f"FlowBook Memory v5: Final NS={final.user_ns_mb:.1f}MB, "
                f"Peak Ckpt={peak_ckpt:.1f}MB, GPU={final.gpu_mb:.1f}MB")

    finally:
        cleanup_kernel(kernel_manager, kernel_client)

    return result


def measure_rerun_overhead(
    kernel_client: BlockingKernelClient,
    code_cells: List[dict],
    rerun_n: int,
    timeout: float = 30.0,
) -> RerunOverheadResult:
    """
    Measure rerun overhead at quartile-boundary cells.

    For each iteration 0..N-1, for each quartile cell, sends the
    %measure_rerun_overhead magic command and captures timing data.

    Args:
        kernel_client: Active FlowBook kernel client
        code_cells: List of code cells from notebook
        rerun_n: Number of measurement iterations
        timeout: Timeout per measurement

    Returns:
        RerunOverheadResult with all measurements
    """
    K = len(code_cells)

    # Calculate quartile indices
    if K == 0:
        quartile_indices = []
    elif K == 1:
        quartile_indices = [0]
    else:
        quartile_indices = sorted(set([
            0,
            (K - 1) // 4,
            (K - 1) // 2,
            3 * (K - 1) // 4,
            K - 1
        ]))

    result = RerunOverheadResult(
        rerun_n=rerun_n,
        quartile_indices=quartile_indices,
    )

    if not quartile_indices or rerun_n <= 0:
        return result

    total_measurements = rerun_n * len(quartile_indices)
    log(f"Rerun Overhead: {rerun_n} iterations x {len(quartile_indices)} quartile cells = {total_measurements} measurements")

    measurement_count = 0
    for iteration in range(rerun_n):
        log(f"Rerun Overhead: Iteration {iteration + 1}/{rerun_n}...")
        for idx in quartile_indices:
            cell = code_cells[idx]
            cell_id = cell.get("id", f"cell_{idx}")

            measurement_count += 1
            log(f"  Measuring cell {idx+1} ({cell_id}) [{measurement_count}/{total_measurements}]...")

            # Send the magic command
            magic_code = f"%measure_rerun_overhead {cell_id}"
            msg_id = kernel_client.execute(magic_code, silent=False)

            # Wait for result
            overhead_data = None
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    msg = kernel_client.get_iopub_msg(timeout=1.0)
                except Exception:
                    continue

                if msg['parent_header'].get('msg_id') != msg_id:
                    continue

                msg_type = msg['header']['msg_type']
                if msg_type == 'display_data':
                    metadata = msg['content'].get('metadata', {})
                    flowbook_meta = metadata.get('flowbook', {})
                    if 'rerun_overhead' in flowbook_meta:
                        overhead_data = flowbook_meta['rerun_overhead']
                        break
                elif msg_type == 'status':
                    if msg['content']['execution_state'] == 'idle':
                        break

            if overhead_data:
                result.measurements.append(RerunOverheadMeasurement(
                    iteration=iteration,
                    cell_id=overhead_data.get("cell_id", cell_id),
                    cell_index=idx,
                    checkpoint_ms=overhead_data.get("checkpoint_ms", 0.0),
                    check_ms=overhead_data.get("check_ms", 0.0),
                    total_overhead_ms=overhead_data.get("total_overhead_ms", 0.0),
                    checkpoint_by_var=overhead_data.get("checkpoint_by_var", {}),
                    checkpoint_var_costs=overhead_data.get("checkpoint_var_costs", {}),
                ))
                log(f"    Checkpoint: {overhead_data.get('checkpoint_ms', 0):.1f}ms, "
                    f"Check: {overhead_data.get('check_ms', 0):.1f}ms, "
                    f"Total: {overhead_data.get('total_overhead_ms', 0):.1f}ms")
            else:
                log(f"    Warning: No overhead data received for cell {cell_id}")
                result.measurements.append(RerunOverheadMeasurement(
                    iteration=iteration,
                    cell_id=cell_id,
                    cell_index=idx,
                    checkpoint_ms=0.0,
                    check_ms=0.0,
                    total_overhead_ms=0.0,
                ))

    return result


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
            "--skip-baseline",
            action="store_true",
            help="Skip baseline memory run (runs by default for cross-run comparison)",
        )
        subparser.add_argument(
            "--baseline-timing",
            action="store_true",
            help="Also run baseline timing phase (skipped by default)",
        )
        subparser.add_argument(
            "--rerun",
            type=int,
            default=0,
            help="Number of rerun overhead measurement iterations at quartile-boundary cells (default: 0)",
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
            default="syntactic",
            help="Staleness computation mode: 'syntactic' (set intersection, lower memory) or 'semantic' (checkpoint diff, precise). Default: syntactic",
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
        do_baseline_memory = False# not kwargs.get("skip_baseline", False)
        run_baseline_timing_flag = True# kwargs.get("baseline_timing", False)
        rerun_n = kwargs.get("rerun", 0)
        num_trials = kwargs.get("trials", 1)
        start_trial = kwargs.get("start", 1)
        staleness_mode = kwargs.get("staleness_mode", "syntactic")

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
            phases = ["FlowBook timing"]
            if run_baseline_timing_flag:
                phases.append("baseline timing")
            if do_baseline_memory:
                phases.append("baseline memory")
            if heapsizer_available:
                phases.append("FlowBook memory")
            log(f"Starting comparison: {', '.join(phases)}")
            log(f"Notebook: {notebook_path}")
            log(f"Cell timeout: {cell_timeout}s" if cell_timeout else "Cell timeout: none")
            log(f"HeapSizer available: {heapsizer_available}")
            log(f"Baseline memory: {do_baseline_memory}, Baseline timing: {run_baseline_timing_flag}")
            if rerun_n > 0:
                # Calculate quartile indices
                K = len(code_cells)
                if K <= 1:
                    quartile_indices = [0] if K == 1 else []
                else:
                    quartile_indices = sorted(set([
                        0,
                        (K - 1) // 4,
                        (K - 1) // 2,
                        3 * (K - 1) // 4,
                        K - 1
                    ]))
                log(f"Rerun overhead: {rerun_n} iterations at quartile cells {quartile_indices}")
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
                flowbook_timing, rerun_overhead = run_flowbook_timing(
                    notebook_content, cell_timeout, staleness_mode, rerun_n=rerun_n
                )
                log("")

                # ============================================================
                # PHASE 2: Baseline Timing (Scalene OFF) - if --baseline-timing
                # ============================================================
                baseline_timing = None
                if run_baseline_timing_flag:
                    log("=" * 60)
                    log("PHASE 2: BASELINE TIMING (Scalene OFF)")
                    log("=" * 60)
                    baseline_timing = run_baseline_timing(notebook_content, cell_timeout)
                    log("")
                else:
                    log("=" * 60)
                    log("PHASE 2: BASELINE TIMING - SKIPPED (use --baseline-timing to enable)")
                    log("=" * 60)
                    log("")

                # ============================================================
                # PHASE 3: Baseline Memory (HeapSizer) - if available and not --skip-baseline
                # ============================================================
                baseline_memory = None
                if heapsizer_available and do_baseline_memory:
                    log("=" * 60)
                    log("PHASE 3: BASELINE MEMORY (HeapSizer)")
                    log("=" * 60)
                    baseline_memory = run_baseline_memory(notebook_content, cell_timeout)
                    log("")
                elif heapsizer_available:
                    log("=" * 60)
                    log("PHASE 3: BASELINE MEMORY - SKIPPED (use default to enable)")
                    log("=" * 60)
                    log("")

                # ============================================================
                # PHASE 4: FlowBook Memory (HeapSizer) - if available
                # ============================================================
                flowbook_memory = None
                if heapsizer_available:
                    log("=" * 60)
                    log("PHASE 4: FLOWBOOK MEMORY (HeapSizer) - v5")
                    log("=" * 60)
                    flowbook_memory = run_flowbook_memory_v5(notebook_content, cell_timeout, staleness_mode)
                    log("")

                # Override gpu_checkpoint_mb using diff-based measurement:
                # GPU overhead = FlowBook timing GPU - Baseline timing GPU per cell.
                # This is more accurate than measuring cudf objects via
                # memory_usage(), which can misreport under managed memory.
                if (flowbook_memory and flowbook_memory.cells
                        and flowbook_timing and flowbook_timing.cells
                        and baseline_timing and baseline_timing.cells):
                    # Build baseline GPU lookup by cell index
                    baseline_gpu_by_idx = {
                        c.cell_index: c.gpu_mb
                        for c in baseline_timing.cells if not c.is_rerun
                    }
                    flowbook_gpu_by_idx = {
                        c.cell_index: c.gpu_mb
                        for c in flowbook_timing.cells if not c.is_rerun
                    }
                    for cell in flowbook_memory.cells:
                        fb_gpu = flowbook_gpu_by_idx.get(cell.cell_index, 0.0)
                        bl_gpu = baseline_gpu_by_idx.get(cell.cell_index, 0.0)
                        if fb_gpu > 0 and bl_gpu > 0:
                            # Diff-based GPU checkpoint overhead (floor at 0)
                            cell.gpu_checkpoint_mb = max(0.0, fb_gpu - bl_gpu)
                            # Per-variable breakdown not available from diff
                            cell.gpu_checkpoint_vars = {}

                # Build comparison result with new structure
                metadata_dict: Dict[str, Any] = {
                    "num_cells": len(code_cells),
                    "timeout_seconds": cell_timeout,
                    "staleness_mode": staleness_mode,
                }
                if rerun_n > 0:
                    metadata_dict["rerun_n"] = rerun_n
                if num_trials > 1:
                    metadata_dict["trial"] = trial_num
                    metadata_dict["num_trials"] = num_trials

                # Create v5.0 format comparison dict
                # Convert baseline_memory to v5 format if present
                baseline_memory_v5 = _convert_memory_results_to_v5(baseline_memory) if baseline_memory else None
                comparison_dict = create_v5_comparison_result(
                    notebook_path=str(notebook_path),
                    timestamp=datetime.now().isoformat(),
                    metadata_dict=metadata_dict,
                    baseline_timing=baseline_timing,
                    flowbook_timing=flowbook_timing,
                    baseline_memory=baseline_memory_v5,
                    flowbook_memory=flowbook_memory,
                )

                # Add rerun_overhead section if present
                if rerun_overhead:
                    comparison_dict["rerun_overhead"] = {
                        "rerun_n": rerun_overhead.rerun_n,
                        "quartile_indices": rerun_overhead.quartile_indices,
                        "measurements": [
                            {
                                "iteration": m.iteration,
                                "cell_id": m.cell_id,
                                "cell_index": m.cell_index,
                                "checkpoint_ms": m.checkpoint_ms,
                                "check_ms": m.check_ms,
                                "total_overhead_ms": m.total_overhead_ms,
                                "checkpoint_by_var": m.checkpoint_by_var,
                                "checkpoint_var_costs": m.checkpoint_var_costs,
                            }
                            for m in rerun_overhead.measurements
                        ],
                    }

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

                if heapsizer_available and flowbook_memory and flowbook_memory.cells:
                    # V5 format: get final cell's total_mb
                    final_cell = flowbook_memory.cells[-1]
                    flowbook_ns = final_cell.user_ns_mb
                    flowbook_ckpt = flowbook_memory.peak_checkpoint_mb
                    flowbook_total = final_cell.total_mb
                    log("MEMORY (from HeapSizer v5):")
                    if baseline_memory and baseline_memory.cells:
                        baseline_ns = baseline_memory.cells[-1].user_ns_mb
                        ns_overhead = flowbook_ns - baseline_ns
                        log(f"  Baseline namespace:   {baseline_ns:,.1f}MB")
                        log(f"  FlowBook namespace:   {flowbook_ns:,.1f}MB (overhead: {ns_overhead:+,.1f}MB)")
                        log(f"  FlowBook checkpoints: {flowbook_ckpt:,.1f}MB (peak)")
                        log(f"  FlowBook total:       {flowbook_total:,.1f}MB")
                    else:
                        log(f"  Baseline:             SKIPPED")
                        log(f"  FlowBook namespace:   {flowbook_ns:,.1f}MB")
                        log(f"  FlowBook checkpoints: {flowbook_ckpt:,.1f}MB (peak)")
                        log(f"  FlowBook total:       {flowbook_total:,.1f}MB")

                    # GPU overhead from timing phase diff
                    if flowbook_timing and baseline_timing:
                        fb_gpu_cells = [c for c in flowbook_timing.cells if not c.is_rerun]
                        bl_gpu_cells = [c for c in baseline_timing.cells if not c.is_rerun]
                        if fb_gpu_cells and bl_gpu_cells and fb_gpu_cells[-1].gpu_mb > 0:
                            fb_final_gpu = fb_gpu_cells[-1].gpu_mb
                            bl_final_gpu = bl_gpu_cells[-1].gpu_mb
                            gpu_overhead = max(0.0, fb_final_gpu - bl_final_gpu)
                            log(f"  GPU (FlowBook):       {fb_final_gpu:,.1f}MB")
                            log(f"  GPU (Baseline):       {bl_final_gpu:,.1f}MB")
                            log(f"  GPU overhead:         {gpu_overhead:,.1f}MB")
                    log("")

                if rerun_overhead and rerun_overhead.measurements:
                    total_measurements = len(rerun_overhead.measurements)
                    avg_checkpoint = sum(m.checkpoint_ms for m in rerun_overhead.measurements) / total_measurements
                    avg_check = sum(m.check_ms for m in rerun_overhead.measurements) / total_measurements
                    avg_total = sum(m.total_overhead_ms for m in rerun_overhead.measurements) / total_measurements

                    log(f"RERUN OVERHEAD ({rerun_overhead.rerun_n} iterations x {len(rerun_overhead.quartile_indices)} quartile cells = {total_measurements} measurements):")
                    log(f"  Average checkpoint:   {avg_checkpoint:,.1f}ms")
                    log(f"  Average check:        {avg_check:,.1f}ms")
                    log(f"  Average total:        {avg_total:,.1f}ms")
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
