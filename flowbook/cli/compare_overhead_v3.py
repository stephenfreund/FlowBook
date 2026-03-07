#!/usr/bin/env python3
"""
Process v3.0 FlowBook baseline comparison JSON files and generate statistics and plots.

v3.0 adds pre-cell memory measurements enabling cross-run memory overhead comparison:
- Base_i = baseline pre_namespace + pre_gpu (before step i)
- Flow_i = FlowBook pre_namespace + pre_gpu + pre_checkpoint + pre_enforcer (before step i)
- MemoryOverhead_i = Flow_i - Base_i
- Checkpoint_i = MemoryOverhead_{i+1} - MemoryOverhead_i

Dispatched from compare_overhead.main() when v3 data is detected.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flowbook.cli.compare_overhead import (
    # Data classes
    FileStats,
    AggregateStats,
    # Data loading and extraction
    load_comparison_json,
    is_v2_format,
    extract_warnings,
    extract_checkpoint_timing_var_data,
    extract_checkpoint_var_data,
    # Statistics
    compute_aggregate_stats,
    # Formatting
    format_table,
    format_json_output,
    format_csv,
)


# Minimum base memory (MB) below which we report ratio as 0.0
MIN_MEANINGFUL_BASE_MB = 1.0


def _get_cell_field(cell: Dict, field: str, fallback: float = 0.0) -> float:
    """Get a field from a memory cell dict, with fallback."""
    return cell.get(field, fallback)


def _get_checkpoint_mb(cell: Dict) -> float:
    """Get checkpoint cumulative MB with fallback to checkpoint_var_costs.

    The checkpoint_cumulative_mb field may be 0 even when checkpoint data exists
    in checkpoint_var_costs. This mirrors v2's get_checkpoint_mb() fallback logic.
    """
    # 1. Try explicit field if > 0
    checkpoint_cumulative = cell.get("checkpoint_cumulative_mb", 0)
    if checkpoint_cumulative > 0:
        return checkpoint_cumulative
    # 2. Try overhead_breakdown.checkpoints_mb
    overhead = cell.get("overhead_breakdown") or {}
    checkpoints_mb = overhead.get("checkpoints_mb")
    if checkpoints_mb is not None and checkpoints_mb > 0:
        return checkpoints_mb
    # 3. Fall back to summing cumulative_by_var (bytes -> MB)
    cumulative_by_var = cell.get("cumulative_by_var") or {}
    if cumulative_by_var:
        return sum(cumulative_by_var.values()) / (1024 * 1024)
    # 4. Fall back to summing cumulative_by_type (bytes -> MB)
    cumulative_by_type = cell.get("cumulative_by_type") or {}
    if cumulative_by_type:
        return sum(cumulative_by_type.values()) / (1024 * 1024)
    # 5. Fall back to summing checkpoint_var_costs (per-cell total)
    var_costs = cell.get("checkpoint_var_costs") or {}
    if var_costs:
        return sum(v.get("bytes", 0) for v in var_costs.values() if isinstance(v, dict)) / (1024 * 1024)
    return 0.0


def _get_pre_checkpoint_mb(cell: Dict) -> float:
    """Get pre-cell checkpoint cumulative MB with fallback.

    For cross-run comparison, we need pre_checkpoint_cumulative_mb. If that's 0,
    we fall back to the previous cell's checkpoint_var_costs total.
    Note: This is an approximation since we're using post-cell data as pre-cell estimate.
    """
    pre_ckpt = cell.get("pre_checkpoint_cumulative_mb", 0)
    if pre_ckpt > 0:
        return pre_ckpt
    # No direct pre_ field - will be handled at call site using previous cell's data
    return 0.0


def _compute_cross_run_overhead(
    baseline_mem_cells: List[Dict],
    flowbook_mem_cells: List[Dict],
    baseline_totals: Dict,
    flowbook_totals: Dict,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute cross-run memory overhead and per-cell checkpoint cost.

    Args:
        baseline_mem_cells: Baseline memory cells (must have pre_namespace_mb, pre_gpu_mb)
        flowbook_mem_cells: FlowBook memory cells (must have pre_namespace_mb, pre_gpu_mb,
                           pre_checkpoint_cumulative_mb, pre_enforcer_overhead_mb)
        baseline_totals: Baseline memory totals (final_namespace_mb, final_gpu_mb)
        flowbook_totals: FlowBook memory totals (final_namespace_mb, final_gpu_mb,
                        final_checkpoint_cumulative_mb, final_enforcer_overhead_mb)

    Returns:
        Tuple of (base_values, checkpoint_costs, ratios) where:
        - base_values[i] = Base_i (baseline memory before step i)
        - checkpoint_costs[i] = Checkpoint_i = MemoryOverhead_{i+1} - MemoryOverhead_i
        - ratios[i] = Checkpoint_i / Base_i (0 if Base_i < MIN_MEANINGFUL_BASE_MB)
    """
    num_cells = min(len(baseline_mem_cells), len(flowbook_mem_cells))

    # Check if pre_checkpoint_cumulative_mb is populated
    has_pre_checkpoint = any(
        _get_cell_field(fc, "pre_checkpoint_cumulative_mb") > 0
        for fc in flowbook_mem_cells[:num_cells]
    )

    # If pre_checkpoint_cumulative_mb is 0, derive from checkpoint_var_costs
    # pre_checkpoint for cell i ≈ checkpoint_var_costs total from cell i-1
    if not has_pre_checkpoint:
        ckpt_totals = []
        for fc in flowbook_mem_cells[:num_cells]:
            costs = fc.get("checkpoint_var_costs") or {}
            total_bytes = sum(v.get("bytes", 0) for v in costs.values() if isinstance(v, dict))
            ckpt_totals.append(total_bytes / (1024 * 1024))
        # Make cumulative
        for i in range(1, len(ckpt_totals)):
            ckpt_totals[i] = max(ckpt_totals[i], ckpt_totals[i - 1])
        # pre_checkpoint[i] = ckpt_totals[i-1] (0 for first cell)
        pre_checkpoint_derived = [0.0] + ckpt_totals[:-1] if ckpt_totals else []
    else:
        pre_checkpoint_derived = None

    # Build MemoryOverhead sequence: overhead[i] = Flow_i - Base_i
    overhead = []
    base_values = []
    for i in range(num_cells):
        bc = baseline_mem_cells[i]
        fc = flowbook_mem_cells[i]

        base_i = _get_cell_field(bc, "pre_namespace_mb") + _get_cell_field(bc, "pre_gpu_mb")

        # Use derived pre_checkpoint if pre_checkpoint_cumulative_mb is 0
        if pre_checkpoint_derived is not None:
            pre_ckpt = pre_checkpoint_derived[i] if i < len(pre_checkpoint_derived) else 0.0
        else:
            pre_ckpt = _get_cell_field(fc, "pre_checkpoint_cumulative_mb")

        flow_i = (_get_cell_field(fc, "pre_namespace_mb")
                  + _get_cell_field(fc, "pre_gpu_mb")
                  + pre_ckpt
                  + _get_cell_field(fc, "pre_enforcer_overhead_mb"))

        base_values.append(base_i)
        overhead.append(flow_i - base_i)

    # Final overhead (after last cell)
    base_final = baseline_totals.get("final_namespace_mb", 0) + baseline_totals.get("final_gpu_mb", 0)

    # For final checkpoint, use derived value or explicit field
    if pre_checkpoint_derived is not None and ckpt_totals:
        final_ckpt = ckpt_totals[-1]
    else:
        final_ckpt = flowbook_totals.get("final_checkpoint_cumulative_mb", 0)

    flow_final = (flowbook_totals.get("final_namespace_mb", 0)
                  + flowbook_totals.get("final_gpu_mb", 0)
                  + final_ckpt
                  + flowbook_totals.get("final_enforcer_overhead_mb", 0))
    overhead.append(flow_final - base_final)

    # Checkpoint_i = overhead[i+1] - overhead[i]
    checkpoint_costs = [overhead[i + 1] - overhead[i] for i in range(num_cells)]

    # Ratios
    ratios = []
    for i in range(num_cells):
        if base_values[i] >= MIN_MEANINGFUL_BASE_MB:
            ratios.append(checkpoint_costs[i] / base_values[i])
        else:
            ratios.append(0.0)

    return base_values, checkpoint_costs, ratios


def _compute_fallback_ratios(flowbook_mem_cells: List[Dict]) -> List[float]:
    """Compute checkpoint overhead ratios using FlowBook-only data (fallback).

    Uses checkpoint_delta_mb when non-zero. Otherwise derives from checkpoint_var_costs
    bytes (logical checkpoint size delta between consecutive cells).
    Denominator is prev_namespace_mb + prev_gpu_mb.
    """
    # Check if checkpoint_delta_mb has any non-zero values
    has_delta = any(_get_cell_field(c, "checkpoint_delta_mb") > 0 for c in flowbook_mem_cells)

    if has_delta:
        # Use checkpoint_delta_mb directly
        ratios = []
        for i, c in enumerate(flowbook_mem_cells):
            delta_mb = _get_cell_field(c, "checkpoint_delta_mb")
            if i == 0:
                base_mb = 0.0
            else:
                prev = flowbook_mem_cells[i - 1]
                base_mb = _get_cell_field(prev, "namespace_mb") + _get_cell_field(prev, "gpu_mb")
            if base_mb >= MIN_MEANINGFUL_BASE_MB:
                ratios.append(delta_mb / base_mb)
            else:
                ratios.append(0.0)
        return ratios

    # Fallback: derive from checkpoint_var_costs bytes
    # Compute logical checkpoint size at each cell, then delta between consecutive cells
    ckpt_sizes = []
    for c in flowbook_mem_cells:
        costs = c.get("checkpoint_var_costs") or {}
        total_bytes = sum(info.get("bytes", 0) for info in costs.values() if isinstance(info, dict))
        ckpt_sizes.append(total_bytes / (1024 * 1024))

    ratios = []
    for i in range(len(flowbook_mem_cells)):
        if i == 0:
            delta_mb = ckpt_sizes[0]
            base_mb = 0.0
        else:
            delta_mb = max(ckpt_sizes[i] - ckpt_sizes[i - 1], 0)
            prev = flowbook_mem_cells[i - 1]
            base_mb = _get_cell_field(prev, "namespace_mb") + _get_cell_field(prev, "gpu_mb")

        if base_mb >= MIN_MEANINGFUL_BASE_MB:
            ratios.append(delta_mb / base_mb)
        else:
            ratios.append(0.0)

    return ratios


def compute_file_stats_v3(data: Dict[str, Any], file_path: str) -> FileStats:
    """Compute statistics from a v3.0 comparison file.

    Timing stats are identical to v2. Memory stats use cross-run comparison
    when baseline memory is available, falling back to FlowBook-only.
    """
    notebook_path = data.get("notebook_path", file_path)
    notebook_name = Path(notebook_path).name

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    # --- Timing (identical to v2) ---
    baseline_timing = baseline.get("timing", {})
    flowbook_timing = flowbook.get("timing", {})
    baseline_totals = baseline_timing.get("totals", {}) if baseline_timing else {}
    flowbook_totals = flowbook_timing.get("totals", {}) if flowbook_timing else {}

    baseline_runtime = baseline_totals.get("execute_duration_ms", 0.0)
    flowbook_runtime = flowbook_totals.get("execute_duration_ms", 0.0)
    state_overhead = flowbook_totals.get("state_duration_ms", 0.0)
    check_overhead = flowbook_totals.get("check_duration_ms", 0.0)
    flowbook_total = flowbook_runtime

    # Checking summary
    checking_summary = flowbook_totals.get("checking_summary", {})
    checking_clean_cells = checking_summary.get("clean_cells", 0)
    checking_stale_cells = checking_summary.get("stale_cells", 0)
    checking_error_cells = checking_summary.get("error_cells", 0)
    checking_reason_counts = checking_summary.get("reason_counts", {})
    checking_error_counts = checking_summary.get("error_counts", {})

    if baseline_runtime > 0:
        slowdown = flowbook_total / baseline_runtime
        state_pct = (state_overhead / baseline_runtime) * 100
        check_pct = (check_overhead / baseline_runtime) * 100
    elif flowbook_runtime > 0:
        slowdown = (flowbook_runtime + state_overhead + check_overhead) / flowbook_runtime
        state_pct = (state_overhead / flowbook_runtime) * 100
        check_pct = (check_overhead / flowbook_runtime) * 100
    else:
        slowdown = 0.0
        state_pct = 0.0
        check_pct = 0.0

    # --- Memory (v3 cross-run) ---
    baseline_memory_data = baseline.get("memory", {})
    flowbook_memory_data = flowbook.get("memory", {})
    baseline_mem_cells = baseline_memory_data.get("cells", []) if baseline_memory_data else []
    flowbook_mem_cells = flowbook_memory_data.get("cells", []) if flowbook_memory_data else []
    has_baseline_memory = bool(baseline_mem_cells)

    baseline_mem_totals = baseline_memory_data.get("totals", {}) if baseline_memory_data else {}
    flowbook_mem_totals = flowbook_memory_data.get("totals", {}) if flowbook_memory_data else {}

    mb_to_bytes = 1024 * 1024

    if has_baseline_memory and flowbook_mem_cells:
        # Cross-run comparison for overall stats
        # Overall memory stats from cross-run final totals
        baseline_final_mb = (baseline_mem_totals.get("final_namespace_mb", 0)
                             + baseline_mem_totals.get("final_gpu_mb", 0))
        flowbook_final_mb = (flowbook_mem_totals.get("final_namespace_mb", 0)
                             + flowbook_mem_totals.get("final_gpu_mb", 0)
                             + flowbook_mem_totals.get("final_checkpoint_cumulative_mb", 0)
                             + flowbook_mem_totals.get("final_enforcer_overhead_mb", 0))
        overhead_mb = max(flowbook_final_mb - baseline_final_mb, 0)

        baseline_memory = int(baseline_final_mb * mb_to_bytes)
        memory_overhead = int(overhead_mb * mb_to_bytes)
        flowbook_memory = int(flowbook_final_mb * mb_to_bytes)

        if baseline_memory > 0:
            memory_overhead_ratio = flowbook_memory / baseline_memory
            memory_pct = (memory_overhead / baseline_memory) * 100
        else:
            memory_overhead_ratio = 1.0
            memory_pct = 0.0

        # Per-cell ratios: use same formula as v2 (checkpoint_delta / prev_namespace)
        # This ensures CDFs match between v2 and v3
        per_cell_memory_overhead = _compute_fallback_ratios(flowbook_mem_cells)

    elif flowbook_mem_cells:
        # FlowBook-only fallback
        per_cell_memory_overhead = _compute_fallback_ratios(flowbook_mem_cells)

        ns_mb = flowbook_mem_totals.get("final_namespace_mb", 0)
        ckpt_mb = flowbook_mem_totals.get("final_checkpoint_cumulative_mb", 0)

        # When checkpoint_cumulative_mb is 0 (CoW), derive from checkpoint_var_costs
        if ckpt_mb == 0 and flowbook_mem_cells:
            last_cell = flowbook_mem_cells[-1]
            costs = last_cell.get("checkpoint_var_costs") or {}
            ckpt_mb = sum(info.get("bytes", 0) for info in costs.values() if isinstance(info, dict)) / (1024 * 1024)

        baseline_memory = int(ns_mb * mb_to_bytes)
        memory_overhead = int(ckpt_mb * mb_to_bytes)
        flowbook_memory = baseline_memory + memory_overhead

        if baseline_memory > 0:
            memory_overhead_ratio = flowbook_memory / baseline_memory
            memory_pct = (memory_overhead / baseline_memory) * 100
        else:
            memory_overhead_ratio = 1.0
            memory_pct = 0.0
    else:
        per_cell_memory_overhead = []
        baseline_memory = 0
        flowbook_memory = 0
        memory_overhead = 0
        memory_pct = 0.0
        memory_overhead_ratio = 0.0

    num_cells = data.get("metadata", {}).get("num_cells", 0)
    if num_cells == 0 and flowbook_timing:
        num_cells = len(flowbook_timing.get("cells", []))

    num_trials = data.get("metadata", {}).get("averaged_trials", 1)

    # Rerun statistics
    rerun_baseline_cells = baseline_timing.get("rerun_cells", []) if baseline_timing else []
    rerun_flowbook_cells = flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    num_reruns = len(rerun_flowbook_cells)

    rerun_baseline_runtime = sum(c.get("execute_duration_ms", 0) for c in rerun_baseline_cells)
    rerun_flowbook_runtime = sum(c.get("execute_duration_ms", 0) for c in rerun_flowbook_cells)
    rerun_state_overhead = sum(c.get("state_duration_ms", 0) for c in rerun_flowbook_cells)
    rerun_check_overhead = sum(c.get("check_duration_ms", 0) for c in rerun_flowbook_cells)
    rerun_flowbook_total = rerun_flowbook_runtime

    # Final checkpoint from rerun memory data
    rerun_final_checkpoint = 0
    flowbook_mem_rerun = flowbook_memory_data.get("rerun_cells", []) if flowbook_memory_data else []
    if flowbook_mem_rerun:
        last = flowbook_mem_rerun[-1]
        rerun_final_checkpoint = int(_get_cell_field(last, "checkpoint_cumulative_mb") * mb_to_bytes)

    # Last cell overhead from timing
    baseline_timing_cells = baseline_timing.get("cells", []) if baseline_timing else []
    flowbook_timing_cells = flowbook_timing.get("cells", []) if flowbook_timing else []

    last_cell_state_pct = 0.0
    last_cell_check_pct = 0.0
    last_cell_memory_pct = 0.0

    if flowbook_timing_cells and baseline_timing_cells:
        last_fc = flowbook_timing_cells[-1]
        last_bc = baseline_timing_cells[-1]
        last_baseline_runtime = last_bc.get("execute_duration_ms", 0.0)
        if last_baseline_runtime > 0:
            last_cell_state_pct = (last_fc.get("state_duration_ms", 0.0) / last_baseline_runtime) * 100
            last_cell_check_pct = (last_fc.get("check_duration_ms", 0.0) / last_baseline_runtime) * 100

    if flowbook_mem_cells and baseline_mem_cells:
        last_baseline_mem = _get_cell_field(baseline_mem_cells[-1], "namespace_mb")
        last_flowbook_mem = _get_cell_field(flowbook_mem_cells[-1], "namespace_mb")
        if last_baseline_mem > 0:
            last_cell_memory_pct = ((last_flowbook_mem - last_baseline_mem) / last_baseline_mem) * 100

    # Per-cell timing overhead
    per_cell_checkpoint_overhead_ms: List[float] = []
    per_cell_total_overhead_ms: List[float] = []

    for fc in flowbook_timing_cells:
        state_ms = fc.get("state_duration_ms", 0.0)
        check_ms = fc.get("check_duration_ms", 0.0)
        execute_ms = fc.get("execute_duration_ms", 0.0)
        code_ms = fc.get("code_duration_ms")
        if code_ms is None:
            code_ms = max(execute_ms - state_ms - check_ms, 0)
        other_ms = max(execute_ms - (code_ms + state_ms + check_ms), 0)
        per_cell_checkpoint_overhead_ms.append(state_ms)
        per_cell_total_overhead_ms.append(state_ms + check_ms + other_ms)

    # Compute peak memory overhead percentage (max ratio * 100)
    peak_memory_overhead_pct = (
        max(per_cell_memory_overhead) * 100
        if per_cell_memory_overhead
        else 0.0
    )

    return FileStats(
        notebook_path=notebook_path,
        notebook_name=notebook_name,
        num_cells=num_cells,
        baseline_runtime_ms=baseline_runtime,
        flowbook_runtime_ms=flowbook_runtime,
        state_overhead_ms=state_overhead,
        check_overhead_ms=check_overhead,
        flowbook_total_ms=flowbook_total,
        slowdown=slowdown,
        state_overhead_pct=state_pct,
        check_overhead_pct=check_pct,
        baseline_memory_bytes=baseline_memory,
        flowbook_memory_bytes=flowbook_memory,
        memory_overhead_bytes=memory_overhead,
        memory_overhead_pct=memory_pct,
        memory_overhead_ratio=memory_overhead_ratio,
        last_cell_state_overhead_pct=last_cell_state_pct,
        last_cell_check_overhead_pct=last_cell_check_pct,
        last_cell_memory_overhead_pct=last_cell_memory_pct,
        peak_memory_overhead_pct=peak_memory_overhead_pct,
        num_reruns=num_reruns,
        rerun_baseline_runtime_ms=rerun_baseline_runtime,
        rerun_flowbook_runtime_ms=rerun_flowbook_runtime,
        rerun_state_overhead_ms=rerun_state_overhead,
        rerun_check_overhead_ms=rerun_check_overhead,
        rerun_flowbook_total_ms=rerun_flowbook_total,
        rerun_final_checkpoint_bytes=rerun_final_checkpoint,
        num_trials=num_trials,
        per_cell_checkpoint_overhead_ms=per_cell_checkpoint_overhead_ms,
        per_cell_total_overhead_ms=per_cell_total_overhead_ms,
        per_cell_memory_overhead_mb=per_cell_memory_overhead,
        checking_clean_cells=checking_clean_cells,
        checking_stale_cells=checking_stale_cells,
        checking_error_cells=checking_error_cells,
        checking_reason_counts=checking_reason_counts,
        checking_error_counts=checking_error_counts,
    )


def plot_combined_v3(
    data: Dict[str, Any],
    output_path: Optional[str] = None,
    large_fonts: bool = True,
    top_n: int = 10
) -> Optional[Any]:
    """
    Create combined 6-panel plot (2x3 grid) for v3.0 data.

    Panels 1, 2, 5 are identical to v2 (timing data unchanged).
    Panel 3 (Plot 4): Memory overhead using FlowBook run's own namespace as baseline layer.
        Note: namespace layer is from the FlowBook run, not the separate baseline run.
    Panel 4: Checkpoint memory by variable (same as v2).
    Panel 6 (Plot 6): Checkpoint_i / Base_i using cross-run data when baseline available.
        Falls back to FlowBook-only delta / (prev_ns + prev_gpu) when no baseline.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette()

    baseline = data["kernels"]["baseline"]
    flowbook = data["kernels"]["flowbook"]

    # Extract timing data (including reruns)
    baseline_timing = baseline.get("timing", {})
    flowbook_timing = flowbook.get("timing", {})
    baseline_initial_cells = baseline_timing.get("cells", []) if baseline_timing else []
    flowbook_initial_cells = flowbook_timing.get("cells", []) if flowbook_timing else []
    baseline_rerun_cells = baseline_timing.get("rerun_cells", []) if baseline_timing else []
    flowbook_rerun_cells = flowbook_timing.get("rerun_cells", []) if flowbook_timing else []
    baseline_cells = baseline_initial_cells + baseline_rerun_cells
    flowbook_cells = flowbook_initial_cells + flowbook_rerun_cells
    timing_initial_count = len(baseline_initial_cells)

    # Extract memory data (including reruns)
    baseline_memory = baseline.get("memory", {})
    flowbook_memory = flowbook.get("memory", {})
    baseline_mem_initial = baseline_memory.get("cells", []) if baseline_memory else []
    flowbook_mem_initial = flowbook_memory.get("cells", []) if flowbook_memory else []
    baseline_mem_rerun = baseline_memory.get("rerun_cells", []) if baseline_memory else []
    flowbook_mem_rerun = flowbook_memory.get("rerun_cells", []) if flowbook_memory else []
    baseline_mem_cells = baseline_mem_initial + baseline_mem_rerun
    flowbook_mem_cells = flowbook_mem_initial + flowbook_mem_rerun
    memory_initial_count = len(flowbook_mem_initial)
    has_baseline_memory = bool(baseline_mem_cells)

    # Font sizes
    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    has_memory = bool(flowbook_mem_cells)
    timing_var_data = extract_checkpoint_timing_var_data(data, top_n=top_n)
    var_data = extract_checkpoint_var_data(data, top_n=top_n)

    fig, axes_2d = plt.subplots(3, 2, figsize=(14, 18))
    axes = [
        axes_2d[0, 0], axes_2d[0, 1],
        axes_2d[1, 0], axes_2d[1, 1],
        axes_2d[2, 0], axes_2d[2, 1],
    ]

    # --- Prepare timing data (same as v2) ---
    cell_data_map = {}
    has_baseline = bool(baseline_cells)
    if baseline_cells and flowbook_cells:
        for c in baseline_cells:
            cell_data_map[c["cell_id"]] = {"baseline_ms": c.get("execute_duration_ms", 0)}
        for c in flowbook_cells:
            if c["cell_id"] in cell_data_map:
                execute_ms = c.get("execute_duration_ms", 0)
                state_ms = c.get("state_duration_ms", 0)
                check_ms = c.get("check_duration_ms", 0)
                code_ms = c.get("code_duration_ms")
                if code_ms is None:
                    code_ms = max(execute_ms - state_ms - check_ms, 0)
                cell_data_map[c["cell_id"]].update({
                    "execute_ms": execute_ms, "code_ms": code_ms,
                    "state_ms": state_ms, "check_ms": check_ms,
                })
    elif flowbook_cells:
        for c in flowbook_cells:
            execute_ms = c.get("execute_duration_ms", 0)
            state_ms = c.get("state_duration_ms", 0)
            check_ms = c.get("check_duration_ms", 0)
            code_ms = c.get("code_duration_ms")
            if code_ms is None:
                code_ms = max(execute_ms - state_ms - check_ms, 0)
            cell_data_map[c["cell_id"]] = {
                "baseline_ms": code_ms, "execute_ms": execute_ms,
                "code_ms": code_ms, "state_ms": state_ms, "check_ms": check_ms,
            }

    cell_ids = list(cell_data_map.keys())
    cells_arr = np.arange(1, len(cell_ids) + 1) if cell_ids else np.array([])

    baseline_arr = np.array([cell_data_map[cid].get("baseline_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    code_arr = np.array([cell_data_map[cid].get("code_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    state_arr = np.array([cell_data_map[cid].get("state_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    check_arr = np.array([cell_data_map[cid].get("check_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    execute_arr = np.array([cell_data_map[cid].get("execute_ms", 0) for cid in cell_ids]) if cell_ids else np.array([])
    other_arr = np.maximum(execute_arr - (code_arr + state_arr + check_arr), 0) if cell_ids else np.array([])

    baseline_cumsum = np.cumsum(baseline_arr) if len(baseline_arr) > 0 else np.array([])
    code_cumsum = np.cumsum(code_arr) if len(code_arr) > 0 else np.array([])
    state_cumsum = np.cumsum(state_arr) if len(state_arr) > 0 else np.array([])
    check_cumsum = np.cumsum(check_arr) if len(check_arr) > 0 else np.array([])
    other_cumsum = np.cumsum(other_arr) if len(other_arr) > 0 else np.array([])

    # ========== Panel 1: Timing Comparison (top-left) ==========
    ax = axes[0]
    if len(cells_arr) > 0:
        baseline_label = "Baseline" if has_baseline else "Code (no baseline)"
        ax.plot(cells_arr, baseline_cumsum / 1000, color=colors[0], linewidth=2, marker='o', markersize=4, label=baseline_label)
        ax.fill_between(cells_arr, 0, code_cumsum / 1000, alpha=0.3, color=colors[1], label="FlowBook Code")
        ax.fill_between(cells_arr, code_cumsum / 1000, (code_cumsum + state_cumsum) / 1000, alpha=0.4, color=colors[2], label="State")
        ax.fill_between(cells_arr, (code_cumsum + state_cumsum) / 1000, (code_cumsum + state_cumsum + check_cumsum) / 1000, alpha=0.4, color=colors[3], label="Check")
        ax.fill_between(cells_arr, (code_cumsum + state_cumsum + check_cumsum) / 1000, (code_cumsum + state_cumsum + check_cumsum + other_cumsum) / 1000, alpha=0.4, color=colors[4], label="Other")

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)
        title = "Timing Comparison" if has_baseline else "Timing (FlowBook only)"
        if timing_initial_count < len(cells_arr):
            title += f" (cells 1-{timing_initial_count} + {len(cells_arr) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)

        if timing_initial_count < len(cells_arr):
            ax.axvline(x=timing_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')
        ax.legend(loc="upper left", fontsize=legend_size)

        total_code_s = code_cumsum[-1] / 1000
        total_state_s = state_cumsum[-1] / 1000
        total_check_s = check_cumsum[-1] / 1000
        total_other_s = other_cumsum[-1] / 1000
        total_flowbook_s = total_code_s + total_state_s + total_check_s + total_other_s
        total_baseline_s = baseline_cumsum[-1] / 1000

        if has_baseline:
            textstr = f'Baseline: {total_baseline_s:.2f}s\nFlowBook: {total_flowbook_s:.2f}s'
        else:
            textstr = f'Code: {total_code_s:.2f}s\nTotal: {total_flowbook_s:.2f}s'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.70, textstr, transform=ax.transAxes, fontsize=legend_size,
                verticalalignment='top', horizontalalignment='left', bbox=props)

        total_overhead_s = total_state_s + total_check_s + total_other_s
        if total_code_s > 0:
            overhead_pct = (total_overhead_s / total_code_s) * 100
            ax.annotate(f'{overhead_pct:.1f}% overhead (vs code)',
                        xy=(cells_arr[-1], total_flowbook_s),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=legend_size, va='center', ha='left', color=colors[1])
    else:
        ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Timing Comparison", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 2: Checkpoint Time by Variable (top-right) ==========
    ax = axes[1]
    if timing_var_data is not None:
        var_colors = sns.color_palette("husl", len(timing_var_data["vars_ordered"]))
        timing_cells_x = np.array(timing_var_data["cells"])
        timing_var_types = timing_var_data.get("var_types", {})

        stacked = [np.array(timing_var_data["by_var"][v]) / 1000 for v in timing_var_data["vars_ordered"]]
        cumulative_timing = np.zeros(len(timing_cells_x))
        for i, (v, data_sec) in enumerate(zip(timing_var_data["vars_ordered"], stacked)):
            var_type = timing_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(timing_cells_x, cumulative_timing, cumulative_timing + data_sec, alpha=0.7, color=var_colors[i], label=label)
            cumulative_timing = cumulative_timing + data_sec

        ax.plot(timing_cells_x, cumulative_timing, color='black', linewidth=1.5, linestyle='--', label='Total')

        timing_var_initial_count = timing_var_data.get("initial_count", len(timing_cells_x))
        if timing_var_initial_count < len(timing_cells_x):
            ax.axvline(x=timing_var_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Checkpoint Time (seconds)", fontsize=label_size)
        title = "Checkpoint Time by Variable"
        if timing_var_initial_count < len(timing_cells_x):
            title += f" (cells 1-{timing_var_initial_count} + {len(timing_cells_x) - timing_var_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.legend(loc="upper left", fontsize=legend_size - 4, ncol=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No checkpoint timing data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Checkpoint Time by Variable", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 3: Memory Overhead (middle-left) — Plot 4 ==========
    # When baseline memory available: bottom = baseline namespace, overhead = flowbook - baseline.
    # When no baseline: use FlowBook namespace + checkpoint_cumulative_mb or checkpoint_var_costs.
    ax = axes[2]
    if has_memory:
        mem_cells_x = np.arange(1, len(flowbook_mem_cells) + 1)

        flowbook_namespace_mb = np.array([_get_cell_field(c, "namespace_mb") for c in flowbook_mem_cells])
        flowbook_gpu_mb = np.array([_get_cell_field(c, "gpu_mb") for c in flowbook_mem_cells])
        # Use helper with fallback to checkpoint_var_costs
        checkpoint_cumulative_mb = np.array([_get_checkpoint_mb(c) for c in flowbook_mem_cells])
        # Make cumulative (carry forward max) for semantic mode
        staleness_mode = data.get("metadata", {}).get("staleness_mode", "semantic")
        if staleness_mode == "semantic":
            checkpoint_cumulative_mb = np.maximum.accumulate(checkpoint_cumulative_mb)

        if has_baseline_memory and len(baseline_mem_cells) >= len(flowbook_mem_cells):
            # Cross-run: show baseline namespace as bottom, FlowBook overhead on top
            baseline_namespace_mb = np.array([_get_cell_field(c, "namespace_mb") for c in baseline_mem_cells[:len(flowbook_mem_cells)]])
            baseline_gpu_mb = np.array([_get_cell_field(c, "gpu_mb") for c in baseline_mem_cells[:len(flowbook_mem_cells)]])

            base_total = baseline_namespace_mb + baseline_gpu_mb
            flow_total = flowbook_namespace_mb + flowbook_gpu_mb + checkpoint_cumulative_mb

            overhead_mb = np.maximum(flow_total - base_total, 0)

            ax.fill_between(mem_cells_x, 0, base_total, alpha=0.4, color='gray', label='Baseline Namespace')
            ax.fill_between(mem_cells_x, base_total, base_total + overhead_mb, alpha=0.5, color='steelblue', label='FlowBook Overhead')
            ax.plot(mem_cells_x, base_total, color='gray', linewidth=2, linestyle='--')

            peak_overhead = np.max(overhead_mb)
            peak_idx = np.argmax(overhead_mb)
            if base_total[peak_idx] > 0:
                peak_overhead_pct = peak_overhead / base_total[peak_idx] * 100
                ax.annotate(f'{peak_overhead_pct:.1f}% peak overhead',
                            xy=(mem_cells_x[peak_idx], base_total[peak_idx] + overhead_mb[peak_idx]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=legend_size, va='bottom', ha='left', color=colors[1])

            ax.set_title('Memory Overhead (Cross-Run)', fontsize=title_size)
        else:
            # No baseline: use FlowBook namespace + checkpoint overhead
            has_gpu = np.any(flowbook_gpu_mb > 0)

            ax.fill_between(mem_cells_x, 0, flowbook_namespace_mb, alpha=0.4, color='gray', label='User Namespace')
            cumulative_mem = flowbook_namespace_mb.copy()

            if has_gpu:
                next_level = cumulative_mem + flowbook_gpu_mb
                ax.fill_between(mem_cells_x, cumulative_mem, next_level, alpha=0.4, color='orange', label='GPU Memory')
                cumulative_mem = next_level

            # Use checkpoint_cumulative_mb if non-zero, otherwise derive from checkpoint_var_costs
            if np.any(checkpoint_cumulative_mb > 0):
                ckpt_layer = checkpoint_cumulative_mb
            else:
                # Derive from checkpoint_var_costs bytes (logical checkpoint size)
                ckpt_layer = np.zeros(len(flowbook_mem_cells))
                for i, c in enumerate(flowbook_mem_cells):
                    costs = c.get("checkpoint_var_costs") or {}
                    total_bytes = sum(info.get("bytes", 0) for info in costs.values() if isinstance(info, dict))
                    ckpt_layer[i] = total_bytes / (1024 * 1024)
                # Make cumulative (carry forward max)
                ckpt_layer = np.maximum.accumulate(ckpt_layer)

            next_level = cumulative_mem + ckpt_layer
            ax.fill_between(mem_cells_x, cumulative_mem, next_level, alpha=0.5, color='steelblue', label='Checkpoints')
            cumulative_mem = next_level

            ax.plot(mem_cells_x, flowbook_namespace_mb, color='gray', linewidth=2, linestyle='--')

            base_mem = flowbook_namespace_mb + flowbook_gpu_mb
            peak_overhead = np.max(cumulative_mem - base_mem)
            peak_idx = np.argmax(cumulative_mem - base_mem)
            if base_mem[peak_idx] > 0:
                peak_overhead_pct = peak_overhead / base_mem[peak_idx] * 100
                ax.annotate(f'{peak_overhead_pct:.1f}% peak overhead',
                            xy=(mem_cells_x[peak_idx], cumulative_mem[peak_idx]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=legend_size, va='bottom', ha='left', color=colors[1])

            ax.set_title('Memory Overhead (FlowBook Only)', fontsize=title_size)

            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
            ax.text(0.98, 0.02, 'Namespace from FlowBook run', transform=ax.transAxes,
                    fontsize=tick_size - 2, va='bottom', ha='right', bbox=props, style='italic')

        ax.set_xlabel('Cell Number', fontsize=label_size)
        ax.set_ylabel('Memory (MB)', fontsize=label_size)

        if memory_initial_count < len(mem_cells_x):
            ax.axvline(x=memory_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.legend(loc='upper left', fontsize=legend_size - 2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 4: Checkpoint Memory by Variable (middle-right) ==========
    ax = axes[3]
    if var_data is not None:
        var_colors_mem = sns.color_palette("husl", len(var_data["vars_ordered"]))
        var_cells_x = np.array(var_data["cells"])
        mb = 1024 * 1024
        mem_var_types = var_data.get("var_types", {})

        # Namespace reference layer
        namespace_var = np.zeros(len(var_cells_x))
        gpu_mem_var = np.zeros(len(var_cells_x))
        if has_baseline_memory and len(baseline_mem_cells) >= len(var_cells_x):
            namespace_var = np.array([_get_cell_field(c, "namespace_mb") for c in baseline_mem_cells[:len(var_cells_x)]])
            gpu_mem_var = np.array([_get_cell_field(c, "gpu_mb") for c in baseline_mem_cells[:len(var_cells_x)]])
        elif has_memory and len(flowbook_mem_cells) >= len(var_cells_x):
            namespace_var = np.array([_get_cell_field(c, "namespace_mb") for c in flowbook_mem_cells[:len(var_cells_x)]])
            gpu_mem_var = np.array([_get_cell_field(c, "gpu_mb") for c in flowbook_mem_cells[:len(var_cells_x)]])

        has_gpu_var = any(g > 0 for g in gpu_mem_var)

        ax.fill_between(var_cells_x, 0, namespace_var, alpha=0.3, color='gray', label='User Namespace')
        cumulative_var = namespace_var.copy()

        if has_gpu_var:
            next_level = cumulative_var + gpu_mem_var
            ax.fill_between(var_cells_x, cumulative_var, next_level, alpha=0.4, color='orange', label='GPU Memory')
            cumulative_var = next_level

        stacked_mem = [np.array(var_data["by_var"][v]) / mb for v in var_data["vars_ordered"]]
        for i, (v, data_mb) in enumerate(zip(var_data["vars_ordered"], stacked_mem)):
            var_type = mem_var_types.get(v, "")
            label = f"{v} ({var_type})" if var_type else v
            ax.fill_between(var_cells_x, cumulative_var, cumulative_var + data_mb, alpha=0.7, color=var_colors_mem[i], label=label)
            cumulative_var = cumulative_var + data_mb

        ax.plot(var_cells_x, namespace_var, color='gray', linewidth=2, linestyle='--')

        var_initial_count = var_data.get("initial_count", len(var_cells_x))
        if var_initial_count < len(var_cells_x):
            ax.axvline(x=var_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Memory (MB)", fontsize=label_size)
        title = "Checkpoint Memory by Variable"
        if var_initial_count < len(var_cells_x):
            title += f" (cells 1-{var_initial_count} + {len(var_cells_x) - var_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)
        ax.legend(loc="upper left", fontsize=legend_size - 4, ncol=2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No checkpoint memory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Checkpoint Memory by Variable", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 5: Overhead Time per Cell (bottom-left) ==========
    ax = axes[4]
    if len(cells_arr) > 0:
        bar_width = 0.6
        ax.bar(cells_arr, state_arr / 1000, width=bar_width, alpha=0.7, color=colors[2], label='State')
        ax.bar(cells_arr, check_arr / 1000, width=bar_width, alpha=0.7, color=colors[3], label='Check', bottom=state_arr / 1000)
        ax.bar(cells_arr, other_arr / 1000, width=bar_width, alpha=0.7, color=colors[4], label='Other', bottom=(state_arr + check_arr) / 1000)

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Overhead per Cell (seconds)", fontsize=label_size)
        title = "Overhead Time per Cell"
        if timing_initial_count < len(cells_arr):
            title += f" (cells 1-{timing_initial_count} + {len(cells_arr) - timing_initial_count} reruns)"
        ax.set_title(title, fontsize=title_size)

        if timing_initial_count < len(cells_arr):
            ax.axvline(x=timing_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')
        ax.legend(loc="upper right", fontsize=legend_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5, right=len(cells_arr) + 0.5)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Overhead Time per Cell", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # ========== Panel 6: Checkpoint Overhead Ratio per Cell (bottom-right) — Plot 6 ==========
    # Always use v2-style ratio (checkpoint_delta / prev_namespace) for consistency with CDFs
    ax = axes[5]

    if flowbook_mem_cells:
        # Use same formula as v2: checkpoint_delta_mb / prev_namespace_mb
        ratios = _compute_fallback_ratios(flowbook_mem_cells)
        ratio_title = "Checkpoint Overhead Ratio"

        cell_nums = list(range(1, len(ratios) + 1))
        bar_width = 0.6
        ax.bar(cell_nums, ratios, width=bar_width, alpha=0.7, color='#66c2a5')

        ax.set_xlabel("Cell Number", fontsize=label_size)
        ax.set_ylabel("Checkpoint / Base Memory", fontsize=label_size)

        if memory_initial_count < len(flowbook_mem_cells):
            ratio_title += f" (cells 1-{memory_initial_count} + {len(flowbook_mem_cells) - memory_initial_count} reruns)"
            ax.axvline(x=memory_initial_count + 0.5, color='red', linestyle='--', linewidth=2, label='Rerun Start')
            ax.legend(loc="upper right", fontsize=legend_size)
        ax.set_title(ratio_title, fontsize=title_size)

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(left=0.5, right=len(cell_nums) + 0.5)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Checkpoint Overhead Ratio", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Figure title
    notebook_name = Path(data.get("notebook_path", "notebook")).stem
    fig.suptitle(notebook_name, fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Combined v3 plot saved to: {output_path}")
        return None
    else:
        return fig


def plot_overhead_cdfs_v3(
    aggregate: AggregateStats,
    output_path: Optional[str] = None,
    large_fonts: bool = True
) -> Optional[List[Any]]:
    """Create CDF plots for per-cell overhead distributions (v3 data).

    Creates three figures (matching v2):
    1. Two side-by-side CDFs: Time overhead (log scale) + Memory ratio
    2. Single CDF: Peak memory overhead % per notebook

    Memory overhead ratio uses checkpoint_delta / prev_namespace (same as v2).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    tick_size = 14 if large_fonts else 10
    annotation_size = 12 if large_fonts else 9

    total_overhead = np.array(aggregate.all_total_overhead_per_cell)
    memory_overhead = np.array(aggregate.all_memory_overhead_per_cell)

    # Prepare time data
    if len(total_overhead) > 0:
        total_data = np.sort(total_overhead)
        total_xlabel = "Total Overhead per Cell (ms)"
        total_cdf = np.arange(1, len(total_data) + 1) / len(total_data)
        total_stats = {
            'P50': np.percentile(total_data, 50),
            'P90': np.percentile(total_data, 90),
            'P95': np.percentile(total_data, 95),
            'P99': np.percentile(total_data, 99),
        }
    else:
        total_data = None

    # Prepare memory ratio data
    if len(memory_overhead) > 0:
        memory_data = np.sort(memory_overhead)
        memory_cdf = np.arange(1, len(memory_data) + 1) / len(memory_data)
        memory_stats = {
            'P50': np.percentile(memory_data, 50),
            'P90': np.percentile(memory_data, 90),
            'P95': np.percentile(memory_data, 95),
            'P99': np.percentile(memory_data, 99),
        }
    else:
        memory_data = None

    def format_ratio(r):
        return f'{r:.2f}'

    def add_percentile_markers(ax, data_arr, cdf_arr, stats, unit_fmt, color, legend_fontsize):
        percentiles = ['P50', 'P90', 'P95', 'P99']
        y_positions = [0.5, 0.9, 0.95, 0.99]
        label_offsets = [(5, 5), (5, -15), (5, 5), (5, -15)]
        label_vas = ['bottom', 'top', 'bottom', 'top']

        for pname, y_val, offset, va in zip(percentiles, y_positions, label_offsets, label_vas):
            if pname not in stats:
                continue
            x_val = stats[pname]
            ax.vlines(x_val, 0, y_val, color=color, linestyle='--', linewidth=1, alpha=0.4)
            ax.scatter([x_val], [y_val], color=color, s=30, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
            ax.annotate(pname, (x_val, y_val), textcoords='offset points',
                       xytext=offset, fontsize=annotation_size, ha='left', va=va, fontweight='bold')

        if len(data_arr) > 0:
            max_val = np.max(data_arr)
            ax.scatter([max_val], [1.0], color=color, s=30, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
            ax.annotate('Max', (max_val, 1.0), textcoords='offset points',
                       xytext=(5, -15), fontsize=annotation_size, ha='left', va='top', fontweight='bold')

        formatted_values = {pname: unit_fmt(stats[pname]) for pname in percentiles if pname in stats}
        if len(data_arr) > 0:
            formatted_values['Max'] = unit_fmt(np.max(data_arr))
        max_val_len = max(len(v) for v in formatted_values.values()) if formatted_values else 0
        legend_keys = [p for p in percentiles if p in formatted_values] + (['Max'] if 'Max' in formatted_values else [])
        legend_lines = [f'{pname}: {formatted_values[pname]:>{max_val_len}}' for pname in legend_keys]
        legend_text = '\n'.join(legend_lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=legend_fontsize,
                verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

    def add_percentile_gridlines(ax):
        for y in [0.5, 0.9, 0.95, 0.99]:
            ax.axhline(y, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    figures = []

    # --- Figure 1: Log scale ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes1[0]
    if total_data is not None:
        pos_mask = total_data > 0
        if np.any(pos_mask):
            ax.fill_between(total_data[pos_mask], 0, total_cdf[pos_mask], alpha=0.3, color='steelblue', edgecolor='none')
            ax.plot(total_data[pos_mask], total_cdf[pos_mask], color='steelblue', linewidth=2)
            add_percentile_markers(ax, total_data[pos_mask], total_cdf[pos_mask], total_stats, lambda x: f'{x:.1f}ms', 'black', tick_size)
            add_percentile_gridlines(ax)
            ax.set_xscale('log')
            ax.set_xlabel(total_xlabel, fontsize=label_size)
            ax.set_ylabel("Cumulative Probability", fontsize=label_size)
            ax.set_title("Total Overhead (Log Scale)", fontsize=title_size)
            ax.set_ylim(0, 1.05)
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, f'N={len(total_data)}', transform=ax.transAxes, fontsize=tick_size,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Total Overhead (Log Scale)", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    ax = axes1[1]
    if memory_data is not None:
        pos_mask = memory_data > 0
        if np.any(pos_mask):
            ax.fill_between(memory_data[pos_mask], 0, memory_cdf[pos_mask], alpha=0.3, color='seagreen', edgecolor='none')
            ax.plot(memory_data[pos_mask], memory_cdf[pos_mask], color='seagreen', linewidth=2)
            add_percentile_markers(ax, memory_data[pos_mask], memory_cdf[pos_mask], memory_stats, format_ratio, 'black', tick_size)
            add_percentile_gridlines(ax)
            ax.set_xlabel("Checkpoint / Base Memory Ratio", fontsize=label_size)
            ax.set_ylabel("Cumulative Probability", fontsize=label_size)
            ax.set_title("Memory Overhead Ratio", fontsize=title_size)
            ax.set_ylim(0, 1.05)
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, f'N={len(memory_data)}', transform=ax.transAxes, fontsize=tick_size,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead Ratio", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    fig1.suptitle("Per-Cell Overhead CDFs (Full Distribution)", fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig1)

    # --- Figure 2: Peak Memory Overhead % CDF (per notebook) ---
    peak_pcts = np.array(aggregate.all_peak_memory_overhead_pct)
    if len(peak_pcts) > 0:
        peak_data = np.sort(peak_pcts)
        peak_cdf = np.arange(1, len(peak_data) + 1) / len(peak_data)
        peak_stats = {
            'P50': np.percentile(peak_data, 50),
            'P90': np.percentile(peak_data, 90),
            'P95': np.percentile(peak_data, 95),
            'P99': np.percentile(peak_data, 99),
        }

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

        ax2.fill_between(peak_data, 0, peak_cdf, alpha=0.3, color='darkorange', edgecolor='none')
        ax2.plot(peak_data, peak_cdf, color='darkorange', linewidth=2)

        # Format percentage values for legend
        def format_pct(v):
            if v >= 100:
                return f'{v:.0f}%'
            elif v >= 10:
                return f'{v:.1f}%'
            elif v >= 1:
                return f'{v:.2f}%'
            else:
                return f'{v:.3f}%'

        add_percentile_markers(ax2, peak_data, peak_cdf, peak_stats, format_pct, 'darkorange', tick_size)
        add_percentile_gridlines(ax2)

        ax2.set_xlabel("Checkpoints / Namespace Size", fontsize=label_size)
        ax2.set_ylabel("Cumulative Probability", fontsize=label_size)
        ax2.set_title("Peak Memory Overhead Distribution", fontsize=title_size)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, 100)
        ax2.set_xticks([0, 25, 50, 75, 100])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax2.text(0.02, 0.98, f'N={len(peak_data)}', transform=ax2.transAxes, fontsize=tick_size,
                verticalalignment='top', horizontalalignment='left', bbox=props)
        ax2.tick_params(axis='both', labelsize=tick_size)

        plt.tight_layout()
        figures.append(fig2)

    if output_path is not None:
        import matplotlib.pyplot as plt
        plt.figure(fig1.number)
        plt.savefig(output_path, dpi=150)
        for f in figures:
            plt.close(f)
        print(f"CDF plot saved to: {output_path}")
        return None
    else:
        return figures


def plot_overhead_histograms_v3(
    aggregate: AggregateStats,
    output_path: Optional[str] = None,
    large_fonts: bool = True
) -> Optional[Any]:
    """Create histogram plots for per-cell overhead distributions (v3 data).

    Memory overhead uses cross-run Checkpoint_i / Base_i ratios.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    tick_size = 14 if large_fonts else 10

    total_overhead = np.array(aggregate.all_total_overhead_per_cell)
    memory_overhead = np.array(aggregate.all_memory_overhead_per_cell)

    def remove_outliers_iqr(data_arr: np.ndarray) -> np.ndarray:
        if len(data_arr) == 0:
            return data_arr
        q1 = np.percentile(data_arr, 25)
        q3 = np.percentile(data_arr, 75)
        iqr = q3 - q1
        return data_arr[(data_arr >= q1 - 1.5 * iqr) & (data_arr <= q3 + 1.5 * iqr)]

    total_filtered = remove_outliers_iqr(total_overhead)
    memory_filtered = remove_outliers_iqr(memory_overhead)

    fig, axes_hist = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram 1: Total Overhead
    ax = axes_hist[0]
    if len(total_filtered) > 0:
        if np.max(total_filtered) > 1000:
            data_plot = total_filtered / 1000
            xlabel = "Total Overhead per Cell (seconds)"
        else:
            data_plot = total_filtered
            xlabel = "Total Overhead per Cell (ms)"

        ax.hist(data_plot, bins=30, alpha=0.3, color='steelblue', edgecolor='black')
        ax.axvline(np.median(data_plot), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(data_plot):.2f}')
        ax.axvline(np.mean(data_plot), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(data_plot):.2f}')
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel("Frequency", fontsize=label_size)
        ax.set_title("Total Overhead per Cell Distribution", fontsize=title_size)
        ax.legend(fontsize=tick_size)

        n_total = len(total_overhead)
        n_filtered = len(total_filtered)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.95, f'N={n_filtered} (removed {n_total - n_filtered} outliers)',
                transform=ax.transAxes, fontsize=tick_size, va='top', ha='right', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Total Overhead per Cell Distribution", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    # Histogram 2: Memory Overhead Ratio
    ax = axes_hist[1]
    if len(memory_filtered) > 0:
        ax.hist(memory_filtered, bins=30, alpha=0.3, color='seagreen', edgecolor='black')
        ax.axvline(np.median(memory_filtered), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(memory_filtered):.2f}')
        ax.axvline(np.mean(memory_filtered), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(memory_filtered):.2f}')
        ax.set_xlabel("Checkpoint / Base Memory Ratio", fontsize=label_size)
        ax.set_ylabel("Frequency", fontsize=label_size)
        ax.set_title("Memory Overhead Ratio Distribution", fontsize=title_size)
        ax.legend(fontsize=tick_size)

        n_total = len(memory_overhead)
        n_filtered = len(memory_filtered)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.95, f'N={n_filtered} (removed {n_total - n_filtered} outliers)',
                transform=ax.transAxes, fontsize=tick_size, va='top', ha='right', bbox=props)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Memory Overhead Ratio Distribution", fontsize=title_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    fig.suptitle("Per-Cell Overhead Distributions (Outliers Removed)", fontsize=title_size + 2, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is not None:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Histogram plot saved to: {output_path}")
        return None
    else:
        return fig


def process_v3(file_data: Dict[str, Dict[str, Any]], args) -> None:
    """Main entry point for v3 data processing.

    Called from compare_overhead.main() when v3 format is detected.
    Recomputes stats using v3 cross-run methodology, then formats output
    and generates plots.
    """
    import matplotlib.pyplot as plt

    # Compute v3 stats for each file
    stats_list: List[FileStats] = []
    for file_path, data in file_data.items():
        try:
            stats = compute_file_stats_v3(data, file_path)
            stats_list.append(stats)
        except Exception as e:
            print(f"Warning: Error computing v3 stats for {file_path}: {e}", file=sys.stderr)
            continue

    if not stats_list:
        print("Error: No valid v3 comparison files found", file=sys.stderr)
        sys.exit(1)

    # Sort
    sort_key = {
        "slowdown": lambda s: s.slowdown,
        "memory": lambda s: s.memory_overhead_pct,
        "runtime": lambda s: s.baseline_runtime_ms,
        "name": lambda s: s.notebook_name,
    }[args.sort_by]
    stats_list.sort(key=sort_key, reverse=(args.sort_by != "name"))

    # Aggregate (reuse v2 aggregate function — same FileStats structure)
    aggregate = compute_aggregate_stats(stats_list)

    # Output statistics
    if args.format == "table":
        print(format_table(stats_list, aggregate))
    elif args.format == "json":
        print(format_json_output(stats_list, aggregate))
    elif args.format == "csv":
        print(format_csv(stats_list, aggregate))

    # Generate plots
    if args.plot:
        from matplotlib.backends.backend_pdf import PdfPages

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_figures = []

        for file_path, data in file_data.items():
            try:
                fig = plot_combined_v3(data, output_path=None, large_fonts=args.large_fonts, top_n=args.top_n)
                if fig is not None:
                    combined_figures.append(fig)
            except Exception as e:
                print(f"Warning: Could not generate v3 plot for {file_path}: {e}", file=sys.stderr)

        if combined_figures:
            combined_path = output_dir / args.output
            with PdfPages(str(combined_path)) as pdf:
                for fig in combined_figures:
                    pdf.savefig(fig, dpi=150)
                    plt.close(fig)

                # Aggregate histograms
                if aggregate.all_total_overhead_per_cell or aggregate.all_memory_overhead_per_cell:
                    hist_fig = plot_overhead_histograms_v3(aggregate, output_path=None, large_fonts=args.large_fonts)
                    if hist_fig is not None:
                        pdf.savefig(hist_fig, dpi=150)
                        plt.close(hist_fig)

                    # CDF plots
                    cdf_figs = plot_overhead_cdfs_v3(aggregate, output_path=None, large_fonts=args.large_fonts)
                    if cdf_figs is not None:
                        for cdf_fig in cdf_figs:
                            pdf.savefig(cdf_fig, dpi=150)
                            plt.close(cdf_fig)

            print(f"Combined v3 overhead plots saved to: {combined_path}")
