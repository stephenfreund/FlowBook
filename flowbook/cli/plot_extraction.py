"""Plot data extraction functions for compare_overhead.

This module contains functions to extract plot-ready data from ComparisonResult
objects. Each extract_plotN_data function returns a PlotNData dataclass that
contains all the data needed to render that plot.

Usage:
    from flowbook.cli.models import ComparisonResult
    from flowbook.cli.plot_extraction import extract_plot3_data, extract_plot6_data

    result = ComparisonResult.from_dict(json.load(f))
    p3 = extract_plot3_data(result)
    p6 = extract_plot6_data(result)
"""

from typing import Dict, List, Optional, Set
import numpy as np

from flowbook.cli.models import (
    ComparisonResult,
    FlowBookMemoryResult,
    BaselineMemoryResult,
    Plot1Data,
    Plot2Data,
    Plot3Data,
    Plot4Data,
    Plot5Data,
    Plot6Data,
    CDFData,
    # V5 models
    V5CellMemory,
    V5MemoryResult,
)
from flowbook.util.output import log


# Minimum thresholds to avoid division by zero or noise
MIN_RUN_SEC = 0.01
MIN_BASE_MB = 0.1  # 100KB - low enough for small notebooks, high enough to filter noise


def extract_gpu_overhead_from_timing(data: Dict) -> Optional[List[float]]:
    """Extract per-cell GPU checkpoint overhead from timing phase diff.

    Computes gpu_overhead[i] = max(0, flowbook_timing_gpu[i] - baseline_timing_gpu[i])
    using pynvml-measured GPU memory captured during timing phases.

    This is more accurate than measuring cudf objects via memory_usage()
    because it measures actual GPU memory impact (pynvml / nvidia-smi level).

    Args:
        data: Full comparison JSON dict

    Returns:
        List of per-cell GPU overhead in MB, or None if timing GPU data unavailable
    """
    kernels = data.get("kernels", {})
    fb_timing = kernels.get("flowbook", {}).get("timing", {})
    bl_timing = kernels.get("baseline", {}).get("timing", {})

    if not fb_timing or not bl_timing:
        return None

    fb_cells = [c for c in fb_timing.get("cells", []) if not c.get("is_rerun")]
    bl_cells = [c for c in bl_timing.get("cells", []) if not c.get("is_rerun")]

    if not fb_cells or not bl_cells:
        return None

    # Check that timing cells have gpu_mb data
    if not any(c.get("gpu_mb", 0) > 0 for c in fb_cells):
        return None

    # Build baseline GPU lookup by cell index
    bl_gpu = {c["cell_index"]: c.get("gpu_mb", 0.0) for c in bl_cells}

    result = []
    for c in fb_cells:
        fb_gpu = c.get("gpu_mb", 0.0)
        base_gpu = bl_gpu.get(c["cell_index"], 0.0)
        result.append(max(0.0, fb_gpu - base_gpu))

    return result


def extract_plot1_data(result: ComparisonResult) -> Optional[Plot1Data]:
    """Extract data for Plot 1: Execution Time per Cell.

    Shows stacked bar chart of run time + overhead components.

    Args:
        result: ComparisonResult with timing data

    Returns:
        Plot1Data or None if no timing data
    """
    timing = result.timing
    if not timing:
        return None

    # Get FlowBook timing cells
    fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
    cells_data = fb_timing.get("cells", [])
    rerun_data = fb_timing.get("rerun_cells", [])
    all_cells = cells_data + rerun_data

    if not all_cells:
        return None

    cells = []
    run_time = []
    state_time = []
    check_time = []
    other_time = []

    for cell in all_cells:
        idx = cell.get("cell_index", 0)
        cells.append(idx + 1)  # 1-indexed

        # Get timing values with fallbacks for different JSON format versions
        execute_ms = cell.get("execute_duration_ms", 0) or 0
        state_ms = cell.get("state_ms", 0) or cell.get("state_duration_ms", 0) or 0
        check_ms = cell.get("check_ms", 0) or cell.get("check_duration_ms", 0) or 0

        # Code time: try code_duration_ms first, then legacy field names
        code_ms = cell.get("code_duration_ms")
        if code_ms is None:
            code_ms = cell.get("run_ms", 0) or cell.get("cell_runtime_ms", 0) or 0
            # If still no code time but have execute, derive it
            if code_ms == 0 and execute_ms > 0:
                code_ms = max(execute_ms - state_ms - check_ms, 0)

        # Other overhead = execute - (code + state + check)
        other_ms = max(0, execute_ms - (code_ms + state_ms + check_ms))

        run_time.append(code_ms / 1000)
        state_time.append(state_ms / 1000)
        check_time.append(check_ms / 1000)
        other_time.append(other_ms / 1000)

    log(f"Plot 1: {len(cells)} cells, total run time {sum(run_time):.1f}s")

    return Plot1Data(
        cells=cells,
        run_time_sec=run_time,
        state_time_sec=state_time,
        check_time_sec=check_time,
        other_time_sec=other_time,
        initial_count=len(cells_data),
    )


def extract_plot2_data(result: ComparisonResult, top_n: int = 10) -> Optional[Plot2Data]:
    """Extract data for Plot 2: Checkpoint Time by Variable.

    Shows stacked area chart of deepcopy time per variable.

    Args:
        result: ComparisonResult with timing data
        top_n: Number of top variables to show (rest aggregated as "other")

    Returns:
        Plot2Data or None if no timing data
    """
    timing = result.timing
    if not timing:
        return None

    fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
    cells_data = fb_timing.get("cells", [])
    rerun_data = fb_timing.get("rerun_cells", [])
    all_cells = cells_data + rerun_data

    if not all_cells:
        return None

    # Collect all variable names
    all_vars: Set[str] = set()
    for cell in all_cells:
        var_timing = cell.get("checkpoint_var_timing", {}) or cell.get("checkpoint_var_costs", {})
        all_vars.update(var_timing.keys())

    if not all_vars:
        return None

    # Build per-variable time series
    cells = []
    var_series: Dict[str, List[float]] = {v: [] for v in all_vars}

    for cell in all_cells:
        idx = cell.get("cell_index", 0)
        cells.append(idx + 1)

        var_timing = cell.get("checkpoint_var_timing", {}) or cell.get("checkpoint_var_costs", {})
        for v in all_vars:
            # Time in seconds
            info = var_timing.get(v, {})
            if isinstance(info, dict):
                ms = info.get("deepcopy_ms", 0) or info.get("time_ms", 0) or 0
            else:
                ms = 0
            var_series[v].append(ms / 1000)

    # Order by total time
    var_totals = {v: sum(var_series[v]) for v in all_vars}
    vars_ordered = sorted(var_totals.keys(), key=lambda v: var_totals[v], reverse=True)

    # Aggregate "other" if needed
    if len(vars_ordered) > top_n:
        top_vars = vars_ordered[:top_n]
        other_vars = vars_ordered[top_n:]

        other_series = [0.0] * len(cells)
        for v in other_vars:
            for i, val in enumerate(var_series[v]):
                other_series[i] += val
            del var_series[v]

        var_series["other"] = other_series
        vars_ordered = top_vars + ["other"]

    log(f"Plot 2: {len(vars_ordered)} vars, top: {vars_ordered[:3]}")

    return Plot2Data(
        cells=cells,
        var_series={v: var_series[v] for v in vars_ordered},
        vars_ordered=vars_ordered,
        initial_count=len(cells_data),
    )


def extract_plot3_data(result: ComparisonResult) -> Optional[Plot3Data]:
    """Extract data for Plot 3: Memory Overhead.

    Shows stacked area chart with three layers:
    - user_ns_mb: User namespace memory
    - gpu_mb: GPU memory
    - overhead_mb: FlowBook checkpoint overhead

    When baseline available:
        overhead = flowbook.total - baseline.total

    When no baseline:
        overhead = flowbook.overhead_mb

    Args:
        result: ComparisonResult with memory data

    Returns:
        Plot3Data or None if no memory data
    """
    fb = result.flowbook
    bl = result.baseline

    if not fb or not fb.cells:
        return None

    has_baseline = bl is not None and len(bl.cells) > 0

    cells = []
    user_ns_mb = []
    gpu_mb = []
    overhead_mb = []

    fb_all = fb.all_cells
    bl_all = bl.all_cells if bl else []

    for i, fb_cell in enumerate(fb_all):
        cells.append(fb_cell.cell_index + 1)

        if has_baseline and i < len(bl_all):
            # Cross-run comparison
            bl_total = bl_all[i].post.total_mb
            flow_total = fb_cell.post.total_mb
            overhead = max(0, flow_total - bl_total)
        else:
            # FlowBook only
            overhead = fb_cell.post.overhead_mb

        user_ns_mb.append(fb_cell.post.user_ns_mb)
        gpu_mb.append(fb_cell.post.gpu_mb)
        overhead_mb.append(overhead)

    # Find peak (peak overhead relative to base = user_ns + gpu)
    if overhead_mb:
        peak_idx = overhead_mb.index(max(overhead_mb))
        peak_overhead = overhead_mb[peak_idx]
        base_at_peak = user_ns_mb[peak_idx] + gpu_mb[peak_idx]
        peak_pct = 100 * peak_overhead / base_at_peak if base_at_peak > 0 else 0
    else:
        peak_idx = 0
        peak_overhead = 0
        peak_pct = 0

    log(f"Plot 3: {len(cells)} cells, peak overhead {peak_pct:.1f}% at cell {peak_idx + 1}")

    return Plot3Data(
        cells=cells,
        user_ns_mb=user_ns_mb,
        gpu_mb=gpu_mb,
        overhead_mb=overhead_mb,
        has_baseline=has_baseline,
        peak_overhead_mb=peak_overhead,
        peak_overhead_pct=peak_pct,
        peak_cell=peak_idx,
        initial_count=len(fb.cells),
    )


def extract_plot4_data(result: ComparisonResult, top_n: int = 10) -> Optional[Plot4Data]:
    """Extract data for Plot 4: Checkpoint Memory by Variable.

    Shows stacked area chart:
    - Bottom: user namespace (gray)
    - Middle: GPU memory (orange)
    - Top: per-variable checkpoint sizes (colors)

    Variable sizes are summed across all checkpoints at each cell.

    Args:
        result: ComparisonResult with memory data
        top_n: Number of top variables to show

    Returns:
        Plot4Data or None if no memory data
    """
    fb = result.flowbook
    if not fb or not fb.cells:
        return None

    fb_all = fb.all_cells

    cells = []
    namespace_mb = []
    gpu_mb = []

    # Collect all variable names
    all_vars: Set[str] = set()
    for cell in fb_all:
        all_vars.update(cell.post.var_totals().keys())

    if not all_vars:
        log("Plot 4: No checkpoint variables found")
        return None

    # Build per-variable series
    var_series: Dict[str, List[float]] = {v: [] for v in all_vars}
    var_types: Dict[str, str] = {}

    for cell in fb_all:
        cells.append(cell.cell_index + 1)
        namespace_mb.append(cell.post.user_ns_mb)
        gpu_mb.append(cell.post.gpu_mb)

        totals = cell.post.var_totals()
        for v in all_vars:
            var_series[v].append(totals.get(v, 0.0))

        # Capture var types
        types = cell.post.var_types()
        for v, t in types.items():
            if v not in var_types:
                var_types[v] = t

    # Order by max size
    var_maxes = {v: max(var_series[v]) for v in all_vars}
    vars_ordered = sorted(var_maxes.keys(), key=lambda v: var_maxes[v], reverse=True)

    # Aggregate "other" if needed
    if len(vars_ordered) > top_n:
        top_vars = vars_ordered[:top_n]
        other_vars = vars_ordered[top_n:]

        other_series = [0.0] * len(cells)
        for v in other_vars:
            for i, val in enumerate(var_series[v]):
                other_series[i] += val
            del var_series[v]

        var_series["other"] = other_series
        vars_ordered = top_vars + ["other"]

    log(f"Plot 4: {len(vars_ordered)} vars, top: {vars_ordered[:3]}")

    return Plot4Data(
        cells=cells,
        namespace_mb=namespace_mb,
        gpu_mb=gpu_mb,
        var_series={v: var_series[v] for v in vars_ordered},
        vars_ordered=vars_ordered,
        var_types=var_types,
        initial_count=len(fb.cells),
    )


def extract_plot5_data(result: ComparisonResult) -> Optional[Plot5Data]:
    """Extract data for Plot 5: Overhead Time per Cell.

    Shows stacked bar chart of state/check/other overhead.

    Args:
        result: ComparisonResult with timing data

    Returns:
        Plot5Data or None if no timing data
    """
    timing = result.timing
    if not timing:
        return None

    fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
    cells_data = fb_timing.get("cells", [])
    rerun_data = fb_timing.get("rerun_cells", [])
    all_cells = cells_data + rerun_data

    if not all_cells:
        return None

    cells = []
    state_sec = []
    check_sec = []
    other_sec = []

    for cell in all_cells:
        idx = cell.get("cell_index", 0)
        cells.append(idx + 1)

        # Get timing values with fallbacks for different JSON format versions
        execute_ms = cell.get("execute_duration_ms", 0) or 0
        state_ms = cell.get("state_ms", 0) or cell.get("state_duration_ms", 0) or 0
        check_ms = cell.get("check_ms", 0) or cell.get("check_duration_ms", 0) or 0

        # Code time for computing "other" overhead
        code_ms = cell.get("code_duration_ms")
        if code_ms is None:
            code_ms = cell.get("run_ms", 0) or cell.get("cell_runtime_ms", 0) or 0
            if code_ms == 0 and execute_ms > 0:
                code_ms = max(execute_ms - state_ms - check_ms, 0)

        # Other overhead = execute - (code + state + check)
        other_ms = max(0, execute_ms - (code_ms + state_ms + check_ms))

        state_sec.append(state_ms / 1000)
        check_sec.append(check_ms / 1000)
        other_sec.append(other_ms / 1000)

    log(f"Plot 5: {len(cells)} cells, total overhead {sum(state_sec) + sum(check_sec) + sum(other_sec):.2f}s")

    return Plot5Data(
        cells=cells,
        state_sec=state_sec,
        check_sec=check_sec,
        other_sec=other_sec,
        initial_count=len(cells_data),
    )


def extract_plot6_data(result: ComparisonResult) -> Optional[Plot6Data]:
    """Extract data for Plot 6: Checkpoint Overhead Ratio per Cell.

    For each cell, compute:
        ratio = checkpoint_delta_mb / prev_cell_base_mb

    This shows how much new checkpoint data was created relative to the
    base memory (namespace + GPU) at the start of that cell.

    Args:
        result: ComparisonResult with memory data

    Returns:
        Plot6Data or None if no memory data
    """
    fb = result.flowbook
    if not fb or not fb.cells:
        return None

    fb_all = fb.all_cells
    cells = []
    ratios = []

    for i, cell in enumerate(fb_all):
        cells.append(cell.cell_index + 1)

        # Compute checkpoint delta (new checkpoint data at this cell)
        if i == 0:
            delta_mb = cell.post.total_checkpoint_mb
            base_mb = 0.0  # No prior namespace for first cell
        else:
            prev_cell = fb_all[i - 1]
            delta_mb = max(0, cell.post.total_checkpoint_mb - prev_cell.post.total_checkpoint_mb)
            base_mb = prev_cell.post.user_ns_mb + prev_cell.post.gpu_mb

        # Compute ratio (0 if base is too small for meaningful ratio)
        if base_mb >= MIN_BASE_MB:
            ratio = delta_mb / base_mb
        else:
            ratio = 0.0
        ratios.append(ratio)

    if not cells:
        log("Plot 6: No cells found")
        return None

    nonzero_count = sum(1 for r in ratios if r > 0)
    log(f"Plot 6: {len(cells)} cells, {nonzero_count} with nonzero ratios")

    return Plot6Data(
        cells=cells,
        ratios=ratios,
        initial_count=len(fb.cells),
    )


def extract_cdf_data(
    results: List[ComparisonResult],
    raw_data: Optional[List[Dict]] = None,
) -> Optional[CDFData]:
    """Extract data for aggregate CDF plots across multiple notebooks.

    Builds CDFs for:
    - Time overhead ratio (overhead_sec / run_sec)
    - Memory overhead ratio (overhead_mb / base_mb)
    - Peak memory overhead % per notebook

    Args:
        results: List of ComparisonResults from multiple notebooks
        raw_data: Optional list of raw JSON dicts (for v5 extraction)

    Returns:
        CDFData or None if no data
    """
    time_overhead_ms = []
    memory_ratios = []
    memory_abs_mb = []
    peak_memory_pct = []

    for i, result in enumerate(results):
        # Time overhead from timing data (raw ms values)
        timing = result.timing
        if timing:
            fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
            for cell in fb_timing.get("cells", []):
                # Get timing values with fallbacks for different JSON format versions
                state_ms = cell.get("state_ms", 0) or cell.get("state_duration_ms", 0) or 0
                check_ms = cell.get("check_ms", 0) or cell.get("check_duration_ms", 0) or 0

                overhead_ms = state_ms + check_ms
                if overhead_ms > 0:
                    time_overhead_ms.append(overhead_ms)

        # Memory ratios - use v5 extraction if raw data provided
        p3 = None
        if raw_data and i < len(raw_data):
            data = raw_data[i]
            version = str(data.get("version", "4.0"))
            if version.startswith("5"):
                v5_memory = extract_v5_memory_result(data)
                if v5_memory and v5_memory.all_cells:
                    gpu_from_timing = extract_gpu_overhead_from_timing(data)
                    p3 = extract_plot3_data_v5(
                        v5_memory.all_cells,
                        gpu_overhead_from_timing=gpu_from_timing,
                    )

        # Fall back to v4 extraction
        if p3 is None:
            p3 = extract_plot3_data(result)

        if p3:
            # Use same delta-based ratio as Plot 6: checkpoint_delta / prev_base
            # This measures incremental overhead per cell, not cumulative
            for j in range(len(p3.overhead_mb)):
                if j == 0:
                    # First cell: delta = overhead, prev_base = 0 (skip)
                    continue
                prev_base = p3.base_mb[j - 1]
                if prev_base >= MIN_BASE_MB:
                    delta = max(0, p3.overhead_mb[j] - p3.overhead_mb[j - 1])
                    memory_ratios.append(delta / prev_base)
                    # Absolute per-cell memory overhead (Checkpoint - Base, in MB).
                    # overhead_mb is flowbook.total - baseline.total, i.e. the
                    # checkpoint memory present at this cell. Gated by the same
                    # condition as memory_ratios so both CDFs share one cell
                    # population (identical N).
                    memory_abs_mb.append(p3.overhead_mb[j])

            peak_memory_pct.append(p3.peak_overhead_pct)

    # GPU checkpoint memory ratios and peak
    gpu_memory_ratios = []
    gpu_peak_memory_pct = []
    for i, result in enumerate(results):
        p3 = None
        if raw_data and i < len(raw_data):
            data = raw_data[i]
            version = str(data.get("version", "4.0"))
            if version.startswith("5"):
                v5_memory = extract_v5_memory_result(data)
                if v5_memory and v5_memory.all_cells:
                    gpu_from_timing = extract_gpu_overhead_from_timing(data)
                    p3 = extract_plot3_data_v5(
                        v5_memory.all_cells,
                        gpu_overhead_from_timing=gpu_from_timing,
                    )

        if p3 and p3.gpu_checkpoint_mb:
            # Per-cell GPU checkpoint delta / prev_gpu ratio
            # Use GPU-only base (not CPU+GPU) for GPU-specific overhead
            for j in range(1, len(p3.gpu_checkpoint_mb)):
                prev_gpu_base = p3.gpu_mb[j - 1]
                if prev_gpu_base >= MIN_BASE_MB:
                    delta = max(0, p3.gpu_checkpoint_mb[j] - p3.gpu_checkpoint_mb[j - 1])
                    gpu_memory_ratios.append(delta / prev_gpu_base)

            # Peak GPU checkpoint overhead %
            # Use GPU-only base (not CPU+GPU) for GPU-specific overhead
            gpu_base_totals = p3.gpu_mb
            gpu_totals = [g + gc for g, gc in zip(gpu_base_totals, p3.gpu_checkpoint_mb)]
            peak_gpu_base = max(gpu_base_totals) if gpu_base_totals else 0
            peak_gpu_total = max(gpu_totals) if gpu_totals else 0
            if peak_gpu_base > 0:
                gpu_peak_memory_pct.append((peak_gpu_total / peak_gpu_base - 1) * 100)

    # Per-cell overhead percentage: (state + check) / base * 100
    overhead_pct = []
    for result in results:
        timing = result.timing
        if timing:
            fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
            for cell in fb_timing.get("cells", []):
                # Get timing values
                state_ms = cell.get("state_ms", 0) or cell.get("state_duration_ms", 0) or 0
                check_ms = cell.get("check_ms", 0) or cell.get("check_duration_ms", 0) or 0

                # Get base (code execution time)
                code_ms = cell.get("code_duration_ms")
                if code_ms is None:
                    code_ms = cell.get("run_ms", 0) or cell.get("cell_runtime_ms", 0) or 0

                # Compute overhead percentage: (state + check) / (base + 150) * 100
                # Add 150ms to base to reflect Jupyter frontend overhead
                if code_ms > 0:
                    pct = (state_ms + check_ms) / (code_ms + 150) * 100
                    overhead_pct.append(pct)

    # Base runtime CDF (code execution time per cell)
    base_runtime_ms = []
    for result in results:
        timing = result.timing
        if timing:
            fb_timing = timing.get("kernels", {}).get("flowbook", {}).get("timing", {})
            for cell in fb_timing.get("cells", []):
                code_ms = cell.get("code_duration_ms")
                if code_ms is None:
                    code_ms = cell.get("run_ms", 0) or cell.get("cell_runtime_ms", 0) or 0
                if code_ms > 0:
                    base_runtime_ms.append(code_ms)

    if not time_overhead_ms and not memory_ratios and not overhead_pct and not base_runtime_ms:
        return None

    def build_cdf(values: List[float]):
        if not values:
            return [], []
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        pcts = [(i + 1) / n for i in range(n)]
        return sorted_vals, pcts

    time_sorted, time_pct = build_cdf(time_overhead_ms)
    memory_sorted, memory_pct = build_cdf(memory_ratios)
    memory_abs_sorted, memory_abs_pct = build_cdf(memory_abs_mb)
    peak_sorted, peak_pct = build_cdf(peak_memory_pct)
    gpu_memory_sorted, gpu_memory_pct = build_cdf(gpu_memory_ratios)
    gpu_peak_sorted, gpu_peak_pct = build_cdf(gpu_peak_memory_pct)
    overhead_pct_sorted, overhead_pct_pct = build_cdf(overhead_pct)
    base_runtime_sorted, base_runtime_pct = build_cdf(base_runtime_ms)

    log(f"CDF: {len(time_overhead_ms)} time samples, {len(memory_ratios)} memory ratios, {len(overhead_pct)} overhead pct, {len(base_runtime_ms)} base runtimes")

    return CDFData(
        time_overhead_ms=time_overhead_ms,
        time_sorted=time_sorted,
        time_percentiles=time_pct,
        memory_ratios=memory_ratios,
        memory_sorted=memory_sorted,
        memory_percentiles=memory_pct,
        peak_memory_pct=peak_memory_pct,
        peak_sorted=peak_sorted,
        peak_percentiles=peak_pct,
        gpu_memory_ratios=gpu_memory_ratios,
        gpu_memory_sorted=gpu_memory_sorted,
        gpu_memory_percentiles=gpu_memory_pct,
        gpu_peak_memory_pct=gpu_peak_memory_pct,
        gpu_peak_sorted=gpu_peak_sorted,
        gpu_peak_percentiles=gpu_peak_pct,
        overhead_pct=overhead_pct,
        overhead_pct_sorted=overhead_pct_sorted,
        overhead_pct_percentiles=overhead_pct_pct,
        base_runtime_ms=base_runtime_ms,
        base_runtime_sorted=base_runtime_sorted,
        base_runtime_percentiles=base_runtime_pct,
        memory_abs_mb=memory_abs_mb,
        memory_abs_sorted=memory_abs_sorted,
        memory_abs_percentiles=memory_abs_pct,
    )


# ============ V5 Extraction Functions ============


def extract_plot2_data_v5(
    cells: List[V5CellMemory],
    top_n: int = 10,
) -> Optional[Plot2Data]:
    """Extract data for Plot 2: Checkpoint Time by Variable from v5 cells.

    Shows stacked area chart of deepcopy time per variable.

    Args:
        cells: List of V5CellMemory objects with checkpoint_var_timing
        top_n: Number of top variables to show (rest aggregated as "other")

    Returns:
        Plot2Data or None if no timing data
    """
    if not cells:
        return None

    # Collect all variable names with timing
    all_vars: Set[str] = set()
    for cell in cells:
        all_vars.update(cell.checkpoint_var_timing.keys())

    if not all_vars:
        return None

    # Build per-variable time series (convert ms to seconds for plotting)
    cell_nums: List[int] = []
    var_series: Dict[str, List[float]] = {v: [] for v in all_vars}
    var_types: Dict[str, str] = {}

    for cell in cells:
        cell_nums.append(cell.cell_index + 1)
        for v in all_vars:
            ms = cell.checkpoint_var_timing.get(v, 0.0)
            var_series[v].append(ms / 1000)  # Convert to seconds

        # Aggregate var types from all cells (first type found wins)
        for v, t in cell.checkpoint_var_types.items():
            if v not in var_types and t:
                var_types[v] = t

    # Order by total time
    var_totals = {v: sum(var_series[v]) for v in all_vars}
    vars_ordered = sorted(var_totals.keys(), key=lambda v: var_totals[v], reverse=True)

    # Aggregate "other" if too many variables
    if len(vars_ordered) > top_n:
        top_vars = vars_ordered[:top_n]
        other_vars = vars_ordered[top_n:]

        other_series = [0.0] * len(cell_nums)
        for v in other_vars:
            for i, val in enumerate(var_series[v]):
                other_series[i] += val
            del var_series[v]

        var_series["other"] = other_series
        vars_ordered = top_vars + ["other"]

    # Count initial (non-rerun) cells
    initial_count = len([c for c in cells if not getattr(c, 'is_rerun', False)])

    log(f"Plot 2 v5: {len(vars_ordered)} vars, top: {vars_ordered[:3]}")

    return Plot2Data(
        cells=cell_nums,
        var_series={v: var_series[v] for v in vars_ordered},
        vars_ordered=vars_ordered,
        initial_count=initial_count if initial_count > 0 else len(cells),
        var_types=var_types,
    )


def extract_plot3_data_v5(
    cells: List[V5CellMemory],
    baseline_cells: Optional[List] = None,
    gpu_overhead_from_timing: Optional[List[float]] = None,
) -> Optional[Plot3Data]:
    """Extract data for Plot 3 from v5 memory cells.

    Shows stacked area chart with three layers:
    - user_ns_mb: User namespace memory
    - gpu_mb: GPU memory
    - overhead_mb: FlowBook checkpoint overhead

    When baseline_cells provided (cross-run comparison):
        overhead = flowbook.total - baseline.total

    When no baseline (FlowBook only):
        overhead = flowbook.checkpoint_mb (direct from kernel API)

    Args:
        cells: List of V5CellMemory objects (FlowBook)
        baseline_cells: Optional list of baseline cell memory (for cross-run comparison)
        gpu_overhead_from_timing: Optional per-cell GPU overhead computed from
            timing phase diff (flowbook_gpu - baseline_gpu). When provided,
            overrides the cudf-based gpu_checkpoint_mb measurement.

    Returns:
        Plot3Data or None if no cells
    """
    if not cells:
        return None

    has_baseline = baseline_cells is not None and len(baseline_cells) > 0

    cell_nums = []
    user_ns_mb = []
    gpu_mb = []
    overhead_mb = []

    for i, cell in enumerate(cells):
        cell_nums.append(cell.cell_index + 1)

        if has_baseline and i < len(baseline_cells):
            # Cross-run comparison: overhead = flowbook.total - baseline.total
            bl_cell = baseline_cells[i]
            if hasattr(bl_cell, 'post'):
                bl_total = bl_cell.post.total_mb
            else:
                bl_total = getattr(bl_cell, 'user_ns_mb', 0) + getattr(bl_cell, 'gpu_mb', 0)
            flow_total = cell.total_mb  # user_ns + gpu + checkpoint
            overhead = max(0, flow_total - bl_total)
        else:
            # FlowBook only: use checkpoint_mb directly
            overhead = cell.checkpoint_mb

        user_ns_mb.append(cell.user_ns_mb)
        gpu_mb.append(cell.gpu_mb)
        overhead_mb.append(overhead)

    # Compute peak FlowBook total and peak base total across all cells
    # flowbook_total = user_ns + gpu + checkpoint
    # base_total = user_ns + gpu
    flowbook_totals = [cell.user_ns_mb + cell.gpu_mb + cell.checkpoint_mb for cell in cells]
    base_totals = [cell.user_ns_mb + cell.gpu_mb for cell in cells]
    peak_flowbook_mb = max(flowbook_totals) if flowbook_totals else 0
    peak_base_mb = max(base_totals) if base_totals else 0

    # Peak overhead percentage: (max_flowbook / max_base - 1) * 100
    if peak_base_mb > 0:
        peak_pct = (peak_flowbook_mb / peak_base_mb - 1) * 100
    else:
        peak_pct = 0

    # Find cell with max overhead for annotation placement
    if overhead_mb:
        peak_idx = overhead_mb.index(max(overhead_mb))
        peak_overhead = overhead_mb[peak_idx]
    else:
        peak_idx = 0
        peak_overhead = 0

    mode = "cross-run" if has_baseline else "FlowBook-only"
    log(f"Plot 3 v5 ({mode}): {len(cells)} cells, peak overhead {peak_pct:.1f}% "
        f"(FlowBook: {peak_flowbook_mb:.1f} MB, Base: {peak_base_mb:.1f} MB)")

    # GPU checkpoint overhead per cell
    # Prefer timing-derived diff (flowbook_gpu - baseline_gpu) over cudf-based measurement
    if gpu_overhead_from_timing is not None and len(gpu_overhead_from_timing) == len(cells):
        gpu_ckpt_mb = list(gpu_overhead_from_timing)
    else:
        gpu_ckpt_mb = [getattr(cell, 'gpu_checkpoint_mb', 0.0) for cell in cells]

    return Plot3Data(
        cells=cell_nums,
        user_ns_mb=user_ns_mb,
        gpu_mb=gpu_mb,
        overhead_mb=overhead_mb,
        has_baseline=has_baseline,
        peak_overhead_mb=peak_overhead,
        peak_overhead_pct=peak_pct,
        peak_cell=peak_idx,
        initial_count=len(cells),
        peak_flowbook_mb=peak_flowbook_mb,
        peak_base_mb=peak_base_mb,
        gpu_checkpoint_mb=gpu_ckpt_mb,
    )


def extract_plot4_data_v5(cells: List[V5CellMemory], top_n: int = 10) -> Optional[Plot4Data]:
    """Extract data for Plot 4 from v5 memory cells.

    Shows stacked area chart:
    - Bottom: user namespace (gray)
    - Middle: GPU memory (orange)
    - Top: per-variable checkpoint sizes (colors)

    Args:
        cells: List of V5CellMemory objects
        top_n: Number of top variables to show

    Returns:
        Plot4Data or None if no cells
    """
    if not cells:
        return None

    # Collect all variable names
    all_vars: Set[str] = set()
    for cell in cells:
        all_vars.update(cell.checkpoint_vars.keys())

    if not all_vars:
        log("Plot 4 v5: No checkpoint variables found")
        return None

    cell_nums = []
    namespace_mb = []
    gpu_mb = []
    var_series: Dict[str, List[float]] = {v: [] for v in all_vars}
    var_types: Dict[str, str] = {}

    for cell in cells:
        cell_nums.append(cell.cell_index + 1)
        namespace_mb.append(cell.user_ns_mb)
        gpu_mb.append(cell.gpu_mb)

        for v in all_vars:
            var_series[v].append(cell.checkpoint_vars.get(v, 0.0))

        # Aggregate var types from all cells (first type found wins)
        for v, t in cell.checkpoint_var_types.items():
            if v not in var_types and t:
                var_types[v] = t

    # Order by max size
    var_maxes = {v: max(var_series[v]) for v in all_vars}
    vars_ordered = sorted(var_maxes.keys(), key=lambda v: var_maxes[v], reverse=True)

    # Aggregate "other" if needed
    if len(vars_ordered) > top_n:
        top_vars = vars_ordered[:top_n]
        other_vars = vars_ordered[top_n:]

        other_series = [0.0] * len(cells)
        for v in other_vars:
            for i, val in enumerate(var_series[v]):
                other_series[i] += val
            del var_series[v]

        var_series["other"] = other_series
        vars_ordered = top_vars + ["other"]

    log(f"Plot 4 v5: {len(vars_ordered)} vars, top: {vars_ordered[:3]}")

    return Plot4Data(
        cells=cell_nums,
        namespace_mb=namespace_mb,
        gpu_mb=gpu_mb,
        var_series={v: var_series[v] for v in vars_ordered},
        vars_ordered=vars_ordered,
        var_types=var_types,
        initial_count=len(cells),
    )


def extract_plot6_data_v5(cells: List[V5CellMemory]) -> Optional[Plot6Data]:
    """Extract data for Plot 6 from v5 memory cells.

    For each cell, compute:
        ratio = checkpoint_delta_mb / prev_cell_base_mb

    This shows how much new checkpoint data was created relative to the
    base memory at the start of that cell.

    Args:
        cells: List of V5CellMemory objects

    Returns:
        Plot6Data or None if no cells
    """
    if not cells:
        return None

    cell_nums = []
    ratios = []
    gpu_ratios = []

    for i, cell in enumerate(cells):
        cell_nums.append(cell.cell_index + 1)

        # Compute checkpoint delta (new checkpoint data at this cell)
        if i == 0:
            delta_mb = cell.checkpoint_mb
            gpu_delta_mb = getattr(cell, 'gpu_checkpoint_mb', 0.0)
            prev_base_mb = 0.0
        else:
            prev_cell = cells[i - 1]
            delta_mb = max(0, cell.checkpoint_mb - prev_cell.checkpoint_mb)
            gpu_delta_mb = max(0, getattr(cell, 'gpu_checkpoint_mb', 0.0) - getattr(prev_cell, 'gpu_checkpoint_mb', 0.0))
            prev_base_mb = prev_cell.base_mb

        # Compute ratio (0 if base is too small)
        if prev_base_mb >= MIN_BASE_MB:
            ratio = delta_mb / prev_base_mb
            gpu_ratio = gpu_delta_mb / prev_base_mb
        else:
            ratio = 0.0
            gpu_ratio = 0.0
        ratios.append(ratio)
        gpu_ratios.append(gpu_ratio)

    nonzero_count = sum(1 for r in ratios if r > 0)
    log(f"Plot 6 v5: {len(cells)} cells, {nonzero_count} with nonzero ratios")

    return Plot6Data(
        cells=cell_nums,
        ratios=ratios,
        gpu_ratios=gpu_ratios,
        initial_count=len(cells),
    )


def extract_baseline_cells(data: Dict) -> Optional[List]:
    """Extract baseline cell memory from JSON data.

    Returns cells in a format compatible with cross-run comparison.

    Args:
        data: JSON data dict (full comparison result)

    Returns:
        List of baseline cells or None
    """
    kernels = data.get("kernels", {})
    bl_data = kernels.get("baseline", {})
    memory_data = bl_data.get("memory", {})
    if not memory_data:
        return None

    cells_data = memory_data.get("cells", [])
    rerun_data = memory_data.get("rerun_cells", [])

    # Convert to simple objects with post.total_mb
    class SimpleCell:
        def __init__(self, d):
            self.cell_id = d.get("cell_id", "")
            self.cell_index = d.get("cell_index", 0)
            # V4 format has post_user_ns_mb, post_gpu_mb
            user_ns = d.get("post_user_ns_mb", 0.0)
            gpu = d.get("post_gpu_mb", 0.0)
            self.post = type('Post', (), {'total_mb': user_ns + gpu, 'user_ns_mb': user_ns, 'gpu_mb': gpu})()

    all_cells = [SimpleCell(c) for c in cells_data + rerun_data]
    return all_cells if all_cells else None


def extract_v5_memory_result(data: Dict) -> Optional[V5MemoryResult]:
    """Extract V5MemoryResult from JSON data.

    Handles both v5.0 format and attempts conversion from v4.0.

    Args:
        data: JSON data dict (full comparison result or kernels section)

    Returns:
        V5MemoryResult or None
    """
    # Check version
    version = data.get("version", "")

    if version.startswith("5"):
        # Native v5 format
        kernels = data.get("kernels", {})
        fb_data = kernels.get("flowbook", {})
        memory_data = fb_data.get("memory", {})
        if memory_data:
            return V5MemoryResult.from_dict(memory_data)
        return None

    elif version.startswith("4"):
        # Convert from v4 format
        kernels = data.get("kernels", {})
        fb_data = kernels.get("flowbook", {})
        memory_data = fb_data.get("memory", {})
        if not memory_data:
            return None

        # Convert v4 cells to v5
        v5_cells = []
        for cell_data in memory_data.get("cells", []):
            # V4 has post_user_ns_mb, post_gpu_mb, post_overhead_mb, post_checkpoint_vars
            user_ns_mb = cell_data.get("post_user_ns_mb", 0.0)
            gpu_mb = cell_data.get("post_gpu_mb", 0.0)
            overhead_mb = cell_data.get("post_overhead_mb", 0.0)

            # Extract checkpoint_vars from nested structure
            checkpoint_vars = {}
            post_ckpt = cell_data.get("post_checkpoint_vars", {})
            for ckpt_name, ckpt_vars in post_ckpt.items():
                for var_name, var_info in ckpt_vars.items():
                    if isinstance(var_info, dict):
                        size_mb = var_info.get("size_mb", 0.0)
                    else:
                        size_mb = float(var_info)
                    checkpoint_vars[var_name] = checkpoint_vars.get(var_name, 0.0) + size_mb

            v5_cells.append(V5CellMemory(
                cell_id=cell_data.get("cell_id", ""),
                cell_index=cell_data.get("cell_index", 0),
                user_ns_mb=user_ns_mb,
                gpu_mb=gpu_mb,
                checkpoint_mb=overhead_mb,
                checkpoint_vars=checkpoint_vars,
            ))

        # Convert rerun cells similarly
        v5_reruns = []
        for cell_data in memory_data.get("rerun_cells", []):
            user_ns_mb = cell_data.get("post_user_ns_mb", 0.0)
            gpu_mb = cell_data.get("post_gpu_mb", 0.0)
            overhead_mb = cell_data.get("post_overhead_mb", 0.0)

            checkpoint_vars = {}
            post_ckpt = cell_data.get("post_checkpoint_vars", {})
            for ckpt_name, ckpt_vars in post_ckpt.items():
                for var_name, var_info in ckpt_vars.items():
                    if isinstance(var_info, dict):
                        size_mb = var_info.get("size_mb", 0.0)
                    else:
                        size_mb = float(var_info)
                    checkpoint_vars[var_name] = checkpoint_vars.get(var_name, 0.0) + size_mb

            v5_reruns.append(V5CellMemory(
                cell_id=cell_data.get("cell_id", ""),
                cell_index=cell_data.get("cell_index", 0),
                user_ns_mb=user_ns_mb,
                gpu_mb=gpu_mb,
                checkpoint_mb=overhead_mb,
                checkpoint_vars=checkpoint_vars,
            ))

        return V5MemoryResult(cells=v5_cells, rerun_cells=v5_reruns)

    return None
