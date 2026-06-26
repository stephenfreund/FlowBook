"""Plot rendering functions for compare_overhead.

This module takes PlotNData dataclasses from plot_extraction.py and renders
matplotlib figures. Each render_plotN function takes a PlotNData and axes,
producing a visualization.

Usage:
    from flowbook.cli.plot_extraction import extract_plot3_data
    from flowbook.cli.plot_rendering import render_plot3

    p3 = extract_plot3_data(result)
    fig, ax = plt.subplots()
    render_plot3(ax, p3)
"""

from typing import List, Optional

import numpy as np

from flowbook.cli.models import (
    Plot1Data,
    Plot2Data,
    Plot3Data,
    Plot4Data,
    Plot5Data,
    Plot6Data,
    CDFData,
)


def render_plot1(
    ax,
    data: Plot1Data,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    notebook_name: str = "",
) -> None:
    """Render Plot 1: Execution Time per Cell.

    Stacked area chart showing cumulative time:
    - Code execution (bottom)
    - State management
    - Check operations
    - Other overhead (top)

    Args:
        ax: Matplotlib axes
        data: Plot1Data with timing arrays
        colors: Color palette (uses seaborn default if None)
        large_fonts: Use larger fonts for readability
        show_legend: Whether to show legend
        notebook_name: Optional notebook name for title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if colors is None:
        colors = sns.color_palette()

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    cells = np.array(data.cells)
    code_arr = np.array(data.run_time_sec)
    state_arr = np.array(data.state_time_sec)
    check_arr = np.array(data.check_time_sec)
    other_arr = np.array(data.other_time_sec)

    # Cumulative sums
    code_cumsum = np.cumsum(code_arr)
    state_cumsum = np.cumsum(state_arr)
    check_cumsum = np.cumsum(check_arr)
    other_cumsum = np.cumsum(other_arr)

    # Blue line with markers showing code time (baseline-like reference)
    ax.plot(
        cells,
        code_cumsum,
        color=colors[0],
        linewidth=2,
        marker="o",
        markersize=4,
        label="Code (no baseline)",
    )

    # Stacked areas: code (bottom) + state + check + other (top)
    ax.fill_between(
        cells, 0, code_cumsum, alpha=0.3, color=colors[1], label="FlowBook Code"
    )
    ax.fill_between(
        cells,
        code_cumsum,
        code_cumsum + state_cumsum,
        alpha=0.4,
        color=colors[2],
        label="State",
    )
    ax.fill_between(
        cells,
        code_cumsum + state_cumsum,
        code_cumsum + state_cumsum + check_cumsum,
        alpha=0.4,
        color=colors[3],
        label="Check",
    )
    ax.fill_between(
        cells,
        code_cumsum + state_cumsum + check_cumsum,
        code_cumsum + state_cumsum + check_cumsum + other_cumsum,
        alpha=0.4,
        color=colors[4],
        label="Other",
    )

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Cumulative Time (seconds)", fontsize=label_size)

    title = "Execution Time"
    if data.initial_count < len(cells):
        title += f" (cells 1-{data.initial_count} + {len(cells) - data.initial_count} reruns)"
    ax.set_title(title, fontsize=title_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Rerun separator
    if data.initial_count < len(cells):
        ax.axvline(
            x=data.initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

    if show_legend:
        ax.legend(loc="upper left", fontsize=legend_size)

    # Summary text
    total_code = code_cumsum[-1] if len(code_cumsum) > 0 else 0
    total_overhead = (
        (state_cumsum[-1] + check_cumsum[-1] + other_cumsum[-1])
        if len(state_cumsum) > 0
        else 0
    )
    total = total_code + total_overhead

    textstr = f"Code: {total_code:.2f}s\nTotal: {total:.2f}s"
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(
        0.02,
        0.70,
        textstr,
        transform=ax.transAxes,
        fontsize=legend_size,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
    )

    # Overhead percentage (vs code time)
    if total_code > 0:
        overhead_pct = (total_overhead / total_code) * 100
        ax.annotate(
            f"{overhead_pct:.1f}% overhead (vs code)",
            xy=(cells[-1], total),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=legend_size,
            va="center",
            ha="left",
            color=colors[1],
        )


def render_plot2(
    ax,
    data: Plot2Data,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    notebook_name: str = "",
) -> None:
    """Render Plot 2: Checkpoint Time by Variable.

    Stacked area chart showing checkpoint deepcopy time per variable.

    Args:
        ax: Matplotlib axes
        data: Plot2Data with per-variable timing
        colors: Color palette
        large_fonts: Use larger fonts
        show_legend: Whether to show legend
        notebook_name: Optional notebook name for title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if colors is None:
        colors = sns.color_palette("husl", len(data.vars_ordered))

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    cells = np.array(data.cells)
    cumulative = np.zeros(len(cells))

    for i, var in enumerate(data.vars_ordered):
        var_data = np.array(data.var_series[var])
        var_type = data.var_types.get(var, "") if hasattr(data, "var_types") else ""
        if var == "other":
            label = f"other ({len(data.vars_ordered)} vars)"
        elif var_type:
            label = f"{var} ({var_type})"
        else:
            label = var
        ax.fill_between(
            cells,
            cumulative,
            cumulative + var_data,
            alpha=0.7,
            color=colors[i % len(colors)],
            label=label,
        )
        cumulative = cumulative + var_data

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Checkpoint Time (seconds)", fontsize=label_size)

    title = "Checkpoint Time by Variable"
    ax.set_title(title, fontsize=title_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Rerun separator
    if data.initial_count < len(cells):
        ax.axvline(x=data.initial_count + 0.5, color="red", linestyle="--", linewidth=2)

    if show_legend:
        ax.legend(loc="upper left", fontsize=legend_size, ncol=2)


def render_plot3(
    ax,
    data: Plot3Data,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    notebook_name: str = "",
) -> None:
    """Render Plot 3: Memory Overhead.

    Stacked area chart showing three layers:
    - User namespace (gray)
    - GPU memory (orange)
    - Checkpoint overhead (blue)

    Args:
        ax: Matplotlib axes
        data: Plot3Data with memory arrays
        colors: Color palette
        large_fonts: Use larger fonts
        show_legend: Whether to show legend
        notebook_name: Optional notebook name for title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if colors is None:
        colors = sns.color_palette()

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    cells = np.array(data.cells)
    user_ns = np.array(data.user_ns_mb)
    gpu = np.array(data.gpu_mb)
    overhead = np.array(data.overhead_mb)

    # Stack layers: namespace, then GPU, then CPU overhead, then GPU checkpoint
    layer1 = user_ns
    layer2 = user_ns + gpu
    layer3 = user_ns + gpu + overhead

    gpu_ckpt = (
        np.array(data.gpu_checkpoint_mb)
        if data.gpu_checkpoint_mb
        else np.zeros_like(cells, dtype=float)
    )
    has_gpu_ckpt = np.any(gpu_ckpt > 0)
    layer4 = layer3 + gpu_ckpt

    # Fixed colors: gray for namespace, orange for GPU, steelblue for overhead, red for GPU checkpoint
    ns_color = "gray"
    gpu_color = "orange"
    overhead_color = "steelblue"
    gpu_ckpt_color = "red"

    ax.fill_between(cells, 0, layer1, alpha=0.3, color=ns_color, label="User Namespace")
    ax.fill_between(
        cells, layer1, layer2, alpha=0.3, color=gpu_color, label="GPU Memory"
    )
    ax.fill_between(
        cells,
        layer2,
        layer3,
        alpha=0.3,
        color=overhead_color,
        label="FlowBook Overhead",
    )
    if has_gpu_ckpt:
        ax.fill_between(
            cells,
            layer3,
            layer4,
            alpha=0.3,
            color=gpu_ckpt_color,
            label="GPU Checkpoint",
        )

    ax.plot(cells, layer1, color=ns_color, linewidth=1.5, marker="o", markersize=3)
    ax.plot(cells, layer2, color=gpu_color, linewidth=1.5, marker="o", markersize=3)
    ax.plot(cells, layer3, color=overhead_color, linewidth=2, marker="o", markersize=4)
    if has_gpu_ckpt:
        ax.plot(
            cells, layer4, color=gpu_ckpt_color, linewidth=2, marker="o", markersize=4
        )

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Memory (MB)", fontsize=label_size)

    # Title without notebook name (notebook name goes in figure suptitle)
    title = "Memory Overhead"
    if data.initial_count < len(cells):
        title += f" (cells 1-{data.initial_count} + {len(cells) - data.initial_count} reruns)"
    ax.set_title(title, fontsize=title_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Rerun separator
    if data.initial_count < len(cells):
        ax.axvline(
            x=data.initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

    if show_legend:
        ax.legend(loc="upper left", fontsize=legend_size)

    # Peak overhead annotation showing FlowBook total and Base total
    if data.peak_flowbook_mb > 0 and data.peak_base_mb > 0:
        ax.annotate(
            f"Peak: {data.peak_overhead_pct:.1f}% ({data.peak_flowbook_mb:.1f} / {data.peak_base_mb:.1f} MB)",
            xy=(data.peak_cell + 1, layer3[data.peak_cell]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=legend_size,
            va="bottom",
            ha="left",
            color=overhead_color,
            fontweight="bold",
        )
    elif data.peak_overhead_mb > 0:
        # Fallback for older data without peak_flowbook_mb/peak_base_mb
        ax.annotate(
            f"Peak: {data.peak_overhead_pct:.1f}%",
            xy=(data.peak_cell + 1, layer3[data.peak_cell]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=legend_size,
            va="bottom",
            ha="left",
            color=overhead_color,
            fontweight="bold",
        )


def render_plot4(
    ax,
    data: Plot4Data,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    notebook_name: str = "",
) -> None:
    """Render Plot 4: Checkpoint Memory by Variable.

    Stacked area chart showing:
    - User namespace (bottom, gray)
    - GPU memory (orange)
    - Per-variable checkpoint sizes (colors)

    Args:
        ax: Matplotlib axes
        data: Plot4Data with per-variable memory
        colors: Color palette
        large_fonts: Use larger fonts
        show_legend: Whether to show legend
        notebook_name: Optional notebook name for title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    cells = np.array(data.cells)
    namespace = np.array(data.namespace_mb)
    gpu = np.array(data.gpu_mb)

    # Variable colors
    var_colors = sns.color_palette("husl", len(data.vars_ordered))

    cumulative = np.zeros(len(cells))

    # Namespace (gray)
    ax.fill_between(
        cells, cumulative, namespace, alpha=0.3, color="gray", label="Namespace"
    )
    cumulative = namespace.copy()

    # GPU (orange)
    if np.any(gpu > 0):
        ax.fill_between(
            cells, cumulative, cumulative + gpu, alpha=0.4, color="orange", label="GPU"
        )
        cumulative = cumulative + gpu

    # Per-variable checkpoints
    for i, var in enumerate(data.vars_ordered):
        var_data = np.array(data.var_series[var])
        var_type = data.var_types.get(var, "")
        label = f"{var} ({var_type})" if var_type else var
        if var == "other":
            label = f"other ({len(data.vars_ordered)} vars)"

        ax.fill_between(
            cells,
            cumulative,
            cumulative + var_data,
            alpha=0.7,
            color=var_colors[i % len(var_colors)],
            label=label,
        )
        cumulative = cumulative + var_data

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Memory (MB)", fontsize=label_size)

    title = "Checkpoint Memory by Variable"
    ax.set_title(title, fontsize=title_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Rerun separator
    if data.initial_count < len(cells):
        ax.axvline(x=data.initial_count + 0.5, color="red", linestyle="--", linewidth=2)

    if show_legend:
        ax.legend(loc="upper left", fontsize=legend_size, ncol=2)


def render_plot5(
    ax,
    data: Plot5Data,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    notebook_name: str = "",
) -> None:
    """Render Plot 5: Overhead Time per Cell.

    Bar chart showing overhead breakdown per cell:
    - State management time
    - Check time
    - Other overhead

    Args:
        ax: Matplotlib axes
        data: Plot5Data with overhead arrays
        colors: Color palette
        large_fonts: Use larger fonts
        show_legend: Whether to show legend
        notebook_name: Optional notebook name for title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if colors is None:
        colors = sns.color_palette()

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    cells = np.array(data.cells)
    state = np.array(data.state_sec)
    check = np.array(data.check_sec)
    other = np.array(data.other_sec)

    width = 0.8
    ax.bar(cells, state, width, label="State", color=colors[2], alpha=0.7)
    ax.bar(cells, check, width, bottom=state, label="Check", color=colors[3], alpha=0.7)
    ax.bar(
        cells,
        other,
        width,
        bottom=state + check,
        label="Other",
        color=colors[4],
        alpha=0.7,
    )

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Overhead Time (seconds)", fontsize=label_size)

    title = "Overhead Time per Cell"
    ax.set_title(title, fontsize=title_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=0.5, right=len(cells) + 0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Rerun separator
    if data.initial_count < len(cells):
        ax.axvline(
            x=data.initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

    if show_legend:
        ax.legend(loc="upper right", fontsize=legend_size)


def render_plot6(
    ax,
    data: Plot6Data,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    notebook_name: str = "",
) -> None:
    """Render Plot 6: Checkpoint Overhead Ratio per Cell.

    Bar chart showing checkpoint_delta / base_memory ratio for each cell.

    Args:
        ax: Matplotlib axes
        data: Plot6Data with per-cell ratios
        colors: Color palette
        large_fonts: Use larger fonts
        show_legend: Whether to show legend
        notebook_name: Optional notebook name for title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if colors is None:
        colors = sns.color_palette()

    label_size = 18 if large_fonts else 12
    title_size = 20 if large_fonts else 14
    legend_size = 14 if large_fonts else 10
    tick_size = 14 if large_fonts else 10

    cells = np.array(data.cells)
    ratios = np.array(data.ratios)
    gpu_ratios = np.array(data.gpu_ratios) if data.gpu_ratios else np.zeros_like(ratios)
    has_gpu = np.any(gpu_ratios > 0)

    if has_gpu:
        # Grouped bars: CPU and GPU side by side
        bar_width = 0.35
        ax.bar(
            cells - bar_width / 2,
            ratios,
            width=bar_width,
            alpha=0.7,
            color="#66c2a5",
            label="CPU Checkpoint",
        )
        ax.bar(
            cells + bar_width / 2,
            gpu_ratios,
            width=bar_width,
            alpha=0.7,
            color="red",
            label="GPU Checkpoint",
        )
    else:
        # Single bar (no GPU data)
        bar_width = 0.6
        ax.bar(cells, ratios, width=bar_width, alpha=0.7, color="#66c2a5")

    ax.set_xlabel("Cell Number", fontsize=label_size)
    ax.set_ylabel("Checkpoint / Base Memory", fontsize=label_size)

    title = "Checkpoint Overhead Ratio"
    ax.set_title(title, fontsize=title_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(left=0.5, right=len(cells) + 0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Rerun separator
    if data.initial_count < len(cells):
        ax.axvline(
            x=data.initial_count + 0.5,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Rerun Start",
        )

    if show_legend and (has_gpu or data.initial_count < len(cells)):
        ax.legend(loc="upper right", fontsize=legend_size)


def render_time_cdf(
    ax,
    sorted_vals: List[float],
    percentiles: List[float],
    n: int,
    color: str = "steelblue",
    title: str = "Analysis Time Distribution",
    xlabel: str = "Analysis Time (ms, log scale)",
    large_fonts: bool = True,
    show_sample_size: bool = True,
    value_fmt=None,
    tick_fmt=None,
) -> None:
    """Render a value-based CDF panel with log scale x-axis.

    This is a shared helper for both Analysis Time Distribution and
    Rerun Overhead Time Distribution CDFs.

    Args:
        ax: Matplotlib axes
        sorted_vals: Sorted time values in ms
        percentiles: Corresponding CDF percentiles (0-1)
        n: Number of data points (for display)
        color: Line/fill color
        title: Plot title
        xlabel: X-axis label
        large_fonts: Use larger fonts
    """
    from matplotlib.ticker import FuncFormatter

    # Font sizes
    # label_size = 22 if large_fonts else 14
    # title_size = 24 if large_fonts else 16
    # tick_size = 18 if large_fonts else 12
    # annotation_size = 18 if large_fonts else 12
    # legend_fontsize = 14 if large_fonts else 10

    label_size = 18 if large_fonts else 14
    title_size = 20 if large_fonts else 16
    tick_size = 14 if large_fonts else 12
    annotation_size = 14 if large_fonts else 12
    legend_fontsize = 12 if large_fonts else 10

    if not sorted_vals:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=title_size)
        return

    sorted_arr = np.array(sorted_vals)
    cdf_arr = np.array(percentiles)

    # Compute percentile stats (P50, P95, P99 only - no P90)
    stats = {
        "P50": np.percentile(sorted_arr, 50),
        "P95": np.percentile(sorted_arr, 95),
        "P99": np.percentile(sorted_arr, 99),
    }

    # For log scale, filter positive values
    pos_mask = sorted_arr > 0
    if not np.any(pos_mask):
        ax.text(
            0.5,
            0.5,
            "No positive data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=title_size)
        return
    plot_data = sorted_arr[pos_mask]
    plot_cdf = cdf_arr[pos_mask]

    # Plot CDF line and fill
    ax.fill_between(plot_data, 0, plot_cdf, alpha=0.12, color=color, edgecolor="none")
    ax.plot(plot_data, plot_cdf, color=color, linewidth=2.5)

    # Disable standard gridlines, use only custom reference lines
    ax.grid(False)

    # Subtle spines
    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)

    # Percentile config (P50, P95, P99 only)
    pct_config = [
        ("P50", 0.50, (12, 8), "bottom"),
        ("P95", 0.95, (12, -18), "top"),
        ("P99", 0.99, (12, 8), "bottom"),
    ]
    max_val = np.max(plot_data)

    # Faint reference lines at each point of interest
    line_color = "gray"
    line_alpha = 0.3
    line_width = 0.6
    for pname, y_val, _, _ in pct_config:
        x_val = stats[pname]
        ax.axhline(
            y=y_val,
            color=line_color,
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )
        ax.axvline(
            x=x_val,
            color=line_color,
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )
    # Max reference lines
    ax.axhline(
        y=1.0,
        color=line_color,
        linestyle="--",
        linewidth=line_width,
        alpha=line_alpha,
        zorder=1,
    )
    ax.axvline(
        x=max_val,
        color=line_color,
        linestyle="--",
        linewidth=line_width,
        alpha=line_alpha,
        zorder=1,
    )

    # Percentile markers with leader line annotations
    for pname, y_val, offset, va in pct_config:
        x_val = stats[pname]
        ax.scatter(
            [x_val],
            [y_val],
            color=color,
            s=40,
            marker="o",
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            pname,
            (x_val, y_val),
            textcoords="offset points",
            xytext=offset,
            fontsize=annotation_size,
            ha="left",
            va=va,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3),
        )

    # Max marker
    ax.scatter(
        [max_val],
        [1.0],
        color=color,
        s=40,
        marker="o",
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
    )
    ax.annotate(
        "Max",
        (max_val, 1.0),
        textcoords="offset points",
        xytext=(12, -18),
        fontsize=annotation_size,
        ha="left",
        va="top",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3),
    )

    # Format function for milliseconds (default when no formatter supplied)
    def fmt_ms(v):
        if v >= 1:
            return f"{v:.1f}ms"
        else:
            return f"{v:.2f}ms"

    fmt = value_fmt or fmt_ms

    # Legend box with values (right-aligned, monospace)
    formatted = {p: fmt(stats[p]) for p in ["P50", "P95", "P99"]}
    formatted["Max"] = fmt(max_val)
    max_val_len = max(len(v) for v in formatted.values())
    legend_lines = [
        f"{p:>3}  {formatted[p]:>{max_val_len}}" for p in ["P50", "P95", "P99", "Max"]
    ]
    legend_text = "\n".join(legend_lines)
    props = dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray")
    ax.text(
        0.98,
        0.02,
        legend_text,
        transform=ax.transAxes,
        fontsize=legend_fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    # N count in top left
    if show_sample_size:
        textstr = f"N={n:,}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=tick_size,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"
            ),
        )

    # Axes config
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel("Cumulative Proportion", fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_xscale("log")
    if tick_fmt is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(tick_fmt))
    else:
        # Format with commas (e.g., "1,000" instead of "1000")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))


def render_overhead_pct_cdf(
    ax, data: CDFData, large_fonts: bool = True, show_sample_size: bool = True
) -> None:
    """Render CDF for per-cell overhead percentage distribution.

    Overhead percentage = (state + check) / base * 100

    Args:
        ax: Matplotlib axes
        data: CDFData with overhead_pct_* fields
        large_fonts: Use larger fonts
        show_sample_size: Whether to show N= annotation
    """
    from matplotlib.ticker import FuncFormatter

    label_size = 22 if large_fonts else 14
    title_size = 24 if large_fonts else 16
    tick_size = 18 if large_fonts else 12
    annotation_size = 18 if large_fonts else 12
    legend_fontsize = 14 if large_fonts else 10

    sorted_vals = data.overhead_pct_sorted
    percentiles = data.overhead_pct_percentiles
    n = len(data.overhead_pct)
    color = "darkgreen"

    if not sorted_vals:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Cell Overhead Distribution", fontsize=title_size)
        return

    import numpy as np
    sorted_arr = np.array(sorted_vals)
    cdf_arr = np.array(percentiles)

    # Compute percentile stats
    stats = {
        "P50": np.percentile(sorted_arr, 50),
        "P95": np.percentile(sorted_arr, 95),
        "P99": np.percentile(sorted_arr, 99),
    }
    max_val = np.max(sorted_arr)

    # Plot CDF line and fill
    ax.fill_between(sorted_arr, 0, cdf_arr, alpha=0.12, color=color, edgecolor="none")
    ax.plot(sorted_arr, cdf_arr, color=color, linewidth=2.5)

    # Disable standard gridlines
    ax.grid(False)

    # Subtle spines
    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)

    # Percentile config
    pct_config = [
        ("P50", 0.50, (12, 8), "bottom"),
        ("P95", 0.95, (12, -18), "top"),
        ("P99", 0.99, (12, 8), "bottom"),
    ]

    # Faint reference lines
    line_color = "gray"
    line_alpha = 0.3
    line_width = 0.6
    for pname, y_val, _, _ in pct_config:
        x_val = stats[pname]
        ax.axhline(y=y_val, color=line_color, linestyle="--", linewidth=line_width, alpha=line_alpha, zorder=1)
        ax.axvline(x=x_val, color=line_color, linestyle="--", linewidth=line_width, alpha=line_alpha, zorder=1)
    ax.axhline(y=1.0, color=line_color, linestyle="--", linewidth=line_width, alpha=line_alpha, zorder=1)
    ax.axvline(x=max_val, color=line_color, linestyle="--", linewidth=line_width, alpha=line_alpha, zorder=1)

    # Percentile markers
    for pname, y_val, offset, va in pct_config:
        x_val = stats[pname]
        ax.scatter([x_val], [y_val], color=color, s=40, marker="o", zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(
            pname, (x_val, y_val),
            textcoords="offset points", xytext=offset,
            fontsize=annotation_size, ha="left", va=va, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3)
        )

    # Max marker
    ax.scatter([max_val], [1.0], color=color, s=40, marker="o", zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate(
        "Max", (max_val, 1.0),
        textcoords="offset points", xytext=(12, -18),
        fontsize=annotation_size, ha="left", va="top", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3)
    )

    # Format percentage
    def fmt_pct(v):
        if v >= 10:
            return f"{v:.1f}%"
        elif v >= 1:
            return f"{v:.2f}%"
        else:
            return f"{v:.3f}%"

    # Legend box with values
    formatted = {p: fmt_pct(stats[p]) for p in ["P50", "P95", "P99"]}
    formatted["Max"] = fmt_pct(max_val)
    max_val_len = max(len(v) for v in formatted.values())
    legend_lines = [f"{p:>3}  {formatted[p]:>{max_val_len}}" for p in ["P50", "P95", "P99", "Max"]]
    legend_text = "\n".join(legend_lines)
    props = dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray")
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=legend_fontsize,
            verticalalignment="bottom", horizontalalignment="right", bbox=props,
            family="monospace")

    # N count in top left
    if show_sample_size:
        textstr = f"N={n:,}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=tick_size,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"))

    # Axes config
    ax.set_xlabel("Overhead Percentage", fontsize=label_size)
    ax.set_ylabel("Cumulative Proportion", fontsize=label_size)
    ax.set_title("Per-Cell Overhead Distribution", fontsize=title_size)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.tick_params(axis="both", labelsize=tick_size)

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}%"))


def render_base_runtime_cdf(
    ax, data: CDFData, large_fonts: bool = True, show_sample_size: bool = True
) -> None:
    """Render CDF for base runtime (code execution time) distribution.

    Args:
        ax: Matplotlib axes
        data: CDFData with base_runtime_* fields
        large_fonts: Use larger fonts
        show_sample_size: Whether to show N= annotation
    """
    render_time_cdf(
        ax,
        sorted_vals=list(data.base_runtime_sorted),
        percentiles=list(data.base_runtime_percentiles),
        n=len(data.base_runtime_ms),
        color="darkgreen",
        title="Base Runtime Distribution",
        xlabel="Code Execution Time (ms, log scale)",
        large_fonts=large_fonts,
        show_sample_size=show_sample_size,
    )


def render_cdf_panel(
    ax,
    data: CDFData,
    metric: str,
    colors=None,
    large_fonts: bool = True,
    show_legend: bool = True,
    color_override: str = None,
    title_override: str = None,
    show_sample_size: bool = True,
) -> None:
    """Render a single CDF panel for aggregate data.

    Args:
        ax: Matplotlib axes
        data: CDFData with aggregate ratios
        metric: One of "time", "memory", "peak"
        colors: Color palette (ignored, uses fixed colors per metric)
        large_fonts: Use larger fonts
        show_legend: Whether to show legend
        color_override: Optional color to use instead of the default for this metric
        title_override: Optional title to use instead of the default for this metric
        show_sample_size: Whether to show N= annotation
    """
    from matplotlib.ticker import FuncFormatter

    # Font sizes
    # label_size = 22 if large_fonts else 14
    # title_size = 24 if large_fonts else 16
    # tick_size = 18 if large_fonts else 12
    # annotation_size = 18 if large_fonts else 12
    # legend_fontsize = 14 if large_fonts else 10

    label_size = 18 if large_fonts else 14
    title_size = 20 if large_fonts else 16
    tick_size = 14 if large_fonts else 12
    annotation_size = 14 if large_fonts else 12
    legend_fontsize = 12 if large_fonts else 10

    # For time metric, use the shared helper
    if metric == "time":
        render_time_cdf(
            ax,
            sorted_vals=list(data.time_sorted),
            percentiles=list(data.time_percentiles),
            n=len(data.time_overhead_ms),
            color=color_override or "steelblue",
            title=title_override or "Analysis Time Distribution",
            xlabel="Analysis Time (ms, log scale)",
            large_fonts=large_fonts,
            show_sample_size=show_sample_size,
        )
        return

    # Absolute per-cell memory overhead (Checkpoint - Base, MB) on a log axis,
    # reusing the log-scale time helper with MB formatters.
    if metric == "memory_abs":
        def fmt_mb(v):
            if v >= 100:
                return f"{v:,.0f} MB"
            elif v >= 10:
                return f"{v:.1f} MB"
            elif v >= 1:
                return f"{v:.2f} MB"
            elif v > 0:
                return f"{v:.3f} MB"
            else:
                return "0 MB"

        def tick_mb(x, _pos):
            if x >= 1:
                return f"{x:,.0f}"
            elif x >= 0.1:
                return f"{x:.1f}"
            elif x >= 0.01:
                return f"{x:.2f}"
            else:
                return f"{x:.3f}"

        render_time_cdf(
            ax,
            sorted_vals=list(data.memory_abs_sorted),
            percentiles=list(data.memory_abs_percentiles),
            n=len(data.memory_abs_mb),
            color=color_override or "darkorange",
            title=title_override or "Memory Overhead Distribution",
            xlabel="Checkpoint - Base Memory Size (MB, log scale)",
            large_fonts=large_fonts,
            show_sample_size=show_sample_size,
            value_fmt=fmt_mb,
            tick_fmt=tick_mb,
        )
        return

    # Metric-specific config for non-time metrics
    if metric == "memory":
        sorted_vals = list(data.memory_sorted)
        percentiles = list(data.memory_percentiles)
        title = "Memory Overhead Distribution"
        xlabel = "Checkpoint / Base Memory Size"
        n = len(data.memory_ratios)
        color = "seagreen"
    elif metric == "peak":
        sorted_vals = list(data.peak_sorted)
        percentiles = list(data.peak_percentiles)
        title = "Peak Memory Overhead Distribution"
        xlabel = "Peak Checkpoint / Base Memory Size"
        n = len(data.peak_memory_pct)
        color = "darkorange"
    elif metric == "gpu_memory":
        sorted_vals = list(data.gpu_memory_sorted)
        percentiles = list(data.gpu_memory_percentiles)
        title = "GPU Checkpoint Memory Distribution"
        xlabel = "GPU Checkpoint / Base Memory Size"
        n = len(data.gpu_memory_ratios)
        color = "red"
    elif metric == "gpu_peak":
        sorted_vals = list(data.gpu_peak_sorted)
        percentiles = list(data.gpu_peak_percentiles)
        title = "Peak GPU Checkpoint Overhead Distribution"
        xlabel = "Peak GPU Checkpoint / Base Memory Size"
        n = len(data.gpu_peak_memory_pct)
        color = "darkred"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Apply overrides if provided
    if color_override:
        color = color_override
    if title_override:
        title = title_override

    if not sorted_vals:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=title_size)
        return

    sorted_arr = np.array(sorted_vals)
    cdf_arr = np.array(percentiles)

    # Compute percentile stats (P50, P95, P99 only - no P90)
    stats = {
        "P50": np.percentile(sorted_arr, 50),
        "P95": np.percentile(sorted_arr, 95),
        "P99": np.percentile(sorted_arr, 99),
    }

    plot_data = sorted_arr
    plot_cdf = cdf_arr

    # Plot CDF line and fill
    ax.fill_between(plot_data, 0, plot_cdf, alpha=0.12, color=color, edgecolor="none")
    ax.plot(plot_data, plot_cdf, color=color, linewidth=2.5)

    # Disable standard gridlines, use only custom reference lines
    ax.grid(False)

    # Subtle spines
    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)

    # Percentile config (P50, P95, P99 only)
    pct_config = [
        ("P50", 0.50, (12, 8), "bottom"),
        ("P95", 0.95, (12, -18), "top"),
        ("P99", 0.99, (12, 8), "bottom"),
    ]
    max_val = np.max(plot_data)

    # Faint reference lines at each point of interest
    line_color = "gray"
    line_alpha = 0.3
    line_width = 0.6
    for pname, y_val, _, _ in pct_config:
        x_val = stats[pname]
        ax.axhline(
            y=y_val,
            color=line_color,
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )
        ax.axvline(
            x=x_val,
            color=line_color,
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )
    # Max reference lines
    ax.axhline(
        y=1.0,
        color=line_color,
        linestyle="--",
        linewidth=line_width,
        alpha=line_alpha,
        zorder=1,
    )
    ax.axvline(
        x=max_val,
        color=line_color,
        linestyle="--",
        linewidth=line_width,
        alpha=line_alpha,
        zorder=1,
    )

    # Percentile markers with leader line annotations
    for pname, y_val, offset, va in pct_config:
        x_val = stats[pname]
        ax.scatter(
            [x_val],
            [y_val],
            color=color,
            s=40,
            marker="o",
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            pname,
            (x_val, y_val),
            textcoords="offset points",
            xytext=offset,
            fontsize=annotation_size,
            ha="left",
            va=va,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3),
        )

    # Max marker
    ax.scatter(
        [max_val],
        [1.0],
        color=color,
        s=40,
        marker="o",
        zorder=5,
        edgecolors="white",
        linewidths=1.5,
    )
    ax.annotate(
        "Max",
        (max_val, 1.0),
        textcoords="offset points",
        xytext=(12, -18),
        fontsize=annotation_size,
        ha="left",
        va="top",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3),
    )

    # Format functions
    def fmt_ratio_pct(r):
        if r >= 1:
            return f"{r * 100:.0f}%"
        elif r >= 0.1:
            return f"{r * 100:.0f}%"
        elif r >= 0.01:
            return f"{r * 100:.1f}%"
        elif r >= 0.001:
            return f"{r * 100:.2f}%"
        elif r > 0:
            return "<0.01%"
        else:
            return "0%"

    def fmt_pct(v):
        if v >= 100:
            return f"{v:.0f}%"
        elif v >= 10:
            return f"{v:.1f}%"
        elif v >= 1:
            return f"{v:.2f}%"
        else:
            return f"{v:.3f}%"

    # Select formatter based on metric
    if metric in ("memory", "gpu_memory"):
        unit_fmt = fmt_ratio_pct
    elif metric in ("peak", "gpu_peak"):
        unit_fmt = fmt_pct
    else:
        unit_fmt = fmt_pct

    # Legend box with values (right-aligned, monospace)
    formatted = {p: unit_fmt(stats[p]) for p in ["P50", "P95", "P99"]}
    formatted["Max"] = unit_fmt(max_val)
    max_val_len = max(len(v) for v in formatted.values())
    legend_lines = [
        f"{p:>3}  {formatted[p]:>{max_val_len}}" for p in ["P50", "P95", "P99", "Max"]
    ]
    legend_text = "\n".join(legend_lines)
    props = dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray")
    ax.text(
        0.98,
        0.02,
        legend_text,
        transform=ax.transAxes,
        fontsize=legend_fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    # N count in top left
    if show_sample_size:
        textstr = f"N={n:,}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=tick_size,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.95, edgecolor="lightgray"
            ),
        )

    # Axes config
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel("Cumulative Proportion", fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_size)

    if metric in ("memory", "gpu_memory"):
        # Extend x-axis beyond 100% if data exceeds it
        x_max = max(1.05, max_val * 1.05)
        ax.set_xlim(-0.05, x_max)
        # Dynamic ticks based on range
        if x_max <= 1.05:
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        else:
            # Generate ticks up to max value
            tick_step = 0.5 if x_max <= 2.5 else 1.0
            ticks = np.arange(0, x_max + tick_step, tick_step)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t * 100)}%" for t in ticks])
    elif metric in ("peak", "gpu_peak"):
        # Extend x-axis beyond 100% if data exceeds it
        x_max = max(105, max_val * 1.05)
        ax.set_xlim(0, x_max)
        # Dynamic ticks based on range
        if x_max <= 105:
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        else:
            # Generate ticks up to max value
            tick_step = 50 if x_max <= 250 else 100
            ticks = np.arange(0, x_max + tick_step, tick_step)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t)}%" for t in ticks])


def render_combined_6panel(
    fig,
    axes,
    p1: Optional[Plot1Data],
    p2: Optional[Plot2Data],
    p3: Optional[Plot3Data],
    p4: Optional[Plot4Data],
    p5: Optional[Plot5Data],
    p6: Optional[Plot6Data],
    large_fonts: bool = True,
    notebook_name: str = "",
) -> None:
    """Render all 6 panels in a 2x3 grid.

    Layout:
    - Row 1: Timing (P1) | Checkpoint Time by Variable (P2)
    - Row 2: Memory Overhead (P3) | Checkpoint Memory by Variable (P4)
    - Row 3: Overhead per Cell (P5) | Checkpoint Ratio CDF (P6)

    Args:
        fig: Matplotlib figure
        axes: 6-element list of axes in order [P1, P2, P3, P4, P5, P6]
        p1-p6: Plot data (can be None for missing data)
        large_fonts: Use larger fonts
        notebook_name: Optional notebook name for figure suptitle
    """
    import matplotlib.pyplot as plt

    title_size = 20 if large_fonts else 14

    # Add notebook name as figure suptitle
    if notebook_name:
        fig.suptitle(
            notebook_name,
            fontsize=22 if large_fonts else 16,
            fontweight="bold",
            y=0.995,
        )

    # Panel 1: Timing
    if p1 is not None:
        render_plot1(axes[0], p1, large_fonts=large_fonts)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No timing data",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("Execution Time", fontsize=title_size)

    # Panel 2: Checkpoint Time by Variable
    if p2 is not None:
        render_plot2(axes[1], p2, large_fonts=large_fonts)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No checkpoint timing data",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Checkpoint Time by Variable", fontsize=title_size)

    # Panel 3: Memory Overhead
    if p3 is not None:
        render_plot3(axes[2], p3, large_fonts=large_fonts)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No memory data",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Memory Overhead", fontsize=title_size)

    # Panel 4: Checkpoint Memory by Variable
    if p4 is not None:
        render_plot4(axes[3], p4, large_fonts=large_fonts)
    else:
        axes[3].text(
            0.5,
            0.5,
            "No checkpoint memory data",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
        axes[3].set_title("Checkpoint Memory by Variable", fontsize=title_size)

    # Panel 5: Overhead per Cell
    if p5 is not None:
        render_plot5(axes[4], p5, large_fonts=large_fonts)
    else:
        axes[4].text(
            0.5,
            0.5,
            "No overhead timing data",
            ha="center",
            va="center",
            transform=axes[4].transAxes,
        )
        axes[4].set_title("Overhead Time per Cell", fontsize=title_size)

    # Panel 6: Checkpoint Ratio CDF
    if p6 is not None:
        render_plot6(axes[5], p6, large_fonts=large_fonts)
    else:
        axes[5].text(
            0.5,
            0.5,
            "No ratio data",
            ha="center",
            va="center",
            transform=axes[5].transAxes,
        )
        axes[5].set_title("Checkpoint Overhead Ratio", fontsize=title_size)

    # Adjust layout to make room for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.97] if notebook_name else None)
