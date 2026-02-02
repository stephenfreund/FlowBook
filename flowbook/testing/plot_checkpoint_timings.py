"""
Plot checkpoint timing results as side-by-side stacked cumulative line plots.

Usage:
    python -m flowbook.testing.plot_checkpoint_timings file1.csv file2.csv
    python -m flowbook.testing.plot_checkpoint_timings file1.csv file2.csv -o plot.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single(ax, csv_path: str, colors) -> None:
    """
    Plot a single stacked cumulative timing chart on the given axes.

    Args:
        ax: Matplotlib axes to plot on
        csv_path: Path to CSV file with cell_runtime_s and commit_time_s columns
        colors: Color palette to use
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Compute cumulative sums
    df["cumulative_runtime"] = df["cell_runtime_s"].cumsum()
    df["cumulative_commit"] = df["commit_time_s"].cumsum()
    df["cumulative_total"] = df["cumulative_runtime"] + df["cumulative_commit"]

    # Create cell index for x-axis
    df["cell"] = range(1, len(df) + 1)

    # Plot stacked area
    ax.fill_between(
        df["cell"],
        0,
        df["cumulative_runtime"],
        alpha=0.7,
        label="Cell Runtime",
        color=colors[0],
    )
    ax.fill_between(
        df["cell"],
        df["cumulative_runtime"],
        df["cumulative_total"],
        alpha=0.7,
        label="Checkpoint Time",
        color=colors[1],
    )

    # Plot lines on top for clarity
    ax.plot(df["cell"], df["cumulative_runtime"], color=colors[0], linewidth=1.5)
    ax.plot(df["cell"], df["cumulative_total"], color=colors[1], linewidth=1.5)

    # Labels and title
    ax.set_xlabel("Cell Number")
    ax.set_ylabel("Cumulative Time (seconds)")
    ax.set_title(os.path.basename(csv_path))
    ax.legend(loc="upper left")

    # Set x-axis to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def plot_cumulative_timings(csv_path1: str, csv_path2: str, output_path: str = None) -> None:
    """
    Create side-by-side stacked line plots of cumulative timing for two CSV files.

    Args:
        csv_path1: Path to first CSV file
        csv_path2: Path to second CSV file
        output_path: Path to save plot (if None, displays interactively)
    """
    # Set up the plot with shared axes
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    colors = sns.color_palette()

    plot_single(ax1, csv_path1, colors)
    plot_single(ax2, csv_path2, colors)

    # Add super title with two levels of directory names
    cwd = os.getcwd()
    parent = os.path.basename(os.path.dirname(cwd))
    current = os.path.basename(cwd)
    fig.suptitle(f"{parent}/{current}", fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot side-by-side cumulative checkpoint timing results"
    )
    parser.add_argument(
        "csv1",
        help="Path to first CSV file with cell_runtime_s and commit_time_s columns"
    )
    parser.add_argument(
        "csv2",
        help="Path to second CSV file with cell_runtime_s and commit_time_s columns"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output image file (default: display interactively)"
    )

    args = parser.parse_args()
    plot_cumulative_timings(args.csv1, args.csv2, args.output)


if __name__ == "__main__":
    main()
