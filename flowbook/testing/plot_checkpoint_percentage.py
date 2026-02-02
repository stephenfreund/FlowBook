"""
Plot cumulative checkpoint overhead percentage.

Takes baseline and flowbook timing CSVs and plots the cumulative percentage
of time spent in checkpointing versus cell number.

Usage:
    python -m flowbook.testing.plot_checkpoint_percentage baseline.csv flowbook.csv
    python -m flowbook.testing.plot_checkpoint_percentage baseline.csv flowbook.csv -o output.pdf
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_merge(baseline_csv: str, flowbook_csv: str) -> pd.DataFrame:
    """Load and merge baseline and flowbook timing data."""
    df_base = pd.read_csv(baseline_csv)
    df_fb = pd.read_csv(flowbook_csv)

    # Merge by cell_id if available, otherwise positional
    if "cell_id" in df_base.columns and "cell_id" in df_fb.columns:
        df = df_base[["cell_id", "cell_runtime_s"]].merge(
            df_fb[["cell_id", "commit_time_s"]],
            on="cell_id",
            how="inner"
        )
    else:
        df = pd.DataFrame()
        df["cell_runtime_s"] = df_base["cell_runtime_s"]
        df["commit_time_s"] = df_fb["commit_time_s"]

    return df


def create_percentage_plot(
    baseline_csv: str,
    flowbook_csv: str,
    output_path: str,
) -> None:
    """Create plot of cumulative checkpoint overhead percentage."""
    df = load_and_merge(baseline_csv, flowbook_csv)

    # Calculate cumulative values
    df["cumulative_runtime"] = df["cell_runtime_s"].cumsum()
    df["cumulative_commit"] = df["commit_time_s"].cumsum()
    df["cumulative_total"] = df["cumulative_runtime"] + df["cumulative_commit"]

    # Calculate percentage of time spent in checkpointing
    df["checkpoint_pct"] = (df["cumulative_commit"] / df["cumulative_total"]) * 100
    df["cell"] = range(1, len(df) + 1)

    # Create plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df["cell"], df["checkpoint_pct"], linewidth=2, marker='o', markersize=4)
    ax.fill_between(df["cell"], 0, df["checkpoint_pct"], alpha=0.3)

    ax.set_xlabel("Cell Number")
    ax.set_ylabel("Cumulative Checkpoint Overhead (%)")
    ax.set_title("Checkpoint Time as Percentage of Total Execution Time")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    # Add annotation for final percentage
    final_pct = df["checkpoint_pct"].iloc[-1]
    final_cell = df["cell"].iloc[-1]
    ax.annotate(
        f"{final_pct:.1f}%",
        xy=(final_cell, final_pct),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
    )

    # Add super title with directory info
    cwd = os.getcwd()
    parent = os.path.basename(os.path.dirname(cwd))
    current = os.path.basename(cwd)
    fig.suptitle(f"{parent}/{current}", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot cumulative checkpoint overhead percentage"
    )
    parser.add_argument(
        "baseline_csv",
        help="Path to baseline timings CSV"
    )
    parser.add_argument(
        "flowbook_csv",
        help="Path to flowbook timings CSV"
    )
    parser.add_argument(
        "-o", "--output",
        default="checkpoint_percentage.pdf",
        help="Output plot file (default: checkpoint_percentage.pdf)"
    )

    args = parser.parse_args()
    create_percentage_plot(args.baseline_csv, args.flowbook_csv, args.output)


if __name__ == "__main__":
    main()
