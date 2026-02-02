"""
Common table rendering functions for optimization statistics display.

These functions are used by both optimize_cli.py (to display results after optimization)
and stats_cli.py (to display results from notebook metadata).
"""

from typing import Optional

from flowbook.cli.optimization_metadata import (
    FlowbookOptimizationMetadata,
    LLMCostSummary,
    OptimizationResultsSummary,
    SplitResultsSummary,
)


def render_split_results_table(split_results: SplitResultsSummary) -> None:
    """
    Display split preprocessing results table.

    Args:
        split_results: Split results summary to display
    """
    print(f"\n{'='*70}")
    print("## Split Preprocessing Results")
    print(f"{'='*70}")
    print(f"Cells analyzed:  {split_results.cells_analyzed}")
    print(f"Cells split:     {split_results.cells_split}")
    print(f"New cells added: {split_results.total_new_cells}")
    print(f"LLM Cost:        ${split_results.llm_cost:.4f}")
    print(f"Time:            {split_results.time:.2f}s")
    print(f"{'='*70}")


def render_optimization_results_table(opt_results: OptimizationResultsSummary) -> None:
    """
    Display optimization results table with all cells.

    Args:
        opt_results: Optimization results summary to display
    """
    if not opt_results.cells:
        return

    print(f"\n{'='*100}")
    print("## Optimization Results\n")
    # Column widths: Cell ID=19, Initial=13, Potential=11, Optimized=15, Speedup=9, Status=16
    print("| Cell ID             | Initial (s) | Potential | Optimized (s) | Speedup | Status           |")
    print("|---------------------|-------------|-----------|---------------|---------|------------------|")

    # Status display mapping
    status_map = {
        'optimized': '✓ Optimized',
        'error': '✗ Error',
        'no improvement': '✗ No improvement',
        'not attempted': '- Not optimized'
    }

    # Display each cell
    for cell_result in opt_results.cells:
        # Truncate cell ID to 17 chars + '..' if needed
        cell_id = cell_result.cell_id[:17] + '..' if len(cell_result.cell_id) > 19 else cell_result.cell_id

        # Format potential as "X/5" or "-"
        potential_str = f"{cell_result.potential}/5" if cell_result.potential is not None else "-"

        # Get display status
        status = status_map.get(cell_result.status, '- Unknown')

        print(
            f"| {cell_id:<19} | {cell_result.initial_time:>11.2f} | {potential_str:>9} | "
            f"{cell_result.final_time:>13.2f} | {cell_result.speedup:>6.2f}x | {status:<16} |"
        )

    # Display totals
    print("|---------------------|-------------|-----------|---------------|---------|------------------|")
    print(
        f"| {'TOTAL':<19} | {opt_results.total_initial_time:>11.2f} | {'':<9} | "
        f"{opt_results.total_final_time:>13.2f} | {opt_results.overall_speedup:>6.2f}x | {'':<16} |"
    )

    # Display summary
    print(f"\n**Time saved:** {opt_results.time_saved:.2f}s ({opt_results.time_saved_percent:.1f}%)")
    print(f"{'='*100}")


def render_llm_cost_summary_table(llm_costs: LLMCostSummary) -> None:
    """
    Display LLM cost summary table.

    Args:
        llm_costs: LLM cost summary to display
    """
    print(f"\n{'='*70}")
    print("## LLM Cost Summary")
    print(f"{'='*70}")

    # Show breakdown if split was performed
    if llm_costs.split_cost is not None and llm_costs.split_time is not None:
        print(f"Split Cost:        ${llm_costs.split_cost:.4f}")
        print(f"Optimization Cost: ${llm_costs.optimization_cost:.4f}")
        print(f"Total Cost:        ${llm_costs.total_cost:.4f}")
        print(f"")
        print(f"Split Time:        {llm_costs.split_time:.2f}s")
        print(f"Optimization Time: {llm_costs.optimization_time:.2f}s")
        print(f"Total Time:        {llm_costs.total_time:.2f}s")
    else:
        # Only optimization (no split)
        print(f"Total Cost:  ${llm_costs.total_cost:.4f}")
        print(f"Total Time:  {llm_costs.total_time:.2f}s")

    print(f"{'='*70}")


def display_all_stats(metadata: FlowbookOptimizationMetadata) -> None:
    """
    Display all statistics tables from optimization metadata.

    Args:
        metadata: Complete optimization metadata
    """
    # Display split results if available
    if metadata.split_results:
        render_split_results_table(metadata.split_results)

    # Display optimization results
    render_optimization_results_table(metadata.optimization_results)

    # Display LLM cost summary
    render_llm_cost_summary_table(metadata.llm_costs)

    # Display metadata info
    print(f"\n{'='*70}")
    print("## Optimization Info")
    print(f"{'='*70}")
    print(f"Timestamp:   {metadata.timestamp}")
    print(f"Model:       {metadata.model}")
    print(f"Fast Model:  {metadata.fast_model}")
    print(f"{'='*70}")
