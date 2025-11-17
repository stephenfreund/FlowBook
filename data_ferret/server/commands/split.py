"""
Split command for DataFerret.

This command analyzes code cells and splits them into logical, self-contained steps
using LLM analysis to improve notebook readability and maintainability.

Example usage:
    From JupyterLab: Use the "Split Cells" button in the toolbar
    From CLI: data_ferret_split notebook.ipynb -o notebook_split.ipynb
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from data_ferret.agent.agent import FerretAgent, FerretStats
from data_ferret.agent.llm_cost import Usage
from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.output import error, indent, log, timer
from data_ferret.util.prompts import get_prompt


# ============================================================================
# Pydantic Models for LLM Responses
# ============================================================================


class SplitCellInfo(BaseModel):
    """Information about a single split cell."""

    cell_type: str = Field(description="Cell type: 'code' or 'markdown'")
    description: str = Field(description="Brief description of what this cell does")
    source: str = Field(description="The source code or markdown for this cell")


class SplitCellResponse(BaseModel):
    """Response from LLM for splitting a single cell."""

    pesky_variables: List[str] = Field(
        description="List of variables that contain streams, iterators, or matplotlib figures/axes"
    )
    explanation: str = Field(
        description="Explanation of how the cell was split and why"
    )
    should_split: bool = Field(
        description="Whether the cell should actually be split (False if already well-structured)"
    )
    split_cells: List[SplitCellInfo] = Field(
        description="List of cells to replace the original cell with"
    )


# ============================================================================
# Helper Classes
# ============================================================================


class StatsAggregator:
    """Aggregates statistics from multiple LLM calls."""

    def __init__(self):
        self.all_stats: List[FerretStats] = []

    def add_stats(self, stats: FerretStats) -> None:
        """Add stats from one LLM call."""
        self.all_stats.append(stats)

    def get_aggregated_stats(self) -> FerretStats:
        """Get aggregated stats across all calls."""
        if not self.all_stats:
            # Create a minimal stats object
            stats = FerretStats(
                model="unknown",
                time=0.0,
                usage=Usage(input_tokens=0, output_tokens=0),
                log_path=None,
            )
            return stats

        total_time = sum(s.time for s in self.all_stats)
        total_input = sum(s.usage.input_tokens for s in self.all_stats)
        total_output = sum(s.usage.output_tokens for s in self.all_stats)

        # Create aggregated stats (cost will be calculated automatically)
        return FerretStats(
            model=self.all_stats[0].model,
            time=total_time,
            usage=Usage(input_tokens=total_input, output_tokens=total_output),
            log_path=None,
        )


class ContextBuilder:
    """Builds context from previous cells for LLM."""

    MAX_CONTEXT_CELLS = 3  # Maximum number of previous cells to include
    MAX_CELL_LINES = 10  # Maximum lines per cell in context

    @staticmethod
    def build_context(cells: List[Dict[str, Any]], current_index: int) -> str:
        """
        Build context string from previous cells.

        Args:
            cells: All cells in the notebook
            current_index: Index of current cell being processed

        Returns:
            Context string summarizing previous cells
        """
        if current_index == 0:
            return "This is the first cell in the notebook."

        # Collect previous code cells
        prev_cells = []
        start_index = max(0, current_index - 5)

        for i in range(start_index, current_index):
            cell = cells[i]
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                # Truncate long cells
                lines = source.split("\n")
                if len(lines) > ContextBuilder.MAX_CELL_LINES:
                    source = (
                        "\n".join(lines[: ContextBuilder.MAX_CELL_LINES])
                        + "\n... (truncated)"
                    )
                prev_cells.append(f"Cell {i}:\n{source}")

        if not prev_cells:
            return "No previous code cells."

        # Return last N cells
        return "\n\n".join(prev_cells[-ContextBuilder.MAX_CONTEXT_CELLS :])


class SplitCellResult(BaseModel):
    """Result of splitting a single cell."""

    cell_id: str
    cell_index: int
    response: SplitCellResponse
    stats: FerretStats


# ============================================================================
# Main Command Class
# ============================================================================


class SplitCommand(NotebookCommand):
    """Split code cells into logical, self-contained steps using LLM analysis."""

    @property
    def command_name(self) -> str:
        return "split"

    @property
    def display_name(self) -> str:
        return "Split Cells"

    @property
    def icon_name(self) -> str:
        return "ui-components:cut"

    @property
    def tooltip(self) -> str:
        return "Split code cells into logical steps for better readability"

    @property
    def requires_kernel(self) -> bool:
        return False  # Static analysis only

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main entry point for splitting cells.

        Args:
            notebook_content: The notebook to process
            kernel_client: Optional kernel client (not used for split)
            selected_cell_ids: Optional list of cell IDs to process (if None, process all)
            config: Optional configuration object
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'notebook' (modified notebook) and 'metadata' (results)
        """
        with timer(key="split_cells_total", message="Splitting cells"):
            log("Starting cell splitting process")

            # Get configuration
            model = (
                config.model
                if config and hasattr(config, 'model')
                else "claude-3-5-sonnet-20241022"
            )
            log(f"Using model: {model}")

            # Initialize stats aggregator
            stats_agg = StatsAggregator()

            # Process cells
            try:
                new_notebook, split_results = await self._split_cells(
                    notebook_content=notebook_content,
                    selected_cell_ids=selected_cell_ids,
                    model=model,
                    stats_agg=stats_agg,
                )
            except Exception as e:
                error(f"Error during cell splitting: {e}")
                return {
                    "notebook": notebook_content,
                    "metadata": {
                        "status": "error",
                        "command": self.command_name,
                        "error": str(e),
                    },
                }

            # Aggregate final stats
            total_stats = stats_agg.get_aggregated_stats()

            log("\nSplit complete:")
            log(f"  - Cells analyzed: {split_results['cells_analyzed']}")
            log(f"  - Cells split: {split_results['cells_split']}")
            log(f"  - Total new cells: {split_results['total_new_cells']}")
            log(f"  - LLM cost: ${total_stats.cost:.4f}")
            log(f"  - Total time: {total_stats.time:.2f}s")

            metadata = {
                "status": "success",
                "command": self.command_name,
                "cells_analyzed": split_results["cells_analyzed"],
                "cells_split": split_results["cells_split"],
                "total_new_cells": split_results["total_new_cells"],
                "llm_stats": {
                    "model": total_stats.model,
                    "cost": total_stats.cost,
                    "time": total_stats.time,
                    "input_tokens": total_stats.usage.input_tokens,
                    "output_tokens": total_stats.usage.output_tokens,
                },
            }

            return {"notebook": new_notebook, "metadata": metadata}

    async def split_cell(
        self,
        cell: Dict[str, Any],
        cell_index: int,
        cells: List[Dict[str, Any]],
        model: str,
    ) -> SplitCellResult:
        """
        Split a single cell using LLM.

        Args:
            cell: The notebook cell to split
            cell_index: Index of the cell in the notebook
            cells: All cells in the notebook (for context)
            model: LLM model to use

        Returns:
            SplitCellResult with response and stats
        """
        with timer(
            key=f"split_cell_{cell_index}", message=f"Analyzing cell {cell_index}"
        ):
            cell_id = cell.get("id", f"cell_{cell_index}")
            cell_source = "".join(cell.get("source", []))

            # Build context from previous cells
            context = ContextBuilder.build_context(cells, cell_index)

            # Create agent for this cell
            agent = FerretAgent[SplitCellResponse](
                key="cell_splitting",
                model=model,
                instructions=get_prompt("split_instructions"),
                output_type=SplitCellResponse,
            )

            # Build input for LLM
            input_text = get_prompt(
                "split_input",
                cell_index=cell_index,
                cell_id=cell_id,
                cell_source=cell_source,
                context=context,
            )

            # Call LLM
            log(f"  Analyzing cell {cell_index} ({len(cell_source)} chars)")
            response, stats = await agent.run(input_text)

            log(f"  → Should split: {response.should_split}")
            if response.should_split:
                log(f"  → Splitting into {len(response.split_cells)} cells")

            return SplitCellResult(
                cell_id=cell_id,
                cell_index=cell_index,
                response=response,
                stats=stats,
            )

    async def _split_cells(
        self,
        notebook_content: Dict[str, Any],
        selected_cell_ids: Optional[List[str]],
        model: str,
        stats_agg: StatsAggregator,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process all cells and perform splitting.

        Args:
            notebook_content: The notebook to process
            selected_cell_ids: Optional list of cell IDs to process
            model: LLM model to use
            stats_agg: Stats aggregator to track LLM usage

        Returns:
            Tuple of (modified_notebook, results_dict)
        """
        new_notebook = copy.deepcopy(notebook_content)
        cells = new_notebook.get("cells", [])

        # Collect tasks for cells to process
        tasks = []
        cells_to_process = []

        with indent(message="Processing cells"):
            for i, cell in enumerate(cells):
                # Skip non-code cells
                if cell.get("cell_type") != "code":
                    continue

                # Skip if not in selected cells (when specified)
                cell_id = cell.get("id")
                if selected_cell_ids and cell_id not in selected_cell_ids:
                    continue

                # Add task for this cell
                tasks.append(self.split_cell(cell, i, cells, model))
                cells_to_process.append((i, cell))

            # Run all tasks concurrently using gather
            if not tasks:
                return new_notebook, {
                    "cells_analyzed": 0,
                    "cells_split": 0,
                    "total_new_cells": 0,
                }

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and build new cell list
            cells_analyzed = len(results)
            cells_split = 0
            total_new_cells = 0

            # Create a map of cell_id -> split result
            split_results_map = {}
            for result in results:
                if isinstance(result, Exception):
                    error(f"  Error splitting cell: {result}")
                    continue

                # Add stats
                stats_agg.add_stats(result.stats)

                # Store result
                split_results_map[result.cell_id] = result

                # Count splits
                if result.response.should_split:
                    cells_split += 1
                    total_new_cells += len(result.response.split_cells)

            # Build new cell list with split cells
            new_cells_list = []
            for i, cell in enumerate(cells):
                cell_id = cell.get("id")

                # Check if this cell was split
                if cell_id in split_results_map:
                    split_result = split_results_map[cell_id]

                    if split_result.response.should_split:
                        # Replace with split cells
                        for split_cell_info in split_result.response.split_cells:
                            new_cell = self._create_cell_from_split_info(
                                split_cell_info, cell_id
                            )
                            new_cells_list.append(new_cell)

                        log(
                            f"  Split cell {i} into {len(split_result.response.split_cells)} cells"
                        )
                    else:
                        # Keep original cell
                        new_cells_list.append(cell)
                else:
                    # Cell was not processed, keep as is
                    new_cells_list.append(cell)

        # Update notebook with new cells
        new_notebook["cells"] = new_cells_list

        results_summary = {
            "cells_analyzed": cells_analyzed,
            "cells_split": cells_split,
            "total_new_cells": total_new_cells,
        }

        return new_notebook, results_summary

    def _create_cell_from_split_info(
        self, split_cell_info: SplitCellInfo, original_cell_id: str
    ) -> Dict[str, Any]:
        """
        Create a notebook cell from SplitCellInfo.

        Args:
            split_cell_info: Information about the split cell
            original_cell_id: ID of the original cell that was split

        Returns:
            Notebook cell dictionary
        """
        # Generate a unique ID for the new cell
        cell_id = str(uuid.uuid4())

        new_cell = {
            "id": cell_id,
            "cell_type": split_cell_info.cell_type,
            "metadata": {
                "split_from": original_cell_id,
                "split_description": split_cell_info.description,
            },
            "source": split_cell_info.source,
        }

        # Add execution_count and outputs for code cells
        if split_cell_info.cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []

        return new_cell
