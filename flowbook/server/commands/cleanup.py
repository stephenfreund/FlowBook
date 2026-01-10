"""
Cleanup command implementation.
"""

import copy
import json
from typing import Any, Dict, Optional, List, Tuple
import asyncio

import nbformat
from pydantic import BaseModel, Field

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.agent.agent import FlowbookAgent, FlowbookStats
from flowbook.util.prompts import get_prompt
from flowbook.util.output import log


class CleanupResult(BaseModel):
    """Result from cleaning up a code cell."""
    description: str = Field(description="Markdown description of improvements made")
    improved_code: str = Field(description="The improved/cleaned up code")


class CleanupResultAndStats(BaseModel):
    cleanup_result: CleanupResult
    stats: FlowbookStats


class CleanupCommand(NotebookCommand):
    """Uses LLM to generate improved code with suggestions."""

    @property
    def command_name(self) -> str:
        return "cleanup"

    @property
    def display_name(self) -> str:
        return "Cleanup"

    @property
    def icon_name(self) -> str:
        return "ui-components:clean"

    @property
    def tooltip(self) -> str:
        return "Generate improved code with AI suggestions"

    @property
    def requires_kernel(self) -> bool:
        return False

    async def cleanup_cell(
        self, index: int, cells: List[nbformat.NotebookNode], model: Any
    ) -> Tuple[int, CleanupResultAndStats]:
        """Clean up a single cell using the LLM."""
        cell = cells[index]
        agent = FlowbookAgent[CleanupResult](
            key="cell_cleanup",
            model=model,
            instructions=get_prompt("cleanup_instructions"),
            output_type=CleanupResult,
        )

        prefix = "\n".join([cell["source"] for cell in cells[:index]])
        profile_data = json.dumps(
            cell.get("metadata", {}).get("flowbook", {}).get("profile", {}), indent=2
        )

        input_text = get_prompt(
            "cleanup_input",
            prefix=prefix,
            cell_source=cell["source"],
            profile_data=profile_data,
        )

        final_output, stats = await agent.run(input_text)

        log(
            f"Cleaned up cell {index} | Tokens: {stats.usage.total_tokens} | Cost: ${stats.cost:.4f}"
        )

        return index, CleanupResultAndStats(cleanup_result=final_output, stats=stats)

    async def cleanup_cells(
        self,
        nb: nbformat.NotebookNode,
        model: Any,
        selected_cell_ids: Optional[List[str]] = None,
    ) -> Tuple[nbformat.NotebookNode, float]:
        """Clean up code cells in the notebook."""
        log("Cleaning up code cells with AI suggestions")

        tasks = []
        cell_indices_to_cleanup = []

        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                if cell["source"].strip():
                    if selected_cell_ids is None or cell["id"] in selected_cell_ids:
                        tasks.append(self.cleanup_cell(index, nb["cells"], model))
                        cell_indices_to_cleanup.append(index)

        if not tasks:
            log("No cells to clean up")
            return nb, 0.0

        results = await asyncio.gather(*tasks)

        # Create new notebook with cleaned cells inserted
        new_nb = nbformat.v4.new_notebook()
        new_nb["metadata"] = nb.get("metadata", {})
        new_cells = []

        result_map = {index: result for index, result in results}

        for index, cell in enumerate(nb["cells"]):
            # Add original cell
            new_cells.append(cell)

            # If this cell was cleaned up, add the markdown description and improved code
            if index in result_map:
                cleanup_result = result_map[index].cleanup_result

                # Create markdown cell with description
                markdown_cell = nbformat.v4.new_markdown_cell(
                    source=f"## Code Improvements\n\n{cleanup_result.description}"
                )
                new_cells.append(markdown_cell)

                # Create code cell with improved code
                code_cell = nbformat.v4.new_code_cell(source=cleanup_result.improved_code)
                new_cells.append(code_cell)

        new_nb["cells"] = new_cells

        total_cost = sum([result.stats.cost for _, result in results])
        log(f"Total cleanup cost: ${total_cost:.4f}")

        return new_nb, total_cost

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Process the cleanup command."""
        with self.timing_context() as get_elapsed:
            new_nb, total_cost = await self.cleanup_cells(
                notebook_content, config.model, selected_cell_ids
            )

            metadata = {
                "status": "success",
                "command": self.command_name,
                "total_cost": total_cost,
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_nb,
            metadata=metadata,
            total_cost=total_cost,
            total_time=total_time
        )
