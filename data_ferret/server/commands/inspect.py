"""
Inspect cells command implementation.
"""

import copy
import json
import random
from typing import Any, Dict, Optional

from agents import Usage

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.util.notebook_tools import NotebookTools
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.ferret_metadata import (
    FerretMetadata,
    OptimizationPotential,
    OptimizationStep,
    set_optimization_potential_ferret_metadata,
)
from data_ferret.agent.agent import FerretAgent, FerretStats
from data_ferret.util.output import log, timer
from data_ferret.util.prompts import get_prompt

from typing import List, Tuple, Dict, Any, Optional
import asyncio

import nbformat
from pydantic import BaseModel, Field


class InspectionResultAndStats(BaseModel):
    inspection_metadata: OptimizationPotential
    stats: FerretStats


class InspectCommand(NotebookCommand):
    """Adds inspection metadata to all cells in the notebook."""

    @property
    def command_name(self) -> str:
        return "inspect"

    @property
    def display_name(self) -> str:
        return "Inspect Cells"

    @property
    def icon_name(self) -> str:
        return "ui-components:search"

    @property
    def tooltip(self) -> str:
        return "Add inspection metadata to cells"

    async def inspect_cell(
        self, index: int, nb: nbformat.NotebookNode, model: Any
    ) -> Tuple[str, InspectionResultAndStats]:
        cells = nb["cells"]
        cell = cells[index]

        profile_data = FerretMetadata.from_cell(cell).profile
        if profile_data is not None and profile_data.duration < 3.0:
            final_output = OptimizationPotential(potential=0, optimization_plan=[])
            stats = FerretStats(
                model=model, usage=Usage(total_tokens=0), time=0, log_path=""
            )

        else:
            with NotebookTools(nb) as tools:

                agent = FerretAgent[OptimizationPotential](
                    key="cell_inspection",
                    model=model,
                    instructions=get_prompt("cell_inspection_instructions"),
                    output_type=OptimizationPotential,
                    tools=tools.tools(include_profile=True),
                )

                prefix = "\n".join(
                    [f'Cell {cell["id"]}:\n{cell["source"]}' for cell in cells[:index]]
                )

                profile_duration = (
                    "Total duration: {profile_data.duration:.2f} seconds"
                    if profile_data is not None
                    else ""
                )
                profile_trace = profile_data.profile if profile_data is not None else ""
                profile_env = (
                    "\n".join(
                        [f"- {key}: {value}" for key, value in profile_data.env.items()]
                    )
                    if profile_data is not None
                    else ""
                )

                input_text = get_prompt(
                    "cell_inspection_input",
                    cell_id=cell["id"],
                    prefix=prefix,
                    cell_source=cell["source"],
                    profile_duration=profile_duration,
                    profile_trace=profile_trace,
                    profile_env=profile_env,
                )

                final_output, stats = await agent.run(input_text)

                # Merge optimization steps with the same target_cell_id and function_name into a single string,
                # with each description separated by a newline, in Markdown list format.
                merged_steps: Dict[Tuple[str, Optional[str]], List[str]] = {}
                for step in final_output.optimization_plan:
                    key = (step.target_cell_id, step.function_name)
                    if key not in merged_steps:
                        merged_steps[key] = []
                    merged_steps[key].extend(step.description)

                # Create new optimization plan with merged descriptions
                new_optimization_plan = []
                for (
                    target_cell_id,
                    function_name,
                ), descriptions in merged_steps.items():
                    new_optimization_plan.append(
                        OptimizationStep(
                            target_cell_id=target_cell_id,
                            function_name=function_name,
                            description=descriptions,
                        )
                    )

                final_output.optimization_plan = new_optimization_plan

        # print(
        #     f"| {index:<9}| {final_output.potential:<9}| {stats.usage.total_tokens:<9}| {stats.time:<9.1f}| {stats.cost:<9.4f}|"
        # )
        log(
            f"Cell {index} Potential:{final_output.potential} Tokens:{stats.usage.total_tokens} Time:{stats.time:.2f} Cost:{stats.cost:.4f}"
        )

        return cell["id"], InspectionResultAndStats(
            inspection_metadata=final_output, stats=stats
        )

    async def inspect_cells(
        self,
        nb: nbformat.NotebookNode,
        model: Any,
        cell_ids: Optional[List[str]] = None,
    ) -> Tuple[nbformat.NotebookNode, float]:
        # print()
        # print("# Inspecting Cells for Optimization Potential")
        # print()

        tasks = []
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                source = "\n".join(
                    cell["source"]
                    if isinstance(cell["source"], list)
                    else [cell["source"]]
                )
                if source.strip():
                    # Skip cells that already have inspection data unless --all is specified
                    if cell_ids is None or cell["id"] in cell_ids:
                        tasks.append(self.inspect_cell(index, nb, model))
                else:
                    set_optimization_potential_ferret_metadata(
                        cell, OptimizationPotential(potential=0, optimization_plan=[])
                    )

        # print(
        #     "|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|".format(
        #         "Index", "Potential", "Tokens", "Time (s)", "Cost ($)"
        #     )
        # )
        # print("|{:-^10}|{:-^10}|{:-^10}|{:-^10}|{:-^10}|".format("", "", "", "", ""))
        results = await asyncio.gather(*tasks)
        # print()
        new_nb: nbformat.NotebookNode = nb.copy()  # type: ignore

        # Update each cell with its inspection results
        cell_map = {cell["id"]: cell for cell in new_nb["cells"]}
        for cell_id, cell_result in results:
            cell = cell_map.get(cell_id)
            assert cell is not None, f"Cell {cell_id} not found in notebook"
            set_optimization_potential_ferret_metadata(
                cell, cell_result.inspection_metadata
            )

        total_cost = sum([cell_result.stats.cost for _, cell_result in results])

        return new_nb, total_cost

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Add inspection metadata to each cell."""
        with self.timing_context() as get_elapsed:
            new_nb, total_cost = await self.inspect_cells(
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
