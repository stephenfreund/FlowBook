"""
Optimize cells command implementation.
"""

import asyncio
import copy
from typing import Any, Dict, Optional, List, Tuple

from data_ferret.server.base import NotebookCommand
from data_ferret.util.notebook_tools import NotebookTools
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.ferret_metadata import (
    FerretMetadata,
    OptimizationPotential,
    OptimizationStep,
    CodeSnippet,
    OptimizedCodeResponse,
    OptimizedCodeMetadata,
    OptimizationAppliedMetadata,
    set_optimized_ferret_metadata,
    set_optimization_applied_ferret_metadata,
)
from data_ferret.agent.agent import FerretAgent, FerretStats
from data_ferret.util.prompts import get_prompt

import nbformat
from pydantic import BaseModel, Field
import ast


def extract_function(source_code: str, function_name: str) -> Optional[str]:
    """Extract a function definition from source code.

    Args:
        source_code: The full source code
        function_name: The name of the function to extract

    Returns:
        The function source code (including decorators), or None if not found
    """
    try:
        tree = ast.parse(source_code)
        lines = source_code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the start line (including decorators if present)
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                else:
                    start_line = node.lineno - 1

                end_line = node.end_lineno
                return "\n".join(lines[start_line:end_line])
        return None
    except Exception:
        return None


def replace_function(source_code: str, function_name: str, new_function_code: str) -> str:
    """Replace a function definition in source code.

    Args:
        source_code: The full source code
        function_name: The name of the function to replace
        new_function_code: The new function source code

    Returns:
        The modified source code
    """
    try:
        tree = ast.parse(source_code)
        lines = source_code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get the start line (including decorators if present)
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                else:
                    start_line = node.lineno - 1

                end_line = node.end_lineno

                # Build new source - split new_function_code into lines
                before = lines[:start_line]
                after = lines[end_line:]
                new_function_lines = new_function_code.split("\n")
                new_lines = before + new_function_lines + after

                return "\n".join(new_lines)

        # Function not found, return original
        return source_code
    except Exception:
        # On error, return original
        return source_code


class OptimizationResultAndStats(BaseModel):
    original_code: List[CodeSnippet] = Field(
        description="Original code snippets that will be replaced"
    )
    optimized_code: List[CodeSnippet] = Field(
        description="Optimized code snippets"
    )
    stats: FerretStats


class OptimizeCommand(NotebookCommand):
    """Optimizes cells in the notebook based on their optimization plans."""

    @property
    def command_name(self) -> str:
        return "optimize"

    @property
    def display_name(self) -> str:
        return "Optimize Cells"

    @property
    def icon_name(self) -> str:
        return "ui-components:flash"

    @property
    def tooltip(self) -> str:
        return "Optimize cells based on inspection metadata"

    async def optimize_cell(
        self, index: int, nb: nbformat.NotebookNode, model: Any
    ) -> Tuple[str, OptimizationResultAndStats]:
        """Optimize a single cell based on its optimization plan.

        This method iterates over the optimization steps for a cell and uses
        the LLM to generate optimized code for each step.

        Args:
            index: Index of the cell in the notebook
            nb: The notebook containing the cell
            model: The LLM model to use for optimization

        Returns:
            Tuple of (cell_id, OptimizationResultAndStats)
        """
        cells = nb["cells"]
        cell = cells[index]

        # Get optimization plan from cell metadata
        ferret_metadata = FerretMetadata.from_cell(cell)
        optimization_potential = ferret_metadata.get_optimization_potential()

        if not optimization_potential or not optimization_potential.optimization_plan:
            # No optimization plan, return empty results
            # Create a minimal FerretStats with no usage
            from agents import Usage
            empty_stats = FerretStats(
                model=model,
                time=0.0,
                usage=Usage(input_tokens=0, output_tokens=0),
                log_path=""
            )
            return cell["id"], OptimizationResultAndStats(
                original_code=[],
                optimized_code=[],
                stats=empty_stats
            )

        from agents import Usage

        original_snippets = []
        optimized_snippets = []
        total_cost = 0.0
        total_time = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        all_log_paths = []

        with NotebookTools(nb) as tools:
            # Iterate over each optimization step (non-concurrent)
            for step in optimization_potential.optimization_plan:
                # Find the target cell
                target_cell = None
                target_index = None
                for idx, c in enumerate(cells):
                    if c["id"] == step.target_cell_id:
                        target_cell = c
                        target_index = idx
                        break

                if target_cell is None:
                    print(f"Warning: Target cell {step.target_cell_id} not found")
                    continue

                # Extract profile metadata to get environment information
                target_ferret_metadata = FerretMetadata.from_cell(target_cell)
                target_profile = target_ferret_metadata.get_profile()
                env_data = target_profile.env if target_profile else None

                # Build context prefix (all cells before the target cell)
                prefix = "\n\n".join(
                    [f'# Cell {c["id"]}\n{c["source"]}' for c in cells[:target_index]]
                )

                # Get the source code to optimize and store original
                if step.function_name:
                    # Extract the specific function from the cell
                    original_function_code = extract_function(target_cell["source"], step.function_name)
                    if original_function_code:
                        cell_source = original_function_code
                        optimization_context = f"Optimize only the function '{step.function_name}'. Return only the complete function definition, nothing else."
                    else:
                        # Couldn't extract function, optimize whole cell
                        print(f"Warning: Could not extract function '{step.function_name}' from cell {step.target_cell_id}")
                        cell_source = target_cell["source"]
                        optimization_context = f"Focus on optimizing the function '{step.function_name}'"
                        original_function_code = target_cell["source"]
                else:
                    # Optimize the entire cell
                    cell_source = target_cell["source"]
                    original_function_code = target_cell["source"]
                    optimization_context = "Optimize the entire cell code. Return only the optimized code, nothing else."

                # Store original code snippet
                original_snippets.append(
                    CodeSnippet(
                        cell_id=step.target_cell_id,
                        function_name=step.function_name,
                        source=original_function_code
                    )
                )

                # Build the optimization descriptions
                if isinstance(step.description, list):
                    optimization_descriptions = "Optimizations to apply:\n" + "\n".join(f"- {desc}" for desc in step.description)
                else:
                    optimization_descriptions = f"Optimizations to apply:\n- {step.description}"

                # Format environment information from profile metadata
                if env_data:
                    env_lines = [f"  {var}: {type_}" for var, type_ in env_data.items()]
                    env_section = "Available variables in the environment (from profiling):\n" + "\n".join(env_lines)
                else:
                    env_section = ""

                # Create agent with structured output
                agent = FerretAgent[OptimizedCodeResponse](
                    key="cell_optimization",
                    model=model,
                    instructions=get_prompt("optimization_instructions"),
                    output_type=OptimizedCodeResponse,
                    tools=tools.tools(include_profile=True),
                )

                input_text = get_prompt(
                    "optimization_input",
                    prefix=prefix,
                    kind="cell" if step.function_name is None else "function",
                    cell_source=cell_source,
                    env_section=env_section,
                    optimization_descriptions=optimization_descriptions,
                )

                # Run the optimization
                optimization_response, stats = await agent.run(input_text)

                total_cost += stats.cost
                total_time += stats.time
                if stats.usage:
                    total_input_tokens += stats.usage.input_tokens
                    total_output_tokens += stats.usage.output_tokens
                all_log_paths.append(stats.log_path)

                # Store optimized code snippet with optimizations applied
                optimized_snippets.append(
                    CodeSnippet(
                        cell_id=step.target_cell_id,
                        function_name=step.function_name,
                        source=optimization_response.optimized_code.strip(),
                        optimizations_applied=optimization_response.optimizations_applied
                    )
                )

                # Print the applied optimizations
                if optimization_response.optimizations_applied:
                    print(f"\n  Optimizations applied:")
                    for opt in optimization_response.optimizations_applied:
                        print(f"    - {opt}")

                print(
                    f"| {index:<9}| {step.target_cell_id:<15}| {step.function_name or 'whole cell':<20}| {stats.usage.total_tokens if stats.usage else 0:<9}| {stats.time:<9.1f}| {stats.cost:<9.4f}|"
                )

        # Create aggregated FerretStats
        aggregated_usage = Usage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens
        )
        aggregated_stats = FerretStats(
            model=model,
            time=total_time,
            usage=aggregated_usage,
            log_path=", ".join(all_log_paths) if all_log_paths else ""
        )

        return cell["id"], OptimizationResultAndStats(
            original_code=original_snippets,
            optimized_code=optimized_snippets,
            stats=aggregated_stats
        )

    async def optimize_cells(
        self,
        nb: nbformat.NotebookNode,
        model: Any,
        cell_ids: Optional[List[str]] = None,
    ) -> Tuple[nbformat.NotebookNode, float]:
        """Optimize cells in the notebook.

        Args:
            nb: The notebook to optimize
            model: The LLM model to use
            cell_ids: Optional list of specific cell IDs to optimize

        Returns:
            Tuple of (modified_notebook, total_cost)
        """
        print()
        print("# Optimizing Cells")
        print()

        new_nb: nbformat.NotebookNode = copy.deepcopy(nb)

        # Track which cells we need to process
        cells_to_process = []
        for index, cell in enumerate(new_nb["cells"]):
            if cell["cell_type"] == "code":
                # Check if cell has optimization plan
                ferret_metadata = FerretMetadata.from_cell(cell)
                opt_potential = ferret_metadata.get_optimization_potential()

                if opt_potential and opt_potential.optimization_plan:
                    # Only process if in cell_ids list (or if no filter)
                    if cell_ids is None or cell["id"] in cell_ids:
                        cells_to_process.append(index)

        if not cells_to_process:
            print("No cells to optimize (no optimization plans found)")
            return new_nb, 0.0

        print(
            "|{:<10}|{:<16}|{:<21}|{:<10}|{:<10}|{:<10}|".format(
                "Index", "Target Cell", "Function", "Tokens", "Time (s)", "Cost ($)"
            )
        )
        print("|{:-^10}|{:-^16}|{:-^21}|{:-^10}|{:-^10}|{:-^10}|".format("", "", "", "", "", ""))

        # Process cells sequentially (non-concurrent)
        all_results = []
        for index in cells_to_process:
            cell_id, result = await self.optimize_cell(index, new_nb, model)
            all_results.append((cell_id, result))

        print()

        # Apply optimization results to the notebook
        # Create a mapping from cell_id to cell
        cell_map = {cell["id"]: cell for cell in new_nb["cells"]}

        # Group optimizations by target cell (the cell that was actually modified)
        from collections import defaultdict
        target_cell_metadata = defaultdict(lambda: {
            'original': [],
            'optimized': [],
            'optimizations': []
        })

        # Track which cells were modified for each triggering cell
        triggering_cell_modifications = defaultdict(set)

        # First pass: apply optimizations and collect metadata per target cell
        for cell_id, result_and_stats in all_results:
            for opt_snippet in result_and_stats.optimized_code:
                target_cell = cell_map.get(opt_snippet.cell_id)
                if target_cell:
                    # Find the matching original snippet
                    original_snippet = next(
                        (orig for orig in result_and_stats.original_code
                         if orig.cell_id == opt_snippet.cell_id and orig.function_name == opt_snippet.function_name),
                        None
                    )

                    # Accumulate metadata for this target cell
                    target_cell_id = opt_snippet.cell_id
                    if original_snippet:
                        target_cell_metadata[target_cell_id]['original'].append(original_snippet.source)
                        target_cell_metadata[target_cell_id]['optimized'].append(opt_snippet.source)

                    if opt_snippet.optimizations_applied:
                        target_cell_metadata[target_cell_id]['optimizations'].extend(opt_snippet.optimizations_applied)

                    # Track that this triggering cell caused modification of target_cell_id
                    triggering_cell_modifications[cell_id].add(target_cell_id)

                    # Apply the optimization to the source
                    if opt_snippet.function_name:
                        # Replace just the function in the cell
                        target_cell["source"] = replace_function(
                            target_cell["source"],
                            opt_snippet.function_name,
                            opt_snippet.source
                        )
                    else:
                        # Replace the whole cell
                        target_cell["source"] = opt_snippet.source

        # Second pass: store metadata on each modified cell
        for target_cell_id, metadata_parts in target_cell_metadata.items():
            if target_cell_id in cell_map and metadata_parts['original']:
                optimized_metadata = OptimizedCodeMetadata(
                    original_code="\n\n".join(metadata_parts['original']),
                    optimized_code="\n\n".join(metadata_parts['optimized']),
                    optimizations_applied=metadata_parts['optimizations'],
                )
                set_optimized_ferret_metadata(cell_map[target_cell_id], optimized_metadata)

        # Third pass: store list of modified cells on each triggering cell
        for triggering_cell_id, modified_cell_ids in triggering_cell_modifications.items():
            if triggering_cell_id in cell_map and modified_cell_ids:
                optimization_applied_metadata = OptimizationAppliedMetadata(
                    modified_cell_ids=list(modified_cell_ids)
                )
                set_optimization_applied_ferret_metadata(cell_map[triggering_cell_id], optimization_applied_metadata)

        total_cost = sum([result.stats.cost for _, result in all_results])

        return new_nb, total_cost

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize cells in the notebook based on their optimization plans."""
        new_nb, total_cost = await self.optimize_cells(
            notebook_content, config.model, selected_cell_ids
        )

        metadata = {
            "status": "success",
            "command": self.command_name,
            "total_cost": total_cost,
        }

        return {"notebook": new_nb, "metadata": metadata}
