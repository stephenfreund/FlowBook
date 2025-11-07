"""
Optimize cells command implementation.
"""

import asyncio
import copy
from typing import Any, Dict, Optional, List, Tuple, Set

from data_ferret.server.base import NotebookCommand
from data_ferret.util.notebook_tools import NotebookTools
from data_ferret.server.kernel_manager import FerretKernelClient, TestCodeData
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
from data_ferret.util.dependencies import analyze_notebook, CellDependencies
from data_ferret.kernel.types import DiffResult, format_diff_as_markdown
from data_ferret.util.output import log, error, timer

import nbformat
from pydantic import BaseModel, Field
import ast


class ASTHelper:
    """Helper class for AST-based code manipulation."""

    @staticmethod
    def _find_function_node(
        source_code: str, function_name: str
    ) -> Optional[Tuple[ast.FunctionDef, List[str]]]:
        """Find a function node and return it with source lines.

        Returns:
            Tuple of (function_node, source_lines) or None if not found
        """
        try:
            tree = ast.parse(source_code)
            lines = source_code.splitlines()

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return node, lines
            return None
        except Exception:
            return None

    @staticmethod
    def _get_function_line_range(node: ast.FunctionDef) -> Tuple[int, int]:
        """Get the start and end line numbers for a function (including decorators).

        Returns:
            Tuple of (start_line, end_line)
        """
        start_line = (
            node.decorator_list[0].lineno - 1
            if node.decorator_list
            else node.lineno - 1
        )
        end_line = node.end_lineno
        return start_line, end_line

    @classmethod
    def extract_function(cls, source_code: str, function_name: str) -> Optional[str]:
        """Extract a function definition from source code.

        Args:
            source_code: The full source code
            function_name: The name of the function to extract

        Returns:
            The function source code (including decorators), or None if not found
        """
        result = cls._find_function_node(source_code, function_name)
        if result is None:
            return None

        node, lines = result
        start_line, end_line = cls._get_function_line_range(node)
        return "\n".join(lines[start_line:end_line])

    @classmethod
    def replace_function(
        cls, source_code: str, function_name: str, new_function_code: str
    ) -> str:
        """Replace a function definition in source code.

        Args:
            source_code: The full source code
            function_name: The name of the function to replace
            new_function_code: The new function source code

        Returns:
            The modified source code
        """
        result = cls._find_function_node(source_code, function_name)
        if result is None:
            return source_code

        node, lines = result
        start_line, end_line = cls._get_function_line_range(node)

        # Build new source
        before = lines[:start_line]
        after = lines[end_line:]
        new_function_lines = new_function_code.split("\n")
        new_lines = before + new_function_lines + after

        return "\n".join(new_lines)


class OptimizationResultAndStats(BaseModel):
    original_code: List[CodeSnippet] = Field(
        description="Original code snippets that will be replaced"
    )
    optimized_code: List[CodeSnippet] = Field(description="Optimized code snippets")
    stats: FerretStats


class TestCodeRequest(BaseModel):
    """Request model for test_code comm message."""

    original_code: str = Field(..., description="The original cell's code")
    modified_code: str = Field(..., description="The modified cell's code")
    output_variables: List[str] = Field(
        ..., description="List of variable names to compare"
    )


class TestCodeResponse(BaseModel):
    """Response model for test_code comm message."""

    ok: bool = Field(..., description="Whether the test succeeded")
    result: Optional[DiffResult] = Field(
        None, description="The diff result if successful"
    )
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        arbitrary_types_allowed = True


class ValidationHelper:
    """Helper class for optimization validation operations."""

    # System variables to exclude from validation
    SYSTEM_VARIABLES = {
        "get_ipython",
        "In",
        "Out",
        "exit",
        "quit",
        "_",
        "__",
        "___",
        "_i",
        "_ii",
        "_iii",
        "_dh",
    }

    @classmethod
    def get_modified_globals_for_cell(
        cls, cell_id: str, dependencies_dict: Dict[str, CellDependencies]
    ) -> Set[str]:
        """Extract global variables written by a cell, filtered for validation."""
        if cell_id not in dependencies_dict:
            return set()

        deps = dependencies_dict[cell_id]

        # Filter out private and system variables
        globals_written = {
            var
            for var in deps.globals_written
            if not var.startswith("_") and var not in cls.SYSTEM_VARIABLES
        }

        return globals_written

    @staticmethod
    def build_optimized_code_for_validation(
        original_snippets: List[CodeSnippet],
        optimized_snippets: List[CodeSnippet],
        cell_map: Dict[str, Any],
        triggering_cell_id: str,
    ) -> str:
        """
        Build complete code for validation.

        Strategy:
        1. First, include all optimized functions from OTHER cells (dependencies)
        2. Then, include the optimized version of the triggering cell
        3. This ensures dependencies are defined before the main cell runs
        """
        code_parts = []

        # Separate snippets by whether they're from the triggering cell or not
        dependency_snippets = [
            s for s in optimized_snippets if s.cell_id != triggering_cell_id
        ]
        triggering_cell_snippets = [
            s for s in optimized_snippets if s.cell_id == triggering_cell_id
        ]

        # Add optimized dependency functions first
        for snippet in dependency_snippets:
            code_parts.append(f"# Modified from cell {snippet.cell_id}")
            code_parts.append(snippet.source)
            code_parts.append("")  # blank line

        # Add optimized triggering cell code
        for snippet in triggering_cell_snippets:
            if snippet.function_name:
                code_parts.append(f"# Optimized function {snippet.function_name}")
            else:
                code_parts.append(f"# Optimized cell {snippet.cell_id}")
            code_parts.append(snippet.source)
            code_parts.append("")

        # If no snippets from triggering cell, use the current cell source
        # (case where only dependencies were optimized)
        if not triggering_cell_snippets:
            triggering_cell = cell_map.get(triggering_cell_id)
            if triggering_cell:
                code_parts.append(f"# Original cell {triggering_cell_id}")
                code_parts.append(triggering_cell["source"])

        return "\n".join(code_parts)

    @staticmethod
    def validate_optimization(
        original_code: str,
        optimized_code: str,
        modified_globals: Set[str],
        test_code_func,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that optimization preserves semantics.

        Args:
            original_code: The original code
            optimized_code: The optimized code
            modified_globals: Set of global variables to check
            test_code_func: Function to call for testing code (takes original, modified, variables)

        Returns:
            (True, None) if validation passes
            (False, error_message) if validation fails
        """
        if not modified_globals:
            # No globals to validate, automatically pass
            return (True, None)

        try:
            result = test_code_func(
                original_code=original_code,
                modified_code=optimized_code,
                output_variables=sorted(list(modified_globals)),
            )

            if result.ok and isinstance(result.result, DiffResult):
                # Check if all variables are equal
                if not result.result.differences:
                    return (True, None)
                else:
                    # Format which variables differ
                    diff_vars = list(result.result.differences.keys())
                    error_msg = f"Variables changed: {', '.join(diff_vars)}"
                    error_msg += f"\n\n{format_diff_as_markdown(result.result)}"
                    return (False, error_msg)
            else:
                # test_code failed with error
                error_msg = (
                    str(result.result)
                    if isinstance(result.result, str)
                    else "Unknown error"
                )
                return (False, error_msg)

        except Exception as e:
            return (False, f"Validation exception: {str(e)}")


class StatsAggregator:
    """Helper class for aggregating optimization statistics."""

    def __init__(self, model: Any):
        self.model = model
        self.total_cost = 0.0
        self.total_time = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.log_paths = []

    def add_stats(self, stats: FerretStats) -> None:
        """Add statistics from a single optimization step."""
        self.total_cost += stats.cost
        self.total_time += stats.time
        if stats.usage:
            self.total_input_tokens += stats.usage.input_tokens
            self.total_output_tokens += stats.usage.output_tokens
        self.log_paths.append(stats.log_path)

    def get_aggregated_stats(self) -> FerretStats:
        """Get the aggregated statistics."""
        from agents import Usage

        return FerretStats(
            model=self.model,
            time=self.total_time,
            usage=Usage(
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            ),
            log_path=", ".join(self.log_paths) if self.log_paths else "",
        )

    def get_empty_stats(self) -> FerretStats:
        """Get empty stats for when no optimization is performed."""
        from agents import Usage

        return FerretStats(
            model=self.model,
            time=0.0,
            usage=Usage(input_tokens=0, output_tokens=0),
            log_path="",
        )


class CodeExtractor:
    """Helper class for extracting and preparing code for optimization."""

    @staticmethod
    def extract_optimization_target(
        cell: Any, step: OptimizationStep
    ) -> Tuple[str, str, str]:
        """
        Extract the code to be optimized and context.

        Returns:
            Tuple of (cell_source, original_function_code, optimization_context)
        """
        if step.function_name:
            # Extract the specific function from the cell
            original_function_code = ASTHelper.extract_function(
                cell["source"], step.function_name
            )
            if original_function_code:
                return (
                    original_function_code,
                    original_function_code,
                    f"Optimize only the function '{step.function_name}'. Return only the complete function definition, nothing else.",
                )
            else:
                # Couldn't extract function, optimize whole cell
                log(
                    f"Warning: Could not extract function '{step.function_name}' from cell {step.target_cell_id}"
                )
                return (
                    cell["source"],
                    cell["source"],
                    f"Focus on optimizing the function '{step.function_name}'",
                )
        else:
            # Optimize the entire cell
            return (
                cell["source"],
                cell["source"],
                "Optimize the entire cell code. Return only the optimized code, nothing else.",
            )

    @staticmethod
    def build_context_prefix(cells: List[Any], target_index: int) -> str:
        """Build context prefix from all cells before the target cell."""
        return "\n\n".join(
            [f'# Cell {c["id"]}\n{c["source"]}' for c in cells[:target_index]]
        )

    @staticmethod
    def format_optimization_descriptions(description) -> str:
        """Format optimization descriptions into a string."""
        if isinstance(description, list):
            return "Optimizations to apply:\n" + "\n".join(
                f"- {desc}" for desc in description
            )
        else:
            return f"Optimizations to apply:\n- {description}"

    @staticmethod
    def format_environment_section(env_data: Optional[Dict[str, str]]) -> str:
        """Format environment information from profile metadata."""
        if env_data:
            env_lines = [f"  {var}: {type_}" for var, type_ in env_data.items()]
            return (
                "Available variables in the environment (from profiling):\n"
                + "\n".join(env_lines)
            )
        return ""


class MetadataManager:
    """Helper class for managing optimization metadata."""

    @staticmethod
    def apply_optimizations_to_notebook(
        cell_map: Dict[str, Any],
        all_results: List[Tuple[str, OptimizationResultAndStats]],
    ) -> None:
        """Apply optimization results to notebook cells and manage metadata."""
        from collections import defaultdict

        # Group optimizations by target cell
        target_cell_metadata = defaultdict(
            lambda: {"original": [], "optimized": [], "optimizations": []}
        )

        # Track which cells were modified for each triggering cell
        triggering_cell_modifications = defaultdict(set)

        # First pass: apply optimizations and collect metadata
        for cell_id, result_and_stats in all_results:
            for opt_snippet in result_and_stats.optimized_code:
                target_cell = cell_map.get(opt_snippet.cell_id)
                if not target_cell:
                    continue

                # Find matching original snippet
                original_snippet = next(
                    (
                        orig
                        for orig in result_and_stats.original_code
                        if orig.cell_id == opt_snippet.cell_id
                        and orig.function_name == opt_snippet.function_name
                    ),
                    None,
                )

                # Accumulate metadata
                target_cell_id = opt_snippet.cell_id
                if original_snippet:
                    target_cell_metadata[target_cell_id]["original"].append(
                        original_snippet.source
                    )
                    target_cell_metadata[target_cell_id]["optimized"].append(
                        opt_snippet.source
                    )

                if opt_snippet.optimizations_applied:
                    target_cell_metadata[target_cell_id]["optimizations"].extend(
                        opt_snippet.optimizations_applied
                    )

                # Track modifications
                triggering_cell_modifications[cell_id].add(target_cell_id)

                # Apply the optimization to the source
                if opt_snippet.function_name:
                    target_cell["source"] = ASTHelper.replace_function(
                        target_cell["source"],
                        opt_snippet.function_name,
                        opt_snippet.source,
                    )
                else:
                    target_cell["source"] = opt_snippet.source

        # Second pass: store metadata on modified cells
        for target_cell_id, metadata_parts in target_cell_metadata.items():
            if target_cell_id in cell_map and metadata_parts["original"]:
                optimized_metadata = OptimizedCodeMetadata(
                    original_code="\n\n".join(metadata_parts["original"]),
                    optimized_code="\n\n".join(metadata_parts["optimized"]),
                    optimizations_applied=metadata_parts["optimizations"],
                )
                set_optimized_ferret_metadata(
                    cell_map[target_cell_id], optimized_metadata
                )

        # Third pass: store list of modified cells on triggering cells
        for (
            triggering_cell_id,
            modified_cell_ids,
        ) in triggering_cell_modifications.items():
            if triggering_cell_id in cell_map and modified_cell_ids:
                optimization_applied_metadata = OptimizationAppliedMetadata(
                    modified_cell_ids=list(modified_cell_ids)
                )
                set_optimization_applied_ferret_metadata(
                    cell_map[triggering_cell_id], optimization_applied_metadata
                )


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

    @property
    def requires_kernel(self) -> bool:
        return True

    def _send_test_code_comm(
        self,
        kernel_client: FerretKernelClient,
        original_code: str,
        modified_code: str,
        output_variables: List[str],
    ) -> TestCodeData:
        """
        Send test_code comm message to kernel and return response.

        Uses the base class _send_comm_message method with type-safe
        Pydantic models for request and response.

        Args:
            kernel_client: The kernel client to send the message to
            original_code: The original cell's code
            modified_code: The modified (next) cell's code
            output_variables: List of variable names to compare

        Returns:
            TestCodeData with ok and result/error fields
        """
        # Create type-safe request model
        request = TestCodeRequest(
            original_code=original_code,
            modified_code=modified_code,
            output_variables=output_variables,
        )

        # Send comm and receive validated response
        response: TestCodeResponse = self._send_comm_message(
            kernel_client,
            target_name="test_code",
            request=request,
            response_type=TestCodeResponse,
        )

        # Validate response state
        if response.ok:
            if response.result is None:
                raise RuntimeError("test_code succeeded but returned no result")
            result = response.result
        else:
            if response.error is None:
                raise RuntimeError("test_code failed but returned no error message")
            result = response.error

        return TestCodeData(ok=response.ok, result=result)

    def _find_target_cell(
        self, cells: List[Any], target_cell_id: str
    ) -> Optional[Tuple[Any, int]]:
        """Find a target cell by ID.

        Returns:
            Tuple of (cell, index) or None if not found
        """
        for idx, c in enumerate(cells):
            if c["id"] == target_cell_id:
                return c, idx
        return None

    async def _process_optimization_step(
        self,
        step: OptimizationStep,
        cells: List[Any],
        index: int,
        model: Any,
        tools: NotebookTools,
        stats_aggregator: StatsAggregator,
    ) -> Tuple[CodeSnippet, CodeSnippet]:
        """Process a single optimization step.

        Returns:
            Tuple of (original_snippet, optimized_snippet)
        """
        target_desc = step.function_name or "whole cell"
        with timer(
            key="process_optimization_step",
            message=f"Processing optimization for {step.target_cell_id} ({target_desc})",
        ):
            # Find the target cell
            result = self._find_target_cell(cells, step.target_cell_id)
            if result is None:
                error(f"Target cell {step.target_cell_id} not found")
                raise ValueError(f"Target cell {step.target_cell_id} not found")

            target_cell, target_index = result

            # Extract profile metadata to get environment information
            target_ferret_metadata = FerretMetadata.from_cell(target_cell)
            target_profile = target_ferret_metadata.get_profile()
            env_data = target_profile.env if target_profile else None

            # Build context and extract code
            with timer(
                key="build_context", message="Building context and extracting code"
            ):
                prefix = CodeExtractor.build_context_prefix(cells, target_index)
                cell_source, original_function_code, optimization_context = (
                    CodeExtractor.extract_optimization_target(target_cell, step)
                )

            # Store original code snippet
            original_snippet = CodeSnippet(
                cell_id=step.target_cell_id,
                function_name=step.function_name,
                source=original_function_code,
            )

            # Build optimization descriptions and environment section
            optimization_descriptions = CodeExtractor.format_optimization_descriptions(
                step.description
            )
            env_section = CodeExtractor.format_environment_section(env_data)

            # Create agent and run optimization
            with timer(key="agent_optimization", message="Running LLM optimization"):
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

                optimization_response, stats = await agent.run(input_text)
                stats_aggregator.add_stats(stats)

            # Store optimized code snippet
            optimized_snippet = CodeSnippet(
                cell_id=step.target_cell_id,
                function_name=step.function_name,
                source=optimization_response.optimized_code.strip(),
                optimizations_applied=optimization_response.optimizations_applied,
            )

            # Log the applied optimizations
            if optimization_response.optimizations_applied:
                log("Optimizations applied:")
                for opt in optimization_response.optimizations_applied:
                    log(f"  - {opt}")

            log(
                f"| {index:<9}| {step.target_cell_id:<15}| {step.function_name or 'whole cell':<20}| {stats.usage.total_tokens if stats.usage else 0:<9}| {stats.time:<9.1f}| {stats.cost:<9.4f}|"
            )

            return original_snippet, optimized_snippet

    async def optimize_cell(
        self,
        index: int,
        nb: nbformat.NotebookNode,
        model: Any,
        kernel_client: Optional[FerretKernelClient] = None,
        dependencies_dict: Optional[Dict[str, CellDependencies]] = None,
    ) -> Tuple[str, OptimizationResultAndStats]:
        """Optimize a single cell based on its optimization plan.

        This method iterates over the optimization steps for a cell and uses
        the LLM to generate optimized code for each step.

        Args:
            index: Index of the cell in the notebook
            nb: The notebook containing the cell
            model: The LLM model to use for optimization
            kernel_client: Optional kernel client for validation
            dependencies_dict: Optional pre-computed dependencies for validation

        Returns:
            Tuple of (cell_id, OptimizationResultAndStats)
        """
        cells = nb["cells"]
        cell = cells[index]

        # Get optimization plan from cell metadata
        ferret_metadata = FerretMetadata.from_cell(cell)
        optimization_potential = ferret_metadata.get_optimization_potential()

        stats_aggregator = StatsAggregator(model)

        if not optimization_potential or not optimization_potential.optimization_plan:
            return cell["id"], OptimizationResultAndStats(
                original_code=[],
                optimized_code=[],
                stats=stats_aggregator.get_empty_stats(),
            )

        original_snippets = []
        optimized_snippets = []

        with NotebookTools(nb) as tools:
            # Iterate over each optimization step (non-concurrent)
            for step in optimization_potential.optimization_plan:
                try:
                    original_snippet, optimized_snippet = (
                        await self._process_optimization_step(
                            step, cells, index, model, tools, stats_aggregator
                        )
                    )
                    original_snippets.append(original_snippet)
                    optimized_snippets.append(optimized_snippet)
                except ValueError:
                    # Target cell not found, skip this step
                    continue

        log(f"Kernel client available: {kernel_client is not None}")
        log(f"Dependencies dict available: {dependencies_dict is not None}")

        # VALIDATION: Check if optimization preserves semantics
        if kernel_client and dependencies_dict:
            with timer(
                key="validation", message="Validating optimization preserves semantics"
            ):
                modified_globals = ValidationHelper.get_modified_globals_for_cell(
                    cell["id"], dependencies_dict
                )

                log(f"Modified globals to validate: {modified_globals}")

                if modified_globals:
                    # Build original and optimized code for comparison
                    original_code = cell["source"]
                    cell_map = {c["id"]: c for c in cells}
                    optimized_code = (
                        ValidationHelper.build_optimized_code_for_validation(
                            original_snippets, optimized_snippets, cell_map, cell["id"]
                        )
                    )

                    log(f"Original code length: {len(original_code)} chars")
                    log(f"Optimized code length: {len(optimized_code)} chars")

                    # Validate using lambda to wrap the kernel client call
                    is_valid, error_msg = ValidationHelper.validate_optimization(
                        original_code,
                        optimized_code,
                        modified_globals,
                        lambda **kwargs: self._send_test_code_comm(
                            kernel_client, **kwargs
                        ),
                    )

                    if not is_valid:
                        error(f"Validation failed: {error_msg}")

                        log(f"Skipping optimization for cell {index}")
                        return cell["id"], OptimizationResultAndStats(
                            original_code=[],
                            optimized_code=[],
                            stats=stats_aggregator.get_aggregated_stats(),
                        )
                    else:
                        log("✓ Validation passed")

        return cell["id"], OptimizationResultAndStats(
            original_code=original_snippets,
            optimized_code=optimized_snippets,
            stats=stats_aggregator.get_aggregated_stats(),
        )

    async def optimize_cells(
        self,
        nb: nbformat.NotebookNode,
        model: Any,
        cell_ids: Optional[List[str]] = None,
        kernel_client: Optional[FerretKernelClient] = None,
    ) -> Tuple[nbformat.NotebookNode, float]:
        """Optimize cells in the notebook.

        Args:
            nb: The notebook to optimize
            model: The LLM model to use
            cell_ids: Optional list of specific cell IDs to optimize
            kernel_client: Optional kernel client for validation

        Returns:
            Tuple of (modified_notebook, total_cost)
        """
        with timer(key="optimize_cells_total", message="Optimizing cells"):
            log("=" * 60)
            log("# Optimizing Cells")
            log("=" * 60)

            # Analyze notebook dependencies once for all validations
            dependencies_dict = None
            if kernel_client:
                with timer(
                    key="analyze_dependencies",
                    message="Analyzing notebook dependencies",
                ):
                    dependencies_dict = analyze_notebook(nb)

            new_nb: nbformat.NotebookNode = copy.deepcopy(nb)

            # Track which cells we need to process
            with timer(key="identify_cells", message="Identifying cells to optimize"):
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
                log("No cells to optimize (no optimization plans found)")
                return new_nb, 0.0

            log(f"Found {len(cells_to_process)} cell(s) to optimize")
            log("")
            log(
                "|{:<10}|{:<16}|{:<21}|{:<10}|{:<10}|{:<10}|".format(
                    "Index", "Target Cell", "Function", "Tokens", "Time (s)", "Cost ($)"
                )
            )
            log(
                "|{:-^10}|{:-^16}|{:-^21}|{:-^10}|{:-^10}|{:-^10}|".format(
                    "", "", "", "", "", ""
                )
            )

            # Process cells sequentially (non-concurrent)
            all_results = []
            with timer(
                key="process_all_cells", message="Processing all optimization steps"
            ):
                for index in cells_to_process:
                    cell_id, result = await self.optimize_cell(
                        index, new_nb, model, kernel_client, dependencies_dict
                    )
                    all_results.append((cell_id, result))

            log("")

            # Apply optimization results to the notebook
            with timer(
                key="apply_optimizations", message="Applying optimizations to notebook"
            ):
                cell_map = {cell["id"]: cell for cell in new_nb["cells"]}
                MetadataManager.apply_optimizations_to_notebook(cell_map, all_results)

            total_cost = sum([result.stats.cost for _, result in all_results])
            log(f"Total optimization cost: ${total_cost:.4f}")

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
            notebook_content, config.model, selected_cell_ids, kernel_client
        )

        metadata = {
            "status": "success",
            "command": self.command_name,
            "total_cost": total_cost,
        }

        return {"notebook": new_nb, "metadata": metadata}
