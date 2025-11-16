"""
Optimize cells command implementation.
"""

import asyncio
import copy
import time
import traceback
from typing import Any, Dict, Optional, List, Tuple, Set

from data_ferret.kernel.checkpoint import is_valid_variable_name
from data_ferret.kernel.kernel_command_client import KernelCommandClient
from data_ferret.server.base import NotebookCommand
from data_ferret.util.notebook_tools import NotebookTools
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.ferret_metadata import (
    FerretMetadata,
    OptimizationPotential,
    OptimizationStep,
    CodeSnippet,
    OptimizedCodeResponse,
    BatchOptimizedCodeResponse,
    OptimizedCodeMetadata,
    OptimizationAppliedMetadata,
    set_optimized_ferret_metadata,
    set_optimization_applied_ferret_metadata,
)
from data_ferret.agent.agent import FerretAgent, FerretStats
from data_ferret.util.prompts import get_prompt
from data_ferret.util.dependencies import analyze_notebook, CellDependencies
from data_ferret.util.notebook_analysis import NotebookAnalysis
from data_ferret.kernel.types import (
    DiffResult, TestCodeResult, TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash,
    ExecutionError, format_diff_as_markdown
)
from data_ferret.util.output import log, error, timer

import nbformat
from pydantic import BaseModel, Field
import ast


class PrePostEnvironments(BaseModel):
    """Optional pre-existing checkpoint names to skip original code execution."""

    original_environment: Optional[str] = Field(
        None,
        description="Name of checkpoint containing the pre-execution environment"
    )
    original_result: Optional[str] = Field(
        None,
        description="Name of checkpoint containing the post-execution environment"
    )
    original_duration: Optional[float] = Field(
        None,
        description="Duration of original code execution (from profile metadata)"
    )


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


class CodeExecutionOrchestrator:
    """
    Orchestrates code execution testing using kernel commands.

    Replaces the monolithic test_code command with composable operations.
    """

    def __init__(self, kernel_client: FerretKernelClient):
        self.kernel_client = kernel_client
        self.cmd_client = KernelCommandClient(kernel_client)

    def test_code(
        self,
        original_code: str,
        modified_code: str,
        output_variables: Set[str],
        pre_post_envs: Optional[PrePostEnvironments] = None,
    ) -> TestCodeResult:
        """
        Test original vs modified code by orchestrating kernel operations.

        Args:
            original_code: The original code to execute
            modified_code: The modified code to execute
            output_variables: Set of variable names to compare
            pre_post_envs: Optional pre-existing checkpoints to skip original execution

        Returns:
            TestCodeSuccess, TestCodeOriginalCrash, or TestCodeModifiedCrash
        """
        # Check if we should use pre-existing checkpoints
        use_existing_checkpoints = (
            pre_post_envs is not None
            and pre_post_envs.original_environment is not None
            and pre_post_envs.original_result is not None
        )

        if use_existing_checkpoints:
            # Skip steps 1-3: Use existing checkpoints
            # Use provided duration from profile, or 0.0 if not available
            original_duration = pre_post_envs.original_duration or 0.0
            log(f"Using existing checkpoints: {pre_post_envs.original_environment} and {pre_post_envs.original_result}")
            log(f"Using profile duration: {original_duration:.2f}s")
            env_checkpoint_name = pre_post_envs.original_environment
            result_checkpoint_name = pre_post_envs.original_result
        else:
            # Step 1: Save original environment
            self.cmd_client.checkpoint_save("original_environment")

            # Step 2: Execute original code with crash handling
            original_duration, original_error = self._execute_code_safely(original_code)

            if original_error:
                # Original crashed - return error
                return TestCodeOriginalCrash(
                    error=original_error,
                    original_duration=original_duration
                )

            # Step 3: Save original result
            self.cmd_client.checkpoint_save("original_result")

            env_checkpoint_name = "original_environment"
            result_checkpoint_name = "original_result"

        # Step 4: Restore original environment
        self.cmd_client.checkpoint_restore(env_checkpoint_name)

        # Step 5: Execute modified code with crash handling
        modified_duration, modified_error = self._execute_code_safely(modified_code)

        if modified_error:
            # Modified crashed (original worked) - return error
            return TestCodeModifiedCrash(
                error=modified_error,
                original_duration=original_duration,
                modified_duration=modified_duration
            )

        # Step 6: Save modified result
        self.cmd_client.checkpoint_save("modified_result")

        # Step 7: Compare checkpoints (only check specified output variables)
        compare_response = self.cmd_client.checkpoint_compare(
            result_checkpoint_name,
            "modified_result",
            keys_to_include=output_variables
        )

        # The diff is already filtered by keys_to_include, no need to filter again
        diff_result = compare_response.diff

        # Step 8: Calculate speedup
        speedup = original_duration / modified_duration if modified_duration > 0 else 0.0

        # Step 9: Return success result
        return TestCodeSuccess(
            diff=diff_result,
            original_duration=original_duration,
            modified_duration=modified_duration,
            speedup=speedup
        )

    def _execute_code_safely(
        self,
        code: str
    ) -> Tuple[float, Optional[ExecutionError]]:
        """
        Execute code and capture any errors.

        Args:
            code: The code to execute

        Returns:
            (duration, error) - error is None if execution succeeded
        """
        start_time = time.time()

        try:
            # Execute code in kernel
            msg_id = self.kernel_client.execute(code, store_history=False)

            # Wait for execution to complete and check for errors
            error = self._wait_for_execution(msg_id, code)
            duration = time.time() - start_time

            return duration, error

        except Exception as e:
            duration = time.time() - start_time
            error = ExecutionError(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                code_snippet=code
            )
            return duration, error

    def _wait_for_execution(self, msg_id: str, code: str) -> Optional[ExecutionError]:
        """
        Wait for execution to complete and check for errors.

        Args:
            msg_id: The message ID of the execution request
            code: The code that was executed (for error reporting)

        Returns:
            ExecutionError if execution failed, None if succeeded
        """
        error_occurred = None

        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=60.0)

                # Check if this message is for our execution
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue

                msg_type = msg['header']['msg_type']

                if msg_type == 'error':
                    # Execution error - save it but keep waiting for status: idle
                    content = msg['content']
                    error_occurred = ExecutionError(
                        error_type=content.get('ename', 'UnknownError'),
                        error_message=content.get('evalue', ''),
                        traceback='\n'.join(content.get('traceback', [])),
                        code_snippet=code
                    )

                elif msg_type == 'status':
                    # Check if execution is done
                    execution_state = msg['content']['execution_state']
                    if execution_state == 'idle':
                        # Execution completed - return error if one occurred
                        return error_occurred

            except Exception:
                # Timeout or other error - continue waiting
                continue

    def _filter_diff(
        self,
        diff: Dict[str, Any],
        output_variables: Set[str]
    ) -> DiffResult:
        """
        Filter diff result to only include specified output variables.

        Args:
            diff: Complete diff from checkpoint comparison
            output_variables: Variables to include

        Returns:
            Filtered DiffResult containing only the specified variables
        """
        # The checkpoint_compare returns a diff dictionary
        # We need to filter it to only include variables in output_variables
        if not output_variables:
            # If no output variables specified, return the full diff
            return DiffResult(differences=diff)

        # Filter to only include specified variables
        filtered_diff = {k: v for k, v in diff.items() if k in output_variables}
        return DiffResult(differences=filtered_diff)


class OptimizationResultAndStats(BaseModel):
    original_code: List[CodeSnippet] = Field(
        description="Original code snippets that will be replaced"
    )
    optimized_code: List[CodeSnippet] = Field(description="Optimized code snippets")
    stats: FerretStats
    original_duration: Optional[float] = Field(
        None, description="Original execution time in seconds (from validation)"
    )
    modified_duration: Optional[float] = Field(
        None, description="Modified execution time in seconds (from validation)"
    )
    speedup: Optional[float] = Field(
        None, description="Speedup ratio (original / modified)"
    )


class RepairedOptimizationResponse(BaseModel):
    """Response from LLM when repairing failed optimizations."""
    explanation: str = Field(
        description="Explanation of the repair process and the changes made"
    )

    repaired_snippets: List[CodeSnippet] = Field(
        description="Complete list of repaired code snippets, one for each optimization step"
    )



class ValidationHelper:
    """Helper class for optimization validation operations."""

    @classmethod
    def get_modified_globals_for_cell(
        cls, cell_id: str, analysis: Optional[NotebookAnalysis]
    ) -> Set[str]:
        """
        Extract live global variables written by a cell for validation.

        This returns only variables that are:
        1. Written by this cell
        2. Live after this cell (will be used by subsequent cells)
        3. Valid variable names (not private/system variables)

        Args:
            cell_id: ID of the cell to analyze
            analysis: NotebookAnalysis instance (None if validation disabled)

        Returns:
            Set of variable names that need validation
        """
        if analysis is None or not analysis.has_cell(cell_id):
            return set()

        # Get validation variables (written AND live)
        validation_vars = analysis.get_validation_variables(cell_id)

        # Filter out private and system variables
        filtered_vars = {var for var in validation_vars if is_valid_variable_name(var)}

        return filtered_vars

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
        kernel_client: FerretKernelClient,
        pre_post_envs: Optional[PrePostEnvironments] = None,
    ) -> Tuple[bool, Optional[str], Optional[TestCodeResult]]:
        """
        Validate that optimization preserves semantics.

        Args:
            original_code: The original code
            optimized_code: The optimized code
            modified_globals: Set of global variables to check
            kernel_client: FerretKernelClient for executing operations
            pre_post_envs: Optional pre-existing checkpoints to skip original execution

        Returns:
            (True, None, test_code_result) if validation passes
            (False, error_message, None) if validation fails
        """
        if not modified_globals:
            # No globals to validate, automatically pass
            return (True, None, None)

        try:
            # Create orchestrator
            orchestrator = CodeExecutionOrchestrator(kernel_client)

            # Run test
            result = orchestrator.test_code(
                original_code=original_code,
                modified_code=optimized_code,
                output_variables=modified_globals,
                pre_post_envs=pre_post_envs
            )

            # Handle the three possible result types
            if isinstance(result, TestCodeSuccess):
                # Both codes succeeded - check if outputs match
                diff_result = result.diff

                if not diff_result.differences:
                    # All variables match - validation passed
                    return (True, None, result)
                else:
                    # Variables differ - validation failed
                    diff_vars = list(diff_result.differences.keys())
                    error_msg = f"Variables changed: {', '.join(diff_vars)}"
                    error_msg += f"\n\n{format_diff_as_markdown(diff_result)}"
                    return (False, error_msg, None)

            elif isinstance(result, TestCodeOriginalCrash):
                # Original code crashed - cannot optimize broken code
                error = result.error
                error_msg = "Original code crashes - cannot optimize broken code:\n\n"
                error_msg += f"**{error.error_type}**: {error.error_message}\n\n"
                error_msg += "**Traceback:**\n```\n"
                error_msg += error.traceback
                error_msg += "\n```"
                return (False, error_msg, None)

            elif isinstance(result, TestCodeModifiedCrash):
                # Optimized code crashed - optimization introduced a bug
                error = result.error
                error_msg = "Optimized code crashes (original works) - optimization introduced a bug:\n\n"
                error_msg += f"**{error.error_type}**: {error.error_message}\n\n"
                error_msg += "**Traceback:**\n```\n"
                error_msg += error.traceback
                error_msg += "\n```"
                return (False, error_msg, None)

            else:
                # Unknown result type
                error_msg = f"Unknown test result type: {type(result)}"
                return (False, error_msg, None)

        except Exception as e:
            error_msg = f"Validation exception: {type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return (False, error_msg, None)


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
            env_lines = [f"- {var}: {type_}" for var, type_ in env_data.items()]
            return (
                "Available variables in the environment:\n"
                + "```\n"
                + "\n".join(env_lines)
                + "\n```\n"
            )
        return ""

    @staticmethod
    def format_live_variables_section(
        live_vars: Set[str], env_data: Optional[Dict[str, str]] = None
    ) -> str:
        """Format live variables that must be preserved during optimization.

        Args:
            live_vars: Set of variable names that are live (will be used by subsequent cells)
            env_data: Optional environment data with type information

        Returns:
            Formatted string describing live variables that must be preserved
        """
        if not live_vars:
            return ""

        # Sort for consistent output
        sorted_vars = sorted(live_vars)

        # Add type information if available
        if env_data:
            var_lines = []
            for var in sorted_vars:
                type_info = env_data.get(var, "unknown")
                var_lines.append(f"- {var}: {type_info}")
        else:
            var_lines = [f"- {var}" for var in sorted_vars]

        return (
            "CRITICAL: The following variables MUST have the exact same values after optimization:\n"
            + "```\n"
            + "\n".join(var_lines)
            + "\n```\n"
            + "These variables are used by subsequent cells and cannot be modified or removed."
        )


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

    def _format_snippets_for_repair(self, snippets: List[CodeSnippet]) -> str:
        """Format snippets for the repair prompt."""
        formatted = []
        for i, snippet in enumerate(snippets, 1):
            func_desc = f" (function: {snippet.function_name})" if snippet.function_name else " (whole cell)"
            formatted.append(f"## Snippet {i}: Cell {snippet.cell_id}{func_desc}")
            formatted.append(f"```python")
            formatted.append(snippet.source)
            formatted.append(f"```")
            if snippet.optimizations_applied:
                formatted.append(f"Optimizations applied: {', '.join(snippet.optimizations_applied)}")
            formatted.append("")
        return "\n".join(formatted)

    async def _repair_optimization(
        self,
        cells: List[Any],
        cell: Any,
        original_snippets: List[CodeSnippet],
        failed_snippets: List[CodeSnippet],
        validation_error: str,
        model: Any,
        tools: NotebookTools,
        analysis: Optional[NotebookAnalysis],
        stats_aggregator: StatsAggregator,
    ) -> Optional[List[CodeSnippet]]:
        """
        Attempt to repair failed optimizations using LLM.

        Args:
            cells: All notebook cells
            cell: The cell being optimized
            original_snippets: All original code snippets
            failed_snippets: All optimized snippets that failed validation
            validation_error: The validation error message
            model: The LLM model to use
            tools: NotebookTools instance
            analysis: NotebookAnalysis instance for filtering env to dependencies
            stats_aggregator: Stats aggregator for tracking costs

        Returns:
            List of repaired CodeSnippets or None if repair failed
        """
        with timer(
            key="repair_optimization",
            message=f"Repairing optimization for cell {cell['id']}",
        ):
            # Get profile metadata for environment information
            target_ferret_metadata = FerretMetadata.from_cell(cell)
            target_profile = target_ferret_metadata.get_profile()
            full_env = target_profile.env if target_profile else {}

            # Filter env to only dependencies of this cell
            env_data = analysis.filter_env_to_dependencies(cell["id"], full_env) if analysis else full_env

            # Build context prefix (code before this cell)
            cell_index = next(i for i, c in enumerate(cells) if c["id"] == cell["id"])
            prefix = CodeExtractor.build_context_prefix(cells, cell_index)

            # Format snippets and environment
            original_snippets_text = self._format_snippets_for_repair(original_snippets)
            optimized_snippets_text = self._format_snippets_for_repair(failed_snippets)
            env_section = CodeExtractor.format_environment_section(env_data)

            # Get live variables that must be preserved
            live_vars_section = ""
            if analysis:
                live_vars = analysis.get_validation_variables(cell["id"])
                live_vars_section = CodeExtractor.format_live_variables_section(
                    live_vars, env_data
                )

            # Create repair agent
            agent = FerretAgent[RepairedOptimizationResponse](
                key="optimization_repair",
                model=model,
                instructions=get_prompt("optimization_repair_instructions"),
                output_type=RepairedOptimizationResponse,
                tools=tools.tools(include_profile=True),
            )

            # Build repair input
            input_text = get_prompt(
                "optimization_repair_input",
                prefix=prefix,
                original_snippets=original_snippets_text,
                optimized_snippets=optimized_snippets_text,
                env_section=env_section,
                live_vars_section=live_vars_section,
                validation_error=validation_error,
            )

            try:
                repair_response, stats = await agent.run(input_text)
                stats_aggregator.add_stats(stats)

                log(f"Repair attempt completed with {len(repair_response.repaired_snippets)} snippets")
                log(f"Repair explanation: {repair_response.explanation}")
                # for snippet in repair_response.repaired_snippets:
                #     if snippet.optimizations_applied:
                #         log(f"  {snippet.cell_id}/{snippet.function_name or 'whole'}: {', '.join(snippet.optimizations_applied)}")

                return repair_response.repaired_snippets

            except Exception as e:
                import traceback
                error(f"Repair failed with exception: {type(e).__name__}: {str(e)}")
                error(f"Traceback:\n{traceback.format_exc()}")
                return None

    async def _validate_optimization_with_retry(
        self,
        cell: Any,
        cells: List[Any],
        original_snippets: List[CodeSnippet],
        optimized_snippets: List[CodeSnippet],
        model: Any,
        tools: NotebookTools,
        kernel_client: FerretKernelClient,
        analysis: Optional[NotebookAnalysis],
        stats_aggregator: StatsAggregator,
        pre_post_envs: Optional[PrePostEnvironments] = None,
    ) -> Tuple[bool, List[CodeSnippet], Optional[Dict[str, float]]]:
        """
        Validate optimization with retry logic.

        Args:
            cell: The cell being optimized
            cells: All notebook cells
            original_snippets: Original code snippets
            optimized_snippets: Optimized code snippets
            model: LLM model to use for repairs
            tools: NotebookTools instance
            kernel_client: Kernel client for validation
            analysis: NotebookAnalysis instance for dependency/liveness info
            stats_aggregator: Stats aggregator
            pre_post_envs: Optional pre-existing checkpoints to skip original execution

        Returns:
            Tuple of (success, validated_snippets, timing_data)
            - success: True if validation passed (possibly after repairs)
            - validated_snippets: The snippets that passed validation (may be repaired)
            - timing_data: Timing information from validation
        """
        MAX_RETRIES = 3
        current_snippets = optimized_snippets.copy()

        for attempt in range(MAX_RETRIES + 1):  # 0 = initial, 1-3 = retries
            if attempt == 0:
                log(f"Validating optimization for cell {cell['id']}")
            else:
                log(f"Retry attempt {attempt}/{MAX_RETRIES} for cell {cell['id']}")

            # Get modified globals for validation (only live variables)
            modified_globals = ValidationHelper.get_modified_globals_for_cell(
                cell["id"], analysis
            )

            log(f"Modified globals: {modified_globals}")

            if not modified_globals:
                log("No globals to validate, skipping validation")
                return (True, current_snippets, None)

            log(f"Modified globals to validate: {modified_globals}")

            # Build code for validation
            original_code = cell["source"]
            cell_map = {c["id"]: c for c in cells}
            optimized_code = ValidationHelper.build_optimized_code_for_validation(
                original_snippets, current_snippets, cell_map, cell["id"]
            )

            with timer(key="original_code", message="Original code"):
                log(original_code)
            with timer(key="optimized_code", message="Optimized code"):
                log(optimized_code)

            # Validate
            is_valid, error_msg, test_result = ValidationHelper.validate_optimization(
                original_code,
                optimized_code,
                modified_globals,
                kernel_client,
                pre_post_envs,
            )

            if is_valid:
                log("✓ Validation passed")
                # Extract timing data
                timing_data = None
                if test_result:
                    timing_data = {
                        "original_duration": test_result.original_duration,
                        "modified_duration": test_result.modified_duration,
                        "speedup": test_result.speedup,
                    }
                    log(
                        f"Timing: {test_result.original_duration:.2f}s → "
                        f"{test_result.modified_duration:.2f}s "
                        f"(speedup: {test_result.speedup:.2f}x)"
                    )
                return (True, current_snippets, timing_data)

            # Validation failed
            error(f"Validation failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {error_msg}")

            # If this was the last attempt, give up
            if attempt >= MAX_RETRIES:
                error(
                    f"Validation failed after {MAX_RETRIES} retries, "
                    f"reverting to original code"
                )
                return (False, [], None)

            # Try to repair ALL snippets at once
            log(f"Attempting to repair all snippets (retry {attempt + 1}/{MAX_RETRIES})")

            repaired_snippets = await self._repair_optimization(
                cells,
                cell,
                original_snippets,
                current_snippets,
                error_msg,
                model,
                tools,
                analysis,
                stats_aggregator,
            )

            if not repaired_snippets:
                error("Repair failed, giving up")
                return (False, [], None)

            # Replace all snippets with repaired versions
            current_snippets = repaired_snippets
            log(f"✓ Replaced all snippets with repaired versions")

            # Loop back to validate the repaired code

        # Should not reach here, but just in case
        return (False, [], None)

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

    def _extract_target_cell_and_metadata(
        self, cells: List[Any], step: OptimizationStep
    ) -> Tuple[Any, int, Optional[Dict[str, str]]]:
        """
        Extract target cell and its metadata.

        Args:
            cells: All notebook cells
            step: The optimization step

        Returns:
            Tuple of (target_cell, target_index, env_data)

        Raises:
            ValueError: If target cell not found
        """
        result = self._find_target_cell(cells, step.target_cell_id)
        if result is None:
            error(f"Target cell {step.target_cell_id} not found")
            raise ValueError(f"Target cell {step.target_cell_id} not found")

        target_cell, target_index = result

        # Extract profile metadata to get environment information
        target_ferret_metadata = FerretMetadata.from_cell(target_cell)
        target_profile = target_ferret_metadata.get_profile()
        env_data = target_profile.env if target_profile else None

        return target_cell, target_index, env_data

    def _build_optimization_context(
        self,
        cells: List[Any],
        target_cell: Any,
        target_index: int,
        step: OptimizationStep,
        env_data: Optional[Dict[str, str]],
        analysis: Optional[NotebookAnalysis] = None,
    ) -> Tuple[str, str, str, str, str, str]:
        """
        Build context and extract code for optimization.

        Args:
            cells: All notebook cells
            target_cell: The target cell
            target_index: Index of target cell
            step: The optimization step
            env_data: Environment data from profile
            analysis: Optional NotebookAnalysis for dependency/liveness info

        Returns:
            Tuple of (prefix, original_function_code, optimization_context,
                     optimization_descriptions, env_section, live_vars_section)
        """
        # Build context prefix
        prefix = CodeExtractor.build_context_prefix(cells, target_index)

        # Extract code to optimize
        _, original_function_code, optimization_context = (
            CodeExtractor.extract_optimization_target(target_cell, step)
        )

        # Format optimization descriptions and environment section
        optimization_descriptions = CodeExtractor.format_optimization_descriptions(
            step.description
        )
        env_section = CodeExtractor.format_environment_section(env_data)

        # Get live variables that must be preserved
        live_vars_section = ""
        if analysis:
            live_vars = analysis.get_validation_variables(step.target_cell_id)
            live_vars_section = CodeExtractor.format_live_variables_section(
                live_vars, env_data
            )

        return (
            prefix,
            original_function_code,
            optimization_context,
            optimization_descriptions,
            env_section,
            live_vars_section,
        )

    async def _run_llm_optimization(
        self,
        step: OptimizationStep,
        prefix: str,
        cell_source: str,
        optimization_descriptions: str,
        env_section: str,
        live_vars_section: str,
        model: Any,
        tools: NotebookTools,
        stats_aggregator: StatsAggregator,
    ) -> Tuple[OptimizedCodeResponse, FerretStats]:
        """
        Run LLM optimization.

        Args:
            step: The optimization step
            prefix: Context prefix (code before the cell)
            cell_source: The source code to optimize
            optimization_descriptions: Formatted optimization descriptions
            env_section: Formatted environment section
            live_vars_section: Formatted live variables section
            model: LLM model to use
            tools: NotebookTools instance
            stats_aggregator: Stats aggregator

        Returns:
            Tuple of (optimization_response, stats)
        """
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
            live_vars_section=live_vars_section,
            optimization_descriptions=optimization_descriptions,
        )

        optimization_response, stats = await agent.run(input_text)
        stats_aggregator.add_stats(stats)

        return optimization_response, stats

    def _format_snippets_for_batch(
        self,
        snippets: List[CodeSnippet]
    ) -> str:
        """
        Format multiple code snippets into a single string with headers.

        Format:
        ### Cell ABC123 / Function process_data

        def process_data(df):
            ...

        ### Cell XYZ789

        for i in range(n):
            ...

        Args:
            snippets: List of CodeSnippet objects

        Returns:
            Formatted string with all snippets
        """
        formatted_parts = []

        for snippet in snippets:
            # Create header
            if snippet.function_name:
                header = f"### Cell {snippet.cell_id} / Function {snippet.function_name}"
            else:
                header = f"### Cell {snippet.cell_id}"

            formatted_parts.append(header)
            formatted_parts.append("")  # blank line
            formatted_parts.append(snippet.source)
            formatted_parts.append("")  # blank line between snippets

        return "\n".join(formatted_parts)

    def _format_optimization_descriptions_batch(
        self,
        optimization_plan: List[OptimizationStep]
    ) -> str:
        """
        Format all optimization descriptions from the plan into a numbered list.

        Format:
        1. [Cell ABC123 / Function process_data] Use vectorized operations instead of loops
        2. [Cell ABC123 / Function process_data] Cache expensive computations
        3. [Cell XYZ789] Replace list concatenation with list comprehension

        Args:
            optimization_plan: List of OptimizationStep objects

        Returns:
            Formatted string with numbered optimizations
        """
        optimization_lines = []
        counter = 1

        for step in optimization_plan:
            # Create location identifier
            if step.function_name:
                location = f"Cell {step.target_cell_id} / Function {step.function_name}"
            else:
                location = f"Cell {step.target_cell_id}"

            # Add each description
            for desc in step.description:
                optimization_lines.append(f"{counter}. [{location}] {desc}")
                counter += 1

        return "\n".join(optimization_lines)

    async def _extract_original_snippets_from_plan(
        self,
        optimization_plan: List[OptimizationStep],
        cells: List[Any]
    ) -> List[CodeSnippet]:
        """
        Extract all original code snippets from the optimization plan.

        This is similar to the current loop in _process_all_optimization_steps,
        but only extracts the original code without calling the LLM.

        Args:
            optimization_plan: List of optimization steps
            cells: All notebook cells

        Returns:
            List of original CodeSnippet objects

        Raises:
            ValueError: If a target cell is not found
        """
        original_snippets = []

        for step in optimization_plan:
            # Find target cell
            result = self._find_target_cell(cells, step.target_cell_id)
            if result is None:
                raise ValueError(f"Target cell {step.target_cell_id} not found")

            target_cell, _ = result

            # Extract the code to optimize
            _, original_code, _ = CodeExtractor.extract_optimization_target(
                target_cell, step
            )

            # Create snippet
            snippet = CodeSnippet(
                cell_id=step.target_cell_id,
                function_name=step.function_name,
                source=original_code
            )

            original_snippets.append(snippet)

        return original_snippets

    def _build_batch_optimization_context(
        self,
        cells: List[Any],
        optimization_plan: List[OptimizationStep],
        analysis: Optional[NotebookAnalysis] = None
    ) -> Tuple[str, str, str]:
        """
        Build context for batch optimization (prefix, env_section, live_vars_section).

        Uses the context from the first cell in the optimization plan, or builds
        a comprehensive context if multiple cells are involved.

        Args:
            cells: All notebook cells
            optimization_plan: List of optimization steps
            analysis: Optional NotebookAnalysis for dependency/liveness info

        Returns:
            Tuple of (prefix, env_section, live_vars_section)
        """
        if not optimization_plan:
            return "", "", ""

        # Use the first step to determine context
        first_step = optimization_plan[0]

        # Extract target cell and metadata
        target_cell, target_index, env_data = self._extract_target_cell_and_metadata(
            cells, first_step
        )

        # Filter env to dependencies if analysis available
        if analysis:
            full_env = env_data or {}
            env_data = analysis.filter_env_to_dependencies(
                first_step.target_cell_id, full_env
            )

        # Build context prefix (code before the first target cell)
        prefix = CodeExtractor.build_context_prefix(cells, target_index)

        # Build environment section
        env_section = CodeExtractor.format_environment_section(env_data)

        # Collect all live variables from all affected cells
        all_live_vars = set()
        if analysis:
            for step in optimization_plan:
                live_vars = analysis.get_validation_variables(step.target_cell_id)
                all_live_vars.update(live_vars)

        live_vars_section = CodeExtractor.format_live_variables_section(
            all_live_vars, env_data
        )

        return prefix, env_section, live_vars_section

    async def _run_batch_llm_optimization(
        self,
        formatted_snippets: str,
        optimization_descriptions: str,
        prefix: str,
        env_section: str,
        live_vars_section: str,
        model: Any,
        tools: NotebookTools,
        stats_aggregator: StatsAggregator
    ) -> Tuple[BatchOptimizedCodeResponse, FerretStats]:
        """
        Run batch LLM optimization on multiple snippets.

        Args:
            formatted_snippets: All snippets formatted with headers
            optimization_descriptions: Numbered list of all optimizations
            prefix: Context prefix
            env_section: Environment section
            live_vars_section: Live variables section
            model: LLM model to use
            tools: NotebookTools instance
            stats_aggregator: Stats aggregator

        Returns:
            Tuple of (batch_response, stats)
        """
        agent = FerretAgent[BatchOptimizedCodeResponse](
            key="batch_cell_optimization",
            model=model,
            instructions=get_prompt("batch_optimization_instructions"),
            output_type=BatchOptimizedCodeResponse,
            tools=tools.tools(include_profile=True),
        )

        input_text = get_prompt(
            "batch_optimization_input",
            prefix=prefix,
            formatted_snippets=formatted_snippets,
            optimization_descriptions=optimization_descriptions,
            env_section=env_section,
            live_vars_section=live_vars_section,
        )

        batch_response, stats = await agent.run(input_text)
        stats_aggregator.add_stats(stats)

        return batch_response, stats

    async def _process_optimization_step(
        self,
        step: OptimizationStep,
        cells: List[Any],
        index: int,
        model: Any,
        tools: NotebookTools,
        stats_aggregator: StatsAggregator,
        analysis: Optional[NotebookAnalysis] = None,
    ) -> Tuple[CodeSnippet, CodeSnippet]:
        """Process a single optimization step.

        Args:
            step: The optimization step
            cells: All notebook cells
            index: Index of the cell being optimized
            model: LLM model to use
            tools: NotebookTools instance
            stats_aggregator: Stats aggregator
            analysis: Optional NotebookAnalysis for dependency/liveness info

        Returns:
            Tuple of (original_snippet, optimized_snippet)
        """
        target_desc = step.function_name or "whole cell"
        with timer(
            key="process_optimization_step",
            message=f"Processing optimization for {step.target_cell_id} ({target_desc})",
        ):
            # Extract target cell and metadata
            target_cell, target_index, env_data = self._extract_target_cell_and_metadata(
                cells, step
            )

            # Build context and extract code
            with timer(
                key="build_context", message="Building context and extracting code"
            ):
                (
                    prefix,
                    original_function_code,
                    _,
                    optimization_descriptions,
                    env_section,
                    live_vars_section,
                ) = self._build_optimization_context(
                    cells, target_cell, target_index, step, env_data, analysis
                )

            # Store original code snippet
            original_snippet = CodeSnippet(
                cell_id=step.target_cell_id,
                function_name=step.function_name,
                source=original_function_code,
            )

            # Run LLM optimization
            with timer(key="agent_optimization", message="Running LLM optimization"):
                # Need to get cell_source for the optimization
                cell_source, _, _ = CodeExtractor.extract_optimization_target(
                    target_cell, step
                )

                optimization_response, stats = await self._run_llm_optimization(
                    step,
                    prefix,
                    cell_source,
                    optimization_descriptions,
                    env_section,
                    live_vars_section,
                    model,
                    tools,
                    stats_aggregator,
                )

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

    async def _process_all_optimization_steps(
        self,
        optimization_plan: List[OptimizationStep],
        cells: List[Any],
        index: int,
        model: Any,
        tools: NotebookTools,
        stats_aggregator: StatsAggregator,
        analysis: Optional[NotebookAnalysis] = None,
    ) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:
        """
        Process all optimization steps for a cell using batch optimization.

        NEW APPROACH: Instead of calling the LLM once per step, we:
        1. Extract all original snippets from the plan
        2. Format them into a single string with headers
        3. Create a list of all optimizations to apply
        4. Call the LLM once to optimize all snippets together
        5. Extract optimized snippets from response

        Args:
            optimization_plan: List of optimization steps
            cells: All notebook cells
            index: Index of the cell being optimized (for logging)
            model: LLM model to use
            tools: NotebookTools instance
            stats_aggregator: Stats aggregator
            analysis: Optional NotebookAnalysis for dependency/liveness info

        Returns:
            Tuple of (original_snippets, optimized_snippets)
        """
        if not optimization_plan:
            return [], []

        with timer(
            key="process_all_optimization_steps_batch",
            message=f"Processing {len(optimization_plan)} optimization steps in batch"
        ):
            # Step 1: Extract all original snippets
            with timer(key="extract_snippets", message="Extracting original snippets"):
                try:
                    original_snippets = await self._extract_original_snippets_from_plan(
                        optimization_plan, cells
                    )
                except ValueError as e:
                    error(f"Failed to extract snippets: {e}")
                    return [], []

            if not original_snippets:
                return [], []

            # Step 2: Format all snippets into batch format
            formatted_snippets = self._format_snippets_for_batch(original_snippets)

            # Step 3: Create list of optimization descriptions
            optimization_descriptions = self._format_optimization_descriptions_batch(
                optimization_plan
            )

            # Step 4: Build context (prefix, env, live vars)
            prefix, env_section, live_vars_section = self._build_batch_optimization_context(
                cells, optimization_plan, analysis
            )

            # Step 5: Call LLM with batch optimization
            with timer(key="batch_llm_call", message="Running batch LLM optimization"):
                try:
                    batch_response, stats = await self._run_batch_llm_optimization(
                        formatted_snippets,
                        optimization_descriptions,
                        prefix,
                        env_section,
                        live_vars_section,
                        model,
                        tools,
                        stats_aggregator
                    )
                except Exception as e:
                    error(f"Batch LLM optimization failed: {type(e).__name__}: {str(e)}")
                    import traceback
                    error(f"Traceback:\n{traceback.format_exc()}")
                    return [], []

            # Step 6: Extract optimized snippets from response
            # The response has optimized_snippets list with optimizations_applied already set
            optimized_snippets = batch_response.optimized_snippets

            # Verify we got the right number of snippets
            if len(optimized_snippets) != len(original_snippets):
                error(
                    f"LLM returned {len(optimized_snippets)} snippets, "
                    f"expected {len(original_snippets)}"
                )
                return [], []

            # Log the results
            log(f"Batch optimization completed:")
            log(f"  Snippets processed: {len(optimized_snippets)}")
            log(f"  Total tokens: {stats.usage.total_tokens if stats.usage else 0}")
            log(f"  Time: {stats.time:.1f}s")
            log(f"  Cost: ${stats.cost:.4f}")

            for i, (orig, opt) in enumerate(zip(original_snippets, optimized_snippets)):
                target_desc = opt.function_name or "whole cell"
                log(
                    f"| {index:<9}| {opt.cell_id:<15}| {target_desc:<20}| "
                    f"{'-':<9}| {'-':<9}| {'-':<9}|"
                )
                if opt.optimizations_applied:
                    for opt_desc in opt.optimizations_applied:
                        log(f"    - {opt_desc}")

            return original_snippets, optimized_snippets

    def _build_optimization_result(
        self,
        cell_id: str,
        original_snippets: List[CodeSnippet],
        optimized_snippets: List[CodeSnippet],
        stats_aggregator: StatsAggregator,
        timing_data: Optional[Dict[str, float]] = None,
    ) -> OptimizationResultAndStats:
        """
        Build the optimization result.

        Args:
            cell_id: The cell ID
            original_snippets: Original code snippets
            optimized_snippets: Optimized code snippets
            stats_aggregator: Stats aggregator
            timing_data: Optional timing data from validation

        Returns:
            OptimizationResultAndStats
        """
        return OptimizationResultAndStats(
            original_code=original_snippets,
            optimized_code=optimized_snippets,
            stats=stats_aggregator.get_aggregated_stats(),
            original_duration=timing_data["original_duration"] if timing_data else None,
            modified_duration=timing_data["modified_duration"] if timing_data else None,
            speedup=timing_data["speedup"] if timing_data else None,
        )

    async def optimize_cell(
        self,
        index: int,
        nb: nbformat.NotebookNode,
        model: Any,
        kernel_client: Optional[FerretKernelClient] = None,
        analysis: Optional[NotebookAnalysis] = None,
        pre_post_envs: Optional[PrePostEnvironments] = None,
    ) -> Tuple[str, OptimizationResultAndStats]:
        """Optimize a single cell based on its optimization plan.

        This method iterates over the optimization steps for a cell and uses
        the LLM to generate optimized code for each step. It validates the
        optimizations and retries up to 3 times if validation fails.

        Args:
            index: Index of the cell in the notebook
            nb: The notebook containing the cell
            model: The LLM model to use for optimization
            kernel_client: Optional kernel client for validation
            analysis: Optional NotebookAnalysis for dependency/liveness info
            pre_post_envs: Optional pre-existing checkpoints to skip original execution

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
            return cell["id"], self._build_optimization_result(
                cell["id"], [], [], stats_aggregator
            )

        # Process all optimization steps
        with NotebookTools(nb) as tools:
            original_snippets, optimized_snippets = (
                await self._process_all_optimization_steps(
                    optimization_potential.optimization_plan,
                    cells,
                    index,
                    model,
                    tools,
                    stats_aggregator,
                    analysis,
                )
            )

        log(f"Kernel client available: {kernel_client is not None}")
        log(f"Analysis available: {analysis is not None}")

        # Initialize timing data
        timing_data = None

        # VALIDATION WITH RETRY: Check if optimization preserves semantics
        if kernel_client and analysis and optimized_snippets:
            with timer(
                key="validation", message="Validating optimization with retry support"
            ):
                with NotebookTools(nb) as tools:
                    is_valid, validated_snippets, timing_data = (
                        await self._validate_optimization_with_retry(
                            cell,
                            cells,
                            original_snippets,
                            optimized_snippets,
                            model,
                            tools,
                            kernel_client,
                            analysis,
                            stats_aggregator,
                            pre_post_envs,
                        )
                    )

                if not is_valid:
                    # Validation failed after retries, return empty result
                    log(f"Skipping optimization for cell {index} after validation failures")
                    return cell["id"], self._build_optimization_result(
                        cell["id"], [], [], stats_aggregator
                    )

                # Use validated snippets (may have been repaired)
                optimized_snippets = validated_snippets

        return cell["id"], self._build_optimization_result(
            cell["id"], original_snippets, optimized_snippets, stats_aggregator, timing_data
        )

    def _identify_cells_to_optimize(
        self,
        nb: nbformat.NotebookNode,
        cell_ids: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Identify cells that need optimization.

        Args:
            nb: The notebook
            cell_ids: Optional filter for specific cell IDs

        Returns:
            List of cell indices to process
        """
        cells_to_process = []
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                # Check if cell has optimization plan
                ferret_metadata = FerretMetadata.from_cell(cell)
                opt_potential = ferret_metadata.get_optimization_potential()

                if opt_potential and opt_potential.optimization_plan:
                    # Only process if in cell_ids list (or if no filter)
                    if cell_ids is None or cell["id"] in cell_ids:
                        cells_to_process.append(index)

        return cells_to_process

    async def _process_cells_sequentially(
        self,
        cells_to_process: List[int],
        nb: nbformat.NotebookNode,
        model: Any,
        kernel_client: Optional[FerretKernelClient],
        analysis: Optional[NotebookAnalysis],
        pre_post_envs: Optional[PrePostEnvironments] = None,
    ) -> List[Tuple[str, OptimizationResultAndStats]]:
        """
        Process all cells sequentially.

        Args:
            cells_to_process: List of cell indices to process
            nb: The notebook
            model: LLM model to use
            kernel_client: Optional kernel client
            analysis: Optional NotebookAnalysis instance
            pre_post_envs: Optional pre-existing checkpoints to skip original execution

        Returns:
            List of (cell_id, result) tuples
        """
        all_results = []
        for index in cells_to_process:
            cell_id, result = await self.optimize_cell(
                index, nb, model, kernel_client, analysis, pre_post_envs
            )
            all_results.append((cell_id, result))
        return all_results

    def _apply_and_finalize(
        self,
        nb: nbformat.NotebookNode,
        all_results: List[Tuple[str, OptimizationResultAndStats]],
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """
        Apply optimizations to notebook and compile final metadata.

        Args:
            nb: The notebook
            all_results: List of optimization results

        Returns:
            Tuple of (total_cost, cell_timing)
        """
        # Apply optimization results to the notebook
        cell_map = {cell["id"]: cell for cell in nb["cells"]}
        MetadataManager.apply_optimizations_to_notebook(cell_map, all_results)

        # Calculate total cost
        total_cost = sum([result.stats.cost for _, result in all_results])
        log(f"Total optimization cost: ${total_cost:.4f}")

        # Collect timing data for cells that were optimized
        cell_timing = {}
        for cell_id, result in all_results:
            if result.original_duration is not None and result.modified_duration is not None:
                cell_timing[cell_id] = {
                    "original_duration": result.original_duration,
                    "modified_duration": result.modified_duration,
                    "speedup": result.speedup,
                }

        return total_cost, cell_timing

    async def optimize_cells(
        self,
        nb: nbformat.NotebookNode,
        model: Any,
        cell_ids: Optional[List[str]] = None,
        kernel_client: Optional[FerretKernelClient] = None,
        pre_post_envs: Optional[PrePostEnvironments] = None,
    ) -> Tuple[nbformat.NotebookNode, float, Dict[str, Dict[str, float]]]:
        """Optimize cells in the notebook.

        Args:
            nb: The notebook to optimize
            model: The LLM model to use
            cell_ids: Optional list of specific cell IDs to optimize
            kernel_client: Optional kernel client for validation
            pre_post_envs: Optional pre-existing checkpoints to skip original execution

        Returns:
            Tuple of (modified_notebook, total_cost, cell_timing)
        """
        with timer(key="optimize_cells_total", message="Optimizing cells"):
            log("=" * 60)
            log("# Optimizing Cells")
            log("=" * 60)

            # Analyze notebook dependencies and liveness once for all validations
            analysis = None
            if kernel_client:
                with timer(
                    key="analyze_dependencies",
                    message="Analyzing notebook dependencies and liveness",
                ):
                    analysis = NotebookAnalysis(nb)
                    log(f"Analyzed {len(analysis.get_all_cell_ids())} cells")

            new_nb: nbformat.NotebookNode = copy.deepcopy(nb)

            # Identify cells to optimize
            with timer(key="identify_cells", message="Identifying cells to optimize"):
                cells_to_process = self._identify_cells_to_optimize(new_nb, cell_ids)

            if not cells_to_process:
                log("No cells to optimize (no optimization plans found)")
                return new_nb, 0.0, {}

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

            # Process cells sequentially
            with timer(
                key="process_all_cells", message="Processing all optimization steps"
            ):
                all_results = await self._process_cells_sequentially(
                    cells_to_process, new_nb, model, kernel_client, analysis, pre_post_envs
                )

            log("")

            # Apply optimizations and compile final metadata
            with timer(
                key="apply_optimizations", message="Applying optimizations to notebook"
            ):
                total_cost, cell_timing = self._apply_and_finalize(new_nb, all_results)

            return new_nb, total_cost, cell_timing

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        pre_post_envs: Optional[PrePostEnvironments] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize cells in the notebook based on their optimization plans."""
        new_nb, total_cost, cell_timing = await self.optimize_cells(
            notebook_content, config.model, selected_cell_ids, kernel_client, pre_post_envs
        )

        metadata = {
            "status": "success",
            "command": self.command_name,
            "total_cost": total_cost,
            "cell_timing": cell_timing,
        }

        return {"notebook": new_nb, "metadata": metadata}
