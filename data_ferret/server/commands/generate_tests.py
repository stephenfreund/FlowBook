"""
Generate unit tests command implementation.

Generates comprehensive unit tests for code cells using AI analysis.
Tests are added to existing tests in cell metadata (non-destructive).
"""

import asyncio
from typing import Any, Dict, Optional, List, Tuple, Set

from agents import Usage
import nbformat
from pydantic import BaseModel, Field

from data_ferret.agent.agent import FerretAgent, FerretStats
from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.notebook_analysis import NotebookAnalysis
from data_ferret.util.ferret_metadata import (
    FerretMetadata,
    UnitTest,
    UnitTests,
    set_unit_tests_ferret_metadata,
)
from data_ferret.util.output import log
from data_ferret.util.prompts import get_prompt


class GeneratedTests(BaseModel):
    """LLM-generated unit tests for a code cell."""

    tests: List[UnitTest] = Field(
        description="List of 3-5 unit tests covering various scenarios",
        min_items=3,
        max_items=5,
    )


class GenerateTestsResultAndStats(BaseModel):
    """Result of generating tests for a cell with stats."""

    tests: List[UnitTest] = Field(description="The generated unit tests")
    stats: FerretStats = Field(description="Statistics from the LLM call")


class GenerateTestsCommand(NotebookCommand):
    """Generates comprehensive unit tests for code cells using AI."""

    @property
    def command_name(self) -> str:
        return "generate_tests"

    @property
    def display_name(self) -> str:
        return "Generate Tests"

    @property
    def icon_name(self) -> str:
        return "ui-components:build"

    @property
    def tooltip(self) -> str:
        return "Auto-generate unit tests for cell(s)"

    @property
    def requires_kernel(self) -> bool:
        return False

    @staticmethod
    def format_environment_section(env_data: Optional[Dict[str, str]]) -> str:
        """Format environment information from profile metadata."""
        if env_data:
            env_lines = [f"- {var}: {type_}" for var, type_ in env_data.items()]
            return "".join(env_lines)
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

        return "\n".join(var_lines)

    async def _generate_tests_for_cell(
        self,
        cell: nbformat.NotebookNode,
        cell_id: str,
        analysis: NotebookAnalysis,
        model: str,
        context: str = "",
    ) -> Tuple[List[UnitTest], FerretStats]:
        """Generate unit tests for a single cell using AI."""

        # Get cell source code
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # Skip empty cells
        if not source.strip():
            return [], FerretStats(
                model=model,
                log_path="",
                time=0.0,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        # Get dependencies from analysis
        dependencies = analysis.get_dependencies(cell_id)
        if not dependencies:
            return [], FerretStats(
                model=model,
                log_path="",
                time=0.0,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        # Format variables for the prompt
        globals_read = sorted(dependencies.globals_read)
        globals_written = sorted(dependencies.globals_written)

        read_vars = ", ".join(globals_read) if globals_read else "None"
        written_vars = ", ".join(globals_written) if globals_written else "None"

        # Add context section if provided
        context_section = ""
        if context:
            context_section = f"""Context from previous cells:
```python
{context}
```
"""

        # Extract env_data from cell metadata (profile data)
        ferret_meta = FerretMetadata.from_cell(cell)
        env_data = ferret_meta.profile.env if ferret_meta.profile else None

        # Get live variables from analysis
        live_vars = analysis.get_live_out_variables(cell_id)

        # Get prompts from the prompt manager
        instructions = get_prompt("generate_tests_instructions")
        prompt = get_prompt(
            "generate_tests_input",
            cell_source=source,
            env_section=self.format_environment_section(env_data),
            live_vars_section=self.format_live_variables_section(live_vars, env_data),
        )

        # Generate tests using AI agent
        generated_tests, stats = await FerretAgent.make_and_run_agent(
            key="generate-tests",
            model=model,
            instructions=instructions,
            output_type=GeneratedTests,
            input=prompt,
            log_dir="agent-logs",
        )

        return generated_tests.tests if generated_tests else [], stats

    async def generate_tests_for_cell(
        self,
        index: int,
        nb: nbformat.NotebookNode,
        analysis: NotebookAnalysis,
        model: str,
    ) -> Tuple[str, GenerateTestsResultAndStats]:
        """Generate tests for a single cell and return the result with stats."""
        cells = nb["cells"]
        cell = cells[index]
        cell_id = cell["id"]

        # Check if cell has dependency info
        if not analysis.has_cell(cell_id):
            # No dependency info, return empty
            return cell_id, GenerateTestsResultAndStats(
                tests=[],
                stats=FerretStats(
                    model=model,
                    log_path="",
                    time=0.0,
                    usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                ),
            )

        # Build context from previous cells (for AI to understand variable definitions)
        context_lines = []
        for i, prev_cell in enumerate(cells[:index]):
            if prev_cell.get("cell_type") == "code":
                prev_source = prev_cell.get("source", "")
                if isinstance(prev_source, list):
                    prev_source = "".join(prev_source)
                if prev_source.strip():
                    context_lines.append(f"# Cell {i}")
                    context_lines.append(prev_source)
                    context_lines.append("")

        context = "\n".join(context_lines[-500:])  # Limit context size

        # Generate tests using AI
        tests, stats = await self._generate_tests_for_cell(
            cell, cell_id, analysis, model, context
        )

        # Print progress
        status = "✓" if tests else "✗"
        test_count = len(tests)
        tokens = stats.usage.total_tokens if stats.usage else 0
        log(
            f"Cell {index} ({cell_id[:8]}...): {status} Generated {test_count} tests | "
            f"Tokens: {tokens} | Time: {stats.time:.2f}s | Cost: ${stats.cost:.4f}"
        )

        return cell_id, GenerateTestsResultAndStats(tests=tests, stats=stats)

    async def generate_tests_for_cells(
        self,
        nb: nbformat.NotebookNode,
        analysis: NotebookAnalysis,
        model: str,
        cell_ids: Optional[List[str]] = None,
    ) -> Tuple[nbformat.NotebookNode, float]:
        """Generate tests for code cells concurrently."""
        log("Generating unit tests for cells...")

        # Create tasks for cells to process
        tasks = []
        for index, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)

                # Only process selected cells if specified
                if source.strip() and (cell_ids is None or cell["id"] in cell_ids):
                    tasks.append(
                        self.generate_tests_for_cell(
                            index, nb, analysis, model
                        )
                    )

        if not tasks:
            log("No cells to process")
            return nb, 0.0

        # Execute all generation tasks concurrently
        results = await asyncio.gather(*tasks)

        # Create new notebook with updated metadata
        new_nb: nbformat.NotebookNode = nbformat.from_dict(nb)
        cell_map = {cell["id"]: cell for cell in new_nb["cells"]}

        # Apply generated tests to cells (ADD to existing tests, don't replace)
        tests_added = 0
        for cell_id, result in results:
            cell = cell_map.get(cell_id)
            if cell is None:
                continue

            if result.tests:
                # Get existing tests from metadata
                ferret_meta = FerretMetadata.from_cell(cell)
                existing_tests = (
                    ferret_meta.unit_tests.tests if ferret_meta.unit_tests else []
                )

                # Combine existing and new tests
                combined_tests = existing_tests + result.tests
                tests_added += len(result.tests)

                # Save back to metadata
                unit_tests = UnitTests(tests=combined_tests)
                set_unit_tests_ferret_metadata(cell, unit_tests)

                log(
                    f"Cell {cell_id[:8]}...: Added {len(result.tests)} tests "
                    f"(total now: {len(combined_tests)})"
                )

        total_cost = sum([result.stats.cost for _, result in results])
        log(f"\nTotal: {tests_added} tests added | Cost: ${total_cost:.4f}")

        return new_nb, total_cost

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Generate unit tests for code cells."""
        with self.timing_context() as get_elapsed:
            # Analyze notebook (dependencies + liveness)
            analysis = NotebookAnalysis(notebook_content)

            # Generate tests concurrently
            new_nb, total_cost = await self.generate_tests_for_cells(
                notebook_content,
                analysis,
                config.model if config else "gpt-4o",
                selected_cell_ids,
            )

            cells_processed = (
                len(selected_cell_ids)
                if selected_cell_ids
                else len(
                    [
                        c
                        for c in notebook_content["cells"]
                        if c.get("cell_type") == "code"
                    ]
                )
            )

            metadata = {
                "status": "success",
                "command": self.command_name,
                "cells_processed": cells_processed,
                "total_cost": total_cost,
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=dict(new_nb),
            metadata=metadata,
            total_cost=total_cost,
            total_time=total_time,
        )
