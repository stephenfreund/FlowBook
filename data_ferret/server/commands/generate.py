"""
Generate code from string specification command implementation.
"""

import ast
from typing import Any, Dict, Optional, List

import nbformat
from pydantic import BaseModel, Field

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.agent.agent import FerretAgent, FerretStats
from data_ferret.util.prompts import get_prompt
from data_ferret.util.output import log
from data_ferret.util.ferret_metadata import (
    FerretMetadata,
    GeneratedCodeMetadata,
    set_generated_ferret_metadata,
)


class GenerateCodeResult(BaseModel):
    """Result from generating code from a specification."""
    generated_code: str = Field(description="The generated Python code")
    explanation: str = Field(description="Brief explanation of what the generated code does")


class GenerateCodeResultAndStats(BaseModel):
    """Result and statistics from code generation."""
    generate_result: GenerateCodeResult
    stats: FerretStats


class GenerateCodeCommand(NotebookCommand):
    """Generate code from string specification using LLM."""

    @property
    def command_name(self) -> str:
        return "generate"

    @property
    def display_name(self) -> str:
        return "Generate Code"

    @property
    def icon_name(self) -> str:
        return "ui-components:code"

    @property
    def tooltip(self) -> str:
        return "Generate code from string specification"

    @property
    def requires_kernel(self) -> bool:
        return False

    def is_string_constant_cell(self, source: str) -> Optional[str]:
        """
        Check if a cell contains only a string constant.

        Args:
            source: The source code of the cell

        Returns:
            The string value if it's a string constant cell, None otherwise
        """
        source = source.strip()
        if not source:
            return None

        try:
            # Parse the source as Python code
            tree = ast.parse(source)

            # Check if it's a single expression statement with a constant string
            if (len(tree.body) == 1 and
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)):
                return tree.body[0].value.value
        except SyntaxError:
            pass

        return None

    async def generate_code_for_cell(
        self, cell: nbformat.NotebookNode, model: Any, prefix: str = "", env_data: Optional[Dict[str, str]] = None
    ) -> Optional[GenerateCodeResultAndStats]:
        """
        Generate code for a cell containing a string specification.

        Args:
            cell: The notebook cell
            model: The LLM model to use
            prefix: Code from previous cells for context
            env_data: Environment variables and their types from profiling

        Returns:
            GenerateCodeResultAndStats if successful, None if cell doesn't contain a string spec
        """
        spec = self.is_string_constant_cell(cell["source"])
        if spec is None:
            return None

        log(f"Generating code for specification: {spec[:100]}...")

        # Format environment information from profile metadata
        if env_data:
            env_lines = [f"  {var}: {type_}" for var, type_ in env_data.items()]
            env_section = "Available variables in the environment (from profiling):\n" + "\n".join(env_lines)
        else:
            env_section = ""

        # Create the agent
        agent = FerretAgent[GenerateCodeResult](
            key="code_generation",
            model=model,
            instructions=get_prompt("generate_instructions"),
            output_type=GenerateCodeResult,
        )

        # Format the input
        input_text = get_prompt(
            "generate_input",
            prefix=prefix,
            specification=spec,
            env_section=env_section,
        )

        # Run the agent
        result, stats = await agent.run(input_text)

        log(
            f"Generated code | Tokens: {stats.usage.total_tokens} | Cost: ${stats.cost:.4f}"
        )

        return GenerateCodeResultAndStats(generate_result=result, stats=stats)

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Process the generate code command."""
        with self.timing_context() as get_elapsed:
            log("Generating code from string specifications")

            nb = notebook_content
            new_cells = []
            total_cost = 0.0
            generated_count = 0

            # Build prefix for context
            prefix_lines = []
            # Track the most recent environment from profiled cells
            current_env = None

            for index, cell in enumerate(nb["cells"]):
                # Build context from previous code cells
                if cell["cell_type"] == "code" and cell["source"].strip():
                    prefix = "\n".join(prefix_lines)

                    # Update current environment from profile metadata if available
                    ferret_metadata = FerretMetadata.from_cell(cell)
                    profile = ferret_metadata.get_profile()
                    if profile and profile.env_after:
                        current_env = profile.env_after

                # Check if this cell should be processed
                should_process = (
                    cell["cell_type"] == "code" and
                    cell["source"].strip() and
                    (selected_cell_ids is None or cell["id"] in selected_cell_ids)
                )

                if should_process:
                    result_and_stats = await self.generate_code_for_cell(cell, config.model, prefix, current_env)

                    if result_and_stats:
                        result = result_and_stats.generate_result
                        stats = result_and_stats.stats

                        # Cell contains a string specification
                        # Modify the cell to include the generated code after the specification
                        original_source = cell["source"]
                        new_source = f"{original_source}\n\n# Generated code:\n{result.generated_code}"

                        # Update the cell source
                        cell = cell.copy()
                        cell["source"] = new_source

                        # Create and set the generated metadata using BaseModel
                        generated_metadata = GeneratedCodeMetadata(
                            explanation=result.explanation,
                            original_spec=original_source,
                        )
                        set_generated_ferret_metadata(cell, generated_metadata)

                        generated_count += 1
                        total_cost += stats.cost

                new_cells.append(cell)

                # Add to prefix for next cells
                if cell["cell_type"] == "code" and cell["source"].strip():
                    prefix_lines.append(cell["source"])

            nb["cells"] = new_cells

            metadata = {
                "status": "success",
                "command": self.command_name,
                "generated_count": generated_count,
                "total_cost": total_cost,
            }

            log(f"Generated code for {generated_count} cell(s) | Total cost: ${total_cost:.4f}")

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=nb,
            metadata=metadata,
            total_cost=total_cost,
            total_time=total_time
        )
