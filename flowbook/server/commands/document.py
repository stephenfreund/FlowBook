"""
Document command implementation.

Adds descriptive comments to code cells based on dependency analysis and profiling data.
"""

import asyncio
import copy
from typing import Any, Dict, Optional, List, Tuple

from agents import Usage
from flowbook.util.output import timer
import nbformat
from pydantic import BaseModel, Field

from flowbook.agent.agent import FlowbookAgent, FlowbookStats
from flowbook.kernel_support.checkpoint import is_valid_variable_name
from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.util.dependencies import analyze_notebook
from flowbook.util.flowbook_metadata import FlowbookMetadata
from flowbook.util.prompts import get_prompt


class VariableDescription(BaseModel):
    """Description of a variable."""

    variable_name: str = Field(description="The name of the variable")
    description: str = Field(description="What the variable captures or represents")


class CellDocumentation(BaseModel):
    """LLM-generated documentation for a code cell."""

    title: str = Field(description="A concise title describing what the cell does")
    description: List[str] = Field(
        description="Bullet points describing the cell's functionality"
    )
    variable_descriptions: List[VariableDescription] = Field(
        description="Descriptions of what each output variable captures",
        default_factory=list,
    )


class DocumentationResultAndStats(BaseModel):
    """Result of documenting a cell with stats."""

    comment: str = Field(description="The generated documentation comment")
    stats: FlowbookStats = Field(description="Statistics from the LLM call")


class DocumentCommand(NotebookCommand):
    """Adds descriptive documentation comments to code cells."""

    # Unique markers for documentation comments
    DOC_START_MARKER = (
        "# ======================================================================"
    )
    DOC_END_MARKER = (
        "# ======================================================================"
    )

    @property
    def command_name(self) -> str:
        return "document"

    @property
    def display_name(self) -> str:
        return "Document Cells"

    @property
    def icon_name(self) -> str:
        return "ui-components:text-editor"

    @property
    def tooltip(self) -> str:
        return "Add documentation comments to code cells"

    @property
    def requires_kernel(self) -> bool:
        return False

    def _format_type_info(self, var_name: str, env: Dict[str, str]) -> str:
        """Extract type information for a variable from profile data."""
        if var_name not in env:
            return ""

        type_name = env.get(var_name, "?")

        return type_name

    async def _generate_llm_documentation(
        self,
        source_code: str,
        globals_written: List[str],
        globals_read: List[str],
        model: str,
    ) -> Tuple[Optional[CellDocumentation], FlowbookStats]:
        """Use LLM to generate documentation for a code cell."""
        # Format variables for the prompt
        read_vars = ", ".join(globals_read) if globals_read else "None"
        written_vars = ", ".join(globals_written) if globals_written else "None"

        # Get prompts from the prompt manager
        instructions = get_prompt("document_instructions")
        prompt = get_prompt(
            "document_input",
            source_code=source_code,
            globals_read=read_vars,
            globals_written=written_vars,
        )

        doc, stats = await FlowbookAgent.make_and_run_agent(
            key="document-cell",
            model=model,
            instructions=instructions,
            output_type=CellDocumentation,
            input=prompt,
            log_dir="agent-logs",
        )

        return doc, stats

    async def _generate_cell_comment(
        self,
        cell: nbformat.NotebookNode,
        cell_id: str,
        dependencies: Dict[str, Any],
        model: str,
    ) -> Tuple[str, FlowbookStats]:
        """Generate a documentation comment for a cell."""
        if cell_id not in dependencies:
            return "", FlowbookStats(
                model=model,
                log_path="",
                time=0.0,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        deps = dependencies[cell_id]

        profile_data = FlowbookMetadata.from_cell(cell).get_profile()
        env = profile_data.env if profile_data else {}
        env_after = profile_data.env_after if profile_data else {}

        # Get cell source code
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # Get variables
        globals_read = sorted(deps.globals_read)
        globals_written = sorted(deps.globals_written)

        globals_written = [k for k in globals_written if is_valid_variable_name(k)]

        # Generate LLM documentation
        llm_doc, stats = await self._generate_llm_documentation(
            source, globals_written, globals_read, model
        )

        comment_lines = []
        comment_lines.append(self.DOC_START_MARKER)

        # Add LLM-generated title
        if llm_doc and llm_doc.title:
            comment_lines.append(f"# {llm_doc.title}")
        else:
            # Fallback to first line of code
            first_line = source.strip().split("\n")[0][:60]
            if len(source.strip().split("\n")[0]) > 60:
                first_line += "..."
            comment_lines.append(f"# Cell: {first_line}")

        # Add LLM-generated description
        if llm_doc and llm_doc.description:
            comment_lines.append("#")
            for bullet in llm_doc.description:
                # Wrap long bullets to fit within comment width
                bullet = bullet.strip()
                if not bullet.startswith("-") and not bullet.startswith("•"):
                    bullet = f"- {bullet}"
                comment_lines.append(f"# {bullet}")

        # List dependencies (variables read)
        if globals_read:
            comment_lines.append("#")
            comment_lines.append("# Global Inputs:")
            for var in globals_read:
                type_info = self._format_type_info(var, env)
                if type_info:
                    comment_lines.append(f"#   - {var}: {type_info}")
                else:
                    comment_lines.append(f"#   - {var}")

        # List outputs (variables written) with LLM descriptions
        if globals_written:
            comment_lines.append("#")
            comment_lines.append("# Global Outputs:")

            # Create a mapping of variable names to LLM descriptions
            var_desc_map = {}
            if llm_doc and llm_doc.variable_descriptions:
                var_desc_map = {
                    vd.variable_name: vd.description
                    for vd in llm_doc.variable_descriptions
                }

            for var in globals_written:
                type_info = self._format_type_info(var, env_after)
                llm_desc = var_desc_map.get(var, "")

                if type_info and llm_desc:
                    comment_lines.append(f"#   - {var}: {type_info} - {llm_desc}")
                elif type_info:
                    comment_lines.append(f"#   - {var}: {type_info}")
                elif llm_desc:
                    comment_lines.append(f"#   - {var} - {llm_desc}")
                else:
                    comment_lines.append(f"#   - {var}")

        comment_lines.append(self.DOC_END_MARKER)
        comment_lines.append("")  # Empty line after comment block
        comment_lines.append("")  # Empty line after comment block

        return "\n".join(comment_lines), stats

    def _remove_old_comment_from_source(self, source: str) -> str:
        """Remove old documentation comment from source code."""
        if self.DOC_START_MARKER not in source:
            return source

        lines = source.split("\n")
        # Find the start and end of the comment block
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() == self.DOC_START_MARKER and start_idx is None:
                start_idx = i
            elif (
                line.strip() == self.DOC_END_MARKER
                and start_idx is not None
                and end_idx is None
            ):
                end_idx = i
                return (
                    "\n".join(lines[:start_idx]).rstrip()
                    + "\n".join(lines[end_idx + 1 :]).lstrip()
                )

        return source

    async def document_cell(
        self,
        index: int,
        nb: nbformat.NotebookNode,
        dependencies_dict: Dict[str, Any],
        model: str,
    ) -> Tuple[str, DocumentationResultAndStats]:
        """Document a single cell and return the result with stats."""
        cells = nb["cells"]
        cell = cells[index]
        cell_id = cell["id"]

        # Generate documentation comment
        comment, stats = await self._generate_cell_comment(
            cell, cell_id, dependencies_dict, model
        )

        # Print progress for this cell
        status = "✓" if comment else "✗"
        tokens = stats.usage.total_tokens if stats.usage else 0
        print(
            f"| {index:<9}| {status:<9}| {tokens:<9}| {stats.time:<9.1f}| {stats.cost:<9.4f}|"
        )

        return cell_id, DocumentationResultAndStats(comment=comment, stats=stats)

    async def document_cells(
        self,
        nb: nbformat.NotebookNode,
        dependencies_dict: Dict[str, Any],
        model: str,
        cell_ids: Optional[List[str]] = None,
    ) -> Tuple[nbformat.NotebookNode, float]:
        """Document all code cells concurrently."""
        print()
        print("# Documenting Cells")
        print()

        # Clean old comments first
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                cleaned_source = self._remove_old_comment_from_source(source)
                cell["source"] = cleaned_source

        # Create tasks for cells to document
        tasks = []
        for index, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                source = "\n".join(
                    cell["source"]
                    if isinstance(cell["source"], list)
                    else [cell["source"]]
                )
                if source.strip():
                    # Only document selected cells if specified
                    if cell_ids is None or cell["id"] in cell_ids:
                        tasks.append(
                            self.document_cell(index, nb, dependencies_dict, model)
                        )

        # Print table header
        print(
            "|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|".format(
                "Index", "Status", "Tokens", "Time (s)", "Cost ($)"
            )
        )
        print("|{:-^10}|{:-^10}|{:-^10}|{:-^10}|{:-^10}|".format("", "", "", "", ""))

        # Execute all documentation tasks concurrently
        results = await asyncio.gather(*tasks)
        print()

        # Create new notebook with documentation
        new_nb: nbformat.NotebookNode = nb.copy()  # type: ignore
        cell_map = {cell["id"]: cell for cell in new_nb["cells"]}

        # Apply documentation to cells
        for cell_id, cell_result in results:
            with timer(message=f"Cell {cell_id}"):
                cell = cell_map.get(cell_id)
                assert cell is not None, f"Cell {cell_id} not found in notebook"

                if cell_result.comment:
                    # Get the cleaned source
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    # Add comment at the beginning
                    new_source = cell_result.comment + source
                    cell["source"] = new_source

        total_cost = sum([cell_result.stats.cost for _, cell_result in results])

        return new_nb, total_cost

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Add documentation to code cells as comments."""
        with self.timing_context() as get_elapsed:
            # Analyze dependencies for the entire notebook
            dependencies_dict = analyze_notebook(notebook_content)

            # Document cells concurrently
            new_nb, total_cost = await self.document_cells(
                notebook_content, dependencies_dict, config.model, selected_cell_ids
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
