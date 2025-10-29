"""
Document command implementation.

Adds descriptive comments to code cells based on dependency analysis and profiling data.
"""

import copy
from typing import Any, Dict, Optional, List

import nbformat
from pydantic import BaseModel, Field

from data_ferret.agent.agent import FerretAgent
from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.dependencies import analyze_notebook
from data_ferret.util.ferret_metadata import FerretMetadata
from data_ferret.util.output import log
from data_ferret.util.prompts import get_prompt


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
    ) -> Optional[CellDocumentation]:
        """Use LLM to generate documentation for a code cell."""
        try:
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

            doc, stats = await FerretAgent.make_and_run_agent(
                key="document-cell",
                model="gpt-4o-mini",
                instructions=instructions,
                output_type=CellDocumentation,
                input=prompt,
                log_dir="agent-logs",
            )

            log(f"LLM documentation generated (cost: ${stats.cost:.4f})")
            return doc
        except Exception as e:
            log(f"Failed to generate LLM documentation: {e}")
            return None

    async def _generate_cell_comment(
        self,
        cell: nbformat.NotebookNode,
        cell_id: str,
        dependencies: Dict[str, Any],
    ) -> str:
        """Generate a documentation comment for a cell."""
        if cell_id not in dependencies:
            return ""

        deps = dependencies[cell_id]

        profile_data = FerretMetadata.from_cell(cell).get_profile()
        env = profile_data.env if profile_data else {}
        env_after = profile_data.env_after if profile_data else {}

        # Get cell source code
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # Get variables
        globals_read = sorted(deps.globals_read)
        globals_written = sorted(deps.globals_written)

        globals_written = [
            k
            for k in globals_written
            if not k.startswith("_")
            and k not in ("get_ipython", "In", "Out", "exit", "quit")
        ]

        # Generate LLM documentation
        llm_doc = await self._generate_llm_documentation(
            source, globals_written, globals_read
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

        return "\n".join(comment_lines)

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

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add documentation to code cells as comments."""
        log("Documenting code cells with comments")

        # Analyze dependencies for the entire notebook
        dependencies_dict = analyze_notebook(notebook_content)

        # Create new notebook with documented cells
        new_nb = copy.deepcopy(notebook_content)

        cells_documented = 0

        for cell in new_nb["cells"]:
            if cell.get("cell_type") == "code":
                cell_id = cell.get("id", "")
                source = cell.get("source", "")

                if isinstance(source, list):
                    source = "".join(source)

                # Skip if no source or if not in selected cells
                if not source.strip():
                    continue

                if selected_cell_ids and cell_id not in selected_cell_ids:
                    continue

                cleaned_source = self._remove_old_comment_from_source(source)
                cell["source"] = cleaned_source
                # Generate comment
                comment = await self._generate_cell_comment(
                    cell, cell_id, dependencies_dict
                )

                if comment:
                    # Add new comment at the beginning
                    new_source = comment + cleaned_source
                    cell["source"] = new_source
                    cells_documented += 1

        log(f"Documented {cells_documented} cells")

        metadata = {
            "status": "success",
            "command": self.command_name,
            "cells_documented": cells_documented,
        }

        return {"notebook": new_nb, "metadata": metadata}
