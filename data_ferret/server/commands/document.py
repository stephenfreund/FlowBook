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


class VariableDescription(BaseModel):
    """Description of a variable."""
    variable_name: str = Field(description="The name of the variable")
    description: str = Field(description="What the variable captures or represents")


class CellDocumentation(BaseModel):
    """LLM-generated documentation for a code cell."""
    title: str = Field(description="A concise title describing what the cell does")
    description: List[str] = Field(description="Bullet points describing the cell's functionality")
    variable_descriptions: List[VariableDescription] = Field(
        description="Descriptions of what each output variable captures",
        default_factory=list
    )


class DocumentCommand(NotebookCommand):
    """Adds descriptive documentation comments to code cells."""

    # Documentation mode: "comment" or "markdown"
    DOCUMENTATION_MODE = "markdown"  # Change to "comment" for inline comments

    # Unique markers for documentation comments
    DOC_START_MARKER = "# ======================================================================"
    DOC_END_MARKER =   "# ======================================================================"

    # Marker for generated markdown cells
    MARKDOWN_MARKER = "<!-- DataFerret Generated Documentation -->"

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
            # Build a prompt that includes the code and output variables
            prompt = f"""Analyze the following Python code and provide documentation:

Code:
```python
{source_code}
```

Variables read (inputs): {', '.join(globals_read) if globals_read else 'None'}
Variables written (outputs): {', '.join(globals_written) if globals_written else 'None'}

Provide:
1. A concise title (max 60 characters) describing what this cell does
2. A bullet list (2-4 bullets) describing the functionality
3. For each variable in the outputs list, describe what it captures or represents

Be specific and technical. Focus on what the code accomplishes, not how it does it."""

            doc, stats = await FerretAgent.make_and_run_agent(
                key="document-cell",
                model="gpt-4o-mini",
                instructions="You are a technical documentation assistant. Analyze code and provide clear, concise documentation.",
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

        return "\n".join(comment_lines)

    async def _generate_markdown_cell_content(
        self,
        cell: nbformat.NotebookNode,
        cell_id: str,
        dependencies: Dict[str, Any],
    ) -> str:
        """Generate markdown documentation for a cell."""
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

        # Generate LLM documentation
        llm_doc = await self._generate_llm_documentation(
            source, globals_written, globals_read
        )

        markdown_lines = []

        # Add marker at the top (hidden in rendered markdown)
        markdown_lines.append(self.MARKDOWN_MARKER)
        markdown_lines.append("")

        # Add LLM-generated title
        if llm_doc and llm_doc.title:
            markdown_lines.append(f"## {llm_doc.title}")
        else:
            # Fallback to first line of code
            first_line = source.strip().split("\n")[0][:60]
            if len(source.strip().split("\n")[0]) > 60:
                first_line += "..."
            markdown_lines.append(f"## {first_line}")

        # Add LLM-generated description
        if llm_doc and llm_doc.description:
            markdown_lines.append("")
            for bullet in llm_doc.description:
                bullet = bullet.strip()
                if not bullet.startswith("-") and not bullet.startswith("•"):
                    bullet = f"- {bullet}"
                markdown_lines.append(bullet)

        # List dependencies (variables read)
        if globals_read:
            markdown_lines.append("")
            markdown_lines.append("**Global Inputs:**")
            for var in globals_read:
                type_info = self._format_type_info(var, env)
                if type_info:
                    markdown_lines.append(f"- `{var}`: {type_info}")
                else:
                    markdown_lines.append(f"- `{var}`")

        # List outputs (variables written) with LLM descriptions
        if globals_written:
            markdown_lines.append("")
            markdown_lines.append("**Global Outputs:**")

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
                    markdown_lines.append(f"- `{var}`: {type_info} — {llm_desc}")
                elif type_info:
                    markdown_lines.append(f"- `{var}`: {type_info}")
                elif llm_desc:
                    markdown_lines.append(f"- `{var}` — {llm_desc}")
                else:
                    markdown_lines.append(f"- `{var}`")

        return "\n".join(markdown_lines)

    def _remove_old_comment_from_source(self, source: str) -> str:
        """Remove old documentation comment from source code."""
        if self.DOC_START_MARKER not in source:
            return source

        lines = source.split("\n")
        # Find the start and end of the comment block
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() == self.DOC_START_MARKER:
                start_idx = i
            elif line.strip() == self.DOC_END_MARKER and start_idx is not None:
                end_idx = i
                break

        if start_idx is not None and end_idx is not None:
            # Remove old comment (including the empty line after)
            # Handle case where there might not be an empty line after
            if end_idx + 1 < len(lines) and lines[end_idx + 1].strip() == "":
                lines = lines[:start_idx] + lines[end_idx + 2:]
            else:
                lines = lines[:start_idx] + lines[end_idx + 1:]
            return "\n".join(lines)

        return source

    def _is_generated_markdown_cell(self, cell: nbformat.NotebookNode) -> bool:
        """Check if a cell is a generated markdown documentation cell."""
        if cell.get("cell_type") != "markdown":
            return False

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        return source.strip().startswith(self.MARKDOWN_MARKER)

    def _remove_generated_markdown_cells(
        self, cells: List[nbformat.NotebookNode]
    ) -> List[nbformat.NotebookNode]:
        """Remove all generated markdown documentation cells."""
        return [cell for cell in cells if not self._is_generated_markdown_cell(cell)]

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add documentation to code cells as comments or markdown cells."""
        mode = self.DOCUMENTATION_MODE
        log(f"Documenting code cells in '{mode}' mode")

        # Analyze dependencies for the entire notebook
        dependencies_dict = analyze_notebook(notebook_content)

        # Create new notebook with documented cells
        new_nb = copy.deepcopy(notebook_content)

        # Step 1: Clean up ALL old documentation (both comments and markdown cells)
        # Remove generated markdown cells
        new_nb["cells"] = self._remove_generated_markdown_cells(new_nb["cells"])

        # Remove old comments from all code cells
        for cell in new_nb["cells"]:
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                cleaned_source = self._remove_old_comment_from_source(source)
                cell["source"] = cleaned_source

        # Step 2: Add new documentation based on mode
        cells_documented = 0

        if mode == "comment":
            # Comment mode: Add comments to code cells
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

                    # Generate comment
                    comment = await self._generate_cell_comment(
                        cell, cell_id, dependencies_dict
                    )

                    if comment:
                        # Add new comment at the beginning
                        new_source = comment + source
                        cell["source"] = new_source
                        cells_documented += 1

        elif mode == "markdown":
            # Markdown mode: Insert markdown cells before code cells
            new_cells = []
            for cell in new_nb["cells"]:
                if cell.get("cell_type") == "code":
                    cell_id = cell.get("id", "")
                    source = cell.get("source", "")

                    if isinstance(source, list):
                        source = "".join(source)

                    # Check if should document this cell
                    should_document = (
                        source.strip()
                        and (not selected_cell_ids or cell_id in selected_cell_ids)
                    )

                    if should_document:
                        # Generate markdown documentation
                        markdown_content = await self._generate_markdown_cell_content(
                            cell, cell_id, dependencies_dict
                        )

                        if markdown_content:
                            # Create a new markdown cell
                            markdown_cell = nbformat.v4.new_markdown_cell(
                                source=markdown_content
                            )
                            new_cells.append(markdown_cell)
                            cells_documented += 1

                # Add the original cell (code or otherwise)
                new_cells.append(cell)

            new_nb["cells"] = new_cells

        else:
            log(f"Unknown documentation mode: {mode}")

        log(f"Documented {cells_documented} cells")

        metadata = {
            "status": "success",
            "command": self.command_name,
            "cells_documented": cells_documented,
            "mode": mode,
        }

        return {"notebook": new_nb, "metadata": metadata}
