"""
Document command implementation.

Adds descriptive comments to code cells based on dependency analysis and profiling data.
"""

import copy
from typing import Any, Dict, Optional, List

import nbformat

from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.dependencies import analyze_notebook
from data_ferret.util.ferret_metadata import FerretMetadata
from data_ferret.util.output import log


class DocumentCommand(NotebookCommand):
    """Adds descriptive documentation comments to code cells."""

    # Unique markers for documentation comments
    DOC_START_MARKER = "# ======================================================================"
    DOC_END_MARKER =   "# ======================================================================"

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

    def _generate_cell_comment(
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

        comment_lines = []
        comment_lines.append(self.DOC_START_MARKER)

        # Get cell description from source - try to infer what it does
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # Generate a brief description
        first_line = source.strip().split("\n")[0][:60]
        if len(source.strip().split("\n")[0]) > 60:
            first_line += "..."
        comment_lines.append(f"# Cell: {first_line}")

        # List dependencies (variables read)
        globals_read = sorted(deps.globals_read)
        if globals_read:
            comment_lines.append("#")
            comment_lines.append("# Dependencies (reads):")
            for var in globals_read:
                type_info = self._format_type_info(var, env)
                if type_info:
                    comment_lines.append(f"#   - {var}: {type_info}")
                else:
                    comment_lines.append(f"#   - {var}")

        # List outputs (variables written)
        globals_written = sorted(deps.globals_written)
        if globals_written:
            comment_lines.append("#")
            comment_lines.append("# Outputs (writes):")
            for var in globals_written:
                type_info = self._format_type_info(var, env_after)
                if type_info:
                    comment_lines.append(f"#   - {var}: {type_info}")
                else:
                    comment_lines.append(f"#   - {var}")

        # List functions called
        if deps.functions_called:
            comment_lines.append("#")
            comment_lines.append("# Functions called:")
            for func in sorted(deps.functions_called):
                comment_lines.append(f"#   - {func}()")

        # List functions defined
        if deps.functions_defined:
            comment_lines.append("#")
            comment_lines.append("# Functions defined:")
            for func in sorted(deps.functions_defined):
                comment_lines.append(f"#   - {func}()")

        # List classes defined
        if deps.classes_defined:
            comment_lines.append("#")
            comment_lines.append("# Classes defined:")
            for cls in sorted(deps.classes_defined):
                comment_lines.append(f"#   - {cls}")

        comment_lines.append(self.DOC_END_MARKER)
        comment_lines.append("")  # Empty line after comment block

        return "\n".join(comment_lines)

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Add documentation comments to code cells."""
        log("Documenting code cells")

        # Analyze dependencies for the entire notebook
        dependencies_dict = analyze_notebook(notebook_content)


        # Create new notebook with documented cells
        new_nb = copy.deepcopy(notebook_content)

        cells_documented = 0
        for cell in new_nb.get("cells", []):
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
                comment = self._generate_cell_comment(
                    cell, cell_id, dependencies_dict
                )

                if comment:
                    # Check if cell already has our documentation comment
                    if self.DOC_START_MARKER in source:
                        # Remove old documentation comment
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
                            lines = lines[:start_idx] + lines[end_idx + 2:]
                            source = "\n".join(lines)

                    # Add new comment at the beginning
                    new_source = comment + source
                    cell["source"] = new_source
                    cells_documented += 1

        log(f"Documented {cells_documented} cells")

        metadata = {
            "status": "success",
            "command": self.command_name,
            "cells_documented": cells_documented,
        }

        return {"notebook": new_nb, "metadata": metadata}
