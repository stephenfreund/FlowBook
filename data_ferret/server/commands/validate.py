"""
Validate notebook command implementation.
"""

import copy
from typing import Any, Dict, Optional

from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_manager import FerretKernelClient


class ValidateNotebookCommand(NotebookCommand):
    """Validates notebook structure and checks for common issues."""

    @property
    def command_name(self) -> str:
        return "validate"

    @property
    def display_name(self) -> str:
        return "Validate Notebook"

    @property
    def icon_name(self) -> str:
        return "ui-components:check"

    @property
    def tooltip(self) -> str:
        return "Validate notebook structure and check for issues"

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[list] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Validate the notebook structure."""
        issues = []
        warnings = []

        if "cells" not in notebook_content:
            issues.append("Missing 'cells' field")

        if "metadata" not in notebook_content:
            warnings.append("Missing 'metadata' field")

        if "nbformat" not in notebook_content:
            issues.append("Missing 'nbformat' field")

        cells = notebook_content.get("cells", [])
        for idx, cell in enumerate(cells):
            if "cell_type" not in cell:
                issues.append(f"Cell {idx} missing 'cell_type'")

            if "source" not in cell:
                issues.append(f"Cell {idx} missing 'source'")

        empty_code_cells = [
            idx
            for idx, cell in enumerate(cells)
            if cell.get("cell_type") == "code" and not cell.get("source")
        ]

        if empty_code_cells:
            warnings.append(f"Found {len(empty_code_cells)} empty code cells")

        is_valid = len(issues) == 0

        new_notebook = copy.deepcopy(notebook_content)

        status_emoji = "✅" if is_valid else "❌"
        validation_text = f"""# Validation Results {status_emoji}

**Status**: {"Valid" if is_valid else "Invalid"}

"""

        if issues:
            validation_text += (
                "## Issues\n" + "\n".join(f"- {issue}" for issue in issues) + "\n\n"
            )

        if warnings:
            validation_text += (
                "## Warnings\n"
                + "\n".join(f"- {warning}" for warning in warnings)
                + "\n\n"
            )

        if not issues and not warnings:
            validation_text += (
                "No issues or warnings found. Notebook structure is valid!\n"
            )

        validation_cell = {
            "cell_type": "markdown",
            "metadata": {"generated": True, "command": "validate"},
            "source": validation_text,
        }

        new_notebook["cells"].insert(0, validation_cell)

        metadata = {
            "status": "success",
            "command": self.command_name,
            "validation": {
                "is_valid": is_valid,
                "issues": issues,
                "warnings": warnings,
                "cells_checked": len(cells),
            },
        }

        return {"notebook": new_notebook, "metadata": metadata}

