"""
Analyze notebook command implementation.
"""

import copy
from typing import Any, Dict, Optional

from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.output import log


class AnalyzeNotebookCommand(NotebookCommand):
    """Analyzes notebook structure and content."""

    @property
    def command_name(self) -> str:
        return "analyze"

    @property
    def display_name(self) -> str:
        return "Analyze Notebook"

    @property
    def icon_name(self) -> str:
        return "ui-components:chart"

    @property
    def tooltip(self) -> str:
        return "Analyze notebook structure and statistics"

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[list] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze the notebook and return statistics."""

        log("Analyzing notebook...")

        cells = notebook_content.get("cells", [])

        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        raw_cells = [c for c in cells if c.get("cell_type") == "raw"]

        total_code_lines = sum(
            (
                len(cell.get("source", []))
                if isinstance(cell.get("source"), list)
                else len(cell.get("source", "").split("\n"))
            )
            for cell in code_cells
        )

        new_notebook = copy.deepcopy(notebook_content)

        analysis_text = f"""# Notebook Analysis Results

- **Total Cells**: {len(cells)}
- **Code Cells**: {len(code_cells)}
- **Markdown Cells**: {len(markdown_cells)}
- **Raw Cells**: {len(raw_cells)}
- **Total Lines of Code**: {total_code_lines}
- **Notebook Format**: {notebook_content.get('nbformat')}
- **Kernel**: {notebook_content.get('metadata', {}).get('kernelspec', {}).get('name', 'unknown')}
"""

        analysis_cell = {
            "cell_type": "markdown",
            "metadata": {"generated": True, "command": "analyze"},
            "source": analysis_text,
        }

        new_notebook["cells"].insert(0, analysis_cell)

        metadata = {
            "status": "success",
            "command": self.command_name,
            "analysis": {
                "total_cells": len(cells),
                "code_cells": len(code_cells),
                "markdown_cells": len(markdown_cells),
                "raw_cells": len(raw_cells),
                "total_code_lines": total_code_lines,
                "notebook_format": notebook_content.get("nbformat"),
                "kernel": notebook_content.get("metadata", {})
                .get("kernelspec", {})
                .get("name", "unknown"),
            },
        }

        return {"notebook": new_notebook, "metadata": metadata}

