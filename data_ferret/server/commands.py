"""
Built-in command implementations for notebook processing.
"""

import os
from typing import Any, Dict, Optional
import json
import copy

from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_helper import KernelHelper
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.server.message_broadcaster import get_broadcaster
from data_ferret.util.output import log, timer

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

    def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
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

    def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
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


class ExecuteAllCommand(NotebookCommand):
    """Executes all code cells in the notebook using the kernel."""

    @property
    def command_name(self) -> str:
        return "execute_all"

    @property
    def display_name(self) -> str:
        return "Execute All Cells"

    @property
    def icon_name(self) -> str:
        return "ui-components:run"

    @property
    def tooltip(self) -> str:
        return "Execute all code cells and capture outputs"

    @property
    def requires_kernel(self) -> bool:
        return True

    def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute all code cells."""
        if kernel_client is None:
            return {
                "notebook": notebook_content,
                "metadata": {
                    "status": "error",
                    "command": self.command_name,
                    "error": "Kernel client required but not provided",
                },
            }

        new_notebook = copy.deepcopy(notebook_content)
        cells = new_notebook.get("cells", [])

        execution_results = []
        total_executed = 0

        with timer(key="execute_all", message="Executing all cells"):
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    with timer(key="execute_cell", message=f"Executing cell {idx}:{cell.get('id')}"):
                        source = cell.get("source", "")
                        if isinstance(source, list):
                            source = "".join(source)

                        if source.strip():
                            result = KernelHelper.execute_code(
                                kernel_client,
                                source,
                                cell_id=cell.get("id"),
                                cell_metadata=cell.get("metadata"),
                            )

                            cell["execution_count"] = result["execution_count"]
                            cell["outputs"] = result["outputs"]

                            execution_results.append(
                                {
                                    "cell_index": idx,
                                    "status": result["status"],
                                    "execution_count": result["execution_count"],
                                }
                            )
                            log(f"[{result['execution_count']}]")

                            total_executed += 1

        metadata = {
            "status": "success",
            "command": self.command_name,
            "execution": {
                "total_executed": total_executed,
                "results": execution_results,
            },
        }

        return {"notebook": new_notebook, "metadata": metadata}


class InspectVariablesCommand(NotebookCommand):
    """Inspects variables in the kernel namespace."""

    @property
    def command_name(self) -> str:
        return "inspect_vars"

    @property
    def display_name(self) -> str:
        return "Inspect Variables"

    @property
    def icon_name(self) -> str:
        return "ui-components:inspect"

    @property
    def tooltip(self) -> str:
        return "Inspect variables in the kernel namespace"

    @property
    def requires_kernel(self) -> bool:
        return True

    def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Inspect kernel variables."""
        print("KERNEL CLIENT", kernel_client)
        if kernel_client is None:
            return {
                "notebook": notebook_content,
                "metadata": {
                    "status": "error",
                    "command": self.command_name,
                    "error": "Kernel client required but not provided",
                },
            }

        inspect_code = """
import json
import sys

def get_variable_info():
    vars_info = []
    for name in dir():
        if not name.startswith('_'):
            try:
                obj = eval(name)
                vars_info.append({
                    'name': name,
                    'type': type(obj).__name__,
                    'repr': repr(obj)[:100]
                })
            except:
                pass
    return vars_info

print(json.dumps(get_variable_info()))
"""

        print("INSPECT CODE", inspect_code)

        result = KernelHelper.execute_code(kernel_client, inspect_code)

        print("RESULT", result)

        variables = []
        if result["status"] == "ok" and result["outputs"]:
            for output in result["outputs"]:
                if output["output_type"] == "stream" and output["name"] == "stdout":
                    try:
                        variables = json.loads(output["text"])
                    except:
                        pass

        print("VARIABLES", variables)

        new_notebook = copy.deepcopy(notebook_content)

        print("NEW NOTEBOOK", new_notebook)

        if variables:
            var_table = "| Variable | Type | Value |\n|----------|------|-------|\n"
            for var in variables:
                var_table += f"| {var['name']} | {var['type']} | {var['repr']} |\n"

            report_text = f"""# Variable Inspector

{var_table}

Total variables: {len(variables)}
"""
        else:
            report_text = (
                "# Variable Inspector\n\nNo variables found in kernel namespace."
            )

        report_cell = {
            "cell_type": "markdown",
            "metadata": {"generated": True, "command": "inspect_vars"},
            "source": report_text,
        }

        new_notebook["cells"].insert(0, report_cell)

        metadata = {
            "status": "success",
            "command": self.command_name,
            "variables": variables,
        }

        print("METADATA", metadata)

        return {"notebook": new_notebook, "metadata": metadata}
