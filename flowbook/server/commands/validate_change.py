"""
Validate Change command implementation.

Validates selected cells by comparing their code with the next cell's code,
using the current cell's output variables.
"""

import pprint
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from flowbook.kernel_support.checkpoint import is_valid_variable_name
from flowbook.kernel_support.kernel_command_client import KernelCommandClient
from flowbook.kernel_support.types import (
    MemoryCheckpointDiffResult, TestCodeResult, TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash,
    ExecutionError, format_diff_as_markdown
)
from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.util.dependencies import analyze_notebook, CellDependencies
from flowbook.util.output import log, timer


# Import CodeExecutionOrchestrator from optimize.py
from flowbook.server.commands.optimize import CodeExecutionOrchestrator


class ValidateChangeCommand(NotebookCommand):
    """Validates selected cells with next cell comparison."""

    @property
    def command_name(self) -> str:
        return "validate_change"

    @property
    def display_name(self) -> str:
        return "Validate Change"

    @property
    def icon_name(self) -> str:
        return "ui-components:check"

    @property
    def tooltip(self) -> str:
        return "Validate selected cells with next cell comparison"

    @property
    def requires_kernel(self) -> bool:
        return True

    def _get_next_cell_source(
        self,
        cells: List[Dict[str, Any]],
        current_cell_id: str
    ) -> str:
        """
        Get the next cell's source code.

        Args:
            cells: List of all notebook cells
            current_cell_id: ID of the current cell

        Returns:
            Next cell's source code as string, or empty string if no next cell
        """
        # Find current cell index
        current_index = None
        for idx, cell in enumerate(cells):
            if cell.get("id") == current_cell_id:
                current_index = idx
                break

        # Check if next cell exists
        if current_index is None or current_index + 1 >= len(cells):
            return ""

        # Get next cell
        next_cell = cells[current_index + 1]

        # Only use code cells
        if next_cell.get("cell_type") != "code":
            return ""

        next_source = next_cell.get("source", "")

        # Handle source as list or string
        if isinstance(next_source, list):
            next_source = "".join(next_source)

        return next_source

    def _get_cell_output_variables(
        self,
        dependencies_dict: Dict[str, CellDependencies],
        cell_id: str
    ) -> List[str]:
        """
        Get filtered output variables (globals_written) for a cell.

        Args:
            dependencies_dict: Dictionary mapping cell_id to dependencies
            cell_id: ID of the cell

        Returns:
            List of variable names written by the cell, filtered
        """
        if cell_id not in dependencies_dict:
            return []

        deps = dependencies_dict[cell_id]

        # Filter out private and system variables
        globals_written = [var for var in deps.globals_written if is_valid_variable_name(var)]

        return sorted(globals_written)

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Validate selected cells with next cell comparison.

        Args:
            notebook_content: The notebook content
            kernel_client: The kernel client
            selected_cell_ids: List of selected cell IDs (required)

        Returns:
            ProcessingResult with notebook and metadata containing per-cell results
        """
        with self.timing_context() as get_elapsed:
            if kernel_client is None:
                total_time = get_elapsed()
                return ProcessingResult(
                    notebook=notebook_content,
                    metadata={
                        "status": "error",
                        "command": self.command_name,
                        "error": "Kernel client required but not provided",
                    },
                    total_cost=0.0,
                    total_time=total_time
                )

            # If no cells selected, do no work
            if not selected_cell_ids:
                log("[No cells selected - no work to do]")
                total_time = get_elapsed()
                return ProcessingResult(
                    notebook=notebook_content,
                    metadata={
                        "status": "success",
                        "command": self.command_name,
                        "results": {},
                        "total_processed": 0,
                    },
                    total_cost=0.0,
                    total_time=total_time
                )

            cells = notebook_content.get("cells", [])
            results = {}
            total_processed = 0

            # Analyze dependencies for the entire notebook once
            with timer(key="validate:analyze_deps", message="Analyzing notebook dependencies"):
                dependencies_dict = analyze_notebook(notebook_content)

            log(f"Validating {len(selected_cell_ids)} selected cell(s)...")

            # Process each cell
            with timer(key="validate:cells", message="Validating cells"):
                for idx, cell in enumerate(cells):
                    cell_id = cell.get("id")

                    # Skip if not a code cell
                    if cell.get("cell_type") != "code":
                        continue

                    # Skip if not in selected cells
                    if cell_id not in selected_cell_ids:
                        continue

                    with timer(key=f"validate:cell_{idx}", message=f"Validating cell {idx}:{cell_id}"):
                        # Get current cell source
                        source = cell.get("source", "")
                        if isinstance(source, list):
                            source = "".join(source)

                        # Get next cell source
                        next_source = self._get_next_cell_source(cells, cell_id)

                        # Get output variables from dependencies
                        output_variables = self._get_cell_output_variables(
                            dependencies_dict, cell_id
                        )

                        try:
                            # Create orchestrator and run test
                            orchestrator = CodeExecutionOrchestrator(kernel_client)
                            result = orchestrator.test_code(
                                original_code=source,
                                modified_code=next_source,
                                output_variables=set(output_variables)
                            )

                            # Store result
                            results[cell_id] = {
                                "ok": True,  # Always true - result type discriminates success/failure
                                "result": result.model_dump(),
                                "error": None,
                            }

                            # Log result based on result type
                            if isinstance(result, TestCodeSuccess):
                                # Both codes succeeded - show diff
                                status_str = "✓" if not result.diff.differences else "✗"
                                log(f"[{status_str}] Cell {idx}: {format_diff_as_markdown(result.diff)}")
                            elif isinstance(result, TestCodeOriginalCrash):
                                # Original code crashed
                                error = result.error
                                log(f"[✗] Cell {idx}: Original code crashed - {error.error_type}: {error.error_message}")
                            elif isinstance(result, TestCodeModifiedCrash):
                                # Modified code crashed
                                error = result.error
                                log(f"[✗] Cell {idx}: Modified code crashed - {error.error_type}: {error.error_message}")
                            else:
                                log(f"[?] Cell {idx}: Unknown result type")

                            total_processed += 1

                        except Exception as e:
                            error_details = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                            log(f"[✗] Cell {idx}: Error - {type(e).__name__}: {str(e)}")
                            log(f"Traceback:\n{traceback.format_exc()}")
                            results[cell_id] = {
                                "ok": False,
                                "result": None,
                                "error": error_details,
                            }

            # Return metadata with per-cell results
            metadata = {
                "status": "success",
                "command": self.command_name,
                "results": results,
                "total_processed": total_processed,
            }

            log(f"[Completed: {total_processed} cell(s) validated]")

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=notebook_content,
            metadata=metadata,
            total_cost=0.0,
            total_time=total_time
        )
