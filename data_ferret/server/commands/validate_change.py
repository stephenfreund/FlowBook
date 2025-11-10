"""
Validate Change command implementation.

Validates selected cells by comparing their code with the next cell's code,
using the current cell's output variables.
"""

import pprint
import traceback
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from data_ferret.kernel.checkpoint import is_valid_variable_name
from data_ferret.kernel.types import (
    DiffResult, TestCodeResult, TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash,
    format_diff_as_markdown
)
from data_ferret.server.base import NotebookCommand
from data_ferret.server.kernel_manager import FerretKernelClient, TestCodeData
from data_ferret.util.dependencies import analyze_notebook, CellDependencies
from data_ferret.util.output import log, timer


class TestCodeRequest(BaseModel):
    """Request model for test_code comm message."""
    original_code: str = Field(..., description="The original cell's code")
    modified_code: str = Field(..., description="The modified cell's code")
    output_variables: List[str] = Field(..., description="List of variable names to compare")


class TestCodeResponse(BaseModel):
    """Response model for test_code comm message."""
    ok: bool = Field(..., description="Whether the test succeeded")
    result: Optional[TestCodeResult] = Field(None, description="The test code result with timing info if successful")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        arbitrary_types_allowed = True


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

    def _send_test_code_comm(
        self,
        kernel_client: FerretKernelClient,
        original_code: str,
        modified_code: str,
        output_variables: List[str]
    ) -> TestCodeData:
        """
        Send test_code comm message to kernel and return response.

        Uses the base class _send_comm_message method with type-safe
        Pydantic models for request and response.

        Args:
            kernel_client: The kernel client to send the message to
            original_code: The original cell's code
            modified_code: The modified (next) cell's code
            output_variables: List of variable names to compare

        Returns:
            TestCodeData with ok and result/error fields
        """
        # Create type-safe request model
        request = TestCodeRequest(
            original_code=original_code,
            modified_code=modified_code,
            output_variables=output_variables
        )

        # Send comm and receive validated response
        response: TestCodeResponse = self._send_comm_message(
            kernel_client,
            target_name="test_code",
            request=request,
            response_type=TestCodeResponse
        )

        # Extract result from validated response (with full IDE autocomplete!)
        result = response.result if response.ok else response.error
        return TestCodeData(ok=response.ok, result=result)

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
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate selected cells with next cell comparison.

        Args:
            notebook_content: The notebook content
            kernel_client: The kernel client
            selected_cell_ids: List of selected cell IDs (required)

        Returns:
            Dictionary with notebook and metadata containing per-cell results
        """
        if kernel_client is None:
            return {
                "notebook": notebook_content,
                "metadata": {
                    "status": "error",
                    "command": self.command_name,
                    "error": "Kernel client required but not provided",
                },
            }

        # If no cells selected, do no work
        if not selected_cell_ids:
            log("[No cells selected - no work to do]")
            return {
                "notebook": notebook_content,
                "metadata": {
                    "status": "success",
                    "command": self.command_name,
                    "results": {},
                    "total_processed": 0,
                },
            }

        cells = notebook_content.get("cells", [])
        results = {}
        total_processed = 0

        # Analyze dependencies for the entire notebook once
        with timer(key="analyze_dependencies", message="Analyzing notebook dependencies"):
            dependencies_dict = analyze_notebook(notebook_content)

        log(f"Validating {len(selected_cell_ids)} selected cell(s)...")

        # Process each cell
        with timer(key="validate_cells", message="Validating cells"):
            for idx, cell in enumerate(cells):
                cell_id = cell.get("id")

                # Skip if not a code cell
                if cell.get("cell_type") != "code":
                    continue

                # Skip if not in selected cells
                if cell_id not in selected_cell_ids:
                    continue

                with timer(key=f"validate_cell_{idx}", message=f"Validating cell {idx}:{cell_id}"):
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
                        # Send test_code comm message
                        result = self._send_test_code_comm(
                            kernel_client,
                            original_code=source,
                            modified_code=next_source,
                            output_variables=output_variables
                        )

                        # Store result
                        results[cell_id] = {
                            "ok": result.ok,
                            "result": result.result.model_dump() if result.ok and result.result else None,
                            "error": result.result if not result.ok else None,
                        }

                        # Log result based on result type
                        if result.ok and result.result:
                            if isinstance(result.result, TestCodeSuccess):
                                # Both codes succeeded - show diff
                                status_str = "✓" if not result.result.diff.differences else "✗"
                                log(f"[{status_str}] Cell {idx}: {format_diff_as_markdown(result.result.diff)}")
                            elif isinstance(result.result, TestCodeOriginalCrash):
                                # Original code crashed
                                error = result.result.error
                                log(f"[✗] Cell {idx}: Original code crashed - {error.error_type}: {error.error_message}")
                            elif isinstance(result.result, TestCodeModifiedCrash):
                                # Modified code crashed
                                error = result.result.error
                                log(f"[✗] Cell {idx}: Modified code crashed - {error.error_type}: {error.error_message}")
                            else:
                                log(f"[?] Cell {idx}: Unknown result type")
                        else:
                            log(f"[✗] Cell {idx}: {result.result}")

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

        return {"notebook": notebook_content, "metadata": metadata}
