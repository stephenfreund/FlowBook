"""
Execute all cells command implementation.
"""

import copy
import sys
import traceback
from typing import Any, Dict, Optional

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.util.ferret_metadata import FerretMetadata, ProfileData, set_profile_ferret_metadata
from data_ferret.util.metadata_extractor import extract_and_set_metadata
from data_ferret.server.kernel_helper import KernelHelper
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.output import log, timer


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

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[list] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Execute all code cells."""
        if kernel_client is None:
            return ProcessingResult(
                notebook=notebook_content,
                metadata={
                    "status": "error",
                    "command": self.command_name,
                    "error": "Kernel client required but not provided",
                },
                total_cost=0.0,
                total_time=0.0
            )

        with self.timing_context() as get_elapsed:
            new_notebook = copy.deepcopy(notebook_content)
            cells = new_notebook.get("cells", [])

            execution_results = []
            total_executed = 0
            status = "success"  # Track actual execution status

            kernel_client.execute("%enable_scalene")

            with timer(key="execute_all", message="Executing all cells"):
                for idx, cell in enumerate(cells):
                    if cell.get("cell_type") == "code":
                        if selected_cell_ids and cell.get("id") not in selected_cell_ids:
                            continue

                        with timer(key="execute_cell", message=f"Executing cell {idx}:{cell.get('id')}"):
                            source = cell.get("source", "")
                            if isinstance(source, list):
                                source = "".join(source)

                            metadata = cell.get("metadata", {}).copy()
                            metadata['cell_id'] = cell.get("id")

                            if source.strip():
                                try:
                                    # Use 30-minute timeout to match kernel timeout
                                    with self.timing_context() as cell_get_elapsed:
                                        result = KernelHelper.execute_code(
                                            kernel_client,
                                            source,
                                            timeout=30 * 60,  # 30 minutes
                                            cell_id=cell.get("id"),
                                            cell_metadata=metadata,
                                        )   

                                    cell["execution_count"] = result["execution_count"]
                                    cell["outputs"] = result["outputs"]

                                    # Check for execution errors
                                    if result["status"] == "error":
                                        status = "error"
                                        error_message = result["error_message"]
                                        print()
                                        print(f"--------------------------------")
                                        print(f"{error_message}")
                                        print(f"--------------------------------")
                                        execution_results.append(
                                            {
                                                "cell_index": idx,
                                                "status": "error",
                                                "execution_count": result["execution_count"],
                                                "error_message": error_message,
                                                "execution_time": cell_get_elapsed() * 1000,
                                            }
                                        )
                                        break  # Stop on first error

                                    # Extract all metadata types using generic extractor
                                    extract_and_set_metadata(cell, result["outputs"])

                                    execution_results.append(
                                        {
                                            "cell_index": idx,
                                            "status": result["status"],
                                            "execution_count": result["execution_count"],
                                            "execution_time": cell_get_elapsed() * 1000,
                                        }
                                    )
                                    log(f"[{result['execution_count']}]")

                                    total_executed += 1

                                except Exception as e:
                                    # Catch any Python exceptions during execution
                                    cell["outputs"] = [
                                        {
                                            "output_type": "error",
                                            "ename": e.__class__.__name__,
                                            "evalue": str(e),
                                            "traceback": traceback.format_exception(type(e), e, e.__traceback__),
                                        }
                                    ]

                                    execution_results.append(
                                        {
                                            "cell_index": idx,
                                            "status": "error",
                                            "execution_time": cell_get_elapsed() * 1000,
                                        }
                                    )
                                    status = "error"
                                    break  # Stop on exception

            metadata = {
                "status": status,  # Use tracked status instead of hardcoded "success"
                "command": self.command_name,
                "execution": {
                    "total_executed": total_executed,
                    "results": execution_results,
                },
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_notebook,
            metadata=metadata,
            total_cost=0.0,
            total_time=total_time
        )

