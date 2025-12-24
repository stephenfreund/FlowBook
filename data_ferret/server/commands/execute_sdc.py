"""
Execute all cells command with Sequential Dataflow Consistency enforcement.
"""

import copy
import traceback
from typing import Any, Dict, List, Optional

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.server.kernel_helper import KernelHelper
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.metadata_extractor import extract_and_set_metadata
from data_ferret.util.output import error, log, timer


class ExecuteSDCCommand(NotebookCommand):
    """
    Execute notebook cells with Sequential Dataflow Consistency enforcement.

    Uses FerretSDCKernel to track variable dependencies and enforce
    that cells don't modify state read by earlier cells.
    """

    @property
    def command_name(self) -> str:
        return "execute_sdc"

    @property
    def display_name(self) -> str:
        return "Execute with SDC"

    @property
    def icon_name(self) -> str:
        return "ui-components:check"

    @property
    def tooltip(self) -> str:
        return "Execute cells with Sequential Dataflow Consistency enforcement"

    @property
    def requires_kernel(self) -> bool:
        return True

    @property
    def kernel_name(self) -> str:
        return "ferret_sdc_kernel"

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Execute notebook with SDC enforcement.

        Args:
            notebook_content: Notebook JSON
            kernel_client: FerretKernelClient instance (or FerretSDCKernelClient)
            selected_cell_ids: Optional list of cell IDs to execute
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            ProcessingResult with:
                - notebook: Updated notebook with outputs
                - metadata: Execution metadata including SDC info
        """
        if kernel_client is None:
            return ProcessingResult(
                notebook=notebook_content,
                metadata={
                    "status": "error",
                    "command": self.command_name,
                    "error": "Kernel client required but not provided",
                },
                total_cost=0.0,
                total_time=0.0,
            )

        with self.timing_context() as get_elapsed:
            new_notebook = copy.deepcopy(notebook_content)
            cells = new_notebook.get("cells", [])

            # Extract cell order for SDC
            code_cells = [c for c in cells if c.get("cell_type") == "code"]
            cell_order = [c["id"] for c in code_cells]

            execution_results = []
            sdc_results = []
            all_stale: set = set()
            violations: List[Dict[str, Any]] = []
            total_executed = 0
            status = "success"

            with timer(key="execute_magic", message="Executing magic %continue_after_violation on"):
                kernel_client.execute("%continue_after_violation on")


            with timer(key="execute_sdc", message="Executing all cells with SDC"):
                for idx, cell in enumerate(cells):
                    if cell.get("cell_type") != "code":
                        continue

                    if selected_cell_ids and cell.get("id") not in selected_cell_ids:
                        continue

                    cell_id = cell.get("id")
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    if not source.strip():
                        continue

                    with timer(
                        key="execute_cell", message=f"Executing cell {idx}:{cell_id}"
                    ):
                        # Build metadata with cell_id and cell_order
                        cell_metadata = cell.get("metadata", {}).copy()
                        cell_metadata["cell_id"] = cell_id
                        cell_metadata["cell_order"] = cell_order

                        try:
                            with self.timing_context() as cell_get_elapsed:
                                result = KernelHelper.execute_code(
                                    kernel_client,
                                    source,
                                    self.timeout
                                    cell_id=cell_id,
                                    cell_metadata=cell_metadata,
                                )

                            cell["execution_count"] = result["execution_count"]
                            cell["outputs"] = result["outputs"]

                            # Extract SDC metadata from outputs
                            sdc_meta = self._extract_sdc_metadata(result["outputs"])
                            if sdc_meta:
                                sdc_results.append(
                                    {
                                        "cell_id": cell_id,
                                        "sdc": sdc_meta,
                                    }
                                )

                                # Track stale cells
                                if sdc_meta.get("stale_cells"):
                                    all_stale.update(sdc_meta["stale_cells"])

                                # Track violations
                                if sdc_meta.get("violation"):
                                    violations.append(sdc_meta["violation"])

                            # Check for execution errors
                            if result["status"] == "error":
                                status = "error"
                                error_message = result.get(
                                    "error_message", "Unknown error"
                                )
                                log(f"Error in cell {cell_id}: {error_message}")

                                execution_results.append(
                                    {
                                        "cell_index": idx,
                                        "cell_id": cell_id,
                                        "status": "error",
                                        "execution_count": result["execution_count"],
                                        "error_message": error_message,
                                        "execution_time": cell_get_elapsed() * 1000,
                                    }
                                )
                                break  # Stop on first error

                            # Extract all metadata types using generic extractor
                            extract_and_set_metadata(cell, result["outputs"])

                            # Build execution result with timing
                            cell_result = {
                                "cell_index": idx,
                                "cell_id": cell_id,
                                "status": result["status"],
                                "execution_count": result["execution_count"],
                                "execution_time": cell_get_elapsed() * 1000,
                                "sdc": sdc_meta,
                            }
                            # Add SDC timing if available
                            if sdc_meta:
                                cell_result["run_ms"] = sdc_meta.get("run_duration_ms", 0.0)
                                cell_result["state_ms"] = sdc_meta.get("state_duration_ms", 0.0)
                                cell_result["check_ms"] = sdc_meta.get("check_duration_ms", 0.0)

                            execution_results.append(cell_result)
                            log(f"[{result['execution_count']}]")

                            total_executed += 1

                        except Exception as e:
                            # Catch any Python exceptions during execution
                            cell["outputs"] = [
                                {
                                    "output_type": "error",
                                    "ename": e.__class__.__name__,
                                    "evalue": str(e),
                                    "traceback": traceback.format_exception(
                                        type(e), e, e.__traceback__
                                    ),
                                }
                            ]

                            execution_results.append(
                                {
                                    "cell_index": idx,
                                    "cell_id": cell_id,
                                    "status": "error",
                                    "execution_time": cell_get_elapsed() * 1000,
                                }
                            )
                            status = "error"
                            break  # Stop on exception

            metadata = {
                "status": status,
                "command": self.command_name,
                "sdc_enabled": True,
                "cell_order": cell_order,
                "stale_cells": sorted(all_stale),
                "violations": violations,
                "execution": {
                    "total_executed": total_executed,
                    "results": execution_results,
                },
                "sdc_results": sdc_results,
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_notebook,
            metadata=metadata,
            total_cost=0.0,
            total_time=total_time,
        )

    def _extract_sdc_metadata(
        self, outputs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract SDC metadata from cell outputs.

        Looks for display_data outputs with ferret_sdc in metadata.
        """
        for output in outputs:
            if output.get("output_type") == "display_data":
                output_meta = output.get("metadata", {})
                if "ferret_sdc" in output_meta:
                    return output_meta["ferret_sdc"]
        return None
