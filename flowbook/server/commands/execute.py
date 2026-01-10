"""
Execute all cells command with Sequential Dataflow Consistency enforcement.
"""

import argparse
import copy
import traceback
from typing import Any, Dict, List, Optional

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_helper import KernelHelper
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.util.metadata_extractor import extract_and_set_metadata
from flowbook.util.output import error, log, timer


class ExecuteCommand(NotebookCommand):
    """
    Execute notebook cells with Sequential Dataflow Consistency enforcement.

    Uses FlowbookSDCKernel to track variable dependencies and enforce
    that cells don't modify state read by earlier cells.
    """

    @property
    def command_name(self) -> str:
        return "execute"

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
        return "flowbook_sdc_kernel"

    def make_subparser(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        """Add command-specific CLI arguments."""
        subparser = super().make_subparser(subparsers)
        subparser.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="Timeout in seconds for each cell execution",
        )
        subparser.add_argument(
            "--downsample-csv",
            type=float,
            default=None,
            metavar="PROPORTION",
            help="Proportion of rows to keep from CSV files (e.g., 0.1 for 10%%)",
        )
        return subparser

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Execute notebook with SDC enforcement.

        Args:
            notebook_content: Notebook JSON
            kernel_client: FlowbookKernelClient instance (or FlowbookSDCKernelClient)
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

        # Extract command-specific kwargs
        cell_timeout = kwargs.get("timeout") or self.timeout
        downsample_csv = kwargs.get("downsample_csv")

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
            error_message = None
            error_cell_id = None

            with timer(key="execute:magic", message="Executing magic %continue_after_violation on"):
                kernel_client.execute("%continue_after_violation on")

            # Inject CSV downsampling monkey-patch if requested
            if downsample_csv is not None:
                KernelHelper.inject_csv_downsampling(kernel_client, downsample_csv)

            with timer(key="execute:sdc", message="Executing all cells with SDC"):
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
                        key="execute:cell", message=f"Executing cell {idx}:{cell_id}"
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
                                    cell_timeout,
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

                            # Check for execution errors or timeouts
                            if result["status"] in ("error", "timeout"):
                                status = result["status"]
                                error_message = result.get(
                                    "error_message", "Unknown error"
                                )
                                error_cell_id = cell_id
                                print()
                                print("--------------------------------")
                                print(f"{error_message}")
                                print("--------------------------------")

                                execution_results.append(
                                    {
                                        "cell_index": idx,
                                        "cell_id": cell_id,
                                        "status": status,
                                        "execution_count": result["execution_count"],
                                        "error_message": error_message,
                                        "execution_time": cell_get_elapsed() * 1000,
                                    }
                                )
                                break  # Stop on first error or timeout

                            # Extract all metadata types using generic extractor
                            extract_and_set_metadata(cell, result["outputs"])

                            for output in result["outputs"]:
                                if output["output_type"] == "stream":
                                    log(output["text"])

                                elif output["output_type"] == "execute_result":
                                    if isinstance(output["data"], dict):
                                        if "text/plain" in output["data"]:
                                            log(output["data"]["text/plain"])
                                        else:
                                            log(f"No text/plain in execute_result: {output['data'].keys()}")
                                    else:
                                        log(f"Execute result is not a dict: {output['data']}")
                                elif output["output_type"] == "display_data":
                                    if isinstance(output["data"], dict):
                                        if "text/plain" in output["data"]:
                                            log(output["data"]["text/plain"])
                                        else:
                                            log(f"No text/plain in display_data: {output['data'].keys()}")
                                    else:
                                        log(f"Display data is not a dict: {output['data']}")

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
                            log(f"Execution count: {result['execution_count']}")

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
                "error_message": error_message,
                "error_cell_id": error_cell_id,
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

        Looks for display_data outputs with flowbook_sdc in metadata.
        """
        for output in outputs:
            if output.get("output_type") == "display_data":
                output_meta = output.get("metadata", {})
                if "flowbook_sdc" in output_meta:
                    return output_meta["flowbook_sdc"]
        return None
