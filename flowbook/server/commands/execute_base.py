"""
Execute all cells command implementation.
"""

import argparse
import copy
import sys
import traceback
from typing import Any, Dict, Optional

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.util.flowbook_metadata import FlowbookMetadata, ProfileData, set_profile_flowbook_metadata
from flowbook.util.metadata_extractor import extract_and_set_metadata
from flowbook.server.kernel_helper import KernelHelper
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.util.output import log, timer


class ExecuteBaseCommand(NotebookCommand):
    """Executes all code cells in the notebook using the kernel."""

    @property
    def command_name(self) -> str:
        return "execute_base"

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

    @property
    def kernel_name(self) -> str:
        """
        Return the kernel name to use for this command.

        Override this property to specify a different kernel.
        Default is 'flowbook_kernel'.
        """
        return "python3"


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

        # Extract command-specific kwargs
        cell_timeout = kwargs.get("timeout") or self.timeout
        downsample_csv = kwargs.get("downsample_csv")

        with self.timing_context() as get_elapsed:
            new_notebook = copy.deepcopy(notebook_content)
            cells = new_notebook.get("cells", [])

            execution_results = []
            total_executed = 0
            status = "success"  # Track actual execution status
            error_message = None
            error_cell_id = None

            # No scalene for base runs
            # KernelHelper.execute_code(kernel_client, "%scalene on", store_history=False)

            # Inject CSV downsampling monkey-patch if requested
            if downsample_csv is not None:
                KernelHelper.inject_csv_downsampling(kernel_client, downsample_csv)

            with timer(key="execute:all", message="Executing all cells"):
                for idx, cell in enumerate(cells):
                    if cell.get("cell_type") == "code":
                        if selected_cell_ids and cell.get("id") not in selected_cell_ids:
                            continue

                        with timer(key="execute:cell", message=f"Executing cell {idx}:{cell.get('id')}"):
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
                                            cell_timeout,
                                            cell_id=cell.get("id"),
                                            cell_metadata=metadata,
                                        )   

                                    cell["execution_count"] = result["execution_count"]
                                    cell["outputs"] = result["outputs"]

                                    # Print flowbook protocol messages
                                    self.print_flowbook_messages(result)

                                    # Check for execution errors or timeouts
                                    if result["status"] in ("error", "timeout"):
                                        status = result["status"]
                                        error_message = result["error_message"]
                                        error_cell_id = cell.get("id")
                                        print()
                                        print(f"--------------------------------")
                                        print(f"{error_message}")
                                        print(f"--------------------------------")
                                        execution_results.append(
                                            {
                                                "cell_index": idx,
                                                "cell_id": error_cell_id,
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

                                    execution_results.append(
                                        {
                                            "cell_index": idx,
                                            "status": result["status"],
                                            "execution_count": result["execution_count"],
                                            "execution_time": cell_get_elapsed() * 1000,
                                        }
                                    )
                                    log(f"Execution count: {result['execution_count']}")

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
                "error_message": error_message,
                "error_cell_id": error_cell_id,
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

