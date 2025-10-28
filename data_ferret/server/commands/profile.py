"""
Profile command implementation.
"""

import copy
import traceback
from typing import Any, Dict, Optional

from data_ferret.server.base import NotebookCommand
from data_ferret.util.ferret_metadata import FerretMetadata, ProfileData, set_profile_ferret_metadata
from data_ferret.server.kernel_helper import KernelHelper
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.output import log, timer


class ProfileCommand(NotebookCommand):
    """Profiles code cells with memory and performance tracking."""

    @property
    def command_name(self) -> str:
        return "profile"

    @property
    def display_name(self) -> str:
        return "Profile"

    @property
    def icon_name(self) -> str:
        return "ui-components:run"

    @property
    def tooltip(self) -> str:
        return "Profile code cells with memory and performance tracking"

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
    ) -> Dict[str, Any]:
        """Profile code cells."""
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

        kernel_client.execute("%enable_scalene")

        with timer(key="profile", message="Profiling cells"):
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    if selected_cell_ids and cell.get("id") not in selected_cell_ids:
                        continue

                    with timer(key="profile_cell", message=f"Profiling cell {idx}:{cell.get('id')}"):
                        source = cell.get("source", "")
                        if isinstance(source, list):
                            source = "".join(source)

                        metadata = cell.get("metadata", {}).copy()
                        metadata['cell_id'] = cell.get("id")

                        if source.strip():
                            try:
                                result = KernelHelper.execute_code(
                                    kernel_client,
                                    source,
                                    cell_id=cell.get("id"),
                                    cell_metadata=metadata,
                                )

                                cell["execution_count"] = result["execution_count"]
                                cell["outputs"] = result["outputs"]

                                for output in result["outputs"]:
                                    if 'metadata' in output:
                                        output_metadata = output['metadata']
                                        if 'profile' in output_metadata:
                                            profile_metadata = ProfileData.model_validate(output_metadata['profile'])
                                            set_profile_ferret_metadata(cell, profile_metadata)

                                execution_results.append(
                                    {
                                        "cell_index": idx,
                                        "status": result["status"],
                                        "execution_count": result["execution_count"],
                                    }
                                )
                                log(f"[{result['execution_count']}]")

                                total_executed += 1
                            except Exception as e:
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
                                        "status": "error"
                                    }
                                )

        metadata = {
            "status": "success",
            "command": self.command_name,
            "execution": {
                "total_executed": total_executed,
                "results": execution_results,
            },
        }

        return {"notebook": new_notebook, "metadata": metadata}
