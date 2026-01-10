"""
Inspect variables command implementation.
"""

import copy
import json
from typing import Any, Dict, Optional

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_helper import KernelHelper
from flowbook.server.kernel_manager import FlowbookKernelClient


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

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[list] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Inspect kernel variables."""
        with self.timing_context() as get_elapsed:
            print("KERNEL CLIENT", kernel_client)
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

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_notebook,
            metadata=metadata,
            total_cost=0.0,
            total_time=total_time
        )

