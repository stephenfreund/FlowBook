"""
Run Test Command - Execute a test cell with checkpoint/restore semantics
"""

from typing import Any, Dict, List, Optional
import time

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.server.kernel_manager import FerretKernelClient


class RunTestCommand(NotebookCommand):
    @property
    def command_name(self) -> str:
        return "run_test"

    @property
    def display_name(self) -> str:
        return "Run Test"

    @property
    def icon_name(self) -> str:
        return "ui-components:check"

    @property
    def requires_kernel(self) -> bool:
        return True

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        **kwargs
    ) -> ProcessingResult:
        """Run a test cell with checkpoint/restore semantics."""

        if not kernel_client:
            raise ValueError("Kernel required for running tests")

        test_cell_id = kwargs.get("test_cell_id")
        if not test_cell_id:
            raise ValueError("test_cell_id required")

        # Find test cell and parent cell
        cells = notebook_content.get("cells", [])
        test_cell = None
        parent_cell = None

        for cell in cells:
            if cell.get("id") == test_cell_id:
                test_cell = cell
                parent_cell_id = (
                    cell.get("metadata", {}).get("ferret_test", {}).get("test_for")
                )
                break

        if not test_cell:
            raise ValueError(f"Test cell {test_cell_id} not found")

        for cell in cells:
            if cell.get("id") == parent_cell_id:
                parent_cell = cell
                break

        if not parent_cell:
            raise ValueError(f"Parent cell {parent_cell_id} not found")

        # Parse test source into sections
        test_source = self._get_cell_source(test_cell)
        sections = self._parse_test_source(test_source)

        outputs = []
        test_metadata = test_cell.get("metadata", {}).get("ferret_test", {})

        try:
            start_time = time.time()

            # Create checkpoint
            checkpoint_code = """
import sys
_ferret_test_checkpoint = {}
_ferret_test_checkpoint['globals'] = {k: v for k, v in globals().items()
                                       if not k.startswith('_')}
"""
            await kernel_client.execute(
                checkpoint_code, silent=True, store_history=False
            )

            # Run setup code if present
            if sections["setup"].strip():
                setup_result = await kernel_client.execute(
                    sections["setup"], silent=False, store_history=False
                )
                if setup_result.get("outputs"):
                    outputs.extend(setup_result["outputs"])

            # Run parent cell code
            parent_source = self._get_cell_source(parent_cell)
            parent_result = await kernel_client.execute(
                parent_source, silent=False, store_history=False
            )
            if parent_result.get("outputs"):
                outputs.extend(parent_result["outputs"])

            # Add separator
            outputs.append(
                {"output_type": "display_data", "data": {"text/plain": "# ..."}, "metadata": {}}
            )

            # Run assertions
            assertion_result = await kernel_client.execute(
                sections["assertions"], silent=False, store_history=False
            )

            duration = time.time() - start_time

            # Check for errors
            has_error = False
            error_message = None

            for output in assertion_result.get("outputs", []):
                outputs.append(output)
                if output.get("output_type") == "error":
                    has_error = True
                    traceback = output.get("traceback", [])
                    if traceback:
                        error_message = "\n".join(traceback)
                    else:
                        error_message = output.get("evalue", "Unknown error")

            # Restore checkpoint
            restore_code = """
# Restore globals
_to_delete = [k for k in globals().keys()
              if k not in _ferret_test_checkpoint['globals']
              and not k.startswith('_')]
for k in _to_delete:
    del globals()[k]

for k, v in _ferret_test_checkpoint['globals'].items():
    globals()[k] = v

del _ferret_test_checkpoint
"""
            await kernel_client.execute(restore_code, silent=True, store_history=False)

            # Update test metadata with results
            test_metadata["last_run"] = {
                "status": "failed" if has_error else "passed",
                "duration": duration,
                "error_message": error_message,
                "timestamp": int(time.time() * 1000),
            }

            # Add timing output
            status_symbol = "✗" if has_error else "✓"
            status_text = "Failed" if has_error else "Passed"
            status_color = "#f44336" if has_error else "#4caf50"
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {
                        "text/html": f'<div style="color: {status_color}; '
                        f'font-size: 12px; margin-top: 8px; font-weight: 600;">'
                        f'{status_symbol} {status_text} ({duration:.3f}s)</div>'
                    },
                    "metadata": {},
                }
            )

        except Exception as e:
            # Handle execution error
            test_metadata["last_run"] = {
                "status": "error",
                "duration": 0,
                "error_message": str(e),
                "timestamp": int(time.time() * 1000),
            }
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)],
                }
            )

        # Update test cell in notebook
        test_cell["outputs"] = outputs
        test_cell["metadata"]["ferret_test"] = test_metadata

        return ProcessingResult(
            notebook=notebook_content, metadata={"test_run": test_metadata["last_run"]}
        )

    def _get_cell_source(self, cell: Dict[str, Any]) -> str:
        """Get source code from a cell."""
        source = cell.get("source", [])
        if isinstance(source, list):
            return "".join(source)
        return source

    def _parse_test_source(self, source: str) -> Dict[str, str]:
        """Parse test source into setup and assertions sections."""
        lines = source.split("\n")

        sections = {"setup": "", "assertions": ""}

        current_section = None

        for line in lines:
            if line.strip().startswith("# Setup:"):
                current_section = "setup"
            elif line.strip().startswith("# ..."):
                current_section = None
            elif line.strip().startswith("# Assertions:"):
                current_section = "assertions"
            elif current_section:
                sections[current_section] += line + "\n"

        return sections
