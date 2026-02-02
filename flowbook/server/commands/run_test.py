"""
Run Test Command - Execute a test cell with checkpoint/restore semantics
"""

from typing import Any, Dict, List, Optional
import time

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.kernel_support.kernel_command_client import KernelCommandClient


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
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        **kwargs,
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
                    cell.get("metadata", {}).get("flowbook_test", {}).get("test_for")
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
        test_metadata = test_cell.get("metadata", {}).get("flowbook_test", {})

        # Create kernel command client and checkpoint name
        cmd_client = KernelCommandClient(kernel_client)
        checkpoint_name = f"test_{test_cell_id}_{int(time.time() * 1000)}"

        # Save checkpoint once at the beginning
        cmd_client.checkpoint_save(checkpoint_name)

        try:
            start_time = time.time()
            setup_time = 0.0
            parent_time = 0.0
            assertion_time = 0.0

            # Run setup code if present
            if sections["setup"].strip():
                setup_start = time.time()
                setup_result = await kernel_client.execute(
                    sections["setup"], silent=False, store_history=False
                )
                setup_time = time.time() - setup_start
                if setup_result.get("outputs"):
                    outputs.extend(setup_result["outputs"])

            # Run parent cell code
            parent_start = time.time()
            parent_source = self._get_cell_source(parent_cell)
            parent_result = await kernel_client.execute(
                parent_source, silent=False, store_history=False
            )
            parent_time = time.time() - parent_start
            if parent_result.get("outputs"):
                outputs.extend(parent_result["outputs"])

            # Add separator
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"text/plain": "# ..."},
                    "metadata": {},
                }
            )

            # Run assertions
            assertion_start = time.time()
            assertion_result = await kernel_client.execute(
                sections["assertions"], silent=False, store_history=False
            )
            assertion_time = time.time() - assertion_start

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

            # Restore checkpoint after test execution
            cmd_client.checkpoint_restore(checkpoint_name)

            # Update test metadata with results
            test_metadata["last_run"] = {
                "status": "failed" if has_error else "passed",
                "duration": duration,
                "error_message": error_message,
                "timestamp": int(time.time() * 1000),
            }

            # Add timing output with detailed breakdown
            status_symbol = "✗" if has_error else "✓"
            status_text = "Failed" if has_error else "Passed"
            status_color = "#f44336" if has_error else "#4caf50"

            # Build timing breakdown
            timing_parts = []
            if setup_time > 0:
                timing_parts.append(f"setup: {setup_time:.3f}s")
            timing_parts.append(f"parent: {parent_time:.3f}s")
            timing_parts.append(f"assertions: {assertion_time:.3f}s")
            timing_breakdown = ", ".join(timing_parts)

            # Debug: Log timing information
            print(f"[RunTest] Timing - setup: {setup_time:.3f}s, parent: {parent_time:.3f}s, assertions: {assertion_time:.3f}s, total: {duration:.3f}s")
            print(f"[RunTest] Building timing HTML with breakdown: {timing_breakdown}")

            timing_html = (
                f'<div style="color: {status_color}; '
                f'font-size: 12px; margin-top: 8px; font-weight: 600;">'
                f'{status_symbol} {status_text} ({duration:.3f}s total: {timing_breakdown})</div>'
            )
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {
                        "text/html": timing_html
                    },
                    "metadata": {},
                }
            )
            print(f"[RunTest] Added timing output. Total outputs: {len(outputs)}")

        except Exception as e:
            # Restore checkpoint even on error
            try:
                cmd_client.checkpoint_restore(checkpoint_name)
            except Exception:
                pass  # Best effort restore

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

        finally:
            # Clean up checkpoint at the end
            try:
                cmd_client.checkpoint_delete(checkpoint_name)
            except Exception:
                pass  # Best effort cleanup

        # Update test cell in notebook
        test_cell["outputs"] = outputs
        test_cell["metadata"]["flowbook_test"] = test_metadata

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
