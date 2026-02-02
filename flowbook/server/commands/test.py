"""
Test command implementation.

Executes unit tests for cells with checkpoint/restore.
"""

from typing import Any, Dict, List, Optional
import time
import nbformat

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.server.kernel_helper import KernelHelper
from flowbook.kernel_support.kernel_command_client import KernelCommandClient
from flowbook.util.flowbook_metadata import FlowbookMetadata, UnitTest
from flowbook.util.output import indent, log, print, timer


class TestCommand(NotebookCommand):
    """Execute unit tests for cells with checkpoint/restore."""

    @property
    def command_name(self) -> str:
        return "test"

    @property
    def display_name(self) -> str:
        return "Run Unit Tests"

    @property
    def icon_name(self) -> str:
        return "ui-components:check"

    @property
    def tooltip(self) -> str:
        return "Run unit tests for cell(s)"

    @property
    def requires_kernel(self) -> bool:
        return True

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        **kwargs
    ) -> ProcessingResult:
        """
        Execute unit tests for specified cell(s).

        If selected_cell_ids is provided: test only those cells
        If selected_cell_ids is None: test ALL cells with unit tests
        """

        start_time = time.time()

        # Initialize checkpoint client
        checkpoint_client = KernelCommandClient(kernel_client, timeout=30)

        notebook = nbformat.from_dict(notebook_content)

        # Determine which cells to test
        cells_to_test = []
        if selected_cell_ids:
            # Test only selected cells
            for cell in notebook.cells:
                if cell.cell_type == 'code' and cell.get('id') in selected_cell_ids:
                    cells_to_test.append(cell)
        else:
            # Test all cells that have unit tests
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    flowbook_meta = FlowbookMetadata.from_cell(cell)
                    if flowbook_meta.unit_tests and flowbook_meta.unit_tests.tests:
                        cells_to_test.append(cell)

        print(f"Testing {len(cells_to_test)} cell(s)")

        # Clear existing outputs for cells being tested
        for cell in cells_to_test:
            cell.outputs = []

        # Execute tests for each cell
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for cell in cells_to_test:
            print(f"Testing cell {cell.id}")
            with indent(message=f"Testing cell {cell.id}"):
                passed, failed = self._execute_tests_for_cell(cell, kernel_client, checkpoint_client)
                total_tests += passed + failed
                passed_tests += passed
                failed_tests += failed

        total_time = time.time() - start_time

        return ProcessingResult(
            notebook=dict(notebook),
            metadata={
                "cells_tested": len(cells_to_test),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests
            },
            total_cost=0.0,
            total_time=total_time
        )

    def _execute_tests_for_cell(
        self,
        cell: nbformat.NotebookNode,
        kernel_client: FlowbookKernelClient,
        checkpoint_client: KernelCommandClient
    ) -> tuple[int, int]:
        """
        Execute all tests for a single cell.

        Returns:
            Tuple of (passed_count, failed_count)
        """
        flowbook_meta = FlowbookMetadata.from_cell(cell)

        if not flowbook_meta.unit_tests or not flowbook_meta.unit_tests.tests:
            return (0, 0)

        cell_code = cell.source
        tests = flowbook_meta.unit_tests.tests

        # Calculate max title length for alignment
        max_title_length = max(len(test.title) for test in tests) if tests else 0

        # Add header output
        self._add_output(cell, "stream", {
            "name": "stdout",
            "text": f"Running {len(tests)} test(s) for cell {cell.id[:8]}...\n"
        })

        passed_count = 0
        failed_count = 0

        # Generate checkpoint name for this cell (shared by all tests)
        checkpoint_name = f"__flowbook_test_checkpoint_{cell.id}"

        # Save checkpoint once before all tests
        try:
            response = checkpoint_client.checkpoint_save(checkpoint_name)
            if response.status != "ok":
                log(f"Error: Failed to save checkpoint for cell {cell.id}: {response.message}")
                return (0, 0)  # Can't test without checkpoint
        except Exception as e:
            log(f"Error: Failed to save checkpoint for cell {cell.id}: {e}")
            return (0, 0)

        try:
            # Execute each test
            for i, test in enumerate(tests, 1):
                success = self._execute_single_test(
                    cell, test, cell_code, kernel_client, checkpoint_client,
                    checkpoint_name, i, len(tests), max_title_length
                )
                if success:
                    passed_count += 1
                else:
                    failed_count += 1
        finally:
            # Delete checkpoint after all tests complete
            self._delete_checkpoint(checkpoint_client, checkpoint_name)

        # Add summary
        summary_text = f"\nTest Results: {passed_count} passed, {failed_count} failed\n"
        self._add_output(cell, "stream", {
            "name": "stdout",
            "text": summary_text
        })

        return (passed_count, failed_count)

    def _execute_single_test(
        self,
        cell: nbformat.NotebookNode,
        test: UnitTest,
        cell_code: str,
        kernel_client: FlowbookKernelClient,
        checkpoint_client: KernelCommandClient,
        checkpoint_name: str,
        test_num: int,
        total_tests: int,
        max_title_length: int
    ) -> bool:
        """
        Execute a single test with checkpoint/restore.

        Returns:
            True if test passed, False if test failed
        """
        print(f"Running test {test_num}/{total_tests}: {test.title}...")

        # Track timing for each phase
        start_time = time.time()
        setup_time = 0.0
        cell_time = 0.0
        assertion_time = 0.0

        try:
            # Execute setup code
            if test.setup_code and test.setup_code.strip():
                setup_start = time.time()
                result = KernelHelper.execute_code(kernel_client, test.setup_code, timeout=30.0, store_history=False)
                setup_time = time.time() - setup_start
                if not self._check_execution_success(result):
                    total_time = time.time() - start_time
                    self._add_test_failure(cell, test, "Setup code failed", result, test_num, total_tests, total_time, max_title_length)
                    self._restore_checkpoint_only(checkpoint_client, checkpoint_name)
                    return False

            # 3. Execute cell code
            cell_start = time.time()
            result = KernelHelper.execute_code(kernel_client, cell_code, timeout=30.0, store_history=False)
            cell_time = time.time() - cell_start
            if not self._check_execution_success(result):
                total_time = time.time() - start_time
                self._add_test_failure(cell, test, "Cell execution failed", result, test_num, total_tests, total_time, max_title_length)
                self._restore_checkpoint_only(checkpoint_client, checkpoint_name)
                return False

            # 4. Execute assertion code
            assertion_start = time.time()
            result = KernelHelper.execute_code(kernel_client, test.assertion_code, timeout=30.0, store_history=False)
            assertion_time = time.time() - assertion_start
            if not self._check_execution_success(result):
                total_time = time.time() - start_time
                self._add_test_failure(cell, test, "Assertion failed", result, test_num, total_tests, total_time, max_title_length)
                self._restore_checkpoint_only(checkpoint_client, checkpoint_name)
                return False

            # 5. Test passed!
            total_time = time.time() - start_time
            self._add_test_success(cell, test, test_num, total_tests, total_time, setup_time, cell_time, assertion_time, max_title_length)
            return True

        finally:
            # 6. Always restore checkpoint
            self._restore_checkpoint_only(checkpoint_client, checkpoint_name)

    def _restore_checkpoint_only(self, checkpoint_client: KernelCommandClient, checkpoint_name: str):
        """Restore a checkpoint without deleting it."""
        try:
            response = checkpoint_client.checkpoint_restore(checkpoint_name)
            if response.status != "ok":
                log(f"Warning: Failed to restore checkpoint '{checkpoint_name}': {response.message}")
        except Exception as e:
            log(f"Warning: Error during checkpoint restore: {e}")

    def _delete_checkpoint(self, checkpoint_client: KernelCommandClient, checkpoint_name: str):
        """Delete a checkpoint to free memory."""
        try:
            response = checkpoint_client.checkpoint_delete(checkpoint_name)
            if response.status != "ok":
                log(f"Warning: Failed to delete checkpoint '{checkpoint_name}': {response.message}")
        except Exception as e:
            log(f"Warning: Error during checkpoint delete: {e}")

    def _check_execution_success(self, result: Dict[str, Any]) -> bool:
        """Check if kernel execution succeeded."""
        return result.get('status') == 'ok'

    def _add_test_success(
        self,
        cell: nbformat.NotebookNode,
        test: UnitTest,
        test_num: int,
        total_tests: int,
        total_time: float,
        setup_time: float = 0.0,
        cell_time: float = 0.0,
        assertion_time: float = 0.0,
        max_title_length: int = 0
    ):
        """Add success output to cell with timing information."""
        # Build timing breakdown
        timing_parts = []
        if setup_time > 0:
            timing_parts.append(f"setup: {setup_time:.3f}s")
        timing_parts.append(f"cell: {cell_time:.3f}s")
        timing_parts.append(f"assertions: {assertion_time:.3f}s")
        timing_breakdown = ", ".join(timing_parts)

        # Pad title to align timing information
        padded_title = test.title.ljust(max_title_length)

        self._add_output(cell, "stream", {
            "name": "stdout",
            "text": f"✓ Test {test_num}/{total_tests}: {padded_title} ({total_time:.3f}s total: {timing_breakdown})\n"
        })

    def _add_test_failure(
        self,
        cell: nbformat.NotebookNode,
        test: UnitTest,
        reason: str,
        result: Dict[str, Any],
        test_num: int,
        total_tests: int,
        total_time: float,
        max_title_length: int = 0
    ):
        """Add failure output to cell with timing information."""
        # Pad title to align timing information
        padded_title = test.title.ljust(max_title_length)

        error_text = f"✗ Test {test_num}/{total_tests}: {padded_title} ({total_time:.3f}s)\n"
        error_text += f"  Reason: {reason}\n"

        # Extract error details from result
        if result.get('error_message'):
            error_text += f"  Error:\n{result['error_message']}\n"

        self._add_output(cell, "stream", {
            "name": "stderr",
            "text": error_text
        })

    def _add_output(
        self,
        cell: nbformat.NotebookNode,
        output_type: str,
        output_data: Dict[str, Any]
    ):
        """Add an output to a cell."""
        output = nbformat.v4.new_output(output_type, **output_data)
        cell.outputs.append(output)
