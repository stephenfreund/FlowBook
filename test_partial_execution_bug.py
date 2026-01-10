"""
Test for the specific bug where optimized code partially executes before error.

This test replicates the exact scenario from the output log where:
1. Optimized code runs for 37 lines
2. Line 38 throws NameError
3. Error is not captured by _execute_code_safely
4. Code proceeds to checkpoint comparison
5. Validation detects variable mismatch (df shape changed)

Run with: python -m pytest test_partial_execution_bug.py -v -s
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from flowbook.server.commands.optimize import CodeExecutionOrchestrator
from flowbook.kernel.types import (
    TestCodeSuccess,
    TestCodeModifiedCrash,
    ExecutionError
)


class MockKernelClientWithPartialExecution:
    """
    Mock kernel client that simulates partial code execution.

    This mock simulates:
    - Code executing successfully up to line 37
    - NameError on line 38
    - Error message being sent but potentially not captured
    """

    def __init__(self, error_msg_has_wrong_parent_id=False):
        """
        Args:
            error_msg_has_wrong_parent_id: If True, error message will have
                mismatched parent_header, simulating the bug
        """
        self.error_msg_has_wrong_parent_id = error_msg_has_wrong_parent_id
        self.message_queue = []
        self.current_msg_id = None

    def execute(self, code, store_history=False):
        """Mock execute that tracks message ID."""
        self.current_msg_id = "msg-123"

        # Queue messages based on code content
        if "new_years_model" in code:
            # This code will cause NameError
            self._queue_partial_execution_with_error()
        else:
            # Normal successful execution
            self._queue_successful_execution()

        return self.current_msg_id

    def _queue_partial_execution_with_error(self):
        """Queue messages for partial execution followed by error."""
        msg_id = self.current_msg_id

        # Message 1: execute_input
        self.message_queue.append({
            'header': {'msg_type': 'execute_input', 'msg_id': 'input-1'},
            'parent_header': {'msg_id': msg_id},
            'content': {'execution_count': 1}
        })

        # Message 2: stream output from lines 1-37 executing successfully
        self.message_queue.append({
            'header': {'msg_type': 'stream', 'msg_id': 'stream-1'},
            'parent_header': {'msg_id': msg_id},
            'content': {
                'name': 'stdout',
                'text': 'Processing data up to line 37...\n'
            }
        })

        # Message 3: error on line 38
        # BUG SIMULATION: Use wrong parent_id if flag is set
        error_parent_id = "wrong-msg-456" if self.error_msg_has_wrong_parent_id else msg_id

        self.message_queue.append({
            'header': {'msg_type': 'error', 'msg_id': 'error-1'},
            'parent_header': {'msg_id': error_parent_id},
            'content': {
                'ename': 'NameError',
                'evalue': "name 'new_years_model' is not defined",
                'traceback': [
                    'Traceback (most recent call last):',
                    'Cell In[43], line 38',
                    "    df['new_years_factor'] = new_years_model.predict(df[new_years_columns])",
                    "NameError: name 'new_years_model' is not defined"
                ]
            }
        })

        # Message 4: status idle
        self.message_queue.append({
            'header': {'msg_type': 'status', 'msg_id': 'status-1'},
            'parent_header': {'msg_id': msg_id},
            'content': {'execution_state': 'idle'}
        })

    def _queue_successful_execution(self):
        """Queue messages for successful execution."""
        msg_id = self.current_msg_id

        self.message_queue.append({
            'header': {'msg_type': 'execute_input', 'msg_id': 'input-1'},
            'parent_header': {'msg_id': msg_id},
            'content': {}
        })

        self.message_queue.append({
            'header': {'msg_type': 'status', 'msg_id': 'status-1'},
            'parent_header': {'msg_id': msg_id},
            'content': {'execution_state': 'idle'}
        })

    def get_iopub_msg(self, timeout=60.0):
        """Return queued messages."""
        if not self.message_queue:
            raise TimeoutError("No more messages")
        return self.message_queue.pop(0)


class TestPartialExecutionBug:
    """Test the specific bug scenario from the output log."""

    def test_error_captured_with_correct_parent_header(self):
        """
        Test that error IS captured when parent_header is correct.

        This is the expected behavior.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=False)
        orchestrator = CodeExecutionOrchestrator(client)

        code = """
# Lines 1-37: successful execution
df['ratio'] = base_ratio
# Line 38: NameError
df['new_years_factor'] = new_years_model.predict(df[new_years_columns])
"""

        duration, error = orchestrator._execute_code_safely(code)

        # Error SHOULD be captured
        assert error is not None, "Error should be captured with correct parent_header"
        assert error.error_type == "NameError"
        assert "new_years_model" in error.error_message
        assert "line 38" in error.traceback

    def test_error_missed_with_wrong_parent_header(self):
        """
        Test that error is MISSED when parent_header is wrong.

        This reproduces the BUG from the output log.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=True)
        orchestrator = CodeExecutionOrchestrator(client)

        code = """
# Lines 1-37: successful execution
df['ratio'] = base_ratio
# Line 38: NameError
df['new_years_factor'] = new_years_model.predict(df[new_years_columns])
"""

        duration, error = orchestrator._execute_code_safely(code)

        # Error is MISSED due to wrong parent_header
        # This is the BUG
        assert error is None, "BUG: Error not captured due to parent_header mismatch"


class TestTestCodeWithPartialExecution:
    """Test the full test_code flow with partial execution."""

    @pytest.fixture
    def mock_cmd_client(self):
        """Mock KernelCommandClient for checkpoint operations."""
        mock = Mock()

        # Mock checkpoint operations
        mock.checkpoint_save = Mock()
        mock.checkpoint_restore = Mock()

        # Mock checkpoint comparison
        from flowbook.kernel.types import DiffResult, CompareCheckpointsResponse
        mock.checkpoint_compare = Mock(return_value=CompareCheckpointsResponse(
            diff=DiffResult(differences={})
        ))

        return mock

    def test_test_code_detects_modified_crash(self, mock_cmd_client):
        """
        Test that test_code correctly returns ModifiedCrash when error is captured.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=False)

        # Patch the KernelCommandClient initialization
        with patch('flowbook.server.commands.optimize.KernelCommandClient', return_value=mock_cmd_client):
            orchestrator = CodeExecutionOrchestrator(client)

            original_code = "df['ratio'] = base_ratio"
            modified_code = "df['new_years_factor'] = new_years_model.predict(df)"

            result = orchestrator.test_code(
                original_code=original_code,
                modified_code=modified_code,
                output_variables={'df'},
                pre_post_envs=None
            )

            # Should return ModifiedCrash
            assert isinstance(result, TestCodeModifiedCrash)
            assert result.error.error_type == "NameError"
            assert "new_years_model" in result.error.error_message

    def test_test_code_incorrectly_proceeds_when_error_missed(self, mock_cmd_client):
        """
        Test that when error is MISSED, test_code incorrectly proceeds to validation.

        This reproduces the exact bug scenario from the output log.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=True)

        # Patch the KernelCommandClient initialization
        with patch('flowbook.server.commands.optimize.KernelCommandClient', return_value=mock_cmd_client):
            orchestrator = CodeExecutionOrchestrator(client)

            original_code = "df['ratio'] = base_ratio"
            modified_code = """
# Partial execution changes df
df['ratio'] = base_ratio
# Then error on line 38
df['new_years_factor'] = new_years_model.predict(df)
"""

            # Mock checkpoint comparison to show df changed
            # (simulating partial execution that modified df before error)
            from flowbook.kernel.types import DiffResult, CompareCheckpointsResponse
            mock_cmd_client.checkpoint_compare.return_value = CompareCheckpointsResponse(
                diff=DiffResult(differences={
                    'df': {
                        'type': 'shape_mismatch',
                        'before': '(319809, 67)',
                        'after': '(319809, 68)'
                    }
                })
            )

            result = orchestrator.test_code(
                original_code=original_code,
                modified_code=modified_code,
                output_variables={'df'},
                pre_post_envs=None
            )

            # BUG: Instead of returning ModifiedCrash, returns Success with differences
            # because the error was not captured
            assert isinstance(result, TestCodeSuccess), \
                "BUG: Should be ModifiedCrash, but error wasn't captured"

            # The diff shows df changed (from partial execution)
            assert 'df' in result.diff.differences
            assert result.diff.differences['df']['type'] == 'shape_mismatch'


class TestMessageInspection:
    """
    Tests to help inspect and log actual message formats.

    These tests help diagnose what's happening with messages.
    """

    def test_log_all_messages_from_partial_execution(self, capsys):
        """
        Log all messages received during partial execution.

        Use -s flag to see output: pytest test_partial_execution_bug.py -v -s
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=False)

        # Execute to populate queue
        msg_id = client.execute("df['x'] = new_years_model.predict(df)", store_history=False)

        print(f"\n{'=' * 60}")
        print(f"Expected msg_id: {msg_id}")
        print(f"{'=' * 60}")

        # Read and log all messages
        message_num = 1
        while client.message_queue:
            msg = client.get_iopub_msg(timeout=1.0)

            print(f"\nMessage {message_num}:")
            print(f"  msg_type: {msg['header']['msg_type']}")
            print(f"  msg_id: {msg['header']['msg_id']}")
            print(f"  parent_header.msg_id: {msg['parent_header'].get('msg_id', 'MISSING')}")
            print(f"  parent_id matches: {msg['parent_header'].get('msg_id') == msg_id}")

            if msg['header']['msg_type'] == 'error':
                print(f"  ERROR CONTENT:")
                print(f"    ename: {msg['content']['ename']}")
                print(f"    evalue: {msg['content']['evalue']}")

            message_num += 1

        print(f"\n{'=' * 60}\n")

    def test_demonstrate_parent_header_filtering(self, capsys):
        """
        Demonstrate how parent_header filtering works and when it fails.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=True)
        msg_id = client.execute("df['x'] = new_years_model.predict(df)", store_history=False)

        print(f"\n{'=' * 60}")
        print(f"Demonstrating parent_header filtering BUG")
        print(f"Expected msg_id: {msg_id}")
        print(f"{'=' * 60}")

        # Manually walk through what _wait_for_execution does
        error_captured = None

        while client.message_queue:
            msg = client.get_iopub_msg(timeout=1.0)
            msg_type = msg['header']['msg_type']
            parent_id = msg['parent_header'].get('msg_id')

            print(f"\nMessage: {msg_type}")
            print(f"  parent_header.msg_id: {parent_id}")

            # This is the filter from _wait_for_execution line 303-304
            if parent_id != msg_id:
                print(f"  ❌ FILTERED OUT (parent_id doesn't match)")
                continue

            print(f"  ✓ Passed filter")

            if msg_type == 'error':
                print(f"  ✓ Error captured!")
                error_captured = msg['content']['ename']

            if msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                print(f"  ✓ Execution complete")
                break

        print(f"\n{'=' * 60}")
        print(f"Result: Error captured = {error_captured}")
        if error_captured is None:
            print("BUG: Error was NOT captured due to parent_header mismatch!")
        print(f"{'=' * 60}\n")


class TestProposedFixes:
    """Test potential fixes for the bug."""

    def test_fix_relaxed_parent_header_matching(self):
        """
        Test a potential fix: Don't filter by parent_header for error messages.

        Alternative approach: Accept error messages with any parent_header,
        or only filter for specific message types.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=True)
        msg_id = client.execute("df['x'] = new_years_model.predict(df)", store_history=False)

        # Simulate FIXED version of _wait_for_execution
        error_captured = None

        while client.message_queue:
            msg = client.get_iopub_msg(timeout=1.0)
            msg_type = msg['header']['msg_type']
            parent_id = msg['parent_header'].get('msg_id')

            # PROPOSED FIX: Only check parent_header for status messages
            # Accept error messages regardless of parent_header
            if msg_type not in ['error'] and parent_id != msg_id:
                continue

            if msg_type == 'error':
                error_captured = msg['content']

            if msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                if parent_id == msg_id:
                    break

        # With the fix, error SHOULD be captured even with wrong parent_header
        assert error_captured is not None
        assert error_captured['ename'] == 'NameError'

    def test_fix_track_all_errors_in_window(self):
        """
        Test another potential fix: Track all errors until status:idle.

        This approach captures any error that occurs between execute and idle,
        regardless of parent_header.
        """
        client = MockKernelClientWithPartialExecution(error_msg_has_wrong_parent_id=True)
        msg_id = client.execute("df['x'] = new_years_model.predict(df)", store_history=False)

        # Track all errors in a window
        errors_in_window = []
        execution_complete = False

        while client.message_queue and not execution_complete:
            msg = client.get_iopub_msg(timeout=1.0)
            msg_type = msg['header']['msg_type']

            # Capture ALL error messages, regardless of parent_header
            if msg_type == 'error':
                errors_in_window.append(msg['content'])

            # Only check parent_header for status:idle
            if msg_type == 'status':
                if msg['parent_header'].get('msg_id') == msg_id:
                    if msg['content']['execution_state'] == 'idle':
                        execution_complete = True

        # Should have captured the error
        assert len(errors_in_window) > 0
        assert errors_in_window[0]['ename'] == 'NameError'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
