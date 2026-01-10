"""
Diagnostic tests for execution error capture in CodeExecutionOrchestrator.

These tests diagnose the bug where _wait_for_execution fails to capture
kernel error messages due to parent_header filtering issues.

Run with: python -m pytest test_execution_error_capture.py -v -s
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from flowbook.server.commands.optimize import CodeExecutionOrchestrator
from flowbook.kernel.types import ExecutionError


class MockKernelClient:
    """Mock kernel client for testing message handling."""

    def __init__(self, messages):
        """
        Args:
            messages: List of messages to return from get_iopub_msg
        """
        self.messages = messages
        self.message_index = 0
        self.executed_code = []

    def execute(self, code, store_history=False):
        """Mock execute method."""
        self.executed_code.append(code)
        return "test-msg-id-123"

    def get_iopub_msg(self, timeout=60.0):
        """Return pre-configured messages in sequence."""
        if self.message_index >= len(self.messages):
            raise TimeoutError("No more messages")

        msg = self.messages[self.message_index]
        self.message_index += 1
        return msg


def create_message(msg_type, parent_msg_id, **kwargs):
    """Helper to create Jupyter kernel messages."""
    msg = {
        'header': {
            'msg_type': msg_type,
            'msg_id': f'msg-{msg_type}-{time.time()}'
        },
        'parent_header': {
            'msg_id': parent_msg_id
        },
        'content': kwargs.get('content', {})
    }
    return msg


class TestWaitForExecutionMessageFiltering:
    """Test message filtering in _wait_for_execution."""

    def test_error_captured_with_matching_parent_header(self):
        """Test that errors ARE captured when parent_header matches."""
        msg_id = "test-msg-id-123"

        messages = [
            # Execute input message
            create_message('execute_input', msg_id),

            # Error message with matching parent_header
            create_message('error', msg_id, content={
                'ename': 'NameError',
                'evalue': "name 'new_years_model' is not defined",
                'traceback': [
                    'Traceback (most recent call last):',
                    '  File "<stdin>", line 1',
                    "NameError: name 'new_years_model' is not defined"
                ]
            }),

            # Status idle to complete execution
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = undefined_var")

        # Error SHOULD be captured
        assert error is not None
        assert error.error_type == "NameError"
        assert "new_years_model" in error.error_message

    def test_error_missed_with_mismatched_parent_header(self):
        """Test that errors are MISSED when parent_header doesn't match."""
        msg_id = "test-msg-id-123"
        wrong_msg_id = "wrong-msg-id-456"

        messages = [
            # Execute input message
            create_message('execute_input', msg_id),

            # Error message with WRONG parent_header
            create_message('error', wrong_msg_id, content={
                'ename': 'NameError',
                'evalue': "name 'new_years_model' is not defined",
                'traceback': [
                    'Traceback (most recent call last):',
                    "NameError: name 'new_years_model' is not defined"
                ]
            }),

            # Status idle with correct parent_header
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = undefined_var")

        # Error is MISSED due to parent_header mismatch
        # This test documents the BUG
        assert error is None  # BUG: Should be ExecutionError, but is None

    def test_error_missed_with_empty_parent_header(self):
        """Test that errors are MISSED when parent_header is empty."""
        msg_id = "test-msg-id-123"

        messages = [
            # Execute input message
            create_message('execute_input', msg_id),

            # Error message with EMPTY parent_header
            {
                'header': {
                    'msg_type': 'error',
                    'msg_id': 'error-msg-456'
                },
                'parent_header': {},  # Empty parent_header
                'content': {
                    'ename': 'NameError',
                    'evalue': "name 'new_years_model' is not defined",
                    'traceback': [
                        'Traceback (most recent call last):',
                        "NameError: name 'new_years_model' is not defined"
                    ]
                }
            },

            # Status idle with correct parent_header
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = undefined_var")

        # Error is MISSED due to empty parent_header
        # This test documents the BUG
        assert error is None  # BUG: Should be ExecutionError, but is None

    def test_error_missed_with_missing_parent_header(self):
        """Test that errors are MISSED when parent_header key is missing."""
        msg_id = "test-msg-id-123"

        messages = [
            # Execute input message
            create_message('execute_input', msg_id),

            # Error message with NO parent_header key
            {
                'header': {
                    'msg_type': 'error',
                    'msg_id': 'error-msg-456'
                },
                # parent_header key is missing entirely
                'content': {
                    'ename': 'NameError',
                    'evalue': "name 'new_years_model' is not defined",
                    'traceback': [
                        'Traceback (most recent call last):',
                        "NameError: name 'new_years_model' is not defined"
                    ]
                }
            },

            # Status idle with correct parent_header
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = undefined_var")

        # Error is MISSED due to missing parent_header
        # This test documents the BUG
        assert error is None  # BUG: Should be ExecutionError, but is None

    def test_multiple_errors_only_last_captured(self):
        """Test that when multiple errors occur, only the last one is captured."""
        msg_id = "test-msg-id-123"

        messages = [
            # Execute input
            create_message('execute_input', msg_id),

            # First error
            create_message('error', msg_id, content={
                'ename': 'ValueError',
                'evalue': 'first error',
                'traceback': ['ValueError: first error']
            }),

            # Second error (should overwrite first)
            create_message('error', msg_id, content={
                'ename': 'NameError',
                'evalue': 'second error',
                'traceback': ['NameError: second error']
            }),

            # Status idle
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = 1")

        # Should capture the LAST error
        assert error is not None
        assert error.error_type == "NameError"
        assert "second error" in error.error_message


class TestExecuteCodeSafelyIntegration:
    """Integration tests for _execute_code_safely with realistic scenarios."""

    def test_partial_execution_before_error(self):
        """
        Test scenario where code partially executes before error.
        This simulates the bug from the output log.
        """
        msg_id = "test-msg-id-123"

        # Simulate: First 37 lines execute, line 38 throws NameError
        messages = [
            create_message('execute_input', msg_id),

            # Some output from successful execution
            create_message('stream', msg_id, content={
                'name': 'stdout',
                'text': 'Processing data...\n'
            }),

            # Then error on line 38
            create_message('error', msg_id, content={
                'ename': 'NameError',
                'evalue': "name 'new_years_model' is not defined",
                'traceback': [
                    'Traceback (most recent call last):',
                    'Cell In[43], line 38',
                    "    df['new_years_factor'] = new_years_model.predict(df[new_years_columns])",
                    "NameError: name 'new_years_model' is not defined"
                ]
            }),

            # Status idle
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        code = """
# Line 1-37: Code that executes successfully
df['ratio'] = base_ratio
# Line 38: Error occurs here
df['new_years_factor'] = new_years_model.predict(df[new_years_columns])
"""

        duration, error = orchestrator._execute_code_safely(code)

        # Error SHOULD be captured
        assert error is not None
        assert error.error_type == "NameError"
        assert "new_years_model" in error.error_message
        assert "line 38" in error.traceback


class TestMessageSequenceVariations:
    """Test different message sequence patterns from Jupyter kernel."""

    def test_stream_before_error(self):
        """Test error capture when stream messages precede error."""
        msg_id = "test-msg-id-123"

        messages = [
            create_message('execute_input', msg_id),
            create_message('stream', msg_id, content={'name': 'stdout', 'text': 'output\n'}),
            create_message('stream', msg_id, content={'name': 'stderr', 'text': 'warning\n'}),
            create_message('error', msg_id, content={
                'ename': 'RuntimeError',
                'evalue': 'test error',
                'traceback': ['RuntimeError: test error']
            }),
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("print('hi')\nraise RuntimeError()")

        assert error is not None
        assert error.error_type == "RuntimeError"

    def test_display_data_before_error(self):
        """Test error capture when display_data messages precede error."""
        msg_id = "test-msg-id-123"

        messages = [
            create_message('execute_input', msg_id),
            create_message('display_data', msg_id, content={
                'data': {'text/plain': 'Figure(...)'},
                'metadata': {}
            }),
            create_message('error', msg_id, content={
                'ename': 'ValueError',
                'evalue': 'invalid value',
                'traceback': ['ValueError: invalid value']
            }),
            create_message('status', msg_id, content={'execution_state': 'idle'})
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("plt.plot()\nraise ValueError()")

        assert error is not None
        assert error.error_type == "ValueError"


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_status_idle_before_error_message(self):
        """Test what happens if status:idle arrives before error message."""
        msg_id = "test-msg-id-123"

        messages = [
            create_message('execute_input', msg_id),
            # Status idle arrives FIRST
            create_message('status', msg_id, content={'execution_state': 'idle'}),
            # Error arrives AFTER idle (shouldn't happen, but let's test)
            create_message('error', msg_id, content={
                'ename': 'NameError',
                'evalue': 'undefined',
                'traceback': ['NameError: undefined']
            })
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = undefined")

        # Since idle arrived first, method returns before seeing error
        # This documents current behavior
        assert error is None  # Returns None because loop exits at idle

    def test_timeout_during_execution(self):
        """Test timeout handling."""
        msg_id = "test-msg-id-123"

        # Only send execute_input, no status:idle -> will timeout
        messages = [
            create_message('execute_input', msg_id),
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        # Should handle timeout gracefully
        duration, error = orchestrator._execute_code_safely("while True: pass")

        # Current implementation catches exception in _wait_for_execution
        # and continues loop, but eventually times out
        # Timeout exceptions are caught in except block at line 325
        assert duration >= 0  # Should complete, even if with timeout


class TestRealWorldScenarios:
    """Test scenarios based on real-world kernel behavior."""

    def test_jupyter_kernel_message_format(self):
        """
        Test with message format exactly as Jupyter kernel sends it.

        This tests the most realistic scenario based on Jupyter protocol.
        """
        msg_id = "abc123-def456-ghi789"

        messages = [
            # Real Jupyter message format
            {
                'header': {
                    'msg_id': '1234-5678',
                    'username': 'username',
                    'session': 'session-id',
                    'msg_type': 'execute_input',
                    'version': '5.3'
                },
                'parent_header': {
                    'msg_id': msg_id,
                    'username': 'username',
                    'session': 'session-id',
                    'msg_type': 'execute_request',
                    'version': '5.3'
                },
                'content': {
                    'code': 'x = undefined',
                    'execution_count': 1
                },
                'metadata': {}
            },

            # Error message
            {
                'header': {
                    'msg_id': 'error-1234',
                    'username': 'username',
                    'session': 'session-id',
                    'msg_type': 'error',
                    'version': '5.3'
                },
                'parent_header': {
                    'msg_id': msg_id,
                    'username': 'username',
                    'session': 'session-id',
                    'msg_type': 'execute_request',
                    'version': '5.3'
                },
                'content': {
                    'ename': 'NameError',
                    'evalue': "name 'undefined' is not defined",
                    'traceback': [
                        '\x1b[0;31m---------------------------------------------------------------------------\x1b[0m',
                        '\x1b[0;31mNameError\x1b[0m                                 Traceback (most recent call last)',
                        'Cell In[1], line 1\n\x1b[0;32m----> 1\x1b[0m x \x1b[38;5;241m=\x1b[39m \x1b[43mundefined\x1b[49m\n',
                        "\x1b[0;31mNameError\x1b[0m: name 'undefined' is not defined"
                    ]
                },
                'metadata': {}
            },

            # Status message
            {
                'header': {
                    'msg_id': 'status-5678',
                    'username': 'username',
                    'session': 'session-id',
                    'msg_type': 'status',
                    'version': '5.3'
                },
                'parent_header': {
                    'msg_id': msg_id,
                    'username': 'username',
                    'session': 'session-id',
                    'msg_type': 'execute_request',
                    'version': '5.3'
                },
                'content': {
                    'execution_state': 'idle'
                },
                'metadata': {}
            }
        ]

        client = MockKernelClient(messages)
        orchestrator = CodeExecutionOrchestrator(client)

        duration, error = orchestrator._execute_code_safely("x = undefined")

        # With full Jupyter message format, error SHOULD be captured
        assert error is not None
        assert error.error_type == "NameError"
        assert "undefined" in error.error_message


class TestActualKernelClient:
    """Tests using actual FlowbookKernelClient (requires running kernel)."""

    @pytest.mark.skip(reason="Requires running Jupyter kernel - use for manual testing")
    def test_with_real_kernel_name_error(self):
        """
        Manual test with real kernel to observe actual message format.

        To run this test:
        1. Remove @pytest.mark.skip
        2. Start a Jupyter kernel
        3. Run: pytest test_execution_error_capture.py::TestActualKernelClient -v -s
        """
        from jupyter_client import BlockingKernelClient
        from jupyter_client.manager import start_new_kernel

        # Start a real kernel
        km, kc = start_new_kernel()

        try:
            orchestrator = CodeExecutionOrchestrator(kc)

            # Execute code that will cause NameError
            duration, error = orchestrator._execute_code_safely(
                "df['new_years_factor'] = new_years_model.predict(df[new_years_columns])"
            )

            # Print results for inspection
            print(f"\nDuration: {duration}")
            print(f"Error captured: {error is not None}")
            if error:
                print(f"Error type: {error.error_type}")
                print(f"Error message: {error.error_message}")
                print(f"Traceback:\n{error.traceback}")
            else:
                print("ERROR WAS NOT CAPTURED!")

            # Assert error was captured
            assert error is not None, "Error should have been captured"
            assert error.error_type == "NameError"

        finally:
            kc.stop_channels()
            km.shutdown_kernel()
