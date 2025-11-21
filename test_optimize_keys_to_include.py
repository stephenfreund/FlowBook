"""
Integration tests for optimization validation with keys_to_include.

These tests verify that the optimization validation correctly uses
keys_to_include to only check live/output variables for equivalence.

To run these tests:
    pytest data_ferret/server/commands/test_optimize_keys_to_include.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from data_ferret.server.commands.optimize import CodeExecutionOrchestrator
from data_ferret.kernel.types import TestCodeSuccess, DiffResult


class TestCodeExecutionOrchestratorKeysToInclude:
    """Test that CodeExecutionOrchestrator uses keys_to_include correctly."""

    @pytest.fixture
    def mock_kernel_client(self):
        """Create a mock FerretKernelClient."""
        client = Mock()
        client.execute = Mock(return_value="msg_id_123")
        client.get_iopub_msg = Mock()
        return client

    @pytest.fixture
    def orchestrator(self, mock_kernel_client):
        """Create CodeExecutionOrchestrator with mock kernel client."""
        return CodeExecutionOrchestrator(mock_kernel_client)

    def test_test_code_uses_keys_to_include(self, orchestrator, mock_kernel_client):
        """Test that test_code passes output_variables as keys_to_include."""
        # Setup: Mock the kernel command client's checkpoint_compare
        with patch.object(orchestrator.cmd_client, 'checkpoint_save'):
            with patch.object(orchestrator.cmd_client, 'checkpoint_restore'):
                with patch.object(orchestrator.cmd_client, 'checkpoint_compare') as mock_compare:
                    # Setup mock execution
                    mock_kernel_client.get_iopub_msg.side_effect = [
                        # Original execution completes successfully
                        {'parent_header': {'msg_id': 'msg_id_123'},
                         'header': {'msg_type': 'status'},
                         'content': {'execution_state': 'idle'}},
                        # Modified execution completes successfully
                        {'parent_header': {'msg_id': 'msg_id_123'},
                         'header': {'msg_type': 'status'},
                         'content': {'execution_state': 'idle'}},
                    ]

                    # Mock checkpoint_compare to return no differences
                    mock_compare.return_value = Mock(
                        diff=DiffResult(differences={})
                    )

                    # Define output variables to check
                    output_variables = {'result', 'output_value'}

                    # Execute test_code
                    result = orchestrator.test_code(
                        original_code="result = 1 + 1",
                        modified_code="result = 2",
                        output_variables=output_variables,
                    )

                    # Verify checkpoint_compare was called with keys_to_include
                    mock_compare.assert_called_once()
                    call_args = mock_compare.call_args

                    # Check that keys_to_include was passed
                    assert 'keys_to_include' in call_args.kwargs
                    assert call_args.kwargs['keys_to_include'] == output_variables

    def test_test_code_filters_comparison_to_output_variables(self, orchestrator, mock_kernel_client):
        """Test that only output_variables are compared, not all variables."""
        # This is a more complete integration test that verifies the filtering behavior

        with patch.object(orchestrator.cmd_client, 'checkpoint_save'):
            with patch.object(orchestrator.cmd_client, 'checkpoint_restore'):
                with patch.object(orchestrator.cmd_client, 'checkpoint_compare') as mock_compare:
                    # Setup mock execution
                    mock_kernel_client.get_iopub_msg.side_effect = [
                        # Original execution
                        {'parent_header': {'msg_id': 'msg_id_123'},
                         'header': {'msg_type': 'status'},
                         'content': {'execution_state': 'idle'}},
                        # Modified execution
                        {'parent_header': {'msg_id': 'msg_id_123'},
                         'header': {'msg_type': 'status'},
                         'content': {'execution_state': 'idle'}},
                    ]

                    # Simulate a diff that would include 'temp_var' (which we don't care about)
                    # but the keys_to_include should filter it out
                    mock_compare.return_value = Mock(
                        diff=DiffResult(differences={})  # No differences in output_variables
                    )

                    # Only care about 'result', not 'temp_var' or other intermediates
                    output_variables = {'result'}

                    result = orchestrator.test_code(
                        original_code="""
temp_var = 100
intermediate = temp_var * 2
result = intermediate + 5
""",
                        modified_code="""
temp_var = 999  # Different, but we don't care
intermediate = temp_var * 2  # Also different
result = 205  # Same result!
""",
                        output_variables=output_variables,
                    )

                    # Verify that keys_to_include was set to only output_variables
                    call_args = mock_compare.call_args
                    assert call_args.kwargs['keys_to_include'] == {'result'}

    def test_test_code_empty_output_variables(self, orchestrator, mock_kernel_client):
        """Test behavior when output_variables is empty."""
        with patch.object(orchestrator.cmd_client, 'checkpoint_save'):
            with patch.object(orchestrator.cmd_client, 'checkpoint_restore'):
                with patch.object(orchestrator.cmd_client, 'checkpoint_compare') as mock_compare:
                    # Setup mock execution
                    mock_kernel_client.get_iopub_msg.side_effect = [
                        {'parent_header': {'msg_id': 'msg_id_123'},
                         'header': {'msg_type': 'status'},
                         'content': {'execution_state': 'idle'}},
                        {'parent_header': {'msg_id': 'msg_id_123'},
                         'header': {'msg_type': 'status'},
                         'content': {'execution_state': 'idle'}},
                    ]

                    mock_compare.return_value = Mock(
                        diff=DiffResult(differences={})
                    )

                    # Empty output_variables set
                    result = orchestrator.test_code(
                        original_code="x = 1",
                        modified_code="x = 2",
                        output_variables=set(),
                    )

                    # Verify checkpoint_compare was called with empty set
                    call_args = mock_compare.call_args
                    assert call_args.kwargs['keys_to_include'] == set()


class TestValidationHelperKeysToInclude:
    """Test that ValidationHelper properly uses keys_to_include during validation."""

    def test_validate_optimization_uses_modified_globals(self):
        """Test that validate_optimization passes modified_globals as output_variables."""
        from data_ferret.server.commands.optimize import ValidationHelper

        mock_kernel_client = Mock()

        # Mock CodeExecutionOrchestrator
        with patch('data_ferret.server.commands.optimize.CodeExecutionOrchestrator') as mock_orch_class:
            mock_orch = Mock()
            mock_orch_class.return_value = mock_orch

            # Mock test_code to return success
            mock_orch.test_code.return_value = TestCodeSuccess(
                diff=DiffResult(differences={}),
                original_duration=1.0,
                modified_duration=0.5,
                speedup=2.0
            )

            modified_globals = {'output_var', 'result'}

            # Call validate_optimization
            is_valid, error_msg, test_result = ValidationHelper.validate_optimization(
                original_code="x = 1",
                optimized_code="x = 2",
                modified_globals=modified_globals,
                kernel_client=mock_kernel_client,
            )

            # Verify test_code was called with output_variables=modified_globals
            mock_orch.test_code.assert_called_once()
            call_args = mock_orch.test_code.call_args
            assert call_args.kwargs['output_variables'] == modified_globals

    def test_validate_optimization_no_globals_skips_test(self):
        """Test that validation is skipped when modified_globals is empty."""
        from data_ferret.server.commands.optimize import ValidationHelper

        mock_kernel_client = Mock()

        # Call with empty modified_globals
        is_valid, error_msg, test_result = ValidationHelper.validate_optimization(
            original_code="x = 1",
            optimized_code="x = 2",
            modified_globals=set(),  # Empty
            kernel_client=mock_kernel_client,
        )

        # Should automatically pass without running tests
        assert is_valid is True
        assert error_msg is None
        assert test_result is None


# Note: Additional integration tests could be added here but may require
# a running kernel or more complex setup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
