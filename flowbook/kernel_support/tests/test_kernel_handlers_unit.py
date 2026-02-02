"""
Unit tests for kernel command handlers.

These tests verify the handler implementations work correctly
without needing a running kernel or comm channel.
"""

import pytest
from unittest.mock import Mock, MagicMock

from flowbook.kernel_support.kernel_command_handlers import KernelCommandHandlers
from flowbook.kernel_support.kernel_commands import (
    CheckpointSaveRequest,
    CheckpointRestoreRequest,
    CheckpointDeleteRequest,
    CheckpointListRequest,
    CheckpointCompareRequest,
    CheckpointClearRequest,
    EnableScaleneRequest,
    DisableScaleneRequest,
    ForceCheckpointsRequest,
)


@pytest.fixture
def mock_kernel():
    """Create a mock FlowbookKernel for testing."""
    kernel = Mock()
    kernel.shell = Mock()
    kernel.shell.user_ns = {'x': 1, 'y': 2}
    kernel._checkpoint = Mock()
    kernel._use_scalene = False
    kernel._force_checkpoints = False
    return kernel


@pytest.fixture
def handlers(mock_kernel):
    """Create KernelCommandHandlers with mock kernel."""
    return KernelCommandHandlers(mock_kernel)


class TestCheckpointHandlers:
    """Test checkpoint handler implementations."""

    def test_handle_checkpoint_save(self, handlers, mock_kernel):
        """Test checkpoint save handler."""
        from flowbook.kernel_support.extended_types import AtomicType

        # Setup mock with proper TypeModel objects
        mock_kernel._checkpoint.save.return_value = (
            {'x': AtomicType(kind="Atomic", type_name="int"), 'y': AtomicType(kind="Atomic", type_name="int")},  # saved
            {},  # removed
        )

        # Execute
        req = CheckpointSaveRequest(name="test")
        response = handlers.handle_checkpoint_save(req)

        # Verify
        assert response.status == "ok"
        assert "test" in response.message
        assert len(response.saved) == 2
        assert len(response.removed) == 0
        assert response.duration >= 0
        mock_kernel._checkpoint.save.assert_called_once_with("test", mock_kernel.shell.user_ns)

    def test_handle_checkpoint_restore(self, handlers, mock_kernel):
        """Test checkpoint restore handler."""
        req = CheckpointRestoreRequest(name="test")
        response = handlers.handle_checkpoint_restore(req)

        assert response.status == "ok"
        assert "test" in response.message
        mock_kernel._checkpoint.restore.assert_called_once_with("test", mock_kernel.shell.user_ns)

    def test_handle_checkpoint_delete(self, handlers, mock_kernel):
        """Test checkpoint delete handler."""
        req = CheckpointDeleteRequest(name="test")
        response = handlers.handle_checkpoint_delete(req)

        assert response.status == "ok"
        assert "test" in response.message
        mock_kernel._checkpoint.delete.assert_called_once_with("test")

    def test_handle_checkpoint_list(self, handlers, mock_kernel):
        """Test checkpoint list handler."""
        mock_kernel._checkpoint.list.return_value = ["cp1", "cp2", "cp3"]

        req = CheckpointListRequest()
        response = handlers.handle_checkpoint_list(req)

        assert response.status == "ok"
        assert len(response.checkpoints) == 3
        assert "cp1" in response.checkpoints

    def test_handle_checkpoint_compare(self, handlers, mock_kernel):
        """Test checkpoint compare handler."""
        # Setup mocks
        from flowbook.kernel_support.checkpoint import Checkpoint
        from flowbook.kernel_support.types import DiffResult

        cp1 = Checkpoint("cp1", {'x': 1}, {})
        cp2 = Checkpoint("cp2", {'x': 2}, {})
        mock_kernel._checkpoint.get.side_effect = [cp1, cp2]

        req = CheckpointCompareRequest(name1="cp1", name2="cp2")
        response = handlers.handle_checkpoint_compare(req)

        assert response.status == "ok"
        assert isinstance(response.diff, DiffResult)

    def test_handle_checkpoint_compare_with_keys_to_include(self, handlers, mock_kernel):
        """Test checkpoint compare handler with keys_to_include parameter."""
        from flowbook.kernel_support.checkpoint import Checkpoint
        from flowbook.kernel_support.types import DiffResult

        # Create checkpoints with multiple variables, some different
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 2, 'z': 3}, {})
        mock_kernel._checkpoint.get.side_effect = [cp1, cp2]

        # Compare only 'y' and 'z' (which are the same)
        req = CheckpointCompareRequest(
            name1="cp1",
            name2="cp2",
            keys_to_include={'y', 'z'}
        )
        response = handlers.handle_checkpoint_compare(req)

        assert response.status == "ok"
        assert isinstance(response.diff, DiffResult)
        # Should have no differences since y and z are the same
        assert len(response.diff.differences) == 0

    def test_handle_checkpoint_compare_with_keys_showing_difference(self, handlers, mock_kernel):
        """Test that keys_to_include properly filters to show only requested differences."""
        from flowbook.kernel_support.checkpoint import Checkpoint
        from flowbook.kernel_support.types import DiffResult

        # Create checkpoints with multiple variables, some different
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 999, 'z': 3}, {})
        mock_kernel._checkpoint.get.side_effect = [cp1, cp2]

        # Compare only 'x' (which is different)
        req = CheckpointCompareRequest(
            name1="cp1",
            name2="cp2",
            keys_to_include={'x'}
        )
        response = handlers.handle_checkpoint_compare(req)

        assert response.status == "ok"
        assert isinstance(response.diff, DiffResult)
        # Should have difference in 'x', but not 'y' (even though it's different)
        assert 'x' in response.diff.differences
        assert 'y' not in response.diff.differences

    def test_handle_checkpoint_clear(self, handlers, mock_kernel):
        """Test checkpoint clear handler."""
        req = CheckpointClearRequest()
        response = handlers.handle_checkpoint_clear(req)

        assert response.status == "ok"
        mock_kernel._checkpoint.clear.assert_called_once()


class TestFeatureToggleHandlers:
    """Test feature toggle handler implementations."""

    def test_handle_enable_scalene(self, handlers, mock_kernel):
        """Test enable scalene handler."""
        req = EnableScaleneRequest()
        response = handlers.handle_enable_scalene(req)

        assert response.status == "ok"
        assert mock_kernel._use_scalene is True

    def test_handle_disable_scalene(self, handlers, mock_kernel):
        """Test disable scalene handler."""
        mock_kernel._use_scalene = True

        req = DisableScaleneRequest()
        response = handlers.handle_disable_scalene(req)

        assert response.status == "ok"
        assert mock_kernel._use_scalene is False

    def test_handle_force_checkpoints_enable(self, handlers, mock_kernel):
        """Test force checkpoints enable handler."""
        req = ForceCheckpointsRequest(enabled=True)
        response = handlers.handle_force_checkpoints(req)

        assert response.status == "ok"
        assert response.enabled is True
        assert mock_kernel._force_checkpoints is True

    def test_handle_force_checkpoints_disable(self, handlers, mock_kernel):
        """Test force checkpoints disable handler."""
        mock_kernel._force_checkpoints = True

        req = ForceCheckpointsRequest(enabled=False)
        response = handlers.handle_force_checkpoints(req)

        assert response.status == "ok"
        assert response.enabled is False
        assert mock_kernel._force_checkpoints is False


class TestHandlerRegistry:
    """Test handler registry functionality."""

    def test_get_handler_valid(self, handlers):
        """Test getting a valid handler."""
        handler = handlers.get_handler("checkpoint_save")
        assert callable(handler)

    def test_get_handler_invalid(self, handlers):
        """Test getting an invalid handler raises error."""
        with pytest.raises(ValueError) as exc_info:
            handlers.get_handler("invalid_command")

        assert "Unknown command" in str(exc_info.value)

    def test_all_commands_registered(self, handlers):
        """Test that all expected commands are registered."""
        expected_commands = [
            "checkpoint_save",
            "checkpoint_restore",
            "checkpoint_delete",
            "checkpoint_list",
            "checkpoint_compare",
            "checkpoint_clear",
            "enable_scalene",
            "disable_scalene",
            "force_checkpoints",
        ]

        for cmd in expected_commands:
            handler = handlers.get_handler(cmd)
            assert callable(handler), f"Handler for {cmd} should be callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
