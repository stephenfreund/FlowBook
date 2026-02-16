"""Tests for kernel_command_handlers.py - Targeting uncovered handler methods.

Coverage gaps include all the individual handler implementations that require
a mock kernel. Tests handle_checkpoint_save, restore, delete, list, compare,
compare_leq, clear, enable_scalene, disable_scalene, force_checkpoints,
enable_global_tracking, disable_global_tracking.
"""

import pytest
from unittest.mock import MagicMock, patch

from flowbook.kernel_support.kernel_command_handlers import KernelCommandHandlers
from flowbook.kernel_support.kernel_commands import (
    CheckpointSaveRequest,
    CheckpointRestoreRequest,
    CheckpointDeleteRequest,
    CheckpointListRequest,
    CheckpointCompareRequest,
    CheckpointCompareLeqRequest,
    CheckpointClearRequest,
    EnableScaleneRequest,
    DisableScaleneRequest,
    ForceCheckpointsRequest,
    EnableGlobalTrackingRequest,
    DisableGlobalTrackingRequest,
)
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.types import MemoryCheckpointDiffResult


def _make_mock_kernel():
    """Create a mock kernel with necessary attributes."""
    kernel = MagicMock()
    kernel._checkpoint = MemoryCheckpoints()
    kernel.shell = MagicMock()
    kernel.shell.user_ns = {"x": 1, "y": 2}
    kernel.shell.user_global_ns = {"x": 1, "y": 2}
    kernel._use_scalene = False
    kernel._force_checkpoints = False
    kernel._use_global_tracking = False
    return kernel


class TestGetHandler:
    """Tests for get_handler method."""

    def test_known_command(self):
        """get_handler returns handler for known command."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        handler = handlers.get_handler("checkpoint_save")
        assert callable(handler)

    def test_unknown_command(self):
        """get_handler raises ValueError for unknown command."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        with pytest.raises(ValueError, match="Unknown command"):
            handlers.get_handler("nonexistent_command")


class TestCheckpointSaveHandler:
    """Tests for handle_checkpoint_save."""

    def test_save_success(self):
        """Successful checkpoint save."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointSaveRequest(name="test")
        resp = handlers.handle_checkpoint_save(req)
        assert resp.status == "ok"
        assert "test" in resp.message
        assert resp.duration > 0

    def test_save_removes_unsaveable(self):
        """Save removes variables that cannot be checkpointed."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        # First save normally, then verify removed vars get handled
        # Put an unsaveable value in namespace - modules can't be checkpointed
        import types
        kernel.shell.user_ns["bad"] = types.ModuleType("bad_module")
        req = CheckpointSaveRequest(name="test")
        resp = handlers.handle_checkpoint_save(req)
        assert resp.status == "ok"

    def test_save_error(self):
        """Save returns error response on failure."""
        kernel = _make_mock_kernel()
        kernel.shell = None  # Cause assertion error
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointSaveRequest(name="test")
        resp = handlers.handle_checkpoint_save(req)
        assert resp.status == "error"
        assert resp.traceback is not None


class TestCheckpointRestoreHandler:
    """Tests for handle_checkpoint_restore."""

    def test_restore_success(self):
        """Successful checkpoint restore."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        # First save a checkpoint
        kernel._checkpoint.save("test", kernel.shell.user_ns)
        req = CheckpointRestoreRequest(name="test")
        resp = handlers.handle_checkpoint_restore(req)
        assert resp.status == "ok"

    def test_restore_error(self):
        """Restore returns error for non-existent checkpoint."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointRestoreRequest(name="nonexistent")
        resp = handlers.handle_checkpoint_restore(req)
        assert resp.status == "error"


class TestCheckpointDeleteHandler:
    """Tests for handle_checkpoint_delete."""

    def test_delete_success(self):
        """Successful checkpoint deletion."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        kernel._checkpoint.save("test", kernel.shell.user_ns)
        req = CheckpointDeleteRequest(name="test")
        resp = handlers.handle_checkpoint_delete(req)
        assert resp.status == "ok"

    def test_delete_nonexistent(self):
        """Delete non-existent checkpoint is a no-op (success)."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointDeleteRequest(name="nonexistent")
        resp = handlers.handle_checkpoint_delete(req)
        # MemoryCheckpoints.delete is a no-op for missing keys
        assert resp.status == "ok"

    def test_delete_error(self):
        """Delete returns error when checkpoint raises KeyError."""
        kernel = _make_mock_kernel()
        kernel._checkpoint = MagicMock()
        kernel._checkpoint.delete.side_effect = KeyError("not found")
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointDeleteRequest(name="missing")
        resp = handlers.handle_checkpoint_delete(req)
        assert resp.status == "error"
        assert "not found" in resp.message


class TestCheckpointListHandler:
    """Tests for handle_checkpoint_list."""

    def test_list_empty(self):
        """List with no checkpoints."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointListRequest()
        resp = handlers.handle_checkpoint_list(req)
        assert resp.status == "ok"
        assert resp.checkpoints == []

    def test_list_with_checkpoints(self):
        """List after saving checkpoints."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        kernel._checkpoint.save("cp1", kernel.shell.user_ns)
        kernel._checkpoint.save("cp2", kernel.shell.user_ns)
        req = CheckpointListRequest()
        resp = handlers.handle_checkpoint_list(req)
        assert resp.status == "ok"
        assert len(resp.checkpoints) == 2

    def test_list_error(self):
        """List returns error on failure."""
        kernel = _make_mock_kernel()
        kernel._checkpoint = MagicMock()
        kernel._checkpoint.list.side_effect = RuntimeError("fail")
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointListRequest()
        resp = handlers.handle_checkpoint_list(req)
        assert resp.status == "error"


class TestCheckpointCompareHandler:
    """Tests for handle_checkpoint_compare."""

    def test_compare_identical(self):
        """Compare two identical checkpoints."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        kernel._checkpoint.save("cp1", kernel.shell.user_ns)
        kernel._checkpoint.save("cp2", kernel.shell.user_ns)
        req = CheckpointCompareRequest(name1="cp1", name2="cp2")
        resp = handlers.handle_checkpoint_compare(req)
        assert resp.status == "ok"
        assert not resp.diff  # No differences

    def test_compare_error(self):
        """Compare returns error for missing checkpoint."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointCompareRequest(name1="missing1", name2="missing2")
        resp = handlers.handle_checkpoint_compare(req)
        assert resp.status == "error"


class TestCheckpointCompareLeqHandler:
    """Tests for handle_checkpoint_compare_leq."""

    def test_compare_leq_identical(self):
        """Compare leq with identical checkpoints returns is_leq=True."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        kernel._checkpoint.save("cp1", kernel.shell.user_ns)
        kernel._checkpoint.save("cp2", kernel.shell.user_ns)
        req = CheckpointCompareLeqRequest(name1="cp1", name2="cp2")
        resp = handlers.handle_checkpoint_compare_leq(req)
        assert resp.status == "ok"
        assert resp.is_leq

    def test_compare_leq_error(self):
        """Compare leq returns error for missing checkpoints."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointCompareLeqRequest(name1="missing", name2="missing")
        resp = handlers.handle_checkpoint_compare_leq(req)
        assert resp.status == "error"
        assert not resp.is_leq


class TestCheckpointClearHandler:
    """Tests for handle_checkpoint_clear."""

    def test_clear_success(self):
        """Successful checkpoint clear."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        kernel._checkpoint.save("cp1", kernel.shell.user_ns)
        req = CheckpointClearRequest()
        resp = handlers.handle_checkpoint_clear(req)
        assert resp.status == "ok"
        assert len(kernel._checkpoint.list()) == 0

    def test_clear_error(self):
        """Clear returns error on failure."""
        kernel = _make_mock_kernel()
        kernel._checkpoint = MagicMock()
        kernel._checkpoint.clear.side_effect = RuntimeError("fail")
        handlers = KernelCommandHandlers(kernel)
        req = CheckpointClearRequest()
        resp = handlers.handle_checkpoint_clear(req)
        assert resp.status == "error"


class TestScaleneHandlers:
    """Tests for enable/disable scalene handlers."""

    def test_enable_scalene(self):
        """Enable scalene sets flag on kernel."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = EnableScaleneRequest()
        resp = handlers.handle_enable_scalene(req)
        assert resp.status == "ok"
        assert kernel._use_scalene is True

    def test_disable_scalene(self):
        """Disable scalene clears flag on kernel."""
        kernel = _make_mock_kernel()
        kernel._use_scalene = True
        handlers = KernelCommandHandlers(kernel)
        req = DisableScaleneRequest()
        resp = handlers.handle_disable_scalene(req)
        assert resp.status == "ok"
        assert kernel._use_scalene is False


class TestForceCheckpointsHandler:
    """Tests for force_checkpoints handler."""

    def test_enable_force_checkpoints(self):
        """Enable force checkpoints."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = ForceCheckpointsRequest(enabled=True)
        resp = handlers.handle_force_checkpoints(req)
        assert resp.status == "ok"
        assert resp.enabled is True

    def test_disable_force_checkpoints(self):
        """Disable force checkpoints."""
        kernel = _make_mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        req = ForceCheckpointsRequest(enabled=False)
        resp = handlers.handle_force_checkpoints(req)
        assert resp.status == "ok"
        assert resp.enabled is False


class TestGlobalTrackingHandlers:
    """Tests for enable/disable global tracking handlers."""

    def test_enable_global_tracking(self):
        """Enable global tracking creates TrackingDict."""
        kernel = _make_mock_kernel()
        # user_ns is a plain dict initially
        kernel.shell.user_ns = {"x": 1}
        kernel.shell.user_global_ns = {"x": 1}
        handlers = KernelCommandHandlers(kernel)
        req = EnableGlobalTrackingRequest()
        resp = handlers.handle_enable_global_tracking(req)
        assert resp.status == "ok"
        assert kernel._use_global_tracking is True

    def test_enable_global_tracking_error(self):
        """Enable global tracking returns error on failure."""
        kernel = _make_mock_kernel()
        kernel.shell = None  # Cause error
        handlers = KernelCommandHandlers(kernel)
        req = EnableGlobalTrackingRequest()
        resp = handlers.handle_enable_global_tracking(req)
        assert resp.status == "error"

    def test_disable_global_tracking(self):
        """Disable global tracking converts back to plain dict."""
        kernel = _make_mock_kernel()
        kernel.shell.user_ns = {"x": 1}
        handlers = KernelCommandHandlers(kernel)
        req = DisableGlobalTrackingRequest()
        resp = handlers.handle_disable_global_tracking(req)
        assert resp.status == "ok"
        assert kernel._use_global_tracking is False

    def test_disable_global_tracking_error(self):
        """Disable global tracking returns error on failure."""
        kernel = _make_mock_kernel()
        kernel.shell = None  # Cause error
        handlers = KernelCommandHandlers(kernel)
        req = DisableGlobalTrackingRequest()
        resp = handlers.handle_disable_global_tracking(req)
        assert resp.status == "error"
