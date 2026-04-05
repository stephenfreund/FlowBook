"""Tests for KernelCommandHandlers."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from flowbook.kernel_support.kernel_command_handlers import KernelCommandHandlers
from flowbook.kernel_support.kernel_commands import (
    CheckpointSaveRequest,
    CheckpointRestoreRequest,
    CheckpointDeleteRequest,
    CheckpointListRequest,
    CheckpointClearRequest,
    EnableScaleneRequest,
    DisableScaleneRequest,
    ForceCheckpointsRequest,
)
from flowbook.kernel_support.extended_types import get_type_model


def _mock_kernel():
    """Create a mock FlowbookKernel with essential attributes."""
    kernel = MagicMock()
    kernel.shell = MagicMock()
    kernel.shell.user_ns = {"x": 1, "y": 2}
    kernel.shell.user_global_ns = {"x": 1, "y": 2}
    kernel._checkpoint = MagicMock()
    kernel._use_scalene = False
    kernel._force_checkpoints = False
    kernel._use_global_tracking = False
    return kernel


class TestGetHandler:
    def test_valid_command(self):
        handlers = KernelCommandHandlers(_mock_kernel())
        handler = handlers.get_handler("checkpoint_save")
        assert callable(handler)

    def test_all_commands_registered(self):
        handlers = KernelCommandHandlers(_mock_kernel())
        expected = [
            "checkpoint_save", "checkpoint_restore", "checkpoint_delete",
            "checkpoint_list", "checkpoint_compare", "checkpoint_compare_leq",
            "checkpoint_clear", "enable_scalene", "disable_scalene",
            "force_checkpoints", "enable_global_tracking", "disable_global_tracking",
        ]
        for cmd in expected:
            assert callable(handlers.get_handler(cmd))

    def test_unknown_command_raises(self):
        handlers = KernelCommandHandlers(_mock_kernel())
        with pytest.raises(ValueError, match="Unknown command"):
            handlers.get_handler("nonexistent")


class TestCheckpointSave:
    def test_success(self):
        kernel = _mock_kernel()
        kernel._checkpoint.save.return_value = ({"x": get_type_model(42)}, {})
        handlers = KernelCommandHandlers(kernel)

        resp = handlers.handle_checkpoint_save(CheckpointSaveRequest(name="cp1"))
        assert resp.status == "ok"
        assert "x" in resp.saved
        assert resp.duration > 0

    def test_returns_error_on_exception(self):
        kernel = _mock_kernel()
        kernel._checkpoint.save.side_effect = RuntimeError("boom")
        handlers = KernelCommandHandlers(kernel)

        resp = handlers.handle_checkpoint_save(CheckpointSaveRequest(name="cp1"))
        assert resp.status == "error"
        assert "boom" in resp.message
        assert resp.traceback is not None

    def test_removes_unsaveable_vars(self):
        kernel = _mock_kernel()
        kernel._checkpoint.save.return_value = ({"x": get_type_model(42)}, {"bad_var": get_type_model(object())})
        kernel.shell.user_ns = {"x": 1, "bad_var": object()}
        handlers = KernelCommandHandlers(kernel)

        resp = handlers.handle_checkpoint_save(CheckpointSaveRequest(name="cp1"))
        assert resp.status == "ok"
        assert "bad_var" not in kernel.shell.user_ns


class TestCheckpointRestore:
    def test_success(self):
        kernel = _mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_checkpoint_restore(CheckpointRestoreRequest(name="cp1"))
        assert resp.status == "ok"
        kernel._checkpoint.restore.assert_called_once_with("cp1", kernel.shell.user_ns)

    def test_error_on_missing(self):
        kernel = _mock_kernel()
        kernel._checkpoint.restore.side_effect = KeyError("cp1")
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_checkpoint_restore(CheckpointRestoreRequest(name="cp1"))
        assert resp.status == "error"


class TestCheckpointDelete:
    def test_success(self):
        kernel = _mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_checkpoint_delete(CheckpointDeleteRequest(name="cp1"))
        assert resp.status == "ok"

    def test_not_found(self):
        kernel = _mock_kernel()
        kernel._checkpoint.delete.side_effect = KeyError("cp1")
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_checkpoint_delete(CheckpointDeleteRequest(name="cp1"))
        assert resp.status == "error"
        assert "not found" in resp.message


class TestCheckpointList:
    def test_success(self):
        kernel = _mock_kernel()
        kernel._checkpoint.list.return_value = ["cp1", "cp2"]
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_checkpoint_list(CheckpointListRequest())
        assert resp.status == "ok"
        assert resp.checkpoints == ["cp1", "cp2"]


class TestCheckpointClear:
    def test_success(self):
        kernel = _mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_checkpoint_clear(CheckpointClearRequest())
        assert resp.status == "ok"
        kernel._checkpoint.clear.assert_called_once()


class TestEnableScalene:
    def test_success(self):
        kernel = _mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_enable_scalene(EnableScaleneRequest())
        assert resp.status == "ok"
        assert kernel._use_scalene is True


class TestDisableScalene:
    def test_success(self):
        kernel = _mock_kernel()
        kernel._use_scalene = True
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_disable_scalene(DisableScaleneRequest())
        assert resp.status == "ok"
        assert kernel._use_scalene is False


class TestForceCheckpoints:
    def test_enable(self):
        kernel = _mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_force_checkpoints(ForceCheckpointsRequest(enabled=True))
        assert resp.status == "ok"
        assert resp.enabled is True
        assert kernel._force_checkpoints is True

    def test_disable(self):
        kernel = _mock_kernel()
        handlers = KernelCommandHandlers(kernel)
        resp = handlers.handle_force_checkpoints(ForceCheckpointsRequest(enabled=False))
        assert resp.status == "ok"
        assert resp.enabled is False
