"""
Handler implementations for kernel commands.

This module provides the core implementation logic for all kernel commands,
separated from the comm channel and cell magic interfaces. This allows
the same implementation to be used from multiple entry points (comm, magic, API).

Each handler:
- Takes a typed request object (Pydantic model)
- Performs the operation on the kernel state
- Returns a typed response object (Pydantic model)
- Raises exceptions for errors (caught by comm/magic layer)
"""

import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from data_ferret.kernel.checkpoint import Checkpoint, checkpoint_diff
from data_ferret.kernel.kernel_commands import (
    CheckpointSaveRequest,
    CheckpointSaveResponse,
    CheckpointRestoreRequest,
    CheckpointRestoreResponse,
    CheckpointDeleteRequest,
    CheckpointDeleteResponse,
    CheckpointListRequest,
    CheckpointListResponse,
    CheckpointCompareRequest,
    CheckpointCompareResponse,
    CheckpointClearRequest,
    CheckpointClearResponse,
    EnableScaleneRequest,
    EnableScaleneResponse,
    DisableScaleneRequest,
    DisableScaleneResponse,
    ForceCheckpointsRequest,
    ForceCheckpointsResponse,
    KernelCommandRequest,
    KernelCommandResponse,
    ProgressMessage,
)

if TYPE_CHECKING:
    # Avoid circular import at runtime
    from data_ferret.kernel.ferret_kernel import FerretKernel


class KernelCommandHandlers:
    """
    Handler class for all kernel commands.

    This class encapsulates the implementation logic for kernel commands,
    providing a clean separation between the command protocol and the
    actual operations on kernel state.

    Attributes:
        kernel: Reference to the FerretKernel instance
    """

    def __init__(self, kernel: "FerretKernel"):
        """
        Initialize handler with kernel reference.

        Args:
            kernel: The FerretKernel instance to operate on
        """
        self.kernel = kernel

        # Map command names to handler methods
        self._handlers: Dict[
            str, Callable[[KernelCommandRequest], KernelCommandResponse]
        ] = {
            "checkpoint_save": self.handle_checkpoint_save,
            "checkpoint_restore": self.handle_checkpoint_restore,
            "checkpoint_delete": self.handle_checkpoint_delete,
            "checkpoint_list": self.handle_checkpoint_list,
            "checkpoint_compare": self.handle_checkpoint_compare,
            "checkpoint_clear": self.handle_checkpoint_clear,
            "enable_scalene": self.handle_enable_scalene,
            "disable_scalene": self.handle_disable_scalene,
            "force_checkpoints": self.handle_force_checkpoints,
        }

    def get_handler(
        self, command: str
    ) -> Callable[[KernelCommandRequest], KernelCommandResponse]:
        """
        Get handler function for a command.

        Args:
            command: Command name

        Returns:
            Handler function

        Raises:
            ValueError: If command is not recognized
        """
        if command not in self._handlers:
            raise ValueError(f"Unknown command: {command}")
        return self._handlers[command]

    # ========================================================================
    # Checkpoint Handlers
    # ========================================================================

    def handle_checkpoint_save(
        self, req: CheckpointSaveRequest
    ) -> CheckpointSaveResponse:
        """
        Save a checkpoint of the current kernel state.

        Args:
            req: Save request with checkpoint name

        Returns:
            Response with saved/removed variables and timing

        Raises:
            AssertionError: If shell is not set
        """
        try:
            assert self.kernel.shell is not None, "shell is not set"

            start_time = time.time()
            saved, removed = self.kernel._checkpoint.save(
                req.name, self.kernel.shell.user_ns
            )
            duration = time.time() - start_time

            # Remove variables that couldn't be saved from the namespace
            for k in removed and k in self.kernel.shell.user_ns:
                del self.kernel.shell.user_ns[k]

            return CheckpointSaveResponse(
                status="ok",
                message=f"Checkpoint '{req.name}' saved in {duration:.2f}s",
                saved=saved,
                removed=removed,
                duration=duration,
            )
        except Exception as e:
            return CheckpointSaveResponse(
                status="error",
                message=f"Failed to save checkpoint: {e}",
                saved={},
                removed={},
                duration=0,
            )

    def handle_checkpoint_restore(
        self, req: CheckpointRestoreRequest
    ) -> CheckpointRestoreResponse:
        """
        Restore a previously saved checkpoint.

        Args:
            req: Restore request with checkpoint name

        Returns:
            Response indicating success

        Raises:
            KeyError: If checkpoint doesn't exist
            AssertionError: If shell is not set
        """
        assert self.kernel.shell is not None, "shell is not set"

        self.kernel._checkpoint.restore(req.name, self.kernel.shell.user_ns)

        return CheckpointRestoreResponse(
            status="ok",
            message=f"Checkpoint '{req.name}' restored",
        )

    def handle_checkpoint_delete(
        self, req: CheckpointDeleteRequest
    ) -> CheckpointDeleteResponse:
        """
        Delete a checkpoint.

        Args:
            req: Delete request with checkpoint name

        Returns:
            Response indicating success

        Raises:
            KeyError: If checkpoint doesn't exist
        """
        try:
            self.kernel._checkpoint.delete(req.name)

            return CheckpointDeleteResponse(
                status="ok",
                message=f"Checkpoint '{req.name}' deleted",
            )

        except KeyError:
            return CheckpointDeleteResponse(
                status="error",
                message=f"Checkpoint '{req.name}' not found",
            )

    def handle_checkpoint_list(
        self, req: CheckpointListRequest
    ) -> CheckpointListResponse:
        """
        List all available checkpoints.

        Args:
            req: List request

        Returns:
            Response with list of checkpoint names
        """
        checkpoints = self.kernel._checkpoint.list()

        return CheckpointListResponse(
            status="ok",
            message=f"Found {len(checkpoints)} checkpoint(s)",
            checkpoints=checkpoints,
        )

    def handle_checkpoint_compare(
        self, req: CheckpointCompareRequest
    ) -> CheckpointCompareResponse:
        """
        Compare two checkpoints.

        Args:
            req: Compare request with two checkpoint names and optional keys filter

        Returns:
            Response with diff result

        Raises:
            KeyError: If either checkpoint doesn't exist
        """
        old = self.kernel._checkpoint.get(req.name1)
        new = self.kernel._checkpoint.get(req.name2)
        diff = checkpoint_diff(old, new, keys_to_include=req.keys_to_include)

        return CheckpointCompareResponse(
            status="ok",
            message=f"Compared '{req.name1}' and '{req.name2}'",
            diff=diff,
        )

    def handle_checkpoint_clear(
        self, req: CheckpointClearRequest
    ) -> CheckpointClearResponse:
        """
        Clear all checkpoints.

        Args:
            req: Clear request

        Returns:
            Response indicating success
        """
        self.kernel._checkpoint.clear()

        return CheckpointClearResponse(
            status="ok",
            message="All checkpoints cleared",
        )

    # ========================================================================
    # Feature Toggle Handlers
    # ========================================================================

    def handle_enable_scalene(self, req: EnableScaleneRequest) -> EnableScaleneResponse:
        """
        Enable Scalene profiling.

        Args:
            req: Enable request

        Returns:
            Response indicating success
        """
        self.kernel._use_scalene = True

        return EnableScaleneResponse(
            status="ok",
            message="Scalene profiling enabled",
        )

    def handle_disable_scalene(
        self, req: DisableScaleneRequest
    ) -> DisableScaleneResponse:
        """
        Disable Scalene profiling.

        Args:
            req: Disable request

        Returns:
            Response indicating success
        """
        self.kernel._use_scalene = False

        return DisableScaleneResponse(
            status="ok",
            message="Scalene profiling disabled",
        )

    def handle_force_checkpoints(
        self, req: ForceCheckpointsRequest
    ) -> ForceCheckpointsResponse:
        """
        Enable or disable force checkpoints mode.

        Args:
            req: Force checkpoints request with enabled flag

        Returns:
            Response with current state
        """
        self.kernel._force_checkpoints = req.enabled

        status_text = "enabled" if req.enabled else "disabled"
        return ForceCheckpointsResponse(
            status="ok",
            message=f"Force checkpoints {status_text}",
            enabled=req.enabled,
        )
