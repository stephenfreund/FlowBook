"""
Client for interacting with kernel commands via comm channel.

This module provides a Python client for sending commands to the FerretKernel
through the kernel_command comm channel. It handles the low-level comm protocol
and provides a clean, typed API for each command.

Example usage:
    >>> from jupyter_client import BlockingKernelClient
    >>> kc = BlockingKernelClient()
    >>> kc.load_connection_file('kernel-12345.json')
    >>> kc.start_channels()
    >>>
    >>> client = KernelCommandClient(kc)
    >>> response = client.checkpoint_save("my_checkpoint")
    >>> print(f"Saved {len(response.saved)} variables")
"""

import time
import uuid
from typing import Callable, List, Optional, Set
from jupyter_client import BlockingKernelClient

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
    CheckpointCompareLeqRequest,
    CheckpointCompareLeqResponse,
    CheckpointClearRequest,
    CheckpointClearResponse,
    EnableScaleneRequest,
    EnableScaleneResponse,
    DisableScaleneRequest,
    DisableScaleneResponse,
    ForceCheckpointsRequest,
    ForceCheckpointsResponse,
    ProgressMessage,
    FinalMessage,
)
from data_ferret.util.output import error, timer


class KernelCommandError(Exception):
    """Exception raised when a kernel command fails."""

    pass


class KernelCommandClient:
    """
    Client for sending commands to FerretKernel via comm channel.

    This class provides a high-level API for interacting with kernel commands,
    handling the comm protocol details internally.

    Attributes:
        kernel_client: The Jupyter kernel client to use for communication
        timeout: Default timeout for command responses (seconds)
    """

    def __init__(
        self, kernel_client: BlockingKernelClient, timeout: float = 30, retries: int = 1
    ):
        """
        Initialize the kernel command client.

        Args:
            kernel_client: Jupyter kernel client with active channels
            timeout: Default timeout for responses in seconds
        """
        self.kernel_client = kernel_client
        self.timeout = timeout
        self.retries = retries

    def _log_error_response(self, response, operation: str):
        """
        Log error details from a response with status='error'.

        Args:
            response: Response object with status, message, and optional traceback
            operation: Description of the operation that failed
        """
        error(f"{operation} failed: {response.message}")
        if hasattr(response, 'traceback') and response.traceback:
            error(f"Server traceback:\n{response.traceback}")

    def _send_command(
        self,
        request: dict,
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Send a command and wait for response.

        Args:
            request: Request dictionary (serialized Pydantic model)
            progress_callback: Optional callback for progress messages
            timeout: Optional timeout override

        Returns:
            Response dictionary

        Raises:
            KernelCommandError: If command fails or times out
        """
        timeout = timeout or self.timeout

        # Open comm with kernel
        comm_id = str(uuid.uuid4())
        try:
            # Send comm_open message
            msg = self.kernel_client.session.msg(
                "comm_open",
                {
                    "comm_id": comm_id,
                    "target_name": "kernel_command",
                    "target_module": "",
                    "data": request,
                },
            )
            self.kernel_client.shell_channel.send(msg)

            # Wait for comm_msg responses with our comm_id
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    msg = self.kernel_client.iopub_channel.get_msg(timeout=1.0)

                    # Only process comm_msg messages with our comm_id
                    if msg["msg_type"] == "comm_msg":
                        msg_comm_id = msg["content"].get("comm_id")
                        if msg_comm_id != comm_id:
                            # Not our message, skip it
                            continue

                        data = msg["content"]["data"]

                        # Check message type
                        msg_type = data.get("type")

                        if msg_type == "progress":
                            # Progress message
                            if progress_callback:
                                progress_msg = ProgressMessage(**data)
                                progress_callback(progress_msg.message)

                        elif msg_type == "final":
                            # Final response
                            final_msg = FinalMessage(**data)

                            if not final_msg.ok:
                                raise KernelCommandError(
                                    f"Command failed: {final_msg.error}"
                                )

                            return final_msg.response

                except TimeoutError:
                    # No message yet, continue waiting
                    continue
                except Exception:
                    # Ignore other exceptions (e.g., Empty queue)
                    continue

            raise KernelCommandError(f"Command timed out after {timeout}s")

        finally:
            # Close comm if it was opened
            if comm_id:
                try:
                    close_msg = self.kernel_client.session.msg(
                        "comm_close",
                        {
                            "comm_id": comm_id,
                        },
                    )
                    self.kernel_client.shell_channel.send(close_msg)
                except Exception:
                    pass  # Best effort cleanup

    # ========================================================================
    # Checkpoint Commands
    # ========================================================================

    def checkpoint_save(
        self,
        name: str,
        timeout: Optional[float] = None,
    ) -> CheckpointSaveResponse:
        """
        Save a checkpoint of the current kernel state.

        Args:
            name: Name for the checkpoint
            timeout: Optional timeout override

        Returns:
            Response with saved/removed variables and timing

        Raises:
            KernelCommandError: If command fails
        """
        with timer(
            key="checkpoint_save",
            message=f"KernelCommandClient: Save checkpoint {name}",
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointSaveRequest(name=name)
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointSaveResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint save")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to save checkpoint: {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to save checkpoint after {self.retries} attempts"
        )

    def checkpoint_restore(
        self,
        name: str,
        timeout: Optional[float] = None,
    ) -> CheckpointRestoreResponse:
        """
        Restore a previously saved checkpoint.

        Args:
            name: Name of checkpoint to restore
            timeout: Optional timeout override

        Returns:
            Response indicating success

        Raises:
            KernelCommandError: If command fails or checkpoint doesn't exist
        """
        with timer(
            key="checkpoint_restore",
            message=f"KernelCommandClient: Restore checkpoint {name}",
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointRestoreRequest(name=name)
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointRestoreResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint restore")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to restore checkpoint: {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to restore checkpoint after {self.retries} attempts"
        )

    def checkpoint_delete(
        self,
        name: str,
        timeout: Optional[float] = None,
    ) -> CheckpointDeleteResponse:
        """
        Delete a checkpoint.

        Args:
            name: Name of checkpoint to delete
            timeout: Optional timeout override

        Returns:
            Response indicating success

        Raises:
            KernelCommandError: If command fails or checkpoint doesn't exist
        """
        with timer(
            key="checkpoint_delete",
            message=f"KernelCommandClient: Delete checkpoint {name}",
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointDeleteRequest(name=name)
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointDeleteResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint delete")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to delete checkpoint: {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to delete checkpoint after {self.retries} attempts"
        )

    def checkpoint_list(
        self,
        timeout: Optional[float] = None,
    ) -> CheckpointListResponse:
        """
        List all available checkpoints.

        Args:
            timeout: Optional timeout override

        Returns:
            Response with list of checkpoint names

        Raises:
            KernelCommandError: If command fails
        """
        with timer(
            key="checkpoint_list", message="KernelCommandClient: List checkpoints"
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointListRequest()
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointListResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint list")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to list checkpoints: {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to list checkpoints after {self.retries} attempts"
        )

    def checkpoint_compare(
        self,
        name1: str,
        name2: str,
        keys_to_include: Optional[Set[str]] = None,
        timeout: Optional[float] = None,
    ) -> CheckpointCompareResponse:
        """
        Compare two checkpoints.

        Args:
            name1: First checkpoint name
            name2: Second checkpoint name
            keys_to_include: Optional set of variable names to include in comparison
            timeout: Optional timeout override

        Returns:
            Response with diff result

        Raises:
            KernelCommandError: If command fails or checkpoints don't exist
        """
        with timer(
            key="checkpoint_compare",
            message=f"KernelCommandClient: Compare checkpoints {name1} and {name2}",
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointCompareRequest(
                        name1=name1, name2=name2, keys_to_include=keys_to_include
                    )
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointCompareResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint compare")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to compare checkpoints: {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to compare checkpoints after {self.retries} attempts"
        )

    def checkpoint_compare_leq(
        self,
        name1: str,
        name2: str,
        keys_to_include: Optional[Set[str]] = None,
        timeout: Optional[float] = None,
    ) -> CheckpointCompareLeqResponse:
        """
        Compare two checkpoints using leq semantics.

        Leq mode allows extra keys in the second checkpoint and extra columns
        in DataFrames. This is useful for checking that read-before-write
        variables haven't been modified by a cell.

        Args:
            name1: First checkpoint name (pre-execution state)
            name2: Second checkpoint name (post-execution state)
            keys_to_include: Optional set of variable names to include in comparison
            timeout: Optional timeout override

        Returns:
            Response with diff result and is_leq flag

        Raises:
            KernelCommandError: If command fails or checkpoints don't exist
        """
        with timer(
            key="checkpoint_compare_leq",
            message=f"KernelCommandClient: Compare checkpoints (leq) {name1} and {name2}",
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointCompareLeqRequest(
                        name1=name1, name2=name2, keys_to_include=keys_to_include
                    )
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointCompareLeqResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint compare (leq)")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to compare checkpoints (leq): {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to compare checkpoints (leq) after {self.retries} attempts"
        )

    def checkpoint_clear(
        self,
        timeout: Optional[float] = None,
    ) -> CheckpointClearResponse:
        """
        Clear all checkpoints.

        Args:
            timeout: Optional timeout override

        Returns:
            Response indicating success

        Raises:
            KernelCommandError: If command fails
        """
        with timer(
            key="checkpoint_clear", message="KernelCommandClient: Clear checkpoints"
        ):
            for _ in range(self.retries):
                try:
                    request = CheckpointClearRequest()
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = CheckpointClearResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Checkpoint clear")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to clear checkpoints: {e}")
                    time.sleep(1)
            raise KernelCommandError(
                f"Failed to clear checkpoints after {self.retries} attempts"
            )

    # ========================================================================
    # Feature Toggle Commands
    # ========================================================================

    def enable_scalene(
        self,
        timeout: Optional[float] = None,
    ) -> EnableScaleneResponse:
        """
        Enable Scalene profiling.

        Args:
            timeout: Optional timeout override

        Returns:
            Response indicating success

        Raises:
            KernelCommandError: If command fails
        """
        with timer(
            key="enable_scalene",
            message="KernelCommandClient: Enable Scalene profiling",
        ):
            for _ in range(3):
                try:
                    request = EnableScaleneRequest()
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = EnableScaleneResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Enable Scalene")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to enable Scalene profiling: {e}")
                    time.sleep(1)
        raise KernelCommandError(f"Failed to enable Scalene profiling after 3 attempts")

    def disable_scalene(
        self,
        timeout: Optional[float] = None,
    ) -> DisableScaleneResponse:
        """
        Disable Scalene profiling.

        Args:
            timeout: Optional timeout override

        Returns:
            Response indicating success

        Raises:
            KernelCommandError: If command fails
        """
        with timer(
            key="disable_scalene",
            message="KernelCommandClient: Disable Scalene profiling",
        ):
            for _ in range(3):
                try:
                    request = DisableScaleneRequest()
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = DisableScaleneResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Disable Scalene")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to disable Scalene profiling: {e}")
                    time.sleep(1)
        raise KernelCommandError(
            f"Failed to disable Scalene profiling after 3 attempts"
        )

    def force_checkpoints(
        self,
        enabled: bool,
        timeout: Optional[float] = None,
    ) -> ForceCheckpointsResponse:
        """
        Enable or disable force checkpoints mode.

        Args:
            enabled: Whether to enable force checkpoints
            timeout: Optional timeout override

        Returns:
            Response with current state

        Raises:
            KernelCommandError: If command fails
        """
        with timer(
            key="force_checkpoints",
            message=f"KernelCommandClient: Force checkpoints {enabled}",
        ):
            for _ in range(3):
                try:
                    request = ForceCheckpointsRequest(enabled=enabled)
                    response_dict = self._send_command(
                        request.model_dump(), timeout=timeout
                    )
                    response = ForceCheckpointsResponse(**response_dict)

                    # Check if server returned error status
                    if response.status == "error":
                        self._log_error_response(response, "Force checkpoints")
                        time.sleep(1)
                        continue  # Retry

                    return response
                except Exception as e:
                    error(f"Failed to force checkpoints: {e}")
                    time.sleep(1)
        raise KernelCommandError(f"Failed to force checkpoints after 3 attempts")
