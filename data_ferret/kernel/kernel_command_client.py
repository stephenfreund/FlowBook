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
    CheckpointClearRequest,
    CheckpointClearResponse,
    TestCodeRequest,
    TestCodeResponse,
    EnableScaleneRequest,
    EnableScaleneResponse,
    DisableScaleneRequest,
    DisableScaleneResponse,
    ForceCheckpointsRequest,
    ForceCheckpointsResponse,
    ProgressMessage,
    FinalMessage,
)


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

    def __init__(self, kernel_client: BlockingKernelClient, timeout: float = 30.0):
        """
        Initialize the kernel command client.

        Args:
            kernel_client: Jupyter kernel client with active channels
            timeout: Default timeout for responses in seconds
        """
        self.kernel_client = kernel_client
        self.timeout = timeout

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
        comm_id = None
        try:
            # Send comm_open message
            msg = self.kernel_client.session.msg('comm_open', {
                'target_name': 'kernel_command',
                'data': request,
            })
            self.kernel_client.shell_channel.send(msg)

            # Wait for comm_open reply and subsequent messages
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    msg = self.kernel_client.iopub_channel.get_msg(timeout=1.0)

                    if msg['msg_type'] == 'comm_open':
                        comm_id = msg['content']['comm_id']
                        continue

                    if msg['msg_type'] == 'comm_msg':
                        data = msg['content']['data']

                        # Check message type
                        msg_type = data.get('type')

                        if msg_type == 'progress':
                            # Progress message
                            if progress_callback:
                                progress_msg = ProgressMessage(**data)
                                progress_callback(progress_msg.message)

                        elif msg_type == 'final':
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

            raise KernelCommandError(f"Command timed out after {timeout}s")

        finally:
            # Close comm if it was opened
            if comm_id:
                try:
                    close_msg = self.kernel_client.session.msg('comm_close', {
                        'comm_id': comm_id,
                    })
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
        request = CheckpointSaveRequest(name=name)
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return CheckpointSaveResponse(**response_dict)

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
        request = CheckpointRestoreRequest(name=name)
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return CheckpointRestoreResponse(**response_dict)

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
        request = CheckpointDeleteRequest(name=name)
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return CheckpointDeleteResponse(**response_dict)

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
        request = CheckpointListRequest()
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return CheckpointListResponse(**response_dict)

    def checkpoint_compare(
        self,
        name1: str,
        name2: str,
        timeout: Optional[float] = None,
    ) -> CheckpointCompareResponse:
        """
        Compare two checkpoints.

        Args:
            name1: First checkpoint name
            name2: Second checkpoint name
            timeout: Optional timeout override

        Returns:
            Response with diff result

        Raises:
            KernelCommandError: If command fails or checkpoints don't exist
        """
        request = CheckpointCompareRequest(name1=name1, name2=name2)
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return CheckpointCompareResponse(**response_dict)

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
        request = CheckpointClearRequest()
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return CheckpointClearResponse(**response_dict)

    # ========================================================================
    # Test Code Command
    # ========================================================================

    def test_code(
        self,
        original_code: str,
        modified_code: str,
        output_variables: Optional[Set[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
    ) -> TestCodeResponse:
        """
        Test original vs modified code.

        Args:
            original_code: Original code to execute
            modified_code: Modified code to execute
            output_variables: Variables to compare (None = all)
            progress_callback: Callback for progress messages
            timeout: Optional timeout override

        Returns:
            Response with test results

        Raises:
            KernelCommandError: If command fails
        """
        request = TestCodeRequest(
            original_code=original_code,
            modified_code=modified_code,
            output_variables=output_variables,
        )
        response_dict = self._send_command(
            request.model_dump(),
            progress_callback=progress_callback,
            timeout=timeout,
        )
        return TestCodeResponse(**response_dict)

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
        request = EnableScaleneRequest()
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return EnableScaleneResponse(**response_dict)

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
        request = DisableScaleneRequest()
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return DisableScaleneResponse(**response_dict)

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
        request = ForceCheckpointsRequest(enabled=enabled)
        response_dict = self._send_command(request.model_dump(), timeout=timeout)
        return ForceCheckpointsResponse(**response_dict)
