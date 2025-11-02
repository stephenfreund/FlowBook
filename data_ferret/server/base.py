"""
Abstract base class for notebook processing commands.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.server.config import FerretConfig
from data_ferret.util.output import log
from jupyter_server.serverapp import ServerApp

# Type variables for generic request/response
TRequest = TypeVar('TRequest', bound=BaseModel)
TResponse = TypeVar('TResponse', bound=BaseModel)

class NotebookCommand(ABC):
    """Abstract base class for notebook processing commands."""

    @abstractmethod
    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[FerretConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process the notebook content and return a new notebook with metadata.

        Args:
            notebook_content: The parsed JSON content of a Jupyter notebook
            kernel_client: Optional kernel client for executing code
            selected_cell_ids: Optional list of selected cell IDs
            config: Optional configuration for the command (uses defaults if not provided)
            **kwargs: Additional parameters specific to the command

        Returns:
            Dictionary containing:
                - notebook: The new/modified notebook
                - metadata: JSON metadata object with processing results
        """
        pass

    @property
    @abstractmethod
    def command_name(self) -> str:
        """Return the name of this command."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the display name for UI."""
        pass

    @property
    @abstractmethod
    def icon_name(self) -> str:
        """Return the icon name (Jupyter Lab icon or emoji)."""
        pass

    @property
    def tooltip(self) -> str:
        """Return tooltip text for the command button."""
        return self.display_name

    @property
    def requires_kernel(self) -> bool:
        """Return whether this command requires a kernel connection."""
        return False

    @staticmethod
    def config_from_serverapp(serverapp: ServerApp) -> FerretConfig:
        """Return the configuration from the serverapp."""
        return FerretConfig(
            model=serverapp.web_app.settings["data_ferret"].model,
            fast_model=serverapp.web_app.settings["data_ferret"].fast_model,
        )

    def _send_comm_message(
        self,
        kernel_client: FerretKernelClient,
        target_name: str,
        request: TRequest,
        response_type: Type[TResponse],
        *,
        timeout: int = 60
    ) -> TResponse:
        """
        Send a comm message to the kernel with type-safe request/response.

        This method always logs progress messages if the kernel sends them.
        The kernel should send messages in this format:

        Progress messages (optional, multiple):
            {"type": "progress", "message": "Step description"}

        Final message (required, once):
            {"type": "final", "ok": True, "result": <data>}
            {"type": "final", "ok": False, "error": <error_string>}

        Legacy format (backward compatible):
            {"ok": True, "result": <data>}
            {"ok": False, "error": <error_string>}

        Args:
            kernel_client: The kernel client
            target_name: Comm target name (e.g., "test_code", "debug_command")
            request: Pydantic model instance with request data
            response_type: Pydantic model class for response validation
            timeout: Timeout in seconds (default: 60)

        Returns:
            Instance of response_type with validated data

        Raises:
            Exception if message fails or times out
        """
        comm_id = uuid.uuid4().hex

        # Convert request model to dict
        data = request.model_dump()

        # Build and send the comm_open message
        content = {
            "comm_id": comm_id,
            "target_name": target_name,
            "target_module": "",
            "data": data,
        }
        open_msg = kernel_client.session.msg("comm_open", content)
        kernel_client.shell_channel.send(open_msg)

        # Wait for messages on iopub channel, logging progress until final message
        while True:
            try:
                # Use iopub_channel directly for better compatibility
                reply = kernel_client.iopub_channel.get_msg(timeout=timeout)
                msg_type = reply["header"]["msg_type"]

                # Only process comm_msg messages with our comm_id
                if msg_type == "comm_msg" and reply["content"].get("comm_id") == comm_id:
                    reply_data = reply["content"]["data"]

                    # Check if this is a progress message or final message
                    data_type = reply_data.get("type")

                    if data_type == "progress":
                        # Log progress messages
                        message = reply_data.get("message", "")
                        self._log_progress_message(message)
                    elif data_type == "final":
                        # Final message - validate and return
                        return response_type.model_validate(reply_data)
                    else:
                        # Legacy format (no type field) - validate and return
                        return response_type.model_validate(reply_data)
                # Skip other message types (status, display_data, etc.)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # If we get a timeout or other error, re-raise it
                raise

    def _log_progress_message(self, message: str):
        """
        Log a progress message from the kernel.

        Can be overridden by subclasses for custom progress handling.

        Args:
            message: Progress message to log
        """
        log(message)
