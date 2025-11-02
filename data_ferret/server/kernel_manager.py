"""
Kernel connection manager for the Jupyter server extension.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from jupyter_client.blocking.client import BlockingKernelClient
from jupyter_client.manager import KernelManager
from jupyter_server.serverapp import ServerApp


@dataclass
class TestCodeData:
    """Data class for test_code comm results."""
    ok: bool
    result: str


class FerretKernelClient(BlockingKernelClient):
    """A kernel client for the Ferret server extension."""

    def __init__(self, kernel_id: str):
        super().__init__()
        self.kernel_id = kernel_id

    def execute(
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, Any]] = None,
        allow_stdin: Optional[bool] = None,
        stop_on_error: bool = True,
        *,
        cell_id: Optional[str] = None,
        cell_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Override execute to optionally include cell_id and cell_metadata in the message to the kernel."""
        if user_expressions is None:
            user_expressions = {}
        content = dict(
            code=code,
            silent=silent,
            store_history=store_history,
            user_expressions=user_expressions,
            allow_stdin=allow_stdin if allow_stdin is not None else self.allow_stdin,
            stop_on_error=stop_on_error,
        )

        # Define the metadata, including cell_id and any other custom data
        metadata = {
            "cell_id": cell_id,
        }
        if cell_metadata is not None:
            metadata.update(cell_metadata)

        msg = self.session.msg("execute_request", content, metadata=metadata)
        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]


class KernelConnectionManager:
    """Manages kernel connections for the server extension."""

    def __init__(self, jupyter_server_app: ServerApp):
        self.server_app = jupyter_server_app
        self._kernel_clients: Dict[str, FerretKernelClient] = {}

    def get_kernel_client(self, kernel_id: str) -> FerretKernelClient:
        """Get or create a kernel client for the given kernel ID."""
        if kernel_id in self._kernel_clients:
            return self._kernel_clients[kernel_id]

        kernel_manager: KernelManager = self.server_app.kernel_manager.get_kernel(
            kernel_id
        )

        # Create our custom FerretKernelClient and configure it with connection info
        client = FerretKernelClient(kernel_id=kernel_id)
        client.load_connection_info(kernel_manager.get_connection_info())
        client.start_channels()
        client.wait_for_ready(timeout=30)

        self._kernel_clients[kernel_id] = client

        assert isinstance(client, FerretKernelClient)
        return client

    def cleanup_client(self, kernel_id: str):
        """Clean up a kernel client."""
        if kernel_id in self._kernel_clients:
            client = self._kernel_clients[kernel_id]
            client.stop_channels()
            del self._kernel_clients[kernel_id]
