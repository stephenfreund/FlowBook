"""
Kernel connection manager for the Jupyter server extension.
"""

import threading
from typing import Any, Dict, Optional
from jupyter_client.blocking.client import BlockingKernelClient
from jupyter_client.manager import KernelManager
from jupyter_server.serverapp import ServerApp


class FlowbookKernelClient(BlockingKernelClient):
    """A kernel client for the FlowBook server extension."""

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
    """Manages kernel connections for the server extension.

    Clients are cached per kernel_id so repeated requests reuse one set of
    ZMQ channels instead of leaking a new client per request. Stale entries
    (kernel gone, channels dead) are cleaned up on the next access for that
    kernel_id.

    Because a cached client is shared across requests, callers that talk to
    the kernel must hold the per-kernel lock from ``lock_for()`` for the
    duration of their kernel conversation — ZMQ sockets are not thread-safe,
    and the command handler runs commands in executor threads.
    """

    def __init__(self, jupyter_server_app: ServerApp):
        self.server_app = jupyter_server_app
        self._kernel_clients: Dict[str, FlowbookKernelClient] = {}
        self._locks: Dict[str, threading.Lock] = {}

    def lock_for(self, kernel_id: str) -> threading.Lock:
        """Get the lock serializing kernel conversations for this kernel.

        Must be called from the event loop thread (like get_kernel_client);
        the returned lock is then acquired from worker threads.
        """
        if kernel_id not in self._locks:
            self._locks[kernel_id] = threading.Lock()
        return self._locks[kernel_id]

    def get_kernel_client(self, kernel_id: str) -> FlowbookKernelClient:
        """Get or create a kernel client for the given kernel ID."""
        if kernel_id in self._kernel_clients:
            client = self._kernel_clients[kernel_id]
            try:
                # Raises if the kernel no longer exists on the server.
                self.server_app.kernel_manager.get_kernel(kernel_id)
            except Exception:
                self.cleanup_client(kernel_id)
                raise
            if client.channels_running:
                return client
            # Channels died (e.g. kernel restarted with new ports) — rebuild.
            self.cleanup_client(kernel_id)

        kernel_manager: KernelManager = self.server_app.kernel_manager.get_kernel(
            kernel_id
        )

        # Create our custom FlowbookKernelClient and configure it with connection info
        client = FlowbookKernelClient(kernel_id=kernel_id)
        client.load_connection_info(kernel_manager.get_connection_info())
        client.start_channels()
        try:
            client.wait_for_ready(timeout=30)
        except Exception:
            # Don't leak channels for a client that never became usable.
            client.stop_channels()
            raise

        self._kernel_clients[kernel_id] = client
        return client

    def cleanup_client(self, kernel_id: str):
        """Clean up a kernel client."""
        if kernel_id in self._kernel_clients:
            client = self._kernel_clients[kernel_id]
            try:
                client.stop_channels()
            except Exception:
                pass
            del self._kernel_clients[kernel_id]
