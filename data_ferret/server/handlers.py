"""
Jupyter server API handlers for ferret commands.
"""

import json
import pprint
import asyncio
import uuid
import tornado
from jupyter_server.base.handlers import APIHandler, JupyterHandler
from jupyter_server.utils import url_path_join

from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import (
    FerretKernelClient,
    KernelConnectionManager,
)
from data_ferret.server.message_broadcaster import get_broadcaster, get_broadcast_stream
from data_ferret.util.output import stream_output


# Global kernel manager instance
_kernel_manager = None


class FerretCommandHandler(APIHandler):
    """Handler for ferret command execution."""

    def initialize(self, registry: CommandRegistry):
        """Initialize with command registry and kernel manager."""
        self.registry = registry

    @tornado.web.authenticated
    async def post(self):
        """Handle POST requests to execute commands."""
        try:
            data = self.get_json_body()
            command_name = data.get("command")
            notebook_content = data.get("notebook")
            kernel_id = data.get("kernel_id")
            params = data.get("params", {})

            if not command_name:
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing 'command' field"}))
                return

            if not notebook_content:
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing 'notebook' field"}))
                return

            command = self.registry.get_command(command_name)

            print(
                "MODEL NAME",
                self.serverapp.web_app.settings["data_ferret"].model_name,
            )

            kernel_client = None
            if command.requires_kernel:
                if not kernel_id:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Command requires kernel_id"}))
                    return

                try:
                    kernel_manager = self.kernel_manager.get_kernel(kernel_id)
                    kernel_client = FerretKernelClient(kernel_id=kernel_id)
                    kernel_client.load_connection_info(
                        kernel_manager.get_connection_info()
                    )
                    kernel_client.start_channels()
                    kernel_client.wait_for_ready(timeout=30)
                except Exception as e:
                    self.set_status(400)
                    self.finish(
                        json.dumps({"error": f"Failed to connect to kernel: {str(e)}"})
                    )
                    return

            # Execute command with output streaming to clients
            with stream_output(get_broadcast_stream()):
                result = command.process(
                    notebook_content, kernel_client=kernel_client, **params
                )

            self.finish(json.dumps(result))

        except ValueError as e:
            self.set_status(400)
            self.finish(json.dumps({"error": str(e)}))
        except Exception as e:
            self.set_status(500)
            self.finish(json.dumps({"error": f"Internal error: {str(e)}"}))


class CommandListHandler(APIHandler):
    """Handler for listing available commands."""

    def initialize(self, registry: CommandRegistry):
        """Initialize with command registry."""
        self.registry = registry

    @tornado.web.authenticated
    async def get(self):
        """List available commands with UI information."""
        command_info = self.registry.get_command_info()
        self.finish(json.dumps({"commands": command_info}))


class MessageStreamHandler(JupyterHandler):
    """Handler for Server-Sent Events (SSE) message streaming."""

    def initialize(self):
        """Initialize the handler."""
        self.broadcaster = get_broadcaster()
        self.client_id = str(uuid.uuid4())
        self.queue = None

    @tornado.web.authenticated
    async def get(self):
        """Stream messages to the client via SSE."""
        # Set headers for SSE
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        self.set_header('X-Accel-Buffering', 'no')

        # Register this client with the broadcaster
        self.queue = self.broadcaster.register_client(self.client_id)

        try:
            # Send initial connection message
            self.write(f'data: {{"type":"connected","client_id":"{self.client_id}"}}\n\n')
            await self.flush()

            # Stream messages from the queue
            while True:
                try:
                    # Wait for a message with timeout
                    message = await asyncio.wait_for(self.queue.get(), timeout=30.0)

                    print("MESSAGE", message)

                    # Send the message as SSE event
                    self.write(f'data: {message.to_json()}\n\n')
                    await self.flush()

                except asyncio.TimeoutError:
                    # Send keepalive comment every 30 seconds
                    self.write(': keepalive\n\n')
                    await self.flush()
                except Exception as e:
                    self.log.error(f"Error streaming message: {e}")
                    break

        except Exception as e:
            self.log.error(f"SSE connection error: {e}")
        finally:
            # Unregister client on disconnect
            self.broadcaster.unregister_client(self.client_id)

    def on_connection_close(self):
        """Handle client disconnection."""
        if self.client_id:
            self.broadcaster.unregister_client(self.client_id)


def setup_handlers(web_app):
    """Set up the extension handlers."""
    global _kernel_manager

    pprint.pprint(web_app.settings)

    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    registry = CommandRegistry()
    _kernel_manager = KernelConnectionManager(web_app.settings["serverapp"])

    handlers = [
        (
            url_path_join(base_url, "ferret", "execute"),
            FerretCommandHandler,
            {"registry": registry},
        ),
        (
            url_path_join(base_url, "ferret", "list"),
            CommandListHandler,
            {"registry": registry},
        ),
        (
            url_path_join(base_url, "ferret", "stream"),
            MessageStreamHandler,
            {},
        ),
    ]

    web_app.add_handlers(host_pattern, handlers)


# def _jupyter_server_extension_points():
#     """Entry point for Jupyter Server extension."""
#     return [{"module": "data_ferret.server"}]


# def _load_jupyter_server_extension(server_app):
#     """Load the Jupyter Server extension."""
#     setup_handlers(server_app.web_app)
#     server_app.log.info("Ferret Server Extension loaded!")
