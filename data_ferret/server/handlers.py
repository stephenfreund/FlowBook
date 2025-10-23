"""
Jupyter server API handlers for ferret commands.
"""

import json
import pprint
import tornado
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import (
    FerretKernelClient,
    KernelConnectionManager,
)


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
    ]

    web_app.add_handlers(host_pattern, handlers)


# def _jupyter_server_extension_points():
#     """Entry point for Jupyter Server extension."""
#     return [{"module": "data_ferret.server"}]


# def _load_jupyter_server_extension(server_app):
#     """Load the Jupyter Server extension."""
#     setup_handlers(server_app.web_app)
#     server_app.log.info("Ferret Server Extension loaded!")
