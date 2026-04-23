"""
Jupyter server API handlers for flowbook commands.
"""

import json
import pprint
import asyncio
import traceback
import tornado
import concurrent.futures
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

from flowbook.server.registry import CommandRegistry
from flowbook.server.kernel_manager import (
    FlowbookKernelClient,
    KernelConnectionManager,
)
from flowbook.kernel_discovery import read_discovery, write_discovery
from flowbook.util.output import error, log


# Global kernel manager instance
_kernel_manager = None


class FlowbookCommandHandler(APIHandler):
    """Handler for flowbook command execution."""

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
            selected_cell_ids = data.get("selected_cell_ids", None)
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

            # Normalize notebook (add cell IDs if missing, ensure uniqueness)
            from flowbook.util.cell_ids import normalize_notebook
            notebook_content = normalize_notebook(notebook_content)

            command = self.registry.get_command(command_name)

            kernel_client = None
            if command.requires_kernel:
                if not kernel_id:
                    self.set_status(400)
                    self.finish(json.dumps({"error": "Command requires kernel_id"}))
                    return

                try:
                    kernel_manager = self.kernel_manager.get_kernel(kernel_id)
                    kernel_client = FlowbookKernelClient(kernel_id=kernel_id)
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

            # Run the command in an executor to keep the event loop responsive.
            def run_command():
                log(f"Executing command {command_name}")
                log(f"Selected cell IDs: {selected_cell_ids}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(command.process(
                        notebook_content,
                        kernel_client=kernel_client,
                        selected_cell_ids=selected_cell_ids,
                        **params
                    ))
                    return result
                finally:
                    loop.close()

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            result = await asyncio.get_event_loop().run_in_executor(executor, run_command)
            executor.shutdown(wait=True)

            # Serialize ProcessingResult to JSON
            # Use model_dump() to convert Pydantic model to dict, then json.dumps
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
            self.finish(json.dumps(result_dict))

        except ValueError as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.set_status(400)
            self.finish(json.dumps({"error": f"{str(e)}\n\n{tb_str}"}))
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            error(f"Internal error: {str(e)}")
            traceback.print_exc()
            self.set_status(500)
            self.finish(json.dumps({"error": f"Internal error: {str(e)}\n\n{tb_str}"}))


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


class KernelConnectionFileHandler(APIHandler):
    """Handler for getting kernel connection file path."""

    @tornado.web.authenticated
    async def get(self, kernel_id):
        """Get the connection file path for a kernel."""
        try:
            kernel_manager = self.kernel_manager.get_kernel(kernel_id)
            connection_file = kernel_manager.connection_file

            self.finish(json.dumps({
                "connection_file": connection_file
            }))
        except Exception as e:
            self.set_status(404)
            self.finish(json.dumps({"error": f"Kernel not found: {str(e)}"}))


class KernelDiscoveryHandler(APIHandler):
    """Handler for kernel discovery — check if MCP has a kernel running for a notebook."""

    def _resolve_notebook_path(self, path: str) -> str:
        """Resolve a notebook path to an absolute canonical path.

        Handles tilde expansion in both the path and the server root directory,
        then resolves relative paths against the server root.

        Raises:
            ValueError: If the resolved path escapes the server root directory.
        """
        import os

        path = os.path.expanduser(path)
        root = self.settings.get("server_root_dir", "")
        if root:
            root = os.path.expanduser(root)
        if not os.path.isabs(path):
            if root:
                path = os.path.join(root, path)
        resolved = os.path.abspath(path)
        if root:
            abs_root = os.path.abspath(root)
            if resolved != abs_root and not resolved.startswith(abs_root + os.sep):
                raise ValueError(f"Path escapes server root directory")
        return resolved

    @tornado.web.authenticated
    async def get(self, path):
        """Check for an existing kernel discovery file.

        Args:
            path: Notebook path (relative to server root or absolute).
        """
        abs_path = self._resolve_notebook_path(path)
        disc = read_discovery(abs_path)

        if disc:
            self.finish(json.dumps(disc))
        else:
            self.set_status(404)
            self.finish(json.dumps({"error": "No kernel found for this notebook"}))

    def _get_kernel_pid(self, connection_file: str):
        """Look up the kernel process PID and absolute connection file path.

        Extracts the kernel UUID from the connection file name (kernel-{UUID}.json)
        and queries the Jupyter kernel manager for the actual process PID and
        the full path to the connection file.

        Returns:
            (pid, connection_file) tuple. Falls back to (0, original) on failure.
        """
        import os
        import re

        # Extract kernel UUID from connection file name
        match = re.match(r"kernel-([0-9a-f-]+)\.json", os.path.basename(connection_file))
        if not match:
            return 0, connection_file

        kernel_id = match.group(1)
        try:
            km = self.settings["serverapp"].kernel_manager
            kernel = km.get_kernel(kernel_id)
            pid = getattr(kernel.provisioner, "pid", 0) or 0
            # Use the kernel manager's full connection file path
            abs_conn = getattr(kernel, "connection_file", connection_file)
            return pid, abs_conn
        except Exception:
            return 0, connection_file

    @tornado.web.authenticated
    async def put(self, path):
        """Write a kernel discovery file (called by JupyterLab when it starts a kernel).

        Request body: {"connection_file": "...", "kernel_name": "...", "pid": 123}
        """
        data = self.get_json_body()
        abs_path = self._resolve_notebook_path(path)
        connection_file = data.get("connection_file", "")

        # Look up actual kernel PID and full connection file path
        # (frontend sends pid=0 and a bare filename)
        pid, connection_file = self._get_kernel_pid(connection_file)

        disc_path = write_discovery(
            notebook_path=abs_path,
            connection_file=connection_file,
            kernel_name=data.get("kernel_name", "flowbook_kernel"),
            pid=pid,
            started_by="jupyterlab",
        )
        self.finish(json.dumps({"discovery_file": disc_path}))


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
            url_path_join(base_url, "flowbook", "execute"),
            FlowbookCommandHandler,
            {"registry": registry},
        ),
        (
            url_path_join(base_url, "flowbook", "list"),
            CommandListHandler,
            {"registry": registry},
        ),
        (
            url_path_join(base_url, "flowbook", "kernel", "(.+)", "connection"),
            KernelConnectionFileHandler,
            {},
        ),
        (
            url_path_join(base_url, "flowbook", "kernel-discovery", "(.+)"),
            KernelDiscoveryHandler,
            {},
        ),
    ]

    web_app.add_handlers(host_pattern, handlers)


# def _jupyter_server_extension_points():
#     """Entry point for Jupyter Server extension."""
#     return [{"module": "flowbook.server"}]


# def _load_jupyter_server_extension(server_app):
#     """Load the Jupyter Server extension."""
#     setup_handlers(server_app.web_app)
#     server_app.log.info("FlowBook Server Extension loaded!")
