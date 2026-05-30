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
from flowbook.server.fix_dispatcher import apply_fix
from flowbook.server.fix_models import (
    TOOL_ARG_SCHEMAS,
    ApplyFixResponse,
    PlanValidationError,
)
from flowbook.server.fix_suggester import (
    CustomDoneEvent,
    ErrorEvent,
    FixSuggester,
    PlanEvent,
    TextEvent,
    build_context_from_notebook,
    feature_enabled,
    get_model,
)
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


class SuggestFixHandler(APIHandler):
    """POST /flowbook/suggest-fix — stream a diagnosis + FixPlan via SSE.

    Request body: {"notebook": <full notebook json>, "cell_id": "abcd"}

    Stream protocol (one event per server-sent-event frame):
        event: diagnosis
        data: {"text": "..."}     # one chunk of the diagnosis text

        event: plan
        data: {"fixes": [...]}    # the validated FixPlan, sent once

        event: error
        data: {"message": "..."}  # terminal failure; no plan available

        event: done
        data: {}                  # terminal success marker

    The frontend can abort by closing the connection (AbortController on
    fetch). Tornado surfaces that as RequestFinishedError, which we let
    propagate to stop the upstream LLM stream.
    """

    @tornado.web.authenticated
    async def post(self):
        # Quick guard: feature disabled if no provider key is configured.
        if not feature_enabled(self.settings):
            log(
                "AI fix suggestion requested but disabled: no provider API key "
                f"found for fix_model '{get_model(self.settings)}'. Set "
                "ANTHROPIC_API_KEY (or the env var for your configured "
                "fix_model provider) in the Jupyter server's environment."
            )
            self.set_status(503)
            self.set_header("Content-Type", "application/json")
            self.finish(json.dumps({
                "feature_disabled": True,
                "reason": (
                    "No provider API key found. Set ANTHROPIC_API_KEY (or the "
                    "env var for your configured fix_model provider) in the "
                    "Jupyter server's environment."
                ),
            }))
            return

        try:
            body = self.get_json_body() or {}
            notebook = body.get("notebook")
            cell_id = body.get("cell_id")
            if not notebook or not isinstance(notebook, dict):
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing or invalid 'notebook'"}))
                return
            if not cell_id or not isinstance(cell_id, str):
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing or invalid 'cell_id'"}))
                return
        except Exception as e:
            self.set_status(400)
            self.finish(json.dumps({"error": f"Bad request: {e}"}))
            return

        context = build_context_from_notebook(notebook, cell_id)
        if context is None:
            self.set_status(404)
            self.finish(json.dumps({
                "error": f"Cell {cell_id} has no violation metadata or is not a code cell"
            }))
            return

        # Open the SSE stream.
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")  # nginx: disable buffering

        suggester = FixSuggester(model=get_model(self.settings))

        try:
            # Pass the full notebook so the agentic loop's read-only tools
            # can inspect outputs, distant cells, tracebacks, etc.
            async for event in suggester.stream(context, notebook=notebook):
                if isinstance(event, TextEvent):
                    self._send_event("diagnosis", {"text": event.text})
                elif isinstance(event, PlanEvent):
                    self._send_event("plan", event.plan.model_dump())
                elif isinstance(event, ErrorEvent):
                    self._send_event("error", {"message": event.message})
            self._send_event("done", {})
        except tornado.iostream.StreamClosedError:
            # Client aborted. Nothing to do — the async generator will be
            # garbage-collected, which cancels the underlying LLM request.
            return
        except Exception as e:
            error(f"suggest-fix stream failed: {e}")
            traceback.print_exc()
            try:
                self._send_event("error", {"message": f"Internal error: {e}"})
            except tornado.iostream.StreamClosedError:
                pass
        finally:
            self.finish()

    def _send_event(self, event_type: str, data: dict) -> None:
        """Write one SSE frame and flush it to the client."""
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        self.write(payload)
        self.flush()


class ApplyFixHandler(APIHandler):
    """POST /flowbook/apply-fix — apply a chosen FixSuggestion to a notebook.

    Request body: {"notebook": <nb json>, "tool": "alpha_rename", "args": {...}}

    Always validates `tool` against the allowlist and `args` against the
    schema before dispatching. The notebook in the request body is mutated
    in place by the dispatcher and the full result (modified sources,
    pre-fix snapshot for undo, new cell order if changed) is returned.

    The frontend is responsible for re-running the affected cells.
    """

    @tornado.web.authenticated
    async def post(self):
        try:
            body = self.get_json_body() or {}
            notebook = body.get("notebook")
            tool = body.get("tool")
            args = body.get("args") or {}
        except Exception as e:
            self.set_status(400)
            self.finish(json.dumps({"error": f"Bad request: {e}"}))
            return

        if not notebook or not isinstance(notebook, dict):
            self.set_status(400)
            self.finish(json.dumps({"error": "Missing or invalid 'notebook'"}))
            return
        if tool not in TOOL_ARG_SCHEMAS:
            self.set_status(400)
            self.finish(json.dumps({"error": f"Unknown or unsupported tool: {tool}"}))
            return
        expected = TOOL_ARG_SCHEMAS[tool]
        actual = set(args.keys())
        if expected != actual:
            self.set_status(400)
            self.finish(json.dumps({
                "error": f"Tool '{tool}' args mismatch: expected {sorted(expected)}, got {sorted(actual)}"
            }))
            return

        try:
            response: ApplyFixResponse = apply_fix(notebook, tool, args)
        except ValueError as e:
            self.set_status(400)
            self.finish(json.dumps({"error": str(e)}))
            return
        except Exception as e:
            error(f"apply-fix dispatcher failed: {e}")
            traceback.print_exc()
            self.set_status(500)
            self.finish(json.dumps({"error": f"Internal error: {e}"}))
            return

        # Return the result alongside the (now-mutated) notebook so the
        # frontend can use either.
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({
            "result": response.model_dump(),
            "notebook": notebook,
        }))


class CustomFixHandler(APIHandler):
    """POST /flowbook/custom-fix — run an LLM-driven custom fix per user instruction.

    Request body: {"notebook": <full notebook json>, "cell_id": "abcd",
                   "instruction": "<natural language>"}

    Streams SSE frames during the agentic loop (text + done/error), and on
    success the trailing 'done' event carries the CustomFixResponse payload
    (modified_cells, cells_added/removed, pre/post sources, new cell order,
    free-text summary). The frontend uses pre_fix_sources for Undo exactly
    like the built-in apply path.
    """

    @tornado.web.authenticated
    async def post(self):
        if not feature_enabled(self.settings):
            log(
                "Custom AI fix requested but disabled: no provider API key "
                f"found for fix_model '{get_model(self.settings)}'. Set "
                "ANTHROPIC_API_KEY (or the env var for your configured "
                "fix_model provider) in the Jupyter server's environment."
            )
            self.set_status(503)
            self.set_header("Content-Type", "application/json")
            self.finish(json.dumps({
                "feature_disabled": True,
                "reason": "No provider API key found for the configured fix_model.",
            }))
            return

        try:
            body = self.get_json_body() or {}
            notebook = body.get("notebook")
            cell_id = body.get("cell_id")
            instruction = body.get("instruction")
            if not notebook or not isinstance(notebook, dict):
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing or invalid 'notebook'"}))
                return
            if not cell_id or not isinstance(cell_id, str):
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing or invalid 'cell_id'"}))
                return
            if not instruction or not isinstance(instruction, str) or not instruction.strip():
                self.set_status(400)
                self.finish(json.dumps({"error": "Missing or empty 'instruction'"}))
                return
        except Exception as e:
            self.set_status(400)
            self.finish(json.dumps({"error": f"Bad request: {e}"}))
            return

        # Locate the cell + compute its @-label for the prompt.
        code_cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
        cell_ids = [c.get("id", "") for c in code_cells]
        if cell_id not in cell_ids:
            self.set_status(404)
            self.finish(json.dumps({"error": f"Cell '{cell_id}' not found"}))
            return
        cell_alpha = _index_to_alpha(cell_ids.index(cell_id))

        # SSE headers — same shape as suggest-fix.
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")

        suggester = FixSuggester(model=get_model(self.settings))

        try:
            async for event in suggester.custom_stream(
                notebook=notebook,
                cell_id=cell_id,
                cell_alpha=cell_alpha,
                instruction=instruction.strip(),
            ):
                if isinstance(event, TextEvent):
                    self._send_event("diagnosis", {"text": event.text})
                elif isinstance(event, ErrorEvent):
                    self._send_event("error", {"message": event.message})
                elif isinstance(event, CustomDoneEvent):
                    payload = _build_custom_fix_response(
                        notebook=notebook,
                        instruction=instruction.strip(),
                        summary=event.summary,
                        log=event.log,
                    )
                    self._send_event("done", payload)
        except tornado.iostream.StreamClosedError:
            return
        except Exception as e:
            error(f"custom-fix stream failed: {e}")
            traceback.print_exc()
            try:
                self._send_event("error", {"message": f"Internal error: {e}"})
            except tornado.iostream.StreamClosedError:
                pass
        finally:
            self.finish()

    def _send_event(self, event_type: str, data: dict) -> None:
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        self.write(payload)
        self.flush()


class FixStatusHandler(APIHandler):
    """GET /flowbook/fix-status — report AI-fix availability + configured model.

    Response body: {"enabled": bool, "model": str}

    The frontend uses this to render the fix-it status line in the FlowBook
    panel without making (and failing) an actual suggestion request. ``model``
    is always the resolved fix_model identifier; ``enabled`` reflects whether
    that provider's API key is present in the server environment.
    """

    @tornado.web.authenticated
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({
            "enabled": feature_enabled(self.settings),
            "model": get_model(self.settings),
        }))


def _index_to_alpha(idx: int) -> str:
    """0-based code-cell index → @-label. Mirrors src/cellindexutils.ts indexToAlpha."""
    result = ""
    n = idx
    while True:
        result = chr(ord("A") + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            return result


def _build_custom_fix_response(
    notebook: dict, instruction: str, summary: str, log
) -> dict:
    """Assemble the JSON the frontend needs to drive UI + Undo."""
    # Aggregate diffs from log entries.
    modified: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    for entry in log.entries:
        for cid in entry.modified_cells:
            if cid not in modified:
                modified.append(cid)
        for cid in entry.cells_added:
            if cid not in added:
                added.append(cid)
        for cid in entry.cells_removed:
            if cid not in removed:
                removed.append(cid)

    # post_fix_sources: current source for every modified + added cell still in
    # the notebook (removed ones obviously aren't).
    current_cells = {c.get("id"): c for c in notebook.get("cells", [])}
    post: dict = {}
    from flowbook.scripts.fix_repro_errors import get_cell_source as _gcs

    for cid in list(modified) + list(added):
        if cid in current_cells:
            post[cid] = _gcs(current_cells[cid])

    new_order = [
        c.get("id")
        for c in notebook.get("cells", [])
        if c.get("cell_type") == "code"
    ]

    return {
        "ok": True,
        "instruction": instruction,
        "summary": summary,
        "modified_cells": modified,
        "cells_added": added,
        "cells_removed": removed,
        "pre_fix_sources": dict(log.pre_fix_sources),
        "post_fix_sources": post,
        "new_cell_order": new_order,
        "mutations": [
            {
                "tool": entry.tool,
                "args": entry.args,
                "summary": entry.summary,
                "modified_cells": entry.modified_cells,
                "cells_added": entry.cells_added,
                "cells_removed": entry.cells_removed,
            }
            for entry in log.entries
        ],
    }


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
        (
            url_path_join(base_url, "flowbook", "suggest-fix"),
            SuggestFixHandler,
            {},
        ),
        (
            url_path_join(base_url, "flowbook", "apply-fix"),
            ApplyFixHandler,
            {},
        ),
        (
            url_path_join(base_url, "flowbook", "custom-fix"),
            CustomFixHandler,
            {},
        ),
        (
            url_path_join(base_url, "flowbook", "fix-status"),
            FixStatusHandler,
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
