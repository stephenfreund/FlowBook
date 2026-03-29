"""
Y.js document sync for MCP ↔ JupyterLab collaboration.

Connects the MCP server to a jupyter-collaboration Y.js room so that
cell edits and outputs propagate to JupyterLab in real time.

Requires:
    - jupyter-collaboration extension running on Jupyter Server
    - pycrdt (Y.js CRDT bindings for Python)
    - websockets (WebSocket client)

Usage:
    sync = YDocSync(server_url="http://localhost:8888", token="...")
    await sync.connect("path/to/notebook.ipynb")

    # Edit a cell (appears in JupyterLab instantly)
    sync.set_cell_source(0, "x = 42")

    # Update cell outputs after execution
    sync.set_cell_outputs(0, [{"output_type": "stream", ...}])

    # Read current state (includes JupyterLab edits)
    cell = sync.get_cell(0)

    await sync.disconnect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any, Callable, Dict, List, Optional

import websockets
from jupyter_ydoc import YNotebook
from pycrdt import Doc, Provider

logger = logging.getLogger(__name__)


def _get_collaboration_session(
    server_url: str, token: Optional[str], notebook_path: str
) -> Dict[str, str]:
    """Create/join a collaboration session via Jupyter Server REST API.

    Args:
        server_url: Jupyter Server base URL.
        token: Auth token.
        notebook_path: Relative path to notebook (as Jupyter Server sees it).

    Returns:
        Dict with "sessionId" and "fileId".
    """
    url = f"{server_url}/api/collaboration/session/{notebook_path}"
    body = json.dumps({"format": "json", "type": "notebook"}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


class _WebSocketChannel:
    """Adapter from websockets.WebSocketClientProtocol to pycrdt.Channel protocol."""

    def __init__(self, ws: Any, room_id: str):
        self._ws = ws
        self._room_id = room_id

    @property
    def path(self) -> str:
        return self._room_id

    async def send(self, message: bytes) -> None:
        await self._ws.send(message)

    async def recv(self) -> bytes:
        return await self._ws.recv()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.recv()
        except websockets.exceptions.ConnectionClosed:
            raise StopAsyncIteration


class YDocSync:
    """Manages a Y.js connection to a jupyter-collaboration room."""

    def __init__(
        self,
        server_url: str,
        token: Optional[str] = None,
    ):
        """
        Args:
            server_url: Jupyter Server base URL (e.g., "http://localhost:8888").
            token: Jupyter Server authentication token.
        """
        self._server_url = server_url.rstrip("/")
        self._token = token
        self._ws: Any = None
        self._provider: Optional[Provider] = None
        self._ydoc: Optional[YNotebook] = None
        self._provider_task: Optional[asyncio.Task] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def ydoc(self) -> Optional[YNotebook]:
        return self._ydoc

    async def connect(self, notebook_path: str) -> None:
        """Connect to the Y.js room for a notebook.

        Args:
            notebook_path: Relative path to the notebook (as Jupyter Server sees it).
        """
        if self._connected:
            await self.disconnect()

        # Get collaboration session (creates room if needed)
        session_info = _get_collaboration_session(
            self._server_url, self._token, notebook_path
        )
        session_id = session_info["sessionId"]
        file_id = session_info["fileId"]

        # Build room ID using fileId (not path)
        room_id = f"json:notebook:{file_id}"

        # Build WebSocket URL
        ws_scheme = "wss" if self._server_url.startswith("https") else "ws"
        ws_base = self._server_url.replace("http://", "").replace("https://", "")
        ws_url = f"{ws_scheme}://{ws_base}/api/collaboration/room/{room_id}"

        # Add sessionId and token as query params
        params = f"sessionId={session_id}"
        if self._token:
            params += f"&token={self._token}"
        ws_url += f"?{params}"

        logger.info(f"YDocSync: connecting to room {room_id}")

        # Create Y.js document
        self._ydoc = YNotebook()

        # Connect WebSocket
        self._ws = await websockets.connect(ws_url, max_size=None)

        # Create channel adapter and provider
        channel = _WebSocketChannel(self._ws, room_id)
        self._provider = Provider(self._ydoc.ydoc, channel)

        # Start provider in background task
        self._provider_task = asyncio.create_task(self._run_provider())
        self._connected = True

        # Wait briefly for initial sync
        await asyncio.sleep(0.5)

        logger.info(
            f"YDocSync: connected, {self._ydoc.cell_number} cells in document"
        )

    async def _run_provider(self) -> None:
        """Run the Y.js provider (processes sync messages)."""
        try:
            await self._provider.start()
        except websockets.exceptions.ConnectionClosed:
            logger.info("YDocSync: WebSocket connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"YDocSync: provider error: {e}")
        finally:
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from the Y.js room."""
        if self._provider_task and not self._provider_task.done():
            self._provider_task.cancel()
            try:
                await self._provider_task
            except asyncio.CancelledError:
                pass
            self._provider_task = None

        self._provider = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._ydoc = None
        self._connected = False
        logger.info("YDocSync: disconnected")

    # ------------------------------------------------------------------
    # Cell operations (propagate to JupyterLab via Y.js)
    # ------------------------------------------------------------------

    def get_cell(self, index: int) -> Dict[str, Any]:
        """Get a cell by index from the Y.js document."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        return self._ydoc.get_cell(index)

    def get_cell_count(self) -> int:
        """Get the number of cells in the Y.js document."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        return self._ydoc.cell_number

    def set_cell_source(self, index: int, source: str) -> None:
        """Update a cell's source code (propagates to JupyterLab)."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        cell = self._ydoc.get_cell(index)
        cell["source"] = source
        self._ydoc.set_cell(index, cell)

    def set_cell_outputs(
        self, index: int, outputs: List[Dict[str, Any]]
    ) -> None:
        """Update a cell's outputs (propagates to JupyterLab)."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        cell = self._ydoc.get_cell(index)
        cell["outputs"] = outputs
        self._ydoc.set_cell(index, cell)

    def set_cell_execution_count(
        self, index: int, execution_count: Optional[int]
    ) -> None:
        """Update a cell's execution count (propagates to JupyterLab)."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        cell = self._ydoc.get_cell(index)
        cell["execution_count"] = execution_count
        self._ydoc.set_cell(index, cell)

    def set_cell_metadata(
        self, index: int, key: str, value: Any
    ) -> None:
        """Update a cell metadata key (propagates to JupyterLab)."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        cell = self._ydoc.get_cell(index)
        metadata = cell.get("metadata", {})
        metadata[key] = value
        cell["metadata"] = metadata
        self._ydoc.set_cell(index, cell)

    def get_notebook(self) -> Dict[str, Any]:
        """Get the full notebook from the Y.js document."""
        if not self._ydoc:
            raise RuntimeError("Not connected to Y.js room")
        return self._ydoc.get()

    def find_cell_index(self, cell_id: str) -> Optional[int]:
        """Find a cell's index by ID in the Y.js document."""
        if not self._ydoc:
            return None
        for i in range(self._ydoc.cell_number):
            cell = self._ydoc.get_cell(i)
            if cell.get("id") == cell_id:
                return i
        return None
