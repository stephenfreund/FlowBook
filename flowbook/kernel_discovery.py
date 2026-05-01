"""
Kernel discovery for MCP + JupyterLab shared kernel sessions.

Manages discovery files in the Jupyter runtime directory that enable
either MCP or JupyterLab to find and connect to a kernel started by
the other. Whoever starts the kernel writes the discovery file; the
second participant reads it and connects as a second ZMQ client.

Discovery files live at:
    {jupyter_runtime_dir}/flowbook-{hash}.json

where {hash} is the first 12 hex chars of SHA-256(abs_notebook_path).
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

from jupyter_core.paths import jupyter_runtime_dir


def _discovery_path(notebook_path: str) -> str:
    """Compute the discovery file path for a notebook.

    Args:
        notebook_path: Absolute path to the .ipynb file.

    Returns:
        Path to the discovery JSON in the Jupyter runtime directory.
    """
    h = hashlib.sha256(notebook_path.encode()).hexdigest()[:12]
    return os.path.join(jupyter_runtime_dir(), f"flowbook-{h}.json")


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def write_discovery(
    notebook_path: str,
    connection_file: str,
    kernel_name: str,
    pid: int,
    started_by: str,
) -> str:
    """Write a kernel discovery file.

    Args:
        notebook_path: Absolute path to the notebook.
        connection_file: Path to the ZMQ kernel connection file.
        kernel_name: Kernel spec name (e.g., "flowbook_kernel").
        pid: PID of the kernel process.
        started_by: Who started the kernel ("mcp" or "jupyterlab").

    Returns:
        Path to the discovery file that was written.
    """
    path = _discovery_path(notebook_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    doc = {
        "notebook_path": notebook_path,
        "connection_file": connection_file,
        "kernel_name": kernel_name,
        "pid": pid,
        "started_by": started_by,
        "started_at": time.time(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    return path


def read_discovery(notebook_path: str) -> Optional[Dict[str, Any]]:
    """Read and validate a kernel discovery file.

    Returns the discovery dict if the file exists, the kernel PID is alive,
    and the connection file exists. Otherwise returns None and cleans up
    any stale discovery file.

    Args:
        notebook_path: Absolute path to the notebook.

    Returns:
        Discovery dict with connection_file, pid, etc., or None.
    """
    path = _discovery_path(notebook_path)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, IOError):
        _remove_file(path)
        return None

    # Validate: PID must be alive
    pid = doc.get("pid")
    if not pid or not _is_pid_alive(pid):
        _remove_file(path)
        return None

    # Validate: connection file must exist
    conn_file = doc.get("connection_file")
    if not conn_file or not os.path.exists(conn_file):
        _remove_file(path)
        return None

    return doc


def remove_discovery(notebook_path: str) -> None:
    """Remove the discovery file for a notebook."""
    path = _discovery_path(notebook_path)
    _remove_file(path)


def _remove_file(path: str) -> None:
    """Remove a file, ignoring errors."""
    try:
        os.remove(path)
    except OSError:
        pass
