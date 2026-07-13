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
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

from jupyter_core.paths import jupyter_runtime_dir

logger = logging.getLogger(__name__)


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
    """Check whether a process with the given PID is still running.

    A PermissionError from ``os.kill(pid, 0)`` means the process EXISTS but
    is owned by another user — that is ALIVE, not dead (audit C10). Only
    ProcessLookupError / other OSErrors mean the process is gone.

    Known limitation (PID reuse): a recycled PID can make a dead kernel look
    alive, so a stale discovery file may briefly survive. This is bounded in
    practice by the connect/wait_for_ready timeouts of whoever tries to use
    the stale entry.
    """
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # live process owned by another user
    except OSError:
        return False


def write_discovery(
    notebook_path: str,
    connection_file: str,
    kernel_name: str,
    pid: int,
    started_by: str,
) -> bool:
    """Write a kernel discovery file.

    Refuses to write in two cases:

    - ``pid <= 0``: ``read_discovery`` treats such an entry as stale and
      deletes it on first read, so writing it would silently break sharing.
    - An existing discovery file points at a LIVE kernel with a DIFFERENT
      connection file: clobbering it would hijack another participant's
      running session. Same connection file (restart/refresh) or a dead
      entry is overwritten normally.

    The write itself is atomic (tempfile in the same directory + os.replace)
    so a concurrent ``read_discovery`` never sees a partially written file.

    Args:
        notebook_path: Absolute path to the notebook.
        connection_file: Path to the ZMQ kernel connection file.
        kernel_name: Kernel spec name (e.g., "flowbook_kernel").
        pid: PID of the kernel process.
        started_by: Who started the kernel ("mcp" or "jupyterlab").

    Returns:
        True if the discovery file was written, False if the write was
        refused (invalid pid, or a live entry for a different kernel exists).
    """
    if not isinstance(pid, int) or pid <= 0:
        logger.warning(
            "Refusing to write kernel discovery file for %s: invalid pid %r "
            "(the entry would be treated as stale and deleted on first read)",
            notebook_path,
            pid,
        )
        return False

    path = _discovery_path(notebook_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Don't clobber a live entry that points at a DIFFERENT kernel.
    existing = _read_raw(path)
    if existing is not None:
        existing_pid = existing.get("pid")
        existing_conn = existing.get("connection_file")
        if (
            isinstance(existing_pid, int)
            and existing_pid > 0
            and _is_pid_alive(existing_pid)
            and existing_conn
            and existing_conn != connection_file
        ):
            logger.warning(
                "Refusing to overwrite live kernel discovery file for %s: "
                "existing entry (pid=%s, connection_file=%s, started_by=%s) "
                "points at a different kernel than %s",
                notebook_path,
                existing_pid,
                existing_conn,
                existing.get("started_by"),
                connection_file,
            )
            return False

    doc = {
        "notebook_path": notebook_path,
        "connection_file": connection_file,
        "kernel_name": kernel_name,
        "pid": pid,
        "started_by": started_by,
        "started_at": time.time(),
    }

    # Atomic write: tempfile in the same directory, then rename over the target.
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # Discovery files point at kernel connection files; keep them
            # owner-only. mkstemp already creates the temp file 0o600, but
            # set it explicitly so the final file's mode never depends on
            # that implementation detail (audit S4).
            os.fchmod(f.fileno(), 0o600)
            json.dump(doc, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        _remove_file(tmp_path)
        raise

    return True


def _read_raw(path: str) -> Optional[Dict[str, Any]]:
    """Read a discovery file without validation or cleanup. None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


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
