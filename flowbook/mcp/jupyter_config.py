"""
Auto-discover a running Jupyter Server for Y.js collaboration.

Reads server info files from Jupyter's runtime directory to find
a running server with its URL and token. Falls back to environment
variables JUPYTER_SERVER_URL and JUPYTER_TOKEN.
"""

import json
import os
from typing import Optional, Tuple

from jupyter_core.paths import jupyter_runtime_dir


def discover_jupyter_server() -> Tuple[Optional[str], Optional[str]]:
    """Find a running Jupyter Server and return (url, token).

    Discovery order:
    1. Environment variables JUPYTER_SERVER_URL and JUPYTER_TOKEN
    2. Server info files in Jupyter runtime directory

    Returns:
        (server_url, token) or (None, None) if no server found.
    """
    info = _discover_jupyter_server_info()
    if info:
        return info["url"], info.get("token")
    return None, None


def discover_jupyter_server_root() -> Optional[str]:
    """Find the running Jupyter Server's root directory.

    Returns:
        Absolute path to the server root directory, or None.
    """
    info = _discover_jupyter_server_info()
    if info:
        root = info.get("root_dir", "")
        if root:
            return os.path.abspath(os.path.expanduser(root))
    return None


def _discover_jupyter_server_info() -> Optional[dict]:
    """Find a running Jupyter Server and return its info dict.

    Discovery order:
    1. Environment variables JUPYTER_SERVER_URL and JUPYTER_TOKEN
    2. Server info files in Jupyter runtime directory

    Returns:
        Server info dict with url, token, root_dir, etc., or None.
    """
    # 1. Environment variables
    url = os.environ.get("JUPYTER_SERVER_URL")
    token = os.environ.get("JUPYTER_TOKEN")
    if url:
        return {"url": url.rstrip("/"), "token": token or None}

    # 2. Server info files in runtime directory
    runtime_dir = jupyter_runtime_dir()
    if not os.path.isdir(runtime_dir):
        return None

    # Find server info files (e.g., jpserver-12345.json, nbserver-12345.json)
    server_files = []
    for fname in os.listdir(runtime_dir):
        if (fname.startswith("jpserver-") or fname.startswith("nbserver-")) and fname.endswith(".json"):
            fpath = os.path.join(runtime_dir, fname)
            server_files.append(fpath)

    # Sort by modification time (most recent first)
    server_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    for fpath in server_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                info = json.load(f)
            server_url = info.get("url", "").rstrip("/")
            server_token = info.get("token", "")
            pid = info.get("pid")

            # Verify the server process is still running
            if pid:
                try:
                    os.kill(pid, 0)
                except (OSError, ProcessLookupError):
                    continue  # Server is dead, skip

            if server_url:
                return {
                    "url": server_url,
                    "token": server_token or None,
                    "root_dir": info.get("root_dir", ""),
                    "pid": pid,
                }
        except (json.JSONDecodeError, IOError):
            continue

    return None
