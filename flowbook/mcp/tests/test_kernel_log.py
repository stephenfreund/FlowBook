"""The kernel's fd-level output must not reach the session's process stdout.

FlowBook's logger writes to sys.__stdout__ to bypass IPython's redirection. A
kernel that inherits its parent's stdout therefore writes its logs there; in
the MCP server that stream is the JSON-RPC channel, and a burst of kernel
logging (e.g. the execution-timeout path) interleaved with a tool response
left Claude Code waiting forever. The session now routes kernel output to a
log file in the Jupyter runtime directory.
"""
import json
import os

import pytest

from flowbook.mcp.session import NotebookSession, default_cell_timeout


def _write_notebook(tmp_path, sources):
    nb = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{"cell_type": "code", "source": src, "id": f"c{i:03d}", "metadata": {}, "outputs": [], "execution_count": None}
                  for i, src in enumerate(sources)],
    }
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    return str(p)


def test_kernel_logs_go_to_a_file_not_parent_stdout(tmp_path, capfd):
    s = NotebookSession()
    try:
        s.load(_write_notebook(tmp_path, ["x = 1", "y = x + 1"]))
        s.run_cell("A")
        s.run_cell("B")
        log_path = s.kernel_log_path
    finally:
        s.close()
    out, err = capfd.readouterr()
    assert "Inst-Run" not in out and "Inst-Run" not in err, "kernel logs leaked into the parent's stdio"
    assert log_path and os.path.exists(log_path)
    text = open(log_path, "rb").read().decode("utf-8", "replace")
    assert "Inst-Run" in text


def test_cell_timeout_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOWBOOK_CELL_TIMEOUT_S", "2")
    assert default_cell_timeout() == 2.0
    s = NotebookSession()
    try:
        s.load(_write_notebook(tmp_path, ["import time\ntime.sleep(6)\nz = 1"]))
        r = s.run_cell("A")
        assert r["status"] == "timeout", r
    finally:
        s.close()
