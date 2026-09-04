"""NotebookSession can drive a stock kernel via ``kernel_name``.

Used by the FlowBook-Comparison bench, whose control arm runs the same
session machinery on a plain ``python3`` kernel. These tests start real
kernels and take a few seconds each.
"""
import json
import os

import pytest

from flowbook.kernel_discovery import read_discovery
from flowbook.mcp.session import NotebookSession


def _write_notebook(tmp_path, sources):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "source": src,
                "id": f"c{i:03d}",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            for i, src in enumerate(sources)
        ],
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))
    return str(path)


@pytest.fixture
def chain_notebook(tmp_path):
    return _write_notebook(tmp_path, ["x = 1", "y = x + 1", "print(y)"])


def test_default_kernel_is_flowbook():
    assert NotebookSession().kernel_name == "flowbook_kernel"


def test_plain_kernel_runs_without_flowbook_metadata(chain_notebook):
    session = NotebookSession(kernel_name="python3")
    try:
        info = session.load(chain_notebook)
        assert info["code_cells"] == 3
        assert read_discovery(os.path.abspath(chain_notebook))["kernel_name"] == "python3"

        r1 = session.run_cell("A")
        r2 = session.run_cell("B")
        r3 = session.run_cell("C")
        assert (r1["status"], r2["status"], r3["status"]) == ("ok", "ok", "ok")
        assert "2" in r3["outputs_text"]

        # No reproducibility metadata from a stock kernel
        assert "flowbook" not in r3
        assert session.cell_flowbook_meta == {}
        assert session._stale_cells == set()

        # Trace still records the executions, with no stale information
        runs = [e for e in session._trace if e["kind"] == "run"]
        assert [e["cell_id"] for e in runs] == ["A", "B", "C"]
        assert all(e["stale_cells"] is None for e in runs)
        assert all(e["rejected"] is False for e in runs)

        # FlowBook-only queries degrade gracefully
        status = session.get_status()
        assert isinstance(status, dict)
    finally:
        session.close()


def test_plain_kernel_edit_leaves_staleness_alone(chain_notebook):
    session = NotebookSession(kernel_name="python3")
    try:
        session.load(chain_notebook)
        session.run_cell("A")
        result = session.edit_cell("A", "x = 99")
        # No kernel reply carries staleness, so nothing is marked stale
        assert result["marked_stale"] is False
        edits = [e for e in session._trace if e["kind"] == "edit"]
        assert len(edits) == 1
        assert edits[0]["cell_id"] == "A"
        assert edits[0]["source"] == "x = 99"
        assert edits[0]["executed"] is True
    finally:
        session.close()


def test_flowbook_kernel_still_default_and_reports_metadata(chain_notebook):
    session = NotebookSession()
    try:
        session.load(chain_notebook)
        r = session.run_cell("A")
        assert r["status"] == "ok"
        assert "flowbook" in r
        assert "A" in session.cell_flowbook_meta
    finally:
        session.close()
