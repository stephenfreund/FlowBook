"""Integration tests for MCP NotebookSession with kernel discovery and Y.js.

These tests require a FlowBook kernel to be installable and may take
a few seconds each due to kernel startup time.
"""

import json
import os

import pytest

from flowbook.kernel_discovery import read_discovery
from flowbook.mcp.session import NotebookSession


@pytest.fixture
def simple_notebook(tmp_path):
    """Create a minimal test notebook."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "flowbook_kernel"}},
        "cells": [
            {
                "cell_type": "code",
                "source": "x = 1",
                "id": "aaaa",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "source": "y = x + 1",
                "id": "bbbb",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "source": "print(y)",
                "id": "cccc",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    path = tmp_path / "test.ipynb"
    path.write_text(json.dumps(nb))
    return str(path)


class TestSessionLifecycle:
    def test_load_starts_kernel(self, simple_notebook):
        session = NotebookSession()
        result = session.load(simple_notebook)

        assert result["code_cells"] == 3
        assert session._owns_kernel is True
        assert session.kernel_client is not None

        session.close()

    def test_discovery_file_written_on_load(self, simple_notebook):
        session = NotebookSession()
        session.load(simple_notebook)

        abs_path = os.path.abspath(simple_notebook)
        disc = read_discovery(abs_path)
        assert disc is not None
        assert disc["started_by"] == "mcp"
        assert disc["pid"] > 0

        session.close()

    def test_discovery_file_cleaned_on_close(self, simple_notebook):
        session = NotebookSession()
        session.load(simple_notebook)
        abs_path = os.path.abspath(simple_notebook)

        session.close()

        assert read_discovery(abs_path) is None

    def test_run_cell(self, simple_notebook):
        session = NotebookSession()
        session.load(simple_notebook)

        result = session.run_cell("A")
        assert result["status"] == "ok"

        session.close()

    def test_run_multiple_cells(self, simple_notebook):
        session = NotebookSession()
        session.load(simple_notebook)

        r1 = session.run_cell("A")
        r2 = session.run_cell("B")
        r3 = session.run_cell("C")
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"
        assert r3["status"] == "ok"
        assert "2" in r3["outputs_text"]  # print(y) should output 2

        session.close()

    def test_edit_cell(self, simple_notebook):
        session = NotebookSession()
        session.load(simple_notebook)

        # Run first cell, then edit it
        session.run_cell("A")
        result = session.edit_cell("A", "x = 99")
        assert result["marked_stale"] is True

        session.close()


class TestSharedKernel:
    def test_second_session_joins_existing_kernel(self, simple_notebook):
        s1 = NotebookSession()
        s1.load(simple_notebook)
        assert s1._owns_kernel is True

        s2 = NotebookSession()
        r2 = s2.load(simple_notebook)
        assert s2._owns_kernel is False
        assert r2["joined_existing"] is True

        s2.close()
        s1.close()

    def test_joiner_does_not_kill_kernel(self, simple_notebook):
        s1 = NotebookSession()
        s1.load(simple_notebook)

        s2 = NotebookSession()
        s2.load(simple_notebook)

        # Close joiner
        s2.close()

        # Owner's kernel should still work
        result = s1.run_cell("A")
        assert result["status"] == "ok"

        s1.close()

    def test_both_can_execute_on_shared_kernel(self, simple_notebook):
        s1 = NotebookSession()
        s1.load(simple_notebook)

        s2 = NotebookSession()
        s2.load(simple_notebook)

        # S1 runs cell A (x = 1)
        r1 = s1.run_cell("A")
        assert r1["status"] == "ok"

        # S2's cell IDs are un-normalized (original IDs)
        cell_ids_s2 = s2.get_cell_order()
        # S2 runs cell B equivalent (y = x + 1) on the SAME kernel
        r2 = s2.run_cell(cell_ids_s2[1])
        assert r2["status"] == "ok"

        s2.close()
        s1.close()

    def test_joiner_skips_id_normalization(self, simple_notebook):
        s1 = NotebookSession()
        s1.load(simple_notebook)
        ids_s1 = s1.get_cell_order()

        s2 = NotebookSession()
        s2.load(simple_notebook)
        ids_s2 = s2.get_cell_order()

        # S1 normalized: A, B, C
        assert ids_s1 == ["A", "B", "C"]
        # S2 joined: original IDs from the notebook file
        assert ids_s2 == ["aaaa", "bbbb", "cccc"]

        s2.close()
        s1.close()

    def test_discovery_survives_joiner_close(self, simple_notebook):
        s1 = NotebookSession()
        s1.load(simple_notebook)
        abs_path = os.path.abspath(simple_notebook)

        s2 = NotebookSession()
        s2.load(simple_notebook)

        s2.close()

        # Discovery file should still exist (s1 owns the kernel)
        disc = read_discovery(abs_path)
        assert disc is not None

        s1.close()
        assert read_discovery(abs_path) is None
