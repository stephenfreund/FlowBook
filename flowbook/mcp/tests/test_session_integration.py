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


class TestRefactoringToolsOverKernelController:
    """The 6 refactoring tools run through the unified handlers + KernelController.

    Exercises the real loaded session (kernel + Contents-API batching) end to
    end, verifying the MCP cutover preserves the historical return shapes.
    Cell ids are taken from load() since the session normalizes them.
    """

    def test_alpha_rename_renames_downstream(self, simple_notebook):
        session = NotebookSession()
        ids = session.load(simple_notebook)["cell_ids"]  # [x=1, y=x+1, print(y)]
        try:
            result = session.alpha_rename(ids[1], "y", "y2")
            assert result["total_modified"] == 2  # defines y, and the print(y)
            assert set(result["modified_cells"]) == {ids[1], ids[2]}
        finally:
            session.close()

    def test_alpha_rename_no_match_is_tolerant(self, simple_notebook):
        session = NotebookSession()
        ids = session.load(simple_notebook)["cell_ids"]
        try:
            result = session.alpha_rename(ids[0], "nonexistent", "z")
            assert result["total_modified"] == 0
            assert result["modified_cells"] == []
        finally:
            session.close()

    def test_merge_and_move(self, simple_notebook):
        session = NotebookSession()
        ids = session.load(simple_notebook)["cell_ids"]
        try:
            merged = session.merge_cells([ids[0], ids[1]])
            assert merged["merged_cell_id"] == ids[0]
            assert merged["cells_removed"] == [ids[1]]
            assert merged["new_cell_order"] == [ids[0], ids[2]]

            moved = session.move_cell(ids[0], ids[2])
            assert moved["new_cell_order"] == [ids[2], ids[0]]
        finally:
            session.close()

    def test_mark_diagnostic_idempotent(self, simple_notebook):
        session = NotebookSession()
        ids = session.load(simple_notebook)["cell_ids"]
        try:
            first = session.mark_diagnostic(ids[2])
            assert first["new_source_preview"].startswith("%diagnostic")
            again = session.mark_diagnostic(ids[2])
            assert again.get("already_diagnostic") is True
        finally:
            session.close()


class TestActorAttribution:
    """The kernel echoes the driving actor on flowbook metadata (Phase 4).

    This is the data path that lets a co-located LogBook attribute
    out-of-process MCP executions to origin: 'ai'.
    """

    def _metadata_actor(self, messages):
        for m in messages:
            if m.get("type") == "metadata":
                return m.get("actor")
        return "<<no metadata message>>"

    def test_ai_actor_echoed_when_mcp_runs_a_cell(self, simple_notebook):
        from flowbook.server.kernel_helper import KernelHelper
        session = NotebookSession()
        ids = session.load(simple_notebook)["cell_ids"]
        try:
            result = KernelHelper.execute_code(
                session.kernel_client, "x = 1", cell_id=ids[0], actor="ai"
            )
            assert self._metadata_actor(result["flowbook_messages"]) == "ai"
        finally:
            session.close()

    def test_defaults_to_user_when_actor_absent(self, simple_notebook):
        from flowbook.server.kernel_helper import KernelHelper
        session = NotebookSession()
        ids = session.load(simple_notebook)["cell_ids"]
        try:
            result = KernelHelper.execute_code(
                session.kernel_client, "x = 1", cell_id=ids[0]
            )
            assert self._metadata_actor(result["flowbook_messages"]) == "user"
        finally:
            session.close()
