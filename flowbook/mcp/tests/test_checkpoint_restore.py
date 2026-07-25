"""Tests for NotebookSession.checkpoint()/restore() structural round-trips."""

from unittest.mock import MagicMock, patch

import pytest

from flowbook.mcp.session import NotebookSession
from flowbook.scripts.fix_repro_errors import get_cell_source


def _make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells,
    }


def _make_code_cell(cell_id, source, outputs=None, execution_count=None):
    return {
        "id": cell_id,
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": outputs or [],
        "execution_count": execution_count,
    }


def _make_session(cells):
    session = NotebookSession()
    session.notebook = _make_notebook(cells)
    session.notebook_path = "/abs/path/test.ipynb"
    return session


def _cell_ids(session):
    return [c.get("id") for c in session.notebook["cells"]]


@pytest.fixture(autouse=True)
def no_kernel_calls():
    """Stub out kernel communication for all tests in this module."""
    with patch(
        "flowbook.mcp.session.KernelHelper.execute_code",
        return_value={"flowbook_messages": []},
    ) as mock_exec:
        yield mock_exec


class TestCheckpoint:
    def test_checkpoint_snapshots_full_cells(self):
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ])
        ckpt_id = session.checkpoint()
        snap = session._checkpoints[ckpt_id]
        assert "cells" in snap
        assert [c["id"] for c in snap["cells"]] == ["A", "B"]

    def test_snapshot_is_isolated_from_later_edits(self):
        session = _make_session([_make_code_cell("A", "x = 1")])
        ckpt_id = session.checkpoint()
        session.notebook["cells"][0]["source"] = "x = 999"
        assert get_cell_source(session._checkpoints[ckpt_id]["cells"][0]) == "x = 1"

    def test_list_checkpoints_reports_cell_count(self):
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ])
        session.checkpoint()
        ckpts = session.list_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0]["cell_count"] == 2


class TestRestoreStructural:
    def test_deleted_cell_comes_back_in_order(self):
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
            _make_code_cell("C", "z = 3"),
        ])
        ckpt_id = session.checkpoint()

        # Delete B between checkpoint and restore
        session.notebook["cells"] = [
            c for c in session.notebook["cells"] if c["id"] != "B"
        ]
        assert _cell_ids(session) == ["A", "C"]

        result = session.restore(ckpt_id)

        assert _cell_ids(session) == ["A", "B", "C"]
        assert get_cell_source(session.notebook["cells"][1]) == "y = 2"
        assert "B" in result["changed_cells"]
        assert result["cells_reinserted"] == ["B"]

    def test_added_cell_is_removed(self):
        session = _make_session([_make_code_cell("A", "x = 1")])
        session.executed_cells = {"A"}
        ckpt_id = session.checkpoint()

        new_cell = _make_code_cell("A1", "w = 4")
        session.notebook["cells"].append(new_cell)
        session.executed_cells.add("A1")
        session.cell_flowbook_meta["A1"] = {"read_locs": []}
        session.cell_status["A1"] = "ok"
        session._stale_cells.add("A1")

        result = session.restore(ckpt_id)

        assert _cell_ids(session) == ["A"]
        assert result["cells_removed"] == ["A1"]
        # Bookkeeping for the removed cell is fully cleaned up
        assert "A1" not in session.executed_cells
        assert "A1" not in session.cell_flowbook_meta
        assert "A1" not in session.cell_status
        assert "A1" not in session._stale_cells

    def test_source_edit_reverted_and_marked_stale(self):
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ])
        session.executed_cells = {"A", "B"}
        session.cell_flowbook_meta["A"] = {"errors": [{"type": "VIOLATION"}]}
        session.cell_status["A"] = "error"
        ckpt_id = session.checkpoint()

        session.notebook["cells"][0]["source"] = "x = 42"
        result = session.restore(ckpt_id)

        assert get_cell_source(session.notebook["cells"][0]) == "x = 1"
        assert result["changed_cells"] == ["A"]
        # Same per-cell semantics as before: stale + metadata cleared
        assert "A" in session._stale_cells
        assert "A" not in session.cell_flowbook_meta
        assert "A" not in session.cell_status
        # Untouched cell is left alone
        assert "B" not in session._stale_cells

    def test_reinserted_executed_cell_marked_stale(self):
        """A cell that ran, then was deleted, is stale when reinserted."""
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ])
        session.executed_cells = {"A", "B"}
        ckpt_id = session.checkpoint()

        session.notebook["cells"] = [
            c for c in session.notebook["cells"] if c["id"] != "B"
        ]
        # B is still in executed_cells (kernel state survives deletion)
        result = session.restore(ckpt_id)

        assert _cell_ids(session) == ["A", "B"]
        assert "B" in session._stale_cells
        assert "B" in result["changed_cells"]

    def test_delete_edit_and_add_round_trip(self):
        """Combined structural + source changes are all undone."""
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
            _make_code_cell("C", "z = 3"),
        ])
        ckpt_id = session.checkpoint()

        # Delete B, edit C, insert D
        session.notebook["cells"] = [
            c for c in session.notebook["cells"] if c["id"] != "B"
        ]
        session.notebook["cells"][1]["source"] = "z = 300"
        session.notebook["cells"].append(_make_code_cell("D", "d = 4"))

        result = session.restore(ckpt_id)

        assert _cell_ids(session) == ["A", "B", "C"]
        assert get_cell_source(session.notebook["cells"][1]) == "y = 2"
        assert get_cell_source(session.notebook["cells"][2]) == "z = 3"
        assert set(result["changed_cells"]) == {"B", "C"}
        assert result["cells_reinserted"] == ["B"]
        assert result["cells_removed"] == ["D"]
        assert result["cells_restored"] == 2

    def test_return_payload_shape(self):
        session = _make_session([_make_code_cell("A", "x = 1")])
        ckpt_id = session.checkpoint()
        result = session.restore(ckpt_id)
        assert result["checkpoint_id"] == ckpt_id
        assert result["cells_restored"] == 0
        assert result["changed_cells"] == []

    def test_restore_notifies_kernel_structure(self, no_kernel_calls):
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ])
        session.kernel_client = MagicMock()
        ckpt_id = session.checkpoint()
        session.notebook["cells"] = [session.notebook["cells"][0]]

        session.restore(ckpt_id)

        structure_msgs = [
            call.kwargs.get("flowbook_msg")
            for call in no_kernel_calls.call_args_list
            if call.kwargs.get("flowbook_msg", {}).get("type") == "notebook_structure"
        ]
        assert structure_msgs
        assert structure_msgs[-1]["cell_order"] == ["A", "B"]

    def test_unknown_checkpoint_raises(self):
        session = _make_session([_make_code_cell("A", "x = 1")])
        with pytest.raises(ValueError, match="Unknown checkpoint"):
            session.restore("ckpt_nope")

    def test_restore_twice_is_stable(self):
        """The snapshot is deep-copied on restore, so it can be reused."""
        session = _make_session([
            _make_code_cell("A", "x = 1"),
            _make_code_cell("B", "y = 2"),
        ])
        ckpt_id = session.checkpoint()

        session.notebook["cells"] = [session.notebook["cells"][0]]
        session.restore(ckpt_id)
        assert _cell_ids(session) == ["A", "B"]

        # Mutate the restored notebook, then restore again
        session.notebook["cells"][1]["source"] = "y = 999"
        session.notebook["cells"] = [session.notebook["cells"][1]]
        session.restore(ckpt_id)
        assert _cell_ids(session) == ["A", "B"]
        assert get_cell_source(session.notebook["cells"][1]) == "y = 2"
