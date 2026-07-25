"""Tests for NotebookSession state bookkeeping (audit C9).

(b) get_status clean-count arithmetic: cells that are both error and stale
    were double-counted, and error entries for since-deleted cells could
    drive the count negative.
(c) load() computed abs_path (with ~ expansion) but opened the raw path,
    so ``~/nb.ipynb`` failed.
"""

import json
import os
from unittest.mock import patch

import pytest

from flowbook.mcp.session import NotebookSession


def _make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells,
    }


def _make_code_cell(cell_id, source):
    return {
        "id": cell_id,
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }


def _bare_session(cells):
    """Session with a notebook but no kernel and no Contents API."""
    session = NotebookSession()
    session.notebook = _make_notebook(cells)
    session.notebook_path = "/abs/path/test.ipynb"
    return session


class TestGetStatusCleanCount:
    def test_error_and_stale_cell_not_double_counted(self):
        """A cell that is both error and stale must be subtracted once."""
        session = _bare_session(
            [_make_code_cell("A", "x = 1"), _make_code_cell("B", "y = 2")]
        )
        session.executed_cells = {"A", "B"}
        session.cell_status = {"A": "error", "B": "ok"}
        session._stale_cells = {"A"}  # A is BOTH error and stale

        status = session.get_status()

        # Old arithmetic: 2 executed - 1 error - 1 stale = 0. Correct: B is clean.
        assert "Clean: 1" in status["summary"]

    def test_deleted_error_cells_do_not_go_negative(self):
        """Error entries for cells no longer in the notebook must be ignored."""
        session = _bare_session([_make_code_cell("A", "x = 1")])
        # "gone1"/"gone2" were executed with errors, then deleted.
        session.executed_cells = {"A", "gone1", "gone2"}
        session.cell_status = {"A": "error", "gone1": "error", "gone2": "error"}
        session._stale_cells = {"A"}

        status = session.get_status()

        # Old arithmetic: 1 executed - 3 errors - 1 stale = -3.
        assert "Clean: 0" in status["summary"]
        assert status["executed"] == 1

    def test_all_clean(self):
        session = _bare_session(
            [_make_code_cell("A", "x = 1"), _make_code_cell("B", "y = 2")]
        )
        session.executed_cells = {"A", "B"}
        session.cell_status = {"A": "ok", "B": "ok"}

        status = session.get_status()
        assert "Clean: 2" in status["summary"]

    def test_stale_cell_outside_order_ignored(self):
        """Stale entries for deleted cells must not reduce the clean count."""
        session = _bare_session([_make_code_cell("A", "x = 1")])
        session.executed_cells = {"A"}
        session.cell_status = {"A": "ok"}
        session._stale_cells = {"deleted-cell"}

        status = session.get_status()
        assert "Clean: 1" in status["summary"]


class TestLoadTildeExpansion:
    def test_load_opens_expanded_path(self, tmp_path, monkeypatch):
        """load('~/nb.ipynb') must open the expanded absolute path."""
        monkeypatch.setenv("HOME", str(tmp_path))
        nb = _make_notebook([_make_code_cell("abcd", "x = 1")])
        nb_path = tmp_path / "nb.ipynb"
        nb_path.write_text(json.dumps(nb))

        session = NotebookSession()
        with patch("flowbook.mcp.session.read_discovery", return_value=None), \
             patch("flowbook.mcp.session.setup_kernel", return_value=(None, None)), \
             patch("flowbook.mcp.session.discover_jupyter_server", return_value=(None, None)):
            result = session.load("~/nb.ipynb")

        assert session.notebook is not None
        assert session.notebook_path == str(nb_path)
        assert result["total_cells"] == 1
