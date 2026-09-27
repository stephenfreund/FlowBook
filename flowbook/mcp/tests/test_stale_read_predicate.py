"""NoReadOfStale (opt-in via FLOWBOOK_REJECT_STALE_READS): a cell that reads a location
whose last writer is stale is rejected instead of being recorded clean. These tests start
real FlowBook kernels with the flag set in the kernel's environment."""
import json
import os

import pytest

from flowbook.mcp.session import NotebookSession


def _write_notebook(tmp_path, sources, name="nb.ipynb"):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"cell_type": "code", "source": src, "id": f"c{i:03d}", "metadata": {}, "outputs": [], "execution_count": None}
                    for i, src in enumerate(sources)]}
    path = tmp_path / name
    path.write_text(json.dumps(nb))
    return str(path)


@pytest.fixture
def strict_env(monkeypatch):
    monkeypatch.setenv("FLOWBOOK_REJECT_STALE_READS", "1")


def test_read_of_edited_but_unrun_cell_is_rejected(tmp_path, strict_env):
    path = _write_notebook(tmp_path, ["x = 1", "y = x + 1\nprint(y)", "z = y * 2\nprint(z)"])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        s.edit_cell("A", "x = 5")            # A stale (code changed), never rerun
        r = s.run_cell("B")                  # reads x, whose last writer A is stale
        assert r["status"] == "error", r
        assert "stale_read" in json.dumps(s.cell_flowbook_meta.get("B", {}))
        s.run_cell("A")
        assert s.run_cell("B")["status"] == "ok"
    finally:
        s.close()


def test_read_below_a_stale_but_unrun_writer_is_rejected(tmp_path, strict_env):
    """A reruns with a new value: B is forward-stale. Running C (reads y from B) before B
    is a stale read; after B reruns it is fine."""
    path = _write_notebook(tmp_path, ["x = 1", "y = x + 1\nprint(y)", "z = y * 2\nprint(z)"])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        s.edit_cell("A", "x = 5")
        s.run_cell("A")
        assert "B" in s.get_status()["stale_cells"]
        r = s.run_cell("C")
        assert r["status"] == "error", r
        assert s.run_cell("B")["status"] == "ok"
        assert s.run_cell("C")["status"] == "ok"
        assert s.get_cell("C")["outputs_text"].strip() == "12"
    finally:
        s.close()


def test_default_still_allows_the_read(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOWBOOK_REJECT_STALE_READS", raising=False)
    path = _write_notebook(tmp_path, ["x = 1", "y = x + 1\nprint(y)"])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        s.edit_cell("A", "x = 5")
        assert s.run_cell("B")["status"] == "ok"
    finally:
        s.close()


def test_read_of_a_clean_cell_computed_from_stale_state_is_rejected(tmp_path, strict_env):
    """A reruns with a new value: B is stale. C already ran (clean by the staleness rules)
    but its value came from B's old output. D reads C: a transitively stale read."""
    path = _write_notebook(tmp_path, ["x = 1", "y = x + 1\nprint(y)", "z = y * 2\nprint(z)", "w = z + 1\nprint(w)"])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        s.edit_cell("A", "x = 5")
        s.run_cell("A")                       # B stale; C and D untouched, clean by the rules
        assert set(s.get_status()["stale_cells"]) == {"B"}
        r = s.run_cell("D")                   # reads z from C, which read y from stale B
        assert r["status"] == "error", r
        assert "stale_read" in json.dumps(s.cell_flowbook_meta.get("D", {}))
        s.run_cell("B")
        s.run_cell("C")
        assert s.run_cell("D")["status"] == "ok"
        assert s.get_cell("D")["outputs_text"].strip() == "13"
    finally:
        s.close()
