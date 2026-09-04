"""Rerunning a cell after its execution was rejected must still execute.

Regression: when a rejected cell's pre-checkpoint held file snapshots (an
earlier cell had written a file), the kernel kept that checkpoint for the
rollback, and the next execution of the cell crashed inside the file
checkpoint's directory cleanup ("Directory not empty"): shutil.rmtree deletes
entries by dir_fd-relative name, which the virtual filesystem patches resolved
against the cwd and virtualized. The kernel then answered later executions of
that cell with no output and no metadata, so the stale violation never cleared.
"""
import json

import pytest

from flowbook.mcp.session import NotebookSession


def _write_notebook(tmp_path, sources):
    nb = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{"cell_type": "code", "source": src, "id": f"c{i:03d}", "metadata": {}, "outputs": [], "execution_count": None}
                  for i, src in enumerate(sources)],
    }
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    return str(p)


def test_rerun_after_rejection_with_file_snapshot(tmp_path):
    path = _write_notebook(tmp_path, [
        'x = 1\nwith open("out.json", "w") as f:\n    f.write("{}")',   # A: writes a file
        "print(x)",                                                    # B: reads x
        "x = 2\nx",                                                    # C: writes x after B read it -> rejected
    ])
    s = NotebookSession()
    try:
        s.load(path)
        assert s.run_cell("A")["status"] == "ok"
        assert s.run_cell("B")["status"] == "ok"
        r = s.run_cell("C")
        assert r["status"] == "error" and "flowbook" in r
        assert [e["error_type"] for e in s.cell_flowbook_meta["C"]["errors"]] == ["no_write_after_read"]

        s.edit_cell("C", "y = 2\ny")
        r = s.run_cell("C")
        assert r["status"] == "ok", r
        assert "flowbook" in r, "metadata must come back after the fix"
        assert "2" in r["outputs_text"]
        assert s.cell_flowbook_meta["C"].get("errors", []) == []
        st = s.get_status()
        assert st["violations"] == []

        # and again, plus another cell, to be sure the kernel is healthy
        assert s.run_cell("C")["status"] == "ok"
        assert "1" in s.run_cell("B")["outputs_text"]
    finally:
        s.close()
