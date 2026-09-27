"""Tool output labels cells by numeric position (#n) next to the alpha ID, so a position
can never be passed as an ID. Starts a real FlowBook kernel."""
import json
from types import SimpleNamespace

from flowbook.mcp import server as fb
from flowbook.mcp.session import NotebookSession


def _ctx(session):
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context={"session": session}))


def test_labels_are_numeric_positions(tmp_path):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"cell_type": "code", "source": src, "id": f"c{i:03d}", "metadata": {}, "outputs": [], "execution_count": None}
                    for i, src in enumerate(["x = 1", "y = x + 1", "print(y)"])]}
    path = tmp_path / "nb.ipynb"; path.write_text(json.dumps(nb))
    s = NotebookSession()
    try:
        s.load(str(path))
        out = fb.get_all_cell_sources(_ctx(s))
        assert "── #1 [A]" in out and "── #3 [C]" in out and "@A" not in out
        assert fb._cell_label(s, "B") == "#2"
        s.insert_cell("A", "z = 0", "code")
        assert fb._cell_label(s, "B") == "#3"      # position moved, ID did not
    finally:
        s.close()
