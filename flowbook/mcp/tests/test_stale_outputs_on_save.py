"""FLOWBOOK_STALE_OUTPUTS_ON_SAVE=clear: a saved notebook never shows a stale cell's output
as current. Starts a real FlowBook kernel."""
import json

from flowbook.mcp.session import NotebookSession


def _write_notebook(tmp_path, sources, name="nb.ipynb"):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"cell_type": "code", "source": src, "id": f"c{i:03d}", "metadata": {}, "outputs": [], "execution_count": None}
                    for i, src in enumerate(sources)]}
    path = tmp_path / name
    path.write_text(json.dumps(nb))
    return path


def _outputs(path, cell_id):
    nb = json.loads(path.read_text())
    cell = next(c for c in nb["cells"] if c["id"] == cell_id)
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream").strip()


def test_stale_cell_outputs_are_cleared_on_save(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOWBOOK_STALE_OUTPUTS_ON_SAVE", "clear")
    path = _write_notebook(tmp_path, ["x = 1", "y = x + 1\nprint(y)", "print(x)"])
    s = NotebookSession()
    try:
        s.load(str(path))
        s.run_all()
        s.save()
        assert _outputs(path, "B") == "2"
        s.edit_cell("A", "x = 5")
        s.run_cell("A")                      # B and C stale
        s.save()
        assert _outputs(path, "B") == "" and _outputs(path, "C") == ""
        assert s.get_cell("B")["outputs_text"].strip() == "2"     # the session keeps its copy
        s.run_cell("B"); s.run_cell("C")
        s.save()
        assert _outputs(path, "B") == "6"
    finally:
        s.close()


def test_default_keeps_stale_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOWBOOK_STALE_OUTPUTS_ON_SAVE", raising=False)
    path = _write_notebook(tmp_path, ["x = 1", "y = x + 1\nprint(y)"])
    s = NotebookSession()
    try:
        s.load(str(path)); s.run_all(); s.edit_cell("A", "x = 5"); s.run_cell("A"); s.save()
        assert _outputs(path, "B") == "2"
    finally:
        s.close()
