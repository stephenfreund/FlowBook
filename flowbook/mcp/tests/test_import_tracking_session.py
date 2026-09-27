"""The bench scenario, end to end on the real kernel: a results cell holding `import json`
ends up below a cell that calls json.dump after later inserts; that cell must be rejected."""
import json

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


def test_results_cell_using_json_before_the_import_cell_is_rejected(tmp_path):
    path = _write_notebook(tmp_path, ["x = 1", 'import json\nwith open("r1.json", "w") as f:\n    json.dump({"x": x}, f)'])
    s = NotebookSession()
    try:
        s.load(path)
        assert s.run_cell("A")["status"] == "ok"
        assert s.run_cell("B")["status"] == "ok"           # B: import json + write r1
        new_id = s.insert_cell("A", 'with open("r2.json", "w") as f:\n    json.dump({"x": x}, f)')["new_cell_id"]
        r = s.run_cell(new_id)                             # inserted above B, uses json
        assert r["status"] == "error", r
        assert [e["error_type"] for e in s.cell_flowbook_meta[new_id]["errors"]] == ["no_read_before_write"]
        # a re-import in a later cell is fine and marks nothing stale
        again = s.insert_cell("B", 'import json\nprint(json.dumps([]))')["new_cell_id"]
        assert s.run_cell(again)["status"] == "ok"
        assert "B" not in s._stale_cells
    finally:
        s.close()
