"""Column reads of a DataFrame that is also in IPython's output history are recorded.

Cell B displays X, so X is IPython's `_`. Cell C reads X['a'] and ends in an
expression, so IPython's display hook reads _/__/___ through the TrackingDict.
That read used to register X's DataFrame under '_', and C's read of X['a'] was
then dropped. Starts a real kernel.
"""
import json

from flowbook.mcp.session import NotebookSession


def _write_notebook(tmp_path, sources, name="nb.ipynb"):
    nb = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{"cell_type": "code", "source": src, "id": f"c{i:03d}", "metadata": {}, "outputs": [], "execution_count": None}
                  for i, src in enumerate(sources)],
    }
    path = tmp_path / name
    path.write_text(json.dumps(nb))
    return str(path)


def test_column_read_of_displayed_dataframe_is_recorded(tmp_path):
    path = _write_notebook(tmp_path, [
        "import pandas as pd\nX = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})",
        "X",
        "v = X['a'].sum()\nv",
    ])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        reads = [json.dumps(l, sort_keys=True) for l in (s.cell_flowbook_meta.get("C", {}).get("read_locs") or [])]
        assert any('"col"' in l and '"a"' in l and "X" in l for l in reads), f"read of X['a'] missing: {reads}"
    finally:
        s.close()
