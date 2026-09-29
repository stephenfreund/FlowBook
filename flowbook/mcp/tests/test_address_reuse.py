"""A column write to a temporary must not be reported as a write to another variable.

Regression: the trackers keyed records by id(obj). In the loop below every `ss`
is freed when `ss` is rebound, and CPython gives its address to the next
`X.iloc[...]` slice; the `num_sold` write to `ss` was then reported as
Col(X_train, num_sold), although X (and so X_train) has no num_sold column.
Keeping every `ss` alive made the spurious write disappear. Starts a real kernel.
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


def _col_writes(session, cid):
    locs = session.cell_flowbook_meta.get(cid, {}).get("write_locs") or []
    return [json.dumps(l, sort_keys=True) for l in locs if l.get("type") == "col"]


def test_temporary_column_write_not_attributed_to_later_slice(tmp_path):
    path = _write_notebook(tmp_path, [
        "import pandas as pd\nimport numpy as np\nX = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100)})",
        "for fold in range(20):\n    ss = pd.DataFrame()\n    ss['num_sold'] = np.zeros(3)\n    X_train = X.iloc[fold:fold + 50]",
    ])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        spurious = [l for l in _col_writes(s, "B") if "X_train" in l and "num_sold" in l]
        assert not spurious, f"write to a temporary attributed to X_train: {spurious}"
        assert any("num_sold" in l for l in _col_writes(s, "B")), "the write to ss itself is still recorded"
    finally:
        s.close()
