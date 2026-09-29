"""A cell's column writes to a frame it creates are the same on a first run and a rerun.

    train_transformed = create_corr_features(train_temp)   # columns written inside
    X = train_transformed

The column tracker credits the writes to X. They used to be copied to the names'
aliases in the state before the cell: none on the first run, train_transformed on a
rerun (both names then held the previous run's shared frame). Now they go to the
names holding the new frame after the cell, on both runs. Starts a real kernel.
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


def _col_writes(meta):
    """Col write locations as (column, variable name); loc ids differ per object."""
    return {(l["name"], l.get("var_name", l.get("qualifier")))
            for l in (meta.get("write_locs") or []) if l["type"] == "col"}


def test_new_frame_column_writes_same_on_rerun(tmp_path):
    path = _write_notebook(tmp_path, [
        "import pandas as pd\n"
        "train_temp = pd.DataFrame({'f1': [1.0, 2.0], 'f2': [3.0, 4.0]})\n"
        "def create_corr_features(df):\n"
        "    out = df.copy()\n"
        "    out['f1_x_f2'] = out['f1'] * out['f2']\n"
        "    return out",
        "train_transformed = create_corr_features(train_temp)\n"
        "X = train_transformed",
    ])
    s = NotebookSession()
    try:
        s.load(path)
        a, b = s.get_cell_order()
        s.run_cell(a)
        s.run_cell(b)
        first = _col_writes(s.cell_flowbook_meta.get(b, {}))
        s.run_cell(b)
        rerun = _col_writes(s.cell_flowbook_meta.get(b, {}))
        assert first == rerun, f"first run {first} != rerun {rerun}"
        assert {("f1_x_f2", "X"), ("f1_x_f2", "train_transformed")} <= first, first
    finally:
        s.close()
