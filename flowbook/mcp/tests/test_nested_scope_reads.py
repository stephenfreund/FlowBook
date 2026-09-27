"""Reads of globals from nested scopes (function bodies, comprehensions, generator
expressions) are recorded by the kernel and drive staleness.

Regression: ``user_global_ns`` is a property on IPython's InteractiveShell, so the
kernel's instance-dict shadow of it never took effect and every LOAD_GLOBAL from a
nested scope read the untracked module dict. These tests start real FlowBook kernels.
"""
import json

import pytest

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


def _reads(session, cid):
    return {l["name"] for l in (session.cell_flowbook_meta.get(cid, {}).get("read_locs") or []) if l.get("type") == "var"}


@pytest.mark.parametrize("name, sources, reader", [
    ("comprehension", ["x = 1", "lst = [x * f for f in (1, 2)]\nprint(lst)"], "B"),
    ("genexpr", ["x = 1", "tot = sum(x * f for f in (1, 2))\nprint(tot)"], "B"),
    ("closure", ["x = 1", "def adj(v):\n    return v * x", "y = adj(3)\nprint(y)"], "C"),
    ("lambda", ["x = 1", "f = lambda v: v + x", "y = f(2)\nprint(y)"], "C"),
])
def test_nested_scope_read_is_tracked_and_propagates(tmp_path, name, sources, reader):
    path = _write_notebook(tmp_path, sources, name=f"{name}.ipynb")
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        assert "x" in _reads(s, reader), f"{name}: read of x from a nested scope not recorded"
        s.edit_cell("A", "x = 5")
        s.run_cell("A")
        assert reader in s.get_status()["stale_cells"], f"{name}: reader not marked stale after x changed"
    finally:
        s.close()


def test_top_level_reads_still_tracked(tmp_path):
    path = _write_notebook(tmp_path, ["x = 1", "y = x * 2\nprint(y)"])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        assert _reads(s, "B") == {"x"}
    finally:
        s.close()
