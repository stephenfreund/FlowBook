"""Class bodies, `global` statements and __module__ on a real FlowBook kernel.

Regression for 288f146 (TrackingDict as exec globals): a class body could not see
notebook variables (NameError), `global` writes were lost, `global x; del x` raised,
and notebook classes had __module__ 'builtins' so their instances could not be
pickled. Also checks that a class body's read of a notebook variable is recorded
and drives staleness, and that a `global` read-modify-write is seen by the checker.
These tests start real FlowBook kernels.
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


def _reads(session, cid):
    return {l["name"] for l in (session.cell_flowbook_meta.get(cid, {}).get("read_locs") or []) if l.get("type") == "var"}


def test_class_body_global_statement_and_module(tmp_path):
    path = _write_notebook(tmp_path, [
        "q = 1\ncounter = 0\ngone = 1",
        "class A:\n    v = q\nprint('A.v', A.v)",
        "import pickle\nclass K:\n    pass\ndef f():\n    pass\n"
        "print(K.__module__, f.__module__, type(pickle.loads(pickle.dumps(K()))).__name__)",
        "def inc():\n    global counter\n    counter += 1\ninc()\ninc()\nprint('counter', counter)",
        "def mk():\n    global made\n    made = 42\nmk()\nprint('made', made)",
        "def rm():\n    global gone\n    del gone\nrm()\nprint('gone' in dir())",
    ])
    expected = {"B": "A.v 1", "C": "__main__ __main__ K", "D": "counter 2", "E": "made 42", "F": "False"}
    s = NotebookSession()
    try:
        s.load(path)
        for cid in "ABCDEF":
            r = s.run_cell(cid)
            if cid == "D":
                # `global counter; counter += 1` reads and writes counter: rerunning D changes it.
                # The read and the write were both invisible before the fix.
                assert "no_read_and_write" in (r.get("error_message") or ""), f"D: {r.get('error_message')}"
                assert "counter" in r["error_message"]
            else:
                assert not r.get("error_message"), f"{cid}: {r.get('error_message')}"
            if cid in expected:
                assert expected[cid] in r["outputs_text"], f"{cid}: {r['outputs_text']!r}"
    finally:
        s.close()


def test_class_body_read_is_tracked_and_propagates(tmp_path):
    path = _write_notebook(tmp_path, ["x = 1", "class Cfg:\n    v = x * 2\nprint(Cfg.v)"])
    s = NotebookSession()
    try:
        s.load(path)
        s.run_all()
        assert "x" in _reads(s, "B"), "read of x from a class body not recorded"
        s.edit_cell("A", "x = 5")
        s.run_cell("A")
        assert "B" in s.get_status()["stale_cells"], "class-body reader not marked stale after x changed"
    finally:
        s.close()
