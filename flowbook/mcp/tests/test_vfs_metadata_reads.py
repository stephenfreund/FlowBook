"""File existence checks are reads under the full virtual filesystem, on a real FlowBook kernel.

The FlowBook kernel runs its VFS in full (overlay) mode by default. Full mode used to record
nothing for os.path.exists/os.listdir and did not patch os.stat, so a caching cell
(`if exists(p): load p else: compute and save p`) looked like a plain writer of p on its first
run, although a rerun loads the saved file instead of recomputing it. With existence checks
recorded as reads, the first run reads and writes p: NoReadAndWrite. These tests start real
FlowBook kernels.
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


def _file_reads(session, cid):
    return {l["name"] for l in (session.cell_flowbook_meta.get(cid, {}).get("read_locs") or []) if l.get("type") == "file"}


CACHING_CELL = """\
def predict(i):
    return np.arange(3) * i

def load_or_predict():
    if all(os.path.exists(f"pred{i}.npy") for i in (1, 2, 3)):
        return [np.load(f"pred{i}.npy") for i in (1, 2, 3)]
    preds = [predict(1), predict(2), predict(3)]
    for i, p in enumerate(preds, 1):
        np.save(f"pred{i}.npy", p)
    return preds

preds = load_or_predict()
print(len(preds))
"""


def test_caching_cell_is_read_and_write(tmp_path):
    path = _write_notebook(tmp_path, [
        "import os\nimport numpy as np",
        CACHING_CELL,
        "np.save('other.npy', np.arange(3))",
        "print(os.path.isfile('other.npy'), os.path.exists('other.npy'))",
    ])
    s = NotebookSession()
    try:
        s.load(path)
        assert not s.run_cell("A").get("error_message")

        # First execution: the existence check reads pred1.npy (all() stops at the first
        # missing file) and np.save writes it.
        r = s.run_cell("B")
        msg = r.get("error_message") or ""
        assert "no_read_and_write" in msg, f"B: {msg!r}"
        assert "pred1.npy" in msg, f"B: {msg!r}"

        # A plain writer is not affected.
        r = s.run_cell("C")
        assert not r.get("error_message"), f"C: {r.get('error_message')}"

        # isfile/exists see the overlay copy written by C and are recorded as reads.
        r = s.run_cell("D")
        assert not r.get("error_message"), f"D: {r.get('error_message')}"
        assert "True True" in r["outputs_text"], f"D: {r['outputs_text']!r}"
        assert any(p.endswith("other.npy") for p in _file_reads(s, "D")), _file_reads(s, "D")
    finally:
        s.close()
