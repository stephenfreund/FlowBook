"""Structured execution trace recorded by NotebookSession.

The trace (``session._trace``, written by ``save_event_log`` under
``"trace"``) is the raw, untruncated record of executions, edits, and
structural changes that an offline consumer replays. These tests start
real FlowBook kernels and take a few seconds each.
"""
import json

import pytest

from flowbook.mcp.session import NotebookSession


def _write_notebook(tmp_path, sources, name="nb.ipynb"):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "source": src,
                "id": f"c{i:03d}",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            for i, src in enumerate(sources)
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(nb))
    return str(path)


@pytest.fixture
def chain_notebook(tmp_path):
    # A writes x; B reads x, writes y; C reads y
    return _write_notebook(tmp_path, ["x = 1", "y = x + 1", "print(y)"])


@pytest.fixture
def violating_notebook(tmp_path):
    # C writes x after B read it: NoWriteAfterRead violation when C runs
    return _write_notebook(tmp_path, ["x = 1", "print(x)", "x = 2"])


@pytest.fixture
def session():
    s = NotebookSession()
    yield s
    s.close()


def _runs(session):
    return [e for e in session._trace if e["kind"] == "run"]


def test_run_records_one_per_execution(session, chain_notebook):
    session.load(chain_notebook)
    for cid in ("A", "B", "C"):
        session.run_cell(cid)

    runs = _runs(session)
    assert [e["cell_id"] for e in runs] == ["A", "B", "C"]
    assert [e["seq"] for e in runs] == [0, 1, 2]
    for e in runs:
        assert e["cell_order"] == ["A", "B", "C"]
        assert e["status"] == "ok"
        assert isinstance(e["source"], str) and e["source"]
        assert len(e["source_sha1"]) == 40
        assert isinstance(e["stale_cells"], list)
        assert e["errors"] == []
        assert e["rejected"] is False


def test_bulk_runs_route_through_run_cell(session, chain_notebook):
    session.load(chain_notebook)
    session.run_all()
    assert [e["cell_id"] for e in _runs(session)] == ["A", "B", "C"]
    # run_from skips clean cells, so dirty B first; C goes stale when B reruns
    session.edit_cell("B", "y = x + 5")
    session.run_from("B")
    assert [e["cell_id"] for e in _runs(session)] == ["A", "B", "C", "B", "C"]


def test_edit_record_then_forward_staleness(session, chain_notebook):
    session.load(chain_notebook)
    for cid in ("A", "B", "C"):
        session.run_cell(cid)

    session.edit_cell("A", "x = 99")
    edits = [e for e in session._trace if e["kind"] == "edit"]
    assert len(edits) == 1
    assert edits[0]["cell_id"] == "A"
    assert edits[0]["source"] == "x = 99"
    assert edits[0]["executed"] is True
    assert "A" in edits[0]["stale_after"]

    session.run_cell("A")
    last = _runs(session)[-1]
    assert last["cell_id"] == "A"
    assert last["source"] == "x = 99"
    # ForwardStale marks direct readers of what A wrote; C follows when B reruns
    assert "B" in last["stale_cells"]
    assert "C" not in last["stale_cells"]
    assert last["rejected"] is False

    session.run_cell("B")
    last = _runs(session)[-1]
    assert "C" in last["stale_cells"]


def test_rejected_violation_is_flagged(session, violating_notebook):
    session.load(violating_notebook)
    session.run_cell("A")
    session.run_cell("B")
    r = session.run_cell("C")
    assert r["status"] == "error"

    last = _runs(session)[-1]
    assert last["cell_id"] == "C"
    assert last["status"] == "error"
    assert last["errors"], "metadata should carry the predicate violation"
    assert last["rejected"] is True


def test_accepted_violation_is_not_rejected(session, violating_notebook):
    session.load(violating_notebook)
    session.run_cell("A")
    session.run_cell("B")
    session.set_continue_after_violation(True)
    r = session.run_cell("C")
    assert r["status"] == "ok"

    last = _runs(session)[-1]
    assert last["errors"], "accepted violations are still reported"
    assert last["rejected"] is False


def test_structure_records(session, chain_notebook):
    session.load(chain_notebook)
    inserted = session.insert_cell("A", "z = 0")
    new_id = inserted["new_cell_id"]
    session.delete_cell(new_id)

    structure = [e for e in session._trace if e["kind"] == "structure"]
    assert [e["op"] for e in structure] == ["insert_cell", "delete_cell"]
    assert structure[0]["cell_id"] == new_id
    assert new_id in structure[0]["cell_order"]
    assert new_id not in structure[1]["cell_order"]


def test_save_event_log_includes_trace(session, chain_notebook, tmp_path):
    session.load(chain_notebook)
    session.run_cell("A")
    session.edit_cell("A", "x = 2")

    out = tmp_path / "log.json"
    session.save_event_log(str(out))
    doc = json.loads(out.read_text())
    assert [e["kind"] for e in doc["trace"]] == ["run", "edit"]
    assert doc["trace"][0]["cell_id"] == "A"


def test_trace_resets_on_reload(session, chain_notebook, tmp_path):
    session.load(chain_notebook)
    session.run_cell("A")
    assert session._trace

    other = _write_notebook(tmp_path, ["a = 1"], name="other.ipynb")
    session.load(other)
    assert session._trace == []
