"""Tests for FlowBookSession (checkpoints + event log)."""

import json
import os
import tempfile
import time

import pytest

from flowbook.nbi.session import FlowBookSession


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def session():
    return FlowBookSession()


@pytest.fixture
def sample_cells():
    return [
        {"label": "@A", "cell_type": "code", "source": "x = 1"},
        {"label": "@B", "cell_type": "code", "source": "y = x + 1"},
    ]


# ------------------------------------------------------------------
# Checkpoint tests
# ------------------------------------------------------------------

class TestCheckpoints:
    def test_save_and_retrieve(self, session, sample_cells):
        cp_id = session.save_checkpoint(sample_cells)
        assert cp_id == "ckpt_0"
        assert session.get_checkpoint(cp_id) == sample_cells

    def test_incremental_ids(self, session, sample_cells):
        cp0 = session.save_checkpoint(sample_cells)
        cp1 = session.save_checkpoint(sample_cells)
        cp2 = session.save_checkpoint(sample_cells)
        assert cp0 == "ckpt_0"
        assert cp1 == "ckpt_1"
        assert cp2 == "ckpt_2"

    def test_missing_checkpoint_raises(self, session):
        with pytest.raises(KeyError, match="nonexistent"):
            session.get_checkpoint("nonexistent")

    def test_list_checkpoints(self, session, sample_cells):
        session.save_checkpoint(sample_cells)
        session.save_checkpoint(sample_cells + [{"label": "@C", "cell_type": "code", "source": "z = 3"}])
        listing = session.list_checkpoints()
        assert len(listing) == 2
        assert listing[0]["id"] == "ckpt_0"
        assert listing[0]["cell_count"] == 2
        assert listing[1]["id"] == "ckpt_1"
        assert listing[1]["cell_count"] == 3

    def test_snapshot_is_defensive_copy(self, session):
        """Mutating the input list after save should not affect the checkpoint."""
        cells = [{"label": "@A", "cell_type": "code", "source": "x = 1"}]
        cp_id = session.save_checkpoint(cells)
        cells[0]["source"] = "MUTATED"
        assert session.get_checkpoint(cp_id)[0]["source"] == "x = 1"

    def test_empty_checkpoint(self, session):
        cp_id = session.save_checkpoint([])
        assert session.get_checkpoint(cp_id) == []
        assert session.list_checkpoints()[0]["cell_count"] == 0


# ------------------------------------------------------------------
# Event log tests
# ------------------------------------------------------------------

class TestEventLog:
    def test_log_basic_event(self, session):
        session.log_event("run_cell", {"cell": "@A"}, "ok", 150.5)
        log = session.get_log()
        assert len(log) == 1
        assert log[0]["tool"] == "run_cell"
        assert log[0]["args"] == {"cell": "@A"}
        assert log[0]["result"] == "ok"
        assert log[0]["duration_ms"] == 150.5
        assert log[0]["error"] is None
        assert log[0]["seq"] == 1

    def test_log_with_error(self, session):
        session.log_event("alpha_rename", {"cell": "@B"}, None, 30.0, error="NameError: x")
        log = session.get_log()
        assert log[0]["error"] == "NameError: x"
        assert log[0]["result"] is None

    def test_sequential_numbering(self, session):
        session.log_event("tool_a", {}, "r1", 10.0)
        session.log_event("tool_b", {}, "r2", 20.0)
        session.log_event("tool_c", {}, "r3", 30.0)
        log = session.get_log()
        assert [e["seq"] for e in log] == [1, 2, 3]

    def test_result_truncation(self, session):
        long_result = "x" * 5000
        session.log_event("test", {}, long_result, 1.0)
        assert len(session.get_log()[0]["result"]) == 2000

    def test_none_result_not_truncated(self, session):
        session.log_event("test", {}, None, 1.0)
        assert session.get_log()[0]["result"] is None

    def test_short_result_not_truncated(self, session):
        session.log_event("test", {}, "short", 1.0)
        assert session.get_log()[0]["result"] == "short"

    def test_timestamps_present(self, session):
        session.log_event("test", {}, "ok", 1.0)
        entry = session.get_log()[0]
        assert "timestamp" in entry
        assert "relative_time_s" in entry
        assert isinstance(entry["relative_time_s"], float)

    def test_get_log_returns_copy(self, session):
        session.log_event("test", {}, "ok", 1.0)
        log = session.get_log()
        log.clear()
        assert len(session.get_log()) == 1


# ------------------------------------------------------------------
# format_log tests
# ------------------------------------------------------------------

class TestFormatLog:
    def test_empty_log(self, session):
        assert session.format_log() == ""

    def test_basic_format(self, session):
        session.log_event("run_cell", {"cell": "@A"}, "ok", 150.5)
        formatted = session.format_log()
        assert "[001]" in formatted
        assert "run_cell" in formatted
        assert 'cell="@A"' in formatted
        assert "-> 150ms" in formatted

    def test_error_in_format(self, session):
        session.log_event("bad_tool", {}, None, 5.0, error="SomeError")
        formatted = session.format_log()
        assert "ERROR: SomeError" in formatted

    def test_no_args_format(self, session):
        session.log_event("get_status", {}, "ok", 5.0)
        formatted = session.format_log()
        assert "get_status()" in formatted

    def test_numeric_arg_format(self, session):
        session.log_event("test", {"count": 42}, "ok", 1.0)
        formatted = session.format_log()
        assert "count=42" in formatted


# ------------------------------------------------------------------
# save_log_to_file tests
# ------------------------------------------------------------------

class TestSaveLog:
    def test_save_and_load(self, session):
        session.log_event("tool_a", {"x": 1}, "result_a", 10.0)
        session.log_event("tool_b", {}, "result_b", 20.0, error="oops")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            returned_path = session.save_log_to_file(path)
            assert returned_path == path
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["tool"] == "tool_a"
            assert data[1]["error"] == "oops"
        finally:
            os.unlink(path)

    def test_save_empty_log(self, session):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            session.save_log_to_file(path)
            with open(path) as f:
                data = json.load(f)
            assert data == []
        finally:
            os.unlink(path)
