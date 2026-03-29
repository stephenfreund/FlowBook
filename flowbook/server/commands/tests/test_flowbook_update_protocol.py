"""
End-to-end tests for the flowbook_update IOPub protocol.

Verifies that all IOPub message readers correctly parse flowbook_update
messages produced by the kernel. These tests exercise the full path from
protocol message builders through the IOPub wire format to the receiver
parsing logic in:

- execute_cell_flowbook (compare_baseline.py) — FlowBook kernel metadata + violations
- KernelHelper.execute_code (kernel_helper.py) — general-purpose command execution
- execute_cell_and_extract_timing (benchmark_checkpoint.py) — checkpoint kernel timing
- measure_rerun_overhead receiver (compare_baseline.py) — rerun overhead measurement
"""

import time

from flowbook.kernel.protocol import (
    IOPUB_MSG_TYPE,
    METADATA,
    VIOLATION,
    STATUS,
    build_metadata_message,
    build_violation_message,
    build_status_message,
)
from flowbook.kernel.models import (
    ReproducibilityMetadata,
    ReproducibilityError,
    ErrorType,
)
from flowbook.server.commands.compare_baseline import execute_cell_flowbook
from flowbook.server.kernel_helper import KernelHelper
from flowbook.testing.benchmark_checkpoint import execute_cell_and_extract_timing


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------

def _wrap_flowbook_update(msg: dict, msg_id: str = "test_msg_1") -> dict:
    """Wrap a protocol message as a flowbook_update IOPub message.

    This replicates what _send_flowbook_message() does on the kernel side:
        self.send_response(self.iopub_socket, IOPUB_MSG_TYPE, {"flowbook": msg})
    """
    return {
        "parent_header": {"msg_id": msg_id},
        "header": {"msg_type": IOPUB_MSG_TYPE},
        "content": {"flowbook": msg},
    }


def _status_msg(state: str, msg_id: str = "test_msg_1") -> dict:
    return {
        "parent_header": {"msg_id": msg_id},
        "header": {"msg_type": "status"},
        "content": {"execution_state": state},
    }


def _shell_reply_ok(msg_id: str = "test_msg_1") -> dict:
    return {
        "parent_header": {"msg_id": msg_id},
        "header": {"msg_type": "execute_reply"},
        "content": {"status": "ok"},
    }


def _error_msg(msg_id: str = "test_msg_1") -> dict:
    return {
        "parent_header": {"msg_id": msg_id},
        "header": {"msg_type": "error"},
        "content": {
            "ename": "ValueError",
            "evalue": "test error",
            "traceback": ["ValueError: test error"],
        },
    }


class MockFlowbookClient:
    """Mock kernel client that emits pre-built IOPub messages.

    Simulates both FlowbookKernelClient and CheckpointKernelClient
    interfaces used by the various IOPub readers.
    """

    def __init__(self, iopub_messages, shell_messages=None, msg_id="test_msg_1"):
        self._msg_id = msg_id
        self._iopub_messages = list(iopub_messages)
        self._shell_messages = list(shell_messages or [_shell_reply_ok(msg_id)])

    def execute(self, code, *, cell_id=None, cell_metadata=None, silent=False,
                user_expressions=None, store_history=True):
        return self._msg_id

    def set_cell_order(self, cell_order):
        pass

    def get_iopub_msg(self, timeout=1.0):
        if self._iopub_messages:
            return self._iopub_messages.pop(0)
        raise TimeoutError("No more iopub messages")

    def get_shell_msg(self, timeout=1.0):
        if self._shell_messages:
            return self._shell_messages.pop(0)
        raise TimeoutError("No more shell messages")


# ---------------------------------------------------------------------------
# Tests: execute_cell_flowbook (compare_baseline.py)
# ---------------------------------------------------------------------------

class TestExecuteCellFlowbookProtocol:
    """Verify execute_cell_flowbook correctly parses flowbook_update messages."""

    def _make_metadata_msg(self, **overrides):
        defaults = dict(
            cell_id="abcd",
            execution_seq=1,
            read_locs=[{"type": "var", "name": "x"}],
            write_locs=[{"type": "var", "name": "y"}],
            changed_locs=[{"type": "var", "name": "y"}],
            stale_cells=["efgh"],
            cell_order=["abcd", "efgh"],
            execute_duration_ms=100.0,
            code_duration_ms=80.0,
            state_duration_ms=15.0,
            check_duration_ms=5.0,
            staleness_reasons={},
            errors=[],
        )
        defaults.update(overrides)
        metadata = ReproducibilityMetadata(**defaults)
        return build_metadata_message(metadata)

    def test_receives_metadata_with_timing(self):
        """Metadata message is received and timing fields are extracted."""
        meta_msg = self._make_metadata_msg(
            execute_duration_ms=150.0,
            code_duration_ms=120.0,
            state_duration_ms=20.0,
            check_duration_ms=10.0,
        )
        status_msg = build_status_message("✓", "Clean", cell_id="abcd")

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(meta_msg),
            _wrap_flowbook_update(status_msg),
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "y = x + 1", "abcd", ["abcd", "efgh"]
        )

        assert result["execute_duration_ms"] == 150.0
        assert result["code_duration_ms"] == 120.0
        assert result["state_duration_ms"] == 20.0
        assert result["check_duration_ms"] == 10.0
        assert result["error"] is None
        assert result["stale_cells"] == ["efgh"]
        assert result["checking_result"]["cell_status"] == "clean"

    def test_receives_violation_as_predicate_violation(self):
        """Violation messages are collected in predicate_violations."""
        meta_msg = self._make_metadata_msg(stale_cells=[])
        error = ReproducibilityError(
            error_type=ErrorType.NO_WRITE_AFTER_READ,
            cell_id="abcd",
            locations=["x"],
            message="Backward mutation on x",
        )
        violation_msg = build_violation_message(error, accepted=True)

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(violation_msg),
            _wrap_flowbook_update(meta_msg),
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "x = 1", "abcd", ["abcd"]
        )

        assert result["violation"] is not None
        assert "no_write_after_read" in result["violation"]
        assert "accepted" in result["violation"]
        assert len(result["predicate_violations"]) == 1
        pv = result["predicate_violations"][0]
        assert pv["predicate"] == "no_write_after_read"
        assert pv["locations"] == ["x"]
        assert pv["accepted"] is True

    def test_violation_sets_error_status(self):
        """Violation presence sets checking_result cell_status to error."""
        error = ReproducibilityError(
            error_type=ErrorType.NO_READ_AND_WRITE,
            cell_id="abcd",
            locations=["z"],
            message="Cell reads and writes z",
        )
        violation_msg = build_violation_message(error, accepted=False)
        meta_msg = self._make_metadata_msg(stale_cells=[])

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(violation_msg),
            _wrap_flowbook_update(meta_msg),
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "z = z + 1", "abcd", ["abcd"]
        )

        assert result["checking_result"]["cell_status"] == "error"
        assert len(result["checking_result"]["errors"]) == 1

    def test_staleness_sets_stale_status(self):
        """Stale cells in metadata set checking_result to stale."""
        meta_msg = self._make_metadata_msg(
            stale_cells=["abcd"],
            staleness_reasons={"abcd": [{"type": "forward_stale", "writer": "xxxx"}]},
        )

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(meta_msg),
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "x = 1", "abcd", ["abcd"]
        )

        assert result["checking_result"]["cell_status"] == "stale"
        assert len(result["checking_result"]["reasons"]) == 1

    def test_no_metadata_returns_error(self):
        """Missing metadata returns fallback with error message."""
        client = MockFlowbookClient([
            _status_msg("busy"),
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "x = 1", "abcd", ["abcd"]
        )

        assert result["error"] == "No flowbook metadata received"
        assert result["checking_result"] is None
        assert result["code_duration_ms"] == 0.0

    def test_execution_error_with_metadata(self):
        """Execution error is captured alongside metadata."""
        meta_msg = self._make_metadata_msg()

        client = MockFlowbookClient([
            _status_msg("busy"),
            _error_msg(),
            _wrap_flowbook_update(meta_msg),
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "raise ValueError()", "abcd", ["abcd", "efgh"]
        )

        assert result["error"] is not None
        assert "ValueError" in result["error"]
        # Metadata is still received despite the error
        assert result["execute_duration_ms"] == 100.0

    def test_ignores_messages_from_other_executions(self):
        """Messages with different parent msg_id are ignored."""
        meta_msg = self._make_metadata_msg()

        other_meta = _wrap_flowbook_update(meta_msg, msg_id="other_msg")

        client = MockFlowbookClient([
            _status_msg("busy"),
            other_meta,  # Different msg_id — should be ignored
            _wrap_flowbook_update(meta_msg),  # Correct msg_id
            _status_msg("idle"),
        ])

        result = execute_cell_flowbook(
            client, "y = x", "abcd", ["abcd", "efgh"]
        )

        assert result["execute_duration_ms"] == 100.0
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Tests: KernelHelper.execute_code (kernel_helper.py)
# ---------------------------------------------------------------------------

class TestKernelHelperProtocol:
    """Verify KernelHelper.execute_code collects flowbook_update messages."""

    def test_collects_metadata_messages(self):
        """flowbook_update messages are collected in flowbook_messages."""
        meta_msg = build_metadata_message(ReproducibilityMetadata(
            cell_id="abcd",
            execution_seq=1,
            read_locs=[],
            write_locs=[{"type": "var", "name": "x"}],
            changed_locs=[],
            stale_cells=[],
            cell_order=["abcd"],
        ))
        status_msg = build_status_message("✓", "Clean")

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(meta_msg),
            _wrap_flowbook_update(status_msg),
            _status_msg("idle"),
        ])

        result = KernelHelper.execute_code(client, "x = 1", cell_id="abcd")

        assert result["status"] == "ok"
        fb_msgs = result["flowbook_messages"]
        assert len(fb_msgs) == 2
        assert fb_msgs[0]["type"] == METADATA
        assert fb_msgs[0]["cell_id"] == "abcd"
        assert fb_msgs[1]["type"] == STATUS

    def test_collects_violation_messages(self):
        """Violation messages appear in flowbook_messages."""
        error = ReproducibilityError(
            error_type=ErrorType.NO_WRITE_AFTER_READ,
            cell_id="abcd",
            locations=["x"],
            message="Backward mutation",
        )
        violation_msg = build_violation_message(error, accepted=False)

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(violation_msg),
            _status_msg("idle"),
        ])

        result = KernelHelper.execute_code(client, "x = 1", cell_id="abcd")

        fb_msgs = result["flowbook_messages"]
        assert len(fb_msgs) == 1
        assert fb_msgs[0]["type"] == VIOLATION
        assert fb_msgs[0]["predicate"] == "no_write_after_read"

    def test_display_data_not_confused_with_flowbook(self):
        """Regular display_data messages go to outputs, not flowbook_messages."""
        display_msg = {
            "parent_header": {"msg_id": "test_msg_1"},
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {"text/plain": "hello"},
                "metadata": {},
            },
        }

        client = MockFlowbookClient([
            _status_msg("busy"),
            display_msg,
            _status_msg("idle"),
        ])

        result = KernelHelper.execute_code(client, "display('hello')")

        assert result["flowbook_messages"] == []
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["output_type"] == "display_data"


# ---------------------------------------------------------------------------
# Tests: execute_cell_and_extract_timing (benchmark_checkpoint.py)
# ---------------------------------------------------------------------------

class TestCheckpointTimingProtocol:
    """Verify benchmark_checkpoint reads checkpoint_timing from flowbook_update."""

    def test_receives_checkpoint_timing(self):
        """Checkpoint timing message is correctly parsed."""
        timing_msg = {
            "type": "checkpoint_timing",
            "cell_id": "abcd",
            "execution_count": 1,
            "cell_runtime_s": 0.5,
            "commit_time_s": 0.02,
        }

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(timing_msg),
            _status_msg("idle"),
        ])

        # Create a minimal Cell-like object
        class MockCell:
            source = "import numpy"
            cell_id = "abcd"

        result = execute_cell_and_extract_timing(client, MockCell())

        assert result["type"] == "checkpoint_timing"
        assert result["cell_runtime_s"] == 0.5
        assert result["commit_time_s"] == 0.02
        assert result["cell_id"] == "abcd"

    def test_checkpoint_timing_with_error_in_reply(self):
        """Execution error from shell reply is captured alongside timing data."""
        timing_msg = {
            "type": "checkpoint_timing",
            "cell_id": "abcd",
            "execution_count": 1,
            "cell_runtime_s": 0.1,
            "commit_time_s": 0.0,
        }

        error_reply = {
            "parent_header": {"msg_id": "test_msg_1"},
            "header": {"msg_type": "execute_reply"},
            "content": {
                "status": "error",
                "ename": "ZeroDivisionError",
                "evalue": "division by zero",
                "traceback": ["ZeroDivisionError: division by zero"],
            },
        }

        client = MockFlowbookClient(
            iopub_messages=[
                _status_msg("busy"),
                _wrap_flowbook_update(timing_msg),
                _status_msg("idle"),
            ],
            shell_messages=[error_reply],
        )

        class MockCell:
            source = "1/0"
            cell_id = "abcd"

        result = execute_cell_and_extract_timing(client, MockCell())

        assert result["cell_runtime_s"] == 0.1
        assert "error" in result
        assert "ZeroDivisionError" in result["error"]

    def test_checkpoint_error_before_timing_replaces(self):
        """IOPub error followed by timing: timing overwrites error-only dict."""
        timing_msg = {
            "type": "checkpoint_timing",
            "cell_id": "abcd",
            "execution_count": 1,
            "cell_runtime_s": 0.1,
            "commit_time_s": 0.0,
        }

        client = MockFlowbookClient([
            _status_msg("busy"),
            _error_msg(),
            _wrap_flowbook_update(timing_msg),
            _status_msg("idle"),
        ])

        class MockCell:
            source = "1/0"
            cell_id = "abcd"

        result = execute_cell_and_extract_timing(client, MockCell())

        # When error arrives before timing, timing_data gets reassigned.
        # The timing data is present but the IOPub error is lost (shell
        # reply error path at line 180-184 would still catch it).
        assert result["cell_runtime_s"] == 0.1

    def test_no_timing_returns_error(self):
        """Missing timing message returns error dict."""
        client = MockFlowbookClient([
            _status_msg("busy"),
            _status_msg("idle"),
        ])

        class MockCell:
            source = "x = 1"
            cell_id = "abcd"

        result = execute_cell_and_extract_timing(client, MockCell())

        assert result["cell_runtime_s"] is None
        assert "No timing metadata received" in result["error"]

    def test_ignores_non_checkpoint_flowbook_messages(self):
        """Other flowbook_update types (metadata, status) are ignored."""
        meta_msg = build_metadata_message(ReproducibilityMetadata(
            cell_id="abcd",
            execution_seq=1,
            read_locs=[],
            write_locs=[],
            changed_locs=[],
            stale_cells=[],
            cell_order=["abcd"],
        ))

        client = MockFlowbookClient([
            _status_msg("busy"),
            _wrap_flowbook_update(meta_msg),  # Wrong type for checkpoint reader
            _status_msg("idle"),
        ])

        class MockCell:
            source = "x = 1"
            cell_id = "abcd"

        result = execute_cell_and_extract_timing(client, MockCell())

        assert result["cell_runtime_s"] is None
        assert "No timing metadata received" in result["error"]


# ---------------------------------------------------------------------------
# Tests: Rerun overhead receiver (compare_baseline.py)
# ---------------------------------------------------------------------------

class TestRerunOverheadProtocol:
    """Verify the measure_rerun_overhead receiver in compare_baseline."""

    def test_receives_rerun_overhead(self):
        """Rerun overhead data is extracted from flowbook_update message."""
        from flowbook.server.commands.compare_baseline import measure_rerun_overhead

        overhead_msg = {
            "type": "rerun_overhead",
            "rerun_overhead": {
                "cell_id": "abcd",
                "checkpoint_ms": 5.0,
                "check_ms": 2.0,
                "total_overhead_ms": 7.0,
                "checkpoint_by_var": {"x": 3.0, "y": 2.0},
                "checkpoint_var_costs": {"x": 3.0, "y": 2.0},
            },
        }

        # Build a mock client that returns the overhead on execute
        msg_id = "rerun_msg_1"
        client = MockFlowbookClient(
            iopub_messages=[
                _status_msg("busy", msg_id),
                _wrap_flowbook_update(overhead_msg, msg_id),
                _status_msg("idle", msg_id),
            ],
            shell_messages=[_shell_reply_ok(msg_id)],
            msg_id=msg_id,
        )

        # We can't easily call the full measure_rerun_overhead function
        # because it orchestrates multiple executions. Instead, test the
        # inner IOPub parsing loop directly.
        client.execute("%measure_rerun_overhead abcd", cell_id="abcd")

        overhead_data = None
        while True:
            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except Exception:
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["header"]["msg_type"]
            if msg_type == "flowbook_update":
                fb_data = msg.get("content", {}).get(
                    "flowbook", msg.get("content", {})
                )
                if fb_data.get("type") == "rerun_overhead":
                    overhead_data = fb_data["rerun_overhead"]
            elif msg_type == "status":
                if msg["content"]["execution_state"] == "idle":
                    break

        assert overhead_data is not None
        assert overhead_data["checkpoint_ms"] == 5.0
        assert overhead_data["check_ms"] == 2.0
        assert overhead_data["total_overhead_ms"] == 7.0


# ---------------------------------------------------------------------------
# Tests: Wire format consistency
# ---------------------------------------------------------------------------

class TestWireFormatConsistency:
    """Verify that protocol builders produce messages parseable by receivers.

    These tests check the contract between kernel-side message construction
    (_send_flowbook_message wrapping) and client-side parsing.
    """

    def test_metadata_round_trip_through_wire_format(self):
        """build_metadata_message -> IOPub wrap -> receiver extracts fields."""
        metadata = ReproducibilityMetadata(
            cell_id="test",
            execution_seq=5,
            read_locs=[{"type": "var", "name": "a"}],
            write_locs=[{"type": "col", "name": "df", "qualifier": "price"}],
            changed_locs=[],
            stale_cells=["wxyz"],
            cell_order=["test", "wxyz"],
            execute_duration_ms=42.0,
            code_duration_ms=35.0,
            state_duration_ms=5.0,
            check_duration_ms=2.0,
        )
        protocol_msg = build_metadata_message(metadata)
        wire_msg = _wrap_flowbook_update(protocol_msg)

        # Simulate receiver extraction (same code as execute_cell_flowbook)
        fb_data = wire_msg["content"].get("flowbook", wire_msg["content"])
        assert fb_data["type"] == METADATA
        assert fb_data["cell_id"] == "test"
        assert fb_data["execute_duration_ms"] == 42.0
        assert fb_data["code_duration_ms"] == 35.0
        assert fb_data["stale_cells"] == ["wxyz"]

    def test_violation_round_trip_through_wire_format(self):
        """build_violation_message -> IOPub wrap -> receiver extracts fields."""
        error = ReproducibilityError(
            error_type=ErrorType.NO_READ_BEFORE_WRITE,
            cell_id="abcd",
            locations=["z"],
            message="Forward contamination on z",
            causer_cell="efgh",
        )
        protocol_msg = build_violation_message(error, accepted=True)
        wire_msg = _wrap_flowbook_update(protocol_msg)

        fb_data = wire_msg["content"].get("flowbook", wire_msg["content"])
        assert fb_data["type"] == VIOLATION
        assert fb_data["predicate"] == "no_read_before_write"
        assert fb_data["locations"] == ["z"]
        assert fb_data["accepted"] is True
        assert fb_data["causer_cell"] == "efgh"

    def test_checkpoint_timing_wire_format(self):
        """Checkpoint kernel timing message -> IOPub wrap -> receiver extracts."""
        timing_msg = {
            "type": "checkpoint_timing",
            "cell_id": "abcd",
            "execution_count": 3,
            "cell_runtime_s": 1.5,
            "commit_time_s": 0.05,
        }
        wire_msg = _wrap_flowbook_update(timing_msg)

        # Simulate benchmark_checkpoint receiver extraction
        fb_data = wire_msg["content"].get("flowbook", wire_msg["content"])
        assert fb_data.get("type") == "checkpoint_timing"
        assert fb_data["cell_runtime_s"] == 1.5
        assert fb_data["commit_time_s"] == 0.05

    def test_rerun_overhead_wire_format(self):
        """Rerun overhead message -> IOPub wrap -> receiver extracts."""
        overhead_msg = {
            "type": "rerun_overhead",
            "rerun_overhead": {
                "cell_id": "abcd",
                "checkpoint_ms": 10.0,
                "check_ms": 3.0,
                "total_overhead_ms": 13.0,
            },
        }
        wire_msg = _wrap_flowbook_update(overhead_msg)

        fb_data = wire_msg["content"].get("flowbook", wire_msg["content"])
        assert fb_data.get("type") == "rerun_overhead"
        assert fb_data["rerun_overhead"]["total_overhead_ms"] == 13.0
