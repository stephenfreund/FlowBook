"""Tests for flowbook.kernel.protocol message builders and validation."""

from flowbook.kernel.protocol import (
    METADATA, VIOLATION, STATUS,
    NOTEBOOK_STRUCTURE, CELL_EDITED, CONTINUE_AFTER_VIOLATION, SYNC, EXEC_RESTORE,
    build_metadata_message,
    build_violation_message,
    build_status_message,
    build_notebook_structure_message,
    build_cell_edited_message,
    build_continue_after_violation_message,
    build_sync_message,
    build_exec_restore_message,
    validate_message,
)
from flowbook.kernel.models import (
    ReproducibilityMetadata,
    ReproducibilityError,
    ErrorType,
)


class TestBuildMetadataMessage:
    def test_basic(self):
        metadata = ReproducibilityMetadata(
            cell_id="abcd",
            execution_seq=3,
            read_locs=[{"type": "var", "name": "x"}],
            write_locs=[{"type": "var", "name": "y"}],
            changed_locs=[{"type": "var", "name": "y"}],
            stale_cells=["efgh"],
            cell_order=["abcd", "efgh"],
        )
        msg = build_metadata_message(metadata)
        assert msg["type"] == METADATA
        assert msg["cell_id"] == "abcd"
        assert msg["execution_seq"] == 3
        assert msg["read_locs"] == [{"type": "var", "name": "x"}]
        assert msg["write_locs"] == [{"type": "var", "name": "y"}]
        assert msg["changed_locs"] == [{"type": "var", "name": "y"}]
        assert msg["stale_cells"] == ["efgh"]
        assert msg["cell_order"] == ["abcd", "efgh"]
        assert msg["staleness_reasons"] == {}
        assert msg["errors"] == []

    def test_with_timing(self):
        metadata = ReproducibilityMetadata(
            cell_id="abcd",
            execution_seq=1,
            read_locs=[],
            write_locs=[],
            changed_locs=[],
            stale_cells=[],
            cell_order=["abcd"],
            execute_duration_ms=100.5,
            code_duration_ms=80.0,
            state_duration_ms=15.0,
            check_duration_ms=5.5,
        )
        msg = build_metadata_message(metadata)
        assert msg["execute_duration_ms"] == 100.5
        assert msg["code_duration_ms"] == 80.0
        assert msg["state_duration_ms"] == 15.0
        assert msg["check_duration_ms"] == 5.5

    def test_round_trip_fields(self):
        """All ReproducibilityMetadata fields appear in the message."""
        metadata = ReproducibilityMetadata(
            cell_id="test",
            execution_seq=1,
            read_locs=[],
            write_locs=[],
            changed_locs=[],
            stale_cells=[],
            cell_order=[],
            structural_warnings=["warn1"],
            staleness_reasons={"abc": [{"type": "forward_stale"}]},
            errors=[{"error_type": "no_read_and_write", "message": "test"}],
        )
        msg = build_metadata_message(metadata)
        assert msg["structural_warnings"] == ["warn1"]
        assert msg["staleness_reasons"] == {"abc": [{"type": "forward_stale"}]}
        assert msg["errors"] == [{"error_type": "no_read_and_write", "message": "test"}]


class TestBuildViolationMessage:
    def test_basic(self):
        error = ReproducibilityError(
            error_type=ErrorType.NO_WRITE_AFTER_READ,
            cell_id="abcd",
            locations=["x", "y"],
            message="Backward mutation on x, y",
        )
        msg = build_violation_message(error, accepted=False)
        assert msg["type"] == VIOLATION
        assert msg["predicate"] == "no_write_after_read"
        assert msg["cell_id"] == "abcd"
        assert msg["locations"] == ["x", "y"]
        assert msg["message"] == "Backward mutation on x, y"
        assert msg["accepted"] is False
        assert "causer_cell" not in msg
        assert "detail" not in msg

    def test_with_causer_and_detail(self):
        error = ReproducibilityError(
            error_type=ErrorType.NO_READ_BEFORE_WRITE,
            cell_id="abcd",
            locations=["z"],
            message="Forward contamination",
            causer_cell="efgh",
            detail={"changes_detail": ["z changed"]},
        )
        msg = build_violation_message(error, accepted=True)
        assert msg["accepted"] is True
        assert msg["causer_cell"] == "efgh"
        assert msg["detail"] == {"changes_detail": ["z changed"]}


class TestBuildStatusMessage:
    def test_basic(self):
        msg = build_status_message("✓", "Clean | 3 stale")
        assert msg["type"] == STATUS
        assert msg["icon"] == "✓"
        assert msg["text"] == "Clean | 3 stale"
        assert msg["cell_id"] == ""

    def test_with_cell_id(self):
        msg = build_status_message("✓", "Execute: 42 ms", cell_id="abcd")
        assert msg["cell_id"] == "abcd"


class TestClientToKernelMessages:
    def test_notebook_structure(self):
        msg = build_notebook_structure_message(["a", "b", "c"])
        assert msg["type"] == NOTEBOOK_STRUCTURE
        assert msg["cell_order"] == ["a", "b", "c"]

    def test_cell_edited(self):
        msg = build_cell_edited_message("abcd")
        assert msg["type"] == CELL_EDITED
        assert msg["cell_id"] == "abcd"

    def test_continue_after_violation(self):
        msg = build_continue_after_violation_message(True)
        assert msg["type"] == CONTINUE_AFTER_VIOLATION
        assert msg["enabled"] is True

    def test_sync(self):
        msg = build_sync_message()
        assert msg["type"] == SYNC

    def test_exec_restore(self):
        msg = build_exec_restore_message("abcd")
        assert msg["type"] == EXEC_RESTORE
        assert msg["cell_id"] == "abcd"


class TestValidateMessage:
    def test_valid_messages(self):
        assert validate_message({"type": METADATA, "cell_id": "abc"})
        assert validate_message({"type": VIOLATION, "predicate": "test"})
        assert validate_message({"type": STATUS, "icon": "✓"})
        assert validate_message({"type": NOTEBOOK_STRUCTURE, "cell_order": []})
        assert validate_message({"type": CELL_EDITED, "cell_id": "abc"})
        assert validate_message({"type": CONTINUE_AFTER_VIOLATION, "enabled": True})
        assert validate_message({"type": SYNC})
        assert validate_message({"type": EXEC_RESTORE, "cell_id": "abc"})

    def test_invalid_messages(self):
        assert not validate_message({})
        assert not validate_message({"type": "unknown"})
        assert not validate_message("not a dict")
        assert not validate_message(None)
