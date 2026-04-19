"""Tests for flowbook.tools.format — shared formatting functions."""

import pytest

from flowbook.tools.format import (
    format_loc,
    format_loc_list,
    format_error,
    format_staleness_reasons,
    format_flowbook_meta,
    format_outputs_text,
    format_run_result,
    format_violation_line,
    format_rename_result,
    format_status,
)


# ==================================================================
# format_loc
# ==================================================================

class TestFormatLoc:
    def test_var(self):
        assert format_loc({"type": "var", "name": "df"}) == "df"

    def test_col_with_qualifier(self):
        assert format_loc({"type": "col", "name": "df", "qualifier": "price"}) == "df.price"

    def test_col_without_qualifier(self):
        assert format_loc({"type": "col", "name": "df"}) == "df"

    def test_struct_with_qualifier(self):
        assert format_loc({"type": "struct", "name": "df", "qualifier": "columns"}) == "df[columns]"

    def test_pre_formatted_string(self):
        assert format_loc("df['age']") == "df['age']"

    def test_unknown_type_falls_back_to_name(self):
        assert format_loc({"type": "file", "name": "data.csv"}) == "data.csv"

    def test_missing_name(self):
        assert format_loc({"type": "var"}) == "?"


# ==================================================================
# format_loc_list
# ==================================================================

class TestFormatLocList:
    def test_empty(self):
        assert format_loc_list([]) == "(none)"

    def test_single(self):
        assert format_loc_list([{"type": "var", "name": "x"}]) == "x"

    def test_multiple(self):
        locs = [
            {"type": "var", "name": "x"},
            {"type": "col", "name": "df", "qualifier": "a"},
        ]
        assert format_loc_list(locs) == "x, df.a"


# ==================================================================
# format_error
# ==================================================================

class TestFormatError:
    def test_basic(self):
        err = {"error_type": "NO_READ_AND_WRITE", "message": "reads and writes x"}
        result = format_error(err)
        assert "NO_READ_AND_WRITE" in result
        assert "reads and writes x" in result

    def test_with_locations(self):
        err = {
            "error_type": "NO_WRITE_AFTER_READ",
            "message": "msg",
            "locations": [{"type": "var", "name": "df"}],
        }
        result = format_error(err)
        assert "Locations: df" in result

    def test_with_causer(self):
        err = {
            "error_type": "NO_WRITE_AFTER_READ",
            "message": "msg",
            "causer_cell": "ab12",
        }
        result = format_error(err)
        assert "Causer cell: ab12" in result


# ==================================================================
# format_staleness_reasons
# ==================================================================

class TestFormatStalenessReasons:
    def test_empty(self):
        assert format_staleness_reasons({}) == "(none)"

    def test_single_reason(self):
        reasons = {
            "ab12": [{"type": "forward_stale", "loc": "x", "cell_id": "cd34"}]
        }
        result = format_staleness_reasons(reasons)
        assert "ab12" in result
        assert "forward_stale: x (from cell cd34)" in result

    def test_multiple_reasons(self):
        reasons = {
            "ab12": [
                {"type": "forward_stale", "loc": "x"},
                {"type": "backward_stale", "loc": "y"},
            ]
        }
        result = format_staleness_reasons(reasons)
        assert "forward_stale: x" in result
        assert "backward_stale: y" in result


# ==================================================================
# format_flowbook_meta
# ==================================================================

class TestFormatFlowbookMeta:
    def test_minimal(self):
        meta = {"read_locs": [], "write_locs": [], "errors": [], "stale_cells": []}
        result = format_flowbook_meta(meta)
        assert "Reads: (none)" in result
        assert "Writes: (none)" in result
        assert "Errors: (none)" in result
        assert "Stale cells: (none)" in result

    def test_with_reads_writes(self):
        meta = {
            "read_locs": [{"type": "var", "name": "x"}],
            "write_locs": [{"type": "var", "name": "y"}],
            "errors": [],
            "stale_cells": [],
        }
        result = format_flowbook_meta(meta)
        assert "Reads: x" in result
        assert "Writes: y" in result

    def test_with_errors(self):
        meta = {
            "read_locs": [],
            "write_locs": [],
            "errors": [{"error_type": "NO_READ_AND_WRITE", "message": "bad"}],
            "stale_cells": [],
        }
        result = format_flowbook_meta(meta)
        assert "NO_READ_AND_WRITE" in result

    def test_with_timing(self):
        meta = {
            "read_locs": [],
            "write_locs": [],
            "errors": [],
            "stale_cells": [],
            "execute_duration_ms": 150.5,
            "code_duration_ms": 100.2,
        }
        result = format_flowbook_meta(meta)
        assert "total=150ms" in result or "total=151ms" in result
        assert "code=100ms" in result

    def test_with_changed_locs(self):
        meta = {
            "read_locs": [],
            "write_locs": [],
            "changed_locs": [{"type": "var", "name": "z"}],
            "errors": [],
            "stale_cells": [],
        }
        result = format_flowbook_meta(meta)
        assert "Changed: z" in result

    def test_with_stale_cells_and_reasons(self):
        meta = {
            "read_locs": [],
            "write_locs": [],
            "errors": [],
            "stale_cells": ["ab12"],
            "staleness_reasons": {
                "ab12": [{"type": "forward_stale", "loc": "x"}]
            },
        }
        result = format_flowbook_meta(meta)
        assert "Stale cells: ab12" in result
        assert "forward_stale: x" in result


# ==================================================================
# format_outputs_text
# ==================================================================

class TestFormatOutputsText:
    def test_empty(self):
        assert format_outputs_text([]) == "(no output)"

    def test_stream(self):
        outputs = [{"output_type": "stream", "text": "hello\n"}]
        assert format_outputs_text(outputs) == "hello\n"

    def test_stream_list(self):
        outputs = [{"output_type": "stream", "text": ["hello", " world"]}]
        assert format_outputs_text(outputs) == "hello world"

    def test_execute_result(self):
        outputs = [{"output_type": "execute_result", "data": {"text/plain": "42"}}]
        assert format_outputs_text(outputs) == "42"

    def test_display_data_text(self):
        outputs = [{"output_type": "display_data", "data": {"text/plain": "fig"}}]
        assert format_outputs_text(outputs) == "fig"

    def test_display_data_html(self):
        outputs = [{"output_type": "display_data", "data": {"text/html": "<b>hi</b>"}}]
        result = format_outputs_text(outputs)
        assert "text/html" in result
        assert "get_cell_outputs" in result

    def test_display_data_image(self):
        outputs = [{"output_type": "display_data", "data": {"image/png": "base64..."}}]
        result = format_outputs_text(outputs)
        assert "image/png" in result
        assert "get_cell_outputs" in result

    def test_display_data_html_with_label_includes_label(self):
        outputs = [{"output_type": "display_data", "data": {"text/html": "<b>hi</b>"}}]
        result = format_outputs_text(outputs, cell_label="@C")
        assert 'get_cell_outputs(["@C"])' in result

    def test_display_data_text_and_image_both_rendered(self):
        outputs = [{"output_type": "display_data", "data": {
            "text/plain": "<Figure>", "image/png": "base64..."
        }}]
        result = format_outputs_text(outputs)
        # text/plain passes through
        assert "<Figure>" in result
        # image gets a marker
        assert "image/png" in result
        assert "get_cell_outputs" in result

    def test_error(self):
        outputs = [{"output_type": "error", "ename": "ValueError", "evalue": "bad"}]
        assert format_outputs_text(outputs) == "ValueError: bad"

    def test_multiple_outputs(self):
        outputs = [
            {"output_type": "stream", "text": "line1\n"},
            {"output_type": "execute_result", "data": {"text/plain": "42"}},
        ]
        result = format_outputs_text(outputs)
        assert "line1" in result
        assert "42" in result


# ==================================================================
# format_run_result
# ==================================================================

class TestFormatRunResult:
    def test_ok(self):
        result = {"cell_id": "ab12", "status": "ok", "outputs_text": ""}
        text = format_run_result(result, "@A")
        assert "@A [ab12]: ok" in text

    def test_with_error(self):
        result = {"cell_id": "ab12", "status": "error", "error_message": "NameError: x", "outputs_text": ""}
        text = format_run_result(result, "@A")
        assert "error" in text
        assert "NameError: x" in text

    def test_with_output(self):
        result = {"cell_id": "ab12", "status": "ok", "outputs_text": "hello world"}
        text = format_run_result(result, "@A")
        assert "Output: hello world" in text

    def test_output_truncation(self):
        result = {"cell_id": "ab12", "status": "ok", "outputs_text": "x" * 500}
        text = format_run_result(result, "@A", output_preview_len=100)
        assert "..." in text

    def test_with_meta(self):
        result = {"cell_id": "ab12", "status": "ok", "outputs_text": ""}
        meta = {"read_locs": [{"type": "var", "name": "x"}], "write_locs": [], "errors": [], "stale_cells": []}
        text = format_run_result(result, "@A", meta=meta)
        assert "Reads: x" in text

    def test_with_flowbook_key(self):
        result = {"cell_id": "ab12", "status": "ok", "outputs_text": "", "flowbook": "Reads: y"}
        text = format_run_result(result, "@A")
        assert "Reads: y" in text


# ==================================================================
# format_violation_line
# ==================================================================

class TestFormatViolationLine:
    def test_basic(self):
        err = {"error_type": "NO_READ_AND_WRITE", "locations": []}
        text = format_violation_line(err, "@A", "ab12")
        assert "@A [ab12]: NO_READ_AND_WRITE" in text

    def test_with_locations(self):
        err = {"error_type": "NO_WRITE_AFTER_READ", "locations": [{"type": "var", "name": "x"}]}
        text = format_violation_line(err, "@B", "cd34")
        assert "[x]" in text


# ==================================================================
# format_rename_result
# ==================================================================

class TestFormatRenameResult:
    def test_no_occurrences(self):
        text = format_rename_result("old", "new", [], "@A")
        assert "No occurrences" in text
        assert "@A" in text

    def test_with_modifications(self):
        text = format_rename_result("old", "new", ["@A [ab12]", "@C [ef56]"], "@A")
        assert "Renamed 'old'" in text
        assert "'new'" in text
        assert "2 cells" in text


# ==================================================================
# format_status
# ==================================================================

class TestFormatStatus:
    def test_all_clean(self):
        status = {"executed": 3, "total_code_cells": 3, "violations": [], "stale_cells": {}}
        text = format_status(status, lambda cid: f"@{cid}")
        assert "3/3 executed" in text
        assert "0 violations" in text
        assert "0 stale" in text

    def test_with_violations(self):
        status = {
            "executed": 2,
            "total_code_cells": 3,
            "violations": [{"error_type": "NO_READ_AND_WRITE", "cell_id": "ab", "locations": []}],
            "stale_cells": {},
        }
        text = format_status(status, lambda cid: f"@{cid}")
        assert "1 violations" in text
        assert "NO_READ_AND_WRITE" in text

    def test_with_stale(self):
        status = {
            "executed": 2,
            "total_code_cells": 3,
            "violations": [],
            "stale_cells": {"cd": [{"type": "forward_stale", "loc": "x"}]},
        }
        text = format_status(status, lambda cid: f"@{cid}")
        assert "1 stale" in text
        assert "forward_stale: x" in text
