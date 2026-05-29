"""Tests for the read-only inspection tools.

These tools have no side effects, no kernel access, and operate purely on the
notebook dict. The tests verify correct projection, truncation, and the
graceful error path for unknown cells / unknown tools.
"""

import pytest

from flowbook.server.fix_tools_readonly import (
    TOOL_SCHEMAS,
    ToolError,
    dispatch,
    get_cell_flowbook_meta,
    get_cell_outputs,
    get_cell_source,
    get_cell_traceback,
    list_cells_summary,
    tool_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _code(id_, src, **extra):
    return {
        "cell_type": "code",
        "id": id_,
        "source": src,
        "metadata": extra.pop("metadata", {}),
        "outputs": extra.pop("outputs", []),
        "execution_count": extra.pop("execution_count", None),
        **extra,
    }


def _nb(*cells, with_markdown=False):
    nb = {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    if with_markdown:
        nb["cells"].insert(
            1,
            {"cell_type": "markdown", "id": "md01", "source": "# Header", "metadata": {}},
        )
    return nb


# ---------------------------------------------------------------------------
# list_cells_summary
# ---------------------------------------------------------------------------

class TestListCellsSummary:
    def test_returns_one_entry_per_code_cell(self):
        nb = _nb(
            _code("a000", "x = 1"),
            _code("b000", "y = 2"),
            _code("c000", "z = x + y"),
        )
        out = list_cells_summary(nb)
        assert len(out) == 3
        assert [c["cell_id"] for c in out] == ["a000", "b000", "c000"]
        assert [c["alpha"] for c in out] == ["A", "B", "C"]

    def test_skips_markdown_cells(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"), with_markdown=True)
        out = list_cells_summary(nb)
        assert [c["alpha"] for c in out] == ["A", "B"]  # B is still 2nd code cell

    def test_violation_flagged(self):
        nb = _nb(_code("a000", "train = pd.concat([train, x])",
                       metadata={"flowbook": {"errors": [{"error_type": "no_read_and_write"}]}}))
        out = list_cells_summary(nb)
        assert out[0]["has_violation"] is True
        assert out[0]["violation_types"] == ["no_read_and_write"]

    def test_long_source_truncated_in_preview(self):
        long_source = "x = " + ("a" * 500)
        nb = _nb(_code("a000", long_source))
        preview = list_cells_summary(nb)[0]["source_preview"]
        assert len(preview) < len(long_source)
        assert "truncated" in preview

    def test_alpha_wraps_past_z(self):
        cells = [_code(f"c{i:03d}", "x") for i in range(27)]
        out = list_cells_summary(_nb(*cells))
        assert out[25]["alpha"] == "Z"
        assert out[26]["alpha"] == "AA"


# ---------------------------------------------------------------------------
# get_cell_source
# ---------------------------------------------------------------------------

class TestGetCellSource:
    def test_returns_full_source(self):
        nb = _nb(_code("a000", "import pandas as pd\ntrain = pd.read_csv('x')"))
        assert "pandas" in get_cell_source(nb, "a000")
        assert "read_csv" in get_cell_source(nb, "a000")

    def test_list_source_is_joined(self):
        cell = {"cell_type": "code", "id": "a000",
                "source": ["import pandas\n", "x = 1"], "metadata": {}}
        nb = {"cells": [cell]}
        assert get_cell_source(nb, "a000") == "import pandas\nx = 1"

    def test_unknown_cell_raises(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(ToolError, match="zzzz"):
            get_cell_source(nb, "zzzz")


# ---------------------------------------------------------------------------
# get_cell_outputs
# ---------------------------------------------------------------------------

class TestGetCellOutputs:
    def test_stream_output(self):
        cell = _code("a000", "print('hi')", outputs=[
            {"output_type": "stream", "name": "stdout", "text": "hi\n"},
        ])
        out = get_cell_outputs(_nb(cell), "a000")
        assert out == [{"kind": "stream", "name": "stdout", "text": "hi\n"}]

    def test_execute_result(self):
        cell = _code("a000", "df.head()", outputs=[
            {"output_type": "execute_result", "data": {"text/plain": "   a  b\n0  1  2"}},
        ])
        out = get_cell_outputs(_nb(cell), "a000")
        assert out[0]["kind"] == "execute_result"
        assert "0  1  2" in out[0]["text"]

    def test_error_output(self):
        cell = _code("a000", "1/0", outputs=[
            {"output_type": "error", "ename": "ZeroDivisionError",
             "evalue": "division by zero",
             "traceback": ["Traceback (most recent call last):", "ZeroDivisionError: division by zero"]},
        ])
        out = get_cell_outputs(_nb(cell), "a000")
        assert out[0]["kind"] == "error"
        assert out[0]["ename"] == "ZeroDivisionError"
        assert "ZeroDivisionError" in out[0]["traceback"]

    def test_truncates_long_text(self):
        big = "x" * 10000
        cell = _code("a000", "x", outputs=[
            {"output_type": "stream", "name": "stdout", "text": big},
        ])
        out = get_cell_outputs(_nb(cell), "a000", max_chars=500)
        assert len(out[0]["text"]) <= 500
        assert "truncated" in out[0]["text"]

    def test_unknown_cell_raises(self):
        with pytest.raises(ToolError):
            get_cell_outputs(_nb(), "zzzz")


# ---------------------------------------------------------------------------
# get_cell_flowbook_meta
# ---------------------------------------------------------------------------

class TestFlowbookMeta:
    def test_returns_projected_fields(self):
        cell = _code("a000", "x = 1", metadata={
            "flowbook": {
                "read_locs": [], "write_locs": [{"type": "var", "name": "x"}],
                "errors": [], "execution_seq": 3,
                "junk_field_we_ignore": [1, 2, 3],
            }
        })
        meta = get_cell_flowbook_meta(_nb(cell), "a000")
        assert "write_locs" in meta
        assert "execution_seq" in meta
        assert "junk_field_we_ignore" not in meta

    def test_returns_empty_when_no_flowbook(self):
        cell = _code("a000", "x = 1")
        assert get_cell_flowbook_meta(_nb(cell), "a000") == {}


# ---------------------------------------------------------------------------
# get_cell_traceback
# ---------------------------------------------------------------------------

class TestTraceback:
    def test_extracts_error(self):
        cell = _code("a000", "1/0", outputs=[
            {"output_type": "error", "ename": "ZeroDivisionError",
             "evalue": "division by zero", "traceback": ["line1", "line2"]},
        ])
        tb = get_cell_traceback(_nb(cell), "a000")
        assert tb is not None
        assert tb["ename"] == "ZeroDivisionError"
        assert "line1" in tb["traceback"]

    def test_returns_none_for_clean_cell(self):
        cell = _code("a000", "x = 1", outputs=[
            {"output_type": "stream", "name": "stdout", "text": "ok\n"},
        ])
        assert get_cell_traceback(_nb(cell), "a000") is None


# ---------------------------------------------------------------------------
# dispatch + schemas
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_all_schemas_have_matching_dispatch(self):
        schema_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        assert schema_names == set(tool_names())

    def test_dispatch_invokes_function(self):
        nb = _nb(_code("a000", "x = 1"))
        result = dispatch(nb, "get_cell_source", {"cell_id": "a000"})
        assert result == "x = 1"

    def test_dispatch_with_no_args(self):
        nb = _nb(_code("a000", "x = 1"))
        result = dispatch(nb, "list_cells_summary", {})
        assert len(result) == 1

    def test_dispatch_unknown_tool_raises(self):
        with pytest.raises(ToolError, match="Unknown"):
            dispatch(_nb(), "format_disk", {})

    def test_dispatch_propagates_tool_error(self):
        with pytest.raises(ToolError, match="zzzz"):
            dispatch(_nb(), "get_cell_source", {"cell_id": "zzzz"})

    def test_dispatch_bad_args_becomes_tool_error(self):
        with pytest.raises(ToolError, match="Bad args"):
            dispatch(_nb(), "get_cell_source", {"wrong_key": "x"})
