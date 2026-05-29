"""Tests for the mutator tools used by the custom-fix path.

Each tool is exercised against a synthetic notebook. The MutationLog is
inspected to verify that pre-fix snapshots and per-call diff entries land
correctly — both are what the handler relies on to build the
CustomFixResponse and what the frontend uses to drive Undo.
"""

import pytest

from flowbook.server.fix_tools_mutator import (
    TOOL_SCHEMAS,
    MutationLog,
    MutatorError,
    delete_cell,
    dispatch,
    edit_cell_source,
    insert_cell_after,
    mark_diagnostic,
    merge_cells,
    move_cell,
    tool_names,
)


def _code(id_, src):
    return {
        "cell_type": "code",
        "id": id_,
        "source": src,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }


def _nb(*cells):
    return {
        "cells": list(cells),
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


# ---------------------------------------------------------------------------
# edit_cell_source
# ---------------------------------------------------------------------------

class TestEditCellSource:
    def test_replaces_code(self):
        nb = _nb(_code("a000", "x = 1"))
        log = MutationLog()
        result = edit_cell_source(nb, log, "a000", "x = 2")
        assert _src(nb["cells"][0]) == "x = 2"
        assert result["cell_id"] == "a000"
        assert log.pre_fix_sources == {"a000": "x = 1"}
        assert log.entries[0].modified_cells == ["a000"]

    def test_replaces_markdown(self):
        nb = _nb({"cell_type": "markdown", "id": "m001", "source": "# Old", "metadata": {}})
        log = MutationLog()
        edit_cell_source(nb, log, "m001", "# New")
        assert _src(nb["cells"][0]) == "# New"

    def test_rejects_unparseable_code(self):
        nb = _nb(_code("a000", "x = 1"))
        log = MutationLog()
        with pytest.raises(MutatorError, match="not valid Python"):
            edit_cell_source(nb, log, "a000", "x = (")
        # Original source preserved
        assert _src(nb["cells"][0]) == "x = 1"

    def test_unknown_cell_raises(self):
        with pytest.raises(MutatorError, match="No cell"):
            edit_cell_source(_nb(), MutationLog(), "zzzz", "x = 1")

    def test_snapshot_taken_only_once(self):
        nb = _nb(_code("a000", "x = 1"))
        log = MutationLog()
        edit_cell_source(nb, log, "a000", "x = 2")
        edit_cell_source(nb, log, "a000", "x = 3")
        # Pre-fix should reflect the very first state, not the intermediate one
        assert log.pre_fix_sources["a000"] == "x = 1"


# ---------------------------------------------------------------------------
# insert_cell_after
# ---------------------------------------------------------------------------

class TestInsertCellAfter:
    def test_inserts_new_code_cell(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"))
        log = MutationLog()
        result = insert_cell_after(nb, log, "a000", "z = 0")
        new_id = result["new_cell_id"]
        ids = [c["id"] for c in nb["cells"]]
        assert ids == ["a000", new_id, "b000"]
        assert new_id != "a000" and new_id != "b000"
        assert log.entries[0].cells_added == [new_id]

    def test_id_is_unique(self):
        nb = _nb(_code("a000", "x = 1"), _code("a0001", "y = 2"))
        log = MutationLog()
        r = insert_cell_after(nb, log, "a000", "z = 0")
        # First candidate "a0001" is taken; next_insertion_id picks "a0002"
        assert r["new_cell_id"] not in ("a000", "a0001")

    def test_markdown_kind(self):
        nb = _nb(_code("a000", "x = 1"))
        log = MutationLog()
        r = insert_cell_after(nb, log, "a000", "# Notes", kind="markdown")
        new_cell = [c for c in nb["cells"] if c["id"] == r["new_cell_id"]][0]
        assert new_cell["cell_type"] == "markdown"
        assert "execution_count" not in new_cell

    def test_rejects_unparseable_code(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(MutatorError, match="not valid Python"):
            insert_cell_after(nb, MutationLog(), "a000", "x = (")
        # No new cells added on failure
        assert len(nb["cells"]) == 1

    def test_unknown_after_raises(self):
        with pytest.raises(MutatorError, match="not found"):
            insert_cell_after(_nb(_code("a000", "x")), MutationLog(), "zzzz", "y = 0")


# ---------------------------------------------------------------------------
# delete_cell
# ---------------------------------------------------------------------------

class TestDeleteCell:
    def test_removes_cell(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"))
        log = MutationLog()
        result = delete_cell(nb, log, "a000")
        assert [c["id"] for c in nb["cells"]] == ["b000"]
        assert result == {"cell_id": "a000", "removed": True}
        assert log.entries[0].cells_removed == ["a000"]
        # Source captured for undo
        assert log.pre_fix_sources["a000"] == "x = 1"

    def test_unknown_cell_raises(self):
        with pytest.raises(MutatorError):
            delete_cell(_nb(), MutationLog(), "zzzz")


# ---------------------------------------------------------------------------
# merge_cells / move_cell / mark_diagnostic — thin wrappers
# ---------------------------------------------------------------------------

class TestMergeCells:
    def test_merges_and_logs(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"), _code("c000", "z = x + y"))
        log = MutationLog()
        result = merge_cells(nb, log, ["a000", "b000"])
        assert len(nb["cells"]) == 2
        assert "a000" in log.pre_fix_sources
        assert "b000" in log.pre_fix_sources
        assert log.entries[0].cells_removed == ["b000"]
        assert result["merged_into"] == "a000"

    def test_too_few_raises(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(Exception):  # dispatcher's ValueError → MutatorError
            merge_cells(nb, MutationLog(), ["a000"])


class TestMoveCell:
    def test_reorders_and_logs(self):
        nb = _nb(_code("a000", "x"), _code("b000", "y"), _code("c000", "z"))
        log = MutationLog()
        result = move_cell(nb, log, "b000", "c000")
        ids = [c["id"] for c in nb["cells"]]
        assert ids == ["a000", "c000", "b000"]
        assert log.entries[0].tool == "move_cell"
        assert "b000" in result["new_order"]

    def test_unknown_target_raises(self):
        nb = _nb(_code("a000", "x"), _code("b000", "y"))
        with pytest.raises(MutatorError):
            move_cell(nb, MutationLog(), "b000", "zzzz")


class TestMarkDiagnostic:
    def test_marks_and_logs(self):
        nb = _nb(_code("a000", "df.info()"))
        log = MutationLog()
        mark_diagnostic(nb, log, "a000")
        assert _src(nb["cells"][0]).startswith("%diagnostic")
        assert "a000" in log.pre_fix_sources
        assert log.entries[0].modified_cells == ["a000"]

    def test_already_diagnostic_raises(self):
        nb = _nb(_code("a000", "%diagnostic\ndf.info()"))
        with pytest.raises(MutatorError):
            mark_diagnostic(nb, MutationLog(), "a000")


# ---------------------------------------------------------------------------
# dispatch + schema parity
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_schemas_match_dispatch_table(self):
        names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        assert names == set(tool_names())

    def test_dispatch_invokes_named_tool(self):
        nb = _nb(_code("a000", "x = 1"))
        log = MutationLog()
        result = dispatch(nb, log, "edit_cell_source", {"cell_id": "a000", "new_source": "x = 2"})
        assert result["cell_id"] == "a000"
        assert _src(nb["cells"][0]) == "x = 2"

    def test_dispatch_unknown_tool_raises(self):
        with pytest.raises(MutatorError, match="Unknown"):
            dispatch(_nb(), MutationLog(), "format_disk", {})

    def test_dispatch_bad_args_raises(self):
        with pytest.raises(MutatorError, match="Bad args"):
            dispatch(_nb(), MutationLog(), "edit_cell_source", {"wrong_key": "x"})
