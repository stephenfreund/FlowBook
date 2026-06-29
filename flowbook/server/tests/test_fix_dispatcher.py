"""Tests for fix_dispatcher: each of the six fix tools on synthetic notebooks.

These are the most load-bearing tests in the AI-fix feature — they verify
that a validated FixPlan produces correct notebook transformations
independent of any LLM behavior.
"""

import pytest

from flowbook.server.fix_dispatcher import apply_fix


def _code(id_: str, src: str) -> dict:
    return {
        "cell_type": "code",
        "id": id_,
        "source": src,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }


def _nb(*cells) -> dict:
    return {
        "cells": list(cells),
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


class TestAlphaRename:
    def test_renames_from_target_onwards(self):
        nb = _nb(
            _code("a000", "import pandas as pd\ntrain = pd.read_csv('x.csv')"),
            _code("b000", "train = pd.concat([train, extra])"),
            _code("c000", "y = train.head()"),
        )
        result = apply_fix(nb, "alpha_rename", {
            "cell_id": "b000", "old_name": "train", "new_name": "train_combined",
        })
        assert result.ok
        assert result.modified_cells == ["b000", "c000"]
        # Earlier cell untouched
        assert "train = pd.read_csv" in _src(nb["cells"][0])
        # Renamed in target cell and downstream
        assert "train_combined" in _src(nb["cells"][1])
        assert "train_combined" in _src(nb["cells"][2])

    def test_pre_fix_sources_captured(self):
        nb = _nb(
            _code("a000", "x = 1"),
            _code("b000", "x = 2"),
        )
        result = apply_fix(nb, "alpha_rename", {
            "cell_id": "b000", "old_name": "x", "new_name": "x_b",
        })
        assert "b000" in result.pre_fix_sources
        assert result.pre_fix_sources["b000"] == "x = 2"

    def test_no_effect_raises(self):
        nb = _nb(_code("b000", "y = 1"))
        with pytest.raises(ValueError, match="no effect"):
            apply_fix(nb, "alpha_rename", {
                "cell_id": "b000", "old_name": "missing", "new_name": "x",
            })


class TestRemoveInplace:
    def test_converts_inplace_to_assignment(self):
        nb = _nb(_code("a000", "df.drop(columns=['x'], inplace=True)"))
        result = apply_fix(nb, "remove_inplace", {"cell_id": "a000", "variable": "df"})
        assert result.ok
        new_source = _src(nb["cells"][0])
        assert "inplace" not in new_source
        assert "df = df.drop" in new_source

    def test_no_inplace_raises(self):
        nb = _nb(_code("a000", "df = df.drop(columns=['x'])"))
        with pytest.raises(ValueError, match="no effect"):
            apply_fix(nb, "remove_inplace", {"cell_id": "a000", "variable": "df"})


class TestInsertDeepcopy:
    def test_inserts_copy_and_renames_downstream(self):
        nb = _nb(
            _code("a000", "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()"),
            _code("b000", "model.fit(X, y)"),
            _code("c000", "pred = model.predict(X)"),
        )
        result = apply_fix(nb, "insert_deepcopy", {"cell_id": "b000", "variable": "model"})
        assert result.ok
        b_source = _src(nb["cells"][1])
        assert "copy.deepcopy(model)" in b_source
        assert "model_b000" in b_source
        # Downstream renamed
        c_source = _src(nb["cells"][2])
        assert "model_b000.predict" in c_source


class TestMarkDiagnostic:
    def test_prepends_magic(self):
        nb = _nb(_code("a000", "df.info()"))
        result = apply_fix(nb, "mark_diagnostic", {"cell_id": "a000"})
        assert result.ok
        assert _src(nb["cells"][0]).startswith("%diagnostic")

    def test_already_diagnostic_raises(self):
        nb = _nb(_code("a000", "%diagnostic\ndf.info()"))
        with pytest.raises(ValueError, match="already"):
            apply_fix(nb, "mark_diagnostic", {"cell_id": "a000"})


class TestMergeCells:
    def test_combines_sources_and_removes(self):
        nb = _nb(
            _code("a000", "x = 1"),
            _code("b000", "y = 2"),
            _code("c000", "z = x + y"),
        )
        result = apply_fix(nb, "merge_cells", {"cell_ids": ["a000", "b000"]})
        assert result.ok
        assert result.cells_removed == ["b000"]
        assert len(nb["cells"]) == 2
        assert _src(nb["cells"][0]) == "x = 1\n\ny = 2"
        # Order updated to reflect removal
        assert result.new_cell_order == ["a000", "c000"]

    def test_too_few_raises(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(ValueError, match="at least 2"):
            apply_fix(nb, "merge_cells", {"cell_ids": ["a000"]})


class TestMoveCell:
    def test_reorders_to_after_target(self):
        nb = _nb(
            _code("a000", "x = 1"),
            _code("b000", "df.info()"),  # diagnostic, moved later
            _code("c000", "df = df.dropna()"),
        )
        result = apply_fix(nb, "move_cell", {"cell_id": "b000", "after_cell_id": "c000"})
        assert result.ok
        ids = [c["id"] for c in nb["cells"]]
        assert ids == ["a000", "c000", "b000"]
        assert result.new_cell_order == ["a000", "c000", "b000"]

    def test_unknown_destination_restores(self):
        nb = _nb(
            _code("a000", "x = 1"),
            _code("b000", "y = 2"),
        )
        with pytest.raises(ValueError, match="not found"):
            apply_fix(nb, "move_cell", {"cell_id": "b000", "after_cell_id": "zzzz"})
        # Cell should still be in original place
        assert [c["id"] for c in nb["cells"]] == ["a000", "b000"]


class TestUnknownTool:
    def test_unknown_tool_raises(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(ValueError, match="Unknown tool"):
            apply_fix(nb, "format_disk", {"cell_id": "a000"})
