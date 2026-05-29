"""Unit tests for the unified refactoring handlers over a DictController.

These verify the single-source handler bodies independently of any transport.
The server/MCP/NBI cutovers reuse the same handlers, so passing here is the
load-bearing guarantee for all three surfaces.
"""

import pytest

from flowbook.tools import get
from flowbook.tools.adapters.dict_controller import DictController
from flowbook.tools.controller import ToolError


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
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _run(notebook, tool, args):
    ctrl = DictController(notebook)
    result = get(tool).handler(ctrl, **args)
    return ctrl, result


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


class TestAlphaRename:
    def test_renames_from_target_onwards(self):
        nb = _nb(
            _code("a000", "import pandas as pd\ntrain = pd.read_csv('x.csv')"),
            _code("b000", "train = pd.concat([train, extra])"),
            _code("c000", "y = train.head()"),
        )
        ctrl, result = _run(
            nb, "alpha_rename",
            {"cell_id": "b000", "old_name": "train", "new_name": "train_combined"},
        )
        assert result["modified_cells"] == ["b000", "c000"]
        assert "train = pd.read_csv" in _src(nb["cells"][0])
        assert "train_combined" in _src(nb["cells"][1])
        assert ctrl.pre_sources["b000"] == "train = pd.concat([train, extra])"

    def test_no_effect_raises(self):
        nb = _nb(_code("b000", "y = 1"))
        with pytest.raises(ToolError, match="no effect"):
            _run(nb, "alpha_rename", {"cell_id": "b000", "old_name": "z", "new_name": "x"})

    def test_missing_cell_raises(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(ToolError, match="not found"):
            _run(nb, "alpha_rename", {"cell_id": "zzzz", "old_name": "x", "new_name": "y"})


class TestRemoveInplace:
    def test_converts(self):
        nb = _nb(_code("a000", "df.drop(columns=['x'], inplace=True)"))
        _run(nb, "remove_inplace", {"cell_id": "a000", "variable": "df"})
        s = _src(nb["cells"][0])
        assert "inplace" not in s and "df = df.drop" in s

    def test_no_inplace_raises(self):
        nb = _nb(_code("a000", "df = df.drop(columns=['x'])"))
        with pytest.raises(ToolError, match="no effect"):
            _run(nb, "remove_inplace", {"cell_id": "a000", "variable": "df"})


class TestInsertDeepcopy:
    def test_inserts_and_renames(self):
        nb = _nb(
            _code("a000", "model = LR()"),
            _code("b000", "model.fit(X, y)"),
            _code("c000", "pred = model.predict(X)"),
        )
        _, result = _run(nb, "insert_deepcopy", {"cell_id": "b000", "variable": "model"})
        assert result["new_name"] == "model_b000"
        assert "copy.deepcopy(model)" in _src(nb["cells"][1])
        assert "model_b000.predict" in _src(nb["cells"][2])
        assert result["modified_downstream"] == ["c000"]


class TestMarkDiagnostic:
    def test_prepends(self):
        nb = _nb(_code("a000", "df.info()"))
        _run(nb, "mark_diagnostic", {"cell_id": "a000"})
        assert _src(nb["cells"][0]).startswith("%diagnostic")

    def test_already_raises(self):
        nb = _nb(_code("a000", "%diagnostic\ndf.info()"))
        with pytest.raises(ToolError, match="already"):
            _run(nb, "mark_diagnostic", {"cell_id": "a000"})


class TestMergeCells:
    def test_merges_and_removes(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"), _code("c000", "z = x + y"))
        ctrl, result = _run(nb, "merge_cells", {"cell_ids": ["a000", "b000"]})
        assert result["cells_removed"] == ["b000"]
        assert _src(nb["cells"][0]) == "x = 1\n\ny = 2"
        assert result["new_cell_order"] == ["a000", "c000"]
        # Undo needs the removed cell's pre-source.
        assert ctrl.pre_sources["b000"] == "y = 2"
        assert ctrl.removed == ["b000"]

    def test_too_few_raises(self):
        nb = _nb(_code("a000", "x = 1"))
        with pytest.raises(ToolError, match="at least 2"):
            _run(nb, "merge_cells", {"cell_ids": ["a000"]})


class TestMoveCell:
    def test_reorders(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "df.info()"), _code("c000", "df = df.dropna()"))
        ctrl, result = _run(nb, "move_cell", {"cell_id": "b000", "after_cell_id": "c000"})
        assert [c["id"] for c in nb["cells"]] == ["a000", "c000", "b000"]
        assert result["new_cell_order"] == ["a000", "c000", "b000"]
        assert ctrl.order_changed and not ctrl.post_sources  # order-only change

    def test_unknown_destination_restores(self):
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"))
        with pytest.raises(ToolError, match="not found"):
            _run(nb, "move_cell", {"cell_id": "b000", "after_cell_id": "zzzz"})
        assert [c["id"] for c in nb["cells"]] == ["a000", "b000"]


class TestRegistry:
    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            get("format_disk")

    def test_six_tools_registered(self):
        from flowbook.tools import names
        assert set(names()) == {
            "alpha_rename", "remove_inplace", "insert_deepcopy",
            "mark_diagnostic", "merge_cells", "move_cell",
        }
