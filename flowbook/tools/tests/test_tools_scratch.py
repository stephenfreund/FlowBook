"""Tests for FlowBookTools.scratch_work and FlowBookTools.get_cell_outputs."""

import pytest
from unittest.mock import MagicMock

from flowbook.tools.tools import FlowBookTools


def _session(cell_order=("aa", "bb", "cc")):
    s = MagicMock()
    s.is_loaded = True
    s.get_cell_order.return_value = list(cell_order)
    return s


class TestScratchWork:
    def test_delegates_to_session(self):
        s = _session()
        s.scratch_work.return_value = {
            "status": "ok", "execution_time_ms": 1.0, "outputs": [], "error": None,
        }
        tools = FlowBookTools(s)

        out = tools.scratch_work("x = 1")
        s.scratch_work.assert_called_once_with("x = 1")
        assert out["status"] == "ok"

    def test_returns_dict_not_string(self):
        s = _session()
        s.scratch_work.return_value = {"status": "ok", "outputs": []}
        tools = FlowBookTools(s)
        result = tools.scratch_work("print('hi')")
        assert isinstance(result, dict)


class TestGetCellOutputs:
    def test_resolves_alpha_labels(self):
        s = _session(("aa", "bb", "cc"))
        s.get_cell_outputs.return_value = {"cells": []}
        tools = FlowBookTools(s)

        tools.get_cell_outputs(["@A", "@C"])
        s.get_cell_outputs.assert_called_once_with(["aa", "cc"])

    def test_passes_cell_ids_through(self):
        s = _session(("aa", "bb", "cc"))
        s.get_cell_outputs.return_value = {"cells": []}
        tools = FlowBookTools(s)

        tools.get_cell_outputs(["bb"])
        s.get_cell_outputs.assert_called_once_with(["bb"])

    def test_empty_list(self):
        s = _session()
        s.get_cell_outputs.return_value = {"cells": []}
        tools = FlowBookTools(s)

        tools.get_cell_outputs([])
        s.get_cell_outputs.assert_called_once_with([])

    def test_unresolvable_ref_raises(self):
        s = _session(("aa", "bb"))
        tools = FlowBookTools(s)
        with pytest.raises(ValueError):
            tools.get_cell_outputs(["@Z"])
