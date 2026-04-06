"""Tests for FlowBookNBIExtension registration."""

import pytest
from unittest.mock import MagicMock, call

from flowbook.nbi.extension import FlowBookNBIExtension


class TestFlowBookNBIExtension:
    def test_id(self):
        ext = FlowBookNBIExtension()
        assert ext.id == "flowbook"

    def test_name(self):
        ext = FlowBookNBIExtension()
        assert ext.name == "FlowBook Reproducibility"

    def test_provider(self):
        ext = FlowBookNBIExtension()
        assert ext.provider == "FlowBook"

    def test_activate_disables_builtin_toolsets(self):
        ext = FlowBookNBIExtension()
        host = MagicMock()
        ext.activate(host)

        # Should disable both conflicting NBI toolsets
        host.disable_builtin_toolset.assert_any_call("nbi-notebook-edit")
        host.disable_builtin_toolset.assert_any_call("nbi-notebook-execute")
        assert host.disable_builtin_toolset.call_count == 2

    def test_activate_registers_toolset(self):
        ext = FlowBookNBIExtension()
        host = MagicMock()
        ext.activate(host)

        host.register_toolset.assert_called_once()
        toolset = host.register_toolset.call_args[0][0]
        assert toolset.id == "flowbook-reproducibility"
        assert toolset.name == "FlowBook Reproducibility"
        assert toolset.provider is ext
        assert toolset.instructions is not None
        assert "@A" in toolset.instructions  # Instructions mention @A notation

    def test_activate_registers_all_tools(self):
        ext = FlowBookNBIExtension()
        host = MagicMock()
        ext.activate(host)

        toolset = host.register_toolset.call_args[0][0]
        tool_names = [t.name for t in toolset.tools]

        # Verify all expected tools are present (unified API)
        expected = [
            "get_flowbook_metadata", "get_next_actionable_cell", "get_status",
            "read_cell", "edit_cell_source", "get_all_cell_sources",
            "add_cell", "delete_cell",
            "run_cell", "run_actionable_cell", "run_actionable_cells", "run_all_cells",
            "continue_after_violation",
            "alpha_rename", "remove_inplace", "insert_deepcopy", "mark_diagnostic",
            "merge_cells", "move_cell",
            "save_notebook",
            "checkpoint", "restore", "list_checkpoints", "get_log", "save_log", "print_log",
        ]
        assert set(tool_names) == set(expected)
        assert len(tool_names) == len(expected)

    def test_activate_order(self):
        """Verify disable happens before register (tools shouldn't be visible while conflicting ones exist)."""
        ext = FlowBookNBIExtension()
        host = MagicMock()
        call_order = []
        host.disable_builtin_toolset.side_effect = lambda x: call_order.append(("disable", x))
        host.register_toolset.side_effect = lambda x: call_order.append(("register", x.id))

        ext.activate(host)

        assert call_order[0] == ("disable", "nbi-notebook-edit")
        assert call_order[1] == ("disable", "nbi-notebook-execute")
        assert call_order[2] == ("register", "flowbook-reproducibility")


class TestExtensionMetadata:
    def test_extension_json_exists(self):
        import json
        from pathlib import Path
        metadata_path = Path(__file__).parent.parent / "extension_data" / "extension.json"
        assert metadata_path.exists(), f"extension.json not found at {metadata_path}"

        with open(metadata_path) as f:
            data = json.load(f)
        assert data["class"] == "flowbook.nbi.extension.FlowBookNBIExtension"

    def test_extension_class_loadable(self):
        """Verify NBI's dynamic loading mechanism can import the class."""
        import importlib
        parts = "flowbook.nbi.extension.FlowBookNBIExtension".split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        assert cls is FlowBookNBIExtension
