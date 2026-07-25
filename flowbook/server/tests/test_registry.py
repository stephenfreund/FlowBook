"""Tests for CommandRegistry command discovery.

Audit R5: a command class whose construction fails must be skipped with a
logged warning (naming the class and the exception) instead of being
silently swallowed, and must not break discovery of the other commands.
"""

import logging
import types

import pytest

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.registry import CommandRegistry


class GoodCommand(NotebookCommand):
    @property
    def command_name(self) -> str:
        return "good_command"

    @property
    def display_name(self) -> str:
        return "Good Command"

    @property
    def icon_name(self) -> str:
        return "ui-components:check"

    async def process(self, notebook_content, kernel_client=None,
                      selected_cell_ids=None, **kwargs) -> ProcessingResult:
        return ProcessingResult(notebook=notebook_content, metadata={})


class BrokenCommand(NotebookCommand):
    def __init__(self):
        raise RuntimeError("boom: cannot construct")

    @property
    def command_name(self) -> str:
        return "broken_command"

    @property
    def display_name(self) -> str:
        return "Broken Command"

    @property
    def icon_name(self) -> str:
        return "ui-components:cross"

    async def process(self, notebook_content, kernel_client=None,
                      selected_cell_ids=None, **kwargs) -> ProcessingResult:
        return ProcessingResult(notebook=notebook_content, metadata={})


class StillAbstractCommand(NotebookCommand):
    """Intentionally abstract — must be skipped without a warning."""


def _fake_module():
    module = types.ModuleType("fake_commands_module")
    module.GoodCommand = GoodCommand
    module.BrokenCommand = BrokenCommand
    module.StillAbstractCommand = StillAbstractCommand
    module.NotebookCommand = NotebookCommand  # base class itself is skipped
    return module


class TestRegisterCommandsFromModule:
    def test_broken_command_logs_warning_and_others_register(self, caplog):
        registry = CommandRegistry()
        module = _fake_module()

        with caplog.at_level(logging.WARNING, logger="flowbook.server.registry"):
            registry._register_commands_from_module(module)

        # The good command was registered despite the broken one.
        assert "good_command" in registry.list_commands()
        assert "broken_command" not in registry.list_commands()

        # The failure was logged, naming the class and the exception.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "BrokenCommand" in msg
        assert "boom: cannot construct" in msg

    def test_abstract_class_skipped_silently(self, caplog):
        registry = CommandRegistry()
        module = types.ModuleType("only_abstract")
        module.StillAbstractCommand = StillAbstractCommand

        with caplog.at_level(logging.WARNING, logger="flowbook.server.registry"):
            registry._register_commands_from_module(module)

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_default_discovery_registers_builtin_commands(self):
        registry = CommandRegistry()
        commands = registry.list_commands()
        assert "execute" in commands
