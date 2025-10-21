"""
Command registry for managing available notebook processing commands.
"""
from typing import Dict, List

from data_ferret.server.base import NotebookCommand
from data_ferret.server.commands import (
    AnalyzeNotebookCommand,
    ValidateNotebookCommand,
    ExecuteAllCommand,
    InspectVariablesCommand
)


class CommandRegistry:
    """Registry for all available commands."""

    def __init__(self):
        self._commands: Dict[str, NotebookCommand] = {}
        self._register_default_commands()

    def _register_default_commands(self):
        """Register the default command implementations."""
        self.register(AnalyzeNotebookCommand())
        self.register(ValidateNotebookCommand())
        self.register(ExecuteAllCommand())
        self.register(InspectVariablesCommand())

    def register(self, command: NotebookCommand):
        """Register a command."""
        self._commands[command.command_name] = command

    def get_command(self, name: str) -> NotebookCommand:
        """Get a command by name."""
        if name not in self._commands:
            raise ValueError(f"Unknown command: {name}")
        return self._commands[name]

    def list_commands(self) -> List[str]:
        """List all registered command names."""
        return list(self._commands.keys())

    def get_command_info(self) -> List[dict]:
        """Get information about all commands for the UI."""
        return [
            {
                "id": cmd.command_name,
                "label": cmd.display_name,
                "icon": cmd.icon_name,
                "tooltip": cmd.tooltip,
                "requires_kernel": cmd.requires_kernel
            }
            for cmd in self._commands.values()
        ]
