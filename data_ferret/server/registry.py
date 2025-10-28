"""
Command registry for managing available notebook processing commands.
"""
from typing import Dict, List

from data_ferret.server.base import NotebookCommand
from data_ferret.server.commands import (
    AnalyzeNotebookCommand,
    ValidateNotebookCommand,
    ProfileCommand,
    InspectVariablesCommand,
    InspectCommand,
    CleanupCommand,
    ExampleMessageCommand,
)
from data_ferret.util.output import log, timer

class CommandRegistry:
    """Registry for all available commands."""

    def __init__(self):
        self._commands: Dict[str, NotebookCommand] = {}
        self._register_default_commands()

    def _register_default_commands(self):
        """Register the default command implementations."""
        # Dynamically import and register all commands in the commands subdirectory
        import pkgutil
        import importlib
        import os
        from data_ferret.server import commands

        with timer(key="register_commands", message="Registering commands"):
            commands_dir = os.path.dirname(commands.__file__)
            for _, module_name, is_pkg in pkgutil.iter_modules([commands_dir]):
                if is_pkg:
                    continue
                module = importlib.import_module(f"data_ferret.server.commands.{module_name}")
                # Register all classes in the module that are subclasses of NotebookCommand
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    try:
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, NotebookCommand)
                            and attr is not NotebookCommand
                        ):
                            log(f"{attr_name}...")
                            self.register(attr())
                    except Exception:
                        continue

    def get_command(self, name: str) -> NotebookCommand:
        """Get a command by name."""
        if name not in self._commands:
            raise ValueError(f"Unknown command: {name}")
        return self._commands[name]

    def list_commands(self) -> List[str]:
        """List all registered command names."""
        return list(self._commands.keys())  

    def register(self, command: NotebookCommand):
        """Register a command."""
        self._commands[command.command_name] = command

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
