"""
Built-in command implementations for notebook processing.
"""

from data_ferret.server.commands.analyze import AnalyzeNotebookCommand
from data_ferret.server.commands.validate import ValidateNotebookCommand
from data_ferret.server.commands.profile import ProfileCommand
from data_ferret.server.commands.inspect_variables import InspectVariablesCommand
from data_ferret.server.commands.inspect import InspectCommand
from data_ferret.server.commands.optimize import OptimizeCommand
from data_ferret.server.commands.cleanup import CleanupCommand
from data_ferret.server.commands.document import DocumentCommand
from data_ferret.server.commands.example_message_command import ExampleMessageCommand
from data_ferret.server.commands.validate_change import ValidateChangeCommand
from data_ferret.server.commands.generate import GenerateCodeCommand
from data_ferret.server.commands.split import SplitCommand
from data_ferret.server.commands.test import TestCommand

__all__ = [
    "AnalyzeNotebookCommand",
    "ValidateNotebookCommand",
    "ProfileCommand",
    "InspectVariablesCommand",
    "InspectCommand",
    "OptimizeCommand",
    "CleanupCommand",
    "DocumentCommand",
    "ExampleMessageCommand",
    "ValidateChangeCommand",
    "GenerateCodeCommand",
    "SplitCommand",
    "TestCommand",
]

