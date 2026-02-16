"""
Built-in command implementations for notebook processing.
"""

from flowbook.server.commands.analyze import AnalyzeNotebookCommand
from flowbook.server.commands.validate import ValidateNotebookCommand
from flowbook.server.commands.profile import ProfileCommand
from flowbook.server.commands.inspect_variables import InspectVariablesCommand
from flowbook.server.commands.inspect import InspectCommand
from flowbook.server.commands.optimize import OptimizeCommand
from flowbook.server.commands.cleanup import CleanupCommand
from flowbook.server.commands.document import DocumentCommand
from flowbook.server.commands.example_message_command import ExampleMessageCommand
from flowbook.server.commands.validate_change import ValidateChangeCommand
from flowbook.server.commands.generate import GenerateCodeCommand
from flowbook.server.commands.split import SplitCommand
from flowbook.server.commands.test import TestCommand
from flowbook.server.commands.generate_tests import GenerateTestsCommand
from flowbook.server.commands.prepare_code import PrepareCodeForFlowbookCommand
from flowbook.server.commands.test_leq import TestLeqCommand
from flowbook.server.commands.fix import FixCommand

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
    "GenerateTestsCommand",
    "PrepareCodeForFlowbookCommand",
    "TestLeqCommand",
    "FixCommand",
]

