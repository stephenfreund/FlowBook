"""
Ferret Server - Jupyter Lab Server-Side Extension with Command Pattern and Kernel Communication

This module provides an extensible framework for processing Jupyter notebooks
with custom commands, including kernel communication capabilities.
"""

# Export base classes and interfaces
from data_ferret.server.base import NotebookCommand

# Export helper utilities
from data_ferret.server.kernel_helper import KernelHelper

# Export command implementations
from data_ferret.server.commands import (
    AnalyzeNotebookCommand,
    ValidateNotebookCommand,
    ProfileCommand,
    InspectVariablesCommand,
    InspectCommand,
    OptimizeCommand,
    CleanupCommand,
    ExampleMessageCommand,
)

# Export registry and managers
from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import KernelConnectionManager

# Export handlers and extension setup
from data_ferret.server.handlers import (
    FerretCommandHandler,
    CommandListHandler,
    MessageStreamHandler,
    setup_handlers,
    # _jupyter_server_extension_points,
    # _load_jupyter_server_extension
)

# Export message broadcasting
from data_ferret.server.message_broadcaster import (
    MessageBroadcaster,
    get_broadcaster,
    Message,
    MessageType,
)

# Export CLI
from data_ferret.server.cli import cli_main

__all__ = [
    # Base classes
    "NotebookCommand",
    # Utilities
    "KernelHelper",
    # Command implementations
    "AnalyzeNotebookCommand",
    "ValidateNotebookCommand",
    "ProfileCommand",
    "InspectVariablesCommand",
    "InspectCommand",
    "OptimizeCommand",
    "CleanupCommand",
    # Registry and managers
    "CommandRegistry",
    "KernelConnectionManager",
    # Handlers
    "FerretCommandHandler",
    "CommandListHandler",
    "MessageStreamHandler",
    "setup_handlers",
    "_jupyter_server_extension_points",
    "_load_jupyter_server_extension",
    # Message broadcasting
    "MessageBroadcaster",
    "get_broadcaster",
    "Message",
    "MessageType",
    # CLI
    "cli_main",
]

__version__ = "0.1.0"
