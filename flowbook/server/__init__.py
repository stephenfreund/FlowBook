"""
FlowBook Server - Jupyter Lab Server-Side Extension with Command Pattern and Kernel Communication

This module provides an extensible framework for processing Jupyter notebooks
with custom commands, including kernel communication capabilities.
"""

# Export base classes and interfaces
from flowbook.server.base import NotebookCommand

# Export helper utilities
from flowbook.server.kernel_helper import KernelHelper

# Export command implementations
from flowbook.server.commands import (
    CompareBaselineCommand,
    ExecuteCommand,
    ExecuteBaseCommand,
)

# Export registry and managers
from flowbook.server.registry import CommandRegistry
from flowbook.server.kernel_manager import KernelConnectionManager

# Export handlers and extension setup
from flowbook.server.handlers import (
    FlowbookCommandHandler,
    CommandListHandler,
    setup_handlers,
)

__all__ = [
    # Base classes
    "NotebookCommand",
    # Utilities
    "KernelHelper",
    # Command implementations
    "CompareBaselineCommand",
    "ExecuteCommand",
    "ExecuteBaseCommand",
    # Registry and managers
    "CommandRegistry",
    "KernelConnectionManager",
    # Handlers
    "FlowbookCommandHandler",
    "CommandListHandler",
    "setup_handlers",
]

# Note: CLI functions (cli_main) are in flowbook.cli package
# to avoid circular imports. Import them directly from flowbook.cli if needed.

__version__ = "0.1.0"
