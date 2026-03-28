"""
Abstract base class for notebook processing commands.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import argparse
import time

from pydantic import BaseModel, Field
from flowbook.agent.agent import FlowbookStats
from flowbook.kernel.protocol import format_message_for_cli
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.server.config import FlowbookConfig
from jupyter_server.serverapp import ServerApp
from flowbook.util.output import print

class ProcessingResult(BaseModel):
    """Result of a notebook processing command."""

    notebook: Dict[str, Any] = Field(description="The new/modified notebook")
    metadata: Dict[str, Any] = Field(
        description="JSON metadata object with processing results"
    )
    total_cost: float = Field(default=0.0, description="Total cost of the command")
    total_time: float = Field(
        default=0.0, description="Total time taken to execute the command"
    )


class NotebookCommand(ABC):
    """Abstract base class for notebook processing commands."""

    @abstractmethod
    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[FlowbookConfig] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process the notebook content and return a ProcessingResult.

        Args:
            notebook_content: The parsed JSON content of a Jupyter notebook
            kernel_client: Optional kernel client for executing code
            selected_cell_ids: Optional list of selected cell IDs
            config: Optional configuration for the command (uses defaults if not provided)
            **kwargs: Additional parameters specific to the command

        Returns:
            ProcessingResult containing:
                - notebook: The new/modified notebook
                - metadata: JSON metadata object with processing results
                - total_cost: Total cost in USD
                - total_time: Total time in seconds
        """
        pass

    @staticmethod
    def extract_cost_from_stats(stats: Optional[FlowbookStats]) -> float:
        """
        Extract cost from FlowbookStats object.

        Args:
            stats: Optional FlowbookStats object from agent execution

        Returns:
            Cost in USD, or 0.0 if stats is None
        """
        if stats is None:
            return 0.0
        return stats.cost

    @staticmethod
    @contextmanager
    def timing_context():
        """
        Context manager for timing command execution.

        Yields:
            A callable that returns elapsed time in seconds

        Example:
            with self.timing_context() as get_elapsed:
                # do work
                elapsed = get_elapsed()
        """
        start_time = time.time()

        def get_elapsed():
            return time.time() - start_time

        yield get_elapsed

    @property
    @abstractmethod
    def command_name(self) -> str:
        """Return the name of this command."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the display name for UI."""
        pass

    @property
    @abstractmethod
    def icon_name(self) -> str:
        """Return the icon name (Jupyter Lab icon or emoji)."""
        pass

    @property
    def tooltip(self) -> str:
        """Return tooltip text for the command button."""
        return self.display_name

    @property
    def requires_kernel(self) -> bool:
        """Return whether this command requires a kernel connection."""
        return False

    @property
    def kernel_name(self) -> str:
        """
        Return the kernel name to use for this command.

        Override this property to specify a different kernel.
        Default is 'flowbook_kernel'.
        """
        return "flowbook_kernel"

    @property
    def timeout(self) -> int:
        """Return the timeout for this command."""
        return 8 * 60 * 60  # 8 hours

    @staticmethod
    def config_from_serverapp(serverapp: ServerApp) -> FlowbookConfig:
        """Return the configuration from the serverapp."""
        return FlowbookConfig(
            model=serverapp.web_app.settings["flowbook"].model,
            fast_model=serverapp.web_app.settings["flowbook"].fast_model,
        )

    @staticmethod
    def print_flowbook_messages(
        result: Dict[str, Any],
        cell_order: Optional[List[str]] = None,
    ) -> None:
        """Print flowbook protocol messages from an execution result.

        Call this after KernelHelper.execute_code() to display status and
        violation messages on the CLI.

        Args:
            result: Return value from KernelHelper.execute_code()
            cell_order: Current notebook cell order (for @A notation)
        """
        for fb_msg in result.get("flowbook_messages", []):
            line = format_message_for_cli(fb_msg, cell_order)
            if line:
                print(line)

    @staticmethod
    def extract_flowbook_metadata(
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract flowbook metadata from an execution result.

        Looks for a message with type="metadata" in the flowbook_messages
        list returned by KernelHelper.execute_code().

        Args:
            result: Return value from KernelHelper.execute_code()

        Returns:
            The metadata dict, or None if not found.
        """
        for msg in result.get("flowbook_messages", []):
            if msg.get("type") == "metadata":
                return msg
        return None

    def make_subparser(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        """
        Create and return the subparser for this command.

        Override to add command-specific CLI arguments. The CLI will add
        the 'paths' argument after this returns.

        Args:
            subparsers: The subparsers action from the parent parser

        Returns:
            The configured subparser for this command
        """
        subparser = subparsers.add_parser(
            self.command_name,
            help=self.display_name,
        )
        return subparser
