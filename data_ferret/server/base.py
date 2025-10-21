"""
Abstract base class for notebook processing commands.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from data_ferret.server.kernel_manager import FerretKernelClient


class NotebookCommand(ABC):
    """Abstract base class for notebook processing commands."""

    @abstractmethod
    def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process the notebook content and return a new notebook with metadata.

        Args:
            notebook_content: The parsed JSON content of a Jupyter notebook
            kernel_client: Optional kernel client for executing code
            **kwargs: Additional parameters specific to the command

        Returns:
            Dictionary containing:
                - notebook: The new/modified notebook
                - metadata: JSON metadata object with processing results
        """
        pass

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
