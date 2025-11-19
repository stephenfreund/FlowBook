"""
Abstract base class for notebook processing commands.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agents import Usage
from pydantic import BaseModel, Field
from data_ferret.agent.agent import FerretStats
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.server.config import FerretConfig
from jupyter_server.serverapp import ServerApp

class ProcessingResult(BaseModel):
    """Result of a notebook processing command."""
    notebook: Dict[str, Any] = Field(description="The new/modified notebook")
    metadata: Dict[str, Any] = Field(description="JSON metadata object with processing results")
    total_cost: float = Field(description="Total cost of the command")
    total_time: float = Field(description="Total time taken to execute the command")


class NotebookCommand(ABC):
    """Abstract base class for notebook processing commands."""

    @abstractmethod
    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[FerretConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process the notebook content and return a new notebook with metadata.

        Args:
            notebook_content: The parsed JSON content of a Jupyter notebook
            kernel_client: Optional kernel client for executing code
            selected_cell_ids: Optional list of selected cell IDs
            config: Optional configuration for the command (uses defaults if not provided)
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

    @staticmethod
    def config_from_serverapp(serverapp: ServerApp) -> FerretConfig:
        """Return the configuration from the serverapp."""
        return FerretConfig(
            model=serverapp.web_app.settings["data_ferret"].model,
            fast_model=serverapp.web_app.settings["data_ferret"].fast_model,
        )
