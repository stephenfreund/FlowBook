"""
Example command demonstrating message broadcasting to the client panel.

This shows how to use the message broadcaster to send real-time updates
to the client's message panel during command execution.
"""

import time
from data_ferret.server.base import NotebookCommand
from data_ferret.server.message_broadcaster import get_broadcaster


class ExampleMessageCommand(NotebookCommand):
    """
    Example command that demonstrates message broadcasting.

    This command sends various types of messages to the client panel:
    - Appending text to the current line
    - Creating new lines
    - Signaling completion
    """

    @property
    def command_name(self) -> str:
        return "example_message"

    @property
    def display_name(self) -> str:
        return "Example Message Stream"

    @property
    def icon_name(self) -> str:
        return "ui-components:text"

    @property
    def requires_kernel(self) -> bool:
        return False

    async def process(self, notebook_content: dict, kernel_client=None, selected_cell_ids=None, config=None, **kwargs) -> dict:
        """
        Process the notebook and send example messages.

        Args:
            notebook_content: The notebook content
            kernel_client: Optional kernel client (not used in this example)
            **kwargs: Additional parameters

        Returns:
            Result dict with notebook and metadata
        """
        broadcaster = get_broadcaster()

        # Send a greeting
        broadcaster.append("Starting example command...")
        broadcaster.newline()

        # Simulate some processing with progress updates
        for i in range(5):
            broadcaster.append(f"Processing step {i+1}/5... ")
            # In a real command, you'd do actual work here
            # For demo purposes, we'll just show the progress
            broadcaster.append("done")
            broadcaster.newline()

        # Send completion message
        broadcaster.newline()
        broadcaster.append("Example command completed successfully!")
        broadcaster.end()

        return {
            "notebook": notebook_content,
            "metadata": {
                "status": "success",
                "message": "Example messages sent to panel"
            }
        }


# Example usage in other commands:
"""
from data_ferret.server.message_broadcaster import get_broadcaster

def some_command_process(self, notebook_content, kernel_client=None, **kwargs):
    broadcaster = get_broadcaster()

    # Append text to current line (no newline)
    broadcaster.append("Processing: ")
    broadcaster.append("done")

    # Start a new line
    broadcaster.newline()

    # Append more text on the new line
    broadcaster.append("Analyzing cells...")
    broadcaster.newline()

    # Signal that the command/message is complete
    broadcaster.end()

    # Clear the panel (optional)
    # broadcaster.clear()

    return {...}
"""
