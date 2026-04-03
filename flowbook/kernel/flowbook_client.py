"""
FlowbookKernelClient - Client that sends cell order with executions.

Extends BaseFlowbookClient to inject cell_id and cell_order
into execution requests. Supports the FlowBook protocol for sending
structured commands via execute request metadata.
"""

from typing import Dict, List, Optional

from flowbook.kernel_support.base_client import BaseFlowbookClient


class FlowbookKernelClient(BaseFlowbookClient):
    """
    Kernel client that sends Reproducibility context (cell_id, cell_order) with executions.

    Usage:
        client = FlowbookKernelClient()
        client.set_cell_order(['cell1', 'cell2', 'cell3'])
        client.execute(code, cell_id='cell2')

        # Send a protocol command:
        client.send_flowbook_command({"type": "cell_edited", "cell_id": "abc"})
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cell_order: List[str] = []

    def set_cell_order(self, order: List[str]) -> None:
        """Set the notebook cell order for Reproducibility enforcement."""
        self._cell_order = list(order)

    def _enrich_metadata(self, metadata: dict) -> None:
        """Add cell_order to metadata for Reproducibility enforcement."""
        if self._cell_order:
            metadata["cell_order"] = self._cell_order

    def send_flowbook_command(self, msg: dict) -> str:
        """Send a FlowBook protocol command to the kernel.

        Sends an empty code execution with the protocol message in metadata.
        The kernel's _handle_flowbook_message() dispatches it.

        Args:
            msg: Protocol message dict, e.g. {"type": "cell_edited", "cell_id": "abc"}

        Returns:
            Message ID
        """
        return self.execute(
            "", cell_metadata={"flowbook": msg}, store_history=False
        )

    def execute_with_structure(
        self,
        code: str,
        cell_id: str,
        cell_order: List[str],
        **kwargs,
    ) -> str:
        """
        Convenience method: set order and execute in one call.

        Args:
            code: Code to execute
            cell_id: ID of the cell being executed
            cell_order: Full notebook cell order
            **kwargs: Additional arguments to execute()

        Returns:
            Message ID
        """
        self.set_cell_order(cell_order)
        return self.execute(code, cell_id=cell_id, **kwargs)
