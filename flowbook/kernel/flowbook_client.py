"""
FlowbookKernelClient - Client that sends cell order with executions.

Extends BaseFlowbookClient to inject cell_id and cell_order
into execution requests.
"""

from typing import List

from flowbook.kernel_support.base_client import BaseFlowbookClient


class FlowbookKernelClient(BaseFlowbookClient):
    """
    Kernel client that sends Reproducibility context (cell_id, cell_order) with executions.

    Usage:
        client = FlowbookKernelClient()
        client.set_cell_order(['cell1', 'cell2', 'cell3'])
        client.execute(code, cell_id='cell2')
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
