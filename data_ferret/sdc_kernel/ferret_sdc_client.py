"""
FerretSDCKernelClient - Client that sends cell order with executions.

Extends BlockingKernelClient to inject cell_id and cell_order
into execution requests.
"""

from typing import List, Optional

from jupyter_client.blocking import BlockingKernelClient


class FerretSDCKernelClient(BlockingKernelClient):
    """
    Kernel client that sends SDC context (cell_id, cell_order) with executions.

    Usage:
        client = FerretSDCKernelClient()
        client.set_cell_order(['cell1', 'cell2', 'cell3'])
        client.execute(code, cell_id='cell2')
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cell_order: List[str] = []

    def set_cell_order(self, order: List[str]) -> None:
        """Set the notebook cell order for SDC enforcement."""
        self._cell_order = list(order)

    def execute(
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: Optional[bool] = None,
        stop_on_error: bool = True,
        *,
        cell_id: Optional[str] = None,
        cell_metadata: Optional[dict] = None,
    ) -> str:
        """
        Execute code with SDC context.

        Injects cell_id and cell_order into the execution metadata.

        Args:
            code: Code to execute
            silent: If True, suppress output
            store_history: If True, store in execution history
            user_expressions: Expressions to evaluate
            allow_stdin: Allow stdin requests
            stop_on_error: Stop on error
            cell_id: ID of the cell being executed
            cell_metadata: Additional cell metadata

        Returns:
            Message ID of the execute request
        """
        # Build metadata
        metadata = dict(cell_metadata) if cell_metadata else {}

        if cell_id:
            metadata["cell_id"] = cell_id

        if self._cell_order:
            metadata["cell_order"] = self._cell_order

        # Build content
        content = {
            "code": code,
            "silent": silent,
            "store_history": store_history,
            "user_expressions": user_expressions or {},
            "allow_stdin": (
                allow_stdin if allow_stdin is not None else self.allow_stdin
            ),
            "stop_on_error": stop_on_error,
        }

        # Send with metadata
        msg = self.session.msg("execute_request", content)
        msg["metadata"] = metadata

        self.shell_channel.send(msg)
        return msg["header"]["msg_id"]

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
