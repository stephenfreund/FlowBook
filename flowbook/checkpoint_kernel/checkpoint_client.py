"""
CheckpointKernelClient - Client that sends cell_id with executions.

Extends BlockingKernelClient to inject cell_id into execution requests.
"""

from typing import Optional

from jupyter_client.blocking import BlockingKernelClient


class CheckpointKernelClient(BlockingKernelClient):
    """
    Kernel client that sends cell_id with executions.

    Usage:
        client = CheckpointKernelClient()
        client.execute(code, cell_id='cell_abc')
    """

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
        Execute code with cell_id context.

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
