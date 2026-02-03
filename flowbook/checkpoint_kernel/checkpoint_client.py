"""
CheckpointKernelClient - Client that sends cell_id with executions.

Extends BaseFlowbookClient to inject cell_id into execution requests.
"""

from flowbook.kernel_support.base_client import BaseFlowbookClient


class CheckpointKernelClient(BaseFlowbookClient):
    """
    Kernel client that sends cell_id with executions.

    Usage:
        client = CheckpointKernelClient()
        client.execute(code, cell_id='cell_abc')
    """

    pass
