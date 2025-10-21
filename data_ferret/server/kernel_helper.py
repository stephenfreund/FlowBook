"""
Helper utilities for kernel communication.
"""
from typing import Any, Dict
import time
from data_ferret.server.kernel_manager import FerretKernelClient


class KernelHelper:
    """Helper class for kernel communication."""

    @staticmethod
    def execute_code(kernel_client: FerretKernelClient, code: str, timeout: float = 30.0, *, cell_id: str = None, cell_metadata: dict = None) -> Dict[str, Any]:
        """
        Execute code in the kernel and return results.

        Args:
            kernel_client: The kernel client to use
            code: Code to execute
            timeout: Timeout in seconds
            cell_id: Cell ID
            cell_metadata: Cell metadata
        Returns:
            Dictionary with execution results including outputs and status
        """
        msg_id = kernel_client.execute(code, cell_id=cell_id, cell_metadata=cell_metadata)

        outputs = []
        execution_count = None
        status = 'ok'
        error_message = None

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                status = 'timeout'
                error_message = f'Execution timed out after {timeout} seconds'
                break

            try:
                msg = kernel_client.get_iopub_msg(timeout=1.0)
            except:
                continue

            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            msg_type = msg['header']['msg_type']
            content = msg['content']

            if msg_type == 'execute_input':
                execution_count = content.get('execution_count')

            elif msg_type == 'stream':
                outputs.append({
                    'output_type': 'stream',
                    'name': content['name'],
                    'text': content['text']
                })

            elif msg_type == 'execute_result':
                outputs.append({
                    'output_type': 'execute_result',
                    'execution_count': content['execution_count'],
                    'data': content['data'],
                    'metadata': content.get('metadata', {})
                })

            elif msg_type == 'display_data':
                outputs.append({
                    'output_type': 'display_data',
                    'data': content['data'],
                    'metadata': content.get('metadata', {})
                })

            elif msg_type == 'error':
                status = 'error'
                error_message = '\n'.join(content['traceback'])
                outputs.append({
                    'output_type': 'error',
                    'ename': content['ename'],
                    'evalue': content['evalue'],
                    'traceback': content['traceback']
                })

            elif msg_type == 'status':
                if content['execution_state'] == 'idle':
                    break

        # Get the execute_reply message
        try:
            reply = kernel_client.get_shell_msg(timeout=1.0)
            if reply['content']['status'] == 'error' and status == 'ok':
                status = 'error'
        except:
            pass

        return {
            'status': status,
            'execution_count': execution_count,
            'outputs': outputs,
            'error_message': error_message
        }
