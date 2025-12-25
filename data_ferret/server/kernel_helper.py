"""
Helper utilities for kernel communication.
"""
from typing import Any, Dict
import time
from data_ferret.server.kernel_manager import FerretKernelClient


class KernelHelper:
    """Helper class for kernel communication."""

    @staticmethod
    def execute_code(kernel_client: FerretKernelClient, code: str, timeout: float = 30.0, *, cell_id: str = None, cell_metadata: dict = None, store_history: bool = True) -> Dict[str, Any]:
        """
        Execute code in the kernel and return results.

        Args:
            kernel_client: The kernel client to use
            code: Code to execute
            timeout: Timeout in seconds
            cell_id: Cell ID
            cell_metadata: Cell metadata
            store_history: Whether to store the code in the kernel's history (default: True)
        Returns:
            Dictionary with execution results including outputs and status
        """
        # Merge timeout into cell_metadata so the kernel can use it
        meta_with_timeout = dict(cell_metadata) if cell_metadata else {}
        meta_with_timeout['timeout'] = timeout

        msg_id = kernel_client.execute(code, cell_id=cell_id, cell_metadata=meta_with_timeout, store_history=store_history)

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
                outputs.append({
                    'output_type': 'error',
                    'ename': content['ename'],
                    'evalue': content['evalue'],
                    'traceback': [ line.rstrip() for line in content['traceback'] ]
                })
                error_message = '\n'.join([ line.rstrip() for line in content['traceback'] ])

            elif msg_type == 'status':
                if content['execution_state'] == 'idle':
                    break

        # Get the execute_reply message
        try:
            reply = kernel_client.get_shell_msg(timeout=1.0)
            reply_status = reply['content']['status']
            if reply_status == 'error':
                status = 'error'
                # Extract error details from reply if not already captured
                if error_message is None:
                    error_content = reply['content']
                    if not outputs or outputs[-1].get('output_type') != 'error':
                        outputs.append({
                            'output_type': 'error',
                            'ename': error_content.get('ename', 'UnknownError'),
                            'evalue': error_content.get('evalue', ''),
                            'traceback': error_content.get('traceback', [])
                        })
                        error_message = '\n'.join([ line.rstrip() for line in error_content['traceback'] ])
        except Exception:
            pass

        return {
            'status': status,
            'execution_count': execution_count,
            'outputs': outputs,
            'error_message': error_message
        }
