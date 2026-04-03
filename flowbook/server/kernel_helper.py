"""
Helper utilities for kernel communication.

This module provides utilities for executing code in Jupyter kernels and
injecting runtime modifications like CSV downsampling.
"""
import textwrap
from typing import Any, Dict, Optional

import time
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.util.output import log


# Template for CSV downsampling monkey-patch code.
# Patches both pandas and cuDF (if available) to return only a proportion
# of rows when reading CSV files. This is useful for faster iteration
# during development/testing with large datasets.
CSV_DOWNSAMPLE_PATCH_TEMPLATE = textwrap.dedent('''
    # Patch pandas read_csv
    import pandas as pd
    _original_pd_read_csv = pd.read_csv

    def _downsampled_pd_read_csv(*args, **kwargs):
        df = _original_pd_read_csv(*args, **kwargs)
        n_rows = int(len(df) * {proportion})
        print(f"[pandas] Downsampling CSV: keeping top", n_rows, "of", len(df), "rows")
        return df.head(n_rows)

    pd.read_csv = _downsampled_pd_read_csv

    # Patch cuDF read_csv if available
    try:
        import cudf
        _original_cudf_read_csv = cudf.read_csv

        def _downsampled_cudf_read_csv(*args, **kwargs):
            df = _original_cudf_read_csv(*args, **kwargs)
            n_rows = int(len(df) * {proportion})
            print(f"[cudf] Downsampling CSV: keeping top", n_rows, "of", len(df), "rows")
            return df.head(n_rows)

        cudf.read_csv = _downsampled_cudf_read_csv
        print("CSV downsampling enabled for both pandas and cuDF")
    except ImportError:
        print("CSV downsampling enabled for pandas (cuDF not available)")
''').strip()


class KernelHelper:
    """Helper class for kernel communication."""

    @staticmethod
    def inject_csv_downsampling(
        kernel_client: FlowbookKernelClient,
        proportion: float,
    ) -> Dict[str, Any]:
        """
        Inject CSV downsampling monkey-patch into the kernel.

        This function patches `pd.read_csv` and `cudf.read_csv` (if available)
        to return only the first N rows of any CSV file, where N is determined
        by the proportion parameter. This is useful for:

        - Faster iteration during development with large datasets
        - Testing notebook execution without waiting for full data loads
        - Debugging data pipelines with representative subsets

        The patch is applied once at the start of execution and affects all
        subsequent `read_csv` calls in that kernel session.

        Args:
            kernel_client: The kernel client to inject the patch into.
            proportion: Fraction of rows to keep (0.0 to 1.0).
                - 0.1 = keep first 10% of rows
                - 0.5 = keep first 50% of rows
                - 1.0 = keep all rows (no-op)

        Returns:
            Dictionary with execution results from applying the patch,
            including any outputs or errors.

        Example:
            >>> # Keep only 10% of CSV data for faster testing
            >>> KernelHelper.inject_csv_downsampling(kernel_client, 0.1)

        Note:
            - The patch uses `df.head(n_rows)`, so it always returns the
              FIRST n rows, not a random sample. This ensures reproducibility.
            - Each `read_csv` call prints a message showing how many rows
              were kept vs. the original count.
            - The patch persists for the lifetime of the kernel session.
        """
        patch_code = CSV_DOWNSAMPLE_PATCH_TEMPLATE.format(proportion=proportion)
        result = KernelHelper.execute_code(
            kernel_client,
            patch_code,
            store_history=False
        )
        log(f"CSV downsampling enabled: keeping top {proportion*100:.1f}% of rows")
        return result

    @staticmethod
    def execute_code(
        kernel_client: FlowbookKernelClient,
        code: str,
        timeout: float = 30.0,
        *,
        cell_id: str = None,
        cell_metadata: dict = None,
        store_history: bool = True,
        flowbook_msg: dict = None,
    ) -> Dict[str, Any]:
        """
        Execute code in the kernel and return results.

        Args:
            kernel_client: The kernel client to use
            code: Code to execute
            timeout: Timeout in seconds
            cell_id: Cell ID
            cell_metadata: Cell metadata
            store_history: Whether to store the code in the kernel's history (default: True)
            flowbook_msg: Optional FlowBook protocol message to send via execute metadata.
                e.g. {"type": "cell_edited", "cell_id": "abc"}
        Returns:
            Dictionary with execution results including outputs, status, and
            flowbook_messages (list of protocol messages received from kernel).
        """
        # Merge timeout and flowbook message into cell_metadata
        meta_with_timeout = dict(cell_metadata) if cell_metadata else {}
        meta_with_timeout['timeout'] = timeout
        if flowbook_msg is not None:
            meta_with_timeout['flowbook'] = flowbook_msg

        msg_id = kernel_client.execute(code, cell_id=cell_id, cell_metadata=meta_with_timeout, store_history=store_history)

        outputs = []
        flowbook_messages = []
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

            elif msg_type == 'flowbook_update':
                # FlowBook protocol message from kernel
                fb_data = content.get('flowbook', content)
                flowbook_messages.append(fb_data)

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
            'flowbook_messages': flowbook_messages,
            'error_message': error_message
        }
