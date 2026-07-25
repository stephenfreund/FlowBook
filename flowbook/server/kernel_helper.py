"""
Helper utilities for kernel communication.

This module provides utilities for executing code in Jupyter kernels and
injecting runtime modifications like CSV downsampling.
"""
import textwrap
from queue import Empty
from typing import Any, Callable, Dict, Optional

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


def extract_flowbook_metadata(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the first ``type == "metadata"`` flowbook protocol message
    from a ``KernelHelper.execute_code()`` result, or None.

    Shared per-cell metadata extraction (audit A4) used by both
    ``ExecuteCommand.process`` (flowbook/server/commands/execute.py) and
    ``NotebookSession.run_cell`` (flowbook/mcp/session.py). The two callers
    keep separate execution loops on purpose — see the cross-reference
    comments at those call sites.
    """
    for msg in result.get("flowbook_messages", []):
        if isinstance(msg, dict) and msg.get("type") == "metadata":
            return msg
    return None


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

        Raises:
            ValueError: If proportion is not a number in [0.0, 1.0].

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
        # Coerce and validate BEFORE interpolating into kernel code (audit
        # S5): a string or expression must never reach the template, where
        # it would be executed verbatim in the kernel.
        try:
            proportion = float(proportion)
        except (TypeError, ValueError):
            raise ValueError(
                f"proportion must be a number between 0.0 and 1.0, "
                f"got {proportion!r}"
            )
        if not 0.0 <= proportion <= 1.0:  # also rejects NaN
            raise ValueError(
                f"proportion must be between 0.0 and 1.0, got {proportion!r}"
            )

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
        actor: str = None,
        on_foreign_msg: Optional[Callable[[Dict[str, Any]], None]] = None,
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
            on_foreign_msg: Optional callback invoked with each IOPub message
                whose parent is NOT this execution. On a shared kernel, another
                client's executions (e.g. JupyterLab while MCP is inside this
                call) produce messages on the same IOPub socket; without a
                handler they are drained and lost, silently desyncing the
                caller's state. Callback errors are logged and swallowed.
        Returns:
            Dictionary with execution results including outputs, status, and
            flowbook_messages (list of protocol messages received from kernel).
        """
        # Merge timeout and flowbook message into cell_metadata
        meta_with_timeout = dict(cell_metadata) if cell_metadata else {}
        meta_with_timeout['timeout'] = timeout
        if flowbook_msg is not None:
            meta_with_timeout['flowbook'] = flowbook_msg
        if actor is not None:
            # Tells the kernel who drove this execution; echoed on the
            # flowbook_update so a co-located LogBook attributes it.
            meta_with_timeout['actor'] = actor

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
            except Empty:
                continue

            if msg['parent_header'].get('msg_id') != msg_id:
                # Message from another execution on a shared kernel — hand it
                # to the caller instead of dropping it.
                if on_foreign_msg is not None:
                    try:
                        on_foreign_msg(msg)
                    except Exception as e:
                        log(f"on_foreign_msg handler error: {e}")
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

        # Get the execute_reply message. Replies on the shell channel can only
        # be for OUR requests, but an earlier call that timed out may have
        # abandoned its reply in the queue — discard stale replies until we
        # find the one matching this request.
        try:
            shell_deadline = time.time() + 1.0
            while True:
                remaining = shell_deadline - time.time()
                if remaining <= 0:
                    raise Empty
                reply = kernel_client.get_shell_msg(timeout=remaining)
                if reply['parent_header'].get('msg_id') == msg_id:
                    break
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
