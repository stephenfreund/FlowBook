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


def _filter_mime_bundle(data: dict) -> dict:
    """Convert a Jupyter MIME bundle to the `Output.data` shape used by
    `mcp_content`: each kept MIME type becomes either a text Payload or a
    base64 Payload; dropped types still appear so the adapter can emit a
    marker, but their value is a size-only stub.
    """
    from flowbook.tools.mcp_content import KEEP_IMAGE_MIMES, KEEP_TEXT_MIMES

    out: dict[str, dict] = {}
    for mime, value in data.items():
        if mime in KEEP_IMAGE_MIMES:
            b64 = value if isinstance(value, str) else ""
            out[mime] = {
                "encoding": "base64",
                "bytes": b64,
                "size_bytes": (len(b64) * 3) // 4,
            }
        elif mime in KEEP_TEXT_MIMES:
            text = value if isinstance(value, str) else str(value)
            out[mime] = {"text": text}
        else:
            # Dropped MIME — keep a size stub so the adapter can emit a marker
            as_text = value if isinstance(value, str) else str(value)
            out[mime] = {"size_bytes": len(as_text)}
    return out


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
    def execute_scratch(
        kernel_client: FlowbookKernelClient,
        code: str,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Run code against the live kernel with `silent=True` and a
        kernel-side checkpoint/restore wrapper (flowbook_isolate=True).

        The user namespace is restored after the call, no tracking is
        recorded, no staleness is propagated, and no flowbook_update
        messages are emitted. Stdout/stderr, execute_result, display_data,
        and errors are collected from IOPub and returned as a dict matching
        the `ScratchResult` shape used by `flowbook/tools/mcp_content.py`.
        """
        import time as _time

        cell_metadata = {"timeout": timeout, "flowbook_isolate": True}
        t0 = _time.time()
        msg_id = kernel_client.execute(
            code,
            silent=True,
            store_history=False,
            cell_metadata=cell_metadata,
        )

        outputs: list[dict] = []
        status = "ok"
        error: dict | None = None

        start = _time.time()
        while True:
            if _time.time() - start > timeout:
                status = "error"
                error = {
                    "ename": "TimeoutError",
                    "evalue": f"scratch_work timed out after {timeout}s",
                    "traceback": [],
                }
                break
            try:
                msg = kernel_client.get_iopub_msg(timeout=1.0)
            except Exception:
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            mt = msg["header"]["msg_type"]
            content = msg["content"]

            if mt == "stream":
                outputs.append({
                    "kind": "stream",
                    "stream_name": content["name"],
                    "text": content["text"],
                })
            elif mt == "execute_result":
                outputs.append({
                    "kind": "execute_result",
                    "data": _filter_mime_bundle(content.get("data") or {}),
                })
            elif mt == "display_data":
                outputs.append({
                    "kind": "display_data",
                    "data": _filter_mime_bundle(content.get("data") or {}),
                })
            elif mt == "error":
                status = "error"
                error = {
                    "ename": content.get("ename", ""),
                    "evalue": content.get("evalue", ""),
                    "traceback": [line.rstrip() for line in content.get("traceback", [])],
                }
            elif mt == "status":
                if content["execution_state"] == "idle":
                    break

        # Drain the shell reply (silent requests still produce one)
        try:
            kernel_client.get_shell_msg(timeout=1.0)
        except Exception:
            pass

        return {
            "status": status,
            "execution_time_ms": (_time.time() - t0) * 1000.0,
            "outputs": outputs,
            "error": error,
        }

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
