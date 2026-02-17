"""
Benchmark checkpoint kernel - Measure cell execution and checkpoint times.

Usage:
    python -m flowbook.testing.benchmark_checkpoint notebook.ipynb
    python -m flowbook.testing.benchmark_checkpoint notebook.ipynb -o output.csv
    python -m flowbook.testing.benchmark_checkpoint notebook.ipynb --reruns 1000
"""

import argparse
import ast
import csv
import random
import sys
import time
from typing import List, Optional, TextIO

from jupyter_client import KernelManager

from flowbook import make_kernels
from flowbook.checkpoint_kernel import CheckpointKernelClient
from flowbook.testing.notebook_loader import Cell, load_notebook
from flowbook.util.output import log


def create_checkpoint_kernel() -> tuple[KernelManager, CheckpointKernelClient]:
    """
    Start the checkpoint kernel.

    Returns:
        Tuple of (KernelManager, CheckpointKernelClient)
    """
    make_kernels()

    max_attempts = 3
    kernel_manager = None
    kernel_client = None

    for attempt in range(max_attempts):
        try:
            # Clean up any previous failed attempt
            if kernel_client is not None:
                try:
                    kernel_client.stop_channels()
                except Exception:
                    pass
            if kernel_manager is not None:
                try:
                    kernel_manager.shutdown_kernel(now=True)
                except Exception:
                    pass

            # Start fresh kernel
            kernel_manager = KernelManager(kernel_name="checkpoint_kernel")
            kernel_manager.start_kernel()

            kernel_client = CheckpointKernelClient()
            kernel_client.load_connection_info(kernel_manager.get_connection_info())
            kernel_client.start_channels()

            # Race condition workaround
            time.sleep(2)
            while True:
                try:
                    kernel_client.wait_for_ready(timeout=30)
                    break
                except Exception as e:
                    log(f"Error waiting for kernel to be ready: {e}")
                    time.sleep(0.5)

            return kernel_manager, kernel_client

        except Exception as e:
            log(f"Error on attempt {attempt + 1}/{max_attempts}: {e}")
            if kernel_manager is not None and kernel_manager.is_alive():
                kernel_manager.shutdown_kernel(now=True)
                while kernel_manager.is_alive():
                    time.sleep(1)

            if attempt < max_attempts - 1:
                time.sleep(2)
            else:
                # Clean up before raising
                if kernel_client is not None:
                    try:
                        kernel_client.stop_channels()
                    except Exception:
                        pass
                if kernel_manager is not None:
                    try:
                        kernel_manager.shutdown_kernel(now=True)
                    except Exception:
                        pass
                raise Exception(f"Kernel failed to start after {max_attempts} attempts: {e}")

    raise Exception("Kernel failed to start")


def cleanup_kernel(
    kernel_manager: Optional[KernelManager],
    kernel_client: Optional[CheckpointKernelClient]
) -> None:
    """Clean up kernel resources."""
    if kernel_client:
        try:
            kernel_client.kernel_info()
            time.sleep(0.5)
        except Exception:
            pass

        try:
            kernel_client.stop_channels()
        except Exception as e:
            log(f"Warning: Error stopping kernel channels: {e}")

    if kernel_manager:
        try:
            kernel_manager.shutdown_kernel()
        except Exception as e:
            log(f"Warning: Error shutting down kernel: {e}")


def execute_cell_and_extract_timing(
    kernel_client: CheckpointKernelClient,
    cell: Cell,
    timeout: float = 300.0
) -> dict:
    """
    Execute a cell and extract timing from metadata.

    Returns:
        Dict with keys: execution_count, cell_runtime_s, commit_time_s, error
    """
    msg_id = kernel_client.execute(cell.source, cell_id=cell.cell_id)

    timing_data = None
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return {
                "execution_count": None,
                "cell_runtime_s": None,
                "commit_time_s": None,
                "error": f"Timeout after {timeout}s"
            }

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        # Look for display_data with flowbook_checkpoint metadata
        if msg_type == "display_data":
            metadata = msg.get("content", {}).get("metadata", {})
            if "flowbook_checkpoint" in metadata:
                timing_data = metadata["flowbook_checkpoint"]

        # Check for errors
        if msg_type == "error":
            content = msg["content"]
            error_msg = "\n".join(content.get("traceback", []))
            if timing_data is None:
                timing_data = {"error": error_msg}
            else:
                timing_data["error"] = error_msg

        # Done when kernel is idle
        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Get the execute_reply message
    try:
        reply = kernel_client.get_shell_msg(timeout=1.0)
        if reply["content"]["status"] == "error" and timing_data:
            if "error" not in timing_data:
                error_content = reply["content"]
                timing_data["error"] = "\n".join(error_content.get("traceback", []))
    except Exception:
        pass

    if timing_data is None:
        return {
            "execution_count": None,
            "cell_runtime_s": None,
            "commit_time_s": None,
            "error": "No timing metadata received"
        }

    return timing_data


def execute_silent(
    kernel_client: CheckpointKernelClient,
    code: str,
    timeout: float = 60.0
) -> bool:
    """
    Execute code silently (no cell_id, no timing extraction).

    Returns:
        True if execution succeeded, False on error or timeout
    """
    msg_id = kernel_client.execute(code, silent=True)
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return False

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        if msg_type == "error":
            return False

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Drain shell reply
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass

    return True


def execute_and_get_error(
    kernel_client: CheckpointKernelClient,
    code: str,
    timeout: float = 60.0
) -> Optional[str]:
    """
    Execute code and return error message if any, None if success.
    """
    msg_id = kernel_client.execute(code, silent=True)
    start_time = time.time()
    error_msg = None

    while True:
        if time.time() - start_time > timeout:
            return f"Timeout after {timeout}s"

        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue

        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]

        if msg_type == "error":
            content = msg["content"]
            error_msg = f"{content.get('ename', 'Error')}: {content.get('evalue', 'Unknown')}"

        if msg_type == "status":
            if msg["content"]["execution_state"] == "idle":
                break

    # Drain shell reply
    try:
        kernel_client.get_shell_msg(timeout=1.0)
    except Exception:
        pass

    return error_msg


# ---------------------------------------------------------------------------
# Memory measurement helpers (client-side, injected into kernel)
# ---------------------------------------------------------------------------

_MEMORY_SETUP_CODE = '''
import sys as _sys
import time as _time
import types as _types
import numpy as _np
from pympler import asizeof as _asizeof


_memory_warnings = []

# Primitive types that don't need recursive measurement
_PRIMITIVE_TYPES = (type(None), bool, int, float, complex, str, bytes)

# Names to skip when measuring namespace
_SKIP_NAMES = {'_flowbook_checkpoint', '_flowbook_measure_memory',
               '_measure_object_deep', '_measure_namespace_deep',
               '_get_inner_ndarray', '_collect_user_ns_array_ids',
               '_checkpoint_var_overhead', '_safe_asizeof', '_memory_warnings',
               '_is_shared', '_PRIMITIVE_TYPES', '_SKIP_NAMES',
               '_asizeof', '_time', '_sys', '_types', '_np',
               'In', 'Out', 'get_ipython', 'exit', 'quit'}


def _safe_asizeof(_v):
    """Wrapper for asizeof that handles read-only buffer errors gracefully."""
    try:
        return _asizeof.asizeof(_v)
    except ValueError as _e:
        # Fallback for objects with read-only buffers (e.g., some numpy arrays, LightGBM models)
        _type_name = type(_v).__name__
        _shallow = _sys.getsizeof(_v)
        _memory_warnings.append(f"{_type_name} has read-only buffer, using shallow size ({_shallow} bytes)")
        return _shallow
    except Exception as _e:
        # Catch any other measurement errors
        _type_name = type(_v).__name__
        _shallow = _sys.getsizeof(_v)
        _memory_warnings.append(f"{_type_name} measurement failed ({type(_e).__name__}), using shallow size")
        return _shallow


def _get_inner_ndarray(_arr):
    """Unwrap pandas ExtensionArray to get the underlying ndarray."""
    for _attr in ('_ndarray', '_data'):
        _inner = getattr(_arr, _attr, None)
        if _inner is not None and isinstance(_inner, _np.ndarray):
            return _inner
    if isinstance(_arr, _np.ndarray):
        return _arr
    return None


def _measure_object_deep(_obj, _seen, _shared_array_ids=None):
    """Measure memory of a single object recursively.

    Uses .nbytes for numpy arrays (avoids read-only buffer errors).
    Tracks seen IDs to avoid double-counting shared/aliased objects.
    Handles CoW arrays correctly by checking owndata flag.

    Args:
        _obj: Object to measure
        _seen: Set of object IDs already counted (mutated)
        _shared_array_ids: Set of array IDs that are shared (CoW) - count as 0 data bytes

    Returns:
        Size in bytes (0 if already counted or shared)
    """
    if _shared_array_ids is None:
        _shared_array_ids = set()

    _obj_id = id(_obj)
    if _obj_id in _seen:
        return 0  # Already counted
    _seen.add(_obj_id)

    # Skip modules entirely
    if isinstance(_obj, _types.ModuleType):
        return 0

    # Primitives - just use getsizeof
    if type(_obj) in _PRIMITIVE_TYPES:
        return _sys.getsizeof(_obj)

    # Numpy arrays - use .nbytes (safe, no buffer access)
    if isinstance(_obj, _np.ndarray):
        # Check if this is a view (CoW or slice) - data owned elsewhere
        if not _obj.flags.owndata:
            # It's a view - data buffer is shared, just count wrapper overhead
            return 128
        if _obj_id in _shared_array_ids:
            # Explicitly marked as shared
            return 128
        return _obj.nbytes + 128

    # DataFrames/Series - walk internal arrays
    if hasattr(_obj, '_mgr') and hasattr(_obj._mgr, 'arrays'):
        _total = object.__sizeof__(_obj) + 1024  # DataFrame wrapper overhead
        for _arr in _obj._mgr.arrays:
            _nd = _get_inner_ndarray(_arr)
            _target = _nd if _nd is not None else _arr
            _aid = id(_target)
            if _aid in _seen:
                continue
            _seen.add(_aid)
            if isinstance(_target, _np.ndarray):
                if not _target.flags.owndata or _aid in _shared_array_ids:
                    _total += 128  # View/shared, just wrapper
                else:
                    _total += _target.nbytes + 128
            else:
                _total += _sys.getsizeof(_target)
        return _total

    # Containers - recurse into elements
    if isinstance(_obj, list):
        _total = _sys.getsizeof(_obj)
        for _item in _obj:
            _total += _measure_object_deep(_item, _seen, _shared_array_ids)
        return _total

    if isinstance(_obj, tuple):
        _total = _sys.getsizeof(_obj)
        for _item in _obj:
            _total += _measure_object_deep(_item, _seen, _shared_array_ids)
        return _total

    if isinstance(_obj, dict):
        _total = _sys.getsizeof(_obj)
        for _key, _val in _obj.items():
            _total += _measure_object_deep(_key, _seen, _shared_array_ids)
            _total += _measure_object_deep(_val, _seen, _shared_array_ids)
        return _total

    if isinstance(_obj, set):
        _total = _sys.getsizeof(_obj)
        for _item in _obj:
            _total += _measure_object_deep(_item, _seen, _shared_array_ids)
        return _total

    if isinstance(_obj, frozenset):
        _total = _sys.getsizeof(_obj)
        for _item in _obj:
            _total += _measure_object_deep(_item, _seen, _shared_array_ids)
        return _total

    # Other objects - try asizeof with fallback
    return _safe_asizeof(_obj)


def _measure_namespace_deep(_namespace, _exclude_ids=None, _seen=None, _shared_array_ids=None):
    """Measure total memory of objects in a namespace dict.

    Args:
        _namespace: Dict of {name: object} to measure (e.g., globals(), checkpoint.user_ns)
        _exclude_ids: Set of object IDs to skip entirely (e.g., checkpoint object when measuring user_ns)
        _seen: Set of already-counted object IDs (for cross-namespace dedup)
        _shared_array_ids: Set of array IDs that are shared (CoW) - count as 0 data bytes

    Returns:
        Total size in bytes
    """
    if _seen is None:
        _seen = set()
    if _exclude_ids is None:
        _exclude_ids = set()
    if _shared_array_ids is None:
        _shared_array_ids = set()

    _total = 0
    for _k, _v in _namespace.items():
        # Skip internal names and modules
        if _k.startswith('_') or _k in _SKIP_NAMES:
            continue
        if isinstance(_v, _types.ModuleType):
            continue
        # Skip excluded objects (e.g., checkpoint object)
        if id(_v) in _exclude_ids:
            continue
        _total += _measure_object_deep(_v, _seen, _shared_array_ids)
    return _total


def _collect_user_ns_array_ids():
    """Collect ids of all ndarrays reachable from user namespace variables.

    Returns a set of ndarray ids. Used to detect CoW sharing:
    if a checkpoint array has the same id as a user_ns array, it's shared.
    """
    _ids = set()
    for _k, _v in globals().items():
        if _k.startswith('_') or _k in _SKIP_NAMES or isinstance(_v, _types.ModuleType):
            continue
        if hasattr(_v, '_mgr') and hasattr(_v._mgr, 'arrays'):
            for _arr in _v._mgr.arrays:
                _ids.add(id(_arr))
                _nd = _get_inner_ndarray(_arr)
                if _nd is not None:
                    _ids.add(id(_nd))
        elif isinstance(_v, _np.ndarray):
            _ids.add(id(_v))
    return _ids


def _is_shared(_arr, _nd, _user_ids):
    """Check if an array is shared with user namespace.

    An array is shared if:
    - Its id (or inner ndarray id) matches a user_ns array, OR
    - It's a numpy view (owndata=False), meaning its data lives elsewhere
    """
    if id(_arr) in _user_ids:
        return True
    if _nd is not None and id(_nd) in _user_ids:
        return True
    # numpy views (owndata=False) share their data buffer via .base
    if _nd is not None and isinstance(_nd, _np.ndarray) and not _nd.flags.owndata:
        return True
    if isinstance(_arr, _np.ndarray) and not _arr.flags.owndata:
        return True
    return False


def _checkpoint_var_overhead(_v, _user_ids, _seen=None):
    """Return the true memory overhead for one checkpoint variable.

    For DataFrames/Series: sums nbytes only for arrays that are NOT
    shared with user namespace (by identity or view status).
    For numpy arrays: same check.
    For other types: uses _safe_asizeof.

    _seen tracks ndarray ids already counted to avoid double-counting
    when multiple variables alias the same object (e.g. X = features).
    """
    if _seen is None:
        _seen = set()
    if hasattr(_v, '_mgr') and hasattr(_v._mgr, 'arrays'):
        _overhead = 0
        for _arr in _v._mgr.arrays:
            _nd = _get_inner_ndarray(_arr)
            _aid = id(_nd) if _nd is not None else id(_arr)
            if _aid in _seen:
                continue
            _seen.add(_aid)
            if not _is_shared(_arr, _nd, _user_ids):
                _overhead += _nd.nbytes if _nd is not None else _sys.getsizeof(_arr)
        _overhead += object.__sizeof__(_v) + 1024
        return _overhead
    if isinstance(_v, _np.ndarray):
        _aid = id(_v)
        if _aid in _seen:
            return 128  # wrapper only, data already counted
        _seen.add(_aid)
        if not _is_shared(_v, _v, _user_ids):
            return _v.nbytes + 128
        return 128
    # For other types, use _safe_asizeof (handles read-only buffer errors)
    return _safe_asizeof(_v)


def _flowbook_measure_memory():
    """Measure namespace size with and without checkpoints.

    Returns two totals:
      user_ns_bytes              - globals() with checkpoint objects excluded
                                   (measured via unified deep traversal)
      user_ns_and_checkpoint_bytes - user_ns_bytes + checkpoint overhead
                                   (checkpoint overhead uses ownership-based
                                    accounting to correctly handle CoW/views)

    Returns (user_ns_bytes, user_ns_and_checkpoint_bytes, diagnostics, measurement_time_s).
    """
    global _memory_warnings
    _memory_warnings = []  # Clear warnings from previous measurements
    _t0 = _time.perf_counter()
    _diag = {}

    _cp_obj = None
    if '_flowbook_checkpoint' in globals():
        _cp_obj = globals()['_flowbook_checkpoint']

    # 1. Measure user namespace (excluding checkpoint objects)
    # Build set of object IDs to exclude (checkpoint objects and their contents)
    _exclude_ids = set()
    if _cp_obj is not None:
        _exclude_ids.add(id(_cp_obj))
        if hasattr(_cp_obj, 'saved'):
            _exclude_ids.add(id(_cp_obj.saved))
            for _ckpt in _cp_obj.saved.values():
                _exclude_ids.add(id(_ckpt))
                if hasattr(_ckpt, 'user_ns'):
                    _exclude_ids.add(id(_ckpt.user_ns))

    # Use unified deep measurement (handles read-only buffers, CoW arrays)
    _user_ns_seen = set()
    user_ns_bytes = _measure_namespace_deep(globals(), _exclude_ids=_exclude_ids, _seen=_user_ns_seen)

    # 2. Compute checkpoint overhead by identity-based accounting.
    #    pympler.asizeof overcounts numpy views/CoW copies.  Instead,
    #    we walk checkpoint data manually and only count arrays that are
    #    NOT shared (by object identity) with user namespace.
    #
    #    We compute TWO totals:
    #    - checkpoint_overhead_bytes: per-checkpoint reference accounting
    #      (each checkpoint's data counted independently - may overcount
    #       when the ndarray cache shares the same copy across checkpoints)
    #    - unique_overhead_bytes: actual memory footprint, counting each
    #      unique ndarray object only once across ALL checkpoints
    checkpoint_overhead_bytes = 0
    unique_overhead_bytes = 0
    if _cp_obj is not None and hasattr(_cp_obj, 'saved'):
        _saved = _cp_obj.saved
        _diag['num_checkpoints'] = len(_saved)
        _cp_overheads = {}
        _user_ids = _collect_user_ns_array_ids()

        # Track ndarray ids seen across ALL checkpoints for dedup
        _global_array_seen = set()

        def _container_unique_overhead(_obj, _global_seen, _user_ids):
            """
            Recursively measure container overhead with cross-checkpoint deduplication.

            Walks nested containers (list, tuple, set, dict), tracking each by id().
            Only counts unique objects across all checkpoints.

            Returns the unique overhead in bytes.
            """
            _obj_id = id(_obj)
            if _obj_id in _global_seen:
                return 0  # Already counted in another checkpoint
            _global_seen.add(_obj_id)

            if isinstance(_obj, list):
                _total = _sys.getsizeof(_obj)  # Shallow size of list
                for _item in _obj:
                    if isinstance(_item, (list, tuple, set, dict)):
                        _total += _container_unique_overhead(_item, _global_seen, _user_ids)
                    elif isinstance(_item, _np.ndarray):
                        _aid = id(_item)
                        if _aid not in _global_seen:
                            _global_seen.add(_aid)
                            _total += _item.nbytes + 128
                    elif hasattr(_item, '_mgr') and hasattr(_item._mgr, 'arrays'):
                        # DataFrame inside container
                        for _arr in _item._mgr.arrays:
                            _nd = _get_inner_ndarray(_arr)
                            _aid = id(_nd) if _nd is not None else id(_arr)
                            if _aid not in _global_seen:
                                _global_seen.add(_aid)
                                if not _is_shared(_arr, _nd, _user_ids):
                                    _total += _nd.nbytes if _nd is not None else _sys.getsizeof(_arr)
                        _total += object.__sizeof__(_item) + 1024
                    elif type(_item) not in _PRIMITIVE_TYPES:
                        # Non-primitive leaf - use asizeof
                        _total += _safe_asizeof(_item)
                    # Primitives: already counted in getsizeof of container
                return _total

            elif isinstance(_obj, tuple):
                _total = _sys.getsizeof(_obj)
                for _item in _obj:
                    if isinstance(_item, (list, tuple, set, dict)):
                        _total += _container_unique_overhead(_item, _global_seen, _user_ids)
                    elif isinstance(_item, _np.ndarray):
                        _aid = id(_item)
                        if _aid not in _global_seen:
                            _global_seen.add(_aid)
                            _total += _item.nbytes + 128
                    elif type(_item) not in _PRIMITIVE_TYPES:
                        _total += _safe_asizeof(_item)
                return _total

            elif isinstance(_obj, set):
                _total = _sys.getsizeof(_obj)
                for _item in _obj:
                    # Sets can only contain hashable (usually immutable) items
                    # but check for tuples which can contain mutable nested structures
                    if isinstance(_item, tuple):
                        _total += _container_unique_overhead(_item, _global_seen, _user_ids)
                    elif type(_item) not in _PRIMITIVE_TYPES:
                        _total += _safe_asizeof(_item)
                return _total

            elif isinstance(_obj, dict):
                _total = _sys.getsizeof(_obj)
                for _key, _val in _obj.items():
                    # Keys are hashable, but check for tuples
                    if isinstance(_key, tuple):
                        _total += _container_unique_overhead(_key, _global_seen, _user_ids)
                    elif type(_key) not in _PRIMITIVE_TYPES:
                        _total += _safe_asizeof(_key)
                    # Values can be anything
                    if isinstance(_val, (list, tuple, set, dict)):
                        _total += _container_unique_overhead(_val, _global_seen, _user_ids)
                    elif isinstance(_val, _np.ndarray):
                        _aid = id(_val)
                        if _aid not in _global_seen:
                            _global_seen.add(_aid)
                            _total += _val.nbytes + 128
                    elif hasattr(_val, '_mgr') and hasattr(_val._mgr, 'arrays'):
                        for _arr in _val._mgr.arrays:
                            _nd = _get_inner_ndarray(_arr)
                            _aid = id(_nd) if _nd is not None else id(_arr)
                            if _aid not in _global_seen:
                                _global_seen.add(_aid)
                                if not _is_shared(_arr, _nd, _user_ids):
                                    _total += _nd.nbytes if _nd is not None else _sys.getsizeof(_arr)
                        _total += object.__sizeof__(_val) + 1024
                    elif type(_val) not in _PRIMITIVE_TYPES:
                        _total += _safe_asizeof(_val)
                return _total

            else:
                # Not a container - shouldn't reach here, but fallback
                return _safe_asizeof(_obj)

        for _name, _ckpt in _saved.items():
            _cp_overhead = 0
            _cp_unique = 0
            _seen = set()  # track ndarray ids to avoid double-counting aliases
            if hasattr(_ckpt, 'user_ns'):
                for _k, _v in _ckpt.user_ns.items():
                    _oh = _checkpoint_var_overhead(_v, _user_ids, _seen)
                    _cp_overhead += _oh
                    # Cross-checkpoint dedup: only count unique ndarray objects once
                    if isinstance(_v, _np.ndarray):
                        _aid = id(_v)
                        if _aid not in _global_array_seen:
                            _global_array_seen.add(_aid)
                            _cp_unique += _oh
                        # else: shared with another checkpoint via ndarray cache
                    elif hasattr(_v, '_mgr') and hasattr(_v._mgr, 'arrays'):
                        # DataFrame: dedup individual backing arrays
                        _unique_arr_bytes = 0
                        for _arr in _v._mgr.arrays:
                            _nd = _get_inner_ndarray(_arr)
                            _aid = id(_nd) if _nd is not None else id(_arr)
                            if _aid not in _global_array_seen:
                                _global_array_seen.add(_aid)
                                if not _is_shared(_arr, _nd, _user_ids):
                                    _unique_arr_bytes += _nd.nbytes if _nd is not None else _sys.getsizeof(_arr)
                        _cp_unique += _unique_arr_bytes + object.__sizeof__(_v) + 1024
                    elif isinstance(_v, (list, tuple, set, dict)):
                        # Container: recursively walk and deduplicate nested structures
                        _cp_unique += _container_unique_overhead(_v, _global_array_seen, _user_ids)
                    else:
                        # Other types: use asizeof, track by id to avoid double-counting
                        _aid = id(_v)
                        if _aid not in _global_array_seen:
                            _global_array_seen.add(_aid)
                            _cp_unique += _oh
            # Add overhead for the MemoryCheckpoint object itself
            _ckpt_meta_oh = _sys.getsizeof(_ckpt)
            if hasattr(_ckpt, 'reverse_memo'):
                _ckpt_meta_oh += _sys.getsizeof(_ckpt.reverse_memo)
            _cp_overhead += _ckpt_meta_oh
            _cp_unique += _ckpt_meta_oh
            _cp_overheads[_name] = _cp_overhead
            checkpoint_overhead_bytes += _cp_overhead
            unique_overhead_bytes += _cp_unique

        _diag['checkpoint_overheads'] = _cp_overheads
        _diag['reference_overhead_bytes'] = checkpoint_overhead_bytes
        _diag['unique_overhead_bytes'] = unique_overhead_bytes

        # --- Per-variable overhead for last checkpoint ---
        _last_ckpt = list(_saved.values())[-1] if _saved else None
        _last_name = list(_saved.keys())[-1] if _saved else None
        if _last_ckpt is not None and hasattr(_last_ckpt, 'user_ns'):
            _sharing = []
            _var_overheads = []
            _diag_seen = set()
            for _k, _v in _last_ckpt.user_ns.items():
                _info = {'name': _k, 'type': type(_v).__name__}
                _var_oh = _checkpoint_var_overhead(_v, _user_ids, _diag_seen)
                _var_overheads.append((_k, _var_oh, type(_v).__name__))
                if hasattr(_v, '_mgr') and hasattr(_v._mgr, 'arrays'):
                    _unique = 0
                    _shared = 0
                    for _arr in _v._mgr.arrays:
                        _nd = _get_inner_ndarray(_arr)
                        if _is_shared(_arr, _nd, _user_ids):
                            _shared += 1
                        else:
                            _unique += 1
                    _info['unique_arrays'] = _unique
                    _info['shared_arrays'] = _shared
                elif isinstance(_v, _np.ndarray):
                    _info['shared'] = _is_shared(_v, _v, _user_ids)
                    _info['nbytes'] = int(_v.nbytes)
                _sharing.append(_info)
            _diag['sharing'] = _sharing
            _diag['var_overheads'] = _var_overheads
            # Report reverse_memo size for last checkpoint
            if hasattr(_last_ckpt, 'reverse_memo'):
                _diag['reverse_memo_entries'] = len(_last_ckpt.reverse_memo)
                _diag['reverse_memo_bytes'] = _sys.getsizeof(_last_ckpt.reverse_memo)

    # Use unique (deduplicated) overhead for the primary total so that
    # plots reflect actual memory footprint, not reference-counted refs.
    user_ns_and_checkpoint_bytes = user_ns_bytes + unique_overhead_bytes

    # --- Diagnostics: top user namespace variables by size ---
    _var_sizes = []
    for _k, _v in globals().items():
        if _k.startswith('_') or _k in _SKIP_NAMES or isinstance(_v, _types.ModuleType):
            continue
        # Use _measure_object_deep for consistent measurement with user_ns
        _var_seen = set()
        _var_size = _measure_object_deep(_v, _var_seen)
        _var_sizes.append((_k, _var_size, type(_v).__name__))
    _var_sizes.sort(key=lambda x: x[1], reverse=True)
    _diag['top_vars'] = _var_sizes[:10]

    # Include any warnings from read-only buffer fallbacks
    if _memory_warnings:
        _diag['warnings'] = _memory_warnings[:]

    _elapsed = _time.perf_counter() - _t0
    return (user_ns_bytes, user_ns_and_checkpoint_bytes, _diag, _elapsed)


def _flowbook_checkpoint_details(_top_n=20):
    """Get detailed breakdown of checkpoint contents by variable type.

    Returns dict with:
        num_checkpoints: int
        total_bytes: int - total checkpoint overhead
        by_type: dict mapping type name to {count: int, bytes: int}
        top_variables: list of top N largest variables [{name, type, size_bytes}]
    """
    _result = {
        'num_checkpoints': 0,
        'total_bytes': 0,
        'by_type': {},
        'top_variables': [],
    }

    _cp_obj = None
    if '_flowbook_checkpoint' in globals():
        _cp_obj = globals()['_flowbook_checkpoint']

    if _cp_obj is None or not hasattr(_cp_obj, 'saved'):
        return _result

    _saved = _cp_obj.saved
    _result['num_checkpoints'] = len(_saved)
    _user_ids = _collect_user_ns_array_ids()

    # Collect all variables across all checkpoints
    _all_vars = []  # [(name, type_name, size_bytes, checkpoint_name)]
    _type_agg = {}  # type_name -> {count: int, bytes: int}

    for _ckpt_name, _ckpt in _saved.items():
        if not hasattr(_ckpt, 'user_ns'):
            continue
        _seen = set()
        for _k, _v in _ckpt.user_ns.items():
            _type_name = type(_v).__name__
            _size = _checkpoint_var_overhead(_v, _user_ids, _seen)
            _all_vars.append((_k, _type_name, _size, _ckpt_name))

            # Aggregate by type
            if _type_name not in _type_agg:
                _type_agg[_type_name] = {'count': 0, 'bytes': 0}
            _type_agg[_type_name]['count'] += 1
            _type_agg[_type_name]['bytes'] += _size
            _result['total_bytes'] += _size

    _result['by_type'] = _type_agg

    # Get top N largest variables (sorted by size descending)
    _all_vars.sort(key=lambda x: x[2], reverse=True)
    _result['top_variables'] = [
        {'name': _v[0], 'type': _v[1], 'size_bytes': _v[2]}
        for _v in _all_vars[:_top_n]
    ]

    return _result
'''


def setup_memory_measurement(kernel_client, timeout: float = 60.0) -> bool:
    """Inject pympler measurement helper into the kernel.

    Call once after the kernel is ready.
    Returns True on success.
    """
    return execute_silent(kernel_client, _MEMORY_SETUP_CODE, timeout)


def measure_memory(kernel_client, timeout: float = 300.0) -> dict:
    """Execute the measurement helper and return results.

    Returns dict with user_ns_bytes, user_ns_and_checkpoint_bytes,
    and diagnostics (dict, may be empty).
    On failure returns all zeros with empty diagnostics.
    """
    # Use empty code so the checkpoint kernel treats this as trivial
    # and does NOT take a checkpoint.  The actual work happens in
    # user_expressions which are evaluated without triggering a save.
    msg_id = kernel_client.execute(
        '',
        user_expressions={'_mem': '_flowbook_measure_memory()'},
        silent=True,
    )

    # Wait for idle
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            log('Memory measurement timed out')
            return {'user_ns_bytes': 0, 'user_ns_and_checkpoint_bytes': 0, 'diagnostics': {}}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the shell reply which contains user_expressions
    try:
        reply = kernel_client.get_shell_msg(timeout=5.0)
        expr = reply['content'].get('user_expressions', {}).get('_mem', {})
        if expr.get('status') == 'ok':
            text = expr['data']['text/plain']
            tup = ast.literal_eval(text)
            user_ns_bytes, user_ns_and_checkpoint_bytes, diag, meas_time = tup
            log(f'  Memory measurement took {meas_time*1000:.1f}ms')
            # Log any warnings from read-only buffer fallbacks
            for warning in diag.get('warnings', []):
                log(f'  Memory warning: {warning}')
            return {
                'user_ns_bytes': int(user_ns_bytes),
                'user_ns_and_checkpoint_bytes': int(user_ns_and_checkpoint_bytes),
                'diagnostics': diag,
            }
        else:
            log(f'Memory measurement expression error: {expr}')
    except Exception as e:
        log(f'Memory measurement failed: {e}')

    return {'user_ns_bytes': 0, 'user_ns_and_checkpoint_bytes': 0, 'diagnostics': {}}


def measure_checkpoint_details(kernel_client, timeout: float = 60.0) -> dict:
    """Get detailed breakdown of checkpoint contents by variable type.

    Returns dict with:
        num_checkpoints: int
        total_bytes: int
        by_type: dict mapping type name to {count, bytes}
        top_variables: list of top 20 largest variables [{name, type, size_bytes}]

    On failure returns empty structure.
    """
    msg_id = kernel_client.execute(
        '',
        user_expressions={'_details': '_flowbook_checkpoint_details()'},
        silent=True,
    )

    # Wait for idle
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return {'num_checkpoints': 0, 'total_bytes': 0, 'by_type': {}, 'top_variables': []}
        try:
            msg = kernel_client.get_iopub_msg(timeout=1.0)
        except Exception:
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        if msg['header']['msg_type'] == 'status':
            if msg['content']['execution_state'] == 'idle':
                break

    # Get the shell reply which contains user_expressions
    try:
        reply = kernel_client.get_shell_msg(timeout=5.0)
        expr = reply['content'].get('user_expressions', {}).get('_details', {})
        if expr.get('status') == 'ok':
            text = expr['data']['text/plain']
            return ast.literal_eval(text)
        else:
            log(f'Checkpoint details expression error: {expr}')
    except Exception as e:
        log(f'Checkpoint details failed: {e}')

    return {'num_checkpoints': 0, 'total_bytes': 0, 'by_type': {}, 'top_variables': []}


def run_rerun_trials(
    kernel_client: CheckpointKernelClient,
    cells: List[Cell],
    num_reruns: int,
    num_modifications: int,
    output_file: TextIO,
    cell_timeout: float = 60.0,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Run rerun trials on randomly selected cells (with replacement).

    For each rerun:
    1. Pick a random cell (with replacement)
    2. Restore the post-checkpoint for that cell
    3. Randomly modify the namespace
    4. Trigger a checkpoint and measure timing

    Keeps trying until num_reruns successful measurements are collected.

    Args:
        kernel_client: Connected kernel client
        cells: List of cells that were executed
        num_reruns: Number of successful rerun measurements to collect
        num_modifications: Number of variables to modify per rerun
        output_file: File to write CSV output
        cell_timeout: Timeout per rerun in seconds
        seed: Random seed for reproducibility

    Returns:
        List of timing dicts for each rerun
    """
    if seed is not None:
        random.seed(seed)

    writer = csv.writer(output_file)
    writer.writerow(["cell_id", "commit_time_s", "num_modifications"])

    results = []
    attempts = 0
    max_attempts = num_reruns * 3  # Give up after 3x attempts

    while len(results) < num_reruns and attempts < max_attempts:
        attempts += 1
        # Pick a random cell (with replacement)
        cell = random.choice(cells)
        log(f"Rerun {len(results)+1}/{num_reruns} (attempt {attempts}): Cell {cell.cell_id}...")

        # 1. Restore checkpoint
        restore_code = f'_flowbook_checkpoint.restore("post_{cell.cell_id}", globals())'
        error = execute_and_get_error(kernel_client, restore_code, cell_timeout)
        if error:
            log(f"  FAILED restoring checkpoint: {error}")
            continue

        # 2. Modify namespace
        modify_code = f'''
from flowbook.testing.performance import _randomly_modify_namespace
_randomly_modify_namespace(globals(), {num_modifications},
    exclude={{"__builtins__", "__name__", "__doc__", "_flowbook_checkpoint"}})
'''
        error = execute_and_get_error(kernel_client, modify_code, cell_timeout)
        if error:
            log(f"  FAILED modifying namespace: {error}")
            continue

        # 3. Trigger checkpoint and extract timing
        trigger_cell = Cell(
            cell_id=f"{cell.cell_id}_rerun_{attempts}",
            source="# __flowbook_force_checkpoint__\nz=100",
            cell_type="code",
            index=-1,
        )
        timing = execute_cell_and_extract_timing(
            kernel_client,
            trigger_cell,
            timeout=cell_timeout
        )

        if timing.get("error"):
            log(f"  FAILED checkpoint: {timing['error'][:200]}")
            continue

        commit_time = timing.get("commit_time_s", 0)
        writer.writerow([
            cell.cell_id,
            commit_time,
            num_modifications,
        ])
        output_file.flush()

        results.append({
            "cell_id": cell.cell_id,
            "commit_time_s": commit_time,
            "num_modifications": num_modifications,
        })

        log(f"  Commit: {commit_time*1000:.1f}ms")

    if len(results) < num_reruns:
        log(f"WARNING: Only collected {len(results)}/{num_reruns} successful reruns after {attempts} attempts")

    return results


def run_benchmark(
    notebook_path: str,
    output_file: Optional[TextIO] = None,
    cell_timeout: float = 300.0,
    num_reruns: int = 0,
    rerun_modifications: int = 3,
    rerun_output_file: Optional[TextIO] = None,
    rerun_seed: Optional[int] = None,
) -> List[dict]:
    """
    Run benchmark on a notebook.

    Args:
        notebook_path: Path to .ipynb file
        output_file: File to write CSV output (default: stdout)
        cell_timeout: Timeout per cell in seconds
        num_reruns: Number of rerun measurements to take (0 = skip)
        rerun_modifications: Number of variables to modify per rerun
        rerun_output_file: File to write rerun CSV output
        rerun_seed: Random seed for rerun selection

    Returns:
        List of timing dicts for each cell
    """
    if output_file is None:
        output_file = sys.stdout

    # Load notebook cells
    cells = load_notebook(notebook_path)
    log(f"Loaded {len(cells)} code cells from {notebook_path}")

    # Start kernel
    kernel_manager = None
    kernel_client = None
    results = []
    executed_cells = []

    try:
        log("Starting checkpoint kernel...")
        kernel_manager, kernel_client = create_checkpoint_kernel()
        log("Kernel ready")

        # Setup memory measurement
        if setup_memory_measurement(kernel_client):
            log("Memory measurement helper injected")
        else:
            log("WARNING: Failed to inject memory measurement helper")

        # Write CSV header
        writer = csv.writer(output_file)
        writer.writerow([
            "cell_id", "execution_count", "cell_runtime_s", "commit_time_s",
            "user_ns_bytes", "user_ns_and_checkpoint_bytes",
        ])

        # Execute each cell
        for i, cell in enumerate(cells):
            log(f"Executing cell {i+1}/{len(cells)} ({cell.cell_id})...")
            timing = execute_cell_and_extract_timing(kernel_client, cell, cell_timeout)
            results.append(timing)

            # Write CSV row
            if timing.get("error"):
                log(f"  Error: {timing['error'][:100]}...")
            else:
                mem = measure_memory(kernel_client)
                writer.writerow([
                    cell.cell_id,
                    timing.get("execution_count", ""),
                    timing.get("cell_runtime_s", ""),
                    timing.get("commit_time_s", ""),
                    mem["user_ns_bytes"],
                    mem["user_ns_and_checkpoint_bytes"],
                ])
                mb = 1024 * 1024
                overhead = mem['user_ns_and_checkpoint_bytes'] - mem['user_ns_bytes']
                log(f"  Run: {timing.get('cell_runtime_s', 0)*1000:.1f}ms, Commit: {timing.get('commit_time_s', 0)*1000:.1f}ms")
                ref_oh = mem.get('diagnostics', {}).get('reference_overhead_bytes', overhead)
                log(f"  Memory: user_ns={mem['user_ns_bytes']/mb:,.1f}MB, "
                    f"checkpoint_overhead={overhead/mb:,.1f}MB"
                    + (f" (ref={ref_oh/mb:,.1f}MB)" if ref_oh != overhead else ""))

                # Log diagnostics if available
                diag = mem.get('diagnostics', {})
                if diag:
                    if 'num_checkpoints' in diag:
                        log(f"  Diagnostics: {diag['num_checkpoints']} checkpoints")
                    if 'checkpoint_overheads' in diag:
                        for cp_name, cp_oh in diag['checkpoint_overheads'].items():
                            log(f"    checkpoint '{cp_name}': {cp_oh/mb:,.1f}MB owned")
                    if 'var_overheads' in diag:
                        log(f"  Per-variable overhead (last checkpoint):")
                        for var_name, var_oh, var_type in diag['var_overheads']:
                            if var_oh > 100_000:
                                log(f"    {var_name} ({var_type}): {var_oh/mb:,.1f}MB")
                    if 'reverse_memo_entries' in diag:
                        rm_entries = diag['reverse_memo_entries']
                        rm_bytes = diag['reverse_memo_bytes']
                        log(f"  reverse_memo: {rm_entries:,} entries, {rm_bytes/mb:,.1f}MB hash table")
                    if 'sharing' in diag:
                        log(f"  Array sharing (last checkpoint):")
                        for info in diag['sharing']:
                            name = info['name']
                            vtype = info['type']
                            if 'unique_arrays' in info:
                                log(f"    {name} ({vtype}): unique={info['unique_arrays']}, shared={info['shared_arrays']}")
                            elif 'shared' in info:
                                nbytes = info.get('nbytes', 0)
                                status = 'shared' if info['shared'] else 'UNIQUE'
                                log(f"    {name} ({vtype}): {status} {nbytes/mb:.1f}MB")
                            else:
                                log(f"    {name} ({vtype})")
                    if 'top_vars' in diag:
                        log(f"  Top user namespace variables:")
                        for var_name, var_size, var_type in diag['top_vars']:
                            log(f"    {var_name} ({var_type}): {var_size/mb:,.1f}MB")

                executed_cells.append(cell)

        # Flush output
        output_file.flush()

        # Run rerun trials if requested
        if num_reruns > 0 and executed_cells and rerun_output_file is not None:
            log(f"\nStarting {num_reruns} rerun measurements...")
            run_rerun_trials(
                kernel_client,
                executed_cells,
                num_reruns,
                rerun_modifications,
                rerun_output_file,
                cell_timeout=60.0,
                seed=rerun_seed,
            )

        return results

    finally:
        cleanup_kernel(kernel_manager, kernel_client)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark checkpoint kernel execution and commit times"
    )
    parser.add_argument(
        "notebook",
        help="Path to notebook file (.ipynb)"
    )
    parser.add_argument(
        "-o", "--output",
        default="flowbook_timings.csv",
        help="Output CSV file (default: flowbook_timings.csv)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout per cell in seconds (default: 300)"
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=0,
        help="Number of rerun measurements to take (default: 0 = skip)"
    )
    parser.add_argument(
        "--modifications",
        type=int,
        default=3,
        help="Number of variables to modify per rerun (default: 3)"
    )
    parser.add_argument(
        "--rerun-output",
        default="flowbook_rerun_timings.csv",
        help="Output CSV file for rerun timings (default: flowbook_rerun_timings.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for rerun cell selection (default: None)"
    )

    args = parser.parse_args()

    output_file = None
    rerun_output_file = None
    try:
        output_file = open(args.output, "w", newline="")
        log(f"Writing results to {args.output}")

        if args.reruns > 0:
            rerun_output_file = open(args.rerun_output, "w", newline="")
            log(f"Writing rerun results to {args.rerun_output}")

        run_benchmark(
            args.notebook,
            output_file,
            args.timeout,
            num_reruns=args.reruns,
            rerun_modifications=args.modifications,
            rerun_output_file=rerun_output_file,
            rerun_seed=args.seed,
        )
    finally:
        if output_file:
            output_file.close()
        if rerun_output_file:
            rerun_output_file.close()


if __name__ == "__main__":
    main()
