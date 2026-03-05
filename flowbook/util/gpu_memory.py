"""GPU memory measurement utilities using pynvml.

Measures GPU memory allocated to the current process via NVIDIA Management Library.

Usage:
    from flowbook.util.gpu_memory import get_gpu_memory_mb, has_gpu

    if has_gpu():
        print(f"GPU memory: {get_gpu_memory_mb():.1f} MB")
"""

import os
from typing import Optional

# Module-level state for lazy initialization
_pynvml = None
_pynvml_available: Optional[bool] = None
_current_pid: Optional[int] = None


def _init_pynvml() -> bool:
    """Lazily initialize pynvml. Returns True if available."""
    global _pynvml, _pynvml_available, _current_pid

    if _pynvml_available is not None:
        return _pynvml_available

    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        _current_pid = os.getpid()
        _pynvml_available = True
    except (ImportError, Exception):
        _pynvml_available = False

    return _pynvml_available


def _get_physical_gpu_index() -> int:
    """Get the physical GPU index respecting CUDA_VISIBLE_DEVICES.

    CUDA_VISIBLE_DEVICES remaps logical device 0 to a physical device.
    pynvml bypasses CUDA and needs the physical index directly.

    Returns:
        Physical GPU index (from CUDA_VISIBLE_DEVICES if set, else 0)
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # Take first GPU in the list (e.g., "1,2" -> 1)
        try:
            return int(cuda_visible.split(",")[0])
        except (ValueError, IndexError):
            pass
    return 0


def get_gpu_memory_mb() -> float:
    """Get current process GPU memory usage in MB.

    Uses pynvml to query per-process GPU memory allocation. This captures
    all GPU memory allocated by the process regardless of framework
    (CuPy, PyTorch, cuDF/RMM, CatBoost, XGBoost, etc.).

    Respects CUDA_VISIBLE_DEVICES to query the correct physical GPU when
    running under SLURM or other job schedulers that assign specific GPUs.

    Returns GPU memory in MB, or 0.0 if unavailable.
    """
    if not _init_pynvml():
        return 0.0

    try:
        gpu_index = _get_physical_gpu_index()
        handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        # Query compute running processes
        processes = _pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for proc in processes:
            if proc.pid == _current_pid:
                return proc.usedGpuMemory / (1024 * 1024)
        # Also try graphics processes (some frameworks use this)
        processes = _pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        for proc in processes:
            if proc.pid == _current_pid:
                return proc.usedGpuMemory / (1024 * 1024)
    except Exception:
        pass

    return 0.0


def has_gpu() -> bool:
    """Check if GPU memory measurement is available.

    Returns True if pynvml is available and initialized.
    """
    return _init_pynvml()
