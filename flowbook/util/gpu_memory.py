"""GPU memory measurement utilities using pynvml.

Provides per-process GPU memory measurement for NVIDIA GPUs. Falls back
gracefully to 0.0 when pynvml is unavailable or no GPU is present.

Usage:
    from flowbook.util.gpu_memory import get_gpu_memory_mb, has_gpu

    if has_gpu():
        print(f"GPU memory: {get_gpu_memory_mb():.1f} MB")
"""

import os
from typing import Optional

# Module-level state for lazy initialization
_pynvml = None
_gpu_available: Optional[bool] = None
_current_pid: Optional[int] = None


def _init_pynvml() -> bool:
    """Lazily initialize pynvml. Returns True if GPU is available."""
    global _pynvml, _gpu_available, _current_pid

    if _gpu_available is not None:
        return _gpu_available

    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        _current_pid = os.getpid()
        _gpu_available = True
    except (ImportError, Exception):
        _gpu_available = False

    return _gpu_available


def get_gpu_memory_mb() -> float:
    """Get current process GPU memory usage in MB.

    Returns the GPU memory used by the current process on GPU 0.
    Returns 0.0 if pynvml is unavailable, no GPU is present,
    or the current process is not using the GPU.
    """
    if not _init_pynvml():
        return 0.0

    try:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(0)  # Primary GPU
        # Get per-process memory using nvmlDeviceGetComputeRunningProcesses
        processes = _pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for proc in processes:
            if proc.pid == _current_pid:
                return proc.usedGpuMemory / (1024 * 1024)  # Convert bytes to MB
        return 0.0  # Process not using GPU
    except Exception:
        return 0.0


def has_gpu() -> bool:
    """Check if GPU memory measurement is available.

    Returns True if pynvml is available and at least one NVIDIA GPU is present.
    """
    return _init_pynvml()
