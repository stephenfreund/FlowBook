"""GPU memory measurement utilities.

Provides GPU memory measurement supporting multiple frameworks:
- RAPIDS/cuDF via RMM (RAPIDS Memory Manager)
- CuPy memory pools
- PyTorch CUDA allocator
- General NVIDIA GPU via pynvml

Falls back gracefully to 0.0 when no GPU framework is available.

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


def _get_rmm_memory_mb() -> Optional[float]:
    """Get GPU memory from RMM (RAPIDS Memory Manager)."""
    try:
        import rmm
        # Get the current device resource and check allocated bytes
        mr = rmm.mr.get_current_device_resource()
        # Try to get allocated bytes - works with pool and tracking resources
        if hasattr(mr, 'get_allocated_bytes'):
            return mr.get_allocated_bytes() / (1024 * 1024)
        # For statistics resource wrapper
        if hasattr(mr, 'allocation_counts'):
            stats = mr.allocation_counts
            if hasattr(stats, 'current_bytes'):
                return stats.current_bytes / (1024 * 1024)
    except (ImportError, Exception):
        pass
    return None


def _get_cupy_memory_mb() -> Optional[float]:
    """Get GPU memory from CuPy memory pool."""
    try:
        import cupy
        pool = cupy.get_default_memory_pool()
        return pool.used_bytes() / (1024 * 1024)
    except (ImportError, Exception):
        pass
    return None


def _get_torch_memory_mb() -> Optional[float]:
    """Get GPU memory from PyTorch CUDA allocator."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory allocated by PyTorch on current device
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except (ImportError, Exception):
        pass
    return None


def _get_pynvml_process_memory_mb() -> Optional[float]:
    """Get per-process GPU memory via pynvml."""
    if not _init_pynvml():
        return None

    try:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(0)  # Primary GPU
        # Try compute running processes first
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
    return None


def get_gpu_memory_mb() -> float:
    """Get current process GPU memory usage in MB.

    Tries multiple sources and takes the maximum to capture all allocations:
    1. RMM (RAPIDS Memory Manager) - for cuDF/RAPIDS
    2. CuPy memory pool - for CuPy arrays
    3. PyTorch CUDA allocator - for PyTorch tensors
    4. pynvml per-process query - for CatBoost/XGBoost/LightGBM and others

    Different frameworks use different GPU allocators, so we query all and
    take the maximum to capture whatever is using GPU memory.

    Returns maximum detected GPU memory across all sources, or 0.0 if unavailable.
    """
    measurements = []

    # Try RMM (RAPIDS)
    rmm_mb = _get_rmm_memory_mb()
    if rmm_mb is not None and rmm_mb > 0:
        measurements.append(('rmm', rmm_mb))

    # Try CuPy
    cupy_mb = _get_cupy_memory_mb()
    if cupy_mb is not None and cupy_mb > 0:
        measurements.append(('cupy', cupy_mb))

    # Try PyTorch
    torch_mb = _get_torch_memory_mb()
    if torch_mb is not None and torch_mb > 0:
        measurements.append(('torch', torch_mb))

    # Try pynvml - always try this as it captures CatBoost/XGBoost/LightGBM
    pynvml_mb = _get_pynvml_process_memory_mb()
    if pynvml_mb is not None and pynvml_mb > 0:
        measurements.append(('pynvml', pynvml_mb))

    # Return maximum - different allocators may report different subsets
    # Taking max captures the largest view of GPU memory usage
    if measurements:
        return max(mb for _, mb in measurements)

    return 0.0


def get_gpu_memory_detailed() -> dict:
    """Get detailed GPU memory usage from all sources.

    Returns dict with:
        - total_mb: Maximum across all sources
        - by_source: Dict mapping source name to MB
        - sources: List of sources that reported memory
    """
    by_source = {}

    rmm_mb = _get_rmm_memory_mb()
    if rmm_mb is not None and rmm_mb > 0:
        by_source['rmm'] = rmm_mb

    cupy_mb = _get_cupy_memory_mb()
    if cupy_mb is not None and cupy_mb > 0:
        by_source['cupy'] = cupy_mb

    torch_mb = _get_torch_memory_mb()
    if torch_mb is not None and torch_mb > 0:
        by_source['torch'] = torch_mb

    pynvml_mb = _get_pynvml_process_memory_mb()
    if pynvml_mb is not None and pynvml_mb > 0:
        by_source['pynvml'] = pynvml_mb

    total_mb = max(by_source.values()) if by_source else 0.0

    return {
        'total_mb': total_mb,
        'by_source': by_source,
        'sources': list(by_source.keys()),
    }


def has_gpu() -> bool:
    """Check if GPU memory measurement is available.

    Returns True if any GPU memory source is available.
    """
    # Check pynvml
    if _init_pynvml():
        return True

    # Check if any GPU framework is importable
    try:
        import rmm  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    return False
