"""
Scalene-based memory tracking for FlowBook kernel.

This module provides a wrapper around Scalene's memory profiling API for
precise memory measurement in the FlowBook kernel. Scalene tracks memory
at the native malloc/free level, providing more accurate measurements than
Python-only approaches like pympler.

Requirements:
- Scalene must be installed
- The kernel must be run under Scalene (using `scalene --memory` command)
  which properly initializes memory tracking.

Usage:
    from flowbook.kernel_support.scalene_memory import ScaleneMemoryTracker

    # Check if available
    if ScaleneMemoryTracker.is_available():
        # Enable tracking
        ScaleneMemoryTracker.start()

        # ... execute code ...

        # Get memory stats
        stats = ScaleneMemoryTracker.get_memory()
        print(f"Current: {stats['current_footprint_mb']:.1f} MB")

        # Disable tracking
        ScaleneMemoryTracker.stop()
"""

import os
import platform
import sys
from typing import Any, Dict, Optional


def get_object_size(obj: Any) -> int:
    """Calculate the memory size of an object in bytes.

    Uses type-specific methods for accurate measurement:
    - numpy arrays: nbytes attribute
    - pandas DataFrame/Series: memory_usage(deep=True)
    - Other objects: sys.getsizeof()

    Note: This measures the size of the object itself, not the size
    needed to copy it (which includes internal buffers and temporaries).

    Args:
        obj: Object to measure

    Returns:
        Size in bytes
    """
    try:
        # numpy arrays - use nbytes for accurate native memory size
        if hasattr(obj, 'nbytes') and hasattr(obj, 'dtype'):
            return int(obj.nbytes)

        # pandas DataFrame/Series - use memory_usage with deep=True
        if hasattr(obj, 'memory_usage'):
            try:
                # DataFrame
                if hasattr(obj.memory_usage, '__call__'):
                    usage = obj.memory_usage(deep=True)
                    if hasattr(usage, 'sum'):
                        return int(usage.sum())
                    return int(usage)
            except Exception:
                pass

        # For other types, use sys.getsizeof as a baseline
        # This won't include nested objects but gives a reasonable estimate
        return sys.getsizeof(obj)
    except Exception:
        return 0


class ScaleneMemoryTracker:
    """Wrapper for Scalene's memory profiling API.

    Provides methods to enable/disable memory tracking and retrieve
    memory statistics including total footprint, allocations, and GPU usage.

    This class uses class methods to maintain global tracking state,
    matching how Scalene maintains global profiler state.
    """

    # Internal state
    _tracking_enabled: bool = False
    _scalene_available: Optional[bool] = None  # Cached availability check

    @classmethod
    def is_available(cls) -> bool:
        """Check if Scalene is properly initialized and available.

        Scalene must be properly initialized (running under `scalene` command)
        for memory tracking to work. This method checks:
        1. Platform-specific preload environment variables
        2. Whether Scalene can be imported
        3. Whether Scalene is initialized (not just preloaded)

        Returns:
            True if Scalene is available for memory tracking, False otherwise.
        """
        # Use cached result if available
        if cls._scalene_available is not None:
            return cls._scalene_available

        # Check platform-specific preload env vars
        system = platform.system()
        if system == "Darwin":
            preload = os.environ.get("DYLD_INSERT_LIBRARIES", "")
            if "libscalene" not in preload:
                cls._scalene_available = False
                return False
        elif system == "Linux":
            preload = os.environ.get("LD_PRELOAD", "")
            if "libscalene" not in preload:
                cls._scalene_available = False
                return False
        elif system == "Windows":
            # Windows uses a different mechanism - check if Scalene is importable
            pass

        # Try to import and use Scalene
        try:
            from scalene.scalene_profiler import Scalene
            cls._scalene_available = True
            return True
        except ImportError:
            cls._scalene_available = False
            return False

    @classmethod
    def start(cls) -> bool:
        """Enable memory tracking.

        Note: When running under `scalene --memory`, tracking is always active.
        This method just sets a flag to indicate we want to use the tracking data.

        Returns:
            True if tracking was successfully enabled, False otherwise.
        """
        if not cls.is_available():
            return False

        cls._tracking_enabled = True
        return True

    @classmethod
    def stop(cls) -> bool:
        """Disable memory tracking.

        Note: This just clears our tracking flag. Scalene continues to
        track allocations, but we won't use the data.

        Returns:
            True if tracking was successfully disabled, False otherwise.
        """
        if not cls.is_available():
            return False

        cls._tracking_enabled = False
        return True

    @classmethod
    def is_tracking(cls) -> bool:
        """Check if memory tracking is currently enabled.

        Returns:
            True if tracking is enabled and Scalene is available.
        """
        return cls._tracking_enabled and cls.is_available()

    @classmethod
    def get_malloc_samples(cls) -> float:
        """Get current memory footprint in MB for delta measurement.

        This is used for delta measurement during deepcopy operations.
        By recording the value before and after an operation, you can
        compute the memory allocated by that operation.

        Note: We use current_footprint instead of total_memory_malloc_samples
        because malloc_samples is not reliably populated in all Scalene modes,
        while current_footprint always reflects the actual heap state.

        Returns:
            Current memory footprint (in MB).
            Returns 0.0 if Scalene is not available.
        """
        if not cls.is_available():
            return 0.0

        try:
            from scalene.scalene_profiler import Scalene
            stats = Scalene._Scalene__stats
            # Use current_footprint which is reliably tracked
            return stats.memory_stats.current_footprint
        except Exception:
            return 0.0

    @classmethod
    def get_free_samples(cls) -> float:
        """Get current total free samples in MB.

        Returns:
            Total memory freed (in MB) since tracking started.
            Returns 0.0 if Scalene is not available.
        """
        if not cls.is_available():
            return 0.0

        try:
            from scalene.scalene_profiler import Scalene
            stats = Scalene._Scalene__stats
            return stats.memory_stats.total_memory_free_samples
        except Exception:
            return 0.0

    @classmethod
    def get_memory(cls) -> Dict[str, Any]:
        """Get current memory stats including GPU.

        Returns comprehensive memory statistics from Scalene including
        current footprint, peak usage, allocations/frees, and GPU stats.

        Returns:
            Dictionary with keys:
                - current_footprint_mb: Current memory footprint
                - max_footprint_mb: Peak memory footprint
                - total_malloc_mb: Total memory allocated
                - total_free_mb: Total memory freed
                - net_allocation_mb: Net allocation (malloc - free)
                - gpu_samples: Total GPU compute samples
                - gpu_mem_samples: Total GPU memory samples
                - available: Whether Scalene is available

            If Scalene is not available, returns a dict with available=False
            and zero values for all metrics.
        """
        if not cls.is_available():
            return {
                "current_footprint_mb": 0.0,
                "max_footprint_mb": 0.0,
                "total_malloc_mb": 0.0,
                "total_free_mb": 0.0,
                "net_allocation_mb": 0.0,
                "gpu_samples": 0.0,
                "gpu_mem_samples": 0.0,
                "available": False,
            }

        try:
            from scalene.scalene_profiler import Scalene
            stats = Scalene._Scalene__stats
            mem = stats.memory_stats
            gpu = stats.gpu_stats

            total_malloc = mem.total_memory_malloc_samples
            total_free = mem.total_memory_free_samples

            return {
                "current_footprint_mb": mem.current_footprint,
                "max_footprint_mb": mem.max_footprint,
                "total_malloc_mb": total_malloc,
                "total_free_mb": total_free,
                "net_allocation_mb": total_malloc - total_free,
                "gpu_samples": gpu.total_gpu_samples if hasattr(gpu, 'total_gpu_samples') else 0.0,
                "gpu_mem_samples": gpu.total_gpu_mem_samples if hasattr(gpu, 'total_gpu_mem_samples') else 0.0,
                "available": True,
            }
        except Exception as e:
            return {
                "current_footprint_mb": 0.0,
                "max_footprint_mb": 0.0,
                "total_malloc_mb": 0.0,
                "total_free_mb": 0.0,
                "net_allocation_mb": 0.0,
                "gpu_samples": 0.0,
                "gpu_mem_samples": 0.0,
                "available": False,
                "error": str(e),
            }

    @classmethod
    def reset(cls) -> bool:
        """Clear all memory statistics.

        Resets Scalene's internal statistics counters. Useful when you
        want to start fresh measurement from a known baseline.

        Returns:
            True if reset was successful, False otherwise.
        """
        if not cls.is_available():
            return False

        try:
            from scalene.scalene_profiler import Scalene
            stats = Scalene._Scalene__stats
            stats.clear_all()
            return True
        except Exception:
            return False

    @classmethod
    def format_stats(cls, stats: Optional[Dict[str, Any]] = None) -> str:
        """Format memory statistics as a human-readable string.

        Args:
            stats: Optional stats dict from get_memory(). If None,
                   will call get_memory() to get current stats.

        Returns:
            Formatted string showing memory statistics.
        """
        if stats is None:
            stats = cls.get_memory()

        if not stats.get("available", False):
            return "Scalene memory tracking not available"

        lines = [
            f"Memory: current={stats['current_footprint_mb']:.1f} MB, "
            f"max={stats['max_footprint_mb']:.1f} MB, "
            f"net={stats['net_allocation_mb']:+.1f} MB"
        ]

        gpu_samples = stats.get("gpu_samples", 0)
        gpu_mem = stats.get("gpu_mem_samples", 0)
        if gpu_samples > 0 or gpu_mem > 0:
            lines.append(f"GPU: samples={gpu_samples:.0f}, mem={gpu_mem:.1f}")

        return "\n".join(lines)
