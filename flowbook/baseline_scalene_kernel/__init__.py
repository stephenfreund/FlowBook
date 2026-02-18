"""
Baseline Scalene Kernel - For memory comparison testing.

This package provides a baseline IPython kernel with Scalene memory tracking
capabilities that mirror FlowBook's implementation for fair A/B comparison.

Two kernel specs are provided:
- baseline_scalene_kernel: Standard kernel (timing measurements)
- baseline_scalene_preload_kernel: Kernel with Scalene preloaded (memory measurements)
"""

import os

from flowbook.kernel_support.install import install_kernel

from flowbook.baseline_scalene_kernel.baseline_scalene_kernel import (
    BaselineScaleneKernel,
)

__all__ = [
    "BaselineScaleneKernel",
    "install_baseline_scalene_kernel",
    "install_baseline_scalene_preload_kernel",
]


def install_baseline_scalene_kernel() -> str:
    """
    Install the Baseline Scalene kernel spec (without Scalene preload).

    Returns:
        Path to installed kernel spec directory
    """
    return install_kernel(os.path.dirname(__file__), "baseline_scalene_kernel")


def install_baseline_scalene_preload_kernel() -> str:
    """
    Install the Baseline Scalene kernel spec WITH Scalene library preloaded.

    This kernel spec uses launch_with_scalene.py to ensure Scalene's native
    library is preloaded for accurate memory tracking.

    Returns:
        Path to installed kernel spec directory
    """
    return install_kernel(
        os.path.dirname(__file__),
        "baseline_scalene_preload_kernel",
        spec_subdir="scalene_kernelspec"
    )


# Install kernels on import
try:
    install_baseline_scalene_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails

try:
    install_baseline_scalene_preload_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
