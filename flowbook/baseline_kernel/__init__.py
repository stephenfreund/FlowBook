"""
Baseline Kernel - For comparison testing with FlowBook.

This package provides a baseline IPython kernel for fair A/B comparison
with FlowBook. It has standard IPython behavior without reproducibility
tracking or checkpointing.
"""

import os

from flowbook.kernel_support.install import install_kernel

from flowbook.baseline_kernel.baseline_kernel import (
    BaselineKernel,
)

__all__ = [
    "BaselineKernel",
    "install_baseline_kernel",
]


def install_baseline_kernel() -> str:
    """
    Install the Baseline kernel spec.

    Returns:
        Path to installed kernel spec directory
    """
    return install_kernel(os.path.dirname(__file__), "baseline_kernel")


# Install kernel on import
try:
    install_baseline_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
