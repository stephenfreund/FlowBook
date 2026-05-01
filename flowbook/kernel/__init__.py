"""
FlowBook Kernel - Reproducibility enforcement.

This package provides:
- FlowbookKernel: IPython kernel with reproducibility enforcement
- FlowbookKernelClient: Client that sends cell order with executions
- ReproducibilityEnforcer: Core reproducibility logic (reusable)
"""

import os

from flowbook.kernel_support.install import install_kernel

from flowbook.kernel.flowbook_client import FlowbookKernelClient
from flowbook.kernel.flowbook_kernel import FlowbookKernel
from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer

__all__ = [
    "FlowbookKernel",
    "FlowbookKernelClient",
    "ReproducibilityEnforcer",
    "install_flowbook_kernel",
]


def install_flowbook_kernel() -> str:
    """
    Install the FlowBook kernel spec.

    Returns:
        Path to installed kernel spec directory
    """
    return install_kernel(os.path.dirname(__file__), "flowbook_kernel")


# Install kernels on import
try:
    install_flowbook_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
