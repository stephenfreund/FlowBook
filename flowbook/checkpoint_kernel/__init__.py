"""
Checkpoint Kernel - Execution and checkpoint timing measurement.

This package provides:
- CheckpointKernel: IPython kernel that measures execution and checkpoint time
- CheckpointKernelClient: Client that sends cell_id with executions
"""

import os

from flowbook.kernel_support.install import install_kernel

from flowbook.checkpoint_kernel.checkpoint_client import CheckpointKernelClient
from flowbook.checkpoint_kernel.checkpoint_kernel import CheckpointKernel

__all__ = [
    "CheckpointKernel",
    "CheckpointKernelClient",
    "install_checkpoint_kernel",
]


def install_checkpoint_kernel() -> str:
    """
    Install the Checkpoint kernel spec.

    Returns:
        Path to installed kernel spec directory
    """
    return install_kernel(os.path.dirname(__file__), "checkpoint_kernel")


# Install kernel on import
try:
    install_checkpoint_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
