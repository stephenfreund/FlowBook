"""
Checkpoint Kernel - Execution and checkpoint timing measurement.

This package provides:
- CheckpointKernel: IPython kernel that measures execution and checkpoint time
- CheckpointKernelClient: Client that sends cell_id with executions
"""

import json
import os
import sys

from jupyter_client.kernelspec import KernelSpecManager

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
    ksm = KernelSpecManager()

    # Path to our kernel spec
    spec_dir = os.path.join(os.path.dirname(__file__), "kernelspec")

    # Install it
    dest = ksm.install_kernel_spec(
        spec_dir,
        kernel_name="checkpoint_kernel",
        user=True,
        replace=True,
    )

    # Update argv to use correct python
    kernel_json = os.path.join(dest, "kernel.json")
    with open(kernel_json, "r") as f:
        spec = json.load(f)

    spec["argv"][0] = sys.executable

    with open(kernel_json, "w") as f:
        json.dump(spec, f, indent=2)

    return dest


# Install kernel on import
try:
    install_checkpoint_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
