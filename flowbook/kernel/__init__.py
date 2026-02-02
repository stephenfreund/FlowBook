"""
FlowBook Kernel - Reproducibility enforcement.

This package provides:
- FlowbookKernel: IPython kernel with reproducibility enforcement
- FlowbookKernelClient: Client that sends cell order with executions
- ReproducibilityEnforcer: Core reproducibility logic (reusable)
"""

import json
import os
import sys

from jupyter_client.kernelspec import KernelSpecManager

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
    ksm = KernelSpecManager()

    # Path to our kernel spec
    spec_dir = os.path.join(os.path.dirname(__file__), "kernelspec")

    # Install it
    dest = ksm.install_kernel_spec(
        spec_dir,
        kernel_name="flowbook_kernel",
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
    install_flowbook_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
