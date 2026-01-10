"""
FlowBook SDC Kernel - Sequential Dataflow Consistency enforcement.

This package provides:
- FlowbookSDCKernel: IPython kernel with SDC enforcement
- FlowbookSDCKernelClient: Client that sends cell order with executions
- SDCEnforcer: Core SDC logic (reusable)
"""

import json
import os
import sys

from jupyter_client.kernelspec import KernelSpecManager

from .flowbook_sdc_client import FlowbookSDCKernelClient
from .flowbook_sdc_kernel import FlowbookSDCKernel
from .sdc_enforcer import SDCEnforcer

__all__ = [
    "FlowbookSDCKernel",
    "FlowbookSDCKernelClient",
    "SDCEnforcer",
    "install_sdc_kernel",
]


def install_sdc_kernel() -> str:
    """
    Install the FlowBook SDC kernel spec.

    Returns:
        Path to installed kernel spec directory
    """
    ksm = KernelSpecManager()

    # Path to our kernel spec
    spec_dir = os.path.join(os.path.dirname(__file__), "kernelspec")

    # Install it
    dest = ksm.install_kernel_spec(
        spec_dir,
        kernel_name="flowbook_sdc_kernel",
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
    install_sdc_kernel()
except Exception:
    pass  # Don't fail import if kernel install fails
