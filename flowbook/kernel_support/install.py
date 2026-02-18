"""
Shared kernel spec installation utility.

Provides a common function for installing FlowBook kernel specs.
"""

import json
import os
import sys

from jupyter_client.kernelspec import KernelSpecManager


def install_kernel(
    package_dir: str, kernel_name: str, spec_subdir: str = "kernelspec"
) -> str:
    """
    Install a kernel spec from package_dir/<spec_subdir>/ with the given name.

    Args:
        package_dir: Directory containing the kernel spec subdirectory
        kernel_name: Name to register the kernel as
        spec_subdir: Name of subdirectory containing kernel.json (default: "kernelspec")

    Returns:
        Path to installed kernel spec directory
    """
    ksm = KernelSpecManager()

    # Path to our kernel spec
    spec_dir = os.path.join(package_dir, spec_subdir)

    # Install it
    dest = ksm.install_kernel_spec(
        spec_dir,
        kernel_name=kernel_name,
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
