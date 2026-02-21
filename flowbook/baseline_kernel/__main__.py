"""Entry point for running BaselineKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from flowbook.baseline_kernel.baseline_kernel import (
    BaselineKernel,
)

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=BaselineKernel)
