"""Entry point for running BaselineScaleneKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from flowbook.baseline_scalene_kernel.baseline_scalene_kernel import (
    BaselineScaleneKernel,
)

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=BaselineScaleneKernel)
