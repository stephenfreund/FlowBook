"""Entry point for running FerretKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from .ferret_kernel import FerretKernel

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretKernel)
