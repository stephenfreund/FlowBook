"""Entry point for running FerretSDCKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from .ferret_sdc_kernel import FerretSDCKernel

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretSDCKernel)
