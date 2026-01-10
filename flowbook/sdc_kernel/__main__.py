"""Entry point for running FlowbookSDCKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from .flowbook_sdc_kernel import FlowbookSDCKernel

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FlowbookSDCKernel)
