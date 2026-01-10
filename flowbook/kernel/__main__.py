"""Entry point for running FlowbookKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from .flowbook_kernel import FlowbookKernel

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FlowbookKernel)
