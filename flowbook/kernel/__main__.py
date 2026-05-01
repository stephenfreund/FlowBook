"""Entry point for running FlowbookKernel as a module."""

from ipykernel.kernelapp import IPKernelApp
from traitlets.config import Config

from flowbook.kernel.flowbook_kernel import FlowbookKernel

if __name__ == "__main__":
    config = Config()
    config.IPKernelApp.capture_fd_output = False
    IPKernelApp.launch_instance(kernel_class=FlowbookKernel, config=config)
