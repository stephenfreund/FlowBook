"""Entry point for running ExperimentalKernel as a module."""

from ipykernel.kernelapp import IPKernelApp
from traitlets.config import Config

from flowbook.kernel_support.experimental_kernel import ExperimentalKernel

if __name__ == "__main__":
    config = Config()
    config.IPKernelApp.capture_fd_output = False
    IPKernelApp.launch_instance(kernel_class=ExperimentalKernel, config=config)
