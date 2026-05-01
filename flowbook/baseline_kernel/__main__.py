"""Entry point for running BaselineKernel as a module."""

from ipykernel.kernelapp import IPKernelApp
from traitlets.config import Config

from flowbook.baseline_kernel.baseline_kernel import (
    BaselineKernel,
)

if __name__ == "__main__":
    config = Config()
    config.IPKernelApp.capture_fd_output = False
    IPKernelApp.launch_instance(kernel_class=BaselineKernel, config=config)
