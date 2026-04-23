"""Entry point for running CheckpointKernel as a module."""

from ipykernel.kernelapp import IPKernelApp
from traitlets.config import Config

from flowbook.checkpoint_kernel.checkpoint_kernel import CheckpointKernel

if __name__ == "__main__":
    config = Config()
    config.IPKernelApp.capture_fd_output = False
    IPKernelApp.launch_instance(kernel_class=CheckpointKernel, config=config)
