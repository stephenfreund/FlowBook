"""Entry point for running CheckpointKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from flowbook.checkpoint_kernel.checkpoint_kernel import CheckpointKernel

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=CheckpointKernel)
