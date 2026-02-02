"""Entry point for running ExperimentalKernel as a module."""

from ipykernel.kernelapp import IPKernelApp

from flowbook.kernel_support.experimental_kernel import ExperimentalKernel

if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=ExperimentalKernel)
