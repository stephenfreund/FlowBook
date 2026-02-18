#!/usr/bin/env python3
"""Launch Baseline Scalene kernel with Scalene memory tracking enabled.

This launcher runs the Baseline Scalene kernel under Scalene's memory profiler,
which properly initializes Scalene and enables accurate memory tracking.

Usage:
    python -m flowbook.baseline_scalene_kernel.launch_with_scalene -f connection_file

The launcher:
1. Checks if Scalene is already initialized (running under `scalene` command)
2. If not, re-executes using `scalene --memory` to properly initialize
3. Once under Scalene, starts the Baseline Scalene kernel normally
"""

import os
import platform
import subprocess
import sys
from typing import Optional


def is_scalene_initialized() -> bool:
    """Check if Scalene is properly initialized (not just preloaded).

    Returns True if:
    - Running under `scalene` command (Scalene.__initialized is True), OR
    - The Scalene library is preloaded (fallback check)
    """
    try:
        from scalene.scalene_profiler import Scalene
        if Scalene._Scalene__initialized:
            return True
    except (ImportError, AttributeError):
        pass

    # Fallback: check if library is preloaded
    system = platform.system()
    if system == "Darwin":
        preload = os.environ.get("DYLD_INSERT_LIBRARIES", "")
        return "libscalene" in preload
    elif system == "Linux":
        preload = os.environ.get("LD_PRELOAD", "")
        return "libscalene" in preload
    return False


def launch_kernel():
    """Launch the Baseline Scalene kernel (called when running under Scalene)."""
    from ipykernel.kernelapp import IPKernelApp
    from flowbook.baseline_scalene_kernel.baseline_scalene_kernel import (
        BaselineScaleneKernel,
    )

    IPKernelApp.launch_instance(kernel_class=BaselineScaleneKernel)


def main(argv: Optional[list] = None):
    """Main entry point for the launcher.

    If not running under Scalene, re-executes using `scalene --memory`.
    If already under Scalene, launches the kernel directly.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Check if we're already running under Scalene
    if is_scalene_initialized():
        # Already under Scalene, launch the kernel
        launch_kernel()
        return

    # Not under Scalene - need to re-execute using `scalene` command
    # This properly initializes Scalene's memory tracking

    # Build the scalene command
    # --memory: Enable memory profiling
    # --no-browser: Don't try to open a browser for output
    # --cli: Use CLI output mode
    # --profile-all: Profile all code (including libraries)
    # -m: Run as module
    cmd = [
        sys.executable, "-m", "scalene",
        "--memory",
        "--no-browser",
        "--cli",
        "--profile-all",  # Needed to track kernel code
        "-m", "flowbook.baseline_scalene_kernel.launch_with_scalene",
    ]

    # Add the original arguments (like -f connection_file)
    # Use '---' to separate scalene args from script args
    if argv:
        cmd.append("---")
        cmd.extend(argv)

    # Execute and wait
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
