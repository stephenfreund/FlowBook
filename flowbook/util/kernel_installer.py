"""
Kernel installer for FlowBook's debuggable IPython kernel.
This script is called during the build process to install the kernel.
"""

import json
import sys
import shutil
import tempfile
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager
from filelock import FileLock, Timeout
from jupyter_core.paths import jupyter_data_dir

from flowbook.util.output import log, timer


def install_kernel(kernel_name: str, spec_source_dir: Path):
    with timer(message=f"Installing {kernel_name} kernel..."):

        if not spec_source_dir.exists():
            raise ValueError(
                f"Warning: Kernel spec source not found at {spec_source_dir}"
            )

        # Read the kernel.json template
        kernel_json_path = spec_source_dir / "kernel.json"
        if not kernel_json_path.exists():
            raise ValueError(f"Warning: kernel.json not found at {kernel_json_path}")

        temp_kernelspec = None
        try:
            with open(kernel_json_path, "r") as f:
                spec_content = f.read()

            # Replace the {python} placeholder with the actual Python executable path
            spec_content = spec_content.replace("{python}", sys.executable)

            # Parse the updated content
            spec = json.loads(spec_content)

            # Create a unique temporary kernelspec directory (process-safe)
            temp_kernelspec = Path(tempfile.mkdtemp(prefix="flowbook_kernel_"))

            # Write the updated kernel.json
            temp_kernel_json = temp_kernelspec / "kernel.json"
            with open(temp_kernel_json, "w") as f:
                json.dump(spec, f, indent=2)

            # Copy any other files (e.g., logos)
            for fn in spec_source_dir.iterdir():
                if fn.name != "kernel.json":
                    if fn.is_file():
                        log(f"Copying {fn} to {temp_kernelspec / fn.name}")
                        shutil.copy2(fn, temp_kernelspec / fn.name)
                    elif fn.is_dir():
                        log(f"Copying {fn} to {temp_kernelspec / fn.name}")
                        shutil.copytree(
                            fn, temp_kernelspec / fn.name, dirs_exist_ok=True
                        )



            # Install the kernel
            ksm = KernelSpecManager()
            ksm.install_kernel_spec(
                str(temp_kernelspec),
                kernel_name=kernel_name,
                user=True,
            )

            # Clean up temporary directory
            shutil.rmtree(temp_kernelspec)
            temp_kernelspec = None

        except Exception as e:
            # Clean up temporary directory if it was created
            if temp_kernelspec and temp_kernelspec.exists():
                shutil.rmtree(temp_kernelspec, ignore_errors=True)
            raise ValueError(f"Error installing kernel: {e}")


def update_kernel(kernel_name: str, spec_source_dir: Path):
    """
    Update the kernel installation (useful for development).
    This can be called manually or from a development script.
    """
    # Get the current package directory
    current_dir = Path(__file__).parent
    source_dir = current_dir.parent.parent  # Go up to project root

    install_kernel(kernel_name, spec_source_dir)


def is_kernel_installed_correctly(kernel_name: str) -> bool:
    """Check if kernel is installed and has correct Python path."""
    try:
        ksm = KernelSpecManager()
        spec = ksm.get_kernel_spec(kernel_name)

        # Read the installed kernel.json
        kernel_json_path = Path(spec.resource_dir) / "kernel.json"
        if not kernel_json_path.exists():
            return False

        with open(kernel_json_path, "r") as f:
            kernel_data = json.load(f)

        # Check if Python executable matches current environment
        if not kernel_data.get("argv"):
            return False

        installed_python = kernel_data["argv"][0]
        current_python = sys.executable

        # Compare resolved paths (handle symlinks)
        return Path(installed_python).resolve() == Path(current_python).resolve()

    except Exception:
        # Catch all exceptions: NoSuchKernel, FileNotFoundError, json.JSONDecodeError, KeyError, etc.
        return False


def install_kernel_spec(kernel_name: str, spec_source_dir: Path):
    """Install kernel spec with process-safe idempotent behavior."""

    with timer(message=f"Installing {kernel_name} kernel..."):
        # Fast path: check if already correctly installed
        if is_kernel_installed_correctly(kernel_name):
            log(f"Kernel '{kernel_name}' already correctly installed")
            return

        # Slow path: acquire lock and install
        lock_dir = Path(jupyter_data_dir()) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / f"{kernel_name}.lock"

        lock = FileLock(lock_file, timeout=30)

        try:
            with lock:
                # Double-check: another process may have just installed it
                if is_kernel_installed_correctly(kernel_name):
                    log(f"Kernel '{kernel_name}' was installed by another process")
                    return

                log(f"Installing kernel spec '{kernel_name}'")
                install_kernel(kernel_name, spec_source_dir)
                log(f"Kernel '{kernel_name}' installation complete")

        except Exception as e:
            log(f"Error installing kernel spec '{kernel_name}': {e}")
            log(f"Continuing anyway...")

        except Timeout:
            log(
                f"Warning: Timeout acquiring lock for kernel installation. "
                f"Another process may be installing. Continuing anyway..."
            )
            # Don't fail - the other process will likely complete installation


if __name__ == "__main__":
    # Allow manual kernel updates
    update_kernel(sys.argv[1], Path(sys.argv[2]))
