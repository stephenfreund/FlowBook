"""
Kernel installer for DataFerret's debuggable IPython kernel.
This script is called during the build process to install the kernel.
"""

import json
import sys
import shutil
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager

from data_ferret.util.output import log, timer


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

        try:
            with open(kernel_json_path, "r") as f:
                spec_content = f.read()

            # Replace the {python} placeholder with the actual Python executable path
            spec_content = spec_content.replace("{python}", sys.executable)

            # Parse the updated content
            spec = json.loads(spec_content)

            # Create a temporary kernelspec directory
            temp_kernelspec = spec_source_dir.parent / ".tmp_kernelspec"
            temp_kernelspec.mkdir(exist_ok=True)

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
                replace=True,
            )

            # Clean up temporary directory
            shutil.rmtree(temp_kernelspec)

        except Exception as e:
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


def install_kernel_spec(kernel_name: str, spec_source_dir: Path):
    # ksm = KernelSpecManager()
    # try:
    #     ksm.get_kernel_spec(kernel_name)
    #     log("Kernel spec already installed")
    # except Exception:
    #     log("Installing kernel spec")
    install_kernel(kernel_name, spec_source_dir)


if __name__ == "__main__":
    # Allow manual kernel updates
    update_kernel(sys.argv[1], Path(sys.argv[2]))
