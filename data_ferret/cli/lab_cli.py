"""
ferret_lab - Launch JupyterLab with kernel output streaming to terminal.

This CLI command:
1. Creates a Unix socket for receiving kernel output
2. Launches JupyterLab with the socket path in environment
3. Prints kernel log output to the terminal in real-time
"""

import subprocess
import sys

from data_ferret.util.socket_receiver import setup_socket_receiver


def lab_main():
    """
    Launch JupyterLab with kernel output streaming.

    Creates a Unix socket, sets FERRET_OUTPUT_SOCKET env var,
    launches JupyterLab, and prints kernel output to terminal.
    """
    # Create and start socket receiver with cleanup handlers
    receiver, socket_path = setup_socket_receiver("ferret_lab")

    print(f"[ferret_lab] Starting with output socket: {socket_path}")
    print(f"[ferret_lab] Kernel output will appear below")
    print("-" * 60)

    # Launch JupyterLab, passing through all command line arguments
    # Add -y to auto-confirm exit (avoids stdin read error on Ctrl-C)
    import os
    cmd = ["jupyter", "lab", "-y"] + sys.argv[1:]
    process = subprocess.Popen(
        cmd,
        env=os.environ.copy(),  # Includes FERRET_OUTPUT_SOCKET from setup
    )

    try:
        # Wait for JupyterLab to exit
        return_code = process.wait()
        return return_code

    except KeyboardInterrupt:
        print("\n[ferret_lab] Shutting down...")
        return 0
    except FileNotFoundError:
        print(
            "[ferret_lab] Error: 'jupyter' command not found. Is JupyterLab installed?"
        )
        return 1
    finally:
        process.kill()
        receiver.stop()


if __name__ == "__main__":
    sys.exit(lab_main())
