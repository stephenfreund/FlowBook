"""
ferret_lab - Launch JupyterLab with kernel output streaming to terminal.

This CLI command:
1. Creates a Unix socket for receiving kernel output
2. Launches JupyterLab with the socket path in environment
3. Prints kernel log output to the terminal in real-time
"""

import atexit
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class SocketOutputReceiver:
    """
    Receives output from kernels via Unix domain socket.

    Creates a socket server that accepts connections from kernels and
    prints their output to the terminal.
    """

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.server_socket = None
        self.running = False
        self.listener_thread = None
        self.client_threads = []

    def start(self):
        """Start the socket server."""
        # Remove existing socket file if present
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix domain socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(10)  # Allow multiple kernel connections
        self.server_socket.settimeout(1.0)  # Allow periodic checks for shutdown

        self.running = True
        self.listener_thread = threading.Thread(
            target=self._accept_connections, daemon=True
        )
        self.listener_thread.start()

    def _accept_connections(self):
        """Accept incoming connections from kernels."""
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                # Handle each client in its own thread
                client_thread = threading.Thread(
                    target=self._handle_client, args=(client_socket,), daemon=True
                )
                client_thread.start()
                self.client_threads.append(client_thread)
            except socket.timeout:
                continue
            except OSError:
                # Socket was closed
                break

    def _handle_client(self, client_socket: socket.socket):
        """Handle output from a single kernel connection."""
        client_socket.settimeout(1.0)
        buffer = b""

        while self.running:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break

                buffer += data

                # Process complete lines
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        text = line.decode("utf-8")
                        # Print to terminal (stdout)
                        sys.stdout.write(text + "\n")
                        sys.stdout.flush()
                    except UnicodeDecodeError:
                        pass

            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError, OSError):
                break

        client_socket.close()

    def stop(self):
        """Stop the socket server and cleanup."""
        self.running = False

        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass

        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)

        # Cleanup socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass


def lab_main():
    """
    Launch JupyterLab with kernel output streaming.

    Creates a Unix socket, sets FERRET_OUTPUT_SOCKET env var,
    launches JupyterLab, and prints kernel output to terminal.
    """
    # Create socket path in temp directory
    socket_path = os.path.join(tempfile.gettempdir(), f"ferret_{os.getpid()}.sock")

    # Create and start socket receiver
    receiver = SocketOutputReceiver(socket_path)
    receiver.start()

    # Register cleanup
    def cleanup():
        receiver.stop()

    atexit.register(cleanup)

    # Handle signals for clean shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def signal_handler(signum, frame):
        cleanup()
        # Re-raise to allow normal signal handling
        if signum == signal.SIGINT and original_sigint:
            if callable(original_sigint):
                original_sigint(signum, frame)
        elif signum == signal.SIGTERM and original_sigterm:
            if callable(original_sigterm):
                original_sigterm(signum, frame)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set environment variable for kernels to find
    env = os.environ.copy()
    env["FERRET_OUTPUT_SOCKET"] = socket_path

    print(f"[ferret_lab] Starting with output socket: {socket_path}")
    print(f"[ferret_lab] Kernel output will appear below")
    print("-" * 60)

    # Launch JupyterLab, passing through all command line arguments
    # Add -y to auto-confirm exit (avoids stdin read error on Ctrl-C)
    cmd = ["jupyter", "lab", "-y"] + sys.argv[1:]
    process = subprocess.Popen(
        cmd,
        env=env,
        # Don't capture stdout/stderr - let JupyterLab print normally
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
        cleanup()


if __name__ == "__main__":
    sys.exit(lab_main())
