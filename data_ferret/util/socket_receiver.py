"""
Socket-based output receiver for kernel communication.

Provides a Unix socket server that receives output from kernels and
prints it to the terminal in real-time.
"""

import atexit
import os
import signal
import socket
import sys
import tempfile
import threading
from typing import Callable, Optional


class SocketOutputReceiver:
    """
    Receives output from kernels via Unix domain socket.

    Creates a socket server that accepts connections from kernels and
    prints their output to the terminal.

    Usage:
        receiver = SocketOutputReceiver("/tmp/ferret_123.sock")
        receiver.start()
        # ... run kernel operations ...
        receiver.stop()

    Or as context manager:
        with SocketOutputReceiver.create() as (receiver, socket_path):
            os.environ["FERRET_OUTPUT_SOCKET"] = socket_path
            # ... run kernel operations ...
    """

    def __init__(self, socket_path: str, output_handler: Optional[Callable[[str], None]] = None):
        """
        Initialize the receiver.

        Args:
            socket_path: Path for the Unix domain socket
            output_handler: Optional custom handler for output lines.
                           Default prints to stdout.
        """
        self.socket_path = socket_path
        self.output_handler = output_handler or self._default_handler
        self.server_socket = None
        self.running = False
        self.listener_thread = None
        self.client_threads = []

    @staticmethod
    def _default_handler(text: str) -> None:
        """Default handler: print to stdout."""
        sys.stdout.write(text)
        sys.stdout.flush()

    @classmethod
    def create(cls, prefix: str = "ferret") -> "SocketOutputReceiver":
        """
        Create a receiver with an auto-generated socket path.

        Args:
            prefix: Prefix for the socket filename

        Returns:
            SocketOutputReceiver instance (not yet started)
        """
        socket_path = os.path.join(
            tempfile.gettempdir(), f"{prefix}_{os.getpid()}.sock"
        )
        return cls(socket_path)

    def start(self) -> str:
        """
        Start the socket server.

        Returns:
            The socket path (for setting in environment)
        """
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

        return self.socket_path

    def _accept_connections(self) -> None:
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

    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle output from a single kernel connection."""
        client_socket.settimeout(1.0)

        while self.running:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break

                # Process data immediately without waiting for newlines
                try:
                    text = data.decode("utf-8")
                    self.output_handler(text)
                except UnicodeDecodeError:
                    pass

            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError, OSError):
                break

        client_socket.close()

    def stop(self) -> None:
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

    def __enter__(self) -> "SocketOutputReceiver":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def setup_socket_receiver(prefix: str = "ferret") -> tuple:
    """
    Convenience function to set up a socket receiver with cleanup handlers.

    Creates a receiver, starts it, registers cleanup handlers for atexit
    and signals, and sets the FERRET_OUTPUT_SOCKET environment variable.

    Args:
        prefix: Prefix for socket filename

    Returns:
        Tuple of (receiver, socket_path)
    """
    receiver = SocketOutputReceiver.create(prefix)
    socket_path = receiver.start()

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

    # Set environment variable for kernels
    os.environ["FERRET_OUTPUT_SOCKET"] = socket_path

    return receiver, socket_path
