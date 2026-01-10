"""Timeout handling for cell execution with graceful cleanup."""

import _thread
import os
import threading
from typing import Callable, Optional

import psutil

from flowbook.kernel.process_cleanup import stop_loky_and_all_children


class CellTimeoutHandler:
    """
    Manages cell execution timeout with escalating cleanup.

    On timeout:
    1. Raises KeyboardInterrupt in the main thread
    2. After a grace period, terminates child processes
    """

    def __init__(
        self,
        timeout: float,
        post_kb_grace: float = 1.0,
        kill_timeout: float = 3.0,
        verbose: bool = False,
        max_passes: int = 2,
    ):
        """
        Initialize the timeout handler.

        Args:
            timeout: Seconds before triggering timeout
            post_kb_grace: Seconds to wait after interrupt before killing children
            kill_timeout: Seconds to wait for processes to die
            verbose: If True, print debug info
            max_passes: Number of cleanup passes
        """
        self.timeout = timeout
        self.post_kb_grace = post_kb_grace
        self.kill_timeout = kill_timeout
        self.verbose = verbose
        self.max_passes = max_passes

        self._timer: Optional[threading.Timer] = None
        self._done = False

    def start(self) -> None:
        """Arm the timeout watchdog."""
        self._done = False
        self._timer = threading.Timer(self.timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        """Disarm the timeout watchdog."""
        self._done = True
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass

    def _on_timeout(self) -> None:
        """Handle timeout by interrupting main thread and scheduling cleanup."""
        _thread.interrupt_main()

        def _escalate():
            if self._done:
                return
            try:
                me = psutil.Process(os.getpid())
                kids = me.children(recursive=True)
                for p in kids:
                    try:
                        p.terminate()
                    except Exception:
                        pass
            except Exception:
                pass

        threading.Timer(self.post_kb_grace, _escalate).start()

    async def cleanup_on_error(self) -> None:
        """Perform full cleanup when execution didn't complete normally."""
        try:
            await stop_loky_and_all_children(
                timeout=self.kill_timeout,
                verbose=self.verbose,
                max_passes=self.max_passes,
            )
        except Exception:
            pass
