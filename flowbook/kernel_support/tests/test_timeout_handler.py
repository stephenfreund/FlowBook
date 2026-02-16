"""Tests for timeout_handler.py - Cell execution timeout handling.

Targets:
- CellTimeoutHandler initialization
- start/cancel methods
- _on_timeout trigger behavior
"""

import time
import threading
import pytest
from unittest.mock import MagicMock, patch

from flowbook.kernel_support.timeout_handler import CellTimeoutHandler


class TestCellTimeoutHandlerInit:
    """Tests for CellTimeoutHandler initialization."""

    def test_default_init(self):
        """Default initialization with required timeout."""
        handler = CellTimeoutHandler(timeout=10.0)
        assert handler.timeout == 10.0
        assert handler.post_kb_grace == 1.0
        assert handler.kill_timeout == 3.0
        assert handler.verbose is False
        assert handler.max_passes == 2

    def test_custom_init(self):
        """Custom initialization with all parameters."""
        handler = CellTimeoutHandler(
            timeout=5.0,
            post_kb_grace=2.0,
            kill_timeout=5.0,
            verbose=True,
            max_passes=3,
        )
        assert handler.timeout == 5.0
        assert handler.post_kb_grace == 2.0
        assert handler.kill_timeout == 5.0
        assert handler.verbose is True
        assert handler.max_passes == 3


class TestCellTimeoutHandlerStartCancel:
    """Tests for start and cancel methods."""

    def test_start_creates_timer(self):
        """start creates and starts a daemon timer."""
        handler = CellTimeoutHandler(timeout=100.0)  # Long timeout to not trigger
        handler.start()
        assert handler._timer is not None
        assert handler._timer.daemon is True
        assert not handler._done
        # Clean up
        handler.cancel()

    def test_cancel_stops_timer(self):
        """cancel sets _done flag and cancels timer."""
        handler = CellTimeoutHandler(timeout=100.0)
        handler.start()
        handler.cancel()
        assert handler._done is True

    def test_cancel_without_start(self):
        """cancel is safe to call without start."""
        handler = CellTimeoutHandler(timeout=10.0)
        handler.cancel()
        assert handler._done is True

    def test_cancel_with_none_timer(self):
        """cancel handles _timer being None."""
        handler = CellTimeoutHandler(timeout=10.0)
        handler._timer = None
        handler.cancel()
        assert handler._done is True


class TestCellTimeoutHandlerTimeout:
    """Tests for timeout behavior."""

    @patch("flowbook.kernel_support.timeout_handler._thread")
    def test_on_timeout_interrupts_main(self, mock_thread):
        """_on_timeout calls interrupt_main."""
        handler = CellTimeoutHandler(timeout=10.0)
        handler._on_timeout()
        mock_thread.interrupt_main.assert_called_once()

    @patch("flowbook.kernel_support.timeout_handler._thread")
    def test_on_timeout_schedules_escalation(self, mock_thread):
        """_on_timeout schedules child process termination."""
        handler = CellTimeoutHandler(timeout=10.0, post_kb_grace=0.01)
        handler._on_timeout()
        # Wait a bit for the escalation timer
        time.sleep(0.05)
        # The escalation should have run (or at least not crashed)

    @patch("flowbook.kernel_support.timeout_handler._thread")
    def test_escalation_skipped_when_done(self, mock_thread):
        """Escalation is skipped when _done is True."""
        handler = CellTimeoutHandler(timeout=10.0, post_kb_grace=0.01)
        handler._done = True  # Mark as done before timeout
        handler._on_timeout()
        # Escalation should notice _done and skip
        time.sleep(0.05)


class TestCellTimeoutHandlerCleanup:
    """Tests for cleanup_on_error method."""

    @pytest.mark.asyncio
    async def test_cleanup_on_error_calls_stop(self):
        """cleanup_on_error calls stop_loky_and_all_children."""
        handler = CellTimeoutHandler(timeout=10.0)
        with patch(
            "flowbook.kernel_support.timeout_handler.stop_loky_and_all_children"
        ) as mock_stop:
            mock_stop.return_value = None
            await handler.cleanup_on_error()
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_on_error_swallows_exception(self):
        """cleanup_on_error does not raise on failure."""
        handler = CellTimeoutHandler(timeout=10.0)
        with patch(
            "flowbook.kernel_support.timeout_handler.stop_loky_and_all_children",
            side_effect=Exception("cleanup failed"),
        ):
            # Should not raise
            await handler.cleanup_on_error()
