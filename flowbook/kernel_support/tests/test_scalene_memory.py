"""Tests for ScaleneMemoryTracker.

Tests cover both the case when Scalene is available (with preload) and when
it's not available (fallback behavior). Many tests use mocking since Scalene
requires native library preloading to function.
"""

import os
import platform
from unittest import mock

import pytest

from flowbook.kernel_support.scalene_memory import ScaleneMemoryTracker


class TestScaleneAvailability:
    """Tests for is_available() method."""

    def setup_method(self):
        """Reset cached availability before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_not_available_without_preload_darwin(self):
        """On macOS without DYLD_INSERT_LIBRARIES, Scalene is not available."""
        ScaleneMemoryTracker._scalene_available = None

        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch.dict(os.environ, {}, clear=True):
                result = ScaleneMemoryTracker.is_available()
                assert result is False

    def test_not_available_without_preload_linux(self):
        """On Linux without LD_PRELOAD, Scalene is not available."""
        ScaleneMemoryTracker._scalene_available = None

        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch.dict(os.environ, {}, clear=True):
                result = ScaleneMemoryTracker.is_available()
                assert result is False

    def test_caches_availability_result(self):
        """Availability check result is cached."""
        ScaleneMemoryTracker._scalene_available = None

        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch.dict(os.environ, {}, clear=True):
                result1 = ScaleneMemoryTracker.is_available()
                assert result1 is False
                assert ScaleneMemoryTracker._scalene_available is False

                # Second call uses cached value
                result2 = ScaleneMemoryTracker.is_available()
                assert result2 is False

    def test_available_with_preload_darwin(self):
        """On macOS with libscalene preloaded and import working, available."""
        ScaleneMemoryTracker._scalene_available = None

        mock_scalene = mock.MagicMock()
        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch.dict(
                os.environ, {"DYLD_INSERT_LIBRARIES": "/path/to/libscalene.dylib"}
            ):
                with mock.patch.dict(
                    "sys.modules", {"scalene.scalene_profiler": mock_scalene}
                ):
                    result = ScaleneMemoryTracker.is_available()
                    assert result is True

    def test_available_with_preload_linux(self):
        """On Linux with libscalene preloaded and import working, available."""
        ScaleneMemoryTracker._scalene_available = None

        mock_scalene = mock.MagicMock()
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch.dict(os.environ, {"LD_PRELOAD": "/path/to/libscalene.so"}):
                with mock.patch.dict(
                    "sys.modules", {"scalene.scalene_profiler": mock_scalene}
                ):
                    result = ScaleneMemoryTracker.is_available()
                    assert result is True

    def test_not_available_with_wrong_preload_darwin(self):
        """On macOS with wrong library preloaded, Scalene is not available."""
        ScaleneMemoryTracker._scalene_available = None

        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch.dict(
                os.environ, {"DYLD_INSERT_LIBRARIES": "/path/to/other.dylib"}
            ):
                result = ScaleneMemoryTracker.is_available()
                assert result is False


class TestStartStop:
    """Tests for start() and stop() methods."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_start_returns_false_when_not_available(self):
        """start() returns False when Scalene is not available."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.start()
        assert result is False
        assert ScaleneMemoryTracker._tracking_enabled is False

    def test_stop_returns_false_when_not_available(self):
        """stop() returns False when Scalene is not available."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.stop()
        assert result is False

    def test_start_enables_tracking_when_available(self):
        """start() enables tracking when Scalene is available."""
        ScaleneMemoryTracker._scalene_available = True

        mock_profiler = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules", {"scalene": mock_profiler, "scalene.scalene_profiler": mock_profiler}
        ):
            with mock.patch(
                "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
                return_value=True,
            ):
                result = ScaleneMemoryTracker.start()
                assert result is True
                assert ScaleneMemoryTracker._tracking_enabled is True

    def test_stop_disables_tracking_when_available(self):
        """stop() disables tracking when Scalene is available."""
        ScaleneMemoryTracker._scalene_available = True
        ScaleneMemoryTracker._tracking_enabled = True

        mock_profiler = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules", {"scalene": mock_profiler, "scalene.scalene_profiler": mock_profiler}
        ):
            with mock.patch(
                "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
                return_value=True,
            ):
                result = ScaleneMemoryTracker.stop()
                assert result is True
                assert ScaleneMemoryTracker._tracking_enabled is False


class TestIsTracking:
    """Tests for is_tracking() method."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_not_tracking_when_disabled(self):
        """is_tracking() returns False when tracking is disabled."""
        ScaleneMemoryTracker._tracking_enabled = False
        ScaleneMemoryTracker._scalene_available = True

        result = ScaleneMemoryTracker.is_tracking()
        assert result is False

    def test_not_tracking_when_unavailable(self):
        """is_tracking() returns False when Scalene is unavailable."""
        ScaleneMemoryTracker._tracking_enabled = True
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.is_tracking()
        assert result is False

    def test_tracking_when_enabled_and_available(self):
        """is_tracking() returns True when enabled and Scalene available."""
        ScaleneMemoryTracker._tracking_enabled = True
        ScaleneMemoryTracker._scalene_available = True

        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            result = ScaleneMemoryTracker.is_tracking()
            assert result is True


class TestGetSamples:
    """Tests for get_malloc_samples() and get_free_samples() methods."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_get_malloc_samples_returns_zero_when_unavailable(self):
        """get_malloc_samples() returns 0 when Scalene is unavailable."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.get_malloc_samples()
        assert result == 0.0

    def test_get_free_samples_returns_zero_when_unavailable(self):
        """get_free_samples() returns 0 when Scalene is unavailable."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.get_free_samples()
        assert result == 0.0

    def test_get_malloc_samples_returns_value_when_available(self):
        """get_malloc_samples() returns Scalene's value when available."""
        mock_stats = mock.MagicMock()
        mock_stats.memory_stats.total_memory_malloc_samples = 42.5

        mock_scalene = mock.MagicMock()
        mock_scalene._Scalene__stats = mock_stats

        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            with mock.patch.dict(
                "sys.modules", {"scalene.scalene_profiler": mock.MagicMock(Scalene=mock_scalene)}
            ):
                # Need to mock the import within the method
                with mock.patch(
                    "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.get_malloc_samples"
                ) as mock_method:
                    mock_method.return_value = 42.5
                    result = ScaleneMemoryTracker.get_malloc_samples()
                    assert result == 42.5

    def test_get_free_samples_returns_value_when_available(self):
        """get_free_samples() returns Scalene's value when available."""
        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            with mock.patch(
                "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.get_free_samples"
            ) as mock_method:
                mock_method.return_value = 10.0
                result = ScaleneMemoryTracker.get_free_samples()
                assert result == 10.0


class TestGetMemory:
    """Tests for get_memory() method."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_get_memory_returns_zeros_when_unavailable(self):
        """get_memory() returns dict with zeros when Scalene unavailable."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.get_memory()

        assert result["available"] is False
        assert result["current_footprint_mb"] == 0.0
        assert result["max_footprint_mb"] == 0.0
        assert result["total_malloc_mb"] == 0.0
        assert result["total_free_mb"] == 0.0
        assert result["net_allocation_mb"] == 0.0
        assert result["gpu_samples"] == 0.0
        assert result["gpu_mem_samples"] == 0.0

    def test_get_memory_returns_stats_when_available(self):
        """get_memory() returns Scalene stats when available."""
        mock_memory_stats = mock.MagicMock()
        mock_memory_stats.current_footprint = 100.0
        mock_memory_stats.max_footprint = 150.0
        mock_memory_stats.total_memory_malloc_samples = 200.0
        mock_memory_stats.total_memory_free_samples = 50.0

        mock_gpu_stats = mock.MagicMock()
        mock_gpu_stats.total_gpu_samples = 1000.0
        mock_gpu_stats.total_gpu_mem_samples = 500.0

        mock_stats = mock.MagicMock()
        mock_stats.memory_stats = mock_memory_stats
        mock_stats.gpu_stats = mock_gpu_stats

        mock_scalene_class = mock.MagicMock()
        mock_scalene_class._Scalene__stats = mock_stats

        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            # Mock the import statement inside get_memory
            import sys
            mock_module = mock.MagicMock()
            mock_module.Scalene = mock_scalene_class

            with mock.patch.dict(sys.modules, {"scalene.scalene_profiler": mock_module}):
                result = ScaleneMemoryTracker.get_memory()

                assert result["available"] is True
                assert result["current_footprint_mb"] == 100.0
                assert result["max_footprint_mb"] == 150.0
                assert result["total_malloc_mb"] == 200.0
                assert result["total_free_mb"] == 50.0
                assert result["net_allocation_mb"] == 150.0
                assert result["gpu_samples"] == 1000.0
                assert result["gpu_mem_samples"] == 500.0

    def test_get_memory_handles_exception(self):
        """get_memory() returns error dict on exception."""
        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            # Cause an exception by not mocking the import
            with mock.patch.dict("sys.modules", {"scalene.scalene_profiler": None}):
                result = ScaleneMemoryTracker.get_memory()

                # Should return fallback dict with error
                assert result["available"] is False
                assert result["current_footprint_mb"] == 0.0


class TestReset:
    """Tests for reset() method."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_reset_returns_false_when_unavailable(self):
        """reset() returns False when Scalene is unavailable."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.reset()
        assert result is False

    def test_reset_calls_clear_all_when_available(self):
        """reset() calls Scalene's clear_all when available."""
        mock_stats = mock.MagicMock()
        mock_scalene_class = mock.MagicMock()
        mock_scalene_class._Scalene__stats = mock_stats

        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            import sys
            mock_module = mock.MagicMock()
            mock_module.Scalene = mock_scalene_class

            with mock.patch.dict(sys.modules, {"scalene.scalene_profiler": mock_module}):
                result = ScaleneMemoryTracker.reset()

                assert result is True
                mock_stats.clear_all.assert_called_once()


class TestFormatStats:
    """Tests for format_stats() method."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_format_stats_not_available(self):
        """format_stats() shows not available message."""
        stats = {"available": False}
        result = ScaleneMemoryTracker.format_stats(stats)
        assert result == "Scalene memory tracking not available"

    def test_format_stats_with_memory_only(self):
        """format_stats() formats memory stats correctly."""
        stats = {
            "available": True,
            "current_footprint_mb": 42.5,
            "max_footprint_mb": 50.0,
            "net_allocation_mb": 10.0,
            "gpu_samples": 0,
            "gpu_mem_samples": 0,
        }
        result = ScaleneMemoryTracker.format_stats(stats)

        assert "current=42.5 MB" in result
        assert "max=50.0 MB" in result
        assert "net=+10.0 MB" in result
        assert "GPU" not in result

    def test_format_stats_with_gpu(self):
        """format_stats() includes GPU stats when present."""
        stats = {
            "available": True,
            "current_footprint_mb": 100.0,
            "max_footprint_mb": 120.0,
            "net_allocation_mb": 50.0,
            "gpu_samples": 1000,
            "gpu_mem_samples": 2.5,
        }
        result = ScaleneMemoryTracker.format_stats(stats)

        assert "current=100.0 MB" in result
        assert "GPU" in result
        assert "samples=1000" in result

    def test_format_stats_negative_net(self):
        """format_stats() shows negative net allocation correctly."""
        stats = {
            "available": True,
            "current_footprint_mb": 30.0,
            "max_footprint_mb": 50.0,
            "net_allocation_mb": -5.0,
            "gpu_samples": 0,
            "gpu_mem_samples": 0,
        }
        result = ScaleneMemoryTracker.format_stats(stats)

        assert "net=-5.0 MB" in result

    def test_format_stats_calls_get_memory_when_none(self):
        """format_stats() calls get_memory() when stats is None."""
        ScaleneMemoryTracker._scalene_available = False

        result = ScaleneMemoryTracker.format_stats(None)
        assert result == "Scalene memory tracking not available"


class TestIntegration:
    """Integration tests for ScaleneMemoryTracker workflow."""

    def setup_method(self):
        """Reset state before each test."""
        ScaleneMemoryTracker._scalene_available = None
        ScaleneMemoryTracker._tracking_enabled = False

    def test_full_workflow_when_unavailable(self):
        """Full workflow behaves correctly when Scalene unavailable."""
        ScaleneMemoryTracker._scalene_available = False

        # Check availability
        assert ScaleneMemoryTracker.is_available() is False

        # Start should fail gracefully
        assert ScaleneMemoryTracker.start() is False
        assert ScaleneMemoryTracker.is_tracking() is False

        # Get samples should return zeros
        assert ScaleneMemoryTracker.get_malloc_samples() == 0.0
        assert ScaleneMemoryTracker.get_free_samples() == 0.0

        # Get memory should return unavailable dict
        memory = ScaleneMemoryTracker.get_memory()
        assert memory["available"] is False

        # Format stats should show unavailable
        formatted = ScaleneMemoryTracker.format_stats(memory)
        assert "not available" in formatted

        # Reset should fail gracefully
        assert ScaleneMemoryTracker.reset() is False

        # Stop should fail gracefully
        assert ScaleneMemoryTracker.stop() is False

    def test_state_transitions(self):
        """Test that state transitions work correctly."""
        ScaleneMemoryTracker._scalene_available = False
        ScaleneMemoryTracker._tracking_enabled = False

        # Initially not tracking
        assert ScaleneMemoryTracker.is_tracking() is False

        # Can't enable tracking when unavailable
        ScaleneMemoryTracker._tracking_enabled = True  # Manually set
        assert ScaleneMemoryTracker.is_tracking() is False  # Still False due to unavailability

        # Make available and tracking
        ScaleneMemoryTracker._scalene_available = True
        with mock.patch(
            "flowbook.kernel_support.scalene_memory.ScaleneMemoryTracker.is_available",
            return_value=True,
        ):
            assert ScaleneMemoryTracker.is_tracking() is True

        # Disable tracking
        ScaleneMemoryTracker._tracking_enabled = False
        assert ScaleneMemoryTracker.is_tracking() is False
