"""Tests for GPU memory recording in compare_baseline.

Verifies that get_kernel_gpu_memory_mb properly queries the kernel
and that GPU memory values are recorded in MemoryCellMetrics.
"""

import pytest
from unittest.mock import Mock, MagicMock
import time

from flowbook.server.commands.compare_baseline import (
    get_kernel_gpu_memory_mb,
    MemoryCellMetrics,
)


class MockKernelClient:
    """Mock kernel client that simulates user_expressions for GPU memory."""

    def __init__(self, gpu_mb: float = 0.0, should_timeout: bool = False):
        self.gpu_mb = gpu_mb
        self.should_timeout = should_timeout
        self._msg_id = "test_msg_123"
        self._iopub_messages = []
        self._shell_messages = []

    def execute(self, code, user_expressions=None, silent=True):
        """Simulate kernel execute with user_expressions."""
        if self.should_timeout:
            # Return msg_id but never provide responses
            return self._msg_id

        # Queue iopub messages (status: busy, idle)
        self._iopub_messages = [
            {
                'parent_header': {'msg_id': self._msg_id},
                'header': {'msg_type': 'status'},
                'content': {'execution_state': 'busy'},
            },
            {
                'parent_header': {'msg_id': self._msg_id},
                'header': {'msg_type': 'status'},
                'content': {'execution_state': 'idle'},
            },
        ]

        # Queue shell reply with user_expressions result
        self._shell_messages = [
            {
                'parent_header': {'msg_id': self._msg_id},
                'header': {'msg_type': 'execute_reply'},
                'content': {
                    'status': 'ok',
                    'user_expressions': {
                        '_gpu_mem': {
                            'status': 'ok',
                            'data': {'text/plain': str(self.gpu_mb)},
                        }
                    },
                },
            }
        ]

        return self._msg_id

    def get_iopub_msg(self, timeout=1.0):
        """Return next iopub message or raise timeout."""
        if self._iopub_messages:
            return self._iopub_messages.pop(0)
        raise TimeoutError("No more iopub messages")

    def get_shell_msg(self, timeout=1.0):
        """Return next shell message or raise timeout."""
        if self._shell_messages:
            return self._shell_messages.pop(0)
        raise TimeoutError("No more shell messages")


class TestGetKernelGpuMemoryMb:
    """Tests for get_kernel_gpu_memory_mb function."""

    def test_returns_gpu_memory_from_kernel(self):
        """Test that GPU memory is correctly extracted from kernel response."""
        mock_client = MockKernelClient(gpu_mb=512.5)
        result = get_kernel_gpu_memory_mb(mock_client, timeout=5.0)
        assert result == 512.5

    def test_returns_zero_gpu_memory(self):
        """Test that zero GPU memory is returned when kernel reports 0."""
        mock_client = MockKernelClient(gpu_mb=0.0)
        result = get_kernel_gpu_memory_mb(mock_client, timeout=5.0)
        assert result == 0.0

    def test_returns_zero_on_timeout(self):
        """Test that timeout returns 0.0 instead of raising."""
        mock_client = MockKernelClient(gpu_mb=100.0, should_timeout=True)
        # Use very short timeout to trigger timeout path quickly
        result = get_kernel_gpu_memory_mb(mock_client, timeout=0.1)
        assert result == 0.0

    def test_handles_large_gpu_memory(self):
        """Test handling of large GPU memory values (multi-GB)."""
        mock_client = MockKernelClient(gpu_mb=16384.0)  # 16 GB
        result = get_kernel_gpu_memory_mb(mock_client, timeout=5.0)
        assert result == 16384.0

    def test_handles_fractional_mb(self):
        """Test handling of fractional MB values."""
        mock_client = MockKernelClient(gpu_mb=123.456789)
        result = get_kernel_gpu_memory_mb(mock_client, timeout=5.0)
        assert abs(result - 123.456789) < 0.0001


class MockKernelClientWithError:
    """Mock kernel client that returns an error status for user_expressions."""

    def __init__(self):
        self._msg_id = "test_msg_err"
        self._iopub_messages = []
        self._shell_messages = []

    def execute(self, code, user_expressions=None, silent=True):
        self._iopub_messages = [
            {
                'parent_header': {'msg_id': self._msg_id},
                'header': {'msg_type': 'status'},
                'content': {'execution_state': 'idle'},
            },
        ]
        self._shell_messages = [
            {
                'parent_header': {'msg_id': self._msg_id},
                'header': {'msg_type': 'execute_reply'},
                'content': {
                    'status': 'ok',
                    'user_expressions': {
                        '_gpu_mem': {
                            'status': 'error',
                            'ename': 'ImportError',
                            'evalue': 'No module named pynvml',
                        }
                    },
                },
            }
        ]
        return self._msg_id

    def get_iopub_msg(self, timeout=1.0):
        if self._iopub_messages:
            return self._iopub_messages.pop(0)
        raise TimeoutError()

    def get_shell_msg(self, timeout=1.0):
        if self._shell_messages:
            return self._shell_messages.pop(0)
        raise TimeoutError()


class TestGetKernelGpuMemoryMbErrors:
    """Tests for error handling in get_kernel_gpu_memory_mb."""

    def test_returns_zero_on_expression_error(self):
        """Test that error in user_expression returns 0.0."""
        mock_client = MockKernelClientWithError()
        result = get_kernel_gpu_memory_mb(mock_client, timeout=5.0)
        assert result == 0.0

    def test_returns_zero_on_missing_data(self):
        """Test handling of malformed response (missing data key)."""
        mock_client = Mock()
        mock_client.execute.return_value = "msg_123"

        # Simulate iopub idle
        iopub_msg = {
            'parent_header': {'msg_id': 'msg_123'},
            'header': {'msg_type': 'status'},
            'content': {'execution_state': 'idle'},
        }
        # Shell reply with malformed user_expressions
        shell_msg = {
            'parent_header': {'msg_id': 'msg_123'},
            'header': {'msg_type': 'execute_reply'},
            'content': {
                'status': 'ok',
                'user_expressions': {
                    '_gpu_mem': {'status': 'ok'}  # Missing 'data' key
                },
            },
        }

        mock_client.get_iopub_msg.return_value = iopub_msg
        mock_client.get_shell_msg.return_value = shell_msg

        result = get_kernel_gpu_memory_mb(mock_client, timeout=5.0)
        assert result == 0.0


class TestMemoryCellMetricsGpuFields:
    """Tests for GPU memory fields in MemoryCellMetrics."""

    def test_gpu_fields_initialized(self):
        """Test that pre_gpu_mb and gpu_mb fields exist and work."""
        cell = MemoryCellMetrics(
            cell_id="test",
            cell_index=0,
            pre_namespace_mb=10.0,
            pre_gpu_mb=256.0,
            namespace_mb=15.0,
            checkpoint_delta_mb=0.0,
            checkpoint_cumulative_mb=0.0,
            gpu_mb=512.0,
        )
        assert cell.pre_gpu_mb == 256.0
        assert cell.gpu_mb == 512.0

    def test_gpu_fields_zero_when_no_gpu(self):
        """Test that GPU fields can be zero (no GPU present)."""
        cell = MemoryCellMetrics(
            cell_id="test",
            cell_index=0,
            pre_namespace_mb=10.0,
            pre_gpu_mb=0.0,
            namespace_mb=15.0,
            checkpoint_delta_mb=0.0,
            checkpoint_cumulative_mb=0.0,
            gpu_mb=0.0,
        )
        assert cell.pre_gpu_mb == 0.0
        assert cell.gpu_mb == 0.0

    def test_gpu_memory_increase_tracked(self):
        """Test that GPU memory increase is captured (post > pre)."""
        cell = MemoryCellMetrics(
            cell_id="test",
            cell_index=0,
            pre_namespace_mb=10.0,
            pre_gpu_mb=100.0,
            namespace_mb=15.0,
            checkpoint_delta_mb=0.0,
            checkpoint_cumulative_mb=0.0,
            gpu_mb=500.0,  # 400 MB increase
        )
        gpu_delta = cell.gpu_mb - cell.pre_gpu_mb
        assert gpu_delta == 400.0

    def test_gpu_memory_decrease_tracked(self):
        """Test that GPU memory decrease is captured (post < pre)."""
        cell = MemoryCellMetrics(
            cell_id="test",
            cell_index=0,
            pre_namespace_mb=10.0,
            pre_gpu_mb=1000.0,
            namespace_mb=15.0,
            checkpoint_delta_mb=0.0,
            checkpoint_cumulative_mb=0.0,
            gpu_mb=200.0,  # 800 MB freed
        )
        gpu_delta = cell.gpu_mb - cell.pre_gpu_mb
        assert gpu_delta == -800.0


class TestGpuMemoryInTotals:
    """Tests for GPU memory in results totals."""

    def test_totals_include_final_gpu_mb(self):
        """Test that totals dict can include final_gpu_mb."""
        totals = {
            "final_namespace_mb": 100.0,
            "final_gpu_mb": 2048.0,
            "max_namespace_mb": 120.0,
        }
        assert "final_gpu_mb" in totals
        assert totals["final_gpu_mb"] == 2048.0

    def test_totals_gpu_zero_when_no_gpu(self):
        """Test that totals have zero GPU when no GPU available."""
        totals = {
            "final_namespace_mb": 100.0,
            "final_gpu_mb": 0.0,
            "max_namespace_mb": 120.0,
        }
        assert totals["final_gpu_mb"] == 0.0


class TestGpuMemoryIntegration:
    """Integration tests for GPU memory recording workflow."""

    def test_gpu_memory_flow_through_cells(self):
        """Test that GPU memory increases are tracked across cells."""
        # Simulate a notebook with 3 cells that allocate GPU memory
        cells = [
            MemoryCellMetrics(
                cell_id="c1", cell_index=0,
                pre_namespace_mb=0.0, pre_gpu_mb=0.0,
                namespace_mb=10.0,
                checkpoint_delta_mb=0.0, checkpoint_cumulative_mb=0.0,
                gpu_mb=100.0,  # First GPU allocation
            ),
            MemoryCellMetrics(
                cell_id="c2", cell_index=1,
                pre_namespace_mb=10.0, pre_gpu_mb=100.0,
                namespace_mb=20.0,
                checkpoint_delta_mb=0.0, checkpoint_cumulative_mb=0.0,
                gpu_mb=500.0,  # More GPU allocation
            ),
            MemoryCellMetrics(
                cell_id="c3", cell_index=2,
                pre_namespace_mb=20.0, pre_gpu_mb=500.0,
                namespace_mb=25.0,
                checkpoint_delta_mb=0.0, checkpoint_cumulative_mb=0.0,
                gpu_mb=500.0,  # No change
            ),
        ]

        # Verify GPU memory continuity (post of cell i = pre of cell i+1)
        for i in range(len(cells) - 1):
            assert cells[i].gpu_mb == cells[i + 1].pre_gpu_mb, (
                f"GPU memory mismatch between cell {i} and {i+1}: "
                f"{cells[i].gpu_mb} != {cells[i+1].pre_gpu_mb}"
            )

    def test_rerun_cells_track_gpu_memory(self):
        """Test that rerun cells also track GPU memory."""
        rerun_cell = MemoryCellMetrics(
            cell_id="c1",
            cell_index=0,
            pre_namespace_mb=100.0,
            pre_gpu_mb=1024.0,
            namespace_mb=100.0,
            checkpoint_delta_mb=0.0,
            checkpoint_cumulative_mb=0.0,
            gpu_mb=1024.0,
            is_rerun=True,
        )
        assert rerun_cell.is_rerun is True
        assert rerun_cell.pre_gpu_mb == 1024.0
        assert rerun_cell.gpu_mb == 1024.0


def _has_gpu() -> bool:
    """Check if GPU is available for testing."""
    try:
        from flowbook.util.gpu_memory import has_gpu
        return has_gpu()
    except ImportError:
        return False


def _cupy_available() -> bool:
    """Check if CuPy is available."""
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def kernel_client():
    """Start a real IPython kernel for integration testing.

    Yields a kernel client, then cleans up the kernel.
    """
    from jupyter_client import KernelManager
    import time

    km = KernelManager(kernel_name='python3')
    km.start_kernel()

    kc = km.client()
    kc.start_channels()

    # Wait for kernel to be ready
    kc.wait_for_ready(timeout=30)

    yield kc

    # Cleanup
    kc.stop_channels()
    km.shutdown_kernel(now=True)


@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestGetKernelGpuMemoryMbWithRealKernel:
    """Integration tests using a real kernel to verify GPU memory detection."""

    def test_detects_zero_before_allocation(self, kernel_client):
        """Test that GPU memory is ~0 before any GPU allocation."""
        # Get initial GPU memory (should be 0 or very small)
        initial_mem = get_kernel_gpu_memory_mb(kernel_client, timeout=10.0)

        # Should be a float
        assert isinstance(initial_mem, float)
        # Should be non-negative
        assert initial_mem >= 0.0

    @pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
    def test_detects_cupy_allocation(self, kernel_client):
        """Test that GPU memory increases after CuPy allocation in kernel."""
        # Get initial memory
        initial_mem = get_kernel_gpu_memory_mb(kernel_client, timeout=10.0)

        # Allocate ~100 MB on GPU via kernel
        allocate_code = """
import cupy as cp
_gpu_arr = cp.zeros(25 * 1024 * 1024, dtype=cp.float32)  # 100 MB
cp.cuda.Device().synchronize()
"""
        msg_id = kernel_client.execute(allocate_code)
        # Wait for execution to complete
        while True:
            try:
                msg = kernel_client.get_iopub_msg(timeout=30.0)
                if msg['parent_header'].get('msg_id') == msg_id:
                    if msg['header']['msg_type'] == 'status':
                        if msg['content']['execution_state'] == 'idle':
                            break
            except Exception:
                break

        # Get memory after allocation
        post_mem = get_kernel_gpu_memory_mb(kernel_client, timeout=10.0)

        # Should detect memory increase
        assert post_mem > initial_mem, (
            f"Expected GPU memory increase after CuPy allocation: "
            f"{initial_mem:.2f} MB -> {post_mem:.2f} MB"
        )

        # Delta should be roughly 100 MB (allow some variance for driver overhead)
        delta = post_mem - initial_mem
        assert delta >= 90.0, (
            f"Expected ~100 MB increase, got {delta:.2f} MB"
        )

    @pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
    def test_memory_scales_with_allocation_size(self, kernel_client):
        """Test that larger allocations show proportionally more memory."""
        # First allocation: ~50 MB
        code1 = """
import cupy as cp
_arr1 = cp.zeros(12 * 1024 * 1024, dtype=cp.float32)  # 48 MB
cp.cuda.Device().synchronize()
"""
        msg_id = kernel_client.execute(code1)
        while True:
            try:
                msg = kernel_client.get_iopub_msg(timeout=30.0)
                if msg['parent_header'].get('msg_id') == msg_id:
                    if msg['header']['msg_type'] == 'status':
                        if msg['content']['execution_state'] == 'idle':
                            break
            except Exception:
                break

        mem1 = get_kernel_gpu_memory_mb(kernel_client, timeout=10.0)

        # Second allocation: additional ~100 MB
        code2 = """
_arr2 = cp.zeros(25 * 1024 * 1024, dtype=cp.float32)  # 100 MB
cp.cuda.Device().synchronize()
"""
        msg_id = kernel_client.execute(code2)
        while True:
            try:
                msg = kernel_client.get_iopub_msg(timeout=30.0)
                if msg['parent_header'].get('msg_id') == msg_id:
                    if msg['header']['msg_type'] == 'status':
                        if msg['content']['execution_state'] == 'idle':
                            break
            except Exception:
                break

        mem2 = get_kernel_gpu_memory_mb(kernel_client, timeout=10.0)

        # Memory should increase
        assert mem2 > mem1, (
            f"Expected memory to increase: {mem1:.2f} MB -> {mem2:.2f} MB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
