"""Unit tests for GPU memory measurement utilities using pynvml."""

import pytest
from flowbook.util.gpu_memory import get_gpu_memory_mb, has_gpu, _init_pynvml


class TestHasGpu:
    """Tests for has_gpu function."""

    def test_returns_bool(self):
        """Test that has_gpu returns a boolean."""
        result = has_gpu()
        assert isinstance(result, bool)

    def test_consistent_results(self):
        """Test that multiple calls return consistent results."""
        result1 = has_gpu()
        result2 = has_gpu()
        assert result1 == result2


class TestGetGpuMemoryMb:
    """Tests for get_gpu_memory_mb function."""

    def test_returns_float(self):
        """Test that get_gpu_memory_mb returns a float."""
        result = get_gpu_memory_mb()
        assert isinstance(result, float)

    def test_returns_non_negative(self):
        """Test that get_gpu_memory_mb returns a non-negative value."""
        result = get_gpu_memory_mb()
        assert result >= 0.0

    def test_consistent_when_no_allocation(self):
        """Test that multiple calls without GPU allocation return consistent low values."""
        result1 = get_gpu_memory_mb()
        result2 = get_gpu_memory_mb()
        # Both should be similar (could be 0 or small context overhead)
        assert abs(result1 - result2) < 10.0  # Within 10 MB


class TestLazyInitialization:
    """Tests for lazy pynvml initialization."""

    def test_init_pynvml_returns_bool(self):
        """Test that _init_pynvml returns a boolean."""
        result = _init_pynvml()
        assert isinstance(result, bool)

    def test_init_pynvml_consistent(self):
        """Test that _init_pynvml returns consistent results."""
        result1 = _init_pynvml()
        result2 = _init_pynvml()
        assert result1 == result2


@pytest.mark.skipif(not has_gpu(), reason="No GPU available")
class TestPynvmlDetectsAllocation:
    """Tests that pynvml detects GPU memory allocation from various frameworks."""

    def test_detects_cupy_allocation(self):
        """Test that pynvml detects CuPy array allocation."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        initial_mem = get_gpu_memory_mb()

        # Allocate ~100 MB on GPU
        arr = cp.zeros(25 * 1024 * 1024, dtype=cp.float32)  # 100 MB
        cp.cuda.Device().synchronize()

        mem_after = get_gpu_memory_mb()

        # Should detect memory increase
        assert mem_after > initial_mem, (
            f"Expected memory increase after CuPy allocation: "
            f"{initial_mem:.2f} MB -> {mem_after:.2f} MB"
        )

        del arr
        cp.get_default_memory_pool().free_all_blocks()

    def test_detects_numba_cuda_allocation(self):
        """Test that pynvml detects Numba CUDA allocation."""
        try:
            from numba import cuda
            import numpy as np
        except ImportError:
            pytest.skip("Numba not available")

        initial_mem = get_gpu_memory_mb()

        # Allocate ~100 MB on GPU
        host_array = np.zeros(25 * 1024 * 1024, dtype=np.float32)
        device_array = cuda.to_device(host_array)
        cuda.synchronize()

        mem_after = get_gpu_memory_mb()

        assert mem_after > initial_mem, (
            f"Expected memory increase after Numba CUDA allocation: "
            f"{initial_mem:.2f} MB -> {mem_after:.2f} MB"
        )

        del device_array
        cuda.current_context().deallocations.clear()

    def test_memory_scales_with_allocation_size(self):
        """Test that larger allocations show more memory usage."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        # First allocation: ~50 MB
        arr1 = cp.zeros(12 * 1024 * 1024, dtype=cp.float32)  # 48 MB
        cp.cuda.Device().synchronize()
        mem1 = get_gpu_memory_mb()

        # Second allocation: additional ~100 MB
        arr2 = cp.zeros(25 * 1024 * 1024, dtype=cp.float32)  # 100 MB
        cp.cuda.Device().synchronize()
        mem2 = get_gpu_memory_mb()

        assert mem2 > mem1, f"Expected memory to increase: {mem1:.2f} MB -> {mem2:.2f} MB"

        del arr1, arr2
        cp.get_default_memory_pool().free_all_blocks()


def _cudf_available() -> bool:
    """Check if cuDF is available."""
    try:
        import cudf  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_gpu() or not _cudf_available(), reason="No GPU or cuDF unavailable")
class TestPynvmlDetectsCudf:
    """Tests that pynvml detects cuDF DataFrame GPU memory."""

    def test_detects_cudf_dataframe_allocation(self):
        """Test that cuDF DataFrame GPU memory is detected."""
        import cudf
        import numpy as np

        initial_mem = get_gpu_memory_mb()

        # Create a ~100 MB cuDF DataFrame
        size_mb = 100
        n_rows = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        df = cudf.DataFrame({'col': np.zeros(n_rows, dtype=np.float32)})

        mem_after = get_gpu_memory_mb()

        assert mem_after > initial_mem, (
            f"Expected GPU memory to increase after cuDF allocation: "
            f"{initial_mem:.2f} MB -> {mem_after:.2f} MB"
        )

        # The delta should be roughly the allocation size
        delta = mem_after - initial_mem
        assert delta >= size_mb * 0.9, (
            f"Expected ~{size_mb} MB increase, got {delta:.2f} MB delta"
        )

        del df

    def test_cudf_memory_scales_with_size(self):
        """Test that larger cuDF DataFrames show proportionally more memory."""
        import cudf
        import numpy as np
        import gc

        # First allocation: ~50 MB
        n_rows1 = 50 * 1024 * 1024 // 4
        df1 = cudf.DataFrame({'col': np.zeros(n_rows1, dtype=np.float32)})
        mem1 = get_gpu_memory_mb()

        # Second allocation: additional ~100 MB
        n_rows2 = 100 * 1024 * 1024 // 4
        df2 = cudf.DataFrame({'col': np.zeros(n_rows2, dtype=np.float32)})
        mem2 = get_gpu_memory_mb()

        # Memory should increase by ~100 MB
        delta = mem2 - mem1
        assert delta >= 90, (
            f"Expected ~100 MB increase, got {delta:.2f} MB "
            f"({mem1:.2f} MB -> {mem2:.2f} MB)"
        )

        del df1, df2
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
