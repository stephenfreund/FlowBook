"""Unit tests for GPU memory measurement utilities."""

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
class TestGpuMemoryWithAllocation:
    """Tests that require actual GPU allocation."""

    def test_detects_gpu_memory_allocation(self):
        """Test that GPU memory allocation is detected."""
        from numba import cuda
        import numpy as np

        initial_mem = get_gpu_memory_mb()

        # Allocate ~100 MB on GPU
        size = 25 * 1024 * 1024  # 25M floats = 100 MB
        host_array = np.zeros(size, dtype=np.float32)
        device_array = cuda.to_device(host_array)
        cuda.synchronize()

        mem_after = get_gpu_memory_mb()

        # Should detect significant memory increase (accounting for CUDA context)
        assert mem_after > 50.0, f"Expected >50 MB after allocation, got {mem_after:.2f} MB"

        # Cleanup
        del device_array
        cuda.current_context().deallocations.clear()

    def test_memory_increases_with_larger_allocation(self):
        """Test that larger allocations show more memory usage."""
        from numba import cuda
        import numpy as np

        # First allocation: ~50 MB
        size1 = 12 * 1024 * 1024  # ~48 MB
        arr1 = cuda.to_device(np.zeros(size1, dtype=np.float32))
        cuda.synchronize()
        mem1 = get_gpu_memory_mb()

        # Second allocation: additional ~100 MB
        size2 = 25 * 1024 * 1024  # ~100 MB
        arr2 = cuda.to_device(np.zeros(size2, dtype=np.float32))
        cuda.synchronize()
        mem2 = get_gpu_memory_mb()

        # Memory should increase
        assert mem2 > mem1, f"Expected memory to increase: {mem1:.2f} MB -> {mem2:.2f} MB"

        # Cleanup
        del arr1, arr2
        cuda.current_context().deallocations.clear()


def _cudf_available() -> bool:
    """Check if cuDF is available."""
    try:
        import cudf
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_gpu() or not _cudf_available(), reason="No GPU or cuDF unavailable")
class TestGpuMemoryWithCudf:
    """Tests for GPU memory measurement with cuDF DataFrames."""

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

        # Should detect memory increase (accounting for CUDA context overhead)
        # With CUDA context, expect at least the allocation size
        assert mem_after > initial_mem, (
            f"Expected GPU memory to increase after cuDF allocation: "
            f"{initial_mem:.2f} MB -> {mem_after:.2f} MB"
        )

        # The delta should be roughly the allocation size (allow some overhead)
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

    def test_cudf_memory_decreases_on_delete(self):
        """Test that deleting cuDF DataFrames releases GPU memory."""
        import cudf
        import numpy as np
        import gc

        # Create a 200 MB DataFrame
        n_rows = 200 * 1024 * 1024 // 4
        df = cudf.DataFrame({'col': np.zeros(n_rows, dtype=np.float32)})
        mem_with_df = get_gpu_memory_mb()

        # Delete and collect
        del df
        gc.collect()

        mem_after_delete = get_gpu_memory_mb()

        # Memory should decrease (may not fully release due to RMM pooling)
        # At minimum, check that delete doesn't INCREASE memory
        assert mem_after_delete <= mem_with_df, (
            f"Memory should not increase after delete: "
            f"{mem_with_df:.2f} MB -> {mem_after_delete:.2f} MB"
        )

    def test_cudf_multiple_columns(self):
        """Test GPU memory measurement with multi-column cuDF DataFrame."""
        import cudf
        import numpy as np

        initial_mem = get_gpu_memory_mb()

        # Create a DataFrame with 4 columns, ~100 MB total
        n_rows = 25 * 1024 * 1024 // 4  # 25 MB per column
        df = cudf.DataFrame({
            'a': np.zeros(n_rows, dtype=np.float32),
            'b': np.zeros(n_rows, dtype=np.float32),
            'c': np.zeros(n_rows, dtype=np.float32),
            'd': np.zeros(n_rows, dtype=np.float32),
        })

        mem_after = get_gpu_memory_mb()

        # Should detect ~100 MB increase
        delta = mem_after - initial_mem
        assert delta >= 90, (
            f"Expected ~100 MB for 4x25MB columns, got {delta:.2f} MB delta"
        )

        del df


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
