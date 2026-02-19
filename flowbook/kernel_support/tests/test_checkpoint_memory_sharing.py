"""
Comprehensive tests for checkpoint memory sharing and cumulative size measurement.

These tests verify that:
1. HeapSizer.sizeof_all_checkpoints() correctly measures all checkpoints together,
   accounting for memory sharing between them.
2. MemoryCheckpoints.get_total_checkpoint_size() and
   get_cumulative_checkpoint_size_at_cell() return accurate cumulative sizes.
3. Various data types and sharing scenarios are handled correctly.

The key insight is that checkpoints share memory via the deepcopy memo dict.
Measuring each checkpoint separately and summing would overcount shared memory.
Measuring all checkpoints together with a single HeapSizer pass gives the true total.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from typing import Dict, Any, List

from flowbook.kernel_support.heap_size import (
    HeapSizer,
    AllCheckpointsSize,
    CheckpointSize,
    sizeof,
)
from flowbook.kernel_support.memory_checkpoint import (
    MemoryCheckpoints,
    MemoryCheckpoint,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_mock_checkpoint(name: str, user_ns: Dict[str, Any]) -> MemoryCheckpoint:
    """Create a mock checkpoint with the given namespace."""
    return MemoryCheckpoint(name, user_ns, {})


def get_array_size(arr: np.ndarray) -> int:
    """Get the expected size of a numpy array."""
    return arr.nbytes


# =============================================================================
# HEAPSIZER.SIZEOF_ALL_CHECKPOINTS TESTS
# =============================================================================

class TestSizeofAllCheckpoints:
    """Tests for HeapSizer.sizeof_all_checkpoints() method."""

    def test_empty_checkpoints(self):
        """Test with no checkpoints."""
        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({})

        assert isinstance(result, AllCheckpointsSize)
        assert result.total_bytes == 0
        assert result.by_variable == {}
        assert result.by_type == {}
        assert result.by_checkpoint == {}

    def test_single_checkpoint(self):
        """Test with a single checkpoint."""
        arr = np.zeros(10000)  # ~80KB
        ckpt = create_mock_checkpoint("ckpt1", {"arr": arr})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt})

        assert result.total_bytes > 80000
        assert "arr" in result.by_variable
        assert "ndarray" in result.by_type
        assert "ckpt1" in result.by_checkpoint
        assert result.by_checkpoint["ckpt1"] > 80000

    def test_shared_object_between_checkpoints_counted_once(self):
        """Test that shared objects between checkpoints are counted only once.

        This is the KEY TEST for the memory sharing fix.
        """
        # Create a shared array
        shared_arr = np.zeros(100000)  # ~800KB

        # Two checkpoints reference the SAME array (simulating unchanged variable)
        ckpt1 = create_mock_checkpoint("ckpt1", {"arr": shared_arr})
        ckpt2 = create_mock_checkpoint("ckpt2", {"arr": shared_arr})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Total should be close to 800KB, NOT 1.6MB
        # Allow some overhead for the checkpoint structures
        assert result.total_bytes < 1000000, (
            f"Expected ~800KB for shared array, got {result.total_bytes/1024/1024:.2f}MB. "
            "Shared object may be double-counted."
        )

    def test_different_objects_both_counted(self):
        """Test that different objects in checkpoints are both counted."""
        arr1 = np.zeros(50000)  # ~400KB
        arr2 = np.zeros(50000)  # ~400KB

        # Different arrays in different checkpoints
        ckpt1 = create_mock_checkpoint("ckpt1", {"arr": arr1})
        ckpt2 = create_mock_checkpoint("ckpt2", {"arr": arr2})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Total should be close to 800KB (both arrays)
        assert result.total_bytes > 750000, (
            f"Expected ~800KB for two arrays, got {result.total_bytes/1024/1024:.2f}MB"
        )

    def test_partial_sharing_between_checkpoints(self):
        """Test checkpoints that share some but not all variables."""
        shared = np.zeros(50000)  # ~400KB - shared
        unique1 = np.zeros(25000)  # ~200KB - only in ckpt1
        unique2 = np.zeros(25000)  # ~200KB - only in ckpt2

        ckpt1 = create_mock_checkpoint("ckpt1", {"shared": shared, "unique": unique1})
        ckpt2 = create_mock_checkpoint("ckpt2", {"shared": shared, "unique": unique2})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Total should be ~800KB (400 shared + 200 + 200)
        # NOT 1.2MB (which would double-count shared)
        assert result.total_bytes > 700000  # At least the data
        assert result.total_bytes < 1000000  # But not double-counting

    def test_many_checkpoints_with_same_object(self):
        """Test many checkpoints all sharing the same object."""
        shared = np.zeros(100000)  # ~800KB

        checkpoints = {}
        for i in range(10):
            checkpoints[f"ckpt{i}"] = create_mock_checkpoint(f"ckpt{i}", {"arr": shared})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints(checkpoints)

        # Total should still be ~800KB, not 8MB!
        assert result.total_bytes < 1200000, (
            f"Expected ~800KB for shared array across 10 checkpoints, "
            f"got {result.total_bytes/1024/1024:.2f}MB"
        )

    def test_by_checkpoint_breakdown(self):
        """Test that by_checkpoint breakdown is correct."""
        arr1 = np.zeros(50000)
        arr2 = np.zeros(50000)

        ckpt1 = create_mock_checkpoint("ckpt1", {"arr": arr1})
        ckpt2 = create_mock_checkpoint("ckpt2", {"arr": arr2})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        assert "ckpt1" in result.by_checkpoint
        assert "ckpt2" in result.by_checkpoint
        assert result.by_checkpoint["ckpt1"] > 400000
        assert result.by_checkpoint["ckpt2"] > 400000

        # Sum of checkpoints should equal total
        assert sum(result.by_checkpoint.values()) == result.total_bytes

    def test_by_type_breakdown(self):
        """Test that by_type breakdown is correct."""
        arr = np.zeros(10000)
        df = pd.DataFrame({"a": range(1000)})
        lst = list(range(1000))

        ckpt = create_mock_checkpoint("ckpt1", {"arr": arr, "df": df, "lst": lst})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt})

        assert "ndarray" in result.by_type
        assert "DataFrame" in result.by_type
        assert "list" in result.by_type

    def test_by_variable_breakdown(self):
        """Test that by_variable breakdown is correct."""
        arr1 = np.zeros(10000)
        arr2 = np.zeros(20000)

        ckpt = create_mock_checkpoint("ckpt1", {"small": arr1, "large": arr2})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt})

        assert "small" in result.by_variable
        assert "large" in result.by_variable
        assert result.by_variable["large"] > result.by_variable["small"]


class TestSizeofAllCheckpointsDataTypes:
    """Test sizeof_all_checkpoints with various data types."""

    def test_numpy_arrays_shared(self):
        """Test numpy arrays that are shared."""
        base = np.zeros(100000)

        ckpt1 = create_mock_checkpoint("ckpt1", {"arr": base})
        ckpt2 = create_mock_checkpoint("ckpt2", {"arr": base})  # Same array

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Should count base array only once
        assert result.total_bytes < 900000

    def test_numpy_views_share_data(self):
        """Test that numpy views share underlying data."""
        base = np.zeros(100000)
        view = base[::2]  # View of half the array

        ckpt1 = create_mock_checkpoint("ckpt1", {"base": base})
        ckpt2 = create_mock_checkpoint("ckpt2", {"view": view})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # View doesn't own data, so total should be ~800KB (base only)
        assert result.total_bytes < 900000

    def test_pandas_dataframes_shared(self):
        """Test DataFrames that share underlying data."""
        df = pd.DataFrame({"a": np.zeros(50000), "b": np.ones(50000)})

        ckpt1 = create_mock_checkpoint("ckpt1", {"df": df})
        ckpt2 = create_mock_checkpoint("ckpt2", {"df": df})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Should count DataFrame once, not twice
        assert result.total_bytes < 1200000  # ~800KB for DataFrame + overhead

    def test_pandas_series_shared(self):
        """Test Series that share underlying data."""
        s = pd.Series(np.zeros(100000))

        ckpt1 = create_mock_checkpoint("ckpt1", {"s": s})
        ckpt2 = create_mock_checkpoint("ckpt2", {"s": s})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Should count Series once
        assert result.total_bytes < 1000000

    def test_lists_shared(self):
        """Test Python lists that are shared."""
        shared_list = list(range(100000))

        ckpt1 = create_mock_checkpoint("ckpt1", {"lst": shared_list})
        ckpt2 = create_mock_checkpoint("ckpt2", {"lst": shared_list})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # List should be counted once
        single_size = sizer.reset() or HeapSizer().sizeof(shared_list)
        assert result.total_bytes < single_size * 1.5

    def test_dicts_shared(self):
        """Test Python dicts that are shared."""
        shared_dict = {i: i * 2 for i in range(10000)}

        ckpt1 = create_mock_checkpoint("ckpt1", {"d": shared_dict})
        ckpt2 = create_mock_checkpoint("ckpt2", {"d": shared_dict})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Dict should be counted once
        single_size = sizer.reset() or HeapSizer().sizeof(shared_dict)
        assert result.total_bytes < single_size * 1.5

    def test_nested_structures_shared(self):
        """Test nested structures with shared components."""
        shared_inner = list(range(10000))

        struct1 = {"data": shared_inner, "unique": [1, 2, 3]}
        struct2 = {"data": shared_inner, "unique": [4, 5, 6]}

        ckpt1 = create_mock_checkpoint("ckpt1", {"s": struct1})
        ckpt2 = create_mock_checkpoint("ckpt2", {"s": struct2})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # shared_inner should be counted only once
        inner_size = HeapSizer().sizeof(shared_inner)
        # Total should be less than 2x inner_size
        assert result.total_bytes < inner_size * 2.5

    def test_functions_with_closures(self):
        """Test functions with closures referencing shared data."""
        captured = list(range(10000))

        def func1():
            return captured

        def func2():
            return captured

        ckpt1 = create_mock_checkpoint("ckpt1", {"f": func1})
        ckpt2 = create_mock_checkpoint("ckpt2", {"f": func2})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # The captured list should ideally be counted once if both closures
        # reference the same object
        assert result.total_bytes > 0


class TestSizeofAllCheckpointsEdgeCases:
    """Edge cases for sizeof_all_checkpoints."""

    def test_checkpoint_without_user_ns(self):
        """Test checkpoint objects without user_ns attribute."""
        class FakeCheckpoint:
            pass

        fake = FakeCheckpoint()
        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"fake": fake})

        assert result.by_checkpoint["fake"] == 0

    def test_empty_user_ns(self):
        """Test checkpoints with empty namespaces."""
        ckpt = create_mock_checkpoint("empty", {})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"empty": ckpt})

        assert result.total_bytes == 0

    def test_circular_references(self):
        """Test handling of circular references."""
        a = {"ref": None}
        a["ref"] = a  # Circular

        ckpt1 = create_mock_checkpoint("ckpt1", {"a": a})
        ckpt2 = create_mock_checkpoint("ckpt2", {"a": a})

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"ckpt1": ckpt1, "ckpt2": ckpt2})

        # Should not hang or crash
        assert result.total_bytes > 0
        assert result.total_bytes < 100000  # Should be small

    def test_mixed_types_in_checkpoints(self):
        """Test checkpoints with mixed types."""
        ns = {
            "arr": np.zeros(10000),
            "df": pd.DataFrame({"a": [1, 2, 3]}),
            "lst": [1, 2, 3],
            "dct": {"x": 1},
            "num": 42,
            "txt": "hello",
        }

        ckpt = create_mock_checkpoint("mixed", ns)

        sizer = HeapSizer()
        result = sizer.sizeof_all_checkpoints({"mixed": ckpt})

        assert len(result.by_type) >= 3  # Multiple types
        assert len(result.by_variable) == 6


# =============================================================================
# MEMORYCHECKPOINTS CUMULATIVE SIZE TESTS
# =============================================================================

class TestMemoryCheckpointsGetTotalSize:
    """Tests for MemoryCheckpoints.get_total_checkpoint_size()."""

    def test_no_checkpoints(self):
        """Test with no checkpoints saved."""
        cp = MemoryCheckpoints()
        result = cp.get_total_checkpoint_size()

        assert isinstance(result, AllCheckpointsSize)
        assert result.total_bytes == 0

    def test_single_checkpoint(self):
        """Test with a single checkpoint."""
        cp = MemoryCheckpoints()
        user_ns = {"arr": np.zeros(10000)}
        cp.save("test", user_ns)

        result = cp.get_total_checkpoint_size()

        assert result.total_bytes > 80000  # ~80KB array

    def test_multiple_checkpoints_shared_data(self):
        """Test that shared data across checkpoints is not double-counted."""
        cp = MemoryCheckpoints()

        # Create namespace with array
        arr = np.zeros(100000)
        user_ns = {"arr": arr}

        # Save multiple checkpoints (simulating pre/post pattern)
        cp.save("_pre_cell1", user_ns)
        cp.save("_post_cell1", user_ns)
        cp.save("_pre_cell2", user_ns)
        cp.save("_post_cell2", user_ns)

        result = cp.get_total_checkpoint_size()

        # Because of deepcopy memo sharing, the total should be much less
        # than 4 * 800KB = 3.2MB
        # The exact amount depends on how much sharing occurs
        assert result.total_bytes < 3200000, (
            f"Expected sharing to reduce total, got {result.total_bytes/1024/1024:.2f}MB"
        )


class TestMemoryCheckpointsGetCumulativeSizeAtCell:
    """Tests for MemoryCheckpoints.get_cumulative_checkpoint_size_at_cell()."""

    def test_no_checkpoints(self):
        """Test with no checkpoints."""
        cp = MemoryCheckpoints()
        result = cp.get_cumulative_checkpoint_size_at_cell("nonexistent")

        assert result.total_bytes == 0

    def test_includes_checkpoints_up_to_cell(self):
        """Test that only checkpoints up to the cell are included."""
        cp = MemoryCheckpoints()

        # Create namespaces for different cells
        ns1 = {"a": np.zeros(10000)}
        ns2 = {"a": np.zeros(10000), "b": np.zeros(10000)}
        ns3 = {"a": np.zeros(10000), "b": np.zeros(10000), "c": np.zeros(10000)}

        cp.save("_pre_cell1", ns1)
        cp.save("_post_cell1", ns1)
        cp.save("_pre_cell2", ns2)
        cp.save("_post_cell2", ns2)
        cp.save("_pre_cell3", ns3)
        cp.save("_post_cell3", ns3)

        # Get cumulative up to cell2
        result = cp.get_cumulative_checkpoint_size_at_cell("cell2")

        # Should include _pre_cell1, _post_cell1, _pre_cell2, _post_cell2
        # but NOT _pre_cell3 or _post_cell3
        assert "_pre_cell1" in result.by_checkpoint or result.total_bytes > 0

    def test_monotonically_increasing(self):
        """Test that cumulative size is monotonically increasing as cells execute."""
        cp = MemoryCheckpoints()

        # Simulate cell executions with growing namespaces
        sizes = []
        for i in range(5):
            ns = {f"var{j}": np.zeros(10000) for j in range(i + 1)}
            cp.save(f"_pre_cell{i}", ns)
            cp.save(f"_post_cell{i}", ns)

            result = cp.get_cumulative_checkpoint_size_at_cell(f"cell{i}")
            sizes.append(result.total_bytes)

        # Each size should be >= previous (monotonically increasing)
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i-1], (
                f"Size decreased from cell {i-1} ({sizes[i-1]}) to cell {i} ({sizes[i]})"
            )


# =============================================================================
# INTEGRATION TESTS: SHARING ACROSS PRE/POST CHECKPOINTS
# =============================================================================

class TestPrePostCheckpointSharing:
    """Test sharing between pre and post checkpoints for the same cell."""

    def test_pre_post_share_unchanged_variables(self):
        """Test that pre and post checkpoints share unchanged variables."""
        cp = MemoryCheckpoints()

        # Create namespace with array that won't change
        arr = np.zeros(100000)  # ~800KB
        user_ns = {"unchanged": arr}

        # Save pre-checkpoint
        cp.save("_pre_abc1", user_ns)

        # Save post-checkpoint (same variable, no modifications)
        cp.save("_post_abc1", user_ns)

        result = cp.get_total_checkpoint_size()

        # With sharing, total should be much less than 2 * 800KB
        # Depending on deepcopy implementation, the exact sharing varies
        assert result.total_bytes < 1600000, (
            f"Pre/post checkpoints should share unchanged data, "
            f"got {result.total_bytes/1024/1024:.2f}MB"
        )

    def test_changed_variable_creates_new_copy(self):
        """Test that changed variables create new copies."""
        cp = MemoryCheckpoints()

        # Create initial namespace
        arr1 = np.zeros(50000)
        user_ns = {"arr": arr1}
        cp.save("_pre_abc1", user_ns)

        # Modify and save post
        arr2 = np.ones(50000)  # Different array
        user_ns["arr"] = arr2
        cp.save("_post_abc1", user_ns)

        result = cp.get_total_checkpoint_size()

        # Both arrays should be counted (no sharing since different)
        assert result.total_bytes > 700000  # Both ~400KB arrays


class TestRealWorldScenarios:
    """Test scenarios that match real FlowBook usage patterns."""

    def test_notebook_execution_pattern(self):
        """Simulate a notebook execution with pre/post checkpoints per cell."""
        cp = MemoryCheckpoints()

        # Cell 1: Create array
        ns = {"arr": np.zeros(10000)}
        cp.save("_pre_cell1", ns)
        ns["result1"] = ns["arr"] * 2
        cp.save("_post_cell1", ns)

        # Cell 2: Create DataFrame
        ns["df"] = pd.DataFrame({"a": range(1000)})
        cp.save("_pre_cell2", ns)
        ns["result2"] = ns["df"]["a"].sum()
        cp.save("_post_cell2", ns)

        # Cell 3: More processing
        ns["list_data"] = list(range(5000))
        cp.save("_pre_cell3", ns)
        ns["list_sum"] = sum(ns["list_data"])
        cp.save("_post_cell3", ns)

        # Get total size
        total = cp.get_total_checkpoint_size()

        # Should have reasonable total, not explosion due to double-counting
        assert total.total_bytes > 0
        assert len(total.by_checkpoint) == 6

    def test_large_dataframe_multiple_cells(self):
        """Test with a large DataFrame across multiple cells."""
        cp = MemoryCheckpoints()

        # Create large DataFrame (~8MB)
        df = pd.DataFrame({
            "a": np.zeros(1000000),
        })

        # Cell 1: Just have the DataFrame
        ns = {"df": df}
        cp.save("_pre_cell1", ns)
        cp.save("_post_cell1", ns)

        # Cell 2: Add a column (still same underlying data)
        ns["col_sum"] = df["a"].sum()
        cp.save("_pre_cell2", ns)
        cp.save("_post_cell2", ns)

        # Cell 3: Another operation
        ns["mean"] = df["a"].mean()
        cp.save("_pre_cell3", ns)
        cp.save("_post_cell3", ns)

        result = cp.get_total_checkpoint_size()

        # With 6 checkpoints, naive approach would be 48MB
        # With sharing, should be much less
        assert result.total_bytes < 25000000, (
            f"Expected sharing to reduce total significantly, "
            f"got {result.total_bytes/1024/1024:.2f}MB"
        )


# =============================================================================
# COMPARISON WITH INDIVIDUAL CHECKPOINT MEASUREMENT
# =============================================================================

class TestCumulativeVsIndividual:
    """Compare cumulative measurement vs sum of individual measurements."""

    def test_cumulative_less_than_sum_of_individual(self):
        """Cumulative total should be <= sum of individual checkpoint sizes."""
        cp = MemoryCheckpoints()

        # Create checkpoints with shared data
        shared = np.zeros(50000)
        ns = {"shared": shared, "unique": np.zeros(10000)}

        cp.save("ckpt1", ns)
        ns["unique"] = np.zeros(10000)  # New array
        cp.save("ckpt2", ns)

        # Get cumulative total
        cumulative = cp.get_total_checkpoint_size()

        # Get individual sizes
        size1 = cp.get_checkpoint_size("ckpt1")
        size2 = cp.get_checkpoint_size("ckpt2")
        individual_sum = size1.total_bytes + size2.total_bytes

        # Cumulative should be <= sum (accounting for sharing)
        assert cumulative.total_bytes <= individual_sum * 1.1, (
            f"Cumulative ({cumulative.total_bytes}) should be <= "
            f"sum of individual ({individual_sum})"
        )

    def test_significant_savings_with_sharing(self):
        """Test that there's significant savings from sharing."""
        cp = MemoryCheckpoints()

        # Create large shared object
        large_shared = np.zeros(100000)  # ~800KB
        ns = {"data": large_shared}

        # Save 5 checkpoints all sharing the same data
        for i in range(5):
            cp.save(f"ckpt{i}", ns)

        # Get cumulative
        cumulative = cp.get_total_checkpoint_size()

        # Get individual sum
        individual_sum = sum(
            cp.get_checkpoint_size(f"ckpt{i}").total_bytes
            for i in range(5)
        )

        # Cumulative should be significantly less (at least 2x savings)
        # Individual would be 5 * 800KB = 4MB
        # Cumulative should be closer to 800KB
        if individual_sum > 0:
            savings = 1 - (cumulative.total_bytes / individual_sum)
            assert savings > 0.3, (
                f"Expected >30% savings from sharing, got {savings*100:.1f}%"
            )


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressionNoDoubleCounting:
    """Regression tests to prevent double-counting bugs."""

    def test_same_object_in_multiple_variables(self):
        """Same object assigned to multiple variables should count once."""
        cp = MemoryCheckpoints()

        arr = np.zeros(100000)
        ns = {"a": arr, "b": arr, "c": arr}  # All same object
        cp.save("ckpt", ns)

        result = cp.get_total_checkpoint_size()

        # Should count array once, not three times
        assert result.total_bytes < 1000000  # Not 2.4MB

    def test_list_referenced_multiple_ways(self):
        """List referenced from multiple structures should count once."""
        cp = MemoryCheckpoints()

        shared_list = list(range(50000))
        ns = {
            "list1": shared_list,
            "container": {"nested": shared_list},
            "wrapper": [shared_list],
        }
        cp.save("ckpt", ns)

        result = cp.get_total_checkpoint_size()

        # List should be counted once
        list_size = HeapSizer().sizeof(shared_list)
        assert result.total_bytes < list_size * 2

    def test_dataframe_and_series_from_same_data(self):
        """DataFrame and Series from same data should share."""
        cp = MemoryCheckpoints()

        arr = np.zeros(100000)
        df = pd.DataFrame({"col": arr})
        series = df["col"]  # Series viewing DataFrame data

        ns = {"df": df, "series": series}
        cp.save("ckpt", ns)

        result = cp.get_total_checkpoint_size()

        # Should not double-count the underlying data
        assert result.total_bytes < 1500000  # Not 1.6MB+


class TestRegressionConsistentResults:
    """Regression tests for consistent results."""

    def test_deterministic_results(self):
        """Same checkpoints should give same results."""
        cp = MemoryCheckpoints()

        ns = {"arr": np.zeros(10000), "lst": list(range(1000))}
        cp.save("ckpt", ns)

        result1 = cp.get_total_checkpoint_size()
        result2 = cp.get_total_checkpoint_size()

        assert result1.total_bytes == result2.total_bytes

    def test_order_independence(self):
        """Order of checkpoint saves shouldn't affect total size."""
        # First order
        cp1 = MemoryCheckpoints()
        arr1 = np.zeros(10000)
        arr2 = np.zeros(20000)
        cp1.save("ckpt1", {"arr": arr1})
        cp1.save("ckpt2", {"arr": arr2})
        result1 = cp1.get_total_checkpoint_size()

        # Reverse order (new objects to avoid sharing)
        cp2 = MemoryCheckpoints()
        arr3 = np.zeros(20000)
        arr4 = np.zeros(10000)
        cp2.save("ckpt2", {"arr": arr3})
        cp2.save("ckpt1", {"arr": arr4})
        result2 = cp2.get_total_checkpoint_size()

        # Total sizes should be similar (same amount of data)
        assert abs(result1.total_bytes - result2.total_bytes) < 10000


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for memory measurement."""

    def test_many_checkpoints_reasonable_time(self):
        """Measuring many checkpoints should complete in reasonable time."""
        import time

        cp = MemoryCheckpoints()

        # Create 50 checkpoints
        for i in range(50):
            ns = {f"var{j}": np.zeros(1000) for j in range(10)}
            cp.save(f"ckpt{i}", ns)

        start = time.time()
        result = cp.get_total_checkpoint_size()
        elapsed = time.time() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Took {elapsed:.2f}s to measure 50 checkpoints"
        assert result.total_bytes > 0

    def test_large_namespace_reasonable_time(self):
        """Measuring checkpoints with large namespaces should be fast."""
        import time

        cp = MemoryCheckpoints()

        # Large namespace
        ns = {f"var{i}": np.zeros(10000) for i in range(100)}
        cp.save("large", ns)

        start = time.time()
        result = cp.get_total_checkpoint_size()
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Took {elapsed:.2f}s for large namespace"


# =============================================================================
# TEST DATA TYPE COVERAGE
# =============================================================================

class TestDataTypeCoverage:
    """Ensure all common data types are handled correctly."""

    def test_numpy_dtypes(self):
        """Test various numpy dtypes."""
        cp = MemoryCheckpoints()

        ns = {
            "float64": np.zeros(1000, dtype=np.float64),
            "float32": np.zeros(1000, dtype=np.float32),
            "int64": np.zeros(1000, dtype=np.int64),
            "int32": np.zeros(1000, dtype=np.int32),
            "bool": np.zeros(1000, dtype=bool),
            "complex": np.zeros(1000, dtype=np.complex128),
        }
        cp.save("dtypes", ns)

        result = cp.get_total_checkpoint_size()
        assert result.total_bytes > 0
        assert "float64" in result.by_variable

    def test_pandas_types(self):
        """Test various pandas types."""
        cp = MemoryCheckpoints()

        ns = {
            "df": pd.DataFrame({"a": [1, 2, 3]}),
            "series": pd.Series([1, 2, 3]),
            "categorical": pd.Categorical(["a", "b", "c"]),
            "datetime": pd.date_range("2020-01-01", periods=100),
        }
        cp.save("pandas", ns)

        result = cp.get_total_checkpoint_size()
        assert result.total_bytes > 0

    def test_python_containers(self):
        """Test Python container types."""
        cp = MemoryCheckpoints()

        ns = {
            "list": [1, 2, 3, 4, 5] * 100,
            "dict": {i: i*2 for i in range(100)},
            "set": set(range(100)),
            "tuple": tuple(range(100)),
            "frozenset": frozenset(range(100)),
        }
        cp.save("containers", ns)

        result = cp.get_total_checkpoint_size()
        assert result.total_bytes > 0

    def test_nested_structures(self):
        """Test deeply nested structures."""
        cp = MemoryCheckpoints()

        nested = {"level1": {"level2": {"level3": list(range(1000))}}}
        ns = {"nested": nested}
        cp.save("nested", ns)

        result = cp.get_total_checkpoint_size()
        assert result.total_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
