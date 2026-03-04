"""Integration tests for DataFrame subset checkpoint optimization.

Tests the integration between MemoryCheckpoints and the DataFrame subset
detector, verifying that:
1. Subset optimization can be enabled/disabled
2. Subsets are correctly excluded from deep copy
3. Subsets are correctly reconstructed on restore
4. Save/restore roundtrips preserve data integrity
"""

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


# ============================================================================
# BASIC SAVE/RESTORE WITH SUBSET OPTIMIZATION
# ============================================================================


class TestBasicSubsetOptimization:
    """Test basic save/restore with DataFrame subset optimization."""

    def test_optimization_disabled_by_default(self):
        """Test that optimization is disabled by default."""
        cp = MemoryCheckpoints()
        status = cp.get_df_subset_optimization_status()
        assert status["enabled"] is False

    def test_enable_optimization(self):
        """Test enabling optimization."""
        cp = MemoryCheckpoints()
        cp.set_df_subset_optimization(True)
        status = cp.get_df_subset_optimization_status()
        assert status["enabled"] is True

    def test_disable_optimization(self):
        """Test disabling optimization."""
        cp = MemoryCheckpoints(optimize_df_subsets=True)
        cp.set_df_subset_optimization(False)
        status = cp.get_df_subset_optimization_status()
        assert status["enabled"] is False

    def test_simple_subset_save_restore(self):
        """Test save/restore with a simple row subset."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        # Create parent and child DataFrames
        df = pd.DataFrame({
            "a": range(500),
            "b": range(500, 1000),
            "c": ["x", "y", "z", "w", "v"] * 100,
        })
        df_filtered = df[df["a"] > 250].copy()

        user_ns = {"df": df, "df_filtered": df_filtered}

        # Save checkpoint
        cp.save("test", user_ns)

        # Modify both DataFrames
        user_ns["df"].loc[0, "a"] = 9999
        user_ns["df_filtered"].loc[251, "a"] = 8888

        # Restore
        cp.restore("test", user_ns)

        # Verify parent restored
        assert user_ns["df"].loc[0, "a"] == 0

        # Verify child restored correctly
        assert user_ns["df_filtered"].loc[251, "a"] == 251
        assert len(user_ns["df_filtered"]) == 249

    def test_restore_preserves_values(self):
        """Test that restored subset has correct values."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df = pd.DataFrame({
            "a": range(300),
            "b": np.random.randn(300),
        })
        df_filtered = df[df["a"] > 150].copy()
        original_filtered = df_filtered.copy()

        user_ns = {"df": df, "df_filtered": df_filtered}
        cp.save("test", user_ns)

        # Modify
        user_ns["df_filtered"]["b"] = 0

        # Restore
        cp.restore("test", user_ns)

        # Should match original
        pd.testing.assert_frame_equal(
            user_ns["df_filtered"].reset_index(drop=True),
            original_filtered.reset_index(drop=True),
        )


class TestSubsetWithExtraColumns:
    """Test subset optimization when child has extra columns."""

    def test_extra_columns_preserved(self):
        """Test that extra columns in child are preserved after restore."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=0,  # Allow all relations
        )

        df = pd.DataFrame({"a": range(300)})
        df_filtered = df[df["a"] > 150].copy()
        df_filtered["extra"] = df_filtered["a"] * 2
        df_filtered["extra2"] = "constant"

        original_filtered = df_filtered.copy()

        user_ns = {"df": df, "df_filtered": df_filtered}
        cp.save("test", user_ns)

        # Modify
        user_ns["df_filtered"]["extra"] = 0

        # Restore
        cp.restore("test", user_ns)

        # Extra columns should be restored
        pd.testing.assert_frame_equal(
            user_ns["df_filtered"].reset_index(drop=True),
            original_filtered.reset_index(drop=True),
        )

    def test_column_order_preserved(self):
        """Test that column order is preserved after restore."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=0,
        )

        df = pd.DataFrame({"a": range(300), "b": range(300)})
        df_filtered = df[df["a"] > 150].copy()
        df_filtered["c"] = "new"
        df_filtered = df_filtered[["c", "a", "b"]]  # Reorder columns

        user_ns = {"df": df, "df_filtered": df_filtered}
        cp.save("test", user_ns)
        cp.restore("test", user_ns)

        assert list(user_ns["df_filtered"].columns) == ["c", "a", "b"]


class TestChainOfSubsets:
    """Test chains of subset relationships."""

    def test_two_level_chain(self):
        """Test chain: df -> df_a -> df_b."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df = pd.DataFrame({"x": range(1000)})
        df_a = df[df["x"] > 300].copy()  # ~700 rows
        df_b = df_a[df_a["x"] > 600].copy()  # ~400 rows

        original_a = df_a.copy()
        original_b = df_b.copy()

        user_ns = {"df": df, "df_a": df_a, "df_b": df_b}
        cp.save("test", user_ns)

        # Modify all
        user_ns["df"]["x"] = 0
        user_ns["df_a"]["x"] = 0
        user_ns["df_b"]["x"] = 0

        # Restore
        cp.restore("test", user_ns)

        # All should be restored
        pd.testing.assert_frame_equal(
            user_ns["df_a"].reset_index(drop=True),
            original_a.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            user_ns["df_b"].reset_index(drop=True),
            original_b.reset_index(drop=True),
        )

    def test_multiple_children_same_parent(self):
        """Test multiple children from same parent."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df = pd.DataFrame({
            "category": ["A", "B", "C", "D"] * 200,
            "value": range(800),
        })
        df_a = df[df["category"] == "A"].copy()
        df_b = df[df["category"] == "B"].copy()
        df_c = df[df["category"] == "C"].copy()

        original_a = df_a.copy()
        original_b = df_b.copy()
        original_c = df_c.copy()

        user_ns = {"df": df, "df_a": df_a, "df_b": df_b, "df_c": df_c}
        cp.save("test", user_ns)

        # Modify children
        user_ns["df_a"]["value"] = -1
        user_ns["df_b"]["value"] = -2
        user_ns["df_c"]["value"] = -3

        # Restore
        cp.restore("test", user_ns)

        pd.testing.assert_frame_equal(
            user_ns["df_a"].reset_index(drop=True),
            original_a.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            user_ns["df_b"].reset_index(drop=True),
            original_b.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            user_ns["df_c"].reset_index(drop=True),
            original_c.reset_index(drop=True),
        )


class TestMixedVariables:
    """Test with mix of DataFrames and other variables."""

    def test_non_dataframes_unaffected(self):
        """Test that non-DataFrame variables work normally."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df = pd.DataFrame({"a": range(300)})
        df_filtered = df[df["a"] > 150].copy()

        user_ns = {
            "df": df,
            "df_filtered": df_filtered,
            "x": 42,
            "y": [1, 2, 3],
            "z": {"key": "value"},
        }

        cp.save("test", user_ns)

        user_ns["x"] = 999
        user_ns["y"].append(4)
        user_ns["z"]["key"] = "modified"

        cp.restore("test", user_ns)

        assert user_ns["x"] == 42
        assert user_ns["y"] == [1, 2, 3]
        assert user_ns["z"]["key"] == "value"

    def test_standalone_dataframes_work(self):
        """Test DataFrames with no subset relationship work normally."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df1 = pd.DataFrame({"a": range(200)})
        df2 = pd.DataFrame({"b": range(300, 500)})  # Unrelated to df1

        original_df1 = df1.copy()
        original_df2 = df2.copy()

        user_ns = {"df1": df1, "df2": df2}
        cp.save("test", user_ns)

        user_ns["df1"]["a"] = -1
        user_ns["df2"]["b"] = -1

        cp.restore("test", user_ns)

        pd.testing.assert_frame_equal(user_ns["df1"], original_df1)
        pd.testing.assert_frame_equal(user_ns["df2"], original_df2)


class TestOptimizationDisabled:
    """Test behavior when optimization is disabled."""

    def test_no_optimization_saves_full_copies(self):
        """Verify disabled optimization saves full copies."""
        cp = MemoryCheckpoints(optimize_df_subsets=False)

        df = pd.DataFrame({"a": range(300)})
        df_filtered = df[df["a"] > 150].copy()

        original_filtered = df_filtered.copy()

        user_ns = {"df": df, "df_filtered": df_filtered}
        cp.save("test", user_ns)

        # The checkpoint should have no subset relations
        checkpoint = cp.saved["test"]
        assert checkpoint._df_subset_relations == []

        # But restore should still work correctly
        user_ns["df_filtered"]["a"] = -1
        cp.restore("test", user_ns)

        pd.testing.assert_frame_equal(
            user_ns["df_filtered"].reset_index(drop=True),
            original_filtered.reset_index(drop=True),
        )


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_namespace(self):
        """Test with empty namespace."""
        cp = MemoryCheckpoints(optimize_df_subsets=True)

        user_ns = {}
        cp.save("test", user_ns)
        cp.restore("test", user_ns)

        assert user_ns == {}

    def test_small_dataframes_not_optimized(self):
        """Test that small DataFrames below threshold are not optimized."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=100,  # Higher threshold
        )

        df = pd.DataFrame({"a": range(50)})
        df_filtered = df[df["a"] > 25].copy()

        user_ns = {"df": df, "df_filtered": df_filtered}
        cp.save("test", user_ns)

        # Check no relations were created (both below threshold)
        checkpoint = cp.saved["test"]
        assert checkpoint._df_subset_relations == []

    def test_multiple_checkpoints(self):
        """Test multiple checkpoints with subset optimization."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df = pd.DataFrame({"a": range(300)})
        df_filtered = df[df["a"] > 150].copy()

        user_ns = {"df": df, "df_filtered": df_filtered}

        # Save first checkpoint
        cp.save("cp1", user_ns)

        # Modify
        user_ns["df"]["a"] = user_ns["df"]["a"] + 1000
        df_filtered2 = user_ns["df"][user_ns["df"]["a"] > 1200].copy()
        user_ns["df_filtered"] = df_filtered2

        # Save second checkpoint
        cp.save("cp2", user_ns)

        # Restore first
        cp.restore("cp1", user_ns)
        assert user_ns["df"]["a"].iloc[0] == 0

        # Restore second
        cp.restore("cp2", user_ns)
        assert user_ns["df"]["a"].iloc[0] == 1000

    def test_dtype_preservation(self):
        """Test that dtypes are preserved through save/restore."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=10,
            df_subset_min_savings_bytes=100,
        )

        df = pd.DataFrame({
            "int32": np.array(range(300), dtype=np.int32),
            "float64": np.array(range(300), dtype=np.float64),
            "bool": [True, False] * 150,
            "category": pd.Categorical(["a", "b", "c"] * 100),
        })
        df_filtered = df[df["int32"] > 150].copy()

        original_dtypes = df_filtered.dtypes.copy()

        user_ns = {"df": df, "df_filtered": df_filtered}
        cp.save("test", user_ns)
        cp.restore("test", user_ns)

        # Check dtypes preserved
        for col in df_filtered.columns:
            assert user_ns["df_filtered"][col].dtype == original_dtypes[col]


class TestConfigurationOptions:
    """Test configuration options for subset detection."""

    def test_custom_min_rows(self):
        """Test custom min_rows setting."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_rows=200,
        )

        status = cp.get_df_subset_optimization_status()
        assert status["min_rows"] == 200

    def test_custom_min_savings(self):
        """Test custom min_savings_bytes setting."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_min_savings_bytes=50000,
        )

        status = cp.get_df_subset_optimization_status()
        assert status["min_savings_bytes"] == 50000

    def test_custom_max_dataframes(self):
        """Test custom max_dataframes setting."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_max_dataframes=10,
        )

        status = cp.get_df_subset_optimization_status()
        assert status["max_dataframes"] == 10

    def test_custom_timeout(self):
        """Test custom timeout_ms setting."""
        cp = MemoryCheckpoints(
            optimize_df_subsets=True,
            df_subset_timeout_ms=500,
        )

        status = cp.get_df_subset_optimization_status()
        assert status["timeout_ms"] == 500
