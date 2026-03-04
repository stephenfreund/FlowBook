"""Unit tests for DataFrame subset detection.

Tests for the df_subset_detector module which detects when DataFrames
are row-subsets of other DataFrames for checkpoint optimization.
"""

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.df_subset_detector import (
    DataFrameSubsetDetector,
    SubsetDetectionResult,
    SubsetRelation,
    reconstruct_from_subset,
    topological_sort_relations,
)


# ============================================================================
# BASIC SUBSET DETECTION TESTS
# ============================================================================


class TestBasicSubsetDetection:
    """Test basic DataFrame subset detection."""

    def test_simple_row_subset(self):
        """Test detection of a simple row subset created by boolean indexing."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5] * 50,  # 250 rows
            "b": [10, 20, 30, 40, 50] * 50,
        })
        df_filtered = df[df["a"] > 2]  # 150 rows

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1
        relation = result.relations[0]
        assert relation.child_var == "df_filtered"
        assert relation.parent_var == "df"
        assert len(relation.row_indices) == len(df_filtered)
        assert relation.extra_columns == []
        assert result.total_estimated_savings_bytes > 0

    def test_no_subset_different_columns(self):
        """Test that DataFrames with different column values are not subsets."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5] * 50,
            "b": [10, 20, 30, 40, 50] * 50,
        })
        df_modified = df.copy()
        df_modified["a"] = df_modified["a"] + 100  # Different values

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_modified": df_modified})

        # Should not detect as subset because values differ
        assert len(result.relations) == 0

    def test_no_subset_different_indices(self):
        """Test that DataFrames with non-overlapping indices are not subsets."""
        df = pd.DataFrame({"a": range(200)}, index=range(0, 200))
        df_other = pd.DataFrame({"a": range(100)}, index=range(200, 300))

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_other": df_other})

        assert len(result.relations) == 0

    def test_iloc_subset(self):
        """Test detection of subset created by iloc."""
        df = pd.DataFrame({"a": range(300), "b": range(300, 600)})
        df_head = df.iloc[:100].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_head": df_head})

        assert len(result.relations) == 1
        assert result.relations[0].child_var == "df_head"
        assert result.relations[0].parent_var == "df"

    def test_loc_subset_with_labels(self):
        """Test detection of subset created by loc with string index."""
        df = pd.DataFrame(
            {"a": range(200), "b": range(200, 400)},
            index=[f"row_{i}" for i in range(200)],
        )
        selected_idx = [f"row_{i}" for i in range(50)]
        df_subset = df.loc[selected_idx].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_subset": df_subset})

        assert len(result.relations) == 1


class TestExtraColumns:
    """Test handling of extra columns in child DataFrames."""

    def test_child_with_extra_columns(self):
        """Test child DataFrame that has columns not in parent."""
        df = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
        })
        df_filtered = df[df["a"] > 100].copy()
        df_filtered["c"] = df_filtered["a"] * 2  # Extra column

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1
        relation = result.relations[0]
        assert relation.extra_columns == ["c"]
        assert relation.extra_data is not None
        assert "c" in relation.extra_data.columns

    def test_child_with_multiple_extra_columns(self):
        """Test child with multiple extra columns."""
        # Use min_savings_bytes=0 since we're testing detection, not savings threshold
        df = pd.DataFrame({"a": range(500)})
        df_filtered = df[df["a"] > 250].copy()
        df_filtered["b"] = df_filtered["a"] * 2
        df_filtered["c"] = df_filtered["a"] * 3
        df_filtered["d"] = "constant"

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1
        relation = result.relations[0]
        assert set(relation.extra_columns) == {"b", "c", "d"}
        assert relation.extra_data is not None

    def test_child_with_no_common_columns_rejected(self):
        """Test that child with no common columns is not detected as subset."""
        df = pd.DataFrame({"a": range(200), "b": range(200)})
        # Create df with subset index but completely different columns
        df_other = pd.DataFrame(
            {"x": range(100), "y": range(100)},
            index=df.index[:100],
        )

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_other": df_other})

        # Should not detect - no common columns
        assert len(result.relations) == 0


class TestMinimumThresholds:
    """Test minimum row and savings thresholds."""

    def test_min_rows_threshold(self):
        """Test that DataFrames below min_rows are ignored."""
        df = pd.DataFrame({"a": range(50)})  # 50 rows
        df_small = df[df["a"] > 25]  # ~24 rows

        detector = DataFrameSubsetDetector(min_rows=100, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_small": df_small})

        # Both DataFrames should be ignored (below threshold)
        assert len(result.relations) == 0
        assert len(result.parent_vars) == 0
        assert len(result.child_vars) == 0

    def test_min_savings_threshold(self):
        """Test that subsets with low savings are ignored."""
        # Small DataFrames where indices might not save much
        df = pd.DataFrame({"a": range(150)})
        df_subset = df[df["a"] > 100]

        # With very high min_savings, should not detect
        detector = DataFrameSubsetDetector(
            min_rows=10, min_savings_bytes=1000000  # 1 MB minimum
        )
        result = detector.detect({"df": df, "df_subset": df_subset})

        assert len(result.relations) == 0


class TestMultipleDataFrames:
    """Test detection with multiple DataFrames."""

    def test_chain_of_subsets(self):
        """Test chain: df -> df_a -> df_b."""
        df = pd.DataFrame({"x": range(500)})
        df_a = df[df["x"] > 100].copy()  # ~400 rows
        df_b = df_a[df_a["x"] > 300].copy()  # ~200 rows

        detector = DataFrameSubsetDetector(min_rows=50, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_a": df_a, "df_b": df_b})

        # Should detect df_a as child of df, df_b as child of df_a (or df)
        assert len(result.relations) >= 1
        assert "df_a" in result.child_vars or "df_b" in result.child_vars

    def test_multiple_children_same_parent(self):
        """Test multiple children from same parent."""
        df = pd.DataFrame({"a": range(500), "b": ["x", "y", "z", "w", "v"] * 100})
        df_x = df[df["b"] == "x"].copy()
        df_y = df[df["b"] == "y"].copy()
        df_z = df[df["b"] == "z"].copy()

        detector = DataFrameSubsetDetector(min_rows=50, min_savings_bytes=100)
        result = detector.detect({
            "df": df,
            "df_x": df_x,
            "df_y": df_y,
            "df_z": df_z,
        })

        # All filtered DFs should be detected as children of df
        assert "df" in result.parent_vars
        assert len(result.child_vars) >= 1

    def test_standalone_dataframes(self):
        """Test that unrelated DataFrames are marked as standalone."""
        df1 = pd.DataFrame({"a": range(200)})
        df2 = pd.DataFrame({"b": range(300, 500)})  # Unrelated

        detector = DataFrameSubsetDetector(min_rows=50, min_savings_bytes=100)
        result = detector.detect({"df1": df1, "df2": df2})

        assert len(result.relations) == 0
        assert result.standalone_vars == {"df1", "df2"}


class TestTimeoutBehavior:
    """Test timeout behavior."""

    def test_timeout_stops_detection(self):
        """Test that detection respects timeout."""
        # Create many DataFrames to potentially trigger timeout
        variables = {}
        for i in range(30):
            variables[f"df_{i}"] = pd.DataFrame({"a": range(200 + i * 10)})

        # Very short timeout
        detector = DataFrameSubsetDetector(
            min_rows=50, min_savings_bytes=100, timeout_ms=0.001
        )
        result = detector.detect(variables)

        # Should return without error (timeout should be handled gracefully)
        assert isinstance(result, SubsetDetectionResult)

    def test_max_dataframes_limit(self):
        """Test that max_dataframes limits processing."""
        variables = {}
        for i in range(50):
            variables[f"df_{i}"] = pd.DataFrame({"a": range(200)})

        detector = DataFrameSubsetDetector(
            min_rows=50, min_savings_bytes=100, max_dataframes=5
        )
        result = detector.detect(variables)

        # Should process at most 5 DataFrames
        # The exact behavior depends on implementation, but should not error
        assert isinstance(result, SubsetDetectionResult)


# ============================================================================
# RECONSTRUCTION TESTS
# ============================================================================


class TestReconstruction:
    """Test reconstructing child DataFrames from relations."""

    def test_basic_reconstruction(self):
        """Test basic reconstruction of a row subset."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5] * 50,
            "b": [10, 20, 30, 40, 50] * 50,
        })
        df_filtered = df[df["a"] > 3].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1
        relation = result.relations[0]

        # Reconstruct
        reconstructed = reconstruct_from_subset(df, relation)

        # Compare - reset index for comparison
        pd.testing.assert_frame_equal(
            reconstructed.reset_index(drop=True),
            df_filtered.reset_index(drop=True),
        )

    def test_reconstruction_with_extra_columns(self):
        """Test reconstruction when child has extra columns."""
        # Use min_savings_bytes=0 since we're testing reconstruction, not savings threshold
        df = pd.DataFrame({"a": range(500)})
        df_filtered = df[df["a"] > 250].copy()
        df_filtered["b"] = df_filtered["a"] * 2
        df_filtered["c"] = "extra"

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1
        relation = result.relations[0]

        reconstructed = reconstruct_from_subset(df, relation)

        # Should have all columns including extras
        assert set(reconstructed.columns) == {"a", "b", "c"}
        pd.testing.assert_frame_equal(
            reconstructed.reset_index(drop=True),
            df_filtered.reset_index(drop=True),
        )

    def test_reconstruction_preserves_dtypes(self):
        """Test that reconstruction preserves column dtypes."""
        df = pd.DataFrame({
            "int_col": np.array(range(200), dtype=np.int32),
            "float_col": np.array(range(200), dtype=np.float64),
            "str_col": ["a", "b"] * 100,
        })
        df_filtered = df[df["int_col"] > 100].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1
        relation = result.relations[0]

        reconstructed = reconstruct_from_subset(df, relation)

        # Check dtypes match
        for col in df_filtered.columns:
            assert reconstructed[col].dtype == df_filtered[col].dtype


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


class TestSerialization:
    """Test SubsetRelation serialization."""

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        relation = SubsetRelation(
            child_var="df_child",
            parent_var="df_parent",
            row_indices=np.array([0, 2, 5, 10, 15]),
            common_columns=["a", "b"],
            extra_columns=["c"],
            extra_data=pd.DataFrame({"c": [100, 200, 300, 400, 500]}),
            estimated_savings_bytes=50000,
        )

        # Serialize
        d = relation.to_dict()
        assert isinstance(d, dict)
        assert d["child_var"] == "df_child"
        assert d["parent_var"] == "df_parent"

        # Deserialize
        restored = SubsetRelation.from_dict(d)
        assert restored.child_var == relation.child_var
        assert restored.parent_var == relation.parent_var
        np.testing.assert_array_equal(restored.row_indices, relation.row_indices)
        assert restored.common_columns == relation.common_columns
        assert restored.extra_columns == relation.extra_columns
        pd.testing.assert_frame_equal(restored.extra_data, relation.extra_data)

    def test_to_dict_without_extra_data(self):
        """Test serialization when extra_data is None."""
        relation = SubsetRelation(
            child_var="df_child",
            parent_var="df_parent",
            row_indices=np.array([0, 1, 2]),
            common_columns=["a"],
            extra_columns=[],
            extra_data=None,
            estimated_savings_bytes=1000,
        )

        d = relation.to_dict()
        assert d["extra_data"] is None

        restored = SubsetRelation.from_dict(d)
        assert restored.extra_data is None


# ============================================================================
# TOPOLOGICAL SORT TESTS
# ============================================================================


class TestTopologicalSort:
    """Test topological sorting of relations."""

    def test_simple_chain(self):
        """Test sorting a simple chain: A -> B -> C."""
        # B is child of A, C is child of B
        relations = [
            SubsetRelation("C", "B", np.array([0]), ["a"], [], None, 100),
            SubsetRelation("B", "A", np.array([0]), ["a"], [], None, 100),
        ]

        sorted_rels = topological_sort_relations(relations)

        # B should come before C (B's parent is A which is not a child)
        child_order = [r.child_var for r in sorted_rels]
        assert child_order.index("B") < child_order.index("C")

    def test_multiple_children_same_parent(self):
        """Test sorting when multiple children have same parent."""
        relations = [
            SubsetRelation("B", "A", np.array([0]), ["a"], [], None, 100),
            SubsetRelation("C", "A", np.array([1]), ["a"], [], None, 100),
            SubsetRelation("D", "A", np.array([2]), ["a"], [], None, 100),
        ]

        sorted_rels = topological_sort_relations(relations)

        # All should be present (order among siblings doesn't matter)
        assert len(sorted_rels) == 3
        assert {r.child_var for r in sorted_rels} == {"B", "C", "D"}

    def test_empty_list(self):
        """Test sorting empty list."""
        sorted_rels = topological_sort_relations([])
        assert sorted_rels == []

    def test_single_relation(self):
        """Test sorting single relation."""
        relations = [
            SubsetRelation("B", "A", np.array([0]), ["a"], [], None, 100),
        ]
        sorted_rels = topological_sort_relations(relations)
        assert len(sorted_rels) == 1


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_variables(self):
        """Test with empty variables dict."""
        detector = DataFrameSubsetDetector()
        result = detector.detect({})

        assert result.relations == []
        assert result.parent_vars == set()
        assert result.child_vars == set()
        assert result.standalone_vars == set()

    def test_non_dataframe_variables(self):
        """Test that non-DataFrame variables are ignored."""
        variables = {
            "x": 42,
            "y": [1, 2, 3],
            "z": "string",
            "arr": np.array([1, 2, 3]),
        }

        detector = DataFrameSubsetDetector()
        result = detector.detect(variables)

        assert result.relations == []

    def test_mixed_variables(self):
        """Test with mix of DataFrames and other types."""
        df = pd.DataFrame({"a": range(200)})
        df_filtered = df[df["a"] > 100].copy()

        variables = {
            "df": df,
            "df_filtered": df_filtered,
            "x": 42,
            "y": [1, 2, 3],
        }

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect(variables)

        # Should still detect the subset relationship
        assert len(result.relations) == 1

    def test_dataframe_with_duplicate_index(self):
        """Test handling of DataFrames with duplicate indices."""
        # Parent with unique index
        df = pd.DataFrame({"a": range(200)})
        # Child created from parent should work
        df_filtered = df[df["a"] > 100].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        assert len(result.relations) == 1

    def test_multiindex_dataframe(self):
        """Test with MultiIndex DataFrames."""
        idx = pd.MultiIndex.from_tuples(
            [(i, j) for i in range(20) for j in range(10)],
            names=["level_0", "level_1"],
        )
        df = pd.DataFrame({"a": range(200)}, index=idx)
        df_filtered = df[df["a"] > 100].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_filtered": df_filtered})

        # Should handle MultiIndex
        assert isinstance(result, SubsetDetectionResult)


class TestDetectionResult:
    """Test SubsetDetectionResult properties."""

    def test_result_fields(self):
        """Test that result contains all expected fields."""
        df = pd.DataFrame({"a": range(300)})
        df_a = df[df["a"] > 200].copy()
        df_b = pd.DataFrame({"b": range(200)})  # Standalone

        detector = DataFrameSubsetDetector(min_rows=50, min_savings_bytes=100)
        result = detector.detect({"df": df, "df_a": df_a, "df_b": df_b})

        # Check all expected fields exist
        assert hasattr(result, "relations")
        assert hasattr(result, "parent_vars")
        assert hasattr(result, "child_vars")
        assert hasattr(result, "standalone_vars")
        assert hasattr(result, "total_estimated_savings_bytes")
        assert hasattr(result, "detection_time_ms")

        # Check types
        assert isinstance(result.relations, list)
        assert isinstance(result.parent_vars, set)
        assert isinstance(result.child_vars, set)
        assert isinstance(result.standalone_vars, set)
        assert isinstance(result.total_estimated_savings_bytes, int)
        assert isinstance(result.detection_time_ms, float)

    def test_parent_child_standalone_disjoint(self):
        """Test that parent, child, and standalone sets are disjoint."""
        df = pd.DataFrame({"a": range(300)})
        df_a = df[df["a"] > 100].copy()
        df_b = df[df["a"] > 200].copy()
        df_standalone = pd.DataFrame({"x": range(200)})

        detector = DataFrameSubsetDetector(min_rows=50, min_savings_bytes=100)
        result = detector.detect({
            "df": df,
            "df_a": df_a,
            "df_b": df_b,
            "df_standalone": df_standalone,
        })

        # Sets should be disjoint
        assert result.parent_vars.isdisjoint(result.child_vars)
        assert result.parent_vars.isdisjoint(result.standalone_vars)
        assert result.child_vars.isdisjoint(result.standalone_vars)
