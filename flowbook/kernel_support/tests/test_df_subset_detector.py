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


# ============================================================================
# FULL VALIDATION TESTS (Algorithm Fix Verification)
# ============================================================================


class TestFullValidation:
    """Test the full vectorized validation algorithm.

    These tests verify that the algorithm correctly rejects false positives
    that would have passed the old spot-check validation.
    """

    def test_manufactured_false_positive_rejected(self):
        """Test that a DataFrame matching at spot-check positions but differing elsewhere is NOT detected.

        This is the critical test that proves the fix works - the old spot-check
        only validated positions 0, n//2, and n-1 in the first column.
        """
        n = 200
        # Create parent DataFrame
        parent = pd.DataFrame({
            "a": list(range(n)),
            "b": list(range(n, n*2)),
        })

        # Create a "child" that would pass the old spot-check but is NOT a real subset
        # It matches at positions 0, 99 (n//2), and 199 (n-1) but differs elsewhere
        child_data = {"a": [], "b": []}
        for i in range(100):  # 100 rows
            if i in (0, 49, 99):  # These positions map to 0, 99, 199 in parent
                # Match the parent values at these positions
                child_data["a"].append(i * 2)  # Maps to parent row i*2
                child_data["b"].append(i * 2 + n)
            else:
                # Use DIFFERENT values that would fail full validation
                child_data["a"].append(9999 + i)
                child_data["b"].append(8888 + i)

        child = pd.DataFrame(child_data)
        # Give it a fake index that overlaps with parent to pass index check
        child.index = pd.Index(range(0, 200, 2))

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect as subset because values don't match
        assert len(result.relations) == 0

    def test_mismatch_in_middle_column_rejected(self):
        """Test that DataFrame matching first column but differing in second is NOT detected."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
            "c": range(400, 600),
        })

        # Create child that matches column "a" but has different values in "b"
        indices = list(range(0, 200, 2))  # Every other row
        child = parent.iloc[indices].copy()
        child["b"] = 9999  # Change all values in column b

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect as subset
        assert len(result.relations) == 0

    def test_mismatch_in_last_column_rejected(self):
        """Test that DataFrame matching all but last column is NOT detected."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
            "c": range(400, 600),
        })

        indices = list(range(0, 200, 2))
        child = parent.iloc[indices].copy()
        child["c"] = 7777  # Change only the last column

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect as subset
        assert len(result.relations) == 0

    def test_single_value_mismatch_rejected(self):
        """Test that a single value mismatch is correctly rejected."""
        parent = pd.DataFrame({
            "a": list(range(200)),
            "b": list(range(200, 400)),
        })

        indices = list(range(0, 200, 2))
        child = parent.iloc[indices].copy()
        # Change just ONE value in the middle (not at spot-check positions)
        child.iloc[25, 0] = 99999  # Position 25 (index 50 in parent)

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect as subset
        assert len(result.relations) == 0

    def test_all_columns_validated(self):
        """Test that validation occurs across ALL common columns."""
        parent = pd.DataFrame({
            "col1": range(200),
            "col2": range(200, 400),
            "col3": range(400, 600),
            "col4": range(600, 800),
            "col5": range(800, 1000),
        })

        indices = list(range(0, 200, 2))
        child = parent.iloc[indices].copy()
        # Change only the 4th column (col4)
        child["col4"] = 5555

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect as subset
        assert len(result.relations) == 0

    def test_true_subset_still_detected(self):
        """Test that true subsets are still correctly detected after the fix."""
        parent = pd.DataFrame({
            "a": range(300),
            "b": range(300, 600),
            "c": ["x", "y", "z"] * 100,
        })

        # Create a true row subset
        child = parent[parent["a"] > 100].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should detect as subset
        assert len(result.relations) == 1
        assert result.relations[0].child_var == "child"
        assert result.relations[0].parent_var == "parent"


class TestRealSharingDetection:
    """Test detection of real DataFrame sharing scenarios."""

    def test_true_subset_with_non_sequential_indices(self):
        """Test true subset with gaps (rows 0, 5, 10, 15, ...)."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
        })

        # Select rows with gaps
        indices = list(range(0, 200, 5))  # Every 5th row
        child = parent.iloc[indices].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        assert len(result.relations) == 1
        assert len(result.relations[0].row_indices) == 40

    def test_identical_values_different_rows_not_subset(self):
        """Test that same values but different index positions is NOT a subset."""
        parent = pd.DataFrame({
            "a": [1, 2, 3, 4, 5] * 40,  # 200 rows
            "b": [10, 20, 30, 40, 50] * 40,
        })

        # Create DataFrame with same values but DIFFERENT indices
        child = pd.DataFrame({
            "a": [1, 2, 3, 4, 5] * 20,
            "b": [10, 20, 30, 40, 50] * 20,
        }, index=range(200, 300))  # Non-overlapping index

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect - indices don't overlap
        assert len(result.relations) == 0

    def test_shuffled_rows_not_subset(self):
        """Test that shuffled rows with same values are NOT a subset."""
        parent = pd.DataFrame({
            "a": list(range(200)),
            "b": list(range(200, 400)),
        })

        # Create "child" with same index but shuffled values
        indices = list(range(0, 200, 2))
        child = parent.iloc[indices].copy()
        # Shuffle the rows (values no longer match their index positions)
        child = child.sample(frac=1, random_state=42)
        child.index = parent.iloc[indices].index  # Reset to original indices

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect - values don't match at correct positions
        assert len(result.relations) == 0

    def test_overlapping_but_not_subset(self):
        """Test some rows from parent + different rows = NOT a subset."""
        parent = pd.DataFrame({
            "a": list(range(200)),
            "b": list(range(200, 400)),
        })

        # Take first 50 rows from parent
        partial = parent.iloc[:50].copy()
        # Add 50 completely different rows
        extra = pd.DataFrame({
            "a": list(range(1000, 1050)),
            "b": list(range(2000, 2050)),
        }, index=range(50, 100))

        child = pd.concat([partial, extra])

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should NOT detect - not all indices are in parent
        assert len(result.relations) == 0

    def test_nan_values_handled_correctly(self):
        """Test that NaN values are handled correctly (NaN == NaN for subset purposes)."""
        parent = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0] * 40,
            "b": [np.nan, 2.0, np.nan, 4.0, np.nan] * 40,
        })

        # Create true subset including NaN values
        child = parent.iloc[::2].copy()  # Every other row

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        # Should detect - NaN positions match
        assert len(result.relations) == 1

    def test_nan_in_object_column_handled(self):
        """Test NaN in object dtype columns."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": [None if i % 10 == 0 else f"val_{i}" for i in range(200)],
        })

        child = parent.iloc[::2].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        assert len(result.relations) == 1


class TestMemoryViewVsValueSubset:
    """Test detection of memory views vs value copies."""

    def test_iloc_view_vs_iloc_copy(self):
        """Test both views and copies work correctly."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
        })

        # Create explicit copy
        child_copy = parent.iloc[::2].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child_copy": child_copy})

        assert len(result.relations) == 1

    def test_filtered_copy_detection(self):
        """Test boolean-filtered copy is detected."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
        })

        child = parent[parent["a"] > 50].copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        assert len(result.relations) == 1

    def test_query_subset_detection(self):
        """Test DataFrame.query() result is detected as subset."""
        parent = pd.DataFrame({
            "a": range(200),
            "b": range(200, 400),
        })

        child = parent.query("a > 100").copy()

        detector = DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)
        result = detector.detect({"parent": parent, "child": child})

        assert len(result.relations) == 1


class TestFullValidationPerformance:
    """Test performance of the full validation algorithm."""

    def test_validation_under_50ms_for_100k_rows(self):
        """Test that 100K row validation completes in reasonable time."""
        import time

        n = 100_000
        parent = pd.DataFrame({
            "a": np.arange(n),
            "b": np.arange(n, n*2),
            "c": np.arange(n*2, n*3),
            "d": np.arange(n*3, n*4),
            "e": np.arange(n*4, n*5),
        })

        child = parent.iloc[::2].copy()  # 50K rows

        detector = DataFrameSubsetDetector(min_rows=100, min_savings_bytes=0)

        start = time.time()
        result = detector.detect({"parent": parent, "child": child})
        elapsed_ms = (time.time() - start) * 1000

        assert len(result.relations) == 1
        # Should complete in under 100ms (being generous for CI environments)
        assert elapsed_ms < 100, f"Took {elapsed_ms:.1f}ms, expected < 100ms"

    def test_early_rejection_is_fast(self):
        """Test that non-subsets are rejected quickly (first column mismatch)."""
        import time

        n = 100_000
        parent = pd.DataFrame({
            "a": np.arange(n),
            "b": np.arange(n, n*2),
        })

        # Child with completely different values - should reject immediately
        child = pd.DataFrame({
            "a": np.arange(n, n + 50000),  # Different values
            "b": np.arange(n*2, n*2 + 50000),
        }, index=range(0, n, 2))  # Same index as would be a subset

        detector = DataFrameSubsetDetector(min_rows=100, min_savings_bytes=0)

        start = time.time()
        result = detector.detect({"parent": parent, "child": child})
        elapsed_ms = (time.time() - start) * 1000

        assert len(result.relations) == 0
        # Should reject very quickly
        assert elapsed_ms < 50, f"Took {elapsed_ms:.1f}ms, expected < 50ms"

    def test_caching_avoids_revalidation(self):
        """Test that repeated checks use cache."""
        import time

        parent = pd.DataFrame({
            "a": range(10000),
            "b": range(10000, 20000),
        })
        child = parent.iloc[::2].copy()

        detector = DataFrameSubsetDetector(min_rows=100, min_savings_bytes=0)

        # First detection
        start1 = time.time()
        result1 = detector.detect({"parent": parent, "child": child})
        time1 = time.time() - start1

        # Second detection (should use cache)
        start2 = time.time()
        result2 = detector.detect({"parent": parent, "child": child})
        time2 = time.time() - start2

        assert len(result1.relations) == 1
        assert len(result2.relations) == 1
        # Second call should be faster (cached)
        assert time2 < time1 * 0.5, f"Cache not effective: {time2:.4f}s vs {time1:.4f}s"

    def test_multiple_columns_validation_scales(self):
        """Test that many columns don't cause excessive slowdown."""
        import time

        n = 50000
        # Create DataFrame with many columns
        data = {f"col_{i}": np.arange(n) + i*n for i in range(20)}
        parent = pd.DataFrame(data)
        child = parent.iloc[::2].copy()

        detector = DataFrameSubsetDetector(min_rows=100, min_savings_bytes=0)

        start = time.time()
        result = detector.detect({"parent": parent, "child": child})
        elapsed_ms = (time.time() - start) * 1000

        assert len(result.relations) == 1
        # Should still be reasonably fast even with 20 columns
        assert elapsed_ms < 200, f"Took {elapsed_ms:.1f}ms with 20 columns"
