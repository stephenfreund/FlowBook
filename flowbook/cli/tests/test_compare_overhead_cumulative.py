"""
Tests for compare_overhead extract functions using cumulative checkpoint data.

These tests verify that:
1. extract_checkpoint_type_data_v2 uses cumulative_by_type when available
2. extract_checkpoint_var_data uses cumulative_by_var when available
3. Both functions fall back to checkpoint_var_costs for backwards compatibility
4. The extracted data is consistent with overhead_breakdown.checkpoints_mb
"""

import pytest
from typing import Dict, Any

from flowbook.cli.compare_overhead import (
    extract_checkpoint_type_data_v2,
    extract_checkpoint_var_data,
    is_v2_format,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_v2_comparison_data(cells: list) -> Dict[str, Any]:
    """Create a v2.0 format comparison data structure."""
    return {
        "version": "2.0",  # is_v2_format() checks for "version" or "_version"
        "notebook_path": "/test/notebook.ipynb",
        "kernels": {
            "flowbook": {
                "memory": {
                    "cells": cells
                }
            }
        }
    }


def create_cell_with_old_format(
    cell_id: str,
    cell_index: int,
    checkpoint_var_costs: Dict[str, Dict],
    checkpoints_mb: float = 0.0
) -> Dict[str, Any]:
    """Create a cell dict using the OLD format (checkpoint_var_costs only)."""
    return {
        "cell_id": cell_id,
        "cell_index": cell_index,
        "current_footprint_mb": 10.0,
        "max_footprint_mb": 10.0,
        "allocation_delta_mb": 1.0,
        "gpu_mem_samples": 0.0,
        "checkpoint_var_costs": checkpoint_var_costs,
        "overhead_breakdown": {
            "checkpoints_mb": checkpoints_mb,
            "execution_records_mb": 0.0,
            "tracking_metadata_mb": 0.0,
            "other_mb": 0.0,
        },
        "status": "ok"
    }


def create_cell_with_new_format(
    cell_id: str,
    cell_index: int,
    checkpoint_var_costs: Dict[str, Dict],
    cumulative_by_type: Dict[str, int],
    cumulative_by_var: Dict[str, int],
    checkpoints_mb: float = 0.0
) -> Dict[str, Any]:
    """Create a cell dict using the NEW format (with cumulative fields)."""
    return {
        "cell_id": cell_id,
        "cell_index": cell_index,
        "current_footprint_mb": 10.0,
        "max_footprint_mb": 10.0,
        "allocation_delta_mb": 1.0,
        "gpu_mem_samples": 0.0,
        "checkpoint_var_costs": checkpoint_var_costs,
        "cumulative_by_type": cumulative_by_type,
        "cumulative_by_var": cumulative_by_var,
        "overhead_breakdown": {
            "checkpoints_mb": checkpoints_mb,
            "execution_records_mb": 0.0,
            "tracking_metadata_mb": 0.0,
            "other_mb": 0.0,
        },
        "status": "ok"
    }


# =============================================================================
# EXTRACT_CHECKPOINT_TYPE_DATA_V2 TESTS
# =============================================================================

class TestExtractCheckpointTypeDataWithNewFormat:
    """Tests for extract_checkpoint_type_data_v2 using new cumulative_by_type."""

    def test_uses_cumulative_by_type_when_available(self):
        """Test that cumulative_by_type is used when present."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={"arr": {"bytes": 1000, "type": "ndarray"}},
                cumulative_by_type={"ndarray": 500},  # Different from var_costs!
                cumulative_by_var={"arr": 500},
                checkpoints_mb=0.5
            ),
            create_cell_with_new_format(
                "cell2", 1,
                checkpoint_var_costs={"arr": {"bytes": 1000, "type": "ndarray"}},
                cumulative_by_type={"ndarray": 800},  # Cumulative
                cumulative_by_var={"arr": 800},
                checkpoints_mb=0.8
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data)

        assert result is not None
        # Should use cumulative_by_type values (500, 800) not derive from var_costs
        assert result["by_type"]["ndarray"] == [500, 800]
        assert result["total_bytes"] == [500, 800]

    def test_cumulative_values_are_monotonic(self):
        """Test that cumulative values are properly tracked (should be monotonic)."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={"DataFrame": 1000000, "ndarray": 500000},
                cumulative_by_var={},
                checkpoints_mb=1.5
            ),
            create_cell_with_new_format(
                "cell2", 1,
                checkpoint_var_costs={},
                cumulative_by_type={"DataFrame": 2000000, "ndarray": 1000000},
                cumulative_by_var={},
                checkpoints_mb=3.0
            ),
            create_cell_with_new_format(
                "cell3", 2,
                checkpoint_var_costs={},
                cumulative_by_type={"DataFrame": 3000000, "ndarray": 1500000},
                cumulative_by_var={},
                checkpoints_mb=4.5
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data)

        assert result is not None
        df_values = result["by_type"]["DataFrame"]
        arr_values = result["by_type"]["ndarray"]

        # Each should be monotonically increasing
        for i in range(1, len(df_values)):
            assert df_values[i] >= df_values[i-1], "DataFrame values not monotonic"
            assert arr_values[i] >= arr_values[i-1], "ndarray values not monotonic"

    def test_types_ordered_by_final_size(self):
        """Test that types are ordered by final cumulative size."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={
                    "DataFrame": 5000000,  # Largest
                    "list": 100000,         # Smallest
                    "ndarray": 2000000,     # Middle
                },
                cumulative_by_var={},
                checkpoints_mb=7.1
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data)

        assert result is not None
        # Should be ordered: DataFrame, ndarray, list
        assert result["types_ordered"] == ["DataFrame", "ndarray", "list"]

    def test_top_n_limits_types(self):
        """Test that top_n limits the number of types shown."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={
                    "type1": 1000, "type2": 2000, "type3": 3000,
                    "type4": 4000, "type5": 5000,
                },
                cumulative_by_var={},
                checkpoints_mb=0.015
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data, top_n=3)

        assert result is not None
        # Should have top 3 + "other"
        assert len(result["types_ordered"]) == 4
        assert "other" in result["types_ordered"]
        # Top 3 by size should be type5, type4, type3
        assert result["types_ordered"][:3] == ["type5", "type4", "type3"]


class TestExtractCheckpointTypeDataFallback:
    """Tests for extract_checkpoint_type_data_v2 falling back to old format."""

    def test_falls_back_to_var_costs(self):
        """Test fallback to checkpoint_var_costs when cumulative_by_type missing."""
        cells = [
            create_cell_with_old_format(
                "cell1", 0,
                checkpoint_var_costs={
                    "arr": {"bytes": 1000, "type": "ndarray", "module": "numpy"}
                },
                checkpoints_mb=0.001
            ),
            create_cell_with_old_format(
                "cell2", 1,
                checkpoint_var_costs={
                    "arr": {"bytes": 1000, "type": "ndarray", "module": "numpy"}
                },
                checkpoints_mb=0.002
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data)

        assert result is not None
        # Should derive cumulative from var_costs: 1000, 1000+1000=2000
        assert result["by_type"]["ndarray"] == [1000, 2000]

    def test_returns_none_for_no_data(self):
        """Test returns None when no checkpoint data available."""
        cells = [
            create_cell_with_old_format("cell1", 0, checkpoint_var_costs=None)
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data)

        assert result is None


# =============================================================================
# EXTRACT_CHECKPOINT_VAR_DATA TESTS
# =============================================================================

class TestExtractCheckpointVarDataWithNewFormat:
    """Tests for extract_checkpoint_var_data using new cumulative_by_var."""

    def test_uses_cumulative_by_var_when_available(self):
        """Test that cumulative_by_var is used when present."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={"df": {"bytes": 1000, "type": "DataFrame"}},
                cumulative_by_type={"DataFrame": 500},
                cumulative_by_var={"df": 500},  # Different from var_costs!
                checkpoints_mb=0.5
            ),
            create_cell_with_new_format(
                "cell2", 1,
                checkpoint_var_costs={"df": {"bytes": 1000, "type": "DataFrame"}},
                cumulative_by_type={"DataFrame": 800},
                cumulative_by_var={"df": 800},  # Cumulative
                checkpoints_mb=0.8
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_var_data(data)

        assert result is not None
        # Should use cumulative_by_var values (500, 800)
        assert result["by_var"]["df"] == [500, 800]

    def test_variables_ordered_by_final_size(self):
        """Test that variables are ordered by final cumulative size."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={},
                cumulative_by_var={
                    "large_df": 5000000,
                    "tiny_list": 100,
                    "medium_arr": 1000000,
                },
                checkpoints_mb=6.0
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_var_data(data)

        assert result is not None
        # Should be ordered: large_df, medium_arr, tiny_list
        assert result["vars_ordered"] == ["large_df", "medium_arr", "tiny_list"]

    def test_top_n_limits_variables(self):
        """Test that top_n limits the number of variables shown."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={},
                cumulative_by_var={
                    "var1": 100, "var2": 200, "var3": 300,
                    "var4": 400, "var5": 500, "var6": 600,
                },
                checkpoints_mb=2.1
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_var_data(data, top_n=3)

        assert result is not None
        # Should have top 3 + "other"
        assert len(result["vars_ordered"]) == 4
        assert "other" in result["vars_ordered"]
        # Top 3 by size: var6, var5, var4
        assert result["vars_ordered"][:3] == ["var6", "var5", "var4"]

    def test_other_aggregates_remaining(self):
        """Test that 'other' correctly aggregates remaining variables."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={},
                cumulative_by_var={
                    "big": 1000,
                    "small1": 100,
                    "small2": 200,
                },
                checkpoints_mb=1.3
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_var_data(data, top_n=1)

        assert result is not None
        # Should have "big" and "other" (small1 + small2)
        assert "big" in result["by_var"]
        assert "other" in result["by_var"]
        assert result["by_var"]["other"][0] == 300  # 100 + 200


class TestExtractCheckpointVarDataFallback:
    """Tests for extract_checkpoint_var_data falling back to old format."""

    def test_falls_back_to_var_costs(self):
        """Test fallback to checkpoint_var_costs when cumulative_by_var missing."""
        cells = [
            create_cell_with_old_format(
                "cell1", 0,
                checkpoint_var_costs={
                    "arr": {"bytes": 1000, "type": "ndarray"}
                },
                checkpoints_mb=0.001
            ),
            create_cell_with_old_format(
                "cell2", 1,
                checkpoint_var_costs={
                    "arr": {"bytes": 500, "type": "ndarray"}
                },
                checkpoints_mb=0.0015
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_var_data(data)

        assert result is not None
        # Should derive cumulative: 1000, 1000+500=1500
        assert result["by_var"]["arr"] == [1000, 1500]

    def test_returns_none_for_no_data(self):
        """Test returns None when no checkpoint data available."""
        cells = [
            create_cell_with_old_format("cell1", 0, checkpoint_var_costs=None)
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_var_data(data)

        assert result is None


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestConsistencyBetweenExtracts:
    """Test consistency between type and variable extracts."""

    def test_total_bytes_match(self):
        """Test that total bytes from type extract matches overhead_breakdown."""
        cumulative_by_type = {
            "DataFrame": 1000000,
            "ndarray": 500000,
            "list": 200000,
        }
        total_bytes = sum(cumulative_by_type.values())  # 1.7MB

        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type=cumulative_by_type,
                cumulative_by_var={},
                checkpoints_mb=total_bytes / (1024*1024)
            ),
        ]
        data = create_v2_comparison_data(cells)

        result = extract_checkpoint_type_data_v2(data)

        assert result is not None
        # Sum of types should equal total
        assert result["total_bytes"][0] == total_bytes

    def test_type_and_var_totals_match(self):
        """Test that type total and variable total are consistent."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={"DataFrame": 500, "list": 300},
                cumulative_by_var={"df": 500, "lst": 300},
                checkpoints_mb=0.0008
            ),
        ]
        data = create_v2_comparison_data(cells)

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        assert type_result is not None
        assert var_result is not None

        # Totals should be equal
        type_total = type_result["total_bytes"][0]
        var_total = sum(var_result["by_var"][v][0] for v in var_result["by_var"])
        assert type_total == var_total


class TestBackwardsCompatibility:
    """Test backwards compatibility with old comparison files."""

    def test_old_format_still_works(self):
        """Test that old format files (no cumulative fields) still work."""
        # Old format only has checkpoint_var_costs
        cells = [
            {
                "cell_id": "abc1",
                "cell_index": 0,
                "current_footprint_mb": 10.0,
                "max_footprint_mb": 10.0,
                "checkpoint_var_costs": {
                    "df": {"bytes": 1000000, "type": "DataFrame", "module": "pandas"}
                },
                "overhead_breakdown": {"checkpoints_mb": 1.0},
                "status": "ok"
            },
            {
                "cell_id": "def2",
                "cell_index": 1,
                "current_footprint_mb": 15.0,
                "max_footprint_mb": 15.0,
                "checkpoint_var_costs": {
                    "df": {"bytes": 500000, "type": "DataFrame", "module": "pandas"},
                    "arr": {"bytes": 800000, "type": "ndarray", "module": "numpy"}
                },
                "overhead_breakdown": {"checkpoints_mb": 2.3},
                "status": "ok"
            },
        ]
        data = create_v2_comparison_data(cells)

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        # Should work with old format
        assert type_result is not None
        assert var_result is not None

        # Should derive cumulative from per-cell costs
        assert "DataFrame" in type_result["by_type"]
        assert "df" in var_result["by_var"]


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_cells(self):
        """Test with empty cells list."""
        data = create_v2_comparison_data([])

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        assert type_result is None
        assert var_result is None

    def test_all_zeros(self):
        """Test with all zero values."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={"type1": 0},
                cumulative_by_var={"var1": 0},
                checkpoints_mb=0.0
            ),
        ]
        data = create_v2_comparison_data(cells)

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        assert type_result is not None
        assert var_result is not None

    def test_single_cell(self):
        """Test with single cell."""
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={},
                cumulative_by_type={"DataFrame": 1000000},
                cumulative_by_var={"df": 1000000},
                checkpoints_mb=1.0
            ),
        ]
        data = create_v2_comparison_data(cells)

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        assert type_result is not None
        assert var_result is not None
        assert len(type_result["cells"]) == 1
        assert len(var_result["cells"]) == 1

    def test_mixed_new_and_old_cells(self):
        """Test with mix of new and old format cells."""
        # First cell has new format, second doesn't
        cells = [
            create_cell_with_new_format(
                "cell1", 0,
                checkpoint_var_costs={"x": {"bytes": 1000, "type": "int"}},
                cumulative_by_type={"int": 1000},
                cumulative_by_var={"x": 1000},
                checkpoints_mb=0.001
            ),
            create_cell_with_old_format(
                "cell2", 1,
                checkpoint_var_costs={"x": {"bytes": 500, "type": "int"}},
                checkpoints_mb=0.0015
            ),
        ]
        data = create_v2_comparison_data(cells)

        # Should still use new format path since first cell has cumulative
        type_result = extract_checkpoint_type_data_v2(data)

        assert type_result is not None


class TestNonV2Format:
    """Test handling of non-v2 format data."""

    def test_returns_none_for_v1_format(self):
        """Test that v1 format returns None."""
        data = {
            "format_version": "1.0",
            "cells": []
        }

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        assert type_result is None
        assert var_result is None

    def test_returns_none_for_missing_flowbook_data(self):
        """Test returns None when flowbook data is missing."""
        data = {
            "format_version": "2.0",
            "kernels": {
                "baseline": {"timing": {"cells": []}}
            }
        }

        type_result = extract_checkpoint_type_data_v2(data)
        var_result = extract_checkpoint_var_data(data)

        assert type_result is None
        assert var_result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
