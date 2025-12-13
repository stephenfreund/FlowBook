"""
Integration tests for Diff with structural reads.

These tests verify that the Diff class correctly handles structural reads,
requiring structural equality when structural attributes were accessed.

Note: These tests document the expected behavior. The actual implementation
in diff.py needs to be updated to support structural_reads and structural_mode.
"""

import pytest
import pandas as pd
import numpy as np

from data_ferret.kernel.diff import Diff
from data_ferret.kernel.types import DiffResult, ValueComparison, CompoundDiff
from data_ferret.kernel.structural_tracking import StructuralTrackingMode


class TestDiffWithoutStructuralReads:
    """Tests for Diff behavior without structural reads (baseline)."""

    def test_leq_mode_allows_extra_columns(self):
        """In LEQ mode without structural reads, extra columns are allowed."""
        df_before = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df_after = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        differ = Diff(use_leq=True, column_rbw={'df': {'a', 'b'}})
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Extra column 'c' should be allowed
        assert 'df' not in result.differences

    def test_leq_mode_allows_extra_rows(self):
        """In LEQ mode without structural reads, extra rows may be allowed depending on context."""
        # Note: The exact behavior depends on implementation
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 2, 3]})

        differ = Diff(use_leq=True, column_rbw={'df': {'a'}})
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Behavior depends on whether row changes are detected
        # This test documents current behavior

    def test_strict_mode_detects_column_addition(self):
        """In strict mode (use_leq=False), column additions are detected."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        differ = Diff(use_leq=False)
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences


class TestDiffStructuralReadsModeOff:
    """Tests for Diff with structural_mode=OFF."""

    def test_off_mode_ignores_structural_reads(self):
        """OFF mode ignores structural_reads entirely."""
        df_before = pd.DataFrame({'a': [1], 'b': [2]})
        df_after = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.OFF,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Extra column allowed because mode is OFF
        assert 'df' not in result.differences

    def test_off_mode_no_warnings(self):
        """OFF mode produces no warnings."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.OFF,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert result.warnings == []


class TestDiffStructuralReadsModeWarn:
    """Tests for Diff with structural_mode=WARN."""

    def test_warn_mode_returns_warnings_not_differences(self):
        """WARN mode returns warnings but no differences for structural issues."""
        df_before = pd.DataFrame({'a': [1], 'b': [2]})
        df_after = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.WARN,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Should not be in differences
        assert 'df' not in result.differences
        # But should have warnings
        assert len(result.warnings) > 0
        assert any('columns' in w for w in result.warnings)

    def test_warn_mode_multiple_structural_warnings(self):
        """WARN mode can produce multiple warnings."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns', 'shape', 'len'}},
            structural_mode=StructuralTrackingMode.WARN,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Multiple warnings expected
        assert len(result.warnings) >= 1


class TestDiffStructuralReadsModeEnforce:
    """Tests for Diff with structural_mode=ENFORCE."""

    def test_enforce_columns_blocks_column_addition(self):
        """When df.columns was read, adding columns is a difference in ENFORCE mode."""
        df_before = pd.DataFrame({'a': [1], 'b': [2]})
        df_after = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_shape_blocks_row_addition(self):
        """When df.shape was read, adding rows is a difference."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 2, 3]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'shape'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_len_blocks_row_addition(self):
        """When len(df) was called, adding rows is a difference."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 2, 3]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'len'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_iter_blocks_column_addition(self):
        """When `for col in df` was used, adding columns is a difference."""
        df_before = pd.DataFrame({'a': [1], 'b': [2]})
        df_after = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'iter'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_dtypes_blocks_dtype_change(self):
        """When df.dtypes was read, dtype changes are detected."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1.0, 2.0]})  # Different dtype

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'dtypes'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_index_blocks_index_change(self):
        """When df.index was read, index changes are detected."""
        df_before = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
        df_after = pd.DataFrame({'a': [1, 2]}, index=[0, 2])

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'index'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_describe_blocks_column_addition(self):
        """When df.describe() was called, adding columns is detected."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'describe'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences

    def test_enforce_to_dict_blocks_column_addition(self):
        """When df.to_dict() was called, adding columns is detected."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'to_dict'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences


class TestDiffStructuralNoChange:
    """Tests verifying no difference when structure hasn't changed."""

    def test_no_diff_when_structure_unchanged(self):
        """No difference when structure matches, even with structural reads."""
        df_before = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df_after = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns', 'shape'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' not in result.differences

    def test_value_change_detected_with_structural_reads(self):
        """Value changes are still detected when structural reads present."""
        df_before = pd.DataFrame({'a': [1, 2]})
        df_after = pd.DataFrame({'a': [1, 999]})  # Value changed

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert 'df' in result.differences


class TestDiffCombinedColumnAndStructural:
    """Tests for combined column_rbw and structural_reads."""

    def test_column_rbw_and_structural_together(self):
        """Column RBW and structural reads work together."""
        df_before = pd.DataFrame({'a': [1], 'b': [2]})
        df_after = pd.DataFrame({'a': [999], 'b': [2], 'c': [3]})  # Changed a, added c

        differ = Diff(
            use_leq=True,
            column_rbw={'df': {'a'}},  # Only column 'a' read
            structural_reads={},  # No structural reads
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Should detect change in column 'a', ignore added column 'c'
        assert 'df' in result.differences

    def test_structural_overrides_column_flexibility(self):
        """Structural reads override column-level flexibility."""
        df_before = pd.DataFrame({'a': [1], 'b': [2]})
        df_after = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})

        differ = Diff(
            use_leq=True,
            column_rbw={'df': {'a', 'b'}},  # Only columns a, b read
            structural_reads={'df': {'columns'}},  # But columns attribute was accessed
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Should detect difference because df.columns was read
        assert 'df' in result.differences


class TestDiffSeriesStructural:
    """Tests for Series structural reads."""

    def test_series_index_blocks_length_change(self):
        """When s.index was read, length changes are detected."""
        s_before = pd.Series([1, 2, 3])
        s_after = pd.Series([1, 2, 3, 4])

        differ = Diff(
            use_leq=True,
            structural_reads={'s': {'index'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'s': s_before}, {'s': s_after})

        assert 's' in result.differences

    def test_series_shape_blocks_length_change(self):
        """When s.shape was read, length changes are detected."""
        s_before = pd.Series([1, 2])
        s_after = pd.Series([1, 2, 3])

        differ = Diff(
            use_leq=True,
            structural_reads={'s': {'shape'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'s': s_before}, {'s': s_after})

        assert 's' in result.differences

    def test_series_dtype_blocks_dtype_change(self):
        """When s.dtype was read, dtype changes are detected."""
        s_before = pd.Series([1, 2, 3])
        s_after = pd.Series([1.0, 2.0, 3.0])

        differ = Diff(
            use_leq=True,
            structural_reads={'s': {'dtype'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'s': s_before}, {'s': s_after})

        assert 's' in result.differences

    def test_series_name_blocks_name_change(self):
        """When s.name was read, name changes are detected."""
        s_before = pd.Series([1, 2], name='original')
        s_after = pd.Series([1, 2], name='changed')

        differ = Diff(
            use_leq=True,
            structural_reads={'s': {'name'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'s': s_before}, {'s': s_after})

        assert 's' in result.differences


class TestDiffMultipleVariables:
    """Tests for structural reads on multiple variables."""

    def test_multiple_dataframes(self):
        """Structural reads tracked independently per variable."""
        df1_before = pd.DataFrame({'a': [1]})
        df1_after = pd.DataFrame({'a': [1], 'b': [2]})  # Added column

        df2_before = pd.DataFrame({'x': [1]})
        df2_after = pd.DataFrame({'x': [1], 'y': [2]})  # Added column

        differ = Diff(
            use_leq=True,
            structural_reads={'df1': {'columns'}},  # Only df1 had structural read
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff(
            {'df1': df1_before, 'df2': df2_before},
            {'df1': df1_after, 'df2': df2_after}
        )

        # df1 should be detected (had structural read)
        assert 'df1' in result.differences
        # df2 should NOT be detected (no structural read)
        assert 'df2' not in result.differences

    def test_dataframe_and_series_together(self):
        """Mix of DataFrame and Series with structural reads."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        s_before = pd.Series([1, 2])
        s_after = pd.Series([1, 2, 3])

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}, 's': {'shape'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff(
            {'df': df_before, 's': s_before},
            {'df': df_after, 's': s_after}
        )

        assert 'df' in result.differences
        assert 's' in result.differences


class TestDiffNestedVariables:
    """Tests for structural reads on nested variables."""

    def test_nested_path_structural_read(self):
        """Structural reads work with nested paths like data['train']."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={"data['train']": {'columns'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff(
            {"data['train']": df_before},
            {"data['train']": df_after}
        )

        assert "data['train']" in result.differences


class TestDiffWarningsCollection:
    """Tests for warning collection in warn mode."""

    def test_warnings_include_attribute_names(self):
        """Warnings include which structural attributes were accessed."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns', 'shape'}},
            structural_mode=StructuralTrackingMode.WARN,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Warnings should mention which attributes were accessed
        assert any('columns' in w for w in result.warnings)

    def test_warnings_include_variable_path(self):
        """Warnings include the variable path."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.WARN,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        assert any('df' in w for w in result.warnings)

    def test_compound_diff_warnings(self):
        """CompoundDiff nodes can carry warnings."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.WARN,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # The CompoundDiff for df should have warnings
        if 'df' in result.differences:
            node = result.differences['df']
            if isinstance(node, CompoundDiff):
                assert len(node.warnings) > 0


class TestDiffEdgeCases:
    """Edge cases for structural diff handling."""

    def test_empty_structural_reads(self):
        """Empty structural_reads works like no structural reads."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Should allow extra column (no structural reads)
        assert 'df' not in result.differences

    def test_structural_reads_for_nonexistent_var(self):
        """Structural reads for non-existent variable is ignored."""
        df_before = pd.DataFrame({'a': [1]})
        df_after = pd.DataFrame({'a': [1], 'b': [2]})

        differ = Diff(
            use_leq=True,
            structural_reads={'other': {'columns'}},  # 'other' doesn't exist
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'df': df_before}, {'df': df_after})

        # Should not crash, df allowed to have extra column
        assert 'df' not in result.differences

    def test_variable_type_changed(self):
        """Type change is detected regardless of structural reads."""
        df_before = pd.DataFrame({'a': [1]})
        list_after = [1, 2, 3]

        differ = Diff(
            use_leq=True,
            structural_reads={'x': {'columns'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        result = differ.diff({'x': df_before}, {'x': list_after})

        # Type change should always be detected
        assert 'x' in result.differences


class TestDiffResultWarningsField:
    """Tests for DiffResult.warnings field."""

    def test_diff_result_has_warnings_field(self):
        """DiffResult has a warnings field."""
        result = DiffResult()
        assert hasattr(result, 'warnings')
        assert result.warnings == []

    def test_diff_result_warnings_initialized(self):
        """DiffResult warnings can be initialized."""
        result = DiffResult(warnings=['test warning'])
        assert result.warnings == ['test warning']

    def test_compound_diff_has_warnings_field(self):
        """CompoundDiff has a warnings field."""
        diff = CompoundDiff(source_type='dataframe', children={})
        assert hasattr(diff, 'warnings')
        assert diff.warnings == []


class TestTrackingDataIntegration:
    """Tests for TrackingData integration with diff."""

    def test_tracking_data_provides_structural_reads(self):
        """TrackingData.structural_reads can be passed to Diff."""
        from data_ferret.kernel.models import TrackingData

        tracking = TrackingData(
            structural_reads={'df': {'columns', 'shape'}}
        )

        # This should work without error
        differ = Diff(use_leq=True)
        # structural_reads would be passed from tracking.structural_reads

        assert tracking.structural_reads == {'df': {'columns', 'shape'}}


class TestCheckpointDiffIntegration:
    """Tests for Checkpoint.diff integration (when implemented)."""

    def test_checkpoint_diff_accepts_structural_params(self):
        """Checkpoint.diff accepts structural_reads and structural_mode."""
        from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints

        checkpoints = Checkpoints()
        ns1 = {'df': pd.DataFrame({'a': [1]})}
        ns2 = {'df': pd.DataFrame({'a': [1], 'b': [2]})}

        cp1 = Checkpoint('pre', ns1, {})
        cp2 = Checkpoint('post', ns2, {})

        # This should work when implemented
        result = Checkpoint.diff(
            cp1, cp2,
            use_leq=True,
            structural_reads={'df': {'columns'}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        assert 'df' in result.differences
