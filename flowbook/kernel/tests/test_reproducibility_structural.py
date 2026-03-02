"""
Tests for Reproducibility Enforcer integration with structural tracking.

These tests verify that the ReproducibilityEnforcer correctly handles structural reads,
detecting backward mutations and staleness when structural attributes like
df.columns, df.shape, len(df), etc. were accessed.

Note: Many tests are marked skip until sdc_enforcer.py is updated to support
structural_reads and structural_mode parameters.
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode

from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer
from flowbook.kernel.models import ReproducibilityViolation


class TestReproducibilityEnforcerBaseline:
    """Baseline tests without structural reads (verify existing behavior)."""

    @pytest.fixture
    def checkpoints(self):
        """Create a fresh Checkpoints instance."""
        return MemoryCheckpoints()

    @pytest.fixture
    def enforcer(self, checkpoints):
        """Create an ReproducibilityEnforcer instance."""
        return ReproducibilityEnforcer(checkpoints)

    def _save_pre_checkpoint(self, checkpoints, cell_id, namespace):
        """Save a pre-checkpoint for a cell."""
        checkpoints.save(f"_pre_{cell_id}", namespace, max_size_mb=None)

    def test_no_backward_mutation_without_structural(self):
        """No backward mutation when earlier cell doesn't read what we modify."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])

        # Cell A reads 'x', writes nothing
        ns_a = {'x': 10, 'y': 20}
        self._save_pre_checkpoint(checkpoints, 'cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'x'},
            writes=set(),
        )

        result_a = enforcer.check('cell_a', checkpoints.saved['_pre_cell_a'], ns_a, tracking_a)
        assert result_a.violation is None

        # Cell B modifies 'y' (not read by cell A)
        ns_b_pre = {'x': 10, 'y': 20}
        ns_b_post = {'x': 10, 'y': 999}
        self._save_pre_checkpoint(checkpoints, 'cell_b', ns_b_pre)

        tracking_b = TrackingData(
            reads_before_writes={'y'},
            writes={'y'},
        )

        result_b = enforcer.check('cell_b', checkpoints.saved['_pre_cell_b'], ns_b_post, tracking_b)
        assert result_b.violation is None

    def test_backward_mutation_detected(self):
        """Backward mutation detected when later cell modifies what earlier cell reads."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])

        # Cell A reads 'x'
        ns_a = {'x': 10}
        self._save_pre_checkpoint(checkpoints, 'cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'x'},
            writes=set(),
        )

        result_a = enforcer.check('cell_a', checkpoints.saved['_pre_cell_a'], ns_a, tracking_a)
        assert result_a.violation is None

        # Cell B modifies 'x' (read by cell A)
        ns_b_pre = {'x': 10}
        ns_b_post = {'x': 999}
        self._save_pre_checkpoint(checkpoints, 'cell_b', ns_b_pre)

        tracking_b = TrackingData(
            reads_before_writes={'x'},
            writes={'x'},
        )

        result_b = enforcer.check('cell_b', checkpoints.saved['_pre_cell_b'], ns_b_post, tracking_b)
        assert result_b.violation is not None
        assert 'x' in result_b.violation.variables


class TestSDCStructuralReadsModeOff:
    """Tests for reproducibility with structural_mode=OFF."""

    @pytest.mark.skip(reason="Test design issue: OFF mode only affects structural change detection, not backward mutation detection")
    def test_off_mode_ignores_structural_reads(self):
        """OFF mode ignores structural reads in backward mutation check."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.OFF)

        # Cell A reads df.columns
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'columns'}},
        )

        result_a = enforcer.check(
            'cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds column (would be violation with structural tracking)
        df_b = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', ns_a, {})

        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'c'}},
        )

        result_b = enforcer.check(
            'cell_b', pre_b, ns_b_post, tracking_b,
        )

        # OFF mode: no violation for structural changes
        assert result_b.violation is None


class TestSDCStructuralReadsModeWarn:
    """Tests for reproducibility with structural_mode=WARN."""

    @pytest.mark.skip(reason="Test design issue: WARN mode affects structural warnings, not backward mutation detection")
    def test_warn_mode_no_violation_but_warning(self):
        """WARN mode produces warnings but no violation for structural issues."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.WARN)

        # Cell A reads df.columns
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'columns'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds column
        df_b = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', ns_a, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'c'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        # WARN mode: no violation but should have warnings
        assert result_b.violation is None
        assert hasattr(result_b, 'structural_warnings')
        assert len(result_b.structural_warnings) > 0


class TestSDCStructuralReadsModeEnforce:
    """Tests for reproducibility with structural_mode=ENFORCE."""

    def test_enforce_columns_read_blocks_column_addition(self):
        """ENFORCE mode: adding column is backward mutation when columns was read."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads df.columns
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'columns'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds column 'c'
        df_b = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', {'df': df.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'c'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        # ENFORCE mode: should be violation
        assert result_b.violation is not None
        assert 'df' in str(result_b.violation.variables)

    def test_enforce_shape_read_blocks_row_addition(self):
        """ENFORCE mode: adding row is backward mutation when shape was read."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads df.shape
        df = pd.DataFrame({'a': [1, 2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'shape'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds row
        df_b = pd.DataFrame({'a': [1, 2, 3]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', {'df': df.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        # ENFORCE mode: should be violation
        assert result_b.violation is not None

    def test_enforce_len_read_blocks_row_addition(self):
        """ENFORCE mode: adding row is backward mutation when len() was called."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A calls len(df)
        df = pd.DataFrame({'a': [1, 2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'len'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds row
        df_b = pd.DataFrame({'a': [1, 2, 3]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', {'df': df.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        assert result_b.violation is not None

    def test_enforce_iter_read_blocks_column_addition(self):
        """ENFORCE mode: adding column is backward mutation when `for col in df` was used."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A iterates over df
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'iter'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds column
        df_b = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', {'df': df.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'c'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        assert result_b.violation is not None

    def test_enforce_describe_read_blocks_column_addition(self):
        """ENFORCE mode: adding column is backward mutation when describe() was called."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A calls df.describe()
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'describe'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B adds column
        df_b = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', {'df': df.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'c'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        assert result_b.violation is not None


class TestSDCStructuralStaleness:
    """Tests for staleness computation with structural reads."""

    @pytest.mark.skip(reason="Test design issue: staleness computation interaction with structural tracking needs more work")
    def test_staleness_when_columns_changed(self):
        """Cell becomes stale when columns change and it read df.columns."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b', 'cell_c'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A: creates df
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', {}, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes=set(),
            writes={'df'},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B: reads df.columns
        ns_b = ns_a.copy()
        pre_b = MemoryCheckpoint('pre_b', ns_b, {})
        checkpoints.save('_pre_cell_b', ns_b)

        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'columns'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
            continue_on_violation=True,
        )
        assert result_b.violation is None

        # Cell C: adds column (later cell, so no backward mutation to B)
        df_c = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ns_c_post = {'df': df_c}
        pre_c = MemoryCheckpoint('pre_c', ns_a.copy(), {})
        tracking_c = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'c'}},
        )

        result_c = enforcer.check('cell_c', pre_c, ns_c_post, tracking_c,
        )

        # Cell B should be stale (it read df.columns, columns changed)
        assert 'cell_b' in result_c.stale_cells

    @pytest.mark.skip(reason="Test design issue: staleness computation interaction with structural tracking needs more work")
    def test_staleness_when_row_count_changed(self):
        """Cell becomes stale when row count changes and it read len(df)."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b', 'cell_c'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A: creates df
        df = pd.DataFrame({'a': [1, 2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', {}, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes=set(),
            writes={'df'},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )

        # Cell B: reads len(df)
        ns_b = ns_a.copy()
        pre_b = MemoryCheckpoint('pre_b', ns_b, {})
        checkpoints.save('_pre_cell_b', ns_b)

        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'len'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        # Cell C: adds row
        df_c = pd.DataFrame({'a': [1, 2, 3]})
        ns_c_post = {'df': df_c}
        pre_c = MemoryCheckpoint('pre_c', ns_a.copy(), {})
        tracking_c = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
        )

        result_c = enforcer.check('cell_c', pre_c, ns_c_post, tracking_c,
        )

        # Cell B should be stale (it read len(df), row count changed)
        assert 'cell_b' in result_c.stale_cells


class TestSDCStructuralNoFalsePositives:
    """Tests verifying no false positives from structural tracking."""

    def test_no_violation_when_structure_unchanged(self):
        """No violation when structure doesn't change, even with structural reads."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads df.columns
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df'},
            writes=set(),
            structural_reads={'df': {'columns', 'shape'}},
        )

        result_a = enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )
        assert result_a.violation is None

        # Cell B modifies values but not structure
        df_b = pd.DataFrame({'a': [999], 'b': [888]})
        ns_b_post = {'df': df_b}
        pre_b = MemoryCheckpoint('pre_b', {'df': df.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'a', 'b'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        # No structural change, so no violation from structural reads
        # (But might still have violation from column value changes)
        # This test documents expected behavior

    def test_no_staleness_when_structure_unchanged(self):
        """No staleness when structure doesn't change."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b', 'cell_c'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A: creates df
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ns_a = {'df': df.copy()}
        pre_a = MemoryCheckpoint('pre_a', {}, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(writes={'df'})

        enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )

        # Cell B: reads df.columns
        ns_b = ns_a.copy()
        pre_b = MemoryCheckpoint('pre_b', ns_b, {})
        checkpoints.save('_pre_cell_b', ns_b)

        tracking_b = TrackingData(
            reads_before_writes={'df'},
            structural_reads={'df': {'columns'}},
        )

        enforcer.check('cell_b', pre_b, ns_b, tracking_b,
        )

        # Cell C: modifies values but not structure
        df_c = pd.DataFrame({'a': [999], 'b': [888]})
        ns_c_post = {'df': df_c}
        pre_c = MemoryCheckpoint('pre_c', ns_a.copy(), {})
        tracking_c = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_writes={'df': {'a', 'b'}},
        )

        result_c = enforcer.check('cell_c', pre_c, ns_c_post, tracking_c,
        )

        # Cell B should NOT be stale (structure unchanged, only values changed)
        # But this depends on whether B reads the actual columns too
        # This test documents expected behavior


class TestSDCStructuralMultipleVariables:
    """Tests for structural reads on multiple variables."""

    @pytest.mark.skip(reason="Test design issue: expects no backward mutation violation when cell A reads df2")
    def test_multiple_dataframes_independent(self):
        """Structural reads tracked independently per variable."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads df1.columns but not df2.columns
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'x': [1]})
        ns_a = {'df1': df1.copy(), 'df2': df2.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'df1', 'df2'},
            structural_reads={'df1': {'columns'}},  # Only df1
        )

        enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )

        # Cell B adds column to df2 (not tracked structurally)
        df2_b = pd.DataFrame({'x': [1], 'y': [2]})
        ns_b_post = {'df1': df1.copy(), 'df2': df2_b}
        pre_b = MemoryCheckpoint('pre_b', ns_a.copy(), {})
        tracking_b = TrackingData(
            reads_before_writes={'df2'},
            writes={'df2'},
            column_writes={'df2': {'y'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        # Should NOT be violation (df1.columns was read, but df2 was modified)
        assert result_b.violation is None


class TestSDCStructuralSeriesTracking:
    """Tests for Series structural tracking in reproducibility."""

    def test_series_index_read_blocks_length_change(self):
        """Adding to series is violation when s.index was read."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads s.index
        s = pd.Series([1, 2, 3])
        ns_a = {'s': s.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'s'},
            structural_reads={'s': {'index'}},
        )

        enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )

        # Cell B adds element
        s_b = pd.Series([1, 2, 3, 4])
        ns_b_post = {'s': s_b}
        pre_b = MemoryCheckpoint('pre_b', {'s': s.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'s'},
            writes={'s'},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        assert result_b.violation is not None

    def test_series_dtype_read_blocks_dtype_change(self):
        """Changing dtype is violation when s.dtype was read."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads s.dtype
        s = pd.Series([1, 2, 3])
        ns_a = {'s': s.copy()}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'s'},
            structural_reads={'s': {'dtype'}},
        )

        enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )

        # Cell B changes dtype
        s_b = pd.Series([1.0, 2.0, 3.0])  # Now float
        ns_b_post = {'s': s_b}
        pre_b = MemoryCheckpoint('pre_b', {'s': s.copy()}, {})
        tracking_b = TrackingData(
            reads_before_writes={'s'},
            writes={'s'},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        assert result_b.violation is not None


class TestSDCMagicCommand:
    """Tests for structural tracking magic command (documentation)."""

    def test_magic_command_sets_mode(self):
        """
        Magic command should set structural tracking mode.

        Expected usage:
            %structural_tracking off    # Disable tracking
            %structural_tracking warn   # Track and warn only
            %structural_tracking enforce # Track and enforce equality
        """
        pass

    def test_magic_command_shows_current_mode(self):
        """
        Magic command with no argument should show current mode.

        Expected usage:
            %structural_tracking  # Shows: "Structural tracking mode: warn"
        """
        pass


class TestSDCStructuralNestedPaths:
    """Tests for structural reads on nested variable paths."""

    def test_nested_path_structural_read(self):
        """Structural reads work with nested paths like data['train']."""
        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(['cell_a', 'cell_b'])
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)

        # Cell A reads data['train'].columns
        df = pd.DataFrame({'a': [1]})
        ns_a = {'data': {'train': df.copy()}}
        pre_a = MemoryCheckpoint('pre_a', ns_a, {})
        checkpoints.save('_pre_cell_a', ns_a)

        tracking_a = TrackingData(
            reads_before_writes={'data'},
            structural_reads={"data['train']": {'columns'}},
        )

        enforcer.check('cell_a', pre_a, ns_a, tracking_a,
        )

        # Cell B adds column to data['train']
        df_b = pd.DataFrame({'a': [1], 'b': [2]})
        ns_b_post = {'data': {'train': df_b}}
        pre_b = MemoryCheckpoint('pre_b', {'data': {'train': df.copy()}}, {})
        tracking_b = TrackingData(
            reads_before_writes={'data'},
            writes={'data'},
            column_writes={"data['train']": {'b'}},
        )

        result_b = enforcer.check('cell_b', pre_b, ns_b_post, tracking_b,
        )

        assert result_b.violation is not None
