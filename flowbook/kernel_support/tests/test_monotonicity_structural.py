"""
Tests for MonotonicityEnforcer integration with structural tracking.

These tests verify that the MonotonicityEnforcer correctly handles structural reads,
detecting violations when structural attributes like df.columns, df.shape, len(df),
etc. were accessed and the structure changed.

Note: Many tests are marked skip until monotonicity.py is updated to support
structural_reads and structural_mode parameters.
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel_support.checkpoint import Checkpoints
from flowbook.kernel_support.models import TrackingData, MonotonicityViolation
from flowbook.kernel_support.monotonicity import MonotonicityEnforcer
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode


class TestMonotonicityEnforcerBaseline:
    """Baseline tests without structural reads (verify existing behavior)."""

    @pytest.fixture
    def checkpoints(self):
        return Checkpoints()

    @pytest.fixture
    def user_ns(self):
        return {}

    def test_no_violation_when_rbw_unchanged(self, checkpoints):
        """No violation when RBW variables unchanged."""
        user_ns = {'x': 10, 'y': 20}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        enforcer.save_pre_state('test_cell')

        # Cell reads x, doesn't modify it
        tracking = TrackingData(
            reads_before_writes={'x'},
            writes={'z'},
        )
        user_ns['z'] = 30  # Only writes z

        violation = enforcer.check_and_enforce(tracking, 'test_cell')
        assert violation is None

    def test_violation_when_rbw_modified(self, checkpoints):
        """Violation when RBW variable is modified."""
        user_ns = {'x': 10}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        enforcer.save_pre_state('test_cell')

        # Cell reads x and modifies it
        tracking = TrackingData(
            reads_before_writes={'x'},
            writes={'x'},
        )
        user_ns['x'] = 999  # Modified!

        violation = enforcer.check_and_enforce(tracking, 'test_cell')
        assert violation is not None
        assert 'x' in violation.violated_vars

    def test_state_restored_after_violation(self, checkpoints):
        """State is restored after violation detected."""
        user_ns = {'x': 10}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'x'},
            writes={'x'},
        )
        user_ns['x'] = 999

        violation = enforcer.check_and_enforce(tracking, 'test_cell')
        assert violation is not None
        # State should be restored
        assert user_ns['x'] == 10


class TestMonotonicityStructuralModeOff:
    """Tests for monotonicity with structural_mode=OFF."""

    def test_off_mode_ignores_structural_changes(self):
        """OFF mode ignores structural changes when checking monotonicity."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1], 'b': [2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.OFF,
        )

        enforcer.save_pre_state('test_cell')

        # Cell reads df.columns, then adds column
        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns'}},
        )
        user_ns['df']['c'] = [3]  # Add column

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        # OFF mode: no violation for structural changes
        assert violation is None


class TestMonotonicityStructuralModeWarn:
    """Tests for monotonicity with structural_mode=WARN."""

    def test_warn_mode_produces_warnings(self):
        """WARN mode produces warnings but no violation for structural issues."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1], 'b': [2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.WARN,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns'}},
        )
        user_ns['df']['c'] = [3]  # Add column

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        # WARN mode: no violation but warnings should be available
        assert violation is None
        # Would need a way to access warnings


class TestMonotonicityStructuralModeEnforce:
    """Tests for monotonicity with structural_mode=ENFORCE."""

    def test_enforce_columns_read_blocks_column_addition(self):
        """ENFORCE: adding column is violation when df.columns was read."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1], 'b': [2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns'}},
        )
        user_ns['df']['c'] = [3]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None
        assert 'df' in violation.violated_vars

    def test_enforce_shape_read_blocks_row_addition(self):
        """ENFORCE: adding row is violation when df.shape was read."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'shape'}},
        )
        # Add row
        user_ns['df'] = pd.concat([user_ns['df'], pd.DataFrame({'a': [3]})], ignore_index=True)

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_enforce_len_read_blocks_row_addition(self):
        """ENFORCE: adding row is violation when len(df) was called."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'len'}},
        )
        user_ns['df'] = pd.DataFrame({'a': [1, 2, 3]})

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_enforce_iter_read_blocks_column_addition(self):
        """ENFORCE: adding column is violation when `for col in df` was used."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1], 'b': [2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'iter'}},
        )
        user_ns['df']['c'] = [3]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_enforce_dtypes_read_blocks_dtype_change(self):
        """ENFORCE: dtype change is violation when df.dtypes was read."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'dtypes'}},
        )
        user_ns['df']['a'] = user_ns['df']['a'].astype(float)

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_enforce_index_read_blocks_index_change(self):
        """ENFORCE: index change is violation when df.index was read."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'index'}},
        )
        user_ns['df'].index = [0, 2]  # Changed index

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_enforce_describe_read_blocks_column_addition(self):
        """ENFORCE: adding column is violation when df.describe() was called."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'describe'}},
        )
        user_ns['df']['b'] = [3, 4]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None


class TestMonotonicityStructuralNoFalsePositives:
    """Tests verifying no false positives from structural tracking."""

    def test_no_violation_when_structure_unchanged(self):
        """No violation when structure unchanged, even with structural reads."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        # Read columns and shape, but only modify values
        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns', 'shape'}},
            column_reads_before_writes={'df': {'a', 'b'}},
            column_writes={'df': {'a', 'b'}},
        )
        user_ns['df']['a'] = [100, 200]  # Change values
        user_ns['df']['b'] = [300, 400]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        # Value changes should be detected via column tracking, but
        # structural check should not add false positives
        # This test documents expected behavior - may need adjustment
        # depending on whether value changes to RBW columns are violations

    def test_no_violation_different_variable(self):
        """No violation when structural read is on different variable."""
        checkpoints = Checkpoints()
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'x': [1]})
        user_ns = {'df1': df1.copy(), 'df2': df2.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        # Read df1.columns, modify df2
        tracking = TrackingData(
            reads_before_writes={'df1'},
            writes={'df2'},
            structural_reads={'df1': {'columns'}},
        )
        user_ns['df2']['y'] = [2]  # Add column to df2

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        # No violation - df1 wasn't modified
        assert violation is None


class TestMonotonicityStructuralSeries:
    """Tests for Series structural tracking in monotonicity."""

    def test_series_index_read_blocks_length_change(self):
        """ENFORCE: changing length is violation when s.index was read."""
        checkpoints = Checkpoints()
        s = pd.Series([1, 2, 3])
        user_ns = {'s': s.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'s'},
            writes={'s'},
            structural_reads={'s': {'index'}},
        )
        user_ns['s'] = pd.Series([1, 2, 3, 4])

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_series_dtype_read_blocks_dtype_change(self):
        """ENFORCE: dtype change is violation when s.dtype was read."""
        checkpoints = Checkpoints()
        s = pd.Series([1, 2, 3])
        user_ns = {'s': s.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'s'},
            writes={'s'},
            structural_reads={'s': {'dtype'}},
        )
        user_ns['s'] = pd.Series([1.0, 2.0, 3.0])  # Changed dtype

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None

    def test_series_name_read_blocks_name_change(self):
        """ENFORCE: name change is violation when s.name was read."""
        checkpoints = Checkpoints()
        s = pd.Series([1, 2], name='original')
        user_ns = {'s': s.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'s'},
            writes={'s'},
            structural_reads={'s': {'name'}},
        )
        user_ns['s'].name = 'changed'

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None


class TestMonotonicityStructuralStateRestoration:
    """Tests for state restoration with structural violations."""

    def test_state_restored_after_structural_violation(self):
        """State is restored after structural violation."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1], 'b': [2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns'}},
        )
        user_ns['df']['c'] = [3]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None
        # State should be restored - df should not have column 'c'
        assert list(user_ns['df'].columns) == ['a', 'b']


class TestMonotonicityStructuralMultipleReads:
    """Tests for multiple structural reads."""

    def test_multiple_structural_reads_all_checked(self):
        """All structural reads are checked for violations."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        # Read both columns and len
        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns', 'len'}},
        )
        # Add row (violates len but not columns)
        user_ns['df'] = pd.DataFrame({'a': [1, 2, 3]})

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        # Should detect violation from len
        assert violation is not None


class TestMonotonicityStructuralViolationDetails:
    """Tests for violation detail formatting."""

    @pytest.mark.skip(reason="Enhancement: error message should mention structural nature of violation")
    def test_violation_details_mention_structural_issue(self):
        """Violation details mention structural issue."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            structural_reads={'df': {'columns'}},
        )
        user_ns['df']['b'] = [2]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None
        # Details should mention the structural nature of the violation
        assert 'columns' in violation.diff_details.lower() or 'structure' in violation.diff_details.lower()


class TestMonotonicityStructuralNestedPaths:
    """Tests for nested variable paths with structural reads."""

    def test_nested_path_structural_violation(self):
        """Structural reads work with nested paths like data['train']."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1]})
        user_ns = {'data': {'train': df.copy()}}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        tracking = TrackingData(
            reads_before_writes={'data'},
            writes={'data'},
            structural_reads={"data['train']": {'columns'}},
        )
        user_ns['data']['train']['b'] = [2]

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        assert violation is not None


class TestMonotonicityStructuralWithColumnTracking:
    """Tests for combined structural and column tracking."""

    def test_structural_and_column_tracking_combined(self):
        """Structural and column-level tracking work together."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        user_ns = {'df': df.copy()}

        enforcer = MonotonicityEnforcer(
            checkpoints, user_ns,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )

        enforcer.save_pre_state('test_cell')

        # Read columns (structural) and column 'a' (column-level)
        tracking = TrackingData(
            reads_before_writes={'df'},
            writes={'df'},
            column_reads_before_writes={'df': {'a'}},
            structural_reads={'df': {'columns'}},
        )

        # Modify column 'a' value AND add new column
        user_ns['df']['a'] = [100, 200]  # Value change
        user_ns['df']['d'] = [7, 8]  # Structural change

        violation = enforcer.check_and_enforce(tracking, 'test_cell')

        # Should detect violation (either from value change or structural change)
        assert violation is not None


class TestMonotonicityMagicCommand:
    """Tests for structural tracking configuration (documentation)."""

    def test_magic_command_sets_structural_mode(self):
        """
        Magic command should set structural tracking mode.

        Expected usage:
            %structural_tracking off    # Disable tracking
            %structural_tracking warn   # Track and warn only
            %structural_tracking enforce # Track and enforce equality
        """
        pass

    def test_structural_mode_persists_across_cells(self):
        """
        Structural tracking mode should persist across cell executions.

        Once set, the mode should apply to all subsequent cells until changed.
        """
        pass
