"""
Tests for TrackingDict integration with structural tracking.

These tests verify that the TrackingDict class correctly integrates
with StructuralAccessTracker to track structural attribute accesses
during cell execution.
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel_support.tracking import TrackingDict
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode
from flowbook.kernel_support.models import TrackingData


class TestTrackingDictStructuralSetup:
    """Tests for structural tracking setup in TrackingDict."""

    def test_has_structural_tracker(self):
        """TrackingDict has a structural tracker."""
        td = TrackingDict()
        assert hasattr(td, '_structural_tracker')

    def test_default_mode_is_enforce(self):
        """Default structural tracking mode is ENFORCE."""
        td = TrackingDict()
        assert td.structural_tracking_mode == StructuralTrackingMode.ENFORCE

    def test_set_structural_tracking_mode(self):
        """Can set structural tracking mode."""
        td = TrackingDict()
        td.set_structural_tracking_mode("enforce")
        assert td.structural_tracking_mode == StructuralTrackingMode.ENFORCE

    def test_set_mode_case_insensitive(self):
        """Mode setting is case insensitive."""
        td = TrackingDict()
        td.set_structural_tracking_mode("OFF")
        assert td.structural_tracking_mode == StructuralTrackingMode.OFF

    def test_structural_reads_property(self):
        """TrackingDict has structural_reads property."""
        td = TrackingDict()
        assert isinstance(td.structural_reads, dict)


class TestTrackingDictStructuralExecution:
    """Tests for structural tracking during execution."""

    def test_track_df_columns_access(self):
        """Tracks df.columns access during execution."""
        ns = {'df': pd.DataFrame({'a': [1], 'b': [2]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']

    def test_track_df_shape_access(self):
        """Tracks df.shape access during execution."""
        ns = {'df': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.shape

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'shape' in result.structural_reads['df']

    def test_track_len_df_access(self):
        """Tracks len(df) access during execution."""
        ns = {'df': pd.DataFrame({'a': [1, 2, 3]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = len(df)

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'len' in result.structural_reads['df']

    def test_track_df_iteration(self):
        """Tracks `for col in df` during execution."""
        ns = {'df': pd.DataFrame({'a': [1], 'b': [2]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            for col in df:
                pass

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'iter' in result.structural_reads['df']

    def test_track_df_dtypes(self):
        """Tracks df.dtypes access during execution."""
        ns = {'df': pd.DataFrame({'a': [1], 'b': [1.5]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.dtypes

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'dtypes' in result.structural_reads['df']

    def test_track_df_index(self):
        """Tracks df.index access during execution."""
        ns = {'df': pd.DataFrame({'a': [1, 2]}, index=['x', 'y'])}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.index

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'index' in result.structural_reads['df']

    def test_track_df_describe(self):
        """Tracks df.describe() access during execution."""
        ns = {'df': pd.DataFrame({'a': [1, 2, 3]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.describe()

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'describe' in result.structural_reads['df']

    def test_track_df_to_dict(self):
        """Tracks df.to_dict() access during execution."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.to_dict()

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'to_dict' in result.structural_reads['df']


class TestTrackingDictStructuralModes:
    """Tests for different structural tracking modes."""

    def test_off_mode_no_tracking(self):
        """OFF mode doesn't track structural reads."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("off")

        with td.track_execution():
            df = td['df']
            _ = df.columns
            _ = df.shape
            _ = len(df)

        result = td.get_tracking_data()
        assert result.structural_reads == {}

    def test_warn_mode_tracks(self):
        """WARN mode tracks structural reads."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("warn")

        with td.track_execution():
            df = td['df']
            _ = df.columns

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']

    def test_enforce_mode_tracks(self):
        """ENFORCE mode tracks structural reads."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads


class TestTrackingDictMultipleDataFrames:
    """Tests for tracking multiple DataFrames."""

    def test_track_multiple_dfs_independently(self):
        """Tracks multiple DataFrames independently."""
        ns = {
            'df1': pd.DataFrame({'a': [1]}),
            'df2': pd.DataFrame({'b': [2]}),
        }
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df1 = td['df1']
            df2 = td['df2']
            _ = df1.columns
            _ = df2.shape

        result = td.get_tracking_data()
        assert 'df1' in result.structural_reads
        assert 'columns' in result.structural_reads['df1']
        assert 'df2' in result.structural_reads
        assert 'shape' in result.structural_reads['df2']

    def test_track_df_and_series(self):
        """Tracks DataFrame and Series together."""
        ns = {
            'df': pd.DataFrame({'a': [1]}),
            's': pd.Series([1, 2, 3]),
        }
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            s = td['s']
            _ = df.columns
            _ = s.index

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']
        assert 's' in result.structural_reads
        assert 'index' in result.structural_reads['s']


class TestTrackingDictSeriesTracking:
    """Tests for Series structural tracking.

    Note: These tests are skipped because walk_dataframes only walks DataFrames.
    Series tracking works at the StructuralAccessTracker level but not when
    integrated with TrackingDict (which uses walk_dataframes for registration).
    A future improvement would be to create walk_pandas_objects that yields
    both DataFrames and Series.
    """

    def test_track_series_index(self):
        """Tracks s.index access."""
        ns = {'s': pd.Series([1, 2, 3])}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            s = td['s']
            _ = s.index

        result = td.get_tracking_data()
        assert 's' in result.structural_reads
        assert 'index' in result.structural_reads['s']

    def test_track_series_dtype(self):
        """Tracks s.dtype access."""
        ns = {'s': pd.Series([1, 2, 3])}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            s = td['s']
            _ = s.dtype

        result = td.get_tracking_data()
        assert 's' in result.structural_reads
        assert 'dtype' in result.structural_reads['s']

    def test_track_series_shape(self):
        """Tracks s.shape access."""
        ns = {'s': pd.Series([1, 2, 3])}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            s = td['s']
            _ = s.shape

        result = td.get_tracking_data()
        assert 's' in result.structural_reads
        assert 'shape' in result.structural_reads['s']

    def test_track_series_name(self):
        """Tracks s.name access."""
        ns = {'s': pd.Series([1, 2], name='test')}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            s = td['s']
            _ = s.name

        result = td.get_tracking_data()
        assert 's' in result.structural_reads
        assert 'name' in result.structural_reads['s']

    def test_track_len_series(self):
        """Tracks len(s) access."""
        ns = {'s': pd.Series([1, 2, 3])}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            s = td['s']
            _ = len(s)

        result = td.get_tracking_data()
        assert 's' in result.structural_reads
        assert 'len' in result.structural_reads['s']


class TestTrackingDictNestedDataFrames:
    """Tests for nested DataFrame tracking."""

    def test_track_df_in_dict(self):
        """Tracks DataFrame in a dict."""
        inner_df = pd.DataFrame({'a': [1]})
        ns = {'data': {'train': inner_df}}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            data = td['data']
            df = data['train']
            _ = df.columns

        result = td.get_tracking_data()
        assert "data['train']" in result.structural_reads
        assert 'columns' in result.structural_reads["data['train']"]

    def test_track_df_in_list(self):
        """Tracks DataFrame in a list."""
        df = pd.DataFrame({'a': [1]})
        ns = {'dfs': [df]}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            dfs = td['dfs']
            _ = dfs[0].columns

        result = td.get_tracking_data()
        assert 'dfs[0]' in result.structural_reads
        assert 'columns' in result.structural_reads['dfs[0]']


class TestTrackingDictResetBehavior:
    """Tests for reset behavior."""

    def test_reset_clears_structural_reads(self):
        """reset_tracking clears structural reads."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns

        result_before = td.get_tracking_data()
        assert 'df' in result_before.structural_reads

        td.reset_tracking()
        result_after = td.get_tracking_data()
        assert result_after.structural_reads == {}

    def test_new_execution_clears_previous(self):
        """New track_execution context clears previous tracking."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        # First execution
        with td.track_execution():
            df = td['df']
            _ = df.columns

        # Second execution (should start fresh)
        with td.track_execution():
            pass  # No structural access

        result = td.get_tracking_data()
        # Should be empty from second execution
        assert result.structural_reads == {}


class TestTrackingDictExecIntegration:
    """Tests for exec() integration with structural tracking."""

    def test_exec_tracks_structural_access(self):
        """exec() with TrackingDict tracks structural access."""
        ns = {'df': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        code = "cols = df.columns"

        with td.track_execution():
            exec(code, td)

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']

    def test_exec_tracks_len_call(self):
        """exec() with TrackingDict tracks len() calls."""
        ns = {'df': pd.DataFrame({'a': [1, 2, 3]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        code = "n = len(df)"

        with td.track_execution():
            exec(code, td)

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'len' in result.structural_reads['df']

    def test_exec_tracks_iteration(self):
        """exec() with TrackingDict tracks iteration."""
        ns = {'df': pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        code = "cols = list(df)"

        with td.track_execution():
            exec(code, td)

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'iter' in result.structural_reads['df']

    def test_exec_complex_code(self):
        """exec() with complex code tracks structural access correctly."""
        ns = {'df': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        code = """
n_rows = len(df)
n_cols = len(df.columns)
col_names = list(df)
"""

        with td.track_execution():
            exec(code, td)

        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        # Should have tracked len, columns access, and iteration
        assert 'len' in result.structural_reads['df']
        assert 'columns' in result.structural_reads['df']
        assert 'iter' in result.structural_reads['df']


class TestTrackingDictCombinedTracking:
    """Tests for combined variable and structural tracking."""

    def test_variable_and_structural_tracking(self):
        """Both variable access and structural access are tracked."""
        ns = {
            'df': pd.DataFrame({'a': [1]}),
            'x': 10,
        }
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            x = td['x']
            _ = df.columns
            y = x + 5
            td['y'] = y

        result = td.get_tracking_data()

        # Variable tracking
        assert 'df' in result.reads_before_writes
        assert 'x' in result.reads_before_writes
        assert 'y' in result.writes

        # Structural tracking
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']

    def test_column_and_structural_tracking(self):
        """Both column access and structural access are tracked."""
        ns = {'df': pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df['a']  # Column access
            _ = df.columns  # Structural access

        result = td.get_tracking_data()

        # Column tracking
        assert 'df' in result.column_reads_before_writes
        assert 'a' in result.column_reads_before_writes['df']

        # Structural tracking
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']


class TestTrackingDictNewDataFrames:
    """Tests for tracking newly created DataFrames."""

    def test_track_newly_created_df(self):
        """Tracks structural access on newly created DataFrames.

        Note: Variables created during execution ARE tracked as writes
        when exec() uses the TrackingDict as both globals and locals.
        """
        td = TrackingDict({})
        td['pd'] = pd  # Make pd available
        td.set_structural_tracking_mode("enforce")

        code = """
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
"""

        with td.track_execution():
            exec(code, td)

        result = td.get_tracking_data()
        # The df was written during execution
        assert 'df' in result.writes


class TestTrackingDataModelIntegration:
    """Tests for TrackingData model integration."""

    def test_tracking_data_structural_reads_format(self):
        """TrackingData structural_reads is Dict[str, Set[str]]."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns
            _ = df.shape

        result = td.get_tracking_data()

        assert isinstance(result.structural_reads, dict)
        assert isinstance(result.structural_reads['df'], set)
        # Check that at least our accessed attributes are tracked
        # (pandas may internally access additional attributes like 'index')
        assert 'columns' in result.structural_reads['df']
        assert 'shape' in result.structural_reads['df']

    def test_tracking_data_has_structural_read(self):
        """TrackingData.has_structural_read works correctly."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns

        result = td.get_tracking_data()

        assert result.has_structural_read('df')
        assert not result.has_structural_read('other')

    def test_tracking_data_has_column_structure_read(self):
        """TrackingData.has_column_structure_read works correctly."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns

        result = td.get_tracking_data()

        assert result.has_column_structure_read('df')

    def test_tracking_data_has_row_structure_read(self):
        """TrackingData.has_row_structure_read works correctly."""
        ns = {'df': pd.DataFrame({'a': [1, 2, 3]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = len(df)

        result = td.get_tracking_data()

        assert result.has_row_structure_read('df')


class TestTrackingDictEdgeCases:
    """Tests for edge cases."""

    def test_empty_namespace(self):
        """Works with empty namespace."""
        td = TrackingDict({})
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            pass

        result = td.get_tracking_data()
        assert result.structural_reads == {}

    def test_no_dataframes(self):
        """Works when no DataFrames in namespace."""
        ns = {'x': 10, 'y': [1, 2, 3]}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            _ = td['x']

        result = td.get_tracking_data()
        assert result.structural_reads == {}

    def test_exception_in_execution(self):
        """Structural tracking cleaned up after exception."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        try:
            with td.track_execution():
                df = td['df']
                _ = df.columns
                raise ValueError("test error")
        except ValueError:
            pass

        # Should still get tracking data
        result = td.get_tracking_data()
        assert 'df' in result.structural_reads
        assert 'columns' in result.structural_reads['df']

    def test_structural_tracker_uninstalled_after_execution(self):
        """Structural tracker patches are uninstalled after execution."""
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            pass

        # Verify patches are uninstalled
        assert not td._structural_tracker._installed

    def test_mode_change_during_execution(self):
        """Mode change during execution affects subsequent accesses.

        Note: Changing mode during execution affects tracking because the mode
        is checked at access time. This test documents actual behavior: setting
        mode to OFF after start_column_tracking() means resolve_to_paths()
        returns empty (OFF mode returns empty from resolve_to_paths).
        """
        ns = {'df': pd.DataFrame({'a': [1]})}
        td = TrackingDict(ns)
        td.set_structural_tracking_mode("enforce")

        with td.track_execution():
            df = td['df']
            _ = df.columns
            # Changing mode during execution affects resolve_to_paths result
            td.set_structural_tracking_mode("off")
            _ = df.shape

        result = td.get_tracking_data()
        # OFF mode causes resolve_to_paths to return empty dict
        # This is documented behavior - mode change during execution
        # will affect the final result
        assert result.structural_reads == {}
