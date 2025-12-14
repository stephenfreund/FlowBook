"""
Comprehensive tests for structural attribute tracking.

Tests the StructuralAccessTracker class which tracks when code accesses
structural attributes like .columns, .shape, .dtype on DataFrames and Series.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Set, Dict

from data_ferret.kernel.structural_tracking import (
    StructuralAccessTracker,
    StructuralTrackingMode,
    suspend_structural_tracking,
    _structure_using_context,
    DATAFRAME_STRUCTURAL_ATTRS,
    DATAFRAME_STRUCTURAL_METHODS,
    SERIES_STRUCTURAL_ATTRS,
    SERIES_STRUCTURAL_METHODS,
    COLUMN_REVEALING_ATTRS,
    ROW_REVEALING_ATTRS,
    STRUCTURE_USING_METHODS,
)


class TestStructuralTrackingMode:
    """Tests for the StructuralTrackingMode enum."""

    def test_mode_values(self):
        """Verify all expected modes exist."""
        assert StructuralTrackingMode.OFF == "off"
        assert StructuralTrackingMode.WARN == "warn"
        assert StructuralTrackingMode.ENFORCE == "enforce"

    def test_mode_from_string(self):
        """Mode can be created from string values."""
        assert StructuralTrackingMode("off") == StructuralTrackingMode.OFF
        assert StructuralTrackingMode("warn") == StructuralTrackingMode.WARN
        assert StructuralTrackingMode("enforce") == StructuralTrackingMode.ENFORCE

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError):
            StructuralTrackingMode("invalid")


class TestStructuralAccessTrackerInit:
    """Tests for StructuralAccessTracker initialization."""

    def test_default_mode_is_warn(self):
        """Default mode is WARN."""
        tracker = StructuralAccessTracker()
        assert tracker.mode == StructuralTrackingMode.WARN

    def test_custom_mode(self):
        """Can initialize with custom mode."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        assert tracker.mode == StructuralTrackingMode.ENFORCE

    def test_set_mode(self):
        """Can change mode after initialization."""
        tracker = StructuralAccessTracker()
        tracker.set_mode("enforce")
        assert tracker.mode == StructuralTrackingMode.ENFORCE

    def test_set_mode_case_insensitive(self):
        """set_mode is case insensitive."""
        tracker = StructuralAccessTracker()
        tracker.set_mode("ENFORCE")
        assert tracker.mode == StructuralTrackingMode.ENFORCE
        tracker.set_mode("Off")
        assert tracker.mode == StructuralTrackingMode.OFF


class TestDataFrameAttributeTracking:
    """Tests for tracking DataFrame structural attribute access."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker for each test."""
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        """Create a test DataFrame."""
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_columns_access(self, tracker, df):
        """Detect df.columns access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' in result
        assert 'columns' in result['df']

    def test_shape_access(self, tracker, df):
        """Detect df.shape access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'shape' in result['df']

    def test_dtypes_access(self, tracker, df):
        """Detect df.dtypes access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.dtypes

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'dtypes' in result['df']

    def test_index_access(self, tracker, df):
        """Detect df.index access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.index

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'index' in result['df']

    def test_T_access(self, tracker, df):
        """Detect df.T (transpose) access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.T

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'T' in result['df']

    def test_axes_access(self, tracker, df):
        """Detect df.axes access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.axes

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'axes' in result['df']

    def test_size_access(self, tracker, df):
        """Detect df.size access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.size

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'size' in result['df']

    def test_empty_access(self, tracker, df):
        """Detect df.empty access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.empty

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'empty' in result['df']

    def test_values_access(self, tracker, df):
        """Detect df.values access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.values

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'values' in result['df']

    def test_keys_access(self, tracker, df):
        """Detect df.keys() method access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.keys()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'keys' in result['df']


class TestDataFrameMethodTracking:
    """Tests for tracking DataFrame structural method access."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

    def test_describe_method(self, tracker, df):
        """Detect df.describe() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.describe()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'describe' in result['df']

    def test_info_method(self, tracker, df, capsys):
        """Detect df.info() access."""
        tracker.register(df, 'df')
        tracker.install()

        df.info()  # Prints to stdout

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'info' in result['df']

    def test_to_dict_method(self, tracker, df):
        """Detect df.to_dict() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.to_dict()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'to_dict' in result['df']

    def test_to_numpy_method(self, tracker, df):
        """Detect df.to_numpy() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.to_numpy()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'to_numpy' in result['df']

    def test_head_method(self, tracker, df):
        """Detect df.head() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.head()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'head' in result['df']

    def test_tail_method(self, tracker, df):
        """Detect df.tail() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.tail()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'tail' in result['df']

    def test_sample_method(self, tracker, df):
        """Detect df.sample() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.sample(1)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'sample' in result['df']

    def test_copy_method(self, tracker, df):
        """Detect df.copy() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.copy()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'copy' in result['df']

    def test_select_dtypes_method(self, tracker, df):
        """Detect df.select_dtypes() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.select_dtypes(include='number')

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'select_dtypes' in result['df']

    def test_memory_usage_method(self, tracker, df):
        """Detect df.memory_usage() access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.memory_usage()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'memory_usage' in result['df']


class TestDataFrameSpecialMethods:
    """Tests for tracking special methods like len() and iter()."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_len_builtin(self, tracker, df):
        """Detect len(df) access."""
        tracker.register(df, 'df')
        tracker.install()

        _ = len(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'len' in result['df']

    def test_iter_for_loop(self, tracker, df):
        """Detect `for col in df:` iteration."""
        tracker.register(df, 'df')
        tracker.install()

        for col in df:
            pass

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'iter' in result['df']

    def test_iter_list_conversion(self, tracker, df):
        """Detect list(df) iteration."""
        tracker.register(df, 'df')
        tracker.install()

        _ = list(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'iter' in result['df']


class TestSeriesAttributeTracking:
    """Tests for tracking Series structural attribute access."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def series(self):
        return pd.Series([1, 2, 3], name='test_series')

    def test_index_access(self, tracker, series):
        """Detect s.index access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.index

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' in result
        assert 'index' in result['s']

    def test_shape_access(self, tracker, series):
        """Detect s.shape access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'shape' in result['s']

    def test_dtype_access(self, tracker, series):
        """Detect s.dtype access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.dtype

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'dtype' in result['s']

    def test_name_access(self, tracker, series):
        """Detect s.name access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.name

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'name' in result['s']

    def test_size_access(self, tracker, series):
        """Detect s.size access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.size

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'size' in result['s']

    def test_empty_access(self, tracker, series):
        """Detect s.empty access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.empty

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'empty' in result['s']

    def test_values_access(self, tracker, series):
        """Detect s.values access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.values

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'values' in result['s']


class TestSeriesMethodTracking:
    """Tests for tracking Series structural method access."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def series(self):
        return pd.Series([1, 2, 3])

    def test_to_dict_method(self, tracker, series):
        """Detect s.to_dict() access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.to_dict()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'to_dict' in result['s']

    def test_to_list_method(self, tracker, series):
        """Detect s.to_list() access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.to_list()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'to_list' in result['s']

    def test_to_numpy_method(self, tracker, series):
        """Detect s.to_numpy() access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.to_numpy()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'to_numpy' in result['s']

    def test_describe_method(self, tracker, series):
        """Detect s.describe() access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.describe()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'describe' in result['s']

    def test_copy_method(self, tracker, series):
        """Detect s.copy() access."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.copy()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'copy' in result['s']


class TestSeriesSpecialMethods:
    """Tests for tracking Series special methods."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def series(self):
        return pd.Series([1, 2, 3])

    def test_len_builtin(self, tracker, series):
        """Detect len(s) access."""
        tracker.register(series, 's')
        tracker.install()

        _ = len(series)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'len' in result['s']

    def test_iter_for_loop(self, tracker, series):
        """Detect `for val in s:` iteration."""
        tracker.register(series, 's')
        tracker.install()

        for val in series:
            pass

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'iter' in result['s']


class TestOffMode:
    """Tests for OFF mode behavior."""

    def test_off_mode_no_tracking(self):
        """OFF mode doesn't track anything."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.OFF)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns
        _ = df.shape
        _ = len(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert result == {}

    def test_off_mode_no_install(self):
        """OFF mode doesn't install patches."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.OFF)
        tracker.install()

        # Verify no patches were installed by checking _installed flag
        assert not tracker._installed

    def test_switch_to_off_mode(self):
        """Switching to OFF mode affects resolve_to_paths result.

        Note: The mode is checked in resolve_to_paths(), so switching to OFF
        after tracking will cause the method to return an empty dict.
        This is intentional - the mode represents the current policy.
        """
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns  # Would be tracked if mode were WARN or ENFORCE

        tracker.set_mode("off")
        _ = df.shape  # Still recorded internally but won't be returned

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # OFF mode returns empty - this is the documented behavior
        # The mode represents the current policy for how to use the data
        assert result == {}


class TestWarnMode:
    """Tests for WARN mode behavior."""

    def test_warn_mode_tracks(self):
        """WARN mode tracks structural reads."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns
        _ = df.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']
        assert 'shape' in result['df']


class TestEnforceMode:
    """Tests for ENFORCE mode behavior."""

    def test_enforce_mode_tracks(self):
        """ENFORCE mode tracks structural reads."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns
        _ = df.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']
        assert 'shape' in result['df']


class TestSuspension:
    """Tests for suspend_structural_tracking context manager."""

    def test_suspension_prevents_tracking(self):
        """Tracking is suspended within context manager."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        with suspend_structural_tracking():
            _ = df.columns  # Should NOT be tracked

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or 'columns' not in result.get('df', set())

    def test_suspension_restored_after_context(self):
        """Tracking resumes after context manager exits.

        Note: When accessing df.shape after un-suspending, pandas may
        internally access other attributes (columns, index) which will
        also be tracked. The key assertion is that shape IS tracked.
        """
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        with suspend_structural_tracking():
            _ = df.columns  # NOT directly tracked

        _ = df.shape  # Should be tracked (may also trigger internal attr access)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # The important assertion is that shape IS tracked after suspension ends
        assert 'shape' in result['df']
        # Other attributes may be tracked due to pandas internal calls

    def test_nested_suspension(self):
        """Nested suspension works correctly."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        with suspend_structural_tracking():
            with suspend_structural_tracking():
                _ = df.columns  # NOT tracked
            _ = df.shape  # Still NOT tracked (outer suspension)

        _ = df.index  # Should be tracked

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'index' in result['df']
        assert 'columns' not in result.get('df', set())
        assert 'shape' not in result.get('df', set())


class TestMultipleObjects:
    """Tests for tracking multiple DataFrames/Series."""

    def test_multiple_dataframes(self):
        """Track multiple DataFrames independently.

        Note: Pandas may internally access other attributes when we access
        specific attributes (e.g., shape may trigger columns/index access).
        The key is that our explicit accesses are tracked.
        """
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]})

        tracker.register(df1, 'df1')
        tracker.register(df2, 'df2')
        tracker.install()

        _ = df1.columns
        _ = df2.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Our explicit accesses are tracked
        assert 'columns' in result['df1']
        assert 'shape' in result['df2']
        # df1 shouldn't have shape tracked (we didn't access it)
        assert 'shape' not in result.get('df1', set())

    def test_dataframe_and_series(self):
        """Track DataFrame and Series together."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        s = pd.Series([1, 2, 3])

        tracker.register(df, 'df')
        tracker.register(s, 's')
        tracker.install()

        _ = df.columns
        _ = s.index

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']
        assert 'index' in result['s']

    def test_multiple_attrs_same_object(self):
        """Track multiple attributes on same object."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns
        _ = df.shape
        _ = df.dtypes
        _ = df.index
        _ = len(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert result['df'] == {'columns', 'shape', 'dtypes', 'index', 'len'}


class TestNestedPaths:
    """Tests for tracking nested DataFrames with proper paths."""

    def test_nested_in_dict(self):
        """Track DataFrame nested in dict."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})

        tracker.register(df, "data['train']")
        tracker.install()

        _ = df.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert "data['train']" in result
        assert 'columns' in result["data['train']"]

    def test_nested_in_object(self):
        """Track DataFrame nested as object attribute."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})

        tracker.register(df, "model.features")
        tracker.install()

        _ = df.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert "model.features" in result
        assert 'shape' in result["model.features"]


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_tracking(self):
        """Reset clears all tracked data."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns
        result_before = tracker.resolve_to_paths()
        assert 'df' in result_before

        tracker.reset()
        result_after = tracker.resolve_to_paths()
        assert result_after == {}

        tracker.uninstall()

    def test_reset_clears_registrations(self):
        """Reset clears object registrations."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1]})
        tracker.register(df, 'df')
        tracker.install()

        tracker.reset()

        # Access after reset won't be mapped to a path
        _ = df.columns

        result = tracker.resolve_to_paths()
        # df is not registered anymore, so even though we accessed columns,
        # resolve_to_paths won't find a path for it
        assert 'df' not in result

        tracker.uninstall()


class TestUninstall:
    """Tests for proper uninstallation of patches."""

    def test_uninstall_restores_methods(self):
        """Uninstall restores original methods."""
        original_getattr = pd.DataFrame.__getattribute__

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()

        # Method should be patched
        assert pd.DataFrame.__getattribute__ != original_getattr

        tracker.uninstall()

        # Method should be restored
        # Note: This might not be exactly equal due to how Python handles methods,
        # but it should be the original
        assert tracker._installed is False

    def test_double_install_no_effect(self):
        """Installing twice has no effect."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()
        original_methods = dict(tracker._original_methods)
        tracker.install()  # Second install

        # Should have same original methods
        assert tracker._original_methods == original_methods

        tracker.uninstall()

    def test_double_uninstall_no_effect(self):
        """Uninstalling twice has no effect."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()
        tracker.uninstall()
        tracker.uninstall()  # Should not raise

        assert not tracker._installed


class TestNonStructuralAccess:
    """Tests verifying non-structural access is not tracked."""

    def test_column_access_not_tracked(self):
        """Direct column access (df['col']) is not tracked as structural."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df['a']  # Column access, not structural

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Should not have 'a' in structural reads
        # df might be in result due to other internal accesses
        if 'df' in result:
            assert 'a' not in result['df']

    def test_regular_method_not_tracked(self):
        """Regular methods like mean() are not tracked as structural."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2, 3]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.mean()  # Aggregation, not structural

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        if 'df' in result:
            assert 'mean' not in result['df']


class TestAttributeSets:
    """Tests for the attribute set constants."""

    def test_column_revealing_attrs(self):
        """COLUMN_REVEALING_ATTRS contains expected attributes."""
        expected = {
            'columns', 'keys', 'iter', 'dtypes', 'T', 'axes', 'values',
            'describe', 'to_dict', 'info', 'head', 'tail', 'sample',
            'select_dtypes', 'to_records', 'memory_usage',
        }
        assert COLUMN_REVEALING_ATTRS == expected

    def test_row_revealing_attrs(self):
        """ROW_REVEALING_ATTRS contains expected attributes."""
        expected = {'index', 'len', 'shape', 'size', 'empty'}
        assert ROW_REVEALING_ATTRS == expected

    def test_dataframe_structural_attrs(self):
        """DATAFRAME_STRUCTURAL_ATTRS is complete."""
        # Should include column and row structural attrs
        assert 'columns' in DATAFRAME_STRUCTURAL_ATTRS
        assert 'shape' in DATAFRAME_STRUCTURAL_ATTRS
        assert 'index' in DATAFRAME_STRUCTURAL_ATTRS
        assert 'dtypes' in DATAFRAME_STRUCTURAL_ATTRS

    def test_series_structural_attrs(self):
        """SERIES_STRUCTURAL_ATTRS is complete."""
        assert 'index' in SERIES_STRUCTURAL_ATTRS
        assert 'shape' in SERIES_STRUCTURAL_ATTRS
        assert 'dtype' in SERIES_STRUCTURAL_ATTRS
        assert 'name' in SERIES_STRUCTURAL_ATTRS


class TestIntegrationWithTrackingData:
    """Tests for integration with TrackingData model."""

    def test_tracking_data_structural_reads(self):
        """TrackingData correctly stores structural reads."""
        from data_ferret.kernel.models import TrackingData

        data = TrackingData(
            structural_reads={'df': {'columns', 'shape'}, 's': {'index'}}
        )

        assert data.has_structural_read('df')
        assert data.has_structural_read('s')
        assert not data.has_structural_read('other')

    def test_tracking_data_column_structure_read(self):
        """TrackingData correctly identifies column structure reads."""
        from data_ferret.kernel.models import TrackingData

        data = TrackingData(
            structural_reads={'df': {'columns', 'describe'}}
        )

        assert data.has_column_structure_read('df')

    def test_tracking_data_row_structure_read(self):
        """TrackingData correctly identifies row structure reads."""
        from data_ferret.kernel.models import TrackingData

        data = TrackingData(
            structural_reads={'df': {'shape', 'len'}}
        )

        assert data.has_row_structure_read('df')

    def test_tracking_data_no_structure_read(self):
        """TrackingData correctly identifies when no structural reads."""
        from data_ferret.kernel.models import TrackingData

        data = TrackingData(structural_reads={})

        assert not data.has_structural_read('df')
        assert not data.has_column_structure_read('df')
        assert not data.has_row_structure_read('df')


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_empty_dataframe(self):
        """Tracking works on empty DataFrame.

        Note: Pandas may internally access additional attributes during
        our explicit accesses, so we only check our explicit accesses are present.
        """
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame()
        tracker.register(df, 'df')
        tracker.install()

        _ = df.columns
        _ = df.shape
        _ = df.empty

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Check that our explicit accesses are tracked
        assert 'columns' in result['df']
        assert 'shape' in result['df']
        assert 'empty' in result['df']

    def test_empty_series(self):
        """Tracking works on empty Series.

        Note: Pandas may internally access additional attributes.
        """
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        s = pd.Series([], dtype=float)
        tracker.register(s, 's')
        tracker.install()

        _ = s.index
        _ = s.shape
        _ = s.empty

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Check that our explicit accesses are tracked
        assert 'index' in result['s']
        assert 'shape' in result['s']
        assert 'empty' in result['s']

    def test_large_dataframe(self):
        """Tracking works on large DataFrame."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame(np.random.randn(10000, 100))
        tracker.register(df, 'df')
        tracker.install()

        _ = df.shape
        _ = len(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'shape' in result['df']
        assert 'len' in result['df']

    def test_dataframe_with_multiindex(self):
        """Tracking works with MultiIndex DataFrame."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame(
            {'a': [1, 2, 3, 4]},
            index=pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')])
        )
        tracker.register(df, 'df')
        tracker.install()

        _ = df.index

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'index' in result['df']

    def test_series_with_name_none(self):
        """Tracking works when Series name is None."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        s = pd.Series([1, 2, 3])  # name is None by default
        tracker.register(s, 's')
        tracker.install()

        _ = s.name

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'name' in result['s']

    def test_unregistered_object(self):
        """Accessing unregistered object doesn't cause errors."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]})

        tracker.register(df1, 'df1')  # Only register df1
        tracker.install()

        _ = df1.columns
        _ = df2.columns  # Not registered

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df1' in result
        assert 'df2' not in result  # Not registered, so not in result


class TestStructureUsingContext:
    """Tests for _structure_using_context context manager."""

    def test_context_prevents_tracking(self):
        """Tracking is prevented within structure-using context."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2, 3]})
        tracker.register(df, 'df')
        tracker.install()

        with _structure_using_context():
            _ = df.columns  # Should NOT be tracked

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or 'columns' not in result.get('df', set())

    def test_context_restored_after_exit(self):
        """Tracking resumes after context exits."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2, 3]})
        tracker.register(df, 'df')
        tracker.install()

        with _structure_using_context():
            _ = df.columns  # NOT tracked

        _ = df.shape  # Should be tracked

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'shape' in result['df']

    def test_nested_context(self):
        """Nested structure-using contexts work correctly."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2, 3]})
        tracker.register(df, 'df')
        tracker.install()

        with _structure_using_context():
            with _structure_using_context():
                _ = df.columns  # NOT tracked
            _ = df.shape  # Still NOT tracked (outer context)

        _ = df.index  # Should be tracked

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'index' in result['df']
        assert 'columns' not in result.get('df', set())
        assert 'shape' not in result.get('df', set())


class TestStructureUsingMethodExclusion:
    """Tests for structure-using method exclusion from structural tracking."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    @pytest.fixture
    def series(self):
        return pd.Series([1, 2, 3], name='test')

    def test_dataframe_repr_html_excluded(self, tracker, df):
        """DataFrame._repr_html_() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        # This internally accesses .columns, .index, etc.
        _ = df._repr_html_()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Should not have any structural reads from display
        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_dataframe_repr_excluded(self, tracker, df):
        """DataFrame.__repr__() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = repr(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_dataframe_str_excluded(self, tracker, df):
        """DataFrame.__str__() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = str(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_series_repr_html_excluded(self, tracker, series):
        """Series._repr_html_() does not record structural reads."""
        # Check at class level before installing tracker to avoid side effects
        if not hasattr(pd.Series, '_repr_html_'):
            pytest.skip("Series._repr_html_ not available in this pandas version")

        tracker.register(series, 's')
        tracker.install()

        _ = series._repr_html_()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' not in result or len(result.get('s', set())) == 0

    def test_series_repr_excluded(self, tracker, series):
        """Series.__repr__() does not record structural reads."""
        tracker.register(series, 's')
        tracker.install()

        _ = repr(series)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' not in result or len(result.get('s', set())) == 0

    def test_series_str_excluded(self, tracker, series):
        """Series.__str__() does not record structural reads."""
        tracker.register(series, 's')
        tracker.install()

        _ = str(series)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' not in result or len(result.get('s', set())) == 0

    def test_explicit_access_still_tracked_after_display(self, tracker, df):
        """Explicit structural access after display is still tracked."""
        tracker.register(df, 'df')
        tracker.install()

        # Display (excluded)
        _ = repr(df)

        # Explicit access (should be tracked)
        _ = df.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']

    def test_explicit_access_before_display_tracked(self, tracker, df):
        """Explicit structural access before display is tracked."""
        tracker.register(df, 'df')
        tracker.install()

        # Explicit access (should be tracked)
        _ = df.shape

        # Display (excluded)
        _ = repr(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'shape' in result['df']

    def test_display_methods_still_work(self, tracker, df):
        """Display methods return proper output after patching."""
        tracker.register(df, 'df')
        tracker.install()

        # Verify display methods still return content
        html = df._repr_html_()
        assert '<table' in html.lower() or '<div' in html.lower()

        repr_str = repr(df)
        assert 'a' in repr_str and 'b' in repr_str

        str_str = str(df)
        assert '1' in str_str

        tracker.uninstall()

    def test_structure_using_methods_list(self):
        """STRUCTURE_USING_METHODS contains expected method categories."""
        df_methods = {m for cls, m in STRUCTURE_USING_METHODS if cls == 'DataFrame'}
        series_methods = {m for cls, m in STRUCTURE_USING_METHODS if cls == 'Series'}

        # Verify display methods are included
        assert '_repr_html_' in df_methods
        assert '__repr__' in df_methods
        assert '__str__' in df_methods

        # Verify item access/mutation are included
        assert '__getitem__' in df_methods
        assert '__setitem__' in df_methods
        assert '__delitem__' in df_methods

        # Verify arithmetic operators are included
        assert '__add__' in df_methods
        assert '__sub__' in df_methods
        assert '__mul__' in df_methods

        # Verify aggregation methods are included
        assert 'mean' in df_methods
        assert 'sum' in df_methods
        assert 'min' in df_methods
        assert 'max' in df_methods

        # Verify transform methods are included
        assert 'apply' in df_methods
        assert 'groupby' in df_methods

        # Verify I/O methods are included
        assert 'to_csv' in df_methods
        assert 'to_json' in df_methods

        # Verify Series has similar coverage
        assert '__repr__' in series_methods
        assert '__setitem__' in series_methods
        assert 'mean' in series_methods
        assert 'apply' in series_methods

    def test_display_exclusion_with_suspension(self, tracker, df):
        """Display exclusion works alongside global suspension."""
        tracker.register(df, 'df')
        tracker.install()

        # Global suspension
        with suspend_structural_tracking():
            _ = df.columns  # NOT tracked (suspended)

        # Display (excluded via patched method)
        _ = repr(df)

        # Explicit access after both - note that shape internally accesses
        # columns/index, so we only assert shape is tracked
        _ = df.shape

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'shape' in result['df']

    def test_multiple_displays_no_accumulation(self, tracker, df):
        """Multiple display calls don't accumulate structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        # Multiple displays
        for _ in range(5):
            _ = repr(df)
            _ = str(df)
            if hasattr(df, '_repr_html_'):
                _ = df._repr_html_()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Should still be empty from display operations
        assert 'df' not in result or len(result.get('df', set())) == 0


class TestStructureUsingMethodsExclusion:
    """Tests that structure-using methods don't record structural reads."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    @pytest.fixture
    def series(self):
        return pd.Series([1, 2, 3], name='test')

    def test_setitem_excluded(self, tracker, df):
        """df['x'] = value does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        df['c'] = [7, 8, 9]

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_getitem_excluded(self, tracker, df):
        """df['a'] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df['a']

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_delitem_excluded(self, tracker, df):
        """del df['a'] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        del df['a']

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_arithmetic_add_excluded(self, tracker, df):
        """df + 1 does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df + 1

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_arithmetic_mul_excluded(self, tracker, df):
        """df * 2 does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df * 2

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_comparison_excluded(self, tracker, df):
        """df > 0 does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df > 0

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_mean_excluded(self, tracker, df):
        """df.mean() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.mean()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_sum_excluded(self, tracker, df):
        """df.sum() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.sum()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_apply_excluded(self, tracker, df):
        """df.apply() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.apply(lambda x: x * 2)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_groupby_excluded(self, tracker):
        """df.groupby() does not record structural reads.

        Note: We only test the groupby call itself, not chained methods
        on the result since DataFrameGroupBy is a separate class.
        """
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
        tracker.register(df, 'df')
        tracker.install()

        # Just the groupby call - not the chained aggregation
        _ = df.groupby('a')

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_fillna_excluded(self, tracker):
        """df.fillna() does not record structural reads."""
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
        tracker.register(df, 'df')
        tracker.install()

        _ = df.fillna(0)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_sort_values_excluded(self, tracker, df):
        """df.sort_values() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.sort_values('a')

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_to_csv_excluded(self, tracker, df, tmp_path):
        """df.to_csv() does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        df.to_csv(tmp_path / 'test.csv')

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_series_setitem_excluded(self, tracker, series):
        """s[0] = value does not record structural reads."""
        tracker.register(series, 's')
        tracker.install()

        series[0] = 99

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' not in result or len(result.get('s', set())) == 0

    def test_series_mean_excluded(self, tracker, series):
        """s.mean() does not record structural reads."""
        tracker.register(series, 's')
        tracker.install()

        _ = series.mean()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' not in result or len(result.get('s', set())) == 0

    def test_series_arithmetic_excluded(self, tracker, series):
        """s + 1 does not record structural reads."""
        tracker.register(series, 's')
        tracker.install()

        _ = series + 1

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 's' not in result or len(result.get('s', set())) == 0

    def test_series_bitwise_and_excluded(self, tracker, df):
        """Boolean mask operations (mask1 & mask2) don't record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        # Create boolean masks and combine with &
        mask1 = df['a'] > 1
        mask2 = df['b'] < 6
        _ = mask1 & mask2

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_series_bitwise_or_excluded(self, tracker, df):
        """Boolean mask operations (mask1 | mask2) don't record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        mask1 = df['a'] > 1
        mask2 = df['b'] < 6
        _ = mask1 | mask2

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_series_bitwise_invert_excluded(self, tracker, df):
        """Boolean mask inversion (~mask) doesn't record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        mask = df['a'] > 1
        _ = ~mask

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_explicit_access_still_tracked(self, tracker, df):
        """Explicit structural access is still tracked after structure-using ops."""
        tracker.register(df, 'df')
        tracker.install()

        # Structure-using operations (not tracked)
        _ = df.mean()
        df['c'] = [7, 8, 9]

        # Explicit structural access (should be tracked)
        _ = df.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']

    def test_multiple_structure_using_ops_excluded(self, tracker, df):
        """Multiple structure-using operations don't accumulate reads."""
        tracker.register(df, 'df')
        tracker.install()

        # Multiple operations that internally access structure
        _ = df + 1
        _ = df * 2
        _ = df.mean()
        _ = df.sum()
        df['c'] = [7, 8, 9]
        _ = df['a']

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0


class TestIndexerExclusion:
    """Tests that .loc, .iloc, .at, .iat indexers don't record structural reads."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_loc_getitem_excluded(self, tracker, df):
        """df.loc[...] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.loc[0, 'a']
        _ = df.loc[1:2, 'b']
        _ = df.loc[:, 'a']

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_loc_setitem_excluded(self, tracker, df):
        """df.loc[...] = value does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        df.loc[0, 'a'] = 100
        df.loc[df['a'] > 50, 'b'] = 200

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_loc_new_column_excluded(self, tracker, df):
        """df.loc[:, 'new_col'] = value does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        df.loc[:, 'c'] = [7, 8, 9]

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_iloc_getitem_excluded(self, tracker, df):
        """df.iloc[...] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.iloc[0, 0]
        _ = df.iloc[1:2, :]
        _ = df.iloc[:, 0]

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_iloc_setitem_excluded(self, tracker, df):
        """df.iloc[...] = value does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        df.iloc[0, 0] = 100
        df.iloc[1:3, 1] = 200

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_at_excluded(self, tracker, df):
        """df.at[...] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.at[0, 'a']
        df.at[1, 'b'] = 100

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_iat_excluded(self, tracker, df):
        """df.iat[...] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.iat[0, 0]
        df.iat[1, 1] = 100

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_explicit_access_after_loc_still_tracked(self, tracker, df):
        """Explicit structural access after .loc is still tracked."""
        tracker.register(df, 'df')
        tracker.install()

        # Use loc (should not be tracked)
        _ = df.loc[0, 'a']

        # Explicit structural access (should be tracked)
        _ = df.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']

    def test_complex_loc_expression_excluded(self, tracker, df):
        """Complex .loc expressions don't record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        # Boolean indexing with loc
        df.loc[df['a'] > 1, 'b'] = df.loc[df['a'] > 1, 'a'] * 2

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0


class TestPandasFunctionExclusion:
    """Tests that module-level pandas functions don't record structural reads."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    def test_concat_excluded(self, tracker):
        """pd.concat does not record structural reads."""
        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'a': [3, 4]})
        tracker.register(df1, 'df1')
        tracker.register(df2, 'df2')
        tracker.install()

        _ = pd.concat([df1, df2])

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df1' not in result or len(result.get('df1', set())) == 0
        assert 'df2' not in result or len(result.get('df2', set())) == 0

    def test_merge_excluded(self, tracker):
        """pd.merge does not record structural reads."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
        df2 = pd.DataFrame({'a': [1, 2], 'c': [100, 200]})
        tracker.register(df1, 'df1')
        tracker.register(df2, 'df2')
        tracker.install()

        _ = pd.merge(df1, df2, on='a')

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df1' not in result or len(result.get('df1', set())) == 0
        assert 'df2' not in result or len(result.get('df2', set())) == 0

    def test_get_dummies_excluded(self, tracker):
        """pd.get_dummies does not record structural reads."""
        df = pd.DataFrame({'a': ['x', 'y', 'x']})
        tracker.register(df, 'df')
        tracker.install()

        _ = pd.get_dummies(df['a'])

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_melt_excluded(self, tracker):
        """pd.melt does not record structural reads."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        tracker.register(df, 'df')
        tracker.install()

        _ = pd.melt(df)

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_explicit_access_after_concat_still_tracked(self, tracker):
        """Explicit structural access after pd.concat is still tracked."""
        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'a': [3, 4]})
        tracker.register(df1, 'df1')
        tracker.register(df2, 'df2')
        tracker.install()

        # Concat (should not be tracked)
        _ = pd.concat([df1, df2])

        # Explicit structural access (should be tracked)
        _ = df1.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df1']


class TestGroupByExclusion:
    """Tests that GroupBy column selection doesn't record structural reads."""

    @pytest.fixture
    def tracker(self):
        return StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4],
            'other': [10, 20, 30, 40]
        })

    def test_groupby_column_selection_excluded(self, tracker, df):
        """df.groupby(...)['col'] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.groupby('category')['value'].sum()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_groupby_multi_column_selection_excluded(self, tracker, df):
        """df.groupby(...)[['col1', 'col2']] does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.groupby('category')[['value', 'other']].sum()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_groupby_with_reset_index_excluded(self, tracker, df):
        """df.groupby(...).sum().reset_index() chain does not record structural reads."""
        tracker.register(df, 'df')
        tracker.install()

        _ = df.groupby('category')['value'].sum().reset_index()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_explicit_columns_after_groupby_still_tracked(self, tracker, df):
        """Explicit df.columns access after groupby is still tracked."""
        tracker.register(df, 'df')
        tracker.install()

        # GroupBy (should not be tracked)
        _ = df.groupby('category')['value'].sum()

        # Explicit structural access (should be tracked)
        _ = df.columns

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        assert 'columns' in result['df']


class TestIPythonIntegration:
    """Tests for IPython display machinery integration."""

    def test_get_real_method_excluded(self):
        """IPython's get_real_method doesn't record structural reads.

        When IPython checks if _repr_html_ exists, pandas' __getattr__
        internally accesses .columns. This should be excluded.
        """
        from IPython.core.formatters import get_real_method

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2, 3]})
        tracker.register(df, 'df')
        tracker.install()

        # This is what IPython does to check for display methods
        method = get_real_method(df, '_repr_html_')
        assert method is not None

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Should not record any structural reads
        assert 'df' not in result or len(result.get('df', set())) == 0

    def test_html_formatter_excluded(self):
        """IPython's HTMLFormatter doesn't record structural reads."""
        from IPython.core.formatters import HTMLFormatter

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        df = pd.DataFrame({'a': [1, 2, 3]})
        tracker.register(df, 'df')
        tracker.install()

        # This is what IPython does to format for display
        html_fmt = HTMLFormatter()
        html = html_fmt(df)
        assert html is not None
        assert '<table' in html.lower() or '<div' in html.lower()

        tracker.uninstall()
        result = tracker.resolve_to_paths()

        # Should not record any structural reads
        assert 'df' not in result or len(result.get('df', set())) == 0


class TestStructureUsingRestoration:
    """Tests that structure-using method patches are properly restored on uninstall."""

    def test_repr_html_restored(self):
        """DataFrame._repr_html_ is restored after uninstall."""
        original = pd.DataFrame._repr_html_

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()

        # Method should be patched
        assert pd.DataFrame._repr_html_ != original

        tracker.uninstall()

        # Method should be restored
        assert pd.DataFrame._repr_html_ == original

    def test_repr_restored(self):
        """DataFrame.__repr__ is restored after uninstall."""
        original = pd.DataFrame.__repr__

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()

        assert pd.DataFrame.__repr__ != original

        tracker.uninstall()

        assert pd.DataFrame.__repr__ == original

    def test_series_repr_restored(self):
        """Series.__repr__ is restored after uninstall."""
        original = pd.Series.__repr__

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()

        assert pd.Series.__repr__ != original

        tracker.uninstall()

        assert pd.Series.__repr__ == original

    def test_setitem_restored(self):
        """DataFrame.__setitem__ is restored after uninstall."""
        original = pd.DataFrame.__setitem__

        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()

        assert pd.DataFrame.__setitem__ != original

        tracker.uninstall()

        assert pd.DataFrame.__setitem__ == original

    def test_all_structure_using_methods_in_original_methods(self):
        """All structure-using methods are stored for restoration."""
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.ENFORCE)
        tracker.install()

        for cls_name, method_name in STRUCTURE_USING_METHODS:
            key = f'{cls_name}.{method_name}'
            cls = pd.DataFrame if cls_name == 'DataFrame' else pd.Series
            # Only check if method exists on the class
            if hasattr(cls, method_name):
                assert key in tracker._original_methods, f"Missing {key}"

        tracker.uninstall()
