"""
Tests for kernel/tracking.py - TrackingDict and variable access tracking.

Tests cover:
- TrackingDict basic functionality (dict operations)
- Read/write tracking
- Reads-before-writes detection
- Context manager API (track_execution)
- get_tracking_data() method
- Column-level tracking integration
"""

import pytest
import pandas as pd
import numpy as np
from flowbook.kernel_support.tracking import TrackingDict
from flowbook.kernel_support.models import TrackingData


class TestTrackingDictBasics:
    """Tests for basic dict operations of TrackingDict."""

    def test_init_empty(self):
        """TrackingDict can be initialized empty."""
        td = TrackingDict()
        assert len(td) == 0

    def test_init_with_dict(self):
        """TrackingDict can be initialized with existing dict."""
        td = TrackingDict({"x": 1, "y": 2})
        assert td["x"] == 1
        assert td["y"] == 2
        assert len(td) == 2

    def test_setitem(self):
        """TrackingDict supports item assignment."""
        td = TrackingDict()
        td["x"] = 42
        assert td["x"] == 42

    def test_getitem(self):
        """TrackingDict supports item access."""
        td = TrackingDict({"x": 100})
        assert td["x"] == 100

    def test_delitem(self):
        """TrackingDict supports item deletion."""
        td = TrackingDict({"x": 1})
        del td["x"]
        assert "x" not in td

    def test_contains(self):
        """TrackingDict supports 'in' operator."""
        td = TrackingDict({"x": 1})
        assert "x" in td
        assert "y" not in td

    def test_len(self):
        """TrackingDict supports len()."""
        td = TrackingDict({"a": 1, "b": 2, "c": 3})
        assert len(td) == 3

    def test_keys(self):
        """TrackingDict supports keys()."""
        td = TrackingDict({"x": 1, "y": 2})
        assert set(td.keys()) == {"x", "y"}

    def test_values(self):
        """TrackingDict supports values()."""
        td = TrackingDict({"x": 1, "y": 2})
        assert set(td.values()) == {1, 2}

    def test_items(self):
        """TrackingDict supports items()."""
        td = TrackingDict({"x": 1, "y": 2})
        assert set(td.items()) == {("x", 1), ("y", 2)}

    def test_get(self):
        """TrackingDict supports get() with default."""
        td = TrackingDict({"x": 1})
        assert td.get("x") == 1
        assert td.get("y") is None
        assert td.get("y", 42) == 42

    def test_update(self):
        """TrackingDict supports update()."""
        td = TrackingDict({"x": 1})
        td.update({"y": 2, "z": 3})
        assert td["y"] == 2
        assert td["z"] == 3


class TestReadWriteTracking:
    """Tests for read/write tracking functionality."""

    def test_write_tracking(self):
        """Writes are tracked."""
        td = TrackingDict()
        td.reset_tracking()
        td["x"] = 1
        td["y"] = 2
        assert "x" in td.writes
        assert "y" in td.writes

    def test_read_before_write_tracking(self):
        """Reads before writes are tracked."""
        td = TrackingDict({"x": 1, "y": 2})
        td.reset_tracking()
        # Read x
        _ = td["x"]
        # Read y then write y
        _ = td["y"]
        td["y"] = 10
        # x was read but not written
        assert "x" in td.reads_before_writes
        # y was read then written - still counts as RBW
        assert "y" in td.reads_before_writes

    def test_write_then_read_not_rbw(self):
        """Write then read does not count as RBW."""
        td = TrackingDict()
        td.reset_tracking()
        td["x"] = 1  # Write first
        _ = td["x"]  # Then read
        # x was written before read, so not RBW
        assert "x" not in td.reads_before_writes
        assert "x" in td.writes

    def test_multiple_reads_single_rbw(self):
        """Multiple reads of same variable count as single RBW."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        _ = td["x"]
        _ = td["x"]
        _ = td["x"]
        assert td.reads_before_writes == {"x"}

    def test_reset_tracking_clears_state(self):
        """reset_tracking() clears all tracking state."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        td["y"] = 2
        _ = td["x"]
        assert "y" in td.writes
        assert "x" in td.reads_before_writes

        td.reset_tracking()
        assert len(td.writes) == 0
        assert len(td.reads_before_writes) == 0


class TestTrackExecutionContextManager:
    """Tests for track_execution() context manager."""

    def test_context_manager_resets_tracking(self):
        """Context manager resets tracking at start."""
        td = TrackingDict({"x": 1})
        # Create some tracking state
        td["y"] = 2
        _ = td["x"]

        # Context manager should reset
        with td.track_execution():
            pass

        # After context, tracking should have been reset
        data = td.get_tracking_data()
        assert data.reads_before_writes == set()
        assert data.writes == set()

    def test_context_manager_tracks_operations(self):
        """Context manager tracks operations inside it."""
        td = TrackingDict({"x": 1, "y": 2})

        with td.track_execution():
            _ = td["x"]  # RBW
            td["z"] = 3  # Write

        data = td.get_tracking_data()
        assert "x" in data.reads_before_writes
        assert "z" in data.writes

    def test_context_manager_exception_handling(self):
        """Context manager handles exceptions properly."""
        td = TrackingDict({"x": 1})

        with pytest.raises(ValueError):
            with td.track_execution():
                _ = td["x"]
                td["y"] = 2
                raise ValueError("test error")

        # Should still be able to get tracking data
        data = td.get_tracking_data()
        assert "x" in data.reads_before_writes
        assert "y" in data.writes

    def test_nested_context_managers(self):
        """Nested context managers work correctly."""
        td = TrackingDict({"a": 1, "b": 2})

        with td.track_execution():
            _ = td["a"]
            td["c"] = 3

            # Inner context resets tracking
            with td.track_execution():
                _ = td["b"]
                td["d"] = 4

            # After inner context, only inner operations tracked
            inner_data = td.get_tracking_data()
            assert "b" in inner_data.reads_before_writes
            assert "d" in inner_data.writes
            # Outer operations no longer tracked
            assert "a" not in inner_data.reads_before_writes
            assert "c" not in inner_data.writes


class TestGetTrackingData:
    """Tests for get_tracking_data() method."""

    def test_get_tracking_data_returns_model(self):
        """get_tracking_data returns TrackingData model."""
        td = TrackingDict()
        td.reset_tracking()
        data = td.get_tracking_data()
        assert isinstance(data, TrackingData)

    def test_get_tracking_data_sets(self):
        """get_tracking_data returns sets of variable names."""
        td = TrackingDict({"c": 1, "a": 2, "b": 3})
        td.reset_tracking()
        _ = td["c"]
        _ = td["a"]
        _ = td["b"]
        td["z"] = 10
        td["x"] = 20
        td["y"] = 30

        data = td.get_tracking_data()
        assert data.reads_before_writes == {"a", "b", "c"}
        assert data.writes == {"x", "y", "z"}

    def test_get_tracking_data_filters_private(self):
        """get_tracking_data filters private variables."""
        td = TrackingDict({"_private": 1, "public": 2})
        td.reset_tracking()
        _ = td["_private"]
        _ = td["public"]
        td["_another_private"] = 3
        td["another_public"] = 4

        data = td.get_tracking_data()
        # Private vars should be filtered
        assert "_private" not in data.reads_before_writes
        assert "_another_private" not in data.writes
        # Public vars should be included
        assert "public" in data.reads_before_writes
        assert "another_public" in data.writes

    def test_get_tracking_data_filters_modules(self):
        """get_tracking_data filters module objects."""
        import sys

        td = TrackingDict({"sys": sys, "x": 1})
        td.reset_tracking()
        # Access both
        _ = td["sys"]
        _ = td["x"]

        data = td.get_tracking_data()
        # Module should be filtered
        assert "sys" not in data.reads_before_writes
        # Regular var should be included
        assert "x" in data.reads_before_writes


class TestColumnTracking:
    """Tests for DataFrame column-level tracking."""

    def test_column_tracking_basic(self):
        """Basic column tracking works."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        td = TrackingDict({"df": df})

        with td.track_execution():
            # Read column 'a'
            _ = td["df"]["a"]
            # Write column 'c'
            td["df"]["c"] = td["df"]["a"] * 2

        data = td.get_tracking_data()
        # Check column-level tracking
        assert "df" in data.column_reads_before_writes
        assert "a" in data.column_reads_before_writes["df"]

    def test_column_tracking_write_then_read(self):
        """Column written then read is not RBW."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        td = TrackingDict({"df": df})

        with td.track_execution():
            # Write column 'b' first
            td["df"]["b"] = [4, 5, 6]
            # Then read it
            _ = td["df"]["b"]

        data = td.get_tracking_data()
        # 'b' should not be in RBW since it was written first
        if "df" in data.column_reads_before_writes:
            assert "b" not in data.column_reads_before_writes["df"]

    def test_column_tracking_read_then_write_same_column(self):
        """Column read then written IS in RBW (common pattern: df['x'] = df['x'].transform())."""
        df = pd.DataFrame({"prediction": [1.1, 2.2, 3.3]})
        td = TrackingDict({"df": df})

        with td.track_execution():
            # Read then write same column - common pattern like:
            # df['prediction'] = df['prediction'].astype(float).round().astype(int)
            td["df"]["prediction"] = td["df"]["prediction"].astype(float).round().astype(int)

        data = td.get_tracking_data()
        # 'prediction' should be in BOTH RBW (because it was read) and writes (because it was written)
        assert "df" in data.column_reads_before_writes
        assert "prediction" in data.column_reads_before_writes["df"], \
            "Column read before write should be in column_reads_before_writes"
        assert "df" in data.column_writes
        assert "prediction" in data.column_writes["df"]

    def test_column_tracking_multiple_dataframes(self):
        """Column tracking works with multiple DataFrames."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"y": [3, 4]})
        td = TrackingDict({"df1": df1, "df2": df2})

        with td.track_execution():
            _ = td["df1"]["x"]
            _ = td["df2"]["y"]

        data = td.get_tracking_data()
        assert "df1" in data.column_reads_before_writes
        assert "df2" in data.column_reads_before_writes

    def test_column_tracking_nested_dataframe(self):
        """Column tracking works with nested DataFrames."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        td = TrackingDict({"data": {"train": df}})

        with td.track_execution():
            _ = td["data"]["train"]["col"]

        data = td.get_tracking_data()
        # Should track the nested path
        assert any("train" in k for k in data.column_reads_before_writes.keys())


class TestTrackingEdgeCases:
    """Tests for edge cases in tracking."""

    def test_empty_namespace(self):
        """Tracking works with empty namespace."""
        td = TrackingDict()
        with td.track_execution():
            td["x"] = 1

        data = td.get_tracking_data()
        assert "x" in data.writes

    def test_large_namespace(self):
        """Tracking works with large namespace."""
        initial = {f"var_{i}": i for i in range(100)}
        td = TrackingDict(initial)

        with td.track_execution():
            # Read first 10
            for i in range(10):
                _ = td[f"var_{i}"]
            # Write next 10
            for i in range(10, 20):
                td[f"new_{i}"] = i * 2

        data = td.get_tracking_data()
        assert len(data.reads_before_writes) == 10
        assert len(data.writes) == 10

    def test_overwrite_existing_variable(self):
        """Overwriting existing variable is tracked as write."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        td["x"] = 100

        assert "x" in td.writes
        assert "x" not in td.reads_before_writes

    def test_read_then_overwrite(self):
        """Reading then overwriting is RBW."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        old_val = td["x"]
        td["x"] = old_val + 1

        assert "x" in td.reads_before_writes
        assert "x" in td.writes

    def test_delete_not_tracked(self):
        """Deletion is not tracked as write or read."""
        td = TrackingDict({"x": 1, "y": 2})
        td.reset_tracking()
        del td["x"]

        # Deletion should not appear in tracking
        assert "x" not in td.writes
        assert "x" not in td.reads_before_writes

    def test_numpy_array_in_namespace(self):
        """Tracking works with numpy arrays."""
        arr = np.array([1, 2, 3])
        td = TrackingDict({"arr": arr})

        with td.track_execution():
            _ = td["arr"]
            td["new_arr"] = td["arr"] * 2

        data = td.get_tracking_data()
        assert "arr" in data.reads_before_writes
        assert "new_arr" in data.writes

    def test_mixed_types_in_namespace(self):
        """Tracking works with mixed types."""
        td = TrackingDict(
            {
                "int_var": 42,
                "str_var": "hello",
                "list_var": [1, 2, 3],
                "dict_var": {"a": 1},
                "df_var": pd.DataFrame({"x": [1]}),
            }
        )

        with td.track_execution():
            _ = td["int_var"]
            _ = td["str_var"]
            td["new_var"] = td["list_var"]

        data = td.get_tracking_data()
        assert "int_var" in data.reads_before_writes
        assert "str_var" in data.reads_before_writes
        assert "list_var" in data.reads_before_writes
        assert "new_var" in data.writes


class TestColumnRBWProperty:
    """Tests for column_rbw property."""

    def test_column_rbw_empty(self):
        """column_rbw is empty initially."""
        td = TrackingDict()
        td.reset_tracking()
        assert td.column_reads_before_writes == {}

    def test_column_writes_empty(self):
        """column_writes is empty initially."""
        td = TrackingDict()
        td.reset_tracking()
        assert td.column_writes == {}

    def test_column_reads_exclude_written_variables(self):
        """Column reads from variables written in the cell are excluded.

        If a variable is created in the cell (written), subsequent column reads
        from it should NOT be in column_reads_before_writes.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        td = TrackingDict({"df": df})

        with td.track_execution():
            # Read from existing df - should be tracked
            _ = td["df"]["x"]

            # Create new DataFrame - this is a write
            td["new_df"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

            # Read from new_df - should NOT be tracked (variable was written first)
            _ = td["new_df"]["a"]

        data = td.get_tracking_data()

        # df.x should be in column_reads
        assert "df" in data.column_reads_before_writes
        assert "x" in data.column_reads_before_writes["df"]

        # new_df should NOT be in column_reads (it was written)
        assert "new_df" not in data.column_reads_before_writes

    def test_groupby_merge_pattern_excludes_intermediate_df(self):
        """The groupby().reset_index() then merge() pattern works correctly.

        When a DataFrame is created via groupby and then used in a merge,
        column reads from the intermediate DataFrame should not be tracked.
        """
        df = pd.DataFrame({
            "year": [2020, 2020, 2021, 2021],
            "country": ["US", "CA", "US", "CA"],
            "num_sold": [100, 200, 150, 250]
        })
        td = TrackingDict({"df": df})

        with td.track_execution():
            # Create total_per_day via groupby (WRITE)
            td["total_per_day"] = td["df"].groupby("year")["num_sold"].sum().reset_index()

            # Create grouped_data (WRITE)
            td["grouped_data"] = td["df"].groupby(["year", "country"])["num_sold"].sum().reset_index()

            # Merge uses total_per_day - reads 'year' column but total_per_day was written
            td["grouped_data"] = td["grouped_data"].merge(
                td["total_per_day"],
                on=["year"],
                suffixes=["", "_total"]
            )

        data = td.get_tracking_data()

        # total_per_day should NOT be in column_reads (it was created in this cell)
        assert "total_per_day" not in data.column_reads_before_writes

        # grouped_data should NOT be in column_reads (it was created in this cell)
        assert "grouped_data" not in data.column_reads_before_writes

        # df SHOULD be in column_reads (it existed before the cell)
        assert "df" in data.column_reads_before_writes
        assert "year" in data.column_reads_before_writes["df"]
        assert "num_sold" in data.column_reads_before_writes["df"]

    def test_structural_reads_exclude_written_variables(self):
        """Structural reads from variables written in the cell are excluded."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        td = TrackingDict({"df": df})

        with td.track_execution():
            # Create new DataFrame
            td["new_df"] = pd.DataFrame({"a": [1, 2]})

            # Structural read from new_df - should NOT be tracked
            _ = td["new_df"].columns

        data = td.get_tracking_data()

        # new_df should NOT be in structural_reads (it was written)
        assert "new_df" not in data.structural_reads


class TestStartStopColumnTracking:
    """Tests for manual column tracking control."""

    def test_start_stop_column_tracking(self):
        """Manual start/stop column tracking works."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        td = TrackingDict({"df": df})
        td.reset_tracking()

        td.start_column_tracking()
        _ = td["df"]["a"]
        td.stop_column_tracking()

        # Should have tracked the column access
        assert "df" in td.column_reads_before_writes

    def test_stop_without_start(self):
        """Stopping without starting doesn't crash."""
        td = TrackingDict()
        td.stop_column_tracking()  # Should not raise

    def test_double_start(self):
        """Double start doesn't crash."""
        df = pd.DataFrame({"a": [1]})
        td = TrackingDict({"df": df})
        td.reset_tracking()

        td.start_column_tracking()
        td.start_column_tracking()  # Should not raise
        td.stop_column_tracking()

    def test_double_stop(self):
        """Double stop doesn't crash."""
        td = TrackingDict()
        td.reset_tracking()
        td.start_column_tracking()
        td.stop_column_tracking()
        td.stop_column_tracking()  # Should not raise

    def test_multiple_tracker_instances_no_recursion(self):
        """Multiple TrackingDict instances don't cause patch recursion.

        When multiple TrackingDict instances are created, they should share
        the same patches via class-level tracking, not double-patch and
        cause infinite recursion.
        """
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Create first tracker and install patches
        td1 = TrackingDict({"df": df})
        td1.start_column_tracking()
        _ = td1["df"]["a"]  # This should work
        td1.stop_column_tracking()

        # Create second tracker - should reuse existing patches
        td2 = TrackingDict({"df": df})
        td2.start_column_tracking()
        _ = td2["df"]["a"]  # This should NOT cause infinite recursion
        td2.stop_column_tracking()

        # Verify tracking worked for both
        assert "df" in td1.column_reads_before_writes
        assert "df" in td2.column_reads_before_writes


class TestSuspendedContextManager:
    """Tests for TrackingDict.suspended() context manager."""

    def test_suspended_prevents_tracking(self):
        """suspended() prevents reads and writes from being tracked."""
        td = TrackingDict({"x": 1, "y": 2})
        td.reset_tracking()
        td._tracking_enabled = True

        with td.suspended():
            _ = td["x"]  # Read - should not be tracked
            td["z"] = 3  # Write - should not be tracked

        assert "x" not in td.reads_before_writes
        assert "z" not in td.writes

    def test_suspended_restores_state(self):
        """suspended() restores tracking state after exit."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        td._tracking_enabled = True

        with td.suspended():
            pass

        # Tracking should be restored
        _ = td["x"]
        assert "x" in td.reads_before_writes

    def test_suspended_with_exception(self):
        """suspended() restores state even if exception occurs."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        td._tracking_enabled = True

        try:
            with td.suspended():
                raise ValueError("test")
        except ValueError:
            pass

        # Tracking should be restored despite exception
        _ = td["x"]
        assert "x" in td.reads_before_writes

    def test_suspended_when_already_disabled(self):
        """suspended() works correctly when tracking already disabled."""
        td = TrackingDict({"x": 1})
        td.reset_tracking()
        td._tracking_enabled = False

        with td.suspended():
            _ = td["x"]

        # Should still be disabled after exiting
        assert not td._tracking_enabled

    def test_nested_suspended(self):
        """Nested suspended() contexts work correctly."""
        td = TrackingDict({"x": 1, "y": 2, "z": 3})
        td.reset_tracking()
        td._tracking_enabled = True

        with td.suspended():
            _ = td["x"]  # Not tracked
            with td.suspended():
                _ = td["y"]  # Not tracked
            _ = td["z"]  # Still not tracked (outer context)

        assert "x" not in td.reads_before_writes
        assert "y" not in td.reads_before_writes
        assert "z" not in td.reads_before_writes

        # Verify tracking is re-enabled after both contexts exit
        _ = td["x"]
        assert "x" in td.reads_before_writes
