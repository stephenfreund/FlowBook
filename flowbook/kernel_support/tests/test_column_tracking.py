"""
Tests for column-level DataFrame dependency tracking.
"""

import pandas as pd
import pytest
from typing import Dict, Set

from flowbook.kernel_support.column_tracking import (
    ColumnAccessTracker,
    walk_dataframes,
    _walk_object_attrs,
)


class TestColumnAccessTracker:
    """Tests for ColumnAccessTracker class."""

    def test_record_column_read(self):
        """Test basic column read tracking."""
        tracker = ColumnAccessTracker()
        tracker.record_read(123, ["a", "b"])

        result = tracker.resolve_to_paths()
        # No path registered, so nothing returned
        assert result == {}

        # Register the df
        tracker.register_df(pd.DataFrame(), "df")
        # Need to use the actual id, but for this test we use a mock id
        # Let's test with proper flow

    def test_basic_column_read_with_registration(self):
        """Test column read tracking with proper registration."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.record_read(id(df), ["a"])

        result = tracker.resolve_to_paths()
        assert "df" in result
        assert result["df"] == {"a"}

    def test_column_write_blocks_rbw(self):
        """Test that writing a column before reading prevents RBW tracking."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")

        # Write to 'a' first
        tracker.record_write(id(df), ["a"])
        # Then read 'a' and 'b'
        tracker.record_read(id(df), ["a", "b"])

        result = tracker.resolve_to_paths()
        assert "df" in result
        # 'a' was written first, so only 'b' is RBW
        assert result["df"] == {"b"}

    def test_multi_column_access(self):
        """Test tracking multiple column accesses."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")

        tracker.record_read(id(df), ["a", "b", "c"])

        result = tracker.resolve_to_paths()
        assert result["df"] == {"a", "b", "c"}

    def test_reset_clears_all(self):
        """Test that reset clears all tracking data."""
        df = pd.DataFrame({"a": [1]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.record_read(id(df), ["a"])

        tracker.reset()

        result = tracker.resolve_to_paths()
        assert result == {}

    def test_multiple_dataframes(self):
        """Test tracking multiple DataFrames."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df1, "df1")
        tracker.register_df(df2, "df2")

        tracker.record_read(id(df1), ["a"])
        tracker.record_read(id(df2), ["b"])

        result = tracker.resolve_to_paths()
        assert result["df1"] == {"a"}
        assert result["df2"] == {"b"}

    def test_install_uninstall(self):
        """Test that install/uninstall properly patches and restores methods."""
        tracker = ColumnAccessTracker()
        original_getitem = pd.DataFrame.__getitem__

        tracker.install()
        assert pd.DataFrame.__getitem__ != original_getitem

        tracker.uninstall()
        assert pd.DataFrame.__getitem__ == original_getitem

    def test_tracking_via_patched_getitem(self):
        """Test that patched __getitem__ tracks column access."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()

        try:
            # Access a column - this should be tracked
            _ = df["a"]

            result = tracker.resolve_to_paths()
            assert "df" in result
            assert "a" in result["df"]
        finally:
            tracker.uninstall()

    def test_tracking_via_patched_setitem(self):
        """Test that patched __setitem__ tracks column writes."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()

        try:
            # Write to a column first
            df["c"] = [5, 6]
            # Then read it
            _ = df["c"]

            result = tracker.resolve_to_paths()
            # 'c' was written first, so not RBW
            assert "c" not in result.get("df", set())
        finally:
            tracker.uninstall()

    def test_tracking_multi_column_getitem(self):
        """Test tracking multi-column access with list key."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()

        try:
            _ = df[["a", "b"]]

            result = tracker.resolve_to_paths()
            assert "df" in result
            assert result["df"] == {"a", "b"}
        finally:
            tracker.uninstall()

    def test_merge_tracking_on(self):
        """Test that merge tracks 'on' columns from both DataFrames."""
        df1 = pd.DataFrame({"key": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"key": [1, 2], "b": [100, 200]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df1, "df1")
        tracker.register_df(df2, "df2")
        tracker.install()
        try:
            _ = df1.merge(df2, on="key")
            result = tracker.resolve_to_paths()
            assert "key" in result.get("df1", set())
            assert "key" in result.get("df2", set())
        finally:
            tracker.uninstall()

    def test_merge_tracking_left_right_on(self):
        """Test that merge tracks left_on/right_on columns."""
        df1 = pd.DataFrame({"left_key": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"right_key": [1, 2], "b": [100, 200]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df1, "df1")
        tracker.register_df(df2, "df2")
        tracker.install()
        try:
            _ = df1.merge(df2, left_on="left_key", right_on="right_key")
            result = tracker.resolve_to_paths()
            assert "left_key" in result.get("df1", set())
            assert "right_key" in result.get("df2", set())
        finally:
            tracker.uninstall()

    def test_groupby_column_access(self):
        """Test that groupby column access is tracked."""
        df = pd.DataFrame({"key": [1, 1, 2], "value": [10, 20, 30]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()
        try:
            _ = df.groupby("key")["value"].sum()
            result = tracker.resolve_to_paths()
            assert "key" in result.get("df", set())
            assert "value" in result.get("df", set())
        finally:
            tracker.uninstall()

    def test_groupby_multi_column_access(self):
        """Test that groupby with multi-column access is tracked."""
        df = pd.DataFrame({"key": [1, 1, 2], "a": [10, 20, 30], "b": [1, 2, 3]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()
        try:
            _ = df.groupby("key")[["a", "b"]].sum()
            result = tracker.resolve_to_paths()
            assert "key" in result.get("df", set())
            assert "a" in result.get("df", set())
            assert "b" in result.get("df", set())
        finally:
            tracker.uninstall()

    def test_sort_values_tracking(self):
        """Test that sort_values tracks by columns."""
        df = pd.DataFrame({"a": [3, 1, 2], "b": [1, 2, 3]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()
        try:
            _ = df.sort_values("a")
            result = tracker.resolve_to_paths()
            assert "a" in result.get("df", set())
        finally:
            tracker.uninstall()

    def test_sort_values_multi_column_tracking(self):
        """Test that sort_values tracks multiple by columns."""
        df = pd.DataFrame({"a": [3, 1, 2], "b": [1, 2, 3], "c": [4, 5, 6]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()
        try:
            _ = df.sort_values(["a", "b"])
            result = tracker.resolve_to_paths()
            assert "a" in result.get("df", set())
            assert "b" in result.get("df", set())
        finally:
            tracker.uninstall()

    def test_drop_duplicates_tracking(self):
        """Test that drop_duplicates tracks subset columns."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()
        try:
            _ = df.drop_duplicates(subset=["a"])
            result = tracker.resolve_to_paths()
            assert "a" in result.get("df", set())
        finally:
            tracker.uninstall()

    def test_drop_duplicates_no_subset(self):
        """Test that drop_duplicates without subset doesn't track columns."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()
        try:
            _ = df.drop_duplicates()
            result = tracker.resolve_to_paths()
            # Without subset, no specific columns are tracked
            assert result.get("df", set()) == set()
        finally:
            tracker.uninstall()


class TestWalkDataframes:
    """Tests for walk_dataframes function."""

    def test_top_level_dataframe(self):
        """Test finding top-level DataFrame."""
        df = pd.DataFrame({"a": [1]})
        namespace = {"df": df}

        found = list(walk_dataframes(namespace))
        assert len(found) == 1
        assert found[0] == ("df", df)

    def test_dataframe_in_dict(self):
        """Test finding DataFrame nested in dict."""
        df = pd.DataFrame({"a": [1]})
        namespace = {"data": {"train": df}}

        found = list(walk_dataframes(namespace))
        assert len(found) == 1
        path, found_df = found[0]
        assert path == "data['train']"
        assert found_df is df

    def test_dataframe_in_list(self):
        """Test finding DataFrame in list."""
        df = pd.DataFrame({"a": [1]})
        namespace = {"datasets": [df]}

        found = list(walk_dataframes(namespace))
        assert len(found) == 1
        path, found_df = found[0]
        assert path == "datasets[0]"
        assert found_df is df

    def test_dataframe_in_object(self):
        """Test finding DataFrame as object attribute."""
        df = pd.DataFrame({"a": [1]})

        class Container:
            pass

        obj = Container()
        obj.df = df
        namespace = {"obj": obj}

        found = list(walk_dataframes(namespace))
        assert len(found) == 1
        path, found_df = found[0]
        assert path == "obj.df"
        assert found_df is df

    def test_multiple_dataframes(self):
        """Test finding multiple DataFrames at different levels."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        df3 = pd.DataFrame({"c": [3]})

        namespace = {
            "df1": df1,
            "data": {"train": df2},
            "datasets": [df3],
        }

        found = dict(walk_dataframes(namespace))
        assert len(found) == 3
        assert found["df1"] is df1
        assert found["data['train']"] is df2
        assert found["datasets[0]"] is df3

    def test_skips_private_keys(self):
        """Test that private keys are skipped."""
        df = pd.DataFrame({"a": [1]})
        namespace = {"_private": df, "public": df}

        found = dict(walk_dataframes(namespace))
        assert len(found) == 1
        assert "public" in found
        assert "_private" not in found

    def test_skips_modules(self):
        """Test that module entries are skipped."""
        import numpy as np

        df = pd.DataFrame({"a": [1]})
        # Include a module in the namespace - should be skipped
        namespace = {"np": np, "pd": pd, "df": df}

        found = dict(walk_dataframes(namespace))
        # Should only find the DataFrame, not walk into modules
        assert len(found) == 1
        assert "df" in found
        # No paths starting with module names
        assert not any(path.startswith("np") for path in found)
        assert not any(path.startswith("pd") for path in found)

    def test_skips_modules_in_object_attrs(self):
        """Test that module attributes on objects are skipped."""
        import numpy as np

        df = pd.DataFrame({"a": [1]})

        class Container:
            pass

        c = Container()
        c.np = np  # Module attribute
        c.df = df  # DataFrame attribute

        namespace = {"container": c}
        found = dict(walk_dataframes(namespace))
        assert len(found) == 1
        assert "container.df" in found
        # Should not have walked into numpy module
        assert not any("np" in path for path in found)

    def test_cycle_detection(self):
        """Test that cycles are handled."""
        df = pd.DataFrame({"a": [1]})
        d = {"df": df}
        d["self"] = d  # Create cycle

        # Should not infinite loop
        found = list(walk_dataframes(d))
        assert len(found) == 1

    def test_deeply_nested(self):
        """Test deeply nested structure."""
        df = pd.DataFrame({"a": [1]})

        class Level1:
            pass

        l1 = Level1()
        l1.data = {"datasets": [df]}
        namespace = {"container": l1}

        found = dict(walk_dataframes(namespace))
        assert len(found) == 1
        path = list(found.keys())[0]
        assert "container" in path
        assert "data" in path
        assert "datasets" in path


class TestDiffIntegration:
    """Integration tests for Diff class with column_rbw."""

    def test_diff_with_column_rbw(self):
        """Test that Diff respects column_rbw."""
        from flowbook.kernel_support.diff import Diff

        df_a = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df_b = pd.DataFrame({"a": [1], "b": [999], "c": [3]})  # 'b' changed

        # Only compare column 'a' (which hasn't changed)
        differ = Diff(use_leq=True, column_rbw={"df": {"a"}})
        result = differ.diff({"df": df_a}, {"df": df_b})

        # Should pass - only 'a' is compared, and it's unchanged
        assert len(result.differences) == 0

    def test_diff_without_column_rbw(self):
        """Test that Diff without column_rbw compares all columns."""
        from flowbook.kernel_support.diff import Diff

        df_a = pd.DataFrame({"a": [1], "b": [2]})
        df_b = pd.DataFrame({"a": [1], "b": [999]})

        # No column_rbw, should compare all columns
        differ = Diff(use_leq=True)
        result = differ.diff({"df": df_a}, {"df": df_b})

        # Should fail - 'b' changed
        assert len(result.differences) > 0

    def test_diff_with_deleted_rbw_column(self):
        """Test that Diff catches deleted RBW columns."""
        from flowbook.kernel_support.diff import Diff

        df_a = pd.DataFrame({"a": [1], "b": [2]})
        df_b = pd.DataFrame({"a": [1]})  # 'b' deleted

        # Column 'b' was RBW but got deleted
        differ = Diff(use_leq=True, column_rbw={"df": {"a", "b"}})
        result = differ.diff({"df": df_a}, {"df": df_b})

        # Should fail - RBW column 'b' was deleted
        assert len(result.differences) > 0

    def test_diff_allows_new_columns(self):
        """Test that new columns are allowed in leq mode."""
        from flowbook.kernel_support.diff import Diff

        df_a = pd.DataFrame({"a": [1]})
        df_b = pd.DataFrame({"a": [1], "b": [2]})  # 'b' added

        # Only 'a' is RBW
        differ = Diff(use_leq=True, column_rbw={"df": {"a"}})
        result = differ.diff({"df": df_a}, {"df": df_b})

        # Should pass - 'a' unchanged, 'b' is new (allowed)
        assert len(result.differences) == 0

    def test_diff_write_only_dataframe_skips_comparison(self):
        """Test that write-only DataFrame (empty column_rbw) skips comparison.

        When a cell only writes to a DataFrame without reading any columns,
        the column_rbw entry will have an empty set. In this case, the
        DataFrame should not be compared at all, even if index/shape changed.
        """
        from flowbook.kernel_support.diff import Diff

        # Pre-state: empty DataFrame
        df_a = pd.DataFrame()
        # Post-state: DataFrame with data (different shape and index)
        df_b = pd.DataFrame({"b": [1, 2, 4]})

        # Empty set means no columns were read-before-write
        differ = Diff(use_leq=True, column_rbw={"df": set()})
        result = differ.diff({"df": df_a}, {"df": df_b})

        # Should pass - no columns were read, so nothing needs to match
        assert len(result.differences) == 0

    def test_diff_write_only_tracker_includes_empty_set(self):
        """Test that write-only DataFrames get empty set in column_rbw."""
        tracker = ColumnAccessTracker()
        df = pd.DataFrame({"a": [1]})
        tracker.register_df(df, "df")

        # Only write, no reads
        tracker.record_write(id(df), ["b"])

        result = tracker.resolve_to_paths()

        # "df" should be in result with empty set (write-only)
        assert "df" in result
        assert result["df"] == set()

    def test_resolve_writes_to_paths(self):
        """Test that resolve_writes_to_paths returns column writes."""
        tracker = ColumnAccessTracker()
        df = pd.DataFrame({"a": [1]})
        tracker.register_df(df, "df")

        # Write columns
        tracker.record_write(id(df), ["b", "c"])

        result = tracker.resolve_writes_to_paths()

        assert "df" in result
        assert result["df"] == {"b", "c"}

    def test_column_writes_via_tracking_dict(self):
        """Test that TrackingDict.column_writes returns write tracking data."""
        from flowbook.kernel_support.tracking import TrackingDict

        df = pd.DataFrame({"a": [1, 2]})
        tracking_dict = TrackingDict({"df": df})

        tracking_dict.start_column_tracking()
        try:
            # Write a column
            retrieved_df = tracking_dict["df"]
            retrieved_df["b"] = [3, 4]
        finally:
            tracking_dict.stop_column_tracking()

        result = tracking_dict.column_writes
        assert "df" in result
        assert "b" in result["df"]


class TestTrackingDictIntegration:
    """Integration tests for TrackingDict with column tracking."""

    def test_column_rbw_property(self):
        """Test that TrackingDict.column_rbw returns column tracking data."""
        from flowbook.kernel_support.tracking import TrackingDict

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tracking_dict = TrackingDict({"df": df})

        # Start tracking
        tracking_dict.start_column_tracking()

        try:
            # Access a column
            retrieved_df = tracking_dict["df"]
            _ = retrieved_df["a"]
        finally:
            tracking_dict.stop_column_tracking()

        result = tracking_dict.column_reads_before_writes
        assert "df" in result
        assert "a" in result["df"]

    def test_column_tracking_reset(self):
        """Test that reset_tracking clears column tracking."""
        from flowbook.kernel_support.tracking import TrackingDict

        df = pd.DataFrame({"a": [1]})
        tracking_dict = TrackingDict({"df": df})

        tracking_dict.start_column_tracking()
        try:
            _ = tracking_dict["df"]["a"]
        finally:
            tracking_dict.stop_column_tracking()

        # Should have data
        assert len(tracking_dict.column_reads_before_writes) > 0

        # Reset
        tracking_dict.reset_tracking()

        # Should be empty
        assert len(tracking_dict.column_reads_before_writes) == 0


class TestDeepcopyIntegration:
    """Tests for deepcopy not triggering column tracking patches."""

    def test_deepcopy_does_not_trigger_column_tracking(self):
        """Regression test: deepcopy should not record column reads.

        This tests the fix for a bug where _deepcopy_dataframe() used df[col]
        which triggered the patched __getitem__, causing all columns to be
        recorded as reads during checkpoint creation.
        """
        import numpy as np
        from flowbook.kernel_support import deepcopy as flowbook_deepcopy

        # Create DataFrame with multiple columns
        df = pd.DataFrame()
        df["a"] = np.array([1, 2, 4])
        df["b"] = np.array([1, 2, 4])

        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()

        try:
            # Simulate checkpoint: deepcopy while patches are installed
            _ = flowbook_deepcopy.deepcopy(df)

            # No columns should have been tracked during deepcopy
            result = tracker.resolve_to_paths()
            assert result == {}, f"Deepcopy triggered tracking: {result}"

        finally:
            tracker.uninstall()

    def test_deepcopy_with_object_columns_does_not_trigger_tracking(self):
        """Ensure deepcopy of object-dtype columns doesn't trigger tracking."""
        from flowbook.kernel_support import deepcopy as flowbook_deepcopy

        # DataFrame with object columns (requires special handling in deepcopy)
        df = pd.DataFrame(
            {
                "obj_col": [{"x": 1}, {"x": 2}],
                "str_col": ["a", "b"],
            }
        )

        tracker = ColumnAccessTracker()
        tracker.register_df(df, "df")
        tracker.install()

        try:
            # Deepcopy should iterate columns without triggering patches
            _ = flowbook_deepcopy.deepcopy(df)

            result = tracker.resolve_to_paths()
            assert result == {}, f"Deepcopy triggered tracking: {result}"

        finally:
            tracker.uninstall()
