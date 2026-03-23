"""
Tests for TrackingData conversion methods and change_detector module.

These tests verify that the converters correctly bridge the existing
TrackingData/MemoryCheckpointDiffResult structures to the new typed AccessEvent/Change
hierarchies.
"""

import pytest

from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.types import CompoundDiff, MemoryCheckpointDiffResult, ValueComparison

from flowbook.kernel.access_events import ColumnRead, ColumnWrite, StructuralRead, VariableRead
from flowbook.kernel.change_detector import (
    _analyze_column_change,
    _extract_column_name,
    _parse_column_structural_change,
    _parse_row_change,
    detect_changes,
    get_changed_variables,
    has_any_changes,
)
from flowbook.kernel.changes import (
    ColumnAdded,
    ColumnModified,
    ColumnRemoved,
    RowsAdded,
    RowsRemoved,
    ValueChanged,
)


# =============================================================================
# Test TrackingData conversion methods
# =============================================================================


class TestToAccessEvents:
    """Tests for TrackingData.to_access_events() method."""

    def test_empty_tracking(self):
        """Empty TrackingData produces empty list."""
        tracking = TrackingData()
        events = tracking.to_access_events()
        assert events == []

    def test_column_reads(self):
        """Column reads are converted to ColumnRead events."""
        tracking = TrackingData(
            column_reads_before_writes={"df": {"price", "quantity"}}
        )
        events = tracking.to_access_events()

        column_reads = [e for e in events if isinstance(e, ColumnRead)]
        assert len(column_reads) == 2
        assert ColumnRead(variable="df", column="price") in column_reads
        assert ColumnRead(variable="df", column="quantity") in column_reads

    def test_column_writes(self):
        """Column writes are converted to ColumnWrite events."""
        tracking = TrackingData(column_writes={"df": {"total", "tax"}})
        events = tracking.to_access_events()

        column_writes = [e for e in events if isinstance(e, ColumnWrite)]
        assert len(column_writes) == 2
        assert ColumnWrite(variable="df", column="total") in column_writes
        assert ColumnWrite(variable="df", column="tax") in column_writes

    def test_structural_reads(self):
        """Structural reads are converted to StructuralRead events."""
        tracking = TrackingData(structural_reads={"df": {"columns", "shape"}})
        events = tracking.to_access_events()

        structural_reads = [e for e in events if isinstance(e, StructuralRead)]
        assert len(structural_reads) == 2
        assert StructuralRead(variable="df", attr="columns") in structural_reads
        assert StructuralRead(variable="df", attr="shape") in structural_reads

    def test_multiple_variables(self):
        """Events from multiple variables are all included."""
        tracking = TrackingData(
            column_reads_before_writes={"df1": {"a"}, "df2": {"b"}},
            structural_reads={"df1": {"columns"}},
        )
        events = tracking.to_access_events()

        assert len(events) == 3
        assert ColumnRead(variable="df1", column="a") in events
        assert ColumnRead(variable="df2", column="b") in events
        assert StructuralRead(variable="df1", attr="columns") in events

    def test_deterministic_order(self):
        """Events are in deterministic order (sorted by variable, then column/attr)."""
        tracking = TrackingData(
            column_reads_before_writes={"z": {"b", "a"}, "a": {"x"}},
        )
        events = tracking.to_access_events()

        # Should be sorted: a.x, z.a, z.b
        assert events[0] == ColumnRead(variable="a", column="x")
        assert events[1] == ColumnRead(variable="z", column="a")
        assert events[2] == ColumnRead(variable="z", column="b")


class TestToReadEvents:
    """Tests for TrackingData.to_read_events() method."""

    def test_excludes_writes(self):
        """to_read_events excludes ColumnWrite events."""
        tracking = TrackingData(
            column_reads_before_writes={"df": {"price"}},
            column_writes={"df": {"total"}},
            structural_reads={"df": {"columns"}},
        )
        events = tracking.to_read_events()

        # Should have 2 events (read + structural), not 3 (no write)
        assert len(events) == 2
        assert ColumnRead(variable="df", column="price") in events
        assert StructuralRead(variable="df", attr="columns") in events
        # No ColumnWrite
        assert not any(isinstance(e, ColumnWrite) for e in events)

    def test_includes_variable_reads(self):
        """to_read_events includes VariableRead for non-DataFrame reads."""
        tracking = TrackingData(
            reads_before_writes={"x", "config"},  # Variable-level reads
            column_reads_before_writes={"df": {"price"}},  # df has column detail
        )
        events = tracking.to_read_events()

        # x and config should become VariableRead (no column/structural detail)
        # df should become ColumnRead (has column detail)
        assert VariableRead(variable="x") in events
        assert VariableRead(variable="config") in events
        assert ColumnRead(variable="df", column="price") in events
        # df should NOT also be a VariableRead since it has detail
        assert VariableRead(variable="df") not in events


class TestGetReadVariables:
    """Tests for TrackingData.get_read_variables() method."""

    def test_combines_column_and_structural_reads(self):
        """Variables from both column reads and structural reads are included."""
        tracking = TrackingData(
            column_reads_before_writes={"df1": {"a"}},
            structural_reads={"df2": {"shape"}},
        )
        variables = tracking.get_read_variables()
        assert variables == {"df1", "df2"}

    def test_no_duplicates(self):
        """Same variable in both is only returned once."""
        tracking = TrackingData(
            column_reads_before_writes={"df": {"a"}},
            structural_reads={"df": {"shape"}},
        )
        variables = tracking.get_read_variables()
        assert variables == {"df"}

    def test_includes_variable_level_reads(self):
        """Variable-level reads are also included."""
        tracking = TrackingData(
            reads_before_writes={"x", "y"},
            column_reads_before_writes={"df": {"a"}},
        )
        variables = tracking.get_read_variables()
        assert variables == {"x", "y", "df"}


class TestGetWrittenVariables:
    """Tests for TrackingData.get_written_variables() method."""

    def test_returns_writes(self):
        """Returns the writes set from TrackingData."""
        tracking = TrackingData(writes={"x", "y", "z"})
        variables = tracking.get_written_variables()
        assert variables == {"x", "y", "z"}


class TestToJsonFriendly:
    """Tests for TrackingData.to_json_friendly() method."""

    def test_empty_tracking(self):
        """Empty TrackingData produces empty lists/dicts."""
        tracking = TrackingData()
        result = tracking.to_json_friendly()

        assert result == {
            "reads": [],
            "writes": [],
            "column_reads": {},
            "column_writes": {},
            "structural_reads": {},
            "file_reads": [],
            "file_writes": [],
        }

    def test_converts_sets_to_sorted_lists(self):
        """Sets are converted to sorted lists."""
        tracking = TrackingData(
            reads_before_writes={"z", "a", "m"},
            writes={"y", "b"},
            column_reads_before_writes={"df": {"c", "a", "b"}},
            column_writes={"df": {"x", "w"}},
            structural_reads={"df": {"shape", "columns"}},
        )
        result = tracking.to_json_friendly()

        # All should be sorted lists
        assert result["reads"] == ["a", "m", "z"]
        assert result["writes"] == ["b", "y"]
        assert result["column_reads"] == {"df": ["a", "b", "c"]}
        assert result["column_writes"] == {"df": ["w", "x"]}
        assert result["structural_reads"] == {"df": ["columns", "shape"]}


# =============================================================================
# Test change_detector
# =============================================================================


class TestDetectChanges:
    """Tests for detect_changes function."""

    def test_empty_diff(self):
        """Empty MemoryCheckpointDiffResult produces empty list."""
        diff = MemoryCheckpointDiffResult(differences={})
        changes = detect_changes(diff)
        assert changes == []

    def test_value_comparison_is_value_changed(self):
        """Top-level ValueComparison becomes ValueChanged."""
        diff = MemoryCheckpointDiffResult(
            differences={
                "x": ValueComparison(
                    status="different", value1=1, value2=2, message="values differ"
                )
            }
        )
        changes = detect_changes(diff)
        assert len(changes) == 1
        assert isinstance(changes[0], ValueChanged)
        assert changes[0].variable == "x"

    def test_dataframe_column_modified(self):
        """DataFrame column change becomes ColumnModified."""
        diff = MemoryCheckpointDiffResult(
            differences={
                "df": CompoundDiff(
                    source_type="dataframe",
                    children={
                        "['price']": ValueComparison(
                            status="different",
                            value1="[1,2,3]",
                            value2="[4,5,6]",
                            message="values differ",
                        )
                    },
                )
            }
        )
        changes = detect_changes(diff)
        assert len(changes) == 1
        assert isinstance(changes[0], ColumnModified)
        assert changes[0].variable == "df"
        assert changes[0].column == "price"


class TestParseRowChange:
    """Tests for _parse_row_change helper."""

    def test_rows_added(self):
        """Parses row count increase as RowsAdded."""
        node = ValueComparison(
            status="different",
            value1=5,
            value2=8,
            message="Row count changed from 5 to 8",
        )
        changes = _parse_row_change("df", node)
        assert len(changes) == 1
        assert isinstance(changes[0], RowsAdded)
        assert changes[0].count == 3

    def test_rows_removed(self):
        """Parses row count decrease as RowsRemoved."""
        node = ValueComparison(
            status="different",
            value1=10,
            value2=7,
            message="Row count changed from 10 to 7",
        )
        changes = _parse_row_change("df", node)
        assert len(changes) == 1
        assert isinstance(changes[0], RowsRemoved)
        assert changes[0].count == 3


class TestParseColumnStructuralChange:
    """Tests for _parse_column_structural_change helper."""

    def test_single_column_added(self):
        """Parses single column addition."""
        node = ValueComparison(
            status="different",
            value1=None,
            value2=None,
            message="Columns added: ['new_col']",
        )
        changes = _parse_column_structural_change("df", node)
        assert len(changes) == 1
        assert isinstance(changes[0], ColumnAdded)
        assert changes[0].column == "new_col"

    def test_multiple_columns_added(self):
        """Parses multiple column additions."""
        node = ValueComparison(
            status="different",
            value1=None,
            value2=None,
            message="Columns added: ['a', 'b', 'c']",
        )
        changes = _parse_column_structural_change("df", node)
        assert len(changes) == 3
        cols = {c.column for c in changes}
        assert cols == {"a", "b", "c"}


class TestExtractColumnName:
    """Tests for _extract_column_name helper."""

    def test_single_quotes(self):
        """Extracts column from ['col']."""
        assert _extract_column_name("['price']") == "price"

    def test_double_quotes(self):
        """Extracts column from ['col']."""
        assert _extract_column_name('["price"]') == "price"

    def test_not_column_key(self):
        """Returns None for non-column keys."""
        assert _extract_column_name("_structural_rows") is None
        assert _extract_column_name("something_else") is None


class TestAnalyzeColumnChange:
    """Tests for _analyze_column_change helper."""

    def test_column_removed(self):
        """Detects column removal from message."""
        node = ValueComparison(
            status="different",
            value1="some_data",
            value2=None,
            message="Column 'old_col' only in first DataFrame",
        )
        change = _analyze_column_change("df", "old_col", node)
        assert isinstance(change, ColumnRemoved)

    def test_column_added(self):
        """Detects column addition from message."""
        node = ValueComparison(
            status="different",
            value1=None,
            value2="some_data",
            message="Column 'new_col' only in second DataFrame",
        )
        change = _analyze_column_change("df", "new_col", node)
        assert isinstance(change, ColumnAdded)

    def test_column_modified(self):
        """Defaults to ColumnModified for value changes."""
        node = ValueComparison(
            status="different",
            value1="[1,2,3]",
            value2="[4,5,6]",
            message="Values differ",
        )
        change = _analyze_column_change("df", "price", node)
        assert isinstance(change, ColumnModified)


class TestGetChangedVariables:
    """Tests for get_changed_variables helper."""

    def test_returns_variable_names(self):
        """Returns set of variable names from diff."""
        diff = MemoryCheckpointDiffResult(
            differences={
                "x": ValueComparison(
                    status="different", value1=1, value2=2, message=""
                ),
                "y": ValueComparison(
                    status="different", value1=3, value2=4, message=""
                ),
            }
        )
        vars = get_changed_variables(diff)
        assert vars == {"x", "y"}


class TestHasAnyChanges:
    """Tests for has_any_changes helper."""

    def test_empty_is_false(self):
        """Empty diff has no changes."""
        diff = MemoryCheckpointDiffResult(differences={})
        assert has_any_changes(diff) is False

    def test_non_empty_is_true(self):
        """Non-empty diff has changes."""
        diff = MemoryCheckpointDiffResult(
            differences={
                "x": ValueComparison(
                    status="different", value1=1, value2=2, message=""
                )
            }
        )
        assert has_any_changes(diff) is True


# =============================================================================
# Integration test: full conversion pipeline
# =============================================================================


class TestIntegration:
    """Integration tests for the full conversion pipeline using LocSet operations."""

    def test_column_modification_conflict_detection(self):
        """
        Full pipeline: TrackingData + MemoryCheckpointDiffResult -> Conflict via LocSet ▷

        Scenario:
        - Prior cell read df['price']
        - Current cell modified df['price']
        - Should detect conflict
        """
        from flowbook.kernel.change_detector import changes_to_write_locs
        from flowbook.kernel.locations import tracking_to_readlocset, wlocs_conflict_rlocs

        # Prior cell's tracking
        prior_tracking = TrackingData(
            column_reads_before_writes={"df": {"price", "quantity"}}
        )
        R_prior = tracking_to_readlocset(prior_tracking)

        # Current cell's diff
        diff = MemoryCheckpointDiffResult(
            differences={
                "df": CompoundDiff(
                    source_type="dataframe",
                    children={
                        "['price']": ValueComparison(
                            status="different",
                            value1="[1,2,3]",
                            value2="[4,5,6]",
                            message="values differ",
                        )
                    },
                )
            }
        )
        changes = detect_changes(diff)
        W_i = changes_to_write_locs(changes)

        # Should detect conflict: ColumnModified(df, price) vs Col(df, price)
        conflicting = wlocs_conflict_rlocs(W_i, R_prior)
        assert len(conflicting) == 1
        conflict = next(iter(conflicting))
        assert conflict.qualifier == "df"
        assert conflict.name == "price"

    def test_structural_change_conflict_detection(self):
        """
        Full pipeline with structural changes via LocSet ▷.

        Scenario:
        - Prior cell read df.columns
        - Current cell added a column
        - Should detect conflict (col_add conflicts with Attr(df, columns))
        """
        from flowbook.kernel.change_detector import changes_to_write_locs
        from flowbook.kernel.locations import tracking_to_readlocset, wlocs_conflict_rlocs

        # Prior cell's tracking
        prior_tracking = TrackingData(structural_reads={"df": {"columns"}})
        R_prior = tracking_to_readlocset(prior_tracking)

        # Current cell's diff - column added
        diff = MemoryCheckpointDiffResult(
            differences={
                "df": CompoundDiff(
                    source_type="dataframe",
                    children={
                        "_structural_columns": ValueComparison(
                            status="different",
                            value1=None,
                            value2=None,
                            message="Columns added: ['new_col']",
                        )
                    },
                )
            }
        )
        changes = detect_changes(diff)
        W_i = changes_to_write_locs(changes)

        # Should detect conflict: ColAdd(df, new_col) vs Attr(df, columns)
        conflicting = wlocs_conflict_rlocs(W_i, R_prior)
        assert len(conflicting) == 1
        conflict = next(iter(conflicting))
        assert isinstance(conflict, type(conflict))  # WriteLoc
        assert "new_col" in conflict.name
