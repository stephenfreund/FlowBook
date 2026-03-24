"""Unit tests for ColumnProvenanceTracker."""

import pandas as pd
import pytest

from flowbook.kernel_support.column_provenance import (
    ColumnProvenanceTracker,
    PROVENANCE_KEY,
)


class TestRecordVarWrite:
    """Tests for record_var_write — full DataFrame assignment."""

    def test_sets_all_column_origins(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        origins = ColumnProvenanceTracker.get_origins(df)
        assert origins == {"a": "cell1", "b": "cell1", "c": "cell1"}

    def test_overwrites_previous_origins(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_var_write(df, "cell2")
        origins = ColumnProvenanceTracker.get_origins(df)
        assert origins == {"a": "cell2", "b": "cell2"}

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        assert ColumnProvenanceTracker.get_origins(df) == {}

    def test_handles_non_string_column_names(self):
        df = pd.DataFrame({0: [1], 1: [2]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        origins = ColumnProvenanceTracker.get_origins(df)
        assert origins == {"0": "cell1", "1": "cell1"}


class TestRecordColumnWrite:
    """Tests for record_column_write — first writer wins."""

    def test_records_new_column_origin(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.get_origins(df) == {"x": "cell2"}

    def test_first_writer_wins(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_write(df, "x", "cell1")
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.get_origins(df)["x"] == "cell1"

    def test_does_not_overwrite_var_write_origin(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_column_write(df, "a", "cell2")
        assert ColumnProvenanceTracker.get_origins(df)["a"] == "cell1"

    def test_new_column_after_var_write(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        origins = ColumnProvenanceTracker.get_origins(df)
        assert origins["a"] == "cell1"
        assert origins["x"] == "cell2"

    def test_no_provenance_yet(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_write(df, "x", "cell1")
        assert ColumnProvenanceTracker.get_origins(df) == {"x": "cell1"}


class TestRecordColumnDelete:
    """Tests for record_column_delete."""

    def test_removes_origin(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_column_delete(df, "a")
        origins = ColumnProvenanceTracker.get_origins(df)
        assert "a" not in origins
        assert origins["b"] == "cell1"

    def test_delete_nonexistent_column_no_error(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_delete(df, "nonexistent")

    def test_delete_from_empty_provenance(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_delete(df, "a")


class TestGetColumnsFromCell:
    """Tests for get_columns_from_cell."""

    def test_filters_by_cell_id(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.get_columns_from_cell(df, "cell1") == {"a", "b", "c"}
        assert ColumnProvenanceTracker.get_columns_from_cell(df, "cell2") == {"x"}
        assert ColumnProvenanceTracker.get_columns_from_cell(df, "cell3") == set()


class TestIsColumnAddedBy:
    """Tests for is_column_added_by."""

    def test_true_for_matching_cell(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.is_column_added_by(df, "x", "cell2") is True

    def test_false_for_different_cell(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.is_column_added_by(df, "x", "cell1") is False

    def test_false_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1]})
        assert ColumnProvenanceTracker.is_column_added_by(df, "z", "cell1") is False

    def test_false_for_no_provenance(self):
        df = pd.DataFrame({"a": [1]})
        assert ColumnProvenanceTracker.is_column_added_by(df, "a", "cell1") is False


class TestCopyIsolation:
    """Tests that provenance survives copies correctly."""

    def test_shallow_copy_isolates_origins(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        df2 = df.copy(deep=False)
        ColumnProvenanceTracker.record_column_write(df2, "x", "cell2")
        # df should not have x's origin
        assert "x" not in ColumnProvenanceTracker.get_origins(df)
        assert ColumnProvenanceTracker.get_origins(df2)["x"] == "cell2"

    def test_deep_copy_isolates_origins(self):
        import copy
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        df2 = copy.deepcopy(df)
        ColumnProvenanceTracker.record_column_write(df2, "x", "cell2")
        assert "x" not in ColumnProvenanceTracker.get_origins(df)

    def test_pandas_copy_preserves_origins(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        df2 = df.copy()
        assert ColumnProvenanceTracker.get_origins(df2) == {"a": "cell1"}

    def test_aliasing_shares_origins(self):
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        X = df  # alias
        ColumnProvenanceTracker.record_column_write(X, "x", "cell2")
        # Both names see the same provenance
        assert ColumnProvenanceTracker.get_origins(df)["x"] == "cell2"


class TestEdgeCases:
    """Tests for various edge cases."""

    def test_no_provenance_returns_empty(self):
        df = pd.DataFrame({"a": [1]})
        assert ColumnProvenanceTracker.get_origins(df) == {}

    def test_read_csv_scenario(self):
        """Simulate pd.read_csv: all columns from creator cell."""
        df = pd.DataFrame({"name": ["Alice"], "age": [30], "score": [85]})
        ColumnProvenanceTracker.record_var_write(df, "cell_a")
        origins = ColumnProvenanceTracker.get_origins(df)
        assert all(v == "cell_a" for v in origins.values())
        assert set(origins.keys()) == {"name", "age", "score"}

    def test_merge_scenario(self):
        """Simulate df = df.merge(...): new DataFrame, all columns reset."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        ColumnProvenanceTracker.record_var_write(df1, "cell1")
        df2 = pd.DataFrame({"a": [1], "c": [3]})
        merged = df1.merge(df2, on="a")
        # Var write sets all columns of merged to the merging cell
        ColumnProvenanceTracker.record_var_write(merged, "cell2")
        origins = ColumnProvenanceTracker.get_origins(merged)
        assert origins == {"a": "cell2", "b": "cell2", "c": "cell2"}

    def test_concat_scenario(self):
        """Simulate pd.concat: new DataFrame, all columns from creator."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        result = pd.concat([df1, df2])
        ColumnProvenanceTracker.record_var_write(result, "cell3")
        assert ColumnProvenanceTracker.get_origins(result) == {"a": "cell3"}

    def test_insert_scenario(self):
        """Simulate df.insert(loc, col, val)."""
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        # insert would be captured by the insert patch
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.get_origins(df)["x"] == "cell2"
        assert ColumnProvenanceTracker.get_origins(df)["a"] == "cell1"

    def test_column_overwrite_preserves_first_writer(self):
        """Re-executing df['x'] = new_value doesn't change origin."""
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        # Re-execute same cell
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        assert ColumnProvenanceTracker.get_origins(df)["x"] == "cell2"
        # Different cell tries to write same column
        ColumnProvenanceTracker.record_column_write(df, "x", "cell3")
        assert ColumnProvenanceTracker.get_origins(df)["x"] == "cell2"

    def test_delete_then_recreate_column(self):
        """Column deleted and re-added gets new origin."""
        df = pd.DataFrame({"a": [1]})
        ColumnProvenanceTracker.record_var_write(df, "cell1")
        ColumnProvenanceTracker.record_column_write(df, "x", "cell2")
        ColumnProvenanceTracker.record_column_delete(df, "x")
        assert "x" not in ColumnProvenanceTracker.get_origins(df)
        # Re-create with different cell
        ColumnProvenanceTracker.record_column_write(df, "x", "cell3")
        assert ColumnProvenanceTracker.get_origins(df)["x"] == "cell3"
