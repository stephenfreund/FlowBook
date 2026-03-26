"""Unit tests for DataFrameProvenance and DataFrameProvenanceTracker."""

import copy

import pandas as pd
import pytest

from flowbook.kernel_support.column_provenance import (
    DataFrameProvenance,
    DataFrameProvenanceTracker,
    PROVENANCE_KEY,
)


# ── DataFrameProvenance class tests ────────────────────────────────────


class TestDataFrameProvenance:
    """Tests for the DataFrameProvenance data class itself."""

    def test_default_empty(self):
        prov = DataFrameProvenance()
        assert prov.col_origins == {}
        assert prov.col_deletions == {}
        assert prov.dtype_origins == {}
        assert prov.row_mutators == set()
        assert prov.index_mutators == set()

    def test_copy_isolates(self):
        prov = DataFrameProvenance()
        prov.col_origins = {"a": "cell1"}
        prov.col_deletions = {"b": "cell2"}
        prov.dtype_origins = {"c": "cell3"}
        prov.row_mutators = {"cell4"}
        prov.index_mutators = {"cell5"}

        prov2 = copy.copy(prov)
        prov2.col_origins["x"] = "new"
        prov2.col_deletions["y"] = "new"
        prov2.dtype_origins["z"] = "new"
        prov2.row_mutators.add("new")
        prov2.index_mutators.add("new")

        assert "x" not in prov.col_origins
        assert "y" not in prov.col_deletions
        assert "z" not in prov.dtype_origins
        assert "new" not in prov.row_mutators
        assert "new" not in prov.index_mutators

    def test_deepcopy_isolates(self):
        prov = DataFrameProvenance()
        prov.col_origins = {"a": "cell1"}
        prov.row_mutators = {"cell2"}

        prov2 = copy.deepcopy(prov)
        prov2.col_origins["x"] = "new"
        prov2.row_mutators.add("new")

        assert "x" not in prov.col_origins
        assert "new" not in prov.row_mutators

    def test_copy_preserves_data(self):
        prov = DataFrameProvenance()
        prov.col_origins = {"a": "cell1", "b": "cell2"}
        prov.col_deletions = {"c": "cell3"}
        prov.dtype_origins = {"d": "cell4"}
        prov.row_mutators = {"cell5", "cell6"}
        prov.index_mutators = {"cell7"}

        prov2 = copy.copy(prov)
        assert prov2.col_origins == {"a": "cell1", "b": "cell2"}
        assert prov2.col_deletions == {"c": "cell3"}
        assert prov2.dtype_origins == {"d": "cell4"}
        assert prov2.row_mutators == {"cell5", "cell6"}
        assert prov2.index_mutators == {"cell7"}


# ── Existing column provenance tests (renamed) ────────────────────────


class TestRecordVarWrite:
    """Tests for record_var_write — full DataFrame assignment."""

    def test_sets_all_column_origins(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        origins = DataFrameProvenanceTracker.get_origins(df)
        assert origins == {"a": "cell1", "b": "cell1", "c": "cell1"}

    def test_overwrites_previous_origins(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_var_write(df, "cell2")
        origins = DataFrameProvenanceTracker.get_origins(df)
        assert origins == {"a": "cell2", "b": "cell2"}

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        assert DataFrameProvenanceTracker.get_origins(df) == {}

    def test_handles_non_string_column_names(self):
        df = pd.DataFrame({0: [1], 1: [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        origins = DataFrameProvenanceTracker.get_origins(df)
        assert origins == {"0": "cell1", "1": "cell1"}


class TestRecordVarWriteResetsAll:
    """Tests that record_var_write resets ALL provenance fields."""

    def _setup_full_provenance(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        prov = DataFrameProvenanceTracker._get_or_create(df)
        prov.col_deletions["x"] = "cell2"
        prov.dtype_origins["a"] = "cell3"
        prov.row_mutators.add("cell4")
        prov.index_mutators.add("cell5")
        return df

    def test_clears_deletions(self):
        df = self._setup_full_provenance()
        DataFrameProvenanceTracker.record_var_write(df, "cell_new")
        prov = DataFrameProvenanceTracker.get_provenance(df)
        assert prov.col_deletions == {}

    def test_clears_dtype_origins(self):
        df = self._setup_full_provenance()
        DataFrameProvenanceTracker.record_var_write(df, "cell_new")
        prov = DataFrameProvenanceTracker.get_provenance(df)
        assert prov.dtype_origins == {}

    def test_clears_row_mutators(self):
        df = self._setup_full_provenance()
        DataFrameProvenanceTracker.record_var_write(df, "cell_new")
        prov = DataFrameProvenanceTracker.get_provenance(df)
        assert prov.row_mutators == set()

    def test_clears_index_mutators(self):
        df = self._setup_full_provenance()
        DataFrameProvenanceTracker.record_var_write(df, "cell_new")
        prov = DataFrameProvenanceTracker.get_provenance(df)
        assert prov.index_mutators == set()


class TestRecordColumnWrite:
    """Tests for record_column_write — first writer wins."""

    def test_records_new_column_origin(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.get_origins(df) == {"x": "cell2"}

    def test_first_writer_wins(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell1")
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.get_origins(df)["x"] == "cell1"

    def test_does_not_overwrite_var_write_origin(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_write(df, "a", "cell2")
        assert DataFrameProvenanceTracker.get_origins(df)["a"] == "cell1"

    def test_new_column_after_var_write(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        origins = DataFrameProvenanceTracker.get_origins(df)
        assert origins["a"] == "cell1"
        assert origins["x"] == "cell2"

    def test_no_provenance_yet(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell1")
        assert DataFrameProvenanceTracker.get_origins(df) == {"x": "cell1"}


class TestRecordColumnDelete:
    """Tests for record_column_delete."""

    def test_removes_origin(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_delete(df, "a")
        origins = DataFrameProvenanceTracker.get_origins(df)
        assert "a" not in origins
        assert origins["b"] == "cell1"

    def test_delete_nonexistent_column_no_error(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_delete(df, "nonexistent")

    def test_delete_from_empty_provenance(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_delete(df, "a")


class TestRecordColumnDeleteWithCellId:
    """Tests for record_column_delete with cell_id tracking."""

    def test_records_deleter(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_delete(df, "b", "cell2")
        assert DataFrameProvenanceTracker.is_column_deleted_by(df, "b", "cell2")

    def test_first_deleter_wins(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_delete(df, "b", "cell2")
        DataFrameProvenanceTracker.record_column_delete(df, "b", "cell3")
        assert DataFrameProvenanceTracker.is_column_deleted_by(df, "b", "cell2")
        assert not DataFrameProvenanceTracker.is_column_deleted_by(df, "b", "cell3")

    def test_delete_without_cell_id(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_delete(df, "b")
        # Origin removed but no deletion recorded
        assert "b" not in DataFrameProvenanceTracker.get_origins(df)
        assert not DataFrameProvenanceTracker.is_column_deleted_by(df, "b", "cell1")


class TestRecordDtypeChange:
    """Tests for record_dtype_change."""

    def test_records_dtype_changer(self):
        df = pd.DataFrame({"x": [1]})
        DataFrameProvenanceTracker.record_dtype_change(df, "x", "cell2")
        assert DataFrameProvenanceTracker.is_dtype_changed_by(df, "x", "cell2")

    def test_first_changer_wins(self):
        df = pd.DataFrame({"x": [1]})
        DataFrameProvenanceTracker.record_dtype_change(df, "x", "cell2")
        DataFrameProvenanceTracker.record_dtype_change(df, "x", "cell3")
        assert DataFrameProvenanceTracker.is_dtype_changed_by(df, "x", "cell2")
        assert not DataFrameProvenanceTracker.is_dtype_changed_by(df, "x", "cell3")

    def test_false_for_different_cell(self):
        df = pd.DataFrame({"x": [1]})
        DataFrameProvenanceTracker.record_dtype_change(df, "x", "cell2")
        assert not DataFrameProvenanceTracker.is_dtype_changed_by(df, "x", "other")

    def test_false_for_no_provenance(self):
        df = pd.DataFrame({"x": [1]})
        assert not DataFrameProvenanceTracker.is_dtype_changed_by(df, "x", "cell1")


class TestRecordRowMutation:
    """Tests for record_row_mutation."""

    def test_records_row_mutator(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_row_mutation(df, "cell2")
        assert DataFrameProvenanceTracker.is_row_mutator(df, "cell2")

    def test_multiple_mutators(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_row_mutation(df, "cell2")
        DataFrameProvenanceTracker.record_row_mutation(df, "cell3")
        assert DataFrameProvenanceTracker.is_row_mutator(df, "cell2")
        assert DataFrameProvenanceTracker.is_row_mutator(df, "cell3")

    def test_false_for_non_mutator(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_row_mutation(df, "cell2")
        assert not DataFrameProvenanceTracker.is_row_mutator(df, "other")

    def test_no_provenance(self):
        df = pd.DataFrame({"a": [1]})
        assert not DataFrameProvenanceTracker.is_row_mutator(df, "cell1")


class TestRecordIndexMutation:
    """Tests for record_index_mutation."""

    def test_records_index_mutator(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_index_mutation(df, "cell2")
        assert DataFrameProvenanceTracker.is_index_mutator(df, "cell2")

    def test_multiple_mutators(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_index_mutation(df, "cell2")
        DataFrameProvenanceTracker.record_index_mutation(df, "cell3")
        assert DataFrameProvenanceTracker.is_index_mutator(df, "cell2")
        assert DataFrameProvenanceTracker.is_index_mutator(df, "cell3")

    def test_false_for_non_mutator(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_index_mutation(df, "cell2")
        assert not DataFrameProvenanceTracker.is_index_mutator(df, "other")

    def test_no_provenance(self):
        df = pd.DataFrame({"a": [1]})
        assert not DataFrameProvenanceTracker.is_index_mutator(df, "cell1")


class TestGetColumnsFromCell:
    """Tests for get_columns_from_cell."""

    def test_filters_by_cell_id(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.get_columns_from_cell(df, "cell1") == {"a", "b", "c"}
        assert DataFrameProvenanceTracker.get_columns_from_cell(df, "cell2") == {"x"}
        assert DataFrameProvenanceTracker.get_columns_from_cell(df, "cell3") == set()


class TestIsColumnAddedBy:
    """Tests for is_column_added_by."""

    def test_true_for_matching_cell(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.is_column_added_by(df, "x", "cell2") is True

    def test_false_for_different_cell(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.is_column_added_by(df, "x", "cell1") is False

    def test_false_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1]})
        assert DataFrameProvenanceTracker.is_column_added_by(df, "z", "cell1") is False

    def test_false_for_no_provenance(self):
        df = pd.DataFrame({"a": [1]})
        assert DataFrameProvenanceTracker.is_column_added_by(df, "a", "cell1") is False


class TestCopyIsolation:
    """Tests that provenance survives copies correctly."""

    def test_shallow_copy_isolates_origins(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        df2 = df.copy(deep=False)
        DataFrameProvenanceTracker.record_column_write(df2, "x", "cell2")
        # df should not have x's origin
        assert "x" not in DataFrameProvenanceTracker.get_origins(df)
        assert DataFrameProvenanceTracker.get_origins(df2)["x"] == "cell2"

    def test_deep_copy_isolates_origins(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        df2 = copy.deepcopy(df)
        DataFrameProvenanceTracker.record_column_write(df2, "x", "cell2")
        assert "x" not in DataFrameProvenanceTracker.get_origins(df)

    def test_pandas_copy_preserves_origins(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        df2 = df.copy()
        assert DataFrameProvenanceTracker.get_origins(df2) == {"a": "cell1"}

    def test_aliasing_shares_origins(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        X = df  # alias
        DataFrameProvenanceTracker.record_column_write(X, "x", "cell2")
        # Both names see the same provenance
        assert DataFrameProvenanceTracker.get_origins(df)["x"] == "cell2"


class TestCopyIsolationFull:
    """Tests copy isolation for ALL provenance fields, not just col_origins."""

    def test_copy_isolates_deletions(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_delete(df, "b", "cell2")
        df2 = df.copy(deep=False)
        DataFrameProvenanceTracker.record_column_delete(df2, "a", "cell3")
        # Original should not have 'a' deletion
        assert not DataFrameProvenanceTracker.is_column_deleted_by(df, "a", "cell3")
        assert DataFrameProvenanceTracker.is_column_deleted_by(df2, "a", "cell3")

    def test_copy_isolates_row_mutators(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_row_mutation(df, "cell1")
        df2 = df.copy(deep=False)
        DataFrameProvenanceTracker.record_row_mutation(df2, "cell2")
        assert not DataFrameProvenanceTracker.is_row_mutator(df, "cell2")
        assert DataFrameProvenanceTracker.is_row_mutator(df2, "cell2")

    def test_copy_preserves_all_provenance(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_delete(df, "b", "cell2")
        DataFrameProvenanceTracker.record_dtype_change(df, "a", "cell3")
        DataFrameProvenanceTracker.record_row_mutation(df, "cell4")
        DataFrameProvenanceTracker.record_index_mutation(df, "cell5")

        df2 = df.copy()
        assert DataFrameProvenanceTracker.get_origins(df2) == {"a": "cell1"}
        assert DataFrameProvenanceTracker.is_column_deleted_by(df2, "b", "cell2")
        assert DataFrameProvenanceTracker.is_dtype_changed_by(df2, "a", "cell3")
        assert DataFrameProvenanceTracker.is_row_mutator(df2, "cell4")
        assert DataFrameProvenanceTracker.is_index_mutator(df2, "cell5")


class TestEdgeCases:
    """Tests for various edge cases."""

    def test_no_provenance_returns_empty(self):
        df = pd.DataFrame({"a": [1]})
        assert DataFrameProvenanceTracker.get_origins(df) == {}

    def test_read_csv_scenario(self):
        """Simulate pd.read_csv: all columns from creator cell."""
        df = pd.DataFrame({"name": ["Alice"], "age": [30], "score": [85]})
        DataFrameProvenanceTracker.record_var_write(df, "cell_a")
        origins = DataFrameProvenanceTracker.get_origins(df)
        assert all(v == "cell_a" for v in origins.values())
        assert set(origins.keys()) == {"name", "age", "score"}

    def test_merge_scenario(self):
        """Simulate df = df.merge(...): new DataFrame, all columns reset."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        DataFrameProvenanceTracker.record_var_write(df1, "cell1")
        df2 = pd.DataFrame({"a": [1], "c": [3]})
        merged = df1.merge(df2, on="a")
        # Var write sets all columns of merged to the merging cell
        DataFrameProvenanceTracker.record_var_write(merged, "cell2")
        origins = DataFrameProvenanceTracker.get_origins(merged)
        assert origins == {"a": "cell2", "b": "cell2", "c": "cell2"}

    def test_concat_scenario(self):
        """Simulate pd.concat: new DataFrame, all columns from creator."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        result = pd.concat([df1, df2])
        DataFrameProvenanceTracker.record_var_write(result, "cell3")
        assert DataFrameProvenanceTracker.get_origins(result) == {"a": "cell3"}

    def test_insert_scenario(self):
        """Simulate df.insert(loc, col, val)."""
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        # insert would be captured by the insert patch
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.get_origins(df)["x"] == "cell2"
        assert DataFrameProvenanceTracker.get_origins(df)["a"] == "cell1"

    def test_column_overwrite_preserves_first_writer(self):
        """Re-executing df['x'] = new_value doesn't change origin."""
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        # Re-execute same cell
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        assert DataFrameProvenanceTracker.get_origins(df)["x"] == "cell2"
        # Different cell tries to write same column
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell3")
        assert DataFrameProvenanceTracker.get_origins(df)["x"] == "cell2"

    def test_delete_then_recreate_column(self):
        """Column deleted and re-added gets new origin."""
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell2")
        DataFrameProvenanceTracker.record_column_delete(df, "x")
        assert "x" not in DataFrameProvenanceTracker.get_origins(df)
        # Re-create with different cell
        DataFrameProvenanceTracker.record_column_write(df, "x", "cell3")
        assert DataFrameProvenanceTracker.get_origins(df)["x"] == "cell3"

    def test_get_provenance_returns_none_for_fresh_df(self):
        df = pd.DataFrame({"a": [1]})
        assert DataFrameProvenanceTracker.get_provenance(df) is None

    def test_get_provenance_returns_object(self):
        df = pd.DataFrame({"a": [1]})
        DataFrameProvenanceTracker.record_var_write(df, "cell1")
        prov = DataFrameProvenanceTracker.get_provenance(df)
        assert isinstance(prov, DataFrameProvenance)
        assert prov.col_origins == {"a": "cell1"}
