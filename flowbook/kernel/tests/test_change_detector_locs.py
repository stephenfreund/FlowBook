"""Tests for changes_to_write_locs() in change_detector.py."""

import pytest

from flowbook.kernel.change_detector import changes_to_write_locs
from flowbook.kernel.changes import (
    ColumnAdded,
    ColumnModified,
    ColumnRemoved,
    DtypeChanged,
    IndexChanged,
    RowsAdded,
    RowsRemoved,
    ValueChanged,
)
from flowbook.kernel.locations import WriteLoc


class TestValueChanged:
    def test_single_variable(self):
        result = changes_to_write_locs([ValueChanged(variable="x")])
        assert result == frozenset({WriteLoc.var("x")})

    def test_multiple_variables(self):
        result = changes_to_write_locs([
            ValueChanged(variable="x"),
            ValueChanged(variable="y"),
        ])
        assert result == frozenset({WriteLoc.var("x"), WriteLoc.var("y")})

    def test_duplicate_variable(self):
        result = changes_to_write_locs([
            ValueChanged(variable="x"),
            ValueChanged(variable="x"),
        ])
        assert result == frozenset({WriteLoc.var("x")})


class TestColumnModified:
    def test_single_column(self):
        result = changes_to_write_locs([
            ColumnModified(variable="df", column="price"),
        ])
        assert result == frozenset({WriteLoc.col("df", "price")})

    def test_multiple_columns(self):
        result = changes_to_write_locs([
            ColumnModified(variable="df", column="price"),
            ColumnModified(variable="df", column="qty"),
        ])
        assert result == frozenset({
            WriteLoc.col("df", "price"),
            WriteLoc.col("df", "qty"),
        })


class TestColumnAdded:
    def test_single_column(self):
        result = changes_to_write_locs([
            ColumnAdded(variable="df", column="new_col"),
        ])
        assert result == frozenset({WriteLoc.col_add("df", "new_col")})

    def test_multiple_columns_added(self):
        result = changes_to_write_locs([
            ColumnAdded(variable="df", column="a"),
            ColumnAdded(variable="df", column="b"),
        ])
        assert result == frozenset({
            WriteLoc.col_add("df", "a"),
            WriteLoc.col_add("df", "b"),
        })


class TestColumnRemoved:
    def test_single_column(self):
        result = changes_to_write_locs([
            ColumnRemoved(variable="df", column="old_col"),
        ])
        assert result == frozenset({WriteLoc.col_del("df", "old_col")})


class TestRowsAdded:
    def test_rows_added(self):
        result = changes_to_write_locs([
            RowsAdded(variable="df", count=5),
        ])
        assert result == frozenset({WriteLoc.rows("df")})


class TestRowsRemoved:
    def test_rows_removed(self):
        result = changes_to_write_locs([
            RowsRemoved(variable="df", count=3),
        ])
        assert result == frozenset({WriteLoc.rows("df")})

    def test_rows_added_and_removed_same_var(self):
        """RowsAdded and RowsRemoved both map to WriteLoc.rows, so they deduplicate."""
        result = changes_to_write_locs([
            RowsAdded(variable="df", count=2),
            RowsRemoved(variable="df", count=1),
        ])
        assert result == frozenset({WriteLoc.rows("df")})


class TestIndexChanged:
    def test_index_changed(self):
        result = changes_to_write_locs([
            IndexChanged(variable="df"),
        ])
        assert result == frozenset({WriteLoc.attr_changed("df", "index")})


class TestDtypeChanged:
    def test_dtype_changed_produces_col_and_attr(self):
        result = changes_to_write_locs([
            DtypeChanged(variable="df", column="x", old_dtype="int64", new_dtype="float64"),
        ])
        assert result == frozenset({
            WriteLoc.col("df", "x"),
            WriteLoc.attr_changed("df", "dtypes"),
        })

    def test_dtype_changed_multiple_columns(self):
        result = changes_to_write_locs([
            DtypeChanged(variable="df", column="x", old_dtype="int64", new_dtype="float64"),
            DtypeChanged(variable="df", column="y", old_dtype="object", new_dtype="string"),
        ])
        assert result == frozenset({
            WriteLoc.col("df", "x"),
            WriteLoc.col("df", "y"),
            WriteLoc.attr_changed("df", "dtypes"),
        })


class TestEmptyInput:
    def test_empty_list(self):
        result = changes_to_write_locs([])
        assert result == frozenset()


class TestMixedChanges:
    def test_mixed_change_types(self):
        result = changes_to_write_locs([
            ValueChanged(variable="x"),
            ColumnModified(variable="df", column="price"),
            ColumnAdded(variable="df", column="new"),
            RowsAdded(variable="df2", count=10),
            IndexChanged(variable="df3"),
        ])
        assert result == frozenset({
            WriteLoc.var("x"),
            WriteLoc.col("df", "price"),
            WriteLoc.col_add("df", "new"),
            WriteLoc.rows("df2"),
            WriteLoc.attr_changed("df3", "index"),
        })
