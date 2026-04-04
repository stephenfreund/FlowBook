"""
Tests that in-place DataFrame mutations are detected by the reproducibility
enforcer as backward violations.

Each test follows the same pattern:
  1. Cell A reads df (or a specific column of df)
  2. Cell B mutates df in-place via a specific mechanism
  3. The enforcer must flag B's mutation as a backward violation

This verifies the full pipeline: checkpoint (with CoW) → in-place mutation →
diff (column-level comparison) → change_detector → conflict resolution.
"""

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
)
from flowbook.kernel.models import ErrorType
from flowbook.kernel.tests.conftest import make_tracking


def _get_backward_error(result):
    """Get the first backward mutation (NO_WRITE_AFTER_READ) error from result."""
    for e in result.errors:
        if e.error_type == ErrorType.NO_WRITE_AFTER_READ:
            return e
    return None


class TestInPlaceMutationDetection:
    """Verify that all in-place DataFrame mutation mechanisms trigger backward violations."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b"])

    def _run_backward_mutation_test(
        self,
        mutate_fn,
        *,
        column_reads=None,
        reads=None,
    ):
        """
        Helper: cell A reads df, cell B mutates df in-place.

        Args:
            mutate_fn: callable(df) that mutates df in-place
            column_reads: column-level reads for cell A (default: {"df": {"price"}})
            reads: variable-level reads for cell A (default: {"df"})

        Returns the result from cell B's check (should have a violation).
        """
        if reads is None:
            reads = {"df"}
        if column_reads is None:
            column_reads = {"df": {"price"}}

        # Build the shared DataFrame
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0],
            "quantity": [1, 2, 3],
            "name": ["apple", "banana", "cherry"],
        })

        # --- Cell A: reads df ---
        ns = {"df": df}
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}a", ns, max_size_mb=None
        )
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns,
            tracking=make_tracking(
                reads=reads,
                writes=set(),
                column_reads=column_reads,
            ),
        )
        assert not result_a.has_errors(), "Cell A should have no errors"

        # --- Cell B: mutates df in-place ---
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}b", ns, max_size_mb=None
        )

        # Perform the in-place mutation
        mutate_fn(df)

        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": {"price"}},
                column_writes={"df": {"price"}},
            ),
        )
        return result_b

    # ----------------------------------------------------------------
    # Element-wise assignment operations
    # ----------------------------------------------------------------

    def test_loc_single_element(self):
        """df.loc[row, col] = value"""
        result = self._run_backward_mutation_test(
            lambda df: df.loc.__setitem__((0, "price"), 999.0)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    def test_iloc_single_element(self):
        """df.iloc[row_idx, col_idx] = value"""
        result = self._run_backward_mutation_test(
            lambda df: df.iloc.__setitem__((0, 0), 999.0)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    def test_at_single_element(self):
        """df.at[row, col] = value"""
        result = self._run_backward_mutation_test(
            lambda df: df.at.__setitem__((0, "price"), 999.0)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    def test_iat_single_element(self):
        """df.iat[row_idx, col_idx] = value"""
        result = self._run_backward_mutation_test(
            lambda df: df.iat.__setitem__((0, 0), 999.0)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    def test_loc_slice_assignment(self):
        """df.loc[slice, col] = value (multiple rows)"""
        result = self._run_backward_mutation_test(
            lambda df: df.loc.__setitem__((slice(None, None), "price"), 0.0)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    # ----------------------------------------------------------------
    # inplace=True method operations
    # ----------------------------------------------------------------

    def test_fillna_inplace(self):
        """df.fillna(value, inplace=True)"""
        def mutate(df):
            df.loc.__setitem__((0, "price"), np.nan)  # introduce NaN first
            df.fillna(0.0, inplace=True)

        result = self._run_backward_mutation_test(mutate)
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    def test_replace_inplace(self):
        """df.replace(old, new, inplace=True)"""
        result = self._run_backward_mutation_test(
            lambda df: df.replace(10.0, 999.0, inplace=True)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    def test_rename_inplace(self):
        """df.rename(columns=..., inplace=True) — structural change."""
        def mutate(df):
            df.rename(columns={"price": "cost"}, inplace=True)

        # Cell A reads the "price" column; after rename it becomes "cost",
        # so the whole df is changed from A's perspective.
        result = self._run_backward_mutation_test(
            mutate,
            reads={"df"},
            column_reads={"df": {"price"}},
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    def test_drop_column_inplace(self):
        """df.drop(columns=..., inplace=True) — removes a column A read."""
        def mutate(df):
            df.drop(columns=["price"], inplace=True)

        result = self._run_backward_mutation_test(
            mutate,
            reads={"df"},
            column_reads={"df": {"price"}},
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    def test_drop_rows_inplace(self):
        """df.drop(index=..., inplace=True) — removes rows."""
        def mutate(df):
            df.drop(index=[0], inplace=True)

        result = self._run_backward_mutation_test(mutate)
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    def test_update(self):
        """df.update(other_df) — in-place update from another DataFrame."""
        def mutate(df):
            other = pd.DataFrame({"price": [999.0, 888.0, 777.0]})
            df.update(other)

        result = self._run_backward_mutation_test(mutate)
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"
        assert _get_backward_error(result).causer_cell == "a"

    # ----------------------------------------------------------------
    # Other in-place mechanisms
    # ----------------------------------------------------------------

    def test_sort_values_inplace(self):
        """df.sort_values(by=..., inplace=True)"""
        result = self._run_backward_mutation_test(
            lambda df: df.sort_values(by="price", ascending=False, inplace=True)
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    def test_set_index_inplace(self):
        """df.set_index(col, inplace=True) — moves a column into the index."""
        def mutate(df):
            df.set_index("name", inplace=True)

        result = self._run_backward_mutation_test(
            mutate,
            reads={"df"},
            column_reads={"df": {"price", "name"}},
        )
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    def test_clip_column_reassign(self):
        """df['col'] = df['col'].clip(lo, hi) — clip and reassign."""
        def mutate(df):
            df["price"] = df["price"].clip(15.0, 25.0)

        result = self._run_backward_mutation_test(mutate)
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    def test_numpy_array_copy_reassign(self):
        """Copy column to numpy array, mutate, reassign back."""
        def mutate(df):
            # to_numpy() returns read-only under CoW, so copy explicitly
            arr = df["price"].to_numpy(copy=True)
            arr[0] = 999.0
            df["price"] = arr

        result = self._run_backward_mutation_test(mutate)
        assert result.has_errors()
        assert _get_backward_error(result).cell_id == "b"

    # ----------------------------------------------------------------
    # List mutation (non-DataFrame, simple case)
    # ----------------------------------------------------------------

    def test_list_element_mutation(self):
        """elems[0] = 99 — simple in-place list mutation."""
        elems = [1, 2, 3]
        ns = {"elems": elems}

        # Cell A reads elems
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}a", ns, max_size_mb=None
        )
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns,
            tracking=make_tracking(reads={"elems"}, writes=set()),
        )
        assert not result_a.has_errors()

        # Cell B mutates elems in-place
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}b", ns, max_size_mb=None
        )
        elems[0] = 99
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns,
            tracking=make_tracking(reads={"elems"}, writes={"elems"}),
        )
        assert result_b.has_errors()
        assert _get_backward_error(result_b).cell_id == "b"
        assert _get_backward_error(result_b).causer_cell == "a"
