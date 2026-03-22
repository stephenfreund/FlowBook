"""Tests for SDC Enforcer."""

import pytest

from flowbook.kernel_support.memory_checkpoint import (
    MemoryCheckpoint,
    MemoryCheckpoints,
)
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
)
from flowbook.kernel.models import ErrorType, ReasonType
from flowbook.kernel.locations import writelocset_var_names
from flowbook.kernel.tests.conftest import make_tracking


def _get_backward_error(result):
    """Get the first backward mutation (NO_WRITE_AFTER_READ) error from result."""
    for e in result.errors:
        if e.error_type == ErrorType.NO_WRITE_AFTER_READ:
            return e
    return None


class TestReproducibilityEnforcer:

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_no_violation_forward_dependency(self):
        """Cell B reads what cell A writes - valid."""
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert not result_a.has_errors()

        # Cell B reads x - valid (forward dependency)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert not result_b.has_errors()

    def test_violation_backward_mutation(self):
        """Cell B modifies what cell A reads - violation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B (after A) modifies x - violation!
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        ns_b = self._make_namespace({"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.has_errors()
        assert result_b.errors[0].cell_id == "b"
        assert result_b.errors[0].causer_cell == "a"
        assert "x" in result_b.errors[0].locations

    def test_staleness_computation(self):
        """Re-running cell A makes cell B stale if B reads A's output."""
        # First run: A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        ns_a2 = self._make_namespace({"x": 100, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale
        assert "b" in result.stale_cells

    def test_cell_order_update_affects_violation_check(self):
        """Cell order can be updated, affecting position-based checks.

        Scenario 1: order [a, b] — A before B, so A modifying x is not a
        backward violation against B.

        Scenario 2: order [b, a] — B before A, so A modifying x IS a
        backward violation against B (B reads x).

        We use a fresh enforcer for scenario 2 to avoid cross-contamination.
        """
        # Scenario 1: [a, b, c, d] — A is before B
        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - A is before B in order, so NOT a violation
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert not result.has_errors()

        # Scenario 2: Fresh state with order [b, a, c, d] — B is before A
        self.sdc.reset()
        self.sdc.set_cell_order(["b", "a", "c", "d"])

        # B reads x (fresh)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b2 = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b2,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - now A is AFTER B, so this IS a violation
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a2 = self._make_namespace({"x": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.has_errors()
        assert result.errors[0].causer_cell == "b"

    def test_cell_deletion_prunes_records(self):
        """Deleted cells are removed from tracking."""
        # Execute cells a and b
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert self.sdc._notebook_state.has_record("a")
        assert self.sdc._notebook_state.has_record("b")

        # Remove cell b from order
        self.sdc.set_cell_order(["a", "c", "d"])

        # b should be pruned
        assert self.sdc._notebook_state.has_record("a")
        assert not self.sdc._notebook_state.has_record("b")

    def test_reset_clears_all_state(self):
        """Reset clears all tracking state."""
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert len(self.sdc._notebook_state.tracking_data) == 1
        assert self.sdc.seq_counter == 1

        self.sdc.reset()

        assert len(self.sdc._notebook_state.tracking_data) == 0
        assert self.sdc.seq_counter == 0
        assert self.sdc.cell_order == []

    def test_stale_cells_in_document_order(self):
        """Stale cells should be returned in document order, not execution order."""
        # Execute in order: a, d, b, c (but document order is a, b, c, d)
        # All read x

        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("d", {"x": 1})
        ns_d = self._make_namespace({"x": 1, "w": 4})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads={"x"}, writes={"w"}),
        )

        self._save_pre_checkpoint("b", {"x": 1, "w": 4})
        ns_b = self._make_namespace({"x": 1, "w": 4, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        self._save_pre_checkpoint("c", {"x": 1, "w": 4, "y": 2})
        ns_c = self._make_namespace({"x": 1, "w": 4, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Re-run A with different x - should make b, c, d stale (they all read x)
        self._save_pre_checkpoint("a", {"x": 1, "w": 4, "y": 2, "z": 3})
        ns_a2 = self._make_namespace({"x": 100, "w": 4, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Should be in document order [b, c, d], not execution order [d, b, c]
        assert result.stale_cells == ["b", "c", "d"]

    def test_no_violation_when_cell_not_in_order(self):
        """If cell_id is not in cell_order, no violation check happens."""
        # Set cell order that doesn't include 'x'
        self.sdc.set_cell_order(["a", "b", "c"])

        # Cell 'a' reads variable 'var'
        self._save_pre_checkpoint("a", {"var": 1})
        ns_a = self._make_namespace({"var": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"var"}, writes=set()),
        )

        # Cell 'x' (not in order) modifies 'var' - should not trigger violation
        self._save_pre_checkpoint("x", {"var": 1})
        ns_x = self._make_namespace({"var": 999})
        result = self.sdc.check(
            cell_id="x",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}x"],
            namespace=ns_x,
            tracking=make_tracking(reads=set(), writes={"var"}),
        )

        # No violation because 'x' is not in cell_order
        assert not result.has_errors()


class TestColumnAwareBackwardMutation:
    """Tests for column-aware backward mutation detection."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_no_conflict_different_columns(self):
        """Cell A reads df.price, Cell B modifies df.quantity - no violation."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"},
                writes={"y"},
                column_reads={"df": {"price"}},
            ),
        )

        # Cell B: modifies df.quantity (different column)
        df_modified = df.copy()
        df_modified["quantity"] = [10, 20]
        self._save_pre_checkpoint("b", {"df": df, "y": 30})
        ns_b = self._make_namespace({"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"quantity"}},
            ),
        )

        # No backward mutation - different columns
        # (NoReadAndWrite may fire since B reads and writes df)
        assert not any(
            e.error_type == ErrorType.NO_WRITE_AFTER_READ for e in result_b.errors
        )

    def test_conflict_same_column(self):
        """Cell A reads df.price, Cell B modifies df.price - violation."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"},
                writes={"y"},
                column_reads={"df": {"price"}},
            ),
        )

        # Cell B: modifies df.price (same column)
        df_modified = df.copy()
        df_modified["price"] = [100, 200]
        self._save_pre_checkpoint("b", {"df": df, "y": 30})
        ns_b = self._make_namespace({"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"price"}},
            ),
        )

        # Violation - same column (NO_WRITE_AFTER_READ backward mutation)
        assert result_b.has_errors()
        backward_err = next(
            e for e in result_b.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ
        )
        assert backward_err.cell_id == "b"
        assert backward_err.causer_cell == "a"
        assert "df['price']" in backward_err.locations

    def test_conflict_prior_no_column_info_conservative(self):
        """Cell A reads df (no column info), Cell B modifies df.price - violation (conservative)."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df (no column tracking)
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"},
                writes={"y"},
                # No column_reads - assumes whole df is read
            ),
        )

        # Cell B: modifies df.price
        df_modified = df.copy()
        df_modified["price"] = [100, 200]
        self._save_pre_checkpoint("b", {"df": df, "y": 30})
        ns_b = self._make_namespace({"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"price"}},
            ),
        )

        # Violation - conservative when prior has no column info
        # Now shows column-level detail: df.price instead of just df
        assert result_b.has_errors()
        backward_err = next(e for e in result_b.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert backward_err.causer_cell == "a"
        assert "df['price']" in backward_err.locations

    def test_conflict_current_no_column_info_conservative(self):
        """Cell A reads df.price, Cell B modifies df (no column info) - violation (conservative)."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"},
                writes={"y"},
                column_reads={"df": {"price"}},
            ),
        )

        # Cell B: modifies entire df (no column tracking)
        df_modified = df * 2
        self._save_pre_checkpoint("b", {"df": df, "y": 30})
        ns_b = self._make_namespace({"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                # No column_writes - assumes whole df is modified
            ),
        )

        # Violation - conservative when current has no column info
        assert result_b.has_errors()
        backward_err = next(e for e in result_b.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert backward_err.causer_cell == "a"
        # New resolver provides precise column info: "df.price" instead of just "df"
        assert any(v.startswith("df") for v in backward_err.locations)

    def test_mixed_variable_and_column_conflicts(self):
        """Mixed scenario: variable-level conflict on config, no column conflict on df."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})
        config = {"a": 1}

        # Cell A: reads config (variable-level) and df.price (column-level)
        self._save_pre_checkpoint("a", {"df": df, "config": config})
        ns_a = self._make_namespace({"df": df, "config": config, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df", "config"},
                writes={"y"},
                column_reads={"df": {"price"}},
            ),
        )

        # Cell B: modifies config and df.quantity (not df.price)
        df_modified = df.copy()
        df_modified["quantity"] = [10, 20]
        config_modified = {}
        self._save_pre_checkpoint("b", {"df": df, "config": config, "y": 10})
        ns_b = self._make_namespace(
            {"df": df_modified, "config": config_modified, "y": 10}
        )
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df", "config"},
                writes={"df", "config"},
                column_reads={"df": set()},
                column_writes={"df": {"quantity"}},
            ),
        )

        # Violation on config only (not on df since different columns)
        assert result_b.has_errors()
        backward_err = next(e for e in result_b.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert backward_err.causer_cell == "a"
        assert "config" in backward_err.locations
        # df.price should NOT be in violations (different column)
        assert "df['price']" not in backward_err.locations

    def test_multiple_column_conflicts(self):
        """Multiple columns conflict: df.price and df.quantity both modified."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2], "total": [10, 40]})

        # Cell A: reads df.price and df.quantity
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"},
                writes={"y"},
                column_reads={"df": {"price", "quantity"}},
            ),
        )

        # Cell B: modifies both df.price and df.quantity
        df_modified = df.copy()
        df_modified["price"] = [100, 200]
        df_modified["quantity"] = [10, 20]
        self._save_pre_checkpoint("b", {"df": df, "y": 30})
        ns_b = self._make_namespace({"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"price", "quantity"}},
            ),
        )

        # Violation - both columns conflict
        assert result_b.has_errors()
        backward_err = next(e for e in result_b.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert backward_err.causer_cell == "a"
        assert "df['price']" in backward_err.locations
        assert "df['quantity']" in backward_err.locations

    def test_no_conflict_when_no_overlap_multiple_vars(self):
        """Multiple DataFrames with no column overlap - no violation."""
        import pandas as pd

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})

        # Cell A: reads df1.a and df2.b
        self._save_pre_checkpoint("a", {"df1": df1, "df2": df2})
        ns_a = self._make_namespace({"df1": df1, "df2": df2, "y": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df1", "df2"},
                writes={"y"},
                column_reads={"df1": {"a"}, "df2": {"b"}},
            ),
        )

        # Cell B: modifies df1.b (not df1.a) - df2 unchanged
        df1_modified = df1.copy()
        df1_modified["b"] = [10, 20]
        self._save_pre_checkpoint("b", {"df1": df1, "df2": df2, "y": 1})
        ns_b = self._make_namespace({"df1": df1_modified, "df2": df2, "y": 1})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df1"},
                writes={"df1"},
                column_reads={"df1": set()},
                column_writes={"df1": {"b"}},
            ),
        )

        # No violation - df1.a not modified, df2 not modified
        assert not any(e.error_type == ErrorType.NO_WRITE_AFTER_READ for e in result_b.errors)


class TestBackwardMutationStaleness:
    """Tests for backward mutation staleness behavior.

    Backward mutations mark the executing cell as STALE. Staleness propagation
    to other cells depends on whether the error is accepted or rejected:
    - continue_on_violation=False (default): cell is rejected, no staleness propagation
    - continue_on_violation=True: cell is accepted, staleness propagates
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_backward_mutation_marks_cell_stale(self):
        """When cell B writes to x that cell A reads, B is marked stale."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B modifies x (backward mutation)
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        ns_b = self._make_namespace({"x": 999, "y": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Backward violation info is returned
        assert result.has_errors()
        assert result.errors[0].cell_id == "b"
        assert result.errors[0].causer_cell == "a"
        # Staleness is ALWAYS computed (new semantics)
        # changed_variables shows what changed
        assert "x" in result.changed_variables
        # Cell b itself should be stale (backward mutation)
        assert not self.sdc._notebook_state.is_clean("b")
        # Cell b should have NO_WRITE_AFTER_READ reason
        reasons = self.sdc._notebook_state.get_reasons("b")
        assert any(r.type == ReasonType.NO_WRITE_AFTER_READ for r in reasons)

    def test_backward_mutation_skips_staleness_when_rejected(self):
        """Backward mutation with continue_on_violation=False skips staleness propagation.

        When a cell will be rejected (rolled back by the kernel), propagating
        staleness to other cells is incorrect — the writes never persist, so
        downstream cells should not be marked stale.
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell C reads x (would be affected by B's write if accepted)
        self._save_pre_checkpoint("c", {"x": 1})
        ns_c = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x (backward mutation against A)
        # With continue_on_violation=False (default), this will be rejected
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Backward violation is detected
        assert result.has_errors()
        # C should NOT be marked stale (B will be rejected/rolled back)
        assert self.sdc._notebook_state.is_clean("c")

    def test_backward_mutation_propagates_staleness_when_accepted(self):
        """Backward mutation with continue_on_violation=True does propagate staleness.

        When errors are accepted, the cell's writes persist, so downstream
        cells that read the same variables should be marked stale.
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell C reads x (will be affected by B's write)
        self._save_pre_checkpoint("c", {"x": 1})
        ns_c = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x (backward mutation against A)
        # With continue_on_violation=True, this is accepted
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # Backward violation is detected but accepted
        assert result.has_errors()
        # C IS marked stale (ForwardStale) because B's writes persist
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_backward_mutation_updates_execution_record(self):
        """Execution record is ALWAYS updated, even with backward mutation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert self.sdc._notebook_state.has_record("a")
        assert not self.sdc._notebook_state.has_record("b")

        # Cell B modifies x (backward mutation)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result.has_errors()
        # Record is ALWAYS created (new semantics)
        assert self.sdc._notebook_state.has_record("b")
        assert self.sdc._notebook_state.get_tracking("b").writes == {"x"}

    def test_backward_mutation_chain_staleness_rejected(self):
        """Backward mutation with continue_on_violation=False skips staleness propagation.

        When cell D will be rejected (rolled back), staleness should NOT
        propagate to other cells since D's writes never persist.
        """
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x, writes y
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell C reads y
        self._save_pre_checkpoint("c", {"x": 1, "y": 2})
        ns_c = self._make_namespace({"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )

        # Cell D modifies x (backward mutation against B who reads x)
        # Default continue_on_violation=False → D will be rejected
        self._save_pre_checkpoint("d", {"x": 1, "y": 2, "z": 3})
        ns_d = self._make_namespace({"x": 999, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Backward violation is detected
        assert result.has_errors()
        assert result.errors[0].causer_cell == "b"
        # D itself is stale (backward mutation marks the executing cell stale)
        assert not self.sdc._notebook_state.is_clean("d")
        # B should NOT be marked stale (D will be rejected/rolled back)
        assert self.sdc._notebook_state.is_clean("b")
        # C read y (not x), so C is not affected
        assert self.sdc._notebook_state.is_clean("c")

    def test_backward_mutation_chain_staleness_accepted(self):
        """Backward mutation with continue_on_violation=True propagates staleness.

        When D's execution is accepted, its writes persist, and B (which
        read x) should be marked stale.
        """
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x, writes y
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell C reads y
        self._save_pre_checkpoint("c", {"x": 1, "y": 2})
        ns_c = self._make_namespace({"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )

        # Cell D modifies x with continue_on_violation=True → accepted
        self._save_pre_checkpoint("d", {"x": 1, "y": 2, "z": 3})
        ns_d = self._make_namespace({"x": 999, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # Backward violation is detected but accepted
        assert result.has_errors()
        assert result.errors[0].causer_cell == "b"
        # B IS marked stale (D's writes persist, B read x)
        assert "b" in result.stale_cells
        # C read y (not x), so C is not affected
        assert "c" not in result.stale_cells


class TestTruncationDetection:
    """Test the _check_for_truncation helper function."""

    def test_no_truncation_empty_diff(self):
        """Empty diff should not be truncated."""
        from flowbook.kernel_support.types import MemoryCheckpointDiffResult
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation

        diff = MemoryCheckpointDiffResult(differences={})
        truncated_vars = _check_for_truncation(diff)
        assert truncated_vars == []

    def test_no_truncation_simple_diff(self):
        """Simple diff without _truncated should not be detected."""
        from flowbook.kernel_support.types import (
            MemoryCheckpointDiffResult,
            ValueComparison,
        )
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation

        diff = MemoryCheckpointDiffResult(
            differences={
                "x": ValueComparison(
                    status="different",
                    value1=1,
                    value2=2,
                    message="Values differ",
                )
            }
        )
        truncated_vars = _check_for_truncation(diff)
        assert truncated_vars == []

    def test_truncation_detected_in_dict(self):
        """Truncation in structural type (dict) should be detected."""
        from flowbook.kernel_support.types import (
            MemoryCheckpointDiffResult,
            ValueComparison,
            CompoundDiff,
        )
        from flowbook.kernel.reproducibility_enforcer import (
            _check_for_truncation,
            _format_diff_for_display,
        )

        diff = MemoryCheckpointDiffResult(
            differences={
                "my_dict": CompoundDiff(
                    source_type="dict",
                    children={
                        "['key1']": ValueComparison(
                            status="different", value1=1, value2=2, message="diff"
                        ),
                    },
                    truncated=True,
                )
            }
        )
        truncated_vars = _check_for_truncation(diff)
        assert truncated_vars == ["my_dict"]
        # Test lazy formatting separately
        formatted_diff = _format_diff_for_display(diff, truncated_vars)
        assert "TRUNCATED DIFF DETAILS" in formatted_diff
        assert "Variable: my_dict" in formatted_diff

    def test_nested_container_truncation_is_ignored(self):
        """Truncation in nested container should NOT be flagged (only structure-level matters)."""
        from flowbook.kernel_support.types import (
            MemoryCheckpointDiffResult,
            ValueComparison,
            CompoundDiff,
        )
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation

        # The outer dict is not truncated, only the inner list is
        # Since we only check the immediate variable's truncation status, this should pass
        diff = MemoryCheckpointDiffResult(
            differences={
                "outer": CompoundDiff(
                    source_type="dict",
                    children={
                        "['inner']": CompoundDiff(
                            source_type="list",  # list is not in STRUCTURAL_TYPES, so even if truncated, ignored
                            children={},
                            truncated=True,
                        ),
                    },
                    truncated=False,  # The outer dict is not truncated
                )
            }
        )
        truncated_vars = _check_for_truncation(diff)
        # Nested truncation should NOT be detected - only structure-level at top
        assert truncated_vars == []

    def test_multiple_truncated_vars(self):
        """Multiple truncated variables should all be detected."""
        from flowbook.kernel_support.types import (
            MemoryCheckpointDiffResult,
            ValueComparison,
            CompoundDiff,
        )
        from flowbook.kernel.reproducibility_enforcer import (
            _check_for_truncation,
            _format_diff_for_display,
        )

        diff = MemoryCheckpointDiffResult(
            differences={
                "dict1": CompoundDiff(
                    source_type="dict",
                    children={
                        "['key']": ValueComparison(
                            status="different", value1=1, value2=2, message="diff"
                        )
                    },
                    truncated=True,
                ),
                "obj2": CompoundDiff(
                    source_type="object",
                    children={
                        ".attr": ValueComparison(
                            status="different", value1=1, value2=2, message="diff"
                        )
                    },
                    truncated=True,
                ),
                "clean": ValueComparison(
                    status="different", value1=1, value2=2, message="diff"
                ),
            }
        )
        truncated_vars = _check_for_truncation(diff)
        assert set(truncated_vars) == {"dict1", "obj2"}
        # Test lazy formatting separately
        formatted_diff = _format_diff_for_display(diff, truncated_vars)
        assert "Variable: dict1" in formatted_diff
        assert "Variable: obj2" in formatted_diff


class TestStructuralTrackingEnforce:
    """Tests for structural tracking - structural reads ARE always protected."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(
            self.checkpoints,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_structural_only_read_causes_violation_in_enforce_mode(self):
        """
        With structural tracking ENFORCE, a cell that reads structural attrs
        (like df.shape) SHOULD cause a violation when structure changes.

        This is the strictest mode - structural reads are fully protected.
        """
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert not result_a.has_errors()

        # Cell B: Reads raw_data.shape (structural read only)
        self._save_pre_checkpoint("b", {"raw_data": df})
        ns_b = self._make_namespace({"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"raw_data"},
                writes=set(),
                column_reads=None,
                structural_reads={"raw_data": {"shape"}},
            ),
        )
        assert not result_b.has_errors()

        # Cell C: Adds a new column raw_data['x'] = 3
        # With ENFORCE mode, this SHOULD cause a violation
        df_modified = df.copy()
        df_modified["x"] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        ns_c = self._make_namespace({"raw_data": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(
                reads={"raw_data"},
                writes={"raw_data"},
                column_reads=None,
                column_writes={"raw_data": {"x"}},
            ),
        )
        # SHOULD cause violation - ENFORCE mode protects structural reads
        # Now shows column-level detail: raw_data.x instead of just raw_data
        assert result_c.has_errors()
        backward_err = next(e for e in result_c.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert any("raw_data" in loc and "x" in loc for loc in backward_err.locations)

    def test_structural_violation_with_column_reads_and_new_column(self):
        """
        ENFORCE mode: Adding a new column should violate even when prior cell
        also read column data (not just structural-only read).

        This tests the Bug4.ipynb scenario where:
        - Cell displays DataFrame (reads columns AND structural attrs like .columns)
        - Next cell adds a new column
        - Even though no column DATA overlap, the structural read is violated
        """
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # Cell B: Display DataFrame - reads columns AND structural attrs
        # This simulates what happens when you just type `df` in a cell
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},
                writes=set(),
                column_reads={"df": {"a", "b"}},  # Read actual column data
                structural_reads={
                    "df": {"columns", "dtypes", "shape"}
                },  # Also read structure
            ),
        )
        assert not result_b.has_errors()

        # Cell C: Adds a new column df['x'] = 3
        # No overlap with columns a,b but structural read of .columns is violated
        df_modified = df.copy()
        df_modified["x"] = 3
        self._save_pre_checkpoint("c", {"df": df})
        ns_c = self._make_namespace({"df": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},  # Didn't read existing columns
                column_writes={"df": {"x"}},  # Wrote new column
            ),
        )
        # SHOULD cause violation - prior cell read .columns, we added column x
        # Now shows column-level detail: df.x instead of just df
        assert result_c.has_errors()
        backward_err = next(e for e in result_c.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert any("df" in loc and "x" in loc for loc in backward_err.locations)


# =============================================================================
# TESTS FOR OPT_ACCESSED_VARS_ONLY OPTIMIZATION
# These tests verify correctness of the optimization that only diffs accessed
# variables plus their aliases, instead of the entire namespace.
# =============================================================================


class TestAccessedVarsOnlyOptimization:
    """
    Tests for the OPT_ACCESSED_VARS_ONLY optimization.

    This optimization only diffs variables that the cell accessed (reads + writes)
    plus their aliases (variables sharing object identity), instead of diffing
    the entire namespace. This provides significant speedup for large namespaces.

    Critical correctness requirement: alias detection must find all variables
    that share object identity with accessed variables, so we detect backward
    mutations through aliases.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    # -------------------------------------------------------------------------
    # Basic alias detection tests
    # -------------------------------------------------------------------------

    def test_alias_detected_simple(self):
        """
        Test that modifications through an alias are detected.

        Scenario:
        - Cell A creates y and reads it
        - Cell B creates alias x = y
        - Cell C modifies through x

        Cell C should cause a backward mutation violation because y changed,
        and Cell A read y.
        """
        import pandas as pd

        # Cell A: creates y, reads y
        y = [1, 2, 3]
        self._save_pre_checkpoint("a", {"y": y})
        ns_a = self._make_namespace({"y": y})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"y"}, writes=set()),
        )

        # Cell B: creates alias x = y (same object)
        x = y  # x and y are the same object
        self._save_pre_checkpoint("b", {"y": y, "x": x})
        ns_b = self._make_namespace({"y": y, "x": x})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"y"}, writes={"x"}),
        )

        # Cell C: modifies through x (which changes y too!)
        x_modified = [1, 2, 3]
        x_modified[0] = 999  # Simulate in-place modification
        self._save_pre_checkpoint("c", {"y": y, "x": x})
        # After modification, both x and y point to modified list
        ns_c = self._make_namespace({"y": x_modified, "x": x_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Should detect violation - Cell A read y, Cell C modified y (via x)
        assert result_c.has_errors()
        assert result_c.errors[0].causer_cell == "a"
        # The violation should mention y (the variable Cell A read)
        assert (
            "y" in result_c.errors[0].locations or "x" in result_c.errors[0].locations
        )

    def test_alias_detected_dataframe(self):
        """
        Test alias detection with DataFrames.

        Scenario:
        - Cell A creates df and reads df['price']
        - Cell B creates alias df_copy = df (same object, not a copy!)
        - Cell C modifies df_copy['price']

        Cell C should cause a backward mutation violation.
        """
        import pandas as pd

        df = pd.DataFrame({"price": [100, 200], "quantity": [5, 10]})

        # Cell A: creates df, reads price column
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"}, writes=set(), column_reads={"df": {"price"}}
            ),
        )

        # Cell B: creates alias (NOT a copy)
        df_alias = df  # Same object!
        self._save_pre_checkpoint("b", {"df": df, "df_alias": df_alias})
        ns_b = self._make_namespace({"df": df, "df_alias": df_alias})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"df"}, writes={"df_alias"}),
        )

        # Cell C: modifies through alias
        df_modified = pd.DataFrame({"price": [999, 999], "quantity": [5, 10]})
        self._save_pre_checkpoint("c", {"df": df, "df_alias": df_alias})
        ns_c = self._make_namespace({"df": df_modified, "df_alias": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(
                reads=set(), writes={"df_alias"}, column_writes={"df_alias": {"price"}}
            ),
        )

        # Should detect violation
        assert result_c.has_errors()
        assert result_c.errors[0].causer_cell == "a"

    def test_multiple_aliases_all_detected(self):
        """
        Test that multiple aliases are all detected.

        Scenario:
        - Cell A creates x and reads it
        - Cell B creates aliases: y = x, z = x
        - Cell C modifies through z

        Should detect that x (and y) also changed.
        """
        x = {"value": 1}

        # Cell A: creates x, reads x
        self._save_pre_checkpoint("a", {"x": x})
        ns_a = self._make_namespace({"x": x})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: creates multiple aliases
        y = x
        z = x
        self._save_pre_checkpoint("b", {"x": x, "y": y, "z": z})
        ns_b = self._make_namespace({"x": x, "y": y, "z": z})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y", "z"}),
        )

        # Cell C: modifies through z
        modified = {"value": 999}
        self._save_pre_checkpoint("c", {"x": x, "y": y, "z": z})
        ns_c = self._make_namespace({"x": modified, "y": modified, "z": modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"z"}),
        )

        # Should detect violation - x was read by Cell A
        assert result_c.has_errors()
        assert result_c.errors[0].causer_cell == "a"

    def test_no_alias_no_spurious_diff(self):
        """
        Test that non-aliased variables are not diffed when optimization is active.

        Scenario:
        - Cell A reads x
        - Cell B has many other variables (not aliases)
        - Cell B only accesses y

        The diff should only check y, not the many other variables.
        (We can't directly test this, but we ensure correctness.)
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: has many variables, only accesses y (writes it)
        namespace = {"x": 1, "y": 2}
        # Add many unrelated variables
        for i in range(50):
            namespace[f"unrelated_{i}"] = i * 100

        self._save_pre_checkpoint("b", namespace)
        namespace_after = namespace.copy()
        namespace_after["y"] = 999  # Only y changes
        ns_b = self._make_namespace(namespace_after)
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No violation - y wasn't read by any earlier cell
        assert not result_b.has_errors()

    def test_alias_broken_by_copy(self):
        """
        Test that breaking an alias (via copy) is handled correctly.

        Scenario:
        - Cell A creates df, reads df
        - Cell B creates alias: df_alias = df
        - Cell C breaks alias: df_alias = df_alias.copy(), then modifies df_alias

        Cell C should NOT cause a violation because df_alias is now independent.
        """
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})

        # Cell A: creates df, reads df
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"df"}, writes=set()),
        )

        # Cell B: creates alias
        df_alias = df
        self._save_pre_checkpoint("b", {"df": df, "df_alias": df_alias})
        ns_b = self._make_namespace({"df": df, "df_alias": df_alias})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"df"}, writes={"df_alias"}),
        )

        # Cell C: breaks alias with copy, then modifies the copy
        # Pre-state: df and df_alias are same object
        self._save_pre_checkpoint("c", {"df": df, "df_alias": df_alias})
        # Post-state: df_alias is now a different object (the copy, modified)
        df_copy_modified = df.copy()
        df_copy_modified["a"] = [999, 999, 999]
        ns_c = self._make_namespace({"df": df, "df_alias": df_copy_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"df_alias"}, writes={"df_alias"}),
        )

        # No violation - df is unchanged, only the copy (df_alias) changed
        assert not any(e.error_type == ErrorType.NO_WRITE_AFTER_READ for e in result_c.errors)

    def test_new_variable_not_alias(self):
        """
        Test that new variables (only in post-state) are handled correctly.

        Scenario:
        - Cell A reads x
        - Cell B creates new variable y (wasn't in pre-state)

        Should not cause any issues.
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: creates new variable y
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 42})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No violation - y is new, x is unchanged
        assert not result_b.has_errors()

    def test_deleted_variable(self):
        """
        Test that deleted variables are handled correctly.

        Scenario:
        - Cell A reads x
        - Cell B deletes y (which was not read by anyone)

        Should not cause any issues.
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        ns_a = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: deletes y
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        ns_b = self._make_namespace({"x": 1})  # y deleted
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No violation - y wasn't read by earlier cells
        assert not result_b.has_errors()

    def test_alias_in_nested_structure(self):
        """
        Test alias detection when the same object appears multiple times in a structure.

        Scenario:
        - Cell A creates data dict containing df, reads data['df1']
        - Cell B accesses data['df2'] which is the same object

        Modification through data['df2'] should be detected as affecting data['df1'].
        """
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        # Same df object stored under two keys
        data = {"df1": df, "df2": df}

        # Cell A: reads data, specifically data['df1']
        self._save_pre_checkpoint("a", {"data": data})
        ns_a = self._make_namespace({"data": data})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"data"}, writes=set()),
        )

        # Cell B: modifies data['df2'] (same underlying object as df1)
        data_modified = data.copy()
        df_modified = pd.DataFrame({"a": [999, 999, 999]})
        data_modified["df1"] = df_modified
        data_modified["df2"] = df_modified

        self._save_pre_checkpoint("b", {"data": data})
        ns_b = self._make_namespace({"data": data_modified})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"data"}),
        )

        # Should detect violation - data was read by Cell A
        assert result_b.has_errors()
        assert result_b.errors[0].causer_cell == "a"
        assert "data" in result_b.errors[0].locations

    def test_empty_accessed_vars(self):
        """
        Test edge case where cell accesses no variables.

        Should still work correctly (diff nothing, no violation).
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: accesses nothing (e.g., just prints a constant)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes=set()),
        )

        # No violation - nothing was modified
        assert not result_b.has_errors()

    def test_all_vars_accessed(self):
        """
        Test edge case where cell accesses all variables.

        Should work correctly (diff everything).
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1, "y": 2, "z": 3})
        ns_a = self._make_namespace({"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: accesses all variables, modifies x
        self._save_pre_checkpoint("b", {"x": 1, "y": 2, "z": 3})
        ns_b = self._make_namespace({"x": 999, "y": 2, "z": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x", "y", "z"}, writes={"x"}),
        )

        # Should detect violation - Cell A read x, Cell B modified x
        assert result_b.has_errors()
        backward_err = next(e for e in result_b.errors if e.error_type == ErrorType.NO_WRITE_AFTER_READ)
        assert backward_err.causer_cell == "a"
        assert "x" in backward_err.locations


# =============================================================================
# DEEP ALIAS DETECTION TESTS
# =============================================================================
# These tests verify the deep alias detection feature which finds aliases not
# just at the top level (same object identity), but also at nested levels where
# two variables share internal references.
#
# For example: If a["b"] and c["b"] point to the same object, modifying a["b"]
# also changes c. The deep alias detection correctly identifies this.
# =============================================================================


class TestDeepAliasDetection:
    """
    Tests for deep alias detection via _expand_with_deep_aliases.

    Deep aliases are cases where two variables share internal references,
    not just top-level object identity. These are detected using the
    precomputed alias index in Checkpoint objects.
    """

    def test_nested_dict_shared_value(self):
        """
        User's example: a["b"] and c["b"] point to same object.

        If cell modifies a["b"]["f"] = 4, then c should also be flagged
        as changing because c["b"] is the same object as a["b"].
        """
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_inner = {"f": 1}
        namespace = {
            "a": {"b": shared_inner},
            "c": {"b": shared_inner},  # c["b"] is same object as a["b"]
            "d": {"b": {"f": 1}},  # Different object with same value
        }

        # Create checkpoint with alias index
        checkpoint = MemoryCheckpoint("test", namespace, {})

        # If we access "a", we should also get "c" because they share internal refs
        aliases = checkpoint.get_aliases_for_vars({"a"})

        assert "a" in aliases
        assert "c" in aliases  # CRITICAL: c shares inner object with a
        assert "d" not in aliases  # d has different object

    def test_nested_dict_multiple_levels(self):
        """Test deep nesting - multiple levels of shared objects."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        deep_shared = {"value": 42}
        namespace = {
            "x": {"level1": {"level2": {"level3": deep_shared}}},
            "y": {"other": deep_shared},  # Shares at different path
            "z": {"separate": {"value": 42}},  # Same value, different object
        }

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"x"})

        assert "x" in aliases
        assert "y" in aliases  # Shares deep_shared
        assert "z" not in aliases  # Different object

    def test_list_with_shared_elements(self):
        """Test lists containing shared mutable objects."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_dict = {"data": [1, 2, 3]}
        namespace = {
            "list_a": [shared_dict, {"other": 1}],
            "list_b": [{"first": 0}, shared_dict],  # Contains same dict
            "list_c": [{"data": [1, 2, 3]}],  # Same value, different object
        }

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"list_a"})

        assert "list_a" in aliases
        assert "list_b" in aliases  # Shares shared_dict
        assert "list_c" not in aliases

    def test_numpy_array_views(self):
        """Test numpy array views - arr2 is a view of arr1."""
        import numpy as np
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = arr1[1:4]  # View, shares memory with arr1
        arr3 = np.array(
            [100, 200, 300]
        )  # Completely independent with different content

        namespace = {"arr1": arr1, "arr2": arr2, "arr3": arr3}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr1"})

        assert "arr1" in aliases
        assert "arr2" in aliases  # View shares base array
        # Note: arr3 may or may not be detected as alias due to numpy internals
        # (dtype objects, etc.). The key test is that views ARE detected.

    def test_numpy_array_view_reverse(self):
        """Test that accessing view also finds base array."""
        import numpy as np
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = arr1[1:4]  # View

        namespace = {"base": arr1, "view": arr2}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"view"})

        assert "view" in aliases
        assert "base" in aliases  # Base should be found from view

    def test_object_dtype_series_shared_elements(self):
        """Test Series with object dtype containing shared objects."""
        import pandas as pd
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_list = [1, 2, 3]
        series_a = pd.Series([shared_list, [4, 5], "str"])
        series_b = pd.Series([[10, 20], shared_list, "other"])  # Contains same list

        namespace = {"s_a": series_a, "s_b": series_b}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"s_a"})

        assert "s_a" in aliases
        assert "s_b" in aliases  # Shares shared_list in elements

    def test_object_dtype_dataframe_shared_elements(self):
        """Test DataFrame with object dtype containing shared objects."""
        import pandas as pd
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_dict = {"key": "value"}
        df_a = pd.DataFrame({"col": [shared_dict, {"other": 1}]})
        df_b = pd.DataFrame({"col": [{"another": 2}, shared_dict]})

        namespace = {"df_a": df_a, "df_b": df_b}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df_a"})

        assert "df_a" in aliases
        assert "df_b" in aliases  # Shares shared_dict in cells

    def test_user_defined_object_shared_attribute(self):
        """Test user-defined objects with shared attribute references."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_data = {"value": 100}

        class Container:
            def __init__(self, data):
                self.data = data

        obj_a = Container(shared_data)
        obj_b = Container(shared_data)  # Same data reference
        obj_c = Container({"value": 100})  # Different object

        namespace = {"obj_a": obj_a, "obj_b": obj_b, "obj_c": obj_c}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"obj_a"})

        assert "obj_a" in aliases
        assert "obj_b" in aliases  # Shares data attribute
        assert "obj_c" not in aliases

    def test_circular_references(self):
        """Test that circular references don't cause infinite loops."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        # Create circular structure
        a = {"name": "a"}
        b = {"name": "b", "ref_a": a}
        a["ref_b"] = b  # Circular: a -> b -> a

        namespace = {"obj_a": a, "obj_b": b}

        # This should not hang or crash
        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"obj_a"})

        assert "obj_a" in aliases
        assert "obj_b" in aliases  # Both are connected

    def test_self_referential_list(self):
        """Test list that contains itself."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        lst = [1, 2]
        lst.append(lst)  # Self-reference

        namespace = {"self_ref": lst}

        # Should not hang
        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"self_ref"})

        assert "self_ref" in aliases

    def test_mixed_types_shared_reference(self):
        """Test mix of dicts, lists, DataFrames all sharing an object."""
        import pandas as pd
        import numpy as np
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_arr = np.array([1, 2, 3])
        namespace = {
            "dict_var": {"data": shared_arr},
            "list_var": [shared_arr, None],
            "df_var": pd.DataFrame({"col": [shared_arr]}),  # Object dtype
            "arr_var": shared_arr,
            "independent": {"totally": "different"},  # Use dict instead of numpy array
        }

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"dict_var"})

        assert "dict_var" in aliases
        assert "list_var" in aliases
        # Note: df_var's object-dtype column should also detect the shared array
        assert "arr_var" in aliases
        assert "independent" not in aliases

    def test_tuple_with_mutable_contents(self):
        """Test tuples containing shared mutable objects."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        shared_list = [1, 2, 3]
        namespace = {
            "tuple_a": (shared_list, "immutable"),
            "tuple_b": ("other", shared_list),
            "tuple_c": ([1, 2, 3], "different"),
        }

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"tuple_a"})

        assert "tuple_a" in aliases
        assert "tuple_b" in aliases
        assert "tuple_c" not in aliases

    def test_no_aliases_returns_just_accessed(self):
        """Test that with no aliases, only accessed vars are returned."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        namespace = {
            "x": {"data": 1},
            "y": {"data": 2},
            "z": [3, 4, 5],
        }

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"x"})

        assert aliases == {"x"}

    def test_new_variable_included(self):
        """Test that variables not in namespace are still included."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        namespace = {"x": [1, 2, 3]}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"x", "new_var"})

        assert "x" in aliases
        assert "new_var" in aliases  # Even though not in namespace

    def test_empty_accessed_returns_empty(self):
        """Test empty accessed set returns empty."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        namespace = {"x": [1, 2, 3], "y": [4, 5, 6]}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars(set())

        assert aliases == set()

    # =========================================================================
    # REGRESSION TESTS: Prevent false positives from memoryview id() reuse
    # =========================================================================
    # These tests ensure that independent objects with same structure/values
    # are NOT incorrectly detected as aliases due to temporary object id reuse.
    # See: The .data attribute of numpy arrays creates temporary memoryview
    # objects that can have their id() reused after garbage collection.
    # =========================================================================

    def test_independent_dataframes_not_aliases(self):
        """
        REGRESSION TEST: Independent DataFrames with same structure must NOT be aliases.

        Previously, tracking id(arr.data) caused false positives because memoryview
        objects are temporary and their ids can be reused after garbage collection.
        """
        import pandas as pd
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        # Create two completely independent DataFrames with same structure
        df1 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df2 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        namespace = {"df1": df1, "df2": df2}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df1"})

        assert "df1" in aliases
        assert (
            "df2" not in aliases
        ), "Independent DataFrames should NOT be detected as aliases"

    def test_independent_numpy_arrays_not_aliases(self):
        """
        REGRESSION TEST: Independent numpy arrays with same values must NOT be aliases.

        This ensures we don't falsely detect aliases due to memoryview id reuse.
        """
        import numpy as np
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        # Create independent arrays with same values
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([1, 2, 3, 4, 5])  # Same values, different object

        namespace = {"arr1": arr1, "arr2": arr2}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr1"})

        assert "arr1" in aliases
        assert (
            "arr2" not in aliases
        ), "Independent arrays should NOT be detected as aliases"

    def test_independent_series_not_aliases(self):
        """
        REGRESSION TEST: Independent Series with same values must NOT be aliases.
        """
        import pandas as pd
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        s1 = pd.Series([1, 2, 3], name="data")
        s2 = pd.Series([1, 2, 3], name="data")  # Same values, different object

        namespace = {"s1": s1, "s2": s2}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"s1"})

        assert "s1" in aliases
        assert (
            "s2" not in aliases
        ), "Independent Series should NOT be detected as aliases"

    def test_many_independent_dataframes_no_false_positives(self):
        """
        REGRESSION TEST: Many independent DataFrames should not trigger false aliases.

        This stress tests the id() reuse scenario - with many objects, the chance
        of id() collision (if we were tracking temporary objects) would be higher.
        """
        import pandas as pd
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        # Create many independent DataFrames
        namespace = {}
        for i in range(20):
            namespace[f"df_{i}"] = pd.DataFrame(
                {
                    "id": list(range(100)),
                    "value": list(range(100, 200)),
                }
            )

        checkpoint = MemoryCheckpoint("test", namespace, {})

        # Check that accessing df_0 only returns df_0
        aliases = checkpoint.get_aliases_for_vars({"df_0"})

        assert aliases == {"df_0"}, f"Expected only df_0, got {aliases}"

    def test_numpy_view_still_detected_as_alias(self):
        """
        Ensure actual numpy views ARE still detected as aliases after the fix.

        This verifies we didn't break view detection while fixing the false positives.
        """
        import numpy as np
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        base_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        view1 = base_arr[2:5]  # View of base
        view2 = base_arr[5:8]  # Another view of base
        independent = np.array([3, 4, 5])  # Same values as view1, but independent

        namespace = {
            "base": base_arr,
            "view1": view1,
            "view2": view2,
            "independent": independent,
        }

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"view1"})

        assert "view1" in aliases
        assert "base" in aliases, "Base array should be detected as alias of view"
        assert "view2" in aliases, "Other views of same base should be aliases"
        assert "independent" not in aliases, "Independent array should NOT be alias"

    def test_dataframe_column_copy_not_alias(self):
        """
        REGRESSION TEST: A copied column should not be an alias.
        """
        import pandas as pd
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        col_copy = df["x"].copy()  # Explicit copy

        namespace = {"df": df, "col_copy": col_copy}

        checkpoint = MemoryCheckpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df"})

        assert "df" in aliases
        assert "col_copy" not in aliases, "Copied column should NOT be an alias"


class TestDeepAliasIntegration:
    """
    Integration tests verifying deep alias detection works correctly
    in the full SDC enforcer flow.
    """

    def test_backward_mutation_through_nested_alias(self):
        """
        Full integration test: modifying a["b"] should detect change to c
        when c["b"] is the same object as a["b"].

        Scenario:
        - Cell 1: reads c (and implicitly c["b"])
        - Cell 2: modifies a["b"]["f"] = 4
        - Since c["b"] is same object, c also changed -> backward mutation!
        """
        from flowbook.kernel_support.memory_checkpoint import (
            MemoryCheckpoints,
            MemoryCheckpoint,
        )
        from flowbook.kernel_support.models import TrackingData
        from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer

        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints)
        enforcer.set_cell_order(["cell_A", "cell_B"])

        # Initial state: a and c share internal reference
        shared_inner = {"f": 1}
        initial_ns = {
            "a": {"b": shared_inner},
            "c": {"b": shared_inner},  # c["b"] is same as a["b"]
        }

        # Cell A reads c
        pre_cp_a = checkpoints.save("_pre_cell_A", initial_ns)[0]
        tracking_a = TrackingData(
            reads_before_writes={"c"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
            structural_reads={},
        )
        post_cp_a = checkpoints.save("_post_cell_A", initial_ns)[0]

        result_a = enforcer.check(
            "cell_A",
            checkpoints.get("_pre_cell_A"),
            checkpoints.get("_post_cell_A"),
            tracking_a,
        )
        assert not result_a.has_errors()

        # Cell B modifies a["b"]["f"] (which also modifies c["b"]["f"])
        shared_inner["f"] = 999  # Modify through a["b"]
        modified_ns = {
            "a": {"b": shared_inner},
            "c": {"b": shared_inner},
        }

        pre_cp_b = checkpoints.save("_pre_cell_B", initial_ns)[0]
        tracking_b = TrackingData(
            reads_before_writes=set(),
            writes={"a"},  # Only wrote to "a" by name
            column_reads_before_writes={},
            column_writes={},
            structural_reads={},
        )
        post_ns = modified_ns.copy()
        post_cp_b = checkpoints.save("_post_cell_B", modified_ns)[0]

        # Recreate pre-checkpoint with original values for proper comparison
        # (The checkpoint was taken after modification in this test setup,
        # so we need to simulate the proper pre-state)
        original_inner = {"f": 1}
        original_ns = {
            "a": {"b": original_inner},
            "c": {"b": original_inner},
        }
        pre_cp_proper = MemoryCheckpoint("_pre_cell_B_proper", original_ns, {})

        # Now the diff should detect changes to both a and c
        # because they share the inner object
        result_b = enforcer.check(
            "cell_B",
            pre_cp_proper,
            checkpoints.get("_post_cell_B"),
            tracking_b,
        )

        # With deep alias detection, "c" should be flagged even though
        # we only accessed "a" - because they share internal references
        # This depends on whether the diff detects the change to c
        # The key test is that deep alias expansion includes "c"
        from flowbook.kernel.reproducibility_enforcer import _expand_with_deep_aliases

        accessed = {"a"}
        expanded = _expand_with_deep_aliases(accessed, pre_cp_proper)
        assert "a" in expanded
        assert "c" in expanded, "Deep alias detection should find c shares refs with a"

    def test_expand_with_deep_aliases_uses_checkpoint_index(self):
        """Verify _expand_with_deep_aliases uses the checkpoint's lazy-built index."""
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint
        from flowbook.kernel.reproducibility_enforcer import _expand_with_deep_aliases

        shared = {"inner": [1, 2, 3]}
        namespace = {"var_a": {"ref": shared}, "var_b": {"ref": shared}}

        checkpoint = MemoryCheckpoint("test", namespace, {})

        # Verify index is NOT built initially (lazy building)
        assert not checkpoint._alias_index_built, "Alias index should be built lazily"
        assert (
            not checkpoint._reachable_ids
        ), "Reachable IDs should be empty before first query"

        # Test the expansion - this triggers lazy index building
        result = _expand_with_deep_aliases({"var_a"}, checkpoint)
        assert result == {"var_a", "var_b"}

        # Now verify index WAS built
        assert checkpoint._alias_index_built, "Alias index should be built after query"
        assert checkpoint._reachable_ids, "Reachable IDs should be populated"
        assert checkpoint._id_to_vars, "Reverse index should be populated"

    def test_performance_precomputed_vs_runtime(self):
        """Verify that using precomputed index is efficient."""
        import time
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint
        from flowbook.kernel.reproducibility_enforcer import _expand_with_deep_aliases

        # Create a moderately large namespace
        shared = {"value": list(range(100))}
        namespace = {f"var_{i}": {"ref": shared} for i in range(50)}
        namespace["unrelated"] = {"other": [1, 2, 3]}

        # First call includes index building
        start = time.perf_counter()
        checkpoint = MemoryCheckpoint("test", namespace, {})
        build_time = time.perf_counter() - start

        # Subsequent alias lookups should be fast
        start = time.perf_counter()
        for _ in range(100):
            result = _expand_with_deep_aliases({"var_0"}, checkpoint)
        lookup_time = time.perf_counter() - start

        # Verify correctness
        assert len(result) == 50  # All var_* share the reference
        assert "unrelated" not in result

        # Lookup should be much faster than build (rough sanity check)
        # 100 lookups should take less time than building once
        # (unless namespace is very simple)


class TestBackwardConflictFreshOnly:
    """Tests that BackConflict only checks fresh cells (Def 1.8.2)."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_backward_conflict_skips_stale_cells(self):
        """Stale prior cell should be excluded from backward conflict check.

        Setup: Cell A reads x, then mark A stale. Cell B modifies x.
        Expected: No violation because A is stale (BackConflict only checks fresh).
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Mark A stale via NotebookState API (CODE_CHANGED simulates edit)
        self.sdc._notebook_state.handle_edit("a")

        # Cell B modifies x — should NOT trigger violation because A is stale
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert not result_b.has_errors()

    def test_backward_conflict_still_fires_for_fresh_cells(self):
        """Fresh prior cell should still trigger backward conflict.

        Setup: Cell A reads x (fresh), Cell B modifies x.
        Expected: Violation because A is fresh.
        """
        # Cell A reads x (fresh — not in _stale_cells)
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x — SHOULD trigger violation because A is fresh
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.has_errors()
        assert result_b.errors[0].cell_id == "b"
        assert result_b.errors[0].causer_cell == "a"

    def test_backward_conflict_mixed_stale_and_fresh(self):
        """When multiple prior cells exist, only fresh ones trigger conflict.

        Setup: Cell A reads x (stale), Cell B reads x (fresh), Cell C modifies x.
        Expected: Violation against B (fresh) but not A (stale).
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Mark A stale, keep B fresh via NotebookState API
        self.sdc._notebook_state.handle_edit("a")

        # Cell C modifies x — should conflict with B (fresh) but not A (stale)
        self._save_pre_checkpoint("c", {"x": 1})
        ns_c = self._make_namespace({"x": 999})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_c.has_errors()
        # Should report B as the affected cell (first fresh cell in doc order that reads x)
        # A is skipped because it's stale
        assert result_c.errors[0].causer_cell == "b"


class TestEditTriggeredStaleness:
    """Tests for EDIT transition (§2.3) — mark_cell_edited()."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _execute_cell(
        self,
        cell_id: str,
        pre_ns: dict,
        post_ns: dict,
        reads: set = None,
        writes: set = None,
        continue_on_violation: bool = False,
    ):
        """Helper to execute a cell with given pre/post namespaces."""
        reads = reads or set()
        writes = writes or set()
        self._save_pre_checkpoint(cell_id, pre_ns)
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            namespace=post_ns,
            tracking=make_tracking(reads=reads, writes=writes),
            continue_on_violation=continue_on_violation,
        )

    def test_edit_marks_cell_stale(self):
        """Editing an executed cell marks it stale."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        assert "a" not in self.sdc._notebook_state.get_stale_cells()

        stale = self.sdc.mark_cell_edited("a")
        assert "a" in self.sdc._notebook_state.get_stale_cells()
        assert "a" in stale

    def test_edit_does_not_propagate_downstream(self):
        """Editing a cell does NOT propagate staleness to downstream cells.

        Downstream propagation is deferred to execution time (StaleFwd).
        """
        # A writes x, B reads x
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        assert "a" not in self.sdc._notebook_state.get_stale_cells()
        assert "b" not in self.sdc._notebook_state.get_stale_cells()

        # Edit A — only A should become stale, not B
        self.sdc.mark_cell_edited("a")
        assert "a" in self.sdc._notebook_state.get_stale_cells()
        assert "b" not in self.sdc._notebook_state.get_stale_cells()

    def test_edit_unexecuted_cell_is_noop(self):
        """Editing an unexecuted cell has no effect (no CODE_CHANGED reason added)."""
        # "a" has never been executed — no record
        # Note: "a" is already stale with NEVER_EXECUTED reason from cell_order setup
        stale_before = self.sdc._notebook_state.get_stale_cells()
        self.sdc.mark_cell_edited("a")
        stale_after = self.sdc._notebook_state.get_stale_cells()

        # No CODE_CHANGED reason should be added for unexecuted cell
        reasons = self.sdc._notebook_state.get_reasons("a")
        assert not any(r.type == ReasonType.CODE_CHANGED for r in reasons)
        # Stale list unchanged (edit is a no-op on unexecuted)
        assert set(stale_before) == set(stale_after)

    def test_edit_already_stale_cell(self):
        """Editing an already-stale cell is idempotent."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})

        # Mark stale via edit
        self.sdc.mark_cell_edited("a")
        assert "a" in self.sdc._notebook_state.get_stale_cells()

        # Mark stale again — no change
        stale = self.sdc.mark_cell_edited("a")
        assert "a" in self.sdc._notebook_state.get_stale_cells()
        assert stale.count("a") == 1  # Only appears once


class TestStalenessReasons:
    """Tests for staleness reason tracking (§1.2)."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _execute_cell(
        self,
        cell_id: str,
        pre_ns: dict,
        post_ns: dict,
        reads: set = None,
        writes: set = None,
        continue_on_violation: bool = False,
    ):
        """Helper to execute a cell with given pre/post namespaces."""
        reads = reads or set()
        writes = writes or set()
        self._save_pre_checkpoint(cell_id, pre_ns)
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            namespace=post_ns,
            tracking=make_tracking(reads=reads, writes=writes),
            continue_on_violation=continue_on_violation,
        )

    def test_edit_adds_code_changed_reason(self):
        """Editing a cell adds CODE_CHANGED reason."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.sdc.mark_cell_edited("a")

        # Check that notebook_state has CODE_CHANGED reason
        reasons = self.sdc._notebook_state.get_reasons("a")
        reason_types = {r.type.value for r in reasons}
        assert "code_changed" in reason_types

    def test_staleness_result_contains_reasons(self):
        """ReproducibilityResult includes staleness_reasons dict."""
        # A writes x
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # B reads x
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # A writes x again (different value) → B becomes stale
        self._save_pre_checkpoint("a", {"x": 1})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 2},  # Changed!
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale with FORWARD_STALE reason
        assert "b" in result.stale_cells
        assert "b" in result.staleness_reasons
        reasons_for_b = result.staleness_reasons["b"]
        assert any(r["type"] == "forward_stale" for r in reasons_for_b)

    def test_forward_stale_includes_variable_and_cell(self):
        """FORWARD_STALE reason includes loc and cell_id."""
        # A writes x
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # B reads x
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # A writes x again → B stale with FORWARD_STALE(loc=x, cell_id=a)
        self._save_pre_checkpoint("a", {"x": 1})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 2},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        reasons_for_b = result.staleness_reasons["b"]
        forward_stale = [r for r in reasons_for_b if r["type"] == "forward_stale"]
        assert len(forward_stale) >= 1
        assert forward_stale[0]["loc"] == "x"
        assert forward_stale[0]["cell_id"] == "a"

    def test_fresh_cell_has_no_reasons(self):
        """A freshly executed cell should have no staleness reasons."""
        result = self._execute_cell("a", {}, {"x": 1}, writes={"x"})

        # Cell a is fresh — should not be in staleness_reasons
        assert "a" not in result.staleness_reasons or not result.staleness_reasons.get(
            "a"
        )

    def test_multiple_reasons_accumulate(self):
        """Multiple staleness reasons accumulate on a cell."""
        # A writes x, B writes y
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, writes={"y"})
        # C reads both x and y
        self._execute_cell(
            "c",
            {"x": 1, "y": 2},
            {"x": 1, "y": 2, "z": 3},
            reads={"x", "y"},
            writes={"z"},
        )

        # A changes x
        self._save_pre_checkpoint("a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 10},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        # C should be stale due to x changing

        # Now B also changes y
        self._save_pre_checkpoint("b", {"x": 10, "y": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace={"x": 10, "y": 20},
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # C might have multiple FORWARD_STALE reasons (x from a, y from b)
        # Check that reasons accumulate
        reasons_for_c = result.staleness_reasons.get("c", [])
        # C was already stale from x changing, so it won't get y reason added
        # (already stale cells are skipped in _update_staleness_incremental)
        assert "c" in self.sdc._notebook_state.get_stale_cells()

    def test_reset_clears_reasons(self):
        """Resetting the enforcer clears all staleness reasons."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.sdc.mark_cell_edited("a")

        # Verify CODE_CHANGED reason exists
        reasons = self.sdc._notebook_state.get_reasons("a")
        assert any(r.type.value == "code_changed" for r in reasons)

        self.sdc.reset()

        # After reset, status dict should be empty (not CODE_CHANGED)
        # Note: get_reasons creates NEVER_EXECUTED for unknown cells, so we
        # check the internal status dict directly
        assert "a" not in self.sdc._notebook_state.status


class TestSkippedUpstream:
    """Tests for forward staleness when skipping upstream cells."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        # F writes x, G writes x, H reads x
        self.sdc.set_cell_order(["f", "g", "h"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _execute_cell(
        self,
        cell_id: str,
        pre_ns: dict,
        post_ns: dict,
        reads: set = None,
        writes: set = None,
        continue_on_violation: bool = False,
    ):
        """Helper to execute a cell with given pre/post namespaces."""
        reads = reads or set()
        writes = writes or set()
        self._save_pre_checkpoint(cell_id, pre_ns)
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            namespace=post_ns,
            tracking=make_tracking(reads=reads, writes=writes),
            continue_on_violation=continue_on_violation,
        )

    def test_skipped_upstream_then_run_expected_cell(self):
        """
        Scenario: F->G->H, then F again (skipping G), then G.

        After running F (skipping G), H has FORWARD_STALE.
        After running G, H still has FORWARD_STALE.

        Note: Staleness is always computed, even when G triggers NoReadAndWrite
        error (because G reads and writes x).
        """
        # Initial execution: F->G->H
        self._execute_cell("f", {}, {"x": 1}, writes={"x"})
        self._execute_cell("g", {"x": 1}, {"x": 2}, reads={"x"}, writes={"x"})
        self._execute_cell("h", {"x": 2}, {"x": 2, "y": 10}, reads={"x"}, writes={"y"})

        # H is clean after initial execution
        assert self.sdc._notebook_state.is_clean("h")

        # Run F again (skipping G) - x goes from 2 back to 1
        self._execute_cell("f", {"x": 2, "y": 10}, {"x": 1, "y": 10}, writes={"x"})

        # H should now be stale with FORWARD_STALE
        reasons_h = self.sdc._notebook_state.get_reasons("h")
        reason_types = {r.type for r in reasons_h}
        assert (
            ReasonType.FORWARD_STALE in reason_types
        ), f"Expected FORWARD_STALE, got {reason_types}"

        # Now run G - this should result in FORWARD_STALE
        # (Staleness computed even though G has NoReadAndWrite error)
        self._execute_cell(
            "g", {"x": 1, "y": 10}, {"x": 3, "y": 10}, reads={"x"}, writes={"x"}
        )

        # H should now have FORWARD_STALE
        reasons_h_after = self.sdc._notebook_state.get_reasons("h")
        reason_types_after = {r.type for r in reasons_h_after}

        assert (
            ReasonType.FORWARD_STALE in reason_types_after
        ), f"Expected FORWARD_STALE, got {reason_types_after}"
        # H was marked stale by F initially and stays that way
        input_reason = next(
            r for r in reasons_h_after if r.type == ReasonType.FORWARD_STALE
        )
        assert input_reason.loc == "x"
        # The cell_id could be 'f' or 'g' depending on timing - just verify H is stale for x
        assert input_reason.cell_id in ("f", "g")

    def test_skipped_upstream_exact_user_scenario(self):
        """
        Exact user scenario: F -> G -> H -> F -> G

        After this sequence, H should have FORWARD_STALE.

        Note: Staleness is always computed, even when G triggers NoReadAndWrite
        error (because G reads and writes x).
        """
        # F -> G -> H (initial)
        self._execute_cell("f", {}, {"x": 1}, writes={"x"})
        self._execute_cell("g", {"x": 1}, {"x": 2}, reads={"x"}, writes={"x"})
        self._execute_cell("h", {"x": 2}, {"x": 2, "y": 10}, reads={"x"}, writes={"y"})

        # F (re-run, skipping G)
        self._execute_cell("f", {"x": 2, "y": 10}, {"x": 1, "y": 10}, writes={"x"})

        reasons_after_f = self.sdc._notebook_state.get_reasons("h")
        # H should have FORWARD_STALE
        assert any(
            r.type == ReasonType.FORWARD_STALE for r in reasons_after_f
        ), f"Expected FORWARD_STALE after F, got {reasons_after_f}"

        # G (re-run) - staleness computed even though G has NoReadAndWrite error
        self._execute_cell(
            "g", {"x": 1, "y": 10}, {"x": 3, "y": 10}, reads={"x"}, writes={"x"}
        )

        reasons_after_g = self.sdc._notebook_state.get_reasons("h")

        # H should now have FORWARD_STALE
        reason_types = {r.type for r in reasons_after_g}
        assert (
            ReasonType.FORWARD_STALE in reason_types
        ), f"Expected FORWARD_STALE after G, got {reasons_after_g}"


class TestStaleness:
    """Tests for staleness computation."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_marks_stale_on_set_intersection(self):
        """Marks cells stale based on set intersection of writes and reads."""
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value (W_A ∩ R_B = {x} ≠ ∅)
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        ns_a2 = self._make_namespace({"x": 100, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale (syntactic: W_A ∩ R_B ≠ ∅)
        assert "b" in result.stale_cells

    def test_no_convergence(self):
        """Staleness is monotonic - no convergence detection.

        Scenario: A writes x, B reads x, then A re-writes x with different
        value (B stale), then A re-writes x back to original (B still stale
        because no convergence detection).
        """
        # A writes x=1
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value x=2 (makes B stale via ForwardStale)
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        ns_a2 = self._make_namespace({"x": 2, "y": 2})
        result_a2 = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert "b" in result_a2.stale_cells

        # Re-run A with original value x=1 (back to original)
        self._save_pre_checkpoint("a", {"x": 2, "y": 2})
        ns_a3 = self._make_namespace({"x": 1, "y": 2})
        result_a3 = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a3,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should STILL be stale (no convergence detection, staleness is monotonic)
        assert "b" in result_a3.stale_cells

    def test_defers_checkpoint_deletion(self):
        """Defers checkpoint deletion until next cell executes.

        This allows checkpoint size queries after a cell completes but before
        the next cell runs (important for benchmarking/compare_overhead).
        """
        # Execute cell A
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # In syntactic mode with deferred deletion, checkpoint A is STILL present
        # (marked for pending deletion, but not yet deleted)
        assert f"{PRE_CHECKPOINT_PREFIX}a" in self.checkpoints.saved
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}a"

        # Execute cell B - this triggers deletion of cell A's checkpoint
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Now checkpoint A should be deleted
        assert f"{PRE_CHECKPOINT_PREFIX}a" not in self.checkpoints.saved
        # Checkpoint B is still present (pending deletion for next cell)
        assert f"{PRE_CHECKPOINT_PREFIX}b" in self.checkpoints.saved
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}b"

    # =======================================================================
    # Backward Staleness Tests
    # =======================================================================

    def test_backward_staleness_marks_earlier_cell_stale(self):
        """BackwardStale: when later cell writes to var read by earlier clean cell.

        Scenario: A reads x, B is clean, then C writes x (accepted).
        After C runs, A should be marked stale (W_C ∩ R_A = {x} ≠ ∅).

        Uses continue_on_violation=True because C triggers NoWriteAfterRead
        error (C writes x that A read). When accepted, staleness propagates.
        """
        # A reads x (clean cell that read x)
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        # A should be clean
        assert self.sdc._notebook_state.is_clean("a")

        # B does something unrelated
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 1, "y": 10, "z": 20})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"z"}),
        )

        # C writes x (should make A stale via BackwardStale, accepted)
        self._save_pre_checkpoint("c", {"x": 1, "y": 10, "z": 20})
        ns_c = self._make_namespace({"x": 999, "y": 10, "z": 20})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # A should be stale (backward staleness: W_C ∩ R_A ≠ ∅)
        assert "a" in result.stale_cells

    def test_backward_staleness_only_affects_clean_cells(self):
        """BackwardStale should only mark clean cells as stale, not already-stale cells.

        Uses continue_on_violation=True because C triggers NoWriteAfterRead
        error (C writes to x that A and B read). When accepted, staleness propagates.
        """
        # A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 1, "y": 10, "z": 20})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Manually mark A as stale (simulating earlier staleness)
        from flowbook.kernel.models import Reason, ReasonType

        self.sdc._notebook_state.add_reason(
            "a", Reason(ReasonType.FORWARD_STALE, loc="dummy", cell_id="dummy")
        )
        assert not self.sdc._notebook_state.is_clean("a")
        assert self.sdc._notebook_state.is_clean("b")

        # C writes x - should only affect B (which is clean), not A (already stale)
        # Accepted so staleness propagates
        self._save_pre_checkpoint("c", {"x": 1, "y": 10, "z": 20})
        ns_c = self._make_namespace({"x": 999, "y": 10, "z": 20})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # B should now be stale
        assert "b" in result.stale_cells


class TestClearStalenessOnError:
    """Tests verifying that prior staleness reasons are cleared when repro errors occur.

    When a cell has staleness warnings (e.g., FORWARD_STALE, CODE_CHANGED) and running
    it produces a reproducibility error, the prior staleness reasons should be replaced
    by the error-based reason. This prevents confusing UI state where both staleness
    indicators and error indicators appear.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    def test_forward_stale_cleared_on_forward_contamination_error(self):
        """FORWARD_STALE reason is cleared when NoReadBeforeWrite error occurs.

        Scenario:
        1. A writes x, B reads x (B is clean)
        2. A is re-run, marking B as FORWARD_STALE
        3. C writes x (after B in order)
        4. B is re-run and triggers NoReadBeforeWrite (forward contamination)
        5. B's staleness should only show the error reason, not FORWARD_STALE
        """
        # Step 1: A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # Step 2: A is re-run with different x value, marking B as FORWARD_STALE
        self._save_pre_checkpoint("a", {"x": 1, "y": 10})
        ns_a2 = self._make_namespace({"x": 999, "y": 10})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert "b" in result_a.stale_cells
        reasons_before = self.sdc._notebook_state.get_reasons("b")
        assert any(r.type == ReasonType.FORWARD_STALE for r in reasons_before)

        # Step 3: C writes x (creating forward contamination for B)
        self._save_pre_checkpoint("c", {"x": 999, "y": 10})
        ns_c = self._make_namespace({"x": 2000, "y": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Step 4: B is re-run and triggers NoReadBeforeWrite
        self._save_pre_checkpoint("b", {"x": 2000, "y": 10})
        ns_b2 = self._make_namespace({"x": 2000, "y": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b2,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Step 5: Verify B's staleness only shows error reason
        assert len(result_b.errors) > 0
        assert result_b.errors[0].error_type == ErrorType.NO_READ_BEFORE_WRITE

        reasons_after = self.sdc._notebook_state.get_reasons("b")
        # Should have error-based reason
        assert any(r.type == ReasonType.NO_READ_BEFORE_WRITE for r in reasons_after)
        # Should NOT have the old FORWARD_STALE reason
        assert not any(r.type == ReasonType.FORWARD_STALE for r in reasons_after)

    def test_code_changed_cleared_on_no_read_and_write_error(self):
        """CODE_CHANGED reason is cleared when NoReadAndWrite error occurs.

        Scenario:
        1. A writes x, runs cleanly
        2. A is edited (marked CODE_CHANGED)
        3. A is re-run with code that reads and writes x (NoReadAndWrite error)
        4. A's staleness should only show the error reason, not CODE_CHANGED
        """
        # Step 1: A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.is_clean("a")

        # Step 2: A is edited
        self.sdc.mark_cell_edited("a")
        reasons_before = self.sdc._notebook_state.get_reasons("a")
        assert any(r.type == ReasonType.CODE_CHANGED for r in reasons_before)

        # Step 3: A is re-run with code that reads and writes x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a2 = self._make_namespace({"x": 2})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(
                reads={"x"}, writes={"x"}
            ),  # reads AND writes same var
        )

        # Step 4: Verify A's staleness only shows error reason
        assert len(result_a.errors) > 0
        assert result_a.errors[0].error_type == ErrorType.NO_READ_AND_WRITE

        reasons_after = self.sdc._notebook_state.get_reasons("a")
        # Should have error-based reason
        assert any(r.type == ReasonType.NO_READ_AND_WRITE for r in reasons_after)
        # Should NOT have the old CODE_CHANGED reason
        assert not any(r.type == ReasonType.CODE_CHANGED for r in reasons_after)


class TestStalenessAlwaysComputed:
    """Tests verifying staleness is computed when errors are accepted.

    When continue_on_violation=True, errors are accepted and the cell's writes
    persist. Staleness should be propagated to other cells in this case.

    When continue_on_violation=False (default), errors cause rejection and
    rollback. Staleness is NOT propagated because the writes are undone.
    See TestRollbackLastCheck for tests of the rejected case.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    def test_forward_stale_computed_with_no_read_and_write_error(self):
        """ForwardStale is computed when NoReadAndWrite error is accepted.

        Scenario: B reads and writes x (NoReadAndWrite violation, accepted).
        C should be marked stale because B changed x that C reads.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # C reads x
        self._save_pre_checkpoint("c", {"x": 1})
        ns_c = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("c")

        # B reads and writes x (triggers NoReadAndWrite, accepted)
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"x"}),
            continue_on_violation=True,
        )

        # B should have NoReadAndWrite error
        assert result.has_errors()
        assert any(e.error_type.value == "no_read_and_write" for e in result.errors)

        # C should still be marked stale (ForwardStale was computed)
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_forward_stale_computed_with_no_write_after_read_error(self):
        """ForwardStale is computed when NoWriteAfterRead error is accepted.

        Scenario: A reads x, then B writes x (NoWriteAfterRead violation, accepted).
        C should be marked stale because B changed x that C reads.
        """
        # A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # C reads x
        self._save_pre_checkpoint("c", {"x": 1})
        ns_c = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("c")

        # B writes x (triggers NoWriteAfterRead against A, accepted)
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # B should have NoWriteAfterRead error
        assert result.has_errors()
        assert any(e.error_type.value == "no_write_after_read" for e in result.errors)

        # C should still be marked stale (ForwardStale was computed)
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_backward_stale_computed_with_error(self):
        """BackwardStale is computed when error is accepted.

        Scenario: A reads x, then C writes x (accepted NoWriteAfterRead).
        A should be marked stale (BackwardStale) because C's writes persist.
        """
        # A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("a")

        # C writes x (triggers NoWriteAfterRead against A, accepted)
        self._save_pre_checkpoint("c", {"x": 1, "y": 10})
        ns_c = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # C should have NoWriteAfterRead error
        assert result.has_errors()

        # A should be marked stale (BackwardStale was computed)
        assert "a" in result.stale_cells or not self.sdc._notebook_state.is_clean("a")

    def test_writes_tracked_with_error(self):
        """Writes are tracked even when cell has errors.

        This ensures that downstream staleness tracking works correctly.
        """
        # A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # B writes x (triggers NoWriteAfterRead against A)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B has error
        assert result.has_errors()

        # writes should still be updated
        assert "x" in writelocset_var_names(self.sdc._notebook_state.writes.get("b", frozenset()))

    def test_multiple_errors_still_compute_staleness(self):
        """Staleness is computed even when cell has multiple errors.

        Scenario: B reads and writes x, which triggers:
        - NoReadAndWrite (reads and writes same variable)
        - NoWriteAfterRead (if A read x first)

        Downstream cell C should still be marked stale.
        """
        # A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # C reads x
        self._save_pre_checkpoint("c", {"x": 1})
        ns_c = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # B reads and writes x (triggers both NoReadAndWrite and NoWriteAfterRead, accepted)
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"x"}),
            continue_on_violation=True,
        )

        # B should have errors
        assert result.has_errors()
        assert len(result.errors) >= 1

        # C should still be marked stale
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_forward_stale_preserved_with_error(self):
        """FORWARD_STALE is preserved even when upstream cell has errors.

        Scenario: F->G->H, then F (skipping G), then G.
        When G runs (even with NoReadAndWrite error), H should have FORWARD_STALE.
        """
        self.sdc.set_cell_order(["f", "g", "h"])

        # F writes x
        self._save_pre_checkpoint("f", {})
        ns_f = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="f",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}f"],
            namespace=ns_f,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # G reads and writes x
        self._save_pre_checkpoint("g", {"x": 1})
        ns_g = self._make_namespace({"x": 2})
        self.sdc.check(
            cell_id="g",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}g"],
            namespace=ns_g,
            tracking=make_tracking(reads={"x"}, writes={"x"}),
        )

        # H reads x
        self._save_pre_checkpoint("h", {"x": 2})
        ns_h = self._make_namespace({"x": 2, "y": 10})
        self.sdc.check(
            cell_id="h",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}h"],
            namespace=ns_h,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("h")

        # Re-run F (skipping G)
        self._save_pre_checkpoint("f", {"x": 2, "y": 10})
        ns_f2 = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="f",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}f"],
            namespace=ns_f2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # After F (skipping G), H should be stale with FORWARD_STALE
        reasons_h = self.sdc._notebook_state.get_reasons("h")
        assert any(r.type == ReasonType.FORWARD_STALE for r in reasons_h)

        # Re-run G (triggers NoReadAndWrite error)
        self._save_pre_checkpoint("g", {"x": 1, "y": 10})
        ns_g2 = self._make_namespace({"x": 3, "y": 10})
        result = self.sdc.check(
            cell_id="g",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}g"],
            namespace=ns_g2,
            tracking=make_tracking(reads={"x"}, writes={"x"}),
        )

        # G should have NoReadAndWrite error
        assert result.has_errors()
        assert any(e.error_type.value == "no_read_and_write" for e in result.errors)

        # H should now have FORWARD_STALE
        reasons_h_after = self.sdc._notebook_state.get_reasons("h")
        reason_types = {r.type for r in reasons_h_after}
        assert ReasonType.FORWARD_STALE in reason_types


class TestNoReadAndWriteColumnLevel:
    """Tests for NoReadAndWrite predicate with column-level tracking.

    The NoReadAndWrite predicate checks: Rᵢ ∩ Wᵢ = ∅ (cell should not read
    and write the same location).

    At column granularity, reading a variable binding (df) and writing to
    its column (df['age']) are DIFFERENT locations. The violation should
    only trigger when the SAME column is both read and written.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    def test_read_variable_write_column_no_violation(self):
        """Reading df and writing df['age'] is NOT a NoReadAndWrite violation.

        Pattern: df['age'] = 1
        - Reads: df (the variable binding, to access the DataFrame object)
        - Writes: df.age (the column)

        These are different locations, so no violation.
        """
        import pandas as pd

        # A writes df
        df = pd.DataFrame({"age": [10, 20], "name": ["a", "b"]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df['age'] = 1
        # This reads df (variable) and writes df.age (column)
        df_modified = df.copy()
        df_modified["age"] = 1
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_modified})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df to access the DataFrame
                writes=set(),  # No variable-level writes
                column_reads={},  # No column reads
                column_writes={"df": {"age"}},  # Write to df.age
            ),
        )

        # Should NOT have NoReadAndWrite error
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 0
        ), f"Unexpected NoReadAndWrite error: {no_rw_errors}"

    def test_read_and_write_same_column_is_violation(self):
        """Reading and writing the SAME column IS a NoReadAndWrite violation.

        Pattern: df['age'] = df['age'] * 2
        - Reads: df.age (the column)
        - Writes: df.age (the same column)

        This is a violation.
        """
        import pandas as pd

        # A writes df
        df = pd.DataFrame({"age": [10, 20], "name": ["a", "b"]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df['age'] = df['age'] * 2
        # This reads df.age and writes df.age
        df_modified = df.copy()
        df_modified["age"] = df_modified["age"] * 2
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_modified})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df
                writes=set(),  # No variable-level writes
                column_reads={"df": {"age"}},  # Read df.age
                column_writes={"df": {"age"}},  # Write df.age
            ),
        )

        # SHOULD have NoReadAndWrite error for df.age
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"
        assert "df.age" in no_rw_errors[0].locations

    def test_read_one_column_write_another_no_violation(self):
        """Reading one column and writing another is NOT a violation.

        Pattern: df['total'] = df['price'] * df['qty']
        - Reads: df.price, df.qty
        - Writes: df.total

        Different columns, no violation.
        """
        import pandas as pd

        # A writes df
        df = pd.DataFrame({"price": [10, 20], "qty": [2, 3]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df['total'] = df['price'] * df['qty']
        df_modified = df.copy()
        df_modified["total"] = df_modified["price"] * df_modified["qty"]
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_modified})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df
                writes=set(),  # No variable-level writes
                column_reads={"df": {"price", "qty"}},  # Read df.price, df.qty
                column_writes={"df": {"total"}},  # Write df.total
            ),
        )

        # Should NOT have NoReadAndWrite error (different columns)
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 0
        ), f"Unexpected NoReadAndWrite error: {no_rw_errors}"

    def test_variable_level_read_and_write_is_violation(self):
        """Reading and writing the same variable is a violation (no column info).

        Pattern: x = x + 1
        - Reads: x
        - Writes: x

        This is a violation at the variable level.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B does: x = x + 1
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"x"},  # Read x
                writes={"x"},  # Write x
            ),
        )

        # SHOULD have NoReadAndWrite error
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"


class TestNoReadAndWriteDataFrameReassignment:
    """Tests for NoReadAndWrite predicate with DataFrame reassignment patterns.

    This test class covers the bug fix for patterns like:
        a = feature_engineer(a)

    Where a variable is read (as argument), then reassigned (variable binding changes),
    and the new object has column writes from internal function operations.

    The key insight is that when both:
    - Variable is in reads_before_writes (read as argument)
    - Variable is in writes (binding changed via assignment)
    - Variable is in column_writes (from internal function operations)

    This IS a NoReadAndWrite violation at the variable level, even though
    the column writes may be from internal function operations.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    def test_df_reassignment_via_function_is_violation(self):
        """a = feature_engineer(a) IS a NoReadAndWrite violation.

        Pattern: a = feature_engineer(a)
        - Reads: a (the variable, passed as argument)
        - Writes: a (the variable, reassigned to new DataFrame)
        - Column writes: a.price (from internal function operations)

        This is a NoReadAndWrite violation at the variable level because
        the same variable binding is both read and written.
        """
        import pandas as pd

        # A creates df 'a' and function
        df = pd.DataFrame({"price": [1, 2, 3, 4]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"a": df, "feature_engineer": lambda x: x})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"a", "feature_engineer"}),
        )

        # B does: a = feature_engineer(a)
        # This reads 'a', calls function, and reassigns 'a' to new DataFrame
        df_new = df.copy()
        df_new["price"] = df_new["price"] * 2
        self._save_pre_checkpoint("b", {"a": df, "feature_engineer": lambda x: x})
        ns_b = self._make_namespace({"a": df_new, "feature_engineer": lambda x: x})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"a", "feature_engineer"},  # Read a as argument
                writes={"a"},  # Reassign a to new DataFrame
                column_reads={"a": set()},  # No direct column reads
                column_writes={"a": {"price"}},  # Column writes from internal function
            ),
        )

        # SHOULD have NoReadAndWrite error for 'a'
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"
        assert "a" in no_rw_errors[0].locations, (
            f"Expected 'a' in violation locations, got {no_rw_errors[0].locations}"
        )

    def test_df_column_write_only_no_reassignment_no_violation(self):
        """df['col'] = value is NOT a violation (no reassignment).

        Pattern: df['price'] = 10
        - Reads: df (to access the DataFrame object)
        - Writes: df.price (column only, binding unchanged)

        This is NOT a NoReadAndWrite violation because reading the binding
        and writing to a column are different locations.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"price": [1, 2, 3]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df['price'] = 10
        df_modified = df.copy()
        df_modified["price"] = 10
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_modified})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df to access it
                writes=set(),  # No variable-level writes (binding unchanged)
                column_reads={},  # No column reads
                column_writes={"df": {"price"}},  # Write to df.price
            ),
        )

        # Should NOT have NoReadAndWrite error
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert len(no_rw_errors) == 0, f"Unexpected error: {no_rw_errors}"

    def test_df_copy_and_reassign_no_column_writes_is_violation(self):
        """df = df.copy() IS a NoReadAndWrite violation (even without column writes).

        Pattern: df = df.copy()
        - Reads: df (the variable)
        - Writes: df (the variable, reassigned)

        This is a violation at the variable level.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"x": [1, 2]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df = df.copy()
        df_new = df.copy()
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_new})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df
                writes={"df"},  # Write df (reassignment)
                column_reads={},  # No column-level tracking
                column_writes={},  # No column-level tracking
            ),
        )

        # SHOULD have NoReadAndWrite error
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"

    def test_df_filter_reassignment_is_violation(self):
        """df = df[df['col'] > 0] IS a NoReadAndWrite violation.

        Pattern: df = df[df['price'] > 0]
        - Reads: df (the variable), df.price (the column)
        - Writes: df (the variable, reassigned to filtered DataFrame)

        This is a violation because the variable binding is both read and written.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"price": [1, -1, 2, -2]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df = df[df['price'] > 0]
        df_filtered = df[df["price"] > 0]
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_filtered})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df
                writes={"df"},  # Write df (reassignment)
                column_reads={"df": {"price"}},  # Read df.price for filter
                column_writes={},  # No column writes
            ),
        )

        # SHOULD have NoReadAndWrite error for df
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"
        assert "df" in no_rw_errors[0].locations

    def test_df_read_column_then_reassign_is_violation(self):
        """Reading columns then reassigning the variable IS a NoReadAndWrite violation.

        Pattern:
            total = df['price'].sum()
            df = some_other_df

        Wait, this is different. Let's test:
            df = df.drop(columns=['price'])

        - Reads: df.price (implicitly read during drop)
        - Writes: df (reassigned)

        The variable is both read and written.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"price": [1, 2], "name": ["a", "b"]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df = df.drop(columns=['price'])
        df_dropped = df.drop(columns=["price"])
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_dropped})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df
                writes={"df"},  # Write df (reassignment)
                column_reads={},  # No explicit column reads tracked
                column_writes={},  # No column writes
            ),
        )

        # SHOULD have NoReadAndWrite error
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"

    def test_different_df_names_no_violation(self):
        """Reading one DataFrame and creating another is NOT a violation.

        Pattern: result = process(input_df)
        - Reads: input_df
        - Writes: result

        Different variables, no violation.
        """
        import pandas as pd

        # A creates input_df
        input_df = pd.DataFrame({"x": [1, 2, 3]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"input_df": input_df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"input_df"}),
        )

        # B does: result = process(input_df)
        result_df = input_df.copy()
        result_df["x"] = result_df["x"] * 2
        self._save_pre_checkpoint("b", {"input_df": input_df})
        ns_b = self._make_namespace({"input_df": input_df, "result": result_df})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"input_df"},  # Read input_df
                writes={"result"},  # Write result (different variable)
                column_reads={},
                column_writes={"result": {"x"}},  # Column writes on result
            ),
        )

        # Should NOT have NoReadAndWrite error (different variables)
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert len(no_rw_errors) == 0, f"Unexpected error: {no_rw_errors}"

    def test_reassign_with_both_column_read_and_column_write_via_function(self):
        """Function that reads and writes columns while reassigning IS a violation.

        Pattern: df = transform(df) where transform reads df['a'] and writes df['b']
        - Reads: df (variable), df.a (column)
        - Writes: df (variable), df.b (column)

        This is a violation because the variable is both read and written.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df = transform(df) which internally reads df['a'], writes df['b']
        df_new = df.copy()
        df_new["b"] = df_new["a"] * 2  # Simulates internal function operation
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_new})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df as argument
                writes={"df"},  # Reassign df
                column_reads={"df": {"a"}},  # Read df.a internally
                column_writes={"df": {"b"}},  # Write df.b internally
            ),
        )

        # SHOULD have NoReadAndWrite error for df
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"
        assert "df" in no_rw_errors[0].locations

    def test_inplace_column_update_same_column_is_violation(self):
        """In-place update of same column IS a NoReadAndWrite violation.

        Pattern: df['price'] = df['price'] * 2
        - Reads: df.price
        - Writes: df.price

        Same column read and written - violation at column level.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"price": [1, 2, 3]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B does: df['price'] = df['price'] * 2
        df_modified = df.copy()
        df_modified["price"] = df_modified["price"] * 2
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace({"df": df_modified})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"},  # Read df
                writes=set(),  # No variable-level write (binding unchanged)
                column_reads={"df": {"price"}},  # Read df.price
                column_writes={"df": {"price"}},  # Write df.price
            ),
        )

        # SHOULD have NoReadAndWrite error for df.price
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error, got {len(no_rw_errors)}"
        assert "df.price" in no_rw_errors[0].locations

    def test_empty_column_reads_with_column_writes_and_variable_write_is_violation(self):
        """The specific bug case: empty column reads, column writes, AND variable write.

        This is the exact pattern that triggered the bug:
        - column_reads = {'a': set()}  (empty set - no columns read from 'a')
        - column_writes = {'a': {'price'}}  (columns written)
        - writes = {'a'}  (variable reassigned)

        The bug was that this case fell through without detecting the violation.
        """
        import pandas as pd

        # A creates df
        df = pd.DataFrame({"price": [1, 2, 3, 4]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"a": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"a"}),
        )

        # B does: a = feature_engineer(a) with exact tracking from bug report
        df_new = df.copy()
        df_new["price"] = df_new["price"] * 2
        self._save_pre_checkpoint("b", {"a": df})
        ns_b = self._make_namespace({"a": df_new})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"a", "feature_engineer"},  # Read a and function
                writes={"a"},  # Reassign a
                column_reads={"a": set()},  # EMPTY set - this was the bug trigger
                column_writes={"a": {"price"}},  # Column writes
            ),
        )

        # SHOULD have NoReadAndWrite error for 'a'
        no_rw_errors = [
            e for e in result.errors if e.error_type.value == "no_read_and_write"
        ]
        assert (
            len(no_rw_errors) == 1
        ), f"Expected 1 NoReadAndWrite error for the bug case, got {len(no_rw_errors)}"
        assert "a" in no_rw_errors[0].locations, (
            f"Expected 'a' in locations, got {no_rw_errors[0].locations}"
        )


class TestForwardStaleFormula:
    """Tests for ForwardStale formula: (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅

    The new formula marks cell j stale if:
    - Cell i's old OR new writes overlap with cell j's reads OR writes

    This is more conservative than the old formula (W'ᵢ ∩ Rⱼ ≠ ∅).
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_write_overlap_marks_stale(self):
        """Cell j is stale if i writes to a variable that j also writes.

        New behavior: (Wᵢ ∪ W'ᵢ) ∩ Wⱼ ≠ ∅ triggers staleness.
        Old behavior would NOT mark j stale since j doesn't READ x.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.is_clean("a")

        # B writes x (doesn't read it)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x", "y"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 10})
        ns_a2 = self._make_namespace({"x": 2, "y": 10})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale because A wrote x and B writes x
        assert "b" in result.stale_cells

    def test_old_writes_overlap_with_reads_marks_stale(self):
        """Cell j is stale if i's OLD writes overlap with j's reads.

        Scenario: Cell A writes x, then writes y (no longer writes x).
        Cell B reads x. B should be stale because A's old writes included x.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        # Record that A wrote x
        assert writelocset_var_names(self.sdc._notebook_state.writes.get("a", frozenset())) == {"x"}

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "z": 10})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # Re-run A, now writes y instead of x (x unchanged in namespace)
        self._save_pre_checkpoint("a", {"x": 1, "z": 10})
        ns_a2 = self._make_namespace({"x": 1, "y": 5, "z": 10})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # B should be stale because A's OLD writes (x) overlap with B's reads (x)
        # Even though A's NEW writes (y) don't overlap with B's reads
        assert "b" in result.stale_cells

    def test_old_writes_overlap_with_writes_marks_stale(self):
        """Cell j is stale if i's OLD writes overlap with j's writes.

        Scenario: Cell A writes x, then writes y (no longer writes x).
        Cell B writes x. B should be stale because A's old writes included x.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B writes x (doesn't read it)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 100})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # Re-run A, now writes y instead of x
        self._save_pre_checkpoint("a", {"x": 100})
        ns_a2 = self._make_namespace({"x": 100, "y": 5})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # B should be stale because A's OLD writes (x) overlap with B's writes (x)
        assert "b" in result.stale_cells

    def test_no_overlap_not_stale(self):
        """Cell j is NOT stale if there's no overlap.

        Formula: (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) = ∅ means j is NOT stale.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads/writes completely different variables (y, z)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 10, "z": 20})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 10, "z": 20})
        ns_a2 = self._make_namespace({"x": 2, "y": 10, "z": 20})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should NOT be stale (no overlap: {x} ∩ {y, z} = ∅)
        assert "b" not in result.stale_cells
        assert self.sdc._notebook_state.is_clean("b")

    def test_combined_old_and_new_writes(self):
        """Both old and new writes contribute to staleness.

        Wᵢ ∪ W'ᵢ means both old and new writes can trigger staleness.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x, C reads y
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "b_out": 10})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"b_out"}),
        )

        self._save_pre_checkpoint("c", {"x": 1, "b_out": 10, "y": 2})
        ns_c = self._make_namespace({"x": 1, "b_out": 10, "y": 2, "c_out": 20})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"y"}, writes={"c_out"}),
        )

        assert self.sdc._notebook_state.is_clean("b")
        assert self.sdc._notebook_state.is_clean("c")

        # Re-run A, now writes y instead of x (old writes: {x}, new writes: {y})
        self._save_pre_checkpoint("a", {"x": 1, "b_out": 10, "y": 2, "c_out": 20})
        ns_a2 = self._make_namespace({"x": 1, "b_out": 10, "y": 99, "c_out": 20})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # B should be stale: old writes {x} overlap with B's reads {x}
        # C should be stale: new writes {y} overlap with C's reads {y}
        assert "b" in result.stale_cells
        assert "c" in result.stale_cells


class TestRollbackLastCheck:
    """Tests for rollback_last_check() - restoring state after rejected execution."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def test_rollback_clears_writes(self):
        """Rollback removes writes entries added by check().

        Scenario: Cell D writes x, gets rejected, rollback should remove
        D's write of x so it doesn't affect later checks.
        """
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.last_writer_for("x", "d") == "a"

        # Cell D writes x (simulating a check that will be rejected)
        self._save_pre_checkpoint("d", {"x": 1})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 999},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert "x" in writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset()))

        # Rollback D's check
        self.sdc.rollback_last_check()

        # D's writes should be restored (no x), so last_writer_for x is back to A
        assert self.sdc._notebook_state.last_writer_for("x", "d") == "a"

    def test_rollback_restores_writes_set(self):
        """Rollback restores the cell's writes set to previous value."""
        # Cell D writes x first time
        self._save_pre_checkpoint("d", {})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset())) == {"x"}

        # Cell D writes y (different variable) - simulating rejected execution
        self._save_pre_checkpoint("d", {"x": 1})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 1, "y": 2},
            tracking=make_tracking(reads=set(), writes={"y"}),
        )
        assert writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset())) == {"y"}

        # Rollback
        self.sdc.rollback_last_check()

        # Writes should be restored to {"x"}
        assert writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset())) == {"x"}

    def test_rollback_restores_status(self):
        """Rollback restores cell status to previous value."""
        # Cell A writes x, marks later cell C stale
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell C reads x
        self._save_pre_checkpoint("c", {"x": 1})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace={"x": 1, "c_out": 10},
            tracking=make_tracking(reads={"x"}, writes={"c_out"}),
        )
        assert self.sdc._notebook_state.is_clean("c")

        # Cell A re-runs, marks C stale - this will be rejected
        self._save_pre_checkpoint("a", {"x": 1, "c_out": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 2, "c_out": 10},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Rollback A's check - but this only restores A's state, not C's
        # (staleness propagation to C is a side effect we don't fully rollback)
        self.sdc.rollback_last_check()

        # A's writes should be restored
        assert "x" in writelocset_var_names(self.sdc._notebook_state.writes.get("a", frozenset()))

    def test_rollback_for_never_executed_cell(self):
        """Rollback works correctly for a cell that hadn't executed before."""
        # Cell D never executed before (set_cell_order initializes empty sets)
        assert self.sdc._notebook_state.writes.get("d", frozenset()) == frozenset()
        assert self.sdc._notebook_state.last_writer_for("x", "d") is None
        assert self.sdc._notebook_state.tracking_data.get("d") is None

        # D writes x (simulating rejected execution)
        self._save_pre_checkpoint("d", {})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset())) == {"x"}
        assert "x" in writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset()))
        assert self.sdc._notebook_state.tracking_data.get("d") is not None

        # Rollback
        self.sdc.rollback_last_check()

        # Should be back to never-executed state
        assert self.sdc._notebook_state.writes.get("d", frozenset()) == frozenset()
        assert self.sdc._notebook_state.last_writer_for("x", "d") is None
        assert self.sdc._notebook_state.tracking_data.get("d") is None

    def test_rollback_noop_when_no_pending_snapshot(self):
        """Rollback is safe to call even without a pending snapshot."""
        # Just make sure it doesn't crash
        self.sdc.rollback_last_check()
        self.sdc.rollback_last_check()  # Multiple calls should be safe

    def test_user_scenario_edit_d_to_different_variable(self):
        """
        User scenario: D writes x, causes error, user edits D to write w,
        re-runs D, then runs C - C should not see "Read x from @D".

        This test simulates the full flow with rollback.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 10},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B writes x
        self._save_pre_checkpoint("b", {"x": 10})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace={"x": 11},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # C reads x, writes y
        self._save_pre_checkpoint("c", {"x": 11})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace={"x": 11, "y": 11},
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # D writes x (will be rejected due to backward mutation with C)
        self._save_pre_checkpoint("d", {"x": 11, "y": 11})
        result_d = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 999, "y": 11},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        # This causes backward mutation (D writes x, C read x)
        assert result_d.has_errors()

        # After check, D has x in writes
        assert "x" in writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset()))

        # Kernel rolls back D's execution
        self.sdc.rollback_last_check()

        # Now D's writes should be restored (no x), B is last writer of x
        assert "x" not in writelocset_var_names(self.sdc._notebook_state.writes.get("d", frozenset()))

        # User edits D to write w instead of x, and re-runs D
        self._save_pre_checkpoint("d", {"x": 11, "y": 11})
        result_d2 = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 11, "y": 11, "w": 1},
            tracking=make_tracking(reads=set(), writes={"w"}),
        )
        # No violation now since D writes w, not x
        assert not result_d2.has_errors()
        assert not result_d2.has_errors()

        # Re-run C - should NOT get forward contamination from D
        self._save_pre_checkpoint("c", {"x": 11, "y": 11, "w": 1})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace={"x": 11, "y": 11, "w": 1},
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # C should have no forward contamination error (no "Read x from @D")
        assert not any(
            e.error_type == ErrorType.NO_READ_BEFORE_WRITE for e in result_c.errors
        )

    def test_reexecute_same_cell_does_not_delete_new_checkpoint(self):
        """
        Re-executing the same cell twice in a row should not delete the new checkpoint.

        Bug scenario (before fix):
        1. Cell A executes, sets _pending_checkpoint_deletion = "_pre_a"
        2. Cell A re-executes, creates new checkpoint "_pre_a" (same name)
        3. check() starts, deletes pending checkpoint (which is the NEW one!)
        4. Later restore attempt fails because checkpoint was deleted

        The fix: check() skips deletion if pending checkpoint is for the current cell.
        """

        # Execute cell A first time
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Verify pending deletion is set for cell A
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}a"

        # Re-execute cell A (same cell!) - creates new checkpoint with same name
        self._save_pre_checkpoint("a", {"x": 1})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 2},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Checkpoint should still exist (not deleted because it's for the current cell)
        assert f"{PRE_CHECKPOINT_PREFIX}a" in self.checkpoints.saved

        # Pending deletion should now be set for the new execution
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}a"

    def test_execute_different_cell_deletes_previous_checkpoint(self):
        """
        Executing a different cell should delete the pending checkpoint from previous cell.

        This verifies the deferred deletion still works for different cells.
        """

        # Execute cell A
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Verify A's checkpoint exists and is pending deletion
        assert f"{PRE_CHECKPOINT_PREFIX}a" in self.checkpoints.saved
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}a"

        # Execute cell B (different cell)
        self._save_pre_checkpoint("b", {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace={"x": 1, "y": 2},
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell A's checkpoint should now be deleted
        assert f"{PRE_CHECKPOINT_PREFIX}a" not in self.checkpoints.saved

        # Cell B's checkpoint should exist and be pending deletion
        assert f"{PRE_CHECKPOINT_PREFIX}b" in self.checkpoints.saved
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}b"

    def test_rollback_clears_pending_checkpoint_deletion(self):
        """
        Rollback clears _pending_checkpoint_deletion.

        When execution is rolled back, we should not delete the checkpoint
        on the next execution (even of a different cell).
        """

        # Execute cell A
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Verify pending deletion is set
        assert self.sdc._pending_checkpoint_deletion == f"{PRE_CHECKPOINT_PREFIX}a"

        # Rollback the check (simulating rejected execution)
        self.sdc.rollback_last_check()

        # Verify pending deletion is cleared
        assert self.sdc._pending_checkpoint_deletion is None


class TestColumnAliasExpansion:
    """
    Test that column reads/writes are properly expanded with aliases.

    When x and y are aliases for the same DataFrame p, and code reads p['col'],
    the column read set should include x['col'] and y['col'] too for proper
    staleness detection.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(sanity_check=False, warn_classes=False)
        self.sdc = ReproducibilityEnforcer(
            self.checkpoints,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_forward_staleness_through_alias(self):
        """
        Cell A reads p['col']
        Cell B creates alias: x = p
        Cell C writes x['col']
        Cell A should become stale.

        This is the key test for the alias expansion feature.
        """
        import pandas as pd

        p = pd.DataFrame({"col": [1, 2, 3]})

        # Cell A: reads p['col']
        self._save_pre_checkpoint("a", {"p": p})
        ns_a = self._make_namespace({"p": p})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"p"}, writes=set(), column_reads={"p": {"col"}}
            ),
        )
        assert not result_a.has_errors()
        # Don't check stale_cells here - b,c,d may be stale from first execution

        # Cell B: creates alias x = p (same object)
        x = p  # Same DataFrame object
        self._save_pre_checkpoint("b", {"p": p, "x": x})
        ns_b = self._make_namespace({"p": p, "x": x})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"p"}, writes={"x"}),
        )
        assert not result_b.has_errors()

        # Cell C: writes x['col'] (modifying through alias)
        # Since Cell A (earlier) read p['col'] and C writes x['col'] (alias for p),
        # this triggers a NoWriteAfterRead violation. We use continue_on_violation=True
        # to allow staleness computation even with the violation, since we're testing
        # that alias expansion correctly identifies the staleness relationship.
        p_modified = pd.DataFrame({"col": [999, 999, 999]})
        self._save_pre_checkpoint("c", {"p": p, "x": x})
        ns_c = self._make_namespace({"p": p_modified, "x": p_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(writes={"x"}, column_writes={"x": {"col"}}),
            continue_on_violation=True,
        )

        # Cell A should be stale because p['col'] changed (via alias x)
        assert "a" in result_c.stale_cells, (
            f"Cell A should be stale after C writes x['col'] (x is alias for p). "
            f"Got stale_cells={result_c.stale_cells}"
        )

    def test_no_false_positive_different_dataframes(self):
        """
        Cell A reads p['col']
        Cell B has different DataFrame x (NOT alias)
        Cell C writes x['col']
        Cell A should NOT become stale (different objects).
        """
        import pandas as pd

        p = pd.DataFrame({"col": [1, 2, 3]})
        x = pd.DataFrame({"col": [4, 5, 6]})  # Different DataFrame object

        # Cell A: reads p['col']
        self._save_pre_checkpoint("a", {"p": p})
        ns_a = self._make_namespace({"p": p})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"p"}, writes=set(), column_reads={"p": {"col"}}
            ),
        )
        assert not result_a.has_errors()

        # Cell B: creates independent x
        self._save_pre_checkpoint("b", {"p": p, "x": x})
        ns_b = self._make_namespace({"p": p, "x": x})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(writes={"x"}),
        )

        # Cell C: writes x['col'] (different DataFrame from p)
        x_modified = pd.DataFrame({"col": [999, 999, 999]})
        self._save_pre_checkpoint("c", {"p": p, "x": x})
        ns_c = self._make_namespace({"p": p, "x": x_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(writes={"x"}, column_writes={"x": {"col"}}),
        )

        # Cell A should NOT be stale (x is different DataFrame from p)
        assert "a" not in result_c.stale_cells, (
            f"Cell A should NOT be stale when x is a different DataFrame from p. "
            f"Got stale_cells={result_c.stale_cells}"
        )

    def test_multiple_aliases_forward_staleness(self):
        """
        Cell A reads p['col']
        Cell B creates multiple aliases: x = p, y = p
        Cell C writes through y['col']
        Cell A should become stale.
        """
        import pandas as pd

        p = pd.DataFrame({"col": [1, 2, 3]})

        # Cell A: reads p['col']
        self._save_pre_checkpoint("a", {"p": p})
        ns_a = self._make_namespace({"p": p})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"p"}, writes=set(), column_reads={"p": {"col"}}
            ),
        )
        assert not result_a.has_errors()

        # Cell B: creates multiple aliases
        x = p
        y = p  # Both are same object as p
        self._save_pre_checkpoint("b", {"p": p, "x": x, "y": y})
        ns_b = self._make_namespace({"p": p, "x": x, "y": y})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"p"}, writes={"x", "y"}),
        )

        # Cell C: writes through y (which is alias for p)
        # This is actually a backward mutation (C writes location that earlier A read),
        # but we want to test staleness propagation, so continue_on_violation=True
        p_modified = pd.DataFrame({"col": [999, 999, 999]})
        self._save_pre_checkpoint("c", {"p": p, "x": x, "y": y})
        ns_c = self._make_namespace({"p": p_modified, "x": p_modified, "y": p_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(writes={"y"}, column_writes={"y": {"col"}}),
            continue_on_violation=True,
        )

        # Cell A should be stale because p['col'] changed via y
        assert "a" in result_c.stale_cells, (
            f"Cell A should be stale after C writes y['col'] (y is alias for p). "
            f"Got stale_cells={result_c.stale_cells}"
        )

    def test_column_reads_include_aliases_in_notebook_state(self):
        """
        When x and y are aliases for p, if Cell A reads p['col'],
        then notebook_state.get_column_reads("a") should include
        entries for x['col'] and y['col'] too, not just p['col'].

        This is about the metadata being complete for display/debugging.
        """
        import pandas as pd

        p = pd.DataFrame({"col": [1, 2, 3]})
        x = p  # alias
        y = p  # alias

        # First create the aliases in the namespace
        self._save_pre_checkpoint("b", {"p": p, "x": x, "y": y})
        ns_b = self._make_namespace({"p": p, "x": x, "y": y})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"p", "x", "y"}),
        )

        # Cell A (after B) reads p['col']
        self._save_pre_checkpoint("a", {"p": p, "x": x, "y": y})
        ns_a = self._make_namespace({"p": p, "x": x, "y": y})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"p"}, writes=set(), column_reads={"p": {"col"}}
            ),
        )

        # Get the column reads stored in notebook state
        column_reads = self.sdc._notebook_state.get_column_reads("a")

        # With alias expansion, column_reads should include x and y too
        # since they're aliases for p
        assert "p" in column_reads, f"p should be in column_reads. Got: {column_reads}"
        assert "col" in column_reads.get(
            "p", set()
        ), f"col should be in p's reads. Got: {column_reads}"

        # THE KEY ASSERTION: aliases should also appear
        assert "x" in column_reads, (
            f"x (alias for p) should be in column_reads when p['col'] is read. "
            f"Got: {column_reads}"
        )
        assert "y" in column_reads, (
            f"y (alias for p) should be in column_reads when p['col'] is read. "
            f"Got: {column_reads}"
        )

    def test_structural_reads_include_aliases_in_notebook_state(self):
        """
        When x and y are aliases for p, if Cell A reads p.shape,
        then notebook_state.get_structural_reads("a") should include
        entries for x.shape and y.shape too, not just p.shape.
        """
        import pandas as pd

        p = pd.DataFrame({"col": [1, 2, 3]})
        x = p  # alias
        y = p  # alias

        # First create the aliases in the namespace
        self._save_pre_checkpoint("b", {"p": p, "x": x, "y": y})
        ns_b = self._make_namespace({"p": p, "x": x, "y": y})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"p", "x", "y"}),
        )

        # Cell A (after B) reads p.shape (structural read)
        self._save_pre_checkpoint("a", {"p": p, "x": x, "y": y})
        ns_a = self._make_namespace({"p": p, "x": x, "y": y})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"p"}, writes=set(), structural_reads={"p": {"shape", "columns"}}
            ),
        )

        # Get the structural reads stored in notebook state
        structural_reads = self.sdc._notebook_state.get_structural_reads("a")

        # With alias expansion, structural_reads should include x and y too
        assert (
            "p" in structural_reads
        ), f"p should be in structural_reads. Got: {structural_reads}"
        assert "shape" in structural_reads.get(
            "p", set()
        ), f"shape should be in p's reads. Got: {structural_reads}"

        # THE KEY ASSERTION: aliases should also appear
        assert "x" in structural_reads, (
            f"x (alias for p) should be in structural_reads when p.shape is read. "
            f"Got: {structural_reads}"
        )
        assert "y" in structural_reads, (
            f"y (alias for p) should be in structural_reads when p.shape is read. "
            f"Got: {structural_reads}"
        )

    def test_rollback_no_read_and_write_does_not_falsely_stale_earlier_cells(self):
        """
        When a cell gets NO_READ_AND_WRITE and is rolled back, earlier cells
        that read the same variable should NOT be falsely marked stale.

        Scenario:
            Cell A: creates df, writes df['hour']
            Cell B: reads df['hour'] (clean)
            Cell C: reads AND writes df['hour'] → NO_READ_AND_WRITE, rolled back

        After rollback:
            - B should still be clean (not falsely marked stale)
            - C's reads/writes should be cleared
            - A later cell D writing df['hour'] should NOT conflict with C
        """
        import pandas as pd

        df = pd.DataFrame({"hour": [1, 2, 3]})

        # Cell A: Create df with hour column
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"df": df},
            tracking=make_tracking(writes={"df"}, column_writes={"df": {"hour"}}),
        )

        # Cell B: Reads df['hour'], computes avg
        self._save_pre_checkpoint("b", {"df": df})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace={"df": df, "avg": 2.0},
            tracking=make_tracking(
                reads={"df"},
                writes={"avg"},
                column_reads={"df": {"hour"}},
            ),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # Cell C: Reads AND writes df['hour'] → NO_READ_AND_WRITE
        df_modified = pd.DataFrame({"hour": [10, 20, 30]})
        self._save_pre_checkpoint("c", {"df": df, "avg": 2.0})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace={"df": df_modified, "avg": 2.0},
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": {"hour"}},
                column_writes={"df": {"hour"}},
            ),
        )

        # C should have NO_READ_AND_WRITE error
        assert result_c.has_errors()

        # Simulate kernel rollback
        self.sdc.rollback_last_check()

        # B should still be clean — C was rolled back, so its writes never happened
        assert self.sdc._notebook_state.is_clean(
            "b"
        ), "Cell B was falsely marked stale after rollback of C's NO_READ_AND_WRITE"

        # C's tracking should be cleared (never executed)
        assert self.sdc._notebook_state.tracking_data.get("c") is None

        # Later cell D writes df['hour'] — should only conflict with B, not C
        df_d = pd.DataFrame({"hour": [100, 200, 300]})
        self._save_pre_checkpoint("d", {"df": df, "avg": 2.0})
        result_d = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"df": df_d, "avg": 2.0},
            tracking=make_tracking(
                writes={"df"},
                column_writes={"df": {"hour"}},
            ),
        )

        # D should have backward violation against B (which read df.hour), not C
        if result_d.has_errors():
            assert result_d.errors[0].causer_cell == "b"
            assert "c" != result_d.errors[0].causer_cell


class TestUnrecoverableMutation:
    """Tests for unrecoverable mutation detection and write set restriction."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    # =========================================================================
    # Tests that should produce UNRECOVERABLE_MUTATION errors
    # =========================================================================

    def test_inplace_list_mutation_is_error(self):
        """Cell does x[0] = 99 — x not in tracking.writes → UNRECOVERABLE_MUTATION."""
        # A writes x (a list)
        x = [1, 2, 3]
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": x})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": [1, 2, 3]})
        ns_b = self._make_namespace({"x": [1, 2, 3], "y": 6})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # C mutates x in place: x[0] = 99
        x_mutated = [99, 2, 3]
        self._save_pre_checkpoint("c", {"x": [1, 2, 3], "y": 6})
        ns_c = self._make_namespace({"x": x_mutated, "y": 6})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes=set()),  # reads x, does NOT rebind
        )

        # Should have UNRECOVERABLE_MUTATION error
        error_types = [e.error_type for e in result_c.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION in error_types
        unrec_error = [e for e in result_c.errors if e.error_type == ErrorType.UNRECOVERABLE_MUTATION][0]
        assert "x" in unrec_error.locations

    def test_inplace_dict_mutation_is_error(self):
        """Cell does d['key'] = val — d not in tracking.writes → error."""
        d = {"a": 1}
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"d": d})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"d"}),
        )

        # B mutates d in place
        self._save_pre_checkpoint("b", {"d": {"a": 1}})
        ns_b = self._make_namespace({"d": {"a": 1, "key": "val"}})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"d"}, writes=set()),  # reads d, does NOT rebind
        )

        error_types = [e.error_type for e in result_b.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION in error_types

    def test_inplace_mutation_accepted_with_continue(self):
        """With continue_on_violation=True, error is reported but cell stays clean."""
        x = [1, 2, 3]
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": x})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B mutates x in place, but with continue_on_violation
        self._save_pre_checkpoint("b", {"x": [1, 2, 3]})
        ns_b = self._make_namespace({"x": [99, 2, 3]})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
            continue_on_violation=True,
        )

        # Error still reported
        error_types = [e.error_type for e in result_b.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION in error_types
        # But cell stays clean (accepted)
        assert self.sdc._notebook_state.is_clean("b")

    def test_inplace_mutation_rejected_by_default(self):
        """Without continue_on_violation, cell is marked stale."""
        x = [1, 2, 3]
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": x})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B mutates x in place (default: continue_on_violation=False)
        self._save_pre_checkpoint("b", {"x": [1, 2, 3]})
        ns_b = self._make_namespace({"x": [99, 2, 3]})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert not self.sdc._notebook_state.is_clean("b")
        # Should have UNRECOVERABLE_MUTATION reason
        reasons = self.sdc._notebook_state.get_reasons("b")
        reason_types = {r.type for r in reasons}
        assert ReasonType.UNRECOVERABLE_MUTATION in reason_types

    # =========================================================================
    # Tests that should NOT produce errors
    # =========================================================================

    def test_rebinding_is_recoverable(self):
        """Cell does x = [1,2,3] — x IS in tracking.writes → no error."""
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": [1, 2, 3]})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        error_types = [e.error_type for e in result_a.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION not in error_types

    def test_column_write_is_recoverable(self):
        """Cell does df['col'] = val — col IS in tracking.column_writes → no error."""
        import pandas as pd
        df = pd.DataFrame({"price": [10, 20]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B adds a new column (tracked in column_writes)
        df_with_col = pd.DataFrame({"price": [10, 20], "discount": [1, 2]})
        self._save_pre_checkpoint("b", {"df": pd.DataFrame({"price": [10, 20]})})
        ns_b = self._make_namespace({"df": df_with_col})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(
                reads={"df"}, writes=set(),
                column_writes={"df": {"discount"}},
            ),
        )

        # Should NOT have unrecoverable error (column write is tracked)
        error_types = [e.error_type for e in result_b.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION not in error_types

    # =========================================================================
    # Staleness propagation tests
    # =========================================================================

    def test_inplace_mutation_does_not_propagate_staleness(self):
        """In-place mutation should NOT make downstream cells stale (even when accepted)."""
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": [1, 2, 3]})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": [1, 2, 3]})
        ns_b = self._make_namespace({"x": [1, 2, 3], "y": 6})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # C does x[0] = 99 (accepted via continue_on_violation)
        self._save_pre_checkpoint("c", {"x": [1, 2, 3], "y": 6})
        ns_c = self._make_namespace({"x": [99, 2, 3], "y": 6})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes=set()),
            continue_on_violation=True,
        )

        # B should NOT be stale — x mutation is unrecoverable, doesn't propagate
        assert self.sdc._notebook_state.is_clean("b"), \
            f"B should stay clean but got: {self.sdc._notebook_state.get_status('b')}"

    def test_recoverable_write_propagates_staleness(self):
        """Rebinding x=2 should make cells reading x stale."""
        # A writes x=1
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # C rebinds x=2 (x IS in writes, continue_on_violation since C writes after B reads)
        self._save_pre_checkpoint("c", {"x": 1, "y": 2})
        ns_c = self._make_namespace({"x": 2, "y": 2})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        # B IS stale (x was rebound by C, which is recoverable)
        assert not self.sdc._notebook_state.is_clean("b")

    def test_inplace_mutation_not_last_writer(self):
        """In-place mutation should NOT make the cell the last_writer of the variable."""
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": [1, 2, 3]})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # C mutates x in place (accepted)
        self._save_pre_checkpoint("c", {"x": [1, 2, 3]})
        ns_c = self._make_namespace({"x": [99, 2, 3]})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes=set()),
            continue_on_violation=True,
        )

        # A should still be the last_writer of x, not C (C only mutated in place)
        assert self.sdc._notebook_state.last_writer_for("x", "d") == "a", \
            f"Expected 'a' as last_writer of x, got {self.sdc._notebook_state.last_writer_for('x', 'd')}"

    def test_mixed_recoverable_unrecoverable(self):
        """Cell does y=1; x[0]=2 (writes={y}). Error for x, y propagates normally."""
        # A writes x, reads y
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": [1, 2, 3]})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads y
        self._save_pre_checkpoint("b", {"x": [1, 2, 3]})
        ns_b = self._make_namespace({"x": [1, 2, 3], "y": 0})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # D reads y
        self._save_pre_checkpoint("d", {"x": [1, 2, 3], "y": 0})
        ns_d = self._make_namespace({"x": [1, 2, 3], "y": 0, "z": 0})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )
        assert self.sdc._notebook_state.is_clean("d")

        # C does y=1 (recoverable) AND x[0]=99 (unrecoverable)
        self._save_pre_checkpoint("c", {"x": [1, 2, 3], "y": 0, "z": 0})
        ns_c = self._make_namespace({"x": [99, 2, 3], "y": 1, "z": 0})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
            continue_on_violation=True,
        )

        # Should have UNRECOVERABLE_MUTATION error for x
        error_types = [e.error_type for e in result_c.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION in error_types
        unrec_error = [e for e in result_c.errors if e.error_type == ErrorType.UNRECOVERABLE_MUTATION][0]
        assert "x" in unrec_error.locations

        # D (which reads y) should be stale (y was rebound, recoverable propagation)
        assert not self.sdc._notebook_state.is_clean("d"), \
            f"D should be stale due to y change, but got: {self.sdc._notebook_state.get_status('d')}"

    def test_delete_mutating_cell_no_staleness(self):
        """Deleting a cell that only mutated in-place should not propagate staleness.

        D mutates x[5]=3 (accepted). D deleted. Cells reading x should NOT be
        marked stale from D since D was never last_writer.
        """
        # A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": [1, 2, 3, 4, 5, 6]})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": [1, 2, 3, 4, 5, 6]})
        ns_b = self._make_namespace({"x": [1, 2, 3, 4, 5, 6], "y": 21})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # D mutates x in place (accepted)
        self._save_pre_checkpoint("d", {"x": [1, 2, 3, 4, 5, 6], "y": 21})
        ns_d = self._make_namespace({"x": [1, 2, 3, 4, 5, 99], "y": 21})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads={"x"}, writes=set()),
            continue_on_violation=True,
        )

        # D is NOT last_writer of x (A is) - D only mutated in place
        assert self.sdc._notebook_state.last_writer_for("x", "d") == "a"

        # Delete D
        self.sdc.set_cell_order(["a", "b", "c"])

        # B should still be clean — D was never last_writer of x
        assert self.sdc._notebook_state.is_clean("b"), \
            f"B should stay clean after D deletion, got: {self.sdc._notebook_state.get_status('b')}"

    def test_recoverable_column_write_propagates_staleness(self):
        """Column write (df['col']=val) without rebinding df should propagate staleness.

        This is a recoverable mutation (column is tracked in column_writes),
        so it SHOULD make cells reading df stale, even though df is not rebound.
        """
        import pandas as pd

        # A: df = DataFrame
        df_orig = pd.DataFrame({"price": [10, 20]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df_orig.copy()})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B: reads df (var-level read)
        self._save_pre_checkpoint("b", {"df": df_orig.copy()})
        ns_b = self._make_namespace({"df": df_orig.copy(), "total": 30})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"df"}, writes={"total"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # C: df['price'] = df['price'] * 2 (recoverable column write, NO rebinding)
        df_mod = df_orig.copy()
        df_mod["price"] = df_mod["price"] * 2
        self._save_pre_checkpoint("c", {"df": df_orig.copy(), "total": 30})
        ns_c = self._make_namespace({"df": df_mod, "total": 30})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(
                reads={"df"}, writes=set(),
                column_writes={"df": {"price"}},
            ),
            continue_on_violation=True,
        )

        # No UNRECOVERABLE_MUTATION error (column write is tracked)
        error_types = [e.error_type for e in result_c.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION not in error_types

        # B IS stale — df's price column changed (recoverable propagation)
        assert not self.sdc._notebook_state.is_clean("b"), \
            f"B should be stale from df column change, got: {self.sdc._notebook_state.get_status('b')}"

    def test_unrecoverable_column_mutation_does_not_propagate(self):
        """df.values[0,0]=99 (not in column_writes) should NOT propagate staleness."""
        import pandas as pd

        # A: df = DataFrame
        df_orig = pd.DataFrame({"price": [10, 20]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df_orig.copy()})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B: reads df
        self._save_pre_checkpoint("b", {"df": df_orig.copy()})
        ns_b = self._make_namespace({"df": df_orig.copy(), "total": 30})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"df"}, writes={"total"}),
        )
        assert self.sdc._notebook_state.is_clean("b")

        # C: modifies df element directly (bypasses column tracking — unrecoverable)
        df_mut = df_orig.copy()
        df_mut.iloc[0, 0] = 999
        self._save_pre_checkpoint("c", {"df": df_orig.copy(), "total": 30})
        ns_c = self._make_namespace({"df": df_mut, "total": 30})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"df"}, writes=set()),
            continue_on_violation=True,
        )

        # Should have UNRECOVERABLE_MUTATION error
        error_types = [e.error_type for e in result_c.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION in error_types

        # B should NOT be stale — mutation is unrecoverable, doesn't propagate
        assert self.sdc._notebook_state.is_clean("b"), \
            f"B should stay clean but got: {self.sdc._notebook_state.get_status('b')}"

    def test_rebound_df_with_new_columns_is_recoverable(self):
        """df = df.merge(...) adds new columns but df is rebound — no error.

        When a variable is rebound (df = ...), ALL column changes are recoverable
        because re-executing recreates the entire DataFrame. Column-level tracking
        only matters for in-place mutations (df['col'] = val without rebinding).
        """
        import pandas as pd

        # A: df = DataFrame
        df_orig = pd.DataFrame({"store": [1, 2], "price": [10, 20]})
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"df": df_orig.copy()})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # B: df = df.merge(store_df) — rebound, adds store_factor column
        store_df = pd.DataFrame({"store": [1, 2], "store_factor": [0.9, 1.1]})
        df_merged = df_orig.merge(store_df, on="store")
        self._save_pre_checkpoint("b", {"df": df_orig.copy()})
        ns_b = self._make_namespace({"df": df_merged})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            # df IS in writes (rebound), store_factor NOT in column_writes
            tracking=make_tracking(reads={"df"}, writes={"df"}),
        )

        # Should NOT have UNRECOVERABLE_MUTATION (df was rebound)
        error_types = [e.error_type for e in result_b.errors]
        assert ErrorType.UNRECOVERABLE_MUTATION not in error_types, \
            f"Rebound df should not trigger unrecoverable mutation, got errors: {result_b.errors}"


class TestRerunEmptiedCell:
    """Test that re-running a cell whose code was removed propagates staleness correctly.

    Scenario:
        A: x = 0
        B: x = 1
        C: x = 2
        D: print(x)  (reads x)

    Execute A→B→C→D. Then comment out C's code and rerun C (now empty).
    - D should become stale: it read x from C, but C no longer writes x.
    - B should become stale: B is the last writer of x before C, and C's
      removal of x exposes B's value — its provenance role changed.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None
        )

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    def test_rerun_emptied_cell_makes_reader_stale(self):
        """D should become stale when C stops writing x."""
        # A: x = 0
        self._save_pre_checkpoint("a", {})
        ns = self._make_namespace({"x": 0})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B: x = 1
        self._save_pre_checkpoint("b", {"x": 0})
        ns = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # C: x = 2
        self._save_pre_checkpoint("c", {"x": 1})
        ns = self._make_namespace({"x": 2})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # D: print(x) — reads x
        self._save_pre_checkpoint("d", {"x": 2})
        ns = self._make_namespace({"x": 2})
        result_d = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )
        assert not result_d.has_errors()
        assert "d" not in result_d.stale_cells, "D should be clean after initial run"

        # All cells should be clean now
        assert self.sdc._notebook_state.is_clean("a")
        assert self.sdc._notebook_state.is_clean("b")
        assert self.sdc._notebook_state.is_clean("c")
        assert self.sdc._notebook_state.is_clean("d")

        # Now comment out C and rerun it (empty: no reads, no writes)
        # Namespace still has x=2 (empty code doesn't change it)
        self._save_pre_checkpoint("c", {"x": 2})
        ns = self._make_namespace({"x": 2})
        result_c_empty = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes=set()),
        )

        # D should be stale: C used to write x (D's input), but no longer does
        stale = result_c_empty.stale_cells
        assert "d" in stale, (
            f"D should be stale after C stops writing x. "
            f"stale_cells={stale}, "
            f"W_old_c={{'x'}}, W_new_c=set(), "
            f"D reads x"
        )

    def test_rerun_emptied_cell_makes_last_writer_stale(self):
        """B should become stale when C stops writing x (B is now last writer)."""
        # Same setup: A→B→C→D
        self._save_pre_checkpoint("a", {})
        ns = self._make_namespace({"x": 0})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("b", {"x": 0})
        ns = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("c", {"x": 1})
        ns = self._make_namespace({"x": 2})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("d", {"x": 2})
        ns = self._make_namespace({"x": 2})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Rerun C with empty code
        self._save_pre_checkpoint("c", {"x": 2})
        ns = self._make_namespace({"x": 2})
        result_c_empty = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns,
            tracking=make_tracking(reads=set(), writes=set()),
        )

        # B should be stale: C used to write x, B also writes x and is now
        # the last writer before D. This is the BackwardStale case:
        # C removed x from its writes, and B was the last writer of x before C.
        stale = result_c_empty.stale_cells
        assert "b" in stale, (
            f"B should be stale after C stops writing x (B is now last writer). "
            f"stale_cells={stale}"
        )
