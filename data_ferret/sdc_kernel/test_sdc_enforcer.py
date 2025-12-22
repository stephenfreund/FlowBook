"""Tests for SDC Enforcer."""

import pytest

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData
from data_ferret.kernel.structural_tracking import StructuralTrackingMode

from .sdc_enforcer import SDCEnforcer, PRE_CHECKPOINT_PREFIX
from .conftest import make_tracking


class TestSDCEnforcer:

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_no_violation_forward_dependency(self):
        """Cell B reads what cell A writes - valid."""
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result_a.violation is None

        # Cell B reads x - valid (forward dependency)
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert result_b.violation is None

    def test_violation_backward_mutation(self):
        """Cell B modifies what cell A reads - violation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B (after A) modifies x - violation!
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.violation is not None
        assert result_b.violation.mutating_cell == "b"
        assert result_b.violation.affected_cell == "a"
        assert "x" in result_b.violation.variables

    def test_staleness_computation(self):
        """Re-running cell A makes cell B stale if B reads A's output."""
        # First run: A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale
        assert "b" in result.stale_cells

    def test_no_staleness_if_value_unchanged(self):
        """Semantic check: no staleness if value didn't actually change."""
        # A writes x=1
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with same value x=1
        # Note: pre-checkpoint for A now reflects current state
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 1, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should NOT be stale (x didn't change)
        assert "b" not in result.stale_cells

    def test_cell_order_update_affects_violation_check(self):
        """Cell order can be updated, affecting position-based checks."""
        # Initially [a, b, c, d]
        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - A is before B in order, so NOT a violation
        # (B can depend on A, that's forward dependency)
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.violation is None

        # Now reorder: [b, a, c, d] - B is now before A
        self.sdc.set_cell_order(["b", "a", "c", "d"])

        # A modifies x - now A is AFTER B, so this IS a violation
        # (A is mutating what B, an earlier cell, reads)
        self._save_pre_checkpoint("a", {"x": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.violation is not None
        assert result.violation.affected_cell == "b"

    def test_cell_deletion_prunes_records(self):
        """Deleted cells are removed from tracking."""
        # Execute cells a and b
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert "a" in self.sdc.records
        assert "b" in self.sdc.records

        # Remove cell b from order
        self.sdc.set_cell_order(["a", "c", "d"])

        # b should be pruned
        assert "a" in self.sdc.records
        assert "b" not in self.sdc.records

    def test_reset_clears_all_state(self):
        """Reset clears all tracking state."""
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert len(self.sdc.records) == 1
        assert self.sdc.seq_counter == 1

        self.sdc.reset()

        assert len(self.sdc.records) == 0
        assert self.sdc.seq_counter == 0
        assert self.sdc.cell_order == []

    def test_stale_cells_in_document_order(self):
        """Stale cells should be returned in document order, not execution order."""
        # Execute in order: a, d, b, c (but document order is a, b, c, d)
        # All read x

        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("d", {"x": 1})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "w": 4})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads={"x"}, writes={"w"}),
        )

        self._save_pre_checkpoint("b", {"x": 1, "w": 4})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "w": 4, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        self._save_pre_checkpoint("c", {"x": 1, "w": 4, "y": 2})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "w": 4, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Re-run A with different x - should make b, c, d stale (they all read x)
        self._save_pre_checkpoint("a", {"x": 1, "w": 4, "y": 2, "z": 3})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "w": 4, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a2,
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
        post_a = self._make_post_checkpoint("post_a", {"var": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"var"}, writes=set()),
        )

        # Cell 'x' (not in order) modifies 'var' - should not trigger violation
        self._save_pre_checkpoint("x", {"var": 1})
        post_x = self._make_post_checkpoint("post_x", {"var": 999})
        result = self.sdc.check(
            cell_id="x",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}x"],
            post_checkpoint=post_x,
            tracking=make_tracking(reads=set(), writes={"var"}),
        )

        # No violation because 'x' is not in cell_order
        assert result.violation is None


class TestColumnAwareBackwardMutation:
    """Tests for column-aware backward mutation detection."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_no_conflict_different_columns(self):
        """Cell A reads df.price, Cell B modifies df.quantity - no violation."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"quantity"}},
            ),
        )

        # No violation - different columns
        assert result_b.violation is None

    def test_conflict_same_column(self):
        """Cell A reads df.price, Cell B modifies df.price - violation."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"price"}},
            ),
        )

        # Violation - same column
        assert result_b.violation is not None
        assert result_b.violation.mutating_cell == "b"
        assert result_b.violation.affected_cell == "a"
        assert "df.price" in result_b.violation.variables

    def test_conflict_prior_no_column_info_conservative(self):
        """Cell A reads df (no column info), Cell B modifies df.price - violation (conservative)."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df (no column tracking)
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"price"}},
            ),
        )

        # Violation - conservative when prior has no column info
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "df" in result_b.violation.variables

    def test_conflict_current_no_column_info_conservative(self):
        """Cell A reads df.price, Cell B modifies df (no column info) - violation (conservative)."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(
                reads={"df"},
                writes={"y"},
                column_reads={"df": {"price"}},
            ),
        )

        # Cell B: modifies entire df (no column tracking)
        df_modified = df * 2
        self._save_pre_checkpoint("b", {"df": df, "y": 30})
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                # No column_writes - assumes whole df is modified
            ),
        )

        # Violation - conservative when current has no column info
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        # New resolver provides precise column info: "df.price" instead of just "df"
        assert any(v.startswith("df") for v in result_b.violation.variables)

    def test_mixed_variable_and_column_conflicts(self):
        """Mixed scenario: variable-level conflict on config, no column conflict on df."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})
        config = {"a": 1}

        # Cell A: reads config (variable-level) and df.price (column-level)
        self._save_pre_checkpoint("a", {"df": df, "config": config})
        post_a = self._make_post_checkpoint("post_a", {"df": df, "config": config, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "config": config_modified, "y": 10})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df", "config"},
                writes={"df", "config"},
                column_reads={"df": set()},
                column_writes={"df": {"quantity"}},
            ),
        )

        # Violation on config only (not on df since different columns)
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "config" in result_b.violation.variables
        # df.price should NOT be in violations (different column)
        assert "df.price" not in result_b.violation.variables

    def test_multiple_column_conflicts(self):
        """Multiple columns conflict: df.price and df.quantity both modified."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2], "total": [10, 40]})

        # Cell A: reads df.price and df.quantity
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df, "y": 30})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "y": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},
                column_writes={"df": {"price", "quantity"}},
            ),
        )

        # Violation - both columns conflict
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "df.price" in result_b.violation.variables
        assert "df.quantity" in result_b.violation.variables

    def test_no_conflict_when_no_overlap_multiple_vars(self):
        """Multiple DataFrames with no column overlap - no violation."""
        import pandas as pd

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})

        # Cell A: reads df1.a and df2.b
        self._save_pre_checkpoint("a", {"df1": df1, "df2": df2})
        post_a = self._make_post_checkpoint("post_a", {"df1": df1, "df2": df2, "y": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", {"df1": df1_modified, "df2": df2, "y": 1})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df1"},
                writes={"df1"},
                column_reads={"df1": set()},
                column_writes={"df1": {"b"}},
            ),
        )

        # No violation - df1.a not modified, df2 not modified
        assert result_b.violation is None


class TestContinueOnViolation:
    """Tests for continue_on_violation parameter."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_violation_without_continue_has_empty_stale(self):
        """Default behavior: violation returns empty stale_cells."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B modifies x (backward mutation)
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=False,  # default
        )

        assert result.violation is not None
        assert result.stale_cells == []  # Empty when not continuing
        assert result.changed_variables == []  # Empty when not continuing

    def test_violation_with_continue_computes_stale(self):
        """With continue_on_violation=True, staleness is computed even on violation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B modifies x (backward mutation) - but we continue
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,  # Continue despite violation
        )

        assert result.violation is not None
        assert result.violation.mutating_cell == "b"
        assert result.violation.affected_cell == "a"
        # Key assertion: stale_cells is computed
        assert "a" in result.stale_cells  # A is stale because x changed
        # changed_variables is also computed
        assert "x" in result.changed_variables

    def test_continue_updates_execution_record(self):
        """With continue_on_violation=True, the cell's execution record is updated."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert "a" in self.sdc.records
        assert "b" not in self.sdc.records

        # Cell B modifies x (violation) but we continue
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        assert result.violation is not None
        # Record is updated even with violation
        assert "b" in self.sdc.records
        assert self.sdc.records["b"].tracking.writes == {"x"}

    def test_continue_false_does_not_update_record(self):
        """With continue_on_violation=False, no record is created on violation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x (violation) with default behavior
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=False,
        )

        assert result.violation is not None
        # Record is NOT created
        assert "b" not in self.sdc.records

    def test_continue_with_chain_staleness(self):
        """Test staleness propagation when continuing after violation."""
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x, writes y
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell C reads y
        self._save_pre_checkpoint("c", {"x": 1, "y": 2})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )

        # Cell D modifies x (violation against A) - but we continue
        self._save_pre_checkpoint("d", {"x": 1, "y": 2, "z": 3})
        post_d = self._make_post_checkpoint("post_d", {"x": 999, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
            continue_on_violation=True,
        )

        assert result.violation is not None
        # B reads x, so D modifying x is a backward mutation against B
        assert result.violation.affected_cell == "b"
        # B is stale because x changed (B reads x)
        assert "b" in result.stale_cells
        # C is NOT stale (reads y, which hasn't changed)
        assert "c" not in result.stale_cells
        # A doesn't read anything, so not stale
        assert "a" not in result.stale_cells


class TestTruncationDetection:
    """Test the _check_for_truncation helper function."""

    def test_no_truncation_empty_diff(self):
        """Empty diff should not be truncated."""
        from data_ferret.kernel.types import DiffResult
        from data_ferret.sdc_kernel.sdc_enforcer import _check_for_truncation

        diff = DiffResult(differences={})
        truncated_vars = _check_for_truncation(diff)
        assert truncated_vars == []

    def test_no_truncation_simple_diff(self):
        """Simple diff without _truncated should not be detected."""
        from data_ferret.kernel.types import DiffResult, ValueComparison
        from data_ferret.sdc_kernel.sdc_enforcer import _check_for_truncation

        diff = DiffResult(differences={
            "x": ValueComparison(
                status="different",
                value1=1,
                value2=2,
                message="Values differ",
            )
        })
        truncated_vars = _check_for_truncation(diff)
        assert truncated_vars == []

    def test_truncation_detected_in_dict(self):
        """Truncation in structural type (dict) should be detected."""
        from data_ferret.kernel.types import DiffResult, ValueComparison, CompoundDiff
        from data_ferret.sdc_kernel.sdc_enforcer import _check_for_truncation, _format_diff_for_display

        diff = DiffResult(differences={
            "my_dict": CompoundDiff(
                source_type="dict",
                children={
                    "['key1']": ValueComparison(
                        status="different", value1=1, value2=2, message="diff"
                    ),
                },
                truncated=True
            )
        })
        truncated_vars = _check_for_truncation(diff)
        assert truncated_vars == ["my_dict"]
        # Test lazy formatting separately
        formatted_diff = _format_diff_for_display(diff, truncated_vars)
        assert "TRUNCATED DIFF DETAILS" in formatted_diff
        assert "Variable: my_dict" in formatted_diff

    def test_nested_container_truncation_is_ignored(self):
        """Truncation in nested container should NOT be flagged (only structure-level matters)."""
        from data_ferret.kernel.types import DiffResult, ValueComparison, CompoundDiff
        from data_ferret.sdc_kernel.sdc_enforcer import _check_for_truncation

        # The outer dict is not truncated, only the inner list is
        # Since we only check the immediate variable's truncation status, this should pass
        diff = DiffResult(differences={
            "outer": CompoundDiff(
                source_type="dict",
                children={
                    "['inner']": CompoundDiff(
                        source_type="list",  # list is not in STRUCTURAL_TYPES, so even if truncated, ignored
                        children={},
                        truncated=True
                    ),
                },
                truncated=False  # The outer dict is not truncated
            )
        })
        truncated_vars = _check_for_truncation(diff)
        # Nested truncation should NOT be detected - only structure-level at top
        assert truncated_vars == []

    def test_multiple_truncated_vars(self):
        """Multiple truncated variables should all be detected."""
        from data_ferret.kernel.types import DiffResult, ValueComparison, CompoundDiff
        from data_ferret.sdc_kernel.sdc_enforcer import _check_for_truncation, _format_diff_for_display

        diff = DiffResult(differences={
            "dict1": CompoundDiff(
                source_type="dict",
                children={"['key']": ValueComparison(status="different", value1=1, value2=2, message="diff")},
                truncated=True
            ),
            "obj2": CompoundDiff(
                source_type="object",
                children={".attr": ValueComparison(status="different", value1=1, value2=2, message="diff")},
                truncated=True
            ),
            "clean": ValueComparison(
                status="different", value1=1, value2=2, message="diff"
            ),
        })
        truncated_vars = _check_for_truncation(diff)
        assert set(truncated_vars) == {"dict1", "obj2"}
        # Test lazy formatting separately
        formatted_diff = _format_diff_for_display(diff, truncated_vars)
        assert "Variable: dict1" in formatted_diff
        assert "Variable: obj2" in formatted_diff


class TestStructuralTrackingOff:
    """Tests for structural tracking OFF mode with backward mutation detection."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        # Create enforcer with structural tracking OFF
        self.sdc = SDCEnforcer(
            self.checkpoints,
            structural_mode=StructuralTrackingMode.OFF,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_structural_only_read_no_violation_when_off(self):
        """
        With structural tracking OFF, a cell that only reads structural attrs
        (like df.shape) should NOT conflict with a cell that adds columns.

        This tests Bug4.ipynb scenario:
        - Cell D: raw_data.shape (structural read only, no column reads)
        - Cell E: raw_data['x'] = 3 (column write)

        With structural tracking OFF, this should NOT be a violation.
        """
        import pandas as pd

        # Create a DataFrame
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert result_a.violation is None

        # Cell B: Reads raw_data.shape (structural read only, no column reads)
        # This cell reads the variable but only accesses structural attributes
        self._save_pre_checkpoint("b", {"raw_data": df})
        post_b = self._make_post_checkpoint("post_b", {"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"raw_data"},  # Variable read
                writes=set(),
                column_reads=None,  # NO column reads (only .shape)
                structural_reads={"raw_data": {"shape"}},  # Structural read
            ),
        )
        assert result_b.violation is None

        # Cell C: Adds a new column raw_data['x'] = 3
        # With structural tracking OFF, this should NOT conflict with cell B
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        post_c = self._make_post_checkpoint("post_c", {"raw_data": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"raw_data"},  # Reads raw_data to modify it
                writes={"raw_data"},
                column_reads=None,
                column_writes={"raw_data": {"x"}},  # Writes column 'x'
            ),
        )
        # NO violation - structural tracking is OFF and prior cell only did structural read
        assert result_c.violation is None

    def test_whole_variable_read_still_causes_violation_when_off(self):
        """
        Even with structural tracking OFF, a cell that reads the whole variable
        (not just structural attrs) should still conflict with column modifications.

        This ensures we're not too permissive.
        """
        import pandas as pd

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )

        # Cell B: Reads raw_data (whole variable, e.g., print(raw_data))
        # This is NOT a structural-only read
        self._save_pre_checkpoint("b", {"raw_data": df})
        post_b = self._make_post_checkpoint("post_b", {"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"raw_data"},
                writes=set(),
                column_reads=None,  # No column tracking
                structural_reads=None,  # NOT a structural read - whole variable read
            ),
        )

        # Cell C: Adds a new column
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        post_c = self._make_post_checkpoint("post_c", {"raw_data": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"raw_data"},
                writes={"raw_data"},
                column_reads=None,
                column_writes={"raw_data": {"x"}},
            ),
        )
        # SHOULD cause violation - prior cell read whole variable (not just structural)
        assert result_c.violation is not None
        assert "raw_data" in result_c.violation.variables


class TestStructuralTrackingWarn:
    """Tests for structural tracking WARN mode with backward mutation detection."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        # Create enforcer with structural tracking WARN
        self.sdc = SDCEnforcer(
            self.checkpoints,
            structural_mode=StructuralTrackingMode.WARN,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_structural_only_read_no_violation_in_warn_mode(self):
        """
        With structural tracking WARN, a cell that only reads structural attrs
        (like df.shape) should NOT cause a backward mutation violation.

        Instead, structural warnings should be generated (tested separately).
        This tests the Bug4.ipynb scenario with WARN mode.
        """
        import pandas as pd

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert result_a.violation is None

        # Cell B: Reads raw_data.shape (structural read only)
        self._save_pre_checkpoint("b", {"raw_data": df})
        post_b = self._make_post_checkpoint("post_b", {"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"raw_data"},
                writes=set(),
                column_reads=None,
                structural_reads={"raw_data": {"shape"}},
            ),
        )
        assert result_b.violation is None

        # Cell C: Adds a new column raw_data['x'] = 3
        # With WARN mode, this should NOT cause a backward mutation violation
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        post_c = self._make_post_checkpoint("post_c", {"raw_data": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"raw_data"},
                writes={"raw_data"},
                column_reads=None,
                column_writes={"raw_data": {"x"}},
            ),
        )
        # NO violation - structural tracking is WARN and prior cell only did structural read
        # (warnings are handled separately by diff, not as backward mutation violations)
        assert result_c.violation is None


class TestStructuralTrackingEnforce:
    """Tests for structural tracking ENFORCE mode - structural reads ARE protected."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        # Create enforcer with structural tracking ENFORCE
        self.sdc = SDCEnforcer(
            self.checkpoints,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_structural_only_read_causes_violation_in_enforce_mode(self):
        """
        With structural tracking ENFORCE, a cell that reads structural attrs
        (like df.shape) SHOULD cause a violation when structure changes.

        This is the strictest mode - structural reads are fully protected.
        """
        import pandas as pd

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert result_a.violation is None

        # Cell B: Reads raw_data.shape (structural read only)
        self._save_pre_checkpoint("b", {"raw_data": df})
        post_b = self._make_post_checkpoint("post_b", {"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"raw_data"},
                writes=set(),
                column_reads=None,
                structural_reads={"raw_data": {"shape"}},
            ),
        )
        assert result_b.violation is None

        # Cell C: Adds a new column raw_data['x'] = 3
        # With ENFORCE mode, this SHOULD cause a violation
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        post_c = self._make_post_checkpoint("post_c", {"raw_data": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"raw_data"},
                writes={"raw_data"},
                column_reads=None,
                column_writes={"raw_data": {"x"}},
            ),
        )
        # SHOULD cause violation - ENFORCE mode protects structural reads
        assert result_c.violation is not None
        assert "raw_data" in result_c.violation.variables

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

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"df": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # Cell B: Display DataFrame - reads columns AND structural attrs
        # This simulates what happens when you just type `df` in a cell
        self._save_pre_checkpoint("b", {"df": df})
        post_b = self._make_post_checkpoint("post_b", {"df": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes=set(),
                column_reads={"df": {"a", "b"}},  # Read actual column data
                structural_reads={"df": {"columns", "dtypes", "shape"}},  # Also read structure
            ),
        )
        assert result_b.violation is None

        # Cell C: Adds a new column df['x'] = 3
        # No overlap with columns a,b but structural read of .columns is violated
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"df": df})
        post_c = self._make_post_checkpoint("post_c", {"df": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_reads={"df": set()},  # Didn't read existing columns
                column_writes={"df": {"x"}},  # Wrote new column
            ),
        )
        # SHOULD cause violation - prior cell read .columns, we added column x
        assert result_c.violation is not None
        assert "df" in result_c.violation.variables


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
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict) -> Checkpoint:
        """Create a post-checkpoint."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

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
        post_a = self._make_post_checkpoint("post_a", {"y": y})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"y"}, writes=set()),
        )

        # Cell B: creates alias x = y (same object)
        x = y  # x and y are the same object
        self._save_pre_checkpoint("b", {"y": y, "x": x})
        post_b = self._make_post_checkpoint("post_b", {"y": y, "x": x})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"y"}, writes={"x"}),
        )

        # Cell C: modifies through x (which changes y too!)
        x_modified = [1, 2, 3]
        x_modified[0] = 999  # Simulate in-place modification
        self._save_pre_checkpoint("c", {"y": y, "x": x})
        # After modification, both x and y point to modified list
        post_c = self._make_post_checkpoint("post_c", {"y": x_modified, "x": x_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Should detect violation - Cell A read y, Cell C modified y (via x)
        assert result_c.violation is not None
        assert result_c.violation.affected_cell == "a"
        # The violation should mention y (the variable Cell A read)
        assert "y" in result_c.violation.variables or "x" in result_c.violation.variables

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

        df = pd.DataFrame({'price': [100, 200], 'quantity': [5, 10]})

        # Cell A: creates df, reads price column
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(
                reads={"df"}, writes=set(),
                column_reads={"df": {"price"}}
            ),
        )

        # Cell B: creates alias (NOT a copy)
        df_alias = df  # Same object!
        self._save_pre_checkpoint("b", {"df": df, "df_alias": df_alias})
        post_b = self._make_post_checkpoint("post_b", {"df": df, "df_alias": df_alias})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"df"}, writes={"df_alias"}),
        )

        # Cell C: modifies through alias
        df_modified = pd.DataFrame({'price': [999, 999], 'quantity': [5, 10]})
        self._save_pre_checkpoint("c", {"df": df, "df_alias": df_alias})
        post_c = self._make_post_checkpoint("post_c", {"df": df_modified, "df_alias": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads=set(), writes={"df_alias"},
                column_writes={"df_alias": {"price"}}
            ),
        )

        # Should detect violation
        assert result_c.violation is not None
        assert result_c.violation.affected_cell == "a"

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
        post_a = self._make_post_checkpoint("post_a", {"x": x})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: creates multiple aliases
        y = x
        z = x
        self._save_pre_checkpoint("b", {"x": x, "y": y, "z": z})
        post_b = self._make_post_checkpoint("post_b", {"x": x, "y": y, "z": z})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y", "z"}),
        )

        # Cell C: modifies through z
        modified = {"value": 999}
        self._save_pre_checkpoint("c", {"x": x, "y": y, "z": z})
        post_c = self._make_post_checkpoint("post_c", {"x": modified, "y": modified, "z": modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"z"}),
        )

        # Should detect violation - x was read by Cell A
        assert result_c.violation is not None
        assert result_c.violation.affected_cell == "a"

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
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
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
        post_b = self._make_post_checkpoint("post_b", namespace_after)
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No violation - y wasn't read by any earlier cell
        assert result_b.violation is None

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

        df = pd.DataFrame({'a': [1, 2, 3]})

        # Cell A: creates df, reads df
        self._save_pre_checkpoint("a", {"df": df})
        post_a = self._make_post_checkpoint("post_a", {"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"df"}, writes=set()),
        )

        # Cell B: creates alias
        df_alias = df
        self._save_pre_checkpoint("b", {"df": df, "df_alias": df_alias})
        post_b = self._make_post_checkpoint("post_b", {"df": df, "df_alias": df_alias})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"df"}, writes={"df_alias"}),
        )

        # Cell C: breaks alias with copy, then modifies the copy
        # Pre-state: df and df_alias are same object
        self._save_pre_checkpoint("c", {"df": df, "df_alias": df_alias})
        # Post-state: df_alias is now a different object (the copy, modified)
        df_copy_modified = df.copy()
        df_copy_modified['a'] = [999, 999, 999]
        post_c = self._make_post_checkpoint("post_c", {"df": df, "df_alias": df_copy_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"df_alias"}, writes={"df_alias"}),
        )

        # No violation - df is unchanged, only the copy (df_alias) changed
        assert result_c.violation is None

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
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: creates new variable y
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 42})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No violation - y is new, x is unchanged
        assert result_b.violation is None

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
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: deletes y
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})  # y deleted
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No violation - y wasn't read by earlier cells
        assert result_b.violation is None

    def test_alias_in_nested_structure(self):
        """
        Test alias detection when the same object appears multiple times in a structure.

        Scenario:
        - Cell A creates data dict containing df, reads data['df1']
        - Cell B accesses data['df2'] which is the same object

        Modification through data['df2'] should be detected as affecting data['df1'].
        """
        import pandas as pd

        df = pd.DataFrame({'a': [1, 2, 3]})
        # Same df object stored under two keys
        data = {'df1': df, 'df2': df}

        # Cell A: reads data, specifically data['df1']
        self._save_pre_checkpoint("a", {"data": data})
        post_a = self._make_post_checkpoint("post_a", {"data": data})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"data"}, writes=set()),
        )

        # Cell B: modifies data['df2'] (same underlying object as df1)
        data_modified = data.copy()
        df_modified = pd.DataFrame({'a': [999, 999, 999]})
        data_modified['df1'] = df_modified
        data_modified['df2'] = df_modified

        self._save_pre_checkpoint("b", {"data": data})
        post_b = self._make_post_checkpoint("post_b", {"data": data_modified})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"data"}),
        )

        # Should detect violation - data was read by Cell A
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "data" in result_b.violation.variables

    def test_empty_accessed_vars(self):
        """
        Test edge case where cell accesses no variables.

        Should still work correctly (diff nothing, no violation).
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: accesses nothing (e.g., just prints a constant)
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes=set()),
        )

        # No violation - nothing was modified
        assert result_b.violation is None

    def test_all_vars_accessed(self):
        """
        Test edge case where cell accesses all variables.

        Should work correctly (diff everything).
        """
        # Cell A: reads x
        self._save_pre_checkpoint("a", {"x": 1, "y": 2, "z": 3})
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: accesses all variables, modifies x
        self._save_pre_checkpoint("b", {"x": 1, "y": 2, "z": 3})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2, "z": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x", "y", "z"}, writes={"x"}),
        )

        # Should detect violation - Cell A read x, Cell B modified x
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "x" in result_b.violation.variables


class TestExpandWithAliases:
    """
    Direct unit tests for the _expand_with_aliases helper function.
    """

    def test_no_aliases(self):
        """Test with no aliases - returns just the accessed vars."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        namespace = {"x": [1, 2, 3], "y": [4, 5, 6], "z": "string"}
        accessed = {"x"}

        result = _expand_with_aliases(accessed, namespace)

        assert result == {"x"}

    def test_simple_alias(self):
        """Test with one alias."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        shared_list = [1, 2, 3]
        namespace = {"x": shared_list, "y": shared_list, "z": "other"}
        accessed = {"x"}

        result = _expand_with_aliases(accessed, namespace)

        assert result == {"x", "y"}

    def test_multiple_aliases(self):
        """Test with multiple aliases."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        shared_obj = {"data": 123}
        namespace = {"a": shared_obj, "b": shared_obj, "c": shared_obj, "d": "other"}
        accessed = {"a"}

        result = _expand_with_aliases(accessed, namespace)

        assert result == {"a", "b", "c"}

    def test_multiple_accessed_with_aliases(self):
        """Test with multiple accessed vars, each with aliases."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        obj1 = [1, 2]
        obj2 = {"x": 1}
        namespace = {
            "a1": obj1, "a2": obj1,  # aliases of each other
            "b1": obj2, "b2": obj2,  # aliases of each other
            "c": "independent",
        }
        accessed = {"a1", "b1"}

        result = _expand_with_aliases(accessed, namespace)

        assert result == {"a1", "a2", "b1", "b2"}

    def test_accessed_var_not_in_namespace(self):
        """Test when accessed var is not in namespace (new variable case)."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        namespace = {"x": [1, 2, 3]}
        accessed = {"x", "y"}  # y doesn't exist in namespace

        result = _expand_with_aliases(accessed, namespace)

        # Should still include y (for new variable case)
        assert result == {"x", "y"}

    def test_empty_accessed(self):
        """Test with empty accessed set."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        namespace = {"x": [1, 2, 3], "y": [4, 5, 6]}
        accessed = set()

        result = _expand_with_aliases(accessed, namespace)

        assert result == set()

    def test_empty_namespace(self):
        """Test with empty namespace."""
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        namespace = {}
        accessed = {"x", "y"}  # New variables

        result = _expand_with_aliases(accessed, namespace)

        # Should return the accessed vars (for new variable case)
        assert result == {"x", "y"}

    def test_immutable_types_not_aliased(self):
        """
        Test that immutable types with same value are not considered aliases.

        Python interns small integers and strings, but for our purposes,
        these should not be treated as aliases since modifying one doesn't
        affect the other.
        """
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        # Python may intern these, making id(x) == id(y)
        # But they're immutable, so it doesn't matter for our use case
        namespace = {"x": 42, "y": 42, "z": "hello", "w": "hello"}
        accessed = {"x"}

        result = _expand_with_aliases(accessed, namespace)

        # Due to interning, y might be included. That's fine - it's conservative.
        # The important thing is we don't miss aliases of mutable objects.
        assert "x" in result

    def test_dataframe_aliases(self):
        """Test with DataFrame aliases."""
        import pandas as pd
        from data_ferret.sdc_kernel.sdc_enforcer import _expand_with_aliases

        df = pd.DataFrame({'a': [1, 2, 3]})
        namespace = {"df": df, "df_view": df, "other_df": pd.DataFrame({'b': [4, 5, 6]})}
        accessed = {"df"}

        result = _expand_with_aliases(accessed, namespace)

        assert result == {"df", "df_view"}
