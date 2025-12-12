"""Tests for SDC Enforcer."""

import pytest

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints
from data_ferret.kernel.models import TrackingData

from .sdc_enforcer import SDCEnforcer


def make_tracking(
    reads: set = None,
    writes: set = None,
    column_reads: dict = None,
    column_writes: dict = None,
) -> TrackingData:
    """Helper to create TrackingData with optional column tracking."""
    return TrackingData(
        reads_before_writes=reads or set(),
        writes=writes or set(),
        column_reads_before_writes=column_reads or {},
        column_writes=column_writes or {},
    )


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
        self.checkpoints.save(f"_pre_{cell_id}", namespace, max_size_mb=None)

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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result_a.violation is None

        # Cell B reads x - valid (forward dependency)
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B (after A) modifies x - violation!
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with same value x=1
        # Note: pre-checkpoint for A now reflects current state
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 1, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - A is before B in order, so NOT a violation
        # (B can depend on A, that's forward dependency)
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("d", {"x": 1})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "w": 4})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved["_pre_d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads={"x"}, writes={"w"}),
        )

        self._save_pre_checkpoint("b", {"x": 1, "w": 4})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "w": 4, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        self._save_pre_checkpoint("c", {"x": 1, "w": 4, "y": 2})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "w": 4, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved["_pre_c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Re-run A with different x - should make b, c, d stale (they all read x)
        self._save_pre_checkpoint("a", {"x": 1, "w": 4, "y": 2, "z": 3})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "w": 4, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"var"}, writes=set()),
        )

        # Cell 'x' (not in order) modifies 'var' - should not trigger violation
        self._save_pre_checkpoint("x", {"var": 1})
        post_x = self._make_post_checkpoint("post_x", {"var": 999})
        result = self.sdc.check(
            cell_id="x",
            pre_checkpoint=self.checkpoints.saved["_pre_x"],
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
        self.checkpoints.save(f"_pre_{cell_id}", namespace, max_size_mb=None)

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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
        assert "df" in result_b.violation.variables

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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
        self.checkpoints.save(f"_pre_{cell_id}", namespace, max_size_mb=None)

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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B modifies x (backward mutation)
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B modifies x (backward mutation) - but we continue
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x (violation) with default behavior
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
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
            pre_checkpoint=self.checkpoints.saved["_pre_a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x, writes y
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved["_pre_b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell C reads y
        self._save_pre_checkpoint("c", {"x": 1, "y": 2})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved["_pre_c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
        )

        # Cell D modifies x (violation against A) - but we continue
        self._save_pre_checkpoint("d", {"x": 1, "y": 2, "z": 3})
        post_d = self._make_post_checkpoint("post_d", {"x": 999, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved["_pre_d"],
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
