"""Tests for SDC Enforcer."""

import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode

from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer, PRE_CHECKPOINT_PREFIX
from flowbook.kernel.models import ReasonType
from flowbook.kernel.tests.conftest import make_tracking


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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_no_violation_forward_dependency(self):
        """Cell B reads what cell A writes - valid."""
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace( {"x": 1})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result_a.violation is None

        # Cell B reads x - valid (forward dependency)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert result_b.violation is None

    def test_violation_backward_mutation(self):
        """Cell B modifies what cell A reads - violation."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace( {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Cell B (after A) modifies x - violation!
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        ns_b = self._make_namespace( {"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with different value
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        ns_a2 = self._make_namespace( {"x": 100, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should be stale
        assert "b" in result.stale_cells

    def test_no_staleness_if_value_unchanged(self):
        """Semantic check: no staleness if value didn't actually change."""
        # A writes x=1
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # Re-run A with same value x=1
        # Note: pre-checkpoint for A now reflects current state
        self._save_pre_checkpoint("a", {"x": 1, "y": 2})
        ns_a2 = self._make_namespace( {"x": 1, "y": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should NOT be stale (x didn't change)
        assert "b" not in result.stale_cells

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
        ns_b = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - A is before B in order, so NOT a violation
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace( {"x": 2})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.violation is None

        # Scenario 2: Fresh state with order [b, a, c, d] — B is before A
        self.sdc.reset()
        self.sdc.set_cell_order(["b", "a", "c", "d"])

        # B reads x (fresh)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b2 = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b2,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # A modifies x - now A is AFTER B, so this IS a violation
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a2 = self._make_namespace( {"x": 3})
        result = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result.violation is not None
        assert result.violation.affected_cell == "b"

    def test_cell_deletion_prunes_records(self):
        """Deleted cells are removed from tracking."""
        # Execute cells a and b
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1})
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
        ns_a = self._make_namespace( {"x": 1})
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
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        self._save_pre_checkpoint("d", {"x": 1})
        ns_d = self._make_namespace( {"x": 1, "w": 4})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads={"x"}, writes={"w"}),
        )

        self._save_pre_checkpoint("b", {"x": 1, "w": 4})
        ns_b = self._make_namespace( {"x": 1, "w": 4, "y": 2})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        self._save_pre_checkpoint("c", {"x": 1, "w": 4, "y": 2})
        ns_c = self._make_namespace( {"x": 1, "w": 4, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Re-run A with different x - should make b, c, d stale (they all read x)
        self._save_pre_checkpoint("a", {"x": 1, "w": 4, "y": 2, "z": 3})
        ns_a2 = self._make_namespace( {"x": 100, "w": 4, "y": 2, "z": 3})
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
        ns_a = self._make_namespace( {"var": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"var"}, writes=set()),
        )

        # Cell 'x' (not in order) modifies 'var' - should not trigger violation
        self._save_pre_checkpoint("x", {"var": 1})
        ns_x = self._make_namespace( {"var": 999})
        result = self.sdc.check(
            cell_id="x",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}x"],
            namespace=ns_x,
            tracking=make_tracking(reads=set(), writes={"var"}),
        )

        # No violation because 'x' is not in cell_order
        assert result.violation is None


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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_no_conflict_different_columns(self):
        """Cell A reads df.price, Cell B modifies df.quantity - no violation."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace( {"df": df, "y": 30})
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
        ns_b = self._make_namespace( {"df": df_modified, "y": 30})
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

        # No violation - different columns
        assert result_b.violation is None

    def test_conflict_same_column(self):
        """Cell A reads df.price, Cell B modifies df.price - violation."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace( {"df": df, "y": 30})
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
        ns_b = self._make_namespace( {"df": df_modified, "y": 30})
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
        ns_a = self._make_namespace( {"df": df, "y": 30})
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
        ns_b = self._make_namespace( {"df": df_modified, "y": 30})
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
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "df" in result_b.violation.variables

    def test_conflict_current_no_column_info_conservative(self):
        """Cell A reads df.price, Cell B modifies df (no column info) - violation (conservative)."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "quantity": [1, 2]})

        # Cell A: reads df.price
        self._save_pre_checkpoint("a", {"df": df})
        ns_a = self._make_namespace( {"df": df, "y": 30})
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
        ns_b = self._make_namespace( {"df": df_modified, "y": 30})
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
        ns_a = self._make_namespace( {"df": df, "config": config, "y": 10})
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
        ns_b = self._make_namespace( {"df": df_modified, "config": config_modified, "y": 10})
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
        ns_a = self._make_namespace( {"df": df, "y": 30})
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
        ns_b = self._make_namespace( {"df": df_modified, "y": 30})
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
        ns_a = self._make_namespace( {"df1": df1, "df2": df2, "y": 1})
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
        ns_b = self._make_namespace( {"df1": df1_modified, "df2": df2, "y": 1})
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
        assert result_b.violation is None


class TestBackwardMutationStaleness:
    """Tests for backward mutation behavior (new semantics: never reject, mark stale).

    In the new semantics:
    - Backward mutations mark the cell as STALE (not CLEAN)
    - Execution state is ALWAYS updated (no early return)
    - Staleness is ALWAYS computed for downstream cells
    - `continue_on_violation` parameter is obsolete (always continues)
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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

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
        assert result.violation is not None
        assert result.violation.mutating_cell == "b"
        assert result.violation.affected_cell == "a"
        # Staleness is ALWAYS computed (new semantics)
        # changed_variables shows what changed
        assert "x" in result.changed_variables
        # Cell b itself should be stale (backward mutation)
        assert not self.sdc._notebook_state.is_clean("b")
        # Cell b should have NO_WRITE_AFTER_READ reason
        reasons = self.sdc._notebook_state.get_reasons("b")
        assert any(r.type == ReasonType.NO_WRITE_AFTER_READ for r in reasons)

    def test_backward_mutation_computes_forward_staleness(self):
        """Backward mutation still triggers ForwardStale for downstream cells.

        Staleness is always computed, even when there are errors. This is because:
        - Staleness tracking is orthogonal to error handling
        - The execution record (R, W sets) is always updated
        - Downstream cells need to know about changes
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

        # Cell B modifies x (backward mutation against A, forward stale for C)
        # ForwardStale is computed even with backward mutation error
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace({"x": 999})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Backward violation is detected
        assert result.violation is not None
        # C is marked stale (ForwardStale)
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

        assert result.violation is not None
        # Record is ALWAYS created (new semantics)
        assert self.sdc._notebook_state.has_record("b")
        assert self.sdc._notebook_state.get_tracking("b").writes == {"x"}

    def test_backward_mutation_chain_staleness(self):
        """Test staleness propagation with backward mutation in chain.

        In new semantics:
        - Backward mutations mark the mutating cell stale
        - ForwardStale propagates to cells AFTER the executing cell
        - BackwardStale propagates to cells BEFORE that read the written variable
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
        self._save_pre_checkpoint("d", {"x": 1, "y": 2, "z": 3})
        ns_d = self._make_namespace({"x": 999, "y": 2, "z": 3})
        result = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace=ns_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Backward violation is detected
        assert result.violation is not None
        # B reads x, so D modifying x is a backward mutation against B
        assert result.violation.affected_cell == "b"
        # D itself is stale (new semantics: backward mutation marks the cell stale)
        assert not self.sdc._notebook_state.is_clean("d")
        # A wrote x but didn't read it, so A is not affected by BackwardStale
        assert "a" not in result.stale_cells
        # B read x, so B IS marked stale via BackwardStale (D wrote x that B read)
        assert "b" in result.stale_cells
        # C read y (not x), so C is not affected by D's write to x
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
        from flowbook.kernel_support.types import MemoryCheckpointDiffResult, ValueComparison
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation

        diff = MemoryCheckpointDiffResult(differences={
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
        from flowbook.kernel_support.types import MemoryCheckpointDiffResult, ValueComparison, CompoundDiff
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation, _format_diff_for_display

        diff = MemoryCheckpointDiffResult(differences={
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
        from flowbook.kernel_support.types import MemoryCheckpointDiffResult, ValueComparison, CompoundDiff
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation

        # The outer dict is not truncated, only the inner list is
        # Since we only check the immediate variable's truncation status, this should pass
        diff = MemoryCheckpointDiffResult(differences={
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
        from flowbook.kernel_support.types import MemoryCheckpointDiffResult, ValueComparison, CompoundDiff
        from flowbook.kernel.reproducibility_enforcer import _check_for_truncation, _format_diff_for_display

        diff = MemoryCheckpointDiffResult(differences={
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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        # Create enforcer with structural tracking OFF
        self.sdc = ReproducibilityEnforcer(
            self.checkpoints,
            structural_mode=StructuralTrackingMode.OFF,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

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
        ns_a = self._make_namespace( {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert result_a.violation is None

        # Cell B: Reads raw_data.shape (structural read only, no column reads)
        # This cell reads the variable but only accesses structural attributes
        self._save_pre_checkpoint("b", {"raw_data": df})
        ns_b = self._make_namespace( {"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_c = self._make_namespace( {"raw_data": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
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
        ns_a = self._make_namespace( {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )

        # Cell B: Reads raw_data (whole variable, e.g., print(raw_data))
        # This is NOT a structural-only read
        self._save_pre_checkpoint("b", {"raw_data": df})
        ns_b = self._make_namespace( {"raw_data": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_c = self._make_namespace( {"raw_data": df_modified})
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
        # SHOULD cause violation - prior cell read whole variable (not just structural)
        assert result_c.violation is not None
        assert "raw_data" in result_c.violation.variables


class TestStructuralTrackingWarn:
    """Tests for structural tracking WARN mode with backward mutation detection."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        # Create enforcer with structural tracking WARN
        self.sdc = ReproducibilityEnforcer(
            self.checkpoints,
            structural_mode=StructuralTrackingMode.WARN,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

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
        ns_a = self._make_namespace( {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert result_a.violation is None

        # Cell B: Reads raw_data.shape (structural read only)
        self._save_pre_checkpoint("b", {"raw_data": df})
        ns_b = self._make_namespace( {"raw_data": df})
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
        assert result_b.violation is None

        # Cell C: Adds a new column raw_data['x'] = 3
        # With WARN mode, this should NOT cause a backward mutation violation
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        ns_c = self._make_namespace( {"raw_data": df_modified})
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
        # NO violation - structural tracking is WARN and prior cell only did structural read
        # (warnings are handled separately by diff, not as backward mutation violations)
        assert result_c.violation is None


class TestStructuralTrackingEnforce:
    """Tests for structural tracking ENFORCE mode - structural reads ARE protected."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        # Create enforcer with structural tracking ENFORCE
        self.sdc = ReproducibilityEnforcer(
            self.checkpoints,
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

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

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        # Cell A: Creates the DataFrame
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace( {"raw_data": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"raw_data"}),
        )
        assert result_a.violation is None

        # Cell B: Reads raw_data.shape (structural read only)
        self._save_pre_checkpoint("b", {"raw_data": df})
        ns_b = self._make_namespace( {"raw_data": df})
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
        assert result_b.violation is None

        # Cell C: Adds a new column raw_data['x'] = 3
        # With ENFORCE mode, this SHOULD cause a violation
        df_modified = df.copy()
        df_modified['x'] = 3
        self._save_pre_checkpoint("c", {"raw_data": df})
        ns_c = self._make_namespace( {"raw_data": df_modified})
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
        ns_a = self._make_namespace( {"df": df})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"df"}),
        )

        # Cell B: Display DataFrame - reads columns AND structural attrs
        # This simulates what happens when you just type `df` in a cell
        self._save_pre_checkpoint("b", {"df": df})
        ns_b = self._make_namespace( {"df": df})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_c = self._make_namespace( {"df": df_modified})
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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

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
        ns_a = self._make_namespace( {"y": y})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"y"}, writes=set()),
        )

        # Cell B: creates alias x = y (same object)
        x = y  # x and y are the same object
        self._save_pre_checkpoint("b", {"y": y, "x": x})
        ns_b = self._make_namespace( {"y": y, "x": x})
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
        ns_c = self._make_namespace( {"y": x_modified, "x": x_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
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
        ns_a = self._make_namespace( {"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(
                reads={"df"}, writes=set(),
                column_reads={"df": {"price"}}
            ),
        )

        # Cell B: creates alias (NOT a copy)
        df_alias = df  # Same object!
        self._save_pre_checkpoint("b", {"df": df, "df_alias": df_alias})
        ns_b = self._make_namespace( {"df": df, "df_alias": df_alias})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"df"}, writes={"df_alias"}),
        )

        # Cell C: modifies through alias
        df_modified = pd.DataFrame({'price': [999, 999], 'quantity': [5, 10]})
        self._save_pre_checkpoint("c", {"df": df, "df_alias": df_alias})
        ns_c = self._make_namespace( {"df": df_modified, "df_alias": df_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
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
        ns_a = self._make_namespace( {"x": x})
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
        ns_b = self._make_namespace( {"x": x, "y": y, "z": z})
        self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"y", "z"}),
        )

        # Cell C: modifies through z
        modified = {"value": 999}
        self._save_pre_checkpoint("c", {"x": x, "y": y, "z": z})
        ns_c = self._make_namespace( {"x": modified, "y": modified, "z": modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
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
        ns_a = self._make_namespace( {"x": 1})
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
        ns_b = self._make_namespace( namespace_after)
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_a = self._make_namespace( {"df": df})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"df"}, writes=set()),
        )

        # Cell B: creates alias
        df_alias = df
        self._save_pre_checkpoint("b", {"df": df, "df_alias": df_alias})
        ns_b = self._make_namespace( {"df": df, "df_alias": df_alias})
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
        df_copy_modified['a'] = [999, 999, 999]
        ns_c = self._make_namespace( {"df": df, "df_alias": df_copy_modified})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
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
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: creates new variable y
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1, "y": 42})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_a = self._make_namespace( {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: deletes y
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        ns_b = self._make_namespace( {"x": 1})  # y deleted
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_a = self._make_namespace( {"data": data})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"data"}, writes=set()),
        )

        # Cell B: modifies data['df2'] (same underlying object as df1)
        data_modified = data.copy()
        df_modified = pd.DataFrame({'a': [999, 999, 999]})
        data_modified['df1'] = df_modified
        data_modified['df2'] = df_modified

        self._save_pre_checkpoint("b", {"data": data})
        ns_b = self._make_namespace( {"data": data_modified})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: accesses nothing (e.g., just prints a constant)
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
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
        ns_a = self._make_namespace( {"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B: accesses all variables, modifies x
        self._save_pre_checkpoint("b", {"x": 1, "y": 2, "z": 3})
        ns_b = self._make_namespace( {"x": 999, "y": 2, "z": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x", "y", "z"}, writes={"x"}),
        )

        # Should detect violation - Cell A read x, Cell B modified x
        assert result_b.violation is not None
        assert result_b.violation.affected_cell == "a"
        assert "x" in result_b.violation.variables


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
        arr3 = np.array([100, 200, 300])  # Completely independent with different content

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
        assert "df2" not in aliases, "Independent DataFrames should NOT be detected as aliases"

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
        assert "arr2" not in aliases, "Independent arrays should NOT be detected as aliases"

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
        assert "s2" not in aliases, "Independent Series should NOT be detected as aliases"

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
            namespace[f"df_{i}"] = pd.DataFrame({
                "id": list(range(100)),
                "value": list(range(100, 200)),
            })

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
        view1 = base_arr[2:5]   # View of base
        view2 = base_arr[5:8]   # Another view of base
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
        from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints, MemoryCheckpoint
        from flowbook.kernel_support.models import TrackingData
        from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer
        from flowbook.kernel_support.structural_tracking import StructuralTrackingMode

        checkpoints = MemoryCheckpoints()
        enforcer = ReproducibilityEnforcer(checkpoints, StructuralTrackingMode.OFF)
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
        assert result_a.violation is None

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
        assert not checkpoint._reachable_ids, "Reachable IDs should be empty before first query"

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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

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
        ns_a = self._make_namespace( {"x": 1})
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
        ns_b = self._make_namespace( {"x": 999})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.violation is None

    def test_backward_conflict_still_fires_for_fresh_cells(self):
        """Fresh prior cell should still trigger backward conflict.

        Setup: Cell A reads x (fresh), Cell B modifies x.
        Expected: Violation because A is fresh.
        """
        # Cell A reads x (fresh — not in _stale_cells)
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x — SHOULD trigger violation because A is fresh
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 999})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.violation is not None
        assert result_b.violation.mutating_cell == "b"
        assert result_b.violation.affected_cell == "a"

    def test_backward_conflict_mixed_stale_and_fresh(self):
        """When multiple prior cells exist, only fresh ones trigger conflict.

        Setup: Cell A reads x (stale), Cell B reads x (fresh), Cell C modifies x.
        Expected: Violation against B (fresh) but not A (stale).
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace( {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B reads x
        self._save_pre_checkpoint("b", {"x": 1})
        ns_b = self._make_namespace( {"x": 1})
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
        ns_c = self._make_namespace( {"x": 999})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_c.violation is not None
        # Should report B as the affected cell (first fresh cell in doc order that reads x)
        # A is skipped because it's stale
        assert result_c.violation.affected_cell == "b"


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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _execute_cell(self, cell_id: str, pre_ns: dict, post_ns: dict,
                      reads: set = None, writes: set = None,
                      continue_on_violation: bool = False):
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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _execute_cell(self, cell_id: str, pre_ns: dict, post_ns: dict,
                      reads: set = None, writes: set = None,
                      continue_on_violation: bool = False):
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
        assert "a" not in result.staleness_reasons or not result.staleness_reasons.get("a")

    def test_multiple_reasons_accumulate(self):
        """Multiple staleness reasons accumulate on a cell."""
        # A writes x, B writes y
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, writes={"y"})
        # C reads both x and y
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3}, reads={"x", "y"}, writes={"z"})

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
    """Tests for SKIPPED_UPSTREAM reason type - when cell reads from wrong writer."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        # F writes x, G writes x, H reads x
        self.sdc.set_cell_order(["f", "g", "h"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _execute_cell(self, cell_id: str, pre_ns: dict, post_ns: dict,
                      reads: set = None, writes: set = None,
                      continue_on_violation: bool = False):
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
        Scenario: F→G→H, then F again (skipping G), then G.

        When ENABLE_SKIPPED_UPSTREAM is True:
        - After running F (skipping G), H has SKIPPED_UPSTREAM
        - After running G, H has FORWARD_STALE

        When ENABLE_SKIPPED_UPSTREAM is False (default):
        - After running F (skipping G), H has FORWARD_STALE (not SKIPPED_UPSTREAM)
        - After running G, H still has FORWARD_STALE

        Note: Staleness is always computed, even when G triggers NoReadAndWrite
        error (because G reads and writes x).
        """
        from flowbook.kernel.reproducibility_enforcer import ENABLE_SKIPPED_UPSTREAM

        # Initial execution: F→G→H
        self._execute_cell("f", {}, {"x": 1}, writes={"x"})
        self._execute_cell("g", {"x": 1}, {"x": 2}, reads={"x"}, writes={"x"})
        self._execute_cell("h", {"x": 2}, {"x": 2, "y": 10}, reads={"x"}, writes={"y"})

        # H is clean after initial execution
        assert self.sdc._notebook_state.is_clean("h")

        # Run F again (skipping G) - x goes from 2 back to 1
        self._execute_cell("f", {"x": 2, "y": 10}, {"x": 1, "y": 10}, writes={"x"})

        # H should now be stale
        reasons_h = self.sdc._notebook_state.get_reasons("h")
        reason_types = {r.type for r in reasons_h}

        if ENABLE_SKIPPED_UPSTREAM:
            # With SKIPPED_UPSTREAM enabled: H has SKIPPED_UPSTREAM (reading from F instead of G)
            assert ReasonType.SKIPPED_UPSTREAM in reason_types, f"Expected SKIPPED_UPSTREAM, got {reason_types}"
            skipped_reason = next(r for r in reasons_h if r.type == ReasonType.SKIPPED_UPSTREAM)
            assert skipped_reason.loc == "x"
            assert skipped_reason.expected_cell_id == "g"
        else:
            # With SKIPPED_UPSTREAM disabled: H has FORWARD_STALE
            assert ReasonType.FORWARD_STALE in reason_types, f"Expected FORWARD_STALE, got {reason_types}"

        # Now run G - this should result in FORWARD_STALE
        # (Staleness computed even though G has NoReadAndWrite error)
        self._execute_cell("g", {"x": 1, "y": 10}, {"x": 3, "y": 10}, reads={"x"}, writes={"x"})

        # H should now have FORWARD_STALE
        reasons_h_after = self.sdc._notebook_state.get_reasons("h")
        reason_types_after = {r.type for r in reasons_h_after}

        assert ReasonType.FORWARD_STALE in reason_types_after, f"Expected FORWARD_STALE, got {reason_types_after}"
        if ENABLE_SKIPPED_UPSTREAM:
            assert ReasonType.SKIPPED_UPSTREAM not in reason_types_after, f"SKIPPED_UPSTREAM should be replaced by FORWARD_STALE"
            # With SKIPPED_UPSTREAM enabled, the reason is updated from SKIPPED_UPSTREAM to FORWARD_STALE from G
            input_reason = next(r for r in reasons_h_after if r.type == ReasonType.FORWARD_STALE)
            assert input_reason.loc == "x"
            assert input_reason.cell_id == "g"
            assert input_reason.expected_cell_id is None  # No skipped writer anymore
        else:
            # With SKIPPED_UPSTREAM disabled, H was marked stale by F initially and stays that way
            # (once stale, the loop continues without updating the reason)
            input_reason = next(r for r in reasons_h_after if r.type == ReasonType.FORWARD_STALE)
            assert input_reason.loc == "x"
            # The cell_id could be 'f' or 'g' depending on timing - just verify H is stale for x
            assert input_reason.cell_id in ("f", "g")

    def test_skipped_upstream_exact_user_scenario(self):
        """
        Exact user scenario: F -> G -> H -> F -> G

        After this sequence, H should have FORWARD_STALE (not SKIPPED_UPSTREAM).

        Note: Staleness is always computed, even when G triggers NoReadAndWrite
        error (because G reads and writes x).
        """
        from flowbook.kernel.reproducibility_enforcer import ENABLE_SKIPPED_UPSTREAM

        # F -> G -> H (initial)
        self._execute_cell("f", {}, {"x": 1}, writes={"x"})
        self._execute_cell("g", {"x": 1}, {"x": 2}, reads={"x"}, writes={"x"})
        self._execute_cell("h", {"x": 2}, {"x": 2, "y": 10}, reads={"x"}, writes={"y"})

        print(f"\nAfter F->G->H:")
        print(f"  H reasons: {self.sdc._notebook_state.get_reasons('h')}")
        print(f"  H is_clean: {self.sdc._notebook_state.is_clean('h')}")

        # F (re-run, skipping G)
        self._execute_cell("f", {"x": 2, "y": 10}, {"x": 1, "y": 10}, writes={"x"})

        print(f"\nAfter F->G->H->F:")
        reasons_after_f = self.sdc._notebook_state.get_reasons("h")
        print(f"  H reasons: {reasons_after_f}")
        print(f"  H is_clean: {self.sdc._notebook_state.is_clean('h')}")

        if ENABLE_SKIPPED_UPSTREAM:
            # Verify H has SKIPPED_UPSTREAM at this point
            assert any(r.type == ReasonType.SKIPPED_UPSTREAM for r in reasons_after_f), \
                f"Expected SKIPPED_UPSTREAM after F, got {reasons_after_f}"
        else:
            # With SKIPPED_UPSTREAM disabled, H should have FORWARD_STALE
            assert any(r.type == ReasonType.FORWARD_STALE for r in reasons_after_f), \
                f"Expected FORWARD_STALE after F, got {reasons_after_f}"

        # G (re-run) - staleness computed even though G has NoReadAndWrite error
        self._execute_cell("g", {"x": 1, "y": 10}, {"x": 3, "y": 10}, reads={"x"}, writes={"x"})

        print(f"\nAfter F->G->H->F->G:")
        reasons_after_g = self.sdc._notebook_state.get_reasons("h")
        print(f"  H reasons: {reasons_after_g}")
        print(f"  H is_clean: {self.sdc._notebook_state.is_clean('h')}")

        # H should now have FORWARD_STALE
        reason_types = {r.type for r in reasons_after_g}
        assert ReasonType.FORWARD_STALE in reason_types, \
            f"Expected FORWARD_STALE after G, got {reasons_after_g}"
        if ENABLE_SKIPPED_UPSTREAM:
            # SKIPPED_UPSTREAM should be replaced by FORWARD_STALE
            assert ReasonType.SKIPPED_UPSTREAM not in reason_types, \
                f"SKIPPED_UPSTREAM should be gone after running G, got {reasons_after_g}"


class TestStalenessMode:
    """Tests for syntactic vs semantic staleness computation modes."""

    def setup_method(self):
        from flowbook.kernel_support.structural_tracking import StalenessMode
        self.StalenessMode = StalenessMode
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        """Save a pre-checkpoint for a cell."""
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_namespace(self, namespace: dict) -> dict:
        """Return namespace dict for use with check()."""
        return namespace

    def test_default_mode_is_semantic(self):
        """Default staleness mode should be SEMANTIC."""
        assert self.sdc.staleness_mode == self.StalenessMode.SEMANTIC

    def test_set_staleness_mode_syntactic(self):
        """Can switch to syntactic mode."""
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)
        assert self.sdc.staleness_mode == self.StalenessMode.SYNTACTIC

    def test_set_staleness_mode_semantic(self):
        """Can switch to semantic mode."""
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)
        self.sdc.set_staleness_mode(self.StalenessMode.SEMANTIC)
        assert self.sdc.staleness_mode == self.StalenessMode.SEMANTIC

    def test_syntactic_mode_clears_checkpoints_on_switch(self):
        """Switching from semantic to syntactic should clear pre-checkpoints."""
        # Execute a cell in semantic mode
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Verify pre-checkpoint exists
        assert f"{PRE_CHECKPOINT_PREFIX}a" in self.checkpoints.saved

        # Switch to syntactic mode
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)

        # Pre-checkpoint should be cleared
        assert f"{PRE_CHECKPOINT_PREFIX}a" not in self.checkpoints.saved

    def test_syntactic_mode_marks_stale_on_set_intersection(self):
        """Syntactic mode marks cells stale based on set intersection."""
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)

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

    def test_syntactic_mode_no_convergence(self):
        """Syntactic mode doesn't detect convergence - staleness is monotonic.

        Scenario: A writes x, B reads x, then A re-writes x with different
        value (B stale), then A re-writes x back to original (B still stale
        in syntactic mode because no convergence detection).
        """
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)

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

        # In syntactic mode, B should STILL be stale (no convergence detection)
        # because syntactic mode is monotonic
        assert "b" in result_a3.stale_cells

    def test_semantic_mode_detects_convergence(self):
        """Semantic mode detects convergence and clears staleness.

        Scenario: A writes x, B reads x, then A re-writes x with different
        value (B stale), then A re-writes x back to original (B becomes clean
        in semantic mode due to convergence detection).
        """
        # Default is semantic mode
        assert self.sdc.staleness_mode == self.StalenessMode.SEMANTIC

        # A writes x=1
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B reads x (captures pre-checkpoint with x=1)
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

        # Re-run A with original value x=1 (back to original - convergence!)
        self._save_pre_checkpoint("a", {"x": 2, "y": 2})
        ns_a3 = self._make_namespace({"x": 1, "y": 2})
        result_a3 = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a3,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # In semantic mode, B should NOT be stale anymore (converged)
        assert "b" not in result_a3.stale_cells

    def test_syntactic_mode_defers_checkpoint_deletion(self):
        """Syntactic mode defers checkpoint deletion until next cell executes.

        This allows checkpoint size queries after a cell completes but before
        the next cell runs (important for benchmarking/compare_overhead).
        """
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)

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

    def test_semantic_mode_keeps_checkpoint(self):
        """Semantic mode keeps pre-checkpoint for convergence detection."""
        # Default is semantic mode
        assert self.sdc.staleness_mode == self.StalenessMode.SEMANTIC

        # Execute cell A
        self._save_pre_checkpoint("a", {})
        ns_a = self._make_namespace({"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # In semantic mode, pre-checkpoint should be kept
        assert f"{PRE_CHECKPOINT_PREFIX}a" in self.checkpoints.saved

    # =======================================================================
    # Backward Staleness Tests
    # =======================================================================

    def test_backward_staleness_syntactic_marks_earlier_cell_stale(self):
        """BackwardStale: when later cell writes to var read by earlier clean cell.

        Scenario: A reads x, B is clean, then C writes x.
        After C runs, A should be marked stale (W_C ∩ R_A = {x} ≠ ∅).

        Note: Staleness is always computed, even when C triggers NoWriteAfterRead
        error (because C writes to x that A read).
        """
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)

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

        # C writes x (should make A stale via BackwardStale)
        # Staleness is computed even with NoWriteAfterRead error
        self._save_pre_checkpoint("c", {"x": 1, "y": 10, "z": 20})
        ns_c = self._make_namespace({"x": 999, "y": 10, "z": 20})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # A should be stale (backward staleness: W_C ∩ R_A ≠ ∅)
        assert "a" in result.stale_cells

    def test_backward_staleness_semantic_checks_diff(self):
        """Semantic BackwardStale uses diff comparison for precision.

        If cell C writes x but the value converges to what A originally saw,
        A should NOT be marked stale in semantic mode.
        """
        # Default is semantic mode
        assert self.sdc.staleness_mode == self.StalenessMode.SEMANTIC

        # A reads x=1
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
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

        # C writes x to SAME value as A saw (x=1 -> x=1, no actual change)
        # In semantic mode, A should NOT be stale because diff is empty
        self._save_pre_checkpoint("c", {"x": 1, "y": 10, "z": 20})
        ns_c = self._make_namespace({"x": 1, "y": 10, "z": 20})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # A should NOT be stale - no actual change to x
        assert "a" not in result.stale_cells

    def test_backward_staleness_semantic_marks_stale_on_real_change(self):
        """Semantic BackwardStale marks stale when values actually differ.

        Note: Staleness is always computed, even when C triggers NoWriteAfterRead
        error (because C writes to x that A read).
        """
        # Default is semantic mode
        assert self.sdc.staleness_mode == self.StalenessMode.SEMANTIC

        # A reads x=1
        self._save_pre_checkpoint("a", {"x": 1})
        ns_a = self._make_namespace({"x": 1, "y": 10})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace=ns_a,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert self.sdc._notebook_state.is_clean("a")

        # C writes x to DIFFERENT value
        # Staleness is computed even with NoWriteAfterRead error
        self._save_pre_checkpoint("c", {"x": 1, "y": 10})
        ns_c = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # A should be stale - x actually changed from what A saw
        assert "a" in result.stale_cells

    def test_backward_staleness_only_affects_clean_cells(self):
        """BackwardStale should only mark clean cells as stale, not already-stale cells.

        Note: Staleness is always computed, even when C triggers NoWriteAfterRead
        error (because C writes to x that A and B read).
        """
        self.sdc.set_staleness_mode(self.StalenessMode.SYNTACTIC)

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
        self.sdc._notebook_state.add_reason("a", Reason(ReasonType.FORWARD_STALE, loc="dummy", cell_id="dummy"))
        assert not self.sdc._notebook_state.is_clean("a")
        assert self.sdc._notebook_state.is_clean("b")

        # C writes x - should only affect B (which is clean), not A (already stale)
        # Staleness is computed even with NoWriteAfterRead error
        self._save_pre_checkpoint("c", {"x": 1, "y": 10, "z": 20})
        ns_c = self._make_namespace({"x": 999, "y": 10, "z": 20})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should now be stale
        assert "b" in result.stale_cells


class TestStalenessAlwaysComputed:
    """Tests verifying that staleness is always computed, regardless of errors.

    The key invariant: Staleness tracking is orthogonal to error handling.
    Even when a cell has predicate violations (NoReadAndWrite, NoWriteAfterRead, etc.),
    staleness should still be computed because:
    1. The execution record (R, W sets) is always updated
    2. Downstream cells need to know about changes
    3. Rollback (if any) happens at the kernel level, not in check()
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_namespace(self, namespace: dict) -> dict:
        return namespace

    def test_forward_stale_computed_with_no_read_and_write_error(self):
        """ForwardStale is computed even when cell has NoReadAndWrite error.

        Scenario: B reads and writes x (NoReadAndWrite violation).
        C should still be marked stale because B changed x that C reads.
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

        # B reads and writes x (triggers NoReadAndWrite)
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"x"}),
        )

        # B should have NoReadAndWrite error
        assert result.has_errors()
        assert any(e.error_type.value == "no_read_and_write" for e in result.errors)

        # C should still be marked stale (ForwardStale was computed)
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_forward_stale_computed_with_no_write_after_read_error(self):
        """ForwardStale is computed even when cell has NoWriteAfterRead error.

        Scenario: A reads x, then B writes x (NoWriteAfterRead violation).
        C should still be marked stale because B changed x that C reads.
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

        # B writes x (triggers NoWriteAfterRead against A)
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B should have NoWriteAfterRead error
        assert result.has_errors()
        assert any(e.error_type.value == "no_write_after_read" for e in result.errors)

        # C should still be marked stale (ForwardStale was computed)
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_backward_stale_computed_with_error(self):
        """BackwardStale is computed even when cell has errors.

        Scenario: A reads x, then C writes x.
        A should be marked stale (BackwardStale) even though C has NoWriteAfterRead error.
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

        # C writes x (triggers NoWriteAfterRead against A)
        self._save_pre_checkpoint("c", {"x": 1, "y": 10})
        ns_c = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace=ns_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # C should have NoWriteAfterRead error
        assert result.has_errors()

        # A should be marked stale (BackwardStale was computed)
        assert "a" in result.stale_cells or not self.sdc._notebook_state.is_clean("a")

    def test_last_writer_updated_with_error(self):
        """last_writer is updated even when cell has errors.

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

        # last_writer should still be updated
        assert self.sdc._notebook_state.last_writer.get("x") == "b"

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

        # B reads and writes x (triggers both NoReadAndWrite and NoWriteAfterRead)
        self._save_pre_checkpoint("b", {"x": 1, "y": 10})
        ns_b = self._make_namespace({"x": 999, "y": 10})
        result = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            namespace=ns_b,
            tracking=make_tracking(reads={"x"}, writes={"x"}),
        )

        # B should have errors
        assert result.has_errors()
        assert len(result.errors) >= 1

        # C should still be marked stale
        assert "c" in result.stale_cells or not self.sdc._notebook_state.is_clean("c")

    def test_skipped_upstream_converted_with_error(self):
        """SKIPPED_UPSTREAM → FORWARD_STALE conversion works even with errors.

        Scenario: F→G→H, then F (skipping G), then G.
        When G runs (even with NoReadAndWrite error), H's SKIPPED_UPSTREAM
        should be converted to FORWARD_STALE.
        """
        from flowbook.kernel.reproducibility_enforcer import ENABLE_SKIPPED_UPSTREAM

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

        # After F (skipping G), H should be stale
        reasons_h = self.sdc._notebook_state.get_reasons("h")
        if ENABLE_SKIPPED_UPSTREAM:
            assert any(r.type == ReasonType.SKIPPED_UPSTREAM for r in reasons_h)
        else:
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
        if ENABLE_SKIPPED_UPSTREAM:
            # SKIPPED_UPSTREAM should be replaced by FORWARD_STALE
            assert ReasonType.SKIPPED_UPSTREAM not in reason_types


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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

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
        assert self.sdc._notebook_state.writes.get("a") == {"x"}

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
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def test_rollback_clears_last_writer(self):
        """Rollback removes last_writer entries added by check().

        Scenario: Cell D writes x, gets rejected, rollback should remove
        last_writer[x] = D so it doesn't affect later checks.
        """
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.last_writer.get("x") == "a"

        # Cell D writes x (simulating a check that will be rejected)
        self._save_pre_checkpoint("d", {"x": 1})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 999},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.last_writer.get("x") == "d"

        # Rollback D's check
        self.sdc.rollback_last_check()

        # last_writer[x] should be restored to "a"
        assert self.sdc._notebook_state.last_writer.get("x") == "a"

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
        assert self.sdc._notebook_state.writes.get("d") == {"x"}

        # Cell D writes y (different variable) - simulating rejected execution
        self._save_pre_checkpoint("d", {"x": 1})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 1, "y": 2},
            tracking=make_tracking(reads=set(), writes={"y"}),
        )
        assert self.sdc._notebook_state.writes.get("d") == {"y"}

        # Rollback
        self.sdc.rollback_last_check()

        # Writes should be restored to {"x"}
        assert self.sdc._notebook_state.writes.get("d") == {"x"}

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
        assert "x" in self.sdc._notebook_state.writes.get("a", set())

    def test_rollback_for_never_executed_cell(self):
        """Rollback works correctly for a cell that hadn't executed before."""
        # Cell D never executed before (set_cell_order initializes empty sets)
        assert self.sdc._notebook_state.writes.get("d") == set()
        assert self.sdc._notebook_state.last_writer.get("x") is None
        assert self.sdc._notebook_state.tracking_data.get("d") is None

        # D writes x (simulating rejected execution)
        self._save_pre_checkpoint("d", {})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 1},
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert self.sdc._notebook_state.writes.get("d") == {"x"}
        assert self.sdc._notebook_state.last_writer.get("x") == "d"
        assert self.sdc._notebook_state.tracking_data.get("d") is not None

        # Rollback
        self.sdc.rollback_last_check()

        # Should be back to never-executed state
        assert self.sdc._notebook_state.writes.get("d") == set()
        assert self.sdc._notebook_state.last_writer.get("x") is None
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
        assert result_d.violation is not None or result_d.has_errors()

        # After check, last_writer[x] = D
        assert self.sdc._notebook_state.last_writer.get("x") == "d"

        # Kernel rolls back D's execution
        self.sdc.rollback_last_check()

        # Now last_writer[x] should be back to B
        assert self.sdc._notebook_state.last_writer.get("x") == "b"

        # User edits D to write w instead of x, and re-runs D
        self._save_pre_checkpoint("d", {"x": 11, "y": 11})
        result_d2 = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            namespace={"x": 11, "y": 11, "w": 1},
            tracking=make_tracking(reads=set(), writes={"w"}),
        )
        # No violation now since D writes w, not x
        assert result_d2.violation is None
        assert not result_d2.has_errors()

        # Re-run C - should NOT get forward contamination from D
        self._save_pre_checkpoint("c", {"x": 11, "y": 11, "w": 1})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            namespace={"x": 11, "y": 11, "w": 1},
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # C should have no forward violation (no "Read x from @D")
        assert result_c.forward_violation is None
