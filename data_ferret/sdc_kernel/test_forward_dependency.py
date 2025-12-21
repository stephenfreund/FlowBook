"""
Tests for Forward Dependency Detection.

Forward dependency occurs when a cell reads a variable that a later cell
(in document order) has already written. This means the reading cell is
seeing "future" state that wouldn't exist in top-to-bottom order.

Example:
    Cell order: [A, B, C]
    Execution order: A -> C -> B

    A: x = 10
    B: print(x)  # Reads x
    C: x = 20    # Writes x

    When B executes after C, it prints 20 instead of 10.
    This is a forward dependency: B reads from C (which is later in doc order).
"""

import pytest

from data_ferret.kernel.checkpoint import Checkpoints
from data_ferret.kernel.models import TrackingData

from .sdc_enforcer import SDCEnforcer, PRE_CHECKPOINT_PREFIX, POST_CHECKPOINT_PREFIX, format_forward_dependency_message
from .conftest import make_tracking
from .models import SDCExecutionRecord


class TestForwardDependencyBasic:
    """Basic forward dependency detection tests."""

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

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        """Save a post-checkpoint for a cell."""
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        """Create a post-checkpoint and return it."""
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_basic_forward_dependency_detected(self):
        """
        Basic test: Cell B reads x, Cell C (later) already wrote x.

        Cell order: [a, b, c, d]
        Execution: c first, then b
        Result: b should have forward dependency violation
        """
        # Cell C executes first and writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 20})
        self._save_post_checkpoint("c", {"x": 20})  # Save post checkpoint for forward dep check
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )
        assert result_c.violation is None
        assert result_c.forward_violation is None

        # Cell B executes second and reads x - forward dependency!
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Should have forward dependency violation
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.violation_type == "forward_dependency"
        assert result_b.forward_violation.mutating_cell == "c"
        assert result_b.forward_violation.affected_cell == "b"
        assert "x" in result_b.forward_violation.variables

    def test_no_forward_dependency_when_later_cell_not_executed(self):
        """
        No forward dependency if later cell hasn't executed yet.

        Cell order: [a, b, c]
        Execution: only b
        Result: no violation (c hasn't run yet)
        """
        # Cell B executes and reads x, but C hasn't executed
        self._save_pre_checkpoint("b", {"x": 10})
        post_b = self._make_post_checkpoint("post_b", {"x": 10})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # No forward dependency (c hasn't executed)
        assert result_b.forward_violation is None

    def test_no_forward_dependency_when_no_reads(self):
        """
        No forward dependency if current cell doesn't read anything.
        """
        # Cell C writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 20})
        self._save_post_checkpoint("c", {"x": 20})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B executes but doesn't read x
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20, "y": 5})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # No forward dependency (b doesn't read x)
        assert result_b.forward_violation is None

    def test_no_forward_dependency_when_no_overlap(self):
        """
        No forward dependency if variables don't overlap.

        Cell B reads x, Cell C wrote y - no conflict.
        """
        # Cell C writes y (not x) - value actually changes from nothing to 20
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"y": 20})
        self._save_post_checkpoint("c", {"y": 20})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # Cell B reads x (not y)
        self._save_pre_checkpoint("b", {"x": 10, "y": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 10, "y": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # No forward dependency (different variables)
        assert result_b.forward_violation is None

    def test_forward_dependency_multiple_variables(self):
        """
        Forward dependency with multiple conflicting variables.
        """
        # Cell D writes x, y, z (all values change)
        self._save_pre_checkpoint("d", {})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "y": 2, "z": 3})
        self._save_post_checkpoint("d", {"x": 1, "y": 2, "z": 3})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"x", "y", "z"}),
        )

        # Cell B reads x and y (but not z)
        self._save_pre_checkpoint("b", {"x": 1, "y": 2, "z": 3})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2, "z": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x", "y"}, writes=set()),
        )

        # Forward dependency on x and y
        assert result_b.forward_violation is not None
        assert "x" in result_b.forward_violation.variables
        assert "y" in result_b.forward_violation.variables
        assert "z" not in result_b.forward_violation.variables

    def test_no_forward_dependency_when_value_unchanged(self):
        """
        No forward dependency if later cell wrote but value didn't change.

        This tests the aliasing fix - using checkpoint diffs means we only
        detect actual changes, not just writes.
        """
        # Cell C "writes" x but value doesn't change (x = x scenario)
        self._save_pre_checkpoint("c", {"x": 10})
        post_c = self._make_post_checkpoint("post_c", {"x": 10})  # Same value!
        self._save_post_checkpoint("c", {"x": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),  # tracking says it wrote
        )

        # Cell B reads x
        self._save_pre_checkpoint("b", {"x": 10})
        post_b = self._make_post_checkpoint("post_b", {"x": 10})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # No forward dependency because x didn't actually change
        assert result_b.forward_violation is None


class TestForwardDependencyColumnLevel:
    """Column-level forward dependency detection tests."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_no_conflict_different_columns(self):
        """
        Cell B reads df['price'], Cell C wrote df['qty'] - no conflict.
        """
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "qty": [1, 2]})

        # Cell C writes df['qty']
        self._save_pre_checkpoint("c", {"df": df})
        df_modified = df.copy()
        df_modified["qty"] = [100, 200]
        post_c = self._make_post_checkpoint("post_c", {"df": df_modified})
        self._save_post_checkpoint("c", {"df": df_modified})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_writes={"df": {"qty"}},
            ),
        )

        # Cell B reads df['price'] (different column)
        self._save_pre_checkpoint("b", {"df": df_modified})
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes=set(),
                column_reads={"df": {"price"}},
            ),
        )

        # No forward dependency (different columns)
        assert result_b.forward_violation is None

    def test_conflict_same_column(self):
        """
        Cell B reads df['price'], Cell C wrote df['price'] - conflict!
        """
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "qty": [1, 2]})

        # Cell C writes df['price']
        self._save_pre_checkpoint("c", {"df": df})
        df_modified = df.copy()
        df_modified["price"] = [100, 200]
        post_c = self._make_post_checkpoint("post_c", {"df": df_modified})
        self._save_post_checkpoint("c", {"df": df_modified})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_writes={"df": {"price"}},
            ),
        )

        # Cell B reads df['price'] (same column)
        self._save_pre_checkpoint("b", {"df": df_modified})
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes=set(),
                column_reads={"df": {"price"}},
            ),
        )

        # Forward dependency on df['price']
        assert result_b.forward_violation is not None
        assert "df['price']" in result_b.forward_violation.variables

    def test_conflict_column_read_whole_var_write(self):
        """
        Cell B reads df['price'], Cell C wrote whole df - conflict.

        If later cell wrote the whole variable, column reads conflict.
        """
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "qty": [1, 2]})

        # Cell C writes entire df (no column tracking)
        self._save_pre_checkpoint("c", {"df": df})
        df_new = pd.DataFrame({"a": [1], "b": [2]})
        post_c = self._make_post_checkpoint("post_c", {"df": df_new})
        self._save_post_checkpoint("c", {"df": df_new})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(
                reads=set(),
                writes={"df"},  # Wrote whole variable, no column_writes
            ),
        )

        # Cell B reads df['price']
        self._save_pre_checkpoint("b", {"df": df_new})
        post_b = self._make_post_checkpoint("post_b", {"df": df_new})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes=set(),
                column_reads={"df": {"price"}},
            ),
        )

        # Forward dependency - later cell wrote whole df
        assert result_b.forward_violation is not None
        # The variable name should be in the conflicts
        assert any("df" in v for v in result_b.forward_violation.variables)

    def test_multiple_column_conflicts(self):
        """
        Multiple columns conflict at once.
        """
        import pandas as pd

        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

        # Cell D writes df['a'] and df['b']
        self._save_pre_checkpoint("d", {"df": df})
        df_modified = df.copy()
        df_modified["a"] = [10]
        df_modified["b"] = [20]
        post_d = self._make_post_checkpoint("post_d", {"df": df_modified})
        self._save_post_checkpoint("d", {"df": df_modified})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_writes={"df": {"a", "b"}},
            ),
        )

        # Cell B reads df['a'], df['b'], and df['c']
        self._save_pre_checkpoint("b", {"df": df_modified})
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes=set(),
                column_reads={"df": {"a", "b", "c"}},
            ),
        )

        # Forward dependency on df['a'] and df['b'] but not df['c']
        assert result_b.forward_violation is not None
        assert "df['a']" in result_b.forward_violation.variables
        assert "df['b']" in result_b.forward_violation.variables
        assert "df['c']" not in result_b.forward_violation.variables


class TestForwardDependencyWithBackwardMutation:
    """Tests for interaction between forward dependency and backward mutation."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_both_violations_detected_independently(self):
        """
        Test that both backward mutation and forward dependency are detected.

        Scenario:
        - Cell A reads x
        - Cell C writes x (backward mutation against A) AND wrote y
        - Cell B reads y (forward dependency on C)

        Note: Cell C uses continue_on_violation=True so its record is saved,
        allowing forward dependency detection.
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell C writes x (backward mutation) and y
        # Use continue_on_violation=True so the record is saved
        self._save_pre_checkpoint("c", {"x": 1})
        post_c = self._make_post_checkpoint("post_c", {"x": 999, "y": 2})
        self._save_post_checkpoint("c", {"x": 999, "y": 2})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x", "y"}),
            continue_on_violation=True,  # Save record despite violation
        )
        # C has backward mutation violation against A
        assert result_c.violation is not None
        assert result_c.violation.affected_cell == "a"

        # Cell B reads y (forward dependency on C)
        self._save_pre_checkpoint("b", {"x": 999, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"y"}, writes=set()),
        )

        # B has forward dependency on C
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.mutating_cell == "c"
        assert "y" in result_b.forward_violation.variables

    def test_same_cell_has_both_violations(self):
        """
        Single cell can have both backward mutation AND forward dependency.

        Scenario:
        - Cell A reads x
        - Cell D writes y
        - Cell B writes x (backward) AND reads y (forward)
        """
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell D writes y
        self._save_pre_checkpoint("d", {"x": 1})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "y": 2})
        self._save_post_checkpoint("d", {"x": 1, "y": 2})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"y"}),
        )

        # Cell B: writes x (backward mutation against A) AND reads y (forward dep on D)
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "y": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"y"}, writes={"x"}),
        )

        # Both violations detected
        assert result_b.violation is not None  # Backward mutation
        assert result_b.violation.violation_type == "backward_mutation"
        assert result_b.violation.affected_cell == "a"
        assert "x" in result_b.violation.variables

        assert result_b.forward_violation is not None  # Forward dependency
        assert result_b.forward_violation.violation_type == "forward_dependency"
        assert result_b.forward_violation.mutating_cell == "d"
        assert "y" in result_b.forward_violation.variables


class TestForwardDependencyWithContinueOnViolation:
    """Tests for forward dependency with continue_on_violation parameter."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_forward_dependency_detected_with_continue(self):
        """
        Forward dependency is still detected with continue_on_violation=True.
        """
        # Cell C writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 20})
        self._save_post_checkpoint("c", {"x": 20})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x with continue_on_violation=True
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
            continue_on_violation=True,
        )

        # Forward dependency still detected
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.violation_type == "forward_dependency"


class TestForwardDependencyMessageFormatting:
    """Tests for forward dependency message formatting."""

    def test_message_format_single_variable(self):
        """Test message formatting with single variable."""
        message = format_forward_dependency_message("@B", "@C", ["x"])

        assert "Forward Dependency" in message
        assert "@B" in message
        assert "@C" in message
        assert "x" in message
        assert "top-to-bottom" in message
        assert "reproducibility" in message

    def test_message_format_multiple_variables(self):
        """Test message formatting with multiple variables."""
        message = format_forward_dependency_message("@A", "@D", ["x", "y", "z"])

        assert "@A" in message
        assert "@D" in message
        # Variables should be mentioned
        assert "x" in message or "y" in message or "z" in message


class TestForwardDependencyViolationType:
    """Tests for violation_type field."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_backward_mutation_has_correct_type(self):
        """Backward mutation violations have type 'backward_mutation'."""
        # Cell A reads x
        self._save_pre_checkpoint("a", {"x": 1})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Cell B modifies x
        self._save_pre_checkpoint("b", {"x": 1})
        post_b = self._make_post_checkpoint("post_b", {"x": 999})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        assert result_b.violation is not None
        assert result_b.violation.violation_type == "backward_mutation"

    def test_forward_dependency_has_correct_type(self):
        """Forward dependency violations have type 'forward_dependency'."""
        # Cell C writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 20})
        self._save_post_checkpoint("c", {"x": 20})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert result_b.forward_violation is not None
        assert result_b.forward_violation.violation_type == "forward_dependency"

    def test_violation_to_dict_includes_type(self):
        """SDCViolation.to_dict() includes violation_type."""
        # Cell C writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 20})
        self._save_post_checkpoint("c", {"x": 20})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        violation_dict = result_b.forward_violation.to_dict()
        assert "violation_type" in violation_dict
        assert violation_dict["violation_type"] == "forward_dependency"


class TestForwardDependencyMultipleLaterCells:
    """Tests for forward dependency with multiple later cells."""

    def setup_method(self):
        self.checkpoints = Checkpoints(
            sanity_check=False,
            convert_dtypes=False,
            warn_classes=False,
        )
        self.sdc = SDCEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_reports_first_conflicting_later_cell(self):
        """
        When multiple later cells wrote the variable, report the first one.

        Cell order: [a, b, c, d, e]
        c, d, e all wrote x
        b reads x - should report c (first later cell with conflict)
        """
        # Cell C writes x (first)
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 1})
        self._save_post_checkpoint("c", {"x": 1})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell D also writes x
        self._save_pre_checkpoint("d", {"x": 1})
        post_d = self._make_post_checkpoint("post_d", {"x": 2})
        self._save_post_checkpoint("d", {"x": 2})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell E also writes x
        self._save_pre_checkpoint("e", {"x": 2})
        post_e = self._make_post_checkpoint("post_e", {"x": 3})
        self._save_post_checkpoint("e", {"x": 3})
        self.sdc.check(
            cell_id="e",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}e"],
            post_checkpoint=post_e,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x - should conflict with c (first later cell)
        self._save_pre_checkpoint("b", {"x": 3})
        post_b = self._make_post_checkpoint("post_b", {"x": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        assert result_b.forward_violation is not None
        # Should report c as the mutating cell (first in document order)
        assert result_b.forward_violation.mutating_cell == "c"
