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

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData

from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer, PRE_CHECKPOINT_PREFIX, POST_CHECKPOINT_PREFIX, format_forward_dependency_message
from flowbook.kernel.tests.conftest import make_tracking
from flowbook.kernel.models import ReproducibilityExecutionRecord


class TestForwardDependencyBasic:
    """Basic forward dependency detection tests."""

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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
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

        assert "Forward Contamination" in message
        assert "@B" in message
        assert "@C" in message
        assert "x" in message
        assert "top-to-bottom" in message
        assert "Re-run upstream cells" in message

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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
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
        """ReproducibilityViolation.to_dict() includes violation_type."""
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
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
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


class TestForwardDependencyStaleness:
    """Tests for staleness computation with forward dependencies."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_staleness_after_forward_dependency_with_continue(self):
        """
        When a forward dependency is detected, the violation is recorded and
        a writer_violation is created for the writer cell.

        Scenario:
        - Cell D executes first, writes x
        - Cell B executes second, reads x (forward dependency on D)
        - Forward violation is detected, writer_violation is created for D

        Note: In the new model, forward contamination blocks execution at the
        kernel level (error). The enforcer detects the violation but the cell
        is not automatically added to stale_cells (the kernel handles rejection).
        """
        # Cell D writes x
        self._save_pre_checkpoint("d", {})
        post_d = self._make_post_checkpoint("post_d", {"x": 10})
        self._save_post_checkpoint("d", {"x": 10})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x - forward dependency on D
        self._save_pre_checkpoint("b", {"x": 10})
        post_b = self._make_post_checkpoint("post_b", {"x": 10})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Forward violation detected - D caused the conflict
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.mutating_cell == "d"
        # cell_is_contaminated removed - forward_violation presence indicates contamination
        assert result_b.forward_violation is not None
        # Writer violation is created for the writer cell (same format as backward mutation)
        assert result_b.writer_violation is not None
        assert result_b.writer_violation.mutating_cell == "d"
        assert result_b.writer_violation.affected_cell == "b"
        assert "x" in result_b.writer_violation.variables
        assert result_b.writer_violation.violation_type == "backward_mutation"

        # Now re-execute D with different value
        # B is stale (contaminated), so BackConflict skips it — no violation
        self._save_pre_checkpoint("d", {"x": 10})
        post_d2 = self._make_post_checkpoint("post_d2", {"x": 999})
        result_d2 = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # No backward violation because B is stale (BackConflict only checks fresh)
        assert result_d2.violation is None

    def test_out_of_order_execution_staleness_chain(self):
        """
        Test staleness propagation when cells execute out of document order.

        Cell order: [a, b, c, d, e]
        Dependencies: a writes x, b reads x writes y, c reads y
        Execution: a -> c -> b (c executes before b)

        When b eventually runs, c should become stale.
        """
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell C reads y (y doesn't exist yet, but that's a different issue)
        # Actually let's say C reads x too
        self._save_pre_checkpoint("c", {"x": 1})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Cell B reads x and writes y
        self._save_pre_checkpoint("b", {"x": 1, "z": 3})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2, "z": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # No staleness from B's execution since it didn't change x
        # (y is new, C doesn't read y)
        assert "c" not in result_b.stale_cells

        # Now re-run A with different x
        self._save_pre_checkpoint("a", {"x": 1, "y": 2, "z": 3})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 100, "y": 2, "z": 3})
        result_a2 = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Both B and C read x, so both should be stale
        assert "b" in result_a2.stale_cells
        assert "c" in result_a2.stale_cells
        # Should be in document order
        assert result_a2.stale_cells == ["b", "c"]

    def test_forward_dependency_transitive_staleness(self):
        """
        Test transitive staleness with forward dependency pattern.

        Cell order: [a, b, c, d]
        Execution order: d -> c -> b -> a
        Dependencies: d writes x, c reads x writes y, b reads y writes z, a reads z

        After all execute, changing d's output should make c stale.
        """
        # Execute in reverse order (all forward dependencies)

        # Cell D writes x
        self._save_pre_checkpoint("d", {})
        post_d = self._make_post_checkpoint("post_d", {"x": 1})
        self._save_post_checkpoint("d", {"x": 1})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell C reads x, writes y (forward dep on d)
        self._save_pre_checkpoint("c", {"x": 1})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "y": 2})
        self._save_post_checkpoint("c", {"x": 1, "y": 2})
        result_c = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
            continue_on_violation=True,  # Continue despite forward dep
        )
        assert result_c.forward_violation is not None

        # Cell B reads y, writes z (forward dep on c)
        self._save_pre_checkpoint("b", {"x": 1, "y": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2, "z": 3})
        self._save_post_checkpoint("b", {"x": 1, "y": 2, "z": 3})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"y"}, writes={"z"}),
            continue_on_violation=True,
        )
        assert result_b.forward_violation is not None

        # Cell A reads z (forward dep on b)
        self._save_pre_checkpoint("a", {"x": 1, "y": 2, "z": 3})
        post_a = self._make_post_checkpoint("post_a", {"x": 1, "y": 2, "z": 3, "w": 4})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads={"z"}, writes={"w"}),
            continue_on_violation=True,
        )
        assert result_a.forward_violation is not None

        # All cells (A, B, C) are forward-contaminated → stale
        assert "a" in self.sdc._stale_cells
        assert "b" in self.sdc._stale_cells
        assert "c" in self.sdc._stale_cells

        # Now re-run D with different x
        # C reads x but is stale (EXEC-CONTAMINATED), so BackConflict skips it
        self._save_pre_checkpoint("d", {"x": 1, "y": 2, "z": 3, "w": 4})
        post_d2 = self._make_post_checkpoint("post_d2", {"x": 999, "y": 2, "z": 3, "w": 4})
        result_d2 = self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # No backward violation — C is stale so BackConflict skips it (Def 1.8.2)
        assert result_d2.violation is None

    def test_staleness_when_later_cell_re_executes(self):
        """
        Test staleness when a later cell (source of forward dep) re-executes.

        Cell order: [a, b, c]
        Execution: c writes x, b reads x (forward dep on c)
        B is EXEC-CONTAMINATED (stale). When c re-executes, no backward violation
        because B is stale (BackConflict only checks fresh cells).
        """
        # Cell C writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 10})
        self._save_post_checkpoint("c", {"x": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x (forward dependency) → contaminated
        self._save_pre_checkpoint("b", {"x": 10})
        post_b = self._make_post_checkpoint("post_b", {"x": 10, "y": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.mutating_cell == "c"
        # Writer violation is created for the writer cell (same format as backward mutation)
        assert result_b.writer_violation is not None
        assert result_b.writer_violation.mutating_cell == "c"
        assert result_b.writer_violation.affected_cell == "b"
        assert result_b.writer_violation.violation_type == "backward_mutation"

        # Re-run C with different value
        # B is stale (contaminated), so BackConflict skips it
        self._save_pre_checkpoint("c", {"x": 10, "y": 20})
        post_c2 = self._make_post_checkpoint("post_c2", {"x": 999, "y": 20})
        result_c2 = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # No backward violation — B is stale so BackConflict skips it
        assert result_c2.violation is None

    def test_no_additional_staleness_when_forward_dep_value_unchanged(self):
        """
        No additional staleness when value doesn't change, but B stays stale from contamination.

        Cell order: [a, b, c]
        Execution: c writes x=10, b reads x (forward dep → contaminated/stale)
        When c re-executes with same x=10, B remains stale (from contamination).
        The test verifies C's re-execution doesn't cause additional issues.
        """
        # Cell C writes x
        self._save_pre_checkpoint("c", {})
        post_c = self._make_post_checkpoint("post_c", {"x": 10})
        self._save_post_checkpoint("c", {"x": 10})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x (forward dependency → contaminated)
        self._save_pre_checkpoint("b", {"x": 10})
        post_b = self._make_post_checkpoint("post_b", {"x": 10, "y": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        assert result_b.forward_violation is not None
        # Writer violation is created for the writer cell (same format as backward mutation)
        assert result_b.writer_violation is not None
        assert result_b.writer_violation.mutating_cell == "c"
        assert result_b.writer_violation.affected_cell == "b"
        assert result_b.writer_violation.violation_type == "backward_mutation"

        # Re-run C with SAME value — no violation, no new staleness
        self._save_pre_checkpoint("c", {"x": 10, "y": 20})
        post_c2 = self._make_post_checkpoint("post_c2", {"x": 10, "y": 20})  # x unchanged
        result_c2 = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # No backward violation (B is stale, skipped by BackConflict)
        assert result_c2.violation is None
        # B is still stale (from contamination), but C's re-execution didn't add staleness
        assert "b" in result_c2.stale_cells  # B was already stale

    def test_mixed_forward_backward_staleness(self):
        """
        Test staleness with both forward and backward dependency scenarios.

        Cell order: [a, b, c, d]
        - A writes x
        - B reads x, writes y
        - C reads y (forward dep if C runs before B)
        - D reads x

        Execute: a -> c -> d -> b
        Then re-run a with different x.
        """
        # Cell A writes x
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell C reads y (y doesn't exist yet - this would normally be an error)
        # Let's say C reads x instead for a cleaner test
        self._save_pre_checkpoint("c", {"x": 1})
        post_c = self._make_post_checkpoint("post_c", {"x": 1, "z": 3})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads={"x"}, writes={"z"}),
        )

        # Cell D reads x
        self._save_pre_checkpoint("d", {"x": 1, "z": 3})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "z": 3, "w": 4})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads={"x"}, writes={"w"}),
        )

        # Cell B reads x, writes y
        self._save_pre_checkpoint("b", {"x": 1, "z": 3, "w": 4})
        post_b = self._make_post_checkpoint("post_b", {"x": 1, "y": 2, "z": 3, "w": 4})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )
        # No violations - B doesn't modify what earlier cells read
        assert result_b.violation is None

        # Now re-run A with different x
        self._save_pre_checkpoint("a", {"x": 1, "y": 2, "z": 3, "w": 4})
        post_a2 = self._make_post_checkpoint("post_a2", {"x": 999, "y": 2, "z": 3, "w": 4})
        result_a2 = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a2,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # B, C, D all read x, so all should be stale
        # Should be in document order
        assert "b" in result_a2.stale_cells
        assert "c" in result_a2.stale_cells
        assert "d" in result_a2.stale_cells
        assert result_a2.stale_cells == ["b", "c", "d"]


class TestForwardDependencyColumnStaleness:
    """Tests for column-level staleness with forward dependencies."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _save_pre_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{PRE_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def test_column_level_staleness_with_forward_dependency(self):
        """
        Test column-level staleness in forward dependency scenario.

        Cell order: [a, b, c]
        - C writes df['price']
        - B reads df['price'] (forward dep on C)

        When C re-executes with different price, B should be stale.
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

        # Cell B reads df['price'] (forward dependency on C)
        self._save_pre_checkpoint("b", {"df": df_modified})
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "total": 300})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"total"},
                column_reads={"df": {"price"}},
            ),
            continue_on_violation=True,
        )
        assert result_b.forward_violation is not None

        # Writer violation is created for the writer cell (same format as backward mutation)
        assert result_b.writer_violation is not None
        assert result_b.writer_violation.mutating_cell == "c"
        assert result_b.writer_violation.affected_cell == "b"
        assert result_b.writer_violation.violation_type == "backward_mutation"

        # Re-run C with different price values
        # B is stale (contaminated), so BackConflict skips it — no violation
        df_modified2 = df.copy()
        df_modified2["price"] = [999, 888]
        self._save_pre_checkpoint("c", {"df": df_modified, "total": 300})
        post_c2 = self._make_post_checkpoint("post_c2", {"df": df_modified2, "total": 300})
        result_c2 = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c2,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_writes={"df": {"price"}},
            ),
        )

        # No backward violation — B is stale so BackConflict skips it
        assert result_c2.violation is None

    def test_no_column_staleness_different_columns(self):
        """
        No staleness when different columns are modified.

        Cell order: [a, b, c]
        - C writes df['qty']
        - B reads df['price'] (forward dep on C for whole df)

        When C re-executes and modifies only qty, B should NOT be stale.
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

        # Cell B reads df['price'] (not qty)
        self._save_pre_checkpoint("b", {"df": df_modified})
        post_b = self._make_post_checkpoint("post_b", {"df": df_modified, "total": 30})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(
                reads={"df"},
                writes={"total"},
                column_reads={"df": {"price"}},
            ),
            continue_on_violation=True,
        )
        # Forward dep is on the column that actually changed
        # B reads price, C wrote qty - so no forward violation at column level
        # But there might be at variable level depending on implementation

        # Re-run C with different qty values (price unchanged)
        df_modified2 = df_modified.copy()
        df_modified2["qty"] = [999, 888]  # Only qty changes
        self._save_pre_checkpoint("c", {"df": df_modified, "total": 30})
        post_c2 = self._make_post_checkpoint("post_c2", {"df": df_modified2, "total": 30})
        result_c2 = self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c2,
            tracking=make_tracking(
                reads={"df"},
                writes={"df"},
                column_writes={"df": {"qty"}},
            ),
        )

        # B reads price, C only changed qty - B should NOT be stale
        assert "b" not in result_c2.stale_cells


class TestForwardContaminationExecContaminated:
    """Tests for EXEC-CONTAMINATED rule: forward-contaminated cells are NOT rejected.

    Per the formal rule (§1.8), when a cell has no backward conflict but IS
    forward-contaminated, execution proceeds and the cell is recorded as stale.
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

    def _make_post_checkpoint(self, name: str, namespace: dict):
        self.checkpoints.save(name, namespace, max_size_mb=None)
        return self.checkpoints.saved[name]

    def _save_post_checkpoint(self, cell_id: str, namespace: dict):
        self.checkpoints.save(f"{POST_CHECKPOINT_PREFIX}{cell_id}", namespace, max_size_mb=None)

    def test_forward_contamination_does_not_reject(self):
        """Forward-contaminated cell should proceed (no backward violation).

        Cell C writes x, then Cell B reads x (forward dependency).
        B should NOT have a backward violation — it should be accepted
        with contamination status.
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

        # Cell B reads x — forward dependency but no backward violation
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # No backward violation
        assert result_b.violation is None
        # Forward violation detected
        assert result_b.forward_violation is not None
        # Cell is contaminated
        # cell_is_contaminated removed - forward_violation presence indicates contamination
        assert result_b.forward_violation is not None

    def test_forward_contaminated_cell_is_detected(self):
        """A forward-contaminated cell should have forward_violation set.

        Note: In the new model, forward contamination blocks execution at the
        kernel level (error), so the cell is NOT added to stale_cells by the
        enforcer. The kernel handles rejection.
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

        # Cell B reads x — forward contaminated
        self._save_pre_checkpoint("b", {"x": 20})
        post_b = self._make_post_checkpoint("post_b", {"x": 20})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes=set()),
        )

        # Forward violation is detected
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.mutating_cell == "c"
        # Writer violation is created for the writer cell (same format as backward mutation)
        assert result_b.writer_violation is not None
        assert result_b.writer_violation.mutating_cell == "c"
        assert result_b.writer_violation.affected_cell == "b"
        assert result_b.writer_violation.violation_type == "backward_mutation"

    def test_forward_contamination_still_propagates_stalefwd(self):
        """StaleFwd should still mark downstream cells stale after EXEC-CONTAMINATED.

        Cell C writes x, Cell B reads x and writes y, Cell D reads y.
        When B executes (contaminated), D should become stale if y changed.
        """
        # First execute D (reads y)
        self._save_pre_checkpoint("d", {"y": 0})
        post_d = self._make_post_checkpoint("post_d", {"y": 0})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads={"y"}, writes=set()),
        )

        # Cell C writes x
        self._save_pre_checkpoint("c", {"y": 0})
        post_c = self._make_post_checkpoint("post_c", {"x": 20, "y": 0})
        self._save_post_checkpoint("c", {"x": 20, "y": 0})
        self.sdc.check(
            cell_id="c",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}c"],
            post_checkpoint=post_c,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Cell B reads x (forward dep on C) and writes y
        self._save_pre_checkpoint("b", {"x": 20, "y": 0})
        post_b = self._make_post_checkpoint("post_b", {"x": 20, "y": 99})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"x"}, writes={"y"}),
        )

        # B is contaminated
        # cell_is_contaminated removed - forward_violation presence indicates contamination
        assert result_b.forward_violation is not None
        # D should be stale because B changed y which D reads
        assert "d" in result_b.stale_cells

    def test_backward_conflict_still_rejects_with_both(self):
        """When both backward and forward violations exist, backward wins (EXEC-REJECT).

        Cell A reads x, Cell D writes z. Cell B modifies x (backward) and reads z (forward).
        B should be rejected (backward takes precedence).
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

        # Cell D writes z
        self._save_pre_checkpoint("d", {"x": 1})
        post_d = self._make_post_checkpoint("post_d", {"x": 1, "z": 2})
        self._save_post_checkpoint("d", {"x": 1, "z": 2})
        self.sdc.check(
            cell_id="d",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}d"],
            post_checkpoint=post_d,
            tracking=make_tracking(reads=set(), writes={"z"}),
        )

        # Cell B modifies x (backward violation against A) AND reads z (forward dep on D)
        self._save_pre_checkpoint("b", {"x": 1, "z": 2})
        post_b = self._make_post_checkpoint("post_b", {"x": 999, "z": 2})
        result_b = self.sdc.check(
            cell_id="b",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}b"],
            post_checkpoint=post_b,
            tracking=make_tracking(reads={"z"}, writes={"x"}),
        )

        # Backward violation should be present
        assert result_b.violation is not None
        assert result_b.violation.violation_type == "backward_mutation"
        assert result_b.violation.affected_cell == "a"

        # Forward violation also present
        assert result_b.forward_violation is not None
        assert result_b.forward_violation.violation_type == "forward_dependency"

    def test_non_contaminated_cell_is_fresh(self):
        """A cell without forward contamination should be recorded as fresh (EXEC-ACCEPT)."""
        # Cell A writes x (no forward dependency)
        self._save_pre_checkpoint("a", {})
        post_a = self._make_post_checkpoint("post_a", {"x": 1})
        result_a = self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=post_a,
            tracking=make_tracking(reads=set(), writes={"x"}),
        )

        # Not contaminated
        # cell_is_contaminated removed - forward_violation absence indicates no contamination
        assert result_a.forward_violation is None
        # Not stale
        assert "a" not in result_a.stale_cells
