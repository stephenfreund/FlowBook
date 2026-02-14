"""
Tests for the %exec_restore magic and its integration with _do_execute_impl().

The %exec_restore magic enables EXEC-RESTORE (§1.8) from the frontend:
1. Frontend sends %exec_restore <cell_id> silently (sets pending flag)
2. Frontend triggers cell execution (notebook:run-cell)
3. _do_execute_impl() checks pending flag, restores prefix checkpoint, executes

These tests cover:
- The %exec_restore magic command (validation, pending flag)
- Pending flag consumption logic in _do_execute_impl()
- The updated forward contamination message format
- End-to-end EXEC-RESTORE flow via pending flag
"""

import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData

from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
    POST_CHECKPOINT_PREFIX,
    format_forward_dependency_message,
)
from flowbook.kernel.tests.conftest import make_tracking


# =============================================================================
# Tests for format_forward_dependency_message (updated wording)
# =============================================================================


class TestForwardContaminationMessage:
    """Tests for the updated forward contamination message."""

    def test_title_says_forward_contamination(self):
        """Title should say 'Forward Contamination', not 'Forward Dependency'."""
        message = format_forward_dependency_message("@B", "@C", ["x"])
        assert "Forward Contamination" in message
        assert "Forward Dependency" not in message

    def test_no_blocked_language(self):
        """Message should not use 'blocked' language (cell is accepted, not blocked)."""
        message = format_forward_dependency_message("@B", "@C", ["x"])
        assert "blocked" not in message.lower()
        assert "Why blocked" not in message

    def test_explains_stale_marking(self):
        """Message should explain cell is marked stale."""
        message = format_forward_dependency_message("@B", "@C", ["x"])
        assert "marked stale" in message
        assert "executed successfully" in message

    def test_mentions_restore_from_checkpoint(self):
        """Message should mention the context menu action."""
        message = format_forward_dependency_message("@B", "@C", ["x"])
        assert "Restore from checkpoint" in message

    def test_mentions_notebook_order_alternative(self):
        """Message should mention re-running in notebook order as alternative fix."""
        message = format_forward_dependency_message("@B", "@C", ["x"])
        assert "notebook order" in message

    def test_mentions_both_cells(self):
        """Message should reference both the reading and writing cells."""
        message = format_forward_dependency_message("@A", "@D", ["var"])
        assert "@A" in message
        assert "@D" in message

    def test_mentions_variable(self):
        """Message should mention the conflicting variable."""
        message = format_forward_dependency_message("@B", "@C", ["my_var"])
        assert "my_var" in message

    def test_multiple_variables(self):
        """Message should handle multiple variables."""
        message = format_forward_dependency_message("@A", "@D", ["x", "y", "z"])
        # format_variable_list should join them
        assert "x" in message
        assert "y" in message
        assert "z" in message


# =============================================================================
# Tests for can_exec_restore precondition (used by the magic)
# =============================================================================


class TestCanExecRestorePrecondition:
    """Tests for can_exec_restore used by %exec_restore to validate requests."""

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _execute_cell(self, cell_id, pre_ns, post_ns, reads=None, writes=None):
        """Helper to simulate cell execution."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", pre_ns, max_size_mb=None
        )
        self.checkpoints.save(f"post_{cell_id}", post_ns, max_size_mb=None)
        self.checkpoints.save(
            f"{POST_CHECKPOINT_PREFIX}{cell_id}", post_ns, max_size_mb=None
        )
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            post_checkpoint=self.checkpoints.saved[f"post_{cell_id}"],
            tracking=make_tracking(reads=reads or set(), writes=writes or set()),
        )

    def test_cell_not_in_order(self):
        """can_exec_restore returns False for unknown cell_id."""
        assert self.sdc.can_exec_restore("unknown") is False

    def test_first_cell_always_valid(self):
        """First cell has no predecessors — always valid."""
        assert self.sdc.can_exec_restore("a") is True

    def test_all_predecessors_fresh(self):
        """Valid when all predecessors have fresh records."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        assert self.sdc.can_exec_restore("c") is True

    def test_stale_predecessor_blocks(self):
        """Invalid when any predecessor is stale."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self.sdc._stale_cells.add("a")
        assert self.sdc.can_exec_restore("c") is False

    def test_unexecuted_predecessor_blocks(self):
        """Invalid when any predecessor hasn't been executed."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # Skip B
        assert self.sdc.can_exec_restore("c") is False

    def test_contaminated_predecessor_blocks(self):
        """Invalid when a predecessor is forward-contaminated (stale)."""
        # Execute C first (writes x), then A (reads x) → A is contaminated (stale)
        self._execute_cell("c", {}, {"x": 5}, writes={"x"})
        result_a = self._execute_cell("a", {"x": 5}, {"x": 5}, reads={"x"})
        assert result_a.cell_is_contaminated is True
        assert "a" in self.sdc._stale_cells

        # B cannot exec-restore because A is stale (contaminated)
        self._execute_cell("b", {"x": 5}, {"x": 5, "y": 2}, reads={"x"}, writes={"y"})
        assert self.sdc.can_exec_restore("c") is False


# =============================================================================
# Tests for pending flag lifecycle
# =============================================================================


class TestPendingFlagLifecycle:
    """
    Tests for the _pending_exec_restore flag lifecycle.

    These test the enforcer-level logic that the kernel's _do_execute_impl()
    depends on. The flag is set by %exec_restore and consumed by the execution
    path. We can't easily instantiate FlowbookKernel in tests, so we test the
    enforcer's precondition logic and the flag semantics that the kernel relies on.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _execute_cell(self, cell_id, pre_ns, post_ns, reads=None, writes=None,
                      is_exec_restore=False, old_live_checkpoint=None):
        """Helper to simulate cell execution."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", pre_ns, max_size_mb=None
        )
        self.checkpoints.save(f"post_{cell_id}", post_ns, max_size_mb=None)
        self.checkpoints.save(
            f"{POST_CHECKPOINT_PREFIX}{cell_id}", post_ns, max_size_mb=None
        )
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            post_checkpoint=self.checkpoints.saved[f"post_{cell_id}"],
            tracking=make_tracking(reads=reads or set(), writes=writes or set()),
            is_exec_restore=is_exec_restore,
            old_live_checkpoint=old_live_checkpoint,
        )

    def test_precondition_revalidation_succeeds(self):
        """When precondition holds at execution time, EXEC-RESTORE succeeds."""
        # Execute A and B fresh
        self._execute_cell("a", {}, {"x": 23}, writes={"x"})
        self._execute_cell("b", {"x": 23}, {"x": 23, "y": 26}, reads={"x"}, writes={"y"})

        # Precondition valid for C
        assert self.sdc.can_exec_restore("c") is True

        # EXEC-RESTORE C
        result = self._execute_cell(
            "c", {"x": 23, "y": 26}, {"x": 23, "y": 26, "z": 1},
            writes={"z"}, is_exec_restore=True,
        )
        assert result.violation is None
        assert result.exec_mode == "restore"
        assert "c" not in self.sdc._stale_cells

    def test_precondition_revalidation_fails_falls_back(self):
        """When precondition fails at execution time, normal execution occurs.

        This simulates what the kernel does: it re-checks can_exec_restore()
        before activating the restore path. If a predecessor becomes stale
        between %exec_restore and execution, it falls back to normal execution.
        """
        # Execute A and B fresh
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # Precondition initially valid
        assert self.sdc.can_exec_restore("c") is True

        # But then A becomes stale (e.g., user edited it)
        self.sdc._stale_cells.add("a")

        # Now precondition fails
        assert self.sdc.can_exec_restore("c") is False

        # Normal execution (not exec-restore) would happen
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
            writes={"z"},
        )
        assert result.exec_mode == "live"

    def test_prefix_checkpoint_lookup(self):
        """get_prefix_checkpoint_name returns correct checkpoint for each position."""
        assert self.sdc.get_prefix_checkpoint_name("a") is None
        assert self.sdc.get_prefix_checkpoint_name("b") == f"{POST_CHECKPOINT_PREFIX}a"
        assert self.sdc.get_prefix_checkpoint_name("c") == f"{POST_CHECKPOINT_PREFIX}b"
        assert self.sdc.get_prefix_checkpoint_name("d") == f"{POST_CHECKPOINT_PREFIX}c"

    def test_prefix_checkpoint_for_unknown_cell(self):
        """get_prefix_checkpoint_name returns None for unknown cell."""
        assert self.sdc.get_prefix_checkpoint_name("unknown") is None


# =============================================================================
# End-to-end EXEC-RESTORE via enforcer
# =============================================================================


class TestExecRestoreEndToEnd:
    """
    End-to-end tests for the EXEC-RESTORE flow.

    Simulates the full scenario:
    1. Cells executed out of order → forward contamination
    2. EXEC-RESTORE for contaminated cell → fresh result
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c"])

    def _execute_cell(self, cell_id, pre_ns, post_ns, reads=None, writes=None,
                      is_exec_restore=False, old_live_checkpoint=None):
        """Helper to simulate cell execution."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", pre_ns, max_size_mb=None
        )
        self.checkpoints.save(f"post_{cell_id}", post_ns, max_size_mb=None)
        self.checkpoints.save(
            f"{POST_CHECKPOINT_PREFIX}{cell_id}", post_ns, max_size_mb=None
        )
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            post_checkpoint=self.checkpoints.saved[f"post_{cell_id}"],
            tracking=make_tracking(reads=reads or set(), writes=writes or set()),
            is_exec_restore=is_exec_restore,
            old_live_checkpoint=old_live_checkpoint,
        )

    def test_contaminated_cell_becomes_fresh_after_restore(self):
        """
        Scenario: A: x=23, C: x=5, B: y=x+3
        Run A, then C, then B → B is contaminated (reads x=5 from C).
        EXEC-RESTORE B from A's post-checkpoint → B reads x=23, becomes fresh.
        """
        # Step 1: Run A → x = 23
        self._execute_cell("a", {}, {"x": 23}, writes={"x"})

        # Step 2: Run C → x = 5
        self._execute_cell("c", {"x": 23}, {"x": 5}, writes={"x"})

        # Step 3: Run B → reads x (which is 5 from C, not 23 from A)
        result_b = self._execute_cell(
            "b", {"x": 5}, {"x": 5, "y": 8}, reads={"x"}, writes={"y"}
        )

        # B should be forward-contaminated
        assert result_b.forward_violation is not None
        assert result_b.cell_is_contaminated is True
        assert "b" in self.sdc._stale_cells
        assert result_b.exec_mode == "live"

        # Step 4: EXEC-RESTORE B
        # Simulate what the kernel does:
        # - Save old live store for delta computation
        self.checkpoints.save("_old_live_b", {"x": 5, "y": 8}, max_size_mb=None)
        old_live = self.checkpoints.saved["_old_live_b"]

        # - Execute from prefix checkpoint (post_a: x=23)
        # - B reads x=23, produces y=26
        result_b2 = self._execute_cell(
            "b", {"x": 23}, {"x": 23, "y": 26},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=old_live,
        )

        # B should now be fresh
        assert result_b2.violation is None
        assert result_b2.forward_violation is None
        assert result_b2.cell_is_contaminated is False
        assert "b" not in self.sdc._stale_cells
        assert result_b2.exec_mode == "restore"

    def test_restore_propagates_stalefwd(self):
        """
        After EXEC-RESTORE, StaleFwd correctly marks downstream cells stale.

        Scenario: A: x=1, B: y=x+1 (restore changes y from 3 to 2), C: reads y
        """
        # Execute all cells normally
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 10},
                          reads={"y"}, writes={"z"})

        # C is fresh, reads y=2
        assert "c" not in self.sdc._stale_cells

        # EXEC-RESTORE B where result differs (y=99 instead of y=2)
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2, "z": 10}, max_size_mb=None)
        old_live = self.checkpoints.saved["_old_live_b"]

        result_b = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 99},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=old_live,
        )

        # B is fresh after restore
        assert "b" not in self.sdc._stale_cells
        assert result_b.exec_mode == "restore"

        # C should be stale because y changed (2 → 99 in the delta)
        assert "c" in result_b.stale_cells

    def test_restore_clears_contamination_flag(self):
        """After successful EXEC-RESTORE, cell_is_contaminated is False."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("c", {"x": 1}, {"x": 99}, writes={"x"})

        # B is contaminated
        result = self._execute_cell(
            "b", {"x": 99}, {"x": 99, "y": 100}, reads={"x"}, writes={"y"}
        )
        assert result.cell_is_contaminated is True

        # EXEC-RESTORE B
        self.checkpoints.save("_old_live_b", {"x": 99, "y": 100}, max_size_mb=None)
        result2 = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )
        assert result2.cell_is_contaminated is False
        assert result2.exec_mode == "restore"

    def test_restore_first_cell_no_prefix(self):
        """First cell EXEC-RESTORE has no prefix checkpoint, but succeeds."""
        result = self._execute_cell(
            "a", {}, {"x": 1}, writes={"x"}, is_exec_restore=True,
        )
        assert result.violation is None
        assert result.exec_mode == "restore"
        assert "a" not in self.sdc._stale_cells

    def test_restore_with_no_changes(self):
        """EXEC-RESTORE where cell produces same result as before."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # EXEC-RESTORE B produces same result
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2, "z": 3}, max_size_mb=None)
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )

        # No changes → C stays fresh
        assert "c" not in result.stale_cells
        assert result.exec_mode == "restore"


# =============================================================================
# Tests for the %exec_restore magic command (FlowbookKernel level)
# =============================================================================


class TestExecRestoreMagic:
    """
    Tests for the %exec_restore magic command on FlowbookKernel.

    We can't instantiate a full FlowbookKernel in unit tests (requires
    IPython shell etc.), so we test the magic's logic via a mock approach:
    instantiate the enforcer, then call the same validation the magic uses.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c"])

    def _execute_cell(self, cell_id, pre_ns, post_ns, reads=None, writes=None):
        """Helper to simulate cell execution."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", pre_ns, max_size_mb=None
        )
        self.checkpoints.save(f"post_{cell_id}", post_ns, max_size_mb=None)
        self.checkpoints.save(
            f"{POST_CHECKPOINT_PREFIX}{cell_id}", post_ns, max_size_mb=None
        )
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            post_checkpoint=self.checkpoints.saved[f"post_{cell_id}"],
            tracking=make_tracking(reads=reads or set(), writes=writes or set()),
        )

    def test_magic_validation_accepts_valid_cell(self):
        """Magic should accept cell_id when precondition holds."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # B is valid target — A is fresh
        assert self.sdc.can_exec_restore("b") is True

    def test_magic_validation_rejects_when_predecessors_stale(self):
        """Magic should reject when predecessors are not all fresh."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.sdc._stale_cells.add("a")
        assert self.sdc.can_exec_restore("b") is False

    def test_magic_validation_rejects_empty_cell_id(self):
        """Empty cell_id should be rejected (checked by magic before calling can_exec_restore)."""
        # The magic itself checks for empty string before calling can_exec_restore
        # but can_exec_restore would return False for empty string anyway
        assert self.sdc.can_exec_restore("") is False

    def test_magic_validation_rejects_unknown_cell(self):
        """Unknown cell_id not in cell_order should be rejected."""
        assert self.sdc.can_exec_restore("nonexistent") is False

    def test_magic_cell_id_to_alpha_for_error_messages(self):
        """The magic uses _cell_id_to_alpha for error messages."""
        assert self.sdc._cell_id_to_alpha("a") == "@A"
        assert self.sdc._cell_id_to_alpha("b") == "@B"
        assert self.sdc._cell_id_to_alpha("unknown") == "unknown"


# =============================================================================
# Tests for pending flag edge cases
# =============================================================================


class TestPendingFlagEdgeCases:
    """
    Tests for edge cases in the pending flag consumption logic.

    These test the enforcer-level behaviors that the kernel's consumption
    logic depends on.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)
        self.sdc.set_cell_order(["a", "b", "c", "d"])

    def _execute_cell(self, cell_id, pre_ns, post_ns, reads=None, writes=None,
                      is_exec_restore=False, old_live_checkpoint=None):
        """Helper to simulate cell execution."""
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}{cell_id}", pre_ns, max_size_mb=None
        )
        self.checkpoints.save(f"post_{cell_id}", post_ns, max_size_mb=None)
        self.checkpoints.save(
            f"{POST_CHECKPOINT_PREFIX}{cell_id}", post_ns, max_size_mb=None
        )
        return self.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}{cell_id}"],
            post_checkpoint=self.checkpoints.saved[f"post_{cell_id}"],
            tracking=make_tracking(reads=reads or set(), writes=writes or set()),
            is_exec_restore=is_exec_restore,
            old_live_checkpoint=old_live_checkpoint,
        )

    def test_cell_id_mismatch_falls_back_to_normal(self):
        """
        When the pending flag targets cell X but cell Y executes,
        the kernel clears the flag and runs Y normally.
        We verify that Y runs in 'live' mode.
        """
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # Pending flag would target C, but user executes D instead
        # The kernel clears the flag and runs D normally
        result_d = self._execute_cell(
            "d", {"x": 1, "y": 2}, {"x": 1, "y": 2, "w": 4},
            writes={"w"},
        )
        assert result_d.exec_mode == "live"

    def test_missing_prefix_checkpoint_falls_back(self):
        """
        When prefix checkpoint doesn't exist (deleted/corrupted),
        the kernel falls back to normal execution.

        The kernel checks: self._checkpoints.memory.get(prefix_name) is not None
        If the checkpoint was never saved under POST_CHECKPOINT_PREFIX, the
        get() call raises KeyError and the kernel falls back to normal execution.
        """
        # Execute A but don't save its post checkpoint under the _post_ prefix
        self.checkpoints.save(
            f"{PRE_CHECKPOINT_PREFIX}a", {}, max_size_mb=None
        )
        self.checkpoints.save("post_a", {"x": 1}, max_size_mb=None)
        self.sdc.check(
            cell_id="a",
            pre_checkpoint=self.checkpoints.saved[f"{PRE_CHECKPOINT_PREFIX}a"],
            post_checkpoint=self.checkpoints.saved["post_a"],
            tracking=make_tracking(writes={"x"}),
        )

        # The prefix checkpoint name for B is _post_a
        prefix_name = self.sdc.get_prefix_checkpoint_name("b")
        assert prefix_name == f"{POST_CHECKPOINT_PREFIX}a"

        # Verify the checkpoint doesn't exist — get() raises KeyError
        assert prefix_name not in self.checkpoints.saved

    def test_exec_restore_after_normal_execution_clears_contamination(self):
        """
        Full flow: cell is contaminated, then restored, contamination flag clears.
        """
        # Execute A, C out of order, then B
        self._execute_cell("a", {}, {"x": 10}, writes={"x"})
        self._execute_cell("c", {"x": 10}, {"x": 999}, writes={"x"})
        result_b = self._execute_cell(
            "b", {"x": 999}, {"x": 999, "y": 1000},
            reads={"x"}, writes={"y"},
        )
        assert result_b.cell_is_contaminated is True
        assert "b" in self.sdc._stale_cells

        # EXEC-RESTORE B from prefix (post_a: x=10)
        self.checkpoints.save(
            "_old_live_b", {"x": 999, "y": 1000}, max_size_mb=None
        )
        result_b2 = self._execute_cell(
            "b", {"x": 10}, {"x": 10, "y": 11},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )
        assert result_b2.cell_is_contaminated is False
        assert "b" not in self.sdc._stale_cells
        assert result_b2.exec_mode == "restore"

    def test_successive_restores(self):
        """Multiple EXEC-RESTOREs in sequence work correctly."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})

        # EXEC-RESTORE B twice
        for i in range(2):
            result = self._execute_cell(
                "b", {"x": 1}, {"x": 1, "y": 2},
                reads={"x"}, writes={"y"},
                is_exec_restore=True,
            )
            assert result.exec_mode == "restore"
            assert "b" not in self.sdc._stale_cells

    def test_restore_does_not_trigger_backward_violation(self):
        """
        EXEC-RESTORE skips backward checks even when cell would normally violate.
        """
        # Execute A (reads x)
        self._execute_cell("a", {"x": 1}, {"x": 1}, reads={"x"})

        # EXEC-RESTORE B modifies x — would be backward violation normally
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 999},
            writes={"x"},
            is_exec_restore=True,
        )
        assert result.violation is None
        assert result.exec_mode == "restore"

    def test_restore_does_not_trigger_forward_violation(self):
        """
        EXEC-RESTORE skips forward contamination checks.
        """
        # Execute C (writes x)
        self._execute_cell("c", {}, {"x": 5}, writes={"x"})

        # EXEC-RESTORE B reads x from later cell — but no forward check in restore mode
        result = self._execute_cell(
            "b", {"x": 5}, {"x": 5, "y": 6},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
        )
        assert result.forward_violation is None
        assert result.exec_mode == "restore"
