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

    def test_mentions_context_menu_action(self):
        """Message should mention the context menu action."""
        message = format_forward_dependency_message("@B", "@C", ["x"])
        assert "Run with Previous Cell's state" in message

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

    def test_stale_immediate_predecessor_blocks(self):
        """Invalid when the immediate predecessor is stale."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self.sdc._stale_cells.add("b")  # Immediate predecessor of C
        assert self.sdc.can_exec_restore("c") is False

    def test_stale_non_immediate_predecessor_allowed(self):
        """Valid when a non-immediate predecessor is stale but immediate is fresh."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self.sdc._stale_cells.add("a")  # Non-immediate predecessor of C
        assert self.sdc.can_exec_restore("c") is True

    def test_unexecuted_immediate_predecessor_blocks(self):
        """Invalid when the immediate predecessor hasn't been executed."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # Skip B — immediate predecessor of C
        assert self.sdc.can_exec_restore("c") is False

    def test_unexecuted_non_immediate_predecessor_allowed(self):
        """Valid when a non-immediate predecessor is unexecuted but immediate is fresh."""
        # Skip A, execute B — C's immediate predecessor (B) has a fresh record
        self._execute_cell("b", {}, {"y": 2}, writes={"y"})
        assert self.sdc.can_exec_restore("c") is True

    def test_contaminated_immediate_predecessor_blocks(self):
        """Invalid when the immediate predecessor is forward-contaminated (stale)."""
        # Execute C first (writes x), then B (reads x) → B is contaminated (stale)
        self._execute_cell("c", {}, {"x": 5}, writes={"x"})
        result_b = self._execute_cell("b", {"x": 5}, {"x": 5, "y": 2}, reads={"x"}, writes={"y"})
        assert result_b.cell_is_contaminated is True
        assert "b" in self.sdc._stale_cells

        # C cannot exec-restore because B (immediate predecessor) is stale
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
        before activating the restore path. If the immediate predecessor becomes
        stale between %exec_restore and execution, it falls back to normal execution.
        """
        # Execute A and B fresh
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # Precondition initially valid
        assert self.sdc.can_exec_restore("c") is True

        # But then B (immediate predecessor of C) becomes stale (e.g., user edited it)
        self.sdc._stale_cells.add("b")

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

    def test_restore_with_no_changes_no_writer_conflict(self):
        """
        EXEC-RESTORE where cell produces same result AND no later cell
        writes to the restored cell's reads → no stale cells.
        """
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # EXEC-RESTORE B produces same result, C doesn't write x (B's read)
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2, "z": 3}, max_size_mb=None)
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )

        # C is not stale — no data changes, no writer-conflict
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

    def test_missing_prefix_checkpoint_is_detected(self):
        """
        When prefix checkpoint doesn't exist (predecessor not run),
        the kernel should refuse EXEC-RESTORE.

        The kernel checks: prefix_name in self._checkpoints.memory.saved
        If the checkpoint was never saved under POST_CHECKPOINT_PREFIX,
        the kernel returns an error instead of executing.
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

        # Verify the checkpoint doesn't exist
        assert prefix_name not in self.checkpoints.saved

        # The kernel would detect this and refuse with an error
        # (tested at the kernel level via TestExecRestorePredecessorRequired)

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


# =============================================================================
# Tests for predecessor-required and initial-state EXEC-RESTORE behavior
# =============================================================================


class TestExecRestorePredecessorRequired:
    """
    Tests for the two new EXEC-RESTORE behaviors:

    1. First cell: restore to initial empty state (σ_0)
    2. Non-first cell with predecessor not run: refuse with error

    These test the enforcer/checkpoint-level preconditions that the kernel's
    _do_execute_impl consumption logic depends on.
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

    # --- First cell: initial state ---

    def test_first_cell_prefix_is_none(self):
        """First cell has no predecessor — prefix checkpoint name is None."""
        assert self.sdc.get_prefix_checkpoint_name("a") is None

    def test_first_cell_exec_restore_uses_initial_state(self):
        """
        EXEC-RESTORE on first cell should restore to initial state (σ_0).

        The kernel saves an '_initial_state' checkpoint at tracking init.
        When prefix_name is None, the kernel restores '_initial_state'.
        """
        # Simulate: save an initial state checkpoint (kernel does this at startup)
        initial_ns = {}
        self.checkpoints.save("_initial_state", initial_ns, max_size_mb=None)
        assert "_initial_state" in self.checkpoints.saved

        # Execute A normally first
        self._execute_cell("a", {}, {"x": 42}, writes={"x"})

        # Now EXEC-RESTORE A should work: prefix is None (first cell),
        # so kernel restores _initial_state (empty namespace).
        # Simulated: A executes from empty state, produces x=42.
        self.checkpoints.save("_old_live_a", {"x": 42}, max_size_mb=None)
        result = self._execute_cell(
            "a", {}, {"x": 42},
            writes={"x"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_a"],
        )
        assert result.exec_mode == "restore"
        assert result.violation is None
        assert "a" not in self.sdc._stale_cells

    def test_first_cell_restore_clears_namespace(self):
        """
        After restoring to initial state, the cell should see an empty
        namespace (no user variables from prior executions).

        Verified by: if we simulate execution from {} (initial state),
        the enforcer sees no unexpected reads.
        """
        self.checkpoints.save("_initial_state", {}, max_size_mb=None)

        # A previously executed with polluted namespace
        self._execute_cell("a", {"junk": 999}, {"junk": 999, "x": 1}, writes={"x"})

        # EXEC-RESTORE A from initial state: pre_ns should be {} (clean)
        self.checkpoints.save(
            "_old_live_a", {"junk": 999, "x": 1}, max_size_mb=None
        )
        result = self._execute_cell(
            "a", {}, {"x": 1},
            writes={"x"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_a"],
        )
        assert result.exec_mode == "restore"

    def test_first_cell_can_exec_restore_precondition(self):
        """can_exec_restore for first cell should pass (no predecessors to check)."""
        assert self.sdc.can_exec_restore("a") is True

    # --- Non-first cell: predecessor not run ---

    def test_predecessor_not_run_detected_by_missing_checkpoint(self):
        """
        When predecessor hasn't been executed, its POST checkpoint doesn't
        exist, which the kernel detects.
        """
        # B's prefix is post_a. A was never executed → no _post_a checkpoint.
        prefix_name = self.sdc.get_prefix_checkpoint_name("b")
        assert prefix_name == f"{POST_CHECKPOINT_PREFIX}a"
        assert prefix_name not in self.checkpoints.saved

    def test_predecessor_run_creates_checkpoint(self):
        """After executing predecessor, its POST checkpoint exists."""
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})

        prefix_name = self.sdc.get_prefix_checkpoint_name("b")
        assert prefix_name in self.checkpoints.saved

    def test_skip_predecessor_detected(self):
        """
        Skipping a cell should be detectable: if A runs but B doesn't,
        C's prefix (post_b) is missing.
        """
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # B is skipped — never executed

        # C's prefix is post_b
        prefix_name = self.sdc.get_prefix_checkpoint_name("c")
        assert prefix_name == f"{POST_CHECKPOINT_PREFIX}b"
        assert prefix_name not in self.checkpoints.saved

    def test_predecessor_run_allows_restore(self):
        """
        When predecessor HAS been executed, EXEC-RESTORE proceeds normally.
        """
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})

        # B's prefix (post_a) exists — EXEC-RESTORE B should work
        prefix_name = self.sdc.get_prefix_checkpoint_name("b")
        assert prefix_name in self.checkpoints.saved

        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
        )
        assert result.exec_mode == "restore"
        assert result.violation is None

    def test_deep_predecessor_chain(self):
        """
        Each cell's restore depends only on its immediate predecessor,
        not the entire chain.
        """
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})

        # C's prefix is post_b — B was executed so checkpoint exists
        prefix_name = self.sdc.get_prefix_checkpoint_name("c")
        assert prefix_name == f"{POST_CHECKPOINT_PREFIX}b"
        assert prefix_name in self.checkpoints.saved

        # D's prefix is post_c — C was NOT executed → missing
        prefix_name_d = self.sdc.get_prefix_checkpoint_name("d")
        assert prefix_name_d == f"{POST_CHECKPOINT_PREFIX}c"
        assert prefix_name_d not in self.checkpoints.saved

    def test_predecessor_rerun_updates_checkpoint(self):
        """
        Re-running a predecessor updates its POST checkpoint,
        which affects what EXEC-RESTORE restores for the next cell.
        """
        # A produces x=1
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})

        # A re-executed with x=99
        self._execute_cell("a", {}, {"x": 99}, writes={"x"})

        # B's prefix (post_a) now has x=99
        prefix_name = self.sdc.get_prefix_checkpoint_name("b")
        restored = self.checkpoints.saved[prefix_name]
        assert restored.user_ns["x"] == 99


# =============================================================================
# Tests for EXEC-RESTORE marking all later cells stale
# =============================================================================


class TestExecRestoreMarksLaterCellsStale:
    """
    After EXEC-RESTORE of cell i, later cells j > i that WRITE to any
    variable in cell i's reads-before-writes set are marked stale.

    StaleFwd (the delta-based check) marks later cells that READ changed
    variables.  This additional check catches cells that WRITE to variables
    the restored cell reads — re-running such a cell would cause BackConflict
    (it mutates a location a fresh earlier cell depends on).

    Together, StaleFwd + this writer-check cover all problematic later cells.
    Independent cells (no read/write overlap) stay fresh.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)

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

    # --- The motivating scenario: D writes x, not caught by StaleFwd ---

    def test_writer_cell_marked_stale_after_restore(self):
        """
        Scenario: A: x=23, B: y=x+3, C: z=y, D: y=0
        Execution: A→B→D→C. C is contaminated (reads y from D which overwrote
        B's y — D doesn't read y, only writes it).
        Restore C → D should be stale because D's execution context is invalid.
        D only WRITES y (doesn't read it), so StaleFwd alone wouldn't catch it.
        Without the all-later-cells-stale rule, D would appear fresh after
        restore, but re-running D would cause BackConflict (D writes y, C fresh
        reads y).
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])

        self._execute_cell("a", {}, {"x": 23}, writes={"x"})
        self._execute_cell("b", {"x": 23}, {"x": 23, "y": 26},
                          reads={"x"}, writes={"y"})
        # D writes y=0 (no backward conflict: no earlier cell reads y)
        self._execute_cell("d", {"x": 23, "y": 26}, {"x": 23, "y": 0},
                          writes={"y"})
        # C reads y=0 (from D), but D is later in doc order → contaminated
        result_c = self._execute_cell(
            "c", {"x": 23, "y": 0}, {"x": 23, "y": 0, "z": 0},
            reads={"y"}, writes={"z"}
        )
        assert result_c.cell_is_contaminated is True

        # Restore C from prefix (B's post: x=23, y=26)
        self.checkpoints.save("_old_live_c", {"x": 23, "y": 0, "z": 0},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 23, "y": 26}, {"x": 23, "y": 26, "z": 26},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        assert result.exec_mode == "restore"
        assert "c" not in self.sdc._stale_cells
        # D is marked stale even though it doesn't READ any changed variable
        assert "d" in self.sdc._stale_cells
        assert "d" in result.stale_cells

    # --- Basic: restore marks writer-conflict cells stale ---

    def test_later_cell_writing_to_restored_read_marked_stale(self):
        """
        Execution: A→B→D→C (out of order).
        D writes y. No backward conflict at D's time (no earlier fresh cell reads y).
        C reads y → contaminated. Restore C (reads y) → D (writes y) stale.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 23}, writes={"x"})
        self._execute_cell("b", {"x": 23}, {"x": 23, "y": 26},
                          reads={"x"}, writes={"y"})
        # D out of order: writes y=0 (no backward conflict: B writes y, not reads it)
        self._execute_cell("d", {"x": 23, "y": 26}, {"x": 23, "y": 0},
                          writes={"y"})
        # C reads y=0 → contaminated
        result_c = self._execute_cell(
            "c", {"x": 23, "y": 0}, {"x": 23, "y": 0, "z": 0},
            reads={"y"}, writes={"z"}
        )
        assert result_c.cell_is_contaminated is True

        # Restore C (reads y)
        self.checkpoints.save("_old_live_c", {"x": 23, "y": 0, "z": 0},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 23, "y": 26}, {"x": 23, "y": 26, "z": 26},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # D writes y (which C reads) → stale via writer-check
        assert "d" in result.stale_cells
        assert "a" not in self.sdc._stale_cells
        assert "b" not in self.sdc._stale_cells

    def test_later_cell_not_writing_to_restored_read_stays_fresh(self):
        """After restoring C (reads y), D (writes w, not y) stays fresh."""
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # C reads y, writes z
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        # D writes w — no overlap with C's reads (y)
        self._execute_cell("d", {"x": 1, "y": 2, "z": 3},
                          {"x": 1, "y": 2, "z": 3, "w": 99},
                          writes={"w"})

        # Restore C (reads y)
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 2, "z": 3, "w": 99},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # D doesn't write y → not stale via writer-check
        assert "d" not in self.sdc._stale_cells

    def test_restore_cell_with_no_reads_no_writer_conflicts(self):
        """Restoring A (reads nothing) → no writer-conflict stale cells."""
        self.sdc.set_cell_order(["a", "b", "c"])
        # A reads nothing, writes x
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # Restore A (reads nothing) → no writer-conflicts possible
        result = self._execute_cell(
            "a", {}, {"x": 1},
            writes={"x"},
            is_exec_restore=True,
        )

        # No writer-conflict (A reads nothing), no StaleFwd (no old_live)
        assert "b" not in self.sdc._stale_cells
        assert "c" not in self.sdc._stale_cells

    def test_both_stalefwd_and_writer_check(self):
        """
        StaleFwd and writer-check complement each other.
        Execution: A→B→D→C (out of order).
        D writes y. C reads y → contaminated.
        Restore C with DIFFERENT result (z changes) → D stale (writer-check),
        plus StaleFwd could mark other cells if delta variables match.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # D out of order: writes y=0
        self._execute_cell("d", {"x": 1, "y": 2}, {"x": 1, "y": 0},
                          writes={"y"})
        # C reads y=0 → contaminated
        self._execute_cell("c", {"x": 1, "y": 0}, {"x": 1, "y": 0, "z": 0},
                          reads={"y"}, writes={"z"})

        # Restore C from prefix (B's post: y=2). z changes 0→2 in delta.
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 0, "z": 0},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 2},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # D writes y (C reads y) → stale via writer-check
        assert "d" in result.stale_cells

    def test_restore_last_cell_marks_nothing_stale(self):
        """Restoring the last cell has no later cells to mark stale."""
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # Restore C (last cell)
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 2, "z": 3},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        assert "a" not in self.sdc._stale_cells
        assert "b" not in self.sdc._stale_cells
        assert "c" not in self.sdc._stale_cells

    # --- Only cells with writer-conflict are marked ---

    def test_unexecuted_later_cells_not_marked(self):
        """Later cells that have never been executed are not marked stale."""
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # C and D never executed

        # Restore B
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2},
                            max_size_mb=None)
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )

        # C and D have no records — not in stale set
        assert "c" not in self.sdc._stale_cells
        assert "d" not in self.sdc._stale_cells

    def test_only_writer_conflict_cells_marked(self):
        """Only later cells that write to C's reads are marked stale."""
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # Execute D and E out of order before C to avoid backward violations
        # D writes y — overlaps with C's reads (y)
        self._execute_cell("d", {"x": 1, "y": 2}, {"x": 1, "y": 0},
                          writes={"y"})
        # E writes w — no overlap with C's reads (y)
        self._execute_cell("e", {"x": 1, "y": 0}, {"x": 1, "y": 0, "w": 99},
                          writes={"w"})
        # C reads y=0 → contaminated (D is later in doc order, wrote y)
        self._execute_cell("c", {"x": 1, "y": 0}, {"x": 1, "y": 0, "z": 0},
                          reads={"y"}, writes={"z"})

        # Restore C (reads y)
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 0, "z": 0, "w": 99},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 2},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # D writes y (C's read) → stale
        assert "d" in self.sdc._stale_cells
        # E writes w (not y) → not stale from writer-check
        assert "e" not in self.sdc._stale_cells

    # --- Earlier cells are NOT marked stale ---

    def test_earlier_cells_not_affected(self):
        """Cells before the restored cell are never marked stale by the restore."""
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # Execute D out of order (before C) to avoid backward violation
        # D writes y — overlaps with C's reads
        self._execute_cell("d", {"x": 1, "y": 2}, {"x": 1, "y": 99},
                          writes={"y"})
        # C reads y=99 → contaminated
        self._execute_cell("c", {"x": 1, "y": 99}, {"x": 1, "y": 99, "z": 99},
                          reads={"y"}, writes={"z"})

        # Restore C (reads y)
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 99, "z": 99},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 2},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # A and B unaffected
        assert "a" not in self.sdc._stale_cells
        assert "b" not in self.sdc._stale_cells
        # D writes y (which C reads) → stale via writer-check
        assert "d" in self.sdc._stale_cells

    # --- The restored cell itself is fresh ---

    def test_restored_cell_itself_is_fresh(self):
        """The restored cell is always fresh, even when later cells are marked stale."""
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # Execute C out of order (before B) to avoid backward violation
        # C writes y — overlaps with B's reads for the writer-check
        self._execute_cell("c", {"x": 1}, {"x": 1, "y": 99},
                          writes={"y"})
        # B reads y=99 → contaminated (C is later in doc order)
        self._execute_cell("b", {"x": 1, "y": 99}, {"x": 1, "y": 99, "z": 5},
                          reads={"y"}, writes={"z"})

        # B is contaminated (stale)
        assert "b" in self.sdc._stale_cells

        # Restore B (reads y)
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 99, "z": 5},
                            max_size_mb=None)
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "z": 5},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )

        assert "b" not in self.sdc._stale_cells
        assert result.cell_is_contaminated is False
        assert result.exec_mode == "restore"
        # C writes y (which B reads) → stale via writer-check
        assert "c" in self.sdc._stale_cells

    # --- Cells already stale stay stale ---

    def test_already_stale_later_cell_remains_stale(self):
        """A later cell that was already stale stays stale after restore."""
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # C already stale (e.g. user edited it)
        self.sdc._stale_cells.add("c")
        assert "c" in self.sdc._stale_cells

        # Restore B
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2, "z": 3},
                            max_size_mb=None)
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )

        # C still stale
        assert "c" in self.sdc._stale_cells
        assert "c" in result.stale_cells

    # --- Independent cell stays fresh ---

    def test_independent_later_cell_stays_fresh(self):
        """
        A cell that has no read/write overlap with the restored cell stays fresh.
        The targeted approach is precise: only writer-conflict cells are marked.
        """
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # C writes w — no overlap with B's reads (x)
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "w": 42},
                          writes={"w"})

        # Restore B — C is NOT stale (no writer-conflict, no data change)
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2, "w": 42},
                            max_size_mb=None)
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )

        assert "c" not in result.stale_cells

    # --- Stale list is in document order ---

    def test_stale_list_in_document_order(self):
        """The stale_cells list should be in document order."""
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # Execute D and E out of order (before C) to avoid backward violations
        # D writes y — writer-conflict with C's reads
        self._execute_cell("d", {"x": 1, "y": 2}, {"x": 1, "y": 10},
                          writes={"y"})
        # E also writes y
        self._execute_cell("e", {"x": 1, "y": 10}, {"x": 1, "y": 20},
                          writes={"y"})
        # C reads y=20 → contaminated (D, E later in doc order wrote y)
        self._execute_cell("c", {"x": 1, "y": 20}, {"x": 1, "y": 20, "z": 20},
                          reads={"y"}, writes={"z"})

        # Restore C (reads y)
        self.checkpoints.save("_old_live_c",
                            {"x": 1, "y": 20, "z": 20},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 2},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # D and E write y (C's read) → stale in document order
        assert "d" in result.stale_cells
        assert "e" in result.stale_cells
        # Verify document order
        stale_idx = {cid: i for i, cid in enumerate(result.stale_cells)}
        assert stale_idx["d"] < stale_idx["e"]

    # --- Restore without old_live_checkpoint ---

    def test_restore_without_old_live_writer_check_still_works(self):
        """
        When no old_live_checkpoint is provided, the writer-check still
        marks later cells that write to the restored cell's reads.

        Execution: A→C→B (out of order). C writes y (no backward conflict
        since B not yet executed). B reads y → contaminated.
        Restore B (reads y) without old_live → C (writes y) stale.
        """
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # C out of order: writes y=99 (no backward conflict — B not executed)
        self._execute_cell("c", {"x": 1}, {"x": 1, "y": 99},
                          writes={"y"})
        # B reads y=99 → contaminated (C later in doc order wrote y)
        result_b = self._execute_cell(
            "b", {"x": 1, "y": 99}, {"x": 1, "y": 99, "z": 5},
            reads={"y"}, writes={"z"}
        )
        assert result_b.cell_is_contaminated is True

        # Restore B (reads y) without old_live_checkpoint
        result = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "z": 5},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
        )

        # C writes y (B's read) → stale via writer-check
        assert "c" in result.stale_cells
        # A unaffected
        assert "a" not in self.sdc._stale_cells

    # --- Successive restores ---

    def test_restore_b_then_restore_c_stale_propagation(self):
        """
        Successive restores each mark writer-conflict cells stale.

        Execution: A→D→B→C (out of order). D writes x and y.
        B reads x → contaminated. C reads y → contaminated.
        Restore B (reads x) → D stale (writes x).
        Then restore C (reads y) → D still stale (writes y).
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # D out of order: writes x=99, y=99 (no backward conflict — B,C not executed)
        self._execute_cell("d", {"x": 1}, {"x": 99, "y": 99},
                          writes={"x", "y"})
        # B reads x=99 → contaminated (D later in doc order wrote x)
        result_b = self._execute_cell(
            "b", {"x": 99, "y": 99}, {"x": 99, "y": 99, "z": 5},
            reads={"x"}, writes={"z"})
        assert result_b.cell_is_contaminated is True
        # C reads y=99 → contaminated (D later in doc order wrote y)
        result_c = self._execute_cell(
            "c", {"x": 99, "y": 99, "z": 5},
            {"x": 99, "y": 99, "z": 5, "w": 7},
            reads={"y"}, writes={"w"})
        assert result_c.cell_is_contaminated is True

        # Step 1: Restore B (reads x) → D (writes x) stale
        self.checkpoints.save("_old_live_b",
                            {"x": 99, "y": 99, "z": 5, "w": 7},
                            max_size_mb=None)
        result1 = self._execute_cell(
            "b", {"x": 1}, {"x": 1, "z": 5},
            reads={"x"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )
        assert "d" in result1.stale_cells

        # Step 2: Restore C (reads y) → D (writes y) still stale
        # C's prefix is B's post; after restoring B, B is fresh
        self.checkpoints.save("_old_live_c",
                            {"x": 1, "z": 5, "w": 7},
                            max_size_mb=None)
        result2 = self._execute_cell(
            "c", {"x": 1, "z": 5}, {"x": 1, "z": 5, "w": 7},
            reads={"y"}, writes={"w"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )
        assert "d" in result2.stale_cells

    # --- Backward conflict scenario the user reported ---

    def test_backward_conflict_cell_marked_stale(self):
        """
        Scenario: A: x=23, B: y=x+3, C: z=y, D: y=0
        Execution: A→B→D→C. D writes y (no backward conflict since no earlier
        cell reads y). C reads y=0 from D → contaminated.
        After restoring C, re-running D would cause BackConflict
        (D writes y, C reads y and is now fresh). D must be stale.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 23}, writes={"x"})
        self._execute_cell("b", {"x": 23}, {"x": 23, "y": 26},
                          reads={"x"}, writes={"y"})
        # D writes y=0 (no backward conflict: no earlier cell reads y)
        self._execute_cell("d", {"x": 23, "y": 26}, {"x": 23, "y": 0},
                          writes={"y"})
        # C contaminated (reads y=0 from D, but D is later in doc order)
        result_c = self._execute_cell(
            "c", {"x": 23, "y": 0}, {"x": 23, "y": 0, "z": 0},
            reads={"y"}, writes={"z"}
        )
        assert result_c.cell_is_contaminated is True

        # Before restore: D is fresh (no stale mark)
        assert "d" not in self.sdc._stale_cells

        # Restore C from prefix (B's post: y=26)
        self.checkpoints.save("_old_live_c", {"x": 23, "y": 0, "z": 0},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 23, "y": 26}, {"x": 23, "y": 26, "z": 26},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # C fresh, D stale — user sees D highlighted, knows to re-run
        assert "c" not in self.sdc._stale_cells
        assert "d" in self.sdc._stale_cells
        assert "d" in result.stale_cells
        # A and B unaffected
        assert "a" not in self.sdc._stale_cells
        assert "b" not in self.sdc._stale_cells

    # --- StaleBack: mark earlier fresh cells stale if delta affects their reads ---

    def test_staleback_earlier_cell_reading_changed_var(self):
        """
        EXEC-RESTORE of C writes x (from prefix). A reads x. A is marked stale
        because Δ(Σ, Σ') includes x and Obs_A ∩ Δ ≠ ∅.

        This is the analogue of BackConflict for the restore path: in
        EXEC-ACCEPT, BackConflict would reject C. In EXEC-RESTORE, StaleBack
        marks A stale instead.
        """
        self.sdc.set_cell_order(["a", "b", "c"])
        # A reads x=0 (initial value), writes y
        self._execute_cell("a", {"x": 0}, {"x": 0, "y": 1},
                          reads={"x"}, writes={"y"})
        # B: no-op
        self._execute_cell("b", {"x": 0, "y": 1}, {"x": 0, "y": 1})
        # EXEC-RESTORE C from σ^post_B = {x: 0, y: 1}. C writes x=99.
        self.checkpoints.save("_old_live_c", {"x": 0, "y": 1},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 0, "y": 1}, {"x": 99, "y": 1},
            writes={"x"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # A reads x, x changed → A stale via StaleBack
        assert "a" in self.sdc._stale_cells
        assert "a" in result.stale_cells
        # B doesn't read anything → unaffected
        assert "b" not in self.sdc._stale_cells

    def test_staleback_earlier_cell_not_reading_changed_var(self):
        """Earlier cell that doesn't read any changed variable stays fresh."""
        self.sdc.set_cell_order(["a", "b", "c"])
        # A reads y (not x), writes w
        self._execute_cell("a", {"y": 5}, {"y": 5, "w": 10},
                          reads={"y"}, writes={"w"})
        self._execute_cell("b", {"y": 5, "w": 10}, {"y": 5, "w": 10})
        # EXEC-RESTORE C writes x=99 (A doesn't read x)
        self.checkpoints.save("_old_live_c", {"y": 5, "w": 10},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"y": 5, "w": 10}, {"y": 5, "w": 10, "x": 99},
            writes={"x"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # A reads y, not x → A stays fresh
        assert "a" not in self.sdc._stale_cells

    def test_staleback_and_stalefwd_together(self):
        """
        StaleBack marks earlier cells stale, StaleFwd marks later cells stale.
        Both operate on the same Δ(Σ, Σ').
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        # A reads x
        self._execute_cell("a", {"x": 0}, {"x": 0, "y": 1},
                          reads={"x"}, writes={"y"})
        self._execute_cell("b", {"x": 0, "y": 1}, {"x": 0, "y": 1, "z": 2},
                          writes={"z"})
        # D reads x — will be caught by StaleFwd
        self._execute_cell("d", {"x": 0, "y": 1, "z": 2},
                          {"x": 0, "y": 1, "z": 2, "w": 3},
                          reads={"x"}, writes={"w"})

        # EXEC-RESTORE C: writes x=99. Δ includes x.
        self.checkpoints.save("_old_live_c", {"x": 0, "y": 1, "z": 2, "w": 3},
                            max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 0, "y": 1, "z": 2}, {"x": 99, "y": 1, "z": 2},
            writes={"x"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )

        # A reads x → stale via StaleBack (k < i)
        assert "a" in result.stale_cells
        # D reads x → stale via StaleFwd (k > i)
        assert "d" in result.stale_cells
        # B doesn't read x → fresh
        assert "b" not in self.sdc._stale_cells


class TestImmediatePredecessorPrecondition:
    """
    Tests for the relaxed can_exec_restore precondition.

    The precondition was changed from "ALL prior cells fresh" to
    "immediate predecessor fresh". This test class exhaustively covers
    the new semantics across various notebook topologies and states.
    """

    def setup_method(self):
        self.checkpoints = MemoryCheckpoints(
            sanity_check=False,
            warn_classes=False,
        )
        self.sdc = ReproducibilityEnforcer(self.checkpoints)

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

    # --- Single-cell notebook ---

    def test_single_cell_can_restore(self):
        """A one-cell notebook: the only cell can always restore (first cell)."""
        self.sdc.set_cell_order(["a"])
        assert self.sdc.can_exec_restore("a") is True

    # --- Two-cell notebook ---

    def test_two_cells_predecessor_fresh(self):
        """Two-cell notebook: B can restore when A is fresh."""
        self.sdc.set_cell_order(["a", "b"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        assert self.sdc.can_exec_restore("b") is True

    def test_two_cells_predecessor_stale(self):
        """Two-cell notebook: B cannot restore when A is stale."""
        self.sdc.set_cell_order(["a", "b"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.sdc._stale_cells.add("a")
        assert self.sdc.can_exec_restore("b") is False

    def test_two_cells_predecessor_unexecuted(self):
        """Two-cell notebook: B cannot restore when A is unexecuted."""
        self.sdc.set_cell_order(["a", "b"])
        assert self.sdc.can_exec_restore("b") is False

    # --- The original bug scenario: unexecuted non-immediate predecessor ---

    def test_original_bug_unexecuted_first_cell(self):
        """
        The bug that triggered this fix:
        Notebook: [imports, B, C, D]. User runs B, D, C (skipping imports).
        C's immediate predecessor is B (fresh), so C can exec-restore
        even though 'imports' was never executed (no enforcer record).
        """
        self.sdc.set_cell_order(["imports", "b", "c", "d"])
        # imports is never executed (cell_id=None in the kernel)
        self._execute_cell("b", {}, {"x": 23}, writes={"x"})
        self._execute_cell("d", {"x": 23}, {"x": 5}, writes={"x"})
        result_c = self._execute_cell(
            "c", {"x": 5}, {"x": 5, "y": 8}, reads={"x"}, writes={"y"}
        )
        assert result_c.cell_is_contaminated is True

        # C can exec-restore: immediate predecessor B is fresh
        assert self.sdc.can_exec_restore("c") is True

    def test_original_bug_first_cell_no_record(self):
        """
        Even with 5 cells, if the first has no record but the immediate
        predecessor of the target is fresh, restore is allowed.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])
        # Skip A entirely
        self._execute_cell("b", {}, {"x": 1}, writes={"x"})
        self._execute_cell("c", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("d", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # E's immediate predecessor is D (fresh) — allowed
        assert self.sdc.can_exec_restore("e") is True

    # --- Multiple unexecuted cells before ---

    def test_multiple_gaps_immediate_predecessor_fresh(self):
        """Only the last cell before us needs to be fresh, gaps before are OK."""
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])
        # Skip A, B, C. Only execute D.
        self._execute_cell("d", {}, {"w": 4}, writes={"w"})
        # E's immediate predecessor is D (fresh) — allowed
        assert self.sdc.can_exec_restore("e") is True

    def test_multiple_gaps_immediate_predecessor_missing(self):
        """Gaps are fine, but if the immediate predecessor itself is missing, block."""
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])
        # Execute A and C only (B and D are missing)
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("c", {"x": 1}, {"x": 1, "y": 2},
                          reads={"x"}, writes={"y"})
        # E's immediate predecessor is D — not executed
        assert self.sdc.can_exec_restore("e") is False
        # D's immediate predecessor is C — fresh, so D can restore
        assert self.sdc.can_exec_restore("d") is True

    # --- Staleness patterns ---

    def test_all_predecessors_stale_except_immediate(self):
        """If only the immediate predecessor is fresh, restore is allowed."""
        self.sdc.set_cell_order(["a", "b", "c", "d", "e"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        self._execute_cell("d", {"x": 1, "y": 2, "z": 3},
                          {"x": 1, "y": 2, "z": 3, "w": 4}, writes={"w"})

        # Mark A, B, C all stale but D fresh
        self.sdc._stale_cells.update({"a", "b", "c"})
        # E's immediate predecessor is D (fresh) — allowed
        assert self.sdc.can_exec_restore("e") is True

    def test_immediate_predecessor_stale_even_if_earlier_fresh(self):
        """If the immediate predecessor is stale, block even if all earlier are fresh."""
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})

        # Mark only C stale (immediate predecessor of D)
        self.sdc._stale_cells.add("c")
        assert self.sdc.can_exec_restore("d") is False

    def test_predecessor_freshened_after_being_stale(self):
        """If the predecessor was stale then re-executed (fresh again), allow."""
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # B becomes stale
        self.sdc._stale_cells.add("b")
        assert self.sdc.can_exec_restore("c") is False

        # B re-executed → fresh again (re-execution removes from stale set)
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        assert "b" not in self.sdc._stale_cells
        assert self.sdc.can_exec_restore("c") is True

    # --- Forward contamination interactions ---

    def test_contaminated_non_immediate_predecessor_allowed(self):
        """
        Forward-contaminated cell earlier in the order doesn't block
        restore for a later cell whose immediate predecessor is fresh.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        # D writes x, then A reads x → A is forward-contaminated
        self._execute_cell("d", {}, {"x": 5}, writes={"x"})
        result_a = self._execute_cell("a", {"x": 5}, {"x": 5}, reads={"x"})
        assert result_a.cell_is_contaminated is True
        assert "a" in self.sdc._stale_cells

        # Execute B and C fresh
        self._execute_cell("b", {"x": 5}, {"x": 5, "y": 6}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 5, "y": 6}, {"x": 5, "y": 6, "z": 7},
                          reads={"y"}, writes={"z"})

        # D's immediate predecessor is C (fresh) — allowed despite A being contaminated
        assert self.sdc.can_exec_restore("d") is True

    def test_contaminated_immediate_predecessor_blocks(self):
        """Forward-contaminated immediate predecessor blocks restore."""
        self.sdc.set_cell_order(["a", "b", "c"])
        # C writes x, then B reads x → B contaminated
        self._execute_cell("c", {}, {"x": 5}, writes={"x"})
        result_b = self._execute_cell("b", {"x": 5}, {"x": 5, "y": 6},
                                       reads={"x"}, writes={"y"})
        assert result_b.cell_is_contaminated is True
        # C's immediate predecessor is B (stale/contaminated) — blocked
        assert self.sdc.can_exec_restore("c") is False

    # --- Cell order changes ---

    def test_cell_order_change_shifts_predecessor(self):
        """
        Changing cell order changes who the immediate predecessor is.
        After reordering, the precondition uses the new predecessor.
        """
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # C's predecessor is B (fresh) — allowed
        assert self.sdc.can_exec_restore("c") is True

        # Reorder: C is now between A and B
        self.sdc.set_cell_order(["a", "c", "b"])
        # C's predecessor is now A (fresh) — still allowed
        assert self.sdc.can_exec_restore("c") is True
        # B's predecessor is now C — unexecuted, so blocked
        assert self.sdc.can_exec_restore("b") is False

    def test_cell_order_change_moves_first_to_middle(self):
        """A cell that was first (always valid) may become blocked after reorder."""
        self.sdc.set_cell_order(["a", "b", "c"])
        # First cell always valid
        assert self.sdc.can_exec_restore("a") is True

        # Reorder: A is no longer first
        self.sdc.set_cell_order(["b", "a", "c"])
        # A's predecessor is now B — unexecuted
        assert self.sdc.can_exec_restore("a") is False

    # --- Edge cases ---

    def test_empty_cell_order(self):
        """No cells in order → any cell_id returns False."""
        self.sdc.set_cell_order([])
        assert self.sdc.can_exec_restore("a") is False

    def test_cell_is_both_executed_and_stale(self):
        """A cell that is its own predecessor's blocker scenario — last cell check."""
        self.sdc.set_cell_order(["a", "b"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        # A is both executed and stale (edited after execution)
        self.sdc._stale_cells.add("a")
        # B's predecessor is A — executed but stale
        assert self.sdc.can_exec_restore("b") is False

    def test_restore_target_itself_stale(self):
        """
        The target cell being stale doesn't affect the precondition.
        can_exec_restore checks predecessors, not the target cell.
        """
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        # Mark the target (C) stale — doesn't matter for precondition
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        self.sdc._stale_cells.add("c")
        assert self.sdc.can_exec_restore("c") is True

    def test_restore_target_unexecuted(self):
        """
        The target cell doesn't need to have been executed before.
        can_exec_restore only checks the predecessor.
        """
        self.sdc.set_cell_order(["a", "b", "c"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        # C has never been executed — but its predecessor B is fresh
        assert self.sdc.can_exec_restore("c") is True

    def test_second_cell_first_cell_never_executed(self):
        """
        B is the second cell. A (predecessor) was never executed.
        B cannot restore because A has no record.
        """
        self.sdc.set_cell_order(["a", "b"])
        assert self.sdc.can_exec_restore("b") is False

    def test_long_notebook_last_cell(self):
        """Last cell in a long notebook with mixed states before it."""
        cells = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.sdc.set_cell_order(cells)

        # Execute only some cells: a, c, e, g (skipping b, d, f)
        self._execute_cell("a", {}, {"v": 1}, writes={"v"})
        self._execute_cell("c", {"v": 1}, {"v": 1, "w": 2}, writes={"w"})
        self._execute_cell("e", {"v": 1, "w": 2}, {"v": 1, "w": 2, "x": 3}, writes={"x"})
        self._execute_cell("g", {"v": 1, "w": 2, "x": 3},
                          {"v": 1, "w": 2, "x": 3, "y": 4}, writes={"y"})

        # H's predecessor is G (fresh) — allowed
        assert self.sdc.can_exec_restore("h") is True
        # B's predecessor is A (fresh) — allowed
        assert self.sdc.can_exec_restore("b") is True
        # D's predecessor is C (fresh) — allowed
        assert self.sdc.can_exec_restore("d") is True
        # F's predecessor is E (fresh) — allowed
        assert self.sdc.can_exec_restore("f") is True
        # C's predecessor is B (NOT executed) — blocked
        assert self.sdc.can_exec_restore("c") is False

    # --- Restore then re-check ---

    def test_restore_makes_cell_fresh_unblocks_successor(self):
        """
        After restoring cell C (making it fresh), D's predecessor is fresh
        so D can now restore too.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # C executed but contaminated (stale)
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        self.sdc._stale_cells.add("c")

        # D blocked because C is stale
        assert self.sdc.can_exec_restore("d") is False

        # EXEC-RESTORE C → C becomes fresh
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 2, "z": 3}, max_size_mb=None)
        result = self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )
        assert result.exec_mode == "restore"
        assert "c" not in self.sdc._stale_cells

        # D can now restore — C is fresh
        assert self.sdc.can_exec_restore("d") is True

    def test_chain_of_restores(self):
        """
        Restoring B (makes fresh) enables restoring C, which enables D.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        self._execute_cell("d", {"x": 1, "y": 2, "z": 3},
                          {"x": 1, "y": 2, "z": 3, "w": 4}, reads={"z"}, writes={"w"})

        # Mark B, C stale
        self.sdc._stale_cells.update({"b", "c"})

        # C blocked (B is stale), D blocked (C is stale)
        assert self.sdc.can_exec_restore("c") is False
        assert self.sdc.can_exec_restore("d") is False

        # Restore B
        self.checkpoints.save("_old_live_b", {"x": 1, "y": 2, "z": 3, "w": 4},
                            max_size_mb=None)
        self._execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2},
            reads={"x"}, writes={"y"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_b"],
        )
        assert "b" not in self.sdc._stale_cells

        # C can now restore (B is fresh), D still blocked (C is stale)
        assert self.sdc.can_exec_restore("c") is True
        assert self.sdc.can_exec_restore("d") is False

        # Restore C
        self.checkpoints.save("_old_live_c", {"x": 1, "y": 2, "z": 3, "w": 4},
                            max_size_mb=None)
        self._execute_cell(
            "c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
            reads={"y"}, writes={"z"},
            is_exec_restore=True,
            old_live_checkpoint=self.checkpoints.saved["_old_live_c"],
        )
        assert "c" not in self.sdc._stale_cells

        # D can now restore
        assert self.sdc.can_exec_restore("d") is True

    # --- Symmetry: every position checked ---

    def test_each_position_in_four_cell_notebook(self):
        """
        Systematically verify can_exec_restore for every position in a
        4-cell notebook where all cells have been executed and are fresh.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        self._execute_cell("d", {"x": 1, "y": 2, "z": 3},
                          {"x": 1, "y": 2, "z": 3, "w": 4}, reads={"z"}, writes={"w"})

        # All cells can exec-restore
        for cell in ["a", "b", "c", "d"]:
            assert self.sdc.can_exec_restore(cell) is True, f"Cell {cell} should be restorable"

    def test_each_position_only_that_predecessor_matters(self):
        """
        For each cell, verify that ONLY its immediate predecessor's state
        determines can_exec_restore, not any earlier cell.
        """
        self.sdc.set_cell_order(["a", "b", "c", "d"])
        self._execute_cell("a", {}, {"x": 1}, writes={"x"})
        self._execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self._execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3},
                          reads={"y"}, writes={"z"})
        self._execute_cell("d", {"x": 1, "y": 2, "z": 3},
                          {"x": 1, "y": 2, "z": 3, "w": 4}, reads={"z"}, writes={"w"})

        # Stale A blocks only B, not C or D
        self.sdc._stale_cells.add("a")
        assert self.sdc.can_exec_restore("b") is False
        assert self.sdc.can_exec_restore("c") is True
        assert self.sdc.can_exec_restore("d") is True

        # Also stale B blocks only C
        self.sdc._stale_cells.add("b")
        assert self.sdc.can_exec_restore("c") is False
        assert self.sdc.can_exec_restore("d") is True

        # Also stale C blocks D
        self.sdc._stale_cells.add("c")
        assert self.sdc.can_exec_restore("d") is False

        # A is always restorable (first cell)
        assert self.sdc.can_exec_restore("a") is True
