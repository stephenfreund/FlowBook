"""
Monotonicity enforcement for cell execution.

This module provides MonotonicityEnforcer, which ensures that cells do not
modify variables they read before writing (read-before-write or RBW variables).

Background:
    In notebook execution, monotonicity means that the output of a cell depends
    only on variables that existed before the cell ran, not on side effects from
    the cell itself. When a cell reads a variable and then modifies it, re-running
    that cell could produce different results.

    Monotonicity enforcement detects and prevents this pattern by:
    1. Saving a checkpoint of RBW variables before execution
    2. Executing the cell
    3. Comparing RBW variables to their pre-execution values
    4. If any differ, rolling back to the checkpoint and raising an error

Usage:
    enforcer = MonotonicityEnforcer(checkpoints, user_ns)
    enforcer.save_pre_state(cell_id)

    # ... execute cell ...
    # ... get tracking_data ...

    violation = enforcer.check_and_enforce(tracking_data, cell_id)
    if violation:
        return violation.to_error_result(execution_count)

Column-Level Tracking:
    For DataFrames, monotonicity is checked at the column level. If a cell reads
    columns A and B but only writes column C, only A and B are checked for
    modifications. This allows legitimate patterns like adding computed columns.
"""

from typing import TYPE_CHECKING, Optional

from flowbook.kernel_support.checkpoint import Checkpoint
from flowbook.kernel_support.models import MonotonicityViolation, TrackingData
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode
from flowbook.kernel_support.types import ValueComparison
from flowbook.util.output import log, timer

if TYPE_CHECKING:
    from flowbook.kernel_support.checkpoint import Checkpoints


class MonotonicityEnforcer:
    """
    Enforces monotonicity constraints on cell execution.

    Monotonicity means that read-before-write (RBW) variables must not change
    their values during cell execution. This ensures that cells produce
    consistent results regardless of execution order.

    Attributes:
        _checkpoints: Checkpoint manager for saving/restoring state
        _user_ns: The user namespace dict being monitored
        _pre_checkpoint_name: Name used for the pre-execution checkpoint
        _structural_mode: How to handle structural reads (OFF, WARN, ENFORCE)
    """

    _pre_checkpoint_name = "_monotone_pre"

    def __init__(
        self,
        checkpoints: "Checkpoints",
        user_ns: dict,
        structural_mode: StructuralTrackingMode = StructuralTrackingMode.WARN,
    ):
        """
        Initialize the enforcer.

        Args:
            checkpoints: Checkpoint manager instance
            user_ns: The user namespace dict (may be TrackingDict)
            structural_mode: How to handle structural reads (default: WARN)
        """
        self._checkpoints = checkpoints
        self._user_ns = user_ns
        self._structural_mode = structural_mode

    def set_structural_mode(self, mode: StructuralTrackingMode) -> None:
        """Set the structural tracking mode."""
        self._structural_mode = mode

    def save_pre_state(self, cell_id: str) -> None:
        """
        Save pre-execution state for monotonicity checking.

        Call this before executing the cell code.

        Args:
            cell_id: Identifier of the cell being executed
        """
        with timer(
            key="monotone_save_checkpoint",
            message=f"[monotone] Saving pre-checkpoint for cell {cell_id}"
        ):
            self._checkpoints.save(self._pre_checkpoint_name, self._user_ns)

    def check_and_enforce(
        self,
        tracking_data: TrackingData,
        cell_id: str,
    ) -> Optional[MonotonicityViolation]:
        """
        Check monotonicity and restore state if violated.

        Call this after cell execution with the tracking data captured
        during execution. If RBW variables were modified, the pre-execution
        state is restored and a MonotonicityViolation is returned.

        Args:
            tracking_data: Variable access patterns from execution
            cell_id: Identifier of the cell that was executed

        Returns:
            None if monotonicity check passes, MonotonicityViolation if failed
        """
        with timer(
            key="monotone_check",
            message=f"[monotone] Checking monotonicity for cell {cell_id}"
        ):
            return self._do_check(tracking_data)

    def _do_check(self, tracking_data: TrackingData) -> Optional[MonotonicityViolation]:
        """Internal implementation of monotonicity check."""
        rbw_vars = tracking_data.get_rbw_vars()
        log(f"[monotone] Read-before-write vars: {rbw_vars}")

        if not rbw_vars:
            log("[monotone] No RBW vars to check, skipping")
            self._cleanup()
            return None

        # Get column-level RBW data
        column_rbw = tracking_data.get_column_rbw_sets()

        # Get structural reads
        structural_reads = tracking_data.structural_reads

        # Compare pre and post states
        with timer(key="monotone:diff", message="[monotone] Computing diff"):
            pre = self._checkpoints.get(self._pre_checkpoint_name)
            post = Checkpoint("_monotone_post", self._user_ns, {})
            diff_result = Checkpoint.diff(
                pre,
                post,
                keys_to_include=rbw_vars,
                use_leq=True,
                column_rbw=column_rbw,
                structural_reads=structural_reads,
                structural_mode=self._structural_mode,
            )

        log(
            f"[monotone] Diff result: "
            f"{list(diff_result.differences.keys()) if diff_result.differences else 'no differences'}"
        )

        if diff_result.differences:
            log("[monotone] FAILED - reverting state")
            with timer(key="monotone:restore", message="[monotone] Restoring checkpoint"):
                self._checkpoints.restore(self._pre_checkpoint_name, self._user_ns)
            self._cleanup()

            return MonotonicityViolation(
                violated_vars=list(diff_result.differences.keys()),
                diff_details=self._format_diff_details(diff_result),
                error_summary=f"Monotonicity violation: {list(diff_result.differences.keys())}",
            )

        log("[monotone] PASSED")
        self._cleanup()
        return None

    def _cleanup(self) -> None:
        """Remove temporary checkpoint."""
        self._checkpoints.delete(self._pre_checkpoint_name)

    def _format_diff_details(self, diff_result) -> str:
        """
        Format diff result into human-readable error details.

        Produces a multi-line string describing which RBW variables were
        modified and how their values changed.

        Args:
            diff_result: DiffResult from checkpoint comparison

        Returns:
            Human-readable error description
        """
        def format_node(var_name: str, node, path: str = "") -> list:
            """Recursively format a diff node into lines."""
            lines = []
            full_path = f"{var_name}{path}" if path else var_name

            if isinstance(node, ValueComparison):
                lines.append(f"  {full_path}: {node.message}")
            elif isinstance(node, dict):
                for key, child in node.items():
                    if key == "_truncated":
                        if isinstance(child, ValueComparison):
                            lines.append(f"  {full_path}: (truncated) {child.message}")
                        continue
                    lines.extend(format_node(var_name, child, f"{path}{key}"))

            return lines

        all_lines = [
            "Monotonicity violation - the following read-before-write variables were modified:"
        ]

        for var_name, node in diff_result.differences.items():
            all_lines.append(f"\n{var_name}:")
            detail_lines = format_node(var_name, node)
            # Show first 5 differences, then truncate
            all_lines.extend(detail_lines[:5])
            if len(detail_lines) > 5:
                all_lines.append(f"  ... and {len(detail_lines) - 5} more differences")

        return "\n".join(all_lines)
