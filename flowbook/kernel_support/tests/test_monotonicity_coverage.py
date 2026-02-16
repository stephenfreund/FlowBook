"""Tests for monotonicity.py - Targeting remaining uncovered lines.

Coverage gaps:
- set_structural_mode method (line 85)
- _format_diff_details with truncated/many diffs (lines 203-208, 222)
"""

import pytest
import pandas as pd

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.monotonicity import MonotonicityEnforcer
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.structural_tracking import StructuralTrackingMode
from flowbook.kernel_support.types import ValueComparison, MemoryCheckpointDiffResult


class TestSetStructuralMode:
    """Tests for set_structural_mode method."""

    def test_set_structural_mode_warn(self):
        """set_structural_mode changes the structural mode."""
        checkpoints = MemoryCheckpoints()
        user_ns = {"x": 1}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.set_structural_mode(StructuralTrackingMode.ENFORCE)
        assert enforcer._structural_mode == StructuralTrackingMode.ENFORCE

    def test_set_structural_mode_off(self):
        """set_structural_mode to OFF disables structural checking."""
        checkpoints = MemoryCheckpoints()
        user_ns = {}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.set_structural_mode(StructuralTrackingMode.OFF)
        assert enforcer._structural_mode == StructuralTrackingMode.OFF


class TestFormatDiffDetails:
    """Tests for _format_diff_details method."""

    def test_format_with_value_comparison(self):
        """Format diff details with ValueComparison nodes."""
        checkpoints = MemoryCheckpoints()
        user_ns = {}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        vc = ValueComparison(status="different", value1=1, value2=2, message="1 vs 2")
        diff_result = MemoryCheckpointDiffResult(differences={"x": vc})
        result = enforcer._format_diff_details(diff_result)
        assert "x:" in result
        assert "1 vs 2" in result
        assert "Monotonicity violation" in result

    def test_format_with_nested_dict(self):
        """Format diff details with nested dict nodes."""
        checkpoints = MemoryCheckpoints()
        user_ns = {}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        vc = ValueComparison(status="different", value1=1, value2=2, message="diff")
        diff_result = MemoryCheckpointDiffResult(
            differences={"data": {"[0]": vc, "[1]": vc}}
        )
        result = enforcer._format_diff_details(diff_result)
        assert "data:" in result

    def test_format_truncation_with_many_diffs(self):
        """Format diff details truncates after 5 sub-diffs."""
        checkpoints = MemoryCheckpoints()
        user_ns = {}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        # Create a diff result with many sub-keys
        children = {}
        for i in range(10):
            children[f"[{i}]"] = ValueComparison(
                status="different", value1=i, value2=i + 1, message=f"{i} vs {i+1}"
            )
        diff_result = MemoryCheckpointDiffResult(
            differences={"data": children}
        )
        result = enforcer._format_diff_details(diff_result)
        assert "... and" in result
        assert "more differences" in result

    def test_format_with_truncated_marker(self):
        """Format diff details handles _truncated marker in dict."""
        checkpoints = MemoryCheckpoints()
        user_ns = {}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        vc_truncated = ValueComparison(
            status="different", value1="trunc", value2="info", message="truncated details"
        )
        diff_result = MemoryCheckpointDiffResult(
            differences={"arr": {"_truncated": vc_truncated}}
        )
        result = enforcer._format_diff_details(diff_result)
        assert "truncated" in result.lower()

    def test_format_with_mixed_violation(self):
        """Full monotonicity check that triggers violation and formats it."""
        checkpoints = MemoryCheckpoints()
        user_ns = {"x": 1, "y": "hello"}
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)

        # Save pre state
        enforcer.save_pre_state("cell1")

        # Modify x
        user_ns["x"] = 999

        # Check monotonicity
        tracking = TrackingData(reads_before_writes={"x"}, writes=set())
        violation = enforcer.check_and_enforce(tracking, "cell1")

        assert violation is not None
        assert "x" in violation.violated_vars
        assert "Monotonicity violation" in violation.diff_details
        # State should be restored
        assert user_ns["x"] == 1
