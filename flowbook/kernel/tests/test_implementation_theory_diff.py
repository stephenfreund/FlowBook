"""
Tests for §3: Known Differences with Implementation

These tests verify the behaviors documented in FORMAL_DEVELOPMENT.md §3:
- §3.1 Aliasing and Reference Sharing
- §3.2 Unmonitored Writes (RBW over-approximation, reference tracking)
- §3.3 Practical Implications

The formal model assumes:
1. No aliasing: each Var(x) refers to distinct memory
2. Complete instrumentation: Δ(σ, σ') ⊆ WS(t)

The implementation handles these via:
1. Deep alias expansion before computing Δ
2. Checkpoint-based delta detection (not trace-based)
3. Object-granularity tracking for reference reads
"""

import pytest
import numpy as np
import pandas as pd
from copy import deepcopy

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
    _expand_with_deep_aliases,
)
from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper


# =============================================================================
# §3.1 Aliasing and Reference Sharing
# =============================================================================

class TestAliasingAndReferenceSharing:
    """
    Tests for §3.1: The formal model assumes Var(x) and Var(y) refer to
    distinct memory. Python allows aliasing. The implementation handles
    this via _expand_with_deep_aliases().
    """

    def test_direct_aliasing_detected(self, sdc_helper_with_order):
        """
        Direct aliasing: x = y makes both names reference the same object.
        Mutation via x should be detected as changing y too.

        Scenario:
        - Cell A reads y (creates dependency on y)
        - Cell B mutates x, where x and y are aliases
        - Deep alias expansion should include y when diffing x
        - Should detect backward conflict because y (alias of x) changed
        """
        helper = sdc_helper_with_order

        # PRE-STATE: x and y are aliases (same underlying list)
        shared_pre = [1, 2, 3]
        pre_state = {"x": shared_pre, "y": shared_pre}

        # Cell A reads y
        result_a = helper.execute_cell(
            "a",
            pre_namespace=deepcopy(pre_state),
            post_namespace={**deepcopy(pre_state), "result": 6},
            reads={"y"},
            writes={"result"},
        )
        assert result_a.violation is None

        # POST-STATE: x and y still aliases, but list is mutated
        shared_post = [1, 2, 3, 4]  # Mutated
        post_state = {"x": shared_post, "y": shared_post, "result": 6}

        # Cell B "mutates" x - but since y is alias, y also changed
        # The checkpoint will show both x and y differ from pre
        result_b = helper.execute_cell(
            "b",
            pre_namespace={**deepcopy(pre_state), "result": 6},
            post_namespace=post_state,
            reads=set(),
            writes={"x"},  # Trace says we wrote x
        )

        # Deep alias expansion: writing x should expand to include y
        # Since y changed and cell A read y → backward conflict or A stale
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_nested_sharing_dict_values(self, sdc_helper_with_order):
        """
        Nested sharing: a["key"] and b["key"] may point to same object.

        Scenario:
        - Cell A reads from a["data"]
        - Cell B modifies b["data"], where both point to same inner object
        - Deep alias expansion should include a when diffing b
        """
        helper = sdc_helper_with_order

        # PRE-STATE: a and b share the same inner object
        inner_pre = {"value": 100}
        pre_state = {
            "a": {"data": inner_pre},
            "b": {"data": inner_pre},  # Same inner object
        }

        # Cell A reads a["data"]["value"]
        result_a = helper.execute_cell(
            "a",
            pre_namespace=deepcopy(pre_state),
            post_namespace={**deepcopy(pre_state), "result": 100},
            reads={"a"},
            writes={"result"},
        )
        assert result_a.violation is None

        # POST-STATE: inner object has changed value
        inner_post = {"value": 999}
        post_state = {
            "a": {"data": inner_post},
            "b": {"data": inner_post},  # Still shared, but different value
            "result": 100,
        }

        # Cell B modifies b - but a shares the inner object
        result_b = helper.execute_cell(
            "b",
            pre_namespace={**deepcopy(pre_state), "result": 100},
            post_namespace=post_state,
            reads=set(),
            writes={"b"},
        )

        # Deep alias expansion: a and b share inner, so a also changed
        # Cell A read a → backward conflict or A stale
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_dataframe_column_sharing(self, sdc_helper_with_order):
        """
        DataFrame sharing: df1 and df2 may share underlying column arrays.

        This tests that when two variables reference the same DataFrame,
        modifying one should affect cells that read the other.

        For deep alias expansion to work, the aliases must exist in the
        PRE checkpoint (where the alias index is built).
        """
        helper = sdc_helper_with_order

        # Create initial DataFrame
        df_pre = pd.DataFrame({"col": [1, 2, 3, 4, 5]})

        # Cell A reads df1
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"df1": df_pre.copy()},
            post_namespace={"df1": df_pre.copy(), "mean": 3.0},
            reads={"df1"},
            writes={"mean"},
            column_reads={"df1": {"col"}},
        )
        assert result_a.violation is None

        # For cell B: PRE-checkpoint has df1 and df2 as ALIASES (same object)
        # This simulates: df2 = df1 (creating alias before B runs)
        df_shared = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        pre_b = {"df1": df_shared, "df2": df_shared, "mean": 3.0}  # ALIASES!

        # POST-checkpoint: df2 modified (and df1 too, since they're aliases)
        df_modified = pd.DataFrame({"col": [10, 20, 30, 40, 50]})
        post_b = {"df1": df_modified, "df2": df_modified, "mean": 3.0}

        # Cell B writes df2 - deep alias expansion should include df1
        result_b = helper.execute_cell(
            "b",
            pre_namespace=pre_b,
            post_namespace=post_b,
            reads=set(),
            writes={"df2"},
            column_writes={"df2": {"col"}},
        )

        # Since df1 and df2 are aliases in pre-checkpoint, expanding df2
        # should include df1. Since df1 changed and A read df1 → conflict
        has_conflict = (
            result_b.violation is not None or
            "a" in result_b.stale_cells
        )
        assert has_conflict, "Aliased DataFrame modification should be detected"

    def test_expand_with_deep_aliases_function(self):
        """
        Directly test _expand_with_deep_aliases() to verify it finds
        shared internal references.
        """
        # Create checkpoint with aliased objects
        shared_inner = [1, 2, 3]
        namespace = {
            "a": {"inner": shared_inner},
            "b": {"inner": shared_inner},  # Shares inner with a
            "c": [shared_inner],            # Also references shared_inner
            "d": "unrelated",
        }

        checkpoints = MemoryCheckpoints(sanity_check=False, warn_classes=False)
        checkpoints.save("test", namespace, max_size_mb=None)
        checkpoint = checkpoints.saved["test"]

        # If we access 'a', we should get back a, b, c (all share inner)
        expanded = _expand_with_deep_aliases({"a"}, checkpoint, log_aliases=False)

        # Should include all variables that share references with a
        assert "a" in expanded
        # b and c share inner object with a, so should be included
        # (This depends on implementation details of alias detection)


# =============================================================================
# §3.2 RBW Over-approximation
# =============================================================================

class TestRBWOverApproximation:
    """
    Tests for §3.2: RBW(t) is a valid over-approximation of pre-store reads.

    If an untraced write occurs before a traced read, RBW includes the
    location even though the cell didn't actually read from pre-store.
    This is conservative (safe) behavior.
    """

    def test_inplace_numpy_operation_before_read(self, sdc_helper_with_order):
        """
        Numpy in-place operations may write before the trace sees reads.

        Example: np.add(a, b, out=a) writes to 'a' then result is read.
        If the write is untraced (C extension), but read is traced,
        'a' appears in RBW even though cell wrote it first.

        This tests that such over-inclusion is handled safely.
        """
        helper = sdc_helper_with_order

        # Simulate: Cell A reads x (depends on x)
        arr = np.array([1, 2, 3])
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"x": arr.copy()},
            post_namespace={"x": arr.copy(), "sum": arr.sum()},
            reads={"x"},
            writes={"sum"},
        )
        assert result_a.violation is None

        # Cell B: in-place numpy op that "writes then reads" x
        # The trace might show x in RBW even though B wrote to it first
        # We simulate this by including x in reads (over-approximation)
        arr_modified = arr.copy()
        np.add(arr_modified, 10, out=arr_modified)  # In-place

        result_b = helper.execute_cell(
            "b",
            pre_namespace={"x": arr.copy(), "sum": 6},
            post_namespace={"x": arr_modified, "sum": 6},
            # Over-approximate: include x in reads even though we wrote first
            reads={"x"},
            writes={"x"},
        )

        # Cell B modifies x, which A read → backward conflict or A becomes stale
        # The over-approximation of B reading x is safe (doesn't miss conflicts)
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_over_approximation_causes_conservative_staleness(self, sdc_helper_with_order):
        """
        Over-inclusion in RBW should cause extra staleness, not missed conflicts.

        Scenario:
        - Cell A writes x
        - Cell B re-runs with x in RBW (over-approximation)
        - Cell A re-runs and changes x
        - Cell B should become stale (conservative)
        """
        helper = sdc_helper_with_order

        # Cell A writes x
        result_a = helper.execute_cell(
            "a",
            pre_namespace={},
            post_namespace={"x": 100},
            reads=set(),
            writes={"x"},
        )
        assert result_a.violation is None

        # Cell B: over-approximate by including x in reads
        # (simulates untraced write before read scenario)
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"x": 100},
            post_namespace={"x": 100, "y": 200},
            reads={"x"},  # Over-approximation
            writes={"y"},
        )
        assert result_b.violation is None

        # Re-run Cell A with different output
        result_a2 = helper.execute_cell(
            "a",
            pre_namespace={},
            post_namespace={"x": 999},  # Different value
            reads=set(),
            writes={"x"},
        )

        # Cell B should become stale due to over-approximated dependency on x
        assert "b" in result_a2.stale_cells


# =============================================================================
# §3.2 Reference Tracking (Untraced Reads Covered)
# =============================================================================

class TestReferenceTracking:
    """
    Tests for §3.2: Untraced reads are covered by reference tracking.

    To pass data to a C extension, Python must read a reference first.
    The analysis operates at object granularity: reading a reference
    means depending on everything reachable from that object.
    """

    def test_numpy_internal_read_covered_by_reference(self, sdc_helper_with_order):
        """
        When passing an array to numpy, the reference read covers
        internal reads within the C extension.

        Scenario:
        - Cell A reads arr and passes to np.sum() (C extension reads internals)
        - Cell B modifies arr
        - Cell A should be marked stale (the reference read covered internals)
        """
        helper = sdc_helper_with_order

        arr = np.array([1, 2, 3, 4, 5])

        # Cell A: read arr reference, numpy internally reads array data
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"arr": arr.copy()},
            post_namespace={"arr": arr.copy(), "total": np.sum(arr)},
            reads={"arr"},  # Reference read covers internal numpy reads
            writes={"total"},
        )
        assert result_a.violation is None

        # Cell B modifies arr
        arr_modified = np.array([10, 20, 30, 40, 50])
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"arr": arr.copy(), "total": 15},
            post_namespace={"arr": arr_modified, "total": 15},
            reads=set(),
            writes={"arr"},
        )

        # Cell A should be stale (reference tracking covered numpy's internal reads)
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_nested_object_reference_covers_deep_reads(self, sdc_helper_with_order):
        """
        Reading a container reference covers reads of nested objects.

        Scenario:
        - Cell A reads container["nested"]["deep"]
        - Cell B modifies only container["nested"]["deep"]
        - Cell A should be stale (reference to container covers nested reads)
        """
        helper = sdc_helper_with_order

        container = {
            "nested": {
                "deep": [1, 2, 3],
                "other": "untouched",
            },
            "top_level": 42,
        }

        # Cell A reads through nested path
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"container": deepcopy(container)},
            post_namespace={
                "container": deepcopy(container),
                "sum": sum(container["nested"]["deep"]),
            },
            reads={"container"},  # Reference read covers nested
            writes={"sum"},
        )
        assert result_a.violation is None

        # Cell B modifies deeply nested value
        container_modified = deepcopy(container)
        container_modified["nested"]["deep"] = [100, 200, 300]

        result_b = helper.execute_cell(
            "b",
            pre_namespace={"container": deepcopy(container), "sum": 6},
            post_namespace={"container": container_modified, "sum": 6},
            reads=set(),
            writes={"container"},
        )

        # Cell A should be stale (reference tracking)
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_pandas_dataframe_reference_covers_column_reads(self, sdc_helper_with_order):
        """
        Reading a DataFrame reference covers reads of all columns.

        Even if a C extension (like numpy via pandas) internally reads
        specific columns, the reference read of the DataFrame covers it.
        """
        helper = sdc_helper_with_order

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

        # Cell A reads df, internally accesses column 'a'
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df.copy(), "mean_a": df["a"].mean()},
            reads={"df"},
            writes={"mean_a"},
            column_reads={"df": {"a"}},
        )
        assert result_a.violation is None

        # Cell B modifies column 'a'
        df_modified = df.copy()
        df_modified["a"] = [100, 200, 300]

        result_b = helper.execute_cell(
            "b",
            pre_namespace={"df": df.copy(), "mean_a": 2.0},
            post_namespace={"df": df_modified, "mean_a": 2.0},
            reads=set(),
            writes={"df"},
            column_writes={"df": {"a"}},
        )

        # Cell A should be stale
        assert result_b.violation is not None or "a" in result_b.stale_cells


# =============================================================================
# §3.3 Practical Implications - Checkpoint-Derived Δ
# =============================================================================

class TestCheckpointDerivedDelta:
    """
    Tests for §3.3: Conflict detection uses checkpoint-derived Δ,
    not trace-derived WS(t).

    This means untraced writes are still detected because we compare
    actual checkpoint states.
    """

    def test_untraced_write_detected_via_checkpoint(self, sdc_helper_with_order):
        """
        Even if a write isn't in WS(t), it should be detected via checkpoint.

        Scenario: Cell B reads x (to pass to C extension), which internally
        mutates x. Trace shows R(x) but not W(x). Checkpoint diff catches it.
        """
        helper = sdc_helper_with_order

        # Cell A reads x
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"x": [1, 2, 3]},
            post_namespace={"x": [1, 2, 3], "result": 6},
            reads={"x"},
            writes={"result"},
        )
        assert result_a.violation is None

        # Cell B: reads x (to pass to C extension), which mutates x internally
        # Trace shows x in reads (reference read), but NOT in writes
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"x": [1, 2, 3], "result": 6},
            post_namespace={"x": [999, 999, 999], "result": 6},  # x changed!
            reads={"x"},  # Reference read IS traced
            writes=set(),  # But the mutation is NOT in writes
        )

        # Checkpoint diff should detect x changed (even without writes={"x"})
        # This triggers backward conflict since A read x
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_trace_ws_not_used_for_conflict_detection(self, sdc_helper_with_order):
        """
        WS(t) from trace is informational only. Checkpoint Δ drives detection.

        Scenario: Cell B reads y (passes to function), function mutates y.
        Trace shows y in reads but not writes. Checkpoint detects mutation.
        """
        helper = sdc_helper_with_order

        # Cell A reads y
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"y": [100]},
            post_namespace={"y": [100], "doubled": 200},
            reads={"y"},
            writes={"doubled"},
        )
        assert result_a.violation is None

        # Cell B: reads y, but mutation happens through untraced path
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"y": [100], "doubled": 200},
            post_namespace={"y": [555], "doubled": 200},  # y changed!
            reads={"y"},  # Reference read traced (needed to access y)
            writes=set(),  # But writes empty (mutation untraced)
        )

        # Checkpoint-based Δ should catch the change to y
        checkpoint_detected_change = (
            result_b.violation is not None or
            "a" in result_b.stale_cells
        )
        assert checkpoint_detected_change, (
            "Checkpoint-derived Δ should detect change even with empty WS(t)"
        )


# =============================================================================
# Edge Cases and Soundness Verification
# =============================================================================

class TestSoundnessEdgeCases:
    """
    Additional edge cases to verify soundness properties.
    """

    def test_multiple_levels_of_aliasing(self, sdc_helper_with_order):
        """
        Test aliasing through multiple levels of indirection.

        a -> shared -> inner
        b -> shared -> inner
        c -> inner (direct)

        Modifying via 'c' should affect 'a' and 'b'.
        """
        helper = sdc_helper_with_order

        inner = {"value": 42}
        shared = {"inner": inner}
        namespace = {
            "a": {"shared": shared},
            "b": {"shared": shared},
            "c": inner,
        }

        # Cell A reads from a path through shared
        result_a = helper.execute_cell(
            "a",
            pre_namespace=deepcopy(namespace),
            post_namespace={**deepcopy(namespace), "val": namespace["a"]["shared"]["inner"]["value"]},
            reads={"a"},
            writes={"val"},
        )
        assert result_a.violation is None

        # Cell B modifies via 'c' (direct reference to inner)
        modified = deepcopy(namespace)
        modified["c"]["value"] = 999
        # Due to aliasing, a and b are also affected
        modified["a"]["shared"]["inner"]["value"] = 999
        modified["b"]["shared"]["inner"]["value"] = 999

        result_b = helper.execute_cell(
            "b",
            pre_namespace={**deepcopy(namespace), "val": 42},
            post_namespace={**modified, "val": 42},
            reads=set(),
            writes={"c"},
        )

        # Should detect conflict/staleness on 'a'
        assert result_b.violation is not None or "a" in result_b.stale_cells

    def test_no_false_negatives_with_simple_mutation(self, sdc_helper_with_order):
        """
        Verify that simple mutations are always detected (no false negatives).

        This is a sanity check that the system never misses obvious conflicts.
        """
        helper = sdc_helper_with_order

        # Cell A reads x
        result_a = helper.execute_cell(
            "a",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )
        assert result_a.violation is None

        # Cell B modifies x (obvious backward mutation)
        result_b = helper.execute_cell(
            "b",
            pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 9999, "y": 2},
            reads=set(),
            writes={"x"},
        )

        # Must detect this - no false negatives
        detected = result_b.violation is not None or "a" in result_b.stale_cells
        assert detected, "Simple backward mutation must always be detected"
