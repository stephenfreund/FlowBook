"""Tests for FlowbookKernel."""

import pytest

from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper
from flowbook.kernel.reproducibility_enforcer import PRE_CHECKPOINT_PREFIX


class TestReproducibilityEnforcerGetStaleCells:
    """Tests for get_stale_cells method."""

    def test_get_stale_cells_empty(self, sdc_helper_with_order):
        """Unexecuted cells are stale with NEVER_EXECUTED reason.

        With reason tracking, cells in cell_order start as NeverExecuted.
        """
        # All 4 cells (a, b, c, d) are stale because they haven't executed
        stale = sdc_helper_with_order.sdc.get_stale_cells()
        assert set(stale) == {"a", "b", "c", "d"}

    def test_get_stale_cells_after_staleness(self, sdc_helper_with_order):
        """get_stale_cells returns cached staleness."""
        helper = sdc_helper_with_order

        # A writes x
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})

        # B reads x
        helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # A writes x again with different value -> B becomes stale
        result = helper.execute_cell("a", {"x": 1, "y": 2}, {"x": 100, "y": 2}, writes={"x"})

        assert "b" in result.stale_cells
        # B is stale (FORWARD_STALE), c and d are stale (NEVER_EXECUTED)
        stale = helper.sdc.get_stale_cells()
        assert "b" in stale
        assert "c" in stale  # Never executed
        assert "d" in stale  # Never executed
        assert "a" not in stale  # Just executed, so fresh

    def test_get_stale_cells_in_document_order(self, sdc_helper_with_order):
        """Stale cells are returned in document order."""
        helper = sdc_helper_with_order

        # A writes x
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})

        # Execute cells out of order: d, b, c (all read x)
        helper.execute_cell("d", {"x": 1}, {"x": 1, "w": 4}, reads={"x"}, writes={"w"})
        helper.execute_cell("b", {"x": 1, "w": 4}, {"x": 1, "w": 4, "y": 2}, reads={"x"}, writes={"y"})
        helper.execute_cell("c", {"x": 1, "w": 4, "y": 2}, {"x": 1, "w": 4, "y": 2, "z": 3}, reads={"x"}, writes={"z"})

        # A changes x -> b, c, d all become stale
        result = helper.execute_cell(
            "a",
            {"x": 1, "w": 4, "y": 2, "z": 3},
            {"x": 100, "w": 4, "y": 2, "z": 3},
            writes={"x"}
        )

        # Should be in document order [b, c, d], not execution order [d, b, c]
        assert helper.sdc.get_stale_cells() == ["b", "c", "d"]


class TestReproducibilityEnforcerReset:
    """Tests for reset functionality."""

    def test_reset_clears_stale_cells(self, sdc_helper_with_order):
        """reset() clears the stale cell cache."""
        helper = sdc_helper_with_order

        # Create some stale state
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        helper.execute_cell("a", {"x": 1, "y": 2}, {"x": 100, "y": 2}, writes={"x"})

        # B is stale (FORWARD_STALE), c and d are stale (NEVER_EXECUTED)
        stale = helper.sdc.get_stale_cells()
        assert "b" in stale
        assert "c" in stale  # Never executed
        assert "d" in stale  # Never executed
        assert "a" not in stale  # Just executed, so fresh

        # Reset
        helper.sdc.reset()

        # After reset, cell_order is cleared so no cells are stale
        assert helper.sdc.get_stale_cells() == []
        assert helper.sdc._notebook_state.tracking_data == {}
        assert helper.sdc.seq_counter == 0


class TestReproducibilityEnforcerComputeAllStaleCells:
    """Tests for compute_all_stale_cells method."""

    def test_compute_all_stale_cells_recomputes(self, sdc_helper_with_order):
        """compute_all_stale_cells recomputes from scratch."""
        helper = sdc_helper_with_order

        # Create state
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # Get current namespace checkpoint with modified x
        helper.checkpoints.save("current", {"x": 100, "y": 2}, max_size_mb=None)
        current = helper.checkpoints.saved["current"]

        # Manually mark b as clean to simulate external modification
        helper.sdc._notebook_state.set_clean("b")
        # Verify b is now clean but c, d are still NEVER_EXECUTED
        stale_before = helper.sdc.get_stale_cells()
        assert "b" not in stale_before
        assert "c" in stale_before  # Never executed
        assert "d" in stale_before  # Never executed

        # Recompute should find b stale again (FORWARD_STALE from x)
        result = helper.sdc.compute_all_stale_cells(current)
        assert "b" in result  # b is stale because x changed
        # c and d are also stale (NEVER_EXECUTED)
        stale_after = helper.sdc.get_stale_cells()
        assert "b" in stale_after
        assert "c" in stale_after
        assert "d" in stale_after


class TestHelperFunctions:
    """Tests for ReproducibilityTestHelper convenience methods."""

    def test_execute_cell_returns_result(self, sdc_helper_with_order):
        """execute_cell returns ReproducibilityResult."""
        result = sdc_helper_with_order.execute_cell(
            "a", {}, {"x": 1}, writes={"x"}
        )
        assert not result.has_errors()
        # b, c, d are stale (NEVER_EXECUTED), a is clean (just executed)
        assert "a" not in result.stale_cells
        assert set(result.stale_cells) == {"b", "c", "d"}

    def test_execute_cell_with_column_tracking(self, sdc_helper_with_order):
        """execute_cell supports column tracking."""
        import pandas as pd

        df = pd.DataFrame({"price": [10, 20], "qty": [1, 2]})
        helper = sdc_helper_with_order

        # A reads df.price
        result_a = helper.execute_cell(
            "a", {"df": df}, {"df": df, "total": 30},
            reads={"df"}, writes={"total"},
            column_reads={"df": {"price"}}
        )
        assert not result_a.has_errors()

        # B modifies df.qty (different column) - no backward mutation
        from flowbook.kernel.models import ErrorType
        df_modified = df.copy()
        df_modified["qty"] = [10, 20]
        result_b = helper.execute_cell(
            "b", {"df": df, "total": 30}, {"df": df_modified, "total": 30},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"qty"}}
        )
        # B reads and writes df so NoReadAndWrite fires, but no backward mutation
        assert not any(
            e.error_type == ErrorType.NO_WRITE_AFTER_READ for e in result_b.errors
        )

    def test_execute_cell_detects_violation(self, sdc_helper_with_order):
        """execute_cell detects backward mutation."""
        helper = sdc_helper_with_order

        # A reads x
        helper.execute_cell("a", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # B modifies x - violation
        result = helper.execute_cell(
            "b", {"x": 1, "y": 2}, {"x": 999, "y": 2},
            writes={"x"}
        )
        assert result.has_errors()
        assert result.errors[0].causer_cell == "a"
        assert result.errors[0].cell_id == "b"


class TestMakeTracking:
    """Tests for make_tracking helper."""

    def test_make_tracking_defaults(self):
        """make_tracking with no args returns empty sets."""
        tracking = make_tracking()
        assert tracking.reads_before_writes == set()
        assert tracking.writes == set()
        assert tracking.column_reads_before_writes == {}
        assert tracking.column_writes == {}

    def test_make_tracking_with_values(self):
        """make_tracking sets values correctly."""
        tracking = make_tracking(
            reads={"x", "y"},
            writes={"z"},
            column_reads={"df": {"price"}},
            column_writes={"df": {"qty"}}
        )
        assert tracking.reads_before_writes == {"x", "y"}
        assert tracking.writes == {"z"}
        assert tracking.column_reads_before_writes == {"df": {"price"}}
        assert tracking.column_writes == {"df": {"qty"}}


class TestIsPureMagic:
    """Tests for _is_pure_magic detection."""

    def test_pure_line_magic(self):
        """Single line magic is detected."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        assert kernel._is_pure_magic("%some_magic arg")

    def test_cell_magic_with_code_not_pure(self):
        """Cell magic with code body is NOT pure magic (body is tracked)."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        # Cell magic body contains real Python code that needs tracking
        assert not kernel._is_pure_magic("%%timeit\nx = 1")

    def test_shell_command(self):
        """Shell command is detected."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        assert kernel._is_pure_magic("!ls -la")

    def test_magic_with_comments(self):
        """Magic with comments is detected as pure magic."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        code = """# This is a comment
%structural_tracking off"""
        assert kernel._is_pure_magic(code)

    def test_magic_with_multiline_comments(self):
        """Magic with multiple comment lines is detected."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        code = """# Comment 1
# Comment 2
%some_magic
# Trailing comment"""
        assert kernel._is_pure_magic(code)

    def test_regular_code_not_pure_magic(self):
        """Regular Python code is not pure magic."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        assert not kernel._is_pure_magic("x = 1")

    def test_mixed_code_and_magic_not_pure(self):
        """Code mixed with magic is not pure magic."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        code = """%magic
x = 1"""
        assert not kernel._is_pure_magic(code)

    def test_empty_code(self):
        """Empty code is pure magic (vacuously true)."""
        from flowbook.kernel.flowbook_kernel import FlowbookKernel
        kernel = FlowbookKernel.__new__(FlowbookKernel)
        assert kernel._is_pure_magic("")
        assert kernel._is_pure_magic("   \n  \n  ")
