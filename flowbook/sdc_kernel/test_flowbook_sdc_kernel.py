"""Tests for FlowbookSDCKernel."""

import pytest

from .conftest import make_tracking, SDCTestHelper
from .sdc_enforcer import PRE_CHECKPOINT_PREFIX


class TestSDCEnforcerGetStaleCells:
    """Tests for get_stale_cells method."""

    def test_get_stale_cells_empty(self, sdc_helper_with_order):
        """No stale cells when nothing has been executed."""
        assert sdc_helper_with_order.sdc.get_stale_cells() == []

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
        assert helper.sdc.get_stale_cells() == ["b"]

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


class TestSDCEnforcerReset:
    """Tests for reset functionality."""

    def test_reset_clears_stale_cells(self, sdc_helper_with_order):
        """reset() clears the stale cell cache."""
        helper = sdc_helper_with_order

        # Create some stale state
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        helper.execute_cell("a", {"x": 1, "y": 2}, {"x": 100, "y": 2}, writes={"x"})

        assert helper.sdc.get_stale_cells() == ["b"]

        # Reset
        helper.sdc.reset()

        assert helper.sdc.get_stale_cells() == []
        assert helper.sdc.records == {}
        assert helper.sdc.seq_counter == 0


class TestSDCEnforcerComputeAllStaleCells:
    """Tests for compute_all_stale_cells method."""

    def test_compute_all_stale_cells_recomputes(self, sdc_helper_with_order):
        """compute_all_stale_cells recomputes from scratch."""
        helper = sdc_helper_with_order

        # Create state
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # Get current namespace checkpoint
        helper.checkpoints.save("current", {"x": 100, "y": 2}, max_size_mb=None)
        current = helper.checkpoints.saved["current"]

        # Manually clear stale cache to simulate external modification
        helper.sdc._stale_cells.clear()
        assert helper.sdc.get_stale_cells() == []

        # Recompute should find b stale
        result = helper.sdc.compute_all_stale_cells(current)
        assert result == ["b"]
        assert helper.sdc.get_stale_cells() == ["b"]


class TestHelperFunctions:
    """Tests for SDCTestHelper convenience methods."""

    def test_execute_cell_returns_result(self, sdc_helper_with_order):
        """execute_cell returns SDCResult."""
        result = sdc_helper_with_order.execute_cell(
            "a", {}, {"x": 1}, writes={"x"}
        )
        assert result.violation is None
        assert result.stale_cells == []

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
        assert result_a.violation is None

        # B modifies df.qty (different column) - no violation
        df_modified = df.copy()
        df_modified["qty"] = [10, 20]
        result_b = helper.execute_cell(
            "b", {"df": df, "total": 30}, {"df": df_modified, "total": 30},
            reads={"df"}, writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"qty"}}
        )
        assert result_b.violation is None

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
        assert result.violation is not None
        assert result.violation.affected_cell == "a"
        assert result.violation.mutating_cell == "b"


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
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        assert kernel._is_pure_magic("%some_magic arg")

    def test_cell_magic_with_code_not_pure(self):
        """Cell magic with code body is NOT pure magic (body is tracked)."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        # Cell magic body contains real Python code that needs tracking
        assert not kernel._is_pure_magic("%%timeit\nx = 1")

    def test_shell_command(self):
        """Shell command is detected."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        assert kernel._is_pure_magic("!ls -la")

    def test_magic_with_comments(self):
        """Magic with comments is detected as pure magic."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        code = """# This is a comment
%structural_tracking off"""
        assert kernel._is_pure_magic(code)

    def test_magic_with_multiline_comments(self):
        """Magic with multiple comment lines is detected."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        code = """# Comment 1
# Comment 2
%some_magic
# Trailing comment"""
        assert kernel._is_pure_magic(code)

    def test_regular_code_not_pure_magic(self):
        """Regular Python code is not pure magic."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        assert not kernel._is_pure_magic("x = 1")

    def test_mixed_code_and_magic_not_pure(self):
        """Code mixed with magic is not pure magic."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        code = """%magic
x = 1"""
        assert not kernel._is_pure_magic(code)

    def test_empty_code(self):
        """Empty code is pure magic (vacuously true)."""
        from .flowbook_sdc_kernel import FlowbookSDCKernel
        kernel = FlowbookSDCKernel.__new__(FlowbookSDCKernel)
        assert kernel._is_pure_magic("")
        assert kernel._is_pure_magic("   \n  \n  ")
