"""Tests for base_kernel.py - BaseFlowbookKernel shared kernel infrastructure.

Targets uncovered helper methods without requiring full kernel instantiation.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestExtractCellId:
    """Tests for BaseFlowbookKernel._extract_cell_id (static logic)."""

    def test_cell_id_from_argument(self):
        """cell_id argument takes priority."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        result = BaseFlowbookKernel._extract_cell_id(None, "abcd", None)
        assert result == "abcd"

    def test_cell_id_from_metadata(self):
        """cell_id extracted from cell_meta dict."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        result = BaseFlowbookKernel._extract_cell_id(None, None, {"cell_id": "efgh"})
        assert result == "efgh"

    def test_cell_id_argument_over_metadata(self):
        """cell_id argument takes priority over metadata."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        result = BaseFlowbookKernel._extract_cell_id(None, "abcd", {"cell_id": "efgh"})
        assert result == "abcd"

    def test_cell_id_none_when_both_none(self):
        """Returns None when both cell_id and cell_meta are None."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        result = BaseFlowbookKernel._extract_cell_id(None, None, None)
        assert result is None

    def test_cell_id_none_when_meta_has_no_cell_id(self):
        """Returns None when cell_meta has no cell_id key."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        result = BaseFlowbookKernel._extract_cell_id(None, None, {"other": "data"})
        assert result is None


class TestIsPureMagic:
    """Tests for BaseFlowbookKernel._is_pure_magic."""

    def test_single_magic_line(self):
        """Single magic command is pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        assert BaseFlowbookKernel._is_pure_magic(None, "%timeit x")

    def test_shell_command(self):
        """Shell command is pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        assert BaseFlowbookKernel._is_pure_magic(None, "!ls")

    def test_comment_only(self):
        """Comment-only code is pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        assert BaseFlowbookKernel._is_pure_magic(None, "# just a comment")

    def test_mixed_magic_and_comments(self):
        """Mix of magic, shell, and comments is pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        code = "%magic1\n!shell\n# comment"
        assert BaseFlowbookKernel._is_pure_magic(None, code)

    def test_regular_code_not_magic(self):
        """Regular Python code is not pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        assert not BaseFlowbookKernel._is_pure_magic(None, "x = 1")

    def test_mixed_code_and_magic(self):
        """Mix of regular code and magic is not pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        code = "%timeit x\nx = 1"
        assert not BaseFlowbookKernel._is_pure_magic(None, code)

    def test_force_checkpoint_flag(self):
        """Code with __flowbook_force_checkpoint__ is never pure magic."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        code = "%magic\n# __flowbook_force_checkpoint__"
        assert not BaseFlowbookKernel._is_pure_magic(None, code)

    def test_empty_lines_ignored(self):
        """Empty lines are ignored in magic detection."""
        from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
        code = "%magic1\n\n\n%magic2\n"
        assert BaseFlowbookKernel._is_pure_magic(None, code)
