"""Tests for fix_repro_errors.py, especially cell magic handling."""

import pytest

from flowbook.scripts.fix_repro_errors import (
    split_cell_magic,
    prepend_to_cell_source,
    get_cell_source,
    set_cell_source,
    add_deepcopy_and_rename,
    split_diagnostic_cell,
    alpha_rename_reused_variable,
    FLOWBOOK_FIX_MARKER,
)


class TestSplitCellMagic:
    """Tests for split_cell_magic function."""

    def test_cell_with_time_magic(self):
        """%%time magic should be preserved at the top."""
        source = "%%time\nx = 1\ny = 2\n"
        magic, rest = split_cell_magic(source)
        assert magic == "%%time\n"
        assert rest == "x = 1\ny = 2\n"

    def test_cell_with_timeit_magic(self):
        """%%timeit magic should be preserved."""
        source = "%%timeit\nsum(range(1000))\n"
        magic, rest = split_cell_magic(source)
        assert magic == "%%timeit\n"
        assert rest == "sum(range(1000))\n"

    def test_cell_with_capture_magic(self):
        """%%capture magic should be preserved."""
        source = "%%capture output\nprint('hello')\n"
        magic, rest = split_cell_magic(source)
        assert magic == "%%capture output\n"
        assert rest == "print('hello')\n"

    def test_cell_without_magic(self):
        """Cells without magic should return empty prefix."""
        source = "x = 1\ny = 2\n"
        magic, rest = split_cell_magic(source)
        assert magic == ""
        assert rest == "x = 1\ny = 2\n"

    def test_cell_with_line_magic_not_cell_magic(self):
        """Line magics (%time) should NOT be treated as cell magics."""
        source = "%time x = 1\ny = 2\n"
        magic, rest = split_cell_magic(source)
        assert magic == ""
        assert rest == "%time x = 1\ny = 2\n"

    def test_cell_with_leading_blank_lines_then_magic(self):
        """Blank lines before magic should be included in prefix."""
        source = "\n%%time\nx = 1\n"
        magic, rest = split_cell_magic(source)
        assert magic == "\n%%time\n"
        assert rest == "x = 1\n"

    def test_cell_with_comment_then_magic(self):
        """Comments before magic should be included in prefix."""
        source = "# timing this cell\n%%time\nx = 1\n"
        magic, rest = split_cell_magic(source)
        assert magic == "# timing this cell\n%%time\n"
        assert rest == "x = 1\n"

    def test_cell_with_only_magic(self):
        """Cell with only a magic command."""
        source = "%%time\n"
        magic, rest = split_cell_magic(source)
        assert magic == "%%time\n"
        assert rest == ""

    def test_empty_cell(self):
        """Empty cell should return empty strings."""
        source = ""
        magic, rest = split_cell_magic(source)
        assert magic == ""
        assert rest == ""

    def test_magic_with_indentation(self):
        """Magic with leading whitespace should still be detected."""
        source = "  %%time\nx = 1\n"
        magic, rest = split_cell_magic(source)
        assert magic == "  %%time\n"
        assert rest == "x = 1\n"

    def test_comment_without_magic_not_preserved(self):
        """Leading comments without magic should NOT be preserved as prefix."""
        source = "# just a comment\nx = 1\n"
        magic, rest = split_cell_magic(source)
        assert magic == ""
        assert rest == "# just a comment\nx = 1\n"


class TestPrependToCellSource:
    """Tests for prepend_to_cell_source function."""

    def test_prepend_to_cell_with_magic(self):
        """Prefix should be inserted AFTER magic, not before."""
        source = "%%time\nx = 1\n"
        result = prepend_to_cell_source(source, "# COMMENT\n")
        assert result == "%%time\n# COMMENT\nx = 1\n"

    def test_prepend_to_cell_without_magic(self):
        """Prefix should be at the start when no magic."""
        source = "x = 1\n"
        result = prepend_to_cell_source(source, "# COMMENT\n")
        assert result == "# COMMENT\nx = 1\n"

    def test_prepend_multiline_to_magic_cell(self):
        """Multi-line prefix should be inserted after magic."""
        source = "%%time\nx = 1\n"
        prefix = "# Line 1\n# Line 2\nimport copy\n"
        result = prepend_to_cell_source(source, prefix)
        assert result == "%%time\n# Line 1\n# Line 2\nimport copy\nx = 1\n"

    def test_prepend_preserves_timeit(self):
        """%%timeit should be preserved at top."""
        source = "%%timeit\nsum(range(1000))\n"
        result = prepend_to_cell_source(source, "# FIX\n")
        assert result == "%%timeit\n# FIX\nsum(range(1000))\n"

    def test_prepend_preserves_capture(self):
        """%%capture should be preserved at top."""
        source = "%%capture out\nprint('x')\n"
        result = prepend_to_cell_source(source, "# FIX\n")
        assert result == "%%capture out\n# FIX\nprint('x')\n"


class TestAddDeepcopyAndRenameWithMagic:
    """Tests that add_deepcopy_and_rename preserves cell magics."""

    def _make_notebook(self, cells_source: list[str]) -> dict:
        """Create a minimal notebook structure."""
        cells = []
        for i, source in enumerate(cells_source):
            cells.append({
                "cell_type": "code",
                "id": f"cell{i}",
                "source": source.splitlines(keepends=True) if source else [""],
            })
        return {"cells": cells}

    def test_deepcopy_preserves_time_magic(self):
        """Deep copy fix should insert code after %%time."""
        notebook = self._make_notebook([
            "%%time\nx = df.copy()\n",
            "print(x)\n",
        ])

        add_deepcopy_and_rename(notebook, 0, "df", "abcd")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # %%time must be first
        assert lines[0] == "%%time"
        # Fix comments and code come after
        assert FLOWBOOK_FIX_MARKER in result
        assert "import copy" in result
        assert "deepcopy" in result

    def test_deepcopy_without_magic(self):
        """Deep copy fix should work normally without magic."""
        notebook = self._make_notebook([
            "x = df.copy()\n",
            "print(x)\n",
        ])

        add_deepcopy_and_rename(notebook, 0, "df", "abcd")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # Fix comment should be first (no magic to preserve)
        assert lines[0].startswith(FLOWBOOK_FIX_MARKER)


class TestSplitDiagnosticCellWithMagic:
    """Tests that split_diagnostic_cell preserves cell magics."""

    def _make_notebook(self, cells_source: list[str]) -> dict:
        """Create a minimal notebook structure."""
        cells = []
        for i, source in enumerate(cells_source):
            cells.append({
                "cell_type": "code",
                "id": f"cell{i}",
                "source": source.splitlines(keepends=True) if source else [""],
            })
        return {"cells": cells}

    def test_diagnostic_preserves_time_magic(self):
        """Diagnostic fix should insert after %%time."""
        notebook = self._make_notebook([
            "%%time\ndf.head()\n",
        ])

        split_diagnostic_cell(notebook, 0)

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # %%time must be first
        assert lines[0] == "%%time"
        # Fix comments and %diagnostic come after
        assert FLOWBOOK_FIX_MARKER in result
        assert "%diagnostic" in result

    def test_diagnostic_without_magic(self):
        """Diagnostic fix should work normally without magic."""
        notebook = self._make_notebook([
            "df.head()\n",
        ])

        split_diagnostic_cell(notebook, 0)

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # Fix comment should be first
        assert lines[0].startswith(FLOWBOOK_FIX_MARKER)


class TestAlphaRenameWithMagic:
    """Tests that alpha_rename_reused_variable preserves cell magics."""

    def _make_notebook(self, cells_source: list[str]) -> dict:
        """Create a minimal notebook structure."""
        cells = []
        for i, source in enumerate(cells_source):
            cells.append({
                "cell_type": "code",
                "id": f"cell{i}",
                "source": source.splitlines(keepends=True) if source else [""],
            })
        return {"cells": cells}

    def test_rename_preserves_time_magic(self):
        """Rename fix should insert comment after %%time."""
        notebook = self._make_notebook([
            "%%time\nmodel = train()\n",
            "print(model)\n",
        ])

        alpha_rename_reused_variable(notebook, 0, "model", "abcd")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # %%time must be first
        assert lines[0] == "%%time"
        # Fix comment comes after
        assert FLOWBOOK_FIX_MARKER in result

    def test_rename_without_magic(self):
        """Rename fix should work normally without magic."""
        notebook = self._make_notebook([
            "model = train()\n",
            "print(model)\n",
        ])

        alpha_rename_reused_variable(notebook, 0, "model", "abcd")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # Fix comment should be first
        assert lines[0].startswith(FLOWBOOK_FIX_MARKER)
