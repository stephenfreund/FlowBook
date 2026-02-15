"""
Unit tests for the notebook_to_python conversion utilities.
"""

import pytest
import nbformat

from flowbook.util.notebook_to_python import (
    notebook_to_python,
    python_to_notebook_cells,
    apply_cell_updates,
    CELL_DELIMITER_PATTERN,
)


class TestCellDelimiterPattern:
    """Test the regex pattern for cell delimiters."""

    def test_matches_code_cell(self):
        """Test matching a code cell delimiter."""
        line = "# ====== CELL [abcd] (code) ======"
        match = CELL_DELIMITER_PATTERN.match(line)
        assert match is not None
        assert match.group(1) == "abcd"
        assert match.group(2) == "code"

    def test_matches_markdown_cell(self):
        """Test matching a markdown cell delimiter."""
        line = "# ====== CELL [efgh] (markdown) ======"
        match = CELL_DELIMITER_PATTERN.match(line)
        assert match is not None
        assert match.group(1) == "efgh"
        assert match.group(2) == "markdown"

    def test_no_match_regular_comment(self):
        """Test that regular comments don't match."""
        line = "# This is a regular comment"
        match = CELL_DELIMITER_PATTERN.match(line)
        assert match is None

    def test_no_match_partial_delimiter(self):
        """Test that partial delimiters don't match."""
        line = "# ====== CELL [abcd]"
        match = CELL_DELIMITER_PATTERN.match(line)
        assert match is None


class TestNotebookToPython:
    """Test notebook to Python conversion."""

    def test_single_code_cell(self):
        """Test converting a single code cell."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1")
        cell.id = "abcd"
        nb.cells = [cell]

        result = notebook_to_python(dict(nb))

        assert "# ====== CELL [abcd] (code) ======" in result
        assert "x = 1" in result

    def test_multiple_code_cells(self):
        """Test converting multiple code cells."""
        nb = nbformat.v4.new_notebook()
        cell1 = nbformat.v4.new_code_cell(source="x = 1")
        cell1.id = "aaaa"
        cell2 = nbformat.v4.new_code_cell(source="y = 2")
        cell2.id = "bbbb"
        nb.cells = [cell1, cell2]

        result = notebook_to_python(dict(nb))

        assert "# ====== CELL [aaaa] (code) ======" in result
        assert "# ====== CELL [bbbb] (code) ======" in result
        assert "x = 1" in result
        assert "y = 2" in result
        # First cell should come before second
        assert result.index("aaaa") < result.index("bbbb")

    def test_markdown_cell_prefixed(self):
        """Test that markdown cells have # prefix on each line."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_markdown_cell(source="# Title\n\nParagraph text")
        cell.id = "mdcl"
        nb.cells = [cell]

        result = notebook_to_python(dict(nb))

        assert "# ====== CELL [mdcl] (markdown) ======" in result
        # Each line of markdown should be prefixed
        assert "# # Title" in result
        assert "# Paragraph text" in result

    def test_mixed_cells(self):
        """Test converting mixed code and markdown cells."""
        nb = nbformat.v4.new_notebook()
        md = nbformat.v4.new_markdown_cell(source="# Header")
        md.id = "mark"
        code = nbformat.v4.new_code_cell(source="print('hello')")
        code.id = "code"
        nb.cells = [md, code]

        result = notebook_to_python(dict(nb))

        assert "(markdown)" in result
        assert "(code)" in result

    def test_multiline_code(self):
        """Test converting multiline code cells."""
        nb = nbformat.v4.new_notebook()
        source = """def foo():
    x = 1
    return x * 2"""
        cell = nbformat.v4.new_code_cell(source=source)
        cell.id = "mult"
        nb.cells = [cell]

        result = notebook_to_python(dict(nb))

        assert "def foo():" in result
        assert "    x = 1" in result
        assert "    return x * 2" in result

    def test_empty_notebook(self):
        """Test converting empty notebook."""
        nb = {"cells": [], "metadata": {}}

        result = notebook_to_python(nb)

        assert result == ""

    def test_list_source_format(self):
        """Test converting cells with list source format."""
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "id": "test",
                    "source": ["x = 1\n", "y = 2"]
                }
            ],
            "metadata": {}
        }

        result = notebook_to_python(nb)

        assert "x = 1" in result
        assert "y = 2" in result


class TestPythonToNotebookCells:
    """Test Python to notebook cells conversion."""

    def test_single_cell(self):
        """Test parsing a single cell."""
        python_source = """# ====== CELL [abcd] (code) ======
x = 1
"""
        cells = python_to_notebook_cells(python_source)

        assert len(cells) == 1
        assert cells[0]["id"] == "abcd"
        assert cells[0]["cell_type"] == "code"
        assert cells[0]["source"].strip() == "x = 1"

    def test_multiple_cells(self):
        """Test parsing multiple cells."""
        python_source = """# ====== CELL [aaaa] (code) ======
x = 1

# ====== CELL [bbbb] (code) ======
y = 2
"""
        cells = python_to_notebook_cells(python_source)

        assert len(cells) == 2
        assert cells[0]["id"] == "aaaa"
        assert cells[1]["id"] == "bbbb"

    def test_markdown_unprefix(self):
        """Test that markdown cells have # prefix removed."""
        python_source = """# ====== CELL [mdcl] (markdown) ======
# # Title
#
# Some text
"""
        cells = python_to_notebook_cells(python_source)

        assert len(cells) == 1
        assert cells[0]["cell_type"] == "markdown"
        assert "# Title" in cells[0]["source"]
        assert "Some text" in cells[0]["source"]
        # Should not have double # from prefix
        assert "# # Title" not in cells[0]["source"]

    def test_empty_source(self):
        """Test parsing empty Python source."""
        cells = python_to_notebook_cells("")
        assert cells == []

    def test_no_delimiters(self):
        """Test parsing source with no cell delimiters."""
        python_source = "x = 1\ny = 2"
        cells = python_to_notebook_cells(python_source)
        assert cells == []


class TestRoundtrip:
    """Test roundtrip conversion (notebook -> Python -> cells)."""

    def test_code_cell_roundtrip(self):
        """Test code cell roundtrip preserves content."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1\ny = x + 1")
        cell.id = "test"
        nb.cells = [cell]

        python_source = notebook_to_python(dict(nb))
        parsed = python_to_notebook_cells(python_source)

        assert len(parsed) == 1
        assert parsed[0]["id"] == "test"
        assert parsed[0]["source"].strip() == "x = 1\ny = x + 1"

    def test_multiline_roundtrip(self):
        """Test multiline code roundtrip."""
        nb = nbformat.v4.new_notebook()
        source = """def complex_function(a, b):
    result = a + b
    if result > 10:
        return result * 2
    return result"""
        cell = nbformat.v4.new_code_cell(source=source)
        cell.id = "func"
        nb.cells = [cell]

        python_source = notebook_to_python(dict(nb))
        parsed = python_to_notebook_cells(python_source)

        assert len(parsed) == 1
        assert "def complex_function(a, b):" in parsed[0]["source"]
        assert "if result > 10:" in parsed[0]["source"]


class TestApplyCellUpdates:
    """Test applying cell updates to notebook."""

    def test_update_single_cell(self):
        """Test updating a single cell."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1")
        cell.id = "test"
        nb.cells = [cell]

        updates = [{"id": "test", "cell_type": "code", "source": "x = 2"}]
        result = apply_cell_updates(dict(nb), updates)

        assert result["cells"][0]["source"] == "x = 2"

    def test_update_preserves_other_metadata(self):
        """Test that updates preserve other cell metadata."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1")
        cell.id = "test"
        cell.metadata["custom_key"] = "custom_value"
        nb.cells = [cell]

        updates = [{"id": "test", "cell_type": "code", "source": "x = 2"}]
        result = apply_cell_updates(dict(nb), updates)

        assert result["cells"][0]["source"] == "x = 2"
        assert result["cells"][0]["metadata"]["custom_key"] == "custom_value"

    def test_update_nonexistent_cell_ignored(self):
        """Test that updates to nonexistent cells are ignored."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1")
        cell.id = "test"
        nb.cells = [cell]

        updates = [{"id": "nonexistent", "cell_type": "code", "source": "y = 2"}]
        result = apply_cell_updates(dict(nb), updates)

        # Original cell unchanged
        assert len(result["cells"]) == 1
        assert result["cells"][0]["source"] == "x = 1"

    def test_partial_updates(self):
        """Test updating only some cells."""
        nb = nbformat.v4.new_notebook()
        cell1 = nbformat.v4.new_code_cell(source="x = 1")
        cell1.id = "aaaa"
        cell2 = nbformat.v4.new_code_cell(source="y = 2")
        cell2.id = "bbbb"
        nb.cells = [cell1, cell2]

        # Only update first cell
        updates = [{"id": "aaaa", "cell_type": "code", "source": "x = 100"}]
        result = apply_cell_updates(dict(nb), updates)

        assert result["cells"][0]["source"] == "x = 100"
        assert result["cells"][1]["source"] == "y = 2"


def test_module_imports():
    """Test that all expected functions are importable."""
    from flowbook.util.notebook_to_python import (
        notebook_to_python,
        python_to_notebook_cells,
        apply_cell_updates,
    )
    assert callable(notebook_to_python)
    assert callable(python_to_notebook_cells)
    assert callable(apply_cell_updates)


if __name__ == "__main__":
    print("Running notebook_to_python tests...")

    # Pattern tests
    pattern_tests = TestCellDelimiterPattern()
    pattern_tests.test_matches_code_cell()
    pattern_tests.test_matches_markdown_cell()
    pattern_tests.test_no_match_regular_comment()
    pattern_tests.test_no_match_partial_delimiter()
    print("✓ Cell delimiter pattern tests passed")

    # Notebook to Python tests
    nb_tests = TestNotebookToPython()
    nb_tests.test_single_code_cell()
    nb_tests.test_multiple_code_cells()
    nb_tests.test_markdown_cell_prefixed()
    nb_tests.test_mixed_cells()
    nb_tests.test_multiline_code()
    nb_tests.test_empty_notebook()
    nb_tests.test_list_source_format()
    print("✓ Notebook to Python tests passed")

    # Python to cells tests
    py_tests = TestPythonToNotebookCells()
    py_tests.test_single_cell()
    py_tests.test_multiple_cells()
    py_tests.test_markdown_unprefix()
    py_tests.test_empty_source()
    py_tests.test_no_delimiters()
    print("✓ Python to notebook cells tests passed")

    # Roundtrip tests
    rt_tests = TestRoundtrip()
    rt_tests.test_code_cell_roundtrip()
    rt_tests.test_multiline_roundtrip()
    print("✓ Roundtrip tests passed")

    # Apply updates tests
    apply_tests = TestApplyCellUpdates()
    apply_tests.test_update_single_cell()
    apply_tests.test_update_preserves_other_metadata()
    apply_tests.test_update_nonexistent_cell_ignored()
    apply_tests.test_partial_updates()
    print("✓ Apply cell updates tests passed")

    test_module_imports()
    print("✓ Module imports test passed")

    print("\n✓ All tests passed!")
