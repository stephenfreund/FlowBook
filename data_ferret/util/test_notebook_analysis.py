"""
Tests for NotebookAnalysis class.

To run:
    pytest data_ferret/util/test_notebook_analysis.py -v
"""

import pytest
import nbformat

from data_ferret.util.notebook_analysis import NotebookAnalysis


class TestNotebookAnalysisBasics:
    """Test basic NotebookAnalysis functionality."""

    @pytest.fixture
    def simple_notebook(self):
        """Create a simple notebook for testing."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="cell1", source="x = 1"),
            nbformat.v4.new_code_cell(id="cell2", source="y = x + 1"),
            nbformat.v4.new_code_cell(id="cell3", source="z = y * 2"),
        ]
        return nb

    def test_initialization(self, simple_notebook):
        """Test that NotebookAnalysis initializes correctly."""
        analysis = NotebookAnalysis(simple_notebook)

        assert analysis.notebook is simple_notebook
        assert len(analysis.get_all_cell_ids()) == 3
        assert "cell1" in analysis.get_all_cell_ids()
        assert "cell2" in analysis.get_all_cell_ids()
        assert "cell3" in analysis.get_all_cell_ids()

    def test_has_cell(self, simple_notebook):
        """Test has_cell method."""
        analysis = NotebookAnalysis(simple_notebook)

        assert analysis.has_cell("cell1") is True
        assert analysis.has_cell("cell2") is True
        assert analysis.has_cell("nonexistent") is False

    def test_get_dependencies(self, simple_notebook):
        """Test get_dependencies method."""
        analysis = NotebookAnalysis(simple_notebook)

        deps1 = analysis.get_dependencies("cell1")
        assert deps1 is not None
        assert deps1.cell_id == "cell1"
        assert "x" in deps1.globals_written

        deps2 = analysis.get_dependencies("cell2")
        assert deps2 is not None
        assert "x" in deps2.globals_read
        assert "y" in deps2.globals_written

    def test_get_liveness(self, simple_notebook):
        """Test get_liveness method."""
        analysis = NotebookAnalysis(simple_notebook)

        liveness1 = analysis.get_liveness("cell1")
        assert liveness1 is not None
        assert "x" in liveness1.live_out  # x is used by cell2

        liveness3 = analysis.get_liveness("cell3")
        assert liveness3 is not None
        assert len(liveness3.live_out) == 0  # Nothing live after last cell


class TestNotebookAnalysisConvenienceMethods:
    """Test convenience accessor methods."""

    @pytest.fixture
    def test_notebook(self):
        """Create a notebook for testing accessors."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="a = 1\nb = 2"),
            nbformat.v4.new_code_cell(id="c2", source="c = a + b"),
            nbformat.v4.new_code_cell(id="c3", source="result = c * 2"),
        ]
        return nb

    def test_get_globals_written(self, test_notebook):
        """Test get_globals_written method."""
        analysis = NotebookAnalysis(test_notebook)

        written1 = analysis.get_globals_written("c1")
        assert written1 == {"a", "b"}

        written2 = analysis.get_globals_written("c2")
        assert written2 == {"c"}

        # Nonexistent cell returns empty set
        written_none = analysis.get_globals_written("nonexistent")
        assert written_none == set()

    def test_get_globals_read(self, test_notebook):
        """Test get_globals_read method."""
        analysis = NotebookAnalysis(test_notebook)

        read1 = analysis.get_globals_read("c1")
        assert read1 == set()  # First cell reads nothing

        read2 = analysis.get_globals_read("c2")
        assert read2 == {"a", "b"}

        read3 = analysis.get_globals_read("c3")
        assert read3 == {"c"}

    def test_get_live_out_variables(self, test_notebook):
        """Test get_live_out_variables method."""
        analysis = NotebookAnalysis(test_notebook)

        live_out1 = analysis.get_live_out_variables("c1")
        assert live_out1 == {"a", "b"}  # Both used by c2

        live_out2 = analysis.get_live_out_variables("c2")
        assert live_out2 == {"c"}  # c used by c3

        live_out3 = analysis.get_live_out_variables("c3")
        assert live_out3 == set()  # Nothing live after last cell


class TestValidationVariables:
    """Test get_validation_variables - the key method for optimization."""

    @pytest.fixture
    def optimization_notebook(self):
        """Create a notebook simulating optimization scenarios."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            # Cell 1: writes x, y, temp - temp is never used
            nbformat.v4.new_code_cell(
                id="c1",
                source="x = 1\ny = 2\ntemp = 999"
            ),
            # Cell 2: reads x, y; writes result
            nbformat.v4.new_code_cell(
                id="c2",
                source="result = x + y"
            ),
            # Cell 3: reads result only
            nbformat.v4.new_code_cell(
                id="c3",
                source="final = result * 2"
            ),
        ]
        return nb

    def test_validation_variables_excludes_dead_vars(self, optimization_notebook):
        """Test that validation variables excludes dead variables."""
        analysis = NotebookAnalysis(optimization_notebook)

        # c1 writes {x, y, temp}
        # c1 live_out = {x, y} (temp is never used - dead)
        # validation = live_out = {x, y}
        validation_vars = analysis.get_validation_variables("c1")

        assert "x" in validation_vars
        assert "y" in validation_vars
        assert "temp" not in validation_vars  # Dead variable excluded

        # validation_vars should equal live_out
        assert validation_vars == analysis.get_live_out_variables("c1")

    def test_validation_variables_equals_live_out(self, optimization_notebook):
        """Test that validation variables equals live_out."""
        analysis = NotebookAnalysis(optimization_notebook)

        # c2 writes {result}
        # c2 live_out = {result}
        # validation = live_out = {result}
        validation_vars = analysis.get_validation_variables("c2")

        assert validation_vars == {"result"}
        assert validation_vars == analysis.get_live_out_variables("c2")

    def test_validation_variables_empty_for_last_cell(self):
        """Test that validation is empty for last cell (nothing live after)."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="x = 1"),
            nbformat.v4.new_code_cell(id="c2", source="y = x + 1"),  # Last cell
        ]

        analysis = NotebookAnalysis(nb)

        # c2 is last cell, so live_out = {} (nothing live after last cell)
        # validation = live_out = {}
        validation_vars = analysis.get_validation_variables("c2")
        assert validation_vars == set()
        assert validation_vars == analysis.get_live_out_variables("c2")

    def test_validation_with_function_calls(self):
        """Test validation with function dependencies."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            # Define function that uses global var
            nbformat.v4.new_code_cell(
                id="c1",
                source="def f():\n    return x + 1"
            ),
            # Define x
            nbformat.v4.new_code_cell(id="c2", source="x = 10"),
            # Call function
            nbformat.v4.new_code_cell(id="c3", source="result = f()"),
        ]

        analysis = NotebookAnalysis(nb)

        # c1: live_out = {f} (f is used in c3, but x isn't defined yet)
        validation_c1 = analysis.get_validation_variables("c1")
        assert "f" in validation_c1
        assert validation_c1 == analysis.get_live_out_variables("c1")

        # c2: live_out = {x, f} (both needed by c3)
        validation_c2 = analysis.get_validation_variables("c2")
        assert "x" in validation_c2
        assert "f" in validation_c2  # f from c1 is still live
        assert validation_c2 == analysis.get_live_out_variables("c2")

    def test_validation_catches_accidental_modification(self):
        """Test that validation includes variables from previous cells."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="data = [1, 2, 3]"),
            nbformat.v4.new_code_cell(id="c2", source="result = sum(data)"),  # Should not modify data
            nbformat.v4.new_code_cell(id="c3", source="final = data + result"),  # Uses both data and result
        ]

        analysis = NotebookAnalysis(nb)

        # c2: live_out = {data, result} (both needed by c3)
        # Validation should include 'data' to catch if c2 accidentally modifies it
        validation_c2 = analysis.get_validation_variables("c2")
        assert "data" in validation_c2  # From c1, should remain unchanged
        assert "result" in validation_c2  # Written by c2
        assert validation_c2 == {"data", "result"}


class TestDependencyFiltering:
    """Test dependency filtering methods for LLM context."""

    @pytest.fixture
    def dependency_notebook(self):
        """Create a notebook for testing dependency filtering."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="a = 1\nb = 2\nc = 3"),
            nbformat.v4.new_code_cell(id="c2", source="result = a + b"),  # Only uses a, b
            nbformat.v4.new_code_cell(id="c3", source="final = result + c"),
        ]
        return nb

    def test_get_dependency_variables(self, dependency_notebook):
        """Test get_dependency_variables method."""
        analysis = NotebookAnalysis(dependency_notebook)

        # c1 has no dependencies
        deps1 = analysis.get_dependency_variables("c1")
        assert deps1 == set()

        # c2 depends on a, b (not c)
        deps2 = analysis.get_dependency_variables("c2")
        assert deps2 == {"a", "b"}
        assert "c" not in deps2

        # c3 depends on result, c
        deps3 = analysis.get_dependency_variables("c3")
        assert deps3 == {"result", "c"}

    def test_filter_env_to_dependencies(self, dependency_notebook):
        """Test filter_env_to_dependencies method."""
        analysis = NotebookAnalysis(dependency_notebook)

        # Full environment with type info
        full_env = {
            "a": "int",
            "b": "int",
            "c": "int",
            "result": "int",
            "unrelated": "str",
        }

        # c2 only depends on a, b
        filtered_env = analysis.filter_env_to_dependencies("c2", full_env)

        assert filtered_env == {"a": "int", "b": "int"}
        assert "c" not in filtered_env
        assert "unrelated" not in filtered_env

    def test_filter_env_empty_dependencies(self):
        """Test filtering when cell has no dependencies."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="x = 42"),
        ]

        analysis = NotebookAnalysis(nb)

        full_env = {"a": "int", "b": "str"}
        filtered_env = analysis.filter_env_to_dependencies("c1", full_env)

        assert filtered_env == {}  # No dependencies, so empty


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_notebook(self):
        """Test with an empty notebook."""
        nb = nbformat.v4.new_notebook()
        nb.cells = []

        analysis = NotebookAnalysis(nb)

        assert len(analysis.get_all_cell_ids()) == 0
        assert analysis.has_cell("any") is False

    def test_nonexistent_cell_returns_none_or_empty(self):
        """Test that nonexistent cells return None or empty sets."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(id="c1", source="x = 1")]

        analysis = NotebookAnalysis(nb)

        assert analysis.get_dependencies("nonexistent") is None
        assert analysis.get_liveness("nonexistent") is None
        assert analysis.get_globals_written("nonexistent") == set()
        assert analysis.get_globals_read("nonexistent") == set()
        assert analysis.get_validation_variables("nonexistent") == set()

    def test_markdown_cells_ignored(self):
        """Test that markdown cells are ignored."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell(id="md1", source="# Header"),
            nbformat.v4.new_code_cell(id="c1", source="x = 1"),
            nbformat.v4.new_markdown_cell(id="md2", source="More text"),
        ]

        analysis = NotebookAnalysis(nb)

        # Only code cells should be analyzed
        assert len(analysis.get_all_cell_ids()) == 1
        assert "c1" in analysis.get_all_cell_ids()
        assert "md1" not in analysis.get_all_cell_ids()

    def test_syntax_error_cell(self):
        """Test handling of cells with syntax errors."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="x = "),  # Syntax error
            nbformat.v4.new_code_cell(id="c2", source="y = 1"),
        ]

        # Should not crash
        analysis = NotebookAnalysis(nb)

        # c1 should exist but have no dependencies (can't parse)
        deps = analysis.get_dependencies("c1")
        assert deps is not None
        assert len(deps.globals_written) == 0
        assert len(deps.globals_read) == 0


class TestBackwardCompatibility:
    """Test backward compatibility methods."""

    def test_to_dependencies_dict(self):
        """Test to_dependencies_dict method."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="x = 1"),
            nbformat.v4.new_code_cell(id="c2", source="y = x + 1"),
        ]

        analysis = NotebookAnalysis(nb)
        deps_dict = analysis.to_dependencies_dict()

        assert isinstance(deps_dict, dict)
        assert "c1" in deps_dict
        assert "c2" in deps_dict
        assert deps_dict["c1"].cell_id == "c1"

    def test_to_liveness_dict(self):
        """Test to_liveness_dict method."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="x = 1"),
            nbformat.v4.new_code_cell(id="c2", source="y = x + 1"),
        ]

        analysis = NotebookAnalysis(nb)
        liveness_dict = analysis.to_liveness_dict()

        assert isinstance(liveness_dict, dict)
        assert "c1" in liveness_dict
        assert "c2" in liveness_dict
        assert liveness_dict["c1"].cell_id == "c1"


class TestSummary:
    """Test summary method."""

    def test_get_summary(self):
        """Test get_summary method."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell(id="c1", source="x = 1\ntemp = 2"),
            nbformat.v4.new_code_cell(id="c2", source="y = x + 1"),
        ]

        analysis = NotebookAnalysis(nb)
        summary = analysis.get_summary("c1")

        assert summary["cell_id"] == "c1"
        assert "x" in summary["writes"]
        assert "temp" in summary["writes"]
        assert len(summary["reads"]) == 0
        assert "x" in summary["live_out"]  # x is used by c2
        assert "x" in summary["validation_vars"]  # x is live

    def test_get_summary_nonexistent_cell(self):
        """Test get_summary for nonexistent cell."""
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(id="c1", source="x = 1")]

        analysis = NotebookAnalysis(nb)
        summary = analysis.get_summary("nonexistent")

        assert "error" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
