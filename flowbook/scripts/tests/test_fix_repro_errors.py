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
    add_model_copy_and_rename,
    convert_inplace_to_assignment,
    add_copy_before_structural_assign,
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


# =============================================================================
# Tests for New Fix Types (Unrecoverable Mutations)
# =============================================================================


class TestModelCopyFix:
    """Tests for add_model_copy_and_rename (model-copy fix type)."""

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

    def test_model_copy_basic(self):
        """Model copy should insert safe_model_copy and rename."""
        notebook = self._make_notebook([
            "model.fit(X_train, y_train)\n",
            "predictions = model.predict(X_test)\n",
        ])

        add_model_copy_and_rename(notebook, 0, "model", "abcd")

        result = get_cell_source(notebook["cells"][0])

        # Should have fix marker
        assert FLOWBOOK_FIX_MARKER in result
        # Should import safe_model_copy
        assert "from flowbook.util.model_copy import safe_model_copy" in result
        # Should create renamed variable
        assert "model_flow_abcd = safe_model_copy(model)" in result
        # Should rename model in the cell
        assert "model_flow_abcd.fit(X_train, y_train)" in result

    def test_model_copy_renames_downstream(self):
        """Model copy should rename variable in downstream cells."""
        notebook = self._make_notebook([
            "model.fit(X_train, y_train)\n",
            "predictions = model.predict(X_test)\n",
            "print(model.score(X_test, y_test))\n",
        ])

        add_model_copy_and_rename(notebook, 0, "model", "abcd")

        # Check downstream cells are renamed
        cell1_source = get_cell_source(notebook["cells"][1])
        assert "model_flow_abcd.predict" in cell1_source
        assert "model.predict" not in cell1_source

        cell2_source = get_cell_source(notebook["cells"][2])
        assert "model_flow_abcd.score" in cell2_source

    def test_model_copy_preserves_magic(self):
        """Model copy should preserve cell magics."""
        notebook = self._make_notebook([
            "%%time\nmodel.fit(X_train, y_train)\n",
        ])

        add_model_copy_and_rename(notebook, 0, "model", "abcd")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # %%time must be first
        assert lines[0] == "%%time"

    def test_model_copy_chains_with_previous_fix(self):
        """Model copy should chain from previously renamed variable."""
        notebook = self._make_notebook([
            "model_flow_1234.fit(X, y)\n",
            "model_flow_1234.predict(X)\n",
        ])

        add_model_copy_and_rename(notebook, 0, "model", "abcd")

        result = get_cell_source(notebook["cells"][0])

        # Should copy from the _flow_ variant
        assert "safe_model_copy(model_flow_1234)" in result
        # Should create new name
        assert "model_flow_abcd" in result


class TestInplaceToCopyFix:
    """Tests for convert_inplace_to_assignment (inplace-to-copy fix type)."""

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

    def test_drop_inplace_to_assignment(self):
        """df.drop(inplace=True) should become df = df.drop()."""
        notebook = self._make_notebook([
            "df.drop('col', axis=1, inplace=True)\n",
        ])

        convert_inplace_to_assignment(notebook, 0, "df")

        result = get_cell_source(notebook["cells"][0])

        # Should have fix marker
        assert FLOWBOOK_FIX_MARKER in result
        # Should be converted to assignment (the code, not the comment)
        assert "df = df.drop" in result or "df=df.drop" in result
        # The actual code line should not have inplace=True
        code_lines = [l for l in result.split("\n") if not l.startswith("#")]
        code_part = "\n".join(code_lines)
        assert "inplace=True" not in code_part

    def test_fillna_inplace_to_assignment(self):
        """df.fillna(inplace=True) should become df = df.fillna()."""
        notebook = self._make_notebook([
            "df.fillna(0, inplace=True)\n",
        ])

        convert_inplace_to_assignment(notebook, 0, "df")

        result = get_cell_source(notebook["cells"][0])

        assert FLOWBOOK_FIX_MARKER in result
        # Check the code part (not comments)
        code_lines = [l for l in result.split("\n") if not l.startswith("#")]
        code_part = "\n".join(code_lines)
        assert "inplace=True" not in code_part

    def test_reset_index_inplace(self):
        """df.reset_index(inplace=True) should become df = df.reset_index()."""
        notebook = self._make_notebook([
            "df.reset_index(drop=True, inplace=True)\n",
        ])

        convert_inplace_to_assignment(notebook, 0, "df")

        result = get_cell_source(notebook["cells"][0])

        assert FLOWBOOK_FIX_MARKER in result
        assert "drop=True" in result  # Other args preserved
        # Check the code part (not comments)
        code_lines = [l for l in result.split("\n") if not l.startswith("#")]
        code_part = "\n".join(code_lines)
        assert "inplace=True" not in code_part

    def test_multiple_inplace_ops(self):
        """Multiple inplace operations should all be converted."""
        notebook = self._make_notebook([
            "df.drop('a', inplace=True)\ndf.fillna(0, inplace=True)\n",
        ])

        convert_inplace_to_assignment(notebook, 0, "df")

        result = get_cell_source(notebook["cells"][0])

        # Check the code part (not comments) - both should be converted
        code_lines = [l for l in result.split("\n") if not l.startswith("#")]
        code_part = "\n".join(code_lines)
        assert "inplace=True" not in code_part

    def test_preserves_magic(self):
        """Inplace fix should preserve cell magics."""
        notebook = self._make_notebook([
            "%%time\ndf.drop('col', inplace=True)\n",
        ])

        convert_inplace_to_assignment(notebook, 0, "df")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # %%time must be first
        assert lines[0] == "%%time"

    def test_no_change_without_inplace(self):
        """Cell without inplace=True should get fix marker but minimal changes."""
        notebook = self._make_notebook([
            "df = df.drop('col', axis=1)\n",
        ])

        convert_inplace_to_assignment(notebook, 0, "df")

        result = get_cell_source(notebook["cells"][0])

        # The original assignment pattern should still work
        assert "df = df.drop" in result or "df=df.drop" in result


class TestStructCopyFix:
    """Tests for add_copy_before_structural_assign (struct-copy fix type)."""

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

    def test_struct_copy_columns_assignment(self):
        """Structural assignment should insert copy() before."""
        notebook = self._make_notebook([
            "df.columns = ['a', 'b', 'c']\n",
            "print(df)\n",
        ])

        add_copy_before_structural_assign(notebook, 0, "df", "abcd")

        result = get_cell_source(notebook["cells"][0])

        # Should have fix marker
        assert FLOWBOOK_FIX_MARKER in result
        # Should create copy
        assert "df_flow_abcd = df.copy()" in result
        # Should rename in structural assignment
        assert "df_flow_abcd.columns = ['a', 'b', 'c']" in result

    def test_struct_copy_renames_downstream(self):
        """Structural copy should rename variable in downstream cells."""
        notebook = self._make_notebook([
            "df.columns = ['a', 'b', 'c']\n",
            "print(df.head())\n",
        ])

        add_copy_before_structural_assign(notebook, 0, "df", "abcd")

        cell1_source = get_cell_source(notebook["cells"][1])
        assert "df_flow_abcd.head()" in cell1_source

    def test_struct_copy_preserves_magic(self):
        """Structural copy should preserve cell magics."""
        notebook = self._make_notebook([
            "%%time\ndf.columns = ['a', 'b']\n",
        ])

        add_copy_before_structural_assign(notebook, 0, "df", "abcd")

        result = get_cell_source(notebook["cells"][0])
        lines = result.split("\n")

        # %%time must be first
        assert lines[0] == "%%time"

    def test_struct_copy_chains_with_previous_fix(self):
        """Structural copy should chain from previously renamed variable."""
        notebook = self._make_notebook([
            "df_flow_1234.index = new_index\n",
            "print(df_flow_1234)\n",
        ])

        add_copy_before_structural_assign(notebook, 0, "df", "abcd")

        result = get_cell_source(notebook["cells"][0])

        # Should copy from the _flow_ variant
        assert "df_flow_abcd = df_flow_1234.copy()" in result
        # Should rename to new name
        assert "df_flow_abcd.index = new_index" in result
