"""
Tests for the PrepareCodeForFerret command.
"""

import pytest
from data_ferret.server.commands.prepare_code import PrepareCodeForFerretCommand


@pytest.fixture
def command():
    """Create a PrepareCodeForFerretCommand instance."""
    return PrepareCodeForFerretCommand()


@pytest.fixture
def base_notebook():
    """Create a base notebook structure."""
    return {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


@pytest.mark.asyncio
async def test_simple_column_fillna(command, base_notebook):
    """Test simple column access with fillna."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'].fillna(0, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["status"] == "success"
    assert result.metadata["transformations"]["cells_modified"] == 1
    assert result.metadata["transformations"]["total_transformations"] == 1

    transformed_source = result.notebook["cells"][0]["source"]
    assert "df['col'] = df['col'].fillna(0)" in transformed_source
    assert "inplace" not in transformed_source


@pytest.mark.asyncio
async def test_simple_column_replace(command, base_notebook):
    """Test simple column access with replace."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['column_name'].replace(old, new, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 1
    transformed_source = result.notebook["cells"][0]["source"]
    assert "df['column_name'] = df['column_name'].replace(old, new)" in transformed_source


@pytest.mark.asyncio
async def test_boolean_mask(command, base_notebook):
    """Test with boolean mask subscript."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df[df['x'] > 5]['y'].fillna(0, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 1
    transformed_source = result.notebook["cells"][0]["source"]
    assert "df[df['x'] > 5]['y'] = df[df['x'] > 5]['y'].fillna(0)" in transformed_source


@pytest.mark.asyncio
async def test_nested_subscripts(command, base_notebook):
    """Test with nested subscript access."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df[mask]['col'].sort_values(inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 1
    transformed_source = result.notebook["cells"][0]["source"]
    assert "df[mask]['col'] = df[mask]['col'].sort_values()" in transformed_source


@pytest.mark.asyncio
async def test_multiple_transformations_per_cell(command, base_notebook):
    """Test multiple chained assignments in one cell."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": """df['a'].fillna(0, inplace=True)
df['b'].replace(old, new, inplace=True)
df['c'].drop_duplicates(inplace=True)""",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 1
    assert result.metadata["transformations"]["total_transformations"] == 3

    transformed_source = result.notebook["cells"][0]["source"]
    assert "df['a'] = df['a'].fillna(0)" in transformed_source
    assert "df['b'] = df['b'].replace(old, new)" in transformed_source
    assert "df['c'] = df['c'].drop_duplicates()" in transformed_source


@pytest.mark.asyncio
async def test_no_transformation_direct_method(command, base_notebook):
    """Test that direct DataFrame methods are NOT transformed."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df.fillna(0, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    # Should NOT be transformed (no subscript)
    assert result.metadata["transformations"]["cells_modified"] == 0
    assert result.metadata["transformations"]["total_transformations"] == 0

    # Source should remain unchanged
    assert result.notebook["cells"][0]["source"] == "df.fillna(0, inplace=True)"


@pytest.mark.asyncio
async def test_no_transformation_no_inplace(command, base_notebook):
    """Test that methods without inplace are NOT transformed."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'].fillna(0)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 0
    assert result.notebook["cells"][0]["source"] == "df['col'].fillna(0)"


@pytest.mark.asyncio
async def test_no_transformation_inplace_false(command, base_notebook):
    """Test that methods with inplace=False are NOT transformed."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'].fillna(0, inplace=False)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 0
    assert result.notebook["cells"][0]["source"] == "df['col'].fillna(0, inplace=False)"


@pytest.mark.asyncio
async def test_preserve_comments(command, base_notebook):
    """Test that comments are preserved."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'].fillna(0, inplace=True)  # Fill missing values",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    transformed_source = result.notebook["cells"][0]["source"]
    assert "# Fill missing values" in transformed_source


@pytest.mark.asyncio
async def test_preserve_multiline_formatting(command, base_notebook):
    """Test that multiline formatting is preserved."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": """df['column_name'].fillna(
    0,
    inplace=True
)""",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    transformed_source = result.notebook["cells"][0]["source"]
    # Should still be multiline (LibCST preserves formatting)
    assert "\n" in transformed_source
    assert "df['column_name'] = df['column_name'].fillna" in transformed_source


@pytest.mark.asyncio
async def test_skip_markdown_cells(command, base_notebook):
    """Test that markdown cells are skipped."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "markdown",
            "source": "# This is markdown with df['col'].fillna(0, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 0
    # Markdown cell should remain unchanged
    assert "# This is markdown" in result.notebook["cells"][0]["source"]


@pytest.mark.asyncio
async def test_skip_empty_cells(command, base_notebook):
    """Test that empty cells are skipped."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "",
            "metadata": {},
        },
        {
            "id": "test2",
            "cell_type": "code",
            "source": "   ",
            "metadata": {},
        },
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 0


@pytest.mark.asyncio
async def test_handle_syntax_errors_gracefully(command, base_notebook):
    """Test that syntax errors are handled gracefully."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'.fillna(0, inplace=True)",  # Syntax error: missing ]
            "metadata": {},
        },
        {
            "id": "test2",
            "cell_type": "code",
            "source": "df['valid'].fillna(0, inplace=True)",  # Valid
            "metadata": {},
        },
    ]

    result = await command.process(base_notebook)

    # Should process valid cells and skip invalid ones
    assert result.metadata["status"] == "success"
    assert result.metadata["transformations"]["cells_modified"] == 1

    # First cell should remain unchanged (syntax error)
    assert result.notebook["cells"][0]["source"] == "df['col'.fillna(0, inplace=True)"

    # Second cell should be transformed
    assert "df['valid'] = df['valid'].fillna(0)" in result.notebook["cells"][1]["source"]


@pytest.mark.asyncio
async def test_selected_cells_only(command, base_notebook):
    """Test that only selected cells are transformed."""
    base_notebook["cells"] = [
        {
            "id": "cell1",
            "cell_type": "code",
            "source": "df['a'].fillna(0, inplace=True)",
            "metadata": {},
        },
        {
            "id": "cell2",
            "cell_type": "code",
            "source": "df['b'].fillna(0, inplace=True)",
            "metadata": {},
        },
        {
            "id": "cell3",
            "cell_type": "code",
            "source": "df['c'].fillna(0, inplace=True)",
            "metadata": {},
        },
    ]

    # Only process cell2
    result = await command.process(base_notebook, selected_cell_ids=["cell2"])

    assert result.metadata["transformations"]["cells_modified"] == 1

    # cell1 should remain unchanged
    assert result.notebook["cells"][0]["source"] == "df['a'].fillna(0, inplace=True)"

    # cell2 should be transformed
    assert "df['b'] = df['b'].fillna(0)" in result.notebook["cells"][1]["source"]

    # cell3 should remain unchanged
    assert result.notebook["cells"][2]["source"] == "df['c'].fillna(0, inplace=True)"


@pytest.mark.asyncio
async def test_entire_notebook_when_no_selection(command, base_notebook):
    """Test that all cells are processed when no selection is provided."""
    base_notebook["cells"] = [
        {
            "id": "cell1",
            "cell_type": "code",
            "source": "df['a'].fillna(0, inplace=True)",
            "metadata": {},
        },
        {
            "id": "cell2",
            "cell_type": "code",
            "source": "df['b'].fillna(0, inplace=True)",
            "metadata": {},
        },
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 2
    assert result.metadata["transformations"]["total_transformations"] == 2


@pytest.mark.asyncio
async def test_command_properties(command):
    """Test command properties."""
    assert command.command_name == "prepare_code"
    assert command.display_name == "Prepare Code for Ferret"
    assert command.icon_name == "ui-components:code"
    assert command.requires_kernel is False


@pytest.mark.asyncio
async def test_metadata_structure(command, base_notebook):
    """Test that metadata has the expected structure."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'].fillna(0, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    # Check metadata structure
    assert "status" in result.metadata
    assert "command" in result.metadata
    assert "transformations" in result.metadata
    assert "message" in result.metadata

    transformations = result.metadata["transformations"]
    assert "cells_modified" in transformations
    assert "total_transformations" in transformations
    assert "summary" in transformations

    # Check summary structure
    assert len(transformations["summary"]) == 1
    assert "cell_id" in transformations["summary"][0]
    assert "transformations" in transformations["summary"][0]


@pytest.mark.asyncio
async def test_timing_recorded(command, base_notebook):
    """Test that execution time is recorded."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": "df['col'].fillna(0, inplace=True)",
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.total_time >= 0
    assert result.total_cost == 0.0


@pytest.mark.asyncio
async def test_list_source_format(command, base_notebook):
    """Test that list-format sources are handled correctly."""
    base_notebook["cells"] = [
        {
            "id": "test1",
            "cell_type": "code",
            "source": ["df['col'].fillna(0, inplace=True)\n", "df['x'].replace(1, 2, inplace=True)"],
            "metadata": {},
        }
    ]

    result = await command.process(base_notebook)

    assert result.metadata["transformations"]["cells_modified"] == 1
    assert result.metadata["transformations"]["total_transformations"] == 2

    transformed_source = result.notebook["cells"][0]["source"]
    assert "df['col'] = df['col'].fillna(0)" in transformed_source
    assert "df['x'] = df['x'].replace(1, 2)" in transformed_source
