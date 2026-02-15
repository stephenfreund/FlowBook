"""
Utilities to convert Jupyter notebooks to annotated Python files and back.

The annotated format uses cell delimiters to preserve cell boundaries:
    # ====== CELL [cell_id] (code|markdown) ======
    <source>
"""

from __future__ import annotations

import re
from typing import Dict, List, Any


# Pattern for cell delimiter lines
CELL_DELIMITER_PATTERN = re.compile(
    r'^# ====== CELL \[([a-z]+)\] \((code|markdown)\) ======\s*$'
)


def notebook_to_python(nb: Dict[str, Any]) -> str:
    """
    Convert a notebook to annotated Python with cell delimiters.

    Args:
        nb: Notebook content as a dictionary (parsed JSON)

    Returns:
        Python source with cell delimiters like:
            # ====== CELL [abcd] (code) ======
            x = df['price'].sum()

            # ====== CELL [efgh] (markdown) ======
            # This is a markdown cell
    """
    lines: List[str] = []

    for cell in nb.get("cells", []):
        cell_id = cell.get("id", "xxxx")
        cell_type = cell.get("cell_type", "code")

        # Add cell delimiter
        lines.append(f"# ====== CELL [{cell_id}] ({cell_type}) ======")

        # Get source (can be string or list of strings)
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        # For markdown cells, prefix each line with # to make valid Python
        if cell_type == "markdown":
            for line in source.split("\n"):
                lines.append(f"# {line}")
        else:
            # Code cells - add source as-is
            lines.append(source)

        # Add blank line between cells for readability
        lines.append("")

    return "\n".join(lines)


def python_to_notebook_cells(python_source: str) -> List[Dict[str, Any]]:
    """
    Parse annotated Python back to notebook cell dicts.

    Args:
        python_source: Python source with cell delimiters

    Returns:
        List of cell dictionaries with 'id', 'cell_type', and 'source' keys.
        Each cell dict is suitable for replacing in a notebook's cells array.
    """
    cells: List[Dict[str, Any]] = []
    current_cell: Dict[str, Any] | None = None
    current_lines: List[str] = []

    def flush_cell():
        nonlocal current_cell, current_lines
        if current_cell is not None:
            # Join lines and strip trailing whitespace
            source = "\n".join(current_lines).rstrip()

            # For markdown cells, remove the # prefix we added
            if current_cell["cell_type"] == "markdown":
                markdown_lines = []
                for line in source.split("\n"):
                    # Remove leading "# " or "#" prefix
                    if line.startswith("# "):
                        markdown_lines.append(line[2:])
                    elif line.startswith("#"):
                        markdown_lines.append(line[1:])
                    else:
                        markdown_lines.append(line)
                source = "\n".join(markdown_lines)

            current_cell["source"] = source
            cells.append(current_cell)
            current_cell = None
            current_lines = []

    for line in python_source.split("\n"):
        match = CELL_DELIMITER_PATTERN.match(line)
        if match:
            # Found a new cell delimiter
            flush_cell()
            cell_id = match.group(1)
            cell_type = match.group(2)
            current_cell = {
                "id": cell_id,
                "cell_type": cell_type,
            }
        elif current_cell is not None:
            # Accumulate lines for current cell
            current_lines.append(line)

    # Flush the last cell
    flush_cell()

    return cells


def apply_cell_updates(
    notebook: Dict[str, Any], updated_cells: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply updated cells to a notebook, preserving other cell properties.

    Args:
        notebook: Original notebook content
        updated_cells: List of cells with 'id', 'cell_type', 'source' keys

    Returns:
        Modified notebook with updated cell sources.
        Cells in updated_cells but not in notebook are ignored.
        Cells in notebook but not in updated_cells are preserved.
    """
    # Build a map of updated cells by ID
    updates_by_id = {cell["id"]: cell for cell in updated_cells}

    # Create new cells list
    new_cells = []
    for cell in notebook.get("cells", []):
        cell_id = cell.get("id")
        if cell_id in updates_by_id:
            update = updates_by_id[cell_id]
            # Create updated cell preserving original properties
            new_cell = dict(cell)
            new_cell["source"] = update["source"]
            new_cell["cell_type"] = update["cell_type"]
            new_cells.append(new_cell)
        else:
            new_cells.append(cell)

    # Create new notebook with updated cells
    result = dict(notebook)
    result["cells"] = new_cells
    return result
