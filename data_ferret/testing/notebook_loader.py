"""
Notebook loader - Parse notebooks and extract executable cells.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Cell:
    """Represents a notebook cell."""

    cell_id: str
    source: str
    cell_type: str  # 'code' or 'markdown'
    index: int  # Position in notebook


def load_notebook(path: str) -> List[Cell]:
    """
    Load a notebook and extract executable cells.

    Args:
        path: Path to .ipynb file

    Returns:
        List of Cell objects (code cells only, in order)
    """
    notebook_path = Path(path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = []
    for i, cell_data in enumerate(nb.get("cells", [])):
        cell_type = cell_data.get("cell_type", "")
        if cell_type != "code":
            continue

        # Get cell ID from metadata
        cell_id = cell_data.get("id", "")
        if not cell_id:
            cell_id = cell_data.get("metadata", {}).get("id", f"cell_{i}")

        # Handle source as list or string
        source = cell_data.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        cells.append(
            Cell(
                cell_id=cell_id,
                source=source,
                cell_type=cell_type,
                index=i,
            )
        )

    return cells
