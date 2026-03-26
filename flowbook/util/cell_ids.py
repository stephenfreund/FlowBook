"""
Utilities for generating and managing notebook cell IDs.

This module provides functions to:
- Generate unique 4-character cell IDs
- Validate cell IDs (4 lowercase alphanumeric characters)
- Normalize notebooks by ensuring all cells have unique valid IDs
"""

import random
import string
from typing import Dict, Any, Set


def generate_cell_id(existing_ids: Set[str]) -> str:
    """
    Generate a unique 4-character cell ID.

    Uses lowercase letters a-z (26^4 = 456,976 possible IDs).

    Args:
        existing_ids: Set of IDs already in use

    Returns:
        A unique 4-character ID not in existing_ids
    """
    while True:
        # Generate random 4-character ID (letters only for new IDs)
        cell_id = ''.join(random.choices(string.ascii_lowercase, k=4))
        if cell_id not in existing_ids:
            return cell_id


def is_valid_cell_id(cell_id: str) -> bool:
    """
    Check if a cell ID is valid (4 lowercase alphanumeric characters).

    Args:
        cell_id: The cell ID to validate

    Returns:
        True if valid, False otherwise
    """
    if len(cell_id) != 4:
        return False
    for c in cell_id:
        if not (c.isdigit() or (c.isalpha() and c.islower())):
            return False
    return True


def normalize_notebook(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a notebook by ensuring all cells have unique 4-character IDs.

    This function:
    1. Adds IDs to cells that don't have them
    2. Replaces non-4-character IDs with new 4-character IDs
    3. Ensures all IDs are unique (regenerates duplicates)
    4. Converts source from list to string format
    5. Does NOT modify the input notebook (creates a copy if changes needed)

    Args:
        notebook: Notebook JSON as dict

    Returns:
        Normalized notebook (same object if no changes needed, or new dict)
    """
    # Track whether we need to make changes
    needs_changes = False

    # Collect existing valid IDs and check for duplicates
    existing_ids: Set[str] = set()
    duplicate_ids: Set[str] = set()

    for cell in notebook.get("cells", []):
        cell_id = cell.get("id")
        if cell_id is None:
            needs_changes = True
        elif not is_valid_cell_id(str(cell_id)):
            # Invalid ID format - needs replacement
            needs_changes = True
        elif cell_id in existing_ids:
            duplicate_ids.add(cell_id)
            needs_changes = True
        else:
            existing_ids.add(cell_id)

    # Check if any sources need conversion
    for cell in notebook.get("cells", []):
        if isinstance(cell.get("source"), list):
            needs_changes = True
            break

    # If no changes needed, return original
    if not needs_changes:
        return notebook

    # Make a shallow copy of notebook and deep copy of cells
    import copy
    normalized = {**notebook}
    normalized["cells"] = copy.deepcopy(notebook["cells"])

    # Reset tracking for second pass
    existing_ids = set()

    # Process each cell
    for cell in normalized["cells"]:
        # Handle ID
        cell_id = cell.get("id")

        # Replace if:
        # - No ID
        # - Invalid ID format
        # - Duplicate ID
        # - Already seen in this pass
        needs_new_id = (
            cell_id is None or
            not is_valid_cell_id(str(cell_id)) or
            cell_id in duplicate_ids or
            cell_id in existing_ids
        )

        if needs_new_id:
            # Generate new unique ID
            new_id = generate_cell_id(existing_ids)
            cell["id"] = new_id
            existing_ids.add(new_id)
        else:
            existing_ids.add(cell_id)

        # Convert source from list to string if needed
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])

    return normalized


def _int_to_alpha(n: int) -> str:
    """Convert 0-based index to Excel-style alpha label (A, B, ..., Z, AA, AB, ...).

    0 → A, 25 → Z, 26 → AA, 27 → AB, ...
    """
    if n < 26:
        return chr(ord('A') + n)
    # Two letters
    n -= 26
    return chr(ord('A') + n // 26) + chr(ord('A') + n % 26)


def normalize_notebook_alpha(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a notebook with alphabetic cell IDs for MCP use.

    Code cells get sequential alpha IDs: A, B, C, ... Z, AA, AB, ...
    Markdown/raw cells get prefixed IDs: mA, mB, mC, ...

    Also converts source from list to string format.

    Args:
        notebook: Notebook JSON as dict

    Returns:
        Normalized notebook (new copy with alpha IDs)
    """
    import copy as copy_mod

    normalized = {**notebook}
    normalized["cells"] = copy_mod.deepcopy(notebook["cells"])

    code_idx = 0
    md_idx = 0

    for cell in normalized["cells"]:
        if cell.get("cell_type") == "code":
            cell["id"] = _int_to_alpha(code_idx)
            code_idx += 1
        else:
            cell["id"] = "m" + _int_to_alpha(md_idx)
            md_idx += 1

        # Convert source from list to string if needed
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])

    return normalized


def next_insertion_id(before_id: str, existing_ids: Set[str]) -> str:
    """Generate an insertion ID between two cells.

    Given cell ID "B", produces "B1", "B2", "B3", ... choosing the first
    that doesn't conflict with existing IDs.

    Args:
        before_id: The cell ID that the new cell follows.
        existing_ids: Set of IDs already in use.

    Returns:
        A new unique ID like "B1", "B2", etc.
    """
    for n in range(1, 100):
        candidate = f"{before_id}{n}"
        if candidate not in existing_ids:
            return candidate
    raise ValueError(f"Too many insertions after {before_id}")
