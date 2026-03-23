"""
Change Detector - Converts MemoryCheckpointDiffResult to typed Change list.

This module bridges the existing MemoryCheckpointDiffResult structure (with nested dictionaries
representing diff trees) to the new typed Change hierarchy.

Usage:
    from flowbook.kernel_support.types import MemoryCheckpointDiffResult
    from flowbook.kernel_support.change_detector import detect_changes

    diff = MemoryCheckpointDiffResult(differences={'df': {'[\"price\"]': ValueComparison(...)}})
    changes = detect_changes(diff)
    # [ColumnModified(variable='df', column='price')]

Design:
    The MemoryCheckpointDiffResult contains a nested structure where:
    - Top-level keys are variable names
    - Values can be:
        - ValueComparison: Simple value change
        - CompoundDiff: Complex object with children
        - Dict: Legacy format nested structure

    Special keys in CompoundDiff children:
    - _structural_rows: Row count changed
    - _structural_columns: Columns added/removed
    - _structural_index: Index changed
    - ['col_name']: Column value or existence changed
"""

import re
from typing import Any, Dict, List, Optional

from flowbook.kernel_support.types import CompoundDiff, MemoryCheckpointDiffResult, ValueComparison

from flowbook.kernel.changes import (
    Change,
    ColumnAdded,
    ColumnModified,
    ColumnRemoved,
    DtypeChanged,
    IndexChanged,
    RowsAdded,
    RowsRemoved,
    ValueChanged,
)
from flowbook.kernel.locations import WriteLoc, WriteLocSet


def detect_changes(diff: MemoryCheckpointDiffResult) -> List[Change]:
    """
    Convert a MemoryCheckpointDiffResult to a list of typed Change objects.

    This is the main entry point for converting the diff tree into
    typed changes for conflict detection.

    Args:
        diff: MemoryCheckpointDiffResult from namespace comparison

    Returns:
        List of Change objects describing what changed
    """
    changes: List[Change] = []

    for var_name, diff_tree in diff.differences.items():
        var_changes = _analyze_diff_tree(var_name, diff_tree)
        changes.extend(var_changes)

    return changes


def _analyze_diff_tree(variable: str, node: Any) -> List[Change]:
    """
    Analyze a diff tree node and extract changes.

    Args:
        variable: The variable name this diff is for
        node: The diff node (ValueComparison, CompoundDiff, or dict)

    Returns:
        List of Change objects
    """
    if node is None:
        return []

    # ValueComparison at top level = entire value changed
    if isinstance(node, ValueComparison):
        return [ValueChanged(variable=variable)]

    # CompoundDiff = complex object with structured changes
    if isinstance(node, CompoundDiff):
        return _analyze_compound_diff(variable, node)

    # Dict = legacy format or nested structure
    if isinstance(node, dict):
        return _analyze_dict_diff(variable, node)

    # Unknown type - treat as value change
    return [ValueChanged(variable=variable)]


def _analyze_compound_diff(variable: str, diff: CompoundDiff) -> List[Change]:
    """
    Analyze a CompoundDiff (DataFrame, Series, or other complex object).

    Args:
        variable: The variable name
        diff: The CompoundDiff object

    Returns:
        List of Change objects
    """
    changes: List[Change] = []

    if diff.source_type == "dataframe":
        changes.extend(_analyze_dataframe_diff(variable, diff.children))
    elif diff.source_type == "series":
        changes.extend(_analyze_series_diff(variable, diff.children))
    else:
        # Other compound types (dict, list, etc.) - check for any changes
        for key, child in diff.children.items():
            if isinstance(child, ValueComparison):
                changes.append(ValueChanged(variable=variable))
                break
            elif isinstance(child, CompoundDiff):
                # Nested compound diff means something inside changed
                # For conflict detection, this counts as a change to the whole variable
                changes.append(ValueChanged(variable=variable))
                break

    return changes


def _analyze_dataframe_diff(variable: str, children: Dict[str, Any]) -> List[Change]:
    """
    Analyze DataFrame diff children for specific change types.

    Special keys:
    - _structural_rows: Row count changed
    - _structural_columns: Columns added
    - ['col_name']: Column change

    Args:
        variable: The variable name
        children: Dict of child diffs

    Returns:
        List of Change objects
    """
    changes: List[Change] = []
    modified_cols = set()
    added_cols = set()
    removed_cols = set()

    for key, child in children.items():
        # Structural row changes
        if key == "_structural_rows":
            row_changes = _parse_row_change(variable, child)
            changes.extend(row_changes)
            continue

        # Structural column changes (additions only tracked here)
        if key == "_structural_columns":
            col_changes = _parse_column_structural_change(variable, child)
            for ch in col_changes:
                if isinstance(ch, ColumnAdded):
                    added_cols.add(ch.column)
                changes.append(ch)  # Add to changes list!
            continue

        # Structural index changes
        if key == "_structural_index":
            changes.append(IndexChanged(variable=variable))
            continue

        # Index changes (from diff, could be row add/remove or label change)
        if key == "_index":
            index_changes = _parse_index_change(variable, child)
            changes.extend(index_changes)
            continue

        # Column key (e.g., "['price']" or "[\"price\"]")
        col_name = _extract_column_name(key)
        if col_name:
            col_change = _analyze_column_change(variable, col_name, child)
            if col_change:
                if isinstance(col_change, ColumnModified):
                    modified_cols.add(col_name)
                elif isinstance(col_change, ColumnRemoved):
                    removed_cols.add(col_name)
                elif isinstance(col_change, ColumnAdded):
                    added_cols.add(col_name)
                changes.append(col_change)

    return changes


def _analyze_series_diff(variable: str, children: Dict[str, Any]) -> List[Change]:
    """
    Analyze Series diff children for changes.

    For Series, most changes mean the whole value changed. We look for
    dtype changes specifically.

    Args:
        variable: The variable name
        children: Dict of child diffs

    Returns:
        List of Change objects
    """
    changes: List[Change] = []

    for key, child in children.items():
        if key == "_dtype":
            # Dtype changed
            if isinstance(child, ValueComparison) and child.status == "different":
                old_dtype = str(child.value1) if child.value1 else "unknown"
                new_dtype = str(child.value2) if child.value2 else "unknown"
                # For Series, we use the variable name as the "column"
                changes.append(
                    DtypeChanged(
                        variable=variable,
                        column=variable,  # Series name is the variable
                        old_dtype=old_dtype,
                        new_dtype=new_dtype,
                    )
                )
        elif isinstance(child, ValueComparison) and child.status == "different":
            # Value changed in the series
            changes.append(ValueChanged(variable=variable))
            break  # One ValueChanged is enough

    return changes


def _analyze_dict_diff(variable: str, diff: Dict[str, Any]) -> List[Change]:
    """
    Analyze a raw dict diff (legacy format).

    Args:
        variable: The variable name
        diff: The diff dictionary

    Returns:
        List of Change objects
    """
    changes: List[Change] = []

    for key, child in diff.items():
        # Check for structural keys
        if key == "_structural_rows":
            row_changes = _parse_row_change(variable, child)
            changes.extend(row_changes)
        elif key == "_structural_columns":
            col_changes = _parse_column_structural_change(variable, child)
            changes.extend(col_changes)
        elif key == "_structural_index":
            changes.append(IndexChanged(variable=variable))
        else:
            # Column or value change
            col_name = _extract_column_name(key)
            if col_name:
                col_change = _analyze_column_change(variable, col_name, child)
                if col_change:
                    changes.append(col_change)
            elif isinstance(child, ValueComparison) and child.status == "different":
                changes.append(ValueChanged(variable=variable))

    return changes


def _parse_row_change(variable: str, node: Any) -> List[Change]:
    """
    Parse a _structural_rows node to determine RowsAdded or RowsRemoved.

    Args:
        variable: The variable name
        node: The structural rows diff node

    Returns:
        List containing RowsAdded or RowsRemoved, or empty if unparseable
    """
    if not isinstance(node, ValueComparison):
        return []

    msg = node.message or ""

    # Try to parse "Row count changed from X to Y"
    match = re.search(r"from (\d+) to (\d+)", msg)
    if match:
        old_count = int(match.group(1))
        new_count = int(match.group(2))
        diff = new_count - old_count
        if diff > 0:
            return [RowsAdded(variable=variable, count=diff)]
        elif diff < 0:
            return [RowsRemoved(variable=variable, count=abs(diff))]

    # If we can't parse, assume some rows changed
    return [RowsAdded(variable=variable, count=1)]


def _parse_index_change(variable: str, node: Any) -> List[Change]:
    """
    Parse an _index node to determine row or index changes.

    The _index key in DataFrame diff indicates the index changed.
    This can mean:
    - Rows added (index grew)
    - Rows removed (index shrank)
    - Index labels changed (same length, different values)

    Args:
        variable: The variable name
        node: The index diff node (typically ValueComparison)

    Returns:
        List of RowsAdded, RowsRemoved, or IndexChanged
    """
    if not isinstance(node, ValueComparison):
        return []

    # Try to extract length from RangeIndex or similar
    # value1 = old index, value2 = new index
    old_len = _get_index_length(node.value1)
    new_len = _get_index_length(node.value2)

    if old_len is not None and new_len is not None:
        diff = new_len - old_len
        if diff > 0:
            return [RowsAdded(variable=variable, count=diff)]
        elif diff < 0:
            return [RowsRemoved(variable=variable, count=abs(diff))]

    # If lengths are equal or we can't determine, it's just an index label change
    return [IndexChanged(variable=variable)]


def _get_index_length(index_obj: Any) -> Optional[int]:
    """
    Try to extract the length from an index object.

    Args:
        index_obj: A pandas Index, RangeIndex, or similar

    Returns:
        The length, or None if can't determine
    """
    if index_obj is None:
        return None

    # Try direct len()
    try:
        return len(index_obj)
    except (TypeError, AttributeError):
        pass

    # Try parsing RangeIndex string repr: "RangeIndex(start=0, stop=3, step=1)"
    if hasattr(index_obj, "__str__"):
        s = str(index_obj)
        import re

        match = re.search(r"stop=(\d+)", s)
        if match:
            return int(match.group(1))

    return None


def _parse_column_structural_change(variable: str, node: Any) -> List[Change]:
    """
    Parse a _structural_columns node to determine ColumnAdded changes.

    Args:
        variable: The variable name
        node: The structural columns diff node

    Returns:
        List of ColumnAdded changes
    """
    if not isinstance(node, ValueComparison):
        return []

    msg = node.message or ""

    # Try to parse "Columns added: ['col1', 'col2']"
    match = re.search(r"Columns added: \[([^\]]+)\]", msg)
    if match:
        cols_str = match.group(1)
        # Parse column names from the string representation
        col_names = re.findall(r"'([^']+)'", cols_str)
        return [ColumnAdded(variable=variable, column=col) for col in col_names]

    return []


def _extract_column_name(key: str) -> Optional[str]:
    """
    Extract column name from a diff key like "['price']" or '["price"]'.

    Args:
        key: The diff key

    Returns:
        Column name, or None if not a column key
    """
    # Match ['col'] or ["col"]
    match = re.match(r"\[[\'\"](.+)[\'\"]\]", key)
    if match:
        return match.group(1)
    return None


def _analyze_column_change(variable: str, column: str, node: Any) -> Optional[Change]:
    """
    Analyze a column diff node to determine the type of change.

    Args:
        variable: The variable name
        column: The column name
        node: The column diff node

    Returns:
        The appropriate Change type, or None
    """
    if isinstance(node, ValueComparison):
        if node.status != "different":
            return None

        # Check the message for hints about the change type
        msg = node.message or ""

        if "only in first" in msg.lower() or "missing in second" in msg.lower():
            return ColumnRemoved(variable=variable, column=column)
        elif "only in second" in msg.lower() or "missing in first" in msg.lower():
            return ColumnAdded(variable=variable, column=column)
        elif "missing in pre-state" in msg.lower() or "missing in pre" in msg.lower():
            # New column added - wasn't in pre-checkpoint
            return ColumnAdded(variable=variable, column=column)
        elif "missing in post-state" in msg.lower() or "missing in post" in msg.lower():
            # Column removed - was in pre but not in post
            return ColumnRemoved(variable=variable, column=column)
        elif "deleted in post" in msg.lower():
            return ColumnRemoved(variable=variable, column=column)
        else:
            # Default: column values changed
            return ColumnModified(variable=variable, column=column)

    elif isinstance(node, CompoundDiff):
        # Column has structured diff - means values changed
        return ColumnModified(variable=variable, column=column)

    elif isinstance(node, dict):
        # Nested dict - column changed
        return ColumnModified(variable=variable, column=column)

    return None


def changes_to_write_locs(changes: List[Change]) -> WriteLocSet:
    """Convert typed Change objects to WriteLocSet."""
    locs = set()
    for change in changes:
        if isinstance(change, ValueChanged):
            locs.add(WriteLoc.var(change.variable))
        elif isinstance(change, ColumnModified):
            locs.add(WriteLoc.col(change.variable, change.column))
        elif isinstance(change, ColumnAdded):
            locs.add(WriteLoc.col_add(change.variable, change.column))
        elif isinstance(change, ColumnRemoved):
            locs.add(WriteLoc.col_del(change.variable, change.column))
        elif isinstance(change, RowsAdded):
            locs.add(WriteLoc.rows(change.variable))
        elif isinstance(change, RowsRemoved):
            locs.add(WriteLoc.rows(change.variable))
        elif isinstance(change, IndexChanged):
            locs.add(WriteLoc.attr(change.variable, "index"))
        elif isinstance(change, DtypeChanged):
            locs.add(WriteLoc.col(change.variable, change.column))
            locs.add(WriteLoc.attr(change.variable, "dtypes"))
    return frozenset(locs)


def detect_write_locs(diff: MemoryCheckpointDiffResult) -> WriteLocSet:
    """Convert diff result to WriteLocSet."""
    changes = detect_changes(diff)
    return changes_to_write_locs(changes)


def get_changed_variables(diff: MemoryCheckpointDiffResult) -> set:
    """
    Get all variables that have any changes.

    This is a quick check without parsing the full change types.

    Args:
        diff: MemoryCheckpointDiffResult from namespace comparison

    Returns:
        Set of variable names that changed
    """
    return set(diff.differences.keys())


def has_any_changes(diff: MemoryCheckpointDiffResult) -> bool:
    """
    Check if the diff contains any changes.

    Args:
        diff: MemoryCheckpointDiffResult from namespace comparison

    Returns:
        True if there are any changes
    """
    return bool(diff.differences)
