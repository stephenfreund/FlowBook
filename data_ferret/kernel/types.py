"""
Type definitions for structured diff results.

This module defines the types used to represent differences between Python objects
in a structured, tree-like format with typed path components.

Key Functions:
    - serialize_diff_result(): Convert DiffResult to JSON-compatible dict
    - format_diff_as_markdown(): Convert DiffResult to human-readable markdown list
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator


class PathComponent(ABC):
    """Abstract base class for path components in diff results."""

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of this path component."""
        pass


class RootComponent(PathComponent):
    """Root path component representing a variable name."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class IndexComponent(PathComponent):
    """Path component for list/tuple/array indexing."""

    def __init__(self, index: int):
        self.index = index

    def __str__(self) -> str:
        return f"[{self.index}]"


class KeyComponent(PathComponent):
    """Path component for dictionary key access."""

    def __init__(self, key: str):
        self.key = key

    def __str__(self) -> str:
        # Use repr to properly escape quotes
        return f"[{repr(self.key)}]"


class AttributeComponent(PathComponent):
    """Path component for object attribute access."""

    def __init__(self, attr: str):
        self.attr = attr

    def __str__(self) -> str:
        return f".{self.attr}"


class DataFrameLocation(PathComponent):
    """Path component for pandas DataFrame cell location."""

    def __init__(self, row: Any, col: Any):
        self.row = row
        self.col = col

    def __str__(self) -> str:
        return f"[{repr(self.row)}, {repr(self.col)}]"


class ValueComparison(BaseModel):
    """
    Represents a comparison result between two values.

    Attributes:
        status: "different" if values are not equal, "close" if within tolerance (floats)
        value1: The first value being compared
        value2: The second value being compared
        message: Human-readable description of the difference
    """
    status: Literal["different", "close"] = Field(..., description="Type of difference")
    value1: Any = Field(..., description="First value")
    value2: Any = Field(..., description="Second value")
    message: str = Field(..., description="Description of the difference")

    @property
    def is_close(self) -> bool:
        """Return True if the values are close (within tolerance)."""
        return self.status == "close"

    class Config:
        arbitrary_types_allowed = True


# Type alias for the tree structure of differences
# A DiffNode is either:
# - A ValueComparison (leaf node - actual difference)
# - A Dict mapping path strings to nested DiffNodes (compound structure)
DiffNode = Union[ValueComparison, Dict[str, "DiffNode"]]


class DiffResult(BaseModel):
    """
    Result of a namespace diff operation.

    This is a Pydantic model that wraps a dictionary mapping variable names
    to their diff trees (DiffNode objects). It provides full serialization/
    deserialization capabilities.

    Attributes:
        differences: Dictionary mapping variable names to DiffNode trees

    Example:
        >>> result = DiffResult(differences={
        ...     'x': ValueComparison(status='different', value1=1, value2=2, message='...'),
        ...     'data': {'[0]': ValueComparison(...)}
        ... })
        >>> json_str = result.model_dump_json()
        >>> restored = DiffResult.model_validate_json(json_str)
    """
    differences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary mapping variable names to diff trees (DiffNode)"
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator('differences', mode='before')
    @classmethod
    def convert_dicts_to_comparisons(cls, v):
        """Convert nested dicts to ValueComparison objects during deserialization."""
        if not isinstance(v, dict):
            return v

        def convert_node(node):
            """Recursively convert dicts to ValueComparison where appropriate."""
            if isinstance(node, ValueComparison):
                # Already a ValueComparison
                return node
            elif isinstance(node, dict):
                # Check if this is a ValueComparison dict (has status, message fields)
                if 'status' in node and 'message' in node and 'value1' in node and 'value2' in node:
                    # Convert to ValueComparison
                    return ValueComparison(**node)
                else:
                    # It's a nested diff dict - recurse
                    return {key: convert_node(value) for key, value in node.items()}
            else:
                # Unknown type, return as is
                return node

        # Convert each variable's diff tree
        return {var: convert_node(node) for var, node in v.items()}

    def __bool__(self) -> bool:
        """Return True if there are any differences."""
        return bool(self.differences)

    def __len__(self) -> int:
        """Return the number of variables with differences."""
        return len(self.differences)

    def __contains__(self, key: str) -> bool:
        """Check if a variable has differences."""
        return key in self.differences

    def __getitem__(self, key: str) -> DiffNode:
        """Get the diff tree for a variable."""
        return self.differences[key]

    def __setitem__(self, key: str, value: DiffNode) -> None:
        """Set the diff tree for a variable."""
        self.differences[key] = value

    def __iter__(self):
        """Iterate over variable names."""
        return iter(self.differences)

    def __eq__(self, other):
        """Compare DiffResult with another DiffResult or dict."""
        if isinstance(other, DiffResult):
            return self.differences == other.differences
        elif isinstance(other, dict):
            # Allow comparison with plain dicts for backward compatibility
            return self.differences == other
        return False

    def keys(self):
        """Return variable names."""
        return self.differences.keys()

    def values(self):
        """Return diff trees."""
        return self.differences.values()

    def items(self):
        """Return (variable, diff_tree) pairs."""
        return self.differences.items()

    def get(self, key: str, default=None):
        """Get diff tree with default."""
        return self.differences.get(key, default)


def serialize_diff_result(diff_result: DiffResult) -> Dict[str, Any]:
    """
    Serialize a DiffResult to a JSON-compatible dictionary.

    Args:
        diff_result: The DiffResult to serialize

    Returns:
        JSON-compatible dict representation
    """
    def serialize_node(node: DiffNode) -> Any:
        """Recursively serialize a DiffNode."""
        if isinstance(node, ValueComparison):
            # Serialize ValueComparison to dict
            return {
                "type": "comparison",
                "status": node.status,
                "message": node.message,
                # Don't include value1/value2 to avoid serialization issues
            }
        elif isinstance(node, dict):
            # Recursively serialize nested diffs
            return {key: serialize_node(value) for key, value in node.items()}
        else:
            # Shouldn't happen, but handle gracefully
            return {"type": "unknown", "value": str(node)}

    return {var: serialize_node(node) for var, node in diff_result.items()}


def format_diff_as_markdown(diff_result: DiffResult) -> str:
    """
    Format a DiffResult as a human-readable markdown list.

    Args:
        diff_result: The DiffResult to format

    Returns:
        Markdown-formatted string with bulleted list of all differences

    Example:
        >>> result = {'x': ValueComparison(status='different', value1=1, value2=2, message='...')}
        >>> print(format_diff_as_markdown(result))
        ## Differences Found

        - **x**: Int mismatch: 1 vs 2
    """
    if not diff_result:
        return "## No Differences Found\n\nAll variables are equal."

    lines = ["## Differences Found\n"]

    def format_node(var_name: str, node: DiffNode, path: str = "") -> None:
        """Recursively format a DiffNode and append to lines."""
        if isinstance(node, ValueComparison):
            # Leaf node - actual difference
            full_path = f"{var_name}{path}" if path else var_name

            # Add status indicator for close values
            status_indicator = " *(close)*" if node.is_close else ""

            # Format the bullet point
            lines.append(f"- **{full_path}**{status_indicator}: {node.message}")

        elif isinstance(node, dict):
            # Compound structure - recurse into nested diffs
            for key, child_node in node.items():
                # Skip special truncation markers
                if key == "_truncated":
                    if isinstance(child_node, ValueComparison):
                        full_path = f"{var_name}{path}" if path else var_name
                        lines.append(f"  - *{full_path}: {child_node.message}*")
                    continue

                # Build the new path
                new_path = f"{path}{key}"
                format_node(var_name, child_node, new_path)

    # Process each variable
    for var_name in sorted(diff_result.keys()):
        format_node(var_name, diff_result[var_name])

    return "\n".join(lines)
