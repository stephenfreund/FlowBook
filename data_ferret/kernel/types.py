"""
Type definitions for structured diff results.

This module defines the types used to represent differences between Python objects
in a structured, tree-like format with typed path components.

Key Functions:
    - serialize_diff_result(): Convert DiffResult to JSON-compatible dict
    - format_diff_as_markdown(): Convert DiffResult to human-readable markdown list
"""

from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, Literal, Optional, Union, ForwardRef
from pydantic import BaseModel, Field, field_validator
import traceback
import math


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

    def model_post_init(self, __context: Any) -> None:
        """
        Defensive check: detect if floats are incorrectly marked as different
        when they're actually within tolerance.

        This helps diagnose bugs where tolerance parameters aren't being
        properly applied in the comparison logic.
        """
        # Only check if status is "different" (not "close")
        if self.status != "different":
            return

        # Check if both values are float-like
        try:
            import numpy as np
            is_float1 = isinstance(self.value1, (float, np.floating))
            is_float2 = isinstance(self.value2, (float, np.floating))
        except ImportError:
            is_float1 = isinstance(self.value1, float)
            is_float2 = isinstance(self.value2, float)

        if not (is_float1 and is_float2):
            return

        # Skip NaN values (they're correctly marked as different)
        try:
            if math.isnan(self.value1) or math.isnan(self.value2):
                return
        except (TypeError, ValueError):
            return

        # Check if values are within defensive tolerance (1e-5 for both atol and rtol)
        atol = 1e-5
        rtol = 1e-5

        try:
            diff = abs(self.value1 - self.value2)
            threshold = atol + rtol * abs(self.value2)

            if diff <= threshold:
                # Values are within tolerance but marked as different!
                # This indicates a bug in the comparison logic
                stack = traceback.format_stack()
                stack_str = ''.join(stack)

                # Append stack trace to message for debugging
                self.message += (
                    f"\n\n**TOLERANCE BUG DETECTED**\n"
                    f"Values marked as 'different' but within tolerance:\n"
                    f"  value1: {self.value1}\n"
                    f"  value2: {self.value2}\n"
                    f"  difference: {diff:.2e}\n"
                    f"  threshold: {threshold:.2e}\n"
                    f"  atol: {atol}, rtol: {rtol}\n\n"
                    f"Stack trace:\n{stack_str}"
                )
        except Exception as e:
            # Don't fail if the check itself has an error
            pass

    @property
    def is_close(self) -> bool:
        """Return True if the values are close (within tolerance)."""
        return self.status == "close"

    class Config:
        arbitrary_types_allowed = True


# Source types for compound diffs - helps identify what kind of structure the diff came from
DiffSourceType = Literal[
    "array",      # numpy ndarray
    "list",       # Python list
    "tuple",      # Python tuple
    "dataframe",  # pandas DataFrame
    "series",     # pandas Series
    "dict",       # Python dict
    "object",     # User-defined object (via __dict__ or __slots__)
    "set",        # Python set
    "frozenset",  # Python frozenset
    "complex",    # Complex number (real/imag parts)
]


class CompoundDiff(BaseModel):
    """
    Represents diffs for a compound/container structure.

    This wrapper provides type information about what kind of structure
    the diffs came from, enabling smarter truncation handling. For example,
    truncation in an array's values is OK (we know it changed), but
    truncation in a DataFrame's columns is NOT OK (we might miss columns).

    Attributes:
        source_type: What kind of structure this diff represents
        children: Dict mapping path strings to nested DiffNodes
        truncated: Whether the diff was truncated (hit max_diffs limit)

    Example:
        >>> diff = CompoundDiff(
        ...     source_type="dataframe",
        ...     children={"['A']": ValueComparison(...), "['B']": ValueComparison(...)},
        ...     truncated=False
        ... )
    """
    source_type: DiffSourceType = Field(..., description="Type of structure this diff represents")
    children: Dict[str, "DiffNode"] = Field(default_factory=dict, description="Nested diffs")
    truncated: bool = Field(default=False, description="Whether diff was truncated")

    class Config:
        arbitrary_types_allowed = True


# Type alias for the tree structure of differences
# A DiffNode is either:
# - A ValueComparison (leaf node - actual difference)
# - A CompoundDiff (container node with type info and nested diffs)
DiffNode = Union[ValueComparison, CompoundDiff]


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

    def close_only(self) -> 'DiffResult':
        """
        Return a new DiffResult containing only 'close' comparisons.

        Filters the diff tree to include only ValueComparison nodes with
        status='close', along with all parent paths needed to reach them.

        Returns:
            DiffResult: New DiffResult with only close comparisons

        Example:
            >>> result = differ.diff(a, b)
            >>> close_results = result.close_only()
            >>> # close_results contains only float comparisons within tolerance
        """
        return self._filter_by_status('close')

    def different_only(self) -> 'DiffResult':
        """
        Return a new DiffResult containing only 'different' comparisons.

        Filters the diff tree to include only ValueComparison nodes with
        status='different', along with all parent paths needed to reach them.

        Returns:
            DiffResult: New DiffResult with only different comparisons

        Example:
            >>> result = differ.diff(a, b)
            >>> diff_results = result.different_only()
            >>> # diff_results excludes close float comparisons
        """
        return self._filter_by_status('different')

    def _filter_by_status(self, status: str) -> 'DiffResult':
        """
        Filter diff tree by ValueComparison status.

        Args:
            status: Status to filter for ('close' or 'different')

        Returns:
            DiffResult: New DiffResult with only matching comparisons
        """
        filtered_diffs = {}

        for var_name, diff_node in self.differences.items():
            filtered_node = self._filter_node_by_status(diff_node, status)
            if filtered_node is not None:
                filtered_diffs[var_name] = filtered_node

        return DiffResult(differences=filtered_diffs)

    def _filter_node_by_status(self, node: DiffNode, status: str):
        """
        Recursively filter a DiffNode by status.

        Args:
            node: DiffNode to filter (ValueComparison, CompoundDiff, or dict)
            status: Status to filter for

        Returns:
            Filtered node, or None if no matches found
        """
        if isinstance(node, ValueComparison):
            # Leaf node - return it only if status matches
            return node if node.status == status else None
        elif isinstance(node, CompoundDiff):
            # CompoundDiff node - recursively filter children
            filtered_children: Dict[str, DiffNode] = {}
            for key, child_node in node.children.items():
                filtered_child = self._filter_node_by_status(child_node, status)
                if filtered_child is not None:
                    filtered_children[key] = filtered_child

            # Return filtered CompoundDiff only if it has content
            if filtered_children:
                return CompoundDiff(
                    source_type=node.source_type,
                    children=filtered_children,
                    truncated=node.truncated
                )
            return None
        elif isinstance(node, dict):
            # Legacy dict node - recursively filter children
            filtered_dict = {}
            for key, child_node in node.items():
                filtered_child = self._filter_node_by_status(child_node, status)
                if filtered_child is not None:
                    filtered_dict[key] = filtered_child

            # Return filtered dict only if it has content
            return filtered_dict if filtered_dict else None
        else:
            # Unknown node type, return as-is (shouldn't happen)
            return node


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
        elif isinstance(node, CompoundDiff):
            # Recursively serialize CompoundDiff
            return {
                "type": "compound",
                "source_type": node.source_type,
                "truncated": node.truncated,
                "children": {key: serialize_node(value) for key, value in node.children.items()}
            }
        elif isinstance(node, dict):
            # Legacy dict - recursively serialize nested diffs
            return {key: serialize_node(value) for key, value in node.items()}
        else:
            # Shouldn't happen, but handle gracefully
            return {"type": "unknown", "value": str(node)}

    return {var: serialize_node(node) for var, node in diff_result.items()}


class ExecutionError(BaseModel):
    """
    Details about a code execution error/crash.

    Captures comprehensive error information including the exception type,
    message, and full stack trace for debugging purposes.

    Attributes:
        error_type: The type of exception (e.g., "ValueError", "ZeroDivisionError")
        error_message: The exception message
        traceback: Full formatted stack trace
        code_snippet: Optional snippet of the code that crashed
    """
    error_type: str = Field(..., description="Exception type name")
    error_message: str = Field(..., description="Exception message")
    traceback: str = Field(..., description="Full formatted stack trace")
    code_snippet: Optional[str] = Field(None, description="Code that crashed")

    class Config:
        arbitrary_types_allowed = True


class TestCodeSuccess(BaseModel):
    """
    Result when both original and modified code execute successfully.

    This model contains the diff result comparing the outputs of both
    code versions, along with timing information to calculate speedup.

    Attributes:
        status: Discriminator tag, always "success"
        diff: The diff result comparing variables from both executions
        original_duration: Execution time of the original code in seconds
        modified_duration: Execution time of the modified code in seconds
        speedup: Calculated speedup ratio (original_duration / modified_duration)
    """
    status: Literal["success"] = Field(default="success", description="Result status discriminator")
    diff: DiffResult = Field(..., description="Diff result comparing variables")
    original_duration: float = Field(..., description="Original code execution time in seconds")
    modified_duration: float = Field(..., description="Modified code execution time in seconds")
    speedup: float = Field(..., description="Speedup ratio (original / modified)")

    class Config:
        arbitrary_types_allowed = True


class TestCodeOriginalCrash(BaseModel):
    """
    Result when the original code crashes during execution.

    This indicates the original code has a bug or runtime error,
    so optimization cannot proceed.

    Attributes:
        status: Discriminator tag, always "original_crash"
        error: Detailed information about the crash
        original_duration: Optional partial execution time before crash
    """
    status: Literal["original_crash"] = Field(default="original_crash", description="Result status discriminator")
    error: ExecutionError = Field(..., description="Error details")
    original_duration: Optional[float] = Field(None, description="Time before crash (if measurable)")

    class Config:
        arbitrary_types_allowed = True


class TestCodeModifiedCrash(BaseModel):
    """
    Result when the modified code crashes but the original succeeded.

    This indicates the optimization introduced a bug or behavioral change
    that causes the code to fail.

    Attributes:
        status: Discriminator tag, always "modified_crash"
        error: Detailed information about the crash
        original_duration: Execution time of the original code (succeeded)
        modified_duration: Optional partial execution time before crash
    """
    status: Literal["modified_crash"] = Field(default="modified_crash", description="Result status discriminator")
    error: ExecutionError = Field(..., description="Error details")
    original_duration: float = Field(..., description="Original code execution time in seconds")
    modified_duration: Optional[float] = Field(None, description="Time before crash (if measurable)")

    class Config:
        arbitrary_types_allowed = True


# Discriminated union type for test_code results
# The 'status' field is used to discriminate between the three possible outcomes
TestCodeResult = Annotated[
    Union[TestCodeSuccess, TestCodeOriginalCrash, TestCodeModifiedCrash],
    Field(discriminator="status")
]


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

        elif isinstance(node, CompoundDiff):
            # CompoundDiff structure - recurse into children
            for key, child_node in node.children.items():
                # Build the new path
                new_path = f"{path}{key}"
                format_node(var_name, child_node, new_path)

            # Add truncation message if applicable
            if node.truncated:
                full_path = f"{var_name}{path}" if path else var_name
                lines.append(f"  - *{full_path}: (truncated, more differences exist)*")

        elif isinstance(node, dict):
            # Legacy dict structure - recurse into nested diffs
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
