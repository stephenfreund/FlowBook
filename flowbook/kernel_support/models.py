"""
Pydantic models for kernel execution data structures.

This module defines type-safe models for data that flows through the kernel
during cell execution, including:

- TrackingData: Variable access patterns (reads, writes, column-level tracking)
- ExecutionProfile: Timing and profiling information
- ExecutionMetadata: Complete metadata for cell execution results
- MonotonicityViolation: Details when monotonicity constraints are violated
- ExecutionContext: Pre-execution state and configuration

These models ensure type safety and provide automatic serialization/deserialization
for communication between kernel components and the frontend.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from flowbook.kernel.access_events import (
        AccessEvent,
        ColumnRead,
        ColumnWrite,
        ReadEvent,
        StructuralRead,
        VariableRead,
    )


class TrackingData(BaseModel):
    """
    Captured variable access patterns during cell execution.

    This model represents the dynamic dependency information collected
    by TrackingDict during code execution. It tracks both variable-level
    and column-level (for DataFrames) access patterns.

    Attributes:
        reads_before_writes: Variables read before being written in the cell
        writes: All variables written during cell execution
        column_reads_before_writes: DataFrame columns read before written, by variable path
        column_writes: DataFrame columns written, by variable path
        structural_reads: Structural attributes/methods accessed per variable path

    Example:
        >>> data = TrackingData(
        ...     reads_before_writes=["df", "config"],
        ...     writes=["result", "df"],
        ...     column_reads_before_writes={"df": ["price", "quantity"]},
        ...     column_writes={"df": ["total"]},
        ...     structural_reads={"df": {"columns", "shape"}}
        ... )
    """

    reads_before_writes: Set[str] = Field(
        default_factory=set,
        description="Variables read before being written in this cell",
    )
    writes: Set[str] = Field(
        default_factory=set, description="All variables written during cell execution"
    )
    column_reads_before_writes: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="DataFrame columns read before written, keyed by variable path",
    )
    column_writes: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="DataFrame columns written, keyed by variable path",
    )
    structural_reads: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description=(
            "Structural attributes/methods accessed, keyed by variable path. "
            "When code accesses df.columns, df.shape, df.describe(), etc., "
            "these are recorded here. The diff then requires structural "
            "equality for these variables, not just column value equality. "
            "e.g., {'df': {'columns', 'shape'}, 'data[\"train\"]': {'describe'}}"
        ),
    )
    file_reads_before_writes: Set[str] = Field(
        default_factory=set,
        description="Absolute file paths read before being written in this cell",
    )
    file_writes: Set[str] = Field(
        default_factory=set,
        description="Absolute file paths written during cell execution",
    )

    def get_rbw_vars(self) -> Set[str]:
        """Return read-before-write variables as a set."""
        return set(self.reads_before_writes)

    def get_column_rbw_sets(self) -> Dict[str, Set[str]]:
        """Return column RBW data with sets instead of lists."""
        return {k: set(v) for k, v in self.column_reads_before_writes.items()}

    def has_structural_read(self, var_path: str) -> bool:
        """Check if any structural attribute was read for a variable."""
        return bool(self.structural_reads.get(var_path))

    def has_column_structure_read(self, var_path: str) -> bool:
        """
        Check if column-revealing attributes were read.

        These include: columns, keys, iter, dtypes, T, axes, describe,
        to_dict, info, head, tail, sample, select_dtypes, etc.

        If any of these were accessed, adding columns should be detected.
        """
        attrs = self.structural_reads.get(var_path, set())
        column_revealing = {
            'columns', 'keys', 'iter', 'dtypes', 'T', 'axes', 'values',
            'describe', 'to_dict', 'info', 'head', 'tail', 'sample',
            'select_dtypes', 'to_records', 'memory_usage',
        }
        return bool(attrs & column_revealing)

    def has_row_structure_read(self, var_path: str) -> bool:
        """
        Check if row-revealing attributes were read.

        These include: index, len, shape, size, empty

        If any of these were accessed, adding/removing rows should be detected.
        """
        attrs = self.structural_reads.get(var_path, set())
        row_revealing = {'index', 'len', 'shape', 'size', 'empty'}
        return bool(attrs & row_revealing)

    def to_access_events(self) -> List["AccessEvent"]:
        """
        Convert to a list of typed AccessEvent objects.

        Returns all access events in a deterministic order:
        1. All ColumnRead events (sorted by variable, then column)
        2. All ColumnWrite events (sorted by variable, then column)
        3. All StructuralRead events (sorted by variable, then attr)

        Returns:
            List of AccessEvent objects
        """
        from flowbook.kernel.access_events import (
            ColumnRead,
            ColumnWrite,
            StructuralRead,
        )

        events: List["AccessEvent"] = []

        # Column reads
        for var, columns in sorted(self.column_reads_before_writes.items()):
            for col in sorted(columns):
                events.append(ColumnRead(variable=var, column=col))

        # Column writes
        for var, columns in sorted(self.column_writes.items()):
            for col in sorted(columns):
                events.append(ColumnWrite(variable=var, column=col))

        # Structural reads
        for var, attrs in sorted(self.structural_reads.items()):
            for attr in sorted(attrs):
                events.append(StructuralRead(variable=var, attr=attr))

        return events

    def to_read_events(self) -> List["ReadEvent"]:
        """
        Convert to only read events (for conflict detection).

        This is the subset of access events that can conflict with changes:
        - ColumnRead: conflicts with changes to that column
        - StructuralRead: conflicts with structural changes
        - VariableRead: conflicts with any change to the variable (whole-value read)

        ColumnWrite events are not included because writes don't create
        dependencies on prior values.

        Returns:
            List of ReadEvent objects (ColumnRead, StructuralRead, VariableRead)
        """
        from flowbook.kernel.access_events import (
            ColumnRead,
            StructuralRead,
            VariableRead,
        )

        events: List["ReadEvent"] = []

        # Track which variables have specific read info
        vars_with_detail = set()

        # Column reads
        for var, columns in sorted(self.column_reads_before_writes.items()):
            vars_with_detail.add(var)
            for col in sorted(columns):
                events.append(ColumnRead(variable=var, column=col))

        # Structural reads
        for var, attrs in sorted(self.structural_reads.items()):
            vars_with_detail.add(var)
            for attr in sorted(attrs):
                events.append(StructuralRead(variable=var, attr=attr))

        # Variable-level reads (for variables without column or structural detail)
        # These are non-DataFrame/Series variables that were read
        for var in sorted(self.reads_before_writes):
            if var not in vars_with_detail:
                events.append(VariableRead(variable=var))

        return events

    def to_json_friendly(self) -> Dict[str, Any]:
        """
        Return dict with sorted lists instead of sets for JSON serialization.

        This is the canonical format for sending tracking data to the frontend.

        Returns:
            Dict with keys: reads, writes, column_reads, column_writes, structural_reads
        """
        return {
            "reads": sorted(self.reads_before_writes),
            "writes": sorted(self.writes),
            "column_reads": {k: sorted(v) for k, v in sorted(self.column_reads_before_writes.items())},
            "column_writes": {k: sorted(v) for k, v in sorted(self.column_writes.items())},
            "structural_reads": {k: sorted(v) for k, v in sorted(self.structural_reads.items())},
            "file_reads": sorted(os.path.relpath(p) for p in self.file_reads_before_writes),
            "file_writes": sorted(os.path.relpath(p) for p in self.file_writes),
        }

    def get_read_variables(self) -> Set[str]:
        """
        Get all variables that were read during cell execution.

        Combines:
        - Variables with column reads
        - Variables with structural reads
        - Variable-level reads

        Returns:
            Set of variable names that were read
        """
        variables = set()
        variables.update(self.column_reads_before_writes.keys())
        variables.update(self.structural_reads.keys())
        variables.update(self.reads_before_writes)
        return variables

    model_config = ConfigDict(frozen=False)  # Allow modification after creation

    def get_written_variables(self) -> Set[str]:
        """
        Get all variables that were written during cell execution.

        Returns:
            Set of variable names that were written
        """
        return set(self.writes)


class ExecutionProfile(BaseModel):
    """
    Profiling and timing data for cell execution.

    Captures performance information including execution duration,
    Scalene profiler output, and type information for variables
    before and after execution.

    Attributes:
        duration: Execution time in seconds
        profile: Scalene profiler output (empty if profiling disabled)
        env: Variable types before execution (name -> type string)
        env_after: Variable types after execution (name -> type string)
    """

    duration: float = Field(..., description="Execution time in seconds")
    profile: str = Field(default="", description="Scalene profiler output")
    env: Dict[str, str] = Field(
        default_factory=dict, description="Variable types before execution"
    )
    env_after: Dict[str, str] = Field(
        default_factory=dict, description="Variable types after execution"
    )


class ExecutionMetadata(BaseModel):
    """
    Complete metadata for a cell execution result.

    This is the top-level metadata structure attached to cell execution
    results, containing profiling information and optional dynamic
    dependency tracking data.

    Attributes:
        profile: Profiling and timing information
        dynamic_dependencies: Variable access patterns (if tracking enabled)
    """

    profile: ExecutionProfile = Field(..., description="Profiling data")
    dynamic_dependencies: Optional[TrackingData] = Field(
        None, description="Dynamic dependency tracking data (if enabled)"
    )

    def to_display_metadata(self) -> dict:
        """Convert to dict format expected by display helpers."""
        result = {"profile": self.profile.model_dump()}
        if self.dynamic_dependencies is not None:
            result["dynamic_dependencies"] = self.dynamic_dependencies.model_dump()
        return result


class MonotonicityViolation(BaseModel):
    """
    Details of a monotonicity constraint violation.

    When monotonicity enforcement is enabled, cells that modify read-before-write
    variables are rejected and their effects rolled back. This model captures
    the details of such violations for error reporting.

    Attributes:
        violated_vars: List of variable names that were incorrectly modified
        diff_details: Human-readable description of the differences
        error_summary: Short summary for error message
    """

    violated_vars: List[str] = Field(
        ..., description="Variables that were modified in violation of monotonicity"
    )
    diff_details: str = Field(
        ..., description="Human-readable details of the differences found"
    )
    error_summary: str = Field(..., description="Short summary for error reporting")

    def to_error_result(self, execution_count: int) -> dict:
        """Convert to kernel error result format."""
        return {
            "status": "error",
            "execution_count": execution_count,
            "ename": "MonotonicityError",
            "evalue": self.error_summary,
            "traceback": [self.diff_details],
        }


class ExecutionContext(BaseModel):
    """
    Pre-execution state and configuration for a cell.

    Captures all the context needed to execute a cell, extracted
    from the execution request parameters. This allows clean
    separation between request parsing and execution logic.

    Attributes:
        cell_id: Unique identifier for the cell being executed
        code: The code to execute (with directives stripped)
        timeout: Execution timeout in seconds
        original_code: Original code before directive parsing
    """

    model_config = ConfigDict(frozen=False)

    cell_id: Optional[str] = Field(None, description="Cell identifier")
    code: str = Field(..., description="Code to execute")
    timeout: float = Field(..., description="Execution timeout in seconds")
    original_code: str = Field(..., description="Original code before parsing")

    @property
    def has_cell_magics(self) -> bool:
        """Check if code contains cell magics."""
        return self.code.startswith("%") or "\n%" in self.code

    @property
    def has_shell_magics(self) -> bool:
        """Check if code contains shell commands."""
        return self.code.startswith("!") or "\n!" in self.code

    @property
    def should_profile(self) -> bool:
        """Check if this cell should be profiled."""
        return (
            self.cell_id is not None
            and not self.has_cell_magics
            and not self.has_shell_magics
        )
