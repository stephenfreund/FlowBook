"""
Data models for Reproducibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from flowbook.kernel_support.models import TrackingData

if TYPE_CHECKING:
    from flowbook.kernel.changes import Change


@dataclass
class ReproducibilityViolation:
    """A reproducibility violation (backward mutation or forward dependency)."""

    mutating_cell: str  # cell that caused violation (wrote the variable)
    affected_cell: str  # cell whose reads were affected
    variables: List[str]  # variables involved in the conflict
    message: str  # human-readable description
    violation_type: str = "backward_mutation"  # "backward_mutation" | "forward_dependency"
    truncation_details: Optional[str] = None  # pretty-printed diff if truncation occurred
    # Detailed diagnostic info for better messages
    structural_reads_detail: Dict[str, Dict[str, str]] = field(default_factory=dict)  # var -> {attr -> value_repr}
    changes_detail: List[str] = field(default_factory=list)  # ["Column 'y' added", "Shape: (5,4) → (5,5)"]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "mutating_cell": self.mutating_cell,
            "affected_cell": self.affected_cell,
            "variables": self.variables,
            "message": self.message,
            "violation_type": self.violation_type,
        }
        if self.truncation_details:
            result["truncation_details"] = self.truncation_details
        if self.structural_reads_detail:
            result["structural_reads_detail"] = self.structural_reads_detail
        if self.changes_detail:
            result["changes_detail"] = self.changes_detail
        return result

    def to_error_result(self, execution_count: int) -> dict:
        """Convert to kernel error result format."""
        return {
            "status": "error",
            "execution_count": execution_count,
            "ename": "ReproducibilityViolation",
            "evalue": self.message,
            "traceback": [self.message],
        }


@dataclass
class ReproducibilityExecutionRecord:
    """Record of a cell's most recent execution — Rec[i] in the formalism (§1.6)."""

    cell_id: str
    tracking: TrackingData
    execution_seq: int  # monotonic execution counter
    # Captured values of structural attrs at read time (for better error messages)
    # Format: {var_name: {attr_name: repr_value}}
    structural_reads_values: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Cached typed changes from this cell's execution (for fast forward dependency checks)
    # These are computed once during backward mutation check and reused
    typed_changes: List["Change"] = field(default_factory=list)


@dataclass
class ReproducibilityResult:
    """Result of monitor check — determines transition rule (EXEC-ACCEPT/CONTAMINATED/REJECT)."""

    violation: Optional[ReproducibilityViolation]  # Primary violation (backward mutation)
    stale_cells: List[str]  # cell IDs that need re-execution (document order)
    changed_variables: List[str]  # variables that changed value
    column_changed: Dict[str, List[str]] = field(default_factory=dict)  # var -> [changed columns]
    structural_warnings: List[str] = field(default_factory=list)  # warnings from WARN mode
    forward_violation: Optional[ReproducibilityViolation] = None  # Forward dependency violation (if any)
    cell_is_contaminated: bool = False  # [EXEC-CONTAMINATED] True if cell executed but is forward-contaminated
    exec_mode: str = "live"  # [EXEC-RESTORE] "live" or "restore"


@dataclass
class ReproducibilityMetadata:
    """
    Metadata returned after each cell execution.
    Designed to work with existing metadata viewer.
    """

    cell_id: str
    execution_seq: int
    reads: List[str]
    writes: List[str]
    changed_variables: List[str]
    stale_cells: List[str]
    violation: Optional[Dict[str, Any]]  # ReproducibilityViolation as dict, or None
    cell_order: List[str]  # current notebook structure
    column_reads: Dict[str, List[str]] = field(default_factory=dict)  # var -> [read columns]
    column_writes: Dict[str, List[str]] = field(default_factory=dict)  # var -> [written columns]
    column_changed: Dict[str, List[str]] = field(default_factory=dict)  # var -> [changed columns]
    structural_reads: Dict[str, List[str]] = field(default_factory=dict)  # var -> [structural attrs read]
    structural_warnings: List[str] = field(default_factory=list)  # warnings from WARN mode
    file_reads: List[str] = field(default_factory=list)  # absolute file paths read
    file_writes: List[str] = field(default_factory=list)  # absolute file paths written
    # Timing information (in milliseconds)
    execute_duration_ms: float = 0.0  # Total time in _do_execute_impl
    code_duration_ms: float = 0.0  # Time for _ipython_do_execute (user code)
    state_duration_ms: float = 0.0  # Checkpoint time (pre + post)
    check_duration_ms: float = 0.0  # SDC check time
    cell_is_contaminated: bool = False  # [EXEC-CONTAMINATED] True if forward-contaminated
    exec_mode: str = "live"  # [EXEC-RESTORE] "live" or "restore"

    def to_display_metadata(self) -> dict:
        """Format for display in output metadata."""
        return {
            "flowbook": {
                "cell_id": self.cell_id,
                "execution_seq": self.execution_seq,
                "reads": self.reads,
                "writes": self.writes,
                "changed_variables": self.changed_variables,
                "stale_cells": self.stale_cells,
                "violation": self.violation,
                "cell_order": self.cell_order,
                "column_reads": self.column_reads,
                "column_writes": self.column_writes,
                "column_changed": self.column_changed,
                "structural_reads": self.structural_reads,
                "structural_warnings": self.structural_warnings,
                "file_reads": self.file_reads,
                "file_writes": self.file_writes,
                "execute_duration_ms": self.execute_duration_ms,
                "code_duration_ms": self.code_duration_ms,
                "state_duration_ms": self.state_duration_ms,
                "check_duration_ms": self.check_duration_ms,
                "cell_is_contaminated": self.cell_is_contaminated,
                "exec_mode": self.exec_mode,
            }
        }


@dataclass
class ProvenanceMap:
    """
    Tracks which cell wrote each location (§1.8.5).

    Provenance persists until overwritten by another cell's execution.
    This enables detection of forward contamination even after cells are edited.

    When cell C writes x, then C is edited to write y and re-executed:
    - Prov["x"] = C (from old execution, NOT cleared on edit)
    - Prov["y"] = C (new)
    - When B (earlier cell) reads x: Prov["x"] = C, C is after B → contaminated

    When A (earlier cell) re-executes and writes x:
    - Prov["x"] = A (updated)
    - Now B reads x: A is before B → OK
    """

    # Variable-level: var_name -> cell_id that last wrote it
    variables: Dict[str, str] = field(default_factory=dict)

    # Column-level: var_name -> col_name -> cell_id
    columns: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def update_variable(self, var: str, cell_id: str) -> None:
        """Record that cell_id wrote variable var."""
        self.variables[var] = cell_id

    def update_column(self, var: str, col: str, cell_id: str) -> None:
        """Record that cell_id wrote column col of variable var."""
        if var not in self.columns:
            self.columns[var] = {}
        self.columns[var][col] = cell_id

    def get_variable_writer(self, var: str) -> Optional[str]:
        """Get the cell_id that last wrote variable var, or None."""
        return self.variables.get(var)

    def get_column_writer(self, var: str, col: str) -> Optional[str]:
        """Get the cell_id that last wrote column col of var, or None."""
        return self.columns.get(var, {}).get(col)

    def clear(self) -> None:
        """Clear all provenance (used on kernel reset)."""
        self.variables.clear()
        self.columns.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for debugging/serialization."""
        return {
            "variables": dict(self.variables),
            "columns": {v: dict(c) for v, c in self.columns.items()},
        }
