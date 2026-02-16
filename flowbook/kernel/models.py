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
    run_duration_ms: float = 0.0  # Code execution time
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
                "run_duration_ms": self.run_duration_ms,
                "state_duration_ms": self.state_duration_ms,
                "check_duration_ms": self.check_duration_ms,
                "cell_is_contaminated": self.cell_is_contaminated,
                "exec_mode": self.exec_mode,
            }
        }
