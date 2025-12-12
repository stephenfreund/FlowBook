"""
Data models for Sequential Dataflow Consistency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from data_ferret.kernel.models import TrackingData


@dataclass
class SDCViolation:
    """A backward mutation violation."""

    mutating_cell: str  # cell that caused violation
    affected_cell: str  # earlier cell whose reads were mutated
    variables: List[str]  # variables that were mutated
    message: str  # human-readable description
    truncation_details: Optional[str] = None  # pretty-printed diff if truncation occurred

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "mutating_cell": self.mutating_cell,
            "affected_cell": self.affected_cell,
            "variables": self.variables,
            "message": self.message,
        }
        if self.truncation_details:
            result["truncation_details"] = self.truncation_details
        return result

    def to_error_result(self, execution_count: int) -> dict:
        """Convert to kernel error result format."""
        return {
            "status": "error",
            "execution_count": execution_count,
            "ename": "SDCViolation",
            "evalue": self.message,
            "traceback": [self.message],
        }


@dataclass
class SDCExecutionRecord:
    """Record of a cell's most recent execution."""

    cell_id: str
    tracking: TrackingData
    execution_seq: int  # monotonic execution counter


@dataclass
class SDCResult:
    """Result of SDC check after cell execution."""

    violation: Optional[SDCViolation]
    stale_cells: List[str]  # cell IDs that need re-execution (document order)
    changed_variables: List[str]  # variables that changed value
    column_changed: Dict[str, List[str]] = field(default_factory=dict)  # var -> [changed columns]


@dataclass
class SDCMetadata:
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
    violation: Optional[Dict[str, Any]]  # SDCViolation as dict, or None
    cell_order: List[str]  # current notebook structure
    column_reads: Dict[str, List[str]] = field(default_factory=dict)  # var -> [read columns]
    column_writes: Dict[str, List[str]] = field(default_factory=dict)  # var -> [written columns]
    column_changed: Dict[str, List[str]] = field(default_factory=dict)  # var -> [changed columns]
    # Timing information (in milliseconds)
    run_duration_ms: float = 0.0  # Code execution time
    state_duration_ms: float = 0.0  # Checkpoint time (pre + post)
    check_duration_ms: float = 0.0  # SDC check time

    def to_display_metadata(self) -> dict:
        """Format for display in output metadata."""
        return {
            "ferret_sdc": {
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
                "run_duration_ms": self.run_duration_ms,
                "state_duration_ms": self.state_duration_ms,
                "check_duration_ms": self.check_duration_ms,
            }
        }
