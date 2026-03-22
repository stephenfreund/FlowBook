"""
Data models for Reproducibility.

This module defines the core data structures for the reproducibility system,
mapping to the formal specification in main.tex and FORMAL_DEVELOPMENT.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Union

from flowbook.kernel_support.models import TrackingData

if TYPE_CHECKING:
    from flowbook.kernel.changes import Change


# =============================================================================
# Staleness Reason Types
# =============================================================================


class ReasonType(str, Enum):
    """Why a cell is stale.

    Names align with formal predicates from [Inst-Run] specification:
    - FORWARD_STALE: ForwardStale(R,W,i,j) - cell j>i reads location that i wrote
    - WRITE_OVERLAP: ForwardStale write part - cell j>i writes to location that i wrote
    - BACKWARD_STALE: BackwardStale(W,W',i,j) - cell j<i was last writer of removed write
    - NO_READ_BEFORE_WRITE: ¬NoReadBeforeWrite - reads location written by later cell (forward contamination)
    - NO_READ_AND_WRITE: ¬NoReadAndWrite - cell reads and writes same location
    - WRITE_BEFORE_READ: ¬WriteBeforeRead - reads user var not written by earlier cell
    - NO_WRITE_AFTER_READ: ¬NoWriteAfterRead - wrote location read by earlier fresh cell
    """

    NEVER_EXECUTED = "never_executed"
    CODE_CHANGED = "code_changed"
    FORWARD_STALE = "forward_stale"  # was INPUT_CHANGED - cell reads location that changed
    WRITE_OVERLAP = "write_overlap"  # cell writes to same location as earlier cell (no convergence)
    BACKWARD_STALE = "backward_stale"  # was WRITE_CONFLICT
    NO_READ_BEFORE_WRITE = "no_read_before_write"  # was READS_FROM_LATER - forward contamination
    NO_READ_AND_WRITE = "no_read_and_write"  # cell reads and writes same location
    WRITE_BEFORE_READ = "write_before_read"  # reads user var not written by earlier cell
    ORDER_CHANGED = "order_changed"
    NO_WRITE_AFTER_READ = "no_write_after_read"  # was BACKWARD_MUTATION - cell wrote to location read by earlier cell
    UNRECOVERABLE_MUTATION = "unrecoverable_mutation"  # in-place mutation without rebinding


# =============================================================================
# Error Types (Formal Predicate Violations)
# =============================================================================
# These error types represent violations of the four formal validity predicates
# from FORMAL_DEVELOPMENT.md §3.2. When these predicates fail, execution is
# rejected with rollback (unless continue_after_violation is enabled).
# =============================================================================


class ErrorType(str, Enum):
    """
    Type of reproducibility error (formal predicate violation).

    Formal ref: main.tex §3.2, FORMAL_DEVELOPMENT.md §3.2 (lines 176-179)

    These correspond to the four validity predicates from [Inst-Run]:
    - NO_READ_AND_WRITE: Rᵢ ∩ Wᵢ = ∅ (cell reads and writes same location)
    - WRITE_BEFORE_READ: Rᵢ ⊆ W_{1..i-1} (reads user var not written by earlier cell)
    - NO_READ_BEFORE_WRITE: Rᵢ ∩ W_{i+1..n} = ∅ (forward contamination)
    - NO_WRITE_AFTER_READ: Wᵢ ∩ R_{1..i-1} = ∅ (backward mutation)
    """

    NO_READ_AND_WRITE = "no_read_and_write"
    WRITE_BEFORE_READ = "write_before_read"
    NO_READ_BEFORE_WRITE = "no_read_before_write"  # forward contamination
    NO_WRITE_AFTER_READ = "no_write_after_read"    # backward mutation
    UNRECOVERABLE_MUTATION = "unrecoverable_mutation"  # in-place mutation without rebinding


@dataclass
class ReproducibilityError:
    """
    A reproducibility error (formal predicate violation).

    Formal ref: FORMAL_DEVELOPMENT.md §3.2

    Errors cause execution rejection with rollback (unless continue_after_violation
    is enabled). This is distinct from staleness reasons which are informational.

    Attributes:
        error_type: The violated formal predicate
        cell_id: ID of the cell that caused the error
        locations: Variables/columns involved in the violation
        message: Human-readable error message
        causer_cell: For backward/forward violations, the conflicting cell ID
        detail: Additional diagnostic information
    """

    error_type: ErrorType
    cell_id: str
    locations: List[str]
    message: str
    causer_cell: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = {
            "error_type": self.error_type.value,
            "cell_id": self.cell_id,
            "locations": self.locations,
            "message": self.message,
        }
        if self.causer_cell is not None:
            result["causer_cell"] = self.causer_cell
        if self.detail is not None:
            result["detail"] = self.detail
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReproducibilityError":
        """Create from dict (for deserialization)."""
        return cls(
            error_type=ErrorType(data["error_type"]),
            cell_id=data["cell_id"],
            locations=data["locations"],
            message=data["message"],
            causer_cell=data.get("causer_cell"),
            detail=data.get("detail"),
        )


@dataclass(frozen=True)
class Reason:
    """
    A single reason why a cell is stale.

    Attributes:
        type: The category of staleness reason
        loc: Variable or location involved (e.g., "x", "df.col")
        cell_id: Cell that caused the staleness (actual ID, not @position)
    """

    type: ReasonType
    loc: Optional[str] = None
    cell_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result: Dict[str, Any] = {"type": self.type.value}
        if self.loc is not None:
            result["loc"] = self.loc
        if self.cell_id is not None:
            result["cell_id"] = self.cell_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reason":
        """Create from dict (for deserialization)."""
        return cls(
            type=ReasonType(data["type"]),
            loc=data.get("loc"),
            cell_id=data.get("cell_id"),
        )

    def __str__(self) -> str:
        parts = [self.type.value]
        if self.loc:
            parts.append(f"loc={self.loc}")
        if self.cell_id:
            parts.append(f"cell={self.cell_id}")
        return f"Reason({', '.join(parts)})"


@dataclass
class CellStatus:
    """
    Cell status: Clean or Stale with a set of reasons.

    A cell is Clean if it needs no action. A cell is Stale if it has one
    or more reasons requiring re-execution.
    """

    is_clean: bool
    reasons: Set[Reason] = field(default_factory=set)

    @classmethod
    def clean(cls) -> "CellStatus":
        """Create a Clean status."""
        return cls(is_clean=True, reasons=set())

    @classmethod
    def stale(cls, reasons: Set[Reason]) -> "CellStatus":
        """Create a Stale status with given reasons."""
        return cls(is_clean=False, reasons=reasons)

    @classmethod
    def never_executed(cls) -> "CellStatus":
        """Create a Stale status for never-executed cell."""
        return cls(is_clean=False, reasons={Reason(ReasonType.NEVER_EXECUTED)})

    @classmethod
    def code_changed(cls) -> "CellStatus":
        """Create a Stale status for edited cell."""
        return cls(is_clean=False, reasons={Reason(ReasonType.CODE_CHANGED)})

    def add_reason(self, reason: Reason) -> None:
        """Add a reason (converts to Stale if Clean)."""
        self.is_clean = False
        self.reasons.add(reason)

    def clear_reasons(self) -> None:
        """Clear all reasons and mark as Clean."""
        self.is_clean = True
        self.reasons = set()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "is_clean": self.is_clean,
            "reasons": sorted(
                [r.to_dict() for r in self.reasons],
                key=lambda r: r["type"],
            ),
        }

    def __str__(self) -> str:
        if self.is_clean:
            return "Clean"
        reason_strs = sorted(str(r) for r in self.reasons)
        return f"Stale({{{', '.join(reason_strs)}}})"


@dataclass
class ReproducibilityResult:
    """Result of monitor check — determines transition rule (EXEC-ACCEPT/REJECT)."""

    stale_cells: List[str]  # cell IDs that need re-execution (document order)
    changed_variables: List[str]  # variables that changed value
    column_changed: Dict[str, List[str]] = field(default_factory=dict)  # var -> [changed columns]
    structural_warnings: List[str] = field(default_factory=list)  # warnings from WARN mode
    # Staleness reasons per cell: { cell_id: [reason_dict, ...] }
    staleness_reasons: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Reproducibility errors (formal predicate violations that cause rejection)
    errors: List["ReproducibilityError"] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Return True if any formal predicate violations were detected."""
        return len(self.errors) > 0


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
    # Staleness reasons per cell: { cell_id: [reason_dict, ...] }
    staleness_reasons: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Reproducibility errors (formal predicate violations)
    errors: List[Dict[str, Any]] = field(default_factory=list)

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
                "staleness_reasons": self.staleness_reasons,
                "errors": self.errors,
            }
        }


@dataclass
class MovedCell:
    """Record of a cell that changed position in notebook order."""

    cell_id: str
    old_position: int
    new_position: int

    @property
    def moved_forward(self) -> bool:
        """True if cell moved to a later position (old < new)."""
        return self.old_position < self.new_position

    @property
    def moved_backward(self) -> bool:
        """True if cell moved to an earlier position (new < old)."""
        return self.new_position < self.old_position


@dataclass
class OrderDelta:
    """Delta between old and new cell order."""

    deleted: List[str]  # Cell IDs in old order but not in new
    inserted: List[str]  # Cell IDs in new order but not in old
    moved: List[MovedCell]  # Cells that changed position


@dataclass
class OrderChangeResult:
    """Result of processing a cell order change."""

    newly_stale: List[str]  # Cells marked stale by this order change
    warnings: List[str]  # Human-readable warnings
    delta: OrderDelta  # The computed delta


# =============================================================================
# Cell State Snapshot (for rollback)
# =============================================================================


@dataclass
class CellStateSnapshot:
    """
    Snapshot of a cell's state before execution, used for rollback.

    When a cell execution is rejected (e.g., due to a reproducibility violation),
    the kernel rolls back the namespace. This snapshot allows the enforcer to
    also rollback its analysis state to match.

    Attributes:
        cell_id: The cell whose state was captured
        reads: Previous reads set (None if cell hadn't executed)
        writes: Previous writes set (None if cell hadn't executed)
        status: Previous CellStatus (None if cell hadn't executed)
        tracking_data: Previous TrackingData (None if cell hadn't executed)
        execution_seq: Previous execution sequence number (None if cell hadn't executed)
        structural_reads_values: Previous structural read values (None if none)
        typed_changes: Previous typed changes (None if none)
    """

    cell_id: str
    reads: Optional[Any]  # ReadLocSet (FrozenSet[ReadLoc]) or None
    writes: Optional[Any]  # WriteLocSet (FrozenSet[WriteLoc]) or None
    status: Optional["CellStatus"]
    tracking_data: Optional[TrackingData]
    execution_seq: Optional[int]
    structural_reads_values: Optional[Dict[str, Dict[str, str]]]
    typed_changes: Optional[List["Change"]]


