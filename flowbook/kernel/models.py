"""
Data models for Reproducibility.

This module defines the core data structures for the reproducibility system,
mapping to the formal specification in main.tex and FORMAL_DEVELOPMENT.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING, Union

from flowbook.kernel_support.models import TrackingData

if TYPE_CHECKING:
    from flowbook.kernel.changes import Change


# =============================================================================
# Location Types (Formal: ℓ ∈ Loc)
# =============================================================================
# These types implement the formal location specification:
#   ℓ ∈ Loc ::= Var(x) | Col(df, c) | File(path) | Structural(df, attr)
#
# Formal ref: main.tex §1, FORMAL_DEVELOPMENT.md §1.1, §8.1-8.3
# =============================================================================


class LocType(str, Enum):
    """Type of location in the formal model."""

    VAR = "var"           # Variable: Var(x)
    COLUMN = "column"     # DataFrame column: Col(df, c)
    FILE = "file"         # File path: File(path)
    STRUCTURAL = "struct" # Structural attribute: Structural(df, attr)


@dataclass(frozen=True)
class Loc:
    """
    A location in the formal model.

    Formal ref: main.tex §1, FORMAL_DEVELOPMENT.md §1.1

    Locations represent the granular units of state that cells read and write.
    The formal model uses: ℓ ∈ Loc ::= Var(x) | Col(df, c) | File(path) | Structural(df, attr)

    Examples:
        Loc.var("x")                    # Variable x
        Loc.column("df", "price")       # Column df['price']
        Loc.file("/path/to/data.csv")   # File
        Loc.structural("df", "shape")   # Structural attribute df.shape
    """

    type: LocType
    name: str
    qualifier: Optional[str] = None  # For columns/structural: the variable name

    @classmethod
    def var(cls, name: str) -> "Loc":
        """Create a variable location: Var(name)."""
        return cls(type=LocType.VAR, name=name)

    @classmethod
    def column(cls, var: str, col: str) -> "Loc":
        """Create a column location: Col(var, col)."""
        return cls(type=LocType.COLUMN, name=col, qualifier=var)

    @classmethod
    def file(cls, path: str) -> "Loc":
        """Create a file location: File(path)."""
        return cls(type=LocType.FILE, name=path)

    @classmethod
    def structural(cls, var: str, attr: str) -> "Loc":
        """Create a structural attribute location: Structural(var, attr)."""
        return cls(type=LocType.STRUCTURAL, name=attr, qualifier=var)

    def __str__(self) -> str:
        if self.type == LocType.VAR:
            return f"Var({self.name})"
        elif self.type == LocType.COLUMN:
            return f"Col({self.qualifier}, {self.name})"
        elif self.type == LocType.FILE:
            return f"File({self.name})"
        else:
            return f"Structural({self.qualifier}, {self.name})"


# Type alias for a set of locations
LocSet = FrozenSet[Loc]


def tracking_to_read_locs(tracking: TrackingData) -> LocSet:
    """
    Convert TrackingData reads to a set of Loc objects.

    Formal ref: Rᵢ in main.tex, FORMAL_DEVELOPMENT.md §1.2

    This creates the unified read set Rᵢ that includes:
    - Variable reads (Var)
    - Column reads (Col)
    - File reads (File)
    - Structural reads (Structural)
    """
    locs: Set[Loc] = set()

    # Variable reads
    for var in tracking.reads_before_writes:
        locs.add(Loc.var(var))

    # Column reads
    for var, cols in tracking.column_reads_before_writes.items():
        for col in cols:
            locs.add(Loc.column(var, col))

    # File reads
    for path in tracking.file_reads_before_writes:
        locs.add(Loc.file(path))

    # Structural reads
    for var, attrs in tracking.structural_reads.items():
        for attr in attrs:
            locs.add(Loc.structural(var, attr))

    return frozenset(locs)


def tracking_to_write_locs(tracking: TrackingData) -> LocSet:
    """
    Convert TrackingData writes to a set of Loc objects.

    Formal ref: Wᵢ in main.tex, FORMAL_DEVELOPMENT.md §1.2

    This creates the unified write set Wᵢ that includes:
    - Variable writes (Var)
    - Column writes (Col)
    - File writes (File)
    """
    locs: Set[Loc] = set()

    # Variable writes
    for var in tracking.writes:
        locs.add(Loc.var(var))

    # Column writes
    for var, cols in tracking.column_writes.items():
        for col in cols:
            locs.add(Loc.column(var, col))

    # File writes
    for path in tracking.file_writes:
        locs.add(Loc.file(path))

    return frozenset(locs)


def locs_intersect(a: LocSet, b: LocSet) -> bool:
    """Check if two location sets have any overlap."""
    return bool(a & b)


def get_var_locs(locs: LocSet) -> Set[str]:
    """Extract just the variable names from a LocSet."""
    return {loc.name for loc in locs if loc.type == LocType.VAR}


# =============================================================================
# Staleness Reason Types
# =============================================================================


class ReasonType(str, Enum):
    """Why a cell is stale."""

    NEVER_EXECUTED = "never_executed"
    CODE_CHANGED = "code_changed"
    INPUT_CHANGED = "input_changed"
    WRITE_CONFLICT = "write_conflict"
    READS_FROM_LATER = "reads_from_later"
    SOURCE_DELETED = "source_deleted"
    ORDER_CHANGED = "order_changed"
    SKIPPED_UPSTREAM = "skipped_upstream"  # Cell reads from wrong writer; re-run won't help


@dataclass(frozen=True)
class Reason:
    """
    A single reason why a cell is stale.

    Attributes:
        type: The category of staleness reason
        loc: Variable or location involved (e.g., "x", "df.col")
        cell_id: Cell that caused the staleness (actual ID, not @position)
        expected_cell_id: For skipped writer cases, the cell that should have provided the value
    """

    type: ReasonType
    loc: Optional[str] = None
    cell_id: Optional[str] = None
    expected_cell_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result: Dict[str, Any] = {"type": self.type.value}
        if self.loc is not None:
            result["loc"] = self.loc
        if self.cell_id is not None:
            result["cell_id"] = self.cell_id
        if self.expected_cell_id is not None:
            result["expected_cell_id"] = self.expected_cell_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reason":
        """Create from dict (for deserialization)."""
        return cls(
            type=ReasonType(data["type"]),
            loc=data.get("loc"),
            cell_id=data.get("cell_id"),
            expected_cell_id=data.get("expected_cell_id"),
        )

    def __str__(self) -> str:
        parts = [self.type.value]
        if self.loc:
            parts.append(f"loc={self.loc}")
        if self.cell_id:
            parts.append(f"cell={self.cell_id}")
        if self.expected_cell_id:
            parts.append(f"expected={self.expected_cell_id}")
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
        """Add a reason (converts to Stale if Clean).

        For INPUT_CHANGED and SKIPPED_UPSTREAM reasons with a specific location,
        replaces any existing reason of either type for the same location
        (they are mutually exclusive - only the most recent matters).
        """
        self.is_clean = False

        # INPUT_CHANGED and SKIPPED_UPSTREAM are mutually exclusive for same location
        location_based_types = {ReasonType.INPUT_CHANGED, ReasonType.SKIPPED_UPSTREAM}
        if reason.type in location_based_types and reason.loc is not None:
            self.reasons = {
                r for r in self.reasons
                if not (r.type in location_based_types and r.loc == reason.loc)
            }

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
class ReproducibilityResult:
    """Result of monitor check — determines transition rule (EXEC-ACCEPT/REJECT)."""

    violation: Optional[ReproducibilityViolation]  # Primary violation (backward mutation or forward dependency)
    stale_cells: List[str]  # cell IDs that need re-execution (document order)
    changed_variables: List[str]  # variables that changed value
    column_changed: Dict[str, List[str]] = field(default_factory=dict)  # var -> [changed columns]
    structural_warnings: List[str] = field(default_factory=list)  # warnings from WARN mode
    forward_violation: Optional[ReproducibilityViolation] = None  # Forward dependency violation (if any)
    # Writer violation: backward_mutation violation to store on writer cell (for forward contamination)
    writer_violation: Optional[ReproducibilityViolation] = None
    # Staleness reasons per cell: { cell_id: [reason_dict, ...] }
    staleness_reasons: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


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
    # Writer violation: backward_mutation violation to store on writer cell (for forward contamination)
    writer_violation: Optional[Dict[str, Any]] = None
    # Staleness reasons per cell: { cell_id: [reason_dict, ...] }
    staleness_reasons: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

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
                "writer_violation": self.writer_violation,
                "staleness_reasons": self.staleness_reasons,
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


