"""
Data models for Reproducibility.

This module defines the core data structures for the reproducibility system,
mapping to the formal specification in main.tex and FORMAL_DEVELOPMENT.md.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING, Union

from flowbook.kernel_support.models import TrackingData

if TYPE_CHECKING:
    from flowbook.kernel.changes import Change
    from flowbook.kernel_support.types import MemoryCheckpointDiffResult


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


def get_loc_variables(locs: LocSet) -> Set[str]:
    """
    Extract all variable names from a LocSet (including qualified names).

    For Var(x) → x
    For Col(df, c) → df
    For Structural(df, attr) → df
    For File(path) → path
    """
    result: Set[str] = set()
    for loc in locs:
        if loc.type == LocType.VAR:
            result.add(loc.name)
        elif loc.type in (LocType.COLUMN, LocType.STRUCTURAL):
            if loc.qualifier:
                result.add(loc.qualifier)
        elif loc.type == LocType.FILE:
            result.add(loc.name)
    return result


# =============================================================================
# Loc-based Conflict Detection
# =============================================================================
# These functions implement the formal predicates from FORMAL_DEVELOPMENT.md
# using Loc-based read/write sets for column-level granularity.
# =============================================================================


def check_loc_conflicts(
    W_i: LocSet,
    R_before_i: LocSet,
    structural_mode: "StructuralTrackingMode",
) -> Tuple[LocSet, List[str]]:
    """
    Check for conflicts between writes and reads.

    Implements: FORMAL_DEVELOPMENT.md §3.2, line 179
    NoWriteAfterRead(R, W, i) ≝ Wᵢ ∩ R_{1..i-1} = ∅

    Column-level granularity:
    - Loc.column("df", "price") conflicts only with Loc.column("df", "price")
    - Loc.column("df", "price") does NOT conflict with Loc.column("df", "quantity")
    - Loc.var("x") conflicts with any Loc involving x

    Structural mode handling:
    - OFF: structural conflicts ignored
    - WARN: structural conflicts return warnings, not violations
    - ENFORCE: structural conflicts are violations

    Args:
        W_i: Write set of cell i (locations that changed)
        R_before_i: Union of read sets of clean cells before i
        structural_mode: How to handle structural attribute conflicts

    Returns:
        Tuple of (violation_locs, warning_messages)
    """
    from flowbook.kernel_support.structural_tracking import StructuralTrackingMode

    violations: Set[Loc] = set()
    warnings: List[str] = []

    for write_loc in W_i:
        for read_loc in R_before_i:
            conflict = _locs_conflict(write_loc, read_loc)
            if conflict:
                # Check if this is a structural conflict
                if read_loc.type == LocType.STRUCTURAL:
                    if structural_mode == StructuralTrackingMode.OFF:
                        continue  # Ignore
                    elif structural_mode == StructuralTrackingMode.WARN:
                        warnings.append(
                            f"Structural conflict: {write_loc} affects {read_loc}"
                        )
                        continue  # Warning, not violation
                    # ENFORCE: fall through to add as violation

                violations.add(write_loc)

    return frozenset(violations), warnings


def _locs_conflict(write_loc: Loc, read_loc: Loc) -> bool:
    """
    Check if a write location conflicts with a read location.

    Conflict rules:
    - VAR vs VAR: same name
    - VAR vs COLUMN: var name matches column's qualifier (variable)
    - VAR vs STRUCTURAL: var name matches structural's qualifier
    - COLUMN vs COLUMN: same qualifier AND same column name
    - COLUMN vs STRUCTURAL: same qualifier (any structural change for that var)
    - FILE vs FILE: same path
    """
    # Different location types have different conflict semantics
    if write_loc.type == LocType.VAR:
        # Variable write conflicts with any read of that variable
        if read_loc.type == LocType.VAR:
            return write_loc.name == read_loc.name
        elif read_loc.type == LocType.COLUMN:
            return write_loc.name == read_loc.qualifier
        elif read_loc.type == LocType.STRUCTURAL:
            return write_loc.name == read_loc.qualifier
        elif read_loc.type == LocType.FILE:
            return False

    elif write_loc.type == LocType.COLUMN:
        # Column write conflicts with same column read OR variable read of that df
        if read_loc.type == LocType.VAR:
            return write_loc.qualifier == read_loc.name
        elif read_loc.type == LocType.COLUMN:
            return (write_loc.qualifier == read_loc.qualifier and
                    write_loc.name == read_loc.name)
        elif read_loc.type == LocType.STRUCTURAL:
            # Column change may affect structural reads (depends on attribute)
            # Adding/removing columns affects .columns, .shape, etc.
            return write_loc.qualifier == read_loc.qualifier
        elif read_loc.type == LocType.FILE:
            return False

    elif write_loc.type == LocType.STRUCTURAL:
        # Structural write (e.g., index change) conflicts with structural reads
        if read_loc.type == LocType.STRUCTURAL:
            return (write_loc.qualifier == read_loc.qualifier and
                    write_loc.name == read_loc.name)
        elif read_loc.type == LocType.VAR:
            return write_loc.qualifier == read_loc.name
        else:
            return False

    elif write_loc.type == LocType.FILE:
        # File write conflicts with file read of same path
        if read_loc.type == LocType.FILE:
            return write_loc.name == read_loc.name
        return False

    return False


def diff_to_write_locs(
    diff: "MemoryCheckpointDiffResult",
    tracking: "TrackingData",
) -> LocSet:
    """
    Convert diff result to LocSet of what actually changed.

    Ref: This bridges the gap between runtime diff detection
    and the formal Wᵢ set in FORMAL_DEVELOPMENT.md §1.2

    Mapping:
    - Variable changed → Loc.var(name)
    - Column changed → Loc.column(var, col)
    - Rows added/removed → Loc.var(var) (whole variable affected)
    - Structural change → Loc.structural(var, attr)

    Args:
        diff: MemoryCheckpointDiffResult from namespace comparison
        tracking: TrackingData for column write info

    Returns:
        LocSet of locations that changed
    """
    from flowbook.kernel_support.types import CompoundDiff, ValueComparison

    locs: Set[Loc] = set()

    for var_name, diff_node in diff.differences.items():
        # Check if this is a DataFrame/Series with column-level changes
        if isinstance(diff_node, CompoundDiff):
            if diff_node.source_type in ("dataframe", "series"):
                # Extract column-level changes
                col_changes = _extract_column_locs_from_diff(var_name, diff_node)
                if col_changes:
                    locs.update(col_changes)
                else:
                    # Whole variable changed (e.g., rows added/removed)
                    locs.add(Loc.var(var_name))
            else:
                # Other compound type - treat as variable change
                locs.add(Loc.var(var_name))
        elif isinstance(diff_node, ValueComparison):
            # Simple value changed
            locs.add(Loc.var(var_name))
        else:
            # Unknown diff type - treat as variable change
            locs.add(Loc.var(var_name))

    # Also add column writes from tracking if not covered by diff
    for var, cols in tracking.column_writes.items():
        for col in cols:
            locs.add(Loc.column(var, col))

    return frozenset(locs)


def _extract_column_locs_from_diff(var_name: str, diff: "CompoundDiff") -> Set[Loc]:
    """
    Extract column-level Loc objects from a DataFrame/Series CompoundDiff.

    Returns set of Loc.column() for modified columns, or empty set if
    the change is structural (rows added/removed).
    """
    import re
    from flowbook.kernel_support.types import CompoundDiff, ValueComparison

    locs: Set[Loc] = set()

    for key, child in diff.children.items():
        # Structural changes - return empty (caller will use Loc.var)
        if key in ("_structural_rows", "_structural_columns", "_structural_index", "_index"):
            return set()  # Structural change affects whole variable

        # Column key (e.g., "['price']" or '["price"]')
        match = re.match(r"\[[\'\"](.+)[\'\"]\]", key)
        if match:
            col_name = match.group(1)
            if isinstance(child, (ValueComparison, CompoundDiff)):
                locs.add(Loc.column(var_name, col_name))

    return locs


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
    SKIPPED_UPSTREAM = "skipped_upstream"  # Cell reads from wrong writer; re-run won't help
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

        For FORWARD_STALE and SKIPPED_UPSTREAM reasons with a specific location,
        replaces any existing reason of either type for the same location
        (they are mutually exclusive - only the most recent matters).
        """
        self.is_clean = False

        # FORWARD_STALE and SKIPPED_UPSTREAM are mutually exclusive for same location
        location_based_types = {ReasonType.FORWARD_STALE, ReasonType.SKIPPED_UPSTREAM}
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
        last_writer_entries: Entries in last_writer that pointed to this cell
        column_last_writer_entries: Entries in column_last_writer that pointed to this cell
    """

    cell_id: str
    reads: Optional[Set[str]]
    writes: Optional[Set[str]]
    status: Optional["CellStatus"]
    tracking_data: Optional[TrackingData]
    execution_seq: Optional[int]
    structural_reads_values: Optional[Dict[str, Dict[str, str]]]
    typed_changes: Optional[List["Change"]]
    last_writer_entries: Dict[str, str]  # loc -> cell_id (was this cell)
    column_last_writer_entries: Dict[str, Dict[str, str]]  # var -> {col -> cell_id}


