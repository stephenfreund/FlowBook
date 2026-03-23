"""
Typed Read and Write Locations for Reproducibility Analysis.

This module defines the core location types for FlowBook's reproducibility system:
- ReadLoc: what a cell accessed during execution (4 constructors)
- WriteLoc: what a cell changed and how (7 constructors)
- ▷ (write_conflicts_read): does a write invalidate a read?

The key design insight: reads and writes are different types. Reads describe
"what did I look at?" while writes describe "what did I change and how?"
The "how" (modify vs add vs delete vs rows changed) determines which reads
are invalidated.

Formal ref: LOCSET_UNIFICATION_PLAN.md, FORMAL_DEVELOPMENT.md §8.1

Usage:
    r = ReadLoc.var("x")
    w = WriteLoc.col_add("df", "price")
    assert not write_conflicts_read(w, ReadLoc.col("df", "qty"))  # independent columns
    assert write_conflicts_read(w, ReadLoc.attr("df", "columns")) # structure changed
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set

from flowbook.kernel_support.models import TrackingData


# =============================================================================
# Attribute Constants
# =============================================================================
# These define which DataFrame attributes are affected by structural changes.

# Attributes that reveal column structure
COL_ATTRS: FrozenSet[str] = frozenset({
    "columns",   # Column names Index
    "keys",      # Same as columns
    "dtypes",    # Column dtypes
    "axes",      # [index, columns]
    "T",         # Transpose (exposes columns)
    "values",    # Full array (shape visible)
    "iter",      # Iteration over DataFrame yields columns
    "describe",  # describe() includes all columns
    "shape",     # (rows, cols)
    "size",      # rows * cols
})

# Attributes that reveal row structure
ROW_ATTRS: FrozenSet[str] = frozenset({
    "index",     # Row labels
    "shape",     # (rows, cols)
    "size",      # rows * cols
    "len",       # Number of rows
    "empty",     # Whether empty
    "axes",      # [index, columns] — index is a component
    "values",    # Full array — row count affects shape
    "T",         # Transpose of values
})


# =============================================================================
# ReadLoc — What a cell accessed
# =============================================================================


class ReadLocType(str, Enum):
    """Read location type."""
    VAR = "var"           # Var(x): whole variable
    COLUMN = "column"     # Col(d, c): DataFrame column
    ATTR = "attr"         # Attr(d, a): DataFrame attribute (shape, columns, etc.)
    FILE = "file"         # File(p): file path


@dataclass(frozen=True)
class ReadLoc:
    """
    A read location — identifies what a cell accessed during execution.

    ReadLoc ::= Var(x) | Col(d, c) | Attr(d, a) | File(p)

    Formal ref: LOCSET_UNIFICATION_PLAN.md §Read Location Grammar
    """
    type: ReadLocType
    name: str
    qualifier: Optional[str] = None

    @classmethod
    def var(cls, name: str) -> "ReadLoc":
        """Var(x) — whole variable read."""
        return cls(ReadLocType.VAR, name)

    @classmethod
    def col(cls, var: str, column: str) -> "ReadLoc":
        """Col(d, c) — DataFrame column read."""
        return cls(ReadLocType.COLUMN, column, qualifier=var)

    @classmethod
    def attr(cls, var: str, attribute: str) -> "ReadLoc":
        """Attr(d, a) — DataFrame attribute read (shape, columns, etc.)."""
        return cls(ReadLocType.ATTR, attribute, qualifier=var)

    @classmethod
    def file(cls, path: str) -> "ReadLoc":
        """File(p) — file read."""
        return cls(ReadLocType.FILE, path)

    def var_name(self) -> str:
        """Extract the top-level variable name."""
        if self.type in (ReadLocType.VAR, ReadLocType.FILE):
            return self.name
        return self.qualifier  # COLUMN, ATTR

    def display_name(self) -> str:
        """Human-readable representation for UI display."""
        if self.type == ReadLocType.VAR:
            return self.name
        elif self.type == ReadLocType.COLUMN:
            return f"{self.qualifier}['{self.name}']"
        elif self.type == ReadLocType.ATTR:
            return f"{self.qualifier}.{self.name}"
        elif self.type == ReadLocType.FILE:
            return f"File({self.name})"
        return str(self)

    def to_dict(self) -> Dict[str, str]:
        """Serialize to JSON-friendly dict for frontend metadata."""
        d: Dict[str, str] = {"type": self.type.value, "name": self.name}
        if self.qualifier is not None:
            d["qualifier"] = self.qualifier
        return d

    def __str__(self) -> str:
        if self.type == ReadLocType.VAR:
            return f"Var({self.name})"
        elif self.type == ReadLocType.COLUMN:
            return f"Col({self.qualifier}, {self.name})"
        elif self.type == ReadLocType.ATTR:
            return f"Attr({self.qualifier}, {self.name})"
        elif self.type == ReadLocType.FILE:
            return f"File({self.name})"
        return repr(self)


# =============================================================================
# WriteLoc — What a cell changed and how
# =============================================================================


class WriteLocType(str, Enum):
    """Write location type — encodes HOW a location changed."""
    VAR = "var"              # Var(x): variable completely replaced
    COL = "col"              # Col(d, c): column values modified
    COL_ADD = "col_add"      # ColAdd(d, c): new column added
    COL_DEL = "col_del"      # ColDel(d, c): column removed
    ROWS = "rows"            # Rows(d): rows added or removed
    ATTR = "attr"            # Attr(d, a): attribute changed
    FILE = "file"            # File(p): file written


@dataclass(frozen=True)
class WriteLoc:
    """
    A write location — identifies what a cell changed and how.

    WriteLoc ::= Var(x) | Col(d, c) | ColAdd(d, c) | ColDel(d, c)
               | Rows(d) | Attr(d, a) | File(p)

    The "how" (modify vs add vs delete vs rows) determines which reads
    are invalidated. This is what eliminates the need for a separate
    ConflictResolver — the conflict semantics are encoded in the type.

    Formal ref: LOCSET_UNIFICATION_PLAN.md §Write Location Grammar
    """
    type: WriteLocType
    name: str
    qualifier: Optional[str] = None

    @classmethod
    def var(cls, name: str) -> "WriteLoc":
        """Var(x) — variable completely replaced."""
        return cls(WriteLocType.VAR, name)

    @classmethod
    def col(cls, var: str, column: str) -> "WriteLoc":
        """Col(d, c) — column values modified."""
        return cls(WriteLocType.COL, column, qualifier=var)

    @classmethod
    def col_add(cls, var: str, column: str) -> "WriteLoc":
        """ColAdd(d, c) — new column added."""
        return cls(WriteLocType.COL_ADD, column, qualifier=var)

    @classmethod
    def col_del(cls, var: str, column: str) -> "WriteLoc":
        """ColDel(d, c) — column removed."""
        return cls(WriteLocType.COL_DEL, column, qualifier=var)

    @classmethod
    def rows(cls, var: str) -> "WriteLoc":
        """Rows(d) — rows added or removed."""
        return cls(WriteLocType.ROWS, var)

    @classmethod
    def attr(cls, var: str, attribute: str) -> "WriteLoc":
        """Attr(d, a) — attribute changed (index, dtypes, etc.)."""
        return cls(WriteLocType.ATTR, attribute, qualifier=var)

    @classmethod
    def file(cls, path: str) -> "WriteLoc":
        """File(p) — file written."""
        return cls(WriteLocType.FILE, path)

    def var_name(self) -> str:
        """Extract the top-level variable name.

        Used by LastWriter which operates at the variable level.
        """
        if self.type in (WriteLocType.VAR, WriteLocType.ROWS, WriteLocType.FILE):
            return self.name
        return self.qualifier  # COL, COL_ADD, COL_DEL, ATTR

    def output(self) -> ReadLoc:
        """Convert to the ReadLoc that would observe this write's value.

        Used for write-write overlap in ForwardStale:
          (Wᵢ ∪ W'ᵢ) ▷ output*(Wⱼ) ≠ ∅

        Formal ref: LOCSET_UNIFICATION_PLAN.md §The Output Function
        """
        if self.type == WriteLocType.VAR:
            return ReadLoc.var(self.name)
        elif self.type in (WriteLocType.COL, WriteLocType.COL_ADD, WriteLocType.COL_DEL):
            return ReadLoc.col(self.qualifier, self.name)
        elif self.type == WriteLocType.ROWS:
            return ReadLoc.var(self.name)
        elif self.type == WriteLocType.ATTR:
            return ReadLoc.attr(self.qualifier, self.name)
        elif self.type == WriteLocType.FILE:
            return ReadLoc.file(self.name)
        raise ValueError(f"Unknown write type: {self.type}")

    def display_name(self) -> str:
        """Human-readable representation for UI display."""
        if self.type == WriteLocType.VAR:
            return self.name
        elif self.type == WriteLocType.COL:
            return f"{self.qualifier}['{self.name}']"
        elif self.type == WriteLocType.COL_ADD:
            return f"{self.qualifier}['{self.name}'] (added)"
        elif self.type == WriteLocType.COL_DEL:
            return f"{self.qualifier}['{self.name}'] (removed)"
        elif self.type == WriteLocType.ROWS:
            return f"{self.name} (rows changed)"
        elif self.type == WriteLocType.ATTR:
            return f"{self.qualifier}.{self.name}"
        elif self.type == WriteLocType.FILE:
            return f"File({self.name})"
        return str(self)

    def to_dict(self) -> Dict[str, str]:
        """Serialize to JSON-friendly dict for frontend metadata."""
        d: Dict[str, str] = {"type": self.type.value, "name": self.name}
        if self.qualifier is not None:
            d["qualifier"] = self.qualifier
        return d

    def __str__(self) -> str:
        if self.type == WriteLocType.VAR:
            return f"Var({self.name})"
        elif self.type == WriteLocType.COL:
            return f"Col({self.qualifier}, {self.name})"
        elif self.type == WriteLocType.COL_ADD:
            return f"ColAdd({self.qualifier}, {self.name})"
        elif self.type == WriteLocType.COL_DEL:
            return f"ColDel({self.qualifier}, {self.name})"
        elif self.type == WriteLocType.ROWS:
            return f"Rows({self.name})"
        elif self.type == WriteLocType.ATTR:
            return f"Attr({self.qualifier}, {self.name})"
        elif self.type == WriteLocType.FILE:
            return f"File({self.name})"
        return repr(self)


# Type aliases
ReadLocSet = FrozenSet[ReadLoc]
WriteLocSet = FrozenSet[WriteLoc]


# =============================================================================
# The ▷ Conflict Relation
# =============================================================================
# w ▷ r: does writing w invalidate reading r?
#
# This is the SINGLE conflict check for all reproducibility analysis.
# Supersedes the old ConflictResolver + CONFLICT_RULES table.
#
# The "how" is encoded in the WriteLoc type constructor.
# The matrix has 7 write types × 4 read types = 28 cells.
# =============================================================================


def write_conflicts_read(w: WriteLoc, r: ReadLoc) -> bool:
    """
    w ▷ r — does writing w invalidate reading r?

    This is the core conflict relation. All validity predicates
    and staleness checks are defined in terms of this function.

    Formal ref: LOCSET_UNIFICATION_PLAN.md §The Conflict Relation: ▷
    """
    # --- Var(x) writes: invalidate any read involving x ---
    if w.type == WriteLocType.VAR:
        if r.type == ReadLocType.VAR:
            return w.name == r.name
        elif r.type in (ReadLocType.COLUMN, ReadLocType.ATTR):
            return w.name == r.qualifier
        return False  # Var vs File: no conflict

    # --- Col(d, c) writes: column values modified ---
    # Invalidates: same-column reads
    # Does NOT invalidate: Var reads (binding unchanged), attribute reads (values ≠ structure)
    elif w.type == WriteLocType.COL:
        if r.type == ReadLocType.COLUMN:
            return w.qualifier == r.qualifier and w.name == r.name
        return False

    # --- ColAdd(d, c) writes: new column added ---
    # Invalidates: column-structure attribute reads
    # Does NOT invalidate: Var reads (binding unchanged), existing column reads
    elif w.type == WriteLocType.COL_ADD:
        if r.type == ReadLocType.ATTR:
            return w.qualifier == r.qualifier and r.name in COL_ATTRS
        return False

    # --- ColDel(d, c) writes: column removed ---
    # Invalidates: that column's reads, column-structure attrs
    # Does NOT invalidate: Var reads (binding unchanged)
    elif w.type == WriteLocType.COL_DEL:
        if r.type == ReadLocType.COLUMN:
            return w.qualifier == r.qualifier and w.name == r.name
        elif r.type == ReadLocType.ATTR:
            return w.qualifier == r.qualifier and r.name in COL_ATTRS
        return False

    # --- Rows(d) writes: rows added or removed ---
    # Invalidates: all column reads (data changed), row-structure attrs
    # Does NOT invalidate: Var reads (binding unchanged)
    elif w.type == WriteLocType.ROWS:
        if r.type == ReadLocType.COLUMN:
            return w.name == r.qualifier
        elif r.type == ReadLocType.ATTR:
            return w.name == r.qualifier and r.name in ROW_ATTRS
        return False

    # --- Attr(d, a) writes: attribute changed ---
    # Invalidates: same attribute reads
    # Does NOT invalidate: Var reads (binding unchanged), column value reads
    elif w.type == WriteLocType.ATTR:
        if r.type == ReadLocType.ATTR:
            return w.qualifier == r.qualifier and w.name == r.name
        return False

    # --- File(p) writes ---
    elif w.type == WriteLocType.FILE:
        if r.type == ReadLocType.FILE:
            return w.name == r.name
        return False

    return False


# =============================================================================
# Set-Level Operations
# =============================================================================


def wlocs_conflict_rlocs(writes: WriteLocSet, reads: ReadLocSet) -> WriteLocSet:
    """
    W ▷ R — return write locs in W that conflict with some read in R.

    This is the set-level extension of ▷:
      A ▷ B = { w ∈ A | ∃ r ∈ B . w ▷ r }
    """
    if not writes or not reads:
        return frozenset()
    return frozenset(
        w for w in writes
        if any(write_conflicts_read(w, r) for r in reads)
    )


def has_conflict(writes: WriteLocSet, reads: ReadLocSet) -> bool:
    """
    W ▷ R ≠ ∅ — quick boolean check for any conflict.

    Short-circuits on first conflict found.
    """
    return any(
        write_conflicts_read(w, r)
        for w in writes for r in reads
    )


def output_set(writes: WriteLocSet) -> ReadLocSet:
    """
    output*(W) = { output(w) | w ∈ W }

    Convert writes to the reads they produce.
    Used for write-write overlap in ForwardStale.
    """
    return frozenset(w.output() for w in writes)


# =============================================================================
# Extraction Helpers (for metadata output)
# =============================================================================


def readlocset_var_names(locs: ReadLocSet) -> Set[str]:
    """Extract top-level variable names from a ReadLocSet."""
    return {r.var_name() for r in locs}


def writelocset_var_names(locs: WriteLocSet) -> Set[str]:
    """Extract top-level variable names from a WriteLocSet."""
    return {w.var_name() for w in locs}


def readlocset_to_column_map(locs: ReadLocSet) -> Dict[str, List[str]]:
    """Group Col read locs by variable → [column names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type == ReadLocType.COLUMN and loc.qualifier:
            result.setdefault(loc.qualifier, []).append(loc.name)
    return {k: sorted(v) for k, v in result.items()}


def writelocset_to_column_map(locs: WriteLocSet) -> Dict[str, List[str]]:
    """Group Col/ColAdd/ColDel write locs by variable → [column names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type in (WriteLocType.COL, WriteLocType.COL_ADD, WriteLocType.COL_DEL) and loc.qualifier:
            result.setdefault(loc.qualifier, []).append(loc.name)
    return {k: sorted(v) for k, v in result.items()}


def readlocset_to_attr_map(locs: ReadLocSet) -> Dict[str, List[str]]:
    """Group Attr read locs by variable → [attribute names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type == ReadLocType.ATTR and loc.qualifier:
            result.setdefault(loc.qualifier, []).append(loc.name)
    return {k: sorted(v) for k, v in result.items()}


def readlocset_to_file_list(locs: ReadLocSet) -> List[str]:
    """Extract file paths from a ReadLocSet."""
    return sorted(loc.name for loc in locs if loc.type == ReadLocType.FILE)


def writelocset_to_file_list(locs: WriteLocSet) -> List[str]:
    """Extract file paths from a WriteLocSet."""
    return sorted(loc.name for loc in locs if loc.type == WriteLocType.FILE)


def readlocset_to_list(locs: ReadLocSet) -> List[Dict[str, str]]:
    """Serialize ReadLocSet to sorted list of dicts for frontend metadata."""
    return sorted(
        (loc.to_dict() for loc in locs),
        key=lambda d: (d["type"], d.get("qualifier", ""), d["name"]),
    )


def writelocset_to_list(locs: WriteLocSet) -> List[Dict[str, str]]:
    """Serialize WriteLocSet to sorted list of dicts for frontend metadata."""
    return sorted(
        (loc.to_dict() for loc in locs),
        key=lambda d: (d["type"], d.get("qualifier", ""), d["name"]),
    )


# =============================================================================
# Conversion from TrackingData
# =============================================================================


def tracking_to_readlocset(tracking: TrackingData) -> ReadLocSet:
    """
    Convert TrackingData reads to ReadLocSet.

    This creates the unified read set Rᵢ from runtime tracking data:
    - Variable reads → Var(x) (only for variables WITHOUT column/structural detail)
    - Column reads → Col(d, c)
    - Structural reads → Attr(d, a)
    - File reads → File(p)

    Variables that have column-level or structural-level read detail are NOT
    included as Var(x) reads, since the finer-grained locs already capture
    their read footprint. This matches the semantics of TrackingData.to_read_events().
    """
    locs: Set[ReadLoc] = set()

    # Track variables with finer-grained read info
    vars_with_detail: Set[str] = set()

    for var, cols in (tracking.column_reads_before_writes or {}).items():
        vars_with_detail.add(var)
        for col in cols:
            locs.add(ReadLoc.col(var, col))

    for var, attrs in (tracking.structural_reads or {}).items():
        vars_with_detail.add(var)
        for attr in attrs:
            locs.add(ReadLoc.attr(var, attr))

    # Only emit Var(x) for variables without column/structural detail
    for var in (tracking.reads_before_writes or set()):
        if var not in vars_with_detail:
            locs.add(ReadLoc.var(var))

    for path in (tracking.file_reads_before_writes or set()):
        locs.add(ReadLoc.file(path))

    return frozenset(locs)


def tracking_to_writelocset(tracking: TrackingData) -> WriteLocSet:
    """
    Convert TrackingData writes to WriteLocSet.

    Note: TrackingData doesn't distinguish ColAdd/ColDel/Rows/Attr.
    All column writes are treated as Col (modification). The diff-based
    detect_write_locs() function produces the full typed WriteLocs.

    This function is used when no diff is available (e.g., for the basic
    write set before diff refinement).
    """
    locs: Set[WriteLoc] = set()

    for var in (tracking.writes or set()):
        locs.add(WriteLoc.var(var))

    for var, cols in (tracking.column_writes or {}).items():
        for col in cols:
            locs.add(WriteLoc.col(var, col))

    for path in (tracking.file_writes or set()):
        locs.add(WriteLoc.file(path))

    return frozenset(locs)
