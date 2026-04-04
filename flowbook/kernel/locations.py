"""
Typed Read and Write Locations for Reproducibility Analysis.

This module defines the core location types for FlowBook's reproducibility system:
- ReadLoc: what a cell accessed during execution (5 constructors)
- WriteLoc: what a cell changed and how (5 constructors)
- ▷ (write_conflicts_read): does a write invalidate a read?

Reads and writes share the same symmetric grammar:
  Loc ::= Var(x) | Col(d, c) | Cols(d) | Rows(d) | File(p)

The conflict relation ▷ is a clean 5×5 matrix with no lookup tables.
All cells are either —, a name match, or d≡d'.

Formal ref: FORMAL_DEVELOPMENT.md §8.1, CONFLICT_RELATION.md

Usage:
    r = ReadLoc.var("x")
    w = WriteLoc.col("df", "price")
    assert not write_conflicts_read(w, ReadLoc.col("df", "qty"))  # independent columns
    assert write_conflicts_read(w, ReadLoc.cols("df"))             # structure changed
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Union

from flowbook.kernel.loc_ids import LocRef, StableIdMap, get_qualifier
from flowbook.kernel_support.models import TrackingData

logger = logging.getLogger(__name__)

# Type alias for qualifier: either a variable name string or a LocRef
Qualifier = Union[str, LocRef]


# =============================================================================
# Attribute Classification (tracking layer only)
# =============================================================================
# These classify structural attribute names into Cols/Rows domains.
# Used only in tracking_to_readlocset() — NOT in the conflict relation.

# Attributes that map to Cols(d) read — column structure
COLS_READ_ATTRS: FrozenSet[str] = frozenset({
    "columns",   # Column names Index
    "keys",      # Same as columns
    "dtypes",    # Column dtypes
    "iter",      # Iteration over DataFrame yields columns
})

# Attributes that map to Rows(d) read — row structure
ROWS_READ_ATTRS: FrozenSet[str] = frozenset({
    "index",     # Row labels
    "len",       # Number of rows
    "empty",     # Whether empty
})

# Attributes that map to BOTH Cols(d) AND Rows(d) — cross-cutting
BOTH_READ_ATTRS: FrozenSet[str] = frozenset({
    "shape",     # (rows, cols)
    "size",      # rows * cols
    "axes",      # [index, columns]
    "values",    # Full array (both dimensions)
    "T",         # Transpose (both dimensions)
    "describe",  # describe() — statistics over all columns
})


# =============================================================================
# ReadLoc — What a cell accessed
# =============================================================================


class ReadLocType(str, Enum):
    """Read location type."""
    VAR = "var"           # Var(x): whole variable
    COLUMN = "col"        # Col(d, c): DataFrame column
    COLS = "cols"         # Cols(d): column structure (names, dtypes, count)
    ROWS = "rows"         # Rows(d): row structure (index, length, count)
    FILE = "file"         # File(p): file path


@dataclass(frozen=True)
class ReadLoc:
    """
    A read location — identifies what a cell accessed during execution.

    ReadLoc ::= Var(x) | Col(d, c) | Cols(d) | Rows(d) | File(p)

    The qualifier is either a variable name (str) or a LocRef(loc_id, var_name)
    for stable DataFrame identity. LocRef qualifiers enable correct conflict
    detection for aliased DataFrames (same object, different variable names).

    For Cols(d) and Rows(d), name holds the display variable name and
    qualifier holds the DataFrame identifier (matching WriteLoc.rows pattern).

    Formal ref: FORMAL_DEVELOPMENT.md §8.1
    """
    type: ReadLocType
    name: str
    qualifier: Optional[Qualifier] = None

    @classmethod
    def var(cls, name: str) -> "ReadLoc":
        """Var(x) — whole variable read."""
        return cls(ReadLocType.VAR, name)

    @classmethod
    def col(cls, qualifier: Qualifier, column: str) -> "ReadLoc":
        """Col(d, c) — DataFrame column read."""
        return cls(ReadLocType.COLUMN, column, qualifier=qualifier)

    @classmethod
    def cols(cls, var: str, qualifier: Optional[Qualifier] = None) -> "ReadLoc":
        """Cols(d) — column structure read (columns, dtypes, etc.).

        Args:
            var: Variable name (for display and var_name())
            qualifier: DataFrame identifier (LocRef or str). Defaults to var.
        """
        return cls(ReadLocType.COLS, var, qualifier=qualifier if qualifier is not None else var)

    @classmethod
    def rows(cls, var: str, qualifier: Optional[Qualifier] = None) -> "ReadLoc":
        """Rows(d) — row structure read (index, len, etc.).

        Args:
            var: Variable name (for display and var_name())
            qualifier: DataFrame identifier (LocRef or str). Defaults to var.
        """
        return cls(ReadLocType.ROWS, var, qualifier=qualifier if qualifier is not None else var)

    @classmethod
    def file(cls, path: str) -> "ReadLoc":
        """File(p) — file read."""
        return cls(ReadLocType.FILE, path)

    def var_name(self) -> str:
        """Extract the top-level variable name."""
        if self.type in (ReadLocType.VAR, ReadLocType.FILE):
            return self.name
        if self.type in (ReadLocType.COLS, ReadLocType.ROWS):
            return self.name  # Cols/Rows store display var name in name
        # COLUMN
        if isinstance(self.qualifier, LocRef):
            return self.qualifier.var_name
        return self.qualifier

    def display_name(self) -> str:
        """Human-readable representation for UI display."""
        q = _display_qualifier(self.qualifier)
        if self.type == ReadLocType.VAR:
            return self.name
        elif self.type == ReadLocType.COLUMN:
            return f"{q}['{self.name}']"
        elif self.type == ReadLocType.COLS:
            return f"{self.name} (cols structure)"
        elif self.type == ReadLocType.ROWS:
            return f"{self.name} (rows structure)"
        elif self.type == ReadLocType.FILE:
            return f"File({self.name})"
        return str(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict for frontend metadata."""
        d: Dict[str, Any] = {"type": self.type.value, "name": self.name}
        if self.qualifier is not None:
            if isinstance(self.qualifier, LocRef):
                d["qualifier"] = self.qualifier.loc_id
                d["var_name"] = self.qualifier.var_name
            else:
                d["qualifier"] = self.qualifier
        return d

    def __str__(self) -> str:
        q = _display_qualifier(self.qualifier)
        if self.type == ReadLocType.VAR:
            return f"Var({self.name})"
        elif self.type == ReadLocType.COLUMN:
            return f"Col({q}, {self.name})"
        elif self.type == ReadLocType.COLS:
            return f"Cols({self.name})"
        elif self.type == ReadLocType.ROWS:
            return f"Rows({self.name})"
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
    COLS = "cols"            # Cols(d): column structure changed (dtypes, etc.)
    ROWS = "rows"            # Rows(d): rows added or removed
    FILE = "file"            # File(p): file written


@dataclass(frozen=True)
class WriteLoc:
    """
    A write location — identifies what a cell changed and how.

    WriteLoc ::= Var(x) | Col(d, c) | Cols(d) | Rows(d) | File(p)

    The "how" determines which reads are invalidated. This is what
    eliminates the need for a separate ConflictResolver — the conflict
    semantics are encoded in the type.

    The qualifier is either a variable name (str) or a LocRef(loc_id, var_name)
    for stable DataFrame identity. For Cols(d) and Rows(d), the qualifier
    holds the DataFrame identifier, and name holds the display variable name.

    Formal ref: FORMAL_DEVELOPMENT.md §8.2
    """
    type: WriteLocType
    name: str
    qualifier: Optional[Qualifier] = None

    @classmethod
    def var(cls, name: str) -> "WriteLoc":
        """Var(x) — variable completely replaced."""
        return cls(WriteLocType.VAR, name)

    @classmethod
    def col(cls, qualifier: Qualifier, column: str) -> "WriteLoc":
        """Col(d, c) — column values modified."""
        return cls(WriteLocType.COL, column, qualifier=qualifier)

    @classmethod
    def cols(cls, var: str, qualifier: Optional[Qualifier] = None) -> "WriteLoc":
        """Cols(d) — column structure changed (dtypes, etc.).

        Args:
            var: Variable name (for display and var_name())
            qualifier: DataFrame identifier (LocRef or str). Defaults to var.
        """
        return cls(WriteLocType.COLS, var, qualifier=qualifier if qualifier is not None else var)

    @classmethod
    def rows(cls, var: str, qualifier: Optional[Qualifier] = None) -> "WriteLoc":
        """Rows(d) — rows added or removed.

        Args:
            var: Variable name (for display and var_name())
            qualifier: DataFrame identifier (LocRef or str). Defaults to var.
        """
        return cls(WriteLocType.ROWS, var, qualifier=qualifier if qualifier is not None else var)

    @classmethod
    def file(cls, path: str) -> "WriteLoc":
        """File(p) — file written."""
        return cls(WriteLocType.FILE, path)

    def var_name(self) -> str:
        """Extract the top-level variable name.

        Used by LastWriter which operates at the variable level.
        """
        if self.type in (WriteLocType.VAR, WriteLocType.FILE):
            return self.name
        if self.type in (WriteLocType.COLS, WriteLocType.ROWS):
            return self.name  # Cols/Rows store display var name in name
        # COL
        if isinstance(self.qualifier, LocRef):
            return self.qualifier.var_name
        return self.qualifier

    def display_name(self) -> str:
        """Human-readable representation for UI display."""
        q = _display_qualifier(self.qualifier)
        if self.type == WriteLocType.VAR:
            return self.name
        elif self.type == WriteLocType.COL:
            return f"{q}['{self.name}']"
        elif self.type == WriteLocType.COLS:
            return f"{self.name} (cols changed)"
        elif self.type == WriteLocType.ROWS:
            return f"{self.name} (rows changed)"
        elif self.type == WriteLocType.FILE:
            return f"File({self.name})"
        return str(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict for frontend metadata."""
        d: Dict[str, Any] = {"type": self.type.value, "name": self.name}
        if self.qualifier is not None:
            if isinstance(self.qualifier, LocRef):
                d["qualifier"] = self.qualifier.loc_id
                d["var_name"] = self.qualifier.var_name
            else:
                d["qualifier"] = self.qualifier
        return d

    def __str__(self) -> str:
        q = _display_qualifier(self.qualifier)
        if self.type == WriteLocType.VAR:
            return f"Var({self.name})"
        elif self.type == WriteLocType.COL:
            return f"Col({q}, {self.name})"
        elif self.type == WriteLocType.COLS:
            return f"Cols({self.name})"
        elif self.type == WriteLocType.ROWS:
            return f"Rows({self.name})"
        elif self.type == WriteLocType.FILE:
            return f"File({self.name})"
        return repr(self)


# Type aliases
ReadLocSet = FrozenSet[ReadLoc]
WriteLocSet = FrozenSet[WriteLoc]


# =============================================================================
# Qualifier Comparison Helpers
# =============================================================================


def _display_qualifier(q: Optional[Qualifier]) -> str:
    """Extract display string from a qualifier (str or LocRef)."""
    if q is None:
        return "?"
    if isinstance(q, LocRef):
        return q.var_name
    return q


def _same_dataframe(a: Optional[Qualifier], b: Optional[Qualifier]) -> bool:
    """Compare two DataFrame identifiers (str or LocRef).

    If both are LocRef, compares loc_ids (same object → same id, even
    if accessed through different variable names).
    If mixed or both strings, falls back to var_name comparison.
    """
    if a is None or b is None:
        return a is b
    if isinstance(a, LocRef) and isinstance(b, LocRef):
        return a.loc_id == b.loc_id
    # Mixed or both strings — compare var names
    a_name = a.var_name if isinstance(a, LocRef) else a
    b_name = b.var_name if isinstance(b, LocRef) else b
    return a_name == b_name


# =============================================================================
# The ▷ Conflict Relation
# =============================================================================
# w ▷ r: does writing w invalidate reading r?
#
# This is the SINGLE conflict check for all reproducibility analysis.
#
# The 5×5 matrix (no lookup tables):
#
#            Var(x')  Col(d',c')  Cols(d')  Rows(d')  File(p')
# Var(x)      x=x'      —          —         —         —
# Col(d,c)     —      d≡d'∧c=c'   d≡d'       —         —
# Cols(d)      —        d≡d'       d≡d'       —         —
# Rows(d)      —        d≡d'        —        d≡d'       —
# File(p)      —         —          —         —        p=p'
#
# Var(x) only conflicts with Var(x) reads. Rebinding detection for
# Col/Cols/Rows readers works because Var(x) is always present in the
# read set alongside those reads (see tracking_to_readlocset).
#
# Qualifier comparison uses _same_dataframe() for DataFrame-to-DataFrame
# checks (compares loc_ids when LocRef is available).
# =============================================================================


def write_conflicts_read(w: WriteLoc, r: ReadLoc) -> bool:
    """
    w ▷ r — does writing w invalidate reading r?

    This is the core conflict relation. All validity predicates
    and staleness checks are defined in terms of this function.

    5×5 matrix with no lookup tables — every cell is either —,
    a name match, or d≡d'.

    Formal ref: CONFLICT_RELATION.md §Read-Write Conflict Matrix
    """
    # --- Var(x) writes: only conflict with Var(x) reads ---
    if w.type == WriteLocType.VAR:
        return r.type == ReadLocType.VAR and w.name == r.name

    # --- Col(d, c) writes: column written (may add, modify, or delete) ---
    # Invalidates: same-column reads, column-structure reads (Cols)
    elif w.type == WriteLocType.COL:
        if r.type == ReadLocType.COLUMN:
            return _same_dataframe(w.qualifier, r.qualifier) and w.name == r.name
        if r.type == ReadLocType.COLS:
            return _same_dataframe(w.qualifier, r.qualifier)
        return False

    # --- Cols(d) writes: column structure changed ---
    # Invalidates: all column reads, column-structure reads
    elif w.type == WriteLocType.COLS:
        if r.type == ReadLocType.COLUMN:
            return _same_dataframe(w.qualifier, r.qualifier)
        if r.type == ReadLocType.COLS:
            return _same_dataframe(w.qualifier, r.qualifier)
        return False

    # --- Rows(d) writes: rows added or removed ---
    # Invalidates: all column reads (data changed), row-structure reads
    elif w.type == WriteLocType.ROWS:
        if r.type == ReadLocType.COLUMN:
            return _same_dataframe(w.qualifier, r.qualifier)
        if r.type == ReadLocType.ROWS:
            return _same_dataframe(w.qualifier, r.qualifier)
        return False

    # --- File(p) writes ---
    elif w.type == WriteLocType.FILE:
        return r.type == ReadLocType.FILE and w.name == r.name

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


def write_conflicts_write(w1: WriteLoc, w2: WriteLoc) -> bool:
    """
    w1 ▷▷ w2 — does writing w1 overlap with writing w2?

    This is the direct write-write conflict relation. Cell j should become
    stale if cell i's writes overlap with cell j's writes:
      ∃ w1 ∈ W_i, w2 ∈ W_j . w1 ▷▷ w2

    The 5×5 symmetric matrix:

    | w1 ↓ \\ w2 →  | Var(x') | Col(d',c') | Cols(d') | Rows(d') | File(p') |
    |----------------|---------|------------|----------|----------|----------|
    | Var(x)         | x=x'   | —          | —        | —        | —        |
    | Col(d,c)       | —       | d≡d' ∧ c=c'| d≡d'    | d≡d'     | —        |
    | Cols(d)        | —       | d≡d'       | d≡d'     | —        | —        |
    | Rows(d)        | —       | d≡d'       | —        | d≡d'     | —        |
    | File(p)        | —       | —          | —        | —        | p=p'     |
    """
    # --- Var(x) ---
    if w1.type == WriteLocType.VAR:
        return w2.type == WriteLocType.VAR and w1.name == w2.name

    # --- Col(d, c) ---
    elif w1.type == WriteLocType.COL:
        if w2.type == WriteLocType.COL:
            return _same_dataframe(w1.qualifier, w2.qualifier) and w1.name == w2.name
        if w2.type == WriteLocType.COLS:
            return _same_dataframe(w1.qualifier, w2.qualifier)
        if w2.type == WriteLocType.ROWS:
            return _same_dataframe(w1.qualifier, w2.qualifier)
        return False

    # --- Cols(d) ---
    elif w1.type == WriteLocType.COLS:
        if w2.type == WriteLocType.COL:
            return _same_dataframe(w1.qualifier, w2.qualifier)
        if w2.type == WriteLocType.COLS:
            return _same_dataframe(w1.qualifier, w2.qualifier)
        return False

    # --- Rows(d) ---
    elif w1.type == WriteLocType.ROWS:
        if w2.type == WriteLocType.COL:
            return _same_dataframe(w1.qualifier, w2.qualifier)
        if w2.type == WriteLocType.ROWS:
            return _same_dataframe(w1.qualifier, w2.qualifier)
        return False

    # --- File(p) ---
    elif w1.type == WriteLocType.FILE:
        return w2.type == WriteLocType.FILE and w1.name == w2.name

    return False


def wlocs_conflict_wlocs(writes1: WriteLocSet, writes2: WriteLocSet) -> WriteLocSet:
    """
    W1 ▷▷ W2 — return write locs in W1 that overlap with some write in W2.

    Set-level extension of ▷▷:
      A ▷▷ B = { w1 ∈ A | ∃ w2 ∈ B . w1 ▷▷ w2 }
    """
    if not writes1 or not writes2:
        return frozenset()
    return frozenset(
        w1 for w1 in writes1
        if any(write_conflicts_write(w1, w2) for w2 in writes2)
    )


# =============================================================================
# Typed Formal Predicates
# =============================================================================
# These are the typed (column-aware) versions of the formal predicates from
# FORMAL_DEVELOPMENT.md §3.2-3.3. Each takes ReadLocSet/WriteLocSet and uses
# the ▷ relation for column-level precision.
#
# The four VALIDITY predicates (must all hold for execution to be accepted):
#   no_read_and_write, write_before_read, no_read_before_write, no_write_after_read
#
# The two STALENESS predicates (used after acceptance to propagate staleness):
#   forward_stale_reads, forward_stale_writes


def no_read_and_write(R_i: ReadLocSet, W_i: WriteLocSet) -> WriteLocSet:
    """
    NoReadAndWrite(R, W, i) — Rᵢ ∩ Wᵢ = ∅

    Returns the WriteLocs that conflict with reads in the same cell.
    Empty means predicate holds.

    Formal ref: FORMAL_DEVELOPMENT.md §3.2
    """
    return wlocs_conflict_rlocs(W_i, R_i)


def no_read_before_write(R_i: ReadLocSet, W_after: WriteLocSet) -> WriteLocSet:
    """
    NoReadBeforeWrite(R, W, i) — Rᵢ ∩ W_{i+1..n} = ∅

    Returns the WriteLocs from later cells that conflict with cell i's reads.
    Empty means predicate holds (no forward contamination).

    Formal ref: FORMAL_DEVELOPMENT.md §3.2
    """
    return wlocs_conflict_rlocs(W_after, R_i)


def no_write_after_read(W_i: WriteLocSet, R_before: ReadLocSet) -> WriteLocSet:
    """
    NoWriteAfterRead(R, W, i) — Wᵢ ∩ R_{1..i-1} = ∅

    Returns the WriteLocs from cell i that conflict with earlier cells' reads.
    Empty means predicate holds (no backward mutation).

    Formal ref: FORMAL_DEVELOPMENT.md §3.2
    """
    return wlocs_conflict_rlocs(W_i, R_before)


def forward_stale_reads(W_i: WriteLocSet, R_j: ReadLocSet) -> WriteLocSet:
    """
    ForwardStale read overlap: W'ᵢ ▷ Rⱼ

    Returns the WriteLocs from cell i that invalidate cell j's reads.
    Non-empty means cell j should become stale.

    Formal ref: FORMAL_DEVELOPMENT.md §3.3
    """
    return wlocs_conflict_rlocs(W_i, R_j)


def forward_stale_writes(W_i: WriteLocSet, W_j: WriteLocSet) -> WriteLocSet:
    """
    ForwardStale write overlap: W'ᵢ ▷▷ Wⱼ

    Returns the WriteLocs from cell i that overlap with cell j's writes.
    Non-empty means cell j should become stale (write-write overlap).

    Formal ref: FORMAL_DEVELOPMENT.md §3.3, CONFLICT_RELATION.md §Write-Write Conflict
    """
    return wlocs_conflict_wlocs(W_i, W_j)


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
    """Group Col read locs by variable name → [column names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type == ReadLocType.COLUMN and loc.qualifier:
            key = _display_qualifier(loc.qualifier)
            result.setdefault(key, []).append(loc.name)
    return {k: sorted(v) for k, v in result.items()}


def writelocset_to_column_map(locs: WriteLocSet) -> Dict[str, List[str]]:
    """Group Col write locs by variable name → [column names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type == WriteLocType.COL and loc.qualifier:
            key = _display_qualifier(loc.qualifier)
            result.setdefault(key, []).append(loc.name)
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
        key=lambda d: (d["type"], str(d.get("qualifier", "")), str(d["name"])),
    )


def writelocset_to_list(locs: WriteLocSet) -> List[Dict[str, str]]:
    """Serialize WriteLocSet to sorted list of dicts for frontend metadata."""
    return sorted(
        (loc.to_dict() for loc in locs),
        key=lambda d: (d["type"], str(d.get("qualifier", "")), str(d["name"])),
    )


# =============================================================================
# Conversion from TrackingData
# =============================================================================


def tracking_to_readlocset(
    tracking: TrackingData,
    namespace: Optional[dict] = None,
    stable_map: Optional[StableIdMap] = None,
) -> ReadLocSet:
    """
    Convert TrackingData reads to ReadLocSet.

    This creates the unified read set Rᵢ from runtime tracking data:
    - Variable reads → Var(x) (always emitted for every read variable)
    - Column reads → Col(d, c) with LocRef qualifier when stable_map available
    - Structural reads → Cols(d) / Rows(d) classified from attribute names
    - File reads → File(p)

    Structural attribute names are classified into Cols/Rows domains:
    - COLS_READ_ATTRS (columns, keys, dtypes, iter) → Cols(d)
    - ROWS_READ_ATTRS (index, len, empty) → Rows(d)
    - BOTH_READ_ATTRS (shape, size, axes, values, T, describe) → both
    - Unknown attributes → both (conservative) with warning

    Var(x) is always emitted alongside Col/Cols/Rows reads. This ensures
    variable rebinding is caught by the simple Var(x) ▷ Var(x) = true
    check, without needing a cross-domain bridge rule. Column independence
    is preserved because Col/Cols/Rows ▷ Var = false in the ▷ matrix.

    Args:
        tracking: TrackingData from cell execution
        namespace: Current kernel namespace (optional, for LocRef qualifiers)
        stable_map: StableIdMap instance (optional, for LocRef qualifiers)
    """
    locs: Set[ReadLoc] = set()

    for var, col_set in (tracking.column_reads_before_writes or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        for c in col_set:
            locs.add(ReadLoc.col(q, c))

    for var, attrs in (tracking.structural_reads or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        needs_cols = False
        needs_rows = False
        for attr in attrs:
            if attr in COLS_READ_ATTRS:
                needs_cols = True
            elif attr in ROWS_READ_ATTRS:
                needs_rows = True
            elif attr in BOTH_READ_ATTRS:
                needs_cols = True
                needs_rows = True
            else:
                # Unknown attribute — conservatively map to both
                logger.warning(
                    "Unknown structural attribute %r on %r — mapping to both Cols and Rows",
                    attr, var,
                )
                needs_cols = True
                needs_rows = True
        if needs_cols:
            locs.add(ReadLoc.cols(var, qualifier=q))
        if needs_rows:
            locs.add(ReadLoc.rows(var, qualifier=q))

    # Always emit Var(x) for every variable in reads_before_writes.
    # Var(x) captures the binding read; Col/Cols/Rows locs capture finer detail.
    # Rebinding is caught by Var(x) ▷ Var(x); column independence is
    # preserved because Col/Cols/Rows ▷ Var = false in the ▷ matrix.
    for var in (tracking.reads_before_writes or set()):
        locs.add(ReadLoc.var(var))

    for path in (tracking.file_reads_before_writes or set()):
        locs.add(ReadLoc.file(path))

    return frozenset(locs)


def tracking_to_writelocset(
    tracking: TrackingData,
    namespace: Optional[dict] = None,
    stable_map: Optional[StableIdMap] = None,
) -> WriteLocSet:
    """
    Convert TrackingData writes to WriteLocSet.

    This function is used when no diff is available (e.g., for the basic
    write set before diff refinement).

    Args:
        tracking: TrackingData from cell execution
        namespace: Current kernel namespace (optional, for LocRef qualifiers)
        stable_map: StableIdMap instance (optional, for LocRef qualifiers)
    """
    locs: Set[WriteLoc] = set()

    for var in (tracking.writes or set()):
        locs.add(WriteLoc.var(var))

    for var, col_set in (tracking.column_writes or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        for c in col_set:
            locs.add(WriteLoc.col(q, c))

    # Structural mutations (recorded at operation time by monkey patches)
    for var in (tracking.row_mutations or set()):
        q = get_qualifier(var, namespace, stable_map)
        locs.add(WriteLoc.rows(var, qualifier=q))

    for var in (tracking.index_mutations or set()):
        q = get_qualifier(var, namespace, stable_map)
        locs.add(WriteLoc.rows(var, qualifier=q))

    for var in (tracking.dtype_changes or {}):
        q = get_qualifier(var, namespace, stable_map)
        locs.add(WriteLoc.cols(var, qualifier=q))

    for var, col_set in (tracking.column_deletions or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        for c in col_set:
            locs.add(WriteLoc.col(q, c))

    for path in (tracking.file_writes or set()):
        locs.add(WriteLoc.file(path))

    return frozenset(locs)
