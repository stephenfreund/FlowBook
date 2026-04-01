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
from typing import Any, Dict, FrozenSet, List, Optional, Set, Union

from flowbook.kernel.loc_ids import LocRef, StableIdMap, get_qualifier
from flowbook.kernel_support.models import TrackingData

# Type alias for qualifier: either a variable name string or a LocRef
Qualifier = Union[str, LocRef]


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

# Attributes that depend on column DATA values (not just structure).
# Col(d, c) writes invalidate these because modifying column values
# changes the data these attributes expose.
COL_VALUE_ATTRS: FrozenSet[str] = frozenset({
    "values",    # df.values — 2D array of all column data
    "T",         # df.T — transpose exposes all column data
    "describe",  # df.describe() — statistics computed from column values
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
    COLUMN = "col"        # Col(d, c): DataFrame column
    ATTR = "attr"         # Attr(d, a): DataFrame attribute (shape, columns, etc.)
    FILE = "file"         # File(p): file path


@dataclass(frozen=True)
class ReadLoc:
    """
    A read location — identifies what a cell accessed during execution.

    ReadLoc ::= Var(x) | Col(d, c) | Attr(d, a) | File(p)

    The qualifier is either a variable name (str) or a LocRef(loc_id, var_name)
    for stable DataFrame identity. LocRef qualifiers enable correct conflict
    detection for aliased DataFrames (same object, different variable names).

    Formal ref: LOCSET_UNIFICATION_PLAN.md §Read Location Grammar
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
    def attr(cls, qualifier: Qualifier, attribute: str) -> "ReadLoc":
        """Attr(d, a) — DataFrame attribute read (shape, columns, etc.)."""
        return cls(ReadLocType.ATTR, attribute, qualifier=qualifier)

    @classmethod
    def file(cls, path: str) -> "ReadLoc":
        """File(p) — file read."""
        return cls(ReadLocType.FILE, path)

    def var_name(self) -> str:
        """Extract the top-level variable name."""
        if self.type in (ReadLocType.VAR, ReadLocType.FILE):
            return self.name
        if isinstance(self.qualifier, LocRef):
            return self.qualifier.var_name
        return self.qualifier  # COLUMN, ATTR with str qualifier

    def display_name(self) -> str:
        """Human-readable representation for UI display."""
        q = _display_qualifier(self.qualifier)
        if self.type == ReadLocType.VAR:
            return self.name
        elif self.type == ReadLocType.COLUMN:
            return f"{q}['{self.name}']"
        elif self.type == ReadLocType.ATTR:
            return f"{q}.{self.name}"
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
        elif self.type == ReadLocType.ATTR:
            return f"Attr({q}, {self.name})"
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

    The qualifier is either a variable name (str) or a LocRef(loc_id, var_name)
    for stable DataFrame identity. For Rows(d), the qualifier holds the
    DataFrame identifier (same as Col/ColAdd/ColDel/Attr), and name holds
    the display variable name.

    Formal ref: LOCSET_UNIFICATION_PLAN.md §Write Location Grammar
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
    def col_add(cls, qualifier: Qualifier, column: str) -> "WriteLoc":
        """ColAdd(d, c) — new column added."""
        return cls(WriteLocType.COL_ADD, column, qualifier=qualifier)

    @classmethod
    def col_del(cls, qualifier: Qualifier, column: str) -> "WriteLoc":
        """ColDel(d, c) — column removed."""
        return cls(WriteLocType.COL_DEL, column, qualifier=qualifier)

    @classmethod
    def rows(cls, var: str, qualifier: Optional[Qualifier] = None) -> "WriteLoc":
        """Rows(d) — rows added or removed.

        Args:
            var: Variable name (for display and var_name())
            qualifier: DataFrame identifier (LocRef or str). Defaults to var.
        """
        return cls(WriteLocType.ROWS, var, qualifier=qualifier if qualifier is not None else var)

    @classmethod
    def attr(cls, qualifier: Qualifier, attribute: str) -> "WriteLoc":
        """Attr(d, a) — attribute changed (index, dtypes, etc.)."""
        return cls(WriteLocType.ATTR, attribute, qualifier=qualifier)

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
        if self.type == WriteLocType.ROWS:
            return self.name  # Rows stores display var name in name
        # COL, COL_ADD, COL_DEL, ATTR
        if isinstance(self.qualifier, LocRef):
            return self.qualifier.var_name
        return self.qualifier

    def output(self) -> FrozenSet[ReadLoc]:
        """Return the ReadLocs that would observe this write's effect.

        Used for write-write overlap in ForwardStale:
          (Wᵢ ∪ W'ᵢ) ▷ output*(Wⱼ) ≠ ∅

        Each write type returns exactly the reads it would conflict with,
        so W ▷ output(W') correctly detects write-write overlap.

        Formal ref: CONFLICT_RELATION.md §The output Function
        """
        if self.type == WriteLocType.VAR:
            return frozenset({ReadLoc.var(self.name)})
        elif self.type == WriteLocType.COL:
            # Just the column itself — no COL_VALUE_ATTRS inflation.
            # Including attrs like 'values'/'T' would create false
            # write-write overlap between independent column writes
            # (Col(d,"price") vs Col(d,"qty")), breaking column independence.
            # Rows↔Col overlap is unaffected: Rows ▷ Col is True in ▷,
            # and Col ▷ output(Rows) works via ROW_ATTRS ∩ CVA.
            return frozenset({ReadLoc.col(self.qualifier, self.name)})
        elif self.type == WriteLocType.COL_ADD:
            # ColAdd conflicts with Attr(d, a) for a ∈ COL_ATTRS
            return frozenset(
                ReadLoc.attr(self.qualifier, a) for a in COL_ATTRS
            )
        elif self.type == WriteLocType.COL_DEL:
            # ColDel conflicts with Col(d, c) and Attr(d, a) for a ∈ COL_ATTRS
            return frozenset(
                {ReadLoc.col(self.qualifier, self.name)}
                | {ReadLoc.attr(self.qualifier, a) for a in COL_ATTRS}
            )
        elif self.type == WriteLocType.ROWS:
            # Rows conflicts with Attr(d, a) for a ∈ ROW_ATTRS
            # qualifier holds the DataFrame identifier (LocRef or str)
            return frozenset(
                ReadLoc.attr(self.qualifier, a) for a in ROW_ATTRS
            )
        elif self.type == WriteLocType.ATTR:
            return frozenset({ReadLoc.attr(self.qualifier, self.name)})
        elif self.type == WriteLocType.FILE:
            return frozenset({ReadLoc.file(self.name)})
        raise ValueError(f"Unknown write type: {self.type}")

    def display_name(self) -> str:
        """Human-readable representation for UI display."""
        q = _display_qualifier(self.qualifier)
        if self.type == WriteLocType.VAR:
            return self.name
        elif self.type == WriteLocType.COL:
            return f"{q}['{self.name}']"
        elif self.type == WriteLocType.COL_ADD:
            return f"{q}['{self.name}'] (added)"
        elif self.type == WriteLocType.COL_DEL:
            return f"{q}['{self.name}'] (removed)"
        elif self.type == WriteLocType.ROWS:
            return f"{self.name} (rows changed)"
        elif self.type == WriteLocType.ATTR:
            return f"{q}.{self.name}"
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
        elif self.type == WriteLocType.COL_ADD:
            return f"ColAdd({q}, {self.name})"
        elif self.type == WriteLocType.COL_DEL:
            return f"ColDel({q}, {self.name})"
        elif self.type == WriteLocType.ROWS:
            return f"Rows({self.name})"
        elif self.type == WriteLocType.ATTR:
            return f"Attr({q}, {self.name})"
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
# Supersedes the old ConflictResolver + CONFLICT_RULES table.
#
# The "how" is encoded in the WriteLoc type constructor.
# The matrix has 7 write types × 4 read types = 28 cells.
#
# Var(x) only conflicts with Var(x) reads. Rebinding detection for
# column/attr readers works because Var(x) is always present in the
# read set alongside Col/Attr reads (see tracking_to_readlocset).
#
# Qualifier comparison uses _same_dataframe() for DataFrame-to-DataFrame
# checks (compares loc_ids when LocRef is available).
# =============================================================================


def write_conflicts_read(w: WriteLoc, r: ReadLoc) -> bool:
    """
    w ▷ r — does writing w invalidate reading r?

    This is the core conflict relation. All validity predicates
    and staleness checks are defined in terms of this function.

    Formal ref: LOCSET_UNIFICATION_PLAN.md §The Conflict Relation: ▷
    """
    # --- Var(x) writes: only conflict with Var(x) reads ---
    # Rebinding detection for Col/Attr readers is handled by always
    # including Var(x) in read sets alongside Col/Attr reads.
    if w.type == WriteLocType.VAR:
        return r.type == ReadLocType.VAR and w.name == r.name

    # --- Col(d, c) writes: column written (may add or modify) ---
    # Invalidates: same-column reads on same DataFrame,
    #              all column-related attrs on same DataFrame (shape, columns, dtypes, etc.)
    # Does NOT invalidate: Var reads (binding unchanged)
    elif w.type == WriteLocType.COL:
        if r.type == ReadLocType.COLUMN:
            return _same_dataframe(w.qualifier, r.qualifier) and w.name == r.name
        if r.type == ReadLocType.ATTR:
            return _same_dataframe(w.qualifier, r.qualifier) and r.name in COL_ATTRS
        return False

    # --- ColAdd(d, c) writes: new column added ---
    # Invalidates: column-structure attribute reads on same DataFrame
    # Does NOT invalidate: Var reads (binding unchanged), existing column reads
    elif w.type == WriteLocType.COL_ADD:
        if r.type == ReadLocType.ATTR:
            return _same_dataframe(w.qualifier, r.qualifier) and r.name in COL_ATTRS
        return False

    # --- ColDel(d, c) writes: column removed ---
    # Invalidates: that column's reads, column-structure attrs on same DataFrame
    # Does NOT invalidate: Var reads (binding unchanged)
    elif w.type == WriteLocType.COL_DEL:
        if r.type == ReadLocType.COLUMN:
            return _same_dataframe(w.qualifier, r.qualifier) and w.name == r.name
        elif r.type == ReadLocType.ATTR:
            return _same_dataframe(w.qualifier, r.qualifier) and r.name in COL_ATTRS
        return False

    # --- Rows(d) writes: rows added or removed ---
    # Invalidates: all column reads (data changed), row-structure attrs on same DataFrame
    # Does NOT invalidate: Var reads (binding unchanged)
    elif w.type == WriteLocType.ROWS:
        if r.type == ReadLocType.COLUMN:
            return _same_dataframe(w.qualifier, r.qualifier)
        elif r.type == ReadLocType.ATTR:
            return _same_dataframe(w.qualifier, r.qualifier) and r.name in ROW_ATTRS
        return False

    # --- Attr(d, a) writes: attribute changed ---
    # Invalidates: same attribute reads on same DataFrame
    # Does NOT invalidate: Var reads (binding unchanged), column value reads
    elif w.type == WriteLocType.ATTR:
        if r.type == ReadLocType.ATTR:
            return _same_dataframe(w.qualifier, r.qualifier) and w.name == r.name
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
    output*(W) = ⋃ { output(w) | w ∈ W }

    Convert writes to the reads they produce.
    Used for write-write overlap in ForwardStale.
    """
    result: Set[ReadLoc] = set()
    for w in writes:
        result.update(w.output())
    return frozenset(result)


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
    ForwardStale write overlap: W'ᵢ ▷ output*(Wⱼ)

    Returns the WriteLocs from cell i that overlap with cell j's write effects.
    Non-empty means cell j should become stale (write-write overlap).
    Uses output() to convert j's writes to the reads they produce.

    Formal ref: FORMAL_DEVELOPMENT.md §3.3, CONFLICT_RELATION.md §Write-Write Conflict
    """
    return wlocs_conflict_rlocs(W_i, output_set(W_j))


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
    """Group Col/ColAdd/ColDel write locs by variable name → [column names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type in (WriteLocType.COL, WriteLocType.COL_ADD, WriteLocType.COL_DEL) and loc.qualifier:
            key = _display_qualifier(loc.qualifier)
            result.setdefault(key, []).append(loc.name)
    return {k: sorted(v) for k, v in result.items()}


def readlocset_to_attr_map(locs: ReadLocSet) -> Dict[str, List[str]]:
    """Group Attr read locs by variable name → [attribute names]."""
    result: Dict[str, List[str]] = {}
    for loc in locs:
        if loc.type == ReadLocType.ATTR and loc.qualifier:
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
    - Structural reads → Attr(d, a) with LocRef qualifier when stable_map available
    - File reads → File(p)

    Var(x) is always emitted alongside Col/Attr reads. This ensures
    variable rebinding is caught by the simple Var(x) ▷ Var(x) = true
    check, without needing a cross-domain bridge rule. Column independence
    is preserved because Col/Rows/Attr ▷ Var = false in the ▷ matrix.

    Args:
        tracking: TrackingData from cell execution
        namespace: Current kernel namespace (optional, for LocRef qualifiers)
        stable_map: StableIdMap instance (optional, for LocRef qualifiers)
    """
    locs: Set[ReadLoc] = set()

    for var, cols in (tracking.column_reads_before_writes or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        for col in cols:
            locs.add(ReadLoc.col(q, col))

    for var, attrs in (tracking.structural_reads or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        for attr in attrs:
            locs.add(ReadLoc.attr(q, attr))

    # Always emit Var(x) for every variable in reads_before_writes.
    # Var(x) captures the binding read; Col/Attr locs capture finer detail.
    # Rebinding is caught by Var(x) ▷ Var(x); column independence is
    # preserved because Col/Rows/Attr ▷ Var = false in the ▷ matrix.
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

    Note: TrackingData doesn't distinguish ColAdd/ColDel/Rows/Attr.
    All column writes are treated as Col (modification). The diff-based
    detect_write_locs() function produces the full typed WriteLocs.

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

    for var, cols in (tracking.column_writes or {}).items():
        q = get_qualifier(var, namespace, stable_map)
        for col in cols:
            locs.add(WriteLoc.col(q, col))

    for path in (tracking.file_writes or set()):
        locs.add(WriteLoc.file(path))

    return frozenset(locs)
