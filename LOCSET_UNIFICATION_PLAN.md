# Plan: Unify All Location Tracking with Typed Write Locations

## Current State

Phase 1 (remove provenance) is complete. What remains is the **dual-representation
problem**: the enforcer maintains parallel string-based and Loc-based views of R/W,
with ad-hoc column overlap logic bridging the gap. Additionally, four files
(`changes.py`, `access_events.py`, `conflict_resolver.py`, `conflict_rules.py`)
implement a parallel typed-change pipeline for backward mutation detection that
duplicates concepts already expressible through locations.

## Problem

Two parallel systems exist for the same purpose (conflict detection):

**System 1: String-based R/W + ad-hoc column logic**

- `NotebookState.reads/writes`: `Dict[str, Set[str]]` (variable names only)
- Column info threaded via `changed_vars`, `column_changed` parameters
- `_has_relevant_overlap_by_id()`: 50-line manual column-overlap check
- Used for: staleness propagation, forward contamination

**System 2: Typed Change/AccessEvent pipeline**

- `Change` hierarchy: `ValueChanged`, `ColumnAdded`, `ColumnModified`, `ColumnRemoved`,
  `RowsAdded`, `RowsRemoved`, `IndexChanged`, `DtypeChanged`
- `AccessEvent` hierarchy: `VariableRead`, `ColumnRead`, `StructuralRead`
- `ConflictResolver` + `CONFLICT_RULES` table: 20+ declarative rules
- Used for: backward mutation (NoWriteAfterRead), forward contamination typed-change path

## Goal

**One mechanism. One conflict check. One set of types.**

Replace both systems with:

- `ReadLoc` — what a cell reads (4 constructors)
- `WriteLoc` — what a cell writes and _how_ it changed (7 constructors)
- `▷ : WriteLoc × ReadLoc → bool` — the conflict relation (one 7×4 matrix)
- `output : WriteLoc → ReadLoc` — maps a write to the read it produces (for write-write overlap)

All four validity predicates, both staleness predicates, and backward mutation detection
use the same `▷` relation. The `ConflictResolver`, `Change`, `AccessEvent`, and
`conflict_rules.py` are deleted.

---

## Formal Specification

### Read Location Grammar

Read locations describe what a cell accessed during execution:

```
r ∈ ReadLoc ::= Var(x) | Col(d, c) | Attr(d, a) | File(p)
```

| Constructor  | Meaning                        | Example                  |
| ------------ | ------------------------------ | ------------------------ |
| `Var(x)`     | Whole variable `x`             | `df`, `config`, `model`  |
| `Col(d, c)`  | Column `c` of DataFrame `d`    | `df["price"]`            |
| `Attr(d, a)` | Attribute `a` of DataFrame `d` | `df.shape`, `df.columns` |
| `File(p)`    | File at path `p`               | `data.csv`               |

**R_i ⊆ ReadLoc** — the set of read locations for cell `i`.

### Write Location Grammar

Write locations describe what a cell changed and _how_:

```
w ∈ WriteLoc ::= Var(x) | Col(d, c) | ColAdd(d, c) | ColDel(d, c)
               | Rows(d) | Attr(d, a) | File(p)
```

| Constructor    | Meaning                          | Example                  |
| -------------- | -------------------------------- | ------------------------ |
| `Var(x)`       | Variable `x` completely replaced | `x = 42`                 |
| `Col(d, c)`    | Column `c` values modified       | `df["price"] = [1,2,3]`  |
| `ColAdd(d, c)` | New column `c` added             | `df["new"] = [4,5,6]`    |
| `ColDel(d, c)` | Column `c` removed               | `df.drop("old", axis=1)` |
| `Rows(d)`      | Rows added or removed            | `df.append(...)`         |
| `Attr(d, a)`   | Attribute `a` changed            | `df.reset_index()`       |
| `File(p)`      | File at path `p` written         | `df.to_csv("out.csv")`   |

**W_i ⊆ WriteLoc** — the set of write locations for cell `i`.

### The Conflict Relation: ▷

**Definition.** Write location `w` conflicts with read location `r`,
written `w ▷ r`, iff:

```
Var(x)           ▷  Var(y)            ≝  x = y
Var(x)           ▷  Col(d, c)        ≝  x = d
Var(x)           ▷  Attr(d, a)       ≝  x = d
Var(x)           ▷  File(p)          ≝  false

Col(d, c)        ▷  Var(x)           ≝  d = x
Col(d, c)        ▷  Col(d', c')      ≝  d = d' ∧ c = c'
Col(d, c)        ▷  Attr(d', a)      ≝  false                    ← modifying values doesn't change structure
Col(d, c)        ▷  File(p)          ≝  false

ColAdd(d, c)     ▷  Var(x)           ≝  d = x
ColAdd(d, c)     ▷  Col(d', c')      ≝  false                    ← adding column doesn't affect existing column reads
ColAdd(d, c)     ▷  Attr(d', a)      ≝  d = d' ∧ a ∈ COL_ATTRS  ← adding column changes column-structure attrs
ColAdd(d, c)     ▷  File(p)          ≝  false

ColDel(d, c)     ▷  Var(x)           ≝  d = x
ColDel(d, c)     ▷  Col(d', c')      ≝  d = d' ∧ c = c'         ← removing column invalidates reads of that column
ColDel(d, c)     ▷  Attr(d', a)      ≝  d = d' ∧ a ∈ COL_ATTRS  ← removing column changes column-structure attrs
ColDel(d, c)     ▷  File(p)          ≝  false

Rows(d)          ▷  Var(x)           ≝  d = x
Rows(d)          ▷  Col(d', c)       ≝  d = d'                   ← row change affects all column data
Rows(d)          ▷  Attr(d', a)      ≝  d = d' ∧ a ∈ ROW_ATTRS  ← row change affects row-structure attrs
Rows(d)          ▷  File(p)          ≝  false

Attr(d,a) ▷  Var(x)           ≝  d = x
Attr(d,a) ▷  Col(d', c)       ≝  false                    ← attr change doesn't affect column values
Attr(d,a) ▷  Attr(d', a')     ≝  d = d' ∧ a = a'
Attr(d,a) ▷  File(p)          ≝  false

File(p)          ▷  Var(x)           ≝  false
File(p)          ▷  Col(d, c)        ≝  false
File(p)          ▷  Attr(d, a)       ≝  false
File(p)          ▷  File(q)          ≝  p = q
```

where:

```
COL_ATTRS = { columns, keys, dtypes, axes, T, values, iter, describe, shape, size }
ROW_ATTRS = { index, shape, size, len, empty }
```

**Key design: the "how" is in the write, not in the read.** `Col(df, price)` (modify)
vs `ColAdd(df, price)` (add) have different conflict semantics. This is what eliminates
the need for the `ConflictResolver`'s typed-change rules.

### Attribute Conflicts Are Always Enforced

There is no OFF or WARN mode. `Attr` reads participate in `▷` unconditionally.
The old `StructuralTrackingMode` is removed.

### Conflict Set Operations

Extend `▷` pointwise to sets:

```
A ▷ B  ≝  { w ∈ A | ∃ r ∈ B . w ▷ r }
```

`A ▷ B` returns the write locs in `A` that conflict with some read loc in `B`.
Write `A ▷ B ≠ ∅` to mean "some write in A conflicts with some read in B."

### The Output Function

For ForwardStale's write-write overlap, we need to check whether two cells'
writes "touch the same location." Define `output : WriteLoc → ReadLoc` — the
read location that would observe this write's value:

```
output(Var(x))           = Var(x)
output(Col(d, c))        = Col(d, c)
output(ColAdd(d, c))     = Col(d, c)
output(ColDel(d, c))     = Col(d, c)
output(Rows(d))          = Var(d)
output(Attr(d,a)) = Attr(d, a)
output(File(p))          = File(p)
```

Extend to sets: `output*(W) = { output(w) | w ∈ W }`.

**Key property:** `ColAdd(df, price)` and `ColAdd(df, qty)` have different outputs
(`Col(df, price)` vs `Col(df, qty)`), so they don't overlap. Two independent column
additions do NOT cause false write-write conflicts.

### Revised Validity Predicates

```
NoReadAndWrite(R, W, i)    ≝  Wᵢ ▷ Rᵢ = ∅
WriteBeforeRead(R, W, i)   ≝  ∀ r ∈ Rᵢ . (r ∈ builtins) ∨ (r ∈ ambient) ∨ (∃ j < i . Wⱼ ▷ {r} ≠ ∅)
NoReadBeforeWrite(R, W, i) ≝  W_{i+1..n} ▷ Rᵢ = ∅
NoWriteAfterRead(R, W, i)  ≝  Wᵢ ▷ R_{1..i-1} = ∅
RecoverableMutation(W, i)  ≝  diff(preᵢ, Σ) ⊆ Wᵢ     (all mutations are in write set)
```

Note: `WriteBeforeRead` retains the "ambient" exclusion — variables that exist in the
namespace but weren't written by any cell are allowed (pragmatic, matches current behavior).

### Revised Staleness Predicates

```
ForwardStale(R, W, W', i, j) ≝  j > i ∧ (
    (Wᵢ ∪ W'ᵢ) ▷ Rⱼ ≠ ∅                             — write-read conflict
    ∨ (Wᵢ ∪ W'ᵢ) ▷ output*(Wⱼ) ≠ ∅                   — write-write overlap
)

BackwardStale(W, W', i, j) ≝  j < i ∧ j = LastWriter(W, i, w) for some w ∈ Wᵢ \ W'ᵢ
```

**Note on types:** The paper writes `(Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ)`, unioning reads and
writes freely because they share a single `Loc` type. With typed locations, `Rⱼ` is
`ReadLocSet` and `Wⱼ` is `WriteLocSet` — different types that cannot be unioned.
The split into two disjuncts resolves this: the first checks writes against reads
directly (`▷ : WriteLoc × ReadLoc`), and the second converts writes to their
equivalent reads via `output*` before checking. This is semantically equivalent to
the paper but type-safe.

### LastWriter

`LastWriter` operates at the variable level (per user decision):

```
var(w) = x           if w = Var(x)
       = d           if w = Col(d,c) | ColAdd(d,c) | ColDel(d,c) | Rows(d) | Attr(d,a)
       = p           if w = File(p)

LastWriter(W, i, w) = max { j < i | var(w) ∈ { var(w') | w' ∈ Wⱼ } }
```

### Diff → WriteLoc Mapping

The diff system produces typed changes. Each maps to `WriteLoc` as follows:

| Diff result                     | WriteLoc                        |
| ------------------------------- | ------------------------------- |
| Variable completely replaced    | `Var(x)`                        |
| DataFrame column values changed | `Col(d, c)`                     |
| DataFrame column added          | `ColAdd(d, c)`                  |
| DataFrame column removed        | `ColDel(d, c)`                  |
| DataFrame rows added/removed    | `Rows(d)`                       |
| DataFrame index changed         | `Attr(d, index)`                |
| DataFrame dtype changed         | `Col(d, c)` + `Attr(d, dtypes)` |
| File written                    | `File(p)`                       |

### Tracking → ReadLoc Mapping

The runtime tracking system records what a cell accessed:

| Tracking field                        | ReadLoc           |
| ------------------------------------- | ----------------- |
| `reads_before_writes: {"df"}`         | `Var(df)`         |
| `column_reads: {"df": {"price"}}`     | `Col(df, price)`  |
| `structural_reads: {"df": {"shape"}}` | `Attr(df, shape)` |
| `file_reads: {"data.csv"}`            | `File(data.csv)`  |

This is exactly what `tracking_to_read_locs()` already does (modulo the rename
from `Loc` to `ReadLoc`).

---

## Worked Example: Independent Column Additions

Cells B and C both add columns to `df`:

```python
# Cell B: df["price"] = [10, 20, 30]
# Cell C: df["qty"] = [1, 2, 3]
```

After running B then C:

```
R_B = { Var(df) }                    W_B = { ColAdd(df, price) }
R_C = { Var(df) }                    W_C = { ColAdd(df, qty) }
```

**NoWriteAfterRead for C:** `W_C ▷ R_{before_C}`.
`ColAdd(df, qty) ▷ Var(df)` = `d = x` = `df = df` = **true** → violation!

Wait — is this correct? C adds a column to df, and B read df. Should that be a
violation? Yes: C mutates the DataFrame that B already read. If we re-executed
top-to-bottom, B would see the pre-C DataFrame. This IS a real conflict.

But in practice, B didn't read the `qty` column (it didn't exist yet). The conflict
is at the variable level (`Var(df)`), which is conservative. The user must structure
their notebook so that df mutations flow top-to-bottom.

**ForwardStale after editing B:**
`(W_B_old ∪ W_B_new) ▷ R_C` — if B still writes `ColAdd(df, price)`, then
`ColAdd(df, price) ▷ Var(df) = true` → C is stale (correct: B changed df that C reads).

`(W_B_old ∪ W_B_new) ▷ output*(W_C)` — `ColAdd(df, price) ▷ output(ColAdd(df, qty))`
= `ColAdd(df, price) ▷ Col(df, qty)` = **false** (adding price doesn't affect qty reads).
No false write-write overlap. ✓

---

## Implementation

### Code: ReadLoc and WriteLoc

```python
# flowbook/kernel/models.py

class LocType(str, Enum):
    """Read location types."""
    VAR = "var"
    COLUMN = "column"
    ATTR = "attr"
    FILE = "file"


class WriteType(str, Enum):
    """Write location types — encode HOW a location changed."""
    VAR = "var"              # Variable completely replaced
    COL = "col"              # Column values modified
    COL_ADD = "col_add"      # New column added
    COL_DEL = "col_del"      # Column removed
    ROWS = "rows"            # Rows added or removed
    ATTR = "attr"    # Attribute changed
    FILE = "file"            # File written


@dataclass(frozen=True)
class ReadLoc:
    """A read location. Identifies what a cell accessed.

    Constructors:
        ReadLoc.var("x")                → Var(x)
        ReadLoc.col("df", "price")      → Col(df, price)
        ReadLoc.attr("df", "shape")     → Attr(df, shape)
        ReadLoc.file("data.csv")        → File(data.csv)
    """
    type: LocType
    name: str
    qualifier: Optional[str] = None

    @classmethod
    def var(cls, name: str) -> "ReadLoc":
        return cls(LocType.VAR, name)

    @classmethod
    def col(cls, var: str, column: str) -> "ReadLoc":
        return cls(LocType.COLUMN, column, qualifier=var)

    @classmethod
    def attr(cls, var: str, attribute: str) -> "ReadLoc":
        return cls(LocType.ATTR, attribute, qualifier=var)

    @classmethod
    def file(cls, path: str) -> "ReadLoc":
        return cls(LocType.FILE, path)

    def display_name(self) -> str:
        if self.type == LocType.VAR:
            return self.name
        elif self.type == LocType.COLUMN:
            return f"{self.qualifier}['{self.name}']"
        elif self.type == LocType.ATTR:
            return f"{self.qualifier}.{self.name}"
        elif self.type == LocType.FILE:
            return f"File({self.name})"
        return str(self)


@dataclass(frozen=True)
class WriteLoc:
    """A write location. Identifies what changed and how.

    Constructors:
        WriteLoc.var("x")                    → Var(x)
        WriteLoc.col("df", "price")          → Col(df, price) — values modified
        WriteLoc.col_add("df", "new")        → ColAdd(df, new)
        WriteLoc.col_del("df", "old")        → ColDel(df, old)
        WriteLoc.rows("df")                  → Rows(df)
        WriteLoc.attr("df", "index") → Attr(df, index)
        WriteLoc.file("out.csv")             → File(out.csv)
    """
    type: WriteType
    name: str
    qualifier: Optional[str] = None

    @classmethod
    def var(cls, name: str) -> "WriteLoc":
        return cls(WriteType.VAR, name)

    @classmethod
    def col(cls, var: str, column: str) -> "WriteLoc":
        return cls(WriteType.COL, column, qualifier=var)

    @classmethod
    def col_add(cls, var: str, column: str) -> "WriteLoc":
        return cls(WriteType.COL_ADD, column, qualifier=var)

    @classmethod
    def col_del(cls, var: str, column: str) -> "WriteLoc":
        return cls(WriteType.COL_DEL, column, qualifier=var)

    @classmethod
    def rows(cls, var: str) -> "WriteLoc":
        return cls(WriteType.ROWS, var)

    @classmethod
    def attr(cls, var: str, attribute: str) -> "WriteLoc":
        return cls(WriteType.ATTR, attribute, qualifier=var)

    @classmethod
    def file(cls, path: str) -> "WriteLoc":
        return cls(WriteType.FILE, path)

    def var_name(self) -> str:
        """Extract the top-level variable name."""
        if self.type in (WriteType.VAR, WriteType.ROWS, WriteType.FILE):
            return self.name
        return self.qualifier  # COL, COL_ADD, COL_DEL, ATTR

    def output(self) -> ReadLoc:
        """The ReadLoc that would observe this write's value."""
        if self.type == WriteType.VAR:
            return ReadLoc.var(self.name)
        elif self.type in (WriteType.COL, WriteType.COL_ADD, WriteType.COL_DEL):
            return ReadLoc.col(self.qualifier, self.name)
        elif self.type == WriteType.ROWS:
            return ReadLoc.var(self.name)
        elif self.type == WriteType.ATTR:
            return ReadLoc.attr(self.qualifier, self.name)
        elif self.type == WriteType.FILE:
            return ReadLoc.file(self.name)

    def display_name(self) -> str:
        if self.type == WriteType.VAR:
            return self.name
        elif self.type == WriteType.COL:
            return f"{self.qualifier}['{self.name}']"
        elif self.type == WriteType.COL_ADD:
            return f"{self.qualifier}['{self.name}'] (added)"
        elif self.type == WriteType.COL_DEL:
            return f"{self.qualifier}['{self.name}'] (removed)"
        elif self.type == WriteType.ROWS:
            return f"{self.name} (rows changed)"
        elif self.type == WriteType.ATTR:
            return f"{self.qualifier}.{self.name}"
        elif self.type == WriteType.FILE:
            return f"File({self.name})"
        return str(self)


ReadLocSet = FrozenSet[ReadLoc]
WriteLocSet = FrozenSet[WriteLoc]
```

### Code: The ▷ Conflict Function

```python
# Attribute sets
COL_ATTRS = frozenset({
    "columns", "keys", "dtypes", "axes", "T",
    "values", "iter", "describe", "shape", "size",
})
ROW_ATTRS = frozenset({
    "index", "shape", "size", "len", "empty",
})


def write_conflicts_read(w: WriteLoc, r: ReadLoc) -> bool:
    """w ▷ r — does writing w invalidate reading r?"""

    # --- Var(x) writes: invalidate any read involving x ---
    if w.type == WriteType.VAR:
        if r.type == LocType.VAR:
            return w.name == r.name
        elif r.type in (LocType.COLUMN, LocType.ATTR):
            return w.name == r.qualifier
        return False

    # --- Col(d, c) writes: invalidate same-column reads and whole-var reads ---
    elif w.type == WriteType.COL:
        if r.type == LocType.VAR:
            return w.qualifier == r.name
        elif r.type == LocType.COLUMN:
            return w.qualifier == r.qualifier and w.name == r.name
        return False  # Col write does NOT affect Attr reads (values ≠ structure)

    # --- ColAdd(d, c) writes: structural change, doesn't affect existing columns ---
    elif w.type == WriteType.COL_ADD:
        if r.type == LocType.VAR:
            return w.qualifier == r.name
        elif r.type == LocType.COLUMN:
            return False  # Adding new column doesn't affect existing column reads
        elif r.type == LocType.ATTR:
            return w.qualifier == r.qualifier and r.name in COL_ATTRS
        return False

    # --- ColDel(d, c) writes: invalidates that column + structural attrs ---
    elif w.type == WriteType.COL_DEL:
        if r.type == LocType.VAR:
            return w.qualifier == r.name
        elif r.type == LocType.COLUMN:
            return w.qualifier == r.qualifier and w.name == r.name
        elif r.type == LocType.ATTR:
            return w.qualifier == r.qualifier and r.name in COL_ATTRS
        return False

    # --- Rows(d) writes: affects all columns + row-structure attrs ---
    elif w.type == WriteType.ROWS:
        if r.type == LocType.VAR:
            return w.name == r.name
        elif r.type == LocType.COLUMN:
            return w.name == r.qualifier
        elif r.type == LocType.ATTR:
            return w.name == r.qualifier and r.name in ROW_ATTRS
        return False

    # --- Attr(d, a) writes: invalidates same attr + whole-var reads ---
    elif w.type == WriteType.ATTR:
        if r.type == LocType.VAR:
            return w.qualifier == r.name
        elif r.type == LocType.ATTR:
            return w.qualifier == r.qualifier and w.name == r.name
        return False  # Attr change doesn't affect column value reads

    # --- File(p) writes ---
    elif w.type == WriteType.FILE:
        if r.type == LocType.FILE:
            return w.name == r.name
        return False

    return False


def wlocs_conflict_rlocs(writes: WriteLocSet, reads: ReadLocSet) -> WriteLocSet:
    """W ▷ R — return write locs that conflict with some read."""
    return frozenset(
        w for w in writes
        if any(write_conflicts_read(w, r) for r in reads)
    )


def has_conflict(writes: WriteLocSet, reads: ReadLocSet) -> bool:
    """W ▷ R ≠ ∅ — quick boolean check."""
    return any(
        write_conflicts_read(w, r)
        for w in writes for r in reads
    )


def output_set(writes: WriteLocSet) -> ReadLocSet:
    """output*(W) — convert writes to the reads they produce."""
    return frozenset(w.output() for w in writes)
```

### Code: ForwardStale (simplified enforcer)

```python
def _compute_forward_staleness(
    self,
    W_i_old: WriteLocSet,
    W_i_new: WriteLocSet,
    just_executed: str,
    my_position: int,
) -> List[str]:
    """ForwardStale(R, W, W', i, j) for all j > i."""
    W_union = W_i_old | W_i_new
    newly_stale = []

    for later_cell_id in self._cell_order[my_position + 1:]:
        if not self._notebook_state.is_clean(later_cell_id):
            continue

        R_j = self._notebook_state.reads.get(later_cell_id, frozenset())
        W_j = self._notebook_state.writes.get(later_cell_id, frozenset())

        # Write-read conflict: (Wᵢ ∪ W'ᵢ) ▷ Rⱼ
        conflicting = wlocs_conflict_rlocs(W_union, R_j)

        # Write-write overlap: (Wᵢ ∪ W'ᵢ) ▷ output*(Wⱼ)
        if not conflicting:
            conflicting = wlocs_conflict_rlocs(W_union, output_set(W_j))

        if conflicting:
            for w in conflicting:
                self._notebook_state.add_reason(
                    later_cell_id,
                    Reason(ReasonType.FORWARD_STALE, loc=w.output(), cell_id=just_executed)
                )
            newly_stale.append(later_cell_id)

    return newly_stale
```

### Code: NoWriteAfterRead (replaces ConflictResolver)

```python
def _check_backward_mutation(
    self,
    cell_id: str,
    my_position: int,
    W_i: WriteLocSet,
) -> Optional[ReproducibilityError]:
    """NoWriteAfterRead(R, W, i) ≝ Wᵢ ▷ R_{1..i-1} = ∅"""
    for prior_cell_id in self._cell_order[:my_position]:
        if not self._notebook_state.is_clean(prior_cell_id):
            continue
        R_j = self._notebook_state.reads.get(prior_cell_id, frozenset())
        conflicting = wlocs_conflict_rlocs(W_i, R_j)
        if conflicting:
            locs = sorted(w.display_name() for w in conflicting)
            return ReproducibilityError(
                error_type=ErrorType.NO_WRITE_AFTER_READ,
                cell_id=cell_id,
                locations=locs,
                message=f"Writes {', '.join(locs)} already read by earlier cell",
                causer_cell=prior_cell_id,
            )
    return None
```

### Code: Diff → WriteLocSet

```python
# flowbook/kernel/change_detector.py (rewritten)

def detect_write_locs(diff: MemoryCheckpointDiffResult) -> WriteLocSet:
    """Convert a diff result to WriteLocSet.

    Replaces: detect_changes() → List[Change]
    """
    locs: Set[WriteLoc] = set()

    for var_name, diff_tree in diff.differences.items():
        if isinstance(diff_tree, ValueComparison):
            locs.add(WriteLoc.var(var_name))
        elif isinstance(diff_tree, CompoundDiff):
            if diff_tree.source_type == "dataframe":
                locs.update(_df_diff_to_write_locs(var_name, diff_tree.children))
            else:
                locs.add(WriteLoc.var(var_name))
        else:
            locs.add(WriteLoc.var(var_name))

    return frozenset(locs)


def _df_diff_to_write_locs(var: str, children: Dict) -> Set[WriteLoc]:
    """Convert DataFrame diff children to WriteLocs."""
    locs: Set[WriteLoc] = set()
    for key, child in children.items():
        if key == "_structural_rows":
            locs.add(WriteLoc.rows(var))
        elif key == "_structural_columns":
            for col in _parse_added_columns(child):
                locs.add(WriteLoc.col_add(var, col))
        elif key == "_structural_index":
            locs.add(WriteLoc.attr(var, "index"))
        else:
            col_name = _extract_column_name(key)
            if col_name:
                kind = _classify_column_change(child)
                if kind == "added":
                    locs.add(WriteLoc.col_add(var, col_name))
                elif kind == "removed":
                    locs.add(WriteLoc.col_del(var, col_name))
                else:  # modified
                    locs.add(WriteLoc.col(var, col_name))
    return locs
```

---

## Implementation Phases

### Phase 1: ✅ COMPLETE — Remove dead provenance code

### Phase 2: Define ReadLoc, WriteLoc, and ▷

**Goal**: Implement the type system and conflict relation.

**Files**:

1. **`flowbook/kernel/models.py`**:
   - Add `ReadLoc`, `WriteLoc`, `ReadLocSet`, `WriteLocSet` (code above)
   - Add `write_conflicts_read()`, `wlocs_conflict_rlocs()`, `has_conflict()`, `output_set()`
   - Add `COL_ATTRS`, `ROW_ATTRS` constants
   - Keep old `Loc` temporarily for backward compatibility during migration

2. **`flowbook/kernel/tests/test_loc_conflicts.py`** (NEW):
   - Exhaustive tests of all 28 cells in the ▷ matrix
   - Tests for `output()` function
   - Tests for set-level operations
   - Tests for `display_name()`

### Phase 3: Migrate change_detector.py to produce WriteLocSet

**Goal**: The diff→changes pipeline produces `WriteLocSet` instead of `List[Change]`.

**Files**:

1. **`flowbook/kernel/change_detector.py`**: Rewrite `detect_changes()` → `detect_write_locs()`
   (code above). Keep `detect_changes()` temporarily as a wrapper.

2. **`flowbook/kernel/tests/test_change_detector.py`**: Update for new return type.

### Phase 4: Migrate NotebookState to ReadLocSet/WriteLocSet

**Goal**: Core state uses typed location sets.

**Files**:

1. **`flowbook/kernel/notebook_state.py`**:

   ```python
   reads: Dict[str, ReadLocSet]    # was Set[str]
   writes: Dict[str, WriteLocSet]  # was Set[str]
   ```

   - `record_execution()`: store `ReadLocSet` from `tracking_to_read_locs()`,
     `WriteLocSet` from `detect_write_locs()` (or from tracking)
   - `handle_delete()`: use `has_conflict()` / `wlocs_conflict_rlocs()`
   - `last_writer_for()`: use `var_name()` extraction (variable-level, per user decision)
   - `snapshot_cell_state()` / `restore_cell_state()`: store `ReadLocSet`/`WriteLocSet`

2. **`flowbook/kernel/models.py`**:
   - `CellStateSnapshot.reads`: `Optional[ReadLocSet]`
   - `CellStateSnapshot.writes`: `Optional[WriteLocSet]`

3. **Tests**: Update all tests using `reads`/`writes` as `Set[str]` to `ReadLocSet`/`WriteLocSet`.

### Phase 5: Migrate enforcer to use ▷ exclusively

**Goal**: All predicate checks and staleness computations use `▷`.

**Files**:

1. **`flowbook/kernel/reproducibility_enforcer.py`**:
   - `_check_no_read_and_write()`: `has_conflict(W_i, R_i)`
   - `_check_backward_mutation_new()`: `wlocs_conflict_rlocs(W_i, R_before_i)` (replaces ConflictResolver)
   - `_check_forward_contamination()`: `has_conflict(W_later, R_i)`
   - `_compute_forward_staleness_syntactic()`: new signature, uses `wlocs_conflict_rlocs()`
   - `_compute_backward_staleness_syntactic()`: same
   - **Remove**: `_has_relevant_overlap_by_id()`, parallel `changed_vars`/`column_changed` threading,
     all `ConflictResolver` usage
   - **Remove**: `_conflict_resolver` instance variable

2. **Remove `StructuralTrackingMode`**: Delete the enum, the `%structural_tracking` magic,
   and all mode-dependent branching.

### Phase 6: Delete superseded files

**Goal**: Remove the entire typed-change pipeline.

**Files to DELETE**:

- `flowbook/kernel/conflict_resolver.py` — replaced by `▷`
- `flowbook/kernel/conflict_rules.py` — rules absorbed into `▷` matrix
- `flowbook/kernel/access_events.py` — replaced by `ReadLoc`
- `flowbook/kernel/changes.py` — replaced by `WriteLoc`

**Files to UPDATE**:

- `flowbook/kernel/change_detector.py` — no longer imports `Change` types
- Remove old `Loc`, `LocSet`, `LocType` from `models.py` (replaced by `ReadLoc`/`WriteLoc`)
- Remove `check_loc_conflicts()`, `_locs_conflict()`, `diff_to_write_locs()` from `models.py`

**Tests to DELETE/UPDATE**:

- `flowbook/kernel/tests/test_conflict_rules.py` — delete (rules are now in ▷)
- `flowbook/kernel/tests/test_converters.py` — delete or rewrite for WriteLocSet

### Phase 7: Update Reason.loc, frontend bridge, and spec

**Reason.loc**:

- `Reason.loc`: change from `Optional[str]` to `Optional[ReadLoc]`
- `Reason.to_dict()`: serialize via `loc.display_name()` for frontend

**Frontend output bridge** (helpers in `models.py`):

```python
def readlocset_to_var_names(locs: ReadLocSet) -> List[str]:
def readlocset_to_column_map(locs: ReadLocSet) -> Dict[str, List[str]]:
def readlocset_to_file_list(locs: ReadLocSet) -> List[str]:
def writelocset_to_var_names(locs: WriteLocSet) -> List[str]:
def writelocset_to_column_map(locs: WriteLocSet) -> Dict[str, List[str]]:
```

**FORMAL_DEVELOPMENT.md**:

- §1.2: R_i ⊆ ReadLoc, W_i ⊆ WriteLoc
- §3.2: Predicates use ▷
- §3.3: Staleness uses ▷ + output
- §8.1: Replace with full ReadLoc/WriteLoc grammar and ▷ matrix
- Remove §8.2, §8.4 (absorbed into core)
- Implementation Map: update all references

---

## What Gets Deleted

| File                                | Lines     | Replacement                            |
| ----------------------------------- | --------- | -------------------------------------- |
| `conflict_resolver.py`              | ~394      | `wlocs_conflict_rlocs()` (~5 lines)    |
| `conflict_rules.py`                 | ~415      | `write_conflicts_read()` (~60 lines)   |
| `access_events.py`                  | ~143      | `ReadLoc` (~30 lines)                  |
| `changes.py`                        | ~150      | `WriteLoc` (~60 lines)                 |
| `_has_relevant_overlap_by_id()`     | ~50       | `has_conflict()` (3 lines)             |
| `_locs_conflict()`                  | ~55       | absorbed into `write_conflicts_read()` |
| `check_loc_conflicts()`             | ~50       | `wlocs_conflict_rlocs()`               |
| `StructuralTrackingMode` + branches | ~100      | deleted (always enforce)               |
| **Total removed**                   | **~1350** | **~160 replacement**                   |

## Key Design Decisions

1. **Reads and writes are different types.** `ReadLoc` (4 constructors) vs `WriteLoc`
   (7 constructors). The asymmetry is inherent: reads are "what did I look at?", writes
   are "what did I change and how?"

2. **The "how" is in the write, not in the conflict function.** `Col(df, price)` vs
   `ColAdd(df, price)` have different conflict semantics because they're different
   `WriteLoc` values. The `▷` function is a simple matrix lookup.

3. **`output()` handles write-write overlap.** Converting a `WriteLoc` to the `ReadLoc`
   it produces avoids false positives (independent column adds don't conflict).

4. **Attribute conflicts are always enforced.** No mode parameter anywhere.

5. **`LastWriter` stays at variable level.** Per user decision: `var_name()` extracts the
   top-level variable name from any `WriteLoc`.

6. **`WriteBeforeRead` keeps the ambient exclusion.** Pragmatic: pre-existing namespace
   variables are not flagged.

7. **`TrackingData` is unchanged.** It remains the runtime _collection_ format. Conversion
   to `ReadLocSet`/`WriteLocSet` happens once at `record_execution()` time.

## Risk Assessment

- **Phase 2** (types + ▷): Low risk. New code, no existing behavior changed.
- **Phase 3** (change_detector): Low risk. Well-isolated module.
- **Phase 4** (NotebookState): Medium risk. Many call sites for reads/writes.
- **Phase 5** (enforcer): High risk. Core logic rewrite. Must have full test coverage first.
- **Phase 6** (delete): Low risk. Just removing files that are no longer imported.
- **Phase 7** (cleanup): Low risk. Serialization and spec only.
