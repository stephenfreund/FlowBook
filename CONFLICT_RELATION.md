# The Conflict Relation ▷

FlowBook's reproducibility system tracks what each notebook cell _reads_ and _writes_, then uses a single conflict relation — **▷** (`write_conflicts_read`) — to determine whether executing one cell invalidates another. This document describes the location types, how they are generated, and the full conflict matrix.

**Notation.** We write **w ▷ r** to mean "write `w` invalidates read `r`." The relation is _asymmetric by construction_: the left operand is always a `WriteLoc` and the right is always a `ReadLoc` — they are different types, so `r ▷ w` is not even well-formed. We use the directed triangle ▷ rather than the symmetric-looking ▷ previously used in source code comments to emphasize this: the write _acts on_ the read, not the other way around.

## Location Grammars

Reads and writes share a **symmetric 5-type grammar**. Both grammars use the same five constructors: `Var`, `Col`, `Cols`, `Rows`, and `File`. The distinction between reads and writes is semantic (what was accessed vs. what changed), not structural.

### ReadLoc

```
x ∈ VarName               -- variable name (e.g., "df", "config")
d ∈ Address               -- stable DataFrame address (see §"Stable Object Identity via StableIdMap")
c ∈ ColName               -- column name (e.g., "price", "qty")
p ∈ FilePath              -- file path (e.g., "data.csv")

ReadLoc ::= Var(x)        -- whole-variable read
          | Col(d, c)     -- column c of DataFrame d
          | Cols(d)       -- column structure of DataFrame d (names, dtypes, count)
          | Rows(d)       -- row structure of DataFrame d (index, length, count)
          | File(p)       -- file at path p
```

| Constructor | Fields                  | Semantics                                   |
| ----------- | ----------------------- | ------------------------------------------- |
| `Var(x)`    | name = x                | Cell read variable `x` as an opaque value   |
| `Col(d, c)` | qualifier = d, name = c | Cell read column `c` of DataFrame `d`       |
| `Cols(d)`   | qualifier = d, name = d | Cell read column structure of DataFrame `d` |
| `Rows(d)`   | qualifier = d, name = d | Cell read row structure of DataFrame `d`    |
| `File(p)`   | name = p                | Cell read file at path `p`                  |

**Granularity rule:** If a variable has column-level or structural-level read detail, it is represented _only_ by those finer-grained locs. `Var(x)` is emitted only for variables with no column/structural detail — plain scalars, lists, dicts, etc. This avoids double-counting: a cell that reads `df["price"]` produces `Col(df, price)`, not both `Var(df)` and `Col(df, price)`.

### WriteLoc

Using the same metavariables as ReadLoc:

```
WriteLoc ::= Var(x)        -- variable completely replaced
           | Col(d, c)     -- column written (may add, modify, or delete)
           | Cols(d)       -- column structure changed (dtypes, etc.)
           | Rows(d)       -- rows added or removed from DataFrame d
           | File(p)       -- file at path p written
```

| Constructor | Fields                  | Semantics                                                        |
| ----------- | ----------------------- | ---------------------------------------------------------------- |
| `Var(x)`    | name = x                | Variable `x` was reassigned or is a non-DataFrame mutation       |
| `Col(d, c)` | qualifier = d, name = c | Column `c` of DataFrame `d` was written (add, modify, or delete) |
| `Cols(d)`   | qualifier = d, name = d | Column structure of DataFrame `d` changed (dtypes, etc.)         |
| `Rows(d)`   | qualifier = d, name = d | Rows were added to or removed from DataFrame `d`                 |
| `File(p)`   | name = p                | File at path `p` was written                                     |

## When Each Location Is Generated

### Read Locations

Read locations are recorded by runtime instrumentation during cell execution. FlowBook wraps DataFrame/Series access to observe what each cell touches.

| ReadLoc     | Generated when                                                                  | Examples                                                 |
| ----------- | ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `Var(x)`    | Variable `x` is read and has no column/structural detail (scalars, lists, etc.) | `y = x + 1`, `print(config)`, `len(my_list)`             |
| `Col(d, c)` | Column `c` of DataFrame `d` is accessed for computation                         | `df['price'].sum()`, `df.price.mean()`, `df.loc[:, 'x']` |
| `Cols(d)`   | A column-structure attribute is explicitly accessed                             | `df.columns`, `df.dtypes`, `df.keys()`, `for col in df:` |
| `Rows(d)`   | A row-structure attribute is explicitly accessed                                | `df.index`, `len(df)`, `df.empty`                        |
| `File(p)`   | A file at path `p` is read before being written in the same cell                | `pd.read_csv('data.csv')`, `open('config.json').read()`  |

**Cross-cutting attributes.** Some attributes reveal both column and row structure — `df.shape`, `df.values`, `df.T`, `df.axes`, `df.size`. For these, the tracking layer emits _both_ `Cols(d)` and `Rows(d)`. This ensures the read is sensitive to structural changes on either dimension.

**Structure-revealing vs. structure-using.** `Cols(d)` and `Rows(d)` are only recorded for _explicit_ access to structural attributes like `df.columns` or `df.index`. Internal access by structure-_using_ methods (`repr()`, `__getitem__`, `mean()`) does **not** produce structural reads, even though those methods internally touch structural attributes. This prevents over-staleness: calling `df['price'].mean()` should not make the cell sensitive to column-set changes.

**Attribute classification.** The mapping from individual attribute names (e.g., `"columns"`, `"index"`, `"shape"`) to `Cols(d)` / `Rows(d)` / both happens in the tracking layer (`column_tracking.py`), not in the conflict relation. The ▷ relation only sees `Cols` and `Rows` — it has no knowledge of individual attribute names.

### Write Locations

Write locations come from two sources: (1) diffing memory checkpoints taken before and after cell execution, and (2) runtime monkey patches that record structural mutations as they happen.

The `change_detector` module parses the structured diff tree into typed `Change` objects, which are then converted to `WriteLoc` values. Structural mutations (row changes, index changes, dtype changes, column deletions) are recorded at operation time by monkey patches in `column_tracking.py` and flow through `TrackingData` into `tracking_to_writelocset()`.

| WriteLoc    | Detected when                                                   | Examples                                                      |
| ----------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| `Var(x)`    | Variable `x` was reassigned, or a non-DataFrame object mutated  | `x = 10`, `config['key'] = val`, `df = pd.DataFrame(...)`     |
| `Col(d, c)` | Column `c` was added, modified, or deleted                      | `df['price'] *= 1.1`, `df['new'] = vals`, `del df['old']`     |
| `Cols(d)`   | Column structure changed: dtype change recorded by monkey patch | `df['x'] = df['x'].astype(float)`, `df.rename(columns=...)`   |
| `Rows(d)`   | Row mutations or index changes recorded by monkey patch         | `df.loc[len(df)] = row`, `pd.concat(...)`, `df.reset_index()` |
| `File(p)`   | File at path `p` was written during execution                   | `df.to_csv('out.csv')`, `open('result.json', 'w').write(...)` |

`DtypeChanged(d, c)` produces _two_ write locs: `Col(d, c)` (the column's data is now a different type) and `Cols(d)` (the dtype metadata changed).

**Structural mutation tracking.** Row mutations, index changes, dtype changes, and column deletions are recorded at operation time by monkey patches (e.g., on `__delitem__`, `drop`, `reset_index`, `.loc` row assignment) in `column_tracking.py`. These events are stored in `TrackingData` and converted to `WriteLoc` values by `tracking_to_writelocset()`. This ensures structural writes are correctly detected even on re-execution, where checkpoint diffs would be idempotent.

## The ▷ Conflict Relation

The function `write_conflicts_read(w, r)` answers: **does writing `w` invalidate reading `r`?**

This is a 5 x 5 matrix — 5 write types against 5 read types — and it is the _only_ conflict check in the entire system. All staleness predicates, backward conflict detection, and forward contamination checks are defined in terms of ▷.

### Read-Write Conflict Matrix

> **`True`** means the write invalidates the read (the cell that did the read is now stale).

| Write `w` ↓ \ Read `r` → | **Var(x')** | **Col(d', c')**       | **Cols(d')** | **Rows(d')** | **File(p')** |
| ------------------------ | ----------- | --------------------- | ------------ | ------------ | ------------ |
| **Var(x)**               | `x = x'`    | —                     | —            | —            | —            |
| **Col(d, c)**            | —           | `d ≡ d'` AND `c = c'` | `d ≡ d'`     | —            | —            |
| **Cols(d)**              | —           | `d ≡ d'`              | `d ≡ d'`     | —            | —            |
| **Rows(d)**              | —           | `d ≡ d'`              | —            | `d ≡ d'`     | —            |
| **File(p)**              | —           | —                     | —            | —            | `p = p'`     |

(**—** = never conflicts)

**Comparison operators:**

- **`d ≡ d'`** (identity): Compares `LocRef.loc_id` values — same DataFrame object via `StableIdMap`
- **`=`** (equality): String comparison for names, columns, and paths

### Key Observations

- **`Var(x)` only conflicts with `Var(x)` reads.** Rebinding detection for column/structural readers works because `Var(x)` is always present in read sets alongside `Col`/`Cols`/`Rows` reads (see `tracking_to_readlocset`). When `df = new_value`, the read set `{Var("df"), Col(df, "price"), ...}` ensures `Var("df") ▷ Var("df") = true` catches the rebinding. No cross-domain bridge rule is needed.
- **`Col(d, c)` is conservative.** Writing a column invalidates reads of that _exact_ column, plus _all_ column-structure reads (`Cols`). This covers add, modify, and delete scenarios uniformly. _Column independence_ is preserved: cell A reads `df["qty"]`, cell B writes `df["price"]` → no conflict.
- **`Cols(d)` invalidates all columns.** A column-structure write (e.g., dtype change) invalidates every `Col(d, c)` read on that DataFrame, as well as other `Cols(d)` reads. This is conservative but correct: if the column structure changed, any column-level computation might be affected.
- **`Rows(d)` is column-wide but dimension-independent.** Row changes invalidate all column reads (more or fewer values per column) and row-structure reads. But `Rows(d)` does _not_ conflict with `Cols(d)` reads — adding a row does not change `df.columns` or `df.dtypes`.
- **Cross-domain independence.** `Rows ▷ Cols = false` and `Cols ▷ Rows = false`. Row-structural changes do not invalidate column-structure readers, and vice versa. This is a key design property: `df.reset_index(inplace=True)` (which produces `Rows(d)`) does not stale a cell that only read `df.columns`.
- **No lookup tables.** The ▷ matrix has no attribute-name lookup tables (`COL_ATTRS`, `ROW_ATTRS`). The classification of individual attributes into the `Cols` or `Rows` domain happens in the tracking layer. The conflict relation operates purely on the five location types.

## Write-Write Conflict (Forward Staleness)

The system uses a direct write-write conflict relation — **▷▷** (`write_conflicts_write`) — to determine whether two writes overlap. This is a self-contained 5x5 function, analogous to the 5x5 ▷ relation but operating on two `WriteLoc` operands.

### The ▷▷ Relation

`write_conflicts_write(w1, w2)` answers: **do writes `w1` and `w2` overlap?** An entry shows the condition under which `w₁ ▷▷ w₂` holds — i.e., executing cell `i` (row) makes cell `j`'s write (column) stale. Comparison operators are the same as in the read-write matrix above: `≡` for DataFrame identity, `=` for string equality.

| w₁ ↓ \ w₂ →   | **Var(x')** | **Col(d', c')**       | **Cols(d')** | **Rows(d')** | **File(p')** |
| ------------- | ----------- | --------------------- | ------------ | ------------ | ------------ |
| **Var(x)**    | `x = x'`    | —                     | —            | —            | —            |
| **Col(d, c)** | —           | `d ≡ d'` AND `c = c'` | `d ≡ d'`     | `d ≡ d'`     | —            |
| **Cols(d)**   | —           | `d ≡ d'`              | `d ≡ d'`     | —            | —            |
| **Rows(d)**   | —           | `d ≡ d'`              | —            | `d ≡ d'`     | —            |
| **File(p)**   | —           | —                     | —            | —            | `p = p'`     |

(**—** = no write-write conflict)

**Code:** `write_conflicts_write()` in `kernel/locations.py`

### Forward Staleness Check

When cell `i` executes, for each later cell `j`:

1. **Read-based staleness:** `W'ᵢ ▷ Rⱼ ≠ ∅` — did `i`'s writes invalidate what `j` previously read?
2. **Write-based staleness:** `W'ᵢ ▷▷ Wⱼ ≠ ∅` — do `i`'s writes overlap with what `j` writes?

The first check uses ▷ (write-read); the second uses ▷▷ (write-write). The second catches cases like: cell A and cell B both write `df["price"]`. If cell A re-executes with a new value, cell B is stale — its write was computed from outdated inputs.

### Key Observations

**Column independence is preserved.** Two writes to distinct columns of the same DataFrame do NOT conflict: `Col(d, "price") ▷▷ Col(d', "qty")` requires `c = c'`, which fails.

- **The ▷▷ relation is symmetric.** If `w₁ ▷▷ w₂` then `w₂ ▷▷ w₁`. This follows from the semantics: if two writes touch overlapping state, both cells' results depend on execution order, so staleness propagates in both directions.
- **`Var(x)` only overlaps with `Var(x')`** when `x = x'`. Write-write overlap between `Var("df")` and `Col(df, c)` is not detected by the write-write path; instead, it is caught by the read overlap path because the read set always contains `Var("df")` alongside `Col` reads.
- **`Col` vs `Col`** requires exact column match (`c = c'`) — column independence at the write-write level.
- **`Col` vs `Rows` overlap** is detected bidirectionally: `Col(d, c) ▷▷ Rows(d')` and `Rows(d) ▷▷ Col(d', c')` both hold when `d ≡ d'`, because row changes affect all column data and vice versa.
- **`Col` vs `Cols` overlap** is symmetric: `Col(d, c) ▷▷ Cols(d')` and `Cols(d) ▷▷ Col(d', c')` both hold when `d ≡ d'`, because a column-structure write and a column-value write touch overlapping state.
- **Cross-domain independence in ▷▷.** `Cols(d) ▷▷ Rows(d') = false` and `Rows(d) ▷▷ Cols(d') = false`. Column-structure writes and row-structure writes are independent, matching the ▷ relation.

## False Positives

The `Cols` and `Rows` types are domain-level abstractions that deliberately sacrifice some precision for simplicity. This section documents the known false positives.

### Col(d, c) ▷ Cols(d) — already conservative

Writing a single column invalidates _all_ `Cols(d)` readers, even those that only checked `df.dtypes` while the written column's dtype did not change. This is the same conservatism that existed in the previous `Attr`-based system. It is acceptable because column-structure reads are uncommon and the false positive rate is low in practice.

### Intra-domain conflation within Rows

`Rows(d)` conflates all row-structural attributes into a single domain: index label changes, row count changes, and row-order changes all produce the same `Rows(d)` write. This means an index label change (e.g., `df.index = new_labels`) will stale a cell that only read `len(df)`, even though the row count did not change.

### Intra-domain conflation within Cols

Similarly, `Cols(d)` conflates dtype changes and column-set changes. A dtype change (e.g., `df['x'] = df['x'].astype(float)`) produces `Cols(d)`, which stales a cell that read `df.columns` even though the column set did not change.

### Cross-domain independence is preserved

The important precision guarantee: `Rows ▷ Cols = false` and `Cols ▷ Rows = false`. Row-structural changes do not stale column-structure readers, and column-structural changes do not stale row-structure readers. This is stronger than the previous `Attr`-based system, where the `COL_ATTRS`/`ROW_ATTRS` overlap on shared attributes (`shape`, `axes`, `values`, `T`) created cross-domain coupling.

## Stable Object Identity via StableIdMap

In the grammars above, the qualifier `d` in `Col(d, c)`, `Cols(d)`, `Rows(d)` is a
**`LocRef(loc_id, var_name)`** — a dual-purpose identifier combining stable object
identity with the variable name used to access it. This matches the formal model
where `d ∈ Address` is a stable DataFrame address.

### The `LocRef` qualifier

```python
@dataclass(frozen=True)
class LocRef:
    loc_id: int    # Stable object identity (from StableIdMap)
    var_name: str  # Variable name at access time (for Var ▷ Col bridging)
```

**Two LocRefs with the same `loc_id` refer to the same DataFrame object**, even if
accessed through different variable names (aliases). The `var_name` records which
name was used to access the object, enabling `Var(x) ▷ Col(d, c)` bridging.

### StableIdMap: Surviving deep copy

The challenge: Python's `id()` breaks on checkpoint deep copy — every `deepcopy()`
creates new objects with new ids. The `StableIdMap` solves this with a weakref-based
side-table:

1. **Assignment**: Maps `id(obj)` → `(stable_id, weakref(obj))`
2. **Lookup**: Checks `ref() is obj` to verify same object (detects id reuse after GC)
3. **Checkpoint transfer**: `apply_memo(memo)` copies stable_ids from originals
   to their deep-copy targets using the checkpoint's memo dict

| Scenario                  | Action                                     | Result |
| ------------------------- | ------------------------------------------ | ------ |
| Same object               | `ref() is obj` → return existing stable_id | ✓      |
| Alias (`df2 = df`)        | Same object → same stable_id               | ✓      |
| User copy (`df.copy()`)   | Different object → new stable_id           | ✓      |
| id reuse after GC         | `ref()` dead → new stable_id               | ✓      |
| Our deepcopy (checkpoint) | `apply_memo()` transfers stable_id         | ✓      |

### How ▷ uses qualifiers

The ▷ relation uses one comparison mode for DataFrame-level checks:

- **`_same_dataframe(w.qualifier, r.qualifier)`** — for DataFrame-to-DataFrame checks
  (Col, Cols, Rows). Compares `loc_id`s when both are LocRef.
  Aliased DataFrames match automatically.

`Var(x)` writes only conflict with `Var(x)` reads (simple name equality). Rebinding
detection for column/structural readers works because `Var(x)` is always present in
read sets alongside `Col`/`Cols`/`Rows` reads — see `tracking_to_readlocset()`.

### Backward compatibility

When `StableIdMap` is not available (e.g., in tests), qualifiers fall back to plain
strings. The `_same_dataframe()` helper handles mixed comparisons:
`_same_dataframe(LocRef(42, "df"), "df")` returns `True` via var_name matching.

See `FORMAL_DEVELOPMENT.md` §9.1 for the full design analysis.

**Code:**

- StableIdMap + LocRef: `kernel/loc_ids.py`
- Qualifier comparison: `_same_dataframe()` in `kernel/locations.py`
- Memo transfer: `_apply_restore_memo()` in `kernel/flowbook_kernel.py`
- Alias detection (Phase 1 safety net): `_expand_with_deep_aliases()` in `kernel/reproducibility_enforcer.py`

## Structural Mutation Tracking

Checkpoint diffs are idempotent: re-executing a structural operation (e.g., deleting an already-deleted column) produces no diff. To ensure structural writes are always correctly detected, FlowBook records structural mutations at operation time via monkey patches in `column_tracking.py`.

### How structural writes flow

1. **At operation time:** Monkey patches on DataFrame methods (e.g., `__delitem__`, `drop`, `reset_index`, `.loc` row assignment) record structural events into `TrackingData` during cell execution.
2. **After execution:** `tracking_to_writelocset()` converts `TrackingData` events into typed `WriteLoc` values (`Col`, `Cols`, `Rows`), which are merged with diff-derived writes.
3. **Result:** The enforcer always sees the correct write set, even on re-execution.

This replaces the previous `_inject_structural_writes()` approach, which used `DataFrameProvenance` stored in `df.attrs` to retroactively augment write sets. That method was removed because operation-time tracking is simpler and more reliable.

### DataFrameProvenance (reporting only)

`DataFrameProvenance` still exists in `df.attrs['_flowbook_provenance']` for reporting purposes (e.g., showing which cell first created a column). However, it is **no longer used by the enforcer for conflict detection** — all conflict-relevant structural writes come from `TrackingData`.

**Code:**

- Structural mutation recording: monkey patches in `kernel_support/column_tracking.py`
- TrackingData → WriteLoc conversion: `tracking_to_writelocset()` in `kernel/change_detector.py`
- Provenance class (reporting only): `DataFrameProvenance` in `kernel_support/column_provenance.py`

## Design Summary

The entire conflict system rests on three primitives:

1. **`ReadLoc` (5 types) / `WriteLoc` (5 types)** — symmetric typed locations that encode _what_ was accessed and _how_ it changed. Both grammars share the same five constructors: `Var`, `Col`, `Cols`, `Rows`, `File`.
2. **`▷` (`write_conflicts_read`)** — a 5x5 function: "does this write invalidate this read?"
3. **`▷▷` (`write_conflicts_write`)** — a 5x5 function: "do these two writes overlap?"

All higher-level predicates — `BackConflict`, `FwdContaminated`, `StaleFwd` — are defined compositionally from these primitives. There is no separate conflict rules table, no ad-hoc special cases, no attribute-name lookup tables. The classification of individual DataFrame attributes into the `Cols` or `Rows` domain happens in the tracking layer, not in the conflict relation. Adding a new location type requires only:

- A new arm in `write_conflicts_read()` (which reads it invalidates)
- A new arm in `write_conflicts_write()` (which writes it overlaps with)
- A mapping in `changes_to_write_locs()` or `tracking_to_writelocset()` (how to detect it)
