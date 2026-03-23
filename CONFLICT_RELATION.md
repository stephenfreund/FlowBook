# The Conflict Relation ▷

FlowBook's reproducibility system tracks what each notebook cell *reads* and *writes*, then uses a single conflict relation — **▷** (`write_conflicts_read`) — to determine whether executing one cell invalidates another. This document describes the location types, how they are generated, and the full conflict matrix.

**Notation.** We write **w ▷ r** to mean "write `w` invalidates read `r`." The relation is *asymmetric by construction*: the left operand is always a `WriteLoc` and the right is always a `ReadLoc` — they are different types, so `r ▷ w` is not even well-formed. We use the directed triangle ▷ rather than the symmetric-looking ▷ previously used in source code comments to emphasize this: the write *acts on* the read, not the other way around.

## Location Grammars

Reads and writes are different types. Reads describe *what a cell looked at*; writes describe *what a cell changed and how*. The "how" is what makes column-granular conflict resolution possible: modifying a column's values is a different kind of write than adding a new column, and each invalidates a different set of reads.

### ReadLoc

```
x ∈ VarName               -- variable name (e.g., "df", "config")
d ∈ Address               -- stable DataFrame address (see §"Stable Object Identity via StableIdMap")
c ∈ ColName               -- column name (e.g., "price", "qty")
a ∈ AttrName              -- structural attribute name (e.g., "shape", "columns")
p ∈ FilePath              -- file path (e.g., "data.csv")

ReadLoc ::= Var(x)        -- whole-variable read
          | Col(d, c)     -- column c of DataFrame d
          | Attr(d, a)    -- structural attribute a of DataFrame d
          | File(p)       -- file at path p
```

| Constructor  | Fields               | Semantics                                        |
|-------------|----------------------|--------------------------------------------------|
| `Var(x)`    | name = x             | Cell read variable `x` as an opaque value        |
| `Col(d, c)` | qualifier = d, name = c | Cell read column `c` of DataFrame `d`           |
| `Attr(d, a)` | qualifier = d, name = a | Cell read structural attribute `a` of DataFrame `d` |
| `File(p)`   | name = p             | Cell read file at path `p`                       |

**Granularity rule:** If a variable has column-level or structural-level read detail, it is represented *only* by those finer-grained locs. `Var(x)` is emitted only for variables with no column/structural detail — plain scalars, lists, dicts, etc. This avoids double-counting: a cell that reads `df["price"]` produces `Col(df, price)`, not both `Var(df)` and `Col(df, price)`.

### WriteLoc

Using the same metavariables as ReadLoc:

```
WriteLoc ::= Var(x)            -- variable completely replaced
           | Col(d, c)         -- column c values modified in place
           | ColAdd(d, c)      -- new column c added to DataFrame d
           | ColDel(d, c)      -- column c removed from DataFrame d
           | Rows(d)           -- rows added or removed from DataFrame d
           | Attr(d, a)        -- structural attribute a changed
           | File(p)           -- file at path p written
```

| Constructor         | Fields               | Semantics                                              |
|--------------------|----------------------|--------------------------------------------------------|
| `Var(x)`           | name = x             | Variable `x` was reassigned or is a non-DataFrame mutation |
| `Col(d, c)`        | qualifier = d, name = c | Column `c` of DataFrame `d` had its values modified    |
| `ColAdd(d, c)`     | qualifier = d, name = c | Column `c` was added to DataFrame `d`                  |
| `ColDel(d, c)`     | qualifier = d, name = c | Column `c` was removed from DataFrame `d`              |
| `Rows(d)`          | name = d             | Rows were added to or removed from DataFrame `d`       |
| `Attr(d, a)` | qualifier = d, name = a | Attribute `a` of DataFrame `d` changed (e.g., index)  |
| `File(p)`          | name = p             | File at path `p` was written                           |


## When Each Location Is Generated

### Read Locations

Read locations are recorded by runtime instrumentation during cell execution. FlowBook wraps DataFrame/Series access to observe what each cell touches.

| ReadLoc        | Generated when                                                                  | Examples                                                    |
|---------------|---------------------------------------------------------------------------------|-------------------------------------------------------------|
| `Var(x)`      | Variable `x` is read and has no column/structural detail (scalars, lists, etc.) | `y = x + 1`, `print(config)`, `len(my_list)`               |
| `Col(d, c)`   | Column `c` of DataFrame `d` is accessed for computation                         | `df['price'].sum()`, `df.price.mean()`, `df.loc[:, 'x']`   |
| `Attr(d, a)`  | A structure-*revealing* attribute is explicitly accessed                         | `df.columns`, `df.shape`, `len(df)`, `for col in df:`       |
| `File(p)`     | A file at path `p` is read before being written in the same cell                | `pd.read_csv('data.csv')`, `open('config.json').read()`     |

**Structure-revealing vs. structure-using:** `Attr(d, a)` is only recorded for *explicit* access to structural attributes like `df.columns` or `df.shape`. Internal access by structure-*using* methods (`repr()`, `__getitem__`, `mean()`) does **not** produce `Attr` reads, even though those methods internally touch structural attributes. This prevents over-staleness: calling `df['price'].mean()` should not make the cell sensitive to column-set changes.

### Write Locations

Write locations are determined by diffing memory checkpoints taken before and after cell execution. The `change_detector` module parses the structured diff tree into typed `Change` objects, which are then converted to `WriteLoc` values.

| WriteLoc            | Detected when (checkpoint diff)                                | Examples                                                          |
|--------------------|----------------------------------------------------------------|-------------------------------------------------------------------|
| `Var(x)`           | Variable `x` was reassigned, or a non-DataFrame object mutated | `x = 10`, `config['key'] = val`, `df = pd.DataFrame(...)`        |
| `Col(d, c)`        | Column `c` exists in both pre- and post-checkpoint but values differ | `df['price'] *= 1.1`, `df.loc[:, 'x'] = 0`                     |
| `ColAdd(d, c)`     | Column `c` exists in post-checkpoint but not in pre-checkpoint | `df['new'] = vals`, `df.insert(0, 'col', v)`, `df.assign(...)` |
| `ColDel(d, c)`     | Column `c` exists in pre-checkpoint but not in post-checkpoint | `del df['old']`, `df.drop(columns=['x'], inplace=True)`          |
| `Rows(d)`          | Row count of DataFrame `d` changed between checkpoints         | `df.loc[len(df)] = row`, `pd.concat(...)`, `df.dropna(...)`     |
| `Attr(d, a)` | Attribute value differs (e.g., index labels changed, same length) | `df.reset_index(inplace=True)`, `df.index = new_labels`        |
| `File(p)`          | File at path `p` was written during execution                   | `df.to_csv('out.csv')`, `open('result.json', 'w').write(...)`   |

`DtypeChanged(d, c)` produces *two* write locs: `Col(d, c)` (the column's data is now a different type) and `Attr(d, "dtypes")` (the dtype metadata changed).


## The ▷ Conflict Relation

The function `write_conflicts_read(w, r)` answers: **does writing `w` invalidate reading `r`?**

This is a 7 × 4 matrix — 7 write types against 4 read types — and it is the *only* conflict check in the entire system. All staleness predicates, backward conflict detection, and forward contamination checks are defined in terms of ▷.

### Attribute Groups

Two sets define which DataFrame attributes are sensitive to which kind of structural change:

| Group        | Members                                                                    | Meaning                       |
|-------------|----------------------------------------------------------------------------|-------------------------------|
| `COL_ATTRS` | `columns`, `keys`, `dtypes`, `axes`, `T`, `values`, `iter`, `describe`, `shape`, `size` | Attributes that reveal column structure |
| `COL_VALUE_ATTRS` | `values`, `T`, `describe`                                           | Attributes that depend on column data values |
| `ROW_ATTRS` | `index`, `axes`, `values`, `T`, `shape`, `size`, `len`, `empty`            | Attributes that reveal row structure    |

`shape`, `size`, `axes`, `values`, and `T` appear in both — they expose both dimensions. For example, `axes = [index, columns]` is affected by both row and column structural changes.

### Read-Write Conflict Matrix

> **`True`** means the write invalidates the read (the cell that did the read is now stale).

| Write `w` ↓  \  Read `r` → | **Var(x')**         | **Col(d', c')**                  | **Attr(d', a')**                       | **File(p')**         |
|-----------------------------|---------------------|----------------------------------|-----------------------------------------|----------------------|
| **Var(x)**                  | `x = x'`            | `x = name(d')`                   | `x = name(d')`                          | —                    |
| **Col(d, c)**               | —                   | `d ≡ d'` AND `c = c'`            | `d ≡ d'` AND `a' ∈ COL_VALUE_ATTRS`    | —                    |
| **ColAdd(d, c)**            | —                   | —                                | `d ≡ d'` AND `a' ∈ COL_ATTRS`          | —                    |
| **ColDel(d, c)**            | —                   | `d ≡ d'` AND `c = c'`            | `d ≡ d'` AND `a' ∈ COL_ATTRS`          | —                    |
| **Rows(d)**                 | —                   | `d ≡ d'` (all columns)           | `d ≡ d'` AND `a' ∈ ROW_ATTRS`          | —                    |
| **Attr(d, a)**              | —                   | —                                | `d ≡ d'` AND `a = a'`                  | —                    |
| **File(p)**                 | —                   | —                                | —                                       | `p = p'`             |

(**—** = never conflicts)

**Comparison operators:**
- **`d ≡ d'`** (identity): Compares `LocRef.loc_id` values — same DataFrame object via `StableIdMap`
- **`name(d')`**: Extracts the variable name from an address (`LocRef.var_name`) — used for the `Var(x) ▷ Col/Attr` bridge, where variable rebinding invalidates reads through that name
- **`=`** (equality): String comparison for names, columns, attributes, and paths

Key observations:

- **`Var(x)` is the nuclear option.** Replacing a variable conflicts with *every* read type on that variable — column reads, attribute reads, everything. The entire binding changed.
- **`Col(d, c)` is maximally precise.** Modifying column values invalidates reads of that *exact* column, plus value-dependent attributes (`values`, `T`, `describe`) that expose the raw data. It does not touch structural attributes or other columns. This is what enables *column independence*: cell A reads `df["qty"]`, cell B writes `df["price"]` → no conflict (unless A also reads `df.values`).
- **`ColAdd(d, c)` does not invalidate existing column reads.** The old columns' data is untouched. It only invalidates structural attributes like `columns` and `shape` that would now reflect the extra column.
- **`ColDel(d, c)` is stricter than `ColAdd`.** It invalidates reads of the deleted column (it no longer exists) *plus* the same structural attributes.
- **`Rows(d)` is column-wide.** Every column's data changed (more or fewer values), so all column reads conflict. Row-structural attributes (`index`, `shape`, `len`, `empty`) and shared attributes (`axes`, `values`, `T`) are also affected — but `df.columns` and `df.dtypes` are unchanged by adding a row.
- **`Attr(d, a)` is point-to-point in ▷.** Only the exact same attribute conflicts. Changing the index does not *directly* invalidate reading `dtypes`. However, some attribute changes have *derived effects* — for example, changing the index also changes `axes` (since `axes = [index, columns]`). The change detector handles this by emitting `Attr` writes for all affected derived attributes, not just the root cause. This keeps ▷ simple (point-to-point) while ensuring derived attributes are correctly invalidated.


## Write-Write Conflict (Forward Staleness)

There is no separate write-write conflict function. Instead, the system converts one side's writes into reads via the **output function**, then reuses ▷.

### The Output Function

`output(w)` maps a `WriteLoc` to the set of `ReadLoc`s that would *observe* the effect that `w` produced. Each write type returns exactly the reads it would conflict with in ▷, ensuring `W ▷ output(W')` correctly detects write-write overlap:

| WriteLoc            | `output()` → ReadLoc set |
|--------------------|--------------------------|
| `Var(x)`           | `{ Var(x) }`             |
| `Col(d, c)`        | `{ Col(d, c) } ∪ { Attr(d, a) \| a ∈ COL_VALUE_ATTRS }` |
| `ColAdd(d, c)`     | `{ Attr(d, a) \| a ∈ COL_ATTRS }` |
| `ColDel(d, c)`     | `{ Col(d, c) } ∪ { Attr(d, a) \| a ∈ COL_ATTRS }` |
| `Rows(d)`          | `{ Attr(d, a) \| a ∈ ROW_ATTRS }` |
| `Attr(d, a)`       | `{ Attr(d, a) }`         |
| `File(p)`          | `{ File(p) }`            |

This lifts to sets: `output*(W) = ⋃ { output(w) | w ∈ W }`.

**Key insight:** Structural writes (`ColAdd`, `ColDel`, `Rows`) expand to multiple output reads because they affect shared structural attributes. For example, `output(Rows(d))` = `{Attr(d, a) | a ∈ ROW_ATTRS}` rather than `Var(d)`, because row changes affect shape, index, etc. — not the variable binding itself.

**Note on Rows and columns:** `Rows(d) ▷ Col(d, *)` in ▷, but `output(Rows(d))` does not include column reads because we cannot enumerate column names at the loc level. Write-write overlap between `Rows(d)` and `Col(d, c)` is detected via `Rows(d) ▷ output(Col(d, c))`: since `output(Col(d, c))` contains `Col(d, c)` and `Rows(d) ▷ Col(d, c) = True`, overlap is always detected. The reverse direction (`Col(d,c) ▷ output(Rows(d))`) also works because `output(Rows(d))` contains `Attr(d, values)` and `Attr(d, T)`, both in `COL_VALUE_ATTRS`. Note: the current implementation uses variable-name-level write-write overlap (more conservative than the typed `output()` approach), so this is doubly covered.

### Forward Staleness Check

When cell `i` executes, for each later cell `j`:

1. **Read-based staleness:** `W'ᵢ ▷ Rⱼ ≠ ∅` — did `i`'s writes invalidate what `j` previously read?
2. **Write-based staleness:** `W'ᵢ ▷ output*(Wⱼ) ≠ ∅` — did `i`'s writes overlap with what `j` writes?

Both checks use the same ▷ relation. The second catches cases like: cell A and cell B both write `df["price"]`. If cell A re-executes with a new value, cell B is stale — its write was computed from outdated inputs.

### Effective Write-Write Conflict Matrix

Composing `output()` with ▷ yields the effective write-write table. An entry is `True` when executing cell `i` (row) makes cell `j`'s write (column) stale.

| Cell `i` wrote ↓  \  Cell `j` wrote → | **Var(x)** | **Col(d, c)** | **ColAdd(d, c)** | **ColDel(d, c)** | **Rows(d)** | **Attr(d, a)** | **File(p)** |
|---------------------------------------|------------|---------------|------------------|------------------|-------------|----------------------|-------------|
| **Var(x)**                            | same `x`   | `x == d`      | `x == d`, `a ∈ COL_ATTRS` | `x == d`  | `x == d`, `a ∈ ROW_ATTRS` | `x == d`             | —           |
| **Col(d, c)**                         | —          | same `d,c` or `a ∈ COL_VALUE_ATTRS` | `a ∈ COL_VALUE_ATTRS` | same `d,c` or `a ∈ COL_VALUE_ATTRS` | same `d`, `a ∈ ROW_ATTRS ∩ COL_VALUE_ATTRS` | `a ∈ COL_VALUE_ATTRS` | —           |
| **ColAdd(d, c)**                      | —          | —             | same `d`, `a ∈ COL_ATTRS` | —         | —           | `a ∈ COL_ATTRS`      | —           |
| **ColDel(d, c)**                      | —          | same `d,c`    | same `d`, `a ∈ COL_ATTRS` | same `d,c` or `a ∈ COL_ATTRS` | —  | `a ∈ COL_ATTRS`      | —           |
| **Rows(d)**                           | —          | same `d`      | same `d`, `a ∈ COL_ATTRS` | same `d`  | same `d`    | `a ∈ ROW_ATTRS`      | —           |
| **Attr(d, a)**                        | —          | —             | `a ∈ COL_ATTRS`  | `a ∈ COL_ATTRS`  | `a ∈ ROW_ATTRS` | same `d,a`       | —           |
| **File(p)**                           | —          | —             | —                | —                | —           | —                    | same `p`    |

(**—** = no write-write staleness)

Notable changes from the corrected `output()`:
- **`ColAdd` now detects overlap** with other `ColAdd`, `ColDel`, and `Attr` writes on the same DataFrame, because `output(ColAdd)` projects to `COL_ATTRS` attributes.
- **`Rows` detects overlap** with other `Rows` writes and with `Attr` writes on `ROW_ATTRS`, because `output(Rows)` projects to `ROW_ATTRS` attributes (not `Var(d)`).
- **`Col` vs `Rows`** overlap is asymmetric: `Rows(d) ▷ output(Col(d,c)) = True` (Rows conflicts with Col reads), but `Col(d,c) ▷ output(Rows(d)) = False` (Col doesn't conflict with Attr reads). The direct `▷` check on `Rⱼ` catches the Col→Rows direction if cell j reads any columns.


## Stable Object Identity via StableIdMap

In the grammars above, the qualifier `d` in `Col(d, c)`, `Attr(d, a)`, etc. is a
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

| Scenario | Action | Result |
|----------|--------|--------|
| Same object | `ref() is obj` → return existing stable_id | ✓ |
| Alias (`df2 = df`) | Same object → same stable_id | ✓ |
| User copy (`df.copy()`) | Different object → new stable_id | ✓ |
| id reuse after GC | `ref()` dead → new stable_id | ✓ |
| Our deepcopy (checkpoint) | `apply_memo()` transfers stable_id | ✓ |

### How ▷ uses qualifiers

The ▷ relation uses two comparison modes:

- **`_same_dataframe(w.qualifier, r.qualifier)`** — for DataFrame-to-DataFrame checks
  (Col, ColAdd, ColDel, Rows, Attr). Compares `loc_id`s when both are LocRef.
  Aliased DataFrames match automatically.
- **`_var_targets_ref(w.name, r.qualifier)`** — for `Var(x) ▷ Col(d, c)` bridging.
  Compares `w.name` against `r.qualifier.var_name`. Var rebinding only invalidates
  reads that went through that specific variable name.

### The `Var(x) ▷ Col(d, c)` cross-domain rule

The conflict matrix has `Var(x) ▷ Col(d, c) = True` when `x == d.var_name`.
This catches rebinding: when `df = new_value`, all column reads that went through
the name `"df"` are invalidated. Reads through an alias (`df2 = df`) are NOT
invalidated by `Var("df")` because `"df" ≠ "df2"` — which is correct, since `df2`
still points to the original object.

### Backward compatibility

When `StableIdMap` is not available (e.g., in tests), qualifiers fall back to plain
strings. The `_same_dataframe()` helper handles mixed comparisons:
`_same_dataframe(LocRef(42, "df"), "df")` returns `True` via var_name matching.

See `FORMAL_DEVELOPMENT.md` §9.1 for the full design analysis.

**Code:**
- StableIdMap + LocRef: `kernel/loc_ids.py`
- Qualifier comparison: `_same_dataframe()`, `_var_targets_ref()` in `kernel/locations.py`
- Memo transfer: `_apply_restore_memo()` in `kernel/flowbook_kernel.py`
- Alias detection (Phase 1 safety net): `_expand_with_deep_aliases()` in `kernel/reproducibility_enforcer.py`

## Design Summary

The entire conflict system rests on three primitives:

1. **`ReadLoc` / `WriteLoc`** — typed locations that encode *what* was accessed and *how* it changed.
2. **`▷` (`write_conflicts_read`)** — a single 7×4 function that answers "does this write invalidate this read?"
3. **`output()`** — a projection from writes to reads, enabling write-write overlap to be expressed as `▷` over projected reads.

All higher-level predicates — `BackConflict`, `FwdContaminated`, `StaleFwd` — are defined compositionally from these three primitives. There is no separate conflict rules table, no ad-hoc special cases. Adding a new write type (e.g., `DtypeChanged`) requires only:

- A new arm in `write_conflicts_read()` (which reads invalidate)
- A new arm in `output()` (which read observes the new value)
- A mapping in `changes_to_write_locs()` (how to detect it from diffs)
