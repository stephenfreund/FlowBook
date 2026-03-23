# The Conflict Relation ▷

FlowBook's reproducibility system tracks what each notebook cell *reads* and *writes*, then uses a single conflict relation — **▷** (`write_conflicts_read`) — to determine whether executing one cell invalidates another. This document describes the location types, how they are generated, and the full conflict matrix.

**Notation.** We write **w ▷ r** to mean "write `w` invalidates read `r`." The relation is *asymmetric by construction*: the left operand is always a `WriteLoc` and the right is always a `ReadLoc` — they are different types, so `r ▷ w` is not even well-formed. We use the directed triangle ▷ rather than the symmetric-looking ▷ previously used in source code comments to emphasize this: the write *acts on* the read, not the other way around.

## Location Grammars

Reads and writes are different types. Reads describe *what a cell looked at*; writes describe *what a cell changed and how*. The "how" is what makes column-granular conflict resolution possible: modifying a column's values is a different kind of write than adding a new column, and each invalidates a different set of reads.

### ReadLoc

```
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

```
WriteLoc ::= Var(x)            -- variable completely replaced
           | Col(d, c)         -- column c values modified in place
           | ColAdd(d, c)      -- new column c added to DataFrame d
           | ColDel(d, c)      -- column c removed from DataFrame d
           | Rows(d)           -- rows added or removed from DataFrame d
           | Attr(d, a) -- structural attribute a changed
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
| `ROW_ATTRS` | `index`, `shape`, `size`, `len`, `empty`                                   | Attributes that reveal row structure    |

`shape` and `size` appear in both — they expose both dimensions.

### Read-Write Conflict Matrix

> **`True`** means the write invalidates the read (the cell that did the read is now stale).

| Write `w` ↓  \  Read `r` → | **Var(x)**          | **Col(d, c)**              | **Attr(d, a)**                  | **File(p)**          |
|-----------------------------|---------------------|----------------------------|---------------------------------|----------------------|
| **Var(x)**                  | same `x`            | `x == d`                   | `x == d`                        | —                    |
| **Col(d, c)**               | —                   | same `d` AND same `c`      | —                               | —                    |
| **ColAdd(d, c)**            | —                   | —                          | same `d` AND `a ∈ COL_ATTRS`   | —                    |
| **ColDel(d, c)**            | —                   | same `d` AND same `c`      | same `d` AND `a ∈ COL_ATTRS`   | —                    |
| **Rows(d)**                 | —                   | same `d` (all columns)     | same `d` AND `a ∈ ROW_ATTRS`   | —                    |
| **Attr(d, a)**       | —                   | —                          | same `d` AND same `a`           | —                    |
| **File(p)**                 | —                   | —                          | —                               | same `p`             |

(**—** = never conflicts)

Key observations:

- **`Var(x)` is the nuclear option.** Replacing a variable conflicts with *every* read type on that variable — column reads, attribute reads, everything. The entire binding changed.
- **`Col(d, c)` is maximally precise.** Modifying column values only invalidates reads of that *exact* column. It does not touch attributes or other columns. This is what enables *column independence*: cell A reads `df["qty"]`, cell B writes `df["price"]` → no conflict.
- **`ColAdd(d, c)` does not invalidate existing column reads.** The old columns' data is untouched. It only invalidates structural attributes like `columns` and `shape` that would now reflect the extra column.
- **`ColDel(d, c)` is stricter than `ColAdd`.** It invalidates reads of the deleted column (it no longer exists) *plus* the same structural attributes.
- **`Rows(d)` is column-wide but attribute-narrow.** Every column's data changed (more or fewer values), so all column reads conflict. But only row-structural attributes are affected — `df.columns` is unchanged by adding a row.
- **`Attr(d, a)` is point-to-point.** Only the exact same attribute conflicts. Changing the index does not invalidate reading `dtypes`.


## Write-Write Conflict (Forward Staleness)

There is no separate write-write conflict function. Instead, the system converts one side's writes into reads via the **output function**, then reuses ▷.

### The Output Function

`output(w)` maps each `WriteLoc` to the `ReadLoc` that would *observe* the value that `w` produced:

| WriteLoc            | `output()` → ReadLoc |
|--------------------|----------------------|
| `Var(x)`           | `Var(x)`             |
| `Col(d, c)`        | `Col(d, c)`          |
| `ColAdd(d, c)`     | `Col(d, c)`          |
| `ColDel(d, c)`     | `Col(d, c)`          |
| `Rows(d)`          | `Var(d)`             |
| `Attr(d, a)` | `Attr(d, a)`        |
| `File(p)`          | `File(p)`            |

This lifts to sets: `output*(W) = { output(w) | w ∈ W }`.

### Forward Staleness Check

When cell `i` executes, for each later cell `j`:

1. **Read-based staleness:** `W'ᵢ ▷ Rⱼ ≠ ∅` — did `i`'s writes invalidate what `j` previously read?
2. **Write-based staleness:** `W'ᵢ ▷ output*(Wⱼ) ≠ ∅` — did `i`'s writes overlap with what `j` writes?

Both checks use the same ▷ relation. The second catches cases like: cell A and cell B both write `df["price"]`. If cell A re-executes with a new value, cell B is stale — its write was computed from outdated inputs.

### Effective Write-Write Conflict Matrix

Composing `output()` with ▷ yields the effective write-write table. An entry is `True` when executing cell `i` (row) makes cell `j`'s write (column) stale.

| Cell `i` wrote ↓  \  Cell `j` wrote → | **Var(x)** | **Col(d, c)** | **ColAdd(d, c)** | **ColDel(d, c)** | **Rows(d)** | **Attr(d, a)** | **File(p)** |
|---------------------------------------|------------|---------------|------------------|------------------|-------------|----------------------|-------------|
| **Var(x)**                            | same `x`   | `x == d`      | `x == d`         | `x == d`         | same `x`    | `x == d`             | —           |
| **Col(d, c)**                         | —          | same `d,c`    | same `d,c`       | same `d,c`       | —           | —                    | —           |
| **ColAdd(d, c)**                      | —          | —             | —                | —                | —           | —                    | —           |
| **ColDel(d, c)**                      | —          | same `d,c`    | same `d,c`       | same `d,c`       | —           | —                    | —           |
| **Rows(d)**                           | same `d`   | `d` matches   | `d` matches      | `d` matches      | same `d`    | `d` matches          | —           |
| **Attr(d, a)**                 | —          | —             | —                | —                | —           | same `d,a`           | —           |
| **File(p)**                           | —          | —             | —                | —                | —           | —                    | same `p`    |

(**—** = no write-write staleness)

Notable: **`ColAdd` in cell `i` never triggers write-write staleness for any write type in cell `j`.** Its `output()` is `Col(d, c)`, and `ColAdd ▷ Col(d, c)` is always false — adding a column does not invalidate reading that column's values. This reflects the semantics: if both cells independently add the same column, there is no data-flow dependency between them (though the second execution will overwrite the first's value).


## Design Summary

The entire conflict system rests on three primitives:

1. **`ReadLoc` / `WriteLoc`** — typed locations that encode *what* was accessed and *how* it changed.
2. **`▷` (`write_conflicts_read`)** — a single 7×4 function that answers "does this write invalidate this read?"
3. **`output()`** — a projection from writes to reads, enabling write-write overlap to be expressed as `▷` over projected reads.

All higher-level predicates — `BackConflict`, `FwdContaminated`, `StaleFwd` — are defined compositionally from these three primitives. There is no separate conflict rules table, no ad-hoc special cases. Adding a new write type (e.g., `DtypeChanged`) requires only:

- A new arm in `write_conflicts_read()` (which reads invalidate)
- A new arm in `output()` (which read observes the new value)
- A mapping in `changes_to_write_locs()` (how to detect it from diffs)
