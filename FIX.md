# Unrecoverable Mutation Fix

## The Issue

The reproducibility enforcer uses the write set W_i to propagate staleness to downstream cells. Previously:

```
W_i_union = old_writes | current_writes | changed_vars
```

where `changed_vars = diff.differences.keys()` (what the diff detected as changed). This is **unsound for in-place mutations**: if cell D does `x[5] = 3`, the diff detects `x` changed, so `x` enters the write set. But D only mutated one element — it cannot restore the full value of `x` on re-execution.

### The Problem Scenario

```
Cell A: x = [1, 2, 3, 4, 5, 6]    # writes x
Cell C: x[4] = 99                   # mutates x in place
Cell D: x[5] = 88                   # mutates x in place
Cell E: print(x)                    # prints [1, 2, 3, 4, 99, 88]
```

After running all four cells, `x = [1, 2, 3, 4, 99, 88]` and E prints that.

Now delete cell D. Under the old implementation:
- D was `last_writer` of `x` (because `x` was in D's write set via `changed_vars`)
- Deleting D marks C as stale via BackwardStale (C was the last writer of `x` before D)
- E also becomes stale

If we re-run C and E:
- C executes `x[4] = 99` on the **current** namespace where `x = [1, 2, 3, 4, 99, 88]`
- C only sets `x[4]` — it does NOT restore `x[5]` back to `6` (D's mutation persists)
- E prints `[1, 2, 3, 4, 99, 88]`

But a fresh top-to-bottom run (without D) would produce:
- A: `x = [1, 2, 3, 4, 5, 6]`
- C: `x[4] = 99` → `x = [1, 2, 3, 4, 99, 6]`
- E: prints `[1, 2, 3, 4, 99, 6]`

**The result is non-reproducible.** Re-running "stale" cells does not recover the correct state because partial updates cannot restore the full value of a variable. The cell only mutates one element — it cannot undo mutations made by other (now-deleted) cells.

## The Fix

### 1. New Error Type: UNRECOVERABLE_MUTATION

Added `ErrorType.UNRECOVERABLE_MUTATION` and `ReasonType.UNRECOVERABLE_MUTATION` to the formal model.

An in-place mutation is detected when:
- A variable appears in `diff.differences` (value changed)
- The variable is NOT in `tracking.writes` (not rebound)
- The variable existed before execution (not a new variable)
- The variable has value-level changes (not just column-level)

### 2. Classification: Recoverable vs Unrecoverable

```
recoverable_changed_vars = changed_vars ∩ tracking.writes
unrecoverable_changed_vars = value_level_changed_vars - tracking.writes - new_vars
```

Column-level exception: `df['col'] = val` where `col` IS in `tracking.column_writes` is recoverable.

### 3. Write Set Restriction

Only recoverable changes propagate staleness:
```
W_i_staleness = recoverable_changed_vars  (not all changed_vars)
```

### 4. Last Writer Restriction

Only recoverable changes update `last_writer`:
```
last_writer[x] = cell_id   only if x ∈ tracking.writes
```

A cell that only mutates `x` in place does NOT become `last_writer` of `x`.

### 5. Error Handling

`UNRECOVERABLE_MUTATION` follows the same acceptance pattern as `NO_WRITE_AFTER_READ`:
- **Default**: rejected with rollback, cell marked stale
- **continue_on_violation=True**: accepted, cell stays clean, error still reported

Even when accepted, the staleness restriction still applies — in-place mutations never inflate the write set.

## Why It Works

**Soundness**: Only variables that can be restored by re-execution participate in staleness propagation. If a cell mutates `x` in place, deleting that cell means `x`'s mutation is lost — no remaining cell can reproduce it. By keeping `x` out of the write set, no downstream cell is told "x was written here."

**Backward mutation detection is unchanged**: The backward mutation check (`NoWriteAfterRead`) still uses `diff.differences` (all changes), not just recoverable changes. In-place mutations to variables read by earlier cells are still detected.

**Column tracking preserves precision**: `df['col'] = val` is recoverable because the column write is tracked — re-executing the cell will restore that column.

## Behavioral Differences

| Scenario | Old | New |
|---|---|---|
| `x[5] = 3` (no rebind) | Accepted silently, x enters write set | **ERROR**: unrecoverable mutation |
| `x[5] = 3` with continue_on_violation | Accepted, x in write set, staleness propagates | **ERROR accepted**, x NOT in write set, no staleness |
| `df['col'] = val` (col in column_writes) | Accepted | Accepted (unchanged — recoverable) |
| `df.values[0,0] = 99` (not tracked) | Accepted silently, df enters write set | **ERROR**: unrecoverable mutation |
| `x = x + 1` (rebind) | Staleness propagates | Staleness propagates (unchanged) |
| Backward mutation `x[0]=99` after read | Violation | Violation (unchanged) + additional unrecoverable error |

## How Conflict Detection Works

The system uses a three-level hierarchy of **locations** to determine whether a change in one cell conflicts with a read in another cell. This hierarchy governs both backward mutation detection (NoWriteAfterRead) and staleness propagation (ForwardStale/BackwardStale).

### Location Types

Every read and write is classified into one of four location types:

| Location Type | Notation | Example |
|---|---|---|
| **Variable** `Var(x)` | Whole-variable binding | `x = 5`, `result = compute(x)` |
| **Column** `Col(df, c)` | Single DataFrame column | `df['price']`, `df.loc[:, 'x']` |
| **Structural** `Structural(df, attr)` | DataFrame shape/schema metadata | `df.shape`, `df.columns`, `len(df)` |
| **File** `File(path)` | External file | `pd.read_csv('data.csv')` |

### Access Events (What cells read)

When a cell executes, the tracking system records what it accessed:

- **`VariableRead(x)`** — cell read variable `x` at the whole-variable level (e.g., `y = x + 1`). This is the most conservative: ANY change to `x` is a conflict.
- **`ColumnRead(df, 'price')`** — cell read a specific column (e.g., `df['price'].sum()`). Only changes to that specific column are conflicts.
- **`StructuralRead(df, 'shape')`** — cell read a structural attribute (e.g., `rows, cols = df.shape`). Only structural changes (row/column add/remove) are conflicts; value changes within existing columns are not.

### Change Types (What the diff detects)

After execution, the system diffs the pre- and post-checkpoint to detect what changed:

| Change Type | Detected When | Example |
|---|---|---|
| `ValueChanged(x)` | Variable replaced or non-DataFrame mutated | `x = 10`, `config['key'] = val` |
| `ColumnModified(df, 'price')` | Existing column values changed | `df['price'] = df['price'] * 1.1` |
| `ColumnAdded(df, 'discount')` | New column appeared | `df['discount'] = df['price'] * 0.1` |
| `ColumnRemoved(df, 'temp')` | Column disappeared | `del df['temp']` |
| `RowsAdded(df, count=5)` | More rows than before | `df = pd.concat([df, new_rows])` |
| `RowsRemoved(df, count=3)` | Fewer rows than before | `df = df[df['x'] > 0]` |
| `IndexChanged(df)` | Index labels changed | `df.index = new_index` |
| `DtypeChanged(df, 'x')` | Column dtype changed | `df['x'] = df['x'].astype(float)` |

### Conflict Rules (When does a change conflict with a read?)

Rules are evaluated in order — first match wins. The complete table:

#### Variable-level reads (`VariableRead`)

| Change | Read | Result | Why |
|---|---|---|---|
| **Any change** to `x` | `VariableRead(x)` | **VIOLATION** | Variable-level read is conservative: any modification invalidates it |

> **Example:** Cell A does `y = x + 1` (`VariableRead(x)`). Cell B does `df['price'] *= 2` which is a `ColumnModified(df, 'price')`. Since A reads `x` not `df`, **no conflict** (different variables). But if B changed `x` in any way, it would conflict.

#### Column modifications (`ColumnModified`)

| Change | Read | Result | Why |
|---|---|---|---|
| `ColumnModified(df, 'price')` | `ColumnRead(df, 'price')` | **VIOLATION** | Same column: values that were read have changed |
| `ColumnModified(df, 'price')` | `ColumnRead(df, 'qty')` | **OK** | Different column: modifying price doesn't affect qty |
| `ColumnModified(df, 'price')` | `StructuralRead(df, *)` | **OK** | Modifying values doesn't change structure |

> **Example:** Cell A does `total = df['price'].sum()` (`ColumnRead(df, 'price')`). Cell B does `df['qty'] = df['qty'] * 2` (`ColumnModified(df, 'qty')`). **No conflict** — different columns. But if B did `df['price'] = df['price'] * 1.1`, that IS a conflict.

#### Column additions (`ColumnAdded`)

| Change | Read | Result | Why |
|---|---|---|---|
| `ColumnAdded(df, 'new')` | `ColumnRead(df, 'price')` | **OK** | Adding a new column doesn't change existing column values |
| `ColumnAdded(df, 'new')` | `StructuralRead(df, 'columns')` | **STRUCTURAL** | Adding a column changes `df.columns` |
| `ColumnAdded(df, 'new')` | `StructuralRead(df, 'shape')` | **STRUCTURAL** | Adding a column changes `df.shape` |
| `ColumnAdded(df, 'new')` | `StructuralRead(df, 'len')` | **OK** | Adding a column doesn't change row count |

> **Example:** Cell A does `cols = df.columns` (`StructuralRead(df, 'columns')`). Cell B does `df['discount'] = 0.1` (`ColumnAdded(df, 'discount')`). The structural mode determines the outcome: **VIOLATION** in ENFORCE mode, **WARNING** in WARN mode, **OK** in OFF mode.

#### Column removals (`ColumnRemoved`)

| Change | Read | Result | Why |
|---|---|---|---|
| `ColumnRemoved(df, 'x')` | `ColumnRead(df, 'x')` | **VIOLATION** | Can't read a column that was removed |
| `ColumnRemoved(df, 'x')` | `ColumnRead(df, 'y')` | **OK** | Removing column x doesn't affect reads of column y |
| `ColumnRemoved(df, 'x')` | `StructuralRead(df, 'columns'/'shape')` | **STRUCTURAL** | Removing a column changes structure |

#### Row changes (`RowsAdded`, `RowsRemoved`)

| Change | Read | Result | Why |
|---|---|---|---|
| `RowsAdded/Removed(df)` | `ColumnRead(df, *)` | **VIOLATION** | ALL columns now have more/fewer values |
| `RowsAdded/Removed(df)` | `StructuralRead(df, 'shape'/'len'/'index')` | **STRUCTURAL** | Row count changed |
| `RowsAdded/Removed(df)` | `StructuralRead(df, 'columns'/'dtypes')` | **OK** | Row changes don't affect column names or types |

> **Example:** Cell A does `total = df['price'].sum()` (`ColumnRead(df, 'price')`). Cell B does `df = pd.concat([df, new_rows])` (`RowsAdded(df)`). This is a **VIOLATION** because the price column now has additional values that would change the sum.

#### Index changes (`IndexChanged`)

| Change | Read | Result | Why |
|---|---|---|---|
| `IndexChanged(df)` | `ColumnRead(df, *)` | **OK** | Index labels don't affect column values |
| `IndexChanged(df)` | `StructuralRead(df, 'index'/'axes')` | **STRUCTURAL** | Index is a structural attribute |
| `IndexChanged(df)` | `StructuralRead(df, 'shape'/'columns')` | **OK** | Index change ≠ shape change (same number of rows) |

#### Dtype changes (`DtypeChanged`)

| Change | Read | Result | Why |
|---|---|---|---|
| `DtypeChanged(df, 'x')` | `ColumnRead(df, 'x')` | **WARNING** | Same values, but type behavior may differ |
| `DtypeChanged(df, 'x')` | `ColumnRead(df, 'y')` | **OK** | Different column |
| `DtypeChanged(df, 'x')` | `StructuralRead(df, 'dtypes')` | **STRUCTURAL** | Dtype metadata changed |

### Structural Mode

Conflicts marked **STRUCTURAL** in the tables above have mode-dependent severity:

| Mode | Structural Conflicts Become | Use Case |
|---|---|---|
| `ENFORCE` | **VIOLATION** (rejected) | Strict reproducibility |
| `WARN` | **WARNING** (accepted, shown) | Awareness without blocking |
| `OFF` | **OK** (ignored) | Don't care about structural changes |

Set via `%structural_tracking <off/warn/enforce>`.

### How Conflicts Map to Checks

The conflict rules are used in two places:

1. **Backward mutation check** (`NoWriteAfterRead`): For each earlier **clean** cell, the system matches the current cell's `typed_changes` against the earlier cell's `prior_reads` using the conflict resolver. If any (change, read) pair produces a VIOLATION, the execution is rejected.

2. **Staleness propagation** (`ForwardStale`/`BackwardStale`): Uses set intersection on variable names with column-aware refinement. If cell i changed variable `df` and later cell j reads `df`, staleness depends on whether the specific columns overlap. The `_has_relevant_overlap_by_id` method checks:
   - Variable-level overlap: `changed_vars ∩ cell_reads ≠ ∅`
   - If both sides have column detail: only stale if `changed_columns ∩ read_columns ≠ ∅`
   - If either side lacks column detail: conservative (assume overlap)
   - If cell has structural reads: check structural overlap

### Recoverable vs Unrecoverable (New)

The unrecoverable mutation check adds a layer **before** these conflict checks:

```
For each variable in diff.differences:
  If var ∈ tracking.writes           → RECOVERABLE (rebinding)
  If var has only column changes
    and all changed cols ∈ column_writes → RECOVERABLE (tracked column write)
  Otherwise                           → UNRECOVERABLE (in-place mutation)
```

Only recoverable changes participate in staleness propagation and last_writer updates. Unrecoverable changes still participate in backward mutation detection (NoWriteAfterRead) — the conflict rules above apply to ALL changes regardless of recoverability. The distinction only affects whether the change *propagates forward* to make other cells stale.

## Files Changed

- `flowbook/kernel/models.py` — Added `UNRECOVERABLE_MUTATION` to `ErrorType` and `ReasonType`
- `flowbook/kernel/reproducibility_enforcer.py` — Classification logic, new check method, write set restriction, last_writer restriction, error acceptance
- `flowbook/kernel/tests/test_reproducibility_enforcer.py` — 11 new test cases in `TestUnrecoverableMutation`

## Examples

- `examples/09_UnrecoverableMutation_Lists.ipynb` — Lists and plain variables
- `examples/10_UnrecoverableMutation_Arrays.ipynb` — NumPy arrays
- `examples/11_UnrecoverableMutation_DataFrameColumns.ipynb` — DataFrame column-level distinction
