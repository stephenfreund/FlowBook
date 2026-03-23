# Plan: Fix Var(x) Semantics + DataFrame Method Interception

## Problem

`Var(df)` in a read set conflates two meanings:
1. **Binding read** — `df["y"] = ...` accesses the namespace binding `df` to get the object
2. **Data read** — `df.sum()` reads all column data through an untracked method

With `Col(d,c) ▷ Var(x) = true` (current), NoReadAndWrite incorrectly fires for
every column assignment: `df["y"] = ...` reads `Var(df)` and writes `Col(df, y)`.

## Solution

Two changes:

1. **`Var(x)` means "binding only"** — change 5 rules in ▷ so sub-variable writes
   (Col, ColAdd, ColDel, Rows, Attr) don't conflict with Var reads
2. **Intercept DataFrame methods** (`.sum()`, `.mean()`, `.describe()`, etc.) to record
   individual column reads (`Col(df, price)`, `Col(df, qty)`, ...) instead of nothing

No new ReadLoc type. ReadLoc stays at 4 constructors. ▷ stays at 7×4.

## Phase 1: Change ▷ rules for Var reads

### `flowbook/kernel/locations.py`

Change 5 rules in `write_conflicts_read()`:

```
# BEFORE:                              AFTER:
Col(d, c)        ▷ Var(x) ≝ d = x     Col(d, c)        ▷ Var(x) ≝ false
ColAdd(d, c)     ▷ Var(x) ≝ d = x     ColAdd(d, c)     ▷ Var(x) ≝ false
ColDel(d, c)     ▷ Var(x) ≝ d = x     ColDel(d, c)     ▷ Var(x) ≝ false
Rows(d)          ▷ Var(x) ≝ d = x     Rows(d)          ▷ Var(x) ≝ false
Attr(d,a) ▷ Var(x) ≝ d = x     Attr(d,a) ▷ Var(x) ≝ false
```

Only `Var(x) ▷ Var(x)` (variable completely replaced) conflicts with a Var read.

The updated 7×4 matrix:

| Write ↓ \ Read → | Var(x) | Col(d,c) | Attr(d,a) | File(p) |
|---|---|---|---|---|
| Var(x) | x=x' | x=d | x=d | — |
| Col(d,c) | **false** | d=d'∧c=c' | — | — |
| ColAdd(d,c) | **false** | — | d=d'∧a∈COL | — |
| ColDel(d,c) | **false** | d=d'∧c=c' | d=d'∧a∈COL | — |
| Rows(d) | **false** | d=d' | d=d'∧a∈ROW | — |
| Attr(d,a) | **false** | — | d=d'∧a=a' | — |
| File(p) | — | — | — | p=p' |

(**bold** = changed from `d=x` to `false`)

### Behavioral impact

- **NoReadAndWrite**: `df["y"] = ...` no longer fires. R={Var(df)}, W={Col(df,y)},
  Col(df,y) ▷ Var(df) = false. Correct: reading the binding to do a column
  assignment is not a read-modify-write.

- **ForwardStale**: A cell reading only `Var(df)` (binding) is NOT staled by a
  column write. Correct: the binding didn't change.

- **Known false negatives**: Untracked methods like `df.sum()` produce `Var(df)` in
  reads and are NOT staled by column writes. Fixed in Phase 2.

### Tests

Update `test_locations.py`:
- Change 5 existing tests from `assert True` to `assert False` for the changed rules
- Add 2 new tests verifying Var(df) is NOT conflicted by Col/ColAdd writes

Update enforcer tests:
- Tests expecting NoReadAndWrite for column assignments: remove that expectation

## Phase 2: Intercept Tier 1 DataFrame methods

### Strategy

Patch DataFrame methods that read column data to record individual column reads
via the existing column tracking infrastructure. At execution time, the DataFrame
exists, so we enumerate its columns directly.

### `flowbook/kernel_support/column_tracking.py`

Add patches for these methods in `_patch_dataframe_methods()`:

**Aggregation methods** (read all columns, return Series/scalar):
- `sum`, `mean`, `std`, `var`, `min`, `max`, `median`
- `describe`, `corr`, `cov`
- `quantile`, `nunique`

**Transformation methods** (read all columns, return DataFrame):
- `apply` (axis=0 default: each column), `to_numpy`, `to_dict`, `to_records`

**Property access** (read all columns):
- `values` (property — returns numpy array of all column data)

**Pattern for each patch:**

```python
# Aggregation methods — all follow same pattern
_AGG_METHODS = ['sum', 'mean', 'std', 'var', 'min', 'max', 'median',
                'describe', 'corr', 'cov', 'quantile', 'nunique']

for method_name in _AGG_METHODS:
    original = getattr(pd.DataFrame, method_name)
    self._original_methods[f'DataFrame.{method_name}'] = original

    def make_tracked(orig, name):
        def tracked_method(df_self, *args, **kwargs):
            tracker = _get_active_tracker()
            if tracker is not None:
                df_id = id(df_self)
                if df_id in tracker._registered_ids:
                    # Record read of each column
                    for col in df_self.columns:
                        tracker._reads[df_id].add(str(col))
            return orig(df_self, *args, **kwargs)
        tracked_method.__name__ = name
        tracked_method.__doc__ = orig.__doc__
        return tracked_method

    setattr(pd.DataFrame, method_name, make_tracked(original, method_name))
```

For `values` (property):

```python
original_values = pd.DataFrame.values.fget  # get the getter function
def tracked_values(df_self):
    tracker = _get_active_tracker()
    if tracker is not None:
        df_id = id(df_self)
        if df_id in tracker._registered_ids:
            for col in df_self.columns:
                tracker._reads[df_id].add(str(col))
    return original_values(df_self)
pd.DataFrame.values = property(tracked_values)
```

### What this produces

Cell: `result = df.sum()` where df has columns {price, qty, name}

Before (no interception):
```
reads_before_writes = {"df"}
column_reads_before_writes = {}
→ ReadLocSet = {Var(df)}
```

After (with interception):
```
reads_before_writes = {"df"}
column_reads_before_writes = {"df": {"price", "qty", "name"}}
→ ReadLocSet = {Col(df, price), Col(df, qty), Col(df, name)}
```

Note: `Var(df)` is NOT emitted because `tracking_to_readlocset` already skips
`Var(x)` when column detail exists. Column-level precision is automatic.

### Performance

Each patch: ~1-3µs (check active tracker, check registered, iterate columns).
12 methods × 1 call = ~12-36µs per cell. Negligible vs existing ~10-50µs overhead.

### Unpatch in `_unpatch_dataframe_methods()`

Add corresponding unpatching for each new method:

```python
for method_name in _AGG_METHODS:
    key = f'DataFrame.{method_name}'
    if key in self._original_methods:
        setattr(pd.DataFrame, method_name, self._original_methods[key])
```

### Tests

Add to `flowbook/kernel_support/tests/test_column_tracking.py`:

```python
class TestAggregationMethodTracking:
    def test_sum_records_all_columns(self):
        """df.sum() records reads of all columns."""

    def test_mean_records_all_columns(self):
        """df.mean() records reads of all columns."""

    def test_describe_records_all_columns(self):
        """df.describe() records reads of all columns."""

    def test_corr_records_all_columns(self):
        """df.corr() records reads of all columns."""

    def test_apply_records_all_columns(self):
        """df.apply(func) records reads of all columns."""

    def test_values_records_all_columns(self):
        """df.values records reads of all columns."""

    def test_unregistered_df_no_tracking(self):
        """Unregistered DataFrame doesn't record reads."""

    def test_inactive_tracker_no_tracking(self):
        """Inactive tracker doesn't record reads."""
```

~8 new tests.

## Phase 3: End-to-end integration tests

Add to `test_locset_integration.py`:

```python
class TestVarBindingSemantics:

    def test_column_assignment_no_read_and_write(self):
        """df["y"] = expr does NOT trigger NoReadAndWrite."""
        # R = {Var(df)}, W = {Col(df, y)}
        # Col(df, y) ▷ Var(df) = false → no error

    def test_column_write_doesnt_stale_binding_reader(self):
        """Cell that only reads Var(df) is NOT staled by Col write."""
        # A: z = df  (R_A = {Var(df)})
        # B: df["y"] = [1,2,3]  (W_B = {Col(df, y)})
        # Col(df, y) ▷ Var(df) = false → A NOT stale

    def test_var_write_does_stale_binding_reader(self):
        """Cell that reads Var(df) IS staled by Var(df) write."""
        # A: z = df  (R_A = {Var(df)})
        # B: df = pd.DataFrame(...)  (W_B = {Var(df)})
        # Var(df) ▷ Var(df) = true → A stale
```

~3 new tests. (The tracked-method → column-read → staleness path is already tested
by existing column staleness tests once the methods produce Col reads.)

## Phase 4: Update specs

### `FORMAL_DEVELOPMENT.md`

- §8.3: Update ▷ matrix (5 cells change to false)
- §8: Add note: "Var(x) represents a namespace binding read. Sub-variable writes
  (Col, ColAdd, Rows, etc.) do not conflict with Var reads because they do not
  change the binding."

### `LOCSET_UNIFICATION_PLAN.md`

- Update conflict matrix

## Summary

| Phase | What | Lines | New Tests |
|---|---|---|---|
| 1 | Change 5 ▷ rules to false | ~10 | ~7 update |
| 2 | Patch 12 DataFrame methods | ~150 | ~8 |
| 3 | End-to-end integration | ~40 | ~3 |
| 4 | Spec updates | ~20 | — |
| **Total** | | **~220** | **~18** |

No new types. No new constructors. No ▷ matrix expansion.
Just 5 rules change + method interception that produces existing Col locs.
