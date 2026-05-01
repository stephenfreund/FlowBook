---
description: 'Fix reproducibility violations in a Jupyter notebook using FlowBook MCP tools. Creates a -fixed copy, runs cells incrementally, fixes violations, and produces a clean reproducible notebook.'
---

# Fix Notebook for Reproducibility

You are fixing reproducibility violations in a Jupyter notebook using FlowBook's MCP tools.

**Input**: $ARGUMENTS (path to a .ipynb file)

## What Reproducibility Analysis Guarantees

**Rerun consistency** means: if all cells are CLEAN, then running the notebook top-to-bottom will reproduce every cell's recorded outputs. FlowBook enforces this by tracking what each cell reads and writes, then checking four predicates after every execution:

1. **NoReadAndWrite**: A cell must not read and write the same variable (re-runs would accumulate changes). Example: `train = pd.concat([train, extra])` — each re-run appends more rows.
2. **WriteBeforeRead**: Every variable a cell reads must have been written by an earlier cell (no dangling references).
3. **NoReadBeforeWrite** (forward contamination): A cell must not read a variable that is written by a _later_ cell (execution order dependency).
4. **NoWriteAfterRead** (backward mutation): A cell must not write a variable that was read by an _earlier_ cell (re-running the writer would change the reader's inputs).

Additionally, **UNRECOVERABLE_MUTATION** detects in-place modifications (like `df.drop(inplace=True)` or `model.fit()`) that FlowBook cannot roll back.

When a violation is found, FlowBook marks cells **stale** — meaning their outputs may no longer match what a top-to-bottom re-run would produce. The goal of fixing is to eliminate all violations so every cell is CLEAN.

## Workflow

### Step 1: Copy and Load

```
# Copy the notebook
cp $ARGUMENTS {stem}-fixed.ipynb

# Load it via MCP
load_notebook("{stem}-fixed.ipynb")
```

### Step 2: Run From First Cell (Baseline)

Run all cells to establish the baseline:

```
run_from("A")
```

This runs from cell @A through the end, stopping on the first error or violation.
Note the violations reported — these are what we need to fix.

### Step 3: Fix Loop (max 10 iterations)

For each iteration:

1. **Checkpoint** before attempting any fix:

   ```
   checkpoint()
   ```

2. **Find the next problem**:

   ```
   get_next_actionable_cell()
   ```

   If it returns "All clean", you're done — go to Step 4.

3. **If the actionable cell has a violation or error**, read it and fix:

   ```
   read_cell(cell_id)
   ```

   Categorize and fix using the taxonomy below.

4. **After fixing, run from the fixed cell** to re-run it and all downstream cells:

   ```
   run_from(cell_id)
   ```

5. **If things got worse**, undo:

   ```
   restore(checkpoint_id)
   ```

   Then try a different strategy. After restore, `run_from` the earliest restored cell.

6. **If no progress** after 2 attempts on the same violation, skip it and move on.

### Step 4: Save and Report

```
save_notebook()
```

Print a summary:

- Original violation count
- Fixes applied table, always in this format:

| Cell | Strategy | Change |
|---|---|---|
| D | `insert_deepcopy` | `df` → `df_D` |

Followed by a diagnosis blockquote for each fix:

> **@D**: `no_read_and_write` + `no_write_after_read` on `df['age']` — reads and writes same column, and @C already read it

- Remaining violations (if any)
- Path to the fixed notebook

## Violation Taxonomy and MCP Tool Usage

### 1. In-place Variable Reassignment

**What it looks like**: A cell reads and overwrites the same variable.
**Error type**: `NO_READ_AND_WRITE`
**Example**: `train = pd.concat([train, extra_data])`

**Fix**: Alpha-rename the variable.

```
checkpoint()
read_cell("B")
alpha_rename("B", "train", "train_combined")
run_from(cell_id)
```

### 2. Invalid Mutation (Unrecoverable)

**What it looks like**: In-place pandas operations or model.fit() calls.
**Error type**: `UNRECOVERABLE_MUTATION`

**Fix A** — For `inplace=True` (most common):

```
checkpoint()
remove_inplace("C", "df")
run_from(cell_id)
```

**Fix B** — For `.fit()` or object mutation:

```
checkpoint()
insert_deepcopy("C", "model")
alpha_rename("C", "model_copy", "model_C")  # Use cell ID in name
run_from(cell_id)
```

**Fix C** — If allocation and mutation are in adjacent cells:

```
checkpoint()
merge_cells(["B", "C"])
run_from(cell_id)
```

**Proactive repair**: Even when `inplace=True` doesn't immediately trigger a violation, it can cause staleness tracking issues later. When you see `inplace=True` in any cell (e.g., `df.drop(..., inplace=True)`, `df.fillna(..., inplace=True)`, `df.reset_index(inplace=True)`), proactively convert it:
```
checkpoint()
remove_inplace("J", "store_data")
run_from(cell_id)
```
This converts `df.method(inplace=True)` to `df = df.method()`, making the mutation explicit and trackable.

### 3. Sequential Transformation Chain

**What it looks like**: Multiple cells transform the same variable in sequence.
**Error type**: `NO_WRITE_AFTER_READ` (backward mutation)
**Example**: Cell @B does `df = df.fillna(0)`, Cell @C does `df = df.assign(feature=...)`.

**Fix A** — Merge tightly coupled steps:

```
checkpoint()
merge_cells(["B", "C"])
run_from(cell_id)
```

**Fix B** — Give each step its own output name:

```
checkpoint()
alpha_rename("C", "df", "df_featured")
run_from(cell_id)
```

### 4. Reusing Variable for Different Purposes

**What it looks like**: A variable holds different data at different points.
**Error type**: `NO_WRITE_AFTER_READ`
**Example**: `model` used for LogisticRegression in cell @B, reassigned to RandomForest in cell @D.

**Fix**: Rename from the point of reuse onwards.

```
checkpoint()
alpha_rename("D", "model", "rf_model")
run_from(cell_id)
```

Choose semantically meaningful names when possible (e.g., `lr_model` / `rf_model` rather than `model_v2`).

### 5. Diagnostic Inspection Before Mutation

**What it looks like**: A read-only cell (df.info(), df.head(), print()) sits above a cell that modifies the variable.
**Error type**: `NO_WRITE_AFTER_READ`

**Fix A** — Mark the inspection cell as diagnostic (preferred for pure inspection):

```
checkpoint()
mark_diagnostic("C")
run_from(cell_id)
```

**Fix B** — Move the inspection after the mutation:

```
checkpoint()
move_cell("C", after_cell_id="D")
run_from(cell_id)
```

### Undoing a Failed Fix

If a fix makes things worse (more violations, or introduces runtime errors):

```
restore("ckpt_abc12345")
# Now try a different strategy...
```

## Guidelines

1. **Preserve functionality**: Fixes must not change what the notebook computes, only how variables are named/scoped.
2. **Minimal changes**: Prefer the smallest change that fixes the violation. `alpha_rename` is usually sufficient.
3. **Use the algorithmic tools**: `alpha_rename`, `remove_inplace`, `insert_deepcopy`, `mark_diagnostic`, `merge_cells`, and `move_cell` are AST-based and reliable. Prefer them over manual `edit_cell_source` when possible.
4. **Use `edit_cell_source` for complex cases**: When the algorithmic tools don't fit (e.g., restructuring logic, adding parameters to functions), fall back to reading the cell, modifying the source manually, and using `edit_cell_source`.
5. **Always checkpoint before fixing**: This lets you safely undo if things go wrong.
6. **Use `run_from(cell_id)`** after each fix to re-run from the fixed cell onwards. It skips clean cells and stops on the first error or violation.
7. **Don't fix staleness directly**: Staleness is a *symptom* of violations. Fix the violation and staleness resolves automatically. But stale cells DO need to be re-run to update the kernel state.
8. **Naming convention for deepcopies**: When `insert_deepcopy` creates a copy, rename it to `{var}_{cell_id}` format (e.g., `df_G` for a copy of `df` introduced in cell G). This is clearer than `df_copy_copy_copy`. After `insert_deepcopy`, use `alpha_rename` to fix the generated name:
   ```
   insert_deepcopy("G", "df")      # Creates df_copy
   alpha_rename("G", "df_copy", "df_G")  # Rename to df_G
   run_from("G")
   ```
9. **Proactively fix inplace operations**: Scan for `inplace=True` in cells and convert them with `remove_inplace` even before they cause violations. In-place operations can cause subtle staleness tracking issues.
