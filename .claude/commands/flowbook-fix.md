---
description: 'Fix reproducibility violations in the currently open notebook using FlowBook tools. Works on the active JupyterLab notebook — no file path needed.'
---

# Fix Notebook for Reproducibility

You are fixing reproducibility violations in the currently open Jupyter notebook. The notebook is already open in JupyterLab with `flowbook_kernel`.

**IMPORTANT**: Use ONLY `mcp__nbi__*` tools (e.g., `mcp__nbi__run_actionable_cells`, `mcp__nbi__checkpoint`, `mcp__nbi__read_cell`). Do NOT use `mcp__flowbook__*` tools — those are for CLI mode and will not update the notebook UI.

**To read the full notebook**, call `mcp__nbi__read_cell` with no arguments (returns all cells in one call). To read a single cell, pass its @-label (e.g., `mcp__nbi__read_cell("@C")`).

**Report every tool call**: Before each `mcp__nbi__*` call, print a one-line status message describing what you're doing (e.g., "Running all actionable cells...", "Checkpointing before fix...", "Renaming 'df' → 'df_clean' from @C...").

## What Reproducibility Analysis Guarantees

**Rerun consistency** means: if all cells are CLEAN, then running the notebook top-to-bottom will reproduce every cell's recorded outputs. FlowBook enforces this by tracking what each cell reads and writes, then checking four predicates after every execution:

1. **NoReadAndWrite**: A cell must not read and write the same variable (re-runs would accumulate changes). Example: `train = pd.concat([train, extra])` — each re-run appends more rows.
2. **WriteBeforeRead**: Every variable a cell reads must have been written by an earlier cell (no dangling references).
3. **NoReadBeforeWrite** (forward contamination): A cell must not read a variable that is written by a _later_ cell (execution order dependency).
4. **NoWriteAfterRead** (backward mutation): A cell must not write a variable that was read by an _earlier_ cell (re-running the writer would change the reader's inputs).

Additionally, **UNRECOVERABLE_MUTATION** detects in-place modifications (like `df.drop(inplace=True)` or `model.fit()`) that FlowBook cannot roll back.

When a violation is found, FlowBook marks cells **stale** — meaning their outputs may no longer match what a top-to-bottom re-run would produce. The goal of fixing is to eliminate all violations so every cell is CLEAN.

## Workflow

### Step 1: Baseline Run

Run all cells to establish the baseline:

```
mcp__nbi__run_actionable_cells()
```

This runs all stale and unexecuted cells, stopping on the first error or violation.
Note the violations reported — these are what we need to fix.

### Step 2: Fix Loop (max 10 iterations)

For each iteration:

1. **Checkpoint** before attempting any fix:

   ```
   mcp__nbi__checkpoint()
   ```

2. **Find the next problem**:

   ```
   mcp__nbi__get_next_actionable_cell()
   ```

   If it returns "All clean.", you're done — go to Step 3.

3. **If the actionable cell has a violation or error**, read it and fix:

   ```
   mcp__nbi__read_cell("@C")
   ```

   Categorize and fix using the taxonomy below.

4. **After fixing, re-run** to propagate changes:

   ```
   mcp__nbi__run_actionable_cells()
   ```

5. **If things got worse**, undo:

   ```
   restore(checkpoint_id)
   ```

   Then try a different strategy. After restore, run `mcp__nbi__run_actionable_cells()` again.

6. **If no progress** after 2 attempts on the same violation, skip it and move on.

### Step 3: Save and Report

```
mcp__nbi__save_notebook()
```

Print a summary:

- Original violation count
- Fixes applied table, always in this format:

| Cell | Strategy          | Change           |
| ---- | ----------------- | ---------------- |
| @D   | `mcp__nbi__insert_deepcopy` | `df` → `df_copy` |

Followed by a diagnosis blockquote for each fix:

> **@D**: `no_read_and_write` + `no_write_after_read` on `df['age']` — reads and writes same column, and @C already read it

- Remaining violations (if any)
- Final reproducibility status

## Violation Taxonomy and Tool Usage

### 1. In-place Variable Reassignment

**What it looks like**: A cell reads and overwrites the same variable.
**Error type**: `NO_READ_AND_WRITE`
**Example**: `train = pd.concat([train, extra_data])`

**Fix**: Alpha-rename the variable.

```
mcp__nbi__checkpoint()
mcp__nbi__read_cell("@B")
mcp__nbi__alpha_rename("@B", "train", "train_combined")
mcp__nbi__run_actionable_cells()
```

### 1b. In-place Column Reassignment

**What it looks like**: A cell reads and overwrites the same column.
**Error type**: `NO_READ_AND_WRITE`
**Example**: `df['x'] = df['x'] + 1`

**Fix**: Make a copy of the dataframe.

```
mcp__nbi__checkpoint()
mcp__nbi__insert_deepcopy("@C", "df")
mcp__nbi__run_actionable_cells()
```

### 2. Invalid Mutation (Unrecoverable)

**What it looks like**: In-place pandas operations or model.fit() calls.
**Error type**: `UNRECOVERABLE_MUTATION`

**Fix A** — For `inplace=True` (most common):

```
mcp__nbi__checkpoint()
mcp__nbi__remove_inplace("@C", "df")
mcp__nbi__run_actionable_cells()
```

**Fix B** — For `.fit()` or object mutation:

```
mcp__nbi__checkpoint()
mcp__nbi__insert_deepcopy("@C", "model")
mcp__nbi__run_actionable_cells()
```

**Fix C** — If allocation and mutation are in adjacent cells:

```
mcp__nbi__checkpoint()
mcp__nbi__merge_cells(["@B", "@C"])
mcp__nbi__run_actionable_cells()
```

### 3. Sequential Transformation Chain

**What it looks like**: Multiple cells transform the same variable or dataframe column in sequence.
**Error type**: `NO_WRITE_AFTER_READ` (backward mutation)
**Example**: Cell @B does `df = df.fillna(0)`, Cell @C does `df = df.assign(feature=...)`.

**Fix A** — Merge tightly coupled steps:

```
mcp__nbi__checkpoint()
mcp__nbi__merge_cells(["@B", "@C"])
mcp__nbi__run_actionable_cells()
```

**Fix B** — Give each step its own output name:

```
mcp__nbi__checkpoint()
mcp__nbi__alpha_rename("@C", "df", "df_featured")
mcp__nbi__run_actionable_cells()
```

### 4. Reusing Variable for Different Purposes

**What it looks like**: A variable holds different data at different points.
**Error type**: `NO_WRITE_AFTER_READ`
**Example**: `model` used for LogisticRegression in cell @B, reassigned to RandomForest in cell @D.

**Fix**: Rename from the point of reuse onwards.

```
mcp__nbi__checkpoint()
mcp__nbi__alpha_rename("@D", "model", "rf_model")
mcp__nbi__run_actionable_cells()
```

Choose semantically meaningful names when possible (e.g., `lr_model` / `rf_model` rather than `model_v2`).

### 5. Diagnostic Inspection Before Mutation

**What it looks like**: A read-only cell (df.info(), df.head(), print()) sits above a cell that modifies the variable.
**Error type**: `NO_WRITE_AFTER_READ`

**Fix A** — Mark the inspection cell as diagnostic (preferred for pure inspection):

```
mcp__nbi__checkpoint()
mcp__nbi__mark_diagnostic("@C")
mcp__nbi__run_actionable_cells()
```

**Fix B** — Move the inspection after the mutation:

```
mcp__nbi__checkpoint()
mcp__nbi__move_cell("@C", "@D")
mcp__nbi__run_actionable_cells()
```

### Undoing a Failed Fix

If a fix makes things worse (more violations, or introduces runtime errors):

```
mcp__nbi__restore("ckpt_0")
# Now try a different strategy...
```

## Guidelines

1. **Preserve functionality**: Fixes must not change what the notebook computes, only how variables are named/scoped.
2. **Minimal changes**: Prefer the smallest change that fixes the violation. `mcp__nbi__alpha_rename` is usually sufficient.
3. **Use the algorithmic tools**: `mcp__nbi__alpha_rename`, `mcp__nbi__remove_inplace`, `mcp__nbi__insert_deepcopy`, `mcp__nbi__mark_diagnostic`, `mcp__nbi__merge_cells`, and `mcp__nbi__move_cell` are AST-based and reliable. Prefer them over manual `mcp__nbi__edit_cell_source` when possible.
4. **Use `mcp__nbi__edit_cell_source` for complex cases**: When the algorithmic tools don't fit (e.g., restructuring logic, adding parameters to functions), fall back to reading the cell, modifying the source manually, and using `mcp__nbi__edit_cell_source`.
5. **Always checkpoint before fixing**: This lets you safely undo if things go wrong.
6. **Use `mcp__nbi__run_actionable_cells()`** after each fix to re-run all stale/unexecuted cells. It stops on the first error or violation.
7. **Don't fix staleness directly**: Staleness is a _symptom_ of violations. Fix the violation and staleness resolves automatically. But stale cells DO need to be re-run to update the kernel state.
8. **All cell references use @A notation**: @A = first code cell, @B = second, etc. Markdown cells are not counted.
9. **Always report tool calls you make**
