# FlowBook — reproducibility for Jupyter notebooks

FlowBook is a custom Jupyter kernel (`flowbook_kernel`) and toolset that turns a notebook
into a reproducibility-checked program. It records the read and write set of every cell
execution at fine granularity (variable, column, column-set, row-set, file) and enforces
**rerun consistency**: if every cell is marked CLEAN, running the notebook top-to-bottom
will reproduce the current outputs bit-for-bit.

## Programming model

The notebook is an ordered list of code cells (markdown cells are skipped for indexing).
Cells share a single Python namespace. Each execution of cell `i` is observed by the
kernel, which records:

  - `R[i]` — the set of locations the cell read.
  - `W[i]` — the set of locations the cell wrote.

A *location* is one of: a variable, a named column of a dataframe, a set of columns, a
set of rows, or a file path. When a cell is rejected as violating reproducibility, the
kernel rolls the namespace back to the state it had before the cell ran, so the
notebook's semantics match what a top-to-bottom re-run would produce.

## The four validity predicates

After every execution of cell `i`, the kernel checks four predicates over the current
`R` and `W`. If any predicate fails the execution is rejected (or flagged as a
violation, if continue-after-violation mode is on).

  1. **NoReadAndWrite** — `R[i] ∩ W[i] = ∅`. A cell that both reads and writes the
     same location accumulates state across re-runs (e.g. `train = pd.concat([train, extra])`
     grows `train` every time).
  2. **WriteBeforeRead** — `R[i] ⊆ W[1..i-1]`. Every location a cell reads must have
     been written by some earlier cell. Catches dangling references and typos.
  3. **NoReadBeforeWrite** (forward contamination) — `R[i] ∩ W[i+1..n] = ∅`. A cell
     must not read a location that is only written by a *later* cell; that implies the
     notebook was executed out of order.
  4. **NoWriteAfterRead** (backward mutation) — `W[i] ∩ R[1..i-1] = ∅`. A cell must
     not write a location that an *earlier* cell read; re-running cell `i` would change
     the earlier cell's inputs and break rerun consistency.

In addition, **UNRECOVERABLE_MUTATION** flags in-place mutations (e.g. `df.drop(inplace=True)`,
`model.fit(...)`, `list.append(...)`) that FlowBook cannot roll back. Such cells are
rejected so the namespace is never corrupted.

## Staleness

A CLEAN cell becomes STALE automatically when a nearby cell is edited or re-run in a
way that invalidates its recorded outputs:

  - **Forward staleness** — cell `j > i` reads or writes a location cell `i` just wrote;
    `j`'s outputs may no longer reflect what a top-to-bottom rerun would produce.
  - **Backward staleness** — cell `j < i` was the last writer of a location that `i` no
    longer writes; `j` must be re-run to re-establish its outputs.

Staleness is a *symptom*, not a cause — it resolves automatically once the underlying
violation is fixed and the affected cells are re-run.

## Programming guidelines

To keep every cell CLEAN and preserve rerun consistency, write code that follows these
rules. When proposing code or edits, produce code that obeys them; when diagnosing a
violation, point to the rule that is being broken.

  1. **Give each derived value a new name.** A cell that transforms data should bind
     the result to a fresh variable: prefer `df_imputed = df.fillna(0)` over
     `df = df.fillna(0)`. Reassigning a variable that a later cell still reads triggers
     `NoWriteAfterRead`; reassigning within the same expression triggers `NoReadAndWrite`.

  2. **Never read and write the same location in one cell.** Patterns like
     `train = pd.concat([train, extra])` accumulate across re-runs;
     `df['x'] = df['x'] + 1` mutates a column in place. Introduce a new name, or make a
     copy first (`df2 = df.copy(); df2['x'] = df['x'] + 1`).

  3. **Avoid in-place mutation.** Pandas `inplace=True`, `list.append`, `dict.update`,
     set mutation, and estimator `.fit(...)` calls are unrecoverable — the kernel cannot
     roll them back on failure. Prefer the non-inplace variant that returns a new object
     (`df = df.drop(...)`, `lst = lst + [x]`), or bundle allocation and mutation into a
     single cell so re-running rebuilds the object atomically.

  4. **Keep allocation and its mutation in the same cell.**
     `model = RandomForestClassifier(...)` and `model.fit(X, y)` belong together.
     Splitting them means re-running only the fit cell corrupts an already-fit model
     and flags `UNRECOVERABLE_MUTATION`.

  5. **Break aliasing explicitly.** `X = features` creates two names for the same
     object; mutating either affects both. Use `.copy()` (or `insert_deepcopy`) when you
     need independence. Leave aliases only when you will never mutate either side.

  6. **Read only from above.** A cell may reference only variables defined by an
     earlier cell. Forward references imply out-of-order execution and are rejected as
     forward contamination (`NoReadBeforeWrite`).

  7. **Isolate pure-inspection cells.** `df.info()`, `df.head()`, `print(df)`, and
     plotting calls read the current state. Placing an inspection cell between a
     writer and a later reader of the same variable creates a spurious
     `NoWriteAfterRead`. Either mark the cell diagnostic, or move it after all writers
     of the inspected variable.

  8. **Treat files as tracked locations.** When a cell writes a file, that path is
     part of its write set. Don't read a file in one cell and overwrite the same path
     in a later cell without renaming — the same rules as variables apply.

  9. **One conceptual step per cell.** Small, single-purpose cells keep read/write
     sets minimal and make violations easier to localize. Avoid omnibus cells that
     both build and mutate, or both train and evaluate.

  10. **Verify incrementally.** Run each cell the moment it is written or edited.
      Batching many edits before executing makes errors and violations pile up and
      forces painful bisection; a single-cell failure is almost always easier to
      diagnose than a failure that surfaces after several accumulated changes.

## Fix taxonomy

Each violation has a preferred algorithmic fix (AST-based tools — reliable, prefer
over manual edits):

  - `alpha_rename(@X, old, new)` — rename a variable from cell `@X` onwards.
    Fixes `NoReadAndWrite` on variables and `NoWriteAfterRead` from reusing a name.
  - `insert_deepcopy(@X, var)` — introduce `var = var.copy()` to break aliasing.
    Fixes `NoReadAndWrite` on columns (`df['x'] = df['x'] + 1`) and object mutation.
  - `remove_inplace(@X, var)` — rewrite `df.method(inplace=True)` as `df = df.method()`.
    Fixes `UNRECOVERABLE_MUTATION` from pandas `inplace=True`.
  - `mark_diagnostic(@X)` — exclude a pure-inspection cell (`df.info()`, plots, `print(...)`)
    from read/write tracking. Fixes `NoWriteAfterRead` when the inspection cell precedes
    a legitimate writer.
  - `merge_cells([@X, @Y])` — merge adjacent cells when an allocation and its mutation
    were split across cells. Fixes `UNRECOVERABLE_MUTATION` from `obj = X(); obj.fit(...)`.
  - `move_cell(@X, @Y)` — reorder cells. Useful when inspection cells would be correct
    if placed after the mutating cell.

## Cell addressing

Cells are always referenced by **@-label**: `@A` is the first code cell, `@B` the
second, continuing `@Z, @AA, @AB, ...`. Markdown cells are not counted. Never use raw
integer indices or the kernel's internal cell IDs — always use `@A` notation, and
always use the FlowBook tools (never write-through tools that rewrite cells without
preserving identity).
