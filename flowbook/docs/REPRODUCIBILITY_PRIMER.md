# FlowBook Reproducibility Primer

This document is the canonical prose explanation of what FlowBook's reproducibility
analysis guarantees and the predicates it enforces. It is the single source of truth
referenced by:

- The `/fix-notebook` slash command (`.claude/commands/fix-notebook.md`)
- The reproducibility-fixer agent (`.claude/agents/reproducibility-fixer.md`)
- The in-product AI fix suggester (`flowbook/server/fix_suggester.py`)

For the formal mathematical specification of the predicates and transition rules,
see `FORMAL_DEVELOPMENT.md` at the repo root.

---

## What Reproducibility Analysis Guarantees

**Rerun consistency** means: if all cells in a notebook are CLEAN, then running the
notebook top-to-bottom from a fresh kernel will reproduce every cell's recorded
outputs. FlowBook enforces this by tracking what each cell reads and writes — at
both the variable level and the DataFrame-column / row-set level — and then
checking four predicates after every cell execution.

A cell is **CLEAN** when its last execution passed all four predicates with its
current source. A cell is **STALE** when it has been edited since its last
execution, or when another cell's execution has invalidated its outputs (e.g.,
the value it read was later overwritten). Stale cells need to be re-run to
restore confidence; they are a *symptom* of an unresolved dependency, not a
violation in themselves.

## The Four Validity Predicates

Let R_i and W_i denote the read and write sets of cell i.

### 1. NoReadAndWrite

> A cell must not read and write the same location: R_i ∩ W_i = ∅.

If a cell both reads and writes a variable (or column), each re-run would
accumulate changes — the cell is not idempotent under re-execution from a fresh
kernel.

**Example violation**: `train = pd.concat([train, extra])` — each re-run appends
more rows to `train`.

### 2. WriteBeforeRead

> Every variable a cell reads must have been written by an earlier cell:
> R_i ⊆ W_{1..i-1}.

Reading from a name that no earlier cell defines means a top-to-bottom rerun
would crash with `NameError`.

**Example violation**: cell @C uses `model` but no cell @A or @B ever assigns it.

### 3. NoReadBeforeWrite (forward contamination)

> A cell must not read a location that is written by a *later* cell:
> R_i ∩ W_{i+1..n} = ∅.

If cell @B reads `df` and cell @D later assigns `df = ...`, then running
top-to-bottom @B sees the *old* `df`, but the current session has @D's `df`.
The recorded outputs of @B reflect a value that a rerun would not produce.

### 4. NoWriteAfterRead (backward mutation)

> A cell must not write a location that was read by an *earlier* cell:
> W_i ∩ R_{1..i-1} = ∅.

If cell @D writes a location that @B already read, re-running @D would not
change @B's recorded output — but the kernel state is now different from what
@B's outputs claim.

**Example violation**: cell @B does `df = df.fillna(0)` and cell @C does
`df = df.assign(feature=...)`. @C writes `df`, which @B read.

## UNRECOVERABLE_MUTATION

In-place mutation of an object that another cell already read or holds a
reference to cannot be undone by FlowBook's value-restoration machinery, even
when the four predicates pass syntactically. Examples:

- `df.drop(columns=[...], inplace=True)` — mutates the DataFrame directly.
- `df.fillna(0, inplace=True)`, `df.reset_index(inplace=True)`, etc.
- `model.fit(X, y)` — mutates the estimator object in place.
- `lst.append(x)` on a shared list.

These are flagged as `UNRECOVERABLE_MUTATION` violations. The fix is to make
the mutation explicit and trackable (`df = df.method()`) or to operate on a
fresh copy (`model_v2 = deepcopy(model); model_v2.fit(...)`).

## Staleness as a Symptom

When FlowBook flags a cell **stale**, it is communicating that *something* a
predicate would otherwise catch has happened — but the stale cell itself is
not the violator. Two staleness causes:

- **Forward staleness**: cell i wrote a location that some later cell j read
  or wrote. j is now stale because i's new value invalidates j's outputs.
- **Backward staleness**: cell i no longer writes a location that some
  earlier cell j used to write. j was the last writer; now the namespace
  state differs from what j's outputs claim.

Fixing the *violation* (renaming, removing inplace, etc.) and then re-running
the stale cells resolves the staleness. Marking a cell as `%diagnostic`
(read-only inspection that shouldn't influence reproducibility) is another
way to silence staleness on inspection cells.

## Locations: Variables and Columns

A read or write set element is a typed *location*, not just a name. The
location types are:

- **Var**: a top-level Python name (e.g., `train`, `model`).
- **Col**: a single DataFrame column (e.g., `train.age`).
- **Cols**: a set of columns affected (e.g., `train.[age,sex]` after a
  `drop(columns=[...])`).
- **Rows**: a row-set mutation (e.g., row drops, sorts, index changes).
- **File**: a filesystem read or write.

Two locations *conflict* (under the ▷ relation) when a write to one would
invalidate a read of the other. A write to `train` conflicts with a read of
`train.age`; a write to `train.age` conflicts with a read of `train.age` and
with a read of `train` only when the read inspects the column. This typed
analysis is what lets FlowBook give column-precise violation messages
rather than whole-DataFrame ones.
