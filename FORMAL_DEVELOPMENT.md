# Formal Development: Reproducibility Semantics for Computational Notebooks

This document formalizes notebook execution using a two-layer semantics:
a **standard semantics** for basic notebook operations, and an
**instrumented semantics** that tracks read/write sets and cell staleness
to enforce reproducibility.

**Goals.** We prove:

1. The initial notebook state is well-formed.
2. Every notebook operation preserves well-formedness.
3. Well-formed + all cells clean ⟹ a serial execution exists that reproduces the outputs.

---

## 1. State Representation

### 1.1 Standard State

A standard notebook state is a tuple:

```
S = (C, O, Σ)
```

where:

- `C = C₁, ..., Cₙ` — sequence of cell source code
- `O = O₁, ..., Oₙ` — sequence of cell outputs (⊥ if not yet executed)
- `Σ : Loc → Value` — current variable store (kernel state)

### 1.2 Instrumentation

An instrumentation is a tuple:

```
I = (T, R, W)
```

where:

- `Tᵢ ∈ {CLEAN, STALE}` — status of cell i
- `Rᵢ ⊆ Loc` — set of locations read by cell i
- `Wᵢ ⊆ Loc` — set of locations written by cell i

An **instrumented state** combines both: `S · I`.

### 1.3 Notation

We use the following notation for collecting reads/writes over ranges:

```
W_{i..j} = ⋃_{k ∈ [i..j]} Wₖ
R_{i..j} = ⋃_{k ∈ [i..j]} Rₖ
```

Sequence concatenation: `X_{1..i-1}, c, X_{i..n}` inserts element `c` at position `i`.

### 1.4 Auxiliary Definitions

**Definition 1.4.1 (Overwritten).**

```
Overwritten(W, i) ≝ W_{i+1..n}
```

The set of locations written by cells after position i.

**Definition 1.4.2 (LastWriter).**

```
LastWriter(W, i, y) = max { j < i | y ∈ Wⱼ }
```

The last cell before i that wrote location y, or ⊥ if none.

---

## 2. Standard Semantics

The standard semantics operates on states `S = (C, O, Σ)` using single arrows (→).

### 2.1 Standard Evaluation

- `C; Σ ↓ᵢ o · Σ'` — running cell Cᵢ from store Σ produces output o and store Σ'
- `C ↓ O · Σ'` — running cells 1..n from initial store ∅ produces outputs O and final store Σ'

### 2.2 Standard Transition Rules

**[Std-Edit]**

```
(C, O, Σ) →^{Edit(i, c)} (C[i := c], O, Σ)
```

**[Std-Run]**

```
Cᵢ; Σ ↓ o · Σ'
─────────────────────────────────
(C, O, Σ) →^{Run(i)} (C, O[i := o], Σ')
```

### 2.3 Standard Structural Operations

Let `X_{j..k}` denote the subsequence `Xⱼ, ..., Xₖ`.

**[Std-Insert]**

```
C' = C_{1..i-1}, c, C_{i..n}
O' = O_{1..i-1}, ⊥, O_{i..n}
─────────────────────────────────
(C, O, Σ) →^{Insert(i, c)} (C', O', Σ)
```

**[Std-Delete]**

```
C' = C_{1..i-1}, C_{i+1..n}
O' = O_{1..i-1}, O_{i+1..n}
─────────────────────────────────
(C, O, Σ) →^{Delete(i)} (C', O', Σ)
```

**[Std-Move-Down]** (s < d)

```
(C, O, Σ) →^{Delete(s)} S'' →^{Insert(d-1, Cₛ)} S'
─────────────────────────────────────────────────
(C, O, Σ) →^{Move(s, d)} S'
```

**[Std-Move-Up]** (s > d)

```
(C, O, Σ) →^{Delete(s)} S'' →^{Insert(d, Cₛ)} S'
─────────────────────────────────────────────────
(C, O, Σ) →^{Move(s, d)} S'
```

### 2.4 Batch Operations

Batch operations are sequences of single operations. Let `k = j - i + 1`.

**Batch Insert:** Insert k cells `c₁, ..., cₖ` starting at position i:

```
→^{Insert(i, c₁...cₖ)} = →^{Insert(i, c₁)} →^{Insert(i+1, c₂)} ⋯ →^{Insert(i+k-1, cₖ)}
```

**Batch Delete:** Delete cells at positions i through j:

```
→^{Delete(i..j)} = →^{Delete(i)} →^{Delete(i)} ⋯ →^{Delete(i)}  [k times]
```

After each deletion, remaining cells shift down, so position i always holds the next cell to delete.

**Batch Move:** Move cells i through j to position d.
Decompose as batch delete followed by batch insert, saving cells first.
Let `c₁, ..., cₖ = Cᵢ, ..., Cⱼ`.

_Move down_ (d > j): The destination shifts by k after deletion.

```
→^{Move(i..j, d)} = →^{Delete(i..j)} →^{Insert(d-k, c₁...cₖ)}
```

_Move up_ (d < i): The destination is unchanged after deletion.

```
→^{Move(i..j, d)} = →^{Delete(i..j)} →^{Insert(d, c₁...cₖ)}
```

---

## 3. Instrumented Semantics

The instrumented semantics operates on states `S · I` using double arrows (⇒).

### 3.1 Instrumented Evaluation

- `C; Σ ⇓ᵢ o · Σ' · r · w` — running Cᵢ from Σ produces output o, store Σ', read set r, write set w

### 3.2 Validity Predicates

These predicates check that cell i's execution is valid:

```
NoReadAndWrite(R, W, i)    ≝  Rᵢ ∩ Wᵢ = ∅
WriteBeforeRead(R, W, i)   ≝  Rᵢ ⊆ W_{1..i-1}
NoReadBeforeWrite(R, W, i) ≝  Rᵢ ∩ W_{i+1..n} = ∅
NoWriteAfterRead(R, W, i)  ≝  Wᵢ ∩ R_{1..i-1} = ∅
RecoverableMutation(W, i)  ≝  diff(preᵢ, Σ) ⊆ Wᵢ ∪ ColWᵢ
```

- **RecoverableMutation**: All mutations detected by the diff must be recoverable — either the variable was rebound (in Wᵢ) or the column was tracked (in ColWᵢ). In-place mutations not in either set are unrecoverable errors.

### 3.3 Staleness Predicates

These predicates determine when cells become stale:

```
ForwardStale(R, W, W', i, j)       ≝  j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
BackwardStale(W, W', i, j)     ≝  j < i ∧ j = LastWriter(W, i, y) for some y ∈ Wᵢ \ W'ᵢ
```

- **ForwardStale**: Cell j (after i) becomes stale if i wrote to a location that j reads or writes. Note: only **recoverable** writes (rebound variables and tracked column writes) participate in staleness propagation. In-place mutations that are not recoverable do not propagate staleness. **Typed implementation:** Since ReadLoc ≠ WriteLoc, the implementation uses two checks: `(Wᵢ ∪ W'ᵢ) ▷ Rⱼ ≠ ∅` (read overlap) and `(Wᵢ ∪ W'ᵢ) ▷▷ Wⱼ ≠ ∅` (write-write overlap). See §10 for details.
- **BackwardStale**: Cell j (before i) becomes stale if it was the last writer of a location that i no longer writes.

### 3.4 Instrumented Transition Rules

**[Inst-Edit]**

```
S →^{Edit(i, c)} S'
─────────────────────────────────────────────────
S · (T, R, W) ⇒^{Edit(i, c)} S' · (T[i := STALE], R, W)
```

**[Inst-Run]**

```
Cᵢ; Σ ⇓ o · Σ' · r · w
R' = R[i := r]
W' = W[i := w]
NoReadAndWrite(R', W', i)
WriteBeforeRead(R', W', i)
NoReadBeforeWrite(R', W', i)
NoWriteAfterRead(R', W', i)
T'ⱼ = CLEAN           if j = i
    = STALE           if ForwardStale(R, W, W', i, j)
    = STALE           if BackwardStale(W, W', i, j)
    = Tⱼ              otherwise
─────────────────────────────────────────────────
(C, O, Σ) · (T, R, W) ⇒^{Run(i)} (C, O', Σ') · (T', R', W')
```

### 3.5 Instrumented Structural Operations

**[Inst-Insert]**

```
S →^{Insert(i, c)} S'
T' = T_{1..i-1}, STALE, T_{i..n}
R' = R_{1..i-1}, ∅, R_{i..n}
W' = W_{1..i-1}, ∅, W_{i..n}
─────────────────────────────────────────────────
S · (T, R, W) ⇒^{Insert(i, c)} S' · (T', R', W')
```

The new cell is STALE (never executed) with empty read/write sets.
Since Wᵢ = ∅, inserting cannot invalidate any existing cell.

**[Inst-Delete]**

```
S →^{Delete(i)} S'
w = Wᵢ
R'' = R[i:={}]
W'' = W[i:={}]
T''ⱼ = STALE           if ForwardStale(R, W, W'', i, j)
     = STALE           if BackwardStale(W, W'', i, j)
     = Tⱼ              otherwise
R' = R_{1..i-1}, R_{i+1..n}
W' = W_{1..i-1}, W_{i+1..n}
T' = T''_{1..i-1}, T''_{i+1..n}
─────────────────────────────────────────────────
S · (T, R, W) ⇒^{Delete(i)} S' · (T', R', W')
```

Deleting cell i is modeled as clearing its reads and writes (W''=W[i:={}], R''=R[i:={}]),
then applying the same ForwardStale and BackwardStale predicates used in [Inst-Run].
Since W''ᵢ = {}, ForwardStale simplifies to Wᵢ ∩ (Rⱼ ∪ Wⱼ) ≠ ∅ for j > i,
and BackwardStale checks all y ∈ Wᵢ (since Wᵢ \ {} = Wᵢ).

**[Inst-Move-Down]** (s < d)

```
S · I ⇒^{Delete(s)} S'' · I'' ⇒^{Insert(d-1, Cₛ)} S' · I'
─────────────────────────────────────────────────
S · I ⇒^{Move(s, d)} S' · I'
```

**[Inst-Move-Up]** (s > d)

```
S · I ⇒^{Delete(s)} S'' · I'' ⇒^{Insert(d, Cₛ)} S' · I'
─────────────────────────────────────────────────
S · I ⇒^{Move(s, d)} S' · I'
```

Move is the composition of delete and insert. The delete may mark cells stale
via ForwardStale and BackwardStale, and the insert adds a stale cell at the destination.
Batch operations follow the same decompositions as in the standard semantics.

---

## 4. Well-Formedness Invariant

**Invariant 4.1 (Well-formed).**
A state (C, O, Σ, T, R, W) is _well-formed_ if for every i with Tᵢ = CLEAN,
there exists Σ' such that:

```
Cᵢ, Σ ⇓ Σ', Oᵢ, Rᵢ, Wᵢ
```

and:

1. Σ and Σ' agree except on Overwritten(W, i) = W\_{i+1..n}
2. Rᵢ ∩ W\_{i..n} = ∅
3. Rᵢ ⊆ W\_{1..i-1}

---

## 5. Main Lemma: Well-Formed + All Clean ⟹ Serial Execution

**Lemma 5.1.** If (C, O, Σ, T, R, W) is well-formed and all cells are CLEAN,
then there exists Σ' such that `C ⇓ O, Σ'`.

**Proof.** Define P(i): "C*{1..i} ⇓ O*{1..i}, σᵢ, where Σ and σᵢ agree on W*{1..i} \ W*{i+1..n}."

_Base case._ P(0) holds trivially: ε ⇓ ε, ∅.

_Inductive step._ Assume P(i-1). Since cell i is CLEAN, by the invariant there exists
Σ' such that Cᵢ, Σ ⇓ Σ', Oᵢ, Rᵢ, Wᵢ with Σ and Σ' agreeing except on W\_{i+1..n}.

Because Σ and σ*{i-1} agree on Rᵢ (since Rᵢ ⊆ W*{1..i-1} \ W*{i..n}), we obtain
Cᵢ, σ*{i-1} ⇓ Oᵢ, σᵢ with σᵢ and Σ' agreeing on W\_{1..i}.

Hence σᵢ and Σ agree on W*{1..i} \ W*{i+1..n}, so P(i) holds.

Therefore P(n) holds, i.e., C ⇓ O, σₙ. ∎

---

## 6. Preservation: Operations Preserve Well-Formedness

**Lemma 6.1 (Preservation).** Every notebook operation preserves well-formedness.

The Edit case is trivial (it only marks cells stale). Insert adds a STALE cell,
imposing no well-formedness obligation. Delete may mark cells stale via
ForwardStale and BackwardStale. Move composes delete and insert.

For Run(i), we show that the new state S' is well-formed by case analysis on j:

- **Case j < i:** Cell j was CLEAN before and remains CLEAN (Run only cleans i and may mark
  others stale). The validity predicates ensure no backward conflict.

- **Case j = i:** Cell i is now CLEAN. The validity predicates (WriteBeforeRead, etc.)
  establish the well-formedness conditions directly.

- **Case j > i:** If j remains CLEAN, then ForwardStale did not trigger, meaning
  (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) = ∅. The well-formedness conditions transfer from the pre-state.

---

## 7. Implementation Map

This section maps formal concepts across three representations:

- **main.tex** — LaTeX proof document
- **FORMAL_DEVELOPMENT.md** — This document (Markdown specification)
- **Code** — Python/TypeScript implementation

### 7.1 State Representation

| main.tex            | FORMAL_DEVELOPMENT.md | Code                                                                |
| ------------------- | --------------------- | ------------------------------------------------------------------- |
| S = (C, O, Σ)       | §1.1 Standard State   | `_cell_order`, cell outputs, kernel `namespace`                     |
| I = (T, R, W)       | §1.2 Instrumentation  | `NotebookState` in `kernel/notebook_state.py`                       |
| Tᵢ ∈ {CLEAN, STALE} | §1.2                  | `NotebookState.is_clean(cell_id)`                                   |
| Rᵢ                  | §1.2                  | `TrackingData.reads_before_writes` in `kernel_support/models.py`    |
| Wᵢ                  | §1.2                  | `TrackingData.writes` in `kernel_support/models.py`                 |
| W\_{i..j}           | §1.3                  | `_writes_in_range()` helper in `kernel/reproducibility_enforcer.py` |

### 7.2 Validity Predicates

Validity predicates are implemented inline within `check()`, following the [Inst-Run] structure:

| main.tex                   | FORMAL_DEVELOPMENT.md | Code                                                              |
| -------------------------- | --------------------- | ----------------------------------------------------------------- |
| NoReadAndWrite(R, W, i)    | §3.2                  | `_check_no_read_and_write()` using typed `wlocs_conflict_rlocs()` |
| WriteBeforeRead(R, W, i)   | §3.2                  | Not strictly enforced (would reject reading undefined variables)  |
| NoReadBeforeWrite(R, W, i) | §3.2                  | `_check_forward_contamination()` in `check()`                     |
| NoWriteAfterRead(R, W, i)  | §3.2                  | `_check_backward_mutation_new()` in `check()`                     |
| RecoverableMutation(W, i)  | §3.2                  | `_check_unrecoverable_mutation()` in `check()`                    |

### 7.3 Staleness Predicates

| main.tex                   | FORMAL_DEVELOPMENT.md | Code                                                                                                                               |
| -------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| ForwardStale(R, W, i, j)   | §3.3                  | `_compute_forward_staleness()` in `check()`, `_handle_deletions()` in `kernel/reproducibility_enforcer.py`                         |
| BackwardStale(W, W', i, j) | §3.3                  | Computed via `NotebookState.last_writer_for()` in `_compute_backward_staleness()`, `handle_delete()` in `kernel/notebook_state.py` |
| LastWriter(W, i, y)        | §1.4.2                | `NotebookState.last_writer_for(loc, cell_id)` in `kernel/notebook_state.py` — pure computation over W                              |
| Overwritten(W, i)          | §1.4.1                | Computed on-demand in staleness checks                                                                                             |

### 7.4 Transition Rules

The `check()` method implements [Inst-Run] exactly, with formal citations in comments:

| main.tex          | FORMAL_DEVELOPMENT.md | Code                                                          |
| ----------------- | --------------------- | ------------------------------------------------------------- |
| Inst-Edit         | §3.4 [Inst-Edit]      | `mark_cell_edited()` in `kernel/reproducibility_enforcer.py`  |
| Inst-Run          | §3.4 [Inst-Run]       | `check()` in `kernel/reproducibility_enforcer.py`             |
| Inst-Insert       | §3.5 [Inst-Insert]    | `set_cell_order()` detecting new cells                        |
| Inst-Delete       | §3.5 [Inst-Delete]    | `_handle_deletions()` in `kernel/reproducibility_enforcer.py` |
| Inst-Move-Down/Up | §3.5 [Inst-Move-*]    | `_handle_moves()` in `kernel/reproducibility_enforcer.py`     |

**[Inst-Run] Implementation Structure:**

| Formal Line                | Code Location                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `R' = R[i := r]`           | STEP 3: `record_execution()` call                                                          |
| `W' = W[i := w]`           | STEP 3: `record_execution()` call                                                          |
| NoReadAndWrite check       | STEP 2: `_check_no_read_and_write()`                                                       |
| NoReadBeforeWrite check    | STEP 2: `_check_forward_contamination()`                                                   |
| NoWriteAfterRead check     | STEP 2: `_check_backward_mutation_new()`                                                   |
| RecoverableMutation check  | STEP 2: `_check_unrecoverable_mutation()`                                                  |
| `T'ᵢ = CLEAN`              | STEP 4: `set_clean(cell_id)`                                                               |
| ForwardStale loop (reads)  | STEP 5: `_compute_forward_staleness()` — `wlocs_conflict_rlocs(change_wlocs, R_j)`         |
| ForwardStale loop (writes) | STEP 5: `_compute_forward_staleness()` — `wlocs_conflict_wlocs(change_wlocs, W_j)`         |
| BackwardStale loop         | STEP 5: LastWriter via `NotebookState.last_writer_for()` (variable level — coverage check) |

### 7.5 Invariant and Checks

| main.tex                 | FORMAL_DEVELOPMENT.md | Code                                             |
| ------------------------ | --------------------- | ------------------------------------------------ |
| Well-formed invariant    | §4 Invariant 4.1      | Enforced by staleness tracking + validity checks |
| Preservation lemma       | §6 Lemma 6.1          | Verified by `check()` return values              |
| ForwardStale propagation | §3.4 T'ⱼ cases        | `_compute_forward_staleness()` in `check()`      |
| BackwardStale check      | §3.4 T'ⱼ cases        | `_check_backward_mutation_new()` in `check()`    |

### 7.6 Frontend Communication

| Concept             | Code                                                                   |
| ------------------- | ---------------------------------------------------------------------- |
| Staleness reasons   | `Reason`, `ReasonType` in `kernel/models.py`                           |
| Metadata output     | `flowbook` key in `display_data` output                                |
| TypeScript types    | `IReproducibilityMetadata` in `src/flowbook/types.ts`                  |
| Metadata extraction | `_extractReproducibilityMetadata()` in `src/flowbook/executionhook.ts` |

---

## 8. Typed Read/Write Locations and the ▷ Conflict Relation

The implementation uses typed read and write locations with a conflict relation
▷ that provides column-level granularity for all predicates and staleness checks.

### 8.1 Read Locations

Read locations describe what a cell accessed:

```
r ∈ ReadLoc ::= Var(x) | Col(d, c) | Cols(d) | Rows(d) | File(p)
```

| Constructor | Meaning                         | Example               |
| ----------- | ------------------------------- | --------------------- |
| Var(x)      | Whole variable x                | df, config            |
| Col(d, c)   | Column c of DataFrame d         | df["price"]           |
| Cols(d)     | Column structure of DataFrame d | df.columns, df.dtypes |
| Rows(d)     | Row structure of DataFrame d    | df.index, len(df)     |
| File(p)     | File at path p                  | data.csv              |

Note: Cross-cutting attributes (shape, values, etc.) emit both Cols and Rows reads.

**Code:** `ReadLoc` in `kernel/locations.py`, `tracking_to_readlocset()` converts TrackingData

### 8.2 Write Locations

Write locations describe what changed and _how_:

```
w ∈ WriteLoc ::= Var(x) | Col(d, c) | Cols(d) | Rows(d) | File(p)
```

| Constructor | Meaning                                 | Example               |
| ----------- | --------------------------------------- | --------------------- |
| Var(x)      | Variable completely replaced            | x = 42                |
| Col(d, c)   | Column written (add, modify, or delete) | df["price"] = [1,2,3] |
| Cols(d)     | Column structure changed                | dtype changes         |
| Rows(d)     | Rows added/removed                      | df.append(...)        |
| File(p)     | File written                            | df.to_csv("out.csv")  |

**Code:** `WriteLoc` in `kernel/locations.py`, `changes_to_write_locs()` converts Change objects

**Storage:** `NotebookState.writes[cell_id]` stores the union of tracking-derived WriteLocs (Var, Col, Cols, Rows, File — from `tracking_to_writelocset()`, which converts structural mutations recorded at operation time by TrackingData) and diff-derived WriteLocs (Col, Rows — from `changes_to_write_locs()`), filtered to only include diff-derived locs for variables that tracking also considers writes (recoverable mutations). See `record_execution()` in `kernel/notebook_state.py`.

### 8.3 The ▷ Conflict Relation

`w ▷ r` means "writing w invalidates reading r".

**Var(x) semantics**: `Var(x)` as a read means "read the namespace binding" —
the pointer from name `x` to an object. Sub-variable writes (Col,
Rows, Cols) do NOT change the binding, so they do NOT conflict with `Var(x)`.
Only `Var(x)` writes (replacing the entire variable) conflict with `Var(x)` reads.

DataFrame methods like `df.sum()` that read column data are intercepted to produce
individual `Col(d, c)` reads, not `Var(d)`. This ensures column-level staleness
precision.

Key rules:

| Write     | Read       | Conflicts?                                                   |
| --------- | ---------- | ------------------------------------------------------------ |
| Var(x)    | Var(x)     | **Yes** (replacing variable invalidates binding read)        |
| Var(x)    | Col(d, c)  | **No** (rebinding caught via Var(x) read always in read set) |
| Col(d, c) | Var(x)     | **No** (column write doesn't change binding)                 |
| Col(d, c) | Col(d, c') | Only if c = c' (column-level precision)                      |
| Col(d, c) | Cols(d)    | **Yes** if d ≡ d' (column write affects column structure)    |
| Cols(d)   | Col(d, c)  | **Yes** if d ≡ d' (structure change affects column readers)  |
| Cols(d)   | Cols(d)    | **Yes** if d ≡ d'                                            |
| Rows(d)   | Var(x)     | **No** (row change doesn't change binding)                   |
| Rows(d)   | Col(d, c)  | **Yes** (row change affects all column data)                 |
| Rows(d)   | Rows(d)    | **Yes** if d ≡ d'                                            |

Note: Cols ▷ Rows = false and Rows ▷ Cols = false (column structure and row structure are independent dimensions).

**Code:** `write_conflicts_read()` in `kernel/locations.py`

Set-level operations:

- `wlocs_conflict_rlocs(W, R)` — return writes in W that conflict with some read in R
- `has_conflict(W, R)` — boolean W ▷ R ≠ ∅
- `wlocs_conflict_wlocs(W₁, W₂)` — return writes in W₁ that overlap with some write in W₂

Typed predicate helpers (pure functions for unit testing):

- `no_read_and_write(R_i, W_i)` — returns conflicting writes in Wᵢ ▷ Rᵢ
- `no_read_before_write(R_i, W_after)` — forward contamination W\_{i+1..n} ▷ Rᵢ
- `no_write_after_read(W_i, R_before)` — backward mutation Wᵢ ▷ R\_{1..i-1}
- `forward_stale_reads(W_i, R_j)` — read-based forward staleness
- `forward_stale_writes(W_i, W_j)` — write-write overlap Wᵢ ▷▷ Wⱼ

**Code:** `kernel/locations.py`

### 8.4 The ▷▷ Write-Write Conflict Relation

For ForwardStale's write-write overlap, `▷▷ : WriteLoc × WriteLoc → Bool` determines
whether two writes overlap. This is a self-contained 5×5 matrix:

```
w₁ ▷▷ w₂ — do writes w₁ and w₂ overlap?

| w₁ ↓ \ w₂ →   | Var(x') | Col(d',c')      | Cols(d') | Rows(d') | File(p') |
|----------------|---------|-----------------|----------|----------|----------|
| Var(x)         | x = x'  | —               | —        | —        | —        |
| Col(d, c)      | —       | d ≡ d' ∧ c = c' | d ≡ d'   | d ≡ d'   | —        |
| Cols(d)        | —       | d ≡ d'          | d ≡ d'   | —        | —        |
| Rows(d)        | —       | d ≡ d'          | —        | d ≡ d'   | —        |
| File(p)        | —       | —               | —        | —        | p = p'   |
```

This lifts to sets: `W₁ ▷▷ W₂ = { w₁ ∈ W₁ | ∃ w₂ ∈ W₂ . w₁ ▷▷ w₂ }`.

**Code:** `write_conflicts_write()` and `wlocs_conflict_wlocs()` in `kernel/locations.py`

### 8.5 Staleness Reasons

The implementation tracks _why_ a cell is stale for UI display:

```
Reason = CODE_CHANGED | INPUT_CHANGED(loc, cell) | NEVER_EXECUTED | ...
```

**Code:** `Reason`, `ReasonType` in `kernel/models.py`

---

## 9. Known Differences with Implementation

### 9.1 Stable Object Identity via StableIdMap

In the formal model, `Col(d, c)` uses `d ∈ Address` as a stable DataFrame
identity. The paper assumes: _"DataFrame addresses are immutable: address d
always refers to the same DataFrame object."_

The implementation realizes this with **`StableIdMap`** — a weakref-based
side-table that maps Python `id()` to stable integer identifiers. These
stable identifiers survive checkpoint deep copy (via memo dict transfer) and
correctly detect `id()` reuse after garbage collection (via weakref
validation).

#### The core problem: `id()` breaks on deep copy

Python's `id()` cannot serve directly as a stable address because the
checkpoint system uses `deepcopy()` for isolation. Every checkpoint
save/restore creates new objects with new `id()` values:

```
Cell A executes:  total = df["price"].sum()
  → Records: ReadLoc.col(id=0x7f3a, "price")

Cell B violates → namespace rolled back via deep copy
  → df is now a NEW object: id = 0x8b2c

Cell C executes:  df["price"] = new_values
  → Records: WriteLoc.col(id=0x8b2c, "price")

Staleness check:  Col(0x8b2c, "price") ▷ Col(0x7f3a, "price")
  → 0x8b2c ≠ 0x7f3a → False → CONFLICT MISSED
```

#### Solution: Weakref-validated stable IDs with memo transfer

`StableIdMap` assigns each object a monotonically increasing integer
(`stable_id`) on first encounter. It uses `weakref.ref` to detect when
Python reuses an `id()` for a different object after GC:

```python
def get_stable(self, obj) -> int:
    pid = id(obj)
    entry = self._entries.get(pid)
    if entry is not None:
        stable_id, ref = entry
        if ref() is obj:  # Same object, not id reuse
            return stable_id
    # New object or id reuse → assign fresh stable_id
    stable_id = self._next_id; self._next_id += 1
    self._entries[pid] = (stable_id, weakref.ref(obj))
    return stable_id
```

To survive checkpoint deep copy, the map transfers stable_ids from originals
to their copies using the `deepcopy` memo dict:

```python
def apply_memo(self, memo: Dict[int, Any]) -> None:
    for old_id, new_obj in memo.items():
        entry = self._entries.get(old_id)
        if entry is not None:
            stable_id, _ = entry
            new_pid = id(new_obj)
            self._entries[new_pid] = (stable_id, weakref.ref(new_obj))
```

This is called after every `_take_checkpoint()` and `_restore_checkpoint()`
in `flowbook_kernel.py`.

#### Correctness by scenario

| Scenario                  | `ref() is obj`          | Action                             | Result |
| ------------------------- | ----------------------- | ---------------------------------- | ------ |
| Same object               | True                    | Return existing stable_id          | ✓      |
| Alias (`df2 = df`)        | True (same obj)         | Return same stable_id              | ✓      |
| User copy (`df.copy()`)   | False (different obj)   | Assign new stable_id               | ✓      |
| id reuse after GC         | False (ref dead → None) | Assign new stable_id               | ✓      |
| Our deepcopy (checkpoint) | N/A                     | `apply_memo()` transfers stable_id | ✓      |

#### LocRef: Dual-purpose qualifier

Sub-location qualifiers use `LocRef(loc_id, var_name)` — a frozen dataclass
that carries both the stable identity and the variable name used to access
the object:

```python
@dataclass(frozen=True)
class LocRef:
    loc_id: int    # Stable identity (from StableIdMap)
    var_name: str  # Variable name at access time
```

This dual representation enables two different comparison modes in the ▷
relation:

- **`_same_dataframe(a, b)`**: Compares `LocRef.loc_id` values — used for
  intra-DataFrame conflicts (Col vs Col, Rows vs Col, etc.). Aliases share
  the same loc_id, so `Col(LocRef(42,"df"), "price")` and
  `Col(LocRef(42,"X"), "price")` correctly conflict.

- **`Var(x)` only conflicts with `Var(x)` reads** (simple name equality).
  Rebinding detection for column/attribute readers works because `Var(x)` is
  always present in read sets alongside `Col`/`Cols`/`Rows` reads — see
  `tracking_to_readlocset()`. No cross-domain bridge rule is needed.

#### Relationship with deep alias detection

StableIdMap and the deep alias index (`_build_alias_index` in
`MemoryCheckpoint`) solve **different problems** and are complementary:

- **StableIdMap** gives same-object aliases the same `loc_id`, so the ▷
  relation correctly matches sub-locations across variable names:
  `Col(LocRef(42,"X"), "price") ▷ Col(LocRef(42,"df"), "price")` → True.

- **Deep alias detection** finds different objects that share internal
  mutable state (e.g., two DataFrames sharing an underlying column array,
  or two dicts sharing a nested list). These are different objects with
  different `loc_id`s — the ▷ relation correctly sees them as unrelated.
  But in-place mutation through one affects the other, so the diff step
  must examine both. The deep alias index ensures this.

Neither mechanism subsumes the other. StableIdMap cannot detect shared
internals between different objects. Deep alias detection cannot make ▷
match sub-locations across variable name aliases (it operates at the diff
level, not the conflict relation level).

#### Backward compatibility

When `StableIdMap` is not available (e.g., in unit tests), qualifiers fall
back to plain strings. `_same_dataframe(str, str)` compares strings directly,
preserving the previous behavior. All existing tests pass unchanged.

**Code:**

- StableIdMap, LocRef: `kernel/loc_ids.py`
- Qualifier helpers: `_same_dataframe()`, `_display_qualifier()` in `kernel/locations.py`
- Memo exposure: `MemoryCheckpoints._last_memo` in `kernel_support/memory_checkpoint.py`
- Memo transfer: `_apply_restore_memo()` in `kernel/flowbook_kernel.py`
- Deep alias detection: `_build_alias_index()`, `get_aliases_for_vars()` in `kernel_support/memory_checkpoint.py`

### 9.2 Checkpoint-Based Comparison

Rather than comparing pre/post stores directly, the implementation uses
memory checkpoints that snapshot variable states via deep copy.

The checkpoint system's `deepcopy` creates new Python objects with new
`id()` values. The `StableIdMap` (§9.1) compensates for this by transferring
stable identifiers from originals to copies via the `deepcopy` memo dict.
The memo dict is exposed as `MemoryCheckpoints._last_memo` after each
`save()`, `save_incremental()`, and `restore()` call, and
`flowbook_kernel.py` calls `stable_map.apply_memo(memo)` after every
checkpoint operation.

**Code:** `MemoryCheckpoint` in `kernel_support/memory_checkpoint.py`

### 9.3 Conflict Resolution

The implementation uses typed read/write locations (`ReadLoc`/`WriteLoc`) with a conflict
relation (`▷`) for column-level precision. All conflict detection uses
`wlocs_conflict_rlocs(W, R)` which computes the set of writes in W that conflict
with some read in R, using `write_conflicts_read()` as the per-element check.

With `LocRef` qualifiers (§9.1), the ▷ relation uses two distinct comparison
modes:

- **Intra-DataFrame** (`Col ▷ Col`, `Rows ▷ Col`, `Col ▷ Cols`, etc.):
  Uses `_same_dataframe()` to compare `loc_id` values. Aliased DataFrames
  (`X = df`) share the same `loc_id`, so cross-alias conflicts are detected
  natively without additional alias expansion.

- **Var(x)** only conflicts with `Var(x)` reads (simple name equality).
  No cross-domain bridge is needed because `Var(x)` is always present in
  read sets alongside `Col`/`Cols`/`Rows` reads (see `tracking_to_readlocset()`).
  When `df = new_value`, the read set `{Var("df"), Col(df, "price"), ...}`
  ensures `Var("df") ▷ Var("df") = true` catches the rebinding. Column
  independence is preserved because `Col/Rows/Cols ▷ Var = false`.

**Code:** `write_conflicts_read()`, `wlocs_conflict_rlocs()` in `kernel/locations.py`

---

## 10. Staleness Computation

Staleness is computed using checkpoint-based diffs and pure set operations on
read and write sets.

Uses checkpoint once to compute accurate W_i (what actually changed),
then evaluates all predicates using pure set operations on R and W.

**Checkpoint usage**: Compute diff, then discard.

**Stored state**:

```
R[i] : ReadLocSet     — locations read before write (Var, Col, Cols, Rows, File)
W[i] : WriteLocSet    — locations that actually changed (Var, Col, Cols, Rows, File)
```

**Predicates** (using ▷ conflict relation for column-level precision):

```
NoReadAndWrite(R, W, i)    ≝  Wᵢ ▷ Rᵢ = ∅
WriteBeforeRead(R, W, i)   ≝  ∀ r ∈ Rᵢ . r ∈ ambient ∨ ∃ j < i . Wⱼ ▷ {r} ≠ ∅
NoReadBeforeWrite(R, W, i) ≝  W_{i+1..n} ▷ Rᵢ = ∅
NoWriteAfterRead(R, W, i)  ≝  Wᵢ ▷ R_{1..i-1} = ∅  (clean cells only)

ForwardStale(R, W, W', i, j) ≝  j > i ∧ (
    (Wᵢ ∪ W'ᵢ) ▷ Rⱼ ≠ ∅                   — write-read conflict
    ∨ (Wᵢ ∪ W'ᵢ) ▷▷ Wⱼ ≠ ∅               — write-write overlap
)
```

Note: The paper uses `∩` (set intersection) because R and W share a single Loc type.
The implementation uses `▷` because ReadLoc and WriteLoc are different types.
The `▷▷` relation handles write-write overlap directly (no conversion needed).

**Properties**:

- Staleness is monotonic (once stale, always stale until re-executed)
- Sound but conservative (may over-approximate staleness)
- Memory: O(cells × |variable names|)

### 10.1 Implementation Map

| Concept                 | Code Location                            |
| ----------------------- | ---------------------------------------- |
| Syntactic forward stale | `_compute_forward_staleness_syntactic()` |

## 11. WRITE_OVERLAP: Why Write Overlaps Need Special Handling

The ForwardStale formula marks cell j stale when cell i's writes overlap with j's reads or writes:

```
ForwardStale(R, W, W', i, j) ≝ j > i ∧ (
    (Wᵢ ∪ W'ᵢ) ▷ Rⱼ ≠ ∅                   — read overlap
    ∨ (Wᵢ ∪ W'ᵢ) ▷▷ Wⱼ ≠ ∅               — write-write overlap
)
```

This formula has two distinct overlap cases that require different handling:

### 11.1 Read Overlap vs Write Overlap

**Read Overlap**: `(Wᵢ ∪ W'ᵢ) ▷ Rⱼ ≠ ∅`

- Cell j _reads_ a location that cell i wrote
- The value may or may not have changed
- Reason type: `FORWARD_STALE`

**Write Overlap**: `(Wᵢ ∪ W'ᵢ) ▷▷ Wⱼ ≠ ∅`

- Cell j _writes_ to a location that cell i also writes
- Both cells modify the same location
- Reason type: `WRITE_OVERLAP`

### 11.2 Why Write Overlap Needs Special Handling

1. **Write order determines final state**: The final value of a location depends on which
   cell writes last. If cells i and j both write to x, re-running j may produce a different
   final state than re-running i then j.

2. **Data flow changed**: Even if the values are identical, the provenance (which cell
   "owns" the location) has changed, which may affect downstream analysis.

### 11.3 Removed Writes: Dependency Removal

A special case occurs when cell i _used to_ write a location but no longer does:

```
(W'ᵢ - Wᵢ) ∩ Rⱼ ≠ ∅
```

If cell j reads x, and cell i previously provided x but now doesn't write it:

- The dependency relationship j→i for x is broken
- Cell j's source for x has changed (now comes from elsewhere)
- This is marked as `WRITE_OVERLAP`

**Example**:

```
Cell A (v1): x = 1      # W_A = {x}
Cell B:      print(x)   # R_B = {x}, source of x is A
Cell A (v2): y = 2      # W_A = {y}, no longer writes x
```

After A runs v2, B should be stale: its source of x has changed even though
x's value (from some prior state) hasn't changed.

### 11.4 Implementation

- Computes `read_overlap = W_i_union & cell_reads`
- Computes `write_overlap = W_i_union & cell_writes`
- Read overlaps → `FORWARD_STALE`
- Write-only overlaps → `WRITE_OVERLAP`

### 11.5 Implementation Map

| Concept                 | Code Location                             |
| ----------------------- | ----------------------------------------- |
| WRITE_OVERLAP enum      | `ReasonType.WRITE_OVERLAP` in `models.py` |
| Write overlap detection | `_compute_forward_staleness_syntactic()`  |

## 12. Structural Mutation Tracking

### 12.1 The Re-execution Problem and Solution

Write locations are determined by diffing memory checkpoints. On first execution of
`df['x'] = 5`, column `x` is absent in the pre-checkpoint and present in the
post-checkpoint, producing `ColumnAdded`. On re-execution, `x` exists in both
checkpoints (from the prior run), so the diff produces `ColumnModified` instead.

Both `ColumnAdded` and `ColumnModified` now map to `Col(d, x)` in `changes_to_write_locs()`.
Since `Col(d, c) ▷ Cols(d)`, the conflict with structural reads is
detected regardless of whether the column was added or modified. This eliminates the
re-execution inconsistency that previously required provenance-based write type upgrades.

Other structural changes (row mutations, index changes, dtype changes, column deletions)
are tracked at operation time in `TrackingData`, so their WriteLocs are always
present regardless of whether the diff produces a change on re-execution.

### 12.2 Structural Mutation Tracking via TrackingData

Structural mutations are recorded at operation time by monkey-patched DataFrame hooks.
These hooks write directly into `TrackingData` fields, which are then converted to
WriteLocs by `tracking_to_writelocset()`. This eliminates the need for a separate
post-hoc injection step.

**TrackingData structural fields:**

| Field              | Type              | Tracks                                   |
| ------------------ | ----------------- | ---------------------------------------- |
| `column_writes`    | `{var: set[col]}` | Columns written (add, modify, or delete) |
| `column_deletions` | `{var: set[col]}` | Columns deleted                          |
| `row_mutations`    | `set[var]`        | DataFrames with row count changes        |
| `index_mutations`  | `set[var]`        | DataFrames with index changes            |
| `dtype_changes`    | `{var: set[col]}` | Columns with dtype changes               |

**Conversion to WriteLocs** (`tracking_to_writelocset()`):

| TrackingData field | WriteLoc(s) produced                |
| ------------------ | ----------------------------------- |
| `column_writes`    | `Col(d, c)` for each column         |
| `column_deletions` | `Col(d, c)` for each deleted column |
| `row_mutations`    | `Rows(d)`                           |
| `index_mutations`  | `Rows(d)`                           |
| `dtype_changes`    | `Cols(d)`                           |

Since these WriteLocs come from tracking (not from diffing), they are always present
on re-execution. This solves the empty-diff re-execution problem without requiring
provenance-based injection.

**Monkey-patched hooks recording structural mutations:**

- `__setitem__`: records column write + dtype change detection
- `insert`: records column write
- `__delitem__`: records column deletion
- `.loc.__setitem__`: records row mutation (if row count changes)
- `drop`: records column deletion and/or row mutation
- `_set_axis`: records index mutation (when `axis == 0`, i.e., `df.index = ...`)
- Inplace wrapper (`_wrap_inplace_for_provenance`): applied to `dropna`, `drop_duplicates`,
  `reset_index`, `set_index`, `sort_index`, `sort_values`, `rename`, `fillna`, `replace` —
  checks row count, index identity, and dtypes before/after, records mutations as appropriate

### 12.3 DataFrameProvenance (Reporting Only)

`DataFrameProvenance` still exists in `df.attrs['_flowbook_provenance']` for reporting
which cell first caused each structural effect, but it is no longer used for conflict
detection. Structural conflicts are fully handled by the TrackingData-based WriteLocs
described above.

**DataFrameProvenance fields:**

| Field            | Type                | Tracks                                       |
| ---------------- | ------------------- | -------------------------------------------- |
| `col_origins`    | `{column: cell_id}` | Which cell first created each column         |
| `col_deletions`  | `{column: cell_id}` | Which cell first deleted each column         |
| `dtype_origins`  | `{column: cell_id}` | Which cell first changed each column's dtype |
| `row_mutators`   | `set[cell_id]`      | Cells that mutated the row count             |
| `index_mutators` | `set[cell_id]`      | Cells that mutated the index                 |

**Storage:** Provenance lives in `df.attrs`, which is automatically preserved by
`df.copy(deep=False)` (used by the checkpoint system's deepcopy), `df.copy()`, and
aliasing. Each copy gets an independent `DataFrameProvenance` via `__copy__`/`__deepcopy__`,
so mutations to one do not affect others.

### 12.4 Formal Integration

Because structural mutations are tracked at operation time, the stored write sets W[i]
already contain all structural WriteLocs. The predicates use W directly without any
injection step:

```
NoReadBeforeWrite(R, W, i) ≝  W_{i+1..n} ▷ Rᵢ = ∅
NoWriteAfterRead(R, W, i)  ≝  Wᵢ ▷ R_{1..i-1} = ∅  (clean cells only)
ForwardStale(R, W, W', i, j) ≝  j > i ∧ (
    (Wᵢ ∪ W'ᵢ) ▷ Rⱼ ≠ ∅  ∨  (Wᵢ ∪ W'ᵢ) ▷▷ Wⱼ ≠ ∅
)
```

These are the same predicates from §10 — no additional `Σ` parameter is needed.

### 12.5 Edge Cases

| Case                              | Handling                                                                                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pd.read_csv()` → `df['x'] = 5`   | `record_var_write` sets all CSV columns to cell A. `record_column_write` sets `x` to cell C. WriteLoc for `x` correctly attributed to cell C only.          |
| `df = df.merge(...)`              | New DataFrame → `record_var_write` creates fresh provenance. Prior provenance discarded.                                                                    |
| Checkpoint restore                | `df.copy(deep=False)` preserves `.attrs`. `DataFrameProvenance.__copy__` creates isolated copy. Provenance survives rollback.                               |
| Cell movement/deletion            | Origin references cell_id of a moved/deleted cell. Predicates look up cell position — if not found, cell is skipped. Provenance becomes stale but harmless. |
| Aliasing (`X = df`)               | Same object, same `.attrs` → provenance shared. Correct.                                                                                                    |
| `df.assign(x=val)`                | Returns new DataFrame → `record_var_write` covers it.                                                                                                       |
| Column deleted then recreated     | `record_column_delete` records deleter and clears col_origin. Next `record_column_write` sets new origin.                                                   |
| Idempotent row mutation (re-exec) | `row_mutators` is a set — adding the same cell_id again is a no-op. Provenance persists correctly.                                                          |
| `df.dropna(inplace=True)`         | Inplace wrapper detects row count change → `record_row_mutation`. Dtypes/index also checked.                                                                |
| `df.index = new_idx`              | `_set_axis` patch (axis=0) → `record_index_mutation`.                                                                                                       |
| `df['x'] = df['x'].astype(float)` | `__setitem__` dtype detection compares pre/post dtypes → `record_dtype_change`.                                                                             |

### 12.6 Implementation Map

| Concept                        | Code Location                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| Provenance class               | `DataFrameProvenance` in `kernel_support/column_provenance.py`                                                                     |
| Provenance tracker             | `DataFrameProvenanceTracker` in `kernel_support/column_provenance.py`                                                              |
| Provenance key                 | `PROVENANCE_KEY = '_flowbook_provenance'` in `kernel_support/column_provenance.py`                                                 |
| Column write hook              | `__setitem__` patch in `kernel_support/column_tracking.py`                                                                         |
| Column insert hook             | `insert` patch in `kernel_support/column_tracking.py`                                                                              |
| Column delete hook             | `__delitem__` patch in `kernel_support/column_tracking.py`                                                                         |
| Row mutation hook              | `.loc.__setitem__`, `drop`, inplace wrappers in `kernel_support/column_tracking.py`                                                |
| Index mutation hook            | `_set_axis` patch in `kernel_support/column_tracking.py`                                                                           |
| Dtype change hook              | `__setitem__` dtype detection in `kernel_support/column_tracking.py`                                                               |
| Inplace wrapper                | `_wrap_inplace_for_provenance()` in `kernel_support/column_tracking.py`                                                            |
| Var write hook                 | `TrackingDict.__setitem__` in `kernel_support/tracking.py`                                                                         |
| cell_id threading              | `track_execution(cell_id=...)` in `kernel_support/tracking.py`                                                                     |
| Tracking → WriteLoc conversion | `tracking_to_writelocset()` in `kernel/locations.py`                                                                               |
| Write set storage              | `record_execution()` in `kernel/notebook_state.py`                                                                                 |
| Unit tests                     | `kernel_support/tests/test_column_provenance.py`                                                                                   |
| Integration tests              | `TestForwardContaminationStructuralRead`, `TestStructuralProvenanceIntegration` in `kernel/tests/test_reproducibility_enforcer.py` |
