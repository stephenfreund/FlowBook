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

*Move down* (d > j): The destination shifts by k after deletion.
```
→^{Move(i..j, d)} = →^{Delete(i..j)} →^{Insert(d-k, c₁...cₖ)}
```

*Move up* (d < i): The destination is unchanged after deletion.
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

- **ForwardStale**: Cell j (after i) becomes stale if i wrote to a location that j reads or writes. Note: only **recoverable** writes (rebound variables and tracked column writes) participate in staleness propagation. In-place mutations that are not recoverable do not propagate staleness.
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
A state (C, O, Σ, T, R, W) is *well-formed* if for every i with Tᵢ = CLEAN,
there exists Σ' such that:

```
Cᵢ, Σ ⇓ Σ', Oᵢ, Rᵢ, Wᵢ
```

and:
1. Σ and Σ' agree except on Overwritten(W, i) = W_{i+1..n}
2. Rᵢ ∩ W_{i..n} = ∅
3. Rᵢ ⊆ W_{1..i-1}

---

## 5. Main Lemma: Well-Formed + All Clean ⟹ Serial Execution

**Lemma 5.1.** If (C, O, Σ, T, R, W) is well-formed and all cells are CLEAN,
then there exists Σ' such that `C ⇓ O, Σ'`.

**Proof.** Define P(i): "C_{1..i} ⇓ O_{1..i}, σᵢ, where Σ and σᵢ agree on W_{1..i} \ W_{i+1..n}."

*Base case.* P(0) holds trivially: ε ⇓ ε, ∅.

*Inductive step.* Assume P(i-1). Since cell i is CLEAN, by the invariant there exists
Σ' such that Cᵢ, Σ ⇓ Σ', Oᵢ, Rᵢ, Wᵢ with Σ and Σ' agreeing except on W_{i+1..n}.

Because Σ and σ_{i-1} agree on Rᵢ (since Rᵢ ⊆ W_{1..i-1} \ W_{i..n}), we obtain
Cᵢ, σ_{i-1} ⇓ Oᵢ, σᵢ with σᵢ and Σ' agreeing on W_{1..i}.

Hence σᵢ and Σ agree on W_{1..i} \ W_{i+1..n}, so P(i) holds.

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

| main.tex | FORMAL_DEVELOPMENT.md | Code |
|----------|----------------------|------|
| S = (C, O, Σ) | §1.1 Standard State | `_cell_order`, cell outputs, kernel `namespace` |
| I = (T, R, W) | §1.2 Instrumentation | `NotebookState` in `kernel/notebook_state.py` |
| Tᵢ ∈ {CLEAN, STALE} | §1.2 | `NotebookState.is_clean(cell_id)` |
| Rᵢ | §1.2 | `TrackingData.reads_before_writes` in `kernel_support/models.py` |
| Wᵢ | §1.2 | `TrackingData.writes` in `kernel_support/models.py` |
| W_{i..j} | §1.3 | `_writes_in_range()` helper in `kernel/reproducibility_enforcer.py` |

### 7.2 Validity Predicates

Validity predicates are implemented inline within `check()`, following the [Inst-Run] structure:

| main.tex | FORMAL_DEVELOPMENT.md | Code |
|----------|----------------------|------|
| NoReadAndWrite(R, W, i) | §3.2 | Implicit in `TrackingData.reads_before_writes` (excludes written-after locations) |
| WriteBeforeRead(R, W, i) | §3.2 | Not strictly enforced (would reject reading undefined variables) |
| NoReadBeforeWrite(R, W, i) | §3.2 | `_check_forward_contamination()` in `check()` |
| NoWriteAfterRead(R, W, i) | §3.2 | `_check_backward_mutation_new()` in `check()` |
| RecoverableMutation(W, i) | §3.2 | `_check_unrecoverable_mutation()` in `check()` |

### 7.3 Staleness Predicates

| main.tex | FORMAL_DEVELOPMENT.md | Code |
|----------|----------------------|------|
| ForwardStale(R, W, i, j) | §3.3 | `_compute_forward_staleness()` in `check()`, `_handle_deletions()` in `kernel/reproducibility_enforcer.py` |
| BackwardStale(W, W', i, j) | §3.3 | Computed via `NotebookState.last_writer` in `check()`, `_handle_deletions()` in `kernel/reproducibility_enforcer.py` |
| LastWriter(W, i, y) | §1.4.2 | `NotebookState.last_writer[loc]` in `kernel/notebook_state.py` |
| Overwritten(W, i) | §1.4.1 | Computed on-demand in staleness checks |

### 7.4 Transition Rules

The `check()` method implements [Inst-Run] exactly, with formal citations in comments:

| main.tex | FORMAL_DEVELOPMENT.md | Code |
|----------|----------------------|------|
| Inst-Edit | §3.4 [Inst-Edit] | `mark_cell_edited()` in `kernel/reproducibility_enforcer.py` |
| Inst-Run | §3.4 [Inst-Run] | `check()` in `kernel/reproducibility_enforcer.py` (lines ~938-1183) |
| Inst-Insert | §3.5 [Inst-Insert] | `set_cell_order()` detecting new cells |
| Inst-Delete | §3.5 [Inst-Delete] | `_handle_deletions()` in `kernel/reproducibility_enforcer.py` |
| Inst-Move-Down/Up | §3.5 [Inst-Move-*] | `_handle_moves()` in `kernel/reproducibility_enforcer.py` |

**[Inst-Run] Implementation Structure:**

| Formal Line | Code Location |
|-------------|---------------|
| `R' = R[i := r]` | STEP 3: `record_execution()` call |
| `W' = W[i := w]` | STEP 3: `record_execution()` call |
| NoReadBeforeWrite check | STEP 2: `_check_forward_contamination()` |
| NoWriteAfterRead check | STEP 2: `_check_backward_mutation_new()` |
| RecoverableMutation check | STEP 2: `_check_unrecoverable_mutation()` |
| `T'ᵢ = CLEAN` | STEP 4: `set_clean(cell_id)` |
| ForwardStale loop | STEP 5: `_compute_forward_staleness()` |
| BackwardStale loop | STEP 5: LastWriter tracking via `NotebookState.last_writer` |

### 7.5 Invariant and Checks

| main.tex | FORMAL_DEVELOPMENT.md | Code |
|----------|----------------------|------|
| Well-formed invariant | §4 Invariant 4.1 | Enforced by staleness tracking + validity checks |
| Preservation lemma | §6 Lemma 6.1 | Verified by `check()` return values |
| ForwardStale propagation | §3.4 T'ⱼ cases | `_compute_forward_staleness()` in `check()` |
| BackwardStale check | §3.4 T'ⱼ cases | `_check_backward_mutation_new()` in `check()` |

### 7.6 Frontend Communication

| Concept | Code |
|---------|------|
| Staleness reasons | `Reason`, `ReasonType` in `kernel/models.py` |
| Metadata output | `flowbook` key in `display_data` output |
| TypeScript types | `IReproducibilityMetadata` in `src/flowbook/types.ts` |
| Metadata extraction | `extractFlowbookMetadata()` in `src/flowbook/executionhook.ts` |

---

## 8. Extensions Beyond Core Formalism

The implementation extends the core formalism with:

### 8.1 Location Types (Loc)

Locations are represented as typed tuples for fine-grained conflict detection:
```
ℓ ∈ Loc ::= Var(x) | Col(df, c) | File(path) | Structural(df, attr)
```

**Code:** `Loc` dataclass in `kernel/models.py` with constructors:
- `Loc.var(name)` — variable location
- `Loc.column(var, column)` — column location
- `Loc.file(path)` — file location
- `Loc.structural(var, attr)` — structural attribute location

### 8.2 Column-Level Tracking

Column-level locations allow `df['price']` changes to not conflict with cells reading only `df['quantity']`.

**Code:** `TrackingData.column_reads`, `column_writes` in `kernel_support/models.py`

Loc-based predicates in `kernel/models.py`:
- `check_loc_conflicts(W_i, R_before_i, mode)` — implements NoWriteAfterRead with column-level precision
- `diff_to_write_locs(diff, tracking)` — converts diff result to LocSet
- `tracking_to_read_locs(tracking)` — converts TrackingData to LocSet
- `get_loc_variables(locs)` — extracts variable names from LocSet

### 8.3 File I/O Tracking

File reads/writes are tracked as locations:
```
ℓ ∈ Loc ::= ... | File(path)
```

**Code:** `TrackingData.file_reads_before_writes`, `file_writes`

### 8.4 Structural Attributes

DataFrame structural attributes (shape, columns, dtypes) can be tracked:
```
ℓ ∈ Loc ::= ... | Structural(df, attr)
```

**Code:** `structural_reads` in `TrackingData`, `StructuralMode` in `kernel/conflict_rules.py`

### 8.5 Staleness Reasons

The implementation tracks *why* a cell is stale for UI display:
```
Reason = CODE_CHANGED | INPUT_CHANGED(loc, cell) | NEVER_EXECUTED | ...
```

**Code:** `Reason`, `ReasonType` in `kernel/models.py`

---

## 9. Known Differences with Implementation

### 9.1 Aliasing and Reference Sharing

The formal model assumes distinct locations. Python has aliasing:
- `x = y` makes both names reference the same object
- DataFrame columns may share underlying arrays

**Implementation:** `_expand_with_deep_aliases()` expands accessed variables to
include all their aliases before computing diffs.

### 9.2 Checkpoint-Based Comparison

Rather than comparing pre/post stores directly, the implementation uses
memory checkpoints that snapshot variable states.

**Code:** `MemoryCheckpoint` in `kernel_support/memory_checkpoint.py`

### 9.3 Conflict Resolution Hierarchy

The implementation uses a rule-based conflict resolver for column-level precision:

**Code:** `CONFLICT_RULES` in `kernel/conflict_rules.py`, `ConflictResolver` in `kernel/conflict_resolver.py`

---

## 10. Staleness Computation Modes

The implementation supports two staleness computation modes that differ in
precision and memory usage.

### 10.1 Syntactic Mode

Uses checkpoint once to compute accurate W_i (what actually changed),
then evaluates all predicates using pure set operations on R and W.

**Checkpoint usage**: Compute diff, then discard.

**Stored state**:
```
R[i] : Set[Loc]    — variables read before write
W[i] : Set[Loc]    — variables that actually changed
```

**Predicates**:
```
NoReadAndWrite(R, W, i)    ≝  Rᵢ ∩ Wᵢ = ∅
WriteBeforeRead(R, W, i)   ≝  Rᵢ ⊆ W_{1..i-1}
NoReadBeforeWrite(R, W, i) ≝  Rᵢ ∩ W_{i+1..n} = ∅
NoWriteAfterRead(R, W, i)  ≝  Wᵢ ∩ R_{1..i-1} = ∅  (clean cells only)

ForwardStale(R, W, W', i, j) ≝  j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
```

**Properties**:
- Staleness is monotonic (once stale, always stale until re-executed)
- Sound but conservative (may over-approximate staleness)
- Memory: O(cells × |variable names|)

### 10.2 Semantic Mode

Uses checkpoint to compute R and W, then stores pre-checkpoints
for semantic value comparison.

**Checkpoint usage**: Store permanently for each executed cell.

**Stored state**:
```
R[i], W[i] : Set[Loc]           — for quick filtering
pre_checkpoint[i] : Σ → Value   — namespace state when cell i executed
```

**Predicates** (with semantic override):
```
ForwardStale_semantic(i, j) ≝
    j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅ ∧ diff(pre_checkpoint[j], Σ, Rⱼ) ≠ ∅
```

**Convergence rule**:
```
Converged(j) ≝ diff(pre_checkpoint[j], Σ, Rⱼ) = ∅
```
If `Converged(j)` and cell j was stale, mark j clean.

**Properties**:
- Staleness is non-monotonic (can be cleared when values converge)
- Precise (only marks stale when values actually differ)
- Memory: O(cells × |variable values|)

### 10.3 Comparison

| Property | Syntactic | Semantic |
|----------|-----------|----------|
| Checkpoint retention | Discard after computing W | Keep permanently |
| Staleness check | (W_i ∪ W'_i) ∩ (R_j ∪ W_j) ≠ ∅ | diff(pre_checkpoint[j], Σ, R_j) ≠ ∅ |
| Un-staleness | Never | When diff becomes empty |
| Memory | O(cells × names) | O(cells × values) |
| Precision | Conservative | Exact |

### 10.4 Example: Diverge-Then-Converge

```
Cell A: reads x (pre_checkpoint_A[x] = 1)
Cell B: writes x = 2
Cell C: writes x = 1
```

| After | Syntactic | Semantic |
|-------|-----------|----------|
| B runs | A stale (W_B ∩ R_A = {x}) | A stale (Σ[x]=2 ≠ pre_A[x]=1) |
| C runs | A still stale | A clean (Σ[x]=1 = pre_A[x]=1) |

### 10.5 Implementation Map

| Concept | Code Location |
|---------|---------------|
| StalenessMode enum | `flowbook/kernel_support/structural_tracking.py` |
| Mode property | `ReproducibilityEnforcer.staleness_mode` |
| Mode setter | `ReproducibilityEnforcer.set_staleness_mode()` |
| Syntactic forward stale | `_compute_forward_staleness_syntactic()` |
| Semantic forward stale | `_compute_forward_staleness_semantic()` |
| Convergence detection | `_check_convergence()` |
| Pre-checkpoint cleanup | `_clear_all_pre_checkpoints()` |
| Magic command | `%staleness_mode` in `flowbook_kernel.py` |

## 11. WRITE_OVERLAP: Why Write Overlaps Need Special Handling

The ForwardStale formula marks cell j stale when cell i's writes overlap with j's reads or writes:

```
ForwardStale(R, W, W', i, j) ≝ j > i ∧ (Wᵢ ∪ W'ᵢ) ∩ (Rⱼ ∪ Wⱼ) ≠ ∅
```

This formula has two distinct overlap cases that require different handling:

### 11.1 Read Overlap vs Write Overlap

**Read Overlap**: `(Wᵢ ∪ W'ᵢ) ∩ Rⱼ ≠ ∅`
- Cell j *reads* a location that cell i wrote
- The value may or may not have changed
- In semantic mode: can be cleared by convergence if `diff(pre_j, Σ, Rⱼ) = ∅`
- Reason type: `FORWARD_STALE`

**Write Overlap**: `(Wᵢ ∪ W'ᵢ) ∩ Wⱼ ≠ ∅`
- Cell j *writes* to a location that cell i also writes
- Both cells modify the same location
- Cannot be cleared by convergence (see §11.2)
- Reason type: `WRITE_OVERLAP`

### 11.2 Why Write Overlap Cannot Converge

Convergence clearing (`Converged(j)`) checks whether a cell's *inputs* have returned
to their expected values. For write overlap:

1. **No read dependency to check**: If the overlap is write-only (j doesn't read the location),
   there's no input to diff against. The convergence predicate `diff(pre_j, Σ, Rⱼ) = ∅`
   may trivially pass even though the execution order matters.

2. **Write order determines final state**: The final value of a location depends on which
   cell writes last. If cells i and j both write to x, re-running j may produce a different
   final state than re-running i then j.

3. **Data flow changed**: Even if the values are identical, the provenance (which cell
   "owns" the location) has changed, which may affect downstream analysis.

### 11.3 Removed Writes: Dependency Removal

A special case occurs when cell i *used to* write a location but no longer does:

```
(W'ᵢ - Wᵢ) ∩ Rⱼ ≠ ∅
```

If cell j reads x, and cell i previously provided x but now doesn't write it:
- The dependency relationship j→i for x is broken
- Cell j's source for x has changed (now comes from elsewhere)
- This is marked as `WRITE_OVERLAP` because convergence won't fix it

**Example**:
```
Cell A (v1): x = 1      # W_A = {x}
Cell B:      print(x)   # R_B = {x}, source of x is A
Cell A (v2): y = 2      # W_A = {y}, no longer writes x
```
After A runs v2, B should be stale: its source of x has changed even though
x's value (from some prior state) hasn't changed.

### 11.4 Implementation in Syntactic vs Semantic Modes

**Syntactic Mode**:
- Computes `read_overlap = W_i_union & cell_reads`
- Computes `write_overlap = W_i_union & cell_writes`
- Read overlaps → `FORWARD_STALE` or `SKIPPED_UPSTREAM`
- Write-only overlaps → `WRITE_OVERLAP`

**Semantic Mode**:
- First checks for removed writes: `(W'_i - W_i) & cell_reads`
- Removed writes overlap → `WRITE_OVERLAP` (no convergence check)
- Write-only overlap → `WRITE_OVERLAP` (no convergence check)
- Read overlap with changed vars → semantic diff check → `FORWARD_STALE` if diff non-empty

### 11.5 Implementation Map

| Concept | Code Location |
|---------|---------------|
| WRITE_OVERLAP enum | `ReasonType.WRITE_OVERLAP` in `models.py` |
| Write overlap detection (syntactic) | `_compute_forward_staleness_syntactic()` line ~1946 |
| Write overlap detection (semantic) | `_compute_forward_staleness_semantic()` line ~2020 |
| Removed writes detection | `_compute_forward_staleness_semantic()` line ~2032 |
| Convergence excludes WRITE_OVERLAP | `_check_convergence()` line ~2124 (only clears FORWARD_STALE) |
