# FlowBook: A Guided Tour of the Formalization

This document explains the Lean 4 development in this directory
(everything except the executable kernel `FlowBook/Exec.lean`). It is
meant to be **self-contained**: it introduces the problem, then presents
every definition — with its full body — and every theorem statement,
building up to the three correctness results. Only *proofs* are omitted;
all definitions are shown in full and explained.

Everything here is machine-checked in Lean with no `sorry` and no custom
axioms.

---

## The problem: reproducible notebooks

A *computational notebook* (Jupyter, and similar) is a sequence of code
*cells* that a user runs interactively. Cells share state through a
global namespace: running a cell reads some variables, writes others, and
records an *output*. Crucially, the user may run cells **in any order**,
re-run them, edit them, insert and delete them.

This flexibility breaks *reproducibility*. The outputs currently shown in
a notebook may not be the outputs you would get by restarting the kernel
and running every cell from top to bottom. For example, if you run a cell
that defines `x = 1`, then a cell that prints `x`, then go back and edit
the first cell to `x = 2` **without re-running the second**, the notebook
still displays `1` — but a fresh top-to-bottom run would display `2`. The
notebook is *not reproducible*.

FlowBook is a dynamic analysis that watches the reads and writes of each
cell and marks cells **stale** when their displayed output can no longer
be trusted. Its guarantee: **whenever FlowBook reports every cell as
up-to-date ("clean"), the notebook is reproducible.**

This formalization defines an idealized notebook semantics, defines what
reproducibility means, defines FlowBook's analysis as an *instrumented
semantics*, and proves three theorems: the analysis's invariant is
preserved by every user action (**Preservation**), an all-clean notebook
really is reproducible (**Output Consistency**), and the natural strategy
of re-running stale cells always terminates (**Progress**).

---

## A note on Lean notation

A few pieces of syntax recur throughout. If you know basic functional
programming, this is enough to read every definition below:

| Syntax | Meaning |
|---|---|
| `A → B` | function from `A` to `B`; also logical implication when `A`, `B` are propositions |
| `Prop` | the type of propositions (statements that can be proved) |
| `∃ x, P x` / `∀ x, P x` | there exists / for all |
| `∧` `∨` `¬` `↔` | and, or, not, if-and-only-if |
| `Option A` | either `some a` (a value) or `none` (absent) |
| `List A` | a finite list |
| `xs[i]?` | the `i`-th element of a list as an `Option` (`some x` if in range, `none` otherwise) — 0-based |
| `xs.set i x` | the list `xs` with position `i` replaced by `x` |
| `xs.insertIdx i x` | insert `x` at position `i`, shifting later elements right |
| `xs.eraseIdx i` | remove the element at position `i`, shifting later elements left |
| `xs.length` | the number of elements |
| `{ c with field := v }` | the record `c` with one field overwritten |
| `⟨a, b, c⟩` | an anonymous constructor (builds a structure/tuple from its parts) |
| `fun x => e` | an anonymous function |
| `structure … where` | a record type (a bundle of named fields) |
| `inductive … where` | a datatype or an inductively-defined relation, given by its constructors |

The whole development is generic in four types:

| Type | Stands for |
|---|---|
| `Code` | the source code of a cell (kept abstract) |
| `Output` | the output a cell displays (kept abstract) |
| `L` | *locations* — the units of state cells read and write |
| `V` | the *values* stored at locations |

Cell positions are **0-based** here.

---

# Part 1 — The notebook model and standard semantics
### (`FlowBook/Semantics.lean`)

## 1.1 Locations

A *location* is a unit of state a cell can read or write. FlowBook tracks
two granularities: ordinary top-level variables, and **individual columns
of a DataFrame** (so that two cells touching different columns of the
same table do not appear to conflict). None of the metatheory depends on
this structure — it is generic in `L` — but `Loc` is the intended
instantiation:

```lean
inductive Loc (Var Addr Col : Type) where
  | var (x : Var)             -- a top-level variable  x
  | col (d : Addr) (c : Col)  -- column  c  of the DataFrame at address  d
```

## 1.2 Stores

A *store* maps locations to values. `none` means a location is unbound.
The empty store binds nothing.

```lean
def Store (L V : Type) := L → Option V

def Store.empty : Store L V := fun _ => none
```

Two auxiliary relations describe when stores agree. `AgreeExcept σ σ' X`
says `σ` and `σ'` are equal *everywhere outside* the set `X` (here a set
of locations is represented by its membership predicate `L → Prop`);
`AgreeOn` is the dual.

```lean
def AgreeExcept (σ σ' : Store L V) (X : L → Prop) : Prop :=
  ∀ ℓ, ¬ X ℓ → σ ℓ = σ' ℓ

def AgreeOn (σ σ' : Store L V) (X : L → Prop) : Prop :=
  ∀ ℓ, X ℓ → σ ℓ = σ' ℓ
```

## 1.3 Cell evaluation as a black box

Running one cell is modeled by a judgment written informally as

> `c ; Σ ⇓ o · Σ' · r · w`

meaning: executing code `c` in store `Σ` produces output `o`, new store
`Σ'`, the set `r` of locations *read from the incoming store*, and the
set `w` of locations *written*. The actual language runtime (Python, say)
is treated as a **black box**, and evaluation is a *relation*, not a
function, because a cell may be non-deterministic (e.g. it draws a random
number).

The metatheory needs exactly two facts about this black box. They are
bundled as a type class `CellEval` — an interface a concrete runtime must
implement — carrying the evaluation relation `Eval` plus two guarantees:

* **`frame`**: the store changes *only* at written locations. (This is
  what makes `w` genuinely "the write set".)
* **`locality`**: a cell's behavior depends only on what it reads. If we
  start from a different store `τ` that agrees with `Σ` on the read set
  `r`, then *the same execution is available*: same output, same read and
  write sets, and the same values written. (Fixing the read locations
  pins down one execution, even for a non-deterministic cell.)

```lean
class CellEval (Code Output L V : Type) where
  -- the judgment  c ; σ ⇓ o · σ' · r · w
  Eval : Code → Store L V → Output → Store L V → (L → Prop) → (L → Prop) → Prop
  -- unchanged outside the write set:
  frame : Eval c σ o σ' r w → ∀ ℓ, ¬ w ℓ → σ' ℓ = σ ℓ
  -- behavior depends only on the reads:
  locality : Eval c σ o σ' r w → (∀ ℓ, r ℓ → τ ℓ = σ ℓ) →
    ∃ τ', Eval c τ o τ' r w ∧
      (∀ ℓ, w ℓ → τ' ℓ = σ' ℓ) ∧ (∀ ℓ, ¬ w ℓ → τ' ℓ = τ ℓ)
```

The *standard* (uninstrumented) evaluation just forgets the read/write
sets — it is what a plain kernel does:

```lean
def StdEval (c : Code) (σ : Store L V) (o : Output) (σ' : Store L V) : Prop :=
  ∃ r w, Eval c σ o σ' r w
```

## 1.4 User operations

The user interacts with a notebook through five operations:

```lean
inductive Op (Code : Type) where
  | run (i : Nat)              -- execute cell i
  | edit (i : Nat) (c : Code)  -- replace the source of cell i with c
  | insert (i : Nat) (c : Code)-- insert a new cell with code c at position i
  | delete (i : Nat)           -- delete cell i
  | move (s d : Nat)           -- move the cell at position s to position d
```

## 1.5 Notebook state and the standard semantics

An idealized notebook state is `S = (C, O, Σ)`: the sequence of cell
sources `C`, the sequence of most-recent outputs `O`, and the store `Σ`.
We fuse `C` and `O` into a single list of pairs; an output of `none`
means the cell has never run.

```lean
structure StdState (Code Output L V : Type) where
  cells : List (Code × Option Output)
  store : Store L V
```

The **standard semantics** is a relation `S ⟶op S'` giving the effect of
each operation. It is defined by the following inference rules (each
constructor is one rule; its hypotheses appear above the conclusion). The
defining feature is `[Std-Run]`: it lets the user run **any** cell at any
time, regardless of what ran before — this is exactly the permissiveness
that endangers reproducibility.

```lean
inductive StdStep : StdState … → Op Code → StdState … → Prop where
  -- [Std-Run]: evaluate cell i, record its output, update the store
  | run (hcell : st.cells[i]? = some (c, o₀))
        (heval : StdEval c st.store o σ') :
      StdStep st (.run i) ⟨st.cells.set i (c, some o), σ'⟩

  -- [Std-Edit]: replace the source of cell i; output and store unchanged
  | edit (hcell : st.cells[i]? = some (c₀, o₀)) :
      StdStep st (.edit i c) ⟨st.cells.set i (c, o₀), st.store⟩

  -- [Std-Insert]: insert a fresh, never-run cell
  | insert (hle : i ≤ st.cells.length) :
      StdStep st (.insert i c) ⟨st.cells.insertIdx i (c, none), st.store⟩

  -- [Std-Delete]: remove cell i
  | delete (hcell : st.cells[i]? = some (c₀, o₀)) :
      StdStep st (.delete i) ⟨st.cells.eraseIdx i, st.store⟩

  -- [Std-Move-Down] (s < d): a delete followed by an insert
  | moveDown (hlt : s < d) (hcell : st.cells[s]? = some (c₀, o₀))
             (h1 : StdStep st (.delete s) st'')
             (h2 : StdStep st'' (.insert (d - 1) c₀) st') :
      StdStep st (.move s d) st'

  -- [Std-Move-Up] (d < s): a delete followed by an insert
  | moveUp (hlt : d < s) (hcell : st.cells[s]? = some (c₀, o₀))
           (h1 : StdStep st (.delete s) st'')
           (h2 : StdStep st'' (.insert d c₀) st') :
      StdStep st (.move s d) st'
```

*(Implicit variable binders are elided above for readability; the full
declaration names them explicitly.)*

## 1.6 Top-to-bottom execution and reproducibility

The **reference behavior** is a clean top-to-bottom run from the empty
store. `Runs σ cells σ'` holds when executing the cells in order,
starting from `σ`, produces the store `σ'` **and each cell's recorded
output matches what it produces**. (Note the `cons` rule requires the
cell's recorded output to be `some o` and to equal the output produced —
so a `Runs` derivation certifies the outputs, not just the final store.)

```lean
inductive Runs : Store L V → List (Code × Option Output) → Store L V → Prop where
  | nil  : Runs σ [] σ
  | cons (hc : StdEval c σ o σ₁) (hrest : Runs σ₁ rest σ') :
      Runs σ ((c, some o) :: rest) σ'
```

A notebook is **reproducible** (the paper also says *output consistent*)
exactly when some top-to-bottom execution from the empty store
reproduces its recorded outputs. The final store need not match the
user's interactive store — only the visible outputs must agree, so we
only assert *existence* of a resulting store:

```lean
def Reproducible (st : StdState …) : Prop :=
  ∃ σ', Runs Store.empty st.cells σ'

abbrev OutputConsistent := Reproducible
```

This is the property FlowBook aims to guarantee.

---

# Part 2 — FlowBook's analysis as an instrumented semantics
### (`FlowBook/Analysis.lean`)

FlowBook augments each cell with bookkeeping: a **status tag**, and the
**read set** and **write set** observed on the cell's last run. It uses
these to reject runs that a clean top-to-bottom execution could not have
produced, and to propagate staleness.

## 2.1 Instrumented state

A cell's status is `clean` (its recorded output is still valid) or
`stale` (it must be re-run before it can be trusted).

```lean
inductive Tag where | clean | stale
```

We fuse all five per-cell components into one record — code, recorded
output (`none` = never run), tag, read set `R_i`, and write set `W_i`
(sets are membership predicates `L → Prop`):

```lean
structure Cell (Code Output L : Type) where
  code   : Code
  out    : Option Output
  tag    : Tag
  reads  : L → Prop
  writes : L → Prop
```

An instrumented notebook `S · I` is a list of such cells plus a store:

```lean
structure Notebook (Code Output L V : Type) where
  cells : List (Cell Code Output L)
  store : Store L V
```

Forgetting the instrumentation recovers a plain standard state:

```lean
def Notebook.erase (nb : Notebook …) : StdState … :=
  ⟨nb.cells.map fun c => (c.code, c.out), nb.store⟩
```

## 2.2 Reading the tables

These helpers phrase the paper's set notation (`ℓ ∈ R_i`,
`ℓ ∈ ⋃ W_{i+1..n}`, …) as membership queries on the cell list. Each looks
up a position with `cells[·]?`, so out-of-range positions are simply
false.

```lean
def ReadsAt  (cs) (i) (ℓ) : Prop := ∃ c, cs[i]? = some c ∧ c.reads ℓ   -- ℓ ∈ R_i
def WritesAt (cs) (i) (ℓ) : Prop := ∃ c, cs[i]? = some c ∧ c.writes ℓ  -- ℓ ∈ W_i
def IsClean  (cs) (i)     : Prop := ∃ c, cs[i]? = some c ∧ c.tag = .clean
def IsStale  (cs) (i)     : Prop := ∃ c, cs[i]? = some c ∧ c.tag = .stale

def WritesBelow (cs) (i) (ℓ) : Prop := ∃ j, i < j ∧ WritesAt cs j ℓ  -- ℓ ∈ ⋃ W_{i+1..n}
def WritesAbove (cs) (i) (ℓ) : Prop := ∃ j, j < i ∧ WritesAt cs j ℓ  -- ℓ ∈ ⋃ W_{1..i-1}
def ReadsAbove  (cs) (i) (ℓ) : Prop := ∃ j, j < i ∧ ReadsAt  cs j ℓ  -- ℓ ∈ ⋃ R_{1..i-1}
```

## 2.3 Rerun consistency

FlowBook allows a cell's run only if its observed reads and writes are
consistent with what that cell would have done *in place* during a clean
top-to-bottom run. This is captured by four conditions, bundled as a
structure. Reading each field: cell `i` may not read a location it also
writes; everything it reads must have been written by some cell **above**
it; it may not read a location written by a cell **below** it; and it may
not overwrite a location that some cell **above** it read.

```lean
structure RerunConsistent (cs) (i : Nat) : Prop where
  -- R_i ∩ W_i = ∅
  noReadAndWrite    : ∀ ℓ, ReadsAt cs i ℓ → ¬ WritesAt cs i ℓ
  -- R_i ⊆ ⋃ W_{1..i-1}
  writeBeforeRead   : ∀ ℓ, ReadsAt cs i ℓ → WritesAbove cs i ℓ
  -- R_i ∩ ⋃ W_{i+1..n} = ∅
  noReadBeforeWrite : ∀ ℓ, ReadsAt cs i ℓ → ¬ WritesBelow cs i ℓ
  -- W_i ∩ ⋃ R_{1..i-1} = ∅
  noWriteAfterRead  : ∀ ℓ, WritesAt cs i ℓ → ¬ ReadsAbove cs i ℓ
```

## 2.4 Staleness propagation

After a cell runs (or is deleted), FlowBook marks other cells stale if
their recorded output might no longer be valid. Two situations arise.
Throughout, `w` is the cell's **new** write set (for deletion, `w` is
empty), while the tables in `cs` hold the **old** reads and writes.

**Forward staleness.** A cell `j` *below* `i` is invalidated if it
reads or writes any location that `i` writes now (`w`) or wrote before
(`W_i`):

```lean
-- ForwardStale(R, W, W', i, j) ≜ j > i ∧ (W_i ∪ W'_i) ∩ (R_j ∪ W_j) ≠ ∅
def FwdStale (cs) (i) (w) (j) : Prop :=
  i < j ∧ ∃ ℓ, (WritesAt cs i ℓ ∨ w ℓ) ∧ (ReadsAt cs j ℓ ∨ WritesAt cs j ℓ)
```

**Backward staleness.** Suppose cell `i` *stops* writing a location `ℓ`
it used to write (`ℓ ∈ W_i \ w`). Cells below `i` that read `ℓ` were
depending on `i` to supply it; with that write gone, `ℓ` must instead be
restored by the **nearest cell above `i` that writes `ℓ`**. That cell —
the *last writer* of `ℓ` above `i` — is marked stale so that re-running
it restores `ℓ`. "Last writer" is `j < i` writing `ℓ` with no other
writer of `ℓ` strictly between `j` and `i`:

```lean
-- LastWriter(W, i, ℓ) = max { k < i ∣ ℓ ∈ W_k }
def IsLastWriter (cs) (i) (ℓ) (j) : Prop :=
  j < i ∧ WritesAt cs j ℓ ∧ ∀ k, j < k → k < i → ¬ WritesAt cs k ℓ

-- BackwardStale(W, W', i, j) ≜ j < i ∧ j = LastWriter(W, i, ℓ)  for some  ℓ ∈ W_i \ W'_i
def BwdStale (cs) (i) (w) (j) : Prop :=
  ∃ ℓ, WritesAt cs i ℓ ∧ ¬ w ℓ ∧ IsLastWriter cs i ℓ j
```

A cell is **marked** by a run/delete of `i` if it is forward- or
backward-stale:

```lean
def Marked (cs) (i) (w) (j) : Prop := FwdStale cs i w j ∨ BwdStale cs i w j
```

## 2.5 The tag update, and why it is a *specification*

When cell `i` runs, every other cell `j` keeps its code, output, and
read/write sets, but its tag becomes `stale` exactly when it is `Marked`
or was already stale. Because `Marked` quantifies over locations and
positions and evaluation is a black box, this update is not computable in
general, so we describe it as a **relation** `RetagSpec` (a specification
the new table must satisfy) rather than a function:

```lean
def RetagSpec (cs cs' : List (Cell …)) (i : Nat) (w : L → Prop) : Prop :=
  ∀ j, j ≠ i → ∀ c, cs[j]? = some c →
    ∃ t, cs'[j]? = some { c with tag := t } ∧
      (t = Tag.stale ↔ (Marked cs i w j ∨ c.tag = Tag.stale))
```

The specification is always satisfiable — the rules below never get
stuck for lack of a valid retagging (proof omitted):

```lean
theorem retag_exists (cs) (i) (w) (ci') :
    ∃ cs', cs'.length = cs.length ∧
           (i < cs.length → cs'[i]? = some ci') ∧
           RetagSpec cs cs' i w
```

## 2.6 The instrumented semantics

The instrumented step relation `S · I ⟹op S' · I'` refines the standard
semantics. The most important rule is `[Inst-Run]`: it evaluates cell `i`
(getting output `o`, store `σ'`, reads `r`, writes `w`); it **requires**
the updated table to be rerun consistent for `i` (this is how illegal
runs are rejected — the rule simply does not apply); it marks `i` clean
with its new sets; and it retags every other cell per `RetagSpec`.
`[Inst-Delete]` uses the same retagging with empty new write set, then
removes the cell. `[Inst-Insert]` adds a fresh stale cell whose read and
write sets are empty (`fun _ => False`). `[Inst-Move]` is a delete
followed by an insert.

```lean
inductive InstStep : Notebook … → Op Code → Notebook … → Prop where
  -- [Inst-Run]
  | run (hcell : nb.cells[i]? = some ci)
        (heval : Eval ci.code nb.store o σ' r w)
        (hrc : RerunConsistent
                 (nb.cells.set i { ci with out := some o, tag := .clean,
                                           reads := r, writes := w }) i)
        (hlen : cs'.length = nb.cells.length)
        (hat  : cs'[i]? = some { ci with out := some o, tag := .clean,
                                         reads := r, writes := w })
        (hretag : RetagSpec nb.cells cs' i w) :
      InstStep nb (.run i) ⟨cs', σ'⟩

  -- [Inst-Edit]: mark the edited cell stale (its reads/writes are kept until re-run)
  | edit (hcell : nb.cells[i]? = some ci) :
      InstStep nb (.edit i c)
        ⟨nb.cells.set i { ci with code := c, tag := .stale }, nb.store⟩

  -- [Inst-Insert]: a fresh stale cell with empty read/write sets
  | insert (hle : i ≤ nb.cells.length) :
      InstStep nb (.insert i c)
        ⟨nb.cells.insertIdx i ⟨c, none, .stale, fun _ => False, fun _ => False⟩, nb.store⟩

  -- [Inst-Delete]: retag using the deleted cell's writes (new write set ∅), then erase
  | delete (hcell : nb.cells[i]? = some ci)
           (hlen : cs''.length = nb.cells.length)
           (hretag : RetagSpec nb.cells cs'' i (fun _ => False)) :
      InstStep nb (.delete i) ⟨cs''.eraseIdx i, nb.store⟩

  -- [Inst-Move-Down] (s < d) / [Inst-Move-Up] (d < s): delete then insert
  | moveDown (hlt : s < d) (hcell : nb.cells[s]? = some cs_)
             (h1 : InstStep nb (.delete s) nb'')
             (h2 : InstStep nb'' (.insert (d - 1) cs_.code) nb') :
      InstStep nb (.move s d) nb'
  | moveUp (hlt : d < s) (hcell : nb.cells[s]? = some cs_)
           (h1 : InstStep nb (.delete s) nb'')
           (h2 : InstStep nb'' (.insert d cs_.code) nb') :
      InstStep nb (.move s d) nb'
```

## 2.7 Well-formedness: the analysis invariant

FlowBook maintains an invariant that every **clean** cell is trustworthy.
Concretely, a clean cell `i` must have a *witness*: it can be re-executed
from the current store to reproduce its recorded output and its recorded
read/write sets, changing the store only at locations that later cells
overwrite anyway.

```lean
def Witnessed (cs) (σ) (i) : Prop :=
  ∃ c, cs[i]? = some c ∧
    ∃ o σ', c.out = some o ∧
      Eval c.code σ o σ' c.reads c.writes ∧      -- re-runs to the recorded output/reads/writes
      AgreeExcept σ σ' (WritesBelow cs i)         -- and only disturbs locations written below i
```

A notebook is **well-formed** when every clean cell is both witnessed and
rerun consistent:

```lean
def WellFormed (nb : Notebook …) : Prop :=
  ∀ i, IsClean nb.cells i → Witnessed nb.cells nb.store i ∧ RerunConsistent nb.cells i
```

The invariant holds at the start because a notebook begins with **every
cell stale**, so the "for every clean cell …" condition is vacuous:

```lean
theorem wellFormed_initial :
    (∀ i c, nb.cells[i]? = some c → c.tag = .stale) → WellFormed nb
```

---

# Part 3 — The three correctness theorems

## 3.1 Preservation — every operation keeps the invariant
### (`FlowBook/Preservation.lean`)

> **Theorem (Preservation).** If `S · I` is well-formed and the user
> performs any operation, `S · I ⟹op S' · I'`, then `S' · I'` is
> well-formed.

```lean
theorem preservation (hwf : WellFormed nb) (hstep : InstStep nb op nb') :
    WellFormed nb'
```

This is the workhorse: it says the rerun-consistency-and-staleness
machinery really does maintain the "every clean cell is trustworthy"
invariant, no matter what the user does. The proof is by cases on the
operation, matching the paper's appendix (the `[Inst-Run]` case splits
on whether a clean cell sits before, at, or after the executed cell).

## 3.2 Output Consistency — an all-clean notebook is reproducible
### (`FlowBook/OutputConsistency.lean`)

`AllClean` says every cell is tagged clean:

```lean
def AllClean (cs) : Prop := ∀ i c, cs[i]? = some c → c.tag = Tag.clean
```

> **Theorem (Output Consistency / Reproducibility).** A well-formed
> notebook in which *every* cell is clean is reproducible: its recorded
> outputs match some top-to-bottom execution from the empty store.

```lean
theorem output_consistency (hwf : WellFormed nb) (hclean : AllClean nb.cells) :
    Reproducible nb.erase
```

This is FlowBook's headline promise: **"all clean" implies
"reproducible."** The proof rebuilds a top-to-bottom run cell by cell,
using each clean cell's witness (via `locality`) to show it produces the
same output it recorded.

## 3.3 Progress — re-running stale cells terminates
### (`FlowBook/Progress.lean`)

Preservation and Output Consistency say the invariant is kept and that
all-clean states are good — but not that an all-clean state is
*reachable*. Progress closes the loop, for the `RunToClean` algorithm:

```
E := ∅                                   -- cells executed so far
while some cell is stale:
    i := the first stale cell
    if i is stuck:
        report error
    else:
        Run i with [Inst-Run]
        if i ∈ E and the run marked a cell before i stale:
            report potential non-termination
        E := E ∪ {i}
```

Every execution that reports no potential non-termination terminates,
either all-clean (hence reproducible) or at a report.

The check is necessary: cell evaluation is a relation, and rerunning
stale cells with no check need not terminate.  Two cells with empty
read sets that each write `{a}` or `{b}`, choosing differently on
successive runs, mark each other stale forever — each run of the later
cell that drops a location backward-marks the earlier cell via
`LastWriter`, and each run of the earlier cell forward-marks the
later.  The check forbids exactly the event that repeats: staleness
moves toward earlier cells only via `BackwardStale` (a run dropping a
write owned by an earlier cell), so requiring reruns to mark nothing
before themselves *is* the termination invariant, checked directly —
no footprint comparison is needed.

The first stale cell is the earliest stale position:

```lean
def FirstStale (cs) (i) : Prop := IsStale cs i ∧ ∀ j, j < i → ¬ IsStale cs j
```

Any notebook is either all-clean or has a first stale cell:

```lean
theorem allClean_or_firstStale (cs) : AllClean cs ∨ ∃ i, FirstStale cs i
```

Executions that report nothing are modeled by `RunToClean F nb nb' F'`,
where `F` is the **complement of the algorithm's `E`** — the positions
not yet executed.  Initially `F` is all positions (`E = ∅`).  Each
step runs the first stale cell: a first execution (`first`) is any
successful run; a repeated execution (`rerun`) must leave every
position `≤ i` clean — the algorithm's check, stated directly on the
successor state:

```lean
inductive RunToClean : List Nat → Notebook … → Notebook … → List Nat → Prop where
  | refl : RunToClean F nb nb F                        -- done
  | first (hfs : FirstStale nb.cells i) (hmem : i ∈ F) -- first execution of cell i
          (hstep : InstStep nb (.run i) nb₁)
          (hrest : RunToClean (F.erase i) nb₁ nb' F') : RunToClean F nb nb' F'
  | rerun (hfs : FirstStale nb.cells i) (hmem : i ∉ F) -- a rerun marks nothing before i
          (hstep : InstStep nb (.run i) nb₁)
          (hclean : ∀ j, j ≤ i → IsClean nb₁.cells j)
          (hrest : RunToClean F nb₁ nb' F') : RunToClean F nb nb' F'
```

The algorithm cannot continue without a report when the first stale
cell admits no run of the required kind:

```lean
def RunToCleanStuck (nb) (F) : Prop :=
  ∃ i, FirstStale nb.cells i ∧
    ((i ∈ F ∧ ∀ nb', ¬ InstStep nb (.run i) nb') ∨    -- no run at all: report error
     (i ∉ F ∧ ∀ nb', InstStep nb (.run i) nb' →      -- every run marks a cell at or
        ∃ j, j ≤ i ∧ ¬ IsClean nb'.cells j))          --   before i: report potential
                                                      --   non-termination
```

> **Theorem (Progress).** From any well-formed state, every report-free
> execution of `RunToClean` can be extended to reach a state that is
> well-formed and either all-clean or stuck.

```lean
theorem progress (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', RunToClean F nb nb' F' ∧ WellFormed nb' ∧
      (AllClean nb'.cells ∨ RunToCleanStuck nb' F')

theorem progress_init (hwf : WellFormed nb) :   -- the algorithm's E = ∅
    ∃ nb' F', RunToClean (List.range nb.cells.length) nb nb' F' ∧ WellFormed nb' ∧
      (AllClean nb'.cells ∨ RunToCleanStuck nb' F')
```

Combined with Output Consistency, termination lands in a *reproducible*
notebook or at a report:

```lean
theorem progress_reproducible (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', RunToClean F nb nb' F' ∧ (Reproducible nb'.erase ∨ RunToCleanStuck nb' F')
```

Termination follows the lexicographic measure `(|F|, n − first-stale
position)`: a first execution removes `i` from `F`; a rerun keeps the
prefix clean by its side condition, so the run of clean cells at the
top strictly grows.  The measure yields the paper's bound — at most
`n` first executions and at most `n` reruns between consecutive first
executions, i.e. at most `n(n+2)` runs in total.  The proof's key
supporting lemmas (behind the paper's Stability lemma) show the rerun
condition is satisfiable exactly where the paper's proof needs it:
after running the first stale cell, every cell `j < i` — in particular
every backward-marked one — *can* re-run reproducing its recorded
output, read set, and write set (a `MatchingRunAt`; the run's
`noWriteAfterRead` check keeps its new writes off those cells' read
sets, so `locality` reproduces their recorded behavior), and such a
run marks nothing before itself, so it satisfies the rerun condition
and never triggers the report:

```lean
theorem canRerun_after_run
    (hwf : WellFormed nb) (hfs : FirstStale nb.cells i)
    (hstep : InstStep nb (.run i) nb') :
    ∀ j, j < i → CanRerunAt nb' j

theorem cleanPrefix_run_exists
    (hfs : FirstStale nb.cells i) (h : CanRerunAt nb i) :
    ∃ nb', InstStep nb (.run i) nb' ∧ ∀ j, j ≤ i → IsClean nb'.cells j
```

---

# Part 4 — The analysis refines the standard semantics
### (`FlowBook/Erasure.lean`)

The instrumented rules were built by adding bookkeeping on top of the
standard rules. This theorem confirms the two stay in sync: erasing the
instrumentation from any instrumented step yields a valid standard step.
So FlowBook only *restricts* which ordinary behaviors are allowed — it
never invents new ones.

```lean
theorem instStep_erase (h : InstStep nb op nb') : StdStep nb.erase op nb'.erase
```

---

# Part 5 — The axioms are consistent
### (`FlowBook/Examples.lean`)

Everything above assumed an abstract `CellEval` with its `frame` and
`locality` axioms. To show these assumptions are not vacuous — that a
real runtime can satisfy them — a tiny concrete language is exhibited and
proved to be a `CellEval`.

The language has two commands: assign a constant to a location, or copy
one location to another.

```lean
inductive Cmd (L V : Type) where
  | assign (dst : L) (v : V)  -- dst := v
  | copy (dst src : L)        -- dst := src
```

Its store update, and its evaluation relation (an assignment reads
nothing and writes `dst`; a copy reads `src` and writes `dst`; outputs
are trivial, of type `Unit`):

```lean
def update [DecidableEq L] (σ : Store L V) (ℓ : L) (v : Option V) : Store L V :=
  fun ℓ' => if ℓ' = ℓ then v else σ ℓ'

def CmdEval [DecidableEq L] (c : Cmd L V) (σ : Store L V) (_ : Unit)
    (σ' : Store L V) (r w : L → Prop) : Prop :=
  match c with
  | .assign dst v =>
      σ' = update σ dst (some v) ∧ (∀ ℓ, ¬ r ℓ)        ∧ (∀ ℓ, w ℓ ↔ ℓ = dst)
  | .copy dst src =>
      σ' = update σ dst (σ src)  ∧ (∀ ℓ, r ℓ ↔ ℓ = src) ∧ (∀ ℓ, w ℓ ↔ ℓ = dst)
```

This is registered as a `CellEval` instance, with `frame` and `locality`
proved (proofs omitted):

```lean
instance instCellEval [DecidableEq L] : CellEval (Cmd L V) Unit L V where
  Eval := CmdEval
  frame := …
  locality := …
```

Instantiating the paper's location type and this language gives a
notebook type to which all three theorems apply directly:

```lean
abbrev PaperLoc     := Loc String Nat String
abbrev MiniNotebook := Notebook (Cmd PaperLoc Nat) Unit PaperLoc Nat
```

---

## Summary of the correspondence

| Concept | Lean | File |
|---|---|---|
| location `ℓ ::= x ∣ d.c` | `Loc` | Semantics |
| store, empty store, "agree except on X" | `Store`, `Store.empty`, `AgreeExcept` | Semantics |
| cell evaluation `c ; Σ ⇓ o · Σ' · r · w` | `CellEval.Eval` + `frame`, `locality` | Semantics |
| user operations | `Op` | Semantics |
| standard state and semantics | `StdState`, `StdStep` | Semantics |
| top-to-bottom execution | `Runs` | Semantics |
| reproducible / output-consistent | `Reproducible` | Semantics |
| status tag; instrumented cell/notebook | `Tag`, `Cell`, `Notebook` | Analysis |
| rerun consistency (4 conditions) | `RerunConsistent` | Analysis |
| forward / backward staleness | `FwdStale`, `IsLastWriter`, `BwdStale`, `Marked` | Analysis |
| the tag update / instrumented semantics | `RetagSpec`, `InstStep` | Analysis |
| the analysis invariant | `Witnessed`, `WellFormed` | Analysis |
| notebooks start well-formed | `wellFormed_initial` | Analysis |
| **Preservation** | `preservation` | Preservation |
| **Output Consistency** | `output_consistency` | OutputConsistency |
| **Progress** | `progress`, `progress_init`, `progress_reproducible` | Progress |
| the `RunToClean` algorithm and its check | `RunToClean`, `RunToCleanStuck`, `MatchingRunAt`, `cleanPrefix_run_exists` | Progress |
| analysis refines standard semantics | `instStep_erase` | Erasure |
| the axioms are satisfiable | `Cmd`, `CmdEval`, `instCellEval` | Examples |

Every statement above is proved in Lean; see the `.lean` files for the
proofs, and `FlowBook/Exec.lean` for an executable, separately-verified
version of the analysis that runs the paper's litmus tests.
