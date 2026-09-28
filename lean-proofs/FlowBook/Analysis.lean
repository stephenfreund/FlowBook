/-
# Dynamic Analysis for Reproducibility: the Instrumented Semantics

This file formalizes Section "Dynamic Analysis for Reproducibility":

* the instrumentation state `I = (T, R, W)` — status tags, read sets,
  and write sets — attached to each cell;
* the rerun-consistency predicates `NoReadAndWrite`, `WriteBeforeRead`,
  `NoReadBeforeWrite`, `NoWriteAfterRead`
  (Definition 2.1 of the supplement);
* the staleness predicates `ForwardStale`, `BackwardStale`, and
  `LastWriter`;
* the instrumented semantics `S · I ⟹op S' · I'` of Figure 9;
* the well-formedness invariant (Definition 2.2 of the supplement).

Representation notes:

* The paper's instrumented state `S · I = (C, O, Σ) · (T, R, W)` is six
  parallel sequences plus a store.  We fuse the five per-cell sequences
  into a single list of `Cell` records, which keeps them synchronized by
  construction.  `Notebook.erase` recovers the standard state `(C, O, Σ)`.
* Read and write sets are predicates `L → Prop`.
* The tag updates of `[Inst-Run]` and `[Inst-Delete]` are specified
  relationally (pointwise over positions) rather than as a function,
  because the staleness predicates are not decidable for black-box
  evaluation; `retag_exists` below shows (classically) that the
  specification is always satisfiable, so the rules never fail for want
  of a retagging.
-/
import FlowBook.Semantics

namespace FlowBook

/-- Cell status: `clean` means the recorded output is valid for the
current store; `stale` means the cell must be re-executed. -/
inductive Tag where
  | clean
  | stale
deriving DecidableEq

theorem Tag.clean_or_stale : ∀ t : Tag, t = .clean ∨ t = .stale
  | .clean => .inl rfl
  | .stale => .inr rfl

variable {Code Output L V : Type}

/-- One notebook cell together with its instrumentation: source code
`code`, most recent output `out` (`none` = never executed), status tag
`tag`, and the recorded read and write sets `reads`/`writes` from its
most recent execution. -/
structure Cell (Code Output L : Type) where
  code : Code
  out : Option Output
  tag : Tag
  reads : L → Prop
  writes : L → Prop

/-- An instrumented notebook state `S · I`. -/
structure Notebook (Code Output L V : Type) where
  cells : List (Cell Code Output L)
  store : Store L V

/-- Erasing the instrumentation recovers the standard state `(C, O, Σ)`. -/
def Notebook.erase (nb : Notebook Code Output L V) : StdState Code Output L V :=
  ⟨nb.cells.map fun c => (c.code, c.out), nb.store⟩

theorem lt_length_of_getElem?_eq_some {cs : List (Cell Code Output L)} {j : Nat}
    {c : Cell Code Output L} (h : cs[j]? = some c) : j < cs.length := by
  rcases List.getElem?_eq_some_iff.mp h with ⟨hlt, _⟩
  exact hlt

theorem exists_getElem?_eq_some {cs : List (Cell Code Output L)} {j : Nat}
    (h : j < cs.length) : ∃ c, cs[j]? = some c :=
  ⟨cs[j], List.getElem?_eq_getElem h⟩

/-! ## Positional accessors on the cell list -/

section Tables
variable (cs : List (Cell Code Output L))

/-- `ℓ ∈ R_i`. -/
def ReadsAt (i : Nat) (ℓ : L) : Prop := ∃ c, cs[i]? = some c ∧ c.reads ℓ

/-- `ℓ ∈ W_i`. -/
def WritesAt (i : Nat) (ℓ : L) : Prop := ∃ c, cs[i]? = some c ∧ c.writes ℓ

/-- `T_i = clean`. -/
def IsClean (i : Nat) : Prop := ∃ c, cs[i]? = some c ∧ c.tag = .clean

/-- `T_i = stale`. -/
def IsStale (i : Nat) : Prop := ∃ c, cs[i]? = some c ∧ c.tag = .stale

/-- `ℓ ∈ ⋃ W_{i+1..n}`: some cell strictly below `i` writes `ℓ`. -/
def WritesBelow (i : Nat) (ℓ : L) : Prop := ∃ j, i < j ∧ WritesAt cs j ℓ

/-- `ℓ ∈ ⋃ W_{1..i-1}`: some cell strictly above `i` writes `ℓ`. -/
def WritesAbove (i : Nat) (ℓ : L) : Prop := ∃ j, j < i ∧ WritesAt cs j ℓ

/-- `ℓ ∈ ⋃ R_{1..i-1}`: some cell strictly above `i` reads `ℓ`. -/
def ReadsAbove (i : Nat) (ℓ : L) : Prop := ∃ j, j < i ∧ ReadsAt cs j ℓ

/-! ## Rerun consistency (Definition 2.1 of the supplement) -/

/-- Definition (Rerun Consistent Accesses).  Instrumentation state `R`
and `W` are rerun consistent for cell `i` if:

* `noReadAndWrite`  — `R_i ∩ W_i = ∅`: cell `i` does not read and write
  the same location;
* `writeBeforeRead` — `R_i ⊆ ⋃ W_{1..i-1}`: every location read by `i`
  was written by a cell above `i`;
* `noReadBeforeWrite` — `R_i ∩ (⋃ W_{i+1..n}) = ∅`: cell `i` does not
  read locations written by cells below `i`;
* `noWriteAfterRead` — `W_i ∩ (⋃ R_{1..i-1}) = ∅`: cell `i` does not
  overwrite locations read by cells above `i`. -/
structure RerunConsistent (i : Nat) : Prop where
  noReadAndWrite : ∀ ℓ, ReadsAt cs i ℓ → ¬ WritesAt cs i ℓ
  writeBeforeRead : ∀ ℓ, ReadsAt cs i ℓ → WritesAbove cs i ℓ
  noReadBeforeWrite : ∀ ℓ, ReadsAt cs i ℓ → ¬ WritesBelow cs i ℓ
  noWriteAfterRead : ∀ ℓ, WritesAt cs i ℓ → ¬ ReadsAbove cs i ℓ

/-! ## Staleness propagation

After running (or deleting) cell `i`, the analysis marks stale every
cell whose recorded execution may no longer be rerun consistent.  In
`ForwardStale`/`BackwardStale` below, the argument `w` is the *new*
write set `W'_i` of cell `i` (`w = ∅` for `[Inst-Delete]`, matching the
figure's `W'' = W[i := ∅]`), while the tables in `cs` are the *old*
`R`, `W`. -/

/-- `ForwardStale(R, W, W', i, j) ≜ j > i ∧ (W_i ∪ W'_i) ∩ (R_j ∪ W_j) ≠ ∅`:
cell `j` below `i` reads or writes a location that cell `i` writes now
(`w`) or wrote before (`W_i`). -/
def FwdStale (i : Nat) (w : L → Prop) (j : Nat) : Prop :=
  i < j ∧ ∃ ℓ, (WritesAt cs i ℓ ∨ w ℓ) ∧ (ReadsAt cs j ℓ ∨ WritesAt cs j ℓ)

/-- `LastWriter(W, i, ℓ) = max { k < i | ℓ ∈ W_k }`, as a relation:
`j` is the nearest writer of `ℓ` strictly above `i`. -/
def IsLastWriter (i : Nat) (ℓ : L) (j : Nat) : Prop :=
  j < i ∧ WritesAt cs j ℓ ∧ ∀ k, j < k → k < i → ¬ WritesAt cs k ℓ

/-- `BackwardStale(W, W', i, j) ≜ j < i ∧ j = LastWriter(W, i, ℓ)` for
some `ℓ ∈ W_i \ W'_i`: cell `i` stopped writing `ℓ`, so the nearest
writer of `ℓ` above `i` must re-execute to restore a store consistent
with top-to-bottom execution. -/
def BwdStale (i : Nat) (w : L → Prop) (j : Nat) : Prop :=
  ∃ ℓ, WritesAt cs i ℓ ∧ ¬ w ℓ ∧ IsLastWriter cs i ℓ j

/-- A cell is marked stale by running/deleting cell `i` with new write
set `w` iff it is forward or backward stale. -/
def Marked (i : Nat) (w : L → Prop) (j : Nat) : Prop :=
  FwdStale cs i w j ∨ BwdStale cs i w j

end Tables

/-- Pointwise specification of the tag update `T'` performed by
`[Inst-Run]` and `[Inst-Delete]` at positions `j ≠ i`: every cell keeps
its code, output, and read/write sets, and its new tag is `stale` iff
it is `Marked` (forward or backward stale) or was already stale. -/
def RetagSpec (cs cs' : List (Cell Code Output L)) (i : Nat) (w : L → Prop) : Prop :=
  ∀ j, j ≠ i → ∀ c, cs[j]? = some c →
    ∃ t, cs'[j]? = some { c with tag := t } ∧
      (t = Tag.stale ↔ (Marked cs i w j ∨ c.tag = Tag.stale))

/-! ## The instrumented semantics (Figure 9) -/

open CellEval in
/-- The instrumented semantics `S · I ⟹op S' · I'` of
Figure 9.

* `[Inst-Run]` evaluates cell `i` via the instrumented judgment,
  requires the updated read/write tables to be rerun consistent for
  `i` (Definition 2.1 of the supplement), marks cell `i`
  clean, and marks forward/backward-stale cells stale.
* `[Inst-Edit]` replaces the code and marks the cell stale, leaving
  `R`, `W`, and the recorded output unchanged.
* `[Inst-Insert]` inserts a fresh stale cell with empty read and write
  sets.
* `[Inst-Delete]` applies the same staleness marking as `[Inst-Run]`
  with new write set `∅`, then removes the cell.
* `[Inst-Move-Down]`/`[Inst-Move-Up]` compose a delete and an insert,
  exactly as in the figure. -/
inductive InstStep [CellEval Code Output L V] :
    Notebook Code Output L V → Op Code → Notebook Code Output L V → Prop where
  /-- `[Inst-Run]`. -/
  | run {nb : Notebook Code Output L V} {i ci o σ' r w cs'}
      (hcell : nb.cells[i]? = some ci)
      (heval : Eval ci.code nb.store o σ' r w)
      (hrc : RerunConsistent
        (nb.cells.set i { ci with out := some o, tag := .clean, reads := r, writes := w }) i)
      (hlen : cs'.length = nb.cells.length)
      (hat : cs'[i]? = some { ci with out := some o, tag := .clean, reads := r, writes := w })
      (hretag : RetagSpec nb.cells cs' i w) :
      InstStep nb (.run i) ⟨cs', σ'⟩
  /-- `[Inst-Edit]`. -/
  | edit {nb : Notebook Code Output L V} {i ci c}
      (hcell : nb.cells[i]? = some ci) :
      InstStep nb (.edit i c)
        ⟨nb.cells.set i { ci with code := c, tag := .stale }, nb.store⟩
  /-- `[Inst-Insert]`. -/
  | insert {nb : Notebook Code Output L V} {i c}
      (hle : i ≤ nb.cells.length) :
      InstStep nb (.insert i c)
        ⟨nb.cells.insertIdx i ⟨c, none, .stale, fun _ => False, fun _ => False⟩, nb.store⟩
  /-- `[Inst-Delete]`: retag with new write set `∅`, then erase. -/
  | delete {nb : Notebook Code Output L V} {i ci cs''}
      (hcell : nb.cells[i]? = some ci)
      (hlen : cs''.length = nb.cells.length)
      (hretag : RetagSpec nb.cells cs'' i (fun _ => False)) :
      InstStep nb (.delete i) ⟨cs''.eraseIdx i, nb.store⟩
  /-- `[Inst-Move-Down]` (`s < d`). -/
  | moveDown {nb nb'' nb' : Notebook Code Output L V} {s d cs_}
      (hlt : s < d)
      (hcell : nb.cells[s]? = some cs_)
      (h1 : InstStep nb (.delete s) nb'')
      (h2 : InstStep nb'' (.insert (d - 1) cs_.code) nb') :
      InstStep nb (.move s d) nb'
  /-- `[Inst-Move-Up]` (`d < s`). -/
  | moveUp {nb nb'' nb' : Notebook Code Output L V} {s d cs_}
      (hlt : d < s)
      (hcell : nb.cells[s]? = some cs_)
      (h1 : InstStep nb (.delete s) nb'')
      (h2 : InstStep nb'' (.insert d cs_.code) nb') :
      InstStep nb (.move s d) nb'

open Classical in
/-- The pointwise retagging function realizing `RetagSpec` (classical,
since the staleness predicates are not decidable for black-box
evaluation). -/
private noncomputable def retagFun (cs : List (Cell Code Output L)) (i : Nat)
    (w : L → Prop) (ci' : Cell Code Output L) (j : Nat) (c : Cell Code Output L) :
    Cell Code Output L :=
  if j = i then ci'
  else if Marked cs i w j ∨ c.tag = Tag.stale then { c with tag := .stale }
  else { c with tag := .clean }

/-- The retag specification is always satisfiable: the rules `[Inst-Run]`
and `[Inst-Delete]` never fail for want of a retagging. -/
theorem retag_exists (cs : List (Cell Code Output L)) (i : Nat) (w : L → Prop)
    (ci' : Cell Code Output L) :
    ∃ cs' : List (Cell Code Output L),
      cs'.length = cs.length ∧
      (i < cs.length → cs'[i]? = some ci') ∧
      RetagSpec cs cs' i w := by
  refine ⟨cs.mapFinIdx fun j c _ => retagFun cs i w ci' j c, by simp, ?_, ?_⟩
  · intro hi
    rw [List.getElem?_eq_getElem (by simpa using hi)]
    simp [retagFun]
  · intro j hj c hc
    have hjlt : j < cs.length := by
      rcases Nat.lt_or_ge j cs.length with h | h
      · exact h
      · rw [List.getElem?_eq_none h] at hc; cases hc
    have hcj : cs[j] = c := by
      rw [List.getElem?_eq_getElem hjlt] at hc
      exact Option.some.inj hc
    rw [List.getElem?_eq_getElem (l := cs.mapFinIdx fun j c _ => retagFun cs i w ci' j c)
      (by simpa using hjlt)]
    by_cases hm : Marked cs i w j ∨ c.tag = Tag.stale
    · refine ⟨.stale, ?_, by simpa using hm⟩
      simp [retagFun, hj, hcj, hm]
    · refine ⟨.clean, ?_, by simp [hm]⟩
      simp [retagFun, hj, hcj, hm]

/-! ## Well-formedness (Definition 2.2 of the supplement) -/

section WellFormed
variable [CellEval Code Output L V]

open CellEval in
/-- Clauses (1) and (2) of Definition 2.2 of the supplement for cell `i`:
the cell can be re-executed from the current store `σ` to reproduce its
recorded output and read/write sets, changing the store only at
locations overwritten by cells below `i`. -/
def Witnessed (cs : List (Cell Code Output L)) (σ : Store L V) (i : Nat) : Prop :=
  ∃ c, cs[i]? = some c ∧
    ∃ o σ', c.out = some o ∧
      Eval c.code σ o σ' c.reads c.writes ∧
      AgreeExcept σ σ' (WritesBelow cs i)

/-- Definition (Well-Formed State).  An instrumented state
`S · I = (C, O, Σ) · (T, R, W)` is well-formed if every clean cell `i`

1. can be re-executed from the current store to produce its recorded
   output `O_i` and read and write sets `R_i` and `W_i`,
2. changing the store only at locations written by cells below `i`, and
3. `R` and `W` are rerun consistent for `i`. -/
def WellFormed (nb : Notebook Code Output L V) : Prop :=
  ∀ i, IsClean nb.cells i →
    Witnessed nb.cells nb.store i ∧ RerunConsistent nb.cells i

/-- "Initially, all cells are stale" — and any all-stale state is
well-formed (vacuously), so every notebook starts well-formed. -/
theorem wellFormed_initial {nb : Notebook Code Output L V}
    (h : ∀ i : Nat, ∀ c : Cell Code Output L, nb.cells[i]? = some c → c.tag = .stale) :
    WellFormed nb := by
  rintro i ⟨c, hc, htag⟩
  rw [h i c hc] at htag
  cases htag

end WellFormed

end FlowBook
