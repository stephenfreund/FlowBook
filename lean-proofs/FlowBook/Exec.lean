/-
# An Executable Reference Kernel

The rest of the development models cell evaluation as a *relation* and
read/write sets as *predicates* `L → Prop`, which is faithful to the
paper (non-deterministic black-box runtime) but not runnable.  This
file gives a concrete, **executable** instantiation:

* a small deterministic cell language `LCmd` with a functional
  evaluator `evalCmd` that returns the output, new store, and the read
  and write sets *as finite lists*;
* a `CellEval` instance whose evaluation relation is the graph of
  `evalCmd`, with `frame` and `locality` proved;
* an executable notebook representation `ENotebook` carrying list-backed
  read/write sets, reflected to the relational `Notebook` by `toNb`;
* executable operations (`runCell`, `editCell`, `insertCell`,
  `deleteCell`) returning `Except Violation ·`, where `runCell` reports
  exactly the rerun-consistency violation it detected — the messages of
  Figure 3;
* **soundness**: whenever `runCell` succeeds it produces a genuine
  `[Inst-Run]` step (`runCell_sound`), and likewise for the structural
  operations; combined with `preservation`/`output_consistency`, an
  accepted run from a well-formed state stays well-formed, and an
  all-clean reachable state is reproducible (`allClean_reproducible`);
* `#guard` checks replaying the litmus tests of Figures 3 and 4.

The executable staleness marking is computed by decidable Boolean
functions (`fwdStaleB`, `bwdStaleB`) that are proved to reflect the
relational `FwdStale`/`BwdStale` predicates, so the tags checked by the
`#guard`s are the same tags the soundness theorem certifies.
-/
import FlowBook.Preservation
import FlowBook.OutputConsistency

namespace FlowBook
namespace Exec

deriving instance Repr for Tag

/-! ## A concrete, deterministic cell language -/

/-- A miniature cell language rich enough to express every litmus test:
`const dst v` assigns a constant (writes `dst`); `copy dst src` copies
one location to another (reads `src`, writes `dst`); `use src` is a
read-only expression whose output is the value read (reads `src`); and
`incr dst` reads and writes the same location (the `x = x + 1` case). -/
inductive LCmd (L : Type) where
  | const (dst : L) (v : Nat)
  | copy (dst src : L)
  | use (src : L)
  | incr (dst : L)
deriving Repr, DecidableEq

/-- A cell's output value: `use` shows the number it read; statements
display `0`.  A `Cell`/`ECell` records `Option Out` (with `none` meaning
"never executed"). -/
abbrev Out : Type := Nat

variable {L : Type} [DecidableEq L]

/-- Pointwise store update. -/
def update (σ : Store L Nat) (ℓ : L) (v : Option Nat) : Store L Nat :=
  fun ℓ' => if ℓ' = ℓ then v else σ ℓ'

@[simp] theorem update_self (σ : Store L Nat) (ℓ : L) (v : Option Nat) :
    update σ ℓ v ℓ = v := by simp [update]

@[simp] theorem update_ne (σ : Store L Nat) {ℓ ℓ' : L} (h : ℓ' ≠ ℓ) (v : Option Nat) :
    update σ ℓ v ℓ' = σ ℓ' := by simp [update, h]

/-- The functional evaluator: returns `(output, new store, reads, writes)`. -/
def evalCmd (c : LCmd L) (σ : Store L Nat) : Out × Store L Nat × List L × List L :=
  match c with
  | .const dst v => (0, update σ dst (some v), [], [dst])
  | .copy dst src => (0, update σ dst (σ src), [src], [dst])
  | .use src => ((σ src).getD 0, σ, [src], [])
  | .incr dst => (0, update σ dst ((σ dst).map (· + 1)), [dst], [dst])

/-! ## The evaluation relation and the `CellEval` instance -/

/-- The evaluation relation is the graph of `evalCmd`, with read and
write sets reflected from the returned lists. -/
def LEval (c : LCmd L) (σ : Store L Nat) (o : Out) (σ' : Store L Nat)
    (r w : L → Prop) : Prop :=
  o = (evalCmd c σ).1 ∧ σ' = (evalCmd c σ).2.1 ∧
    (∀ ℓ, r ℓ ↔ ℓ ∈ (evalCmd c σ).2.2.1) ∧ (∀ ℓ, w ℓ ↔ ℓ ∈ (evalCmd c σ).2.2.2)

/-- `evalCmd` only changes the store at locations in its write list. -/
theorem evalCmd_frame (c : LCmd L) (σ : Store L Nat) :
    ∀ ℓ, ℓ ∉ (evalCmd c σ).2.2.2 → (evalCmd c σ).2.1 ℓ = σ ℓ := by
  intro ℓ hℓ
  cases c with
  | const dst v => simp only [evalCmd] at hℓ ⊢; exact update_ne σ (by simpa using hℓ) _
  | copy dst src => simp only [evalCmd] at hℓ ⊢; exact update_ne σ (by simpa using hℓ) _
  | use src => simp [evalCmd]
  | incr dst => simp only [evalCmd] at hℓ ⊢; exact update_ne σ (by simpa using hℓ) _

/-- `evalCmd`'s output and written values depend only on the read
locations. -/
theorem evalCmd_locality (c : LCmd L) (σ τ : Store L Nat)
    (hagree : ∀ ℓ, ℓ ∈ (evalCmd c σ).2.2.1 → τ ℓ = σ ℓ) :
    (evalCmd c τ).1 = (evalCmd c σ).1 ∧
    (evalCmd c τ).2.2.1 = (evalCmd c σ).2.2.1 ∧
    (evalCmd c τ).2.2.2 = (evalCmd c σ).2.2.2 ∧
    (∀ ℓ, ℓ ∈ (evalCmd c σ).2.2.2 → (evalCmd c τ).2.1 ℓ = (evalCmd c σ).2.1 ℓ) := by
  cases c with
  | const dst v =>
    refine ⟨rfl, rfl, rfl, ?_⟩
    intro ℓ hℓ; have : ℓ = dst := by simpa [evalCmd] using hℓ
    subst this; simp [evalCmd]
  | copy dst src =>
    have hsrc : τ src = σ src := hagree src (by simp [evalCmd])
    refine ⟨rfl, rfl, rfl, ?_⟩
    intro ℓ hℓ; have : ℓ = dst := by simpa [evalCmd] using hℓ
    subst this; simp [evalCmd, hsrc]
  | use src =>
    have hsrc : τ src = σ src := hagree src (by simp [evalCmd])
    refine ⟨by simp [evalCmd, hsrc], rfl, rfl, ?_⟩
    intro ℓ hℓ; simp [evalCmd] at hℓ
  | incr dst =>
    have hdst : τ dst = σ dst := hagree dst (by simp [evalCmd])
    refine ⟨rfl, rfl, rfl, ?_⟩
    intro ℓ hℓ; have : ℓ = dst := by simpa [evalCmd] using hℓ
    subst this; simp [evalCmd, hdst]

/-- The concrete language satisfies the black-box evaluation axioms. -/
instance instLCellEval : CellEval (LCmd L) Out L Nat where
  Eval := LEval
  frame := by
    rintro c σ o σ' r w ⟨ho, hσ', hr, hw⟩ ℓ hnw
    have : ℓ ∉ (evalCmd c σ).2.2.2 := fun h => hnw ((hw ℓ).mpr h)
    rw [hσ']; exact evalCmd_frame c σ ℓ this
  locality := by
    rintro c σ o σ' r w τ ⟨ho, hσ', hr, hw⟩ hag
    have hag' : ∀ ℓ, ℓ ∈ (evalCmd c σ).2.2.1 → τ ℓ = σ ℓ :=
      fun ℓ hℓ => hag ℓ ((hr ℓ).mpr hℓ)
    obtain ⟨hoτ, hrτ, hwτ, hvτ⟩ := evalCmd_locality c σ τ hag'
    refine ⟨(evalCmd c τ).2.1, ⟨ho.trans hoτ.symm, rfl, ?_, ?_⟩, ?_, ?_⟩
    · intro ℓ; rw [hrτ]; exact hr ℓ
    · intro ℓ; rw [hwτ]; exact hw ℓ
    · intro ℓ hwℓ
      have : ℓ ∈ (evalCmd c σ).2.2.2 := (hw ℓ).mp hwℓ
      rw [hσ']; exact hvτ ℓ this
    · intro ℓ hnw
      have : ℓ ∉ (evalCmd c τ).2.2.2 := fun h => hnw ((hw ℓ).mpr (hwτ ▸ h))
      exact evalCmd_frame c τ ℓ this

/-! ## Executable notebooks and reflection to the relational model -/

/-- An executable cell: like `Cell`, but read/write sets are finite
`List`s so membership is decidable and the cell is `#eval`-able. -/
structure ECell (L : Type) where
  code : LCmd L
  out : Option Out
  tag : Tag
  reads : List L
  writes : List L
deriving Repr

/-- An executable notebook. -/
structure ENotebook (L : Type) where
  cells : List (ECell L)
  store : Store L Nat

/-- Reflect an executable cell to a relational `Cell` by turning its
read/write lists into membership predicates. -/
def ECell.toCell (c : ECell L) : Cell (LCmd L) Out L :=
  { code := c.code, out := c.out, tag := c.tag,
    reads := fun ℓ => ℓ ∈ c.reads, writes := fun ℓ => ℓ ∈ c.writes }

/-- Reflect an executable notebook to the relational `Notebook`. -/
def ENotebook.toNb (nb : ENotebook L) : Notebook (LCmd L) Out L Nat :=
  { cells := nb.cells.map ECell.toCell, store := nb.store }

@[simp] theorem toNb_cells (nb : ENotebook L) :
    nb.toNb.cells = nb.cells.map ECell.toCell := rfl

@[simp] theorem toNb_store (nb : ENotebook L) : nb.toNb.store = nb.store := rfl

/-- The write list at index `k` (empty if out of range). -/
def wListAt (ecs : List (ECell L)) (k : Nat) : List L :=
  match ecs[k]? with | some c => c.writes | none => []

/-- The read list at index `k` (empty if out of range). -/
def rListAt (ecs : List (ECell L)) (k : Nat) : List L :=
  match ecs[k]? with | some c => c.reads | none => []

theorem writesAt_iff (ecs : List (ECell L)) (j : Nat) (ℓ : L) :
    WritesAt (ecs.map ECell.toCell) j ℓ ↔ ℓ ∈ wListAt ecs j := by
  unfold WritesAt wListAt
  rw [List.getElem?_map]
  cases h : ecs[j]? with
  | none => simp [ECell.toCell]
  | some ec => simp [ECell.toCell]

theorem readsAt_iff (ecs : List (ECell L)) (j : Nat) (ℓ : L) :
    ReadsAt (ecs.map ECell.toCell) j ℓ ↔ ℓ ∈ rListAt ecs j := by
  unfold ReadsAt rListAt
  rw [List.getElem?_map]
  cases h : ecs[j]? with
  | none => simp [ECell.toCell]
  | some ec => simp [ECell.toCell]

/-! ## Decidable rerun-consistency checks (Definition 2.1 of the supplement) -/

/-- `ℓ` is written by some cell strictly above `i`. -/
def writtenAboveB (ecs : List (ECell L)) (i : Nat) (ℓ : L) : Bool :=
  (List.range i).any (fun k => decide (ℓ ∈ wListAt ecs k))

/-- `ℓ` is written by some cell strictly below `i`. -/
def writtenBelowB (ecs : List (ECell L)) (i : Nat) (ℓ : L) : Bool :=
  (List.range ecs.length).any (fun k => decide (i < k) && decide (ℓ ∈ wListAt ecs k))

/-- `ℓ` is read by some cell strictly above `i`. -/
def readAboveB (ecs : List (ECell L)) (i : Nat) (ℓ : L) : Bool :=
  (List.range i).any (fun k => decide (ℓ ∈ rListAt ecs k))

theorem writtenAboveB_iff (ecs : List (ECell L)) (i : Nat) (ℓ : L) :
    writtenAboveB ecs i ℓ = true ↔ WritesAbove (ecs.map ECell.toCell) i ℓ := by
  unfold writtenAboveB WritesAbove
  simp only [List.any_eq_true, List.mem_range, decide_eq_true_eq]
  constructor
  · rintro ⟨k, hk, hw⟩; exact ⟨k, hk, (writesAt_iff ecs k ℓ).mpr hw⟩
  · rintro ⟨k, hk, hw⟩; exact ⟨k, hk, (writesAt_iff ecs k ℓ).mp hw⟩

theorem writtenBelowB_iff (ecs : List (ECell L)) (i : Nat) (ℓ : L) :
    writtenBelowB ecs i ℓ = true ↔ WritesBelow (ecs.map ECell.toCell) i ℓ := by
  unfold writtenBelowB WritesBelow
  simp only [List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨k, _, hk, hw⟩; exact ⟨k, hk, (writesAt_iff ecs k ℓ).mpr hw⟩
  · rintro ⟨k, hk, hw⟩
    have hklt : k < ecs.length := by
      rcases Nat.lt_or_ge k ecs.length with h | h
      · exact h
      · exfalso
        have hnone : ecs[k]? = none := List.getElem?_eq_none h
        rw [(writesAt_iff ecs k ℓ)] at hw
        simp [wListAt, hnone] at hw
    exact ⟨k, hklt, hk, (writesAt_iff ecs k ℓ).mp hw⟩

theorem readAboveB_iff (ecs : List (ECell L)) (i : Nat) (ℓ : L) :
    readAboveB ecs i ℓ = true ↔ ReadsAbove (ecs.map ECell.toCell) i ℓ := by
  unfold readAboveB ReadsAbove
  simp only [List.any_eq_true, List.mem_range, decide_eq_true_eq]
  constructor
  · rintro ⟨k, hk, hr⟩; exact ⟨k, hk, (readsAt_iff ecs k ℓ).mpr hr⟩
  · rintro ⟨k, hk, hr⟩; exact ⟨k, hk, (readsAt_iff ecs k ℓ).mp hr⟩

/-- Rerun-consistency violations, with the diagnostic messages of
Figure 3. -/
inductive Violation (L : Type) where
  | noSuchCell
  | noReadAndWrite (ℓ : L)     -- "reads and writes ℓ"
  | writeBeforeRead (ℓ : L)    -- "ℓ not written by a cell above"
  | noReadBeforeWrite (ℓ : L)  -- "reads ℓ written by a cell below"
  | noWriteAfterRead (ℓ : L)   -- "writes ℓ read by a cell above"
deriving Repr, DecidableEq

/-- Diagnose the first rerun-consistency violation of cell `i` in `ecs`,
or `none` if cell `i` is rerun consistent. -/
def diagnose (ecs : List (ECell L)) (i : Nat) : Option (Violation L) :=
  match ecs[i]? with
  | none => some .noSuchCell
  | some ci =>
    match ci.reads.find? (fun ℓ => decide (ℓ ∈ ci.writes)) with
    | some ℓ => some (.noReadAndWrite ℓ)
    | none =>
    match ci.reads.find? (fun ℓ => writtenBelowB ecs i ℓ) with
    | some ℓ => some (.noReadBeforeWrite ℓ)
    | none =>
    match ci.reads.find? (fun ℓ => !writtenAboveB ecs i ℓ) with
    | some ℓ => some (.writeBeforeRead ℓ)
    | none =>
    match ci.writes.find? (fun ℓ => readAboveB ecs i ℓ) with
    | some ℓ => some (.noWriteAfterRead ℓ)
    | none => none

/-- When `diagnose` reports no violation, cell `i` is genuinely rerun
consistent in the relational model. -/
theorem diagnose_none_sound {ecs : List (ECell L)} {i : Nat}
    (h : diagnose ecs i = none) : RerunConsistent (ecs.map ECell.toCell) i := by
  rw [diagnose] at h
  split at h
  · exact absurd h (by simp)
  · rename_i ci hci
    have hget : ∀ ℓ, ReadsAt (ecs.map ECell.toCell) i ℓ ↔ ℓ ∈ ci.reads := by
      intro ℓ; rw [readsAt_iff]; unfold rListAt; rw [hci]
    have hgetw : ∀ ℓ, WritesAt (ecs.map ECell.toCell) i ℓ ↔ ℓ ∈ ci.writes := by
      intro ℓ; rw [writesAt_iff]; unfold wListAt; rw [hci]
    split at h
    · exact absurd h (by simp)
    · rename_i h1
      split at h
      · exact absurd h (by simp)
      · rename_i h2
        split at h
        · exact absurd h (by simp)
        · rename_i h3
          split at h
          · exact absurd h (by simp)
          · rename_i h4
            have H1 := List.find?_eq_none.mp h1
            have H2 := List.find?_eq_none.mp h2
            have H3 := List.find?_eq_none.mp h3
            have H4 := List.find?_eq_none.mp h4
            refine ⟨?_, ?_, ?_, ?_⟩
            · intro ℓ hr hw
              have := H1 ℓ ((hget ℓ).mp hr)
              simp [(hgetw ℓ).mp hw] at this
            · intro ℓ hr
              have h3ℓ := H3 ℓ ((hget ℓ).mp hr)
              refine (writtenAboveB_iff ecs i ℓ).mp ?_
              cases hcv : writtenAboveB ecs i ℓ with
              | true => rfl
              | false => rw [hcv] at h3ℓ; simp at h3ℓ
            · intro ℓ hr hbelow
              have := H2 ℓ ((hget ℓ).mp hr)
              rw [(writtenBelowB_iff ecs i ℓ).mpr hbelow] at this
              simp at this
            · intro ℓ hw habove
              have := H4 ℓ ((hgetw ℓ).mp hw)
              rw [(readAboveB_iff ecs i ℓ).mpr habove] at this
              simp at this

/-! ## Decidable staleness (`ForwardStale`, `BackwardStale`) -/

/-- Forward staleness of cell `j` after running/deleting cell `i` whose
*new* write list is `wl`. -/
def fwdStaleB (ecs : List (ECell L)) (i : Nat) (wl : List L) (j : Nat) : Bool :=
  decide (i < j) &&
    (wListAt ecs i ++ wl).any
      (fun ℓ => decide (ℓ ∈ rListAt ecs j) || decide (ℓ ∈ wListAt ecs j))

/-- The nearest writer of `ℓ` strictly above `i`, if any. -/
def lastWriterB (ecs : List (ECell L)) (i : Nat) (ℓ : L) : Option Nat :=
  ((List.range i).filter (fun k => decide (ℓ ∈ wListAt ecs k))).max?

/-- Backward staleness of cell `j`: some location `ℓ` cell `i` used to
write but no longer does (`ℓ ∈ W_i \ wl`) has `j` as its nearest writer
above `i`. -/
def bwdStaleB (ecs : List (ECell L)) (i : Nat) (wl : List L) (j : Nat) : Bool :=
  (wListAt ecs i).any
    (fun ℓ => (!decide (ℓ ∈ wl)) && (lastWriterB ecs i ℓ == some j))

/-- `markedB`: cell `j` is marked stale by running/deleting cell `i`. -/
def markedB (ecs : List (ECell L)) (i : Nat) (wl : List L) (j : Nat) : Bool :=
  fwdStaleB ecs i wl j || bwdStaleB ecs i wl j

theorem fwdStaleB_iff {w : L → Prop} {ecs : List (ECell L)} {i : Nat} {wl : List L}
    {j : Nat} (hwl : ∀ ℓ, w ℓ ↔ ℓ ∈ wl) :
    fwdStaleB ecs i wl j = true ↔ FwdStale (ecs.map ECell.toCell) i w j := by
  unfold fwdStaleB FwdStale
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true, List.mem_append,
    Bool.or_eq_true]
  constructor
  · rintro ⟨hij, ℓ, hℓ, hrw⟩
    refine ⟨hij, ℓ, ?_, ?_⟩
    · rcases hℓ with h | h
      · exact Or.inl ((writesAt_iff ecs i ℓ).mpr h)
      · exact Or.inr ((hwl ℓ).mpr h)
    · rcases hrw with h | h
      · exact Or.inl ((readsAt_iff ecs j ℓ).mpr h)
      · exact Or.inr ((writesAt_iff ecs j ℓ).mpr h)
  · rintro ⟨hij, ℓ, hℓ, hrw⟩
    refine ⟨hij, ℓ, ?_, ?_⟩
    · rcases hℓ with h | h
      · exact Or.inl ((writesAt_iff ecs i ℓ).mp h)
      · exact Or.inr ((hwl ℓ).mp h)
    · rcases hrw with h | h
      · exact Or.inl ((readsAt_iff ecs j ℓ).mp h)
      · exact Or.inr ((writesAt_iff ecs j ℓ).mp h)

theorem lastWriterB_iff {ecs : List (ECell L)} {i : Nat} {ℓ : L} {j : Nat} :
    lastWriterB ecs i ℓ = some j ↔ IsLastWriter (ecs.map ECell.toCell) i ℓ j := by
  unfold lastWriterB IsLastWriter
  rw [List.max?_eq_some_iff]
  simp only [List.mem_filter, List.mem_range, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨hji, hwj⟩, hmax⟩
    refine ⟨hji, (writesAt_iff ecs j ℓ).mpr hwj, ?_⟩
    intro k hjk hki hwk
    have := hmax k ⟨hki, (writesAt_iff ecs k ℓ).mp hwk⟩
    omega
  · rintro ⟨hji, hwj, hmax⟩
    refine ⟨⟨hji, (writesAt_iff ecs j ℓ).mp hwj⟩, ?_⟩
    intro b ⟨hbi, hwb⟩
    rcases Nat.lt_or_ge j b with h | h
    · exact absurd ((writesAt_iff ecs b ℓ).mpr hwb) (hmax b h hbi)
    · exact h

theorem bwdStaleB_iff {w : L → Prop} {ecs : List (ECell L)} {i : Nat} {wl : List L}
    {j : Nat} (hwl : ∀ ℓ, w ℓ ↔ ℓ ∈ wl) :
    bwdStaleB ecs i wl j = true ↔ BwdStale (ecs.map ECell.toCell) i w j := by
  unfold bwdStaleB BwdStale
  simp only [List.any_eq_true, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
    beq_iff_eq]
  constructor
  · rintro ⟨ℓ, hℓW, hℓnw, hlw⟩
    exact ⟨ℓ, (writesAt_iff ecs i ℓ).mpr hℓW, fun h => hℓnw ((hwl ℓ).mp h),
      lastWriterB_iff.mp hlw⟩
  · rintro ⟨ℓ, hℓW, hℓnw, hlw⟩
    exact ⟨ℓ, (writesAt_iff ecs i ℓ).mp hℓW, fun h => hℓnw ((hwl ℓ).mpr h),
      lastWriterB_iff.mpr hlw⟩

theorem markedB_iff {w : L → Prop} {ecs : List (ECell L)} {i : Nat} {wl : List L}
    {j : Nat} (hwl : ∀ ℓ, w ℓ ↔ ℓ ∈ wl) :
    markedB ecs i wl j = true ↔ Marked (ecs.map ECell.toCell) i w j := by
  unfold markedB Marked
  rw [Bool.or_eq_true, fwdStaleB_iff hwl, bwdStaleB_iff hwl]

/-! ## The retag computation and its `RetagSpec` -/

/-- The retagged cell list: position `i` becomes `newCell`; every other
cell keeps its data and is marked stale iff `markedB` or already stale. -/
def retagList (ecs : List (ECell L)) (i : Nat) (wl : List L)
    (newCell : ECell L) : List (ECell L) :=
  ecs.mapIdx fun j c =>
    if j = i then newCell
    else { c with tag := if markedB ecs i wl j || decide (c.tag = Tag.stale) then
      .stale else .clean }

@[simp] theorem retagList_length (ecs : List (ECell L)) (i : Nat) (wl : List L)
    (newCell : ECell L) : (retagList ecs i wl newCell).length = ecs.length := by
  simp [retagList]

theorem retagList_get_i {ecs : List (ECell L)} {i : Nat} {wl : List L}
    {newCell : ECell L} (hi : i < ecs.length) :
    (retagList ecs i wl newCell)[i]? = some newCell := by
  have hci : ecs[i]? = some ecs[i] := List.getElem?_eq_getElem hi
  simp [retagList, List.getElem?_mapIdx, hci]

/-- The computed retag satisfies the relational `RetagSpec`. -/
theorem retagList_spec {w : L → Prop} {ecs : List (ECell L)} {i : Nat} {wl : List L}
    {newCell : ECell L} (hwl : ∀ ℓ, w ℓ ↔ ℓ ∈ wl) :
    RetagSpec (ecs.map ECell.toCell)
      ((retagList ecs i wl newCell).map ECell.toCell) i w := by
  intro j hji c hc
  rw [List.getElem?_map] at hc
  cases hej : ecs[j]? with
  | none => rw [hej] at hc; simp at hc
  | some ej =>
    rw [hej] at hc
    simp only [Option.map_some, Option.some.injEq] at hc
    subst hc
    -- compute retagList[j]
    have hrj : (retagList ecs i wl newCell)[j]? =
        some { ej with tag := if markedB ecs i wl j || decide (ej.tag = Tag.stale) then
          .stale else .clean } := by
      simp [retagList, List.getElem?_mapIdx, hej, hji]
    refine ⟨(if markedB ecs i wl j || decide (ej.tag = Tag.stale) then Tag.stale else Tag.clean),
      by rw [List.getElem?_map, hrj]; rfl, ?_⟩
    have htoc : (ECell.toCell ej).tag = ej.tag := rfl
    rw [htoc]
    by_cases hm : (markedB ecs i wl j || decide (ej.tag = Tag.stale)) = true
    · rw [if_pos hm]
      constructor
      · intro _
        rw [Bool.or_eq_true] at hm
        rcases hm with h | h
        · exact Or.inl ((markedB_iff hwl).mp h)
        · exact Or.inr (of_decide_eq_true h)
      · intro _; rfl
    · rw [if_neg hm]
      constructor
      · intro hcontra; exact absurd hcontra (by simp)
      · rintro (h | h)
        · exact absurd (by rw [Bool.or_eq_true]; exact Or.inl ((markedB_iff hwl).mpr h)) hm
        · exact absurd (by rw [Bool.or_eq_true]; exact Or.inr (decide_eq_true h)) hm

/-! ## `map` commutes with `set`/`insertIdx`/`eraseIdx` (generic helpers) -/

theorem map_eraseIdx {α β : Type} (f : α → β) (l : List α) (i : Nat) :
    (l.eraseIdx i).map f = (l.map f).eraseIdx i := by
  apply List.ext_getElem?
  intro j
  rw [List.getElem?_map]
  rcases Nat.lt_or_ge j i with h | h
  · rw [List.getElem?_eraseIdx_of_lt h, List.getElem?_eraseIdx_of_lt h, List.getElem?_map]
  · rw [List.getElem?_eraseIdx_of_ge h, List.getElem?_eraseIdx_of_ge h, List.getElem?_map]

theorem map_insertIdx {α β : Type} (f : α → β) (l : List α) (i : Nat) (a : α) :
    (l.insertIdx i a).map f = (l.map f).insertIdx i (f a) := by
  apply List.ext_getElem?
  intro j
  rw [List.getElem?_map]
  rcases Nat.lt_trichotomy j i with h | h | h
  · rw [List.getElem?_insertIdx_of_lt h, List.getElem?_insertIdx_of_lt h, List.getElem?_map]
  · subst h
    rw [List.getElem?_insertIdx_self, List.getElem?_insertIdx_self, List.length_map]
    by_cases hle : j ≤ l.length <;> simp [hle]
  · rw [List.getElem?_insertIdx_of_gt h, List.getElem?_insertIdx_of_gt h, List.getElem?_map]

/-- The reflection of a fresh executable cell is exactly the fresh cell
`[Inst-Insert]` inserts. -/
theorem toCell_fresh (c : LCmd L) :
    ECell.toCell (⟨c, none, .stale, [], []⟩ : ECell L) =
      ⟨c, none, .stale, fun _ => False, fun _ => False⟩ := by
  have hnil : (fun ℓ : L => ℓ ∈ ([] : List L)) = (fun _ => False) := by
    funext ℓ; exact propext (iff_false_intro List.not_mem_nil)
  show (⟨c, none, .stale, fun ℓ => ℓ ∈ ([] : List L), fun ℓ => ℓ ∈ ([] : List L)⟩ :
    Cell (LCmd L) Out L) = _
  rw [hnil]

/-! ## Executable operations -/

/-- The cell that running `ci` from store `σ` records: its output, reads,
and writes come from `evalCmd`; it is marked clean. -/
def evalNewCell (ci : ECell L) (σ : Store L Nat) : ECell L :=
  { code := ci.code, out := some (evalCmd ci.code σ).1, tag := Tag.clean,
    reads := (evalCmd ci.code σ).2.2.1, writes := (evalCmd ci.code σ).2.2.2 }

/-- `[Inst-Run]`: evaluate cell `i`, reject with the first rerun-consistency
violation, otherwise update outputs, store, and staleness tags. -/
def runCell (nb : ENotebook L) (i : Nat) : Except (Violation L) (ENotebook L) :=
  match nb.cells[i]? with
  | none => .error .noSuchCell
  | some ci =>
    match diagnose (nb.cells.set i (evalNewCell ci nb.store)) i with
    | some v => .error v
    | none =>
      .ok ⟨retagList nb.cells i (evalCmd ci.code nb.store).2.2.2 (evalNewCell ci nb.store),
           (evalCmd ci.code nb.store).2.1⟩

/-- `[Inst-Edit]`: replace code and mark stale. -/
def editCell (nb : ENotebook L) (i : Nat) (c : LCmd L) :
    Except (Violation L) (ENotebook L) :=
  match nb.cells[i]? with
  | none => .error .noSuchCell
  | some ci => .ok { nb with cells := nb.cells.set i { ci with code := c, tag := .stale } }

/-- `[Inst-Insert]`: insert a fresh stale cell. -/
def insertCell (nb : ENotebook L) (i : Nat) (c : LCmd L) :
    Except (Violation L) (ENotebook L) :=
  if i ≤ nb.cells.length then
    .ok { nb with cells := nb.cells.insertIdx i ⟨c, none, .stale, [], []⟩ }
  else .error .noSuchCell

/-- `[Inst-Delete]`: propagate staleness from the deleted cell's writes,
then remove it. -/
def deleteCell (nb : ENotebook L) (i : Nat) : Except (Violation L) (ENotebook L) :=
  match nb.cells[i]? with
  | none => .error .noSuchCell
  | some ci => .ok { nb with cells := (retagList nb.cells i [] ci).eraseIdx i }

/-! ## Soundness: each accepted operation is a genuine instrumented step -/

/-- Generic: a positive `getElem?` witnesses an in-range index. -/
theorem idx_lt {α : Type} {l : List α} {i : Nat} {a : α} (h : l[i]? = some a) :
    i < l.length := by
  rcases Nat.lt_or_ge i l.length with hh | hh
  · exact hh
  · rw [List.getElem?_eq_none hh] at h; exact absurd h (by simp)

theorem runCell_sound {nb nb' : ENotebook L} {i : Nat}
    (h : runCell nb i = .ok nb') : InstStep nb.toNb (.run i) nb'.toNb := by
  rw [runCell] at h
  split at h
  · exact absurd h (by simp)
  · rename_i ci hci
    split at h
    · exact absurd h (by simp)
    · rename_i hdiag
      rw [Except.ok.injEq] at h
      subst h
      have hi : i < nb.cells.length := idx_lt hci
      refine InstStep.run (ci := ECell.toCell ci)
        (o := (evalCmd ci.code nb.store).1) (σ' := (evalCmd ci.code nb.store).2.1)
        (r := fun ℓ => ℓ ∈ (evalCmd ci.code nb.store).2.2.1)
        (w := fun ℓ => ℓ ∈ (evalCmd ci.code nb.store).2.2.2)
        (cs' := (retagList nb.cells i (evalCmd ci.code nb.store).2.2.2
          (evalNewCell ci nb.store)).map ECell.toCell) ?_ ?_ ?_ ?_ ?_ ?_
      · -- hcell
        rw [toNb_cells, List.getElem?_map, hci]; rfl
      · -- heval
        exact ⟨rfl, rfl, fun ℓ => Iff.rfl, fun ℓ => Iff.rfl⟩
      · -- hrc
        have hd := diagnose_none_sound hdiag
        rw [List.map_set] at hd
        rw [toNb_cells]
        exact hd
      · -- hlen
        rw [toNb_cells, List.length_map, List.length_map, retagList_length]
      · -- hat
        rw [List.getElem?_map, retagList_get_i hi]; rfl
      · -- hretag
        rw [toNb_cells]
        exact retagList_spec (fun _ => Iff.rfl)

theorem editCell_sound {nb nb' : ENotebook L} {i : Nat} {c : LCmd L}
    (h : editCell nb i c = .ok nb') : InstStep nb.toNb (.edit i c) nb'.toNb := by
  rw [editCell] at h
  split at h
  · exact absurd h (by simp)
  · rename_i ci hci
    rw [Except.ok.injEq] at h
    subst h
    show InstStep nb.toNb (.edit i c)
      ⟨(nb.cells.set i { ci with code := c, tag := .stale }).map ECell.toCell, nb.store⟩
    rw [List.map_set]
    exact InstStep.edit (ci := ECell.toCell ci)
      (by rw [toNb_cells, List.getElem?_map, hci]; rfl)

theorem insertCell_sound {nb nb' : ENotebook L} {i : Nat} {c : LCmd L}
    (h : insertCell nb i c = .ok nb') : InstStep nb.toNb (.insert i c) nb'.toNb := by
  rw [insertCell] at h
  split at h
  · rename_i hle
    rw [Except.ok.injEq] at h
    subst h
    show InstStep nb.toNb (.insert i c)
      ⟨(nb.cells.insertIdx i ⟨c, none, .stale, [], []⟩).map ECell.toCell, nb.store⟩
    rw [map_insertIdx, toCell_fresh]
    exact InstStep.insert (by rw [toNb_cells, List.length_map]; exact hle)
  · exact absurd h (by simp)

theorem deleteCell_sound {nb nb' : ENotebook L} {i : Nat}
    (h : deleteCell nb i = .ok nb') : InstStep nb.toNb (.delete i) nb'.toNb := by
  rw [deleteCell] at h
  split at h
  · exact absurd h (by simp)
  · rename_i ci hci
    rw [Except.ok.injEq] at h
    subst h
    show InstStep nb.toNb (.delete i)
      ⟨((retagList nb.cells i [] ci).eraseIdx i).map ECell.toCell, nb.store⟩
    rw [map_eraseIdx]
    refine InstStep.delete (ci := ECell.toCell ci)
      (cs'' := (retagList nb.cells i [] ci).map ECell.toCell) ?_ ?_ ?_
    · rw [toNb_cells, List.getElem?_map, hci]; rfl
    · rw [toNb_cells, List.length_map, List.length_map, retagList_length]
    · rw [toNb_cells]
      exact retagList_spec (fun ℓ => ⟨fun hf => hf.elim, fun hm => absurd hm (by simp)⟩)

/-! ## An executable operation runner and the end-to-end guarantee -/

/-- Executable notebook operations. -/
inductive EOp (L : Type) where
  | run (i : Nat)
  | edit (i : Nat) (c : LCmd L)
  | insert (i : Nat) (c : LCmd L)
  | delete (i : Nat)

/-- Reflect an executable operation to a relational one. -/
def EOp.toOp : EOp L → Op (LCmd L)
  | .run i => .run i
  | .edit i c => .edit i c
  | .insert i c => .insert i c
  | .delete i => .delete i

/-- Run one executable operation. -/
def stepOp (nb : ENotebook L) : EOp L → Except (Violation L) (ENotebook L)
  | .run i => runCell nb i
  | .edit i c => editCell nb i c
  | .insert i c => insertCell nb i c
  | .delete i => deleteCell nb i

/-- Run a whole sequence, stopping at the first violation. -/
def runOps (nb : ENotebook L) : List (EOp L) → Except (Violation L) (ENotebook L)
  | [] => .ok nb
  | op :: rest =>
    match stepOp nb op with
    | .ok nb' => runOps nb' rest
    | .error v => .error v

theorem stepOp_sound {nb nb' : ENotebook L} {op : EOp L}
    (h : stepOp nb op = .ok nb') : InstStep nb.toNb op.toOp nb'.toNb := by
  cases op with
  | run i => exact runCell_sound h
  | edit i c => exact editCell_sound h
  | insert i c => exact insertCell_sound h
  | delete i => exact deleteCell_sound h

/-- **Soundness of the runner.**  If the executable kernel accepts an
operation sequence from a well-formed notebook, the resulting notebook
is again well-formed (by `preservation`). -/
theorem runOps_wellFormed {nb nb' : ENotebook L} {ops : List (EOp L)}
    (hwf : WellFormed nb.toNb) (h : runOps nb ops = .ok nb') : WellFormed nb'.toNb := by
  induction ops generalizing nb with
  | nil => rw [runOps] at h; rw [Except.ok.injEq] at h; subst h; exact hwf
  | cons op rest ih =>
    rw [runOps] at h
    split at h
    · rename_i nb1 hstep
      exact ih (preservation hwf (stepOp_sound hstep)) h
    · exact absurd h (by simp)

/-- A fresh, never-run executable cell. -/
def freshCell (c : LCmd L) : ECell L :=
  { code := c, out := none, tag := .stale, reads := [], writes := [] }

/-- Build an all-stale notebook from a list of cell sources. -/
def mkNb (cs : List (LCmd L)) : ENotebook L :=
  { cells := cs.map freshCell, store := Store.empty }

/-- Every freshly built notebook is well-formed (all cells stale). -/
theorem mkNb_wellFormed (cs : List (LCmd L)) : WellFormed (mkNb cs).toNb := by
  apply wellFormed_initial
  intro i c hc
  rw [toNb_cells, List.getElem?_map] at hc
  cases hcc : (mkNb cs).cells[i]? with
  | none => rw [hcc] at hc; simp at hc
  | some ec =>
    rw [hcc] at hc
    simp only [Option.map_some, Option.some.injEq] at hc
    subst hc
    show ec.tag = Tag.stale
    have hmap : (mkNb cs).cells[i]? = (cs[i]?).map freshCell := by
      simp [mkNb, List.getElem?_map]
    rw [hmap] at hcc
    cases hci : cs[i]? with
    | none => rw [hci] at hcc; simp at hcc
    | some cmd =>
      rw [hci] at hcc; simp only [Option.map_some, Option.some.injEq] at hcc
      subst hcc; rfl

/-- **End-to-end guarantee.**  Starting from a freshly built (all-stale)
notebook, if the executable kernel accepts an operation sequence and the
final notebook is all clean, then it is reproducible (its recorded
outputs match a top-to-bottom execution from the empty store). -/
theorem mkNb_runOps_reproducible {cs : List (LCmd L)} {ops : List (EOp L)}
    {nb' : ENotebook L} (h : runOps (mkNb cs) ops = .ok nb')
    (hac : AllClean nb'.toNb.cells) : Reproducible nb'.toNb.erase :=
  output_consistency (runOps_wellFormed (mkNb_wellFormed cs) h) hac

/-! ## Machine-checked replays of the litmus tests -/

section Demos

/-- Locations used by the demos. -/
private def vx : Loc String Nat String := .var "x"
private def vy : Loc String Nat String := .var "y"
private def va : Loc String Nat String := .var "a"
private def vz : Loc String Nat String := .var "z"
private def vother : Loc String Nat String := .var "other"
private def vdf : Loc String Nat String := .var "df"
private def dfy : Loc String Nat String := .col 0 "y"

/-- The observable outcome of a scenario: a rejection with its diagnostic,
or the final per-cell status tags.  (A plain sum, so `#eval` displays it
via `Repr` rather than trying to run it as a monad.) -/
inductive Result (L : Type) where
  | rejected (v : Violation L)
  | tags (ts : List Tag)
deriving Repr, DecidableEq

/-- Run a scenario and report either the rerun-consistency violation or
the final per-cell tags. -/
def scenario (cs : List (LCmd (Loc String Nat String)))
    (ops : List (EOp (Loc String Nat String))) : Result (Loc String Nat String) :=
  match runOps (mkNb cs) ops with
  | .error v => .rejected v
  | .ok nb => .tags (nb.cells.map (·.tag))

/-! Each `#guard` below fails the build unless the kernel reports exactly
the violation or tags shown in the paper's figure. -/

/-! ### Figure 3: rerun-consistency violations -/

/-
Scenario 1 (NoReadAndWrite): running `x = x + 1` after `x = 0` reads
and writes `x`. -/
#guard scenario [.const vx 0, .incr vx] [.run 0, .run 1] =
  .rejected (.noReadAndWrite vx)

/-
Scenario 2 (WriteBeforeRead): running `df.head()` with the cell that
defines `df` never executed. -/
#guard scenario [.const vdf 5, .use vdf] [.run 1] =
  .rejected (.writeBeforeRead vdf)

/-
Scenario 3 (NoReadBeforeWrite): `df["y"].sum()` reads a column written
by a cell below it. -/
#guard scenario [.use dfy, .const dfy 3] [.run 1, .run 0] =
  .rejected (.noReadBeforeWrite dfy)

/-
Scenario 4 (NoWriteAfterRead): `a = 100` overwrites `a` read by a cell
above it. -/
#guard scenario [.const va 1, .use va, .const va 100] [.run 0, .run 1, .run 2] =
  .rejected (.noWriteAfterRead va)

/-! ### Figure 4: staleness -/

/-
Scenario 1 (ForwardStale, write→read): after `J` (`x = 100`) and `K`
(`print(x)`) run, editing `J` to `x = 9999` and rerunning it marks `K`
stale. -/
#guard scenario [.const vx 100, .use vx]
    [.run 0, .run 1, .edit 0 (.const vx 9999), .run 0] =
  .tags [.clean, .stale]

/-
Scenario 2 (ForwardStale, write→write): running `M` (`y = 20`) and then
`L` (`y = 10`) above it marks `M` stale. -/
#guard scenario [.const vy 10, .const vy 20] [.run 1, .run 0] =
  .tags [.clean, .stale]

/-
Scenario 3 (ForwardStale, delete): after `N` (`df = ...`) and `O`
(`df.describe()`) run, deleting `N` marks `O` stale. -/
#guard scenario [.const vdf 5, .use vdf] [.run 0, .run 1, .delete 0] =
  .tags [.stale]

/-
Scenario 4 (BackwardStale + ForwardStale): after `P` (`z = 0`), `Q`
(`z = 99`), and `R` (`print(z)`) run, editing `Q` to `other = 99` and
rerunning it marks `P` stale (BackwardStale, to restore `z`) and `R`
stale (ForwardStale). -/
#guard scenario [.const vz 0, .const vz 99, .use vz]
    [.run 0, .run 1, .run 2, .edit 1 (.const vother 99), .run 1] =
  .tags [.stale, .clean, .stale]

end Demos

end Exec
end FlowBook
