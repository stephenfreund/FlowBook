/-
# A Concrete Model of the Evaluation Axioms

The whole development is parametric in a black-box cell evaluation
relation satisfying the `frame` and `locality` axioms of `CellEval`.
This file exhibits a concrete model — a miniature cell language of
constant assignments `dst := v` and copies `dst := src` — and proves it
satisfies the axioms.  This shows the axiomatization is consistent
(non-vacuous): the three theorems are about a nonempty class of
languages.

The paper's location grammar `ℓ ::= x | d.c` (`FlowBook.Loc`) is used
as the location type of the example instantiation at the bottom.
-/
import FlowBook.Preservation
import FlowBook.OutputConsistency
import FlowBook.Progress
import FlowBook.Erasure

namespace FlowBook
namespace Examples

/-- A miniature cell language: assign a constant to a location, or
copy the contents of one location into another. -/
inductive Cmd (L V : Type) where
  | assign (dst : L) (v : V)
  | copy (dst src : L)

/-- Pointwise store update. -/
def update {L V : Type} [DecidableEq L] (σ : Store L V) (ℓ : L) (v : Option V) :
    Store L V :=
  fun ℓ' => if ℓ' = ℓ then v else σ ℓ'

/-- The evaluation relation of the miniature language: an assignment
reads nothing and writes its destination; a copy reads its source and
writes its destination.  Outputs are trivial. -/
def CmdEval {L V : Type} [DecidableEq L] (c : Cmd L V) (σ : Store L V) (_ : Unit)
    (σ' : Store L V) (r w : L → Prop) : Prop :=
  match c with
  | .assign dst v =>
      σ' = update σ dst (some v) ∧ (∀ ℓ, ¬ r ℓ) ∧ (∀ ℓ, w ℓ ↔ ℓ = dst)
  | .copy dst src =>
      σ' = update σ dst (σ src) ∧ (∀ ℓ, r ℓ ↔ ℓ = src) ∧ (∀ ℓ, w ℓ ↔ ℓ = dst)

/-- The miniature language satisfies the `frame` and `locality` axioms. -/
instance instCellEval (L V : Type) [DecidableEq L] :
    CellEval (Cmd L V) Unit L V where
  Eval := CmdEval
  frame := by
    intro c σ o σ' r w heval ℓ hnw
    cases c with
    | assign dst v =>
      obtain ⟨hσ', _, hw⟩ := heval
      have hne : ℓ ≠ dst := fun h => hnw ((hw ℓ).mpr h)
      rw [hσ']
      simp [update, hne]
    | copy dst src =>
      obtain ⟨hσ', _, hw⟩ := heval
      have hne : ℓ ≠ dst := fun h => hnw ((hw ℓ).mpr h)
      rw [hσ']
      simp [update, hne]
  locality := by
    intro c σ o σ' r w τ heval hagree
    cases c with
    | assign dst v =>
      obtain ⟨hσ', hr, hw⟩ := heval
      refine ⟨update τ dst (some v), ⟨rfl, hr, hw⟩, ?_, ?_⟩
      · intro ℓ hwℓ
        have : ℓ = dst := (hw ℓ).mp hwℓ
        subst this
        rw [hσ']
        simp [update]
      · intro ℓ hnw
        have hne : ℓ ≠ dst := fun h => hnw ((hw ℓ).mpr h)
        simp [update, hne]
    | copy dst src =>
      obtain ⟨hσ', hr, hw⟩ := heval
      have hsrc : τ src = σ src := hagree src ((hr src).mpr rfl)
      refine ⟨update τ dst (τ src), ⟨rfl, hr, hw⟩, ?_, ?_⟩
      · intro ℓ hwℓ
        have : ℓ = dst := (hw ℓ).mp hwℓ
        subst this
        rw [hσ', hsrc]
        simp [update]
      · intro ℓ hnw
        have hne : ℓ ≠ dst := fun h => hnw ((hw ℓ).mpr h)
        simp [update, hne]

/-- The paper's location type, instantiated with string variable names,
numeric DataFrame addresses, and string column names. -/
abbrev PaperLoc : Type := Loc String Nat String

/-- Notebooks over the miniature language, with the paper's location
grammar and numeric values: all three theorems apply to them. -/
abbrev MiniNotebook : Type := Notebook (Cmd PaperLoc Nat) Unit PaperLoc Nat

example {nb nb' : MiniNotebook} {op : Op (Cmd PaperLoc Nat)}
    (hwf : WellFormed nb) (hstep : InstStep nb op nb') : WellFormed nb' :=
  preservation hwf hstep

example {nb : MiniNotebook} (hwf : WellFormed nb) (hclean : AllClean nb.cells) :
    Reproducible nb.erase :=
  output_consistency hwf hclean

example {nb : MiniNotebook} (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', FirstStaleExec F nb nb' F' ∧
      (Reproducible nb'.erase ∨ FirstStaleStuck nb' F') :=
  progress_reproducible hwf F

end Examples
end FlowBook
