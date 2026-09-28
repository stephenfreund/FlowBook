/-
# Theorem 2.4 (Reproducibility / Output Consistency)

If `S · I = (C, O, Σ) · (T, R, W)` is well-formed and `T_i = clean` for
all `i`, then `(C, O, Σ)` is reproducible: there exists `Σ'` such that
executing all cells top-to-bottom from the empty store produces exactly
the recorded outputs (`C ↓ O · Σ'`).

This is the full, machine-checked proof of Theorem 2.4 of the
supplement, corresponding to the proof in §4 of the supplement ("Output
Consistency: Well-Formed All-Clean States Are Output Consistent").  The induction
follows the paper exactly: `P(i)` asserts that the first `i` cells
execute top-to-bottom from the empty store, producing the recorded
outputs and a store `Σ_i` that agrees with the interactive store `Σ` on
`(⋃ W_{1..i}) \ (⋃ W_{i+1..n})`.  The inductive step uses rerun
consistency of cell `i` to show `R_i ⊆ (⋃ W_{1..i-1}) \ (⋃ W_{i..n})`,
so `Σ` and `Σ_{i-1}` agree on `R_i`, and the `locality` property of
cell evaluation replays the well-formedness witness of cell `i` from
`Σ_{i-1}`.
-/
import FlowBook.Preservation

namespace FlowBook

variable {Code Output L V : Type} [CellEval Code Output L V]

open CellEval

/-- All cells are tagged clean. -/
def AllClean (cs : List (Cell Code Output L)) : Prop :=
  ∀ i : Nat, ∀ c : Cell Code Output L, cs[i]? = some c → c.tag = Tag.clean

/-- **Theorem 2.4 (Reproducibility / Output Consistency).**
If `S · I = (C, O, Σ) · (T, R, W)` is well-formed and every cell is
clean, then `(C, O, Σ)` is reproducible: some top-to-bottom execution
from the empty store produces exactly the recorded outputs. -/
theorem output_consistency {nb : Notebook Code Output L V}
    (hwf : WellFormed nb) (hclean : AllClean nb.cells) :
    Reproducible nb.erase := by
  obtain ⟨cs, σ⟩ := nb
  replace hwf : ∀ i, IsClean cs i → Witnessed cs σ i ∧ RerunConsistent cs i := hwf
  replace hclean : ∀ i : Nat, ∀ c : Cell Code Output L, cs[i]? = some c → c.tag = Tag.clean :=
    hclean
  -- The paper's induction hypothesis P(k): the first k cells run
  -- top-to-bottom from ∅, producing the recorded outputs and a store τ
  -- agreeing with σ on (⋃ W_{0..k-1}) \ (⋃ W_{k..n-1}).
  have main : ∀ k, k ≤ cs.length →
      ∃ τ : Store L V,
        Runs Store.empty ((cs.take k).map fun c => (c.code, c.out)) τ ∧
        ∀ ℓ, WritesAbove cs k ℓ → ¬ WritesAt cs k ℓ → ¬ WritesBelow cs k ℓ →
          τ ℓ = σ ℓ := by
    intro k
    induction k with
    | zero =>
      intro _
      refine ⟨Store.empty, by simpa using Runs.nil, ?_⟩
      rintro ℓ ⟨j, hj, _⟩
      omega
    | succ k ih =>
      intro hk1
      obtain ⟨τ, hruns, hagree⟩ := ih (by omega)
      have hkcs : k < cs.length := by omega
      obtain ⟨c, hc⟩ := exists_getElem?_eq_some hkcs
      obtain ⟨hwit, hrc⟩ := hwf k ⟨c, hc, hclean k c hc⟩
      obtain ⟨c', hc', o, σstar, hout, heval, hagreeStar⟩ := hwit
      rw [hc] at hc'; cases hc'
      -- τ agrees with σ on the reads of cell k
      -- (WriteBeforeRead + NoReadAndWrite + NoReadBeforeWrite give
      --  R_k ⊆ (⋃ W_{0..k-1}) \ (⋃ W_{k..n-1}))
      have hreads : ∀ ℓ, c.reads ℓ → τ ℓ = σ ℓ := by
        intro ℓ hr
        have h1 : WritesAbove cs k ℓ := hrc.writeBeforeRead ℓ ⟨c, hc, hr⟩
        have h2 : ¬ WritesAt cs k ℓ := fun hw => hrc.noReadAndWrite ℓ ⟨c, hc, hr⟩ hw
        have h3 : ¬ WritesBelow cs k ℓ := fun hw =>
          hrc.noReadBeforeWrite ℓ ⟨c, hc, hr⟩ hw
        exact hagree ℓ h1 h2 h3
      -- replay the witness from τ
      obtain ⟨τ', heval', hτw, hτnw⟩ := CellEval.locality heval hreads
      refine ⟨τ', ?_, ?_⟩
      · -- extend the top-to-bottom run by cell k
        have htake : cs.take (k + 1) = cs.take k ++ [c] := by
          rw [List.take_add_one, hc]
          rfl
        rw [htake, List.map_append]
        have hsnoc := Runs.snoc (c := c.code) (o := o) hruns ⟨c.reads, c.writes, heval'⟩
        simpa [hout] using hsnoc
      · -- the new store agrees with σ on (⋃ W_{0..k}) \ (⋃ W_{k+1..n-1})
        intro ℓ habove hnotat hnotbelow
        have hnb_k : ¬ WritesBelow cs k ℓ := by
          rintro ⟨j, hj, hw⟩
          by_cases hjk : j = k + 1
          · subst hjk; exact hnotat hw
          · exact hnotbelow ⟨j, by omega, hw⟩
        by_cases hwℓ : c.writes ℓ
        · -- written by cell k: the replayed run writes the witness value,
          -- and the witness agrees with σ off ⋃ W_{k+1..n-1}
          have h1 : τ' ℓ = σstar ℓ := hτw ℓ hwℓ
          have h2 : σ ℓ = σstar ℓ := hagreeStar ℓ hnb_k
          rw [h1, ← h2]
        · -- untouched by cell k: inherited from the previous store
          have h1 : τ' ℓ = τ ℓ := hτnw ℓ hwℓ
          have habove_k : WritesAbove cs k ℓ := by
            obtain ⟨j, hj, hw⟩ := habove
            by_cases hjk : j = k
            · subst hjk
              obtain ⟨cc, hcc, hccw⟩ := hw
              rw [hc] at hcc; cases hcc
              exact absurd hccw hwℓ
            · exact ⟨j, by omega, hw⟩
          have hnotat_k : ¬ WritesAt cs k ℓ := by
            rintro ⟨cc, hcc, hccw⟩
            rw [hc] at hcc; cases hcc
            exact hwℓ hccw
          rw [h1]
          exact hagree ℓ habove_k hnotat_k hnb_k
  obtain ⟨τ, hruns, _⟩ := main cs.length (Nat.le_refl _)
  refine ⟨τ, ?_⟩
  show Runs Store.empty (cs.map fun c => (c.code, c.out)) τ
  rw [List.take_length] at hruns
  exact hruns

end FlowBook
