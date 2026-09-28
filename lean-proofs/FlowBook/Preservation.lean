/-
# Theorem 2.3 (Preservation): Notebook Operations Preserve Well-Formedness

If `S · I` is well-formed and `S · I ⟹op S' · I'`, then `S' · I'` is
well-formed.

This file gives the full, machine-checked proof of Theorem 2.3 of the
supplement, corresponding to the proof in §3 of the supplement
("Preservation: Notebook Operations Preserve Well-Formedness").
The case analysis follows the paper:

* `[Inst-Edit]`  — the edited cell becomes stale; every other clean
  cell keeps its witness verbatim.
* `[Inst-Run]`   — subcases `j < i`, `j = i`, `j > i`.  The paper's
  commuting diagram ("running `C_j` from `Σ'` yields the same behavior
  as from `Σ`") is realized by the `locality` axiom of `CellEval`.
  The case `j < i` additionally uses the maximality of `LastWriter` in
  `BackwardStale`: a clean cell `j < i` whose own write `ℓ` is exposed
  by the run (`ℓ ∈ W_i \ W'_i`) is either the last writer of `ℓ`
  above `i` — and hence marked stale — or shielded by a later writer
  of `ℓ`, which keeps `ℓ` inside `⋃ W_{j+1..n}`.
* `[Inst-Insert]` — index shifting; the new cell has empty read and
  write sets, so no rerun-consistency predicate is disturbed.
* `[Inst-Delete]` — index shifting, plus the same `LastWriter`
  argument as `[Inst-Run]` for residual writes of the deleted cell.
* `[Inst-Move]`  — composition of the delete and insert cases.
-/
import FlowBook.Analysis

namespace FlowBook

variable {Code Output L V : Type}

/-! ## Helpers about positional tables -/

/-- Two cell lists with pointwise-equal read/write tables. -/
def RWEquiv (cs₁ cs₂ : List (Cell Code Output L)) : Prop :=
  (∀ j ℓ, ReadsAt cs₁ j ℓ ↔ ReadsAt cs₂ j ℓ) ∧
  (∀ j ℓ, WritesAt cs₁ j ℓ ↔ WritesAt cs₂ j ℓ)

theorem RWEquiv.writesBelow {cs₁ cs₂ : List (Cell Code Output L)}
    (h : RWEquiv cs₁ cs₂) (i : Nat) (ℓ : L) :
    WritesBelow cs₁ i ℓ ↔ WritesBelow cs₂ i ℓ := by
  constructor <;> rintro ⟨j, hj, hw⟩
  · exact ⟨j, hj, (h.2 j ℓ).mp hw⟩
  · exact ⟨j, hj, (h.2 j ℓ).mpr hw⟩

theorem RWEquiv.writesAbove {cs₁ cs₂ : List (Cell Code Output L)}
    (h : RWEquiv cs₁ cs₂) (i : Nat) (ℓ : L) :
    WritesAbove cs₁ i ℓ ↔ WritesAbove cs₂ i ℓ := by
  constructor <;> rintro ⟨j, hj, hw⟩
  · exact ⟨j, hj, (h.2 j ℓ).mp hw⟩
  · exact ⟨j, hj, (h.2 j ℓ).mpr hw⟩

theorem RWEquiv.readsAbove {cs₁ cs₂ : List (Cell Code Output L)}
    (h : RWEquiv cs₁ cs₂) (i : Nat) (ℓ : L) :
    ReadsAbove cs₁ i ℓ ↔ ReadsAbove cs₂ i ℓ := by
  constructor <;> rintro ⟨j, hj, hw⟩
  · exact ⟨j, hj, (h.1 j ℓ).mp hw⟩
  · exact ⟨j, hj, (h.1 j ℓ).mpr hw⟩

theorem RWEquiv.rerunConsistent {cs₁ cs₂ : List (Cell Code Output L)}
    (h : RWEquiv cs₁ cs₂) (i : Nat)
    (hrc : RerunConsistent cs₁ i) : RerunConsistent cs₂ i where
  noReadAndWrite ℓ hr hw :=
    hrc.noReadAndWrite ℓ ((h.1 i ℓ).mpr hr) ((h.2 i ℓ).mpr hw)
  writeBeforeRead ℓ hr :=
    (h.writesAbove i ℓ).mp (hrc.writeBeforeRead ℓ ((h.1 i ℓ).mpr hr))
  noReadBeforeWrite ℓ hr hw :=
    hrc.noReadBeforeWrite ℓ ((h.1 i ℓ).mpr hr) ((h.writesBelow i ℓ).mpr hw)
  noWriteAfterRead ℓ hw hr :=
    hrc.noWriteAfterRead ℓ ((h.2 i ℓ).mpr hw) ((h.readsAbove i ℓ).mpr hr)

theorem RWEquiv.symm {cs₁ cs₂ : List (Cell Code Output L)} (h : RWEquiv cs₁ cs₂) :
    RWEquiv cs₂ cs₁ :=
  ⟨fun j ℓ => (h.1 j ℓ).symm, fun j ℓ => (h.2 j ℓ).symm⟩

/-! ## The `[Inst-Run]` case -/

section Run
variable [CellEval Code Output L V]

open CellEval

theorem preservation_run
    {cs : List (Cell Code Output L)} {σ : Store L V} {i : Nat}
    {ci ci' : Cell Code Output L}
    {o : Output} {σ' : Store L V} {r w : L → Prop}
    {cs' : List (Cell Code Output L)}
    (hwf : WellFormed (⟨cs, σ⟩ : Notebook Code Output L V))
    (hcell : cs[i]? = some ci)
    (heval : Eval ci.code σ o σ' r w)
    (hci' : ci' = { ci with out := some o, tag := .clean, reads := r, writes := w })
    (hrc : RerunConsistent (cs.set i ci') i)
    (hlen : cs'.length = cs.length)
    (hat : cs'[i]? = some ci')
    (hretag : RetagSpec cs cs' i w) :
    WellFormed (⟨cs', σ'⟩ : Notebook Code Output L V) := by
  have hi : i < cs.length := lt_length_of_getElem?_eq_some hcell
  -- fields of the updated cell
  have hci'code : ci'.code = ci.code := by rw [hci']
  have hci'out : ci'.out = some o := by rw [hci']
  have hci'reads : ci'.reads = r := by rw [hci']
  have hci'writes : ci'.writes = w := by rw [hci']
  -- table facts about `csm := cs.set i ci'`
  have hreads_csm_i : ∀ ℓ, ReadsAt (cs.set i ci') i ℓ ↔ r ℓ := by
    intro ℓ
    constructor
    · rintro ⟨c, hc, hcr⟩
      rw [List.getElem?_set_self hi] at hc; cases hc
      rw [hci'reads] at hcr; exact hcr
    · intro hr
      exact ⟨ci', by simp [List.getElem?_set_self hi], by rw [hci'reads]; exact hr⟩
  have hwrites_csm_i : ∀ ℓ, WritesAt (cs.set i ci') i ℓ ↔ w ℓ := by
    intro ℓ
    constructor
    · rintro ⟨c, hc, hcw⟩
      rw [List.getElem?_set_self hi] at hc; cases hc
      rw [hci'writes] at hcw; exact hcw
    · intro hw
      exact ⟨ci', by simp [List.getElem?_set_self hi], by rw [hci'writes]; exact hw⟩
  have hreads_csm_ne : ∀ j, j ≠ i → ∀ ℓ, (ReadsAt (cs.set i ci') j ℓ ↔ ReadsAt cs j ℓ) := by
    intro j hji ℓ
    constructor <;> rintro ⟨c, hc, hcr⟩
    · rw [List.getElem?_set_ne (by omega)] at hc; exact ⟨c, hc, hcr⟩
    · exact ⟨c, by rw [List.getElem?_set_ne (by omega)]; exact hc, hcr⟩
  have hwrites_csm_ne : ∀ j, j ≠ i → ∀ ℓ, (WritesAt (cs.set i ci') j ℓ ↔ WritesAt cs j ℓ) := by
    intro j hji ℓ
    constructor <;> rintro ⟨c, hc, hcw⟩
    · rw [List.getElem?_set_ne (by omega)] at hc; exact ⟨c, hc, hcw⟩
    · exact ⟨c, by rw [List.getElem?_set_ne (by omega)]; exact hc, hcw⟩
  -- `cs'` and `cs.set i ci'` have identical read/write tables.
  have hequiv : RWEquiv cs' (cs.set i ci') := by
    constructor <;> intro j ℓ
    · by_cases hji : j = i
      · subst hji
        constructor <;> rintro ⟨c, hc, hcr⟩
        · rw [hat] at hc; cases hc
          exact ⟨ci', by simp [List.getElem?_set_self hi], hcr⟩
        · rw [List.getElem?_set_self hi] at hc; cases hc
          exact ⟨ci', hat, hcr⟩
      · constructor <;> rintro ⟨c, hc, hcr⟩
        · have hjlt : j < cs.length := by
            have := lt_length_of_getElem?_eq_some hc; omega
          obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hjlt
          obtain ⟨t, hct, _⟩ := hretag j hji c₀ hc₀
          rw [hct] at hc; cases hc
          exact ⟨c₀, by rw [List.getElem?_set_ne (by omega)]; exact hc₀, hcr⟩
        · rw [List.getElem?_set_ne (by omega)] at hc
          obtain ⟨t, hct, _⟩ := hretag j hji c hc
          exact ⟨{ c with tag := t }, hct, hcr⟩
    · by_cases hji : j = i
      · subst hji
        constructor <;> rintro ⟨c, hc, hcw⟩
        · rw [hat] at hc; cases hc
          exact ⟨ci', by simp [List.getElem?_set_self hi], hcw⟩
        · rw [List.getElem?_set_self hi] at hc; cases hc
          exact ⟨ci', hat, hcw⟩
      · constructor <;> rintro ⟨c, hc, hcw⟩
        · have hjlt : j < cs.length := by
            have := lt_length_of_getElem?_eq_some hc; omega
          obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hjlt
          obtain ⟨t, hct, _⟩ := hretag j hji c₀ hc₀
          rw [hct] at hc; cases hc
          exact ⟨c₀, by rw [List.getElem?_set_ne (by omega)]; exact hc₀, hcw⟩
        · rw [List.getElem?_set_ne (by omega)] at hc
          obtain ⟨t, hct, _⟩ := hretag j hji c hc
          exact ⟨{ c with tag := t }, hct, hcw⟩
  -- disjointness facts from the rerun-consistency check of the run itself
  have hr_w_disj : ∀ ℓ, r ℓ → ¬ w ℓ := by
    intro ℓ hr hw
    exact hrc.noReadAndWrite ℓ ((hreads_csm_i ℓ).mpr hr) ((hwrites_csm_i ℓ).mpr hw)
  -- `NoWriteAfterRead` of the run: `w` avoids reads above `i`
  have hw_reads_above : ∀ j, j < i → ∀ ℓ, ReadsAt cs j ℓ → ¬ w ℓ := by
    intro j hj ℓ hr hw
    exact hrc.noWriteAfterRead ℓ ((hwrites_csm_i ℓ).mpr hw)
      ⟨j, hj, (hreads_csm_ne j (by omega) ℓ).mpr hr⟩
  -- `NoReadBeforeWrite` of the run: `r` avoids writes below `i`
  have hr_writes_below : ∀ j, i < j → ∀ ℓ, WritesAt cs j ℓ → ¬ r ℓ := by
    intro j hj ℓ hw hr
    exact hrc.noReadBeforeWrite ℓ ((hreads_csm_i ℓ).mpr hr)
      ⟨j, hj, (hwrites_csm_ne j (by omega) ℓ).mpr hw⟩
  -- the store update is framed by `w`
  have hframe : ∀ ℓ, ¬ w ℓ → σ' ℓ = σ ℓ := CellEval.frame heval
  -- Now prove well-formedness of the new state.
  intro j hclean
  replace hclean : IsClean cs' j := hclean
  show Witnessed cs' σ' j ∧ RerunConsistent cs' j
  obtain ⟨cj', hcj', htag⟩ := hclean
  by_cases hji : j = i
  · -- Subcase j = i: the run itself is the witness.
    subst hji
    constructor
    · -- Witnessed: re-run cell i from σ'; it reads none of its own writes.
      have hagree : ∀ ℓ, r ℓ → σ' ℓ = σ ℓ := fun ℓ hr => hframe ℓ (hr_w_disj ℓ hr)
      obtain ⟨τ', heval', hτw, hτnw⟩ := CellEval.locality heval hagree
      refine ⟨ci', hat, o, τ', hci'out, ?_, ?_⟩
      · rw [hci'code, hci'reads, hci'writes]; exact heval'
      · intro ℓ _
        by_cases hw : w ℓ
        · exact (hτw ℓ hw).symm
        · exact (hτnw ℓ hw).symm
    · exact hequiv.symm.rerunConsistent _ hrc
  · -- Subcase j ≠ i: cell j was clean before the run.
    have hjlt : j < cs.length := by
      have := lt_length_of_getElem?_eq_some hcj'; omega
    obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hjlt
    obtain ⟨t, hct, hstale⟩ := hretag j hji c₀ hc₀
    rw [hct] at hcj'; cases hcj'
    simp only at htag
    have hnotmarked : ¬ (Marked cs i w j ∨ c₀.tag = Tag.stale) := by
      intro hm
      have := hstale.mpr hm
      rw [htag] at this
      cases this
    have hnotm : ¬ Marked cs i w j := fun h => hnotmarked (.inl h)
    have hc₀clean : c₀.tag = .clean := by
      rcases Tag.clean_or_stale c₀.tag with h | h
      · exact h
      · exact absurd (.inr h) hnotmarked
    obtain ⟨hwit, hrcj⟩ := hwf j ⟨c₀, hc₀, hc₀clean⟩
    replace hwit : Witnessed cs σ j := hwit
    replace hrcj : RerunConsistent cs j := hrcj
    obtain ⟨c₀', hc₀', oj, σ''', hout, hevalj, hagreej⟩ := hwit
    rw [hc₀] at hc₀'; cases hc₀'
    rcases Nat.lt_or_ge j i with hjilt | hjige
    · -- Sub-subcase j < i.
      have hrw : ∀ ℓ, c₀.reads ℓ → ¬ w ℓ := by
        intro ℓ hr hw
        exact hw_reads_above j hjilt ℓ ⟨c₀, hc₀, hr⟩ hw
      have hrcj' : RerunConsistent (cs.set i ci') j := by
        constructor
        · intro ℓ hr hw
          exact hrcj.noReadAndWrite ℓ
            ((hreads_csm_ne j hji ℓ).mp hr) ((hwrites_csm_ne j hji ℓ).mp hw)
        · intro ℓ hr
          obtain ⟨k, hk, hkw⟩ := hrcj.writeBeforeRead ℓ ((hreads_csm_ne j hji ℓ).mp hr)
          exact ⟨k, hk, (hwrites_csm_ne k (by omega) ℓ).mpr hkw⟩
        · rintro ℓ hr ⟨k, hk, hkw⟩
          have hrold : ReadsAt cs j ℓ := (hreads_csm_ne j hji ℓ).mp hr
          by_cases hki : k = i
          · subst hki
            exact hw_reads_above j hjilt ℓ hrold ((hwrites_csm_i ℓ).mp hkw)
          · exact hrcj.noReadBeforeWrite ℓ hrold
              ⟨k, hk, (hwrites_csm_ne k hki ℓ).mp hkw⟩
        · rintro ℓ hw ⟨k, hk, hkr⟩
          have hwold : WritesAt cs j ℓ := (hwrites_csm_ne j hji ℓ).mp hw
          have hki : k ≠ i := by omega
          exact hrcj.noWriteAfterRead ℓ hwold ⟨k, hk, (hreads_csm_ne k hki ℓ).mp hkr⟩
      refine ⟨?_, hequiv.symm.rerunConsistent j hrcj'⟩
      -- Witnessed at j from σ': replay via locality.
      have hagree : ∀ ℓ, c₀.reads ℓ → σ' ℓ = σ ℓ := fun ℓ hr => hframe ℓ (hrw ℓ hr)
      obtain ⟨τ', hevalj', hτw, hτnw⟩ := CellEval.locality hevalj hagree
      refine ⟨{ c₀ with tag := t }, hct, oj, τ', hout, hevalj', ?_⟩
      intro ℓ hnb
      have hnbm : ¬ WritesBelow (cs.set i ci') j ℓ :=
        fun h => hnb ((hequiv.writesBelow j ℓ).mpr h)
      by_cases hwℓ : c₀.writes ℓ
      · -- ℓ is written by cell j itself: value comes from the old witness.
        have hτ : τ' ℓ = σ''' ℓ := hτw ℓ hwℓ
        have hnowriter : ∀ k, j < k → k ≠ i → ¬ WritesAt cs k ℓ := by
          intro k hk hki hkw
          exact hnbm ⟨k, hk, (hwrites_csm_ne k hki ℓ).mpr hkw⟩
        have hnw : ¬ w ℓ := by
          intro hw
          exact hnbm ⟨i, hjilt, (hwrites_csm_i ℓ).mpr hw⟩
        -- ℓ is not written below j in the OLD tables either: the only
        -- candidate writer is i, and then j is the last writer of ℓ
        -- above i, so BackwardStale would have marked j.
        have hnold : ¬ WritesBelow cs j ℓ := by
          rintro ⟨k, hk, hkw⟩
          by_cases hki : k = i
          · subst hki
            apply hnotm
            refine .inr ⟨ℓ, hkw, hnw, hjilt, ⟨c₀, hc₀, hwℓ⟩, ?_⟩
            intro k' hk1 hk2 hk'w
            exact hnowriter k' hk1 (by omega) hk'w
          · exact hnowriter k hk hki hkw
        have h1 : σ ℓ = σ''' ℓ := hagreej ℓ hnold
        have h2 : σ' ℓ = σ ℓ := hframe ℓ hnw
        rw [h2, h1, hτ]
      · exact (hτnw ℓ hwℓ).symm
    · -- Sub-subcase j > i.
      have hij : i < j := by omega
      -- not forward stale: (W_i ∪ w) ∩ (R_j ∪ W_j) = ∅
      have hfwd : ∀ ℓ, (WritesAt cs i ℓ ∨ w ℓ) → ¬ (ReadsAt cs j ℓ ∨ WritesAt cs j ℓ) := by
        intro ℓ h1 h2
        exact hnotm (.inl ⟨hij, ℓ, h1, h2⟩)
      have hrcj' : RerunConsistent (cs.set i ci') j := by
        constructor
        · intro ℓ hr hw
          exact hrcj.noReadAndWrite ℓ
            ((hreads_csm_ne j hji ℓ).mp hr) ((hwrites_csm_ne j hji ℓ).mp hw)
        · intro ℓ hr
          have hrold : ReadsAt cs j ℓ := (hreads_csm_ne j hji ℓ).mp hr
          obtain ⟨k, hk, hkw⟩ := hrcj.writeBeforeRead ℓ hrold
          have hki : k ≠ i := by
            intro hki; subst hki
            exact hfwd ℓ (.inl hkw) (.inl hrold)
          exact ⟨k, hk, (hwrites_csm_ne k hki ℓ).mpr hkw⟩
        · rintro ℓ hr ⟨k, hk, hkw⟩
          have hrold : ReadsAt cs j ℓ := (hreads_csm_ne j hji ℓ).mp hr
          have hki : k ≠ i := by omega
          exact hrcj.noReadBeforeWrite ℓ hrold ⟨k, hk, (hwrites_csm_ne k hki ℓ).mp hkw⟩
        · rintro ℓ hw ⟨k, hk, hkr⟩
          have hwold : WritesAt cs j ℓ := (hwrites_csm_ne j hji ℓ).mp hw
          by_cases hki : k = i
          · subst hki
            exact hr_writes_below j hij ℓ hwold ((hreads_csm_i ℓ).mp hkr)
          · exact hrcj.noWriteAfterRead ℓ hwold ⟨k, hk, (hreads_csm_ne k hki ℓ).mp hkr⟩
      refine ⟨?_, hequiv.symm.rerunConsistent j hrcj'⟩
      have hagree : ∀ ℓ, c₀.reads ℓ → σ' ℓ = σ ℓ := by
        intro ℓ hr
        exact hframe ℓ (fun hw => hfwd ℓ (.inr hw) (.inl ⟨c₀, hc₀, hr⟩))
      obtain ⟨τ', hevalj', hτw, hτnw⟩ := CellEval.locality hevalj hagree
      refine ⟨{ c₀ with tag := t }, hct, oj, τ', hout, hevalj', ?_⟩
      intro ℓ hnb
      have hnbm : ¬ WritesBelow (cs.set i ci') j ℓ :=
        fun h => hnb ((hequiv.writesBelow j ℓ).mpr h)
      by_cases hwℓ : c₀.writes ℓ
      · have hτ : τ' ℓ = σ''' ℓ := hτw ℓ hwℓ
        have hnold : ¬ WritesBelow cs j ℓ := by
          rintro ⟨k, hk, hkw⟩
          have hki : k ≠ i := by omega
          exact hnbm ⟨k, hk, (hwrites_csm_ne k hki ℓ).mpr hkw⟩
        have h1 : σ ℓ = σ''' ℓ := hagreej ℓ hnold
        have h2 : σ' ℓ = σ ℓ :=
          hframe ℓ (fun hw => hfwd ℓ (.inr hw) (.inr ⟨c₀, hc₀, hwℓ⟩))
        rw [h2, h1, hτ]
      · exact (hτnw ℓ hwℓ).symm

end Run

/-! ## The `[Inst-Edit]` case -/

section Edit
variable [CellEval Code Output L V]

theorem preservation_edit
    {cs : List (Cell Code Output L)} {σ : Store L V} {i : Nat}
    {ci ci' : Cell Code Output L} {c : Code}
    (hwf : WellFormed (⟨cs, σ⟩ : Notebook Code Output L V))
    (hcell : cs[i]? = some ci)
    (hci' : ci' = { ci with code := c, tag := .stale }) :
    WellFormed (⟨cs.set i ci', σ⟩ : Notebook Code Output L V) := by
  have hi : i < cs.length := lt_length_of_getElem?_eq_some hcell
  have hci'tag : ci'.tag = .stale := by rw [hci']
  have hci'reads : ci'.reads = ci.reads := by rw [hci']
  have hci'writes : ci'.writes = ci.writes := by rw [hci']
  -- editing changes only code and tag: the read/write tables are unchanged
  have hequiv : RWEquiv (cs.set i ci') cs := by
    constructor <;> intro j ℓ <;> by_cases hji : j = i
    · subst hji
      constructor
      · rintro ⟨cc, hc, hcr⟩
        rw [List.getElem?_set_self hi] at hc; cases hc
        rw [hci'reads] at hcr
        exact ⟨ci, hcell, hcr⟩
      · rintro ⟨cc, hc, hcr⟩
        rw [hcell] at hc; cases hc
        exact ⟨ci', by simp [List.getElem?_set_self hi],
          by rw [hci'reads]; exact hcr⟩
    · constructor
      · rintro ⟨cc, hc, hcr⟩
        rw [List.getElem?_set_ne (by omega)] at hc
        exact ⟨cc, hc, hcr⟩
      · rintro ⟨cc, hc, hcr⟩
        exact ⟨cc, by rw [List.getElem?_set_ne (by omega)]; exact hc, hcr⟩
    · subst hji
      constructor
      · rintro ⟨cc, hc, hcw⟩
        rw [List.getElem?_set_self hi] at hc; cases hc
        rw [hci'writes] at hcw
        exact ⟨ci, hcell, hcw⟩
      · rintro ⟨cc, hc, hcw⟩
        rw [hcell] at hc; cases hc
        exact ⟨ci', by simp [List.getElem?_set_self hi],
          by rw [hci'writes]; exact hcw⟩
    · constructor
      · rintro ⟨cc, hc, hcw⟩
        rw [List.getElem?_set_ne (by omega)] at hc
        exact ⟨cc, hc, hcw⟩
      · rintro ⟨cc, hc, hcw⟩
        exact ⟨cc, by rw [List.getElem?_set_ne (by omega)]; exact hc, hcw⟩
  intro j hclean
  replace hclean : IsClean (cs.set i ci') j := hclean
  show Witnessed (cs.set i ci') σ j ∧ RerunConsistent (cs.set i ci') j
  obtain ⟨cj, hcj, htag⟩ := hclean
  have hji : j ≠ i := by
    intro hji; subst hji
    rw [List.getElem?_set_self hi] at hcj; cases hcj
    rw [hci'tag] at htag
    cases htag
  rw [List.getElem?_set_ne (by omega)] at hcj
  obtain ⟨hwit, hrcj⟩ := hwf j ⟨cj, hcj, htag⟩
  replace hwit : Witnessed cs σ j := hwit
  replace hrcj : RerunConsistent cs j := hrcj
  refine ⟨?_, hequiv.symm.rerunConsistent j hrcj⟩
  obtain ⟨c₀, hc₀, oj, σ''', hout, hevalj, hagreej⟩ := hwit
  rw [hcj] at hc₀; cases hc₀
  refine ⟨cj, by rw [List.getElem?_set_ne (by omega)]; exact hcj,
    oj, σ''', hout, hevalj, ?_⟩
  intro ℓ hnb
  exact hagreej ℓ (fun h => hnb ((hequiv.writesBelow j ℓ).mpr h))

end Edit

/-! ## The `[Inst-Insert]` case -/

section Insert
variable [CellEval Code Output L V]

/-- Preservation for `[Inst-Insert]`, stated for any inserted cell that
is stale with empty read and write sets. -/
theorem preservation_insert
    {cs : List (Cell Code Output L)} {σ : Store L V} {i : Nat}
    {newc : Cell Code Output L}
    (hwf : WellFormed (⟨cs, σ⟩ : Notebook Code Output L V))
    (hle : i ≤ cs.length)
    (htagn : newc.tag = .stale)
    (hreadsn : ∀ ℓ, ¬ newc.reads ℓ)
    (hwritesn : ∀ ℓ, ¬ newc.writes ℓ) :
    WellFormed (⟨cs.insertIdx i newc, σ⟩ : Notebook Code Output L V) := by
  have hget_lt : ∀ j, j < i → (cs.insertIdx i newc)[j]? = cs[j]? := by
    intro j hj; rw [List.getElem?_insertIdx_of_lt hj]
  have hget_self : (cs.insertIdx i newc)[i]? = some newc := by
    rw [List.getElem?_insertIdx_self]; simp [hle]
  have hget_gt : ∀ j, i < j → (cs.insertIdx i newc)[j]? = cs[j - 1]? := by
    intro j hj; rw [List.getElem?_insertIdx_of_gt hj]
  intro j hclean
  replace hclean : IsClean (cs.insertIdx i newc) j := hclean
  show Witnessed (cs.insertIdx i newc) σ j ∧ RerunConsistent (cs.insertIdx i newc) j
  obtain ⟨cj, hcj, htag⟩ := hclean
  have hji : j ≠ i := by
    intro hji; subst hji
    rw [hget_self] at hcj; cases hcj
    rw [htagn] at htag
    cases htag
  rcases Nat.lt_or_ge j i with hjlt | hjge
  · -- j < i
    rw [hget_lt j hjlt] at hcj
    obtain ⟨hwit, hrcj⟩ := hwf j ⟨cj, hcj, htag⟩
    replace hwit : Witnessed cs σ j := hwit
    replace hrcj : RerunConsistent cs j := hrcj
    constructor
    · obtain ⟨c₀, hc₀, oj, σ''', hout, hevalj, hagreej⟩ := hwit
      rw [hcj] at hc₀; cases hc₀
      refine ⟨cj, by rw [hget_lt j hjlt]; exact hcj, oj, σ''', hout, hevalj, ?_⟩
      intro ℓ hnb
      refine hagreej ℓ ?_
      rintro ⟨k, hk, ck, hck, hckw⟩
      by_cases hki : k < i
      · exact hnb ⟨k, hk, ck, by rw [hget_lt k hki]; exact hck, hckw⟩
      · refine hnb ⟨k + 1, by omega, ck, ?_, hckw⟩
        rw [hget_gt (k + 1) (by omega)]
        simpa using hck
    · constructor
      · rintro ℓ ⟨cc, hc, hcr⟩ ⟨cc', hc', hcw⟩
        rw [hget_lt j hjlt] at hc hc'
        exact hrcj.noReadAndWrite ℓ ⟨cc, hc, hcr⟩ ⟨cc', hc', hcw⟩
      · rintro ℓ ⟨cc, hc, hcr⟩
        rw [hget_lt j hjlt] at hc
        obtain ⟨k, hk, ck, hck, hckw⟩ := hrcj.writeBeforeRead ℓ ⟨cc, hc, hcr⟩
        exact ⟨k, hk, ck, by rw [hget_lt k (by omega)]; exact hck, hckw⟩
      · rintro ℓ ⟨cc, hc, hcr⟩ ⟨k, hk, ck, hck, hckw⟩
        rw [hget_lt j hjlt] at hc
        by_cases hki : k = i
        · subst hki; rw [hget_self] at hck; cases hck
          exact hwritesn ℓ hckw
        · rcases Nat.lt_or_ge k i with hklt | hkge
          · rw [hget_lt k hklt] at hck
            exact hrcj.noReadBeforeWrite ℓ ⟨cc, hc, hcr⟩ ⟨k, hk, ck, hck, hckw⟩
          · rw [hget_gt k (by omega)] at hck
            exact hrcj.noReadBeforeWrite ℓ ⟨cc, hc, hcr⟩
              ⟨k - 1, by omega, ck, hck, hckw⟩
      · rintro ℓ ⟨cc, hc, hcw⟩ ⟨k, hk, ck, hck, hckr⟩
        rw [hget_lt j hjlt] at hc
        have hklt : k < i := by omega
        rw [hget_lt k hklt] at hck
        exact hrcj.noWriteAfterRead ℓ ⟨cc, hc, hcw⟩ ⟨k, hk, ck, hck, hckr⟩
  · -- j > i: the cell was at j - 1
    have hij : i < j := by omega
    rw [hget_gt j hij] at hcj
    obtain ⟨hwit, hrcj⟩ := hwf (j - 1) ⟨cj, hcj, htag⟩
    replace hwit : Witnessed cs σ (j - 1) := hwit
    replace hrcj : RerunConsistent cs (j - 1) := hrcj
    constructor
    · obtain ⟨c₀, hc₀, oj, σ''', hout, hevalj, hagreej⟩ := hwit
      rw [hcj] at hc₀; cases hc₀
      refine ⟨cj, by rw [hget_gt j hij]; exact hcj, oj, σ''', hout, hevalj, ?_⟩
      intro ℓ hnb
      refine hagreej ℓ ?_
      rintro ⟨k, hk, ck, hck, hckw⟩
      -- writer k > j-1 ≥ i in cs sits at k+1 > j in the new list
      refine hnb ⟨k + 1, by omega, ck, ?_, hckw⟩
      rw [hget_gt (k + 1) (by omega)]
      simpa using hck
    · constructor
      · rintro ℓ ⟨cc, hc, hcr⟩ ⟨cc', hc', hcw⟩
        rw [hget_gt j hij] at hc hc'
        exact hrcj.noReadAndWrite ℓ ⟨cc, hc, hcr⟩ ⟨cc', hc', hcw⟩
      · rintro ℓ ⟨cc, hc, hcr⟩
        rw [hget_gt j hij] at hc
        obtain ⟨k, hk, ck, hck, hckw⟩ := hrcj.writeBeforeRead ℓ ⟨cc, hc, hcr⟩
        rcases Nat.lt_or_ge k i with hklt | hkge
        · exact ⟨k, by omega, ck, by rw [hget_lt k hklt]; exact hck, hckw⟩
        · refine ⟨k + 1, by omega, ck, ?_, hckw⟩
          rw [hget_gt (k + 1) (by omega)]
          simpa using hck
      · rintro ℓ ⟨cc, hc, hcr⟩ ⟨k, hk, ck, hck, hckw⟩
        rw [hget_gt j hij] at hc
        have hki : i < k := by omega
        rw [hget_gt k hki] at hck
        exact hrcj.noReadBeforeWrite ℓ ⟨cc, hc, hcr⟩ ⟨k - 1, by omega, ck, hck, hckw⟩
      · rintro ℓ ⟨cc, hc, hcw⟩ ⟨k, hk, ck, hck, hckr⟩
        rw [hget_gt j hij] at hc
        by_cases hki : k = i
        · subst hki; rw [hget_self] at hck; cases hck
          exact hreadsn ℓ hckr
        · rcases Nat.lt_or_ge k i with hklt | hkge
          · rw [hget_lt k hklt] at hck
            exact hrcj.noWriteAfterRead ℓ ⟨cc, hc, hcw⟩ ⟨k, by omega, ck, hck, hckr⟩
          · rw [hget_gt k (by omega)] at hck
            exact hrcj.noWriteAfterRead ℓ ⟨cc, hc, hcw⟩
              ⟨k - 1, by omega, ck, hck, hckr⟩

end Insert

/-! ## The `[Inst-Delete]` case -/

section Delete
variable [CellEval Code Output L V]

theorem preservation_delete
    {cs : List (Cell Code Output L)} {σ : Store L V} {i : Nat}
    {ci : Cell Code Output L}
    {cs'' : List (Cell Code Output L)}
    (hwf : WellFormed (⟨cs, σ⟩ : Notebook Code Output L V))
    (hcell : cs[i]? = some ci)
    (hlen : cs''.length = cs.length)
    (hretag : RetagSpec cs cs'' i (fun _ => False)) :
    WellFormed (⟨cs''.eraseIdx i, σ⟩ : Notebook Code Output L V) := by
  have hi : i < cs.length := lt_length_of_getElem?_eq_some hcell
  -- position j of the result corresponds to position jo of cs
  have hget : ∀ j : Nat, ∀ cj : Cell Code Output L, (cs''.eraseIdx i)[j]? = some cj →
      ∃ jo, ((j < i ∧ jo = j) ∨ (i ≤ j ∧ jo = j + 1)) ∧ jo ≠ i ∧
        ∃ c₀, cs[jo]? = some c₀ ∧ cj = { c₀ with tag := cj.tag } ∧
          (cj.tag = Tag.stale ↔
            (Marked cs i (fun _ => False) jo ∨ c₀.tag = Tag.stale)) := by
    intro j cj hcj
    rcases Nat.lt_or_ge j i with hjlt | hjge
    · rw [List.getElem?_eraseIdx_of_lt hjlt] at hcj
      have hjcs : j < cs.length := by
        have := lt_length_of_getElem?_eq_some (cs := cs'') hcj; omega
      obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hjcs
      obtain ⟨t, hct, hstale⟩ := hretag j (by omega) c₀ hc₀
      rw [hct] at hcj; cases hcj
      exact ⟨j, .inl ⟨hjlt, rfl⟩, by omega, c₀, hc₀, rfl, hstale⟩
    · rw [List.getElem?_eraseIdx_of_ge hjge] at hcj
      have hjcs : j + 1 < cs.length := by
        have := lt_length_of_getElem?_eq_some (cs := cs'') hcj; omega
      obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hjcs
      obtain ⟨t, hct, hstale⟩ := hretag (j + 1) (by omega) c₀ hc₀
      rw [hct] at hcj; cases hcj
      exact ⟨j + 1, .inr ⟨hjge, rfl⟩, by omega, c₀, hc₀, rfl, hstale⟩
  -- conversely, old position k ≠ i appears in the result
  have old_to_new : ∀ k, k ≠ i → ∀ ck : Cell Code Output L, cs[k]? = some ck →
      ∃ t k', ((k < i ∧ k' = k) ∨ (i < k ∧ k' = k - 1)) ∧
        (cs''.eraseIdx i)[k']? = some { ck with tag := t } := by
    intro k hk ck hck
    obtain ⟨t, hct, _⟩ := hretag k hk ck hck
    rcases Nat.lt_or_ge k i with hlt | hge
    · refine ⟨t, k, .inl ⟨hlt, rfl⟩, ?_⟩
      rw [List.getElem?_eraseIdx_of_lt hlt]; exact hct
    · refine ⟨t, k - 1, .inr ⟨by omega, rfl⟩, ?_⟩
      rw [List.getElem?_eraseIdx_of_ge (by omega)]
      have hkk : k - 1 + 1 = k := by omega
      rw [hkk]; exact hct
  intro j hclean
  replace hclean : IsClean (cs''.eraseIdx i) j := hclean
  show Witnessed (cs''.eraseIdx i) σ j ∧ RerunConsistent (cs''.eraseIdx i) j
  obtain ⟨cj, hcj, htag⟩ := hclean
  obtain ⟨jo, hjo, hjo_ne, c₀, hc₀, hcjc₀, hstale⟩ := hget j cj hcj
  have hcjr : cj.reads = c₀.reads := by rw [hcjc₀]
  have hcjw : cj.writes = c₀.writes := by rw [hcjc₀]
  have hcjcode : cj.code = c₀.code := by rw [hcjc₀]
  have hcjout : cj.out = c₀.out := by rw [hcjc₀]
  have hnotmarked : ¬ (Marked cs i (fun _ => False) jo ∨ c₀.tag = Tag.stale) := by
    intro hm
    have := hstale.mpr hm
    rw [htag] at this; cases this
  have hnotm : ¬ Marked cs i (fun _ => False) jo := fun h => hnotmarked (.inl h)
  have hc₀clean : c₀.tag = .clean := by
    rcases Tag.clean_or_stale c₀.tag with h | h
    · exact h
    · exact absurd (.inr h) hnotmarked
  obtain ⟨hwit, hrcj⟩ := hwf jo ⟨c₀, hc₀, hc₀clean⟩
  replace hwit : Witnessed cs σ jo := hwit
  replace hrcj : RerunConsistent cs jo := hrcj
  constructor
  · -- Witnessed
    obtain ⟨c₀', hc₀', oj, σ''', hout, hevalj, hagreej⟩ := hwit
    rw [hc₀] at hc₀'; cases hc₀'
    refine ⟨cj, hcj, oj, σ''', by rw [hcjout]; exact hout, ?_, ?_⟩
    · rw [hcjcode, hcjr, hcjw]; exact hevalj
    · intro ℓ hnb
      by_cases hwℓ : c₀.writes ℓ
      · -- every surviving writer below jo yields a writer below j in the result
        have hnowriter : ∀ k, jo < k → k ≠ i → ¬ WritesAt cs k ℓ := by
          rintro k hk hki ⟨ck, hck, hckw⟩
          obtain ⟨t, k', hk', hck'⟩ := old_to_new k hki ck hck
          exact hnb ⟨k', by omega, { ck with tag := t }, hck', hckw⟩
        have hnold : ¬ WritesBelow cs jo ℓ := by
          rintro ⟨k, hk, hkw⟩
          by_cases hki : k = i
          · subst hki
            -- jo is the last writer of ℓ above i: BackwardStale marks jo
            apply hnotm
            refine .inr ⟨ℓ, hkw, by simp, by omega, ⟨c₀, hc₀, hwℓ⟩, ?_⟩
            intro k' hk1 hk2 hk'w
            exact hnowriter k' hk1 (by omega) hk'w
          · exact hnowriter k hk hki hkw
        exact hagreej ℓ hnold
      · -- outside cell jo's own writes: the witness store agrees by frame
        have := CellEval.frame hevalj ℓ hwℓ
        rw [this]
  · -- Rerun consistency at the new position
    constructor
    · rintro ℓ ⟨cc, hc, hcr⟩ ⟨cc', hc', hcw⟩
      rw [hcj] at hc hc'; cases hc; cases hc'
      rw [hcjr] at hcr; rw [hcjw] at hcw
      exact hrcj.noReadAndWrite ℓ ⟨c₀, hc₀, hcr⟩ ⟨c₀, hc₀, hcw⟩
    · rintro ℓ ⟨cc, hc, hcr⟩
      rw [hcj] at hc; cases hc
      rw [hcjr] at hcr
      obtain ⟨k, hk, ck, hck, hckw⟩ := hrcj.writeBeforeRead ℓ ⟨c₀, hc₀, hcr⟩
      have hki : k ≠ i := by
        intro hki; subst hki
        -- cell i wrote something jo reads: jo would be forward stale
        exact hnotm (.inl ⟨by omega, ℓ, .inl ⟨ck, hck, hckw⟩, .inl ⟨c₀, hc₀, hcr⟩⟩)
      obtain ⟨t, k', hk', hck'⟩ := old_to_new k hki ck hck
      exact ⟨k', by omega, { ck with tag := t }, hck', hckw⟩
    · rintro ℓ ⟨cc, hc, hcr⟩ ⟨k', hk', ck', hck', hckw⟩
      rw [hcj] at hc; cases hc
      rw [hcjr] at hcr
      obtain ⟨ko, hko, hko_ne, ck₀, hck₀, hckck₀, _⟩ := hget k' ck' hck'
      have hckw₀ : ck₀.writes ℓ := by
        have hh : ck'.writes = ck₀.writes := by rw [hckck₀]
        rw [hh] at hckw
        exact hckw
      exact hrcj.noReadBeforeWrite ℓ ⟨c₀, hc₀, hcr⟩ ⟨ko, by omega, ck₀, hck₀, hckw₀⟩
    · rintro ℓ ⟨cc, hc, hcw⟩ ⟨k', hk', ck', hck', hckr⟩
      rw [hcj] at hc; cases hc
      rw [hcjw] at hcw
      obtain ⟨ko, hko, hko_ne, ck₀, hck₀, hckck₀, _⟩ := hget k' ck' hck'
      have hckr₀ : ck₀.reads ℓ := by
        have hh : ck'.reads = ck₀.reads := by rw [hckck₀]
        rw [hh] at hckr
        exact hckr
      exact hrcj.noWriteAfterRead ℓ ⟨c₀, hc₀, hcw⟩ ⟨ko, by omega, ck₀, hck₀, hckr₀⟩

end Delete

/-! ## Theorem 2.3 (Preservation) -/

section Main
variable [CellEval Code Output L V]

/-- **Theorem 2.3 (Preservation).**  If `S · I` is well-formed and
`S · I ⟹op S' · I'`, then `S' · I'` is well-formed. -/
theorem preservation {nb nb' : Notebook Code Output L V} {op : Op Code}
    (hwf : WellFormed nb) (hstep : InstStep nb op nb') :
    WellFormed nb' := by
  induction hstep with
  | @run nb i ci o σ' r w cs' hcell heval hrc hlen hat hretag =>
    exact preservation_run (cs := nb.cells) (σ := nb.store) hwf hcell heval rfl
      hrc hlen hat hretag
  | @edit nb i ci c hcell =>
    exact preservation_edit (cs := nb.cells) (σ := nb.store) hwf hcell rfl
  | @insert nb i c hle =>
    exact preservation_insert (cs := nb.cells) (σ := nb.store) hwf hle rfl
      (fun _ h => h) (fun _ h => h)
  | @delete nb i ci cs'' hcell hlen hretag =>
    exact preservation_delete (cs := nb.cells) (σ := nb.store) hwf hcell hlen hretag
  | moveDown hlt hcell h1 h2 ih1 ih2 =>
    exact ih2 (ih1 hwf)
  | moveUp hlt hcell h1 h2 ih1 ih2 =>
    exact ih2 (ih1 hwf)

end Main

end FlowBook
