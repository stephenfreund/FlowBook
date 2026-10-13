/-
# Theorem 2.5 (Progress): Running Stale Cells Terminates

Theorem 2.5 of the supplement (proof in §5 of the supplement):

> Under Assumption (Determinism), in a well-formed notebook, repeatedly
> running the first stale cell terminates in a well-formed notebook in
> which either every cell is clean (and hence the notebook is output
> consistent), or the first stale cell is stuck.

## The determinism assumption

The paper states determinism in terms of read *sets*: rerunning a cell
when the values it read are unchanged reproduces its read set, write
set, and written values.  That set-level property is not enough for
termination: a cell whose read set is `{a, b}`, `{b, c}`, or `{c, a}`
depending on which of `a`, `b`, `c` holds a stale value satisfies it,
yet with three cells above it that write `a`, `b`, `c`, running the
first stale cell cycles forever (each run drops one location,
`BackwardStale` marks its writer, whose rerun restores it and re-marks
the cell, which now reads the next bad location).

The proof in the supplement uses the *order* of reads ("the first such
location `u` reads"), so the mechanization assumes determinism in its
sequential form (`Deterministic`): a cell reads its locations in a
sequence `rseq c σ`, and the next location it reads depends only on
the values of the locations it has already read (`rseq_take`).  Its
write set and written values depend only on the values it read
(`det`).  Outputs are unconstrained.  Every program that reads its
inputs one at a time and branches only on values already read satisfies
this; it implies the set-level property (with `locality`).

## Proof structure

Let `E` be the top-to-bottom execution of the current code (`TopPre`;
unique under determinism, `TopPre.unique`).

* `clean_prefix`: in a well-formed state, every cell `k` of the clean
  prefix is *settled* — its recorded read and write sets are those of
  `E` (`Settled`) — and the store agrees with `E`'s store `σ_m` on every
  location written above `m` and not at or below `m` (`Exposed`).  This
  is the output-consistency induction, so it holds in *every*
  well-formed state reached, and no invariant about values needs to be
  carried across runs.
* `run_analysis`: a run of the first stale cell `f` either reads exactly
  `E`'s reads and writes `E`'s writes (an `E`-run), or it first diverges
  from `E` at a location `p`, read by both, at which the store differs
  from `σ_f`; then `p ∈ W_f` and `p` is not in the new write set
  (`NoReadAndWrite`).
* A *phase* for cell `m` (`Phase`) lasts while the first stale cell is
  at or above `m`.  Cells above `m` stay settled, so their runs are
  `E`-runs that keep their write sets: they mark only cells below
  themselves, and the first stale cell moves down.  Each run of `m`
  that diverges from `E` at the `d`-th read leaves the first `d + 1`
  reads of `E` exposed, so the next run of `m` agrees with `E` on at
  least `d + 1` reads (`Phase.exposed`).  After an `E`-run of `m`, the
  next run of `m` is an `E`-run that keeps its write set.  The phase
  measure is therefore `(|rseq m| + 1 − κ, n − first stale)`, and the
  phase index `m` strictly increases between phases.

`progress_terminates` states termination as accessibility (no infinite
execution of the strategy, whatever outputs the cells produce).
`progress` gives a terminating execution ending in a well-formed state
that is all-clean or stuck; `progress_halted` shows that *every*
execution that cannot continue ends in such a state.
-/
import FlowBook.OutputConsistency

namespace FlowBook

variable {Code Output L V : Type}

/-! ## First stale cell -/

/-- `i` is the position of the first stale cell. -/
def FirstStale (cs : List (Cell Code Output L)) (i : Nat) : Prop :=
  IsStale cs i ∧ ∀ j, j < i → ¬ IsStale cs j

theorem IsClean.not_stale {cs : List (Cell Code Output L)} {j : Nat}
    (h : IsClean cs j) : ¬ IsStale cs j := by
  rintro ⟨c, hc, hst⟩
  obtain ⟨c', hc', hcl⟩ := h
  rw [hc] at hc'; cases hc'
  rw [hcl] at hst
  cases hst

theorem FirstStale.lt_length {cs : List (Cell Code Output L)} {i : Nat}
    (h : FirstStale cs i) : i < cs.length := by
  obtain ⟨c, hc, _⟩ := h.1
  exact lt_length_of_getElem?_eq_some hc

theorem FirstStale.unique {cs : List (Cell Code Output L)} {i j : Nat}
    (hi : FirstStale cs i) (hj : FirstStale cs j) : i = j := by
  rcases Nat.lt_trichotomy i j with h | h | h
  · exact absurd hi.1 (hj.2 i h)
  · exact h
  · exact absurd hj.1 (hi.2 j h)

theorem FirstStale.clean_before {cs : List (Cell Code Output L)} {i : Nat}
    (h : FirstStale cs i) : ∀ k, k < i → IsClean cs k := by
  intro k hk
  obtain ⟨c, hc⟩ := exists_getElem?_eq_some (Nat.lt_trans hk h.lt_length)
  rcases Tag.clean_or_stale c.tag with ht | ht
  · exact ⟨c, hc, ht⟩
  · exact absurd ⟨c, hc, ht⟩ (h.2 k hk)

/-- A minimal witness of a nonempty predicate on `Nat` (classical). -/
theorem exists_min {P : Nat → Prop} (h : ∃ n, P n) :
    ∃ n, P n ∧ ∀ m, m < n → ¬ P m := by
  classical
  obtain ⟨n, hn⟩ := h
  induction n using Nat.strongRecOn with
  | ind n ih =>
    by_cases hmin : ∀ m, m < n → ¬ P m
    · exact ⟨n, hn, hmin⟩
    · obtain ⟨m, hm⟩ := Classical.not_forall.mp hmin
      have hmn : m < n := by
        by_cases h2 : m < n
        · exact h2
        · exact absurd (fun hlt => absurd hlt h2) hm
      have hPm : P m := by
        by_cases h3 : P m
        · exact h3
        · exact absurd (fun _ => h3) hm
      exact ih m hmn hPm

/-- A state is all-clean or has a first stale cell (classical). -/
theorem allClean_or_firstStale (cs : List (Cell Code Output L)) :
    AllClean cs ∨ ∃ i, FirstStale cs i := by
  classical
  by_cases hac : AllClean cs
  · exact .inl hac
  · right
    have hex : ∃ i, IsStale cs i := by
      refine Classical.byContradiction fun hno => hac ?_
      intro i c hc
      rcases Tag.clean_or_stale c.tag with h | h
      · exact h
      · exact absurd ⟨i, ⟨c, hc, h⟩⟩ hno
    obtain ⟨i, h1, h2⟩ := exists_min hex
    exact ⟨i, h1, h2⟩

theorem AllClean.not_firstStale {cs : List (Cell Code Output L)} (h : AllClean cs)
    {i : Nat} : ¬ FirstStale cs i := by
  rintro ⟨⟨c, hc, hst⟩, _⟩
  rw [h i c hc] at hst
  cases hst

/-! ## The determinism assumption -/

/-- **Assumption (Determinism)**, in sequential form.  A run of code `c`
from store `σ` reads the locations of the list `rseq c σ`, in order.

* `reads_iff`: the read set of a run is the set of locations in
  `rseq c σ`.
* `rseq_take`: the next location read depends only on the values of the
  locations already read — if `τ` agrees with `σ` on the first `k`
  locations of `rseq c σ`, then the first `k + 1` locations of
  `rseq c τ` and `rseq c σ` coincide.
* `det`: the write set and the written values depend only on the values
  read.  The output may differ. -/
class Deterministic (Code Output L V : Type) [CellEval Code Output L V] where
  rseq : Code → Store L V → List L
  reads_iff : ∀ {c : Code} {σ : Store L V} {o : Output} {σ' : Store L V}
      {r w : L → Prop},
    Eval c σ o σ' r w → ∀ ℓ, r ℓ ↔ ℓ ∈ rseq c σ
  rseq_take : ∀ {c : Code} {σ τ : Store L V} {k : Nat},
    (∀ ℓ, ℓ ∈ (rseq c σ).take k → τ ℓ = σ ℓ) →
    (rseq c τ).take (k + 1) = (rseq c σ).take (k + 1)
  det : ∀ {c : Code} {σ : Store L V} {o : Output} {σ' : Store L V} {r w : L → Prop}
      {τ : Store L V} {o₂ : Output} {τ' : Store L V} {r₂ w₂ : L → Prop},
    Eval c σ o σ' r w → Eval c τ o₂ τ' r₂ w₂ → (∀ ℓ, r ℓ → τ ℓ = σ ℓ) →
    (∀ ℓ, w₂ ℓ ↔ w ℓ) ∧ ∀ ℓ, w ℓ → τ' ℓ = σ' ℓ

export Deterministic (rseq)

section Seq
variable [CellEval Code Output L V] [Deterministic Code Output L V]

/-- Agreeing on every location read gives the same read sequence. -/
theorem rseq_eq_of_agree {c : Code} {σ τ : Store L V}
    (h : ∀ ℓ, ℓ ∈ rseq (Output := Output) c σ → τ ℓ = σ ℓ) :
    rseq (Output := Output) c τ = rseq (Output := Output) c σ := by
  have h1 := Deterministic.rseq_take (Output := Output) (c := c) (σ := σ) (τ := τ)
    (k := (rseq (Output := Output) c σ).length)
    (by intro ℓ hℓ; rw [List.take_length] at hℓ; exact h ℓ hℓ)
  rw [List.take_of_length_le (Nat.le_succ _)] at h1
  have hlen : (rseq (Output := Output) c τ).length ≤ (rseq (Output := Output) c σ).length := by
    have := congrArg List.length h1
    rw [List.length_take] at this
    omega
  rwa [List.take_of_length_le (by omega)] at h1

theorem take_succ_append {pre rest : List L} {p : L} :
    (pre ++ p :: rest).take (pre.length + 1) = pre ++ [p] := by
  rw [List.take_append, List.take_of_length_le (by omega),
    show pre.length + 1 - pre.length = 1 by omega]
  simp

/-- Agreeing on the reads before `p` means `p` is read next. -/
theorem rseq_append_of_agree {c : Code} {σ τ : Store L V} {pre : List L} {p : L}
    {rest : List L} (hL : rseq (Output := Output) c σ = pre ++ p :: rest)
    (h : ∀ ℓ, ℓ ∈ pre → τ ℓ = σ ℓ) :
    ∃ rest', rseq (Output := Output) c τ = pre ++ p :: rest' := by
  have h1 := Deterministic.rseq_take (Output := Output) (c := c) (σ := σ) (τ := τ)
    (k := pre.length) (by intro ℓ hℓ; rw [hL, List.take_left' rfl] at hℓ; exact h ℓ hℓ)
  rw [hL, take_succ_append] at h1
  refine ⟨(rseq (Output := Output) c τ).drop (pre.length + 1), ?_⟩
  rw [← List.take_append_drop (pre.length + 1) (rseq (Output := Output) c τ), h1]
  simp

end Seq

/-- Splitting a list at the first location where two stores differ. -/
theorem split_first_disagree (l : List L) (f g : Store L V) :
    (∀ ℓ, ℓ ∈ l → f ℓ = g ℓ) ∨
    ∃ pre p rest, l = pre ++ p :: rest ∧ (∀ ℓ, ℓ ∈ pre → f ℓ = g ℓ) ∧ f p ≠ g p := by
  induction l with
  | nil => exact .inl (by simp)
  | cons a t ih =>
    by_cases ha : f a = g a
    · rcases ih with h | ⟨pre, p, rest, ht, hpre, hp⟩
      · left
        intro ℓ hℓ
        rcases List.mem_cons.mp hℓ with rfl | h'
        · exact ha
        · exact h ℓ h'
      · right
        refine ⟨a :: pre, p, rest, by simp [ht], ?_, hp⟩
        intro ℓ hℓ
        rcases List.mem_cons.mp hℓ with rfl | h'
        · exact ha
        · exact hpre ℓ h'
    · exact .inr ⟨[], a, t, rfl, by simp, ha⟩

theorem mem_take_of_lt {pre rest : List L} {p : L} {κ : Nat} (h : pre.length < κ) :
    p ∈ (pre ++ p :: rest).take κ := by
  obtain ⟨j, rfl⟩ : ∃ j, κ = pre.length + 1 + j := ⟨κ - pre.length - 1, by omega⟩
  rw [List.take_append, show pre.length + 1 + j - pre.length = j + 1 by omega]
  simp

/-! ## The top-to-bottom execution `E` and settled cells -/

section TopDown
variable [CellEval Code Output L V]

/-- `TopPre cs k σ`: executing the code of the first `k` cells top to
bottom from the empty store produces the store `σ` (the store `σ^k` that
cell `k` reads in `E`). -/
inductive TopPre (cs : List (Cell Code Output L)) : Nat → Store L V → Prop where
  | zero : TopPre cs 0 Store.empty
  | succ {k : Nat} {σ : Store L V} {c : Cell Code Output L} {o : Output}
      {σ' : Store L V} {r w : L → Prop} :
      TopPre cs k σ → cs[k]? = some c → Eval c.code σ o σ' r w → TopPre cs (k + 1) σ'

/-- Two cell lists with the same code at every position. -/
def SameCodes (cs cs' : List (Cell Code Output L)) : Prop :=
  ∀ j : Nat, (cs'[j]?).map Cell.code = (cs[j]?).map Cell.code

/-- `E` depends only on the code of the cells. -/
theorem TopPre.of_sameCodes {cs cs' : List (Cell Code Output L)} (hs : SameCodes cs cs')
    {k : Nat} {σ : Store L V} (h : TopPre cs k σ) : TopPre cs' k σ := by
  induction h with
  | zero => exact .zero
  | @succ k σ c o σ' r w _ hc heval ih =>
    have hk := hs k
    rw [hc] at hk
    cases hc' : cs'[k]? with
    | none => rw [hc'] at hk; cases hk
    | some c' =>
      rw [hc'] at hk
      have hcode : c'.code = c.code := Option.some.inj hk
      exact .succ ih hc' (by rw [hcode]; exact heval)

/-- Cell `k` is *settled*: its recorded read and write sets are those of
its run in `E`, and they are disjoint. -/
def Settled (V : Type) [CellEval Code Output L V] (cs : List (Cell Code Output L))
    (k : Nat) : Prop :=
  ∃ c, cs[k]? = some c ∧ ∃ (σ : Store L V) (o : Output) (σ' : Store L V),
    TopPre cs k σ ∧ Eval c.code σ o σ' c.reads c.writes ∧
    ∀ ℓ, c.reads ℓ → ¬ c.writes ℓ

theorem Settled.transfer {cs cs' : List (Cell Code Output L)} {k : Nat}
    (hs : Settled V cs k) (hcodes : SameCodes cs cs')
    (hcell : ∀ c, cs[k]? = some c → ∃ c', cs'[k]? = some c' ∧
      c'.code = c.code ∧ c'.reads = c.reads ∧ c'.writes = c.writes) :
    Settled V cs' k := by
  obtain ⟨c, hc, σ, o, σ', htop, heval, hdisj⟩ := hs
  obtain ⟨c', hc', h1, h2, h3⟩ := hcell c hc
  exact ⟨c', hc', σ, o, σ', htop.of_sameCodes hcodes,
    by rw [h1, h2, h3]; exact heval, by rw [h2, h3]; exact hdisj⟩

/-- `ℓ ∈ (⋃ W_{1..m-1}) \ (⋃ W_{m..n})`: written above `m`, and not at
or below `m`. -/
def Exposed (cs : List (Cell Code Output L)) (m : Nat) (ℓ : L) : Prop :=
  WritesAbove cs m ℓ ∧ ¬ WritesAt cs m ℓ ∧ ¬ WritesBelow cs m ℓ

theorem RWEquiv.exposed {cs₁ cs₂ : List (Cell Code Output L)} (h : RWEquiv cs₁ cs₂)
    (m : Nat) (ℓ : L) : Exposed cs₁ m ℓ ↔ Exposed cs₂ m ℓ := by
  unfold Exposed
  rw [h.writesAbove, h.writesBelow, h.2 m ℓ]

theorem RWEquiv.trans {cs₁ cs₂ cs₃ : List (Cell Code Output L)} (h₁ : RWEquiv cs₁ cs₂)
    (h₂ : RWEquiv cs₂ cs₃) : RWEquiv cs₁ cs₃ :=
  ⟨fun j ℓ => (h₁.1 j ℓ).trans (h₂.1 j ℓ), fun j ℓ => (h₁.2 j ℓ).trans (h₂.2 j ℓ)⟩

/-- **The clean prefix agrees with `E`.**  If cells `0..m-1` are clean in
a well-formed state, then they are settled, and the store agrees with
`E`'s store `σ^m` on the locations written above `m` and not at or
below `m`.  This is the induction of Theorem 2.4 (output consistency),
restricted to a prefix. -/
theorem clean_prefix {nb : Notebook Code Output L V} (hwf : WellFormed nb) :
    ∀ m, m ≤ nb.cells.length → (∀ k, k < m → IsClean nb.cells k) →
    ∃ σ, TopPre nb.cells m σ ∧ (∀ k, k < m → Settled V nb.cells k) ∧
      ∀ ℓ, Exposed nb.cells m ℓ → nb.store ℓ = σ ℓ := by
  obtain ⟨cs, σ0⟩ := nb
  replace hwf : ∀ i, IsClean cs i → Witnessed cs σ0 i ∧ RerunConsistent cs i := hwf
  intro m
  induction m with
  | zero =>
    intro _ _
    refine ⟨Store.empty, .zero, fun k hk => absurd hk (Nat.not_lt_zero _), ?_⟩
    rintro ℓ ⟨⟨j, hj, _⟩, _, _⟩
    omega
  | succ k ih =>
    intro hk1 hclean
    dsimp only at hk1 hclean ih ⊢
    obtain ⟨τ, htop, hset, hagree⟩ := ih (by omega) (fun j hj => hclean j (by omega))
    have hkcs : k < cs.length := by omega
    obtain ⟨c, hc⟩ := exists_getElem?_eq_some hkcs
    obtain ⟨hwit, hrc⟩ := hwf k (hclean k (by omega))
    obtain ⟨c', hc', o, σstar, _, heval, hagreeStar⟩ := hwit
    rw [hc] at hc'; cases hc'
    have hreads : ∀ ℓ, c.reads ℓ → τ ℓ = σ0 ℓ := by
      intro ℓ hr
      have h1 : WritesAbove cs k ℓ := hrc.writeBeforeRead ℓ ⟨c, hc, hr⟩
      have h2 : ¬ WritesAt cs k ℓ := fun hw => hrc.noReadAndWrite ℓ ⟨c, hc, hr⟩ hw
      have h3 : ¬ WritesBelow cs k ℓ := fun hw =>
        hrc.noReadBeforeWrite ℓ ⟨c, hc, hr⟩ hw
      exact (hagree ℓ ⟨h1, h2, h3⟩).symm
    obtain ⟨τ', heval', hτw, hτnw⟩ := CellEval.locality heval hreads
    refine ⟨τ', .succ htop hc heval', ?_, ?_⟩
    · intro j hj
      by_cases hjk : j = k
      · subst hjk
        exact ⟨c, hc, τ, o, τ', htop, heval',
          fun ℓ hr hw => hrc.noReadAndWrite ℓ ⟨c, hc, hr⟩ ⟨c, hc, hw⟩⟩
      · exact hset j (by omega)
    · rintro ℓ ⟨habove, hnotat, hnotbelow⟩
      have hnb_k : ¬ WritesBelow cs k ℓ := by
        rintro ⟨j, hj, hw⟩
        by_cases hjk : j = k + 1
        · subst hjk; exact hnotat hw
        · exact hnotbelow ⟨j, by omega, hw⟩
      by_cases hwℓ : c.writes ℓ
      · rw [hτw ℓ hwℓ]
        exact hagreeStar ℓ hnb_k
      · rw [hτnw ℓ hwℓ]
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
        exact hagree ℓ ⟨habove_k, hnotat_k, hnb_k⟩

/-- Under determinism, `E` is unique. -/
theorem TopPre.unique [Deterministic Code Output L V] {cs : List (Cell Code Output L)}
    {k : Nat} {σ τ : Store L V} (h1 : TopPre cs k σ) (h2 : TopPre cs k τ) : σ = τ := by
  induction h1 generalizing τ with
  | zero => cases h2; rfl
  | @succ k σ c o σ' r w _ hc heval ih =>
    cases h2 with
    | @succ _ τ₀ c' o' τ' r' w' hpre' hc' heval' =>
      rw [hc] at hc'; cases hc'
      have := ih hpre'; subst this
      obtain ⟨hw, hv⟩ := Deterministic.det heval heval' (fun _ _ => rfl)
      funext ℓ
      by_cases hwℓ : w ℓ
      · exact (hv ℓ hwℓ).symm
      · rw [CellEval.frame heval ℓ hwℓ,
          CellEval.frame heval' ℓ (fun h => hwℓ ((hw ℓ).mp h))]

end TopDown

/-! ## Facts about a single `[Inst-Run]` step -/

section RunFacts
variable [CellEval Code Output L V]

/-- Any `[Inst-Run]` step preserves the number of cells. -/
theorem instStep_run_length {nb nb' : Notebook Code Output L V} {i : Nat}
    (h : InstStep nb (.run i) nb') : nb'.cells.length = nb.cells.length := by
  cases h with
  | run hcell heval hrc hlen hat hretag => exact hlen

theorem writesAbove_set {cs : List (Cell Code Output L)} {i : Nat} {c : Cell Code Output L}
    {ℓ : L} : WritesAbove (cs.set i c) i ℓ ↔ WritesAbove cs i ℓ := by
  constructor <;> rintro ⟨j, hj, cc, hcc, hw⟩
  · rw [List.getElem?_set_ne (by omega)] at hcc; exact ⟨j, hj, cc, hcc, hw⟩
  · exact ⟨j, hj, cc, by rw [List.getElem?_set_ne (by omega)]; exact hcc, hw⟩

theorem writesBelow_set {cs : List (Cell Code Output L)} {i : Nat} {c : Cell Code Output L}
    {ℓ : L} : WritesBelow (cs.set i c) i ℓ ↔ WritesBelow cs i ℓ := by
  constructor <;> rintro ⟨j, hj, cc, hcc, hw⟩
  · rw [List.getElem?_set_ne (by omega)] at hcc; exact ⟨j, hj, cc, hcc, hw⟩
  · exact ⟨j, hj, cc, by rw [List.getElem?_set_ne (by omega)]; exact hcc, hw⟩

/-- The tables after a run are those of `cs` with cell `i` replaced. -/
theorem run_rwEquiv {cs cs' : List (Cell Code Output L)} {i : Nat}
    {ci' : Cell Code Output L} {w : L → Prop}
    (hi : i < cs.length) (hlen : cs'.length = cs.length) (hat : cs'[i]? = some ci')
    (hretag : RetagSpec cs cs' i w) : RWEquiv cs' (cs.set i ci') := by
  have key : ∀ (P : Cell Code Output L → L → Prop) (k : Nat) (ℓ : L),
      (∀ c t, P { c with tag := t } ℓ ↔ P c ℓ) →
      ((∃ c, cs'[k]? = some c ∧ P c ℓ) ↔ (∃ c, (cs.set i ci')[k]? = some c ∧ P c ℓ)) := by
    intro P k ℓ hP
    by_cases hki : k = i
    · subst hki
      rw [hat, List.getElem?_set_self hi]
    · constructor <;> rintro ⟨c, hc, hcr⟩
      · have hklt : k < cs.length := by
          have := lt_length_of_getElem?_eq_some hc; omega
        obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hklt
        obtain ⟨t', hct', _⟩ := hretag k hki c₀ hc₀
        rw [hct'] at hc; cases hc
        exact ⟨c₀, by rw [List.getElem?_set_ne (by omega)]; exact hc₀, (hP c₀ t').mp hcr⟩
      · rw [List.getElem?_set_ne (by omega)] at hc
        obtain ⟨t', hct', _⟩ := hretag k hki c hc
        exact ⟨{ c with tag := t' }, hct', (hP c t').mpr hcr⟩
  exact ⟨fun k ℓ => key (fun c ℓ => c.reads ℓ) k ℓ (fun _ _ => Iff.rfl),
    fun k ℓ => key (fun c ℓ => c.writes ℓ) k ℓ (fun _ _ => Iff.rfl)⟩

/-- After a run, cell `i` keeps its code, and every other cell keeps its
code, output, and read/write sets. -/
theorem run_cell_shape {cs cs' : List (Cell Code Output L)} {i : Nat}
    {ci ci' : Cell Code Output L} {w : L → Prop}
    (hcell : cs[i]? = some ci) (hat : cs'[i]? = some ci') (hcode : ci'.code = ci.code)
    (hretag : RetagSpec cs cs' i w) :
    ∀ k c, cs[k]? = some c → ∃ c', cs'[k]? = some c' ∧ c'.code = c.code ∧
      (k ≠ i → c'.reads = c.reads ∧ c'.writes = c.writes) := by
  intro k c hc
  by_cases hki : k = i
  · subst hki
    rw [hcell] at hc; cases hc
    exact ⟨ci', hat, hcode, fun h => absurd rfl h⟩
  · obtain ⟨t, hct, _⟩ := hretag k hki c hc
    exact ⟨{ c with tag := t }, hct, rfl, fun _ => ⟨rfl, rfl⟩⟩

theorem run_sameCodes {cs cs' : List (Cell Code Output L)} {i : Nat}
    {ci ci' : Cell Code Output L} {w : L → Prop}
    (hcell : cs[i]? = some ci) (hat : cs'[i]? = some ci') (hcode : ci'.code = ci.code)
    (hlen : cs'.length = cs.length)
    (hretag : RetagSpec cs cs' i w) : SameCodes cs cs' := by
  intro k
  rcases Nat.lt_or_ge k cs.length with hk | hk
  · obtain ⟨c, hc⟩ := exists_getElem?_eq_some hk
    obtain ⟨c', hc', h1, _⟩ := run_cell_shape hcell hat hcode hretag k c hc
    rw [hc, hc']
    simp [h1]
  · rw [List.getElem?_eq_none hk, List.getElem?_eq_none (by omega)]

/-- A run of the first stale cell `i` whose new write set contains its
old one marks nothing at or above `i` (`BackwardStale` needs a dropped
write, and `ForwardStale` marks only cells below `i`). -/
theorem run_prefix_clean {cs cs' : List (Cell Code Output L)} {i : Nat}
    {ci ci' : Cell Code Output L} {w : L → Prop}
    (hfs : FirstStale cs i) (hcell : cs[i]? = some ci) (hat : cs'[i]? = some ci')
    (hclean : ci'.tag = .clean) (hretag : RetagSpec cs cs' i w)
    (hw : ∀ ℓ, ci.writes ℓ → w ℓ) : ∀ j, j ≤ i → IsClean cs' j := by
  intro j hj
  by_cases hji : j = i
  · subst hji; exact ⟨ci', hat, hclean⟩
  · have hjlt : j < i := by omega
    obtain ⟨cj, hcj⟩ := exists_getElem?_eq_some (Nat.lt_trans hjlt hfs.lt_length)
    obtain ⟨t, hct, hstale⟩ := hretag j hji cj hcj
    have hcjclean : cj.tag = Tag.clean := by
      rcases Tag.clean_or_stale cj.tag with h | h
      · exact h
      · exact absurd ⟨cj, hcj, h⟩ (hfs.2 j hjlt)
    have hnot : ¬ (Marked cs i w j ∨ cj.tag = Tag.stale) := by
      rintro ((⟨hij, _⟩ | ⟨ℓ, ⟨cc, hcc, hccw⟩, hnw, _⟩) | hs)
      · omega
      · rw [hcell] at hcc; cases hcc; exact hnw (hw ℓ hccw)
      · rw [hcjclean] at hs; cases hs
    have ht : t = Tag.clean := by
      rcases Tag.clean_or_stale t with h | h
      · exact h
      · exact absurd (hstale.mp h) hnot
    exact ⟨{ cj with tag := t }, hct, ht⟩

/-- A location read by an accepted run of cell `i` is exposed at `i`
afterwards: written above `i` (`WriteBeforeRead`), not by `i`
(`NoReadAndWrite`), and not below `i` (`NoReadBeforeWrite`). -/
theorem exposed_of_run {cs cs' : List (Cell Code Output L)} {i : Nat}
    {ci' : Cell Code Output L} {ℓ : L}
    (hi : i < cs.length) (hequiv : RWEquiv cs' (cs.set i ci'))
    (hrc : RerunConsistent (cs.set i ci') i) (hr : ci'.reads ℓ) : Exposed cs' i ℓ := by
  have hR : ReadsAt (cs.set i ci') i ℓ := ⟨ci', by simp [List.getElem?_set_self hi], hr⟩
  exact (hequiv.exposed i ℓ).mpr
    ⟨hrc.writeBeforeRead ℓ hR, hrc.noReadAndWrite ℓ hR, hrc.noReadBeforeWrite ℓ hR⟩

end RunFacts

/-! ## A run of the first stale cell, compared with `E` -/

section Analysis
variable [CellEval Code Output L V] [Deterministic Code Output L V]

/-- **Run analysis.**  Let the store agree with `E`'s store `σ_i` on the
locations exposed at `i` (`clean_prefix`).  An accepted run of cell `i`
either

1. reads exactly the locations `E` reads, and some run from `σ_i` has
   the same read and write sets (it is an `E`-run); or
2. diverges from `E`: `E`'s read sequence is `pre ++ p :: rest`, the
   run reads `pre` and `p`, and the store differs from `σ_i` at `p`.
   Then `p` is not exposed, so `p ∈ W_i`: cell `i` itself wrote the
   value it now reads. -/
theorem run_analysis {cs : List (Cell Code Output L)} {σ0 σf : Store L V} {i : Nat}
    {ci : Cell Code Output L} {o : Output} {σ' : Store L V} {r w : L → Prop}
    (hcell : cs[i]? = some ci) (heval : Eval ci.code σ0 o σ' r w)
    (hrc : RerunConsistent
      (cs.set i { ci with out := some o, tag := .clean, reads := r, writes := w }) i)
    (hagree : ∀ ℓ, Exposed cs i ℓ → σ0 ℓ = σf ℓ) :
    ((∀ ℓ, r ℓ ↔ ℓ ∈ rseq (Output := Output) ci.code σf) ∧
      ∃ (o' : Output) (σ'' : Store L V), Eval ci.code σf o' σ'' r w) ∨
    ∃ pre p rest, rseq (Output := Output) ci.code σf = pre ++ p :: rest ∧ σ0 p ≠ σf p ∧
      (∀ ℓ, ℓ ∈ pre ++ [p] → r ℓ) ∧ ci.writes p := by
  have hi := lt_length_of_getElem?_eq_some hcell
  have hR : ∀ ℓ, r ℓ ↔ ℓ ∈ rseq (Output := Output) ci.code σ0 :=
    Deterministic.reads_iff heval
  rcases split_first_disagree (rseq (Output := Output) ci.code σf) σ0 σf with
    hall | ⟨pre, p, rest, hL, hpre, hp⟩
  · left
    have heq : rseq (Output := Output) ci.code σ0 = rseq (Output := Output) ci.code σf :=
      rseq_eq_of_agree hall
    refine ⟨fun ℓ => by rw [hR, heq], ?_⟩
    obtain ⟨τ', hτ, _, _⟩ := CellEval.locality heval (τ := σf)
      (fun ℓ hr => (hall ℓ (by rw [← heq, ← hR]; exact hr)).symm)
    exact ⟨o, τ', hτ⟩
  · right
    obtain ⟨rest', hL'⟩ := rseq_append_of_agree hL hpre
    have hrd : ∀ ℓ, ℓ ∈ pre ++ [p] → r ℓ := by
      intro ℓ hℓ
      rw [hR, hL']
      simp only [List.mem_append, List.mem_singleton] at hℓ
      simp only [List.mem_append, List.mem_cons]
      rcases hℓ with h | h
      · exact .inl h
      · exact .inr (.inl h)
    refine ⟨pre, p, rest, hL, hp, hrd, ?_⟩
    refine Classical.byContradiction fun hnw => hp (hagree p ?_)
    have hrp : ReadsAt
        (cs.set i { ci with out := some o, tag := .clean, reads := r, writes := w }) i p :=
      ⟨_, List.getElem?_set_self hi, hrd p (by simp)⟩
    refine ⟨writesAbove_set.mp (hrc.writeBeforeRead p hrp), ?_,
      fun hb => hrc.noReadBeforeWrite p hrp (writesBelow_set.mpr hb)⟩
    rintro ⟨c, hc, hcw⟩
    rw [hcell] at hc; cases hc
    exact hnw hcw

end Analysis

/-! ## The strategy and its termination -/

section Strategy
variable [CellEval Code Output L V]

/-- One step of the strategy: run the first stale cell (any accepted
`[Inst-Run]` step; the cell's output is unconstrained). -/
def StrategyStep (nb nb' : Notebook Code Output L V) : Prop :=
  ∃ i, FirstStale nb.cells i ∧ InstStep nb (.run i) nb'

/-- The converse of `StrategyStep`: `Acc StrategyRel nb` says that every
execution of the strategy from `nb` is finite. -/
def StrategyRel (nb' nb : Notebook Code Output L V) : Prop := StrategyStep nb nb'

/-- Executions of the strategy. -/
inductive StrategyStar : Notebook Code Output L V → Notebook Code Output L V → Prop where
  | refl (nb : Notebook Code Output L V) : StrategyStar nb nb
  | step {nb nb₁ nb' : Notebook Code Output L V} :
      StrategyStep nb nb₁ → StrategyStar nb₁ nb' → StrategyStar nb nb'

/-- The first stale cell is *stuck*: no `[Inst-Run]` step of it exists,
because its run fails a rerun-consistency check or raises an exception. -/
def Stuck (nb : Notebook Code Output L V) : Prop :=
  ∃ i, FirstStale nb.cells i ∧ ∀ nb', ¬ InstStep nb (.run i) nb'

theorem acc_of_no_firstStale {nb : Notebook Code Output L V}
    (h : ∀ i, ¬ FirstStale nb.cells i) : Acc StrategyRel nb :=
  Acc.intro nb fun _ ⟨i, hfs, _⟩ => absurd hfs (h i)

variable [Deterministic Code Output L V]

/-- The invariant of a *phase* for cell `m`, with `E`'s store `σm` before
`m` and the code `c` of `m`.  Let `Lm = rseq c σm` be `E`'s read sequence
for `m`.

* the first stale cell, if any, is at or above `m`;
* every cell above `m` is settled;
* the first `κ` reads of `Lm` are exposed at `m`, so the next run of `m`
  agrees with `E` on at least `κ` reads;
* `κ = |Lm| + 1` records that `m`'s recorded read and write sets are
  those of `E`, so the next run of `m` keeps its write set. -/
structure Phase (n m κ : Nat) (σm : Store L V) (c : Code)
    (nb : Notebook Code Output L V) : Prop where
  len : nb.cells.length = n
  wf : WellFormed nb
  lt : m < n
  first_le : ∀ i, FirstStale nb.cells i → i ≤ m
  settled : ∀ k, k < m → Settled V nb.cells k
  top : TopPre nb.cells m σm
  cell : ∃ cm, nb.cells[m]? = some cm ∧ cm.code = c ∧
    ((rseq (Output := Output) c σm).length < κ →
      ∃ (o : Output) (σ' : Store L V), Eval c σm o σ' cm.reads cm.writes)
  bound : κ ≤ (rseq (Output := Output) c σm).length + 1
  exposed : ∀ ℓ, ℓ ∈ (rseq (Output := Output) c σm).take κ → Exposed nb.cells m ℓ

/-- **Termination of a phase.**  From a state in a phase for `m`, every
execution of the strategy is finite, given that every execution is
finite from the well-formed states whose first stale cell is below `m`
(`outer`).  By induction on the lexicographic measure
`(|Lm| + 1 − κ, n − first stale)`. -/
theorem phase_acc {n m : Nat} {σm : Store L V} {c : Code}
    (outer : ∀ nb : Notebook Code Output L V, nb.cells.length = n → WellFormed nb →
      (∀ f, FirstStale nb.cells f → n - f < n - m) → Acc StrategyRel nb) :
    ∀ K κ (nb : Notebook Code Output L V),
      (rseq (Output := Output) c σm).length + 1 - κ ≤ K → Phase n m κ σm c nb →
      Acc StrategyRel nb := by
  intro K
  induction K using Nat.strongRecOn with
  | ind K ihK =>
  suffices H : ∀ F κ (nb : Notebook Code Output L V),
      (rseq (Output := Output) c σm).length + 1 - κ ≤ K →
      (∀ f, FirstStale nb.cells f → n - f ≤ F) → Phase n m κ σm c nb →
      Acc StrategyRel nb by
    intro κ nb hK hP
    exact H n κ nb hK (fun f _ => Nat.sub_le n f) hP
  intro F
  induction F using Nat.strongRecOn with
  | ind F ihF =>
  intro κ nb hK hF hP
  refine Acc.intro nb ?_
  rintro nb' ⟨f, hfs, hstep⟩
  have hwf' : WellFormed nb' := preservation hP.wf hstep
  have hlen' : nb'.cells.length = n := (instStep_run_length hstep).trans hP.len
  have hfm : f ≤ m := hP.first_le f hfs
  have hfn : f < n := hP.len ▸ hfs.lt_length
  -- the store agrees with `E` on the locations exposed at `f`
  obtain ⟨σf, htopf, _, hagree⟩ :=
    clean_prefix hP.wf f (Nat.le_of_lt hfs.lt_length) hfs.clean_before
  -- leaving the phase: no stale cell at or above `m`
  have exit : (∀ i, FirstStale nb'.cells i → m < i) → Acc StrategyRel nb' := fun h =>
    outer nb' hlen' hwf' (fun i hi => by
      have h1 := h i hi
      have h2 := hi.lt_length
      omega)
  cases hstep with
  | @run _ _ ci o σ' r w cs' hcell heval hrc hlen hat hretag =>
  have hi : f < nb.cells.length := lt_length_of_getElem?_eq_some hcell
  have hshape := run_cell_shape hcell hat rfl hretag
  have hcodes : SameCodes nb.cells cs' := run_sameCodes hcell hat rfl hlen hretag
  have hequiv := run_rwEquiv hi hlen hat hretag
  have htop' : TopPre cs' m σm := hP.top.of_sameCodes hcodes
  -- cells other than `f` stay settled
  have hsettled_ne : ∀ k, k < m → k ≠ f → Settled V cs' k := fun k hk hkf =>
    (hP.settled k hk).transfer hcodes (fun c hc => by
      obtain ⟨c', hc', h1, h2⟩ := hshape k c hc
      exact ⟨c', hc', h1, (h2 hkf).1, (h2 hkf).2⟩)
  rcases Nat.lt_or_eq_of_le hfm with hlt | rfl
  · -- `f < m`: `f` is settled, so its run is an `E`-run that keeps its
    -- write set; the first stale cell moves down.
    obtain ⟨cf, hcf, σs, os, σs', htops, hevals, hdisj⟩ := hP.settled f hlt
    rw [hcell] at hcf; cases hcf
    have hσ : σs = σf := htops.unique htopf
    subst hσ
    rcases run_analysis hcell heval hrc hagree with
      ⟨hrL, o', σ'', hevalE⟩ | ⟨pre, p, rest, hL, _, _, hwp⟩
    · obtain ⟨hw, _⟩ := Deterministic.det hevals hevalE (fun _ _ => rfl)
      have hrr : r = ci.reads := funext fun ℓ =>
        propext ((hrL ℓ).trans (Deterministic.reads_iff hevals ℓ).symm)
      have hww : w = ci.writes := funext fun ℓ => propext (hw ℓ)
      have hpre := run_prefix_clean hfs hcell hat rfl hretag (fun ℓ h => (hw ℓ).mpr h)
      have hequiv' : RWEquiv cs' nb.cells := by
        refine hequiv.trans ⟨fun k ℓ => ?_, fun k ℓ => ?_⟩
        · by_cases hkf : k = f
          · subst hkf
            constructor <;> rintro ⟨cc, hcc, h⟩
            · rw [List.getElem?_set_self hi] at hcc; cases hcc
              exact ⟨ci, hcell, by rw [← hrr]; exact h⟩
            · rw [hcell] at hcc; cases hcc
              exact ⟨_, List.getElem?_set_self hi, by show r ℓ; rw [hrr]; exact h⟩
          · constructor <;> rintro ⟨cc, hcc, h⟩
            · rw [List.getElem?_set_ne (by omega)] at hcc; exact ⟨cc, hcc, h⟩
            · exact ⟨cc, by rw [List.getElem?_set_ne (by omega)]; exact hcc, h⟩
        · by_cases hkf : k = f
          · subst hkf
            constructor <;> rintro ⟨cc, hcc, h⟩
            · rw [List.getElem?_set_self hi] at hcc; cases hcc
              exact ⟨ci, hcell, by rw [← hww]; exact h⟩
            · rw [hcell] at hcc; cases hcc
              exact ⟨_, List.getElem?_set_self hi, by show w ℓ; rw [hww]; exact h⟩
          · constructor <;> rintro ⟨cc, hcc, h⟩
            · rw [List.getElem?_set_ne (by omega)] at hcc; exact ⟨cc, hcc, h⟩
            · exact ⟨cc, by rw [List.getElem?_set_ne (by omega)]; exact hcc, h⟩
      by_cases hin : ∃ i, FirstStale cs' i ∧ i ≤ m
      · obtain ⟨f', hfs', hf'm⟩ := hin
        have hff' : f < f' := by
          rcases Nat.lt_or_ge f f' with h | h
          · exact h
          · exact absurd hfs'.1 (hpre f' h).not_stale
        have hf'n : f' < n := hlen' ▸ hfs'.lt_length
        have hFf := hF f hfs
        refine ihF (n - f') (by omega) κ ⟨cs', σ'⟩ hK
          (fun i hi' => Nat.le_of_eq (by rw [hi'.unique hfs'])) ?_
        obtain ⟨cm, hcm, hcmc, hfull⟩ := hP.cell
        obtain ⟨cm', hcm', hc1, hc2⟩ := hshape m cm hcm
        obtain ⟨hc3, hc4⟩ := hc2 (by omega)
        exact {
          len := hlen'
          wf := hwf'
          lt := hP.lt
          first_le := fun i hi' => by rw [hi'.unique hfs']; exact hf'm
          settled := fun k hk => by
            by_cases hkf : k = f
            · subst hkf
              exact (hP.settled k hk).transfer hcodes (fun cc hcc => by
                rw [hcell] at hcc; cases hcc
                exact ⟨_, hat, rfl, hrr, hww⟩)
            · exact hsettled_ne k hk hkf
          top := htop'
          cell := ⟨cm', hcm', hc1.trans hcmc, by rw [hc3, hc4]; exact hfull⟩
          bound := hP.bound
          exposed := fun ℓ hℓ => (hequiv'.exposed m ℓ).mpr (hP.exposed ℓ hℓ) }
      · exact exit (fun i hi' => by
          rcases Nat.lt_or_ge m i with h | h
          · exact h
          · exact absurd ⟨i, hi', h⟩ hin)
    · -- a settled cell cannot diverge from `E`
      have : ci.reads p :=
        (Deterministic.reads_iff hevals p).mpr (by rw [hL]; simp)
      exact absurd hwp (hdisj p this)
  · -- `f = m`: the run of `m` itself
    obtain ⟨cm, hcm, hcmc, hfull⟩ := hP.cell
    rw [hcell] at hcm; cases hcm
    have hσ : σf = σm := htopf.unique hP.top
    subst hσ hcmc
    -- settled cells above `f` are unchanged
    have hsettled' : ∀ k, k < f → Settled V cs' k := fun k hk => hsettled_ne k hk (by omega)
    rcases run_analysis hcell heval hrc hagree with
      ⟨hrL, o', σ'', hevalE⟩ | ⟨pre, p, rest, hL, hp, hrd, _⟩
    · -- an `E`-run of `m`
      by_cases hκ : (rseq (Output := Output) ci.code σf).length < κ
      · -- the recorded sets were already `E`'s: the write set is kept
        obtain ⟨o₁, σ₁, hev₁⟩ := hfull hκ
        obtain ⟨hw, _⟩ := Deterministic.det hev₁ hevalE (fun _ _ => rfl)
        have hpre := run_prefix_clean hfs hcell hat rfl hretag (fun ℓ h => (hw ℓ).mpr h)
        exact exit (fun i hi' => by
          rcases Nat.lt_or_ge f i with h | h
          · exact h
          · exact absurd hi'.1 (hpre i h).not_stale)
      · by_cases hin : ∃ i, FirstStale cs' i ∧ i ≤ f
        · obtain ⟨f', hfs', hf'f⟩ := hin
          refine ihK _ (by omega) ((rseq (Output := Output) ci.code σf).length + 1)
            ⟨cs', σ'⟩ (Nat.le_refl _) ?_
          exact {
            len := hlen'
            wf := hwf'
            lt := hP.lt
            first_le := fun i hi' => by rw [hi'.unique hfs']; exact hf'f
            settled := hsettled'
            top := htop'
            cell := ⟨_, hat, rfl, fun _ => ⟨o', σ'', hevalE⟩⟩
            bound := Nat.le_refl _
            exposed := fun ℓ hℓ => by
              rw [List.take_of_length_le (Nat.le_succ _)] at hℓ
              exact exposed_of_run hi hequiv hrc ((hrL ℓ).mpr hℓ) }
        · exact exit (fun i hi' => by
            rcases Nat.lt_or_ge f i with h | h
            · exact h
            · exact absurd ⟨i, hi', h⟩ hin)
    · -- the run of `m` diverges from `E` at `p`, after `|pre| ≥ κ` agreeing reads
      have hκ : κ ≤ pre.length := by
        refine Nat.le_of_not_lt fun hlt => hp (hagree p (hP.exposed p ?_))
        rw [hL]
        exact mem_take_of_lt hlt
      have hLlen : (rseq (Output := Output) ci.code σf).length = pre.length + 1 + rest.length := by
        rw [hL]; simp; omega
      by_cases hin : ∃ i, FirstStale cs' i ∧ i ≤ f
      · obtain ⟨f', hfs', hf'f⟩ := hin
        refine ihK _ (by omega) (pre.length + 1) ⟨cs', σ'⟩ (Nat.le_refl _) ?_
        exact {
          len := hlen'
          wf := hwf'
          lt := hP.lt
          first_le := fun i hi' => by rw [hi'.unique hfs']; exact hf'f
          settled := hsettled'
          top := htop'
          cell := ⟨_, hat, rfl, fun h => absurd h (by omega)⟩
          bound := by omega
          exposed := fun ℓ hℓ => by
            rw [hL, take_succ_append] at hℓ
            exact exposed_of_run hi hequiv hrc (hrd ℓ hℓ) }
      · exact exit (fun i hi' => by
          rcases Nat.lt_or_ge f i with h | h
          · exact h
          · exact absurd ⟨i, hi', h⟩ hin)

/-- Every execution of the strategy from a well-formed state whose first
stale cell is within `D` of the bottom is finite.  By induction on `D`:
the first stale cell `f` starts a phase for `f`. -/
theorem acc_aux : ∀ D (nb : Notebook Code Output L V), WellFormed nb →
    (∀ f, FirstStale nb.cells f → nb.cells.length - f < D) → Acc StrategyRel nb := by
  intro D
  induction D with
  | zero =>
    intro nb _ h
    exact acc_of_no_firstStale (fun i hi => by have := h i hi; omega)
  | succ D ih =>
    intro nb hwf hD
    rcases allClean_or_firstStale nb.cells with hac | ⟨f, hfs⟩
    · exact acc_of_no_firstStale (fun _ hi => hac.not_firstStale hi)
    · obtain ⟨σf, htop, hset, _⟩ :=
        clean_prefix hwf f (Nat.le_of_lt hfs.lt_length) hfs.clean_before
      obtain ⟨cf, hcf⟩ := exists_getElem?_eq_some hfs.lt_length
      have hDf := hD f hfs
      refine phase_acc (n := nb.cells.length) (m := f) (σm := σf) (c := cf.code)
        (fun nb' hlen hwf' h => ih nb' hwf' (fun f' hf' => by
          have := h f' hf'
          rw [hlen]
          omega))
        _ 0 nb (Nat.le_refl _) ?_
      exact {
        len := rfl
        wf := hwf
        lt := hfs.lt_length
        first_le := fun i hi => Nat.le_of_eq (hi.unique hfs)
        settled := hset
        top := htop
        cell := ⟨cf, hcf, rfl, fun h => absurd h (Nat.not_lt_zero _)⟩
        bound := Nat.zero_le _
        exposed := fun ℓ hℓ => by simp at hℓ }

/-- **Theorem 2.5 (Progress), termination.**  Under the determinism
assumption, every execution of the strategy "run the first stale cell"
from a well-formed state is finite, whatever outputs the cells produce. -/
theorem progress_terminates {nb : Notebook Code Output L V} (hwf : WellFormed nb) :
    Acc StrategyRel nb :=
  acc_aux (nb.cells.length + 1) nb hwf (fun _ _ => by omega)

end Strategy

section Halting
variable [CellEval Code Output L V]

/-- A state from which the strategy cannot continue is all-clean or stuck. -/
theorem halted_allClean_or_stuck {nb : Notebook Code Output L V}
    (h : ∀ nb', ¬ StrategyStep nb nb') : AllClean nb.cells ∨ Stuck nb := by
  rcases allClean_or_firstStale nb.cells with hac | ⟨i, hfs⟩
  · exact .inl hac
  · exact .inr ⟨i, hfs, fun nb' hs => h nb' ⟨i, hfs, hs⟩⟩

/-- Every state reached by the strategy is well-formed (Preservation). -/
theorem StrategyStar.wellFormed {nb nb' : Notebook Code Output L V}
    (h : StrategyStar nb nb') (hwf : WellFormed nb) : WellFormed nb' := by
  induction h with
  | refl => exact hwf
  | step hs _ ih =>
    obtain ⟨_, _, hrun⟩ := hs
    exact ih (preservation hwf hrun)

/-- Every execution of the strategy that cannot continue ends in a
well-formed state that is all-clean or stuck. -/
theorem progress_halted {nb nb' : Notebook Code Output L V}
    (hwf : WellFormed nb) (hexec : StrategyStar nb nb')
    (hhalt : ∀ nb'', ¬ StrategyStep nb' nb'') :
    WellFormed nb' ∧ (AllClean nb'.cells ∨ Stuck nb') :=
  ⟨hexec.wellFormed hwf, halted_allClean_or_stuck hhalt⟩

variable [Deterministic Code Output L V]

/-- **Theorem 2.5 (Progress).**  Under the determinism assumption,
repeatedly running the first stale cell of a well-formed notebook
terminates (`progress_terminates`) in a well-formed notebook in which
either every cell is clean, or the first stale cell is stuck. -/
theorem progress {nb : Notebook Code Output L V} (hwf : WellFormed nb) :
    ∃ nb', StrategyStar nb nb' ∧ WellFormed nb' ∧ (AllClean nb'.cells ∨ Stuck nb') := by
  induction progress_terminates hwf with
  | intro nb _ ih =>
    by_cases hs : ∃ nb₁, StrategyStep nb nb₁
    · obtain ⟨nb₁, hst⟩ := hs
      have hst' := hst
      obtain ⟨_, _, hrun⟩ := hst'

      obtain ⟨nb', h1, h2, h3⟩ := ih nb₁ hst (preservation hwf hrun)
      exact ⟨nb', .step hst h1, h2, h3⟩
    · exact ⟨nb, .refl nb, hwf, halted_allClean_or_stuck (fun nb' h => hs ⟨nb', h⟩)⟩

/-- Progress combined with Theorem 2.4: the strategy terminates either in
an output-consistent (reproducible) notebook or at a stuck cell. -/
theorem progress_reproducible {nb : Notebook Code Output L V} (hwf : WellFormed nb) :
    ∃ nb', StrategyStar nb nb' ∧ (Reproducible nb'.erase ∨ Stuck nb') := by
  obtain ⟨nb', hexec, hwf', hend⟩ := progress hwf
  refine ⟨nb', hexec, ?_⟩
  rcases hend with hac | hstuck
  · exact .inl (output_consistency hwf' hac)
  · exact .inr hstuck

end Halting

end FlowBook
