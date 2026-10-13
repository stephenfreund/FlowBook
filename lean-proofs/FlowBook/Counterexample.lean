/-
# Set-Level Determinism Does Not Imply Progress

A natural weakening of the determinism assumption of Theorem 2.5
states it in terms of read *sets*: rerunning a cell when the values it
read are unchanged reproduces its read set, write set, and written
values (`SetDeterministic` below).  This file checks the supplement's
example showing that this set-level form does not ensure termination,
which is why the supplement and `FlowBook/Progress.lean` assume
determinism in its sequential form (`Deterministic`).

The notebook has four cells over locations `a`, `b`, `c`, `z`:

    A:  a := 0        B:  b := 0        C:  c := 0
    U:  if a ≠ 0 = b      then read {a, b}, c := 1
        elif b ≠ 0 = c    then read {b, c}, a := 1
        elif c ≠ 0 = a    then read {c, a}, b := 1
        else                   read {a, b, c}, z := 0

The supplement's `U` writes nothing in its last case; here it writes
`z := 0` because every cell of this language writes exactly one
location.  The cycle never takes that case.

`U` behaves like Berry's "Gustave" function: its read set depends on
which input is nonzero, so it cannot be computed by reading its inputs
one at a time.  The evaluation relation satisfies `frame`, `locality`,
and `SetDeterministic`.  From the well-formed state `a = 1`, `b = c = 0`
with `U` stale, running the first stale cell runs `U, A, U, C, U, B` and
returns to the same state (`cycle`), so the strategy does not terminate
(`not_acc`).
-/
import FlowBook.Progress

namespace FlowBook
namespace Counterexample

/-! ## The cell language -/

inductive Lc where
  | a | b | c | z
deriving DecidableEq, Repr

instance (P : Lc → Prop) [DecidablePred P] : Decidable (∀ ℓ, P ℓ) :=
  decidable_of_iff (P .a ∧ P .b ∧ P .c ∧ P .z) (by
    constructor
    · rintro ⟨h1, h2, h3, h4⟩ ℓ
      cases ℓ <;> assumption
    · intro h
      exact ⟨h _, h _, h _, h _⟩)

instance (P : Lc → Prop) [DecidablePred P] : Decidable (∃ ℓ, P ℓ) :=
  decidable_of_iff (P .a ∨ P .b ∨ P .c ∨ P .z) (by
    constructor
    · rintro (h | h | h | h) <;> exact ⟨_, h⟩
    · rintro ⟨ℓ, h⟩
      cases ℓ
      · exact .inl h
      · exact .inr (.inl h)
      · exact .inr (.inr (.inl h))
      · exact .inr (.inr (.inr h)))

inductive Cc where
  | setA | setB | setC | gus
deriving DecidableEq, Repr

abbrev St : Type := Store Lc Nat

def upd (σ : St) (ℓ : Lc) (v : Nat) : St := fun x => if x = ℓ then some v else σ x

/-- Whether a location holds `0`. -/
def is0 (σ : St) (ℓ : Lc) : Bool := σ ℓ == some 0

/-- The branch `U` takes, given whether `a`, `b`, `c` hold `0`: the
location it writes, the value written, and the locations it reads. -/
def gusCase (x y z : Bool) : Lc × Nat × List Lc :=
  if !x && y then (.c, 1, [.a, .b])
  else if !y && z then (.a, 1, [.b, .c])
  else if !z && x then (.b, 1, [.c, .a])
  else (.z, 0, [.a, .b, .c])

/-- The branch depends only on the flags of the locations it reads. -/
theorem gusCase_agree (x y z x' y' z' : Bool)
    (ha : Lc.a ∈ (gusCase x y z).2.2 → x' = x) (hb : Lc.b ∈ (gusCase x y z).2.2 → y' = y)
    (hc : Lc.c ∈ (gusCase x y z).2.2 → z' = z) :
    gusCase x' y' z' = gusCase x y z := by
  revert ha hb hc
  cases x <;> cases y <;> cases z <;> cases x' <;> cases y' <;> cases z' <;> decide

/-- Each command writes one location: which one, the value, and the
locations it reads. -/
def spec : Cc → St → Lc × Nat × List Lc
  | .setA, _ => (.a, 0, [])
  | .setB, _ => (.b, 0, [])
  | .setC, _ => (.c, 0, [])
  | .gus, σ => gusCase (is0 σ .a) (is0 σ .b) (is0 σ .c)

/-- The evaluator: the new store, the read list, and the write list. -/
def ev (k : Cc) (σ : St) : St × List Lc × List Lc :=
  (upd σ (spec k σ).1 (spec k σ).2.1, (spec k σ).2.2, [(spec k σ).1])

theorem ev_frame (k : Cc) (σ : St) : ∀ ℓ, ℓ ∉ (ev k σ).2.2 → (ev k σ).1 ℓ = σ ℓ := by
  intro ℓ hℓ
  simp only [ev, List.mem_singleton] at hℓ ⊢
  simp [upd, hℓ]

theorem spec_agree (k : Cc) (σ τ : St) (h : ∀ ℓ, ℓ ∈ (spec k σ).2.2 → τ ℓ = σ ℓ) :
    spec k τ = spec k σ := by
  cases k
  case setA | setB | setC => rfl
  case gus =>
    have h0 : ∀ ℓ, ℓ ∈ (spec .gus σ).2.2 → is0 τ ℓ = is0 σ ℓ := fun ℓ hℓ => by
      simp [is0, h ℓ hℓ]
    exact gusCase_agree _ _ _ _ _ _ (h0 .a) (h0 .b) (h0 .c)

/-- Agreeing on the locations read gives the same reads, writes, and
written values. -/
theorem ev_agree (k : Cc) (σ τ : St) (h : ∀ ℓ, ℓ ∈ (ev k σ).2.1 → τ ℓ = σ ℓ) :
    (ev k τ).2 = (ev k σ).2 ∧ ∀ ℓ, ℓ ∈ (ev k σ).2.2 → (ev k τ).1 ℓ = (ev k σ).1 ℓ := by
  have hs := spec_agree k σ τ h
  refine ⟨by simp [ev, hs], fun ℓ hℓ => ?_⟩
  simp only [ev, List.mem_singleton] at hℓ
  subst hℓ
  simp [ev, hs, upd]

/-- The evaluation relation is the graph of `ev`; outputs are trivial. -/
def GEval (k : Cc) (σ : St) (_ : Unit) (σ' : St) (r w : Lc → Prop) : Prop :=
  σ' = (ev k σ).1 ∧ (∀ ℓ, r ℓ ↔ ℓ ∈ (ev k σ).2.1) ∧ (∀ ℓ, w ℓ ↔ ℓ ∈ (ev k σ).2.2)

instance instCellEval : CellEval Cc Unit Lc Nat where
  Eval := GEval
  frame := by
    rintro k σ o σ' r w ⟨hσ', _, hw⟩ ℓ hnw
    rw [hσ']
    exact ev_frame k σ ℓ (fun h => hnw ((hw ℓ).mpr h))
  locality := by
    rintro k σ o σ' r w τ ⟨hσ', hr, hw⟩ hag
    obtain ⟨heq, hval⟩ := ev_agree k σ τ (fun ℓ hℓ => hag ℓ ((hr ℓ).mpr hℓ))
    have hrl : (ev k τ).2.1 = (ev k σ).2.1 := congrArg Prod.fst heq
    have hwl : (ev k τ).2.2 = (ev k σ).2.2 := congrArg Prod.snd heq
    refine ⟨(ev k τ).1, ⟨rfl, ?_, ?_⟩, ?_, ?_⟩
    · intro ℓ; rw [hrl]; exact hr ℓ
    · intro ℓ; rw [hwl]; exact hw ℓ
    · intro ℓ hwℓ; rw [hσ']; exact hval ℓ ((hw ℓ).mp hwℓ)
    · intro ℓ hnw
      exact ev_frame k τ ℓ (fun h => hnw ((hw ℓ).mpr (hwl ▸ h)))

/-- **The supplement's determinism assumption**, stated for read sets:
rerunning a cell from a store that agrees on the locations it read
reproduces its read set, write set, and written values. -/
def SetDeterministic (Code Output L V : Type) [CellEval Code Output L V] : Prop :=
  ∀ {k : Code} {σ : Store L V} {o : Output} {σ' : Store L V} {r w : L → Prop}
    {τ : Store L V} {o₂ : Output} {τ' : Store L V} {r₂ w₂ : L → Prop},
    Eval k σ o σ' r w → Eval k τ o₂ τ' r₂ w₂ → (∀ ℓ, r ℓ → τ ℓ = σ ℓ) →
    (∀ ℓ, r₂ ℓ ↔ r ℓ) ∧ (∀ ℓ, w₂ ℓ ↔ w ℓ) ∧ ∀ ℓ, w ℓ → τ' ℓ = σ' ℓ

theorem setDeterministic : SetDeterministic Cc Unit Lc Nat := by
  rintro k σ o σ' r w τ o₂ τ' r₂ w₂ ⟨hσ', hr, hw⟩ ⟨hτ', hr₂, hw₂⟩ hag
  obtain ⟨heq, hval⟩ := ev_agree k σ τ (fun ℓ hℓ => hag ℓ ((hr ℓ).mpr hℓ))
  have hrl : (ev k τ).2.1 = (ev k σ).2.1 := congrArg Prod.fst heq
  have hwl : (ev k τ).2.2 = (ev k σ).2.2 := congrArg Prod.snd heq
  refine ⟨fun ℓ => by rw [hr₂, hr, hrl], fun ℓ => by rw [hw₂, hw, hwl], fun ℓ hwℓ => ?_⟩
  rw [hσ', hτ']
  exact hval ℓ ((hw ℓ).mp hwℓ)

/-! ## Notebooks with list-valued read and write sets -/

/-- A cell with list-valued read and write sets; every cell has run. -/
structure LCell where
  code : Cc
  tag : Tag
  rl : List Lc
  wl : List Lc
deriving DecidableEq

def LCell.toCell (x : LCell) : Cell Cc Unit Lc :=
  ⟨x.code, some (), x.tag, fun ℓ => ℓ ∈ x.rl, fun ℓ => ℓ ∈ x.wl⟩

def rAt (l : List LCell) (j : Nat) : List Lc := ((l[j]?).map LCell.rl).getD []
def wAt (l : List LCell) (j : Nat) : List Lc := ((l[j]?).map LCell.wl).getD []

theorem readsAt_iff (l : List LCell) (j : Nat) (ℓ : Lc) :
    ReadsAt (l.map LCell.toCell) j ℓ ↔ ℓ ∈ rAt l j := by
  unfold ReadsAt rAt
  rw [List.getElem?_map]
  cases l[j]? <;> simp [LCell.toCell]

theorem writesAt_iff (l : List LCell) (j : Nat) (ℓ : Lc) :
    WritesAt (l.map LCell.toCell) j ℓ ↔ ℓ ∈ wAt l j := by
  unfold WritesAt wAt
  rw [List.getElem?_map]
  cases l[j]? <;> simp [LCell.toCell]

theorem wAt_of_ge {l : List LCell} {j : Nat} (h : l.length ≤ j) : wAt l j = [] := by
  simp [wAt, List.getElem?_eq_none h]

/-- A decidable sufficient condition for rerun consistency. -/
theorem rerunConsistent_of (l : List LCell) (i : Nat)
    (h1 : ∀ ℓ, ℓ ∈ rAt l i → ℓ ∉ wAt l i)
    (h2 : ∀ ℓ, ℓ ∈ rAt l i → ∃ j, j < i ∧ ℓ ∈ wAt l j)
    (h3 : ∀ ℓ, ℓ ∈ rAt l i → ∀ j, j < l.length → i < j → ℓ ∉ wAt l j)
    (h4 : ∀ ℓ, ℓ ∈ wAt l i → ∀ j, j < i → ℓ ∉ rAt l j) :
    RerunConsistent (l.map LCell.toCell) i where
  noReadAndWrite ℓ hr hw := h1 ℓ ((readsAt_iff ..).mp hr) ((writesAt_iff ..).mp hw)
  writeBeforeRead ℓ hr := by
    obtain ⟨j, hj, hw⟩ := h2 ℓ ((readsAt_iff ..).mp hr)
    exact ⟨j, hj, (writesAt_iff ..).mpr hw⟩
  noReadBeforeWrite ℓ hr := by
    rintro ⟨j, hj, hw⟩
    have hw' := (writesAt_iff ..).mp hw
    rcases Nat.lt_or_ge j l.length with hl | hl
    · exact h3 ℓ ((readsAt_iff ..).mp hr) j hl hj hw'
    · rw [wAt_of_ge hl] at hw'; simp at hw'
  noWriteAfterRead ℓ hw := by
    rintro ⟨j, hj, hr⟩
    exact h4 ℓ ((writesAt_iff ..).mp hw) j hj ((readsAt_iff ..).mp hr)

/-- `Marked` in list form. -/
def MarkedL (l : List LCell) (i : Nat) (w : List Lc) (j : Nat) : Prop :=
  (i < j ∧ ∃ ℓ, (ℓ ∈ wAt l i ∨ ℓ ∈ w) ∧ (ℓ ∈ rAt l j ∨ ℓ ∈ wAt l j)) ∨
  (∃ ℓ, ℓ ∈ wAt l i ∧ ℓ ∉ w ∧ j < i ∧ ℓ ∈ wAt l j ∧ ∀ k, k < i → j < k → ℓ ∉ wAt l k)

instance (l : List LCell) (i : Nat) (w : List Lc) (j : Nat) : Decidable (MarkedL l i w j) := by
  unfold MarkedL; infer_instance

theorem marked_iff (l : List LCell) (i : Nat) (w : List Lc) (j : Nat) :
    Marked (l.map LCell.toCell) i (fun ℓ => ℓ ∈ w) j ↔ MarkedL l i w j := by
  unfold Marked FwdStale BwdStale IsLastWriter MarkedL
  simp only [readsAt_iff, writesAt_iff]
  constructor
  · rintro (h | ⟨ℓ, h1, h2, h3, h4, h5⟩)
    · exact .inl h
    · exact .inr ⟨ℓ, h1, h2, h3, h4, fun k hk hjk => h5 k hjk hk⟩
  · rintro (h | ⟨ℓ, h1, h2, h3, h4, h5⟩)
    · exact .inl h
    · exact .inr ⟨ℓ, h1, h2, h3, h4, fun k hjk hk => h5 k hk hjk⟩

/-- The tag update of `[Inst-Run]`, computed. -/
def retag (l : List LCell) (i : Nat) (w : List Lc) (x' : LCell) : List LCell :=
  l.mapIdx fun j x =>
    if j = i then x'
    else if MarkedL l i w j ∨ x.tag = .stale then { x with tag := .stale }
    else { x with tag := .clean }

theorem retag_spec (l : List LCell) (i : Nat) (w : List Lc) (x' : LCell) :
    RetagSpec (l.map LCell.toCell) ((retag l i w x').map LCell.toCell) i (fun ℓ => ℓ ∈ w) := by
  intro j hj c hc
  rw [List.getElem?_map] at hc
  cases hx : l[j]? with
  | none => rw [hx] at hc; cases hc
  | some x =>
    rw [hx] at hc
    cases hc
    rw [List.getElem?_map, retag, List.getElem?_mapIdx, hx]
    by_cases hm : MarkedL l i w j ∨ x.tag = .stale
    · refine ⟨.stale, by simp [hj, hm, LCell.toCell], ?_⟩
      rw [marked_iff]
      exact ⟨fun _ => hm, fun _ => rfl⟩
    · refine ⟨.clean, by simp [hj, hm, LCell.toCell], ?_⟩
      rw [marked_iff]
      constructor
      · intro h; cases h
      · intro h; exact absurd h hm

/-- The notebook built from list cells and a store. -/
def nb (l : List LCell) (σ : St) : Notebook Cc Unit Lc Nat := ⟨l.map LCell.toCell, σ⟩

/-- The cell recorded by running `x` from `σ`. -/
def newCell (x : LCell) (σ : St) : LCell :=
  { x with tag := .clean, rl := (ev x.code σ).2.1, wl := (ev x.code σ).2.2 }

/-- The state after running cell `i` from `nb l σ`. -/
def after (l : List LCell) (σ : St) (i : Nat) : List LCell × St :=
  match l[i]? with
  | some x => (retag l i (ev x.code σ).2.2 (newCell x σ), (ev x.code σ).1)
  | none => (l, σ)

/-- A decidable sufficient condition for one strategy step. -/
theorem step_of (l : List LCell) (σ : St) (i : Nat) (x : LCell) (hx : l[i]? = some x)
    (hfs1 : (l[i]?).map LCell.tag = some .stale)
    (hfs2 : ∀ j, j < i → (l[j]?).map LCell.tag ≠ some .stale)
    (h1 : ∀ ℓ, ℓ ∈ rAt (l.set i (newCell x σ)) i → ℓ ∉ wAt (l.set i (newCell x σ)) i)
    (h2 : ∀ ℓ, ℓ ∈ rAt (l.set i (newCell x σ)) i →
      ∃ j, j < i ∧ ℓ ∈ wAt (l.set i (newCell x σ)) j)
    (h3 : ∀ ℓ, ℓ ∈ rAt (l.set i (newCell x σ)) i → ∀ j, j < l.length → i < j →
      ℓ ∉ wAt (l.set i (newCell x σ)) j)
    (h4 : ∀ ℓ, ℓ ∈ wAt (l.set i (newCell x σ)) i → ∀ j, j < i →
      ℓ ∉ rAt (l.set i (newCell x σ)) j) :
    StrategyStep (nb l σ) (nb (after l σ i).1 (after l σ i).2) := by
  have hi : i < l.length := by
    rcases Nat.lt_or_ge i l.length with h | h
    · exact h
    · rw [List.getElem?_eq_none h] at hx; cases hx
  refine ⟨i, ⟨?_, ?_⟩, ?_⟩
  · rw [hx] at hfs1
    exact ⟨x.toCell, by simp [nb, List.getElem?_map, hx], by
      simpa [LCell.toCell] using hfs1⟩
  · rintro j hj ⟨c, hc, hst⟩
    simp only [nb, List.getElem?_map] at hc
    cases hy : l[j]? with
    | none => rw [hy] at hc; cases hc
    | some y =>
      rw [hy] at hc; cases hc
      exact hfs2 j hj (by rw [hy]; simpa [LCell.toCell] using hst)
  · simp only [after, hx]
    refine InstStep.run (ci := x.toCell) (o := ()) (σ' := (ev x.code σ).1)
      (r := fun ℓ => ℓ ∈ (ev x.code σ).2.1) (w := fun ℓ => ℓ ∈ (ev x.code σ).2.2) ?_ ?_ ?_ ?_ ?_ ?_
    · simp [nb, List.getElem?_map, hx]
    · exact ⟨rfl, fun _ => Iff.rfl, fun _ => Iff.rfl⟩
    · have := rerunConsistent_of (l.set i (newCell x σ)) i h1 h2
        (by simpa using h3) h4
      rw [List.map_set] at this
      exact this
    · simp [nb, retag]
    · simp only [List.getElem?_map, retag, List.getElem?_mapIdx, List.getElem?_eq_getElem hi]
      simp [newCell, LCell.toCell]
    · exact retag_spec l i _ _

/-! ## The cycle -/

def cA : LCell := ⟨.setA, .clean, [], [.a]⟩
def cB : LCell := ⟨.setB, .clean, [], [.b]⟩
def cC : LCell := ⟨.setC, .clean, [], [.c]⟩

/-- The start: `U` last read `{b, c}` and wrote `a := 1`, and is stale. -/
def l0 : List LCell := [cA, cB, cC, ⟨.gus, .stale, [.b, .c], [.a]⟩]

/-- The start store: `a = 1`, `b = c = 0`. -/
def σ0 : St := fun
  | .a => some 1
  | .b => some 0
  | .c => some 0
  | .z => none

def s1 := after l0 σ0 3
def s2 := after s1.1 s1.2 0
def s3 := after s2.1 s2.2 3
def s4 := after s3.1 s3.2 2
def s5 := after s4.1 s4.2 3
def s6 := after s5.1 s5.2 1

theorem step1 : StrategyStep (nb l0 σ0) (nb s1.1 s1.2) :=
  step_of _ _ 3 _ rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem step2 : StrategyStep (nb s1.1 s1.2) (nb s2.1 s2.2) :=
  step_of _ _ 0 _ rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem step3 : StrategyStep (nb s2.1 s2.2) (nb s3.1 s3.2) :=
  step_of _ _ 3 _ rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem step4 : StrategyStep (nb s3.1 s3.2) (nb s4.1 s4.2) :=
  step_of _ _ 2 _ rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem step5 : StrategyStep (nb s4.1 s4.2) (nb s5.1 s5.2) :=
  step_of _ _ 3 _ rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem step6 : StrategyStep (nb s5.1 s5.2) (nb s6.1 s6.2) :=
  step_of _ _ 1 _ rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- After six runs, the notebook is back where it started. -/
theorem s6_eq : nb s6.1 s6.2 = nb l0 σ0 := by
  have h1 : s6.1 = l0 := by decide
  have h2 : s6.2 = σ0 := by
    funext ℓ
    cases ℓ <;> decide
  rw [h1, h2]

/-- The strategy runs `U, A, U, C, U, B` and returns to the start. -/
theorem cycle : StrategyStep (nb l0 σ0) (nb s1.1 s1.2) ∧ StrategyStep (nb s1.1 s1.2) (nb s2.1 s2.2) ∧
    StrategyStep (nb s2.1 s2.2) (nb s3.1 s3.2) ∧ StrategyStep (nb s3.1 s3.2) (nb s4.1 s4.2) ∧
    StrategyStep (nb s4.1 s4.2) (nb s5.1 s5.2) ∧ StrategyStep (nb s5.1 s5.2) (nb l0 σ0) :=
  ⟨step1, step2, step3, step4, step5, s6_eq ▸ step6⟩

/-- The start state is well-formed: `A`, `B`, `C` rerun to the same
effect, changing the store only at `a`, which `U` writes below them. -/
theorem wellFormed_start : WellFormed (nb l0 σ0) := by
  intro i hclean
  have hi : i < 3 := by
    obtain ⟨c, hc, hcl⟩ := hclean
    simp only [nb, List.getElem?_map] at hc
    rcases Nat.lt_or_ge i 3 with h | h
    · exact h
    · exfalso
      match i, h, hc with
      | 3, _, hc => simp [l0, LCell.toCell] at hc; subst hc; cases hcl
      | n + 4, _, hc => simp [l0] at hc
  refine ⟨?_, ?_⟩
  · match i, hi with
    | 0, _ =>
      refine ⟨cA.toCell, rfl, (), (ev .setA σ0).1, rfl, ⟨rfl, ?_, ?_⟩, ?_⟩
      · intro ℓ; simp [cA, LCell.toCell, ev, spec]
      · intro ℓ; simp [cA, LCell.toCell, ev, spec]
      · intro ℓ hℓ
        cases ℓ
        · exact absurd ⟨3, by decide, (writesAt_iff _ _ _).mpr (by decide)⟩ hℓ
        all_goals rfl
    | 1, _ =>
      refine ⟨cB.toCell, rfl, (), (ev .setB σ0).1, rfl, ⟨rfl, ?_, ?_⟩, ?_⟩
      · intro ℓ; simp [cB, LCell.toCell, ev, spec]
      · intro ℓ; simp [cB, LCell.toCell, ev, spec]
      · intro ℓ _; cases ℓ <;> rfl
    | 2, _ =>
      refine ⟨cC.toCell, rfl, (), (ev .setC σ0).1, rfl, ⟨rfl, ?_, ?_⟩, ?_⟩
      · intro ℓ; simp [cC, LCell.toCell, ev, spec]
      · intro ℓ; simp [cC, LCell.toCell, ev, spec]
      · intro ℓ _; cases ℓ <;> rfl
  · exact rerunConsistent_of l0 i
      (by match i, hi with | 0, _ | 1, _ | 2, _ => decide)
      (by match i, hi with | 0, _ | 1, _ | 2, _ => decide)
      (by match i, hi with | 0, _ | 1, _ | 2, _ => decide)
      (by match i, hi with | 0, _ | 1, _ | 2, _ => decide)

/-- **The strategy does not terminate** from a well-formed state of a
language satisfying the supplement's set-level determinism assumption. -/
theorem not_acc : ¬ Acc StrategyRel (nb l0 σ0) := by
  let P : Notebook Cc Unit Lc Nat → Prop := fun x =>
    x = nb l0 σ0 ∨ x = nb s1.1 s1.2 ∨ x = nb s2.1 s2.2 ∨ x = nb s3.1 s3.2 ∨
    x = nb s4.1 s4.2 ∨ x = nb s5.1 s5.2
  have hsucc : ∀ x, P x → ∃ y, P y ∧ StrategyStep x y := by
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := cycle
    rintro x (rfl | rfl | rfl | rfl | rfl | rfl)
    · exact ⟨_, .inr (.inl rfl), h1⟩
    · exact ⟨_, .inr (.inr (.inl rfl)), h2⟩
    · exact ⟨_, .inr (.inr (.inr (.inl rfl))), h3⟩
    · exact ⟨_, .inr (.inr (.inr (.inr (.inl rfl)))), h4⟩
    · exact ⟨_, .inr (.inr (.inr (.inr (.inr rfl)))), h5⟩
    · exact ⟨_, .inl rfl, h6⟩
  intro hacc
  have key : ∀ x, Acc StrategyRel x → ¬ P x := by
    intro x hx
    induction hx with
    | intro x _ ih =>
      intro hP
      obtain ⟨y, hPy, hxy⟩ := hsucc x hP
      exact ih y hxy hPy
  exact key _ hacc (.inl rfl)

/-- Summary: the language satisfies `frame`, `locality`, and the
supplement's set-level determinism, the start state is well-formed, and
running the first stale cell never terminates. -/
theorem setDeterminism_insufficient :
    SetDeterministic Cc Unit Lc Nat ∧ WellFormed (nb l0 σ0) ∧ ¬ Acc StrategyRel (nb l0 σ0) :=
  ⟨setDeterministic, wellFormed_start, not_acc⟩

end Counterexample
end FlowBook
