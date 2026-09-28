/-
# Theorem 2.5 (Progress): Running Stale Cells Terminates

Theorem 2.5 of the supplement (proof in §5 of the supplement), for
the `RunToClean` algorithm:

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

From a well-formed state, every execution that reports no potential
non-termination terminates within `n(n+2)` runs, either in a state
where all cells are clean (and hence, by Theorem 2.4, the notebook is
reproducible) or at a cell that is stuck.

## Formalization notes

Cell evaluation is a *relation* (cells may be non-deterministic), so
rerunning stale cells with no check need not terminate: two cells with
empty read sets that each write `{a}` or `{b}`, choosing differently
on successive runs, mark each other stale forever — each run of the
later cell that drops a location backward-marks the earlier cell as
that location's last writer, and each run of the earlier cell
forward-marks the later one.  The check forbids exactly the event that
repeats: staleness moves toward earlier cells only via `BackwardStale`,
so requiring reruns to mark nothing before themselves *is* the
termination invariant, checked directly.

The relation `RunToClean F nb nb' F'` models the executions that
report no potential non-termination.  `F` is the *complement* of the
algorithm's `E`: the positions not yet executed.  A `first` step
executes a cell for the first time (`i ∈ F`; any successful
`[Inst-Run]` step).  A `rerun` step executes a cell again (`i ∉ F`)
and must leave every position `≤ i` clean — the algorithm's check,
verbatim.  Starting from a well-formed state we prove:

* `progress` — every such execution can be extended to termination: a
  state that is all-clean or stuck for the algorithm
  (`RunToCleanStuck`: no run at all for an unexecuted cell, or — for
  an already-executed cell — every available run marks a cell before
  it stale, which is the situation the algorithm reports).
  Termination is by the lexicographic measure
  `(|F|, n − first-stale position)`: a first execution removes `i`
  from `F`, and a rerun keeps the prefix clean by its side condition,
  so the first-stale position strictly increases.  The measure yields
  the paper's bound: at most `n` first executions and at most `n`
  reruns between consecutive first executions.  `progress_init`
  instantiates `F` with all positions — the algorithm's initial state
  `E = ∅`.
* `canRerun_after_run` and `cleanPrefix_run_exists` — the paper's key
  supporting claim (used by the Stability lemma): after any successful
  run of the first stale cell `i`, every cell `j < i` that was marked
  stale (necessarily by `BackwardStale`) can re-run from the new store
  reproducing its recorded output, read set, and write set
  (`MatchingRunAt`), and such a run marks nothing before itself
  (`matching_clean_prefix`) — so it satisfies the algorithm's rerun
  condition, and these reruns need never trigger the report: the
  run's `NoWriteAfterRead` check guarantees the new writes avoid those
  cells' read sets, so `locality` re-produces the recorded behavior.
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

/-! ## Reruns with matching read and write sets -/

section Replay
variable [CellEval Code Output L V]

open CellEval

/-- An `[Inst-Run]` step at cell `i` whose read and write sets equal
the recorded `R_i` and `W_i`.  The output may differ.  For such a run
`W_i \ W'_i = ∅`, so `BackwardStale` marks nothing — these are the
witnesses showing the algorithm's rerun condition is satisfiable
(`cleanPrefix_run_exists`). -/
def MatchingRunAt (nb : Notebook Code Output L V) (i : Nat)
    (nb' : Notebook Code Output L V) : Prop :=
  ∃ ci o σ' cs',
    nb.cells[i]? = some ci ∧
    Eval ci.code nb.store o σ' ci.reads ci.writes ∧
    RerunConsistent
      (nb.cells.set i
        { ci with out := some o, tag := .clean, reads := ci.reads, writes := ci.writes }) i ∧
    cs'.length = nb.cells.length ∧
    cs'[i]? = some
      { ci with out := some o, tag := .clean, reads := ci.reads, writes := ci.writes } ∧
    RetagSpec nb.cells cs' i ci.writes ∧
    nb' = ⟨cs', σ'⟩

/-- A matching run is an instance of `[Inst-Run]`. -/
theorem MatchingRunAt.instStep {nb nb' : Notebook Code Output L V} {i : Nat}
    (h : MatchingRunAt nb i nb') : InstStep nb (.run i) nb' := by
  obtain ⟨ci, o, σ', cs', hcell, heval, hrc, hlen, hat, hretag, rfl⟩ := h
  exact .run hcell heval hrc hlen hat hretag

/-- Cell `i` can re-run from the current store reproducing its recorded
output, read set, and write set, with its recorded accesses rerun
consistent.  Used to show the algorithm need not report at cells marked
stale by `BackwardStale`. -/
def CanRerunAt (nb : Notebook Code Output L V) (i : Nat) : Prop :=
  ∃ ci, nb.cells[i]? = some ci ∧ ∃ o σ',
    ci.out = some o ∧ Eval ci.code nb.store o σ' ci.reads ci.writes ∧
    RerunConsistent nb.cells i

/-- Updating a cell without changing its read/write sets leaves all
read/write tables unchanged. -/
theorem rwEquiv_set_same {cs : List (Cell Code Output L)} {i : Nat}
    {ci ci' : Cell Code Output L}
    (hcell : cs[i]? = some ci)
    (hr : ci'.reads = ci.reads) (hw : ci'.writes = ci.writes) :
    RWEquiv (cs.set i ci') cs := by
  have hi : i < cs.length := lt_length_of_getElem?_eq_some hcell
  constructor <;> intro j ℓ <;> by_cases hji : j = i
  · subst hji
    constructor
    · rintro ⟨cc, hc, hcr⟩
      rw [List.getElem?_set_self hi] at hc; cases hc
      rw [hr] at hcr
      exact ⟨ci, hcell, hcr⟩
    · rintro ⟨cc, hc, hcr⟩
      rw [hcell] at hc; cases hc
      exact ⟨ci', by simp [List.getElem?_set_self hi], by rw [hr]; exact hcr⟩
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
      rw [hw] at hcw
      exact ⟨ci, hcell, hcw⟩
    · rintro ⟨cc, hc, hcw⟩
      rw [hcell] at hc; cases hc
      exact ⟨ci', by simp [List.getElem?_set_self hi], by rw [hw]; exact hcw⟩
  · constructor
    · rintro ⟨cc, hc, hcw⟩
      rw [List.getElem?_set_ne (by omega)] at hc
      exact ⟨cc, hc, hcw⟩
    · rintro ⟨cc, hc, hcw⟩
      exact ⟨cc, by rw [List.getElem?_set_ne (by omega)]; exact hc, hcw⟩

/-- A cell that can re-run with its recorded behavior admits a
matching run. -/
theorem CanRerunAt.step_exists {nb : Notebook Code Output L V} {i : Nat}
    (h : CanRerunAt nb i) : ∃ nb', MatchingRunAt nb i nb' := by
  obtain ⟨ci, hcell, o, σ', hout, heval, hrc⟩ := h
  have hi : i < nb.cells.length := lt_length_of_getElem?_eq_some hcell
  obtain ⟨cs', hlen, hat, hretag⟩ := retag_exists nb.cells i ci.writes
    { ci with out := some o, tag := .clean, reads := ci.reads, writes := ci.writes }
  refine ⟨⟨cs', σ'⟩, ci, o, σ', cs', hcell, heval, ?_, hlen, hat hi, hretag, rfl⟩
  exact (rwEquiv_set_same
    (ci' := { ci with out := some o, tag := .clean, reads := ci.reads, writes := ci.writes })
    hcell rfl rfl).symm.rerunConsistent i hrc

/-- A matching rerun of the first stale cell leaves every position
`≤ i` clean: the write set is unchanged, so `BackwardStale` marks
nothing, and `ForwardStale` only marks cells after `i`.  This is the
strictly growing clean prefix in the paper's proof. -/
theorem matching_clean_prefix {nb nb' : Notebook Code Output L V} {i : Nat}
    (hfs : FirstStale nb.cells i) (h : MatchingRunAt nb i nb') :
    ∀ j, j ≤ i → IsClean nb'.cells j := by
  obtain ⟨ci, o, σ', cs', hcell, heval, hrc, hlen, hat, hretag, rfl⟩ := h
  intro j hj
  show IsClean cs' j
  by_cases hji : j = i
  · subst hji
    exact ⟨_, hat, rfl⟩
  · have hjlt : j < i := by omega
    have hjcs : j < nb.cells.length := by
      have := lt_length_of_getElem?_eq_some hcell; omega
    obtain ⟨cj, hcj⟩ := exists_getElem?_eq_some hjcs
    obtain ⟨t, hct, hstale⟩ := hretag j (by omega) cj hcj
    have hcjclean : cj.tag = Tag.clean := by
      rcases Tag.clean_or_stale cj.tag with h | h
      · exact h
      · exact absurd ⟨cj, hcj, h⟩ (hfs.2 j hjlt)
    have hnotm : ¬ Marked nb.cells i ci.writes j := by
      rintro (⟨hij, _⟩ | ⟨ℓ, hw1, hw2, _⟩)
      · omega
      · obtain ⟨cc, hcc, hccw⟩ := hw1
        rw [hcell] at hcc; cases hcc
        exact hw2 hccw
    have hnot : ¬ (Marked nb.cells i ci.writes j ∨ cj.tag = Tag.stale) := by
      rintro (hm | hs)
      · exact hnotm hm
      · rw [hcjclean] at hs; cases hs
    have ht : t = Tag.clean := by
      rcases Tag.clean_or_stale t with h | h
      · exact h
      · exact absurd (hstale.mp h) hnot
    exact ⟨{ cj with tag := t }, hct, ht⟩

/-- A cell that can re-run with its recorded behavior admits a run
satisfying the algorithm's rerun condition: an `[Inst-Run]` step that
leaves every position `≤ i` clean. -/
theorem cleanPrefix_run_exists {nb : Notebook Code Output L V} {i : Nat}
    (hfs : FirstStale nb.cells i) (h : CanRerunAt nb i) :
    ∃ nb', InstStep nb (.run i) nb' ∧ ∀ j, j ≤ i → IsClean nb'.cells j := by
  obtain ⟨nb', hm⟩ := h.step_exists
  exact ⟨nb', hm.instStep, matching_clean_prefix hfs hm⟩

/-- Any `[Inst-Run]` step preserves the number of cells. -/
theorem instStep_run_length {nb nb' : Notebook Code Output L V} {i : Nat}
    (h : InstStep nb (.run i) nb') : nb'.cells.length = nb.cells.length := by
  cases h with
  | run hcell heval hrc hlen hat hretag => exact hlen

end Replay

/-! ## The paper's key supporting claim

After any successful run of the first stale cell `i`, every cell `j < i`
that became stale (necessarily via `BackwardStale`) can re-run from the
new store reproducing its recorded behavior.  This is the sentence
"running `C_j` from `Σ'` produces the same behavior (same output and
read/write sets)" in the proof of Theorem 2.5 (§5 of the supplement),
and it is why the algorithm need not report at such cells. -/

section Backward
variable [CellEval Code Output L V]

open CellEval

/-- Rerun consistency of a cell above the executed cell, in the updated
tables (the `j < i` transfer used in the preservation proof). -/
theorem rerunConsistent_below_of_run {cs : List (Cell Code Output L)}
    {i j : Nat} {ci' : Cell Code Output L}
    (hji : j < i) (hi : i < cs.length)
    (hrci : RerunConsistent (cs.set i ci') i)
    (hrcj : RerunConsistent cs j) :
    RerunConsistent (cs.set i ci') j := by
  have hne : ∀ k, k ≠ i → ∀ ℓ,
      (ReadsAt (cs.set i ci') k ℓ ↔ ReadsAt cs k ℓ) ∧
      (WritesAt (cs.set i ci') k ℓ ↔ WritesAt cs k ℓ) := by
    intro k hk ℓ
    constructor <;> constructor <;> rintro ⟨c, hc, hp⟩
    · rw [List.getElem?_set_ne (by omega)] at hc; exact ⟨c, hc, hp⟩
    · exact ⟨c, by rw [List.getElem?_set_ne (by omega)]; exact hc, hp⟩
    · rw [List.getElem?_set_ne (by omega)] at hc; exact ⟨c, hc, hp⟩
    · exact ⟨c, by rw [List.getElem?_set_ne (by omega)]; exact hc, hp⟩
  have hji' : j ≠ i := by omega
  constructor
  · intro ℓ hr hw
    exact hrcj.noReadAndWrite ℓ ((hne j hji' ℓ).1.mp hr) ((hne j hji' ℓ).2.mp hw)
  · intro ℓ hr
    obtain ⟨k, hk, hkw⟩ := hrcj.writeBeforeRead ℓ ((hne j hji' ℓ).1.mp hr)
    exact ⟨k, hk, ((hne k (by omega) ℓ).2).mpr hkw⟩
  · rintro ℓ hr ⟨k, hk, hkw⟩
    have hrold : ReadsAt cs j ℓ := (hne j hji' ℓ).1.mp hr
    by_cases hki : k = i
    · subst hki
      -- the new writes of cell i avoid reads above i (NoWriteAfterRead of i)
      exact hrci.noWriteAfterRead ℓ hkw ⟨j, hji, hr⟩
    · exact hrcj.noReadBeforeWrite ℓ hrold ⟨k, hk, ((hne k hki ℓ).2).mp hkw⟩
  · rintro ℓ hw ⟨k, hk, hkr⟩
    have hki : k ≠ i := by omega
    exact hrcj.noWriteAfterRead ℓ ((hne j hji' ℓ).2.mp hw)
      ⟨k, hk, ((hne k hki ℓ).1).mp hkr⟩

/-- **Backward-marked cells can re-run with their recorded behavior.**
If `S · I` is well-formed with first stale cell `i` and
`S · I ⟹Run(i) S' · I'`, then every cell `j < i` (in particular every
cell marked stale by `BackwardStale`) can re-run from the new store
reproducing its recorded output, read set, and write set: the run's
rerun-consistency check guarantees `W'_i ∩ R_j = ∅`, so cell `j`
re-produces its recorded behavior (`locality`), and its recorded
accesses remain rerun consistent in the new tables. -/
theorem canRerun_after_run {nb nb' : Notebook Code Output L V} {i : Nat}
    (hwf : WellFormed nb) (hfs : FirstStale nb.cells i)
    (hstep : InstStep nb (.run i) nb') :
    ∀ j, j < i → CanRerunAt nb' j := by
  cases hstep with
  | @run _ _ ci o σ' r w cs' hcell heval hrc hlen hat hretag =>
    intro j hji
    have hi : i < nb.cells.length := lt_length_of_getElem?_eq_some hcell
    have hjcs : j < nb.cells.length := by omega
    obtain ⟨cj, hcj⟩ := exists_getElem?_eq_some hjcs
    -- cell j was clean (i is the first stale cell)
    have hcjclean : cj.tag = Tag.clean := by
      rcases Tag.clean_or_stale cj.tag with h | h
      · exact h
      · exact absurd ⟨cj, hcj, h⟩ (hfs.2 j hji)
    obtain ⟨hwit, hrcj⟩ := hwf j ⟨cj, hcj, hcjclean⟩
    replace hwit : Witnessed nb.cells nb.store j := hwit
    replace hrcj : RerunConsistent nb.cells j := hrcj
    obtain ⟨cj', hcj', oj, σ''', hout, hevalj, _⟩ := hwit
    rw [hcj] at hcj'; cases hcj'
    -- the new cell at j keeps code/out/reads/writes
    obtain ⟨t, hct, _⟩ := hretag j (by omega) cj hcj
    -- the new store agrees with the old one on cell j's reads
    -- (NoWriteAfterRead check of the run: w avoids reads above i)
    have hreads_csm_i : ∀ ℓ, w ℓ →
        WritesAt (nb.cells.set i
          { ci with out := some o, tag := .clean, reads := r, writes := w }) i ℓ := by
      intro ℓ hw
      exact ⟨{ ci with out := some o, tag := .clean, reads := r, writes := w },
        by simp [List.getElem?_set_self hi], hw⟩
    have hagree : ∀ ℓ, cj.reads ℓ → σ' ℓ = nb.store ℓ := by
      intro ℓ hr
      refine CellEval.frame heval ℓ (fun hw => ?_)
      exact hrc.noWriteAfterRead ℓ (hreads_csm_i ℓ hw)
        ⟨j, hji, ⟨cj, by rw [List.getElem?_set_ne (by omega)]; exact hcj, hr⟩⟩
    obtain ⟨τ', hevalj', _, _⟩ := CellEval.locality hevalj hagree
    -- assemble the recorded-behavior rerun in the new state
    refine ⟨{ cj with tag := t }, hct, oj, τ', hout, hevalj', ?_⟩
    -- rerun consistency in the new tables
    have hequiv : RWEquiv cs'
        (nb.cells.set i { ci with out := some o, tag := .clean, reads := r, writes := w }) := by
      constructor <;> intro k ℓ
      · by_cases hki : k = i
        · subst hki
          constructor <;> rintro ⟨c, hc, hcr⟩
          · rw [hat] at hc; cases hc
            exact ⟨{ ci with out := some o, tag := .clean, reads := r, writes := w },
              by simp [List.getElem?_set_self hi], hcr⟩
          · rw [List.getElem?_set_self hi] at hc; cases hc
            exact ⟨_, hat, hcr⟩
        · constructor <;> rintro ⟨c, hc, hcr⟩
          · have hklt : k < nb.cells.length := by
              have := lt_length_of_getElem?_eq_some hc; omega
            obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hklt
            obtain ⟨t', hct', _⟩ := hretag k hki c₀ hc₀
            rw [hct'] at hc; cases hc
            exact ⟨c₀, by rw [List.getElem?_set_ne (by omega)]; exact hc₀, hcr⟩
          · rw [List.getElem?_set_ne (by omega)] at hc
            obtain ⟨t', hct', _⟩ := hretag k hki c hc
            exact ⟨{ c with tag := t' }, hct', hcr⟩
      · by_cases hki : k = i
        · subst hki
          constructor <;> rintro ⟨c, hc, hcw⟩
          · rw [hat] at hc; cases hc
            exact ⟨{ ci with out := some o, tag := .clean, reads := r, writes := w },
              by simp [List.getElem?_set_self hi], hcw⟩
          · rw [List.getElem?_set_self hi] at hc; cases hc
            exact ⟨_, hat, hcw⟩
        · constructor <;> rintro ⟨c, hc, hcw⟩
          · have hklt : k < nb.cells.length := by
              have := lt_length_of_getElem?_eq_some hc; omega
            obtain ⟨c₀, hc₀⟩ := exists_getElem?_eq_some hklt
            obtain ⟨t', hct', _⟩ := hretag k hki c₀ hc₀
            rw [hct'] at hc; cases hc
            exact ⟨c₀, by rw [List.getElem?_set_ne (by omega)]; exact hc₀, hcw⟩
          · rw [List.getElem?_set_ne (by omega)] at hc
            obtain ⟨t', hct', _⟩ := hretag k hki c hc
            exact ⟨{ c with tag := t' }, hct', hcw⟩
    exact hequiv.symm.rerunConsistent j
      (rerunConsistent_below_of_run hji hi hrc hrcj)

end Backward

/-! ## The algorithm and its termination -/

section Strategy
variable [CellEval Code Output L V]

/-- An execution of the `RunToClean` algorithm that reports no
potential non-termination.  `F` is the complement of the algorithm's
`E`: the positions not yet executed.  A `first` step executes a cell
for the first time (any successful `[Inst-Run]` step); a `rerun` step
executes a cell again and must leave every position `≤ i` clean — the
algorithm's check that the run marked no earlier cell stale, stated
directly on the successor state. -/
inductive RunToClean :
    List Nat → Notebook Code Output L V → Notebook Code Output L V → List Nat → Prop where
  | refl {F : List Nat} {nb : Notebook Code Output L V} : RunToClean F nb nb F
  | first {F F' : List Nat} {nb nb₁ nb' : Notebook Code Output L V} {i : Nat}
      (hfs : FirstStale nb.cells i) (hmem : i ∈ F)
      (hstep : InstStep nb (.run i) nb₁)
      (hrest : RunToClean (F.erase i) nb₁ nb' F') : RunToClean F nb nb' F'
  | rerun {F F' : List Nat} {nb nb₁ nb' : Notebook Code Output L V} {i : Nat}
      (hfs : FirstStale nb.cells i) (hmem : i ∉ F)
      (hstep : InstStep nb (.run i) nb₁)
      (hclean : ∀ j, j ≤ i → IsClean nb₁.cells j)
      (hrest : RunToClean F nb₁ nb' F') : RunToClean F nb nb' F'

/-- The algorithm cannot continue without a report: the first stale
cell admits no `[Inst-Run]` step at all (`report error`), or it has
already been executed and every available run marks a cell at or
before it stale (`report potential non-termination`).  The second
disjunct also covers an executed cell with no run at all, vacuously. -/
def RunToCleanStuck (nb : Notebook Code Output L V) (F : List Nat) : Prop :=
  ∃ i, FirstStale nb.cells i ∧
    ((i ∈ F ∧ ∀ nb', ¬ InstStep nb (.run i) nb') ∨
     (i ∉ F ∧ ∀ nb', InstStep nb (.run i) nb' →
        ∃ j, j ≤ i ∧ ¬ IsClean nb'.cells j))

/-- **Theorem 2.5 (Progress).**  From any well-formed state, every
report-free execution of the algorithm can be extended to reach a
state that is well-formed and either all-clean or stuck.  Termination
is by the lexicographic measure `(|F|, n − first-stale position)`. -/
theorem progress {nb : Notebook Code Output L V}
    (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', RunToClean F nb nb' F' ∧ WellFormed nb' ∧
      (AllClean nb'.cells ∨ RunToCleanStuck nb' F') := by
  classical
  -- strong induction on the measure |F|·(n+1) + (n − first-stale position)
  suffices H : ∀ μ : Nat, ∀ nb : Notebook Code Output L V, ∀ F : List Nat,
      WellFormed nb →
      (∀ i, FirstStale nb.cells i →
        F.length * (nb.cells.length + 1) + (nb.cells.length - i) < μ) →
      ∃ nb' F', RunToClean F nb nb' F' ∧ WellFormed nb' ∧
        (AllClean nb'.cells ∨ RunToCleanStuck nb' F') by
    refine H (F.length * (nb.cells.length + 1) + nb.cells.length + 1) nb F hwf ?_
    intro i _
    have hK : F.length * (nb.cells.length + 1) = F.length * (nb.cells.length + 1) := rfl
    generalize F.length * (nb.cells.length + 1) = K at *
    omega
  intro μ
  induction μ using Nat.strongRecOn with
  | ind μ ih =>
    intro nb F hwf hbound
    rcases allClean_or_firstStale nb.cells with hac | ⟨i, hfs⟩
    · exact ⟨nb, F, .refl, hwf, .inl hac⟩
    · have hiN : i < nb.cells.length := by
        obtain ⟨c, hc, _⟩ := hfs.1
        exact lt_length_of_getElem?_eq_some hc
      by_cases hmem : i ∈ F
      · -- first execution: any successful run
        by_cases hrun : ∃ nb₁, InstStep nb (.run i) nb₁
        · obtain ⟨nb₁, hstep⟩ := hrun
          have hwf₁ : WellFormed nb₁ := preservation hwf hstep
          have hN₁ : nb₁.cells.length = nb.cells.length := instStep_run_length hstep
          have hFlen : (F.erase i).length + 1 = F.length := by
            have h0 : 0 < F.length := List.length_pos_of_mem hmem
            rw [List.length_erase_of_mem hmem]
            omega
          -- new measure
          obtain ⟨nb', F', hexec, hwf', hend⟩ :=
            ih ((F.erase i).length * (nb₁.cells.length + 1) + nb₁.cells.length + 1)
              (by
                -- (|F|−1)(n+1) + n + 1 = |F|(n+1) ≤ measure at i < μ
                have hb := hbound i hfs
                rw [hN₁]
                have hmul : F.length * (nb.cells.length + 1) =
                    (F.erase i).length * (nb.cells.length + 1) + (nb.cells.length + 1) := by
                  rw [← hFlen, Nat.succ_mul]
                generalize hg : (F.erase i).length * (nb.cells.length + 1) = A at *
                omega)
              nb₁ (F.erase i) hwf₁
              (by
                intro p hp
                have hpN : p < nb₁.cells.length := by
                  obtain ⟨c, hc, _⟩ := hp.1
                  exact lt_length_of_getElem?_eq_some hc
                generalize (F.erase i).length * (nb₁.cells.length + 1) = A
                omega)
          exact ⟨nb', F', .first hfs hmem hstep hexec, hwf', hend⟩
        · -- no run at all: stuck (report error)
          refine ⟨nb, F, .refl, hwf, .inr ⟨i, hfs, .inl ⟨hmem, ?_⟩⟩⟩
          intro nb' h
          exact hrun ⟨nb', h⟩
      · -- already executed: the rerun must mark nothing at or before i
        by_cases hrep : ∃ nb₁, InstStep nb (.run i) nb₁ ∧
            ∀ j, j ≤ i → IsClean nb₁.cells j
        · obtain ⟨nb₁, hstep, hprefix⟩ := hrep
          have hwf₁ : WellFormed nb₁ := preservation hwf hstep
          have hN₁ : nb₁.cells.length = nb.cells.length :=
            instStep_run_length hstep
          obtain ⟨nb', F', hexec, hwf', hend⟩ :=
            ih (F.length * (nb₁.cells.length + 1) + (nb₁.cells.length - i))
              (by
                have hb := hbound i hfs
                rw [hN₁]
                exact hb)
              nb₁ F hwf₁
              (by
                intro p hp
                -- the clean prefix grew: p > i
                have hpi : i < p := by
                  rcases Nat.lt_or_ge i p with h | h
                  · exact h
                  · exact absurd hp.1 (hprefix p (by omega)).not_stale
                have hpN : p < nb₁.cells.length := by
                  obtain ⟨c, hc, _⟩ := hp.1
                  exact lt_length_of_getElem?_eq_some hc
                generalize F.length * (nb₁.cells.length + 1) = A
                omega)
          exact ⟨nb', F', .rerun hfs hmem hstep hprefix hexec, hwf', hend⟩
        · -- every available run marks a cell at or before i stale (the
          -- report), or no run exists at all (vacuous)
          refine ⟨nb, F, .refl, hwf, .inr ⟨i, hfs, .inr ⟨hmem, ?_⟩⟩⟩
          intro nb' hstep
          exact Classical.byContradiction fun hno =>
            hrep ⟨nb', hstep, fun j hj =>
              Classical.byContradiction fun hc => hno ⟨j, hj, hc⟩⟩

/-- The algorithm as written starts with `E = ∅`: every position may
still be executed for the first time. -/
theorem progress_init {nb : Notebook Code Output L V} (hwf : WellFormed nb) :
    ∃ nb' F', RunToClean (List.range nb.cells.length) nb nb' F' ∧ WellFormed nb' ∧
      (AllClean nb'.cells ∨ RunToCleanStuck nb' F') :=
  progress hwf _

/-- Progress combined with Theorem 2.4: the algorithm terminates either
in a *reproducible* notebook or at a cell where it reports. -/
theorem progress_reproducible {nb : Notebook Code Output L V}
    (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', RunToClean F nb nb' F' ∧
      (Reproducible nb'.erase ∨ RunToCleanStuck nb' F') := by
  obtain ⟨nb', F', hexec, hwf', hend⟩ := progress hwf F
  refine ⟨nb', F', hexec, ?_⟩
  rcases hend with hac | hstuck
  · exact .inl (output_consistency hwf' hac)
  · exact .inr hstuck

end Strategy

end FlowBook
