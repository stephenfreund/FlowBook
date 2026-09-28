/-
# The Instrumented Semantics Refines the Standard Semantics

Figure `fig:inst-semantics` defines the instrumented rules on top of
the standard rules of Figure `fig:std-semantics`: `[Inst-Edit]`,
`[Inst-Insert]`, and `[Inst-Delete]` take a standard step `S ─op→ S'`
as a premise, and `[Inst-Run]` extends `[Std-Run]` with the
instrumented evaluation judgment.  This file makes that relationship a
theorem: erasing the instrumentation from an instrumented step yields a
standard step,

  `S · I ⟹op S' · I'   implies   ⌊S · I⌋ ─op→ ⌊S' · I'⌋`

where `⌊·⌋ = Notebook.erase` projects out `(C, O, Σ)`.  In particular,
the analysis only constrains *which* standard behaviors are allowed; it
never invents new ones.
-/
import FlowBook.Analysis

namespace FlowBook

variable {Code Output L V : Type}

private theorem erase_cells (nb : Notebook Code Output L V) :
    nb.erase.cells = nb.cells.map fun c => (c.code, c.out) := rfl

variable [CellEval Code Output L V]

open CellEval

/-- Erasing the instrumentation from an instrumented step yields a
standard step on the erased states. -/
theorem instStep_erase {nb nb' : Notebook Code Output L V} {op : Op Code}
    (h : InstStep nb op nb') : StdStep nb.erase op nb'.erase := by
  induction h with
  | @run nb i ci o σ' r w cs' hcell heval hrc hlen hat hretag =>
    have hi : i < nb.cells.length := lt_length_of_getElem?_eq_some hcell
    show StdStep nb.erase (.run i) ⟨cs'.map fun c => (c.code, c.out), σ'⟩
    have hcells : (cs'.map fun c => (c.code, c.out)) =
        nb.erase.cells.set i (ci.code, some o) := by
      apply List.ext_getElem?
      intro j
      by_cases hji : j = i
      · subst hji
        rw [List.getElem?_map, hat,
          List.getElem?_set_self (by simpa [erase_cells] using hi)]
        rfl
      · rw [List.getElem?_map, List.getElem?_set_ne (by omega), erase_cells,
          List.getElem?_map]
        rcases Nat.lt_or_ge j nb.cells.length with hj | hj
        · obtain ⟨cj, hcj⟩ := exists_getElem?_eq_some hj
          obtain ⟨t, hct, _⟩ := hretag j hji cj hcj
          rw [hcj, hct]
          rfl
        · rw [List.getElem?_eq_none (l := cs') (by omega),
            List.getElem?_eq_none (l := nb.cells) (by omega)]
    rw [hcells]
    refine StdStep.run (c := ci.code) (o₀ := ci.out) ?_ ⟨r, w, heval⟩
    rw [erase_cells, List.getElem?_map, hcell]
    rfl
  | @edit nb i ci c hcell =>
    have hi : i < nb.cells.length := lt_length_of_getElem?_eq_some hcell
    show StdStep nb.erase (.edit i c)
      ⟨(nb.cells.set i { ci with code := c, tag := .stale }).map
        fun c => (c.code, c.out), nb.store⟩
    have hcells :
        ((nb.cells.set i { ci with code := c, tag := .stale }).map
          fun c => (c.code, c.out)) =
        nb.erase.cells.set i (c, ci.out) := by
      apply List.ext_getElem?
      intro j
      by_cases hji : j = i
      · subst hji
        rw [List.getElem?_map, List.getElem?_set_self hi,
          List.getElem?_set_self (by simpa [erase_cells] using hi)]
        rfl
      · rw [List.getElem?_map, List.getElem?_set_ne (by omega),
          List.getElem?_set_ne (by omega), erase_cells, List.getElem?_map]
    rw [hcells]
    refine StdStep.edit (c₀ := ci.code) (o₀ := ci.out) ?_
    rw [erase_cells, List.getElem?_map, hcell]
    rfl
  | @insert nb i c hle =>
    show StdStep nb.erase (.insert i c)
      ⟨(nb.cells.insertIdx i ⟨c, none, .stale, fun _ => False, fun _ => False⟩).map
        fun c => (c.code, c.out), nb.store⟩
    have hcells :
        ((nb.cells.insertIdx i ⟨c, none, .stale, fun _ => False, fun _ => False⟩).map
          fun c => (c.code, c.out)) =
        nb.erase.cells.insertIdx i (c, none) := by
      apply List.ext_getElem?
      intro j
      rcases Nat.lt_or_ge j i with hj | hj
      · rw [List.getElem?_map, List.getElem?_insertIdx_of_lt hj,
          List.getElem?_insertIdx_of_lt hj, erase_cells, List.getElem?_map]
      · rcases Nat.lt_or_ge i j with hj' | hj'
        · rw [List.getElem?_map, List.getElem?_insertIdx_of_gt hj',
            List.getElem?_insertIdx_of_gt hj', erase_cells, List.getElem?_map]
        · have hji : j = i := by omega
          subst hji
          rw [List.getElem?_map, List.getElem?_insertIdx_self,
            List.getElem?_insertIdx_self, erase_cells]
          simp [hle]
    rw [hcells]
    exact StdStep.insert (by simpa [erase_cells] using hle)
  | @delete nb i ci cs'' hcell hlen hretag =>
    show StdStep nb.erase (.delete i)
      ⟨(cs''.eraseIdx i).map fun c => (c.code, c.out), nb.store⟩
    have hcells : ((cs''.eraseIdx i).map fun c => (c.code, c.out)) =
        nb.erase.cells.eraseIdx i := by
      apply List.ext_getElem?
      intro j
      rcases Nat.lt_or_ge j i with hj | hj
      · rw [List.getElem?_map, List.getElem?_eraseIdx_of_lt hj,
          List.getElem?_eraseIdx_of_lt hj, erase_cells, List.getElem?_map]
        rcases Nat.lt_or_ge j nb.cells.length with hjn | hjn
        · obtain ⟨cj, hcj⟩ := exists_getElem?_eq_some hjn
          obtain ⟨t, hct, _⟩ := hretag j (by omega) cj hcj
          rw [hcj, hct]
          rfl
        · rw [List.getElem?_eq_none (l := cs'') (by omega),
            List.getElem?_eq_none (l := nb.cells) (by omega)]
      · rw [List.getElem?_map, List.getElem?_eraseIdx_of_ge hj,
          List.getElem?_eraseIdx_of_ge hj, erase_cells, List.getElem?_map]
        rcases Nat.lt_or_ge (j + 1) nb.cells.length with hjn | hjn
        · obtain ⟨cj, hcj⟩ := exists_getElem?_eq_some hjn
          have hji : j + 1 ≠ i := by omega
          obtain ⟨t, hct, _⟩ := hretag (j + 1) hji cj hcj
          rw [hcj, hct]
          rfl
        · rw [List.getElem?_eq_none (l := cs'') (by omega),
            List.getElem?_eq_none (l := nb.cells) (by omega)]
    rw [hcells]
    refine StdStep.delete (c₀ := ci.code) (o₀ := ci.out) ?_
    rw [erase_cells, List.getElem?_map, hcell]
    rfl
  | @moveDown nb nb'' nb' s d cs_ hlt hcell h1 h2 ih1 ih2 =>
    refine StdStep.moveDown (c₀ := cs_.code) (o₀ := cs_.out) hlt ?_ ih1 ih2
    rw [erase_cells, List.getElem?_map, hcell]
    rfl
  | @moveUp nb nb'' nb' s d cs_ hlt hcell h1 h2 ih1 ih2 =>
    refine StdStep.moveUp (c₀ := cs_.code) (o₀ := cs_.out) hlt ?_ ih1 ih2
    rw [erase_cells, List.getElem?_map, hcell]
    rfl

end FlowBook
