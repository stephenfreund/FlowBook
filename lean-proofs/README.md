# Lean Formalization of the FlowBook Formal Development

This directory contains a complete [Lean 4](https://lean-lang.org)
formalization of the mathematical development in the FlowBook paper:
the notebook semantics (`semantics-col.tex`), the dynamic analysis for
reproducibility / output consistency (`analysis-col.tex`,
`flowbook-analysis.tex`), and the three correctness theorems with the
full proofs from the appendix (`appendix.tex`, `supplemental.tex`).

It also includes an **executable reference kernel** (`FlowBook/Exec.lean`)
that is proved sound against the relational semantics and `#eval`s the
paper's litmus tests (see the dedicated section below).

All proofs are complete and machine-checked: there are **no `sorry`s
and no custom axioms**.  Every theorem depends only on Lean's three
standard axioms (`propext`, `Classical.choice`, `Quot.sound`), as
reported by `#print axioms`.

## Building

The project uses Lake with Lean `4.30.0` (pinned in `lean-toolchain`)
and has **no external dependencies** (no Mathlib — core Lean only):

```bash
cd lean-proofs
lake build
```

## Files and correspondence to the paper

| Lean file | Paper source | Contents |
|---|---|---|
| `FlowBook/Semantics.lean` | §Semantics (`semantics-col.tex`) | Locations `ℓ ::= x \| d.c` (`Loc`), stores, black-box cell evaluation (`CellEval`), notebook operations (`Op`), the standard semantics of Fig. `fig:std-semantics` (`StdStep`), top-to-bottom execution (`Runs`), Def. *Reproducible / Output-Consistent State* (`Reproducible`) |
| `FlowBook/Analysis.lean` | §Analysis (`analysis-col.tex`) | Instrumentation `I = (T, R, W)` (fused into `Cell` records), Def. *Rerun Consistent Accesses* (`RerunConsistent`), `ForwardStale`/`LastWriter`/`BackwardStale` (`FwdStale`, `IsLastWriter`, `BwdStale`), the instrumented semantics of Fig. `fig:inst-semantics` (`InstStep`), Def. *Well-Formed State* (`WellFormed`), and the paper's initial-state observation (`wellFormed_initial`) |
| `FlowBook/Preservation.lean` | Theorem 1 + appendix proof | `preservation`: well-formedness is preserved by every operation, by cases `[Inst-Run]` (subcases `j < i`, `j = i`, `j > i`), `[Inst-Edit]`, `[Inst-Insert]`, `[Inst-Delete]`, `[Inst-Move]` |
| `FlowBook/OutputConsistency.lean` | Theorem 2 + appendix proof | `output_consistency`: well-formed all-clean states are reproducible, by the paper's induction on prefixes with the agreement invariant on `(⋃ W_{1..i}) \ (⋃ W_{i+1..n})` |
| `FlowBook/Progress.lean` | Theorem 3 + appendix proof | `progress`, `progress_init`, `progress_reproducible`: every report-free execution of the `RunToClean` algorithm terminates in an all-clean (hence reproducible) state or at a report; `canRerun_after_run`: the paper's key claim that cells marked stale by `BackwardStale` can be re-run reproducing their recorded behavior |
| `FlowBook/Erasure.lean` | Figs. `fig:std-semantics` / `fig:inst-semantics` | `instStep_erase`: the instrumented semantics refines the standard semantics (erasing `I` from an instrumented step yields a standard step) |
| `FlowBook/Examples.lean` | — | A concrete model (assignments and copies) satisfying the evaluation axioms, showing the axiomatization is consistent; instantiations of all three theorems at that model |
| `FlowBook/Exec.lean` | Figs. `fig:litmus-tests` / `fig:staleness-litmus-tests` | An **executable** reference kernel: a concrete cell language with a functional evaluator, decidable rerun-consistency and staleness, functional operations (`runCell`/`editCell`/`insertCell`/`deleteCell`), soundness theorems tying each accepted operation to a real `InstStep`, and `#eval` demos replaying every litmus test |

## The executable reference kernel (`FlowBook/Exec.lean`)

The metatheory above models cell evaluation as a *relation* and
read/write sets as *predicates* `L → Prop` — faithful to the paper, but
not runnable.  `Exec.lean` instantiates the whole framework with a
concrete, `#eval`-able kernel and proves it *sound* against the
relational semantics, so the executable checker and the proved theorems
are the same object.

* **Concrete language.**  `LCmd` is a small deterministic language
  (`const`/`copy`/`use`/`incr`) with a functional evaluator `evalCmd`
  that returns the output, new store, and the read/write sets *as finite
  lists*.  `instLCellEval` proves this satisfies the `frame` and
  `locality` axioms, so it is a legal `CellEval`.
* **Executable notebooks.**  `ENotebook` carries list-backed read/write
  sets; `ENotebook.toNb` reflects it to the relational `Notebook`.
* **Decidable analysis.**  The four rerun-consistency predicates and the
  forward/backward staleness predicates are computed by Boolean
  functions, each proved to reflect its relational counterpart
  (`writtenAboveB_iff`, `writtenBelowB_iff`, `fwdStaleB_iff`,
  `bwdStaleB_iff`, `markedB_iff`).  Because the staleness tags shown by
  the demos are computed by these *proved-correct* functions, the tags
  are exactly the ones the soundness theorem certifies.
* **Soundness.**  `runCell_sound`, `editCell_sound`, `insertCell_sound`,
  and `deleteCell_sound` each prove that when the executable operation
  accepts, it realizes a genuine `InstStep` on the reflected notebooks.
  `runOps_wellFormed` chains this with `preservation`, and
  `mkNb_runOps_reproducible` gives the end-to-end statement: starting
  from a freshly built (all-stale, hence well-formed) notebook, if the
  kernel accepts an operation sequence and the final notebook is all
  clean, then it is reproducible.
* **Litmus tests.**  The `#eval` block at the end replays every scenario
  from Figures `fig:litmus-tests` and `fig:staleness-litmus-tests`.  The
  four rerun-consistency scenarios each print the exact violation the
  paper labels (`noReadAndWrite`, `writeBeforeRead`, `noReadBeforeWrite`,
  `noWriteAfterRead`); the staleness scenarios print the resulting
  per-cell tags — forward staleness `[clean, stale]` and the combined
  backward+forward scenario `[stale, clean, stale]`.  Run them with:

  ```bash
  lake env lean FlowBook/Exec.lean
  ```

Because `runCell` reports the *first* violation it finds, `diagnose`
checks `NoReadBeforeWrite` before `WriteBeforeRead` so that a top cell
reading a below-written location is reported as the paper labels it
(both predicates fail in that case).  This ordering is invisible to the
soundness proof, which only uses the "no violation" case.

## Modeling decisions

* **Black-box cell evaluation.**  The paper treats cell evaluation
  `c ; Σ ⇓ o · Σ' · r · w` as a black box over the language runtime and
  notes it is a *relation* (cells may be non-deterministic).  The
  formalization axiomatizes exactly the two properties the paper's
  proofs use, as fields of the `CellEval` class:
  - `frame` — the store is unchanged outside the write set `w` (this is
    the definition of `w`: "the set of write locations where `Σ'` is
    updated from `Σ`");
  - `locality` — evaluation depends only on the locations read: from
    any store agreeing on `r`, the same execution (same output, same
    read and write sets, same written values) is available.  This
    property is stated in a remark in `supplemental.tex` and is what
    realizes the commuting diagrams in the appendix proofs.

  `FlowBook/Examples.lean` proves both axioms hold for a concrete
  language, so the axiomatization is non-vacuous.

* **State representation.**  The paper's `S · I = (C, O, Σ) · (T, R, W)`
  is six parallel sequences; the formalization fuses the five per-cell
  components into a single `List Cell` (fields `code`, `out`, `tag`,
  `reads`, `writes`), which keeps them synchronized by construction.
  `Notebook.erase` projects back to the standard state `(C, O, Σ)`.
  `O_i = ⊥` is `out = none`.  Read/write sets are predicates
  `L → Prop`.  Cell positions are 0-based (the paper is 1-based).

* **Relational tag updates.**  The `T'` updates of `[Inst-Run]` and
  `[Inst-Delete]` are specified pointwise (`RetagSpec`) rather than
  computed, because the staleness predicates are not decidable for a
  black-box evaluation relation; `retag_exists` shows (classically)
  the specification is always satisfiable, so the rules never fail for
  want of a retagging.

* **`Move` as composition.**  Exactly as in both figures,
  `Move(s, d)` is a `Delete` followed by an `Insert`, so its
  preservation case is the composition of the other two.

## The three theorems

```lean
theorem preservation (hwf : WellFormed nb) (hstep : InstStep nb op nb') :
    WellFormed nb'

theorem output_consistency (hwf : WellFormed nb) (hclean : AllClean nb.cells) :
    Reproducible nb.erase

theorem progress (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', RunToClean F nb nb' F' ∧ WellFormed nb' ∧
      (AllClean nb'.cells ∨ RunToCleanStuck nb' F')
```

## Notes on Theorem 3 (Progress)

Theorem 3 is stated for the `RunToClean` algorithm:

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

The check is necessary because cell evaluation is non-deterministic:
without it, three cells suffice for a non-terminating execution in
which every run succeeds (let cells 2 and 3 each alternate between
writing `{a}` and `{b}`; each run of one backward-marks the other via
`LastWriter`).  Staleness moves toward earlier cells only via
`BackwardStale` (a run dropping a write owned by an earlier cell), so
requiring reruns to mark nothing before themselves is the termination
invariant itself, checked directly.

The formalization models the report-free executions: `RunToClean`
carries `F`, the complement of the algorithm's `E` (initially all
positions); a `first` step executes a cell for the first time with any
successful `[Inst-Run]`, and a `rerun` step is an `[Inst-Run]` step
that leaves every position `≤ i` clean — the algorithm's check, stated
on the successor state.

* `progress` proves every report-free execution **terminates**, by the
  paper's measure: a first execution shrinks `F`, and a rerun keeps
  the prefix clean by its side condition, so the clean prefix strictly
  grows — at most `n` first executions and at most `n` reruns between
  them.
* `canRerun_after_run` and `cleanPrefix_run_exists` prove the paper's
  supporting claim behind the Stability lemma: after any successful
  run of the first stale cell `i`, every cell `j < i` — in particular
  every cell marked by `BackwardStale` — can re-run from the new store
  reproducing its recorded output, read set, and write set
  (`MatchingRunAt`), and such a run marks nothing before itself,
  satisfying the rerun condition, because the run's `NoWriteAfterRead`
  check keeps the new writes away from those cells' read sets.  Hence
  such reruns need never trigger the report; for deterministic
  runtimes no rerun does, so the report never fires.

## Verifying the axiom footprint

```bash
lake env lean -q <(echo 'import FlowBook
#print axioms FlowBook.preservation
#print axioms FlowBook.output_consistency
#print axioms FlowBook.progress')
```

Each reports `[propext, Classical.choice, Quot.sound]` — Lean's
standard axioms only.
