# Lean Formalization of the FlowBook Formal Development

This directory contains a complete [Lean 4](https://lean-lang.org)
formalization of the mathematical development in the FlowBook paper:
the notebook semantics, the dynamic analysis for reproducibility /
output consistency, and the three correctness theorems with the full
proofs from the supplemental material.

It also includes an **executable reference kernel** (`FlowBook/Exec.lean`)
that is proved sound against the relational semantics and `#eval`s the
paper's litmus tests (see the dedicated section below).

All proofs are complete and machine-checked: there are **no `sorry`s
and no custom axioms**. Every theorem depends only on Lean's three
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

| Lean file                         | Paper source                                                 | Contents                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FlowBook/Semantics.lean`         | §Semantics                                                   | Locations `ℓ ::= x \| d.c` (`Loc`), stores, black-box cell evaluation (`CellEval`), notebook operations (`Op`), the standard semantics of Fig. 8 (`StdStep`), top-to-bottom execution (`Runs`), Def. _Reproducible / Output-Consistent State_ (`Reproducible`)                                                                                                        |
| `FlowBook/Analysis.lean`          | §Analysis                                                    | Instrumentation `I = (T, R, W)` (fused into `Cell` records), Def. _Rerun Consistent Accesses_ (`RerunConsistent`), `ForwardStale`/`LastWriter`/`BackwardStale` (`FwdStale`, `IsLastWriter`, `BwdStale`), the instrumented semantics of Fig. 9 (`InstStep`), Def. _Well-Formed State_ (`WellFormed`), and the paper's initial-state observation (`wellFormed_initial`) |
| `FlowBook/Preservation.lean`      | Theorem 2.3 of the supplement, proof in §3 of the supplement | `preservation`: well-formedness is preserved by every operation, by cases `[Inst-Run]` (subcases `j < i`, `j = i`, `j > i`), `[Inst-Edit]`, `[Inst-Insert]`, `[Inst-Delete]`, `[Inst-Move]`                                                                                                                                                                           |
| `FlowBook/OutputConsistency.lean` | Theorem 2.4 of the supplement, proof in §4 of the supplement | `output_consistency`: well-formed all-clean states are reproducible, by the paper's induction on prefixes with the agreement invariant on `(⋃ W_{1..i}) \ (⋃ W_{i+1..n})`                                                                                                                                                                                             |
| `FlowBook/Progress.lean`          | Theorem 2.5 of the supplement, proof in §5 of the supplement | `progress`, `progress_init`, `progress_reproducible`: every execution of the first-stale strategy (`FirstStaleExec`), under the paper's rerun assumption, terminates in an all-clean (hence reproducible) state or at a stuck cell; `canRerun_after_run`: the assumption is satisfiable for cells marked stale by `BackwardStale`                                        |
| `FlowBook/Erasure.lean`           | Figs. 8 / 9                                                  | `instStep_erase`: the instrumented semantics refines the standard semantics (erasing `I` from an instrumented step yields a standard step)                                                                                                                                                                                                                            |
| `FlowBook/Examples.lean`          | —                                                            | A concrete model (assignments and copies) satisfying the evaluation axioms, showing the axiomatization is consistent; instantiations of all three theorems at that model                                                                                                                                                                                              |
| `FlowBook/Exec.lean`              | Figs. 3 / 4                                                  | An **executable** reference kernel: a concrete cell language with a functional evaluator, decidable rerun-consistency and staleness, functional operations (`runCell`/`editCell`/`insertCell`/`deleteCell`), soundness theorems tying each accepted operation to a real `InstStep`, and `#eval` demos replaying every litmus test                                     |

## The executable reference kernel (`FlowBook/Exec.lean`)

The metatheory above models cell evaluation as a _relation_ and
read/write sets as _predicates_ `L → Prop` — faithful to the paper, but
not runnable. `Exec.lean` instantiates the whole framework with a
concrete, `#eval`-able kernel and proves it _sound_ against the
relational semantics, so the executable checker and the proved theorems
are the same object.

- **Concrete language.** `LCmd` is a small deterministic language
  (`const`/`copy`/`use`/`incr`) with a functional evaluator `evalCmd`
  that returns the output, new store, and the read/write sets _as finite
  lists_. `instLCellEval` proves this satisfies the `frame` and
  `locality` axioms, so it is a legal `CellEval`.
- **Executable notebooks.** `ENotebook` carries list-backed read/write
  sets; `ENotebook.toNb` reflects it to the relational `Notebook`.
- **Decidable analysis.** The four rerun-consistency predicates and the
  forward/backward staleness predicates are computed by Boolean
  functions, each proved to reflect its relational counterpart
  (`writtenAboveB_iff`, `writtenBelowB_iff`, `fwdStaleB_iff`,
  `bwdStaleB_iff`, `markedB_iff`). Because the staleness tags shown by
  the demos are computed by these _proved-correct_ functions, the tags
  are exactly the ones the soundness theorem certifies.
- **Soundness.** `runCell_sound`, `editCell_sound`, `insertCell_sound`,
  and `deleteCell_sound` each prove that when the executable operation
  accepts, it realizes a genuine `InstStep` on the reflected notebooks.
  `runOps_wellFormed` chains this with `preservation`, and
  `mkNb_runOps_reproducible` gives the end-to-end statement: starting
  from a freshly built (all-stale, hence well-formed) notebook, if the
  kernel accepts an operation sequence and the final notebook is all
  clean, then it is reproducible.
- **Litmus tests.** The `#eval` block at the end replays every scenario
  from Figures 3 and 4. The
  four rerun-consistency scenarios each print the exact violation the
  paper labels (`noReadAndWrite`, `writeBeforeRead`, `noReadBeforeWrite`,
  `noWriteAfterRead`); the staleness scenarios print the resulting
  per-cell tags — forward staleness `[clean, stale]` and the combined
  backward+forward scenario `[stale, clean, stale]`. Run them with:

  ```bash
  lake env lean FlowBook/Exec.lean
  ```

Because `runCell` reports the _first_ violation it finds, `diagnose`
checks `NoReadBeforeWrite` before `WriteBeforeRead` so that a top cell
reading a below-written location is reported as the paper labels it
(both predicates fail in that case). This ordering is invisible to the
soundness proof, which only uses the "no violation" case.

## Modeling decisions

- **Black-box cell evaluation.** The paper treats cell evaluation
  `c ; Σ ⇓ o · Σ' · r · w` as a black box over the language runtime and
  notes it is a _relation_ (cells may be non-deterministic). The
  formalization axiomatizes exactly the two properties the paper's
  proofs use, as fields of the `CellEval` class:
  - `frame` — the store is unchanged outside the write set `w` (this is
    the definition of `w`: "the set of write locations where `Σ'` is
    updated from `Σ`");
  - `locality` — evaluation depends only on the locations read: from
    any store agreeing on `r`, the same execution (same output, same
    read and write sets, same written values) is available. This
    property is stated in a remark in the paper and is what
    realizes the commuting diagrams in the proofs of §§3–5 of the supplement.

  `FlowBook/Examples.lean` proves both axioms hold for a concrete
  language, so the axiomatization is non-vacuous.

- **State representation.** The paper's `S · I = (C, O, Σ) · (T, R, W)`
  is six parallel sequences; the formalization fuses the five per-cell
  components into a single `List Cell` (fields `code`, `out`, `tag`,
  `reads`, `writes`), which keeps them synchronized by construction.
  `Notebook.erase` projects back to the standard state `(C, O, Σ)`.
  `O_i = ⊥` is `out = none`. Read/write sets are predicates
  `L → Prop`. Cell positions are 0-based (the paper is 1-based).

- **Relational tag updates.** The `T'` updates of `[Inst-Run]` and
  `[Inst-Delete]` are specified pointwise (`RetagSpec`) rather than
  computed, because the staleness predicates are not decidable for a
  black-box evaluation relation; `retag_exists` shows (classically)
  the specification is always satisfiable, so the rules never fail for
  want of a retagging.

- **`Move` as composition.** Exactly as in both figures,
  `Move(s, d)` is a `Delete` followed by an `Insert`, so its
  preservation case is the composition of the other two.

## The three theorems

```lean
theorem preservation (hwf : WellFormed nb) (hstep : InstStep nb op nb') :
    WellFormed nb'

theorem output_consistency (hwf : WellFormed nb) (hclean : AllClean nb.cells) :
    Reproducible nb.erase

theorem progress (hwf : WellFormed nb) (F : List Nat) :
    ∃ nb' F', FirstStaleExec F nb nb' F' ∧ WellFormed nb' ∧
      (AllClean nb'.cells ∨ FirstStaleStuck nb' F')
```

## Notes on Theorem 2.5 (Progress)

Theorem 2.5 of the supplement (proved in §5 of the supplement) states
that repeatedly running the first stale cell terminates, either with
all cells clean or at a stuck cell, under the supplement's assumption
that rerunning a previously clean cell whose reads are unchanged
reproduces its recorded read and write sets (automatic for
deterministic evaluation).

The assumption is necessary because cell evaluation is
non-deterministic: without it, three cells suffice for a
non-terminating execution in which every run succeeds (let cells 2
and 3 each alternate between writing `{a}` and `{b}`; each run of one
backward-marks the other via `LastWriter`).

The formalization makes the assumption a side condition:
`FirstStaleExec` carries `F`, the positions with no validated recorded
behavior (initially all positions); a `first` step runs a cell in `F`
with any successful `[Inst-Run]`, and a `rerun` step must be a
`MatchingRunAt` step, reproducing the recorded read and write sets.
`FirstStaleStuck` holds when the first stale cell has no run of the
required kind.

- `progress` proves every such execution **terminates**, by the
  lexicographic measure `(|F|, n − first-stale position)`: a first run
  shrinks `F`, and a matching rerun has `W_i \ W'_i = ∅`, so
  `BackwardStale` marks nothing and the clean prefix strictly grows
  (`matching_clean_prefix`).
- `canRerun_after_run` proves the assumption is satisfiable where the
  paper's proof uses it: after any successful run of the first stale
  cell `i`, every cell `j < i` — in particular every cell marked by
  `BackwardStale` — can re-run from the new store reproducing its
  recorded output, read set, and write set, because the run's
  `NoWriteAfterRead` check keeps the new writes away from those cells'
  read sets.
- A matching rerun may forward-mark later cells; whether those remain
  matchable when the sweep reaches them is not proved, and a validated
  cell with no matching run counts as stuck.

## Verifying the axiom footprint

```bash
lake env lean -q <(echo 'import FlowBook
#print axioms FlowBook.preservation
#print axioms FlowBook.output_consistency
#print axioms FlowBook.progress')
```

Each reports `[propext, Classical.choice, Quot.sound]` — Lean's
standard axioms only.
