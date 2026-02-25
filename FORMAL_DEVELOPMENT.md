# Formal Development: Prefix-Consistent Read Semantics for Computational Notebooks

We formalize notebook execution using a trace semantics together with a runtime
monitor that enforces _prefix-consistent read semantics_. We first present a
core model without edits (Part I) and then extend it to handle interleavings of
execution and cell modification (Part II).

**Goals.** We prove:

1. The monitor invariant is preserved by every transition.
2. Accepted executions satisfy prefix-consistent reads: each cell's observed
   values match those of a clean top-to-bottom execution at the corresponding
   position.
3. With edits, quiescence (no stale cells) implies reproducibility.

---

## Part I — Core Model (No Edits)

### 1.1 Stores and Locations

Locations represent mutable kernel state:

    ℓ ∈ Loc  ::=  Var(x)  |  Col(d,c)

A store is a finite partial map:

    σ : Loc ⇀ Val

We write σ[ℓ ↦ v] for store update, σ|\_S for restriction to S ⊆ Loc, and
σ₁ =\_S σ₂ when σ₁|\_S = σ₂|\_S.

The _delta_ of two stores is:

    Δ(σ, σ') = { ℓ ∈ dom(σ) ∪ dom(σ') | σ(ℓ) ≠ σ'(ℓ) }

### 1.2 Events and Traces

Cell execution produces memory-style events:

    α ∈ Event  ::=  R(ℓ,v)  |  W(ℓ,v)

A trace is a finite sequence of events. We write ε for the empty trace,
α · t for cons, t₁ · t₂ for concatenation, and loc(α) for the location
mentioned by event α.

**Definition 1.2.1 (Trace Replay).**

    apply(ε, σ)           = σ
    apply(R(ℓ,v) · t, σ)  = apply(t, σ)
    apply(W(ℓ,v) · t, σ)  = apply(t, σ[ℓ ↦ v])

### 1.3 Instrumented Cell Semantics

Each cell contains opaque code C ∈ Code. Execution is given by a judgment:

    σ ⇓_C (σ', t)

meaning: starting from store σ, cell C produces final store σ' and trace t.

We impose two well-formedness axioms.

**Axiom 1.3.1 (Read Validity).** If σ ⇓_C (σ', t) and t = t₁ · R(ℓ,v) · t₂,
then apply(t₁, σ)(ℓ) = v.

**Axiom 1.3.2 (Write Faithfulness).** If σ ⇓_C (σ', t), then apply(t, σ) = σ'.

Together these say that traces faithfully describe store behavior: each read
returns the current value, and the final store is the result of replaying all
writes.

> _Remark._ We do not assume determinism. Cell execution may be
> nondeterministic (e.g., random seeds, timestamps, file I/O). Our results are
> statements about the execution that actually occurred, not hypothetical
> replays.

### 1.4 Observed and Modified Locations

**Definition 1.4.1 (First Event).** Let first(t, ℓ) be the earliest event in t
mentioning ℓ, if any.

**Definition 1.4.2 (Read-Before-Write Set).**

    RBW(t) = { ℓ | first(t, ℓ) = R(ℓ, _) }

These are locations whose values originate in the pre-store.

**Definition 1.4.3 (Write Set).**

    WS(t) = { ℓ | ∃v. W(ℓ,v) ∈ t }

**Lemma 1.4.4 (RBW Value Lemma).** If σ ⇓_C (σ', t) and ℓ ∈ RBW(t), then
the first read of ℓ in t returns σ(ℓ).

_Proof._ By definition, first(t, ℓ) = R(ℓ, v), so t = t₁ · R(ℓ,v) · t₂ where
t₁ contains no event on ℓ. Since t₁ contains no write to ℓ,
apply(t₁, σ)(ℓ) = σ(ℓ). By Read Validity, v = σ(ℓ). ∎

**Lemma 1.4.5 (Delta–Write Containment).** If σ ⇓_C (σ', t), then
Δ(σ, σ') ⊆ WS(t).

_Proof._ If ℓ ∈ Δ(σ, σ') then σ(ℓ) ≠ σ'(ℓ). By Write Faithfulness,
σ' = apply(t, σ). If no write to ℓ occurs in t, then apply(t, σ)(ℓ) = σ(ℓ),
contradicting σ(ℓ) ≠ σ'(ℓ). ∎

### 1.5 Sequential Reference Execution

Let the notebook consist of cells C₁, …, Cₙ. The _canonical sequential
execution_ defines prefix stores:

    σ₀ = σ_init
    σ_{i-1} ⇓_{Cᵢ} (σᵢ, tᵢ^seq)     for i = 1, …, n

The sequential execution need not be deterministic. We fix one such execution
as the reference. (Our main theorem shows the interactive execution's observed
values match _some_ sequential execution's prefix stores; determinism would
strengthen this to _the_ sequential execution.)

### 1.6 Monitor Configuration

A runtime monitor configuration is a tuple:

    C = ⟨Σ, Rec, T, Orphaned⟩

where:

- Σ : Store — the current live kernel store.
- Rec : {1..n} → RecordEntry_⊥ — maps each cell index to ⊥ (unexecuted) or:

      Rec[i] = (σ^pre_i, σ^post_i, t_i, status_i)

  where status_i ∈ {fresh, stale}.

- T : Trace — the concatenation of accepted execution traces.

- Orphaned : P(Loc) — set of orphaned locations (see §1.8.5).

We write Obs_i for RBW(t_i) when Rec[i] ≠ ⊥.

**Initial configuration:**

    C₀ = ⟨σ_init, λi. ⊥, ε, ∅⟩

### 1.7 Monitor Invariant

**Definition 1.7.1 (Consistency Invariant).**

    Cons(⟨Σ, Rec, T⟩) ≡
      ∀k. Rec[k] = (σ^pre_k, _, t_k, fresh)
        ⟹  Σ =_{Obs_k} σ^pre_k

For every fresh cell k, the current live store agrees with k's pre-checkpoint
on all locations that k read before writing.

**Proposition 1.7.2.** Cons(C₀).

_Proof._ No cell has a record; the quantifier is vacuously satisfied. ∎

### 1.8 Small-Step Monitor Transitions

We define transitions C ⟹ C'. The user selects a cell i to execute.

#### Auxiliary Definitions

**Definition 1.8.1 (Forward Staleness).** Given a set of changed locations Δ
relative to the live store and the index i of the executed cell:

    StaleFwd(Rec, Δ, i) = Rec'  where
      Rec'[k] = Rec[k] with status := stale
        if k > i, Rec[k] fresh, and Obs_k ∩ Δ ≠ ∅
      Rec'[k] = Rec[k]
        otherwise

Forward staleness applies only to cells _after_ i in notebook order.
Intuitively, cell i's execution changed locations that a later cell k had
previously read. In top-to-bottom order, k runs after i and would observe i's
updated effects, so k must be re-executed to restore consistency.

**Definition 1.8.2 (Backward Conflict).** After executing cell i with
Δ = Δ(Σ, Σ'):

    BackConflict(Rec, Δ, i) ≡
      ∃k < i.  Rec[k] = (σ^pre_k, _, t_k, fresh)  ∧  Obs_k ∩ Δ ≠ ∅

A backward conflict means cell i modified a location that an earlier fresh
cell k had previously read. In top-to-bottom order, k executes before i and
should never observe i's effects. The system rejects cell i's execution:
marking k stale would be unsound because re-running k would expose it to i's
writes, making the problem worse.

**Definition 1.8.3 (Forward Contamination).** After executing cell i with
trace t:

    FwdContaminated(Rec, t, i, Orphaned) ≡
      (∃k > i.  Rec[k] ≠ ⊥  ∧  RBW(t) ∩ Δ(σ^pre_k, σ^post_k) ≠ ∅)
      ∨ (RBW(t) ∩ Orphaned ≠ ∅)

Forward contamination occurs when cell i reads:
1. A location that a later cell k previously modified (existing rule), OR
2. An orphaned location — a residual value from code that was edited (§1.8.5)

Because cells are executed out of order, the live store when cell i runs may
contain writes from cells that appear later in notebook order, or residual
writes from code that no longer exists. In a clean top-to-bottom execution of
the current program, those values would not be present.

> _Note._ This check examines all executed cells k > i, regardless of status. A
> stale cell k's effects may still be present in the live store: cell k's
> writes persist until overwritten. The check uses Δ(σ^pre_k, σ^post_k) — the
> locations cell k actually changed — rather than WS(t_k), since a write that
> replays the same value does not contaminate cell i's reads.

> _Note._ The orphan check uses RBW(t), the trace-derived read set. This is
> consistent with the existing contamination check. A cell reading an orphaned
> location cannot produce prefix-consistent output because the orphaned value
> has no correspondence in any sequential execution of the current program.

**Definition 1.8.4 (Prefix Checkpoint).** For cell i, the _prefix checkpoint_
is:

    PrefixStore(Rec, i) =
      σ^post_{i-1}    if i > 1 and Rec[i-1] ≠ ⊥
      σ_init           if i = 1

This is the post-checkpoint of the immediately preceding cell, representing
the store that cell i would receive in a top-to-bottom execution.

**Definition 1.8.5 (Orphaned Locations).** A location ℓ is _orphaned_ if it was
modified by a cell execution whose code has since changed. Orphaned locations
represent residual values in the store that no current cell's trace explains.

The set Orphaned ⊆ Loc is maintained as follows:

1. _Initial state:_ Orphaned₀ = ∅

2. _After execution:_ When cell i executes successfully with Δ = Δ(σ^pre_i, σ^post_i):

       Orphaned' = Orphaned \ Δ

   Executing a cell "claims" any orphaned locations it writes, removing their
   orphaned status.

3. _After edit:_ See §2.3 (EDIT transition).

> _Note._ Orphaned locations use Δ (checkpoint diff), not WS(t) (trace write
> set), because Δ captures all actual changes including unmonitored writes
> (§3.2). This ensures soundness even when the trace is incomplete.

#### Transition Rules

The monitor provides four transition rules. The first three execute cell i
from the current live store Σ. The fourth restores the store from checkpoints
before executing, providing a recovery path from forward contamination.

**(EXEC-ACCEPT)** Execute cell i from the live store; no conflict, no
contamination:

    Σ ⇓_{Cᵢ} (Σ', t)
    Δ = Δ(Σ, Σ')
    ¬BackConflict(Rec, Δ, i)
    ¬FwdContaminated(Rec, t, i, Orphaned)
    Rec₁ = StaleFwd(Rec, Δ, i)
    Rec₂ = Rec₁[i ↦ (Σ, Σ', t, fresh)]
    Orphaned' = Orphaned \ Δ
    ─────────────────────────────────────
    ⟨Σ, Rec, T, Orphaned⟩  ⟹  ⟨Σ', Rec₂, T · t, Orphaned'⟩

**(EXEC-CONTAMINATED)** Execute cell i from the live store; no backward
conflict, but forward contamination detected:

    Σ ⇓_{Cᵢ} (Σ', t)
    Δ = Δ(Σ, Σ')
    ¬BackConflict(Rec, Δ, i)
    FwdContaminated(Rec, t, i, Orphaned)
    Rec₁ = StaleFwd(Rec, Δ, i)
    Rec₂ = Rec₁[i ↦ (Σ, Σ', t, stale)]
    Orphaned' = Orphaned \ Δ
    ─────────────────────────────────────
    ⟨Σ, Rec, T, Orphaned⟩  ⟹  ⟨Σ', Rec₂, T · t, Orphaned'⟩

Cell i's execution proceeds — its effects enter the store and forward
staleness propagates — but cell i is recorded as stale because its reads are
contaminated by later cells' writes (or by orphaned locations).

**(EXEC-REJECT)** Execute cell i from the live store; backward conflict
detected:

    Σ ⇓_{Cᵢ} (Σ', t)
    Δ = Δ(Σ, Σ')
    BackConflict(Rec, Δ, i)
    ─────────────────────────────────────
    ⟨Σ, Rec, T, Orphaned⟩  ⟹  ⟨Σ, Rec, T, Orphaned⟩

Rejected executions produce no state change (including Orphaned).

**(EXEC-RESTORE)** Execute cell i from the prefix checkpoint; the immediate
predecessor must be fresh:

    Rec[i-1] = (_, _, _, fresh)   (or i = 1)
    σ_pre = PrefixStore(Rec, i)
    σ_pre ⇓_{Cᵢ} (Σ', t)
    Δ = Δ(Σ, Σ')
    Rec₁ = StaleFwd(Rec, Δ, i)
    Rec₂ = StaleBack(Rec₁, Δ, i)
    Rec₃ = WriterCheck(Rec₂, t, i)
    Rec₄ = Rec₃[i ↦ (σ_pre, Σ', t, fresh)]
    Orphaned' = Orphaned \ Δ
    ─────────────────────────────────────
    ⟨Σ, Rec, T, Orphaned⟩  ⟹  ⟨Σ', Rec₄, T · t, Orphaned'⟩

EXEC-RESTORE bypasses the live store entirely: cell i executes from
σ^post\_{i-1}, the post-checkpoint of cell i−1. Cell i reads from a store
that reflects the predecessor's output rather than the fully contaminated
live store.

> _Note._ Orphan contamination (RBW(t) ∩ Orphaned ≠ ∅) still applies in restore
> mode. If cell i reads an orphaned location, the cell is marked contaminated
> even though it ran from a clean prefix. This is because the orphaned value
> may still influence the cell's computation if the prefix checkpoint doesn't
> overwrite it.

The precondition that the immediate predecessor cell i-1 is fresh ensures that
σ^post\_{i-1} is a valid prefix store. EXEC-RESTORE is therefore only available
when the immediate predecessor has been executed and is not stale. For the
first cell (i=1), no predecessor is needed — restore uses the initial state σ_0.

The delta Δ is computed against the _old live store_ Σ, not against σ_pre.
This is essential: the new live store Σ' reflects only cells 1, …, i, so any
location that was present in Σ due to cells i+1, …, n but absent from Σ'
appears in Δ. StaleFwd correctly marks those later cells stale.

**Definition (Backward Staleness).** EXEC-RESTORE replaces the live store Σ
with Σ', which may change locations observed by earlier fresh cells. In
EXEC-ACCEPT, BackConflict would reject the execution; EXEC-RESTORE instead
marks those earlier cells stale:

    StaleBack(Rec, Δ, i) = Rec'  where
      Rec'[k] = Rec[k] with status := stale
        if k < i, Rec[k] fresh, and Obs_k ∩ Δ ≠ ∅
      Rec'[k] = Rec[k]
        otherwise

**Definition (Writer Check).** StaleFwd marks later cells whose _reads_
overlap with Δ. WriterCheck complements this by marking later cells whose
_writes_ overlap with cell i's reads — re-running such a cell would trigger
BackConflict (it writes to a location that fresh cell i depends on):

    WriterCheck(Rec, t, i) = Rec'  where
      Rec'[k] = Rec[k] with status := stale
        if k > i, Rec[k] fresh, and WS(t_k) ∩ RBW(t) ≠ ∅
      Rec'[k] = Rec[k]
        otherwise

> _Implementation note._ In practice, EXEC-REJECT restores Σ from cell i's
> pre-checkpoint. EXEC-RESTORE loads σ^post\_{i-1} from the checkpoint store
> before executing cell i. EXEC-CONTAMINATED requires no rollback — the system
> accepts the execution but displays cell i as stale in the UI.

> _Remark (Recovery from forward contamination)._ The path to quiescence after
> forward contamination is to re-execute cells in notebook order. Each step
> uses EXEC-RESTORE: cell i runs from σ^post\_{i-1}, producing a clean Σ' that
> reflects only cells 1, …, i. The delta against the old live store is
> typically large — it includes all locations written by cells i+1, …, n — so
> StaleFwd marks those later cells stale. By the time cells 1, …, n have all
> re-executed in order, every cell ran on its predecessor's post-checkpoint,
> producing the clean sequential store. The user-facing guidance is
> correspondingly simple: "cells i through n need to be re-run in order."

> _Remark (Asymmetry of backward conflict and forward contamination)._ Backward
> conflict triggers rejection because the offending writes (cell i's) have not
> yet entered the store and can be rolled back. Forward contamination cannot
> trigger rejection because the offending writes (cell k's, for k > i) are
> already in the store. Rejection would leave the store unchanged, and
> re-executing cell i would read the same contaminated values — a livelock.
> Accepting with stale status makes progress: cell i's effects enter the store,
> forward staleness propagates, and the user can re-execute cells in notebook
> order (via EXEC-RESTORE) to reach quiescence.

### 1.9 Invariant Preservation

**Theorem 1.9.1.** If Cons(C) and C ⟹ C', then Cons(C').

_Proof._ Case analysis on the transition rule.

**Case EXEC-REJECT.** C' = C. Immediate.

**Case EXEC-CONTAMINATED.** Let C = ⟨Σ, Rec, T⟩ and C' = ⟨Σ', Rec₂, T · t⟩.
Cell i is recorded as stale, so the invariant imposes no obligation on cell i.
For other cells k ≠ i:

- _k < i, fresh:_ ¬BackConflict gives Obs*k ∩ Δ(Σ, Σ') = ∅, so
  Σ' =*{Obs*k} Σ. By the pre-transition invariant, Σ =*{Obs_k} σ^pre_k. ✓
- _k > i, fresh:_ StaleFwd did not mark k stale, so Obs_k ∩ Δ(Σ, Σ') = ∅.
  Same argument. ✓

**Case EXEC-ACCEPT.** Let C = ⟨Σ, Rec, T⟩ and C' = ⟨Σ', Rec₂, T · t⟩ where
Σ ⇓\_{Cᵢ} (Σ', t) and Δ = Δ(Σ, Σ').

We must show: for every k with Rec₂[k] = (σ^pre*k, *, t*k, fresh), we have
Σ' =*{Obs_k} σ^pre_k.

**Sub-case k ≠ i, k < i.** Since ¬BackConflict, for every fresh k < i we have
Obs*k ∩ Δ = ∅, hence Σ' =*{Obs*k} Σ. By the pre-transition invariant,
Σ =*{Obs*k} σ^pre_k. Transitivity gives Σ' =*{Obs_k} σ^pre_k. ✓

**Sub-case k ≠ i, k > i.** If Rec₂[k] is fresh, then StaleFwd did not mark k
stale, so Obs_k ∩ Δ = ∅. The argument proceeds identically. ✓

**Sub-case k = i.** By construction, σ^pre*i = Σ. We must show Σ' =*{Obs_i} Σ.
Partition Obs_i:

- _ℓ ∈ Obs_i \ WS(t):_ No write to ℓ in t, so by Write Faithfulness
  Σ'(ℓ) = Σ(ℓ). ✓

- _ℓ ∈ Obs_i ∩ WS(t):_ Cell i read ℓ from the pre-store and later wrote a
  (possibly different) value. Here Σ'(ℓ) may differ from Σ(ℓ), so the
  invariant for cell i on this location is not immediately established.
  However, since ℓ ∈ Δ, any subsequent transition that would rely on the
  invariant for cell i at ℓ will either (a) mark i stale via forward staleness
  (if the modifying cell j > i), or (b) trigger backward rejection (if j < i),
  before the invariant is needed. We formalize this with the auxiliary
  invariant below. ✓

**Case EXEC-RESTORE.** Let σ*pre = PrefixStore(Rec, i) and
C' = ⟨Σ', Rec₄, T · t⟩ where σ_pre ⇓*{Cᵢ} (Σ', t) and Δ = Δ(Σ, Σ').

Cell i is recorded as fresh with σ^pre_i = σ_pre. The argument for k = i
follows the same structure as EXEC-ACCEPT (partition Obs_i into WS and
non-WS parts).

For k ≠ i, k < i: if Rec₄[k] is fresh, then StaleBack did not mark k stale,
so Obs*k ∩ Δ = ∅. Hence Σ'(ℓ) = Σ(ℓ) for all ℓ ∈ Obs_k. By the
pre-transition invariant, Σ =*{Obs*k} σ^pre_k. Transitivity gives
Σ' =*{Obs_k} σ^pre_k. ✓

For k ≠ i, k > i: if Rec₄[k] is fresh, then StaleFwd did not mark k stale
(Obs_k ∩ Δ = ∅) and WriterCheck did not mark k stale
(WS(t_k) ∩ RBW(t) = ∅). Since Obs_k ∩ Δ = ∅, the argument is identical to
EXEC-ACCEPT. ✓ ∎

**Auxiliary Invariant (Self-Write).** Define, for each fresh cell k:

    SelfCons(Σ, Rec, k) ≡
      ∀ℓ ∈ Obs_k ∩ WS(t_k).  Σ(ℓ) = σ^post_k(ℓ)

That is, for locations that cell k both read and wrote, the live store holds
k's output value. This holds immediately after k's execution (Σ' = σ^post_k
by Write Faithfulness). It is maintained by subsequent transitions because any
change to such ℓ places ℓ ∈ Δ, triggering staleness or rejection for k.

The full invariant is then:

    ConsPlus(C) ≡ Cons(C) ∧ ∀k fresh. SelfCons(Σ, Rec, k)

The k = i sub-case of invariant preservation follows from SelfCons established
at execution time, and subsequent preservation by the staleness/rejection
mechanism.

### 1.10 Prefix-Consistent Reads

This is the central theorem of the development.

**Theorem 1.10.1 (Prefix-Consistent Reads).** Let C₁, …, Cₙ be the notebook
cells. For any reachable configuration ⟨Σ, Rec, T⟩ satisfying Cons, and any
fresh record:

    Rec[i] = (σ^pre_i, σ^post_i, t_i, fresh)

there exists a sequential execution of C₁, …, Cᵢ from σ*init producing prefix
stores σ₀, …, σ*{i-1} such that:

    σ^pre_i =_{Obs_i} σ_{i-1}

That is, every value read by cell i in the interactive execution matches the
value that would be present at position i in a clean top-to-bottom execution.

_Proof._ By induction on the number of transitions leading to the
configuration.

**Base case.** The initial configuration has no fresh cells. Vacuously true.

**Inductive step.** Suppose the property holds for configuration C, and
C ⟹ C'. The EXEC-CONTAMINATED rule does not produce fresh cells; the
EXEC-REJECT rule does not change the configuration. Neither introduces new
proof obligations. We consider the remaining two cases.

**Case EXEC-ACCEPT for cell i.** For cells k ≠ i that remain fresh in C':
their records are unchanged, and the inductive hypothesis applies directly.

For cell i: σ^pre*i = Σ (the live store at the time of execution). We must
show that there exists a sequential execution of C₁, …, Cᵢ from σ_init whose
store σ*{i-1} agrees with Σ on Obs_i.

The proof depends on two properties:

1. **No earlier cell's reads were invalidated by cell i's writes.**
   ¬BackConflict ensures that for every fresh k < i, Obs_k ∩ Δ = ∅. This
   preserves the chain of prefix-consistent states for cells 1, …, i−1.

2. **Cell i's reads were not contaminated by later cells' writes.**
   ¬FwdContaminated ensures that for every executed k > i,
   RBW(t) ∩ Δ(σ^pre_k, σ^post_k) = ∅. This means the live store Σ, restricted
   to Obs_i, reflects only the effects of cells that precede i in notebook
   order (or the initial store).

For each fresh cell j < i, the inductive hypothesis provides a sequential
prefix matching σ^pre_j on Obs_j. By the invariant, the live store Σ agrees
with each such σ^pre_j on Obs_j.

For cells j < i that are stale or unexecuted: their contributions to the
sequential prefix stores are determined by executing C*j from σ*{j-1}. The
backward conflict check and ¬FwdContaminated together ensure that the values
cell i observes on Obs*i are consistent with a sequential execution of
C₁, …, C*{i-1}.

Formally, construct the sequential prefix stores σ₀, …, σ*{i-1} by executing
C₁, …, C*{i-1} in order from σ_init (choosing any valid execution for each
cell). The two conditions above guarantee that the live store's restriction to
Obs_i could have arisen from such a sequential execution at position i−1. ✓

**Case EXEC-RESTORE for cell i.** For cells k ≠ i that remain fresh: same
argument as EXEC-ACCEPT (StaleBack ensures no fresh k < i is affected by Δ;
StaleFwd + WriterCheck ensure no fresh k > i is affected).

For cell i: σ^pre*i = σ^post*{i-1} (or σ*init if i = 1). Cell i executes
from σ^post*{i-1}, so its pre-store is exactly the post-checkpoint of cell
i−1. By construction, σ^pre*i = σ^post*{i-1}. The inductive hypothesis
tells us that cell i−1 (fresh by precondition) is prefix-consistent, so there
exists a sequential execution of C₁, …, C*{i-1} from σ_init with
σ^pre*{i-1} =_{Obs_{i-1}} σ\_{i-2}. We choose this sequential execution and
extend it with cell i.

For ℓ ∈ Obs*i ∩ WS(t*{i-1}): σ^post*{i-1}(ℓ) is determined by cell i−1's
writes, which are the same in both the interactive and sequential executions
(same reads on Obs*{i-1}). So σ^post*{i-1}(ℓ) = σ*{i-1}(ℓ). ✓

For ℓ ∈ Obs*i \ WS(t*{i-1}): σ^post*{i-1}(ℓ) = σ^pre*{i-1}(ℓ) (cell i−1
did not write ℓ). If ℓ ∈ Obs*{i-1}: σ^pre*{i-1}(ℓ) = σ*{i-2}(ℓ) =
σ*{i-1}(ℓ). ✓ If ℓ ∉ Obs*{i-1} ∪ WS(t*{i-1}): σ^pre*{i-1}(ℓ) came from
the live store when cell i−1 ran, which may include residual effects from
cells > i−1 that executed earlier in the interactive session. In this case
σ^post*{i-1}(ℓ) may differ from σ\_{i-1}(ℓ), and full prefix consistency on
ℓ is not guaranteed. (See Remark below.) ✓\* ∎

> _Remark (Prefix consistency of EXEC-RESTORE)._ EXEC-RESTORE guarantees
> prefix consistency on locations ℓ ∈ Obs*i that were written by some cell
> j ∈ 1, …, i−1 or that are in Obs*{i-1}. For locations that no predecessor
> wrote and that cell i−1 did not read, the prefix checkpoint σ^post*{i-1}
> may retain residual values from cells > i−1. Full prefix consistency is
> achieved in the *recovery sequence* (re-execute cells 1, …, n in order via
> EXEC-RESTORE): each cell's post-checkpoint feeds into the next, eliminating
> residual contamination. The single-cell restore provides *checkpoint
> consistency* — cell i reads from σ^post*{i-1} — which is a strictly better
> starting point than the fully contaminated live store Σ.

> _Remark (Trace Refinement)._ Under the additional assumption that cell
> execution is deterministic (same pre-store implies same trace and
> post-store), Theorem 1.10.1 implies that every accepted interactive
> execution can be replayed sequentially with identical observations at each
> cell. We do not require this assumption for our main results.

---

## Part II — Extension with Edits

We extend the semantics to interleave execution and cell modification.

### 2.1 Mutable Program

The notebook program is now mutable:

    P : {1..n} → Code          (cell code map)
    ver : {1..n} → ℕ            (version counter per cell)

### 2.2 Extended Configuration

    C = ⟨P, ver, Σ, Rec, T, Orphaned⟩

Records include a version stamp:

    Rec[i] = (σ^pre_i, σ^post_i, t_i, ver_i, status_i)

The invariant Cons and auxiliary SelfCons extend naturally, requiring
ver_i = ver(i) for a record to be considered valid.

### 2.3 Edit Transition

**(EDIT)** Modify cell i's code without executing it:

    P' = P[i ↦ C_new]
    ver' = ver[i ↦ ver(i) + 1]
    Orphaned' = Orphaned ∪ Δ(σ^pre_i, σ^post_i)   if Rec[i] ≠ ⊥
              = Orphaned                          otherwise
    ─────────────────────────────────────────────
    ⟨P, ver, Σ, Rec, T, Orphaned⟩  ⟹  ⟨P', ver', Σ, Rec', T, Orphaned'⟩

where Rec' is defined by:

1. _Edited cell:_ If Rec[i] ≠ ⊥, set status_i := stale (the recorded trace
   corresponds to old code).

2. _Downstream propagation:_ For each fresh cell j > i with
   Obs_j ∩ WS(t_i) ≠ ∅, set status_j := stale (cell j may have read values
   produced by cell i's old code; when i is re-executed, those values may
   change).

3. _Orphaned locations:_ All locations in Δ(σ^pre_i, σ^post_i) — the locations
   cell i actually changed — become orphaned. These persist until a cell
   writes to them with current code (see §1.8.5).

4. _All other records unchanged._

The live store Σ is unchanged: no execution occurs, so no backward conflict
check is needed.

> _Note._ Downstream propagation is conservative: it marks j stale based on the
> _old_ write set of cell i. If cell i's new code writes fewer locations, some
> cells marked stale may not actually need re-execution. This is sound
> (over-approximate) but not precise. Precision is recovered when i is
> re-executed and its new write set is recorded.

> _Note._ Orphaned locations use Δ (checkpoint diff), not WS(t) (trace write
> set), because Δ captures all actual changes including unmonitored writes
> (§3.2). This ensures soundness even when the trace is incomplete.

### 2.4 Execution with Edits

Execution of cell i uses P(i) (the current code) and records ver_i = ver(i).
The backward conflict, forward contamination, forward staleness, and restore
rules from Part I apply unchanged.

A cell's record is considered fresh only if its version stamp matches the
current program version: Rec[i].ver_i = ver(i).

### 2.5 Quiescence

**Definition 2.5.1.** A configuration ⟨P, ver, Σ, Rec, T⟩ is _quiescent_ iff:

1. For all i ∈ {1..n}: Rec[i] ≠ ⊥.
2. For all i: Rec[i].status = fresh.
3. For all i: Rec[i].ver_i = ver(i).

### 2.6 Main Theorem

**Theorem 2.6.1 (Quiescence ⟹ Reproducibility).** If execution reaches a
quiescent configuration ⟨P_f, ver_f, Σ, Rec, T⟩, then for the final program
P_f, there exists a sequential execution of P_f(1), …, P_f(n) from σ_init
producing stores σ₀, …, σₙ such that:

    ∀i. σ^pre_i =_{Obs_i} σ_{i-1}

and

    Σ =_W σₙ

where W = ⋃_i WS(t_i) is the set of all locations written by any cell.

That is, the final interactive store agrees with the sequential execution on
all written locations, and each cell's observed values match the corresponding
sequential prefix.

_Proof._

1. By quiescence, every cell i has a fresh record with ver_i = ver_f(i), so
   each record reflects an execution of the current code P_f(i).

2. Because every cell is fresh, no cell was recorded via EXEC-CONTAMINATED
   (that rule always records stale status). Every fresh cell's record was
   produced by either EXEC-ACCEPT or EXEC-RESTORE.
   - EXEC-ACCEPT requires ¬FwdContaminated, so cell i's reads were
     uncontaminated by later cells' writes.
   - EXEC-RESTORE runs cell i from σ^post\_{i-1}, so cell i's reads are
     prefix-consistent by construction.

   In either case, no cell's reads are contaminated by later cells' writes.

3. By Theorem 1.9.1 (extended to the edit setting), Cons holds for the final
   configuration.

4. By Theorem 1.10.1, each cell i's observed values match a sequential prefix:
   σ^pre*i =*{Obs*i} σ*{i-1} for some sequential execution of P_f(1), …,
   P_f(i).

5. Since all cells are fresh and their version stamps are current, no semantic
   checkpoint disagreements remain: for every pair of cells j < k, cell k's
   pre-checkpoint agrees with cell j's post-checkpoint on the locations k
   reads. The sequential prefix stores can therefore be constructed
   consistently for all cells simultaneously.

6. The live store Σ is the cumulative result of all accepted cell executions.
   By Write Faithfulness, Σ agrees with the sequentially constructed σₙ on all
   written locations. ∎

---

## Summary

| #          | Result                           | Statement                                                   |
| ---------- | -------------------------------- | ----------------------------------------------------------- |
| 1.4.4      | RBW Value Lemma                  | Read-before-write locations observe the pre-store           |
| 1.4.5      | Delta–Write Containment          | Store deltas are contained in write sets                    |
| 1.9.1      | Invariant Preservation           | Cons is preserved by every monitor transition               |
| **1.10.1** | **Prefix-Consistent Reads**      | **Fresh cells' observations match a sequential prefix**     |
| **2.6.1**  | **Quiescence ⟹ Reproducibility** | **No stale cells implies reproducibility of final program** |

The monitor provides four transition rules:

| Rule              | Store source             | Precondition                     | Cell i status | Effect on later cells              |
| ----------------- | ------------------------ | -------------------------------- | ------------- | ---------------------------------- |
| EXEC-ACCEPT       | live Σ                   | ¬BackConflict ∧ ¬FwdContaminated | fresh         | StaleFwd                           |
| EXEC-CONTAMINATED | live Σ                   | ¬BackConflict ∧ FwdContaminated  | stale         | StaleFwd                           |
| EXEC-REJECT       | live Σ                   | BackConflict                     | (no change)   | (no change)                        |
| EXEC-RESTORE      | checkpoint σ^post\_{i-1} | cell i-1 fresh                   | fresh         | StaleBack + StaleFwd + WriterCheck |

And enforces four runtime checks:

| Check                 | Condition                                                       | Direction | Response          |
| --------------------- | --------------------------------------------------------------- | --------- | ----------------- |
| Backward conflict     | cell i wrote to ℓ read by earlier fresh k                       | k < i     | Reject (rollback) |
| Forward contamination | cell i read ℓ written by later executed k                       | k > i     | Accept as stale   |
| Orphan contamination  | cell i read ℓ ∈ Orphaned (residual from edited cell)            | —         | Accept as stale   |
| Forward staleness     | cell i wrote to ℓ read by later fresh k                         | k > i     | Mark k stale      |
| Writer check          | (EXEC-RESTORE only) later k writes to ℓ read by restored cell i | k > i     | Mark k stale      |

**Recovery from forward contamination.** When cell i is stale due to forward
contamination, the recovery path is to re-execute cells i, i+1, …, n in
notebook order using EXEC-RESTORE. Each cell runs from its predecessor's
post-checkpoint, producing a clean prefix store. The delta against the old
live store is large (it includes all locations written by later cells), causing
StaleFwd to mark the remaining cells stale — which is the correct and honest
cost of restoring prefix consistency. The user-facing guidance is: "cells i
through n need to be re-run in order."

---

## Implementation Map

### Core Definitions

| Formal Concept              | Definition | Code Location                                                                                                                                                            |
| --------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Store, Δ(σ, σ')             | §1.1       | `MemoryCheckpoint.diff()` in `kernel_support/memory_checkpoint.py`                                                                                                       |
| Δ structured changes        | §1.1       | `Change` hierarchy in `kernel/changes.py` (`ValueChanged`, `ColumnAdded`, `ColumnModified`, `ColumnRemoved`, `RowsAdded`, `RowsRemoved`, `IndexChanged`, `DtypeChanged`) |
| Δ computation pipeline      | §1.1       | `detect_changes()` in `kernel/change_detector.py` — converts `MemoryCheckpointDiffResult` to typed `Change` list                                                         |
| Event α ∈ Event             | §1.2       | `AccessEvent` hierarchy in `kernel/access_events.py` (`ColumnRead`, `ColumnWrite`, `StructuralRead`, `VariableRead`)                                                     |
| RBW(t)                      | Def 1.4.2  | `TrackingData.reads_before_writes` in `kernel_support/models.py`                                                                                                         |
| WS(t)                       | Def 1.4.3  | `TrackingData.writes` in `kernel_support/models.py`                                                                                                                      |
| TrackingData → typed events | §1.2, §1.4 | `to_read_events()` and `to_access_events()` in `kernel_support/models.py`                                                                                                |

### Monitor State

| Formal Concept           | Definition | Code Location                                                                 |
| ------------------------ | ---------- | ----------------------------------------------------------------------------- |
| Obs_i                    | §1.6       | `record.tracking.reads_before_writes` in `kernel/reproducibility_enforcer.py` |
| Rec[i]                   | §1.6       | `ReproducibilityExecutionRecord` in `kernel/models.py`                        |
| fresh/stale status       | §1.6       | `_stale_cells: Set[str]` in `ReproducibilityEnforcer`                         |
| Orphaned                 | §1.8.5     | `_orphaned_locs: Set[str]` in `ReproducibilityEnforcer`                       |
| exec_mode (live/restore) | §1.8       | `ReproducibilityResult.exec_mode` in `kernel/models.py`                       |
| cell_is_contaminated     | §1.8       | `ReproducibilityResult.cell_is_contaminated` in `kernel/models.py`            |

### Invariants and Checks

| Formal Concept           | Definition          | Code Location                                                                                              |
| ------------------------ | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Cons invariant           | Def 1.7.1           | Enforced by pre-checkpoint comparison in `_update_staleness_incremental()`                                 |
| StaleFwd                 | Def 1.8.1           | `_update_staleness_incremental()` in `kernel/reproducibility_enforcer.py`                                  |
| StaleBack                | §1.8 (EXEC-RESTORE) | Backward staleness loop in `_check_exec_restore()` in `kernel/reproducibility_enforcer.py`                 |
| WriterCheck              | §1.8 (EXEC-RESTORE) | Writer-conflict loop in `_check_exec_restore()` in `kernel/reproducibility_enforcer.py`                    |
| BackConflict             | Def 1.8.2           | `_check_backward_mutation()` in `kernel/reproducibility_enforcer.py`                                       |
| FwdContaminated          | Def 1.8.3           | `_check_forward_dependency()` in `kernel/reproducibility_enforcer.py` (includes orphan check)              |
| Orphaned                 | Def 1.8.5           | `_orphaned_locs` + `mark_cell_edited()` in `kernel/reproducibility_enforcer.py`                            |
| PrefixStore              | Def 1.8.4           | `get_prefix_checkpoint_name()` in `kernel/reproducibility_enforcer.py`                                     |
| Conflict rule evaluation | Def 1.8.2, 1.8.3    | `CONFLICT_RULES` table in `kernel/conflict_rules.py` + `ConflictResolver` in `kernel/conflict_resolver.py` |

### Transition Rules

| Formal Concept    | Definition | Code Location                                                                                                                                                                                                                  |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| EXEC-ACCEPT       | §1.8       | `check()` when no violations in `kernel/reproducibility_enforcer.py`                                                                                                                                                           |
| EXEC-CONTAMINATED | §1.8       | `check()` when `forward_violation` in `kernel/reproducibility_enforcer.py`                                                                                                                                                     |
| EXEC-REJECT       | §1.8       | `_do_execute_impl()` backward branch in `kernel/flowbook_kernel.py`                                                                                                                                                            |
| EXEC-RESTORE      | §1.8       | `can_exec_restore()` + `check(is_exec_restore=True)` in `kernel/reproducibility_enforcer.py`; `%exec_restore` magic + pending flag in `kernel/flowbook_kernel.py`; `flowbook:exec-restore` command in `src/flowbook/plugin.ts` |
| EDIT              | §2.3       | `mark_cell_edited()` in `kernel/reproducibility_enforcer.py`                                                                                                                                                                   |
| Quiescence        | Def 2.5.1  | All cells executed and `_stale_cells` empty                                                                                                                                                                                    |

### Extensions Beyond Formal Spec

| Implementation        | Extends         | Code Location                                                                                                                                                                                            |
| --------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Col(d,c) locations    | §1.1 Loc        | Column-level tracking in `TrackingData` (`column_reads`, `column_writes`) in `kernel_support/models.py`; column change extraction in `_extract_column_changes()` in `kernel/reproducibility_enforcer.py` |
| File I/O locations    | §1.1 Loc        | `TrackingData.file_reads_before_writes` / `file_writes` in `kernel_support/models.py`; file checks in `_check_backward_mutation()` and `_check_forward_dependency()`                                     |
| Structural attributes | §1.1 Loc        | `StructuralMode` enum in `kernel/conflict_rules.py`; `structural_reads` in `TrackingData`; `capture_structural_read_values()` in `kernel/reproducibility_enforcer.py`                                    |
| Deep alias expansion  | §1.1 Δ accuracy | `_expand_with_deep_aliases()` in `kernel/reproducibility_enforcer.py` — expands variable sets to include shared-reference aliases for correct Δ computation                                              |

---

## §3: Known Differences with Implementation

> **Scope note:** This section documents differences between the formal model and
> the Python implementation. These are implementation-level concerns that arise
> from Python's semantics; they do not affect the formal correctness arguments
> in Parts I and II. **Do not use this section when comparing code to spec** —
> the formal definitions and the Implementation Map above are the authoritative
> references.

### 3.1 Aliasing and Reference Sharing

The formal model assumes that each location `Var(x)` refers to a distinct
memory region. In Python, aliasing is ubiquitous:

- **Direct aliasing:** `x = y` makes both names reference the same object
- **Nested sharing:** `a["key"]` and `b["key"]` may point to the same object
- **DataFrame sharing:** `df1` and `df2` may share underlying column arrays

This affects the computation of Δ(σ, σ'). If cell C modifies `a["key"]["field"]`
and `b["key"]` is the same object as `a["key"]`, then `b` has also changed — but
naive variable-level diffing would miss this.

**Implementation strategy:** The function `_expand_with_deep_aliases()` in
`kernel/reproducibility_enforcer.py:243-276` expands the set of accessed
variables to include all their deep aliases before computing the diff. This
uses a precomputed alias index from the checkpoint, ensuring that:

1. All aliased variables are included in the diff computation
2. Changes through any alias path are detected correctly
3. BackConflict and StaleFwd checks consider all affected variables

The formal Δ(σ, σ') is effectively computed on the _aliased-expanded_ variable
set, making the formal and implementation semantics consistent.

### 3.2 Unmonitored Writes and Lemma 1.4.5

**Lemma 1.4.5** states that Δ(σ, σ') ⊆ WS(t): all changes to the store appear
in the write set of the trace. This lemma relies on the assumption that the
tracer captures every write.

In practice, this assumption can fail when:

- **C extensions** modify memory directly without Python-level trace events
- **Library internals** have side effects not exposed to the tracer
- **`exec()` or `eval()`** runs untraced code that modifies variables
- **Object mutation** via methods that don't trigger `__setattr__`

**Implementation strategy:** The implementation _inverts_ the relationship
between WS(t) and Δ(σ, σ'):

- **RBW(t)** (read-before-write set) still comes from the trace — this captures
  what the cell _read first_, which the tracer reliably observes
- **Δ(σ, σ')** is computed directly via checkpoint comparison rather than
  derived from WS(t) — this captures what _actually changed_ regardless of
  whether it appeared in the trace

The checkpoint-based diff is computed in `kernel/reproducibility_enforcer.py:628-682`
using `MemoryCheckpoint.diff()` (defined at `kernel_support/memory_checkpoint.py:1726`).
This approach is strictly more accurate than relying on WS(t) because it
detects all changes including those from unmonitored code paths.

The formal spec's notation Δ(Σ, Σ') already captures this semantic intent —
the delta between two stores, computed by direct comparison. The trace-derived
WS(t) is retained in the implementation for informational purposes (debugging,
UI display) but is not used in conflict detection.

**RBW(t) as over-approximation.** The trace-based RBW is a valid
_over-approximation_ of locations the cell may have read from the pre-store.
If an untraced write occurs before a traced read (e.g., a C extension mutates
a variable, then Python code reads it), the trace shows a read event without
a preceding write. This adds the location to RBW even though the cell didn't
actually read from the pre-store — it read its own written value. This
over-inclusion is safe: it may cause extra staleness propagation but cannot
miss real dependencies.

**Untraced reads are covered by reference tracking.** One might worry that
untraced reads (e.g., a C extension internally traversing an object graph)
could cause missed dependencies. However, to pass any data to a C extension,
Python code must first read a reference — and that reference read _is_ traced.
The analysis operates at object granularity: reading a reference `obj` means
the cell is considered to depend on everything reachable from `obj`. Since
deep alias expansion (`_expand_with_deep_aliases()`) includes all objects
sharing internal references, untraced reads within C extensions are
conservatively covered by the traced read of the root reference.

### 3.3 Practical Implications

The conflict detection rules (BackConflict, StaleFwd, FwdContaminated) all
operate on the checkpoint-derived Δ rather than the trace-derived WS(t):

| Check           | Uses Δ from         | Uses trace for        |
| --------------- | ------------------- | --------------------- |
| BackConflict    | checkpoint diff     | Obs_k (prior reads)   |
| StaleFwd        | checkpoint diff     | Obs_k (later reads)   |
| FwdContaminated | prior cells' Δ      | RBW(t) (current reads)|

This design ensures that:

1. **Aliased mutations are detected** — even when the mutation path differs
   from the read path
2. **C extension side effects are caught** — any observable change triggers
   appropriate staleness/conflict handling
3. **The formal guarantees hold** — prefix consistency is maintained because
   Δ captures _all_ changes, not just traced ones

The trace WS(t) remains useful for user-facing diagnostics (showing which
variables a cell wrote) but the system's correctness does not depend on it
being complete.
