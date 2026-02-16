# Plan: Heap-Based Semantics with Explicit Aliasing (Simplified)

## Motivation

The current formal model uses `Var(x)` and `Col(d,c)` as locations but doesn't model:
1. **Aliasing**: Multiple variables can reference the same heap object
2. **Unmonitored mutations**: The heap may change in ways not reflected in the trace

This extension adds a heap-based semantics that explicitly handles both.

## Scope

**Supported values:**
- Primitives (int, float, string, bool, None)
- DataFrames (with named columns)
- Arrays (numpy ndarrays)

**Trace granularity:**
- Top-level variable accesses: `R(x)`, `W(x)`
- DataFrame column accesses: `R(x.c)`, `W(x.c)`

No tracking of nested dict/list paths or arbitrary object attributes.

**Key assumption:** All variable-level reads and writes appear in the trace. However, the *heap mutations* caused by those writes may not be fully described by the trace (e.g., C extension side effects).

---

## Part III — Heap Semantics with Aliasing

### 3.1 Addresses and Heap

**Definition 3.1.1 (Addresses).** Let `Addr` be a set of heap addresses.

**Definition 3.1.2 (Values).**

```
v ∈ Val ::= prim                     -- primitive value
          | DataFrame(cols)          -- cols : ColName → Addr (column buffers)
          | Array(data)              -- data is the buffer contents
```

**Definition 3.1.3 (Heap).**

```
H : Addr ⇀ Val
```

**Definition 3.1.4 (Environment).** Maps variable names to addresses:

```
E : VarName → Addr
```

This is Python's `user_ns`.

**Definition 3.1.5 (State).**

```
S = (E, H)
```

### 3.2 Locations

**Definition 3.2.1 (Locations).** Two kinds:

```
ℓ ∈ Loc ::= Var(a)      -- variable at address a
          | Col(a, c)   -- column c of DataFrame at address a
```

**Definition 3.2.2 (Variable Location).** For variable x:

```
loc(E, x) = Var(E(x))
```

**Definition 3.2.3 (Column Location).** For column c of DataFrame variable d:

```
col_loc(E, d, c) = Col(E(d), c)
```

### 3.3 Aliasing

**Definition 3.3.1 (Variable Aliasing).** Variables x and y are aliased in E iff:

```
E(x) = E(y)
```

**Definition 3.3.2 (Alias Set).**

```
Aliases(E, x) = { y ∈ dom(E) | E(x) = E(y) }
```

**Definition 3.3.3 (Alias Expansion).** Expand a set of variables to include aliases:

```
Expand(E, X) = ⋃ { Aliases(E, x) | x ∈ X }
```

### 3.4 Events and Traces

**Definition 3.4.1 (Events).** Events track locations:

```
α ∈ Event ::= R(ℓ)      -- read location ℓ
            | W(ℓ)      -- write location ℓ
```

where `ℓ ∈ {Var(a), Col(a,c)}`.

**Definition 3.4.2 (Variable Projection).** Project address-based events to variable names using environment E:

```
vars(E, R(Var(a)))   = { x | E(x) = a }
vars(E, W(Var(a)))   = { x | E(x) = a }
vars(E, R(Col(a,c))) = { x | E(x) = a }  -- the DataFrame variable
vars(E, W(Col(a,c))) = { x | E(x) = a }
```

### 3.5 Read-Before-Write Sets

RBW is derived from the trace. Since all variable reads appear in the trace,
RBW is complete (no unmonitored reads at variable level).

**Definition 3.5.1 (RBW at Address Level).**

```
RBW_loc(t) = { ℓ | first(t, ℓ) = R(ℓ) }
```

**Definition 3.5.2 (RBW at Variable Level).** Using environment E at cell start:

```
RBW_var(E, t) = Expand(E, ⋃ { vars(E, R(ℓ)) | ℓ ∈ RBW_loc(t) })
```

The alias expansion ensures that if we read address `a` and both `x` and `y` point to `a`, both are included.

**Lemma 3.5.3 (RBW Completeness).** Since all variable reads appear in the trace:

```
{ x | cell actually read from E(x) } ⊆ RBW_var(E, t)
```

RBW may over-approximate (include variables read after being written internally),
but cannot under-approximate. This is the safe direction for dependency tracking.

**Definition 3.5.4 (RBW Over-Approximation).** If an unmonitored write occurs
before a traced read of the same variable:

```
Trace shows:  R(x)  with no preceding W(x)
Reality:      Internal W(x) happened before R(x)
Result:       x ∈ RBW_var even though cell didn't read from pre-store
```

This over-inclusion is safe: it may cause extra staleness but cannot miss dependencies.

### 3.6 State Delta and Unmonitored Mutations

The trace records variable-level events, but the heap may change in ways not
captured by the trace. We distinguish two notions of "what changed":

**Definition 3.6.1 (Trace-Derived Write Set).** Variables the trace says were written:

```
WS_trace(t) = { x | W(Var(E(x))) ∈ t or W(Col(E(x), c)) ∈ t for some c }
```

**Definition 3.6.2 (Heap Delta).** Addresses that *actually* changed (from checkpoint comparison):

```
Δ_H(H, H') = { a | H(a) ≠ H'(a) }
```

**Definition 3.6.3 (Delta at Variable Level).** Variables whose referent changed:

```
Δ_var(E, H, H') = { x ∈ dom(E) | E(x) ∈ Δ_H(H, H') }
```

**Definition 3.6.4 (Expanded Delta).** Include aliases:

```
Δ_expanded(E, H, H') = Expand(E, Δ_var(E, H, H'))
```

**Lemma 3.6.5 (Write Set Containment — May Fail).**

In the idealized model: `Δ_var(E, H, H') ⊆ WS_trace(t)`

In practice, this can fail when:
- C extensions modify heap objects without Python-level trace events
- Library internals have side effects not exposed to the tracer
- Methods mutate objects without triggering `__setattr__`

**Definition 3.6.6 (Unmonitored Writes).** The set of variables changed but not in trace:

```
Unmonitored(E, H, H', t) = Δ_var(E, H, H') \ WS_trace(t)
```

**Key Design Decision:** Conflict detection uses `Δ_expanded` (checkpoint-derived),
NOT `WS_trace` (trace-derived). This ensures unmonitored writes are still detected.

**Lemma 3.6.7 (Soundness Despite Unmonitored Writes).** For conflict detection:

```
BackConflict uses:   Δ_expanded(E, H, H')     -- catches all mutations
StaleFwd uses:       Δ_expanded(E, H, H')     -- catches all mutations
FwdContaminated uses: Δ_expanded from prior cells
```

Since all checks use checkpoint-derived Δ, unmonitored writes cannot cause
missed conflicts. They may cause false negatives in `WS_trace` (for display
purposes) but not in conflict detection.

### 3.7 Extended Monitor Configuration

**Definition 3.7.1 (Record).** Each cell record includes environment:

```
Rec[i] = (E^pre_i, H^pre_i, H^post_i, t_i, status_i)
```

Note: `E^post_i` not needed if we assume no rebinding during cell execution.

**Definition 3.7.2 (Observed Variables).**

```
Obs_i = RBW_var(E^pre_i, t_i)
```

This is alias-expanded by definition.

### 3.8 Conflict Detection

**Definition 3.8.1 (Backward Conflict).** After cell i executes with `Δ = Δ_expanded(E^pre_i, H, H')`:

```
BackConflict(Rec, Δ, i) ≡ ∃k < i. Rec[k] fresh ∧ Obs_k ∩ Δ ≠ ∅
```

**Definition 3.8.2 (Forward Staleness).** Mark later cells stale:

```
StaleFwd(Rec, Δ, i)[k] = stale  if k > i ∧ Rec[k] fresh ∧ Obs_k ∩ Δ ≠ ∅
```

**Definition 3.8.3 (Forward Contamination).**

```
FwdContaminated(Rec, t, i) ≡ ∃k > i. RBW_var(E^pre_i, t) ∩ Δ_expanded(E^pre_k, H^pre_k, H^post_k) ≠ ∅
```

### 3.9 Soundness

**Theorem 3.9.1 (Invariant Preservation).** If `Cons(C)` and `C ⟹ C'`, then `Cons(C')`.

**Theorem 3.9.2 (Prefix Consistency with Aliasing).** For fresh cell i:

```
∀ℓ ∈ Obs_loc_i.  H^pre_i(ℓ) = H_{i-1}(ℓ)
```

where `H_{i-1}` is the heap from a sequential execution of cells 1..i-1.

---

## Implementation Mapping

| Formal Concept | Definition | Code Location |
|----------------|------------|---------------|
| `Addr` | §3.1.1 | Python `id(obj)` |
| `E` (Environment) | §3.1.4 | `user_ns` dict |
| `H` (Heap) | §3.1.3 | `MemoryCheckpoint.variables` (deep-copied) |
| `Aliases(E, x)` | §3.3.2 | `checkpoint.get_aliases_for_vars({x})` |
| `Expand(E, X)` | §3.3.3 | `_expand_with_deep_aliases(X, pre_checkpoint)` |
| `RBW_var` | §3.5.2 | `TrackingData.reads_before_writes` |
| `WS_trace` | §3.6.1 | `TrackingData.writes` |
| `Δ_H(H, H')` | §3.6.2 | `MemoryCheckpoint.diff()` |
| `Δ_expanded` | §3.6.4 | Diff with `keys_to_include` from alias expansion |
| `Unmonitored` | §3.6.6 | `Δ_var \ writes` (implicit in checkpoint-based detection) |
| `Obs_i` | §3.7.2 | `record.tracking.reads_before_writes` (pre-expanded) |
| `Col(a, c)` | §3.2.1 | `TrackingData.column_reads_before_writes` |

**Key Implementation Insight:** The code uses `MemoryCheckpoint.diff()` for conflict
detection, not `TrackingData.writes`. This is the implementation of using `Δ_expanded`
instead of `WS_trace`, ensuring unmonitored writes are caught.

---

## Work Items

### Phase 1: Definitions (1-2 days)
1. [ ] Add §3.1 (Addr, Val, Heap, Environment, State) to FORMAL_DEVELOPMENT.md
2. [ ] Add §3.2 (Locations: Var(a), Col(a,c))
3. [ ] Add §3.3 (Aliasing, Alias Set, Expand)

### Phase 2: Traces and RBW (1 day)
4. [ ] Add §3.4 (Events with locations)
5. [ ] Add §3.5 (RBW_var with alias expansion, RBW Completeness lemma, over-approximation)

### Phase 3: Delta and Unmonitored Writes (1 day)
6. [ ] Add §3.6.1-3.6.4 (WS_trace, Heap delta, Δ_var, Δ_expanded)
7. [ ] Add §3.6.5-3.6.7 (Write Set Containment failure, Unmonitored definition, Soundness lemma)
8. [ ] Clarify that conflict detection uses Δ_expanded, not WS_trace

### Phase 4: Monitor Revision (1 day)
9. [ ] Add §3.7 (Extended record with E^pre)
10. [ ] Add §3.8 (Revised BackConflict, StaleFwd, FwdContaminated using Δ_expanded)

### Phase 5: Soundness (1-2 days)
11. [ ] State and prove Invariant Preservation (§3.9.1)
12. [ ] State and prove Prefix Consistency (§3.9.2)
13. [ ] Prove soundness despite unmonitored writes (relies on checkpoint-based Δ)
14. [ ] Prove RBW over-approximation is safe (conservative direction)

### Phase 6: Integration (1 day)
15. [ ] Update Implementation Map with new mappings
16. [ ] Revise current §3 "Known Differences":
    - §3.1 Aliasing → reference new Part III
    - §3.2 Unmonitored Writes → reference §3.6.5-3.6.7
    - §3.2 RBW Over-approximation → reference §3.5.4
17. [ ] Verify existing tests against formal definitions

---

## Relationship to Parts I-II

**Part I-II** remain valid as a **simplified model** under the assumption:

> **No-Aliasing Assumption:** For all x ≠ y in dom(E), E(x) ≠ E(y).

Under this assumption:
- `Expand(E, X) = X`
- `Var(E(x))` bijects with `Var(x)`
- Part I locations and Part III locations coincide

**Part III** generalizes Part I-II by removing the no-aliasing assumption and adding explicit alias expansion.

---

## What Remains in "Known Differences" (Current §3)

After adding Part III, the current §3 becomes redundant and should be revised:

| Current §3 Section | Status | Action |
|--------------------|--------|--------|
| §3.1 Aliasing | ✓ Formally modeled | Replace with reference to Part III §3.3 |
| §3.2 Unmonitored Writes | ✓ Formally modeled | Replace with reference to Part III §3.6.5-3.6.7 |
| §3.2 RBW Over-approximation | ✓ Formally modeled | Replace with reference to Part III §3.5.4 |
| §3.2 Reference Tracking | ✓ Subsumed by aliasing | Replace with reference to Part III §3.3, §3.5.2 |
| §3.3 Practical Implications | ✓ Formalized | Replace with reference to Part III §3.6.7 (Soundness lemma) |

**New §3 (after revision):** A brief note stating that Part III provides the complete
heap-based semantics, and directing readers there. The detailed "known differences"
become formal definitions rather than implementation notes.
