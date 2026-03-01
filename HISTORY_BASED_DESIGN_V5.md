# Notebook Reproducibility Semantics (V5)

A notebook is **reproducible** when all cells are Clean—executing top-to-bottom
would produce the same outputs as currently recorded.

---

## Part I — State

### 1.1 State Components

```
S = ⟨C, O, Σ, T, R, W, L⟩

Program State:
  C : Cell → Code           -- source code per cell
  O : Cell → Output         -- recorded output per cell (or ⊥)
  Σ : Loc → Value           -- live kernel state

Instrumentation State:
  T : Cell → Status         -- cell status (see below)
  R : Cell → P(Loc)         -- locations read by each cell
  W : Cell → P(Loc)         -- locations written by each cell
  L : Loc ⇀ Cell            -- last writer: which cell last wrote each location
```

Cells are numbered 1..n in document order.

### 1.2 Status and Reasons

```
Reason = NeverExecuted
       | CodeChanged
       | InputChanged(loc: Loc, writer: Cell)
       | WriteConflict(loc: Loc, writer: Cell)
       | ReadsFromLater(loc: Loc, writer: Cell)
       | SourceDeleted(loc: Loc)
       | OrderChanged

Status = Clean | Stale(reasons: P(Reason))
```

A cell is **Clean** if it needs no action, or **Stale** with a non-empty set of
reasons explaining why it needs re-execution.

**Reason meanings:**
| Reason | Meaning |
|--------|---------|
| `NeverExecuted` | Cell has no recorded output |
| `CodeChanged` | Source code modified since last execution |
| `InputChanged(x, j)` | Cell j wrote x, which this cell reads |
| `WriteConflict(x, j)` | Cell j wrote x, which this cell also writes |
| `ReadsFromLater(x, j)` | This cell reads x from cell j, but j comes later |
| `SourceDeleted(x)` | This cell reads x, but x's writer was deleted |
| `OrderChanged` | Cell positions changed, affecting dependencies |

**Helper function:**
```
-- Add a reason to a cell's status (idempotent for Clean→Stale)
AddReason(i, r):
  match T_i:
    Clean        → T_i := Stale({r})
    Stale(rs)    → T_i := Stale(rs ∪ {r})
```

### 1.3 Derived Functions

```
-- Locations written by cells after i
Overwritten(W, i) = W_{i+1} ∪ W_{i+2} ∪ ... ∪ W_n

-- Which cell before i should have written x (based on document order)
LastWriter(W, i, x) = max { j < i | x ∈ W_j }, or ⊥ if none

-- Cell i is runnable if its reads come from expected writers
Runnable(L, W, R, i) ≡ ∀x ∈ R_i. L(x) = LastWriter(W, i, x)

-- Check if cell is stale
IsStale(T_i) ≡ T_i ≠ Clean
```

### 1.4 Initial State

```
S_init = ⟨C, O, Σ, T, R, W, L⟩ where:
  C_i = initial code for cell i
  O_i = ⊥                         -- no outputs yet
  Σ   = Σ_init                    -- initial kernel state
  T_i = Stale({NeverExecuted})    -- all cells need to run
  R_i = ∅                         -- no reads recorded
  W_i = ∅                         -- no writes recorded
  L   = ∅                         -- no writers yet
```

---

## Part II — Transitions

### 2.1 Edit

User modifies cell i's code:

```
EDIT(i, code'):
  C_i := code'
  T_i := Stale({CodeChanged})
```

Editing replaces any prior reasons with `CodeChanged`. Other cells unchanged.

### 2.2 Execute

User executes cell i:

```
EXEC(i):
  -- Precondition: cell must be runnable (using recorded R_i)
  require Runnable(L, W, R_i, i)

  -- Save state for potential rollback
  Σ_save := Σ
  L_save := L

  -- Run code in current kernel state
  (Σ', R', W', out) := eval(C_i, Σ)

  -- Check Runnable with NEW reads
  if ¬Runnable(L, W, R', i):
    -- Rollback: restore pre-execution state
    Σ := Σ_save
    L := L_save
    R_i := R'          -- Record reads (for future Runnable checks)

    -- Mark contaminated with specific reasons
    T_i := Stale({ ReadsFromLater(x, L(x)) | x ∈ R', L(x) > i })
    return Error("Cell reads from later cell")

  -- Commit: update program state
  Σ   := Σ'
  O_i := out

  -- Update instrumentation
  R_i := R'
  W_i := W'

  -- Update last-writer map
  for x ∈ W':
    L(x) := i

  -- Cell is Clean (all reasons cleared)
  T_i := Clean

  -- Propagate staleness to later cells that touch what we wrote
  for j in {i+1, ..., n}:
    for x ∈ W' ∩ R_j:
      AddReason(j, InputChanged(x, i))
    for x ∈ W' ∩ W_j:
      AddReason(j, WriteConflict(x, i))
```

**Precondition:** Runnable(L, W, R_i, i) must hold (using recorded R_i).

- Unexecuted cells: R_i = ∅, so Runnable is vacuously true.
- Previously-executed cells: R_i contains recorded reads. If contaminated
  (L ≠ LastWriter for some read), execution is blocked.

**Postcondition:** After execution, check Runnable with NEW reads R'.
- If Runnable: commit changes, T_i = Clean
- If ¬Runnable: **rollback** Σ and L, record R_i, set T_i with contamination reasons

**Explanation:**
1. Check precondition — prevents re-executing contaminated cells
2. Save Σ and L for potential rollback
3. Execute code, capturing reads R', writes W', and output
4. Check Runnable with R' — if contaminated, rollback and abort
5. Commit: update Σ, O_i, R_i, W_i, L
6. T_i = Clean (reads came from expected cells)
7. Propagate staleness to later Clean cells that read or write what we wrote

### 2.3 Insert

```
INSERT(pos, code):
  n := n + 1

  -- Shift cells: positions pos..n-1 become pos+1..n
  for k in {n, n-1, ..., pos+1}:      -- iterate backwards
    C_k := C_{k-1}
    O_k := O_{k-1}
    T_k := T_{k-1}
    R_k := R_{k-1}
    W_k := W_{k-1}

  -- Update L: cell references shift up
  for x where L(x) ≥ pos:
    L(x) := L(x) + 1

  -- Initialize new cell at pos
  C_pos := code
  O_pos := ⊥
  T_pos := Stale({NeverExecuted})
  R_pos := ∅
  W_pos := ∅

  -- Invalidate Clean cells whose Runnable changed
  -- (LastWriter shifts for cells after pos)
  for j in {pos+1, ..., n}:
    if T_j = Clean ∧ ¬Runnable(L, W, R_j, j):
      AddReason(j, OrderChanged)
```

**Explanation:** Inserting a cell shifts all later cells down. The L map must update
cell references (a cell that was at position 5 is now at 6). Clean cells after the
insertion point may become Stale if their LastWriter computation changed.

### 2.4 Delete

```
DELETE(pos):
  -- Capture which locations the deleted cell wrote
  deleted_writes := W_pos

  -- Clear L for locations written by deleted cell
  for x ∈ deleted_writes:
    if L(x) = pos:
      L(x) := ⊥

  -- Shift cells: positions pos+1..n become pos..n-1
  for k in {pos, ..., n-1}:
    C_k := C_{k+1}
    O_k := O_{k+1}
    T_k := T_{k+1}
    R_k := R_{k+1}
    W_k := W_{k+1}

  n := n - 1

  -- Update L: cell references shift down
  for x where L(x) > pos:
    L(x) := L(x) - 1

  -- Mark cells as Stale if they read orphaned locations
  -- (locations whose writer was deleted)
  for j in {1, ..., n}:
    for x ∈ R_j:
      if L(x) = ⊥:
        AddReason(j, SourceDeleted(x))
```

**Orphan semantics:** When a cell is deleted, any location x it was the last writer
of becomes "orphaned" (L(x) = ⊥). Cells that read orphaned locations become Stale
because on replay, no cell would provide that value—it came from a cell that no
longer exists.

**Example:** If cell 2 wrote y and cell 3 reads y, deleting cell 2 sets L(y) = ⊥.
Cell 3 becomes Stale because its recorded read of y has no source.

### 2.5 Move

```
MOVE(src, dst):
  require src ≠ dst

  -- Save the moving cell's state
  C_saved := C_src
  O_saved := O_src
  T_saved := T_src
  R_saved := R_src
  W_saved := W_src

  if src < dst:
    -- Moving forward: shift cells src+1..dst backward
    for k in {src, ..., dst-1}:
      C_k := C_{k+1}
      O_k := O_{k+1}
      T_k := T_{k+1}
      R_k := R_{k+1}
      W_k := W_{k+1}

    -- Update L: cells src+1..dst shift to src..dst-1
    for x where src < L(x) ≤ dst:
      L(x) := L(x) - 1
    -- The moved cell goes from src to dst
    for x where L(x) = src:
      L(x) := dst

  else:  -- src > dst
    -- Moving backward: shift cells dst..src-1 forward
    for k in {src, src-1, ..., dst+1}:  -- iterate backwards
      C_k := C_{k-1}
      O_k := O_{k-1}
      T_k := T_{k-1}
      R_k := R_{k-1}
      W_k := W_{k-1}

    -- Update L: cells dst..src-1 shift to dst+1..src
    for x where dst ≤ L(x) < src:
      L(x) := L(x) + 1
    -- The moved cell goes from src to dst
    for x where L(x) = src:
      L(x) := dst

  -- Place saved cell at destination
  C_dst := C_saved
  O_dst := O_saved
  T_dst := T_saved
  R_dst := R_saved
  W_dst := W_saved

  -- Invalidate Clean cells whose Runnable changed
  lo := min(src, dst)
  hi := max(src, dst)
  for j in {lo, ..., hi}:
    if T_j = Clean ∧ ¬Runnable(L, W, R_j, j):
      AddReason(j, OrderChanged)

  -- Also invalidate cells outside the range that read from moved cell
  -- (their LastWriter changed if moved cell was their expected source)
  for j in {1, ..., n}:
    if T_j = Clean ∧ ∃x ∈ R_j. L(x) = dst ∧ LastWriter(W, j, x) ≠ dst:
      AddReason(j, OrderChanged)
```

**Explanation:** Moving a cell changes document order, which affects LastWriter
computations. A Clean cell becomes Stale if:
1. Its Runnable check now fails (L(x) ≠ LastWriter for some read)
2. It reads from the moved cell, but that cell is no longer in the right position

**Example:** If cell 3 reads x from cell 1, moving cell 1 to position 4 means
LastWriter(W, 3, x) changes. Cell 3 becomes Stale because on replay, it would
read x from a different source (or none).

---

## Part III — Reproducibility

### 3.1 Definition

```
Reproducible(S) ≡ ∀i ∈ {1..n}. T_i = Clean
```

### 3.2 Soundness

**Theorem.** If Reproducible(S), then executing cells 1..n in order from Σ_init
produces the same outputs as O.

**Proof sketch.**

All cells Clean ⟹ all cells satisfy Runnable.

Runnable(L, W, R_i, i) means: ∀x ∈ R_i. L(x) = LastWriter(W, i, x)

This says: for every location cell i reads, the current last writer L(x)
equals the cell that *should* have written it in document order.

By induction on i:
- Cell 1: R_1's values come from L (which points to appropriate earlier cells or init)
- Cell i: By IH, cells 1..i-1 produced correct outputs.
  Since L(x) = LastWriter(W, i, x) for all reads, cell i reads the same values
  it read when it was recorded as Clean.

Same inputs ⟹ same outputs (determinism assumed). ∎

### 3.3 Completeness

**Theorem.** If ¬Reproducible(S), then some cell's output may differ on replay.

**Proof by cases** on the reason(s) in T_i = Stale(reasons):

**Case: NeverExecuted ∈ reasons**
  O_i = ⊥. Replay produces output where none recorded.

**Case: CodeChanged ∈ reasons**
  Current code ≠ executed code. Different code may produce different output.

**Case: InputChanged(x, j) ∈ reasons**
  Cell j wrote x after cell i recorded its read.
  On replay, cell i reads a different value for x. Different output.

**Case: WriteConflict(x, j) ∈ reasons**
  Cell j wrote x, changing L(x). Cell i's write order is now inconsistent
  with document order. Different final state on replay.

**Case: ReadsFromLater(x, j) ∈ reasons**
  Cell i read x from cell j > i. On replay, cell j hasn't run when cell i
  executes, so x has a different (or undefined) value. Different output.

**Case: SourceDeleted(x) ∈ reasons**
  Cell i read x, but the cell that wrote x was deleted.
  On replay, no cell provides x. Error or different output.

**Case: OrderChanged ∈ reasons**
  Cell positions changed, affecting LastWriter computation.
  Runnable may fail, or reads may come from different cells. ∎

---

## Part IV — Examples

### Notation

```
Rows:    Cells (by position) and L (last-writer map)
Columns: Steps (init, Exec i, Edit i)
Symbols: !  = Stale (with one or more reasons)
         ✓  = Clean
         X  = Rollback (contaminated, execution rejected)

Reasons are shown in step explanations, e.g.:
  [1] W={x}. AddReason(2, InputChanged(x, 1))
```

### 4.1 In-Order Execution

```
Cells: 1: x=1    2: y=x+1    3: print(y)

         | init | Exec 1  | Exec 2  | Exec 3
---------+------+---------+---------+---------
1: x=1   |  !   | [1] ✓   |    ✓    |    ✓
2: y=x+1 |  !   |    !    | [2] ✓   |    ✓
3: pr(y) |  !   |    !    |    !    | [3] ✓
---------+------+---------+---------+---------
L(x)     |  -   |    1    |    1    |    1
L(y)     |  -   |    -    |    2    |    2

[1] R={}, W={x}. Runnable ✓ (vacuous)
[2] R={x}, W={y}. L(x)=1 = LastWriter(W,2,x) ✓
[3] R={y}, W={}. L(y)=2 = LastWriter(W,3,y) ✓

All Clean → Reproducible
```

### 4.2 Edit Then Re-execute

```
Starting from 4.1: all Clean

         | all ✓ | Edit 1 | Exec 1  | Exec 2  | Exec 3
---------+-------+--------+---------+---------+---------
1: x=1   |   ✓   | x=2  ! | [4] ✓   |    ✓    |    ✓
2: y=x+1 |   ✓   |    ✓   |    !    | [5] ✓   |    ✓
3: pr(y) |   ✓   |    ✓   |    ✓    |    !    | [6] ✓
---------+-------+--------+---------+---------+---------
L(x)     |   1   |    1   |    1    |    1    |    1
L(y)     |   2   |    2   |    2    |    2    |    2

Edit 1:  Code changed → T_1 := Stale
[4] W={x}. Propagate: x ∈ R_2 → T_2 := Stale
[5] W={y}. Propagate: y ∈ R_3 → T_3 := Stale
[6] Clean

All Clean → Reproducible
```

### 4.3 Out-of-Order (Rollback)

```
Cells: 1: print(x)    2: x=1

         | init | Exec 2  | Exec 1
---------+------+---------+---------
1: pr(x) |  !   |    !    | [2] X !
2: x=1   |  !   | [1] ✓   |    ✓
---------+------+---------+---------
L(x)     |  -   |    2    |    2

[1] R={}, W={x}. Runnable ✓. L(x):=2
[2] R={x}. L(x)=2, LastWriter(W,1,x)=⊥
    2 ≠ ⊥ → ROLLBACK. R_1:={x}, T_1:=Stale

Cell 1 reads x from cell 2 (later) → Contaminated
Future Exec 1 blocked: R_1={x}, L(x)=2 ≠ LastWriter=⊥
```

### 4.4 Re-execute Earlier Cell

```
Cells: 1: x=1    2: x=2    3: print(x)
Starting: all Clean, L(x)=2

         | all ✓ | Exec 1  | Exec 2  | Exec 3
---------+-------+---------+---------+---------
1: x=1   |   ✓   | [4] ✓   |    ✓    |    ✓
2: x=2   |   ✓   |    !    | [5] ✓   |    ✓
3: pr(x) |   ✓   |    !    |    !    | [6] ✓
---------+-------+---------+---------+---------
L(x)     |   2   |    1    |    2    |    2

[4] W={x}, L(x):=1
    Propagate: x ∈ W_2 → T_2 := Stale  (write-write)
    Propagate: x ∈ R_3 → T_3 := Stale
[5] W={x}, L(x):=2
    Propagate: x ∈ R_3 → T_3 := Stale
[6] Clean

Re-running cell 1 invalidates both 2 (writes x) and 3 (reads x).
```

### 4.5 Skip Middle Cell

```
Cells: 1: x=1    2: x=2    3: print(x)

         | init | Exec 1  | Exec 3
---------+------+---------+---------
1: x=1   |  !   | [1] ✓   |    ✓
2: x=2   |  !   |    !    |    !
3: pr(x) |  !   |    !    | [2] ✓
---------+------+---------+---------
L(x)     |  -   |    1    |    1

[1] W={x}, L(x):=1. Clean
[2] R={x}. L(x)=1, LastWriter(W,3,x)=?
    W_2 not recorded (cell 2 never ran)!
    LastWriter only sees recorded W, so LastWriter(W,3,x)=1
    L(x)=1 = 1 ✓ → Clean

But cell 2 is Stale → NOT Reproducible!
Replay runs cell 2, which writes x=2, then cell 3 sees different value.
```

### 4.6 Fix Contamination via Edit

```
Cells: 1: print(x)    2: x=1
Starting: T=[Stale, Clean] from 4.3 (cell 1 contaminated, R_1={x})

         | start | Edit 1     | Exec 1
---------+-------+------------+---------
1: pr(x) |   !   | pr("hi") ! | [3] ✓
2: x=1   |   ✓   |     ✓      |    ✓
---------+-------+------------+---------
L(x)     |   2   |     2      |    2

Edit 1:  Code no longer reads x. T_1 := Stale
[3] R={}. Runnable ✓ (vacuous). Clean

Contamination fixed by removing the problematic dependency.
```

### 4.7 Propagation Rules

A cell j > i becomes Stale when EXEC(i) commits if:

| Condition | Reason |
|-----------|--------|
| W_i ∩ R_j ≠ ∅ | j reads what i wrote → input changed |
| W_i ∩ W_j ≠ ∅ | j writes what i wrote → L changed |

Both mean: document-order replay differs from recorded state.

---

## Part V — Extended Examples

### 5.1 Out-of-Order Then Fix

```
Cells: 1: x=1    2: x=x+1    3: print(x)

         | init | Exec 1  | Exec 3  | Exec 2  | Exec 3
---------+------+---------+---------+---------+---------
1: x=1   |  !   | [1] ✓   |    ✓    |    ✓    |    ✓
2: x=x+1 |  !   |    !    |    !    | [3] ✓   |    ✓
3: pr(x) |  !   |    !    | [2] ✓   |    !    | [4] ✓
---------+------+---------+---------+---------+---------
L(x)     |  -   |    1    |    1    |    2    |    2

[1] W={x}, L(x):=1
[2] R={x}. L(x)=1, LastWriter=1 ✓ (cell 2 not run, W_2 unknown)
    But cell 2 is Stale → Not Reproducible
[3] W={x}, L(x):=2. Propagate: x ∈ R_3 → T_3:=Stale
[4] R={x}. L(x)=2, LastWriter=2 ✓

All Clean → Reproducible
```

### 5.2 Re-execution Cascade

```
Cells: 1: x=2    2: x=3    3: print(x)    4: x=4

         | all ✓ | Exec 1
---------+-------+---------
1: x=2   |   ✓   | [5] ✓
2: x=3   |   ✓   |    !
3: pr(x) |   ✓   |    !
4: x=4   |   ✓   |    !
---------+-------+---------
L(x)     |   4   |    1

Starting: L(x)=4 (cell 4 was last writer)
[5] W={x}, L(x):=1
    Propagate: x ∈ W_2 → T_2:=Stale
    Propagate: x ∈ R_3 → T_3:=Stale
    Propagate: x ∈ W_4 → T_4:=Stale

Re-running cell 1 invalidates ALL later cells that touch x.
```

### 5.3 Contamination (Forward Read)

```
Cells: 1: print(x)    2: x=1

         | init | Exec 2  | Exec 1
---------+------+---------+---------
1: pr(x) |  !   |    !    | [2] X !
2: x=1   |  !   | [1] ✓   |    ✓
---------+------+---------+---------
L(x)     |  -   |    2    |    2

[1] W={x}, L(x):=2
[2] R={x}. L(x)=2, LastWriter(W,1,x)=⊥
    2 ≠ ⊥ → ROLLBACK

Cell 1 reads from cell 2 (later) → Contaminated.
Cannot be fixed by re-running. Must restructure.
```

### 5.4 Fix Contamination by Moving Cell

```
After 5.3: N=(1,2), T=[Stale,Clean], R_1={x}
Move cell 1 after cell 2: N=(2,1)

         | before | Move 1→2 | Exec 1
---------+--------+----------+---------
2: x=1   |   ✓    |    ✓     |    ✓
1: pr(x) |   !    |    !     | [3] ✓
---------+--------+----------+---------
L(x)     |   2    |    2     |    2

After move: LastWriter(W, 1, x) = 2 (cell 2 now before cell 1)
[3] R={x}. L(x)=2, LastWriter=2 ✓

All Clean → Reproducible
```

### 5.5 Edit Cascade

```
Cells: 1: x=1    2: y=x    3: z=y    4: print(z)

         | all ✓ | Edit 2
---------+-------+----------
1: x=1   |   ✓   |    ✓
2: y=x   |   ✓   | y=x*2  !
3: z=y   |   ✓   |    ✓
4: pr(z) |   ✓   |    ✓
---------+-------+----------
L(x)     |   1   |    1
L(y)     |   2   |    2
L(z)     |   3   |    3

Edit 2: Code changed → T_2:=Stale
        Cells 3,4 still Clean (Edit doesn't propagate)

But NOT Reproducible: cell 2 is Stale.

Exec 2 would propagate: y ∈ R_3 → T_3:=Stale
Then Exec 3 would propagate: z ∈ R_4 → T_4:=Stale
```

### 5.6 Multiple Edits Then Execute

```
Cells: 1: x=1    2: y=x    3: print(y)

         | all ✓ | Edit 1 | Edit 2  | Edit 3  | Exec 1  | Exec 2  | Exec 3
---------+-------+--------+---------+---------+---------+---------+---------
1: x=1   |   ✓   | x=2  ! |    !    |    !    | [4] ✓   |    ✓    |    ✓
2: y=x   |   ✓   |    ✓   | y=x+1 ! |    !    |    !    | [5] ✓   |    ✓
3: pr(y) |   ✓   |    ✓   |    ✓    | pr(x) ! |    ✓    |    !    | [6] ✓
---------+-------+--------+---------+---------+---------+---------+---------
L(x)     |   1   |    1   |    1    |    1    |    1    |    1    |    1
L(y)     |   2   |    2   |    2    |    2    |    2    |    2    |    2

Edit 1: T_1:=Stale
Edit 2: T_2:=Stale (code changed)
Edit 3: T_3:=Stale (code changed, now reads x not y)
[4] W={x}. Propagate: x ∈ R_2 → T_2:=Stale (already stale)
    T_3 stays Clean? No—T_3 was Stale from edit.
[5] W={y}. Propagate: y ∈ R_3? No, R_3={x} after edit.
    But T_3 is still Stale (code changed).
[6] R={x}. L(x)=1, LastWriter=1 ✓

All Clean → Reproducible
```

### 5.7 Delete Cell (Orphan)

```
Cells: 1: x=2    2: y=x    3: print(y)
Starting: all Clean, L(y)=2 (cell 2 wrote y)

         | all ✓ | Delete 2
---------+-------+----------
1: x=2   |   ✓   |    ✓
2: y=x   |   ✓   |   ---
3: pr(y) |   ✓   |    !
---------+-------+----------
L(x)     |   1   |    1
L(y)     |   2   |    ⊥

Delete 2:
  1. Cell 2 wrote y, so set L(y) := ⊥ (orphaned)
  2. Check for orphan reads: R_3 = {y}, L(y) = ⊥
  3. Cell 3 reads orphaned location → T_3 := Stale

Why Stale? Cell 3's recorded output came from reading y, which was
provided by cell 2. With cell 2 deleted, there's no source for y.
On replay, cell 3 would fail or read a different value.

Path to Clean:
  - Insert a cell before 3 that writes y, OR
  - Edit cell 3 to not read y
```

### 5.8 Insert Cell

```
Cells: 1: print(x)    2: x=1
Starting: T=[Stale,Clean] (cell 1 contaminated from 5.3)

         | start | Insert 0 | Exec 0  | Exec 1
---------+-------+----------+---------+---------
0: x=0   |  ---  |    !     | [3] ✓   |    ✓
1: pr(x) |   !   |    !     |    !    | [4] ✓
2: x=1   |   ✓   |    ✓     |    ✓    |    ✓
---------+-------+----------+---------+---------
L(x)     |   2   |    2     |    0    |    0

Insert 0 (new cell "x=0" at position 0): T_0:=Stale
[3] W={x}, L(x):=0. Propagate: x ∈ R_1, x ∈ W_2
    T_1:=Stale (was already), T_2:=Stale
[4] R={x}. L(x)=0, LastWriter(W,1,x)=0 ✓
    T_1:=Clean

Cell 1 no longer contaminated—reads from cell 0 (earlier).
Cell 2 now Stale (its write will overwrite cell 0's).
```

### 5.9 Summary of All Staleness Causes

```
| Reason              | When Added                         | Fix               |
|---------------------|------------------------------------|-------------------|
| NeverExecuted       | Initial state, INSERT              | Run cell          |
| CodeChanged         | EDIT                               | Run cell          |
| InputChanged(x, j)  | EXEC(j) wrote x that we read       | Run cell          |
| WriteConflict(x, j) | EXEC(j) wrote x that we write      | Run cell          |
| ReadsFromLater(x,j) | EXEC rollback: read x from j > i   | Restructure       |
| SourceDeleted(x)    | DELETE removed writer of x         | Add writer / edit |
| OrderChanged        | INSERT or MOVE changed positions   | Run cell          |
```

**Multiple reasons:** A cell can accumulate multiple reasons. For example, after
EXEC(1) writes x, cell 3 gets `InputChanged(x, 1)`. If cell 2 is then deleted
and cell 3 also read y from cell 2, it gets `SourceDeleted(y)` added to its
reasons: `Stale({InputChanged(x, 1), SourceDeleted(y)})`.

**Clearing reasons:** When a cell is executed successfully (T_i := Clean), all
reasons are cleared. The cell starts fresh.

---

## Part VI — Summary

**State:** S = ⟨C, O, Σ, T, R, W, L⟩

**Key insight:** L tracks *actual* last writer, LastWriter computes *expected*
last writer from document order. Runnable checks they match.

**Status with reasons:**
```
Status = Clean | Stale(reasons: P(Reason))

Reason = NeverExecuted | CodeChanged | InputChanged(x, j)
       | WriteConflict(x, j) | ReadsFromLater(x, j)
       | SourceDeleted(x) | OrderChanged
```

| Status | Meaning | Action |
|--------|---------|--------|
| Clean | Reproducible | None |
| Stale({...}) | Needs re-execution | Run cell (reasons explain why) |

**Transitions and reasons:**
| Transition | Reason(s) added |
|------------|-----------------|
| EDIT(i, _) | `CodeChanged` (replaces all) |
| EXEC(i) rollback | `ReadsFromLater(x, j)` for each contaminating read |
| EXEC(i) propagate | `InputChanged(x, i)` or `WriteConflict(x, i)` |
| INSERT(pos, _) | `NeverExecuted` for new cell; `OrderChanged` for affected |
| DELETE(pos) | `SourceDeleted(x)` for each orphaned read |
| MOVE(src, dst) | `OrderChanged` for affected cells |

**Execution rule:**
- Precondition: Runnable(L, W, R_i, i) — blocks re-executing contaminated cells
- Postcondition: Check Runnable with new R' — rollback if contaminated
- Propagation: AddReason for each x ∈ W_i ∩ R_j or x ∈ W_i ∩ W_j

**Reproducible:** ∀i. T_i = Clean
