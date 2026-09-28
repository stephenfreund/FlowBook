/-
# A Semantics for Interactive Computational Notebook Execution

This file formalizes Section "A Semantics for Interactive Computational
Notebook Execution" of the FlowBook paper:

* the notebook model: cells, outputs, stores over locations;
* black-box cell evaluation as a big-step *relation* (the paper treats
  the underlying language runtime as a black box, and notes that cells
  may be non-deterministic);
* the notebook operations `Run/Edit/Insert/Delete/Move` and the
  standard semantics of Figure 8;
* top-to-bottom execution (`Std-Top-to-Bottom-Execution`) and
  reproducibility / output consistency (Definition 1.1 of the supplement).

The development is parametric in the types of cell source code `Code`,
cell outputs `Output`, locations `L`, and values `V`.  The paper's
location grammar `ℓ ::= x | d.c` is provided as the inductive type
`FlowBook.Loc`, the intended instantiation for `L`; none of the
metatheory depends on the structure of locations.
-/

namespace FlowBook

/-! ## Locations

The paper: "A location `ℓ ∈ Loc` identifies a unit of state that cells
may read or write: `ℓ ::= x | d.c`" where `x` is a top-level variable,
`d` a DataFrame address, and `c` a column name.  The metatheory below is
parametric in an arbitrary type `L` of locations; `Loc` is the intended
instantiation. -/

/-- The paper's location grammar: a top-level variable binding `x`, or
column `c` of the DataFrame at address `d`. -/
inductive Loc (Var Addr Col : Type) where
  | var (x : Var)
  | col (d : Addr) (c : Col)
deriving DecidableEq, Repr

/-! ## Stores -/

/-- A store maps locations to values; `none` means the location is
unbound.  The empty store binds nothing. -/
def Store (L V : Type) := L → Option V

/-- The empty store `∅`. -/
def Store.empty {L V : Type} : Store L V := fun _ => none

/-- `AgreeExcept σ σ' X`: stores `σ` and `σ'` agree on every location
outside `X`.  This is the paper's "`σ` and `σ'` agree except on `X`". -/
def AgreeExcept {L V : Type} (σ σ' : Store L V) (X : L → Prop) : Prop :=
  ∀ ℓ, ¬ X ℓ → σ ℓ = σ' ℓ

/-- `AgreeOn σ σ' X`: stores `σ` and `σ'` agree on every location in `X`. -/
def AgreeOn {L V : Type} (σ σ' : Store L V) (X : L → Prop) : Prop :=
  ∀ ℓ, X ℓ → σ ℓ = σ' ℓ

/-! ## Black-box cell evaluation

The paper defines instrumented cell evaluation
`c ; Σ ⇓ o · Σ' · r · w`: executing code `c` in store `Σ` produces
output `o`, store `Σ'`, the set `r ⊆ Loc` of locations *read from the
incoming store*, and the set `w ⊆ Loc` of locations where `Σ'` is
updated from `Σ`.  Evaluation "models the underlying language runtime
(Python)" as a black box and "is a *relation* rather than a function"
because cells may be non-deterministic.

Treating evaluation as a black box, the metatheory needs exactly two
assumptions, both implicit in the paper's proofs (the `locality`/frame
property is stated in a remark in the paper):

* `frame`: the resulting store agrees with the incoming store outside
  the write set (this is the defining property of the write set: "the
  set of write locations `w` where `Σ'` is updated from `Σ`");
* `locality`: a cell's behavior depends only on the locations it reads.
  If `Σ̂` agrees with `Σ` on `r`, the same execution is available from
  `Σ̂`: same output, same read and write sets, writing the same values.
  Fixing the values of the read locations pins down a particular
  execution even for non-deterministic cells. -/
class CellEval (Code Output L V : Type) where
  /-- The instrumented big-step judgment `c ; σ ⇓ o · σ' · r · w`. -/
  Eval : Code → Store L V → Output → Store L V → (L → Prop) → (L → Prop) → Prop
  /-- The store is unchanged outside the write set. -/
  frame : ∀ {c σ o σ' r w}, Eval c σ o σ' r w →
    ∀ ℓ, ¬ w ℓ → σ' ℓ = σ ℓ
  /-- Evaluation depends only on the read locations: from any store
  agreeing on `r`, the same execution (same output, reads, writes, and
  written values) is available. -/
  locality : ∀ {c σ o σ' r w τ}, Eval c σ o σ' r w →
    (∀ ℓ, r ℓ → τ ℓ = σ ℓ) →
    ∃ τ', Eval c τ o τ' r w ∧
      (∀ ℓ, w ℓ → τ' ℓ = σ' ℓ) ∧ (∀ ℓ, ¬ w ℓ → τ' ℓ = τ ℓ)

export CellEval (Eval)

variable {Code Output L V : Type}

/-- Standard (uninstrumented) cell evaluation `c ; σ ⇓ o · σ'`
(the paper's `\StdEvalCell`), obtained from instrumented evaluation by
erasing the read and write sets. -/
def StdEval [CellEval Code Output L V]
    (c : Code) (σ : Store L V) (o : Output) (σ' : Store L V) : Prop :=
  ∃ r w, Eval c σ o σ' r w

/-! ## Notebook operations

`op ::= Run(i) | Edit(i, c) | Insert(i, c) | Delete(i) | Move(i, j)`.

Cell positions are 0-based here; the paper uses 1-based indices. -/
inductive Op (Code : Type) where
  | run (i : Nat)
  | edit (i : Nat) (c : Code)
  | insert (i : Nat) (c : Code)
  | delete (i : Nat)
  | move (s d : Nat)

/-! ## The standard semantics (Figure 8)

A standard notebook state is `S = (C, O, Σ)`.  We store the code and
output sequences as a single list of pairs; `O_i = none` means cell `i`
has not yet been executed (the paper's `O_i = ⊥`). -/

/-- Standard notebook state `S = (C, O, Σ)`. -/
structure StdState (Code Output L V : Type) where
  cells : List (Code × Option Output)
  store : Store L V

/-- The standard semantics `S ─op→ S'` of Figure 8.

`Std-Run` permits executing *any* cell at *any* time; `Std-Move` is the
composition of a delete and an insert, exactly as in the figure. -/
inductive StdStep [CellEval Code Output L V] :
    StdState Code Output L V → Op Code → StdState Code Output L V → Prop where
  /-- `[Std-Run]`: evaluate cell `i` in the current store, record its
  output, and update the store. -/
  | run {st : StdState Code Output L V} {i c o₀ o σ'}
      (hcell : st.cells[i]? = some (c, o₀))
      (heval : StdEval c st.store o σ') :
      StdStep st (.run i) ⟨st.cells.set i (c, some o), σ'⟩
  /-- `[Std-Edit]`: replace the source code of cell `i`; the recorded
  output and the store are unchanged. -/
  | edit {st : StdState Code Output L V} {i c₀ o₀ c}
      (hcell : st.cells[i]? = some (c₀, o₀)) :
      StdStep st (.edit i c) ⟨st.cells.set i (c, o₀), st.store⟩
  /-- `[Std-Insert]`: insert a fresh, never-executed cell at position `i`. -/
  | insert {st : StdState Code Output L V} {i c}
      (hle : i ≤ st.cells.length) :
      StdStep st (.insert i c) ⟨st.cells.insertIdx i (c, none), st.store⟩
  /-- `[Std-Delete]`: remove the cell at position `i`. -/
  | delete {st : StdState Code Output L V} {i c₀ o₀}
      (hcell : st.cells[i]? = some (c₀, o₀)) :
      StdStep st (.delete i) ⟨st.cells.eraseIdx i, st.store⟩
  /-- `[Std-Move-Down]` (`s < d`): a delete composed with an insert. -/
  | moveDown {st st'' st' : StdState Code Output L V} {s d c₀ o₀}
      (hlt : s < d)
      (hcell : st.cells[s]? = some (c₀, o₀))
      (h1 : StdStep st (.delete s) st'')
      (h2 : StdStep st'' (.insert (d - 1) c₀) st') :
      StdStep st (.move s d) st'
  /-- `[Std-Move-Up]` (`d < s`): a delete composed with an insert. -/
  | moveUp {st st'' st' : StdState Code Output L V} {s d c₀ o₀}
      (hlt : d < s)
      (hcell : st.cells[s]? = some (c₀, o₀))
      (h1 : StdStep st (.delete s) st'')
      (h2 : StdStep st'' (.insert d c₀) st') :
      StdStep st (.move s d) st'

/-! ## Top-to-bottom execution and reproducibility -/

/-- `Runs σ cells σ'` is the paper's top-to-bottom execution judgment
(`Std-Top-to-Bottom-Execution`), relativized to a starting store:
executing the cells in order from `σ` produces *exactly the recorded
outputs* and the final store `σ'`.  In particular every cell must have
a recorded output (`O_i ≠ ⊥`). -/
inductive Runs [CellEval Code Output L V] :
    Store L V → List (Code × Option Output) → Store L V → Prop where
  | nil {σ : Store L V} : Runs σ [] σ
  | cons {σ σ₁ σ' : Store L V} {c : Code} {o : Output}
      {rest : List (Code × Option Output)}
      (hc : StdEval c σ o σ₁) (hrest : Runs σ₁ rest σ') :
      Runs σ ((c, some o) :: rest) σ'

/-- Definition (Reproducible / Output-Consistent State).  A notebook
state `S = (C, O, Σ)` is reproducible iff some top-to-bottom execution
of all cells from the empty store produces exactly the outputs
currently recorded in the notebook.  The final stores need not agree;
only the outputs must. -/
def Reproducible [CellEval Code Output L V] (st : StdState Code Output L V) : Prop :=
  ∃ σ' : Store L V, Runs Store.empty st.cells σ'

/-- The paper also calls reproducibility "output consistency"
(Definition 1.1 of the supplement). -/
abbrev OutputConsistent [CellEval Code Output L V] (st : StdState Code Output L V) :
    Prop :=
  Reproducible st

/-- Appending one executed cell to a top-to-bottom run. -/
theorem Runs.snoc [CellEval Code Output L V]
    {σ τ τ' : Store L V} {l : List (Code × Option Output)} {c : Code} {o : Output}
    (h : Runs σ l τ) (hc : StdEval c τ o τ') :
    Runs σ (l ++ [(c, some o)]) τ' := by
  induction h with
  | nil => exact .cons hc .nil
  | cons hc' _ ih => exact .cons hc' (ih hc)

end FlowBook
