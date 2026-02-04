# Flowbook Execution Guarantees and Restrictions Report

## Executive Summary

Flowbook (DataFerret) is a JupyterLab extension with a custom IPython kernel that provides **deterministic notebook execution** through state checkpointing, namespace diffing, and serial execution enforcement. The system guarantees that notebook state changes can be tracked, compared, and restored, but this comes with specific restrictions on what Python objects can participate in these guarantees.

---

## Part 1: Execution Guarantees

### 1.1 Serial Execution Guarantee

**Location:** `ferret_kernel.py:474-612`

Flowbook enforces strictly serial cell execution with the following properties:

- **Single-threaded execution**: Each cell runs synchronously and blocks until complete
- **Per-cell timeout enforcement**: Default 30 minutes, configurable via `# timeout N` comment at cell start
- **Watchdog thread mechanism**: A daemon timer thread monitors execution time
- **Escalating interruption strategy**:
  1. First, raises `KeyboardInterrupt` in the main thread
  2. After a grace period (`_post_kb_grace = 1.0s`), terminates child processes
  3. Finally kills any stubborn child processes

**Child Process Cleanup** (`ferret_kernel.py:37-125`):

- Shuts down joblib/loky parallel executors cleanly
- Resets the executor singleton to ensure fresh state
- Recursively terminates and kills all child processes via `psutil`

### 1.2 State Checkpointing Guarantee

**Location:** `checkpoint.py:87-208`

Checkpoints provide **deep copy semantics** with these guarantees:

- **Snapshot isolation**: Checkpoints are independent copies; modifications don't affect saved state
- **Alias preservation**: The `reverse_memo` tracks which checkpointed variables are aliases of each other
- **Atomic save/restore**: Either a complete checkpoint is saved/restored or the operation fails
- **Optional sanity checking**: Can verify that deep copies match originals (`sanity_check=True`)

```python
# The checkpoint stores:
class Checkpoint:
    name: str                      # Checkpoint identifier
    user_ns: Dict[str, Any]        # Deep copied namespace
    reverse_memo: Dict[int, Any]   # Maps id() to original keys for alias tracking
```

### 1.3 Deterministic Diff Guarantee

**Location:** `diff.py:26-1113`

The `Diff` class provides **isomorphic pointer structure comparison**:

- **Deterministic output**: Same inputs always produce same diff results
- **Sorted key iteration**: Common keys compared in sorted order for reproducibility
- **NaN equality**: `NaN == NaN` is treated as true throughout the system
- **Configurable tolerance**: Float comparisons use `rtol` (relative) and `atol` (absolute) tolerances
- **Pointer structure verification**: Detects when two namespaces have identical values but different aliasing patterns

**Example of pointer structure mismatch:**

```python
# Namespace A
a = [1, 2, 3]
b = a  # Same object - aliased

# Namespace B - same VALUES but different structure
x = [1, 2, 3]
y = [1, 2, 3]  # Different objects - NOT aliased

# differ.diff(ns_a, ns_b) → ERROR: pointer structure mismatch
```

### 1.4 Type Tracking Guarantee

**Location:** `extended_types.py:1-320`

Every variable's type is captured with rich metadata:

- **NumPy arrays**: shape, dtype
- **Pandas DataFrames**: row count, column names and dtypes
- **Pandas Series**: dtype, length
- **Functions**: name, parameters, return type (if annotated)
- **Containers**: element type unions for lists, dicts, sets

### 1.5 Test Code Guarantee

**Location:** `ferret_kernel.py:362-407`

The `test_code` operation provides **A/B comparison** with guarantees:

1. Original environment is saved
2. Original code executes, result checkpointed
3. Environment restored to original state
4. Modified code executes, result checkpointed
5. Both results diffed for equivalence
6. Timing information captured (speedup ratio calculated)

---

## Part 2: Restrictions

### 2.1 Non-Checkpointable Types

**Location:** `checkpoint.py:125-147`

The following **cannot be checkpointed** and are silently removed:

| Type                                           | Reason                      | Code Location                                 |
| ---------------------------------------------- | --------------------------- | --------------------------------------------- |
| **Modules**                                    | Cannot deep copy            | `isinstance(v, types.ModuleType)`             |
| **Matplotlib objects**                         | Complex internal state      | `type(v).__module__.startswith("matplotlib")` |
| **NumPy arrays containing matplotlib objects** | Nested matplotlib reference | Check on `object` dtype arrays                |
| **Any type failing `copy.deepcopy()`**         | Serialization impossible    | Try/except in save loop                       |

**Practical implications:**

- Variables referencing `import numpy as np` are excluded
- Matplotlib figures, axes, artists cannot be checkpointed
- Open file handles, sockets, database connections fail

### 2.2 Filtered System Variables

**Location:** `checkpoint.py:16-29`

These IPython system variables are **always excluded**:

```python
SYSTEM_VARIABLES = {
    "get_ipython", "In", "Out", "exit", "quit",
    "_", "__", "___", "_i", "_ii", "_iii", "_dh"
}
```

Additionally, any variable starting with `_` (underscore) is filtered out.

### 2.3 Aliasing Requirements

**Location:** `diff.py:164-219`, `test_diff.py:416-595`

For two namespaces to be considered equal, they must have **isomorphic pointer structure**:

**Restriction:** If variable `a` and `b` reference the same object in namespace A, then variables `a` and `b` must also reference the same object in namespace B.

```python
# INVALID - aliasing not preserved
a = {'list': [1,2,3], 'alias': [1,2,3]}  # Different objects
# vs
b = {'list': obj, 'alias': obj}           # Same object
# → Pointer structure mismatch ERROR

# VALID - aliasing matches
shared = [1,2,3]
a = {'list': shared, 'alias': shared}
shared2 = [1,2,3]
b = {'list': shared2, 'alias': shared2}
# → Equal (both have matching alias structure)
```

### 2.4 Type Strictness Restrictions

**Location:** `diff.py:221-256, 307-339`

By default (`strict=True`), types must match exactly:

- `int` vs `float` → **Type mismatch error**
- `list` vs `tuple` → **Type mismatch error**
- `list` vs `np.ndarray` → **Type mismatch error**
- `np.int32` vs `np.int64` → **Type mismatch error**

In non-strict mode (`strict=False`), these become compatible:

- `int` ↔ `float` (int converted to float for comparison)
- `list`/`tuple` ↔ `np.ndarray` (structural comparison)

### 2.5 NumPy Array Restrictions

**Location:** `diff.py:593-640`

NumPy arrays must match in:

- **Shape**: Exactly equal dimensions
- **Dtype**: Exactly matching data types
- **Values**: Element-wise comparison (with tolerance for floats)

**Floating point specifics:**

- `np.allclose()` used with configurable `rtol=1e-5`, `atol=1e-8`
- `equal_nan=True` - NaN values are considered equal

### 2.6 Pandas Object Restrictions

**Location:** `diff.py:642-838`

**Series requirements:**

- Index must be equal (via `.equals()`)
- Name must match exactly
- Dtype must match
- Values must match (NaN-aware)

**DataFrame requirements:**

- Shape must match
- Columns must match (order and names)
- Index must match
- Per-column values must match

**GroupBy special handling:**

- Internal cache (`_cache`, `_grouper._cache`) is **ignored**
- Only semantic properties compared: underlying data, grouper config, selection

### 2.7 Container Size Limits

**Location:** `diff.py:37-38`, `extended_types.py:12`

- `max_diffs_per_container = 1000`: Stops collecting differences after 1000 per container
- `INSPECTION_LIMIT = 10`: Only first 10 elements inspected for type inference

### 2.8 Callable Comparison Restrictions

**Location:** `diff.py:547-591`

- **Functions**: Currently **ignored** in comparison (line 578-579 returns `None`)
- **Bound methods**: Compare `__func__` and `__self__` recursively
- **Method type mismatch**: Bound method vs regular function → Error

### 2.9 Object Comparison Requirements

**Location:** `diff.py:1034-1112`

Custom objects must have `__dict__` for comparison:

- Objects without `__dict__` fall back to `!=` operator
- Objects where `!=` raises an exception cause comparison errors
- All public attributes are recursively compared

### 2.10 Class Variables Are Not Tracked

**Location:** `diff.py:1066-1069`

Object comparison only examines instance attributes (`__dict__`), **not class-level attributes**:

```python
# Only instance attributes are compared:
dict_a = val_a.__dict__
dict_b = val_b.__dict__
```

Class variables defined on the class itself (via `cls.attr` or in the class body) are not tracked or compared.

#### Justification: Why This Is Acceptable for Notebooks

**1. Notebooks operate on data instances, not class definitions**

The primary workflow in data science notebooks is:

- Load/create data → Transform data → Analyze results
- Users work with DataFrame instances, array instances, model instances
- Class-level state mutation is extremely rare in notebook workflows

**2. Class definitions are code, not state**

In the notebook execution model, classes are defined once (like functions) and then instantiated. The meaningful state that changes between cells lives in the instances:

```python
# Cell 1: Define class (code)
class Model:
    learning_rate = 0.01  # Class variable - rarely mutated

# Cell 2: Create instance (state)
model = Model()
model.weights = trained_weights  # Instance attribute - this is the state we care about
```

**3. Re-executing a cell redefines the class entirely**

When a user re-runs a cell containing a class definition, Python creates an entirely new class object. Any "changes" to class variables are reset:

```python
# Cell 1 (first execution)
class Config:
    debug = False

# Cell 2
Config.debug = True  # Mutate class variable

# Cell 1 (re-execution) - Creates NEW class, debug is False again
class Config:
    debug = False
```

Tracking class variable changes across this re-definition boundary would be confusing and inconsistent with how notebooks work.

**4. Most classes come from modules, which are already excluded**

Classes from imported libraries (`sklearn.linear_model.LinearRegression`, `pandas.DataFrame`, etc.) live in modules. Since modules are excluded from checkpointing, class variables on imported classes wouldn't be tracked anyway. Flowbook is consistent by also not tracking class variables on user-defined classes.

**5. Implementation complexity would be substantial**

Properly tracking class variables would require:

- Traversing the Method Resolution Order (MRO) for inherited class variables
- Handling metaclasses and descriptors
- Distinguishing class variables from class methods
- Dealing with `__slots__` classes
- Managing the class object's own `__dict__`

This complexity is not justified given the rarity of class variable mutation in notebook workflows.

**6. Stateful class patterns are anti-patterns in notebooks**

Using class variables as mutable shared state is generally considered poor practice:

```python
# Anti-pattern: Mutable class-level state
class Counter:
    count = 0  # Shared across all instances - problematic

    def increment(self):
        Counter.count += 1
```

Notebooks encourage functional, data-centric programming where state flows through variables in the namespace, not hidden in class hierarchies. By not tracking class variables, Flowbook implicitly discourages this anti-pattern.

**7. Instance state captures what matters**

For machine learning and data science use cases, the important state is always in instances:

- Model weights → `model.coef_`, `model.weights`
- Training history → `history.losses`
- Configuration → `config.learning_rate` (instance attribute)
- Data → DataFrame/array instances in the namespace

Class variables typically hold defaults or constants that don't change during execution.

### 2.11 Execution Time Restrictions

**Location:** `ferret_kernel.py:295-296, 497-504`

- **Default timeout**: 30 minutes (`_default_cell_timeout = 30 * 60`)
- **Per-cell override**: `# timeout N` as first line
- **Post-interrupt grace**: 1 second before escalating to process killing

---

## Part 3: Immutable Atomic Types (No Pointer Tracking)

**Location:** `diff.py:77-110`

For these types, **only value equality matters**, not object identity:

- `None`
- `bool`
- `int` (including `np.integer`)
- `float` (including `np.floating`)
- `complex` (including `np.complexfloating`)
- `str`
- `bytes`

This means two integers with the same value are always equal, regardless of whether they're the same object.

---

## Part 4: Float Comparison Semantics

**Location:** `diff.py:477-515`

Floating point comparison follows this priority:

1. **NaN handling**: Both NaN → Equal; one NaN → Different with special message
2. **Exact equality**: `val_a == val_b` → Equal
3. **Tolerance check**: `math.isclose(val_a, val_b, rel_tol=rtol, abs_tol=atol)`
   - If close and `report_close=True` → Status "close" (reported but not error)
   - If close and `report_close=False` → Equal (not reported)
4. **Different**: Outside tolerance → Status "different"

Default tolerances:

- `rtol = 1e-5` (relative)
- `atol = 1e-8` (absolute)

For checkpoint diffs specifically (`checkpoint.py:115`):

- `atol = 1e-6`
- `report_close = False` (close values treated as equal)

---

## Part 5: How Components Work Together

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Executes Cell                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FerretKernel.do_execute()                                      │
│  - Parse timeout directive                                       │
│  - Arm watchdog timer                                           │
│  - Optionally save pre-execution checkpoint                     │
│  - Execute code with Scalene profiling (optional)               │
│  - Capture type models before/after                             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐    ┌──────────────────────────────────┐
│   Normal Exit           │    │   Timeout/Exception              │
│   - Disarm watchdog     │    │   - KeyboardInterrupt raised     │
│   - Display timing      │    │   - Child processes terminated   │
│   - Diff checkpoints    │    │   - Loky/joblib reset            │
│     (if force mode)     │    │   - Exception info via Comm      │
└─────────────────────────┘    └──────────────────────────────────┘
```

### Checkpoint Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    checkpoint.save(name, user_ns)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Filter namespace (remove system vars, underscores, modules) │
│  2. Filter values (remove matplotlib, non-copyable)             │
│  3. Deep copy each variable with shared memo                    │
│  4. Store Checkpoint(name, copied_ns, reverse_memo)             │
│  5. Return (saved_types, removed_types)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Diff Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Diff().diff(namespace_a, namespace_b)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Reset identity tracking maps (id_map_a, id_map_b)           │
│  2. Find variables only in A → "removed"                        │
│  3. Find variables only in B → "added"                          │
│  4. For common variables (sorted):                              │
│     a. Check if immutable atomic → skip pointer tracking        │
│     b. Register object IDs for pointer structure                │
│     c. Dispatch to type-specific comparator                     │
│     d. Detect pointer structure mismatches                      │
│  5. Return DiffResult with tree of differences                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary Table: Key Guarantees vs Restrictions

| Guarantee                                 | Restriction                                                   |
| ----------------------------------------- | ------------------------------------------------------------- |
| Serial, single-threaded execution         | No multi-process state sharing between cells                  |
| Deep copy checkpoints with alias tracking | Modules, matplotlib, non-copyable objects excluded            |
| Deterministic namespace diffs             | Pointer structure must be isomorphic                          |
| NaN equality (NaN == NaN)                 | Floating point tolerance required for numeric comparison      |
| Type tracking for all variables           | Only first 10 elements inspected for containers               |
| Per-cell timeout enforcement              | Maximum 30 minutes default; child processes killed on timeout |
| GroupBy cache-independent comparison      | Internal cache fields ignored                                 |
| Test code A/B comparison                  | Requires checkpointable environment                           |
| Instance attribute tracking               | Class variables not tracked (by design - see §2.10)           |
