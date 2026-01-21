"""
Checkpoint management for kernel state snapshots.

This module provides functionality to save and restore kernel namespace state,
enabling features like undo, state comparison, and reproducible execution.

================================================================================
                              DESIGN DOCUMENT
================================================================================

1. OVERVIEW
-----------
The checkpoint module implements a snapshotting system for Jupyter kernel
namespaces. It allows saving the complete state of user-defined variables
at any point during execution and restoring them later. This enables:

  - Undo/Redo functionality for notebook execution
  - State comparison between execution points (via diff)
  - Reproducible execution by resetting to known states
  - Debugging by inspecting historical variable values
  - Speculative execution with easy rollback

The core challenge is creating true deep copies of complex Python objects
while maintaining reasonable performance for large data science workloads.


2. CRITICAL LIMITATIONS (READ THIS FIRST!)
-------------------------------------------
Before using checkpoints, be aware of these fundamental limitations:

❌ CLASS VARIABLES ARE NOT RESTORED (Section 8.4)
   - User-defined classes can be checkpointed
   - BUT mutable class variables (class-level attributes) will NOT be restored
   - Only instance attributes are properly checkpointed
   - Workaround: Use instance attributes instead of class variables
   - Example of what DOESN'T work:
       class Counter:
           count = 0  # ← This won't be restored!
       cp.save('before', user_ns)
       Counter.count = 100
       cp.restore('before', user_ns)
       # Counter.count is STILL 100 (not restored to 0)

⚠️  NOT THREAD-SAFE
   - Concurrent save/restore operations will corrupt data
   - Use external locking if checkpointing from multiple threads

⚠️  MATPLOTLIB OBJECTS EXCLUDED (Section 8.2)
   - Matplotlib figures, axes, etc. are automatically filtered out
   - They cannot be reliably deep copied

⚠️  GENERATORS & ITERATORS MAY FAIL (Section 7.7)
   - Generators cannot be pickled (maintain execution state)
   - Iterators may produce unexpected results after restore
   - These will be tracked in the "removed" dictionary


3. ARCHITECTURE
---------------
Key Components:

  Checkpoint          - Single snapshot containing deep-copied namespace
  Checkpoints         - Manager for multiple named checkpoints
  filter_user_namespace() - Filters namespace to user-defined variables
  deepcopy module     - Custom deepcopy with pandas/function support

Data Flow:
  user_ns → filter → deep copy → Checkpoint
  Checkpoint → deep copy → user_ns (on restore)

Dependency Graph:
  checkpoint.py
    ├── deepcopy.py      - Custom deepcopy implementation
    ├── diff.py          - Structured comparison between checkpoints
    ├── extended_types.py - TypeModel generation for reporting
    └── output.py        - Logging and timing utilities


3. DESIGN GOALS
---------------
  a) Correctness: Mutations to restored variables must NOT affect the
     stored checkpoint. Conversely, mutations to the live namespace
     must NOT affect stored checkpoints.

  b) Performance: Checkpointing should be fast enough for interactive use.
     Target: <1s for typical data science notebooks (DataFrames up to ~1M rows).

  c) Memory Efficiency: Use copy-on-write for non-object DataFrame/Series
     columns to minimize memory overhead until mutation occurs.

  d) Transparency: Users shouldn't need to understand checkpointing internals.
     Variable behavior should be identical before/after checkpoint/restore.


4. DESIGN DECISIONS & RATIONALE
-------------------------------

4.1 Custom Deepcopy Module
---------------------------
Problem: Python's copy.deepcopy() doesn't handle pandas objects optimally.
  - DataFrames need special handling for object dtype columns
  - Series need special handling for object dtype
  - Functions need closure and mutable default copying

Solution: Custom deepcopy module (flowbook.kernel.deepcopy) that:
  - Follows standard library's dispatch pattern for consistency
  - Adds custom handlers for pd.DataFrame, pd.Series, types.FunctionType
  - Falls back to standard deepcopy logic for everything else
  - Maintains shared memo dictionary for circular reference handling

The memo dictionary is shared across all copies to ensure that if the same
object appears multiple times in the namespace, it's only copied once and
all references point to the same copy.


4.2 Copy-on-Write (CoW) Integration
-----------------------------------
Pandas 2.0+ supports copy-on-write mode (pd.options.mode.copy_on_write = True).
When enabled:
  - Shallow copies of DataFrames share underlying data until mutation
  - First write to either copy triggers actual data duplication
  - This makes shallow copy O(1) for memory/time initially

We enable CoW globally and rely on it for non-object columns. Object columns
still require explicit deep copying because the objects they contain could
be mutable (lists, dicts, custom objects).


4.3 Restore Creates New Copies
------------------------------
When restoring a checkpoint, we deep copy from the checkpoint rather than
just updating references. This ensures:
  - Multiple restores from the same checkpoint work correctly
  - The stored checkpoint remains pristine
  - Mutations after restore don't affect the checkpoint


4.5 Namespace Filtering
-----------------------
Not all variables in the kernel namespace should be checkpointed:

Excluded by name:
  - Variables starting with underscore (_private, __dunder__)
  - IPython system variables (In, Out, _, __, ___, _i, _ii, etc.)

Excluded by type:
  - Module objects (import numpy as np → np is not checkpointed)
  - Matplotlib objects (figures, axes - can't be reliably deep copied)

This filtering happens at checkpoint time and restore time to ensure
consistency.


4.6 Reverse Memo for Identity Tracking
--------------------------------------
The Checkpoint class stores a reverse_memo mapping copied object IDs back
to original memo keys. This enables:
  - Tracking which objects are shared references vs. independent copies
  - Diff algorithms to detect structural changes (aliasing)
  - Debugging copy behavior

Example:
  original_list = [1, 2, 3]
  namespace = {"a": original_list, "b": original_list}  # Same object

After copying, both copies should point to the same copied list.
The reverse_memo maps: id(copied_list) → id(original_list)


5. IMPLEMENTATION DETAILS
-------------------------

5.1 _deep_copy_user_ns() Method
-------------------------------
Central method that handles all deep copying:

  def _deep_copy_user_ns(variables):
      copied = {}
      memo = {}      # Shared across all copies
      failed = {}    # Track failures

      for name, value in variables.items():
          try:
              copied[name] = deepcopy(value, memo)
          except Exception as e:
              failed[name] = e

      return copied, memo, failed

The memo dictionary is crucial: it ensures objects referenced multiple
times in the namespace are copied only once.


5.2 Deepcopy Module Implementation
----------------------------------
The deepcopy module uses a dispatch table pattern identical to Python's
standard library copy module:

  _deepcopy_dispatch = {
      pd.DataFrame: _deepcopy_dataframe,
      pd.Series: _deepcopy_series,
      types.FunctionType: _deepcopy_function,
      list: _deepcopy_list,
      dict: _deepcopy_dict,
      # ... etc
  }

Custom handlers:
  - _deepcopy_dataframe: shallow copy + deep copy object columns
  - _deepcopy_series: shallow copy + deep copy if object dtype
  - _deepcopy_function: deep copy closure and mutable defaults

See flowbook/kernel/deepcopy.py for full implementation.


5.3 Sanity Check Mode
---------------------
Optional sanity_check=True mode in Checkpoints:
  - After saving, compares original namespace to checkpoint
  - Uses Diff class from diff.py for structured comparison
  - Raises ValueError if differences found
  - Useful for debugging but expensive (doubles comparison work)


6. ASSUMPTIONS
--------------
  a) Pandas CoW is enabled globally (we set it in deepcopy.py)
  b) Object dtype columns contain only picklable/deepcopyable objects
  c) Variables follow Python naming conventions (underscore prefix = private)
  d) Namespaces are dict-like with string keys
  e) Matplotlib objects are not critical to checkpoint (excluded)
  f) Users don't rely on exact object identity across checkpoint/restore


7. CORNER CASES
---------------

7.1 Circular References
-----------------------
Handled correctly via deepcopy's memo mechanism:
  a = []
  a.append(a)  # Self-referential list

The memo ensures the copied list references itself, not the original.


7.2 Object Columns with Mixed Types
-----------------------------------
  df["mixed"] = [1, "string", [1, 2, 3], None]

All object columns are deep copied, so the list is properly isolated.


7.3 Namespace Changes During Checkpoint
---------------------------------------
If code modifies the namespace while save() is running (threading),
behavior is undefined. This module is NOT thread-safe.


7.4 Variables with Custom __deepcopy__
--------------------------------------
Objects implementing __deepcopy__ are handled by deepcopy module's
standard path. The memo is passed through, so object-level customization
is respected.


7.5 Extension Types (Cython, C extensions)
------------------------------------------
May or may not be deepcopyable. Failures are caught and tracked in the
`failed` dict returned by _deep_copy_user_ns(). The save() method reports
these as `removed` variables.


7.6 Large Objects
-----------------
No size limits are enforced. A 10GB DataFrame will be copied (slowly).
Consider this when using checkpoints with large data.


7.7 Generators and Iterators
----------------------------
Cannot be meaningfully copied - their state is internal/hidden.
Will likely appear in `failed` dict or produce empty/exhausted copies.


7.8 Open File Handles, Network Connections
------------------------------------------
These are excluded if they have matplotlib-like modules or fail deepcopy.
No special handling - rely on deepcopy failures.


8. KNOWN ISSUES & LIMITATIONS
-----------------------------

8.1 Memory Overhead
-------------------
Each checkpoint duplicates all mutable data. With CoW, non-mutated data
shares memory, but:
  - Object columns are always fully copied
  - Modifying either original or copy triggers full copy of modified column
  - Multiple checkpoints multiply memory usage

Mitigation: Clear unneeded checkpoints via delete() or clear()


8.2 Matplotlib Exclusion
------------------------
Matplotlib figures/axes are excluded entirely because:
  - They contain circular references to backend objects
  - Their state includes C-level resources
  - Copying often fails or produces broken objects

Users must recreate plots after restore.


8.3 Function/Lambda Checkpointing
---------------------------------
Functions are handled via deepcopy module's _deepcopy_function():
  - Closure cell contents are deep copied using the shared memo
  - New cell objects are created with the copied contents
  - Mutable default arguments ([], {}, set()) are deep copied
  - A new function object is built with the new closure and defaults
  - Functions without closures AND without mutable defaults are unchanged

This ensures that restored functions have isolated closure and default state.

Design notes (NOT problems in normal usage):
  - __globals__ is shared and points to user_ns. Since restore() modifies
    user_ns in place, functions automatically see restored global values.
  - Lambdas are FunctionType objects, handled identically to regular functions.
  - Bound methods: deepcopy + memo ensures __self__ points to the same
    copied object as the namespace variable.
  - Recursive functions: work correctly because __globals__ (user_ns) is
    modified in place during restore, so recursive calls find the restored
    function.

EDGE CASE WARNING - Direct checkpoint access:
  If you access checkpoint.user_ns['func'] directly WITHOUT restoring,
  the function's __globals__ still points to the LIVE user_ns, not the
  checkpoint's namespace. Calling such functions will:
    - Use LIVE global variable values (not checkpoint values)
    - Make recursive calls to LIVE functions (not checkpoint copies)
    - Modify LIVE namespace (not checkpoint)

  This is only an issue for unusual usage patterns. Normal usage
  (restore then use) works correctly.


8.4 Class Definitions Not Properly Checkpointed
------------------------------------------------
User-defined classes are NOT properly deep copied. Python's copy.deepcopy()
returns the same class object for types. This causes several issues:

PROBLEM 1: Mutable class variables are NOT restored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class Counter:
      count = 0  # Mutable class variable
      instances = []  # Another mutable class variable

  obj = Counter()
  checkpoint.save('test', user_ns)

  Counter.count = 100
  Counter.instances.append(obj)

  checkpoint.restore('test', user_ns)
  # Counter.count is STILL 100, not 0!
  # Counter.instances is STILL [obj], not []!

PROBLEM 2: Methods added/removed from classes persist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class Extensible:
      def original(self):
          return "original"

  checkpoint.save('test', user_ns)

  Extensible.new_method = lambda self: "new"
  del Extensible.original

  checkpoint.restore('test', user_ns)
  # Extensible.new_method STILL exists!
  # Extensible.original is STILL deleted!

PROBLEM 3: Class redefinition DOES work (but may confuse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  class MyClass:
      value = 1

  checkpoint.save('test', user_ns)

  class MyClass:  # Completely new class object
      value = 999

  checkpoint.restore('test', user_ns)
  # MyClass.value is 1 (restored correctly because it's a different object)

WHY THIS HAPPENS:
  - copy.deepcopy(SomeClass) returns SomeClass (same object)
  - The checkpoint stores a reference to the live class, not a copy
  - Modifying the class modifies the checkpoint's reference too

WORKAROUND 1: Use instance attributes instead of class variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # BAD - class variable won't be restored
  class Counter:
      count = 0

  # GOOD - instance attribute will be restored
  class Counter:
      def __init__(self):
          self.count = 0

WORKAROUND 2: Store mutable state in a separate variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # BAD
  class Config:
      settings = {'debug': False}

  # GOOD
  class Config:
      pass
  config_settings = {'debug': False}  # Stored separately, will be restored

WORKAROUND 3: Redefine the class entirely after restore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # If you need to reset a class, redefine it after restore:
  checkpoint.restore('test', user_ns)
  class Counter:  # Redefine to get fresh class
      count = 0

INSTANCES ARE CHECKPOINTED CORRECTLY:
  - Instance attributes (__dict__) are deep copied
  - Only class-level state has issues
  - If your class only uses instance attributes, checkpointing works fine


9. PERFORMANCE CONSIDERATIONS
-----------------------------

9.1 Timing Characteristics
--------------------------
Operation timings for typical data science notebook:
  - Empty namespace: <1ms
  - 100 scalar variables: ~10ms
  - 1MB DataFrame (no object cols): ~50ms (mostly CoW setup)
  - 1MB DataFrame (object cols): ~500ms (must iterate cells)
  - 100MB DataFrame (no object cols): ~100ms
  - 100MB DataFrame (object cols): ~5s

Key insight: object dtype columns dominate checkpoint time.


9.2 Optimization Flags
----------------------
Constructor options:
  - convert_object_to_specialized=True (default): Convert object→specialized dtype

Disabling this may be useful for debugging but hurts performance.


9.3 Memory/Time Tradeoff
------------------------
  - Shallow copy + CoW: Fast, memory-efficient until mutation
  - Deep copy: Slow, doubles memory immediately

The current strategy (shallow + selective deep) balances both.


9.4 Large Primitive List Caching
--------------------------------
For large lists (>= 1000 elements) containing only PRIMITIVE immutable types
(None, bool, int, float, complex, str, bytes), the deepcopy module caches
the checkpoint copy and reuses it on subsequent checkpoints if the list
hasn't changed.

IMPORTANT: Only primitive types are cached - NOT tuples, frozensets, etc.
This keeps the eligibility check very fast (O(n) type checks, no recursion).

How it works:
  1. First checkpoint: O(n) primitive check + O(n) hash + O(n) shallow copy
  2. Subsequent checkpoints (unchanged): O(n) hash check only → cache hit
  3. If list modified: O(n) hash detects change → new copy created

Alias traversal optimization:
  The cache is also used by alias detection (_collect_reachable_ids):
  - Cached lists contain only primitives → no nested mutable references
  - Alias detection skips element-by-element traversal for cached lists
  - This turns O(n) traversal into O(1) cache lookup + O(n) hash

Benefits:
  - Repeated checkpoints of unchanged large lists are much faster
  - Alias detection is faster for large primitive lists
  - Common in data science: large lists of indices, feature names, etc.

Cache management:
  - Cache cleared when all checkpoints are deleted (Checkpoints.clear())
  - Cache cleared when last checkpoint is deleted (Checkpoints.delete())
  - Cache auto-prunes when it exceeds 100 entries
  - Cache entries keyed by list id + content hash for change detection

Configuration (in flowbook.kernel.deepcopy):
  - _LARGE_LIST_THRESHOLD = 1000 (minimum list size for caching)
  - _MAX_LIST_CACHE_SIZE = 100 (maximum cached lists)

Diagnostics:
  - get_list_cache_stats() returns cache size and configuration
  - Log messages show cache hits/misses when FLOWBOOK_LOG=1


10. USAGE EXAMPLES
------------------

Example 1: Basic Save/Restore
    from flowbook.kernel.checkpoint import Checkpoints

    cp = Checkpoints()

    # Save current state
    saved, removed = cp.save("before_experiment", user_ns)
    print(f"Saved {len(saved)} variables, removed {len(removed)}")

    # ... run experiments that modify user_ns ...

    # Restore to saved state
    cp.restore("before_experiment", user_ns)

Example 2: Undo/Redo Pattern
    cp = Checkpoints()

    # Stack of checkpoints for undo
    undo_stack = []

    def save_checkpoint(name):
        cp.save(name, user_ns)
        undo_stack.append(name)

    def undo():
        if undo_stack:
            name = undo_stack.pop()
            cp.restore(name, user_ns)

    # Usage
    save_checkpoint("state_1")
    # ... make changes ...
    save_checkpoint("state_2")
    # ... make more changes ...
    undo()  # Back to state_2
    undo()  # Back to state_1

Example 3: Speculative Execution with Rollback
    cp = Checkpoints()

    # Save before risky operation
    cp.save("before_experiment", user_ns)

    try:
        # Run experimental code that might fail or produce bad results
        df = complex_data_transformation(df)
        model = train_model(df)

        if model.score < 0.8:
            # Results not good enough, rollback
            cp.restore("before_experiment", user_ns)
            print("Rolled back due to poor results")
        else:
            print("Success! Keeping changes")
    except Exception as e:
        # Error occurred, rollback
        cp.restore("before_experiment", user_ns)
        print(f"Rolled back due to error: {e}")

Example 4: Debugging with Historical State
    cp = Checkpoints()

    # Save state at various execution points
    cp.save("after_data_load", user_ns)
    # ... process data ...
    cp.save("after_cleaning", user_ns)
    # ... train model ...
    cp.save("after_training", user_ns)

    # Later, inspect historical values
    cleaning_state = cp.get("after_cleaning")
    print(f"Data shape after cleaning: {cleaning_state.user_ns['df'].shape}")

    # Restore to investigate specific point
    cp.restore("after_cleaning", user_ns)
    # ... debug data cleaning issues ...

Example 5: Optional Features
    # Disable size warnings for large checkpoints
    cp.save("big_data", user_ns, max_size_mb=None)

    # Enable sanity checking (expensive, for debugging)
    cp = Checkpoints(sanity_check=True)

    # Disable class warnings if you know what you're doing
    cp = Checkpoints(warn_classes=False)

Example 6: Comparing Checkpoints
    from flowbook.kernel.checkpoint import Checkpoint

    cp = Checkpoints()

    cp.save("version_1", user_ns)
    # ... make changes ...
    cp.save("version_2", user_ns)

    # Compare two versions
    cp1 = cp.get("version_1")
    cp2 = cp.get("version_2")

    diff_result = Checkpoint.diff(cp1, cp2)
    if diff_result.differences:
        print("Variables changed:")
        for diff in diff_result.differences:
            print(f"  {diff}")

Example 7: Managing Multiple Checkpoints
    cp = Checkpoints()

    # Save multiple states
    for i in range(5):
        # ... process iteration ...
        cp.save(f"iteration_{i}", user_ns)

    # List all checkpoints
    print(f"Saved checkpoints: {cp.list()}")

    # Check if checkpoint exists
    if cp.exists("iteration_3"):
        cp.restore("iteration_3", user_ns)

    # Delete old checkpoints
    cp.delete("iteration_0")
    cp.delete("iteration_1")

    # Clear all checkpoints
    # cp.clear()

Example 8: Performance Tips
    cp = Checkpoints()

    # For large DataFrames, checkpointing object columns with mutable values is expensive
    # Consider using specialized dtypes when possible
    df['int_col'] = df['int_col'].astype('Int64')  # Not object
    df['str_col'] = df['str_col'].astype('string')  # Not object

    # Monitor checkpoint time
    import time
    start = time.time()
    cp.save("big_checkpoint", user_ns)
    print(f"Checkpoint took {time.time() - start:.2f} seconds")


11. DEPENDENCIES
----------------
Internal:
  - flowbook.kernel.deepcopy: Custom deepcopy implementation
  - flowbook.kernel.diff.Diff: Structured diff between checkpoints
  - flowbook.kernel.extended_types.TypeModel, get_type_model: Type introspection
  - flowbook.util.output.log, timer: Logging and performance instrumentation

External:
  - datetime, decimal, time, types: Standard library utilities
  - numpy: Array handling and scalar types
  - pandas: DataFrame/Series handling and dtype inference


12. DEEP ALIAS DETECTION
------------------------
Checkpoints support deep alias detection - identifying when variables share
ANY internal references, not just top-level object identity. This is critical
for the SDC enforcer's backward mutation checking.

12.1 Why It Matters
~~~~~~~~~~~~~~~~~~~
When a cell accesses only some variables, we must still detect changes to
variables that share internal references. Example:

  a = {"b": {"data": 1}}
  c = {"b": a["b"]}  # c["b"] is same object as a["b"]

  # Cell accesses only 'a' and modifies a["b"]["data"]
  a["b"]["data"] = 2  # This ALSO changes c!

Without deep alias detection, we'd miss the change to 'c' and could allow
backward mutations that should be blocked.

12.2 Data Structures
~~~~~~~~~~~~~~~~~~~~
The alias index is built LAZILY on first query and stored in Checkpoint:

  _reachable_ids: Dict[var_name, Set[obj_id]]
      All object IDs reachable from each variable via nested traversal.

  _id_to_vars: Dict[obj_id, Set[var_name]]
      Reverse index: maps each object ID to all variables containing it.
      Enables O(1) lookup: "which variables contain this object?"

  _id_to_paths: Dict[obj_id, Dict[var_name, path_str]]
      Path tracking for logging (e.g., "a['b'] ↔ c['b']").
      Only populated when FLOWBOOK_LOG_DEEP_ALIASES=1.

12.3 What Gets Tracked
~~~~~~~~~~~~~~~~~~~~~~
  - Containers: dict, list, tuple, set, frozenset
  - Pandas: DataFrame (via _mgr), Series, object-dtype columns/elements
  - NumPy: ndarray (via .base for views), object-dtype array elements
  - Custom objects: via __dict__ and __slots__

12.4 What Gets Skipped
~~~~~~~~~~~~~~~~~~~~~~
  - Immutable atomics: None, bool, int, float, str, bytes
      Can't be mutated in-place, no aliasing concern.

  - Singleton types: type, FunctionType, ModuleType, etc.
      Class objects, functions, and modules are singletons in Python.
      Finding them in multiple variables doesn't indicate meaningful aliasing.

      Example: If user creates `base = LGBMRegressor(...)`, the instance
      `base` contains references to the `LGBMRegressor` class object. The
      class object is the same singleton everywhere. Without skipping types,
      we'd falsely mark `base` as aliasing with `LGBMRegressor` and every
      other ML class that shares internal type machinery.

      This was causing 24-minute diff times when a 1-variable diff expanded
      to 47 variables due to false class aliases.

  - Temporary objects: .values, .data properties
      These create temporary arrays/memoryviews whose id() can be reused
      by Python's memory allocator after garbage collection.

  - Circular references: Handled via visited set to prevent infinite loops.

  - Large primitive containers: lists, sets, dicts with >= 1000 elements
      containing only primitive types (None, bool, int, float, str, bytes).
      These are detected via deepcopy's is_primitive_container() function.
      Since primitives can't contain nested mutable refs, element-by-element
      traversal is skipped entirely. This optimization applies to both
      original containers and their cached checkpoint copies.

12.5 Usage
~~~~~~~~~~
  checkpoint = Checkpoint(name, user_ns, memo)

  # Alias index is built lazily on first call
  accessed = {"a", "b"}
  all_relevant = checkpoint.get_aliases_for_vars(accessed, log_aliases=True)
  # Returns accessed + any variables sharing internal references

12.6 Performance
~~~~~~~~~~~~~~~~
  - Index built once per checkpoint (checkpoints are immutable)
  - Lookup is O(accessed_vars + number_of_aliases)
  - Much faster than O(total_objects_in_namespace) runtime traversal

12.7 Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~
  FLOWBOOK_LOG_DEEP_ALIASES=1
      Enable detailed logging of discovered alias relationships.
      Shows paths like "a['b'] ↔ c['b'] (share internal ref)".

  FLOWBOOK_SLOW_ALIAS_THRESHOLD=1000
      Warn if alias collection takes too long for a variable.
      Default: 1000 object-dtype elements.


13. CUDF (GPU DATAFRAME) SUPPORT
---------------------------------
cuDF objects are transparently handled during checkpoint operations.

13.1 Checkpoint Save
~~~~~~~~~~~~~~~~~~~~
When saving, cuDF objects are converted to pandas:
- cudf.DataFrame → pd.DataFrame (GPU→CPU transfer)
- cudf.Series → pd.Series
- cudf.Index → pd.Index

This conversion uses cudf_compat.to_pandas() which handles both:
- Native cuDF objects (calls .to_pandas())
- cudf.pandas proxy objects (extracts _fsproxy_slow)

13.2 Origin Tracking
~~~~~~~~~~~~~~~~~~~~
CuDFOriginTracker records which variables were originally cuDF:
- Before save: cudf_origins.record(var_name, value)
- After restore: cudf_origins.should_convert_back(var_name) → bool

This enables optional GPU restore (not currently implemented).

13.3 cudf.pandas Proxy Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~
cudf.pandas provides transparent GPU acceleration. Proxies wrap pandas
objects and route operations to GPU when beneficial. Key challenges:
- type(proxy).__name__ == 'DataFrame' but may not pass isinstance checks
- Proxies have _fsproxy_slow (pandas) and _fsproxy_fast (cudf) attributes
- Accessing attributes can trigger infinite recursion in wrapped code

Solution: cudf_compat provides safe detection and extraction utilities.


14. OPAQUE OBJECT PATTERN
--------------------------
Some complex objects have simple mutable state. The opaque pattern
provides efficient handling without deep traversal.

14.1 Architecture
~~~~~~~~~~~~~~~~~
OpaqueRegistry (in opaque.py) manages handlers:
- OpaqueHandler.can_handle(obj) → True if handler applies
- OpaqueHandler.get_mutable_state(obj) → extract checkpointable state
- OpaqueHandler.copy_with_state(obj, state, memo) → create copy
- OpaqueHandler.states_equal(state1, state2) → equality check

14.2 Keras Model Handler
~~~~~~~~~~~~~~~~~~~~~~~~
KerasModelHandler checkpoints Keras Sequential/Functional models:
- Only model weights change during training
- Weights extracted via model.get_weights()
- Copy via keras.models.clone_model() + set_weights()
- Rejects unbuilt models (architecture not frozen)

This reduces checkpointing from millions of objects to weight arrays.

14.3 Deferred Keras Import
~~~~~~~~~~~~~~~~~~~~~~~~~~
Importing Keras/TensorFlow takes ~3 seconds. To avoid this penalty:
- _is_keras_model(obj) detects Keras models WITHOUT importing Keras
- Uses module name inspection: 'keras' in type(obj).__module__
- Keras handlers registered lazily when first Keras model encountered

This is implemented in both deepcopy.py and diff.py.


15. CHECKPOINT.DIFF TIMERS
---------------------------
Checkpoint.diff() includes timing for debugging performance issues:

  with timer(key="checkpoint_diff:setup", message="[diff] Setup"):
      # Import and mode initialization
  with timer(key="checkpoint_diff:create", message="[diff] Create Diff object"):
      # Diff constructor
  with timer(key="checkpoint_diff:compare", message="[diff] Compare namespaces"):
      # Actual comparison

Additional profiling available via FLOWBOOK_PROFILE_DIFF=1 environment variable.


16. FUTURE WORK / TODOS
-----------------------
  - [ ] Incremental checkpointing (store only changes from previous)
  - [ ] Checkpoint compression for memory efficiency
  - [ ] Async checkpoint creation (background deep copy)
  - [ ] Checkpoint serialization (save to disk)
  - [ ] Configurable exclusion patterns (not just matplotlib)
  - [ ] Thread safety (at least via locking)
  - [ ] Size estimation before checkpoint (warn on large data)


17. MULTIINDEX COLUMN SUPPORT
-----------------------------
DataFrames with MultiIndex columns (hierarchical column labels) are fully
supported throughout the checkpoint system. This includes:

  - Deep copying of MultiIndex DataFrames
  - Alias detection for variables containing MultiIndex DataFrames
  - Checkpoint diffing between MultiIndex DataFrames

17.1 Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~
When iterating over DataFrame columns, we use positional indexing
(`.iloc[:, col_idx]`) rather than label-based indexing (`df[col]`).
This avoids issues where `df[tuple]` might return a DataFrame instead
of a Series when pandas interprets the tuple as a partial key.

Example patterns used:
  # CORRECT - always returns a Series
  for col_idx in range(len(df.columns)):
      series = df.iloc[:, col_idx]

  # PROBLEMATIC - may return DataFrame for MultiIndex
  for col in df.columns:
      series = df[col]  # Could be DataFrame!

17.2 Supported MultiIndex Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - 2-level and 3+ level column hierarchies
  - Mixed-type level values (strings, integers, None)
  - Duplicate column names within MultiIndex
  - Object-dtype columns with mutable elements
  - Empty MultiIndex DataFrames


================================================================================
                            END DESIGN DOCUMENT
================================================================================
"""

from __future__ import annotations

import os
import time
import types
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from flowbook.kernel.deepcopy import (
    deepcopy,
    _IMMUTABLE_INFERRED_KINDS,
    _convert_object_column_dtype,
    _has_mutable_defaults,
    is_primitive_container,
    _LARGE_LIST_THRESHOLD,
)
from flowbook.kernel.diff import Diff
from flowbook.kernel.opaque import OpaqueRegistry
from pandas.api.types import infer_dtype
from flowbook.kernel.extended_types import TypeModel, get_type_model
from flowbook.util.output import log, output, timer


# Enable copy-on-write mode for better performance with DataFrame copies
pd.options.mode.copy_on_write = True

# Infer string columns as StringDtype instead of object (for better performance)
pd.options.future.infer_string = True

# Environment variable to enable detailed checkpoint profiling
# Set FLOWBOOK_PROFILE_CHECKPOINT=1 to record deepcopy timings keyed by type
_PROFILE_CHECKPOINT = os.environ.get("FLOWBOOK_PROFILE_CHECKPOINT", "0") == "1"


# System variables to filter out from user namespace
SYSTEM_VARIABLES = {
    "get_ipython",
    "In",
    "Out",
    "exit",
    "quit",
    "_",
    "__",
    "___",
    "_i",
    "_ii",
    "_iii",
    "_dh",
}


# =============================================================================
# OBJECT TYPE CONVERSION HELPERS
# =============================================================================


def convert_series_object_to_specialized(series: pd.Series) -> pd.Series:
    """
    Convert object dtype Series to appropriate dtypes when possible.

    Handles: integers, floats, strings, decimals, complex, booleans,
    datetimes, timedeltas, and categorical data. Does NOT parse strings to numbers.

    Args:
        series: Series to convert

    Returns:
        Converted Series (or original if no conversion possible)
    """
    return _convert_object_column_dtype(series)


def convert_dataframe_object_to_specialized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object dtype columns in DataFrame to appropriate dtypes when possible.

    Args:
        df: DataFrame to convert

    Returns:
        DataFrame with converted columns
    """
    df2 = df.copy()
    obj_cols = df2.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        df2[col] = convert_series_object_to_specialized(df2[col])

    return df2


def _deep_copy_function(
    func: types.FunctionType, memo: dict[int, Any], skip_immutable: bool = True
) -> types.FunctionType:
    """
    Deep copy a function, including its closure contents.

    Standard deepcopy doesn't copy closure cell contents, leaving the copied
    function referencing the same objects as the original. This function
    creates a true deep copy by:
    1. Deep copying each cell's contents using the shared memo
    2. Creating new cell objects with the copied contents
    3. Building a new function with the new closure

    Args:
        func: Function to copy
        memo: Shared memo dict for tracking copied objects
        skip_immutable: If True, skip copying immutable objects in closure

    Returns:
        New function with deep-copied closure
    """
    # If no closure and no mutable defaults, return the same function
    if func.__closure__ is None and not _has_mutable_defaults(func):
        return func

    # If no closure but has mutable defaults, need to copy defaults only
    if func.__closure__ is None:
        new_func = types.FunctionType(
            func.__code__,
            func.__globals__,
            func.__name__,
            func.__defaults__,
            None,
        )
        memo[id(func)] = new_func

        new_defaults = (
            tuple(deepcopy(d, memo) for d in func.__defaults__)
            if func.__defaults__
            else None
        )
        new_kwdefaults = (
            {k: deepcopy(v, memo) for k, v in func.__kwdefaults__.items()}
            if func.__kwdefaults__
            else None
        )

        new_func.__defaults__ = new_defaults
        new_func.__kwdefaults__ = new_kwdefaults
        new_func.__annotations__ = func.__annotations__.copy() if func.__annotations__ else {}
        for k, v in func.__dict__.items():
            new_func.__dict__[k] = deepcopy(v, memo)
        new_func.__doc__ = func.__doc__

        return new_func

    # Create a temporary function and register in memo to handle circular refs
    temp_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )
    memo[id(func)] = temp_func

    # Deep copy closure contents
    new_cells = []
    for cell in func.__closure__:
        try:
            copied_contents = deepcopy(cell.cell_contents, memo)
            new_cells.append(types.CellType(copied_contents))
        except ValueError:
            # Empty cell (variable referenced but not yet bound)
            new_cells.append(types.CellType())

    new_closure = tuple(new_cells)

    # Deep copy defaults
    new_defaults = (
        tuple(deepcopy(d, memo) for d in func.__defaults__)
        if func.__defaults__
        else None
    )
    new_kwdefaults = (
        {k: deepcopy(v, memo) for k, v in func.__kwdefaults__.items()}
        if func.__kwdefaults__
        else None
    )

    # Create the actual new function with deep-copied closure and defaults
    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        new_defaults,
        new_closure,
    )
    new_func.__kwdefaults__ = new_kwdefaults
    new_func.__annotations__ = func.__annotations__.copy() if func.__annotations__ else {}
    for k, v in func.__dict__.items():
        new_func.__dict__[k] = deepcopy(v, memo)
    new_func.__doc__ = func.__doc__

    # Update memo to point to the actual function
    memo[id(func)] = new_func

    return new_func


def is_valid_variable_name(name: str) -> bool:
    """
    Check if a variable name should be included in processing.

    Filters out:
    - Names starting with underscore (private/internal)
    - IPython system variables

    Args:
        name: Variable name to check

    Returns:
        True if the variable should be included, False otherwise
    """
    return not name.startswith("_") and name not in SYSTEM_VARIABLES


def is_valid_variable(name: str, value: Any) -> bool:
    """
    Check if a variable (name and value) should be included in processing.

    Filters out:
    - Names starting with underscore (private/internal)
    - IPython system variables
    - Module objects

    Args:
        name: Variable name to check
        value: Variable value to check

    Returns:
        True if the variable should be included, False otherwise
    """
    return is_valid_variable_name(name) and not isinstance(value, types.ModuleType)


def filter_user_namespace(user_ns: dict[str, Any]) -> dict[str, Any]:
    """
    Filter a user namespace to include only valid variables.

    This is a convenience function that applies is_valid_variable() to
    an entire namespace dictionary.

    Args:
        user_ns: User namespace dictionary

    Returns:
        Filtered dictionary with only valid variables
    """
    return {k: v for k, v in user_ns.items() if is_valid_variable(k, v)}


# =============================================================================
# DEEP ALIAS DETECTION
# =============================================================================
# These helpers support the OPT_ACCESSED_VARS_ONLY optimization by detecting
# deep/nested aliases - cases where two variables share internal references.
#
# Example: If a["b"] and c["b"] point to the same object, modifying a["b"]
# also changes c. We need to detect this to ensure correctness.
#
# The index is built ONCE during checkpoint creation (checkpoints are immutable)
# and reused for all subsequent alias expansion queries.
# =============================================================================

# Immutable types that don't need alias tracking (can't be mutated in-place)
_IMMUTABLE_ATOMIC_TYPES = (type(None), bool, int, float, complex, str, bytes)

# Singleton types that should be SKIPPED for alias tracking.
# These are shared objects - finding them in multiple variables does NOT
# indicate meaningful aliasing that could cause backward mutation issues.
#
# Key insight: A user creating `base = LGBMRegressor(...)` will have `base`
# contain references to the class `LGBMRegressor` (through __class__, methods,
# etc.). The `LGBMRegressor` class object is a singleton - it's the same object
# everywhere. If we track its id(), we'll incorrectly mark `base` as aliasing
# with `LGBMRegressor`, `CatBoostClassifier`, and any other class that shares
# internal type machinery.
#
# This caused 24-minute diff times when ML model classes were falsely detected
# as aliases, expanding a 1-variable diff to 47 variables.
_SINGLETON_TYPES = (
    type,  # Class objects (LGBMRegressor, pd.DataFrame, etc.)
    types.FunctionType,  # User-defined functions
    types.BuiltinFunctionType,  # Built-in functions like len, print
    types.BuiltinMethodType,  # Built-in methods
    types.MethodType,  # Bound methods
    types.ModuleType,  # Imported modules
    types.MethodDescriptorType,  # Method descriptors on types
    types.WrapperDescriptorType,  # Wrapper descriptors on types
    types.GetSetDescriptorType,  # getset descriptors
    types.MemberDescriptorType,  # Member descriptors
)

# Environment variable to control deep alias logging
_LOG_DEEP_ALIASES = os.environ.get("FLOWBOOK_LOG_DEEP_ALIASES", "").lower() in ("1", "true", "yes", "on")

# Threshold for warning about slow object-dtype traversal.
# If a container has more object-dtype elements than this, log a warning.
_SLOW_OBJECT_DTYPE_THRESHOLD = int(os.environ.get("FLOWBOOK_SLOW_ALIAS_THRESHOLD", "1000"))


def _collect_reachable_ids_with_paths(
    obj: Any,
    path: str,
    visited: Set[int],
    id_to_path: Dict[int, str],
) -> None:
    """
    Recursively collect all object IDs reachable from obj, tracking paths.

    This traverses the object graph and records both the object ID and
    the path to reach it (e.g., "['key'].attr[0]").

    Args:
        obj: The object to traverse
        path: Current path string (e.g., "['key']" or ".attr")
        visited: Set of already-visited object IDs (to handle cycles)
        id_to_path: Dict mapping object ID to the first path that reached it
    """
    obj_id = id(obj)
    if obj_id in visited:
        return  # Already visited (handles circular refs)

    # Skip immutable atomics - can't be mutated, no aliasing concern
    if isinstance(obj, _IMMUTABLE_ATOMIC_TYPES):
        return

    # Skip numpy scalar types
    if isinstance(obj, (np.integer, np.floating, np.complexfloating, np.bool_)):
        return

    # Skip singleton types (classes, functions, modules, etc.)
    # These are shared objects - finding them in multiple variables doesn't
    # indicate meaningful aliasing. See _SINGLETON_TYPES comment for details.
    if isinstance(obj, _SINGLETON_TYPES):
        return

    # Handle opaque objects (e.g., Keras/PyTorch models) - treat internal structure as
    # atomic, but DO traverse custom attributes for alias detection
    handler = OpaqueRegistry.get_handler(obj)
    if handler is not None:
        visited.add(obj_id)
        if obj_id not in id_to_path:
            id_to_path[obj_id] = path

        # Traverse custom attributes that we capture in get_mutable_state()
        # These may contain aliased references to other notebook variables
        try:
            custom_attrs = handler.get_traversable_attrs(obj)
            for attr_name, attr_value in custom_attrs.items():
                _collect_reachable_ids_with_paths(
                    attr_value, f"{path}.{attr_name}", visited, id_to_path
                )
        except Exception:
            pass  # Best-effort traversal

        return  # Don't recurse into internal opaque object structure

    visited.add(obj_id)
    # Record the first path to this object
    if obj_id not in id_to_path:
        id_to_path[obj_id] = path

    # Recurse into containers
    try:
        if isinstance(obj, dict):
            # OPTIMIZATION: Skip large dicts with only primitive values.
            if len(obj) >= _LARGE_LIST_THRESHOLD and is_primitive_container(obj):
                return
            for k, v in obj.items():
                key_repr = repr(k) if not isinstance(k, str) else f"'{k}'"
                _collect_reachable_ids_with_paths(v, f"{path}[{key_repr}]", visited, id_to_path)
        elif isinstance(obj, (list, tuple)):
            # OPTIMIZATION: Skip large lists/tuples known to contain only primitives.
            if len(obj) >= _LARGE_LIST_THRESHOLD and is_primitive_container(obj):
                return
            for i, item in enumerate(obj):
                _collect_reachable_ids_with_paths(item, f"{path}[{i}]", visited, id_to_path)
        elif isinstance(obj, (set, frozenset)):
            # OPTIMIZATION: Skip large sets with only primitive elements.
            if isinstance(obj, set) and len(obj) >= _LARGE_LIST_THRESHOLD and is_primitive_container(obj):
                return
            for item in obj:
                _collect_reachable_ids_with_paths(item, f"{path}<set>", visited, id_to_path)
        elif isinstance(obj, pd.DataFrame):
            # Track internal block manager (persistent object)
            if hasattr(obj, '_mgr'):
                mgr_id = id(obj._mgr)
                visited.add(mgr_id)
                if mgr_id not in id_to_path:
                    id_to_path[mgr_id] = f"{path}._mgr"
            # NOTE: We do NOT track df[col].values because .values creates a
            # TEMPORARY array whose id() can be reused by the memory allocator.
            # We only recurse into object-dtype columns to find nested references.
            # Optimization: Check dtype directly, iterate Series (avoids .values array creation)
            # Use iloc to avoid issues with MultiIndex columns
            for col_idx in range(len(obj.columns)):
                try:
                    col = obj.columns[col_idx]
                    series = obj.iloc[:, col_idx]
                    if series.dtype == object:
                        # FAST PATH: Skip object columns with immutable scalars (strings, ints, etc.)
                        # These can't have nested mutable references
                        kind = infer_dtype(series, skipna=True)
                        if kind in _IMMUTABLE_INFERRED_KINDS:
                            continue  # All scalars are immutable, no nested refs possible
                        col_repr = repr(col) if not isinstance(col, str) else f"'{col}'"
                        for i, item in enumerate(series):
                            _collect_reachable_ids_with_paths(
                                item, f"{path}[{col_repr}][{i}]", visited, id_to_path
                            )
                except Exception:
                    pass
        elif isinstance(obj, pd.Series):
            # NOTE: We do NOT track series.values because .values creates a
            # TEMPORARY array whose id() can be reused by the memory allocator.
            # We only recurse into object-dtype series to find nested references.
            # Optimization: Check dtype directly, iterate Series (avoids .values array creation)
            try:
                if obj.dtype == object:
                    # FAST PATH: Skip object Series with immutable scalars
                    kind = infer_dtype(obj, skipna=True)
                    if kind not in _IMMUTABLE_INFERRED_KINDS:
                        for i, item in enumerate(obj):
                            _collect_reachable_ids_with_paths(item, f"{path}[{i}]", visited, id_to_path)
            except Exception:
                pass
        elif isinstance(obj, np.ndarray):
            # Track base array if this is a view (for proper view aliasing)
            # The base array is a persistent object, not a temporary.
            if obj.base is not None:
                base_id = id(obj.base)
                visited.add(base_id)
                if base_id not in id_to_path:
                    id_to_path[base_id] = f"{path}.base"
                _collect_reachable_ids_with_paths(obj.base, f"{path}.base", visited, id_to_path)
            if obj.dtype == object:
                try:
                    for i, item in enumerate(obj.flat):
                        _collect_reachable_ids_with_paths(item, f"{path}[{i}]", visited, id_to_path)
                except Exception:
                    pass
        elif hasattr(obj, '__dict__'):
            for attr, v in obj.__dict__.items():
                _collect_reachable_ids_with_paths(v, f"{path}.{attr}", visited, id_to_path)
        elif hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    _collect_reachable_ids_with_paths(
                        getattr(obj, slot), f"{path}.{slot}", visited, id_to_path
                    )
    except Exception:
        pass


def _collect_reachable_ids(obj: Any, visited: Set[int]) -> None:
    """
    Recursively collect all object IDs reachable from obj.

    This traverses the object graph and adds the ID of every mutable object
    to the visited set. Used to build the alias detection index.

    Args:
        obj: The object to traverse
        visited: Set to add object IDs to (modified in place)
    """
    obj_id = id(obj)
    if obj_id in visited:
        return  # Already visited (handles circular refs)

    # Skip immutable atomics - can't be mutated, no aliasing concern
    if isinstance(obj, _IMMUTABLE_ATOMIC_TYPES):
        return

    # Skip numpy scalar types
    if isinstance(obj, (np.integer, np.floating, np.complexfloating, np.bool_)):
        return

    # Skip singleton types (classes, functions, modules, etc.)
    # These are shared objects - finding them in multiple variables doesn't
    # indicate meaningful aliasing. See _SINGLETON_TYPES comment for details.
    if isinstance(obj, _SINGLETON_TYPES):
        return

    # Handle opaque objects (e.g., Keras/PyTorch models) - treat internal structure as
    # atomic, but DO traverse custom attributes for alias detection
    handler = OpaqueRegistry.get_handler(obj)
    if handler is not None:
        visited.add(obj_id)

        # Traverse custom attributes that we capture in get_mutable_state()
        # These may contain aliased references to other notebook variables
        try:
            custom_attrs = handler.get_traversable_attrs(obj)
            for attr_value in custom_attrs.values():
                _collect_reachable_ids(attr_value, visited)
        except Exception:
            pass  # Best-effort traversal

        return  # Don't recurse into internal opaque object structure

    visited.add(obj_id)

    # Recurse into containers
    try:
        if isinstance(obj, dict):
            # OPTIMIZATION: Skip large dicts with only primitive values.
            if len(obj) >= _LARGE_LIST_THRESHOLD and is_primitive_container(obj):
                return
            for v in obj.values():
                _collect_reachable_ids(v, visited)
        elif isinstance(obj, (list, tuple)):
            # OPTIMIZATION: Skip large lists/tuples known to contain only primitives.
            if len(obj) >= _LARGE_LIST_THRESHOLD and is_primitive_container(obj):
                return
            for item in obj:
                _collect_reachable_ids(item, visited)
        elif isinstance(obj, (set, frozenset)):
            # OPTIMIZATION: Skip large sets with only primitive elements.
            if isinstance(obj, set) and len(obj) >= _LARGE_LIST_THRESHOLD and is_primitive_container(obj):
                return
            for item in obj:
                _collect_reachable_ids(item, visited)
        elif isinstance(obj, pd.DataFrame):
            # Track internal block manager (persistent object)
            if hasattr(obj, '_mgr'):
                visited.add(id(obj._mgr))
            # NOTE: We do NOT track df[col].values because .values creates a
            # TEMPORARY array whose id() can be reused by the memory allocator.
            # We only recurse into object-dtype columns to find nested references.
            # Optimization: Check dtype directly, iterate Series (avoids .values array creation)
            # Use iloc to avoid issues with MultiIndex columns
            for col_idx in range(len(obj.columns)):
                try:
                    series = obj.iloc[:, col_idx]
                    if series.dtype == object:
                        # FAST PATH: Skip object columns with immutable scalars (strings, ints, etc.)
                        # These can't have nested mutable references
                        kind = infer_dtype(series, skipna=True)
                        if kind in _IMMUTABLE_INFERRED_KINDS:
                            continue  # All scalars are immutable, no nested refs possible
                        for item in series:
                            _collect_reachable_ids(item, visited)
                except Exception:
                    pass
        elif isinstance(obj, pd.Series):
            # NOTE: We do NOT track series.values because .values creates a
            # TEMPORARY array whose id() can be reused by the memory allocator.
            # We only recurse into object-dtype series to find nested references.
            # Optimization: Check dtype directly, iterate Series (avoids .values array creation)
            try:
                if obj.dtype == object:
                    # FAST PATH: Skip object Series with immutable scalars
                    kind = infer_dtype(obj, skipna=True)
                    if kind not in _IMMUTABLE_INFERRED_KINDS:
                        for item in obj:
                            _collect_reachable_ids(item, visited)
            except Exception:
                pass
        elif isinstance(obj, np.ndarray):
            # Track base array if this is a view (persistent object)
            if obj.base is not None:
                visited.add(id(obj.base))
                _collect_reachable_ids(obj.base, visited)
            # For object-dtype arrays, recurse into elements
            if obj.dtype == object:
                try:
                    for item in obj.flat:
                        _collect_reachable_ids(item, visited)
                except Exception:
                    pass
        elif hasattr(obj, '__dict__'):
            # User-defined objects
            for v in obj.__dict__.values():
                _collect_reachable_ids(v, visited)
        elif hasattr(obj, '__slots__'):
            # Slots-based objects
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    _collect_reachable_ids(getattr(obj, slot), visited)
    except Exception:
        # If traversal fails, we've at least captured the top-level object
        pass


class Checkpoint:
    """
    A snapshot of the kernel's user namespace at a point in time.

    Checkpoints store deep copies of variables along with metadata for
    tracking object identity across copies (via reverse_memo) and deep
    alias detection (via precomputed alias index).

    Attributes:
        name: Identifier for this checkpoint
        user_ns: Deep-copied user namespace variables
        reverse_memo: Maps copied object IDs back to original memo keys

    Deep Alias Detection Attributes (built lazily on first query, see section 12):
        _reachable_ids: Dict[var_name, Set[obj_id]] - all object IDs reachable
            from each variable via traversing nested containers. Used to find
            which variables share internal references.
        _id_to_vars: Dict[obj_id, Set[var_name]] - reverse index mapping each
            object ID to all variable names that contain it. Enables efficient
            lookup of aliases for a given set of accessed variables.
        _id_to_paths: Dict[obj_id, Dict[var_name, path_str]] - maps each object
            ID to the paths within each variable that reach it (e.g., "a['b'][0]").
            Used for detailed logging of alias relationships (only when FLOWBOOK_LOG_DEEP_ALIASES=1).
        _alias_index_built: bool - True if the alias index has been built.

    Note:
        The alias index only tracks PERSISTENT objects - not temporary objects
        created by property accessors like .values or .data which can have their
        id() reused by Python's memory allocator after garbage collection.
    """

    def __init__(
        self,
        name: str,
        user_ns: dict[str, Any],
        memo: dict[int, Any],
        cudf_origins: Optional['cudf_compat.CuDFOriginTracker'] = None,
    ):
        """
        Create a new checkpoint.

        Args:
            name: Identifier for this checkpoint
            user_ns: Deep-copied user namespace variables
            memo: Dictionary mapping original object IDs to their copies
            cudf_origins: Optional tracker for cudf object origins (for restore)
        """
        self.name = name
        self.user_ns = user_ns
        self.reverse_memo = {id(v): k for k, v in memo.items()}

        # cuDF origin tracking (for restore)
        from . import cudf_compat
        self._cudf_origins = cudf_origins or cudf_compat.CuDFOriginTracker()

        # Deep alias detection index (built lazily on first query)
        self._reachable_ids: Dict[str, Set[int]] = {}
        self._id_to_vars: Dict[int, Set[str]] = {}
        # Path tracking for alias logging: maps (var_name, obj_id) -> path string
        self._id_to_paths: Dict[int, Dict[str, str]] = {}
        self._alias_index_built: bool = False

    def _build_alias_index(self) -> None:
        """
        Build the deep alias detection index.

        This computes:
        - _reachable_ids: Maps each variable name to set of all object IDs reachable from it
        - _id_to_vars: Reverse index - maps object ID to all variable names containing it
        - _id_to_paths: Maps object ID to dict of {var_name: path_string} for logging

        These indexes are used by _expand_with_deep_aliases() in sdc_enforcer.py to
        efficiently find all variables that share internal references with a given
        set of accessed variables.

        Called once during checkpoint creation. Since checkpoints are immutable,
        the index never needs to be recomputed.
        """
        from collections import defaultdict

        with timer(key="alias:build_index", message=f"Building alias index for checkpoint '{self.name}'"):
            self._reachable_ids = {}
            self._id_to_vars = defaultdict(set)
            self._id_to_paths = defaultdict(dict)

            total_ids_collected = 0

            # Phase 1: Collect all reachable IDs for each variable
            # Optimization: Only track paths when logging is enabled (saves ~30-50% time)
            track_paths = _LOG_DEEP_ALIASES
            slow_vars = []

            with timer(key="alias:collect", message="Collecting reachable IDs"):
                for var_name, var_value in self.user_ns.items():
                    visited: Set[int] = set()

                    t0 = time.perf_counter()
                    if track_paths:
                        # Full path tracking for detailed alias logging
                        id_to_path: Dict[int, str] = {}
                        _collect_reachable_ids_with_paths(var_value, var_name, visited, id_to_path)
                        # Store paths
                        for obj_id, path in id_to_path.items():
                            self._id_to_paths[obj_id][var_name] = path
                    else:
                        # Fast path: no path tracking
                        _collect_reachable_ids(var_value, visited)
                    var_collect_time = time.perf_counter() - t0

                    self._reachable_ids[var_name] = visited
                    total_ids_collected += len(visited)

                    # Track slow variables for warning
                    if len(visited) > _SLOW_OBJECT_DTYPE_THRESHOLD or var_collect_time > 0.005:
                        slow_vars.append((var_name, len(visited), var_collect_time * 1000))

            # Log warnings for slow variables
            for var_name, num_ids, time_ms in slow_vars:
                # Identify the cause
                var_value = self.user_ns[var_name]
                cause = ""
                if isinstance(var_value, pd.DataFrame):
                    # Use iloc to avoid issues with MultiIndex columns
                    obj_cols = [var_value.columns[i] for i in range(len(var_value.columns))
                                if var_value.iloc[:, i].dtype == object]
                    if obj_cols:
                        total_obj_elements = sum(len(var_value) for _ in obj_cols)
                        cause = f" (DataFrame with {len(obj_cols)} object-dtype columns, {total_obj_elements} elements)"
                elif isinstance(var_value, pd.Series) and var_value.dtype == object:
                    cause = f" (Series with {len(var_value)} object-dtype elements)"
                elif isinstance(var_value, np.ndarray) and var_value.dtype == object:
                    cause = f" (ndarray with {var_value.size} object-dtype elements)"
                log(f"[alias-index] WARNING: slow var '{var_name}' (type={type(var_value).__name__}): {num_ids} IDs in {time_ms:.1f}ms{cause}")

            # Phase 2: Build reverse index in one pass
            # This is O(total_ids) but we batch the set.add calls per variable
            with timer(key="alias:reverse", message="Building reverse index"):
                for var_name, visited in self._reachable_ids.items():
                    for obj_id in visited:
                        self._id_to_vars[obj_id].add(var_name)

            # Convert defaultdicts to regular dicts
            self._id_to_vars = dict(self._id_to_vars)
            self._id_to_paths = dict(self._id_to_paths)

            # Log summary
            num_vars = len(self.user_ns)
            num_unique_ids = len(self._id_to_vars)
            log(f"[alias-index] {num_vars} vars, {total_ids_collected} total IDs, {num_unique_ids} unique IDs")

            self._alias_index_built = True

    def get_aliases_for_vars(
        self, accessed_vars: Set[str], log_aliases: bool = False
    ) -> Set[str]:
        """
        Get all variables that share internal references with the given variables.

        This uses the precomputed alias index to efficiently find aliases.
        The index is built lazily on first call.
        O(accessed_vars + number_of_aliases) instead of O(total_objects_in_namespace).

        Args:
            accessed_vars: Set of variable names to find aliases for
            log_aliases: If True, log discovered alias relationships

        Returns:
            Set of all variable names that share any internal reference with
            the accessed variables (including the original accessed_vars)
        """
        # Lazy index building: only build when first needed
        if not self._alias_index_built:
            self._build_alias_index()

        if not self._reachable_ids:
            # Empty namespace - just return accessed_vars
            return accessed_vars

        # Step 1: Collect all reachable IDs from accessed vars
        all_reachable_ids: Set[int] = set()
        accessed_ids_by_var: Dict[str, Set[int]] = {}
        for var_name in accessed_vars:
            if var_name in self._reachable_ids:
                var_ids = self._reachable_ids[var_name]
                all_reachable_ids |= var_ids
                accessed_ids_by_var[var_name] = var_ids

        # Step 2: Find all vars containing any of these IDs
        all_relevant_vars: Set[str] = set()
        # Track which IDs caused each alias for logging
        alias_reasons: Dict[str, List[tuple]] = {}  # alias_var -> [(accessed_var, path1, path2)]

        for obj_id in all_reachable_ids:
            vars_with_id = self._id_to_vars.get(obj_id, set())
            for alias_var in vars_with_id:
                if alias_var not in accessed_vars and alias_var not in all_relevant_vars:
                    # This is a newly discovered alias
                    if log_aliases or _LOG_DEEP_ALIASES:
                        # Find which accessed var shares this ID
                        for accessed_var, acc_ids in accessed_ids_by_var.items():
                            if obj_id in acc_ids:
                                # Get paths for both variables
                                paths = self._id_to_paths.get(obj_id, {})
                                acc_path = paths.get(accessed_var, accessed_var)
                                alias_path = paths.get(alias_var, alias_var)
                                if alias_var not in alias_reasons:
                                    alias_reasons[alias_var] = []
                                alias_reasons[alias_var].append((accessed_var, acc_path, alias_path))
                                break
                all_relevant_vars.add(alias_var)

        # Log discovered aliases
        if (log_aliases or _LOG_DEEP_ALIASES) and alias_reasons:
            for alias_var, reasons in sorted(alias_reasons.items()):
                for accessed_var, acc_path, alias_path in reasons:
                    log(f"[deep-alias] {acc_path} ↔ {alias_path} (share internal ref)")

        # Step 3: Include original accessed vars (for new variable case)
        all_relevant_vars |= accessed_vars

        return all_relevant_vars

    def get_original_id(self, obj_id: int) -> int:
        """
        Map a copied object's ID back to its original memo key.

        Args:
            obj_id: ID of a copied object

        Returns:
            Original memo key, or obj_id if not found in memo
        """
        return self.reverse_memo.get(obj_id, obj_id)

    @staticmethod
    def diff(
        a: Checkpoint, b: Checkpoint, keys_to_include: set[str] | None = None,
        use_leq: bool = False,
        column_rbw: Optional[Dict[str, Set[str]]] = None,
        structural_reads: Optional[Dict[str, Set[str]]] = None,
        structural_mode: Optional["StructuralTrackingMode"] = None,
    ):
        """
        Compare two checkpoints and return structured diff results.

        Args:
            a: First checkpoint to compare
            b: Second checkpoint to compare
            keys_to_include: Optional set of keys to limit comparison to
            use_leq: If True, use leq mode where extra keys in b are allowed
                     and DataFrames in b can have extra columns
            column_rbw: Optional column-level reads-before-writes mapping.
                       Maps variable path to set of column names that were RBW.
                       When provided with use_leq=True, only these columns are
                       compared for each DataFrame.
            structural_reads: Optional structural attribute reads mapping.
                       Maps variable path to set of structural attributes read
                       (e.g., 'columns', 'shape', 'len'). Used with structural_mode.
            structural_mode: How to handle structural reads (OFF, WARN, ENFORCE).
                       If None, defaults to OFF.

        Returns:
            DiffResult: Structured diff tree with only differences
        """
        with timer(key="checkpoint_diff:setup", message="[diff] Setup"):
            from .structural_tracking import StructuralTrackingMode
            if structural_mode is None:
                structural_mode = StructuralTrackingMode.OFF

        with timer(key="checkpoint_diff:create", message="[diff] Create Diff object"):
            differ = Diff(
                strict=False,
                report_close=False,
                atol=1e-5,
                rtol=1e-5,
                use_leq=use_leq,
                column_rbw=column_rbw,
                structural_reads=structural_reads or {},
                structural_mode=structural_mode,
            )

        with timer(key="checkpoint_diff:compare", message="[diff] Compare namespaces"):
            result = differ.diff(a.user_ns, b.user_ns, keys_to_include)

        return result


class Checkpoints:
    """
    Manager for multiple named checkpoints of kernel state.

    Provides save, restore, and comparison operations for kernel namespace
    snapshots. Handles deep copying with special optimizations for pandas
    objects via custom deepcopy module.

    Attributes:
        sanity_check: If True, verify copies match originals after save
        saved: Dictionary mapping checkpoint names to Checkpoint objects
    """

    def __init__(
        self,
        sanity_check: bool = False,
        warn_classes: bool = True,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            sanity_check: If True, verify copies match originals after save
            warn_classes: If True, warn when user-defined classes are checkpointed, since
                class variables won't be properly restored. Default True.
        """
        self.sanity_check = sanity_check
        self.warn_classes = warn_classes
        self.saved: dict[str, Checkpoint] = {}

        # Ensure copy-on-write is enabled for performance
        if not pd.options.mode.copy_on_write:
            log("WARNING: pandas copy_on_write was disabled - re-enabling for checkpoint performance")
            pd.options.mode.copy_on_write = True

    def _estimate_size(self, variables: dict[str, Any]) -> int:
        """
        Estimate the memory size of variables in bytes.

        This is a rough estimate using sys.getsizeof() which may underestimate
        for complex nested structures. It's meant for warning purposes only.

        Args:
            variables: Dictionary of variables to estimate

        Returns:
            Estimated size in bytes
        """
        import sys

        total_size = 0
        for k, v in variables.items():
            try:
                # Get size of the object
                size = sys.getsizeof(v)

                # For pandas DataFrames/Series, use memory_usage()
                if isinstance(v, pd.DataFrame):
                    size = v.memory_usage(deep=True).sum()
                elif isinstance(v, pd.Series):
                    size = v.memory_usage(deep=True)
                elif isinstance(v, np.ndarray):
                    size = v.nbytes

                total_size += size
            except Exception:
                # If we can't estimate, skip this variable
                pass

        return total_size

    def _deep_copy_user_ns(
        self, variables: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[int, Any], dict[str, Exception]]:
        """
        Deep copy a dictionary of variables using custom deepcopy module.

        Uses flowbook.kernel.deepcopy which has special handling for:
        - pandas DataFrames: shallow copy + deep copy object columns
        - pandas Series: shallow copy + deep copy if object dtype
        - Functions: deep copy closure and mutable defaults

        Args:
            variables: Dictionary of variables to copy

        Returns:
            Tuple of (copied dictionary, memo dictionary for tracking copied objects,
                     dictionary of failed variables with their exceptions)
        """
        copied = {}
        memo = {}
        failed = {}

        loop_start = time.time()
        for k, v in variables.items():
            try:
                start_time = time.time()
                # Use custom deepcopy which handles pandas and functions specially
                copied[k] = deepcopy(v, memo)

                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds

                if _PROFILE_CHECKPOINT:
                    # Record timing keyed by type name (in milliseconds)
                    type_module = type(v).__module__
                    type_name = type(v).__name__
                    output.add_timing(f"deepcopy:{type_module}.{type_name}", duration_ms)

                if duration_ms > 10 or _PROFILE_CHECKPOINT:  # 10ms threshold
                    log(f"Deep copying variable {k} took {duration_ms:.1f} ms")
            except Exception as e:
                error_msg = f"Failed to deep copy variable {k}: {type(e).__name__}: {e}"

                # Add helpful hints based on the type
                if isinstance(v, types.GeneratorType):
                    error_msg += "\n  Hint: Generators cannot be checkpointed (they maintain execution state)"
                elif hasattr(type(v), '__module__') and type(v).__module__.startswith('matplotlib'):
                    error_msg += "\n  Hint: Matplotlib objects are excluded from checkpoints"
                elif isinstance(v, types.ModuleType):
                    error_msg += "\n  Hint: Modules cannot be checkpointed"
                elif hasattr(v, '__iter__') and not isinstance(v, (str, bytes, list, tuple, dict, set, frozenset)):
                    error_msg += "\n  Hint: Iterator objects may not be checkpointable"
                elif 'thread' in str(type(v)).lower() or 'lock' in str(type(v)).lower():
                    error_msg += "\n  Hint: Thread/lock objects cannot be pickled"

                log(error_msg)
                # Track variables that failed to copy
                failed[k] = e

        if _PROFILE_CHECKPOINT:
            loop_duration_ms = (time.time() - loop_start) * 1000
            output.add_timing("deepcopy:loop_total", loop_duration_ms)

        return copied, memo, failed

    def _is_user_defined_class(self, v: Any) -> bool:
        """
        Check if a value is a user-defined class (not an instance).

        Args:
            v: Value to check

        Returns:
            True if v is a user-defined class, False otherwise
        """
        # Check if it's a type/class
        if not isinstance(v, type):
            return False

        # Exclude built-in types
        if v.__module__ in ('builtins', '__builtin__'):
            return False

        # Exclude common library classes
        if v.__module__.startswith(('pandas', 'numpy', 'matplotlib', 'sklearn')):
            return False

        # It's a user-defined class
        return True

    def checkpointable_value(self, v: Any) -> bool:
        """
        Check if a value can be included in a checkpoint.

        Filters out modules and matplotlib objects which cannot be
        reliably deep copied.

        Args:
            v: Value to check

        Returns:
            True if the value can be checkpointed, False otherwise
        """
        # Skip modules
        if isinstance(v, types.ModuleType):
            return False

        # Skip matplotlib objects (safe access to __module__)
        module = getattr(type(v), "__module__", "")
        if module.startswith("matplotlib"):
            return False

        # Skip numpy arrays containing matplotlib objects
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                try:
                    for item in v.flat:
                        item_module = getattr(type(item), "__module__", "")
                        if item_module.startswith("matplotlib"):
                            return False
                except (AttributeError, TypeError):
                    # If we can't iterate, be conservative and skip
                    return False

        return True

    def checkpointable_vars(self, user_ns: dict[str, Any]) -> dict[str, Any]:
        """
        Filter namespace to variables with valid names.

        Args:
            user_ns: User namespace dictionary

        Returns:
            Filtered dictionary excluding private/system variables
        """
        return filter_user_namespace(user_ns)

    def checkpointable_values(self, user_ns: dict[str, Any]) -> dict[str, Any]:
        """
        Filter namespace to values that can be checkpointed.

        Args:
            user_ns: User namespace dictionary

        Returns:
            Filtered dictionary excluding non-checkpointable values
        """
        return {k: v for k, v in user_ns.items() if self.checkpointable_value(v)}

    def save(
        self, name: str, user_ns: dict[str, Any], max_size_mb: int | None = 1000
    ) -> tuple[dict[str, TypeModel], dict[str, TypeModel]]:
        """
        Save a checkpoint of the current namespace.

        Note: Object dtype columns in DataFrames/Series are automatically converted
        to specialized dtypes (Int64, string, datetime64, etc.) during the deepcopy
        operation. This happens for all DataFrames, including those nested in
        other data structures.

        Args:
            name: Identifier for this checkpoint (overwrites if exists)
            user_ns: User namespace dictionary to checkpoint
            max_size_mb: Warn if estimated checkpoint size exceeds this many MB.
                Set to None to disable size warnings. Default: 1000 MB.

        Returns:
            Tuple of (saved variables with type models, removed variables with type models).
            Removed includes variables that couldn't be checkpointed or failed to copy.

        Raises:
            ValueError: If checkpoint name is empty or whitespace-only
        """
        # Validate checkpoint name
        if not name or not name.strip():
            raise ValueError("Checkpoint name cannot be empty or whitespace-only")

        # Estimate size and warn if needed
        if max_size_mb is not None:
            checkpointable_vars = self.checkpointable_vars(user_ns)
            checkpointable_values = self.checkpointable_values(checkpointable_vars)

            estimated_bytes = self._estimate_size(checkpointable_values)
            estimated_mb = estimated_bytes / (1024 * 1024)

            if estimated_mb > max_size_mb:
                log(f"WARNING: Checkpoint '{name}' estimated at {estimated_mb:.1f} MB (threshold: {max_size_mb} MB)")
                log(f"         Large checkpoints may consume significant memory and time")

        # Warn about user-defined classes if enabled
        if self.warn_classes:
            checkpointable_vars_temp = self.checkpointable_vars(user_ns)
            checkpointable_values_temp = self.checkpointable_values(checkpointable_vars_temp)

            for var_name, var_value in checkpointable_values_temp.items():
                if self._is_user_defined_class(var_value):
                    log(f"WARNING: Variable '{var_name}' is a user-defined class ({var_value.__name__})")
                    log(f"         Class variables (mutable class attributes) will NOT be properly restored")
                    log(f"         Only instance attributes will be checkpointed. See documentation section 8.4")

        with timer(key="checkpoint:deep_copy", message="Deep copying user namespace"):
            saved = {}
            removed = {}

            with timer(key="checkpoint:filter_vars", message="Filtering variables"):
                checkpointable_vars = self.checkpointable_vars(user_ns)
                checkpointable_values = self.checkpointable_values(checkpointable_vars)

            for k in checkpointable_vars.keys() - checkpointable_values.keys():
                removed[k] = get_type_model(user_ns[k])

            # Record cudf origins before deep copy (cudf objects become pandas)
            with timer(key="checkpoint:cudf_origins", message="Recording cudf origins"):
                from . import cudf_compat
                cudf_origins = cudf_compat.CuDFOriginTracker()
                for k, v in checkpointable_values.items():
                    cudf_origins.record(k, v)

            # Use helper to deep copy all variables
            with timer(key="checkpoint:deepcopy", message="Deep copying variables"):
                cp, memo, failed = self._deep_copy_user_ns(checkpointable_values)

            # Track successfully copied variables
            with timer(key="checkpoint:type_models", message="Generating type models"):
                for k in cp:
                    saved[k] = get_type_model(checkpointable_values[k])

                # Track variables that failed to copy
                for k in failed:
                    removed[k] = get_type_model(checkpointable_values[k])

            self.saved[name] = Checkpoint(name, cp, memo, cudf_origins)

        if self.sanity_check:
            with timer(key="checkpoint:sanity_check", message="Running sanity check"):
                original = {k: v for k, v in checkpointable_values.items() if k in saved}
                differ = Diff(strict=False, report_close=False, atol=1e-5, rtol=1e-5)
                diff_result = differ.diff(original, cp)
                if diff_result.differences:
                    raise ValueError(f"Sanity check failed: {diff_result.differences}")

        return saved, removed

    def restore(self, name: str, user_ns: dict[str, Any]) -> None:
        """
        Restore a checkpoint to the namespace.

        Clears all checkpointable variables from the namespace and replaces
        them with deep copies from the checkpoint.

        Args:
            name: Name of checkpoint to restore
            user_ns: User namespace dictionary to restore into

        Raises:
            KeyError: If checkpoint name doesn't exist
        """
        cp = self.saved[name]
        checkpointable_vars = self.checkpointable_vars(user_ns)

        for k in checkpointable_vars.keys():
            del user_ns[k]

        # Deep copy the checkpoint before restoring to keep the checkpoint pristine
        # This ensures that modifications to restored variables don't affect the checkpoint
        restored_vars, _, _ = self._deep_copy_user_ns(cp.user_ns)

        # Convert pandas objects back to cudf if they originated from cudf
        for k in list(restored_vars.keys()):
            restored_vars[k] = cp._cudf_origins.restore_value(k, restored_vars[k])

        user_ns.update(restored_vars)

    def type_models(self, user_ns: dict[str, Any]) -> dict[str, TypeModel]:
        """
        Get type models for all checkpointable variables in namespace.

        Args:
            user_ns: User namespace dictionary

        Returns:
            Dictionary mapping variable names to their TypeModel representations
        """
        return {
            k: get_type_model(v) for k, v in self.checkpointable_vars(user_ns).items()
        }

    def delete(self, name: str) -> None:
        """
        Delete a checkpoint by name.

        Also clears the large list cache when the last checkpoint is deleted,
        since cached list copies may reference data from deleted checkpoints.

        Args:
            name: Name of checkpoint to delete (no-op if doesn't exist)
        """
        if name in self.saved:
            del self.saved[name]
            # Clear list cache when last checkpoint is deleted to free memory
            # and avoid stale references
            if not self.saved:
                from flowbook.kernel.deepcopy import clear_list_cache
                clear_list_cache()

    def list(self) -> list[str]:
        """
        List all checkpoint names.

        Returns:
            List of checkpoint names in insertion order
        """
        return list(self.saved.keys())

    def clear(self) -> None:
        """
        Delete all checkpoints.

        Also clears the large list cache to free memory and avoid stale
        references to data from deleted checkpoints.
        """
        self.saved.clear()
        # Clear list cache to free memory held by cached list copies
        from flowbook.kernel.deepcopy import clear_list_cache
        clear_list_cache()

    def get(self, name: str) -> Checkpoint:
        """
        Get a checkpoint by name.

        Args:
            name: Name of checkpoint to retrieve

        Returns:
            The Checkpoint object

        Raises:
            KeyError: If checkpoint name doesn't exist
        """
        return self.saved[name]

    def exists(self, name: str) -> bool:
        """
        Check if a checkpoint with the given name exists.

        Args:
            name: Name of checkpoint to check

        Returns:
            True if checkpoint exists, False otherwise
        """
        return name in self.saved
