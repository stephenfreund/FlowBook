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

❌ CLASS VARIABLES ARE NOT RESTORED (Section 8.5)
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

⚠️  IN-PLACE DTYPE CONVERSION (Section 8.3)
   - By default, object dtype columns are converted to specialized types
   - This modifies your ORIGINAL DataFrames during checkpointing
   - object → Int64, string, datetime64, etc.
   - Set convert_dtypes=False to preserve object dtypes
   - Example:
       df = pd.DataFrame({'data': pd.Series([1, 2, 3], dtype=object)})
       cp.save('test', user_ns)  # df['data'].dtype is now Int64!

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
  convert_*_to_specialized() - Converts object dtypes for efficiency
  deepcopy module     - Custom deepcopy with pandas/function support

Data Flow:
  user_ns → filter → convert dtypes → deep copy → Checkpoint
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

Solution: Custom deepcopy module (data_ferret.kernel.deepcopy) that:
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


4.3 Object Dtype Conversion
---------------------------
Problem: Object dtype columns are slow to copy and compare because each
element must be handled individually.

Solution: Before checkpointing, convert object columns to specialized
dtypes when possible using infer_dtype():
  - Integer/mixed-integer → Int64 (nullable)
  - Float/mixed-float → float64
  - String → StringDtype
  - Boolean → boolean (nullable)
  - Datetime → datetime64[ns]
  - Etc.

This conversion:
  - Makes subsequent copies faster (contiguous memory)
  - Makes comparisons faster (vectorized operations)
  - Reduces memory for common cases
  - Is done in-place on user_ns (intentional side effect)


4.4 Restore Creates New Copies
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

See data_ferret/kernel/deepcopy.py for full implementation.


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


8.3 In-Place Dtype Conversion
-----------------------------
convert_object_to_specialized() modifies user_ns IN PLACE.
This is intentional (avoids another copy) but may surprise users:
  - DataFrame columns may have different dtypes after checkpoint
  - This is generally an improvement (proper types vs. object)
  - But could break code relying on exact object dtype


8.4 Function/Lambda Checkpointing
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


8.5 Class Definitions Not Properly Checkpointed
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


10. USAGE EXAMPLES
------------------

Example 1: Basic Save/Restore
    from data_ferret.kernel.checkpoint import Checkpoints

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
    # Disable dtype conversion to preserve object dtypes
    cp = Checkpoints(convert_dtypes=False)

    # Disable size warnings for large checkpoints
    cp.save("big_data", user_ns, max_size_mb=None)

    # Enable sanity checking (expensive, for debugging)
    cp = Checkpoints(sanity_check=True)

    # Disable class warnings if you know what you're doing
    cp = Checkpoints(warn_classes=False)

Example 6: Comparing Checkpoints
    from data_ferret.kernel.checkpoint import Checkpoint

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

    # For large DataFrames, checkpointing object columns is expensive
    # Consider converting to specialized dtypes first
    df['int_col'] = df['int_col'].astype('Int64')  # Not object
    df['str_col'] = df['str_col'].astype('string')  # Not object

    # Or disable conversion and keep object dtype
    cp = Checkpoints(convert_dtypes=False)

    # Monitor checkpoint time
    import time
    start = time.time()
    cp.save("big_checkpoint", user_ns)
    print(f"Checkpoint took {time.time() - start:.2f} seconds")


11. DEPENDENCIES
----------------
Internal:
  - data_ferret.kernel.deepcopy: Custom deepcopy implementation
  - data_ferret.kernel.diff.Diff: Structured diff between checkpoints
  - data_ferret.kernel.extended_types.TypeModel, get_type_model: Type introspection
  - data_ferret.util.output.log, timer: Logging and performance instrumentation

External:
  - datetime, decimal, time, types: Standard library utilities
  - numpy: Array handling and scalar types
  - pandas: DataFrame/Series handling and dtype inference


12. FUTURE WORK / TODOS
-----------------------
  - [ ] Incremental checkpointing (store only changes from previous)
  - [ ] Checkpoint compression for memory efficiency
  - [ ] Async checkpoint creation (background deep copy)
  - [ ] Checkpoint serialization (save to disk)
  - [ ] Configurable exclusion patterns (not just matplotlib)
  - [ ] Thread safety (at least via locking)
  - [ ] Size estimation before checkpoint (warn on large data)


13. TESTING NOTES
-----------------
Key test scenarios:

BASIC FUNCTIONALITY:
  1. Basic save/restore roundtrip
  2. Mutation isolation (change original, verify checkpoint unchanged)
  3. Repeated restore from same checkpoint
  4. Checkpoint deletion and memory cleanup
  5. Multiple checkpoints with different states
  6. Checkpoint overwrite
  7. Empty namespace save/restore
  8. Restore into empty namespace
  9. Sanity check mode verification

CIRCULAR REFERENCES & SHARED OBJECTS (7.1, 4.6):
  10. Self-referential list (list containing itself)
  11. Mutually referential lists (A contains B, B contains A)
  12. Self-referential dict
  13. Circular reference in DataFrame cells
  14. Same list shared by multiple variables (identity preservation)
  15. Shared dict in multiple DataFrame cells
  16. Reverse memo tracking for object identity

PANDAS DATA STRUCTURES:
  17. DataFrame with lists in cells
  18. DataFrame with dicts in cells
  19. DataFrame with nested mutable structures
  20. DataFrame with mixed dtypes (int, float, str, object)
  21. Multiple DataFrames with mutable objects
  22. Series with lists
  23. Series with dicts
  24. Series with non-object dtypes
  25. Empty DataFrame
  26. Empty Series
  27. DataFrame with None values
  28. Large DataFrames (100+ rows) with mutable objects
  29. MultiIndex DataFrames (row and column MultiIndex)
  30. DataFrame with custom index
  31. Series with custom index
  32. Nested DataFrames (DataFrame containing DataFrames)
  33. List of DataFrames
  34. Sparse DataFrames

OBJECT DTYPE CONVERSION (4.3, 7.2):
  35. Object column with truly mixed types (can't convert)
  36. Object column with integers (converts to Int64)
  37. Object column with strings (converts to string dtype)
  38. Object column with floats (converts to float64)
  39. Object column with booleans (converts to boolean)
  40. Object column with datetimes (converts to datetime64)

EXTENSION DTYPES:
  41. Categorical dtype
  42. Nullable integer dtype (Int64)
  43. String dtype (StringDtype)
  44. Boolean dtype (nullable boolean)

SPECIAL NUMERIC VALUES:
  45. NaN values in DataFrames
  46. Infinity values (inf, -inf)
  47. Complex numbers (in arrays and DataFrames)
  48. Decimal objects

FUNCTION CHECKPOINTING (8.4):
  49. Function with closure variables - verify isolation
  50. Function with mutable default list []
  51. Function with mutable default dict {}
  52. Lambda with captured variables
  53. Nested functions with shared closure
  54. Recursive function (factorial, etc.)
  55. Function without closure or defaults (optimization check)
  56. Bound method (method references object)
  57. Function with nested mutable closure (dict containing list)
  58. Function with mutable __dict__ attributes

CLASS DEFINITIONS (8.5 - KNOWN ISSUES):
  59. Class variable not restored (known issue)
  60. Instance attributes ARE restored correctly
  61. Class method modification persists (known issue)
  62. Class redefinition works correctly
  63. Instance with mutable attributes in __dict__

CUSTOM DEEPCOPY & PICKLE (7.4):
  64. Object with custom __deepcopy__ method
  65. Object with __getstate__ and __setstate__
  66. Object with __reduce__ or __reduce_ex__

GENERATORS & ITERATORS (7.7):
  67. Generator objects (may fail or exhaust)
  68. Iterator over list (may produce unexpected results)

COLLECTION TYPES:
  69. Sets (mutable)
  70. Frozensets (immutable)
  71. Named tuples
  72. collections.deque
  73. Tuples with mutable contents
  74. Regular lists with nested lists
  75. Regular dicts with nested dicts
  76. Numpy arrays (numeric and object dtype)

DATETIME TYPES:
  77. datetime.datetime objects
  78. datetime.timedelta objects
  79. datetime64 in DataFrames
  80. timedelta64 in DataFrames

FILTERING & ERROR HANDLING:
  81. Filters out modules
  82. Filters out system variables (_, __, get_ipython, etc.)
  83. Filters out private variables (_prefix)
  84. Filters out matplotlib objects
  85. Uncopyable object tracked as removed
  86. Restore removes variables not in checkpoint

EDGE CASES:
  87. Very deeply nested structures (dict in dict in dict...)
  88. None values
  89. Unicode strings
  90. Bytes objects
  91. Restore preserves private variables (doesn't delete them)
  92. Variable deletion between save and restore

INTEGRATION TESTS:
  93. Save/restore/delete cycle with multiple checkpoints
  94. Checkpoint diff after modification
  95. Type models generation
  96. Mixed types in single namespace

TEST FILES:
  - test_checkpoint.py: Basic functionality and common scenarios
  - test_checkpoint_comprehensive.py: All corner cases and edge cases from above


================================================================================
                            END DESIGN DOCUMENT
================================================================================
"""

from __future__ import annotations

import time
import types
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd

from data_ferret.kernel.deepcopy import deepcopy
from data_ferret.kernel.diff import Diff
from data_ferret.kernel.extended_types import TypeModel, get_type_model
from data_ferret.util.output import log, timer

# Enable copy-on-write mode for better performance with DataFrame copies
pd.options.mode.copy_on_write = True


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


class Checkpoint:
    """
    A snapshot of the kernel's user namespace at a point in time.

    Checkpoints store deep copies of variables along with metadata for
    tracking object identity across copies (via reverse_memo).

    Attributes:
        name: Identifier for this checkpoint
        user_ns: Deep-copied user namespace variables
        reverse_memo: Maps copied object IDs back to original memo keys
    """

    def __init__(self, name: str, user_ns: dict[str, Any], memo: dict[int, Any]):
        """
        Create a new checkpoint.

        Args:
            name: Identifier for this checkpoint
            user_ns: Deep-copied user namespace variables
            memo: Dictionary mapping original object IDs to their copies
        """
        self.name = name
        self.user_ns = user_ns
        self.reverse_memo = {id(v): k for k, v in memo.items()}

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
        from .structural_tracking import StructuralTrackingMode
        if structural_mode is None:
            structural_mode = StructuralTrackingMode.OFF
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
        return differ.diff(a.user_ns, b.user_ns, keys_to_include)


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
        convert_dtypes: bool = True,
        warn_classes: bool = True,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            sanity_check: If True, verify copies match originals after save
            convert_dtypes: If True, automatically convert object dtype columns to specialized
                dtypes (Int64, string, datetime64, etc.) during checkpointing. Default True.
                Set to False to preserve original object dtypes.
            warn_classes: If True, warn when user-defined classes are checkpointed, since
                class variables won't be properly restored. Default True.
        """
        self.sanity_check = sanity_check
        self.convert_dtypes = convert_dtypes
        self.warn_classes = warn_classes
        self.saved: dict[str, Checkpoint] = {}

        # Ensure copy-on-write is enabled for performance
        if not pd.options.mode.copy_on_write:
            log("WARNING: pandas copy_on_write was disabled - re-enabling for checkpoint performance")
            pd.options.mode.copy_on_write = True

        # Set global dtype conversion flag in deepcopy module
        import data_ferret.kernel.deepcopy as deepcopy_module
        deepcopy_module._convert_object_dtypes = convert_dtypes

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

        Uses data_ferret.kernel.deepcopy which has special handling for:
        - pandas DataFrames: shallow copy + deep copy object columns
        - pandas Series: shallow copy + deep copy if object dtype
        - Functions: deep copy closure and mutable defaults

        Args:
            variables: Dictionary of variables to copy

        Returns:
            Tuple of (copied dictionary, memo dictionary for tracking copied objects,
                     dictionary of failed variables with their exceptions)
        """
        # Temporarily set the dtype conversion flag for this operation
        import data_ferret.kernel.deepcopy as deepcopy_module
        old_convert_flag = deepcopy_module._convert_object_dtypes
        deepcopy_module._convert_object_dtypes = self.convert_dtypes

        try:
            copied = {}
            memo = {}
            failed = {}

            for k, v in variables.items():
                try:
                    start_time = time.time()
                    # Use custom deepcopy which handles pandas and functions specially
                    copied[k] = deepcopy(v, memo)

                    end_time = time.time()
                    duration = end_time - start_time
                    if duration > 0.010:
                        log(f"Deep copying variable {k} took {duration:.3f} seconds")
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

            return copied, memo, failed
        finally:
            # Restore the old flag
            deepcopy_module._convert_object_dtypes = old_convert_flag

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
                    log(f"         Only instance attributes will be checkpointed. See documentation section 8.5")

        with timer(key="deep_copy_user_ns", message="Deep copying user namespace"):
            saved = {}
            removed = {}
            checkpointable_vars = self.checkpointable_vars(user_ns)

            checkpointable_values = self.checkpointable_values(checkpointable_vars)
            for k in checkpointable_vars.keys() - checkpointable_values.keys():
                removed[k] = get_type_model(user_ns[k])

            # Use helper to deep copy all variables
            cp, memo, failed = self._deep_copy_user_ns(checkpointable_values)

            # Track successfully copied variables
            for k in cp:
                saved[k] = get_type_model(checkpointable_values[k])

            # Track variables that failed to copy
            for k in failed:
                removed[k] = get_type_model(checkpointable_values[k])

            self.saved[name] = Checkpoint(name, cp, memo)

        if self.sanity_check:
            with timer(key="sanity_check", message="Running sanity check"):
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

        Args:
            name: Name of checkpoint to delete (no-op if doesn't exist)
        """
        if name in self.saved:
            del self.saved[name]

    def list(self) -> list[str]:
        """
        List all checkpoint names.

        Returns:
            List of checkpoint names in insertion order
        """
        return list(self.saved.keys())

    def clear(self) -> None:
        """Delete all checkpoints."""
        self.saved.clear()

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
