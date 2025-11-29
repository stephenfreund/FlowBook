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


2. ARCHITECTURE
---------------
Key Components:

  Checkpoint          - Single snapshot containing deep-copied namespace
  Checkpoints         - Manager for multiple named checkpoints
  filter_user_namespace() - Filters namespace to user-defined variables
  is_immutable_type() - Optimizes copying by skipping immutable objects
  convert_*_to_specialized() - Converts object dtypes for efficiency

Data Flow:
  user_ns → filter → convert dtypes → deep copy → Checkpoint
  Checkpoint → deep copy → user_ns (on restore)

Dependency Graph:
  checkpoint.py
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

  c) Memory Efficiency: Use copy-on-write and skip copying immutable data
     where possible to minimize memory overhead.

  d) Transparency: Users shouldn't need to understand checkpointing internals.
     Variable behavior should be identical before/after checkpoint/restore.


4. DESIGN DECISIONS & RATIONALE
-------------------------------

4.1 Deep Copy Strategy
----------------------
Problem: Python's copy.deepcopy() doesn't handle pandas objects optimally.
  - DataFrames with copy(deep=True) still share underlying numpy arrays
  - Object dtype columns can contain mutable objects that need deep copying

Solution: Custom deep copy strategy per type:
  - Standard objects: copy.deepcopy() with shared memo
  - DataFrames: shallow copy + CoW + explicit deep copy of object columns
  - Series: shallow copy + CoW + explicit deep copy if object dtype
  - Immutable types: reference sharing (no copy needed)

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


4.3 Immutable Type Optimization
-------------------------------
Immutable objects don't need copying - they can be safely shared:
  - Python primitives: int, float, str, bool, bytes, complex, frozenset, range
  - NumPy scalars: np.int64, np.float32, etc. (all np.generic subclasses)
  - Date/time: datetime.date, datetime.time, datetime.timedelta
  - Pandas scalars: pd.Timestamp, pd.Timedelta, pd.Period
  - Decimal: decimal.Decimal

For pandas Series/DataFrame with object dtype, we check if ALL values are
immutable before skipping the deep copy. This is done via is_column_all_immutable().


4.4 Object Dtype Conversion
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


4.5 Restore Creates New Copies
------------------------------
When restoring a checkpoint, we deep copy from the checkpoint rather than
just updating references. This ensures:
  - Multiple restores from the same checkpoint work correctly
  - The stored checkpoint remains pristine
  - Mutations after restore don't affect the checkpoint


4.6 Namespace Filtering
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


4.7 Reverse Memo for Identity Tracking
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
          if isinstance(value, pd.DataFrame):
              # 1. Shallow copy (CoW handles non-object columns)
              # 2. Deep copy object columns only if they contain mutable objects
              # 3. Track in memo
          elif isinstance(value, pd.Series):
              # Similar to DataFrame but simpler
          else:
              # Standard deepcopy with memo
              copied[name] = copy.deepcopy(value, memo=memo)

      return copied, memo, failed

The memo dictionary is crucial: it ensures objects referenced multiple
times in the namespace are copied only once.


5.2 Large Column Sampling
-------------------------
For very large Series (>10M rows), checking every value for immutability
is expensive. The is_column_all_immutable() function:
  1. Checks the first non-NA value to get expected type
  2. For small series: checks all elements
  3. For large series: randomly samples _IMMUTABLE_CHECK_SAMPLE_SIZE elements
  4. Uses type identity check as fast path (type(val) is first_type)

This is probabilistic - a column with 99.9% immutable values and one mutable
value might be incorrectly classified as all-immutable. In practice:
  - Data science columns are typically homogeneous
  - The mutable object would need to be in the unsampled portion
  - The consequence is a shallow copy instead of deep copy


5.3 Sanity Check Mode
---------------------
Optional sanity_check=True mode in Checkpoints:
  - After saving, compares original namespace to checkpoint
  - Uses Diff class from diff.py for structured comparison
  - Raises ValueError if differences found
  - Useful for debugging but expensive (doubles comparison work)


6. ASSUMPTIONS
--------------
  a) Pandas CoW is enabled globally (we set it in this module)
  b) Object dtype columns contain only picklable/deepcopyable objects
  c) Variables follow Python naming conventions (underscore prefix = private)
  d) Namespaces are dict-like with string keys
  e) Matplotlib objects are not critical to checkpoint (excluded)
  f) Users don't rely on exact object identity across checkpoint/restore
  g) The random seed is not controlled (sampling may vary between runs)


7. CORNER CASES
---------------

7.1 Circular References
-----------------------
Handled correctly via copy.deepcopy's memo mechanism:
  a = []
  a.append(a)  # Self-referential list

The memo ensures the copied list references itself, not the original.


7.2 Object Columns with Mixed Types
-----------------------------------
  df["mixed"] = [1, "string", [1, 2, 3], None]

The immutability check finds [1, 2, 3] (mutable) and triggers deep copy.
Even if it misses the list in sampling, the shallow copy still works -
just with shared references to mutable objects.


7.3 Namespace Changes During Checkpoint
---------------------------------------
If code modifies the namespace while save() is running (threading),
behavior is undefined. This module is NOT thread-safe.


7.4 Variables with Custom __deepcopy__
--------------------------------------
Objects implementing __deepcopy__ are handled by copy.deepcopy().
The memo is passed through, so object-level customization is respected.


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


8.4 Sampling Probabilistic Errors
---------------------------------
Large column immutability check uses sampling. In pathological cases:
  - Column with millions of strings + one list
  - If list is not sampled, column treated as immutable
  - Shallow copy instead of deep copy
  - Mutations to the list affect both original and checkpoint

In practice, this is rare because:
  - Data columns are usually homogeneous
  - Sample size is large (10M)
  - The pathological case requires deliberate construction


8.5 Function/Lambda Checkpointing
---------------------------------
Functions ARE checkpointed (not excluded), but:
  - They reference their original closure via __closure__
  - deepcopy doesn't copy closure variables by default
  - Closures reference the same objects as before checkpoint

This means restored functions may see modified closure state.


8.6 Class Definitions Not Checkpointed
--------------------------------------
If user defines a class in a cell, class instances are checkpointed
but the class definition itself isn't (it's a type, not excluded,
but not deeply copied meaningfully).


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
  - skip_immutable_copy=True (default): Skip deep copy for immutable objects
  - convert_object_to_specialized=True (default): Convert object→specialized dtype

Disabling these may be useful for debugging but hurts performance.


9.3 Memory/Time Tradeoff
------------------------
  - Shallow copy + CoW: Fast, memory-efficient until mutation
  - Deep copy: Slow, doubles memory immediately

The current strategy (shallow + selective deep) balances both.


10. USAGE EXAMPLES
------------------

Basic usage:
    from data_ferret.kernel.checkpoint import Checkpoints

    checkpoints = Checkpoints()

    # Save current state
    saved, removed = checkpoints.save("before_experiment", user_ns)

    # ... run experiments that modify user_ns ...

    # Restore to saved state
    checkpoints.restore("before_experiment", user_ns)

Comparing checkpoints:
    from data_ferret.kernel.checkpoint import Checkpoint

    cp1 = checkpoints.get("v1")
    cp2 = checkpoints.get("v2")

    diff_result = Checkpoint.diff(cp1, cp2)
    # diff_result contains structured tree of changes

Type inspection:
    type_models = checkpoints.type_models(user_ns)
    for name, model in type_models.items():
        print(f"{name}: {model}")


11. DEPENDENCIES
----------------
Internal:
  - data_ferret.kernel.diff.Diff: Structured diff between checkpoints
  - data_ferret.kernel.extended_types.TypeModel, get_type_model: Type introspection
  - data_ferret.util.output.log, timer: Logging and performance instrumentation

External:
  - copy: Python standard library deep copy
  - datetime, decimal, random, time, types: Standard library utilities
  - numpy: Array handling and scalar types
  - pandas: DataFrame/Series handling and dtype inference


12. FUTURE WORK / TODOS
-----------------------
  - [ ] Incremental checkpointing (store only changes from previous)
  - [ ] Checkpoint compression for memory efficiency
  - [ ] Async checkpoint creation (background deep copy)
  - [ ] Checkpoint serialization (save to disk)
  - [ ] Better handling of closures in functions
  - [ ] Configurable exclusion patterns (not just matplotlib)
  - [ ] Thread safety (at least via locking)
  - [ ] Size estimation before checkpoint (warn on large data)


13. TESTING NOTES
-----------------
Key test scenarios:
  1. Basic save/restore roundtrip
  2. Mutation isolation (change original, verify checkpoint unchanged)
  3. Multiple references (list shared by two variables)
  4. Circular references
  5. Large DataFrames with various dtypes
  6. Object columns with mixed mutable/immutable content
  7. Repeated restore from same checkpoint
  8. Checkpoint deletion and memory cleanup
  9. Error handling for non-copyable objects
  10. Sanity check mode verification


================================================================================
                            END DESIGN DOCUMENT
================================================================================
"""

from __future__ import annotations

import copy
import datetime
import decimal
import random
import time
import types
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

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


def is_immutable_type(obj: Any) -> bool:
    """
    Check if an object is of an immutable type that's safe to skip copying.

    Returns True for basic immutable types commonly found in data science code,
    including Python primitives, NumPy scalars, and pandas temporal types.

    Args:
        obj: Object to check

    Returns:
        True if the object is immutable and safe to skip deep copying
    """
    if obj is None or obj is pd.NA:
        return True

    # Basic immutable types
    if isinstance(obj, (int, float, str, bool, bytes, complex, frozenset, range)):
        return True

    # NumPy scalar types (all are immutable)
    if isinstance(obj, np.generic):
        return True

    # Date/time types
    if isinstance(obj, (datetime.date, datetime.time, datetime.timedelta)):
        return True

    # Pandas temporal types
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return True

    # Decimal
    if isinstance(obj, decimal.Decimal):
        return True

    return False


# Number of random samples to check for large series
_IMMUTABLE_CHECK_SAMPLE_SIZE = 10_000_000


def _is_na(val) -> bool:
    """Check if a value is NA/None, handling non-scalar types safely."""
    if val is None:
        return True
    try:
        result = pd.isna(val)
        # pd.isna returns array for sequences - treat as not NA
        if isinstance(result, (bool, np.bool_)):
            return result
        return False
    except (ValueError, TypeError):
        # pd.isna fails for some types - they're not NA
        return False


def is_column_all_immutable(series: pd.Series) -> bool:
    """
    Check if all non-null values in a Series are immutable types.

    For small series, checks all elements. For large series, samples a random
    subset for performance. Uses type consistency as a fast path since data
    science columns are typically homogeneous.

    Args:
        series: pandas Series to check

    Returns:
        True if all sampled non-null values are immutable types
    """
    n = len(series)
    if n == 0:
        return True

    # Find first non-NA value without creating new Series
    first_val = None
    first_type = None
    first_idx = 0
    for i, val in enumerate(series):
        if not _is_na(val):
            if not is_immutable_type(val):
                return False
            first_val = val
            first_type = type(val)
            first_idx = i
            break

    if first_val is None:
        return True  # All NA

    log(f"first_type: {first_type}")

    # For small series, check all remaining elements
    if n <= _IMMUTABLE_CHECK_SAMPLE_SIZE:
        for val in series.iloc[first_idx + 1:]:
            if not _is_na(val):
                if type(val) is not first_type and not is_immutable_type(val):
                    log(f"val: {val} type: {type(val)}")
                    return False
        return True

    # For large series, randomly sample indices
    remaining_indices = list(range(first_idx + 1, n))
    sample_size = min(_IMMUTABLE_CHECK_SAMPLE_SIZE, len(remaining_indices))
    if sample_size > 0:
        sampled_indices = random.sample(remaining_indices, sample_size)
        for idx in sampled_indices:
            val = series.iloc[idx]
            if not _is_na(val):
                if type(val) is not first_type and not is_immutable_type(val):
                    log(f"val: {val} type: {type(val)}")
                    return False

    return True


def convert_series_object_to_specialized(series: pd.Series) -> pd.Series:
    """
    Convert object dtype Series to appropriate dtypes when possible.

    Handles all reasonable infer_dtype results:
    - integer/mixed-integer → Int64
    - floating/mixed-integer-float → float64
    - string → string (StringDtype)
    - bytes → bytes (object, could be optimized in future)
    - decimal → float64
    - complex → complex128
    - boolean → boolean
    - datetime64/datetime/date → datetime64[ns]
    - timedelta64/timedelta → timedelta64[ns]
    - categorical → category
    - period → period (already proper)
    - mixed/time/unknown-array → object (no conversion)

    Does NOT parse strings to numbers.

    Args:
        series: Series to convert

    Returns:
        Converted Series (or original if no conversion possible)
    """
    if series.dtype != object or series.empty:
        return series

    kind = infer_dtype(series, skipna=True)

    try:
        # Integers: ints + None/NaN
        if kind in {"integer", "mixed-integer"}:
            return series.astype("Int64")

        # Floats or int+float mixture
        elif kind in {"floating", "mixed-integer-float"}:
            return series.astype(float)

        # Decimal: convert to float
        elif kind == "decimal":
            return series.astype(float)

        # Complex numbers
        elif kind == "complex":
            return series.astype(complex)

        # Strings: convert to pandas string dtype
        elif kind == "string":
            return series.astype("string")

        # Booleans
        elif kind == "boolean":
            return series.astype("boolean")

        # Datetime types
        elif kind in {"datetime64", "datetime", "date"}:
            return pd.to_datetime(series)

        # Timedelta types
        elif kind in {"timedelta64", "timedelta"}:
            return pd.to_timedelta(series)

        # Categorical: convert to category dtype
        elif kind == "categorical":
            return series.astype("category")

        # Period: already a proper dtype, but ensure it's converted
        elif kind == "period":
            # Period arrays should already be proper dtype, but try to ensure
            return series  # Usually already PeriodDtype

        # Mixed types, time objects, bytes, unknown-array: leave as object
        # These are either too heterogeneous or don't have better representations
        else:
            return series

    except (TypeError, ValueError, Exception):
        # If any conversion fails, return original series
        return series


def convert_dataframe_object_to_specialized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object dtype columns in DataFrame to appropriate dtypes when possible.

    Returns a copy of the DataFrame where object columns are converted based on
    infer_dtype results. See convert_series_object_to_specialized for full list of
    conversions.

    Handles: integers, floats, strings, decimals, complex, booleans, datetimes,
    timedeltas, and categorical data. Does NOT parse strings to numbers.

    Args:
        df: DataFrame to convert

    Returns:
        DataFrame with converted columns
    """
    df2 = df.copy()
    obj_cols = df2.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        df2[col] = convert_series_object_to_specialized(df2[col])
        if df2[col].dtype != object:
            log(f"Converted column {col} from object to {df2[col].dtype}")
        else:
            log(f"Column {col} still has object dtype")

    return df2


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
        a: Checkpoint, b: Checkpoint, keys_to_include: set[str] | None = None
    ):
        """
        Compare two checkpoints and return structured diff results.

        Args:
            a: First checkpoint to compare
            b: Second checkpoint to compare
            keys_to_include: Optional set of keys to limit comparison to

        Returns:
            DiffResult: Structured diff tree with only differences
        """
        differ = Diff(strict=False, report_close=False, atol=1e-5, rtol=1e-5)
        return differ.diff(a.user_ns, b.user_ns, keys_to_include)


class Checkpoints:
    """
    Manager for multiple named checkpoints of kernel state.

    Provides save, restore, and comparison operations for kernel namespace
    snapshots. Handles deep copying with special optimizations for pandas
    objects and immutable types.

    Attributes:
        sanity_check: If True, verify copies match originals after save
        skip_immutable_copy: If True, skip deep copy for immutable objects in pandas columns
        convert_object_to_specialized: If True, convert object dtypes to specialized types before copying
        saved: Dictionary mapping checkpoint names to Checkpoint objects
    """

    def __init__(
        self,
        sanity_check: bool = False,
        skip_immutable_copy: bool = True,
        convert_object_to_specialized: bool = True,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            sanity_check: If True, verify copies match originals after save
            skip_immutable_copy: If True, skip deep copy for immutable objects
            convert_object_to_specialized: If True, convert object dtypes to specialized types before copying
        """
        self.sanity_check = sanity_check
        self.skip_immutable_copy = skip_immutable_copy
        self.convert_object_to_specialized = convert_object_to_specialized
        self.saved: dict[str, Checkpoint] = {}

    def _deep_copy_user_ns(
        self, variables: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[int, Any], dict[str, Exception]]:
        """
        Deep copy a dictionary of variables, with special handling for pandas objects.

        Ensures that mutable objects inside pandas DataFrames and Series are fully
        deep copied to prevent shared references. For pandas objects with object dtype,
        this method ensures that mutable objects stored in cells (like lists, dicts,
        custom objects) are properly deep copied rather than creating shallow references.

        Args:
            variables: Dictionary of variables to copy

        Returns:
            Tuple of (copied dictionary, memo dictionary for tracking copied objects,
                     dictionary of failed variables with their exceptions)
        """
        copied = {}
        memo = {}
        failed = {}

        for k, v in variables.items():
            # with timer(key="deep_copy_variable", message=f"Deep copying variable {k}"):
            try:
                start_time = time.time()
                if isinstance(v, pd.DataFrame):
                    # Check if DataFrame has any object dtype columns
                    has_object_columns = any(
                        v[col].dtype == object for col in v.columns
                    )
                    # Shallow copy is sufficient: CoW handles non-object columns,
                    # and we replace object columns entirely with apply(deepcopy) below
                    df_copy = v.copy(deep=False)

                    if has_object_columns:
                        # Deep copy object columns to ensure mutable objects in cells
                        # are truly independent
                        for col in df_copy.columns:
                            if df_copy[col].dtype == object:
                                # Skip deepcopy if optimization enabled and column contains only immutable objects
                                if (
                                    self.skip_immutable_copy
                                    and is_column_all_immutable(df_copy[col])
                                ):
                                    continue
                                log(f"Deep copying column {col}")
                                df_copy[col] = df_copy[col].apply(
                                    lambda x: copy.deepcopy(x, memo=memo)
                                )

                    memo[id(v)] = df_copy
                    copied[k] = df_copy

                elif isinstance(v, pd.Series):
                    # Shallow copy is sufficient: CoW handles non-object Series,
                    # and we replace object Series entirely with apply(deepcopy) below
                    series_copy = v.copy(deep=False)
                    if v.dtype == object:
                        # Skip deepcopy if optimization enabled and series contains only immutable objects
                        if not (
                            self.skip_immutable_copy
                            and is_column_all_immutable(series_copy)
                        ):
                            series_copy = series_copy.apply(
                                lambda x: copy.deepcopy(x, memo=memo)
                            )
                    memo[id(v)] = series_copy
                    copied[k] = series_copy
                else:
                    # For all other types, use standard deepcopy with memo tracking
                    copied[k] = copy.deepcopy(v, memo=memo)

                end_time = time.time()
                duration = end_time - start_time
                if duration > 0.010:
                    log(f"Deep copying variable {k} took {duration:.3f} seconds")
            except Exception as e:
                # Track variables that failed to copy
                failed[k] = e

        return copied, memo, failed

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

    def _convert_objects_to_specialized(self, user_ns: dict[str, Any]) -> None:
        """
        Convert object dtypes to specialized types in-place in user_ns.

        Iterates through all variables in user_ns and converts DataFrames and
        Series with object dtypes to more specialized types (Int64, float64,
        string, boolean, datetime64, etc.) based on infer_dtype results.

        Modifies user_ns in-place. Logs conversions and errors.

        Args:
            user_ns: User namespace dictionary to modify in-place
        """
        for k, v in list(user_ns.items()):
            try:
                if isinstance(v, pd.DataFrame):
                    # Check if DataFrame has any object dtype columns
                    obj_cols = v.select_dtypes(include=["object"]).columns
                    if len(obj_cols) > 0:
                        with timer(
                            key="convert_dataframe_object_to_specialized",
                            message=f"Converting DataFrame {k} object columns",
                        ):
                            user_ns[k] = convert_dataframe_object_to_specialized(v)
                elif isinstance(v, pd.Series) and v.dtype == object:
                    with timer(
                        key="convert_series_object_to_specialized",
                        message=f"Converting Series {k} object column",
                    ):
                        user_ns[k] = convert_series_object_to_specialized(v)
            except Exception as e:
                # If conversion fails, just use the original value
                log(f"Error converting {k}: {e}")

    def save(
        self, name: str, user_ns: dict[str, Any]
    ) -> tuple[dict[str, TypeModel], dict[str, TypeModel]]:
        """
        Save a checkpoint of the current namespace.

        Args:
            name: Identifier for this checkpoint (overwrites if exists)
            user_ns: User namespace dictionary to checkpoint (modified in-place if convert_object_to_specialized is True)

        Returns:
            Tuple of (saved variables with type models, removed variables with type models).
            Removed includes variables that couldn't be checkpointed or failed to copy.
        """
        with timer(key="deep_copy_user_ns", message="Deep copying user namespace"):
            if self.convert_object_to_specialized:
                self._convert_objects_to_specialized(user_ns)

            saved = {}
            removed = {}
            checkpointable_vars = self.checkpointable_vars(user_ns)

            checkpointable_values = self.checkpointable_values(checkpointable_vars)
            for k in checkpointable_vars.keys() - checkpointable_values.keys():
                removed[k] = get_type_model(user_ns[k])


            # Use helper to deep copy all variables with pandas awareness
            cp, memo, failed = self._deep_copy_user_ns(checkpointable_values)

            # Track successfully copied variables
            for k in cp:
                saved[k] = get_type_model(checkpointable_values[k])

            # Track variables that failed to copy
            for k in failed:
                removed[k] = get_type_model(checkpointable_values[k])

            self.saved[name] = Checkpoint(name, cp, memo)

        if self.sanity_check:
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
