"""
Data Ferret Kernel Namespace Diff Comparator
Compares Jupyter kernel user namespaces for equality with isomorphic pointer structure.
"""

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas.core.groupby.ops import BaseGrouper
from typing import Any, Dict, Set, Tuple, Optional
import math

from data_ferret.kernel.types import (
    ValueComparison,
    CompoundDiff,
    DiffNode,
    DiffResult,
    IndexComponent,
    KeyComponent,
    AttributeComponent,
    DataFrameLocation,
)
from data_ferret.util.output import log, timer


def _is_floating_dtype(dtype) -> bool:
    """
    Safely check if a dtype is a floating point type.

    Handles both numpy dtypes and pandas extension dtypes.

    Args:
        dtype: A numpy dtype or pandas extension dtype

    Returns:
        True if the dtype represents floating point numbers
    """
    try:
        # Try pandas first (works for both numpy and extension dtypes)
        from pandas.api.types import is_float_dtype
        return is_float_dtype(dtype)
    except Exception:
        # Fallback to numpy (but this will fail for extension dtypes)
        try:
            return np.issubdtype(dtype, np.floating)
        except (TypeError, AttributeError):
            return False


def _is_integer_dtype(dtype) -> bool:
    """
    Safely check if a dtype is an integer type.

    Handles both numpy dtypes and pandas extension dtypes.

    Args:
        dtype: A numpy dtype or pandas extension dtype

    Returns:
        True if the dtype represents integers
    """
    try:
        # Try pandas first (works for both numpy and extension dtypes)
        from pandas.api.types import is_integer_dtype
        return is_integer_dtype(dtype)
    except Exception:
        # Fallback to numpy (but this will fail for extension dtypes)
        try:
            return np.issubdtype(dtype, np.integer)
        except (TypeError, AttributeError):
            return False


def _all_elements_are_floats(arr) -> bool:
    """
    Check if all elements in an object-dtype array/series are floats.

    Args:
        arr: numpy array or pandas Series with object dtype

    Returns:
        True if all non-NaN elements are float instances, False otherwise
    """
    if isinstance(arr, pd.Series):
        arr = arr.values

    # Check each element
    for elem in arr.flat:
        # Skip NaN values (they're allowed in float arrays)
        if pd.isna(elem):
            continue
        # Check if element is a float (Python float or numpy floating)
        if not isinstance(elem, (float, np.floating)):
            return False

    return True


def are_compatible_dtypes(arr1, arr2) -> bool:
    """
    Check if two numpy arrays or pandas Series have compatible dtypes for equality comparison.

    Returns True if:
    - Both are integer types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, Int64, etc.)
    - Both are floating types (float16, float32, float64, Float64, etc.)
    - Both are string types (object with strings, StringDtype)
    - One is object dtype containing all floats, and the other is a floating type
    - Both are the exact same type

    Args:
        arr1: First numpy array or pandas Series
        arr2: Second numpy array or pandas Series
    """
    # Extract dtypes
    dtype1 = arr1.dtype
    dtype2 = arr2.dtype

    # Same dtype - always compatible
    if dtype1 == dtype2:
        return True

    # Handle pandas extension dtypes (StringDtype, Int64Dtype, etc.)
    # These need special handling before numpy dtype checks
    from pandas.api.types import is_extension_array_dtype, is_string_dtype, is_integer_dtype, is_float_dtype

    # Both are string types (object with strings or StringDtype)
    if is_string_dtype(dtype1) and is_string_dtype(dtype2):
        return True

    # Both are integer types (numpy int or pandas Int64, etc.)
    if is_integer_dtype(dtype1) and is_integer_dtype(dtype2):
        return True

    # Both are floating types (numpy float or pandas Float64, etc.)
    if is_float_dtype(dtype1) and is_float_dtype(dtype2):
        return True

    # If either is an extension dtype that we haven't handled above,
    # they're not compatible unless they're exactly the same (already checked)
    if is_extension_array_dtype(dtype1) or is_extension_array_dtype(dtype2):
        return False

    # For numpy dtypes only (not extension dtypes):
    # Both are integer types (signed or unsigned)
    if _is_integer_dtype(dtype1) and _is_integer_dtype(dtype2):
        return True

    # Both are floating types
    if _is_floating_dtype(dtype1) and _is_floating_dtype(dtype2):
        return True

    # Check object dtype compatibility with float dtypes
    # If one is object and the other is floating, check if object contains all floats
    if dtype1 == np.object_ and _is_floating_dtype(dtype2):
        return _all_elements_are_floats(arr1)

    if dtype2 == np.object_ and _is_floating_dtype(dtype1):
        return _all_elements_are_floats(arr2)

    return False


class Diff:
    """
    Compare two Jupyter kernel user namespaces for equality.
    Checks value equality and isomorphic pointer structure.
    Returns structured diff results with all differences found.
    """

    def __init__(
        self,
        rtol=1e-5,
        atol=1e-8,
        max_diffs_per_container: int = 10,
        max_diffs_per_structure: int = 100,
        sample_large_arrays: bool = False,
        strict: bool = True,
        report_close: bool = True,
        use_leq: bool = False,
        column_rbw: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Initialize the Diff comparator.

        Args:
            rtol: Relative tolerance for floating point comparisons (default: 1e-5)
            atol: Absolute tolerance for floating point comparisons (default: 1e-8)
            max_diffs_per_container: Maximum differences to collect per container (default: 1000)
            max_diffs_per_structure: Maximum differences to collect per structured data
                                     (arrays, Series, DataFrames) (default: 5)
            sample_large_arrays: Whether to sample large arrays instead of full comparison (default: False)
            strict: If True, require exact type matches. If False, allow compatible types
                    (e.g., int vs float, list vs ndarray) (default: True)
            report_close: If True, report floats that are close (within tolerance) with status='close'.
                         If False, treat close values as equal and don't report them (default: True)
            use_leq: If True, check if b is a conservative extension of a (default: False).
                     This means: (1) extra keys in b are allowed, (2) DataFrames in b can have
                     extra columns as long as all columns from a are present and equal.
            column_rbw: Column-level reads-before-writes for DataFrames (default: None).
                       Maps variable path (e.g., 'df', 'data["train"]') to set of column names
                       that were read-before-write. When provided with use_leq=True, only these
                       columns are compared for each DataFrame.

        Example:
            >>> # Default behavior - reports close values
            >>> differ = Diff(rtol=1e-5)
            >>> result = differ.diff({'x': 1.0000001}, {'x': 1.0000002})
            >>> 'x' in result  # True - close value is reported

            >>> # With report_close=False - treats close as equal
            >>> differ = Diff(rtol=1e-5, report_close=False)
            >>> result = differ.diff({'x': 1.0000001}, {'x': 1.0000002})
            >>> 'x' in result  # False - close value not reported
        """
        self.rtol = rtol
        self.atol = atol
        self.max_diffs_per_container = max_diffs_per_container
        self.max_diffs_per_structure = max_diffs_per_structure
        self.sample_large_arrays = sample_large_arrays
        self.strict = strict
        self.report_close = report_close
        self.use_leq = use_leq
        self.column_rbw = column_rbw or {}
        # Track object identities to ensure pointer structure matches
        self.id_map_a = {}  # Maps id(obj_a) -> canonical_id
        self.id_map_b = {}  # Maps id(obj_b) -> canonical_id
        self.next_canonical_id = 0

    def _is_immutable_atomic(self, val: Any) -> bool:
        """
        Check if a value is an immutable atomic type that doesn't need pointer tracking.

        Immutable atomics are values where identity doesn't matter - only value matters.
        These include: None, bool, int, float, complex, str, bytes

        Args:
            val: The value to check

        Returns:
            True if val is an immutable atomic, False otherwise
        """
        # None is always immutable atomic
        if val is None:
            return True

        # bool (must check before int since bool is subclass of int)
        if isinstance(val, bool):
            return True

        # Numeric types
        if isinstance(val, (int, np.integer)):
            return True
        if isinstance(val, (float, np.floating)):
            return True
        if isinstance(val, (complex, np.complexfloating)):
            return True

        # String and bytes
        if isinstance(val, (str, bytes)):
            return True

        return False

    def _get_type_category(self, val: Any) -> str:
        """
        Get the type category for immutable atomic values.

        This groups semantically equivalent types together:
        - All integer types (int, np.int8, np.int16, np.int32, np.int64) → "integer"
        - All float types (float, np.float16, np.float32, np.float64) → "float"
        - All complex types (complex, np.complex64, np.complex128) → "complex"

        This allows numpy scalar type variants to be compared as the same type category,
        so that np.int64(5) == np.int32(5) or int(5) == np.int64(5).

        Returns:
            String representing the type category
        """
        if val is None:
            return "none"
        # Check bool before int (bool is subclass of int in Python)
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, (int, np.integer)):
            return "integer"
        if isinstance(val, (float, np.floating)):
            return "float"
        if isinstance(val, (complex, np.complexfloating)):
            return "complex"
        if isinstance(val, str):
            return "str"
        if isinstance(val, bytes):
            return "bytes"
        return "other"

    def diff(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        keys_to_include: Set[str] | None = None,
    ) -> DiffResult:
        """
        Compare two user namespaces.

        Args:
            a: First namespace dictionary
            b: Second namespace dictionary
            keys_to_include: Optional set of keys to compare (default: all keys)

        Returns:
            DiffResult instance containing diff trees for variables with differences.
            The differences dict is empty if all variables are equal.
        """
        if keys_to_include is None:
            keys_to_include = set(a.keys()) | set(b.keys())

        # Reset identity tracking for each comparison
        self.id_map_a = {}
        self.id_map_b = {}
        self.next_canonical_id = 0

        differences: Dict[str, DiffNode] = {}

        # Check for variables only in a
        only_in_a = set(a.keys()) - set(b.keys())
        for var in only_in_a & keys_to_include:
            differences[var] = ValueComparison(
                status="different",
                value1=a[var],
                value2=None,
                message="Variable was removed",
            )

        # Check for variables only in b (skip if use_leq since extra keys are allowed)
        if not self.use_leq:
            only_in_b = set(b.keys()) - set(a.keys())
            for var in only_in_b & keys_to_include:
                differences[var] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=b[var],
                    message="Variable was added",
                )

        # Compare common variables - only add to differences if not equal
        common_vars = set(a.keys()) & set(b.keys())
        for var in sorted(
            common_vars & keys_to_include
        ):  # Sort for deterministic output
            # with timer(key="compare_values", message=f"Comparing {var}"):
            diff_result = self._compare_values(a[var], b[var], path=var)
            if diff_result:  # Only include if there are differences
                differences[var] = diff_result

        return DiffResult(differences=differences)

    def _compare_values(
        self, val_a: Any, val_b: Any, path: str = ""
    ) -> Optional[DiffNode]:
        """
        Compare two values, dispatching to type-specific methods.
        Returns None if equal, otherwise returns DiffNode with differences.
        """
        # Skip pointer tracking for immutable atomic values
        # For these types, only value equality matters, not object identity
        both_immutable_atomic = self._is_immutable_atomic(
            val_a
        ) and self._is_immutable_atomic(val_b)

        # Get IDs for all values (needed for error handling even if not tracking)
        id_a, id_b = id(val_a), id(val_b)
        registered_ids = False  # Track whether we registered these IDs

        if not both_immutable_atomic:
            # Pointer tracking for mutable containers and complex objects

            # If we've seen val_a before, check if pointer structure matches
            if id_a in self.id_map_a:
                canonical_a = self.id_map_a[id_a]
                if id_b in self.id_map_b:
                    canonical_b = self.id_map_b[id_b]
                    if canonical_a != canonical_b:
                        return ValueComparison(
                            status="different",
                            value1=val_a,
                            value2=val_b,
                            message=f"Pointer structure mismatch at {path}",
                        )
                else:
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Pointer structure mismatch at {path} (first namespace has reference to earlier object)",
                    )
                return None  # Already compared, and structure matches

            # If we've seen val_b before but not val_a
            if id_b in self.id_map_b:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Pointer structure mismatch at {path} (second namespace has reference to earlier object)",
                )

            # Register these objects with the same canonical ID
            # We do this before comparing to handle circular references
            canonical_id = self.next_canonical_id
            self.next_canonical_id += 1
            self.id_map_a[id_a] = canonical_id
            self.id_map_b[id_b] = canonical_id
            registered_ids = True  # Mark that we registered these IDs

        # Type checking
        # For immutable atomics, use category-based type checking
        # This allows np.int64 to match np.int32, int to match np.int64, etc.
        if both_immutable_atomic:
            type_a_category = self._get_type_category(val_a)
            type_b_category = self._get_type_category(val_b)

            if type_a_category != type_b_category:
                # Different type categories (e.g., int vs str, int vs float)
                # In non-strict mode, check if types are compatible (e.g., int vs float)
                if not self.strict:
                    is_compatible, compat_type = self._types_compatible(val_a, val_b)
                    if is_compatible:
                        # Use flexible comparison for compatible types
                        if compat_type == "numeric":
                            result = self._compare_numeric_flexible(val_a, val_b, path)
                        else:
                            # Should not happen for atomics, but handle gracefully
                            result = ValueComparison(
                                status="different",
                                value1=val_a,
                                value2=val_b,
                                message=f"Type category mismatch at {path}: {type_a_category} vs {type_b_category} ({type(val_a).__name__} vs {type(val_b).__name__})",
                            )
                        return result

                # Strict mode or incompatible types
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Type category mismatch at {path}: {type_a_category} vs {type_b_category} ({type(val_a).__name__} vs {type(val_b).__name__})",
                )
            # Same type category - continue to value comparison below
        else:
            # For non-atomic types (containers), use exact type matching
            if type(val_a) != type(val_b):
                # In non-strict mode, check if types are compatible
                if not self.strict:
                    is_compatible, compat_type = self._types_compatible(val_a, val_b)
                    if is_compatible:
                        # Use flexible comparison for compatible types
                        if compat_type == "numeric":
                            result = self._compare_numeric_flexible(val_a, val_b, path)
                        elif compat_type in ("list_array", "tuple_array"):
                            result = self._compare_list_array_flexible(
                                val_a, val_b, path
                            )
                        else:
                            # Should not happen, but handle gracefully
                            result = ValueComparison(
                                status="different",
                                value1=val_a,
                                value2=val_b,
                                message=f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}",
                            )

                        # If comparison found a difference, unregister these objects
                        if result is not None and registered_ids:
                            del self.id_map_a[id_a]
                            del self.id_map_b[id_b]

                        return result

                # Strict mode or incompatible types - unregister and return type mismatch
                if registered_ids:
                    del self.id_map_a[id_a]
                    del self.id_map_b[id_b]
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}",
                )

        # Dispatch to type-specific methods
        result: Optional[DiffNode] = None
        if val_a is None:
            result = None  # Both None (type check passed), so equal
        elif isinstance(val_a, bool):
            result = self._compare_bool(val_a, val_b, path)
        elif isinstance(val_a, (int, np.integer)):
            result = self._compare_int(val_a, val_b, path)
        elif isinstance(val_a, (float, np.floating)):
            result = self._compare_float(val_a, val_b, path)
        elif isinstance(val_a, complex):
            result = self._compare_complex(val_a, val_b, path)
        elif isinstance(val_a, str):
            result = self._compare_str(val_a, val_b, path)
        elif isinstance(val_a, bytes):
            result = self._compare_bytes(val_a, val_b, path)
        # Pandas scalar types (must be before callable check)
        elif isinstance(val_a, pd.Timestamp):
            result = self._compare_timestamp(val_a, val_b, path)
        elif isinstance(val_a, pd.Timedelta):
            result = self._compare_timedelta(val_a, val_b, path)
        elif callable(val_a):
            result = self._compare_callable(val_a, val_b, path)
        elif isinstance(val_a, np.ndarray):
            result = self._compare_ndarray(val_a, val_b, path)
        elif isinstance(val_a, (DataFrameGroupBy, SeriesGroupBy)):
            result = self._compare_groupby(val_a, val_b, path)
        elif isinstance(val_a, pd.Index):
            result = self._compare_index(val_a, val_b, path)
        elif isinstance(val_a, pd.Series):
            result = self._compare_series(val_a, val_b, path)
        elif isinstance(val_a, pd.DataFrame):
            result = self._compare_dataframe(val_a, val_b, path)
        elif isinstance(val_a, list):
            result = self._compare_list(val_a, val_b, path)
        elif isinstance(val_a, tuple):
            result = self._compare_tuple(val_a, val_b, path)
        elif isinstance(val_a, frozenset):
            result = self._compare_frozenset(val_a, val_b, path)
        elif isinstance(val_a, set):
            result = self._compare_set(val_a, val_b, path)
        elif isinstance(val_a, dict):
            result = self._compare_dict(val_a, val_b, path)
        else:
            # User-defined objects
            result = self._compare_object(val_a, val_b, path)

        # If comparison found a difference, unregister these objects
        # so they don't pollute future comparisons
        # Only unregister if we actually registered (i.e., not immutable atomics)
        if result is not None and registered_ids:
            del self.id_map_a[id_a]
            del self.id_map_b[id_b]

        return result

    def _types_compatible(self, val_a: Any, val_b: Any) -> Tuple[bool, str]:
        """
        Check if two values have compatible types for non-strict comparison.

        Returns:
            Tuple of (is_compatible, compatibility_type) where compatibility_type is:
            - "numeric": int vs float compatibility
            - "list_array": list vs ndarray compatibility
            - "tuple_array": tuple vs ndarray compatibility
            - "": not compatible
        """
        # Check numeric compatibility (int vs float)
        is_int_a = isinstance(val_a, (int, np.integer)) and not isinstance(val_a, bool)
        is_int_b = isinstance(val_b, (int, np.integer)) and not isinstance(val_b, bool)
        is_float_a = isinstance(val_a, (float, np.floating))
        is_float_b = isinstance(val_b, (float, np.floating))

        if (is_int_a and is_float_b) or (is_float_a and is_int_b):
            return (True, "numeric")

        # Check list vs array compatibility
        if isinstance(val_a, list) and isinstance(val_b, np.ndarray):
            return (True, "list_array")
        if isinstance(val_a, np.ndarray) and isinstance(val_b, list):
            return (True, "list_array")

        # Check tuple vs array compatibility
        if isinstance(val_a, tuple) and isinstance(val_b, np.ndarray):
            return (True, "tuple_array")
        if isinstance(val_a, np.ndarray) and isinstance(val_b, tuple):
            return (True, "tuple_array")

        return (False, "")

    def _get_list_depth(self, lst: Any) -> int:
        """
        Determine the nesting depth of a list/tuple.

        Returns:
            0 for non-list/tuple
            1 for flat list/tuple
            2 for list/tuple of lists/tuples
            etc.
        """
        if not isinstance(lst, (list, tuple)):
            return 0

        if len(lst) == 0:
            return 1

        # Check first element to determine depth
        max_depth = 0
        for item in lst:
            if isinstance(item, (list, tuple)):
                depth = 1 + self._get_list_depth(item)
                max_depth = max(max_depth, depth)
            else:
                max_depth = max(max_depth, 1)

        return max_depth

    def _compare_numeric_flexible(
        self, val_a: Any, val_b: Any, path: str
    ) -> Optional[ValueComparison]:
        """
        Compare int vs float values in non-strict mode.
        Converts int to float and uses float comparison logic.
        """
        # Convert both to float for comparison
        float_a = float(val_a)
        float_b = float(val_b)

        return self._compare_float(float_a, float_b, path)

    def _compare_list_array_flexible(
        self, val_a: Any, val_b: Any, path: str
    ) -> Optional[DiffNode]:
        """
        Compare list/tuple vs ndarray in non-strict mode.
        Validates structure matches and compares element-by-element.
        """
        # Determine which is the list/tuple and which is the array
        if isinstance(val_a, (list, tuple)):
            lst, arr = val_a, val_b
            lst_is_a = True
        else:
            lst, arr = val_b, val_a
            lst_is_a = False

        # Check that list depth matches array dimensions
        list_depth = self._get_list_depth(lst)
        array_ndim = arr.ndim

        if list_depth != array_ndim:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Structure mismatch at {path}: {'list' if isinstance(lst, list) else 'tuple'} depth {list_depth} vs array ndim {array_ndim}",
            )

        # Convert list/tuple to array for shape comparison
        try:
            lst_as_array = np.array(lst)
        except (ValueError, TypeError) as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Cannot convert {'list' if isinstance(lst, list) else 'tuple'} to array at {path}: {str(e)}",
            )

        # Check shapes match
        if lst_as_array.shape != arr.shape:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Shape mismatch at {path}: {'list' if isinstance(lst, list) else 'tuple'} shape {lst_as_array.shape} vs array shape {arr.shape}",
            )

        # Compare element-by-element
        # Flatten both for easier iteration
        flat_lst = lst_as_array.ravel()
        flat_arr = arr.ravel()

        for i in range(len(flat_lst)):
            lst_val = flat_lst[i]
            arr_val = flat_arr[i]

            # Use flexible comparison for elements
            elem_diff = self._compare_values(lst_val, arr_val, f"{path}[{i}]")
            if elem_diff:
                message = elem_diff.message
                # Return first difference found
                idx = np.unravel_index(i, arr.shape)
                idx_tuple = tuple(int(x) for x in idx)
                return ValueComparison(
                    status="different",
                    value1=lst_val,
                    value2=arr_val,
                    message=f"Element mismatch at {path}[{idx_tuple}]: {lst_val} vs {arr_val} - {type(lst_val)} vs {type(arr_val)}"
                    + message,
                )

        # All elements match
        return None

    def _compare_bool(
        self, val_a: bool, val_b: bool, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Bool mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_int(
        self, val_a: int, val_b: int, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Integer mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_float(
        self, val_a: float, val_b: float, path: str
    ) -> Optional[ValueComparison]:
        # Handle NaN
        is_nan_a = math.isnan(val_a) if isinstance(val_a, float) else np.isnan(val_a)
        is_nan_b = math.isnan(val_b) if isinstance(val_b, float) else np.isnan(val_b)

        if is_nan_a and is_nan_b:
            return None  # Both NaN, considered equal
        if is_nan_a or is_nan_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Float mismatch at {path}: {val_a} vs {val_b} (one is NaN)",
            )

        # Check exact equality first
        if val_a == val_b:
            return None  # Exactly equal

        # Check if close within tolerance
        is_close = (
            math.isclose(val_a, val_b, rel_tol=self.rtol, abs_tol=self.atol)
            if isinstance(val_a, float)
            else np.isclose(val_a, val_b, rel_tol=self.rtol, abs_tol=self.atol)
        )
        if is_close:
            # If report_close is False, treat close values as equal (no difference)
            if not self.report_close:
                return None
            return ValueComparison(
                status="close",
                value1=val_a,
                value2=val_b,
                message=f"Float close at {path}: {val_a} vs {val_b} (within tolerance)",
            )

        # Not equal and not close
        return ValueComparison(
            status="different",
            value1=val_a,
            value2=val_b,
            message=f"Float mismatch at {path}: {val_a} vs {val_b}",
        )

    def _compare_complex(
        self, val_a: complex, val_b: complex, path: str
    ) -> Optional[DiffNode]:
        children: Dict[str, DiffNode] = {}
        real_diff = self._compare_float(val_a.real, val_b.real, f"{path}.real")
        if real_diff:
            children[".real"] = real_diff
        imag_diff = self._compare_float(val_a.imag, val_b.imag, f"{path}.imag")
        if imag_diff:
            children[".imag"] = imag_diff
        if children:
            return CompoundDiff(source_type="complex", children=children, truncated=False)
        return None

    def _compare_str(
        self, val_a: str, val_b: str, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"String mismatch at {path}: '{val_a}' vs '{val_b}'",
            )
        return None

    def _compare_bytes(
        self, val_a: bytes, val_b: bytes, path: str
    ) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Bytes mismatch at {path}",
            )
        return None

    def _compare_callable(
        self, val_a: Any, val_b: Any, path: str
    ) -> Optional[DiffNode]:
        """
        Compare callables.
        - For type objects (classes): compare by identity
        - For functions: ignore differences (redefinitions are common)
        - For bound methods: compare __func__ and __self__
        """
        # Check if both are type objects (classes)
        if isinstance(val_a, type) and isinstance(val_b, type):
            if val_a is not val_b:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Type mismatch at {path}: {val_a.__name__} vs {val_b.__name__}",
                )
            return None

        # Check if both are bound methods
        is_method_a = hasattr(val_a, "__self__") and hasattr(val_a, "__func__")
        is_method_b = hasattr(val_b, "__self__") and hasattr(val_b, "__func__")

        if is_method_a and is_method_b:
            # Both are bound methods - compare the underlying function and instance
            diffs = {}
            func_diff = self._compare_values(
                val_a.__func__, val_b.__func__, f"{path}.__func__"
            )
            if func_diff:
                diffs[".__func__"] = func_diff

            self_diff = self._compare_values(
                val_a.__self__, val_b.__self__, f"{path}.__self__"
            )
            if self_diff:
                diffs[".__self__"] = self_diff

            return diffs if diffs else None
        elif is_method_a != is_method_b:
            # One is a bound method, the other isn't
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Callable type mismatch at {path}: bound method vs function",
            )
        else:
            # ignore function/callable mismatch
            return None

    def _compare_ndarray(
        self, val_a: np.ndarray, val_b: np.ndarray, path: str
    ) -> Optional[DiffNode]:
        """Compare numpy arrays, collecting up to max_diffs_per_container differences.

        This method records structural mismatches (shape, dtype) but continues
        comparing elements where possible.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Check shape - record mismatch but can't compare elements with different shapes
        if val_a.shape != val_b.shape:
            children["_shape"] = ValueComparison(
                status="different",
                value1=val_a.shape,
                value2=val_b.shape,
                message=f"Array shape mismatch at {path}: {val_a.shape} vs {val_b.shape}",
            )
            # Can't compare elements with different shapes
            return CompoundDiff(source_type="array", children=children, truncated=False)

        # Check dtype compatibility - record mismatch but can't compare incompatible dtypes
        if not are_compatible_dtypes(val_a, val_b):
            children["_dtype"] = ValueComparison(
                status="different",
                value1=val_a.dtype,
                value2=val_b.dtype,
                message=f"Array dtype mismatch at {path}: {val_a.dtype} vs {val_b.dtype}",
            )
            # Can't compare elements with incompatible dtypes
            return CompoundDiff(source_type="array", children=children, truncated=False)

        # Cast to common type if dtypes are compatible but different
        if val_a.dtype != val_b.dtype:
            # Special handling for object dtype with floats
            if val_a.dtype == np.object_ and _is_floating_dtype(val_b.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            elif val_b.dtype == np.object_ and _is_floating_dtype(val_a.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            else:
                # Check if either is an extension dtype
                from pandas.api.types import is_extension_array_dtype
                if is_extension_array_dtype(val_a.dtype) or is_extension_array_dtype(val_b.dtype):
                    # Extension dtypes can't use np.promote_types, compare as-is
                    val_a_cmp = val_a
                    val_b_cmp = val_b
                else:
                    # Safe to use np.promote_types for numpy dtypes
                    common_dtype = np.promote_types(val_a.dtype, val_b.dtype)
                    val_a_cmp = val_a.astype(common_dtype)
                    val_b_cmp = val_b.astype(common_dtype)
        else:
            val_a_cmp = val_a
            val_b_cmp = val_b

        # Fast path: use vectorized operations to check if arrays are equal
        try:
            if _is_floating_dtype(val_a_cmp.dtype) or np.issubdtype(
                val_a_cmp.dtype, np.complexfloating
            ):
                # Fast vectorized check for floats
                if np.allclose(
                    val_a_cmp, val_b_cmp, rtol=self.rtol, atol=self.atol, equal_nan=True
                ):
                    return None  # Arrays are equal
            else:
                # Fast vectorized check for other types
                if np.array_equal(val_a_cmp, val_b_cmp):
                    return None  # Arrays are equal
        except Exception:
            pass  # Fall through to element-by-element comparison

        # Arrays are different - collect up to max_diffs_per_container differences
        diff_count = 0

        try:
            if _is_floating_dtype(val_a_cmp.dtype) or np.issubdtype(
                val_a_cmp.dtype, np.complexfloating
            ):
                # Iterate to find specific differences
                flat_a = val_a_cmp.ravel()
                flat_b = val_b_cmp.ravel()
                flat_a_orig = val_a.ravel()
                flat_b_orig = val_b.ravel()

                for i in range(len(flat_a)):
                    a_val, b_val = flat_a[i], flat_b[i]
                    both_nan = np.isnan(a_val) and np.isnan(b_val)

                    if not both_nan and not np.allclose(
                        [a_val], [b_val], rtol=self.rtol, atol=self.atol, equal_nan=True
                    ):
                        idx = np.unravel_index(i, val_a.shape)
                        idx_tuple = tuple(int(x) for x in idx)

                        children[f"[{idx_tuple}]"] = ValueComparison(
                            status="different",
                            value1=flat_a_orig[i],
                            value2=flat_b_orig[i],
                            message=f"Array values mismatch at {path}[{idx_tuple}]: {flat_a_orig[i]} vs {flat_b_orig[i]}",
                        )

                        diff_count += 1
                        if diff_count >= self.max_diffs_per_container:
                            truncated = True
                            break
            else:
                # Iterate to find specific differences
                # Use _compare_values() to properly handle floats with tolerance even in object dtype arrays
                flat_a = val_a_cmp.ravel()
                flat_b = val_b_cmp.ravel()
                flat_a_orig = val_a.ravel()
                flat_b_orig = val_b.ravel()

                for i in range(len(flat_a)):
                    idx = np.unravel_index(i, val_a.shape)
                    idx_tuple = tuple(int(x) for x in idx)

                    elem_diff = self._compare_values(flat_a_orig[i], flat_b_orig[i], f"{path}[{idx_tuple}]")
                    if elem_diff:
                        children[f"[{idx_tuple}]"] = elem_diff

                        diff_count += 1
                        if diff_count >= self.max_diffs_per_container:
                            truncated = True
                            break
        except Exception as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Array comparison error at {path}: {str(e)}",
            )

        if children:
            return CompoundDiff(source_type="array", children=children, truncated=truncated)
        return None

    def _compare_series(
        self, val_a: pd.Series, val_b: pd.Series, path: str
    ) -> Optional[DiffNode]:
        """Compare pandas Series, collecting up to max_diffs_per_container differences.

        This method records structural mismatches (index, name, dtype) but continues
        comparing values where possible.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Check index - if indexes differ, can't compare values by label
        indexes_match = val_a.index.equals(val_b.index)
        if not indexes_match:
            children["_index"] = ValueComparison(
                status="different",
                value1=val_a.index,
                value2=val_b.index,
                message=f"Series index mismatch at {path}",
            )
            # Can't compare values when indexes differ (labels don't match)
            return CompoundDiff(source_type="series", children=children, truncated=False)

        # Check name - record mismatch but continue
        if val_a.name != val_b.name:
            children["_name"] = ValueComparison(
                status="different",
                value1=val_a.name,
                value2=val_b.name,
                message=f"Series name mismatch at {path}: {val_a.name} vs {val_b.name}",
            )

        # Check dtype compatibility - record mismatch but can't compare incompatible dtypes
        if not are_compatible_dtypes(val_a, val_b):
            children["_dtype"] = ValueComparison(
                status="different",
                value1=val_a.dtype,
                value2=val_b.dtype,
                message=f"Series dtype mismatch at {path}: {val_a.dtype} vs {val_b.dtype}",
            )
            # Can't compare values with incompatible dtypes
            return CompoundDiff(source_type="series", children=children, truncated=False)

        # Cast to common type if dtypes are compatible but different
        if val_a.dtype != val_b.dtype:
            # Special handling for object dtype with floats
            if val_a.dtype == np.object_ and _is_floating_dtype(val_b.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            elif val_b.dtype == np.object_ and _is_floating_dtype(val_a.dtype):
                val_a_cmp = val_a.astype(np.float64)
                val_b_cmp = val_b.astype(np.float64)
            else:
                # Check if either is an extension dtype
                from pandas.api.types import is_extension_array_dtype
                if is_extension_array_dtype(val_a.dtype) or is_extension_array_dtype(val_b.dtype):
                    # Extension dtypes can't use np.promote_types, compare as-is
                    val_a_cmp = val_a
                    val_b_cmp = val_b
                else:
                    # Safe to use np.promote_types for numpy dtypes
                    common_dtype = np.promote_types(val_a.dtype, val_b.dtype)
                    val_a_cmp = val_a.astype(common_dtype)
                    val_b_cmp = val_b.astype(common_dtype)
        else:
            val_a_cmp = val_a
            val_b_cmp = val_b

        # Fast path: check if series are equal using vectorized operations
        try:
            if pd.api.types.is_float_dtype(val_a_cmp.dtype):
                # For floats, check NaN positions and values
                mask_nan_a = pd.isna(val_a_cmp)
                mask_nan_b = pd.isna(val_b_cmp)
                if mask_nan_a.equals(mask_nan_b):
                    non_nan_a = val_a_cmp[~mask_nan_a]
                    non_nan_b = val_b_cmp[~mask_nan_b]
                    if len(non_nan_a) == 0 or np.allclose(
                        non_nan_a, non_nan_b, rtol=self.rtol, atol=self.atol
                    ):
                        # Values equal, return any structural diffs
                        if children:
                            return CompoundDiff(source_type="series", children=children, truncated=False)
                        return None
            else:
                # For non-float types, use equals
                if val_a_cmp.equals(val_b_cmp):
                    # Values equal, return any structural diffs
                    if children:
                        return CompoundDiff(source_type="series", children=children, truncated=False)
                    return None
        except Exception:
            pass  # Fall through to element-by-element comparison

        # Series are different - collect up to max_diffs_per_container differences
        # Use vectorized operations to find diff indices, then only iterate over actual diffs

        try:
            if pd.api.types.is_float_dtype(val_a_cmp.dtype):
                # OPTIMIZED: Vectorized diff detection for float dtypes
                # Extract numpy arrays for fast access
                arr_a = val_a_cmp.values
                arr_b = val_b_cmp.values
                arr_a_orig = val_a.values  # Original values for output
                arr_b_orig = val_b.values
                index = val_a_cmp.index

                # Vectorized NaN detection
                nan_a = np.isnan(arr_a)
                nan_b = np.isnan(arr_b)

                # Find all difference indices at once:
                # 1. NaN position mismatches (one is NaN, other is not)
                nan_mismatch = nan_a != nan_b
                # 2. Value mismatches (both non-NaN but values differ beyond tolerance)
                both_valid = ~nan_a & ~nan_b
                value_mismatch = both_valid & ~np.isclose(
                    arr_a, arr_b, rtol=self.rtol, atol=self.atol
                )

                # Combined: all differences
                all_diffs = nan_mismatch | value_mismatch
                diff_indices = np.where(all_diffs)[0]

                # Check if truncation needed
                truncated = len(diff_indices) > self.max_diffs_per_container

                # Only iterate over actual differences (up to max_diffs)
                for i in diff_indices[: self.max_diffs_per_container]:
                    idx_label = index[i]
                    v1, v2 = arr_a_orig[i], arr_b_orig[i]

                    if nan_mismatch[i]:
                        children[f"[{repr(idx_label)}]"] = ValueComparison(
                            status="different",
                            value1=v1,
                            value2=v2,
                            message=f"Series NaN positions mismatch at {path}[{repr(idx_label)}]: is_nan={nan_a[i]} vs is_nan={nan_b[i]}",
                        )
                    else:
                        children[f"[{repr(idx_label)}]"] = ValueComparison(
                            status="different",
                            value1=v1,
                            value2=v2,
                            message=f"Series values mismatch at {path}[{repr(idx_label)}]: {v1} vs {v2}",
                        )
            else:
                # For non-float dtypes, use vectorized equality check first
                # then only call _compare_values on different elements
                arr_a = val_a_cmp.values
                arr_b = val_b_cmp.values
                index = val_a_cmp.index

                # Try vectorized comparison first for simple types
                try:
                    # For object dtype or other types, try pandas not-equal
                    not_equal = val_a_cmp != val_b_cmp
                    # Handle case where comparison returns non-boolean (e.g., nested objects)
                    if hasattr(not_equal, 'values'):
                        diff_mask = not_equal.values
                    else:
                        diff_mask = np.array([True] * len(arr_a))  # Fall back to check all
                except (TypeError, ValueError):
                    # Comparison failed, check all elements
                    diff_mask = np.array([True] * len(arr_a))

                diff_indices = np.where(diff_mask)[0]
                diff_count = 0

                # Only iterate over potentially different elements
                for i in diff_indices:
                    idx_label = index[i]
                    elem_diff = self._compare_values(
                        arr_a[i], arr_b[i], f"{path}[{repr(idx_label)}]"
                    )
                    if elem_diff:
                        children[f"[{repr(idx_label)}]"] = elem_diff
                        diff_count += 1
                        if diff_count >= self.max_diffs_per_container:
                            truncated = True
                            break
        except Exception as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Series comparison error at {path}: {str(e)}",
            )

        if children:
            return CompoundDiff(source_type="series", children=children, truncated=truncated)
        return None

    def _compare_dataframe(
        self, val_a: pd.DataFrame, val_b: pd.DataFrame, path: str
    ) -> Optional[DiffNode]:
        """Compare pandas DataFrames, collecting up to max_diffs_per_structure differences.

        This method NEVER returns early - it always compares all columns that exist in both
        DataFrames, accumulating all differences found.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False
        total_diff_count = 0

        # Determine which columns to compare
        cols_a = set(val_a.columns)
        cols_b = set(val_b.columns)

        if self.use_leq:
            # Check if we have column-level RBW info for this variable
            if path in self.column_rbw:
                rbw_cols = self.column_rbw[path]

                if not rbw_cols:
                    # No columns tracked - compare ALL common columns (conservative)
                    cols_to_compare = cols_a & cols_b
                else:
                    # Track missing RBW columns as differences (but don't return early!)
                    missing_in_a = rbw_cols - cols_a
                    for col in sorted(missing_in_a):
                        children[f"['{col}']"] = ValueComparison(
                            status="different",
                            value1=None,
                            value2=None,
                            message=f"DataFrame column '{col}' missing in pre-state at {path}",
                        )
                        total_diff_count += 1

                    missing_in_b = rbw_cols - cols_b
                    for col in sorted(missing_in_b):
                        children[f"['{col}']"] = ValueComparison(
                            status="different",
                            value1=None,
                            value2=None,
                            message=f"DataFrame column '{col}' deleted in post-state at {path}",
                        )
                        total_diff_count += 1

                    # Compare RBW columns that exist in both
                    cols_to_compare = rbw_cols & cols_a & cols_b
            else:
                # No column-level RBW info - in leq mode, compare all of val_a's columns
                # Track columns missing in b
                missing_cols = cols_a - cols_b
                for col in sorted(missing_cols):
                    children[f"['{col}']"] = ValueComparison(
                        status="different",
                        value1=val_a[col] if col in val_a.columns else None,
                        value2=None,
                        message=f"DataFrame column '{col}' missing in second DataFrame at {path}",
                    )
                    total_diff_count += 1

                # Compare columns that exist in both
                cols_to_compare = cols_a & cols_b
        else:
            # Standard mode: track column differences
            only_in_a = cols_a - cols_b
            only_in_b = cols_b - cols_a

            for col in sorted(only_in_a):
                children[f"['{col}']"] = ValueComparison(
                    status="different",
                    value1=val_a[col],
                    value2=None,
                    message=f"Column '{col}' only in first DataFrame",
                )
                total_diff_count += 1

            for col in sorted(only_in_b):
                children[f"['{col}']"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=val_b[col],
                    message=f"Column '{col}' only in second DataFrame",
                )
                total_diff_count += 1

            # Compare common columns
            cols_to_compare = cols_a & cols_b

        # Check index mismatch (record as difference but continue comparing columns)
        if not val_a.index.equals(val_b.index):
            children["_index"] = ValueComparison(
                status="different",
                value1=val_a.index,
                value2=val_b.index,
                message=f"DataFrame index mismatch at {path}",
            )
            total_diff_count += 1
            # We still try to compare columns if they have the same length
            if len(val_a) != len(val_b):
                # Can't compare columns with different row counts
                if children:
                    return CompoundDiff(source_type="dataframe", children=children, truncated=False)
                return None

        # Check dtype compatibility for each column (record mismatches but continue)
        cols_with_dtype_issues = set()
        for col in cols_to_compare:
            if not are_compatible_dtypes(val_a[col], val_b[col]):
                children[f"['{col}']._dtype"] = ValueComparison(
                    status="different",
                    value1=val_a[col].dtype,
                    value2=val_b[col].dtype,
                    message=f"DataFrame column '{col}' dtype mismatch at {path}: {val_a[col].dtype} vs {val_b[col].dtype}",
                )
                cols_with_dtype_issues.add(col)
                total_diff_count += 1

        # Only compare columns that don't have dtype issues
        cols_to_compare_values = cols_to_compare - cols_with_dtype_issues

        # Compare each column - _compare_series handles dtype casting internally
        for col in sorted(cols_to_compare_values):
            # with timer(key="compare_series", message=f"Comparing {col}"):
            col_diff = self._compare_series(
                val_a[col], val_b[col], f"{path}['{col}']"
            )
            # log(f"col_diff: {col_diff}")
            if col_diff:
                # Keep nested structure - don't flatten series diffs
                children[f"['{col}']"] = col_diff

                # Count diffs for truncation limit
                if isinstance(col_diff, CompoundDiff):
                    # Count non-metadata keys in the nested children
                    total_diff_count += len(col_diff.children) # len([k for k in col_diff.children if not k.startswith("_")])
                else:
                    total_diff_count += 1

                # Check if we've hit the limit
                if total_diff_count >= self.max_diffs_per_structure:
                    truncated = True
                    return CompoundDiff(source_type="dataframe", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="dataframe", children=children, truncated=truncated)
        return None

    def _compare_index(
        self, val_a: pd.Index, val_b: pd.Index, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas Index objects using .equals() method."""
        if not val_a.equals(val_b):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Index mismatch at {path}: {list(val_a)} vs {list(val_b)}",
            )
        return None

    def _compare_timestamp(
        self, val_a: pd.Timestamp, val_b: pd.Timestamp, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas Timestamp objects."""
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Timestamp mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_timedelta(
        self, val_a: pd.Timedelta, val_b: pd.Timedelta, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas Timedelta objects."""
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Timedelta mismatch at {path}: {val_a} vs {val_b}",
            )
        return None

    def _compare_groupby(self, val_a, val_b, path: str) -> Optional[ValueComparison]:
        """Compare pandas GroupBy objects (TODO: implement full diff collection)."""
        result = self._compare_groupby_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(
                status="different", value1=val_a, value2=val_b, message=result
            )
        return None

    def _compare_groupby_legacy(self, val_a, val_b, path: str) -> str:
        """
        Compare DataFrameGroupBy or SeriesGroupBy objects by their semantic properties.

        Excludes internal cache fields (_cache, _grouper._cache) that can differ
        between otherwise equivalent groupby objects.

        Compares:
        - The underlying data object (obj)
        - The grouper configuration (keys, sort, dropna)
        - Selection state
        """
        # Compare the underlying object (DataFrame or Series)
        if not hasattr(val_a, "obj") or not hasattr(val_b, "obj"):
            return f"GroupBy structure mismatch at {path}: missing obj attribute"

        obj_diff = self._compare_values(val_a.obj, val_b.obj, f"{path}.obj")
        if obj_diff:
            # _compare_values returns DiffNode now, extract message for legacy
            if isinstance(obj_diff, ValueComparison):
                return obj_diff.message
            elif isinstance(obj_diff, dict):
                # Compound diff - just indicate there's a difference
                return f"GroupBy data mismatch at {path}.obj"
            return obj_diff

        # Compare the grouper (excluding cache)
        if not hasattr(val_a, "_grouper") or not hasattr(val_b, "_grouper"):
            return f"GroupBy structure mismatch at {path}: missing _grouper attribute"

        grouper_diff = self._compare_grouper(
            val_a._grouper, val_b._grouper, f"{path}._grouper"
        )
        if grouper_diff:
            # _compare_grouper returns ValueComparison now, but we need string for legacy
            if isinstance(grouper_diff, ValueComparison):
                return grouper_diff.message
            return grouper_diff

        # Compare selection (which columns are selected)
        if hasattr(val_a, "_selection") and hasattr(val_b, "_selection"):
            if val_a._selection != val_b._selection:
                return f"GroupBy selection mismatch at {path}: {val_a._selection} vs {val_b._selection}"

        return ""

    def _compare_grouper(
        self, val_a: BaseGrouper, val_b: BaseGrouper, path: str
    ) -> Optional[ValueComparison]:
        """Compare pandas BaseGrouper objects (legacy wrapper)."""
        result = self._compare_grouper_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(
                status="different", value1=val_a, value2=val_b, message=result
            )
        return None

    def _compare_grouper_legacy(
        self, val_a: BaseGrouper, val_b: BaseGrouper, path: str
    ) -> str:
        """
        Compare BaseGrouper objects, excluding their internal cache.

        Compares the semantic properties that define the grouping:
        - groupings (keys)
        - sort flag
        - dropna flag
        - axis
        """
        # Compare axis (typically a pandas Index)
        if hasattr(val_a, "axis") and hasattr(val_b, "axis"):
            # Use pandas Index.equals() for proper comparison
            axes_equal = False
            if isinstance(val_a.axis, pd.Index) and isinstance(val_b.axis, pd.Index):
                axes_equal = val_a.axis.equals(val_b.axis)
            elif isinstance(val_a.axis, np.ndarray) and isinstance(
                val_b.axis, np.ndarray
            ):
                axes_equal = np.array_equal(val_a.axis, val_b.axis)
            else:
                axes_equal = val_a.axis == val_b.axis

            if not axes_equal:
                return f"Grouper axis mismatch at {path}: {val_a.axis} vs {val_b.axis}"

        # Compare sort flag
        if hasattr(val_a, "_sort") and hasattr(val_b, "_sort"):
            if val_a._sort != val_b._sort:
                return (
                    f"Grouper sort mismatch at {path}: {val_a._sort} vs {val_b._sort}"
                )

        # Compare dropna flag
        if hasattr(val_a, "dropna") and hasattr(val_b, "dropna"):
            if val_a.dropna != val_b.dropna:
                return f"Grouper dropna mismatch at {path}: {val_a.dropna} vs {val_b.dropna}"

        # Compare groupings (the actual grouping keys)
        if hasattr(val_a, "_groupings") and hasattr(val_b, "_groupings"):
            groupings_a = val_a._groupings
            groupings_b = val_b._groupings

            if len(groupings_a) != len(groupings_b):
                return f"Grouper groupings count mismatch at {path}: {len(groupings_a)} vs {len(groupings_b)}"

            for i, (grp_a, grp_b) in enumerate(zip(groupings_a, groupings_b)):
                # Compare key/name
                if hasattr(grp_a, "name") and hasattr(grp_b, "name"):
                    if grp_a.name != grp_b.name:
                        return f"Grouping name mismatch at {path}._groupings[{i}]: {grp_a.name} vs {grp_b.name}"

                # Compare key object if available
                if hasattr(grp_a, "key") and hasattr(grp_b, "key"):
                    if grp_a.key != grp_b.key:
                        return f"Grouping key mismatch at {path}._groupings[{i}]: {grp_a.key} vs {grp_b.key}"

        return ""

    def _compare_list(self, val_a: list, val_b: list, path: str) -> Optional[DiffNode]:
        """Compare lists, recording length mismatch but comparing common elements."""
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record length mismatch but continue comparing common elements
        if len(val_a) != len(val_b):
            children["_length"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"List length mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Compare common elements, collecting differences
        min_len = min(len(val_a), len(val_b))
        diff_count = 0
        for i in range(min_len):
            item_a, item_b = val_a[i], val_b[i]
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                children[f"[{i}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    break

        if children:
            return CompoundDiff(source_type="list", children=children, truncated=truncated)
        return None

    def _compare_tuple(
        self, val_a: tuple, val_b: tuple, path: str
    ) -> Optional[DiffNode]:
        """Compare tuples, recording length mismatch but comparing common elements."""
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record length mismatch but continue comparing common elements
        if len(val_a) != len(val_b):
            children["_length"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"Tuple length mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Compare common elements, collecting differences
        min_len = min(len(val_a), len(val_b))
        diff_count = 0
        for i in range(min_len):
            item_a, item_b = val_a[i], val_b[i]
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                children[f"[{i}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    break

        if children:
            return CompoundDiff(source_type="tuple", children=children, truncated=truncated)
        return None

    def _compare_set(
        self, val_a: set, val_b: set, path: str
    ) -> Optional[DiffNode]:
        """
        Compare sets by finding matching elements and comparing them recursively.
        This properly handles pointer structure within sets.
        Records size mismatch but continues to find unmatched elements.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record size mismatch but continue
        if len(val_a) != len(val_b):
            children["_size"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"Set size mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Convert to lists for matching
        list_a = list(val_a)
        list_b = list(val_b)

        # Try to find a matching between elements
        used_b = set()
        unmatched_a = []
        diff_count = len(children)

        for i, item_a in enumerate(list_a):
            found_match = False

            for j, item_b in enumerate(list_b):
                if j in used_b:
                    continue

                # Try to match item_a with item_b
                diff = self._compare_values(item_a, item_b, f"{path}{{element {i}}}")
                if not diff:
                    # Found a match
                    used_b.add(j)
                    found_match = True
                    break

            if not found_match:
                unmatched_a.append((i, item_a))

        # Record unmatched elements from set A
        for i, item_a in unmatched_a:
            children[f"{{unmatched_a_{i}}}"] = ValueComparison(
                status="different",
                value1=item_a,
                value2=None,
                message=f"Set element {repr(item_a)} from first set has no match in second set",
            )
            diff_count += 1
            if diff_count >= self.max_diffs_per_container:
                truncated = True
                return CompoundDiff(source_type="set", children=children, truncated=truncated)

        # Record unmatched elements from set B
        for j, item_b in enumerate(list_b):
            if j not in used_b:
                children[f"{{unmatched_b_{j}}}"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=item_b,
                    message=f"Set element {repr(item_b)} from second set has no match in first set",
                )
                diff_count += 1
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    return CompoundDiff(source_type="set", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="set", children=children, truncated=truncated)
        return None

    def _compare_frozenset(
        self, val_a: frozenset, val_b: frozenset, path: str
    ) -> Optional[DiffNode]:
        """
        Compare frozensets by finding matching elements and comparing them recursively.
        Records size mismatch but continues to find unmatched elements.
        """
        children: Dict[str, DiffNode] = {}
        truncated = False

        # Record size mismatch but continue
        if len(val_a) != len(val_b):
            children["_size"] = ValueComparison(
                status="different",
                value1=len(val_a),
                value2=len(val_b),
                message=f"Frozenset size mismatch at {path}: {len(val_a)} vs {len(val_b)}",
            )

        # Convert to lists for matching
        list_a = list(val_a)
        list_b = list(val_b)

        # Try to find a matching between elements
        used_b = set()
        unmatched_a = []
        diff_count = len(children)

        for i, item_a in enumerate(list_a):
            found_match = False

            for j, item_b in enumerate(list_b):
                if j in used_b:
                    continue

                # Try to match item_a with item_b
                diff = self._compare_values(item_a, item_b, f"{path}{{element {i}}}")
                if not diff:
                    # Found a match
                    used_b.add(j)
                    found_match = True
                    break

            if not found_match:
                unmatched_a.append((i, item_a))

        # Record unmatched elements from frozenset A
        for i, item_a in unmatched_a:
            children[f"{{unmatched_a_{i}}}"] = ValueComparison(
                status="different",
                value1=item_a,
                value2=None,
                message=f"Frozenset element {repr(item_a)} from first set has no match in second set",
            )
            diff_count += 1
            if diff_count >= self.max_diffs_per_container:
                truncated = True
                return CompoundDiff(source_type="frozenset", children=children, truncated=truncated)

        # Record unmatched elements from frozenset B
        for j, item_b in enumerate(list_b):
            if j not in used_b:
                children[f"{{unmatched_b_{j}}}"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=item_b,
                    message=f"Frozenset element {repr(item_b)} from second set has no match in first set",
                )
                diff_count += 1
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    return CompoundDiff(source_type="frozenset", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="frozenset", children=children, truncated=truncated)
        return None

    def _compare_dict(self, val_a: dict, val_b: dict, path: str) -> Optional[DiffNode]:
        children: Dict[str, DiffNode] = {}
        truncated = False
        keys_a = set(val_a.keys())
        keys_b = set(val_b.keys())

        # Check for key mismatches
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a

        for key in only_a:
            children[f"[{repr(key)}]"] = ValueComparison(
                status="different",
                value1=val_a[key],
                value2=None,
                message=f"Key {repr(key)} only in first dict",
            )

        for key in only_b:
            children[f"[{repr(key)}]"] = ValueComparison(
                status="different",
                value1=None,
                value2=val_b[key],
                message=f"Key {repr(key)} only in second dict",
            )

        # Compare common keys, collecting ALL differences
        common_keys = keys_a & keys_b
        diff_count = len(children)  # Count keys already different
        for key in sorted(common_keys, key=str):  # Sort for deterministic output
            diff = self._compare_values(val_a[key], val_b[key], f"{path}[{repr(key)}]")
            if diff:
                children[f"[{repr(key)}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    break

        if children:
            return CompoundDiff(source_type="dict", children=children, truncated=truncated)
        return None

    def _compare_object(self, val_a: Any, val_b: Any, path: str) -> Optional[DiffNode]:
        """
        Compare user-defined objects by recursively comparing their attributes.
        Handles both __dict__ and __slots__ based objects.
        """
        has_dict_a = hasattr(val_a, "__dict__")
        has_dict_b = hasattr(val_b, "__dict__")

        # Get __slots__ from the class, not the instance
        slots_a = getattr(type(val_a), "__slots__", None)
        slots_b = getattr(type(val_b), "__slots__", None)

        # If neither has __dict__ nor __slots__, try direct equality
        if not has_dict_a and not slots_a:
            try:
                if val_a != val_b:
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Object mismatch at {path}: {val_a} != {val_b}",
                    )
                return None
            except Exception:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Object comparison not supported at {path} (type: {type(val_a).__name__})",
                )

        children: Dict[str, DiffNode] = {}
        truncated = False
        diff_count = 0

        # Compare __dict__ attributes if available
        if has_dict_a and has_dict_b:
            dict_a = val_a.__dict__
            dict_b = val_b.__dict__

            keys_a = set(dict_a.keys())
            keys_b = set(dict_b.keys())

            # Check for attribute mismatches
            only_a = keys_a - keys_b
            only_b = keys_b - keys_a

            for key in only_a:
                children[f".{key}"] = ValueComparison(
                    status="different",
                    value1=dict_a[key],
                    value2=None,
                    message=f"Attribute {key} only in first object",
                )
                diff_count += 1

            for key in only_b:
                children[f".{key}"] = ValueComparison(
                    status="different",
                    value1=None,
                    value2=dict_b[key],
                    message=f"Attribute {key} only in second object",
                )
                diff_count += 1

            # Compare common attributes
            common_keys = keys_a & keys_b
            for key in sorted(common_keys, key=str):
                diff = self._compare_values(dict_a[key], dict_b[key], f"{path}.{key}")
                if diff:
                    children[f".{key}"] = diff
                    diff_count += 1
                    if diff_count >= self.max_diffs_per_container:
                        truncated = True
                        return CompoundDiff(source_type="object", children=children, truncated=truncated)
        elif has_dict_a != has_dict_b:
            # One has __dict__, the other doesn't (and no slots to fall back on)
            if not slots_a and not slots_b:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Object mismatch at {path}: __dict__ availability differs",
                )

        # Compare __slots__ attributes
        if slots_a or slots_b:
            # Normalize slots to tuples
            if slots_a is None:
                slots_a = ()
            elif isinstance(slots_a, str):
                slots_a = (slots_a,)
            else:
                slots_a = tuple(slots_a)

            if slots_b is None:
                slots_b = ()
            elif isinstance(slots_b, str):
                slots_b = (slots_b,)
            else:
                slots_b = tuple(slots_b)

            all_slots = set(slots_a) | set(slots_b)

            for slot in sorted(all_slots):
                has_a = hasattr(val_a, slot)
                has_b = hasattr(val_b, slot)

                if has_a and not has_b:
                    children[f".{slot}"] = ValueComparison(
                        status="different",
                        value1=getattr(val_a, slot),
                        value2=None,
                        message=f"Slot {slot} only in first object",
                    )
                    diff_count += 1
                elif has_b and not has_a:
                    children[f".{slot}"] = ValueComparison(
                        status="different",
                        value1=None,
                        value2=getattr(val_b, slot),
                        message=f"Slot {slot} only in second object",
                    )
                    diff_count += 1
                elif has_a and has_b:
                    diff = self._compare_values(
                        getattr(val_a, slot), getattr(val_b, slot), f"{path}.{slot}"
                    )
                    if diff:
                        children[f".{slot}"] = diff
                        diff_count += 1

                if diff_count >= self.max_diffs_per_container:
                    truncated = True
                    return CompoundDiff(source_type="object", children=children, truncated=truncated)

        if children:
            return CompoundDiff(source_type="object", children=children, truncated=truncated)
        return None


# Example usage
if __name__ == "__main__":
    # Create test namespaces
    import numpy as np
    import pandas as pd

    # Namespace A
    a = {}
    a["x"] = 42
    a["y"] = 3.14159
    a["z"] = np.array([1.0, 2.0, np.nan, 4.0])
    a["df"] = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, np.nan]})
    a["list_obj"] = [1, 2, 3]
    a["ref1"] = a["list_obj"]  # Create pointer reference

    # Namespace B (identical)
    b = {}
    b["x"] = 42
    b["y"] = 3.14159
    b["z"] = np.array([1.0, 2.0, np.nan, 4.0])
    b["df"] = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, np.nan]})
    b["list_obj"] = [1, 2, 3]
    b["ref1"] = b["list_obj"]  # Maintain pointer structure

    differ = Diff()
    differences = differ.diff(a, b)

    if differences:
        print("Differences found:")
        for var, msg in differences.items():
            print(f"  {var}: {msg}")
    else:
        print("Namespaces are equal!")

    # Test with differences
    b["x"] = 43  # Change value
    differences = differ.diff(a, b)
    print("\nAfter changing b['x']:")
    for var, msg in differences.items():
        print(f"  {var}: {msg}")
