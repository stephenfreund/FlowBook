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
    DiffNode,
    DiffResult,
    IndexComponent,
    KeyComponent,
    AttributeComponent,
    DataFrameLocation,
)

    


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
        max_diffs_per_container: int = 1000,
        sample_large_arrays: bool = True,
        strict: bool = True,
        report_close: bool = True
    ):
        """
        Initialize the Diff comparator.

        Args:
            rtol: Relative tolerance for floating point comparisons (default: 1e-5)
            atol: Absolute tolerance for floating point comparisons (default: 1e-8)
            max_diffs_per_container: Maximum differences to collect per container (default: 1000)
            sample_large_arrays: Whether to sample large arrays instead of full comparison (default: True)
            strict: If True, require exact type matches. If False, allow compatible types
                    (e.g., int vs float, list vs ndarray) (default: True)
            report_close: If True, report floats that are close (within tolerance) with status='close'.
                         If False, treat close values as equal and don't report them (default: True)

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
        self.sample_large_arrays = sample_large_arrays
        self.strict = strict
        self.report_close = report_close
        # Track object identities to ensure pointer structure matches
        self.id_map_a = {}  # Maps id(obj_a) -> canonical_id
        self.id_map_b = {}  # Maps id(obj_b) -> canonical_id
        self.next_canonical_id = 0
    
    def diff(self, a: Dict[str, Any], b: Dict[str, Any], keys_to_include: Set[str] | None = None) -> DiffResult:
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
                message="Variable was removed"
            )

        # Check for variables only in b
        only_in_b = set(b.keys()) - set(a.keys())
        for var in only_in_b & keys_to_include:
            differences[var] = ValueComparison(
                status="different",
                value1=None,
                value2=b[var],
                message="Variable was added"
            )

        # Compare common variables - only add to differences if not equal
        common_vars = set(a.keys()) & set(b.keys())
        for var in sorted(common_vars & keys_to_include):  # Sort for deterministic output
            diff_result = self._compare_values(a[var], b[var], path=var)
            if diff_result:  # Only include if there are differences
                differences[var] = diff_result

        return DiffResult(differences=differences)
    
    def _compare_values(self, val_a: Any, val_b: Any, path: str = "") -> Optional[DiffNode]:
        """
        Compare two values, dispatching to type-specific methods.
        Returns None if equal, otherwise returns DiffNode with differences.
        """
        # Check pointer structure
        id_a, id_b = id(val_a), id(val_b)

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
                        message=f"Pointer structure mismatch at {path}"
                    )
            else:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Pointer structure mismatch at {path} (first namespace has reference to earlier object)"
                )
            return None  # Already compared, and structure matches

        # If we've seen val_b before but not val_a
        if id_b in self.id_map_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Pointer structure mismatch at {path} (second namespace has reference to earlier object)"
            )

        # Register these objects with the same canonical ID
        # We do this before comparing to handle circular references
        canonical_id = self.next_canonical_id
        self.next_canonical_id += 1
        self.id_map_a[id_a] = canonical_id
        self.id_map_b[id_b] = canonical_id

        # Type checking
        if type(val_a) != type(val_b):
            # In non-strict mode, check if types are compatible
            if not self.strict:
                is_compatible, compat_type = self._types_compatible(val_a, val_b)
                if is_compatible:
                    # Use flexible comparison for compatible types
                    if compat_type == "numeric":
                        result = self._compare_numeric_flexible(val_a, val_b, path)
                    elif compat_type in ("list_array", "tuple_array"):
                        result = self._compare_list_array_flexible(val_a, val_b, path)
                    else:
                        # Should not happen, but handle gracefully
                        result = ValueComparison(
                            status="different",
                            value1=val_a,
                            value2=val_b,
                            message=f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}"
                        )

                    # If comparison found a difference, unregister these objects
                    if result is not None:
                        del self.id_map_a[id_a]
                        del self.id_map_b[id_b]

                    return result

            # Strict mode or incompatible types - unregister and return type mismatch
            del self.id_map_a[id_a]
            del self.id_map_b[id_b]
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}"
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
        elif callable(val_a):
            result = self._compare_callable(val_a, val_b, path)
        elif isinstance(val_a, np.ndarray):
            result = self._compare_ndarray(val_a, val_b, path)
        elif isinstance(val_a, (DataFrameGroupBy, SeriesGroupBy)):
            result = self._compare_groupby(val_a, val_b, path)
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
        if result is not None:
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

    def _compare_numeric_flexible(self, val_a: Any, val_b: Any, path: str) -> Optional[ValueComparison]:
        """
        Compare int vs float values in non-strict mode.
        Converts int to float and uses float comparison logic.
        """
        # Convert both to float for comparison
        float_a = float(val_a)
        float_b = float(val_b)

        return self._compare_float(float_a, float_b, path)

    def _compare_list_array_flexible(self, val_a: Any, val_b: Any, path: str) -> Optional[DiffNode]:
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
                message=f"Structure mismatch at {path}: {'list' if isinstance(lst, list) else 'tuple'} depth {list_depth} vs array ndim {array_ndim}"
            )

        # Convert list/tuple to array for shape comparison
        try:
            lst_as_array = np.array(lst)
        except (ValueError, TypeError) as e:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Cannot convert {'list' if isinstance(lst, list) else 'tuple'} to array at {path}: {str(e)}"
            )

        # Check shapes match
        if lst_as_array.shape != arr.shape:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Shape mismatch at {path}: {'list' if isinstance(lst, list) else 'tuple'} shape {lst_as_array.shape} vs array shape {arr.shape}"
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
                # Return first difference found
                idx = np.unravel_index(i, arr.shape)
                idx_tuple = tuple(int(x) for x in idx)
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Element mismatch at {path}[{idx_tuple}]: {lst_val} vs {arr_val}"
                )

        # All elements match
        return None

    def _compare_bool(self, val_a: bool, val_b: bool, path: str) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Bool mismatch at {path}: {val_a} vs {val_b}"
            )
        return None

    def _compare_int(self, val_a: int, val_b: int, path: str) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Integer mismatch at {path}: {val_a} vs {val_b}"
            )
        return None
    
    def _compare_float(self, val_a: float, val_b: float, path: str) -> Optional[ValueComparison]:
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
                message=f"Float mismatch at {path}: {val_a} vs {val_b} (one is NaN)"
            )

        # Check exact equality first
        if val_a == val_b:
            return None  # Exactly equal

        # Check if close within tolerance
        if math.isclose(val_a, val_b, rel_tol=self.rtol, abs_tol=self.atol):
            # If report_close is False, treat close values as equal (no difference)
            if not self.report_close:
                return None
            return ValueComparison(
                status="close",
                value1=val_a,
                value2=val_b,
                message=f"Float close at {path}: {val_a} vs {val_b} (within tolerance)"
            )

        # Not equal and not close
        return ValueComparison(
            status="different",
            value1=val_a,
            value2=val_b,
            message=f"Float mismatch at {path}: {val_a} vs {val_b}"
        )
    
    def _compare_complex(self, val_a: complex, val_b: complex, path: str) -> Optional[DiffNode]:
        diffs = {}
        real_diff = self._compare_float(val_a.real, val_b.real, f"{path}.real")
        if real_diff:
            diffs[".real"] = real_diff
        imag_diff = self._compare_float(val_a.imag, val_b.imag, f"{path}.imag")
        if imag_diff:
            diffs[".imag"] = imag_diff
        return diffs if diffs else None

    def _compare_str(self, val_a: str, val_b: str, path: str) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"String mismatch at {path}: '{val_a}' vs '{val_b}'"
            )
        return None

    def _compare_bytes(self, val_a: bytes, val_b: bytes, path: str) -> Optional[ValueComparison]:
        if val_a != val_b:
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Bytes mismatch at {path}"
            )
        return None
    
    def _compare_callable(self, val_a: Any, val_b: Any, path: str) -> Optional[DiffNode]:
        """
        Compare callables.
        - For functions: use identity (is)
        - For bound methods: compare __func__ and __self__
        """
        # Check if both are bound methods
        is_method_a = hasattr(val_a, '__self__') and hasattr(val_a, '__func__')
        is_method_b = hasattr(val_b, '__self__') and hasattr(val_b, '__func__')

        if is_method_a and is_method_b:
            # Both are bound methods - compare the underlying function and instance
            diffs = {}
            func_diff = self._compare_values(val_a.__func__, val_b.__func__, f"{path}.__func__")
            if func_diff:
                diffs[".__func__"] = func_diff

            self_diff = self._compare_values(val_a.__self__, val_b.__self__, f"{path}.__self__")
            if self_diff:
                diffs[".__self__"] = self_diff

            return diffs if diffs else None
        elif is_method_a != is_method_b:
            # One is a bound method, the other isn't
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Callable type mismatch at {path}: bound method vs function"
            )
        else:
            # Both are regular functions/callables - use identity
            if val_a is not val_b:
                name_a = getattr(val_a, '__name__', repr(val_a))
                name_b = getattr(val_b, '__name__', repr(val_b))
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Callable mismatch at {path}: {name_a} vs {name_b} (different objects)"
                )
            return None
    
    def _compare_ndarray(self, val_a: np.ndarray, val_b: np.ndarray, path: str) -> Optional[ValueComparison]:
        """Compare numpy arrays (TODO: implement full diff collection with sampling)."""
        result = self._compare_ndarray_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(status="different", value1=val_a, value2=val_b, message=result)
        return None

    def _compare_ndarray_legacy(self, val_a: np.ndarray, val_b: np.ndarray, path: str) -> str:
        # Check shape
        if val_a.shape != val_b.shape:
            return f"Array shape mismatch at {path}: {val_a.shape} vs {val_b.shape}"
        
        # Check dtype
        if val_a.dtype != val_b.dtype:
            return f"Array dtype mismatch at {path}: {val_a.dtype} vs {val_b.dtype}"
        
        # Compare values
        try:
            if np.issubdtype(val_a.dtype, np.floating) or np.issubdtype(val_a.dtype, np.complexfloating):
                # Use allclose for floating point, treating NaN as equal
                if not np.allclose(val_a, val_b, rtol=self.rtol, atol=self.atol, equal_nan=True):
                    # Find first mismatch
                    flat_a = val_a.ravel()
                    flat_b = val_b.ravel()
                    for i in range(len(flat_a)):
                        a_val, b_val = flat_a[i], flat_b[i]
                        both_nan = np.isnan(a_val) and np.isnan(b_val)
                        if not both_nan and not np.allclose([a_val], [b_val], rtol=self.rtol, atol=self.atol, equal_nan=True):
                            idx = np.unravel_index(i, val_a.shape)
                            idx_tuple = tuple(int(x) for x in idx)
                            return f"Array values mismatch at {path}[{idx_tuple}]: {a_val} vs {b_val}"
                    return f"Array values mismatch at {path}"
            else:
                # For other types, use array_equal
                if not np.array_equal(val_a, val_b):
                    # Find first mismatch
                    flat_a = val_a.ravel()
                    flat_b = val_b.ravel()
                    for i in range(len(flat_a)):
                        if flat_a[i] != flat_b[i]:
                            idx = np.unravel_index(i, val_a.shape)
                            idx_tuple = tuple(int(x) for x in idx)
                            return f"Array values mismatch at {path}[{idx_tuple}]: {flat_a[i]} vs {flat_b[i]}"
                    return f"Array values mismatch at {path}"
        except Exception as e:
            return f"Array comparison error at {path}: {str(e)}"
        
        return ""
    
    def _compare_series(self, val_a: pd.Series, val_b: pd.Series, path: str) -> Optional[ValueComparison]:
        """Compare pandas Series (TODO: implement full diff collection)."""
        result = self._compare_series_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(status="different", value1=val_a, value2=val_b, message=result)
        return None

    def _compare_series_legacy(self, val_a: pd.Series, val_b: pd.Series, path: str) -> str:
        # Check index
        if not val_a.index.equals(val_b.index):
            return f"Series index mismatch at {path}"
        
        # Check name
        if val_a.name != val_b.name:
            return f"Series name mismatch at {path}: {val_a.name} vs {val_b.name}"
        
        # Check dtype
        if val_a.dtype != val_b.dtype:
            return f"Series dtype mismatch at {path}: {val_a.dtype} vs {val_b.dtype}"
        
        # Compare values
        try:
            if pd.api.types.is_float_dtype(val_a.dtype):
                # For float dtypes, use allclose with NaN handling
                mask_nan_a = pd.isna(val_a)
                mask_nan_b = pd.isna(val_b)
                if not mask_nan_a.equals(mask_nan_b):
                    # Find first NaN position mismatch
                    for idx in val_a.index:
                        if mask_nan_a[idx] != mask_nan_b[idx]:
                            return f"Series NaN positions mismatch at {path}[{repr(idx)}]: is_nan={mask_nan_a[idx]} vs is_nan={mask_nan_b[idx]}"
                    return f"Series NaN positions mismatch at {path}"
                
                non_nan_a = val_a[~mask_nan_a]
                non_nan_b = val_b[~mask_nan_b]
                if len(non_nan_a) > 0:
                    if not np.allclose(non_nan_a, non_nan_b, rtol=self.rtol, atol=self.atol):
                        # Find first value mismatch
                        for idx in non_nan_a.index:
                            if not np.allclose([non_nan_a[idx]], [non_nan_b[idx]], rtol=self.rtol, atol=self.atol):
                                return f"Series values mismatch at {path}[{repr(idx)}]: {non_nan_a[idx]} vs {non_nan_b[idx]}"
                        return f"Series values mismatch at {path}"
            else:
                if not val_a.equals(val_b):
                    # Find first value mismatch
                    for idx in val_a.index:
                        if val_a[idx] != val_b[idx]:
                            return f"Series values mismatch at {path}[{repr(idx)}]: {val_a[idx]} vs {val_b[idx]}"
                    return f"Series values mismatch at {path}"
        except Exception as e:
            return f"Series comparison error at {path}: {str(e)}"
        
        return ""
    
    def _compare_dataframe(self, val_a: pd.DataFrame, val_b: pd.DataFrame, path: str) -> Optional[ValueComparison]:
        """Compare pandas DataFrames (TODO: implement full diff collection with sampling)."""
        result = self._compare_dataframe_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(status="different", value1=val_a, value2=val_b, message=result)
        return None

    def _compare_dataframe_legacy(self, val_a: pd.DataFrame, val_b: pd.DataFrame, path: str) -> str:
        # Check shape
        if val_a.shape != val_b.shape:
            return f"DataFrame shape mismatch at {path}: {val_a.shape} vs {val_b.shape}"

        # Check columns
        if not val_a.columns.equals(val_b.columns):
            return f"DataFrame columns mismatch at {path}"

        # Check index
        if not val_a.index.equals(val_b.index):
            return f"DataFrame index mismatch at {path}"

        # Compare each column
        for col in val_a.columns:
            col_diff = self._compare_series(val_a[col], val_b[col], f"{path}['{col}']")
            if col_diff:
                # _compare_series returns ValueComparison now, extract message for legacy
                if isinstance(col_diff, ValueComparison):
                    return col_diff.message
                return col_diff

        return ""

    def _compare_groupby(self, val_a, val_b, path: str) -> Optional[ValueComparison]:
        """Compare pandas GroupBy objects (TODO: implement full diff collection)."""
        result = self._compare_groupby_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(status="different", value1=val_a, value2=val_b, message=result)
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
        if not hasattr(val_a, 'obj') or not hasattr(val_b, 'obj'):
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
        if not hasattr(val_a, '_grouper') or not hasattr(val_b, '_grouper'):
            return f"GroupBy structure mismatch at {path}: missing _grouper attribute"

        grouper_diff = self._compare_grouper(val_a._grouper, val_b._grouper, f"{path}._grouper")
        if grouper_diff:
            # _compare_grouper returns ValueComparison now, but we need string for legacy
            if isinstance(grouper_diff, ValueComparison):
                return grouper_diff.message
            return grouper_diff

        # Compare selection (which columns are selected)
        if hasattr(val_a, '_selection') and hasattr(val_b, '_selection'):
            if val_a._selection != val_b._selection:
                return f"GroupBy selection mismatch at {path}: {val_a._selection} vs {val_b._selection}"

        return ""

    def _compare_grouper(self, val_a: BaseGrouper, val_b: BaseGrouper, path: str) -> Optional[ValueComparison]:
        """Compare pandas BaseGrouper objects (legacy wrapper)."""
        result = self._compare_grouper_legacy(val_a, val_b, path)
        if result:
            return ValueComparison(status="different", value1=val_a, value2=val_b, message=result)
        return None

    def _compare_grouper_legacy(self, val_a: BaseGrouper, val_b: BaseGrouper, path: str) -> str:
        """
        Compare BaseGrouper objects, excluding their internal cache.

        Compares the semantic properties that define the grouping:
        - groupings (keys)
        - sort flag
        - dropna flag
        - axis
        """
        # Compare axis (typically a pandas Index)
        if hasattr(val_a, 'axis') and hasattr(val_b, 'axis'):
            # Use pandas Index.equals() for proper comparison
            axes_equal = False
            if isinstance(val_a.axis, pd.Index) and isinstance(val_b.axis, pd.Index):
                axes_equal = val_a.axis.equals(val_b.axis)
            elif isinstance(val_a.axis, np.ndarray) and isinstance(val_b.axis, np.ndarray):
                axes_equal = np.array_equal(val_a.axis, val_b.axis)
            else:
                axes_equal = (val_a.axis == val_b.axis)

            if not axes_equal:
                return f"Grouper axis mismatch at {path}: {val_a.axis} vs {val_b.axis}"

        # Compare sort flag
        if hasattr(val_a, '_sort') and hasattr(val_b, '_sort'):
            if val_a._sort != val_b._sort:
                return f"Grouper sort mismatch at {path}: {val_a._sort} vs {val_b._sort}"

        # Compare dropna flag
        if hasattr(val_a, 'dropna') and hasattr(val_b, 'dropna'):
            if val_a.dropna != val_b.dropna:
                return f"Grouper dropna mismatch at {path}: {val_a.dropna} vs {val_b.dropna}"

        # Compare groupings (the actual grouping keys)
        if hasattr(val_a, '_groupings') and hasattr(val_b, '_groupings'):
            groupings_a = val_a._groupings
            groupings_b = val_b._groupings

            if len(groupings_a) != len(groupings_b):
                return f"Grouper groupings count mismatch at {path}: {len(groupings_a)} vs {len(groupings_b)}"

            for i, (grp_a, grp_b) in enumerate(zip(groupings_a, groupings_b)):
                # Compare key/name
                if hasattr(grp_a, 'name') and hasattr(grp_b, 'name'):
                    if grp_a.name != grp_b.name:
                        return f"Grouping name mismatch at {path}._groupings[{i}]: {grp_a.name} vs {grp_b.name}"

                # Compare key object if available
                if hasattr(grp_a, 'key') and hasattr(grp_b, 'key'):
                    if grp_a.key != grp_b.key:
                        return f"Grouping key mismatch at {path}._groupings[{i}]: {grp_a.key} vs {grp_b.key}"

        return ""

    def _compare_list(self, val_a: list, val_b: list, path: str) -> Optional[DiffNode]:
        diffs = {}

        # Check length mismatch
        if len(val_a) != len(val_b):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"List length mismatch at {path}: {len(val_a)} vs {len(val_b)}"
            )

        # Compare each element, collecting ALL differences
        diff_count = 0
        for i, (item_a, item_b) in enumerate(zip(val_a, val_b)):
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                diffs[f"[{i}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    diffs["_truncated"] = ValueComparison(
                        status="different",
                        value1=None,
                        value2=None,
                        message=f"Truncated after {diff_count} differences (max_diffs_per_container={self.max_diffs_per_container})"
                    )
                    break

        return diffs if diffs else None
    
    def _compare_tuple(self, val_a: tuple, val_b: tuple, path: str) -> Optional[DiffNode]:
        diffs = {}

        # Check length mismatch
        if len(val_a) != len(val_b):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Tuple length mismatch at {path}: {len(val_a)} vs {len(val_b)}"
            )

        # Compare each element, collecting ALL differences
        diff_count = 0
        for i, (item_a, item_b) in enumerate(zip(val_a, val_b)):
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                diffs[f"[{i}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    diffs["_truncated"] = ValueComparison(
                        status="different",
                        value1=None,
                        value2=None,
                        message=f"Truncated after {diff_count} differences (max_diffs_per_container={self.max_diffs_per_container})"
                    )
                    break

        return diffs if diffs else None
    
    def _compare_set(self, val_a: set, val_b: set, path: str) -> Optional[ValueComparison]:
        """
        Compare sets by finding matching elements and comparing them recursively.
        This properly handles pointer structure within sets.
        """
        if len(val_a) != len(val_b):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Set size mismatch at {path}: {len(val_a)} vs {len(val_b)}"
            )

        # Convert to lists for matching
        list_a = list(val_a)
        list_b = list(val_b)

        # Try to find a matching between elements
        used_b = set()

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
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Set contents mismatch at {path}: element {repr(item_a)} has no matching element in second set"
                )

        return None
    
    def _compare_frozenset(self, val_a: frozenset, val_b: frozenset, path: str) -> Optional[ValueComparison]:
        """Compare frozensets using the same recursive approach as sets."""
        if len(val_a) != len(val_b):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Frozenset size mismatch at {path}: {len(val_a)} vs {len(val_b)}"
            )

        # Convert to lists for matching
        list_a = list(val_a)
        list_b = list(val_b)

        # Try to find a matching between elements
        used_b = set()

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
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Frozenset contents mismatch at {path}: element {repr(item_a)} has no matching element in second set"
                )

        return None
    
    def _compare_dict(self, val_a: dict, val_b: dict, path: str) -> Optional[DiffNode]:
        diffs = {}
        keys_a = set(val_a.keys())
        keys_b = set(val_b.keys())

        # Check for key mismatches
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a

        for key in only_a:
            diffs[f"[{repr(key)}]"] = ValueComparison(
                status="different",
                value1=val_a[key],
                value2=None,
                message=f"Key {repr(key)} only in first dict"
            )

        for key in only_b:
            diffs[f"[{repr(key)}]"] = ValueComparison(
                status="different",
                value1=None,
                value2=val_b[key],
                message=f"Key {repr(key)} only in second dict"
            )

        # Compare common keys, collecting ALL differences
        common_keys = keys_a & keys_b
        diff_count = len(diffs)  # Count keys already different
        for key in sorted(common_keys, key=str):  # Sort for deterministic output
            diff = self._compare_values(val_a[key], val_b[key], f"{path}[{repr(key)}]")
            if diff:
                diffs[f"[{repr(key)}]"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    diffs["_truncated"] = ValueComparison(
                        status="different",
                        value1=None,
                        value2=None,
                        message=f"Truncated after {diff_count} differences (max_diffs_per_container={self.max_diffs_per_container})"
                    )
                    break

        return diffs if diffs else None
    
    def _compare_object(self, val_a: Any, val_b: Any, path: str) -> Optional[DiffNode]:
        """
        Compare user-defined objects by recursively comparing their __dict__.
        """
        # Check if objects have __dict__
        if not hasattr(val_a, '__dict__'):
            # Try direct equality
            try:
                if val_a != val_b:
                    return ValueComparison(
                        status="different",
                        value1=val_a,
                        value2=val_b,
                        message=f"Object mismatch at {path}: {val_a} != {val_b}"
                    )
                return None
            except:
                return ValueComparison(
                    status="different",
                    value1=val_a,
                    value2=val_b,
                    message=f"Object comparison not supported at {path} (type: {type(val_a).__name__})"
                )

        if not hasattr(val_b, '__dict__'):
            return ValueComparison(
                status="different",
                value1=val_a,
                value2=val_b,
                message=f"Object mismatch at {path}: first has __dict__, second does not"
            )

        # Recursively compare __dict__ attributes
        diffs = {}
        dict_a = val_a.__dict__
        dict_b = val_b.__dict__

        keys_a = set(dict_a.keys())
        keys_b = set(dict_b.keys())

        # Check for attribute mismatches
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a

        for key in only_a:
            diffs[f".{key}"] = ValueComparison(
                status="different",
                value1=dict_a[key],
                value2=None,
                message=f"Attribute {key} only in first object"
            )

        for key in only_b:
            diffs[f".{key}"] = ValueComparison(
                status="different",
                value1=None,
                value2=dict_b[key],
                message=f"Attribute {key} only in second object"
            )

        # Compare common attributes, collecting ALL differences
        common_keys = keys_a & keys_b
        diff_count = len(diffs)
        for key in sorted(common_keys, key=str):
            diff = self._compare_values(dict_a[key], dict_b[key], f"{path}.{key}")
            if diff:
                diffs[f".{key}"] = diff
                diff_count += 1
                # Stop if we hit the limit
                if diff_count >= self.max_diffs_per_container:
                    diffs["_truncated"] = ValueComparison(
                        status="different",
                        value1=None,
                        value2=None,
                        message=f"Truncated after {diff_count} differences (max_diffs_per_container={self.max_diffs_per_container})"
                    )
                    break

        return diffs if diffs else None


# Example usage
if __name__ == "__main__":
    # Create test namespaces
    import numpy as np
    import pandas as pd
    
    # Namespace A
    a = {}
    a['x'] = 42
    a['y'] = 3.14159
    a['z'] = np.array([1.0, 2.0, np.nan, 4.0])
    a['df'] = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, np.nan]})
    a['list_obj'] = [1, 2, 3]
    a['ref1'] = a['list_obj']  # Create pointer reference
    
    # Namespace B (identical)
    b = {}
    b['x'] = 42
    b['y'] = 3.14159
    b['z'] = np.array([1.0, 2.0, np.nan, 4.0])
    b['df'] = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, np.nan]})
    b['list_obj'] = [1, 2, 3]
    b['ref1'] = b['list_obj']  # Maintain pointer structure
    
    differ = Diff()
    differences = differ.diff(a, b)
    
    if differences:
        print("Differences found:")
        for var, msg in differences.items():
            print(f"  {var}: {msg}")
    else:
        print("Namespaces are equal!")
    
    # Test with differences
    b['x'] = 43  # Change value
    differences = differ.diff(a, b)
    print("\nAfter changing b['x']:")
    for var, msg in differences.items():
        print(f"  {var}: {msg}")