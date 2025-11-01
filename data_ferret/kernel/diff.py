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


class Diff:
    """
    Compare two Jupyter kernel user namespaces for equality.
    Checks value equality and isomorphic pointer structure.
    """
    
    def __init__(self, rtol=1e-5, atol=1e-5):
        """
        Initialize the Diff comparator.
        
        Args:
            rtol: Relative tolerance for floating point comparisons (default: 1e-5)
            atol: Absolute tolerance for floating point comparisons (default: 1e-5)
        """
        self.rtol = rtol
        self.atol = atol
        # Track object identities to ensure pointer structure matches
        self.id_map_a = {}  # Maps id(obj_a) -> canonical_id
        self.id_map_b = {}  # Maps id(obj_b) -> canonical_id
        self.next_canonical_id = 0
    
    def diff(self, a: Dict[str, Any], b: Dict[str, Any], keys_to_include: Set[str] | None = None) -> Dict[str, str]:
        """
        Compare two user namespaces.
        
        Args:
            a: First namespace dictionary
            b: Second namespace dictionary
            
        Returns:
            Dictionary mapping variable names to difference descriptions.
            Empty dict if namespaces are equal.
        """
        if keys_to_include is None:
            keys_to_include = set(a.keys()) | set(b.keys())

        # Reset identity tracking for each comparison
        self.id_map_a = {}
        self.id_map_b = {}
        self.next_canonical_id = 0
        
        differences = {}
        
        # Check for variables only in a
        only_in_a = set(a.keys()) - set(b.keys())
        for var in only_in_a & keys_to_include:
            differences[var] = f"Variable was removed"
        
        # Check for variables only in b
        only_in_b = set(b.keys()) - set(a.keys())
        for var in only_in_b & keys_to_include:
            differences[var] = f"Variable was added"
        
        # Compare common variables
        common_vars = set(a.keys()) & set(b.keys())
        for var in sorted(common_vars & keys_to_include):  # Sort for deterministic output
            diff_msg = self._compare_values(a[var], b[var], path=var)
            if diff_msg:
                differences[var] = diff_msg
        
        return differences
    
    def _compare_values(self, val_a: Any, val_b: Any, path: str = "") -> str:
        """
        Compare two values, dispatching to type-specific methods.
        Returns empty string if equal, otherwise returns difference description.
        """
        # Check pointer structure
        id_a, id_b = id(val_a), id(val_b)
        
        # If we've seen val_a before, check if pointer structure matches
        if id_a in self.id_map_a:
            canonical_a = self.id_map_a[id_a]
            if id_b in self.id_map_b:
                canonical_b = self.id_map_b[id_b]
                if canonical_a != canonical_b:
                    return f"Pointer structure mismatch at {path}"
            else:
                return f"Pointer structure mismatch at {path} (first namespace has reference to earlier object)"
            return ""  # Already compared, and structure matches
        
        # If we've seen val_b before but not val_a
        if id_b in self.id_map_b:
            return f"Pointer structure mismatch at {path} (second namespace has reference to earlier object)"
        
        # Register these objects with the same canonical ID
        # We do this before comparing to handle circular references
        canonical_id = self.next_canonical_id
        self.next_canonical_id += 1
        self.id_map_a[id_a] = canonical_id
        self.id_map_b[id_b] = canonical_id
        
        # Type checking
        if type(val_a) != type(val_b):
            # Unregister since they're not equal
            del self.id_map_a[id_a]
            del self.id_map_b[id_b]
            return f"Type mismatch at {path}: {type(val_a).__name__} vs {type(val_b).__name__}"
        
        # Dispatch to type-specific methods
        result = ""
        if val_a is None:
            result = ""
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
        if result:
            del self.id_map_a[id_a]
            del self.id_map_b[id_b]
        
        return result
    
    def _compare_bool(self, val_a: bool, val_b: bool, path: str) -> str:
        if val_a != val_b:
            return f"Bool mismatch at {path}: {val_a} vs {val_b}"
        return ""
    
    def _compare_int(self, val_a: int, val_b: int, path: str) -> str:
        if val_a != val_b:
            return f"Integer mismatch at {path}: {val_a} vs {val_b}"
        return ""
    
    def _compare_float(self, val_a: float, val_b: float, path: str) -> str:
        # Handle NaN
        is_nan_a = math.isnan(val_a) if isinstance(val_a, float) else np.isnan(val_a)
        is_nan_b = math.isnan(val_b) if isinstance(val_b, float) else np.isnan(val_b)
        
        if is_nan_a and is_nan_b:
            return ""  # Both NaN, considered equal
        if is_nan_a or is_nan_b:
            return f"Float mismatch at {path}: {val_a} vs {val_b} (one is NaN)"
        
        # Use isclose for comparison
        if not math.isclose(val_a, val_b, rel_tol=self.rtol, abs_tol=self.atol):
            return f"Float mismatch at {path}: {val_a} vs {val_b}"
        return ""
    
    def _compare_complex(self, val_a: complex, val_b: complex, path: str) -> str:
        real_diff = self._compare_float(val_a.real, val_b.real, f"{path}.real")
        if real_diff:
            return real_diff
        imag_diff = self._compare_float(val_a.imag, val_b.imag, f"{path}.imag")
        if imag_diff:
            return imag_diff
        return ""
    
    def _compare_str(self, val_a: str, val_b: str, path: str) -> str:
        if val_a != val_b:
            return f"String mismatch at {path}: '{val_a}' vs '{val_b}'"
        return ""
    
    def _compare_bytes(self, val_a: bytes, val_b: bytes, path: str) -> str:
        if val_a != val_b:
            return f"Bytes mismatch at {path}"
        return ""
    
    def _compare_callable(self, val_a: Any, val_b: Any, path: str) -> str:
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
            func_diff = self._compare_values(val_a.__func__, val_b.__func__, f"{path}.__func__")
            if func_diff:
                return func_diff
            
            self_diff = self._compare_values(val_a.__self__, val_b.__self__, f"{path}.__self__")
            if self_diff:
                return self_diff
            
            return ""
        elif is_method_a != is_method_b:
            # One is a bound method, the other isn't
            return f"Callable type mismatch at {path}: bound method vs function"
        else:
            # Both are regular functions/callables - use identity
            if val_a is not val_b:
                name_a = getattr(val_a, '__name__', repr(val_a))
                name_b = getattr(val_b, '__name__', repr(val_b))
                return f"Callable mismatch at {path}: {name_a} vs {name_b} (different objects)"
            return ""
    
    def _compare_ndarray(self, val_a: np.ndarray, val_b: np.ndarray, path: str) -> str:
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
    
    def _compare_series(self, val_a: pd.Series, val_b: pd.Series, path: str) -> str:
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
    
    def _compare_dataframe(self, val_a: pd.DataFrame, val_b: pd.DataFrame, path: str) -> str:
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
                return col_diff

        return ""

    def _compare_groupby(self, val_a, val_b, path: str) -> str:
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
            return obj_diff

        # Compare the grouper (excluding cache)
        if not hasattr(val_a, '_grouper') or not hasattr(val_b, '_grouper'):
            return f"GroupBy structure mismatch at {path}: missing _grouper attribute"

        grouper_diff = self._compare_grouper(val_a._grouper, val_b._grouper, f"{path}._grouper")
        if grouper_diff:
            return grouper_diff

        # Compare selection (which columns are selected)
        if hasattr(val_a, '_selection') and hasattr(val_b, '_selection'):
            if val_a._selection != val_b._selection:
                return f"GroupBy selection mismatch at {path}: {val_a._selection} vs {val_b._selection}"

        return ""

    def _compare_grouper(self, val_a: BaseGrouper, val_b: BaseGrouper, path: str) -> str:
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

    def _compare_list(self, val_a: list, val_b: list, path: str) -> str:
        if len(val_a) != len(val_b):
            return f"List length mismatch at {path}: {len(val_a)} vs {len(val_b)}"
        
        for i, (item_a, item_b) in enumerate(zip(val_a, val_b)):
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                return diff
        
        return ""
    
    def _compare_tuple(self, val_a: tuple, val_b: tuple, path: str) -> str:
        if len(val_a) != len(val_b):
            return f"Tuple length mismatch at {path}: {len(val_a)} vs {len(val_b)}"
        
        for i, (item_a, item_b) in enumerate(zip(val_a, val_b)):
            diff = self._compare_values(item_a, item_b, f"{path}[{i}]")
            if diff:
                return diff
        
        return ""
    
    def _compare_set(self, val_a: set, val_b: set, path: str) -> str:
        """
        Compare sets by finding matching elements and comparing them recursively.
        This properly handles pointer structure within sets.
        """
        if len(val_a) != len(val_b):
            return f"Set size mismatch at {path}: {len(val_a)} vs {len(val_b)}"
        
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
                return f"Set contents mismatch at {path}: element {repr(item_a)} has no matching element in second set"
        
        return ""
    
    def _compare_frozenset(self, val_a: frozenset, val_b: frozenset, path: str) -> str:
        """Compare frozensets using the same recursive approach as sets."""
        if len(val_a) != len(val_b):
            return f"Frozenset size mismatch at {path}: {len(val_a)} vs {len(val_b)}"
        
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
                return f"Frozenset contents mismatch at {path}: element {repr(item_a)} has no matching element in second set"
        
        return ""
    
    def _compare_dict(self, val_a: dict, val_b: dict, path: str) -> str:
        keys_a = set(val_a.keys())
        keys_b = set(val_b.keys())
        
        if keys_a != keys_b:
            only_a = keys_a - keys_b
            only_b = keys_b - keys_a
            msg_parts = []
            if only_a:
                msg_parts.append(f"keys only in first: {only_a}")
            if only_b:
                msg_parts.append(f"keys only in second: {only_b}")
            return f"Dict keys mismatch at {path}: {', '.join(msg_parts)}"
        
        for key in sorted(keys_a, key=str):  # Sort for deterministic output
            diff = self._compare_values(val_a[key], val_b[key], f"{path}[{repr(key)}]")
            if diff:
                return diff
        
        return ""
    
    def _compare_object(self, val_a: Any, val_b: Any, path: str) -> str:
        """
        Compare user-defined objects by recursively comparing their __dict__.
        """
        # Check if objects have __dict__
        if not hasattr(val_a, '__dict__'):
            # Try direct equality
            try:
                if val_a != val_b:
                    return f"Object mismatch at {path}: {val_a} != {val_b}"
                return ""
            except:
                return f"Object comparison not supported at {path} (type: {type(val_a).__name__})"
        
        if not hasattr(val_b, '__dict__'):
            return f"Object mismatch at {path}: first has __dict__, second does not"
        
        # Recursively compare __dict__ attributes
        dict_a = val_a.__dict__
        dict_b = val_b.__dict__
        
        keys_a = set(dict_a.keys())
        keys_b = set(dict_b.keys())
        
        if keys_a != keys_b:
            only_a = keys_a - keys_b
            only_b = keys_b - keys_a
            msg_parts = []
            if only_a:
                msg_parts.append(f"attributes only in first: {only_a}")
            if only_b:
                msg_parts.append(f"attributes only in second: {only_b}")
            return f"Object attributes mismatch at {path}: {', '.join(msg_parts)}"
        
        # Compare each attribute recursively
        for key in sorted(keys_a, key=str):
            diff = self._compare_values(dict_a[key], dict_b[key], f"{path}.{key}")
            if diff:
                return diff
        
        return ""


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