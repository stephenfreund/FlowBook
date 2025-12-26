"""
Column-level dependency tracking for DataFrames.

This module provides tracking of which DataFrame columns are read and written
during cell execution, using monkey-patching of pandas DataFrame methods.

The tracking works by:
1. Patching DataFrame methods (__getitem__, __setitem__, loc, iloc, etc.) before execution
2. Recording column access by object ID during execution
3. Resolving object IDs to variable paths after execution
4. Restoring original methods

This handles DataFrames at any nesting level: top-level, in dicts, lists, or object attributes.
"""

import types
import pandas as pd
from typing import Dict, Set, Iterable, Tuple, Optional, Generator, Any
from collections import defaultdict

from data_ferret.util.output import log, error


class ColumnAccessTracker:
    """Tracks DataFrame column access via monkey-patching."""

    # Class-level flag to suspend tracking globally (for deepcopy operations)
    _suspended = False

    # Class-level tracking of patch state to prevent double-patching across instances
    _patches_installed = False
    _class_original_methods: Dict[str, Any] = {}

    def __init__(self):
        self._reads_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._writes_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._id_to_path: Dict[int, str] = {}
        self._original_methods: Dict[str, Any] = {}
        self._installed = False
        # Mapping from GroupBy object id -> source DataFrame id
        # Used for cudf compatibility where gb.obj may not be accessible
        self._groupby_to_df: Dict[int, int] = {}

    def install(self) -> None:
        """Monkey-patch DataFrame methods to track column access."""
        if self._installed:
            return
        # Use class-level check to prevent double-patching across instances
        if not ColumnAccessTracker._patches_installed:
            self._patch_dataframe_methods()
            ColumnAccessTracker._patches_installed = True
        else:
            # Patches already installed by another instance - just copy the originals
            self._original_methods = ColumnAccessTracker._class_original_methods.copy()
        self._installed = True

        # Install cudf tracking if available (all cudf logic in cudf_compat)
        from . import cudf_compat
        cudf_compat.install_cudf_tracking(self)

    def uninstall(self) -> None:
        """Restore original DataFrame methods."""
        if not self._installed:
            return

        # Uninstall cudf tracking if available (all cudf logic in cudf_compat)
        from . import cudf_compat
        cudf_compat.uninstall_cudf_tracking()

        # Only restore if we have the original methods stored
        if self._original_methods:
            self._restore_dataframe_methods()
            ColumnAccessTracker._patches_installed = False
            ColumnAccessTracker._class_original_methods.clear()
        self._installed = False

    def register_df(self, df: pd.DataFrame, path: str) -> None:
        """Register a DataFrame with its namespace path."""
        self._id_to_path[id(df)] = path

    def record_read(self, df_id: int, columns: Iterable[str]) -> None:
        """Record column reads for a DataFrame by ID."""
        if ColumnAccessTracker._suspended:
            return
        for col in columns:
            # Only record as RBW if not already written
            if col not in self._writes_by_id[df_id]:
                self._reads_by_id[df_id].add(col)
                # DEBUG: Uncomment to trace where column reads come from
                # import traceback
                # print(f"DEBUG: record_read df_id={df_id} col={col}")
                # traceback.print_stack(limit=10)

    def record_write(self, df_id: int, columns: Iterable[str]) -> None:
        """Record column writes for a DataFrame by ID."""
        if ColumnAccessTracker._suspended:
            return
        for col in columns:
            self._writes_by_id[df_id].add(col)

    def resolve_to_paths(self) -> Dict[str, Set[str]]:
        """Convert id-based tracking to path-based column_rbw.

        Returns a dict mapping variable paths to sets of columns that were
        read-before-write. Includes entries for DataFrames that had column
        writes but no reads (with empty sets), so callers can distinguish
        "write-only DataFrame" from "untracked DataFrame".
        """
        result: Dict[str, Set[str]] = {}

        # Get all DataFrame IDs that had any column activity
        all_df_ids = set(self._reads_by_id.keys()) | set(self._writes_by_id.keys())

        for df_id in all_df_ids:
            if df_id not in self._id_to_path:
                continue

            path = self._id_to_path[df_id]
            read_cols = self._reads_by_id.get(df_id, set())

            # record_read() already ensures only reads-before-writes are recorded
            # (it checks if column was already written before adding to reads)
            # So _reads_by_id already contains the correct RBW set - no subtraction needed
            result[path] = read_cols.copy()

        return result

    def resolve_writes_to_paths(self) -> Dict[str, Set[str]]:
        """Convert id-based tracking to path-based column writes.

        Returns a dict mapping variable paths to sets of columns that were
        written during cell execution.
        """
        result: Dict[str, Set[str]] = {}

        for df_id, written_cols in self._writes_by_id.items():
            if df_id not in self._id_to_path:
                continue
            if written_cols:  # Only include if there were actual writes
                path = self._id_to_path[df_id]
                result[path] = written_cols.copy()

        return result

    def reset(self) -> None:
        """Reset tracking for new cell execution."""
        self._reads_by_id.clear()
        self._writes_by_id.clear()
        self._id_to_path.clear()
        self._groupby_to_df.clear()

        # Reset cudf tracking state (all cudf logic in cudf_compat)
        from . import cudf_compat
        cudf_compat.reset_cudf_tracking()

    def _patch_dataframe_methods(self) -> None:
        """Apply monkey-patches to DataFrame and related classes."""
        tracker = self

        # ========== DataFrame.__getitem__ ==========
        self._original_methods['DataFrame.__getitem__'] = pd.DataFrame.__getitem__
        original_df_getitem = self._original_methods['DataFrame.__getitem__']

        def tracked_df_getitem(df: pd.DataFrame, key):
            # Track column access
            if isinstance(key, str):
                tracker.record_read(id(df), [key])
            elif isinstance(key, list):
                str_keys = [k for k in key if isinstance(k, str)]
                if str_keys:
                    tracker.record_read(id(df), str_keys)
            elif isinstance(key, pd.Index):
                str_keys = [k for k in key if isinstance(k, str)]
                if str_keys:
                    tracker.record_read(id(df), str_keys)
            return original_df_getitem(df, key)

        pd.DataFrame.__getitem__ = tracked_df_getitem

        # ========== DataFrame.__setitem__ ==========
        self._original_methods['DataFrame.__setitem__'] = pd.DataFrame.__setitem__
        original_df_setitem = self._original_methods['DataFrame.__setitem__']

        def tracked_df_setitem(df: pd.DataFrame, key, value):
            # Track column writes
            if isinstance(key, str):
                tracker.record_write(id(df), [key])
            elif isinstance(key, list):
                str_keys = [k for k in key if isinstance(k, str)]
                if str_keys:
                    tracker.record_write(id(df), str_keys)
            return original_df_setitem(df, key, value)

        pd.DataFrame.__setitem__ = tracked_df_setitem

        # ========== DataFrame.assign ==========
        # Note: assign() returns a NEW DataFrame, it does NOT modify the original.
        # So we don't record any writes to the original DataFrame.
        self._original_methods['DataFrame.assign'] = pd.DataFrame.assign
        original_assign = self._original_methods['DataFrame.assign']

        def tracked_assign(df: pd.DataFrame, **kwargs):
            # assign() returns a new DataFrame - no writes to original
            return original_assign(df, **kwargs)

        pd.DataFrame.assign = tracked_assign

        # ========== DataFrame.drop ==========
        self._original_methods['DataFrame.drop'] = pd.DataFrame.drop
        original_drop = self._original_methods['DataFrame.drop']

        def tracked_drop(df: pd.DataFrame, labels=None, *, axis=0, index=None,
                         columns=None, level=None, inplace=False, errors='raise'):
            # Track column drops as reads (need to know what columns exist)
            if columns is not None:
                cols = [columns] if isinstance(columns, str) else list(columns)
                tracker.record_read(id(df), cols)
            return original_drop(df, labels=labels, axis=axis, index=index,
                                 columns=columns, level=level, inplace=inplace, errors=errors)

        pd.DataFrame.drop = tracked_drop

        # ========== DataFrame.groupby ==========
        self._original_methods['DataFrame.groupby'] = pd.DataFrame.groupby
        original_groupby = self._original_methods['DataFrame.groupby']

        def tracked_groupby(df: pd.DataFrame, by=None, *args, **kwargs):
            # Track groupby columns as reads
            if by is not None:
                if isinstance(by, str):
                    tracker.record_read(id(df), [by])
                elif isinstance(by, list):
                    str_keys = [k for k in by if isinstance(k, str)]
                    if str_keys:
                        tracker.record_read(id(df), str_keys)
            result = original_groupby(df, by=by, *args, **kwargs)
            # Store mapping from GroupBy -> DataFrame for cudf compatibility
            tracker._groupby_to_df[id(result)] = id(df)
            return result

        pd.DataFrame.groupby = tracked_groupby

        # ========== _LocIndexer.__getitem__ ==========
        try:
            from pandas.core.indexing import _LocIndexer
            self._original_methods['_LocIndexer.__getitem__'] = _LocIndexer.__getitem__
            original_loc_getitem = self._original_methods['_LocIndexer.__getitem__']

            def tracked_loc_getitem(loc_indexer, key):
                # Extract DataFrame from the indexer
                df = loc_indexer.obj
                if isinstance(df, pd.DataFrame):
                    columns = _extract_columns_from_loc_key(key, df)
                    if columns:
                        tracker.record_read(id(df), columns)
                return original_loc_getitem(loc_indexer, key)

            _LocIndexer.__getitem__ = tracked_loc_getitem
        except (ImportError, AttributeError):
            pass  # pandas version doesn't have this

        # ========== _LocIndexer.__setitem__ ==========
        try:
            from pandas.core.indexing import _LocIndexer
            self._original_methods['_LocIndexer.__setitem__'] = _LocIndexer.__setitem__
            original_loc_setitem = self._original_methods['_LocIndexer.__setitem__']

            def tracked_loc_setitem(loc_indexer, key, value):
                df = loc_indexer.obj
                if isinstance(df, pd.DataFrame):
                    columns = _extract_columns_from_loc_key(key, df)
                    if columns:
                        tracker.record_write(id(df), columns)
                return original_loc_setitem(loc_indexer, key, value)

            _LocIndexer.__setitem__ = tracked_loc_setitem
        except (ImportError, AttributeError):
            pass

        # ========== _iLocIndexer.__getitem__ ==========
        try:
            from pandas.core.indexing import _iLocIndexer
            self._original_methods['_iLocIndexer.__getitem__'] = _iLocIndexer.__getitem__
            original_iloc_getitem = self._original_methods['_iLocIndexer.__getitem__']

            def tracked_iloc_getitem(iloc_indexer, key):
                df = iloc_indexer.obj
                if isinstance(df, pd.DataFrame):
                    columns = _extract_columns_from_iloc_key(key, df)
                    if columns:
                        tracker.record_read(id(df), columns)
                return original_iloc_getitem(iloc_indexer, key)

            _iLocIndexer.__getitem__ = tracked_iloc_getitem
        except (ImportError, AttributeError):
            pass

        # ========== _iLocIndexer.__setitem__ ==========
        try:
            from pandas.core.indexing import _iLocIndexer
            self._original_methods['_iLocIndexer.__setitem__'] = _iLocIndexer.__setitem__
            original_iloc_setitem = self._original_methods['_iLocIndexer.__setitem__']

            def tracked_iloc_setitem(iloc_indexer, key, value):
                df = iloc_indexer.obj
                if isinstance(df, pd.DataFrame):
                    columns = _extract_columns_from_iloc_key(key, df)
                    if columns:
                        tracker.record_write(id(df), columns)
                return original_iloc_setitem(iloc_indexer, key, value)

            _iLocIndexer.__setitem__ = tracked_iloc_setitem
        except (ImportError, AttributeError):
            pass

        # ========== DataFrame.merge ==========
        self._original_methods['DataFrame.merge'] = pd.DataFrame.merge
        original_merge = self._original_methods['DataFrame.merge']

        def tracked_merge(df: pd.DataFrame, right, how='inner', on=None,
                          left_on=None, right_on=None, left_index=False,
                          right_index=False, sort=False, suffixes=('_x', '_y'),
                          copy=None, indicator=False, validate=None):
            # Track columns read from left DataFrame (self)
            if on is not None:
                cols = [on] if isinstance(on, str) else list(on)
                tracker.record_read(id(df), cols)
            if left_on is not None:
                cols = [left_on] if isinstance(left_on, str) else list(left_on)
                tracker.record_read(id(df), cols)

            # Track columns read from right DataFrame
            if isinstance(right, pd.DataFrame):
                if on is not None:
                    cols = [on] if isinstance(on, str) else list(on)
                    tracker.record_read(id(right), cols)
                if right_on is not None:
                    cols = [right_on] if isinstance(right_on, str) else list(right_on)
                    tracker.record_read(id(right), cols)

            return original_merge(df, right, how=how, on=on, left_on=left_on,
                                  right_on=right_on, left_index=left_index,
                                  right_index=right_index, sort=sort, suffixes=suffixes,
                                  copy=copy, indicator=indicator, validate=validate)

        pd.DataFrame.merge = tracked_merge

        # ========== DataFrameGroupBy.__getitem__ ==========
        try:
            from pandas.core.groupby import DataFrameGroupBy
            self._original_methods['DataFrameGroupBy.__getitem__'] = DataFrameGroupBy.__getitem__
            original_gb_getitem = self._original_methods['DataFrameGroupBy.__getitem__']

            def tracked_gb_getitem(gb, key):
                # Import cudf_compat for proxy detection
                from . import cudf_compat

                # Check for cudf GroupBy/proxy FIRST to avoid recursion
                # The cudf.pandas proxy system causes infinite recursion when our
                # patched method calls original_gb_getitem on a proxy object
                if cudf_compat.is_cudf_groupby(gb) or cudf_compat.is_cudf_proxy(gb):
                    # Still track column access using our stored mapping
                    df_id = tracker._groupby_to_df.get(id(gb))
                    if df_id is not None:
                        if isinstance(key, str):
                            tracker.record_read(df_id, [key])
                        elif isinstance(key, list):
                            str_keys = [k for k in key if isinstance(k, str)]
                            if str_keys:
                                tracker.record_read(df_id, str_keys)
                    # Call cudf's native method directly (bypass proxy recursion)
                    return cudf_compat.call_native_groupby_getitem(gb, key)

                # Standard pandas handling below
                df_id = None
                try:
                    df = gb.obj
                    if isinstance(df, pd.DataFrame):
                        df_id = id(df)
                except AttributeError:
                    # Fallback: look up DataFrame id from groupby mapping
                    df_id = tracker._groupby_to_df.get(id(gb))

                if df_id is not None:
                    if isinstance(key, str):
                        tracker.record_read(df_id, [key])
                    elif isinstance(key, list):
                        str_keys = [k for k in key if isinstance(k, str)]
                        if str_keys:
                            tracker.record_read(df_id, str_keys)
                return original_gb_getitem(gb, key)

            DataFrameGroupBy.__getitem__ = tracked_gb_getitem
        except (ImportError, AttributeError):
            pass

        # ========== DataFrame.sort_values ==========
        self._original_methods['DataFrame.sort_values'] = pd.DataFrame.sort_values
        original_sort_values = self._original_methods['DataFrame.sort_values']

        def tracked_sort_values(df: pd.DataFrame, by, **kwargs):
            cols = [by] if isinstance(by, str) else list(by)
            tracker.record_read(id(df), cols)
            return original_sort_values(df, by, **kwargs)

        pd.DataFrame.sort_values = tracked_sort_values

        # ========== DataFrame.drop_duplicates ==========
        self._original_methods['DataFrame.drop_duplicates'] = pd.DataFrame.drop_duplicates
        original_drop_duplicates = self._original_methods['DataFrame.drop_duplicates']

        def tracked_drop_duplicates(df: pd.DataFrame, subset=None, **kwargs):
            if subset is not None:
                cols = [subset] if isinstance(subset, str) else list(subset)
                tracker.record_read(id(df), cols)
            return original_drop_duplicates(df, subset=subset, **kwargs)

        pd.DataFrame.drop_duplicates = tracked_drop_duplicates

        # Save to class-level storage for other instances
        ColumnAccessTracker._class_original_methods = self._original_methods.copy()

    def _restore_dataframe_methods(self) -> None:
        """Restore original DataFrame methods."""
        # Restore DataFrame methods
        if 'DataFrame.__getitem__' in self._original_methods:
            pd.DataFrame.__getitem__ = self._original_methods['DataFrame.__getitem__']
        if 'DataFrame.__setitem__' in self._original_methods:
            pd.DataFrame.__setitem__ = self._original_methods['DataFrame.__setitem__']
        if 'DataFrame.assign' in self._original_methods:
            pd.DataFrame.assign = self._original_methods['DataFrame.assign']
        if 'DataFrame.drop' in self._original_methods:
            pd.DataFrame.drop = self._original_methods['DataFrame.drop']
        if 'DataFrame.groupby' in self._original_methods:
            pd.DataFrame.groupby = self._original_methods['DataFrame.groupby']
        if 'DataFrame.merge' in self._original_methods:
            pd.DataFrame.merge = self._original_methods['DataFrame.merge']
        if 'DataFrame.sort_values' in self._original_methods:
            pd.DataFrame.sort_values = self._original_methods['DataFrame.sort_values']
        if 'DataFrame.drop_duplicates' in self._original_methods:
            pd.DataFrame.drop_duplicates = self._original_methods['DataFrame.drop_duplicates']

        # Restore indexer methods
        try:
            from pandas.core.indexing import _LocIndexer, _iLocIndexer
            if '_LocIndexer.__getitem__' in self._original_methods:
                _LocIndexer.__getitem__ = self._original_methods['_LocIndexer.__getitem__']
            if '_LocIndexer.__setitem__' in self._original_methods:
                _LocIndexer.__setitem__ = self._original_methods['_LocIndexer.__setitem__']
            if '_iLocIndexer.__getitem__' in self._original_methods:
                _iLocIndexer.__getitem__ = self._original_methods['_iLocIndexer.__getitem__']
            if '_iLocIndexer.__setitem__' in self._original_methods:
                _iLocIndexer.__setitem__ = self._original_methods['_iLocIndexer.__setitem__']
        except (ImportError, AttributeError):
            pass

        # Restore DataFrameGroupBy methods
        try:
            from pandas.core.groupby import DataFrameGroupBy
            if 'DataFrameGroupBy.__getitem__' in self._original_methods:
                DataFrameGroupBy.__getitem__ = self._original_methods['DataFrameGroupBy.__getitem__']
        except (ImportError, AttributeError):
            pass

        self._original_methods.clear()


def _extract_columns_from_loc_key(key, df: pd.DataFrame) -> list:
    """Extract column names from a .loc key."""
    if not isinstance(key, tuple):
        # Single key - could be row selector only
        return []

    if len(key) < 2:
        return []

    col_key = key[1]

    if isinstance(col_key, str):
        return [col_key]
    elif isinstance(col_key, list):
        return [k for k in col_key if isinstance(k, str)]
    elif isinstance(col_key, pd.Index):
        return [k for k in col_key if isinstance(k, str)]
    elif isinstance(col_key, slice):
        # Slice of columns - need to resolve against DataFrame columns
        try:
            cols = df.columns[col_key]
            return list(cols)
        except Exception:
            return []

    return []


def _extract_columns_from_iloc_key(key, df: pd.DataFrame) -> list:
    """Extract column names from a .iloc key (positional)."""
    if not isinstance(key, tuple):
        return []

    if len(key) < 2:
        return []

    col_key = key[1]
    columns = df.columns

    try:
        if isinstance(col_key, int):
            return [columns[col_key]]
        elif isinstance(col_key, list):
            return [columns[i] for i in col_key if isinstance(i, int)]
        elif isinstance(col_key, slice):
            return list(columns[col_key])
    except (IndexError, KeyError):
        pass

    return []


from contextlib import contextmanager


@contextmanager
def suspend_column_tracking():
    """Context manager to temporarily suspend column access tracking.

    Use this when code needs to access DataFrame columns without triggering
    tracking, such as during deepcopy operations for checkpoints.

    Example:
        with suspend_column_tracking():
            df_copy = df.copy()
            for col in df.columns:
                # Access columns without recording reads
                data = df[col]
    """
    was_suspended = ColumnAccessTracker._suspended
    ColumnAccessTracker._suspended = True
    try:
        yield
    finally:
        ColumnAccessTracker._suspended = was_suspended


def walk_dataframes(
    namespace: dict,
    prefix: str = "",
    visited: Optional[Set[int]] = None
) -> Generator[Tuple[str, pd.DataFrame], None, None]:
    """
    Recursively find all DataFrames in namespace, including nested in objects.

    Handles:
    - Top-level variables: df
    - Dict values: data['train']
    - List/tuple items: datasets[0]
    - Object attributes: obj.df

    Args:
        namespace: The namespace dict to walk
        prefix: Current path prefix for nested access
        visited: Set of visited object IDs to prevent cycles

    Yields:
        Tuples of (path, DataFrame) for each DataFrame found
    """
    # Local import to avoid circular dependency
    from data_ferret.kernel.checkpoint import is_valid_variable_name

    if visited is None:
        visited = set()

    for key, val in namespace.items():
        # Skip IPython special variables and private variables
        if not isinstance(key, str) or not is_valid_variable_name(key):
            continue

        # Avoid cycles
        val_id = id(val)
        if val_id in visited:
            continue
        visited.add(val_id)

        # Build path
        if prefix:
            path = f"{prefix}['{key}']"
        else:
            path = str(key)

        # Skip modules - they can have DataFrames but we don't want to walk into them
        if isinstance(val, types.ModuleType):
            continue

        if isinstance(val, pd.DataFrame):
            yield path, val
        elif isinstance(val, dict):
            yield from walk_dataframes(val, path, visited)
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                item_id = id(item)
                if item_id in visited:
                    continue
                visited.add(item_id)
                # Skip modules in lists/tuples
                if isinstance(item, types.ModuleType):
                    continue
                item_path = f"{path}[{i}]"
                if isinstance(item, pd.DataFrame):
                    yield item_path, item
                elif isinstance(item, dict):
                    yield from walk_dataframes(item, item_path, visited)
                elif hasattr(item, '__dict__') and not callable(item):
                    yield from _walk_object_attrs(item, item_path, visited)
        elif hasattr(val, '__dict__') and not callable(val):
            # Recurse into object attributes
            yield from _walk_object_attrs(val, path, visited)


def _walk_object_attrs(
    obj: Any,
    prefix: str,
    visited: Set[int]
) -> Generator[Tuple[str, pd.DataFrame], None, None]:
    """
    Walk object attributes looking for DataFrames.

    Args:
        obj: The object to inspect
        prefix: Current path prefix
        visited: Set of visited object IDs

    Yields:
        Tuples of (path, DataFrame) for each DataFrame found
    """
    try:
        attrs = vars(obj)  # Get instance __dict__
    except TypeError:
        return  # Can't get vars for this object

    # Make a copy of items to avoid "dictionary changed size during iteration" errors
    for attr_name, attr_val in list(attrs.items()):
        # Skip private attributes
        if attr_name.startswith('_'):
            continue

        val_id = id(attr_val)
        if val_id in visited:
            continue
        visited.add(val_id)

        # Skip modules
        if isinstance(attr_val, types.ModuleType):
            continue

        path = f"{prefix}.{attr_name}"

        if isinstance(attr_val, pd.DataFrame):
            yield path, attr_val
        elif isinstance(attr_val, dict):
            yield from walk_dataframes(attr_val, path, visited)
        elif isinstance(attr_val, (list, tuple)):
            for i, item in enumerate(attr_val):
                item_id = id(item)
                if item_id in visited:
                    continue
                visited.add(item_id)
                # Skip modules in lists/tuples
                if isinstance(item, types.ModuleType):
                    continue
                item_path = f"{path}[{i}]"
                if isinstance(item, pd.DataFrame):
                    yield item_path, item
                elif hasattr(item, '__dict__') and not callable(item):
                    yield from _walk_object_attrs(item, item_path, visited)
        elif hasattr(attr_val, '__dict__') and not callable(attr_val):
            yield from _walk_object_attrs(attr_val, path, visited)


def walk_pandas_objects(
    namespace: dict,
    prefix: str = "",
    visited: Optional[Set[int]] = None
) -> Generator[Tuple[str, Any], None, None]:
    """
    Recursively find all DataFrames AND Series in namespace, including nested in objects.

    This is similar to walk_dataframes but also yields Series objects, which is
    needed for structural tracking (Series have index, dtype, name, etc.).

    Handles:
    - Top-level variables: df, s
    - Dict values: data['train']
    - List/tuple items: datasets[0]
    - Object attributes: obj.df

    Args:
        namespace: The namespace dict to walk
        prefix: Current path prefix for nested access
        visited: Set of visited object IDs to prevent cycles

    Yields:
        Tuples of (path, pandas_object) for each DataFrame or Series found
    """
    # Local import to avoid circular dependency
    from data_ferret.kernel.checkpoint import is_valid_variable_name

    if visited is None:
        visited = set()

    for key, val in namespace.items():
        # Skip IPython special variables and private variables
        if not isinstance(key, str) or not is_valid_variable_name(key):
            continue

        # Avoid cycles
        val_id = id(val)
        if val_id in visited:
            continue
        visited.add(val_id)

        # Build path
        if prefix:
            path = f"{prefix}['{key}']"
        else:
            path = str(key)

        # Skip modules - they can have DataFrames but we don't want to walk into them
        if isinstance(val, types.ModuleType):
            continue

        if isinstance(val, pd.DataFrame):
            yield path, val
        elif isinstance(val, pd.Series):
            yield path, val
        elif isinstance(val, dict):
            yield from walk_pandas_objects(val, path, visited)
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                item_id = id(item)
                if item_id in visited:
                    continue
                visited.add(item_id)
                # Skip modules in lists/tuples
                if isinstance(item, types.ModuleType):
                    continue
                item_path = f"{path}[{i}]"
                if isinstance(item, pd.DataFrame):
                    yield item_path, item
                elif isinstance(item, pd.Series):
                    yield item_path, item
                elif isinstance(item, dict):
                    yield from walk_pandas_objects(item, item_path, visited)
                elif hasattr(item, '__dict__') and not callable(item):
                    yield from _walk_object_attrs_pandas(item, item_path, visited)
        elif hasattr(val, '__dict__') and not callable(val):
            # Recurse into object attributes
            yield from _walk_object_attrs_pandas(val, path, visited)


def _walk_object_attrs_pandas(
    obj: Any,
    prefix: str,
    visited: Set[int]
) -> Generator[Tuple[str, Any], None, None]:
    """
    Walk object attributes looking for DataFrames and Series.

    Args:
        obj: The object to inspect
        prefix: Current path prefix
        visited: Set of visited object IDs

    Yields:
        Tuples of (path, pandas_object) for each DataFrame or Series found
    """
    try:
        attrs = vars(obj)  # Get instance __dict__
    except TypeError:
        return  # Can't get vars for this object

    # Make a copy of items to avoid "dictionary changed size during iteration" errors
    for attr_name, attr_val in list(attrs.items()):
        # Skip private attributes
        if attr_name.startswith('_'):
            continue

        val_id = id(attr_val)
        if val_id in visited:
            continue
        visited.add(val_id)

        # Skip modules
        if isinstance(attr_val, types.ModuleType):
            continue

        path = f"{prefix}.{attr_name}"

        if isinstance(attr_val, pd.DataFrame):
            yield path, attr_val
        elif isinstance(attr_val, pd.Series):
            yield path, attr_val
        elif isinstance(attr_val, dict):
            yield from walk_pandas_objects(attr_val, path, visited)
        elif isinstance(attr_val, (list, tuple)):
            for i, item in enumerate(attr_val):
                item_id = id(item)
                if item_id in visited:
                    continue
                visited.add(item_id)
                # Skip modules in lists/tuples
                if isinstance(item, types.ModuleType):
                    continue
                item_path = f"{path}[{i}]"
                if isinstance(item, pd.DataFrame):
                    yield item_path, item
                elif isinstance(item, pd.Series):
                    yield item_path, item
                elif hasattr(item, '__dict__') and not callable(item):
                    yield from _walk_object_attrs_pandas(item, item_path, visited)
        elif hasattr(attr_val, '__dict__') and not callable(attr_val):
            yield from _walk_object_attrs_pandas(attr_val, path, visited)
