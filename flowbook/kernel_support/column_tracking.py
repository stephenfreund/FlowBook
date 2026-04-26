"""
Column-level dependency tracking for DataFrames.

This module provides tracking of which DataFrame columns are read and written
during cell execution, using monkey-patching of pandas DataFrame methods.

The tracking works by:
1. Patching DataFrame methods (__getitem__, __setitem__, loc, iloc, etc.) ONCE at startup
2. Using activate()/deactivate() to enable/disable tracking per cell (~0.1µs)
3. Registering DataFrames lazily when accessed from namespace (no walking needed)
4. Resolving object IDs to variable paths after execution

This handles DataFrames at any nesting level: top-level, in dicts, lists, or object attributes.

Performance optimization (always-on pattern):
- Patches are installed once and never uninstalled during normal operation
- Per-cell overhead reduced from ~760µs to ~10-50µs (95% reduction)
- activate()/deactivate() just set/clear a thread-local pointer
- Lazy registration eliminates namespace walking at start/stop
"""

import threading
import types
import pandas as pd
from pandas._libs import lib
from typing import Dict, Set, Iterable, Tuple, Optional, Generator, Any
from collections import defaultdict

from flowbook.util.output import log, error, timer

# Threshold for skipping large primitive lists/tuples during dataframe walks.
# Below this size, the overhead of checking isn't worth it.
_LARGE_CONTAINER_THRESHOLD = 1000

# Primitive immutable types - lists/tuples containing only these cannot
# contain DataFrames or Series, so we can skip iterating through them.
_PRIMITIVE_TYPES = frozenset({
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
})


def _is_primitive_list(lst: list) -> bool:
    """Check if all elements in a list are primitive types."""
    for item in lst:
        if type(item) not in _PRIMITIVE_TYPES:
            return False
    return True


def _is_primitive_tuple(t: tuple) -> bool:
    """Check if all elements in a tuple are primitive types."""
    for item in t:
        if type(item) not in _PRIMITIVE_TYPES:
            return False
    return True


# Thread-local storage for the active tracker
_thread_local = threading.local()


class ColumnAccessTracker:
    """Tracks DataFrame column access via monkey-patching.

    Uses an always-on pattern for performance:
    - Patches are installed once at first use (class-level)
    - activate()/deactivate() control which tracker instance receives events
    - Lazy registration: DataFrames are registered when accessed from namespace
    """

    # Class-level flag to suspend tracking globally (for deepcopy operations)
    _suspended = False

    # Class-level tracking of patch state to prevent double-patching across instances
    _patches_installed = False
    _class_original_methods: Dict[str, Any] = {}

    @classmethod
    def _get_active_tracker(cls) -> Optional['ColumnAccessTracker']:
        """Get the currently active tracker (if any)."""
        return getattr(_thread_local, 'column_tracker', None)

    @classmethod
    def _set_active_tracker(cls, tracker: Optional['ColumnAccessTracker']) -> None:
        """Set the currently active tracker."""
        _thread_local.column_tracker = tracker

    def __init__(self, namespace_ref: Optional[dict] = None):
        self._reads_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._writes_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._id_to_path: Dict[int, str] = {}
        self._original_methods: Dict[str, Any] = {}
        self._installed = False
        # Reference to namespace for lazy fallback walk in resolve_to_paths()
        self._namespace_ref = namespace_ref
        # Mapping from GroupBy object id -> source DataFrame id
        # Used for cudf compatibility where gb.obj may not be accessible
        self._groupby_to_df: Dict[int, int] = {}
        # Current cell ID (set during activate, used for column provenance)
        self._cell_id: Optional[str] = None
        # Structural mutation tracking (recorded at operation time)
        self._row_mutations_by_id: Set[int] = set()
        self._index_mutations_by_id: Set[int] = set()
        self._dtype_changes_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._column_deletions_by_id: Dict[int, Set[str]] = defaultdict(set)

    def set_namespace_ref(self, namespace_ref: dict) -> None:
        """Set the namespace reference for lazy fallback walks."""
        self._namespace_ref = namespace_ref

    def _ensure_patches_installed(self) -> None:
        """Ensure patches are installed (idempotent, first call only)."""
        if ColumnAccessTracker._patches_installed:
            return
        with timer(key="tracking:patch_dataframe_methods", message="Patch DataFrame methods"):
            self._patch_dataframe_methods()
        ColumnAccessTracker._patches_installed = True
        # Install cudf tracking if available (all cudf logic in cudf_compat)
        with timer(key="tracking:cudf_install", message="Install cudf tracking"):
            from flowbook.kernel_support import cudf_compat
            cudf_compat.install_cudf_tracking(self)

    def activate(self, cell_id: Optional[str] = None) -> None:
        """Activate this tracker instance for the current thread.

        This makes this tracker the active one that receives column access events.
        Very fast (~0.1µs) - just sets a thread-local pointer.

        Args:
            cell_id: The ID of the cell being executed. Used for column
                provenance tracking (recording which cell created each column).
        """
        self._cell_id = cell_id
        self._ensure_patches_installed()
        ColumnAccessTracker._set_active_tracker(self)

    def deactivate(self) -> None:
        """Deactivate this tracker instance.

        Very fast (~0.1µs) - just clears the thread-local pointer.
        Does NOT uninstall patches (they stay installed for next cell).
        """
        if ColumnAccessTracker._get_active_tracker() is self:
            ColumnAccessTracker._set_active_tracker(None)
        self._cell_id = None

    def install(self) -> None:
        """Monkey-patch DataFrame methods to track column access.

        For backward compatibility. Prefer activate() for new code.
        """
        if self._installed:
            return
        # Use class-level check to prevent double-patching across instances
        if not ColumnAccessTracker._patches_installed:
            with timer(key="tracking:patch_dataframe_methods", message="Patch DataFrame methods"):
                self._patch_dataframe_methods()
            ColumnAccessTracker._patches_installed = True
            # Install cudf tracking if available (all cudf logic in cudf_compat)
            with timer(key="tracking:cudf_install", message="Install cudf tracking"):
                from flowbook.kernel_support import cudf_compat
                cudf_compat.install_cudf_tracking(self)
        else:
            # Patches already installed by another instance - just copy the originals
            self._original_methods = ColumnAccessTracker._class_original_methods.copy()
        self._installed = True
        # Also activate this tracker
        ColumnAccessTracker._set_active_tracker(self)

    def uninstall(self) -> None:
        """Restore original DataFrame methods.

        NOTE: In normal operation, patches are never uninstalled (they persist
        across cells for performance). This method is primarily for testing.
        """
        if not self._installed:
            return

        # Deactivate this tracker
        self.deactivate()

        # Uninstall cudf tracking if available (all cudf logic in cudf_compat)
        from flowbook.kernel_support import cudf_compat
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

        If there are unregistered DataFrame IDs (e.g., nested DataFrames not
        accessed via namespace), performs a lazy walk to resolve their paths.
        """
        result: Dict[str, Set[str]] = {}

        # Get all DataFrame IDs that had any column activity
        all_df_ids = set(self._reads_by_id.keys()) | set(self._writes_by_id.keys())

        # Check if any IDs are unregistered
        unregistered_ids = all_df_ids - set(self._id_to_path.keys())
        if unregistered_ids and self._namespace_ref is not None:
            # Lazy walk to find paths for unregistered DataFrames
            # This happens rarely (only for nested DataFrames not accessed via namespace)
            for path, df in walk_dataframes(self._namespace_ref):
                if id(df) in unregistered_ids:
                    self._id_to_path[id(df)] = path

        for df_id in all_df_ids:
            if df_id not in self._id_to_path:
                continue  # Still unresolved (deleted DataFrame?)

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

    # --- Structural mutation recording ---

    def record_row_mutation(self, df_id: int) -> None:
        """Record that a DataFrame had rows added or removed."""
        self._row_mutations_by_id.add(df_id)

    def record_index_mutation(self, df_id: int) -> None:
        """Record that a DataFrame had its index mutated."""
        self._index_mutations_by_id.add(df_id)

    def record_dtype_change(self, df_id: int, column: str) -> None:
        """Record that a DataFrame column's dtype changed."""
        self._dtype_changes_by_id[df_id].add(column)

    def record_column_deletion(self, df_id: int, column: str) -> None:
        """Record that a column was deleted from a DataFrame."""
        self._column_deletions_by_id[df_id].add(column)

    def resolve_row_mutations_to_paths(self) -> Set[str]:
        """Convert id-based row mutations to variable paths."""
        result: Set[str] = set()
        for df_id in self._row_mutations_by_id:
            if df_id in self._id_to_path:
                result.add(self._id_to_path[df_id])
        return result

    def resolve_index_mutations_to_paths(self) -> Set[str]:
        """Convert id-based index mutations to variable paths."""
        result: Set[str] = set()
        for df_id in self._index_mutations_by_id:
            if df_id in self._id_to_path:
                result.add(self._id_to_path[df_id])
        return result

    def resolve_dtype_changes_to_paths(self) -> Dict[str, Set[str]]:
        """Convert id-based dtype changes to variable path → column sets."""
        result: Dict[str, Set[str]] = {}
        for df_id, cols in self._dtype_changes_by_id.items():
            if df_id in self._id_to_path and cols:
                result[self._id_to_path[df_id]] = cols.copy()
        return result

    def resolve_column_deletions_to_paths(self) -> Dict[str, Set[str]]:
        """Convert id-based column deletions to variable path → column sets."""
        result: Dict[str, Set[str]] = {}
        for df_id, cols in self._column_deletions_by_id.items():
            if df_id in self._id_to_path and cols:
                result[self._id_to_path[df_id]] = cols.copy()
        return result

    def reset(self) -> None:
        """Reset tracking for new cell execution."""
        self._reads_by_id.clear()
        self._writes_by_id.clear()
        self._id_to_path.clear()
        self._groupby_to_df.clear()
        self._row_mutations_by_id.clear()
        self._index_mutations_by_id.clear()
        self._dtype_changes_by_id.clear()
        self._column_deletions_by_id.clear()

        # Reset cudf tracking state (all cudf logic in cudf_compat)
        from flowbook.kernel_support import cudf_compat
        cudf_compat.reset_cudf_tracking()

    def _patch_dataframe_methods(self) -> None:
        """Apply monkey-patches to DataFrame and related classes.

        IMPORTANT: These patches use ColumnAccessTracker._get_active_tracker() to
        look up the active tracker from thread-local storage. This allows the patches
        to stay installed permanently while different tracker instances are activated
        for different cell executions.
        """
        # Note: We don't use `self` for tracking in wrappers - instead we look up
        # the active tracker dynamically. This enables the always-on pattern.

        # ========== DataFrame.__getitem__ ==========
        self._original_methods['DataFrame.__getitem__'] = pd.DataFrame.__getitem__
        original_df_getitem = self._original_methods['DataFrame.__getitem__']

        def tracked_df_getitem(df: pd.DataFrame, key):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None:
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
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None:
                # Snapshot dtype before write for dtype-change provenance.
                # Use df.dtypes[key] instead of df[key].dtype to avoid
                # triggering tracked_df_getitem (which would record a
                # spurious column read).
                old_dtypes = {}
                if tracker._cell_id is not None:
                    dtypes = df.dtypes
                    if isinstance(key, str) and key in df.columns:
                        old_dtypes[key] = dtypes[key]
                    elif isinstance(key, list):
                        for k in key:
                            if isinstance(k, str) and k in df.columns:
                                old_dtypes[k] = dtypes[k]
                # Track column writes
                if isinstance(key, str):
                    tracker.record_write(id(df), [key])
                    # Record column provenance (first writer wins)
                    if tracker._cell_id is not None:
                        from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                        DataFrameProvenanceTracker.record_column_write(df, key, tracker._cell_id)
                elif isinstance(key, list):
                    str_keys = [k for k in key if isinstance(k, str)]
                    if str_keys:
                        tracker.record_write(id(df), str_keys)
                        if tracker._cell_id is not None:
                            from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                            for k in str_keys:
                                DataFrameProvenanceTracker.record_column_write(df, k, tracker._cell_id)
            result = original_df_setitem(df, key, value)
            # Check for dtype changes after write.
            # Use df.dtypes[col] instead of df[col].dtype to avoid triggering
            # tracked_df_getitem (which would record a spurious column read).
            if tracker is not None and tracker._cell_id is not None and old_dtypes:
                from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                new_dtypes = df.dtypes
                for col, old_dt in old_dtypes.items():
                    if col in df.columns and new_dtypes[col] != old_dt:
                        DataFrameProvenanceTracker.record_dtype_change(df, col, tracker._cell_id)
                        tracker.record_dtype_change(id(df), col)
            return result

        pd.DataFrame.__setitem__ = tracked_df_setitem

        # ========== DataFrame.__delitem__ ==========
        self._original_methods['DataFrame.__delitem__'] = pd.DataFrame.__delitem__
        original_df_delitem = self._original_methods['DataFrame.__delitem__']

        def tracked_df_delitem(df: pd.DataFrame, key):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None and isinstance(key, str):
                from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                DataFrameProvenanceTracker.record_column_delete(df, key, tracker._cell_id)
                tracker.record_column_deletion(id(df), key)
            return original_df_delitem(df, key)

        pd.DataFrame.__delitem__ = tracked_df_delitem

        # ========== DataFrame.insert ==========
        self._original_methods['DataFrame.insert'] = pd.DataFrame.insert
        original_df_insert = self._original_methods['DataFrame.insert']

        def tracked_df_insert(df: pd.DataFrame, loc, column, value, allow_duplicates=False):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None and isinstance(column, str):
                tracker.record_write(id(df), [column])
                if tracker._cell_id is not None:
                    from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                    DataFrameProvenanceTracker.record_column_write(df, column, tracker._cell_id)
            return original_df_insert(df, loc, column, value, allow_duplicates=allow_duplicates)

        pd.DataFrame.insert = tracked_df_insert

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
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None and columns is not None:
                # Track column drops as reads (need to know what columns exist)
                cols = [columns] if isinstance(columns, str) else list(columns)
                tracker.record_read(id(df), cols)

            # Snapshot state for provenance detection on inplace drops
            pre_len = None
            pre_cols = None
            if inplace and tracker is not None and tracker._cell_id is not None:
                pre_len = len(df)
                pre_cols = set(str(c) for c in df.columns)

            result = original_drop(df, labels=labels, axis=axis, index=index,
                                   columns=columns, level=level, inplace=inplace, errors=errors)

            if pre_len is not None:
                from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                cell_id = tracker._cell_id
                # Row drop provenance
                if len(df) != pre_len:
                    DataFrameProvenanceTracker.record_row_mutation(df, cell_id)
                    tracker.record_row_mutation(id(df))
                # Column deletion provenance
                post_cols = set(str(c) for c in df.columns)
                for col in pre_cols - post_cols:
                    DataFrameProvenanceTracker.record_column_delete(df, col, cell_id)
                    tracker.record_column_deletion(id(df), col)

            return result

        pd.DataFrame.drop = tracked_drop

        # ========== DataFrame.groupby ==========
        self._original_methods['DataFrame.groupby'] = pd.DataFrame.groupby
        original_groupby = self._original_methods['DataFrame.groupby']

        def tracked_groupby(df: pd.DataFrame, by=None, *args, **kwargs):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None and by is not None:
                # Track groupby columns as reads
                if isinstance(by, str):
                    tracker.record_read(id(df), [by])
                elif isinstance(by, list):
                    str_keys = [k for k in by if isinstance(k, str)]
                    if str_keys:
                        tracker.record_read(id(df), str_keys)
            result = original_groupby(df, by=by, *args, **kwargs)
            # Store mapping from GroupBy -> DataFrame for cudf compatibility
            if tracker is not None:
                tracker._groupby_to_df[id(result)] = id(df)
            return result

        pd.DataFrame.groupby = tracked_groupby

        # ========== _LocIndexer.__getitem__ ==========
        try:
            from pandas.core.indexing import _LocIndexer
            self._original_methods['_LocIndexer.__getitem__'] = _LocIndexer.__getitem__
            original_loc_getitem = self._original_methods['_LocIndexer.__getitem__']

            def tracked_loc_getitem(loc_indexer, key):
                tracker = ColumnAccessTracker._get_active_tracker()
                if tracker is not None:
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
                tracker = ColumnAccessTracker._get_active_tracker()
                pre_len = None
                if tracker is not None:
                    df = loc_indexer.obj
                    if isinstance(df, pd.DataFrame):
                        columns = _extract_columns_from_loc_key(key, df)
                        if columns:
                            tracker.record_write(id(df), columns)
                        if tracker._cell_id is not None:
                            pre_len = len(df)
                    del df  # pandas 2.x's chained-assignment check is sys.getrefcount(self.obj) <= 2; don't pin a third ref
                original_loc_setitem(loc_indexer, key, value)
                if pre_len is not None:
                    df = loc_indexer.obj
                    if len(df) != pre_len:
                        from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                        DataFrameProvenanceTracker.record_row_mutation(df, tracker._cell_id)
                        tracker.record_row_mutation(id(df))

            _LocIndexer.__setitem__ = tracked_loc_setitem
        except (ImportError, AttributeError):
            pass

        # ========== _iLocIndexer.__getitem__ ==========
        try:
            from pandas.core.indexing import _iLocIndexer
            self._original_methods['_iLocIndexer.__getitem__'] = _iLocIndexer.__getitem__
            original_iloc_getitem = self._original_methods['_iLocIndexer.__getitem__']

            def tracked_iloc_getitem(iloc_indexer, key):
                tracker = ColumnAccessTracker._get_active_tracker()
                if tracker is not None:
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
                tracker = ColumnAccessTracker._get_active_tracker()
                if tracker is not None:
                    df = iloc_indexer.obj
                    if isinstance(df, pd.DataFrame):
                        columns = _extract_columns_from_iloc_key(key, df)
                        if columns:
                            tracker.record_write(id(df), columns)
                    del df  # pandas 2.x's chained-assignment check is sys.getrefcount(self.obj) <= 2; don't pin a third ref
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
                          copy=lib.no_default, indicator=False, validate=None):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None:
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
                tracker = ColumnAccessTracker._get_active_tracker()

                # Import cudf_compat for proxy detection
                from flowbook.kernel_support import cudf_compat

                # Check for cudf GroupBy/proxy FIRST to avoid recursion
                # The cudf.pandas proxy system causes infinite recursion when our
                # patched method calls original_gb_getitem on a proxy object
                if cudf_compat.is_cudf_groupby(gb) or cudf_compat.is_cudf_proxy(gb):
                    # Still track column access using our stored mapping
                    if tracker is not None:
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
                if tracker is not None:
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
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None:
                cols = [by] if isinstance(by, str) else list(by)
                tracker.record_read(id(df), cols)
            return original_sort_values(df, by, **kwargs)

        pd.DataFrame.sort_values = tracked_sort_values

        # ========== DataFrame.drop_duplicates ==========
        self._original_methods['DataFrame.drop_duplicates'] = pd.DataFrame.drop_duplicates
        original_drop_duplicates = self._original_methods['DataFrame.drop_duplicates']

        def tracked_drop_duplicates(df: pd.DataFrame, subset=None, **kwargs):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None and subset is not None:
                cols = [subset] if isinstance(subset, str) else list(subset)
                tracker.record_read(id(df), cols)
            return original_drop_duplicates(df, subset=subset, **kwargs)

        pd.DataFrame.drop_duplicates = tracked_drop_duplicates

        # ========== DataFrame aggregation/transformation methods ==========
        # These methods read all (or all numeric) columns. We record reads of
        # every column so that column-level staleness works correctly.
        # Without this, df.sum() would only produce Var(df) in the read set,
        # and Col writes wouldn't trigger staleness (Col ▷ Var = false).
        _ALL_COL_METHODS = [
            'sum', 'mean', 'std', 'var', 'min', 'max', 'median',
            'describe', 'corr', 'cov', 'quantile', 'nunique',
            'apply', 'to_numpy', 'to_dict', 'to_records',
        ]

        for method_name in _ALL_COL_METHODS:
            original = getattr(pd.DataFrame, method_name, None)
            if original is None:
                continue
            self._original_methods[f'DataFrame.{method_name}'] = original

            def _make_all_col_tracked(orig, name):
                def tracked_method(df_self, *args, **kwargs):
                    tracker = ColumnAccessTracker._get_active_tracker()
                    if tracker is not None:
                        df_id = id(df_self)
                        if df_id in tracker._id_to_path:
                            cols = [str(c) for c in df_self.columns]
                            tracker.record_read(df_id, cols)
                    return orig(df_self, *args, **kwargs)
                tracked_method.__name__ = name
                tracked_method.__doc__ = orig.__doc__
                return tracked_method

            setattr(pd.DataFrame, method_name, _make_all_col_tracked(original, method_name))

        # ========== DataFrame.values (property) ==========
        original_values_fget = pd.DataFrame.values.fget
        if original_values_fget is not None:
            self._original_methods['DataFrame.values'] = original_values_fget

            def tracked_values(df_self):
                tracker = ColumnAccessTracker._get_active_tracker()
                if tracker is not None:
                    df_id = id(df_self)
                    if df_id in tracker._id_to_path:
                        cols = [str(c) for c in df_self.columns]
                        tracker.record_read(df_id, cols)
                return original_values_fget(df_self)

            pd.DataFrame.values = property(tracked_values)

        # ========== DataFrame._set_axis (index/columns setter) ==========
        # pd.DataFrame.index is an AxisProperty (Cython descriptor), not a
        # Python property, so we can't patch fget/fset. Instead patch _set_axis,
        # which is called when df.index = ... (axis=0) or df.columns = ... (axis=1).
        self._original_methods['DataFrame._set_axis'] = pd.DataFrame._set_axis
        original_set_axis = self._original_methods['DataFrame._set_axis']

        def patched_set_axis(df_self, axis, labels):
            tracker = ColumnAccessTracker._get_active_tracker()
            if tracker is not None and tracker._cell_id is not None and axis == 0:
                from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                DataFrameProvenanceTracker.record_index_mutation(df_self, tracker._cell_id)
                tracker.record_index_mutation(id(df_self))
            return original_set_axis(df_self, axis, labels)

        pd.DataFrame._set_axis = patched_set_axis

        # ========== Inplace method provenance wrapper ==========
        # Wraps DataFrame methods that accept inplace=True to detect
        # row mutations, index mutations, and dtype changes.
        import functools

        def _wrap_inplace_for_provenance(original, method_name):
            @functools.wraps(original)
            def wrapper(df_self, *args, inplace=False, **kwargs):
                tracker = ColumnAccessTracker._get_active_tracker()
                if not inplace or tracker is None or tracker._cell_id is None:
                    return original(df_self, *args, inplace=inplace, **kwargs)

                pre_len = len(df_self)
                pre_index = df_self.index.copy()
                pre_dtypes = {str(c): df_self[c].dtype for c in df_self.columns}

                result = original(df_self, *args, inplace=True, **kwargs)

                from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                cell_id = tracker._cell_id
                df_id = id(df_self)
                if len(df_self) != pre_len:
                    DataFrameProvenanceTracker.record_row_mutation(df_self, cell_id)
                    tracker.record_row_mutation(df_id)
                if not df_self.index.equals(pre_index):
                    DataFrameProvenanceTracker.record_index_mutation(df_self, cell_id)
                    tracker.record_index_mutation(df_id)
                post_dtypes = {str(c): df_self[c].dtype for c in df_self.columns}
                for col in post_dtypes:
                    if col in pre_dtypes and pre_dtypes[col] != post_dtypes[col]:
                        DataFrameProvenanceTracker.record_dtype_change(df_self, col, cell_id)
                        tracker.record_dtype_change(df_id, col)

                return result
            return wrapper

        _INPLACE_METHODS = [
            'dropna', 'drop_duplicates', 'reset_index', 'set_index',
            'sort_index', 'sort_values', 'rename', 'fillna', 'replace',
        ]

        for method_name in _INPLACE_METHODS:
            original = getattr(pd.DataFrame, method_name, None)
            if original is None:
                continue
            key = f'DataFrame.{method_name}_provenance'
            # Only save original if not already saved by an earlier patch
            if f'DataFrame.{method_name}' not in self._original_methods:
                self._original_methods[f'DataFrame.{method_name}'] = original
            self._original_methods[key] = original
            wrapped = _wrap_inplace_for_provenance(
                self._original_methods[f'DataFrame.{method_name}'], method_name
            )
            setattr(pd.DataFrame, method_name, wrapped)

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

        # Restore aggregation/transformation methods
        for method_name in ['sum', 'mean', 'std', 'var', 'min', 'max', 'median',
                            'describe', 'corr', 'cov', 'quantile', 'nunique',
                            'apply', 'to_numpy', 'to_dict', 'to_records']:
            key = f'DataFrame.{method_name}'
            if key in self._original_methods:
                setattr(pd.DataFrame, method_name, self._original_methods[key])

        # Restore values property
        if 'DataFrame.values' in self._original_methods:
            pd.DataFrame.values = property(self._original_methods['DataFrame.values'])

        # Restore _set_axis
        if 'DataFrame._set_axis' in self._original_methods:
            pd.DataFrame._set_axis = self._original_methods['DataFrame._set_axis']

        # Restore inplace methods
        for method_name in ['dropna', 'drop_duplicates', 'reset_index', 'set_index',
                            'sort_index', 'sort_values', 'rename', 'fillna', 'replace']:
            key = f'DataFrame.{method_name}'
            if key in self._original_methods:
                setattr(pd.DataFrame, method_name, self._original_methods[key])

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
    from flowbook.kernel_support.checkpoint import is_valid_variable_name

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
            # OPTIMIZATION: Skip large lists/tuples that contain only primitives
            # They cannot contain DataFrames, so no point iterating through them
            if len(val) >= _LARGE_CONTAINER_THRESHOLD:
                if isinstance(val, list) and _is_primitive_list(val):
                    continue
                if isinstance(val, tuple) and _is_primitive_tuple(val):
                    continue
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
            # OPTIMIZATION: Skip large lists/tuples that contain only primitives
            if len(attr_val) >= _LARGE_CONTAINER_THRESHOLD:
                if isinstance(attr_val, list) and _is_primitive_list(attr_val):
                    continue
                if isinstance(attr_val, tuple) and _is_primitive_tuple(attr_val):
                    continue
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
    from flowbook.kernel_support.checkpoint import is_valid_variable_name

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
            # OPTIMIZATION: Skip large lists/tuples that contain only primitives
            # They cannot contain DataFrames/Series, so no point iterating through them
            if len(val) >= _LARGE_CONTAINER_THRESHOLD:
                if isinstance(val, list) and _is_primitive_list(val):
                    continue
                if isinstance(val, tuple) and _is_primitive_tuple(val):
                    continue
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
            # OPTIMIZATION: Skip large lists/tuples that contain only primitives
            if len(attr_val) >= _LARGE_CONTAINER_THRESHOLD:
                if isinstance(attr_val, list) and _is_primitive_list(attr_val):
                    continue
                if isinstance(attr_val, tuple) and _is_primitive_tuple(attr_val):
                    continue
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
