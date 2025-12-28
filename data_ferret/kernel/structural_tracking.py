"""
Structural attribute tracking for DataFrames and Series.

Tracks when code accesses structural attributes like .columns, .shape, .dtype
or methods like .describe(), .to_dict() that reveal the structure of data objects.

When these are accessed, the monotonicity/SDC diff must require structural
equality, not just value equality for accessed columns.

The Problem This Solves:
------------------------
Consider this code:
    cols = df.columns.tolist()  # Returns ['a', 'b']

If another cell adds a column to df, then df.columns returns ['a', 'b', 'c'].
The standard LEQ diff would say "df is still valid" because columns a and b
have the same values. But the BEHAVIOR of the code changes because df.columns
returns a different list.

By tracking that df.columns was accessed, we can require that df.columns
remains identical (not just that existing columns have same values).

Modes:
- "off": Don't track structural reads at all
- "warn": Track and log warnings, but don't enforce in diff
- "enforce": Track and require structural equality in diff

Note: numpy arrays don't need special tracking - they're fully compared,
so accessing arr.shape is covered by normal variable-level read tracking.
"""

import pandas as pd
import threading
from typing import Dict, Set, Any, Optional
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum

from data_ferret.util.output import timer


def _unwrap_cudf_proxy(obj: Any) -> Any:
    """
    Unwrap a cudf.pandas proxy object to get the underlying pandas object.

    cudf.pandas creates proxy objects that wrap pandas objects. When calling
    original pandas methods, we need the underlying pandas object, not the proxy.

    The proxy stores the slow (pandas) object in _fsproxy_slow attribute.

    Args:
        obj: Any object, possibly a cudf proxy

    Returns:
        The underlying pandas object if obj is a cudf proxy, otherwise obj unchanged
    """
    # Check if this is a cudf proxy by looking for the _fsproxy_slow attribute
    # IMPORTANT: Use object.__getattribute__ to bypass our wrapped __getattribute__
    # Otherwise hasattr() triggers our wrapper -> _unwrap_cudf_proxy -> infinite recursion
    try:
        slow_obj = object.__getattribute__(obj, '_fsproxy_slow')
        # _fsproxy_slow can be a callable that returns the slow object
        if callable(slow_obj):
            return slow_obj()
        return slow_obj
    except AttributeError:
        return obj


# =============================================================================
# Thread-local for structure-using methods
# =============================================================================

_in_structure_using_method = threading.local()


@contextmanager
def _structure_using_context():
    """Context manager to mark we're inside a structure-using method.

    Structure-using methods (like __setitem__, arithmetic, display) internally
    access structural attributes but shouldn't count as explicit user access.
    """
    prev = getattr(_in_structure_using_method, 'active', False)
    _in_structure_using_method.active = True
    try:
        yield
    finally:
        _in_structure_using_method.active = prev


class StructuralTrackingMode(str, Enum):
    """Mode for structural attribute tracking."""
    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


# =============================================================================
# Structural attributes/methods to track
# =============================================================================

# DataFrame attributes that reveal column structure
DATAFRAME_COLUMN_STRUCTURAL = frozenset({
    'columns',      # Index of column names
    'keys',         # Same as columns
    'dtypes',       # Series with dtype per column
    'T',            # Transpose exposes columns as rows
    'axes',         # [index, columns]
    'values',       # Full 2D array (shape visible)
})

# DataFrame attributes that reveal row structure
DATAFRAME_ROW_STRUCTURAL = frozenset({
    'index',        # Row labels
    'shape',        # (rows, cols)
    'size',         # rows * cols
    'empty',        # True if no rows/cols
})

# DataFrame methods that reveal structure (tracked when accessed)
DATAFRAME_STRUCTURAL_METHODS = frozenset({
    'info',         # Prints structure
    'describe',     # Stats DataFrame with column per numeric col
    'to_dict',      # Dict with column names as keys
    'to_records',   # RecArray based on columns
    'to_numpy',     # Array with shape (rows, cols)
    'head',         # Exposes column structure
    'tail',         # Exposes column structure
    'sample',       # Exposes column structure
    'copy',         # Preserves structure
    'select_dtypes', # Returns subset of columns
    'memory_usage', # Series with one entry per column
})

# All DataFrame structural attributes
DATAFRAME_STRUCTURAL_ATTRS = DATAFRAME_COLUMN_STRUCTURAL | DATAFRAME_ROW_STRUCTURAL

# Combined: attrs that reveal column structure (for diff checks)
COLUMN_REVEALING_ATTRS = frozenset({
    'columns', 'keys', 'iter', 'dtypes', 'T', 'axes', 'values',
    'describe', 'to_dict', 'info', 'head', 'tail', 'sample',
    'select_dtypes', 'to_records', 'memory_usage',
})

# Combined: attrs that reveal row structure (for diff checks)
ROW_REVEALING_ATTRS = frozenset({
    'index', 'len', 'shape', 'size', 'empty',
})

# Series attributes that reveal structure
SERIES_STRUCTURAL_ATTRS = frozenset({
    'index',        # Element labels
    'shape',        # (length,)
    'dtype',        # Data type
    'name',         # Series name
    'size',         # Number of elements
    'empty',        # True if no elements
    'values',       # Array (length visible)
})

# Series methods that reveal structure
SERIES_STRUCTURAL_METHODS = frozenset({
    'to_dict',      # Dict with index as keys
    'to_list',      # List (length visible)
    'to_numpy',     # Array (length visible)
    'describe',     # Stats based on dtype
    'copy',         # Preserves structure
})

# =============================================================================
# Structure-using methods (should NOT record structural reads)
# =============================================================================
#
# These methods internally access structural attributes (.columns, .index, etc.)
# as implementation details, but their primary purpose is NOT to reveal structure.
# We only want to track EXPLICIT user access to structural attributes.
#
# Categories:
# - Display: repr, str, html rendering
# - Attribute lookup: __getattr__ (pandas checks if attr is a column name)
# - Item access: __getitem__, __setitem__, __delitem__
# - Arithmetic: +, -, *, /, etc.
# - Comparison: ==, !=, <, >, etc.
# - Aggregation: mean, sum, min, max, etc.
# - Transform: apply, map, transform, pipe
# - Reshape: merge, join, groupby, pivot, etc.
# - I/O: to_csv, to_json, etc.

STRUCTURE_USING_METHODS = [
    # --- Display ---
    ('DataFrame', '_repr_html_'),
    ('DataFrame', '__repr__'),
    ('DataFrame', '__str__'),
    ('DataFrame', '_repr_latex_'),
    ('DataFrame', '_repr_data_resource_'),
    ('DataFrame', 'to_html'),

    # --- Attribute lookup ---
    ('DataFrame', '__getattr__'),

    # --- Item access/mutation ---
    ('DataFrame', '__getitem__'),
    ('DataFrame', '__setitem__'),
    ('DataFrame', '__delitem__'),
    ('DataFrame', 'insert'),
    ('DataFrame', 'drop'),
    ('DataFrame', 'assign'),
    ('DataFrame', 'pop'),

    # --- Arithmetic ---
    ('DataFrame', '__add__'),
    ('DataFrame', '__radd__'),
    ('DataFrame', '__sub__'),
    ('DataFrame', '__rsub__'),
    ('DataFrame', '__mul__'),
    ('DataFrame', '__rmul__'),
    ('DataFrame', '__truediv__'),
    ('DataFrame', '__rtruediv__'),
    ('DataFrame', '__floordiv__'),
    ('DataFrame', '__rfloordiv__'),
    ('DataFrame', '__mod__'),
    ('DataFrame', '__rmod__'),
    ('DataFrame', '__pow__'),
    ('DataFrame', '__rpow__'),
    ('DataFrame', '__neg__'),
    ('DataFrame', '__pos__'),
    ('DataFrame', '__abs__'),

    # --- Comparison ---
    ('DataFrame', '__eq__'),
    ('DataFrame', '__ne__'),
    ('DataFrame', '__lt__'),
    ('DataFrame', '__le__'),
    ('DataFrame', '__gt__'),
    ('DataFrame', '__ge__'),

    # --- Aggregation ---
    ('DataFrame', 'mean'),
    ('DataFrame', 'sum'),
    ('DataFrame', 'min'),
    ('DataFrame', 'max'),
    ('DataFrame', 'std'),
    ('DataFrame', 'var'),
    ('DataFrame', 'count'),
    ('DataFrame', 'median'),
    ('DataFrame', 'mode'),
    ('DataFrame', 'prod'),
    ('DataFrame', 'cumsum'),
    ('DataFrame', 'cumprod'),
    ('DataFrame', 'cummax'),
    ('DataFrame', 'cummin'),
    ('DataFrame', 'diff'),
    ('DataFrame', 'pct_change'),
    ('DataFrame', 'rank'),
    ('DataFrame', 'quantile'),
    ('DataFrame', 'sem'),
    ('DataFrame', 'skew'),
    ('DataFrame', 'kurt'),
    ('DataFrame', 'corr'),
    ('DataFrame', 'cov'),
    ('DataFrame', 'nunique'),
    ('DataFrame', 'value_counts'),

    # --- Transform/Apply ---
    ('DataFrame', 'apply'),
    ('DataFrame', 'map'),
    ('DataFrame', 'applymap'),
    ('DataFrame', 'transform'),
    ('DataFrame', 'pipe'),
    ('DataFrame', 'agg'),
    ('DataFrame', 'aggregate'),

    # --- Reshape ---
    ('DataFrame', 'merge'),
    ('DataFrame', 'join'),
    ('DataFrame', 'pivot'),
    ('DataFrame', 'pivot_table'),
    ('DataFrame', 'melt'),
    ('DataFrame', 'stack'),
    ('DataFrame', 'unstack'),
    ('DataFrame', 'groupby'),
    ('DataFrame', 'resample'),
    ('DataFrame', 'rolling'),
    ('DataFrame', 'expanding'),
    ('DataFrame', 'ewm'),
    ('DataFrame', 'transpose'),
    ('DataFrame', 'sort_values'),
    ('DataFrame', 'sort_index'),
    ('DataFrame', 'reset_index'),
    ('DataFrame', 'set_index'),
    ('DataFrame', 'reindex'),
    ('DataFrame', 'rename'),
    ('DataFrame', 'fillna'),
    ('DataFrame', 'dropna'),
    ('DataFrame', 'duplicated'),
    ('DataFrame', 'drop_duplicates'),
    ('DataFrame', 'replace'),
    ('DataFrame', 'clip'),
    ('DataFrame', 'where'),
    ('DataFrame', 'mask'),
    ('DataFrame', 'query'),
    ('DataFrame', 'eval'),

    # --- I/O ---
    ('DataFrame', 'to_csv'),
    ('DataFrame', 'to_json'),
    ('DataFrame', 'to_pickle'),
    ('DataFrame', 'to_parquet'),
    ('DataFrame', 'to_excel'),
    ('DataFrame', 'to_sql'),
    ('DataFrame', 'to_feather'),
    ('DataFrame', 'to_hdf'),
    ('DataFrame', 'to_latex'),
    ('DataFrame', 'to_markdown'),
    ('DataFrame', 'to_clipboard'),
    ('DataFrame', 'to_string'),

    # --- Boolean ---
    ('DataFrame', '__bool__'),
    ('DataFrame', 'any'),
    ('DataFrame', 'all'),
    ('DataFrame', 'isin'),
    ('DataFrame', 'isna'),
    ('DataFrame', 'isnull'),
    ('DataFrame', 'notna'),
    ('DataFrame', 'notnull'),

    # --- Bitwise (used for boolean mask operations like mask1 & mask2) ---
    ('DataFrame', '__and__'),
    ('DataFrame', '__rand__'),
    ('DataFrame', '__or__'),
    ('DataFrame', '__ror__'),
    ('DataFrame', '__xor__'),
    ('DataFrame', '__rxor__'),
    ('DataFrame', '__invert__'),

    # --- Series ---
    ('Series', '_repr_html_'),
    ('Series', '__repr__'),
    ('Series', '__str__'),
    ('Series', 'to_string'),
    ('Series', '__getattr__'),
    ('Series', '__getitem__'),
    ('Series', '__setitem__'),
    ('Series', '__delitem__'),
    ('Series', '__add__'),
    ('Series', '__radd__'),
    ('Series', '__sub__'),
    ('Series', '__rsub__'),
    ('Series', '__mul__'),
    ('Series', '__rmul__'),
    ('Series', '__truediv__'),
    ('Series', '__rtruediv__'),
    ('Series', '__floordiv__'),
    ('Series', '__rfloordiv__'),
    ('Series', '__mod__'),
    ('Series', '__rmod__'),
    ('Series', '__pow__'),
    ('Series', '__rpow__'),
    ('Series', '__neg__'),
    ('Series', '__pos__'),
    ('Series', '__abs__'),
    ('Series', '__eq__'),
    ('Series', '__ne__'),
    ('Series', '__lt__'),
    ('Series', '__le__'),
    ('Series', '__gt__'),
    ('Series', '__ge__'),
    ('Series', 'mean'),
    ('Series', 'sum'),
    ('Series', 'min'),
    ('Series', 'max'),
    ('Series', 'std'),
    ('Series', 'var'),
    ('Series', 'count'),
    ('Series', 'median'),
    ('Series', 'mode'),
    ('Series', 'prod'),
    ('Series', 'cumsum'),
    ('Series', 'cumprod'),
    ('Series', 'cummax'),
    ('Series', 'cummin'),
    ('Series', 'diff'),
    ('Series', 'pct_change'),
    ('Series', 'rank'),
    ('Series', 'quantile'),
    ('Series', 'apply'),
    ('Series', 'map'),
    ('Series', 'transform'),
    ('Series', 'pipe'),
    ('Series', 'agg'),
    ('Series', 'aggregate'),
    ('Series', 'unique'),
    ('Series', 'nunique'),
    ('Series', 'groupby'),
    ('Series', 'sort_values'),
    ('Series', 'sort_index'),
    ('Series', 'reset_index'),
    ('Series', 'fillna'),
    ('Series', 'dropna'),
    ('Series', 'duplicated'),
    ('Series', 'drop_duplicates'),
    ('Series', 'replace'),
    ('Series', 'clip'),
    ('Series', 'where'),
    ('Series', 'mask'),
    ('Series', '__bool__'),
    ('Series', 'any'),
    ('Series', 'all'),
    ('Series', 'isin'),
    ('Series', 'isna'),
    ('Series', 'isnull'),
    ('Series', 'notna'),
    ('Series', 'notnull'),

    # --- Bitwise (used for boolean mask operations like mask1 & mask2) ---
    ('Series', '__and__'),
    ('Series', '__rand__'),
    ('Series', '__or__'),
    ('Series', '__ror__'),
    ('Series', '__xor__'),
    ('Series', '__rxor__'),
    ('Series', '__invert__'),

    ('Series', 'to_csv'),
    ('Series', 'to_json'),
    ('Series', 'to_pickle'),
    ('Series', 'to_frame'),
]

# Indexer classes that need wrapping
# These are accessed via df.loc, df.iloc, etc.
INDEXER_METHODS_TO_WRAP = [
    '__getitem__',
    '__setitem__',
]

# Module-level pandas functions that internally access structural attrs
# These are functions like pd.concat, pd.merge that operate on DataFrames
# but shouldn't count as explicit structural reads
PANDAS_FUNCTIONS_TO_WRAP = [
    'concat',
    'merge',
    'merge_ordered',
    'merge_asof',
    'get_dummies',
    'crosstab',
    'cut',
    'qcut',
    'melt',
    'wide_to_long',
    'pivot',
    'pivot_table',
]


class StructuralAccessTracker:
    """
    Tracks structural attribute/method access via monkey-patching.

    Similar to ColumnAccessTracker, but for structural attributes and methods
    like .columns, .shape, .describe(), etc.

    Attributes:
        mode: Current tracking mode ("off", "warn", "enforce")

    Usage:
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
        tracker.register(df, 'df')
        tracker.install()

        # ... execute user code ...
        _ = df.columns  # Tracked!

        tracker.uninstall()
        result = tracker.resolve_to_paths()
        # result = {'df': {'columns'}}
    """

    # Class-level flag to suspend tracking (for deepcopy operations)
    _suspended = False

    # Class-level tracking of patch state to prevent double-patching across instances
    _patches_installed = False
    _class_original_methods: Dict[str, Any] = {}

    def __init__(self, mode: StructuralTrackingMode = StructuralTrackingMode.WARN):
        """
        Initialize the tracker.

        Args:
            mode: Tracking mode - OFF, WARN, or ENFORCE
        """
        self._mode = mode
        self._reads_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._id_to_path: Dict[int, str] = {}
        self._original_methods: Dict[str, Any] = {}
        self._installed = False

    @property
    def mode(self) -> StructuralTrackingMode:
        """Current tracking mode."""
        return self._mode

    @mode.setter
    def mode(self, value: StructuralTrackingMode) -> None:
        """Set tracking mode."""
        self._mode = value

    def set_mode(self, mode: str) -> None:
        """
        Set mode from string. Convenience for magic commands.

        Args:
            mode: One of "off", "warn", "enforce"
        """
        self._mode = StructuralTrackingMode(mode.lower())

    def install(self) -> None:
        """Monkey-patch DataFrame and Series to track structural access."""
        if self._installed:
            return
        if self._mode == StructuralTrackingMode.OFF:
            return
        # Use class-level check to prevent double-patching across instances
        if not StructuralAccessTracker._patches_installed:
            with timer(key="tracking:patch_structural", message="Patch structural methods"):
                self._patch_dataframe()
                self._patch_series()
                self._patch_structure_using_methods()
                self._patch_indexers()
                self._patch_pandas_functions()
                self._patch_groupby()
            # Save to class-level storage for other instances
            StructuralAccessTracker._class_original_methods = self._original_methods.copy()
            StructuralAccessTracker._patches_installed = True
        else:
            # Patches already installed by another instance - just copy the originals
            self._original_methods = StructuralAccessTracker._class_original_methods.copy()
        self._installed = True

    def uninstall(self) -> None:
        """Restore original methods."""
        if not self._installed:
            return
        # Only restore if we have the original methods stored
        if self._original_methods:
            self._restore_all()
            StructuralAccessTracker._patches_installed = False
            StructuralAccessTracker._class_original_methods.clear()
        self._installed = False

    def register(self, obj: Any, path: str) -> None:
        """
        Register an object with its namespace path.

        Args:
            obj: The object (DataFrame, Series)
            path: Variable path like 'df' or 'data["train"]'
        """
        if self._mode != StructuralTrackingMode.OFF:
            self._id_to_path[id(obj)] = path

    def record_structural_read(self, obj_id: int, attr: str) -> None:
        """
        Record that a structural attribute/method was accessed.

        Args:
            obj_id: The id() of the object
            attr: Name of the attribute or method accessed
        """
        if StructuralAccessTracker._suspended:
            return
        if self._mode == StructuralTrackingMode.OFF:
            return
        if getattr(_in_structure_using_method, 'active', False):
            return  # Inside structure-using method, don't record
        self._reads_by_id[obj_id].add(attr)

    def resolve_to_paths(self) -> Dict[str, Set[str]]:
        """
        Convert id-based tracking to path-based.

        Returns:
            Dict mapping variable paths to sets of structural attrs accessed.
            e.g., {'df': {'columns', 'shape'}, 'data["train"]': {'describe'}}
        """
        if self._mode == StructuralTrackingMode.OFF:
            return {}
        result: Dict[str, Set[str]] = {}
        for obj_id, attrs in self._reads_by_id.items():
            if obj_id in self._id_to_path:
                path = self._id_to_path[obj_id]
                result[path] = attrs.copy()
        return result

    def reset(self) -> None:
        """Reset tracking for new cell execution."""
        self._reads_by_id.clear()
        self._id_to_path.clear()

    def _patch_dataframe(self) -> None:
        """Patch DataFrame to track structural attribute and method access."""
        tracker = self

        # --- Patch __getattribute__ for attribute access ---
        original_getattr = pd.DataFrame.__getattribute__
        self._original_methods['DataFrame.__getattribute__'] = original_getattr

        def tracked_getattribute(df, name):
            result = original_getattr(df, name)
            # Track structural attributes
            if name in DATAFRAME_STRUCTURAL_ATTRS:
                tracker.record_structural_read(id(df), name)
            # Track when structural methods are accessed
            elif name in DATAFRAME_STRUCTURAL_METHODS:
                tracker.record_structural_read(id(df), name)
            return result

        pd.DataFrame.__getattribute__ = tracked_getattribute

        # --- Patch __len__ for len(df) ---
        original_len = pd.DataFrame.__len__
        self._original_methods['DataFrame.__len__'] = original_len

        def tracked_len(df):
            tracker.record_structural_read(id(df), 'len')
            return original_len(df)

        pd.DataFrame.__len__ = tracked_len

        # --- Patch __iter__ for `for col in df:` ---
        original_iter = pd.DataFrame.__iter__
        self._original_methods['DataFrame.__iter__'] = original_iter

        def tracked_iter(df):
            tracker.record_structural_read(id(df), 'iter')
            return original_iter(df)

        pd.DataFrame.__iter__ = tracked_iter

    def _patch_series(self) -> None:
        """Patch Series to track structural attribute and method access."""
        tracker = self

        # --- Patch __getattribute__ for attribute access ---
        original_getattr = pd.Series.__getattribute__
        self._original_methods['Series.__getattribute__'] = original_getattr

        def tracked_getattribute(s, name):
            result = original_getattr(s, name)
            if name in SERIES_STRUCTURAL_ATTRS:
                tracker.record_structural_read(id(s), name)
            elif name in SERIES_STRUCTURAL_METHODS:
                tracker.record_structural_read(id(s), name)
            return result

        pd.Series.__getattribute__ = tracked_getattribute

        # --- Patch __len__ for len(s) ---
        original_len = pd.Series.__len__
        self._original_methods['Series.__len__'] = original_len

        def tracked_len(s):
            tracker.record_structural_read(id(s), 'len')
            return original_len(s)

        pd.Series.__len__ = tracked_len

        # --- Patch __iter__ for `for val in s:` ---
        original_iter = pd.Series.__iter__
        self._original_methods['Series.__iter__'] = original_iter

        def tracked_iter(s):
            tracker.record_structural_read(id(s), 'iter')
            return original_iter(s)

        pd.Series.__iter__ = tracked_iter

    def _patch_structure_using_methods(self) -> None:
        """Patch structure-using methods to exclude their internals from tracking.

        Structure-using methods internally access structural attributes but
        their primary purpose is NOT to reveal structure. We wrap them to
        set a context flag that prevents structural read recording.
        """
        for cls_name, method_name in STRUCTURE_USING_METHODS:
            cls = pd.DataFrame if cls_name == 'DataFrame' else pd.Series
            original = getattr(cls, method_name, None)
            if original is None:
                continue

            key = f'{cls_name}.{method_name}'
            self._original_methods[key] = original

            def make_wrapper(orig):
                def wrapper(obj, *args, **kwargs):
                    with _structure_using_context():
                        # Unwrap cudf proxy to get underlying pandas object
                        unwrapped = _unwrap_cudf_proxy(obj)
                        return orig(unwrapped, *args, **kwargs)
                return wrapper

            setattr(cls, method_name, make_wrapper(original))

    def _patch_indexers(self) -> None:
        """Patch indexer classes (loc, iloc, at, iat) to suppress structural reads.

        Indexers like df.loc[...] internally access structural attributes like
        .columns, .index, .axes but from the user's perspective these are just
        data access operations, not structure-revealing operations.
        """
        try:
            from pandas.core.indexing import _LocIndexer, _iLocIndexer, _AtIndexer, _iAtIndexer
            indexer_classes = [
                ('_LocIndexer', _LocIndexer),
                ('_iLocIndexer', _iLocIndexer),
                ('_AtIndexer', _AtIndexer),
                ('_iAtIndexer', _iAtIndexer),
            ]
        except ImportError:
            # Older pandas versions may have different structure
            return

        for cls_name, cls in indexer_classes:
            for method_name in INDEXER_METHODS_TO_WRAP:
                original = getattr(cls, method_name, None)
                if original is None:
                    continue

                key = f'{cls_name}.{method_name}'
                self._original_methods[key] = original

                def make_wrapper(orig):
                    def wrapper(obj, *args, **kwargs):
                        with _structure_using_context():
                            # Unwrap cudf proxy to get underlying pandas object
                            unwrapped = _unwrap_cudf_proxy(obj)
                            return orig(unwrapped, *args, **kwargs)
                    return wrapper

                setattr(cls, method_name, make_wrapper(original))

    def _patch_pandas_functions(self) -> None:
        """Patch module-level pandas functions to suppress structural reads.

        Functions like pd.concat, pd.merge internally access structural attributes
        of input DataFrames but from the user's perspective these are data
        transformation operations, not structure-revealing operations.
        """
        for func_name in PANDAS_FUNCTIONS_TO_WRAP:
            original = getattr(pd, func_name, None)
            if original is None:
                continue

            key = f'pd.{func_name}'
            self._original_methods[key] = original

            def make_wrapper(orig):
                def wrapper(*args, **kwargs):
                    with _structure_using_context():
                        return orig(*args, **kwargs)
                return wrapper

            setattr(pd, func_name, make_wrapper(original))

    def _patch_groupby(self) -> None:
        """Patch GroupBy classes to suppress structural reads.

        When you do df.groupby(...)['col'], the __getitem__ on the GroupBy
        object internally accesses df.columns to validate the column selection.
        Aggregation methods like .sum(), .mean() also access structure internally.
        This should not count as an explicit structural read.
        """
        try:
            from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
            groupby_classes = [
                ('DataFrameGroupBy', DataFrameGroupBy),
                ('SeriesGroupBy', SeriesGroupBy),
            ]
        except ImportError:
            return

        # Methods on GroupBy that should suppress structural reads
        groupby_methods = [
            '__getitem__',
            # Aggregation methods
            'sum', 'mean', 'median', 'std', 'var', 'min', 'max',
            'count', 'size', 'first', 'last', 'nth', 'prod',
            'sem', 'ohlc', 'describe',
            # Transform methods
            'transform', 'apply', 'agg', 'aggregate',
            'filter', 'pipe',
            # Other methods that might access structure
            'cumsum', 'cumprod', 'cummax', 'cummin', 'cumcount',
            'diff', 'pct_change', 'rank', 'shift', 'fillna',
            'ffill', 'bfill', 'head', 'tail', 'nunique', 'value_counts',
        ]

        for cls_name, cls in groupby_classes:
            for method_name in groupby_methods:
                original = getattr(cls, method_name, None)
                if original is None:
                    continue

                key = f'{cls_name}.{method_name}'
                self._original_methods[key] = original

                def make_wrapper(orig):
                    def wrapper(obj, *args, **kwargs):
                        with _structure_using_context():
                            # Unwrap cudf proxy to get underlying pandas object
                            # cudf.pandas proxies cause issues when passed to original pandas methods
                            unwrapped = _unwrap_cudf_proxy(obj)
                            return orig(unwrapped, *args, **kwargs)
                    return wrapper

                setattr(cls, method_name, make_wrapper(original))

    def _restore_all(self) -> None:
        """Restore all original methods."""
        # Try to import indexer classes for restoration
        try:
            from pandas.core.indexing import _LocIndexer, _iLocIndexer, _AtIndexer, _iAtIndexer
            indexer_map = {
                '_LocIndexer': _LocIndexer,
                '_iLocIndexer': _iLocIndexer,
                '_AtIndexer': _AtIndexer,
                '_iAtIndexer': _iAtIndexer,
            }
        except ImportError:
            indexer_map = {}

        # Try to import GroupBy classes for restoration
        try:
            from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
            groupby_map = {
                'DataFrameGroupBy': DataFrameGroupBy,
                'SeriesGroupBy': SeriesGroupBy,
            }
        except ImportError:
            groupby_map = {}

        for key, original in self._original_methods.items():
            cls_name, method_name = key.split('.', 1)
            if cls_name == 'DataFrame':
                setattr(pd.DataFrame, method_name, original)
            elif cls_name == 'Series':
                setattr(pd.Series, method_name, original)
            elif cls_name == 'pd':
                # Module-level pandas functions
                setattr(pd, method_name, original)
            elif cls_name in indexer_map:
                setattr(indexer_map[cls_name], method_name, original)
            elif cls_name in groupby_map:
                setattr(groupby_map[cls_name], method_name, original)
        self._original_methods.clear()


@contextmanager
def suspend_structural_tracking():
    """
    Context manager to temporarily suspend structural access tracking.

    Use during deepcopy operations to avoid polluting tracking data.

    Example:
        with suspend_structural_tracking():
            df_copy = df.copy()
    """
    was_suspended = StructuralAccessTracker._suspended
    StructuralAccessTracker._suspended = True
    try:
        yield
    finally:
        StructuralAccessTracker._suspended = was_suspended
