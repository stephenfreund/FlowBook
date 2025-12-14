"""
TrackingDict - Variable access tracking for dynamic dependency analysis.

This module provides TrackingDict, a dict subclass that tracks variable access
patterns during cell execution. It records:

- reads_before_writes: Variables read before being written (input dependencies)
- writes: All variables written during execution
- Column-level tracking: Which DataFrame columns are read/written

Architecture:
    TrackingDict uses a DELEGATION pattern - it delegates all storage to an
    underlying namespace (user_global_ns) while intercepting access to track
    read/write patterns. This ensures:

    1. Single source of truth: All data lives in user_global_ns
    2. Python scoping works: List comprehensions and functions find variables
       because they're stored in the real globals namespace
    3. No synchronization issues: No shadow namespace to keep in sync

    Column-level tracking is handled by ColumnAccessTracker, which monkey-patches
    pandas DataFrame methods.

Usage:
    The kernel enables tracking by wrapping user_global_ns:

        tracking_dict = TrackingDict(shell.user_global_ns)
        shell.user_ns = tracking_dict

    For each cell execution, use the context manager:

        with tracking_dict.track_execution():
            exec(code, tracking_dict)
        tracking_data = tracking_dict.get_tracking_data()
"""

from contextlib import contextmanager
from typing import Dict, Generator, Optional, Set

from .column_tracking import ColumnAccessTracker, walk_dataframes, walk_pandas_objects
from .structural_tracking import StructuralAccessTracker, StructuralTrackingMode


class TrackingDict(dict):
    """
    A dict that delegates storage to an underlying namespace while tracking access.

    This inherits from dict for isinstance compatibility, but ALL storage is
    delegated to _real_ns. The dict inheritance is just for type compatibility
    with IPython internals that check isinstance(user_ns, dict).

    Key design: We don't store data ourselves - we delegate to user_global_ns.
    This means list comprehensions and functions automatically find variables
    because they look in user_global_ns, which IS where we store everything.
    """

    # Use __slots__ to prevent attribute access from going through __getattr__
    # Actually, we can't use __slots__ with dict subclass easily, so we use
    # a prefix convention and careful attribute access

    def __init__(self, real_ns: Optional[dict] = None):
        """
        Initialize TrackingDict as a wrapper around the real namespace.

        Args:
            real_ns: The real namespace to delegate storage to. This should be
                     shell.user_global_ns (which is user_module.__dict__).
                     If None, creates a new empty dict for storage.
        """
        # Don't call super().__init__() with data - we delegate storage
        super().__init__()

        # Use object.__setattr__ to avoid triggering our __setitem__
        # If no real_ns provided, create one (for tests and standalone use)
        object.__setattr__(self, '_real_ns', real_ns if real_ns is not None else {})
        object.__setattr__(self, '_reads_before_writes', set())
        object.__setattr__(self, '_writes', set())
        object.__setattr__(self, '_tracking_enabled', True)  # Track by default
        object.__setattr__(self, '_column_tracker', ColumnAccessTracker())
        object.__setattr__(self, '_structural_tracker', StructuralAccessTracker())

    # =========================================================================
    # Core dict protocol - delegate to _real_ns
    # =========================================================================

    def __getitem__(self, key):
        value = self._real_ns[key]
        if self._tracking_enabled and key not in self._writes:
            self._reads_before_writes.add(key)
        return value

    def __setitem__(self, key, value):
        if self._tracking_enabled:
            self._writes.add(key)
        self._real_ns[key] = value

    def __delitem__(self, key):
        del self._real_ns[key]

    def __contains__(self, key):
        return key in self._real_ns

    def __iter__(self):
        return iter(self._real_ns)

    def __len__(self):
        return len(self._real_ns)

    def __repr__(self):
        return f"TrackingDict({repr(self._real_ns)})"

    # =========================================================================
    # Dict methods - all delegate to _real_ns
    # =========================================================================

    def keys(self):
        return self._real_ns.keys()

    def values(self):
        return self._real_ns.values()

    def items(self):
        return self._real_ns.items()

    def get(self, key, default=None):
        """Get with default - does NOT track reads (used by IPython internals)."""
        return self._real_ns.get(key, default)

    def update(self, other=None, **kwargs):
        """Update the namespace. Uses __setitem__ to ensure tracking."""
        if other is not None:
            if hasattr(other, 'items'):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *args):
        try:
            value = self[key]  # Track the read
            del self[key]
            return value
        except KeyError:
            if args:
                return args[0]
            raise

    def popitem(self):
        key, value = self._real_ns.popitem()
        return key, value

    def clear(self):
        self._real_ns.clear()

    def copy(self):
        return dict(self._real_ns)

    # =========================================================================
    # Tracking control
    # =========================================================================

    def reset_tracking(self):
        """Reset tracking state for a new cell execution."""
        self._reads_before_writes.clear()
        self._writes.clear()
        self._column_tracker.reset()
        self._structural_tracker.reset()

    @property
    def reads_before_writes(self) -> Set[str]:
        return self._reads_before_writes

    @property
    def writes(self) -> Set[str]:
        return self._writes

    @property
    def column_reads_before_writes(self) -> Dict[str, Set[str]]:
        """Get column-level reads-before-writes, keyed by variable path."""
        return self._column_tracker.resolve_to_paths()

    @property
    def column_writes(self) -> Dict[str, Set[str]]:
        """Get column-level writes, keyed by variable path."""
        return self._column_tracker.resolve_writes_to_paths()

    @property
    def structural_reads(self) -> Dict[str, Set[str]]:
        """Get structural attribute reads, keyed by variable path."""
        return self._structural_tracker.resolve_to_paths()

    @property
    def structural_tracking_mode(self) -> StructuralTrackingMode:
        """Get current structural tracking mode."""
        return self._structural_tracker.mode

    def set_structural_tracking_mode(self, mode: str) -> None:
        """
        Set structural tracking mode.

        Args:
            mode: One of "off", "warn", "enforce"
        """
        self._structural_tracker.set_mode(mode)

    # =========================================================================
    # Column tracking
    # =========================================================================

    def start_column_tracking(self) -> None:
        """Call before cell execution to enable column and structural tracking.

        This registers all existing DataFrames and installs monkey-patches
        on DataFrame methods to track column access and structural attribute access.
        """
        # Ensure clean state before tracking (guards against leaked state from
        # previous cells if patches remained installed due to exceptions)
        self._column_tracker.reset()
        self._structural_tracker.reset()
        # Register all existing DataFrames with their paths (for column tracking)
        for path, df in walk_dataframes(self._real_ns):
            self._column_tracker.register_df(df, path)
        # Register all pandas objects (DataFrames AND Series) for structural tracking
        for path, obj in walk_pandas_objects(self._real_ns):
            self._structural_tracker.register(obj, path)
        self._column_tracker.install()
        self._structural_tracker.install()

    def stop_column_tracking(self) -> None:
        """Call after cell execution to finalize column and structural tracking.

        This re-registers DataFrames (to catch newly created ones),
        resolves tracking data, and restores original DataFrame methods.
        """
        # Re-register DataFrames (new DFs may have been created during execution)
        for path, df in walk_dataframes(self._real_ns):
            self._column_tracker.register_df(df, path)
        # Re-register all pandas objects for structural tracking
        for path, obj in walk_pandas_objects(self._real_ns):
            self._structural_tracker.register(obj, path)
        self._column_tracker.uninstall()
        self._structural_tracker.uninstall()

    # =========================================================================
    # Context Manager API
    # =========================================================================

    @contextmanager
    def track_execution(self) -> Generator[None, None, None]:
        """
        Context manager for tracking a cell execution.

        Handles the full lifecycle of tracking: reset, enable tracking,
        start column tracking, execute (yield), stop column tracking,
        disable tracking. After the context exits, call get_tracking_data()
        to retrieve the captured data.

        Usage:
            with user_ns.track_execution():
                exec(code, user_ns)
            data = user_ns.get_tracking_data()

        Yields:
            None - execute your code inside the with block
        """
        self.reset_tracking()
        self._tracking_enabled = True
        self.start_column_tracking()
        try:
            yield
        finally:
            self.stop_column_tracking()
            self._tracking_enabled = False

    @contextmanager
    def suspended(self):
        """
        Temporarily suspend all tracking.

        Use this to prevent reads/writes during infrastructure code
        (like magic commands) from being recorded.

        Usage:
            with user_ns.suspended():
                # do stuff that shouldn't be tracked
                pass
        """
        prev_enabled = self._tracking_enabled
        self._tracking_enabled = False
        try:
            yield
        finally:
            self._tracking_enabled = prev_enabled

    def get_tracking_data(self) -> "TrackingData":
        """
        Return captured tracking data as a Pydantic model.

        Call this after cell execution (outside the track_execution context)
        to get the captured variable access patterns.

        Returns:
            TrackingData model with reads_before_writes, writes, column data, and structural reads
        """
        from .checkpoint import is_valid_variable
        from .models import TrackingData

        # Filter column reads: exclude DataFrames that were WRITTEN in this cell
        # If a variable like `total_per_day` was created in this cell, column reads
        # from it are not "reads before writes" because the whole variable is new
        column_rbw = {
            k: set(v)
            for k, v in self.column_reads_before_writes.items()
            if k not in self._writes  # Exclude variables that were written
        }

        # Same for structural reads: exclude variables that were written
        struct_reads = {
            k: set(v)
            for k, v in self.structural_reads.items()
            if k not in self._writes
        }

        return TrackingData(
            reads_before_writes=set(
                k
                for k in self._reads_before_writes
                if is_valid_variable(k, self._real_ns.get(k))
            ),
            writes=set(
                k
                for k in self._writes
                if is_valid_variable(k, self._real_ns.get(k))
            ),
            column_reads_before_writes=column_rbw,
            column_writes={k: set(v) for k, v in self.column_writes.items()},
            structural_reads=struct_reads,
        )
