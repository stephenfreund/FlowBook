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

Performance optimization (always-on pattern):
    - Patches are installed once at first use (not per-cell)
    - Per-cell uses activate()/deactivate() (~0.1µs each)
    - DataFrames are registered lazily when accessed from namespace
    - Eliminates namespace walking at start/stop time
    - Overhead reduced from ~760µs to ~10-50µs per cell (95% reduction)

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

import pandas as pd

from flowbook.util.output import timer
from flowbook.kernel_support.column_tracking import ColumnAccessTracker, walk_dataframes, walk_pandas_objects
from flowbook.kernel_support.structural_tracking import StructuralAccessTracker, StructuralTrackingMode


def _is_ipython_result_var(key: str) -> bool:
    """Check if key is an IPython auto-result variable.

    These variables are automatically created by IPython to store cell outputs
    and history. We should not let them overwrite real variable paths in column
    tracking, because that would break NoReadAndWrite detection (e.g., if cell
    reads 'train' and writes to 'train', but 'train' gets re-registered as '_3'
    when IPython stores the cell result, we'd miss the read-write conflict).

    IPython special variables include:
    - _  : last output
    - __ : second-to-last output
    - ___: third-to-last output
    - _1, _2, etc.: numbered output history
    - _i, _ii, _iii: input history (strings, not DataFrames, but check anyway)
    - _oh, _ih: output/input history dicts
    """
    if not key.startswith('_'):
        return False
    # Single underscore
    if key == '_':
        return True
    # Double/triple underscore (__, ___)
    if key in ('__', '___'):
        return True
    # Numbered outputs: _1, _2, _3, etc.
    if len(key) > 1 and key[1:].isdigit():
        return True
    # Input history: _i, _ii, _iii
    if key in ('_i', '_ii', '_iii'):
        return True
    # History dicts: _ih, _oh
    if key in ('_ih', '_oh'):
        return True
    return False


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
        real_ns_actual = real_ns if real_ns is not None else {}
        object.__setattr__(self, '_real_ns', real_ns_actual)
        object.__setattr__(self, '_reads_before_writes', set())
        object.__setattr__(self, '_writes', set())
        object.__setattr__(self, '_tracking_enabled', True)  # Track by default
        # Pass namespace reference to trackers for lazy fallback walks
        object.__setattr__(self, '_column_tracker', ColumnAccessTracker(namespace_ref=real_ns_actual))
        object.__setattr__(self, '_structural_tracker', StructuralAccessTracker(namespace_ref=real_ns_actual))

    # =========================================================================
    # Core dict protocol - delegate to _real_ns
    # =========================================================================

    def __getitem__(self, key):
        value = self._real_ns[key]
        if self._tracking_enabled:
            if key not in self._writes:
                self._reads_before_writes.add(key)
            # Lazy registration: register DataFrames/Series when accessed from namespace
            # This eliminates the need to walk the namespace at start/stop time
            if isinstance(value, pd.DataFrame):
                self._column_tracker.register_df(value, key)
                self._structural_tracker.register(value, key)
            elif isinstance(value, pd.Series):
                self._structural_tracker.register(value, key)
        return value

    def __setitem__(self, key, value):
        if self._tracking_enabled:
            self._writes.add(key)
            # Lazy registration: register DataFrames/Series when assigned to namespace
            # This eliminates the need to walk the namespace at start/stop time
            # Skip IPython result variables (_1, _2, etc.) to avoid overwriting real paths
            if isinstance(value, pd.DataFrame):
                if not _is_ipython_result_var(key):
                    self._column_tracker.register_df(value, key)
                    self._structural_tracker.register(value, key)
            elif isinstance(value, pd.Series):
                if not _is_ipython_result_var(key):
                    self._structural_tracker.register(value, key)
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

        Uses the always-on pattern for performance:
        - Patches are installed once (idempotent) and stay installed across cells
        - Per-cell: just reset tracking state and activate this tracker (~10µs total)
        - DataFrames are registered lazily when accessed from namespace (no walking)
        """
        # Reset tracking state for new cell
        with timer(key="tracking:reset", message="Track reset"):
            self._column_tracker.reset()
            self._structural_tracker.reset()

        # Activate tracking for this cell execution
        # Patches are installed idempotently (first call only)
        with timer(key="tracking:activate_trackers", message="Activate trackers"):
            self._column_tracker.activate()
            self._structural_tracker.activate()

    def stop_column_tracking(self) -> None:
        """Call after cell execution to finalize column and structural tracking.

        Uses the always-on pattern for performance:
        - Just deactivates tracking (~0.1µs)
        - Does NOT uninstall patches (they stay for next cell)
        - Does NOT walk namespace (resolution uses lazy fallback if needed)
        """
        # Deactivate tracking (patches stay installed for next cell)
        with timer(key="tracking:deactivate_trackers", message="Deactivate trackers"):
            self._column_tracker.deactivate()
            self._structural_tracker.deactivate()

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
        from flowbook.kernel_support.checkpoint import is_valid_variable
        from flowbook.kernel_support.models import TrackingData

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
