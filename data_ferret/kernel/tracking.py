"""
TrackingDict - Variable access tracking for dynamic dependency analysis.

This module provides TrackingDict, a dict subclass that tracks variable access
patterns during cell execution. It records:

- reads_before_writes: Variables read before being written (input dependencies)
- writes: All variables written during execution
- Column-level tracking: Which DataFrame columns are read/written

Architecture:
    TrackingDict wraps the IPython user namespace and intercepts __getitem__
    and __setitem__ to track access patterns. Column-level tracking is handled
    by ColumnAccessTracker, which monkey-patches pandas DataFrame methods.

Usage:
    The kernel enables tracking by replacing user_ns with a TrackingDict:

        user_ns = TrackingDict(user_ns)

    For each cell execution, use the context manager:

        with user_ns.track_execution():
            exec(code, user_ns)
        tracking_data = user_ns.get_tracking_data()

    Or manually:

        user_ns.reset_tracking()
        user_ns.start_column_tracking()
        try:
            exec(code, user_ns)
        finally:
            user_ns.stop_column_tracking()
        tracking_data = user_ns.get_tracking_data()

Note:
    This implementation uses single storage (the parent dict) to avoid
    synchronization issues that occur with dual storage when IPython's
    internal methods bypass our overridden methods.
"""

from contextlib import contextmanager
from typing import Dict, Generator, Set

from .column_tracking import ColumnAccessTracker, walk_dataframes


class TrackingDict(dict):
    """A dict subclass that tracks variable access patterns during cell execution."""

    def __init__(self, initial_ns=None):
        super().__init__()
        if initial_ns is not None:
            # Copy all contents from the existing namespace
            self.update(initial_ns)
        self._initial_keys = set(self.keys())
        self._column_tracker = ColumnAccessTracker()
        self.reset_tracking()

    def reset_tracking(self):
        """Reset tracking state for a new cell execution."""
        self._reads_before_writes = set()
        self._writes = set()
        self._column_tracker.reset()

    def start_column_tracking(self) -> None:
        """Call before cell execution to enable column tracking.

        This registers all existing DataFrames and installs monkey-patches
        on DataFrame methods to track column access.
        """
        # Ensure clean state before tracking (guards against leaked state from
        # previous cells if patches remained installed due to exceptions)
        self._column_tracker.reset()
        # Register all existing DataFrames with their paths
        for path, df in walk_dataframes(self):
            self._column_tracker.register_df(df, path)
        self._column_tracker.install()

    def stop_column_tracking(self) -> None:
        """Call after cell execution to finalize column tracking.

        This re-registers DataFrames (to catch newly created ones),
        resolves tracking data, and restores original DataFrame methods.
        """
        # Re-register DataFrames (new DFs may have been created during execution)
        for path, df in walk_dataframes(self):
            self._column_tracker.register_df(df, path)
        self._column_tracker.uninstall()

    @property
    def column_reads_before_writes(self) -> Dict[str, Set[str]]:
        """Get column-level reads-before-writes, keyed by variable path."""
        return self._column_tracker.resolve_to_paths()

    @property
    def column_writes(self) -> Dict[str, Set[str]]:
        """Get column-level writes, keyed by variable path."""
        return self._column_tracker.resolve_writes_to_paths()

    @property
    def reads_before_writes(self):
        return self._reads_before_writes

    @property
    def writes(self):
        return self._writes

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if key not in self._writes:
            self._reads_before_writes.add(key)
        return val

    def __setitem__(self, key, value):
        # Track writes of non-private variables
        self._writes.add(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        super().__delitem__(key)

    # =========================================================================
    # Context Manager API
    # =========================================================================

    @contextmanager
    def track_execution(self) -> Generator[None, None, None]:
        """
        Context manager for tracking a cell execution.

        Handles the full lifecycle of tracking: reset, start column tracking,
        execute (yield), and stop column tracking. After the context exits,
        call get_tracking_data() to retrieve the captured data.

        Usage:
            with user_ns.track_execution():
                exec(code, user_ns)
            data = user_ns.get_tracking_data()

        Yields:
            None - execute your code inside the with block
        """
        self.reset_tracking()
        self.start_column_tracking()
        try:
            yield
        finally:
            self.stop_column_tracking()

    def get_tracking_data(self) -> "TrackingData":
        """
        Return captured tracking data as a Pydantic model.

        Call this after cell execution (outside the track_execution context)
        to get the captured variable access patterns.

        Returns:
            TrackingData model with reads_before_writes, writes, and column data
        """
        from .checkpoint import is_valid_variable
        from .models import TrackingData

        return TrackingData(
            reads_before_writes=set(
                k
                for k in self._reads_before_writes
                if is_valid_variable(k, self.get(k))
            ),
            writes=set(k for k in self._writes if is_valid_variable(k, self.get(k))),
            column_reads_before_writes={
                k: set(v) for k, v in self.column_reads_before_writes.items()
            },
            column_writes={k: set(v) for k, v in self.column_writes.items()},
        )


# from IPython import get_ipython


# def pre_run_cell(info):
#     ip = get_ipython()
#     # before each cell, just clear out last cell’s logs
#     ip.user_ns.reset_tracking()


# def post_run_cell(result):
#     ip = get_ipython()
#     # after each cell you can inspect:
#     print("reads-before-writes:", ip.user_ns.reads_before_writes)
#     print("writes:", ip.user_ns.writes)
#     print("new_vars:", ip.user_ns.new_vars)


# def load_ipython_extension(ipython):
#     """Load the tracking extension into IPython."""
#     ipython.user_ns = TrackingDict(ipython.user_ns)
#     ipython.events.register("pre_run_cell", pre_run_cell)
#     ipython.events.register("post_run_cell", post_run_cell)
#     print("✅ tracking_ns loaded: now tracking variable accesses.")


# def unload_ipython_extension(ipython):
#     """Unload the tracking extension from IPython."""
#     ipython.events.unregister("pre_run_cell", pre_run_cell)
#     ipython.events.unregister("post_run_cell", post_run_cell)
#     # Restore a plain dict namespace
#     plain = dict(ipython.user_ns)
#     ipython.user_ns = plain
#     print("🛑 tracking_ns unloaded: namespace restored.")
