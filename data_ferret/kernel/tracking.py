"""
TrackingDict is a dictionary that tracks reads-before-write, new_vars, and writes.

This implementation uses single storage (the parent dict) to avoid synchronization
issues that occur with dual storage when IPython's internal methods bypass our
overridden methods.

Also supports column-level tracking for DataFrames via ColumnAccessTracker.
"""


import sys
from typing import Dict, Set

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
    def column_rbw(self) -> Dict[str, Set[str]]:
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
