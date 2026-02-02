"""
Always-On Column Tracking - Prototype

This module demonstrates an alternative approach to column tracking where:
1. Patches are installed ONCE at module import time (or kernel startup)
2. A global flag controls whether tracking is active
3. No install/uninstall overhead per cell execution

Key insight from benchmarks:
- Install/uninstall costs ~500µs per cycle
- Patched operations have ZERO overhead (within noise)
- Therefore, keeping patches always installed is essentially free

Usage:
    # Install once at kernel startup
    ensure_patches_installed()

    # Per-cell execution (fast - just sets a flag)
    tracker = ColumnAccessTrackerAlwaysOn()
    tracker.activate()

    # ... execute cell ...

    tracker.deactivate()
    results = tracker.resolve_to_paths()
"""

import pandas as pd
from typing import Dict, Set, Iterable, Optional, Any
from collections import defaultdict
import threading


# =============================================================================
# Global State
# =============================================================================

# Thread-local active tracker (supports multiple notebooks in same kernel)
_thread_local = threading.local()

# Global patch installation state
_patches_installed = False
_original_methods: Dict[str, Any] = {}


def _get_active_tracker() -> Optional['ColumnAccessTrackerAlwaysOn']:
    """Get the currently active tracker for this thread."""
    return getattr(_thread_local, 'active_tracker', None)


def _set_active_tracker(tracker: Optional['ColumnAccessTrackerAlwaysOn']) -> None:
    """Set the currently active tracker for this thread."""
    _thread_local.active_tracker = tracker


# =============================================================================
# Patch Installation (called once)
# =============================================================================

def ensure_patches_installed() -> None:
    """
    Install patches once. Safe to call multiple times.

    This should be called at kernel startup. After this, all DataFrame
    operations will check the global flag and track if active.
    """
    global _patches_installed, _original_methods

    if _patches_installed:
        return

    # ========== DataFrame.__getitem__ ==========
    _original_methods['DataFrame.__getitem__'] = pd.DataFrame.__getitem__
    original_df_getitem = _original_methods['DataFrame.__getitem__']

    def tracked_df_getitem(df: pd.DataFrame, key):
        tracker = _get_active_tracker()
        if tracker is not None:
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
    _original_methods['DataFrame.__setitem__'] = pd.DataFrame.__setitem__
    original_df_setitem = _original_methods['DataFrame.__setitem__']

    def tracked_df_setitem(df: pd.DataFrame, key, value):
        tracker = _get_active_tracker()
        if tracker is not None:
            if isinstance(key, str):
                tracker.record_write(id(df), [key])
            elif isinstance(key, list):
                str_keys = [k for k in key if isinstance(k, str)]
                if str_keys:
                    tracker.record_write(id(df), str_keys)
        return original_df_setitem(df, key, value)

    pd.DataFrame.__setitem__ = tracked_df_setitem

    # ========== DataFrame.groupby ==========
    _original_methods['DataFrame.groupby'] = pd.DataFrame.groupby
    original_groupby = _original_methods['DataFrame.groupby']

    def tracked_groupby(df: pd.DataFrame, by=None, *args, **kwargs):
        tracker = _get_active_tracker()
        if tracker is not None and by is not None:
            if isinstance(by, str):
                tracker.record_read(id(df), [by])
            elif isinstance(by, list):
                str_keys = [k for k in by if isinstance(k, str)]
                if str_keys:
                    tracker.record_read(id(df), str_keys)
        return original_groupby(df, by=by, *args, **kwargs)

    pd.DataFrame.groupby = tracked_groupby

    # ========== DataFrame.sort_values ==========
    _original_methods['DataFrame.sort_values'] = pd.DataFrame.sort_values
    original_sort_values = _original_methods['DataFrame.sort_values']

    def tracked_sort_values(df: pd.DataFrame, by, **kwargs):
        tracker = _get_active_tracker()
        if tracker is not None:
            cols = [by] if isinstance(by, str) else list(by)
            tracker.record_read(id(df), cols)
        return original_sort_values(df, by, **kwargs)

    pd.DataFrame.sort_values = tracked_sort_values

    # Add more methods as needed...
    # (This is a prototype - full implementation would patch all methods)

    _patches_installed = True


def uninstall_patches() -> None:
    """Restore original methods. Mainly for testing."""
    global _patches_installed, _original_methods

    if not _patches_installed:
        return

    if 'DataFrame.__getitem__' in _original_methods:
        pd.DataFrame.__getitem__ = _original_methods['DataFrame.__getitem__']
    if 'DataFrame.__setitem__' in _original_methods:
        pd.DataFrame.__setitem__ = _original_methods['DataFrame.__setitem__']
    if 'DataFrame.groupby' in _original_methods:
        pd.DataFrame.groupby = _original_methods['DataFrame.groupby']
    if 'DataFrame.sort_values' in _original_methods:
        pd.DataFrame.sort_values = _original_methods['DataFrame.sort_values']

    _original_methods.clear()
    _patches_installed = False


# =============================================================================
# Tracker Class (lightweight, per-execution)
# =============================================================================

class ColumnAccessTrackerAlwaysOn:
    """
    Lightweight tracker that uses globally-installed patches.

    Unlike ColumnAccessTracker, this class does NOT install/uninstall patches.
    Instead, it activates/deactivates itself as the global active tracker.

    This makes activate()/deactivate() nearly instant (just setting a pointer).
    """

    # Class-level suspension flag (for deepcopy operations)
    _suspended = False

    def __init__(self):
        self._reads_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._writes_by_id: Dict[int, Set[str]] = defaultdict(set)
        self._id_to_path: Dict[int, str] = {}
        self._active = False

    def activate(self) -> None:
        """Activate this tracker. Very fast - just sets a pointer."""
        if self._active:
            return
        ensure_patches_installed()  # Ensure patches are there (idempotent)
        _set_active_tracker(self)
        self._active = True

    def deactivate(self) -> None:
        """Deactivate this tracker. Very fast - just clears a pointer."""
        if not self._active:
            return
        if _get_active_tracker() is self:
            _set_active_tracker(None)
        self._active = False

    def register_df(self, df: pd.DataFrame, path: str) -> None:
        """Register a DataFrame with its namespace path."""
        self._id_to_path[id(df)] = path

    def record_read(self, df_id: int, columns: Iterable[str]) -> None:
        """Record column reads for a DataFrame by ID."""
        if ColumnAccessTrackerAlwaysOn._suspended:
            return
        for col in columns:
            if col not in self._writes_by_id[df_id]:
                self._reads_by_id[df_id].add(col)

    def record_write(self, df_id: int, columns: Iterable[str]) -> None:
        """Record column writes for a DataFrame by ID."""
        if ColumnAccessTrackerAlwaysOn._suspended:
            return
        for col in columns:
            self._writes_by_id[df_id].add(col)

    def resolve_to_paths(self) -> Dict[str, Set[str]]:
        """Convert id-based tracking to path-based column_rbw."""
        result: Dict[str, Set[str]] = {}
        all_df_ids = set(self._reads_by_id.keys()) | set(self._writes_by_id.keys())

        for df_id in all_df_ids:
            if df_id not in self._id_to_path:
                continue
            path = self._id_to_path[df_id]
            read_cols = self._reads_by_id.get(df_id, set())
            result[path] = read_cols.copy()

        return result

    def reset(self) -> None:
        """Reset tracking for new cell execution."""
        self._reads_by_id.clear()
        self._writes_by_id.clear()
        self._id_to_path.clear()


# =============================================================================
# Context Manager for easy usage
# =============================================================================

from contextlib import contextmanager

@contextmanager
def track_columns():
    """
    Context manager for column tracking with always-on patches.

    Usage:
        tracker = ColumnAccessTrackerAlwaysOn()
        for path, df in walk_dataframes(namespace):
            tracker.register_df(df, path)

        with track_columns_active(tracker):
            # execute user code
            pass

        results = tracker.resolve_to_paths()
    """
    tracker = ColumnAccessTrackerAlwaysOn()
    tracker.activate()
    try:
        yield tracker
    finally:
        tracker.deactivate()


# =============================================================================
# Benchmark comparison
# =============================================================================

if __name__ == '__main__':
    import time
    import statistics
    import numpy as np
    from flowbook.kernel_support.column_tracking import ColumnAccessTracker, walk_dataframes

    print("=" * 60)
    print("Always-On vs Current Approach Comparison")
    print("=" * 60)

    # Create test namespace
    ns = {f'df_{i}': pd.DataFrame(np.random.randn(100, 10),
                                   columns=[f'col_{j}' for j in range(10)])
          for i in range(10)}

    n_iterations = 100

    # Current approach
    def current_approach():
        tracker = ColumnAccessTracker()
        for path, df in walk_dataframes(ns):
            tracker.register_df(df, path)
        tracker.install()
        for key in list(ns.keys())[:5]:
            _ = ns[key]['col_0']
        tracker.uninstall()
        return tracker.resolve_to_paths()

    # Always-on approach
    ensure_patches_installed()

    def always_on_approach():
        tracker = ColumnAccessTrackerAlwaysOn()
        for path, df in walk_dataframes(ns):
            tracker.register_df(df, path)
        tracker.activate()
        for key in list(ns.keys())[:5]:
            _ = ns[key]['col_0']
        tracker.deactivate()
        return tracker.resolve_to_paths()

    # Warmup
    for _ in range(10):
        current_approach()
        always_on_approach()

    # Benchmark current
    current_times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        current_approach()
        end = time.perf_counter_ns()
        current_times.append((end - start) / 1000)  # µs

    # Benchmark always-on
    always_on_times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        always_on_approach()
        end = time.perf_counter_ns()
        always_on_times.append((end - start) / 1000)  # µs

    # Cleanup
    uninstall_patches()

    current_mean = statistics.mean(current_times)
    always_on_mean = statistics.mean(always_on_times)

    print(f"\nCurrent approach:   {current_mean:.1f} µs/cycle")
    print(f"Always-on approach: {always_on_mean:.1f} µs/cycle")
    print(f"Speedup:            {current_mean/always_on_mean:.2f}x faster")
    print(f"Savings:            {current_mean - always_on_mean:.1f} µs ({(1 - always_on_mean/current_mean)*100:.1f}%)")
