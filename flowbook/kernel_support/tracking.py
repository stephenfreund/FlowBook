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

import os
import types
import pandas as pd

from flowbook.util.output import timer
from flowbook.kernel_support.column_tracking import (
    ColumnAccessTracker, walk_dataframes, walk_pandas_objects, _is_dataframe, _is_series,
)
from flowbook.kernel_support.structural_tracking import StructuralAccessTracker, StructuralTrackingMode


class UncopyableReadError(NameError):
    """Raised when user code reads a variable whose value could not be
    checkpointed.

    FlowBook cannot restore such a value on rollback, so allowing reads
    would let unreproducible state flow into downstream cells. The variable
    stays alive in the namespace; rebinding it (or deleting it) lifts the
    block. See FORMAL_DEVELOPMENT.md (Uncopyable Variables).
    """


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


_UNBOUND = object()  # sentinel: the name had no binding before this cell
_MISSING = object()  # sentinel: no entry in a mapping (raw-storage mirror)
_NOT_DIRTY = object()  # sentinel: the raw storage adds nothing to _real_ns for a name

# Names TrackingDict keeps in its own dict storage between cells. CPython reads
# them from a function's globals with raw dict access when it creates a function
# (__module__ from __name__, and __builtins__), and functions defined in a cell
# keep this TrackingDict as their globals after the cell ends.
_RAW_PERSISTENT = ("__name__", "__builtins__")


def imports_tracked() -> bool:
    """Module bindings participate in read/write tracking unless FLOWBOOK_TRACK_IMPORTS=0."""
    return os.environ.get("FLOWBOOK_TRACK_IMPORTS", "1") != "0"


def rollback_module_bindings(tracking_dict) -> list:
    """Undo the module (re)bindings of the cell just executed through tracking_dict.

    Modules are not checkpointed, so restoring the pre-execution checkpoint
    cannot remove an import a rejected cell introduced; without this, a later
    cell would read `json` as an ambient name although no committed cell
    imports it. Uses the previous-binding snapshot taken on first write, so a
    name that was unbound is deleted and a name that pointed at another module
    is pointed back. Returns the names touched.
    """
    touched = []
    real = tracking_dict._real_ns
    for name, prev in list(tracking_dict._prev_bindings.items()):
        now = real.get(name, _UNBOUND)
        if not (isinstance(now, types.ModuleType) or isinstance(prev, types.ModuleType)):
            continue
        if now is prev:
            continue
        if prev is _UNBOUND:
            real.pop(name, None)
        else:
            real[name] = prev
        touched.append(name)
    return touched


class TrackingDict(dict):
    """
    A dict that delegates storage to an underlying namespace while tracking access.

    This inherits from dict for isinstance compatibility, but ALL storage is
    delegated to _real_ns. The dict inheritance is just for type compatibility
    with IPython internals that check isinstance(user_ns, dict).

    Key design: We don't store data ourselves - we delegate to user_global_ns.
    This means list comprehensions and functions automatically find variables
    because they look in user_global_ns, which IS where we store everything.

    Exception: when a TrackingDict is used as exec *globals*, CPython bypasses
    the mapping protocol in a few places and uses the dict's own storage. See
    "Raw-storage mirror" below: while a cell runs, that storage mirrors
    _real_ns so those accesses see (and can change) the real values.
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
        # First-write-in-cell snapshot of a name's previous binding, so a
        # rebinding to the identical object (re-running `import pandas as pd`)
        # can be recognized as a no-op write. See get_tracking_data().
        object.__setattr__(self, '_prev_bindings', {})
        object.__setattr__(self, '_writes', set())
        object.__setattr__(self, '_tracking_enabled', True)  # Track by default
        # Variables whose values could not be checkpointed (name -> type
        # description). Tracked reads of these raise UncopyableReadError;
        # rebinding or deleting the name lifts the block.
        object.__setattr__(self, '_blocked_reads', {})
        # Raw-storage mirror (see begin_mirror): nesting depth of mirrored
        # run_code calls, and the snapshot of _real_ns the mirror started from.
        object.__setattr__(self, '_mirror_depth', 0)
        object.__setattr__(self, '_mirror_base', {})
        self._reset_raw()
        # Pass namespace reference to trackers for lazy fallback walks
        object.__setattr__(self, '_column_tracker', ColumnAccessTracker(namespace_ref=real_ns_actual))
        object.__setattr__(self, '_structural_tracker', StructuralAccessTracker(namespace_ref=real_ns_actual))

    # =========================================================================
    # Core dict protocol - delegate to _real_ns
    # =========================================================================

    def _check_blocked(self, key) -> None:
        """Raise if key is read-blocked as uncopyable (tracking enabled)."""
        if self._tracking_enabled and key in self._blocked_reads:
            # Paper semantics for non-checkpointable objects: warn once and
            # block all subsequent reads (the value cannot be restored on
            # rollback, so reads would leak unreproducible state downstream).
            type_desc = self._blocked_reads[key]
            raise UncopyableReadError(
                f"FlowBook blocked this read of '{key}': its value "
                f"({type_desc}) cannot be checkpointed, so state depending "
                f"on it cannot be restored or reproduced. Rebind '{key}' to "
                f"a new value (or delete it) to use the name again."
            )

    def _track_read(self, key, value) -> None:
        """Record a read of key (in-cell masked) and lazily register pandas
        objects for column/structural tracking."""
        if key not in self._writes:
            self._reads_before_writes.add(key)
        # IPython's result variables (_, __, ___, _N) alias objects that real
        # variables hold, and IPython's display hook reads _/__/___ through this
        # dict whenever a cell ends in an expression. Registering the object under
        # that name would replace its real path (e.g. X), and the column reads of X
        # would then be dropped when the display hook writes ___. Same guard as
        # __setitem__.
        if _is_ipython_result_var(key):
            return
        # Lazy registration: register DataFrames/Series when accessed from
        # namespace. This eliminates namespace walking at start/stop time.
        if _is_dataframe(value):
            self._column_tracker.register_df(value, key)
            self._structural_tracker.register(value, key)
        elif _is_series(value):
            self._structural_tracker.register(value, key)

    def __getitem__(self, key):
        self._check_blocked(key)
        self._apply_raw_key(key)
        value = self._real_ns[key]
        if self._tracking_enabled:
            self._track_read(key, value)
        return value

    def __setitem__(self, key, value):
        # Rebinding replaces the uncopyable value — lift the read block.
        self._blocked_reads.pop(key, None)
        if self._tracking_enabled:
            if key not in self._prev_bindings:
                self._prev_bindings[key] = self._real_ns.get(key, _UNBOUND)
            self._writes.add(key)
            # Lazy registration: register DataFrames/Series when assigned to namespace
            # This eliminates the need to walk the namespace at start/stop time
            # Skip IPython result variables (_1, _2, etc.) to avoid overwriting real paths
            if _is_dataframe(value):
                if not _is_ipython_result_var(key):
                    self._column_tracker.register_df(value, key)
                    self._structural_tracker.register(value, key)
                    # Record provenance: all columns attributed to current cell
                    if self._column_tracker._cell_id is not None:
                        from flowbook.kernel_support.column_provenance import DataFrameProvenanceTracker
                        DataFrameProvenanceTracker.record_var_write(value, self._column_tracker._cell_id)
            elif _is_series(value):
                if not _is_ipython_result_var(key):
                    self._structural_tracker.register(value, key)
        self._real_ns[key] = value
        if self._mirror_depth:
            dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._apply_raw_key(key)
        if self._tracking_enabled and key not in self._prev_bindings:
            self._prev_bindings[key] = self._real_ns.get(key, _UNBOUND)
        del self._real_ns[key]
        dict.pop(self, key, None)
        self._blocked_reads.pop(key, None)
        if self._tracking_enabled:
            # `del x` is a write to x in the formal model: it changes what
            # downstream readers of x observe. Recording it puts x in the
            # diff's keys_to_include (OPT_ACCESSED_VARS_ONLY), so the diff
            # reports "Variable was removed" and staleness propagates to
            # readers of x. Recorded after the delete so a KeyError does
            # not record a phantom write.
            self._writes.add(key)

    def __contains__(self, key):
        self._apply_raw_key(key)
        return key in self._real_ns

    def __iter__(self):
        return iter(self._view())

    def __len__(self):
        return len(self._view())

    def __repr__(self):
        return f"TrackingDict({repr(self._view())})"

    # =========================================================================
    # Dict methods - all delegate to _real_ns
    # =========================================================================

    def keys(self):
        # Names-only access: reveals which variables exist, not their
        # values. There is no location type for the namespace key set, so
        # this is intentionally untracked (documented escape hatch).
        return self._view().keys()

    def _track_all_reads(self) -> None:
        """Record reads of every user variable (values()/items() reveal all
        values — the honest read set is 'everything')."""
        from flowbook.kernel_support.checkpoint import is_valid_variable
        for key, value in list(self._view().items()):
            if is_valid_variable(key, value):
                self._track_read(key, value)

    def values(self):
        if self._tracking_enabled:
            # Iterating values reads every variable's value (audit:
            # namespace iteration escaped read tracking).
            self._track_all_reads()
        return self._view().values()

    def items(self):
        if self._tracking_enabled:
            # Iterating items reads every variable's value (audit:
            # namespace iteration escaped read tracking).
            self._track_all_reads()
        return self._view().items()

    def get(self, key, default=None):
        """Get with default — tracked like __getitem__ (audit:
        globals().get('x') escaped read tracking and the uncopyable
        read block)."""
        self._check_blocked(key)
        self._apply_raw_key(key)
        if key not in self._real_ns:
            return default
        value = self._real_ns[key]
        if self._tracking_enabled:
            self._track_read(key, value)
        return value

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
        dict.pop(self, key, None)
        return key, value

    def clear(self):
        self._real_ns.clear()
        if self._mirror_depth:
            dict.clear(self)

    def copy(self):
        return dict(self._view())

    # =========================================================================
    # Raw-storage mirror (TrackingDict as exec globals)
    # =========================================================================
    #
    # The kernel runs cells with this TrackingDict as exec *globals* as well as
    # locals, so reads from nested scopes (LOAD_GLOBAL in functions, lambdas,
    # comprehensions) go through __getitem__ and are tracked. But CPython does
    # not always use the mapping protocol on a globals dict subclass. It uses
    # the dict's own storage directly for
    #   - a class body's fallback from its namespace to globals (LOAD_NAME),
    #   - `global x; x = ...` and `global x; del x` in a function
    #     (STORE_GLOBAL / DELETE_GLOBAL),
    #   - the __module__ of a new function (globals['__name__']) and of a new
    #     class (LOAD_NAME __name__), and a new function's __builtins__.
    # With empty storage those fail: a class body cannot see any global, a
    # `global` write is lost, and notebook functions and classes get
    # __module__ None / 'builtins' (so their instances cannot be pickled).
    #
    # So while a cell runs (begin_mirror .. end_mirror) the dict's own storage
    # holds a copy of _real_ns, kept current by __setitem__ / __delitem__. A
    # name whose raw value is neither the value the mirror started from nor
    # the value in _real_ns was changed by a raw write: reads see the raw
    # value: the first access through the mapping protocol applies it to
    # _real_ns as a tracked write (_apply_raw_key), so the write is recorded
    # before any later read of the name, and end_mirror applies the rest. Between cells the storage holds only _RAW_PERSISTENT
    # (plus any raw write made by a notebook function called outside a cell,
    # which the next begin_mirror applies untracked); it is never a stale
    # copy, because FlowBook restores checkpoints into _real_ns directly.
    # Class-body reads that reach the raw storage are not tracked; the
    # kernel's class-body AST transformer (class_scope.py) routes them
    # through __getitem__ instead.

    def _raw_override(self, key):
        """What the raw storage says about key beyond _real_ns.

        _NOT_DIRTY when a raw write has not touched key (read _real_ns),
        _MISSING when a raw delete removed it, otherwise the raw value.
        """
        raw = dict.get(self, key, _MISSING)
        if raw is self._mirror_base.get(key, _MISSING):
            return _NOT_DIRTY
        if raw is self._real_ns.get(key, _MISSING):
            return _NOT_DIRTY
        return raw

    def _apply_raw_key(self, key) -> None:
        """Apply a pending raw write/delete of key to _real_ns (tracked like any write)."""
        pending = self._raw_override(key)
        if pending is _NOT_DIRTY:
            return
        if pending is _MISSING:
            self._record_raw_delete(key)
        else:
            self[key] = pending
        if not self._mirror_depth:
            # Between cells the raw storage must not keep copies (FlowBook
            # restores checkpoints into _real_ns directly).
            dict.pop(self, key, None)
            if key in _RAW_PERSISTENT and key in self._real_ns:
                dict.__setitem__(self, key, self._real_ns[key])
                self._mirror_base[key] = self._real_ns[key]

    def _record_raw_delete(self, key) -> None:
        if self._tracking_enabled and key not in self._prev_bindings:
            self._prev_bindings[key] = self._real_ns.get(key, _UNBOUND)
        self._real_ns.pop(key, None)
        self._blocked_reads.pop(key, None)
        if self._tracking_enabled:
            self._writes.add(key)

    def _raw_changes(self):
        """(written, deleted): raw writes and raw deletes not yet in _real_ns."""
        base, real = self._mirror_base, self._real_ns
        written = {}
        for key, raw in dict.items(self):
            if raw is not base.get(key, _MISSING) and raw is not real.get(key, _MISSING):
                written[key] = raw
        deleted = [key for key, b in base.items()
                   if not dict.__contains__(self, key) and real.get(key, _MISSING) is b]
        return written, deleted

    def _view(self) -> dict:
        """The namespace as user code sees it: _real_ns plus pending raw changes."""
        written, deleted = self._raw_changes()
        if not written and not deleted:
            return self._real_ns
        view = dict(self._real_ns)
        view.update(written)
        for key in deleted:
            view.pop(key, None)
        return view

    def _apply_raw_changes(self) -> None:
        """Apply pending raw writes/deletes to _real_ns through the tracked paths."""
        written, deleted = self._raw_changes()
        for key in deleted:
            self._record_raw_delete(key)
        for key, value in written.items():
            self[key] = value

    def _reset_raw(self) -> None:
        dict.clear(self)
        for key in _RAW_PERSISTENT:
            if key in self._real_ns:
                dict.__setitem__(self, key, self._real_ns[key])
        object.__setattr__(self, '_mirror_base', dict(dict.items(self)))

    def begin_mirror(self) -> None:
        """Start mirroring _real_ns into the raw storage (before running cell code).

        Nested calls (a magic that runs code inside a cell) only count depth.
        """
        object.__setattr__(self, '_mirror_depth', self._mirror_depth + 1)
        if self._mirror_depth > 1:
            return
        # Raw writes made between cells (a notebook function with a `global`
        # statement called from outside any cell) belong to no cell.
        with self.suspended():
            self._apply_raw_changes()
        dict.clear(self)
        dict.update(self, self._real_ns)
        object.__setattr__(self, '_mirror_base', dict(self._real_ns))

    def end_mirror(self) -> None:
        """Apply the cell's raw writes to _real_ns (tracked) and stop mirroring."""
        if not self._mirror_depth:
            return
        object.__setattr__(self, '_mirror_depth', self._mirror_depth - 1)
        if self._mirror_depth:
            return
        try:
            self._apply_raw_changes()
        finally:
            self._reset_raw()

    # =========================================================================
    # Uncopyable variable read blocking
    # =========================================================================

    def block_variable(self, name: str, type_desc: str = "uncopyable object") -> None:
        """Block tracked reads of a variable whose value cannot be checkpointed.

        The value stays alive in the namespace (open files keep working,
        displayed figures stay displayed), but user-code reads raise
        UncopyableReadError until the name is rebound or deleted.
        Idempotent.
        """
        self._blocked_reads[name] = type_desc

    def unblock_variable(self, name: str) -> None:
        """Lift the read block for a variable (no-op if not blocked)."""
        self._blocked_reads.pop(name, None)

    @property
    def blocked_variables(self) -> Set[str]:
        """Names currently read-blocked as uncopyable."""
        return set(self._blocked_reads)

    # =========================================================================
    # Tracking control
    # =========================================================================

    def reset_tracking(self):
        """Reset tracking state for a new cell execution."""
        self._reads_before_writes.clear()
        self._writes.clear()
        self._prev_bindings.clear()
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

    def start_column_tracking(self, cell_id: Optional[str] = None) -> None:
        """Call before cell execution to enable column and structural tracking.

        Uses the always-on pattern for performance:
        - Patches are installed once (idempotent) and stay installed across cells
        - Per-cell: just reset tracking state and activate this tracker (~10µs total)
        - DataFrames are registered lazily when accessed from namespace (no walking)

        Args:
            cell_id: The ID of the cell being executed. Passed to column
                tracker for provenance recording.
        """
        # Reset tracking state for new cell
        with timer(key="tracking:reset", message="Track reset"):
            self._column_tracker.reset()
            self._structural_tracker.reset()

        # Activate tracking for this cell execution
        # Patches are installed idempotently (first call only)
        with timer(key="tracking:activate_trackers", message="Activate trackers"):
            self._column_tracker.activate(cell_id=cell_id)
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
    def track_execution(self, cell_id: Optional[str] = None) -> Generator[None, None, None]:
        """
        Context manager for tracking a cell execution.

        Handles the full lifecycle of tracking: reset, enable tracking,
        start column tracking, execute (yield), stop column tracking,
        disable tracking. After the context exits, call get_tracking_data()
        to retrieve the captured data.

        Usage:
            with user_ns.track_execution(cell_id="abc"):
                exec(code, user_ns)
            data = user_ns.get_tracking_data()

        Args:
            cell_id: The ID of the cell being executed. Passed through to
                column tracker for provenance recording.

        Yields:
            None - execute your code inside the with block
        """
        self.reset_tracking()
        self._tracking_enabled = True
        self.start_column_tracking(cell_id=cell_id)
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
        from flowbook.kernel_support.memory_checkpoint import is_valid_variable_name
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

        # Structural mutations: exclude variables that were wholly written
        # (their mutations are subsumed by the Var(x) write)
        row_muts = {
            v for v in self._column_tracker.resolve_row_mutations_to_paths()
            if v not in self._writes
        } if self._column_tracker else set()
        index_muts = {
            v for v in self._column_tracker.resolve_index_mutations_to_paths()
            if v not in self._writes
        } if self._column_tracker else set()
        dtype_chg = {
            k: set(v)
            for k, v in (self._column_tracker.resolve_dtype_changes_to_paths() if self._column_tracker else {}).items()
            if k not in self._writes
        }
        col_dels = {
            k: set(v)
            for k, v in (self._column_tracker.resolve_column_deletions_to_paths() if self._column_tracker else {}).items()
            if k not in self._writes
        }

        track_imports = imports_tracked()

        def keep(name: str) -> bool:
            value = self._real_ns.get(name)
            if track_imports and isinstance(value, types.ModuleType):
                # Module bindings are locations for the ordering predicates
                # (a cell using `json` before the cell that imports it is a
                # forward contamination), even though they are not checkpointed.
                return is_valid_variable_name(name)
            return is_valid_variable(name, value)

        def is_noop_rebinding(name: str) -> bool:
            # `import pandas as pd` re-executed rebinds pd to the very same
            # module object. Reported in rebound_same; the enforcer classifies
            # it against the notebook's recorded writers (see check()).
            value = self._real_ns.get(name, _UNBOUND)
            prev = self._prev_bindings.get(name, _UNBOUND)
            return (
                track_imports
                and isinstance(value, types.ModuleType)
                and prev is value
            )

        kept_writes = set(k for k in self._writes if keep(k))
        return TrackingData(
            reads_before_writes=set(k for k in self._reads_before_writes if keep(k)),
            writes=kept_writes,
            rebound_same=set(k for k in kept_writes if is_noop_rebinding(k)),
            # Names now bound to a different object than before the cell (identity:
            # `x = x` followed by an in-place change is a mutation, not a rebinding).
            rebound=set(k for k in kept_writes
                        if self._real_ns.get(k, _UNBOUND) is not self._prev_bindings.get(k, _UNBOUND)),
            column_reads_before_writes=column_rbw,
            column_writes={k: set(v) for k, v in self.column_writes.items()},
            structural_reads=struct_reads,
            row_mutations=row_muts,
            index_mutations=index_muts,
            dtype_changes=dtype_chg,
            column_deletions=col_dels,
        )
