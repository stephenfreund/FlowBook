"""
FlowbookKernel - IPython kernel with Reproducibility enforcement.

================================================================================
REPRODUCIBILITY - SPECIFICATION
================================================================================

For a formal treatment with algorithms and proofs, see analysis.md in the
project root. This docstring provides implementation-level documentation.

Reproducibility ensures notebook reproducibility by enforcing dataflow rules that prevent
hidden state dependencies. When Reproducibility is enforced, running cells top-to-bottom
always produces the same result as running them in any order.

================================================================================
THE THREE Reproducibility RULES
================================================================================

Rule 1: Reproducibility Invariant (Goal)
----------------------------------------
A notebook is reproducible if running all cells in document order from a fresh
kernel produces identical results every time. This is the goal Reproducibility enforces.

Rule 2: Staleness Propagation (Computed)
----------------------------------------
A cell becomes "stale" when any variable it reads has a different value than
when the cell last executed. Stale cells are displayed in the UI; users should
re-execute them to reflect current state.

A cell X is stale if:
    - X was previously executed
    - Some variable V was in X's "reads_before_writes" set
    - V's current value differs from when X was executed

Rule 3: No Backward Mutation (Enforced)
----------------------------------------
A cell may NOT modify a variable that an earlier cell (in document order)
reads. This prevents "hidden" dependencies where earlier cells depend on later
cells having run first.

VIOLATION CONDITION: Cell X at position P causes a violation if:
    - X modifies variable V (detected via checkpoint diff)
    - Cell Y at position Q < P exists such that:
        - Y was previously executed
        - Y read V (V is in Y's reads_before_writes set)
        - The modification affects values Y depends on (see conflict rules below)

================================================================================
WHAT IS TRACKED AND CHECKED
================================================================================

1. GLOBAL VARIABLES
-------------------
Every cell execution tracks:
    - reads_before_writes: Variables read before being written in that cell
    - writes: All variables written in that cell

When a cell accesses a variable from the namespace (e.g., `x`), it's recorded
as a read if the cell hasn't already written to it. Writing to a variable
records it as a write.

CONFLICT RULE: If Cell B modifies variable V, and earlier Cell A read V, this
is a backward mutation violation (unless column-level tracking exempts it).

Example (VIOLATION):
    Cell A: y = x + 1       # reads_before_writes = {x}
    Cell B: x = 10          # writes = {x} → VIOLATES because A read x

Example (OK):
    Cell A: x = 1           # writes = {x}
    Cell B: y = x + 1       # reads_before_writes = {x}
    Cell A re-run: x = 2    # B becomes STALE (not a violation, just needs re-run)

2. DATAFRAME COLUMN-LEVEL TRACKING
----------------------------------
For pandas DataFrames, Reproducibility tracks individual columns for precision:
    - column_reads_before_writes: {var: {columns read}}
    - column_writes: {var: {columns written}}

CONFLICT RULE: Modifying df['col_a'] does NOT conflict with reading df['col_b'].
Only overlapping columns cause violations.

Example (OK - different columns):
    Cell A: total = df['price'].sum()           # column_reads = {df: {price}}
    Cell B: df['quantity'] = df['quantity'] * 2 # column_writes = {df: {quantity}}
    → No violation: price ≠ quantity

Example (VIOLATION - same column):
    Cell A: total = df['price'].sum()           # column_reads = {df: {price}}
    Cell B: df['price'] = df['price'] * 1.1     # column_writes = {df: {price}}
    → VIOLATION: B modifies price which A read

CONSERVATIVE FALLBACK: If column information is unavailable on either side,
Reproducibility conservatively treats it as a variable-level conflict.

3. STRUCTURAL ATTRIBUTE TRACKING
--------------------------------
Structural tracking detects when code accesses attributes that reveal a
DataFrame/Series structure (shape, columns, index, dtypes).

TRACKED ATTRIBUTES (explicit access only):

DataFrame column-revealing:
    columns, keys, dtypes, T, axes, values

DataFrame row-revealing:
    index, shape, size, empty

DataFrame methods:
    describe(), to_dict(), to_records(), head(), tail(), sample(),
    info(), select_dtypes(), memory_usage()

Series:
    index, shape, dtype, name, size, empty, values, to_dict(), to_list()

STRUCTURE-USING vs STRUCTURE-REVEALING:
    - "Structure-revealing" methods (e.g., df.columns) ARE tracked
    - "Structure-using" methods (e.g., df['col'], df.mean()) are NOT tracked
      because they use structure internally but don't expose it to the user

Example: df['x'] = 3 internally checks df.columns, but this is NOT recorded
as a structural read because the primary purpose is mutation, not inspection.

STRUCTURAL TRACKING MODES:

%structural_tracking off     - Don't track structural attributes
%structural_tracking warn    - Track and warn, but don't block (DEFAULT)
%structural_tracking enforce - Track and block violations

WARN mode example:
    Cell A: cols = df.columns.tolist()  # structural_reads = {df: {columns}}
    Cell B: df['new_col'] = 1           # adds column
    → WARNING: "Cell @B modified 'df' which Cell @A previously read."
    → Shows what was read (df.columns → ['a', 'b']) and what changed

ENFORCE mode example:
    Cell A: n = len(df)                 # structural_reads = {df: {len}}
    Cell B: df.loc[len(df)] = [1, 2]    # adds row
    → VIOLATION: "Cell @B modified 'df' which Cell @A (earlier) reads."
    → Cell B is rolled back

================================================================================
HOW CHANGE DETECTION WORKS
================================================================================

Reproducibility uses checkpoint-based diffing to detect actual changes:

1. PRE-CHECKPOINT: Deep copy of namespace before cell execution
2. EXECUTION: Run cell with TrackingDict recording reads/writes
3. POST-CHECKPOINT: Deep copy of namespace after cell execution
4. DIFF: Compare pre vs post to find actual changes

DIFF SEMANTICS:
    - Value equality for primitives (int, str, etc.)
    - Deep equality for containers (list, dict, etc.)
    - Element-wise equality for numpy arrays (with dtype tolerance)
    - Column-by-column equality for DataFrames
    - LEQ (Less-or-Equal) semantics for column tracking: new columns OK for writes

STALENESS CHECK:
    For each previously-executed cell X with reads_before_writes R:
    Compare X's pre-checkpoint (what X saw) vs current namespace
    If any variable in R differs → X is stale

================================================================================
EXECUTION FLOW
================================================================================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         FlowbookKernel                             │
    │                                                                     │
    │  1. do_execute() receives code + cell_id + cell_order               │
    │                         │                                           │
    │                         ▼                                           │
    │  2. Take PRE-checkpoint (deep copy namespace)                       │
    │                         │                                           │
    │                         ▼                                           │
    │  3. Execute code with TrackingDict (records reads/writes)           │
    │     - Track variable access via __getitem__/__setitem__             │
    │     - Track column access via monkey-patched DataFrame methods      │
    │     - Track structural access via monkey-patched properties         │
    │                         │                                           │
    │                         ▼                                           │
    │  4. Take POST-checkpoint (deep copy namespace)                      │
    │                         │                                           │
    │                         ▼                                           │
    │  5. ReproducibilityEnforcer.check():                                            │
    │     a. Diff pre vs post to find actual changes                      │
    │     b. Check Rule 3 violations against earlier cells                │
    │     c. Update staleness cache (Rule 2)                              │
    │     d. Capture structural read values for error messages            │
    │                         │                                           │
    │                         ▼                                           │
    │  6. On violation:                                                   │
    │     - Restore PRE-checkpoint (rollback cell effects)                │
    │     - Return error with detailed diagnostics                        │
    │     On success:                                                     │
    │     - Display metadata (reads, writes, stale cells)                 │
    │     - Send structural warnings if any                               │
    └─────────────────────────────────────────────────────────────────────┘

================================================================================
MAGIC COMMANDS
================================================================================

%notebook_structure cell1 cell2 ...  - Set cell order (auto-injected by client)
%flowbook_status                     - Display current Reproducibility state
%flowbook_stale                      - Show which cells are currently stale

Boolean toggle commands (no arg = enable, ? = show status):
%continue_after_violation            - Enable continuing after violations
%continue_after_violation off        - Stop on violation (default behavior)
%continue_after_violation ?          - Show current setting

Mode selection command:
%structural_tracking [off|warn|enforce] - Set structural tracking mode
    off     - Don't track structural attributes
    warn    - Track and warn, but don't block (default)
    enforce - Track and block violations

================================================================================
ASSUMPTIONS AND LIMITATIONS
================================================================================

ASSUMPTIONS:
------------
1. Cell order is provided correctly by the client (via cell_order metadata)
2. All code executes synchronously (no background threads modifying state)
3. Variables accessed via the global namespace dict are the primary data
4. Pandas DataFrames/Series are the primary data structures for column tracking

KNOWN LIMITATIONS:
------------------

1. CLASS VARIABLES NOT TRACKED
   Class-level attributes (class variables) are not restored on rollback:
       class Counter:
           count = 0  # This won't be restored on rollback!
   Workaround: Use instance attributes instead.

2. NESTED OBJECT MUTATIONS
   Mutations to objects stored inside other objects may not be fully tracked:
       data['nested']['key'] = value  # Tracked at 'data' level, not 'nested'
   The outer variable is tracked, but we can't always detect which inner
   part changed.

3. GENERATOR/ITERATOR STATE
   Generators and iterators cannot be checkpointed (they have execution state).
   Restored iterators may behave unexpectedly.

4. EXTERNAL SIDE EFFECTS
   File I/O, network calls, database modifications are NOT rolled back:
       f.write(data)  # File is modified even if cell is rolled back
   Reproducibility only manages Python namespace state.

5. MATPLOTLIB OBJECTS EXCLUDED
   Matplotlib figures/axes are not checkpointed (unpicklable).
   Plot state is not restored on rollback.

6. APPROXIMATE COLUMN TRACKING
   Column tracking uses monkey-patching and may miss edge cases:
   - Custom DataFrame subclasses may bypass tracking
   - Very complex chained operations might not track correctly
   - df.values mutations bypass column tracking entirely

7. STRUCTURAL TRACKING LIMITATIONS
   - Only tracks explicit attribute access (df.columns), not implicit use
   - Structure-using methods (df['x']) internally access .columns but aren't
     tracked because tracking internal implementation details would cause
     excessive false positives
   - df.attrs (user-defined metadata) is not currently tracked

8. PERFORMANCE OVERHEAD
   - Each cell execution requires two deep copies (pre/post checkpoint)
   - Large DataFrames increase checkpoint time significantly
   - Column tracking adds overhead to every DataFrame operation

9. NOT THREAD-SAFE
   Concurrent cell executions would corrupt tracking state.

================================================================================
AREAS FOR IMPROVEMENT
================================================================================

DESIGN IMPROVEMENTS:
--------------------

1. INCREMENTAL CHECKPOINTING
   Currently: Full deep copy before and after every cell
   Improvement: Copy-on-write or incremental snapshots to reduce overhead
   Challenge: Detecting mutations without full copies is complex

2. FINER-GRAINED TRACKING
   Currently: Variable and column level
   Improvement: Track array indices, dict keys, object attributes
   Challenge: Performance cost of fine-grained tracking; complexity

3. STRUCTURAL ATTRIBUTE VALUES
   Currently: Capture values at read time for better error messages
   Improvement: Show before/after values in all violation messages
   Status: Partially implemented (structural_reads_values in ReproducibilityExecutionRecord)

4. ASYNC EXECUTION SUPPORT
   Currently: Assumes synchronous execution
   Improvement: Support await/async code with proper state tracking
   Challenge: Interleaved execution complicates read/write ordering

IMPLEMENTATION IMPROVEMENTS:
----------------------------

1. ATTRS TRACKING
   df.attrs (DataFrame metadata dict) should be tracked like columns
   Currently not tracked at all

2. INDEX TRACKING
   df.index is tracked structurally but not at element level
   Could track index element access like columns

3. MULTIINDEX SUPPORT
   MultiIndex columns/indices may need special handling for tracking

4. SERIES AS COLUMN PROXY
   When user does `s = df['col']`, mutations to s should trigger df warnings
   Currently only direct df mutations are tracked

5. BETTER ERROR RECOVERY
   Currently: Rollback entire cell on violation
   Improvement: Partial execution recovery, show what succeeded

6. VISUALIZATION OF DEPENDENCIES
   Add command to visualize cell dependency graph
   Show which cells would become stale if a variable changes

7. UNDO HISTORY
   Allow undoing multiple cells, not just the current one
   Keep checkpoint history for user-initiated undo

================================================================================
SPECIAL TYPE SUPPORT
================================================================================

cuDF (GPU DataFrames)
---------------------
cuDF objects are transparently handled via cudf_compat module:
- Supports both native cudf and cudf.pandas proxy mode
- Checkpoints convert GPU→CPU (cudf→pandas) for storage
- Column tracking works with cudf.DataFrame.__getitem__/__setitem__

Keras Models
------------
Keras/TensorFlow models use the opaque object pattern:
- Only weights are checkpointed (not millions of TensorFlow objects)
- Deferred import avoids ~3s penalty for non-Keras notebooks
- Detection via module inspection: _is_keras_model() in deepcopy.py/diff.py

See checkpoint.py sections 13-14 for implementation details.

================================================================================
"""

import os
import re
import time
import traceback
from typing import Optional, Tuple

from IPython.core.magic import Magics, line_magic, magics_class
from ipykernel.kernelapp import IPKernelApp

from flowbook.kernel_support import extended_types
from flowbook.kernel_support.base_kernel import BaseFlowbookKernel
from flowbook.kernel_support.checkpoint import filter_user_namespace
from flowbook.kernel_support.deepcopyable import check_deepcopyable
from flowbook.kernel_support.timeout_handler import CellTimeoutHandler
from flowbook.kernel_support.tracking import TrackingDict
from flowbook.util.cell_index import index_to_alpha
from flowbook.util.output import error, log, timer, output

from flowbook.kernel.models import ReproducibilityMetadata
from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
)


@magics_class
class FlowbookKernel(BaseFlowbookKernel, Magics):
    """
    IPython kernel with Reproducibility enforcement.

    Features:
    - Variable access tracking (reads/writes per cell)
    - Reproducibility Rule 3 enforcement (no backward mutations)
    - Staleness computation and reporting
    - Cell order management via magic command

    Reproducibility is always enabled. No profiling or checkpoint magics.
    """

    implementation = "flowbook_kernel"
    implementation_version = "0.1"
    banner = "FlowBook Reproducibility Kernel - Reproducibility"

    # =========================================================================
    # Configuration Constants
    # =========================================================================

    _default_cell_timeout = 30 * 60  # 30 minutes
    _post_kb_grace = 1.0  # Grace period after KeyboardInterrupt
    _kill_timeout = 3.0  # Time to wait before force kill
    _verbose = False
    _max_passes = 2  # Max timeout handler passes

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.shell is not None
        self.shell.register_magics(self)

        # Expose checkpoint object to user code for memory measurement
        # (consistent with checkpoint_kernel for benchmark_checkpoint.py compatibility)
        self.shell.user_ns["_flowbook_checkpoint"] = self._checkpoint

        # Tracking
        self._tracking = TrackingDict(self.shell.user_ns)

        # Reproducibility enforcement
        self._enforcer = ReproducibilityEnforcer(self._checkpoint)

        # Continue after violation flag (default: stop on violation)
        self._continue_after_violation: bool = False

        # Ensure filesystem magics are registered
        self._ensure_fs_magics()

        # Ensure VFS patches are applied to user namespace
        self._ensure_vfs_namespace_patched()

        # Ensure tracking is initialized (done lazily on first execution)
        self._ensure_tracking_initialized()

        # Pre-load cudf if available (import takes ~3s for CUDA init)
        # Doing this here avoids charging the overhead to the first cell
        from flowbook.kernel_support.cudf_compat import has_cudf
        has_cudf()

    # =========================================================================
    # Magic Commands
    # =========================================================================

    @line_magic
    def notebook_structure(self, line: str) -> None:
        """
        Set the notebook cell order for Reproducibility enforcement.

        Usage:
            %notebook_structure cell1 cell2 cell3 ...
        """
        cell_order = line.split()
        self._enforcer.set_cell_order(cell_order)

    @line_magic
    def flowbook_status(self, line: str) -> None:
        """Display current Reproducibility state."""
        order = self._enforcer.cell_order
        records = self._enforcer.records

        status_lines = [
            f"Cell order: {order}",
            f"Executed cells: {list(records.keys())}",
            f"Execution counter: {self._enforcer.seq_counter}",
        ]

        for cell_id, record in records.items():
            status_lines.append(
                f"  {cell_id}: reads={sorted(record.reads)}, "
                f"writes={sorted(record.writes)}, seq={record.execution_seq}"
            )

        self._display.display_icon_and_text(
            "ℹ️", "Reproducibility Status", "\n".join(status_lines)
        )

    @line_magic
    def continue_after_violation(self, line: str) -> None:
        """
        Control whether execution continues after Reproducibility violations.

        Usage:
            %continue_after_violation        - Enable (continue after violations, default)
            %continue_after_violation on     - Enable (continue after violations)
            %continue_after_violation off    - Disable (stop on violation)
            %continue_after_violation ?      - Show current status
        """
        arg = line.strip().lower()

        if arg == "?":
            status = "on" if self._continue_after_violation else "off"
            self._display.display_icon_and_text(
                "ℹ️", f"Continue after violation: {status}"
            )
            return

        if not arg or arg in ("on", "true", "1", "enable"):
            self._continue_after_violation = True
            self._display.display_icon_and_text(
                "ℹ️", "Continue after violation: enabled"
            )
        elif arg in ("off", "false", "0", "disable"):
            self._continue_after_violation = False
            self._display.display_icon_and_text(
                "ℹ️", "Continue after violation: disabled"
            )
        else:
            self._display.display_icon_and_text(
                "⚠️", f"Invalid: '{arg}'. Use 'on', 'off', or '?'"
            )

    @line_magic
    def flowbook_stale(self, line: str) -> None:
        """
        Show which cells are currently stale.

        Usage:
            %flowbook_stale
        """
        stale_cells = self._enforcer.get_stale_cells()
        if not stale_cells:
            self._display.display_icon_and_text("✓", "No stale cells")
            return

        # Convert to @A notation
        stale_refs = []
        for cell_id in stale_cells:
            try:
                idx = self._enforcer.cell_order.index(cell_id)
                stale_refs.append(index_to_alpha(idx))
            except (ValueError, IndexError):
                stale_refs.append(cell_id)

        self._display.display_icon_and_text(
            "⚠️", f"Stale cells: {', '.join(stale_refs)}"
        )

    @line_magic
    def cell_edited(self, line: str) -> None:
        """[EDIT transition] Mark a cell as edited (stale) (§2.3).

        Usage:
            %cell_edited <cell_id>
        """
        cell_id = line.strip()
        if not cell_id:
            return

        stale_cells = self._enforcer.mark_cell_edited(cell_id)
        if cell_id in [c for c in stale_cells]:
            # Send updated staleness info to frontend
            metadata = ReproducibilityMetadata(
                cell_id=cell_id,
                execution_seq=self._enforcer.seq_counter,
                reads=[],
                writes=[],
                changed_variables=[],
                stale_cells=stale_cells,
                violation=None,
                cell_order=self._enforcer.cell_order,
            )
            self._display.display_icon_and_text(
                "✏️",
                f"Cell edited, marked stale",
                metadata=metadata.to_display_metadata(),
            )

    @line_magic
    def exec_restore(self, line: str) -> None:
        """Deprecated: EXEC-RESTORE has been removed.

        Forward contamination now blocks execution. To fix:
        1. Run the upstream cells in document order
        2. Then run this cell
        """
        self._display.display_icon_and_text(
            "❌",
            "EXEC-RESTORE is deprecated. Run upstream cells in document order to fix forward contamination."
        )

    @line_magic
    def structural_tracking(self, line: str) -> None:
        """
        Set structural tracking mode for DataFrame/Series attribute monitoring.

        Structural tracking detects when code accesses attributes that reveal
        DataFrame/Series structure (like df.columns, df.shape, len(df)).
        When structural tracking is enabled and these attributes are read,
        subsequent changes to the structure (adding columns, changing row count)
        are either warned about or treated as Reproducibility violations.

        Usage:
            %structural_tracking           - Show current mode
            %structural_tracking off       - Disable structural tracking
            %structural_tracking warn      - Track and warn only (default)
            %structural_tracking enforce   - Track and treat changes as violations
        """
        from flowbook.kernel_support.structural_tracking import StructuralTrackingMode

        # Suspend tracking during magic execution to avoid recording infrastructure reads
        tracking = self._tracking
        if tracking is not None and hasattr(tracking, "suspended"):
            ctx = tracking.suspended()
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            mode_str = line.strip().lower()

            if not mode_str:
                # Show current mode
                current_mode = self._enforcer.structural_mode.value
                self._display.display_icon_and_text(
                    "🔍", f"Structural tracking mode: {current_mode}"
                )
                return

            try:
                mode = StructuralTrackingMode(mode_str)
            except ValueError:
                self._display.display_icon_and_text(
                    "❌", f"Invalid mode: {mode_str}. Use 'off', 'warn', or 'enforce'"
                )
                return

            # Update Reproducibility enforcer
            self._enforcer.set_structural_mode(mode)

            # Update TrackingDict if it exists
            if tracking is not None:
                tracking.set_structural_tracking_mode(mode_str)

            self._display.display_icon_and_text(
                "✅", f"Structural tracking mode set to: {mode.value}"
            )

    # =========================================================================
    # Memory introspection magic (using HeapSizer)
    # =========================================================================

    @line_magic
    def memory(self, line: str) -> None:
        """
        Show memory usage using HeapSizer heap traversal.

        HeapSizer provides accurate memory measurement by traversing the
        object graph with proper handling of numpy views, pandas CoW, and
        shared references.

        Usage:
            %memory          - Show namespace summary
            %memory vars     - Show per-variable memory breakdown
            %memory vars 10  - Show top 10 variables by size
            %memory ckpt     - Show checkpoint overhead breakdown
            %memory cache    - Show deepcopy cache sizes
            %memory internal - Show FlowBook internal overhead
        """
        from flowbook.kernel_support.heap_size import HeapSizer

        cmd = line.strip().lower()
        args = cmd.split()
        subcmd = args[0] if args else ""

        if subcmd == "" or subcmd == "?" or subcmd == "status":
            # Show namespace summary
            sizer = HeapSizer()
            user_ns = self.shell.user_ns
            checkpointable = self._checkpoints.checkpointable_vars(user_ns)
            ns_size = sizer.sizeof_namespace(checkpointable)

            msg = f"Namespace memory: {ns_size.total_bytes / (1024*1024):.1f} MB\n"
            msg += f"Variables: {len(ns_size.by_variable)}\n"
            msg += f"Types: {len(ns_size.by_type)}"
            self._display.display_icon_and_text("📊", msg)

        elif subcmd == "vars":
            # Show per-variable memory breakdown
            limit = 20
            if len(args) > 1:
                try:
                    limit = int(args[1])
                except ValueError:
                    pass
            self._show_var_memory_breakdown(limit)

        elif subcmd == "ckpt":
            # Show checkpoint costs
            self._show_checkpoint_memory_costs()

        elif subcmd == "cache":
            # Show deepcopy cache sizes
            self._show_cache_sizes()

        elif subcmd == "internal":
            # Show FlowBook internal overhead
            self._show_internal_sizes()

        else:
            self._display.display_icon_and_text(
                "❓", f"Unknown command: {subcmd}. Use vars/ckpt/cache/internal"
            )

    def _show_var_memory_breakdown(self, limit: int = 20) -> None:
        """Show per-variable memory breakdown from user namespace using HeapSizer."""
        from flowbook.kernel_support.heap_size import HeapSizer

        sizer = HeapSizer()
        user_ns = self.shell.user_ns
        checkpointable = self._checkpoints.checkpointable_vars(user_ns)
        ns_size = sizer.sizeof_namespace(checkpointable)

        # Build list of (name, type, size)
        var_sizes = []
        for name, size in ns_size.by_variable.items():
            type_name = type(checkpointable[name]).__name__
            var_sizes.append((name, type_name, size))

        # Sort by size descending
        var_sizes.sort(key=lambda x: x[2], reverse=True)
        var_sizes = var_sizes[:limit]

        # Format output
        lines = ["Variable         Type            Size"]
        lines.append("─" * 50)
        for name, type_name, size in var_sizes:
            size_str = self._format_bytes(size)
            lines.append(f"{name:<16} {type_name:<15} {size_str:>10}")

        self._display.display_icon_and_text("📊", "\n".join(lines))

    def _show_checkpoint_memory_costs(self) -> None:
        """Show checkpoint memory costs using HeapSizer."""
        internal_sizes = self._checkpoints.memory.get_internal_sizes()

        lines = ["Checkpoint Memory Costs (measured by HeapSizer):"]
        lines.append("")
        lines.append(
            f"Checkpoint data:     {self._format_bytes(internal_sizes['checkpoints_total'])}"
        )
        lines.append(
            f"Deepcopy cache:      {self._format_bytes(internal_sizes['deepcopy_cache_total'])}"
        )
        lines.append(
            f"Alias index:         {self._format_bytes(internal_sizes['alias_index_total'])}"
        )
        lines.append(
            f"Var costs cache:     {self._format_bytes(internal_sizes['var_costs_cache'])}"
        )

        total = sum(internal_sizes.values())
        lines.append("")
        lines.append(f"Total overhead:      {self._format_bytes(total)}")

        self._display.display_icon_and_text("📊", "\n".join(lines))

    def _show_cache_sizes(self) -> None:
        """Show deepcopy cache sizes."""
        from flowbook.kernel_support.deepcopy import (
            get_container_cache_stats,
            get_cache_sizes,
            get_cached_objects_size,
        )

        stats = get_container_cache_stats()
        sizes = get_cache_sizes()
        total_cached = get_cached_objects_size()

        lines = ["Deepcopy Cache Statistics:"]
        lines.append("")
        lines.append(f"List cache:    {stats['list_cache_size']} entries")
        lines.append(f"Set cache:     {stats['set_cache_size']} entries")
        lines.append(f"Dict cache:    {stats['dict_cache_size']} entries")
        lines.append(f"ndarray cache: {stats['ndarray_cache_size']} entries")
        lines.append("")
        lines.append(f"Cache structure:     {self._format_bytes(sum(sizes.values()))}")
        lines.append(f"Cached objects:      {self._format_bytes(total_cached)}")

        self._display.display_icon_and_text("📊", "\n".join(lines))

    def _show_internal_sizes(self) -> None:
        """Show FlowBook internal memory overhead."""
        internal_sizes = self._checkpoints.memory.get_internal_sizes()
        overhead = self._checkpoints.memory.get_overhead_breakdown()

        lines = ["FlowBook Internal Memory:"]
        lines.append("")
        lines.append("From HeapSizer:")
        for key, value in internal_sizes.items():
            lines.append(f"  {key}: {self._format_bytes(value)}")
        lines.append("")
        lines.append("From cached costs:")
        for key, value in overhead.items():
            lines.append(f"  {key}: {self._format_bytes(value)}")

        self._display.display_icon_and_text("📊", "\n".join(lines))

    def _format_bytes(self, size: int) -> str:
        """Format bytes as human-readable string."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"

    @line_magic
    def tracking(self, line: str) -> None:
        """Stub for global tracking (always on in Reproducibility kernel)."""
        self._display.display_icon_and_text(
            "ℹ️", "Global tracking is always enabled in Reproducibility kernel"
        )

    @line_magic
    def monotone(self, line: str) -> None:
        """Stub for monotone enforcement (use Reproducibility rules instead)."""
        self._display.display_icon_and_text(
            "ℹ️",
            "Reproducibility kernel uses Reproducibility rules instead of monotone enforcement",
        )

    @line_magic
    def force_checkpoints(self, line: str) -> None:
        """Stub for force checkpoints (Reproducibility kernel always checkpoints)."""
        self._display.display_icon_and_text(
            "ℹ️", "Reproducibility kernel always takes checkpoints"
        )

    @line_magic
    def checkpoint(self, line: str) -> None:
        """Stub for checkpoint magic (not supported in Reproducibility kernel)."""
        self._display.display_icon_and_text(
            "ℹ️", "Manual checkpoints not supported in Reproducibility kernel"
        )

    @line_magic
    def restore(self, line: str) -> None:
        """Stub for restore magic (not supported in Reproducibility kernel)."""
        self._display.display_icon_and_text(
            "ℹ️", "Restore not supported in Reproducibility kernel"
        )

    @line_magic
    def list_checkpoints(self, line: str) -> None:
        """Stub for list checkpoints magic (not supported in Reproducibility kernel)."""
        self._display.display_icon_and_text(
            "ℹ️", "List checkpoints not supported in Reproducibility kernel"
        )

    # =========================================================================
    # Tracking Initialization
    # =========================================================================

    def _ensure_tracking_initialized(self) -> None:
        """
        Initialize variable tracking with full comprehension support.

        This patches the shell's run_code method to use our TrackingDict for
        BOTH globals and locals. This is necessary because:

        1. Python's exec(code, globals, locals) uses globals for LOAD_GLOBAL
        2. List comprehensions and functions use LOAD_GLOBAL for free variables
        3. If globals != our TrackingDict, comprehension reads aren't tracked

        By using TrackingDict for both, all variable access is tracked including
        reads inside list comprehensions and nested functions.
        """
        if not isinstance(self.shell.user_ns, TrackingDict):
            # Create TrackingDict wrapping user_global_ns
            tracking_dict = TrackingDict(self.shell.user_global_ns)
            self.shell.user_ns = tracking_dict

            # Patch run_code to use TrackingDict for both globals and locals
            self._patch_run_code(tracking_dict)

            # Save initial state checkpoint (σ_0) for EXEC-RESTORE on the first cell
            self._take_checkpoint("_initial_state")

    def _patch_run_code(self, tracking_dict: TrackingDict) -> None:
        """
        Patch shell.run_code to use TrackingDict for both globals and locals.

        This enables tracking of variable reads inside list comprehensions
        and nested functions, which would otherwise bypass our TrackingDict.
        """
        shell = self.shell
        original_run_code = shell.run_code

        def patched_run_code(code_obj, result=None, *, async_=False):
            """
            Execute code using TrackingDict for both globals and locals.

            This ensures all variable access is tracked, including reads
            inside list comprehensions which use LOAD_GLOBAL.
            """
            # Temporarily replace both user_ns and inject tracking_dict
            # as the globals dict for exec. We do this by temporarily
            # swapping what user_global_ns returns.

            # Store original
            old_user_ns = shell.user_ns

            try:
                # Set both to tracking_dict so exec sees it as globals
                shell.user_ns = tracking_dict
                # user_global_ns is a property, but we shadow it in __dict__
                shell.__dict__["user_global_ns"] = tracking_dict

                # Call original run_code - it will use our tracking_dict
                return original_run_code(code_obj, result, async_=async_)

            finally:
                # Restore
                shell.user_ns = old_user_ns
                # Remove the shadow
                if "user_global_ns" in shell.__dict__:
                    del shell.__dict__["user_global_ns"]

        # Replace the method
        shell.run_code = patched_run_code

    # =========================================================================
    # Execution
    # =========================================================================

    async def _do_execute_impl(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
        cell_meta: Optional[dict] = None,
    ) -> dict:
        """
        Execute code with Reproducibility tracking and enforcement.
        """
        start_time = time.perf_counter() * 1000
        execution_time = None

        with timer(message=f"do_execute: {self._cell_id}"):
            try:
                # Update cell order if provided in metadata
                if cell_meta and "cell_order" in cell_meta:
                    self._enforcer.set_cell_order(cell_meta["cell_order"])

                # Check for notebook_structure magic (parse and remove if present)
                code = self._process_structure_magic(code)

                # Extract timeout from code directive or cell_meta
                code, timeout = self._extract_timeout(code, cell_meta)

                # Skip Reproducibility for empty code or pure magic
                if not code.strip() or self._is_pure_magic(code):
                    return await self._execute_without_enforcer(
                        code,
                        silent,
                        store_history,
                        user_expressions,
                        allow_stdin,
                        cell_meta,
                    )

                # Take pre-execution snapshot
                user_ns = self.shell.user_ns
                with timer(
                    key="kernel:checkpoint", message="Pre-execution checkpoint"
                ) as pre_timer:
                    pre_checkpoint = self._take_checkpoint(
                        f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}"
                    )

                # Reset tracking for this execution
                if isinstance(user_ns, TrackingDict):
                    user_ns.reset_tracking()
                # Reset VFS per-cell tracking
                self._vfs.reset_cell_tracking()

                # Setup timeout handler
                timeout_handler = CellTimeoutHandler(
                    timeout=timeout,
                    post_kb_grace=self._post_kb_grace,
                    kill_timeout=self._kill_timeout,
                    verbose=self._verbose,
                    max_passes=self._max_passes,
                )
                timeout_handler.start()
                normal_exit = False

                try:
                    # Execute with tracking
                    if isinstance(user_ns, TrackingDict):
                        with user_ns.track_execution():
                            with timer(
                                key="kernel:track_execution",
                                message="Run cell code",
                            ):
                                with timer(key="kernel:execute") as run_timer:
                                    result = await self._ipython_do_execute(
                                        code,
                                        silent,
                                        store_history,
                                        user_expressions,
                                        allow_stdin,
                                        cell_meta=cell_meta,
                                        cell_id=self._cell_id,
                                    )
                                execution_time = run_timer.duration()
                            with timer(
                                key="kernel:get_tracking_data",
                                message="Get tracking data",
                            ):
                                tracking = user_ns.get_tracking_data()
                    else:
                        with timer(key="kernel:execute") as run_timer:
                            result = await self._ipython_do_execute(
                                code,
                                silent,
                                store_history,
                                user_expressions,
                                allow_stdin,
                                cell_meta=cell_meta,
                                cell_id=self._cell_id,
                            )
                        execution_time = run_timer.duration()
                        tracking = None
                    normal_exit = True

                    # Merge VFS file tracking into TrackingData
                    if tracking is not None and (
                        self._vfs.enabled or self._vfs.tracking_only
                    ):
                        file_tracking = self._vfs.get_cell_file_tracking()
                        tracking.file_reads_before_writes = (
                            file_tracking.file_reads_before_writes
                        )
                        tracking.file_writes = file_tracking.file_writes
                except KeyboardInterrupt:
                    # Timeout occurred - restore pre-state
                    self._restore_checkpoint(f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}")
                    return self._handle_timeout_error(timeout)
                finally:
                    timeout_handler.cancel()
                    if not normal_exit:
                        await timeout_handler.cleanup_on_error()

                # If execution had an error, restore pre-state and skip Reproducibility checks
                if result.get("status") == "error":
                    self._restore_checkpoint(f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}")
                    return result

                # Run Reproducibility check if we have tracking data and cell_id
                # NOTE: We diff pre_checkpoint against the live namespace (user_ns)
                # instead of creating a post-checkpoint. This eliminates ~50% of
                # checkpoint overhead by avoiding the second deep copy.
                if tracking and self._cell_id:
                    with timer(key="kernel:check") as check_timer:
                        sdc_result = self._enforcer.check(
                            cell_id=self._cell_id,
                            pre_checkpoint=pre_checkpoint,
                            namespace=self.shell.user_ns,
                            tracking=tracking,
                            continue_on_violation=self._continue_after_violation,
                        )

                    # Handle violations (backward mutation and/or forward dependency)
                    has_backward = sdc_result and sdc_result.violation
                    has_forward = sdc_result and sdc_result.forward_violation

                    # [EXEC-REJECT] Backward conflict → rollback (Def 1.8.2)
                    if has_backward:
                        if self._continue_after_violation:
                            # Warn about backward violation but continue
                            if sdc_result.violation.truncation_details:
                                error(
                                    f"Reproducibility truncation: {sdc_result.violation.message}"
                                )
                                self._send_truncation_details(
                                    sdc_result.violation.truncation_details
                                )
                            error(
                                f"Reproducibility violation (continuing): {sdc_result.violation.message}"
                            )
                            self._send_violation_warning(sdc_result.violation)
                            if has_forward:
                                error(
                                    f"Forward dependency (continuing): {sdc_result.forward_violation.message}"
                                )
                                self._send_violation_warning(
                                    sdc_result.forward_violation
                                )
                        else:
                            # Block on backward violation
                            # Log truncation issues to terminal
                            if sdc_result.violation.truncation_details:
                                error(
                                    f"Reproducibility truncation: {sdc_result.violation.message}"
                                )
                                self._send_truncation_details(
                                    sdc_result.violation.truncation_details
                                )

                            error(
                                f"Reproducibility violation: {sdc_result.violation.message}"
                            )

                            self._restore_checkpoint(
                                f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}"
                            )

                            self._send_violation_error(sdc_result.violation)

                            # Also report forward violation if both exist
                            if has_forward:
                                self._send_violation_warning(
                                    sdc_result.forward_violation
                                )

                            return self._make_error_result(sdc_result.violation)

                    # Forward contamination → block execution (error, not warning)
                    # User must run upstream cells in document order to fix
                    if has_forward and not has_backward:
                        error(
                            f"Forward dependency (blocked): {sdc_result.forward_violation.message}"
                        )

                        self._restore_checkpoint(
                            f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}"
                        )

                        self._send_forward_violation_error(sdc_result.forward_violation, sdc_result.writer_violation)

                        return self._make_error_result(sdc_result.forward_violation)

                    # Display results (skip if silent, error, or backward violation with rollback)
                    skip_display = (has_backward) and not self._continue_after_violation
                    if (
                        not silent
                        and result.get("status") != "error"
                        and not skip_display
                    ):
                        pre_ms = pre_timer.duration()
                        post_ms = post_timer.duration()
                        state_ms = pre_ms + post_ms
                        self._display_execution_result(
                            execute_duration_ms=time.perf_counter() * 1000 - start_time,
                            code_duration_ms=execution_time or 0.0,
                            state_duration_ms=state_ms,
                            check_duration_ms=check_timer.duration(),
                            tracking=tracking,
                            sdc_result=sdc_result,
                            pre_state_ms=pre_ms,
                            post_state_ms=post_ms,
                        )

                return result
            except Exception as e:
                error(
                    f"Reproducibility error in cell {self._cell_id}: {e}\n{traceback.format_exc()}"
                )
                raise
            finally:
                end_time = time.perf_counter() * 1000
                if execution_time is not None:
                    duration = end_time - start_time
                    output.add_timing(
                        key="kernel:checking_total_time",
                        duration=duration - execution_time,
                    )
                    slowdown = duration / execution_time
                    output.add_timing(key="kernel:slowdown", duration=slowdown)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_cell_alpha(self) -> str:
        """Get @A notation for current cell."""
        if self._cell_id is None:
            return "unknown"
        try:
            index = self._enforcer.cell_order.index(self._cell_id)
            return index_to_alpha(index)
        except (ValueError, IndexError):
            return self._cell_id

    def _process_structure_magic(self, code: str) -> str:
        """
        Process %notebook_structure magic if present at start of code.
        Removes the magic line and updates cell order.
        Returns remaining code.
        """
        lines = code.split("\n")
        if lines and lines[0].strip().startswith("%notebook_structure"):
            # Extract cell order from magic line
            magic_line = lines[0].strip()
            parts = magic_line.split()[1:]  # Skip the magic name
            if parts:
                self._enforcer.set_cell_order(parts)
            return "\n".join(lines[1:])
        return code

    def _parse_timeout_from_code(self, code: str) -> Tuple[str, float]:
        """Parse timeout directive from code if present."""
        match = re.match(r"# timeout (\d+)\n", code)
        if match:
            timeout = int(match.group(1))
            code = code.replace(match.group(0), "", 1)
        else:
            timeout = self._default_cell_timeout
        return code, timeout

    def _extract_timeout(
        self, code: str, cell_meta: Optional[dict]
    ) -> Tuple[str, float]:
        """
        Extract timeout from code directive or cell_meta.

        Priority:
        1. # timeout N directive in code (highest)
        2. timeout from cell_meta (from command)
        3. _default_cell_timeout (fallback)

        Returns:
            Tuple of (code with directive removed, timeout in seconds)
        """
        # Parse timeout from code (highest priority)
        parsed_code, code_timeout = self._parse_timeout_from_code(code)

        # Determine timeout: code directive > cell_meta > default
        if code_timeout != self._default_cell_timeout:
            # Code had explicit # timeout directive
            timeout = code_timeout
        elif cell_meta and "timeout" in cell_meta:
            # Use timeout from cell_metadata (from command)
            timeout = float(cell_meta["timeout"])
        else:
            # Fall back to default
            timeout = self._default_cell_timeout

        return parsed_code, timeout

    def _handle_timeout_error(self, timeout: float) -> dict:
        """Create timeout error result."""
        timeout_msg = f"Cell execution timed out after {timeout} seconds"
        self.send_response(
            self.iopub_socket,
            "error",
            {
                "ename": "TimeoutError",
                "evalue": timeout_msg,
                "traceback": [timeout_msg],
            },
        )
        return {
            "status": "timeout",
            "execution_count": self.execution_count,
            "ename": "TimeoutError",
            "evalue": timeout_msg,
            "traceback": [timeout_msg],
        }

    def _warn_non_deepcopyable_objects(self) -> None:
        """Warn about objects that can't be deep copied (won't be checkpointed)."""
        user_ns = filter_user_namespace(self.shell.user_ns)

        # Collect non-copyable variables with their types and reasons
        non_copyable = []
        for k, v in user_ns.items():
            reason = check_deepcopyable(v)
            if reason:
                non_copyable.append((k, type(v).__name__, reason))

        if non_copyable:
            for k, typ, reason in non_copyable:
                message = (
                    f"The object {k} (type {typ}) cannot be checkpointed: {reason}"
                )
                log(message)
                self._display.display_icon_and_text("\u26a0\ufe0f", message)

    async def _execute_without_enforcer(
        self,
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> dict:
        """Execute without Reproducibility tracking (for magics, empty code)."""
        result = await self._ipython_do_execute(
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
            cell_meta=cell_meta,
            cell_id=self._cell_id,
        )

        # Send empty metadata to clear any stale metadata from previous executions
        # This ensures the frontend shows empty reads/writes for magic-only cells
        if not silent and self._cell_id:
            empty_metadata = ReproducibilityMetadata(
                cell_id=self._cell_id,
                execution_seq=self._enforcer.seq_counter,
                reads=[],
                writes=[],
                changed_variables=[],
                stale_cells=self._enforcer.get_stale_cells(),
                violation=None,
                cell_order=self._enforcer.cell_order,
            )
            # Use display_icon_and_text with metadata to send to frontend
            self._display.display_icon_and_text(
                "✓",
                "Magic cell",
                metadata=empty_metadata.to_display_metadata(),
            )

        return result

    def _format_var_with_columns(
        self,
        var: str,
        column_reads: dict,
        column_writes: dict,
    ) -> str:
        """
        Format variable with inline column info: df[price,qty].

        Args:
            var: Variable name
            column_reads: Dict mapping var names to sets/lists of read columns
            column_writes: Dict mapping var names to sets/lists of written columns

        Returns:
            Formatted string like "df[price,qty]" or "x" if no columns
        """
        # Handle both sets and lists
        read_cols = column_reads.get(var, [])
        write_cols = column_writes.get(var, [])
        all_cols = set(read_cols) | set(write_cols)

        if all_cols:
            cols_list = sorted(all_cols)[:3]  # Show first 3
            cols_str = ",".join(cols_list)
            if len(all_cols) > 3:
                cols_str += f",+{len(all_cols) - 3}"
            return f"{var}[{cols_str}]"
        return var

    def _display_execution_result(
        self,
        execute_duration_ms: float,
        code_duration_ms: float,
        state_duration_ms: float,
        check_duration_ms: float,
        tracking,
        sdc_result,
        pre_state_ms: float = 0.0,
        post_state_ms: float = 0.0,
    ) -> None:
        """Display execution timing and Reproducibility metadata."""
        # Build metadata for display
        structural_warnings = sdc_result.structural_warnings if sdc_result else []

        # Use TrackingData.to_json_friendly() for clean serialization
        tracking_json = (
            tracking.to_json_friendly()
            if tracking
            else {
                "reads": [],
                "writes": [],
                "column_reads": {},
                "column_writes": {},
                "structural_reads": {},
            }
        )

        # Get file tracking data (separate from variable tracking)
        # Only include files that were read before being written (important for reproducibility)
        file_rbw = (
            sorted(os.path.relpath(p) for p in tracking.file_reads_before_writes)
            if tracking
            else []
        )
        file_writes_list = (
            sorted(os.path.relpath(p) for p in tracking.file_writes) if tracking else []
        )

        metadata = ReproducibilityMetadata(
            cell_id=self._cell_id or "",
            execution_seq=self._enforcer.seq_counter,
            reads=tracking_json["reads"],  # Variable reads only
            writes=tracking_json["writes"],  # Variable writes only
            changed_variables=sdc_result.changed_variables if sdc_result else [],
            stale_cells=sdc_result.stale_cells if sdc_result else [],
            violation=(
                sdc_result.violation.to_dict()
                if (sdc_result and sdc_result.violation)
                else (
                    sdc_result.forward_violation.to_dict()
                    if (sdc_result and sdc_result.forward_violation)
                    else None
                )
            ),
            cell_order=self._enforcer.cell_order,
            column_reads=tracking_json["column_reads"],
            column_writes=tracking_json["column_writes"],
            column_changed=sdc_result.column_changed if sdc_result else {},
            structural_reads=tracking_json["structural_reads"],
            structural_warnings=structural_warnings,
            file_reads=file_rbw,  # File reads (separate from variables)
            file_writes=file_writes_list,  # File writes (separate from variables)
            execute_duration_ms=execute_duration_ms,
            code_duration_ms=code_duration_ms,
            state_duration_ms=state_duration_ms,
            check_duration_ms=check_duration_ms,
            writer_violation=(
                sdc_result.writer_violation.to_dict()
                if (sdc_result and sdc_result.writer_violation)
                else None
            ),
        )

        # Log and display structural warnings
        if structural_warnings:
            for warning in structural_warnings:
                error(f"[structural] {warning}")
            self._send_structural_warnings(structural_warnings)

        # Build display text
        state_detail = f"State: {state_duration_ms:.0f} ms (pre={pre_state_ms:.0f}, post={post_state_ms:.0f})"
        parts = [
            f"Execute: {execute_duration_ms:.0f} ms",
            f"Code: {code_duration_ms:.0f} ms",
            state_detail,
            f"Check: {check_duration_ms:.0f} ms",
        ]

        # Variable reads (separate from file reads)
        if tracking_json["reads"] or tracking_json["column_reads"]:
            read_vars = set(tracking_json["reads"]) | set(
                tracking_json["column_reads"].keys()
            )
            if read_vars:
                reads_preview = [
                    self._format_var_with_columns(v, tracking_json["column_reads"], {})
                    for v in list(read_vars)[:3]
                ]
                parts.append(f"Reads: {','.join(reads_preview)}")

        # Variable writes (separate from file writes)
        write_vars = set(tracking_json["writes"]) | set(
            tracking_json["column_writes"].keys()
        )
        if write_vars:
            writes_preview = [
                self._format_var_with_columns(v, {}, tracking_json["column_writes"])
                for v in list(write_vars)[:3]
            ]
            parts.append(f"Writes: {','.join(writes_preview)}")

        # File I/O (separate section)
        if file_rbw or file_writes_list:
            if file_rbw:
                parts.append(f"File Reads: {','.join(file_rbw)}")
            if file_writes_list:
                parts.append(f"File Writes: {','.join(file_writes_list)}")

        if sdc_result and sdc_result.stale_cells:
            # Convert cell IDs to @A references for display
            stale_refs = []
            for cell_id in sdc_result.stale_cells:
                try:
                    idx = self._enforcer.cell_order.index(cell_id)
                    stale_refs.append(index_to_alpha(idx))
                except (ValueError, IndexError):
                    stale_refs.append(cell_id)  # Fallback to ID if not in order
            parts.append(f"Stale: {','.join(stale_refs)}")

        icon = "✓" if not (sdc_result and sdc_result.violation) else "✗"

        self._display.display_icon_and_text(
            icon,
            " | ".join(parts),
            metadata=metadata.to_display_metadata(),
        )

    def _send_violation_error(self, violation) -> None:
        """Send Reproducibility violation as error via iopub.

        Emits structured flowbook metadata before the error so the frontend
        can store and update violation info (e.g., when cells are reordered).
        """
        # Emit structured metadata for the frontend
        metadata = ReproducibilityMetadata(
            cell_id=self._cell_id or "",
            execution_seq=self._enforcer.seq_counter,
            reads=[],
            writes=[],
            changed_variables=[],
            stale_cells=self._enforcer.get_stale_cells(),
            violation=violation.to_dict(),
            cell_order=self._enforcer.cell_order,
        )
        self._display.display_icon_and_text(
            "❌",
            "Backward violation",
            metadata=metadata.to_display_metadata(),
        )

        # Then send the error
        self.send_response(
            self.iopub_socket,
            "error",
            {
                "ename": "ReproducibilityViolation",
                "evalue": violation.message,
                "traceback": [violation.message],
            },
        )

    def _send_forward_violation_error(self, violation, writer_violation) -> None:
        """Send forward contamination violation as error via iopub.

        Emits structured flowbook metadata including writer_violation so the
        frontend can store the backward_mutation violation on the writer cell.
        """
        # Emit structured metadata for the frontend
        metadata = ReproducibilityMetadata(
            cell_id=self._cell_id or "",
            execution_seq=self._enforcer.seq_counter,
            reads=[],
            writes=[],
            changed_variables=[],
            stale_cells=self._enforcer.get_stale_cells(),
            violation=violation.to_dict(),
            cell_order=self._enforcer.cell_order,
            writer_violation=writer_violation.to_dict() if writer_violation else None,
        )
        self._display.display_icon_and_text(
            "❌",
            "Forward contamination",
            metadata=metadata.to_display_metadata(),
        )

        # Then send the error
        self.send_response(
            self.iopub_socket,
            "error",
            {
                "ename": "ReproducibilityViolation",
                "evalue": violation.message,
                "traceback": [violation.message],
            },
        )

    def _send_violation_warning(self, violation) -> None:
        """Send Reproducibility violation as warning via iopub (when continue_after_violation is enabled).

        Uses the violation.message from the enforcer verbatim.
        """
        self.send_response(
            self.iopub_socket,
            "stream",
            {"name": "stderr", "text": violation.message + "\n"},
        )

    def _send_truncation_details(self, truncation_details: str) -> None:
        """Send truncation details to stderr for user visibility."""
        self.send_response(
            self.iopub_socket,
            "stream",
            {"name": "stderr", "text": f"\n{truncation_details}\n"},
        )

    def _send_structural_warnings(self, warnings: list) -> None:
        """Send structural warnings to stderr for user visibility."""
        if not warnings:
            return
        # Separator line for visual separation (between warnings and from any preceding violation)
        separator = "\n" + "═" * 80 + "\n\n"
        # Start with separator to separate from any preceding output (like violation message)
        warning_text = separator + separator.join(warnings) + "\n"
        self.send_response(
            self.iopub_socket,
            "stream",
            {"name": "stderr", "text": warning_text},
        )

    def _make_error_result(self, violation) -> dict:
        """Create error result dict for Reproducibility violation."""
        return {
            "status": "error",
            "execution_count": self.execution_count,
            "ename": "ReproducibilityViolation",
            "evalue": violation.message,
            "traceback": [violation.message],
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def do_shutdown(self, restart: bool) -> dict:
        """Handle kernel shutdown/restart."""
        if restart:
            # Clear Reproducibility state on restart for clean slate
            self._enforcer.reset()

        # Explicitly flush timings before shutdown - atexit may not run
        # if the kernel is killed by jupyter_client after timeout
        output._print_timings()

        return super().do_shutdown(restart)


# Entry point
if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FlowbookKernel)
