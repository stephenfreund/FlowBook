"""
FerretSDCKernel - IPython kernel with Sequential Dataflow Consistency enforcement.

================================================================================
SEQUENTIAL DATAFLOW CONSISTENCY (SDC) - SPECIFICATION
================================================================================

For a formal treatment with algorithms and proofs, see analysis.md in the
project root. This docstring provides implementation-level documentation.

SDC ensures notebook reproducibility by enforcing dataflow rules that prevent
hidden state dependencies. When SDC is enforced, running cells top-to-bottom
always produces the same result as running them in any order.

================================================================================
THE THREE SDC RULES
================================================================================

Rule 1: Reproducibility Invariant (Goal)
----------------------------------------
A notebook is reproducible if running all cells in document order from a fresh
kernel produces identical results every time. This is the goal SDC enforces.

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
For pandas DataFrames, SDC tracks individual columns for precision:
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
SDC conservatively treats it as a variable-level conflict.

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

SDC uses checkpoint-based diffing to detect actual changes:

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
    │                         FerretSDCKernel                             │
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
    │  5. SDCEnforcer.check():                                            │
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
%sdc_status                          - Display current SDC state
%sdc_stale                           - Show which cells are currently stale

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
   SDC only manages Python namespace state.

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
   Status: Partially implemented (structural_reads_values in SDCExecutionRecord)

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
"""

import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

from IPython.core.magic import Magics, line_magic, magics_class
from ipykernel.ipkernel import IPythonKernel
from ipykernel.kernelapp import IPKernelApp

from data_ferret.kernel.checkpoint import Checkpoint, Checkpoints, filter_user_namespace
from data_ferret.kernel.deepcopyable import check_deepcopyable
from data_ferret.kernel.display_helpers import DisplayHelper
from data_ferret.kernel.timeout_handler import CellTimeoutHandler
from data_ferret.kernel.tracking import TrackingDict
from data_ferret.util.cell_index import index_to_alpha
from data_ferret.util.output import error, log, timer, output

from .models import SDCMetadata
from .sdc_enforcer import SDCEnforcer, PRE_CHECKPOINT_PREFIX, POST_CHECKPOINT_PREFIX


@magics_class
class FerretSDCKernel(IPythonKernel, Magics):
    """
    IPython kernel with Sequential Dataflow Consistency enforcement.

    Features:
    - Variable access tracking (reads/writes per cell)
    - SDC Rule 3 enforcement (no backward mutations)
    - Staleness computation and reporting
    - Cell order management via magic command

    SDC is always enabled. No profiling or checkpoint magics.
    """

    implementation = "ferret_sdc_kernel"
    implementation_version = "0.1"
    banner = "Ferret SDC Kernel - Sequential Dataflow Consistency"

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

        # Display helper
        self._display = DisplayHelper()

        # Current cell being executed
        self._cell_id: Optional[str] = None

        # Checkpointing (for pre/post comparison)
        self._checkpoint = Checkpoints(
            sanity_check=False,
            warn_classes=False,
        )

        # Tracking
        self._tracking = TrackingDict(self.shell.user_ns)

        # SDC enforcement
        self._sdc = SDCEnforcer(self._checkpoint)

        # Continue after violation flag (default: stop on violation)
        self._continue_after_violation: bool = False

    # =========================================================================
    # Magic Commands
    # =========================================================================

    @line_magic
    def notebook_structure(self, line: str) -> None:
        """
        Set the notebook cell order for SDC enforcement.

        Usage:
            %notebook_structure cell1 cell2 cell3 ...
        """
        cell_order = line.split()
        self._sdc.set_cell_order(cell_order)

    @line_magic
    def sdc_status(self, line: str) -> None:
        """Display current SDC state."""
        order = self._sdc.cell_order
        records = self._sdc.records

        status_lines = [
            f"Cell order: {order}",
            f"Executed cells: {list(records.keys())}",
            f"Execution counter: {self._sdc.seq_counter}",
        ]

        for cell_id, record in records.items():
            status_lines.append(
                f"  {cell_id}: reads={sorted(record.reads)}, "
                f"writes={sorted(record.writes)}, seq={record.execution_seq}"
            )

        self._display.display_icon_and_text(
            "ℹ️", "SDC Status", "\n".join(status_lines)
        )

    @line_magic
    def continue_after_violation(self, line: str) -> None:
        """
        Control whether execution continues after SDC violations.

        Usage:
            %continue_after_violation        - Enable (continue after violations, default)
            %continue_after_violation on     - Enable (continue after violations)
            %continue_after_violation off    - Disable (stop on violation)
            %continue_after_violation ?      - Show current status
        """
        arg = line.strip().lower()

        if arg == "?":
            status = "on" if self._continue_after_violation else "off"
            self._display.display_icon_and_text("ℹ️", f"Continue after violation: {status}")
            return

        if not arg or arg in ("on", "true", "1", "enable"):
            self._continue_after_violation = True
            self._display.display_icon_and_text("ℹ️", "Continue after violation: enabled")
        elif arg in ("off", "false", "0", "disable"):
            self._continue_after_violation = False
            self._display.display_icon_and_text("ℹ️", "Continue after violation: disabled")
        else:
            self._display.display_icon_and_text(
                "⚠️", f"Invalid: '{arg}'. Use 'on', 'off', or '?'"
            )

    @line_magic
    def sdc_stale(self, line: str) -> None:
        """
        Show which cells are currently stale.

        Usage:
            %sdc_stale
        """
        stale_cells = self._sdc.get_stale_cells()
        if not stale_cells:
            self._display.display_icon_and_text("✓", "No stale cells")
            return

        # Convert to @A notation
        stale_refs = []
        for cell_id in stale_cells:
            try:
                idx = self._sdc.cell_order.index(cell_id)
                stale_refs.append(index_to_alpha(idx))
            except (ValueError, IndexError):
                stale_refs.append(cell_id)

        self._display.display_icon_and_text("⚠️", f"Stale cells: {', '.join(stale_refs)}")

    @line_magic
    def structural_tracking(self, line: str) -> None:
        """
        Set structural tracking mode for DataFrame/Series attribute monitoring.

        Structural tracking detects when code accesses attributes that reveal
        DataFrame/Series structure (like df.columns, df.shape, len(df)).
        When structural tracking is enabled and these attributes are read,
        subsequent changes to the structure (adding columns, changing row count)
        are either warned about or treated as SDC violations.

        Usage:
            %structural_tracking           - Show current mode
            %structural_tracking off       - Disable structural tracking
            %structural_tracking warn      - Track and warn only (default)
            %structural_tracking enforce   - Track and treat changes as violations
        """
        from data_ferret.kernel.structural_tracking import StructuralTrackingMode

        # Suspend tracking during magic execution to avoid recording infrastructure reads
        tracking = self._tracking
        if tracking is not None and hasattr(tracking, 'suspended'):
            ctx = tracking.suspended()
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            mode_str = line.strip().lower()

            if not mode_str:
                # Show current mode
                current_mode = self._sdc.structural_mode.value
                self._display.display_icon_and_text(
                    "🔍",
                    f"Structural tracking mode: {current_mode}"
                )
                return

            try:
                mode = StructuralTrackingMode(mode_str)
            except ValueError:
                self._display.display_icon_and_text(
                    "❌",
                    f"Invalid mode: {mode_str}. Use 'off', 'warn', or 'enforce'"
                )
                return

            # Update SDC enforcer
            self._sdc.set_structural_mode(mode)

            # Update TrackingDict if it exists
            if tracking is not None:
                tracking.set_structural_tracking_mode(mode_str)

            self._display.display_icon_and_text(
                "✅",
                f"Structural tracking mode set to: {mode.value}"
            )

    # =========================================================================
    # Stub magics for compatibility with FerretKernel notebooks
    # =========================================================================

    @line_magic
    def scalene(self, line: str) -> None:
        """Stub for Scalene profiling (not supported in SDC kernel)."""
        self._display.display_icon_and_text("ℹ️", "Scalene profiling not supported in SDC kernel")

    @line_magic
    def tracking(self, line: str) -> None:
        """Stub for global tracking (always on in SDC kernel)."""
        self._display.display_icon_and_text("ℹ️", "Global tracking is always enabled in SDC kernel")

    @line_magic
    def monotone(self, line: str) -> None:
        """Stub for monotone enforcement (use SDC rules instead)."""
        self._display.display_icon_and_text("ℹ️", "SDC kernel uses SDC rules instead of monotone enforcement")

    @line_magic
    def force_checkpoints(self, line: str) -> None:
        """Stub for force checkpoints (SDC kernel always checkpoints)."""
        self._display.display_icon_and_text("ℹ️", "SDC kernel always takes checkpoints")

    @line_magic
    def checkpoint(self, line: str) -> None:
        """Stub for checkpoint magic (not supported in SDC kernel)."""
        self._display.display_icon_and_text("ℹ️", "Manual checkpoints not supported in SDC kernel")

    @line_magic
    def restore(self, line: str) -> None:
        """Stub for restore magic (not supported in SDC kernel)."""
        self._display.display_icon_and_text("ℹ️", "Restore not supported in SDC kernel")

    @line_magic
    def list_checkpoints(self, line: str) -> None:
        """Stub for list checkpoints magic (not supported in SDC kernel)."""
        self._display.display_icon_and_text("ℹ️", "List checkpoints not supported in SDC kernel")

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

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[dict] = None,
        allow_stdin: bool = False,
        *,
        cell_meta: Optional[dict] = None,
        cell_id: Optional[str] = None,
    ) -> dict:
        """
        Execute code with SDC tracking and enforcement.
        """
        try:
            # Ensure tracking is initialized (done lazily on first execution)
            self._ensure_tracking_initialized()

            # Extract cell context
            self._cell_id = self._extract_cell_id(cell_id, cell_meta)

            # Update cell order if provided in metadata
            if cell_meta and "cell_order" in cell_meta:
                self._sdc.set_cell_order(cell_meta["cell_order"])

            # Check for notebook_structure magic (parse and remove if present)
            code = self._process_structure_magic(code)

            # Extract timeout from code directive or cell_meta
            code, timeout = self._extract_timeout(code, cell_meta)

            # Skip SDC for empty code or pure magic
            if not code.strip() or self._is_pure_magic(code):
                return await self._execute_without_sdc(
                    code,
                    silent,
                    store_history,
                    user_expressions,
                    allow_stdin,
                    cell_meta,
                )

            # Take pre-execution snapshot
            user_ns = self.shell.user_ns
            with timer(key="sdc_pre_checkpoint") as pre_timer:
                pre_checkpoint = self._take_checkpoint(f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}")

            # Reset tracking for this execution
            if isinstance(user_ns, TrackingDict):
                user_ns.reset_tracking()

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
                with timer(key="sdc_execute") as run_timer:
                    if isinstance(user_ns, TrackingDict):
                        with user_ns.track_execution():
                            result = await super().do_execute(
                                code,
                                silent,
                                store_history,
                                user_expressions,
                                allow_stdin,
                                cell_meta=cell_meta,
                                cell_id=self._cell_id,
                            )
                        tracking = user_ns.get_tracking_data()
                    else:
                        result = await super().do_execute(
                            code,
                            silent,
                            store_history,
                            user_expressions,
                            allow_stdin,
                            cell_meta=cell_meta,
                            cell_id=self._cell_id,
                        )
                        tracking = None
                normal_exit = True
            except KeyboardInterrupt:
                # Timeout occurred - restore pre-state
                self._sdc.checkpoints.restore(
                    f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}", self.shell.user_ns
                )
                return self._handle_timeout_error(timeout)
            finally:
                timeout_handler.cancel()
                if not normal_exit:
                    await timeout_handler.cleanup_on_error()

            # If execution had an error, restore pre-state and skip SDC checks
            if result.get("status") == "error":
                self._sdc.checkpoints.restore(
                    f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}", self.shell.user_ns
                )
                return result

            # Warn about non-deepcopyable objects after successful execution
            with timer(key="warn_non_deepcopyable", message="Warn non-deepcopyable"):
                self._warn_non_deepcopyable_objects()

            # Take post-execution snapshot
            with timer(key="sdc_post_checkpoint") as post_timer:
                post_checkpoint = self._take_checkpoint(f"{POST_CHECKPOINT_PREFIX}{self._cell_id}")

            # Run SDC check if we have tracking data and cell_id
            if tracking and self._cell_id:
                with timer(key="sdc_check") as check_timer:
                    sdc_result = self._sdc.check(
                        cell_id=self._cell_id,
                        pre_checkpoint=pre_checkpoint,
                        post_checkpoint=post_checkpoint,
                        tracking=tracking,
                        continue_on_violation=self._continue_after_violation,
                        namespace=self.shell.user_ns,  # For capturing structural read values
                    )

                # Handle violations (backward mutation and/or forward dependency)
                has_backward = sdc_result and sdc_result.violation
                has_forward = sdc_result and sdc_result.forward_violation

                if has_backward or has_forward:
                    if self._continue_after_violation:
                        # Warn about both violations but continue
                        if has_backward:
                            if sdc_result.violation.truncation_details:
                                error(f"SDC truncation: {sdc_result.violation.message}")
                                self._send_truncation_details(sdc_result.violation.truncation_details)
                            error(f"SDC violation (continuing): {sdc_result.violation.message}")
                            self._send_violation_warning(sdc_result.violation)
                        if has_forward:
                            error(f"Forward dependency (continuing): {sdc_result.forward_violation.message}")
                            self._send_violation_warning(sdc_result.forward_violation)
                    else:
                        # Block on violation - backward takes precedence
                        primary = sdc_result.violation if has_backward else sdc_result.forward_violation

                        # Log truncation issues to terminal (for backward mutations)
                        if has_backward and sdc_result.violation.truncation_details:
                            error(f"SDC truncation: {sdc_result.violation.message}")
                            self._send_truncation_details(sdc_result.violation.truncation_details)

                        error(f"SDC violation: {primary.message}")

                        # Only rollback for backward mutations (forward deps didn't change anything)
                        if has_backward:
                            self._sdc.checkpoints.restore(
                                f"{PRE_CHECKPOINT_PREFIX}{self._cell_id}", self.shell.user_ns
                            )

                        self._send_violation_error(primary)

                        # Also report forward violation if both exist
                        if has_backward and has_forward:
                            self._send_violation_warning(sdc_result.forward_violation)

                        return self._make_error_result(primary)

                # Display results (skip if silent, error, or violation with rollback)
                skip_display = (has_backward or has_forward) and not self._continue_after_violation
                if not silent and result.get("status") != "error" and not skip_display:
                    state_ms = pre_timer.duration() + post_timer.duration()
                    self._display_execution_result(
                        run_timer.duration(),
                        state_ms,
                        check_timer.duration(),
                        tracking,
                        sdc_result,
                    )

            return result
        except Exception as e:
            error(f"SDC error in cell {self._cell_id}: {e}\n{traceback.format_exc()}")
            raise

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_cell_alpha(self) -> str:
        """Get @A notation for current cell."""
        if self._cell_id is None:
            return "unknown"
        try:
            index = self._sdc.cell_order.index(self._cell_id)
            return index_to_alpha(index)
        except (ValueError, IndexError):
            return self._cell_id

    def _extract_cell_id(
        self, cell_id: Optional[str], cell_meta: Optional[dict]
    ) -> Optional[str]:
        """Extract cell ID from arguments or metadata."""
        if cell_id is not None:
            return cell_id
        if cell_meta is not None:
            return cell_meta.get("cell_id")
        return None

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
                self._sdc.set_cell_order(parts)
            return "\n".join(lines[1:])
        return code

    def _is_pure_magic(self, code: str) -> bool:
        """Check if code is only magic commands (with optional comments)."""
        lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
        # Allow magics (%), shell commands (!), and comments (#)
        return all(
            line.startswith("%") or line.startswith("!") or line.startswith("#")
            for line in lines
        )

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
            {"ename": "TimeoutError", "evalue": timeout_msg, "traceback": [timeout_msg]},
        )
        return {
            "status": "error",
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
                message = f"The object {k} (type {typ}) cannot be checkpointed: {reason}"
                log(message)
                self._display.display_icon_and_text(
                    "\u26A0\uFE0F",
                    message
                )

    def _take_checkpoint(self, checkpoint_name: str) -> Checkpoint:
        """
        Take a snapshot of checkpointable variables before execution.

        Uses Checkpoints.save() to properly deep copy the namespace.
        This is critical for correct operation with TrackingDict.
        """
        # Convert TrackingDict to regular dict for checkpointing
        self._checkpoint.save(
            checkpoint_name, dict(self.shell.user_ns), max_size_mb=None
        )
        return self._checkpoint.saved[checkpoint_name]

    async def _execute_without_sdc(
        self,
        code: str,
        silent: bool,
        store_history: bool,
        user_expressions: Optional[dict],
        allow_stdin: bool,
        cell_meta: Optional[dict],
    ) -> dict:
        """Execute without SDC tracking (for magics, empty code)."""
        result = await super().do_execute(
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
            empty_metadata = SDCMetadata(
                cell_id=self._cell_id,
                execution_seq=self._sdc.seq_counter,
                reads=[],
                writes=[],
                changed_variables=[],
                stale_cells=self._sdc.get_stale_cells(),
                violation=None,
                cell_order=self._sdc.cell_order,
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
        run_duration: float,
        state_duration: float,
        check_duration: float,
        tracking,
        sdc_result,
    ) -> None:
        """Display execution timing and SDC metadata."""
        # Build metadata for display
        structural_warnings = (
            sdc_result.structural_warnings if sdc_result else []
        )

        # Use TrackingData.to_json_friendly() for clean serialization
        tracking_json = tracking.to_json_friendly() if tracking else {
            "reads": [],
            "writes": [],
            "column_reads": {},
            "column_writes": {},
            "structural_reads": {},
        }

        metadata = SDCMetadata(
            cell_id=self._cell_id or "",
            execution_seq=self._sdc.seq_counter,
            reads=tracking_json["reads"],
            writes=tracking_json["writes"],
            changed_variables=sdc_result.changed_variables if sdc_result else [],
            stale_cells=sdc_result.stale_cells if sdc_result else [],
            violation=(
                sdc_result.violation.to_dict()
                if (sdc_result and sdc_result.violation)
                else None
            ),
            cell_order=self._sdc.cell_order,
            column_reads=tracking_json["column_reads"],
            column_writes=tracking_json["column_writes"],
            column_changed=sdc_result.column_changed if sdc_result else {},
            structural_reads=tracking_json["structural_reads"],
            structural_warnings=structural_warnings,
            run_duration_ms=run_duration,
            state_duration_ms=state_duration,
            check_duration_ms=check_duration,
        )

        # Log and display structural warnings
        if structural_warnings:
            for warning in structural_warnings:
                error(f"[structural] {warning}")
            self._send_structural_warnings(structural_warnings)

        # Build display text
        parts = [
            f"Run: {run_duration:.0f} ms",
            f"State: {state_duration:.0f} ms",
            f"Check: {check_duration:.0f} ms",
        ]
        if tracking_json["reads"] or tracking_json["column_reads"]:
            # Combine variable-level reads with column-level read variables
            read_vars = set(tracking_json["reads"]) | set(tracking_json["column_reads"].keys())
            if read_vars:
                reads_preview = [
                    self._format_var_with_columns(v, tracking_json["column_reads"], {})
                    for v in list(read_vars)[:3]
                ]
                parts.append(f"Reads: {','.join(reads_preview)}")
        # Show writes if there are variable-level writes OR column-level writes
        write_vars = set(tracking_json["writes"]) | set(tracking_json["column_writes"].keys())
        if write_vars:
            writes_preview = [
                self._format_var_with_columns(v, {}, tracking_json["column_writes"])
                for v in list(write_vars)[:3]
            ]
            parts.append(f"Writes: {','.join(writes_preview)}")

        if sdc_result and sdc_result.stale_cells:
            # Convert cell IDs to @A references for display
            stale_refs = []
            for cell_id in sdc_result.stale_cells:
                try:
                    idx = self._sdc.cell_order.index(cell_id)
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
        """Send SDC violation as error via iopub.

        Uses the violation.message from the enforcer verbatim.
        """
        self.send_response(
            self.iopub_socket,
            "error",
            {
                "ename": "SDCViolation",
                "evalue": violation.message,
                "traceback": [violation.message],
            },
        )

    def _send_violation_warning(self, violation) -> None:
        """Send SDC violation as warning via iopub (when continue_after_violation is enabled).

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
        """Create error result dict for SDC violation."""
        return {
            "status": "error",
            "execution_count": self.execution_count,
            "ename": "SDCViolation",
            "evalue": violation.message,
            "traceback": [violation.message],
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def do_shutdown(self, restart: bool) -> dict:
        """Handle kernel shutdown/restart."""
        if restart:
            # Clear SDC state on restart for clean slate
            self._sdc.reset()

        # Explicitly flush timings before shutdown - atexit may not run
        # if the kernel is killed by jupyter_client after timeout
        output._print_timings()

        return super().do_shutdown(restart)


# Entry point
if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FerretSDCKernel)
