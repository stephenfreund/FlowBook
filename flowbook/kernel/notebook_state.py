"""
Notebook state for reproducibility tracking.

This module implements the core state model for tracking notebook reproducibility:
    S = ⟨C, O, Σ, T, R, W⟩

Where:
    C, O, Σ: Managed elsewhere (notebook content, outputs, kernel namespace)
    T: Cell → Status (managed here)
    R: Cell → P(Loc) - reads per cell (managed here)
    W: Cell → P(Loc) - writes per cell (managed here)

This module also stores per-cell TrackingData for:
    - Column-level reads/writes for DataFrames
    - Structural attribute reads (df.columns, df.shape, etc.)
    - File I/O tracking

And additional per-cell metadata:
    - execution_seq: Monotonic execution counter
    - structural_reads_values: Captured values for error messages
    - typed_changes: Cached Change objects for fast forward dependency checks
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, TYPE_CHECKING

from flowbook.kernel.loc_ids import StableIdMap
from flowbook.kernel.models import CellStateSnapshot, CellStatus, Reason, ReasonType
from flowbook.kernel.locations import (
    ReadLoc, ReadLocSet, WriteLoc, WriteLocSet,
    has_conflict, wlocs_conflict_rlocs, wlocs_conflict_wlocs,
    writelocset_var_names, readlocset_var_names,
    tracking_to_readlocset, tracking_to_writelocset,
)
from flowbook.kernel.change_detector import changes_to_write_locs
from flowbook.kernel_support.models import TrackingData

if TYPE_CHECKING:
    from flowbook.kernel.changes import Change


@dataclass
class NotebookState:
    """
    Instrumentation state for reproducibility tracking.

    This class maintains the state needed to track cell staleness and detect
    reproducibility issues. It implements the formal model transitions:
    - EDIT: Mark cell stale with CodeChanged reason
    - EXEC: Record reads/writes, propagate staleness
    - INSERT/DELETE/MOVE: Handle structural changes

    Attributes:
        # Formal model state (T, R, W)
        status: Map from cell ID to CellStatus (T in formal model)
        reads: Map from cell ID to set of locations read (R in formal model)
        writes: Map from cell ID to set of locations written (W in formal model)
        cell_order: List of cell IDs in document order

        # Per-cell tracking data
        tracking_data: TrackingData per cell (Pydantic model for frontend communication)
        execution_seq: Execution sequence number per cell
        structural_reads_values: Captured structural values for error messages
        typed_changes: Cached typed Change objects for fast forward dependency checks
    """

    # Formal model state (T, R, W)
    status: Dict[str, CellStatus] = field(default_factory=dict)
    reads: Dict[str, ReadLocSet] = field(default_factory=dict)
    writes: Dict[str, WriteLocSet] = field(default_factory=dict)
    cell_order: List[str] = field(default_factory=list)

    # Per-cell tracking data
    tracking_data: Dict[str, TrackingData] = field(default_factory=dict)  # cell_id -> TrackingData
    execution_seq: Dict[str, int] = field(default_factory=dict)  # cell_id -> seq number
    structural_reads_values: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)  # cell_id -> var -> attr -> value
    typed_changes: Dict[str, List["Change"]] = field(default_factory=dict)  # cell_id -> changes
    # Canonical AST fingerprint of the source last executed for each cell.
    # Used by [Inst-Edit] to decide whether a source edit is meaningful: an edit
    # whose fingerprint matches this baseline is treated as cosmetic (clears
    # CODE_CHANGED) rather than marking the cell stale.
    fingerprints: Dict[str, str] = field(default_factory=dict)  # cell_id -> AST fingerprint

    # =========================================================================
    # Status Access
    # =========================================================================

    def get_status(self, cell_id: str) -> CellStatus:
        """Get cell status, defaulting to Stale(NeverExecuted)."""
        if cell_id not in self.status:
            self.status[cell_id] = CellStatus.never_executed()
        return self.status[cell_id]

    def is_clean(self, cell_id: str) -> bool:
        """Check if cell is Clean."""
        return self.get_status(cell_id).is_clean

    def set_clean(self, cell_id: str) -> None:
        """Mark cell as Clean (clears all reasons)."""
        self.status[cell_id] = CellStatus.clean()

    def set_stale(self, cell_id: str, reasons: Set[Reason]) -> None:
        """Mark cell as Stale with given reasons (replaces existing)."""
        self.status[cell_id] = CellStatus.stale(reasons)

    def add_reason(self, cell_id: str, reason: Reason) -> None:
        """Add a reason to a cell (accumulates with existing reasons)."""
        self.get_status(cell_id).add_reason(reason)

    def get_reasons(self, cell_id: str) -> Set[Reason]:
        """Get all reasons for a cell's staleness."""
        return self.get_status(cell_id).reasons

    def set_fingerprint(self, cell_id: str, fingerprint: Optional[str]) -> None:
        """Store the AST fingerprint of the source last executed for a cell.

        Passing None (unparseable source) clears any stored fingerprint, which
        makes subsequent edits fall through to the conservative mark-stale path.
        """
        if fingerprint is None:
            self.fingerprints.pop(cell_id, None)
        else:
            self.fingerprints[cell_id] = fingerprint

    def get_fingerprint(self, cell_id: str) -> Optional[str]:
        """Return the AST fingerprint of the source last executed for a cell."""
        return self.fingerprints.get(cell_id)

    def clear_pre_execution_reasons(self, cell_id: str) -> None:
        """Clear pre-execution reasons (NEVER_EXECUTED, CODE_CHANGED) from a cell.

        Called when a cell executes - these reasons are no longer valid since
        the cell has been executed. If this was the only reason, marks cell clean.
        """
        status = self.get_status(cell_id)
        pre_exec_types = {ReasonType.NEVER_EXECUTED, ReasonType.CODE_CHANGED}
        status.reasons = {r for r in status.reasons if r.type not in pre_exec_types}
        # If no reasons remain, mark clean
        if not status.reasons:
            status.is_clean = True

    # =========================================================================
    # Stale Cell Queries
    # =========================================================================

    def get_stale_cells(self) -> List[str]:
        """Return list of stale cell IDs in document order."""
        return [c for c in self.cell_order if not self.is_clean(c)]

    def get_all_reasons(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get reasons for all stale cells (for metadata output).

        Reasons are sorted by priority so the most specific reason comes first:
        1. FORWARD_STALE (tells you which variable changed)
        2. CODE_CHANGED (cell was edited)
        3. BACKWARD_STALE, NO_READ_BEFORE_WRITE
        4. ORDER_CHANGED (least specific)
        5. NEVER_EXECUTED
        """
        # Priority order: lower = higher priority (shown first)
        priority = {
            ReasonType.FORWARD_STALE: 1,
            ReasonType.CODE_CHANGED: 2,
            ReasonType.BACKWARD_STALE: 3,
            ReasonType.NO_READ_BEFORE_WRITE: 4,
            ReasonType.ORDER_CHANGED: 5,
            ReasonType.NEVER_EXECUTED: 6,
        }

        result: Dict[str, List[Dict[str, Any]]] = {}
        for cell_id in self.cell_order:
            status = self.get_status(cell_id)
            if not status.is_clean:
                # Sort reasons by priority
                sorted_reasons = sorted(
                    status.reasons,
                    key=lambda r: (priority.get(r.type, 99), r.loc or "", r.cell_id or "")
                )
                result[cell_id] = [r.to_dict() for r in sorted_reasons]
        return result

    # =========================================================================
    # Derived Functions (from formal model)
    # =========================================================================

    def last_writer_for(self, var_name: str, before_cell: str) -> Optional[str]:
        """
        LastWriter(W, i, x) = max { j < i | x ∈ W_j }, or None

        Find which cell before `before_cell` should have written `var_name`
        based on document order and recorded writes.

        Operates at variable level: checks if any WriteLoc in a cell's writes
        has a matching var_name().
        """
        if before_cell not in self.cell_order:
            return None

        before_pos = self.cell_order.index(before_cell)
        result: Optional[str] = None
        result_pos = -1

        for cell_id, cell_writes in self.writes.items():
            if any(w.var_name() == var_name for w in cell_writes) and cell_id in self.cell_order:
                pos = self.cell_order.index(cell_id)
                if pos < before_pos and pos > result_pos:
                    result = cell_id
                    result_pos = pos

        return result

    # =========================================================================
    # Transitions
    # =========================================================================

    def record_execution(
        self,
        cell_id: str,
        tracking: TrackingData,
        execution_seq: Optional[int] = None,
        structural_reads_values: Optional[Dict[str, Dict[str, str]]] = None,
        typed_changes: Optional[List["Change"]] = None,
        namespace: Optional[dict] = None,
        stable_map: Optional[StableIdMap] = None,
    ) -> None:
        """
        Record cell execution: update R, W, and tracking data.

        Called after successful execution (not on rollback).
        This corresponds to the "commit" phase of EXEC.

        Args:
            cell_id: The cell that executed
            tracking: TrackingData from cell execution (contains reads, writes, columns, etc.)
            execution_seq: Execution sequence number
            structural_reads_values: Captured structural values for error messages
            typed_changes: Cached typed changes for fast forward dependency checks
            namespace: Current kernel namespace (for LocRef qualifiers)
            stable_map: StableIdMap instance (for LocRef qualifiers)
        """
        # Store the TrackingData object directly
        self.tracking_data[cell_id] = tracking

        # Core R, W tracking (derived from TrackingData) — typed LocSets
        self.reads[cell_id] = tracking_to_readlocset(tracking, namespace, stable_map)
        tracking_wlocs = tracking_to_writelocset(tracking, namespace, stable_map)
        # Merge diff-derived WriteLocs (Col, Rows, Attr) when available.
        # tracking_to_writelocset only produces Var + Col + File; the diff provides
        # the full typed set needed for column-level write-write overlap via output().
        # Only include diff-derived locs for variables that tracking also considers
        # as writes — otherwise unrecoverable mutations (in-place changes not tracked
        # as writes) would incorrectly appear as writes in last_writer_for().
        if typed_changes:
            tracking_write_vars = (tracking.writes or set()) | set(tracking.column_writes.keys() if tracking.column_writes else [])
            diff_wlocs = changes_to_write_locs(typed_changes, namespace, stable_map)
            recoverable_diff_wlocs = frozenset(
                w for w in diff_wlocs if w.var_name() in tracking_write_vars
            )
            self.writes[cell_id] = tracking_wlocs | recoverable_diff_wlocs
        else:
            self.writes[cell_id] = tracking_wlocs

        # Additional per-cell metadata (not in TrackingData)
        if execution_seq is not None:
            self.execution_seq[cell_id] = execution_seq
        if structural_reads_values is not None:
            self.structural_reads_values[cell_id] = structural_reads_values
        if typed_changes is not None:
            self.typed_changes[cell_id] = typed_changes

    def snapshot_cell_state(self, cell_id: str) -> CellStateSnapshot:
        """
        Capture current state for a cell before execution.

        Used to enable rollback if execution is rejected. Returns a
        CellStateSnapshot that can be passed to restore_cell_state().

        Args:
            cell_id: The cell about to execute

        Returns:
            CellStateSnapshot containing all state that will be modified
        """
        return CellStateSnapshot(
            cell_id=cell_id,
            reads=self.reads.get(cell_id),
            writes=self.writes.get(cell_id),
            status=self.status.get(cell_id),
            tracking_data=self.tracking_data.get(cell_id),
            execution_seq=self.execution_seq.get(cell_id),
            structural_reads_values=self.structural_reads_values.get(cell_id),
            typed_changes=self.typed_changes.get(cell_id),
        )

    def restore_cell_state(self, snapshot: CellStateSnapshot) -> None:
        """
        Restore cell state from a snapshot (undo record_execution).

        Called when kernel rolls back a rejected execution. This restores
        the enforcer's analysis state to match the rolled-back namespace.

        Args:
            snapshot: CellStateSnapshot from snapshot_cell_state()
        """
        cell_id = snapshot.cell_id

        # Restore or clear per-cell state
        if snapshot.reads is None:
            self.reads.pop(cell_id, None)
        else:
            self.reads[cell_id] = snapshot.reads

        if snapshot.writes is None:
            self.writes.pop(cell_id, None)
        else:
            self.writes[cell_id] = snapshot.writes

        if snapshot.status is None:
            self.status.pop(cell_id, None)
        else:
            self.status[cell_id] = snapshot.status

        if snapshot.tracking_data is None:
            self.tracking_data.pop(cell_id, None)
        else:
            self.tracking_data[cell_id] = snapshot.tracking_data

        if snapshot.execution_seq is None:
            self.execution_seq.pop(cell_id, None)
        else:
            self.execution_seq[cell_id] = snapshot.execution_seq

        if snapshot.structural_reads_values is None:
            self.structural_reads_values.pop(cell_id, None)
        else:
            self.structural_reads_values[cell_id] = snapshot.structural_reads_values

        if snapshot.typed_changes is None:
            self.typed_changes.pop(cell_id, None)
        else:
            self.typed_changes[cell_id] = snapshot.typed_changes

    # =========================================================================
    # Per-Cell Tracking Data Access
    # =========================================================================

    def get_tracking(self, cell_id: str) -> Optional[TrackingData]:
        """Get TrackingData for a cell, or None if not executed."""
        return self.tracking_data.get(cell_id)

    def get_column_reads(self, cell_id: str) -> Dict[str, Set[str]]:
        """Get column-level reads for a cell {var: {cols}}."""
        tracking = self.tracking_data.get(cell_id)
        if tracking is None:
            return {}
        return tracking.get_column_rbw_sets()

    def get_column_writes(self, cell_id: str) -> Dict[str, Set[str]]:
        """Get column-level writes for a cell {var: {cols}}."""
        tracking = self.tracking_data.get(cell_id)
        if tracking is None:
            return {}
        return {k: set(v) for k, v in tracking.column_writes.items()}

    def get_structural_reads(self, cell_id: str) -> Dict[str, Set[str]]:
        """Get structural attribute reads for a cell {var: {attrs}}."""
        tracking = self.tracking_data.get(cell_id)
        if tracking is None:
            return {}
        return {k: set(v) for k, v in tracking.structural_reads.items()}

    def get_file_reads(self, cell_id: str) -> Set[str]:
        """Get file paths read by a cell."""
        tracking = self.tracking_data.get(cell_id)
        if tracking is None:
            return set()
        return set(tracking.file_reads_before_writes)

    def get_file_writes(self, cell_id: str) -> Set[str]:
        """Get file paths written by a cell."""
        tracking = self.tracking_data.get(cell_id)
        if tracking is None:
            return set()
        return set(tracking.file_writes)

    def get_execution_seq(self, cell_id: str) -> Optional[int]:
        """Get execution sequence number for a cell."""
        return self.execution_seq.get(cell_id)

    def get_structural_reads_values(self, cell_id: str) -> Dict[str, Dict[str, str]]:
        """Get captured structural values for a cell {var: {attr: value}}."""
        return self.structural_reads_values.get(cell_id, {})

    def get_typed_changes(self, cell_id: str) -> List["Change"]:
        """Get cached typed changes for a cell."""
        return self.typed_changes.get(cell_id, [])

    def has_record(self, cell_id: str) -> bool:
        """Check if a cell has been executed (has tracking data)."""
        return cell_id in self.tracking_data

    def propagate_staleness(self, writer_cell: str, written_locs: WriteLocSet) -> None:
        """
        Propagate staleness to later cells.

        For j in {i+1, ..., n}:
            for w ∈ W' ▷ R_j: AddReason(j, InputChanged(w, i))
            for w ∈ W' ▷ output*(W_j): AddReason(j, WriteConflict(w, i))
        """
        if writer_cell not in self.cell_order:
            return

        writer_pos = self.cell_order.index(writer_cell)

        for later_cell in self.cell_order[writer_pos + 1:]:
            if not self.is_clean(later_cell):
                continue  # Already stale, skip

            later_reads = self.reads.get(later_cell, frozenset())
            later_writes = self.writes.get(later_cell, frozenset())

            # ForwardStale: W' ▷ R_j
            conflicting_writes = wlocs_conflict_rlocs(written_locs, later_reads)
            for w in conflicting_writes:
                self.add_reason(later_cell, Reason(
                    ReasonType.FORWARD_STALE, loc=w.display_name(), cell_id=writer_cell
                ))

            # BackwardStale: W' ▷▷ W_j (direct write-write conflict)
            write_conflicting = wlocs_conflict_wlocs(written_locs, later_writes)
            for w in write_conflicting - conflicting_writes:
                self.add_reason(later_cell, Reason(
                    ReasonType.BACKWARD_STALE, loc=w.display_name(), cell_id=writer_cell
                ))

    def handle_edit(self, cell_id: str) -> None:
        """
        EDIT transition [Inst-Edit]: T' = T[i := stale], R and W unchanged.

        Per the formal semantics, editing a cell only changes its status to stale.
        R and W are preserved so that:
        - ForwardStale on rerun can compute W_i \\ W'_i (removed writes)
        - BackwardStale on rerun can detect when a cell stops writing a variable
        - Other cells' staleness propagation still sees the old R/W
        """
        self.set_stale(cell_id, {Reason(ReasonType.CODE_CHANGED)})

    def handle_delete(self, deleted_cell: str) -> None:
        """
        DELETE transition — [Inst-Delete]:
        1. Compute ForwardStale and BackwardStale using same predicates as [Inst-Run]
        2. Keep L(x) pointing to deleted cell (for orphan detection)
        3. Remove deleted cell from status/reads/writes/cell_order

        ForwardStale: j > i, Wᵢ ▷ Rⱼ ≠ ∅ or Wᵢ ▷ output*(Wⱼ) ≠ ∅
        BackwardStale: j < i, j = LastWriter(W, i, y) for y ∈ Wᵢ

        Note: We intentionally keep last_writer pointing to the deleted cell
        so that forward dependency checks can detect "orphaned values" -
        values that came from a cell that no longer exists.
        """
        deleted_writes = self.writes.get(deleted_cell, frozenset())

        if deleted_writes and deleted_cell in self.cell_order:
            my_position = self.cell_order.index(deleted_cell)

            # ForwardStale: j > i, Wᵢ ▷ (Rⱼ ∪ output*(Wⱼ)) ≠ ∅
            for cell_id in self.cell_order[my_position + 1:]:
                cell_reads = self.reads.get(cell_id, frozenset())
                cell_writes = self.writes.get(cell_id, frozenset())
                read_conflicting = wlocs_conflict_rlocs(deleted_writes, cell_reads)
                write_conflicting = wlocs_conflict_wlocs(deleted_writes, cell_writes)

                for w in read_conflicting:
                    self.add_reason(cell_id, Reason(
                        ReasonType.FORWARD_STALE, loc=w.display_name(), cell_id=deleted_cell
                    ))
                for w in write_conflicting - read_conflicting:
                    self.add_reason(cell_id, Reason(
                        ReasonType.WRITE_OVERLAP, loc=w.display_name(), cell_id=deleted_cell
                    ))

            # BackwardStale: j < i, j = LastWriter(W, i, y) for y ∈ Wᵢ
            # Snapshot clean state — formal rule checks original Tⱼ
            originally_clean = {
                cell_id for cell_id in self.cell_order[:my_position]
                if self.is_clean(cell_id)
            }
            for w in deleted_writes:
                var_name = w.var_name()
                last_j = None
                for cell_id in self.cell_order[:my_position]:
                    cell_w = self.writes.get(cell_id, frozenset())
                    if any(ww.var_name() == var_name for ww in cell_w):
                        last_j = cell_id
                if last_j is not None and last_j in originally_clean:
                    self.add_reason(last_j, Reason(
                        ReasonType.BACKWARD_STALE, loc=w.display_name(), cell_id=deleted_cell
                    ))

        # Remove deleted cell from state
        self.status.pop(deleted_cell, None)
        self.reads.pop(deleted_cell, None)
        self.writes.pop(deleted_cell, None)
        # Also remove per-cell tracking data
        self.tracking_data.pop(deleted_cell, None)
        self.execution_seq.pop(deleted_cell, None)
        self.structural_reads_values.pop(deleted_cell, None)
        self.typed_changes.pop(deleted_cell, None)
        self.fingerprints.pop(deleted_cell, None)
        if deleted_cell in self.cell_order:
            self.cell_order.remove(deleted_cell)

    def handle_insert(self, cell_id: str, position: int) -> None:
        """
        INSERT transition: new cell at position with Stale({NeverExecuted})

        Also checks if insertion affects Runnable for later cells.
        """
        # Insert into cell_order at position
        if position < 0:
            position = 0
        if position > len(self.cell_order):
            position = len(self.cell_order)

        self.cell_order.insert(position, cell_id)
        self.status[cell_id] = CellStatus.never_executed()
        self.reads[cell_id] = frozenset()
        self.writes[cell_id] = frozenset()

        # Cells after insertion point may be affected by order change
        # but without provenance tracking we cannot check Runnable here.

    def handle_move(self, cell_id: str, new_position: int) -> None:
        """
        MOVE transition: reposition cell and check affected cells.
        """
        if cell_id not in self.cell_order:
            return

        old_position = self.cell_order.index(cell_id)
        if old_position == new_position:
            return

        # Remove from old position
        self.cell_order.remove(cell_id)

        # new_position is the target index in the final list (no adjustment needed)

        # Clamp to valid range
        if new_position < 0:
            new_position = 0
        if new_position > len(self.cell_order):
            new_position = len(self.cell_order)

        # Insert at new position
        self.cell_order.insert(new_position, cell_id)

        # Cells in affected range may be affected by order change
        # but without provenance tracking we cannot check Runnable here.

    def set_cell_order(self, new_order: List[str]) -> List[str]:
        """
        Update cell order and handle structural changes.

        Detects deletions, insertions, and reorderings.
        Returns list of cells that became newly stale.

        Args:
            new_order: New list of cell IDs in document order

        Returns:
            List of cell IDs that became stale due to this change
        """
        old_set = set(self.cell_order)
        new_set = set(new_order)
        previously_stale = set(self.get_stale_cells())

        # Handle deletions
        for deleted in old_set - new_set:
            self.handle_delete(deleted)

        # Handle insertions (new cells not in old order)
        for inserted in new_set - old_set:
            pos = new_order.index(inserted)
            # Don't use handle_insert as it modifies cell_order
            # Just initialize the new cell's state
            self.status[inserted] = CellStatus.never_executed()
            self.reads[inserted] = frozenset()
            self.writes[inserted] = frozenset()

        # Check if order actually changed (not just same cells in same positions)
        old_order = self.cell_order
        order_changed = old_order != list(new_order)

        # Update order
        self.cell_order = list(new_order)

        # Note: without provenance tracking, we cannot check Runnable here.
        # ORDER_CHANGED detection now relies on structural changes only.

        # Return newly stale cells
        currently_stale = set(self.get_stale_cells())
        return list(currently_stale - previously_stale)

    def clear(self) -> None:
        """Reset all state (e.g., on kernel restart)."""
        # Formal model state
        self.status.clear()
        self.reads.clear()
        self.writes.clear()
        self.cell_order.clear()
        # Per-cell tracking data
        self.tracking_data.clear()
        self.execution_seq.clear()
        self.structural_reads_values.clear()
        self.typed_changes.clear()
        self.fingerprints.clear()

    # =========================================================================
    # Debug/Inspection
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire state to dict for debugging."""
        return {
            "cell_order": self.cell_order,
            "status": {k: v.to_dict() for k, v in self.status.items()},
            "reads": {k: sorted(str(loc) for loc in v) for k, v in self.reads.items()},
            "writes": {k: sorted(str(loc) for loc in v) for k, v in self.writes.items()},
        }

    def __str__(self) -> str:
        lines = ["NotebookState:"]
        lines.append(f"  cell_order: {self.cell_order}")
        lines.append("  status:")
        for cell_id in self.cell_order:
            status = self.get_status(cell_id)
            lines.append(f"    {cell_id}: {status}")
        return "\n".join(lines)
