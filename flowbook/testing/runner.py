"""
Simulated reproducibility execution engine.

Provides ReproducibilitySimulator class that executes notebook cells with full reproducibility
tracking (checkpoints, read/write tracking, enforcement) without requiring
a running Jupyter kernel.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time

import numpy as np

from flowbook.kernel_support.checkpoint import Checkpoint, Checkpoints
from flowbook.util.output import log, timer
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.tracking import TrackingDict
from flowbook.kernel.reproducibility_enforcer import (
    ReproducibilityEnforcer,
    PRE_CHECKPOINT_PREFIX,
    POST_CHECKPOINT_PREFIX,
)
from flowbook.kernel.models import ReproducibilityResult

from flowbook.testing.notebook_loader import Cell


@dataclass
class CellRecord:
    """Record of a cell execution in the simulator."""

    cell_id: str
    source: str
    tracking: TrackingData
    sdc_result: ReproducibilityResult
    execution_time_ms: float
    checkpoint_time_ms: float
    check_time_ms: float
    error: Optional[str] = None


@dataclass
class ExecutionLog:
    """Log entry for detailed execution tracking."""

    event: str
    cell_id: str
    timestamp: float
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class ReproducibilitySimulator:
    """
    Simulates reproducibility kernel execution for testing.

    This class executes notebook cells using Python's exec() while maintaining
    full reproducibility infrastructure: checkpoints, read/write tracking, and enforcement.
    It mimics the behavior of the real reproducibility kernel without requiring Jupyter.

    Usage:
        simulator = ReproducibilitySimulator()
        cells = load_notebook("notebook.ipynb")
        simulator.execute_notebook(cells)

        # Access checkpoints and records
        pre = simulator.get_pre_checkpoint("cell_a")
        post = simulator.get_post_checkpoint("cell_a")
        record = simulator.cell_records["cell_a"]
    """

    def __init__(self):
        """Initialize the reproducibility simulator."""
        self.checkpoints = Checkpoints(sanity_check=False, warn_classes=False)
        self.enforcer = ReproducibilityEnforcer(self.checkpoints)
        self.namespace: Dict[str, Any] = {}
        self.cell_records: Dict[str, CellRecord] = {}
        self.cells: List[Cell] = []
        self.execution_log: List[ExecutionLog] = []
        self._tracking_dict: Optional[TrackingDict] = None

    def execute_notebook(self, cells: List[Cell]) -> None:
        """
        Execute all cells in order, creating checkpoints for each.

        Args:
            cells: List of Cell objects to execute
        """
        self.cells = cells
        self.namespace = {}
        self.cell_records = {}
        self.execution_log = []

        # Set cell order for reproducibility enforcement
        cell_order = [c.cell_id for c in cells]
        self.enforcer.set_cell_order(cell_order)

        # Create tracking dict wrapping namespace
        self._tracking_dict = TrackingDict(self.namespace)

        # Initialize matplotlib inline mode (equivalent to %matplotlib inline)
        self._init_matplotlib()

        for cell in cells:
            self.execute_cell(cell)

    def _init_matplotlib(self) -> None:
        """Initialize matplotlib with inline/agg backend for non-interactive use."""
        init_code = """
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()
"""
        try:
            exec(init_code, self.namespace)
        except Exception:
            pass  # matplotlib may not be installed

    def _set_random_seeds(self, cell_index: int) -> None:
        """
        Set random seeds for stdlib and numpy based on cell index.

        This ensures deterministic execution for cells that use random functions,
        making re-execution produce the same results.

        Args:
            cell_index: The index of the cell in the notebook
        """
        random.seed(cell_index)
        np.random.seed(cell_index)

    def execute_cell(self, cell: Cell) -> CellRecord:
        """
        Execute a single cell with full reproducibility tracking.

        Args:
            cell: Cell to execute

        Returns:
            CellRecord with execution details
        """
        if self._tracking_dict is None:
            self._tracking_dict = TrackingDict(self.namespace)

        with timer(key="sim:execute_cell", message=f"Execute cell {cell.cell_id}") as t_total:
            self._log("execute_start", cell.cell_id)

            # 1. Save pre-checkpoint
            with timer(key="sim:pre_checkpoint") as t_pre:
                pre_name = f"{PRE_CHECKPOINT_PREFIX}{cell.cell_id}"
                self.checkpoints.save(pre_name, self.namespace, max_size_mb=None)
                pre_checkpoint = self.checkpoints.saved[pre_name]

            # 2. Set random seeds based on cell index for deterministic execution
            self._set_random_seeds(cell.index)

            # 3. Execute code with tracking
            with timer(key="sim:exec") as t_exec:
                error = None
                with self._tracking_dict.track_execution():
                    try:
                        exec(cell.source, self._tracking_dict)
                    except Exception as e:
                        error = str(e)

            exec_time = t_exec.duration()

            # 4. Get tracking data
            tracking = self._tracking_dict.get_tracking_data()

            # 5. Save post-checkpoint
            with timer(key="sim:post_checkpoint") as t_post:
                post_name = f"{POST_CHECKPOINT_PREFIX}{cell.cell_id}"
                self.checkpoints.save(post_name, self.namespace, max_size_mb=None)
                post_checkpoint = self.checkpoints.saved[post_name]

            checkpoint_time = t_pre.duration() + t_post.duration()

            # 6. Run SDC check
            with timer(key="sim:sdc_check") as t_check:
                sdc_result = self.enforcer.check(
                    cell_id=cell.cell_id,
                    pre_checkpoint=pre_checkpoint,
                    post_checkpoint=post_checkpoint,
                    tracking=tracking,
                    continue_on_violation=True,  # Continue to compute staleness
                    namespace=self.namespace,
                )
            check_time = t_check.duration()

        total_time = t_total.duration()

        # Create record
        record = CellRecord(
            cell_id=cell.cell_id,
            source=cell.source,
            tracking=tracking,
            sdc_result=sdc_result,
            execution_time_ms=exec_time,
            checkpoint_time_ms=checkpoint_time,
            check_time_ms=check_time,
            error=error,
        )
        self.cell_records[cell.cell_id] = record

        self._log(
            "execute_complete",
            cell.cell_id,
            duration_ms=total_time,
            details={
                "exec_ms": exec_time,
                "checkpoint_ms": checkpoint_time,
                "check_ms": check_time,
                "reads": list(tracking.reads_before_writes),
                "writes": list(tracking.writes),
                "changed": sdc_result.changed_variables,
                "stale": sdc_result.stale_cells,
                "violation": sdc_result.has_errors(),
                "error": error,
            },
        )

        status = "ERROR" if error else ("VIOLATION" if sdc_result.has_errors() else "OK")
        log(f"[{status}] {cell.cell_id}: {total_time:.2f}ms")

        return record

    def restore_pre_checkpoint(self, cell_id: str) -> None:
        """
        Restore namespace to the pre-checkpoint state for a cell.

        Args:
            cell_id: ID of the cell whose pre-checkpoint to restore
        """
        pre_name = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
        if pre_name not in self.checkpoints.saved:
            raise ValueError(f"No pre-checkpoint for cell {cell_id}")

        self._log("restore_pre", cell_id)
        self.checkpoints.restore(pre_name, self.namespace)

        # Recreate tracking dict with restored namespace
        self._tracking_dict = TrackingDict(self.namespace)

    def restore_post_checkpoint(self, cell_id: str) -> None:
        """
        Restore namespace to the post-checkpoint state for a cell.

        Args:
            cell_id: ID of the cell whose post-checkpoint to restore
        """
        post_name = f"{POST_CHECKPOINT_PREFIX}{cell_id}"
        if post_name not in self.checkpoints.saved:
            raise ValueError(f"No post-checkpoint for cell {cell_id}")

        self._log("restore_post", cell_id)
        self.checkpoints.restore(post_name, self.namespace)

        # Recreate tracking dict with restored namespace
        self._tracking_dict = TrackingDict(self.namespace)

    def get_pre_checkpoint(self, cell_id: str) -> Checkpoint:
        """Get the pre-checkpoint for a cell."""
        pre_name = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
        if pre_name not in self.checkpoints.saved:
            raise ValueError(f"No pre-checkpoint for cell {cell_id}")
        return self.checkpoints.saved[pre_name]

    def get_post_checkpoint(self, cell_id: str) -> Checkpoint:
        """Get the post-checkpoint for a cell."""
        post_name = f"{POST_CHECKPOINT_PREFIX}{cell_id}"
        if post_name not in self.checkpoints.saved:
            raise ValueError(f"No post-checkpoint for cell {cell_id}")
        return self.checkpoints.saved[post_name]

    def get_cell_ids(self) -> List[str]:
        """Get list of executed cell IDs in order."""
        return [c.cell_id for c in self.cells]

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        """Get a cell by ID."""
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def get_current_namespace(self) -> Dict[str, Any]:
        """Get a copy of the current namespace."""
        return dict(self.namespace)

    def _log(
        self,
        event: str,
        cell_id: str,
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an entry to the execution log."""
        self.execution_log.append(
            ExecutionLog(
                event=event,
                cell_id=cell_id,
                timestamp=time.perf_counter(),
                duration_ms=duration_ms,
                details=details or {},
            )
        )

    def re_execute_cell(self, cell_id: str) -> CellRecord:
        """
        Re-execute a cell after restoring its pre-checkpoint.

        This is a convenience method that:
        1. Restores the pre-checkpoint for the cell
        2. Executes the cell code
        3. Returns the new execution record

        Args:
            cell_id: ID of the cell to re-execute

        Returns:
            New CellRecord from re-execution
        """
        cell = self.get_cell(cell_id)
        if cell is None:
            raise ValueError(f"Cell not found: {cell_id}")

        self.restore_pre_checkpoint(cell_id)
        return self.execute_cell(cell)
