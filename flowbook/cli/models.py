"""Data models for compare_baseline and compare_overhead.

This module defines dataclasses for all memory measurement data structures,
ensuring type safety and clear interfaces between data collection and plotting.

Memory Data Format (v4):

{
  "version": "4.0",
  "metadata": {
    "staleness_mode": "syntactic" | "semantic",
    "num_cells": int,
    "timeout_seconds": float,
    "notebook_path": str,
    "timestamp": str
  },
  "kernels": {
    "baseline": {
      "memory": {
        "cells": [BaselineCellMemory as dict, ...],
        "rerun_cells": [...]
      }
    },
    "flowbook": {
      "memory": {
        "cells": [FlowBookCellMemory as dict, ...],
        "rerun_cells": [...]
      }
    }
  }
}
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class StalenessMode(str, Enum):
    """Staleness tracking mode."""
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"


# ============ Baseline Memory Models ============

@dataclass
class BaselineMemorySnapshot:
    """Memory snapshot at a point in time (pre or post cell).

    Attributes:
        user_ns_mb: Size of user namespace objects (HeapSizer.sizeof_user_namespace)
        gpu_mb: GPU memory usage
    """
    user_ns_mb: float
    gpu_mb: float

    @property
    def total_mb(self) -> float:
        """Total memory: user namespace + GPU."""
        return self.user_ns_mb + self.gpu_mb

    def to_dict(self) -> Dict[str, float]:
        return {"user_ns_mb": self.user_ns_mb, "gpu_mb": self.gpu_mb}

    @classmethod
    def from_dict(cls, d: Dict) -> "BaselineMemorySnapshot":
        return cls(user_ns_mb=d.get("user_ns_mb", 0.0), gpu_mb=d.get("gpu_mb", 0.0))


@dataclass
class BaselineCellMemory:
    """Memory measurements for a single baseline cell.

    Attributes:
        cell_id: Unique cell identifier
        cell_index: Position in notebook (0-indexed)
        pre: Memory snapshot before cell execution
        post: Memory snapshot after cell execution
        status: "ok" or "error"
        error: Error message if status is "error"
    """
    cell_id: str
    cell_index: int
    pre: BaselineMemorySnapshot
    post: BaselineMemorySnapshot
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_index": self.cell_index,
            "pre_user_ns_mb": self.pre.user_ns_mb,
            "pre_gpu_mb": self.pre.gpu_mb,
            "post_user_ns_mb": self.post.user_ns_mb,
            "post_gpu_mb": self.post.gpu_mb,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BaselineCellMemory":
        return cls(
            cell_id=d["cell_id"],
            cell_index=d["cell_index"],
            pre=BaselineMemorySnapshot(
                user_ns_mb=d.get("pre_user_ns_mb", 0.0),
                gpu_mb=d.get("pre_gpu_mb", 0.0),
            ),
            post=BaselineMemorySnapshot(
                user_ns_mb=d.get("post_user_ns_mb", 0.0),
                gpu_mb=d.get("post_gpu_mb", 0.0),
            ),
            status=d.get("status", "ok"),
            error=d.get("error"),
        )


@dataclass
class BaselineMemoryResult:
    """All baseline memory measurements.

    Attributes:
        cells: Memory measurements for initial cell executions
        rerun_cells: Memory measurements for rerun cells
    """
    cells: List[BaselineCellMemory] = field(default_factory=list)
    rerun_cells: List[BaselineCellMemory] = field(default_factory=list)

    @property
    def all_cells(self) -> List[BaselineCellMemory]:
        """All cells including reruns."""
        return self.cells + self.rerun_cells

    @property
    def final_user_ns_mb(self) -> float:
        """User namespace size after last cell."""
        all_c = self.all_cells
        return all_c[-1].post.user_ns_mb if all_c else 0.0

    @property
    def max_user_ns_mb(self) -> float:
        """Peak user namespace size."""
        all_c = self.all_cells
        return max((c.post.user_ns_mb for c in all_c), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "rerun_cells": [c.to_dict() for c in self.rerun_cells],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BaselineMemoryResult":
        return cls(
            cells=[BaselineCellMemory.from_dict(c) for c in d.get("cells", [])],
            rerun_cells=[BaselineCellMemory.from_dict(c) for c in d.get("rerun_cells", [])],
        )


# ============ FlowBook Memory Models ============

@dataclass
class CheckpointVarInfo:
    """Information about a single variable in a checkpoint.

    Attributes:
        size_mb: Size of the variable in MB
        type_name: Python type name (e.g., "DataFrame")
        module: Module of the type (e.g., "pandas.core.frame")
    """
    size_mb: float
    type_name: str = ""
    module: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"size_mb": self.size_mb, "type_name": self.type_name, "module": self.module}

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointVarInfo":
        if isinstance(d, (int, float)):
            # Simple format: just the size
            return cls(size_mb=float(d))
        return cls(
            size_mb=d.get("size_mb", 0.0),
            type_name=d.get("type_name", ""),
            module=d.get("module", ""),
        )


@dataclass
class CheckpointVarSizes:
    """Per-variable sizes within a single checkpoint.

    Attributes:
        vars: Mapping of variable name to size info
    """
    vars: Dict[str, CheckpointVarInfo] = field(default_factory=dict)

    @property
    def total_mb(self) -> float:
        """Total size of all variables in this checkpoint."""
        return sum(v.size_mb for v in self.vars.values())

    def __getitem__(self, var_name: str) -> float:
        """Get size of a variable (0 if not present)."""
        return self.vars.get(var_name, CheckpointVarInfo(0.0)).size_mb

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.vars.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointVarSizes":
        return cls(vars={k: CheckpointVarInfo.from_dict(v) for k, v in d.items()})


@dataclass
class FlowBookMemorySnapshot:
    """Memory snapshot at a point in time (pre or post cell).

    Attributes:
        user_ns_mb: Size of user namespace objects
        gpu_mb: GPU memory usage
        overhead_mb: Total FlowBook overhead (checkpoints + caches)
        checkpoint_vars: Per-checkpoint, per-variable breakdown
    """
    user_ns_mb: float
    gpu_mb: float
    overhead_mb: float
    checkpoint_vars: Dict[str, CheckpointVarSizes] = field(default_factory=dict)

    @property
    def total_mb(self) -> float:
        """Total memory: user namespace + GPU + overhead."""
        return self.user_ns_mb + self.gpu_mb + self.overhead_mb

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints."""
        return len(self.checkpoint_vars)

    @property
    def total_checkpoint_mb(self) -> float:
        """Sum of all checkpoint var sizes."""
        return sum(cv.total_mb for cv in self.checkpoint_vars.values())

    def var_totals(self) -> Dict[str, float]:
        """Sum each variable's size across all checkpoints.

        Returns:
            {var_name: total_mb_across_all_checkpoints}
        """
        totals: Dict[str, float] = {}
        for ckpt in self.checkpoint_vars.values():
            for var_name, info in ckpt.vars.items():
                totals[var_name] = totals.get(var_name, 0.0) + info.size_mb
        return totals

    def var_types(self) -> Dict[str, str]:
        """Get type names for all variables.

        Returns:
            {var_name: type_name}
        """
        types: Dict[str, str] = {}
        for ckpt in self.checkpoint_vars.values():
            for var_name, info in ckpt.vars.items():
                if var_name not in types and info.type_name:
                    types[var_name] = info.type_name
        return types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_ns_mb": self.user_ns_mb,
            "gpu_mb": self.gpu_mb,
            "overhead_mb": self.overhead_mb,
            "checkpoint_vars": {k: v.to_dict() for k, v in self.checkpoint_vars.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FlowBookMemorySnapshot":
        return cls(
            user_ns_mb=d.get("user_ns_mb", 0.0),
            gpu_mb=d.get("gpu_mb", 0.0),
            overhead_mb=d.get("overhead_mb", 0.0),
            checkpoint_vars={
                k: CheckpointVarSizes.from_dict(v)
                for k, v in d.get("checkpoint_vars", {}).items()
            },
        )


@dataclass
class FlowBookCellMemory:
    """Memory measurements for a single FlowBook cell.

    Attributes:
        cell_id: Unique cell identifier
        cell_index: Position in notebook (0-indexed)
        pre: Memory snapshot before cell execution
        post: Memory snapshot after cell execution
        status: "ok" or "error"
        error: Error message if status is "error"
    """
    cell_id: str
    cell_index: int
    pre: FlowBookMemorySnapshot
    post: FlowBookMemorySnapshot
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_index": self.cell_index,
            "pre_user_ns_mb": self.pre.user_ns_mb,
            "pre_gpu_mb": self.pre.gpu_mb,
            "pre_overhead_mb": self.pre.overhead_mb,
            "pre_checkpoint_vars": {k: v.to_dict() for k, v in self.pre.checkpoint_vars.items()},
            "post_user_ns_mb": self.post.user_ns_mb,
            "post_gpu_mb": self.post.gpu_mb,
            "post_overhead_mb": self.post.overhead_mb,
            "post_checkpoint_vars": {k: v.to_dict() for k, v in self.post.checkpoint_vars.items()},
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FlowBookCellMemory":
        return cls(
            cell_id=d["cell_id"],
            cell_index=d["cell_index"],
            pre=FlowBookMemorySnapshot(
                user_ns_mb=d.get("pre_user_ns_mb", 0.0),
                gpu_mb=d.get("pre_gpu_mb", 0.0),
                overhead_mb=d.get("pre_overhead_mb", 0.0),
                checkpoint_vars={
                    k: CheckpointVarSizes.from_dict(v)
                    for k, v in d.get("pre_checkpoint_vars", {}).items()
                },
            ),
            post=FlowBookMemorySnapshot(
                user_ns_mb=d.get("post_user_ns_mb", 0.0),
                gpu_mb=d.get("post_gpu_mb", 0.0),
                overhead_mb=d.get("post_overhead_mb", 0.0),
                checkpoint_vars={
                    k: CheckpointVarSizes.from_dict(v)
                    for k, v in d.get("post_checkpoint_vars", {}).items()
                },
            ),
            status=d.get("status", "ok"),
            error=d.get("error"),
        )


@dataclass
class FlowBookMemoryResult:
    """All FlowBook memory measurements.

    Attributes:
        cells: Memory measurements for initial cell executions
        rerun_cells: Memory measurements for rerun cells
    """
    cells: List[FlowBookCellMemory] = field(default_factory=list)
    rerun_cells: List[FlowBookCellMemory] = field(default_factory=list)

    @property
    def all_cells(self) -> List[FlowBookCellMemory]:
        """All cells including reruns."""
        return self.cells + self.rerun_cells

    @property
    def final_overhead_mb(self) -> float:
        """Overhead after last cell."""
        all_c = self.all_cells
        return all_c[-1].post.overhead_mb if all_c else 0.0

    @property
    def final_user_ns_mb(self) -> float:
        """User namespace size after last cell."""
        all_c = self.all_cells
        return all_c[-1].post.user_ns_mb if all_c else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "rerun_cells": [c.to_dict() for c in self.rerun_cells],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FlowBookMemoryResult":
        return cls(
            cells=[FlowBookCellMemory.from_dict(c) for c in d.get("cells", [])],
            rerun_cells=[FlowBookCellMemory.from_dict(c) for c in d.get("rerun_cells", [])],
        )


# ============ V5 Simplified Memory Models ============

@dataclass
class V5CellMemory:
    """Simplified cell memory for v5 format.

    V5 simplifies memory measurement by:
    - Removing pre_* fields (only post-execution state needed)
    - Flattening nested checkpoint structure to single checkpoint_mb
    - Aggregating per-variable costs across all checkpoints

    Attributes:
        cell_id: Unique cell identifier
        cell_index: Position in notebook (0-indexed)
        user_ns_mb: Size of user namespace objects
        gpu_mb: GPU memory usage
        checkpoint_mb: Total checkpoint overhead BEYOND namespace
        checkpoint_vars: Per-variable checkpoint sizes aggregated across all checkpoints
        checkpoint_var_timing: Per-variable deepcopy times in milliseconds
    """
    cell_id: str
    cell_index: int
    user_ns_mb: float
    gpu_mb: float
    checkpoint_mb: float
    checkpoint_vars: Dict[str, float] = field(default_factory=dict)
    checkpoint_var_timing: Dict[str, float] = field(default_factory=dict)

    @property
    def base_mb(self) -> float:
        """Base memory: user namespace + GPU."""
        return self.user_ns_mb + self.gpu_mb

    @property
    def total_mb(self) -> float:
        """Total memory: base + checkpoint overhead."""
        return self.base_mb + self.checkpoint_mb

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "cell_id": self.cell_id,
            "cell_index": self.cell_index,
            "user_ns_mb": self.user_ns_mb,
            "gpu_mb": self.gpu_mb,
            "checkpoint_mb": self.checkpoint_mb,
            "checkpoint_vars": self.checkpoint_vars,
        }
        # Only include timing if present (for backward compatibility)
        if self.checkpoint_var_timing:
            d["checkpoint_var_timing"] = self.checkpoint_var_timing
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "V5CellMemory":
        return cls(
            cell_id=d["cell_id"],
            cell_index=d["cell_index"],
            user_ns_mb=d.get("user_ns_mb", 0.0),
            gpu_mb=d.get("gpu_mb", 0.0),
            checkpoint_mb=d.get("checkpoint_mb", 0.0),
            checkpoint_vars=d.get("checkpoint_vars", {}),
            checkpoint_var_timing=d.get("checkpoint_var_timing", {}),
        )


@dataclass
class V5MemoryResult:
    """V5 simplified memory measurements for a kernel.

    Attributes:
        cells: Memory measurements for initial cell executions
        rerun_cells: Memory measurements for rerun cells
    """
    cells: List[V5CellMemory] = field(default_factory=list)
    rerun_cells: List[V5CellMemory] = field(default_factory=list)

    @property
    def all_cells(self) -> List[V5CellMemory]:
        """All cells including reruns."""
        return self.cells + self.rerun_cells

    @property
    def final_checkpoint_mb(self) -> float:
        """Checkpoint overhead after last cell."""
        all_c = self.all_cells
        return all_c[-1].checkpoint_mb if all_c else 0.0

    @property
    def final_user_ns_mb(self) -> float:
        """User namespace size after last cell."""
        all_c = self.all_cells
        return all_c[-1].user_ns_mb if all_c else 0.0

    @property
    def peak_checkpoint_mb(self) -> float:
        """Peak checkpoint overhead across all cells."""
        all_c = self.all_cells
        return max((c.checkpoint_mb for c in all_c), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "rerun_cells": [c.to_dict() for c in self.rerun_cells],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "V5MemoryResult":
        return cls(
            cells=[V5CellMemory.from_dict(c) for c in d.get("cells", [])],
            rerun_cells=[V5CellMemory.from_dict(c) for c in d.get("rerun_cells", [])],
        )


# ============ Plot Data Models ============

@dataclass
class Plot1Data:
    """Data for Plot 1: Execution Time per Cell.

    Shows stacked bar chart of run time + overhead components.
    """
    cells: List[int]  # 1-indexed cell numbers
    run_time_sec: List[float]
    state_time_sec: List[float]
    check_time_sec: List[float]
    other_time_sec: List[float]
    initial_count: int  # cells before reruns


@dataclass
class Plot2Data:
    """Data for Plot 2: Checkpoint Time by Variable.

    Shows stacked area chart of deepcopy time per variable.
    """
    cells: List[int]
    var_series: Dict[str, List[float]]  # {var_name: [sec_at_cell_0, ...]}
    vars_ordered: List[str]  # ordered by total time
    initial_count: int


@dataclass
class Plot3Data:
    """Data for Plot 3: Memory Overhead.

    Shows stacked area chart:
    - Bottom: user namespace (gray)
    - Middle: GPU memory (orange)
    - Top: FlowBook checkpoint overhead (blue)
    """
    cells: List[int]
    user_ns_mb: List[float]  # user namespace size
    gpu_mb: List[float]  # GPU memory
    overhead_mb: List[float]  # flowbook checkpoint overhead
    has_baseline: bool
    peak_overhead_mb: float
    peak_overhead_pct: float
    peak_cell: int
    initial_count: int

    @property
    def base_mb(self) -> List[float]:
        """Base memory (namespace + GPU) for backward compatibility."""
        return [ns + gpu for ns, gpu in zip(self.user_ns_mb, self.gpu_mb)]


@dataclass
class Plot4Data:
    """Data for Plot 4: Checkpoint Memory by Variable.

    Shows stacked area chart:
    - Bottom: user namespace (gray)
    - Middle: GPU memory (orange)
    - Top: per-variable checkpoint sizes (colors)
    """
    cells: List[int]
    namespace_mb: List[float]
    gpu_mb: List[float]
    var_series: Dict[str, List[float]]  # {var_name: [mb_at_cell_0, ...]}
    vars_ordered: List[str]  # ordered by max size
    var_types: Dict[str, str]  # {var_name: type_name}
    initial_count: int


@dataclass
class Plot5Data:
    """Data for Plot 5: Overhead Time per Cell.

    Shows stacked bar chart of state/check/other overhead.
    """
    cells: List[int]
    state_sec: List[float]
    check_sec: List[float]
    other_sec: List[float]
    initial_count: int


@dataclass
class Plot6Data:
    """Data for Plot 6: Checkpoint Overhead Ratio per Cell.

    Bar chart of checkpoint_delta_mb / base_mb for each cell.
    """
    cells: List[int]  # 1-indexed cell numbers
    ratios: List[float]  # ratio per cell (0 if base < threshold)
    initial_count: int  # cells before reruns


@dataclass
class CDFData:
    """Data for aggregate CDF plots across multiple notebooks."""
    # Time overhead ratio CDF
    time_ratios: List[float]
    time_sorted: List[float]
    time_percentiles: List[float]

    # Memory overhead ratio CDF
    memory_ratios: List[float]
    memory_sorted: List[float]
    memory_percentiles: List[float]

    # Peak memory overhead CDF (per notebook)
    peak_memory_pct: List[float]
    peak_sorted: List[float]
    peak_percentiles: List[float]


# ============ Comparison Result Model ============

@dataclass
class ComparisonMetadata:
    """Metadata for a comparison run."""
    staleness_mode: str  # "syntactic" or "semantic"
    num_cells: int
    timeout_seconds: float
    notebook_path: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "staleness_mode": self.staleness_mode,
            "num_cells": self.num_cells,
            "timeout_seconds": self.timeout_seconds,
            "notebook_path": self.notebook_path,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ComparisonMetadata":
        return cls(
            staleness_mode=d.get("staleness_mode", "semantic"),
            num_cells=d.get("num_cells", 0),
            timeout_seconds=d.get("timeout_seconds", 0.0),
            notebook_path=d.get("notebook_path", ""),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class ComparisonResult:
    """Complete comparison result for a notebook.

    This is the top-level structure serialized to JSON.
    """
    version: str = "4.0"
    metadata: Optional[ComparisonMetadata] = None
    baseline: Optional[BaselineMemoryResult] = None
    flowbook: Optional[FlowBookMemoryResult] = None
    # Timing data kept as dict for now (not refactored)
    timing: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"version": self.version}

        if self.metadata:
            result["metadata"] = self.metadata.to_dict()

        kernels: Dict[str, Any] = {}
        if self.baseline:
            kernels["baseline"] = {"memory": self.baseline.to_dict()}
        if self.flowbook:
            kernels["flowbook"] = {"memory": self.flowbook.to_dict()}
        if kernels:
            result["kernels"] = kernels

        if self.timing:
            result["timing"] = self.timing

        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "ComparisonResult":
        metadata = None
        if "metadata" in d:
            metadata = ComparisonMetadata.from_dict(d["metadata"])

        baseline = None
        flowbook = None
        kernels = d.get("kernels", {})

        if "baseline" in kernels and "memory" in kernels["baseline"]:
            baseline = BaselineMemoryResult.from_dict(kernels["baseline"]["memory"])

        if "flowbook" in kernels and "memory" in kernels["flowbook"]:
            flowbook = FlowBookMemoryResult.from_dict(kernels["flowbook"]["memory"])

        # Timing data is inside kernels, not at root level
        # Reconstruct the timing dict structure expected by plot_extraction
        timing = None
        if kernels:
            timing = {"kernels": kernels}

        return cls(
            version=d.get("version", "4.0"),
            metadata=metadata,
            baseline=baseline,
            flowbook=flowbook,
            timing=timing,
        )
