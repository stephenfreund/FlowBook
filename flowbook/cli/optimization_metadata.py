"""
Pydantic models for optimization metadata stored in notebooks.

These models define the structure of optimization statistics that are embedded
in the notebook metadata and displayed by the stats CLI tool.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SplitResultsSummary(BaseModel):
    """Summary of split preprocessing results."""

    cells_analyzed: int = Field(description="Number of cells analyzed for splitting")
    cells_split: int = Field(description="Number of cells that were split")
    total_new_cells: int = Field(description="Total number of new cells created")
    llm_cost: float = Field(description="LLM cost for split operation")
    time: float = Field(description="Time taken for split operation in seconds")


class CellOptimizationResult(BaseModel):
    """Optimization result for a single cell."""

    cell_id: str = Field(description="Cell ID")
    potential: Optional[int] = Field(
        description="Optimization potential score (1-5)", default=None
    )
    initial_time: float = Field(description="Initial execution time in seconds")
    final_time: float = Field(description="Final execution time in seconds")
    speedup: float = Field(description="Speedup factor (initial/final)")
    status: str = Field(
        description="Optimization status: 'optimized', 'error', 'no improvement', 'not attempted'"
    )


class OptimizationResultsSummary(BaseModel):
    """Summary of optimization results for all cells."""

    cells: List[CellOptimizationResult] = Field(
        description="List of per-cell optimization results"
    )
    total_initial_time: float = Field(
        description="Total initial execution time across all cells"
    )
    total_final_time: float = Field(
        description="Total final execution time across all cells"
    )
    overall_speedup: float = Field(description="Overall speedup factor")
    time_saved: float = Field(description="Total time saved in seconds")
    time_saved_percent: float = Field(description="Percentage of time saved")


class LLMCostSummary(BaseModel):
    """Summary of LLM costs for the optimization pipeline."""

    split_cost: Optional[float] = Field(
        description="Cost for split preprocessing", default=None
    )
    split_time: Optional[float] = Field(
        description="Time for split preprocessing in seconds", default=None
    )
    optimization_cost: float = Field(description="Cost for optimization pipeline")
    optimization_time: float = Field(
        description="Time for optimization pipeline in seconds"
    )
    total_cost: float = Field(description="Total cost (split + optimization)")
    total_time: float = Field(description="Total time in seconds")


class FlowbookOptimizationMetadata(BaseModel):
    """Top-level metadata for flowbook optimization stored in notebook."""

    split_results: Optional[SplitResultsSummary] = Field(
        description="Split preprocessing results", default=None
    )
    optimization_results: OptimizationResultsSummary = Field(
        description="Optimization results"
    )
    llm_costs: LLMCostSummary = Field(description="LLM cost summary")
    timestamp: str = Field(description="ISO format timestamp of optimization")
    model: str = Field(description="Primary model used for optimization")
    fast_model: str = Field(description="Fast model used for lightweight operations")

    @classmethod
    def create_timestamp(cls) -> str:
        """Create ISO format timestamp for current time."""
        return datetime.utcnow().isoformat() + "Z"
