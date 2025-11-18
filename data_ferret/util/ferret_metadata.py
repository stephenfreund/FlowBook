from __future__ import annotations
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

import nbformat


class ProfileData(BaseModel):
    duration: float = Field(description="The duration of the profile in seconds.")
    profile: str = Field(description="The profile contents.")
    env: Dict[str, str] = Field(
        description="The global variables and their types before the cell was executed."
    )
    env_after: Dict[str, str] = Field(
        description="The global variables and their types after the cell was executed."
    )


class OptimizationStep(BaseModel):
    target_cell_id: str = Field(description="The ID of the cell to modify")
    function_name: Optional[str] = Field(
        default=None,
        description="The name of the top-level function to optimize. None if the optimization applies to top-level code in the cell.",
    )
    description: List[str] = Field(description="The optimizations to apply")


class OptimizationPotential(BaseModel):
    potential: int = Field(
        ge=0, le=5, description="Optimization potential rating (0-5)"
    )
    optimization_plan: List[OptimizationStep] = Field(
        default_factory=list,
        description="A list of concrete optimization steps. Empty if potential < 4.",
    )


class CodeSnippet(BaseModel):
    cell_id: str = Field(description="The ID of the cell")
    function_name: Optional[str] = Field(
        default=None,
        description="The name of the function. None if this is the entire cell.",
    )
    source: str = Field(description="The source code for the function or cell")
    optimizations_applied: Optional[List[str]] = Field(
        default=None,
        description="List of optimizations that were applied (for optimized code only)",
    )


class OptimizedCodeResponse(BaseModel):
    optimizations_applied: List[str] = Field(
        description="A bullet list of optimizations that were applied"
    )
    optimized_code: str = Field(
        description="The optimized Python code, ready to run with no additional text"
    )


class BatchOptimizedCodeResponse(BaseModel):
    """Response from LLM when optimizing multiple code snippets at once."""
    explanation: str = Field(
        description="Detailed explanation of the optimization strategy, including a correctness argument and reasoning about the changes made"
    )

    optimized_snippets: List[CodeSnippet] = Field(
        description="Complete list of optimized code snippets, one for each input snippet in the same order"
    )


class GeneratedCodeMetadata(BaseModel):
    """Metadata for AI-generated code from string specifications."""

    explanation: str = Field(
        description="Brief explanation of what the generated code does"
    )
    original_spec: str = Field(
        description="The original string specification that was used to generate the code"
    )


class OptimizedCodeMetadata(BaseModel):
    """Metadata for AI-optimized code."""

    original_code: str = Field(description="The original code before optimization")
    optimized_code: str = Field(
        description="The optimized code that replaced the original"
    )
    optimizations_applied: List[str] = Field(
        description="List of optimizations that were applied"
    )


class OptimizationAppliedMetadata(BaseModel):
    """Metadata for cells that triggered optimizations applied to other cells."""

    modified_cell_ids: List[str] = Field(
        description="List of cell IDs that were modified as a result of optimizing this cell"
    )


class FerretMetadata(BaseModel):
    optimization_potential: Optional[OptimizationPotential] = None
    profile: Optional[ProfileData] = None
    generated: Optional[GeneratedCodeMetadata] = None
    optimized: Optional[OptimizedCodeMetadata] = None
    optimization_applied: Optional[OptimizationAppliedMetadata] = None

    def get_optimization_potential(self) -> Optional[OptimizationPotential]:
        return self.optimization_potential

    def set_optimization_potential(
        self, metadata: OptimizationPotential
    ) -> FerretMetadata:
        return self.model_copy(update={"optimization_potential": metadata})

    def get_profile(self) -> Optional[ProfileData]:
        return self.profile

    def set_profile(self, metadata: ProfileData) -> FerretMetadata:
        return self.model_copy(update={"profile": metadata})

    def get_generated(self) -> Optional[GeneratedCodeMetadata]:
        return self.generated

    def set_generated(self, metadata: GeneratedCodeMetadata) -> FerretMetadata:
        return self.model_copy(update={"generated": metadata})

    def get_optimized(self) -> Optional[OptimizedCodeMetadata]:
        return self.optimized

    def set_optimized(self, metadata: OptimizedCodeMetadata) -> FerretMetadata:
        return self.model_copy(update={"optimized": metadata})

    def get_optimization_applied(self) -> Optional[OptimizationAppliedMetadata]:
        return self.optimization_applied

    def set_optimization_applied(
        self, metadata: OptimizationAppliedMetadata
    ) -> FerretMetadata:
        return self.model_copy(update={"optimization_applied": metadata})

    @staticmethod
    def from_cell(cell: nbformat.NotebookNode) -> FerretMetadata:
        if "metadata" not in cell or "ferret" not in cell["metadata"]:
            return FerretMetadata()
        else:
            return FerretMetadata.model_validate(cell["metadata"]["ferret"])


def set_optimization_potential_ferret_metadata(
    cell: nbformat.NotebookNode, metadata: OptimizationPotential
) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["optimization_potential"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret


def set_profile_ferret_metadata(
    cell: nbformat.NotebookNode, metadata: ProfileData
) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["profile"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret


def set_generated_ferret_metadata(
    cell: nbformat.NotebookNode, metadata: GeneratedCodeMetadata
) -> None:
    """Set the generated code metadata for a cell."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["generated"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret


def set_optimized_ferret_metadata(
    cell: nbformat.NotebookNode, metadata: OptimizedCodeMetadata
) -> None:
    """Set the optimized code metadata for a cell."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["optimized"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret


def set_optimization_applied_ferret_metadata(
    cell: nbformat.NotebookNode, metadata: OptimizationAppliedMetadata
) -> None:
    """Set the optimization applied metadata for a cell that triggered optimizations."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["optimization_applied"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret


def get_ferret_metadata_from_cell(cell: nbformat.NotebookNode) -> dict | None:
    if (
        "metadata" not in cell
        or "ferret" not in cell["metadata"]
        or cell["metadata"]["ferret"] is None
    ):
        return None
    ferret = cell["metadata"]["ferret"]
    if isinstance(ferret, dict):
        return ferret
    # fallback, it is a BaseModel
    return FerretMetadata.model_validate(ferret).model_dump()
