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
    description: List[str] = Field(
        description="The optimizations to apply"
    )


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
    source: str = Field(
        description="The source code for the function or cell"
    )
    optimizations_applied: Optional[List[str]] = Field(
        default=None,
        description="List of optimizations that were applied (for optimized code only)"
    )


class OptimizedCodeResponse(BaseModel):
    optimized_code: str = Field(
        description="The optimized Python code, ready to run with no additional text"
    )
    optimizations_applied: List[str] = Field(
        description="A bullet list of optimizations that were applied"
    )


class FerretMetadata(BaseModel):
    optimization_potential: Optional[OptimizationPotential] = None
    profile: Optional[ProfileData] = None

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
