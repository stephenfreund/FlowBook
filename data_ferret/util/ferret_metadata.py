from __future__ import annotations
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

import nbformat

class ProfileData(BaseModel):
    duration: float = Field(description="The duration of the profile in seconds.")
    profile: str = Field(description="The profile contents.")
    env: Dict[str, str] = Field(description="The global variables and their types.")

class OptimizationStep(BaseModel):
    target_cell_id: str = Field(description="The ID of the cell to modify")
    function_name: Optional[str] = Field(default=None, description="The name of the top-level function to optimize. None if the optimization applies to top-level code in the cell.")
    description: str = Field(description="Description of the optimization step to apply")

class OptimizationPotential(BaseModel):
    potential: int = Field(ge=0, le=5, description="Optimization potential rating (0-5)")
    optimization_plan: List[OptimizationStep] = Field(default_factory=list, description="A list of concrete optimization steps. Empty if potential < 4.")

class FerretMetadata(BaseModel):
    optimization_potential: Optional[OptimizationPotential] = None
    profile: Optional[ProfileData] = None

    def get_optimization_potential(self) -> Optional[OptimizationPotential]:
        return self.optimization_potential

    def set_optimization_potential(self, metadata: OptimizationPotential) -> FerretMetadata:
        return self.model_copy(update={"optimization_potential": metadata})

    def get_profile(self) -> Optional[ProfileData]:
        return self.profile

    def set_profile(self, metadata: ProfileData) -> FerretMetadata:
        return self.model_copy(update={"profile": metadata})

    @staticmethod
    def from_cell(cell: nbformat.NotebookNode) -> FerretMetadata:
        if "metadata" not in cell:
            return FerretMetadata()
        else:
            return FerretMetadata.model_validate(cell["metadata"]['ferret'])


def set_optimization_potential_ferret_metadata(cell: nbformat.NotebookNode, metadata: OptimizationPotential) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["optimization_potential"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret

def set_profile_ferret_metadata(cell: nbformat.NotebookNode, metadata: ProfileData) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["profile"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret

def get_ferret_metadata_from_cell(cell: nbformat.NotebookNode) -> dict | None:
    if "metadata" not in cell or "ferret" not in cell["metadata"] or cell["metadata"]["ferret"] is None:
        return None
    ferret = cell["metadata"]["ferret"]
    if isinstance(ferret, dict):
        return ferret
    # fallback, it is a BaseModel
    return FerretMetadata.model_validate(ferret).model_dump()