from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

import nbformat

class ProfileMetadata(BaseModel):
    start_time: float = Field(description="The start time of the profile.")
    end_time: float = Field(description="The end time of the profile.")
    duration: float = Field(description="The duration of the profile in seconds.")
    profile: str = Field(description="The profile contents.")

class InspectMetadata(BaseModel):
    optimizability: int = Field(ge=0, le=5, description="How optimizable is this cell? (0-5)")
    readability: int = Field(ge=0, le=5, description="How readable is this cell? (0-5)")
    complexity: int = Field(ge=0, le=5, description="How complex is this cell? (0-5)")

    improvements: List[str] = Field(default_factory=list, description="A list of concrete suggestions for how to improve this cell.")

class FerretMetadata(BaseModel):
    inspect: Optional[InspectMetadata] = None
    profile: Optional[ProfileMetadata] = None

    def get_inspect_metadata(self) -> Optional[InspectMetadata]:
        return self.inspect

    def set_inspect_metadata(self, metadata: InspectMetadata) -> FerretMetadata:  
        return self.model_copy(update={"inspect": metadata})

    def get_profile_metadata(self) -> Optional[ProfileMetadata]:
        return self.profile

    def set_profile_metadata(self, metadata: ProfileMetadata) -> FerretMetadata:
        return self.model_copy(update={"profile": metadata})

    @staticmethod
    def from_cell(cell: nbformat.NotebookNode) -> FerretMetadata:
        if "metadata" not in cell:
            return FerretMetadata()
        else:
            return FerretMetadata.model_validate(cell["metadata"])


def set_inspect_ferret_metadata(cell: nbformat.NotebookNode, metadata: InspectMetadata) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    ferret = cell["metadata"].get("ferret", {})
    if not isinstance(ferret, dict):
        ferret = FerretMetadata.model_validate(ferret).model_dump()
    ferret["inspect"] = metadata.model_dump()
    cell["metadata"]["ferret"] = ferret

def set_profile_ferret_metadata(cell: nbformat.NotebookNode, metadata: ProfileMetadata) -> None:
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