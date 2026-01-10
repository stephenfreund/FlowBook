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


class DynamicDependencies(BaseModel):
    reads_before_writes: List[str] = Field(
        default_factory=list,
        description="Variables read before being written in this cell"
    )
    writes: List[str] = Field(
        default_factory=list,
        description="Variables written in this cell"
    )
    column_reads_before_writes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="For DataFrame variables, which columns were read before being written. "
                    "Keys are variable paths (e.g., 'df', 'data[\"train\"]'), values are lists of column names."
    )
    column_writes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="For DataFrame variables, which columns were written. "
                    "Keys are variable paths, values are lists of column names."
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


class UnitTest(BaseModel):
    """A single unit test for a cell."""

    title: str = Field(description="Short title for the test")
    description: str = Field(description="English description of what this test validates")
    setup_code: str = Field(
        description="Python code to set up the globals used by the cell under test"
    )
    assertion_code: str = Field(
        description="Python code to assert the appropriate properties to ensure the test worked correctly"
    )


class UnitTests(BaseModel):
    """Collection of unit tests for a cell."""

    tests: List[UnitTest] = Field(
        default_factory=list, description="List of unit tests for the cell"
    )


class FlowbookMetadata(BaseModel):
    optimization_potential: Optional[OptimizationPotential] = None
    profile: Optional[ProfileData] = None
    dynamic_dependencies: Optional[DynamicDependencies] = None
    generated: Optional[GeneratedCodeMetadata] = None
    optimized: Optional[OptimizedCodeMetadata] = None
    optimization_applied: Optional[OptimizationAppliedMetadata] = None
    unit_tests: Optional[UnitTests] = None

    def get_optimization_potential(self) -> Optional[OptimizationPotential]:
        return self.optimization_potential

    def set_optimization_potential(
        self, metadata: OptimizationPotential
    ) -> FlowbookMetadata:
        return self.model_copy(update={"optimization_potential": metadata})

    def get_profile(self) -> Optional[ProfileData]:
        return self.profile

    def set_profile(self, metadata: ProfileData) -> FlowbookMetadata:
        return self.model_copy(update={"profile": metadata})

    def get_dynamic_dependencies(self) -> Optional[DynamicDependencies]:
        return self.dynamic_dependencies

    def set_dynamic_dependencies(self, metadata: DynamicDependencies) -> FlowbookMetadata:
        return self.model_copy(update={"dynamic_dependencies": metadata})

    def get_generated(self) -> Optional[GeneratedCodeMetadata]:
        return self.generated

    def set_generated(self, metadata: GeneratedCodeMetadata) -> FlowbookMetadata:
        return self.model_copy(update={"generated": metadata})

    def get_optimized(self) -> Optional[OptimizedCodeMetadata]:
        return self.optimized

    def set_optimized(self, metadata: OptimizedCodeMetadata) -> FlowbookMetadata:
        return self.model_copy(update={"optimized": metadata})

    def get_optimization_applied(self) -> Optional[OptimizationAppliedMetadata]:
        return self.optimization_applied

    def set_optimization_applied(
        self, metadata: OptimizationAppliedMetadata
    ) -> FlowbookMetadata:
        return self.model_copy(update={"optimization_applied": metadata})

    def get_unit_tests(self) -> Optional[UnitTests]:
        return self.unit_tests

    def set_unit_tests(self, metadata: UnitTests) -> FlowbookMetadata:
        return self.model_copy(update={"unit_tests": metadata})

    @staticmethod
    def from_cell(cell: nbformat.NotebookNode) -> FlowbookMetadata:
        if "metadata" not in cell or "flowbook" not in cell["metadata"]:
            return FlowbookMetadata()
        else:
            return FlowbookMetadata.model_validate(cell["metadata"]["flowbook"])


def set_optimization_potential_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: OptimizationPotential
) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["optimization_potential"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def set_profile_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: ProfileData
) -> None:
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["profile"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def set_dynamic_dependencies_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: DynamicDependencies
) -> None:
    """Set the dynamic dependencies metadata for a cell."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["dynamic_dependencies"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def set_generated_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: GeneratedCodeMetadata
) -> None:
    """Set the generated code metadata for a cell."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["generated"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def set_optimized_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: OptimizedCodeMetadata
) -> None:
    """Set the optimized code metadata for a cell."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["optimized"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def set_optimization_applied_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: OptimizationAppliedMetadata
) -> None:
    """Set the optimization applied metadata for a cell that triggered optimizations."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["optimization_applied"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def set_unit_tests_flowbook_metadata(
    cell: nbformat.NotebookNode, metadata: UnitTests
) -> None:
    """Set the unit tests metadata for a cell."""
    if "metadata" not in cell:
        cell["metadata"] = {}
    flowbook = cell["metadata"].get("flowbook", {})
    if not isinstance(flowbook, dict):
        flowbook = FlowbookMetadata.model_validate(flowbook).model_dump()
    flowbook["unit_tests"] = metadata.model_dump()
    cell["metadata"]["flowbook"] = flowbook


def get_flowbook_metadata_from_cell(cell: nbformat.NotebookNode) -> dict | None:
    if (
        "metadata" not in cell
        or "flowbook" not in cell["metadata"]
        or cell["metadata"]["flowbook"] is None
    ):
        return None
    flowbook = cell["metadata"]["flowbook"]
    if isinstance(flowbook, dict):
        return flowbook
    # fallback, it is a BaseModel
    return FlowbookMetadata.model_validate(flowbook).model_dump()
