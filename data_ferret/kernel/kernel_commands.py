"""
Pydantic models for kernel command requests and responses.

This module defines the command protocol for the kernel_command comm channel,
providing type-safe request/response models for all kernel operations.

Commands are organized into categories:
- Checkpoint management (save, restore, delete, list, compare, clear)
- Test code execution
- Feature toggles (scalene, force_checkpoints)
"""

from typing import Any, Dict, List, Literal, Optional, Set, Union
from pydantic import BaseModel, Field

from data_ferret.kernel.types import DiffResult, TestCodeResult
from data_ferret.kernel.extended_types import TypeModel


# ============================================================================
# Base Command Models
# ============================================================================


class KernelCommandRequest(BaseModel):
    """
    Base class for all kernel command requests.

    All command requests must include a 'command' field that identifies
    which operation to perform.
    """

    command: str = Field(..., description="Command identifier")

    class Config:
        arbitrary_types_allowed = True


class KernelCommandResponse(BaseModel):
    """
    Base class for all kernel command responses.

    All responses include a status field and optional message.
    Additional data is provided in subclass-specific fields.
    """

    status: Literal["ok", "error"] = Field(..., description="Response status")
    message: str = Field(default="", description="Human-readable message")
    traceback: Optional[str] = Field(default=None, description="Stack trace for errors")

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Checkpoint Commands
# ============================================================================


class CheckpointSaveRequest(KernelCommandRequest):
    """Request to save a checkpoint of the current kernel state."""

    command: Literal["checkpoint_save"] = "checkpoint_save"
    name: str = Field(..., description="Name for the checkpoint")


class CheckpointSaveResponse(KernelCommandResponse):
    """Response from saving a checkpoint."""

    saved: Dict[str, TypeModel] = Field(
        default_factory=dict, description="Variables that were successfully saved"
    )
    removed: Dict[str, TypeModel] = Field(
        default_factory=dict, description="Variables that could not be saved"
    )
    duration: float = Field(..., description="Time taken to save checkpoint (seconds)")


class CheckpointRestoreRequest(KernelCommandRequest):
    """Request to restore a previously saved checkpoint."""

    command: Literal["checkpoint_restore"] = "checkpoint_restore"
    name: str = Field(..., description="Name of the checkpoint to restore")


class CheckpointRestoreResponse(KernelCommandResponse):
    """Response from restoring a checkpoint."""

    pass  # Success indicated by status="ok"


class CheckpointDeleteRequest(KernelCommandRequest):
    """Request to delete a checkpoint."""

    command: Literal["checkpoint_delete"] = "checkpoint_delete"
    name: str = Field(..., description="Name of the checkpoint to delete")


class CheckpointDeleteResponse(KernelCommandResponse):
    """Response from deleting a checkpoint."""

    pass  # Success indicated by status="ok"


class CheckpointListRequest(KernelCommandRequest):
    """Request to list all available checkpoints."""

    command: Literal["checkpoint_list"] = "checkpoint_list"


class CheckpointListResponse(KernelCommandResponse):
    """Response with list of available checkpoints."""

    checkpoints: List[str] = Field(
        default_factory=list, description="Names of all available checkpoints"
    )


class CheckpointCompareRequest(KernelCommandRequest):
    """Request to compare two checkpoints."""

    command: Literal["checkpoint_compare"] = "checkpoint_compare"
    name1: str = Field(..., description="Name of first checkpoint")
    name2: str = Field(..., description="Name of second checkpoint")
    keys_to_include: Optional[Set[str]] = Field(
        None,
        description="Optional set of variable names to include in comparison. If None, all variables are compared.",
    )


class CheckpointCompareResponse(KernelCommandResponse):
    """Response with diff between two checkpoints."""

    diff: DiffResult = Field(..., description="Differences between checkpoints")


class CheckpointCompareLeqRequest(KernelCommandRequest):
    """Request to compare two checkpoints using leq semantics.

    Leq mode allows extra keys in the second checkpoint and extra columns
    in DataFrames. This is useful for checking that read-before-write
    variables haven't been modified by a cell.
    """

    command: Literal["checkpoint_compare_leq"] = "checkpoint_compare_leq"
    name1: str = Field(..., description="Name of first checkpoint (pre-execution state)")
    name2: str = Field(..., description="Name of second checkpoint (post-execution state)")
    keys_to_include: Optional[Set[str]] = Field(
        None,
        description="Optional set of variable names to include in comparison (typically read-before-write set).",
    )
    column_rbw: Optional[Dict[str, Set[str]]] = Field(
        None,
        description="Optional column-level reads-before-writes. Maps variable path to set of "
                    "column names that were read-before-write. When provided, only these columns "
                    "are compared for each DataFrame.",
    )


class CheckpointCompareLeqResponse(KernelCommandResponse):
    """Response with leq diff between two checkpoints."""

    diff: DiffResult = Field(..., description="Differences between checkpoints")
    is_leq: bool = Field(..., description="True if pre <= post for the specified keys (no differences found)")


class CheckpointClearRequest(KernelCommandRequest):
    """Request to clear all checkpoints."""

    command: Literal["checkpoint_clear"] = "checkpoint_clear"


class CheckpointClearResponse(KernelCommandResponse):
    """Response from clearing all checkpoints."""

    pass  # Success indicated by status="ok"


# ============================================================================
# Feature Toggle Commands
# ============================================================================


class EnableScaleneRequest(KernelCommandRequest):
    """Request to enable Scalene profiling."""

    command: Literal["enable_scalene"] = "enable_scalene"


class EnableScaleneResponse(KernelCommandResponse):
    """Response from enabling Scalene."""

    pass  # Success indicated by status="ok"


class DisableScaleneRequest(KernelCommandRequest):
    """Request to disable Scalene profiling."""

    command: Literal["disable_scalene"] = "disable_scalene"


class DisableScaleneResponse(KernelCommandResponse):
    """Response from disabling Scalene."""

    pass  # Success indicated by status="ok"


class ForceCheckpointsRequest(KernelCommandRequest):
    """Request to enable/disable force checkpoints mode."""

    command: Literal["force_checkpoints"] = "force_checkpoints"
    enabled: bool = Field(..., description="Whether to enable force checkpoints")


class ForceCheckpointsResponse(KernelCommandResponse):
    """Response from setting force checkpoints mode."""

    enabled: bool = Field(..., description="Current force checkpoints state")


class EnableGlobalTrackingRequest(KernelCommandRequest):
    """Request to enable global variable tracking."""

    command: Literal["enable_global_tracking"] = "enable_global_tracking"


class EnableGlobalTrackingResponse(KernelCommandResponse):
    """Response from enabling global tracking."""

    pass  # Success indicated by status="ok"


class DisableGlobalTrackingRequest(KernelCommandRequest):
    """Request to disable global variable tracking."""

    command: Literal["disable_global_tracking"] = "disable_global_tracking"


class DisableGlobalTrackingResponse(KernelCommandResponse):
    """Response from disabling global tracking."""

    pass  # Success indicated by status="ok"


# ============================================================================
# Command Union Types
# ============================================================================

# Union of all request types
KernelCommandRequestUnion = Union[
    CheckpointSaveRequest,
    CheckpointRestoreRequest,
    CheckpointDeleteRequest,
    CheckpointListRequest,
    CheckpointCompareRequest,
    CheckpointCompareLeqRequest,
    CheckpointClearRequest,
    EnableScaleneRequest,
    DisableScaleneRequest,
    ForceCheckpointsRequest,
    EnableGlobalTrackingRequest,
    DisableGlobalTrackingRequest,
]

# Union of all response types
KernelCommandResponseUnion = Union[
    CheckpointSaveResponse,
    CheckpointRestoreResponse,
    CheckpointDeleteResponse,
    CheckpointListResponse,
    CheckpointCompareResponse,
    CheckpointCompareLeqResponse,
    CheckpointClearResponse,
    EnableScaleneResponse,
    DisableScaleneResponse,
    ForceCheckpointsResponse,
    EnableGlobalTrackingResponse,
    DisableGlobalTrackingResponse,
]


# ============================================================================
# Progress Messages
# ============================================================================


class ProgressMessage(BaseModel):
    """
    Progress message sent during long-running operations.

    Allows operations like test_code to send status updates
    while they're still executing.
    """

    type: Literal["progress"] = "progress"
    message: str = Field(..., description="Progress update message")

    class Config:
        arbitrary_types_allowed = True


class FinalMessage(BaseModel):
    """
    Final message wrapper for comm responses.

    Wraps the actual response with metadata indicating
    this is the final message of the operation.
    """

    type: Literal["final"] = "final"
    ok: bool = Field(..., description="Whether the operation succeeded")
    response: Optional[Dict[str, Any]] = Field(
        None, description="Response data (present if ok=True)"
    )
    error: Optional[str] = Field(
        None, description="Error message (present if ok=False)"
    )

    class Config:
        arbitrary_types_allowed = True
