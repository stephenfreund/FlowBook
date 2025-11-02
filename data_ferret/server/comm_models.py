"""
Pydantic models for comm message requests and responses.

These models provide type-safe communication between server commands
and kernel comm handlers.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DebugCommandRequest(BaseModel):
    """Request model for debug_command comm message."""
    cmd: str = Field(..., description="Debugger command to execute")


class DebugCommandResponse(BaseModel):
    """Response model for debug_command comm message."""
    ok: bool = Field(..., description="Whether the command succeeded")
    result: Optional[str] = Field(None, description="Command result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
