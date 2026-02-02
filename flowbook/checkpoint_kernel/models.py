"""
Data models for checkpoint benchmarking kernel.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CheckpointMetadata:
    """Metadata returned after each cell execution with timing info."""

    cell_id: str
    execution_count: int
    cell_runtime_s: float
    commit_time_s: float
    error: Optional[str] = None

    def to_display_metadata(self) -> Dict[str, Any]:
        """Format for display in output metadata."""
        result = {
            "flowbook_checkpoint": {
                "cell_id": self.cell_id,
                "execution_count": self.execution_count,
                "cell_runtime_s": self.cell_runtime_s,
                "commit_time_s": self.commit_time_s,
            }
        }
        if self.error:
            result["flowbook_checkpoint"]["error"] = self.error
        return result
