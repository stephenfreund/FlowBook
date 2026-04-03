"""
FlowBook communication protocol.

Defines the unified JSON message format for kernel <-> client communication.
All messages use a "type" discriminator field.

Transport layer:
- Frontend (TypeScript) <-> Kernel: Comm channel ("flowbook" target)
- Python clients <-> Kernel: Execute request metadata (client->kernel),
  custom IOPub msg_type "flowbook_update" (kernel->client)

The kernel always emits both: custom IOPub (for Python clients) and comm
(for frontend, if open). This dual-send approach is intentional — the
frontend uses the comm, Python clients use IOPub, and both see the same
payload.
"""

from typing import Any, Dict, List, Optional

# ===========================================================================
# Message type constants
# ===========================================================================

# Kernel -> Client
METADATA = "metadata"      # Post-execution reproducibility data
VIOLATION = "violation"    # Predicate violation
STATUS = "status"          # Status line (icon + text)

# Client -> Kernel
NOTEBOOK_STRUCTURE = "notebook_structure"
CELL_EDITED = "cell_edited"
CONTINUE_AFTER_VIOLATION = "continue_after_violation"
SYNC = "sync"
EXEC_RESTORE = "exec_restore"

# IOPub message type for custom messages
IOPUB_MSG_TYPE = "flowbook_update"

# Comm target name
COMM_TARGET = "flowbook"


# ===========================================================================
# Kernel -> Client message builders
# ===========================================================================

def build_metadata_message(metadata: "ReproducibilityMetadata") -> dict:
    """Build a metadata message from a ReproducibilityMetadata instance."""
    return {
        "type": METADATA,
        "cell_id": metadata.cell_id,
        "execution_seq": metadata.execution_seq,
        "read_locs": metadata.read_locs,
        "write_locs": metadata.write_locs,
        "changed_locs": metadata.changed_locs,
        "stale_cells": metadata.stale_cells,
        "cell_order": metadata.cell_order,
        "structural_warnings": metadata.structural_warnings,
        "execute_duration_ms": metadata.execute_duration_ms,
        "code_duration_ms": metadata.code_duration_ms,
        "state_duration_ms": metadata.state_duration_ms,
        "check_duration_ms": metadata.check_duration_ms,
        "staleness_reasons": metadata.staleness_reasons,
        "errors": metadata.errors,
    }


def build_violation_message(
    error: "ReproducibilityError",
    accepted: bool = False,
) -> dict:
    """Build a violation message from a ReproducibilityError instance."""
    msg: Dict[str, Any] = {
        "type": VIOLATION,
        "predicate": error.error_type.value,
        "cell_id": error.cell_id,
        "locations": error.locations,
        "message": error.message,
        "accepted": accepted,
    }
    if error.causer_cell:
        msg["causer_cell"] = error.causer_cell
    if error.detail:
        msg["detail"] = error.detail
    return msg


def build_status_message(icon: str, text: str, cell_id: str = "") -> dict:
    """Build a status message (icon + text line).

    Args:
        icon: Status icon (e.g. "✓", "✗")
        text: Status text (e.g. "Execute: 42 ms | Code: 38 ms")
        cell_id: Cell ID that produced this status (for @A display)
    """
    return {
        "type": STATUS,
        "icon": icon,
        "text": text,
        "cell_id": cell_id,
    }


# ===========================================================================
# Client -> Kernel message builders
# ===========================================================================

def build_notebook_structure_message(cell_order: List[str]) -> dict:
    """Build a notebook_structure message."""
    return {
        "type": NOTEBOOK_STRUCTURE,
        "cell_order": cell_order,
    }


def build_cell_edited_message(cell_id: str) -> dict:
    """Build a cell_edited message."""
    return {
        "type": CELL_EDITED,
        "cell_id": cell_id,
    }


def build_continue_after_violation_message(enabled: bool) -> dict:
    """Build a continue_after_violation message."""
    return {
        "type": CONTINUE_AFTER_VIOLATION,
        "enabled": enabled,
    }


def build_sync_message() -> dict:
    """Build a sync message (request full state from kernel)."""
    return {"type": SYNC}


def build_exec_restore_message(cell_id: str) -> dict:
    """Build an exec_restore message."""
    return {
        "type": EXEC_RESTORE,
        "cell_id": cell_id,
    }


# ===========================================================================
# Validation
# ===========================================================================

KERNEL_TO_CLIENT_TYPES = {METADATA, VIOLATION, STATUS}
CLIENT_TO_KERNEL_TYPES = {
    NOTEBOOK_STRUCTURE, CELL_EDITED, CONTINUE_AFTER_VIOLATION,
    SYNC, EXEC_RESTORE,
}
ALL_TYPES = KERNEL_TO_CLIENT_TYPES | CLIENT_TO_KERNEL_TYPES


def validate_message(msg: dict) -> bool:
    """Check that a message has a valid type field."""
    return isinstance(msg, dict) and msg.get("type") in ALL_TYPES


# ===========================================================================
# CLI formatting
# ===========================================================================

def format_message_for_cli(msg: dict, cell_order: Optional[List[str]] = None) -> Optional[str]:
    """Format a kernel-to-client protocol message for CLI display.

    Returns a single-line string, or None if the message should not be printed.
    """
    msg_type = msg.get("type")

    if msg_type == STATUS:
        cell_id = msg.get("cell_id", "")
        cell_ref = _cell_id_to_ref(cell_id, cell_order) if cell_id else ""
        prefix = f"{cell_ref} " if cell_ref else ""
        return f"{prefix}{msg.get('icon', '')} {msg.get('text', '')}"

    if msg_type == VIOLATION:
        cell_id = msg.get("cell_id", "")
        cell_ref = _cell_id_to_ref(cell_id, cell_order) if cell_id else cell_id
        predicate = msg.get("predicate", "")
        message = msg.get("message", "")
        accepted = msg.get("accepted", False)
        tag = "ACCEPTED" if accepted else "REJECTED"
        return f"{cell_ref} ✗ [{tag}] {predicate}: {message}"

    # Don't print metadata messages — the status message covers the summary
    return None


def _cell_id_to_ref(cell_id: str, cell_order: Optional[List[str]]) -> str:
    """Convert a cell ID to @A notation if cell_order is available."""
    if not cell_order or not cell_id:
        return cell_id
    try:
        from flowbook.util.cell_index import index_to_alpha
        idx = cell_order.index(cell_id)
        return index_to_alpha(idx)
    except (ValueError, IndexError):
        return cell_id
