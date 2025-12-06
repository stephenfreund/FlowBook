"""Generic metadata extraction from kernel outputs."""
from typing import Any, Dict, List, Callable, Tuple
import nbformat
from data_ferret.util.ferret_metadata import (
    ProfileData,
    DynamicDependencies,
    set_profile_ferret_metadata,
    set_dynamic_dependencies_ferret_metadata,
)

# Registry mapping metadata keys to (validator, setter) pairs
METADATA_HANDLERS: Dict[str, Tuple[Callable, Callable]] = {
    "profile": (
        lambda data: ProfileData.model_validate(data),
        set_profile_ferret_metadata
    ),
    "tracking": (
        lambda data: DynamicDependencies.model_validate(data),
        set_dynamic_dependencies_ferret_metadata
    ),
    # Future metadata types just need to be added here
}


def extract_and_set_metadata(cell: nbformat.NotebookNode, outputs: List[Dict[str, Any]]) -> None:
    """
    Extract all known metadata types from outputs and set them on the cell.

    This function loops through outputs, finds metadata, and applies all registered
    handlers automatically. Adding new metadata types only requires updating the
    METADATA_HANDLERS registry.

    Args:
        cell: The notebook cell to update
        outputs: List of output dicts from kernel execution
    """
    for output in outputs:
        if 'metadata' not in output:
            continue

        output_metadata = output['metadata']

        # Process all registered metadata types
        for metadata_key, (validator, setter) in METADATA_HANDLERS.items():
            if metadata_key in output_metadata:
                try:
                    validated = validator(output_metadata[metadata_key])
                    setter(cell, validated)
                except Exception as e:
                    print(f"Warning: Failed to process {metadata_key} metadata: {e}")
