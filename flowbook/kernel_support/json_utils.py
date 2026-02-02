"""JSON serialization utilities for kernel data."""

import numpy as np


def make_json_safe(obj):
    """Convert an object to a JSON-safe format, handling numpy arrays and NaN values."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size > 100:
            return {
                "_type": "ndarray",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "size": int(obj.size),
                "summary": f"Array of shape {obj.shape}"
            }
        try:
            result = obj.tolist()
            return make_json_safe(result)
        except Exception:
            return {
                "_type": "ndarray",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "size": int(obj.size)
            }
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        else:
            return obj.item()
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        else:
            return obj
    else:
        return obj
