import re

import json
from typing import Any, Union


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def transform_json(obj: Any) -> Any:
    """
    Recursively transform:
      1. Strings containing valid JSON → parsed & transformed
      2. Strings with '\\n' → list of lines each ending in '\\n'
    """
    # Handle strings
    if isinstance(obj, str):
        s = obj
        trimmed = s.strip()

        # Rule #1: try JSON parse if it looks like an object/array
        if trimmed.startswith("{") or trimmed.startswith("["):
            try:
                parsed = json.loads(s)
                return transform_json(parsed)
            except json.JSONDecodeError:
                pass

        # Rule #2: split on newlines (keep line endings)
        if "\n" in s:
            # splitlines(True) keeps the newline characters
            return s.splitlines(True)

        # otherwise unchanged
        return s

    # Handle lists/tuples: transform each element
    if isinstance(obj, list):
        return [transform_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(transform_json(item) for item in obj)

    # Handle dicts: transform each value
    if isinstance(obj, dict):
        return {key: transform_json(val) for key, val in obj.items()}

    # Other primitives (int, float, bool, None): unchanged
    return obj
