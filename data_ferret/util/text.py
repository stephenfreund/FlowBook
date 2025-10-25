import re

import json
from typing import Any, Union, Tuple, Optional, Dict


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


# ANSI color code mappings
ANSI_COLORS = {
    '30': 'black',
    '31': 'red',
    '32': 'green',
    '33': 'yellow',
    '34': 'blue',
    '35': 'magenta',
    '36': 'cyan',
    '37': 'white',
    '90': 'bright-black',
    '91': 'bright-red',
    '92': 'bright-green',
    '93': 'bright-yellow',
    '94': 'bright-blue',
    '95': 'bright-magenta',
    '96': 'bright-cyan',
    '97': 'bright-white',
}


def parse_ansi_codes(code_str: str) -> Optional[Dict[str, str]]:
    """
    Parse ANSI escape code string and extract color/style information.

    Args:
        code_str: The numeric part of the ANSI code (e.g., "31" for red)

    Returns:
        Dictionary with color/style info, or None if not a color code
    """
    codes = code_str.split(';')
    result = {}

    for code in codes:
        if code in ANSI_COLORS:
            result['color'] = ANSI_COLORS[code]
        elif code == '1':
            result['bold'] = True
        elif code == '0':
            # Reset code
            return None

    return result if result else None


def parse_ansi_text(text: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Parse text with ANSI codes and extract the first color/style found.

    Args:
        text: Text potentially containing ANSI escape codes

    Returns:
        Tuple of (stripped_text, style_metadata)
        where style_metadata is a dict with 'color' and/or 'bold' keys, or None
    """
    # Pattern to match ANSI escape sequences
    ansi_pattern = r'\x1B\[([0-9;]+)m'

    # Find the first ANSI code
    match = re.search(ansi_pattern, text)
    metadata = None

    if match:
        code_str = match.group(1)
        metadata = parse_ansi_codes(code_str)

    # Strip all ANSI codes
    stripped = strip_ansi(text)

    return stripped, metadata


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
