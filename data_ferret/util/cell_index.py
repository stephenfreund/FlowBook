"""
Cell index conversion utilities.

Converts between 0-based numeric indices and Excel-style alphabetic notation.
"""


def index_to_alpha(index: int) -> str:
    """
    Convert 0-based index to Excel-style alpha notation.

    Args:
        index: 0-based cell index

    Returns:
        Excel-style alpha string with @ prefix (e.g., @A, @B, @AA, @AB)

    Examples:
        0 → @A
        25 → @Z
        26 → @AA
        51 → @AZ
        52 → @BA
        701 → @ZZ
        702 → @AAA

    Raises:
        ValueError: If index is negative or too large
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative (got: {index})")

    # Handle single letter (0-25): A-Z
    if index < 26:
        return "@" + chr(ord('A') + index)

    # Handle two letters (26-701): AA-ZZ
    if index < 26 + 26 * 26:
        offset = index - 26
        first = chr(ord('A') + offset // 26)
        second = chr(ord('A') + offset % 26)
        return "@" + first + second

    # Handle three letters (702-18277): AAA-ZZZ
    if index < 26 + 26 * 26 + 26 * 26 * 26:
        offset = index - (26 + 26 * 26)
        first = chr(ord('A') + offset // (26 * 26))
        second = chr(ord('A') + (offset // 26) % 26)
        third = chr(ord('A') + offset % 26)
        return "@" + first + second + third

    # Index too large
    raise ValueError(f"Index {index} is too large (max supported: 18277 for @ZZZ)")


def alpha_to_index(alpha: str) -> int:
    """
    Convert Excel-style alpha notation to 0-based index.

    Args:
        alpha: Excel-style alpha string with @ prefix (e.g., @A, @B, @AA)

    Returns:
        0-based cell index

    Examples:
        @A → 0
        @Z → 25
        @AA → 26
        @AZ → 51
        @BA → 52
        @ZZ → 701
        @AAA → 702

    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(alpha, str):
        raise ValueError(f"Expected string, got {type(alpha).__name__}")

    if not alpha.startswith('@'):
        raise ValueError(f"Invalid format: must start with '@' (got: {alpha})")

    letters = alpha[1:]  # Remove @ prefix

    if not letters:
        raise ValueError("Invalid format: no letters after '@'")

    if not letters.isalpha():
        raise ValueError(f"Invalid format: must contain only letters (got: {alpha})")

    if not letters.isupper():
        raise ValueError(f"Invalid format: letters must be uppercase (got: {alpha})")

    length = len(letters)

    if length == 1:
        # Single letter: A=0, B=1, ..., Z=25
        return ord(letters[0]) - ord('A')
    elif length == 2:
        # Two letters: AA=26, AB=27, ..., ZZ=701
        first = ord(letters[0]) - ord('A')
        second = ord(letters[1]) - ord('A')
        return 26 + first * 26 + second
    elif length == 3:
        # Three letters: AAA=702, AAB=703, ..., ZZZ=18277
        first = ord(letters[0]) - ord('A')
        second = ord(letters[1]) - ord('A')
        third = ord(letters[2]) - ord('A')
        return 26 + 26 * 26 + first * 26 * 26 + second * 26 + third
    else:
        raise ValueError(f"Invalid format: too many letters (max 3, got {length} in {alpha})")
