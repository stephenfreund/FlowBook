"""Cell index conversion utilities.

Converts between 0-based code-cell indices and @-label notation.
Mirrors the TypeScript implementation in src/cellindexutils.ts.

The @A notation indexes code cells only -- markdown cells are skipped.
@A = first code cell, @B = second code cell, etc.

Mapping:
  Single letter  (0-25):      A=0, B=1, ..., Z=25
  Two letters    (26-701):    AA=26, AB=27, ..., ZZ=701
  Three letters  (702-18277): AAA=702, AAB=703, ..., ZZZ=18277
"""

import re

_MAX_INDEX = 26 + 26 * 26 + 26 * 26 * 26 - 1  # 18277 (@ZZZ)


def index_to_alpha(index: int) -> str:
    """Convert 0-based code-cell index to @-label.

    0 -> '@A', 25 -> '@Z', 26 -> '@AA', 701 -> '@ZZ', 702 -> '@AAA'

    Raises ValueError if index is negative or > 18277.
    """
    if index < 0:
        raise ValueError(f'Index must be non-negative (got: {index})')

    # Single letter (0-25): A-Z
    if index < 26:
        return '@' + chr(ord('A') + index)

    # Two letters (26-701): AA-ZZ
    if index < 26 + 26 * 26:
        offset = index - 26
        first = chr(ord('A') + offset // 26)
        second = chr(ord('A') + offset % 26)
        return '@' + first + second

    # Three letters (702-18277): AAA-ZZZ
    if index <= _MAX_INDEX:
        offset = index - (26 + 26 * 26)
        first = chr(ord('A') + offset // (26 * 26))
        second = chr(ord('A') + (offset // 26) % 26)
        third = chr(ord('A') + offset % 26)
        return '@' + first + second + third

    raise ValueError(
        f'Index {index} is too large (max supported: {_MAX_INDEX} for @ZZZ)'
    )


def alpha_to_index(label: str) -> int:
    """Convert @-label to 0-based code-cell index.

    '@A' -> 0, '@Z' -> 25, '@AA' -> 26, '@ZZ' -> 701, '@AAA' -> 702

    Accepts with or without @ prefix: '@C' and 'C' both -> 2.
    Raises ValueError for invalid format.
    """
    if not isinstance(label, str):
        raise ValueError(f'Expected string, got {type(label).__name__}')

    # Strip optional @ prefix
    letters = label[1:] if label.startswith('@') else label

    if len(letters) == 0:
        raise ValueError(f'Invalid format: no letters after \'@\' (got: {label!r})')

    # Normalize to uppercase
    letters = letters.upper()

    if not re.fullmatch(r'[A-Z]+', letters):
        raise ValueError(
            f'Invalid format: must contain only letters (got: {label!r})'
        )

    length = len(letters)

    if length == 1:
        # Single letter: A=0, B=1, ..., Z=25
        return ord(letters[0]) - ord('A')

    if length == 2:
        # Two letters: AA=26, AB=27, ..., ZZ=701
        first = ord(letters[0]) - ord('A')
        second = ord(letters[1]) - ord('A')
        return 26 + first * 26 + second

    if length == 3:
        # Three letters: AAA=702, AAB=703, ..., ZZZ=18277
        first = ord(letters[0]) - ord('A')
        second = ord(letters[1]) - ord('A')
        third = ord(letters[2]) - ord('A')
        return 26 + 26 * 26 + first * 26 * 26 + second * 26 + third

    raise ValueError(
        f'Invalid format: too many letters (max 3, got {length} in {label!r})'
    )


def parse_cell_ref(cell: str) -> int:
    """Parse a cell reference in any supported format.

    Accepts:
    - @-labels: '@C' -> 2, '@AA' -> 26
    - Plain letters: 'C' -> 2, 'AA' -> 26
    - Numeric strings: '2' -> 2

    Returns 0-based code-cell index.
    Raises ValueError for invalid input.
    """
    if not isinstance(cell, str) or not cell:
        raise ValueError(f'Invalid cell reference: {cell!r}')

    cell = cell.strip()

    # If it starts with @, treat as alpha label
    if cell.startswith('@'):
        return alpha_to_index(cell)

    # If it's purely numeric, treat as integer index
    if cell.isdigit():
        return int(cell)

    # Otherwise treat as alpha label without @ prefix
    return alpha_to_index(cell)
