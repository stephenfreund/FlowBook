/**
 * Cell index conversion utilities.
 *
 * Converts between 0-based numeric indices and Excel-style alphabetic notation.
 */

/**
 * Convert 0-based index to Excel-style alpha notation.
 *
 * @param index - 0-based cell index
 * @returns Excel-style alpha string with @ prefix (e.g., @A, @B, @AA, @AB)
 *
 * @example
 * indexToAlpha(0)   // "@A"
 * indexToAlpha(25)  // "@Z"
 * indexToAlpha(26)  // "@AA"
 * indexToAlpha(51)  // "@AZ"
 * indexToAlpha(52)  // "@BA"
 * indexToAlpha(701) // "@ZZ"
 * indexToAlpha(702) // "@AAA"
 *
 * @throws {Error} If index is negative or too large
 */
export function indexToAlpha(index: number): string {
  if (index < 0) {
    throw new Error(`Index must be non-negative (got: ${index})`);
  }

  // Handle single letter (0-25): A-Z
  if (index < 26) {
    return '@' + String.fromCharCode('A'.charCodeAt(0) + index);
  }

  // Handle two letters (26-701): AA-ZZ
  if (index < 26 + 26 * 26) {
    const offset = index - 26;
    const first = String.fromCharCode('A'.charCodeAt(0) + Math.floor(offset / 26));
    const second = String.fromCharCode('A'.charCodeAt(0) + (offset % 26));
    return '@' + first + second;
  }

  // Handle three letters (702-18277): AAA-ZZZ
  if (index < 26 + 26 * 26 + 26 * 26 * 26) {
    const offset = index - (26 + 26 * 26);
    const first = String.fromCharCode('A'.charCodeAt(0) + Math.floor(offset / (26 * 26)));
    const second = String.fromCharCode('A'.charCodeAt(0) + Math.floor((offset / 26) % 26));
    const third = String.fromCharCode('A'.charCodeAt(0) + (offset % 26));
    return '@' + first + second + third;
  }

  // Index too large
  throw new Error(`Index ${index} is too large (max supported: 18277 for @ZZZ)`);
}

/**
 * Convert Excel-style alpha notation to 0-based index.
 *
 * @param alpha - Excel-style alpha string with @ prefix (e.g., @A, @B, @AA)
 * @returns 0-based cell index
 *
 * @example
 * alphaToIndex('@A')   // 0
 * alphaToIndex('@Z')   // 25
 * alphaToIndex('@AA')  // 26
 * alphaToIndex('@AZ')  // 51
 * alphaToIndex('@BA')  // 52
 * alphaToIndex('@ZZ')  // 701
 * alphaToIndex('@AAA') // 702
 *
 * @throws {Error} If format is invalid
 */
export function alphaToIndex(alpha: string): number {
  if (typeof alpha !== 'string') {
    throw new Error(`Expected string, got ${typeof alpha}`);
  }

  if (!alpha.startsWith('@')) {
    throw new Error(`Invalid format: must start with '@' (got: ${alpha})`);
  }

  const letters = alpha.substring(1); // Remove @ prefix

  if (letters.length === 0) {
    throw new Error('Invalid format: no letters after \'@\'');
  }

  if (!/^[A-Z]+$/.test(letters)) {
    throw new Error(`Invalid format: must contain only uppercase letters (got: ${alpha})`);
  }

  const length = letters.length;

  if (length === 1) {
    // Single letter: A=0, B=1, ..., Z=25
    return letters.charCodeAt(0) - 'A'.charCodeAt(0);
  } else if (length === 2) {
    // Two letters: AA=26, AB=27, ..., ZZ=701
    const first = letters.charCodeAt(0) - 'A'.charCodeAt(0);
    const second = letters.charCodeAt(1) - 'A'.charCodeAt(0);
    return 26 + first * 26 + second;
  } else if (length === 3) {
    // Three letters: AAA=702, AAB=703, ..., ZZZ=18277
    const first = letters.charCodeAt(0) - 'A'.charCodeAt(0);
    const second = letters.charCodeAt(1) - 'A'.charCodeAt(0);
    const third = letters.charCodeAt(2) - 'A'.charCodeAt(0);
    return 26 + 26 * 26 + first * 26 * 26 + second * 26 + third;
  } else {
    throw new Error(`Invalid format: too many letters (max 3, got ${length} in ${alpha})`);
  }
}
