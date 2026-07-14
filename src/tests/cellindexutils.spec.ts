/**
 * Unit tests for the @A cell-index alpha conversions.
 */

import { indexToAlpha, alphaToIndex } from '../cellindexutils';

describe('indexToAlpha', () => {
  it('maps single letters', () => {
    expect(indexToAlpha(0)).toBe('@A');
    expect(indexToAlpha(1)).toBe('@B');
    expect(indexToAlpha(25)).toBe('@Z');
  });

  it('maps two-letter range', () => {
    expect(indexToAlpha(26)).toBe('@AA');
    expect(indexToAlpha(27)).toBe('@AB');
    expect(indexToAlpha(26 + 26 * 26 - 1)).toBe('@ZZ');
  });

  it('maps three-letter range start', () => {
    expect(indexToAlpha(26 + 26 * 26)).toBe('@AAA');
  });

  it('throws on negative index', () => {
    expect(() => indexToAlpha(-1)).toThrow();
  });
});

describe('alphaToIndex', () => {
  it('round-trips every index through indexToAlpha', () => {
    for (const i of [0, 1, 25, 26, 27, 700, 701, 702, 1000]) {
      expect(alphaToIndex(indexToAlpha(i))).toBe(i);
    }
  });
});
