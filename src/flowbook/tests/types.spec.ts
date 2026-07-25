/**
 * Unit tests for the TS side of the ▷ conflict relation and helpers.
 *
 * The ▷ cases come from the SHARED parity fixture
 * (tests/fixtures/conflict_cases.json), which
 * flowbook/kernel/tests/test_frontend_parity.py runs against the Python
 * write_conflicts_read — either implementation drifting fails a test on
 * its own side.
 */

import {
  IReadLoc,
  IWriteLoc,
  writeConflictsRead,
  findConflictingReads,
  formatReadLoc,
  escapeHtml
} from '../types';

import conflictCases from './fixtures/conflict_cases.json';

interface IConflictCase {
  desc: string;
  write: IWriteLoc;
  read: IReadLoc;
  conflicts: boolean;
}

const CASES = (conflictCases as { cases: IConflictCase[] }).cases;

describe('writeConflictsRead (▷) — shared fixture', () => {
  it('covers the load-bearing matrix pairs', () => {
    // Every ▷ pair with interesting semantics must appear in the fixture
    // so both language implementations stay pinned on it.
    const required = [
      'var:var',
      'var:col',
      'var:cols',
      'var:rows',
      'col:col',
      'col:cols',
      'col:var',
      'col:rows',
      'cols:col',
      'cols:cols',
      'cols:rows',
      'cols:var',
      'rows:col',
      'rows:rows',
      'rows:cols',
      'rows:var',
      'file:file',
      'file:var'
    ];
    for (const pair of required) {
      const [w, r] = pair.split(':');
      expect(CASES.some(c => c.write.type === w && c.read.type === r)).toBe(
        true
      );
    }
  });

  it.each(CASES.map(c => [c.desc, c] as [string, IConflictCase]))(
    '%s',
    (_desc, c) => {
      expect(writeConflictsRead(c.write, c.read)).toBe(c.conflicts);
    }
  );
});

describe('findConflictingReads', () => {
  it('returns each conflicting read once, preserving read order', () => {
    const writes: IWriteLoc[] = [
      { type: 'col', name: 'price', qualifier: 'df' },
      { type: 'var', name: 'x' }
    ];
    const reads: IReadLoc[] = [
      { type: 'var', name: 'x' },
      { type: 'col', name: 'qty', qualifier: 'df' },
      { type: 'col', name: 'price', qualifier: 'df' }
    ];
    const result = findConflictingReads(writes, reads);
    expect(result).toEqual([
      { type: 'var', name: 'x' },
      { type: 'col', name: 'price', qualifier: 'df' }
    ]);
  });

  it('returns empty for no conflicts', () => {
    expect(
      findConflictingReads(
        [{ type: 'col', name: 'a', qualifier: 'df' }],
        [{ type: 'var', name: 'df' }]
      )
    ).toEqual([]);
  });
});

describe('formatReadLoc', () => {
  it('formats qualified and bare locs', () => {
    expect(formatReadLoc({ type: 'col', name: 'price', qualifier: 'df' })).toBe(
      'df.price'
    );
    expect(formatReadLoc({ type: 'var', name: 'x' })).toBe('x');
  });

  it('uses var_name for LocRef (numeric) qualifiers', () => {
    expect(
      formatReadLoc({
        type: 'col',
        name: 'price',
        qualifier: 7,
        var_name: 'df'
      })
    ).toBe('df.price');
  });
});

describe('escapeHtml', () => {
  it('escapes all HTML-special characters', () => {
    expect(escapeHtml('<img src=x onerror="alert(\'p\')" & more>')).toBe(
      '&lt;img src=x onerror=&quot;alert(&#39;p&#39;)&quot; &amp; more&gt;'
    );
  });

  it('escapes & first (no double escaping)', () => {
    expect(escapeHtml('&lt;')).toBe('&amp;lt;');
  });

  it('leaves backticks alone (used for code conversion downstream)', () => {
    expect(escapeHtml('`x`')).toBe('`x`');
  });
});
