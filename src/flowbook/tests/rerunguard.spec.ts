/**
 * Tests for the RunToClean rerun footprint guard.
 *
 * Good programs must not be flagged (deterministic reruns, random values
 * written to fixed variables, DataFrame recreation churning loc_ids);
 * write/read set changes on re-execution must be flagged.
 */

import {
  RunToCleanGuard,
  canonicalFootprint,
  canonicalLocKey,
  formatFootprintChange
} from '../rerunguard';
import { IReadLoc, IReproducibilityMetadata } from '../types';

function varLoc(name: string): IReadLoc {
  return { type: 'var', name };
}

function colLoc(df: string, col: string, locId: number): IReadLoc {
  return { type: 'col', name: col, qualifier: locId, var_name: df };
}

function meta(reads: IReadLoc[], writes: IReadLoc[]): IReproducibilityMetadata {
  return {
    cell_id: 'test',
    execution_seq: 0,
    read_locs: reads,
    write_locs: writes,
    changed_locs: [],
    stale_cells: []
  } as unknown as IReproducibilityMetadata;
}

describe('canonicalLocKey', () => {
  it('keys variables by name', () => {
    expect(canonicalLocKey(varLoc('x'))).toBe('var||x');
  });

  it('drops numeric loc_id qualifiers (object identity churns)', () => {
    expect(canonicalLocKey(colLoc('df', 'price', 101))).toBe(
      canonicalLocKey(colLoc('df', 'price', 202))
    );
  });

  it('keeps string qualifiers (variable names)', () => {
    const loc: IReadLoc = { type: 'col', name: 'price', qualifier: 'df' };
    expect(canonicalLocKey(loc)).toBe('col|df|price');
    expect(canonicalLocKey(loc)).toBe(
      canonicalLocKey(colLoc('df', 'price', 7))
    );
  });
});

describe('canonicalFootprint', () => {
  it('deduplicates and canonicalizes', () => {
    const fp = canonicalFootprint([
      varLoc('x'),
      varLoc('x'),
      colLoc('df', 'a', 1)
    ]);
    expect(fp.size).toBe(2);
    expect(fp.has('var||x')).toBe(true);
    expect(fp.has('col|df|a')).toBe(true);
  });
});

describe('RunToCleanGuard', () => {
  it('never flags a first run', () => {
    const guard = new RunToCleanGuard();
    expect(guard.noteRun('A', meta([varLoc('x')], [varLoc('y')]))).toBeNull();
  });

  it('does not flag an identical rerun', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('A', meta([varLoc('x')], [varLoc('y')]));
    expect(guard.noteRun('A', meta([varLoc('x')], [varLoc('y')]))).toBeNull();
  });

  it('does not flag random values written to fixed variables', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('A', meta([], [varLoc('x')]));
    expect(guard.noteRun('A', meta([], [varLoc('x')]))).toBeNull();
  });

  it('does not flag DataFrame recreation (loc_id churn)', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun(
      'A',
      meta([colLoc('df', 'a', 101)], [colLoc('df', 'b', 101)])
    );
    expect(
      guard.noteRun(
        'A',
        meta([colLoc('df', 'a', 202)], [colLoc('df', 'b', 202)])
      )
    ).toBeNull();
  });

  it('flags a write set flip', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('A', meta([], [varLoc('a')]));
    const change = guard.noteRun('A', meta([], [varLoc('b')]));
    expect(change).not.toBeNull();
    expect(change!.writesAdded).toEqual(['b']);
    expect(change!.writesRemoved).toEqual(['a']);
    expect(change!.readsAdded).toEqual([]);
  });

  it('flags a read set change', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('A', meta([varLoc('x')], [varLoc('y')]));
    const change = guard.noteRun('A', meta([varLoc('z')], [varLoc('y')]));
    expect(change).not.toBeNull();
    expect(change!.readsAdded).toEqual(['z']);
    expect(change!.readsRemoved).toEqual(['x']);
  });

  it('skips and forgets when metadata is missing', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('A', meta([], [varLoc('a')]));
    expect(guard.noteRun('A', null)).toBeNull();
    // Record dropped: a differing run now counts as a first run...
    expect(guard.noteRun('A', meta([], [varLoc('b')]))).toBeNull();
    // ...but comparisons resume afterwards.
    expect(guard.noteRun('A', meta([], [varLoc('c')]))).not.toBeNull();
  });

  it('tracks cells independently', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('A', meta([], [varLoc('a')]));
    expect(guard.noteRun('B', meta([], [varLoc('b')]))).toBeNull();
    expect(guard.noteRun('A', meta([], [varLoc('a')]))).toBeNull();
  });
});

describe('formatFootprintChange', () => {
  it('renders the changed sets', () => {
    const text = formatFootprintChange({
      readsAdded: [],
      readsRemoved: [],
      writesAdded: ['b'],
      writesRemoved: ['a']
    });
    expect(text).toContain('writes +{b}');
    expect(text).toContain('writes -{a}');
  });
});
