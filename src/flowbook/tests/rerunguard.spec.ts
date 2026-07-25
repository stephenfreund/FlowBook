/**
 * Tests for the RunToClean rerun check.
 *
 * The trigger: a re-executed cell (already run this sweep) that leaves
 * a cell before itself stale. Footprints are remembered only to explain
 * reports. Good programs must not be flagged (deterministic reruns,
 * random values written to fixed variables, DataFrame recreation,
 * footprint drift that marks nothing backward); reruns that re-mark
 * earlier cells must be.
 */

import {
  RunToCleanGuard,
  canonicalFootprint,
  canonicalLocKey,
  formatFootprintChange
} from '../rerunguard';
import { IReadLoc, IReproducibilityMetadata } from '../types';

const ORDER = ['A', 'B', 'C'];

function varLoc(name: string): IReadLoc {
  return { type: 'var', name };
}

function colLoc(df: string, col: string, locId: number): IReadLoc {
  return { type: 'col', name: col, qualifier: locId, var_name: df };
}

function meta(
  reads: IReadLoc[],
  writes: IReadLoc[],
  staleCells: string[] = []
): IReproducibilityMetadata {
  return {
    cell_id: 'test',
    execution_seq: 0,
    read_locs: reads,
    write_locs: writes,
    changed_locs: [],
    stale_cells: staleCells
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
  it('never reports a first execution, even with backward marks', () => {
    const guard = new RunToCleanGuard();
    expect(
      guard.noteRun('C', meta([], [varLoc('y')], ['A']), ORDER)
    ).toBeNull();
  });

  it('does not report a rerun with forward marks only', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('B', meta([], [varLoc('y')], ['C']), ORDER);
    expect(
      guard.noteRun('B', meta([], [varLoc('y')], ['C']), ORDER)
    ).toBeNull();
  });

  it('reports a rerun that marks an earlier cell stale', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('C', meta([], [varLoc('b')]), ORDER);
    const report = guard.noteRun('C', meta([], [varLoc('a')], ['B']), ORDER);
    expect(report).not.toBeNull();
    expect(report!.backwardStale).toEqual(['B']);
    expect(report!.change).not.toBeNull();
    expect(report!.change!.prevWrites).toEqual(['b']);
    expect(report!.change!.newWrites).toEqual(['a']);
  });

  it('does not report footprint drift without backward marks', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('B', meta([varLoc('x')], [varLoc('y')]), ORDER);
    expect(
      guard.noteRun('B', meta([varLoc('x')], [varLoc('z')], ['C']), ORDER)
    ).toBeNull();
  });

  it('reports even without read/write tracking metadata', () => {
    const guard = new RunToCleanGuard();
    const bare = { stale_cells: [] } as unknown as IReproducibilityMetadata;
    const bareMarked = {
      stale_cells: ['A', 'B']
    } as unknown as IReproducibilityMetadata;
    guard.noteRun('C', bare, ORDER);
    const report = guard.noteRun('C', bareMarked, ORDER);
    expect(report).not.toBeNull();
    expect(report!.backwardStale).toEqual(['A', 'B']);
    expect(report!.change).toBeNull();
  });

  it('does not flag DataFrame recreation (loc_id churn)', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun(
      'C',
      meta([colLoc('df', 'a', 101)], [colLoc('df', 'b', 101)]),
      ORDER
    );
    const report = guard.noteRun(
      'C',
      meta([colLoc('df', 'a', 202)], [colLoc('df', 'b', 202)], ['B']),
      ORDER
    );
    // The backward mark still reports, but the loc_id churn must not
    // show up as a phantom footprint change.
    expect(report).not.toBeNull();
    expect(report!.change).toBeNull();
  });

  it('never reports when metadata is missing entirely', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('C', meta([], [varLoc('a')]), ORDER);
    expect(guard.noteRun('C', null, ORDER)).toBeNull();
  });

  it('tracks cells independently', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('B', meta([], [varLoc('y')]), ORDER);
    expect(
      guard.noteRun('C', meta([], [varLoc('z')], ['A']), ORDER)
    ).toBeNull(); // first execution of C
  });
});

describe('formatFootprintChange', () => {
  it('spells out both runs read and write sets in full', () => {
    const guard = new RunToCleanGuard();
    guard.noteRun('C', meta([varLoc('x')], [varLoc('a')]), ORDER);
    const report = guard.noteRun('C', meta([], [varLoc('b')], ['B']), ORDER);
    expect(formatFootprintChange(report!.change!)).toBe(
      'the previous run read x and wrote a, ' +
        'but this run read nothing and wrote b'
    );
  });
});
