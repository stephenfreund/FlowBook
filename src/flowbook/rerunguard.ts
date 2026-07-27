/**
 * Read/write footprint comparison for the run-until-clean loop.
 *
 * Implements the check from the paper's Progress theorem (RunToClean; see
 * FORMAL_DEVELOPMENT.md): the loop executes the first stale cell in
 * notebook order, recording the set E of cells it has executed. If it
 * executes a cell in E a second time, the new run's read and write sets
 * must match those recorded by the cell's previous run. If they do not,
 * the loop may fail to terminate — a rerun that drops a write can mark an
 * earlier cell stale (BackwardStale) again and again — so the guard
 * reports the rerun as a *warning* at that cell and the loop continues.
 * The hard stop is the per-cell run counter: once a cell has run more
 * than MAX_RERUNS_PER_CELL times in one sweep, the loop fails with a
 * potential non-termination error.
 *
 * Deterministic cells never trigger the report, and neither do
 * nondeterministic cells whose read and write sets are fixed while the
 * values they write vary (e.g. cells that draw random numbers): only the
 * location sets are compared, never values.
 *
 * The comparison is at name level: loc dicts carry an object-identity
 * `qualifier` (a stable_id) that legitimately changes when a rerun
 * recreates an object (`df = pd.read_csv(...)` allocates a new DataFrame
 * on every run), so numeric qualifiers are dropped and each location is
 * keyed by `type|owner|name` where the owner is the accessing variable.
 */

import { IReadLoc, IReproducibilityMetadata, IWriteLoc } from './types';

/**
 * Maximum runs of a single cell within one run-until-clean sweep. The
 * paper's Progress theorem is qualitative (no run bound exists for
 * footprint-unstable cells), so this is a pragmatic limit: warnings are
 * issued while under it, and exceeding it stops the sweep.
 */
export const MAX_RERUNS_PER_CELL = 10;

/** Reduce a ReadLoc/WriteLoc to a name-level key stable across reruns. */
export function canonicalLocKey(loc: IReadLoc | IWriteLoc): string {
  const owner =
    typeof loc.qualifier === 'string' ? loc.qualifier : (loc.var_name ?? '');
  return `${loc.type}|${owner}|${loc.name}`;
}

/** Canonicalize a loc list to a comparable set of name-level keys. */
export function canonicalFootprint(
  locs: (IReadLoc | IWriteLoc)[] | undefined | null
): Set<string> {
  const keys = new Set<string>();
  for (const loc of locs ?? []) {
    keys.add(canonicalLocKey(loc));
  }
  return keys;
}

/** Human-readable rendering of a name-level key. */
export function formatFootprintKey(key: string): string {
  const [type, owner, name] = key.split('|');
  if (type === 'var') {
    return name;
  }
  if (type === 'col') {
    return owner ? `${owner}.${name}` : `.${name}`;
  }
  if (type === 'cols' || type === 'rows') {
    return `${type}(${owner || name})`;
  }
  if (type === 'file') {
    return `file:${name}`;
  }
  return `${type}:${name}`;
}

export interface IFootprintChange {
  /** Full name-level footprints of the two runs, formatted and sorted. */
  prevReads: string[];
  prevWrites: string[];
  newReads: string[];
  newWrites: string[];
  /** The differences, formatted and sorted. */
  readsAdded: string[];
  readsRemoved: string[];
  writesAdded: string[];
  writesRemoved: string[];
}

function fmtAll(keys: Set<string>): string[] {
  return Array.from(keys, formatFootprintKey).sort();
}

function diff(prev: Set<string>, next: Set<string>): [string[], string[]] {
  const added: string[] = [];
  const removed: string[] = [];
  for (const k of next) {
    if (!prev.has(k)) {
      added.push(formatFootprintKey(k));
    }
  }
  for (const k of prev) {
    if (!next.has(k)) {
      removed.push(formatFootprintKey(k));
    }
  }
  return [added.sort(), removed.sort()];
}

function fmtSet(items: string[]): string {
  return items.length ? items.join(', ') : 'nothing';
}

/**
 * Describe a footprint change by spelling out both runs' read and
 * write sets in full.
 */
export function formatFootprintChange(change: IFootprintChange): string {
  return (
    `the previous run read ${fmtSet(change.prevReads)} and wrote ` +
    `${fmtSet(change.prevWrites)}, but this run read ` +
    `${fmtSet(change.newReads)} and wrote ${fmtSet(change.newWrites)}`
  );
}

/** A warning report from the rerun check. */
export interface IRerunReport {
  /** Cells before the rerun cell that it marked stale, document order. */
  backwardStale: string[];
  /** How many times the rerun cell has run this sweep. */
  runCount: number;
  /** The footprint change, when tracking metadata allows spelling it out. */
  change: IFootprintChange | null;
}

/**
 * Implements the RunToClean rerun check for one sweep.
 *
 * Call `noteRun` after every committed cell execution. The check
 * triggers — returning a warning report — exactly when a *re-executed*
 * cell (one already run this sweep) leaves a cell before itself stale:
 * the only way staleness moves backward is a run dropping a write owned
 * by an earlier cell, and when a rerun does that, the sweep may never
 * terminate. The caller warns and continues, stopping only when
 * `capExceeded` becomes true.
 *
 * Footprints are remembered purely to explain a report; the trigger
 * never compares them. The caller must run cells first-stale-first, so
 * that any stale cell before the rerun cell afterwards was marked by
 * that run.
 */
export class RunToCleanGuard {
  private _executed = new Set<string>();
  private _runCounts = new Map<string, number>();
  private _recorded = new Map<
    string,
    { reads: Set<string>; writes: Set<string> }
  >();

  /** Number of times `noteRun` has seen `cellId` this sweep. */
  runCount(cellId: string): number {
    return this._runCounts.get(cellId) ?? 0;
  }

  /** True once `cellId` has run more than MAX_RERUNS_PER_CELL times. */
  capExceeded(cellId: string): boolean {
    return this.runCount(cellId) > MAX_RERUNS_PER_CELL;
  }

  private _noteFootprint(
    cellId: string,
    meta: IReproducibilityMetadata | null | undefined
  ): IFootprintChange | null {
    if (
      !meta ||
      (meta.read_locs === undefined && meta.write_locs === undefined)
    ) {
      // No tracking info: nothing to compare; drop any stale record so a
      // later run is not compared against outdated data.
      this._recorded.delete(cellId);
      return null;
    }
    const reads = canonicalFootprint(meta.read_locs);
    const writes = canonicalFootprint(meta.write_locs);
    const previous = this._recorded.get(cellId);
    this._recorded.set(cellId, { reads, writes });
    if (!previous) {
      return null;
    }
    const [readsAdded, readsRemoved] = diff(previous.reads, reads);
    const [writesAdded, writesRemoved] = diff(previous.writes, writes);
    if (
      !readsAdded.length &&
      !readsRemoved.length &&
      !writesAdded.length &&
      !writesRemoved.length
    ) {
      return null;
    }
    return {
      prevReads: fmtAll(previous.reads),
      prevWrites: fmtAll(previous.writes),
      newReads: fmtAll(reads),
      newWrites: fmtAll(writes),
      readsAdded,
      readsRemoved,
      writesAdded,
      writesRemoved
    };
  }

  noteRun(
    cellId: string,
    meta: IReproducibilityMetadata | null | undefined,
    cellOrder: string[]
  ): IRerunReport | null {
    const rerun = this._executed.has(cellId);
    this._executed.add(cellId);
    this._runCounts.set(cellId, this.runCount(cellId) + 1);
    const change = this._noteFootprint(cellId, meta);
    if (!rerun || !meta) {
      return null;
    }
    const position = cellOrder.indexOf(cellId);
    if (position < 0) {
      return null;
    }
    const backwardStale = (meta.stale_cells ?? [])
      .filter(cid => {
        const p = cellOrder.indexOf(cid);
        return p >= 0 && p < position;
      })
      .sort((a, b) => cellOrder.indexOf(a) - cellOrder.indexOf(b));
    if (!backwardStale.length) {
      return null;
    }
    return { backwardStale, runCount: this.runCount(cellId), change };
  }
}
