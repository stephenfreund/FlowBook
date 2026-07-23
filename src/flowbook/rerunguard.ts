/**
 * Read/write footprint comparison for the run-until-clean loop.
 *
 * Implements the check from the paper's Progress theorem (RunToClean; see
 * FORMAL_DEVELOPMENT.md): the loop executes the first stale cell in
 * notebook order, recording the set E of cells it has executed. If it
 * executes a cell in E a second time, the new run's read and write sets
 * must match those recorded by the cell's previous run. If they do not,
 * the loop may fail to terminate — a rerun that drops a write can mark an
 * earlier cell stale (BackwardStale) again and again — so the loop reports
 * potential non-termination at that cell.
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
  readsAdded: string[];
  readsRemoved: string[];
  writesAdded: string[];
  writesRemoved: string[];
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

/** One-line description of a footprint change for reports. */
export function formatFootprintChange(change: IFootprintChange): string {
  const parts: string[] = [];
  if (change.readsAdded.length) {
    parts.push(`reads +{${change.readsAdded.join(', ')}}`);
  }
  if (change.readsRemoved.length) {
    parts.push(`reads -{${change.readsRemoved.join(', ')}}`);
  }
  if (change.writesAdded.length) {
    parts.push(`writes +{${change.writesAdded.join(', ')}}`);
  }
  if (change.writesRemoved.length) {
    parts.push(`writes -{${change.writesRemoved.join(', ')}}`);
  }
  return parts.length ? parts.join('; ') : 'footprint changed';
}

/**
 * Tracks per-cell footprints across one run-until-clean sweep.
 *
 * Call `noteRun` after every committed cell execution. The first
 * execution of a cell records its footprint and returns null; a
 * re-execution whose read and write sets match returns null; a
 * re-execution whose sets changed returns a change summary, which the
 * loop reports as potential non-termination.
 */
export class RunToCleanGuard {
  private _recorded = new Map<
    string,
    { reads: Set<string>; writes: Set<string> }
  >();

  noteRun(
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
    return { readsAdded, readsRemoved, writesAdded, writesRemoved };
  }
}
