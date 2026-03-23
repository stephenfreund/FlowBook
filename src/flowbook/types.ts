/**
 * Type definitions for FlowBook kernel extension (reproducibility)
 */

/**
 * A typed read location from the ReadLoc grammar:
 *   Var(x) | Col(d, c) | Attr(d, a) | File(p)
 */
export interface IReadLoc {
  type: 'var' | 'col' | 'attr' | 'file';
  name: string;
  qualifier?: string;
}

/**
 * A typed write location from the WriteLoc grammar:
 *   Var(x) | Col(d, c) | ColAdd(d, c) | ColDel(d, c) | Rows(d) | Attr(d, a) | File(p)
 */
export interface IWriteLoc {
  type: 'var' | 'col' | 'col_add' | 'col_del' | 'rows' | 'attr' | 'file';
  name: string;
  qualifier?: string;
}

export interface IReproducibilityMetadata {
  cell_id: string;
  execution_seq: number;
  read_locs: IReadLoc[];
  write_locs: IWriteLoc[];
  changed_locs: IWriteLoc[];
  stale_cells: string[];
  cell_order: string[];
  structural_warnings?: string[];
  // Timing information (in milliseconds)
  execute_duration_ms?: number; // Total time in _do_execute_impl
  code_duration_ms?: number; // Time for _ipython_do_execute (user code)
  state_duration_ms?: number;
  check_duration_ms?: number;
  // Staleness reasons per cell: { cell_id: [reason, ...] }
  staleness_reasons?: { [cell_id: string]: IBackendStalenessReason[] };
  // Reproducibility errors (formal predicate violations)
  errors?: IReproducibilityError[];
}

/**
 * Reproducibility error from kernel (formal predicate violation).
 */
export interface IReproducibilityError {
  error_type: string;
  cell_id: string;
  locations: string[];
  message: string;
  causer_cell?: string;
  detail?: Record<string, unknown>;
}

export interface IReproducibilityCellState {
  cellId: string;
  executionSeq: number;
  readLocs: IReadLoc[];
  writeLocs: IWriteLoc[];
  isStale: boolean;
}

/**
 * Reason types from the formal model (§1.2).
 * Maps to ReasonType enum in flowbook/kernel/models.py
 *
 * Names align with formal predicates from [Inst-Run] specification:
 * - forward_stale: ForwardStale(R,W,i,j) - cell j>i reads/writes location that i wrote
 * - backward_stale: BackwardStale(W,W',i,j) - cell j<i was last writer of removed write
 * - no_read_before_write: ¬NoReadBeforeWrite - reads location written by later cell
 * - no_write_after_read: ¬NoWriteAfterRead - wrote location read by earlier fresh cell
 */
export type BackendReasonType =
  | 'never_executed' // Cell has never been run
  | 'code_changed' // Cell source code was edited
  | 'forward_stale' // A variable this cell reads was modified by another cell (was input_changed)
  | 'write_overlap' // Cell writes to same location as earlier cell (no convergence)
  | 'backward_stale' // Another cell wrote to a variable this cell also writes (was write_conflict)
  | 'no_read_before_write' // Cell reads a value written by a later cell (was reads_from_later)
  | 'order_changed' // Cell order changed affecting data flow
  | 'no_write_after_read'; // Cell wrote to location read by earlier cell (was backward_mutation)

/**
 * Frontend-computed reason types with human-readable formatting.
 * These are computed by executionhook.ts from kernel metadata.
 */
export type FrontendReasonType =
  | 'source_edited' // Source code was edited (mapped from code_changed)
  | 'variable_modified' // A variable was modified (mapped from input_changed)
  | 'writer_conflict' // This cell writes what another cell reads
  | 'unknown'; // Fallback for unclassified staleness

/**
 * Union of all reason types (backend + frontend).
 */
export type ReasonType = BackendReasonType | FrontendReasonType;

/**
 * Backend reason from kernel (as sent in staleness_reasons).
 * Minimal structure matching flowbook/kernel/models.py Reason dataclass.
 */
export interface IBackendStalenessReason {
  type: BackendReasonType;
  loc?: string; // Variable or location involved (e.g., "x", "df")
  cell_id?: string; // Cell that caused the staleness (actual ID, not @position)
}

/**
 * Frontend-computed reason with rich context for UI display.
 * Built from kernel metadata by executionhook.ts.
 */
export interface IFrontendStalenessReason {
  type: FrontendReasonType;
  causing_cell?: string; // actual cell ID
  variables?: string[];
  columns?: { [key: string]: string[] };
  message: string; // human-readable (@A notation)
}

/**
 * Union type for staleness reasons - can be either backend or frontend format.
 * Use type guards to distinguish:
 *   'message' in reason → IFrontendStalenessReason
 *   otherwise → IBackendStalenessReason
 */
export type IStalenessReason =
  | IBackendStalenessReason
  | IFrontendStalenessReason;

export interface IProposedFixEntry {
  cell_ids: string[];
  modified_source: string;
  explanation: string;
}

export interface IProposedFix {
  violation_type: string;
  mutating_cell: string;
  affected_cell: string;
  strategy: string; // "alpha_rename" | "copy_value" | "merge_cells" | "reorder"
  fix_entries: IProposedFixEntry[];
  explanation: string;
}

/**
 * Predicate types for formal predicate violations.
 * These match ErrorType enum in flowbook/kernel/models.py
 */
export type PredicateType =
  | 'no_read_and_write' // Cell reads and writes same location
  | 'write_before_read' // Reads user var not written by earlier cell
  | 'no_read_before_write' // Forward contamination
  | 'no_write_after_read'; // Backward mutation

/**
 * Unified predicate violation sent by kernel.
 *
 * All four formal predicates produce the same structure:
 * - NO_READ_AND_WRITE: Cell reads and writes same location
 * - WRITE_BEFORE_READ: Reads undefined variable
 * - NO_READ_BEFORE_WRITE: Forward contamination
 * - NO_WRITE_AFTER_READ: Backward mutation
 *
 * The `accepted` field indicates how the violation was handled:
 * - accepted=false: Rejected - execution rolled back
 * - accepted=true: Accepted (continue_after_violation) - execution continues
 * Both are shown as errors (red) in the UI.
 */
export interface IPredicateViolation {
  predicate: PredicateType;
  cell_id: string;
  locations: string[];
  message: string;
  accepted: boolean;
  causer_cell?: string;
  detail?: {
    structural_reads_detail?: { [key: string]: { [key: string]: string } };
    changes_detail?: string[];
  };
}

// ============================================================================
// Conflict Relation (▷) — TypeScript port of write_conflicts_read
// ============================================================================

/**
 * Attributes that reveal column structure.
 */
const COL_ATTRS = new Set([
  'columns',
  'keys',
  'dtypes',
  'axes',
  'T',
  'values',
  'iter',
  'describe',
  'shape',
  'size'
]);

/**
 * Attributes that reveal row structure.
 */
const ROW_ATTRS = new Set([
  'index',
  'shape',
  'size',
  'len',
  'empty',
  'axes',
  'values',
  'T'
]);

/**
 * Does write `w` invalidate read `r`?
 *
 * This is the ▷ conflict relation — the single 7×4 function that determines
 * all staleness in the system. Port of Python write_conflicts_read().
 */
export function writeConflictsRead(w: IWriteLoc, r: IReadLoc): boolean {
  switch (w.type) {
    case 'var':
      // Var(x) write conflicts with any read on the same variable
      if (r.type === 'var' || r.type === 'file') {
        return w.name === r.name;
      }
      // Col(d,c) or Attr(d,a): conflicts if x == d (qualifier)
      return w.name === r.qualifier;

    case 'col':
      // Col(d,c) only conflicts with Col(d,c) — same dataframe AND same column
      return r.type === 'col' && w.qualifier === r.qualifier && w.name === r.name;

    case 'col_add':
      // ColAdd(d,c) only conflicts with Attr reads on COL_ATTRS
      return (
        r.type === 'attr' &&
        w.qualifier === r.qualifier &&
        COL_ATTRS.has(r.name)
      );

    case 'col_del':
      // ColDel(d,c) conflicts with Col(d,c) AND Attr(d, COL_ATTRS)
      if (r.type === 'col') {
        return w.qualifier === r.qualifier && w.name === r.name;
      }
      return (
        r.type === 'attr' &&
        w.qualifier === r.qualifier &&
        COL_ATTRS.has(r.name)
      );

    case 'rows':
      // Rows(d) conflicts with all Col(d,*) AND Attr(d, ROW_ATTRS)
      if (r.type === 'col') {
        return w.name === r.qualifier;
      }
      if (r.type === 'attr') {
        return w.name === r.qualifier && ROW_ATTRS.has(r.name);
      }
      return false;

    case 'attr':
      // Attr(d,a) only conflicts with Attr(d,a) — same dataframe AND same attr
      return (
        r.type === 'attr' &&
        w.qualifier === r.qualifier &&
        w.name === r.name
      );

    case 'file':
      // File(p) only conflicts with File(p)
      return r.type === 'file' && w.name === r.name;

    default:
      return false;
  }
}

/**
 * Check if any write loc in `wlocs` conflicts with any read loc in `rlocs`.
 */
export function hasConflict(
  wlocs: IWriteLoc[],
  rlocs: IReadLoc[]
): boolean {
  for (const w of wlocs) {
    for (const r of rlocs) {
      if (writeConflictsRead(w, r)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Find all read locs from `rlocs` that are invalidated by any write in `wlocs`.
 * Returns the conflicting read locs.
 */
export function findConflictingReads(
  wlocs: IWriteLoc[],
  rlocs: IReadLoc[]
): IReadLoc[] {
  const result: IReadLoc[] = [];
  for (const r of rlocs) {
    for (const w of wlocs) {
      if (writeConflictsRead(w, r)) {
        result.push(r);
        break;
      }
    }
  }
  return result;
}

/**
 * Format a ReadLoc for display (e.g., "df.price", "x", "data.csv").
 */
export function formatReadLoc(loc: IReadLoc): string {
  if (loc.qualifier) {
    return `${loc.qualifier}.${loc.name}`;
  }
  return loc.name;
}

/**
 * Format a WriteLoc for display with type annotation.
 */
export function formatWriteLoc(loc: IWriteLoc): string {
  if (loc.qualifier) {
    return `${loc.qualifier}.${loc.name}`;
  }
  return loc.name;
}

/**
 * Map a WriteLoc to its output ReadLoc (for write-write conflict via ▷).
 */
export function writeLocOutput(w: IWriteLoc): IReadLoc {
  switch (w.type) {
    case 'var':
      return { type: 'var', name: w.name };
    case 'col':
    case 'col_add':
    case 'col_del':
      return { type: 'col', name: w.name, qualifier: w.qualifier };
    case 'rows':
      return { type: 'var', name: w.name };
    case 'attr':
      return { type: 'attr', name: w.name, qualifier: w.qualifier };
    case 'file':
      return { type: 'file', name: w.name };
    default:
      return { type: 'var', name: w.name };
  }
}
