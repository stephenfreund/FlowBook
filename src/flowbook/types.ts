/**
 * Type definitions for FlowBook kernel extension (reproducibility)
 */

/**
 * A typed read location from the ReadLoc grammar:
 *   Var(x) | Col(d, c) | Attr(d, a) | File(p)
 *
 * The qualifier identifies the DataFrame:
 * - string: variable name (legacy, or when StableIdMap is not available)
 * - number: stable loc_id from StableIdMap (with var_name for display)
 *
 * When qualifier is a number (loc_id), var_name holds the variable name
 * used to access the object. Two locs with the same loc_id refer to the
 * same DataFrame even if accessed through different variable names (aliases).
 */
export interface IReadLoc {
  type: 'var' | 'col' | 'attr' | 'file';
  name: string;
  qualifier?: string | number;
  var_name?: string; // Present when qualifier is a loc_id (number)
}

/**
 * A typed write location from the WriteLoc grammar:
 *   Var(x) | Col(d, c) | ColAdd(d, c) | ColDel(d, c) | Rows(d) | Attr(d, a) | File(p)
 *
 * Same qualifier semantics as IReadLoc.
 */
export interface IWriteLoc {
  type: 'var' | 'col' | 'col_add' | 'col_del' | 'rows' | 'attr' | 'file';
  name: string;
  qualifier?: string | number;
  var_name?: string; // Present when qualifier is a loc_id (number)
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
 * Attributes that depend on column DATA values (not just structure).
 * Col(d, c) writes invalidate these.
 */
const COL_VALUE_ATTRS = new Set([
  'values', // df.values — 2D array of all column data
  'T', // df.T — transpose exposes all column data
  'describe' // df.describe() — statistics computed from column values
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
 * Get the display variable name from a qualifier.
 * If qualifier is a loc_id (number), use var_name. Otherwise use qualifier directly.
 */
function _displayQualifier(loc: {
  qualifier?: string | number;
  var_name?: string;
}): string | undefined {
  if (loc.var_name !== undefined) {
    return loc.var_name;
  }
  if (typeof loc.qualifier === 'string') {
    return loc.qualifier;
  }
  return undefined;
}

/**
 * Compare two DataFrame identifiers (qualifiers).
 * If both are numbers (loc_ids from StableIdMap), compare by identity.
 * Otherwise fall back to variable name comparison.
 */
function _sameDataframe(
  a: { qualifier?: string | number; var_name?: string },
  b: { qualifier?: string | number; var_name?: string }
): boolean {
  // Both have numeric qualifiers (loc_ids) → compare by object identity
  if (typeof a.qualifier === 'number' && typeof b.qualifier === 'number') {
    return a.qualifier === b.qualifier;
  }
  // Fall back to var name comparison
  const aName = _displayQualifier(a);
  const bName = _displayQualifier(b);
  return aName === bName;
}

/**
 * Does Var(x) target the DataFrame accessed through ref?
 * Var rebinding uses variable name matching, not object identity.
 */
function _varTargetsRef(
  varName: string,
  ref: { qualifier?: string | number; var_name?: string }
): boolean {
  const refName = _displayQualifier(ref);
  return varName === refName;
}

/**
 * Does write `w` invalidate read `r`?
 *
 * This is the ▷ conflict relation — the single 7×4 function that determines
 * all staleness in the system. Port of Python write_conflicts_read().
 *
 * Uses _sameDataframe() for DataFrame-to-DataFrame qualifier comparison
 * (compares loc_ids when available) and _varTargetsRef() for Var-to-DataFrame
 * comparison (compares variable names).
 */
export function writeConflictsRead(w: IWriteLoc, r: IReadLoc): boolean {
  switch (w.type) {
    case 'var':
      // Var(x) write conflicts with any read on the same variable
      if (r.type === 'var') {
        return w.name === r.name;
      }
      if (r.type === 'file') {
        return false;
      }
      // Col(d,c) or Attr(d,a): conflicts if x targets same DataFrame via var name
      return _varTargetsRef(w.name, r);

    case 'col':
      // Col(d,c) conflicts with Col(d,c) AND Attr(d, a) for a ∈ COL_ATTRS
      if (r.type === 'col') {
        return _sameDataframe(w, r) && w.name === r.name;
      }
      if (r.type === 'attr') {
        return _sameDataframe(w, r) && COL_ATTRS.has(r.name);
      }
      return false;

    case 'col_add':
      // ColAdd(d,c) only conflicts with Attr reads on COL_ATTRS
      return r.type === 'attr' && _sameDataframe(w, r) && COL_ATTRS.has(r.name);

    case 'col_del':
      // ColDel(d,c) conflicts with Col(d,c) AND Attr(d, COL_ATTRS)
      if (r.type === 'col') {
        return _sameDataframe(w, r) && w.name === r.name;
      }
      return r.type === 'attr' && _sameDataframe(w, r) && COL_ATTRS.has(r.name);

    case 'rows':
      // Rows(d) conflicts with all Col(d,*) AND Attr(d, ROW_ATTRS)
      if (r.type === 'col') {
        return _sameDataframe(w, r);
      }
      if (r.type === 'attr') {
        return _sameDataframe(w, r) && ROW_ATTRS.has(r.name);
      }
      return false;

    case 'attr':
      // Attr(d,a) only conflicts with Attr(d,a) — same dataframe AND same attr
      return r.type === 'attr' && _sameDataframe(w, r) && w.name === r.name;

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
export function hasConflict(wlocs: IWriteLoc[], rlocs: IReadLoc[]): boolean {
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
  const q = _displayQualifier(loc);
  if (q) {
    return `${q}.${loc.name}`;
  }
  return loc.name;
}

/**
 * Format a WriteLoc for display with type annotation.
 */
export function formatWriteLoc(loc: IWriteLoc): string {
  const q = _displayQualifier(loc);
  if (q) {
    return `${q}.${loc.name}`;
  }
  return loc.name;
}

/**
 * Compare two ReadLoc qualifiers for equality.
 * Uses _sameDataframe() logic for proper loc_id comparison.
 */
export function readLocsMatchQualifier(a: IReadLoc, b: IReadLoc): boolean {
  // Both have numeric qualifiers → compare loc_ids
  if (typeof a.qualifier === 'number' && typeof b.qualifier === 'number') {
    return a.qualifier === b.qualifier;
  }
  // Fall back to display name comparison
  const aName = _displayQualifier(a);
  const bName = _displayQualifier(b);
  return aName === bName;
}

/**
 * Map a WriteLoc to the ReadLocs that would observe its effect.
 * Returns multiple locs for structural writes (ColAdd, ColDel, Rows).
 * Used for write-write overlap detection via ▷.
 */
export function writeLocOutputs(w: IWriteLoc): IReadLoc[] {
  switch (w.type) {
    case 'var':
      return [{ type: 'var', name: w.name }];
    case 'col':
      return [
        { type: 'col', name: w.name, qualifier: w.qualifier },
        ...[...COL_VALUE_ATTRS].map(a => ({
          type: 'attr' as const,
          name: a,
          qualifier: w.qualifier
        }))
      ];
    case 'col_add':
      // ColAdd conflicts with Attr(d, a) for a ∈ COL_ATTRS
      return [...COL_ATTRS].map(a => ({
        type: 'attr' as const,
        name: a,
        qualifier: w.qualifier
      }));
    case 'col_del':
      // ColDel conflicts with Col(d, c) and Attr(d, a) for a ∈ COL_ATTRS
      return [
        { type: 'col', name: w.name, qualifier: w.qualifier },
        ...[...COL_ATTRS].map(a => ({
          type: 'attr' as const,
          name: a,
          qualifier: w.qualifier
        }))
      ];
    case 'rows':
      // Rows conflicts with Attr(d, a) for a ∈ ROW_ATTRS
      // qualifier holds the DataFrame identifier; propagate var_name for display
      return [...ROW_ATTRS].map(a => ({
        type: 'attr' as const,
        name: a,
        qualifier: w.qualifier ?? w.name,
        var_name: w.var_name
      }));
    case 'attr':
      return [{ type: 'attr', name: w.name, qualifier: w.qualifier }];
    case 'file':
      return [{ type: 'file', name: w.name }];
    default:
      return [{ type: 'var', name: w.name }];
  }
}
