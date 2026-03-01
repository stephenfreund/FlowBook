/**
 * Type definitions for FlowBook kernel extension (reproducibility)
 */

export interface IReproducibilityViolation {
  mutating_cell: string;
  affected_cell: string;
  variables: string[];
  message: string;
  violation_type?:
    | 'backward_mutation'
    | 'forward_dependency'
    | 'deleted_cell_dependency';
}

export interface IReproducibilityMetadata {
  cell_id: string;
  execution_seq: number;
  reads: string[];
  writes: string[];
  changed_variables: string[];
  stale_cells: string[];
  violation: IReproducibilityViolation | null;
  cell_order: string[];
  column_reads?: { [key: string]: string[] };
  column_writes?: { [key: string]: string[] };
  column_changed?: { [key: string]: string[] };
  structural_reads?: { [key: string]: string[] };
  structural_warnings?: string[];
  // File I/O tracking
  file_reads?: string[];
  file_writes?: string[];
  // Timing information (in milliseconds)
  execute_duration_ms?: number; // Total time in _do_execute_impl
  code_duration_ms?: number; // Time for _ipython_do_execute (user code)
  state_duration_ms?: number;
  check_duration_ms?: number;
  // Writer violation: backward_mutation violation to store on writer cell (for forward contamination)
  writer_violation?: IReproducibilityViolation;
  // Staleness reasons per cell: { cell_id: [reason, ...] }
  staleness_reasons?: { [cell_id: string]: IBackendStalenessReason[] };
  // Whether this cell is contaminated (reads from later cell)
  cell_is_contaminated?: boolean;
  // Proposed fix for violations
  proposed_fix?: IProposedFix;
}

export interface IReproducibilityCellState {
  cellId: string;
  executionSeq: number;
  reads: string[];
  writes: string[];
  isStale: boolean;
}

/**
 * Reason types from the formal model (§1.2).
 * Maps to ReasonType enum in flowbook/kernel/models.py
 */
export type BackendReasonType =
  | 'never_executed' // Cell has never been run
  | 'code_changed' // Cell source code was edited
  | 'input_changed' // A variable this cell reads was modified by another cell
  | 'write_conflict' // Another cell wrote to a variable this cell also writes
  | 'reads_from_later' // Cell reads a value written by a later cell (forward contamination)
  | 'source_deleted' // The cell that wrote a variable this cell reads was deleted
  | 'order_changed' // Cell order changed affecting data flow
  | 'skipped_upstream'; // Cell reads from wrong writer; re-running won't help, run expected cell first

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
  expected_cell_id?: string; // For skipped writer: cell that should have provided the value
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

export interface IViolationInfo {
  type: string; // "backward_mutation" | "forward_dependency" | "truncation"
  mutating_cell: string; // actual cell ID
  affected_cell: string; // actual cell ID
  variables: string[];
  message: string; // human-readable (@A notation)
  // Detailed diagnostic info for enhanced messages
  structural_reads_detail?: { [key: string]: { [key: string]: string } }; // var -> {attr -> value_repr}
  changes_detail?: string[]; // ["Column 'y' added", "Shape: (5,4) → (5,5)"]
}

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
