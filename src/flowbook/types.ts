/**
 * Type definitions for FlowBook kernel extension (reproducibility)
 */

export interface IReproducibilityViolation {
  mutating_cell: string;
  affected_cell: string;
  variables: string[];
  message: string;
  violation_type?: 'backward_mutation' | 'forward_dependency';
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
  run_duration_ms?: number;
  state_duration_ms?: number;
  check_duration_ms?: number;
  // Forward contamination flag (EXEC-CONTAMINATED)
  cell_is_contaminated?: boolean;
  // Execution mode (EXEC-RESTORE)
  exec_mode?: 'live' | 'restore';
}

export interface IReproducibilityCellState {
  cellId: string;
  executionSeq: number;
  reads: string[];
  writes: string[];
  isStale: boolean;
}

export interface IStalenessReason {
  type: string; // "variable_modified" | "source_edited" | "contaminated" | "writer_conflict" | "unknown"
  causing_cell?: string; // actual cell ID
  variables?: string[];
  columns?: { [key: string]: string[] };
  message: string; // human-readable (@A notation)
}

export interface IViolationInfo {
  type: string; // "backward_mutation" | "forward_dependency" | "truncation"
  mutating_cell: string; // actual cell ID
  affected_cell: string; // actual cell ID
  variables: string[];
  message: string; // human-readable (@A notation)
}
