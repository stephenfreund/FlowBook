/**
 * Type definitions for SDC kernel extension
 */

export interface ISDCViolation {
  mutating_cell: string;
  affected_cell: string;
  variables: string[];
  message: string;
}

export interface ISDCMetadata {
  cell_id: string;
  execution_seq: number;
  reads: string[];
  writes: string[];
  changed_variables: string[];
  stale_cells: string[];
  violation: ISDCViolation | null;
  cell_order: string[];
  column_reads?: { [key: string]: string[] };
  column_writes?: { [key: string]: string[] };
  column_changed?: { [key: string]: string[] };
  structural_reads?: { [key: string]: string[] };
  structural_warnings?: string[];
  // Timing information (in milliseconds)
  run_duration_ms?: number;
  state_duration_ms?: number;
  check_duration_ms?: number;
}

export interface ISDCCellState {
  cellId: string;
  executionSeq: number;
  reads: string[];
  writes: string[];
  isStale: boolean;
}
