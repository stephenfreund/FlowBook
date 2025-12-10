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
}

export interface ISDCCellState {
  cellId: string;
  executionSeq: number;
  reads: string[];
  writes: string[];
  isStale: boolean;
}
