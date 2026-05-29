/**
 * FlowBook communication protocol types.
 *
 * Defines the unified JSON message format for kernel <-> client communication.
 * All messages use a "type" discriminator field.
 *
 * Transport:
 * - Frontend <-> Kernel: Comm channel ("flowbook" target)
 * - Python clients <-> Kernel: Execute request metadata / custom IOPub
 */

import { IReproducibilityMetadata, IPredicateViolation } from './types';

// ===========================================================================
// Constants
// ===========================================================================

export const COMM_TARGET = 'flowbook';

// ===========================================================================
// Kernel -> Client messages
// ===========================================================================

/**
 * Metadata message — post-execution reproducibility data.
 * Replaces display_data with metadata.flowbook.
 */
export interface IMetadataMessage extends IReproducibilityMetadata {
  type: 'metadata';
  /**
   * Who drove the execution that produced this metadata: 'ai' for an LLM
   * (MCP / NBI tool call / fix agent), 'user' otherwise. Used by the LogBook
   * bridge to attribute out-of-process AI activity. Absent when unknown.
   */
  actor?: 'user' | 'ai';
}

/**
 * Violation message — predicate violation.
 * Replaces display_data with metadata.predicate_violation.
 */
export interface IViolationMessage extends IPredicateViolation {
  type: 'violation';
}

/**
 * Status message — icon + text line.
 * Replaces display_icon_and_text() calls with metadata payload.
 */
export interface IStatusMessage {
  type: 'status';
  icon: string;
  text: string;
  cell_id: string; // Cell that produced this status (for @A display)
}

/**
 * Discriminated union of all kernel -> client messages.
 */
export type FlowbookKernelMessage =
  | IMetadataMessage
  | IViolationMessage
  | IStatusMessage;

// ===========================================================================
// Client -> Kernel messages
// ===========================================================================

export interface INotebookStructureMessage {
  type: 'notebook_structure';
  cell_order: string[];
}

export interface ICellEditedMessage {
  type: 'cell_edited';
  cell_id: string;
}

export interface IContinueAfterViolationMessage {
  type: 'continue_after_violation';
  enabled: boolean;
}

export interface ISyncMessage {
  type: 'sync';
}

export interface IExecRestoreMessage {
  type: 'exec_restore';
  cell_id: string;
}

/**
 * Discriminated union of all client -> kernel messages.
 */
export type FlowbookClientMessage =
  | INotebookStructureMessage
  | ICellEditedMessage
  | IContinueAfterViolationMessage
  | ISyncMessage
  | IExecRestoreMessage;
