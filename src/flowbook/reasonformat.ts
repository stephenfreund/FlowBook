/**
 * Canonical formatting of backend staleness reasons.
 *
 * Single source of truth for mapping backend reason types
 * (BackendReasonType) to human-readable messages and frontend reason
 * structures. Both executionhook.ts (which needs an
 * IFrontendStalenessReason to store on cells) and stalenessnotice.ts
 * (which needs a display string) build on the one table below via thin
 * view functions.
 */

import {
  FrontendReasonType,
  IBackendStalenessReason,
  IFrontendStalenessReason
} from './types';
import { indexToAlpha } from '../cellindexutils';

/**
 * How the causing cell should be referenced in messages.
 */
interface ICausingRef {
  // '@A'-style alpha reference, or 'a deleted cell' when the causing cell
  // is no longer in the notebook. Empty when the reason names no cell.
  ref: string;
  // ' above' | ' below' | '' — relative position of the causing cell to
  // the stale cell. Empty when unknown (no currentCellId, deleted cell).
  direction: string;
}

/**
 * Resolve a causing cell id to a display reference + direction suffix.
 */
function resolveCausingRef(
  causingCellId: string | undefined,
  cellOrder: string[],
  currentCellId?: string
): ICausingRef {
  if (!causingCellId) {
    return { ref: '', direction: '' };
  }
  const causingIdx = cellOrder.indexOf(causingCellId);
  if (causingIdx < 0) {
    return { ref: 'a deleted cell', direction: '' };
  }
  const ref = indexToAlpha(causingIdx);
  let direction = '';
  if (currentCellId) {
    const currentIdx = cellOrder.indexOf(currentCellId);
    if (currentIdx >= 0) {
      direction = causingIdx < currentIdx ? ' above' : ' below';
    }
  }
  return { ref, direction };
}

/**
 * The canonical view of a backend reason: frontend type, message, and the
 * locations involved. Both public views derive from this.
 */
interface IReasonView {
  type: FrontendReasonType;
  message: string;
  variables?: string[];
}

/**
 * The one table: backend reason type -> frontend type + message.
 */
function buildReasonView(
  reason: IBackendStalenessReason,
  cellOrder: string[],
  currentCellId?: string
): IReasonView {
  const loc = reason.loc;
  const { ref, direction } = resolveCausingRef(
    reason.cell_id,
    cellOrder,
    currentCellId
  );
  const where = `${ref}${direction}`;

  switch (reason.type) {
    case 'never_executed':
      return { type: 'unknown', message: 'Cell has never been executed' };

    case 'code_changed':
      return { type: 'source_edited', message: 'Source code was edited' };

    case 'forward_stale':
      // ForwardStale: an input this cell reads was modified.
      if (loc && ref) {
        return {
          type: 'variable_modified',
          variables: [loc],
          message: `\`${loc}\` modified by ${where}`
        };
      }
      return {
        type: 'variable_modified',
        message: ref ? `Input modified by ${where}` : 'Input was modified'
      };

    case 'write_overlap':
      // Cell writes a location that another cell also writes.
      if (loc && ref) {
        return {
          type: 'writer_conflict',
          variables: [loc],
          message: `\`${loc}\` also written by ${where}`
        };
      }
      return {
        type: 'writer_conflict',
        message: ref
          ? `Writes conflict with ${where}`
          : 'Write conflict detected'
      };

    case 'backward_stale':
      if (loc && ref) {
        return {
          type: 'writer_conflict',
          variables: [loc],
          message: `Write conflict on \`${loc}\` with ${where}`
        };
      }
      return { type: 'writer_conflict', message: 'Write conflict detected' };

    case 'no_read_before_write':
      // NoReadBeforeWrite failed — reads from a later cell (forward
      // contamination).
      if (loc && ref) {
        return {
          type: 'unknown',
          message: `Reads \`${loc}\` written by ${where}`
        };
      }
      return {
        type: 'unknown',
        message: 'Reads a value written by a later cell'
      };

    case 'order_changed':
      return { type: 'unknown', message: 'Cell order changed' };

    case 'no_write_after_read':
      // NoWriteAfterRead failed — wrote a location read by an earlier cell
      // (backward mutation).
      if (loc && ref) {
        return {
          type: 'variable_modified',
          variables: [loc],
          message: `Writes \`${loc}\` already read by ${where}`
        };
      }
      return {
        type: 'unknown',
        message: ref
          ? `Writes a variable already read by ${where}`
          : 'Writes a variable already read by another cell'
      };

    default:
      return {
        type: 'unknown',
        message: ref ? `Dependencies changed by ${where}` : 'Cell is stale'
      };
  }
}

/**
 * View 1: convert a backend staleness reason to the frontend reason
 * structure stored on cells and in the staleness manager.
 */
export function backendReasonToFrontend(
  reason: IBackendStalenessReason,
  cellOrder: string[],
  currentCellId?: string
): IFrontendStalenessReason {
  const view = buildReasonView(reason, cellOrder, currentCellId);
  const result: IFrontendStalenessReason = {
    type: view.type,
    message: view.message
  };
  if (reason.cell_id) {
    result.causing_cell = reason.cell_id;
  }
  if (view.variables) {
    result.variables = view.variables;
  }
  return result;
}

/**
 * View 2: format a backend staleness reason as a display string for the
 * staleness notice.
 */
export function formatBackendReasonMessage(
  reason: IBackendStalenessReason,
  cellOrder: string[],
  currentCellId: string
): string {
  return buildReasonView(reason, cellOrder, currentCellId).message;
}
