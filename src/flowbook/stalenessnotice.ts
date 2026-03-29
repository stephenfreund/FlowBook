/**
 * Manages staleness notice outputs in cell output areas.
 *
 * Extracted from CellHighlighter to separate output-management concerns
 * from CSS highlighting.
 */

import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { StalenessManager } from './stalenessmanager';
import { IStalenessReason, IFrontendStalenessReason } from './types';
import { indexToAlpha } from '../cellindexutils';

/**
 * Type guard to check if a staleness reason is a frontend reason with message.
 */
function isFrontendReason(
  reason: IStalenessReason
): reason is IFrontendStalenessReason {
  return 'message' in reason;
}

/**
 * Manages staleness notice display_data outputs in cell output areas.
 */
export class StalenessNoticeManager {
  /**
   * Add or remove the staleness notice display_data output at index 0.
   */
  updateStalenessNotice(
    cell: Cell,
    isStale: boolean,
    stalenessManager: StalenessManager,
    cellOrder: string[]
  ): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Check if there's a violation (either in metadata or existing notice)
    // Violation implies specific issue, so skip staleness notice
    const hasViolationMetadata =
      cell.model.getMetadata('flowbook_violation') !== undefined;
    let hasViolationNotice = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as any;
      if (out.metadata?.flowbook_violation_notice === true) {
        hasViolationNotice = true;
        break;
      }
    }

    // Remove staleness notice if violation is present (it's more specific)
    if (hasViolationMetadata || hasViolationNotice) {
      this._removeNoticesByKey(outputs, 'flowbook_staleness_notice');
      return;
    }

    // Check if first output is already a staleness notice
    const hasNotice =
      outputs.length > 0 &&
      (outputs.get(0).toJSON() as any).metadata?.flowbook_staleness_notice ===
        true;

    if (isStale) {
      const reason = stalenessManager.getReason(cell.model.id) || {
        type: 'unknown',
        message: 'Dependencies changed'
      };

      // Don't display notice for never_executed cells
      if (reason.type === 'never_executed') {
        if (hasNotice) {
          this._removeNoticesByKey(outputs, 'flowbook_staleness_notice');
        }
        return;
      }

      const message = this.formatStalenessMessage(
        reason,
        cellOrder,
        cell.model.id
      );

      // Escape HTML in the message but preserve backtick-wrapped code
      const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Use different label for writer_conflict (potential violation vs stale dependency)
      const isWriterConflict = reason.type === 'writer_conflict';
      const label = isWriterConflict ? 'Unresolved Violation' : '';
      const plainText = label
        ? `\u26a0\ufe0f ${label}: ${message}`
        : `\u26a0\ufe0f ${message}`;

      const stalenessOutput: IOutput = {
        output_type: 'display_data',
        data: {
          'text/html': label
            ? `<div class="flowbook-staleness-notice">\u26a0\ufe0f <b>${label}</b>: ${htmlMessage} </div>`
            : `<div class="flowbook-staleness-notice">\u26a0\ufe0f ${htmlMessage} </div>`,
          'text/plain': plainText
        },
        metadata: { flowbook_staleness_notice: true }
      };

      if (hasNotice) {
        // Check if message matches current notice
        const existingPlain = (outputs.get(0).toJSON() as any).data?.[
          'text/plain'
        ];
        if (existingPlain === plainText) {
          return; // Already up to date
        }
      }

      // Build new output array: [notice, ...existing non-notice outputs]
      const allOutputs: IOutput[] = [stalenessOutput];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        if (!(out as any).metadata?.flowbook_staleness_notice) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
    } else if (hasNotice) {
      this._removeNoticesByKey(outputs, 'flowbook_staleness_notice');
    }
  }

  /**
   * Format staleness message with dynamic @A references from current cell order.
   */
  formatStalenessMessage(
    reason: IStalenessReason,
    cellOrder: string[],
    currentCellId: string
  ): string {
    if (isFrontendReason(reason)) {
      return this._formatFrontendReason(reason, cellOrder, currentCellId);
    }
    return this._formatBackendReason(reason, cellOrder, currentCellId);
  }

  /**
   * Recompute the flowbook_staleness metadata message with current @A references.
   */
  updateStalenessMetadata(cell: Cell, cellOrder: string[]): void {
    const staleness = cell.model.getMetadata('flowbook_staleness') as
      | IStalenessReason
      | undefined;
    if (!staleness) {
      return;
    }

    if (!isFrontendReason(staleness) || !staleness.causing_cell) {
      return;
    }

    const newMessage = this.formatStalenessMessage(
      staleness,
      cellOrder,
      cell.model.id
    );
    if (newMessage !== staleness.message) {
      cell.model.setMetadata('flowbook_staleness', {
        ...staleness,
        message: newMessage
      });
    }
  }

  private _formatFrontendReason(
    reason: IFrontendStalenessReason,
    cellOrder: string[],
    currentCellId: string
  ): string {
    if (reason.type === 'source_edited') {
      return 'Source code was edited';
    }

    if (!reason.causing_cell) {
      return reason.message;
    }

    const causingIdx = cellOrder.indexOf(reason.causing_cell);
    const currentIdx = cellOrder.indexOf(currentCellId);
    const isDeleted = causingIdx < 0;
    const causingRef = isDeleted ? 'a deleted cell' : indexToAlpha(causingIdx);
    const direction =
      !isDeleted && currentIdx >= 0 && causingIdx < currentIdx
        ? ' above'
        : !isDeleted
          ? ' below'
          : '';

    const parts: string[] = [];
    if (reason.variables) {
      for (const v of reason.variables) {
        parts.push('`' + v + '`');
      }
    }
    if (reason.columns) {
      for (const [dfName, cols] of Object.entries(reason.columns)) {
        for (const col of cols) {
          parts.push('`' + dfName + '.' + col + '`');
        }
      }
    }

    if (reason.type === 'writer_conflict' && parts.length > 0) {
      return `Writes ${parts.join(', ')} already read by ${causingRef}${direction}`;
    }

    if (parts.length > 0) {
      return `${parts.join(', ')} modified by ${causingRef}${direction}`;
    }

    if (reason.type === 'unknown') {
      return `Dependencies modified by ${causingRef}`;
    }

    return reason.message;
  }

  private _formatBackendReason(
    reason: IStalenessReason,
    cellOrder: string[],
    currentCellId: string
  ): string {
    const cellId = 'cell_id' in reason ? reason.cell_id : undefined;
    const loc = 'loc' in reason ? reason.loc : undefined;
    const currentIdx = cellOrder.indexOf(currentCellId);

    let causingRef = '';
    let causingDirection = '';
    let causingIsDeleted = false;
    if (cellId) {
      const causingIdx = cellOrder.indexOf(cellId);
      causingIsDeleted = causingIdx < 0;
      causingRef = causingIsDeleted
        ? 'a deleted cell'
        : indexToAlpha(causingIdx);
      if (!causingIsDeleted && currentIdx >= 0) {
        causingDirection = causingIdx < currentIdx ? ' above' : ' below';
      }
    }

    switch (reason.type) {
      case 'never_executed':
        return 'Cell has never been executed';
      case 'code_changed':
        return 'Source code was edited';
      case 'forward_stale':
        if (loc && causingRef) {
          return `\`${loc}\` modified by ${causingRef}${causingDirection}`;
        }
        return causingRef
          ? `Input modified by ${causingRef}${causingDirection}`
          : 'Input was modified';
      case 'write_overlap':
        if (loc && causingRef) {
          return `\`${loc}\` also written by ${causingRef}`;
        }
        return causingRef
          ? `Writes conflict with ${causingRef}`
          : 'Write conflict detected';
      case 'backward_stale':
        if (loc && causingRef) {
          return `\`${loc}\` write conflict with ${causingRef}`;
        }
        return 'Write conflict detected';
      case 'no_read_before_write':
        if (loc && causingRef) {
          return `Reads \`${loc}\` written by ${causingRef} ${causingDirection}`;
        }
        return 'Reads value written by another cell';
      case 'order_changed':
        return 'Cell order changed';
      case 'no_write_after_read':
        if (loc && causingRef) {
          return `Writes \`${loc}\` already read by ${causingRef} ${causingDirection}`;
        }
        return causingRef
          ? `Writes variable already read by ${causingRef} ${causingDirection}`
          : 'Writes variable already read by another cell';
      default:
        return 'Cell is stale';
    }
  }

  /**
   * Remove all outputs with a given metadata key set to true.
   */
  private _removeNoticesByKey(
    outputs: ICodeCellModel['outputs'],
    key: string
  ): void {
    const allOutputs: IOutput[] = [];
    let removed = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as IOutput;
      if ((out as any).metadata?.[key]) {
        removed = true;
      } else {
        allOutputs.push(out);
      }
    }
    if (removed) {
      outputs.fromJSON(allOutputs);
    }
  }
}
