/**
 * Manages staleness notice outputs in cell output areas.
 *
 * Extracted from CellHighlighter to separate output-management concerns
 * from CSS highlighting.
 */

import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { StalenessManager } from './stalenessmanager';
import {
  IStalenessReason,
  IFrontendStalenessReason,
  IReproducibilityMetadata,
  asFlowbookOutput,
  escapeHtml
} from './types';
import { indexToAlpha } from '../cellindexutils';
import { formatBackendReasonMessage } from './reasonformat';

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
    const flowbookMeta = cell.model.getMetadata('flowbook') as
      | IReproducibilityMetadata
      | undefined;
    const hasViolationMetadata =
      flowbookMeta?.errors && flowbookMeta.errors.length > 0;
    let hasViolationNotice = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = asFlowbookOutput(outputs.get(i).toJSON());
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
      asFlowbookOutput(outputs.get(0).toJSON()).metadata
        ?.flowbook_staleness_notice === true;

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

      // Escape HTML in the message (variable/column names come from user
      // data), then convert backtick-wrapped code to <code> — backticks
      // are not HTML-special, so they survive the escaping.
      const htmlMessage = escapeHtml(message).replace(
        /`([^`]+)`/g,
        '<code>$1</code>'
      );

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
        const existingPlain = asFlowbookOutput(outputs.get(0).toJSON()).data?.[
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
        if (!asFlowbookOutput(out).metadata?.flowbook_staleness_notice) {
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
    return formatBackendReasonMessage(reason, cellOrder, currentCellId);
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
      if (asFlowbookOutput(out).metadata?.[key]) {
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
