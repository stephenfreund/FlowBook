/**
 * Manages violation notice outputs in cell output areas.
 *
 * Extracted from CellHighlighter to separate output-management concerns
 * from CSS highlighting.
 */

import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { IReproducibilityError, IReproducibilityMetadata, asFlowbookOutput } from './types';
import { indexToAlpha } from '../cellindexutils';

/**
 * Manages violation notice display_data outputs in cell output areas.
 */
export class ViolationNoticeManager {
  /**
   * Add or remove the violation notice display_data output.
   * Reads errors from flowbook metadata (the canonical source of violation data).
   *
   * Returns true if the cell has violations (so caller can add error CSS class).
   */
  updateViolationNotice(cell: Cell, cellOrder: string[]): boolean {
    if (cell.model.type !== 'code') {
      return false;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    const flowbookMeta = cell.model.getMetadata('flowbook') as
      | IReproducibilityMetadata
      | undefined;
    const violations = flowbookMeta?.errors;

    // Check if we already have a violation notice
    let hasViolationNotice = false;
    let existingPlainText = '';
    for (let i = 0; i < outputs.length; i++) {
      const out = asFlowbookOutput(outputs.get(i).toJSON());
      if (out.metadata?.flowbook_violation_notice === true) {
        hasViolationNotice = true;
        existingPlainText = out.data?.['text/plain'] || '';
        break;
      }
    }

    if (violations && violations.length > 0) {
      const { noticeOutput, plainText } = this._buildViolationNotice(
        violations,
        cellOrder
      );

      // Check if message matches current notice
      if (hasViolationNotice && existingPlainText === plainText) {
        return true; // Already up to date, but still has violations
      }

      // Build new output array (also remove staleness notices — violation is more specific)
      const allOutputs: IOutput[] = [noticeOutput];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const fbOut = asFlowbookOutput(out);
        const isViolationNotice =
          fbOut.metadata?.flowbook_violation_notice === true;
        const isStalenessNotice =
          fbOut.metadata?.flowbook_staleness_notice === true;
        const isKernelError =
          out.output_type === 'error' &&
          (fbOut.ename === 'ReproducibilityError' ||
            fbOut.ename === 'ReproducibilityViolation');
        const isKernelPredicateViolation =
          out.output_type === 'display_data' &&
          fbOut.metadata?.predicate_violation;

        if (
          !isViolationNotice &&
          !isStalenessNotice &&
          !isKernelError &&
          !isKernelPredicateViolation
        ) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
      return true;
    }

    // No violations — clear any existing violation notices
    if (hasViolationNotice) {
      const allOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const fbOut = asFlowbookOutput(out);
        if (!fbOut.metadata?.flowbook_violation_notice) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
    }
    return false;
  }

  /**
   * Build the violation notice output, grouping violations by (predicate, locations).
   */
  private _buildViolationNotice(
    violations: IReproducibilityError[],
    cellOrder: string[]
  ): { noticeOutput: IOutput; plainText: string } {
    const icon = '\u274c';
    const cssClass = 'flowbook-error-notice';

    // Group violations by (error_type, locations) to merge common ones
    const grouped = new Map<
      string,
      { errorType: string; locs: string[]; causers: string[] }
    >();

    for (const violation of violations) {
      let causerRef: string | null = null;
      if (violation.causer_cell && typeof violation.causer_cell === 'string') {
        const rawCauser = violation.causer_cell.startsWith('@')
          ? violation.causer_cell.slice(1)
          : violation.causer_cell;
        const causerIdx = cellOrder.indexOf(rawCauser);
        causerRef = causerIdx >= 0 ? indexToAlpha(causerIdx) : 'a deleted cell';
      }

      const locsKey = [...violation.locations].sort().join(',');
      const groupKey = `${violation.error_type}:${locsKey}`;

      if (!grouped.has(groupKey)) {
        grouped.set(groupKey, {
          errorType: violation.error_type,
          locs: violation.locations,
          causers: []
        });
      }

      if (causerRef && !grouped.get(groupKey)!.causers.includes(causerRef)) {
        grouped.get(groupKey)!.causers.push(causerRef);
      }
    }

    // Build messages from grouped violations
    const htmlMessages: string[] = [];
    const plainMessages: string[] = [];

    for (const group of grouped.values()) {
      const locs = group.locs.map(l => '`' + l + '`').join(', ');
      const htmlLocs = locs.replace(/`([^`]+)`/g, '<code>$1</code>');
      const causersStr = group.causers.join(', ');

      let message: string;
      switch (group.errorType) {
        case 'no_write_after_read': {
          message = causersStr
            ? `Writes ${htmlLocs} already read by ${causersStr}`
            : `Writes ${htmlLocs} already read by cell above`;
          for (const loc of group.locs) {
            if (loc.includes('.')) {
              const [dfName, colName] = loc.split('.');
              message += `<br>Use <code>${dfName}["${colName}"]</code> = ... for full-column assignment`;
            }
          }
          break;
        }
        case 'no_read_before_write':
          message = causersStr
            ? `Reads ${htmlLocs} written by ${causersStr} below`
            : `Reads ${htmlLocs} written by cell below`;
          break;
        case 'no_read_and_write':
          message = `Reads and writes ${htmlLocs}`;
          break;
        case 'write_before_read':
          message = `${htmlLocs} not defined by any cell above`;
          break;
        default:
          message = `Violation on ${htmlLocs}`;
      }

      htmlMessages.push(message);
      plainMessages.push(message.replace(/<code>([^<]+)<\/code>/g, '`$1`'));
    }

    const combinedHtml = htmlMessages
      .map(m => `<div>${icon} ${m}</div>`)
      .join('');
    const plainText = plainMessages.map(m => `${icon} ${m}`).join('\n');

    const noticeOutput: IOutput = {
      output_type: 'display_data',
      data: {
        'text/html': `<div class="${cssClass}">${combinedHtml}</div>`,
        'text/plain': plainText
      },
      metadata: {
        flowbook_violation_notice: true,
        flowbook_predicate_accepted: violations[0].accepted,
        flowbook_violation_count: violations.length
      }
    };

    return { noticeOutput, plainText };
  }
}
