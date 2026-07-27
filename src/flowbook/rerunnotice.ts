/**
 * Manages rerun warning notice outputs in cell output areas.
 *
 * The run-until-clean loop shows this orange notice under a cell whose
 * rerun marked an earlier cell stale (unstable read/write footprint).
 * Unlike the staleness and violation notices, it reflects sweep state
 * rather than kernel state, so it is owned by the loop (toolbar / NBI
 * bridge) instead of CellHighlighter: cleared for every cell at sweep
 * start, updated as reports fire, and left in place when the sweep ends
 * as the visible trace of the instability. It is stripped from saved
 * notebooks like the other notices (see plugin.ts noticeKeys).
 */

import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { NotebookPanel } from '@jupyterlab/notebook';
import { IOutput } from '@jupyterlab/nbformat';
import { asFlowbookOutput, escapeHtml } from './types';

export const RERUN_WARNING_FLAG = 'flowbook_rerun_warning_notice';

/**
 * Manages rerun warning display_data outputs in cell output areas.
 */
export class RerunWarningNoticeManager {
  /**
   * Insert or update the orange warning notice at output index 0.
   */
  showWarning(cell: Cell, message: string): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const outputs = (cell.model as ICodeCellModel).outputs;

    const htmlMessage = escapeHtml(message).replace(
      /`([^`]+)`/g,
      '<code>$1</code>'
    );
    const plainText = `\u26a0\ufe0f Unstable footprint: ${message}`;
    const noticeOutput: IOutput = {
      output_type: 'display_data',
      data: {
        'text/html': `<div class="flowbook-rerun-warning-notice">\u26a0\ufe0f <b>Unstable footprint</b>: ${htmlMessage}</div>`,
        'text/plain': plainText
      },
      metadata: { [RERUN_WARNING_FLAG]: true }
    };

    // Skip the rewrite when the current notice already says this
    const existing = this._findNotice(outputs);
    if (existing !== null) {
      const existingPlain = asFlowbookOutput(outputs.get(existing).toJSON())
        .data?.['text/plain'];
      if (existingPlain === plainText) {
        return;
      }
    }

    const allOutputs: IOutput[] = [noticeOutput];
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as IOutput;
      if (!asFlowbookOutput(out).metadata?.[RERUN_WARNING_FLAG]) {
        allOutputs.push(out);
      }
    }
    outputs.fromJSON(allOutputs);
  }

  /**
   * Remove the warning notice from one cell, preserving other outputs.
   */
  clearWarning(cell: Cell): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const outputs = (cell.model as ICodeCellModel).outputs;
    if (this._findNotice(outputs) === null) {
      return;
    }
    const allOutputs: IOutput[] = [];
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as IOutput;
      if (!asFlowbookOutput(out).metadata?.[RERUN_WARNING_FLAG]) {
        allOutputs.push(out);
      }
    }
    outputs.fromJSON(allOutputs);
  }

  /**
   * Remove the warning notice from every cell (sweep start).
   */
  clearAll(panel: NotebookPanel): void {
    for (const widget of panel.content.widgets) {
      this.clearWarning(widget as Cell);
    }
  }

  private _findNotice(outputs: ICodeCellModel['outputs']): number | null {
    for (let i = 0; i < outputs.length; i++) {
      const out = asFlowbookOutput(outputs.get(i).toJSON());
      if (out.metadata?.[RERUN_WARNING_FLAG] === true) {
        return i;
      }
    }
    return null;
  }
}
