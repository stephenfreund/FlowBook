/**
 * Notebook toolbar extension for FlowBook - Run All Stale/Unrun button
 */

import { NotebookPanel, NotebookActions } from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { IDisposable } from '@lumino/disposable';
import { showErrorMessage, ToolbarButton } from '@jupyterlab/apputils';
import { stepIntoIcon } from '@jupyterlab/ui-components';

import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { IReproducibilityMetadata, waitForFlowbookMetadata } from './types';
import { KernelDetector } from '../shared/kerneldetection';
import {
  MAX_RERUNS_PER_CELL,
  RunToCleanGuard,
  formatFootprintChange
} from './rerunguard';
import { RerunWarningNoticeManager } from './rerunnotice';
import { indexToAlpha } from '../cellindexutils';

/**
 * Extension that adds "Run Next Stale" button to the notebook toolbar
 */
export class FlowbookToolbarExtension implements DocumentRegistry.IWidgetExtension<
  NotebookPanel,
  DocumentRegistry.IModel
> {
  private _highlighter: ReproducibilityCellHighlighter | null = null;
  private _kernelDetector: KernelDetector;
  private _rerunNotice = new RerunWarningNoticeManager();

  constructor(kernelDetector: KernelDetector) {
    this._kernelDetector = kernelDetector;
  }

  /**
   * Set the highlighter reference (called when plugin activates).
   * Pass null on deactivation so the button doesn't touch a disposed
   * highlighter.
   */
  setHighlighter(highlighter: ReproducibilityCellHighlighter | null): void {
    this._highlighter = highlighter;
  }

  /**
   * Create the toolbar button for a new notebook panel
   */
  createNew(
    panel: NotebookPanel,
    _context: DocumentRegistry.IContext<DocumentRegistry.IModel>
  ): IDisposable {
    // Create "Run All Stale/Unrun" button
    const button = new ToolbarButton({
      icon: stepIntoIcon,
      tooltip: 'Run all stale and unrun cells',
      onClick: async () => {
        await this._runAllActionable(panel);
      }
    });

    // Start hidden, show only for flowbook_kernel
    button.node.style.display = 'none';

    // Add to toolbar (position 10 is after fast-forward/run-all button)
    panel.toolbar.insertItem(10, 'flowbook-run-next-stale', button);

    // Function to update button visibility
    const updateButtonVisibility = () => {
      const shouldShow = this._kernelDetector.isFlowbookKernel(panel);
      button.node.style.display = shouldShow ? '' : 'none';
    };

    // Initial visibility update when session is ready
    panel.sessionContext.ready
      .then(() => {
        updateButtonVisibility();
      })
      .catch(err => {
        console.warn(
          'FlowBook toolbar: session failed to initialize; leaving button hidden',
          err
        );
      });

    // Listen for kernel changes
    panel.sessionContext.kernelChanged.connect(() => {
      updateButtonVisibility();
    });

    return {
      dispose: () => {
        button.dispose();
      },
      get isDisposed() {
        return button.isDisposed;
      }
    };
  }

  /**
   * Run all stale and unrun code cells in document order — the
   * RunToClean loop. Stops on hard error always. Stops on violation if
   * continue_after_violation is false. If a cell is executed a second
   * time and marks an earlier cell stale (its read or write sets
   * changed), the loop shows an orange warning notice under that cell
   * and continues; it stops with a potential non-termination error only
   * once a cell has run more than MAX_RERUNS_PER_CELL times this sweep.
   * The iteration cap is a backstop for many-cell rerun cycles.
   * User can cancel mid-loop via kernel interrupt.
   */
  private async _runAllActionable(panel: NotebookPanel): Promise<void> {
    const notebook = panel.content;
    const guard = new RunToCleanGuard();
    const warnedCells = new Set<string>();
    this._rerunNotice.clearAll(panel);
    let nCodeCells = 0;
    for (const w of notebook.widgets) {
      if (w.model.type === 'code') {
        nCodeCells++;
      }
    }
    const maxIterations = Math.max(25, nCodeCells * (nCodeCells + 3));

    for (let iter = 0; iter < maxIterations; iter++) {
      // Find next actionable cell
      let staleCells: ReadonlySet<string> = new Set();
      if (this._highlighter) {
        const stalenessManager = this._highlighter.getStalenessManager(panel);
        staleCells = stalenessManager.staleCells;
      }

      let targetWidgetIdx = -1;
      let targetCodeIdx = -1;
      let codeIdx = 0;
      const widgets = notebook.widgets;
      for (let i = 0; i < widgets.length; i++) {
        const cell = widgets[i];
        if (cell.model.type !== 'code') {
          continue;
        }
        const currentCodeIdx = codeIdx;
        codeIdx++;
        const codeModel = cell.model as ICodeCellModel;
        const source = codeModel.sharedModel.getSource();
        if (!source || source.trim() === '') {
          continue;
        }
        const cellId = cell.model.id;
        const needsRun =
          staleCells.has(cellId) || codeModel.executionCount === null;
        if (needsRun) {
          targetWidgetIdx = i;
          targetCodeIdx = currentCodeIdx;
          break;
        }
      }

      if (targetWidgetIdx < 0) {
        break;
      }

      // Capture identity + metadata before the run: the widget index can
      // go stale if cells are added/removed during execution, and the
      // flowbook metadata is written asynchronously by the comm handler
      // (the shell reply can beat it, leaving us reading the PREVIOUS
      // run's errors).
      const targetCell = widgets[targetWidgetIdx];
      const targetModelId = targetCell.model.id;
      const beforeMeta = targetCell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;

      // Run the cell
      notebook.activeCellIndex = targetWidgetIdx;
      notebook.scrollToCell(targetCell);
      await NotebookActions.run(notebook, panel.sessionContext);

      // Re-find the cell by model ID — indices may have shifted.
      const cell = notebook.widgets.find(w => w.model.id === targetModelId);
      if (!cell) {
        // Cell was removed during the run — look for the next actionable.
        continue;
      }

      // Check for hard error (outputs land with the shell reply, no race)
      const outputs = (cell.model as ICodeCellModel).outputs;
      let hasError = false;
      if (outputs) {
        for (let j = 0; j < outputs.length; j++) {
          const output = outputs.get(j);
          if (output && output.type === 'error') {
            hasError = true;
            break;
          }
        }
      }
      if (hasError) {
        break;
      }

      // Wait for this run's metadata to land before checking violations.
      const meta = await waitForFlowbookMetadata(cell.model, beforeMeta);
      if (meta?.errors && meta.errors.length > 0) {
        break;
      }

      // RunToClean check: a re-executed cell marking an earlier cell
      // stale means the sweep may never terminate — warn and continue.
      const cellOrder = notebook.widgets
        .filter(w => w.model.type === 'code')
        .map(w => w.model.id);
      const report = guard.noteRun(targetModelId, meta ?? null, cellOrder);
      if (report) {
        warnedCells.add(targetModelId);
        const markedLabels = report.backwardStale
          .map(cid => indexToAlpha(cellOrder.indexOf(cid)))
          .join(', ');
        let message =
          `Re-running this cell marked earlier cell(s) ${markedLabels} ` +
          `stale again (run ${report.runCount} of at most ` +
          `${MAX_RERUNS_PER_CELL} this sweep).`;
        if (report.change) {
          const detail = formatFootprintChange(report.change);
          message += ` ${detail.charAt(0).toUpperCase()}${detail.slice(1)}.`;
        }
        message +=
          ' Varying values are fine, but the set of variables a cell ' +
          'reads and writes must be the same on every run. Run-all ' +
          'continues; it stops if this cell runs more than ' +
          `${MAX_RERUNS_PER_CELL} times.`;
        this._rerunNotice.showWarning(cell, message);
      }

      // Per-cell run cap: the hard stop for potential non-termination.
      if (guard.capExceeded(targetModelId)) {
        const label = indexToAlpha(targetCodeIdx);
        const count = guard.runCount(targetModelId);
        const prefix =
          `Cell ${label} ran ${count} times in this sweep ` +
          `(limit ${MAX_RERUNS_PER_CELL})`;
        const message = warnedCells.has(targetModelId)
          ? `${prefix}: its re-runs kept marking earlier cell(s) stale, ` +
            'so repeatedly running stale cells may never make the ' +
            'notebook clean. Varying values are fine, but the set of ' +
            'variables a cell reads and writes must be the same on ' +
            'every run. See the warning under the cell for the ' +
            'footprint change.'
          : warnedCells.size > 0
            ? `${prefix} without the notebook reaching a clean state; ` +
              'the orange warning notice(s) mark the cell(s) whose ' +
              're-runs kept marking earlier cells stale.'
            : `${prefix} without the notebook reaching a clean state ` +
              '(possible untracked nondeterminism).';
        await showErrorMessage('FlowBook: Potential Non-Termination', message);
        break;
      }
    }
  }
}
