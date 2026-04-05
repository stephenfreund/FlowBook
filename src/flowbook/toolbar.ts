/**
 * Notebook toolbar extension for FlowBook - Run All Stale/Unrun button
 */

import { NotebookPanel, NotebookActions } from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { IDisposable } from '@lumino/disposable';
import { ToolbarButton } from '@jupyterlab/apputils';
import { stepIntoIcon } from '@jupyterlab/ui-components';

import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { IReproducibilityMetadata } from './types';
import { KernelDetector } from '../shared/kerneldetection';

/**
 * Extension that adds "Run Next Stale" button to the notebook toolbar
 */
export class FlowbookToolbarExtension implements DocumentRegistry.IWidgetExtension<
  NotebookPanel,
  DocumentRegistry.IModel
> {
  private _highlighter: ReproducibilityCellHighlighter | null = null;
  private _kernelDetector: KernelDetector;

  constructor(kernelDetector: KernelDetector) {
    this._kernelDetector = kernelDetector;
  }

  /**
   * Set the highlighter reference (called when plugin activates)
   */
  setHighlighter(highlighter: ReproducibilityCellHighlighter): void {
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
    panel.sessionContext.ready.then(() => {
      updateButtonVisibility();
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
   * Run all stale and unrun code cells in document order.
   * Stops on hard error always. Stops on violation if continue_after_violation is false.
   * User can cancel mid-loop via kernel interrupt.
   */
  private async _runAllActionable(panel: NotebookPanel): Promise<void> {
    const notebook = panel.content;
    const maxIterations = 500;

    for (let iter = 0; iter < maxIterations; iter++) {
      // Find next actionable cell
      let staleCells: ReadonlySet<string> = new Set();
      if (this._highlighter) {
        const stalenessManager = this._highlighter.getStalenessManager(panel);
        staleCells = stalenessManager.staleCells;
      }

      let targetWidgetIdx = -1;
      const widgets = notebook.widgets;
      for (let i = 0; i < widgets.length; i++) {
        const cell = widgets[i];
        if (cell.model.type !== 'code') {
          continue;
        }
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
          break;
        }
      }

      if (targetWidgetIdx < 0) {
        break;
      }

      // Run the cell
      notebook.activeCellIndex = targetWidgetIdx;
      notebook.scrollToCell(widgets[targetWidgetIdx]);
      await NotebookActions.run(notebook, panel.sessionContext);

      // Check for hard error
      const cell = widgets[targetWidgetIdx];
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

      // Check for violation
      const meta = cell.model.getMetadata('flowbook') as IReproducibilityMetadata | undefined;
      if (meta?.errors && meta.errors.length > 0) {
        break;
      }
    }
  }
}
