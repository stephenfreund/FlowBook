/**
 * Notebook toolbar extension for FlowBook - Run Next Stale/Unrun button
 */

import { NotebookPanel, NotebookActions } from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { IDisposable } from '@lumino/disposable';
import { ToolbarButton } from '@jupyterlab/apputils';
import { stepIntoIcon } from '@jupyterlab/ui-components';

import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { KernelDetector } from '../shared/kerneldetection';

/**
 * Extension that adds "Run Next Stale" button to the notebook toolbar
 */
export class FlowbookToolbarExtension
  implements
    DocumentRegistry.IWidgetExtension<NotebookPanel, DocumentRegistry.IModel>
{
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
    // Create "Run Next Stale/Unrun" button
    const button = new ToolbarButton({
      icon: stepIntoIcon,
      tooltip: 'Run next stale or unrun cell',
      onClick: async () => {
        await this._runNextStaleOrUnrun(panel);
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
   * Find and run the first stale or unrun code cell in document order
   */
  private async _runNextStaleOrUnrun(panel: NotebookPanel): Promise<void> {
    const notebook = panel.content;
    const widgets = notebook.widgets;

    // Get stale cells from staleness manager if available
    let staleCells: ReadonlySet<string> = new Set();
    if (this._highlighter) {
      const stalenessManager = this._highlighter.getStalenessManager(panel);
      staleCells = stalenessManager.staleCells;
    }

    // Find first non-empty code cell that is stale or has never been run
    for (let i = 0; i < widgets.length; i++) {
      const cell = widgets[i];
      if (cell.model.type !== 'code') {
        continue;
      }

      const codeModel = cell.model as ICodeCellModel;

      // Skip empty cells (whitespace-only counts as empty)
      const source = codeModel.sharedModel.getSource();
      if (!source || source.trim() === '') {
        continue;
      }

      const cellId = cell.model.id;
      const isStale = staleCells.has(cellId);
      // Check if cell has been executed by looking at executionCount
      const hasBeenExecuted = codeModel.executionCount !== null;

      // Cell needs to run if it's stale OR if it's never been executed
      const needsRun = isStale || !hasBeenExecuted;

      if (needsRun) {
        console.log(
          `FlowBook: Running ${isStale ? 'stale' : 'unrun'} cell at index ${i}`
        );

        // Activate the cell
        notebook.activeCellIndex = i;

        // Scroll to make cell visible
        notebook.scrollToCell(cell);

        // Run the cell
        await NotebookActions.run(notebook, panel.sessionContext);

        return;
      }
    }

    console.log('FlowBook: No stale or unrun cells to run');
  }
}
