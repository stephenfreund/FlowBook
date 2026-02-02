/**
 * Execution hook for FlowBook kernel - extracts reproducibility metadata
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker, Notebook, NotebookActions } from '@jupyterlab/notebook';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { IReproducibilityMetadata } from './types';

export class ReproducibilityExecutionHookManager {
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter;

  constructor(
    _app: JupyterFrontEnd,
    tracker: INotebookTracker,
    highlighter: ReproducibilityCellHighlighter
  ) {
    this._tracker = tracker;
    this._highlighter = highlighter;
    this._setupHooks();
  }

  private _setupHooks(): void {
    // Listen for cell execution completion
    NotebookActions.executed.connect(this._onCellExecuted, this);

    // Listen for cell execution start to set cell_order via magic
    NotebookActions.executionScheduled.connect(this._onExecutionScheduled, this);

    console.log('ReproducibilityExecutionHookManager: Execution hooks installed');
  }

  /**
   * Called before cell execution - send %notebook_structure magic to set cell order
   */
  private _onExecutionScheduled(_sender: any, args: { notebook: Notebook; cell: Cell }): void {
    const { notebook } = args;

    // Get the notebook panel
    const panel = this._tracker.currentWidget;
    if (!panel || panel.content !== notebook) {
      return;
    }

    // Build cell order array (only code cells)
    const cellOrder: string[] = [];
    for (let i = 0; i < notebook.widgets.length; i++) {
      const c = notebook.widgets[i];
      if (c.model.type === 'code') {
        cellOrder.push(c.model.id);
      }
    }

    // Send %notebook_structure magic to kernel
    const session = panel.sessionContext.session;
    if (session && session.kernel && cellOrder.length > 0) {
      const magicCommand = `%notebook_structure ${cellOrder.join(' ')}`;
      session.kernel.requestExecute({ code: magicCommand, silent: true, store_history: false });
      console.log(`ReproducibilityExecutionHook: Sent notebook_structure with ${cellOrder.length} cells`);
    }
  }

  private _extractReproducibilityMetadata(outputs: IOutput[]): IReproducibilityMetadata | null {
    console.log(`ReproducibilityExecutionHook: Checking ${outputs.length} outputs for flowbook metadata`);

    for (const output of outputs) {
      console.log(`ReproducibilityExecutionHook: Output type = ${output.output_type}`);

      if (output.output_type !== 'display_data') {
        continue;
      }

      const metadata = (output as any).metadata;
      console.log('ReproducibilityExecutionHook: display_data metadata =', metadata);

      if (!metadata?.flowbook) {
        console.log('ReproducibilityExecutionHook: No flowbook in metadata');
        continue;
      }

      console.log('ReproducibilityExecutionHook: Found flowbook metadata!', metadata.flowbook);
      return metadata.flowbook as IReproducibilityMetadata;
    }
    console.log('ReproducibilityExecutionHook: No flowbook metadata found in any output');
    return null;
  }

  private _onCellExecuted(_sender: any, args: { notebook: Notebook; cell: Cell }): void {
    const { notebook, cell } = args;

    if (cell.model.type !== 'code') {
      return;
    }

    // Get the notebook panel
    const panel = this._tracker.currentWidget;
    if (!panel || panel.content !== notebook) {
      return;
    }

    // Get outputs
    const codeModel = cell.model as ICodeCellModel;
    const outputs: IOutput[] = [];
    for (let i = 0; i < codeModel.outputs.length; i++) {
      outputs.push(codeModel.outputs.get(i).toJSON() as IOutput);
    }

    // Extract reproducibility metadata
    const reproducibilityMetadata = this._extractReproducibilityMetadata(outputs);
    if (!reproducibilityMetadata) {
      return;
    }

    // Store metadata on cell
    cell.model.setMetadata('flowbook', reproducibilityMetadata);

    // Update staleness manager
    const stalenessManager = this._highlighter.getStalenessManager(panel);
    stalenessManager.updateFromMetadata(reproducibilityMetadata);

    console.log(`ReproducibilityExecutionHook: Extracted metadata for cell ${cell.model.id}:`, reproducibilityMetadata);
  }
}
