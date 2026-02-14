/**
 * Execution hook for FlowBook kernel - extracts reproducibility metadata
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import {
  INotebookTracker,
  Notebook,
  NotebookActions
} from '@jupyterlab/notebook';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { IReproducibilityMetadata } from './types';

export class ReproducibilityExecutionHookManager {
  private _app: JupyterFrontEnd;
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter;
  private _editTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private _executedCells: Set<string> = new Set();

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    highlighter: ReproducibilityCellHighlighter
  ) {
    this._app = app;
    this._tracker = tracker;
    this._highlighter = highlighter;
    this._setupHooks();
  }

  private _setupHooks(): void {
    // Listen for cell execution completion
    NotebookActions.executed.connect(this._onCellExecuted, this);

    // Listen for cell execution start to set cell_order via magic
    NotebookActions.executionScheduled.connect(
      this._onExecutionScheduled,
      this
    );

    // [EDIT transition (§2.3)] Listen for cell content changes
    this._tracker.currentChanged.connect(this._setupCellEditListener, this);

    console.log(
      'ReproducibilityExecutionHookManager: Execution hooks installed'
    );
  }

  /**
   * [EDIT transition (§2.3)] Set up listeners for cell content changes.
   * When a code cell's source changes and the cell was previously executed,
   * send %cell_edited <cell_id> to the kernel with debouncing.
   */
  private _setupCellEditListener(): void {
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    const notebook = panel.content;
    for (let i = 0; i < notebook.widgets.length; i++) {
      const cell = notebook.widgets[i];
      if (cell.model.type !== 'code') {
        continue;
      }
      const cellId = cell.model.id;
      const model = cell.model as ICodeCellModel;

      // Listen for source changes
      model.sharedModel.changed.connect(() => {
        this._onCellContentChanged(cellId);
      });
    }
  }

  /**
   * [EDIT transition (§2.3)] Handle cell content change with debouncing.
   */
  private _onCellContentChanged(cellId: string): void {
    // Only notify kernel about cells that have been previously executed
    if (!this._executedCells.has(cellId)) {
      return;
    }

    // Debounce: cancel previous timer for this cell
    const existing = this._editTimers.get(cellId);
    if (existing) {
      clearTimeout(existing);
    }

    // Set new timer (1s debounce)
    const timer = setTimeout(() => {
      this._sendCellEdited(cellId);
      this._editTimers.delete(cellId);
    }, 1000);

    this._editTimers.set(cellId, timer);
  }

  /**
   * [EDIT transition (§2.3)] Send %cell_edited magic to kernel.
   */
  private _sendCellEdited(cellId: string): void {
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    const session = panel.sessionContext.session;
    if (session && session.kernel) {
      session.kernel.requestExecute({
        code: `%cell_edited ${cellId}`,
        silent: true,
        store_history: false
      });
      console.log(
        `ReproducibilityExecutionHook: Sent cell_edited for ${cellId}`
      );
    }
  }

  /**
   * Called before cell execution - send %notebook_structure magic to set cell order
   */
  private _onExecutionScheduled(
    _sender: any,
    args: { notebook: Notebook; cell: Cell }
  ): void {
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
      session.kernel.requestExecute({
        code: magicCommand,
        silent: true,
        store_history: false
      });
      console.log(
        `ReproducibilityExecutionHook: Sent notebook_structure with ${cellOrder.length} cells`
      );
    }
  }

  private _extractReproducibilityMetadata(
    outputs: IOutput[]
  ): IReproducibilityMetadata | null {
    console.log(
      `ReproducibilityExecutionHook: Checking ${outputs.length} outputs for flowbook metadata`
    );

    for (const output of outputs) {
      console.log(
        `ReproducibilityExecutionHook: Output type = ${output.output_type}`
      );

      if (output.output_type !== 'display_data') {
        continue;
      }

      const metadata = (output as any).metadata;
      console.log(
        'ReproducibilityExecutionHook: display_data metadata =',
        metadata
      );

      if (!metadata?.flowbook) {
        console.log('ReproducibilityExecutionHook: No flowbook in metadata');
        continue;
      }

      console.log(
        'ReproducibilityExecutionHook: Found flowbook metadata!',
        metadata.flowbook
      );
      return metadata.flowbook as IReproducibilityMetadata;
    }
    console.log(
      'ReproducibilityExecutionHook: No flowbook metadata found in any output'
    );
    return null;
  }

  private _onCellExecuted(
    _sender: any,
    args: { notebook: Notebook; cell: Cell }
  ): void {
    const { notebook, cell } = args;

    if (cell.model.type !== 'code') {
      return;
    }

    // [EDIT transition (§2.3)] Track executed cells for edit detection
    this._executedCells.add(cell.model.id);

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
    const reproducibilityMetadata =
      this._extractReproducibilityMetadata(outputs);
    if (!reproducibilityMetadata) {
      return;
    }

    // Store metadata on cell
    cell.model.setMetadata('flowbook', reproducibilityMetadata);

    // Update staleness manager
    const stalenessManager = this._highlighter.getStalenessManager(panel);
    stalenessManager.updateFromMetadata(reproducibilityMetadata);

    // Notify command system so context menu items re-evaluate isEnabled
    this._app.commands.notifyCommandChanged('flowbook:exec-restore');

    console.log(
      `ReproducibilityExecutionHook: Extracted metadata for cell ${cell.model.id}:`,
      reproducibilityMetadata
    );
  }
}
