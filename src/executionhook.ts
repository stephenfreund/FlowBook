/**
 * Execution hook for intercepting cell execution to auto-generate code from string specifications
 * and extract ferret metadata from kernel outputs
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel, Notebook, NotebookActions } from '@jupyterlab/notebook';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { FerretCommandsManager } from './manager';
import { IFerretMetadata, IFerretProfileData, IDynamicDependencies } from './types';

/**
 * Manages execution hooks to auto-generate code before execution
 */
export class ExecutionHookManager {
  private app: JupyterFrontEnd;
  private tracker: INotebookTracker;
  private manager: FerretCommandsManager;
  private executingCells: Set<string> = new Set();
  private processing: boolean = false;

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    manager: FerretCommandsManager
  ) {
    this.app = app;
    this.tracker = tracker;
    this.manager = manager;
    this.setupHooks();
  }

  /**
   * Check if a cell contains only a string constant
   */
  private isStringConstantCell(source: string): boolean {
    const trimmed = source.trim();
    if (!trimmed) {
      return false;
    }

    // Check for single-line string (with single, double, or triple quotes)
    const singleLinePattern = /^["'](?:[^"'\\]|\\.)*["']$/;
    const tripleQuoteSingle = /^'''[\s\S]*'''$/;
    const tripleQuoteDouble = /^"""[\s\S]*"""$/;

    return (
      singleLinePattern.test(trimmed) ||
      tripleQuoteSingle.test(trimmed) ||
      tripleQuoteDouble.test(trimmed)
    );
  }

  /**
   * Setup execution hooks by wrapping notebook execution commands
   */
  private setupHooks(): void {
    // Wrap the run-cell command
    this.wrapCommand('notebook:run-cell');
    this.wrapCommand('notebook:run-cell-and-select-next');
    this.wrapCommand('notebook:run-all-cells');
    this.wrapCommand('notebook:run-all-above');
    this.wrapCommand('notebook:run-all-below');

    // Listen for cell execution completion to extract ferret metadata
    NotebookActions.executed.connect(this.onCellExecuted, this);

    console.log('ExecutionHookManager: Execution command hooks installed');
  }

  /**
   * Wrap a notebook execution command to intercept and generate code if needed
   */
  private wrapCommand(commandId: string): void {
    if (!this.app.commands.hasCommand(commandId)) {
      console.warn(
        `ExecutionHookManager: Command ${commandId} not found`
      );
      return;
    }

    // Patch the internal command registry to intercept execution
    const commands = (this.app.commands as any)._commands;
    const originalCmd = commands.get(commandId);

    if (originalCmd) {
      const origExecute = originalCmd.execute;
      originalCmd.execute = async (args?: any) => {
        // Get the current notebook panel
        const panel = this.tracker.currentWidget;
        if (panel) {
          // Check cells before execution
          await this.checkAndGenerateCells(panel, commandId);
        }

        // Execute the original command
        return origExecute.call(originalCmd, args);
      };

      console.log(`ExecutionHookManager: Wrapped command ${commandId}`);
    }
  }

  /**
   * Check cells and generate code if needed before execution
   */
  private async checkAndGenerateCells(
    panel: NotebookPanel,
    commandId: string
  ): Promise<void> {
    if (this.processing) {
      return;
    }

    this.processing = true;

    try {
      const notebook = panel.content;
      const cellsToCheck: Cell[] = [];

      // Determine which cells to check based on the command
      if (commandId === 'notebook:run-all-cells') {
        // Check all code cells
        for (let i = 0; i < notebook.widgets.length; i++) {
          const cell = notebook.widgets[i];
          if (cell.model.type === 'code') {
            cellsToCheck.push(cell);
          }
        }
      } else if (commandId === 'notebook:run-all-above') {
        // Check all code cells above active cell
        const activeIndex = notebook.activeCellIndex;
        for (let i = 0; i < activeIndex; i++) {
          const cell = notebook.widgets[i];
          if (cell.model.type === 'code') {
            cellsToCheck.push(cell);
          }
        }
      } else if (commandId === 'notebook:run-all-below') {
        // Check all code cells from active cell down
        const activeIndex = notebook.activeCellIndex;
        for (let i = activeIndex; i < notebook.widgets.length; i++) {
          const cell = notebook.widgets[i];
          if (cell.model.type === 'code') {
            cellsToCheck.push(cell);
          }
        }
      } else {
        // For run-cell commands, check only the active cell
        const activeCell = notebook.activeCell;
        if (activeCell && activeCell.model.type === 'code') {
          cellsToCheck.push(activeCell);
        }
      }

      // Process each cell
      for (const cell of cellsToCheck) {
        await this.checkAndGenerateCell(panel, cell);
      }
    } finally {
      this.processing = false;
    }
  }

  /**
   * Check a single cell and generate code if it's a string constant
   */
  private async checkAndGenerateCell(
    panel: NotebookPanel,
    cell: Cell
  ): Promise<void> {
    const source = cell.model.sharedModel.getSource();
    if (!this.isStringConstantCell(source)) {
      return;
    }

    const cellId = cell.model.id;

    // If the cell is ONLY a string constant, we should generate code
    // even if it was generated before (user may have deleted the code)
    // The isStringConstantCell check ensures the cell has ONLY the string,
    // so if generated code exists, this check would be false

    // Check if we're already processing this cell
    if (this.executingCells.has(cellId)) {
      return;
    }

    console.log(
      `ExecutionHook: Detected string constant cell, generating code for cell ${cellId}...`
    );

    this.executingCells.add(cellId);

    try {
      await this.manager.executeCommand('generate', panel, cellId);
      console.log(`ExecutionHook: Code generation complete for cell ${cellId}`);
    } catch (error) {
      console.error('ExecutionHook: Generation failed:', error);
    } finally {
      this.executingCells.delete(cellId);
    }
  }

  /**
   * Extract ferret metadata from cell outputs.
   * Mirrors the server-side extract_and_set_metadata() function.
   */
  private extractFerretMetadata(outputs: IOutput[]): Partial<IFerretMetadata> | null {
    for (const output of outputs) {
      if (output.output_type !== 'display_data') {
        continue;
      }

      const metadata = (output as any).metadata;
      if (!metadata) {
        continue;
      }

      const ferretMeta: Partial<IFerretMetadata> = {};

      // Extract profile metadata
      if (metadata.profile) {
        ferretMeta.profile = metadata.profile as IFerretProfileData;
      }

      // Extract tracking/dynamic_dependencies metadata
      if (metadata.tracking) {
        ferretMeta.dynamic_dependencies = {
          reads_before_writes: metadata.tracking.reads_before_writes || [],
          writes: metadata.tracking.writes || []
        } as IDynamicDependencies;
      }

      if (Object.keys(ferretMeta).length > 0) {
        return ferretMeta;
      }
    }
    return null;
  }

  /**
   * Handle cell execution completion to extract ferret metadata from outputs
   */
  private onCellExecuted(
    _sender: any,
    args: { notebook: Notebook; cell: Cell }
  ): void {
    const { cell } = args;

    // Only process code cells
    if (cell.model.type !== 'code') {
      return;
    }

    // Get outputs from the cell model
    const codeModel = cell.model as ICodeCellModel;
    const outputs: IOutput[] = [];
    for (let i = 0; i < codeModel.outputs.length; i++) {
      outputs.push(codeModel.outputs.get(i).toJSON() as IOutput);
    }

    // Extract ferret metadata from outputs
    const ferretMeta = this.extractFerretMetadata(outputs);
    if (!ferretMeta) {
      return;
    }

    // Merge with existing ferret metadata
    const existingMeta = (cell.model.getMetadata('ferret') as IFerretMetadata) || {};
    const mergedMeta = { ...existingMeta, ...ferretMeta };

    // Set the merged metadata on the cell
    cell.model.setMetadata('ferret', mergedMeta);

    console.log(
      `ExecutionHook: Extracted ferret metadata for cell ${cell.model.id}:`,
      ferretMeta
    );
  }
}
