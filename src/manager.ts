/**
 * Manager for Ferret commands
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { ICommandPalette } from '@jupyterlab/apputils';
import { showDialog, Dialog } from '@jupyterlab/apputils';
import { Notification } from '@jupyterlab/apputils';

import { FerretAPI } from './api';
import { KernelUtils } from './kernel';
import {
  CommandInfo,
  CommandResult,
  ExecuteCommandRequest,
  FERRET_COMMANDS
} from './types';
import { NotebookHistoryManager } from './history';
import React from 'react';

/**
 * Manages Ferret commands, their registration, and execution
 */
export class FerretCommandsManager {
  private commands: CommandInfo[] = FERRET_COMMANDS;
  private app: JupyterFrontEnd;
  private tracker: INotebookTracker;
  private historyManager: NotebookHistoryManager;

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    historyManager: NotebookHistoryManager
  ) {
    this.app = app;
    this.tracker = tracker;
    this.historyManager = historyManager;
  }

  /**
   * Get all loaded commands
   */
  getCommands(): CommandInfo[] {
    return this.commands;
  }

  /**
   * Execute a command on a notebook
   * @param commandId - The command to execute
   * @param notebook - The notebook panel
   * @param cellIdsOrSpecificCell - Either a single cell ID (from cell toolbar) or array of cell IDs (from notebook toolbar with selection)
   */
  async executeCommand(
    commandId: string,
    notebook: NotebookPanel,
    cellIdsOrSpecificCell?: string | string[]
  ): Promise<CommandResult | null> {
    try {
      const notebookContent = notebook.content.model?.toJSON();

      if (!notebookContent) {
        console.error('Could not get notebook content');
        showDialog({
          title: 'Error',
          body: 'Could not get notebook content',
          buttons: [Dialog.okButton()]
        });
        return null;
      }

      const commandInfo = this.commands.find(cmd => cmd.id === commandId);

      // Flush any pending user edits before executing command
      this.historyManager.flushPendingEdit(notebook.context.path, notebook);

      // Determine selected cell IDs based on context
      let selectedCellIds: string[] | undefined;
      if (cellIdsOrSpecificCell) {
        if (typeof cellIdsOrSpecificCell === 'string') {
          // Single cell ID from cell toolbar
          selectedCellIds = [cellIdsOrSpecificCell];
          console.log('Executing command on specific cell:', cellIdsOrSpecificCell);
        } else {
          // Array of cell IDs from notebook toolbar with selection
          selectedCellIds = cellIdsOrSpecificCell;
          console.log('Executing command on selected cells:', selectedCellIds);
        }
      } else {
        // Command invoked from notebook toolbar with no selection
        selectedCellIds = undefined;
        console.log('Executing command on entire notebook (no specific cells)');
      }

      // Build the request
      const request: ExecuteCommandRequest = {
        command: commandId,
        notebook: notebookContent,
        params: {}
      };

      // Only include selected_cell_ids if defined
      if (selectedCellIds !== undefined) {
        request.selected_cell_ids = selectedCellIds;
      }

      // Handle kernel requirement
      if (commandInfo?.requires_kernel) {
        const kernelInfo = await KernelUtils.ensureKernel(notebook);
        if (!kernelInfo) {
          return null;
        }
        request.kernel_id = kernelInfo.kernel_id;
      }

      // Execute the command
      const result = await FerretAPI.executeCommand(request);

      // Update the notebook with results
      if (result.notebook) {
        // Add history entry for this command
        const affectedCells = selectedCellIds || this.getAllCellIds(result.notebook);
        this.historyManager.addCommandEntry(notebook.context.path, {
          id: `cmd-${Date.now()}`,
          timestamp: Date.now(),
          commandId: commandId,
          commandLabel: commandInfo?.label || commandId,
          icon: commandInfo?.icon || 'ui-components:edit',
          notebookSnapshot: result.notebook,
          affectedCells: affectedCells,
          metadata: result.metadata,
          description: this.generateCommandDescription(
            commandId,
            commandInfo?.label,
            result.notebook,
            affectedCells
          )
        });

        notebook.content.model?.fromJSON(result.notebook);

        console.log('Command metadata:', result.metadata);
        console.log(`Command cost: $${result.total_cost.toFixed(4)}, time: ${result.total_time.toFixed(2)}s`);

        // Format cost and time for display
        const costStr = result.total_cost > 0 ? ` (Cost: $${result.total_cost.toFixed(4)}, Time: ${result.total_time.toFixed(1)}s)` : '';
        Notification.success(`${commandInfo?.label || 'Command'} Complete${costStr}`, { autoClose: 3000 });

      }

      return result;
    } catch (error) {
      console.error(`Failed to execute command ${commandId}:`, error);

      // Extract error message from various error formats
      let errorMessage = 'Unknown error';
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      } else if (error && typeof error === 'object') {
        // Try to extract from common error formats
        const errorObj = error as any;
        errorMessage = errorObj.error || errorObj.message || errorObj.toString();
      }

      showDialog({
        title: 'Command Error',
        buttons: [Dialog.okButton()],
        body: React.createElement('div', { dangerouslySetInnerHTML: { __html: `<pre style="white-space: pre-wrap; word-break: break-word;">${errorMessage}</pre>` } })
      });
      return null;
    }
  }

  /**
   * Register all commands with JupyterLab's command registry
   */
  registerCommands(): void {
    this.commands.forEach(cmdInfo => {
      const commandId = `data_ferret:${cmdInfo.id}`;

      this.app.commands.addCommand(commandId, {
        label: `Ferret: ${cmdInfo.label}`,
        caption: cmdInfo.tooltip,
        execute: async (args?: any) => {
          const current = this.tracker.currentWidget;
          if (current) {
            // Determine which cell(s) to operate on:
            // 1. If cellId is explicitly provided (from cell toolbar), use it
            // 2. If fromContextMenu is true, use the active cell
            // 3. Otherwise, operate on the entire notebook (no cellId)
            let cellId: string | undefined;

            if (args?.cellId) {
              // Explicit cell ID from cell toolbar
              cellId = args.cellId as string;
            } else if (args?.fromContextMenu) {
              // From context menu - use active cell
              const activeCell = current.content.activeCell;
              if (activeCell) {
                cellId = activeCell.model.id;
              }
            }
            // else: undefined, which means operate on entire notebook

            await this.executeCommand(cmdInfo.id, current, cellId);
          }
        }
      });
    });
  }

  /**
   * Add commands to the command palette
   */
  addToPalette(palette: ICommandPalette): void {
    this.commands.forEach(cmdInfo => {
      const commandId = `data_ferret:${cmdInfo.id}`;
      palette.addItem({
        command: commandId,
        category: 'Ferret Commands',
        args: {},
      });
    });
  }

  /**
   * Add commands to the code cell context menu
   */
  addToContextMenu(tracker: INotebookTracker): void {
    // Add each Ferret command to the context menu
    [...this.commands].reverse().forEach((cmdInfo, index) => {
      const commandId = `data_ferret:${cmdInfo.id}`;

      // Add menu item to the context menu
      // Using rank 0-3 to place at the top of the context menu
      this.app.contextMenu.addItem({
        command: commandId,
        selector: '.jp-Cell.jp-CodeCell',
        rank: 0,
        // Pass fromContextMenu flag to indicate this execution is from context menu
        args: { fromContextMenu: true }
      });
    });

    // Add a separator after Ferret commands to create a distinct section
    this.app.contextMenu.addItem({
      type: 'separator',
      selector: '.jp-Cell.jp-CodeCell',
      rank: this.commands.length
    });
  }

  /**
   * Get all cell IDs from a notebook
   */
  private getAllCellIds(notebook: any): string[] {
    return notebook.cells?.map((cell: any) => cell.id) || [];
  }

  /**
   * Generate a human-readable description for a command execution
   */
  private generateCommandDescription(
    commandId: string,
    label?: string,
    notebook?: any,
    affectedCells?: string[]
  ): string {
    if (!affectedCells || affectedCells.length === 0) {
      return label || commandId;
    }

    const indices = this.getCellIndices(notebook, affectedCells);
    const indicesText = indices.length > 0 ? ` [${indices.join(', ')}]` : '';
    return `${label || commandId}${indicesText}`;
  }

  /**
   * Get 1-based cell indices from cell IDs
   */
  private getCellIndices(notebook: any, cellIds: string[]): number[] {
    if (!notebook || !notebook.cells || !cellIds) {
      return [];
    }

    const indices: number[] = [];
    const cellIdToIndex = new Map<string, number>();

    notebook.cells.forEach((cell: any, index: number) => {
      cellIdToIndex.set(cell.id, index + 1); // 1-based indexing
    });

    cellIds.forEach(cellId => {
      const index = cellIdToIndex.get(cellId);
      if (index !== undefined) {
        indices.push(index);
      }
    });

    return indices.sort((a, b) => a - b);
  }
}
