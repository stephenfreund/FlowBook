/**
 * Manager for Ferret commands
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { ICommandPalette } from '@jupyterlab/apputils';
import { showDialog, Dialog } from '@jupyterlab/apputils';

import { FerretAPI } from './api';
import { KernelUtils } from './kernel';
import {
  CommandInfo,
  CommandResult,
  ExecuteCommandRequest,
  FERRET_COMMANDS
} from './types';

/**
 * Manages Ferret commands, their registration, and execution
 */
export class FerretCommandsManager {
  private commands: CommandInfo[] = FERRET_COMMANDS;
  private app: JupyterFrontEnd;
  private tracker: INotebookTracker;

  constructor(app: JupyterFrontEnd, tracker: INotebookTracker) {
    this.app = app;
    this.tracker = tracker;
  }

  /**
   * Get all loaded commands
   */
  getCommands(): CommandInfo[] {
    return this.commands;
  }

  /**
   * Execute a command on a notebook
   */
  async executeCommand(
    commandId: string,
    notebook: NotebookPanel
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

      // Get selected cell IDs
      const selectedCells = notebook.content.selectedCells;
      const selectedCellIds = selectedCells.map(cell => cell.model.id);

      // Build the request
      const request: ExecuteCommandRequest = {
        command: commandId,
        notebook: notebookContent,
        params: {},
        selected_cell_ids: selectedCellIds
      };

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
        notebook.content.model?.fromJSON(result.notebook);

        console.log('Command metadata:', result.metadata);

        showDialog({
          title: `${commandInfo?.label || 'Command'} Complete`,
          body: 'Command executed successfully. Check the console for detailed metadata.',
          buttons: [Dialog.okButton()]
        });
      }

      return result;
    } catch (error) {
      console.error(`Failed to execute command ${commandId}:`, error);
      showDialog({
        title: 'Command Error',
        body: `Failed to execute command: ${error}`,
        buttons: [Dialog.okButton()]
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
        label: cmdInfo.label,
        caption: cmdInfo.tooltip,
        execute: async () => {
          const current = this.tracker.currentWidget;
          if (current) {
            await this.executeCommand(cmdInfo.id, current);
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
        args: {}
      });
    });
  }
}
