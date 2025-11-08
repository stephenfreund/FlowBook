/**
 * Notebook toolbar extension for Ferret commands
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { IDisposable } from '@lumino/disposable';
import { ToolbarButton, Clipboard, showErrorMessage } from '@jupyterlab/apputils';

import { FerretCommandsManager } from './manager';
import { FerretAPI } from './api';

/**
 * Extension that adds Ferret command buttons to the notebook toolbar
 */
export class NotebookToolbarExtension
  implements
    DocumentRegistry.IWidgetExtension<NotebookPanel, DocumentRegistry.IModel>
{
  private manager: FerretCommandsManager;

  constructor(manager: FerretCommandsManager) {
    this.manager = manager;
  }

  /**
   * Create the toolbar buttons for a new notebook panel
   */
  createNew(
    panel: NotebookPanel,
    context: DocumentRegistry.IContext<DocumentRegistry.IModel>
  ): IDisposable {
    const commands = this.manager.getCommands();

    commands.forEach(cmdInfo => {
      const button = new ToolbarButton({
        label: cmdInfo.label,
        tooltip: cmdInfo.tooltip,
        // icon: cmdInfo.icon,
        onClick: async () => {
          // Get selected cell IDs for commands that need them
          // const selectedCellIds = this.getSelectedCellIds(panel);
          await this.manager.executeCommand(cmdInfo.id, panel, undefined);
        }
      });

      panel.toolbar.insertItem(10, `ferret-${cmdInfo.id}-button`, button);
    });

    // Add Copy Connection File button
    const copyConnectionButton = new ToolbarButton({
      label: 'Copy Connection',
      tooltip: 'Copy kernel connection file path to clipboard',
      onClick: async () => {
        await this.copyKernelConnectionFile(panel);
      }
    });

    panel.toolbar.insertItem(100, 'ferret-copy-connection-button', copyConnectionButton);

    return {
      dispose: () => {},
      get isDisposed() {
        return false;
      }
    };
  }

  /**
   * Copy the kernel connection file path to clipboard
   */
  private async copyKernelConnectionFile(panel: NotebookPanel): Promise<void> {
    try {
      const session = panel.sessionContext.session;

      if (!session || !session.kernel) {
        await showErrorMessage(
          'No Kernel',
          'No kernel is running. Please start a kernel first.'
        );
        return;
      }

      const kernelId = session.kernel.id;
      const connectionFile = await FerretAPI.getKernelConnectionFile(kernelId);

      Clipboard.copyToSystem(connectionFile);

      console.log(`Copied kernel connection file to clipboard: ${connectionFile}`);
    } catch (error) {
      await showErrorMessage(
        'Error',
        `Failed to get kernel connection file: ${error}`
      );
    }
  }

  // /**
  //  * Get the IDs of all selected cells in the notebook
  //  */
  // private getSelectedCellIds(panel: NotebookPanel): string[] | undefined {
  //   const notebook = panel.content;
  //   const selectedCells: string[] = [];

  //   // Iterate through all cells
  //   for (let i = 0; i < notebook.widgets.length; i++) {
  //     const cell = notebook.widgets[i];
  //     if (notebook.isSelectedOrActive(cell)) {
  //       selectedCells.push(cell.model.id);
  //     }
  //   }

  //   // Return undefined if no cells selected (for backward compatibility)
  //   return selectedCells.length > 0 ? selectedCells : undefined;
  // }
}
