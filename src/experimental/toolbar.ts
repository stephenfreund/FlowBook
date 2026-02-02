/**
 * Notebook toolbar extension for FlowBook commands
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { IDisposable } from '@lumino/disposable';
import { ToolbarButton, Clipboard, showErrorMessage } from '@jupyterlab/apputils';

import { FlowbookCommandsManager } from './manager';
import { FlowbookAPI } from '../api';

/**
 * Extension that adds FlowBook command buttons to the notebook toolbar
 */
export class NotebookToolbarExtension
  implements
    DocumentRegistry.IWidgetExtension<NotebookPanel, DocumentRegistry.IModel>
{
  private manager: FlowbookCommandsManager;

  constructor(manager: FlowbookCommandsManager) {
    this.manager = manager;
  }

  /**
   * Create the toolbar buttons for a new notebook panel
   */
  createNew(
    panel: NotebookPanel,
    context: DocumentRegistry.IContext<DocumentRegistry.IModel>
  ): IDisposable {
    const buttons: ToolbarButton[] = [];
    const commands = this.manager.getCommands();

    // Function to check if current kernel is flowbook_kernel
    const isFlowbookKernel = () => {
      const kernelName = panel.sessionContext.session?.kernel?.name;
      return kernelName === 'flowbook_kernel';
    };

    // Function to update button visibility
    const updateButtonVisibility = () => {
      const shouldShow = isFlowbookKernel();
      buttons.forEach(btn => {
        btn.node.style.display = shouldShow ? '' : 'none';
      });
    };

    // Create buttons
    commands.forEach(cmdInfo => {
      const button = new ToolbarButton({
        label: cmdInfo.label,
        tooltip: cmdInfo.tooltip,
        // icon: cmdInfo.icon,
        onClick: async () => {
          await this.manager.executeCommand(cmdInfo.id, panel, undefined);
        }
      });

      panel.toolbar.insertItem(10, `flowbook-${cmdInfo.id}-button`, button);
      buttons.push(button);
    });

    // Add Copy Connection File button
    const copyConnectionButton = new ToolbarButton({
      label: 'Copy Connection',
      tooltip: 'Copy kernel connection file path to clipboard',
      onClick: async () => {
        await this.copyKernelConnectionFile(panel);
      }
    });

    panel.toolbar.insertItem(100, 'flowbook-copy-connection-button', copyConnectionButton);
    buttons.push(copyConnectionButton);

    // Initial visibility update
    panel.sessionContext.ready.then(() => {
      updateButtonVisibility();
    });

    // Listen for kernel changes
    panel.sessionContext.kernelChanged.connect(() => {
      updateButtonVisibility();
    });

    return {
      dispose: () => {
        buttons.forEach(btn => btn.dispose());
      },
      get isDisposed() {
        return buttons.length === 0;
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
      const connectionFile = await FlowbookAPI.getKernelConnectionFile(kernelId);

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
