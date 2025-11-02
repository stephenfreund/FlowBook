/**
 * Notebook toolbar extension for Ferret commands
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { IDisposable } from '@lumino/disposable';
import { ToolbarButton } from '@jupyterlab/apputils';

import { FerretCommandsManager } from './manager';

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
          const selectedCellIds = this.getSelectedCellIds(panel);
          await this.manager.executeCommand(cmdInfo.id, panel, selectedCellIds);
        }
      });

      panel.toolbar.insertItem(10, `ferret-${cmdInfo.id}-button`, button);
    });

    return {
      dispose: () => {},
      get isDisposed() {
        return false;
      }
    };
  }

  /**
   * Get the IDs of all selected cells in the notebook
   */
  private getSelectedCellIds(panel: NotebookPanel): string[] | undefined {
    const notebook = panel.content;
    const selectedCells: string[] = [];

    // Iterate through all cells
    for (let i = 0; i < notebook.widgets.length; i++) {
      const cell = notebook.widgets[i];
      if (notebook.isSelectedOrActive(cell)) {
        selectedCells.push(cell.model.id);
      }
    }

    // Return undefined if no cells selected (for backward compatibility)
    return selectedCells.length > 0 ? selectedCells : undefined;
  }
}
