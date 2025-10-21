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
  implements DocumentRegistry.IWidgetExtension<NotebookPanel, DocumentRegistry.IModel>
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
        icon: cmdInfo.icon,
        onClick: async () => {
          await this.manager.executeCommand(cmdInfo.id, panel);
        }
      });

      panel.toolbar.insertItem(10, `ferret-${cmdInfo.id}-button`, button);
    });

    return {
      dispose: () => {},
      get isDisposed() { return false; }
    };
  }
}
