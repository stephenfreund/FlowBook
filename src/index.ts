/**
 * Ferret JupyterLab Frontend Extension
 * Adds toolbar buttons for ferret commands with kernel communication
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { INotebookTracker } from '@jupyterlab/notebook';
import { ICommandPalette } from '@jupyterlab/apputils';

import { FerretCommandsManager } from './manager';
import { NotebookToolbarExtension } from './toolbar';
import { CellToolbarExtension } from './celltoolbar';

/**
 * The main Ferret extension plugin
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'ferret:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette],
  activate: async (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    palette: ICommandPalette | null
  ) => {
    console.log('JupyterLab extension ferret is activated!');

    // Create the command manager
    const manager = new FerretCommandsManager(app, tracker);

    // Load commands from the server
    await manager.loadCommands();

    // Register commands with JupyterLab
    manager.registerCommands();

    // Add to command palette if available
    if (palette) {
      manager.addToPalette(palette);
    }

    // Add toolbar buttons to notebooks
    const toolbarExtension = new NotebookToolbarExtension(manager);
    app.docRegistry.addWidgetExtension('Notebook', toolbarExtension);

    // Add cell toolbar buttons
    new CellToolbarExtension(manager, tracker);

    console.log('Ferret toolbar buttons added');
  }
};

export default extension;
