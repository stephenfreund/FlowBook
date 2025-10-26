/**
 * Ferret JupyterLab Frontend Extension
 * Adds toolbar buttons for ferret commands with kernel communication
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { INotebookTracker } from '@jupyterlab/notebook';
import { ICommandPalette, IToolbarWidgetRegistry } from '@jupyterlab/apputils';

import { FerretCommandsManager } from './manager';
import { NotebookToolbarExtension } from './toolbar';
import { CellToolbarExtension } from './celltoolbar';
import { MessagePanel } from './panel';
import { FerretMetadataPanel } from './metadatapanel';
import { CellMetadataHighlighter } from './cellhighlighter';

/**
 * The main Ferret extension plugin
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'data_ferret:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette, IToolbarWidgetRegistry],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    palette: ICommandPalette | null,
    toolbarRegistry: IToolbarWidgetRegistry | null
  ) => {
    console.log('JupyterLab extension ferret is activated!');

    // Create the command manager
    const manager = new FerretCommandsManager(app, tracker);

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
    if (toolbarRegistry) {
      new CellToolbarExtension(manager, tracker, toolbarRegistry);
    }

    // Create and add the message panel to the right area
    const messagePanel = new MessagePanel();
    app.shell.add(messagePanel, 'right', { rank: 500 });

    // Create and add the metadata panel to the right area
    const metadataPanel = new FerretMetadataPanel();
    app.shell.add(metadataPanel, 'right', { rank: 501 });

    // Create cell metadata highlighter for visual indicators
    new CellMetadataHighlighter(tracker, metadataPanel);

    console.log('Ferret toolbar buttons, panels, and cell highlighter added');

  }
};

export default extension;
