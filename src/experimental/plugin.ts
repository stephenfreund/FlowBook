/**
 * Experimental Kernel Plugin - Activates only for experimental_kernel
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { ICommandPalette, IToolbarWidgetRegistry } from '@jupyterlab/apputils';

import { KernelDetector } from '../shared/kerneldetection';
import { FlowbookCommandsManager } from './manager';
import { NotebookToolbarExtension } from './toolbar';
import { CellToolbarExtension } from './celltoolbar';
import { MessagePanel } from '../logpanel';
import { FlowbookMetadataPanel } from './metadatapanel';
import { UnitTestPanel } from './unittestpanel';
import { UnitTestPanelTracker } from './unittesttracker';
import { CellMetadataHighlighter } from './cellhighlighter';
import { ExecutionHookManager } from './executionhook';
import { NotebookHistoryManager } from './history';
import { HistoryPanel } from './historypanel';
import { CellIndexManager } from '../cellindex';

/**
 * Track activation state and manage UI components
 */
class ExperimentalActivationManager {
  private _app: JupyterFrontEnd;
  private _tracker: INotebookTracker;
  private _palette: ICommandPalette | null;
  private _kernelDetector: KernelDetector;

  // Shared managers (always active)
  private _historyManager: NotebookHistoryManager;
  private _cellIndexManager: CellIndexManager;
  private _manager: FlowbookCommandsManager;

  // UI components (created/destroyed based on kernel)
  private _messagePanel: MessagePanel | null = null;
  private _metadataPanel: FlowbookMetadataPanel | null = null;
  private _historyPanel: HistoryPanel | null = null;
  private _unitTestPanel: UnitTestPanel | null = null;

  private _isUIActive = false;

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    palette: ICommandPalette | null,
    toolbarRegistry: IToolbarWidgetRegistry | null
  ) {
    this._app = app;
    this._tracker = tracker;
    this._palette = palette;
    this._kernelDetector = new KernelDetector(tracker);

    // Initialize shared managers
    this._historyManager = new NotebookHistoryManager();
    this._cellIndexManager = new CellIndexManager();
    this._manager = new FlowbookCommandsManager(app, tracker, this._historyManager);

    // Register commands (they will check kernel before executing)
    this._manager.registerCommands();

    // Add toolbar extension (always registered, but buttons check kernel)
    const toolbarExtension = new NotebookToolbarExtension(this._manager);
    app.docRegistry.addWidgetExtension('Notebook', toolbarExtension);

    // Add cell toolbar buttons if registry available
    if (toolbarRegistry) {
      new CellToolbarExtension(this._manager, tracker, toolbarRegistry);
    }

    this._setupKernelChangeListener();
    this._setupNotebookMonitoring();
    this._checkCurrentNotebook();
  }

  private _setupKernelChangeListener(): void {
    this._kernelDetector.kernelChanged.connect((_, info) => {
      if (info.currentKernel === 'experimental_kernel') {
        this._activateUI();
      } else if (info.previousKernel === 'experimental_kernel') {
        this._deactivateUI();
      }
    });

    // Also check when current widget changes
    this._tracker.currentChanged.connect(() => {
      this._checkCurrentNotebook();
    });
  }

  private _setupNotebookMonitoring(): void {
    // Monitor notebooks for history and cell index
    this._tracker.widgetAdded.connect((_, widget: NotebookPanel) => {
      widget.context.ready.then(() => {
        this._historyManager.startMonitoring(widget.context.path, widget);
        this._cellIndexManager.startMonitoring(widget.context.path, widget);
      });

      widget.disposed.connect(() => {
        this._historyManager.stopMonitoring(widget.context.path);
        this._cellIndexManager.stopMonitoring(widget.context.path);
      });
    });

    // Start monitoring any already open notebooks
    this._tracker.forEach((widget: NotebookPanel) => {
      widget.context.ready.then(() => {
        this._historyManager.startMonitoring(widget.context.path, widget);
        this._cellIndexManager.startMonitoring(widget.context.path, widget);
      });

      widget.disposed.connect(() => {
        this._historyManager.stopMonitoring(widget.context.path);
        this._cellIndexManager.stopMonitoring(widget.context.path);
      });
    });
  }

  private _checkCurrentNotebook(): void {
    const notebook = this._tracker.currentWidget;
    if (notebook) {
      notebook.sessionContext.ready.then(() => {
        if (this._kernelDetector.isExperimentalKernel(notebook)) {
          this._activateUI();
        } else {
          this._deactivateUI();
        }
      });
    }
  }

  private _activateUI(): void {
    if (this._isUIActive) {
      return;
    }

    console.log('Experimental Plugin: Activating UI for experimental_kernel');

    // Add to command palette if available
    if (this._palette) {
      this._manager.addToPalette(this._palette);
    }

    // Add to context menu
    this._manager.addToContextMenu(this._tracker);

    // Create and add the message panel to the right area
    this._messagePanel = new MessagePanel();
    this._app.shell.add(this._messagePanel, 'right', { rank: 500 });

    // Create and add the metadata panel to the right area
    this._metadataPanel = new FlowbookMetadataPanel();
    this._app.shell.add(this._metadataPanel, 'right', { rank: 501 });

    // Create and add the history panel to the right area
    this._historyPanel = new HistoryPanel(this._tracker, this._historyManager);
    this._app.shell.add(this._historyPanel, 'right', { rank: 502 });

    // Create and add the unit test panel to the right area
    this._unitTestPanel = new UnitTestPanel(this._app, this._tracker);
    this._app.shell.add(this._unitTestPanel, 'right', { rank: 503 });

    // Create cell metadata highlighter for visual indicators
    new CellMetadataHighlighter(this._tracker, this._metadataPanel);

    // Create unit test panel tracker for monitoring cell selection
    new UnitTestPanelTracker(this._tracker, this._unitTestPanel);

    // Create execution hook manager for auto-generating code from string specs
    new ExecutionHookManager(this._app, this._tracker, this._manager);

    this._isUIActive = true;
    console.log('Experimental Plugin: UI activated');
  }

  private _deactivateUI(): void {
    if (!this._isUIActive) {
      return;
    }

    console.log('Experimental Plugin: Deactivating UI');

    // Remove panels
    if (this._messagePanel) {
      this._messagePanel.dispose();
      this._messagePanel = null;
    }
    if (this._metadataPanel) {
      this._metadataPanel.dispose();
      this._metadataPanel = null;
    }
    if (this._historyPanel) {
      this._historyPanel.dispose();
      this._historyPanel = null;
    }
    if (this._unitTestPanel) {
      this._unitTestPanel.dispose();
      this._unitTestPanel = null;
    }

    this._isUIActive = false;
    console.log('Experimental Plugin: UI deactivated');
  }
}

/**
 * Experimental Plugin definition
 */
export const experimentalPlugin: JupyterFrontEndPlugin<void> = {
  id: 'flowbook:experimental',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette, IToolbarWidgetRegistry],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    palette: ICommandPalette | null,
    toolbarRegistry: IToolbarWidgetRegistry | null
  ) => {
    console.log('Experimental Plugin: Extension registered (will activate UI when experimental_kernel is used)');
    new ExperimentalActivationManager(app, tracker, palette, toolbarRegistry);
  }
};
