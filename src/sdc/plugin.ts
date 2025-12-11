/**
 * SDC Kernel Plugin - Activates only for ferret_sdc_kernel
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';

import { KernelDetector } from '../shared/kerneldetection';
import { MessagePanel } from '../logpanel';
import { SDCMetadataPanel } from './metadatapanel';
import { SDCCellHighlighter } from './cellhighlighter';
import { SDCExecutionHookManager } from './executionhook';
import { CellIndexManager } from '../cellindex';

/**
 * Track activation state per notebook
 */
class SDCActivationManager {
  private _app: JupyterFrontEnd;
  private _tracker: INotebookTracker;
  private _kernelDetector: KernelDetector;
  private _panel: SDCMetadataPanel | null = null;
  private _highlighter: SDCCellHighlighter | null = null;
  private _cellIndexManager: CellIndexManager;
  private _isActive = false;
  private _activeNotebookPath: string | null = null;

  constructor(app: JupyterFrontEnd, tracker: INotebookTracker) {
    this._app = app;
    this._tracker = tracker;
    this._kernelDetector = new KernelDetector(tracker);
    this._cellIndexManager = new CellIndexManager();

    this._setupKernelChangeListener();
    this._checkCurrentNotebook();
  }

  private _setupKernelChangeListener(): void {
    this._kernelDetector.kernelChanged.connect((_, info) => {
      console.log(`SDC Plugin: Kernel changed from ${info.previousKernel} to ${info.currentKernel}`);
      if (info.currentKernel === 'ferret_sdc_kernel') {
        this._activate();
      } else if (info.previousKernel === 'ferret_sdc_kernel') {
        this._deactivate();
      }
    });

    // Also check when current widget changes
    this._tracker.currentChanged.connect(() => {
      console.log('SDC Plugin: Current notebook changed, checking kernel...');
      this._checkCurrentNotebook();
    });
  }

  private _checkCurrentNotebook(): void {
    const notebook = this._tracker.currentWidget;
    if (notebook) {
      const kernelName = notebook.sessionContext.session?.kernel?.name;
      console.log(`SDC Plugin: Checking notebook, kernel = ${kernelName}`);

      // Wait for session to be ready
      notebook.sessionContext.ready.then(() => {
        const isSDC = this._kernelDetector.isSDCKernel(notebook);
        const currentKernelName = notebook.sessionContext.session?.kernel?.name;
        console.log(`SDC Plugin: Session ready, kernel = ${currentKernelName}, isSDC = ${isSDC}`);

        if (isSDC) {
          this._activate();
        } else {
          this._deactivate();
        }
      });
    }
  }

  private _activate(): void {
    if (this._isActive) {
      return;
    }

    console.log('SDC Plugin: Activating for ferret_sdc_kernel');

    // Create panel
    this._panel = new SDCMetadataPanel();
    this._app.shell.add(this._panel, 'right', { rank: 510 });

    // Create highlighter
    this._highlighter = new SDCCellHighlighter(this._tracker, this._panel);

    // Create execution hook
    new SDCExecutionHookManager(
      this._app,
      this._tracker,
      this._highlighter
    );

    // Start cell index overlays for current notebook
    const widget = this._tracker.currentWidget;
    if (widget) {
      this._activeNotebookPath = widget.context.path;
      this._cellIndexManager.startMonitoring(this._activeNotebookPath, widget);
    }

    this._isActive = true;
    console.log('SDC Plugin: Activated');
  }

  private _deactivate(): void {
    if (!this._isActive) {
      return;
    }

    console.log('SDC Plugin: Deactivating');

    // Stop cell index overlays
    if (this._activeNotebookPath) {
      this._cellIndexManager.stopMonitoring(this._activeNotebookPath);
      this._activeNotebookPath = null;
    }

    // Remove panel
    if (this._panel) {
      this._panel.dispose();
      this._panel = null;
    }

    // Highlighter cleanup
    this._highlighter = null;

    this._isActive = false;
    console.log('SDC Plugin: Deactivated');
  }
}

/**
 * SDC Plugin definition
 */
export const sdcPlugin: JupyterFrontEndPlugin<void> = {
  id: 'data_ferret:sdc',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker) => {
    console.log('SDC Plugin: Extension registered (will activate when ferret_sdc_kernel is used)');
    new SDCActivationManager(app, tracker);
  }
};
