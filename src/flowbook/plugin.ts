/**
 * FlowBook Kernel Plugin - Activates only for flowbook_kernel (reproducibility)
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';

import { KernelDetector } from '../shared/kerneldetection';
import { ReproducibilityMetadataPanel } from './metadatapanel';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { ReproducibilityExecutionHookManager } from './executionhook';
import { CellIndexManager } from '../cellindex';

/**
 * Track activation state per notebook
 */
class FlowbookActivationManager {
  private _app: JupyterFrontEnd;
  private _tracker: INotebookTracker;
  private _kernelDetector: KernelDetector;
  private _panel: ReproducibilityMetadataPanel | null = null;
  private _highlighter: ReproducibilityCellHighlighter | null = null;
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
      console.log(`FlowBook Plugin: Kernel changed from ${info.previousKernel} to ${info.currentKernel}`);
      if (info.currentKernel === 'flowbook_kernel') {
        this._activate();
      } else if (info.previousKernel === 'flowbook_kernel') {
        this._deactivate();
      }
    });

    // Also check when current widget changes
    this._tracker.currentChanged.connect(() => {
      console.log('FlowBook Plugin: Current notebook changed, checking kernel...');
      this._checkCurrentNotebook();
    });
  }

  private _checkCurrentNotebook(): void {
    const notebook = this._tracker.currentWidget;
    if (notebook) {
      const kernelName = notebook.sessionContext.session?.kernel?.name;
      console.log(`FlowBook Plugin: Checking notebook, kernel = ${kernelName}`);

      // Wait for session to be ready
      notebook.sessionContext.ready.then(() => {
        const isFlowbook = this._kernelDetector.isFlowbookKernel(notebook);
        const currentKernelName = notebook.sessionContext.session?.kernel?.name;
        console.log(`FlowBook Plugin: Session ready, kernel = ${currentKernelName}, isFlowbook = ${isFlowbook}`);

        if (isFlowbook) {
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

    console.log('FlowBook Plugin: Activating for flowbook_kernel');

    // Create panel
    this._panel = new ReproducibilityMetadataPanel();
    this._app.shell.add(this._panel, 'right', { rank: 510 });

    // Create highlighter
    this._highlighter = new ReproducibilityCellHighlighter(this._tracker, this._panel);

    // Create execution hook
    new ReproducibilityExecutionHookManager(
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
    console.log('FlowBook Plugin: Activated');
  }

  private _deactivate(): void {
    if (!this._isActive) {
      return;
    }

    console.log('FlowBook Plugin: Deactivating');

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
    console.log('FlowBook Plugin: Deactivated');
  }
}

/**
 * FlowBook Plugin definition
 */
export const flowbookPlugin: JupyterFrontEndPlugin<void> = {
  id: 'flowbook:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker) => {
    console.log('FlowBook Plugin: Extension registered (will activate when flowbook_kernel is used)');
    new FlowbookActivationManager(app, tracker);
  }
};
