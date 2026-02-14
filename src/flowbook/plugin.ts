/**
 * FlowBook Kernel Plugin - Activates only for flowbook_kernel (reproducibility)
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';
import { IDisposable } from '@lumino/disposable';

import { KernelDetector } from '../shared/kerneldetection';
import { ReproducibilityMetadataPanel } from './metadatapanel';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { ReproducibilityExecutionHookManager } from './executionhook';
import { CellIndexManager } from '../cellindex';
import { IReproducibilityMetadata } from './types';

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
  private _execRestoreDisposables: IDisposable[] = [];

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
      console.log(
        `FlowBook Plugin: Kernel changed from ${info.previousKernel} to ${info.currentKernel}`
      );
      if (info.currentKernel === 'flowbook_kernel') {
        this._activate();
      } else if (info.previousKernel === 'flowbook_kernel') {
        this._deactivate();
      }
    });

    // Also check when current widget changes
    this._tracker.currentChanged.connect(() => {
      console.log(
        'FlowBook Plugin: Current notebook changed, checking kernel...'
      );
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
        console.log(
          `FlowBook Plugin: Session ready, kernel = ${currentKernelName}, isFlowbook = ${isFlowbook}`
        );

        if (isFlowbook) {
          this._activate();
        } else {
          this._deactivate();
        }
      });

      // Also listen for status changes in case kernel starts after ready
      notebook.sessionContext.statusChanged.connect(() => {
        const isFlowbook = this._kernelDetector.isFlowbookKernel(notebook);
        const currentKernelName = notebook.sessionContext.session?.kernel?.name;
        console.log(
          `FlowBook Plugin: Status changed, kernel = ${currentKernelName}, isFlowbook = ${isFlowbook}`
        );

        if (isFlowbook && !this._isActive) {
          this._activate();
        } else if (!isFlowbook && this._isActive) {
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
    this._highlighter = new ReproducibilityCellHighlighter(
      this._tracker,
      this._panel
    );

    // Create execution hook
    new ReproducibilityExecutionHookManager(
      this._app,
      this._tracker,
      this._highlighter
    );

    // Register exec-restore command and context menu
    this._registerExecRestoreCommand();

    // Start cell index overlays for current notebook
    const widget = this._tracker.currentWidget;
    if (widget) {
      this._activeNotebookPath = widget.context.path;
      this._cellIndexManager.startMonitoring(this._activeNotebookPath, widget);
    }

    this._isActive = true;
    console.log('FlowBook Plugin: Activated');
  }

  /**
   * Register the flowbook:exec-restore command and context menu item.
   *
   * The command sends %exec_restore <cellId> silently, then triggers
   * notebook:run-cell to re-execute the active cell from its prefix
   * checkpoint. The context menu item is only visible when the active
   * cell has cell_is_contaminated === true in its flowbook metadata.
   */
  private _registerExecRestoreCommand(): void {
    const commandId = 'flowbook:exec-restore';
    const tracker = this._tracker;
    const app = this._app;

    const commandDisposable = app.commands.addCommand(commandId, {
      label: 'Restore from checkpoint',
      isVisible: () => {
        const panel = tracker.currentWidget;
        if (!panel) {
          return false;
        }
        const activeCell = panel.content.activeCell;
        if (!activeCell || activeCell.model.type !== 'code') {
          return false;
        }
        const metadata = activeCell.model.metadata as any;
        const meta = metadata?.flowbook as
          | IReproducibilityMetadata
          | undefined;
        return meta?.cell_is_contaminated === true;
      },
      execute: async () => {
        const panel = tracker.currentWidget;
        if (!panel) {
          return;
        }
        const activeCell = panel.content.activeCell;
        if (!activeCell || activeCell.model.type !== 'code') {
          return;
        }
        const cellId = activeCell.model.id;
        const session = panel.sessionContext.session;
        if (!session || !session.kernel) {
          return;
        }

        // Send %exec_restore magic silently — sets the pending flag in the kernel.
        // The kernel's ZMQ queue ensures ordering:
        //   1. %exec_restore <cell_id>
        //   2. %notebook_structure <order>  (sent by _onExecutionScheduled)
        //   3. Cell code                    (_do_execute_impl consumes the flag)
        const future = session.kernel.requestExecute({
          code: `%exec_restore ${cellId}`,
          silent: true,
          store_history: false
        });
        await future.done;

        // Now trigger cell execution via the standard notebook command.
        // The right-click context menu already makes the cell active,
        // so notebook:run-cell targets the correct cell.
        await app.commands.execute('notebook:run-cell');
      }
    });

    const menuDisposable = app.contextMenu.addItem({
      command: commandId,
      selector: '.jp-Cell.jp-CodeCell',
      rank: 0
    });

    this._execRestoreDisposables.push(commandDisposable, menuDisposable);
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

    // Remove exec-restore command and context menu item
    for (const d of this._execRestoreDisposables) {
      d.dispose();
    }
    this._execRestoreDisposables = [];

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
    console.log(
      'FlowBook Plugin: Extension registered (will activate when flowbook_kernel is used)'
    );
    new FlowbookActivationManager(app, tracker);
  }
};
