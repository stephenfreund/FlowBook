/**
 * FlowBook Kernel Plugin - Activates only for flowbook_kernel (reproducibility)
 */

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';

import { KernelDetector } from '../shared/kerneldetection';
import { ReproducibilityMetadataPanel } from './metadatapanel';
import { DependenciesPanel } from './dependenciespanel';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { ReproducibilityExecutionHookManager } from './executionhook';
import { CellIndexManager } from '../cellindex';
import { IPredicateViolation } from './types';
import { FlowbookToolbarExtension } from './toolbar';

/**
 * Register the flowbook:exec-restore command at plugin startup.
 *
 * The command is registered once, but its isEnabled check gates on:
 * 1. Current kernel is flowbook_kernel
 * 2. Active cell has a no_read_before_write predicate violation in flowbook_violations
 *
 * The context menu item is defined declaratively in schema/plugin.json.
 */
function registerExecRestoreCommand(
  app: JupyterFrontEnd,
  tracker: INotebookTracker,
  kernelDetector: KernelDetector
): void {
  const commandId = 'flowbook:exec-restore';

  app.commands.addCommand(commandId, {
    label: 'Run with upstream state',
    isEnabled: () => {
      try {
        const panel = tracker.currentWidget;
        if (!panel) {
          return false;
        }
        // Gate on flowbook_kernel
        if (!kernelDetector.isFlowbookKernel(panel)) {
          return false;
        }
        const activeCell = panel.content.activeCell;
        if (!activeCell || activeCell.model.type !== 'code') {
          return false;
        }
        const violations = activeCell.model.getMetadata(
          'flowbook_violations'
        ) as IPredicateViolation[] | undefined;
        return (
          violations !== undefined &&
          violations.some(v => v.predicate === 'no_read_before_write')
        );
      } catch (e) {
        console.error('FlowBook exec-restore isEnabled error:', e);
        return false;
      }
    },
    isVisible: () => {
      try {
        const panel = tracker.currentWidget;
        if (!panel) {
          return false;
        }
        return kernelDetector.isFlowbookKernel(panel);
      } catch {
        return false;
      }
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

      // Send exec_restore via protocol (deprecated — kernel shows error message).
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
}

/**
 * Track activation state per notebook
 */
class FlowbookActivationManager {
  private _app: JupyterFrontEnd;
  private _tracker: INotebookTracker;
  private _kernelDetector: KernelDetector;
  private _panel: ReproducibilityMetadataPanel | null = null;
  private _dependenciesPanel: DependenciesPanel | null = null;
  private _highlighter: ReproducibilityCellHighlighter | null = null;
  private _executionHook: ReproducibilityExecutionHookManager | null = null;
  private _cellIndexManager: CellIndexManager;
  private _toolbarExtension: FlowbookToolbarExtension;
  private _isActive = false;
  private _activeNotebookPath: string | null = null;

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    toolbarExtension: FlowbookToolbarExtension
  ) {
    this._app = app;
    this._tracker = tracker;
    this._kernelDetector = new KernelDetector(tracker);
    this._cellIndexManager = new CellIndexManager();
    this._toolbarExtension = toolbarExtension;

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

    // Create panels
    this._panel = new ReproducibilityMetadataPanel();
    this._app.shell.add(this._panel, 'right', { rank: 510 });

    this._dependenciesPanel = new DependenciesPanel();
    this._app.shell.add(this._dependenciesPanel, 'right', { rank: 520 });

    // Create highlighter (dependencies panel must be set before _initialize runs)
    this._highlighter = new ReproducibilityCellHighlighter(
      this._tracker,
      this._panel
    );
    this._highlighter.setDependenciesPanel(this._dependenciesPanel);

    // Set highlighter on toolbar extension so it can access staleness manager
    this._toolbarExtension.setHighlighter(this._highlighter);

    // Create execution hook (store reference for sendCommand access)
    this._executionHook = new ReproducibilityExecutionHookManager(
      this._app,
      this._tracker,
      this._highlighter
    );

    // Trigger initial dependency graph update (highlighter's constructor may
    // have called _updateAllCells before the dependencies panel was set)
    this._highlighter.refreshDependencies();

    // Start cell index overlays for current notebook
    const widget = this._tracker.currentWidget;
    if (widget) {
      this._activeNotebookPath = widget.context.path;
      this._cellIndexManager.startMonitoring(this._activeNotebookPath, widget);

      // Sync initial staleness state from kernel
      this._syncInitialState(widget);
    }

    this._isActive = true;
    console.log('FlowBook Plugin: Activated');
  }

  /**
   * Sync initial staleness state from kernel on notebook load.
   * Sends cell order and requests current state via comm channel.
   */
  private _syncInitialState(panel: NotebookPanel): void {
    if (!this._executionHook) {
      console.log('FlowBook Plugin: No execution hook available for sync');
      return;
    }

    // Get cell order
    const cells = panel.content.widgets;
    const cellOrder = cells
      .filter((cell: Cell) => cell.model.type === 'code')
      .map((cell: Cell) => cell.model.id);

    if (cellOrder.length === 0) {
      return;
    }

    // Send cell order first, then request sync via comm
    this._executionHook.sendCommand({
      type: 'notebook_structure',
      cell_order: cellOrder
    });
    this._executionHook.sendCommand({ type: 'sync' });

    console.log('FlowBook Plugin: Sent initial sync via comm');
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

    // Remove panels
    if (this._panel) {
      this._panel.dispose();
      this._panel = null;
    }
    if (this._dependenciesPanel) {
      this._dependenciesPanel.dispose();
      this._dependenciesPanel = null;
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

    const kernelDetector = new KernelDetector(tracker);

    // Register exec-restore command at startup (context menu item in schema)
    registerExecRestoreCommand(app, tracker, kernelDetector);

    // Create and register toolbar extension
    const toolbarExtension = new FlowbookToolbarExtension(kernelDetector);
    app.docRegistry.addWidgetExtension('Notebook', toolbarExtension);

    new FlowbookActivationManager(app, tracker, toolbarExtension);
  }
};
