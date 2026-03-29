/**
 * Cell highlighter for reproducibility staleness visualization.
 *
 * Responsibilities:
 * - CSS class management (stale/unexecuted/error)
 * - StalenessManager lifecycle (per notebook)
 * - Active cell change → metadata panel updates
 * - Kernel restart → clear metadata
 * - Coordination: delegates to StalenessNoticeManager and ViolationNoticeManager
 */

import {
  INotebookTracker,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { StalenessManager } from './stalenessmanager';
import { ReproducibilityMetadataPanel } from './metadatapanel';
import { DependenciesPanel, ICellGraphData } from './dependenciespanel';
import { IReproducibilityMetadata, IPredicateViolation } from './types';
import { StalenessNoticeManager } from './stalenessnotice';
import { ViolationNoticeManager } from './violationnotice';
import { getCodeCellOrder } from '../cellindexutils';
import { indexToAlpha } from '../cellindexutils';

export class ReproducibilityCellHighlighter {
  private _tracker: INotebookTracker;
  private _panel: ReproducibilityMetadataPanel;
  private _dependenciesPanel: DependenciesPanel | null = null;
  private _depPanelFrameId: number | null = null;
  private _stalenessManagers = new Map<string, StalenessManager>();
  private _pendingRestartUpdate = new Set<string>();
  private _executedInSession = new Map<string, Set<string>>();
  private _monitoredNotebooks = new Set<string>();
  private _stalenessNotice = new StalenessNoticeManager();
  private _violationNotice = new ViolationNoticeManager();
  private _isDisposed = false;

  constructor(tracker: INotebookTracker, panel: ReproducibilityMetadataPanel) {
    this._tracker = tracker;
    this._panel = panel;
    this._initialize();
  }

  setDependenciesPanel(panel: DependenciesPanel): void {
    this._dependenciesPanel = panel;
  }

  /**
   * Schedule a dependency graph refresh on the next animation frame.
   */
  refreshDependencies(): void {
    if (this._depPanelFrameId !== null) {
      cancelAnimationFrame(this._depPanelFrameId);
    }
    this._depPanelFrameId = requestAnimationFrame(() => {
      this._depPanelFrameId = null;
      const notebook = this._tracker.currentWidget;
      if (notebook) {
        const stalenessManager = this.getStalenessManager(notebook);
        const cellOrder = getCodeCellOrder(notebook);
        this._updateDependenciesPanel(notebook, stalenessManager, cellOrder);
      }
    });
  }

  /**
   * Update the status display in the metadata panel header.
   */
  updateStatus(icon: string, text: string, cellId?: string): void {
    this._panel.updateStatus(icon, text, cellId);
  }

  /**
   * Get or create staleness manager for a notebook.
   */
  getStalenessManager(notebook: NotebookPanel): StalenessManager {
    const path = notebook.context.path;
    let manager = this._stalenessManagers.get(path);

    if (!manager) {
      manager = new StalenessManager(notebook);
      this._stalenessManagers.set(path, manager);

      manager.stalenessChanged.connect(() => {
        this._updateAllCells(notebook);
      });

      notebook.disposed.connect(() => {
        manager?.dispose();
        this._stalenessManagers.delete(path);
        this._monitoredNotebooks.delete(path);
      });
    }

    return manager;
  }

  /**
   * Update cell highlighting and notices.
   * Single entry point for all cell rendering updates.
   */
  updateCell(
    cell: Cell,
    stalenessManager: StalenessManager,
    cellOrder: string[],
    notebookPath: string
  ): void {
    const cellId = cell.model.id;
    const isStale = stalenessManager.isCellStale(cellId);
    const executedInSession = this._executedInSession.get(notebookPath);
    const hasFlowbookMetadata =
      cell.model.getMetadata('flowbook') !== undefined;
    const wasExecutedInSession =
      (executedInSession && executedInSession.has(cellId)) ||
      hasFlowbookMetadata;
    const isUnexecuted = !wasExecutedInSession;

    const codeModel = cell.model as ICodeCellModel;
    const source = codeModel.sharedModel.getSource();
    const isEmpty = !source || source.trim() === '';

    // Remove existing highlight classes
    cell.node.classList.remove('flowbook-cell-stale');
    cell.node.classList.remove('flowbook-cell-unexecuted');
    cell.node.classList.remove('flowbook-cell-error');

    if (isEmpty) {
      this._stalenessNotice.updateStalenessNotice(
        cell,
        false,
        stalenessManager,
        cellOrder
      );
    } else if (isStale) {
      cell.node.classList.add('flowbook-cell-stale');
      this._stalenessNotice.updateStalenessNotice(
        cell,
        true,
        stalenessManager,
        cellOrder
      );
    } else if (isUnexecuted) {
      cell.node.classList.add('flowbook-cell-unexecuted');
      this._stalenessNotice.updateStalenessNotice(
        cell,
        false,
        stalenessManager,
        cellOrder
      );
    } else {
      this._stalenessNotice.updateStalenessNotice(
        cell,
        false,
        stalenessManager,
        cellOrder
      );
    }

    // Add/remove violation notice output
    const hasViolations = this._violationNotice.updateViolationNotice(
      cell,
      cellOrder
    );
    if (hasViolations) {
      cell.node.classList.add('flowbook-cell-error');
    }

    // Recompute staleness metadata messages with current @A references
    this._stalenessNotice.updateStalenessMetadata(cell, cellOrder);

    // Update panel if this is the active cell
    if (this._tracker.activeCell === cell) {
      const reproducibilityMetadata = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      const notebook = this._tracker.currentWidget;

      if (reproducibilityMetadata && notebook) {
        this._panel.updateMetadata(reproducibilityMetadata, cellId, cellOrder);
      }
    }
  }

  /**
   * Disconnect all signal listeners and clean up.
   */
  dispose(): void {
    if (this._isDisposed) {
      return;
    }
    this._isDisposed = true;

    this._tracker.currentChanged.disconnect(this._onNotebookChanged, this);
    this._tracker.activeCellChanged.disconnect(this._onActiveCellChanged, this);
    NotebookActions.executed.disconnect(this._onExecuted, this);

    if (this._depPanelFrameId !== null) {
      cancelAnimationFrame(this._depPanelFrameId);
    }

    for (const manager of this._stalenessManagers.values()) {
      manager.dispose();
    }
    this._stalenessManagers.clear();
    this._monitoredNotebooks.clear();
    this._executedInSession.clear();
  }

  private _initialize(): void {
    this._tracker.currentChanged.connect(this._onNotebookChanged, this);
    this._tracker.activeCellChanged.connect(this._onActiveCellChanged, this);
    NotebookActions.executed.connect(this._onExecuted, this);

    if (this._tracker.currentWidget) {
      this._monitorNotebook(this._tracker.currentWidget);
    }
  }

  private _onNotebookChanged(
    _tracker: INotebookTracker,
    notebook: NotebookPanel | null
  ): void {
    if (notebook) {
      this._monitorNotebook(notebook);
    }
  }

  private _onActiveCellChanged(
    tracker: INotebookTracker,
    cell: Cell | null
  ): void {
    const notebook = tracker.currentWidget;
    if (!notebook) {
      this._panel.clear();
      return;
    }

    if (cell && cell.model.type === 'code') {
      const reproducibilityMetadata = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      const cellId = cell.model.id;
      const currentCellOrder = getCodeCellOrder(notebook);

      if (reproducibilityMetadata) {
        this._panel.updateMetadata(
          reproducibilityMetadata,
          cellId,
          currentCellOrder
        );
      } else {
        this._panel.clear();
      }
    } else {
      this._panel.clear();
    }
  }

  /**
   * Track executed cells across all notebooks.
   * Uses a named method so it can be disconnected on dispose.
   */
  private _onExecuted(_sender: any, args: { notebook: any; cell: Cell }): void {
    // Find the notebook panel that owns this notebook widget
    const panel = this._tracker.currentWidget;
    if (!panel || panel.content !== args.notebook) {
      return;
    }
    const path = panel.context.path;
    let executed = this._executedInSession.get(path);
    if (!executed) {
      executed = new Set<string>();
      this._executedInSession.set(path, executed);
    }
    executed.add(args.cell.model.id);
  }

  /**
   * Start monitoring a notebook for cell changes and kernel restarts.
   * Idempotent per notebook path — duplicate calls are no-ops.
   */
  private _monitorNotebook(notebook: NotebookPanel): void {
    const path = notebook.context.path;
    if (this._monitoredNotebooks.has(path)) {
      // Already monitored — just update cells for current state
      this._updateAllCells(notebook);
      return;
    }
    this._monitoredNotebooks.add(path);
    this._updateAllCells(notebook);

    notebook.content.model?.cells.changed.connect(() => {
      this._updateAllCells(notebook);
      this._updatePanelWithCurrentCellOrder(notebook);
    });

    // Listen for kernel restart
    notebook.sessionContext.statusChanged.connect((_, status) => {
      if (status === 'restarting' || status === 'autorestarting') {
        this._pendingRestartUpdate.add(path);
        this._executedInSession.delete(path);
        this._clearAllFlowbookMetadata(notebook);
      } else if (status === 'idle' && this._pendingRestartUpdate.has(path)) {
        this._pendingRestartUpdate.delete(path);
        this._updateAllCells(notebook);
      }
    });
  }

  private _updatePanelWithCurrentCellOrder(notebook: NotebookPanel): void {
    const activeCell = this._tracker.activeCell;
    if (!activeCell || activeCell.model.type !== 'code') {
      return;
    }

    const reproducibilityMetadata = activeCell.model.getMetadata('flowbook') as
      | IReproducibilityMetadata
      | undefined;

    if (reproducibilityMetadata) {
      const cellId = activeCell.model.id;
      const currentCellOrder = getCodeCellOrder(notebook);
      this._panel.updateMetadata(
        reproducibilityMetadata,
        cellId,
        currentCellOrder
      );
    }
  }

  private _updateAllCells(notebook: NotebookPanel): void {
    const stalenessManager = this.getStalenessManager(notebook);
    const cellOrder = getCodeCellOrder(notebook);
    const cells = notebook.content.widgets;
    const path = notebook.context.path;

    cells.forEach(cell => {
      if (cell.model.type === 'code') {
        this.updateCell(cell, stalenessManager, cellOrder, path);
      }
    });

    this.refreshDependencies();
  }

  /**
   * Collect graph data from all code cells and update the dependencies panel.
   */
  private _updateDependenciesPanel(
    notebook: NotebookPanel,
    stalenessManager: StalenessManager,
    cellOrder: string[]
  ): void {
    if (!this._dependenciesPanel) {
      return;
    }

    const cellData: ICellGraphData[] = [];
    const cells = notebook.content.widgets;

    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i];
      if (cell.model.type !== 'code') {
        continue;
      }

      const cellId = cell.model.id;
      const orderIndex = cellOrder.indexOf(cellId);
      if (orderIndex < 0) {
        continue;
      }

      const flowbook = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      const violations =
        (cell.model.getMetadata('flowbook_violations') as
          | IPredicateViolation[]
          | undefined) || [];

      let label: string;
      try {
        label = indexToAlpha(orderIndex);
      } catch {
        label = cellId;
      }

      cellData.push({
        cellId,
        index: orderIndex,
        label,
        readLocs: flowbook?.read_locs || [],
        writeLocs: flowbook?.write_locs || [],
        isStale: stalenessManager.isCellStale(cellId),
        isExecuted: flowbook !== undefined,
        hasError: violations.length > 0,
        violations
      });
    }

    cellData.sort((a, b) => a.index - b.index);
    this._dependenciesPanel.updateGraph(cellData);
  }

  /**
   * Clear all flowbook metadata from cells on kernel restart.
   */
  private _clearAllFlowbookMetadata(notebook: NotebookPanel): void {
    const stalenessManager = this.getStalenessManager(notebook);
    stalenessManager.clear();

    const cells = notebook.content.widgets;
    cells.forEach(cell => {
      if (cell.model.type !== 'code') {
        return;
      }

      cell.model.deleteMetadata('flowbook');
      cell.model.deleteMetadata('flowbook_staleness');
      cell.model.deleteMetadata('flowbook_violation');
      cell.model.deleteMetadata('flowbook_violations');

      const codeModel = cell.model as ICodeCellModel;
      const outputs = codeModel.outputs;
      const cleanOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const meta = (out as any).metadata || {};
        if (
          !meta.flowbook_staleness_notice &&
          !meta.flowbook_violation_notice
        ) {
          cleanOutputs.push(out);
        }
      }
      if (cleanOutputs.length !== outputs.length) {
        outputs.fromJSON(cleanOutputs);
      }

      cell.node.classList.remove('flowbook-stale-cell');
      cell.node.classList.remove('flowbook-unexecuted-cell');
      cell.node.classList.remove('flowbook-cell-stale');
      cell.node.classList.remove('flowbook-cell-unexecuted');
      cell.node.classList.remove('flowbook-cell-error');
    });
  }
}
