/**
 * Cell highlighter for reproducibility staleness visualization
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { StalenessManager } from './stalenessmanager';
import { ReproducibilityMetadataPanel } from './metadatapanel';
import { IReproducibilityMetadata } from './types';

export class ReproducibilityCellHighlighter {
  private _tracker: INotebookTracker;
  private _panel: ReproducibilityMetadataPanel;
  private _stalenessManagers = new Map<string, StalenessManager>();

  constructor(tracker: INotebookTracker, panel: ReproducibilityMetadataPanel) {
    this._tracker = tracker;
    this._panel = panel;
    this._initialize();
  }

  private _initialize(): void {
    this._tracker.currentChanged.connect(this._onNotebookChanged, this);
    this._tracker.activeCellChanged.connect(this._onActiveCellChanged, this);

    if (this._tracker.currentWidget) {
      this._monitorNotebook(this._tracker.currentWidget);
    }
  }

  /**
   * Get or create staleness manager for a notebook
   */
  getStalenessManager(notebook: NotebookPanel): StalenessManager {
    const path = notebook.context.path;
    let manager = this._stalenessManagers.get(path);

    if (!manager) {
      manager = new StalenessManager(notebook);
      this._stalenessManagers.set(path, manager);

      // Listen for staleness changes to update highlighting
      manager.stalenessChanged.connect(() => {
        this._updateAllCells(notebook);
      });

      notebook.disposed.connect(() => {
        manager?.dispose();
        this._stalenessManagers.delete(path);
      });
    }

    return manager;
  }

  private _onNotebookChanged(tracker: INotebookTracker, notebook: NotebookPanel | null): void {
    if (notebook) {
      this._monitorNotebook(notebook);
    }
  }

  /**
   * Get current cell order from notebook (only code cells)
   */
  private _getCurrentCellOrder(notebook: NotebookPanel): string[] {
    const cellOrder: string[] = [];
    const cells = notebook.content.widgets;
    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i];
      if (cell.model.type === 'code') {
        cellOrder.push(cell.model.id);
      }
    }
    return cellOrder;
  }

  private _onActiveCellChanged(tracker: INotebookTracker, cell: Cell | null): void {
    const notebook = tracker.currentWidget;
    if (!notebook) {
      this._panel.clear();
      return;
    }

    if (cell && cell.model.type === 'code') {
      const metadata = cell.model.metadata as any;
      const reproducibilityMetadata = metadata?.flowbook as IReproducibilityMetadata | undefined;
      const cellId = cell.model.id;
      const currentCellOrder = this._getCurrentCellOrder(notebook);

      if (reproducibilityMetadata) {
        this._panel.updateMetadata(reproducibilityMetadata, cellId, currentCellOrder);
      } else {
        this._panel.clear();
      }
    } else {
      this._panel.clear();
    }
  }

  private _monitorNotebook(notebook: NotebookPanel): void {
    this._updateAllCells(notebook);

    notebook.content.model?.cells.changed.connect(() => {
      this._updateAllCells(notebook);
      // Update panel with new cell order when cells are added/removed/reordered
      this._updatePanelWithCurrentCellOrder(notebook);
    });
  }

  /**
   * Update the panel with current cell order (if active cell has metadata)
   */
  private _updatePanelWithCurrentCellOrder(notebook: NotebookPanel): void {
    const activeCell = this._tracker.activeCell;
    if (!activeCell || activeCell.model.type !== 'code') {
      return;
    }

    const metadata = activeCell.model.metadata as any;
    const reproducibilityMetadata = metadata?.flowbook as IReproducibilityMetadata | undefined;

    if (reproducibilityMetadata) {
      const cellId = activeCell.model.id;
      const currentCellOrder = this._getCurrentCellOrder(notebook);
      this._panel.updateMetadata(reproducibilityMetadata, cellId, currentCellOrder);
    }
  }

  private _updateAllCells(notebook: NotebookPanel): void {
    const stalenessManager = this.getStalenessManager(notebook);
    const cells = notebook.content.widgets;

    cells.forEach(cell => {
      if (cell.model.type === 'code') {
        this._updateCell(cell, stalenessManager);
      }
    });
  }

  private _updateCell(cell: Cell, stalenessManager: StalenessManager): void {
    const cellId = cell.model.id;
    const isStale = stalenessManager.isCellStale(cellId);

    console.log(`CellHighlighter: Updating cell ${cellId}, isStale=${isStale}`);

    // Remove existing stale class
    cell.node.classList.remove('flowbook-cell-stale');

    // Add stale class if needed
    if (isStale) {
      cell.node.classList.add('flowbook-cell-stale');
      console.log(`CellHighlighter: Added .flowbook-cell-stale class to cell ${cellId}`);
    } else {
      console.log(`CellHighlighter: Removed .flowbook-cell-stale class from cell ${cellId}`);
    }

    // Update panel if this is the active cell
    if (this._tracker.activeCell === cell) {
      const metadata = cell.model.metadata as any;
      const reproducibilityMetadata = metadata?.flowbook as IReproducibilityMetadata | undefined;
      const notebook = this._tracker.currentWidget;

      if (reproducibilityMetadata && notebook) {
        const currentCellOrder = this._getCurrentCellOrder(notebook);
        this._panel.updateMetadata(reproducibilityMetadata, cellId, currentCellOrder);
      }
    }
  }
}
