/**
 * Cell metadata highlighter for Ferret inspection data
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { IFerretMetadata } from './types';
import { FerretMetadataPanel } from './metadatapanel';

/**
 * Class that monitors notebook cells and applies visual indicators
 * based on Ferret inspection metadata
 */
export class CellMetadataHighlighter {
  private tracker: INotebookTracker;
  private panel: FerretMetadataPanel;

  constructor(tracker: INotebookTracker, panel: FerretMetadataPanel) {
    this.tracker = tracker;
    this.panel = panel;
    this.initialize();
  }

  /**
   * Initialize the highlighter
   */
  private initialize(): void {
    // Update cells when the current notebook changes
    this.tracker.currentChanged.connect(this.onNotebookChanged, this);

    // Update cells for the current notebook if one is active
    if (this.tracker.currentWidget) {
      this.monitorNotebook(this.tracker.currentWidget);
    }

    // Monitor active cell changes
    this.tracker.activeCellChanged.connect(this.onActiveCellChanged, this);
  }

  /**
   * Handle notebook change events
   */
  private onNotebookChanged(
    tracker: INotebookTracker,
    notebook: NotebookPanel | null
  ): void {
    if (notebook) {
      this.monitorNotebook(notebook);
    }
  }

  /**
   * Handle active cell change events
   */
  private onActiveCellChanged(
    tracker: INotebookTracker,
    cell: Cell | null
  ): void {
    if (cell && cell.model.type === 'code') {
      const metadata = cell.model.metadata as any;
      const ferretMetadata = metadata?.ferret as IFerretMetadata | undefined;

      if (ferretMetadata && (ferretMetadata.inspect || ferretMetadata.profile)) {
        this.panel.updateMetadata(ferretMetadata);
      } else {
        this.panel.clear();
      }
    } else {
      this.panel.clear();
    }
  }

  /**
   * Monitor a notebook for cell changes
   */
  private monitorNotebook(notebook: NotebookPanel): void {
    // Update all cells initially
    this.updateAllCells(notebook);

    // Watch for content changes (cell additions, deletions, metadata changes)
    notebook.content.model?.cells.changed.connect(() => {
      this.updateAllCells(notebook);
    });

    // Watch for metadata changes on individual cells
    const cells = notebook.content.widgets;
    cells.forEach(cell => {
      if (cell.model.type === 'code') {
        cell.model.metadataChanged.connect(() => {
          this.updateCell(cell);
        });
      }
    });
  }

  /**
   * Update all code cells in a notebook
   */
  private updateAllCells(notebook: NotebookPanel): void {
    const cells = notebook.content.widgets;
    cells.forEach(cell => {
      if (cell.model.type === 'code') {
        this.updateCell(cell);
      }
    });
  }

  /**
   * Update a single cell's styling
   */
  private updateCell(cell: Cell): void {
    const metadata = cell.model.metadata as any;
    const ferretMetadata = metadata?.ferret as IFerretMetadata | undefined;

    // Remove existing classes
    cell.node.classList.remove('ferret-cell-warning', 'ferret-cell-danger');

    // Check if any ferret metadata exists (inspect or profile)
    if (ferretMetadata && (ferretMetadata.inspect || ferretMetadata.profile)) {
      // Apply CSS class based on inspect metadata if present
      if (ferretMetadata.inspect) {
        const { optimizability, readability, complexity } = ferretMetadata.inspect;
        const maxValue = Math.max(optimizability, readability, complexity);

        // Apply CSS class based on max value
        if (3 < maxValue && maxValue <= 4) {
          cell.node.classList.add('ferret-cell-warning');
        } else if (maxValue === 5) {
          cell.node.classList.add('ferret-cell-danger');
        }
      }
    }

    // Update panel if this is the active cell
    if (this.tracker.activeCell === cell) {
      if (ferretMetadata && (ferretMetadata.inspect || ferretMetadata.profile)) {
        this.panel.updateMetadata(ferretMetadata);
      } else {
        this.panel.clear();
      }
    }
  }

}
