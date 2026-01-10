/**
 * Cell metadata highlighter for FlowBook inspection data
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { IFlowbookMetadata } from './types';
import { FlowbookMetadataPanel } from './metadatapanel';

/**
 * Class that monitors notebook cells and applies visual indicators
 * based on FlowBook inspection metadata
 */
export class CellMetadataHighlighter {
  private tracker: INotebookTracker;
  private panel: FlowbookMetadataPanel;

  constructor(tracker: INotebookTracker, panel: FlowbookMetadataPanel) {
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
      const flowbookMetadata = metadata?.flowbook as IFlowbookMetadata | undefined;
      const cellId = cell.model.id;

      if (flowbookMetadata && (flowbookMetadata.optimization_potential || flowbookMetadata.profile || flowbookMetadata.dynamic_dependencies)) {
        this.panel.updateMetadata(flowbookMetadata, cellId);
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
    const flowbookMetadata = metadata?.flowbook as IFlowbookMetadata | undefined;
    const cellId = cell.model.id;

    // Remove existing classes
    cell.node.classList.remove('flowbook-cell-warning', 'flowbook-cell-danger');

    // Check if any flowbook metadata exists (optimization potential or profile)
    if (flowbookMetadata && (flowbookMetadata.optimization_potential || flowbookMetadata.profile)) {
      // Apply CSS class based on optimization potential metadata if present
      if (flowbookMetadata.optimization_potential) {
        const { potential } = flowbookMetadata.optimization_potential;
        const maxValue = potential;

        // Apply CSS class based on max value
        if (3 < maxValue && maxValue <= 4) {
          cell.node.classList.add('flowbook-cell-warning');
        } else if (maxValue === 5) {
          cell.node.classList.add('flowbook-cell-danger');
        }
      }
    }

    // Update panel if this is the active cell
    if (this.tracker.activeCell === cell) {
      if (flowbookMetadata && (flowbookMetadata.optimization_potential || flowbookMetadata.profile || flowbookMetadata.dynamic_dependencies)) {
        this.panel.updateMetadata(flowbookMetadata, cellId);
      } else {
        this.panel.clear();
      }
    }
  }

}
