/**
 * Unit test panel tracker for monitoring cell selection
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { UnitTestPanel } from './unittestpanel';

/**
 * Class that monitors active cell changes and updates the unit test panel
 */
export class UnitTestPanelTracker {
  private tracker: INotebookTracker;
  private panel: UnitTestPanel;

  constructor(tracker: INotebookTracker, panel: UnitTestPanel) {
    this.tracker = tracker;
    this.panel = panel;
    this.initialize();
  }

  /**
   * Initialize the tracker
   */
  private initialize(): void {
    // Monitor active cell changes
    this.tracker.activeCellChanged.connect(this.onActiveCellChanged, this);

    // Update for the current active cell if one exists
    if (this.tracker.activeCell) {
      this.onActiveCellChanged(this.tracker, this.tracker.activeCell);
    }

    // Monitor notebook changes to update when switching notebooks
    this.tracker.currentChanged.connect(this.onNotebookChanged, this);
  }

  /**
   * Handle notebook change events
   */
  private onNotebookChanged(
    tracker: INotebookTracker,
    notebook: NotebookPanel | null
  ): void {
    if (notebook && notebook.content.activeCell) {
      this.onActiveCellChanged(tracker, notebook.content.activeCell);
    } else {
      this.panel.clear();
    }
  }

  /**
   * Handle active cell change events
   */
  private onActiveCellChanged(
    tracker: INotebookTracker,
    cell: Cell | null
  ): void {
    if (!cell || cell.model.type !== 'code') {
      this.panel.clear();
      return;
    }

    // Just pass the cell directly - the panel will read/write metadata
    this.panel.updateCell(cell);
  }
}
