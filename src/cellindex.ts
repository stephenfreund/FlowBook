/**
 * Cell Index Overlay Manager
 * Adds 1-based cell index overlays to code cells
 */

import { NotebookPanel } from '@jupyterlab/notebook';

/**
 * Manages cell index overlays for notebooks
 */
export class CellIndexManager {
  private _overlays: Map<string, Map<string, HTMLElement>> = new Map();
  private _listeners: Map<string, any[]> = new Map();

  constructor() {
    // Notebooks are passed directly to methods
  }

  /**
   * Start monitoring a notebook for cell index overlays
   */
  startMonitoring(notebookPath: string, notebook: NotebookPanel): void {
    console.log(`[CellIndex] Start monitoring: ${notebookPath}`);

    // Initialize overlay map for this notebook
    this._overlays.set(notebookPath, new Map());

    // Create initial overlays (with delay to ensure DOM is ready)
    setTimeout(() => {
      this.updateAllOverlays(notebookPath, notebook);
    }, 100);

    // Listen to cell list changes
    const listeners: any[] = [];

    const cellsChangedListener = () => {
      // Use setTimeout to wait for DOM to be ready
      setTimeout(() => {
        this.updateAllOverlays(notebookPath, notebook);
      }, 100);
    };

    notebook.content.model?.cells.changed.connect(cellsChangedListener);
    listeners.push({ signal: notebook.content.model?.cells.changed, callback: cellsChangedListener });

    this._listeners.set(notebookPath, listeners);
  }

  /**
   * Stop monitoring a notebook
   */
  stopMonitoring(notebookPath: string): void {
    console.log(`[CellIndex] Stop monitoring: ${notebookPath}`);

    // Remove all overlays
    const overlays = this._overlays.get(notebookPath);
    if (overlays) {
      overlays.forEach(overlay => {
        overlay.remove();
      });
      this._overlays.delete(notebookPath);
    }

    // Disconnect listeners
    const listeners = this._listeners.get(notebookPath);
    if (listeners) {
      listeners.forEach(({ signal, callback }) => {
        try {
          signal?.disconnect(callback);
        } catch (e) {
          // Ignore errors from already disconnected signals
        }
      });
      this._listeners.delete(notebookPath);
    }
  }

  /**
   * Update all cell index overlays for a notebook
   */
  private updateAllOverlays(notebookPath: string, notebook: NotebookPanel): void {
    const overlays = this._overlays.get(notebookPath);
    if (!overlays) return;

    // Remove old overlays
    overlays.forEach(overlay => {
      overlay.remove();
    });
    overlays.clear();

    // Add overlays to all code cells
    let codeCellIndex = 1;
    notebook.content.widgets.forEach((cell) => {
      // Only add to code cells
      if (cell.model.type === 'code') {
        const overlay = this.createOverlay(codeCellIndex);
        const editorNode = cell.node.querySelector('.jp-InputArea-editor');

        if (editorNode) {
          // Make sure the editor has position relative
          (editorNode as HTMLElement).style.position = 'relative';
          editorNode.appendChild(overlay);
          overlays.set(cell.model.id, overlay);
        } else {
          console.warn(`[CellIndex] Editor node not found for cell ${cell.model.id}`);
        }
        codeCellIndex++;
      }
    });
  }

  /**
   * Create an overlay element with the cell index
   */
  private createOverlay(index: number): HTMLElement {
    const overlay = document.createElement('div');
    overlay.className = 'ferret-cell-index-overlay';
    overlay.textContent = `#${index}`;
    return overlay;
  }
}
