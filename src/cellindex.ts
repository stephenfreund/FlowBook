/**
 * Cell Index Overlay Manager
 * Adds 0-based cell index overlays to code cells using @A notation
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { indexToAlpha } from './cellindexutils';

/**
 * Manages cell index overlays for notebooks
 */
export class CellIndexManager {
  private _overlays: Map<string, Map<string, HTMLElement>> = new Map();
  private _observers: Map<string, MutationObserver> = new Map();
  private _notebooks: Map<string, NotebookPanel> = new Map();
  private _listeners: Map<string, any[]> = new Map();
  private _domListeners: Map<
    string,
    { node: HTMLElement; handlers: { event: string; handler: EventListener }[] }
  > = new Map();

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
    this._notebooks.set(notebookPath, notebook);

    // Create initial overlays
    this.updateAllOverlays(notebookPath, notebook);

    // Use MutationObserver to watch for editor elements being added to DOM
    const observer = new MutationObserver(mutations => {
      for (const mutation of mutations) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          // Check if any added nodes contain editor elements
          for (const node of mutation.addedNodes) {
            if (node instanceof HTMLElement) {
              if (
                node.classList.contains('jp-InputArea-editor') ||
                node.querySelector('.jp-InputArea-editor')
              ) {
                // Editor element was added, update overlays
                this.updateAllOverlays(notebookPath, notebook);
                return;
              }
            }
          }
        }
      }
    });

    // Observe the notebook content for DOM changes
    observer.observe(notebook.content.node, {
      childList: true,
      subtree: true
    });
    this._observers.set(notebookPath, observer);

    // Listen to cell list changes
    const listeners: any[] = [];

    const cellsChangedListener = () => {
      this.updateAllOverlays(notebookPath, notebook);
    };

    notebook.content.model?.cells.changed.connect(cellsChangedListener);
    listeners.push({
      signal: notebook.content.model?.cells.changed,
      callback: cellsChangedListener
    });

    this._listeners.set(notebookPath, listeners);

    // Listen for clipboard and undo/redo operations that can detach overlays
    const contentNode = notebook.content.node;
    const deferredRefresh = () => {
      // Wait for the DOM to settle after the operation
      requestAnimationFrame(() => {
        this.updateAllOverlays(notebookPath, notebook);
      });
    };
    const domHandlers: { event: string; handler: EventListener }[] = [];

    for (const event of ['cut', 'paste'] as const) {
      contentNode.addEventListener(event, deferredRefresh, true);
      domHandlers.push({ event, handler: deferredRefresh });
    }

    // Catch undo/redo (Cmd+Z / Ctrl+Z / Ctrl+Shift+Z / Cmd+Shift+Z)
    const keyHandler = (e: Event) => {
      const ke = e as KeyboardEvent;
      if ((ke.metaKey || ke.ctrlKey) && ke.key === 'z') {
        deferredRefresh();
      }
    };
    contentNode.addEventListener('keydown', keyHandler, true);
    domHandlers.push({ event: 'keydown', handler: keyHandler });

    this._domListeners.set(notebookPath, {
      node: contentNode,
      handlers: domHandlers
    });
  }

  /**
   * Stop monitoring a notebook
   */
  stopMonitoring(notebookPath: string): void {
    console.log(`[CellIndex] Stop monitoring: ${notebookPath}`);

    // Stop MutationObserver
    const observer = this._observers.get(notebookPath);
    if (observer) {
      observer.disconnect();
      this._observers.delete(notebookPath);
    }

    // Remove all overlays
    const overlays = this._overlays.get(notebookPath);
    if (overlays) {
      overlays.forEach(overlay => {
        overlay.remove();
      });
      this._overlays.delete(notebookPath);
    }

    // Remove notebook reference
    this._notebooks.delete(notebookPath);

    // Disconnect signal listeners
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

    // Remove DOM event listeners
    const domEntry = this._domListeners.get(notebookPath);
    if (domEntry) {
      for (const { event, handler } of domEntry.handlers) {
        domEntry.node.removeEventListener(event, handler, true);
      }
      this._domListeners.delete(notebookPath);
    }
  }

  /**
   * Update all cell index overlays for a notebook
   */
  private updateAllOverlays(
    notebookPath: string,
    notebook: NotebookPanel
  ): void {
    const overlays = this._overlays.get(notebookPath);
    if (!overlays) {
      return;
    }

    // Build map of cell ID to code cell index
    const cellIndexMap = new Map<string, number>();
    let codeCellIndex = 0;
    notebook.content.widgets.forEach(cell => {
      if (cell.model.type === 'code') {
        cellIndexMap.set(cell.model.id, codeCellIndex);
        codeCellIndex++;
      }
    });

    // Add overlays to all code cells that have editor elements
    notebook.content.widgets.forEach(cell => {
      if (cell.model.type === 'code') {
        const cellId = cell.model.id;
        const index = cellIndexMap.get(cellId);

        if (index === undefined) {
          return;
        }

        // Check if overlay already exists for this cell
        const existingOverlay = overlays.get(cellId);
        if (existingOverlay && existingOverlay.parentNode) {
          existingOverlay.textContent = indexToAlpha(index);
          return;
        }

        // Try to add overlay
        const editorNode = cell.node.querySelector('.jp-InputArea-editor');
        if (editorNode) {
          // Remove old overlay if it exists but isn't attached
          if (existingOverlay) {
            existingOverlay.remove();
          }

          const overlay = this.createOverlay(index);
          (editorNode as HTMLElement).style.position = 'relative';
          editorNode.appendChild(overlay);
          overlays.set(cellId, overlay);
        }
      }
    });
  }

  /**
   * Create an overlay element with the cell index
   */
  private createOverlay(index: number): HTMLElement {
    const overlay = document.createElement('div');
    overlay.className = 'flowbook-cell-index-overlay';
    overlay.textContent = indexToAlpha(index);
    return overlay;
  }
}
