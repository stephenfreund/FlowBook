/**
 * History manager for notebook operations and user edits
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { IHistoryEntry, IHistoryState } from './types';
import { ISignal, Signal } from '@lumino/signaling';

/**
 * Generate a unique ID for history entries
 */
function generateId(): string {
  return `history-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Manages history for notebooks with undo/redo functionality
 */
export class NotebookHistoryManager {
  private _history: Map<string, IHistoryState> = new Map();
  private _changeListeners: Map<string, any[]> = new Map();
  private _debounceTimers: Map<string, any> = new Map();
  private _isRestoring: boolean = false;
  private _historyChanged: Signal<this, string> = new Signal(this);

  /**
   * Signal emitted when history changes for a notebook
   */
  get historyChanged(): ISignal<this, string> {
    return this._historyChanged;
  }

  /**
   * Initialize history for a notebook
   */
  private ensureHistory(notebookPath: string): IHistoryState {
    if (!this._history.has(notebookPath)) {
      this._history.set(notebookPath, {
        entries: [],
        currentIndex: -1,
        maxEntries: 50,
        pendingEdit: false,
        lastSnapshotTime: 0,
        editDebounceMs: 2000
      });
    }
    return this._history.get(notebookPath)!;
  }

  /**
   * Add a command entry to history
   */
  addCommandEntry(notebookPath: string, entry: Omit<IHistoryEntry, 'type'>): void {
    const state = this.ensureHistory(notebookPath);

    // If we've undone and now executing a command, discard future history
    if (state.currentIndex < state.entries.length - 1) {
      state.entries = state.entries.slice(0, state.currentIndex + 1);
    }

    // Add the command entry
    const commandEntry: IHistoryEntry = {
      ...entry,
      type: 'command'
    };

    state.entries.push(commandEntry);
    state.currentIndex = state.entries.length - 1;
    state.lastSnapshotTime = Date.now();

    // Prune if exceeds max
    this.pruneOldEntries(notebookPath);

    this._historyChanged.emit(notebookPath);
  }

  /**
   * Capture a user edit (called after debounce or on command execution)
   */
  captureUserEdit(notebookPath: string, notebook: NotebookPanel): void {
    if (this._isRestoring) return;

    const state = this.ensureHistory(notebookPath);
    if (state.currentIndex < 0) return; // No initial state yet

    const currentSnapshot = notebook.content.model?.toJSON();
    if (!currentSnapshot) return;

    const lastEntry = state.entries[state.currentIndex];
    if (!lastEntry) return;

    // Compare snapshots
    const changes = this.compareSnapshots(lastEntry.notebookSnapshot, currentSnapshot);

    if (!changes.hasChanges) return;

    // If we've undone and user makes edit, discard future history
    if (state.currentIndex < state.entries.length - 1) {
      state.entries = state.entries.slice(0, state.currentIndex + 1);
    }

    // Check if we should combine with the previous user edit entry
    const shouldCombine = lastEntry.type === 'user-edit' && this.hasCellOverlap(
      lastEntry.affectedCells,
      changes.affectedCells
    );

    if (shouldCombine) {
      // Update the existing user edit entry
      lastEntry.notebookSnapshot = currentSnapshot;
      lastEntry.timestamp = Date.now();

      // Merge affected cells (unique cells only)
      const combinedCells = new Set([...lastEntry.affectedCells, ...changes.affectedCells]);
      lastEntry.affectedCells = Array.from(combinedCells);

      // Combine edit summaries
      if (lastEntry.editSummary && changes.summary) {
        lastEntry.editSummary.cellsAdded += changes.summary.cellsAdded;
        lastEntry.editSummary.cellsDeleted += changes.summary.cellsDeleted;
        lastEntry.editSummary.cellsModified += changes.summary.cellsModified;
        lastEntry.editSummary.cellsMoved += changes.summary.cellsMoved;
        lastEntry.description = this.generateEditDescription(lastEntry.editSummary);
      }

      state.lastSnapshotTime = Date.now();
      state.pendingEdit = false;
    } else {
      // Create new user edit entry
      const editEntry: IHistoryEntry = {
        id: generateId(),
        timestamp: Date.now(),
        type: 'user-edit',
        icon: 'ui-components:edit',
        notebookSnapshot: currentSnapshot,
        affectedCells: changes.affectedCells,
        description: this.generateEditDescription(changes.summary),
        editSummary: changes.summary
      };

      state.entries.push(editEntry);
      state.currentIndex = state.entries.length - 1;
      state.lastSnapshotTime = Date.now();
      state.pendingEdit = false;

      // Prune if exceeds max
      this.pruneOldEntries(notebookPath);
    }

    this._historyChanged.emit(notebookPath);
  }

  /**
   * Flush any pending debounced edit immediately
   */
  flushPendingEdit(notebookPath: string, notebook: NotebookPanel): void {
    if (this._debounceTimers.has(notebookPath)) {
      clearTimeout(this._debounceTimers.get(notebookPath));
      this._debounceTimers.delete(notebookPath);
      this.captureUserEdit(notebookPath, notebook);
    }
  }

  /**
   * Start monitoring a notebook for user edits
   */
  startMonitoring(notebookPath: string, notebook: NotebookPanel): void {
    console.log(`[History] Start monitoring: ${notebookPath}`);

    // Create initial snapshot
    const initialSnapshot = notebook.content.model?.toJSON();
    if (initialSnapshot) {
      const state = this.ensureHistory(notebookPath);
      if (state.entries.length === 0) {
        // Add initial state entry
        const initialEntry: IHistoryEntry = {
          id: generateId(),
          timestamp: Date.now(),
          type: 'command',
          commandId: 'initial',
          commandLabel: 'Initial State',
          icon: 'ui-components:save',
          notebookSnapshot: initialSnapshot,
          affectedCells: [],
          description: 'Initial notebook state'
        };
        state.entries.push(initialEntry);
        state.currentIndex = 0;
        this._historyChanged.emit(notebookPath);
      }
    }

    const listeners: any[] = [];

    // Listen to cell list changes
    const cellsChangedListener = () => {
      this.onNotebookChanged(notebookPath, notebook);
    };
    notebook.content.model?.cells.changed.connect(cellsChangedListener);
    listeners.push({ signal: notebook.content.model?.cells.changed, callback: cellsChangedListener });

    // Listen to content changes
    const contentChangedListener = () => {
      this.onNotebookChanged(notebookPath, notebook);
    };
    notebook.content.model?.contentChanged.connect(contentChangedListener);
    listeners.push({ signal: notebook.content.model?.contentChanged, callback: contentChangedListener });

    // Listen to metadata changes on existing cells
    notebook.content.widgets.forEach(cell => {
      const metadataChangedListener = () => {
        this.onNotebookChanged(notebookPath, notebook);
      };
      cell.model.metadataChanged.connect(metadataChangedListener);
      listeners.push({ signal: cell.model.metadataChanged, callback: metadataChangedListener });

      const cellContentChangedListener = () => {
        this.onNotebookChanged(notebookPath, notebook);
      };
      cell.model.contentChanged.connect(cellContentChangedListener);
      listeners.push({ signal: cell.model.contentChanged, callback: cellContentChangedListener });
    });

    this._changeListeners.set(notebookPath, listeners);
  }

  /**
   * Stop monitoring a notebook
   */
  stopMonitoring(notebookPath: string): void {
    console.log(`[History] Stop monitoring: ${notebookPath}`);

    // Clear debounce timer
    if (this._debounceTimers.has(notebookPath)) {
      clearTimeout(this._debounceTimers.get(notebookPath));
      this._debounceTimers.delete(notebookPath);
    }

    // Disconnect listeners
    const listeners = this._changeListeners.get(notebookPath);
    if (listeners) {
      listeners.forEach(({ signal, callback }) => {
        try {
          signal?.disconnect(callback);
        } catch (e) {
          // Ignore errors from already disconnected signals
        }
      });
      this._changeListeners.delete(notebookPath);
    }
  }

  /**
   * Handle notebook changes (with debouncing)
   */
  private onNotebookChanged(notebookPath: string, notebook: NotebookPanel): void {
    if (this._isRestoring) return;

    const state = this._history.get(notebookPath);
    if (!state) return;

    // Clear existing debounce timer
    if (this._debounceTimers.has(notebookPath)) {
      clearTimeout(this._debounceTimers.get(notebookPath));
    }

    state.pendingEdit = true;

    // Set new debounce timer
    const timer = setTimeout(() => {
      this.captureUserEdit(notebookPath, notebook);
      this._debounceTimers.delete(notebookPath);
    }, state.editDebounceMs);

    this._debounceTimers.set(notebookPath, timer);
  }

  /**
   * Compare two notebook snapshots and return what changed
   */
  private compareSnapshots(
    before: any,
    after: any
  ): {
    hasChanges: boolean;
    affectedCells: string[];
    summary: {
      cellsAdded: number;
      cellsDeleted: number;
      cellsModified: number;
      cellsMoved: number;
    };
  } {
    const beforeCells = new Map<string, any>(before.cells?.map((c: any) => [c.id, c]) || []);
    const afterCells = new Map<string, any>(after.cells?.map((c: any) => [c.id, c]) || []);

    let added = 0;
    let deleted = 0;
    let modified = 0;
    const affectedCells: string[] = [];

    // Check for deletions and modifications
    beforeCells.forEach((cell: any, id: string) => {
      if (!afterCells.has(id)) {
        deleted++;
        affectedCells.push(id);
      } else {
        const afterCell = afterCells.get(id);
        if (JSON.stringify(cell) !== JSON.stringify(afterCell)) {
          modified++;
          affectedCells.push(id);
        }
      }
    });

    // Check for additions
    afterCells.forEach((cell: any, id: string) => {
      if (!beforeCells.has(id)) {
        added++;
        affectedCells.push(id);
      }
    });

    // Simple move detection (cell order changed but same cells)
    let moved = 0;
    if (added === 0 && deleted === 0 && before.cells && after.cells) {
      const beforeOrder = before.cells.map((c: any) => c.id);
      const afterOrder = after.cells.map((c: any) => c.id);
      if (JSON.stringify(beforeOrder) !== JSON.stringify(afterOrder)) {
        moved = beforeOrder.length;
      }
    }

    return {
      hasChanges: added > 0 || deleted > 0 || modified > 0 || moved > 0,
      affectedCells,
      summary: {
        cellsAdded: added,
        cellsDeleted: deleted,
        cellsModified: modified,
        cellsMoved: moved
      }
    };
  }

  /**
   * Generate a human-readable description of an edit
   */
  private generateEditDescription(summary: {
    cellsAdded: number;
    cellsDeleted: number;
    cellsModified: number;
    cellsMoved: number;
  }): string {
    const parts: string[] = [];

    if (summary.cellsAdded > 0) {
      parts.push(`${summary.cellsAdded} cell${summary.cellsAdded > 1 ? 's' : ''} added`);
    }
    if (summary.cellsDeleted > 0) {
      parts.push(`${summary.cellsDeleted} cell${summary.cellsDeleted > 1 ? 's' : ''} deleted`);
    }
    if (summary.cellsModified > 0) {
      parts.push(`${summary.cellsModified} cell${summary.cellsModified > 1 ? 's' : ''} modified`);
    }
    if (summary.cellsMoved > 0) {
      parts.push('cells reordered');
    }

    return parts.length > 0 ? parts.join(', ') : 'No changes';
  }

  /**
   * Jump to a specific history entry
   */
  jumpToEntry(notebookPath: string, index: number, notebook: NotebookPanel): IHistoryEntry | null {
    const state = this._history.get(notebookPath);
    if (!state || index < 0 || index >= state.entries.length) {
      return null;
    }

    const entry = state.entries[index];

    this._isRestoring = true;
    try {
      // Restore notebook state
      notebook.content.model?.fromJSON(entry.notebookSnapshot);
      state.currentIndex = index;
      this._historyChanged.emit(notebookPath);
      return entry;
    } finally {
      this._isRestoring = false;
    }
  }

  /**
   * Undo to previous entry
   */
  undo(notebookPath: string, notebook: NotebookPanel): IHistoryEntry | null {
    const state = this._history.get(notebookPath);
    if (!state || state.currentIndex <= 0) {
      return null;
    }
    return this.jumpToEntry(notebookPath, state.currentIndex - 1, notebook);
  }

  /**
   * Redo to next entry
   */
  redo(notebookPath: string, notebook: NotebookPanel): IHistoryEntry | null {
    const state = this._history.get(notebookPath);
    if (!state || state.currentIndex >= state.entries.length - 1) {
      return null;
    }
    return this.jumpToEntry(notebookPath, state.currentIndex + 1, notebook);
  }

  /**
   * Get history for a notebook
   */
  getHistory(notebookPath: string): IHistoryEntry[] {
    const state = this._history.get(notebookPath);
    return state ? state.entries : [];
  }

  /**
   * Get current index
   */
  getCurrentIndex(notebookPath: string): number {
    const state = this._history.get(notebookPath);
    return state ? state.currentIndex : -1;
  }

  /**
   * Check if can undo
   */
  canUndo(notebookPath: string): boolean {
    const state = this._history.get(notebookPath);
    return state ? state.currentIndex > 0 : false;
  }

  /**
   * Check if can redo
   */
  canRedo(notebookPath: string): boolean {
    const state = this._history.get(notebookPath);
    return state ? state.currentIndex < state.entries.length - 1 : false;
  }

  /**
   * Clear history for a notebook
   */
  clearHistory(notebookPath: string): void {
    this._history.delete(notebookPath);
    this.stopMonitoring(notebookPath);
    this._historyChanged.emit(notebookPath);
  }

  /**
   * Prune old entries if exceeds max
   */
  private pruneOldEntries(notebookPath: string): void {
    const state = this._history.get(notebookPath);
    if (!state) return;

    if (state.entries.length > state.maxEntries) {
      const removeCount = state.entries.length - state.maxEntries;
      state.entries = state.entries.slice(removeCount);
      state.currentIndex = Math.max(0, state.currentIndex - removeCount);
    }
  }

  /**
   * Check if two cell ID lists have any overlap
   */
  private hasCellOverlap(cells1: string[], cells2: string[]): boolean {
    if (cells1.length === 0 || cells2.length === 0) {
      return false;
    }
    const set1 = new Set(cells1);
    return cells2.some(cellId => set1.has(cellId));
  }
}
