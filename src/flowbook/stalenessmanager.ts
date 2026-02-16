/**
 * Manages staleness state across a notebook for FlowBook kernel
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { ISignal, Signal } from '@lumino/signaling';
import { IReproducibilityMetadata, IStalenessReason } from './types';

export interface IStalenessChange {
  added: string[];
  removed: string[];
  current: string[];
}

export class StalenessManager {
  private _staleCells = new Set<string>();
  private _stalenessReasons = new Map<string, IStalenessReason>();
  private _stalenessChanged = new Signal<this, IStalenessChange>(this);
  private _notebook: NotebookPanel;

  constructor(notebook: NotebookPanel) {
    this._notebook = notebook;
    this._setupKernelRestartListener();
  }

  get stalenessChanged(): ISignal<this, IStalenessChange> {
    return this._stalenessChanged;
  }

  get staleCells(): ReadonlySet<string> {
    return this._staleCells;
  }

  /**
   * Check if a cell is currently stale
   */
  isCellStale(cellId: string): boolean {
    return this._staleCells.has(cellId);
  }

  /**
   * Store the reason a cell became stale
   */
  setReason(cellId: string, reason: IStalenessReason): void {
    this._stalenessReasons.set(cellId, reason);
  }

  /**
   * Get the reason a cell is stale
   */
  getReason(cellId: string): IStalenessReason | undefined {
    return this._stalenessReasons.get(cellId);
  }

  /**
   * Update staleness from reproducibility metadata
   *
   * The metadata contains the ABSOLUTE set of all currently stale cells
   * as computed by the kernel. We replace our entire set with this truth.
   */
  updateFromMetadata(reproducibilityMetadata: IReproducibilityMetadata): void {
    console.log('StalenessManager: Before update, stale cells =', [
      ...this._staleCells
    ]);
    console.log(
      'StalenessManager: Metadata stale_cells =',
      reproducibilityMetadata.stale_cells
    );

    // Track previous state for diff
    const previousStale = new Set(this._staleCells);

    // Replace entire set with kernel's absolute truth
    this._staleCells = new Set(reproducibilityMetadata.stale_cells);

    // Compute diff for event
    const currentStale = new Set(reproducibilityMetadata.stale_cells);
    const added = [...currentStale].filter(id => !previousStale.has(id));
    const removed = [...previousStale].filter(id => !currentStale.has(id));

    console.log('StalenessManager: After update, stale cells =', [
      ...this._staleCells
    ]);
    // Clear reasons for cells that are no longer stale
    for (const id of removed) {
      this._stalenessReasons.delete(id);
    }

    console.log('StalenessManager: Added =', added, ', Removed =', removed);

    if (added.length > 0 || removed.length > 0) {
      this._stalenessChanged.emit({
        added,
        removed,
        current: [...this._staleCells]
      });
    }
  }

  /**
   * Clear all staleness state
   */
  clear(): void {
    const removed = [...this._staleCells];
    this._staleCells.clear();
    this._stalenessReasons.clear();

    if (removed.length > 0) {
      this._stalenessChanged.emit({
        added: [],
        removed,
        current: []
      });
    }
  }

  /**
   * Listen for kernel restart to clear staleness
   */
  private _setupKernelRestartListener(): void {
    this._notebook.sessionContext.statusChanged.connect((_, status) => {
      if (status === 'restarting' || status === 'autorestarting') {
        this.clear();
      }
    });
  }

  dispose(): void {
    this._staleCells.clear();
    this._stalenessReasons.clear();
  }
}
