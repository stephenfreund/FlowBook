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
  private _stalenessReasons = new Map<string, IStalenessReason[]>();
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
   * Store the reasons a cell is stale
   */
  setReasons(cellId: string, reasons: IStalenessReason[]): void {
    this._stalenessReasons.set(cellId, reasons);
  }

  /**
   * Get all reasons a cell is stale
   */
  getReasons(cellId: string): IStalenessReason[] {
    return this._stalenessReasons.get(cellId) || [];
  }

  /**
   * Store a single reason a cell is stale (wraps in array internally).
   */
  setReason(cellId: string, reason: IStalenessReason): void {
    this._stalenessReasons.set(cellId, [reason]);
  }

  /**
   * Get the primary reason a cell is stale, or undefined if none.
   */
  getReason(cellId: string): IStalenessReason | undefined {
    const reasons = this._stalenessReasons.get(cellId);
    return reasons && reasons.length > 0 ? reasons[0] : undefined;
  }

  /**
   * Eagerly mark a single cell as stale and record a reason.
   *
   * Used by callers that mutate cell source outside the kernel's
   * cell_edited round-trip (e.g. the AI fix suggester) so the staleness
   * UI updates immediately rather than after the debounced kernel
   * notification. The next kernel-driven updateFromMetadata() will
   * reconcile (and likely keep the cell stale, since cell_edited will
   * have been sent in the meantime).
   */
  markStale(cellId: string, reason: IStalenessReason): void {
    const wasStale = this._staleCells.has(cellId);
    this._staleCells.add(cellId);
    this._stalenessReasons.set(cellId, [reason]);
    if (!wasStale) {
      this._stalenessChanged.emit({
        added: [cellId],
        removed: [],
        current: [...this._staleCells]
      });
    }
  }

  /**
   * Update staleness from reproducibility metadata
   *
   * The metadata contains the ABSOLUTE set of all currently stale cells
   * as computed by the kernel. We replace our entire set with this truth.
   *
   * The metadata also contains `staleness_reasons` mapping each stale cell
   * to an array of reasons WHY it is stale (§1.2 in formal spec).
   */
  updateFromMetadata(reproducibilityMetadata: IReproducibilityMetadata): void {
    // Track previous state for diff
    const previousStale = new Set(this._staleCells);

    // Replace entire set with kernel's absolute truth
    this._staleCells = new Set(reproducibilityMetadata.stale_cells);

    // Compute diff for event
    const currentStale = new Set(reproducibilityMetadata.stale_cells);
    const added = [...currentStale].filter(id => !previousStale.has(id));
    const removed = [...previousStale].filter(id => !currentStale.has(id));

    // Clear reasons for cells that are no longer stale
    for (const id of removed) {
      this._stalenessReasons.delete(id);
    }

    // Update reasons from metadata and detect reason changes
    let reasonsChanged = false;
    if (reproducibilityMetadata.staleness_reasons) {
      for (const [cellId, reasons] of Object.entries(
        reproducibilityMetadata.staleness_reasons
      )) {
        const oldReasons = this._stalenessReasons.get(cellId);
        // Check if reasons actually changed (compare serialized form)
        const oldJson = JSON.stringify(oldReasons || []);
        const newJson = JSON.stringify(reasons);
        if (oldJson !== newJson) {
          reasonsChanged = true;
        }
        this._stalenessReasons.set(cellId, reasons);
      }
    }

    // Emit signal if cells changed OR reasons changed (e.g., reason type updated)
    if (added.length > 0 || removed.length > 0 || reasonsChanged) {
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
