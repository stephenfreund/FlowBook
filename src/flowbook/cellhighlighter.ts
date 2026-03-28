/**
 * Cell highlighter for reproducibility staleness visualization
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
import {
  IReproducibilityMetadata,
  IStalenessReason,
  IFrontendStalenessReason,
  IPredicateViolation
} from './types';
import { indexToAlpha } from '../cellindexutils';

/**
 * Type guard to check if a staleness reason is a frontend reason with message.
 */
function isFrontendReason(
  reason: IStalenessReason
): reason is IFrontendStalenessReason {
  return 'message' in reason;
}

export class ReproducibilityCellHighlighter {
  private _tracker: INotebookTracker;
  private _panel: ReproducibilityMetadataPanel;
  private _dependenciesPanel: DependenciesPanel | null = null;
  private _depPanelFrameId: number | null = null; // debounce for dep panel updates
  private _stalenessManagers = new Map<string, StalenessManager>();
  private _pendingRestartUpdate = new Set<string>(); // notebook paths awaiting update after restart
  private _executedInSession = new Map<string, Set<string>>(); // notebook path -> set of executed cell IDs

  constructor(tracker: INotebookTracker, panel: ReproducibilityMetadataPanel) {
    this._tracker = tracker;
    this._panel = panel;
    this._initialize();
  }

  /**
   * Set the dependencies panel for graph updates.
   */
  setDependenciesPanel(panel: DependenciesPanel): void {
    this._dependenciesPanel = panel;
  }

  /**
   * Schedule a dependency graph refresh on the next animation frame.
   * Multiple calls within the same frame collapse into one update,
   * ensuring the panel always reads the latest metadata regardless
   * of signal ordering between IOPub, executed, and staleness handlers.
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
        const cellOrder = this._getCurrentCellOrder(notebook);
        this._updateDependenciesPanel(notebook, stalenessManager, cellOrder);
      }
    });
  }

  /**
   * Update the status display in the metadata panel header.
   * Called when a "status" protocol message arrives from the kernel.
   */
  updateStatus(icon: string, text: string, cellId?: string): void {
    this._panel.updateStatus(icon, text, cellId);
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

  private _onNotebookChanged(
    tracker: INotebookTracker,
    notebook: NotebookPanel | null
  ): void {
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
      const metadata = cell.model.metadata as any;
      const reproducibilityMetadata = metadata?.flowbook as
        | IReproducibilityMetadata
        | undefined;
      const cellId = cell.model.id;
      const currentCellOrder = this._getCurrentCellOrder(notebook);

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

  private _monitorNotebook(notebook: NotebookPanel): void {
    this._updateAllCells(notebook);

    notebook.content.model?.cells.changed.connect(() => {
      this._updateAllCells(notebook);
      // Update panel with new cell order when cells are added/removed/reordered
      this._updatePanelWithCurrentCellOrder(notebook);
    });

    // Listen for cell execution to track executed cells.
    // Note: updateCell is NOT called here — it is called by
    // ReproducibilityExecutionHookManager._onCellExecuted after it has
    // finished extracting and storing violation/flowbook metadata.
    // Calling updateCell here would run _updateViolationOutput too early,
    // destroying the kernel's predicate_violation display_data before the
    // execution hook can read it.
    NotebookActions.executed.connect((_sender, args) => {
      if (args.notebook === notebook.content) {
        let executed = this._executedInSession.get(path);
        if (!executed) {
          executed = new Set<string>();
          this._executedInSession.set(path, executed);
        }
        executed.add(args.cell.model.id);
      }
    });

    // Listen for kernel restart to update all cells
    // We track 'restarting' and wait for 'idle' when execution counts are actually cleared
    const path = notebook.context.path;
    notebook.sessionContext.statusChanged.connect((_, status) => {
      console.log(
        `CellHighlighter: Kernel status changed to '${status}' for ${path}`
      );
      if (status === 'restarting' || status === 'autorestarting') {
        // Mark this notebook as pending update after restart
        this._pendingRestartUpdate.add(path);
        // Clear executed cells tracking for this notebook
        this._executedInSession.delete(path);
        // Clear all flowbook metadata since it's session-specific
        this._clearAllFlowbookMetadata(notebook);
        console.log(
          `CellHighlighter: Kernel restarting - cleared all flowbook metadata for ${path}`
        );
      } else if (status === 'idle' && this._pendingRestartUpdate.has(path)) {
        // Kernel is ready after restart
        this._pendingRestartUpdate.delete(path);
        console.log(
          'CellHighlighter: Kernel idle after restart, updating cells'
        );
        this._updateAllCells(notebook);
      }
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
    const reproducibilityMetadata = metadata?.flowbook as
      | IReproducibilityMetadata
      | undefined;

    if (reproducibilityMetadata) {
      const cellId = activeCell.model.id;
      const currentCellOrder = this._getCurrentCellOrder(notebook);
      this._panel.updateMetadata(
        reproducibilityMetadata,
        cellId,
        currentCellOrder
      );
    }
  }

  private _updateAllCells(notebook: NotebookPanel): void {
    const stalenessManager = this.getStalenessManager(notebook);
    const cellOrder = this._getCurrentCellOrder(notebook);
    const cells = notebook.content.widgets;
    const path = notebook.context.path;

    cells.forEach(cell => {
      if (cell.model.type === 'code') {
        this.updateCell(cell, stalenessManager, cellOrder, path);
      }
    });

    // Schedule dependency panel refresh (debounced to next frame)
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

      const metadata = cell.model.metadata as any;
      const flowbook = metadata?.flowbook as
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

    // Sort by program order index
    cellData.sort((a, b) => a.index - b.index);

    this._dependenciesPanel.updateGraph(cellData);
  }

  /**
   * Clear all flowbook metadata from cells on kernel restart.
   * Flowbook metadata is session-specific and invalid after restart.
   */
  private _clearAllFlowbookMetadata(notebook: NotebookPanel): void {
    const stalenessManager = this.getStalenessManager(notebook);
    stalenessManager.clear();

    const cells = notebook.content.widgets;
    cells.forEach(cell => {
      if (cell.model.type !== 'code') {
        return;
      }

      // Clear flowbook metadata keys
      cell.model.deleteMetadata('flowbook');
      cell.model.deleteMetadata('flowbook_staleness');
      cell.model.deleteMetadata('flowbook_violation');
      cell.model.deleteMetadata('flowbook_violations');

      // Remove flowbook notice outputs (staleness and violation notices)
      const codeModel = cell.model as ICodeCellModel;
      const outputs = codeModel.outputs;
      const cleanOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const meta = (out as any).metadata || {};
        // Keep outputs that are NOT flowbook notices
        if (
          !meta.flowbook_staleness_notice &&
          !meta.flowbook_violation_notice
        ) {
          cleanOutputs.push(out);
        }
      }
      // Only update if we removed something
      if (cleanOutputs.length !== outputs.length) {
        outputs.fromJSON(cleanOutputs);
      }

      // Remove staleness and error CSS classes
      cell.node.classList.remove('flowbook-stale-cell');
      cell.node.classList.remove('flowbook-unexecuted-cell');
      cell.node.classList.remove('flowbook-cell-stale');
      cell.node.classList.remove('flowbook-cell-unexecuted');
      cell.node.classList.remove('flowbook-cell-error');
    });
  }

  /**
   * Update cell highlighting and notices.
   * This is the single entry point for all cell rendering updates.
   * Can be called externally after storing metadata on the cell.
   */
  updateCell(
    cell: Cell,
    stalenessManager: StalenessManager,
    cellOrder: string[],
    notebookPath: string
  ): void {
    const cellId = cell.model.id;
    const isStale = stalenessManager.isCellStale(cellId);
    // Check if cell was executed in current kernel session
    // Use both frontend tracking AND flowbook metadata (which persists across browser refresh)
    const executedInSession = this._executedInSession.get(notebookPath);
    const hasFlowbookMetadata =
      cell.model.getMetadata('flowbook') !== undefined;
    const wasExecutedInSession =
      (executedInSession && executedInSession.has(cellId)) ||
      hasFlowbookMetadata;
    const isUnexecuted = !wasExecutedInSession;

    // Check if cell is empty (whitespace-only counts as empty)
    const codeModel = cell.model as ICodeCellModel;
    const source = codeModel.sharedModel.getSource();
    const isEmpty = !source || source.trim() === '';

    console.log(
      `CellHighlighter: Updating cell ${cellId}, isStale=${isStale}, isUnexecuted=${isUnexecuted}, isEmpty=${isEmpty}`
    );

    // Remove existing highlight classes
    cell.node.classList.remove('flowbook-cell-stale');
    cell.node.classList.remove('flowbook-cell-unexecuted');
    cell.node.classList.remove('flowbook-cell-error');

    // Skip highlighting and notices for empty cells - they are always clean
    if (isEmpty) {
      console.log(
        `CellHighlighter: Cell ${cellId} is empty, skipping highlight and notices`
      );
      // Clear any existing staleness notice from empty cells
      this._updateCellOutput(cell, false, stalenessManager, cellOrder);
    } else if (isStale) {
      // Add appropriate class
      cell.node.classList.add('flowbook-cell-stale');
      console.log(
        `CellHighlighter: Added .flowbook-cell-stale class to cell ${cellId}`
      );
      // Add staleness notice output
      this._updateCellOutput(cell, isStale, stalenessManager, cellOrder);
    } else if (isUnexecuted) {
      cell.node.classList.add('flowbook-cell-unexecuted');
      console.log(
        `CellHighlighter: Added .flowbook-cell-unexecuted class to cell ${cellId}`
      );
      // No staleness notice for unexecuted non-empty cells
      this._updateCellOutput(cell, false, stalenessManager, cellOrder);
    } else {
      console.log(
        `CellHighlighter: Cell ${cellId} is fresh (no highlight class)`
      );
      // Remove staleness notice
      this._updateCellOutput(cell, false, stalenessManager, cellOrder);
    }

    // Add/remove violation notice output (unified predicate violations)
    this._updateViolationOutput(cell, cellOrder);

    // Recompute metadata messages with current @A references
    this._updateStalenessMetadata(cell, cellOrder);
    this._updateViolationMetadata(cell, cellOrder);

    // Update panel if this is the active cell
    if (this._tracker.activeCell === cell) {
      const metadata = cell.model.metadata as any;
      const reproducibilityMetadata = metadata?.flowbook as
        | IReproducibilityMetadata
        | undefined;
      const notebook = this._tracker.currentWidget;

      if (reproducibilityMetadata && notebook) {
        this._panel.updateMetadata(reproducibilityMetadata, cellId, cellOrder);
      }
    }
  }

  /**
   * Format staleness message with dynamic @A references from current cell order.
   * Recomputes the human-readable message so it stays correct when cells are reordered.
   */
  private _formatStalenessMessage(
    reason: IStalenessReason,
    cellOrder: string[],
    currentCellId: string
  ): string {
    // Handle frontend reasons with full context
    if (isFrontendReason(reason)) {
      return this._formatFrontendReason(reason, cellOrder, currentCellId);
    }

    // Handle backend reasons (from kernel staleness_reasons)
    return this._formatBackendReason(reason, cellOrder, currentCellId);
  }

  /**
   * Format a frontend-computed staleness reason.
   */
  private _formatFrontendReason(
    reason: IFrontendStalenessReason,
    cellOrder: string[],
    currentCellId: string
  ): string {
    if (reason.type === 'source_edited') {
      return 'Source code was edited';
    }

    if (!reason.causing_cell) {
      return reason.message;
    }

    const causingIdx = cellOrder.indexOf(reason.causing_cell);
    const currentIdx = cellOrder.indexOf(currentCellId);
    // Note: indexToAlpha already returns with @ prefix
    // For deleted cells, don't include direction (it's meaningless)
    const isDeleted = causingIdx < 0;
    const causingRef = isDeleted ? 'a deleted cell' : indexToAlpha(causingIdx);
    // Determine relative position: "above" if causing cell is before current cell
    const direction =
      !isDeleted && currentIdx >= 0 && causingIdx < currentIdx
        ? ' above'
        : !isDeleted
          ? ' below'
          : '';

    // Build variable parts
    const parts: string[] = [];
    if (reason.variables) {
      for (const v of reason.variables) {
        parts.push('`' + v + '`');
      }
    }
    if (reason.columns) {
      for (const [dfName, cols] of Object.entries(reason.columns)) {
        for (const col of cols) {
          parts.push('`' + dfName + '.' + col + '`');
        }
      }
    }

    // WriterCheck: this cell writes to variables the causing cell reads
    // Running this cell would trigger BackConflict
    if (reason.type === 'writer_conflict' && parts.length > 0) {
      return `Writes ${parts.join(', ')} already read by ${causingRef}${direction}`;
    }

    // StaleFwd: the causing cell modified variables this cell reads
    if (parts.length > 0) {
      return `${parts.join(', ')} modified by ${causingRef}${direction}`;
    }

    if (reason.type === 'unknown') {
      return `Dependencies modified by ${causingRef}`;
    }

    return reason.message;
  }

  /**
   * Format a backend staleness reason from kernel.
   */
  private _formatBackendReason(
    reason: IStalenessReason,
    cellOrder: string[],
    currentCellId: string
  ): string {
    // Backend reasons have cell_id instead of causing_cell
    const cellId = 'cell_id' in reason ? reason.cell_id : undefined;
    const loc = 'loc' in reason ? reason.loc : undefined;

    const currentIdx = cellOrder.indexOf(currentCellId);

    // Note: indexToAlpha already returns with @ prefix
    // For deleted cells, don't include direction (it's meaningless)
    let causingRef = '';
    let causingDirection = '';
    let causingIsDeleted = false;
    if (cellId) {
      const causingIdx = cellOrder.indexOf(cellId);
      causingIsDeleted = causingIdx < 0;
      causingRef = causingIsDeleted
        ? 'a deleted cell'
        : indexToAlpha(causingIdx);
      // Determine relative position (only if cell still exists)
      if (!causingIsDeleted && currentIdx >= 0) {
        causingDirection = causingIdx < currentIdx ? ' above' : ' below';
      }
    }

    switch (reason.type) {
      case 'never_executed':
        return 'Cell has never been executed';
      case 'code_changed':
        return 'Source code was edited';
      case 'forward_stale':
        // ForwardStale: show "x modified by @F"
        if (loc && causingRef) {
          return `\`${loc}\` modified by ${causingRef}${causingDirection}`;
        }
        return causingRef
          ? `Input modified by ${causingRef}${causingDirection}`
          : 'Input was modified';
      case 'write_overlap':
        // Write overlap: both cells write to same location
        if (loc && causingRef) {
          return `\`${loc}\` also written by ${causingRef}`;
        }
        return causingRef
          ? `Writes conflict with ${causingRef}`
          : 'Write conflict detected';
      case 'backward_stale':
        if (loc && causingRef) {
          return `\`${loc}\` write conflict with ${causingRef}`;
        }
        return 'Write conflict detected';
      case 'no_read_before_write':
        // NoReadBeforeWrite failed - reads from another cell (forward contamination)
        if (loc && causingRef) {
          return `Reads \`${loc}\` written by ${causingRef} ${causingDirection}`;
        }
        return 'Reads value written by another cell';
      case 'order_changed':
        return 'Cell order changed';
      case 'no_write_after_read':
        // NoWriteAfterRead failed - writes to location read by another cell (backward mutation)
        if (loc && causingRef) {
          return `Writes \`${loc}\` already read by ${causingRef} ${causingDirection}`;
        }
        return causingRef
          ? `Writes variable already read by ${causingRef} ${causingDirection}`
          : 'Writes variable already read by another cell';
      default:
        return 'Cell is stale';
    }
  }

  /**
   * Add or remove the staleness notice display_data output at index 0.
   */
  private _updateCellOutput(
    cell: Cell,
    isStale: boolean,
    stalenessManager: StalenessManager,
    cellOrder: string[]
  ): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Check if there's a violation (either in metadata or existing notice)
    // Violation implies specific issue, so skip staleness notice
    const hasViolationMetadata =
      cell.model.getMetadata('flowbook_violation') !== undefined;
    let hasViolationNotice = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as any;
      if (out.metadata?.flowbook_violation_notice === true) {
        hasViolationNotice = true;
        break;
      }
    }

    // Remove staleness notice if violation is present (it's more specific)
    if (hasViolationMetadata || hasViolationNotice) {
      // Remove any existing staleness notice that may have been added by an
      // earlier update cycle (e.g., before the violation metadata was set)
      const allOutputs: IOutput[] = [];
      let removed = false;
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        if ((out as any).metadata?.flowbook_staleness_notice) {
          removed = true;
        } else {
          allOutputs.push(out);
        }
      }
      if (removed) {
        outputs.fromJSON(allOutputs);
      }
      return;
    }

    // Check if first output is already a staleness notice
    const hasNotice =
      outputs.length > 0 &&
      (outputs.get(0).toJSON() as any).metadata?.flowbook_staleness_notice ===
        true;

    if (isStale) {
      const reason = stalenessManager.getReason(cell.model.id) || {
        type: 'unknown',
        message: 'Dependencies changed'
      };

      // Don't display notice for never_executed cells
      if (reason.type === 'never_executed') {
        if (hasNotice) {
          // Remove existing notice
          const allOutputs: IOutput[] = [];
          for (let i = 0; i < outputs.length; i++) {
            const out = outputs.get(i).toJSON() as IOutput;
            if (!(out as any).metadata?.flowbook_staleness_notice) {
              allOutputs.push(out);
            }
          }
          outputs.fromJSON(allOutputs);
        }
        return;
      }

      const message = this._formatStalenessMessage(
        reason,
        cellOrder,
        cell.model.id
      );

      // Escape HTML in the message but preserve backtick-wrapped code
      const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Use different label for writer_conflict (potential violation vs stale dependency)
      const isWriterConflict = reason.type === 'writer_conflict';
      const label = isWriterConflict ? 'Unresolved Violation' : '';
      const plainText = label
        ? `\u26a0\ufe0f ${label}: ${message}`
        : `\u26a0\ufe0f ${message}`;

      const infoIcon = '';
      //        '<span class="flowbook-staleness-info-icon" title="This cell may produce different results than shown">ⓘ</span>';
      const stalenessOutput: IOutput = {
        output_type: 'display_data',
        data: {
          'text/html': label
            ? `<div class="flowbook-staleness-notice">\u26a0\ufe0f <b>${label}</b>: ${htmlMessage} ${infoIcon}</div>`
            : `<div class="flowbook-staleness-notice">\u26a0\ufe0f ${htmlMessage} ${infoIcon}</div>`,
          'text/plain': plainText
        },
        metadata: { flowbook_staleness_notice: true }
      };

      if (hasNotice) {
        // Check if message matches current notice
        const existingPlain = (outputs.get(0).toJSON() as any).data?.[
          'text/plain'
        ];
        if (existingPlain === plainText) {
          return; // Already up to date
        }
      }

      // Build new output array: [notice, ...existing non-notice outputs]
      const allOutputs: IOutput[] = [stalenessOutput];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        if (!(out as any).metadata?.flowbook_staleness_notice) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
    } else if (hasNotice) {
      // Remove staleness notice
      const allOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        if (!(out as any).metadata?.flowbook_staleness_notice) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
    }
  }

  /**
   * Add or remove the violation notice display_data output.
   * Handles IPredicateViolation[] format (multiple violations shown in a single error box).
   *
   * Both accepted=true (continue mode) and accepted=false (rejected) are
   * shown as red error boxes. A violation is a violation regardless.
   */
  private _updateViolationOutput(cell: Cell, cellOrder: string[]): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Get predicate violations from cell metadata
    const violations = cell.model.getMetadata('flowbook_violations') as
      | IPredicateViolation[]
      | undefined;

    // Check if we already have a violation notice
    let hasViolationNotice = false;
    let existingPlainText = '';
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as any;
      if (out.metadata?.flowbook_violation_notice === true) {
        hasViolationNotice = true;
        existingPlainText = out.data?.['text/plain'] || '';
        break;
      }
    }

    // Handle new predicate violation format (array of violations)
    if (violations && violations.length > 0) {
      const icon = '\u274c';
      const cssClass = 'flowbook-error-notice';

      // Group violations by (predicate, locations) to merge common ones
      // e.g., "Writes x read by @A" + "Writes x read by @B" => "Writes x read by @A, @B"
      const grouped = new Map<
        string,
        { predicate: string; locs: string[]; causers: string[] }
      >();

      for (const violation of violations) {
        // Get causer cell reference
        let causerRef: string | null = null;
        if (
          violation.causer_cell &&
          typeof violation.causer_cell === 'string'
        ) {
          const rawCauser = violation.causer_cell.startsWith('@')
            ? violation.causer_cell.slice(1)
            : violation.causer_cell;
          const causerIdx = cellOrder.indexOf(rawCauser);
          causerRef =
            causerIdx >= 0 ? indexToAlpha(causerIdx) : 'a deleted cell';
        }

        // Create grouping key from predicate and sorted locations
        const locsKey = [...violation.locations].sort().join(',');
        const groupKey = `${violation.predicate}:${locsKey}`;

        if (!grouped.has(groupKey)) {
          grouped.set(groupKey, {
            predicate: violation.predicate,
            locs: violation.locations,
            causers: []
          });
        }

        // Add causer if present and not already in list
        if (causerRef && !grouped.get(groupKey)!.causers.includes(causerRef)) {
          grouped.get(groupKey)!.causers.push(causerRef);
        }
      }

      // Build messages from grouped violations
      const htmlMessages: string[] = [];
      const plainMessages: string[] = [];

      for (const group of grouped.values()) {
        const locs = group.locs.map(l => '`' + l + '`').join(', ');
        const htmlLocs = locs.replace(/`([^`]+)`/g, '<code>$1</code>');
        const causersStr = group.causers.join(', ');

        // Build message based on predicate type
        let message: string;
        switch (group.predicate) {
          case 'no_write_after_read': {
            message = causersStr
              ? `Writes ${htmlLocs} already read by ${causersStr}`
              : `Writes ${htmlLocs} already read by cell above`;
            // Add column assignment hints for DataFrame column mutations only
            for (const loc of group.locs) {
              if (loc.includes('.')) {
                // Location has column info: "df.x" -> df["x"]
                const [dfName, colName] = loc.split('.');
                message += `<br>Use <code>${dfName}["${colName}"]</code> = ... for full-column assignment`;
              }
            }
            break;
          }
          case 'no_read_before_write':
            message = causersStr
              ? `Reads ${htmlLocs} written by ${causersStr} below`
              : `Reads ${htmlLocs} written by cell below`;
            break;
          case 'no_read_and_write':
            message = `Reads and writes ${htmlLocs}`;
            break;
          case 'write_before_read':
            message = `${htmlLocs} not defined by any cell above`;
            break;
          default:
            message = `Violation on ${htmlLocs}`;
        }

        htmlMessages.push(message);
        plainMessages.push(message.replace(/<code>([^<]+)<\/code>/g, '`$1`'));
      }

      // Combine all violations into a single notice
      const combinedHtml = htmlMessages
        .map(m => `<div>${icon} ${m}</div>`)
        .join('');
      const plainText = plainMessages.map(m => `${icon} ${m}`).join('\n');

      const noticeOutput: IOutput = {
        output_type: 'display_data',
        data: {
          'text/html': `<div class="${cssClass}">${combinedHtml}</div>`,
          'text/plain': plainText
        },
        metadata: {
          flowbook_violation_notice: true,
          flowbook_predicate_accepted: violations[0].accepted,
          flowbook_violation_count: violations.length
        }
      };

      // Check if message matches current notice
      if (hasViolationNotice && existingPlainText === plainText) {
        return; // Already up to date
      }

      // Add error class to cell for violations
      cell.node.classList.add('flowbook-cell-error');

      // Build new output array (also remove staleness notices — violation is more specific)
      const allOutputs: IOutput[] = [noticeOutput];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const isViolationNotice =
          (out as any).metadata?.flowbook_violation_notice === true;
        const isStalenessNotice =
          (out as any).metadata?.flowbook_staleness_notice === true;
        const isKernelError =
          out.output_type === 'error' &&
          ((out as any).ename === 'ReproducibilityError' ||
            (out as any).ename === 'ReproducibilityViolation');
        // Filter kernel's predicate violation display_data (empty text/plain)
        const isKernelPredicateViolation =
          out.output_type === 'display_data' &&
          (out as any).metadata?.predicate_violation;

        if (
          !isViolationNotice &&
          !isStalenessNotice &&
          !isKernelError &&
          !isKernelPredicateViolation
        ) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
      return;
    }

    // No predicate violations array - clear any existing violation notices
    let hasOurNotices = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as any;
      if (out.metadata?.flowbook_violation_notice) {
        hasOurNotices = true;
        break;
      }
    }

    if (hasOurNotices) {
      // Remove only our notices (not kernel's predicate_violation or errors)
      cell.node.classList.remove('flowbook-cell-error');
      const allOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const meta = (out as any).metadata || {};
        const isOurNotice = meta.flowbook_violation_notice;
        if (!isOurNotice) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
    }
  }

  /**
   * Recompute the flowbook_staleness metadata message with current @A references.
   */
  private _updateStalenessMetadata(cell: Cell, cellOrder: string[]): void {
    const staleness = cell.model.getMetadata('flowbook_staleness') as
      | IStalenessReason
      | undefined;
    if (!staleness) {
      return;
    }

    // Only frontend reasons (with causing_cell) need message updates
    if (!isFrontendReason(staleness) || !staleness.causing_cell) {
      return;
    }

    const newMessage = this._formatStalenessMessage(
      staleness,
      cellOrder,
      cell.model.id
    );
    if (newMessage !== staleness.message) {
      cell.model.setMetadata('flowbook_staleness', {
        ...staleness,
        message: newMessage
      });
    }
  }

  /**
   * Recompute violation metadata messages with current @A references.
   * Predicate violations are re-rendered by _updateViolationOutput using current cell order.
   */
  private _updateViolationMetadata(_cell: Cell, _cellOrder: string[]): void {
    // Predicate violations are rendered dynamically by _updateViolationOutput
    // using the current cellOrder, so no metadata update is needed.
  }
}
