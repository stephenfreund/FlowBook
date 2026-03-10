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
import {
  IReproducibilityMetadata,
  IStalenessReason,
  IFrontendStalenessReason,
  IViolationInfo,
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

/**
 * Type guard to check if a violation is the new predicate violation format.
 */
function isPredicateViolation(
  violation: IViolationInfo | IPredicateViolation | undefined
): violation is IPredicateViolation {
  return (
    violation !== undefined &&
    'predicate' in violation &&
    'accepted' in violation
  );
}

export class ReproducibilityCellHighlighter {
  private _tracker: INotebookTracker;
  private _panel: ReproducibilityMetadataPanel;
  private _stalenessManagers = new Map<string, StalenessManager>();
  private _pendingRestartUpdate = new Set<string>(); // notebook paths awaiting update after restart
  private _executedInSession = new Map<string, Set<string>>(); // notebook path -> set of executed cell IDs

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

    // Listen for cell execution to track executed cells and update highlight
    NotebookActions.executed.connect((_sender, args) => {
      if (args.notebook === notebook.content) {
        // Track this cell as executed in current session
        let executed = this._executedInSession.get(path);
        if (!executed) {
          executed = new Set<string>();
          this._executedInSession.set(path, executed);
        }
        executed.add(args.cell.model.id);

        const stalenessManager = this.getStalenessManager(notebook);
        const cellOrder = this._getCurrentCellOrder(notebook);
        this.updateCell(args.cell, stalenessManager, cellOrder, path);
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

      // Remove flowbook notice outputs (staleness and violation notices)
      const codeModel = cell.model as ICodeCellModel;
      const outputs = codeModel.outputs;
      const cleanOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const meta = (out as any).metadata || {};
        // Keep outputs that are NOT flowbook notices (including legacy flowbook_error_notice)
        if (
          !meta.flowbook_staleness_notice &&
          !meta.flowbook_violation_notice &&
          !meta.flowbook_error_notice
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
    cellOrder: string[]
  ): string {
    // Handle frontend reasons with full context
    if (isFrontendReason(reason)) {
      return this._formatFrontendReason(reason, cellOrder);
    }

    // Handle backend reasons (from kernel staleness_reasons)
    return this._formatBackendReason(reason, cellOrder);
  }

  /**
   * Format a frontend-computed staleness reason.
   */
  private _formatFrontendReason(
    reason: IFrontendStalenessReason,
    cellOrder: string[]
  ): string {
    if (reason.type === 'source_edited') {
      return 'Source code was edited';
    }

    if (!reason.causing_cell) {
      return reason.message;
    }

    const causingIdx = cellOrder.indexOf(reason.causing_cell);
    // Note: indexToAlpha already returns with @ prefix
    const causingRef =
      causingIdx >= 0 ? indexToAlpha(causingIdx) : 'a deleted cell';

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
      return `Writes ${parts.join(', ')}, which was read by ${causingRef}`;
    }

    // StaleFwd: the causing cell modified variables this cell reads
    if (parts.length > 0) {
      return `${parts.join(', ')} modified by ${causingRef}`;
    }

    if (reason.type === 'unknown') {
      return `Dependencies changed by ${causingRef}`;
    }

    return reason.message;
  }

  /**
   * Format a backend staleness reason from kernel.
   */
  private _formatBackendReason(
    reason: IStalenessReason,
    cellOrder: string[]
  ): string {
    // Backend reasons have cell_id instead of causing_cell
    const cellId = 'cell_id' in reason ? reason.cell_id : undefined;
    const expectedCellId =
      'expected_cell_id' in reason ? reason.expected_cell_id : undefined;
    const loc = 'loc' in reason ? reason.loc : undefined;

    // Note: indexToAlpha already returns with @ prefix
    let causingRef = '';
    if (cellId) {
      const causingIdx = cellOrder.indexOf(cellId);
      causingRef =
        causingIdx >= 0 ? indexToAlpha(causingIdx) : 'a deleted cell';
    }

    let expectedRef = '';
    let expectedCellDeleted = false;
    if (expectedCellId) {
      const expectedIdx = cellOrder.indexOf(expectedCellId);
      if (expectedIdx >= 0) {
        expectedRef = indexToAlpha(expectedIdx);
      } else {
        expectedCellDeleted = true;
        expectedRef = 'a deleted cell';
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
          return `\`${loc}\` was modified by ${causingRef}`;
        }
        return causingRef
          ? `Input modified by ${causingRef}`
          : 'Input was modified';
      case 'write_overlap':
        // Write overlap: both cells write to same location
        if (loc && causingRef) {
          return `Write overlap: \`${loc}\` also written by ${causingRef}`;
        }
        return causingRef
          ? `Write overlap with ${causingRef}`
          : 'Write overlap detected';
      case 'skipped_upstream':
        // Re-running won't help - need to run the expected cell first
        // If expected cell was deleted, say so clearly
        if (expectedCellDeleted) {
          return loc
            ? `\`${loc}\` is from a deleted cell`
            : 'Source cell was deleted';
        }
        if (loc && expectedRef) {
          return `Run ${expectedRef} first (\`${loc}\` is from wrong source)`;
        }
        return expectedRef
          ? `Run ${expectedRef} first`
          : 'Upstream cell was skipped';
      case 'backward_stale':
        if (loc && causingRef) {
          return `Write conflict on \`${loc}\` with ${causingRef}`;
        }
        return 'Write conflict detected';
      case 'no_read_before_write':
        // NoReadBeforeWrite failed - reads from later cell (forward contamination)
        if (loc && causingRef) {
          return `Reads \`${loc}\` from later cell ${causingRef} (forward contamination)`;
        }
        return 'Reads from a later cell';
      case 'reads_residual_write':
        return loc
          ? `Source of \`${loc}\` was deleted`
          : 'Source cell was deleted';
      case 'order_changed':
        return 'Cell order changed';
      case 'no_write_after_read':
        // NoWriteAfterRead failed - wrote to location read by earlier cell (backward mutation)
        if (loc && causingRef) {
          return `Wrote \`${loc}\` read by earlier cell ${causingRef} (backward mutation)`;
        }
        return causingRef
          ? `Wrote to variable read by ${causingRef}`
          : 'Backward mutation detected';
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

    // Skip staleness notice if violation is present (it's more specific)
    if (hasViolationMetadata || hasViolationNotice) {
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

      const message = this._formatStalenessMessage(reason, cellOrder);

      // Escape HTML in the message but preserve backtick-wrapped code
      const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Use different label for writer_conflict (potential violation vs stale dependency)
      const isWriterConflict = reason.type === 'writer_conflict';
      const label = isWriterConflict ? 'Unresolved Violation' : '';
      const plainText = label
        ? `\u26a0\ufe0f ${label}: ${message}`
        : `\u26a0\ufe0f ${message}`;

      const infoIcon =
        '<span class="flowbook-staleness-info-icon" title="This cell may produce different results than shown">ⓘ</span>';
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
   * Handles both legacy IViolationInfo and new IPredicateViolation formats.
   *
   * For IPredicateViolation:
   * - accepted=true: yellow info box (violation accepted, cell stays CLEAN)
   * - accepted=false: red error box (violation rejected, execution rolled back)
   */
  private _updateViolationOutput(cell: Cell, cellOrder: string[]): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Get violation metadata from cell (can be IViolationInfo or IPredicateViolation)
    const violation = cell.model.getMetadata('flowbook_violation') as
      | IViolationInfo
      | IPredicateViolation
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

    // Handle new predicate violation format
    if (isPredicateViolation(violation)) {
      const locs = violation.locations.map(l => '`' + l + '`').join(', ');
      const htmlLocs = locs.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Get causer cell reference (strip any existing @ prefix)
      // If cell was deleted, show "a deleted cell" instead of raw ID
      let causerRef: string | null = null;
      if (violation.causer_cell && typeof violation.causer_cell === 'string') {
        const rawCauser = violation.causer_cell.startsWith('@')
          ? violation.causer_cell.slice(1)
          : violation.causer_cell;
        const causerIdx = cellOrder.indexOf(rawCauser);
        causerRef = causerIdx >= 0 ? indexToAlpha(causerIdx) : 'a deleted cell';
      }

      // Build message based on predicate type
      // Note: indexToAlpha already returns with @ prefix, causerRef is either "@A" or "a deleted cell"
      let message: string;
      switch (violation.predicate) {
        case 'no_write_after_read':
          message = causerRef
            ? `Wrote ${htmlLocs} read by ${causerRef}`
            : `Wrote ${htmlLocs} read by earlier cell`;
          break;
        case 'no_read_before_write':
          message = causerRef
            ? `Read ${htmlLocs} from ${causerRef}`
            : `Read ${htmlLocs} from later cell`;
          break;
        case 'no_read_and_write':
          message = `Read and wrote ${htmlLocs} in same cell`;
          break;
        case 'write_before_read':
          message = `${htmlLocs} not defined by earlier cells`;
          break;
        default:
          message = violation.message;
      }

      const plainMessage = message.replace(/<code>([^<]+)<\/code>/g, '`$1`');

      // Style based on whether violation was accepted
      const isAccepted = violation.accepted;
      const icon = isAccepted ? '\u26a0\ufe0f' : '\u274c';
      const cssClass = isAccepted
        ? 'flowbook-staleness-notice'
        : 'flowbook-error-notice';

      const plainText = `${icon} ${plainMessage}`;
      const noticeOutput: IOutput = {
        output_type: 'display_data',
        data: {
          'text/html': `<div class="${cssClass}">${icon} ${message}</div>`,
          'text/plain': plainText
        },
        metadata: {
          flowbook_violation_notice: true,
          flowbook_predicate_accepted: isAccepted
        }
      };

      // Check if message matches current notice
      if (hasViolationNotice && existingPlainText === plainText) {
        return; // Already up to date
      }

      // Add error class to cell if violation was rejected
      if (!isAccepted) {
        cell.node.classList.add('flowbook-cell-error');
      } else {
        cell.node.classList.remove('flowbook-cell-error');
      }

      // Build new output array
      const allOutputs: IOutput[] = [noticeOutput];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const isViolationNotice =
          (out as any).metadata?.flowbook_violation_notice === true;
        const isErrorNotice =
          (out as any).metadata?.flowbook_error_notice === true;
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
          !isErrorNotice &&
          !isKernelError &&
          !isKernelPredicateViolation
        ) {
          allOutputs.push(out);
        }
      }
      outputs.fromJSON(allOutputs);
      return;
    }

    // Handle legacy IViolationInfo format
    if (violation && 'mutating_cell' in violation && violation.mutating_cell) {
      // Compute message with current @A references
      // Note: indexToAlpha already returns with @ prefix
      // Use "a deleted cell" for cells that no longer exist
      const mutIdx = cellOrder.indexOf(violation.mutating_cell);
      const affIdx = cellOrder.indexOf(violation.affected_cell);
      const mutRef = mutIdx >= 0 ? indexToAlpha(mutIdx) : 'a deleted cell';
      const affRef = affIdx >= 0 ? indexToAlpha(affIdx) : 'a deleted cell';
      const vars = violation.variables.map(v => '`' + v + '`').join(', ');

      // Different message format based on violation type
      // If mutating cell is no longer in notebook, treat as deleted cell dependency
      const mutatingCellDeleted =
        mutIdx < 0 && violation.mutating_cell !== '<deleted>';
      const isForwardContamination =
        violation.type === 'forward_dependency' && !mutatingCellDeleted;
      const isDeletedCellDependency =
        violation.type === 'deleted_cell_dependency' || mutatingCellDeleted;
      const isContaminationLike =
        isForwardContamination || isDeletedCellDependency;
      let plainText: string;
      let noticeOutput: IOutput;

      if (isDeletedCellDependency) {
        // Deleted cell dependency: reading a variable written by a cell that was deleted
        const htmlVars = vars.replace(/`([^`]+)`/g, '<code>$1</code>');
        plainText = `\u274c Deleted Cell Dependency: ${vars} written by a deleted cell. Re-run an upstream cell that defines these variables.`;
        noticeOutput = {
          output_type: 'display_data',
          data: {
            'text/html': `<div class="flowbook-violation-notice"><b>\u274c Deleted Cell Dependency: ${htmlVars} written by a deleted cell. Re-run an upstream cell that defines these variables.</b></div>`,
            'text/plain': plainText
          },
          metadata: {
            flowbook_violation_notice: true,
            flowbook_is_contamination: true
          }
        };
      } else if (isForwardContamination) {
        // Forward contamination: reading cell read from later writing cell
        const htmlVars = vars.replace(/`([^`]+)`/g, '<code>$1</code>');
        plainText = `\u274c Forward Contamination: ${vars} written by downstream cell ${mutRef}. Re-run upstream cells to restore reproducible values.`;
        noticeOutput = {
          output_type: 'display_data',
          data: {
            'text/html': `<div class="flowbook-violation-notice"><b>\u274c Forward Contamination: ${htmlVars} written by downstream cell ${mutRef}. Re-run upstream cells to restore reproducible values.</b></div>`,
            'text/plain': plainText
          },
          metadata: {
            flowbook_violation_notice: true,
            flowbook_is_contamination: true
          }
        };
      } else {
        // Backward violation: build detailed message
        const htmlParts: string[] = [];
        const plainParts: string[] = [];

        // Header line
        const headerMsg = `Cell ${mutRef} modified ${vars} which Cell ${affRef} (earlier) reads.`;
        const headerHtml = headerMsg.replace(/`([^`]+)`/g, '<code>$1</code>');
        htmlParts.push(
          `<div class="flowbook-violation-header">\u274c <b>Not Reproducible</b>: ${headerHtml}</div>`
        );
        plainParts.push(`\u274c Not Reproducible: ${headerMsg}`);

        // What the earlier cell reads (structural_reads_detail)
        if (violation.structural_reads_detail) {
          const readsHtml: string[] = [];
          const readsPlain: string[] = [];
          for (const [varName, attrs] of Object.entries(
            violation.structural_reads_detail
          )) {
            for (const [attr, value] of Object.entries(attrs)) {
              readsHtml.push(
                `<li><code>${varName}.${attr}</code> \u2192 ${this._escapeHtml(value)}</li>`
              );
              readsPlain.push(`  \u2022 ${varName}.${attr} \u2192 ${value}`);
            }
          }
          if (readsHtml.length > 0) {
            htmlParts.push(
              `<div class="flowbook-violation-section"><b>What Cell ${affRef} read:</b><ul>${readsHtml.join('')}</ul></div>`
            );
            plainParts.push(`What Cell ${affRef} read:`);
            plainParts.push(...readsPlain);
          }
        }

        // What this cell changed (changes_detail)
        if (violation.changes_detail && violation.changes_detail.length > 0) {
          const changesHtml = violation.changes_detail
            .map(c => `<li>${this._escapeHtml(c)}</li>`)
            .join('');
          const changesPlain = violation.changes_detail.map(
            c => `  \u2022 ${c}`
          );
          htmlParts.push(
            `<div class="flowbook-violation-section"><b>What Cell ${mutRef} changed:</b><ul>${changesHtml}</ul></div>`
          );
          plainParts.push(`What Cell ${mutRef} changed:`);
          plainParts.push(...changesPlain);
        }

        plainText = plainParts.join('\n');

        noticeOutput = {
          output_type: 'display_data',
          data: {
            'text/html': `<div class="flowbook-violation-notice">${htmlParts.join('')}</div>`,
            'text/plain': plainText
          },
          metadata: { flowbook_violation_notice: true }
        };
      }

      // Check if message matches current notice
      if (hasViolationNotice && existingPlainText === plainText) {
        return; // Already up to date
      }

      // Build new output array:
      // 1. Staleness notice (if exists AND not showing contamination - contamination implies stale)
      // 2. Violation/Contamination notice
      // 3. Other outputs (excluding old notices and kernel outputs)
      const allOutputs: IOutput[] = [];

      // First, add staleness notice if present (but skip if contamination - it already implies stale)
      if (!isContaminationLike) {
        for (let i = 0; i < outputs.length; i++) {
          const out = outputs.get(i).toJSON() as IOutput;
          if ((out as any).metadata?.flowbook_staleness_notice === true) {
            allOutputs.push(out);
            break;
          }
        }
      }

      // Add the new notice
      allOutputs.push(noticeOutput);

      // Add remaining outputs, filtering out:
      // - Old violation/contamination notice
      // - Staleness notice (already added)
      // - Kernel's ReproducibilityViolation error
      // - Kernel's brief "Backward violation" display_data
      // - Kernel's "Forward Contamination" or "Deleted cell" stderr stream
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const isViolationNotice =
          (out as any).metadata?.flowbook_violation_notice === true;
        const isStalenessNotice =
          (out as any).metadata?.flowbook_staleness_notice === true;
        const isKernelError =
          out.output_type === 'error' &&
          (out as any).ename === 'ReproducibilityViolation';
        const plainText = (out as any).data?.['text/plain'] || '';
        const isKernelBriefViolation =
          out.output_type === 'display_data' &&
          (plainText.includes('Backward violation') ||
            plainText.includes('Forward contamination') ||
            plainText.includes('Deleted cell conflict'));
        // Stream text can be string or string[] - normalize to string
        const streamText = Array.isArray((out as any).text)
          ? (out as any).text.join('')
          : (out as any).text || '';
        const isKernelContaminationStderr =
          out.output_type === 'stream' &&
          (out as any).name === 'stderr' &&
          (streamText.includes('Forward Contamination') ||
            streamText.includes('Deleted cell'));

        if (
          !isViolationNotice &&
          !isStalenessNotice &&
          !isKernelError &&
          !isKernelBriefViolation &&
          !isKernelContaminationStderr
        ) {
          allOutputs.push(out);
        }
      }

      outputs.fromJSON(allOutputs);
    } else {
      // No violation metadata - only clear notices that WE created (not kernel's predicate_violation)
      // The kernel's predicate_violation output should remain until we process it
      let hasOurNotices = false;
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as any;
        if (
          out.metadata?.flowbook_violation_notice ||
          out.metadata?.flowbook_error_notice
        ) {
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
          const isOurNotice =
            meta.flowbook_violation_notice || meta.flowbook_error_notice;
          if (!isOurNotice) {
            allOutputs.push(out);
          }
        }
        outputs.fromJSON(allOutputs);
      }
    }
  }

  /**
   * Escape HTML special characters in a string.
   */
  private _escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
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

    const newMessage = this._formatStalenessMessage(staleness, cellOrder);
    if (newMessage !== staleness.message) {
      cell.model.setMetadata('flowbook_staleness', {
        ...staleness,
        message: newMessage
      });
    }
  }

  /**
   * Recompute the flowbook_violation metadata message with current @A references.
   */
  private _updateViolationMetadata(cell: Cell, cellOrder: string[]): void {
    const violation = cell.model.getMetadata('flowbook_violation') as
      | IViolationInfo
      | undefined;
    if (!violation || !violation.mutating_cell) {
      return;
    }

    const mutIdx = cellOrder.indexOf(violation.mutating_cell);
    const affIdx = cellOrder.indexOf(violation.affected_cell);
    // Note: indexToAlpha already returns with @ prefix
    // Use "a deleted cell" for cells that no longer exist
    const mutRef = mutIdx >= 0 ? indexToAlpha(mutIdx) : 'a deleted cell';
    const affRef = affIdx >= 0 ? indexToAlpha(affIdx) : 'a deleted cell';
    const vars = violation.variables.map(v => '`' + v + '`').join(', ');
    const newMessage = `Cell ${mutRef} modified ${vars} read by ${affRef}`;

    if (newMessage !== violation.message) {
      cell.model.setMetadata('flowbook_violation', {
        ...violation,
        message: newMessage
      });
    }
  }
}
