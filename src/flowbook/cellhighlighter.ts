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
  IViolationInfo
} from './types';
import { indexToAlpha } from '../cellindexutils';

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
        this._updateCell(args.cell, stalenessManager, cellOrder, path);
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
        this._updateCell(cell, stalenessManager, cellOrder, path);
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
        // Keep outputs that are NOT flowbook notices
        if (!meta.flowbook_staleness_notice && !meta.flowbook_violation_notice) {
          cleanOutputs.push(out);
        }
      }
      // Only update if we removed something
      if (cleanOutputs.length !== outputs.length) {
        outputs.fromJSON(cleanOutputs);
      }

      // Remove staleness CSS class
      cell.node.classList.remove('flowbook-stale-cell');
      cell.node.classList.remove('flowbook-unexecuted-cell');
    });
  }

  private _updateCell(
    cell: Cell,
    stalenessManager: StalenessManager,
    cellOrder: string[],
    notebookPath: string
  ): void {
    const cellId = cell.model.id;
    const isStale = stalenessManager.isCellStale(cellId);
    // Check if cell was executed in current kernel session (not just has an execution count)
    const executedInSession = this._executedInSession.get(notebookPath);
    const isUnexecuted = !executedInSession || !executedInSession.has(cellId);

    console.log(
      `CellHighlighter: Updating cell ${cellId}, isStale=${isStale}, isUnexecuted=${isUnexecuted}`
    );

    // Remove existing highlight classes
    cell.node.classList.remove('flowbook-cell-stale');
    cell.node.classList.remove('flowbook-cell-unexecuted');

    // Add appropriate class
    if (isStale) {
      cell.node.classList.add('flowbook-cell-stale');
      console.log(
        `CellHighlighter: Added .flowbook-cell-stale class to cell ${cellId}`
      );
    } else if (isUnexecuted) {
      cell.node.classList.add('flowbook-cell-unexecuted');
      console.log(
        `CellHighlighter: Added .flowbook-cell-unexecuted class to cell ${cellId}`
      );
    } else {
      console.log(
        `CellHighlighter: Cell ${cellId} is fresh (no highlight class)`
      );
    }

    // Add/remove staleness notice output
    this._updateCellOutput(cell, isStale, stalenessManager, cellOrder);

    // Add/remove violation notice output (styled version of kernel's error)
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
    if (reason.type === 'source_edited') {
      return 'Source code was edited';
    }

    if (!reason.causing_cell) {
      return reason.message;
    }

    const causingIdx = cellOrder.indexOf(reason.causing_cell);
    const causingRef =
      causingIdx >= 0 ? indexToAlpha(causingIdx) : reason.causing_cell;

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

    // Check if there's a violation notice (violation implies specific issue, so skip staleness notice)
    let hasViolationNotice = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as any;
      if (out.metadata?.flowbook_violation_notice === true) {
        hasViolationNotice = true;
        break;
      }
    }

    // Skip staleness notice if violation notice is present (it's more specific)
    if (hasViolationNotice) {
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
      const message = this._formatStalenessMessage(reason, cellOrder);

      // Escape HTML in the message but preserve backtick-wrapped code
      const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');

      // Use different label for writer_conflict (potential violation vs stale dependency)
      const isWriterConflict = reason.type === 'writer_conflict';
      const label = isWriterConflict ? 'Unresolved Violation' : 'Stale';
      const plainText = `\u26a0\ufe0f ${label}: ${message}`;

      const stalenessOutput: IOutput = {
        output_type: 'display_data',
        data: {
          'text/html': `<div class="flowbook-staleness-notice">\u26a0\ufe0f <b>${label}</b>: ${htmlMessage}</div>`,
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
   * This replaces the kernel's raw error output with a styled notice.
   */
  private _updateViolationOutput(cell: Cell, cellOrder: string[]): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Get violation metadata from cell
    const violation = cell.model.getMetadata('flowbook_violation') as
      | IViolationInfo
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

    if (violation && violation.mutating_cell) {
      // Compute message with current @A references
      const mutIdx = cellOrder.indexOf(violation.mutating_cell);
      const affIdx = cellOrder.indexOf(violation.affected_cell);
      const mutRef =
        mutIdx >= 0 ? indexToAlpha(mutIdx) : violation.mutating_cell;
      const affRef =
        affIdx >= 0 ? indexToAlpha(affIdx) : violation.affected_cell;
      const vars = violation.variables.map(v => '`' + v + '`').join(', ');

      // Different message format for forward contamination vs backward violation
      const isForwardContamination = violation.type === 'forward_dependency';
      let message: string;
      let plainText: string;
      let noticeOutput: IOutput;

      if (isForwardContamination) {
        // Forward contamination: reading cell read from later writing cell
        message = `${vars} written by downstream cell ${mutRef}`;
        const hint = 'Right-click → "Run with upstream state" to fix';
        plainText = `\u26a0\ufe0f Contaminated: ${message}. ${hint}.`;
        const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');
        noticeOutput = {
          output_type: 'display_data',
          data: {
            'text/html': `<div class="flowbook-contamination-notice">\u26a0\ufe0f <b>Contaminated</b>: ${htmlMessage}. ${hint}.</div>`,
            'text/plain': plainText
          },
          metadata: { flowbook_violation_notice: true, flowbook_is_contamination: true }
        };
      } else {
        // Backward violation: mutating cell modified variable read by earlier cell
        message = `Cell ${mutRef} modified ${vars} read by earlier cell ${affRef}`;
        plainText = `\u274c Violation: ${message}`;
        const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');
        noticeOutput = {
          output_type: 'display_data',
          data: {
            'text/html': `<div class="flowbook-violation-notice">\u274c <b>Violation</b>: ${htmlMessage}</div>`,
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
      if (!isForwardContamination) {
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
      // - Kernel's "Forward Contamination" stderr stream
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        const isViolationNotice =
          (out as any).metadata?.flowbook_violation_notice === true;
        const isStalenessNotice =
          (out as any).metadata?.flowbook_staleness_notice === true;
        const isKernelError =
          out.output_type === 'error' &&
          (out as any).ename === 'ReproducibilityViolation';
        const isKernelBriefViolation =
          out.output_type === 'display_data' &&
          ((out as any).data?.['text/plain'] || '').includes(
            'Backward violation'
          );
        // Stream text can be string or string[] - normalize to string
        const streamText = Array.isArray((out as any).text)
          ? (out as any).text.join('')
          : (out as any).text || '';
        const isKernelContaminationStderr =
          out.output_type === 'stream' &&
          (out as any).name === 'stderr' &&
          streamText.includes('Forward Contamination');

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
    } else if (hasViolationNotice) {
      // Remove violation notice (violation was cleared)
      const allOutputs: IOutput[] = [];
      for (let i = 0; i < outputs.length; i++) {
        const out = outputs.get(i).toJSON() as IOutput;
        if (!(out as any).metadata?.flowbook_violation_notice) {
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
    if (!staleness || !staleness.causing_cell) {
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
    const mutRef = mutIdx >= 0 ? indexToAlpha(mutIdx) : violation.mutating_cell;
    const affRef = affIdx >= 0 ? indexToAlpha(affIdx) : violation.affected_cell;
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
