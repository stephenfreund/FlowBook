/**
 * Execution hook for FlowBook kernel - extracts reproducibility metadata
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import {
  INotebookTracker,
  Notebook,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { CellChange } from '@jupyter/ydoc';
import { IOutput } from '@jupyterlab/nbformat';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import {
  IReproducibilityMetadata,
  IStalenessReason,
  IViolationInfo
} from './types';
import { indexToAlpha } from '../cellindexutils';

export class ReproducibilityExecutionHookManager {
  private _app: JupyterFrontEnd;
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter;
  private _editTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private _executedCells: Set<string> = new Set();
  private _attachedKernel: Kernel.IKernelConnection | null = null;
  private _listenedCellIds: Set<string> = new Set();
  private _silentEditMsgIds: Set<string> = new Set();

  constructor(
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    highlighter: ReproducibilityCellHighlighter
  ) {
    this._app = app;
    this._tracker = tracker;
    this._highlighter = highlighter;
    this._setupHooks();
  }

  private _setupHooks(): void {
    // Listen for cell execution completion
    NotebookActions.executed.connect(this._onCellExecuted, this);

    // Listen for cell execution start to set cell_order via magic
    NotebookActions.executionScheduled.connect(
      this._onExecutionScheduled,
      this
    );

    // [EDIT transition (§2.3)] Listen for cell content changes
    this._tracker.currentChanged.connect(this._setupCellEditListener, this);

    // Listen for IOPub messages (catches silent magic responses like %cell_edited)
    this._tracker.currentChanged.connect(this._setupIOPubListener, this);

    // Also set up listeners for already-open notebook (signal may have fired before we subscribed)
    console.log(
      'ReproducibilityExecutionHookManager: currentWidget =',
      this._tracker.currentWidget?.context?.path
    );
    if (this._tracker.currentWidget) {
      this._setupCellEditListener();
      this._setupIOPubListener();
    }

    console.log(
      'ReproducibilityExecutionHookManager: Execution hooks installed'
    );
  }

  /**
   * [EDIT transition (§2.3)] Set up listeners for cell content changes.
   * When a code cell's source changes and the cell was previously executed,
   * send %cell_edited <cell_id> to the kernel with debouncing.
   *
   * Also watches for newly inserted cells so they get listeners too.
   */
  private _setupCellEditListener(): void {
    const panel = this._tracker.currentWidget;
    console.log(
      'ReproducibilityExecutionHook: _setupCellEditListener called, panel =',
      panel?.context?.path
    );
    if (!panel) {
      return;
    }

    const notebook = panel.content;
    console.log(
      `ReproducibilityExecutionHook: Setting up edit listeners for ${notebook.widgets.length} cells`
    );

    // Attach listeners to all existing code cells
    for (let i = 0; i < notebook.widgets.length; i++) {
      this._attachCellEditListener(notebook.widgets[i]);
    }

    // Watch for newly inserted cells
    notebook.model?.cells.changed.connect(() => {
      for (let i = 0; i < notebook.widgets.length; i++) {
        this._attachCellEditListener(notebook.widgets[i]);
      }
    });
  }

  /**
   * Attach a content-change listener to a single cell (idempotent).
   */
  private _attachCellEditListener(cell: Cell): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const cellId = cell.model.id;
    if (this._listenedCellIds.has(cellId)) {
      return;
    }
    this._listenedCellIds.add(cellId);
    console.log(
      `ReproducibilityExecutionHook: Attached edit listener to cell ${cellId}`
    );

    const model = cell.model as ICodeCellModel;
    model.sharedModel.changed.connect((_sender: any, change: CellChange) => {
      console.log(
        `ReproducibilityExecutionHook: Cell ${cellId} changed, sourceChange=${!!change.sourceChange}`
      );
      // Only react to source text edits, not output/metadata/executionCount changes
      if (change.sourceChange) {
        this._onCellContentChanged(cellId);
      }
    });
  }

  /**
   * [EDIT transition (§2.3)] Handle cell content change with debouncing.
   */
  private _onCellContentChanged(cellId: string): void {
    console.log(
      `ReproducibilityExecutionHook: _onCellContentChanged(${cellId}), inExecutedCells=${this._executedCells.has(cellId)}`
    );
    // Only notify kernel about cells that have been previously executed
    if (!this._executedCells.has(cellId)) {
      return;
    }

    // Debounce: cancel previous timer for this cell
    const existing = this._editTimers.get(cellId);
    if (existing) {
      clearTimeout(existing);
    }

    // Set new timer (1s debounce)
    const timer = setTimeout(() => {
      console.log(
        `ReproducibilityExecutionHook: Debounce complete for ${cellId}, sending cell_edited`
      );
      this._sendCellEdited(cellId);
      this._editTimers.delete(cellId);
    }, 1000);

    this._editTimers.set(cellId, timer);
  }

  /**
   * [EDIT transition (§2.3)] Send %cell_edited magic to kernel.
   */
  private _sendCellEdited(cellId: string): void {
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    const session = panel.sessionContext.session;
    if (session && session.kernel) {
      const future = session.kernel.requestExecute({
        code: `%cell_edited ${cellId}`,
        silent: true,
        store_history: false
      });
      const msgId = future.msg.header.msg_id;
      this._silentEditMsgIds.add(msgId);
      console.log(
        `ReproducibilityExecutionHook: Sent cell_edited for ${cellId}, msgId = ${msgId}`
      );
    }
  }

  /**
   * Set up an IOPub listener on the current kernel to catch display_data
   * messages containing flowbook metadata — including those from silent
   * magic executions like %cell_edited that never reach a cell's output area.
   */
  private _setupIOPubListener(): void {
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    this._connectIOPub(panel);

    // Re-attach when the kernel restarts or changes
    panel.sessionContext.kernelChanged.connect(() => {
      this._connectIOPub(panel);
    });
  }

  private _connectIOPub(panel: NotebookPanel): void {
    const kernel = panel.sessionContext.session?.kernel;
    if (!kernel || kernel === this._attachedKernel) {
      return;
    }

    this._attachedKernel = kernel;
    kernel.iopubMessage.connect(this._onIOPubMessage, this);

    console.log(
      'ReproducibilityExecutionHook: IOPub listener attached to kernel'
    );
  }

  private _onIOPubMessage(
    _sender: Kernel.IKernelConnection,
    msg: KernelMessage.IIOPubMessage
  ): void {
    if (msg.header.msg_type !== 'display_data') {
      return;
    }

    // Only process messages from our silent %cell_edited executions.
    // Regular cell execution metadata is handled by _onCellExecuted.
    const parentHeader = msg.parent_header;
    const parentMsgId = 'msg_id' in parentHeader ? parentHeader.msg_id : '';

    console.log(
      'ReproducibilityExecutionHook: IOPub display_data received, parentMsgId =',
      parentMsgId,
      ', tracked =',
      this._silentEditMsgIds.has(parentMsgId),
      ', trackedIds =',
      [...this._silentEditMsgIds]
    );

    if (!this._silentEditMsgIds.has(parentMsgId)) {
      return;
    }
    this._silentEditMsgIds.delete(parentMsgId);

    const content = (msg as KernelMessage.IDisplayDataMsg).content;
    const flowbook = content.metadata?.flowbook as
      | IReproducibilityMetadata
      | undefined;
    if (!flowbook) {
      return;
    }

    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    this._processMetadataUpdate(panel, flowbook);

    console.log(
      'ReproducibilityExecutionHook: IOPub cell_edited update, stale_cells =',
      flowbook.stale_cells
    );
  }

  /**
   * Called before cell execution - send %notebook_structure magic to set cell order
   */
  private _onExecutionScheduled(
    _sender: any,
    args: { notebook: Notebook; cell: Cell }
  ): void {
    const { notebook } = args;

    // Get the notebook panel
    const panel = this._tracker.currentWidget;
    if (!panel || panel.content !== notebook) {
      return;
    }

    // Build cell order array (only code cells)
    const cellOrder: string[] = [];
    for (let i = 0; i < notebook.widgets.length; i++) {
      const c = notebook.widgets[i];
      if (c.model.type === 'code') {
        cellOrder.push(c.model.id);
      }
    }

    // Send %notebook_structure magic to kernel
    const session = panel.sessionContext.session;
    if (session && session.kernel && cellOrder.length > 0) {
      const magicCommand = `%notebook_structure ${cellOrder.join(' ')}`;
      session.kernel.requestExecute({
        code: magicCommand,
        silent: true,
        store_history: false
      });
      console.log(
        `ReproducibilityExecutionHook: Sent notebook_structure with ${cellOrder.length} cells`
      );
    }
  }

  private _extractReproducibilityMetadata(
    outputs: IOutput[]
  ): IReproducibilityMetadata | null {
    console.log(
      `ReproducibilityExecutionHook: Checking ${outputs.length} outputs for flowbook metadata`
    );

    for (const output of outputs) {
      console.log(
        `ReproducibilityExecutionHook: Output type = ${output.output_type}`
      );

      if (output.output_type !== 'display_data') {
        continue;
      }

      const metadata = (output as any).metadata;
      console.log(
        'ReproducibilityExecutionHook: display_data metadata =',
        metadata
      );

      if (!metadata?.flowbook) {
        console.log('ReproducibilityExecutionHook: No flowbook in metadata');
        continue;
      }

      console.log(
        'ReproducibilityExecutionHook: Found flowbook metadata!',
        metadata.flowbook
      );
      return metadata.flowbook as IReproducibilityMetadata;
    }
    console.log(
      'ReproducibilityExecutionHook: No flowbook metadata found in any output'
    );
    return null;
  }

  private _onCellExecuted(
    _sender: any,
    args: { notebook: Notebook; cell: Cell }
  ): void {
    const { notebook, cell } = args;

    if (cell.model.type !== 'code') {
      return;
    }

    // [EDIT transition (§2.3)] Track executed cells for edit detection
    this._executedCells.add(cell.model.id);

    // Get the notebook panel
    const panel = this._tracker.currentWidget;
    if (!panel || panel.content !== notebook) {
      return;
    }

    // Get outputs
    const codeModel = cell.model as ICodeCellModel;
    const outputs: IOutput[] = [];
    for (let i = 0; i < codeModel.outputs.length; i++) {
      outputs.push(codeModel.outputs.get(i).toJSON() as IOutput);
    }

    // Extract reproducibility metadata
    const reproducibilityMetadata =
      this._extractReproducibilityMetadata(outputs);
    if (!reproducibilityMetadata) {
      return;
    }

    // Store metadata on cell
    cell.model.setMetadata('flowbook', reproducibilityMetadata);

    // Process staleness reasons and update manager
    this._processMetadataUpdate(panel, reproducibilityMetadata);

    // Store or clear violation metadata on the executing cell
    const cellOrder = this._getCurrentCellOrder(panel);
    if (reproducibilityMetadata.violation) {
      const v = reproducibilityMetadata.violation;
      const mutIdx = cellOrder.indexOf(v.mutating_cell);
      const affIdx = cellOrder.indexOf(v.affected_cell);
      const mutRef = mutIdx >= 0 ? indexToAlpha(mutIdx) : v.mutating_cell;
      const affRef = affIdx >= 0 ? indexToAlpha(affIdx) : v.affected_cell;
      const violationInfo: IViolationInfo = {
        type: v.violation_type || 'backward_mutation',
        mutating_cell: v.mutating_cell,
        affected_cell: v.affected_cell,
        variables: v.variables,
        message: `Cell ${mutRef} modified ${v.variables.map(vv => '`' + vv + '`').join(', ')} read by ${affRef}`
      };
      cell.model.setMetadata('flowbook_violation', violationInfo);

      // Immediately update violation output (don't wait for staleness signal)
      this._updateViolationOutput(cell, violationInfo, cellOrder);
    } else {
      cell.model.deleteMetadata('flowbook_violation');
      // Clear any existing violation notice
      this._clearViolationOutput(cell);
    }

    console.log(
      `ReproducibilityExecutionHook: Extracted metadata for cell ${cell.model.id}:`,
      reproducibilityMetadata
    );
  }

  /**
   * Get current cell order from notebook (only code cells)
   */
  private _getCurrentCellOrder(panel: NotebookPanel): string[] {
    const cellOrder: string[] = [];
    const cells = panel.content.widgets;
    for (let i = 0; i < cells.length; i++) {
      if (cells[i].model.type === 'code') {
        cellOrder.push(cells[i].model.id);
      }
    }
    return cellOrder;
  }

  /**
   * Shared method to compute staleness reasons, store metadata, and update staleness manager.
   * Used by both _onCellExecuted (cell output path) and _onIOPubMessage (silent magic path).
   */
  private _processMetadataUpdate(
    panel: NotebookPanel,
    metadata: IReproducibilityMetadata
  ): void {
    const stalenessManager = this._highlighter.getStalenessManager(panel);

    // Reason computation and metadata storage are best-effort.
    // updateFromMetadata MUST always run to keep staleness CSS correct.
    try {
      const oldStale = new Set(stalenessManager.staleCells);
      const newStaleSet = new Set(metadata.stale_cells);
      const cellOrder = this._getCurrentCellOrder(panel);

      // Compute newly-stale cells
      const newlyStale = [...newStaleSet].filter(id => !oldStale.has(id));

      // Compute reason for each newly-stale cell
      for (const staleCellId of newlyStale) {
        const reason = this._computeStalenessReason(
          panel,
          staleCellId,
          metadata,
          cellOrder
        );
        stalenessManager.setReason(staleCellId, reason);

        // Store structured metadata on the cell
        const staleCell = this._findCell(panel, staleCellId);
        if (staleCell) {
          staleCell.model.setMetadata('flowbook_staleness', reason);
        }
      }

      // Clear flowbook_staleness metadata from cells that became fresh
      const freshened = [...oldStale].filter(id => !newStaleSet.has(id));
      for (const freshCellId of freshened) {
        const freshCell = this._findCell(panel, freshCellId);
        if (freshCell) {
          freshCell.model.deleteMetadata('flowbook_staleness');
        }
      }
    } catch (e) {
      console.error(
        'ReproducibilityExecutionHook: Error computing staleness reasons:',
        e
      );
    }

    // Update staleness manager (triggers signal → CellHighlighter)
    stalenessManager.updateFromMetadata(metadata);

    // Notify command system so context menu items re-evaluate isEnabled
    this._app.commands.notifyCommandChanged('flowbook:exec-restore');
  }

  /**
   * Compute why a cell became stale.
   */
  private _computeStalenessReason(
    panel: NotebookPanel,
    staleCellId: string,
    metadata: IReproducibilityMetadata,
    cellOrder: string[]
  ): IStalenessReason {
    const causingCellId = metadata.cell_id;

    // Edit case: the cell that triggered the update is the stale cell itself
    // and there are no changed_variables (pure source edit)
    if (
      staleCellId === causingCellId &&
      (!metadata.changed_variables || metadata.changed_variables.length === 0)
    ) {
      return {
        type: 'source_edited',
        causing_cell: causingCellId,
        message: 'Source code was edited'
      };
    }

    // StaleFwd case: look up stale cell's stored reads, intersect with changed_variables
    const staleCell = this._findCell(panel, staleCellId);
    const storedMeta = staleCell?.model.metadata as any;
    const storedFlowbook = storedMeta?.flowbook as
      | IReproducibilityMetadata
      | undefined;

    const causingIdx = cellOrder.indexOf(causingCellId);
    const causingRef =
      causingIdx >= 0 ? indexToAlpha(causingIdx) : causingCellId;

    // Intersect variable reads with changed_variables
    const changedVars = metadata.changed_variables || [];
    const cellReads = storedFlowbook?.reads || [];
    const intersectVars = changedVars.filter(v => cellReads.includes(v));

    // Intersect column reads with column_changed
    const columnChanged = metadata.column_changed || {};
    const cellColumnReads = storedFlowbook?.column_reads || {};
    const intersectCols: { [key: string]: string[] } = {};
    for (const [dfName, changedCols] of Object.entries(columnChanged)) {
      const readCols = cellColumnReads[dfName];
      if (readCols) {
        const overlap = changedCols.filter(c => readCols.includes(c));
        if (overlap.length > 0) {
          intersectCols[dfName] = overlap;
        }
      }
    }

    // Build variable parts for the message
    const parts: string[] = [];
    for (const v of intersectVars) {
      parts.push('`' + v + '`');
    }
    for (const [dfName, cols] of Object.entries(intersectCols)) {
      for (const col of cols) {
        parts.push('`' + dfName + '.' + col + '`');
      }
    }

    if (parts.length > 0) {
      return {
        type: 'variable_modified',
        causing_cell: causingCellId,
        variables: intersectVars.length > 0 ? intersectVars : undefined,
        columns:
          Object.keys(intersectCols).length > 0 ? intersectCols : undefined,
        message: `${parts.join(', ')} modified by ${causingRef}`
      };
    }

    // WriterCheck case: stale cell WRITES to variables the causing cell READS
    // (EXEC-RESTORE marks cells that would cause BackConflict if run)
    const causingCellReads = metadata.reads || [];
    const staleCellWrites = storedFlowbook?.writes || [];
    const writerConflictVars = staleCellWrites.filter(v =>
      causingCellReads.includes(v)
    );
    if (writerConflictVars.length > 0) {
      const varParts = writerConflictVars.map(v => '`' + v + '`');
      return {
        type: 'writer_conflict',
        causing_cell: causingCellId,
        variables: writerConflictVars,
        message: `Writes ${varParts.join(', ')}, which was read by ${causingRef}`
      };
    }

    // Fallback: we know there was a change but can't identify the specific variables
    return {
      type: 'unknown',
      causing_cell: causingCellId,
      message: `Dependencies changed by ${causingRef}`
    };
  }

  /**
   * Find a cell widget by ID in the notebook
   */
  private _findCell(panel: NotebookPanel, cellId: string): Cell | null {
    const cells = panel.content.widgets;
    for (let i = 0; i < cells.length; i++) {
      if (cells[i].model.id === cellId) {
        return cells[i];
      }
    }
    return null;
  }

  /**
   * Update cell outputs to show styled violation notice.
   * Replaces kernel's raw error output with a styled display_data.
   */
  private _updateViolationOutput(
    cell: Cell,
    violation: IViolationInfo,
    cellOrder: string[]
  ): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Compute message with current @A references
    const mutIdx = cellOrder.indexOf(violation.mutating_cell);
    const affIdx = cellOrder.indexOf(violation.affected_cell);
    const mutRef = mutIdx >= 0 ? indexToAlpha(mutIdx) : violation.mutating_cell;
    const affRef = affIdx >= 0 ? indexToAlpha(affIdx) : violation.affected_cell;
    const vars = violation.variables.map(v => '`' + v + '`').join(', ');

    // Different message format for forward contamination vs backward violation
    const isForwardContamination = violation.type === 'forward_dependency';
    let message: string;
    let noticeOutput: IOutput;

    if (isForwardContamination) {
      // Forward contamination: reading cell read from later writing cell
      message = `${vars} written by downstream cell ${mutRef}`;
      const hint = 'Right-click → "Run with upstream state" to fix';
      const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');
      noticeOutput = {
        output_type: 'display_data',
        data: {
          'text/html': `<div class="flowbook-contamination-notice">\u26a0\ufe0f <b>Contaminated</b>: ${htmlMessage}. ${hint}.</div>`,
          'text/plain': `\u26a0\ufe0f Contaminated: ${message}. ${hint}.`
        },
        metadata: { flowbook_violation_notice: true, flowbook_is_contamination: true }
      };
    } else {
      // Backward violation: mutating cell modified variable read by earlier cell
      message = `Cell ${mutRef} modified ${vars} read by earlier cell ${affRef}`;
      const htmlMessage = message.replace(/`([^`]+)`/g, '<code>$1</code>');
      noticeOutput = {
        output_type: 'display_data',
        data: {
          'text/html': `<div class="flowbook-violation-notice">\u274c <b>Violation</b>: ${htmlMessage}</div>`,
          'text/plain': `\u274c Violation: ${message}`
        },
        metadata: { flowbook_violation_notice: true }
      };
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
    // - Staleness notice (already added, or skip if contamination)
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
      // Also filter staleness notice if showing contamination (contamination implies stale)
      const skipStalenessForContamination =
        isStalenessNotice && isForwardContamination;

      if (
        !isViolationNotice &&
        !isStalenessNotice &&
        !isKernelError &&
        !isKernelBriefViolation &&
        !isKernelContaminationStderr &&
        !skipStalenessForContamination
      ) {
        allOutputs.push(out);
      }
    }

    outputs.fromJSON(allOutputs);
  }

  /**
   * Clear any violation notice from cell outputs.
   */
  private _clearViolationOutput(cell: Cell): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const codeModel = cell.model as ICodeCellModel;
    const outputs = codeModel.outputs;

    // Check if there's a violation notice to remove
    let hasViolationNotice = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as any;
      if (out.metadata?.flowbook_violation_notice === true) {
        hasViolationNotice = true;
        break;
      }
    }

    if (!hasViolationNotice) {
      return;
    }

    // Remove violation notice
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
