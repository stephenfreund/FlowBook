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
  IFrontendStalenessReason,
  IViolationInfo,
  IPredicateViolation
} from './types';
import { indexToAlpha } from '../cellindexutils';

export class ReproducibilityExecutionHookManager {
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter;
  private _editTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private _executedCells: Set<string> = new Set();
  private _attachedKernel: Kernel.IKernelConnection | null = null;
  private _listenedCellIds: Set<string> = new Set();
  private _silentEditMsgIds: Set<string> = new Set();

  constructor(
    _app: JupyterFrontEnd,
    tracker: INotebookTracker,
    highlighter: ReproducibilityCellHighlighter
  ) {
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

    const content = (msg as KernelMessage.IDisplayDataMsg).content;
    const flowbook = content.metadata?.flowbook as
      | IReproducibilityMetadata
      | undefined;

    // Process any display_data with flowbook metadata (from %cell_edited, %flowbook_sync, etc.)
    if (!flowbook) {
      return;
    }

    // Track which messages we've processed from our silent executions
    const parentHeader = msg.parent_header;
    const parentMsgId = 'msg_id' in parentHeader ? parentHeader.msg_id : '';
    if (this._silentEditMsgIds.has(parentMsgId)) {
      this._silentEditMsgIds.delete(parentMsgId);
    }

    console.log(
      'ReproducibilityExecutionHook: IOPub flowbook metadata received, stale_cells =',
      flowbook.stale_cells
    );

    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    this._processMetadataUpdate(panel, flowbook);
  }

  /**
   * Called before cell execution - send %notebook_structure magic to set cell order
   */
  private _onExecutionScheduled(
    _sender: any,
    args: { notebook: Notebook; cell: Cell }
  ): void {
    const { notebook, cell } = args;

    // Get the notebook panel
    const panel = this._tracker.currentWidget;
    if (!panel || panel.content !== notebook) {
      return;
    }

    // Cancel any pending edit timer for this cell.
    // If user edited and then immediately ran, the execution makes the cell fresh,
    // so there's no need to send %cell_edited (which would incorrectly mark it stale).
    const cellId = cell.model.id;
    const pendingTimer = this._editTimers.get(cellId);
    if (pendingTimer) {
      clearTimeout(pendingTimer);
      this._editTimers.delete(cellId);
      console.log(
        `ReproducibilityExecutionHook: Cancelled pending cell_edited for ${cellId} (cell is executing)`
      );
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

  /**
   * Extract reproducibility metadata from cell outputs.
   * Returns the metadata and the index of the output containing it (for removal).
   */
  private _extractReproducibilityMetadata(outputs: IOutput[]): {
    metadata: IReproducibilityMetadata | null;
    outputIndex: number | null;
  } {
    console.log(
      `ReproducibilityExecutionHook: Checking ${outputs.length} outputs for flowbook metadata`
    );

    for (let i = 0; i < outputs.length; i++) {
      const output = outputs[i];
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
      return {
        metadata: metadata.flowbook as IReproducibilityMetadata,
        outputIndex: i
      };
    }
    console.log(
      'ReproducibilityExecutionHook: No flowbook metadata found in any output'
    );
    return { metadata: null, outputIndex: null };
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

    // Check for predicate violations from kernel (new unified format)
    const predicateViolations = this._extractPredicateViolations(outputs);
    if (predicateViolations.length > 0) {
      // Store all predicate violations in cell metadata (array)
      cell.model.setMetadata('flowbook_violations', predicateViolations);
      // Also store first one in singular key for backward compatibility
      cell.model.setMetadata('flowbook_violation', predicateViolations[0]);

      // Let cellhighlighter handle the rendering
      const cellOrder = this._getCurrentCellOrder(panel);
      const stalenessManager = this._highlighter.getStalenessManager(panel);
      this._highlighter.updateCell(
        cell,
        stalenessManager,
        cellOrder,
        panel.context.path
      );

      console.log(
        `ReproducibilityExecutionHook: Handled ${predicateViolations.length} predicate violation(s) for cell ${cell.model.id}, accepted=${predicateViolations[0].accepted}`
      );

      // If rejected (not accepted), no further processing needed
      if (!predicateViolations[0].accepted) {
        return;
      }
    }

    // Extract reproducibility metadata
    const {
      metadata: reproducibilityMetadata,
      outputIndex: metadataOutputIndex
    } = this._extractReproducibilityMetadata(outputs);
    if (!reproducibilityMetadata) {
      // If we had predicate violations but no metadata, we're done
      if (predicateViolations.length > 0) {
        return;
      }
      return;
    }

    // Remove the metadata output from cell display (keep it clean for the user)
    if (metadataOutputIndex !== null) {
      const codeModel = cell.model as ICodeCellModel;
      codeModel.outputs.remove(metadataOutputIndex);
      console.log(
        `ReproducibilityExecutionHook: Removed flowbook metadata output at index ${metadataOutputIndex}`
      );
    }

    // Store metadata on cell
    cell.model.setMetadata('flowbook', reproducibilityMetadata);

    // Process staleness reasons and update manager
    this._processMetadataUpdate(panel, reproducibilityMetadata);

    // Store or clear violation metadata on the executing cell (legacy format)
    const cellOrder = this._getCurrentCellOrder(panel);
    const hasPredicateViolations = predicateViolations.length > 0;
    if (reproducibilityMetadata.violation && !hasPredicateViolations) {
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
        message: `Cell ${mutRef} modified ${v.variables.map(vv => '`' + vv + '`').join(', ')} read by ${affRef}`,
        structural_reads_detail: (v as any).structural_reads_detail,
        changes_detail: (v as any).changes_detail
      };
      cell.model.setMetadata('flowbook_violation', violationInfo);
    } else if (!hasPredicateViolations) {
      cell.model.deleteMetadata('flowbook_violation');
    }

    // If there's a writer_violation, store it on the writer cell (legacy format)
    if (reproducibilityMetadata.writer_violation) {
      const wv = reproducibilityMetadata.writer_violation;
      const writerCell = this._findCell(panel, wv.mutating_cell);
      if (writerCell) {
        // Build IViolationInfo - same structure as for normal violations
        const mutIdx = cellOrder.indexOf(wv.mutating_cell);
        const affIdx = cellOrder.indexOf(wv.affected_cell);
        const mutRef = mutIdx >= 0 ? indexToAlpha(mutIdx) : wv.mutating_cell;
        const affRef = affIdx >= 0 ? indexToAlpha(affIdx) : wv.affected_cell;

        const writerViolationInfo: IViolationInfo = {
          type: wv.violation_type || 'backward_mutation',
          mutating_cell: wv.mutating_cell,
          affected_cell: wv.affected_cell,
          variables: wv.variables,
          message: `Cell ${mutRef} modified ${wv.variables.map(v => '`' + v + '`').join(', ')} read by ${affRef}`,
          structural_reads_detail: (wv as any).structural_reads_detail,
          changes_detail: (wv as any).changes_detail
        };

        // Store in metadata
        writerCell.model.setMetadata('flowbook_violation', writerViolationInfo);

        // Let cellhighlighter handle the rendering
        const stalenessManager = this._highlighter.getStalenessManager(panel);
        this._highlighter.updateCell(
          writerCell,
          stalenessManager,
          cellOrder,
          panel.context.path
        );

        console.log(
          `ReproducibilityExecutionHook: Stored writer_violation on cell ${wv.mutating_cell}`
        );
      }
    }

    // Let cellhighlighter handle all cell rendering
    const stalenessManager = this._highlighter.getStalenessManager(panel);
    this._highlighter.updateCell(
      cell,
      stalenessManager,
      cellOrder,
      panel.context.path
    );

    console.log(
      `ReproducibilityExecutionHook: Extracted metadata for cell ${cell.model.id}:`,
      reproducibilityMetadata
    );
  }

  /**
   * Extract all predicate violations from kernel outputs (new unified format).
   * Returns array of violations (may be empty).
   */
  private _extractPredicateViolations(
    outputs: IOutput[]
  ): IPredicateViolation[] {
    const violations: IPredicateViolation[] = [];
    for (const output of outputs) {
      if (output.output_type !== 'display_data') {
        continue;
      }
      const metadata = (output as any).metadata;
      if (metadata?.predicate_violation) {
        const pv = metadata.predicate_violation;
        violations.push({
          predicate: pv.predicate,
          cell_id: pv.cell_id,
          locations: pv.locations || [],
          message: pv.message,
          accepted: pv.accepted,
          causer_cell: pv.causer_cell,
          detail: pv.detail
        });
      }
    }
    return violations;
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

      console.log(
        'ReproducibilityExecutionHook: staleness_reasons from backend:',
        metadata.staleness_reasons
      );
      console.log(
        'ReproducibilityExecutionHook: newlyStale cells:',
        newlyStale
      );

      // Compute reason for each newly-stale cell
      // Prefer backend-provided staleness_reasons, fall back to local computation
      for (const staleCellId of newlyStale) {
        // Skip empty cells - they are always clean
        const staleCell = this._findCell(panel, staleCellId);
        if (staleCell) {
          const codeModel = staleCell.model as ICodeCellModel;
          const source = codeModel.sharedModel.getSource();
          const isEmpty = !source || source.trim() === '';
          if (isEmpty) {
            console.log(
              `ReproducibilityExecutionHook: cell ${staleCellId} is empty, skipping staleness reason`
            );
            continue;
          }
        }

        const backendReasons = metadata.staleness_reasons?.[staleCellId];
        let reason: IFrontendStalenessReason;

        console.log(
          `ReproducibilityExecutionHook: cell ${staleCellId} backendReasons:`,
          backendReasons
        );

        if (backendReasons && backendReasons.length > 0) {
          // Use backend reason - convert to frontend format for cell metadata
          reason = this._backendReasonToFrontend(backendReasons[0], cellOrder);
          console.log(
            `ReproducibilityExecutionHook: cell ${staleCellId} using backend reason:`,
            reason
          );
        } else {
          // Fall back to local computation
          reason = this._computeStalenessReason(
            panel,
            staleCellId,
            metadata,
            cellOrder
          );
          console.log(
            `ReproducibilityExecutionHook: cell ${staleCellId} using local reason:`,
            reason
          );
        }

        stalenessManager.setReason(staleCellId, reason);

        // Store structured metadata on the cell (staleCell already found above)
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
  }

  /**
   * Convert a backend staleness reason to frontend format.
   * Maps backend reason types to frontend types with human-readable messages.
   */
  private _backendReasonToFrontend(
    backendReason: {
      type: string;
      loc?: string;
      cell_id?: string;
      expected_cell_id?: string;
    },
    cellOrder: string[]
  ): IFrontendStalenessReason {
    const cellId = backendReason.cell_id;
    const expectedCellId = backendReason.expected_cell_id;
    const loc = backendReason.loc;

    let causingRef = '';
    if (cellId) {
      const causingIdx = cellOrder.indexOf(cellId);
      causingRef = causingIdx >= 0 ? indexToAlpha(causingIdx) : cellId;
    }

    let expectedRef = '';
    let expectedCellDeleted = false;
    if (expectedCellId) {
      const expectedIdx = cellOrder.indexOf(expectedCellId);
      if (expectedIdx >= 0) {
        expectedRef = indexToAlpha(expectedIdx);
      } else {
        expectedCellDeleted = true;
      }
    }

    switch (backendReason.type) {
      case 'never_executed':
        return {
          type: 'unknown', // Use 'unknown' for non-variable-specific reasons
          message: 'Cell has never been executed'
        };
      case 'code_changed':
        return {
          type: 'source_edited',
          message: 'Source code was edited'
        };
      case 'forward_stale':
        // ForwardStale: show "x modified by @F"
        if (loc && causingRef) {
          return {
            type: 'variable_modified',
            causing_cell: cellId,
            variables: [loc],
            message: `\`${loc}\` was modified by ${causingRef}`
          };
        }
        return {
          type: 'variable_modified',
          causing_cell: cellId,
          message: causingRef
            ? `Input modified by ${causingRef}`
            : 'Input was modified'
        };
      case 'write_overlap':
        // Write overlap: cell writes to location that earlier cell also writes
        if (loc && causingRef) {
          return {
            type: 'writer_conflict',
            causing_cell: cellId,
            variables: [loc],
            message: `Write overlap: \`${loc}\` also written by ${causingRef}`
          };
        }
        return {
          type: 'writer_conflict',
          causing_cell: cellId,
          message: causingRef
            ? `Write overlap with ${causingRef}`
            : 'Write overlap detected'
        };
      case 'skipped_upstream':
        // Re-running won't help - need to run the expected cell first
        // If expected cell was deleted, say so clearly
        if (expectedCellDeleted) {
          return {
            type: 'variable_modified',
            causing_cell: cellId,
            variables: loc ? [loc] : undefined,
            message: loc
              ? `\`${loc}\` is from a deleted cell`
              : 'Source cell was deleted'
          };
        }
        if (loc && expectedRef) {
          return {
            type: 'variable_modified',
            causing_cell: cellId,
            variables: [loc],
            message: `Run ${expectedRef} first (\`${loc}\` is from wrong source)`
          };
        }
        return {
          type: 'variable_modified',
          causing_cell: cellId,
          message: expectedRef
            ? `Run ${expectedRef} first`
            : 'Upstream cell was skipped'
        };
      case 'backward_stale':
        if (loc && causingRef) {
          return {
            type: 'writer_conflict',
            causing_cell: cellId,
            variables: [loc],
            message: `Write conflict on \`${loc}\` with ${causingRef}`
          };
        }
        return {
          type: 'writer_conflict',
          causing_cell: cellId,
          message: 'Write conflict detected'
        };
      case 'no_read_before_write':
        // NoReadBeforeWrite failed - reads from later cell (forward contamination)
        if (loc && causingRef) {
          return {
            type: 'unknown',
            causing_cell: cellId,
            message: `Reads \`${loc}\` from later cell ${causingRef} (forward contamination)`
          };
        }
        return {
          type: 'unknown',
          causing_cell: cellId,
          message: 'Reads from a later cell'
        };
      case 'order_changed':
        return {
          type: 'unknown',
          message: 'Cell order changed'
        };
      case 'no_write_after_read':
        // NoWriteAfterRead failed - wrote to location read by earlier cell (backward mutation)
        if (loc && causingRef) {
          return {
            type: 'variable_modified',
            causing_cell: cellId,
            variables: [loc],
            message: `Wrote \`${loc}\` read by earlier cell ${causingRef} (backward mutation)`
          };
        }
        return {
          type: 'unknown',
          causing_cell: cellId,
          message: causingRef
            ? `Wrote to variable read by ${causingRef}`
            : 'Backward mutation detected'
        };
      default:
        return {
          type: 'unknown',
          causing_cell: cellId,
          message: causingRef
            ? `Dependencies changed by ${causingRef}`
            : 'Cell is stale'
        };
    }
  }

  /**
   * Compute why a cell became stale.
   */
  private _computeStalenessReason(
    panel: NotebookPanel,
    staleCellId: string,
    metadata: IReproducibilityMetadata,
    cellOrder: string[]
  ): IFrontendStalenessReason {
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
}
