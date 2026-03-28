/**
 * Execution hook for FlowBook kernel — comm-based protocol communication.
 *
 * Uses a Jupyter comm channel ("flowbook" target) for bidirectional
 * kernel <-> frontend communication, replacing the old display_data
 * metadata and magic command approach.
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
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import {
  IReproducibilityMetadata,
  IFrontendStalenessReason,
  IPredicateViolation,
  IReadLoc,
  IWriteLoc,
  findConflictingReads,
  formatReadLoc,
  writeLocOutputs,
  readLocsMatchQualifier
} from './types';
import {
  COMM_TARGET,
  FlowbookKernelMessage,
  FlowbookClientMessage
} from './protocol';
import { indexToAlpha } from '../cellindexutils';

export class ReproducibilityExecutionHookManager {
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter;
  private _editTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private _executedCells: Set<string> = new Set();
  private _attachedKernel: Kernel.IKernelConnection | null = null;
  private _listenedCellIds: Set<string> = new Set();
  private _comm: Kernel.IComm | null = null;

  // Pending violations received via comm before _onCellExecuted fires.
  // _onCellExecuted picks these up and stores them on the cell.
  private _pendingViolations: IPredicateViolation[] = [];

  constructor(
    _app: JupyterFrontEnd,
    tracker: INotebookTracker,
    highlighter: ReproducibilityCellHighlighter
  ) {
    this._tracker = tracker;
    this._highlighter = highlighter;
    this._setupHooks();
  }

  /**
   * Send a FlowBook protocol command to the kernel via the comm channel.
   * Used by plugin.ts for sync, exec-restore, etc.
   */
  sendCommand(msg: FlowbookClientMessage): void {
    if (this._comm) {
      this._comm.send(msg as any);
    } else {
      console.warn(
        'ReproducibilityExecutionHook: No comm channel, cannot send command:',
        msg
      );
    }
  }

  private _setupHooks(): void {
    // Listen for cell execution completion
    NotebookActions.executed.connect(this._onCellExecuted, this);

    // Listen for cell execution start to send cell order via comm
    NotebookActions.executionScheduled.connect(
      this._onExecutionScheduled,
      this
    );

    // [EDIT transition (§2.3)] Listen for cell content changes
    this._tracker.currentChanged.connect(this._setupCellEditListener, this);

    // Set up comm channel for kernel communication
    this._tracker.currentChanged.connect(this._setupComm, this);

    // Also set up listeners for already-open notebook (signal may have fired before we subscribed)
    console.log(
      'ReproducibilityExecutionHookManager: currentWidget =',
      this._tracker.currentWidget?.context?.path
    );
    if (this._tracker.currentWidget) {
      this._setupCellEditListener();
      this._setupComm();
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

    // Watch for cell changes (insert/delete) to update kernel and attach listeners
    notebook.model?.cells.changed.connect((_sender, change) => {
      // Attach edit listeners to any new cells
      for (let i = 0; i < notebook.widgets.length; i++) {
        this._attachCellEditListener(notebook.widgets[i]);
      }

      // If cells were added or removed, notify the kernel about the new cell order
      // This ensures staleness is updated immediately (e.g., when a cell is deleted,
      // cells that read from it should be marked stale)
      if (change.type === 'add' || change.type === 'remove') {
        this._sendNotebookStructure(panel);
      }
    });
  }

  /**
   * Send notebook_structure command to kernel via comm.
   * Called when cells are added/removed to update staleness immediately.
   */
  private _sendNotebookStructure(panel: NotebookPanel): void {
    const notebook = panel.content;

    // Build cell order array (only code cells)
    const cellOrder: string[] = [];
    for (let i = 0; i < notebook.widgets.length; i++) {
      const c = notebook.widgets[i];
      if (c.model.type === 'code') {
        cellOrder.push(c.model.id);
      }
    }

    if (cellOrder.length > 0) {
      this.sendCommand({ type: 'notebook_structure', cell_order: cellOrder });
      console.log(
        `ReproducibilityExecutionHook: Sent notebook_structure via comm with ${cellOrder.length} cells`
      );
    }
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
   * [EDIT transition (§2.3)] Send cell_edited command to kernel via comm.
   */
  private _sendCellEdited(cellId: string): void {
    this.sendCommand({ type: 'cell_edited', cell_id: cellId });
    console.log(
      `ReproducibilityExecutionHook: Sent cell_edited via comm for ${cellId}`
    );
  }

  /**
   * Set up a comm channel to the kernel's "flowbook" target.
   * This replaces the old IOPub listener for metadata and the
   * silent magic executions for sending commands.
   */
  private _setupComm(): void {
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    this._connectComm(panel);

    // Re-open comm when the kernel object changes (e.g., switching kernels)
    panel.sessionContext.kernelChanged.connect(() => {
      this._attachedKernel = null; // force reconnect
      this._connectComm(panel);
    });

    // Re-open comm after kernel restart. The kernel object stays the same
    // on restart, so kernelChanged doesn't fire — we must watch statusChanged.
    panel.sessionContext.statusChanged.connect((_sender, status) => {
      if (status === 'restarting') {
        // Clear the guard so _connectComm will re-open on next idle
        this._attachedKernel = null;
        this._comm = null;
      } else if (status === 'idle' && this._comm === null) {
        this._connectComm(panel);
      }
    });
  }

  private _connectComm(panel: NotebookPanel): void {
    const kernel = panel.sessionContext.session?.kernel;
    if (!kernel || kernel === this._attachedKernel) {
      return;
    }

    this._attachedKernel = kernel;

    // Open a comm to the kernel's "flowbook" target
    this._comm = kernel.createComm(COMM_TARGET);
    this._comm.onMsg = this._onCommMessage.bind(this);
    this._comm.open();

    console.log(
      'ReproducibilityExecutionHook: Comm channel opened to kernel'
    );
  }

  /**
   * Handle incoming comm messages from the kernel.
   * Dispatches on message type: metadata, violation, or status.
   */
  private _onCommMessage(msg: KernelMessage.ICommMsgMsg): void {
    const data = msg.content.data as unknown as FlowbookKernelMessage;
    if (!data || !data.type) {
      return;
    }

    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    switch (data.type) {
      case 'metadata': {
        // Strip the "type" field to get IReproducibilityMetadata
        const { type: _type, ...metadata } = data;
        const reproMeta = metadata as unknown as IReproducibilityMetadata;

        // Store metadata on the relevant cell
        if (reproMeta.cell_id) {
          const cell = this._findCell(panel, reproMeta.cell_id);
          if (cell) {
            cell.model.setMetadata('flowbook', reproMeta);
          }
        }

        // Process staleness
        this._processMetadataUpdate(panel, reproMeta);

        console.log(
          'ReproducibilityExecutionHook: Comm metadata received, stale_cells =',
          reproMeta.stale_cells
        );
        break;
      }

      case 'violation': {
        const { type: _type, ...violation } = data;
        const pv = violation as unknown as IPredicateViolation;
        // Buffer violation — _onCellExecuted will pick it up and store on cell
        this._pendingViolations.push(pv);
        console.log(
          `ReproducibilityExecutionHook: Comm violation received for cell ${pv.cell_id}, predicate=${pv.predicate}`
        );
        break;
      }

      case 'status': {
        // Update the metadata panel status header
        this._highlighter.updateStatus(data.icon, data.text, data.cell_id);
        console.log(
          `ReproducibilityExecutionHook: Comm status: ${data.icon} ${data.text}`
        );
        break;
      }
    }
  }

  /**
   * Called before cell execution — send notebook_structure via comm to set cell order.
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
    // so there's no need to send cell_edited (which would incorrectly mark it stale).
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

    // Send notebook_structure via comm
    if (cellOrder.length > 0) {
      this.sendCommand({ type: 'notebook_structure', cell_order: cellOrder });
      console.log(
        `ReproducibilityExecutionHook: Sent notebook_structure via comm with ${cellOrder.length} cells`
      );
    }
  }

  // _extractReproducibilityMetadata removed — metadata now arrives via comm

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

    // Collect pending violations that arrived via comm during execution.
    // Filter to violations for this cell (comm may deliver violations for
    // other cells in rare race conditions).
    const cellId = cell.model.id;
    const cellViolations = this._pendingViolations.filter(
      v => v.cell_id === cellId
    );
    // Keep violations for other cells (if any) for their _onCellExecuted call
    this._pendingViolations = this._pendingViolations.filter(
      v => v.cell_id !== cellId
    );

    if (cellViolations.length > 0) {
      cell.model.setMetadata('flowbook_violations', cellViolations);
      cell.model.setMetadata('flowbook_violation', cellViolations[0]);
      console.log(
        `ReproducibilityExecutionHook: Applied ${cellViolations.length} violation(s) from comm to cell ${cellId}`
      );
    } else {
      // Clear any previous violation metadata
      cell.model.deleteMetadata('flowbook_violations');
      cell.model.deleteMetadata('flowbook_violation');
    }

    // Metadata is now stored on cell by _onCommMessage when it arrives.
    // No need to scan cell outputs.

    // Let cellhighlighter handle all cell rendering (staleness + violations).
    const cellOrder = this._getCurrentCellOrder(panel);
    const stalenessManager = this._highlighter.getStalenessManager(panel);
    this._highlighter.updateCell(
      cell,
      stalenessManager,
      cellOrder,
      panel.context.path
    );

    // Refresh dependency graph
    this._highlighter.refreshDependencies();
  }

  // _extractPredicateViolations removed — violations now arrive via comm

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
    },
    cellOrder: string[]
  ): IFrontendStalenessReason {
    const cellId = backendReason.cell_id;
    const loc = backendReason.loc;

    let causingRef = '';
    if (cellId) {
      const causingIdx = cellOrder.indexOf(cellId);
      causingRef = causingIdx >= 0 ? indexToAlpha(causingIdx) : cellId;
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
   * Compute why a cell became stale using the ▷ conflict relation.
   *
   * Uses typed ReadLoc/WriteLoc sets and writeConflictsRead() to determine
   * which specific locations were invalidated.
   */
  private _computeStalenessReason(
    panel: NotebookPanel,
    staleCellId: string,
    metadata: IReproducibilityMetadata,
    cellOrder: string[]
  ): IFrontendStalenessReason {
    const causingCellId = metadata.cell_id;

    // Edit case: the cell that triggered the update is the stale cell itself
    // and there are no changed_locs (pure source edit)
    if (
      staleCellId === causingCellId &&
      (!metadata.changed_locs || metadata.changed_locs.length === 0)
    ) {
      return {
        type: 'source_edited',
        causing_cell: causingCellId,
        message: 'Source code was edited'
      };
    }

    // Look up the stale cell's stored read_locs
    const staleCell = this._findCell(panel, staleCellId);
    const storedMeta = staleCell?.model.metadata as any;
    const storedFlowbook = storedMeta?.flowbook as
      | IReproducibilityMetadata
      | undefined;

    const causingIdx = cellOrder.indexOf(causingCellId);
    const causingRef =
      causingIdx >= 0 ? indexToAlpha(causingIdx) : causingCellId;

    // StaleFwd case: use ▷ to find which of the stale cell's reads are
    // invalidated by the causing cell's changed_locs
    const changedLocs: IWriteLoc[] = metadata.changed_locs || [];
    const cellReadLocs: IReadLoc[] = storedFlowbook?.read_locs || [];

    const conflicting = findConflictingReads(changedLocs, cellReadLocs);

    if (conflicting.length > 0) {
      const parts = conflicting.map(r => '`' + formatReadLoc(r) + '`');
      return {
        type: 'variable_modified',
        causing_cell: causingCellId,
        variables: conflicting.map(r => formatReadLoc(r)),
        message: `${parts.join(', ')} modified by ${causingRef}`
      };
    }

    // WriterCheck case: stale cell's write outputs conflict with causing cell's reads
    // (EXEC-RESTORE marks cells that would cause BackConflict if run)
    const staleCellWriteLocs: IWriteLoc[] = storedFlowbook?.write_locs || [];
    const causingCellReadLocs: IReadLoc[] = metadata.read_locs || [];

    // Project stale cell's writes to reads via output(), then intersect
    const staleOutputs: IReadLoc[] = staleCellWriteLocs.flatMap(w =>
      writeLocOutputs(w)
    );
    // Check which of the stale cell's output reads match the causing cell's reads
    const writerConflicts: string[] = [];
    const seen = new Set<string>();
    for (const output of staleOutputs) {
      const key = `${output.type}:${output.qualifier || ''}:${output.name}`;
      if (seen.has(key)) {
        continue;
      }
      for (const r of causingCellReadLocs) {
        if (
          output.type === r.type &&
          output.name === r.name &&
          readLocsMatchQualifier(output, r)
        ) {
          writerConflicts.push(formatReadLoc(output));
          seen.add(key);
          break;
        }
      }
    }

    if (writerConflicts.length > 0) {
      const varParts = writerConflicts.map(v => '`' + v + '`');
      return {
        type: 'writer_conflict',
        causing_cell: causingCellId,
        variables: writerConflicts,
        message: `Writes ${varParts.join(', ')}, which was read by ${causingRef}`
      };
    }

    // Fallback
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
