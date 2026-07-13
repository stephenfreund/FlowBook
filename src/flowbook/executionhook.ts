/**
 * Execution hook for FlowBook kernel — comm-based protocol communication.
 *
 * Uses a Jupyter comm channel ("flowbook" target) for bidirectional
 * kernel <-> frontend communication, replacing the old display_data
 * metadata and magic command approach.
 */

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
import { FixSuggester } from './fixsuggester';
import {
  IReproducibilityMetadata,
  IFrontendStalenessReason,
  IPredicateViolation,
  IReadLoc,
  IWriteLoc,
  findConflictingReads,
  formatReadLoc,
  writeConflictsRead
} from './types';
import {
  COMM_TARGET,
  FlowbookKernelMessage,
  FlowbookClientMessage
} from './protocol';
import { indexToAlpha, getCodeCellOrder } from '../cellindexutils';
import { emitAiActivity } from './aiattribution';

export class ReproducibilityExecutionHookManager {
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter;
  private _fixSuggester: FixSuggester | null = null;
  private _editTimers: Map<
    string,
    { timer: ReturnType<typeof setTimeout>; model: ICodeCellModel }
  > = new Map();
  private _executedCells: Set<string> = new Set();
  private _attachedKernel: Kernel.IKernelConnection | null = null;
  private _comm: Kernel.IComm | null = null;
  private _isDisposed = false;

  // Per-cell sharedModel.changed handlers, kept so dispose() can disconnect
  // them (and so attachment stays idempotent per cell id).
  private _cellEditHandlers: Map<
    string,
    {
      sharedModel: ICodeCellModel['sharedModel'];
      handler: (sender: unknown, change: CellChange) => void;
    }
  > = new Map();

  // Per-panel sessionContext listeners (kernelChanged/statusChanged) and
  // cells.changed listeners. Stored so they are attached once per panel and
  // disconnected on dispose — previously a fresh anonymous closure was
  // connected on every currentChanged and never removed.
  private _sessionHandlers: Map<
    NotebookPanel,
    {
      onKernelChanged: () => void;
      onStatusChanged: (sender: unknown, status: string) => void;
    }
  > = new Map();
  private _cellsChangedHandlers: Map<
    NotebookPanel,
    (sender: unknown, change: { type: string }) => void
  > = new Map();

  // Pending violations received via comm before _onCellExecuted fires.
  // _onCellExecuted picks these up and stores them on the cell.
  private _pendingViolations: IPredicateViolation[] = [];

  constructor(
    tracker: INotebookTracker,
    highlighter: ReproducibilityCellHighlighter
  ) {
    this._tracker = tracker;
    this._highlighter = highlighter;
    this._setupHooks();
  }

  /**
   * Wire in the AI fix suggester. Called by the activation manager after
   * both the execution hook and the suggester are constructed.
   */
  setFixSuggester(suggester: FixSuggester | null): void {
    this._fixSuggester = suggester;
  }

  /**
   * Disconnect all signal listeners and clean up.
   */
  dispose(): void {
    if (this._isDisposed) {
      return;
    }
    this._isDisposed = true;

    NotebookActions.executed.disconnect(this._onCellExecuted, this);
    NotebookActions.executionScheduled.disconnect(
      this._onExecutionScheduled,
      this
    );
    this._tracker.currentChanged.disconnect(this._setupCellEditListener, this);
    this._tracker.currentChanged.disconnect(this._setupComm, this);

    // Disconnect per-panel session and cells.changed listeners
    for (const [panel, handlers] of this._sessionHandlers) {
      if (!panel.isDisposed) {
        panel.sessionContext.kernelChanged.disconnect(handlers.onKernelChanged);
        panel.sessionContext.statusChanged.disconnect(handlers.onStatusChanged);
      }
    }
    this._sessionHandlers.clear();
    for (const [panel, handler] of this._cellsChangedHandlers) {
      if (!panel.isDisposed) {
        panel.content.model?.cells.changed.disconnect(handler);
      }
    }
    this._cellsChangedHandlers.clear();

    // Disconnect per-cell edit listeners
    for (const { sharedModel, handler } of this._cellEditHandlers.values()) {
      try {
        sharedModel.changed.disconnect(handler);
      } catch {
        // Model may already be disposed
      }
    }
    this._cellEditHandlers.clear();

    // Flush (not drop) pending edit notifications: an edit made just before
    // deactivation must still reach the kernel, or it will keep treating the
    // old source as CLEAN.
    for (const [cellId, pending] of this._editTimers) {
      clearTimeout(pending.timer);
      if (this._comm) {
        this._sendCellEdited(cellId, pending.model);
      }
    }
    this._editTimers.clear();

    // Close comm channel
    this._closeComm();
    this._attachedKernel = null;
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
    if (this._tracker.currentWidget) {
      this._setupCellEditListener();
      this._setupComm();
    }
  }

  /**
   * [EDIT transition (§2.3)] Set up listeners for cell content changes.
   * When a code cell's source changes and the cell was previously executed,
   * send %cell_edited <cell_id> to the kernel with debouncing.
   *
   * Also watches for newly inserted cells so they get listeners too.
   */
  private _setupCellEditListener(): void {
    if (this._isDisposed) {
      return;
    }
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    const notebook = panel.content;

    // Attach listeners to all existing code cells
    for (let i = 0; i < notebook.widgets.length; i++) {
      this._attachCellEditListener(notebook.widgets[i]);
    }

    // Watch for cell changes (insert/delete) to update kernel and attach
    // listeners. Attached once per panel (revisiting a notebook must not
    // stack duplicate listeners) and disconnected on dispose.
    if (this._cellsChangedHandlers.has(panel)) {
      return;
    }
    const onCellsChanged = (_sender: unknown, change: { type: string }) => {
      if (this._isDisposed || panel.isDisposed) {
        return;
      }
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
    };
    notebook.model?.cells.changed.connect(onCellsChanged);
    this._cellsChangedHandlers.set(panel, onCellsChanged);
    panel.disposed.connect(() => {
      this._cellsChangedHandlers.delete(panel);
      this._sessionHandlers.delete(panel);
    });
  }

  /**
   * Send notebook_structure command to kernel via comm.
   * Called when cells are added/removed to update staleness immediately.
   */
  private _sendNotebookStructure(panel: NotebookPanel): void {
    const cellOrder = getCodeCellOrder(panel);

    if (cellOrder.length > 0) {
      this.sendCommand({ type: 'notebook_structure', cell_order: cellOrder });
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
    if (this._cellEditHandlers.has(cellId)) {
      return;
    }

    const model = cell.model as ICodeCellModel;
    const handler = (_sender: unknown, change: CellChange) => {
      if (this._isDisposed) {
        return;
      }
      // Only react to source text edits, not output/metadata/executionCount changes
      if (change.sourceChange) {
        this._onCellContentChanged(cellId, model);
      }
    };
    model.sharedModel.changed.connect(handler);
    this._cellEditHandlers.set(cellId, {
      sharedModel: model.sharedModel,
      handler
    });
  }

  /**
   * [EDIT transition (§2.3)] Handle cell content change with debouncing.
   */
  private _onCellContentChanged(cellId: string, model: ICodeCellModel): void {
    // Cancel any in-flight AI fix suggestion — the violation it was
    // diagnosing is about to be invalidated by this edit.
    if (this._fixSuggester) {
      this._fixSuggester.cancel(cellId);
    }

    // Only notify kernel about cells that have been previously executed
    if (!this._executedCells.has(cellId)) {
      return;
    }

    // Debounce: cancel previous timer for this cell
    const existing = this._editTimers.get(cellId);
    if (existing) {
      clearTimeout(existing.timer);
    }

    // Set new timer (1s debounce)
    const timer = setTimeout(() => {
      this._sendCellEdited(cellId, model);
      this._editTimers.delete(cellId);
    }, 1000);

    this._editTimers.set(cellId, { timer, model });
  }

  /**
   * [EDIT transition (§2.3)] Send cell_edited command to kernel via comm.
   * Includes the cell's current source so the kernel can tell a meaningful
   * edit (AST changed) from a cosmetic one (whitespace/comments).
   */
  private _sendCellEdited(cellId: string, model: ICodeCellModel): void {
    const source = model.sharedModel.getSource();
    this.sendCommand({ type: 'cell_edited', cell_id: cellId, source });
  }

  /**
   * Set up a comm channel to the kernel's "flowbook" target.
   * This replaces the old IOPub listener for metadata and the
   * silent magic executions for sending commands.
   */
  private _setupComm(): void {
    if (this._isDisposed) {
      return;
    }
    const panel = this._tracker.currentWidget;
    if (!panel) {
      return;
    }

    this._connectComm(panel);

    // Session listeners are attached once per panel (revisiting a notebook
    // must not stack duplicates) and disconnected on dispose.
    if (this._sessionHandlers.has(panel)) {
      return;
    }

    // Re-open comm when the kernel object changes (e.g., switching kernels).
    // Only react for the panel the user is looking at — a background panel's
    // kernel change must not steal the comm.
    const onKernelChanged = () => {
      if (this._isDisposed || panel.isDisposed) {
        return;
      }
      if (this._tracker.currentWidget !== panel) {
        return;
      }
      this._attachedKernel = null; // force reconnect
      this._connectComm(panel);
    };
    panel.sessionContext.kernelChanged.connect(onKernelChanged);

    // Re-open comm after kernel restart. The kernel object stays the same
    // on restart, so kernelChanged doesn't fire — we must watch statusChanged.
    const onStatusChanged = (_sender: unknown, status: string) => {
      if (this._isDisposed || panel.isDisposed) {
        return;
      }
      if (status === 'restarting') {
        // The comm died with the kernel process; drop it so _connectComm
        // will re-open on next idle.
        this._attachedKernel = null;
        this._closeComm();
      } else if (
        status === 'idle' &&
        this._comm === null &&
        this._tracker.currentWidget === panel
      ) {
        this._connectComm(panel);
      }
    };
    panel.sessionContext.statusChanged.connect(onStatusChanged);

    this._sessionHandlers.set(panel, { onKernelChanged, onStatusChanged });
    panel.disposed.connect(() => {
      this._sessionHandlers.delete(panel);
      this._cellsChangedHandlers.delete(panel);
    });
  }

  private _connectComm(panel: NotebookPanel): void {
    if (this._isDisposed || panel.isDisposed) {
      return;
    }
    const kernel = panel.sessionContext.session?.kernel;
    if (!kernel || kernel === this._attachedKernel) {
      return;
    }

    // Close any comm to a previous kernel before opening a new one.
    // An abandoned comm's onMsg handler stays live and would keep applying
    // that kernel's updates.
    this._closeComm();

    this._attachedKernel = kernel;

    // Open a comm to the kernel's "flowbook" target. Bind the handler to
    // THIS panel — messages must be applied to the notebook that owns the
    // sending kernel, not whichever notebook is focused when they arrive.
    const comm = kernel.createComm(COMM_TARGET);
    comm.onMsg = msg => this._onCommMessage(panel, msg);
    comm.onClose = () => {
      // Kernel-side teardown: clear our reference so sendCommand doesn't
      // silently write into a dead channel and reconnect can happen.
      if (this._comm === comm) {
        this._comm = null;
        this._attachedKernel = null;
      }
    };
    this._comm = comm;
    comm.open();
  }

  /**
   * Close the current comm (if any), tolerating dead kernels.
   */
  private _closeComm(): void {
    const comm = this._comm;
    if (!comm) {
      return;
    }
    this._comm = null;
    try {
      if (!comm.isDisposed) {
        comm.close();
      }
    } catch {
      // Kernel may already be gone
    }
  }

  /**
   * Handle incoming comm messages from the kernel.
   * Dispatches on message type: metadata, violation, or status.
   *
   * `panel` is the notebook that owned the kernel when the comm was opened —
   * NOT the currently focused notebook. Applying updates to the focused
   * notebook would write one notebook's staleness into another whenever the
   * user switches tabs while a cell finishes.
   */
  private _onCommMessage(
    panel: NotebookPanel,
    msg: KernelMessage.ICommMsgMsg
  ): void {
    if (this._isDisposed || panel.isDisposed) {
      return;
    }
    const data = msg.content.data as unknown as FlowbookKernelMessage;
    if (!data || !data.type) {
      return;
    }

    switch (data.type) {
      case 'metadata': {
        // Strip the "type" field to get IReproducibilityMetadata
        const { type: _type, ...metadata } = data;
        const reproMeta = metadata as unknown as IReproducibilityMetadata;

        // Store metadata on the relevant cell
        if (reproMeta.cell_id) {
          // Mark the cell as executed even when the run was driven
          // externally (e.g. MCP on the shared kernel): NotebookActions
          // signals don't fire for those, and without this, later user
          // edits to the cell would never be reported via cell_edited.
          this._executedCells.add(reproMeta.cell_id);

          // This metadata is canonical; drop any buffered violations for
          // the cell (for external runs _onCellExecuted never fires to
          // clear them, and the buffer would grow without bound).
          this._pendingViolations = this._pendingViolations.filter(
            v => v.cell_id !== reproMeta.cell_id
          );

          const cell = this._findCell(panel, reproMeta.cell_id);
          if (cell) {
            cell.model.setMetadata('flowbook', reproMeta);

            // Refresh cell UI — needed for external executions (e.g. MCP)
            // where _onCellExecuted doesn't fire on this client.
            const cellOrder = this._getCurrentCellOrder(panel);
            const stalenessManager =
              this._highlighter.getStalenessManager(panel);
            this._highlighter.updateCell(
              cell,
              stalenessManager,
              cellOrder,
              panel.context.path
            );
            this._highlighter.refreshDependencies();

            // If an out-of-process agent (MCP on the shared kernel) drove this
            // execution, announce it for an optional observer (e.g. LogBook).
            // Frontend execution signals (NotebookActions) don't fire for ZMQ
            // runs, so this DOM-event is the only signal an observer gets. It
            // carries enough to record the run; dependency-free, no-op when
            // unobserved.
            if ((data as { actor?: string }).actor === 'ai') {
              const codeModel =
                cell.model.type === 'code'
                  ? (cell.model as ICodeCellModel)
                  : null;
              const hasError = !!(
                reproMeta.errors && reproMeta.errors.length > 0
              );
              emitAiActivity({
                path: panel.context.path,
                cellId: reproMeta.cell_id,
                kind: 'execute',
                status: hasError ? 'error' : 'ok',
                executionCount: codeModel ? codeModel.executionCount : null,
                outputCount: codeModel ? codeModel.outputs.length : undefined
              });
            }
          }
        }

        // Process staleness
        this._processMetadataUpdate(panel, reproMeta);
        break;
      }

      case 'violation': {
        const { type: _type, ...violation } = data;
        const pv = violation as unknown as IPredicateViolation;
        // Buffer violation — the metadata message (which follows) carries
        // the canonical errors in flowbook.errors and triggers updateCell.
        this._pendingViolations.push(pv);
        break;
      }

      case 'status': {
        // Update the metadata panel status header
        this._highlighter.updateStatus(data.icon, data.text, data.cell_id);
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

    // Flush (not drop) any pending edit notification for this cell.
    // The comm message travels on the shell channel ahead of the
    // execute_request, so a completed run still ends CLEAN — but if the
    // execution is aborted (kernel interrupted/died, or the run-all queue
    // stopped on an earlier error), the kernel must still know the source
    // changed. Dropping the notification here left the kernel treating the
    // old source as CLEAN in those cases.
    const cellId = cell.model.id;
    const pending = this._editTimers.get(cellId);
    if (pending) {
      clearTimeout(pending.timer);
      this._editTimers.delete(cellId);
      this._sendCellEdited(cellId, pending.model);
    }

    const cellOrder = getCodeCellOrder(panel);

    // Send notebook_structure via comm
    if (cellOrder.length > 0) {
      this.sendCommand({ type: 'notebook_structure', cell_order: cellOrder });
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

    // Clear pending violations for this cell (they were buffered from the
    // violation comm message; the canonical data is in flowbook.errors
    // from the metadata comm message).
    const cellId = cell.model.id;
    this._pendingViolations = this._pendingViolations.filter(
      v => v.cell_id !== cellId
    );

    // Metadata (including errors) is stored on cell by _onCommMessage
    // when the metadata message arrives. No need to write violation
    // metadata separately.

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

    // AI fix suggestion: kick off a streaming diagnosis if this cell now has
    // violations, or clear any stale suggestion if the cell is clean.
    if (this._fixSuggester) {
      const meta = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      const hasErrors = !!meta?.errors && meta.errors.length > 0;
      if (hasErrors) {
        this._fixSuggester.request(panel, cell);
      } else {
        this._fixSuggester.clear(cell);
      }
    }
  }

  // _extractPredicateViolations removed — violations now arrive via comm

  /**
   * Get current cell order from notebook (only code cells).
   */
  private _getCurrentCellOrder(panel: NotebookPanel): string[] {
    return getCodeCellOrder(panel);
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
      // Prefer backend-provided staleness_reasons, fall back to local computation
      for (const staleCellId of newlyStale) {
        // Skip empty cells - they are always clean
        const staleCell = this._findCell(panel, staleCellId);
        if (staleCell) {
          const codeModel = staleCell.model as ICodeCellModel;
          const source = codeModel.sharedModel.getSource();
          const isEmpty = !source || source.trim() === '';
          if (isEmpty) {
            continue;
          }
        }

        const backendReasons = metadata.staleness_reasons?.[staleCellId];
        let reason: IFrontendStalenessReason;

        if (backendReasons && backendReasons.length > 0) {
          // Use backend reason - convert to frontend format for cell metadata
          reason = this._backendReasonToFrontend(backendReasons[0], cellOrder);
        } else {
          // Fall back to local computation
          reason = this._computeStalenessReason(
            panel,
            staleCellId,
            metadata,
            cellOrder
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

    // Find which causing cell reads are invalidated by the stale cell's writes
    const writerConflicts: string[] = [];
    const seen = new Set<string>();
    for (const r of causingCellReadLocs) {
      const key = `${r.type}:${r.qualifier || ''}:${r.name}`;
      if (seen.has(key)) {
        continue;
      }
      for (const w of staleCellWriteLocs) {
        if (writeConflictsRead(w, r)) {
          writerConflicts.push(formatReadLoc(r));
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
