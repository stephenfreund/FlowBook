/**
 * Frontend FixSuggester: requests AI fix suggestions for violations,
 * streams diagnosis text into the violation notice DOM, renders fix buttons,
 * applies fixes via the backend dispatcher, and supports one-click undo.
 *
 * State machine (per cell_id):
 *
 *   IDLE → REQUESTING → STREAMING → READY → APPLYING → APPLIED ┐
 *                                       ↑                       │
 *                                       └─── (undo) ────────────┘
 *
 * Any state can transition back to IDLE on cancel() (cell edited, re-run,
 * notebook closed, violation cleared).
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell, ICodeCellModel } from '@jupyterlab/cells';
import { IOutput } from '@jupyterlab/nbformat';
import { ServerConnection } from '@jupyterlab/services';
import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { IReproducibilityMetadata } from './types';
import { getCodeCellOrder } from '../cellindexutils';

// Metadata flag identifying our dedicated AI-fix display_data output.
const AI_FIX_NOTICE_FLAG = 'flowbook_ai_fix_notice';

// Inner HTML for the AI-fix notice. Three regions, all targeted by the
// suggester via data-flowbook-region queries.
const AI_FIX_NOTICE_HTML = `<div class="fb-ai-fix">
  <div class="fb-diagnosis" data-flowbook-region="diagnosis" hidden></div>
  <div class="fb-fix-buttons" data-flowbook-region="buttons" hidden></div>
  <div class="fb-undo" data-flowbook-region="undo" hidden></div>
</div>`;

interface IFixSuggestion {
  label: string;
  rationale: string;
  tool:
    | 'alpha_rename'
    | 'remove_inplace'
    | 'insert_deepcopy'
    | 'mark_diagnostic'
    | 'merge_cells'
    | 'move_cell';
  args: Record<string, unknown>;
}

interface IFixPlan {
  fixes: IFixSuggestion[];
}

interface IApplyFixResult {
  ok: boolean;
  tool: string;
  args: Record<string, unknown>;
  modified_cells: string[];
  pre_fix_sources: Record<string, string>;
  post_fix_sources: Record<string, string>;
  cells_removed?: string[];
  new_cell_order?: string[];
  error?: string;
}

type CellState =
  | 'idle'
  | 'requesting'
  | 'streaming'
  | 'ready'
  | 'applying'
  | 'applied'
  | 'failed';

/**
 * Full cell snapshot taken before an apply. We keep the entire cell JSON
 * (source + metadata + id) so undoFix() can recreate cells that were
 * removed by merge_cells, and move cells back to their original positions
 * after a move_cell. For source-only fixes (alpha_rename, remove_inplace,
 * etc.) the source field is what undo actually uses; the rest is harmless.
 */
interface IPreFixSnapshot {
  // Code-cell IDs in their original order.
  codeCellOrder: string[];
  // cell_id -> full cell JSON (cell_type, id, source, metadata, ...).
  cells: Map<string, any>;
}

interface ICellEntry {
  state: CellState;
  abort: AbortController;
  diagnosis: string;
  plan: IFixPlan | null;
  // Snapshot taken just before applying a fix. Used by undoFix().
  preFixSnapshot: IPreFixSnapshot | null;
  undoTimer: ReturnType<typeof setTimeout> | null;
}

const UNDO_TTL_MS = 30_000;

export class FixSuggester {
  private _tracker: INotebookTracker;
  private _highlighter: ReproducibilityCellHighlighter | null = null;
  private _state: Map<string, ICellEntry> = new Map();
  private _disposed = false;
  // Cells whose source we are currently mutating programmatically as part
  // of an apply. cancel() ignores ids in this set so our own setSource
  // calls don't tear down the entry mid-apply (sharedModel.changed fires
  // synchronously inside setSource and reaches the edit listener, which
  // would otherwise cancel us).
  private _programmaticEdits: Set<string> = new Set();

  constructor(tracker: INotebookTracker) {
    this._tracker = tracker;
  }

  /**
   * Wire in the highlighter so we can clear violations and trigger
   * staleness/render updates after a fix lands.
   */
  setHighlighter(highlighter: ReproducibilityCellHighlighter | null): void {
    this._highlighter = highlighter;
  }

  /**
   * True while a fix has been applied to this cell and the user hasn't yet
   * run it, edited it manually, or clicked undo. The cellhighlighter uses
   * this to suppress staleness rendering for the affected cell — the
   * "Source updated" + undo notice is the only thing we want visible.
   */
  isFixApplied(cellId: string): boolean {
    const entry = this._state.get(cellId);
    return entry !== undefined && entry.state === 'applied';
  }

  dispose(): void {
    this._disposed = true;
    for (const cellId of [...this._state.keys()]) {
      this.cancel(cellId);
    }
    this._state.clear();
  }

  /**
   * Kick off (or refresh) a suggestion request for the given cell.
   * If a request is already in flight for this cell, it is cancelled first.
   * Safe to call repeatedly — internal dedup avoids redundant work.
   */
  request(panel: NotebookPanel, cell: Cell): void {
    if (this._disposed) {
      return;
    }
    const cellId = cell.model.id;
    this.cancel(cellId);

    const entry: ICellEntry = {
      state: 'requesting',
      abort: new AbortController(),
      diagnosis: '',
      plan: null,
      preFixSnapshot: null,
      undoTimer: null
    };
    this._state.set(cellId, entry);

    // Make sure the dedicated AI-fix output exists on the cell, then
    // render the initial placeholder so the user sees activity.
    this._ensureAIFixOutput(cell);
    this._renderDiagnosisRegion(cell, 'Analyzing…');
    this._renderFixButtons(cell, []);

    void this._streamSuggestion(panel, cell, entry);
  }

  /**
   * Cancel any in-flight suggestion for this cell and reset its UI.
   *
   * Skipped when the source change came from our own apply/undo — those
   * mutations would otherwise wipe the entry we are about to update.
   */
  cancel(cellId: string): void {
    if (this._programmaticEdits.has(cellId)) {
      return;
    }
    const entry = this._state.get(cellId);
    if (!entry) {
      return;
    }
    entry.abort.abort();
    if (entry.undoTimer !== null) {
      clearTimeout(entry.undoTimer);
    }
    this._state.delete(cellId);
    // Manual edit / external cancel: take down the AI fix output too,
    // since the user is moving on.
    const cell = this._findCellByIdAnywhere(cellId);
    if (cell) {
      this._removeAIFixOutput(cell);
    }
  }

  /**
   * Cancel + clear any UI regions for this cell. Called when the violation
   * notice is removed (violation resolved).
   */
  clear(cell: Cell): void {
    const cellId = cell.model.id;
    this.cancel(cellId);
    this._removeAIFixOutput(cell);
  }

  /**
   * Apply the fix at `idx` of the cell's current plan.
   */
  async applyFix(cell: Cell, idx: number): Promise<void> {
    const cellId = cell.model.id;
    const entry = this._state.get(cellId);
    if (!entry || entry.state !== 'ready' || !entry.plan) {
      return;
    }
    const fix = entry.plan.fixes[idx];
    if (!fix) {
      return;
    }

    const panel = this._panelForCell(cell);
    if (!panel) {
      return;
    }

    entry.state = 'applying';
    this._renderDiagnosisRegion(cell, `Applying: ${fix.label}…`);
    this._renderFixButtons(cell, [], { disabled: true });

    // Snapshot the full pre-fix state of every code cell BEFORE the apply.
    // This is what undoFix() uses to reverse source-only, structural, and
    // ordering changes uniformly.
    const snapshot = this._takeSnapshot(panel);

    let result: IApplyFixResult;
    try {
      const nb = panel.model?.toJSON();
      const response = await this._postJson<{
        result: IApplyFixResult;
        notebook: unknown;
      }>('apply-fix', { notebook: nb, tool: fix.tool, args: fix.args });
      result = response.result;
    } catch (err) {
      entry.state = 'failed';
      this._renderDiagnosisRegion(cell, `Apply failed: ${err}`);
      this._renderFixButtons(cell, entry.plan.fixes);
      entry.state = 'ready';
      return;
    }

    entry.preFixSnapshot = snapshot;

    // Apply source changes by mutating cell models (which propagates to Y.js).
    this._applySourceChanges(panel, result.post_fix_sources);
    if (result.cells_removed && result.cells_removed.length > 0) {
      this._removeCells(panel, result.cells_removed);
    }
    // Reorder only when the tool actually changed order (move_cell). For
    // merge_cells, the new_cell_order is just the result of removals and
    // doesn't need a separate reorder pass.
    if (fix.tool === 'move_cell') {
      this._applyMoveCell(
        panel,
        fix.args as { cell_id: string; after_cell_id: string }
      );
    }

    // Transition to 'applied' BEFORE the post-apply cleanup so that
    // cellhighlighter sees isFixApplied=true when it re-renders, and
    // therefore suppresses any staleness notice for these cells. Until
    // the user re-runs / manually edits / undoes, the only thing they
    // should see is the AI-fix notice with the "Source updated" message
    // and the Undo button.
    entry.state = 'applied';
    this._postApplyCleanup(panel, fix.tool, result);

    // Now render the post-apply message into the AI-fix notice. It lives
    // on the violating cell, so we don't iterate `affected` here.
    this._ensureAIFixOutput(cell);
    this._renderDiagnosisRegion(
      cell,
      `Source updated (${fix.label}). Re-run the affected cells when you're ready.`
    );
    this._renderFixButtons(cell, []);
    this._renderUndo(cell, true);

    entry.undoTimer = setTimeout(() => {
      this._renderUndo(cell, false);
      entry.undoTimer = null;
    }, UNDO_TTL_MS);
  }

  /**
   * Revert the most recent applied fix on this cell.
   *
   * Uses the pre-fix snapshot to:
   *  - Restore the source of every code cell that still exists.
   *  - Re-insert cells that were removed (by merge_cells), in their
   *    original positions, with their original source + metadata.
   *  - Reorder cells back to their original sequence (by move_cell).
   */
  async undoFix(cell: Cell): Promise<void> {
    const cellId = cell.model.id;
    const entry = this._state.get(cellId);
    if (!entry || entry.state !== 'applied' || !entry.preFixSnapshot) {
      return;
    }
    const panel = this._panelForCell(cell);
    if (!panel || !panel.model) {
      return;
    }

    const snapshot = entry.preFixSnapshot;

    // ── 1. Classify each snapshot cell ──
    //   "missing"  - present in snapshot but removed by the fix (merge_cells)
    //   "changed"  - source OR flowbook metadata differs from snapshot
    //   "untouched"- identical to snapshot, MUST be left alone
    // Leaving "untouched" cells alone is critical: any setSource we call
    // synchronously fires sharedModel.changed → debounced cell_edited →
    // kernel marks them stale, which is exactly the bug the user saw.
    const currentSources = new Map<string, string>();
    const currentFlowbook = new Map<string, unknown>();
    for (let i = 0; i < panel.model.cells.length; i++) {
      const m = panel.model.cells.get(i);
      if (m.type === 'code') {
        currentSources.set(m.id, (m as ICodeCellModel).sharedModel.getSource());
        currentFlowbook.set(m.id, m.getMetadata('flowbook'));
      }
    }

    const missing: string[] = [];
    const changed: string[] = [];
    for (const id of snapshot.codeCellOrder) {
      if (!currentSources.has(id)) {
        missing.push(id);
        continue;
      }
      const snapCell = snapshot.cells.get(id);
      const snapSrc = this._sourceOf(snapCell);
      const snapFlowbook = snapCell?.metadata?.flowbook;
      const curSrc = currentSources.get(id);
      const curFlowbook = currentFlowbook.get(id);
      if (
        curSrc !== snapSrc ||
        !this._flowbookMetaEqual(curFlowbook, snapFlowbook)
      ) {
        changed.push(id);
      }
    }

    // ── 2. Restore source AND flowbook metadata for "changed" cells. ──
    // Restoring errors is what makes the original violation notice come
    // back. Without it, the cell ends up showing the kernel's staleness
    // formatting of the same predicate, which is exactly what the user
    // saw before this fix.
    for (const id of changed) {
      const cellModel = this._findCellModelInPanel(panel, id);
      const snapCell = snapshot.cells.get(id);
      if (!cellModel || !snapCell) {
        continue;
      }
      this._programmaticEdits.add(id);
      try {
        const snapSrc = this._sourceOf(snapCell);
        if (cellModel.sharedModel.getSource() !== snapSrc) {
          cellModel.sharedModel.setSource(snapSrc);
        }
        const snapFlowbook = snapCell?.metadata?.flowbook;
        if (snapFlowbook !== undefined) {
          cellModel.setMetadata('flowbook', snapFlowbook as any);
        } else {
          cellModel.deleteMetadata('flowbook');
        }
      } finally {
        this._programmaticEdits.delete(id);
      }
    }

    // ── 3. Re-insert removed cells in their original positions. ──
    for (const id of missing) {
      const snapPos = snapshot.codeCellOrder.indexOf(id);
      if (snapPos < 0) {
        continue;
      }
      const insertAt = this._findInsertIndex(panel, snapshot, snapPos);
      panel.model.sharedModel.insertCell(insertAt, snapshot.cells.get(id));
    }

    // ── 4. Restore code-cell order (only matters after move_cell). ──
    this._restoreOrder(panel, snapshot.codeCellOrder);

    // ── 5. Take down the AI-fix notice and the suggester entry. ──
    if (entry.undoTimer !== null) {
      clearTimeout(entry.undoTimer);
      entry.undoTimer = null;
    }
    this._removeAIFixOutput(cell);
    this._state.delete(cellId);

    // ── 6. Re-render the cells we touched. isFixApplied is now false, ──
    // so updateCell takes its normal path. For cells with errors metadata
    // restored, the violation notice reappears and the staleness notice
    // self-suppresses (hasViolationMetadata check in StalenessNoticeManager).
    if (this._highlighter) {
      const stalenessManager = this._highlighter.getStalenessManager(panel);
      const cellOrder = getCodeCellOrder(panel);
      const path = panel.context.path;
      const touched = new Set<string>([...changed, ...missing]);
      for (const id of touched) {
        const c = this._findCellInPanel(panel, id);
        if (c) {
          this._highlighter.updateCell(c, stalenessManager, cellOrder, path);
        }
      }
    }
  }

  /** Structural equality check for flowbook metadata snapshots. */
  private _flowbookMetaEqual(a: unknown, b: unknown): boolean {
    if (a === b) {
      return true;
    }
    if (a === undefined || b === undefined) {
      return a === b;
    }
    try {
      return JSON.stringify(a) === JSON.stringify(b);
    } catch {
      return false;
    }
  }

  /** Find a code-cell model in the given panel by id, or null. */
  private _findCellModelInPanel(
    panel: NotebookPanel,
    id: string
  ): ICodeCellModel | null {
    if (!panel.model) {
      return null;
    }
    for (let i = 0; i < panel.model.cells.length; i++) {
      const m = panel.model.cells.get(i);
      if (m.type === 'code' && m.id === id) {
        return m as ICodeCellModel;
      }
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Streaming
  // ─────────────────────────────────────────────────────────────────────────

  private async _streamSuggestion(
    panel: NotebookPanel,
    cell: Cell,
    entry: ICellEntry
  ): Promise<void> {
    const cellId = cell.model.id;
    const settings = ServerConnection.makeSettings();
    const url = `${settings.baseUrl}flowbook/suggest-fix`;

    const nb = panel.model?.toJSON();
    if (!nb) {
      this._renderDiagnosisRegion(cell, 'No notebook model.');
      entry.state = 'failed';
      return;
    }

    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          // Tornado requires the XSRF cookie/header for authenticated POSTs.
          ...this._xsrfHeader(settings)
        },
        body: JSON.stringify({ notebook: nb, cell_id: cellId }),
        signal: entry.abort.signal
      });
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        return;
      }
      this._renderDiagnosisRegion(cell, 'Suggestion request failed.');
      entry.state = 'failed';
      return;
    }

    if (response.status === 503) {
      // Feature disabled — remove the AI-fix output we just added.
      this._removeAIFixOutput(cell);
      this._state.delete(cellId);
      return;
    }
    if (!response.ok || !response.body) {
      this._renderDiagnosisRegion(
        cell,
        `Suggestion service error (${response.status}).`
      );
      entry.state = 'failed';
      return;
    }

    entry.state = 'streaming';
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        // SSE frames are delimited by a blank line.
        let sep;
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          this._handleFrame(cell, entry, frame);
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        return;
      }
      this._renderDiagnosisRegion(cell, 'Stream interrupted.');
      entry.state = 'failed';
    }
  }

  private _handleFrame(cell: Cell, entry: ICellEntry, frame: string): void {
    let eventType = '';
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (!eventType || dataLines.length === 0) {
      return;
    }
    let data: unknown;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      return;
    }

    if (eventType === 'diagnosis') {
      const chunk = (data as { text?: string }).text || '';
      entry.diagnosis += chunk;
      this._renderDiagnosisRegion(cell, entry.diagnosis);
    } else if (eventType === 'plan') {
      entry.plan = data as IFixPlan;
      entry.state = 'ready';
      this._renderFixButtons(cell, entry.plan.fixes);
    } else if (eventType === 'error') {
      const msg = (data as { message?: string }).message || 'unknown error';
      if (!entry.diagnosis) {
        this._renderDiagnosisRegion(cell, `(${msg})`);
      }
      this._renderFixButtons(cell, []);
      entry.state = 'failed';
    } else if (eventType === 'done') {
      // Terminal marker; nothing to do.
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // DOM helpers — locate the regions placed by violationnotice.ts
  // ─────────────────────────────────────────────────────────────────────────

  private _findRegion(
    cell: Cell,
    region: 'diagnosis' | 'buttons' | 'undo'
  ): HTMLElement | null {
    // Outputs live inside cell.node. The violation notice tags its regions
    // with data-flowbook-region.
    return cell.node.querySelector(
      `[data-flowbook-region="${region}"]`
    ) as HTMLElement | null;
  }

  private _renderDiagnosisRegion(cell: Cell, text: string): void {
    const node = this._findRegion(cell, 'diagnosis');
    if (node) {
      node.textContent = text;
      node.hidden = false;
    }
  }

  private _renderFixButtons(
    cell: Cell,
    fixes: IFixSuggestion[],
    opts: { disabled?: boolean } = {}
  ): void {
    const node = this._findRegion(cell, 'buttons');
    if (!node) {
      return;
    }
    node.innerHTML = '';
    if (fixes.length === 0) {
      node.hidden = true;
      return;
    }
    node.hidden = false;
    fixes.forEach((fix, i) => {
      const btn = document.createElement('button');
      btn.className = 'fb-fix-button jp-Button';
      btn.textContent = fix.label;
      btn.title = fix.rationale;
      btn.dataset.fixIdx = String(i);
      btn.dataset.cellId = cell.model.id;
      btn.disabled = !!opts.disabled;
      node.appendChild(btn);
    });
  }

  private _renderUndo(cell: Cell, visible: boolean): void {
    const node = this._findRegion(cell, 'undo');
    if (!node) {
      return;
    }
    node.innerHTML = '';
    if (!visible) {
      node.hidden = true;
      return;
    }
    node.hidden = false;
    const btn = document.createElement('button');
    btn.className = 'fb-undo-button jp-Button';
    btn.textContent = 'Undo fix';
    btn.dataset.cellId = cell.model.id;
    btn.dataset.flowbookAction = 'undo';
    node.appendChild(btn);
  }

  /**
   * Ensure a dedicated AI-fix display_data output exists on the cell.
   * Idempotent — does nothing if one is already present.
   */
  private _ensureAIFixOutput(cell: Cell): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const outputs = (cell.model as ICodeCellModel).outputs;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as IOutput;
      const meta = (out.metadata || {}) as Record<string, unknown>;
      if (meta[AI_FIX_NOTICE_FLAG]) {
        return;
      }
    }
    const noticeOutput: IOutput = {
      output_type: 'display_data',
      data: {
        'text/html': AI_FIX_NOTICE_HTML,
        'text/plain': ''
      },
      metadata: { [AI_FIX_NOTICE_FLAG]: true }
    };
    // Append at the end so existing kernel outputs (errors, prints) stay
    // visible above it.
    const all: IOutput[] = [];
    for (let i = 0; i < outputs.length; i++) {
      all.push(outputs.get(i).toJSON() as IOutput);
    }
    all.push(noticeOutput);
    outputs.fromJSON(all);
  }

  /** Remove any AI-fix display_data output from the cell. */
  private _removeAIFixOutput(cell: Cell): void {
    if (cell.model.type !== 'code') {
      return;
    }
    const outputs = (cell.model as ICodeCellModel).outputs;
    const surviving: IOutput[] = [];
    let removed = false;
    for (let i = 0; i < outputs.length; i++) {
      const out = outputs.get(i).toJSON() as IOutput;
      const meta = (out.metadata || {}) as Record<string, unknown>;
      if (meta[AI_FIX_NOTICE_FLAG]) {
        removed = true;
        continue;
      }
      surviving.push(out);
    }
    if (removed) {
      outputs.fromJSON(surviving);
    }
  }

  /**
   * Search every tracked notebook for a cell with the given id. Returns
   * null when the cell has been removed (e.g. via merge_cells) or no
   * notebook owns it.
   */
  private _findCellByIdAnywhere(cellId: string): Cell | null {
    let found: Cell | null = null;
    this._tracker.forEach(panel => {
      if (found) {
        return;
      }
      for (const widget of panel.content.widgets) {
        if (widget.model.id === cellId) {
          found = widget;
          return;
        }
      }
    });
    return found;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Notebook mutation
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * After source/structural changes land, clear violation metadata on the
   * modified cells and trigger a highlighter re-render. We deliberately
   * do NOT mark the cells stale — until the user runs or edits, the only
   * notice on these cells should be the AI-fix notice with the Undo
   * button. The cellhighlighter sees isFixApplied(cellId)=true and skips
   * staleness rendering accordingly (even when the kernel echoes back a
   * stale flag a moment later).
   */
  private _postApplyCleanup(
    panel: NotebookPanel,
    tool: IFixSuggestion['tool'],
    result: IApplyFixResult
  ): void {
    if (!this._highlighter || !panel.model) {
      return;
    }
    const stalenessManager = this._highlighter.getStalenessManager(panel);
    const cellOrder = getCodeCellOrder(panel);
    const path = panel.context.path;

    // Affected cells: everything the dispatcher rewrote, plus the moved
    // cell when the tool is move_cell (which doesn't change source).
    const affected = new Set<string>(result.modified_cells);
    if (tool === 'move_cell') {
      const movedId = (result.args as { cell_id?: string })?.cell_id;
      if (typeof movedId === 'string') {
        affected.add(movedId);
      }
    }

    for (const cellId of affected) {
      const cell = this._findCellInPanel(panel, cellId);
      if (!cell) {
        continue;
      }
      // Clear violation entries on the cell's flowbook metadata so the
      // next updateViolationNotice pass removes the red error box.
      // Preserve the rest (read/write locs etc.) for the metadata panel.
      const meta = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      if (meta && meta.errors && meta.errors.length > 0) {
        const cleared: IReproducibilityMetadata = { ...meta, errors: [] };
        cell.model.setMetadata('flowbook', cleared as any);
      }

      // Re-render through the highlighter. Because entry.state was set
      // to 'applied' before this call, the highlighter's isFixApplied
      // check will skip stale rendering for these cells.
      this._highlighter.updateCell(cell, stalenessManager, cellOrder, path);
    }
  }

  private _findCellInPanel(panel: NotebookPanel, cellId: string): Cell | null {
    for (const widget of panel.content.widgets) {
      if (widget.model.id === cellId) {
        return widget;
      }
    }
    return null;
  }

  private _applySourceChanges(
    panel: NotebookPanel,
    sources: Record<string, string>
  ): void {
    if (!panel.model) {
      return;
    }
    for (let i = 0; i < panel.model.cells.length; i++) {
      const cellModel = panel.model.cells.get(i);
      if (cellModel.type !== 'code') {
        continue;
      }
      const newSource = sources[cellModel.id];
      if (newSource === undefined) {
        continue;
      }
      // Mark this id as a programmatic edit just long enough for the
      // synchronous sharedModel.changed signal to fire. cancel() will
      // see the flag and skip. The kernel-side _sendCellEdited debounce
      // (also in _onCellContentChanged) still runs — the cell really
      // did change, and the kernel should mark it stale accordingly.
      this._programmaticEdits.add(cellModel.id);
      try {
        (cellModel as ICodeCellModel).sharedModel.setSource(newSource);
      } finally {
        this._programmaticEdits.delete(cellModel.id);
      }
    }
  }

  private _removeCells(panel: NotebookPanel, cellIds: string[]): void {
    if (!panel.model) {
      return;
    }
    // sharedModel.deleteCellRange wants indices; collect them.
    const ids = new Set(cellIds);
    for (let i = panel.model.cells.length - 1; i >= 0; i--) {
      if (ids.has(panel.model.cells.get(i).id)) {
        panel.model.sharedModel.deleteCell(i);
      }
    }
  }

  /**
   * Move a single cell so it sits immediately after another cell.
   * Mirrors the backend's _move_cell semantics.
   */
  private _applyMoveCell(
    panel: NotebookPanel,
    args: { cell_id: string; after_cell_id: string }
  ): void {
    if (!panel.model) {
      return;
    }
    const srcIdx = this._indexOfCell(panel, args.cell_id);
    const dstAfterIdx = this._indexOfCell(panel, args.after_cell_id);
    if (srcIdx < 0 || dstAfterIdx < 0 || srcIdx === dstAfterIdx) {
      return;
    }
    // sharedModel.moveCell(fromIndex, toIndex) interprets toIndex as the
    // post-removal target. We want the cell to land immediately after
    // dstAfterIdx, so the post-removal target is dstAfterIdx if the source
    // was after the destination, or dstAfterIdx (without +1, since the
    // removal already shifts subsequent cells down by one) if it was before.
    const toIndex = srcIdx < dstAfterIdx ? dstAfterIdx : dstAfterIdx + 1;
    panel.model.sharedModel.moveCell(srcIdx, toIndex);
  }

  /**
   * Reorder code cells back to the given sequence. Tolerates extra cells
   * (e.g. markdown cells interleaved between code cells) by only moving
   * code cells. Called from undoFix() to reverse move_cell.
   */
  private _restoreOrder(panel: NotebookPanel, codeOrder: string[]): void {
    if (!panel.model) {
      return;
    }
    // Walk the desired sequence. For each desired ID, find where it
    // currently sits and where the corresponding code-slot in the live
    // notebook is, then move if mismatched.
    for (let pos = 0; pos < codeOrder.length; pos++) {
      const want = codeOrder[pos];
      const actualIdx = this._indexOfCell(panel, want);
      const targetIdx = this._absIndexOfCodeSlot(panel, pos);
      if (actualIdx < 0 || targetIdx < 0 || actualIdx === targetIdx) {
        continue;
      }
      panel.model.sharedModel.moveCell(actualIdx, targetIdx);
    }
  }

  /** Find the absolute index of the cell with the given id, or -1. */
  private _indexOfCell(panel: NotebookPanel, id: string): number {
    if (!panel.model) {
      return -1;
    }
    for (let i = 0; i < panel.model.cells.length; i++) {
      if (panel.model.cells.get(i).id === id) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Return the absolute index of the n-th code cell (0-based). Used to map
   * code-cell positions to sharedModel indices when markdown cells may be
   * interleaved.
   */
  private _absIndexOfCodeSlot(panel: NotebookPanel, codePos: number): number {
    if (!panel.model) {
      return -1;
    }
    let seen = 0;
    for (let i = 0; i < panel.model.cells.length; i++) {
      if (panel.model.cells.get(i).type === 'code') {
        if (seen === codePos) {
          return i;
        }
        seen++;
      }
    }
    // If codePos is one past the end of existing code cells, return one past
    // the last code cell — that's the right insertion point.
    return panel.model.cells.length;
  }

  /**
   * Capture every code cell's full JSON plus the code-cell ordering.
   * Called just before applying a fix so undoFix() can restore state.
   */
  private _takeSnapshot(panel: NotebookPanel): IPreFixSnapshot {
    const snap: IPreFixSnapshot = { codeCellOrder: [], cells: new Map() };
    if (!panel.model) {
      return snap;
    }
    for (let i = 0; i < panel.model.cells.length; i++) {
      const m = panel.model.cells.get(i);
      if (m.type !== 'code') {
        continue;
      }
      snap.codeCellOrder.push(m.id);
      // toJSON gives a plain object we can pass back to insertCell.
      snap.cells.set(m.id, m.toJSON());
    }
    return snap;
  }

  /**
   * Find the insertion index for a snapshot cell at position `snapPos`.
   * Counts surviving snapshot-cells that precede it in the original
   * ordering and that exist in the current notebook — that count tells us
   * which code-slot the cell should land in.
   */
  private _findInsertIndex(
    panel: NotebookPanel,
    snapshot: IPreFixSnapshot,
    snapPos: number
  ): number {
    let codePos = 0;
    for (let i = 0; i < snapPos; i++) {
      if (this._indexOfCell(panel, snapshot.codeCellOrder[i]) >= 0) {
        codePos++;
      }
    }
    return this._absIndexOfCodeSlot(panel, codePos);
  }

  private _sourceOf(cellJSON: any): string {
    const s = cellJSON?.source;
    if (Array.isArray(s)) {
      return s.join('');
    }
    return s || '';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Small helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _panelForCell(cell: Cell): NotebookPanel | null {
    let found: NotebookPanel | null = null;
    this._tracker.forEach(panel => {
      if (found || !panel.model) {
        return;
      }
      for (let i = 0; i < panel.model.cells.length; i++) {
        if (panel.model.cells.get(i).id === cell.model.id) {
          found = panel;
          return;
        }
      }
    });
    return found;
  }

  private async _postJson<T>(endpoint: string, body: unknown): Promise<T> {
    const settings = ServerConnection.makeSettings();
    const url = `${settings.baseUrl}flowbook/${endpoint}`;
    const response = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...this._xsrfHeader(settings)
      },
      body: JSON.stringify(body)
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    return (await response.json()) as T;
  }

  private _xsrfHeader(
    _settings: ServerConnection.ISettings
  ): Record<string, string> {
    const match = document.cookie.match(/_xsrf=([^;]+)/);
    if (match) {
      return { 'X-XSRFToken': decodeURIComponent(match[1]) };
    }
    return {};
  }
}
