/**
 * NBI Bridge — JupyterLab commands that expose FlowBook's internal state.
 *
 * These commands are called from the NBI extension backend via run_ui_command().
 * All cell indices are code-cell-only (markdown cells are skipped).
 * All returned labels use @A notation via indexToAlpha().
 */

import { JupyterFrontEnd } from '@jupyterlab/application';
import {
  INotebookTracker,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { ICodeCellModel } from '@jupyterlab/cells';

import { ReproducibilityCellHighlighter } from './cellhighlighter';
import { ReproducibilityExecutionHookManager } from './executionhook';
import { KernelDetector } from '../shared/kerneldetection';
import { IReproducibilityMetadata, IReproducibilityError } from './types';
import { indexToAlpha, getCodeCellOrder } from '../cellindexutils';
import { StalenessManager } from './stalenessmanager';

// ---------------------------------------------------------------------------
// Command result types (returned by the commands defined below)
// ---------------------------------------------------------------------------

interface IActionableResult {
  done: boolean;
  index?: number;
  label?: string;
  cell_id?: string;
  reason?: string;
}

interface IRunCellResult {
  status: 'ok' | 'error' | 'violation';
  outputs_text?: string;
  label?: string;
  cell_id?: string;
  flowbook_meta?: IReproducibilityMetadata | null;
  errors?: IReproducibilityError[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert a code-cell index to a notebook widget index.
 * Code-cell index 0 = first code cell, skipping markdown cells.
 */
function codeCellToWidgetIndex(
  panel: NotebookPanel,
  codeCellIndex: number
): number {
  const widgets = panel.content.widgets;
  let codeIdx = 0;
  for (let i = 0; i < widgets.length; i++) {
    if (widgets[i].model.type === 'code') {
      if (codeIdx === codeCellIndex) {
        return i;
      }
      codeIdx++;
    }
  }
  throw new Error(
    `Code cell index ${codeCellIndex} out of range (have ${codeIdx} code cells)`
  );
}

// ---------------------------------------------------------------------------
// MIME bundle → Output dict (mirrors flowbook/server/kernel_helper.py:_filter_mime_bundle)
// ---------------------------------------------------------------------------

const KEEP_IMAGE_MIMES = ['image/png', 'image/jpeg', 'image/svg+xml'];
const KEEP_TEXT_MIMES = ['text/html', 'text/plain'];

function filterMimeBundle(
  data: Record<string, any>
): Record<string, any> {
  const out: Record<string, any> = {};
  for (const mime in data) {
    const value = data[mime];
    if (KEEP_IMAGE_MIMES.includes(mime)) {
      const b64 = typeof value === 'string' ? value : '';
      out[mime] = {
        encoding: 'base64',
        bytes: b64,
        size_bytes: Math.floor((b64.length * 3) / 4)
      };
    } else if (KEEP_TEXT_MIMES.includes(mime)) {
      const text = typeof value === 'string' ? value : String(value ?? '');
      out[mime] = { text };
    } else {
      const asText =
        typeof value === 'string'
          ? value
          : JSON.stringify(value ?? {}) || String(value ?? '');
      out[mime] = { size_bytes: asText.length };
    }
  }
  return out;
}

/**
 * Convert one notebook-model output entry into the shared Output dict shape
 * used by flowbook/tools/mcp_content.py (stream / execute_result /
 * display_data / error).
 */
function outputToDict(output: any): any {
  if (!output) {
    return null;
  }
  const type = output.type;
  if (type === 'stream') {
    return {
      kind: 'stream',
      stream_name: output.name ?? 'stdout',
      text: output.text ?? output.data?.['text/plain'] ?? ''
    };
  }
  if (type === 'execute_result' || type === 'display_data') {
    return { kind: type, data: filterMimeBundle(output.data ?? {}) };
  }
  if (type === 'error') {
    const esc = String.fromCharCode(27);
    const ansiRegex = new RegExp(esc + '\\[[0-9;]*m', 'g');
    const tb = Array.isArray(output.traceback)
      ? output.traceback.map((l: string) => l.replace(ansiRegex, ''))
      : [];
    const text = [
      `${output.ename ?? ''}: ${output.evalue ?? ''}`,
      ...tb
    ].join('\n');
    return { kind: 'error', data: { 'text/plain': { text } } };
  }
  return null;
}

/**
 * Format cell outputs as text for tool responses.
 */
function formatOutputsText(cell: any): string {
  const outputs = cell.model?.outputs;
  if (!outputs) {
    return '';
  }
  const parts: string[] = [];
  for (let i = 0; i < outputs.length; i++) {
    const output = outputs.get(i);
    if (!output) {
      continue;
    }
    const outputType = output.type;
    if (outputType === 'stream') {
      parts.push(output.data?.['text/plain'] || output.text || '');
    } else if (
      outputType === 'execute_result' ||
      outputType === 'display_data'
    ) {
      const text = output.data?.['text/plain'];
      if (text) {
        parts.push(text);
      }
    } else if (outputType === 'error') {
      const traceback = output.traceback;
      if (traceback) {
        // Strip ANSI escape codes (ESC[ sequences)
        const esc = String.fromCharCode(27);
        const ansiRegex = new RegExp(esc + '\\[[0-9;]*m', 'g');
        parts.push((traceback as string[]).join('\n').replace(ansiRegex, ''));
      } else {
        parts.push(`${output.ename}: ${output.evalue}`);
      }
    }
  }
  return parts.join('\n');
}

/**
 * Check if a cell has execution errors in its outputs.
 */
function cellHasError(panel: NotebookPanel, widgetIndex: number): boolean {
  const cell = panel.content.widgets[widgetIndex];
  if (!cell || cell.model.type !== 'code') {
    return false;
  }
  const outputs = (cell.model as ICodeCellModel).outputs;
  if (!outputs) {
    return false;
  }
  for (let i = 0; i < outputs.length; i++) {
    const output = outputs.get(i);
    if (output && output.type === 'error') {
      return true;
    }
  }
  return false;
}

/**
 * Check if a cell has FlowBook violations (predicate errors).
 */
function cellHasViolation(
  panel: NotebookPanel,
  widgetIndex: number
): IReproducibilityError[] {
  const cell = panel.content.widgets[widgetIndex];
  if (!cell || cell.model.type !== 'code') {
    return [];
  }
  const meta = cell.model.getMetadata('flowbook') as
    | IReproducibilityMetadata
    | undefined;
  return meta?.errors || [];
}

// ---------------------------------------------------------------------------
// Bridge context — set by plugin.ts during activation
// ---------------------------------------------------------------------------

let _highlighter: ReproducibilityCellHighlighter | null = null;
let _executionHook: ReproducibilityExecutionHookManager | null = null;
let _kernelDetector: KernelDetector | null = null;
let _tracker: INotebookTracker | null = null;

/**
 * Update the bridge's references to FlowBook internals.
 * Called from plugin.ts when FlowBook activates/deactivates.
 */
export function setBridgeContext(
  highlighter: ReproducibilityCellHighlighter | null,
  executionHook: ReproducibilityExecutionHookManager | null,
  kernelDetector: KernelDetector | null,
  tracker: INotebookTracker | null
): void {
  _highlighter = highlighter;
  _executionHook = executionHook;
  _kernelDetector = kernelDetector;
  _tracker = tracker;
}

function getPanel(): NotebookPanel {
  const panel = _tracker?.currentWidget;
  if (!panel) {
    throw new Error('No active notebook');
  }
  return panel;
}

function getStalenessManager(): StalenessManager {
  if (!_highlighter) {
    throw new Error('FlowBook not active');
  }
  return _highlighter.getStalenessManager(getPanel());
}

// ---------------------------------------------------------------------------
// Command registration
// ---------------------------------------------------------------------------

/**
 * Register all FlowBook NBI bridge commands.
 * Called once at plugin startup.
 */
export function registerBridgeCommands(
  app: JupyterFrontEnd,
  tracker: INotebookTracker,
  kernelDetector: KernelDetector
): void {
  _tracker = tracker;
  _kernelDetector = kernelDetector;

  // ------------------------------------------------------------------
  // flowbook:is-active
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:is-active', {
    execute: () => {
      const panel = tracker.currentWidget;
      if (!panel || !_kernelDetector) {
        return { active: false };
      }
      const isFlowbook = _kernelDetector.isFlowbookKernel(panel);
      return {
        active: isFlowbook,
        kernel: isFlowbook
          ? 'flowbook_kernel'
          : panel.sessionContext.session?.kernel?.name || 'unknown'
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-metadata — get FlowBook metadata for a code cell
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-metadata', {
    execute: args => {
      const panel = getPanel();
      const codeCellIndex = args.cellIndex as number;
      const widgetIdx = codeCellToWidgetIndex(panel, codeCellIndex);
      const cell = panel.content.widgets[widgetIdx];
      const meta = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      const label = indexToAlpha(codeCellIndex);

      return {
        label,
        cell_id: cell.model.id,
        ...(meta || {}),
        // Override stale_cells with @-labels
        stale_cells_labels: meta?.stale_cells?.map((id: string) => {
          const cellOrder = getCodeCellOrder(panel);
          const idx = cellOrder.indexOf(id);
          return idx >= 0 ? indexToAlpha(idx) : id;
        })
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-stale-cells
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-stale-cells', {
    execute: () => {
      const panel = getPanel();
      const mgr = getStalenessManager();
      const staleCells = mgr.staleCells;
      const cellOrder = getCodeCellOrder(panel);

      const result: any[] = [];
      cellOrder.forEach((cellId, idx) => {
        if (staleCells.has(cellId)) {
          const reason = mgr.getReason(cellId);
          const reasonMsg =
            reason && 'message' in reason
              ? reason.message
              : reason?.type || 'stale';
          result.push({
            index: idx,
            label: indexToAlpha(idx),
            cell_id: cellId,
            reason: reasonMsg
          });
        }
      });
      return result;
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-next-actionable — error > stale > unexecuted
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-next-actionable', {
    execute: () => {
      const panel = getPanel();
      const widgets = panel.content.widgets;
      const staleCells = _highlighter
        ? getStalenessManager().staleCells
        : new Set<string>();

      // Priority 1: cells with errors
      // Priority 2: stale cells
      // Priority 3: unexecuted cells
      let codeIdx = 0;
      let firstStale: any = null;
      let firstUnexecuted: any = null;

      for (let i = 0; i < widgets.length; i++) {
        const cell = widgets[i];
        if (cell.model.type !== 'code') {
          continue;
        }
        const codeModel = cell.model as ICodeCellModel;
        const source = codeModel.sharedModel.getSource();
        const cellId = cell.model.id;
        const currentCodeIdx = codeIdx;
        codeIdx++;

        // Skip empty cells
        if (!source || source.trim() === '') {
          continue;
        }

        // Check for errors in FlowBook metadata
        const meta = cell.model.getMetadata('flowbook') as
          | IReproducibilityMetadata
          | undefined;
        if (meta?.errors && meta.errors.length > 0) {
          return {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: 'error',
            error: meta.errors[0].message
          };
        }

        // Check for output errors
        if (cellHasError(panel, i)) {
          return {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: 'error'
          };
        }

        // Track first stale
        if (!firstStale && staleCells.has(cellId)) {
          firstStale = {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: 'stale'
          };
        }

        // Track first unexecuted
        if (!firstUnexecuted && codeModel.executionCount === null) {
          firstUnexecuted = {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: 'unexecuted'
          };
        }
      }

      if (firstStale) {
        return firstStale;
      }
      if (firstUnexecuted) {
        return firstUnexecuted;
      }
      return { done: true };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-status
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-status', {
    execute: () => {
      const panel = getPanel();
      const widgets = panel.content.widgets;
      const staleCells = _highlighter
        ? getStalenessManager().staleCells
        : new Set<string>();

      let totalCodeCells = 0;
      let executed = 0;
      let stale = 0;
      let withErrors = 0;
      let empty = 0;

      for (let i = 0; i < widgets.length; i++) {
        const cell = widgets[i];
        if (cell.model.type !== 'code') {
          continue;
        }
        totalCodeCells++;
        const codeModel = cell.model as ICodeCellModel;
        const source = codeModel.sharedModel.getSource();

        if (!source || source.trim() === '') {
          empty++;
          continue;
        }

        if (codeModel.executionCount !== null) {
          executed++;
        }
        if (staleCells.has(cell.model.id)) {
          stale++;
        }
        const meta = cell.model.getMetadata('flowbook') as
          | IReproducibilityMetadata
          | undefined;
        if (meta?.errors && meta.errors.length > 0) {
          withErrors++;
        }
        if (cellHasError(panel, i)) {
          withErrors++;
        }
      }

      const nonEmpty = totalCodeCells - empty;
      const clean = executed - stale;
      const reproducible =
        nonEmpty > 0 &&
        executed === nonEmpty &&
        stale === 0 &&
        withErrors === 0;

      return {
        total_code_cells: totalCodeCells,
        non_empty: nonEmpty,
        executed,
        stale,
        clean,
        with_errors: withErrors,
        reproducible
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-cell-count
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-cell-count', {
    execute: () => {
      const panel = getPanel();
      const widgets = panel.content.widgets;
      let codeCells = 0;
      let markdownCells = 0;
      for (let i = 0; i < widgets.length; i++) {
        if (widgets[i].model.type === 'code') {
          codeCells++;
        } else if (widgets[i].model.type === 'markdown') {
          markdownCells++;
        }
      }
      return {
        total: widgets.length,
        code_cells: codeCells,
        markdown_cells: markdownCells
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-cell — full cell data by code-cell index
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-cell', {
    execute: args => {
      const panel = getPanel();
      const codeCellIndex = args.cellIndex as number;
      const widgetIdx = codeCellToWidgetIndex(panel, codeCellIndex);
      const cell = panel.content.widgets[widgetIdx];
      const label = indexToAlpha(codeCellIndex);
      const codeModel = cell.model as ICodeCellModel;
      const meta = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;

      return {
        label,
        cell_id: cell.model.id,
        cell_type: cell.model.type,
        source: codeModel.sharedModel.getSource(),
        execution_count: codeModel.executionCount,
        outputs_text: formatOutputsText(cell),
        flowbook_meta: meta || null
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-cell-output
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-cell-output', {
    execute: args => {
      const panel = getPanel();
      const codeCellIndex = args.cellIndex as number;
      const widgetIdx = codeCellToWidgetIndex(panel, codeCellIndex);
      const cell = panel.content.widgets[widgetIdx];
      const label = indexToAlpha(codeCellIndex);

      return {
        label,
        outputs_text: formatOutputsText(cell)
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:edit-cell-source — identity-safe source modification
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:edit-cell-source', {
    execute: args => {
      const panel = getPanel();
      const codeCellIndex = args.cellIndex as number;
      const newSource = args.source as string;
      const widgetIdx = codeCellToWidgetIndex(panel, codeCellIndex);
      const cell = panel.content.widgets[widgetIdx];

      // setSource() modifies in-place — preserves cell ID and triggers
      // FlowBook's sharedModel.changed listener for edit detection
      cell.model.sharedModel.setSource(newSource);

      return {
        label: indexToAlpha(codeCellIndex),
        cell_id: cell.model.id
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:move-cell — identity-safe reorder
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:move-cell', {
    execute: args => {
      const panel = getPanel();
      const fromCodeIdx = args.fromIndex as number;
      const toCodeIdx = args.toIndex as number;
      const fromWidget = codeCellToWidgetIndex(panel, fromCodeIdx);
      const toWidget = codeCellToWidgetIndex(panel, toCodeIdx);

      const sharedModel = panel.content.model?.sharedModel;
      if (!sharedModel) {
        throw new Error('No shared model');
      }

      sharedModel.moveCell(fromWidget, toWidget);

      return {
        label: indexToAlpha(fromCodeIdx),
        newIndex: toCodeIdx
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:notify-structure — send cell order to kernel
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:notify-structure', {
    execute: () => {
      const panel = getPanel();
      if (_executionHook) {
        const cellOrder = getCodeCellOrder(panel);
        _executionHook.sendCommand({
          type: 'notebook_structure',
          cell_order: cellOrder
        });
      }
    }
  });

  // ------------------------------------------------------------------
  // flowbook:set-continue-after-violation
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:set-continue-after-violation', {
    execute: args => {
      const enabled = args.enabled as boolean;
      if (_executionHook) {
        _executionHook.sendCommand({
          type: 'continue_after_violation',
          enabled
        });
      }
      return { enabled };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:enforcer-checkpoint — snapshot kernel enforcer state
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:enforcer-checkpoint', {
    execute: () => {
      if (_executionHook) {
        _executionHook.sendCommand({ type: 'enforcer_checkpoint' });
      }
      // The kernel responds via comm with enforcer_checkpoint_result
      // which the executionHook processes. Return void for now.
      return {};
    }
  });

  // ------------------------------------------------------------------
  // flowbook:enforcer-restore — restore kernel enforcer state
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:enforcer-restore', {
    execute: args => {
      const checkpointId = args.checkpointId as string;
      if (_executionHook) {
        _executionHook.sendCommand({
          type: 'enforcer_restore',
          checkpoint_id: checkpointId
        });
      }
      return {};
    }
  });

  // ------------------------------------------------------------------
  // flowbook:run-cell — run a code cell and return metadata
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:run-cell', {
    execute: async args => {
      const panel = getPanel();
      const codeCellIndex = args.cellIndex as number;
      const widgetIdx = codeCellToWidgetIndex(panel, codeCellIndex);
      const label = indexToAlpha(codeCellIndex);

      // Activate and run via JupyterLab's native mechanism
      panel.content.activeCellIndex = widgetIdx;
      await NotebookActions.run(panel.content, panel.sessionContext);

      // Read results after execution completes
      const cell = panel.content.widgets[widgetIdx];
      const meta = cell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;
      const hasError = cellHasError(panel, widgetIdx);
      const violations = cellHasViolation(panel, widgetIdx);

      return {
        label,
        cell_id: cell.model.id,
        status: hasError ? 'error' : violations.length > 0 ? 'violation' : 'ok',
        outputs_text: formatOutputsText(cell),
        flowbook_meta: meta || null,
        errors: violations
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:run-actionable-cells — loop until clean or error
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:run-actionable-cells', {
    execute: async () => {
      getPanel(); // Verify a notebook is open
      const results: any[] = [];
      let totalRun = 0;
      const maxIterations = 500; // safety limit

      while (totalRun < maxIterations) {
        try {
          // Find next actionable
          const actionable = (await app.commands.execute(
            'flowbook:get-next-actionable'
          )) as IActionableResult;
          if (actionable.done) {
            break;
          }

          const codeCellIndex = actionable.index as number;
          const label = actionable.label as string;

          // Run the cell
          const runResult = (await app.commands.execute('flowbook:run-cell', {
            cellIndex: codeCellIndex
          })) as IRunCellResult;

          totalRun++;
          results.push({
            label,
            status: runResult.status,
            outputs_preview: (runResult.outputs_text || '').slice(0, 200)
          });

          // Stop on hard error (exception/syntax)
          if (runResult.status === 'error') {
            break;
          }

          // Stop on violation if continue_after_violation is false
          // (when it's true, violations are accepted and the cell still becomes "ok")
          if (runResult.status === 'violation') {
            break;
          }
        } catch (error) {
          console.error('Error in run-actionable-cells:', error);
          break;
        }
      }

      // Get final status
      const status = await app.commands.execute('flowbook:get-status');

      return {
        results,
        cells_run: totalRun,
        summary: status
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:add-cell — insert a new cell at a specific position.
  //
  // Semantics mirror the MCP / CLI add_cell path:
  //   - cellIndex: widget index at which to insert (cell at and after
  //     that index is pushed down by 1).
  //   - When called with afterCodeCellIndex=N, inserts AT
  //     widgetIndex(codeCellN) + 1 — i.e. immediately after @<N>.
  //   - When both fields are omitted, appends at the end.
  // The NBI add_cell tool passes one of these fields; we handle both
  // here to keep the command self-contained.
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:add-cell', {
    execute: args => {
      const panel = getPanel();
      if (!panel.model) {
        throw new Error('Notebook model not available');
      }
      const model = panel.model.sharedModel;
      const source = (args.source as string) ?? '';
      const cellType = (args.cellType as 'code' | 'markdown') ?? 'code';

      let insertAt: number;
      if (typeof args.afterCodeCellIndex === 'number') {
        const widgetIdx = codeCellToWidgetIndex(
          panel,
          args.afterCodeCellIndex as number
        );
        insertAt = widgetIdx + 1;
      } else if (typeof args.cellIndex === 'number') {
        insertAt = args.cellIndex as number;
      } else {
        insertAt = model.cells.length;
      }

      const inserted = model.insertCell(insertAt, {
        cell_type: cellType,
        metadata: { trusted: true },
        source
      });

      return {
        cell_id: (inserted as any)?.id ?? null,
        inserted_at: insertAt
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:scratch-work — run code against the live kernel with
  // silent=True and namespace checkpoint/restore (flowbook_isolate).
  // Returns a ScratchResult dict (see flowbook/tools/mcp_content.py).
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:scratch-work', {
    execute: async args => {
      const panel = getPanel();
      const code = (args.code as string) || '';
      const session = panel.sessionContext.session;
      const kernel = session?.kernel;
      if (!kernel) {
        return {
          status: 'error',
          execution_time_ms: 0,
          outputs: [],
          error: {
            ename: 'NoKernel',
            evalue: 'No active kernel',
            traceback: []
          }
        };
      }

      const t0 = performance.now();
      const outputs: any[] = [];
      let status: 'ok' | 'error' = 'ok';
      let error: any = null;

      // Third arg is the message metadata — FlowBook's silent fast-path
      // reads `flowbook_isolate` from there and wraps execution in
      // checkpoint/restore.
      const future = kernel.requestExecute(
        { code, silent: true, store_history: false },
        true,
        { flowbook_isolate: true }
      );

      future.onIOPub = (msg: any) => {
        const mt = msg.header.msg_type;
        const content = msg.content;
        if (mt === 'stream') {
          outputs.push({
            kind: 'stream',
            stream_name: content.name,
            text: content.text
          });
        } else if (mt === 'execute_result') {
          outputs.push({
            kind: 'execute_result',
            data: filterMimeBundle(content.data ?? {})
          });
        } else if (mt === 'display_data') {
          outputs.push({
            kind: 'display_data',
            data: filterMimeBundle(content.data ?? {})
          });
        } else if (mt === 'error') {
          status = 'error';
          error = {
            ename: content.ename ?? '',
            evalue: content.evalue ?? '',
            traceback: (content.traceback ?? []).map((l: string) => l)
          };
        }
      };

      await future.done;
      return {
        status,
        execution_time_ms: performance.now() - t0,
        outputs,
        error
      };
    }
  });

  // ------------------------------------------------------------------
  // flowbook:get-cell-outputs — full outputs for the given cells,
  // including images and HTML. Pure notebook-model read; no kernel call.
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-cell-outputs', {
    execute: args => {
      const panel = getPanel();
      const cellIds = (args.cellIds as string[]) || [];
      const widgets = panel.content.widgets;

      const byId = new Map<string, { label: string; cell: any }>();
      let codeIdx = 0;
      for (let i = 0; i < widgets.length; i++) {
        const w = widgets[i];
        if (w.model.type !== 'code') {
          continue;
        }
        byId.set(w.model.id, {
          label: `@${indexToAlpha(codeIdx)}`,
          cell: w
        });
        codeIdx++;
      }

      const cells: any[] = [];
      for (const cid of cellIds) {
        const entry = byId.get(cid);
        if (!entry) {
          cells.push({
            cell_id: cid,
            label: cid,
            outputs: [
              {
                kind: 'error',
                data: {
                  'text/plain': { text: `cell not found: ${cid}` }
                }
              }
            ]
          });
          continue;
        }
        const model = entry.cell.model as ICodeCellModel;
        const outputs = model.outputs;
        const outArr: any[] = [];
        if (outputs) {
          for (let i = 0; i < outputs.length; i++) {
            const o = outputs.get(i);
            const asDict = outputToDict(o);
            if (asDict) {
              outArr.push(asDict);
            }
          }
        }
        cells.push({ cell_id: cid, label: entry.label, outputs: outArr });
      }
      return { cells };
    }
  });
}
