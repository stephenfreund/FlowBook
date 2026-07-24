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
import {
  IReproducibilityMetadata,
  IReproducibilityError,
  waitForFlowbookMetadata
} from './types';
import { aiTransact } from './aiattribution';
import { indexToAlpha, getCodeCellOrder } from '../cellindexutils';
import { StalenessManager } from './stalenessmanager';
import { RunToCleanGuard, formatFootprintChange } from './rerunguard';

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
  //
  // With { documentOrder: true }, stale and unexecuted cells are merged
  // in document order (errors still first). The RunToClean loop uses
  // this so every cell sees its final inputs before it first runs;
  // rerunning a later stale cell before an earlier unexecuted one can
  // legitimately change the later cell's behavior twice, which would
  // trip the rerun footprint check on a deterministic notebook.
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:get-next-actionable', {
    execute: args => {
      const documentOrder = (args?.documentOrder as boolean) ?? false;
      const panel = getPanel();
      const widgets = panel.content.widgets;
      const staleCells = _highlighter
        ? getStalenessManager().staleCells
        : new Set<string>();

      // Priority 1: cells with errors
      // Priority 2: stale cells
      // Priority 3: unexecuted cells
      // (2 and 3 merge into document order when documentOrder is set)
      let codeIdx = 0;
      let firstStale: any = null;
      let firstUnexecuted: any = null;
      let firstStaleOrUnexecuted: any = null;

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
        const isStale = staleCells.has(cellId);
        if (!firstStale && isStale) {
          firstStale = {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: 'stale'
          };
        }

        // Track first unexecuted
        const isUnexecuted = codeModel.executionCount === null;
        if (!firstUnexecuted && isUnexecuted) {
          firstUnexecuted = {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: 'unexecuted'
          };
        }

        // Track first stale-or-unexecuted (document order)
        if (!firstStaleOrUnexecuted && (isStale || isUnexecuted)) {
          firstStaleOrUnexecuted = {
            index: currentCodeIdx,
            label: indexToAlpha(currentCodeIdx),
            cell_id: cellId,
            reason: isStale ? 'stale' : 'unexecuted'
          };
        }
      }

      if (documentOrder) {
        return firstStaleOrUnexecuted ?? { done: true };
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
      // FlowBook's sharedModel.changed listener for edit detection. Tag the
      // transaction so LogBook attributes this NBI-driven edit to the AI.
      aiTransact(panel.model?.sharedModel, () =>
        cell.model.sharedModel.setSource(newSource)
      );

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

      aiTransact(sharedModel, () => sharedModel.moveCell(fromWidget, toWidget));

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
  // flowbook:run-cell — run a code cell and return metadata
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:run-cell', {
    execute: async args => {
      const panel = getPanel();
      const codeCellIndex = args.cellIndex as number;
      const widgetIdx = codeCellToWidgetIndex(panel, codeCellIndex);
      const label = indexToAlpha(codeCellIndex);

      // Capture pre-run metadata: it is written asynchronously by the
      // comm handler, so the shell reply can beat it and an immediate
      // read would return the PREVIOUS run's errors.
      const targetCell = panel.content.widgets[widgetIdx];
      const beforeMeta = targetCell.model.getMetadata('flowbook') as
        | IReproducibilityMetadata
        | undefined;

      // Activate and run via JupyterLab's native mechanism
      panel.content.activeCellIndex = widgetIdx;
      await NotebookActions.run(panel.content, panel.sessionContext);

      // Wait for this run's metadata to land before evaluating errors
      const cell = panel.content.widgets[widgetIdx];
      const meta = await waitForFlowbookMetadata(cell.model, beforeMeta);
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
  // flowbook:run-actionable-cells — the RunToClean loop
  //
  // Repeatedly executes the first stale-or-unexecuted cell in document
  // order. If a cell is executed a second time and its read or write
  // sets changed, the loop may never terminate (reruns keep marking
  // earlier cells stale), so it stops and reports potential
  // non-termination at that cell. Report-free executions finish within
  // n(n+2) runs (Progress theorem); the iteration cap is a backstop for
  // cells with no tracking metadata.
  // ------------------------------------------------------------------
  app.commands.addCommand('flowbook:run-actionable-cells', {
    execute: async () => {
      const panel = getPanel(); // Verify a notebook is open
      const results: any[] = [];
      let totalRun = 0;
      let potentialNonTermination: any = null;
      const guard = new RunToCleanGuard();
      const nCodeCells = getCodeCellOrder(panel).length;
      const maxIterations = Math.max(25, nCodeCells * (nCodeCells + 3));

      while (totalRun < maxIterations) {
        try {
          // Find next cell to run (stale/unexecuted merged, document order)
          const actionable = (await app.commands.execute(
            'flowbook:get-next-actionable',
            { documentOrder: true }
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

          // RunToClean check: a re-executed cell must not mark an
          // earlier cell stale, else the loop may never terminate.
          const cellOrder = getCodeCellOrder(getPanel());
          const report = runResult.cell_id
            ? guard.noteRun(
                runResult.cell_id,
                runResult.flowbook_meta,
                cellOrder
              )
            : null;
          if (report) {
            const markedLabels = report.backwardStale
              .map(cid => indexToAlpha(cellOrder.indexOf(cid)))
              .join(', ');
            let message =
              `Potential non-termination at ${label}: re-running this ` +
              `cell marked earlier cell(s) ${markedLabels} stale again, ` +
              'so repeatedly running stale cells may never make the ' +
              'notebook clean.';
            if (report.change) {
              const detail = formatFootprintChange(report.change);
              message += ` ${detail.charAt(0).toUpperCase()}${detail.slice(1)}.`;
            }
            message +=
              ' Varying values are fine, but the set of variables a ' +
              'cell reads and writes must be the same on every run.';
            potentialNonTermination = {
              label,
              cell_id: runResult.cell_id,
              backward_stale: report.backwardStale,
              change: report.change,
              message
            };
            results[results.length - 1].status = 'potential-non-termination';
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
        potential_non_termination: potentialNonTermination,
        summary: status
      };
    }
  });
}
