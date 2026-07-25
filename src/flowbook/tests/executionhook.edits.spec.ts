/**
 * Edit-detection tests for ReproducibilityExecutionHookManager
 * ([Inst-Edit], §2.3): the 1 s cell_edited debounce, flush-on-schedule and
 * flush-on-dispose semantics, executed-cell gating (including external
 * executions delivered via comm), and listener attachment for cells added
 * after activation. See FRONTEND_TESTING.md §8, phase 2.
 */

import { ReproducibilityExecutionHookManager } from '../executionhook';
import {
  FakeCell,
  FakeComm,
  FakeExecutionSignals,
  FakeHighlighter,
  FakeKernel,
  FakePanel,
  FakeTracker,
  asHighlighter,
  asSignals,
  asTracker,
  metadataFor
} from './testutils';

interface ISetup {
  tracker: FakeTracker;
  highlighter: FakeHighlighter;
  signals: FakeExecutionSignals;
  hook: ReproducibilityExecutionHookManager;
  panel: FakePanel;
  cell: FakeCell;
  comm: FakeComm;
}

function setup(): ISetup {
  const tracker = new FakeTracker();
  const highlighter = new FakeHighlighter();
  const signals = new FakeExecutionSignals();
  const hook = new ReproducibilityExecutionHookManager(
    asTracker(tracker),
    asHighlighter(highlighter),
    asSignals(signals)
  );
  const panel = new FakePanel('a.ipynb');
  const cell = panel.addCell('c1', 'code', 'x = 1');
  tracker.setCurrent(panel);
  const comm = (panel.kernel as FakeKernel).comms[0];
  return { tracker, highlighter, signals, hook, panel, cell, comm };
}

function editsSent(
  comm: FakeComm
): Array<{ type: string; cell_id: string; source: string }> {
  return (
    comm.sent as Array<{ type: string; cell_id: string; source: string }>
  ).filter(m => m.type === 'cell_edited');
}

function markExecuted(s: ISetup, cell: FakeCell): void {
  s.signals.executed.emit({ notebook: s.panel.content, cell });
}

beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

describe('cell_edited debouncing', () => {
  it('sends cell_edited 1 s after an edit to an executed cell', () => {
    const s = setup();
    markExecuted(s, s.cell);

    s.cell.model.sharedModel.setSource('x = 2');
    expect(editsSent(s.comm)).toHaveLength(0);

    jest.advanceTimersByTime(999);
    expect(editsSent(s.comm)).toHaveLength(0);

    jest.advanceTimersByTime(1);
    expect(editsSent(s.comm)).toEqual([
      { type: 'cell_edited', cell_id: 'c1', source: 'x = 2' }
    ]);
    s.hook.dispose();
  });

  it('a second edit within 1 s resets the timer (single send, latest source)', () => {
    const s = setup();
    markExecuted(s, s.cell);

    s.cell.model.sharedModel.setSource('x = 2');
    jest.advanceTimersByTime(600);
    s.cell.model.sharedModel.setSource('x = 3');
    // 1.2 s after the first edit but only 0.6 s after the second — the
    // timer was reset, so nothing has been sent yet.
    jest.advanceTimersByTime(600);
    expect(editsSent(s.comm)).toHaveLength(0);

    jest.advanceTimersByTime(400);
    const edits = editsSent(s.comm);
    expect(edits).toHaveLength(1);
    expect(edits[0].source).toBe('x = 3');
    s.hook.dispose();
  });

  it('edits to a never-executed cell send nothing', () => {
    const s = setup();

    s.cell.model.sharedModel.setSource('x = 2');
    jest.advanceTimersByTime(5000);

    expect(editsSent(s.comm)).toHaveLength(0);
    s.hook.dispose();
  });
});

describe('flush semantics', () => {
  it('scheduling execution flushes the pending edit before the debounce elapses', () => {
    const s = setup();
    markExecuted(s, s.cell);
    s.cell.model.sharedModel.setSource('x = 2');
    expect(editsSent(s.comm)).toHaveLength(0);

    s.signals.executionScheduled.emit({
      notebook: s.panel.content,
      cell: s.cell
    });

    // Flushed immediately (no timer advance), and BEFORE the
    // notebook_structure message that precedes the execution.
    const sent = s.comm.sent as Array<{ type: string }>;
    const editIdx = sent.findIndex(m => m.type === 'cell_edited');
    const structIdx = sent.findIndex(m => m.type === 'notebook_structure');
    expect(editIdx).toBeGreaterThanOrEqual(0);
    expect(structIdx).toBeGreaterThan(editIdx);

    // The debounce timer was cancelled — no duplicate send later.
    jest.advanceTimersByTime(2000);
    expect(editsSent(s.comm)).toHaveLength(1);
    s.hook.dispose();
  });

  it('dispose flushes pending edits to the comm', () => {
    const s = setup();
    markExecuted(s, s.cell);
    s.cell.model.sharedModel.setSource('x = 2');
    expect(editsSent(s.comm)).toHaveLength(0);

    s.hook.dispose();

    expect(editsSent(s.comm)).toEqual([
      { type: 'cell_edited', cell_id: 'c1', source: 'x = 2' }
    ]);
  });
});

describe('executed-cell tracking', () => {
  it('a cell executed externally (metadata via comm) becomes edit-reported', () => {
    const s = setup();
    // No NotebookActions.executed emission — the run happened on the
    // shared kernel (e.g. MCP); only the metadata comm message arrives.
    s.comm.deliver(metadataFor('c1'));

    s.cell.model.sharedModel.setSource('x = 2');
    jest.advanceTimersByTime(1000);

    expect(editsSent(s.comm)).toEqual([
      { type: 'cell_edited', cell_id: 'c1', source: 'x = 2' }
    ]);
    s.hook.dispose();
  });

  it('cells added after activation get edit listeners', () => {
    const s = setup();
    const late = s.panel.addCell('c2', 'code', 'y = 1');
    s.panel.content.model.cells.changed.emit({ type: 'add' });

    // The add also notified the kernel of the new structure.
    const structures = (
      s.comm.sent as Array<{ type: string; cell_order?: string[] }>
    ).filter(m => m.type === 'notebook_structure');
    expect(structures).toEqual([
      { type: 'notebook_structure', cell_order: ['c1', 'c2'] }
    ]);

    markExecuted(s, late);
    late.model.sharedModel.setSource('y = 2');
    jest.advanceTimersByTime(1000);

    expect(editsSent(s.comm)).toEqual([
      { type: 'cell_edited', cell_id: 'c2', source: 'y = 2' }
    ]);
    s.hook.dispose();
  });
});
