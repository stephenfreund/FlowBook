/**
 * Regression tests for the comm lifecycle in
 * ReproducibilityExecutionHookManager (audit F1/F2 + pending-command
 * buffering). See FRONTEND_TESTING.md §8, phase 2.
 */

import { ReproducibilityExecutionHookManager } from '../executionhook';
import { COMM_TARGET } from '../protocol';
import {
  FakeExecutionSignals,
  FakeKernel,
  FakePanel,
  FakeTracker,
  asSignals,
  asTracker
} from './testutils';

// ---------------------------------------------------------------------------
// Minimal highlighter fake (only what the hook touches)
// ---------------------------------------------------------------------------

class FakeStalenessManager {
  staleCells = new Set<string>();
  reasons = new Map<string, unknown>();

  setReason(cellId: string, reason: unknown): void {
    this.reasons.set(cellId, reason);
  }

  updateFromMetadata(meta: { stale_cells: string[] }): void {
    this.staleCells = new Set(meta.stale_cells);
  }
}

class FakeHighlighter {
  managers = new Map<unknown, FakeStalenessManager>();
  cellUpdates: Array<{ cellId: string; path: string }> = [];
  statusUpdates: Array<{ icon: string; text: string }> = [];

  getStalenessManager(panel: unknown): FakeStalenessManager {
    let m = this.managers.get(panel);
    if (!m) {
      m = new FakeStalenessManager();
      this.managers.set(panel, m);
    }
    return m;
  }

  updateCell(
    cell: { model: { id: string } },
    _mgr: unknown,
    _order: string[],
    path: string
  ): void {
    this.cellUpdates.push({ cellId: cell.model.id, path });
  }

  refreshDependencies(): void {
    // no-op
  }

  updateStatus(icon: string, text: string): void {
    this.statusUpdates.push({ icon, text });
  }
}

// ---------------------------------------------------------------------------

function metadataFor(cellId: string, staleCells: string[] = []): unknown {
  return {
    type: 'metadata',
    cell_id: cellId,
    execution_seq: 1,
    read_locs: [],
    write_locs: [],
    changed_locs: [],
    stale_cells: staleCells,
    cell_order: [cellId]
  };
}

function makeHook(): {
  tracker: FakeTracker;
  highlighter: FakeHighlighter;
  signals: FakeExecutionSignals;
  hook: ReproducibilityExecutionHookManager;
} {
  const tracker = new FakeTracker();
  const highlighter = new FakeHighlighter();
  const signals = new FakeExecutionSignals();
  const hook = new ReproducibilityExecutionHookManager(
    asTracker(tracker),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    highlighter as any,
    asSignals(signals)
  );
  return { tracker, highlighter, signals, hook };
}

describe('comm lifecycle', () => {
  it('opens a comm on the flowbook target for the current panel', () => {
    const { tracker, hook } = makeHook();
    const panel = new FakePanel('a.ipynb');

    tracker.setCurrent(panel);

    const kernel = panel.kernel as FakeKernel;
    expect(kernel.comms).toHaveLength(1);
    expect(kernel.comms[0].targetName).toBe(COMM_TARGET);
    expect(kernel.comms[0].opened).toBe(true);
    hook.dispose();
  });

  it('closes the superseded comm when switching to another panel (F2) and keeps late messages bound to the OWNING panel (F1)', () => {
    const { tracker, highlighter, hook } = makeHook();
    const panelA = new FakePanel('a.ipynb');
    const cellA = panelA.addCell('cell-a');
    const panelB = new FakePanel('b.ipynb');
    panelB.addCell('cell-a'); // same id in B — must NOT receive A's metadata

    tracker.setCurrent(panelA);
    const commA = (panelA.kernel as FakeKernel).comms[0];

    tracker.setCurrent(panelB);

    // F2: A's comm was closed when B's kernel took over.
    expect(commA.closed).toBe(true);
    expect((panelB.kernel as FakeKernel).comms).toHaveLength(1);

    // F1: a message still delivered through A's handler (in flight when
    // the user switched tabs) is applied to A's cell, not B's.
    commA.deliver(metadataFor('cell-a'));

    expect(cellA.model.getMetadata('flowbook')).toBeDefined();
    const cellB = panelB.content.widgets[0];
    expect(cellB.model.getMetadata('flowbook')).toBeUndefined();
    // And the staleness update went to A's manager, not B's.
    expect(highlighter.managers.has(panelA)).toBe(true);
    expect(highlighter.managers.has(panelB)).toBe(false);
    hook.dispose();
  });

  it('clears the comm reference when the kernel side closes it', () => {
    const { tracker, hook } = makeHook();
    const panel = new FakePanel('a.ipynb');
    tracker.setCurrent(panel);
    const comm = (panel.kernel as FakeKernel).comms[0];

    comm.close();
    comm.onClose?.();

    // Sends now buffer (no throw, nothing reaches the dead comm).
    hook.sendCommand({ type: 'sync' });
    expect(comm.sent).toHaveLength(0);
    hook.dispose();
  });
});

describe('pending-command buffering', () => {
  it('buffers while no comm exists and flushes in order on reconnect', () => {
    const { tracker, hook } = makeHook();
    const panel = new FakePanel('a.ipynb', null); // no kernel yet
    tracker.setCurrent(panel);

    hook.sendCommand({ type: 'notebook_structure', cell_order: ['x'] });
    hook.sendCommand({ type: 'cell_edited', cell_id: 'x', source: 's' });

    // Kernel arrives (e.g. session started) — handler reconnects and flushes.
    const kernel = new FakeKernel();
    panel.sessionContext.setKernel(kernel);

    const comm = kernel.comms[0];
    expect(comm.sent).toEqual([
      { type: 'notebook_structure', cell_order: ['x'] },
      { type: 'cell_edited', cell_id: 'x', source: 's' }
    ]);
    hook.dispose();
  });

  it('buffers across a kernel restart window', () => {
    const { tracker, hook } = makeHook();
    const panel = new FakePanel('a.ipynb');
    tracker.setCurrent(panel);
    const kernel = panel.kernel as FakeKernel;

    panel.sessionContext.setStatus('restarting');
    hook.sendCommand({ type: 'sync' });
    expect(kernel.comms[0].sent).toHaveLength(0); // old comm got nothing

    panel.sessionContext.setStatus('idle');
    const newComm = kernel.comms[1];
    expect(newComm).toBeDefined();
    expect(newComm.sent).toEqual([{ type: 'sync' }]);
    hook.dispose();
  });

  it('drops the oldest entries beyond the cap', () => {
    const { tracker, hook } = makeHook();
    const panel = new FakePanel('a.ipynb', null);
    tracker.setCurrent(panel);

    for (let i = 0; i < 105; i++) {
      hook.sendCommand({ type: 'cell_edited', cell_id: `c${i}` });
    }
    const kernel = new FakeKernel();
    panel.sessionContext.setKernel(kernel);

    const sent = kernel.comms[0].sent as Array<{ cell_id: string }>;
    expect(sent).toHaveLength(100);
    expect(sent[0].cell_id).toBe('c5'); // c0..c4 dropped
    expect(sent[99].cell_id).toBe('c104');
    hook.dispose();
  });
});

describe('reconnection guards', () => {
  it('a background panel kernel change does not steal the comm', () => {
    const { tracker, hook } = makeHook();
    const panelA = new FakePanel('a.ipynb');
    const panelB = new FakePanel('b.ipynb');

    tracker.setCurrent(panelA);
    tracker.setCurrent(panelB);
    const commB = (panelB.kernel as FakeKernel).comms[0];

    // A (now background) swaps its kernel — must not affect B's comm.
    const newKernelA = new FakeKernel();
    panelA.sessionContext.setKernel(newKernelA);

    expect(newKernelA.comms).toHaveLength(0);
    expect(commB.closed).toBe(false);
    hook.dispose();
  });
});
