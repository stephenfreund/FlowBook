/**
 * Lifecycle tests for ReproducibilityCellHighlighter (audit F3 + rename
 * migration): monitor listeners attach once per panel, dispose disconnects
 * them, post-dispose emissions mutate nothing, pathChanged migrates the
 * staleness manager, and kernel restart clears flowbook state from cells.
 * See FRONTEND_TESTING.md §8, phase 2.
 */

import { ReproducibilityCellHighlighter } from '../cellhighlighter';
import { StalenessManager } from '../stalenessmanager';
import {
  FakeExecutionSignals,
  FakePanel,
  FakeTracker,
  asPanel,
  asSignals,
  asTracker
} from './testutils';

/** Records calls; the highlighter drives it through updateCell/dispose. */
class FakeMetadataPanel {
  metadataUpdates: Array<{ cellId: string }> = [];
  clears = 0;
  statusUpdates: Array<{ icon: string; text: string }> = [];

  updateMetadata(_meta: unknown, cellId: string, _cellOrder: string[]): void {
    this.metadataUpdates.push({ cellId });
  }

  clear(): void {
    this.clears += 1;
  }

  updateStatus(icon: string, text: string): void {
    this.statusUpdates.push({ icon, text });
  }
}

function makeHighlighter(): {
  tracker: FakeTracker;
  metadataPanel: FakeMetadataPanel;
  signals: FakeExecutionSignals;
  highlighter: ReproducibilityCellHighlighter;
} {
  const tracker = new FakeTracker();
  const metadataPanel = new FakeMetadataPanel();
  const signals = new FakeExecutionSignals();
  const highlighter = new ReproducibilityCellHighlighter(
    asTracker(tracker),
    metadataPanel as any, // eslint-disable-line @typescript-eslint/no-explicit-any
    asSignals(signals)
  );
  return { tracker, metadataPanel, signals, highlighter };
}

// Fake timers keep refreshDependencies' requestAnimationFrame callbacks
// inert so tests stay synchronous and deterministic.
beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

describe('monitor listener lifecycle', () => {
  it('attaches monitor listeners once per panel across repeated currentChanged', () => {
    const { tracker, highlighter } = makeHighlighter();
    const panel = new FakePanel('a.ipynb');
    panel.addCell('c1', 'code', 'x = 1');

    tracker.setCurrent(panel);
    tracker.setCurrent(panel);
    tracker.setCurrent(panel);

    expect(panel.content.model.cells.changed.live).toBe(1);
    expect(panel.context.pathChanged.live).toBe(1);
    // statusChanged carries exactly two listeners: the monitor handler
    // plus the per-notebook StalenessManager's restart listener — each
    // attached once despite three currentChanged emissions.
    expect(panel.sessionContext.statusChanged.live).toBe(2);
    highlighter.dispose();
  });

  it('dispose disconnects all monitor listeners and staleness managers', () => {
    const { tracker, signals, highlighter } = makeHighlighter();
    const panel = new FakePanel('a.ipynb');
    panel.addCell('c1', 'code', 'x = 1');
    tracker.setCurrent(panel);

    highlighter.dispose();

    expect(panel.content.model.cells.changed.live).toBe(0);
    expect(panel.context.pathChanged.live).toBe(0);
    expect(panel.sessionContext.statusChanged.live).toBe(0);
    expect(tracker.currentChanged.live).toBe(0);
    expect(tracker.activeCellChanged.live).toBe(0);
    expect(signals.executed.live).toBe(0);
  });

  it('post-dispose emissions mutate nothing', () => {
    const { tracker, highlighter } = makeHighlighter();
    const panel = new FakePanel('a.ipynb');
    const cell = panel.addCell('c1', 'code', 'x = 1');
    tracker.setCurrent(panel);
    cell.model.setMetadata('flowbook', { cell_id: 'c1' });

    highlighter.dispose();
    const classesBefore = [...cell.node.classList._set].sort();

    panel.content.model.cells.changed.emit({ type: 'add' });
    panel.sessionContext.setStatus('restarting');
    panel.sessionContext.setStatus('idle');
    panel.context.rename('b.ipynb');

    expect([...cell.node.classList._set].sort()).toEqual(classesBefore);
    // A live highlighter would have cleared this on 'restarting'.
    expect(cell.model.getMetadata('flowbook')).toEqual({ cell_id: 'c1' });
  });
});

describe('rename migration', () => {
  it('context.rename migrates the staleness manager to the new path key', () => {
    const { tracker, highlighter } = makeHighlighter();
    const panel = new FakePanel('a.ipynb');
    panel.addCell('c1', 'code', 'x = 1');
    tracker.setCurrent(panel);

    const before = highlighter.getStalenessManager(asPanel(panel));
    panel.context.rename('sub/b.ipynb');
    const after = highlighter.getStalenessManager(asPanel(panel));

    // Same instance survives the rename — no state reset.
    expect(after).toBe(before);

    // Internal map is keyed under the new path only.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const managers = (highlighter as any)['_stalenessManagers'] as Map<
      string,
      StalenessManager
    >;
    expect([...managers.keys()]).toEqual(['sub/b.ipynb']);

    // A NEW panel opened at the old path gets its own manager, not the
    // renamed notebook's.
    const reopened = new FakePanel('a.ipynb');
    reopened.addCell('c9', 'code', 'y = 1');
    tracker.setCurrent(reopened);
    const otherManager = highlighter.getStalenessManager(asPanel(reopened));
    expect(otherManager).not.toBe(before);
    expect(highlighter.getStalenessManager(asPanel(panel))).toBe(before);
    highlighter.dispose();
  });
});

describe('kernel restart', () => {
  it('clears flowbook metadata, notice outputs and classes from cells', () => {
    const { tracker, highlighter } = makeHighlighter();
    const panel = new FakePanel('a.ipynb');
    const cell = panel.addCell('c1', 'code', 'x = 1');
    tracker.setCurrent(panel);

    cell.model.setMetadata('flowbook', { cell_id: 'c1', execution_seq: 1 });
    cell.model.setMetadata('flowbook_staleness', {
      type: 'unknown',
      message: 'Dependencies changed'
    });
    cell.model.outputs.fromJSON([
      {
        output_type: 'display_data',
        data: { 'text/plain': 'stale notice' },
        metadata: { flowbook_staleness_notice: true }
      },
      { output_type: 'stream', name: 'stdout', text: 'hi' }
    ]);
    cell.node.classList.add('flowbook-cell-stale');

    panel.sessionContext.setStatus('restarting');

    expect(cell.model.getMetadata('flowbook')).toBeUndefined();
    expect(cell.model.getMetadata('flowbook_staleness')).toBeUndefined();
    // Notice output removed; the real output survives.
    expect(cell.model.outputs.list).toEqual([
      { output_type: 'stream', name: 'stdout', text: 'hi' }
    ]);
    expect(cell.node.classList.contains('flowbook-cell-stale')).toBe(false);
    highlighter.dispose();
  });
});
