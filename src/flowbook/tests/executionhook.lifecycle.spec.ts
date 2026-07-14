/**
 * Listener-lifecycle tests for ReproducibilityExecutionHookManager (audit
 * F3): repeated currentChanged must not stack duplicate listeners, dispose
 * must disconnect everything, and post-dispose emissions must be no-ops.
 * Connection hygiene is asserted via the CountingSignal counters.
 * See FRONTEND_TESTING.md §8, phase 2.
 */

import { ReproducibilityExecutionHookManager } from '../executionhook';
import {
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
    asHighlighter(highlighter),
    asSignals(signals)
  );
  return { tracker, highlighter, signals, hook };
}

describe('listener lifecycle', () => {
  it('repeated currentChanged on the same panel attaches each listener once', () => {
    const { tracker, hook } = makeHook();
    const panel = new FakePanel('a.ipynb');
    const cell = panel.addCell('c1', 'code', 'x = 1');

    tracker.setCurrent(panel);
    tracker.setCurrent(panel);
    tracker.setCurrent(panel);

    expect(panel.sessionContext.kernelChanged.live).toBe(1);
    expect(panel.sessionContext.statusChanged.live).toBe(1);
    expect(panel.content.model.cells.changed.live).toBe(1);
    expect(cell.model.sharedModel.changed.live).toBe(1);
    hook.dispose();
  });

  it('dispose disconnects every listener', () => {
    const { tracker, signals, hook } = makeHook();
    const panel = new FakePanel('a.ipynb');
    const cell = panel.addCell('c1', 'code', 'x = 1');
    tracker.setCurrent(panel);

    hook.dispose();

    expect(panel.sessionContext.kernelChanged.live).toBe(0);
    expect(panel.sessionContext.statusChanged.live).toBe(0);
    expect(panel.content.model.cells.changed.live).toBe(0);
    expect(cell.model.sharedModel.changed.live).toBe(0);
    expect(signals.executed.live).toBe(0);
    expect(signals.executionScheduled.live).toBe(0);
    expect(tracker.currentChanged.live).toBe(0);
  });

  it('emissions after dispose are no-ops (no sends, no metadata writes)', () => {
    jest.useFakeTimers();
    try {
      const { tracker, highlighter, signals, hook } = makeHook();
      const panel = new FakePanel('a.ipynb');
      const cell = panel.addCell('c1', 'code', 'x = 1');
      tracker.setCurrent(panel);
      const kernel = panel.kernel as FakeKernel;
      const comm = kernel.comms[0];

      // Mark the cell executed pre-dispose so an edit WOULD schedule a
      // cell_edited send if the listener were still attached.
      signals.executed.emit({ notebook: panel.content, cell });

      hook.dispose();
      const sentBefore = comm.sent.length;
      const updatesBefore = highlighter.cellUpdates.length;

      signals.executed.emit({ notebook: panel.content, cell });
      signals.executionScheduled.emit({ notebook: panel.content, cell });
      cell.model.sharedModel.setSource('x = 2');
      jest.advanceTimersByTime(5000);
      panel.sessionContext.setStatus('idle');
      comm.deliver(metadataFor('c1'));

      expect(comm.sent.length).toBe(sentBefore);
      expect(kernel.comms).toHaveLength(1); // no reconnect after dispose
      expect(highlighter.cellUpdates.length).toBe(updatesBefore);
      expect(cell.model.getMetadata('flowbook')).toBeUndefined();
    } finally {
      jest.useRealTimers();
    }
  });
});
