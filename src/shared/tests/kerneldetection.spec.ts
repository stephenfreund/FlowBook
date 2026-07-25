/**
 * Tests for KernelDetector: kernel-name classification, kernelChanged
 * emission semantics, and — the audit point — panel-identity keying (a
 * rename followed by a new panel at the old path must NOT share state).
 * See FRONTEND_TESTING.md §8, phase 2.
 */

import { KernelDetector, IKernelChangeInfo } from '../kerneldetection';
import {
  FakeKernel,
  FakePanel,
  FakeTracker,
  asPanel,
  asTracker
} from '../../flowbook/tests/testutils';

function makeDetector(): {
  tracker: FakeTracker;
  detector: KernelDetector;
  events: IKernelChangeInfo[];
} {
  const tracker = new FakeTracker();
  const detector = new KernelDetector(asTracker(tracker));
  const events: IKernelChangeInfo[] = [];
  detector.kernelChanged.connect((_, info) => {
    events.push(info);
  });
  return { tracker, detector, events };
}

describe('kernel classification', () => {
  it('classifies kernels by name', () => {
    const { detector } = makeDetector();
    const flowbook = new FakePanel('fb.ipynb', new FakeKernel());
    const python = new FakePanel('py.ipynb', new FakeKernel('python3'));
    const none = new FakePanel('none.ipynb', null);

    expect(detector.getKernelType(asPanel(flowbook))).toBe('flowbook_kernel');
    expect(detector.isFlowbookKernel(asPanel(flowbook))).toBe(true);
    expect(detector.getKernelType(asPanel(python))).toBe('other');
    expect(detector.isFlowbookKernel(asPanel(python))).toBe(false);
    expect(detector.getKernelType(asPanel(none))).toBe('none');
  });
});

describe('kernelChanged emission', () => {
  it('emits only when the classification changes', () => {
    const { tracker, events } = makeDetector();
    const panel = new FakePanel('a.ipynb', new FakeKernel('python3'));
    tracker.addWidget(panel);

    // other → other: same classification, no emission.
    panel.sessionContext.setKernel(new FakeKernel('python2'));
    expect(events).toHaveLength(0);

    // other → flowbook_kernel: emitted with the correct transition.
    panel.sessionContext.setKernel(new FakeKernel());
    expect(events).toHaveLength(1);
    expect(events[0].previousKernel).toBe('other');
    expect(events[0].currentKernel).toBe('flowbook_kernel');
    expect(events[0].notebook as unknown).toBe(panel);

    // flowbook_kernel → none.
    panel.sessionContext.setKernel(null);
    expect(events).toHaveLength(2);
    expect(events[1].previousKernel).toBe('flowbook_kernel');
    expect(events[1].currentKernel).toBe('none');
  });

  it('monitors notebooks that existed before construction', () => {
    const tracker = new FakeTracker();
    const panel = new FakePanel('a.ipynb', new FakeKernel('python3'));
    tracker.addWidget(panel); // present before the detector exists

    const detector = new KernelDetector(asTracker(tracker));
    const events: IKernelChangeInfo[] = [];
    detector.kernelChanged.connect((_, info) => {
      events.push(info);
    });

    panel.sessionContext.setKernel(new FakeKernel());
    expect(events).toHaveLength(1);
    expect(events[0].previousKernel).toBe('other');
  });
});

describe('panel-identity keying', () => {
  it('a rename followed by a new panel at the old path does not share state', () => {
    const { tracker, events } = makeDetector();
    const original = new FakePanel('a.ipynb', new FakeKernel());
    tracker.addWidget(original);

    original.context.rename('b.ipynb');

    // A DIFFERENT notebook opened at the original path, no kernel yet.
    const reopened = new FakePanel('a.ipynb', null);
    tracker.addWidget(reopened);

    // If state were path-keyed, reopened would inherit original's
    // 'flowbook_kernel' entry and this none→flowbook transition would
    // be invisible (previous === current → no emission).
    reopened.sessionContext.setKernel(new FakeKernel());
    expect(events).toHaveLength(1);
    expect(events[0].previousKernel).toBe('none');
    expect(events[0].currentKernel).toBe('flowbook_kernel');
    expect(events[0].notebook as unknown).toBe(reopened);

    // And the renamed panel's own state is untouched by the newcomer.
    original.sessionContext.setKernel(new FakeKernel('python3'));
    expect(events).toHaveLength(2);
    expect(events[1].previousKernel).toBe('flowbook_kernel');
    expect(events[1].currentKernel).toBe('other');
    expect(events[1].notebook as unknown).toBe(original);
  });
});
